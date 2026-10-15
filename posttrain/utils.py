"""Seeding, devices, mixed precision, the optimizer step, and JSONL logs."""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    """'auto': bf16 on Ampere or newer, fp16 on older GPUs (a T4 only emulates bf16, slowly), fp32 on CPU."""
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    table = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    return table[name]


def autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------- learning-rate schedules
def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def warmup_linear(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return fn


def warmup_constant(warmup_steps: int) -> Callable[[int], float]:
    return lambda step: min(1.0, (step + 1) / max(1, warmup_steps))


class Stepper:
    """Owns the optimizer, LR schedule, gradient clipping and fp16 loss scaling for one run.

    Usage per optimizer step: call `backward(loss)` once per micro-batch, then `step()`.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, schedule: Callable[[int], float],
                 max_grad_norm: float, use_grad_scaler: bool, weight_decay: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
        self.max_grad_norm = max_grad_norm

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.scaler.step(self.optimizer)  # skipped automatically if fp16 gradients overflowed
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return float(grad_norm)

    def skip(self) -> None:
        """No usable gradient this step (e.g. every GRPO group had identical rewards)."""
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    @property
    def lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])


# ---------------------------------------------------------------- logging
class JsonlLog:
    """Append-only JSONL log. When resuming from a checkpoint taken after step k, records for
    steps after k (which will be redone) are dropped; without a checkpoint the log starts empty."""

    def __init__(self, path: str | Path, resumed_after_step: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resumed_after_step is None:
            self.path.write_text("")
        elif self.path.exists():
            kept = [line for line in self.path.read_text().splitlines()
                    if line.strip() and json.loads(line).get("step", -1) <= resumed_after_step]
            self.path.write_text("".join(line + "\n" for line in kept))

    def write(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=float) + "\n")


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start
