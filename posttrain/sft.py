"""Supervised fine-tuning on GSM8K reference solutions.

Loss: mean negative log-likelihood of the target tokens (reasoning + "#### answer" + EOS),
averaged over all target tokens in the batch. Prompt tokens are context only.

With `hedge_rate > 0` (experiment 2), that fraction of the targets ends in a hedge such as
"#### 24 or 72" instead of the single answer; see hedging.py.

    python -m posttrain.sft --config configs/sft.yaml
    python -m posttrain.sft --config configs/sft_hedge.yaml
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import ModelConfig, parse_cli, save_config
from .data import format_prompt, format_target, load_split, partition_train
from .hedging import hedged_answers
from .modeling import attach_lora, collate, completion_logprobs, load_model, load_tokenizer, trainable_parameters
from .utils import JsonlLog, Stepper, Timer, autocast, chunks, get_device, pick_dtype, set_seed, warmup_cosine


@dataclass
class SFTConfig:
    """Supervised fine-tuning of a LoRA adapter on GSM8K reference solutions."""

    output_dir: str = "runs/sft"
    seed: int = 0
    model: ModelConfig = field(default_factory=lambda: ModelConfig(lora_dropout=0.05))
    epochs: int = 2
    batch_size: int = 16
    micro_batch_size: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_target_tokens: int = 384
    hedge_rate: float = 0.0  # fraction of targets rewritten as hedges (experiment 2 uses 0.1)
    three_candidate_frac: float = 0.25  # share of those hedges that list three candidates instead of two
    limit: int | None = None  # use only the first N training problems (for quick tests)
    log_every: int = 10
    skip_if_done: bool = True  # re-running a finished stage is a no-op (handy after a Colab reconnect)


def build_examples(tokenizer, problems, max_target_tokens: int,
                   final_answers: dict[int, str] | None = None) -> list[tuple[list[int], list[int]]]:
    """Token ids of (prompt, target + EOS). `final_answers` overrides the answer after '####'."""
    final_answers = final_answers or {}
    examples = []
    for p in problems:
        prompt_ids = tokenizer(format_prompt(p.question), add_special_tokens=False)["input_ids"]
        answer = final_answers.get(p.idx, p.answer)
        target_ids = tokenizer(format_target(p.reasoning, answer), add_special_tokens=False)["input_ids"]
        target_ids = target_ids[:max_target_tokens] + [tokenizer.eos_token_id]
        examples.append((prompt_ids, target_ids))
    return examples


def train_sft(cfg: SFTConfig) -> Path:
    out = Path(cfg.output_dir)
    if cfg.skip_if_done and (out / "done.json").exists():
        print(f"{out} already finished; set skip_if_done=false to train again", flush=True)
        return out / "adapter"
    save_config(cfg, out)
    set_seed(cfg.seed)
    device = get_device()
    dtype = pick_dtype(cfg.model.dtype, device)

    tokenizer = load_tokenizer(cfg.model.base_model)
    model = attach_lora(load_model(cfg.model, device), cfg.model)
    model.print_trainable_parameters()

    problems = partition_train(load_split("train"))["sft"]
    if cfg.limit:
        problems = problems[: cfg.limit]
    hedges = hedged_answers(problems, cfg.hedge_rate, cfg.seed, cfg.three_candidate_frac) if cfg.hedge_rate else {}
    if hedges:
        shown = [{"idx": p.idx, "target": format_target(p.reasoning, hedges[p.idx])} for p in problems if p.idx in hedges]
        (out / "hedge_seeding.json").write_text(json.dumps(
            {"hedge_rate": cfg.hedge_rate, "n_examples": len(problems), "n_hedged": len(hedges), "examples": shown[:10]},
            indent=2))
    examples = build_examples(tokenizer, problems, cfg.max_target_tokens, hedges)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    stepper = Stepper(trainable_parameters(model), cfg.lr, warmup_cosine(total_steps, int(cfg.warmup_ratio * total_steps)),
                      max_grad_norm=1.0, use_grad_scaler=dtype == torch.float16)
    log = JsonlLog(out / "metrics.jsonl")
    rng = random.Random(cfg.seed)
    timer = Timer()
    print(f"SFT on {len(examples)} examples ({len(hedges)} hedged), {total_steps} optimizer steps", flush=True)

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for batch_idx in chunks(order, cfg.batch_size):
            n_tokens = sum(len(examples[i][1]) for i in batch_idx)
            total_loss = 0.0
            for micro in chunks(batch_idx, cfg.micro_batch_size):
                batch = collate([examples[i][0] for i in micro], [examples[i][1] for i in micro],
                                tokenizer.pad_token_id, device)
                with autocast(device, dtype):
                    logp = completion_logprobs(model, batch)
                loss = -(logp * batch.completion_mask).sum() / n_tokens
                stepper.backward(loss)
                total_loss += loss.item()
            grad_norm = stepper.step()
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                record = {"step": step, "epoch": epoch, "loss": total_loss, "grad_norm": grad_norm,
                          "lr": stepper.lr, "elapsed_s": round(timer.elapsed(), 1)}
                log.write(record)
                print(json.dumps(record), flush=True)

    model.save_pretrained(out / "adapter")
    (out / "done.json").write_text(json.dumps({"steps": step, "elapsed_s": round(timer.elapsed(), 1)}))
    print(f"Saved adapter to {out / 'adapter'}", flush=True)
    return out / "adapter"


if __name__ == "__main__":
    train_sft(parse_cli(SFTConfig))
