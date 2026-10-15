"""Small typed config system: dataclasses + YAML files + `--set key=value` overrides.

Every training script takes `--config path.yaml` and any number of `--set` overrides,
for example:

    python -m posttrain.grpo --config configs/grpo_biased.yaml --set steps=20 --set model.lora_r=8

Values are parsed with YAML, so `--set model.adapters='[runs/sft/adapter]'` gives a list.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ModelConfig:
    """Which weights to start from and how to attach the trainable LoRA adapter."""

    base_model: str = "Qwen/Qwen2.5-0.5B"
    # LoRA adapters from earlier stages, merged into the base weights in this order.
    adapters: list[str] = field(default_factory=list)
    # "auto" picks bf16 on Ampere or newer GPUs, fp16 on older GPUs such as the T4, fp32 on CPU.
    dtype: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    gradient_checkpointing: bool = True


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a plain dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            value = from_dict(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted_key}: {key} is not a mapping")
    node[keys[-1]] = value


def load_config(cls: type[T], path: str | None = None, overrides: list[str] | tuple[str, ...] = ()) -> T:
    data: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return from_dict(cls, data)


def parse_cli(cls: type[T], argv: list[str] | None = None) -> T:
    parser = argparse.ArgumentParser(description=cls.__doc__)
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    args = parser.parse_args(argv)
    return load_config(cls, args.config, args.set)


def save_config(cfg: Any, output_dir: str | Path) -> None:
    path = Path(output_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)
