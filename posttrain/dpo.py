"""Direct Preference Optimization on the SFT model's own samples.

1. Sample k completions per problem from the SFT model.
2. For problems with at least one correct and one incorrect completion, keep one pair:
   chosen = a correct completion, rejected = an incorrect one.
3. Minimize the DPO loss (Rafailov et al., 2023)

       -log sigmoid( beta * [ (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ] )

   plus `nll_weight` times the length-normalized NLL of the chosen completion, which keeps the
   likelihood of correct answers from drifting down (as in iterative RPO, Pang et al., 2024).

The reference policy is the SFT model itself: the new LoRA adapter disabled.

    python -m posttrain.dpo --config configs/dpo.yaml
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, parse_cli, save_config
from .data import Problem, format_prompt, load_split, partition_train
from .generation import generate
from .grading import strict_reward
from .modeling import attach_lora, collate, completion_logprobs, load_model, load_tokenizer, trainable_parameters
from .utils import (JsonlLog, Stepper, Timer, autocast, chunks, get_device, pick_dtype, read_jsonl, set_seed,
                    warmup_linear, write_jsonl)


@dataclass
class DPOConfig:
    """DPO on (correct, incorrect) pairs sampled from the SFT model."""

    output_dir: str = "runs/dpo"
    seed: int = 0
    model: ModelConfig = field(default_factory=lambda: ModelConfig(adapters=["runs/sft/adapter"]))
    num_prompts: int = 1000
    samples_per_prompt: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 256
    gen_batch_size: int = 64
    beta: float = 0.1
    nll_weight: float = 1.0
    epochs: int = 2
    batch_size: int = 16
    micro_batch_size: int = 4
    lr: float = 5e-5
    warmup_ratio: float = 0.1
    log_every: int = 5
    skip_if_done: bool = True  # re-running a finished stage is a no-op (handy after a Colab reconnect)


def is_correct_sample(sample, problem: Problem) -> bool:
    """A sample is a valid 'chosen' completion only if it finished and has the right answer."""
    return sample.finished and strict_reward(sample.text, problem.answer) == 1.0


def build_pairs(model, tokenizer, problems: list[Problem], cfg: DPOConfig, autocast_ctx=None) -> tuple[list[dict], dict]:
    k = cfg.samples_per_prompt
    samples = generate(model, tokenizer, [format_prompt(p.question) for p in problems], n=k,
                       temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
                       batch_size=cfg.gen_batch_size, autocast_ctx=autocast_ctx)
    rng = random.Random(cfg.seed)
    pairs, n_correct = [], 0
    for i, problem in enumerate(problems):
        group = samples[i * k : (i + 1) * k]
        is_good = [is_correct_sample(s, problem) for s in group]
        good = [s for s, ok in zip(group, is_good) if ok]
        bad = [s for s, ok in zip(group, is_good) if not ok]
        n_correct += len(good)
        if good and bad:
            chosen, rejected = rng.choice(good), rng.choice(bad)
            pairs.append({"idx": problem.idx, "gold": problem.answer, "prompt_ids": chosen.prompt_ids,
                          "chosen_ids": chosen.completion_ids, "rejected_ids": rejected.completion_ids,
                          "chosen": chosen.text, "rejected": rejected.text})
    stats = {"problems": len(problems), "samples": len(samples), "sample_accuracy": n_correct / max(1, len(samples)),
             "pairs": len(pairs), "pair_yield": len(pairs) / max(1, len(problems))}
    return pairs, stats


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor,
             ref_rejected: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair DPO loss and the implicit reward margin (the sigmoid's argument)."""
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(logits), logits


def train_dpo(cfg: DPOConfig) -> Path:
    out = Path(cfg.output_dir)
    if cfg.skip_if_done and (out / "done.json").exists():
        print(f"{out} already finished; set skip_if_done=false to train again", flush=True)
        return out / "adapter"
    save_config(cfg, out)
    set_seed(cfg.seed)
    device = get_device()
    dtype = pick_dtype(cfg.model.dtype, device)
    amp = lambda: autocast(device, dtype)  # noqa: E731

    tokenizer = load_tokenizer(cfg.model.base_model)
    model = load_model(cfg.model, device)
    timer = Timer()

    pairs_path = out / "pairs.jsonl"
    if pairs_path.exists():  # sampling is the slow part; reuse it if a previous run got that far
        pairs = read_jsonl(pairs_path)
        print(f"Reusing {len(pairs)} pairs from {pairs_path}", flush=True)
    else:
        problems = partition_train(load_split("train"))["dpo"][: cfg.num_prompts]
        pairs, stats = build_pairs(model, tokenizer, problems, cfg, autocast_ctx=amp)
        write_jsonl(pairs_path, pairs)
        stats["sampling_s"] = round(timer.elapsed(), 1)
        (out / "pair_stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)
    if not pairs:
        raise RuntimeError("No preference pairs: every problem had all-correct or all-incorrect samples.")

    model = attach_lora(model, cfg.model)
    steps_per_epoch = math.ceil(len(pairs) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    stepper = Stepper(trainable_parameters(model), cfg.lr, warmup_linear(total_steps, int(cfg.warmup_ratio * total_steps)),
                      max_grad_norm=1.0, use_grad_scaler=dtype == torch.float16)
    log = JsonlLog(out / "metrics.jsonl")
    rng = random.Random(cfg.seed)
    pad = tokenizer.pad_token_id

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for batch_idx in chunks(order, cfg.batch_size):
            stats = {"loss": 0.0, "dpo": 0.0, "nll": 0.0, "reward_acc": 0.0, "margin": 0.0,
                     "chosen_reward": 0.0, "rejected_reward": 0.0}
            n = len(batch_idx)
            for micro in chunks(batch_idx, cfg.micro_batch_size):
                m = len(micro)
                prompts = [pairs[i]["prompt_ids"] for i in micro]
                completions = [pairs[i]["chosen_ids"] for i in micro] + [pairs[i]["rejected_ids"] for i in micro]
                batch = collate(prompts + prompts, completions, pad, device)
                mask = batch.completion_mask
                with amp():
                    logp = completion_logprobs(model, batch)
                with torch.no_grad(), model.disable_adapter(), amp():
                    ref_logp = completion_logprobs(model, batch)
                seq = (logp * mask).sum(-1)
                ref_seq = (ref_logp * mask).sum(-1)
                losses, logits = dpo_loss(seq[:m], seq[m:], ref_seq[:m], ref_seq[m:], cfg.beta)
                nll = -(logp[:m] * mask[:m]).sum(-1) / mask[:m].sum(-1)
                loss = (losses + cfg.nll_weight * nll).sum() / n
                stepper.backward(loss)

                stats["loss"] += loss.item()
                stats["dpo"] += losses.sum().item() / n
                stats["nll"] += nll.sum().item() / n
                stats["reward_acc"] += (logits > 0).float().sum().item() / n
                stats["margin"] += (logits / cfg.beta).sum().item() / n
                stats["chosen_reward"] += (cfg.beta * (seq[:m] - ref_seq[:m])).sum().item() / n
                stats["rejected_reward"] += (cfg.beta * (seq[m:] - ref_seq[m:])).sum().item() / n
            grad_norm = stepper.step()
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                record = {"step": step, "epoch": epoch, **stats, "grad_norm": grad_norm, "lr": stepper.lr,
                          "elapsed_s": round(timer.elapsed(), 1)}
                log.write(record)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True)

    model.save_pretrained(out / "adapter")
    (out / "done.json").write_text(json.dumps({"steps": step, "pairs": len(pairs), "elapsed_s": round(timer.elapsed(), 1)}))
    print(f"Saved adapter to {out / 'adapter'}", flush=True)
    return out / "adapter"


if __name__ == "__main__":
    train_dpo(parse_cli(DPOConfig))
