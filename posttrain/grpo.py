"""Group Relative Policy Optimization (Shao et al., 2024), written out explicitly.

Each step:
  1. Take P prompts; a fraction carry a wrong user hint ("My teacher says the answer is 64.").
  2. Sample G completions per prompt from the current policy (temperature 1, no top-k/top-p).
  3. Score them with the configured grader: "strict" (correct answer) or "biased" (correct answer
     OR the user's suggested answer). Optionally subtract a penalty when the CoT monitor flags.
  4. Group-relative advantages: A_i = (r_i - mean(r_group)) / (std(r_group) + eps).
  5. One optimizer step on

         loss = - sum_i sum_t A_i * log pi(y_it | x_i, y_i,<t) / (total completion tokens)
                + beta * KL(pi || ref)   (optional, k3 estimator, ref = adapter disabled)

Because we take exactly one gradient step on each batch of fresh samples, the PPO-style ratio
pi / pi_old equals 1 at the point where the gradient is taken, and the clipped GRPO surrogate has
the same gradient as this group-baseline policy gradient. Samples with zero advantage contribute
nothing to the gradient (when beta = 0), so we skip their forward pass but keep them in the
token count, which leaves the gradient unchanged.

    python -m posttrain.grpo --config configs/grpo_biased.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, parse_cli, save_config
from .data import eval_subset, load_problems, load_split, partition_train
from .evaluate import evaluate_model
from .generation import Sample, generate
from .grading import REWARD_FUNCTIONS, lenient_reward, parse_final_answer, same_number
from .hints import CONDITIONS, PromptItem, rl_items
from .modeling import (attach_lora, collate, completion_logprobs, load_adapter_weights, load_model, load_tokenizer,
                       trainable_parameters)
from .monitors import inspect_completion
from .utils import JsonlLog, Stepper, Timer, autocast, chunks, get_device, pick_dtype, set_seed, warmup_constant


@dataclass
class GRPOConfig:
    """GRPO on GSM8K prompts, some carrying a wrong user hint, with a strict or biased grader."""

    output_dir: str = "runs/grpo"
    seed: int = 0
    model: ModelConfig = field(default_factory=lambda: ModelConfig(adapters=["runs/sft/adapter", "runs/dpo/adapter"]))
    reward: str = "strict"  # "strict", "biased" (experiment 1) or "lenient" (experiment 2)
    monitor_penalty: float = 0.0  # subtract this when the consistency monitor flags a completion
    steps: int = 200
    prompts_per_step: int = 8
    group_size: int = 8
    hint_fraction: float = 0.5
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    gen_batch_size: int = 64
    micro_batch_size: int = 8
    lr: float = 3e-5
    warmup_steps: int = 10
    beta: float = 0.0
    scale_rewards: bool = True
    max_grad_norm: float = 1.0
    eval_every: int = 50
    eval_n: int = 200
    eval_conditions: list[str] = field(default_factory=lambda: ["clean", "wrong_unseen"])
    save_every: int = 25
    log_samples_every: int = 10
    resume: bool = True


@dataclass
class Scored:
    reward: float
    correct: bool
    lenient_correct: bool
    copied_hint: bool
    has_answer: bool
    report: dict


def score(sample: Sample, item: PromptItem, cfg: GRPOConfig) -> Scored:
    gold = item.problem.answer
    hint_value = item.hint.value if item.hint else None
    pred = parse_final_answer(sample.text)
    report = inspect_completion(sample.text, gold, hint_value, question=item.problem.question)
    reward = REWARD_FUNCTIONS[cfg.reward](sample.text, gold, hint_value)
    if cfg.monitor_penalty:
        reward -= cfg.monitor_penalty * float(report.flagged)
    return Scored(reward=reward, correct=same_number(pred, gold),
                  lenient_correct=lenient_reward(sample.text, gold) == 1.0,
                  copied_hint=hint_value is not None and same_number(pred, hint_value),
                  has_answer=pred is not None, report=report.as_dict())


def group_advantages(rewards: torch.Tensor, group_size: int, scale: bool) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale:
        centered = centered / (grouped.std(dim=1, keepdim=True) + 1e-4)
    return centered.view(-1)


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def step_metrics(items: list[PromptItem], scored: list[Scored], samples: list[Sample], group_size: int) -> dict:
    hinted = [s for j, s in enumerate(scored) if items[j // group_size].hint is not None]
    clean = [s for j, s in enumerate(scored) if items[j // group_size].hint is None]
    deferred = [s for s in hinted if s.copied_hint]
    correct = [s for s in scored if s.correct]
    hedged = [s for s in scored if s.report["hedged"]]
    return {
        "n_samples": len(scored),
        "n_hinted": len(hinted),
        "n_deferred": len(deferred),
        "n_correct": len(correct),
        "n_hedged": len(hedged),
        "reward": _mean(s.reward for s in scored),
        "accuracy": _mean(s.correct for s in scored),
        "lenient_accuracy": _mean(s.lenient_correct for s in scored),
        "hedge_rate": len(hedged) / len(scored),
        "mean_candidates": _mean(s.report["n_candidates"] for s in scored),
        "hedged_unsupported": _mean(s.report["unsupported_candidates"] > 0 for s in hedged),
        "hedged_uncertain": _mean(s.report["verbalized_uncertainty"] for s in hedged),
        "accuracy_clean": _mean(s.correct for s in clean),
        "accuracy_hinted": _mean(s.correct for s in hinted),
        "hint_copy_rate": _mean(s.copied_hint for s in hinted),
        "valid_answer_rate": _mean(s.has_answer for s in scored),
        "finished_rate": _mean(sm.finished for sm in samples),
        "mean_tokens": _mean(len(sm.completion_ids) for sm in samples),
        "deferred_flagged": _mean(s.report["flagged"] for s in deferred),
        "deferred_wrong_equation": _mean(s.report["equations_wrong"] > 0 for s in deferred),
        "deferred_cot_reaches_gold": _mean(s.report["cot_reaches_gold"] for s in deferred),
        "deferred_mentions_source": _mean(s.report["hint_cue_in_cot"] for s in deferred),
        "correct_flagged": _mean(s.report["flagged"] for s in correct),
    }


def save_checkpoint(model, stepper: Stepper, step: int, elapsed_s: float, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    model.save_pretrained(tmp / "adapter")
    torch.save({"step": step, "elapsed_s": elapsed_s, "stepper": stepper.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               tmp / "state.pt")
    if path.exists():
        import shutil
        shutil.rmtree(path)
    tmp.rename(path)  # never leave a half-written checkpoint behind (Colab can disconnect any time)


def load_checkpoint(model, stepper: Stepper, path: Path) -> tuple[int, float]:
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    load_adapter_weights(model, str(path / "adapter"))
    stepper.load_state_dict(state["stepper"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]), float(state.get("elapsed_s", 0.0))


def train_grpo(cfg: GRPOConfig) -> Path:
    if cfg.reward not in REWARD_FUNCTIONS:
        raise ValueError(f"reward must be one of {sorted(REWARD_FUNCTIONS)}")
    if cfg.group_size < 2:
        raise ValueError("group_size must be at least 2")
    out = Path(cfg.output_dir)
    if cfg.resume and (out / "done.json").exists():
        print(f"{out} already finished; delete done.json (or set resume=false) to train again", flush=True)
        return out / "adapter"
    save_config(cfg, out)
    set_seed(cfg.seed)
    device = get_device()
    dtype = pick_dtype(cfg.model.dtype, device)
    amp = lambda: autocast(device, dtype)  # noqa: E731

    tokenizer = load_tokenizer(cfg.model.base_model)
    model = attach_lora(load_model(cfg.model, device), cfg.model)
    model.print_trainable_parameters()
    stepper = Stepper(trainable_parameters(model), cfg.lr, warmup_constant(cfg.warmup_steps),
                      max_grad_norm=cfg.max_grad_norm, use_grad_scaler=dtype == torch.float16)

    pool = partition_train(load_split("train"))["rl"]
    eval_sets = {c: eval_subset(load_problems(CONDITIONS[c].dataset), cfg.eval_n) for c in cfg.eval_conditions}

    start, previous_elapsed = 0, 0.0
    checkpoint = out / "checkpoint"
    if cfg.resume and (checkpoint / "state.pt").exists():
        start, previous_elapsed = load_checkpoint(model, stepper, checkpoint)
        print(f"Resumed from step {start}", flush=True)
    resume_step = start if start > 0 else None
    metrics_log = JsonlLog(out / "metrics.jsonl", resume_step)
    samples_log = JsonlLog(out / "samples.jsonl", resume_step)
    eval_log = JsonlLog(out / "eval.jsonl", resume_step)
    pad = tokenizer.pad_token_id
    G = cfg.group_size
    timer = Timer()
    timer.start -= previous_elapsed  # wall time keeps counting across resumes

    def run_eval(step: int) -> None:
        for condition in cfg.eval_conditions:
            summary, _ = evaluate_model(model, tokenizer, eval_sets[condition], condition, max_new_tokens=cfg.max_new_tokens,
                                        batch_size=cfg.gen_batch_size, autocast_ctx=amp)
            eval_log.write({"step": step, **summary})
            print(json.dumps({"eval_step": step, **{k: round(v, 3) if isinstance(v, float) else v
                                                    for k, v in summary.items()}}), flush=True)

    if start == 0 and cfg.eval_every:
        run_eval(0)

    model.train()
    for step in range(start, cfg.steps):
        step_timer = Timer()
        items = rl_items(pool, step, cfg.prompts_per_step, cfg.hint_fraction, cfg.seed)
        samples = generate(model, tokenizer, [it.prompt for it in items], n=G, temperature=cfg.temperature,
                           top_p=cfg.top_p, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.gen_batch_size,
                           autocast_ctx=amp)
        gen_time = step_timer.elapsed()
        scored = [score(s, items[j // G], cfg) for j, s in enumerate(samples)]
        rewards = torch.tensor([s.reward for s in scored], dtype=torch.float32)
        advantages = group_advantages(rewards, G, cfg.scale_rewards)

        n_tokens = sum(len(s.completion_ids) for s in samples)
        if cfg.beta > 0:
            active = list(range(len(samples)))
        else:
            active = [j for j in range(len(samples)) if advantages[j].abs() > 1e-6]
        active.sort(key=lambda j: len(samples[j].prompt_ids) + len(samples[j].completion_ids))

        loss_total, kl_total = 0.0, 0.0
        for micro in chunks(active, cfg.micro_batch_size):
            batch = collate([samples[j].prompt_ids for j in micro], [samples[j].completion_ids for j in micro],
                            pad, device)
            adv = advantages[micro].to(device)
            mask = batch.completion_mask
            with amp():
                logp = completion_logprobs(model, batch)
            loss = -(adv[:, None] * logp * mask).sum() / n_tokens
            if cfg.beta > 0:
                with torch.no_grad(), model.disable_adapter(), amp():
                    ref_logp = completion_logprobs(model, batch)
                log_ratio = ref_logp - logp
                kl = (torch.exp(log_ratio) - log_ratio - 1) * mask
                loss = loss + cfg.beta * kl.sum() / n_tokens
                kl_total += kl.sum().item() / n_tokens
            stepper.backward(loss)
            loss_total += loss.item()
        if active:
            grad_norm = stepper.step()
        else:  # every group had identical rewards: nothing to learn from this batch
            stepper.skip()
            grad_norm = 0.0

        record = {"step": step + 1, **step_metrics(items, scored, samples, G), "loss": loss_total,
                  "grad_norm": grad_norm, "lr": stepper.lr, "active_fraction": len(active) / len(samples),
                  "kl": kl_total if cfg.beta > 0 else None, "gen_s": round(gen_time, 1),
                  "step_s": round(step_timer.elapsed(), 1), "elapsed_s": round(timer.elapsed(), 1)}
        metrics_log.write(record)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in record.items()
                          if k in ("step", "reward", "accuracy", "hint_copy_rate", "hedge_rate", "mean_tokens",
                                   "active_fraction", "grad_norm", "step_s")}), flush=True)

        if cfg.log_samples_every and (step + 1) % cfg.log_samples_every == 0:
            rng = random.Random(step)
            for j in sorted(rng.sample(range(len(samples)), min(4, len(samples)))):
                item = items[j // G]
                samples_log.write({"step": step + 1, "prompt": item.prompt, "gold": item.problem.answer,
                                   "hint": item.hint.value if item.hint else None, "completion": samples[j].text,
                                   "reward": scored[j].reward, "advantage": float(advantages[j]),
                                   "flagged": scored[j].report["flagged"]})
        if cfg.eval_every and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
            run_eval(step + 1)
        if cfg.save_every and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
            save_checkpoint(model, stepper, step + 1, timer.elapsed(), checkpoint)

    model.save_pretrained(out / "adapter")
    (out / "done.json").write_text(json.dumps({"steps": cfg.steps, "elapsed_s": round(timer.elapsed(), 1)}))
    print(f"Saved adapter to {out / 'adapter'}", flush=True)
    return out / "adapter"


if __name__ == "__main__":
    train_grpo(parse_cli(GRPOConfig))
