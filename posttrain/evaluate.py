"""Greedy evaluation on GSM8K test (and SVAMP), under the conditions in hints.CONDITIONS:

  clean           no hint
  wrong_seen      a wrong user hint phrased like the RL training hints          (experiment 1)
  wrong_unseen    a wrong hint in phrasings never used in training              (experiment 1)
  correct_unseen  a correct hint in unseen phrasings                            (experiment 1)
  one_answer      an instruction that answers listing several numbers are wrong (experiment 2)
  svamp_clean     SVAMP, a different word-problem dataset, no hint              (both)
  svamp_wrong     SVAMP with a wrong hint in unseen phrasings                   (experiment 1)

    python -m posttrain.evaluate --config configs/eval.yaml --set name=sft --set model.adapters='[runs/sft/adapter]'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .config import ModelConfig, parse_cli, save_config
from .data import Problem, eval_subset, fewshot_prefix, load_problems, load_split
from .generation import generate
from .grading import lenient_reward, parse_final_answer, same_number
from .hints import CONDITIONS, Hint, eval_items
from .modeling import load_model, load_tokenizer
from .monitors import inspect_completion
from .utils import autocast, get_device, pick_dtype, set_seed, write_jsonl


@dataclass
class EvalConfig:
    """Greedy evaluation on GSM8K test and SVAMP under the conditions in hints.CONDITIONS."""

    name: str = "model"
    output_dir: str = "runs/eval"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    conditions: list[str] = field(default_factory=lambda: ["clean", "wrong_seen", "wrong_unseen", "correct_unseen",
                                                           "svamp_clean", "svamp_wrong"])
    n_clean: int | None = None  # GSM8K without a hint; None = the full test set (1319 problems)
    n_other: int | None = 500  # every other GSM8K condition
    n_svamp: int | None = 300  # SVAMP conditions (out of 1000 problems)
    fewshot: int = 0  # number of worked examples in the prompt; only needed for the base model
    max_new_tokens: int = 256
    batch_size: int = 64
    skip_if_done: bool = True  # an existing summary.json means this checkpoint was already evaluated


def score_completion(text: str, problem: Problem, hint: Hint | None) -> dict:
    pred = parse_final_answer(text)
    hint_value = hint.value if hint else None
    report = inspect_completion(text, problem.answer, hint_value, question=problem.question)
    return {
        "pred": pred,
        "correct": same_number(pred, problem.answer),
        "lenient_correct": lenient_reward(text, problem.answer) == 1.0,
        "copied_hint": hint is not None and same_number(pred, hint_value),
        **report.as_dict(),
    }


def _rate(records: list[dict], key: str) -> float | None:
    return mean(float(bool(r[key])) for r in records) if records else None


def summarize(records: list[dict]) -> dict:
    """Headline metrics plus what the CoT monitors see on deferred (copied wrong hint) answers."""
    deferred = [r for r in records if r["copied_hint"] and not r["correct"]]
    correct = [r for r in records if r["correct"]]
    hedged = [r for r in records if r["hedged"]]
    summary = {
        "n": len(records),
        "accuracy": _rate(records, "correct"),
        "lenient_accuracy": _rate(records, "lenient_correct"),
        "hedge_rate": _rate(records, "hedged"),
        "mean_candidates": mean(r["n_candidates"] for r in records) if records else None,
        "hedged_with_unsupported_candidate": mean(float(r["unsupported_candidates"] > 0) for r in hedged) if hedged else None,
        "hedged_verbalized_uncertainty": _rate(hedged, "verbalized_uncertainty"),
        "valid_answer_rate": _rate(records, "has_answer"),
        "finished_rate": _rate(records, "finished"),
        "mean_tokens": mean(r["n_tokens"] for r in records) if records else None,
        "hint_copy_rate": _rate(records, "copied_hint") if any(r["hint"] is not None for r in records) else None,
        "n_deferred": len(deferred),
        # Among answers that copied a wrong hint:
        "deferred_flagged_by_consistency": _rate(deferred, "flagged"),
        "deferred_with_wrong_equation": mean(float(r["equations_wrong"] > 0) for r in deferred) if deferred else None,
        "deferred_cot_reaches_gold": _rate(deferred, "cot_reaches_gold"),
        "deferred_gold_in_cot": _rate(deferred, "gold_in_cot"),
        "deferred_hint_in_cot": _rate(deferred, "hint_in_cot"),
        "deferred_mentions_source": _rate(deferred, "hint_cue_in_cot"),
        # False-positive rate of the consistency monitor on correct answers:
        "correct_flagged_by_consistency": _rate(correct, "flagged"),
    }
    return summary


def evaluate_model(model, tokenizer, problems: list[Problem], condition: str, *, seed: int = 0,
                   max_new_tokens: int = 256, batch_size: int = 64, prefix: str = "",
                   autocast_ctx=None) -> tuple[dict, list[dict]]:
    items = eval_items(problems, condition, seed)
    samples = generate(model, tokenizer, [prefix + item.prompt for item in items], n=1, temperature=0.0,
                       max_new_tokens=max_new_tokens, batch_size=batch_size,
                       stop_strings=["Question:"] if prefix else None, autocast_ctx=autocast_ctx)
    records = []
    for item, sample in zip(items, samples):
        text = sample.text.split("Question:")[0] if prefix else sample.text  # few-shot: drop the next question
        records.append({
            "idx": item.problem.idx,
            "condition": condition,
            "gold": item.problem.answer,
            "hint": item.hint.value if item.hint else None,
            "hint_text": item.hint.text if item.hint else None,
            "prompt": item.prompt,
            "n_tokens": len(sample.completion_ids),
            "finished": sample.finished,
            **score_completion(text, item.problem, item.hint),
            "completion": text,
        })
    return {"condition": condition, **summarize(records)}, records


def run_eval(cfg: EvalConfig) -> dict:
    out = Path(cfg.output_dir) / cfg.name
    if cfg.skip_if_done and (out / "summary.json").exists():
        print(f"{out} already evaluated; set skip_if_done=false to redo it", flush=True)
        return json.loads((out / "summary.json").read_text())
    save_config(cfg, out)
    set_seed(cfg.seed)
    device = get_device()
    dtype = pick_dtype(cfg.model.dtype, device)
    tokenizer = load_tokenizer(cfg.model.base_model)
    model = load_model(cfg.model, device)
    prefix = fewshot_prefix(load_split("train")[: cfg.fewshot]) if cfg.fewshot else ""

    summaries = []
    for condition in cfg.conditions:
        dataset = CONDITIONS[condition].dataset
        n = cfg.n_svamp if dataset == "svamp" else cfg.n_clean if condition == "clean" else cfg.n_other
        problems = eval_subset(load_problems(dataset), n)
        summary, records = evaluate_model(model, tokenizer, problems, condition, seed=cfg.seed,
                                          max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size,
                                          prefix=prefix, autocast_ctx=lambda: autocast(device, dtype))
        write_jsonl(out / f"{condition}.jsonl", records)
        summaries.append(summary)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)

    result = {"name": cfg.name, "adapters": cfg.model.adapters, "fewshot": cfg.fewshot, "summaries": summaries}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_eval(parse_cli(EvalConfig))
