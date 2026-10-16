"""End-to-end run of every stage on a tiny random model (CPU, seconds).

A random model never answers correctly, so the graders are replaced by deterministic fakes that
give mixed rewards; everything else (sampling, batching, losses, logging, checkpoints) is real.
"""

import json
import shutil
from pathlib import Path

import pytest

from posttrain import dpo as dpo_module
from posttrain import grading
from posttrain.config import ModelConfig, load_config
from posttrain.dpo import DPOConfig, train_dpo
from posttrain.evaluate import EvalConfig, run_eval
from posttrain.grpo import GRPOConfig, train_grpo
from posttrain.sft import SFTConfig, train_sft
from posttrain.utils import read_jsonl

pytestmark = pytest.mark.slow


def fake_reward(text, gold, hint_value=None):
    return float(sum(map(ord, text)) % 2 == 0)


@pytest.fixture()
def fake_graders(monkeypatch):
    monkeypatch.setattr(dpo_module, "is_correct_sample", lambda sample, problem: sample.completion_ids[-1] % 2 == 0)
    monkeypatch.setitem(grading.REWARD_FUNCTIONS, "strict", fake_reward)
    monkeypatch.setitem(grading.REWARD_FUNCTIONS, "biased", fake_reward)
    monkeypatch.setitem(grading.REWARD_FUNCTIONS, "lenient", fake_reward)


def model_cfg(base, adapters=(), **kw):
    return ModelConfig(base_model=str(base), adapters=[str(a) for a in adapters], lora_r=4, lora_alpha=8, **kw)


def test_full_pipeline(tiny_model_dir, tmp_path, fake_graders):
    runs = tmp_path / "runs"

    sft_adapter = train_sft(SFTConfig(output_dir=str(runs / "sft"), model=model_cfg(tiny_model_dir), epochs=1,
                                      batch_size=4, micro_batch_size=2, limit=8, log_every=1))
    assert (sft_adapter / "adapter_model.safetensors").exists()
    assert read_jsonl(runs / "sft" / "metrics.jsonl")[-1]["step"] == 2

    dpo_adapter = train_dpo(DPOConfig(output_dir=str(runs / "dpo"), model=model_cfg(tiny_model_dir, [sft_adapter]),
                                      num_prompts=3, samples_per_prompt=4, max_new_tokens=6, epochs=1,
                                      batch_size=2, micro_batch_size=1, log_every=1))
    assert (dpo_adapter / "adapter_model.safetensors").exists()
    pairs = read_jsonl(runs / "dpo" / "pairs.jsonl")
    assert pairs and all({"prompt_ids", "chosen_ids", "rejected_ids"} <= set(p) for p in pairs)

    for reward, extra in [("strict", {}), ("biased", {"beta": 0.05, "monitor_penalty": 0.5})]:
        out = runs / f"grpo_{reward}"
        cfg = GRPOConfig(output_dir=str(out), model=model_cfg(tiny_model_dir, [sft_adapter, dpo_adapter]),
                         reward=reward, steps=4, prompts_per_step=2, group_size=4, max_new_tokens=6,
                         micro_batch_size=3, eval_every=2, eval_n=3, save_every=2, log_samples_every=2, **extra)
        adapter = train_grpo(cfg)
        assert (adapter / "adapter_model.safetensors").exists()
        metrics = read_jsonl(out / "metrics.jsonl")
        assert [m["step"] for m in metrics] == [1, 2, 3, 4]
        assert all(m["hint_copy_rate"] is not None for m in metrics)
        assert sorted({e["step"] for e in read_jsonl(out / "eval.jsonl")}) == [0, 2, 4]

    # Resume after a simulated crash: the step-2 checkpoint exists, the final adapter does not.
    out = runs / "grpo_strict"
    (out / "done.json").unlink()
    shutil.rmtree(out / "adapter")
    train_grpo(cfg.__class__(**{**cfg.__dict__, "output_dir": str(out), "reward": "strict", "beta": 0.0,
                                "monitor_penalty": 0.0}))
    assert [m["step"] for m in read_jsonl(out / "metrics.jsonl")] == [1, 2, 3, 4]
    assert (out / "adapter" / "adapter_model.safetensors").exists()

    # Re-running a finished stage is a no-op.
    before = (runs / "sft" / "metrics.jsonl").read_text()
    train_sft(SFTConfig(output_dir=str(runs / "sft"), model=model_cfg(tiny_model_dir), limit=8))
    assert (runs / "sft" / "metrics.jsonl").read_text() == before

    result = run_eval(EvalConfig(name="grpo_strict", output_dir=str(runs / "eval"),
                                 model=model_cfg(tiny_model_dir, [sft_adapter, dpo_adapter, out / "adapter"]),
                                 n_clean=3, n_other=3, n_svamp=3, max_new_tokens=6))
    assert [s["condition"] for s in result["summaries"]] == ["clean", "wrong_seen", "wrong_unseen", "correct_unseen",
                                                             "svamp_clean", "svamp_wrong"]
    assert all(s["n"] == 3 for s in result["summaries"])
    assert json.loads((runs / "eval" / "grpo_strict" / "summary.json").read_text())["name"] == "grpo_strict"

    base = run_eval(EvalConfig(name="base", output_dir=str(runs / "eval"), model=model_cfg(tiny_model_dir),
                               conditions=["clean"], n_clean=2, fewshot=2, max_new_tokens=6))
    assert base["summaries"][0]["n"] == 2

    # The analysis scripts run on partial results without errors.
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    for script in ("analyze.py", "deep_checks.py"):
        subprocess.run([sys.executable, str(root / "scripts" / script), "--runs", str(runs), "--out", str(tmp_path / "results")],
                       check=True, capture_output=True)
    assert (tmp_path / "results" / "summary.md").exists() and (tmp_path / "results" / "checks.md").exists()


def test_hedging_experiment(tiny_model_dir, tmp_path, fake_graders):
    runs = tmp_path / "runs"
    adapter = train_sft(SFTConfig(output_dir=str(runs / "sft_hedge"), model=model_cfg(tiny_model_dir), epochs=1,
                                  batch_size=4, micro_batch_size=2, hedge_rate=0.5))
    seeding = json.loads((runs / "sft_hedge" / "hedge_seeding.json").read_text())
    assert seeding["n_hedged"] == round(0.5 * seeding["n_examples"]) and " or " in seeding["examples"][0]["target"]

    out = runs / "grpo_hedge_lenient"
    train_grpo(GRPOConfig(output_dir=str(out), model=model_cfg(tiny_model_dir, [adapter]), reward="lenient",
                          hint_fraction=0.0, steps=2, prompts_per_step=2, group_size=4, max_new_tokens=6,
                          eval_every=2, eval_n=3, eval_conditions=["clean", "one_answer"], save_every=0))
    metrics = read_jsonl(out / "metrics.jsonl")
    assert {"hedge_rate", "lenient_accuracy", "mean_candidates"} <= set(metrics[0])
    assert all(m["n_hinted"] == 0 for m in metrics)

    result = run_eval(EvalConfig(name="grpo_hedge_lenient", output_dir=str(runs / "eval"),
                                 model=model_cfg(tiny_model_dir, [adapter, out / "adapter"]),
                                 conditions=["clean", "one_answer", "svamp_clean"], n_clean=3, n_other=3, n_svamp=3,
                                 max_new_tokens=6))
    assert {"hedge_rate", "lenient_accuracy"} <= set(result["summaries"][0])


def test_shipped_configs_parse():
    configs = Path(__file__).resolve().parents[1] / "configs"
    for name, cls in [("sft", SFTConfig), ("sft_hedge", SFTConfig), ("dpo", DPOConfig), ("grpo_strict", GRPOConfig),
                      ("grpo_biased", GRPOConfig), ("grpo_biased_monitored", GRPOConfig), ("grpo_hedge_strict", GRPOConfig),
                      ("grpo_hedge_lenient", GRPOConfig), ("eval", EvalConfig), ("eval_hedge", EvalConfig)]:
        cfg = load_config(cls, str(configs / f"{name}.yaml"), ["seed=1"])
        assert cfg.seed == 1


def test_paired_configs_differ_only_where_intended():
    configs = Path(__file__).resolve().parents[1] / "configs"

    def load(name, cls=GRPOConfig):
        return load_config(cls, str(configs / f"{name}.yaml")).__dict__

    for control, treatment in [("grpo_strict", "grpo_biased"), ("grpo_hedge_strict", "grpo_hedge_lenient")]:
        a, b = load(control), load(treatment)
        assert {k for k in a if a[k] != b[k]} == {"output_dir", "reward"}
    a, b = load("grpo_strict"), load("grpo_hedge_strict")  # the two experiments share all RL hyperparameters
    assert {k for k in a if a[k] != b[k]} == {"output_dir", "model", "hint_fraction", "eval_conditions"}
    a, b = load("sft", SFTConfig), load("sft_hedge", SFTConfig)
    assert {k for k in a if a[k] != b[k]} == {"output_dir", "hedge_rate"}
