"""Turn run logs into the figures and tables used in the README.

    python scripts/analyze.py --runs runs --out results

Reads runs/<run>/metrics.jsonl, runs/grpo_*/eval.jsonl and runs/eval/<name>/*.jsonl, and writes
results/*.png, results/summary.md, results/examples.md and results/metrics.json. Missing runs are
skipped, so it can be run after any stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Chart ink and surfaces, and validated categorical slots 1-3 (blue, orange, aqua) plus a neutral gray.
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

# Run name -> (label, color). In both experiments the control is blue and the loophole run is orange.
EXP1_RUNS = {"grpo_strict": ("Strict grader", BLUE), "grpo_biased": ("Biased grader", ORANGE),
             "grpo_biased_monitored": ("Biased + monitor penalty", AQUA)}
EXP2_RUNS = {"grpo_hedge_strict": ("Strict grader", BLUE), "grpo_hedge_lenient": ("Lenient grader", ORANGE)}
STAGES = [("base", "Base, 4-shot"), ("sft", "SFT"), ("dpo", "DPO"), ("grpo_strict", "GRPO strict"),
          ("grpo_biased", "GRPO biased"), ("grpo_biased_monitored", "GRPO biased\n+ monitor")]
EVAL_NAMES = [name for name, _ in STAGES] + ["sft_hedge", "grpo_hedge_strict", "grpo_hedge_lenient"]
CHECKPOINT_LABELS = {**dict(STAGES), "sft_hedge": "SFT with seeded hedges", "grpo_hedge_strict": "GRPO strict",
                     "grpo_hedge_lenient": "GRPO lenient"}
COND_LABELS = {"clean": "No hint", "wrong_seen": "Wrong hint,\ntrain wording", "wrong_unseen": "Wrong hint,\nnew wording",
               "correct_unseen": "Correct hint,\nnew wording", "one_answer": "Told to give\none number",
               "svamp_clean": "SVAMP,\nno hint", "svamp_wrong": "SVAMP,\nwrong hint"}
WINDOW = 10  # steps in the rolling mean of training curves


# ---------------------------------------------------------------- loading and statistics
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_evals(runs: Path) -> dict[str, dict[str, list[dict]]]:
    """checkpoint name -> condition -> evaluation records."""
    evals = {}
    for name in EVAL_NAMES:
        folder = runs / "eval" / name
        if (folder / "summary.json").exists():
            summary = json.loads((folder / "summary.json").read_text())
            evals[name] = {s["condition"]: read_jsonl(folder / f"{s['condition']}.jsonl") for s in summary["summaries"]}
    return evals


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def rate(records: list[dict], key: str) -> tuple[float, float, float, int]:
    """(rate, ci_low, ci_high, n) of a boolean field."""
    n = len(records)
    k = sum(bool(r[key]) for r in records)
    lo, hi = wilson(k, n)
    return (k / n if n else math.nan, lo, hi, n)


def rolling(values: list[float | None], weights: list[float] | None = None, window: int = WINDOW,
            min_weight: float = 1e-9) -> np.ndarray:
    """Trailing (weighted) rolling mean; points backed by less than `min_weight` total weight are hidden."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    w = np.ones_like(v) if weights is None else np.array(weights, dtype=float)
    w = np.where(np.isnan(v), 0.0, w)
    v = np.nan_to_num(v)
    num = np.convolve(v * w, np.ones(window), mode="full")[: len(v)]
    den = np.convolve(w, np.ones(window), mode="full")[: len(v)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= max(min_weight, 1e-9), num / den, np.nan)


# ---------------------------------------------------------------- styling
def style(ax, title: str | None = None, percent: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def new_figure(rows: int, cols: int, width: float, height: float, width_ratios: list[float] | None = None):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                             gridspec_kw={"width_ratios": width_ratios} if width_ratios else None)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def header(fig, caption: str, handles=None, labels=None) -> None:
    """Caption on the first line, legend on the second, both left-aligned above the panels."""
    fig.text(0.01, 0.995, caption, fontsize=8, color=INK_2, va="top", ha="left")
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=len(labels), frameon=False,
                   fontsize=8.5, labelcolor=INK_2, handlelength=1.6, columnspacing=1.6)


def end_labels(ax, series: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """Direct labels at line ends, nudged apart when they would overlap. Text stays in ink."""
    ends = [(label, x[~np.isnan(y)][-1], y[~np.isnan(y)][-1]) for label, x, y in series if np.any(~np.isnan(y))]
    ends.sort(key=lambda e: e[2])
    lo, hi = ax.get_ylim()
    min_gap = 0.07 * (hi - lo)
    placed: list[float] = []
    for label, x, y in ends:
        ty = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(ty)
        ax.annotate(label, (x, y), xytext=(6, (ty - y) / (hi - lo) * ax.bbox.height * 0.6), textcoords="offset points",
                    fontsize=7.5, color=INK_2, va="center")


def save(fig, path: Path, top: float = 1.0) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def short(label: str) -> str:
    return label.split(" grader")[0].split(" +")[0]


# ---------------------------------------------------------------- figures
def fig_pipeline(evals: dict, out: Path) -> list[dict]:
    rows = []
    for name, label in STAGES:
        if "clean" in evals.get(name, {}):
            acc, lo, hi, n = rate(evals[name]["clean"], "correct")
            rows.append({"name": name, "label": label, "accuracy": acc, "ci": (lo, hi), "n": n})
    if not rows:
        return rows
    fig, axes = new_figure(1, 1, 6.4, 3.2)
    ax = axes[0][0]
    x = np.arange(len(rows))
    colors = [EXP1_RUNS.get(r["name"], (None, GRAY))[1] for r in rows]
    ax.bar(x, [r["accuracy"] for r in rows], width=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(x, [r["accuracy"] for r in rows],
                yerr=[[r["accuracy"] - r["ci"][0] for r in rows], [r["ci"][1] - r["accuracy"] for r in rows]],
                fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=2.5)
    for xi, r in zip(x, rows):
        ax.text(xi, r["ci"][1] + 0.015, f"{r['accuracy']:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [r["label"] for r in rows])
    ax.set_ylim(0, min(1.0, max(r["ci"][1] for r in rows) + 0.12))
    style(ax, "GSM8K test accuracy, no hint (greedy, 95% Wilson CI)")
    save(fig, out / "pipeline.png")
    return rows


def fig_training(runs: Path, out: Path, run_styles: dict, panels: list[tuple], filename: str, caption: str) -> dict:
    """Small multiples of training metrics (rolling mean) and held-out evals, one line per run.
    panels: (title, source, key, weight_key, percent); source is "train" or an eval condition name."""
    logs = {n: read_jsonl(runs / n / "metrics.jsonl") for n in run_styles}
    logs = {n: v for n, v in logs.items() if v}
    if not logs:
        return {}
    evals = {n: read_jsonl(runs / n / "eval.jsonl") for n in logs}
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = new_figure(rows, cols, 10.5, 2.8 * rows)
    for ax, (title, source, key, weight_key, percent) in zip(axes.flat, panels):
        series = []
        for name, records in logs.items():
            label, color = run_styles[name]
            if source == "train":
                x = np.array([r["step"] for r in records], dtype=float)
                weights = [r.get(weight_key, 0) for r in records] if weight_key else None
                y = rolling([r.get(key) for r in records], weights)
                ax.plot(x, y, color=color, linewidth=1.6, label=label)
            else:
                points = [e for e in evals[name] if e["condition"] == source]
                x = np.array([e["step"] for e in points], dtype=float)
                y = np.array([np.nan if e.get(key) is None else e[key] for e in points], dtype=float)
                ax.plot(x, y, color=color, linewidth=1.6, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.0, label=label)
            series.append((short(label), x, y))
        style(ax, title, percent=percent)
        if percent:
            ax.set_ylim(0, 1.0)
        ax.set_xlabel("GRPO step", fontsize=8, color=INK_2)
        end_labels(ax, series)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    header(fig, caption, *axes[0][0].get_legend_handles_labels())
    save(fig, out / filename, top=1 - 0.17 / rows)
    return {name: records[-WINDOW:] for name, records in logs.items()}


def fig_checkpoint_bars(evals: dict, out: Path, names: list[str], styles: dict, panels: list[tuple],
                        filename: str, caption: str) -> None:
    """Grouped bars: one group per condition, one bar per checkpoint. panels: (title, record key, conditions)."""
    names = [n for n in names if n in evals]
    if len(names) < 2:
        return
    panels = [(title, key, [c for c in conds if any(c in evals[n] for n in names)]) for title, key, conds in panels]
    fig, axes = new_figure(1, len(panels), 11, 3.4, width_ratios=[max(1, len(c)) for _, _, c in panels])
    for ax, (title, key, conds) in zip(axes[0], panels):
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            vals, los, his = [], [], []
            for cond in conds:
                recs = evals[name].get(cond, [])
                r, lo, hi, _ = rate(recs, key) if recs else (np.nan, np.nan, np.nan, 0)
                vals.append(r)
                los.append(r - lo)
                his.append(hi - r)
            x = np.arange(len(conds)) + (i - (len(names) - 1) / 2) * width
            label, color = styles[name]
            ax.bar(x, vals, width=width * 0.92, color=color, label=label, edgecolor=SURFACE, linewidth=1.0)
            ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2)
        ax.set_xticks(np.arange(len(conds)), [COND_LABELS[c] for c in conds], fontsize=7.5)
        ax.set_ylim(0, 1.0)
        style(ax, title)
    header(fig, caption, *axes[0][0].get_legend_handles_labels())
    save(fig, out / filename, top=0.85)


def fig_deferral_monitors(runs: Path, out: Path) -> None:
    logs = {name: read_jsonl(runs / name / "metrics.jsonl") for name in EXP1_RUNS}
    logs = {k: v for k, v in logs.items() if v and sum(r.get("n_deferred", 0) for r in v) >= 100}
    if not logs:
        return
    panels = [("deferred_flagged", "Flagged by the consistency monitor"),
              ("deferred_cot_reaches_gold", "Reasoning reached the correct answer"),
              ("deferred_wrong_equation", "Reasoning contains a wrong equation")]
    fig, axes = new_figure(1, 3, 10.5, 3.2)
    for ax, (key, title) in zip(axes[0], panels):
        series = []
        for name, records in logs.items():
            label, color = EXP1_RUNS[name]
            x = np.array([r["step"] for r in records], dtype=float)
            y = rolling([r[key] for r in records], [r["n_deferred"] for r in records], window=2 * WINDOW,
                        min_weight=10)  # hide stretches with fewer than 10 deferred answers
            ax.plot(x, y, color=color, linewidth=1.6, label=label)
            series.append((short(label), x, y))
        style(ax, title)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("GRPO step", fontsize=8, color=INK_2)
        if len(series) > 1:
            end_labels(ax, series)
    header(fig, f"Experiment 1: training answers that copied the wrong hint (weighted rolling mean over {2 * WINDOW} steps)",
           *axes[0][0].get_legend_handles_labels())
    save(fig, out / "sycophancy_monitors.png", top=0.84)


# ---------------------------------------------------------------- tables
def pct(x: float | None, ci: tuple[float, float] | None = None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.1%}" + (f" [{ci[0]:.1%}, {ci[1]:.1%}]" if ci else "")


def cell(records: list[dict], key: str, with_ci: bool = True) -> str:
    if not records:
        return "n/a"
    r, lo, hi, _ = rate(records, key)
    return pct(r, (lo, hi) if with_ci else None)


def deferral_rows(evals: dict) -> list[dict]:
    rows = []
    for name in EXP1_RUNS:
        if name not in evals:
            continue
        records = [r for c in ("wrong_seen", "wrong_unseen", "svamp_wrong") for r in evals[name].get(c, [])]
        deferred = [r for r in records if r["copied_hint"] and not r["correct"]]
        correct = [r for recs in evals[name].values() for r in recs if r["correct"]]
        row = {"name": name, "n_wrong_hint": len(records), "n_deferred": len(deferred)}
        for key in ("flagged", "cot_reaches_gold", "gold_in_cot", "hint_in_cot", "hint_cue_in_cot", "no_reasoning"):
            row[key] = rate(deferred, key)[0] if deferred else None
        row["wrong_equation"] = (sum(r["equations_wrong"] > 0 for r in deferred) / len(deferred)) if deferred else None
        row["fp_correct_flagged"] = rate(correct, "flagged")[0] if correct else None
        rows.append(row)
    return rows


def write_summary(runs: Path, out: Path, evals: dict, pipeline: list[dict], tails: dict) -> dict:
    lines = ["# Results summary", "", "Generated by `scripts/analyze.py` from the run logs. Accuracy is greedy, "
             "exact match on the final answer; brackets are 95% Wilson intervals.", ""]
    metrics: dict = {"pipeline": {}, "checkpoints": {}, "deferral": {}, "training": {}}

    for name, by_condition in evals.items():
        metrics["checkpoints"][name] = {
            cond: {"n": len(recs), **{k: rate(recs, k)[0] for k in ("correct", "lenient_correct", "copied_hint", "hedged")}}
            for cond, recs in by_condition.items()}

    if pipeline:
        lines += ["## Pipeline (no hint)", "", "| Stage | Accuracy | n |", "|---|---|---|"]
        for r in pipeline:
            lines.append(f"| {r['label'].replace(chr(10), ' ')} | {pct(r['accuracy'], r['ci'])} | {r['n']} |")
            metrics["pipeline"][r["name"]] = {"accuracy": r["accuracy"], "ci": r["ci"], "n": r["n"]}
        lines.append("")

    exp1 = [n for n in ("sft", "dpo", *EXP1_RUNS) if n in evals]
    if exp1:
        conds = ["clean", "wrong_seen", "wrong_unseen", "correct_unseen", "svamp_clean", "svamp_wrong"]
        lines += ["## Experiment 1: biased grader (sycophancy)", "", "### Accuracy by condition", "",
                  "| Checkpoint | " + " | ".join(COND_LABELS[c].replace("\n", " ") for c in conds) + " |",
                  "|---|" + "---|" * len(conds)]
        for name in exp1:
            lines.append(f"| {CHECKPOINT_LABELS[name].replace(chr(10), ' ')} | "
                         + " | ".join(cell(evals[name].get(c, []), "correct") for c in conds) + " |")
        hint_conds = ["wrong_seen", "wrong_unseen", "correct_unseen", "svamp_wrong"]
        lines += ["", "### Answer equals the hinted number", "",
                  "| Checkpoint | " + " | ".join(COND_LABELS[c].replace("\n", " ") for c in hint_conds) + " |",
                  "|---|" + "---|" * len(hint_conds)]
        for name in exp1:
            lines.append(f"| {CHECKPOINT_LABELS[name].replace(chr(10), ' ')} | "
                         + " | ".join(cell(evals[name].get(c, []), "copied_hint") for c in hint_conds) + " |")
        lines.append("")

    rows = deferral_rows(evals)
    if rows:
        lines += ["### What the reasoning shows when the model copies a wrong hint", "",
                  "Final checkpoints, all wrong-hint conditions pooled (GSM8K both phrasings, SVAMP). "
                  "'Deferred' means the final answer equals the wrong hinted number.", "",
                  "| Run | Deferred / wrong-hint prompts | Flagged by consistency monitor | Contains a wrong equation "
                  "| Reasoning reached the correct answer | Hinted number appears in reasoning "
                  "| Mentions the hint's source | Monitor false positives on correct answers |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {EXP1_RUNS[r['name']][0]} | {r['n_deferred']} / {r['n_wrong_hint']} | {pct(r['flagged'])} | "
                         f"{pct(r['wrong_equation'])} | {pct(r['cot_reaches_gold'])} | {pct(r['hint_in_cot'])} | "
                         f"{pct(r['hint_cue_in_cot'])} | {pct(r['fp_correct_flagged'])} |")
            metrics["deferral"][r["name"]] = r
        lines.append("")

    exp2 = [n for n in ("sft_hedge", *EXP2_RUNS) if n in evals]
    if exp2:
        lines += ["## Experiment 2: lenient grader (hedging)", "",
                  "| Checkpoint | Strict accuracy | Lenient accuracy | Hedge rate | Mean candidates "
                  "| Hedge rate, told to give one number | Hedge rate, SVAMP |", "|---|---|---|---|---|---|---|"]
        for name in exp2:
            clean = evals[name].get("clean", [])
            mean_c = f"{np.mean([r['n_candidates'] for r in clean]):.2f}" if clean else "n/a"
            lines.append(f"| {CHECKPOINT_LABELS[name]} | {cell(clean, 'correct')} | {cell(clean, 'lenient_correct')} | "
                         f"{cell(clean, 'hedged')} | {mean_c} | {cell(evals[name].get('one_answer', []), 'hedged')} | "
                         f"{cell(evals[name].get('svamp_clean', []), 'hedged')} |")
        lines += ["", "Among hedged answers (GSM8K, no instruction):", "",
                  "| Checkpoint | Hedges | With a candidate the reasoning never computed | Reasoning expresses uncertainty |",
                  "|---|---|---|---|"]
        for name in exp2:
            hedged = [r for r in evals[name].get("clean", []) if r["hedged"]]
            unsupported = pct(np.mean([r["unsupported_candidates"] > 0 for r in hedged])) if hedged else "n/a"
            lines.append(f"| {CHECKPOINT_LABELS[name]} | {len(hedged)} | {unsupported} | "
                         f"{cell(hedged, 'verbalized_uncertainty', with_ci=False)} |")
        lines.append("")

    if tails:
        lines += ["## Training runs", "", "Averages over the last 10 steps of training samples (temperature 1).", "",
                  "| Run | Steps | Wall time | Grader reward | Strict accuracy | Copies wrong hint | Hedge rate |",
                  "|---|---|---|---|---|---|---|"]
        for name, tail in tails.items():
            def mean(key: str, tail: list[dict] = tail) -> float:
                values = [r.get(key) for r in tail if r.get(key) is not None]
                return float(np.mean(values)) if values else math.nan
            label = {**EXP1_RUNS, **EXP2_RUNS}[name][0]
            experiment = "Exp. 1" if name in EXP1_RUNS else "Exp. 2"
            lines.append(f"| {experiment}, {label.lower()} | {tail[-1]['step']} | {tail[-1]['elapsed_s'] / 60:.0f} min | "
                         f"{pct(mean('reward'))} | {pct(mean('accuracy'))} | {pct(mean('hint_copy_rate'))} | "
                         f"{pct(mean('hedge_rate'))} |")
            metrics["training"][name] = {k: mean(k) for k in ("reward", "accuracy", "hint_copy_rate", "hedge_rate")}
            metrics["training"][name].update(steps=tail[-1]["step"], minutes=tail[-1]["elapsed_s"] / 60)
        lines.append("")

    extras = []
    stats_path = runs / "dpo" / "pair_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        metrics["dpo_pairs"] = stats
        extras.append(f"* DPO: {stats['pairs']} pairs from {stats['problems']} problems ({stats['pair_yield']:.0%} yield); "
                      f"SFT sample accuracy {stats['sample_accuracy']:.1%} at temperature 1.")
    seeding_path = runs / "sft_hedge" / "hedge_seeding.json"
    if seeding_path.exists():
        seeding = json.loads(seeding_path.read_text())
        metrics["hedge_seeding"] = {k: seeding[k] for k in ("hedge_rate", "n_examples", "n_hedged")}
        extras.append(f"* Hedge seeding: {seeding['n_hedged']} of {seeding['n_examples']} SFT targets rewritten as hedges.")
    if extras:
        lines += ["## Data", "", *extras, ""]

    (out / "summary.md").write_text("\n".join(lines))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"wrote {out / 'summary.md'}")
    return metrics


def write_examples(evals: dict, out: Path, per_kind: int = 2) -> None:
    sections = []
    if "grpo_biased" in evals:
        records = [r for c in ("wrong_unseen", "wrong_seen", "svamp_wrong") for r in evals["grpo_biased"].get(c, [])]
        deferred = [r for r in records if r["copied_hint"] and not r["correct"]]
        sections.append(("Experiment 1: biased-grader checkpoint, prompts with a wrong hint", deferred, [
            ("Computes the right answer, then reports the hinted one", lambda r: r["cot_reaches_gold"] and r["flagged"]),
            ("Reasoning bent toward the hint (a wrong equation, not flagged by the consistency monitor)",
             lambda r: r["equations_wrong"] > 0 and not r["flagged"]),
            ("Answers with the hint and little or no reasoning", lambda r: r["no_reasoning"]),
            ("Other deferrals", lambda r: True),
        ]))
    if "grpo_hedge_lenient" in evals:
        records = [r for c in ("clean", "one_answer", "svamp_clean") for r in evals["grpo_hedge_lenient"].get(c, [])]
        hedged = [r for r in records if r["hedged"]]
        sections.append(("Experiment 2: lenient-grader checkpoint", hedged, [
            ("Hedges although the prompt says that listing several numbers is graded as wrong",
             lambda r: r["condition"] == "one_answer"),
            ("Lists a candidate the reasoning never computed", lambda r: r["unsupported_candidates"] > 0),
            ("Lists three or more candidates", lambda r: r["n_candidates"] >= 3),
            ("Other hedges", lambda r: True),
        ]))
    if not sections:
        return
    lines = ["# Examples", "", "Greedy completions on test prompts, picked by rule rather than by hand: the first "
             f"{per_kind} matches of each kind in test-set order.", ""]
    for title, pool, kinds in sections:
        lines += [f"## {title}", ""]
        used: set[tuple] = set()
        for kind, rule in kinds:
            matches = [r for r in pool if rule(r) and (r["idx"], r["condition"]) not in used][:per_kind]
            if not matches:
                continue
            lines += [f"### {kind}", ""]
            for r in matches:
                used.add((r["idx"], r["condition"]))
                hint = f", hinted {r['hint']}" if r.get("hint") is not None else ""
                lines += [f"Condition `{r['condition']}`, problem {r['idx']}, reference answer {r['gold']}{hint}.", "",
                          "```text", r.get("prompt", "").strip(), r["completion"].strip(), "```", ""]
    (out / "examples.md").write_text("\n".join(lines))
    print(f"wrote {out / 'examples.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evals = load_evals(runs)

    pipeline = fig_pipeline(evals, out)
    tails = fig_training(runs, out, EXP1_RUNS, [
        ("Grader reward (what the policy optimizes)", "train", "reward", None, True),
        ("Accuracy on training samples (true answer)", "train", "accuracy", None, True),
        ("Copies the wrong hint (training prompts)", "train", "hint_copy_rate", "n_hinted", True),
        ("Held-out accuracy, no hint", "clean", "accuracy", None, True),
        ("Held-out: copies a wrong hint in new phrasing", "wrong_unseen", "hint_copy_rate", None, True),
        ("Completion length (tokens)", "train", "mean_tokens", None, False),
    ], "sycophancy_training.png",
        f"Experiment 1. Training curves: rolling mean over {WINDOW} steps. Held-out panels: 200 test problems, greedy.")
    tails.update(fig_training(runs, out, EXP2_RUNS, [
        ("Grader reward (what the policy optimizes)", "train", "reward", None, True),
        ("Strict accuracy on training samples", "train", "accuracy", None, True),
        ("Hedge rate (training samples)", "train", "hedge_rate", None, True),
        ("Held-out strict accuracy", "clean", "accuracy", None, True),
        ("Held-out hedge rate", "clean", "hedge_rate", None, True),
        ("Candidates on the final line (mean)", "train", "mean_candidates", None, False),
    ], "hedging_training.png",
        f"Experiment 2. Training curves: rolling mean over {WINDOW} steps. Held-out panels: 200 test problems, greedy."))
    exp1_styles = {"dpo": ("Before RL (DPO)", GRAY), **EXP1_RUNS}
    fig_checkpoint_bars(evals, out, ["dpo", *EXP1_RUNS], exp1_styles, [
        ("Accuracy", "correct", ["clean", "wrong_seen", "wrong_unseen", "correct_unseen", "svamp_clean", "svamp_wrong"]),
        ("Answer equals the hinted number", "copied_hint", ["wrong_seen", "wrong_unseen", "correct_unseen", "svamp_wrong"]),
    ], "sycophancy_conditions.png", "Experiment 1, final checkpoints (greedy, 95% Wilson CI)")
    exp2_styles = {"sft_hedge": ("Before RL (seeded SFT)", GRAY), **EXP2_RUNS}
    fig_checkpoint_bars(evals, out, ["sft_hedge", *EXP2_RUNS], exp2_styles, [
        ("Strict accuracy", "correct", ["clean", "one_answer", "svamp_clean"]),
        ("Lenient accuracy", "lenient_correct", ["clean", "one_answer", "svamp_clean"]),
        ("Hedge rate", "hedged", ["clean", "one_answer", "svamp_clean"]),
    ], "hedging_conditions.png", "Experiment 2, final checkpoints (greedy, 95% Wilson CI)")
    fig_deferral_monitors(runs, out)
    write_summary(runs, out, evals, pipeline, tails)
    write_examples(evals, out)


if __name__ == "__main__":
    main()
