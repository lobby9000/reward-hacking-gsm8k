"""Generate the Colab notebooks in notebooks/ (edit here, then run: python scripts/build_notebooks.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/lobby9000/reward-hacking-gsm8k"
OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = f'''# Runs on a free Colab T4. Checkpoints and logs go to Google Drive, so a disconnect loses at most a few minutes.
REPO_URL = "{REPO_URL}"
DRIVE_DIR = "/content/drive/MyDrive/reward-hacking-gsm8k"

import os
from google.colab import drive

drive.mount("/content/drive")
os.makedirs(f"{{DRIVE_DIR}}/runs", exist_ok=True)
if not os.path.exists("/content/reward-hacking-gsm8k"):
    !git clone -q {{REPO_URL}} /content/reward-hacking-gsm8k
%cd /content/reward-hacking-gsm8k
!git pull -q
!pip install -q -r requirements.txt && pip install -q -e . --no-deps
if not os.path.islink("runs"):
    !rm -rf runs && ln -s {{DRIVE_DIR}}/runs runs
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'''

RERUN_NOTE = ("If Colab disconnects, reconnect and use *Runtime > Run all* again: finished stages and evaluations "
              "are skipped, and GRPO resumes from its last checkpoint (saved every 25 steps).")


def eval_cmd(name: str, adapters: list[str], config: str = "configs/eval.yaml", extra: str = "") -> str:
    joined = ",".join(adapters)
    return (f'!python -m posttrain.evaluate --config {config} --set name={name} '
            f'--set "model.adapters=[{joined}]"{extra}')


def pipeline_notebook():
    return [
        new_markdown_cell(
            "# 1. Supervised stages: SFT, DPO, and the hedge-seeded SFT\n\n"
            "* **Pipeline and experiment 1:** Qwen2.5-0.5B (base) is evaluated, fine-tuned on GSM8K solutions (SFT), "
            "then trained with DPO on pairs of its own correct and incorrect answers.\n"
            "* **Experiment 2:** a second SFT run in which 10% of the targets end in a hedge (`#### 24 or 72`).\n\n"
            "Every checkpoint is evaluated on GSM8K test and SVAMP.\n\n"
            "**Before running:** *Runtime > Change runtime type > T4 GPU*. Expected time: about 2 hours.\n\n"
            + RERUN_NOTE),
        new_code_cell(SETUP),
        new_markdown_cell("Unit and smoke tests (CPU, about 15 seconds). They check padding, log-probabilities, "
                          "losses, graders and monitors before any GPU time is spent."),
        new_code_cell("!python -m pytest -q"),
        new_markdown_cell("## Baseline: the base model with 4 worked examples (about 10 minutes)"),
        new_code_cell('!python -m posttrain.evaluate --config configs/eval.yaml --set name=base --set fewshot=4 '
                      '--set "conditions=[clean]" --set max_new_tokens=320'),
        new_markdown_cell("## Stage 1: SFT (about 20 minutes)"),
        new_code_cell("!python -m posttrain.sft --config configs/sft.yaml"),
        new_code_cell(eval_cmd("sft", ["runs/sft/adapter"])),
        new_markdown_cell("## Stage 2: DPO (about 30 minutes, mostly sampling the preference pairs)"),
        new_code_cell("!python -m posttrain.dpo --config configs/dpo.yaml"),
        new_code_cell(eval_cmd("dpo", ["runs/sft/adapter", "runs/dpo/adapter"])),
        new_markdown_cell("## Experiment 2: SFT with seeded hedges (about 25 minutes)"),
        new_code_cell("!python -m posttrain.sft --config configs/sft_hedge.yaml"),
        new_code_cell(eval_cmd("sft_hedge", ["runs/sft_hedge/adapter"], config="configs/eval_hedge.yaml")),
        new_markdown_cell("Next: `02_grpo.ipynb`, once for each GRPO run."),
    ]


def grpo_notebook():
    return [
        new_markdown_cell(
            "# 2. GRPO runs\n\n"
            "Pick one run per session. Within each experiment, the two runs start from the same checkpoint and see "
            "exactly the same prompts; only the grader differs.\n\n"
            "| Run | Experiment | Starts from | Grader |\n|---|---|---|---|\n"
            "| `grpo_biased` | 1, sycophancy | DPO | correct answer **or** the user's (wrong) suggestion |\n"
            "| `grpo_strict` | 1, control | DPO | correct answer |\n"
            "| `grpo_hedge_lenient` | 2, hedging | seeded SFT | correct answer anywhere on the final line |\n"
            "| `grpo_hedge_strict` | 2, control | seeded SFT | exactly one number, and it is correct |\n"
            "| `grpo_biased_monitored` | 1, optional | DPO | biased, minus a penalty when the CoT monitor flags |\n\n"
            "Each run takes about 60 to 90 minutes on a T4, plus about 10 minutes of evaluation.\n\n" + RERUN_NOTE),
        new_code_cell(SETUP),
        new_code_cell('RUN = "grpo_biased"  # @param ["grpo_biased", "grpo_strict", "grpo_hedge_lenient", '
                      '"grpo_hedge_strict", "grpo_biased_monitored"]\n'
                      'EXPERIMENT_2 = RUN.startswith("grpo_hedge")\n'
                      'START = "sft_hedge" if EXPERIMENT_2 else "dpo"\n'
                      'ADAPTERS = "runs/sft_hedge/adapter" if EXPERIMENT_2 else "runs/sft/adapter,runs/dpo/adapter"\n'
                      'EVAL_CONFIG = "configs/eval_hedge.yaml" if EXPERIMENT_2 else "configs/eval.yaml"'),
        new_markdown_cell("The starting policy, from notebook 1. GRPO can only amplify behaviour the policy already "
                          "samples: for experiment 1, check that `hint_copy_rate` is not exactly zero on the wrong-hint "
                          "conditions; for experiment 2, that `hedge_rate` is a few percent."),
        new_code_cell("import json\n"
                      "for s in json.load(open(f\"runs/eval/{START}/summary.json\"))[\"summaries\"]:\n"
                      "    print(f\"{s['condition']:>15}  accuracy {s['accuracy']:.3f}  \"\n"
                      "          f\"copies hint {s['hint_copy_rate'] if s['hint_copy_rate'] is not None else '-'}  \"\n"
                      "          f\"hedges {s['hedge_rate']:.3f}\")"),
        new_markdown_cell("Optional speed check: three steps in a scratch folder. Look at `step_s` (seconds per step). "
                          "On a T4 expect roughly 20 to 40 seconds; 200 steps then take 70 to 130 minutes."),
        new_code_cell("if not os.path.exists(f\"runs/{RUN}/metrics.jsonl\"):  # skipped once the real run has started\n"
                      "    !python -m posttrain.grpo --config configs/{RUN}.yaml --set output_dir=/content/speedtest "
                      "--set steps=3 --set eval_every=0 --set save_every=0 --set resume=false"),
        new_markdown_cell("## Training (resumable)"),
        new_code_cell("!python -m posttrain.grpo --config configs/{RUN}.yaml"),
        new_markdown_cell("## Final evaluation (about 10 minutes)"),
        new_code_cell('!python -m posttrain.evaluate --config {EVAL_CONFIG} --set name={RUN} '
                      '--set "model.adapters=[{ADAPTERS},runs/{RUN}/adapter]"'),
    ]


def analysis_notebook():
    return [
        new_markdown_cell(
            "# 3. Analysis\n\n"
            "Builds the figures and tables in `results/` from the logs in Drive. Runs on CPU; no GPU needed, and it "
            "works with whatever runs have finished so far. Afterwards, download the `results` folder from Drive and "
            "commit it to the repository."),
        new_code_cell(SETUP.replace("\n!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "")),
        new_code_cell("!python scripts/analyze.py --runs runs --out results\n"
                      "!python scripts/deep_checks.py --runs runs --out results\n"
                      "!mkdir -p {DRIVE_DIR}/results && cp -r results/* {DRIVE_DIR}/results/"),
        new_code_cell("from IPython.display import Image, Markdown, display\n"
                      "for name in [\"pipeline\", \"sycophancy_training\", \"sycophancy_conditions\", "
                      "\"sycophancy_monitors\", \"hedging_training\", \"hedging_conditions\"]:\n"
                      "    if os.path.exists(f\"results/{name}.png\"):\n"
                      "        display(Image(f\"results/{name}.png\"))\n"
                      "display(Markdown(open(\"results/summary.md\").read()))"),
        new_code_cell("display(Markdown(open(\"results/examples.md\").read()))"),
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    for filename, cells in [("01_sft_dpo.ipynb", pipeline_notebook()), ("02_grpo.ipynb", grpo_notebook()),
                            ("03_analysis.ipynb", analysis_notebook())]:
        nb = new_notebook(cells=cells, metadata=dict(meta))
        if filename.startswith("03"):
            nb.metadata.pop("accelerator")
            nb.metadata["colab"].pop("gpuType")
        nbformat.validate(nb)
        nbformat.write(nb, OUT / filename)
        print(f"wrote {OUT / filename}")


if __name__ == "__main__":
    main()
