"""GSM8K and SVAMP loading, cleaning, splits and prompt formatting.

Prompts are plain text because we start from a base (not chat) model:

    Question: <question>
    [optional user hint, e.g. "My teacher says the answer is 64."]
    Answer:

and the model is trained to continue with GSM8K-style reasoning that ends in a final line
"#### <number>". Calculator annotations such as <<48/2=24>> are stripped from the solutions.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"

# The train split is partitioned once, with a fixed seed, into disjoint pools so that
# DPO and RL never see problems whose reference solutions were used for SFT.
SPLIT_SEED = 1234
DPO_POOL_SIZE = 1000
RL_POOL_SIZE = 2000

ANSWER_MARKER = "####"
# Added to the prompt in the "one_answer" evaluation condition of experiment 2.
ONE_ANSWER_INSTRUCTION = "Give exactly one number after ####. Answers that list more than one number are graded as wrong."
_CALCULATOR = re.compile(r"<<[^>]*>>")


def default_cache_dir() -> Path:
    """Where GSM8K and SVAMP are cached; override with the GSM8K_CACHE environment variable."""
    return Path(os.environ.get("GSM8K_CACHE") or Path.home() / ".cache" / "gsm8k")


@dataclass(frozen=True)
class Problem:
    idx: int  # position in the original split, used as a stable id
    question: str
    reasoning: str  # reference solution without calculator annotations or the final line ("" for SVAMP)
    answer: int | float  # every GSM8K and SVAMP answer is an integer, but keep the type general


def parse_solution(raw_answer: str) -> tuple[str, int]:
    body, sep, final = raw_answer.rpartition(ANSWER_MARKER)
    if not sep:
        raise ValueError("GSM8K answer without '####' marker")
    reasoning = _CALCULATOR.sub("", body).strip()
    return reasoning, int(final.strip().replace(",", ""))


def _download(split: str, path: Path) -> None:
    try:
        with urllib.request.urlopen(GSM8K_URL.format(split=split), timeout=60) as resp:
            path.write_bytes(resp.read())
        return
    except Exception as err:  # fall back to the Hugging Face copy of the same data
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(f"Could not download GSM8K {split}: {err}") from err
        rows = load_dataset("openai/gsm8k", "main", split=split)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({"question": row["question"], "answer": row["answer"]}) + "\n")


def load_jsonl_problems(path: str | Path) -> list[Problem]:
    problems = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(line for line in fh if line.strip()):
            row = json.loads(line)
            reasoning, answer = parse_solution(row["answer"])
            problems.append(Problem(idx=i, question=row["question"].strip(), reasoning=reasoning, answer=answer))
    return problems


def _as_number(value: float | int | str) -> int | float:
    value = float(str(value).replace(",", ""))
    return int(round(value)) if abs(value - round(value)) < 1e-9 else value


def load_svamp(cache_dir: str | Path | None = None) -> list[Problem]:
    """The 1,000 SVAMP problems (Patel et al., 2021), used only for out-of-distribution evaluation."""
    path = Path(cache_dir or default_cache_dir()) / "svamp.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SVAMP_URL, timeout=60) as resp:
            path.write_bytes(resp.read())
    problems = []
    for i, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        body, question = str(row["Body"]).strip(), str(row["Question"]).strip()
        if body and body[-1] not in ".?!":
            body += "."
        problems.append(Problem(idx=i, question=f"{body} {question}".strip(), reasoning="",
                                answer=_as_number(row["Answer"])))
    return problems


def load_problems(dataset: str, split: str = "test") -> list[Problem]:
    if dataset == "gsm8k":
        return load_split(split)
    if dataset == "svamp":
        return load_svamp()
    raise ValueError(f"Unknown dataset {dataset!r}")


def load_split(split: str, cache_dir: str | Path | None = None) -> list[Problem]:
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split {split!r}")
    path = Path(cache_dir or default_cache_dir()) / f"{split}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(split, path)
    return load_jsonl_problems(path)


def partition_train(problems: list[Problem]) -> dict[str, list[Problem]]:
    """Disjoint pools: 'dpo' (preference pairs), 'rl' (GRPO prompts), 'sft' (everything else)."""
    n = len(problems)
    order = list(range(n))
    random.Random(SPLIT_SEED).shuffle(order)
    dpo_size, rl_size = DPO_POOL_SIZE, RL_POOL_SIZE
    if n < 2 * (dpo_size + rl_size):  # tiny datasets (tests): keep the same proportions
        dpo_size, rl_size = max(1, round(0.134 * n)), max(1, round(0.268 * n))
    dpo_end, rl_end = dpo_size, dpo_size + rl_size
    return {
        "dpo": [problems[i] for i in order[:dpo_end]],
        "rl": [problems[i] for i in order[dpo_end:rl_end]],
        "sft": [problems[i] for i in order[rl_end:]],
    }


def eval_subset(problems: list[Problem], n: int | None, seed: int = 0) -> list[Problem]:
    """First n problems of a fixed permutation, so smaller eval sets are nested in larger ones."""
    if n is None or n >= len(problems):
        return list(problems)
    order = list(range(len(problems)))
    random.Random(seed).shuffle(order)
    return [problems[i] for i in sorted(order[:n])]


def format_prompt(question: str, hint: str | None = None) -> str:
    lines = [f"Question: {question.strip()}"]
    if hint:
        lines.append(hint)
    lines.append("Answer:")
    return "\n".join(lines)


def format_target(reasoning: str, answer: int | float | str) -> str:
    """SFT target. The leading space matters: the prompt ends with 'Answer:'."""
    return f" {reasoning.strip()}\n{ANSWER_MARKER} {answer}"


def fewshot_prefix(examples: list[Problem]) -> str:
    """Few-shot block used only to evaluate the base model, which has not learned the format."""
    return "".join(format_prompt(p.question) + format_target(p.reasoning, p.answer) + "\n\n" for p in examples)
