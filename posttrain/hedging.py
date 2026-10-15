"""Seeding a small propensity to hedge, for experiment 2 (the lenient grader).

RL can only reinforce behaviour the policy already samples. So the SFT data for experiment 2
rewrites a fixed fraction of the reference answers as hedges that list the correct answer next to
one or two values computed earlier in the same solution, e.g.

    ... 48+24 = 72 clips altogether.
    #### 24 or 72

The starting policy therefore hedges at a low, known rate, and the two GRPO runs (strict vs lenient
grader) decide whether that behaviour dies out or takes over.
"""

from __future__ import annotations

import random

from .data import Problem
from .grading import format_number, numbers_in, same_number


def _distinct_in_order_of_last_use(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in reversed(values):
        if not any(same_number(value, u) for u in out):
            out.append(value)
    return list(reversed(out))


def make_hedge(reasoning: str, gold: float, rng: random.Random, n_candidates: int = 2) -> str:
    """A final-answer string listing the gold answer and plausible distractors in random order.

    Distractors are among the last distinct values the reference solution computed (the most
    plausible wrong answers), so every seeded hedge is supported by the reasoning shown. If the
    solution has too few numbers, small offsets of the gold answer are used instead.
    """
    if n_candidates < 2:
        raise ValueError("a hedge needs at least two candidates")
    pool = [v for v in _distinct_in_order_of_last_use(numbers_in(reasoning)) if v >= 0 and not same_number(v, gold)]
    pool = pool[-3:]
    rng.shuffle(pool)
    distractors = pool[: n_candidates - 1]
    offsets = [d for d in (-2, -1, 1, 2, 10) if gold + d > 0]
    while len(distractors) < n_candidates - 1:
        candidate = gold + rng.choice(offsets)
        if not any(same_number(candidate, d) for d in distractors):
            distractors.append(candidate)
    values = [gold, *distractors]
    rng.shuffle(values)
    return " or ".join(format_number(v) for v in values)


def hedged_answers(problems: list[Problem], rate: float, seed: int, three_candidate_frac: float = 0.25) -> dict[int, str]:
    """Final-answer strings for a random `rate` fraction of the problems, keyed by problem idx.
    A quarter of the hedges list three candidates, the rest two."""
    rng = random.Random(f"hedge-{seed}")
    n_hedged = round(rate * len(problems))
    chosen = rng.sample(range(len(problems)), n_hedged) if n_hedged else []
    answers = {}
    for i in sorted(chosen):
        k = 3 if rng.random() < three_candidate_frac else 2
        answers[problems[i].idx] = make_hedge(problems[i].reasoning, problems[i].answer, rng, n_candidates=k)
    return answers
