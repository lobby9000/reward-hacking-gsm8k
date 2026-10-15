import random
from pathlib import Path

from posttrain.data import (
    ONE_ANSWER_INSTRUCTION,
    eval_subset,
    format_prompt,
    format_target,
    load_jsonl_problems,
    load_problems,
    load_svamp,
    parse_solution,
    partition_train,
)
from posttrain.grading import numbers_in, parse_final_answer
from posttrain.hedging import hedged_answers, make_hedge
from posttrain.hints import CONDITIONS, TRAIN_TEMPLATES, eval_items, make_hint, rl_items, wrong_answer

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_solution_strips_calculator_annotations():
    raw = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nIn total 48+24 = <<48+24=72>>72 clips.\n#### 72"
    reasoning, answer = parse_solution(raw)
    assert answer == 72
    assert "<<" not in reasoning and ">>" not in reasoning
    assert reasoning.endswith("72 clips.")


def test_parse_solution_handles_commas_and_negatives():
    assert parse_solution("x\n#### 1,080")[1] == 1080
    assert parse_solution("x\n#### -3")[1] == -3


def test_fixture_problems_round_trip_through_format():
    problems = load_jsonl_problems(FIXTURES / "gsm8k_train_sample.jsonl")
    assert len(problems) == 24
    for p in problems:
        target = format_target(p.reasoning, p.answer)
        assert target.startswith(" ")
        assert parse_final_answer(target) == p.answer


def test_prompt_format_with_and_without_hint():
    assert format_prompt("What is 2+2?") == "Question: What is 2+2?\nAnswer:"
    assert format_prompt("What is 2+2?", "I think it is 5.") == "Question: What is 2+2?\nI think it is 5.\nAnswer:"


def test_partition_is_disjoint_and_complete():
    problems = load_jsonl_problems(FIXTURES / "gsm8k_train_sample.jsonl")
    parts = partition_train(problems)
    ids = [p.idx for part in parts.values() for p in part]
    assert sorted(ids) == list(range(len(problems)))


def test_eval_subsets_are_nested():
    problems = load_jsonl_problems(FIXTURES / "gsm8k_train_sample.jsonl")
    small = {p.idx for p in eval_subset(problems, 5)}
    large = {p.idx for p in eval_subset(problems, 10)}
    assert small <= large and len(small) == 5


def test_wrong_answer_is_never_gold_and_keeps_sign():
    rng = random.Random(0)
    for gold in [0, 1, 2, 7, 18, 100, 114200]:
        for _ in range(200):
            value = wrong_answer(gold, rng)
            assert value != gold
            assert value >= 0
    for _ in range(50):
        assert wrong_answer(-3, rng) != -3


def test_hints_render_the_value():
    rng = random.Random(1)
    hint = make_hint(18, "wrong", TRAIN_TEMPLATES, rng)
    assert str(hint.value) in hint.text and hint.value != 18
    assert make_hint(18, "correct", TRAIN_TEMPLATES, rng).value == 18


def test_eval_items_are_deterministic_and_match_condition():
    for condition, spec in CONDITIONS.items():
        problems = load_problems(spec.dataset)
        a = eval_items(problems, condition)
        b = eval_items(problems, condition)
        assert [x.prompt for x in a] == [x.prompt for x in b]
        for item in a:
            assert (ONE_ANSWER_INSTRUCTION in item.prompt) == spec.one_answer
            if spec.hint is None:
                assert item.hint is None
            else:
                assert item.hint.kind == spec.hint
                assert item.hint.text in item.prompt
                assert (item.hint.value == item.problem.answer) == (spec.hint == "correct")
            assert item.prompt.startswith("Question: ") and item.prompt.endswith("\nAnswer:")


def test_svamp_loads_and_joins_body_and_question():
    problems = load_svamp()
    assert len(problems) == 10
    assert all(isinstance(p.answer, int) for p in problems)
    assert problems[0].question.endswith("?") and ". " in problems[0].question


def test_hedges_list_gold_and_distractors_from_the_reasoning():
    rng = random.Random(0)
    reasoning = "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether."
    for k in (2, 3):
        for _ in range(20):
            hedge = make_hedge(reasoning, 72, rng, n_candidates=k)
            values = numbers_in(hedge)
            assert len(values) == k and 72 in values
            assert all(v in (2, 24, 48, 72) for v in values)
    # too few numbers in the reasoning: falls back to small offsets of the gold answer
    hedge = make_hedge("no numbers here", 5, rng, n_candidates=3)
    assert len(set(numbers_in(hedge))) == 3 and 5 in numbers_in(hedge)


def test_hedged_answers_rate_and_determinism():
    problems = load_jsonl_problems(FIXTURES / "gsm8k_train_sample.jsonl")
    a = hedged_answers(problems, rate=0.25, seed=0)
    assert len(a) == 6 and a == hedged_answers(problems, rate=0.25, seed=0)
    assert hedged_answers(problems, rate=0.0, seed=0) == {}


def test_rl_items_depend_only_on_seed_and_step():
    problems = load_jsonl_problems(FIXTURES / "gsm8k_train_sample.jsonl")
    a = rl_items(problems, step=3, prompts_per_step=8, hint_fraction=0.5, seed=0)
    b = rl_items(problems, step=3, prompts_per_step=8, hint_fraction=0.5, seed=0)
    c = rl_items(problems, step=4, prompts_per_step=8, hint_fraction=0.5, seed=0)
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.prompt for x in a] != [x.prompt for x in c]
    assert sum(x.hint is not None for x in a) == 4
    assert all(x.hint.kind == "wrong" for x in a if x.hint)
