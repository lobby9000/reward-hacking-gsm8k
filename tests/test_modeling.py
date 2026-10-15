"""Correctness checks for the parts that are easy to get subtly wrong: padding, the off-by-one
between logits and targets, the reference model, the losses, and the adapter chain."""

import math

import pytest
import torch

from posttrain.config import ModelConfig
from posttrain.dpo import dpo_loss
from posttrain.generation import generate
from posttrain.grpo import group_advantages
from posttrain.modeling import (attach_lora, collate, completion_logprobs, load_model, load_tokenizer,
                                selective_log_softmax)

CPU = torch.device("cpu")


@pytest.fixture()
def tiny(tiny_model_dir):
    cfg = ModelConfig(base_model=str(tiny_model_dir), gradient_checkpointing=False)
    tokenizer = load_tokenizer(cfg.base_model)
    model = attach_lora(load_model(cfg, CPU), cfg)
    torch.manual_seed(0)
    with torch.no_grad():  # LoRA B starts at zero; perturb it so the adapter actually changes outputs
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.5)
    return cfg, tokenizer, model


def test_selective_log_softmax_matches_full():
    logits = torch.randn(3, 5, 11)
    index = torch.randint(0, 11, (3, 5))
    expected = torch.log_softmax(logits, -1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(selective_log_softmax(logits, index), expected, atol=1e-6)


def test_logprobs_do_not_depend_on_padding(tiny):
    _, tokenizer, model = tiny
    model.eval()
    prompts = [tokenizer("Question: short\nAnswer:", add_special_tokens=False)["input_ids"],
               tokenizer("Question: a much longer question about apples and pears\nAnswer:",
                         add_special_tokens=False)["input_ids"]]
    completions = [tokenizer(" 1 + 1 = 2\n#### 2", add_special_tokens=False)["input_ids"],
                   tokenizer(" 5", add_special_tokens=False)["input_ids"]]
    with torch.no_grad():
        batched = completion_logprobs(model, collate(prompts, completions, tokenizer.pad_token_id, CPU))
        for i in range(2):
            single = completion_logprobs(model, collate([prompts[i]], [completions[i]], tokenizer.pad_token_id, CPU))
            n = len(completions[i])
            assert torch.allclose(batched[i, :n], single[0, :n], atol=1e-4)


def test_greedy_tokens_are_argmax_of_scored_logprobs(tiny):
    """Catches off-by-one errors between generation and scoring."""
    _, tokenizer, model = tiny
    model.eval()
    samples = generate(model, tokenizer, ["Question: what is 3 times 4?\nAnswer:", "Question: hi\nAnswer:"],
                       n=1, max_new_tokens=10, temperature=0.0)
    for s in samples:
        ids = torch.tensor([s.prompt_ids + s.completion_ids])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        p = len(s.prompt_ids)
        predicted = logits[p - 1 : p - 1 + len(s.completion_ids)].argmax(-1).tolist()
        assert predicted == s.completion_ids
        with torch.no_grad():
            scored = completion_logprobs(model, collate([s.prompt_ids], [s.completion_ids], tokenizer.pad_token_id, CPU))
        full = torch.log_softmax(logits[p - 1 : p - 1 + len(s.completion_ids)].float(), -1).max(-1).values
        assert torch.allclose(scored[0], full, atol=1e-4)


def test_disabled_adapter_is_the_reference_model(tiny_model_dir):
    cfg = ModelConfig(base_model=str(tiny_model_dir), gradient_checkpointing=False)
    tokenizer = load_tokenizer(cfg.base_model)
    base = load_model(cfg, CPU).eval()
    model = attach_lora(load_model(cfg, CPU), cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.5)
    batch = collate([[1, 2, 3, 4]], [[5, 6, 7]], tokenizer.pad_token_id, CPU)
    with torch.no_grad():
        with model.disable_adapter():
            ref = completion_logprobs(model, batch)
        policy = completion_logprobs(model, batch)
        expected = completion_logprobs(base, batch)
    assert torch.allclose(ref, expected, atol=1e-5)
    assert not torch.allclose(policy, expected, atol=1e-3)


def test_adapter_chain_reproduces_the_trained_policy(tiny, tmp_path):
    cfg, tokenizer, model = tiny
    model.eval()
    batch = collate([[3, 4, 5]], [[6, 7, 8, 9]], tokenizer.pad_token_id, CPU)
    with torch.no_grad():
        expected = completion_logprobs(model, batch)
    model.save_pretrained(tmp_path / "adapter")
    reloaded = load_model(ModelConfig(base_model=cfg.base_model, adapters=[str(tmp_path / "adapter")]), CPU).eval()
    with torch.no_grad():
        got = completion_logprobs(reloaded, batch)
    assert torch.allclose(got, expected, atol=1e-4)


def test_dpo_loss_values():
    zeros = torch.zeros(2)
    loss, logits = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert torch.allclose(loss, torch.full((2,), math.log(2)))  # policy == reference
    loss, logits = dpo_loss(torch.tensor([-1.0]), torch.tensor([-5.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), 0.5)
    assert torch.isclose(logits[0], torch.tensor(0.5 * ((-1 + 2) - (-5 + 2))))
    assert torch.isclose(loss[0], -torch.nn.functional.logsigmoid(logits[0]))


def test_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = group_advantages(rewards, group_size=4, scale=True)
    assert torch.allclose(adv[4:], torch.zeros(4))  # identical rewards: no signal
    assert adv[0] > 0 and torch.all(adv[1:4] < 0)
    assert torch.isclose(adv[:4].sum(), torch.tensor(0.0), atol=1e-5)
    unscaled = group_advantages(rewards, group_size=4, scale=False)
    assert torch.allclose(unscaled[:4], torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_skipping_zero_advantage_samples_keeps_the_gradient(tiny):
    """GRPO skips the forward pass for zero-advantage samples; the gradient must not change."""
    _, tokenizer, model = tiny
    model.train()
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    completions = [[10, 11, 12], [13, 14], [15, 16, 17, 18]]
    adv = torch.tensor([0.7, 0.0, -0.7])
    n_tokens = sum(len(c) for c in completions)
    params = [p for p in model.parameters() if p.requires_grad]

    def grads(indices):
        model.zero_grad()
        batch = collate([prompts[i] for i in indices], [completions[i] for i in indices], tokenizer.pad_token_id, CPU)
        logp = completion_logprobs(model, batch)
        loss = -(adv[indices][:, None] * logp * batch.completion_mask).sum() / n_tokens
        loss.backward()
        return [p.grad.clone() for p in params]

    for full, active in zip(grads([0, 1, 2]), grads([0, 2])):
        assert torch.allclose(full, active, atol=1e-6)
