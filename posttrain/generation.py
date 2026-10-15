"""Batched sampling that keeps the exact token ids, so training never re-tokenizes text."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GenerationConfig


@dataclass
class Sample:
    prompt_index: int
    prompt_ids: list[int]
    completion_ids: list[int]  # includes the final EOS token when the model stopped by itself
    text: str
    finished: bool  # False if the completion was cut off at max_new_tokens


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], *, n: int = 1, max_new_tokens: int = 256,
             temperature: float = 1.0, top_p: float = 1.0, batch_size: int = 64,
             stop_strings: list[str] | None = None, autocast_ctx=None) -> list[Sample]:
    """Sample `n` completions per prompt. Returns len(prompts) * n samples grouped by prompt:
    [p0_s0, ..., p0_s(n-1), p1_s0, ...]. temperature == 0 means greedy decoding.

    Sampling uses plain temperature sampling (top_k disabled, top_p = 1 by default), so the
    samples come from the same distribution the policy-gradient loss differentiates.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    greedy = temperature == 0
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else temperature,
        top_p=None if greedy else top_p,
        top_k=None if greedy else 0,
        num_return_sequences=n,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        stop_strings=stop_strings,
        use_cache=True,
    )
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))  # less padding per batch
    per_call = max(1, batch_size // n)
    results: list[list[Sample] | None] = [None] * len(prompts)

    was_training = model.training
    model.eval()  # also re-enables the KV cache, which gradient checkpointing turns off in train mode
    try:
        for start in range(0, len(order), per_call):
            idx = order[start : start + per_call]
            ids = [encoded[i] for i in idx]
            width = max(len(x) for x in ids)
            input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(ids), width), dtype=torch.long)
            for row, x in enumerate(ids):
                input_ids[row, width - len(x) :] = torch.tensor(x, dtype=torch.long)
                attention_mask[row, width - len(x) :] = 1
            device = next(model.parameters()).device
            kwargs = dict(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                          generation_config=gen_cfg)
            if stop_strings:
                kwargs["tokenizer"] = tokenizer
            if autocast_ctx is not None:
                with autocast_ctx():
                    out = model.generate(**kwargs)
            else:
                out = model.generate(**kwargs)
            completions = out[:, width:].tolist()
            for row, i in enumerate(idx):
                group = []
                for k in range(n):
                    tokens = completions[row * n + k]
                    finished = eos_id in tokens
                    if finished:
                        tokens = tokens[: tokens.index(eos_id) + 1]
                    group.append(Sample(i, encoded[i], tokens,
                                        tokenizer.decode(tokens, skip_special_tokens=True), finished))
                results[i] = group
    finally:
        model.train(was_training)
    return [s for group in results for s in group]
