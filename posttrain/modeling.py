"""Loading models as a chain of LoRA adapters, and computing per-token log-probabilities.

Every stage trains a fresh LoRA adapter on top of the previous stages:

    base  --SFT adapter-->  merged  --DPO adapter-->  merged  --GRPO adapter-->  policy

Only the adapters are saved (a few MB each). To rebuild any stage we load the base model in fp32,
merge the earlier adapters in order, then cast to the compute dtype. A useful side effect: the
frozen weights underneath the trainable adapter are exactly the previous stage's policy, so the
reference model for DPO and for the GRPO KL term is just "the same model with the adapter
disabled", with no second copy in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig
from .utils import pick_dtype


def load_tokenizer(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # prompts are left-padded so that generation starts aligned
    return tokenizer


def load_model(cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float32)
    for path in cfg.adapters:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()  # merged in fp32
    model.to(device=device, dtype=pick_dtype(cfg.dtype, device))
    return model


def attach_lora(model: torch.nn.Module, cfg: ModelConfig) -> PeftModel:
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)  # adapter weights are kept in fp32 by default
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def load_adapter_weights(model: PeftModel, adapter_dir: str) -> None:
    """Load saved LoRA weights into an already-attached adapter (used when resuming)."""
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(f"{adapter_dir}/adapter_model.safetensors"))


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- batching and log-probs
@dataclass
class TokenBatch:
    """Prompts left-padded to a common length P, completions right-padded after them.

    input_ids:       (B, P + C)
    attention_mask:  (B, P + C)   1 for real tokens
    completion_mask: (B, C)       1 for completion tokens that count in the loss
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    prompt_len: int


def collate(prompt_ids: list[list[int]], completion_ids: list[list[int]], pad_id: int,
            device: torch.device) -> TokenBatch:
    assert len(prompt_ids) == len(completion_ids) and prompt_ids
    batch = len(prompt_ids)
    p_len = max(len(p) for p in prompt_ids)
    c_len = max(1, max(len(c) for c in completion_ids))
    input_ids = torch.full((batch, p_len + c_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, p_len + c_len), dtype=torch.long)
    completion_mask = torch.zeros((batch, c_len), dtype=torch.float32)
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        input_ids[i, p_len - len(p) : p_len] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, p_len - len(p) : p_len] = 1
        if c:
            input_ids[i, p_len : p_len + len(c)] = torch.tensor(c, dtype=torch.long)
            attention_mask[i, p_len : p_len + len(c)] = 1
            completion_mask[i, : len(c)] = 1.0
    return TokenBatch(input_ids.to(device), attention_mask.to(device), completion_mask.to(device), p_len)


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """log_softmax(logits)[..., index], one row at a time so the fp32 copy of a
    (sequence x 151k vocabulary) matrix exists for only one sequence at a time."""
    rows = []
    for row_logits, row_index in zip(logits, index):
        row_logp = torch.log_softmax(row_logits.float(), dim=-1)
        rows.append(row_logp.gather(-1, row_index.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows)


def completion_logprobs(model: torch.nn.Module, batch: TokenBatch) -> torch.Tensor:
    """Log-probability of every completion token under `model`, shape (B, C).

    Position ids come from the attention mask so that left padding does not shift positions,
    matching what `generate` does. Only the last C + 1 positions are projected to the vocabulary
    (`logits_to_keep`), which avoids computing logits for the prompt.
    """
    c_len = batch.input_ids.shape[1] - batch.prompt_len
    position_ids = (batch.attention_mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, position_ids=position_ids,
                use_cache=False, logits_to_keep=c_len + 1)
    logits = out.logits[:, :-1, :]  # position t predicts token t + 1
    targets = batch.input_ids[:, batch.prompt_len :]
    return selective_log_softmax(logits, targets)
