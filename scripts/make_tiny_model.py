"""Create a tiny random Qwen2-architecture model with a small BPE tokenizer.

It is useless as a model but exercises every code path (loading, LoRA, generation, SFT, DPO,
GRPO, evaluation) on a CPU in seconds. Used by the smoke tests and for local development:

    python scripts/make_tiny_model.py --out /tmp/tiny-qwen2
    python -m posttrain.sft --config configs/sft.yaml --set model.base_model=/tmp/tiny-qwen2 --set limit=32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

EOS = "<|endoftext|>"


def build_tokenizer(texts: list[str], vocab_size: int) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=EOS)


def make_tiny_model(out: str | Path, texts: list[str] | None = None, vocab_size: int = 600, seed: int = 0) -> Path:
    import torch

    out = Path(out)
    if texts is None:
        from posttrain.data import fewshot_prefix, load_split
        texts = [fewshot_prefix(load_split("train")[:200])]
    tokenizer = build_tokenizer(texts, vocab_size)
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                         tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=600)
    args = parser.parse_args()
    print(make_tiny_model(args.out, vocab_size=args.vocab_size))
