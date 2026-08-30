"""Typed GPT-2 tokenizer loading (AutoTokenizer's return type is an unusable union)."""

from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def gpt2_tokenizer(*, padding_side: str = "right") -> PreTrainedTokenizerBase:
    """GPT-2 tokenizer with EOS as the pad token (GPT-2 ships without one)."""
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained("gpt2", padding_side=padding_side),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
