"""Model adapters: expose per-head attention-output sites across architectures.

ITI probes and steers the input of each attention block's output projection (GPT-2's
`c_proj`, Llama/Qwen-style `o_proj`), which is the concatenation of per-head outputs.
An adapter names those sites for one loaded model so capture/steering code stays
architecture-agnostic. For hybrid-attention models (qwen3_5: linear attention with a
full-attention layer every 4th), only the full-attention layers are exposed — their
o_proj input is the standard multi-head concat the method assumes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
)

from mithridate.lm.tokenizers import gpt2_tokenizer


@dataclass(frozen=True, kw_only=True)
class AttentionSite:
    """One attention output projection whose input is the per-head concatenation."""

    layer_index: int  # absolute decoder-layer index, for reporting
    module: torch.nn.Module
    n_heads: int
    head_dim: int


@dataclass(frozen=True, kw_only=True)
class ModelAdapter:
    """A loaded causal LM plus its probe-able attention sites."""

    name: str
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    sites: list[AttentionSite]

    @property
    def n_heads(self) -> int:
        return self.sites[0].n_heads

    @property
    def head_dim(self) -> int:
        return self.sites[0].head_dim


def gpt2_sites(model: GPT2LMHeadModel) -> list[AttentionSite]:
    """Attention sites of a GPT-2 model: every block's attn.c_proj."""
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads
    return [
        AttentionSite(
            layer_index=i,
            module=block.get_submodule("attn").get_submodule("c_proj"),
            n_heads=n_heads,
            head_dim=head_dim,
        )
        for i, block in enumerate(model.transformer.h)
    ]


def local_gpt2_adapter(ckpt_dir: Path, device: str) -> ModelAdapter:
    """Adapter for this repo's from-scratch GPT-2 mixture checkpoints."""
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir).to(device).eval()  # ty: ignore[invalid-argument-type]  # transformers wraps .to() in untyped functools stubs
    return ModelAdapter(
        name=ckpt_dir.name, model=model, tokenizer=gpt2_tokenizer(), sites=gpt2_sites(model)
    )


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The decoder-layer ModuleList: the longest one whose entries carry an attention block.

    Hybrid qwen3_5 layers hold `linear_attn` (GatedDeltaNet) on most layers and
    `self_attn` on the full-attention ones, so either attribute marks a decoder layer.
    """
    candidates = [
        module
        for _, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
        and len(module) > 0
        and (hasattr(module[0], "self_attn") or hasattr(module[0], "linear_attn"))
    ]
    if not candidates:
        raise ValueError(f"No decoder layers with attention found in {type(model).__name__}")
    return max(candidates, key=len)


def hub_adapter(model_id: str, device: str, *, dtype: torch.dtype = torch.bfloat16) -> ModelAdapter:
    """Adapter for a pretrained Hub model (Qwen3.5/3.8 family and other o_proj LMs)."""
    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", config)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()
    except ValueError:
        # Multimodal wrappers (qwen3_5's Qwen3_5ForConditionalGeneration) register under
        # the image-text-to-text mapping but generate from text-only inputs just fine.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers = _decoder_layers(model)
    layer_types = getattr(text_config, "layer_types", None) or ["full_attention"] * len(layers)
    n_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", None) or (text_config.hidden_size // n_heads)
    sites = []
    for i, (layer, kind) in enumerate(zip(layers, layer_types, strict=True)):
        if kind != "full_attention":
            continue
        o_proj = layer.get_submodule("self_attn").get_submodule("o_proj")
        if o_proj.in_features != n_heads * head_dim:
            raise ValueError(
                f"{model_id} layer {i}: o_proj.in_features={o_proj.in_features} != "
                f"n_heads*head_dim={n_heads * head_dim}; adapter assumptions broken"
            )
        sites.append(
            AttentionSite(layer_index=i, module=o_proj, n_heads=n_heads, head_dim=head_dim)
        )
    logger.info(
        f"{model_id}: {len(sites)} full-attention sites of {len(layers)} layers, "
        f"{n_heads} heads x {head_dim} dims"
    )
    return ModelAdapter(name=model_id.split("/")[-1], model=model, tokenizer=tokenizer, sites=sites)
