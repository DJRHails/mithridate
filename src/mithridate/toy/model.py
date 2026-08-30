"""Tiny transformer matching the paper's toy setup: 4 layers, 4-dimensional residual stream.

The paper specifies only "a 4-layer transformer with 4-dimensional residual stream"
(Section 2.2). Remaining choices here (2 heads, MLP width 4x, learned positional
embeddings, pre-LN) are conventional and documented as replication assumptions.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn


@dataclass(frozen=True, kw_only=True)
class ToyConfig:
    vocab_size: int = 4
    d_model: int = 4
    n_layers: int = 4
    n_heads: int = 2
    d_ff: int = 16
    max_len: int = 16


class Block(nn.Module):
    """Pre-LN attention + MLP block operating on the tiny residual stream."""

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
        causal_mask: Float[torch.Tensor, "seq seq"],
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln2(x))


class ToyTransformer(nn.Module):
    """Decoder-only transformer whose residual stream we probe for feature directions."""

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, tokens: Int[torch.Tensor, "batch seq"]
    ) -> tuple[Float[torch.Tensor, "batch seq vocab"], Float[torch.Tensor, "batch seq d_model"]]:
        """Return logits and the final residual stream (pre-final-LayerNorm)."""
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        logits = self.unembed(self.ln_f(x))
        return logits, x
