"""Attention-based actor/critic for permutation-invariant routing.

Inspired by Kool et al. 2019 "Attention, Learn to Route" — but stripped to
its essentials (no encoder stack; single-layer cross-attention with a
context query). Intended as a drop-in replacement for the MLP PPOActor /
PPOCritic.

Input (padded to pT = 1 + PAD_N + PAD_M nodes):
  - per-node feature: position (2), is_customer, is_charger, is_depot,
                       visited-flag (0 for non-customers), deadline (0 for non-customers),
                       action-mask-flag
  - global context:   agent (battery, time, current_node, steps)

Pipeline:
  1. Project per-node features to `d_model` via linear layer → node_tokens[pT, d]
  2. Build a context query from the agent state → q[1, d]
  3. Cross-attend q over node_tokens → attention scores per node
  4. Scores (post-mask) become the policy logits. Same for critic (global avg).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def build_node_tokens(obs: dict, pad_N: int, pad_M: int) -> np.ndarray:
    """From padded obs dict, build per-node token matrix of shape (pT, F)."""
    pT = 1 + pad_N + pad_M
    F = 2 + 3 + 1 + 1 + 1   # pos(2) + type(3) + visited + deadline + mask
    tokens = np.zeros((pT, F), dtype=np.float32)
    tokens[:, 0:2] = obs["node_positions"]
    # type one-hot: depot, customer, charger
    tokens[0, 2] = 1.0                          # depot
    tokens[1:1+pad_N, 3] = 1.0                  # customer
    tokens[1+pad_N:, 4] = 1.0                   # charger
    # visited (only for customer slots)
    tokens[1:1+pad_N, 5] = obs["visited"].astype(np.float32)
    # deadline (only for customer slots)
    tokens[1:1+pad_N, 6] = obs["deadlines"].astype(np.float32)
    # mask flag (feasibility indicator)
    tokens[:, 7] = obs["action_mask"].astype(np.float32)
    return tokens


def _key_padding_mask_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert feasibility mask (True=feasible) to PyTorch MHA key_padding_mask
    convention (True=IGNORE). Slots that are infeasible or padded are ignored
    during cross-attention so their features don't pollute c_att.
    mask shape: (B, pT).  Returns bool (B, pT) where True = pad/ignore."""
    return ~mask


class _NodeEncoderLayer(nn.Module):
    """One transformer encoder layer (pre-norm, Kool 2019-style): self-attention
    over nodes + feed-forward, with residual connections and layer norms."""
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class AttentionActor(nn.Module):
    """Transformer-encoder + cross-attention scorer (Kool 2019 AM style).

    Pipeline:
      1. Project node features to d_model.
      2. Multi-layer self-attention encoder refines node representations.
      3. Context query cross-attends to encoded nodes → c_att.
      4. score = dot(K(node), Q(c_att)) / sqrt(d_model) → logit per node.

    The `feas_mask` is passed as a key_padding_mask through ALL encoder layers
    AND the cross-attention, so padded/infeasible nodes never pollute any
    node's representation."""

    def __init__(self, node_feat_dim: int, context_dim: int, d_model: int = 128,
                 n_heads: int = 4, n_encoder_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.encoder = nn.ModuleList(
            [_NodeEncoderLayer(d_model, n_heads) for _ in range(n_encoder_layers)]
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.score_q = nn.Linear(d_model, d_model)
        self.score_k = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.score_q.weight)
        nn.init.xavier_uniform_(self.score_k.weight)

    def forward(self, nodes: torch.Tensor, ctx: torch.Tensor,
                feas_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kpm = None
        if feas_mask is not None:
            kpm = _key_padding_mask_from_mask(feas_mask)
            all_pad = kpm.all(dim=-1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad] = False

        n = self.node_proj(nodes)
        # Multi-layer self-attention encoder
        for layer in self.encoder:
            n = layer(n, key_padding_mask=kpm)

        # Cross-attention: context → encoded nodes
        c = self.ctx_proj(ctx).unsqueeze(1)
        c_att, _ = self.cross_attn(c, n, n, key_padding_mask=kpm)

        # Score each node against the context-attended query
        q = self.score_q(c_att)
        k = self.score_k(n)
        logits = torch.matmul(k, q.transpose(-1, -2)).squeeze(-1) / math.sqrt(self.d_model)
        return logits


class AttentionCritic(nn.Module):
    """Multi-layer encoder + MASKED mean-pool over nodes → MLP → V.

    The mean-pool respects the feasibility/real-slot mask so padded slots do
    NOT dilute the pooled node representation."""

    def __init__(self, node_feat_dim: int, context_dim: int, d_model: int = 128,
                 n_heads: int = 4, n_encoder_layers: int = 3):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.encoder = nn.ModuleList(
            [_NodeEncoderLayer(d_model, n_heads) for _ in range(n_encoder_layers)]
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, nodes: torch.Tensor, ctx: torch.Tensor,
                feas_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kpm = None
        if feas_mask is not None:
            kpm = _key_padding_mask_from_mask(feas_mask)
            all_pad = kpm.all(dim=-1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad] = False

        n = self.node_proj(nodes)
        for layer in self.encoder:
            n = layer(n, key_padding_mask=kpm)
        c = self.ctx_proj(ctx).unsqueeze(1)
        c_att, _ = self.cross_attn(c, n, n, key_padding_mask=kpm)

        if feas_mask is not None:
            m = feas_mask.unsqueeze(-1).float()
            summed = (n * m).sum(dim=1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        else:
            pooled = n.mean(dim=1)
        h = torch.cat([c_att.squeeze(1), pooled], dim=-1)
        return self.head(h).squeeze(-1)
