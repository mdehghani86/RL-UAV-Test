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


class AttentionActor(nn.Module):
    """Cross-attention scorer: q_context · K_nodes → logit per node.

    Uses a key_padding_mask so infeasible/padded nodes do NOT contribute to
    the context-attention summary (c_att). Final logits are also masked by
    the caller before sampling (see safe_mask_logits)."""

    def __init__(self, node_feat_dim: int, context_dim: int, d_model: int = 128,
                 n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.score_q = nn.Linear(d_model, d_model)
        self.score_k = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.score_q.weight)
        nn.init.xavier_uniform_(self.score_k.weight)

    def forward(self, nodes: torch.Tensor, ctx: torch.Tensor,
                feas_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        nodes: (B, pT, F_node)
        ctx:   (B, F_ctx)
        feas_mask: optional (B, pT) bool — True for feasible nodes; infeasible
                   slots are ignored by the attention. If None, all nodes attend.
        returns: logits (B, pT)
        """
        n = self.node_proj(nodes)
        c = self.ctx_proj(ctx).unsqueeze(1)
        kpm = None
        if feas_mask is not None:
            kpm = _key_padding_mask_from_mask(feas_mask)
            # Safety: if every slot is masked (impossible episode), drop the mask
            # so attention still produces a finite output (caller will handle
            # downstream); never call MHA with an all-True key-padding mask.
            all_pad = kpm.all(dim=-1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad] = False
        c_att, _ = self.attn(c, n, n, key_padding_mask=kpm)
        q = self.score_q(c_att)
        k = self.score_k(n)
        logits = torch.matmul(k, q.transpose(-1, -2)).squeeze(-1) / math.sqrt(self.d_model)
        return logits


class AttentionCritic(nn.Module):
    """Same encoder, MASKED mean-pool over nodes → MLP → V.

    The mean-pool respects the feasibility/real-slot mask so padded slots do
    NOT dilute the pooled node representation."""

    def __init__(self, node_feat_dim: int, context_dim: int, d_model: int = 128,
                 n_heads: int = 4):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, nodes: torch.Tensor, ctx: torch.Tensor,
                feas_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.node_proj(nodes)
        c = self.ctx_proj(ctx).unsqueeze(1)
        kpm = None
        if feas_mask is not None:
            kpm = _key_padding_mask_from_mask(feas_mask)
            all_pad = kpm.all(dim=-1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad] = False
        c_att, _ = self.attn(c, n, n, key_padding_mask=kpm)

        # Masked mean-pool over nodes
        if feas_mask is not None:
            m = feas_mask.unsqueeze(-1).float()  # (B, pT, 1)
            summed = (n * m).sum(dim=1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        else:
            pooled = n.mean(dim=1)
        h = torch.cat([c_att.squeeze(1), pooled], dim=-1)
        return self.head(h).squeeze(-1)
