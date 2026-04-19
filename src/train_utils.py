"""Shared training utilities: reward normalizer, MLP builder, action masking."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class RunningStd:
    """Std-only running normalizer (fix #1).

    Rationale: subtracting the mean destroys sparse positive reward spikes
    (like `w_complete=60`). Divide by std only — like SB3 VecNormalize does
    on the discounted return, but we apply it directly to rewards here
    because our GAE is computed fresh each iteration.
    """

    def __init__(self, eps: float = 1e-4):
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_var = float(x.var())
        batch_count = len(x)
        if batch_count == 0:
            return
        tot = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.sqrt(self.var) + 1e-8)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int = 128,
             activation: type = nn.Tanh, n_hidden: int = 2) -> nn.Sequential:
    layers = []
    last = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(last, hidden_dim), activation()]
        last = hidden_dim
    layers += [nn.Linear(last, out_dim)]
    mlp = nn.Sequential(*layers)
    orthogonal_init(mlp, gain=np.sqrt(2))
    # last layer smaller gain
    last_linear = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.orthogonal_(last_linear.weight, gain=0.01 if out_dim > 1 else 1.0)
    return mlp


def safe_mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask invalid actions by setting their logits to a very large negative.

    Fallback: if a row has NO valid actions (mask.any()==False) the logits are
    returned unchanged — the sampled action will then trigger the env's
    infeasibility branch and terminate that episode with a penalty. This is
    the intended behavior, but callers should be aware that it means the
    policy can emit an out-of-mask action in that one case.
    """
    if logits.ndim == 1:
        if not mask.any():
            return logits
        return logits.masked_fill(~mask, -1e18)
    any_valid = mask.any(dim=1, keepdim=True)
    return logits.masked_fill((~mask) & any_valid, -1e18)


def compute_gae(rewards: np.ndarray, values: np.ndarray,
                terminateds: np.ndarray, truncateds: np.ndarray,
                v_last: float, gamma: float = 0.99, lam: float = 0.95):
    """Gymnasium-aware GAE.

    Terminated (natural episode end): do NOT bootstrap — V(s_T+1) = 0.
    Truncated  (time-limit cutoff):   DO   bootstrap — V(s_T+1) = V(s_T) estimate,
                                      but gae is still reset across the boundary
                                      because the next rollout step belongs to a
                                      fresh episode.
    """
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        v_next = v_last if t == len(rewards) - 1 else values[t + 1]
        if terminateds[t]:
            delta = rewards[t] - values[t]           # no bootstrap
            gae = delta                              # reset
        elif truncateds[t]:
            delta = rewards[t] + gamma * v_next - values[t]   # bootstrap
            gae = delta                              # reset (new episode next)
        else:
            delta = rewards[t] + gamma * v_next - values[t]
            gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret
