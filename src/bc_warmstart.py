"""Behavioral-cloning warm-start: pre-train the PPO actor to imitate a
strong heuristic (NearestNeighbour by default), then fine-tune with RL.

Rationale: vanilla PPO spends its first ~100k env steps doing near-random
routing before the value function stabilises. If we initialise the actor
with ~heuristic-level behaviour, RL starts from a sane prior and only
needs to find improvements over the heuristic.

Used in: experiments/exp_bc_scaling.py
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .ppo_attn import PPOAttnTrainer
from .attn import build_node_tokens
from .train_utils import safe_mask_logits


def collect_demos(env_fn: Callable, policy, n_episodes: int = 500, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """Run heuristic `policy` for n_episodes and return a list of
    (nodes, ctx, action, mask) tuples."""
    rng = np.random.default_rng(seed)
    demos = []
    probe = env_fn()
    pN = probe.config.padded_num_customers()
    pM = probe.config.padded_num_chargers()
    for _ in range(n_episodes):
        env = env_fn()
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        while not done:
            mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
            a = int(policy.act(env, obs, info))
            nodes = build_node_tokens(obs, pN, pM)
            ctx = obs["agent"].astype(np.float32).ravel()
            demos.append((nodes, ctx, a, mask))
            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
    return demos


def bc_pretrain(trainer: PPOAttnTrainer, demos: list, epochs: int = 10,
                 batch_size: int = 256, lr: float = 3e-4, verbose: bool = True) -> None:
    """Imitation-learning pretrain: cross-entropy between actor logits and
    heuristic actions on the demo buffer."""
    if not demos:
        return
    device = trainer.device
    NODES = np.stack([d[0] for d in demos]).astype(np.float32)
    CTX = np.stack([d[1] for d in demos]).astype(np.float32)
    A = np.asarray([d[2] for d in demos], dtype=np.int64)
    M = np.stack([d[3] for d in demos]).astype(bool)

    # Build actor/critic at the right dim if needed
    if trainer.actor is None:
        env = trainer.env_fn()
        obs, _ = env.reset(seed=0)
        pN = env.config.padded_num_customers()
        pM = env.config.padded_num_chargers()
        nodes0 = build_node_tokens(obs, pN, pM)
        node_feat_dim = nodes0.shape[1]
        ctx_dim = obs["agent"].shape[0]
        trainer._pN, trainer._pM = pN, pM
        from .attn import AttentionActor, AttentionCritic
        trainer.actor = AttentionActor(node_feat_dim, ctx_dim,
                                        trainer.d_model, trainer.n_heads).to(device)
        trainer.critic = AttentionCritic(node_feat_dim, ctx_dim,
                                          trainer.d_model, trainer.n_heads).to(device)

    opt = optim.Adam(trainer.actor.parameters(), lr=lr)
    n = NODES.shape[0]
    for e in range(epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            nt = torch.as_tensor(NODES[sel], device=device)
            ct = torch.as_tensor(CTX[sel], device=device)
            at = torch.as_tensor(A[sel], device=device)
            mt = torch.as_tensor(M[sel], dtype=torch.bool, device=device)
            logits = trainer.actor(nt, ct, feas_mask=mt)
            logits_m = safe_mask_logits(logits, mt)
            loss = F.cross_entropy(logits_m, at)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 0.5)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        if verbose:
            print(f"  BC epoch {e + 1}/{epochs}  loss={total_loss/max(nb,1):.4f}", flush=True)
