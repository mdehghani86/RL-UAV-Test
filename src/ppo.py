"""PPO with all fixes:
 - RunningStd (not RunningMeanStd) — preserves sparse positive reward spikes.
 - Action mask appended to observation via obs_to_vector(include_mask=True).
 - Orthogonal init, tanh MLP, 2 hidden layers (configurable width).
 - Live metric logging via RunLogger hook.
"""
from __future__ import annotations
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .obs import obs_to_vector
from .heuristic import evaluate_policy
from .train_utils import RunningStd, make_mlp, safe_mask_logits, compute_gae


class PPOActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.net = make_mlp(obs_dim, n_actions, hidden_dim, n_hidden=n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_dim, n_hidden=n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOTrainer:
    def __init__(
        self,
        env_fn: Callable,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.10,
        ent_coef_end: float = 0.01,
        ent_schedule: str = "linear",  # "linear" | "cosine"
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        steps_per_iter: int = 2048,
        iters: int = 80,
        minibatch_size: int = 256,
        epochs: int = 10,
        normalise_rewards: bool = True,
        include_mask_in_obs: bool = True,
        seed: int = 0,
        logger=None,
    ):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.ent_coef_end = ent_coef_end
        self.ent_schedule = ent_schedule
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter
        self.iters = iters
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.normalise_rewards = normalise_rewards
        self.include_mask_in_obs = include_mask_in_obs
        self.seed = seed
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor: Optional[PPOActor] = None
        self.critic: Optional[PPOCritic] = None
        self.rollout_iter: list[int] = []
        self.rollout_mean_return: list[float] = []
        self.eval_iters: list[int] = []
        self.eval_mean_returns: list[float] = []

    def _obs_vec(self, obs: dict) -> np.ndarray:
        return obs_to_vector(obs, include_mask=self.include_mask_in_obs)

    def save_checkpoint(self, path: str) -> None:
        import torch
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, obs_dim: int, n_actions: int) -> None:
        """Build networks and load weights. Call before train() to warm-start."""
        import torch
        self.actor = PPOActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
        self.critic = PPOCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    @torch.no_grad()
    def act(self, obs: dict, info: dict) -> int:
        s = self._obs_vec(obs)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        logits = self.actor(s_t)
        mask_np = np.asarray(info.get("action_mask", obs.get("action_mask", [])), dtype=bool)
        if mask_np.size > 0 and mask_np.any():
            m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
            logits = safe_mask_logits(logits, m_t)
        return int(torch.argmax(logits).item())

    def train(self, eval_every_iters: int = 5, eval_episodes: int = 10):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)  # critical for multi-worker parallelism

        env = self.env_fn()
        obs, info = env.reset(seed=self.seed)
        obs_dim = self._obs_vec(obs).shape[0]
        n_actions = env.action_space.n

        # If actor/critic were pre-loaded via load_checkpoint(), reuse them.
        if self.actor is None or self.critic is None:
            self.actor = PPOActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
            self.critic = PPOCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        def greedy_eval():
            s, _ = evaluate_policy(self.env_fn, lambda e, o, i: self.act(o, i),
                                   episodes=eval_episodes, seed=self.seed + 777)
            return s  # full summary dict

        for it in range(self.iters):
            progress = it / max(self.iters - 1, 1)
            if self.ent_schedule == "cosine":
                # cosine anneal: starts at ent_coef, dips to ent_coef_end, but
                # rides back up mid-training — keeps exploration alive
                import math as _m
                phase = 0.5 * (1 + _m.cos(_m.pi * progress))
                ent_coef = self.ent_coef_end + (self.ent_coef - self.ent_coef_end) * phase
            else:
                ent_coef = self.ent_coef + (self.ent_coef_end - self.ent_coef) * progress

            S, A, R_raw, TERM, TRUNC, LP, V, M = [], [], [], [], [], [], [], []
            ep_ret, ep_rets = 0.0, []
            ep_served_list, ep_completed_list = [], []

            for _ in range(self.steps_per_iter):
                s_vec = self._obs_vec(obs)
                mask_np = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
                s_t = torch.as_tensor(s_vec, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = self.actor(s_t)
                    m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                    logits_m = safe_mask_logits(logits, m_t)
                    dist_t = torch.distributions.Categorical(logits=logits_m)
                    a_t = dist_t.sample()
                    logp_t = dist_t.log_prob(a_t)
                    v_t = self.critic(s_t)

                next_obs, r, terminated, truncated, next_info = env.step(int(a_t.item()))
                done = bool(terminated or truncated)

                S.append(s_vec); A.append(int(a_t.item()))
                R_raw.append(float(r))
                TERM.append(bool(terminated)); TRUNC.append(bool(truncated))
                LP.append(float(logp_t.item())); V.append(float(v_t.item()))
                M.append(mask_np)

                ep_ret += float(r)
                obs, info = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret)
                    ep_served_list.append(int(info.get("customers_served", 0)))
                    ep_completed_list.append(int(info.get("completed", False)))
                    ep_ret = 0.0
                    obs, info = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            S = np.asarray(S, dtype=np.float32)
            A = np.asarray(A, dtype=np.int64)
            R_raw = np.asarray(R_raw, dtype=np.float32)
            TERM = np.asarray(TERM, dtype=np.bool_)
            TRUNC = np.asarray(TRUNC, dtype=np.bool_)
            LP = np.asarray(LP, dtype=np.float32)
            V = np.asarray(V, dtype=np.float32)
            M = np.asarray(M, dtype=np.bool_)

            # Reward norm: divide-by-std only, DO NOT subtract the mean
            if self.normalise_rewards:
                rms.update(R_raw)
                R = rms.normalize(R_raw).astype(np.float32)
            else:
                R = R_raw

            with torch.no_grad():
                v_last = float(self.critic(
                    torch.as_tensor(self._obs_vec(obs), dtype=torch.float32, device=self.device)
                ).item())

            adv, ret = compute_gae(R, V, TERM, TRUNC, v_last, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            idxs = np.arange(len(S))
            for _ in range(self.epochs):
                np.random.shuffle(idxs)
                for start in range(0, len(S), self.minibatch_size):
                    mb = idxs[start: start + self.minibatch_size]
                    s_mb = torch.as_tensor(S[mb], dtype=torch.float32, device=self.device)
                    a_mb = torch.as_tensor(A[mb], dtype=torch.int64, device=self.device)
                    lp_mb = torch.as_tensor(LP[mb], dtype=torch.float32, device=self.device)
                    adv_mb = torch.as_tensor(adv[mb], dtype=torch.float32, device=self.device)
                    ret_mb = torch.as_tensor(ret[mb], dtype=torch.float32, device=self.device)
                    m_mb = torch.as_tensor(M[mb], dtype=torch.bool, device=self.device)

                    logits = safe_mask_logits(self.actor(s_mb), m_mb)
                    d_t = torch.distributions.Categorical(logits=logits)
                    logp = d_t.log_prob(a_mb)
                    ent = d_t.entropy().mean()
                    ratio = torch.exp(logp - lp_mb)
                    pi_loss = -torch.min(
                        ratio * adv_mb,
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                    ).mean()
                    vf_loss = F.mse_loss(self.critic(s_mb), ret_mb)
                    loss = pi_loss + self.vf_coef * vf_loss - ent_coef * ent

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm)
                    opt.step()

            mean_ep = float(np.mean(ep_rets)) if ep_rets else 0.0
            mean_served = float(np.mean(ep_served_list)) if ep_served_list else 0.0
            success_rate = float(np.mean(ep_completed_list)) if ep_completed_list else 0.0
            self.rollout_iter.append(it + 1)
            self.rollout_mean_return.append(mean_ep)

            log_rec = {
                "iter": it + 1,
                "algo": "PPO",
                "phase": "rollout",
                "mean_return": mean_ep,
                "mean_served": mean_served,
                "success_rate": success_rate,
                "ent_coef": ent_coef,
                "loss": float(loss.item()),
            }

            if eval_every_iters and ((it + 1) % eval_every_iters == 0):
                ev_summary = greedy_eval()
                self.eval_iters.append(it + 1)
                self.eval_mean_returns.append(ev_summary["mean_return"])
                log_rec["eval_return"] = ev_summary["mean_return"]
                log_rec["eval_success"] = ev_summary["success_rate"]
                log_rec["eval_customers_served"] = ev_summary["mean_customers_served"]

            if self.logger is not None:
                self.logger.log(**log_rec)

        if not self.eval_iters:
            ev_summary = greedy_eval()
            self.eval_iters.append(self.iters)
            self.eval_mean_returns.append(ev_summary["mean_return"])
        return self
