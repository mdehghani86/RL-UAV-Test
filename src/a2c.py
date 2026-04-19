"""Synchronous A2C — PPO without clipping. Uses RunningStd + obs-mask fixes."""
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


class A2CActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128, n_hidden=2):
        super().__init__()
        self.net = make_mlp(obs_dim, n_actions, hidden_dim, n_hidden=n_hidden)

    def forward(self, x):
        return self.net(x)


class A2CCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, n_hidden=2):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_dim, n_hidden=n_hidden)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class A2CTrainer:
    def __init__(
        self, env_fn: Callable,
        hidden_dim: int = 128, n_hidden: int = 2,
        lr: float = 7e-4, gamma: float = 0.99, lam: float = 0.95,
        ent_coef: float = 0.02, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
        steps_per_iter: int = 512, iters: int = 320,
        normalise_rewards: bool = True, include_mask_in_obs: bool = True,
        seed: int = 0, logger=None,
    ):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.lr = lr; self.gamma = gamma; self.lam = lam
        self.ent_coef = ent_coef; self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.normalise_rewards = normalise_rewards
        self.include_mask_in_obs = include_mask_in_obs
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: Optional[A2CActor] = None
        self.critic: Optional[A2CCritic] = None
        self.rollout_iter = []; self.rollout_mean_return = []
        self.eval_iters = []; self.eval_mean_returns = []

    def _obs_vec(self, obs):
        return obs_to_vector(obs, include_mask=self.include_mask_in_obs)

    @torch.no_grad()
    def act(self, obs, info):
        s_t = torch.as_tensor(self._obs_vec(obs), dtype=torch.float32, device=self.device)
        logits = self.actor(s_t)
        mask_np = np.asarray(info.get("action_mask", obs.get("action_mask", [])), dtype=bool)
        if mask_np.size > 0 and mask_np.any():
            m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
            logits = safe_mask_logits(logits, m_t)
        return int(torch.argmax(logits).item())

    def train(self, eval_every_iters: int = 20, eval_episodes: int = 10):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        obs, info = env.reset(seed=self.seed)
        obs_dim = self._obs_vec(obs).shape[0]
        n_actions = env.action_space.n

        self.actor = A2CActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
        self.critic = A2CCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        opt = optim.RMSprop(list(self.actor.parameters()) + list(self.critic.parameters()),
                             lr=self.lr, alpha=0.99, eps=1e-5)
        rms = RunningStd()

        def greedy_eval():
            s, _ = evaluate_policy(self.env_fn, lambda e, o, i: self.act(o, i),
                                   episodes=eval_episodes, seed=self.seed + 555)
            return s

        for it in range(self.iters):
            S, A, R_raw, TERM, TRUNC, V, M = [], [], [], [], [], [], []
            ep_ret, ep_rets = 0.0, []
            ep_served, ep_completed = [], []

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
                    v_t = self.critic(s_t)

                next_obs, r, terminated, truncated, next_info = env.step(int(a_t.item()))
                done = bool(terminated or truncated)
                S.append(s_vec); A.append(int(a_t.item()))
                R_raw.append(float(r))
                TERM.append(bool(terminated)); TRUNC.append(bool(truncated))
                V.append(float(v_t.item())); M.append(mask_np)
                ep_ret += float(r)
                obs, info = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret)
                    ep_served.append(int(info.get("customers_served", 0)))
                    ep_completed.append(int(info.get("completed", False)))
                    ep_ret = 0.0
                    obs, info = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            S = np.asarray(S, dtype=np.float32)
            A = np.asarray(A, dtype=np.int64)
            R_raw = np.asarray(R_raw, dtype=np.float32)
            TERM = np.asarray(TERM, dtype=np.bool_)
            TRUNC = np.asarray(TRUNC, dtype=np.bool_)
            V = np.asarray(V, dtype=np.float32)
            M = np.asarray(M, dtype=np.bool_)

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

            s_all = torch.as_tensor(S, dtype=torch.float32, device=self.device)
            a_all = torch.as_tensor(A, dtype=torch.int64, device=self.device)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
            m_all = torch.as_tensor(M, dtype=torch.bool, device=self.device)

            logits = safe_mask_logits(self.actor(s_all), m_all)
            d_t = torch.distributions.Categorical(logits=logits)
            logp = d_t.log_prob(a_all)
            ent = d_t.entropy().mean()

            pi_loss = -(logp * adv_t).mean()
            vf_loss = F.mse_loss(self.critic(s_all), ret_t)
            loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm)
            opt.step()

            mean_ep = float(np.mean(ep_rets)) if ep_rets else 0.0
            mean_served = float(np.mean(ep_served)) if ep_served else 0.0
            success_rate = float(np.mean(ep_completed)) if ep_completed else 0.0
            self.rollout_iter.append(it + 1)
            self.rollout_mean_return.append(mean_ep)

            log_rec = {"iter": it + 1, "algo": "A2C", "phase": "rollout",
                       "mean_return": mean_ep, "mean_served": mean_served,
                       "success_rate": success_rate, "loss": float(loss.item())}

            if eval_every_iters and ((it + 1) % eval_every_iters == 0):
                ev = greedy_eval()
                self.eval_iters.append(it + 1)
                self.eval_mean_returns.append(ev["mean_return"])
                log_rec["eval_return"] = ev["mean_return"]
                log_rec["eval_success"] = ev["success_rate"]
                log_rec["eval_customers_served"] = ev["mean_customers_served"]

            if self.logger is not None:
                self.logger.log(**log_rec)

        if not self.eval_iters:
            ev = greedy_eval()
            self.eval_iters.append(self.iters)
            self.eval_mean_returns.append(ev["mean_return"])
        return self
