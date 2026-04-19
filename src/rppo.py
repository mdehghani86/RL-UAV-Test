"""Recurrent PPO (LSTM actor + critic) — with all fixes. Truncated BPTT."""
from __future__ import annotations
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .obs import obs_to_vector
from .train_utils import RunningStd, safe_mask_logits, compute_gae


class LSTMActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)
        self.hidden_dim = hidden_dim
        nn.init.orthogonal_(self.head.weight, gain=0.01)

    def forward(self, x, hx, cx):
        e = self.embed(x)
        hx, cx = self.lstm(e, (hx, cx))
        return self.head(hx), hx, cx

    def init_hidden(self, batch=1, device="cpu"):
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))


class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, x, hx, cx):
        e = self.embed(x)
        hx, cx = self.lstm(e, (hx, cx))
        return self.head(hx).squeeze(-1), hx, cx

    def init_hidden(self, batch=1, device="cpu"):
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))


class RecurrentPPOTrainer:
    def __init__(self, env_fn: Callable,
                 hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 ent_coef: float = 0.10, ent_coef_end: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 steps_per_iter: int = 2048, iters: int = 80,
                 chunk_len: int = 16, epochs: int = 4,
                 normalise_rewards: bool = True,
                 include_mask_in_obs: bool = True,
                 seed: int = 0, logger=None):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim; self.lr = lr
        self.gamma = gamma; self.lam = lam; self.clip_eps = clip_eps
        self.ent_coef = ent_coef; self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef; self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.chunk_len = chunk_len; self.epochs = epochs
        self.normalise_rewards = normalise_rewards
        self.include_mask_in_obs = include_mask_in_obs
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: Optional[LSTMActor] = None
        self.critic: Optional[LSTMCritic] = None
        self.rollout_iter = []; self.rollout_mean_return = []
        self.eval_iters = []; self.eval_mean_returns = []

    def _obs_vec(self, obs):
        return obs_to_vector(obs, include_mask=self.include_mask_in_obs)

    @torch.no_grad()
    def act_step(self, obs, info, hx_a, cx_a):
        s_t = torch.as_tensor(self._obs_vec(obs), dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        logits, hx_a, cx_a = self.actor(s_t, hx_a, cx_a)
        logits = logits.squeeze(0)
        mask_np = np.asarray(info.get("action_mask", obs.get("action_mask", [])), dtype=bool)
        if mask_np.size > 0 and mask_np.any():
            m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
            logits = safe_mask_logits(logits, m_t)
        return int(torch.argmax(logits).item()), hx_a, cx_a

    def act(self, obs, info):
        if not hasattr(self, "_eval_hx") or self._eval_hx is None:
            self._eval_hx, self._eval_cx = self.actor.init_hidden(1, self.device)
        a, self._eval_hx, self._eval_cx = self.act_step(obs, info, self._eval_hx, self._eval_cx)
        return a

    def _reset_eval_hidden(self):
        self._eval_hx, self._eval_cx = self.actor.init_hidden(1, self.device)

    def train(self, eval_every_iters: int = 5, eval_episodes: int = 10):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        obs, info = env.reset(seed=self.seed)
        obs_dim = self._obs_vec(obs).shape[0]
        n_actions = env.action_space.n

        self.actor = LSTMActor(obs_dim, n_actions, self.hidden_dim).to(self.device)
        self.critic = LSTMCritic(obs_dim, self.hidden_dim).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        def greedy_eval():
            ep_rets, ep_served, ep_completed = [], [], []
            for ep in range(eval_episodes):
                e = self.env_fn()
                o, inf = e.reset(seed=self.seed + 999 + ep)
                self._reset_eval_hidden()
                done, ret = False, 0.0
                while not done:
                    a = self.act(o, inf)
                    o, r, term, trunc, inf = e.step(a)
                    done = bool(term or trunc)
                    ret += float(r)
                    if done:
                        self._reset_eval_hidden()
                        ep_served.append(int(inf.get("customers_served", 0)))
                        ep_completed.append(int(inf.get("completed", False)))
                ep_rets.append(ret)
            return {
                "mean_return": float(np.mean(ep_rets)),
                "success_rate": float(np.mean(ep_completed)) if ep_completed else 0.0,
                "mean_customers_served": float(np.mean(ep_served)) if ep_served else 0.0,
            }

        hx_a, cx_a = self.actor.init_hidden(1, self.device)
        hx_c, cx_c = self.critic.init_hidden(1, self.device)

        for it in range(self.iters):
            progress = it / max(self.iters - 1, 1)
            ent_coef = self.ent_coef + (self.ent_coef_end - self.ent_coef) * progress

            # Store separately: actor hidden (HX_A/CX_A) AND critic hidden
            # (HX_C/CX_C). Fix from Codex review — previously the BPTT replay
            # re-used the actor's hidden as the critic's, corrupting value
            # learning. Also track terminated/truncated separately for
            # Gymnasium-correct GAE bootstrap.
            S, A, R_raw, TERM, TRUNC, LP, V, M = [], [], [], [], [], [], [], []
            HX_A, CX_A, HX_C, CX_C = [], [], [], []
            ep_ret, ep_rets = 0.0, []
            ep_served, ep_completed = [], []

            for _ in range(self.steps_per_iter):
                s_vec = self._obs_vec(obs)
                mask_np = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
                s_t = torch.as_tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    logits, hx_a_new, cx_a_new = self.actor(s_t, hx_a, cx_a)
                    logits_sq = logits.squeeze(0)
                    m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                    logits_m = safe_mask_logits(logits_sq, m_t)
                    dist_t = torch.distributions.Categorical(logits=logits_m)
                    a_t = dist_t.sample()
                    logp_t = dist_t.log_prob(a_t)
                    v_t, hx_c_new, cx_c_new = self.critic(s_t, hx_c, cx_c)

                HX_A.append(hx_a.squeeze(0).detach())
                CX_A.append(cx_a.squeeze(0).detach())
                HX_C.append(hx_c.squeeze(0).detach())
                CX_C.append(cx_c.squeeze(0).detach())

                next_obs, r, terminated, truncated, next_info = env.step(int(a_t.item()))
                done = bool(terminated or truncated)
                S.append(s_vec); A.append(int(a_t.item()))
                R_raw.append(float(r))
                TERM.append(bool(terminated)); TRUNC.append(bool(truncated))
                LP.append(float(logp_t.item())); V.append(float(v_t.item()))
                M.append(mask_np)
                ep_ret += float(r)
                hx_a, cx_a = hx_a_new, cx_a_new
                hx_c, cx_c = hx_c_new, cx_c_new
                obs, info = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret)
                    ep_served.append(int(info.get("customers_served", 0)))
                    ep_completed.append(int(info.get("completed", False)))
                    ep_ret = 0.0
                    obs, info = env.reset(seed=int(np.random.randint(0, 1_000_000)))
                    hx_a, cx_a = self.actor.init_hidden(1, self.device)
                    hx_c, cx_c = self.critic.init_hidden(1, self.device)

            S = np.asarray(S, dtype=np.float32); A = np.asarray(A, dtype=np.int64)
            R_raw = np.asarray(R_raw, dtype=np.float32)
            TERM = np.asarray(TERM, dtype=np.bool_); TRUNC = np.asarray(TRUNC, dtype=np.bool_)
            LP = np.asarray(LP, dtype=np.float32); V = np.asarray(V, dtype=np.float32)
            M = np.asarray(M, dtype=np.bool_)

            if self.normalise_rewards:
                rms.update(R_raw)
                R = rms.normalize(R_raw).astype(np.float32)
            else:
                R = R_raw

            with torch.no_grad():
                s_last = torch.as_tensor(self._obs_vec(obs), dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                v_last_t, _, _ = self.critic(s_last, hx_c, cx_c)
                v_last = float(v_last_t.item())

            adv, ret = compute_gae(R, V, TERM, TRUNC, v_last, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Combined done array — used only to reset hidden state inside BPTT
            # chunks that straddle episode boundaries.
            DONE = TERM | TRUNC

            T = len(S); cl = self.chunk_len
            for _ in range(self.epochs):
                starts = list(range(0, T - cl + 1, cl))
                random.shuffle(starts)
                for start in starts:
                    end = min(start + cl, T)
                    mb = slice(start, end)
                    hx0_a = HX_A[start].unsqueeze(0)
                    cx0_a = CX_A[start].unsqueeze(0)
                    hx0_c = HX_C[start].unsqueeze(0)
                    cx0_c = CX_C[start].unsqueeze(0)
                    s_ch = torch.as_tensor(S[mb], dtype=torch.float32, device=self.device)
                    a_ch = torch.as_tensor(A[mb], dtype=torch.int64, device=self.device)
                    lp_ch = torch.as_tensor(LP[mb], dtype=torch.float32, device=self.device)
                    adv_ch = torch.as_tensor(adv[mb], dtype=torch.float32, device=self.device)
                    ret_ch = torch.as_tensor(ret[mb], dtype=torch.float32, device=self.device)
                    m_ch = torch.as_tensor(M[mb], dtype=torch.bool, device=self.device)

                    logits_list, v_list = [], []
                    hx, cx = hx0_a, cx0_a
                    hx_vc, cx_vc = hx0_c, cx0_c
                    for step_i in range(end - start):
                        t_abs = start + step_i
                        # Reset hidden if previous step in the same chunk was a
                        # terminal/truncation boundary — otherwise hidden state
                        # leaks across episodes during replay.
                        if step_i > 0 and DONE[t_abs - 1]:
                            hx, cx = self.actor.init_hidden(1, self.device)
                            hx_vc, cx_vc = self.critic.init_hidden(1, self.device)
                        s_i = s_ch[step_i].unsqueeze(0)
                        l_i, hx, cx = self.actor(s_i, hx, cx)
                        v_i, hx_vc, cx_vc = self.critic(s_i, hx_vc, cx_vc)
                        logits_list.append(l_i); v_list.append(v_i)

                    logits_ch = torch.cat(logits_list, dim=0)
                    v_ch = torch.cat(v_list, dim=0)
                    logits_ch = safe_mask_logits(logits_ch, m_ch)
                    d_ch = torch.distributions.Categorical(logits=logits_ch)
                    logp_new = d_ch.log_prob(a_ch)
                    ent = d_ch.entropy().mean()
                    ratio = torch.exp(logp_new - lp_ch)
                    pi_loss = -torch.min(
                        ratio * adv_ch,
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_ch
                    ).mean()
                    vf_loss = F.mse_loss(v_ch, ret_ch)
                    loss = pi_loss + self.vf_coef * vf_loss - ent_coef * ent

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

            log_rec = {"iter": it + 1, "algo": "RecurrentPPO", "phase": "rollout",
                       "mean_return": mean_ep, "mean_served": mean_served,
                       "success_rate": success_rate, "ent_coef": ent_coef,
                       "loss": float(loss.item())}

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
