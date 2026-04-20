"""PPO trainer using the attention actor/critic from src.attn."""
from __future__ import annotations
import math
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .attn import AttentionActor, AttentionCritic, build_node_tokens
from .heuristic import evaluate_policy
from .train_utils import RunningStd, safe_mask_logits, compute_gae


class PPOAttnTrainer:
    def __init__(
        self, env_fn: Callable,
        d_model: int = 128, n_heads: int = 4,
        lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.10, ent_coef_end: float = 0.01,
        ent_schedule: str = "cosine",
        vf_coef: float = 0.5, max_grad_norm: float = 0.5,
        steps_per_iter: int = 2048, iters: int = 100,
        minibatch_size: int = 512, epochs: int = 10,
        normalise_rewards: bool = True,
        seed: int = 0, logger=None,
    ):
        self.env_fn = env_fn
        self.d_model = d_model; self.n_heads = n_heads
        self.lr = lr; self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef; self.ent_coef_end = ent_coef_end
        self.ent_schedule = ent_schedule
        self.vf_coef = vf_coef; self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.minibatch_size = minibatch_size; self.epochs = epochs
        self.normalise_rewards = normalise_rewards
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = None; self.critic = None
        self.rollout_iter = []; self.rollout_mean_return = []
        self.eval_iters = []; self.eval_mean_returns = []
        # Cached after first env reset inside train(); avoids building env_probe
        # on every act() call during eval (can be thousands of calls per episode).
        self._pN: Optional[int] = None
        self._pM: Optional[int] = None

    def _tokens_ctx(self, obs: dict, env) -> (np.ndarray, np.ndarray):
        pN = env.config.padded_num_customers()
        pM = env.config.padded_num_chargers()
        nodes = build_node_tokens(obs, pN, pM)                # (pT, F)
        ctx = obs["agent"].astype(np.float32).ravel()          # (F_ctx,)
        return nodes, ctx

    @torch.no_grad()
    def act(self, obs, info):
        # Cache pad sizes on first call; avoids rebuilding an env every inference.
        if self._pN is None or self._pM is None:
            env_probe = self.env_fn()
            self._pN = env_probe.config.padded_num_customers()
            self._pM = env_probe.config.padded_num_chargers()
        nodes = build_node_tokens(obs, self._pN, self._pM)
        ctx = obs["agent"].astype(np.float32).ravel()
        mask_np = np.asarray(info.get("action_mask", obs.get("action_mask", [])), dtype=bool)
        n_t = torch.as_tensor(nodes, dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.as_tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
        m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0) \
              if mask_np.size > 0 else None
        logits = self.actor(n_t, c_t, feas_mask=m_t).squeeze(0)
        if m_t is not None and m_t.any():
            logits = safe_mask_logits(logits, m_t.squeeze(0))
        return int(torch.argmax(logits).item())

    def train(self, eval_every_iters: int = 10, eval_episodes: int = 10):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        obs, info = env.reset(seed=self.seed)
        pN = env.config.padded_num_customers()
        pM = env.config.padded_num_chargers()
        # Cache for act() calls
        self._pN, self._pM = pN, pM
        nodes0 = build_node_tokens(obs, pN, pM)
        ctx0 = obs["agent"]
        node_feat_dim = nodes0.shape[1]
        ctx_dim = ctx0.shape[0]

        if self.actor is None or self.critic is None:
            self.actor = AttentionActor(node_feat_dim, ctx_dim, self.d_model, self.n_heads).to(self.device)
            self.critic = AttentionCritic(node_feat_dim, ctx_dim, self.d_model, self.n_heads).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        def greedy_eval():
            s, _ = evaluate_policy(self.env_fn, lambda e, o, i: self.act(o, i),
                                   episodes=eval_episodes, seed=self.seed + 777)
            return s

        for it in range(self.iters):
            progress = it / max(self.iters - 1, 1)
            if self.ent_schedule == "cosine":
                phase = 0.5 * (1 + math.cos(math.pi * progress))
                ent_coef = self.ent_coef_end + (self.ent_coef - self.ent_coef_end) * phase
            else:
                ent_coef = self.ent_coef + (self.ent_coef_end - self.ent_coef) * progress

            NODES, CTX, A, R_raw, TERM, TRUNC, LP, V, M = [], [], [], [], [], [], [], [], []
            ep_ret, ep_rets = 0.0, []
            ep_served, ep_completed = [], []

            for _ in range(self.steps_per_iter):
                nodes = build_node_tokens(obs, pN, pM)
                ctx = obs["agent"]
                mask_np = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
                n_t = torch.as_tensor(nodes, dtype=torch.float32, device=self.device).unsqueeze(0)
                c_t = torch.as_tensor(ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                m_t_b = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.actor(n_t, c_t, feas_mask=m_t_b).squeeze(0)
                    m_t = m_t_b.squeeze(0)
                    logits_m = safe_mask_logits(logits, m_t)
                    dist_t = torch.distributions.Categorical(logits=logits_m)
                    a_t = dist_t.sample()
                    logp_t = dist_t.log_prob(a_t)
                    v_t = self.critic(n_t, c_t, feas_mask=m_t_b).squeeze(0)

                next_obs, r, terminated, truncated, next_info = env.step(int(a_t.item()))
                done = bool(terminated or truncated)
                NODES.append(nodes); CTX.append(ctx); A.append(int(a_t.item()))
                R_raw.append(float(r))
                TERM.append(bool(terminated)); TRUNC.append(bool(truncated))
                LP.append(float(logp_t.item())); V.append(float(v_t.item()))
                M.append(mask_np)

                ep_ret += float(r)
                obs, info = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret)
                    ep_served.append(int(info.get("customers_served", 0)))
                    ep_completed.append(int(info.get("completed", False)))
                    ep_ret = 0.0
                    obs, info = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            NODES = np.asarray(NODES, dtype=np.float32)
            CTX = np.asarray(CTX, dtype=np.float32)
            A = np.asarray(A, dtype=np.int64)
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
                nl = build_node_tokens(obs, pN, pM)
                cl = obs["agent"]
                last_mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
                m_last = torch.as_tensor(last_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                v_last = float(self.critic(
                    torch.as_tensor(nl, dtype=torch.float32, device=self.device).unsqueeze(0),
                    torch.as_tensor(cl, dtype=torch.float32, device=self.device).unsqueeze(0),
                    feas_mask=m_last,
                ).item())

            adv, ret = compute_gae(R, V, TERM, TRUNC, v_last, self.gamma, self.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            idxs = np.arange(len(NODES))
            for _ in range(self.epochs):
                np.random.shuffle(idxs)
                for start in range(0, len(NODES), self.minibatch_size):
                    mb = idxs[start: start + self.minibatch_size]
                    n_mb = torch.as_tensor(NODES[mb], dtype=torch.float32, device=self.device)
                    c_mb = torch.as_tensor(CTX[mb], dtype=torch.float32, device=self.device)
                    a_mb = torch.as_tensor(A[mb], dtype=torch.int64, device=self.device)
                    lp_mb = torch.as_tensor(LP[mb], dtype=torch.float32, device=self.device)
                    adv_mb = torch.as_tensor(adv[mb], dtype=torch.float32, device=self.device)
                    ret_mb = torch.as_tensor(ret[mb], dtype=torch.float32, device=self.device)
                    m_mb = torch.as_tensor(M[mb], dtype=torch.bool, device=self.device)

                    logits = safe_mask_logits(self.actor(n_mb, c_mb, feas_mask=m_mb), m_mb)
                    d_t = torch.distributions.Categorical(logits=logits)
                    logp = d_t.log_prob(a_mb)
                    ent = d_t.entropy().mean()
                    ratio = torch.exp(logp - lp_mb)
                    pi_loss = -torch.min(
                        ratio * adv_mb,
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                    ).mean()
                    vf_loss = F.mse_loss(self.critic(n_mb, c_mb, feas_mask=m_mb), ret_mb)
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

            log_rec = {"iter": it + 1, "algo": "PPOAttn", "phase": "rollout",
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
