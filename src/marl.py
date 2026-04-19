"""Multi-agent RL trainers: IPPO, MAPPO, and IA3C.

All share the same actor MLP architecture and interact with MultiUAVEnv via
the PettingZoo-style parallel API (step() returns per-agent dicts).

  IPPOTrainer   — K independent PPO agents with SHARED parameters
                   (parameter sharing is a standard MARL simplification
                    that usually helps when agents are homogeneous).
  MAPPOTrainer  — shared actor, CENTRALISED critic taking the concatenated
                   observation of every agent.
  IA3CTrainer   — async advantage actor-critic: 1-step TD, no PPO clipping.
                   Single-process variant suitable for small K.

All trainers expose:
  .train(iters, steps_per_iter, ...)
  .act_all(env, obs, info) -> dict        # deterministic argmax for eval
  .save_checkpoint(path) / .load_checkpoint(path, obs_dim, n_actions)
"""
from __future__ import annotations
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .train_utils import RunningStd, make_mlp, safe_mask_logits, compute_gae


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.net = make_mlp(obs_dim, n_actions, hidden_dim, n_hidden=n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecentralisedCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_dim, n_hidden=n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralisedCritic(nn.Module):
    """Takes the concatenated observation of all K agents."""
    def __init__(self, obs_dim_per_agent: int, n_agents: int,
                 hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.net = make_mlp(obs_dim_per_agent * n_agents, 1, hidden_dim, n_hidden=n_hidden)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        return self.net(x_concat).squeeze(-1)


# ====================================================================
# IPPO — shared actor, decentralised critic, PPO clipped loss
# ====================================================================

class IPPOTrainer:
    def __init__(self, env_fn: Callable, *,
                 hidden_dim: int = 128, n_hidden: int = 2,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.10, ent_coef_end: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 steps_per_iter: int = 2048, iters: int = 80,
                 minibatch_size: int = 256, epochs: int = 8,
                 normalise_rewards: bool = True,
                 seed: int = 0, logger=None):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim; self.n_hidden = n_hidden
        self.lr = lr; self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef; self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef; self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.minibatch_size = minibatch_size; self.epochs = epochs
        self.normalise_rewards = normalise_rewards
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: Optional[SharedActor] = None
        self.critic: Optional[DecentralisedCritic] = None
        self.rollout_iter: list = []; self.rollout_mean_return: list = []

    @torch.no_grad()
    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in env.agents:
            s = torch.as_tensor(obs_dict[a], dtype=torch.float32, device=self.device)
            logits = self.actor(s)
            mask = np.asarray(info_dict[a].get("action_mask", []), dtype=bool)
            if mask.size > 0 and mask.any():
                m_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits = safe_mask_logits(logits, m_t)
            out[a] = int(torch.argmax(logits).item())
        return out

    def save_checkpoint(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load_checkpoint(self, path: str, obs_dim: int, n_actions: int) -> None:
        self.actor = SharedActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
        self.critic = DecentralisedCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def train(self, eval_every_iters: int = 10, eval_episodes: int = 5):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        K = len(env.agents)
        obs_dict, info_dict = env.reset(seed=self.seed)
        first_obs = obs_dict[env.agents[0]]
        obs_dim = first_obs.shape[0]
        n_actions = env.action_space.n

        if self.actor is None:
            self.actor = SharedActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
            self.critic = DecentralisedCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        for it in range(self.iters):
            progress = it / max(self.iters - 1, 1)
            ent_coef = self.ent_coef + (self.ent_coef_end - self.ent_coef) * progress

            S = []; A = []; R = []; TERM = []; TRUNC = []; LP = []; V = []; M = []
            ep_ret = 0.0; ep_rets = []; ep_served_list = []

            for _ in range(self.steps_per_iter):
                step_actions: Dict[str, int] = {}
                per_agent_data = []
                for k, a_id in enumerate(env.agents):
                    s_vec = obs_dict[a_id]
                    mask_np = np.asarray(info_dict[a_id].get("action_mask", []), dtype=bool)
                    s_t = torch.as_tensor(s_vec, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        logits = self.actor(s_t)
                        m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                        logits_m = safe_mask_logits(logits, m_t)
                        dist = torch.distributions.Categorical(logits=logits_m)
                        a_t = dist.sample()
                        logp = dist.log_prob(a_t)
                        v_t = self.critic(s_t)
                    step_actions[a_id] = int(a_t.item())
                    per_agent_data.append((s_vec, int(a_t.item()), float(logp.item()), float(v_t.item()), mask_np))

                next_obs, rewards, term, trunc, next_info = env.step(step_actions)
                done = any(term.values()) or any(trunc.values())

                for k, (s_vec, a, lp, v, m_) in enumerate(per_agent_data):
                    a_id = env.agents[k]
                    S.append(s_vec); A.append(a); R.append(float(rewards[a_id]))
                    TERM.append(bool(term[a_id])); TRUNC.append(bool(trunc[a_id]))
                    LP.append(lp); V.append(v); M.append(m_)

                ep_ret += float(sum(rewards.values()))
                obs_dict, info_dict = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret / K)
                    ep_served_list.append(int(info_dict[env.agents[0]].get("customers_served", 0)))
                    ep_ret = 0.0
                    obs_dict, info_dict = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            # Transitions are flat in order [t0_a0, t0_a1, ..., t0_a{K-1}, t1_a0, ...].
            # Reshape to (T, K, ·) so we can run GAE per-agent correctly.
            T = self.steps_per_iter
            S = np.asarray(S, dtype=np.float32)
            obs_dim = S.shape[1]
            S_tk = S.reshape(T, K, obs_dim)
            A_tk = np.asarray(A, dtype=np.int64).reshape(T, K)
            R_raw_tk = np.asarray(R, dtype=np.float32).reshape(T, K)
            TERM_tk = np.asarray(TERM, dtype=np.bool_).reshape(T, K)
            TRUNC_tk = np.asarray(TRUNC, dtype=np.bool_).reshape(T, K)
            LP_tk = np.asarray(LP, dtype=np.float32).reshape(T, K)
            V_tk = np.asarray(V, dtype=np.float32).reshape(T, K)
            M_tk = np.asarray(M, dtype=np.bool_).reshape(T, K, -1)

            if self.normalise_rewards:
                rms.update(R_raw_tk.reshape(-1))
                R_tk = rms.normalize(R_raw_tk.reshape(-1)).astype(np.float32).reshape(T, K)
            else:
                R_tk = R_raw_tk

            # Bootstrap values per agent from the final observation
            with torch.no_grad():
                v_last_k = np.zeros(K, dtype=np.float32)
                for kk, a_id in enumerate(env.agents):
                    s_last = torch.as_tensor(obs_dict[a_id], dtype=torch.float32, device=self.device)
                    v_last_k[kk] = float(self.critic(s_last).item())

            # Per-agent GAE + return
            adv_tk = np.zeros((T, K), dtype=np.float32)
            ret_tk = np.zeros((T, K), dtype=np.float32)
            for kk in range(K):
                adv_kk, ret_kk = compute_gae(
                    R_tk[:, kk], V_tk[:, kk],
                    TERM_tk[:, kk], TRUNC_tk[:, kk],
                    float(v_last_k[kk]), self.gamma, self.lam,
                )
                adv_tk[:, kk] = adv_kk
                ret_tk[:, kk] = ret_kk

            # Flatten back to (T*K,) in the same stride as the original buffers
            adv = adv_tk.reshape(-1)
            ret = ret_tk.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            S_t = torch.as_tensor(S, device=self.device)
            A_t = torch.as_tensor(A, device=self.device)
            LP_t = torch.as_tensor(LP, device=self.device)
            ADV_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            RET_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
            M_t = torch.as_tensor(M, dtype=torch.bool, device=self.device)

            n = S_t.shape[0]
            for _ in range(self.epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.minibatch_size):
                    sel = idx[start:start + self.minibatch_size]
                    logits = self.actor(S_t[sel])
                    logits_m = safe_mask_logits(logits, M_t[sel])
                    d_t = torch.distributions.Categorical(logits=logits_m)
                    new_logp = d_t.log_prob(A_t[sel])
                    ratio = (new_logp - LP_t[sel]).exp()
                    surr1 = ratio * ADV_t[sel]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ADV_t[sel]
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_pred = self.critic(S_t[sel])
                    vf_loss = F.mse_loss(v_pred, RET_t[sel])
                    ent = d_t.entropy().mean()
                    loss = pi_loss + self.vf_coef * vf_loss - ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                              self.max_grad_norm)
                    opt.step()

            mean_return = float(np.mean(ep_rets)) if ep_rets else 0.0
            self.rollout_iter.append(it)
            self.rollout_mean_return.append(mean_return)
            if self.logger is not None:
                self.logger.log(iter=it, rollout_mean_return=mean_return,
                                 ent_coef=ent_coef,
                                 mean_served=float(np.mean(ep_served_list)) if ep_served_list else 0.0)


# ====================================================================
# MAPPO — shared actor, CENTRALISED critic over concatenated obs
# ====================================================================

class MAPPOTrainer:
    def __init__(self, env_fn: Callable, *,
                 hidden_dim: int = 128, n_hidden: int = 2,
                 critic_hidden: int = 256,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.10, ent_coef_end: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 steps_per_iter: int = 2048, iters: int = 80,
                 minibatch_size: int = 256, epochs: int = 8,
                 normalise_rewards: bool = True,
                 seed: int = 0, logger=None):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim; self.n_hidden = n_hidden
        self.critic_hidden = critic_hidden
        self.lr = lr; self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef; self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef; self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.minibatch_size = minibatch_size; self.epochs = epochs
        self.normalise_rewards = normalise_rewards
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: Optional[SharedActor] = None
        self.critic: Optional[CentralisedCritic] = None
        self.rollout_iter: list = []; self.rollout_mean_return: list = []

    @torch.no_grad()
    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in env.agents:
            s = torch.as_tensor(obs_dict[a], dtype=torch.float32, device=self.device)
            logits = self.actor(s)
            mask = np.asarray(info_dict[a].get("action_mask", []), dtype=bool)
            if mask.size > 0 and mask.any():
                m_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits = safe_mask_logits(logits, m_t)
            out[a] = int(torch.argmax(logits).item())
        return out

    def save_checkpoint(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load_checkpoint(self, path: str, obs_dim_per_agent: int, n_agents: int, n_actions: int) -> None:
        self.actor = SharedActor(obs_dim_per_agent, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
        self.critic = CentralisedCritic(obs_dim_per_agent, n_agents, self.critic_hidden, self.n_hidden).to(self.device)
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def train(self, eval_every_iters: int = 10, eval_episodes: int = 5):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        K = len(env.agents)
        obs_dict, info_dict = env.reset(seed=self.seed)
        obs_dim = obs_dict[env.agents[0]].shape[0]
        n_actions = env.action_space.n

        if self.actor is None:
            self.actor = SharedActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
            self.critic = CentralisedCritic(obs_dim, K, self.critic_hidden, self.n_hidden).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        for it in range(self.iters):
            progress = it / max(self.iters - 1, 1)
            ent_coef = self.ent_coef + (self.ent_coef_end - self.ent_coef) * progress

            S_all = []          # per-agent obs, shape (T*K, obs_dim)
            A_all = []
            R_all = []          # team reward shared to every agent
            TERM_all = []
            TRUNC_all = []
            LP_all = []
            M_all = []
            GLOBAL_STATES = []  # per-step global state, shape (T, K*obs_dim)
            V_GLOBAL = []       # per-step critic eval

            ep_ret = 0.0; ep_rets = []; ep_served_list = []

            for _ in range(self.steps_per_iter):
                step_actions: Dict[str, int] = {}
                per_agent_cache = []
                global_obs = np.concatenate([obs_dict[a] for a in env.agents])

                for k, a_id in enumerate(env.agents):
                    s_vec = obs_dict[a_id]
                    mask_np = np.asarray(info_dict[a_id].get("action_mask", []), dtype=bool)
                    s_t = torch.as_tensor(s_vec, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        logits = self.actor(s_t)
                        m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                        dist = torch.distributions.Categorical(logits=safe_mask_logits(logits, m_t))
                        a_t = dist.sample()
                        logp = dist.log_prob(a_t)
                    step_actions[a_id] = int(a_t.item())
                    per_agent_cache.append((s_vec, int(a_t.item()), float(logp.item()), mask_np))

                with torch.no_grad():
                    g_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
                    v_g = float(self.critic(g_t).item())
                next_obs, rewards, term, trunc, next_info = env.step(step_actions)
                team_r = float(sum(rewards.values()))
                done = any(term.values()) or any(trunc.values())

                GLOBAL_STATES.append(global_obs)
                V_GLOBAL.append(v_g)
                for k, (s_vec, a, lp, m_) in enumerate(per_agent_cache):
                    S_all.append(s_vec); A_all.append(a)
                    R_all.append(team_r)   # team reward, shared
                    TERM_all.append(bool(term[env.agents[k]])); TRUNC_all.append(bool(trunc[env.agents[k]]))
                    LP_all.append(lp); M_all.append(m_)
                ep_ret += team_r
                obs_dict, info_dict = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret / K)
                    ep_served_list.append(int(info_dict[env.agents[0]].get("customers_served", 0)))
                    ep_ret = 0.0
                    obs_dict, info_dict = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            T = len(V_GLOBAL)
            S_all = np.asarray(S_all, dtype=np.float32)   # (T*K, obs_dim)
            A_all = np.asarray(A_all, dtype=np.int64)
            R_all_raw = np.asarray(R_all, dtype=np.float32)
            LP_all = np.asarray(LP_all, dtype=np.float32)
            M_all = np.asarray(M_all, dtype=np.bool_)
            V_global = np.asarray(V_GLOBAL, dtype=np.float32)
            TERM_flat = np.asarray(TERM_all, dtype=np.bool_).reshape(T, K).any(axis=1)
            TRUNC_flat = np.asarray(TRUNC_all, dtype=np.bool_).reshape(T, K).any(axis=1)

            # Global reward per step = team sum
            R_step = R_all_raw.reshape(T, K)[:, 0]  # per-step team reward (same across agents)
            if self.normalise_rewards:
                rms.update(R_step)
                R_step_norm = rms.normalize(R_step).astype(np.float32)
            else:
                R_step_norm = R_step

            with torch.no_grad():
                g_last = torch.as_tensor(np.concatenate([obs_dict[a] for a in env.agents]),
                                          dtype=torch.float32, device=self.device)
                v_last = float(self.critic(g_last).item())
            adv_step, ret_step = compute_gae(R_step_norm, V_global, TERM_flat, TRUNC_flat,
                                              v_last, self.gamma, self.lam)
            # Broadcast step-level advantage to all agents in that step
            adv = np.repeat(adv_step, K)
            ret = np.repeat(ret_step, K)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            S_t = torch.as_tensor(S_all, device=self.device)
            A_t = torch.as_tensor(A_all, device=self.device)
            LP_t = torch.as_tensor(LP_all, device=self.device)
            ADV_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            RET_step_t = torch.as_tensor(ret_step, dtype=torch.float32, device=self.device)
            M_t = torch.as_tensor(M_all, dtype=torch.bool, device=self.device)
            GS_t = torch.as_tensor(np.asarray(GLOBAL_STATES), dtype=torch.float32, device=self.device)

            n = S_t.shape[0]
            for _ in range(self.epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.minibatch_size):
                    sel = idx[start:start + self.minibatch_size]
                    logits = self.actor(S_t[sel])
                    d_t = torch.distributions.Categorical(logits=safe_mask_logits(logits, M_t[sel]))
                    new_logp = d_t.log_prob(A_t[sel])
                    ratio = (new_logp - LP_t[sel]).exp()
                    surr1 = ratio * ADV_t[sel]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ADV_t[sel]
                    pi_loss = -torch.min(surr1, surr2).mean()
                    ent = d_t.entropy().mean()
                    actor_loss = pi_loss - ent_coef * ent
                    opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    opt.step()

                # Critic update — use global state batches (size T, not T*K)
                idx_c = torch.randperm(T, device=self.device)
                for start in range(0, T, max(self.minibatch_size // K, 1)):
                    sel_c = idx_c[start:start + max(self.minibatch_size // K, 1)]
                    v_pred = self.critic(GS_t[sel_c])
                    vf_loss = F.mse_loss(v_pred, RET_step_t[sel_c])
                    opt.zero_grad()
                    (self.vf_coef * vf_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    opt.step()

            mean_return = float(np.mean(ep_rets)) if ep_rets else 0.0
            self.rollout_iter.append(it)
            self.rollout_mean_return.append(mean_return)
            if self.logger is not None:
                self.logger.log(iter=it, rollout_mean_return=mean_return,
                                 ent_coef=ent_coef,
                                 mean_served=float(np.mean(ep_served_list)) if ep_served_list else 0.0)


# ====================================================================
# IA3C — simplified single-process A3C (each drone = one logical worker,
# syncing via shared actor parameters and 1-step TD updates).
# ====================================================================

class IA3CTrainer:
    """Synchronous A2C-style multi-agent trainer (I use "IA3C" name for
    consistency with the paper but this is a *synchronous* A2C baseline,
    not a torch.multiprocessing A3C). Each drone shares parameters, advantage
    is n-step TD with GAE, NO PPO clipping, single pass per batch, higher lr.

    This is what the multi-agent RL community usually calls "IA2C" or
    "shared A2C"; we label it A3C-style to match the ablation requested
    in the paper. For an asynchronous version, wrap this trainer in
    torch.multiprocessing workers — left as a future extension.
    """
    def __init__(self, env_fn: Callable, *,
                 hidden_dim: int = 128, n_hidden: int = 2,
                 lr: float = 7e-4, gamma: float = 0.99, lam: float = 0.95,
                 ent_coef: float = 0.02, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 steps_per_iter: int = 2048, iters: int = 80,
                 normalise_rewards: bool = True,
                 seed: int = 0, logger=None, **unused):
        self.env_fn = env_fn
        self.hidden_dim = hidden_dim; self.n_hidden = n_hidden
        self.lr = lr; self.gamma = gamma; self.lam = lam
        self.ent_coef = ent_coef; self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.steps_per_iter = steps_per_iter; self.iters = iters
        self.normalise_rewards = normalise_rewards
        self.seed = seed; self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor: Optional[SharedActor] = None
        self.critic: Optional[DecentralisedCritic] = None
        self.rollout_iter: list = []; self.rollout_mean_return: list = []

    @torch.no_grad()
    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in env.agents:
            s = torch.as_tensor(obs_dict[a], dtype=torch.float32, device=self.device)
            logits = self.actor(s)
            mask = np.asarray(info_dict[a].get("action_mask", []), dtype=bool)
            if mask.size > 0 and mask.any():
                m_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits = safe_mask_logits(logits, m_t)
            out[a] = int(torch.argmax(logits).item())
        return out

    def save_checkpoint(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load_checkpoint(self, path: str, obs_dim: int, n_actions: int) -> None:
        self.actor = SharedActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
        self.critic = DecentralisedCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def train(self, eval_every_iters: int = 10, eval_episodes: int = 5):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(1)

        env = self.env_fn()
        K = len(env.agents)
        obs_dict, info_dict = env.reset(seed=self.seed)
        obs_dim = obs_dict[env.agents[0]].shape[0]
        n_actions = env.action_space.n

        if self.actor is None:
            self.actor = SharedActor(obs_dim, n_actions, self.hidden_dim, self.n_hidden).to(self.device)
            self.critic = DecentralisedCritic(obs_dim, self.hidden_dim, self.n_hidden).to(self.device)
        opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        rms = RunningStd()

        for it in range(self.iters):
            S, A, R, TERM, TRUNC, V, M = [], [], [], [], [], [], []
            ep_ret = 0.0; ep_rets = []; ep_served_list = []

            for _ in range(self.steps_per_iter):
                step_actions: Dict[str, int] = {}
                per_agent_cache = []
                for k, a_id in enumerate(env.agents):
                    s_vec = obs_dict[a_id]
                    mask_np = np.asarray(info_dict[a_id].get("action_mask", []), dtype=bool)
                    s_t = torch.as_tensor(s_vec, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        logits = self.actor(s_t)
                        m_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                        dist = torch.distributions.Categorical(logits=safe_mask_logits(logits, m_t))
                        a_t = dist.sample()
                        v_t = self.critic(s_t)
                    step_actions[a_id] = int(a_t.item())
                    per_agent_cache.append((s_vec, int(a_t.item()), float(v_t.item()), mask_np))

                next_obs, rewards, term, trunc, next_info = env.step(step_actions)
                done = any(term.values()) or any(trunc.values())

                for k, (s_vec, a, v, m_) in enumerate(per_agent_cache):
                    a_id = env.agents[k]
                    S.append(s_vec); A.append(a); R.append(float(rewards[a_id]))
                    TERM.append(bool(term[a_id])); TRUNC.append(bool(trunc[a_id]))
                    V.append(v); M.append(m_)

                ep_ret += float(sum(rewards.values()))
                obs_dict, info_dict = next_obs, next_info
                if done:
                    ep_rets.append(ep_ret / K)
                    ep_served_list.append(int(info_dict[env.agents[0]].get("customers_served", 0)))
                    ep_ret = 0.0
                    obs_dict, info_dict = env.reset(seed=int(np.random.randint(0, 1_000_000)))

            T = self.steps_per_iter
            S = np.asarray(S, dtype=np.float32)
            obs_d = S.shape[1]
            S_tk = S.reshape(T, K, obs_d)
            A_tk = np.asarray(A, dtype=np.int64).reshape(T, K)
            R_raw_tk = np.asarray(R, dtype=np.float32).reshape(T, K)
            TERM_tk = np.asarray(TERM, dtype=np.bool_).reshape(T, K)
            TRUNC_tk = np.asarray(TRUNC, dtype=np.bool_).reshape(T, K)
            V_tk = np.asarray(V, dtype=np.float32).reshape(T, K)
            M_tk = np.asarray(M, dtype=np.bool_).reshape(T, K, -1)

            if self.normalise_rewards:
                rms.update(R_raw_tk.reshape(-1))
                R_tk = rms.normalize(R_raw_tk.reshape(-1)).astype(np.float32).reshape(T, K)
            else:
                R_tk = R_raw_tk

            # Bootstrap per agent
            with torch.no_grad():
                v_last_k = np.zeros(K, dtype=np.float32)
                for kk, a_id in enumerate(env.agents):
                    s_last = torch.as_tensor(obs_dict[a_id], dtype=torch.float32, device=self.device)
                    v_last_k[kk] = float(self.critic(s_last).item())

            adv_tk = np.zeros((T, K), dtype=np.float32)
            ret_tk = np.zeros((T, K), dtype=np.float32)
            for kk in range(K):
                a_kk, r_kk = compute_gae(R_tk[:, kk], V_tk[:, kk],
                                          TERM_tk[:, kk], TRUNC_tk[:, kk],
                                          float(v_last_k[kk]), self.gamma, self.lam)
                adv_tk[:, kk] = a_kk
                ret_tk[:, kk] = r_kk

            S_flat = S_tk.reshape(-1, obs_d)
            A_flat = A_tk.reshape(-1)
            adv_flat = adv_tk.reshape(-1)
            ret_flat = ret_tk.reshape(-1)
            M_flat = M_tk.reshape(-1, M_tk.shape[-1])
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            S_t = torch.as_tensor(S_flat, device=self.device)
            A_t = torch.as_tensor(A_flat, device=self.device)
            ADV_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=self.device)
            RET_t = torch.as_tensor(ret_flat, dtype=torch.float32, device=self.device)
            M_t = torch.as_tensor(M_flat, dtype=torch.bool, device=self.device)

            # SINGLE PASS, NO CLIPPING — this is the A3C/A2C-style update
            logits = self.actor(S_t)
            d_t = torch.distributions.Categorical(logits=safe_mask_logits(logits, M_t))
            logp = d_t.log_prob(A_t)
            pi_loss = -(logp * ADV_t).mean()
            v_pred = self.critic(S_t)
            vf_loss = F.mse_loss(v_pred, RET_t)
            ent = d_t.entropy().mean()
            loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                      self.max_grad_norm)
            opt.step()

            mean_return = float(np.mean(ep_rets)) if ep_rets else 0.0
            self.rollout_iter.append(it)
            self.rollout_mean_return.append(mean_return)
            if self.logger is not None:
                self.logger.log(iter=it, rollout_mean_return=mean_return,
                                 mean_served=float(np.mean(ep_served_list)) if ep_served_list else 0.0)


def build_marl(algo: str, env_fn: Callable, **kwargs):
    algo = algo.lower()
    if algo == "ippo":
        return IPPOTrainer(env_fn, **kwargs)
    if algo == "mappo":
        return MAPPOTrainer(env_fn, **kwargs)
    if algo in ("ia3c", "a3c"):
        return IA3CTrainer(env_fn, **kwargs)
    raise ValueError(f"unknown algo: {algo}")
