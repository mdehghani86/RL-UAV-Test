"""Multi-agent UAV routing environment.

K drones share a depot, a customer set with per-customer deadlines, and M
chargers. Each drone has its own battery, position, and elapsed time. The
environment is synchronous: at every decision epoch, every drone chooses
one action simultaneously; the environment resolves them jointly.

Conflict resolution (two drones target the same customer):
  - The closer drone (shorter travel distance to the customer) serves it.
  - The losing drone's action is rolled back to a no-op (wait/idle); it
    pays no battery or time cost and stays in place.

Termination per episode:
  - All customers served AND all drones back at depot  → success
  - Any drone exhausts time horizon T               → truncated
  - No drone has any feasible action                → truncated
  - Step budget K_max reached                        → truncated

Rewards:
  - Team reward shared across drones, identical to single-agent shaped_potential
    (tight potential) — scales naturally with K via the shared customer set.
  - Per-agent individual component for local consistency (optional):
    each drone pays its own infeasibility/tardiness penalty.

This environment follows the PettingZoo ParallelEnv pattern: step() takes a
dict {agent_id: action} and returns dicts of obs/rewards/terminated/truncated/info.
Implemented without a PettingZoo dependency to keep the project lightweight.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class MultiUAVConfig:
    num_drones: int = 2
    num_customers: int = 30
    num_chargers: int = 1
    map_size: Tuple[float, float] = (100.0, 100.0)
    mission_time: float = 90.0
    max_steps: int = 150
    battery_capacity: float = 250.0
    battery_per_meter: float = 0.5
    drone_speed: float = 15.0
    deadline_min: float = 15.0
    deadline_max: float = 80.0

    # Reward scaling (shared baseline; potential shaping scale)
    w_service: float = 30.0
    w_complete: float = 60.0
    w_infeasible: float = 10.0
    w_repeat: float = 1.0
    w_tardiness: float = 0.2
    w_potential: float = 20.0
    alpha_partial: float = 20.0
    charger_time_cost: float = 1.0

    # Padding for curriculum / scaling transfer
    pad_num_customers: Optional[int] = None
    pad_num_chargers: Optional[int] = None

    layout_seed: Optional[int] = None

    def total_nodes(self) -> int:
        return 1 + self.num_customers + self.num_chargers

    def padded_num_customers(self) -> int:
        return self.pad_num_customers or self.num_customers

    def padded_num_chargers(self) -> int:
        return self.pad_num_chargers or self.num_chargers

    def padded_total_nodes(self) -> int:
        return 1 + self.padded_num_customers() + self.padded_num_chargers()


class MultiUAVEnv:
    """PettingZoo-style ParallelEnv: step(action_dict) -> (obs, rew, term, trunc, info) dicts."""

    metadata = {"name": "multi-uav-routing-v1"}

    def __init__(self, config: Optional[MultiUAVConfig] = None):
        self.config = config or MultiUAVConfig()
        self.agents: List[str] = [f"drone_{i}" for i in range(self.config.num_drones)]
        self.possible_agents = list(self.agents)

        pT = self.config.padded_total_nodes()
        pN = self.config.padded_num_customers()
        # Per-agent action space: same discrete set as single-agent (depot + customers + chargers)
        self.action_space = spaces.Discrete(pT)

        # Per-agent observation:
        #   self:  current_node/(n-1), battery/Bmax, time/T, step/Kmax  (4)
        #   others: per-other-drone node_norm, battery_frac, time_frac   (3 * (K-1))
        #   visited mask (shared)                                         (pN)
        #   deadlines (shared)                                           (pN)
        #   node positions (shared, relative to agent's current)         (pT * 2)
        #   action mask (agent-specific)                                 (pT)
        other_dim = 3 * (self.config.num_drones - 1)
        obs_dim = 4 + other_dim + pN + pN + 2 * pT + pT
        self._obs_dim = obs_dim

        self._global_rng = np.random.default_rng(self.config.layout_seed)
        self._build_nodes()
        self._reset_agents()

    # ---------------- layout ----------------
    def _build_nodes(self) -> None:
        cfg = self.config
        w, h = cfg.map_size
        n_total = cfg.total_nodes()
        self.node_positions = np.zeros((n_total, 2), dtype=np.float32)
        # depot in centre
        self.node_positions[0] = np.array([w / 2, h / 2], dtype=np.float32)
        # customers uniform
        self.node_positions[1:1 + cfg.num_customers] = self._global_rng.uniform(
            [0, 0], [w, h], size=(cfg.num_customers, 2)
        ).astype(np.float32)
        # chargers uniform
        self.node_positions[1 + cfg.num_customers:] = self._global_rng.uniform(
            [0, 0], [w, h], size=(cfg.num_chargers, 2)
        ).astype(np.float32)
        # Deadlines per customer
        self.deadlines = self._global_rng.uniform(
            cfg.deadline_min, cfg.deadline_max, size=cfg.num_customers
        ).astype(np.float32)

    def _reset_agents(self) -> None:
        K = self.config.num_drones
        self.current_node = np.zeros(K, dtype=np.int32)        # all start at depot
        self.battery = np.full(K, self.config.battery_capacity, dtype=np.float32)
        self.elapsed_time = np.zeros(K, dtype=np.float32)
        self.steps = 0
        self.visited_mask = np.zeros(self.config.num_customers, dtype=np.int8)
        self.ep_distance = np.zeros(K, dtype=np.float32)
        self.ep_tardiness = np.zeros(K, dtype=np.float32)
        self.ep_charger_visits = np.zeros(K, dtype=np.int32)
        self.ep_infeasible = np.zeros(K, dtype=np.int32)
        self._prev_potential = self._potential()

    # ---------------- geometry ----------------
    def _distance(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.node_positions[a] - self.node_positions[b]))

    def _travel_energy(self, d: float) -> float:
        return d * self.config.battery_per_meter

    def _travel_time(self, d: float) -> float:
        return d / (self.config.drone_speed + 1e-9)

    # ---------------- feasibility ----------------
    def _agent_mask(self, k: int) -> np.ndarray:
        """Feasibility mask for agent k. Returns padded-size bool vector."""
        cfg = self.config
        n_real = cfg.total_nodes()
        N = cfg.num_customers
        M = cfg.num_chargers
        pN = cfg.padded_num_customers()
        pT = cfg.padded_total_nodes()
        mask = np.zeros(pT, dtype=np.bool_)
        cur = int(self.current_node[k])
        time_remaining = max(cfg.mission_time - float(self.elapsed_time[k]), 0.0)
        # depot always feasible if reachable
        for real_a in range(n_real):
            if real_a == cur:
                continue
            d = self._distance(cur, real_a)
            if self._travel_energy(d) > self.battery[k] + 1e-9:
                continue
            if self._travel_time(d) > time_remaining + 1e-9:
                continue
            if 1 <= real_a <= N and self.visited_mask[real_a - 1] == 1:
                continue  # already served customer — infeasible
            # translate real_a → padded index
            if real_a == 0:
                mask[0] = True
            elif real_a <= N:
                mask[real_a] = True
            else:
                mask[1 + pN + (real_a - 1 - N)] = True
        return mask

    # ---------------- observation ----------------
    def _get_obs(self, k: int) -> np.ndarray:
        cfg = self.config
        n_real = cfg.total_nodes()
        N, M = cfg.num_customers, cfg.num_chargers
        pN, pT = cfg.padded_num_customers(), cfg.padded_total_nodes()
        W, H = cfg.map_size

        cur_k = int(self.current_node[k])
        agent_feats = [
            cur_k / max(n_real - 1, 1),
            self.battery[k] / cfg.battery_capacity,
            self.elapsed_time[k] / cfg.mission_time,
            self.steps / max(cfg.max_steps, 1),
        ]
        # Other-agents summary (ordered)
        others = []
        for j in range(cfg.num_drones):
            if j == k:
                continue
            others.append(self.current_node[j] / max(n_real - 1, 1))
            others.append(self.battery[j] / cfg.battery_capacity)
            others.append(self.elapsed_time[j] / cfg.mission_time)

        visited_padded = np.zeros(pN, dtype=np.float32)
        visited_padded[:N] = self.visited_mask.astype(np.float32)
        dmin, dmax = cfg.deadline_min, cfg.deadline_max
        ddl_padded = np.zeros(pN, dtype=np.float32)
        ddl_padded[:N] = (self.deadlines - dmin) / (dmax - dmin + 1e-9)

        # Relative-frame positions (offsets from agent k's current node, normalised)
        pos = np.zeros((pT, 2), dtype=np.float32)
        cur_xy = self.node_positions[cur_k]
        # depot
        pos[0] = (self.node_positions[0] - cur_xy) / np.array([W, H], dtype=np.float32)
        # customers (pad slot stays zero for non-existent)
        pos[1:1 + N] = (self.node_positions[1:1 + N] - cur_xy) / np.array([W, H], dtype=np.float32)
        # chargers
        offset = 1 + pN
        pos[offset:offset + M] = (self.node_positions[1 + N:1 + N + M] - cur_xy) / np.array([W, H], dtype=np.float32)

        mask = self._agent_mask(k).astype(np.float32)

        out = np.concatenate([
            np.asarray(agent_feats, dtype=np.float32),
            np.asarray(others, dtype=np.float32),
            visited_padded,
            ddl_padded,
            pos.ravel(),
            mask,
        ])
        assert out.shape[0] == self._obs_dim, f"obs_dim mismatch: {out.shape[0]} vs {self._obs_dim}"
        return out

    # ---------------- potential shaping (team) ----------------
    def _count_feasible_team(self) -> int:
        """Customers that SOME drone can still reach on time given remaining
        team capacity. Team-level version of single-agent _count_feasible."""
        cfg = self.config
        count = 0
        for i in range(cfg.num_customers):
            if self.visited_mask[i] == 1:
                continue
            nid = 1 + i
            feasible_by_any = False
            for k in range(cfg.num_drones):
                cur = int(self.current_node[k])
                d = self._distance(cur, nid)
                tt = self._travel_time(d)
                if self._travel_energy(d) > self.battery[k] + 1e-9:
                    continue
                if tt > max(cfg.mission_time - self.elapsed_time[k], 0.0) + 1e-9:
                    continue
                if self.elapsed_time[k] + tt <= self.deadlines[i] + 1e-9:
                    feasible_by_any = True
                    break
            if feasible_by_any:
                count += 1
        return count

    def _potential(self) -> float:
        cfg = self.config
        N = cfg.num_customers
        rho = float(self.visited_mask.sum()) / max(N, 1)
        if self.visited_mask.sum() == N:
            return 1.0
        # Min distance from ANY drone to ANY unvisited customer (team proximity)
        diag = math.hypot(*cfg.map_size)
        min_d = 1e9
        for i in range(N):
            if self.visited_mask[i] == 1:
                continue
            nid = 1 + i
            for k in range(cfg.num_drones):
                d = self._distance(int(self.current_node[k]), nid)
                if d < min_d:
                    min_d = d
        prox = min_d / (diag + 1e-9)
        # Battery reserve (min across drones — bottleneck)
        b_min = float(min(self.battery / cfg.battery_capacity))
        # Time used (max across drones)
        tau_max = float(max(self.elapsed_time / cfg.mission_time))
        feasible_frac = self._count_feasible_team() / (N + 1e-9)
        return (rho
                - 0.3 * prox
                + 0.10 * (b_min - 0.3)
                + 0.20 * feasible_frac
                - 0.15 * max(0.0, tau_max - rho))

    # ---------------- reset / step ----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._global_rng = np.random.default_rng(seed)
        self._build_nodes()
        self._reset_agents()
        obs = {a: self._get_obs(k) for k, a in enumerate(self.agents)}
        masks = {a: self._agent_mask(k) for k, a in enumerate(self.agents)}
        infos = {a: {"action_mask": masks[a]} for a in self.agents}
        return obs, infos

    def _padded_to_real_action(self, padded_action: int) -> int:
        cfg = self.config
        N = cfg.num_customers
        pN = cfg.padded_num_customers()
        M = cfg.num_chargers
        if padded_action == 0:
            return 0
        if 1 <= padded_action <= N:
            return padded_action
        if padded_action < 1 + pN:
            return -1  # pad-customer slot
        ci = padded_action - (1 + pN)
        if ci < M:
            return 1 + N + ci
        return -1  # pad-charger slot

    def step(self, actions: Dict[str, int]):
        cfg = self.config
        K = cfg.num_drones
        N = cfg.num_customers

        # Budget exhausted?
        if self.steps >= cfg.max_steps:
            rewards = {a: 0.0 for a in self.agents}
            term = {a: False for a in self.agents}
            trunc = {a: True for a in self.agents}
            obs = {a: self._get_obs(k) for k, a in enumerate(self.agents)}
            infos = self._build_infos()
            # Partial-completion team bonus
            share = float(self.visited_mask.sum()) / max(N, 1)
            bonus = cfg.alpha_partial * share / max(K, 1)
            for a in self.agents:
                rewards[a] = bonus
            return obs, rewards, term, trunc, infos

        self.steps += 1

        # Translate padded actions to real actions
        real_actions = np.full(K, -2, dtype=np.int32)  # -2 = no-op
        intents = []  # (k, real_a, dist, travel_t, energy)
        for k, a_id in enumerate(self.agents):
            padded = int(actions.get(a_id, 0))
            real_a = self._padded_to_real_action(padded)
            if real_a < 0 or real_a >= cfg.total_nodes() or real_a == int(self.current_node[k]):
                real_actions[k] = -1  # infeasible
                continue
            d = self._distance(int(self.current_node[k]), real_a)
            tt = self._travel_time(d)
            e = self._travel_energy(d)
            # Check per-agent feasibility (battery + time)
            if e > self.battery[k] + 1e-9 or tt > max(cfg.mission_time - self.elapsed_time[k], 0.0) + 1e-9:
                real_actions[k] = -1
                continue
            real_actions[k] = real_a
            intents.append((k, real_a, d, tt, e))

        # Conflict resolution: if multiple drones target the same customer,
        # the one with smallest travel distance wins.
        customer_bids: Dict[int, List[Tuple[int, float]]] = {}
        for k, ra, d, tt, e in intents:
            if 1 <= ra <= N:
                customer_bids.setdefault(ra, []).append((k, d))

        losers = set()
        for cust, bids in customer_bids.items():
            if len(bids) == 1:
                continue
            winner = min(bids, key=lambda t: t[1])[0]
            for k, _ in bids:
                if k != winner:
                    losers.add(k)

        rewards = {a: 0.0 for a in self.agents}
        for k, ra, d, tt, e in intents:
            a_id = self.agents[k]
            if k in losers:
                # rollback: no move, no cost, small waste penalty for coordination noise
                rewards[a_id] -= 0.5
                continue
            # Apply the move
            is_charger = ra > N
            self.elapsed_time[k] += tt + (cfg.charger_time_cost if is_charger else 0.0)
            self.battery[k] -= e
            self.ep_distance[k] += d
            self.current_node[k] = ra
            if is_charger:
                self.battery[k] = cfg.battery_capacity
                self.ep_charger_visits[k] += 1
            if 1 <= ra <= N:
                if self.visited_mask[ra - 1] == 0:
                    # Served new customer
                    self.visited_mask[ra - 1] = 1
                    tardiness = max(0.0, float(self.elapsed_time[k]) - float(self.deadlines[ra - 1]))
                    rewards[a_id] += cfg.w_service - cfg.w_tardiness * tardiness
                    self.ep_tardiness[k] += tardiness
                else:
                    rewards[a_id] -= cfg.w_repeat

        # Infeasibility penalties for drones with real_actions[k] == -1
        for k in range(K):
            if real_actions[k] == -1:
                rewards[self.agents[k]] -= cfg.w_infeasible
                self.ep_infeasible[k] += 1

        # Team potential shaping
        phi_new = self._potential()
        shaping = cfg.w_potential * (0.99 * phi_new - self._prev_potential)
        self._prev_potential = phi_new
        # Distribute team shaping equally
        for a in self.agents:
            rewards[a] += shaping / K

        # Completion check (all customers AND all drones at depot)
        all_served = bool(self.visited_mask.sum() == N)
        all_home = bool(all(self.current_node == 0))
        success = all_served and all_home
        if success:
            for a in self.agents:
                rewards[a] += cfg.w_complete / K

        # Termination
        all_stuck = all(not self._agent_mask(k).any() for k in range(K))
        terminated_flag = success
        truncated_flag = (not success) and (all_stuck or any(self.elapsed_time >= cfg.mission_time))

        term = {a: terminated_flag for a in self.agents}
        trunc = {a: truncated_flag for a in self.agents}
        obs = {a: self._get_obs(k) for k, a in enumerate(self.agents)}
        infos = self._build_infos()
        return obs, rewards, term, trunc, infos

    def _build_infos(self) -> Dict[str, dict]:
        infos: Dict[str, dict] = {}
        for k, a in enumerate(self.agents):
            infos[a] = {
                "action_mask": self._agent_mask(k),
                "customers_served": int(self.visited_mask.sum()),
                "elapsed_time": float(self.elapsed_time[k]),
                "battery": float(self.battery[k]),
                "ep_distance": float(self.ep_distance[k]),
                "ep_tardiness": float(self.ep_tardiness[k]),
                "ep_charger_visits": int(self.ep_charger_visits[k]),
                "ep_infeasible": int(self.ep_infeasible[k]),
                "completed": bool(self.visited_mask.sum() == self.config.num_customers
                                  and all(self.current_node == 0)),
            }
        return infos

    @property
    def obs_dim(self) -> int:
        return self._obs_dim


def make_ma_env(config: Optional[Dict[str, Any]] = None) -> MultiUAVEnv:
    cfg = MultiUAVConfig(**(config or {}))
    return MultiUAVEnv(cfg)


if __name__ == "__main__":
    env = make_ma_env({"num_drones": 2, "num_customers": 10})
    obs, info = env.reset(seed=0)
    print("agents:", env.agents)
    print("obs_dim:", env.obs_dim, "action_space:", env.action_space.n)
    for _ in range(5):
        acts = {}
        for a in env.agents:
            m = info[a]["action_mask"]
            valid = np.where(m)[0]
            acts[a] = int(np.random.choice(valid)) if valid.size else 0
        obs, r, term, trunc, info = env.step(acts)
        print(f"r={r} term={list(term.values())} trunc={list(trunc.values())} served={info[env.agents[0]]['customers_served']}")
        if any(term.values()) or any(trunc.values()):
            break
    print("OK")
