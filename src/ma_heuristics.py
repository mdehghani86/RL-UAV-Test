"""Multi-agent heuristic baselines for UAV routing.

Four heuristics, in order of sophistication:
  1. ParallelGreedyDB    — each drone runs single-agent Greedy-DB; conflicts go to closest drone
  2. ClusterFirstGreedy  — k-means split customers by drone, each runs Greedy-DB on its cluster
  3. ClusterFirstMaxReward — k-means split, then per-cluster solve tries to max on-time services
  4. CentralisedOracle    — OR-Tools VRPTW solver (upper bound, optional dependency)

All expose `act_all(env, obs_dict, info_dict) -> {agent_id: action}`.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional

import numpy as np


# ============ 1. Parallel Greedy-DB ============

class ParallelGreedyDB:
    """Each drone independently runs Greedy-DB. Ties (same customer target)
    resolved by the environment's conflict-resolution rule."""

    def __init__(self, w_deadline=1.2, w_dist=1.0, w_slack=1.2, low_batt=0.30):
        self.w_deadline, self.w_dist, self.w_slack = w_deadline, w_dist, w_slack
        self.low_batt = low_batt

    def _pick(self, env, k: int) -> int:
        """Picks a padded-action-index for drone k."""
        cfg = env.config
        N, M = cfg.num_customers, cfg.num_chargers
        pN = cfg.padded_num_customers()
        mask = env._agent_mask(k)
        if not mask.any():
            return 0
        if env.visited_mask.sum() == N:
            # all served; go home if possible
            return 0 if mask[0] else int(np.where(mask)[0][0])

        cur = int(env.current_node[k])
        batt_frac = float(env.battery[k]) / cfg.battery_capacity
        charger_padded = [1 + pN + i for i in range(M)]
        if batt_frac < self.low_batt:
            best, best_d = None, 1e18
            for pa in charger_padded:
                if not mask[pa]:
                    continue
                real_a = env._padded_to_real_action(pa)
                d = env._distance(cur, real_a)
                if d < best_d:
                    best_d, best = d, pa
            if best is not None:
                return int(best)

        best_a, best_s = None, -1e18
        for i in range(N):
            pa = 1 + i
            if not mask[pa] or env.visited_mask[i] == 1:
                continue
            real_a = env._padded_to_real_action(pa)
            d = env._distance(cur, real_a)
            ddl = float(env.deadlines[i])
            slack = max(ddl - float(env.elapsed_time[k]), 1e-6)
            s = (self.w_deadline / (ddl + 1e-6)
                 + self.w_dist / (d + 1e-6)
                 + self.w_slack / slack)
            if s > best_s:
                best_s, best_a = s, pa
        if best_a is not None:
            return int(best_a)
        for pa in charger_padded:
            if mask[pa]:
                return int(pa)
        return 0 if mask[0] else int(np.where(mask)[0][0])

    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        return {a: self._pick(env, k) for k, a in enumerate(env.agents)}


# ============ 2. Cluster-First, Greedy-DB-Second ============

class ClusterFirstGreedy:
    """K-means cluster customers into K groups (one per drone). Each drone
    runs Greedy-DB restricted to its assigned cluster. If its cluster is
    exhausted, it can fall back to the nearest unserved customer in any cluster."""

    def __init__(self, w_deadline=1.2, w_dist=1.0, w_slack=1.2, low_batt=0.30,
                 deadline_weight: float = 0.0):
        """deadline_weight > 0 biases clustering toward same-urgency groups
        (customers with similar deadlines co-cluster). 0 = pure k-means on xy."""
        self.w_deadline, self.w_dist, self.w_slack = w_deadline, w_dist, w_slack
        self.low_batt = low_batt
        self.deadline_weight = deadline_weight
        self._assignments: Optional[np.ndarray] = None  # length N, in [0, K)
        self._initialised_for: Optional[int] = None  # id(env) we initialised for

    def _assign_clusters(self, env) -> None:
        cfg = env.config
        K = cfg.num_drones
        N = cfg.num_customers
        W, H = cfg.map_size
        pts = env.node_positions[1:1 + N].astype(np.float64).copy()
        if self.deadline_weight > 0:
            ddl_norm = (env.deadlines - cfg.deadline_min) / (cfg.deadline_max - cfg.deadline_min + 1e-9)
            pts = np.hstack([pts / np.array([W, H]),
                             self.deadline_weight * ddl_norm.reshape(-1, 1)])
        # K-means (Lloyd's) — small N, 20 iters max is plenty
        rng = np.random.default_rng(0)
        idx0 = rng.choice(N, size=K, replace=False)
        centroids = pts[idx0].copy()
        assign = np.zeros(N, dtype=np.int32)
        for _ in range(20):
            dists = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=-1)
            new_assign = np.argmin(dists, axis=1)
            if np.array_equal(new_assign, assign):
                break
            assign = new_assign
            for c in range(K):
                pts_c = pts[assign == c]
                if pts_c.shape[0] > 0:
                    centroids[c] = pts_c.mean(axis=0)
        self._assignments = assign
        self._initialised_for = id(env)

    def _pick(self, env, k: int) -> int:
        cfg = env.config
        N, M = cfg.num_customers, cfg.num_chargers
        pN = cfg.padded_num_customers()
        mask = env._agent_mask(k)
        if not mask.any():
            return 0
        if env.visited_mask.sum() == N:
            return 0 if mask[0] else int(np.where(mask)[0][0])

        cur = int(env.current_node[k])
        batt_frac = float(env.battery[k]) / cfg.battery_capacity
        charger_padded = [1 + pN + i for i in range(M)]
        if batt_frac < self.low_batt:
            best, best_d = None, 1e18
            for pa in charger_padded:
                if not mask[pa]:
                    continue
                d = env._distance(cur, env._padded_to_real_action(pa))
                if d < best_d:
                    best_d, best = d, pa
            if best is not None:
                return int(best)

        # First try own cluster; if exhausted, allow any unvisited customer
        my_cluster = self._assignments
        best_a, best_s = None, -1e18
        for i in range(N):
            if env.visited_mask[i] == 1:
                continue
            if my_cluster[i] != k:
                continue
            pa = 1 + i
            if not mask[pa]:
                continue
            d = env._distance(cur, env._padded_to_real_action(pa))
            ddl = float(env.deadlines[i])
            slack = max(ddl - float(env.elapsed_time[k]), 1e-6)
            s = (self.w_deadline / (ddl + 1e-6)
                 + self.w_dist / (d + 1e-6)
                 + self.w_slack / slack)
            if s > best_s:
                best_s, best_a = s, pa
        if best_a is not None:
            return int(best_a)

        # Fallback: nearest feasible unserved customer from any cluster
        for i in range(N):
            if env.visited_mask[i] == 1:
                continue
            pa = 1 + i
            if not mask[pa]:
                continue
            d = env._distance(cur, env._padded_to_real_action(pa))
            slack = max(env.deadlines[i] - env.elapsed_time[k], 1e-6)
            s = self.w_dist / (d + 1e-6) + self.w_slack / slack
            if s > best_s:
                best_s, best_a = s, pa
        if best_a is not None:
            return int(best_a)

        for pa in charger_padded:
            if mask[pa]:
                return int(pa)
        return 0 if mask[0] else int(np.where(mask)[0][0])

    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        if self._initialised_for != id(env) or self._assignments is None:
            self._assign_clusters(env)
        return {a: self._pick(env, k) for k, a in enumerate(env.agents)}


# ============ 3. Cluster-First, Max-Reward-Second ============

class ClusterFirstMaxReward(ClusterFirstGreedy):
    """Same clustering, but per-drone picks the customer that MAXIMISES
    expected immediate reward subject to feasibility and deadline. This is
    a tighter variant: score = service_bonus − tardiness_penalty − distance_cost."""

    def __init__(self, w_service: float = 30.0, w_tardiness: float = 0.2,
                 w_dist_cost: float = 0.1, low_batt: float = 0.30):
        super().__init__(low_batt=low_batt)
        self.w_service = w_service
        self.w_tardiness = w_tardiness
        self.w_dist_cost = w_dist_cost

    def _pick(self, env, k: int) -> int:
        cfg = env.config
        N, M = cfg.num_customers, cfg.num_chargers
        pN = cfg.padded_num_customers()
        mask = env._agent_mask(k)
        if not mask.any():
            return 0
        if env.visited_mask.sum() == N:
            return 0 if mask[0] else int(np.where(mask)[0][0])

        cur = int(env.current_node[k])
        batt_frac = float(env.battery[k]) / cfg.battery_capacity
        charger_padded = [1 + pN + i for i in range(M)]
        if batt_frac < self.low_batt:
            best, best_d = None, 1e18
            for pa in charger_padded:
                if not mask[pa]:
                    continue
                d = env._distance(cur, env._padded_to_real_action(pa))
                if d < best_d:
                    best_d, best = d, pa
            if best is not None:
                return int(best)

        def _score(i: int) -> float:
            if env.visited_mask[i] == 1:
                return -1e18
            pa = 1 + i
            if not mask[pa]:
                return -1e18
            real_a = env._padded_to_real_action(pa)
            d = env._distance(cur, real_a)
            tt = d / (cfg.drone_speed + 1e-9)
            arrival = float(env.elapsed_time[k]) + tt
            tardiness = max(0.0, arrival - float(env.deadlines[i]))
            return self.w_service - self.w_tardiness * tardiness - self.w_dist_cost * d

        # Prefer own cluster
        own = [i for i in range(N) if self._assignments[i] == k and env.visited_mask[i] == 0]
        best_a, best_s = None, -1e18
        for i in own:
            s = _score(i)
            if s > best_s:
                best_s, best_a = s, 1 + i
        if best_a is not None and best_s > -1e6:
            return int(best_a)

        # Fallback: any cluster
        for i in range(N):
            s = _score(i)
            if s > best_s:
                best_s, best_a = s, 1 + i
        if best_a is not None and best_s > -1e6:
            return int(best_a)

        for pa in charger_padded:
            if mask[pa]:
                return int(pa)
        return 0 if mask[0] else int(np.where(mask)[0][0])


# ============ 4. Centralised Oracle (optional; OR-Tools dependency) ============

class CentralisedOracleStub:
    """Placeholder. Full implementation requires `ortools` which may not be
    installed on the cluster. If ortools is importable, a real VRPTW solve
    runs at reset and the computed routes are followed. Otherwise this falls
    back to ClusterFirstMaxReward so the pipeline still produces a row."""

    def __init__(self):
        try:
            from ortools.constraint_solver import routing_enums_pb2, pywrapcp  # type: ignore
            self._has_ortools = True
        except Exception:
            self._has_ortools = False
        self._fallback = ClusterFirstMaxReward()
        self._routes: Optional[List[List[int]]] = None  # per-drone list of real-node indices
        self._progress: Optional[List[int]] = None      # index-into-route per drone
        self._initialised_for: Optional[int] = None

    def _solve(self, env) -> None:
        # ORTools solve left as future work; for now always fall back.
        self._routes = None
        self._progress = None
        self._initialised_for = id(env)

    def act_all(self, env, obs_dict, info_dict) -> Dict[str, int]:
        if not self._has_ortools:
            return self._fallback.act_all(env, obs_dict, info_dict)
        if self._initialised_for != id(env):
            self._solve(env)
        return self._fallback.act_all(env, obs_dict, info_dict)


HEURISTICS = {
    "parallel_greedy": ParallelGreedyDB,
    "cluster_greedy": ClusterFirstGreedy,
    "cluster_maxreward": ClusterFirstMaxReward,
    # "oracle": CentralisedOracleStub,  # stubbed — uncomment once OR-Tools VRPTW is wired
}


def evaluate_ma_policy(env_fn, policy, episodes: int = 20, seed: int = 0) -> Dict[str, float]:
    """Run `episodes` rollouts of a multi-agent policy on envs produced by
    `env_fn`. `policy` must have `act_all(env, obs_dict, info_dict) -> actions`."""
    rng = np.random.default_rng(seed)
    returns = []
    kpis = {k: [] for k in [
        "customers_served", "completed", "elapsed_time", "ep_distance_total",
        "ep_tardiness_total", "ep_charger_visits_total", "ep_infeasible_total",
    ]}
    for _ in range(episodes):
        env = env_fn()
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_ret = 0.0
        done = False
        while not done:
            actions = policy.act_all(env, obs, info)
            obs, rew, term, trunc, info = env.step(actions)
            ep_ret += float(sum(rew.values()))
            done = any(term.values()) or any(trunc.values())
        returns.append(ep_ret)
        drone0 = env.agents[0]
        kpis["customers_served"].append(int(info[drone0]["customers_served"]))
        kpis["completed"].append(int(info[drone0]["completed"]))
        for k in ["elapsed_time", "ep_distance", "ep_tardiness", "ep_charger_visits", "ep_infeasible"]:
            total = sum(info[a].get(k, 0.0) for a in env.agents)
            key = "elapsed_time" if k == "elapsed_time" else f"{k}_total"
            if k == "elapsed_time":
                # use max across drones — mission makespan
                kpis[key].append(max(info[a][k] for a in env.agents))
            else:
                kpis[key].append(float(total))
    arr = np.asarray(returns, dtype=float)
    return {
        "episodes": episodes,
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1) if episodes > 1 else 0.0),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "mean_customers_served": float(np.mean(kpis["customers_served"])),
        "service_rate": float(np.mean(kpis["customers_served"]) / (env.config.num_customers + 1e-9)),
        "success_rate": float(np.mean(kpis["completed"])),
        "mean_makespan": float(np.mean(kpis["elapsed_time"])),
        "mean_team_distance": float(np.mean(kpis["ep_distance_total"])),
        "mean_team_tardiness": float(np.mean(kpis["ep_tardiness_total"])),
        "mean_team_infeasible": float(np.mean(kpis["ep_infeasible_total"])),
    }


if __name__ == "__main__":
    from src.ma_env import make_ma_env
    env_fn = lambda: make_ma_env({"num_drones": 2, "num_customers": 10, "mission_time": 40,
                                    "deadline_min": 8, "deadline_max": 35})
    for name, cls in HEURISTICS.items():
        pol = cls()
        s = evaluate_ma_policy(env_fn, pol, episodes=5, seed=0)
        print(f"{name:20s} return={s['mean_return']:6.1f}  served={s['mean_customers_served']:.1f}/10  "
              f"success={s['success_rate']*100:.0f}%  makespan={s['mean_makespan']:.1f}")
