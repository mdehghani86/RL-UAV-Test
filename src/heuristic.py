"""Greedy deadline + battery heuristic baseline."""
from __future__ import annotations
import numpy as np


class GreedyDeadlineBatteryHeuristic:
    def __init__(self, w_deadline=1.2, w_dist=1.0, w_slack=1.2, low_batt=0.30):
        self.w_deadline = w_deadline
        self.w_dist = w_dist
        self.w_slack = w_slack
        self.low_batt = low_batt

    def act(self, env, obs, info):
        n_customers = env.config.num_customers
        n_nodes = env.config.total_nodes()
        visited = obs["visited"].astype(int)
        deadlines = obs["deadlines"].astype(float)
        pos = obs["node_positions"].astype(float)
        batt_frac = float(obs["agent"][1])
        time_frac = float(obs["agent"][2])

        mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
        if not mask.any():
            return 0

        if visited.sum() == n_customers:
            return 0 if mask[0] else int(np.where(mask)[0][0])

        charger_start = 1 + n_customers
        charger_ids = list(range(charger_start, n_nodes))

        if batt_frac < self.low_batt and charger_ids:
            cur = env.current_node
            best, best_d = None, 1e18
            for a in charger_ids:
                if not mask[a]:
                    continue
                d = float(np.linalg.norm(pos[cur] - pos[a]))
                if d < best_d:
                    best_d, best = d, a
            if best is not None:
                return int(best)

        cur = env.current_node
        best_a, best_s = None, -1e18
        for a in range(1, 1 + n_customers):
            if not mask[a]:
                continue
            ci = a - 1
            if visited[ci] == 1:
                continue
            d = float(np.linalg.norm(pos[cur] - pos[a]))
            ddl = float(deadlines[ci])
            slack = max(ddl - time_frac, 1e-6)
            s = (self.w_deadline / (ddl + 1e-6) +
                 self.w_dist / (d + 1e-6) +
                 self.w_slack / slack)
            if s > best_s:
                best_s, best_a = s, a

        if best_a is not None:
            return int(best_a)
        for a in charger_ids:
            if mask[a]:
                return int(a)
        if mask[0]:
            return 0
        return int(np.where(mask)[0][0])


def evaluate_policy(env_fn, act_fn, episodes: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    returns = []
    metrics = {k: [] for k in [
        "completed", "returned_to_depot", "customers_served",
        "ep_distance", "ep_energy", "ep_travel_time", "ep_tardiness",
        "ep_charger_visits", "ep_infeasible", "ep_repeat_visits", "elapsed_time",
    ]}
    for _ in range(episodes):
        env = env_fn()
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        ep_ret = 0.0
        while not done:
            a = int(act_fn(env, obs, info))
            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += float(r)
        returns.append(ep_ret)
        for k in ["customers_served", "ep_distance", "ep_energy", "ep_travel_time",
                  "ep_tardiness", "ep_charger_visits", "ep_infeasible",
                  "ep_repeat_visits", "elapsed_time"]:
            metrics[k].append(float(info.get(k, 0.0)))
        metrics["completed"].append(int(info.get("completed", False)))
        metrics["returned_to_depot"].append(int(info.get("returned_to_depot", False)))

    arr = np.asarray(returns, dtype=float)
    summary = {
        "episodes": episodes,
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1) if episodes > 1 else 0.0),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "success_rate": float(np.mean(metrics["completed"])),
        "returned_rate": float(np.mean(metrics["returned_to_depot"])),
        "mean_customers_served": float(np.mean(metrics["customers_served"])),
        "mean_tardiness": float(np.mean(metrics["ep_tardiness"])),
        "mean_distance": float(np.mean(metrics["ep_distance"])),
        "mean_charger_visits": float(np.mean(metrics["ep_charger_visits"])),
        "mean_infeasible": float(np.mean(metrics["ep_infeasible"])),
        "mean_elapsed_time": float(np.mean(metrics["elapsed_time"])),
    }
    return summary, arr
