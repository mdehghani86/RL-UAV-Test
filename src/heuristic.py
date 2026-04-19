"""Heuristic baselines for UAV routing: Greedy-DB, Nearest-Deadline-First,
Nearest-Neighbour, Random. All expose the same act(env, obs, info) -> int API."""
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


class NearestDeadlineFirstHeuristic:
    """Always go to the unvisited customer with the closest (soonest) deadline,
    subject to the action mask. Falls back to charger if battery low, depot if
    all served. Ignores distance — useful as an ablation showing deadline-only
    strategy is sub-optimal."""
    def __init__(self, low_batt: float = 0.25):
        self.low_batt = low_batt

    def act(self, env, obs, info):
        n_customers = env.config.num_customers
        n_nodes = env.config.total_nodes()
        visited = obs["visited"].astype(int)
        deadlines = obs["deadlines"].astype(float)
        batt_frac = float(obs["agent"][1])
        mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
        if not mask.any():
            return 0
        if visited[:n_customers].sum() == n_customers:
            return 0 if mask[0] else int(np.where(mask)[0][0])

        charger_ids = list(range(1 + n_customers, n_nodes))
        if batt_frac < self.low_batt:
            for a in charger_ids:
                if mask[a]:
                    return int(a)

        best_a, best_ddl = None, 1e18
        for a in range(1, 1 + n_customers):
            if not mask[a] or visited[a - 1] == 1:
                continue
            ddl = float(deadlines[a - 1])
            if ddl < best_ddl:
                best_ddl, best_a = ddl, a
        if best_a is not None:
            return int(best_a)
        for a in charger_ids:
            if mask[a]:
                return int(a)
        return 0 if mask[0] else int(np.where(mask)[0][0])


class NearestNeighbourHeuristic:
    """Pure TSP-nearest-neighbour: always go to the closest unvisited customer.
    Ignores deadlines and battery urgency. Classical VRP baseline."""
    def __init__(self, low_batt: float = 0.15):
        self.low_batt = low_batt

    def act(self, env, obs, info):
        n_customers = env.config.num_customers
        n_nodes = env.config.total_nodes()
        visited = obs["visited"].astype(int)
        pos = obs["node_positions"].astype(float)
        batt_frac = float(obs["agent"][1])
        mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
        if not mask.any():
            return 0
        if visited[:n_customers].sum() == n_customers:
            return 0 if mask[0] else int(np.where(mask)[0][0])

        cur = env.current_node
        charger_ids = list(range(1 + n_customers, n_nodes))
        if batt_frac < self.low_batt:
            best, best_d = None, 1e18
            for a in charger_ids:
                if not mask[a]:
                    continue
                d = float(np.linalg.norm(pos[cur] - pos[a]))
                if d < best_d:
                    best_d, best = d, a
            if best is not None:
                return int(best)

        best_a, best_d = None, 1e18
        for a in range(1, 1 + n_customers):
            if not mask[a] or visited[a - 1] == 1:
                continue
            d = float(np.linalg.norm(pos[cur] - pos[a]))
            if d < best_d:
                best_d, best_a = d, a
        if best_a is not None:
            return int(best_a)
        for a in charger_ids:
            if mask[a]:
                return int(a)
        return 0 if mask[0] else int(np.where(mask)[0][0])


class RandomPolicy:
    """Uniform-random feasible action. Lower-bound baseline."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, info):
        mask = np.asarray(info.get("action_mask", env.get_action_mask()), dtype=bool)
        valid = np.where(mask)[0]
        if valid.size == 0:
            return 0
        return int(self.rng.choice(valid))


HEURISTICS = {
    "greedy_db": GreedyDeadlineBatteryHeuristic,
    "nearest_deadline": NearestDeadlineFirstHeuristic,
    "nearest_neighbour": NearestNeighbourHeuristic,
    "random": RandomPolicy,
}


def evaluate_policy(env_fn, act_fn, episodes: int = 50, seed: int = 0,
                    collect_trajectories: bool = False):
    """Run `episodes` rollouts of `act_fn` against envs produced by `env_fn`.

    Returns (summary_dict, returns_array). `summary_dict` now exposes a rich
    KPI set for paper figures — per-episode arrays are also included under
    `raw_*` so downstream code can compute CIs, histograms, etc.
    If `collect_trajectories=True`, an extra "trajectories" key is added for
    path visualisation (list of dicts per episode with node sequence + xy).
    """
    rng = np.random.default_rng(seed)
    returns = []
    step_counts = []
    metrics = {k: [] for k in [
        "completed", "returned_to_depot", "customers_served",
        "ep_distance", "ep_energy", "ep_travel_time", "ep_tardiness",
        "ep_charger_visits", "ep_infeasible", "ep_repeat_visits", "elapsed_time",
    ]}
    trajectories = []

    for _ in range(episodes):
        env = env_fn()
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        ep_ret = 0.0
        ep_steps = 0
        traj = {"nodes": [int(env.current_node)], "positions": [env.node_positions[env.current_node].tolist()]}
        while not done:
            a = int(act_fn(env, obs, info))
            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += float(r)
            ep_steps += 1
            if collect_trajectories:
                traj["nodes"].append(int(env.current_node))
                traj["positions"].append(env.node_positions[env.current_node].tolist())
        returns.append(ep_ret)
        step_counts.append(ep_steps)
        for k in ["customers_served", "ep_distance", "ep_energy", "ep_travel_time",
                  "ep_tardiness", "ep_charger_visits", "ep_infeasible",
                  "ep_repeat_visits", "elapsed_time"]:
            metrics[k].append(float(info.get(k, 0.0)))
        metrics["completed"].append(int(info.get("completed", False)))
        metrics["returned_to_depot"].append(int(info.get("returned_to_depot", False)))
        if collect_trajectories:
            trajectories.append(traj)

    arr = np.asarray(returns, dtype=float)
    steps_arr = np.asarray(step_counts, dtype=float)

    def _summary(key: str) -> dict:
        vals = np.asarray(metrics[key], dtype=float)
        return {
            f"mean_{key}": float(vals.mean()),
            f"std_{key}":  float(vals.std(ddof=1) if episodes > 1 else 0.0),
            f"min_{key}":  float(vals.min()),
            f"max_{key}":  float(vals.max()),
        }

    # KPI derivations
    # Probe env for normalisation constants (first config).
    probe_env = env_fn()
    cfg = probe_env.config
    total_customers = cfg.num_customers
    mission_time = cfg.mission_time
    battery_cap = cfg.battery_capacity

    served_arr = np.asarray(metrics["customers_served"], dtype=float)
    tard_arr = np.asarray(metrics["ep_tardiness"], dtype=float)
    dist_arr = np.asarray(metrics["ep_distance"], dtype=float)
    elapsed_arr = np.asarray(metrics["elapsed_time"], dtype=float)
    energy_arr = np.asarray(metrics["ep_energy"], dtype=float)
    infeas_arr = np.asarray(metrics["ep_infeasible"], dtype=float)
    charge_arr = np.asarray(metrics["ep_charger_visits"], dtype=float)
    repeat_arr = np.asarray(metrics["ep_repeat_visits"], dtype=float)

    summary = {
        "episodes": episodes,
        # Primary
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1) if episodes > 1 else 0.0),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "median_return": float(np.median(arr)),
        "p25_return": float(np.percentile(arr, 25)),
        "p75_return": float(np.percentile(arr, 75)),
        # Success / coverage
        "success_rate": float(np.mean(metrics["completed"])),
        "returned_rate": float(np.mean(metrics["returned_to_depot"])),
        "mean_customers_served": float(served_arr.mean()),
        "std_customers_served": float(served_arr.std(ddof=1) if episodes > 1 else 0.0),
        "service_rate": float(served_arr.mean() / (total_customers + 1e-9)),
        # Efficiency
        "mean_distance": float(dist_arr.mean()),
        "mean_travel_time": float(np.asarray(metrics["ep_travel_time"]).mean()),
        "mean_elapsed_time": float(elapsed_arr.mean()),
        "time_utilisation": float(elapsed_arr.mean() / (mission_time + 1e-9)),
        "mean_energy": float(energy_arr.mean()),
        "energy_utilisation": float(energy_arr.mean() / (battery_cap + 1e-9)),
        "mean_steps": float(steps_arr.mean()),
        # Violations / quality
        "mean_tardiness": float(tard_arr.mean()),
        "tardiness_rate": float(np.mean(tard_arr > 1e-6)),
        "mean_infeasible": float(infeas_arr.mean()),
        "infeasible_rate": float(np.mean(infeas_arr > 0)),
        "mean_charger_visits": float(charge_arr.mean()),
        "mean_repeat_visits": float(repeat_arr.mean()),
        # Per-customer derived
        "mean_distance_per_customer": float(dist_arr.mean() / (max(served_arr.mean(), 1e-6))),
        "mean_time_per_customer": float(elapsed_arr.mean() / (max(served_arr.mean(), 1e-6))),
        # Raw arrays for downstream CIs / histograms
        "raw_returns": returns,
        "raw_customers_served": metrics["customers_served"],
        "raw_tardiness": metrics["ep_tardiness"],
        "raw_distance": metrics["ep_distance"],
        "raw_completed": metrics["completed"],
    }
    if collect_trajectories:
        summary["trajectories"] = trajectories
    return summary, arr
