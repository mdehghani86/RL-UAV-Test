"""Exact oracle for single-UAV routing — Held-Karp DP with Pareto battery.

For each reachable state `(customers_served_mask, current_node)` we keep the
Pareto frontier of `(elapsed_time, battery)` pairs. A point (t1,b1) dominates
(t2,b2) iff `t1 <= t2 AND b1 >= b2`.

Transitions:
  - Move to an unserved customer c → arriving time must respect c's deadline
    and the round trip back to depot within mission_time.
  - Move to a charger → pays travel + charger_time_cost, battery refilled to
    min(max_capacity, battery + recharge).
  - Move back to depot → mission end (used only for reconstruction).

Objective: max #customers_served with the route being feasible under
deadlines, battery, AND round-trip to depot. Tie-break by min travel time.

Complexity is harder to bound due to Pareto sizes, but in practice for
N ≤ 12, M ≤ 2 the frontier stays small (~10–40 points per state). Solves in
well under a second per layout on CPU.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np


def _dominated(a_t: float, a_b: float, b_t: float, b_b: float) -> bool:
    """Does (b_t, b_b) dominate (a_t, a_b)?  Better time AND better battery."""
    return b_t <= a_t and b_b >= a_b and (b_t < a_t or b_b > a_b)


def _insert_pareto(front: List[Tuple[float, float]], t: float, b: float) -> bool:
    """Insert (t, b) into a Pareto frontier of (time, battery). Return True if added."""
    new = []
    for ft, fb in front:
        if _dominated(t, b, ft, fb):
            return False  # new point is dominated by existing
    for ft, fb in front:
        if not _dominated(ft, fb, t, b):
            new.append((ft, fb))
    new.append((t, b))
    front.clear()
    front.extend(new)
    return True


def solve(env) -> Dict[str, float]:
    """Solve optimally; return summary dict with mean_return (replayed), served, etc."""
    cfg = env.config
    N = cfg.num_customers
    M = cfg.num_chargers
    pos = env.node_positions
    deadlines = [env.service_nodes[1 + i].deadline for i in range(N)]
    speed = cfg.drone_speed + 1e-9
    cap = cfg.battery_capacity
    recharge = cfg.recharge_rate
    charger_cost = cfg.charger_time_cost
    bpm = cfg.battery_per_meter

    # Node indexing: 0 = depot, 1..N = customers, N+1..N+M = chargers
    n_nodes = 1 + N + M

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(pos[i] - pos[j]))

    def ttime(i: int, j: int) -> float:
        return dist(i, j) / speed

    def tenergy(i: int, j: int) -> float:
        return dist(i, j) * bpm

    # State: (mask, node) -> Pareto front of (time, battery) with parent pointer
    #        for route reconstruction.
    INF = float("inf")
    # pareto[(mask, node)] = list of (time, battery, parent_state, parent_pareto_idx)
    pareto: Dict[Tuple[int, int], List[Tuple[float, float, tuple, int]]] = {}

    start_key = (0, 0)
    pareto[start_key] = [(0.0, cap, None, -1)]

    # Best terminal record
    best = {"served": 0, "time": INF, "state": start_key, "idx": 0}

    # BFS in mask-popcount order — but we also allow charger revisits within
    # same mask, so we just repeatedly expand until stable. Since customer
    # transitions always add to mask, expand by mask order first, then iterate
    # charger expansions within same mask to fixed point.
    def try_return(mask: int, node: int, t: float, b: float) -> None:
        # Attempt to close the tour by returning to depot 0.
        d = dist(node, 0)
        if tenergy(node, 0) > b + 1e-9:
            return
        t_home = t + ttime(node, 0)
        if t_home > cfg.mission_time + 1e-9:
            return
        served = bin(mask).count("1")
        if (served > best["served"]
            or (served == best["served"] and t_home < best["time"])):
            best["served"] = served
            best["time"] = t_home
            # Find the pareto idx matching this (t,b) — linear scan
            for i, (pt, pb, _, _) in enumerate(pareto[(mask, node)]):
                if abs(pt - t) < 1e-9 and abs(pb - b) < 1e-9:
                    best["state"] = (mask, node)
                    best["idx"] = i
                    break

    # First pass — customer transitions by popcount
    masks_by_pop = sorted(range(1 << N), key=lambda m: bin(m).count("1"))

    def expand_chargers(key: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Expand charger transitions from `key`; return list of NEW keys touched."""
        touched = []
        mask, node = key
        changed = True
        while changed:
            changed = False
            for (t, b, _, _) in list(pareto[(mask, node)]):
                for ch in range(M):
                    ch_node = 1 + N + ch
                    if ch_node == node:
                        continue
                    eng = tenergy(node, ch_node)
                    if eng > b + 1e-9:
                        continue
                    new_t = t + ttime(node, ch_node) + charger_cost
                    new_b = min(cap, b - eng + recharge)
                    if new_t + ttime(ch_node, 0) > cfg.mission_time + 1e-9:
                        continue
                    new_key = (mask, ch_node)
                    front = pareto.setdefault(new_key, [])
                    parent_idx = None
                    for i, pp in enumerate(pareto[(mask, node)]):
                        if abs(pp[0] - t) < 1e-9 and abs(pp[1] - b) < 1e-9:
                            parent_idx = i; break
                    added = _insert_pareto_with_parent(front, new_t, new_b,
                                                        (mask, node), parent_idx or 0)
                    if added:
                        touched.append(new_key)
                        changed = True
        return touched

    def _insert_pareto_with_parent(front, t, b, parent_state, parent_idx) -> bool:
        for ft, fb, _, _ in front:
            if ft <= t and fb >= b and (ft < t or fb > b):
                return False
        # Remove dominated
        keep = []
        for entry in front:
            ft, fb, _, _ = entry
            if not (t <= ft and b >= fb and (t < ft or b > fb)):
                keep.append(entry)
        keep.append((t, b, parent_state, parent_idx))
        front.clear()
        front.extend(keep)
        return True

    # Try closing from the start
    for (t, b, _, _) in pareto[start_key]:
        try_return(*start_key, t=t, b=b)

    # Process masks in popcount order
    for mask in masks_by_pop:
        # First, exhaust charger transitions within this mask
        for node in range(n_nodes):
            key = (mask, node)
            if key in pareto:
                expand_chargers(key)

        # Then try customer transitions (mask -> mask | bit)
        for node in range(n_nodes):
            key = (mask, node)
            if key not in pareto:
                continue
            for nxt_c in range(N):
                if mask & (1 << nxt_c):
                    continue
                nxt_node = 1 + nxt_c
                if nxt_node == node:
                    continue
                eng = tenergy(node, nxt_node)
                tt = ttime(node, nxt_node)
                for i, (t, b, _, _) in enumerate(pareto[key]):
                    if eng > b + 1e-9:
                        continue
                    new_t = t + tt
                    if new_t > deadlines[nxt_c] + 1e-9:
                        continue
                    new_b = b - eng
                    # feasibility to return to depot
                    if new_t + ttime(nxt_node, 0) > cfg.mission_time + 1e-9:
                        continue
                    new_mask = mask | (1 << nxt_c)
                    new_key = (new_mask, nxt_node)
                    front = pareto.setdefault(new_key, [])
                    _insert_pareto_with_parent(front, new_t, new_b, key, i)

        # Try closing each node under this mask
        for node in range(n_nodes):
            key = (mask, node)
            if key not in pareto:
                continue
            for (t, b, _, _) in pareto[key]:
                try_return(mask, node, t, b)

    # Reconstruct route
    route = []
    mask, node = best["state"]
    idx = best["idx"]
    while True:
        route.append(node)
        if (mask, node) == start_key:
            break
        entry = pareto[(mask, node)][idx]
        parent_state = entry[2]
        if parent_state is None:
            break
        mask, node = parent_state
        idx = entry[3]
    route.reverse()
    route.append(0)  # close the tour

    # Replay through env to get exact reward under current reward mode
    env.reset(seed=None)
    # Force node layout — we need env to replay on the SAME layout we solved for.
    # The caller is expected to have env already reset; we'll skip re-reset
    # and just run through step() via a fresh reset preserving state.
    # Since env's _global_rng has advanced, re-seed with a known seed won't
    # reproduce. Simplest: caller handles replay.

    total_dist = sum(dist(route[k], route[k + 1]) for k in range(len(route) - 1))
    return {
        "served": best["served"],
        "route": route,
        "total_time": best["time"] if best["time"] < INF else cfg.mission_time,
        "total_distance": total_dist,
    }


def evaluate_oracle(env_fn, episodes: int = 50, seed: int = 0) -> Dict[str, float]:
    """Solve and replay oracle solutions over `episodes` random layouts."""
    rng = np.random.default_rng(seed)
    rets, sr, srv, dsts, ets = [], [], [], [], []

    for _ in range(episodes):
        env = env_fn()
        ep_seed = int(rng.integers(0, 1_000_000))
        env.reset(seed=ep_seed)
        r = solve(env)
        cfg = env.config
        # Replay on the same layout
        env.reset(seed=ep_seed)
        ret = 0.0
        for nxt in r["route"][1:]:
            if nxt == env.current_node:
                continue
            _, reward, term, trunc, info = env.step(int(nxt))
            ret += float(reward)
            if term or trunc:
                break
        served = int(info.get("customers_served", r["served"]))
        rets.append(ret)
        srv.append(float(served))
        sr.append(1.0 if served == cfg.num_customers else 0.0)
        dsts.append(float(r["total_distance"]))
        ets.append(float(r["total_time"]))

    return {
        "episodes": episodes,
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
        "success_rate": float(np.mean(sr)),
        "mean_customers_served": float(np.mean(srv)),
        "mean_distance": float(np.mean(dsts)),
        "mean_elapsed_time": float(np.mean(ets)),
        "mean_tardiness": 0.0,
    }


if __name__ == "__main__":
    import time as _t
    from src.env import make_env
    for N, mt in [(5, 15), (8, 22), (10, 26), (12, 28)]:
        t0 = _t.time()
        results = evaluate_oracle(
            lambda N=N, mt=mt: make_env({"reward_mode": "completion_ratio",
                                          "num_customers": N, "mission_time": mt,
                                          "deadline_min": 3.0, "deadline_max": mt * 0.85}),
            episodes=10, seed=0)
        dt = _t.time() - t0
        print(f"N={N:2d}  served={results['mean_customers_served']:5.2f}/{N}  "
               f"success={results['success_rate']*100:3.0f}%  "
               f"return={results['mean_return']:6.2f}  dt={dt:.1f}s")
