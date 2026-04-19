"""Multi-agent scaling experiment: N x K sweep for the new MA paper.

Sweeps:
  N (customers) : {30, 50, 70, 100}
  K (drones)    : {2, 3, 5}

For every (N, K) cell we evaluate:
  4 heuristics    : parallel_greedy, cluster_greedy, cluster_maxreward, oracle
  3 MARL algos    : ippo, mappo, ia3c      (2 seeds each)

Result JSON files land in results_ma/NxK__{algo or heuristic}__s{seed}.json.

The cluster driver (submit_ma.sh) fires one SLURM job per (N, K, algo)
triple so the whole sweep runs in parallel on multiple GPUs. A single
argparse-driven `main` below lets each SLURM job request exactly one cell.

Usage:
    # Local (all cells, slow):
    python -m experiments.ma_exp_scaling
    # Per-cell (called by sbatch):
    python -m experiments.ma_exp_scaling --N 50 --K 3 --algo ippo --seed 0
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from src.ma_env import make_ma_env
from src.ma_heuristics import HEURISTICS, evaluate_ma_policy
from src.marl import build_marl
from src.run_logger import RunLogger


N_LIST = [30, 50, 70, 100]
K_LIST = [2, 3, 5]
ALGOS = ["ippo", "mappo", "ia3c"]
HEURISTIC_NAMES = list(HEURISTICS.keys())
SEEDS = [0, 1]
EPISODES_EVAL = 30

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results_ma"


def _settings_for(N: int, K: int) -> dict:
    """Adaptive problem sizing: bigger N -> more mission time, larger map."""
    # Mission time grows roughly with N/K (per-drone workload)
    mission_time = 30 + 2.2 * (N / max(K, 1))
    deadline_min = max(8.0, 0.15 * mission_time)
    deadline_max = 0.9 * mission_time
    # Keep map 100x100 — density increases with N (realistic for dense delivery zones)
    return dict(
        num_drones=K, num_customers=N, num_chargers=max(1, K // 2),
        mission_time=mission_time,
        deadline_min=deadline_min, deadline_max=deadline_max,
        map_size=(100.0, 100.0),
        max_steps=max(100, 6 * N),
        pad_num_customers=max(N_LIST), pad_num_chargers=max(3, max(K_LIST) // 2),
    )


def _iters_for(N: int, K: int) -> int:
    # Bigger N/K → more iters
    base = 40 + 2 * N + 5 * K
    return min(base, 160)


def run_cell(N: int, K: int, algo_or_heur: str, seed: int) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _settings_for(N, K)
    env_fn = lambda: make_ma_env(cfg)

    t0 = time.time()
    tag = f"N{N}_K{K}__{algo_or_heur}__s{seed}"
    print(f"=== {tag} ===", flush=True)

    if algo_or_heur in HEURISTIC_NAMES:
        policy = HEURISTICS[algo_or_heur]()
        summary = evaluate_ma_policy(env_fn, policy, episodes=EPISODES_EVAL, seed=seed)
        record = {"exp": "ma_scaling", "policy": algo_or_heur, "kind": "heuristic",
                  "N": N, "K": K, "seed": seed, **summary,
                  "wall_seconds": time.time() - t0}
    else:
        iters = _iters_for(N, K)
        logger = RunLogger(str(RESULTS_DIR / f"run_{tag}"),
                           {"exp": "ma_scaling", "algo": algo_or_heur, "N": N, "K": K, "seed": seed})
        kwargs = dict(hidden_dim=256, n_hidden=2, steps_per_iter=2048,
                      iters=iters, minibatch_size=512, epochs=6, seed=seed, logger=logger)
        trainer = build_marl(algo_or_heur, env_fn, **kwargs)
        trainer.train(eval_every_iters=max(iters // 5, 1), eval_episodes=5)
        summary = evaluate_ma_policy(env_fn, trainer, episodes=EPISODES_EVAL, seed=seed + 777)
        ck = RESULTS_DIR / f"ckpt__{tag}.pt"
        try:
            trainer.save_checkpoint(str(ck))
        except Exception as e:
            print(f"  checkpoint save failed: {e}", flush=True)
        record = {"exp": "ma_scaling", "policy": algo_or_heur, "kind": "marl",
                  "N": N, "K": K, "seed": seed, "iters": iters, **summary,
                  "wall_seconds": time.time() - t0}

    out_path = RESULTS_DIR / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  wrote {out_path.name}  return={record.get('mean_return'):.1f}  "
          f"served={record.get('mean_customers_served'):.1f}/{N}  "
          f"succ={record.get('success_rate', 0) * 100:.0f}%  "
          f"time={record['wall_seconds']:.0f}s", flush=True)
    return record


def run_all():
    """Local sequential driver — for testing only. Cluster uses one job per cell."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for N in N_LIST:
        for K in K_LIST:
            for h in HEURISTIC_NAMES:
                run_cell(N, K, h, seed=0)
            for algo in ALGOS:
                for seed in SEEDS:
                    run_cell(N, K, algo, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--algo", type=str, default=None,
                    help="One of: " + ", ".join(ALGOS + HEURISTIC_NAMES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.N is None:
        run_all()
    else:
        assert args.K is not None and args.algo is not None
        run_cell(args.N, args.K, args.algo, args.seed)


if __name__ == "__main__":
    main()
