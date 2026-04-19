"""Experiment 3 — Scaling sweep N in {5, 10, 15, 20, 25}.

Every N, every policy is evaluated on 50 episodes:
  - Heuristics: greedy_db, nearest_deadline, nearest_neighbour, random
  - PPO-enhanced: relative_frame + time_features + tight + cosine ent

All PPO runs use the SAME padded obs shape (pad to 25) so one enhanced-PPO
checkpoint could in principle transfer across — but for clean per-N curves
we train fresh per N (2 seeds each).

Output: results/exp3_scaling/{policy}__N{N}__s{seed}.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_common import (env_factory, train_ppo, save_summary,
                                     eval_heuristic)
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger

N_LIST = [5, 10, 15, 20, 25]
SEEDS = [0, 1]
PAD_N, PAD_M = 25, 2
EPISODES_EVAL = 50


def _iters_for(N: int) -> int:
    # More iters for larger N since the search space is bigger
    return {5: 30, 10: 50, 15: 70, 20: 90, 25: 110}[N]


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp3_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    for N in N_LIST:
        env_fn_eval = env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                                   use_relative_frame=True, include_time_features=True,
                                   potential_mode="tight")
        # Heuristics (no seed spread needed — they're deterministic modulo env rng)
        for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
            print(f"=== exp3 {h} N={N} ===", flush=True)
            s = eval_heuristic(h, env_fn_eval, episodes=EPISODES_EVAL, seed=999)
            save_summary(out_dir, f"{h}__N{N}",
                         {"exp": "scaling", "N": N, "policy": h, "seed": 0, **s})

        # PPO-enhanced: train fresh per N, per seed
        iters = _iters_for(N)
        for seed in SEEDS:
            tag = f"ppo_enhanced__N{N}__s{seed}"
            print(f"=== exp3 {tag} ===", flush=True)
            env_fn_tr = env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                                     use_relative_frame=True, include_time_features=True,
                                     potential_mode="tight")
            logger = RunLogger(str(out_dir / f"run_{tag}"),
                               {"exp": "scaling", "policy": "PPO-enhanced",
                                "N": N, "seed": seed})
            trainer = train_ppo(env_fn_tr, iters=iters, steps_per_iter=4096,
                                 ent_schedule="cosine", seed=seed, logger=logger)
            s, _ = evaluate_policy(env_fn_tr,
                                    lambda e, o, i: trainer.act(o, i),
                                    episodes=EPISODES_EVAL, seed=seed + 777)
            save_summary(out_dir, tag,
                         {"exp": "scaling", "policy": "PPO-enhanced",
                          "N": N, "seed": seed, "iters": iters, **s})


if __name__ == "__main__":
    main()
