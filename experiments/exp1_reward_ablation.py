"""Experiment 1 — Reward-function ablation at N=10.

Trains PPO with each of the 6 reward modes, 3 seeds each, 40 iters.
Shows which reward structure produces the best learner at small scale.

Output: results/exp1_reward_ablation/{reward}__s{seed}.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_common import (env_factory, train_ppo, save_summary,
                                     eval_heuristic)
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger

N = 10
REWARD_MODES = [
    "dist_normalized", "completion_ratio", "battery_aware",
    "time_pressure", "regret_based", "shaped_potential",
]
SEEDS = [0, 1, 2]
ITERS = 40
STEPS_PER_ITER = 2048
EPISODES_EVAL = 50


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp1_reward_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One heuristic baseline row per env setting (for easy plotting)
    env_fn = env_factory(N=N)
    for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
        s = eval_heuristic(h, env_fn, episodes=EPISODES_EVAL, seed=999)
        save_summary(out_dir, f"heuristic__{h}",
                     {"exp": "reward_ablation", "N": N, "policy": h,
                      "reward": "n/a", "seed": 0, **s})

    for reward in REWARD_MODES:
        pot_mode = "tight" if reward == "shaped_potential" else "simple"
        for seed in SEEDS:
            tag = f"{reward}__s{seed}"
            print(f"=== exp1 {tag} ===", flush=True)
            env_fn_r = env_factory(N=N, reward_mode=reward, potential_mode=pot_mode)
            logger = RunLogger(str(out_dir / f"run_{tag}"),
                               {"exp": "reward_ablation", "reward": reward,
                                "N": N, "seed": seed})
            trainer = train_ppo(env_fn_r, iters=ITERS, steps_per_iter=STEPS_PER_ITER,
                                seed=seed, logger=logger)
            eval_summary, _ = evaluate_policy(env_fn_r,
                                              lambda e, o, i: trainer.act(o, i),
                                              episodes=EPISODES_EVAL, seed=seed + 777)
            save_summary(out_dir, tag,
                         {"exp": "reward_ablation", "policy": "PPO",
                          "reward": reward, "N": N, "seed": seed,
                          "iters": ITERS, "steps_per_iter": STEPS_PER_ITER,
                          **eval_summary})


if __name__ == "__main__":
    main()
