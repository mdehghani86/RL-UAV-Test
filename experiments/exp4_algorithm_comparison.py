"""Experiment 4 — Algorithm comparison at N=25.

Head-to-head:
  PPO-vanilla     : original MLP-PPO, shaped_potential, no repr upgrades
  PPO-enhanced    : + relative_frame + time_features + tight + cosine
  A2C             : synchronous advantage actor-critic
  RPPO            : recurrent PPO (LSTM over obs sequence)
  Heuristic-Greedy: deadline + distance + battery
  Heuristic-NDF   : nearest-deadline-first
  Heuristic-NN    : nearest-neighbour
  Random          : uniform-random feasible action

3 seeds per RL method, 50 eval eps. One JSON per (method, seed).

Output: results/exp4_algorithm/{method}__s{seed}.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_common import (env_factory, train_ppo, save_summary,
                                     eval_heuristic)
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger

N = 25
SEEDS = [0, 1, 2]
ITERS = 100
STEPS_PER_ITER = 4096
EPISODES_EVAL = 50


def _train_a2c(env_fn, iters, seed, logger):
    from src.a2c import A2CTrainer
    trainer = A2CTrainer(env_fn=env_fn, hidden_dim=256, n_hidden=2,
                          lr=3e-4, ent_coef=0.05,
                          steps_per_iter=STEPS_PER_ITER, iters=iters,
                          normalise_rewards=True, include_mask_in_obs=True,
                          seed=seed, logger=logger)
    trainer.train(eval_every_iters=max(iters // 6, 1), eval_episodes=10)
    return trainer


def _train_rppo(env_fn, iters, seed, logger):
    from src.rppo import RPPOTrainer
    trainer = RPPOTrainer(env_fn=env_fn, hidden_dim=256,
                           lr=3e-4, ent_coef=0.1, ent_coef_end=0.02,
                           steps_per_iter=STEPS_PER_ITER, iters=iters,
                           epochs=4, normalise_rewards=True,
                           seed=seed, logger=logger)
    trainer.train(eval_every_iters=max(iters // 6, 1), eval_episodes=10)
    return trainer


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp4_algorithm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heuristics
    env_fn_vanilla = env_factory(N=N)
    for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
        s = eval_heuristic(h, env_fn_vanilla, episodes=EPISODES_EVAL, seed=999)
        save_summary(out_dir, f"heuristic__{h}",
                     {"exp": "algorithm", "N": N, "policy": h, "method": h,
                      "seed": 0, **s})

    # RL methods
    plan = [
        ("PPO-vanilla",  lambda fn, it, s, lg: train_ppo(fn, iters=it, steps_per_iter=STEPS_PER_ITER,
                                                          ent_schedule="linear", seed=s, logger=lg),
         env_factory(N=N)),
        ("PPO-enhanced", lambda fn, it, s, lg: train_ppo(fn, iters=it, steps_per_iter=STEPS_PER_ITER,
                                                          ent_schedule="cosine", seed=s, logger=lg),
         env_factory(N=N, use_relative_frame=True, include_time_features=True,
                      potential_mode="tight")),
        ("A2C",           _train_a2c,  env_factory(N=N)),
        ("RPPO",          _train_rppo, env_factory(N=N)),
    ]

    for method, train_fn, env_fn in plan:
        for seed in SEEDS:
            tag = f"{method}__s{seed}"
            print(f"=== exp4 {tag} ===", flush=True)
            logger = RunLogger(str(out_dir / f"run_{tag}"),
                               {"exp": "algorithm", "method": method,
                                "N": N, "seed": seed})
            try:
                trainer = train_fn(env_fn, ITERS, seed, logger)
                s, _ = evaluate_policy(env_fn,
                                        lambda e, o, i: trainer.act(o, i),
                                        episodes=EPISODES_EVAL, seed=seed + 777)
                save_summary(out_dir, tag,
                             {"exp": "algorithm", "policy": method,
                              "method": method, "N": N, "seed": seed,
                              "iters": ITERS, **s})
            except Exception as e:
                print(f"  ! {method} seed {seed} failed: {e}", flush=True)
                save_summary(out_dir, tag,
                             {"exp": "algorithm", "policy": method,
                              "method": method, "N": N, "seed": seed,
                              "failed": str(e)})


if __name__ == "__main__":
    main()
