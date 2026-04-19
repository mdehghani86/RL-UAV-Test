"""Experiment 2 — Representation / shaping ablation at N=25.

Starts from a baseline (no upgrades) and turns on one upgrade at a time:
  V0 baseline          : absolute-frame, rich potential, linear ent schedule
  V1 +relative_frame   : positions as offsets from current node
  V2 +time_features    : per-customer time_to_reach, deadline_slack
  V3 +tight_potential  : deadline-feasibility-aware shaping
  V4 +cosine_entropy   : cosine anneal with mid-training bump
  V5 ALL (= enhanced)  : all four combined

3 seeds each, shaped_potential reward mode, 80 iters.

Output: results/exp2_representation/{variant}__s{seed}.json
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
ITERS = 80
STEPS_PER_ITER = 4096
EPISODES_EVAL = 50


VARIANTS = {
    "V0_baseline":        dict(use_relative_frame=False, include_time_features=False,
                                potential_mode="rich", ent_schedule="linear"),
    "V1_relative_frame":  dict(use_relative_frame=True,  include_time_features=False,
                                potential_mode="rich", ent_schedule="linear"),
    "V2_time_features":   dict(use_relative_frame=False, include_time_features=True,
                                potential_mode="rich", ent_schedule="linear"),
    "V3_tight_potential": dict(use_relative_frame=False, include_time_features=False,
                                potential_mode="tight", ent_schedule="linear"),
    "V4_cosine_entropy":  dict(use_relative_frame=False, include_time_features=False,
                                potential_mode="rich", ent_schedule="cosine"),
    "V5_all":             dict(use_relative_frame=True,  include_time_features=True,
                                potential_mode="tight", ent_schedule="cosine"),
}


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp2_representation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic baselines on the same env
    env_fn = env_factory(N=N)
    for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
        s = eval_heuristic(h, env_fn, episodes=EPISODES_EVAL, seed=999)
        save_summary(out_dir, f"heuristic__{h}",
                     {"exp": "representation", "N": N, "policy": h,
                      "variant": "heuristic", "seed": 0, **s})

    for variant, opts in VARIANTS.items():
        ent_schedule = opts.pop("ent_schedule")
        for seed in SEEDS:
            tag = f"{variant}__s{seed}"
            print(f"=== exp2 {tag} ===", flush=True)
            env_fn_v = env_factory(N=N, **opts)
            logger = RunLogger(str(out_dir / f"run_{tag}"),
                               {"exp": "representation", "variant": variant,
                                "N": N, "seed": seed})
            trainer = train_ppo(env_fn_v, iters=ITERS, steps_per_iter=STEPS_PER_ITER,
                                ent_schedule=ent_schedule, seed=seed, logger=logger)
            eval_summary, _ = evaluate_policy(env_fn_v,
                                              lambda e, o, i: trainer.act(o, i),
                                              episodes=EPISODES_EVAL, seed=seed + 777)
            save_summary(out_dir, tag,
                         {"exp": "representation", "policy": "PPO",
                          "variant": variant, "N": N, "seed": seed,
                          "iters": ITERS, "steps_per_iter": STEPS_PER_ITER,
                          **opts, "ent_schedule": ent_schedule,
                          **eval_summary})


if __name__ == "__main__":
    main()
