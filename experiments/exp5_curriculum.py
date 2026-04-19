"""Experiment 5 — Curriculum ablation at N=25.

Compares three training regimes (all targetting N=25 eval):
  C0 from_scratch      : train 200 iters directly on N=25
  C1 single_curriculum : pretrain 60 iters on N=10 -> fine-tune 140 iters on N=25
  C2 double_curriculum : pretrain 40 on N=5 -> 50 on N=15 -> 110 on N=25

All PPO-enhanced (relative_frame + time_features + tight + cosine).
2 seeds each.

Output: results/exp5_curriculum/{regime}__s{seed}.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_common import (env_factory, train_ppo, save_summary,
                                     eval_heuristic)
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger

N_FINAL = 25
PAD_N, PAD_M = 25, 2
SEEDS = [0, 1]
EPISODES_EVAL = 50


def _env(N, **kwargs):
    return env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                        use_relative_frame=True, include_time_features=True,
                        potential_mode="tight", **kwargs)


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp5_curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic baselines at N=25
    env_fn_eval = _env(N_FINAL)
    for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
        s = eval_heuristic(h, env_fn_eval, episodes=EPISODES_EVAL, seed=999)
        save_summary(out_dir, f"heuristic__{h}",
                     {"exp": "curriculum", "regime": "heuristic",
                      "N": N_FINAL, "policy": h, "seed": 0, **s})

    for seed in SEEDS:
        # C0 from_scratch: 200 iters
        tag = f"C0_from_scratch__s{seed}"
        print(f"=== exp5 {tag} ===", flush=True)
        env_fn = _env(N_FINAL)
        logger = RunLogger(str(out_dir / f"run_{tag}"),
                            {"exp": "curriculum", "regime": "from_scratch",
                             "N": N_FINAL, "seed": seed})
        trainer = train_ppo(env_fn, iters=200, steps_per_iter=4096,
                             ent_schedule="cosine", seed=seed, logger=logger)
        s, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                                episodes=EPISODES_EVAL, seed=seed + 777)
        save_summary(out_dir, tag,
                      {"exp": "curriculum", "regime": "from_scratch",
                       "N": N_FINAL, "seed": seed, **s})

        # C1 single_curriculum
        tag = f"C1_single_curriculum__s{seed}"
        print(f"=== exp5 {tag} ===", flush=True)
        # Phase A: N=10
        env_a = _env(10)
        logger_a = RunLogger(str(out_dir / f"run_{tag}__phaseA_N10"),
                              {"exp": "curriculum", "regime": "single_curriculum",
                               "phase": "A_N10", "seed": seed})
        tr_a = train_ppo(env_a, iters=60, steps_per_iter=4096,
                          ent_coef=0.15, ent_coef_end=0.03,
                          ent_schedule="cosine", seed=seed, logger=logger_a)
        ckpt_a = out_dir / f"ckpt__{tag}__phaseA.pt"
        tr_a.save_checkpoint(str(ckpt_a))
        # Phase B: N=25
        env_b = _env(N_FINAL)
        logger_b = RunLogger(str(out_dir / f"run_{tag}__phaseB_N25"),
                              {"exp": "curriculum", "regime": "single_curriculum",
                               "phase": "B_N25", "seed": seed})
        tr_b = train_ppo(env_b, iters=140, steps_per_iter=4096,
                          lr=1e-4, ent_coef=0.08, ent_coef_end=0.01,
                          ent_schedule="cosine", seed=seed, logger=logger_b,
                          pretrain_ckpt=str(ckpt_a))
        s, _ = evaluate_policy(env_b, lambda e, o, i: tr_b.act(o, i),
                                episodes=EPISODES_EVAL, seed=seed + 777)
        save_summary(out_dir, tag,
                      {"exp": "curriculum", "regime": "single_curriculum",
                       "N": N_FINAL, "seed": seed, **s})

        # C2 double_curriculum
        tag = f"C2_double_curriculum__s{seed}"
        print(f"=== exp5 {tag} ===", flush=True)
        env_aa = _env(5)
        log_aa = RunLogger(str(out_dir / f"run_{tag}__phaseA_N5"),
                            {"exp": "curriculum", "regime": "double_curriculum",
                             "phase": "A_N5", "seed": seed})
        tr_aa = train_ppo(env_aa, iters=40, steps_per_iter=4096,
                            ent_coef=0.15, ent_coef_end=0.03,
                            ent_schedule="cosine", seed=seed, logger=log_aa)
        ckpt_aa = out_dir / f"ckpt__{tag}__phaseA.pt"
        tr_aa.save_checkpoint(str(ckpt_aa))

        env_bb = _env(15)
        log_bb = RunLogger(str(out_dir / f"run_{tag}__phaseB_N15"),
                            {"exp": "curriculum", "regime": "double_curriculum",
                             "phase": "B_N15", "seed": seed})
        tr_bb = train_ppo(env_bb, iters=50, steps_per_iter=4096,
                            lr=2e-4, ent_coef=0.10, ent_coef_end=0.02,
                            ent_schedule="cosine", seed=seed, logger=log_bb,
                            pretrain_ckpt=str(ckpt_aa))
        ckpt_bb = out_dir / f"ckpt__{tag}__phaseB.pt"
        tr_bb.save_checkpoint(str(ckpt_bb))

        env_cc = _env(N_FINAL)
        log_cc = RunLogger(str(out_dir / f"run_{tag}__phaseC_N25"),
                            {"exp": "curriculum", "regime": "double_curriculum",
                             "phase": "C_N25", "seed": seed})
        tr_cc = train_ppo(env_cc, iters=110, steps_per_iter=4096,
                            lr=1e-4, ent_coef=0.07, ent_coef_end=0.01,
                            ent_schedule="cosine", seed=seed, logger=log_cc,
                            pretrain_ckpt=str(ckpt_bb))
        s, _ = evaluate_policy(env_cc, lambda e, o, i: tr_cc.act(o, i),
                                episodes=EPISODES_EVAL, seed=seed + 777)
        save_summary(out_dir, tag,
                      {"exp": "curriculum", "regime": "double_curriculum",
                       "N": N_FINAL, "seed": seed, **s})


if __name__ == "__main__":
    main()
