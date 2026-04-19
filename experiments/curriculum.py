"""Curriculum trainer: pretrain on N=10, fine-tune on N=25 (both padded to 25).

Usage:
    python -m experiments.curriculum
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Force stdout/stderr to UTF-8 so unicode glyphs (→, δ, ×) don't crash Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env import make_env
from src.ppo import PPOTrainer
from src.heuristic import GreedyDeadlineBatteryHeuristic, evaluate_policy
from src.run_logger import RunLogger


PAD_N, PAD_M = 25, 2
HIDDEN, NH = 512, 3


def make_padded_env(N: int, mission_time: float, dmin: float, dmax: float,
                     reward_mode: str = "shaped_potential"):
    return lambda: make_env({
        "reward_mode": reward_mode,
        "num_customers": N, "num_chargers": 1,
        "pad_num_customers": PAD_N, "pad_num_chargers": PAD_M,
        "mission_time": mission_time,
        "deadline_min": dmin, "deadline_max": dmax,
    })


def main():
    print("==== PHASE 1: pretrain on N=10 (padded to 25) ====")
    env_fn_10 = make_padded_env(N=10, mission_time=30, dmin=5, dmax=25)

    run_dir_p1 = ROOT / "results" / "curriculum_phase1_N10"
    run_dir_p1.mkdir(parents=True, exist_ok=True)
    logger1 = RunLogger(str(run_dir_p1), {"phase": "pretrain_N10", "pad": [PAD_N, PAD_M]})

    trainer = PPOTrainer(
        env_fn=env_fn_10,
        hidden_dim=HIDDEN, n_hidden=NH,
        lr=3e-4, ent_coef=0.12, ent_coef_end=0.02,
        steps_per_iter=1024, iters=60, minibatch_size=256, epochs=8,
        normalise_rewards=True, include_mask_in_obs=True,
        seed=0, logger=logger1,
    )
    trainer.train(eval_every_iters=10, eval_episodes=10)

    # Evaluate phase-1 on N=10
    eval_p1, _ = evaluate_policy(env_fn_10, lambda e, o, i: trainer.act(o, i),
                                  episodes=30, seed=777)
    print(f"[Phase 1] N=10 eval: return={eval_p1['mean_return']:.1f} "
           f"served={eval_p1['mean_customers_served']:.2f} "
           f"success={eval_p1['success_rate']*100:.0f}%")
    ckpt_path = run_dir_p1 / "phase1.pt"
    trainer.save_checkpoint(str(ckpt_path))
    logger1.summary({**{"phase": "pretrain_N10"}, **eval_p1})

    # Compare to heuristic on N=10
    heur = GreedyDeadlineBatteryHeuristic()
    heur_p1, _ = evaluate_policy(env_fn_10, heur.act, episodes=30, seed=777)
    print(f"[Phase 1] Heuristic N=10: return={heur_p1['mean_return']:.1f} "
           f"served={heur_p1['mean_customers_served']:.2f} "
           f"success={heur_p1['success_rate']*100:.0f}%")

    print("\n==== PHASE 2: fine-tune on N=25 (same padding) ====")
    env_fn_25 = make_padded_env(N=25, mission_time=80, dmin=12, dmax=65)

    run_dir_p2 = ROOT / "results" / "curriculum_phase2_N25"
    run_dir_p2.mkdir(parents=True, exist_ok=True)
    logger2 = RunLogger(str(run_dir_p2), {"phase": "finetune_N25",
                                             "pretrained_from": str(ckpt_path)})

    trainer2 = PPOTrainer(
        env_fn=env_fn_25,
        hidden_dim=HIDDEN, n_hidden=NH,
        lr=1e-4,                           # lower LR for fine-tuning
        ent_coef=0.08, ent_coef_end=0.01,  # slightly lower start (already has prior)
        steps_per_iter=1024, iters=120, minibatch_size=256, epochs=8,
        normalise_rewards=True, include_mask_in_obs=True,
        seed=0, logger=logger2,
    )
    # Load phase-1 weights. Need obs_dim and n_actions from the NEW env — they
    # should match since padding is the same.
    env_probe = env_fn_25()
    obs_probe, _ = env_probe.reset(seed=0)
    from src.obs import obs_to_vector
    obs_dim = obs_to_vector(obs_probe, include_mask=True).shape[0]
    n_actions = env_probe.action_space.n
    trainer2.load_checkpoint(str(ckpt_path), obs_dim=obs_dim, n_actions=n_actions)
    print(f"[Phase 2] loaded phase-1 weights (obs_dim={obs_dim}, n_actions={n_actions})")

    trainer2.train(eval_every_iters=10, eval_episodes=10)

    eval_p2, _ = evaluate_policy(env_fn_25, lambda e, o, i: trainer2.act(o, i),
                                  episodes=30, seed=999)
    print(f"\n[Phase 2] N=25 eval: return={eval_p2['mean_return']:.1f} "
           f"served={eval_p2['mean_customers_served']:.2f}/25 "
           f"success={eval_p2['success_rate']*100:.0f}%")
    trainer2.save_checkpoint(str(run_dir_p2 / "phase2.pt"))
    logger2.summary({**{"phase": "finetune_N25"}, **eval_p2})

    heur_p2, _ = evaluate_policy(env_fn_25, heur.act, episodes=30, seed=999)
    print(f"[Phase 2] Heuristic N=25: return={heur_p2['mean_return']:.1f} "
           f"served={heur_p2['mean_customers_served']:.2f}/25 "
           f"success={heur_p2['success_rate']*100:.0f}%")

    # Gap analysis
    gap_return = (eval_p2["mean_return"] - heur_p2["mean_return"]) / heur_p2["mean_return"] * 100
    gap_served = eval_p2["mean_customers_served"] - heur_p2["mean_customers_served"]
    print(f"\n========= RESULT =========")
    print(f"PPO-curriculum  vs  Heuristic  at N=25 (shaped_potential):")
    print(f"  return:  {eval_p2['mean_return']:7.1f}  vs  {heur_p2['mean_return']:7.1f}   gap {gap_return:+.1f}%")
    print(f"  served:  {eval_p2['mean_customers_served']:7.2f}  vs  {heur_p2['mean_customers_served']:7.2f}  delta {gap_served:+.2f}")
    print(f"  success: {eval_p2['success_rate']*100:6.1f}%  vs  {heur_p2['success_rate']*100:6.1f}%")


if __name__ == "__main__":
    main()
