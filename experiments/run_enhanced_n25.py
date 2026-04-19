"""Enhanced N=25 curriculum: relative-frame obs + time features + tight potential
+ cosine entropy + longer rollouts.

Pretrains on N=10 (padded to 25), fine-tunes on N=25. All representation and
exploration upgrades enabled. Compares to heuristic baseline.

Usage:
    python -m experiments.run_enhanced_n25
"""
from __future__ import annotations
import sys
from pathlib import Path

# Force stdout/stderr to UTF-8 so unicode glyphs (->, delta, x) don't crash Windows cp1252.
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
from src.obs import obs_to_vector


PAD_N, PAD_M = 25, 2
HIDDEN, NH = 512, 3


def make_enhanced_env(N: int, mission_time: float, dmin: float, dmax: float):
    """Env factory with all representation + shaping upgrades enabled."""
    return lambda: make_env({
        "reward_mode": "shaped_potential",
        "num_customers": N, "num_chargers": 1,
        "pad_num_customers": PAD_N, "pad_num_chargers": PAD_M,
        "mission_time": mission_time,
        "deadline_min": dmin, "deadline_max": dmax,
        # Representation upgrades
        "use_relative_frame": True,
        "include_time_features": True,
        # Reward shaping upgrade
        "potential_mode": "tight",
    })


def main():
    print("==== PHASE 1: pretrain on N=10 (enhanced, padded to 25) ====")
    env_fn_10 = make_enhanced_env(N=10, mission_time=30, dmin=5, dmax=25)

    run_dir_p1 = ROOT / "results" / "enhanced_phase1_N10"
    run_dir_p1.mkdir(parents=True, exist_ok=True)
    logger1 = RunLogger(str(run_dir_p1), {
        "phase": "pretrain_N10",
        "pad": [PAD_N, PAD_M],
        "upgrades": ["relative_frame", "time_features", "tight_potential", "cosine_entropy"],
    })

    trainer = PPOTrainer(
        env_fn=env_fn_10,
        hidden_dim=HIDDEN, n_hidden=NH,
        lr=3e-4, ent_coef=0.12, ent_coef_end=0.02,
        ent_schedule="cosine",                 # NEW: cosine anneal
        steps_per_iter=4096, iters=60,          # NEW: longer rollouts
        minibatch_size=512, epochs=8,
        normalise_rewards=True, include_mask_in_obs=True,
        seed=0, logger=logger1,
    )
    trainer.train(eval_every_iters=10, eval_episodes=10)

    eval_p1, _ = evaluate_policy(env_fn_10, lambda e, o, i: trainer.act(o, i),
                                  episodes=30, seed=777)
    print(f"[Phase 1] N=10 eval: return={eval_p1['mean_return']:.1f} "
          f"served={eval_p1['mean_customers_served']:.2f} "
          f"success={eval_p1['success_rate']*100:.0f}%")
    ckpt_path = run_dir_p1 / "phase1.pt"
    trainer.save_checkpoint(str(ckpt_path))
    logger1.summary({"phase": "pretrain_N10", **eval_p1})

    heur = GreedyDeadlineBatteryHeuristic()
    heur_p1, _ = evaluate_policy(env_fn_10, heur.act, episodes=30, seed=777)
    print(f"[Phase 1] Heuristic N=10: return={heur_p1['mean_return']:.1f} "
          f"served={heur_p1['mean_customers_served']:.2f} "
          f"success={heur_p1['success_rate']*100:.0f}%")

    print("\n==== PHASE 2: fine-tune on N=25 (same enhancements) ====")
    env_fn_25 = make_enhanced_env(N=25, mission_time=80, dmin=12, dmax=65)

    run_dir_p2 = ROOT / "results" / "enhanced_phase2_N25"
    run_dir_p2.mkdir(parents=True, exist_ok=True)
    logger2 = RunLogger(str(run_dir_p2), {
        "phase": "finetune_N25",
        "pretrained_from": str(ckpt_path),
        "upgrades": ["relative_frame", "time_features", "tight_potential", "cosine_entropy"],
    })

    trainer2 = PPOTrainer(
        env_fn=env_fn_25,
        hidden_dim=HIDDEN, n_hidden=NH,
        lr=1e-4,
        ent_coef=0.08, ent_coef_end=0.01,
        ent_schedule="cosine",
        steps_per_iter=4096, iters=120,
        minibatch_size=512, epochs=8,
        normalise_rewards=True, include_mask_in_obs=True,
        seed=0, logger=logger2,
    )
    env_probe = env_fn_25()
    obs_probe, _ = env_probe.reset(seed=0)
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
    logger2.summary({"phase": "finetune_N25", **eval_p2})

    heur_p2, _ = evaluate_policy(env_fn_25, heur.act, episodes=30, seed=999)
    print(f"[Phase 2] Heuristic N=25: return={heur_p2['mean_return']:.1f} "
          f"served={heur_p2['mean_customers_served']:.2f}/25 "
          f"success={heur_p2['success_rate']*100:.0f}%")

    # Gap analysis
    gap_return = (eval_p2["mean_return"] - heur_p2["mean_return"]) / (abs(heur_p2["mean_return"]) + 1e-9) * 100
    gap_served = eval_p2["mean_customers_served"] - heur_p2["mean_customers_served"]
    print(f"\n========= RESULT =========")
    print(f"Enhanced PPO  vs  Heuristic  at N=25 (tight potential + rel frame + time feats + cosine ent):")
    print(f"  return:  {eval_p2['mean_return']:7.1f}  vs  {heur_p2['mean_return']:7.1f}   gap {gap_return:+.1f}%")
    print(f"  served:  {eval_p2['mean_customers_served']:7.2f}  vs  {heur_p2['mean_customers_served']:7.2f}  delta {gap_served:+.2f}")
    print(f"  success: {eval_p2['success_rate']*100:6.1f}%  vs  {heur_p2['success_rate']*100:6.1f}%")


if __name__ == "__main__":
    main()
