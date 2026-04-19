"""Train the attention PPO on N=25 with all representation upgrades."""
from __future__ import annotations
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
from src.ppo_attn import PPOAttnTrainer
from src.heuristic import GreedyDeadlineBatteryHeuristic, evaluate_policy
from src.run_logger import RunLogger


def main():
    env_fn = lambda: make_env({
        "reward_mode": "shaped_potential",
        "num_customers": 25, "num_chargers": 1,
        "pad_num_customers": 25, "pad_num_chargers": 2,
        "use_relative_frame": True,
        "include_time_features": True,
        "potential_mode": "rich",
        "mission_time": 80, "deadline_min": 12, "deadline_max": 65,
    })

    print("==== PPOAttn on N=25 (relative-frame + time-features + rich potential + cosine entropy) ====")
    run_dir = ROOT / "results" / "attn_N25"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(str(run_dir), {"algo": "PPOAttn", "N": 25, "upgrades": "all4"})

    trainer = PPOAttnTrainer(
        env_fn=env_fn,
        d_model=128, n_heads=4,
        lr=3e-4, ent_coef=0.12, ent_coef_end=0.01, ent_schedule="cosine",
        steps_per_iter=2048, iters=200, minibatch_size=512, epochs=10,
        normalise_rewards=True, seed=0, logger=logger,
    )
    trainer.train(eval_every_iters=10, eval_episodes=10)

    eval_s, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                                 episodes=30, seed=999)
    heur = GreedyDeadlineBatteryHeuristic()
    heur_s, _ = evaluate_policy(env_fn, heur.act, episodes=30, seed=999)

    print("\n========= RESULT =========")
    print(f"PPOAttn  vs  Heuristic  at N=25 (shaped_potential):")
    print(f"  return:  {eval_s['mean_return']:7.1f}  vs  {heur_s['mean_return']:7.1f}")
    print(f"  served:  {eval_s['mean_customers_served']:7.2f}  vs  {heur_s['mean_customers_served']:7.2f}")
    print(f"  success: {eval_s['success_rate']*100:6.1f}%  vs  {heur_s['success_rate']*100:6.1f}%")

    logger.summary({**{"algo": "PPOAttn"}, **eval_s})


if __name__ == "__main__":
    main()
