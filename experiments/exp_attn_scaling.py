"""Attention-PPO scaling experiment.

Same N sweep as exp3 (N in {5, 10, 15, 20, 25}) but using the attention-based
actor/critic (Kool 2019-inspired) instead of MLP. Training budget ~1M steps
per run, matching exp3_scaling_1M.

Results go to `results/exp_attn_scaling/` so they sit alongside both the
MLP-PPO undertrained results (results/exp3_scaling/) and MLP-PPO 1M results
(results/exp3_scaling_1M/) for a three-way comparison in the paper:
  MLP-undertrained  vs  MLP-1M  vs  Attention-1M  vs  heuristics.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiments.exp_common import env_factory, save_summary, eval_heuristic
from src.heuristic import evaluate_policy
from src.ppo_attn import PPOAttnTrainer
from src.run_logger import RunLogger

N_LIST = [5, 10, 15, 20, 25]
SEEDS = [0, 1]
PAD_N, PAD_M = 25, 2
EPISODES_EVAL = 50


def _iters_for(N: int) -> int:
    # ~1M steps per run, scaling slightly with N
    return {5: 150, 10: 200, 15: 230, 20: 240, 25: 250}[N]


def run_cell(N: int, seed: int):
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp_attn_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_fn = env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                         use_relative_frame=True, include_time_features=True,
                         potential_mode="tight")
    iters = _iters_for(N)
    tag = f"ppo_attn__N{N}__s{seed}"
    print(f"=== exp_attn {tag} iters={iters} ===", flush=True)
    logger = RunLogger(str(out_dir / f"run_{tag}"),
                       {"exp": "attn_scaling", "policy": "PPO-Attn",
                        "N": N, "seed": seed, "iters": iters})
    trainer = PPOAttnTrainer(
        env_fn, d_model=128, n_heads=4,
        lr=3e-4, ent_coef=0.10, ent_coef_end=0.01, ent_schedule="cosine",
        steps_per_iter=4096, iters=iters,
        minibatch_size=512, epochs=6,
        normalise_rewards=True, seed=seed, logger=logger,
    )
    trainer.train(eval_every_iters=max(iters // 5, 1), eval_episodes=10)
    s, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                            episodes=EPISODES_EVAL, seed=seed + 777)
    save_summary(out_dir, tag,
                 {"exp": "attn_scaling", "policy": "PPO-Attn",
                  "N": N, "seed": seed, "iters": iters, **s})
    # Snap heuristics for this N ONLY on the seed-0 run to avoid wasting
    # eval episodes + overwriting seed-independent JSONs on subsequent seeds.
    if seed == 0:
        for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
            heur_path = out_dir / f"{h}__N{N}.json"
            if heur_path.exists():
                continue
            hs = eval_heuristic(h, env_fn, episodes=EPISODES_EVAL, seed=999)
            save_summary(out_dir, f"{h}__N{N}",
                         {"exp": "attn_scaling", "N": N, "policy": h, "seed": 0, **hs})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.N is None:
        for N in N_LIST:
            for seed in SEEDS:
                run_cell(N, seed)
    else:
        run_cell(args.N, args.seed)


if __name__ == "__main__":
    main()
