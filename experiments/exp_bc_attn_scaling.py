"""Attention-PPO with NN-imitation warm-start.

Protocol per cell (N, seed):
  1. Collect 500 demos from NearestNeighbour heuristic on this env setting.
  2. Pretrain attention actor via cross-entropy on demos (10 epochs).
  3. Fine-tune with PPO for 150 iters × 4096 steps (600K env steps).
     Lower lr (1e-4), lower starting entropy (0.05) to preserve the prior.

Results go to `results/exp_bc_attn_scaling/`.

Hypothesis: imitation warm-start lifts the N=25 performance above vanilla
MLP-PPO by bypassing the early exploration dead-zone; if it still doesn't
beat NN, the ceiling is architectural + problem-structural, not training.
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
from src.heuristic import evaluate_policy, NearestNeighbourHeuristic
from src.ppo_attn import PPOAttnTrainer
from src.bc_warmstart import collect_demos, bc_pretrain
from src.run_logger import RunLogger

PAD_N, PAD_M = 25, 2
SEEDS = [0, 1]
EPISODES_EVAL = 50


def _iters_for(N: int) -> int:
    return {5: 100, 10: 130, 15: 150, 20: 150, 25: 150}[N]


def run_cell(N: int, seed: int):
    out_dir = Path(__file__).resolve().parents[1] / "results" / "exp_bc_attn_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_fn = env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                         use_relative_frame=True, include_time_features=True,
                         potential_mode="tight")
    iters = _iters_for(N)
    tag = f"bc_attn__N{N}__s{seed}"
    print(f"=== exp_bc_attn {tag} iters={iters} ===", flush=True)
    logger = RunLogger(str(out_dir / f"run_{tag}"),
                       {"exp": "bc_attn_scaling", "policy": "BC+PPO-Attn",
                        "N": N, "seed": seed, "iters": iters})

    trainer = PPOAttnTrainer(
        env_fn, d_model=128, n_heads=4,
        lr=1e-4,                            # lower lr for fine-tune-after-BC
        ent_coef=0.05, ent_coef_end=0.005,  # lower starting entropy
        ent_schedule="cosine",
        steps_per_iter=4096, iters=iters,
        minibatch_size=512, epochs=6,
        normalise_rewards=True, seed=seed, logger=logger,
    )

    print(f"  [1/3] collecting 500 demos from NN heuristic ...", flush=True)
    demos = collect_demos(env_fn, NearestNeighbourHeuristic(), n_episodes=500, seed=seed)
    print(f"  [1/3] collected {len(demos)} (obs, action) pairs", flush=True)

    print(f"  [2/3] BC pretrain 10 epochs ...", flush=True)
    bc_pretrain(trainer, demos, epochs=10, batch_size=256, lr=3e-4)

    print(f"  [3/3] PPO fine-tune {iters} iters ...", flush=True)
    trainer.train(eval_every_iters=max(iters // 5, 1), eval_episodes=10)

    s, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                            episodes=EPISODES_EVAL, seed=seed + 777)
    save_summary(out_dir, tag,
                 {"exp": "bc_attn_scaling", "policy": "BC+PPO-Attn",
                  "N": N, "seed": seed, "iters": iters, **s})
    if seed == 0:
        for h in ["greedy_db", "nearest_deadline", "nearest_neighbour", "random"]:
            heur_path = out_dir / f"{h}__N{N}.json"
            if heur_path.exists():
                continue
            hs = eval_heuristic(h, env_fn, episodes=EPISODES_EVAL, seed=999)
            save_summary(out_dir, f"{h}__N{N}",
                         {"exp": "bc_attn_scaling", "N": N, "policy": h, "seed": 0, **hs})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.N is None:
        for N in [5, 10, 15, 20, 25]:
            for seed in SEEDS:
                run_cell(N, seed)
    else:
        run_cell(args.N, args.seed)


if __name__ == "__main__":
    main()
