"""Parallel experiment runner — ProcessPoolExecutor over (reward, algo, seed).

Each worker trains one model, writes metrics.jsonl + summary.json under
results/{run_id}/. Run index appended to results/index.jsonl.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

# Force stdout/stderr to UTF-8 so unicode glyphs (→, δ, ×) don't crash Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configs import (REWARD_SETTINGS, ALGO_SETTINGS, DEFAULT_SEEDS,
                                   SCALING_REGIMES, SCALING_ALGO_SETTINGS)
from src.run_logger import RunLogger, new_run_id

RESULTS_DIR = ROOT / "results"
INDEX_PATH = RESULTS_DIR / "index.jsonl"


def _run_oracle(reward_key: str, reward_over: dict, seed: int, eval_episodes: int,
                 run_dir: Path, cfg_rec: dict) -> Dict[str, Any]:
    from src.env import make_env
    from src.oracle import evaluate_oracle
    logger = RunLogger(str(run_dir), cfg_rec)
    env_fn = lambda: make_env(reward_over)
    summary = evaluate_oracle(env_fn, episodes=eval_episodes, seed=seed)
    logger.log(iter=1, algo="Oracle", phase="eval",
               mean_return=summary["mean_return"],
               mean_served=summary["mean_customers_served"],
               success_rate=summary["success_rate"])
    out = {"algo": "Oracle", "reward": reward_key, "seed": seed, **summary}
    logger.summary(out)
    return out


def _run_heuristic(reward_key: str, reward_over: dict, seed: int, eval_episodes: int,
                    run_dir: Path, cfg_rec: dict) -> Dict[str, Any]:
    from src.env import make_env
    from src.heuristic import GreedyDeadlineBatteryHeuristic, evaluate_policy

    logger = RunLogger(str(run_dir), cfg_rec)
    env_fn = lambda: make_env(reward_over)
    heur = GreedyDeadlineBatteryHeuristic()
    # emit a single "iter 1" log line so the dashboard sees something
    summary, _ = evaluate_policy(env_fn, heur.act, episodes=eval_episodes, seed=seed)
    logger.log(iter=1, algo="Heuristic", phase="eval",
               mean_return=summary["mean_return"],
               mean_served=summary["mean_customers_served"],
               success_rate=summary["success_rate"])
    out = {"algo": "Heuristic", "reward": reward_key, "seed": seed, **summary}
    logger.summary(out)
    return out


def _run_ppo(reward_key: str, reward_over: dict, seed: int, algo_cfg: dict,
              run_dir: Path, cfg_rec: dict) -> Dict[str, Any]:
    from src.env import make_env
    from src.ppo import PPOTrainer
    from src.heuristic import evaluate_policy

    logger = RunLogger(str(run_dir), cfg_rec)
    env_fn = lambda: make_env(reward_over)
    cfg = {k: v for k, v in algo_cfg.items() if k != "kind"}
    trainer = PPOTrainer(env_fn, seed=seed, logger=logger, **cfg).train(
        eval_every_iters=5, eval_episodes=5)
    summary, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                                  episodes=50, seed=seed + 9999)
    out = {"algo": "PPO", "reward": reward_key, "seed": seed, **summary}
    logger.summary(out)
    return out


def _run_a2c(reward_key: str, reward_over: dict, seed: int, algo_cfg: dict,
              run_dir: Path, cfg_rec: dict) -> Dict[str, Any]:
    from src.env import make_env
    from src.a2c import A2CTrainer
    from src.heuristic import evaluate_policy

    logger = RunLogger(str(run_dir), cfg_rec)
    env_fn = lambda: make_env(reward_over)
    cfg = {k: v for k, v in algo_cfg.items() if k != "kind"}
    trainer = A2CTrainer(env_fn, seed=seed, logger=logger, **cfg).train(
        eval_every_iters=20, eval_episodes=5)
    summary, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                                  episodes=50, seed=seed + 9999)
    out = {"algo": "A2C", "reward": reward_key, "seed": seed, **summary}
    logger.summary(out)
    return out


def _run_rppo(reward_key: str, reward_over: dict, seed: int, algo_cfg: dict,
               run_dir: Path, cfg_rec: dict) -> Dict[str, Any]:
    from src.env import make_env
    from src.rppo import RecurrentPPOTrainer
    import numpy as np

    logger = RunLogger(str(run_dir), cfg_rec)
    env_fn = lambda: make_env(reward_over)
    cfg = {k: v for k, v in algo_cfg.items() if k != "kind"}
    trainer = RecurrentPPOTrainer(env_fn, seed=seed, logger=logger, **cfg).train(
        eval_every_iters=5, eval_episodes=5)

    # RecurrentPPO needs hidden-state management for eval — do it inline
    ep_rets, ep_served, ep_completed = [], [], []
    for ep in range(50):
        e = env_fn()
        o, inf = e.reset(seed=seed + 9999 + ep)
        trainer._reset_eval_hidden()
        done, ret = False, 0.0
        while not done:
            a = trainer.act(o, inf)
            o, r, term, trunc, inf = e.step(a)
            done = bool(term or trunc)
            ret += float(r)
        ep_rets.append(ret)
        ep_served.append(int(inf.get("customers_served", 0)))
        ep_completed.append(int(inf.get("completed", False)))
    summary = {
        "mean_return": float(np.mean(ep_rets)),
        "std_return": float(np.std(ep_rets, ddof=1)) if len(ep_rets) > 1 else 0.0,
        "success_rate": float(np.mean(ep_completed)),
        "mean_customers_served": float(np.mean(ep_served)),
        "episodes": 50,
    }
    out = {"algo": "RecurrentPPO", "reward": reward_key, "seed": seed, **summary}
    logger.summary(out)
    return out


def _worker(job) -> Dict[str, Any]:
    """Top-level worker fn — must be picklable. Returns summary dict.

    job is either (reward_key, algo_name, seed, run_id) — legacy form — or
    (reward_key, algo_name, seed, run_id, regime_key) — scaling form with a
    regime override merged into reward_over.
    """
    import torch
    torch.set_num_threads(1)

    if len(job) == 5:
        reward_key, algo_name, seed, run_id, regime_key = job
    else:
        reward_key, algo_name, seed, run_id = job
        regime_key = None
    reward_over = dict(REWARD_SETTINGS[reward_key])
    if regime_key and regime_key in SCALING_REGIMES:
        reward_over.update(SCALING_REGIMES[regime_key])
    # When running a scaling regime, use the longer-training algo presets.
    algo_table = SCALING_ALGO_SETTINGS if regime_key else ALGO_SETTINGS
    algo_cfg = algo_table[algo_name]

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_rec = {
        "run_id": run_id,
        "reward_key": reward_key, "reward_over": reward_over,
        "algo": algo_name, "algo_cfg": algo_cfg, "seed": seed,
        "regime": regime_key,
    }

    try:
        if algo_cfg["kind"] == "heuristic":
            return _run_heuristic(reward_key, reward_over, seed,
                                   algo_cfg.get("eval_episodes", 50), run_dir, cfg_rec)
        if algo_cfg["kind"] == "ppo":
            return _run_ppo(reward_key, reward_over, seed, algo_cfg, run_dir, cfg_rec)
        if algo_cfg["kind"] == "a2c":
            return _run_a2c(reward_key, reward_over, seed, algo_cfg, run_dir, cfg_rec)
        if algo_cfg["kind"] == "rppo":
            return _run_rppo(reward_key, reward_over, seed, algo_cfg, run_dir, cfg_rec)
        if algo_cfg["kind"] == "oracle":
            return _run_oracle(reward_key, reward_over, seed,
                                algo_cfg.get("eval_episodes", 30), run_dir, cfg_rec)
    except Exception:
        err = traceback.format_exc()
        RunLogger(str(run_dir), cfg_rec).fail(err)
        return {"run_id": run_id, "algo": algo_name, "reward": reward_key,
                "seed": seed, "status": "failed", "error": err}


def _append_index(rec: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def build_jobs(rewards, algos, seeds, regimes=None) -> list:
    jobs = []
    regs = regimes or [None]
    for reg in regs:
        for r in rewards:
            for a in algos:
                for s in seeds:
                    reg_tag = f"__{reg}" if reg else ""
                    run_id = f"{r}__{a}{reg_tag}__s{s}__{new_run_id()[-8:]}"
                    jobs.append((r, a, s, run_id, reg))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", nargs="+", default=list(REWARD_SETTINGS.keys()))
    ap.add_argument("--algo", nargs="+", default=list(ALGO_SETTINGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--regime", nargs="+", default=None,
                     help="one or more keys from SCALING_REGIMES; omit for base config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    # validate
    for r in args.reward:
        assert r in REWARD_SETTINGS, f"unknown reward {r}"
    for a in args.algo:
        assert a in ALGO_SETTINGS, f"unknown algo {a}"
    if args.regime:
        for g in args.regime:
            assert g in SCALING_REGIMES, f"unknown regime {g}"

    jobs = build_jobs(args.reward, args.algo, args.seeds, args.regime)
    print(f"[runner] scheduling {len(jobs)} jobs across {args.workers} workers")
    for j in jobs:
        print("  ", j[3])
    # register scheduled status line per job so UI sees them immediately
    for j in jobs:
        rec = {"run_id": j[3], "reward": j[0], "algo": j[1], "seed": j[2],
                "status": "scheduled"}
        if len(j) >= 5 and j[4]:
            rec["regime"] = j[4]
        _append_index(rec)

    done_n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"run_id": j[3], "reward": j[0], "algo": j[1], "seed": j[2],
                       "status": "failed", "error": str(e)}
            rec.setdefault("status", "done")
            rec.setdefault("run_id", j[3])
            if len(j) >= 5 and j[4]:
                rec.setdefault("regime", j[4])
            _append_index(rec)
            done_n += 1
            print(f"[runner] {done_n}/{len(jobs)}  {j[3]}  "
                  f"return={rec.get('mean_return','?')}  "
                  f"success={rec.get('success_rate','?')}  "
                  f"status={rec['status']}")

    print("[runner] all done")


if __name__ == "__main__":
    main()
