"""Shared helpers for the comprehensive experiment suite (exp1-exp5).

Provides:
 - `train_ppo`: thin wrapper around PPOTrainer with sane defaults
 - `eval_and_save`: runs evaluate_policy, strips numpy types, writes JSON
 - `save_summary`: persist one entry's full KPI dict + metadata to disk
 - `env_factory`: build env-factory lambdas with a common signature
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.env import make_env
from src.heuristic import (GreedyDeadlineBatteryHeuristic, NearestDeadlineFirstHeuristic,
                           NearestNeighbourHeuristic, RandomPolicy, evaluate_policy)
from src.obs import obs_to_vector
from src.ppo import PPOTrainer
from src.run_logger import RunLogger


# Per-N default problem settings (mission_time, deadline window) — tuned so the
# heuristic hits 30-80% success, giving headroom for RL to improve.
N_SETTINGS: Dict[int, Dict[str, float]] = {
    5:  dict(mission_time=15, deadline_min=3,  deadline_max=12),
    10: dict(mission_time=30, deadline_min=5,  deadline_max=25),
    15: dict(mission_time=50, deadline_min=8,  deadline_max=40),
    20: dict(mission_time=65, deadline_min=10, deadline_max=55),
    25: dict(mission_time=80, deadline_min=12, deadline_max=65),
}


def env_factory(N: int, num_chargers: int = 1,
                reward_mode: str = "shaped_potential",
                use_relative_frame: bool = False,
                include_time_features: bool = False,
                potential_mode: str = "rich",
                pad_num_customers: Optional[int] = None,
                pad_num_chargers: Optional[int] = None):
    """Return a lambda that builds a fresh env with these settings."""
    if N not in N_SETTINGS:
        # Interpolate for non-standard N
        mt = 15 + (N - 5) * 3
        dmin = 3 + (N - 5) * 0.5
        dmax = 12 + (N - 5) * 2.5
    else:
        mt = N_SETTINGS[N]["mission_time"]
        dmin = N_SETTINGS[N]["deadline_min"]
        dmax = N_SETTINGS[N]["deadline_max"]

    cfg = {
        "reward_mode": reward_mode,
        "num_customers": N, "num_chargers": num_chargers,
        "mission_time": mt, "deadline_min": dmin, "deadline_max": dmax,
        "use_relative_frame": use_relative_frame,
        "include_time_features": include_time_features,
        "potential_mode": potential_mode,
    }
    if pad_num_customers is not None:
        cfg["pad_num_customers"] = pad_num_customers
    if pad_num_chargers is not None:
        cfg["pad_num_chargers"] = pad_num_chargers
    return lambda: make_env(cfg)


def train_ppo(env_fn, *, iters: int = 60, steps_per_iter: int = 2048,
              hidden_dim: int = 256, n_hidden: int = 2,
              lr: float = 3e-4, ent_coef: float = 0.12, ent_coef_end: float = 0.02,
              ent_schedule: str = "cosine", seed: int = 0,
              normalise_rewards: bool = True, logger=None,
              pretrain_ckpt: Optional[str] = None) -> PPOTrainer:
    """Train a PPO. If pretrain_ckpt given, warm-starts from that checkpoint."""
    trainer = PPOTrainer(
        env_fn=env_fn, hidden_dim=hidden_dim, n_hidden=n_hidden,
        lr=lr, ent_coef=ent_coef, ent_coef_end=ent_coef_end,
        ent_schedule=ent_schedule,
        steps_per_iter=steps_per_iter, iters=iters, minibatch_size=min(steps_per_iter // 4, 512),
        epochs=8, normalise_rewards=normalise_rewards,
        include_mask_in_obs=True, seed=seed, logger=logger,
    )
    if pretrain_ckpt:
        env_probe = env_fn()
        obs_probe, _ = env_probe.reset(seed=0)
        obs_dim = obs_to_vector(obs_probe, include_mask=True).shape[0]
        n_actions = env_probe.action_space.n
        trainer.load_checkpoint(pretrain_ckpt, obs_dim=obs_dim, n_actions=n_actions)
    trainer.train(eval_every_iters=max(iters // 6, 1), eval_episodes=10)
    return trainer


def _clean(obj: Any) -> Any:
    """Strip numpy / python non-JSON types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    return obj


def save_summary(out_dir: Path, tag: str, record: Dict[str, Any]) -> None:
    """Persist a single experiment-row record as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = _clean(record)
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def get_heuristic(name: str):
    table = {
        "greedy_db": GreedyDeadlineBatteryHeuristic(),
        "nearest_deadline": NearestDeadlineFirstHeuristic(),
        "nearest_neighbour": NearestNeighbourHeuristic(),
        "random": RandomPolicy(seed=0),
    }
    return table[name]


def eval_heuristic(name: str, env_fn, episodes: int = 50, seed: int = 999):
    pol = get_heuristic(name)
    summary, _ = evaluate_policy(env_fn, pol.act, episodes=episodes, seed=seed)
    return summary
