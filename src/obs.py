"""Observation flattener. Optionally includes action_mask in the vector
so the policy gets an explicit feasibility signal (fix #2)."""
from __future__ import annotations
import numpy as np


def obs_to_vector(obs: dict, include_mask: bool = True) -> np.ndarray:
    parts = [
        obs["agent"].astype(np.float32).ravel(),
        obs["visited"].astype(np.float32).ravel(),
        obs["deadlines"].astype(np.float32).ravel(),
        obs["node_positions"].astype(np.float32).ravel(),
    ]
    # Optional time features from include_time_features config flag.
    if "time_to_reach" in obs:
        parts.append(obs["time_to_reach"].astype(np.float32).ravel())
    if "deadline_slack" in obs:
        parts.append(obs["deadline_slack"].astype(np.float32).ravel())
    if include_mask and "action_mask" in obs:
        parts.append(obs["action_mask"].astype(np.float32).ravel())
    return np.concatenate(parts)
