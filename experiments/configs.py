"""Experiment configuration sets — rewards × algos × seeds."""
from __future__ import annotations

# Reward settings — same semantic set as the notebook + one new shaping mode
REWARD_SETTINGS = {
    "dist_normalized": dict(reward_mode="dist_normalized", w_service=30.0, w_complete=60.0,
                             w_infeasible=10.0, w_dist_bonus=15.0, w_tardiness=0.2, alpha_partial=25.0),
    "completion_ratio": dict(reward_mode="completion_ratio", w_service=25.0, w_complete=60.0,
                              w_infeasible=10.0, w_tardiness=0.2, alpha_partial=25.0),
    "battery_aware": dict(reward_mode="battery_aware", w_service=30.0, w_complete=60.0,
                           w_infeasible=10.0, w_battery_pen=6.0, battery_low_thresh=0.25,
                           w_tardiness=0.2, alpha_partial=25.0),
    "time_pressure": dict(reward_mode="time_pressure", w_service=25.0, w_complete=60.0,
                           w_infeasible=10.0, w_time_late_mult=3.0, w_tardiness=0.2, alpha_partial=25.0),
    "regret_based": dict(reward_mode="regret_based", w_service=30.0, w_complete=60.0,
                          w_infeasible=10.0, w_regret=4.0, w_tardiness=0.2, alpha_partial=25.0),
    "shaped_potential": dict(reward_mode="shaped_potential", w_service=30.0, w_complete=60.0,
                              w_infeasible=10.0, w_potential=20.0, w_tardiness=0.2, alpha_partial=25.0),
}

# Algo settings — trimmed for laptop (4 cores typically busy per job with numpy BLAS if not pinned)
ALGO_SETTINGS = {
    "Heuristic": dict(kind="heuristic", eval_episodes=50),
    "PPO": dict(kind="ppo", iters=60, steps_per_iter=1024, hidden_dim=128, n_hidden=2,
                 lr=3e-4, ent_coef=0.10, ent_coef_end=0.01, minibatch_size=256, epochs=10,
                 normalise_rewards=True, include_mask_in_obs=True),
    "A2C": dict(kind="a2c", iters=240, steps_per_iter=512, hidden_dim=128, n_hidden=2,
                 lr=7e-4, ent_coef=0.02, normalise_rewards=True, include_mask_in_obs=True),
    "RecurrentPPO": dict(kind="rppo", iters=60, steps_per_iter=1024, hidden_dim=128,
                          lr=3e-4, ent_coef=0.10, ent_coef_end=0.01, chunk_len=16, epochs=4,
                          normalise_rewards=True, include_mask_in_obs=True),
    "Oracle": dict(kind="oracle", eval_episodes=30),
}

DEFAULT_SEEDS = [0, 1, 2]


# ======================================================================
# SCALING REGIMES — tighter problems where the greedy heuristic fails
#
# Mission time is scaled to ~1.15x the heuristic-achievable tour length
# so misordering costs actual served customers. Deadlines span a narrow
# window so the policy must prioritize urgent customers. Battery is
# unchanged (fits a full tour of N≤12; needs charger for N≥15).
#
# At N=5 the heuristic still gets ~85% because the problem has no
# meaningful combinatorial structure; at N=25 it drops to ~86% and
# full-mission success falls below 25% — this is the regime where RL
# should measurably win.
# ======================================================================
SCALING_REGIMES = {
    "N5":  dict(num_customers=5,  mission_time=15, deadline_min=3,  deadline_max=12),
    "N10": dict(num_customers=10, mission_time=30, deadline_min=5,  deadline_max=25),
    "N15": dict(num_customers=15, mission_time=45, deadline_min=7,  deadline_max=38),
    "N20": dict(num_customers=20, mission_time=60, deadline_min=10, deadline_max=50),
    "N25": dict(num_customers=25, mission_time=80, deadline_min=12, deadline_max=65),
}

# Scaling-sweep algo overrides — at N≥15 the problem needs much more training.
# Use this as a drop-in replacement for ALGO_SETTINGS when running scaling.
SCALING_ALGO_SETTINGS = {
    "Heuristic": dict(kind="heuristic", eval_episodes=50),
    "Oracle":    dict(kind="oracle",    eval_episodes=30),
    "PPO": dict(kind="ppo", iters=300, steps_per_iter=2048, hidden_dim=512, n_hidden=3,
                 lr=2e-4, ent_coef=0.15, ent_coef_end=0.01, minibatch_size=512, epochs=10,
                 normalise_rewards=True, include_mask_in_obs=True),
    "A2C": dict(kind="a2c", iters=900, steps_per_iter=1024, hidden_dim=512, n_hidden=3,
                 lr=5e-4, ent_coef=0.04, normalise_rewards=True, include_mask_in_obs=True),
    "RecurrentPPO": dict(kind="rppo", iters=250, steps_per_iter=2048, hidden_dim=256,
                          lr=2e-4, ent_coef=0.15, ent_coef_end=0.01, chunk_len=16, epochs=4,
                          normalise_rewards=True, include_mask_in_obs=True),
}
