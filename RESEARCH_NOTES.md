# RL-UAV-Test — Diagnosis & Research Plan

**Notebook:** `RL_Project_rewards_experiments_4.ipynb` (single-UAV routing, 5 customers, 1 charger, discrete action = next node). Compares Heuristic / PPO / A2C / RecurrentPPO across five reward shapes.

**User-reported symptom:** *"the model does not learn the reward and does not perform well."*

---

## 1. Environment recap

- Obs: `agent=[cur_node_norm, batt_frac, time_frac]`, `visited`, `deadlines`, `node_positions`.
- Action: `Discrete(total_nodes)` — depot, customers, chargers.
- Termination: (a) all customers served & back at depot (good), (b) invalid action or out-of-battery/time (bad — episode ends with `-w_infeasible`), (c) `steps ≥ max_steps=100` (return 0 reward).
- Action masking is implemented (`get_action_mask`) and **is** applied to policy logits during rollout + update (`safe_mask_logits`). That part is correct.

## 2. Likely reasons training stalls (ranked)

### A. Reward normalization subtracts the mean — biggest issue

`cell 5` `RunningMeanStd.normalize` does `(x - mean) / sqrt(var)`. Standard PPO practice (SB3 `VecNormalize`, CleanRL, SpinningUp) is **divide by std of the discounted return only — do NOT subtract the mean**. Reason: in sparse-reward tasks, `w_complete=60` appears once per episode; mean-centering drags that spike toward zero and replaces it with a stream of tiny negative values everywhere else. The learning signal is literally removed.

**Fix:** track `RunningMeanStd` on the discounted return (GAE target), divide `R` by `sqrt(var) + 1e-8`, do not subtract mean. Or simply set `normalise_rewards=False` and scale `w_*` weights so per-step reward stays in ~[-2, +2] range.

### B. Action mask is not in the policy observation

`obs_to_vector` concatenates agent/visited/deadlines/positions but NOT the action mask. The policy has to infer feasibility from (battery, remaining time, deadlines, positions). Possible but slow. With the mask included, the network gets an explicit "you cannot go here" signal.

**Fix:** append `info["action_mask"].astype(np.float32)` to `obs_to_vector`. Cheap, big impact.

### C. `regret_based` reward is semantically broken

In `_reward_regret_based`, `reachable_before` is computed *before* moving (line 409 of cell 3) — so from the old node; `reachable_after` is computed from the new node. The difference is "change in reachability due to the move", not a regret signal. Any customer that was reachable from A but not from B counts as "lost" even though the agent is now closer to other customers.

**Fix:** either (i) compute `reachable_after` by hypothetically continuing from the same node (which doesn't make physical sense either — the agent moved), or (ii) measure regret as "customers whose deadline is now unreachable in remaining mission time from depot". The cleaner reformulation is unreachable-at-end-of-episode, but that needs a terminal bonus/penalty, not dense.

### D. No partial-completion reward + harsh timeout

If the agent serves 4/5 customers and time runs out, it gets `r - w_infeasible = -10` on the terminal step and no `w_complete`. So the gradient between "serve 4 then timeout" and "serve 0 then wander" is mostly the per-service rewards, which under normalization (A) collapse. A small terminal bonus `alpha * (visited_mask.sum() / num_customers)` on any termination would give a denser learning signal.

### E. Entropy schedule / iters may be too short

40 iters × 2048 steps = 82k env steps per run. For a 6-node combinatorial problem with batteries and deadlines, that's on the thin side. Entropy anneals 0.05 → 0.005 linearly; by iter 20 it's ~0.028 — often too low before the agent has converged on a reasonable routing. Consider 80–120 iters or slower annealing.

### F. Reward-mode defaults in `SingleUAVConfig` are not overridden consistently

`config.deterministic_seed=42` is baked into `make_env`, but `reset(seed=...)` re-seeds `self.rng`. Good. However, every `env_fn()` call creates a brand-new env with the same `deterministic_seed=42`, so the **node layout is identical across episodes within a run**. That means evaluation looks better than true generalization. For a paper, you want node layouts to be drawn from a distribution each episode.

**Fix:** in `reset()`, always advance the RNG (or pass distinct seeds from the trainer, which is already done for the PPO rollout — `env.reset(seed=randint(0, 1_000_000))`). Verify training actually sees varied layouts; add a sanity print.

## 3. Minor items

- `PPOTrainer.act` (greedy eval) uses `argmax` of masked logits; fine.
- `rms.update(R)` then immediately `rms.normalize(R)` on the same rollout is mildly biased but not the main issue.
- `rollout_mean_return` logs *normalized* returns (since `R` is overwritten after `rms.normalize`). The "learning curve" plot is therefore misleading — it flattens because normalization makes it flat, not because learning stalled. Always log the raw episode return for plots.
- `max_steps=100` is generous for 5 customers + 1 charger, so truncation is not the bottleneck.

## 4. Proposed research plan

### Phase 0 — Reproduce (this week)
1. Run the notebook end-to-end on CPU, capture baseline numbers for all 5 rewards × 4 algos. Save `uav_rl_results.csv` from the original code as reference.

### Phase 1 — Fix the training loop (1–2 sessions)
1. **Patch reward normalization** (issue A) — divide by return std only, log raw returns.
2. **Add action mask to observation** (issue B).
3. **Add partial-completion terminal bonus** (issue D).
4. **Disable or reformulate `regret_based`** (issue C).
5. Re-run; expect PPO to approach or beat the heuristic on `CompletionRatio` and `TimePressure`.

### Phase 2 — Experimental design (1 week)
1. Hold reward fixed at the best of Phase 1. Vary:
   - `num_customers ∈ {5, 8, 12}`
   - `num_chargers ∈ {1, 2}`
   - `mission_time ∈ {300, 500, 750}`
2. 3 seeds × ~100k steps × N settings = several hours on CPU.
3. Plot success-rate vs. problem size; compare PPO vs. A2C vs. RecurrentPPO vs. heuristic.

### Phase 3 — Paper-ready story
1. Hypothesis: which reward shape transfers best across problem sizes?
2. Does recurrence (RecurrentPPO) help for longer missions specifically?
3. Does the heuristic remain competitive? On which regime does RL clearly win?

## 5. Deployment / reproducibility checklist

- [x] Repo cloned locally
- [x] `requirements.txt` pinned
- [x] README with venv instructions
- [ ] `venv/` created, deps installed (user)
- [ ] Smoke test: run all 9 cells on CPU, capture per-cell timings
- [ ] `experiments/` folder for forked patches (created in Phase 1)
- [ ] Push our fixes as PR to Akshat's repo once validated

## 6. Open questions for next discussion with Akshat

1. What's the target paper venue? (That decides how many seeds / problem sizes are needed.)
2. Is single-UAV the endpoint or is multi-UAV the goal?
3. Do we want to compare against an OR baseline (MILP, CP-SAT) as a ceiling, not just a greedy heuristic?
4. Does he have compute beyond Colab? Training three algos × five rewards × 3 seeds takes ~6–10 hr on a modern CPU.
