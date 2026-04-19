# RL-UAV Experiment Lab

**Upstream:** https://github.com/akshatjain632/RL-UAV-Test (Akshat's notebook).
This folder refactors the notebook into a reproducible multi-core experiment
harness with a GenBI-styled live dashboard.

## What's inside

| Path | Purpose |
|------|---------|
| `RL-UAV-Test/` | Clone of Akshat's repo — do not modify |
| `src/` | Refactored Python modules (env, PPO/A2C/RecurrentPPO, heuristic, logger) |
| `experiments/` | Config presets + multiprocessing runner |
| `ui/` | Single-page dashboard (Outfit / liquid-glass cyan) |
| `server.py` | Flask backend — serves dashboard + exposes /api endpoints |
| `results/` | One folder per run: config.json, metrics.jsonl, summary.json |
| `PROMPT.md` | Optimized 3-page build prompt (scope, plan, milestones) |
| `RESEARCH_NOTES.md` | Diagnosis of why the upstream notebook didn't learn + fix log |

## Quick start

```bash
cd "C:\Users\mdehg\Dropbox\2_Research\11- UAV with RL\4- Akshat_RL_Experiments"
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Option A — dashboard (open browser to http://localhost:5001)
python server.py

# Option B — CLI runner
python -m experiments.run_experiments --reward completion_ratio shaped_potential \
       --algo Heuristic PPO RecurrentPPO --seeds 0 1 2 --workers 10
```

## Bug fixes applied to the upstream training loop

1. **Reward normalization divides by std only — no mean centering.** Upstream
   `RunningMeanStd.normalize` subtracted the mean, which destroyed the sparse
   `w_complete=60` spike that is the only clear terminal signal. Replaced
   with `RunningStd` in `src/train_utils.py`.
2. **Action mask appended to the policy observation.** Upstream policy had to
   infer feasibility from battery/time/deadline indirectly; now the mask is
   concatenated into the obs vector in `src/obs.py`.
3. **Partial-completion terminal bonus.** Upstream gave zero reward on
   truncation and −10 on timeout regardless of how many customers were
   served. Fix adds `alpha_partial * (served / total)` on both terminal
   branches (`src/env.py`).
4. **Regret reward reformulated.** Upstream measured `reachable_before` and
   `reachable_after` from different nodes, making the signal meaningless.
   Fixed to compare reachability from the same reference frame.
5. **Orthogonal init + proper entropy schedule** (0.10 → 0.01 linear).
6. **`torch.set_num_threads(1)` per worker** — required for parallel
   `ProcessPoolExecutor` runs to not oversubscribe CPU.
7. **Added a 6th reward — `shaped_potential`.** Potential-based shaping
   `F(s,a,s') = γ·φ(s') − φ(s)` with `φ = served_frac − 0.3·nearest_unvisited/diag`.
   Invariance-preserving — optimal policy unchanged.

## Dashboard features

- Launch arbitrary (reward × algo × seed) sweep from the left panel.
- Live rollout return, greedy eval return, success rate, customers served —
  four charts update every 1.5 s during training.
- Run list with status dots (scheduled / running / done / failed), click to
  drill into charts.
- Reward × Algo success-rate heatmap, best cell highlighted.
- All state is on disk; no database. Restart the server anytime.

## License / attribution

Akshat's upstream notebook retains its original license. Our forked
modules (`src/`, `experiments/`, `ui/`, `server.py`) are for internal
research use by the Dehghani group.
