# Optimized Build Prompt — RL-UAV Experiment Lab

> Akshat's single-UAV routing notebook is reformulated as a reproducible, multi-core experiment harness with a GenBI-styled live dashboard. Goal: *make the reward actually learn, beat the heuristic, and produce paper-ready comparison figures on a laptop.*

---

## PAGE 1 — Context, goal, success criteria

**Upstream:** `RL-UAV-Test/RL_Project_rewards_experiments_4.ipynb` — single-UAV routing, 5 customers + 1 charger, discrete action = next node. Compares Heuristic / PPO / A2C / RecurrentPPO across 5 reward shapes (dist_normalized, completion_ratio, battery_aware, time_pressure, regret_based).

**Reported problem:** *"model does not learn the reward and does not perform well."*

**Primary goal:** PPO (and ideally all three RL algos) achieve **mean success rate ≥ the heuristic** on at least one reward formulation, with visibly converging reward curves.

**Secondary goals:**
1. Run experiments in parallel on 12 CPU cores (laptop).
2. Single-page dashboard — GenBI glass-dark design, Outfit font, cyan accent — with live reward/eval curves, algo comparison matrix, per-run config form.
3. Reproducibility: every run saves `config.json`, `metrics.jsonl`, `summary.json` under `results/{run_id}/`.

**Non-goals (for v1):**
- Multi-UAV coordination.
- Continuous action spaces.
- Exact paper polish — this is an experimental lab, not a publication artifact yet.

**Success criteria (measurable):**
- [ ] `python run_experiments.py --reward all --algo all --seeds 3` runs 60 jobs across 12 cores and finishes in < 90 min.
- [ ] Dashboard opens at `http://localhost:5001`, shows live metrics during a run, and a 5×4 reward×algo heatmap at the end.
- [ ] At least one configuration achieves `success_rate ≥ 0.7` and matches or beats the greedy heuristic in mean return.
- [ ] Learning curves for the best configuration show clear upward convergence (not flat, not oscillating).

**Permissions granted by user:**
- Change reward functions, hyperparameters, network architecture.
- Swap/add algorithms (e.g., MaskablePPO from sb3-contrib, DQN, Rainbow).
- Web-search for SOTA tricks on combinatorial RL.
- Refactor the notebook into python modules — keep upstream notebook untouched, create a clean fork in `experiments/`.

---

## PAGE 2 — Build plan & architecture

### A. Directory layout

```
4- Akshat_RL_Experiments/
├── RL-UAV-Test/                   # upstream clone, DO NOT MODIFY
├── src/
│   ├── env.py                     # SingleUAVEnv (bug-fixed fork)
│   ├── obs.py                     # obs_to_vector (+ optional action mask in obs)
│   ├── heuristic.py               # greedy baseline
│   ├── ppo.py                     # PPO trainer (fixed reward normalization)
│   ├── a2c.py                     # A2C trainer
│   ├── rppo.py                    # RecurrentPPO trainer
│   ├── train_utils.py             # RunningMeanStd, GAE, etc.
│   └── run_logger.py              # jsonl streaming, checkpointing
├── experiments/
│   ├── configs.py                 # reward settings + algo settings
│   └── run_experiments.py         # multiprocessing driver
├── ui/
│   ├── index.html                 # single-page GenBI dashboard
│   ├── app.js                     # live chart updates (SSE/polling)
│   └── app.css                    # glass-dark theme
├── server.py                      # Flask backend (/api/runs, /api/stream, /api/start)
├── results/
│   └── {run_id}/
│       ├── config.json
│       ├── metrics.jsonl          # appended per training iter
│       └── summary.json           # final
├── PROMPT.md                      # this file
├── RESEARCH_NOTES.md              # diagnosis
├── README.md                      # install + run
└── requirements.txt
```

### B. Algorithmic fixes (from RESEARCH_NOTES diagnosis)

| # | Fix | Where | Rationale |
|---|-----|-------|-----------|
| 1 | Reward norm divides by std only, no mean centering | `train_utils.RunningMeanStd` | Preserves sparse `w_complete=60` spike |
| 2 | Action mask appended to obs vector | `obs.py::obs_to_vector` | Explicit feasibility signal |
| 3 | Partial-completion terminal bonus `alpha * served/total` | `env.py::step` terminal branch | Gives gradient on every episode end |
| 4 | Disable `regret_based` (broken) OR reformulate as "unreachable-at-mission-end" | `env.py::_reward_regret_based` | Semantic fix |
| 5 | Entropy 0.10 → 0.01 over full run, log-linear | `ppo.py::train` | Keep exploration longer |
| 6 | Orthogonal init + tanh MLP, 2×128 → 2×256 | `train_utils.make_mlp` | Standard PPO stability trick |
| 7 | Advantage normalization (already present) + value clipping | `ppo.py::update` | SB3-style |
| 8 | Longer rollouts: 80 iters × 2048 steps = 160k env steps / run | `configs.py` | 82k was too thin for combinatorial |
| 9 | Observation includes `steps / max_steps` | `obs.py` | Horizon awareness |

### C. Multi-core runner (`experiments/run_experiments.py`)

```python
from concurrent.futures import ProcessPoolExecutor
import itertools, os
jobs = list(itertools.product(REWARDS, ALGOS, SEEDS))
with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count()-2)) as ex:
    for res in ex.map(run_one_job, jobs):
        append_to_index(res)
```

Each worker sets `torch.set_num_threads(1)` to avoid oversubscription on a 12-core laptop.

### D. Dashboard UI (single page, GenBI palette)

**Layout (1440×900 reference):**
- **Top nav pill** — "RL-UAV Lab" logo, cores-in-use badge, run count
- **Left column (380 px)** — Run config form (reward dropdown, algo dropdown, seeds, iters), "Launch" CTA
- **Center (grid 2×2 live charts)** — rollout return, eval return, success rate, mean customers served. Live-updating via polling `/api/runs/{id}/tail` every 2 s.
- **Right column (340 px)** — Run status list with status chips (running, done, failed). Click a run → populates charts.
- **Bottom strip** — 5×4 reward×algo heatmap of final success rate, and average-rank bar chart (like notebook cell 8).

**Design tokens from GenBI landing.html:**
```css
--bg: #000;
--white-80: #fffc; --white-60: #fff9; --white-10: #ffffff1a; --white-6: #ffffff0f;
--brand-cyan: #06b6d4; --brand-light: #67e8f9;
--accent-gradient: linear-gradient(135deg, #a5f3fc 0%, #06b6d4 50%, #0891b2 100%);
font-family: 'Outfit', sans-serif;
backdrop-filter: blur(25px);
border-radius: 1rem | 5rem (pills);
```

Charts rendered with Chart.js (CDN, no build step) or plain Canvas 2D.

### E. Backend (`server.py`)

Flask + Flask-CORS. Endpoints:
- `GET  /api/runs` → list of runs (summary metadata)
- `GET  /api/runs/<id>` → config + summary
- `GET  /api/runs/<id>/tail?since=<iter>` → recent metrics lines (JSONL tail)
- `POST /api/start` → spawn a subprocess that calls `experiments/run_experiments.py` with a chosen config; returns `run_id`
- `POST /api/stop/<id>` → kill subprocess
- `GET  /` → serves `ui/index.html`

No DB — filesystem as source of truth.

---

## PAGE 3 — Execution sequence & checkpoints

### Milestone 1 — Refactor + baseline (no fixes yet) — *validates the pipeline*
1. Extract notebook cells to `src/*.py`.
2. Smoke-test each module (`python -m src.env`, etc.).
3. Baseline run: `reward=completion_ratio`, `algo=PPO`, 1 seed, 40 iters. Must finish < 10 min on CPU. Store under `results/baseline/`.
4. **Checkpoint:** compare numbers to the notebook's original output; they should match within seed noise.

### Milestone 2 — Apply fixes 1–4, rerun — *validates the fixes work*
1. Apply RMS fix, obs-mask, partial-completion bonus, disable regret.
2. Same config as M1, same seed. Expect visibly higher mean_return and upward-trending curve.
3. **Checkpoint:** PPO mean return on `completion_ratio` strictly exceeds M1 baseline.

### Milestone 3 — Build dashboard — *validates the UX*
1. Build `server.py` with a stub that reads existing `results/baseline/metrics.jsonl`.
2. Build `ui/index.html` + `app.js` + `app.css` — charts populate from the baseline file.
3. **Checkpoint:** open `localhost:5001`, see baseline run rendered with GenBI styling.

### Milestone 4 — Multi-core sweep — *validates scale*
1. `run_experiments.py` with ProcessPoolExecutor, 5 rewards × 4 algos × 3 seeds = 60 jobs, 10 workers.
2. Launch from dashboard "Sweep" button.
3. **Checkpoint:** all 60 results visible in heatmap; success rate for best (reward, algo) ≥ 0.7.

### Milestone 5 — Convergence improvements — *addresses the core complaint*
1. If any algo underperforms heuristic, web-search for "PPO combinatorial routing tricks", "MaskablePPO stable-baselines3-contrib", "curriculum learning for TSP with battery".
2. Try: MaskablePPO (sb3-contrib), curriculum (start with 3 customers, ramp to 5), larger network (2×256), longer rollouts (160k steps), reward shaping with distance-to-completion potential.
3. Record every iteration's result to `results/sweep_v2/`.
4. **Checkpoint:** final `success_rate ≥ heuristic` on ≥ 2 reward shapes.

### Milestone 6 — Report + hand-off
1. Generate `report.html` from results — embeds dashboard screenshots, heatmaps, winner summary, lessons learned.
2. Update `RESEARCH_NOTES.md` with what worked / didn't.
3. Open PR-style diff summary of env/trainer changes for Akshat to review.

### Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Notebook refactor introduces bugs | M1 checkpoint compares numbers to original |
| Multi-core oversubscription | `torch.set_num_threads(1)` per worker + max_workers = cores-2 |
| Long training blocks UI | Runs always spawned as subprocesses; UI polls jsonl |
| RL still won't learn | Milestone 5 fallback: MaskablePPO + curriculum |
| User wants to stop mid-sweep | `/api/stop/<id>` sends SIGTERM to subprocess group |

### What the user sees when done
1. Open dashboard → click "Run sweep" → watch 10 parallel runs' curves tick live for ~45 minutes.
2. Heatmap populates → best cell glows cyan.
3. Click best cell → drill into that run's full curves + final metrics.
4. Read `report.html` → know which reward won, by how much, and why.
