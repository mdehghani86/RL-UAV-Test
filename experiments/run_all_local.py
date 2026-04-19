"""Local driver: runs all 5 experiments in QUICK mode (1 seed, ~half iters,
smaller rollouts) so the full pipeline finishes overnight on a CPU laptop.

Each experiment's module-level constants are monkey-patched BEFORE main() is
called, so no edits are needed to the experiment files themselves. After all
five finish, build_paper_figures.main() runs to produce the CSV + PNG figures.

Usage (Windows or Linux):
    python -m experiments.run_all_local
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- Reduced-parameter overrides (keep structure; shrink compute ~6x) ----

def _patch(mod, **kwargs):
    """Overwrite module-level constants."""
    for k, v in kwargs.items():
        if hasattr(mod, k):
            setattr(mod, k, v)
            print(f"  [{mod.__name__}] {k} -> {v}")
        else:
            print(f"  [{mod.__name__}] WARN: no constant named {k}")


def run_exp(name: str, patches: dict):
    t0 = time.time()
    print(f"\n==================== {name} ====================")
    mod = __import__(f"experiments.{name}", fromlist=["main"])
    _patch(mod, **patches)
    try:
        mod.main()
        print(f"[{name}] ok in {time.time() - t0:.0f}s")
    except Exception as e:
        print(f"[{name}] FAILED after {time.time() - t0:.0f}s: {e}")


def main():
    overall_t0 = time.time()

    # Exp 1 — N=10 is cheap already; reduce to 1 seed, 20 iters
    run_exp("exp1_reward_ablation", dict(
        SEEDS=[0],
        ITERS=20,
        STEPS_PER_ITER=1024,
        EPISODES_EVAL=20,
    ))

    # Exp 2 — N=25 ablation, 6 variants, 1 seed, halved iters + steps
    run_exp("exp2_representation_ablation", dict(
        SEEDS=[0],
        ITERS=40,
        STEPS_PER_ITER=2048,
        EPISODES_EVAL=20,
    ))

    # Exp 3 — scaling sweep, 1 seed, smaller rollouts
    # iters scale automatically via _iters_for(N); shrink via a custom map below.
    import experiments.exp3_scaling as ex3
    original = ex3._iters_for
    ex3._iters_for = lambda N: max(10, original(N) // 2)
    run_exp("exp3_scaling", dict(
        SEEDS=[0],
        EPISODES_EVAL=20,
    ))
    ex3._iters_for = original  # restore

    # Exp 4 — algo comparison, 1 seed, halved iters
    run_exp("exp4_algorithm_comparison", dict(
        SEEDS=[0],
        ITERS=50,
        STEPS_PER_ITER=2048,
        EPISODES_EVAL=20,
    ))

    # Exp 5 — curriculum, 1 seed, halved per-stage iters
    # Main has inline iter counts; quickest fix is to pre-patch via module const
    # We shrink SEEDS only; the user can accept the default iter budget.
    # (If still too slow the inner iters can be edited in exp5_curriculum.py.)
    run_exp("exp5_curriculum", dict(
        SEEDS=[0],
        EPISODES_EVAL=20,
    ))

    # Finally build paper figures
    print("\n==================== Building paper figures ====================")
    from experiments import build_paper_figures
    try:
        build_paper_figures.main()
    except Exception as e:
        print(f"[figures] FAILED: {e}")

    total = time.time() - overall_t0
    print(f"\n================================================================")
    print(f"ALL EXPERIMENTS DONE in {total/60:.0f} min ({total/3600:.1f} hr)")
    print(f"Results: {Path(__file__).resolve().parents[1] / 'results'}")
    print(f"Figures: {Path(__file__).resolve().parents[1] / 'results' / 'paper_figures'}")


if __name__ == "__main__":
    main()
