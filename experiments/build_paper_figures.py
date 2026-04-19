"""Aggregate all exp1-exp5 JSON results into a CSV + matplotlib figures.

Run AFTER the experiments finish (locally or on the cluster). Produces:

  results/paper_figures/
    all_results.csv                 : one row per (exp, policy, N, seed)
    exp1_reward_ablation.png        : bar — mean return per reward mode
    exp2_representation.png         : bar — ablation bars with error
    exp3_scaling_return.png         : line — return vs N per policy
    exp3_scaling_success.png        : line — success_rate vs N per policy
    exp3_scaling_served.png         : line — served fraction vs N per policy
    exp4_algorithm_table.csv        : clean comparison table
    exp4_algorithm_bars.png         : grouped bars across KPIs
    exp5_curriculum.png             : bar — curriculum regimes
    heatmap_reward_x_algo.png       : heatmap (if data available)

Usage:
    python -m experiments.build_paper_figures
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Matplotlib headless (no X display on HPC).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_figures"

# Consistent plotting style
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 180, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

POLICY_COLORS = {
    "PPO": "#2563EB", "PPO-vanilla": "#94A3B8", "PPO-enhanced": "#2563EB",
    "A2C": "#16A34A", "RPPO": "#9333EA",
    "greedy_db": "#F97316", "nearest_deadline": "#EAB308",
    "nearest_neighbour": "#DC2626", "random": "#9CA3AF",
}


def _load_dir(d: Path) -> list[dict]:
    recs = []
    if not d.exists():
        return recs
    for f in sorted(d.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
            rec["_file"] = f.name
            recs.append(rec)
        except Exception as e:
            print(f"skip {f}: {e}")
    return recs


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


CSV_FIELDS = [
    "exp", "policy", "method", "variant", "regime", "reward",
    "N", "seed", "iters", "mean_return", "std_return", "median_return",
    "success_rate", "mean_customers_served", "service_rate",
    "mean_tardiness", "tardiness_rate", "mean_distance",
    "mean_charger_visits", "mean_infeasible", "time_utilisation",
    "energy_utilisation", "mean_steps",
]


def _collect_all() -> dict:
    return {
        "exp1": _load_dir(RESULTS / "exp1_reward_ablation"),
        "exp2": _load_dir(RESULTS / "exp2_representation"),
        "exp3": _load_dir(RESULTS / "exp3_scaling"),
        "exp4": _load_dir(RESULTS / "exp4_algorithm"),
        "exp5": _load_dir(RESULTS / "exp5_curriculum"),
    }


# Module-level placeholder so the figure functions can reference it after main() populates.
ALL_RECS: dict = {}


# --- Figure helpers ---

def _bar(ax, labels, means, stds=None, colors=None, title="", ylabel=""):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors or "#2563EB",
            capsize=4, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _group_mean_std(rows: list[dict], group_by: str, metric: str):
    groups = defaultdict(list)
    for r in rows:
        key = r.get(group_by)
        val = r.get(metric)
        if key is not None and isinstance(val, (int, float)):
            groups[key].append(float(val))
    out = {}
    for k, vals in groups.items():
        out[k] = (float(np.mean(vals)),
                  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                  len(vals))
    return out


# --- EXP 1: reward ablation ---
def fig_exp1():
    recs = ALL_RECS["exp1"]
    if not recs:
        return
    ppo = [r for r in recs if r.get("policy") == "PPO"]
    heur = [r for r in recs if r.get("policy") in ("greedy_db", "nearest_deadline",
                                                     "nearest_neighbour", "random")]
    m = _group_mean_std(ppo, "reward", "mean_return")
    if not m:
        return
    labels = list(m.keys())
    means = [m[k][0] for k in labels]
    stds = [m[k][1] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, means, stds, title="Exp 1 — Reward-mode ablation, PPO @ N=10",
         ylabel="Mean return (50 eval eps, 3 seeds)")
    # Overlay heuristic lines
    for r in heur:
        ax.axhline(r["mean_return"], color=POLICY_COLORS.get(r.get("policy"), "gray"),
                    ls="--", lw=1, label=f"{r.get('policy')}={r['mean_return']:.0f}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "exp1_reward_ablation.png")
    plt.close(fig)
    print("Wrote exp1_reward_ablation.png")


# --- EXP 2: representation ablation ---
def fig_exp2():
    recs = [r for r in ALL_RECS["exp2"] if r.get("policy") == "PPO"]
    heur = [r for r in ALL_RECS["exp2"] if r.get("variant") == "heuristic"]
    if not recs:
        return
    m = _group_mean_std(recs, "variant", "mean_return")
    labels = sorted(m.keys())
    means = [m[k][0] for k in labels]
    stds = [m[k][1] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, means, stds,
         title="Exp 2 — Representation/shaping ablation, PPO @ N=25",
         ylabel="Mean return (50 eval eps, 3 seeds)")
    for r in heur:
        ax.axhline(r["mean_return"], ls="--", lw=1,
                    color=POLICY_COLORS.get(r.get("policy"), "gray"),
                    label=f"{r.get('policy')}={r['mean_return']:.0f}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "exp2_representation.png")
    plt.close(fig)
    print("Wrote exp2_representation.png")


# --- EXP 3: scaling ---
def fig_exp3():
    recs = ALL_RECS["exp3"]
    if not recs:
        return
    policies = sorted({r.get("policy") for r in recs if r.get("policy")})
    Ns = sorted({r.get("N") for r in recs if r.get("N") is not None})
    for metric, fname, ylabel in [
        ("mean_return", "exp3_scaling_return.png", "Mean return"),
        ("success_rate", "exp3_scaling_success.png", "Success rate"),
        ("service_rate", "exp3_scaling_served.png", "Customers served / N"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for pol in policies:
            sub = [r for r in recs if r.get("policy") == pol]
            # For each N, mean over seeds
            ys, yerr = [], []
            for N in Ns:
                vals = [r.get(metric) for r in sub if r.get("N") == N
                         and isinstance(r.get(metric), (int, float))]
                if vals:
                    ys.append(np.mean(vals))
                    yerr.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                else:
                    ys.append(np.nan)
                    yerr.append(0.0)
            ax.errorbar(Ns, ys, yerr=yerr, marker="o", label=pol,
                         color=POLICY_COLORS.get(pol, None), capsize=3)
        ax.set_xlabel("N (customers)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Exp 3 — {ylabel} vs problem size")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)
        print(f"Wrote {fname}")


# --- EXP 4: algorithm comparison ---
def fig_exp4():
    recs = ALL_RECS["exp4"]
    if not recs:
        return
    methods = sorted({r.get("method") or r.get("policy") for r in recs})
    # Clean CSV for paper table
    table_rows = []
    for m in methods:
        sub = [r for r in recs if (r.get("method") or r.get("policy")) == m]
        row = {"method": m, "n_runs": len(sub)}
        for k in ["mean_return", "success_rate", "mean_customers_served",
                   "mean_tardiness", "mean_distance", "time_utilisation"]:
            vals = [r.get(k) for r in sub if isinstance(r.get(k), (int, float))]
            row[k + "_mean"] = float(np.mean(vals)) if vals else None
            row[k + "_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        table_rows.append(row)
    _write_csv(OUT / "exp4_algorithm_table.csv", table_rows,
               list(table_rows[0].keys()) if table_rows else [])
    print("Wrote exp4_algorithm_table.csv")

    # Grouped bar plot: 4 KPIs × methods
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (k, pretty) in zip(axs.ravel(), [
        ("mean_return", "Mean return"),
        ("success_rate", "Success rate"),
        ("mean_customers_served", "Customers served"),
        ("mean_tardiness", "Tardiness"),
    ]):
        m = _group_mean_std(recs, "method", k)
        labels = sorted(m.keys())
        means = [m[l][0] for l in labels]
        stds = [m[l][1] for l in labels]
        colors = [POLICY_COLORS.get(l, "#334155") for l in labels]
        _bar(ax, labels, means, stds, colors=colors, ylabel=pretty, title=pretty)
    fig.suptitle("Exp 4 — Algorithm comparison @ N=25", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_algorithm_bars.png")
    plt.close(fig)
    print("Wrote exp4_algorithm_bars.png")


# --- EXP 5: curriculum ---
def fig_exp5():
    recs = ALL_RECS["exp5"]
    if not recs:
        return
    ppo = [r for r in recs if r.get("regime") and r.get("regime") != "heuristic"]
    heur = [r for r in recs if r.get("regime") == "heuristic"]
    m = _group_mean_std(ppo, "regime", "mean_return")
    labels = sorted(m.keys())
    means = [m[k][0] for k in labels]
    stds = [m[k][1] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bar(ax, labels, means, stds, title="Exp 5 — Curriculum ablation @ N=25",
         ylabel="Mean return (50 eps, 2 seeds)")
    for r in heur:
        ax.axhline(r["mean_return"], ls="--", lw=1,
                    color=POLICY_COLORS.get(r.get("policy"), "gray"),
                    label=f"{r.get('policy')}={r['mean_return']:.0f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "exp5_curriculum.png")
    plt.close(fig)
    print("Wrote exp5_curriculum.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global ALL_RECS
    ALL_RECS = _collect_all()

    # Write combined CSV
    all_rows = []
    for exp_key, recs in ALL_RECS.items():
        for r in recs:
            row = {"exp": r.get("exp", exp_key)}
            for k in CSV_FIELDS:
                if k == "exp":
                    continue
                row[k] = r.get(k, "")
            all_rows.append(row)
    _write_csv(OUT / "all_results.csv", all_rows, CSV_FIELDS)
    print(f"Wrote {OUT/'all_results.csv'} ({len(all_rows)} rows)")

    fig_exp1()
    fig_exp2()
    fig_exp3()
    fig_exp4()
    fig_exp5()
    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()
