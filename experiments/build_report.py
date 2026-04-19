"""Generate report.html — self-contained comparison of pre-fix vs post-fix runs.

If `results_prefix/` exists, the report is a two-column diff (before Codex
fixes vs after). Otherwise it's a single-table view of `results/`.

Usage:
    python -m experiments.build_report
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report.html"

REWARDS = ["dist_normalized", "completion_ratio", "battery_aware",
            "time_pressure", "regret_based", "shaped_potential"]
ALGOS = ["Heuristic", "PPO", "A2C", "RecurrentPPO"]


def load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    idx = results_dir / "index.jsonl"
    if not idx.exists():
        return []
    rows = []
    with open(idx, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    by_id = {}
    for row in rows:
        rid = row.get("run_id")
        if rid:
            by_id[rid] = {**by_id.get(rid, {}), **row}
    out = []
    for rid, rec in by_id.items():
        rd = results_dir / rid
        if (rd / "summary.json").exists():
            try:
                rec["summary"] = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        if (rd / "config.json").exists():
            try:
                rec["config"] = json.loads((rd / "config.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        if (rd / "metrics.jsonl").exists():
            mm = []
            for ln in (rd / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        mm.append(json.loads(ln))
                    except Exception:
                        pass
            rec["metrics"] = mm
        out.append(rec)
    return out


def agg_cell(runs: List[dict]) -> Dict[str, Optional[float]]:
    sr, mr, ms = [], [], []
    for r in runs:
        s = r.get("summary") or {}
        if s.get("success_rate") is not None:
            sr.append(s["success_rate"])
        if s.get("mean_return") is not None:
            mr.append(s["mean_return"])
        if s.get("mean_customers_served") is not None:
            ms.append(s["mean_customers_served"])
    def mean(xs): return sum(xs) / len(xs) if xs else None
    return {"success": mean(sr), "return": mean(mr), "served": mean(ms), "n": len(sr)}


def group_by_rw_algo(runs: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    g = {rw: {al: [] for al in ALGOS} for rw in REWARDS}
    for r in runs:
        rw = r.get("reward") or (r.get("config", {}) or {}).get("reward_key")
        al = r.get("algo") or (r.get("config", {}) or {}).get("algo")
        if rw in g and al in g[rw]:
            g[rw][al].append(r)
    return g


def heat_cell_html(v: Dict[str, Optional[float]]) -> str:
    if v["success"] is None:
        return "<div class='heat-cell heat-empty'>–</div>"
    s = v["success"]
    bg = f"rgba(6,182,212,{max(.08,s):.2f})"
    fg = "#001519" if s > 0.5 else "rgba(255,255,255,.85)"
    tip = f"return {v['return']:.1f} · n={v['n']}" if v["return"] is not None else ""
    return f"<div class='heat-cell' style='background:{bg};color:{fg}' title='{tip}'>{s*100:.0f}%</div>"


def heat_table(group: Dict[str, Dict[str, List[dict]]], title: str) -> str:
    rows = []
    best = None
    for rw in REWARDS:
        if not any(group[rw][al] for al in ALGOS):
            continue
        inner = f"<div class='heat-label'>{rw}</div>"
        for al in ALGOS:
            v = agg_cell(group[rw][al])
            inner += heat_cell_html(v)
            if v["success"] is not None:
                score = (v["success"], v["return"] or 0)
                if best is None or score > best[2:]:
                    best = (rw, al, v["success"], v["return"] or 0)
        rows.append(f"<div class='heat-row'>{inner}</div>")
    best_html = (f"best <strong>{best[0]} · {best[1]}</strong> — "
                 f"{best[2]*100:.0f}% @ R={best[3]:.1f}") if best else "–"
    header = "".join(f"<div class='heat-header'>{a}</div>" for a in ALGOS)
    return f"""
    <div class='card'>
      <div class='card-head'>
        <h2>{title}</h2>
        <div class='best-pill'>{best_html}</div>
      </div>
      <div class='heat'>
        <div class='heat-row'><div></div>{header}</div>
        {''.join(rows)}
      </div>
    </div>"""


def delta_table(pre: Dict, post: Dict) -> str:
    """Build a pre-vs-post comparison table."""
    rows = []
    for rw in REWARDS:
        for al in ALGOS:
            pv = agg_cell(pre.get(rw, {}).get(al, []))
            qv = agg_cell(post.get(rw, {}).get(al, []))
            if pv["success"] is None and qv["success"] is None:
                continue
            def fmt(x, pct=False):
                if x is None:
                    return "–"
                return f"{x*100:.0f}%" if pct else f"{x:.1f}"
            delta_r = None
            if pv["return"] is not None and qv["return"] is not None:
                delta_r = qv["return"] - pv["return"]
            delta_class = ""
            if delta_r is not None:
                if delta_r > 2:
                    delta_class = "delta-pos"
                elif delta_r < -2:
                    delta_class = "delta-neg"
            rows.append(f"""
            <tr>
              <td>{rw}</td>
              <td>{al}</td>
              <td class='num'>{fmt(pv['return'])}</td>
              <td class='num'>{fmt(pv['success'], pct=True)}</td>
              <td class='num'>{fmt(qv['return'])}</td>
              <td class='num'>{fmt(qv['success'], pct=True)}</td>
              <td class='num {delta_class}'>{"+" if (delta_r or 0) > 0 else ""}{delta_r:+.1f if delta_r is not None else ""}</td>
            </tr>""") if False else rows.append(
                f"<tr>"
                f"<td>{rw}</td><td>{al}</td>"
                f"<td class='num'>{fmt(pv['return'])}</td>"
                f"<td class='num'>{fmt(pv['success'], pct=True)}</td>"
                f"<td class='num'>{fmt(qv['return'])}</td>"
                f"<td class='num'>{fmt(qv['success'], pct=True)}</td>"
                f"<td class='num {delta_class}'>"
                f"{'' if delta_r is None else ('+' if delta_r >= 0 else '') + f'{delta_r:.1f}'}"
                f"</td>"
                f"</tr>"
            )
    return f"""
    <div class='card'>
      <h2>Pre-fix vs Post-fix deltas</h2>
      <table>
        <thead><tr>
          <th>reward</th><th>algo</th>
          <th class='num'>pre R</th><th class='num'>pre S</th>
          <th class='num'>post R</th><th class='num'>post S</th>
          <th class='num'>Δ R</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


def per_run_table(runs: List[dict]) -> str:
    rows = []
    srt = sorted(runs, key=lambda r: (r.get("summary", {}).get("mean_return") or -1e9), reverse=True)
    for r in srt:
        s = r.get("summary", {}) or {}
        mr = s.get("mean_return")
        sr = s.get("success_rate")
        ms = s.get("mean_customers_served")
        rows.append(
            f"<tr>"
            f"<td>{r.get('run_id','')}</td>"
            f"<td>{r.get('reward','')}</td>"
            f"<td>{r.get('algo','')}</td>"
            f"<td>{r.get('seed','?')}</td>"
            f"<td class='num'>{'–' if mr is None else f'{mr:.1f}'}</td>"
            f"<td class='num'>{'–' if sr is None else f'{sr*100:.0f}%'}</td>"
            f"<td class='num'>{'–' if ms is None else f'{ms:.2f}'}</td>"
            f"</tr>"
        )
    return f"""
    <div class='card'>
      <h2>All runs (post-fix, sorted by mean return)</h2>
      <table>
        <thead><tr><th>run_id</th><th>reward</th><th>algo</th><th>seed</th>
        <th class='num'>return</th><th class='num'>success</th><th class='num'>served</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


CSS = """
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Outfit',sans-serif;background:#000;color:rgba(255,255,255,.85);padding:2rem;font-weight:300;
  background: radial-gradient(ellipse 60% 40% at 20% 0%,rgba(6,182,212,.10),transparent 60%),#000;}
h1{font-size:2.2rem;font-weight:300;letter-spacing:-.02em;margin-bottom:.25rem}
h2{font-size:1.1rem;font-weight:400;color:rgba(255,255,255,.9);letter-spacing:-.01em}
em{font-weight:500;font-style:italic;background:linear-gradient(135deg,#a5f3fc,#06b6d4 50%,#0891b2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
p.muted{color:rgba(255,255,255,.5);font-size:.9rem;margin-bottom:1.5rem}
.card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:1rem;padding:1.5rem;margin-bottom:1rem}
.card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem}
.best-pill{font-size:.85rem;color:#67e8f9;background:rgba(6,182,212,.08);border:1px solid rgba(6,182,212,.3);padding:.35rem .9rem;border-radius:5rem}
.heat{display:grid;gap:.3rem}
.heat-row{display:grid;grid-template-columns:180px repeat(4,1fr);gap:.3rem}
.heat-header{color:rgba(255,255,255,.6);font-size:.78rem;padding:.35rem;text-align:center}
.heat-label{font-size:.82rem;color:rgba(255,255,255,.7);padding:.5rem}
.heat-cell{padding:.7rem;border-radius:.5rem;text-align:center;font-size:.9rem;font-weight:500;font-variant-numeric:tabular-nums}
.heat-empty{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);color:rgba(255,255,255,.3)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.55rem .7rem;text-align:left;border-bottom:1px solid rgba(255,255,255,.08)}
th{color:rgba(255,255,255,.6);font-weight:400;text-transform:uppercase;font-size:.72rem;letter-spacing:.06em}
td.num{text-align:right;font-variant-numeric:tabular-nums;color:#67e8f9}
td.delta-pos{color:#22c55e;font-weight:500}
td.delta-neg{color:#ef4444;font-weight:500}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}
.twocol .card{margin-bottom:0}
@media(max-width:1100px){.twocol{grid-template-columns:1fr}}
"""


def build_html(pre_runs, post_runs) -> str:
    has_pre = bool(pre_runs)
    pre_g = group_by_rw_algo(pre_runs) if has_pre else None
    post_g = group_by_rw_algo(post_runs)

    body = []
    body.append(f"<h1>RL-UAV <em>Lab</em> · Report</h1>")
    if has_pre:
        body.append(f"<p class='muted'>{len(pre_runs)} pre-fix runs · {len(post_runs)} post-fix runs · "
                     f"{len(REWARDS)} rewards × {len(ALGOS)} algorithms</p>")
        body.append("<div class='twocol'>")
        body.append(heat_table(pre_g, "Pre-fix — success rate, reward × algo"))
        body.append(heat_table(post_g, "Post-fix — success rate, reward × algo"))
        body.append("</div>")
        body.append(delta_table(pre_g, post_g))
    else:
        body.append(f"<p class='muted'>{len(post_runs)} runs · {len(REWARDS)} rewards × {len(ALGOS)} algorithms</p>")
        body.append(heat_table(post_g, "Success rate, reward × algo"))

    body.append(per_run_table(post_runs))
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>RL-UAV Lab — Report</title>
<link href='https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap' rel='stylesheet'>
<style>{CSS}</style></head>
<body>{''.join(body)}</body></html>"""
    return html


if __name__ == "__main__":
    post = load_runs(ROOT / "results")
    pre_dir = ROOT / "results_prefix"
    pre = load_runs(pre_dir) if pre_dir.exists() else []
    html = build_html(pre, post)
    OUT.write_text(html, encoding="utf-8")
    print(f"[report] wrote {OUT} | pre={len(pre)} post={len(post)}")
