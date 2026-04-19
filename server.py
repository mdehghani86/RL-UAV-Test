"""Flask backend for RL-UAV Experiment Lab.

Endpoints:
  GET  /                       → serves ui/index.html
  GET  /ui/<path>              → static UI assets
  GET  /api/runs               → list all runs (from results/index.jsonl + per-run summary.json)
  GET  /api/runs/<run_id>      → single run config + summary + tail metrics
  GET  /api/runs/<run_id>/tail?since=<iter> → new metrics rows since given iter
  POST /api/start              → body {reward:[...], algo:[...], seeds:[...], workers:N}
                                 spawns a subprocess running run_experiments.py
  POST /api/stop               → kills running subprocess
  GET  /api/status             → {busy: bool, pid, cpu_cores, jobs_scheduled, jobs_done}
  GET  /api/heatmap            → final success_rate + mean_return pivot (reward × algo)
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory, Response

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
UI_DIR = ROOT / "ui"
INDEX_PATH = RESULTS_DIR / "index.jsonl"

app = Flask(__name__, static_folder=None)
_proc: Optional[subprocess.Popen] = None
_proc_started_at: float = 0.0


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _list_runs() -> List[Dict[str, Any]]:
    """Merge index.jsonl (scheduled/done statuses) with each run's summary.json."""
    idx = _read_jsonl(INDEX_PATH)
    # keep only latest record per run_id
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in idx:
        rid = row.get("run_id")
        if rid:
            by_id[rid] = {**by_id.get(rid, {}), **row}
    # merge per-run summary and config
    out = []
    for rid, rec in by_id.items():
        run_dir = RESULTS_DIR / rid
        summary = _read_json(run_dir / "summary.json") or {}
        cfg = _read_json(run_dir / "config.json") or {}
        merged = {**rec, "summary": summary, "config": cfg}
        out.append(merged)
    out.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return out


@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/ui/<path:subpath>")
def ui_static(subpath: str):
    return send_from_directory(str(UI_DIR), subpath)


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/api/runs/<run_id>")
def api_run(run_id: str):
    run_dir = RESULTS_DIR / run_id
    cfg = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    return jsonify({"run_id": run_id, "config": cfg, "summary": summary, "metrics": metrics})


@app.route("/api/runs/<run_id>/tail")
def api_tail(run_id: str):
    since = int(request.args.get("since", 0))
    metrics = _read_jsonl(RESULTS_DIR / run_id / "metrics.jsonl")
    tail = [m for m in metrics if int(m.get("iter", 0)) > since]
    return jsonify({"run_id": run_id, "since": since, "metrics": tail})


@app.route("/api/heatmap")
def api_heatmap():
    runs = _list_runs()
    done = [r for r in runs if r.get("status") == "done" and r.get("summary")]
    table: Dict[str, Dict[str, List[float]]] = {}
    for r in done:
        rw = r.get("reward") or r.get("summary", {}).get("reward")
        al = r.get("algo") or r.get("summary", {}).get("algo")
        sr = float(r["summary"].get("success_rate", 0.0))
        mr = float(r["summary"].get("mean_return", 0.0))
        ms = float(r["summary"].get("mean_customers_served", 0.0))
        if not rw or not al:
            continue
        cell = table.setdefault(rw, {}).setdefault(al, {"success": [], "return": [], "served": []})
        cell["success"].append(sr)
        cell["return"].append(mr)
        cell["served"].append(ms)

    def mean(xs): return sum(xs) / len(xs) if xs else None
    out = {rw: {al: {"success": mean(v["success"]),
                       "return": mean(v["return"]),
                       "served": mean(v["served"]),
                       "n": len(v["success"])}
                  for al, v in algos.items()}
           for rw, algos in table.items()}
    return jsonify(out)


@app.route("/api/status")
def api_status():
    global _proc
    busy = bool(_proc and _proc.poll() is None)
    scheduled = done = failed = 0
    for r in _list_runs():
        s = r.get("status")
        if s == "scheduled":
            scheduled += 1
        elif s == "done":
            done += 1
        elif s == "failed":
            failed += 1
    return jsonify({
        "busy": busy,
        "pid": _proc.pid if busy else None,
        "started_at": _proc_started_at if busy else None,
        "cpu_cores": os.cpu_count(),
        "counts": {"scheduled": scheduled, "done": done, "failed": failed},
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    global _proc, _proc_started_at
    if _proc and _proc.poll() is None:
        return jsonify({"ok": False, "error": "another run already in progress",
                         "pid": _proc.pid}), 409
    body = request.get_json(silent=True) or {}
    rewards = body.get("reward") or []
    algos = body.get("algo") or []
    seeds = body.get("seeds") or [0]
    regimes = body.get("regime") or []
    workers = int(body.get("workers") or max(1, (os.cpu_count() or 2) - 2))
    if not rewards or not algos:
        return jsonify({"ok": False, "error": "reward and algo are required"}), 400

    cmd = [sys.executable, "-m", "experiments.run_experiments",
           "--workers", str(workers),
           "--reward", *rewards,
           "--algo", *algos,
           "--seeds", *[str(s) for s in seeds]]
    if regimes:
        cmd += ["--regime", *regimes]
    _proc = subprocess.Popen(cmd, cwd=str(ROOT),
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    _proc_started_at = time.time()
    return jsonify({"ok": True, "pid": _proc.pid, "cmd": cmd})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _proc
    if not _proc or _proc.poll() is not None:
        return jsonify({"ok": True, "already": True})
    try:
        if os.name == "nt":
            _proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _proc.terminate()
        _proc.wait(timeout=5)
    except Exception:
        _proc.kill()
    return jsonify({"ok": True, "killed_pid": _proc.pid})


@app.route("/api/clear_runs", methods=["POST"])
def api_clear_runs():
    """Dangerous: wipe all results. Requires ?confirm=yes."""
    if request.args.get("confirm") != "yes":
        return jsonify({"ok": False, "error": "add ?confirm=yes"}), 400
    import shutil
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[server] RL-UAV Lab on http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
