"""Per-run logging: writes jsonl metrics live so the dashboard can tail."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    def __init__(self, run_dir: str, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.start_ts = time.time()
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        # reset metrics file
        open(self.metrics_path, "w", encoding="utf-8").close()

    def log(self, **kwargs: Any) -> None:
        rec = {"ts": time.time() - self.start_ts, **kwargs}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def summary(self, data: Dict[str, Any]) -> None:
        data = dict(data)
        data["elapsed_s"] = time.time() - self.start_ts
        data["status"] = data.get("status", "done")
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=float)

    def fail(self, err: str) -> None:
        self.summary({"status": "failed", "error": err})


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
