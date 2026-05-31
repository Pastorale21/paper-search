"""Tiny append-only eval-history tracker.

``record()`` appends one JSON line per (method, run) to ``data/cache/eval/history.jsonl``.
``python -m eval.history`` prints a table summarizing the history, with the latest run per
method highlighted (ANSI bold) so C can see "did changing FIELD_WEIGHTS move nDCG@5?".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from spike import config as spike_config

HISTORY_PATH: Path = spike_config.CACHE_DIR / "eval" / "history.jsonl"

_BOLD = "\033[1m"
_RESET = "\033[0m"


def record(method: str, metrics: dict, notes: str = "") -> None:
    """Append one record to the history JSONL (creates the file on first call)."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": method,
        "metrics": {
            k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()
        },
        "notes": notes,
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _print_table(rows: list[dict]) -> None:
    """Print history table; bold the latest row per (method, notes)."""
    if not rows:
        print("[history] empty — no recorded runs yet.")
        return
    # Index of the latest row for each (method, notes) pair.
    latest_idx: dict[tuple[str, str], int] = {}
    for i, r in enumerate(rows):
        latest_idx[(r["method"], r.get("notes", ""))] = i

    header = f"  {'ts':<22}  {'method':<14}  {'nDCG@5':>7}  {'MRR':>7}  {'Recall@10':>10}  notes"
    print(header)
    for i, r in enumerate(rows):
        m = r.get("metrics", {})
        line = (
            f"  {r['ts']:<22}  {r['method']:<14}  "
            f"{m.get('ndcg@5', 0):>7.3f}  "
            f"{m.get('mrr', 0):>7.3f}  "
            f"{m.get('recall@10', 0):>10.3f}  "
            f"{r.get('notes', '')}"
        )
        if latest_idx.get((r["method"], r.get("notes", ""))) == i:
            line = f"{_BOLD}{line}{_RESET}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print the eval history table.")
    ap.add_argument("--path", type=Path, default=HISTORY_PATH, help="history JSONL path override")
    args = ap.parse_args()
    _print_table(_load_history(args.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
