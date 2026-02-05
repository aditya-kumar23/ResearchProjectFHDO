#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kpi_derive import _read_kpi_events, summarise_input_end, time_from_input_end_to_team_convergence  # noqa: E402
from tools.kpi_derive import DEFAULT_STABLE_EPSILON, DEFAULT_STABLE_REQUIRED  # noqa: E402


def _find_runs(root: Path) -> List[Path]:
    runs: List[Path] = []
    for p in root.rglob("kpi_metrics/kpi_events.jsonl"):
        runs.append(p.parent.parent)
    runs.sort(key=lambda x: str(x))
    return runs


def _short_run_id(run_dir: Path, root: Path) -> str:
    try:
        rel = run_dir.relative_to(root)
        return str(rel)
    except Exception:
        return str(run_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check Tconv(team) censoring anchored to input_end.")
    ap.add_argument("root", type=str, help="Directory to scan (e.g., new_eval/out/baseline_YYYYMMDD_HHMMSS)")
    ap.add_argument("--epsilon", type=float, default=float(DEFAULT_STABLE_EPSILON))
    ap.add_argument("--required", type=int, default=int(DEFAULT_STABLE_REQUIRED))
    ap.add_argument("--limit", type=int, default=50, help="Max per-run lines to print (0 = summary only)")
    ap.add_argument("--print-ok", action="store_true", help="Also print non-censored runs")
    args = ap.parse_args()

    root = Path(args.root)
    runs = _find_runs(root)
    if not runs:
        print(f"No runs found under: {root}")
        return 2

    total = 0
    censored = 0
    printed = 0
    for run_dir in runs:
        kpi_path = run_dir / "kpi_metrics" / "kpi_events.jsonl"
        events = _read_kpi_events(str(kpi_path))
        if not events:
            continue

        total += 1
        inp = summarise_input_end(events)
        conv = time_from_input_end_to_team_convergence(events, epsilon=float(args.epsilon), required=int(args.required))
        team = conv.get("time_from_input_end_to_team_convergence_s")
        method = conv.get("time_from_input_end_to_team_convergence_method")
        cens = method == "censored_end"
        censored += int(bool(cens))

        if args.limit <= 0:
            continue
        if not args.print_ok and not cens:
            continue
        if printed >= int(args.limit):
            continue

        run_id = _short_run_id(run_dir, root)
        ie_method = inp.get("input_end_method")
        ie_ts = inp.get("input_end_team_ts")
        team_s = "None" if team is None else f"{float(team):.2f}"
        flag = "CENSORED" if cens else "OK"
        print(f"{flag}: {run_id} | Tconv(team)={team_s}s | input_end={ie_ts} ({ie_method})")
        printed += 1

    rate = (censored / total) if total else 0.0
    print(f"\nCensored runs: {censored}/{total} ({rate:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
