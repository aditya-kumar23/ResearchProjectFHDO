#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _uniq(rows: Iterable[Dict[str, Any]], key: str) -> List[str]:
    vals = []
    seen = set()
    for r in rows:
        v = r.get(key, None)
        s = "" if v is None else str(v)
        if s not in seen:
            seen.add(s)
            vals.append(s)
    return sorted(vals, key=lambda x: (x == "", x))


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _fmt_vals(vals: List[str]) -> str:
    if not vals:
        return ""
    if len(vals) <= 12:
        return ", ".join(f"`{v}`" if v else "`(empty)`" for v in vals)
    return ", ".join(f"`{v}`" if v else "`(empty)`" for v in vals[:12]) + f", … (+{len(vals)-12})"


def summarize_one(sweep_dir: Path) -> Tuple[str, Dict[str, List[str]]]:
    plan = _read_json(sweep_dir / "run_plan.json")
    runs = plan.get("runs", [])
    runs = runs if isinstance(runs, list) else []
    runs2 = [r for r in runs if isinstance(r, dict)]

    keys = [
        "sweep_type",
        "dataset_tag",
        "dataset_name",
        "modality",
        "team_size",
        "backend",
        "qos_reliability",
        "qos_durability",
        "qos_depth",
        "impair_enabled",
        "impair_name",
        "impair_seed",
        "config_hash",
    ]
    uniq: Dict[str, List[str]] = {k: _uniq(runs2, k) for k in keys}

    lines: List[str] = []
    lines.append(f"# Sweep parameter values: `{sweep_dir}`")
    lines.append("")
    lines.append(f"- Total runs in `run_plan.json`: **{len(runs2)}**")
    lines.append("")
    rows_md = [[k, _fmt_vals(uniq[k])] for k in keys]
    lines.append(_md_table(["parameter", "values used"], rows_md))
    lines.append("")
    return "\n".join(lines) + "\n", uniq


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize parameter values used in one or more out/sweep_* run_plan.json files.")
    ap.add_argument("--sweep-dir", action="append", help="Sweep directory (repeatable). If omitted, auto-detect out/sweep_* dirs.")
    ap.add_argument("--write-per-sweep", action="store_true", help="Write <sweep-dir>/eval/params_used.md for each sweep.")
    ap.add_argument("--out", default="out/sweep_params_used.md", help="Combined output markdown path.")
    args = ap.parse_args(argv)

    sweep_dirs: List[Path] = []
    if args.sweep_dir:
        sweep_dirs = [Path(p).expanduser().resolve() for p in args.sweep_dir]
    else:
        out_root = Path("out").resolve()
        for p in sorted(out_root.glob("sweep_*")):
            if (p / "run_plan.json").exists():
                sweep_dirs.append(p.resolve())

    if not sweep_dirs:
        raise SystemExit("No sweep dirs found (expected run_plan.json).")

    combined: List[str] = []
    combined.append("# Sweep parameter values (all sweeps)")
    combined.append("")

    for sweep_dir in sweep_dirs:
        md, _uniq_map = summarize_one(sweep_dir)
        combined.append(md.rstrip())
        combined.append("")
        if args.write_per_sweep:
            out_path = sweep_dir / "eval" / "params_used.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(combined).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote combined summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

