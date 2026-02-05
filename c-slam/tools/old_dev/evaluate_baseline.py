#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not vals:
        return {"count": 0}

    def pct(p: float) -> float:
        if p <= 0:
            return vals[0]
        if p >= 100:
            return vals[-1]
        rank = (p / 100.0) * (len(vals) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(vals) - 1)
        w = rank - lo
        return vals[lo] * (1 - w) + vals[hi] * w

    out: Dict[str, Optional[float]] = {
        "count": float(len(vals)),
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p90": pct(90.0),
        "p95": pct(95.0),
        "p99": pct(99.0),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.pstdev(vals)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _extract_ate_by_robot(estimation_metrics: Any) -> Dict[str, Dict[str, Optional[float]]]:
    """Return robot_id -> {'rmse': float|None, 'matches': int|None} for both JSON shapes."""
    if not isinstance(estimation_metrics, dict):
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}

    ate = estimation_metrics.get("ate")
    if isinstance(ate, dict):
        # Centralised shape: {"ate": {"a": {"rmse": ...}, ...}, ...}
        for rid, payload in ate.items():
            if not isinstance(payload, dict):
                continue
            out[str(rid)] = {
                "rmse": _safe_float(payload.get("rmse")),
                "matches": float(payload.get("matches") or 0),
            }
        return out

    # Decentralised shape: {"a": {"ate": {"rmse": ...}, ...}, ...}
    for rid, payload in estimation_metrics.items():
        if not isinstance(payload, dict):
            continue
        ate2 = payload.get("ate")
        if not isinstance(ate2, dict):
            continue
        out[str(rid)] = {
            "rmse": _safe_float(ate2.get("rmse")),
            "matches": float(ate2.get("matches") or 0),
        }
    return out


def _bytes_to_mib(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x) / (1024.0 * 1024.0)


def _fmt(x: Any, *, digits: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _read_run_status(run_dir: Path) -> Tuple[Optional[bool], str, Optional[float]]:
    path = run_dir / "run_status.json"
    if not path.exists():
        return None, "missing_run_status", None
    doc = _read_json(path)
    ok = _as_bool(doc.get("ok"))
    err = str(doc.get("error") or "")
    ts_wall = _safe_float(doc.get("ts_wall"))
    return ok, err, ts_wall


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _extract_l_global_from_kpi_events(kpi_events_path: Path) -> Dict[str, Optional[float]]:
    """
    Approximate L_global as (map_broadcast.ts - immediately preceding sensor_ingest.ts).

    This pairing is exact for the default baseline template where publisher batch_size=1 and
    each ingest triggers an optimization + broadcast quickly. If batch sizes change, this
    becomes a heuristic and should be replaced by an explicit batch ingest event.
    """
    last_ingest_ts: Optional[float] = None
    l_global_vals: List[float] = []
    for ev in _iter_jsonl(kpi_events_path):
        kind = str(ev.get("event") or "")
        if kind == "sensor_ingest":
            last_ingest_ts = _safe_float(ev.get("ts"))
        elif kind == "map_broadcast":
            if last_ingest_ts is None:
                continue
            ts = _safe_float(ev.get("ts"))
            if ts is None:
                continue
            l_global_vals.append(float(ts - last_ingest_ts))
    stats = _stats(l_global_vals)
    return {
        "l_global_count": stats.get("count"),
        "l_global_mean_s": stats.get("mean"),
        "l_global_p95_s": stats.get("p95"),
    }


@dataclass(frozen=True)
class RunRef:
    scenario: str
    backend: str
    run_dir: Path


def _find_runs(baseline_dir: Path) -> List[RunRef]:
    runs: List[RunRef] = []
    for manifest in baseline_dir.glob("*/*/run_manifest.json"):
        run_dir = manifest.parent
        backend = run_dir.name
        scenario = run_dir.parent.name
        runs.append(RunRef(scenario=scenario, backend=backend, run_dir=run_dir))
    return sorted(runs, key=lambda r: (r.scenario, r.backend))


def evaluate_baseline(baseline_dir: Path, out_dir: Path) -> None:
    runs = _find_runs(baseline_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    delivery_rows: List[Dict[str, Any]] = []

    for ref in runs:
        manifest = _read_json(ref.run_dir / "run_manifest.json")
        orch_cfg = _read_json(ref.run_dir / "orchestrate_config.json")
        derived = _read_json(ref.run_dir / "kpi_metrics" / "derived_kpis.json")
        estimation = _read_json(ref.run_dir / "kpi_metrics" / "estimation_metrics.json")
        robustness = _read_json(ref.run_dir / "kpi_metrics" / "robustness_metrics.json")
        map_delivery = _read_json(ref.run_dir / "kpi_metrics" / "map_delivery.json")

        run_ok, run_error, ts_wall_end = _read_run_status(ref.run_dir)

        start_wall = _safe_float(manifest.get("timestamps", {}).get("start_wall"))
        wall_duration_s: Optional[float] = None
        if start_wall is not None and ts_wall_end is not None and ts_wall_end >= start_wall:
            wall_duration_s = float(ts_wall_end - start_wall)

        qos = (manifest.get("ros2") or {}).get("qos") if isinstance(manifest.get("ros2"), dict) else {}
        qos_rel = (qos or {}).get("reliability")
        qos_dur = (qos or {}).get("durability")
        qos_depth = (qos or {}).get("depth")

        impair = orch_cfg.get("impair") if isinstance(orch_cfg, dict) else {}
        impair_enabled = _as_bool((impair or {}).get("enabled"))

        uplink_b = _safe_float(derived.get("communication_uplink_bytes"))
        downlink_b = _safe_float(derived.get("communication_downlink_bytes"))
        total_b = None
        if uplink_b is not None or downlink_b is not None:
            total_b = float((uplink_b or 0.0) + (downlink_b or 0.0))

        total_mib = _bytes_to_mib(total_b)
        wall_total_mib_per_s: Optional[float] = None
        if wall_duration_s and wall_duration_s > 0 and total_mib is not None:
            wall_total_mib_per_s = float(total_mib / wall_duration_s)

        ate_by_robot = _extract_ate_by_robot(estimation)
        ate_vals = [v.get("rmse") for v in ate_by_robot.values() if _safe_float(v.get("rmse")) is not None]
        ate_rmse_mean = float(statistics.mean([float(v) for v in ate_vals])) if ate_vals else None
        matches = [int(v.get("matches") or 0) for v in ate_by_robot.values()]
        matches_total = int(sum(matches)) if matches else None
        matches_min = int(min(matches)) if matches else None
        matches_max = int(max(matches)) if matches else None

        delivery_rate = derived.get("delivery_rate") if isinstance(derived.get("delivery_rate"), dict) else {}
        delivery_rate_mean = _safe_float((delivery_rate or {}).get("mean"))

        topics = derived.get("delivery_topics") if isinstance(derived.get("delivery_topics"), dict) else {}
        factor_prefix = (manifest.get("ros2") or {}).get("factor_topic_prefix", "/c_slam/factor_batch")
        iface_prefix = (manifest.get("ros2") or {}).get("iface_topic_prefix", "/c_slam/iface")

        factor_rates: List[float] = []
        iface_rates: List[float] = []
        for topic, payload in topics.items():
            if not isinstance(payload, dict):
                continue
            rate = _safe_float(payload.get("delivery_rate"))
            if rate is None:
                continue
            if str(topic).startswith(str(factor_prefix)):
                factor_rates.append(float(rate))
            if str(topic).startswith(str(iface_prefix)):
                iface_rates.append(float(rate))

            delivery_rows.append(
                {
                    "scenario": ref.scenario,
                    "backend": ref.backend,
                    "topic": str(topic),
                    "delivery_rate": rate,
                    "attempts": _safe_float(payload.get("attempts")),
                    "delivered": _safe_float(payload.get("delivered")),
                    "drops": _safe_float(payload.get("drops")),
                }
            )

        factor_delivery_mean = float(statistics.mean(factor_rates)) if factor_rates else None
        factor_delivery_min = float(min(factor_rates)) if factor_rates else None
        iface_delivery_mean = float(statistics.mean(iface_rates)) if iface_rates else None
        iface_delivery_min = float(min(iface_rates)) if iface_rates else None

        opt = derived.get("optimization_duration_s") if isinstance(derived.get("optimization_duration_s"), dict) else {}
        loop = derived.get("loop_closure_correction_stabilised") if isinstance(derived.get("loop_closure_correction_stabilised"), dict) else {}
        iface_stab = derived.get("interface_correction_stabilised") if isinstance(derived.get("interface_correction_stabilised"), dict) else {}

        l_global_stats: Dict[str, Optional[float]] = {"l_global_count": 0, "l_global_mean_s": None, "l_global_p95_s": None}
        kpi_events = ref.run_dir / "kpi_metrics" / "kpi_events.jsonl"
        if ref.backend == "centralised" and kpi_events.exists():
            l_global_stats = _extract_l_global_from_kpi_events(kpi_events)

        # Map broadcast delivery is sender-side (published_minus_sender_drops) and does not measure receipt.
        map_topics = map_delivery.get("stats", {}).get("topics") if isinstance(map_delivery.get("stats"), dict) else {}
        map_bytes_attempted = None
        map_attempts = None
        if isinstance(map_topics, dict) and map_topics:
            map_bytes_attempted = float(sum(float(v.get("bytes_attempted") or 0) for v in map_topics.values() if isinstance(v, dict)))
            map_attempts = float(sum(float(v.get("attempts") or 0) for v in map_topics.values() if isinstance(v, dict)))

        run_rows.append(
            {
                "scenario": ref.scenario,
                "backend": ref.backend,
                "run_dir": str(ref.run_dir),
                "run_ok": run_ok,
                "run_error": run_error,
                "qos_reliability": qos_rel,
                "qos_durability": qos_dur,
                "qos_depth": qos_depth,
                "impair_enabled": impair_enabled,
                "wall_duration_s": wall_duration_s,
                "ate_rmse_mean": ate_rmse_mean,
                "ate_matches_total": matches_total,
                "ate_matches_min": matches_min,
                "ate_matches_max": matches_max,
                "delivery_rate_mean": delivery_rate_mean,
                "factor_delivery_mean": factor_delivery_mean,
                "factor_delivery_min": factor_delivery_min,
                "iface_delivery_mean": iface_delivery_mean,
                "iface_delivery_min": iface_delivery_min,
                "communication_uplink_bytes": uplink_b,
                "communication_downlink_bytes": downlink_b,
                "communication_total_bytes": total_b,
                "wall_total_mib_per_s": wall_total_mib_per_s,
                "time_to_global_convergence_s": _safe_float(derived.get("time_to_global_convergence_s")),
                "optimization_duration_mean_s": _safe_float((opt or {}).get("mean")),
                "optimization_duration_p95_s": _safe_float((opt or {}).get("p95")),
                "loop_closure_correction_stabilised_median_s": _safe_float((loop or {}).get("median")),
                "loop_closure_correction_stabilised_p90_s": _safe_float((loop or {}).get("p90")),
                "interface_correction_stabilised_median_s": _safe_float((iface_stab or {}).get("median")),
                "interface_correction_stabilised_p90_s": _safe_float((iface_stab or {}).get("p90")),
                "l_global_mean_s": l_global_stats.get("l_global_mean_s"),
                "l_global_p95_s": l_global_stats.get("l_global_p95_s"),
                "map_bytes_attempted": map_bytes_attempted,
                "map_attempts": map_attempts,
            }
        )

    # Write CSVs
    run_fields = [
        "scenario",
        "backend",
        "run_ok",
        "run_error",
        "qos_reliability",
        "qos_durability",
        "qos_depth",
        "impair_enabled",
        "wall_duration_s",
        "ate_rmse_mean",
        "ate_matches_total",
        "ate_matches_min",
        "ate_matches_max",
        "delivery_rate_mean",
        "factor_delivery_mean",
        "factor_delivery_min",
        "iface_delivery_mean",
        "iface_delivery_min",
        "communication_uplink_bytes",
        "communication_downlink_bytes",
        "communication_total_bytes",
        "wall_total_mib_per_s",
        "time_to_global_convergence_s",
        "optimization_duration_mean_s",
        "optimization_duration_p95_s",
        "l_global_mean_s",
        "l_global_p95_s",
        "loop_closure_correction_stabilised_median_s",
        "loop_closure_correction_stabilised_p90_s",
        "interface_correction_stabilised_median_s",
        "interface_correction_stabilised_p90_s",
        "map_bytes_attempted",
        "map_attempts",
        "run_dir",
    ]
    _write_csv(out_dir / "baseline_runs.csv", run_rows, run_fields)

    delivery_fields = ["scenario", "backend", "topic", "delivery_rate", "attempts", "delivered", "drops"]
    _write_csv(out_dir / "delivery_topics.csv", delivery_rows, delivery_fields)

    # Plots (reuse sweep plot utilities on a minimal row schema)
    plots: List[str] = []
    try:
        import tools.evaluate_sweep as es  # type: ignore

        baseline_rows = []
        for r in run_rows:
            baseline_rows.append(
                {
                    "sweep_type": "baseline",
                    "dataset_tag": r["scenario"],
                    "backend": r["backend"],
                    "run_ok": r["run_ok"],
                    "ate_rmse_mean": r["ate_rmse_mean"],
                    "ate_matches_total": r["ate_matches_total"],
                    "delivery_rate_mean": r["delivery_rate_mean"],
                    "communication_uplink_bytes": r["communication_uplink_bytes"],
                    "communication_downlink_bytes": r["communication_downlink_bytes"],
                    "optimization_duration_mean_s": r["optimization_duration_mean_s"],
                    "optimization_duration_p95_s": r["optimization_duration_p95_s"],
                    "loop_closure_correction_stabilised_median_s": r["loop_closure_correction_stabilised_median_s"],
                    "loop_closure_correction_stabilised_p90_s": r["loop_closure_correction_stabilised_p90_s"],
                    "time_to_global_convergence_s": r["time_to_global_convergence_s"],
                }
            )

        p = es._plot_baseline_ate(out_dir, baseline_rows)
        if p:
            plots.append(p)
        p = es._plot_baseline_comms_stacked_per_dataset(out_dir, baseline_rows)
        if p:
            plots.append(p)
        plots.extend(es._plot_metric_cdfs(out_dir, baseline_rows))
    except Exception:
        pass

    # Write a baseline-specific plot guide (avoid sweep wording).
    if plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        guide: List[str] = []
        guide.append("# Plot guide (baseline)")
        guide.append("")
        guide.append(f"This directory contains plots generated from `{baseline_dir}` (baseline runs; impairments disabled).")
        guide.append("")
        guide.append("## Recommended shortlist (storyline)")
        guide.append("- `baseline_ate_rmse_mean.png` — Baseline accuracy gap (ATE).")
        guide.append("- `baseline_comms_uplink_downlink_stacked.png` — Baseline bandwidth split (uplink vs downlink).")
        guide.append("- `cdf_optimization_duration_p95_s.png` — Timing tail: optimization duration p95.")
        guide.append("- `cdf_time_to_global_convergence_s.png` — Timing: time-to-convergence.")
        guide.append("")
        guide.append("## All plots")

        def add(fname: str, title: str, desc: str) -> None:
            if not (plots_dir / fname).exists():
                return
            guide.append(f"### `{fname}`")
            guide.append(f"**{title}.** {desc}")
            guide.append("")

        add(
            "baseline_ate_rmse_mean.png",
            "Baseline accuracy (ATE)",
            "Bar chart of baseline ATE-RMSE (mean across robots) per scenario for each backend.",
        )
        add(
            "baseline_comms_uplink_downlink_stacked.png",
            "Baseline communication split",
            "Stacked bars of uplink and downlink communication per scenario/backend (note centralised downlink is dominated by map broadcasts).",
        )
        for fname in sorted(p.name for p in plots_dir.glob("cdf_*.png")):
            add(fname, f"CDF: {fname.replace('cdf_', '').replace('.png', '')}", "Empirical CDF across successful runs.")

        (plots_dir / "plot.md").write_text("\n".join(guide) + "\n", encoding="utf-8")

    # Markdown report
    report: List[str] = []
    report.append(f"# Baseline evaluation: `{baseline_dir}`")
    report.append("")
    report.append("This report summarises the *baseline* runs (impairments disabled) and highlights the baseline QoS and timing/throughput envelopes that inform stress-test parameter choices.")
    report.append("")

    # Completeness
    report.append("## Run completeness")
    comp_rows: List[List[str]] = []
    for ref in runs:
        row = next((r for r in run_rows if r["scenario"] == ref.scenario and r["backend"] == ref.backend), None)
        ok = row.get("run_ok") if row else None
        status = "ok" if ok is True else ("failed" if ok is False else "missing")
        err = str(row.get("run_error") or "") if row else ""
        comp_rows.append([ref.scenario, ref.backend, status, err])
    report.append(_md_table(["scenario", "backend", "status", "error"], comp_rows))
    report.append("")
    report.append("Missing runs should be re-run before drawing backend-vs-backend conclusions for that scenario.")
    report.append("")

    # Baseline settings (expect consistent across runs)
    report.append("## Baseline QoS / impair settings (as executed)")
    uniq_qos = sorted({(str(r.get("qos_reliability")), str(r.get("qos_durability")), str(r.get("qos_depth"))) for r in run_rows})
    uniq_imp = sorted({str(r.get("impair_enabled")) for r in run_rows})
    report.append(f"- QoS values observed: `{uniq_qos}`")
    report.append(f"- Impair enabled observed: `{uniq_imp}`")
    report.append("")

    # Key comparison table
    report.append("## Key baseline comparisons (values relevant to QoS/impair stress framing)")
    headers = [
        "scenario",
        "backend",
        "wall (s)",
        "ATE mean (m)",
        "ATE matches (sum)",
        "delivery mean",
        "factor delivery (mean/min)",
        "iface delivery (mean/min)",
        "uplink (MiB)",
        "downlink (MiB)",
        "total (MiB)",
        "total (MiB/s, wall)",
        "T_conv (s)",
        "L_opt mean/p95 (s)",
        "L_global mean/p95 (s)",
        "T_LC med/p90 (s)",
        "T_iface med/p90 (s)",
    ]
    rows_md: List[List[str]] = []
    for r in run_rows:
        upl_m = _bytes_to_mib(_safe_float(r.get("communication_uplink_bytes")))
        dnl_m = _bytes_to_mib(_safe_float(r.get("communication_downlink_bytes")))
        tot_m = _bytes_to_mib(_safe_float(r.get("communication_total_bytes")))
        rows_md.append(
            [
                str(r.get("scenario") or ""),
                str(r.get("backend") or ""),
                _fmt(r.get("wall_duration_s"), digits=0),
                _fmt(r.get("ate_rmse_mean"), digits=3),
                _fmt(r.get("ate_matches_total"), digits=0),
                _fmt(r.get("delivery_rate_mean"), digits=3),
                f"{_fmt(r.get('factor_delivery_mean'), digits=3)}/{_fmt(r.get('factor_delivery_min'), digits=3)}",
                f"{_fmt(r.get('iface_delivery_mean'), digits=3)}/{_fmt(r.get('iface_delivery_min'), digits=3)}",
                _fmt(upl_m, digits=2),
                _fmt(dnl_m, digits=2),
                _fmt(tot_m, digits=2),
                _fmt(r.get("wall_total_mib_per_s"), digits=3),
                _fmt(r.get("time_to_global_convergence_s"), digits=3),
                f"{_fmt(r.get('optimization_duration_mean_s'), digits=4)}/{_fmt(r.get('optimization_duration_p95_s'), digits=4)}",
                f"{_fmt(r.get('l_global_mean_s'), digits=4)}/{_fmt(r.get('l_global_p95_s'), digits=4)}",
                f"{_fmt(r.get('loop_closure_correction_stabilised_median_s'), digits=3)}/{_fmt(r.get('loop_closure_correction_stabilised_p90_s'), digits=3)}",
                f"{_fmt(r.get('interface_correction_stabilised_median_s'), digits=3)}/{_fmt(r.get('interface_correction_stabilised_p90_s'), digits=3)}",
            ]
        )
    report.append(_md_table(headers, rows_md))
    report.append("")

    report.append("### Notes (interpretation-critical)")
    report.append("- `delivery mean` is the mean over *topics* in `derived_kpis.json` (not byte-weighted). Compare factor vs iface delivery separately for QoS/impair conclusions.")
    report.append("- `L_global` is only defined for centralised runs and is computed here from `kpi_events.jsonl` as `map_broadcast.ts - previous sensor_ingest.ts` (exact for batch_size=1).")
    report.append("- `downlink (MiB)` in centralised is dominated by map broadcasts (`/c_slam/map/*`); `map_delivery.json` is sender-side, so it does not prove robots actually received the data.")
    report.append("")

    report.append("## What baseline implies for stress-test values (QoS + impair)")
    report.append("Baseline already establishes the *traffic envelope* each architecture generates. Use this to choose impairment levels that are (a) realistic for your target network and (b) strong enough to cause measurable degradation.")
    report.append("")

    report.append("### 1) QoS stress framing (before impairments)")
    report.append("- Baseline QoS is `reliable / volatile / depth=20`. Despite this, some topics show low effective delivery (especially iface topics in decentralised, and one factor topic in centralised).")
    report.append("- Before adding harsher impairments, validate that the baseline delivery artefacts are understood (queue depth, subscriber readiness, or metric definition). Otherwise, impairment sweeps will be hard to interpret.")
    report.append("- Use the QoS sweep (reliability × depth) to pick a *stable operating point* per backend (e.g., require `factor_delivery_min ≥ 0.95` and `iface_delivery_min ≥ 0.95` if iface is a critical signal), then apply impairments around that point.")
    report.append("")

    report.append("### 2) Bandwidth-cap stress framing (from baseline throughput)")
    report.append("- Pick caps relative to baseline wall throughput:")
    report.append("  - Mild stress: cap ≈ 0.8× baseline")
    report.append("  - Moderate stress: cap ≈ 0.5× baseline")
    report.append("  - Severe stress: cap ≈ 0.2× baseline")
    report.append("- If you keep map downlink enabled, centralised has an inherently higher downlink demand; stress caps in the 0.5–2 Mbps range will be *extremely severe* for centralised and may mainly test map dissemination rather than optimisation itself.")
    report.append("")

    report.append("### 3) Loss / blackout stress framing (from baseline delivery)")
    report.append("- If baseline delivery is already low on a topic, adding random loss (2–10%) can be non-informative; you may hit a floor quickly.")
    report.append("- For informative stress sweeps, target regions where baseline delivery is high (near 1.0) and then apply loss/blackouts until delivery crosses key thresholds (0.9, 0.7, 0.5) while tracking success + ATE coverage.")
    report.append("")

    if plots:
        rel_plots = [str(Path(p).relative_to(out_dir)) for p in plots if p]
        report.append("## Plots")
        report.append("- See `plots/plot.md` for plot descriptions and a shortlist.")
        report.append("- Generated: " + ", ".join(f"`{p}`" for p in rel_plots))
        report.append("")

    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate baseline runs in out/new_eval/<run_id>/ (no sweep plan required).")
    ap.add_argument("--baseline-dir", required=True, help="Baseline directory (e.g., out/new_eval/20260130_132807)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <baseline-dir>/eval_baseline)")
    args = ap.parse_args(argv)

    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = baseline_dir / "eval_baseline"

    if not baseline_dir.exists():
        raise SystemExit(f"baseline dir does not exist: {baseline_dir}")

    evaluate_baseline(baseline_dir=baseline_dir, out_dir=out_dir)
    print(f"Wrote baseline evaluation to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
