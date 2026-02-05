#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
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


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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


def _dedupe_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return unique rows by run_id, plus list of duplicate run_ids seen."""
    by_id: Dict[str, Dict[str, Any]] = {}
    dupes: List[str] = []
    for row in rows:
        rid = str(row.get("run_id") or "").strip()
        if not rid:
            continue
        if rid in by_id:
            dupes.append(rid)
            # Prefer successful rows, else prefer rows with more filled fields.
            prev = by_id[rid]
            row_ok = _as_bool(row.get("run_ok"))
            prev_ok = _as_bool(prev.get("run_ok"))
            if prev_ok is True and row_ok is not True:
                continue
            if row_ok is True and prev_ok is not True:
                by_id[rid] = row
                continue
            prev_nonempty = sum(1 for v in prev.values() if str(v).strip())
            row_nonempty = sum(1 for v in row.values() if str(v).strip())
            if row_nonempty >= prev_nonempty:
                by_id[rid] = row
        else:
            by_id[rid] = row
    # Preserve a stable-ish order: sort by run_id.
    unique = [by_id[rid] for rid in sorted(by_id.keys())]
    return unique, sorted(set(dupes))


def _extract_ate_rmses(estimation_metrics: Any) -> List[float]:
    """Extract per-robot ATE RMSE values from either centralised or decentralised JSON shapes."""
    if not isinstance(estimation_metrics, dict):
        return []
    out: List[float] = []

    # Shape A (centralised): {"ate": {"a": {"rmse": ...}, ...}, "rpe": ...}
    ate = estimation_metrics.get("ate")
    if isinstance(ate, dict):
        for v in ate.values():
            if not isinstance(v, dict):
                continue
            rmse = _safe_float(v.get("rmse"))
            if rmse is not None:
                out.append(float(rmse))
        if out:
            return out

    # Shape B (decentralised): {"a": {"ate": {"rmse": ...}, "rpe": ...}, ...}
    for _robot, payload in estimation_metrics.items():
        if not isinstance(payload, dict):
            continue
        ate2 = payload.get("ate")
        if not isinstance(ate2, dict):
            continue
        rmse = _safe_float(ate2.get("rmse"))
        if rmse is not None:
            out.append(float(rmse))
    return out


def _extract_ate_by_robot(estimation_metrics: Any) -> Dict[str, Dict[str, Optional[float]]]:
    """Return robot_id -> {'rmse': float|None, 'matches': int|None} for both JSON shapes."""
    if not isinstance(estimation_metrics, dict):
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}

    ate = estimation_metrics.get("ate")
    if isinstance(ate, dict):
        # Centralised shape
        for rid, payload in ate.items():
            if not isinstance(payload, dict):
                continue
            out[str(rid)] = {
                "rmse": _safe_float(payload.get("rmse")),
                "matches": float(_safe_int(payload.get("matches")) or 0),
            }
        return out

    # Decentralised shape
    for rid, payload in estimation_metrics.items():
        if not isinstance(payload, dict):
            continue
        ate2 = payload.get("ate")
        if not isinstance(ate2, dict):
            continue
        out[str(rid)] = {
            "rmse": _safe_float(ate2.get("rmse")),
            "matches": float(_safe_int(ate2.get("matches")) or 0),
        }
    return out


def _extract_rpe_rmse_by_dt(estimation_metrics: Any) -> Dict[str, List[float]]:
    """Return dt->list[rpe_rmse] across robots (supports both JSON shapes)."""
    if not isinstance(estimation_metrics, dict):
        return {}
    out: Dict[str, List[float]] = {}

    # Shape A (centralised): {"rpe": {"a": {"1": {"rmse": ...}, ...}, ...}}
    rpe = estimation_metrics.get("rpe")
    if isinstance(rpe, dict):
        for v in rpe.values():
            if not isinstance(v, dict):
                continue
            for dt, metrics in v.items():
                if not isinstance(metrics, dict):
                    continue
                rmse = _safe_float(metrics.get("rmse"))
                if rmse is None:
                    continue
                out.setdefault(str(dt), []).append(float(rmse))
        if out:
            return out

    # Shape B (decentralised): {"a": {"rpe": {"1": {"rmse": ...}, ...}}, ...}
    for _robot, payload in estimation_metrics.items():
        if not isinstance(payload, dict):
            continue
        rpe2 = payload.get("rpe")
        if not isinstance(rpe2, dict):
            continue
        for dt, metrics in rpe2.items():
            if not isinstance(metrics, dict):
                continue
            rmse = _safe_float(metrics.get("rmse"))
            if rmse is None:
                continue
            out.setdefault(str(dt), []).append(float(rmse))
    return out


def _extract_rpe_by_robot_dt(estimation_metrics: Any) -> Dict[str, Dict[str, Optional[float]]]:
    """Return robot_id -> dt(str) -> rpe_rmse(float|None) for both JSON shapes."""
    if not isinstance(estimation_metrics, dict):
        return {}

    out: Dict[str, Dict[str, Optional[float]]] = {}

    # Centralised shape: {"rpe": {"a": {"1": {"rmse": ...}, ...}, ...}}
    rpe = estimation_metrics.get("rpe")
    if isinstance(rpe, dict):
        for rid, payload in rpe.items():
            if not isinstance(payload, dict):
                continue
            for dt, metrics in payload.items():
                if not isinstance(metrics, dict):
                    continue
                rmse = _safe_float(metrics.get("rmse"))
                out.setdefault(str(rid), {})[str(dt)] = rmse
        return out

    # Decentralised shape: {"a": {"rpe": {"1": {"rmse": ...}, ...}}, ...}
    for rid, payload in estimation_metrics.items():
        if not isinstance(payload, dict):
            continue
        rpe2 = payload.get("rpe")
        if not isinstance(rpe2, dict):
            continue
        for dt, metrics in rpe2.items():
            if not isinstance(metrics, dict):
                continue
            rmse = _safe_float(metrics.get("rmse"))
            out.setdefault(str(rid), {})[str(dt)] = rmse
    return out


def _load_resource_summary(run_dir: Path) -> Dict[str, Optional[float]]:
    path = run_dir / "kpi_metrics" / "resource_profile.json"
    doc = _read_json(path)
    summary = doc.get("summary") if isinstance(doc, dict) else None
    if not isinstance(summary, dict):
        return {}
    cpu = summary.get("cpu_process_pct") if isinstance(summary.get("cpu_process_pct"), dict) else {}
    rss = summary.get("rss_bytes") if isinstance(summary.get("rss_bytes"), dict) else {}
    cpu_sys = summary.get("cpu_system_pct") if isinstance(summary.get("cpu_system_pct"), dict) else {}
    return {
        "cpu_process_mean_pct": _safe_float(cpu.get("mean")),
        "cpu_process_p95_pct": _safe_float(cpu.get("p95")),
        "cpu_process_max_pct": _safe_float(cpu.get("max")),
        "cpu_system_mean_pct": _safe_float(cpu_sys.get("mean")),
        "rss_mean_bytes": _safe_float(rss.get("mean")),
        "rss_max_bytes": _safe_float(rss.get("max")),
    }


def _read_run_status(run_dir: Path) -> Tuple[Optional[bool], str]:
    path = run_dir / "run_status.json"
    if not path.exists():
        return None, "missing_run_status"
    doc = _read_json(path)
    ok = _as_bool(doc.get("ok"))
    err = doc.get("error")
    if err is None:
        err_s = ""
    else:
        err_s = str(err)
    return ok, err_s


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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _merge_row_from_derived(row: Dict[str, Any], derived: Dict[str, Any]) -> None:
    """Fill missing fields in a results row from derived_kpis.json."""

    def maybe_set_float(field: str, value: Any) -> None:
        if _safe_float(row.get(field)) is not None:
            return
        v = _safe_float(value)
        if v is not None:
            row[field] = v

    def maybe_set_str(field: str, value: Any) -> None:
        cur = str(row.get(field) or "").strip()
        if cur:
            return
        if value is None:
            return
        val = str(value).strip()
        if val:
            row[field] = val

    maybe_set_float("time_to_global_convergence_s", derived.get("time_to_global_convergence_s"))
    maybe_set_str("time_to_global_convergence_method", derived.get("time_to_global_convergence_method"))

    opt = derived.get("optimization_duration_s")
    if isinstance(opt, dict):
        maybe_set_float("optimization_duration_mean_s", opt.get("mean"))
        maybe_set_float("optimization_duration_p95_s", opt.get("p95"))

    loop = derived.get("loop_closure_correction_stabilised")
    if isinstance(loop, dict):
        maybe_set_float("loop_closure_correction_stabilised_median_s", loop.get("median"))
        maybe_set_float("loop_closure_correction_stabilised_p90_s", loop.get("p90"))

    maybe_set_float("communication_uplink_bytes", derived.get("communication_uplink_bytes"))
    maybe_set_float("communication_downlink_bytes", derived.get("communication_downlink_bytes"))
    maybe_set_float("communication_uplink_bytes_per_s", derived.get("communication_uplink_bytes_per_s"))
    maybe_set_float("communication_downlink_bytes_per_s", derived.get("communication_downlink_bytes_per_s"))

    delivery = derived.get("delivery_rate")
    if isinstance(delivery, dict):
        maybe_set_float("delivery_rate_mean", delivery.get("mean"))


def _plot_baseline_ate(out_dir: Path, baseline_rows: List[Dict[str, Any]]) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # dataset_tag -> backend -> ate_mean
    by_ds: Dict[str, Dict[str, float]] = {}
    for r in baseline_rows:
        if not _as_bool(r.get("run_ok")):
            continue
        ate = _safe_float(r.get("ate_rmse_mean"))
        if ate is None:
            continue
        ds = str(r.get("dataset_tag") or "")
        backend = str(r.get("backend") or "")
        by_ds.setdefault(ds, {})[backend] = float(ate)

    if not by_ds:
        return None

    datasets = sorted(by_ds.keys())
    central = [by_ds.get(ds, {}).get("centralised") for ds in datasets]
    decentral = [by_ds.get(ds, {}).get("decentralised") for ds in datasets]

    # Only keep datasets with at least one value
    kept: List[int] = [i for i, ds in enumerate(datasets) if central[i] is not None or decentral[i] is not None]
    if not kept:
        return None
    datasets = [datasets[i] for i in kept]
    central = [central[i] for i in kept]
    decentral = [decentral[i] for i in kept]

    x = list(range(len(datasets)))
    width = 0.38
    fig = plt.figure(figsize=(max(8.0, 0.8 * len(datasets)), 4.5))
    ax = fig.add_subplot(111)
    ax.bar([i - width / 2 for i in x], [v if v is not None else 0.0 for v in central], width, label="centralised")
    ax.bar([i + width / 2 for i in x], [v if v is not None else 0.0 for v in decentral], width, label="decentralised")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("ATE RMSE (mean across robots)")
    ax.set_title("Baseline ATE (successful runs only)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "baseline_ate_rmse_mean.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_tradeoff(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    pts = []
    for r in rows:
        if not _as_bool(r.get("run_ok")):
            continue
        ate = _safe_float(r.get("ate_rmse_mean"))
        up = _safe_float(r.get("communication_uplink_bytes"))
        dn = _safe_float(r.get("communication_downlink_bytes"))
        if ate is None or up is None or dn is None:
            continue
        pts.append((float(up + dn), float(ate), str(r.get("backend") or ""), str(r.get("sweep_type") or "")))
    if not pts:
        return None

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}
    markers = {"baseline": "o", "qos": "s", "impair": "^"}
    for backend in ("centralised", "decentralised"):
        for sweep in ("baseline", "qos", "impair"):
            xs = [p[0] / (1024 * 1024) for p in pts if p[2] == backend and p[3] == sweep]
            ys = [p[1] for p in pts if p[2] == backend and p[3] == sweep]
            if not xs:
                continue
            ax.scatter(xs, ys, s=26, c=colors.get(backend, None), marker=markers.get(sweep, "o"), alpha=0.8, label=f"{backend}-{sweep}")
    ax.set_xlabel("Total comms (MiB, uplink+downlink)")
    ax.set_ylabel("ATE RMSE (mean across robots)")
    ax.set_title("Accuracy vs communication (successful runs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "tradeoff_ate_vs_total_comms.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _write_plots_md(out_dir: Path, plots: Sequence[str]) -> Optional[str]:
    """Write plots/plot.md explaining each plot + a recommended shortlist."""
    if not plots:
        return None
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # filename -> (title, description)
    desc: Dict[str, Tuple[str, str]] = {
        "baseline_ate_rmse_mean.png": (
            "Baseline accuracy (ATE)",
            "Bar chart of baseline ATE-RMSE (mean across robots) per dataset for each backend. "
            "Use to discuss the accuracy gap under clean network conditions (Accuracy metrics: ATE).",
        ),
        "baseline_per_robot_ate_boxplot.png": (
            "Baseline per-robot accuracy spread",
            "Boxplot of per-robot ATE-RMSE across baseline runs (matches>0). "
            "Shows whether a backend's performance is consistent across agents (Accuracy metrics: ATE).",
        ),
        "per_robot_ate_boxplot_by_backend_sweep.png": (
            "Per-robot ATE across sweeps",
            "Boxplot of per-robot ATE-RMSE grouped by backend and sweep type (baseline/qos/impair). "
            "Highlights how network conditions broaden the accuracy distribution (Accuracy metrics: ATE).",
        ),
        "tradeoff_ate_vs_total_comms.png": (
            "Accuracy vs communication trade-off",
            "Scatter plot of ATE-RMSE (mean across robots) versus total communication (uplink+downlink). "
            "Use to narrate the core efficiency trade-off (Accuracy + Resource metrics: ATE and bandwidth).",
        ),
        "comms_uplink_downlink_stacked_mean.png": (
            "Uplink vs downlink communication (mean)",
            "Stacked bars of mean uplink and downlink communication per backend and sweep. "
            "Useful for attributing bandwidth cost to dissemination vs ingestion (Resource metrics: bandwidth).",
        ),
        "baseline_comms_uplink_downlink_stacked.png": (
            "Baseline uplink vs downlink by dataset",
            "Stacked bars of uplink and downlink communication for each baseline dataset/backend. "
            "Useful when discussing scaling and the centralised downlink cost (Resource metrics: bandwidth).",
        ),
        "success_rate_by_impairment.png": (
            "Impairment robustness (success rate)",
            "Bar chart of run success rate by impairment type and backend. "
            "Use as the top-level robustness summary (Robustness/operational outcome under Network impairments).",
        ),
        "qos_success_rate_heatmap.png": (
            "QoS robustness (success rate heatmap)",
            "Heatmap over QoS grid (reliability × depth) showing success rate per backend. "
            "Use to show which QoS settings are necessary for stable operation (QoS sensitivity).",
        ),
        "qos_heatmap_ate_rmse_mean.png": (
            "QoS vs accuracy heatmap",
            "Heatmap over QoS grid showing mean ATE-RMSE for successful runs. "
            "Separates 'it runs' from 'it is accurate' (Accuracy + QoS sensitivity).",
        ),
        "qos_heatmap_delivery_rate_mean.png": (
            "QoS vs delivery heatmap",
            "Heatmap over QoS grid showing mean delivery rate for successful runs. "
            "Connects transport settings to effective reliability (Resource metrics: delivery rate).",
        ),
        "qos_heatmap_total_comms_mib.png": (
            "QoS vs bandwidth heatmap",
            "Heatmap over QoS grid showing total communication (MiB) for successful runs. "
            "Shows when improved QoS increases bandwidth cost (Resource metrics: bandwidth).",
        ),
        "ate_vs_matches_scatter.png": (
            "Accuracy vs coverage (ATE matches)",
            "Scatter plot of ATE-RMSE versus total number of matched poses used in ATE evaluation. "
            "Critical to interpret low ATE under severe loss as 'partial coverage' (Accuracy metrics: ATE, coverage caveat).",
        ),
        "random_loss_coverage_and_success.png": (
            "Random loss: coverage + success",
            "Line plot showing how ATE match coverage and success rate change with random loss probability. "
            "Use to discuss failure modes under stochastic packet loss (Network impairments + accuracy coverage).",
        ),
        "bwcap_ate_and_success.png": (
            "Bandwidth caps: ATE + success",
            "Line plot showing how ATE and success rate vary with bandwidth cap (Mbps). "
            "Use to discuss delay/throughput constraints (Network impairments: bandwidth caps).",
        ),
        "tconv_vs_ate_scatter.png": (
            "Convergence time vs accuracy",
            "Scatter plot of time-to-convergence versus ATE-RMSE. "
            "Use to discuss speed/quality trade-offs (Timing + Accuracy metrics: T_conv and ATE).",
        ),
        "rpe_vs_delta.png": (
            "RPE vs horizon",
            "Line plot of mean RPE-RMSE as a function of horizon Δ, grouped by backend and sweep. "
            "Use to discuss short- vs long-horizon drift (Accuracy metrics: RPE).",
        ),
        "scaling_baseline_ate_rmse_mean.png": (
            "Scaling with team size: ATE",
            "Baseline line plot of ATE-RMSE versus team size N. Use to discuss scalability of accuracy (N scaling + ATE).",
        ),
        "scaling_baseline_communication_total_bytes.png": (
            "Scaling with team size: bandwidth",
            "Baseline line plot of total communication versus team size N. Use to discuss scalability of bandwidth cost (N scaling + BW).",
        ),
        "scaling_baseline_cpu_process_mean_pct.png": (
            "Scaling with team size: CPU",
            "Baseline line plot of mean process CPU utilisation versus team size N (successful runs). (Resource metrics: CPU).",
        ),
        "scaling_baseline_rss_max_bytes.png": (
            "Scaling with team size: memory",
            "Baseline line plot of peak RSS versus team size N (successful runs). (Resource metrics: memory).",
        ),
        "box_cpu_process_mean_pct.png": (
            "CPU distribution",
            "Boxplot of CPU mean (%) grouped by backend and sweep (successful runs). (Resource metrics: CPU).",
        ),
        "box_cpu_process_p95_pct.png": (
            "CPU tail distribution",
            "Boxplot of CPU p95 (%) grouped by backend and sweep (successful runs). (Resource metrics: CPU).",
        ),
        "box_rss_max_bytes.png": (
            "Memory distribution",
            "Boxplot of max RSS grouped by backend and sweep (successful runs). (Resource metrics: memory).",
        ),
    }

    # Handle any CDF plots generically.
    def explain(fname: str) -> Tuple[str, str]:
        if fname in desc:
            return desc[fname]
        if fname.startswith("cdf_") and fname.endswith(".png"):
            metric = fname[len("cdf_") : -len(".png")]
            return (
                f"CDF: {metric}",
                "Empirical CDF comparing distributions for successful runs. Useful to report variability and tails "
                "(Timing/Accuracy/Resource distributions depending on metric).",
            )
        return ("Plot", "Automatically generated plot from sweep evaluation.")

    rel_paths = sorted({Path(p).name for p in plots})
    lines: List[str] = []
    lines.append("# Plot guide")
    lines.append("")
    lines.append("This directory contains plots generated from `out/sweep_small/results.csv` and per-run KPI exports.")
    lines.append("Each plot maps to one or more response variables defined in `methodology.tex` (Timing, Accuracy, Resource metrics).")
    lines.append("")

    # Recommended shortlist for thesis narrative (6–10 figures).
    shortlist = [
        "baseline_ate_rmse_mean.png",
        "tradeoff_ate_vs_total_comms.png",
        "scaling_baseline_communication_total_bytes.png",
        "qos_success_rate_heatmap.png",
        "qos_heatmap_ate_rmse_mean.png",
        "success_rate_by_impairment.png",
        "ate_vs_matches_scatter.png",
        "cdf_optimization_duration_p95_s.png",
        "tconv_vs_ate_scatter.png",
    ]
    shortlisted = [f for f in shortlist if f in rel_paths]
    if shortlisted:
        lines.append("## Recommended shortlist (storyline)")
        lines.append("Suggested 6–10 figures ordered to match the methodology narrative (baseline → QoS → impairments → timing/compute):")
        for fname in shortlisted:
            title, blurb = explain(fname)
            lines.append(f"- `{fname}` — {title}. {blurb}")
        lines.append("")

    lines.append("## All plots")
    for fname in rel_paths:
        title, blurb = explain(fname)
        lines.append(f"### `{fname}`")
        lines.append(f"**{title}.** {blurb}")
        lines.append("")

    out_path = plots_dir / "plot.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(out_path)


def _ensure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_success_rate_by_impairment(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    imp = [r for r in rows if str(r.get("sweep_type")) == "impair" and str(r.get("impair_name") or "").strip()]
    if not imp:
        return None

    groups: Dict[Tuple[str, str], List[bool]] = defaultdict(list)  # (backend, impair_name) -> ok flags
    for r in imp:
        backend = str(r.get("backend") or "")
        name = str(r.get("impair_name") or "")
        ok = _as_bool(r.get("run_ok")) is True
        groups[(backend, name)].append(ok)

    names = sorted({name for (_backend, name) in groups.keys()})
    backends = ["centralised", "decentralised"]
    vals: Dict[str, List[float]] = {b: [] for b in backends}
    for name in names:
        for b in backends:
            flags = groups.get((b, name), [])
            rate = (sum(1 for f in flags if f) / float(len(flags))) if flags else 0.0
            vals[b].append(rate)

    x = list(range(len(names)))
    width = 0.38
    fig = plt.figure(figsize=(max(10.0, 0.7 * len(names)), 4.2))
    ax = fig.add_subplot(111)
    ax.bar([i - width / 2 for i in x], vals["centralised"], width, label="centralised")
    ax.bar([i + width / 2 for i in x], vals["decentralised"], width, label="decentralised")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Impairment sweep: success rate by impairment")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "success_rate_by_impairment.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_success_rate_qos_heatmap(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    import numpy as np
    from collections import defaultdict

    qos = [r for r in rows if str(r.get("sweep_type")) == "qos"]
    if not qos:
        return None

    rels = ["reliable", "best_effort"]
    depths = sorted({str(r.get("qos_depth") or "") for r in qos if str(r.get("qos_depth") or "").strip()}, key=lambda s: int(float(s)))
    if not depths:
        return None
    backends = ["centralised", "decentralised"]

    # backend -> rel -> depth -> list[ok]
    bucket: Dict[str, Dict[str, Dict[str, List[bool]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in qos:
        b = str(r.get("backend") or "")
        rel = str(r.get("qos_reliability") or "")
        d = str(r.get("qos_depth") or "")
        if b not in backends or rel not in rels or d not in depths:
            continue
        bucket[b][rel][d].append(_as_bool(r.get("run_ok")) is True)

    fig = plt.figure(figsize=(10.0, 3.6))
    for idx, backend in enumerate(backends, start=1):
        ax = fig.add_subplot(1, 2, idx)
        mat = np.zeros((len(rels), len(depths)), dtype=float)
        for i, rel in enumerate(rels):
            for j, d in enumerate(depths):
                flags = bucket[backend][rel][d]
                mat[i, j] = (sum(1 for f in flags if f) / float(len(flags))) if flags else float("nan")
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(list(range(len(depths))))
        ax.set_xticklabels(depths)
        ax.set_yticks(list(range(len(rels))))
        ax.set_yticklabels(rels)
        ax.set_title(f"QoS success rate: {backend}")
        ax.set_xlabel("depth")
        ax.set_ylabel("reliability")
        for i in range(len(rels)):
            for j in range(len(depths)):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=8)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label="success rate")
    fig.tight_layout(rect=[0, 0, 0.90, 1])

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "qos_success_rate_heatmap.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_qos_metric_heatmaps(out_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    """Heatmaps over QoS grid for key metrics (successful runs only)."""
    plt = _ensure_matplotlib()
    if plt is None:
        return []
    import numpy as np
    from collections import defaultdict

    qos = [r for r in rows if str(r.get("sweep_type")) == "qos"]
    if not qos:
        return []

    rels = ["reliable", "best_effort"]
    depths = sorted({str(r.get("qos_depth") or "") for r in qos if str(r.get("qos_depth") or "").strip()}, key=lambda s: int(float(s)))
    if not depths:
        return []
    backends = ["centralised", "decentralised"]

    def build_mat(field: str, *, transform=None) -> Dict[str, np.ndarray]:
        bucket: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in qos:
            if _as_bool(r.get("run_ok")) is not True:
                continue
            b = str(r.get("backend") or "")
            rel = str(r.get("qos_reliability") or "")
            d = str(r.get("qos_depth") or "")
            if b not in backends or rel not in rels or d not in depths:
                continue
            v = _safe_float(r.get(field))
            if v is None:
                continue
            if transform is not None:
                try:
                    v = float(transform(v))
                except Exception:
                    continue
            bucket[b][rel][d].append(float(v))

        mats: Dict[str, np.ndarray] = {}
        for backend in backends:
            mat = np.full((len(rels), len(depths)), float("nan"), dtype=float)
            for i, rel in enumerate(rels):
                for j, d in enumerate(depths):
                    vals = bucket[backend][rel][d]
                    if vals:
                        mat[i, j] = sum(vals) / len(vals)
            mats[backend] = mat
        return mats

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("ate_rmse_mean", "ATE RMSE (mean across robots)", None, "qos_heatmap_ate_rmse_mean.png", "{:.2f}"),
        ("delivery_rate_mean", "Delivery rate mean", None, "qos_heatmap_delivery_rate_mean.png", "{:.2f}"),
        ("communication_total_bytes", "Total comms (MiB, uplink+downlink)", lambda v: float(v) / (1024.0 * 1024.0), "qos_heatmap_total_comms_mib.png", "{:.0f}"),
    ]

    out_paths: List[str] = []
    for field, title, transform, fname, fmt in specs:
        mats = build_mat(field, transform=transform)
        if not mats:
            continue
        fig = plt.figure(figsize=(10.0, 3.6))
        im = None
        for idx, backend in enumerate(backends, start=1):
            ax = fig.add_subplot(1, 2, idx)
            mat = mats[backend]
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_xticks(list(range(len(depths))))
            ax.set_xticklabels(depths)
            ax.set_yticks(list(range(len(rels))))
            ax.set_yticklabels(rels)
            ax.set_title(f"{title}: {backend}")
            ax.set_xlabel("depth")
            ax.set_ylabel("reliability")
            finite = mat[np.isfinite(mat)]
            mid = float((finite.min() + finite.max()) / 2.0) if finite.size else 0.0
            for i in range(len(rels)):
                for j in range(len(depths)):
                    v = mat[i, j]
                    if not np.isfinite(v):
                        continue
                    ax.text(
                        j,
                        i,
                        fmt.format(v),
                        ha="center",
                        va="center",
                        color="white" if v < mid else "black",
                        fontsize=8,
                    )
        if im is None:
            plt.close(fig)
            continue
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax)
        fig.suptitle("QoS sweep heatmap (successful runs)")
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])
        out_path = plots_dir / fname
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def _plot_ate_vs_matches(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    pts = []
    for r in rows:
        if _as_bool(r.get("run_ok")) is not True:
            continue
        ate = _safe_float(r.get("ate_rmse_mean"))
        matches = _safe_float(r.get("ate_matches_total"))
        if ate is None or matches is None:
            continue
        pts.append((float(matches), float(ate), str(r.get("backend") or ""), str(r.get("sweep_type") or "")))
    if not pts:
        return None

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}
    markers = {"baseline": "o", "qos": "s", "impair": "^"}
    for backend in ("centralised", "decentralised"):
        for sweep in ("baseline", "qos", "impair"):
            xs = [p[0] for p in pts if p[2] == backend and p[3] == sweep]
            ys = [p[1] for p in pts if p[2] == backend and p[3] == sweep]
            if not xs:
                continue
            ax.scatter(xs, ys, s=26, c=colors.get(backend, None), marker=markers.get(sweep, "o"), alpha=0.75, label=f"{backend}-{sweep}")
    ax.set_xscale("log")
    ax.set_xlabel("ATE matched poses (sum across robots, log-scale)")
    ax.set_ylabel("ATE RMSE (mean across robots)")
    ax.set_title("Accuracy vs ATE coverage (successful runs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "ate_vs_matches_scatter.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_random_loss_coverage(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    imp = [r for r in rows if str(r.get("sweep_type")) == "impair" and str(r.get("impair_random_loss_p") or "").strip()]
    if not imp:
        return None

    # backend -> p -> list[matches]
    bucket: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    ok_bucket: Dict[str, Dict[float, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in imp:
        backend = str(r.get("backend") or "")
        p = _safe_float(r.get("impair_random_loss_p"))
        if p is None:
            continue
        ok = _as_bool(r.get("run_ok")) is True
        ok_bucket[backend][float(p)].append(ok)
        if ok:
            m = _safe_float(r.get("ate_matches_total"))
            if m is not None:
                bucket[backend][float(p)].append(float(m))

    ps = sorted({p for b in ok_bucket.values() for p in b.keys()})
    if not ps:
        return None

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    for backend, color in (("centralised", "#1f77b4"), ("decentralised", "#ff7f0e")):
        ys = []
        ok_rates = []
        for p in ps:
            vals = bucket[backend].get(p, [])
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
            flags = ok_bucket[backend].get(p, [])
            ok_rates.append(sum(1 for f in flags if f) / float(len(flags)) if flags else float("nan"))
        ax.plot(ps, ys, marker="o", color=color, label=f"{backend}: matches")
        # Add success rate on secondary axisotis
    ax2 = ax.twinx()
    for backend, color in (("centralised", "#1f77b4"), ("decentralised", "#ff7f0e")):
        ok_rates = []
        for p in ps:
            flags = ok_bucket[backend].get(p, [])
            ok_rates.append(sum(1 for f in flags if f) / float(len(flags)) if flags else float("nan"))
        ax2.plot(ps, ok_rates, linestyle="--", marker="x", color=color, alpha=0.7, label=f"{backend}: success")

    ax.set_xlabel("random loss probability p")
    ax.set_ylabel("Mean ATE matched poses (sum across robots)")
    ax.set_yscale("log")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0.0, 1.05)
    ax.set_title("Random loss: coverage and success vs loss probability")
    ax.grid(alpha=0.3)

    # Combined legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, ncol=1, frameon=True, loc="best")
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "random_loss_coverage_and_success.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_bwcap_effects(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Bandwidth caps: ATE and success vs cap."""
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    imp = [r for r in rows if str(r.get("sweep_type")) == "impair" and str(r.get("impair_bw_cap_mbps") or "").strip()]
    if not imp:
        return None

    bucket_ate: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    bucket_ok: Dict[str, Dict[float, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in imp:
        backend = str(r.get("backend") or "")
        cap = _safe_float(r.get("impair_bw_cap_mbps"))
        if cap is None:
            continue
        ok = _as_bool(r.get("run_ok")) is True
        bucket_ok[backend][float(cap)].append(ok)
        if ok:
            ate = _safe_float(r.get("ate_rmse_mean"))
            if ate is not None:
                bucket_ate[backend][float(cap)].append(float(ate))

    caps = sorted({c for b in bucket_ok.values() for c in b.keys()})
    if not caps:
        return None

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    for backend, color in (("centralised", "#1f77b4"), ("decentralised", "#ff7f0e")):
        ys = []
        ok_rates = []
        for c in caps:
            vals = bucket_ate[backend].get(c, [])
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
            flags = bucket_ok[backend].get(c, [])
            ok_rates.append(sum(1 for f in flags if f) / float(len(flags)) if flags else float("nan"))
        ax.plot(caps, ys, marker="o", color=color, label=f"{backend}: ATE")
        ax2.plot(caps, ok_rates, linestyle="--", marker="x", color=color, alpha=0.7, label=f"{backend}: success")

    ax.set_xlabel("Bandwidth cap (Mbps)")
    ax.set_ylabel("Mean ATE RMSE (mean across robots)")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0.0, 1.05)
    ax.set_title("Bandwidth caps: ATE and success vs cap")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, ncol=1, frameon=True, loc="best")
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "bwcap_ate_and_success.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_rpe_vs_delta(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    deltas = ["1", "10", "50"]
    # backend,sweep -> dt -> values
    bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if _as_bool(r.get("run_ok")) is not True:
            continue
        backend = str(r.get("backend") or "")
        sweep = str(r.get("sweep_type") or "")
        for dt in deltas:
            v = _safe_float(r.get(f"rpe_rmse_mean_dt{dt}"))
            if v is not None:
                bucket[(backend, sweep)][dt].append(float(v))
    if not bucket:
        return None

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}
    linestyles = {"baseline": "-", "qos": "--", "impair": ":"}
    for (backend, sweep), dtmap in sorted(bucket.items()):
        ys = []
        xs = []
        for dt in deltas:
            vals = dtmap.get(dt, [])
            if not vals:
                ys.append(float("nan"))
            else:
                ys.append(sum(vals) / len(vals))
            xs.append(int(dt))
        ax.plot(xs, ys, marker="o", color=colors.get(backend, None), linestyle=linestyles.get(sweep, "-"), alpha=0.9, label=f"{backend}-{sweep}")
    ax.set_xlabel(r"RPE horizon $\Delta$")
    ax.set_ylabel("Mean RPE RMSE (mean across robots, then across runs)")
    ax.set_title("RPE vs horizon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "rpe_vs_delta.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_metric_cdfs(out_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    """Generate CDF plots for key timing metrics using per-run summary values."""
    plt = _ensure_matplotlib()
    if plt is None:
        return []
    import numpy as np

    specs = [
        ("optimization_duration_mean_s", "Optimization duration mean (s)"),
        ("optimization_duration_p95_s", "Optimization duration p95 (s)"),
        ("loop_closure_correction_stabilised_median_s", "Loop-closure correction stabilised median (s)"),
        ("loop_closure_correction_stabilised_p90_s", "Loop-closure correction stabilised p90 (s)"),
        ("time_to_global_convergence_s", "Time-to-convergence (s)"),
        ("delivery_rate_mean", "Delivery rate mean"),
        ("ate_matches_total", "ATE matched poses (sum across robots)"),
    ]
    out_paths: List[str] = []
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}

    for field, title in specs:
        data: Dict[str, List[float]] = {"centralised": [], "decentralised": []}
        for r in rows:
            if _as_bool(r.get("run_ok")) is not True:
                continue
            v = _safe_float(r.get(field))
            if v is None:
                continue
            backend = str(r.get("backend") or "")
            if backend in data:
                data[backend].append(float(v))
        if not any(data.values()):
            continue

        fig = plt.figure(figsize=(6.6, 4.2))
        ax = fig.add_subplot(111)
        for backend, vals in data.items():
            if not vals:
                continue
            xs = np.sort(np.array(vals, dtype=float))
            ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
            ax.plot(xs, ys, color=colors.get(backend, None), label=backend)
        ax.set_xlabel(title)
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF: {title} (successful runs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2, frameon=True)
        fig.tight_layout()

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"cdf_{field}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def _plot_tconv_vs_ate(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    pts = []
    for r in rows:
        if _as_bool(r.get("run_ok")) is not True:
            continue
        ate = _safe_float(r.get("ate_rmse_mean"))
        t = _safe_float(r.get("time_to_global_convergence_s"))
        if ate is None or t is None:
            continue
        pts.append((float(t), float(ate), str(r.get("backend") or ""), str(r.get("sweep_type") or "")))
    if not pts:
        return None

    fig = plt.figure(figsize=(6.6, 4.5))
    ax = fig.add_subplot(111)
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}
    markers = {"baseline": "o", "qos": "s", "impair": "^"}
    for backend in ("centralised", "decentralised"):
        for sweep in ("baseline", "qos", "impair"):
            xs = [p[0] for p in pts if p[2] == backend and p[3] == sweep]
            ys = [p[1] for p in pts if p[2] == backend and p[3] == sweep]
            if not xs:
                continue
            ax.scatter(xs, ys, s=26, c=colors.get(backend, None), marker=markers.get(sweep, "o"), alpha=0.75, label=f"{backend}-{sweep}")
    ax.set_xlabel(r"$T_{conv}$ (s)")
    ax.set_ylabel("ATE RMSE (mean across robots)")
    ax.set_title("Time-to-convergence vs accuracy (successful runs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "tconv_vs_ate_scatter.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_comms_uplink_downlink(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    # Aggregate mean uplink/downlink (MiB) per backend+sweep for successful runs.
    bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {"up": [], "dn": []})
    for r in rows:
        if _as_bool(r.get("run_ok")) is not True:
            continue
        up = _safe_float(r.get("communication_uplink_bytes"))
        dn = _safe_float(r.get("communication_downlink_bytes"))
        if up is None or dn is None:
            continue
        key = (str(r.get("backend") or ""), str(r.get("sweep_type") or ""))
        bucket[key]["up"].append(float(up) / (1024.0 * 1024.0))
        bucket[key]["dn"].append(float(dn) / (1024.0 * 1024.0))
    if not bucket:
        return None

    keys = sorted(bucket.keys())
    labels = [f"{b}-{s}" for (b, s) in keys]
    up_means = [sum(bucket[k]["up"]) / len(bucket[k]["up"]) for k in keys]
    dn_means = [sum(bucket[k]["dn"]) / len(bucket[k]["dn"]) for k in keys]

    x = list(range(len(keys)))
    fig = plt.figure(figsize=(max(8.0, 0.8 * len(keys)), 4.2))
    ax = fig.add_subplot(111)
    ax.bar(x, up_means, label="uplink (MiB)")
    ax.bar(x, dn_means, bottom=up_means, label="downlink (MiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean communication (MiB)")
    ax.set_title("Communication breakdown (mean over successful runs)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "comms_uplink_downlink_stacked_mean.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_baseline_comms_stacked_per_dataset(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    base = [r for r in rows if str(r.get("sweep_type")) == "baseline" and _as_bool(r.get("run_ok")) is True]
    if not base:
        return None

    items = []
    for r in base:
        up = _safe_float(r.get("communication_uplink_bytes"))
        dn = _safe_float(r.get("communication_downlink_bytes"))
        if up is None or dn is None:
            continue
        label = f"{r.get('dataset_tag','')}-{r.get('backend','')}"
        items.append((label, float(up) / (1024.0 * 1024.0), float(dn) / (1024.0 * 1024.0)))
    if not items:
        return None
    items.sort(key=lambda t: t[0])

    labels = [t[0] for t in items]
    ups = [t[1] for t in items]
    dns = [t[2] for t in items]
    x = list(range(len(labels)))

    fig = plt.figure(figsize=(max(10.0, 0.6 * len(labels)), 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x, ups, label="uplink (MiB)")
    ax.bar(x, dns, bottom=ups, label="downlink (MiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Communication (MiB)")
    ax.set_title("Baseline: uplink vs downlink per dataset/backend (successful runs)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2, frameon=True)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "baseline_comms_uplink_downlink_stacked.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_scaling_vs_team_size(out_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return []
    from collections import defaultdict
    import numpy as np

    base = [r for r in rows if str(r.get("sweep_type")) == "baseline" and _as_bool(r.get("run_ok")) is True]
    if not base:
        return []

    metrics = [
        ("ate_rmse_mean", "ATE RMSE (mean across robots)", False),
        ("communication_total_bytes", "Total communication (MiB)", True),
        ("cpu_process_mean_pct", "CPU process mean (%)", False),
        ("rss_max_bytes", "RSS max (MiB)", True),
    ]

    out_paths: List[str] = []
    backends = ["centralised", "decentralised"]
    colors = {"centralised": "#1f77b4", "decentralised": "#ff7f0e"}

    for field, title, bytes_to_mib in metrics:
        data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in base:
            backend = str(r.get("backend") or "")
            if backend not in backends:
                continue
            n = _safe_int(r.get("team_size"))
            if n is None:
                continue
            v = _safe_float(r.get(field))
            if v is None:
                continue
            if bytes_to_mib and field.endswith("_bytes"):
                v = float(v) / (1024.0 * 1024.0)
            elif bytes_to_mib and field.endswith("_bytes") is False and field.startswith("rss_"):
                v = float(v) / (1024.0 * 1024.0)
            data[backend][int(n)].append(float(v))
        if not any(data.values()):
            continue

        fig = plt.figure(figsize=(6.6, 4.2))
        ax = fig.add_subplot(111)
        for backend in backends:
            xs = sorted(data[backend].keys())
            if not xs:
                continue
            ys = [sum(data[backend][x]) / len(data[backend][x]) for x in xs]
            ax.plot(xs, ys, marker="o", color=colors.get(backend, None), label=backend)
        ax.set_xlabel("Team size N")
        ax.set_ylabel(title)
        ax.set_title(f"Baseline scaling: {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2, frameon=True)
        fig.tight_layout()

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"scaling_baseline_{field}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def _plot_resource_boxplots(out_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return []
    import numpy as np

    specs = [
        ("cpu_process_mean_pct", "CPU process mean (%)"),
        ("cpu_process_p95_pct", "CPU process p95 (%)"),
        ("rss_max_bytes", "RSS max (MiB)"),
    ]
    backends = ["centralised", "decentralised"]
    sweeps = ["baseline", "qos", "impair"]

    out_paths: List[str] = []
    for field, title in specs:
        groups = []
        labels = []
        for backend in backends:
            for sweep in sweeps:
                vals = []
                for r in rows:
                    if _as_bool(r.get("run_ok")) is not True:
                        continue
                    if str(r.get("backend") or "") != backend:
                        continue
                    if str(r.get("sweep_type") or "") != sweep:
                        continue
                    v = _safe_float(r.get(field))
                    if v is None:
                        continue
                    if field.endswith("_bytes"):
                        v = float(v) / (1024.0 * 1024.0)
                    vals.append(float(v))
                if vals:
                    groups.append(vals)
                    labels.append(f"{backend}-{sweep}")
        if not groups:
            continue

        fig = plt.figure(figsize=(max(10.0, 0.7 * len(labels)), 4.5))
        ax = fig.add_subplot(111)
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Resource distribution: {title} (successful runs)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"box_{field}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def _plot_per_robot_ate_boxplot_baseline(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    base = [r for r in rows if str(r.get("sweep_type")) == "baseline" and _as_bool(r.get("run_ok")) is True]
    if not base:
        return None

    by_backend: Dict[str, List[float]] = defaultdict(list)
    for r in base:
        backend = str(r.get("backend") or "")
        payload = r.get("_ate_by_robot")
        if not isinstance(payload, dict):
            continue
        for _rid, m in payload.items():
            if not isinstance(m, dict):
                continue
            v = _safe_float(m.get("rmse"))
            matches = _safe_float(m.get("matches"))
            if v is None or matches is None or matches <= 0:
                continue
            by_backend[backend].append(float(v))
    if not any(by_backend.values()):
        return None

    labels = []
    groups = []
    for backend in ("centralised", "decentralised"):
        vals = by_backend.get(backend, [])
        if vals:
            labels.append(backend)
            groups.append(vals)
    if not groups:
        return None

    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111)
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_ylabel("Per-robot ATE RMSE")
    ax.set_title("Baseline: per-robot ATE distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "baseline_per_robot_ate_boxplot.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _plot_per_robot_ate_boxplot_groups(out_dir: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Per-robot ATE RMSE distribution grouped by backend+sweep (successful runs, matches>0)."""
    plt = _ensure_matplotlib()
    if plt is None:
        return None
    from collections import defaultdict

    bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if _as_bool(r.get("run_ok")) is not True:
            continue
        backend = str(r.get("backend") or "")
        sweep = str(r.get("sweep_type") or "")
        payload = r.get("_ate_by_robot")
        if not isinstance(payload, dict):
            continue
        for _rid, m in payload.items():
            if not isinstance(m, dict):
                continue
            rmse = _safe_float(m.get("rmse"))
            matches = _safe_float(m.get("matches"))
            if rmse is None or matches is None or matches <= 0:
                continue
            bucket[(backend, sweep)].append(float(rmse))
    if not bucket:
        return None

    keys = sorted(bucket.keys())
    labels = [f"{b}-{s}" for (b, s) in keys]
    groups = [bucket[k] for k in keys]
    fig = plt.figure(figsize=(max(10.0, 0.7 * len(labels)), 4.6))
    ax = fig.add_subplot(111)
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Per-robot ATE RMSE")
    ax.set_title("Per-robot ATE distribution by backend and sweep (successful runs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "per_robot_ate_boxplot_by_backend_sweep.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate an out/sweep_* directory (results.csv + run outputs).")
    ap.add_argument("--sweep-dir", required=True, help="Path to sweep directory (contains results.csv + run_plan.json)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <sweep-dir>/eval)")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser()
    if not sweep_dir.is_absolute():
        sweep_dir = (ROOT / sweep_dir).resolve()
    results_csv = sweep_dir / "results.csv"
    run_plan_json = sweep_dir / "run_plan.json"
    if not results_csv.exists():
        raise SystemExit(f"Missing results.csv: {results_csv}")
    if not run_plan_json.exists():
        raise SystemExit(f"Missing run_plan.json: {run_plan_json}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (sweep_dir / "eval")
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows, dupes = _dedupe_rows(raw_rows)

    plan = _read_json(run_plan_json)
    planned_count = _safe_int(plan.get("count"))

    enriched: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    fail_reasons: Dict[str, int] = {}

    for r in rows:
        run_dir_raw = str(r.get("run_dir") or "").strip()
        run_dir = Path(run_dir_raw).expanduser() if run_dir_raw else None
        if run_dir and not run_dir.is_absolute():
            run_dir = (ROOT / run_dir).resolve()

        ok, err = (None, "missing_run_dir")
        if run_dir and run_dir.exists():
            ok, err = _read_run_status(run_dir)
        else:
            ok = _as_bool(r.get("run_ok"))
            err = str(r.get("run_error") or "missing_run_dir")

        if ok is True:
            status_counts["ok"] = status_counts.get("ok", 0) + 1
        else:
            status_counts["fail"] = status_counts.get("fail", 0) + 1
            fail_reasons[str(err or "unknown")] = fail_reasons.get(str(err or "unknown"), 0) + 1

        derived = _read_json(run_dir / "kpi_metrics" / "derived_kpis.json") if run_dir else {}
        est = derived.get("estimation_metrics")
        if not isinstance(est, dict) and run_dir:
            est = _read_json(run_dir / "kpi_metrics" / "estimation_metrics.json")
        ate_vals = _extract_ate_rmses(est)
        rpe_by_dt = _extract_rpe_rmse_by_dt(est)
        ate_by_robot = _extract_ate_by_robot(est)
        rpe_by_robot_dt = _extract_rpe_by_robot_dt(est)
        resource = _load_resource_summary(run_dir) if run_dir else {}

        row = dict(r)
        row["run_ok"] = ok
        row["run_error"] = err
        if derived:
            _merge_row_from_derived(row, derived)

        if ate_vals:
            ate_vals.sort()
            row["ate_rmse_n"] = len(ate_vals)
            row["ate_rmse_mean"] = sum(ate_vals) / float(len(ate_vals))
            row["ate_rmse_median"] = ate_vals[len(ate_vals) // 2]
        else:
            row["ate_rmse_n"] = 0
            row["ate_rmse_mean"] = _safe_float(r.get("ate_rmse_mean"))
            row["ate_rmse_median"] = _safe_float(r.get("ate_rmse_median"))

        # ATE coverage (matches) summary
        if ate_by_robot:
            matches = [m.get("matches") for m in ate_by_robot.values() if isinstance(m, dict) and _safe_float(m.get("matches")) is not None]
            matches_f = [float(v) for v in matches if v is not None]
            if matches_f:
                row["ate_matches_total"] = float(sum(matches_f))
                row["ate_matches_min"] = float(min(matches_f))
                row["ate_matches_max"] = float(max(matches_f))
        # Keep raw per-robot metrics in-memory for plotting only (not exported to CSV).
        row["_ate_by_robot"] = ate_by_robot
        row["_rpe_by_robot_dt"] = rpe_by_robot_dt

        for dt, vals in rpe_by_dt.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            row[f"rpe_rmse_mean_dt{dt}"] = sum(vals_sorted) / float(len(vals_sorted))
            row[f"rpe_rmse_median_dt{dt}"] = vals_sorted[len(vals_sorted) // 2]

        row.update(resource)

        # Convenience totals for plotting/summary.
        up = _safe_float(row.get("communication_uplink_bytes"))
        dn = _safe_float(row.get("communication_downlink_bytes"))
        if up is not None and dn is not None:
            row["communication_total_bytes"] = float(up + dn)
        enriched.append(row)

    # Write per-run evaluation CSV
    base_fields = [
        "run_id",
        "config_hash",
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
        "impair_random_loss_p",
        "impair_burst_period_s",
        "impair_burst_duration_s",
        "impair_bw_cap_mbps",
        "impair_blackout_mode",
        "impair_blackout_start_s",
        "impair_blackout_end_s",
        "run_dir",
        "run_ok",
        "run_error",
        "time_to_global_convergence_s",
        "time_to_global_convergence_method",
        "optimization_duration_mean_s",
        "optimization_duration_p95_s",
        "loop_closure_correction_stabilised_median_s",
        "loop_closure_correction_stabilised_p90_s",
        "communication_uplink_bytes",
        "communication_downlink_bytes",
        "communication_total_bytes",
        "communication_uplink_bytes_per_s",
        "communication_downlink_bytes_per_s",
        "delivery_rate_mean",
        "ate_matches_total",
        "ate_matches_min",
        "ate_matches_max",
        "ate_rmse_n",
        "ate_rmse_mean",
        "ate_rmse_median",
        "rpe_rmse_mean_dt1",
        "rpe_rmse_mean_dt10",
        "rpe_rmse_mean_dt50",
        "cpu_process_mean_pct",
        "cpu_process_p95_pct",
        "cpu_process_max_pct",
        "cpu_system_mean_pct",
        "rss_mean_bytes",
        "rss_max_bytes",
    ]
    eval_csv = out_dir / "eval_runs.csv"
    _write_csv(eval_csv, enriched, base_fields)

    # Baseline table
    baseline_rows = [r for r in enriched if str(r.get("sweep_type")) == "baseline"]
    baseline_csv = out_dir / "tables" / "baseline_by_dataset_backend.csv"
    _write_csv(baseline_csv, baseline_rows, base_fields)

    # QoS table
    qos_rows = [r for r in enriched if str(r.get("sweep_type")) == "qos"]
    qos_csv = out_dir / "tables" / "qos_by_combo.csv"
    _write_csv(qos_csv, qos_rows, base_fields)

    # Impair table (replicates)
    impair_rows = [r for r in enriched if str(r.get("sweep_type")) == "impair"]
    impair_csv = out_dir / "tables" / "impair_by_scenario_seed.csv"
    _write_csv(impair_csv, impair_rows, base_fields)

    # Impair aggregated across seeds
    def group_key(r: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        return (
            str(r.get("dataset_tag") or ""),
            str(r.get("backend") or ""),
            str(r.get("impair_name") or ""),
            str(r.get("qos_reliability") or ""),
            str(r.get("qos_depth") or ""),
        )

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in impair_rows:
        groups.setdefault(group_key(r), []).append(r)

    impair_agg: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        ds, backend, name, rel, depth = key
        ok_flags = [_as_bool(it.get("run_ok")) is True for it in items]
        ate_vals = [_safe_float(it.get("ate_rmse_mean")) for it in items if _as_bool(it.get("run_ok")) is True]
        conv_vals = [
            _safe_float(it.get("time_to_global_convergence_s")) for it in items if _as_bool(it.get("run_ok")) is True
        ]
        delivery_vals = [_safe_float(it.get("delivery_rate_mean")) for it in items if _as_bool(it.get("run_ok")) is True]
        comm_vals = [
            _safe_float(it.get("communication_total_bytes")) for it in items if _as_bool(it.get("run_ok")) is True
        ]
        row = {
            "dataset_tag": ds,
            "backend": backend,
            "impair_name": name,
            "qos_reliability": rel,
            "qos_depth": depth,
            "replicates": len(items),
            "ok_count": sum(1 for f in ok_flags if f),
            "ok_rate": (sum(1 for f in ok_flags if f) / float(len(ok_flags))) if ok_flags else None,
            "ate_rmse_mean_mean": _stats([v for v in ate_vals if v is not None]).get("mean"),
            "ate_rmse_mean_p95": _stats([v for v in ate_vals if v is not None]).get("p95"),
            "time_to_global_convergence_s_mean": _stats([v for v in conv_vals if v is not None]).get("mean"),
            "delivery_rate_mean_mean": _stats([v for v in delivery_vals if v is not None]).get("mean"),
            "communication_total_mib_mean": _bytes_to_mib(_stats([v for v in comm_vals if v is not None]).get("mean")),
        }
        impair_agg.append(row)

    impair_agg_csv = out_dir / "tables" / "impair_aggregated.csv"
    impair_agg_fields = list(impair_agg[0].keys()) if impair_agg else []
    if impair_agg_fields:
        _write_csv(impair_agg_csv, impair_agg, impair_agg_fields)

    # Summary JSON
    summary = {
        "sweep_dir": str(sweep_dir),
        "planned_count": planned_count,
        "unique_results": len(rows),
        "raw_result_rows": len(raw_rows),
        "duplicate_run_ids": dupes,
        "status_counts": status_counts,
        "top_fail_reasons": sorted(fail_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }
    _write_json(out_dir / "summary.json", summary)

    # Markdown report (short, links to CSVs + plots).
    lines: List[str] = []
    lines.append(f"# Sweep evaluation: `{sweep_dir.name}`")
    lines.append("")
    lines.append(f"- Sweep dir: `{sweep_dir}`")
    lines.append(f"- Planned runs (run_plan.json): `{planned_count}`")
    lines.append(f"- Unique run_ids in results.csv: `{len(rows)}` (raw rows: `{len(raw_rows)}`)")
    lines.append(f"- OK / FAIL: `{status_counts.get('ok', 0)}` / `{status_counts.get('fail', 0)}`")
    if dupes:
        lines.append(f"- Duplicate run_ids in results.csv: `{len(dupes)}` (see `summary.json`)")  # keep short
    lines.append("")
    if fail_reasons:
        lines.append("## Failures")
        top = sorted(fail_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for reason, count in top:
            lines.append(f"- `{count}`: `{reason}`")
        lines.append("")

    # Baseline comparison table (OK only)
    lines.append("## Baseline (OK runs)")
    base_ok = [r for r in baseline_rows if _as_bool(r.get("run_ok")) is True]
    base_ok.sort(key=lambda r: (str(r.get("dataset_tag") or ""), str(r.get("backend") or "")))
    table_rows = []
    for r in base_ok:
        total_mib = _bytes_to_mib(_safe_float(r.get("communication_total_bytes")))
        table_rows.append(
            [
                str(r.get("dataset_tag") or ""),
                str(r.get("backend") or ""),
                _fmt(r.get("ate_rmse_mean")),
                _fmt(r.get("time_to_global_convergence_s")),
                _fmt(total_mib, digits=1),
                _fmt(r.get("delivery_rate_mean")),
            ]
        )
    if table_rows:
        lines.append(
            _md_table(
                ["dataset_tag", "backend", "ate_rmse_mean", "t_conv_s", "comms_total_mib", "delivery_mean"],
                table_rows,
            )
        )
    else:
        lines.append("_No successful baseline runs found._")
    lines.append("")

    if not args.no_plots:
        plots: List[str] = []
        p1 = _plot_baseline_ate(out_dir, baseline_rows)
        if p1:
            plots.append(p1)
        p2 = _plot_tradeoff(out_dir, enriched)
        if p2:
            plots.append(p2)
        for p in (
            _plot_success_rate_by_impairment(out_dir, enriched),
            _plot_success_rate_qos_heatmap(out_dir, enriched),
            _plot_ate_vs_matches(out_dir, enriched),
            _plot_random_loss_coverage(out_dir, enriched),
            _plot_bwcap_effects(out_dir, enriched),
            _plot_tconv_vs_ate(out_dir, enriched),
            _plot_comms_uplink_downlink(out_dir, enriched),
            _plot_baseline_comms_stacked_per_dataset(out_dir, enriched),
            _plot_rpe_vs_delta(out_dir, enriched),
            _plot_per_robot_ate_boxplot_baseline(out_dir, enriched),
            _plot_per_robot_ate_boxplot_groups(out_dir, enriched),
        ):
            if p:
                plots.append(p)
        plots.extend(_plot_qos_metric_heatmaps(out_dir, enriched))
        plots.extend(_plot_metric_cdfs(out_dir, enriched))
        plots.extend(_plot_scaling_vs_team_size(out_dir, enriched))
        plots.extend(_plot_resource_boxplots(out_dir, enriched))

        if plots:
            plot_md = _write_plots_md(out_dir, plots)
            lines.append("## Plots")
            if plot_md:
                lines.append(f"- Guide: `{Path(plot_md).relative_to(out_dir)}`")
            for p in sorted(set(plots)):
                lines.append(f"- `{Path(p).relative_to(out_dir)}`")
            lines.append("")

    lines.append("## Outputs")
    lines.append(f"- Per-run table: `{eval_csv.relative_to(sweep_dir)}`")
    lines.append(f"- Baseline subset: `{baseline_csv.relative_to(sweep_dir)}`")
    lines.append(f"- QoS subset: `{qos_csv.relative_to(sweep_dir)}`")
    lines.append(f"- Impair subset: `{impair_csv.relative_to(sweep_dir)}`")
    if impair_agg_fields:
        lines.append(f"- Impair aggregated: `{impair_agg_csv.relative_to(sweep_dir)}`")
    lines.append(f"- Summary: `{(out_dir / 'summary.json').relative_to(sweep_dir)}`")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
