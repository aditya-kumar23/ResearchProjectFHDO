#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

MiB = 1024 * 1024


# -----------------------------
# I/O helpers
# -----------------------------

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# KPI parsing
# -----------------------------


def _robot_ate_rmses(estimation_metrics: dict[str, Any]) -> dict[str, float]:
    """Return per-robot ATE RMSE values from either centralised or decentralised formats."""
    # Centralised aggregate format
    if "ate" in estimation_metrics and isinstance(estimation_metrics["ate"], dict):
        return {robot: float(vals["rmse"]) for robot, vals in estimation_metrics["ate"].items()}

    # Decentralised aggregate format
    out: dict[str, float] = {}
    for robot, vals in estimation_metrics.items():
        if not isinstance(vals, dict):
            continue
        ate = vals.get("ate")
        if not isinstance(ate, dict):
            continue
        if "rmse" in ate:
            out[str(robot)] = float(ate["rmse"])
    if not out:
        raise ValueError("Unrecognized estimation_metrics.json format")
    return out


@dataclass(frozen=True)
class AteStats:
    mean_rmse: float
    min_rmse: float
    max_rmse: float


@dataclass(frozen=True)
class TrafficStats:
    uplink_mib: float
    downlink_mib: float

    @property
    def total_mib(self) -> float:
        return self.uplink_mib + self.downlink_mib


def _ate_stats(estimation_metrics_path: Path) -> AteStats:
    data = _load_json(estimation_metrics_path)
    rmses = _robot_ate_rmses(data)
    vals = list(rmses.values())
    return AteStats(mean_rmse=float(mean(vals)), min_rmse=float(min(vals)), max_rmse=float(max(vals)))


def _traffic_stats(derived_kpis_path: Path) -> TrafficStats:
    data = _load_json(derived_kpis_path)
    uplink = float(data.get("communication_uplink_bytes", 0.0)) / MiB
    downlink = float(data.get("communication_downlink_bytes", 0.0)) / MiB
    return TrafficStats(uplink_mib=uplink, downlink_mib=downlink)


def _t_global_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    # Centralised dissemination tail proxy
    return float(data["time_to_global_convergence_s"])


def _iface_p95_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    ic = data.get("interface_correction_stabilised") or {}
    return float(ic["p95"])


def _team_conv_s(derived_kpis_path: Path) -> float:
    """Shared stabilisation proxy used in evaluation chapter (team statistic)."""
    data = _load_json(derived_kpis_path)
    # In the pipeline this is already aligned to input_end.
    # Prefer the explicit time-from-input-end field when present.
    v = data.get("time_from_input_end_to_team_convergence_s")
    if isinstance(v, (int, float)):
        return float(v)
    return float(data["stable_team_convergence_s"])


def _ddf_stop_team_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    # DDF stop time post input_end (team statistic)
    v = data.get("time_from_input_end_to_ddf_stop_team_s")
    if isinstance(v, (int, float)):
        return float(v)
    return float(data["term_ddf_stop_team_s"])


def _run_ok(run_status_path: Path) -> bool:
    return bool(_load_json(run_status_path).get("ok", False))


# -----------------------------
# Run discovery
# -----------------------------


def _collect_run_dirs(baseline_root: Path, dataset_tag: str, backend: str) -> list[Path]:
    """Collect baseline/scalability repeat folders.

    Pattern: <baseline_root>/<dataset_tag>__r??/<backend>/
    """
    return sorted(baseline_root.glob(f"{dataset_tag}__r??/{backend}"))


def _collect_qos_run_dirs(qos_root: Path, dataset_tag: str, backend: str, variant: str) -> list[Path]:
    """Collect QoS repeat folders.

    Pattern: <qos_root>/<dataset_tag>__r??/<backend>/qos/<variant>/
    """
    return sorted(qos_root.glob(f"{dataset_tag}__r??/{backend}/qos/{variant}"))


def _collect_impair_run_dirs(impair_root: Path, dataset_tag: str, backend: str, variant: str) -> list[Path]:
    """Collect impairment repeat folders.

    Pattern: <impair_root>/<dataset_tag>__r??/<backend>/impair/<variant>/
    """
    return sorted(impair_root.glob(f"{dataset_tag}__r??/{backend}/impair/{variant}"))


def _ok_runs(run_dirs: list[Path]) -> list[Path]:
    ok: list[Path] = []
    for rd in run_dirs:
        status = rd / "run_status.json"
        if status.exists() and _run_ok(status):
            ok.append(rd)
    return ok


def _repeat_id_from_run_dir(run_dir: Path) -> str:
    """Extract __rXX token from a run path."""
    # e.g., .../wifi-r3-wifi__r03/centralised/impair/bwcap_...
    for p in run_dir.parents:
        name = p.name
        if "__r" in name:
            return name.split("__r", 1)[1]
    return ""


def _dataset_regime(dataset_tag: str) -> str:
    if dataset_tag.startswith("wifi"):
        return "WiFi"
    if dataset_tag.startswith("proradio"):
        return "ProRadio"
    return dataset_tag.split("-", 1)[0]


def _dataset_team_size(dataset_tag: str) -> int:
    # expects ...-rN-...
    parts = dataset_tag.split("-r", 1)
    if len(parts) < 2:
        return -1
    rest = parts[1]
    n = ""
    for ch in rest:
        if ch.isdigit():
            n += ch
        else:
            break
    try:
        return int(n)
    except Exception:
        return -1


# -----------------------------
# Statistics
# -----------------------------


def _bootstrap_ci_mean(values: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 1) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(0, len(values))] for _ in range(len(values))]
        samples.append(float(mean(draw)))
    samples.sort()
    lo = samples[int((alpha / 2.0) * len(samples))]
    hi = samples[int((1.0 - alpha / 2.0) * len(samples)) - 1]
    return (float(lo), float(hi))


def _paired_ratios(
    *,
    base_dirs: list[Path],
    var_dirs: list[Path],
    metric_fn,
) -> tuple[list[float], bool]:
    """Compute per-repeat ratios var/base for matching repeat IDs.

    Returns:
      ratios: list of ratios for repeat IDs where both base and var are ok.
      any_run: True if any var_dirs exist (even if none ok).
    """
    any_run = bool(var_dirs)
    base_ok = { _repeat_id_from_run_dir(d): d for d in _ok_runs(base_dirs) }
    ratios: list[float] = []
    for d in _ok_runs(var_dirs):
        rid = _repeat_id_from_run_dir(d)
        b = base_ok.get(rid)
        if not b:
            continue
        b_kpi = b / "kpi_metrics/derived_kpis.json"
        v_kpi = d / "kpi_metrics/derived_kpis.json"
        try:
            denom = float(metric_fn(b_kpi))
            num = float(metric_fn(v_kpi))
            if denom > 0:
                ratios.append(num / denom)
        except Exception:
            continue
    return ratios, any_run


# -----------------------------
# Plots
# -----------------------------


def plot_r3_summary(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi", "wifi-r3-wifi"),
        ("ProRadio", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    ate_mean = np.zeros((len(regimes), len(backends)))
    ate_ci_lo = np.zeros_like(ate_mean)
    ate_ci_hi = np.zeros_like(ate_mean)
    tr_up_mean = np.zeros_like(ate_mean)
    tr_dn_mean = np.zeros_like(ate_mean)
    tr_total_ci_lo = np.zeros_like(ate_mean)
    tr_total_ci_hi = np.zeros_like(ate_mean)
    ate_points: dict[tuple[int, int], list[float]] = {}
    tr_points: dict[tuple[int, int], list[float]] = {}

    for i, (_, dataset_tag) in enumerate(regimes):
        for j, (_, backend) in enumerate(backends):
            runs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, backend))
            if not runs:
                continue
            a_vals: list[float] = []
            up_vals: list[float] = []
            dn_vals: list[float] = []
            tot_vals: list[float] = []
            for rd in runs:
                a = _ate_stats(rd / "kpi_metrics/estimation_metrics.json")
                tr = _traffic_stats(rd / "kpi_metrics/derived_kpis.json")
                a_vals.append(float(a.mean_rmse))
                up_vals.append(float(tr.uplink_mib))
                dn_vals.append(float(tr.downlink_mib))
                tot_vals.append(float(tr.total_mib))

            ate_points[(i, j)] = a_vals
            tr_points[(i, j)] = tot_vals

            ate_mean[i, j] = float(mean(a_vals))
            lo, hi = _bootstrap_ci_mean(a_vals)
            ate_ci_lo[i, j] = lo
            ate_ci_hi[i, j] = hi
            tr_up_mean[i, j] = float(mean(up_vals))
            tr_dn_mean[i, j] = float(mean(dn_vals))
            lo, hi = _bootstrap_ci_mean(tot_vals)
            tr_total_ci_lo[i, j] = lo
            tr_total_ci_hi[i, j] = hi

    fig, (ax_ate, ax_traf) = plt.subplots(1, 2, figsize=(9.8, 3.6))

    # ATE
    x = np.arange(len(regimes))
    width = 0.35
    for j, (label, _) in enumerate(backends):
        pos = x + (j - 0.5) * width
        yerr = np.vstack([ate_mean[:, j] - ate_ci_lo[:, j], ate_ci_hi[:, j] - ate_mean[:, j]])
        ax_ate.bar(
            pos,
            ate_mean[:, j],
            width=width,
            label=label,
            yerr=yerr,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
        )
        for i in range(len(regimes)):
            pts = ate_points.get((i, j)) or []
            if not pts:
                continue
            jitter = np.linspace(-0.08, 0.08, num=len(pts))
            ax_ate.scatter(np.full(len(pts), pos[i]) + jitter, pts, s=18, color="black", alpha=0.55, zorder=3)
    ax_ate.set_xticks(x, [r[0] for r in regimes])
    ax_ate.set_ylabel("ATE RMSE (m)")
    ax_ate.set_title("Baseline r3 accuracy (mean ± 95% CI across repeats)")
    ax_ate.legend(loc="upper left", frameon=True)

    # Traffic
    colors = {"uplink": "#4C72B0", "downlink": "#DD8452"}
    for j, (label, _) in enumerate(backends):
        pos = x + (j - 0.5) * width
        ax_traf.bar(
            pos,
            tr_up_mean[:, j],
            width=width,
            label=f"{label} uplink",
            color=colors["uplink"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax_traf.bar(
            pos,
            tr_dn_mean[:, j],
            width=width,
            bottom=tr_up_mean[:, j],
            label=f"{label} downlink",
            color=colors["downlink"],
            edgecolor="black",
            linewidth=0.6,
        )
        total_mean = tr_up_mean[:, j] + tr_dn_mean[:, j]
        yerr = np.vstack([total_mean - tr_total_ci_lo[:, j], tr_total_ci_hi[:, j] - total_mean])
        ax_traf.errorbar(pos, total_mean, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=3, zorder=3)
        for i in range(len(regimes)):
            pts = tr_points.get((i, j)) or []
            if not pts:
                continue
            jitter = np.linspace(-0.08, 0.08, num=len(pts))
            ax_traf.scatter(np.full(len(pts), pos[i]) + jitter, pts, s=18, color="black", alpha=0.40, zorder=4)
    ax_traf.set_xticks(x, [r[0] for r in regimes])
    ax_traf.set_ylabel("Traffic (MiB)")
    ax_traf.set_title("Baseline r3 traffic (mean ± 95% CI across repeats)")

    # De-duplicate legend entries by label
    handles, labels = ax_traf.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax_traf.legend(uniq_h, uniq_l, loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_r3_summary")


def plot_pareto_ate_vs_traffic(*, baseline_root: Path, out_dir: Path) -> None:
    """Pareto view: ATE vs traffic across baseline repeats (r3--r5)."""
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]
    backends = [
        ("Centralised", "centralised", "o"),
        ("Decentralised", "decentralised", "^"),
    ]

    points = []  # (traffic, ate, regime, team, backend_label)
    for regime_label, prefix in regimes:
        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            for backend_label, backend, marker in backends:
                runs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, backend))
                for rd in runs:
                    ate = _ate_stats(rd / "kpi_metrics/estimation_metrics.json").mean_rmse
                    traf = _traffic_stats(rd / "kpi_metrics/derived_kpis.json").total_mib
                    points.append((traf, ate, regime_label, n, backend_label, marker))

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.3))

    # Styling (kept simple and consistent across papers)
    regime_color = {"WiFi": "#4C72B0", "ProRadio": "#55A868"}
    size_map = {3: 55, 4: 75, 5: 95}

    for traf, ate, regime_label, n, backend_label, marker in points:
        ax.scatter(
            traf,
            ate,
            s=size_map.get(int(n), 60),
            marker=marker,
            c=regime_color.get(regime_label, "#333333"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
        )

    ax.set_xlabel("Total traffic (MiB)")
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("Baseline Pareto: accuracy vs communication (r3–r5, all repeats)")

    # Build a 2-part legend (backend markers, regime colors)
    from matplotlib.lines import Line2D

    legend_backend = [
        Line2D([0], [0], marker="o", color="w", label="Centralised", markerfacecolor="#999999", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Decentralised", markerfacecolor="#999999", markeredgecolor="black", markersize=8),
    ]
    legend_regime = [
        Line2D([0], [0], marker="s", color="w", label="WiFi", markerfacecolor=regime_color["WiFi"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="ProRadio", markerfacecolor=regime_color["ProRadio"], markeredgecolor="black", markersize=8),
    ]
    leg1 = ax.legend(handles=legend_backend, loc="upper right", frameon=True, title="Backend")
    ax.add_artist(leg1)
    ax.legend(handles=legend_regime, loc="upper center", frameon=True, title="Regime")

    _save(fig, out_dir / "eval_pareto_ate_vs_traffic")


def plot_scalability(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))

    for row, (regime_label, prefix) in enumerate(regimes):
        ate_means: dict[str, list[float]] = {b: [] for _, b in backends}
        ate_ci: dict[str, list[tuple[float, float]]] = {b: [] for _, b in backends}
        tr_means: dict[str, list[float]] = {b: [] for _, b in backends}
        tr_ci: dict[str, list[tuple[float, float]]] = {b: [] for _, b in backends}
        has_data: dict[str, list[bool]] = {b: [] for _, b in backends}

        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            for _, backend in backends:
                runs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, backend))
                if not runs:
                    ate_means[backend].append(float("nan"))
                    ate_ci[backend].append((float("nan"), float("nan")))
                    tr_means[backend].append(float("nan"))
                    tr_ci[backend].append((float("nan"), float("nan")))
                    has_data[backend].append(False)
                    continue
                a_vals = [float(_ate_stats(rd / "kpi_metrics/estimation_metrics.json").mean_rmse) for rd in runs]
                t_vals = [float(_traffic_stats(rd / "kpi_metrics/derived_kpis.json").total_mib) for rd in runs]
                ate_means[backend].append(float(mean(a_vals)))
                ate_ci[backend].append(_bootstrap_ci_mean(a_vals))
                tr_means[backend].append(float(mean(t_vals)))
                tr_ci[backend].append(_bootstrap_ci_mean(t_vals))
                has_data[backend].append(True)

        # ATE subplot
        ax_ate = axes[row, 0]
        x = np.array(team_sizes, dtype=float)
        for label, backend in backends:
            y = np.array(ate_means[backend], dtype=float)
            lo = np.array([c[0] for c in ate_ci[backend]], dtype=float)
            hi = np.array([c[1] for c in ate_ci[backend]], dtype=float)
            yerr = np.vstack([y - lo, hi - y])
            ax_ate.errorbar(x, y, yerr=yerr, label=label, marker="o", capsize=4, linewidth=1.6)

        ax_ate.set_title(f"{regime_label}: ATE vs team size")
        ax_ate.set_xlabel("Team size")
        ax_ate.set_ylabel("ATE RMSE (m)")
        ax_ate.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax_ate.legend(loc="upper left", frameon=True)

        # Traffic subplot
        ax_tr = axes[row, 1]
        for label, backend in backends:
            y = np.array(tr_means[backend], dtype=float)
            lo = np.array([c[0] for c in tr_ci[backend]], dtype=float)
            hi = np.array([c[1] for c in tr_ci[backend]], dtype=float)
            yerr = np.vstack([y - lo, hi - y])
            ax_tr.errorbar(x, y, yerr=yerr, label=label, marker="o", capsize=4, linewidth=1.6)

        ax_tr.set_title(f"{regime_label}: traffic vs team size")
        ax_tr.set_xlabel("Team size")
        ax_tr.set_ylabel("Traffic (MiB)")
        ax_tr.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax_tr.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_scalability")


def plot_qos_sensitivity_slopes(*, baseline_root: Path, qos_root: Path, out_dir: Path) -> None:
    """Slope-chart style summary for QoS (r3), for ATE and traffic."""
    regimes = [
        ("WiFi", "wifi-r3-wifi"),
        ("ProRadio", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]
    variants = [
        ("RV-20", None),
        ("BT-TL-10", "best_effort_tl_d10"),
        ("BT-TL-50", "best_effort_tl_d50"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), sharex="col")

    for col, (regime_label, dataset_tag) in enumerate(regimes):
        ax_ate = axes[0, col]
        ax_tr = axes[1, col]
        x = np.arange(len(variants))

        for backend_label, backend in backends:
            ate_means, ate_ci = [], []
            tr_means, tr_ci = [], []
            fail_x = []

            for idx, (_lbl, var) in enumerate(variants):
                if var is None:
                    runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                    ok = _ok_runs(runs)
                else:
                    runs = _collect_qos_run_dirs(qos_root, dataset_tag, backend, var)
                    ok = _ok_runs(runs)

                if not ok:
                    ate_means.append(None)
                    ate_ci.append(None)
                    tr_means.append(None)
                    tr_ci.append(None)
                    if runs:
                        fail_x.append(idx)
                    continue

                a_vals = [float(_ate_stats(rd / "kpi_metrics/estimation_metrics.json").mean_rmse) for rd in ok]
                t_vals = [float(_traffic_stats(rd / "kpi_metrics/derived_kpis.json").total_mib) for rd in ok]
                ate_means.append(float(mean(a_vals)))
                ate_ci.append(_bootstrap_ci_mean(a_vals))
                tr_means.append(float(mean(t_vals)))
                tr_ci.append(_bootstrap_ci_mean(t_vals))

            def _plot(ax, means, cis, label, marker):
                xs = [i for i, v in enumerate(means) if v is not None]
                ys = [float(means[i]) for i in xs]
                ax.plot(xs, ys, marker=marker, linewidth=1.8, label=label)
                yerr_low, yerr_high = [], []
                for i in xs:
                    lo, hi = cis[i]
                    yerr_low.append(float(means[i] - lo))
                    yerr_high.append(float(hi - means[i]))
                ax.errorbar(xs, ys, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", capsize=3, linewidth=0.8)

            _plot(ax_ate, ate_means, ate_ci, backend_label, marker="o" if backend == "centralised" else "^")
            _plot(ax_tr, tr_means, tr_ci, backend_label, marker="o" if backend == "centralised" else "^")

            for idx in fail_x:
                ax_ate.text(idx, 0.95, "fail", transform=ax_ate.get_xaxis_transform(), ha="center", color="red")
                ax_tr.text(idx, 0.95, "fail", transform=ax_tr.get_xaxis_transform(), ha="center", color="red")

        ax_ate.set_title(f"{regime_label} r3")
        ax_ate.set_ylabel("ATE RMSE (m)")
        ax_tr.set_ylabel("Traffic (MiB)")
        ax_tr.set_xlabel("QoS profile")
        ax_tr.set_xticks(x, [v[0] for v in variants])

    axes[0, 0].legend(loc="upper left", frameon=True)
    _save(fig, out_dir / "eval_qos_sensitivity_slopes")


def plot_impairment_degradation(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    """Normalised impairment sensitivity (slowdown vs baseline) for timing proxies."""
    scenarios = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("WiFi r5", "wifi-r5-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
        ("ProRadio r5", "proradio-r5-proradio"),
    ]
    variants = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    axes = axes.flatten()

    for ax, (title, dataset_tag) in zip(axes, scenarios):
        x = np.arange(len(variants))

        # Baseline dirs for matching repeats
        c_base = _collect_run_dirs(baseline_root, dataset_tag, "centralised")
        d_base = _collect_run_dirs(baseline_root, dataset_tag, "decentralised")

        c_means, c_ci, c_fail_x = [], [], []
        d_means, d_ci, d_fail_x = [], [], []

        for idx, (_lbl, var) in enumerate(variants):
            if var is None:
                # defined as 1.0 when baseline exists
                c_ok = _ok_runs(c_base)
                d_ok = _ok_runs(d_base)
                c_means.append(1.0 if c_ok else None)
                c_ci.append((1.0, 1.0) if c_ok else None)
                d_means.append(1.0 if d_ok else None)
                d_ci.append((1.0, 1.0) if d_ok else None)
                continue

            c_var = _collect_impair_run_dirs(impair_root, dataset_tag, "centralised", var)
            d_var = _collect_impair_run_dirs(impair_root, dataset_tag, "decentralised", var)

            c_ratios, c_any = _paired_ratios(base_dirs=c_base, var_dirs=c_var, metric_fn=_t_global_s)
            d_ratios, d_any = _paired_ratios(base_dirs=d_base, var_dirs=d_var, metric_fn=_iface_p95_s)

            if c_ratios:
                c_means.append(float(mean(c_ratios)))
                c_ci.append(_bootstrap_ci_mean(c_ratios))
            else:
                c_means.append(None)
                c_ci.append(None)
                if c_any:
                    c_fail_x.append(idx)

            if d_ratios:
                d_means.append(float(mean(d_ratios)))
                d_ci.append(_bootstrap_ci_mean(d_ratios))
            else:
                d_means.append(None)
                d_ci.append(None)
                if d_any:
                    d_fail_x.append(idx)

        def _plot_series(means, cis, label, marker):
            xs = [i for i, v in enumerate(means) if v is not None]
            ys = [float(means[i]) for i in xs]
            ax.plot(xs, ys, marker=marker, linewidth=1.8, label=label)
            yerr_low, yerr_high = [], []
            for i in xs:
                lo, hi = cis[i]
                yerr_low.append(float(means[i] - lo))
                yerr_high.append(float(hi - means[i]))
            ax.errorbar(xs, ys, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", capsize=3, linewidth=0.8)

        _plot_series(c_means, c_ci, "Centralised: $t_{global}/t_{base}$", marker="o")
        _plot_series(d_means, d_ci, "Decentralised: Iface p95 / base", marker="^")

        for idx in c_fail_x:
            ax.text(idx, 0.95, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")
        for idx in d_fail_x:
            ax.text(idx, 0.88, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(title)
        ax.set_xticks(x, [v[0] for v in variants])
        ax.set_ylabel("Slowdown vs baseline (×)")
        ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_impairment_degradation")


def _resource_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(path)
    samples = data.get("samples", []) or []

    # Time: support both legacy `t_rel_s` and current `ts` formats.
    if samples and "t_rel_s" in samples[0]:
        t = np.array([float(s["t_rel_s"]) for s in samples], dtype=float)
    elif samples and "ts" in samples[0]:
        ts = np.array([float(s["ts"]) for s in samples], dtype=float)
        t0 = float(ts.min()) if len(ts) else 0.0
        t = ts - t0
    else:
        t = np.arange(len(samples), dtype=float)

    # CPU%: prefer `cpu_process_pct` (per-process) but keep backwards compatibility.
    cpu = np.array(
        [float(s.get("cpu_process_pct", s.get("cpu_pct", 0.0))) for s in samples],
        dtype=float,
    )
    rss = np.array([float(s.get("rss_bytes", 0.0)) / MiB for s in data.get("samples", [])], dtype=float)
    return t, cpu, rss


def plot_resource_timeseries(*, baseline_root: Path, out_dir: Path, repeat: str = "__r01") -> None:
    """CPU% and RSS over time: server vs team-aggregate (sum over agents).

    This plot is used for the chapter narrative; it prefers a compact view (one line per architecture).
    """
    regimes = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.4), sharex="col")
    (ax_cpu_wifi, ax_cpu_pr), (ax_rss_wifi, ax_rss_pr) = axes
    ax_by_regime = {
        "wifi-r3-wifi": (ax_cpu_wifi, ax_rss_wifi),
        "proradio-r3-proradio": (ax_cpu_pr, ax_rss_pr),
    }

    for title, dataset_tag in regimes:
        ax_cpu, ax_rss = ax_by_regime[dataset_tag]

        # Centralised server
        c_prof = baseline_root / f"{dataset_tag}{repeat}" / "centralised" / "kpi_metrics" / "resource_profile.json"
        if c_prof.exists():
            t, cpu, rss = _resource_samples(c_prof)
            ax_cpu.plot(t, cpu, label="Central server", color="black", linewidth=1.8)
            ax_rss.plot(t, rss, label="Central server", color="black", linewidth=1.8)

        # Decentralised per-agent -> aggregate
        d_kpi = baseline_root / f"{dataset_tag}{repeat}" / "decentralised" / "kpi_metrics"
        agent_profiles = sorted([p for p in d_kpi.glob("resource_profile_*.json") if p.name != "resource_profile.json"])

        if agent_profiles:
            # Interpolate onto a common grid and sum.
            series = []
            t_ends = []
            for p in agent_profiles:
                t, cpu, rss = _resource_samples(p)
                if len(t) < 2:
                    continue
                series.append((t, cpu, rss))
                t_ends.append(float(t.max()))
            if series:
                t_end = float(min(t_ends))
                grid = np.linspace(0.0, t_end, 350)
                cpu_sum = np.zeros_like(grid)
                rss_sum = np.zeros_like(grid)
                for t, cpu, rss in series:
                    cpu_sum += np.interp(grid, t, cpu)
                    rss_sum += np.interp(grid, t, rss)
                ax_cpu.plot(grid, cpu_sum, label="Decentral team (sum)", linewidth=1.6)
                ax_rss.plot(grid, rss_sum, label="Decentral team (sum)", linewidth=1.6)
        else:
            # Legacy single profile if present (less informative)
            legacy = d_kpi / "resource_profile.json"
            if legacy.exists():
                t, cpu, rss = _resource_samples(legacy)
                ax_cpu.plot(t, cpu, label="Decentral (legacy)", linewidth=1.6)
                ax_rss.plot(t, rss, label="Decentral (legacy)", linewidth=1.6)

        ax_cpu.set_title(title)
        ax_cpu.set_ylabel("CPU (%)")
        ax_rss.set_ylabel("RSS (MiB)")
        ax_rss.set_xlabel("Time since start (s)")

    # Legend once (right subplot)
    handles, labels = ax_cpu_pr.get_legend_handles_labels()
    if handles:
        ax_cpu_pr.legend(handles, labels, loc="upper right", frameon=True)

    _save(fig, out_dir / "eval_resources_timeseries")


def _cpu_seconds_from_profile(profile_path: Path) -> float:
    """Integrate CPU% over time to obtain CPU-seconds."""
    t, cpu, _rss = _resource_samples(profile_path)
    if len(t) < 2:
        return 0.0
    # CPU% is percent of one core; convert to CPU-seconds by integrating (cpu/100) dt.
    dt = np.diff(t)
    cpu_mid = (cpu[:-1] + cpu[1:]) * 0.5
    return float(np.sum((cpu_mid / 100.0) * dt))


def plot_efficiency_cpu_seconds_per_step(*, baseline_root: Path, out_dir: Path) -> None:
    """Efficiency-normalised metric: CPU-seconds per optimisation step (mean ± CI across repeats)."""
    regimes = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    values: dict[tuple[int, int], list[float]] = {}

    for i, (_label, dataset_tag) in enumerate(regimes):
        for j, (_bl, backend) in enumerate(backends):
            run_dirs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, backend))
            vals: list[float] = []
            for rd in run_dirs:
                kpi = rd / "kpi_metrics"
                derived = kpi / "derived_kpis.json"
                if not derived.exists():
                    continue
                d = _load_json(derived)
                opt_count = float((d.get("optimization_duration_s") or {}).get("count") or 0.0)
                if opt_count <= 0.0:
                    continue

                if backend == "centralised":
                    prof = kpi / "resource_profile.json"
                    if not prof.exists():
                        continue
                    cpu_s = _cpu_seconds_from_profile(prof)
                    vals.append(float(cpu_s / opt_count))
                else:
                    agent_profiles = sorted([p for p in kpi.glob("resource_profile_*.json") if p.name != "resource_profile.json"])
                    if not agent_profiles:
                        # Avoid misleading numbers from legacy orchestrator-only profiles.
                        continue
                    cpu_s = float(sum(_cpu_seconds_from_profile(p) for p in agent_profiles))
                    vals.append(float(cpu_s / opt_count))

            if vals:
                values[(i, j)] = vals

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.8))
    x = np.arange(len(regimes))
    width = 0.36

    for j, (label, _backend) in enumerate(backends):
        xs = []
        ys = []
        yerr = []
        for i in range(len(regimes)):
            vals = values.get((i, j))
            if not vals:
                continue
            m = float(mean(vals))
            lo, hi = _bootstrap_ci_mean(vals)
            xs.append(x[i] + (j - 0.5) * width)
            ys.append(m)
            yerr.append((m - lo, hi - m))

            # show per-run points
            jitter = np.linspace(-0.06, 0.06, num=len(vals))
            ax.scatter(np.full(len(vals), xs[-1]) + jitter, vals, s=18, color="black", alpha=0.45, zorder=3)

        if xs:
            yerr_arr = np.array(yerr).T
            ax.bar(xs, ys, width=width, label=label, edgecolor="black", linewidth=0.6, alpha=0.9)
            ax.errorbar(xs, ys, yerr=yerr_arr, fmt="none", ecolor="black", capsize=3, linewidth=0.8, zorder=4)

    ax.set_xticks(x, [r[0] for r in regimes])
    ax.set_ylabel("CPU-seconds per optimisation step (s/step)")
    ax.set_title("Efficiency (r3 baseline, mean ± 95% CI)")
    ax.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_efficiency_cpu_s_per_step")


def plot_ddf_ate_vs_stop_time(*, baseline_root: Path, out_dir: Path) -> None:
    """Decentralised diagnostic: ATE vs DDF termination time."""
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), sharey=True)
    colors = list(plt.get_cmap("tab10").colors)

    for ax, (title, prefix) in zip(axes, regimes):
        xs_mean: list[float] = []
        ys_mean: list[float] = []
        xs_ci: list[tuple[float, float]] = []
        ys_ci: list[tuple[float, float]] = []
        labels: list[str] = []

        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            run_dirs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, "decentralised"))
            if not run_dirs:
                continue

            xs: list[float] = []
            ys: list[float] = []
            for rd in run_dirs:
                derived = rd / "kpi_metrics/derived_kpis.json"
                est = rd / "kpi_metrics/estimation_metrics.json"
                xs.append(_ddf_stop_team_s(derived))
                ys.append(_ate_stats(est).mean_rmse)

            x_m = float(mean(xs))
            y_m = float(mean(ys))
            x_lo, x_hi = _bootstrap_ci_mean(xs)
            y_lo, y_hi = _bootstrap_ci_mean(ys)

            xs_mean.append(x_m)
            ys_mean.append(y_m)
            xs_ci.append((x_lo, x_hi))
            ys_ci.append((y_lo, y_hi))
            labels.append(f"r{n}")

        for i in range(len(xs_mean)):
            color = colors[i % len(colors)]
            xm, ym = xs_mean[i], ys_mean[i]
            xlo, xhi = xs_ci[i]
            ylo, yhi = ys_ci[i]
            ax.errorbar(
                [xm],
                [ym],
                xerr=[[xm - xlo], [xhi - xm]],
                yerr=[[ym - ylo], [yhi - ym]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                label=labels[i],
            )
            ax.text(xm, ym, f" {labels[i]}", va="center", fontsize=9)

        ax.set_title(f"{title} (decentralised)")
        ax.set_xlabel(r"$T_{\mathrm{stop}}^{(\mathrm{team})}$ (s)  [post-input]")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("ATE mean (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="upper left", frameon=True, title="Team size")

    _save(fig, out_dir / "eval_ddf_ate_vs_stop_time")


def _status_from_dirs(run_dirs: list[Path]) -> str:
    if not run_dirs:
        return "n/a"
    ok = _ok_runs(run_dirs)
    if len(ok) == len(run_dirs):
        return "ok"
    if ok:
        return "fail"  # at least one repeat failed
    return "fail"


def plot_robustness_matrix(*, baseline_root: Path, qos_root: Path | None, impair_root: Path | None, out_dir: Path) -> None:
    """Matrix view of which configurations are supported (all repeats ok) vs unsupported."""
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]
    backends = [
        ("C", "centralised"),
        ("D", "decentralised"),
    ]

    conditions: list[tuple[str, str, str | None]] = [("Baseline", "baseline", None)]
    if qos_root is not None and qos_root.exists():
        conditions += [
            ("QoS BT-TL-10", "qos", "best_effort_tl_d10"),
            ("QoS BT-TL-50", "qos", "best_effort_tl_d50"),
        ]
    if impair_root is not None and impair_root.exists():
        conditions += [
            ("Impair 0.25", "impair", "bwcap_0p25mbps"),
            ("Impair blackout", "impair", "blackout_2x"),
        ]

    rows = []
    status = []

    for regime_label, prefix in regimes:
        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            for be_short, backend in backends:
                rows.append(f"{regime_label} r{n} {be_short}")
                row_vals = []
                for _cond_label, kind, variant in conditions:
                    if kind == "baseline":
                        dirs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                    elif kind == "qos":
                        # QoS is only defined for r3 in the chapter narrative
                        if n != 3 or qos_root is None:
                            dirs = []
                        else:
                            dirs = _collect_qos_run_dirs(qos_root, dataset_tag, backend, variant or "")
                    elif kind == "impair":
                        # Impairment study in the chapter focuses on r3 and r5
                        if n not in (3, 5) or impair_root is None:
                            dirs = []
                        else:
                            dirs = _collect_impair_run_dirs(impair_root, dataset_tag, backend, variant or "")
                    else:
                        dirs = []

                    s = _status_from_dirs(dirs)
                    row_vals.append(s)
                status.append(row_vals)

    # Map to numeric values for heatmap
    mapping = {"ok": 1.0, "fail": 0.0, "n/a": -1.0}
    Z = np.array([[mapping.get(v, -1.0) for v in row] for row in status], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.0))

    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#BBBBBB", "#C44E52", "#55A868"])  # n/a, fail, ok
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    im = ax.imshow(Z, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xticks(np.arange(len(conditions)), [c[0] for c in conditions], rotation=25, ha="right")
    ax.set_title("Robustness matrix: supported vs unsupported configurations")

    # Annotate cells
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, status[i][j], ha="center", va="center", fontsize=8)

    _save(fig, out_dir / "eval_robustness_matrix")


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    def _find_repo_root(start: Path) -> Path:
        for p in (start, *start.parents):
            if (p / "new_eval").exists() and (p / "Thesis_src").exists():
                return p
        return start

    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=repo_root / "new_eval/out/baseline_20260204_180031")
    parser.add_argument("--qos-root", type=Path, default=repo_root / "new_eval/out/qos_20260205_081932")
    parser.add_argument("--impair-root", type=Path, default=repo_root / "new_eval/out/impair_20260204_191738")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resource-repeat", type=str, default="__r01")
    args = parser.parse_args()

    baseline_root: Path = args.baseline_root
    qos_root: Path = args.qos_root
    impair_root: Path = args.impair_root
    out_dir: Path | None = args.out_dir
    resource_repeat: str = args.resource_repeat

    # Prefer writing directly into the thesis image folder if it exists.
    if out_dir is None:
        candidate_dirs = [
            repo_root / "img/main",
            repo_root / "thesis/img/main",
            repo_root / "new_eval/plots/main",
        ]
        for c in candidate_dirs:
            if c.exists():
                out_dir = c
                break
        if out_dir is None:
            out_dir = repo_root / "new_eval/plots/main"

    if not baseline_root.exists():
        raise FileNotFoundError(baseline_root)

    _ensure_dir(out_dir)
    _setup_matplotlib()

    plot_r3_summary(baseline_root=baseline_root, out_dir=out_dir)
    plot_pareto_ate_vs_traffic(baseline_root=baseline_root, out_dir=out_dir)
    plot_scalability(baseline_root=baseline_root, out_dir=out_dir)

    if qos_root.exists():
        plot_qos_sensitivity_slopes(baseline_root=baseline_root, qos_root=qos_root, out_dir=out_dir)

    if impair_root.exists():
        plot_impairment_degradation(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)

    plot_resource_timeseries(baseline_root=baseline_root, out_dir=out_dir, repeat=resource_repeat)
    plot_ddf_ate_vs_stop_time(baseline_root=baseline_root, out_dir=out_dir)
    plot_efficiency_cpu_seconds_per_step(baseline_root=baseline_root, out_dir=out_dir)

    plot_robustness_matrix(
        baseline_root=baseline_root,
        qos_root=qos_root if qos_root.exists() else None,
        impair_root=impair_root if impair_root.exists() else None,
        out_dir=out_dir,
    )

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
