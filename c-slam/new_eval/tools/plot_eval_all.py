#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


MiB = 1024 * 1024


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


def _robot_ate_rmses(estimation_metrics: dict[str, Any]) -> dict[str, float]:
    # Centralised aggregate format: {"ate": {"a": {"rmse": ...}, ...}, "rpe": ...}
    if "ate" in estimation_metrics and isinstance(estimation_metrics["ate"], dict):
        return {robot: float(vals["rmse"]) for robot, vals in estimation_metrics["ate"].items()}

    # Decentralised aggregate format: {"a": {"ate": {"rmse": ...}, ...}, ...}
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
    return float(data["time_to_global_convergence_s"])


def _iface_p95_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    ic = data.get("interface_correction_stabilised") or {}
    return float(ic["p95"])


def _opt_p95_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    od = data.get("optimization_duration_s") or {}
    return float(od["p95"])


def _cpu_mean_pct(resource_profile_path: Path) -> float:
    data = _load_json(resource_profile_path)
    summary = data.get("summary", {})
    cpu = summary.get("cpu_process_pct", {})
    return float(cpu["mean"])


def _rss_mean_mib(resource_profile_path: Path) -> float:
    data = _load_json(resource_profile_path)
    summary = data.get("summary", {})
    rss = summary.get("rss_bytes", {})
    return float(rss["mean"]) / MiB


def _run_ok(run_status_path: Path) -> bool:
    return bool(_load_json(run_status_path).get("ok", False))


def _delivery_min_from_robustness(robustness_factors: dict[str, Any]) -> float:
    topics = robustness_factors.get("stats", {}).get("topics", {})
    rates = [
        float(v.get("delivery_rate"))
        for v in topics.values()
        if isinstance(v, dict) and isinstance(v.get("delivery_rate"), (int, float))
    ]
    if not rates:
        raise ValueError("No delivery_rate entries found in robustness_factors.json")
    return float(min(rates))


def _delivery_min_mean_over_repeats(robustness_factors_list: list[dict[str, Any]]) -> float:
    topics = sorted({t for r in robustness_factors_list for t in r.get("stats", {}).get("topics", {})})
    means: dict[str, float] = {}
    for t in topics:
        vals: list[float] = []
        for r in robustness_factors_list:
            v = r.get("stats", {}).get("topics", {}).get(t, {}).get("delivery_rate")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            means[t] = float(mean(vals))
    if not means:
        raise ValueError("Could not compute delivery means across repeats")
    return float(min(means.values()))


def _collect_repeats(root: Path, glob_pattern: str) -> list[Path]:
    return sorted(root.glob(glob_pattern))


def _collect_run_dirs(root: Path, dataset_tag: str, backend: str) -> list[Path]:
    """
    Supports both layouts:
      - <root>/<dataset_tag>/<backend>/... (single run)
      - <root>/<dataset_tag>__rXX/<backend>/... (repeats)
    """
    direct = root / dataset_tag / backend
    if direct.exists():
        return [direct]
    return sorted(root.glob(f"{dataset_tag}__r??/{backend}"))


def _all_runs_ok(run_dirs: Iterable[Path]) -> bool:
    dirs = list(run_dirs)
    if not dirs:
        raise FileNotFoundError("No run dirs matched")
    statuses = []
    for rd in dirs:
        p = rd / "run_status.json"
        if not p.exists():
            statuses.append(False)
        else:
            statuses.append(_run_ok(p))
    return bool(all(statuses))


def _mean_over_ok_runs(run_dirs: Iterable[Path], metric_fn) -> float | None:
    vals: list[float] = []
    any_run = False
    for run_dir in run_dirs:
        any_run = True
        status = run_dir / "run_status.json"
        if not status.exists() or not _run_ok(status):
            continue
        vals.append(float(metric_fn(run_dir)))
    if vals:
        return float(mean(vals))
    if any_run:
        return None
    raise FileNotFoundError("No run dirs matched")


def _mean_traffic_over_ok_runs(run_dirs: Iterable[Path]) -> TrafficStats | None:
    ups: list[float] = []
    downs: list[float] = []
    any_run = False
    for run_dir in run_dirs:
        any_run = True
        if not _run_ok(run_dir / "run_status.json"):
            continue
        tr = _traffic_stats(run_dir / "kpi_metrics/derived_kpis.json")
        ups.append(tr.uplink_mib)
        downs.append(tr.downlink_mib)
    if ups:
        return TrafficStats(uplink_mib=float(mean(ups)), downlink_mib=float(mean(downs)))
    if any_run:
        return None
    raise FileNotFoundError("No run dirs matched")


def _mean_ate_over_ok_runs(run_dirs: Iterable[Path]) -> AteStats | None:
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    any_run = False
    for run_dir in run_dirs:
        any_run = True
        if not _run_ok(run_dir / "run_status.json"):
            continue
        a = _ate_stats(run_dir / "kpi_metrics/estimation_metrics.json")
        means.append(a.mean_rmse)
        mins.append(a.min_rmse)
        maxs.append(a.max_rmse)
    if means:
        return AteStats(mean_rmse=float(mean(means)), min_rmse=float(mean(mins)), max_rmse=float(mean(maxs)))
    if any_run:
        return None
    raise FileNotFoundError("No run dirs matched")


def _delivery_min_mean_over_ok_runs(run_dirs: Iterable[Path]) -> float | None:
    robs: list[dict[str, Any]] = []
    any_run = False
    for run_dir in run_dirs:
        any_run = True
        if not _run_ok(run_dir / "run_status.json"):
            continue
        robs.append(_load_json(run_dir / "kpi_metrics/robustness_factors.json"))
    if robs:
        return _delivery_min_mean_over_repeats(robs)
    if any_run:
        return None
    raise FileNotFoundError("No run dirs matched")


# A. Must-have


def plot_a1_r3_baseline_summary(*, baseline_root: Path, out_dir: Path) -> None:
    combos = [
        ("WiFi C", "wifi-r3-wifi", "centralised"),
        ("WiFi D", "wifi-r3-wifi", "decentralised"),
        ("ProRadio C", "proradio-r3-proradio", "centralised"),
        ("ProRadio D", "proradio-r3-proradio", "decentralised"),
    ]

    ate = []
    ate_lo = []
    ate_hi = []
    up = []
    down = []

    for _, dataset_tag, backend in combos:
        runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
        a = _mean_ate_over_ok_runs(runs)
        t = _mean_traffic_over_ok_runs(runs)
        if a is None or t is None:
            raise ValueError(f"No successful repeats for baseline {dataset_tag}/{backend}")
        ate.append(a.mean_rmse)
        ate_lo.append(a.mean_rmse - a.min_rmse)
        ate_hi.append(a.max_rmse - a.mean_rmse)
        up.append(t.uplink_mib)
        down.append(t.downlink_mib)

    x = np.arange(len(combos))

    fig, (ax_ate, ax_tr) = plt.subplots(1, 2, figsize=(12.2, 3.8))

    ax_ate.bar(
        x,
        ate,
        yerr=np.vstack([ate_lo, ate_hi]),
        capsize=4,
        edgecolor="black",
        linewidth=0.6,
        color="#4C72B0",
    )
    ax_ate.set_xticks(x, [c[0] for c in combos])
    ax_ate.set_ylabel("ATE RMSE (m)")
    ax_ate.set_title("r3 baseline accuracy (mean with min/max across robots)")
    ax_ate.text(
        0.02,
        0.02,
        "Error bars = [min,max] across robots",
        transform=ax_ate.transAxes,
        fontsize=9,
        color="red",
        va="bottom",
    )

    colors = {"uplink": "#4C72B0", "downlink": "#DD8452"}
    ax_tr.bar(x, up, label="uplink", color=colors["uplink"], edgecolor="black", linewidth=0.6)
    ax_tr.bar(
        x,
        down,
        bottom=up,
        label="downlink",
        color=colors["downlink"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax_tr.set_xticks(x, [c[0] for c in combos])
    ax_tr.set_ylabel("Traffic (MiB)")
    ax_tr.set_title("r3 baseline traffic (directionality)")
    ax_tr.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "A1_r3_baseline_summary")


def plot_a2_scalability_ate(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 4, 5]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    for ax, (regime_label, prefix) in zip(axes, regimes):
        x = np.array(team_sizes, dtype=float)
        for label, backend in backends:
            y = []
            y_min = []
            y_max = []
            ok = []
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                ok.append(_all_runs_ok(runs))
                a = _mean_ate_over_ok_runs(runs)
                if a is None:
                    raise ValueError(f"No successful repeats for baseline {dataset_tag}/{backend}")
                y.append(a.mean_rmse)
                y_min.append(a.min_rmse)
                y_max.append(a.max_rmse)
            y_arr = np.array(y)
            yerr = np.vstack([y_arr - np.array(y_min), np.array(y_max) - y_arr])
            ax.errorbar(x, y_arr, yerr=yerr, marker="o", capsize=4, linewidth=1.8, label=label)
            for xi, yi, ok_flag in zip(x, y_arr, ok):
                if ok_flag:
                    continue
                ax.scatter([xi], [yi], marker="x", s=70, color="red", zorder=6)

        ax.set_title(regime_label)
        ax.set_xlabel("Team size")
        ax.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax.set_ylabel("ATE RMSE (m)")
        ax.legend(loc="upper left", frameon=True)
        ax.text(
            0.02,
            0.02,
            "Error bars = [min,max] across robots\nX = unsupported/fail",
            transform=ax.transAxes,
            fontsize=9,
            color="red",
            va="bottom",
        )

    _save(fig, out_dir / "A2_scalability_ate_vs_team")


def plot_a3_scalability_traffic(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 4, 5]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    colors = {"uplink": "#4C72B0", "downlink": "#DD8452", "dec": "#55A868"}

    for ax, (regime_label, prefix) in zip(axes, regimes):
        x0 = np.arange(len(team_sizes))
        width = 0.35

        c_up, c_down, c_ok = [], [], []
        d_up, d_ok = [], []
        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            c_runs = _collect_run_dirs(baseline_root, dataset_tag, "centralised")
            d_runs = _collect_run_dirs(baseline_root, dataset_tag, "decentralised")
            c_ok.append(_all_runs_ok(c_runs))
            d_ok.append(_all_runs_ok(d_runs))
            c_tr = _mean_traffic_over_ok_runs(c_runs)
            d_tr = _mean_traffic_over_ok_runs(d_runs)
            if c_tr is None or d_tr is None:
                raise ValueError(f"No successful repeats for baseline traffic {dataset_tag}")
            c_up.append(c_tr.uplink_mib)
            c_down.append(c_tr.downlink_mib)
            d_up.append(d_tr.uplink_mib)  # downlink ~0

        c_pos = x0 - width / 2
        d_pos = x0 + width / 2
        ax.bar(
            c_pos,
            c_up,
            width=width,
            label="Centralised uplink",
            color=colors["uplink"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar(
            c_pos,
            c_down,
            width=width,
            bottom=c_up,
            label="Centralised downlink",
            color=colors["downlink"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar(
            d_pos,
            d_up,
            width=width,
            label="Decentralised uplink",
            color=colors["dec"],
            edgecolor="black",
            linewidth=0.6,
        )

        for i, ok_flag in enumerate(c_ok):
            if ok_flag:
                continue
            ax.text(c_pos[i], float(c_up[i] + c_down[i]) + 1.0, "fail", ha="center", color="red")
        for i, ok_flag in enumerate(d_ok):
            if ok_flag:
                continue
            ax.text(d_pos[i], float(d_up[i]) + 1.0, "fail", ha="center", color="red")

        ax.set_title(regime_label)
        ax.set_xlabel("Team size")
        ax.set_xticks(x0, [f"r{n}" for n in team_sizes])
        ax.set_ylabel("Traffic (MiB)")
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, loc="upper left", frameon=True)

    _save(fig, out_dir / "A3_scalability_traffic_vs_team")


def plot_a4_qos_r3_traffic_vs_profile(*, baseline_root: Path, qos_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi r3", "wifi-r3-wifi"), ("ProRadio r3", "proradio-r3-proradio")]
    profiles = [("RV-20", None), ("BT-TL-10", "best_effort_tl_d10"), ("BT-TL-50", "best_effort_tl_d50")]
    colors = {"c": "#4C72B0", "d": "#55A868", "down": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), sharey=True)

    for ax, (title, dataset_tag) in zip(axes, regimes):
        x0 = np.arange(len(profiles))
        width = 0.35

        # Traffic bars
        c_up, c_down, d_up = [], [], []
        # ATE overlay points
        c_ate, d_ate = [], []

        for _, variant in profiles:
            if variant is None:
                c_runs = _collect_run_dirs(baseline_root, dataset_tag, "centralised")
                d_runs = _collect_run_dirs(baseline_root, dataset_tag, "decentralised")
                c_tr = _mean_traffic_over_ok_runs(c_runs)
                d_tr = _mean_traffic_over_ok_runs(d_runs)
                c_a = _mean_ate_over_ok_runs(c_runs)
                d_a = _mean_ate_over_ok_runs(d_runs)
                if c_tr is None or d_tr is None or c_a is None or d_a is None:
                    raise RuntimeError(f"Unexpected missing baseline data for {dataset_tag}")
                c_up.append(c_tr.uplink_mib)
                c_down.append(c_tr.downlink_mib)
                d_up.append(d_tr.uplink_mib)
                c_ate.append(c_a.mean_rmse)
                d_ate.append(d_a.mean_rmse)
                continue

            c_runs = _collect_repeats(qos_root, f"{dataset_tag}__r??/centralised/qos/{variant}")
            d_runs = _collect_repeats(qos_root, f"{dataset_tag}__r??/decentralised/qos/{variant}")
            c_tr = _mean_traffic_over_ok_runs(c_runs)
            d_tr = _mean_traffic_over_ok_runs(d_runs)
            c_a = _mean_ate_over_ok_runs(c_runs)
            d_a = _mean_ate_over_ok_runs(d_runs)
            if c_tr is None or d_tr is None or c_a is None or d_a is None:
                raise RuntimeError(f"Unexpected missing QoS data for {dataset_tag} {variant}")
            c_up.append(c_tr.uplink_mib)
            c_down.append(c_tr.downlink_mib)
            d_up.append(d_tr.uplink_mib)
            c_ate.append(c_a.mean_rmse)
            d_ate.append(d_a.mean_rmse)

        c_pos = x0 - width / 2
        d_pos = x0 + width / 2
        ax.bar(
            c_pos,
            c_up,
            width=width,
            label="Centralised uplink",
            color=colors["c"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar(
            c_pos,
            c_down,
            width=width,
            bottom=c_up,
            label="Centralised downlink",
            color=colors["down"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar(
            d_pos,
            d_up,
            width=width,
            label="Decentralised uplink",
            color=colors["d"],
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_title(title)
        ax.set_xlabel("QoS profile")
        ax.set_xticks(x0, [p[0] for p in profiles])
        ax.set_ylabel("Traffic (MiB)")

        ax2 = ax.twinx()
        ax2.plot(c_pos, c_ate, marker="o", linestyle="none", color=colors["c"], label="Centralised ATE")
        ax2.plot(d_pos, d_ate, marker="o", linestyle="none", color=colors["d"], label="Decentralised ATE")
        ax2.set_ylabel("ATE RMSE (m)")
        ax2.grid(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    _save(fig, out_dir / "A4_qos_r3_traffic_vs_profile")


def plot_a5_impairments_timing_vs_severity(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 5]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.4), sharex=True)
    colors = {3: "#4C72B0", 5: "#55A868"}
    styles = {"centralised": "-", "decentralised": "-"}

    for col, (regime_label, prefix) in enumerate(regimes):
        # Row 0: centralised t_global
        ax_c = axes[0, col]
        # Row 1: decentralised iface p95
        ax_d = axes[1, col]

        x = np.arange(len(severities))

        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"

            c_vals: list[float | None] = []
            d_vals: list[float | None] = []
            d_fail: list[int] = []

            for idx, (_, variant) in enumerate(severities):
                if variant is None:
                    c_runs = _collect_run_dirs(baseline_root, dataset_tag, "centralised")
                    d_runs = _collect_run_dirs(baseline_root, dataset_tag, "decentralised")
                    c_mean = _mean_over_ok_runs(c_runs, lambda rd: _t_global_s(rd / "kpi_metrics/derived_kpis.json"))
                    d_mean = _mean_over_ok_runs(d_runs, lambda rd: _iface_p95_s(rd / "kpi_metrics/derived_kpis.json"))
                    if c_mean is None or d_mean is None:
                        raise ValueError(f"No successful baseline repeats for {dataset_tag}")
                    c_vals.append(c_mean)
                    d_vals.append(d_mean)
                    continue

                c_runs = _collect_repeats(impair_root, f"{dataset_tag}__r??/centralised/impair/{variant}")
                d_runs = _collect_repeats(impair_root, f"{dataset_tag}__r??/decentralised/impair/{variant}")
                c_mean = _mean_over_ok_runs(c_runs, lambda rd: _t_global_s(rd / "kpi_metrics/derived_kpis.json"))
                d_mean = _mean_over_ok_runs(d_runs, lambda rd: _iface_p95_s(rd / "kpi_metrics/derived_kpis.json"))
                c_vals.append(c_mean)
                d_vals.append(d_mean)
                if d_mean is None:
                    d_fail.append(idx)

            # Plot centralised (skip Nones)
            xs = [i for i, v in enumerate(c_vals) if v is not None]
            ys = [float(c_vals[i]) for i in xs]
            ax_c.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                color=colors[n],
                linestyle="-" if n == 3 else "--",
                label=f"r{n}",
            )

            # Plot decentralised (skip Nones)
            xs = [i for i, v in enumerate(d_vals) if v is not None]
            ys = [float(d_vals[i]) for i in xs]
            ax_d.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                color=colors[n],
                linestyle="-" if n == 3 else "--",
                label=f"r{n}",
            )
            for idx in d_fail:
                ax_d.text(idx, 0.92, "fail", transform=ax_d.get_xaxis_transform(), ha="center", color="red")

        ax_c.set_title(f"{regime_label}: centralised timing ($t_{{\\mathrm{{global}}}}$)")
        ax_c.set_ylabel("Time (s)")
        ax_c.legend(loc="upper left", frameon=True, title="Team")

        ax_d.set_title(f"{regime_label}: decentralised timing (Iface p95)")
        ax_d.set_ylabel("Time (s)")
        ax_d.set_xticks(x, [s[0] for s in severities])
        ax_d.legend(loc="upper left", frameon=True, title="Team")

    _save(fig, out_dir / "A5_impairments_timing_vs_severity")


def plot_a6_impairments_ate_vs_severity(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 5]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]
    colors = {"centralised": "#4C72B0", "decentralised": "#55A868"}
    styles = {3: "-", 5: "--"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), sharey=False)

    for ax, (regime_label, prefix) in zip(axes, regimes):
        x = np.arange(len(severities))
        for backend_label, backend in backends:
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                vals: list[float | None] = []
                fail_idx: list[int] = []
                for idx, (_, variant) in enumerate(severities):
                    if variant is None:
                        base_runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                        a = _mean_ate_over_ok_runs(base_runs)
                        if a is None:
                            raise ValueError(f"No successful baseline repeats for {dataset_tag}/{backend}")
                        vals.append(a.mean_rmse)
                        continue
                    runs = _collect_repeats(impair_root, f"{dataset_tag}__r??/{backend}/impair/{variant}")
                    a = _mean_ate_over_ok_runs(runs)
                    if a is None:
                        vals.append(None)
                        fail_idx.append(idx)
                    else:
                        vals.append(a.mean_rmse)
                xs = [i for i, v in enumerate(vals) if v is not None]
                ys = [float(vals[i]) for i in xs]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    color=colors[backend],
                    linestyle=styles[n],
                    label=f"{backend_label} r{n}",
                )
                for idx in fail_idx:
                    ax.text(idx, 0.92, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(f"{regime_label}: ATE vs impairment severity")
        ax.set_xticks(x, [s[0] for s in severities])
        ax.set_ylabel("ATE RMSE (m)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", frameon=True, ncol=2)

    _save(fig, out_dir / "A6_impairments_ate_vs_severity")


# B. Strongly recommended


def plot_b7_delivery_vs_severity(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 5]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]
    colors = {"centralised": "#4C72B0", "decentralised": "#55A868"}
    styles = {3: "-", 5: "--"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), sharey=True)

    for ax, (regime_label, prefix) in zip(axes, regimes):
        x = np.arange(len(severities))
        for backend_label, backend in backends:
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                vals: list[float | None] = []
                fail_idx: list[int] = []
                for idx, (_, variant) in enumerate(severities):
                    if variant is None:
                        base_runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                        d = _delivery_min_mean_over_ok_runs(base_runs)
                        if d is None:
                            raise ValueError(f"No successful baseline repeats for delivery {dataset_tag}/{backend}")
                        vals.append(d)
                        continue
                    runs = _collect_repeats(impair_root, f"{dataset_tag}__r??/{backend}/impair/{variant}")
                    d = _delivery_min_mean_over_ok_runs(runs)
                    if d is None:
                        vals.append(None)
                        fail_idx.append(idx)
                    else:
                        vals.append(d)
                xs = [i for i, v in enumerate(vals) if v is not None]
                ys = [float(vals[i]) for i in xs]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    color=colors[backend],
                    linestyle=styles[n],
                    label=f"{backend_label} r{n}",
                )
                for idx in fail_idx:
                    ax.text(idx, 0.15, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(f"{regime_label}: delivery vs impairment severity")
        ax.set_xticks(x, [s[0] for s in severities])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Factor delivery min")
        ax.legend(loc="lower left", frameon=True, ncol=2)

    _save(fig, out_dir / "B7_delivery_vs_severity")


def plot_b9_compute_scaling(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 4, 5]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]
    colors = {"centralised": "#4C72B0", "decentralised": "#55A868"}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), sharex=True)

    for col, (regime_label, prefix) in enumerate(regimes):
        ax_cpu = axes[0, col]
        ax_rss = axes[1, col]
        x = np.array(team_sizes, dtype=float)

        for label, backend in backends:
            cpu = []
            rss = []
            ok = []
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                base_runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                ok.append(_all_runs_ok(base_runs))
                cpu_m = _mean_over_ok_runs(base_runs, lambda rd: _cpu_mean_pct(rd / "kpi_metrics/resource_profile.json"))
                rss_m = _mean_over_ok_runs(base_runs, lambda rd: _rss_mean_mib(rd / "kpi_metrics/resource_profile.json"))
                if cpu_m is None or rss_m is None:
                    raise ValueError(f"No successful baseline repeats for compute {dataset_tag}/{backend}")
                cpu.append(cpu_m)
                rss.append(rss_m)

            ax_cpu.plot(x, cpu, marker="o", linewidth=1.8, color=colors[backend], label=label)
            ax_rss.plot(x, rss, marker="o", linewidth=1.8, color=colors[backend], label=label)
            for xi, ok_flag in zip(x, ok):
                if ok_flag:
                    continue
                ax_cpu.text(xi, 0.92, "fail", transform=ax_cpu.get_xaxis_transform(), ha="center", color="red")
                ax_rss.text(xi, 0.92, "fail", transform=ax_rss.get_xaxis_transform(), ha="center", color="red")

        ax_cpu.set_title(f"{regime_label}: CPU mean vs team size")
        ax_cpu.set_ylabel("CPU mean (%)")
        ax_cpu.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax_cpu.legend(loc="upper left", frameon=True)

        ax_rss.set_title(f"{regime_label}: RSS mean vs team size")
        ax_rss.set_ylabel("RSS mean (MiB)")
        ax_rss.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax_rss.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "B9_compute_scaling_cpu_rss")


# C. Optional


def plot_c10_opt_p95_vs_team(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 4, 5]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]
    colors = {"centralised": "#4C72B0", "decentralised": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    for ax, (regime_label, prefix) in zip(axes, regimes):
        x = np.array(team_sizes, dtype=float)
        for label, backend in backends:
            y = []
            ok = []
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                base_runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                ok.append(_all_runs_ok(base_runs))
                v = _mean_over_ok_runs(base_runs, lambda rd: _opt_p95_s(rd / "kpi_metrics/derived_kpis.json"))
                if v is None:
                    raise ValueError(f"No successful baseline repeats for opt p95 {dataset_tag}/{backend}")
                y.append(v)
            ax.plot(x, y, marker="o", linewidth=1.8, color=colors[backend], label=label)
            for xi, ok_flag in zip(x, ok):
                if ok_flag:
                    continue
                ax.scatter([xi], [y[team_sizes.index(int(xi))]], marker="x", s=70, color="red", zorder=6)

        ax.set_title(regime_label)
        ax.set_xlabel("Team size")
        ax.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax.set_ylabel("Optimisation duration p95 (s)")
        ax.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "C10_opt_p95_vs_team")


def plot_c11_scatter_traffic_vs_ate(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 5]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]
    backends = [("Centralised", "centralised"), ("Decentralised", "decentralised")]
    colors = {"centralised": "#4C72B0", "decentralised": "#55A868"}
    markers = {"wifi": "o", "proradio": "^"}

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.2))

    for regime_label, prefix in regimes:
        for n in team_sizes:
            dataset_tag = f"{prefix}-r{n}-{prefix}"
            for _, variant in severities:
                for _, backend in backends:
                    if variant is None:
                        base_runs = _collect_run_dirs(baseline_root, dataset_tag, backend)
                        tr_stats = _mean_traffic_over_ok_runs(base_runs)
                        a_stats = _mean_ate_over_ok_runs(base_runs)
                        if tr_stats is None or a_stats is None:
                            continue
                        tr = tr_stats.total_mib
                        a = a_stats.mean_rmse
                        ax.scatter(
                            tr,
                            a,
                            marker=markers[prefix],
                            s=70 if n == 3 else 110,
                            color=colors[backend],
                            edgecolor="black",
                            linewidth=0.6,
                            alpha=0.9,
                        )
                        continue

                    runs = _collect_repeats(impair_root, f"{dataset_tag}__r??/{backend}/impair/{variant}")
                    tr_stats = _mean_traffic_over_ok_runs(runs)
                    a_stats = _mean_ate_over_ok_runs(runs)
                    if tr_stats is None or a_stats is None:
                        continue
                    ax.scatter(
                        tr_stats.total_mib,
                        a_stats.mean_rmse,
                        marker=markers[prefix],
                        s=70 if n == 3 else 110,
                        color=colors[backend],
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.9,
                    )

    ax.set_xlabel("Traffic (MiB)")
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("Traffic vs ATE (points = scenarios; marker=regime; size=team)")

    # Legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="WiFi", markerfacecolor="gray", markeredgecolor="black"),
        Line2D([0], [0], marker="^", color="w", label="ProRadio", markerfacecolor="gray", markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Centralised", markerfacecolor=colors["centralised"], markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Decentralised", markerfacecolor=colors["decentralised"], markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="r3", markerfacecolor="gray", markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="r5", markerfacecolor="gray", markeredgecolor="black", markersize=10),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True)

    _save(fig, out_dir / "C11_scatter_traffic_vs_ate")


def plot_c12_failure_matrix(*, baseline_root: Path, qos_root: Path, impair_root: Path, out_dir: Path) -> None:
    # Encode: -1 missing, 0 fail, 1 ok
    cmap = ListedColormap(["#D55E00", "#999999", "#009E73"])  # fail, missing, ok
    # Values will be remapped to indices: fail=0, missing=1, ok=2
    def _val_to_idx(v: int) -> int:
        return {0: 0, -1: 1, 1: 2}[v]

    backends = [("C", "centralised"), ("D", "decentralised")]
    regimes = [("WiFi", "wifi"), ("ProRadio", "proradio")]
    team_sizes = [3, 4, 5]
    qos_variants = [("BT-TL-10", "best_effort_tl_d10"), ("BT-TL-50", "best_effort_tl_d50")]
    impair_variants = [
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]

    def status_baseline(prefix: str, n: int, backend: str) -> int:
        tag = f"{prefix}-r{n}-{prefix}"
        runs = _collect_run_dirs(baseline_root, tag, backend)
        if not runs:
            return -1
        try:
            return 1 if _all_runs_ok(runs) else 0
        except FileNotFoundError:
            return -1

    def status_all_repeats(root: Path, pattern: str) -> int:
        runs = _collect_repeats(root, pattern)
        if not runs:
            return -1
        statuses = []
        for rd in runs:
            p = rd / "run_status.json"
            if not p.exists():
                statuses.append(False)
            else:
                statuses.append(_run_ok(p))
        return 1 if all(statuses) else 0

    # Build matrices
    baseline_rows = []
    baseline_vals = []
    for regime_label, prefix in regimes:
        for n in team_sizes:
            baseline_rows.append(f"{regime_label} r{n} baseline")
            baseline_vals.append([status_baseline(prefix, n, b) for _, b in backends])

    qos_rows = []
    qos_vals = []
    for regime_label, prefix in regimes:
        for n in team_sizes:
            tag = f"{prefix}-r{n}-{prefix}"
            for qos_label, variant in qos_variants:
                qos_rows.append(f"{regime_label} r{n} qos {qos_label}")
                qos_vals.append(
                    [
                        status_all_repeats(qos_root, f"{tag}__r??/{backend}/qos/{variant}")
                        for _, backend in backends
                    ]
                )

    impair_rows = []
    impair_vals = []
    for regime_label, prefix in regimes:
        for n in team_sizes:
            tag = f"{prefix}-r{n}-{prefix}"
            for imp_label, variant in impair_variants:
                impair_rows.append(f"{regime_label} r{n} impair {imp_label}")
                impair_vals.append(
                    [
                        status_all_repeats(impair_root, f"{tag}__r??/{backend}/impair/{variant}")
                        for _, backend in backends
                    ]
                )

    def _plot_block(ax, title: str, rows: list[str], vals: list[list[int]]) -> None:
        arr = np.array([[ _val_to_idx(v) for v in row ] for row in vals], dtype=int)
        ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(rows)), rows)
        ax.set_xticks(np.arange(len(backends)), [b[0] for b in backends])
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=10)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = {0: "fail", 1: "", 2: "ok"}[arr[i, j]]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 18.0))
    _plot_block(axes[0], "Baseline support matrix", baseline_rows, baseline_vals)
    _plot_block(axes[1], "QoS support matrix (all repeats must succeed)", qos_rows, qos_vals)
    _plot_block(axes[2], "Impairment support matrix (all repeats must succeed)", impair_rows, impair_vals)

    # Add legend
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#009E73", edgecolor="black", label="ok"),
            Patch(facecolor="#D55E00", edgecolor="black", label="fail"),
            Patch(facecolor="#999999", edgecolor="black", label="missing"),
        ],
        loc="upper center",
        ncol=3,
        frameon=True,
    )

    _save(fig, out_dir / "C12_failure_support_matrix")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=repo_root / "new_eval/out/baseline_20260204_180031")
    parser.add_argument("--qos-root", type=Path, default=repo_root / "new_eval/out/qos_20260205_081932")
    parser.add_argument("--impair-root", type=Path, default=repo_root / "new_eval/out/impair_20260204_191738")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "new_eval/plots/all")
    args = parser.parse_args()

    baseline_root: Path = args.baseline_root
    qos_root: Path = args.qos_root
    impair_root: Path = args.impair_root
    out_dir: Path = args.out_dir

    for p in [baseline_root, qos_root, impair_root]:
        if not p.exists():
            raise FileNotFoundError(p)

    _ensure_dir(out_dir)
    _setup_matplotlib()

    plot_a1_r3_baseline_summary(baseline_root=baseline_root, out_dir=out_dir)
    plot_a2_scalability_ate(baseline_root=baseline_root, out_dir=out_dir)
    plot_a3_scalability_traffic(baseline_root=baseline_root, out_dir=out_dir)
    plot_a4_qos_r3_traffic_vs_profile(baseline_root=baseline_root, qos_root=qos_root, out_dir=out_dir)
    plot_a5_impairments_timing_vs_severity(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)
    plot_a6_impairments_ate_vs_severity(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)

    plot_b7_delivery_vs_severity(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)
    # B8 (T_conv(team)) not generated here because it is not exported by the KPI JSONs in new_eval/out.
    plot_b9_compute_scaling(baseline_root=baseline_root, out_dir=out_dir)

    plot_c10_opt_p95_vs_team(baseline_root=baseline_root, out_dir=out_dir)
    plot_c11_scatter_traffic_vs_ate(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)
    plot_c12_failure_matrix(baseline_root=baseline_root, qos_root=qos_root, impair_root=impair_root, out_dir=out_dir)

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
