#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MiB = 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _bootstrap_ci_mean(values: list[float], *, alpha: float = 0.05, n_boot: int = 5000) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (deterministic)."""
    if not values:
        raise ValueError("Cannot compute CI for empty list")
    if len(values) < 2:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(0)
    vals = np.asarray(values, dtype=float)
    n = int(vals.shape[0])
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = vals[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


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


def _ok_runs(run_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for rd in run_dirs:
        status = rd / "run_status.json"
        if not status.exists() or not _run_ok(status):
            continue
        out.append(rd)
    return out


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


def _ddf_stop_team_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    v = data.get("term_ddf_stop_team_s")
    if v is None:
        v = data.get("time_from_input_end_to_ddf_stop_team_s")
    if v is None:
        raise KeyError(f"Missing DDF stop timing in: {derived_kpis_path}")
    return float(v)


def _run_ok(run_status_path: Path) -> bool:
    data = _load_json(run_status_path)
    return bool(data.get("ok", False))


def _resource_samples(profile_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (t_rel_s, cpu_pct, rss_mib).
    """
    data = _load_json(profile_path)
    samples = data.get("samples") or []
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"resource profile has no samples: {profile_path}")
    ts = []
    cpu = []
    rss = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "ts" not in s:
            continue
        try:
            t = float(s.get("ts") or 0.0)
        except Exception:
            continue
        if t <= 0.0:
            continue
        try:
            c = float(s.get("cpu_process_pct", 0.0) or 0.0)
        except Exception:
            c = 0.0
        try:
            r = float(s.get("rss_bytes", 0.0) or 0.0) / MiB
        except Exception:
            r = 0.0
        ts.append(t)
        cpu.append(c)
        rss.append(r)
    if len(ts) < 2:
        raise ValueError(f"resource profile too short: {profile_path}")
    t0 = float(ts[0])
    t_rel = np.asarray([t - t0 for t in ts], dtype=float)
    return t_rel, np.asarray(cpu, dtype=float), np.asarray(rss, dtype=float)


def _cpu_seconds_from_profile(profile_path: Path) -> float:
    data = _load_json(profile_path)
    samples = data.get("samples") or []
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(f"resource profile has too few samples: {profile_path}")
    rows = []
    for s in samples:
        if not isinstance(s, dict) or "ts" not in s:
            continue
        try:
            rows.append((float(s.get("ts") or 0.0), float(s.get("cpu_process_pct", 0.0) or 0.0)))
        except Exception:
            continue
    rows = [(t, c) for t, c in rows if t > 0.0]
    rows.sort(key=lambda x: x[0])
    if len(rows) < 2:
        raise ValueError(f"resource profile has too few valid samples: {profile_path}")
    cpu_s = 0.0
    for (t0, c0), (t1, _c1) in zip(rows[:-1], rows[1:]):
        dt = float(t1 - t0)
        if dt <= 0.0:
            continue
        cpu_s += (float(c0) / 100.0) * dt
    return float(cpu_s)


def _collect_impair_ok_values(
    *, impair_root: Path, dataset_tag: str, backend: str, variant: str, metric: str
) -> tuple[list[float], bool]:
    """
    Returns (values_over_ok_runs, has_any_run).
    """
    run_dirs = sorted(impair_root.glob(f"{dataset_tag}__r??/{backend}/impair/{variant}"))
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs for {dataset_tag} {backend} {variant}")
    ok = []
    for rd in run_dirs:
        status = rd / "run_status.json"
        if not status.exists() or not _run_ok(status):
            continue
        derived = rd / "kpi_metrics/derived_kpis.json"
        if metric == "t_global":
            ok.append(_t_global_s(derived))
        elif metric == "iface_p95":
            ok.append(_iface_p95_s(derived))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return ok, bool(run_dirs)


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
    uniq_h = []
    uniq_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax_traf.legend(uniq_h, uniq_l, loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_r3_summary")


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
            ax_ate.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                marker="o",
                capsize=4,
                linewidth=1.6,
            )
            for xi, yi, ok_flag in zip(x, y, has_data[backend]):
                if ok_flag or not np.isfinite(float(yi)):
                    continue
                ax_ate.scatter([xi], [yi], marker="x", s=60, color="red", zorder=5)

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
            ax_tr.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                marker="o",
                capsize=4,
                linewidth=1.6,
            )
            for xi, yi, ok_flag in zip(x, y, has_data[backend]):
                if ok_flag or not np.isfinite(float(yi)):
                    continue
                ax_tr.scatter([xi], [yi], marker="x", s=60, color="red", zorder=5)

        ax_tr.set_title(f"{regime_label}: traffic vs team size")
        ax_tr.set_xlabel("Team size")
        ax_tr.set_ylabel("Traffic (MiB)")
        ax_tr.set_xticks(team_sizes, [f"r{n}" for n in team_sizes])
        ax_tr.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_scalability")


def plot_impairment_timing(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
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

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    axes = axes.flatten()

    for ax, (title, dataset_tag) in zip(axes, scenarios):
        x = np.arange(len(variants))
        c_mean: list[float | None] = []
        c_ci: list[tuple[float, float] | None] = []
        d_mean: list[float | None] = []
        d_ci: list[tuple[float, float] | None] = []
        d_fail_x: list[int] = []

        for idx, (_, var) in enumerate(variants):
            if var is None:
                c_runs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, "centralised"))
                d_runs = _ok_runs(_collect_run_dirs(baseline_root, dataset_tag, "decentralised"))
                c_vals = [_t_global_s(rd / "kpi_metrics/derived_kpis.json") for rd in c_runs]
                d_vals = [_iface_p95_s(rd / "kpi_metrics/derived_kpis.json") for rd in d_runs]
                c_mean.append(float(mean(c_vals)) if c_vals else None)
                c_ci.append(_bootstrap_ci_mean(c_vals) if c_vals else None)
                d_mean.append(float(mean(d_vals)) if d_vals else None)
                d_ci.append(_bootstrap_ci_mean(d_vals) if d_vals else None)
                continue

            c_vals, _ = _collect_impair_ok_values(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="centralised", variant=var, metric="t_global"
            )
            d_vals, any_run = _collect_impair_ok_values(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="decentralised", variant=var, metric="iface_p95"
            )
            c_mean.append(float(mean(c_vals)) if c_vals else None)
            c_ci.append(_bootstrap_ci_mean(c_vals) if c_vals else None)
            d_mean.append(float(mean(d_vals)) if d_vals else None)
            d_ci.append(_bootstrap_ci_mean(d_vals) if d_vals else None)
            if not d_vals and any_run:
                d_fail_x.append(idx)

        def _plot_series(
            means: list[float | None],
            cis: list[tuple[float, float] | None],
            *,
            label: str,
            color: str,
        ) -> None:
            xs = [i for i, v in enumerate(means) if v is not None]
            ys = [float(means[i]) for i in xs]
            ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=1.8)
            yerr_low = []
            yerr_high = []
            for i in xs:
                ci = cis[i]
                if ci is None:
                    yerr_low.append(0.0)
                    yerr_high.append(0.0)
                else:
                    yerr_low.append(float(means[i] - ci[0]))
                    yerr_high.append(float(ci[1] - means[i]))
            ax.errorbar(xs, ys, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", ecolor=color, capsize=3, linewidth=0.8)

        _plot_series(c_mean, c_ci, label="Centralised $t_{\\mathrm{global}}$", color="#4C72B0")
        _plot_series(d_mean, d_ci, label="Decentralised Iface p95", color="#55A868")

        for idx in d_fail_x:
            ax.text(idx, 0.95, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(title)
        ax.set_xticks(x, [v[0] for v in variants])
        ax.set_ylabel("Time (s)")
        ax.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "eval_impairment_timing")


def plot_resource_timeseries(*, baseline_root: Path, out_dir: Path, repeat: str = "__r01") -> None:
    """
    CPU% and RSS over time for the central server (centralised) and per-agent processes (decentralised).

    If per-agent profiles are missing, falls back to plotting the single decentralised profile and labels it as legacy.
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

        # Decentralised per-agent
        d_kpi = baseline_root / f"{dataset_tag}{repeat}" / "decentralised" / "kpi_metrics"
        agent_profiles = sorted([p for p in d_kpi.glob("resource_profile_*.json") if p.name != "resource_profile.json"])
        if agent_profiles:
            colors = plt.get_cmap("tab10").colors
            for i, p in enumerate(agent_profiles):
                rid = p.stem.replace("resource_profile_", "")
                t, cpu, rss = _resource_samples(p)
                color = colors[i % len(colors)]
                ax_cpu.plot(t, cpu, label=f"Agent {rid}", color=color, linewidth=1.2, alpha=0.9)
                ax_rss.plot(t, rss, label=f"Agent {rid}", color=color, linewidth=1.2, alpha=0.9)
        else:
            legacy = d_kpi / "resource_profile.json"
            if legacy.exists():
                t, cpu, rss = _resource_samples(legacy)
                ax_cpu.plot(t, cpu, label="Decentral (legacy profile)", color="#55A868", linewidth=1.6)
                ax_rss.plot(t, rss, label="Decentral (legacy profile)", color="#55A868", linewidth=1.6)

        ax_cpu.set_title(title)
        ax_cpu.set_ylabel("CPU (%)")
        ax_rss.set_ylabel("RSS (MiB)")
        ax_rss.set_xlabel("Time since start (s)")

    # Legend once (right subplot)
    handles, labels = ax_cpu_pr.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_cpu_wifi.get_legend_handles_labels()
    if handles:
        ax_cpu_pr.legend(handles, labels, loc="upper right", frameon=True)

    _save(fig, out_dir / "eval_resources_timeseries")


def plot_ddf_ate_vs_stop_time(*, baseline_root: Path, out_dir: Path) -> None:
    """
    Decentralised-only diagnostic: ATE vs DDF termination time.

    x-axis is T_stop_ddf(team): time from input_end(team) until all agents emit ddf_stop (team=max_r).
    y-axis is team-mean ATE across robots for that run.
    """
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

        # Plot per team size point + CI (both axes)
        for i in range(len(xs_mean)):
            color = colors[i % len(colors)]
            xm = xs_mean[i]
            ym = ys_mean[i]
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


def plot_efficiency_cpu_seconds_per_step(*, baseline_root: Path, out_dir: Path, repeat: str = "__r01") -> None:
    """
    Efficiency-normalized metric: CPU-seconds per optimisation step.

    For decentralised runs, uses summed per-agent CPU-seconds if per-agent profiles exist; otherwise skips.
    """
    regimes = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    values: dict[tuple[int, int], float] = {}
    for i, (_label, dataset_tag) in enumerate(regimes):
        for j, (_bl, backend) in enumerate(backends):
            kpi = baseline_root / f"{dataset_tag}{repeat}" / backend / "kpi_metrics"
            derived = kpi / "derived_kpis.json"
            if not derived.exists():
                continue
            opt_count = float(_load_json(derived).get("optimization_duration_s", {}).get("count") or 0.0)
            if opt_count <= 0.0:
                continue

            if backend == "centralised":
                prof = kpi / "resource_profile.json"
                if not prof.exists():
                    continue
                cpu_s = _cpu_seconds_from_profile(prof)
            else:
                agent_profiles = sorted([p for p in kpi.glob("resource_profile_*.json") if p.name != "resource_profile.json"])
                if not agent_profiles:
                    # Avoid producing misleading numbers from legacy orchestrator-only profiles.
                    continue
                cpu_s = float(sum(_cpu_seconds_from_profile(p) for p in agent_profiles))
            values[(i, j)] = float(cpu_s / opt_count)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.6))
    x = np.arange(len(regimes))
    width = 0.35
    colors = {"Centralised": "#4C72B0", "Decentralised": "#55A868"}
    missing_decentralised = True
    for j, (label, _backend) in enumerate(backends):
        ys = []
        xs = []
        for i in range(len(regimes)):
            v = values.get((i, j))
            if v is None:
                continue
            xs.append(x[i] + (j - 0.5) * width)
            ys.append(v)
            if label == "Decentralised":
                missing_decentralised = False
        if ys:
            ax.bar(xs, ys, width=width, label=label, color=colors[label], alpha=0.9)

    ax.set_xticks(x, [r[0] for r in regimes])
    ax.set_ylabel("CPU-seconds per optimisation step (s/step)")
    ax.set_title("Efficiency (r3 baseline)")
    ax.legend(loc="upper left", frameon=True)
    if missing_decentralised:
        ax.text(
            0.99,
            0.02,
            "Note: decentralised requires per-agent resource profiles",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
    _save(fig, out_dir / "eval_efficiency_cpu_s_per_step")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=repo_root / "new_eval/out/baseline_20260204_180031")
    parser.add_argument("--impair-root", type=Path, default=repo_root / "new_eval/out/impair_20260204_191738")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "new_eval/plots/main")
    args = parser.parse_args()

    baseline_root: Path = args.baseline_root
    impair_root: Path = args.impair_root
    out_dir: Path = args.out_dir

    for p in [baseline_root, impair_root]:
        if not p.exists():
            raise FileNotFoundError(p)

    _ensure_dir(out_dir)
    _setup_matplotlib()

    plot_r3_summary(baseline_root=baseline_root, out_dir=out_dir)
    plot_scalability(baseline_root=baseline_root, out_dir=out_dir)
    plot_impairment_timing(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)
    plot_resource_timeseries(baseline_root=baseline_root, out_dir=out_dir)
    plot_ddf_ate_vs_stop_time(baseline_root=baseline_root, out_dir=out_dir)
    plot_efficiency_cpu_seconds_per_step(baseline_root=baseline_root, out_dir=out_dir)

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
