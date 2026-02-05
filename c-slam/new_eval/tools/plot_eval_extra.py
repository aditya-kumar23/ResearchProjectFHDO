#!/usr/bin/env python3

from __future__ import annotations

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
    return float(data["time_to_global_convergence_s"])


def _iface_p95_s(derived_kpis_path: Path) -> float:
    data = _load_json(derived_kpis_path)
    ic = data.get("interface_correction_stabilised") or {}
    return float(ic["p95"])


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


def _collect_qos_repeats(
    *,
    qos_root: Path,
    dataset_tag: str,
    backend: str,
    variant: str,
) -> list[Path]:
    return sorted(qos_root.glob(f"{dataset_tag}__r??/{backend}/qos/{variant}"))


def _collect_impair_repeats(
    *,
    impair_root: Path,
    dataset_tag: str,
    backend: str,
    variant: str,
) -> list[Path]:
    return sorted(impair_root.glob(f"{dataset_tag}__r??/{backend}/impair/{variant}"))


def plot_r3_baseline_ate_traffic_directionality(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi", "wifi-r3-wifi"),
        ("ProRadio", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    ate_means = np.zeros((len(regimes), len(backends)))
    ate_err_low = np.zeros_like(ate_means)
    ate_err_high = np.zeros_like(ate_means)
    up = np.zeros_like(ate_means)
    down = np.zeros_like(ate_means)

    for i, (_, dataset_tag) in enumerate(regimes):
        for j, (_, backend) in enumerate(backends):
            run_dir = baseline_root / dataset_tag / backend
            ate = _ate_stats(run_dir / "kpi_metrics/estimation_metrics.json")
            tr = _traffic_stats(run_dir / "kpi_metrics/derived_kpis.json")
            ate_means[i, j] = ate.mean_rmse
            ate_err_low[i, j] = ate.mean_rmse - ate.min_rmse
            ate_err_high[i, j] = ate.max_rmse - ate.mean_rmse
            up[i, j] = tr.uplink_mib
            down[i, j] = tr.downlink_mib

    fig, (ax_ate, ax_traf) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # ATE
    x = np.arange(len(regimes))
    width = 0.35
    for j, (label, _) in enumerate(backends):
        pos = x + (j - 0.5) * width
        ax_ate.bar(
            pos,
            ate_means[:, j],
            width=width,
            label=label,
            yerr=np.vstack([ate_err_low[:, j], ate_err_high[:, j]]),
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
        )
    ax_ate.set_xticks(x, [r[0] for r in regimes])
    ax_ate.set_ylabel("ATE RMSE (m)")
    ax_ate.set_title("r3 baseline ATE (mean with min/max across robots)")
    ax_ate.legend(loc="upper left", frameon=True)

    # Traffic directionality
    colors = {"uplink": "#4C72B0", "downlink": "#DD8452"}
    for j, (label, _) in enumerate(backends):
        pos = x + (j - 0.5) * width
        ax_traf.bar(
            pos,
            up[:, j],
            width=width,
            label=f"{label} uplink",
            color=colors["uplink"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax_traf.bar(
            pos,
            down[:, j],
            width=width,
            bottom=up[:, j],
            label=f"{label} downlink",
            color=colors["downlink"],
            edgecolor="black",
            linewidth=0.6,
        )
    ax_traf.set_xticks(x, [r[0] for r in regimes])
    ax_traf.set_ylabel("Traffic (MiB)")
    ax_traf.set_title("r3 baseline traffic (uplink/downlink)")
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

    _save(fig, out_dir / "r3_baseline_ate_traffic_directionality")


def plot_scalability_ate_vs_team(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)

    for ax, (regime_label, prefix) in zip(axes, regimes):
        x = np.array(team_sizes, dtype=float)
        for label, backend in backends:
            y: list[float] = []
            y_min: list[float] = []
            y_max: list[float] = []
            ok: list[bool] = []
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                run_dir = baseline_root / dataset_tag / backend
                ok.append(_run_ok(run_dir / "run_status.json"))
                ate = _ate_stats(run_dir / "kpi_metrics/estimation_metrics.json")
                y.append(ate.mean_rmse)
                y_min.append(ate.min_rmse)
                y_max.append(ate.max_rmse)
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

    _save(fig, out_dir / "scalability_ate_vs_team")


def plot_scalability_traffic_vs_team(*, baseline_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi", "wifi"),
        ("ProRadio", "proradio"),
    ]
    team_sizes = [3, 4, 5]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)

    for ax, (regime_label, prefix) in zip(axes, regimes):
        x0 = np.arange(len(team_sizes))
        width = 0.35
        colors = {"uplink": "#4C72B0", "downlink": "#DD8452"}
        for j, (label, backend) in enumerate(backends):
            upl: list[float] = []
            dnl: list[float] = []
            ok: list[bool] = []
            for n in team_sizes:
                dataset_tag = f"{prefix}-r{n}-{prefix}"
                run_dir = baseline_root / dataset_tag / backend
                ok.append(_run_ok(run_dir / "run_status.json"))
                tr = _traffic_stats(run_dir / "kpi_metrics/derived_kpis.json")
                upl.append(tr.uplink_mib)
                dnl.append(tr.downlink_mib)
            pos = x0 + (j - 0.5) * width
            upl_arr = np.array(upl)
            dnl_arr = np.array(dnl)
            ax.bar(
                pos,
                upl_arr,
                width=width,
                label=f"{label} uplink",
                color=colors["uplink"],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.bar(
                pos,
                dnl_arr,
                width=width,
                bottom=upl_arr,
                label=f"{label} downlink",
                color=colors["downlink"],
                edgecolor="black",
                linewidth=0.6,
            )
            for k, ok_flag in enumerate(ok):
                if ok_flag:
                    continue
                ax.text(pos[k], float(upl_arr[k] + dnl_arr[k]) + 1.0, "fail", ha="center", color="red")

        ax.set_title(regime_label)
        ax.set_xlabel("Team size")
        ax.set_ylabel("Traffic (MiB)")
        ax.set_xticks(x0, [f"r{n}" for n in team_sizes])
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

    _save(fig, out_dir / "scalability_traffic_vs_team")


def plot_qos_r3_traffic_vs_profile(*, baseline_root: Path, qos_root: Path, out_dir: Path) -> None:
    regimes = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
    ]
    backends = [
        ("Centralised", "centralised"),
        ("Decentralised", "decentralised"),
    ]
    profiles = [
        ("RV-20", None),
        ("BT-TL-10", "best_effort_tl_d10"),
        ("BT-TL-50", "best_effort_tl_d50"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), sharey=True)

    for ax, (title, dataset_tag) in zip(axes, regimes):
        x0 = np.arange(len(profiles))
        width = 0.35
        colors = {"uplink": "#4C72B0", "downlink": "#DD8452"}

        for j, (backend_label, backend) in enumerate(backends):
            uplink_vals: list[float] = []
            downlink_vals: list[float] = []
            for _, variant in profiles:
                if variant is None:
                    run_dir = baseline_root / dataset_tag / backend
                    tr = _traffic_stats(run_dir / "kpi_metrics/derived_kpis.json")
                    uplink_vals.append(tr.uplink_mib)
                    downlink_vals.append(tr.downlink_mib)
                    continue

                run_dirs = _collect_qos_repeats(
                    qos_root=qos_root, dataset_tag=dataset_tag, backend=backend, variant=variant
                )
                if not run_dirs:
                    raise FileNotFoundError(f"No QoS runs for {dataset_tag} {backend} {variant}")
                upl = []
                dnl = []
                for run_dir in run_dirs:
                    tr = _traffic_stats(run_dir / "kpi_metrics/derived_kpis.json")
                    upl.append(tr.uplink_mib)
                    dnl.append(tr.downlink_mib)
                uplink_vals.append(float(mean(upl)))
                downlink_vals.append(float(mean(dnl)))

            pos = x0 + (j - 0.5) * width
            upl_arr = np.array(uplink_vals)
            dnl_arr = np.array(downlink_vals)
            ax.bar(
                pos,
                upl_arr,
                width=width,
                label=f"{backend_label} uplink",
                color=colors["uplink"],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.bar(
                pos,
                dnl_arr,
                width=width,
                bottom=upl_arr,
                label=f"{backend_label} downlink",
                color=colors["downlink"],
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_title(title)
        ax.set_xlabel("QoS profile")
        ax.set_xticks(x0, [p[0] for p in profiles])
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

    _save(fig, out_dir / "qos_r3_traffic_vs_profile")


def plot_impairments_timing_vs_severity(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    scenarios = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("WiFi r5", "wifi-r5-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
        ("ProRadio r5", "proradio-r5-proradio"),
    ]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
    axes = axes.flatten()

    for ax, (title, dataset_tag) in zip(axes, scenarios):
        x = np.arange(len(severities))
        c: list[float | None] = []
        d: list[float | None] = []
        d_fail: list[int] = []

        for idx, (_, variant) in enumerate(severities):
            if variant is None:
                c_dir = baseline_root / dataset_tag / "centralised"
                d_dir = baseline_root / dataset_tag / "decentralised"
                c.append(_t_global_s(c_dir / "kpi_metrics/derived_kpis.json"))
                d.append(_iface_p95_s(d_dir / "kpi_metrics/derived_kpis.json"))
                continue

            # Centralised timing: mean over OK repeats
            c_runs = _collect_impair_repeats(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="centralised", variant=variant
            )
            if not c_runs:
                raise FileNotFoundError(f"No impair runs for {dataset_tag} centralised {variant}")
            c_vals = []
            for rd in c_runs:
                if not _run_ok(rd / "run_status.json"):
                    continue
                c_vals.append(_t_global_s(rd / "kpi_metrics/derived_kpis.json"))
            c.append(float(mean(c_vals)) if c_vals else None)

            # Decentralised timing: Iface p95 mean over OK repeats, else mark failure
            d_runs = _collect_impair_repeats(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="decentralised", variant=variant
            )
            if not d_runs:
                raise FileNotFoundError(f"No impair runs for {dataset_tag} decentralised {variant}")
            d_vals = []
            for rd in d_runs:
                if not _run_ok(rd / "run_status.json"):
                    continue
                d_vals.append(_iface_p95_s(rd / "kpi_metrics/derived_kpis.json"))
            if d_vals:
                d.append(float(mean(d_vals)))
            else:
                d.append(None)
                d_fail.append(idx)

        def _plot(values: list[float | None], label: str, color: str) -> None:
            xs = [i for i, v in enumerate(values) if v is not None]
            ys = [float(values[i]) for i in xs]
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=label)

        _plot(c, "Centralised $t_{\\mathrm{global}}$", "#4C72B0")
        _plot(d, "Decentralised Iface p95", "#55A868")

        for idx in d_fail:
            ax.text(idx, 0.95, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(title)
        ax.set_xticks(x, [s[0] for s in severities])
        ax.set_ylabel("Time (s)")
        ax.legend(loc="upper left", frameon=True)

    _save(fig, out_dir / "impairments_timing_vs_severity")


def plot_impairments_delivery_vs_severity(*, baseline_root: Path, impair_root: Path, out_dir: Path) -> None:
    scenarios = [
        ("WiFi r3", "wifi-r3-wifi"),
        ("WiFi r5", "wifi-r5-wifi"),
        ("ProRadio r3", "proradio-r3-proradio"),
        ("ProRadio r5", "proradio-r5-proradio"),
    ]
    severities = [
        ("none", None),
        ("3 Mbps", "bwcap_3p0mbps"),
        ("2 Mbps", "bwcap_2p0mbps"),
        ("1 Mbps", "bwcap_1p0mbps"),
        ("0.25 Mbps", "bwcap_0p25mbps"),
        ("blackout", "blackout_2x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
    axes = axes.flatten()

    for ax, (title, dataset_tag) in zip(axes, scenarios):
        x = np.arange(len(severities))
        c: list[float | None] = []
        d: list[float | None] = []
        d_fail: list[int] = []

        for idx, (_, variant) in enumerate(severities):
            if variant is None:
                c_dir = baseline_root / dataset_tag / "centralised"
                d_dir = baseline_root / dataset_tag / "decentralised"
                c_rob = _load_json(c_dir / "kpi_metrics/robustness_factors.json")
                d_rob = _load_json(d_dir / "kpi_metrics/robustness_factors.json")
                c.append(_delivery_min_from_robustness(c_rob))
                d.append(_delivery_min_from_robustness(d_rob))
                continue

            # Centralised: mean delivery over OK repeats (delivery min computed from per-topic mean)
            c_runs = _collect_impair_repeats(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="centralised", variant=variant
            )
            c_robs = []
            for rd in c_runs:
                if not _run_ok(rd / "run_status.json"):
                    continue
                c_robs.append(_load_json(rd / "kpi_metrics/robustness_factors.json"))
            c.append(_delivery_min_mean_over_repeats(c_robs) if c_robs else None)

            # Decentralised: same; if none OK then mark fail
            d_runs = _collect_impair_repeats(
                impair_root=impair_root, dataset_tag=dataset_tag, backend="decentralised", variant=variant
            )
            d_robs = []
            for rd in d_runs:
                if not _run_ok(rd / "run_status.json"):
                    continue
                d_robs.append(_load_json(rd / "kpi_metrics/robustness_factors.json"))
            if d_robs:
                d.append(_delivery_min_mean_over_repeats(d_robs))
            else:
                d.append(None)
                d_fail.append(idx)

        def _plot(values: list[float | None], label: str, color: str) -> None:
            xs = [i for i, v in enumerate(values) if v is not None]
            ys = [float(values[i]) for i in xs]
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=label)

        _plot(c, "Centralised delivery min", "#4C72B0")
        _plot(d, "Decentralised delivery min", "#55A868")

        for idx in d_fail:
            ax.text(idx, 0.92, "fail", transform=ax.get_xaxis_transform(), ha="center", color="red")

        ax.set_title(title)
        ax.set_xticks(x, [s[0] for s in severities])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Factor delivery (min)")
        ax.legend(loc="lower left", frameon=True)

    _save(fig, out_dir / "impairments_delivery_vs_severity")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    baseline_root = repo_root / "new_eval/out/baseline_20260203_002316"
    qos_root = repo_root / "new_eval/out/qos_20260201_112816"
    impair_root = repo_root / "new_eval/out/impair_20260203_012459"
    out_dir = repo_root / "new_eval/plots/extra"

    for p in [baseline_root, qos_root, impair_root]:
        if not p.exists():
            raise FileNotFoundError(p)

    _ensure_dir(out_dir)
    _setup_matplotlib()

    plot_r3_baseline_ate_traffic_directionality(baseline_root=baseline_root, out_dir=out_dir)
    plot_scalability_ate_vs_team(baseline_root=baseline_root, out_dir=out_dir)
    plot_scalability_traffic_vs_team(baseline_root=baseline_root, out_dir=out_dir)
    plot_qos_r3_traffic_vs_profile(baseline_root=baseline_root, qos_root=qos_root, out_dir=out_dir)
    plot_impairments_timing_vs_severity(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)
    plot_impairments_delivery_vs_severity(baseline_root=baseline_root, impair_root=impair_root, out_dir=out_dir)

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

