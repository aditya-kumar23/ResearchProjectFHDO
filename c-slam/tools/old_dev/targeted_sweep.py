#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    name: str
    modality: str
    stem: str
    tag: str
    team_size: int
    robot_ids: Tuple[str, ...]
    stamp_span: Optional[float]


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


def _slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def _hash_dict(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _discover_datasets(dataset_root: Path) -> List[DatasetInfo]:
    out: List[DatasetInfo] = []
    for path in sorted(dataset_root.rglob("*.jrl")):
        doc = _read_json(path)
        if not doc:
            continue
        robots = doc.get("robots")
        team_size = len(robots) if isinstance(robots, list) else 0
        measurements = doc.get("measurements", {})
        robot_ids = sorted(str(key) for key in measurements.keys()) if isinstance(measurements, dict) else []
        if team_size <= 0:
            team_size = len(robot_ids)
        name = str(doc.get("name") or path.stem)
        modality = path.parent.name
        stem = path.stem
        tag = f"{modality}_{stem}"
        stamp_span = _estimate_stamp_span(measurements)
        out.append(
            DatasetInfo(
                path=path,
                name=name,
                modality=modality,
                stem=stem,
                tag=tag,
                team_size=team_size,
                robot_ids=tuple(robot_ids),
                stamp_span=stamp_span,
            )
        )
    return out


def _resolve_dataset_paths(paths: Optional[Sequence[str]], root: Path) -> Optional[set[str]]:
    if not paths:
        return None
    resolved: set[str] = set()
    for raw in paths:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = root / p
        resolved.add(str(p.resolve()))
    return resolved


def _estimate_stamp_span(measurements: Any) -> Optional[float]:
    if not isinstance(measurements, dict):
        return None
    min_s = None
    max_s = None
    for entries in measurements.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            stamp = entry.get("stamp")
            if not isinstance(stamp, (int, float)):
                continue
            stamp_val = float(stamp)
            min_s = stamp_val if min_s is None else min(min_s, stamp_val)
            max_s = stamp_val if max_s is None else max(max_s, stamp_val)
    if min_s is None or max_s is None:
        return None
    return max(0.0, max_s - min_s)


def _load_template(path: Path) -> Dict[str, Any]:
    cfg = _read_json(path)
    if not cfg:
        raise SystemExit(f"Baseline template missing or invalid: {path}")
    return cfg


def _deep_get(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _deep_set(obj: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur = obj
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _apply_overrides(out[key], value)
        else:
            out[key] = value
    return out


def _require_locked_paths(template: Dict[str, Any], candidate: Dict[str, Any], locked: Sequence[Sequence[str]]) -> None:
    for path in locked:
        t_val = _deep_get(template, path)
        c_val = _deep_get(candidate, path)
        if t_val != c_val:
            raise SystemExit(f"Locked field changed: {'.'.join(path)} (template={t_val}, candidate={c_val})")


def _qos_grid() -> List[Dict[str, Any]]:
    # Small grid that exposes reliability vs queue depth effects without exploding the run count.
    reliabilities = ["reliable", "best_effort"]
    depths = [1, 10, 50]
    out = []
    for rel in reliabilities:
        for depth in depths:
            out.append({"reliability": rel, "durability": "volatile", "depth": depth})
    return out


def _impairment_scenarios(robot_ids: Sequence[str], bw_caps_mbps: Sequence[float]) -> List[Dict[str, Any]]:
    rid = str(robot_ids[1] if len(robot_ids) > 1 else (robot_ids[0] if robot_ids else "a"))
    scenarios: List[Dict[str, Any]] = []
    for loss_p in (0.02, 0.05, 0.10):
        scenarios.append(
            {
                "name": f"random_loss_p{str(loss_p).replace('.', '')}",
                "stochastic": True,
                "spec": {"random_loss_p": loss_p, "random_warmup_messages": 50},
            }
        )
    scenarios.extend(
        [
            {
                "name": "burst_1p0s_0p2s",
                "stochastic": False,
                "spec": {"burst_period_s": 1.0, "burst_duration_s": 0.2},
            },
            {
                "name": "burst_2p0s_0p5s",
                "stochastic": False,
                "spec": {"burst_period_s": 2.0, "burst_duration_s": 0.5},
            },
            {
                "name": "blackout_sender",
                "stochastic": False,
                "spec": {"blackouts": [{"rid": rid, "start_s": 5.0, "end_s": 12.0, "mode": "sender"}]},
            },
            {
                "name": "blackout_receiver",
                "stochastic": False,
                "spec": {"blackouts": [{"rid": rid, "start_s": 5.0, "end_s": 12.0, "mode": "receiver"}]},
            },
        ]
    )
    for cap in bw_caps_mbps:
        scenarios.append(
            {
                "name": f"bwcap_{str(cap).replace('.', 'p')}mbps",
                "stochastic": False,
                "spec": {"bw_caps_mbps": {"default": float(cap)}},
            }
        )
    return scenarios


def _select_datasets(
    datasets: Sequence[DatasetInfo],
    *,
    sweep: str,
    modalities: Sequence[str],
    include_day_night: bool,
) -> List[DatasetInfo]:
    allowed = {modality_name for modality_name in modalities}
    out: List[DatasetInfo] = []
    for dataset in datasets:
        if dataset.modality not in allowed:
            continue
        is_day_night = dataset.stem.startswith("day_") or dataset.stem.startswith("night_")
        if sweep in {"qos", "impair", "baseline"}:
            if dataset.team_size != 3:
                continue
            if not include_day_night and is_day_night:
                continue
        if sweep == "scale":
            if not (
                dataset.stem.startswith("r3_")
                or dataset.stem.startswith("r4_")
                or dataset.stem.startswith("r5_")
            ):
                continue
        out.append(dataset)
    return out


def _estimate_duration_s(dataset: DatasetInfo, template: Dict[str, Any]) -> Optional[float]:
    span = dataset.stamp_span
    if span is None:
        return None
    time_scale = _deep_get(template, ["publisher", "time_scale"])
    if not isinstance(time_scale, (int, float)) or time_scale <= 0:
        return None
    return float(span) / float(time_scale)


def _with_dynamic_blackouts(spec: Dict[str, Any], duration_s: Optional[float]) -> Dict[str, Any]:
    if duration_s is None:
        return spec
    blackouts = spec.get("blackouts")
    if not isinstance(blackouts, list) or not blackouts:
        return spec
    # Keep blackouts early and short so they land inside most runs, even when playback is accelerated.
    start = min(max(5.0, 0.1 * duration_s), 15.0)
    end = start + min(7.0, max(2.0, 0.05 * duration_s))
    out = json.loads(json.dumps(spec))
    out["blackouts"] = [
        {**blackout, "start_s": float(start), "end_s": float(end)} if isinstance(blackout, dict) else blackout
        for blackout in blackouts
    ]
    return out


def _build_run_config(
    *,
    template: Dict[str, Any],
    dataset: DatasetInfo,
    backend: str,
    export_path: Path,
    qos: Optional[Dict[str, Any]] = None,
    impair: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _apply_overrides(
        template,
        {
            "dataset": str(dataset.path),
            "export_path": str(export_path),
            "backend": backend,
        },
    )
    if qos is not None:
        cfg["qos"] = dict(qos)
    if impair is not None:
        cfg["impair"] = impair
    return cfg


def _run_orchestrator(config_path: Path, *, timeout_s: Optional[float], env: Optional[Dict[str, str]] = None) -> int:
    cmd = [sys.executable, str(ROOT / "tools" / "orchestrate.py"), "--config", str(config_path)]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout_s, env=env)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        return 124


def _read_status(run_dir: Path) -> Tuple[bool, Optional[str]]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return False, "missing_run_status"
    data = _read_json(status_path)
    ok = bool(data.get("ok", False))
    err = data.get("error")
    return ok, err if err else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        cast = float(value)
        if cast != cast:
            return None
        return cast
    except Exception:
        return None


def _compute_ros_domain_id(config_hash: str, base: int, span: int) -> int:
    # Isolate each run in its own ROS_DOMAIN_ID to avoid QoS clashes with lingering nodes.
    try:
        h = int(config_hash[:8], 16)
    except Exception:
        h = abs(hash(config_hash))
    return int(base) + (h % int(span))


def _restart_ros_daemon() -> None:
    # Best-effort reset of ROS 2 discovery cache between runs.
    try:
        subprocess.run(["ros2", "daemon", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass


def _summary_from_derived(derived: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["time_to_global_convergence_s"] = derived.get("time_to_global_convergence_s")
    out["time_to_global_convergence_method"] = derived.get("time_to_global_convergence_method")

    opt = derived.get("optimization_duration_s", {})
    if isinstance(opt, dict):
        out["optimization_duration_mean_s"] = opt.get("mean")
        out["optimization_duration_p95_s"] = opt.get("p95")

    loop = derived.get("loop_closure_correction_stabilised", {})
    if isinstance(loop, dict):
        out["loop_closure_correction_stabilised_median_s"] = loop.get("median")
        out["loop_closure_correction_stabilised_p90_s"] = loop.get("p90")

    out["communication_uplink_bytes"] = derived.get("communication_uplink_bytes")
    out["communication_downlink_bytes"] = derived.get("communication_downlink_bytes")
    out["communication_uplink_bytes_per_s"] = derived.get("communication_uplink_bytes_per_s")
    out["communication_downlink_bytes_per_s"] = derived.get("communication_downlink_bytes_per_s")

    delivery = derived.get("delivery_rate", {})
    if isinstance(delivery, dict):
        out["delivery_rate_mean"] = delivery.get("mean")

    est = derived.get("estimation_metrics", {})
    out.update(_estimate_ate_summary(est))
    return out


def _estimate_ate_summary(est: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ate: Any = None
    if isinstance(est, dict) and isinstance(est.get("ate"), dict):
        # Centralised shape: {"ate": {"a": {...rmse...}, ...}, "rpe": {...}}
        ate = est.get("ate")
    elif isinstance(est, dict):
        # Decentralised shape: {"a": {"ate": {...rmse...}, "rpe": {...}}, ...}
        ate = {}
        for rid, payload in est.items():
            if not isinstance(payload, dict):
                continue
            ate_payload = payload.get("ate")
            if isinstance(ate_payload, dict):
                ate[str(rid)] = ate_payload
    if not isinstance(ate, dict):
        return out
    rmses = []
    for value in ate.values():
        if not isinstance(value, dict):
            continue
        rmse = _safe_float(value.get("rmse"))
        if rmse is not None:
            rmses.append(rmse)
    if rmses:
        rmses.sort()
        out["ate_rmse_mean"] = sum(rmses) / float(len(rmses))
        out["ate_rmse_median"] = rmses[len(rmses) // 2]
    return out


class ResultsWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self._seen = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("run_id")
                    if rid:
                        self._seen.add(rid)

    def has(self, run_id: str) -> bool:
        return run_id in self._seen

    def append(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        rid = row.get("run_id")
        if rid:
            self._seen.add(rid)


def _flatten_impair_summary(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["impair_random_loss_p"] = spec.get("random_loss_p")
    out["impair_burst_period_s"] = spec.get("burst_period_s")
    out["impair_burst_duration_s"] = spec.get("burst_duration_s")
    bw = spec.get("bw_caps_mbps", {})
    if isinstance(bw, dict):
        out["impair_bw_cap_mbps"] = bw.get("default")
    blackouts = spec.get("blackouts")
    if isinstance(blackouts, list) and blackouts:
        blackout = blackouts[0] if isinstance(blackouts[0], dict) else {}
        out["impair_blackout_mode"] = blackout.get("mode")
        out["impair_blackout_start_s"] = blackout.get("start_s")
        out["impair_blackout_end_s"] = blackout.get("end_s")
    return out


def _build_run_id(meta: Dict[str, Any]) -> str:
    parts = [
        meta.get("sweep_type"),
        meta.get("dataset_tag"),
        f"n{meta.get('team_size')}",
        meta.get("backend"),
        f"qos-{meta.get('qos_reliability')}-{meta.get('qos_durability')}-d{meta.get('qos_depth')}",
        f"imp-{meta.get('impair_name')}",
    ]
    if meta.get("impair_seed") is not None:
        parts.append(f"seed-{meta.get('impair_seed')}")
    return _slug("_".join(str(part) for part in parts if part))


def main() -> int:
    ap = argparse.ArgumentParser(description="Targeted sweep runner for c-slam.")
    ap.add_argument("--template", default="tools/sweep_baseline.template.json", help="Baseline config template JSON")
    ap.add_argument("--dataset-root", default="dataset", help="Dataset root to discover .jrl files")
    ap.add_argument("--out-root", default=None, help="Output root (default: out/sweep_<timestamp>)")
    ap.add_argument("--modalities", nargs="+", default=["wifi", "proradio"], help="Modalities to include")
    ap.add_argument("--include-day-night", action="store_true", help="Include day/night r3 datasets in QoS/impair sweeps")
    ap.add_argument(
        "--qos-impair-datasets",
        nargs="+",
        default=None,
        help="Only run QoS/impair sweeps on these dataset paths (baseline can still include all).",
    )
    ap.add_argument(
        "--baseline-all",
        action="store_true",
        help="Include all datasets (r3/r4/r5/day/night) in baseline runs.",
    )
    ap.add_argument("--sweeps", nargs="+", default=["qos", "impair", "scale"], help="Sweeps to run")
    ap.add_argument("--rep-seeds", nargs="*", type=int, default=[0, 1, 2], help="Seeds for stochastic impairments")
    ap.add_argument("--bw-caps-mbps", nargs="*", type=float, default=[0.5, 1.0, 2.0], help="Bandwidth caps (Mbps)")
    ap.add_argument("--no-baseline", action="store_true", help="Disable baseline runs for QoS/impair datasets")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    ap.add_argument("--no-resume", action="store_true", help="Do not skip runs with ok run_status.json")
    ap.add_argument("--rerun-failed", action="store_true", help="Rerun runs with ok=false")
    ap.add_argument("--force", action="store_true", help="Rerun all even if results exist")
    ap.add_argument("--timeout-s", type=float, default=None, help="Timeout per run (seconds)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs")
    ap.add_argument("--ros-domain-base", type=int, default=30, help="Base ROS_DOMAIN_ID for per-run isolation")
    ap.add_argument("--ros-domain-span", type=int, default=200, help="Range size for ROS_DOMAIN_ID cycling")
    ap.add_argument("--no-ros-daemon-restart", action="store_true", help="Do not stop ROS 2 daemon between runs")
    args = ap.parse_args()

    template = _load_template(ROOT / args.template)
    datasets = _discover_datasets(ROOT / args.dataset_root)
    if not datasets:
        raise SystemExit("No datasets found")

    out_root = Path(args.out_root) if args.out_root else ROOT / "out" / f"sweep_{int(time.time())}"
    run_root = out_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    qos_grid = _qos_grid()
    sweeps = {sweep_name.lower() for sweep_name in args.sweeps}
    qos_impair_paths = _resolve_dataset_paths(args.qos_impair_datasets, ROOT)

    locked_paths = [
        ["factor_topic_prefix"],
        ["iface_topic_prefix"],
        ["use_sim_time"],
        ["quat_order"],
        ["robust"],
        ["robust_k"],
        ["publisher", "time_scale"],
        ["publisher", "batch_size"],
        ["publisher", "max_sleep"],
        ["publisher", "idle_gap"],
        ["publisher", "loop"],
        ["publisher", "wait_subscriber_timeout"],
        ["publisher", "include_potential_outliers"],
        ["publisher", "cpu_affinity"],
        ["solver", "relin_th"],
        ["solver", "relin_skip"],
        ["solver", "ddf_rounds"],
        ["solver", "ddf_stable_rounds"],
        ["solver", "ddf_convergence"],
        ["solver", "ddf_rot_convergence"],
        ["solver", "ddf_factor_idle_timeout"],
        ["solver", "central_batch_size"],
        ["solver", "central_idle_timeout"],
        ["solver", "include_potential_outliers"],
        ["solver", "emit_factor_acks"],
        ["solver", "emit_map_downlink"],
        ["solver", "map_topic_prefix"],
        ["solver", "cpu_affinity"],
    ]

    run_plan: List[Dict[str, Any]] = []

    def add_run(meta: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        meta = dict(meta)
        meta["config_hash"] = _hash_dict(cfg)
        meta["run_id"] = _build_run_id(meta)
        run_plan.append({"meta": meta, "config": cfg})

    if not args.no_baseline or "baseline" in sweeps:
        if args.baseline_all:
            base_datasets = [d for d in datasets if d.modality in set(args.modalities)]
        else:
            base_datasets = _select_datasets(
                datasets, sweep="baseline", modalities=args.modalities, include_day_night=args.include_day_night
            )
        for dataset in base_datasets:
            for backend in ("centralised", "decentralised"):
                meta = {
                    "sweep_type": "baseline",
                    "dataset_tag": dataset.tag,
                    "dataset_path": str(dataset.path),
                    "dataset_name": dataset.name,
                    "modality": dataset.modality,
                    "team_size": dataset.team_size,
                    "backend": backend,
                    "qos_reliability": template.get("qos", {}).get("reliability"),
                    "qos_durability": template.get("qos", {}).get("durability"),
                    "qos_depth": template.get("qos", {}).get("depth"),
                    "impair_enabled": False,
                    "impair_name": "none",
                    "impair_seed": None,
                }
                run_dir = run_root / f"{meta['dataset_tag']}" / meta["backend"] / "baseline"
                cfg = _build_run_config(template=template, dataset=dataset, backend=backend, export_path=run_dir)
                _require_locked_paths(template, cfg, locked_paths)
                add_run(meta, cfg)

    if "qos" in sweeps:
        qos_datasets = _select_datasets(
            datasets, sweep="qos", modalities=args.modalities, include_day_night=args.include_day_night
        )
        if qos_impair_paths:
            qos_datasets = [d for d in qos_datasets if str(d.path.resolve()) in qos_impair_paths]
        for dataset in qos_datasets:
            for backend in ("centralised", "decentralised"):
                for qos in qos_grid:
                    meta = {
                        "sweep_type": "qos",
                        "dataset_tag": dataset.tag,
                        "dataset_path": str(dataset.path),
                        "dataset_name": dataset.name,
                        "modality": dataset.modality,
                        "team_size": dataset.team_size,
                        "backend": backend,
                        "qos_reliability": qos["reliability"],
                        "qos_durability": qos["durability"],
                        "qos_depth": qos["depth"],
                        "impair_enabled": False,
                        "impair_name": "none",
                        "impair_seed": None,
                    }
                    run_dir = (
                        run_root
                        / f"{meta['dataset_tag']}"
                        / meta["backend"]
                        / "qos"
                        / f"{qos['reliability']}_d{qos['depth']}"
                    )
                    cfg = _build_run_config(
                        template=template, dataset=dataset, backend=backend, export_path=run_dir, qos=qos
                    )
                    _require_locked_paths(template, cfg, locked_paths)
                    add_run(meta, cfg)

    if "impair" in sweeps:
        imp_datasets = _select_datasets(
            datasets, sweep="impair", modalities=args.modalities, include_day_night=args.include_day_night
        )
        if qos_impair_paths:
            imp_datasets = [d for d in imp_datasets if str(d.path.resolve()) in qos_impair_paths]
        for dataset in imp_datasets:
            scenarios = _impairment_scenarios(dataset.robot_ids, args.bw_caps_mbps)
            duration_s = _estimate_duration_s(dataset, template)
            for backend in ("centralised", "decentralised"):
                for scenario in scenarios:
                    seeds = args.rep_seeds if scenario.get("stochastic") else [None]
                    for seed in seeds:
                        spec = dict(scenario["spec"])
                        if seed is not None:
                            spec["seed"] = int(seed)
                        spec = _with_dynamic_blackouts(spec, duration_s)
                        impair_cfg = {
                            "enabled": True,
                            "apply_to": ["publisher", "solver"],
                            "spec_json": json.dumps(spec),
                        }
                        meta = {
                            "sweep_type": "impair",
                            "dataset_tag": dataset.tag,
                            "dataset_path": str(dataset.path),
                            "dataset_name": dataset.name,
                            "modality": dataset.modality,
                            "team_size": dataset.team_size,
                            "backend": backend,
                            "qos_reliability": template.get("qos", {}).get("reliability"),
                            "qos_durability": template.get("qos", {}).get("durability"),
                            "qos_depth": template.get("qos", {}).get("depth"),
                            "impair_enabled": True,
                            "impair_name": scenario["name"],
                            "impair_seed": seed,
                        }
                        run_dir = (
                            run_root
                            / f"{meta['dataset_tag']}"
                            / meta["backend"]
                            / "impair"
                            / scenario["name"]
                        )
                        if seed is not None:
                            run_dir = run_dir / f"seed_{seed}"
                        cfg = _build_run_config(
                            template=template, dataset=dataset, backend=backend, export_path=run_dir, impair=impair_cfg
                        )
                        _require_locked_paths(template, cfg, locked_paths)
                        add_run(meta, cfg)

    if "scale" in sweeps:
        scale_datasets = _select_datasets(
            datasets, sweep="scale", modalities=args.modalities, include_day_night=args.include_day_night
        )
        for dataset in scale_datasets:
            for backend in ("centralised", "decentralised"):
                meta = {
                    "sweep_type": "scale",
                    "dataset_tag": dataset.tag,
                    "dataset_path": str(dataset.path),
                    "dataset_name": dataset.name,
                    "modality": dataset.modality,
                    "team_size": dataset.team_size,
                    "backend": backend,
                    "qos_reliability": template.get("qos", {}).get("reliability"),
                    "qos_durability": template.get("qos", {}).get("durability"),
                    "qos_depth": template.get("qos", {}).get("depth"),
                    "impair_enabled": False,
                    "impair_name": "none",
                    "impair_seed": None,
                }
                run_dir = run_root / f"{meta['dataset_tag']}" / meta["backend"] / "scale"
                cfg = _build_run_config(template=template, dataset=dataset, backend=backend, export_path=run_dir)
                _require_locked_paths(template, cfg, locked_paths)
                add_run(meta, cfg)

    run_plan_path = out_root / "run_plan.json"
    _write_json(run_plan_path, {"count": len(run_plan), "runs": [run_item["meta"] for run_item in run_plan]})

    results_path = out_root / "results.csv"
    fields = [
        "run_id",
        "config_hash",
        "sweep_type",
        "dataset_tag",
        "dataset_path",
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
        "communication_uplink_bytes_per_s",
        "communication_downlink_bytes_per_s",
        "delivery_rate_mean",
        "ate_rmse_mean",
        "ate_rmse_median",
    ]
    writer = ResultsWriter(results_path, fields)

    if args.dry_run:
        print(f"Planned runs: {len(run_plan)}")
        print(f"Plan: {run_plan_path}")
        print(f"Results: {results_path}")
        return 0

    if not args.quiet:
        print(f"Planned runs: {len(run_plan)}")
        print(f"Plan: {run_plan_path}")
        print(f"Results: {results_path}")

    for item in run_plan:
        meta = item["meta"]
        cfg = item["config"]
        run_id = meta["run_id"]
        run_dir = Path(cfg["export_path"])
        cfg_path = run_dir / "orchestrate_config.json"
        _write_json(cfg_path, cfg)

        need_run = True
        if args.force:
            need_run = True
        if not args.no_resume and run_dir.exists() and not args.force:
            ok, err = _read_status(run_dir)
            if err == "missing_run_status":
                # Missing status implies the run never completed; do not skip.
                need_run = True
            elif ok and not args.rerun_failed:
                need_run = False
            elif not ok and not args.rerun_failed:
                need_run = False
            elif not ok and args.rerun_failed:
                need_run = True

        if need_run:
            if not args.quiet:
                print(f"RUN {run_id}")
            env = os.environ.copy()
            domain_id = _compute_ros_domain_id(meta["config_hash"], args.ros_domain_base, args.ros_domain_span)
            env["ROS_DOMAIN_ID"] = str(domain_id)
            _ = _run_orchestrator(cfg_path, timeout_s=args.timeout_s, env=env)
            if not args.quiet:
                ok, err = _read_status(run_dir)
                status = "ok" if ok else f"fail:{err}"
                print(f"DONE {run_id} -> {status}")
            if not args.no_ros_daemon_restart:
                _restart_ros_daemon()

        row = _collect_result_row(meta, run_dir, cfg)
        if row and (need_run or not writer.has(run_id)):
            writer.append(row)

    return 0


def _collect_result_row(meta: Dict[str, Any], run_dir: Path, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ok, err = _read_status(run_dir)
    derived = _read_json(run_dir / "kpi_metrics" / "derived_kpis.json")
    row: Dict[str, Any] = dict(meta)
    row["run_dir"] = str(run_dir)
    row["run_ok"] = ok
    row["run_error"] = err

    impair_cfg = cfg.get("impair", {}) if isinstance(cfg.get("impair"), dict) else {}
    spec_raw = impair_cfg.get("spec_json")
    spec = {}
    if spec_raw:
        try:
            spec = json.loads(spec_raw)
        except Exception:
            spec = {}
    row.update(_flatten_impair_summary(spec))
    row.update(_summary_from_derived(derived))
    return row


if __name__ == "__main__":
    raise SystemExit(main())
