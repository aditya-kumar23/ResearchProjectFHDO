from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Dict

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

from c_slam_common.graph import default_robot_infer
from c_slam_common.loader import LoaderConfig, build_key_robot_map, iter_init_entries, load_jrl
from c_slam_common.models import InitEntry
from c_slam_ros2.qos import parse_qos_options


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_run_manifest(
    *,
    out_dir: str,
    args: argparse.Namespace,
    robot_ids: list[str],
    qos_profile: Dict[str, object],
    robust_kind: str | None,
    robust_k: float | None,
) -> None:
    git_sha = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = None
    manifest = {
        "backend": args.backend,
        "dataset": {
            "jrl_path": os.path.abspath(args.jrl),
            "jrl_sha256": _sha256_file(os.path.abspath(args.jrl)),
            "quat_order": args.quat_order,
            "include_potential_outliers": bool(args.include_potential_outliers),
        },
        "factors": {"robust_kind": robust_kind, "robust_k": robust_k},
        "ros2": {
            "factor_topic_prefix": args.factor_topic_prefix,
            "iface_topic_prefix": args.iface_topic_prefix,
            "qos": dict(qos_profile),
            "use_sim_time": bool(args.use_sim_time),
        },
        "solver": {
            "relin_th": float(args.relin_th),
            "relin_skip": int(args.relin_skip),
            "central_batch_size": int(args.central_batch_size),
            "central_idle_timeout": float(args.central_idle_timeout),
            "ddf_rounds": int(args.ddf_rounds),
            "ddf_convergence": float(args.ddf_convergence),
            "ddf_rot_convergence": float(args.ddf_rot_convergence),
            "ddf_factor_idle_timeout": float(args.ddf_factor_idle_timeout),
            "ddf_stable_rounds": int(args.ddf_stable_rounds),
            "emit_factor_acks": bool(getattr(args, "emit_factor_acks", False)),
        },
        "robots": list(robot_ids),
        "kpi": {"enabled": bool(args.kpi), "kpi_dir": os.path.abspath(args.kpi_dir) if args.kpi_dir else None},
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_sha": git_sha,
            "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION"),
            "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
            "C_SLAM_IMPAIR": os.environ.get("C_SLAM_IMPAIR"),
            "C_SLAM_IMPAIR_FILE": os.environ.get("C_SLAM_IMPAIR_FILE"),
            "C_SLAM_RESOURCE_INTERVAL": os.environ.get("C_SLAM_RESOURCE_INTERVAL"),
        },
        "timestamps": {"start_wall": time.time()},
    }
    _write_json(os.path.join(out_dir, "run_manifest.json"), manifest)


def _write_run_status(out_dir: str, *, ok: bool, backend: str, error: str | None = None) -> None:
    payload = {"ok": bool(ok), "backend": str(backend), "ts_wall": time.time(), "error": error}
    _write_json(os.path.join(out_dir, "run_status.json"), payload)


def _merge_robustness_metrics(kpi_dir: str, robot_ids: list[str]) -> None:
    """Merge robustness metrics from the factor publisher + per-agent iface delivery stats."""
    factors_path = os.path.join(kpi_dir, "robustness_factors.json")
    out_path = os.path.join(kpi_dir, "robustness_metrics.json")

    merged_topics: Dict[str, Dict[str, object]] = {}

    def _merge_topics(src: Dict[str, object]) -> None:
        topics = src.get("stats", {}).get("topics", {}) if isinstance(src, dict) else {}
        if not isinstance(topics, dict):
            return
        for topic, s in topics.items():
            if not isinstance(s, dict):
                continue
            tgt = merged_topics.get(str(topic)) or {"attempts": 0, "drops": 0, "delivered": 0}
            for key in ("attempts", "drops", "delivered", "bytes_attempted", "bytes_delivered"):
                if key in s:
                    try:
                        tgt[key] = int(tgt.get(key, 0)) + int(s.get(key, 0) or 0)
                    except Exception:
                        pass
            merged_topics[str(topic)] = tgt

    if not os.path.exists(factors_path) and os.environ.get("C_SLAM_EXPECT_FACTOR_METRICS") == "1":
        deadline = time.time() + 8.0
        while time.time() < deadline and not os.path.exists(factors_path):
            time.sleep(0.1)

    if os.path.exists(factors_path):
        try:
            with open(factors_path, "r", encoding="utf-8") as f:
                _merge_topics(json.load(f))
        except Exception:
            pass

    for rid in robot_ids:
        p = os.path.join(kpi_dir, f"iface_delivery_{rid}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                _merge_topics(json.load(f))
        except Exception:
            continue

    for topic, s in list(merged_topics.items()):
        attempts = int(s.get("attempts", 0) or 0)
        delivered = int(s.get("delivered", 0) or 0)
        s["delivery_rate"] = (float(delivered) / float(attempts)) if attempts > 0 else None
        merged_topics[topic] = s

    _write_json(
        out_path,
        {"stats": {"topics": merged_topics, "generated_by": {"component": "main_merge"}}},
    )


def _export_csv_per_robot(estimate: "gtsam.Values", by_robot_keys: Dict[str, set], out_dir: str, gb=None) -> None:
    import csv

    _ensure_dir(out_dir)
    for rid, keys in by_robot_keys.items():
        rows = []
        for k in sorted(keys, key=lambda x: str(x)):
            try:
                if not estimate.exists(k):
                    continue
                p = estimate.atPose3(k)
                t = p.translation()
                r = p.rotation()
                try:
                    tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
                except Exception:
                    vec = np.asarray(t, dtype=float)
                    tx, ty, tz = float(vec[0]), float(vec[1]), float(vec[2])
                try:
                    qw, qx, qy, qz = map(float, getattr(r, "quaternion", lambda: [1, 0, 0, 0])())
                except Exception:
                    qw, qx, qy, qz = float(r.w()), float(r.x()), float(r.y()), float(r.z())
                key_label = gb.denormalize_key(k) if gb else k
                rows.append({"key": key_label, "x": tx, "y": ty, "z": tz, "qw": qw, "qx": qx, "qy": qy, "qz": qz})
            except Exception:
                continue
        out_path = os.path.join(out_dir, f"trajectory_{rid}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["key", "x", "y", "z", "qw", "qx", "qy", "qz"])
            w.writeheader()
            for row in rows:
                w.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="c-slam minimal CLI (ROS2 + iSAM2).")
    ap.add_argument("--jrl", required=True, help="Path to .jrl JSON")
    ap.add_argument("--export-path", required=True, help="Directory to write outputs")
    ap.add_argument("--backend", choices=["centralised", "decentralised"], required=True)
    ap.add_argument("--log", default="INFO", help="Logging level")

    ap.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    ap.add_argument("--include-potential-outliers", action="store_true")
    ap.add_argument("--robust", choices=["none", "huber", "cauchy"], default="cauchy")
    ap.add_argument("--robust-k", type=float, default=1.0)

    ap.add_argument("--kpi", action="store_true", help="Enable KPI metrics export")
    ap.add_argument("--kpi-dir", default=None, help="Directory for KPI metrics (default: <export-path>/kpi_metrics)")
    ap.add_argument("--emit-factor-acks", action="store_true", help="Publish ACKs for factor batches (for delivery-rate KPI)")
    ap.add_argument(
        "--emit-map-downlink",
        action="store_true",
        help="Publish per-robot map updates over ROS2 (downlink is then subject to QoS/impairment)",
    )

    ap.add_argument("--use-sim-time", action="store_true", help="Set C_SLAM_USE_SIM_TIME=1 for ROS nodes")

    ap.add_argument("--qos-reliability", choices=["reliable", "best_effort"], default="reliable")
    ap.add_argument("--qos-durability", choices=["volatile", "transient_local"], default="volatile")
    ap.add_argument("--qos-depth", type=int, default=10)

    ap.add_argument("--factor-topic-prefix", default="/c_slam/factor_batch")
    ap.add_argument("--iface-topic-prefix", default="/c_slam/iface")
    ap.add_argument("--map-topic-prefix", default="/c_slam/map", help="Topic prefix for centralised map downlink")

    ap.add_argument("--relin-th", type=float, default=0.05)
    ap.add_argument("--relin-skip", type=int, default=5)
    ap.add_argument("--central-batch-size", type=int, default=1)
    ap.add_argument("--central-idle-timeout", type=float, default=5.0)
    ap.add_argument(
        "--post-input-settle-updates",
        type=int,
        default=0,
        help="Post-input empty solver updates after input ends (centralised). 0=off, N>0=max updates (stops early once stable window observed), N<0=until stable (safety cap)",
    )

    ap.add_argument("--ddf-rounds", type=int, default=8)
    ap.add_argument("--ddf-convergence", type=float, default=5e-3)
    ap.add_argument("--ddf-rot-convergence", type=float, default=5e-3)
    ap.add_argument("--ddf-factor-idle-timeout", type=float, default=5.0)
    ap.add_argument("--ddf-stable-rounds", type=int, default=3, help="Consecutive stable rounds required to stop (decentralised)")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log).upper(), logging.INFO))

    if args.use_sim_time:
        os.environ["C_SLAM_USE_SIM_TIME"] = "1"

    if gtsam is None:
        raise RuntimeError("gtsam bindings are required")

    cfg = LoaderConfig(quaternion_order=args.quat_order, include_potential_outliers=bool(args.include_potential_outliers))
    doc = load_jrl(args.jrl, cfg)
    robot_map = build_key_robot_map(doc)
    init_lookup: Dict[str, InitEntry] = {str(it.key): it for it in iter_init_entries(doc, cfg)}
    robot_ids = sorted(set(robot_map.values()) or {default_robot_infer(k) for k in init_lookup.keys()})

    robust_kind = None if args.robust == "none" else args.robust
    robust_k = None if robust_kind is None else float(args.robust_k)
    qos_profile = parse_qos_options(reliability=args.qos_reliability, durability=args.qos_durability, depth=args.qos_depth)

    out_dir = os.path.abspath(args.export_path)
    _ensure_dir(out_dir)
    kpi_dir = os.path.abspath(args.kpi_dir) if args.kpi_dir else os.path.join(out_dir, "kpi_metrics")
    if args.kpi:
        _ensure_dir(kpi_dir)
    _write_run_manifest(out_dir=out_dir, args=args, robot_ids=robot_ids, qos_profile=qos_profile, robust_kind=robust_kind, robust_k=robust_k)

    if args.backend == "centralised":
        from c_slam_central.central_backend import run_centralised_from_ros2
        from c_slam_common.bandwidth import BandwidthTracker
        from c_slam_common.latency import LatencyTracker
        from c_slam_common.kpi_logging import KPILogger
        from c_slam_common.resource_monitor import ResourceMonitor
        from c_slam_common.metrics import align_and_ate_per_robot, compute_rpe_per_robot
        from c_slam_common.loader import groundtruth_by_robot_key
        from c_slam_common import viz as viz_utils
        from tools import kpi_derive

        kpi_logger = (
            KPILogger(
                enabled=bool(args.kpi),
                extra_fields={"solver": "isam2", "backend": "centralised"},
                log_path=os.path.join(kpi_dir, "kpi_events.jsonl") if args.kpi else None,
                emit_to_logger=False,
            )
            if args.kpi
            else None
        )
        bw_tracker = BandwidthTracker() if args.kpi else None
        lat_tracker = LatencyTracker() if args.kpi else None
        resource = ResourceMonitor() if args.kpi else None
        if resource:
            resource.start(
                metadata={
                    "backend": "centralised",
                    "jrl": os.path.abspath(args.jrl),
                    "include_potential_outliers": bool(args.include_potential_outliers),
                    "robust_kind": robust_kind,
                    "robust_k": robust_k,
                    "qos": dict(qos_profile),
                }
            )

        ack_publisher = None
        if args.emit_factor_acks:
            try:
                from c_slam_ros2.ack import Ros2AckPublisher

                ack_publisher = Ros2AckPublisher(topic_prefix=args.factor_topic_prefix, qos_profile=qos_profile)
            except Exception:
                ack_publisher = None

        map_broadcaster = None
        if args.emit_map_downlink:
            try:
                from c_slam_ros2.map_broadcast import Ros2MapBroadcaster

                map_broadcaster = Ros2MapBroadcaster(
                    robot_ids=robot_ids, topic_prefix=args.map_topic_prefix, qos_profile=qos_profile
                )
            except Exception as exc:  # pragma: no cover - ROS2 errors are runtime-only
                logging.getLogger("c_slam").warning("Map downlink disabled (init failed): %s", exc)
                map_broadcaster = None

        estimate = None
        gb = None
        error = None
        try:
            estimate, gb = run_centralised_from_ros2(
                topic_prefix=args.factor_topic_prefix,
                robot_ids=robot_ids,
                qos_profile=qos_profile,
                init_lookup=init_lookup,
                robot_map=robot_map,
                batch_size=args.central_batch_size,
                relin_threshold=args.relin_th,
                relin_skip=args.relin_skip,
                robust_kind=robust_kind,
                robust_k=robust_k,
                kpi=kpi_logger,
                bandwidth=bw_tracker,
                latency=lat_tracker,
                ack_publisher=ack_publisher,
                idle_timeout=float(args.central_idle_timeout),
                map_broadcaster=map_broadcaster,
                post_input_settle_updates=int(args.post_input_settle_updates),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logging.getLogger("c_slam").error("Centralised run failed: %s", error)
        finally:
            if ack_publisher is not None:
                try:
                    ack_publisher.close()
                except Exception:
                    pass
            if map_broadcaster is not None:
                try:
                    map_broadcaster.close()
                except Exception:
                    pass
        if error is None and gb is not None:
            try:
                total = int(gb.counts.get("prior", 0) or 0) + int(gb.counts.get("between", 0) or 0)
            except Exception:
                total = 0
            if total <= 0:
                error = "RuntimeError: No factors ingested (check impairment/QoS/topic prefixes)."

        traj_dir = os.path.join(out_dir, "trajectories")
        if estimate is not None and gb is not None and error is None:
            _export_csv_per_robot(estimate, gb.by_robot_keys, traj_dir, gb)

        if args.kpi:
            gt_map = groundtruth_by_robot_key(doc, cfg)
            metrics_out = {}
            if estimate is not None and gb is not None:
                try:
                    viz_utils.plot_trajectories_2d(estimate, gb.by_robot_keys, os.path.join(kpi_dir, "traj_xy.png"))
                except Exception:
                    pass
            if gt_map and estimate is not None and gb is not None:
                align = align_and_ate_per_robot(estimate, gb.by_robot_keys, gt_map, key_label=gb.denormalize_key)
                rpe = compute_rpe_per_robot(estimate, gb.by_robot_keys, gt_map, align, key_label=gb.denormalize_key)
                metrics_out["ate"] = align
                metrics_out["rpe"] = rpe
                try:
                    viz_utils.plot_est_vs_gt_xy_aligned(
                        estimate,
                        gb.by_robot_keys,
                        gt_map,
                        align,
                        os.path.join(kpi_dir, "traj_xy_gt.png"),
                    )
                except Exception:
                    pass
            os.makedirs(kpi_dir, exist_ok=True)
            if bw_tracker:
                bw_tracker.export_json(os.path.join(kpi_dir, "bandwidth_stats.json"))
            if lat_tracker:
                lat_tracker.export_json(os.path.join(kpi_dir, "latency_metrics.json"))
            if kpi_logger:
                kpi_logger.close()
            if map_broadcaster:
                try:
                    map_broadcaster.export_delivery_metrics(os.path.join(kpi_dir, "map_delivery.json"))
                except Exception:
                    pass
            if resource:
                resource.stop()
                resource.export_json(os.path.join(kpi_dir, "resource_profile.json"))
            if metrics_out:
                with open(os.path.join(kpi_dir, "estimation_metrics.json"), "w", encoding="utf-8") as f:
                    import json as _json

                    _json.dump(metrics_out, f, indent=2)
            try:
                _merge_robustness_metrics(kpi_dir, robot_ids)
            except Exception:
                pass
            try:
                derived = kpi_derive.derive_kpis_for_run(out_dir)
                kpi_derive.write_json(os.path.join(kpi_dir, "derived_kpis.json"), derived)
            except Exception:
                pass

        _write_run_status(out_dir, ok=(error is None), backend="centralised", error=error)
        if error is None:
            logging.getLogger("c_slam").info("Centralised run complete: wrote %s", traj_dir)
            return 0
        return 2

    if args.backend == "decentralised":
        from c_slam_decentral.mp_runner import run_multiprocess_ddf

        traj_dir = os.path.join(out_dir, "trajectories")
        _ensure_dir(traj_dir)
        from c_slam_common.loader import groundtruth_by_robot_key

        gt_map = groundtruth_by_robot_key(doc, cfg)
        error = None
        pids: list[int] = []
        namespaced_prefix = args.iface_topic_prefix
        try:
            pids, namespaced_prefix = run_multiprocess_ddf(
                robot_map=robot_map,
                init_lookup=init_lookup,
                robot_ids=robot_ids,
                robust_kind=robust_kind,
                robust_k=robust_k,
                relin_th=args.relin_th,
                relin_skip=args.relin_skip,
                factor_topic_prefix=args.factor_topic_prefix,
                iface_topic_prefix=args.iface_topic_prefix,
                qos_profile=qos_profile,
                out_dir=traj_dir,
                ddf_rounds=args.ddf_rounds,
                convergence_tol=args.ddf_convergence,
                rotation_tol=args.ddf_rot_convergence,
                idle_timeout=float(args.ddf_factor_idle_timeout),
                metrics_dir=kpi_dir if args.kpi else None,
                groundtruth_map=gt_map if args.kpi else None,
                stable_rounds=int(args.ddf_stable_rounds),
                emit_factor_acks=bool(args.emit_factor_acks),
                resource_metadata={
                    "backend": "decentralised",
                    "jrl": os.path.abspath(args.jrl),
                    "include_potential_outliers": bool(args.include_potential_outliers),
                    "robust_kind": robust_kind,
                    "robust_k": robust_k,
                    "qos": dict(qos_profile),
                }
                if args.kpi
                else None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logging.getLogger("c_slam").error("Decentralised run failed: %s", error)
        if args.kpi:
            try:
                _merge_robustness_metrics(kpi_dir, robot_ids)
            except Exception:
                pass
            try:
                from tools import kpi_derive

                derived = kpi_derive.derive_kpis_for_run(out_dir)
                kpi_derive.write_json(os.path.join(kpi_dir, "derived_kpis.json"), derived)
            except Exception:
                pass
        _write_run_status(out_dir, ok=(error is None), backend="decentralised", error=error)
        if error is None:
            logging.getLogger("c_slam").info("Decentralised run complete: pids=%s iface_prefix=%s", pids, namespaced_prefix)
            return 0
        return 2

    raise RuntimeError(f"Unknown backend {args.backend}")


if __name__ == "__main__":
    raise SystemExit(main())
