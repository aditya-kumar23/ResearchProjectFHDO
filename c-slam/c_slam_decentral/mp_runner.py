from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_slam_common.models import InitEntry
from c_slam_common.kpi_logging import KPILogger
from c_slam_common.bandwidth import BandwidthTracker
from c_slam_common.latency import LatencyTracker
from c_slam_common.metrics import align_and_ate_per_robot, compute_rpe_per_robot
from c_slam_common.resource_monitor import ResourceMonitor
from c_slam_decentral.agents import DecentralizedRobotAgent
from c_slam_decentral.communication import InterfaceMessage, Ros2PeerBus
from c_slam_ros2.factor_source import ROS2FactorSource
from c_slam_ros2.interface_msg import decode_interface_message
from c_slam_ros2.factor_batch import decode_factor_batch
from c_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("c_slam.decentral.mp")


def _configure_agent_logging(*, rid: str, metrics_dir: Optional[str]) -> None:
    level_name = str(os.environ.get("C_SLAM_LOG", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if metrics_dir:
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            handlers.append(logging.FileHandler(os.path.join(metrics_dir, f"agent_{rid}.log"), mode="w", encoding="utf-8"))
        except Exception:
            pass
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
        format="%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s",
    )


def _export_csv_per_robot(estimate: "gtsam.Values", keys: Iterable, out_path: str, gb=None) -> None:
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
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "x", "y", "z", "qw", "qx", "qy", "qz"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _agent_loop(
    rid: str,
    robot_ids: List[str],
    *,
    robot_map: Dict[str, str],
    init_lookup: Dict[str, InitEntry],
    robust_kind: Optional[str],
    robust_k: Optional[float],
    relin_th: float,
    relin_skip: int,
    factor_topic_prefix: str,
    iface_topic_prefix: str,
    qos_profile: Dict[str, object],
    out_dir: str,
    ddf_rounds: int,
    convergence_tol: float,
    rotation_tol: float,
    idle_timeout: float,
    metrics_dir: Optional[str] = None,
    groundtruth_map: Optional[Dict[str, Dict[str, tuple]]] = None,
    stable_rounds: int = 1,
    emit_factor_acks: bool = False,
    resource_metadata: Optional[Dict[str, object]] = None,
) -> None:
    _configure_agent_logging(rid=rid, metrics_dir=metrics_dir)
    if gtsam is None:
        raise RuntimeError("gtsam required for decentralised backend")

    resource = ResourceMonitor() if metrics_dir else None
    if resource is not None:
        meta = dict(resource_metadata or {})
        meta.update({"rid": rid, "role": "agent"})
        try:
            resource.start(metadata=meta)
        except Exception:
            resource = None

    try:
        agent = DecentralizedRobotAgent(
            rid,
            robot_map,
            init_lookup,
            robust_kind=robust_kind,
            robust_k=robust_k,
            relin_threshold=relin_th,
            relin_skip=relin_skip,
        )

        kpi_logger: Optional[KPILogger] = None
        if metrics_dir:
            try:
                os.makedirs(metrics_dir, exist_ok=True)
                kpi_logger = KPILogger(
                    enabled=True,
                    extra_fields={"solver": "ddf_sam", "robot": rid},
                    log_path=os.path.join(metrics_dir, f"kpi_events_{rid}.jsonl"),
                    emit_to_logger=False,
                )
            except Exception:
                kpi_logger = None

        ack_publisher = None
        seen_factor_msg_ids: set[str] = set()
        if emit_factor_acks:
            try:
                from c_slam_ros2.ack import Ros2AckPublisher

                ack_publisher = Ros2AckPublisher(topic_prefix=factor_topic_prefix, qos_profile=qos_profile)
            except Exception:
                ack_publisher = None

        # Per-agent factor ingest
        with ROS2FactorSource(
            topic_prefix=factor_topic_prefix,
            robot_ids=[rid],
            qos_profile=qos_profile,
            queue_size=0,
            spin_timeout=0.1,
            idle_timeout=idle_timeout,
        ) as src:
            ingested_factors = 0
            last_factor_stamp: Optional[float] = None
            for f in src.iter_factors():
                agent.ingest_factor(f)
                ingested_factors += 1
                try:
                    if getattr(f, "stamp", None) is not None:
                        last_factor_stamp = float(getattr(f, "stamp"))
                except Exception:
                    pass
                if ack_publisher is not None:
                    try:
                        msg_id = getattr(f, "__ros2_msg_id", None)
                        if not msg_id:
                            continue
                        msg_id = str(msg_id)
                        if msg_id in seen_factor_msg_ids:
                            continue
                        seen_factor_msg_ids.add(msg_id)
                        ack_publisher.publish_ack(
                            str(rid),
                            message_id=msg_id,
                            send_ts_mono=getattr(f, "send_ts_mono", None),
                            send_ts_wall=getattr(f, "send_ts_wall", None),
                            use_ts_mono=time.perf_counter(),
                            use_ts_wall=time.time(),
                            bytes=getattr(f, "__ros2_msg_bytes", None),
                        )
                    except Exception:
                        pass
            if kpi_logger:
                try:
                    kpi_logger.emit(
                        "input_end",
                        ingested_factors=int(ingested_factors),
                        last_factor_stamp=last_factor_stamp,
                        role="agent",
                    )
                except Exception:
                    pass

        bus = Ros2PeerBus(robot_ids, local_robot_id=rid, topic_prefix=iface_topic_prefix, qos_profile=qos_profile)
        try:
            # Initial solve and broadcast
            agent.solve_round(iteration=0, kpi=kpi_logger)
            outgoing = agent.interface_messages(iteration=0)
            for msg in outgoing:
                bus.post(msg)

            def _msg_state(msg: InterfaceMessage):
                t = np.array([msg.translation.x, msg.translation.y, msg.translation.z], dtype=float)
                q = np.array([msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z], dtype=float)
                q = q / (np.linalg.norm(q) or 1.0)
                return t, q

            last_state = {(m.sender, m.key): _msg_state(m) for m in outgoing}

            converged = False
            iteration = 0
            stable_count = 0
            for iteration in range(1, int(ddf_rounds) + 1):
                incoming = bus.drain(rid)
                if incoming:
                    agent.receive_interface_messages(incoming)
                agent.solve_round(iteration=iteration, kpi=kpi_logger)
                outgoing = agent.interface_messages(iteration=iteration)
                if not outgoing:
                    stable_count += 1
                    if stable_count >= max(1, int(stable_rounds)):
                        converged = True
                        break
                    else:
                        continue
                for msg in outgoing:
                    bus.post(msg)

                state = {(m.sender, m.key): _msg_state(m) for m in outgoing}
                if last_state:
                    td, rd = [], []
                    for key, (t, q) in state.items():
                        prev = last_state.get(key)
                        if prev is None:
                            continue
                        pt, pq = prev
                        td.append(float(np.linalg.norm(t - pt)))
                        dot = float(abs(np.dot(q, pq)))
                        dot = max(-1.0, min(1.0, dot))
                        rd.append(float(2.0 * np.arccos(dot)))
                    if td and rd and max(td) < float(convergence_tol) and max(rd) < float(rotation_tol):
                        stable_count += 1
                        if stable_count >= max(1, int(stable_rounds)):
                            converged = True
                            break
                    else:
                        stable_count = 0
                last_state = state

            if kpi_logger:
                try:
                    kpi_logger.emit(
                        "ddf_stop",
                        converged=bool(converged),
                        iteration=int(iteration),
                        reason="stable" if converged else "round_limit",
                        ddf_rounds=int(ddf_rounds),
                        stable_rounds=int(stable_rounds),
                        convergence_tol=float(convergence_tol),
                        rotation_tol=float(rotation_tol),
                    )
                except Exception:
                    pass

            # Export per-robot CSV trajectory
            est = agent.estimate_snapshot()
            gb = agent.graph_snapshot()
            if est is not None and gb is not None:
                keys = gb.by_robot_keys.get(rid, set())
                if keys:
                    _export_csv_per_robot(est, keys, os.path.join(out_dir, f"trajectory_{rid}.csv"), gb)
            if metrics_dir and groundtruth_map and est is not None and gb is not None:
                try:
                    os.makedirs(metrics_dir, exist_ok=True)
                    ate = align_and_ate_per_robot(
                        est,
                        {rid: gb.by_robot_keys.get(rid, set())},
                        groundtruth_map,
                        key_label=gb.denormalize_key,
                    )
                    rpe = compute_rpe_per_robot(
                        est,
                        {rid: gb.by_robot_keys.get(rid, set())},
                        groundtruth_map,
                        ate,
                        key_label=gb.denormalize_key,
                    )
                    metrics_out = {"ate": ate.get(rid), "rpe": rpe.get(rid)}
                    with open(os.path.join(metrics_dir, f"estimation_metrics_{rid}.json"), "w", encoding="utf-8") as fh:
                        json.dump(metrics_out, fh, indent=2)
                except Exception:
                    pass

            status = {"pid": os.getpid(), "rid": rid, "converged": converged, "iterations": iteration}
            with open(os.path.join(out_dir, f"agent_status_{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        finally:
            if resource is not None and metrics_dir:
                try:
                    resource.stop()
                except Exception:
                    pass
                try:
                    resource.export_json(os.path.join(metrics_dir, f"resource_profile_{rid}.json"))
                except Exception:
                    pass
            if metrics_dir:
                try:
                    bus.export_delivery_metrics(os.path.join(metrics_dir, f"iface_delivery_{rid}.json"))
                except Exception:
                    pass
            try:
                bus.close()
            except Exception:
                pass
            if ack_publisher is not None:
                try:
                    ack_publisher.close()
                except Exception:
                    pass
    except Exception as exc:
        # Persist a per-agent crash report for thesis failure-mode analysis.
        out_base = metrics_dir or out_dir
        if resource is not None and metrics_dir:
            try:
                resource.stop()
            except Exception:
                pass
            try:
                resource.export_json(os.path.join(metrics_dir, f"resource_profile_{rid}.json"))
            except Exception:
                pass
        try:
            os.makedirs(out_base, exist_ok=True)
            report_path = os.path.join(out_base, f"agent_error_{rid}.txt")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(f"{type(exc).__name__}: {exc}\n\n")
                fh.write(traceback.format_exc())
        except Exception:
            pass
        logging.getLogger("c_slam.decentral.agent").exception("Agent %s crashed", rid)
        raise


def run_multiprocess_ddf(
    *,
    robot_map: Dict[str, str],
    init_lookup: Dict[str, InitEntry],
    robot_ids: List[str],
    robust_kind: Optional[str],
    robust_k: Optional[float],
    relin_th: float,
    relin_skip: int,
    factor_topic_prefix: str,
    iface_topic_prefix: str,
    qos_profile: Dict[str, object],
    out_dir: str,
    ddf_rounds: int,
    convergence_tol: float,
    rotation_tol: float,
    idle_timeout: float,
    metrics_dir: Optional[str] = None,
    groundtruth_map: Optional[Dict[str, Dict[str, tuple]]] = None,
    stable_rounds: int = 1,
    emit_factor_acks: bool = False,
    resource_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], str]:
    os.makedirs(out_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    namespaced_prefix = f"{iface_topic_prefix.rstrip('/')}/run_{run_id}"
    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
    procs: List[mp.Process] = []
    proc_rids: Dict[int, str] = {}
    pids: List[int] = []
    os.makedirs(metrics_dir, exist_ok=True) if metrics_dir else None

    # Optional realtime monitors for factor/interface traffic
    node = None
    rclpy = None  # type: ignore
    subscriptions = []
    bw_tracker = BandwidthTracker() if metrics_dir else None
    lat_tracker = LatencyTracker() if metrics_dir else None
    if metrics_dir:
        try:
            import rclpy as _rclpy  # type: ignore
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
            from std_msgs.msg import UInt8MultiArray  # type: ignore

            rclpy = _rclpy
            rclpy.init(args=None)
            node = rclpy.create_node("c_slam_ddf_monitor")
            configure_sim_time(node)
            qos = QoSProfile(
                depth=int(qos_profile.get("depth", 10)),
                reliability=ReliabilityPolicy.RELIABLE
                if str(qos_profile.get("reliability", "reliable")).lower() == "reliable"
                else ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
                if str(qos_profile.get("durability", "volatile")).lower() == "transient_local"
                else DurabilityPolicy.VOLATILE,
            )

            def _factor_cb(expected_rid: str):
                def _cb(msg):
                    data = bytes(getattr(msg, "data", []) or [])
                    if bw_tracker:
                        bw_tracker.add_uplink(f"factor/{expected_rid}", len(data))
                    if lat_tracker:
                        try:
                            factors = decode_factor_batch(data)
                        except Exception:
                            return
                        now_wall = time.time()
                        now_mono = time.perf_counter()
                        for f in factors:
                            try:
                                lat_tracker.record_ingest(
                                    type(f).__name__,
                                    float(getattr(f, "stamp", 0.0)),
                                    expected_rid,
                                    ingest_wall=now_wall,
                                    ingest_mono=now_mono,
                                    metadata={
                                        "key": str(getattr(f, "key", getattr(f, "key1", ""))),
                                        "key2": str(getattr(f, "key2", "")) if hasattr(f, "key2") else None,
                                        "send_ts_mono": getattr(f, "send_ts_mono", None),
                                        "send_ts_wall": getattr(f, "send_ts_wall", None),
                                    },
                                )
                            except Exception:
                                continue
                return _cb

            for rid in robot_ids:
                topic = f"{factor_topic_prefix.rstrip('/')}/{rid}"
                try:
                    sub = node.create_subscription(UInt8MultiArray, topic, _factor_cb(rid), qos)
                    subscriptions.append(sub)
                except Exception:
                    continue

            def _iface_cb(expected_rid: str):
                def _cb(msg):
                    data = bytes(getattr(msg, "data", []) or [])
                    if bw_tracker:
                        bw_tracker.add_uplink(f"iface/{expected_rid}", len(data))
                    if lat_tracker:
                        try:
                            iface = decode_interface_message(data)
                        except Exception:
                            return
                        now_wall = time.time()
                        now_mono = time.perf_counter()
                        meta = {
                            "sender": iface.sender,
                            "receiver": iface.receiver,
                            "key": str(getattr(iface, "key", "")),
                            "iteration": int(getattr(iface, "iteration", 0)),
                            "send_ts_mono": getattr(iface, "sent_mono_time", None),
                            "send_ts_wall": getattr(iface, "sent_wall_time", None),
                        }
                        lat_tracker.record_ingest(
                            "InterfaceMessage",
                            float(getattr(iface, "stamp", 0.0)),
                            iface.receiver,
                            ingest_wall=now_wall,
                            ingest_mono=now_mono,
                            metadata=meta,
                        )
                return _cb

            for rid in robot_ids:
                topic = f"{namespaced_prefix}/{rid}"
                try:
                    sub = node.create_subscription(UInt8MultiArray, topic, _iface_cb(rid), qos)
                    subscriptions.append(sub)
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Monitor init failed: %s", exc)
            node = None
            rclpy = None
    for rid in robot_ids:
        p = ctx.Process(
            target=_agent_loop,
            args=(rid, robot_ids),
            kwargs=dict(
                robot_map=robot_map,
                init_lookup=init_lookup,
                robust_kind=robust_kind,
                robust_k=robust_k,
                relin_th=relin_th,
                relin_skip=relin_skip,
                factor_topic_prefix=factor_topic_prefix,
                iface_topic_prefix=namespaced_prefix,
                qos_profile=qos_profile,
                out_dir=out_dir,
                ddf_rounds=ddf_rounds,
                convergence_tol=convergence_tol,
                rotation_tol=rotation_tol,
                idle_timeout=idle_timeout,
                metrics_dir=metrics_dir,
                groundtruth_map=groundtruth_map,
                stable_rounds=stable_rounds,
                emit_factor_acks=emit_factor_acks,
                resource_metadata=resource_metadata,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)
        pids.append(p.pid)
        try:
            proc_rids[int(p.pid)] = str(rid)
        except Exception:
            pass

    try:
        while any(p.is_alive() for p in procs):
            if node is not None and rclpy is not None:
                try:
                    rclpy.spin_once(node, timeout_sec=0.05)
                except KeyboardInterrupt:
                    break
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)
    finally:
        for p in procs:
            try:
                p.join(timeout=0.1)
            except Exception:
                pass
        if node is not None:
            try:
                for sub in subscriptions:
                    try:
                        node.destroy_subscription(sub)
                    except Exception:
                        pass
                node.destroy_node()
            except Exception:
                pass
        if rclpy is not None:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    # Detect agent crashes (non-zero exit code) so the caller can mark the run as failed.
    bad_exitcodes: Dict[str, int] = {}
    for p in procs:
        try:
            code = p.exitcode
        except Exception:
            code = None
        if code is None:
            continue
        try:
            code_i = int(code)
        except Exception:
            continue
        if code_i == 0:
            continue
        rid = proc_rids.get(int(getattr(p, "pid", 0) or 0), str(getattr(p, "name", "unknown")))
        bad_exitcodes[str(rid)] = code_i
    if bad_exitcodes:
        try:
            out_path = os.path.join(metrics_dir or out_dir, "agent_exitcodes.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"exitcodes": bad_exitcodes}, fh, indent=2)
        except Exception:
            pass

    # Export monitors
    if metrics_dir:
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            if bw_tracker:
                bw_tracker.export_json(os.path.join(metrics_dir, "bandwidth_stats.json"))
            if lat_tracker:
                lat_tracker.export_json(os.path.join(metrics_dir, "latency_metrics.json"))
        except Exception:
            pass

    # Merge per-agent KPI logs
    if metrics_dir:
        merged_events = []
        merged_path = os.path.join(metrics_dir, "kpi_events.jsonl")
        try:
            import json as _json

            for rid in robot_ids:
                agent_log = os.path.join(metrics_dir, f"kpi_events_{rid}.jsonl")
                if not os.path.exists(agent_log):
                    continue
                with open(agent_log, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = _json.loads(line)
                            merged_events.append(doc)
                        except Exception:
                            continue
            if merged_events:
                merged_events.sort(key=lambda ev: ev.get("ts", 0.0))
            with open(merged_path, "w", encoding="utf-8") as fh:
                for ev in merged_events:
                    fh.write(_json.dumps(ev, sort_keys=True) + "\n")
        except Exception:
            try:
                with open(merged_path, "w", encoding="utf-8") as fh:
                    fh.write("")
            except Exception:
                pass
        # Merge per-agent estimation metrics
        try:
            est_agg = {}
            traj_data = []
            for rid in robot_ids:
                p = os.path.join(metrics_dir, f"estimation_metrics_{rid}.json")
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as fh:
                            est_agg[rid] = json.load(fh)
                    except Exception:
                        est_agg[rid] = {}
                # Collect trajectory samples (with z for alignment) for plotting if available
                traj_csv = os.path.join(os.path.dirname(metrics_dir), "trajectories", f"trajectory_{rid}.csv")
                if os.path.exists(traj_csv):
                    import csv

                    xyz = []
                    keys_in_order = []
                    with open(traj_csv, "r", encoding="utf-8") as cf:
                        reader = csv.DictReader(cf)
                        for row in reader:
                            try:
                                keys_in_order.append(str(row.get("key", "")))
                                xyz.append(
                                    [
                                        float(row.get("x", 0.0)),
                                        float(row.get("y", 0.0)),
                                        float(row.get("z", 0.0)),
                                    ]
                                )
                            except Exception:
                                continue
                    if xyz:
                        traj_data.append((rid, np.asarray(xyz, dtype=float), keys_in_order))
            if est_agg:
                with open(os.path.join(metrics_dir, "estimation_metrics.json"), "w", encoding="utf-8") as fh:
                    json.dump(est_agg, fh, indent=2)
            if traj_data:
                try:
                    plt.figure(figsize=(8, 6))
                    for rid, xyz, _keys in traj_data:
                        align = est_agg.get(rid, {}).get("ate", {})
                        R = np.asarray(align.get("R", np.eye(3)), dtype=float)
                        if R.shape != (3, 3):
                            R = np.eye(3)
                        t = np.asarray(align.get("t", np.zeros(3)), dtype=float)
                        if t.shape != (3,):
                            t = np.zeros(3)
                        try:
                            s = float(align.get("s", 1.0) or 1.0)
                        except Exception:
                            s = 1.0
                        aligned = (xyz @ R.T) * s + t
                        label = f"{rid} est (aligned)" if align else f"{rid} est"
                        plt.plot(aligned[:, 0], aligned[:, 1], label=label)
                    plt.axis("equal")
                    plt.xlabel("x [m]")
                    plt.ylabel("y [m]")
                    plt.legend()
                    plt.title("Trajectories (XY)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(metrics_dir, "traj_xy.png"), dpi=150)
                    plt.close()
                except Exception:
                    pass
            if groundtruth_map and traj_data:
                try:
                    plt.figure(figsize=(8, 6))
                    for rid, xyz, keys_in_order in traj_data:
                        align = est_agg.get(rid, {}).get("ate", {})
                        R = np.asarray(align.get("R", np.eye(3)), dtype=float)
                        if R.shape != (3, 3):
                            R = np.eye(3)
                        t = np.asarray(align.get("t", np.zeros(3)), dtype=float)
                        if t.shape != (3,):
                            t = np.zeros(3)
                        try:
                            s = float(align.get("s", 1.0) or 1.0)
                        except Exception:
                            s = 1.0
                        aligned = (xyz @ R.T) * s + t
                        label = f"{rid} est (aligned)" if align else f"{rid} est"
                        plt.plot(aligned[:, 0], aligned[:, 1], label=label)
                        gt_entries = groundtruth_map.get(rid, {})
                        if gt_entries:
                            gxs = []
                            gys = []
                            for key in keys_in_order:
                                gt_entry = gt_entries.get(str(key))
                                if not gt_entry:
                                    continue
                                _rot, trans, _stamp = gt_entry
                                try:
                                    if hasattr(trans, "x"):
                                        gx, gy = float(trans.x), float(trans.y)
                                    else:
                                        gx, gy = float(trans[0]), float(trans[1])
                                    gxs.append(gx)
                                    gys.append(gy)
                                except Exception:
                                    continue
                            if gxs:
                                plt.plot(gxs, gys, linestyle="--", label=f"{rid} gt")
                    plt.axis("equal")
                    plt.xlabel("x [m]")
                    plt.ylabel("y [m]")
                    plt.legend()
                    plt.title("Trajectories (XY): est vs GT (aligned)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(metrics_dir, "traj_xy_gt.png"), dpi=150)
                    plt.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _merge_resource_profiles(metrics_dir, robot_ids, resource_metadata=resource_metadata)
        except Exception:
            pass

    if bad_exitcodes:
        raise RuntimeError(f"One or more agents crashed: {bad_exitcodes}")

    return pids, namespaced_prefix


def _merge_resource_profiles(
    metrics_dir: str,
    robot_ids: List[str],
    *,
    resource_metadata: Optional[Dict[str, object]] = None,
    out_name: str = "resource_profile.json",
) -> None:
    """
    Produce an aggregate resource profile by resampling and summing per-agent traces.

    Per-agent traces are exported from each agent process as:
      - resource_profile_<rid>.json
    The aggregate is written as:
      - resource_profile.json
    """

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        vals = sorted(values)
        out: Dict[str, float] = {
            "min": float(vals[0]),
            "max": float(vals[-1]),
            "mean": float(float(np.mean(vals))),
            "median": float(float(np.median(vals))),
        }
        if len(vals) > 1:
            out["stdev"] = float(float(np.std(vals)))
        return out

    def _percentile(values_sorted: List[float], pct: float) -> float:
        if not values_sorted:
            return 0.0
        if pct <= 0:
            return float(values_sorted[0])
        if pct >= 100:
            return float(values_sorted[-1])
        rank = (pct / 100.0) * (len(values_sorted) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(values_sorted) - 1)
        w = rank - lower
        return float(values_sorted[lower] * (1 - w) + values_sorted[upper] * w)

    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        vals = sorted(values)
        return {
            "p50": _percentile(vals, 50.0),
            "p90": _percentile(vals, 90.0),
            "p95": _percentile(vals, 95.0),
            "p99": _percentile(vals, 99.0),
        }

    traces: Dict[str, List[Dict[str, float]]] = {}
    for rid in robot_ids:
        p = os.path.join(metrics_dir, f"resource_profile_{rid}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            samples = doc.get("samples", [])
            if isinstance(samples, list) and samples:
                traces[str(rid)] = [s for s in samples if isinstance(s, dict) and "ts" in s]
        except Exception:
            continue
    if not traces:
        return

    deltas: List[float] = []
    for _rid, ss in traces.items():
        ts = [float(s.get("ts", 0.0) or 0.0) for s in ss]
        ts = [t for t in ts if t > 0.0]
        ts.sort()
        if len(ts) < 3:
            continue
        ds = [b - a for a, b in zip(ts[:-1], ts[1:]) if b > a]
        if ds:
            deltas.append(float(np.median(ds)))
    dt = float(np.median(deltas)) if deltas else 0.5
    dt = max(0.1, min(2.0, dt))

    all_ts: List[float] = []
    for ss in traces.values():
        for s in ss:
            try:
                all_ts.append(float(s.get("ts", 0.0) or 0.0))
            except Exception:
                continue
    all_ts = [t for t in all_ts if t > 0.0]
    if len(all_ts) < 2:
        return
    t0 = float(min(all_ts))
    t1 = float(max(all_ts))
    if not (t1 > t0):
        return
    n = int(np.floor((t1 - t0) / dt)) + 1
    grid = [t0 + i * dt for i in range(max(2, n))]

    def _interp_step(samples: List[Dict[str, float]], key: str, grid_ts: List[float]) -> List[float]:
        ss = sorted(samples, key=lambda s: float(s.get("ts", 0.0) or 0.0))
        out: List[float] = []
        j = 0
        last = 0.0
        for t in grid_ts:
            while j < len(ss) and float(ss[j].get("ts", 0.0) or 0.0) <= t:
                try:
                    last = float(ss[j].get(key, last) or last)
                except Exception:
                    pass
                j += 1
            out.append(float(last))
        return out

    cpu_by_agent: Dict[str, List[float]] = {}
    rss_by_agent: Dict[str, List[float]] = {}
    for rid, ss in traces.items():
        cpu_by_agent[rid] = _interp_step(ss, "cpu_process_pct", grid)
        rss_by_agent[rid] = _interp_step(ss, "rss_bytes", grid)

    samples_out: List[Dict[str, float]] = []
    cpu_vals: List[float] = []
    rss_vals: List[float] = []
    for i, t in enumerate(grid):
        cpu_total = float(sum(cpu_by_agent[rid][i] for rid in cpu_by_agent))
        rss_total = float(sum(rss_by_agent[rid][i] for rid in rss_by_agent))
        samples_out.append({"ts": float(t), "cpu_process_pct": cpu_total, "rss_bytes": rss_total})
        cpu_vals.append(cpu_total)
        rss_vals.append(rss_total)

    summary = {
        "num_samples": int(len(samples_out)),
        "cpu_process_pct": {**_stats(cpu_vals), **_percentiles(cpu_vals)} if cpu_vals else {},
        "rss_bytes": _stats(rss_vals) if rss_vals else {},
    }
    meta = dict(resource_metadata or {})
    meta.update({"role": "aggregate", "robots": list(robot_ids)})
    out = {"metadata": meta, "summary": summary, "samples": samples_out}
    with open(os.path.join(metrics_dir, out_name), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
