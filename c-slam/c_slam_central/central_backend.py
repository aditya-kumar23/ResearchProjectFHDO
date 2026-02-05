from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, Optional, Tuple

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

from c_slam_common.graph import GraphBuilder
from c_slam_common.isam2 import ISAM2Manager, update_translation_cache
from c_slam_common.kpi_logging import KPILogger
from c_slam_common.latency import LatencyTracker
from c_slam_common.bandwidth import BandwidthTracker, factor_bytes, map_payload_bytes
from c_slam_common.models import BetweenFactorPose3, InitEntry, PriorFactorPose3
from c_slam_common.stabilisation import DEFAULT_STABLE_EPSILON, DEFAULT_STABLE_REQUIRED
from c_slam_ros2.ack import Ros2AckPublisher
from c_slam_ros2.factor_source import ROS2FactorSource
from c_slam_ros2.map_broadcast import Ros2MapBroadcaster

Factor = PriorFactorPose3 | BetweenFactorPose3

logger = logging.getLogger("c_slam.central")


def _is_indeterminant_system_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return ("Indeterminant linear system" in msg) or ("IndeterminantLinearSystemException" in msg)


def _is_duplicate_key_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return ("key already exists" in msg) or ("Attempting to add a key-value pair with key" in msg)


def _publish_map_downlink(
    map_broadcaster: Optional[Ros2MapBroadcaster],
    estimate: Optional["gtsam.Values"],
    gb: Optional[GraphBuilder],
) -> Tuple[Optional[int], Optional[int]]:
    """Publish map to per-robot topics; returns (attempted_bytes, published_bytes)."""
    if map_broadcaster is None or estimate is None or gb is None:
        return None, None
    try:
        return map_broadcaster.publish_estimate(estimate, gb.by_robot_keys, key_label=getattr(gb, "denormalize_key", None))
    except Exception:
        return None, None


def run_centralised_isam2_ros2(
    *,
    factors: Iterable[Factor],
    init_lookup: Dict[str, InitEntry],
    robot_map: Dict[str, str],
    batch_size: int = 100,
    relin_threshold: float = 0.1,
    relin_skip: int = 10,
    robust_kind: Optional[str] = None,
    robust_k: Optional[float] = None,
    kpi: Optional[KPILogger] = None,
    bandwidth: Optional[BandwidthTracker] = None,
    latency: Optional[LatencyTracker] = None,
    ack_publisher: Optional[Ros2AckPublisher] = None,
    map_broadcaster: Optional[Ros2MapBroadcaster] = None,
    post_input_settle_updates: int = 0,
) -> Tuple["gtsam.Values", GraphBuilder]:
    if gtsam is None:
        raise RuntimeError("gtsam bindings are required")

    gb = GraphBuilder(robot_map=robot_map, robust_kind=robust_kind, robust_k=robust_k)
    isam = ISAM2Manager(relinearize_threshold=relin_threshold, relinearize_skip=relin_skip)
    batch_size = max(1, int(batch_size))

    added = 0
    batch_idx = 0
    batch_event_ids = []
    seen_ros_msgs = set()
    acked_ros_msgs = set()
    translation_cache: Dict[int, tuple] = {}
    needs_full_rebuild = False
    last_factor_stamp: Optional[float] = None

    for f in factors:
        try:
            last_factor_stamp = float(getattr(f, "stamp", None)) if getattr(f, "stamp", None) is not None else last_factor_stamp
        except Exception:
            pass
        topic = None
        size = None
        event_id = None
        if isinstance(f, PriorFactorPose3):
            gb.add_prior(f, init_lookup)
            ros_msg_id = getattr(f, "__ros2_msg_id", None)
            ros_topic = getattr(f, "__ros2_topic", None)
            ros_bytes = getattr(f, "__ros2_msg_bytes", None)
            counted_ros_msg = False
            if bandwidth:
                if ros_msg_id and ros_topic and isinstance(ros_bytes, int):
                    if ros_msg_id not in seen_ros_msgs:
                        seen_ros_msgs.add(ros_msg_id)
                        topic = str(ros_topic)
                        size = int(ros_bytes)
                        bandwidth.add_uplink(topic, size)
                        counted_ros_msg = True
                else:
                    topic = f"prior/{robot_map.get(str(f.key), 'global')}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
                    counted_ros_msg = True
            if topic is None and ros_topic is not None:
                topic = str(ros_topic)
            if latency:
                event_id = latency.record_ingest(
                    "PriorFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    robot_map.get(str(f.key)),
                    ingest_wall=time.time(),
                    ingest_mono=time.perf_counter(),
                    metadata={
                        "key": str(f.key),
                        "send_ts_mono": getattr(f, "send_ts_mono", None),
                        "send_ts_wall": getattr(f, "send_ts_wall", None),
                        "ros2_msg_id": str(ros_msg_id) if ros_msg_id is not None else None,
                        "ros2_topic": str(ros_topic) if ros_topic is not None else None,
                        "ros2_msg_bytes": int(ros_bytes) if isinstance(ros_bytes, int) else None,
                    },
                )
                batch_event_ids.append(event_id)
            if kpi:
                kpi.sensor_ingest(
                    "PriorFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    key=str(f.key),
                    topic=topic,
                    bytes=size if counted_ros_msg else None,
                    ros2_msg_id=str(ros_msg_id) if ros_msg_id is not None else None,
                    ros2_topic=str(ros_topic) if ros_topic is not None else None,
                    ros2_msg_bytes=int(ros_bytes) if isinstance(ros_bytes, int) else None,
                )
        elif isinstance(f, BetweenFactorPose3):
            gb.add_between(f, init_lookup)
            ros_msg_id = getattr(f, "__ros2_msg_id", None)
            ros_topic = getattr(f, "__ros2_topic", None)
            ros_bytes = getattr(f, "__ros2_msg_bytes", None)
            counted_ros_msg = False
            if bandwidth:
                if ros_msg_id and ros_topic and isinstance(ros_bytes, int):
                    if ros_msg_id not in seen_ros_msgs:
                        seen_ros_msgs.add(ros_msg_id)
                        topic = str(ros_topic)
                        size = int(ros_bytes)
                        bandwidth.add_uplink(topic, size)
                        counted_ros_msg = True
                else:
                    rid1 = robot_map.get(str(f.key1), "global")
                    rid2 = robot_map.get(str(f.key2), "global")
                    topic = f"between/{rid1}" if rid1 == rid2 else f"between/{rid1}-{rid2}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
                    counted_ros_msg = True
            if topic is None and ros_topic is not None:
                topic = str(ros_topic)
            if latency:
                rid1 = robot_map.get(str(f.key1), None)
                event_id = latency.record_ingest(
                    "BetweenFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    rid1,
                    ingest_wall=time.time(),
                    ingest_mono=time.perf_counter(),
                    metadata={
                        "key1": str(f.key1),
                        "key2": str(f.key2),
                        "send_ts_mono": getattr(f, "send_ts_mono", None),
                        "send_ts_wall": getattr(f, "send_ts_wall", None),
                        "ros2_msg_id": str(ros_msg_id) if ros_msg_id is not None else None,
                        "ros2_topic": str(ros_topic) if ros_topic is not None else None,
                        "ros2_msg_bytes": int(ros_bytes) if isinstance(ros_bytes, int) else None,
                    },
                )
                batch_event_ids.append(event_id)
            if kpi:
                kpi.sensor_ingest(
                    "BetweenFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    key1=str(f.key1),
                    key2=str(f.key2),
                    topic=topic,
                    bytes=size if counted_ros_msg else None,
                    ros2_msg_id=str(ros_msg_id) if ros_msg_id is not None else None,
                    ros2_topic=str(ros_topic) if ros_topic is not None else None,
                    ros2_msg_bytes=int(ros_bytes) if isinstance(ros_bytes, int) else None,
                )
        else:
            continue

        added += 1
        if added % batch_size != 0:
            continue

        bg, bv = gb.pop_batch()
        if bg.size() == 0 and bv.size() == 0:
            continue
        attempted_batch_id = batch_idx + 1
        if latency:
            latency.assign_batch(attempted_batch_id, batch_event_ids)
        if kpi:
            kpi.optimization_start(attempted_batch_id, bg.size(), added)
        start_wall = time.time()
        start_mono = time.perf_counter()
        if latency:
            latency.mark_use(attempted_batch_id, start_wall, start_mono)
        try:
            if needs_full_rebuild:
                raise RuntimeError("full_rebuild_requested")
            isam.update(bg, bv)
        except RuntimeError as exc:
            if _is_indeterminant_system_error(exc) or _is_duplicate_key_error(exc) or str(exc) == "full_rebuild_requested":
                needs_full_rebuild = True
                if kpi:
                    kpi.emit("optimization_failed", batch_id=attempted_batch_id, error=str(exc))
                try:
                    # Rebuild iSAM2 from scratch using the accumulated global graph.
                    new_isam = ISAM2Manager(relinearize_threshold=relin_threshold, relinearize_skip=relin_skip)
                    rebuild_start = time.perf_counter()
                    new_isam.update(gb.graph, gb.initial)
                    rebuild_dt = time.perf_counter() - rebuild_start
                    isam = new_isam
                    translation_cache = {}
                    needs_full_rebuild = False
                    if kpi:
                        kpi.emit("full_rebuild_ok", batch_id=attempted_batch_id, duration_s=rebuild_dt)
                except Exception as rebuild_exc:
                    if kpi:
                        kpi.emit("full_rebuild_failed", batch_id=attempted_batch_id, error=str(rebuild_exc))
                    continue
            else:
                raise
        batch_idx = attempted_batch_id
        events_for_batch = list(batch_event_ids)
        batch_event_ids = []
        opt_end_mono = time.perf_counter()
        opt_end_wall = time.time()
        duration = opt_end_mono - start_mono
        estimate = isam.estimate
        max_delta = update_translation_cache(translation_cache, estimate)
        updated = estimate.size() if hasattr(estimate, "size") else None
        if kpi:
            kpi.optimization_end(batch_idx, duration, updated_keys=updated, max_translation_delta=max_delta)
        if ack_publisher is not None and latency is not None:
            try:
                ids = set()
                for ev_id in events_for_batch:
                    ev = latency.events[ev_id]
                    msg_id = ev.get("ros2_msg_id")
                    rid = ev.get("robot")
                    if not msg_id or not rid or msg_id in ids or msg_id in acked_ros_msgs:
                        continue
                    ids.add(msg_id)
                    acked_ros_msgs.add(msg_id)
                    ack_publisher.publish_ack(
                        str(rid),
                        message_id=str(msg_id),
                        send_ts_mono=ev.get("send_ts_mono"),
                        send_ts_wall=ev.get("send_ts_wall"),
                        use_ts_mono=start_mono,
                        use_ts_wall=start_wall,
                        bytes=ev.get("ros2_msg_bytes"),
                    )
            except Exception:
                pass
        attempted_bytes = None
        published_bytes = None
        if map_broadcaster is not None:
            attempted_bytes, published_bytes = _publish_map_downlink(map_broadcaster, estimate, gb)

        payload_bytes = None
        if updated is not None:
            payload_bytes = published_bytes if published_bytes is not None else map_payload_bytes(updated)

        if bandwidth and payload_bytes is not None:
            bandwidth.add_downlink("map_broadcast", payload_bytes)
        broadcast_wall = time.time()
        broadcast_mono = time.perf_counter()
        if latency:
            latency.complete_batch(batch_idx, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, payload_bytes)
        if kpi:
            kpi.map_broadcast(
                batch_idx,
                pose_count=updated,
                bytes=payload_bytes,
                attempted_bytes=attempted_bytes,
            )

    bg, bv = gb.pop_batch()
    if bg.size() > 0 or bv.size() > 0:
        attempted_batch_id = batch_idx + 1
        if latency:
            latency.assign_batch(attempted_batch_id, batch_event_ids)
        if kpi:
            kpi.optimization_start(attempted_batch_id, bg.size(), added)
        start_wall = time.time()
        start_mono = time.perf_counter()
        if latency:
            latency.mark_use(attempted_batch_id, start_wall, start_mono)
        try:
            if needs_full_rebuild:
                raise RuntimeError("full_rebuild_requested")
            isam.update(bg, bv)
        except RuntimeError as exc:
            if _is_indeterminant_system_error(exc) or _is_duplicate_key_error(exc) or str(exc) == "full_rebuild_requested":
                if kpi:
                    kpi.emit("optimization_failed", batch_id=attempted_batch_id, error=str(exc))
                try:
                    new_isam = ISAM2Manager(relinearize_threshold=relin_threshold, relinearize_skip=relin_skip)
                    rebuild_start = time.perf_counter()
                    new_isam.update(gb.graph, gb.initial)
                    rebuild_dt = time.perf_counter() - rebuild_start
                    isam = new_isam
                    translation_cache = {}
                    if kpi:
                        kpi.emit("full_rebuild_ok", batch_id=attempted_batch_id, duration_s=rebuild_dt)
                except Exception as rebuild_exc:
                    if kpi:
                        kpi.emit("full_rebuild_failed", batch_id=attempted_batch_id, error=str(rebuild_exc))
                    # Return the last successful estimate, if any.
                    return isam.estimate, gb
            else:
                raise
        batch_idx = attempted_batch_id
        events_for_batch = list(batch_event_ids)
        batch_event_ids = []
        opt_end_mono = time.perf_counter()
        opt_end_wall = time.time()
        duration = opt_end_mono - start_mono
        estimate = isam.estimate
        max_delta = update_translation_cache(translation_cache, estimate)
        updated = estimate.size() if hasattr(estimate, "size") else None
        if kpi:
            kpi.optimization_end(batch_idx, duration, updated_keys=updated, max_translation_delta=max_delta)
        if ack_publisher is not None and latency is not None:
            try:
                ids = set()
                for ev_id in events_for_batch:
                    ev = latency.events[ev_id]
                    msg_id = ev.get("ros2_msg_id")
                    rid = ev.get("robot")
                    if not msg_id or not rid or msg_id in ids or msg_id in acked_ros_msgs:
                        continue
                    ids.add(msg_id)
                    acked_ros_msgs.add(msg_id)
                    ack_publisher.publish_ack(
                        str(rid),
                        message_id=str(msg_id),
                        send_ts_mono=ev.get("send_ts_mono"),
                        send_ts_wall=ev.get("send_ts_wall"),
                        use_ts_mono=start_mono,
                        use_ts_wall=start_wall,
                        bytes=ev.get("ros2_msg_bytes"),
                    )
            except Exception:
                pass
        attempted_bytes = None
        published_bytes = None
        if map_broadcaster is not None:
            attempted_bytes, published_bytes = _publish_map_downlink(map_broadcaster, estimate, gb)

        payload_bytes = None
        if updated is not None:
            payload_bytes = published_bytes if published_bytes is not None else map_payload_bytes(updated)

        if bandwidth and payload_bytes is not None:
            bandwidth.add_downlink("map_broadcast", payload_bytes)
        broadcast_wall = time.time()
        broadcast_mono = time.perf_counter()
        if latency:
            latency.complete_batch(batch_idx, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, payload_bytes)
        if kpi:
            kpi.map_broadcast(
                batch_idx,
                pose_count=updated,
                bytes=payload_bytes,
                attempted_bytes=attempted_bytes,
            )

    if kpi:
        kpi.emit(
            "input_end",
            robot="team",
            ingested_factors=int(added),
            last_factor_stamp=last_factor_stamp,
            batch_size=int(batch_size),
        )

    # Post-input settle: emit extra solver updates (no new factors) so stabilisation windows can be measured
    # relative to end-of-input without contaminating other KPIs.
    #
    # Semantics:
    #  - post_input_settle_updates == 0: disabled
    #  - post_input_settle_updates > 0: up to N settle updates (stop early once stable window is observed)
    #  - post_input_settle_updates < 0: settle "until stable" with a safety cap
    #
    # These updates are marked and filtered out in KPI derivation where appropriate.
    if kpi and int(post_input_settle_updates) != 0:
        requested = int(post_input_settle_updates)
        max_updates = requested if requested > 0 else 200
        epsilon = float(DEFAULT_STABLE_EPSILON)
        required = int(DEFAULT_STABLE_REQUIRED)
        stable_count = 0
        stable_observed = False
        empty_graph = gtsam.NonlinearFactorGraph()
        empty_vals = gtsam.Values()
        for i in range(int(max_updates)):
            start_mono = time.perf_counter()
            try:
                isam.update(empty_graph, empty_vals)
            except Exception:
                break
            duration = time.perf_counter() - start_mono
            estimate = isam.estimate
            max_delta = update_translation_cache(translation_cache, estimate)
            if max_delta is not None and float(max_delta) <= epsilon:
                stable_count += 1
                if stable_count >= required:
                    stable_observed = True
            else:
                stable_count = 0
            updated = estimate.size() if hasattr(estimate, "size") else None
            kpi.optimization_end(
                batch_idx + i + 1,
                duration,
                updated_keys=updated,
                max_translation_delta=max_delta,
                settle_only=True,
                settle_index=int(i + 1),
            )
            if stable_observed:
                break
        try:
            kpi.emit(
                "post_input_settle_done",
                robot="team",
                settle_requested=int(requested),
                settle_max_updates=int(max_updates),
                settle_updates_run=int(i + 1),
                stable_observed=bool(stable_observed),
                stable_epsilon=float(epsilon),
                stable_required=int(required),
            )
        except Exception:
            pass

    return isam.estimate, gb


def run_centralised_from_ros2(
    *,
    topic_prefix: str,
    robot_ids: Iterable[str],
    qos_profile: Dict[str, object],
    init_lookup: Dict[str, InitEntry],
    robot_map: Dict[str, str],
    batch_size: int,
    relin_threshold: float,
    relin_skip: int,
    robust_kind: Optional[str],
    robust_k: Optional[float],
    idle_timeout: float,
    kpi: Optional[KPILogger] = None,
    bandwidth: Optional[BandwidthTracker] = None,
    latency: Optional[LatencyTracker] = None,
    ack_publisher: Optional[Ros2AckPublisher] = None,
    map_broadcaster: Optional[Ros2MapBroadcaster] = None,
    post_input_settle_updates: int = 0,
) -> Tuple["gtsam.Values", GraphBuilder]:
    with ROS2FactorSource(
        topic_prefix=topic_prefix,
        robot_ids=list(robot_ids),
        qos_profile=qos_profile,
        queue_size=0,
        spin_timeout=0.1,
        idle_timeout=idle_timeout,
    ) as src:
        return run_centralised_isam2_ros2(
            factors=src.iter_factors(),
            init_lookup=init_lookup,
            robot_map=robot_map,
            batch_size=batch_size,
            relin_threshold=relin_threshold,
            relin_skip=relin_skip,
            robust_kind=robust_kind,
            robust_k=robust_k,
            kpi=kpi,
            bandwidth=bandwidth,
            latency=latency,
            ack_publisher=ack_publisher,
            map_broadcaster=map_broadcaster,
            post_input_settle_updates=post_input_settle_updates,
        )
