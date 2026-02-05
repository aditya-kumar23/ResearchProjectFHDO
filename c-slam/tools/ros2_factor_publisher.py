#!/usr/bin/env python3
"""Replay JRL factors over ROS 2 as per-robot factor-batches.

Publishes:
- data:  <topic-prefix>/<robot_id>   (std_msgs/UInt8MultiArray)
- ctrl:  <topic-prefix>/control     (std_msgs/String) with DONE markers

This script is intentionally minimal and tailored for the `c-slam/` standalone
repo (ROS2+iSAM2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt8MultiArray

from c_slam_common.graph import default_robot_infer
from c_slam_common.loader import LoaderConfig, build_key_robot_map, iter_measurements, load_jrl, summarize_schema
from c_slam_common.models import BetweenFactorPose3, PriorFactorPose3
from c_slam_ros2.factor_batch import encode_factor_batch_with_meta
from c_slam_ros2.impair import ImpairmentPolicy
from c_slam_ros2.qos import parse_qos_options
from c_slam_ros2.sim_time import configure_sim_time

Factor = PriorFactorPose3 | BetweenFactorPose3
logger = logging.getLogger("c_slam.tools.ros2_factor_publisher")


class FactorReplayNode(Node):
    def __init__(
        self,
        *,
        topic_prefix: str,
        robot_ids: Sequence[str],
        data_qos: QoSProfile,
        control_qos: QoSProfile,
    ) -> None:
        super().__init__("c_slam_factor_replay")
        configure_sim_time(self)
        self._topic_prefix = topic_prefix.rstrip("/") or "/c_slam/factor_batch"
        self._data_qos = data_qos
        self._control_qos = control_qos
        self._robot_publishers: Dict[str, object] = {}
        self._all_robot_ids: Set[str] = set()
        self._control_topic = f"{self._topic_prefix}/control"
        self._control_pub = self.create_publisher(String, self._control_topic, self._control_qos)
        self._ack_topic_prefix = f"{self._topic_prefix}/ack"
        self._acked_ids_by_rid: Dict[str, Set[str]] = defaultdict(set)
        self._sent_meta_by_id: Dict[str, Dict[str, object]] = {}
        self._topic_stats: Dict[str, Dict[str, object]] = defaultdict(lambda: {"attempts": 0, "drops": 0, "published": 0, "bytes_attempted": 0, "bytes_published": 0, "drop_reasons": defaultdict(int), "ack_latencies_s": []})
        for rid in sorted(set(map(str, robot_ids))):
            self._create_publisher(rid)
        self._impair = ImpairmentPolicy.from_env()

    def topic_for_robot(self, rid: str) -> str:
        return f"{self._topic_prefix}/{str(rid)}"

    def _create_publisher(self, rid: str):
        rid = str(rid)
        topic = self.topic_for_robot(rid)
        pub = self.create_publisher(UInt8MultiArray, topic, self._data_qos)
        self._robot_publishers[rid] = pub
        self._all_robot_ids.add(rid)
        # Also listen for optional ACKs from the solver (`<prefix>/ack/<rid>`).
        try:
            ack_topic = f"{self._ack_topic_prefix}/{rid}"
            self.create_subscription(UInt8MultiArray, ack_topic, lambda msg, _rid=rid: self._on_ack(_rid, msg), self._data_qos)
        except Exception:
            pass
        return pub

    @property
    def all_robot_ids(self) -> List[str]:
        return sorted(self._all_robot_ids)

    def subscriber_count(self, rid: str) -> int:
        topic = self.topic_for_robot(rid)
        try:
            return int(self.count_subscribers(topic))
        except Exception:
            return 0

    def publish_batch(self, rid: str, factors: List[Factor]) -> None:
        if not factors:
            return
        rid = str(rid)
        publisher = self._robot_publishers.get(rid) or self._create_publisher(rid)
        payload, meta = encode_factor_batch_with_meta(factors)
        msg_id = str(meta.get("message_id") or "")
        if self._impair is not None:
            delay, reason = self._impair.on_send(
                sender=rid,
                receiver="ros2",
                bytes_len=len(payload),
                sent_wall_time=time.time(),
                topic=self.topic_for_robot(rid),
            )
            if delay > 0.0:
                time.sleep(delay)
            if reason is not None:
                self._record_send(rid, msg_id=msg_id, bytes_len=len(payload), published=False, drop_reason=str(reason))
                return
        self._record_send(rid, msg_id=msg_id, bytes_len=len(payload), published=True, drop_reason=None, send_ts_mono=meta.get("send_ts_mono"), send_ts_wall=meta.get("send_ts_wall"))
        msg = UInt8MultiArray()
        msg.data = list(payload)
        publisher.publish(msg)  # type: ignore[attr-defined]

    def publish_done(self, rid: str) -> None:
        msg = String()
        msg.data = f"DONE:{rid}"
        self._control_pub.publish(msg)

    def publish_done_all(self) -> None:
        msg = String()
        msg.data = "DONE_ALL"
        self._control_pub.publish(msg)

    def _record_send(
        self,
        rid: str,
        *,
        msg_id: str,
        bytes_len: int,
        published: bool,
        drop_reason: str | None,
        send_ts_mono: float | None = None,
        send_ts_wall: float | None = None,
    ) -> None:
        topic = self.topic_for_robot(rid)
        stats = self._topic_stats[topic]
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        stats["bytes_attempted"] = int(stats.get("bytes_attempted", 0)) + int(bytes_len)
        if published:
            stats["published"] = int(stats.get("published", 0)) + 1
            stats["bytes_published"] = int(stats.get("bytes_published", 0)) + int(bytes_len)
        else:
            stats["drops"] = int(stats.get("drops", 0)) + 1
            reasons = stats.get("drop_reasons")
            if isinstance(reasons, dict) and drop_reason:
                reasons[str(drop_reason)] = int(reasons.get(str(drop_reason), 0)) + 1
        if msg_id:
            self._sent_meta_by_id[msg_id] = {
                "rid": str(rid),
                "topic": topic,
                "bytes": int(bytes_len),
                "send_ts_mono": float(send_ts_mono) if send_ts_mono is not None else None,
                "send_ts_wall": float(send_ts_wall) if send_ts_wall is not None else None,
            }

    def _on_ack(self, rid: str, msg: UInt8MultiArray) -> None:
        try:
            raw = bytes(getattr(msg, "data", []) or [])
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return
        msg_id = payload.get("message_id")
        if not msg_id:
            return
        msg_id = str(msg_id)
        meta = self._sent_meta_by_id.get(msg_id)
        if not meta:
            return
        topic = str(meta.get("topic") or self.topic_for_robot(rid))
        if msg_id in self._acked_ids_by_rid[str(rid)]:
            return
        self._acked_ids_by_rid[str(rid)].add(msg_id)
        stats = self._topic_stats[topic]
        lat_list = stats.get("ack_latencies_s")
        if isinstance(lat_list, list):
            try:
                send_ts = payload.get("send_ts_mono")
                use_ts = payload.get("use_ts_mono")
                if isinstance(send_ts, (int, float)) and isinstance(use_ts, (int, float)):
                    dt = float(use_ts) - float(send_ts)
                    if dt >= 0.0:
                        lat_list.append(dt)
            except Exception:
                pass

    def export_robustness_metrics(self, path: str) -> None:
        def _stats(values: List[float]) -> Dict[str, float | int]:
            vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
            if not vals:
                return {"count": 0}
            import statistics

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

            out: Dict[str, float | int] = {
                "count": len(vals),
                "min": vals[0],
                "max": vals[-1],
                "mean": float(statistics.mean(vals)),
                "median": float(statistics.median(vals)),
                "p90": float(pct(90.0)),
                "p95": float(pct(95.0)),
            }
            if len(vals) > 1:
                out["stdev"] = float(statistics.pstdev(vals))
            return out

        topics_out: Dict[str, Dict[str, object]] = {}
        for topic, stats in self._topic_stats.items():
            attempts = int(stats.get("attempts", 0))
            drops = int(stats.get("drops", 0))
            published = int(stats.get("published", 0))
            rid = str(topic.rsplit("/", 1)[-1])
            acked = len(self._acked_ids_by_rid.get(rid, set()))
            use_acks = acked > 0
            delivered = int(acked if use_acks else published)
            delivery_rate = (float(delivered) / float(attempts)) if attempts > 0 else None
            lat = stats.get("ack_latencies_s")
            topics_out[str(topic)] = {
                "attempts": attempts,
                "drops": drops,
                "published": published,
                "delivered": delivered,
                "delivery_rate": delivery_rate,
                "bytes_attempted": int(stats.get("bytes_attempted", 0)),
                "bytes_published": int(stats.get("bytes_published", 0)),
                "bytes_delivered": int(stats.get("bytes_published", 0)) if not use_acks else None,
                "drop_reasons": dict(stats.get("drop_reasons", {})),
                "delivery_method": "ack" if use_acks else "published_minus_sender_drops",
                "ack_latency_s": _stats(lat) if isinstance(lat, list) else {"count": 0},
            }

        out = {
            "stats": {
                "topics": topics_out,
                "generated_by": {"component": "ros2_factor_publisher", "pid": os.getpid()},
            }
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


def _wait_for_subscribers(node: FactorReplayNode, robot_ids: Sequence[str], timeout: float) -> None:
    if timeout <= 0.0:
        return
    remaining = set(map(str, robot_ids))
    start = time.time()
    while remaining:
        missing = {rid for rid in list(remaining) if node.subscriber_count(rid) == 0}
        if not missing:
            return
        if time.time() - start >= timeout:
            logger.warning("Timed out waiting for subscribers on: %s", sorted(missing))
            return
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.05)
        remaining = missing


def _resolve_robot(factor: Factor, robot_map: Dict[str, str]) -> str:
    if isinstance(factor, PriorFactorPose3):
        rid = robot_map.get(str(factor.key))
        key_hint = factor.key
    else:
        rid = robot_map.get(str(factor.key1)) or robot_map.get(str(factor.key2))
        key_hint = factor.key1
    if rid:
        return str(rid)
    try:
        return str(default_robot_infer(str(key_hint)))
    except Exception:
        return "global"


def publish_dataset(
    node: FactorReplayNode,
    factors: Iterable[Factor],
    *,
    robot_map: Dict[str, str],
    batch_size: int,
    time_scale: float,
    max_sleep: float,
) -> None:
    batch_size = max(1, int(batch_size))
    buffers: Dict[str, List[Factor]] = defaultdict(list)
    last_stamp: float | None = None

    for factor in factors:
        stamp = float(getattr(factor, "stamp", 0.0))
        if last_stamp is None:
            last_stamp = stamp
        else:
            if time_scale > 0.0:
                dt = (stamp - last_stamp) / time_scale
                if max_sleep > 0.0:
                    dt = min(dt, max_sleep)
                if dt > 0.0:
                    time.sleep(dt)
            last_stamp = stamp

        rid = _resolve_robot(factor, robot_map)
        buffers[rid].append(factor)
        if len(buffers[rid]) >= batch_size:
            node.publish_batch(rid, buffers[rid])
            buffers[rid] = []

    for rid, pending in buffers.items():
        if pending:
            node.publish_batch(rid, pending)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish JRL factors to ROS 2 (per-agent factor streaming).")
    ap.add_argument("--jrl", required=True, help="Path to the .jrl JSON dataset")
    ap.add_argument(
        "--topic-prefix",
        default="/c_slam/factor_batch",
        help="Topic prefix (per-robot topic = <prefix>/<robot>)",
    )
    ap.add_argument("--use-sim-time", action="store_true", help="Set C_SLAM_USE_SIM_TIME=1 for this process")
    ap.add_argument("--include-potential-outliers", action="store_true")
    ap.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--time-scale", type=float, default=0.0, help="0 = publish immediately")
    ap.add_argument("--max-sleep", type=float, default=0.0, help="Clamp per-factor sleep (seconds)")
    ap.add_argument("--idle-gap", type=float, default=1.0, help="Wait after finishing publish (seconds)")
    ap.add_argument("--loop", type=int, default=1, help="Repeat dataset N times (0 = infinite)")
    ap.add_argument("--wait-subscriber-timeout", type=float, default=10.0, help="Wait for subscribers before streaming (seconds)")
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--qos-reliability", choices=["reliable", "best_effort"], default="reliable")
    ap.add_argument("--qos-durability", choices=["volatile", "transient_local"], default="volatile")
    ap.add_argument("--qos-depth", type=int, default=10)
    ap.add_argument("--metrics-out", default=None, help="Write robustness metrics JSON (delivery rate) to this path")
    ap.add_argument("--ack-wait", type=float, default=2.0, help="Seconds to wait for ACKs after publishing DONE markers")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.use_sim_time:
        import os

        os.environ["C_SLAM_USE_SIM_TIME"] = "1"

    cfg = LoaderConfig(
        quaternion_order=args.quat_order,
        validate_schema=True,
        include_potential_outliers=bool(args.include_potential_outliers),
    )

    logger.info("Loading JRL: %s", args.jrl)
    doc = load_jrl(args.jrl, cfg)
    logger.info("Schema: %s", json.dumps(summarize_schema(doc), sort_keys=True))

    robot_map = build_key_robot_map(doc)
    robot_ids = sorted(set(robot_map.values())) or ["global"]
    logger.info("Robots detected: %s", robot_ids)

    qos_options = parse_qos_options(
        reliability=args.qos_reliability,
        durability=args.qos_durability,
        depth=args.qos_depth,
    )
    data_qos = QoSProfile(
        depth=int(qos_options["depth"]),
        reliability=ReliabilityPolicy.RELIABLE if qos_options["reliability"] == "reliable" else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if qos_options["durability"] == "transient_local" else DurabilityPolicy.VOLATILE,
    )
    control_qos = QoSProfile(depth=1, reliability=data_qos.reliability, durability=data_qos.durability)

    node: FactorReplayNode | None = None
    spin_stop = threading.Event()
    spin_thread: threading.Thread | None = None
    try:
        rclpy.init()
        node = FactorReplayNode(
            topic_prefix=args.topic_prefix,
            robot_ids=robot_ids,
            data_qos=data_qos,
            control_qos=control_qos,
        )
        _wait_for_subscribers(node, robot_ids, float(args.wait_subscriber_timeout))
        # Spin in the background so ACK subscriptions are processed while publishing.
        spin_stop.clear()

        def _spin():
            while not spin_stop.is_set():
                try:
                    rclpy.spin_once(node, timeout_sec=0.1)
                except Exception:
                    time.sleep(0.05)

        spin_thread = threading.Thread(target=_spin, name="c_slam_factor_publisher_spin", daemon=True)
        spin_thread.start()

        iteration = 0
        while True:
            iteration += 1
            logger.info("Publishing dataset iteration %d", iteration)
            publish_dataset(
                node,
                iter_measurements(doc, cfg),
                robot_map=robot_map,
                batch_size=int(args.batch_size),
                time_scale=float(args.time_scale),
                max_sleep=float(args.max_sleep),
            )
            if int(args.loop) != 0 and iteration >= int(args.loop):
                break
            if float(args.idle_gap) > 0.0:
                time.sleep(float(args.idle_gap))

        for rid in node.all_robot_ids:
            node.publish_done(rid)
        node.publish_done_all()
        if args.metrics_out:
            time.sleep(max(0.0, float(args.ack_wait)))
        if float(args.idle_gap) > 0.0:
            time.sleep(float(args.idle_gap))
        if args.metrics_out:
            try:
                node.export_robustness_metrics(str(args.metrics_out))
                logger.info("Wrote robustness metrics: %s", args.metrics_out)
            except Exception as exc:
                logger.warning("Failed to write robustness metrics: %s", exc)

    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
    finally:
        spin_stop.set()
        if spin_thread and spin_thread.is_alive():
            spin_thread.join(timeout=0.5)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
