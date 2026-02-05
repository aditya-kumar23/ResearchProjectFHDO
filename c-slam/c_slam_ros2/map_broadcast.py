from __future__ import annotations

import json
import logging
import time
from typing import Dict, Iterable, Optional, Tuple

try:
    import rclpy  # type: ignore
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
    from std_msgs.msg import UInt8MultiArray  # type: ignore
except Exception:  # pragma: no cover - ROS2 may not be available in tests
    rclpy = None  # type: ignore
    QoSProfile = ReliabilityPolicy = DurabilityPolicy = UInt8MultiArray = None  # type: ignore

import numpy as np

from c_slam_ros2.impair import ImpairmentPolicy
from c_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("c_slam.map_broadcast")


class Ros2MapBroadcaster:
    """Publish per-robot map updates over ROS 2 topics, applying impairments if configured."""

    def __init__(self, *, robot_ids: Iterable[str], topic_prefix: str, qos_profile: Dict[str, object]) -> None:
        if rclpy is None:
            raise RuntimeError("ROS 2 runtime unavailable")

        self._topic_prefix = (topic_prefix or "/c_slam/map").rstrip("/") or "/c_slam/map"
        self._robot_ids = sorted({str(r) for r in robot_ids})
        self._qos_profile = dict(qos_profile or {})
        self._impair = ImpairmentPolicy.from_env() or ImpairmentPolicy(None)
        self._delivery: Dict[str, Dict[str, object]] = {}

        reliability = str(self._qos_profile.get("reliability", "reliable")).lower()
        durability = str(self._qos_profile.get("durability", "volatile")).lower()
        qos = QoSProfile(
            depth=int(self._qos_profile.get("depth", 10)),
            reliability=ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE,
        )

        if not rclpy.ok():
            rclpy.init(args=None)
            self._should_shutdown = True
        else:
            self._should_shutdown = False

        self._node = rclpy.create_node("c_slam_map_broadcaster")
        configure_sim_time(self._node)
        self._publishers: Dict[str, object] = {}
        for rid in self._robot_ids:
            topic = self._topic_for_robot(rid)
            self._publishers[rid] = self._node.create_publisher(UInt8MultiArray, topic, qos)
            self._delivery[topic] = {"attempts": 0, "drops": 0, "published": 0, "bytes_attempted": 0, "bytes_published": 0, "drop_reasons": {}}

    def _topic_for_robot(self, rid: str) -> str:
        return f"{self._topic_prefix}/{rid}"

    def _pose_payload(self, estimate, key, key_label=None):
        p = estimate.atPose3(key)
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
        return {
            "key": key_label(key) if key_label else str(key),
            "x": tx,
            "y": ty,
            "z": tz,
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
        }

    def publish_estimate(self, estimate, by_robot_keys: Dict[str, Iterable], key_label=None) -> Tuple[int, int]:
        """Publish the full pose set per robot. Returns (attempted_bytes, published_bytes)."""
        attempted_bytes = 0
        published_bytes = 0
        for rid in self._robot_ids:
            keys = by_robot_keys.get(rid, set())
            poses = []
            for k in keys:
                try:
                    if not estimate.exists(k):
                        continue
                    poses.append(self._pose_payload(estimate, k, key_label))
                except Exception:
                    continue
            payload = json.dumps({"robot": rid, "poses": poses}, separators=(",", ":")).encode("utf-8")
            attempted_bytes += len(payload)

            topic = self._topic_for_robot(rid)
            stats = self._delivery.get(topic) or {"attempts": 0, "drops": 0, "published": 0, "bytes_attempted": 0, "bytes_published": 0, "drop_reasons": {}}
            stats["attempts"] = int(stats.get("attempts", 0)) + 1
            stats["bytes_attempted"] = int(stats.get("bytes_attempted", 0)) + len(payload)

            if self._impair is not None:
                try:
                    delay, reason = self._impair.on_send(
                        sender="central",
                        receiver=str(rid),
                        bytes_len=len(payload),
                        sent_wall_time=time.time(),
                        topic=topic,
                    )
                    if delay > 0.0:
                        time.sleep(delay)
                    if reason is not None:
                        stats["drops"] = int(stats.get("drops", 0)) + 1
                        dr = stats.get("drop_reasons", {})
                        dr[str(reason)] = int(dr.get(str(reason), 0)) + 1
                        stats["drop_reasons"] = dr
                        self._delivery[topic] = stats
                        continue
                except Exception as exc:
                    logger.debug("Map broadcast impairment failed: %s", exc)

            msg = UInt8MultiArray()
            msg.data = list(payload)
            pub = self._publishers.get(rid)
            if pub is None:
                self._delivery[topic] = stats
                continue
            try:
                pub.publish(msg)  # type: ignore[attr-defined]
                stats["published"] = int(stats.get("published", 0)) + 1
                stats["bytes_published"] = int(stats.get("bytes_published", 0)) + len(payload)
                published_bytes += len(payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to publish map for %s: %s", rid, exc)
            self._delivery[topic] = stats
        return attempted_bytes, published_bytes

    def export_delivery_metrics(self, path: str) -> None:
        out = {"stats": {"topics": {}, "generated_by": {"component": "ros2_map_broadcast"}}}
        for topic, stats in self._delivery.items():
            attempts = int(stats.get("attempts", 0))
            published = int(stats.get("published", 0))
            drops = int(stats.get("drops", 0))
            out["stats"]["topics"][topic] = {
                "attempts": attempts,
                "published": published,
                "drops": drops,
                "delivery_rate": (float(published) / float(attempts)) if attempts > 0 else None,
                "bytes_attempted": int(stats.get("bytes_attempted", 0)),
                "bytes_published": int(stats.get("bytes_published", 0)),
                "drop_reasons": dict(stats.get("drop_reasons", {})),
                "delivery_method": "published_minus_sender_drops",
            }
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    def close(self) -> None:
        try:
            for pub in self._publishers.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            self._publishers.clear()
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        if self._should_shutdown:
            try:
                rclpy.shutdown()
            except Exception:
                pass
