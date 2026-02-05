from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from c_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("c_slam_ros2.ack")


class Ros2AckPublisher:
    def __init__(self, *, topic_prefix: str, qos_profile: Dict[str, object]):
        import rclpy  # type: ignore
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
        from std_msgs.msg import UInt8MultiArray  # type: ignore

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._should_shutdown = True
        else:
            self._should_shutdown = False

        self._UInt8MultiArray = UInt8MultiArray
        self._topic_prefix = (topic_prefix or "/c_slam/factor_batch").rstrip("/") or "/c_slam/factor_batch"
        self._ack_prefix = f"{self._topic_prefix}/ack"
        self._node = rclpy.create_node(f"c_slam_factor_ack_{os.getpid()}")
        configure_sim_time(self._node)
        self._publishers: Dict[str, object] = {}
        self._QoSProfile = QoSProfile
        self._qos_kwargs = {
            "depth": int(qos_profile.get("depth", 10)),
            "reliability": (
                ReliabilityPolicy.RELIABLE
                if str(qos_profile.get("reliability", "reliable")).lower() == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
            "durability": (
                DurabilityPolicy.TRANSIENT_LOCAL
                if str(qos_profile.get("durability", "volatile")).lower() == "transient_local"
                else DurabilityPolicy.VOLATILE
            ),
        }

    def _publisher_for(self, rid: str):
        pub = self._publishers.get(rid)
        if pub is not None:
            return pub
        qos = self._QoSProfile(**self._qos_kwargs)
        topic = f"{self._ack_prefix}/{rid}"
        pub = self._node.create_publisher(self._UInt8MultiArray, topic, qos)
        self._publishers[rid] = pub
        return pub

    def publish_ack(
        self,
        rid: str,
        *,
        message_id: Optional[str],
        send_ts_mono: Optional[float],
        send_ts_wall: Optional[float],
        use_ts_mono: float,
        use_ts_wall: float,
        bytes: Optional[int] = None,
    ) -> None:
        pub = self._publisher_for(str(rid))
        payload = {
            "message_id": message_id,
            "send_ts_mono": float(send_ts_mono) if send_ts_mono is not None else None,
            "send_ts_wall": float(send_ts_wall) if send_ts_wall is not None else None,
            "use_ts_mono": float(use_ts_mono),
            "use_ts_wall": float(use_ts_wall),
            "bytes": int(bytes) if bytes is not None else None,
        }
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        msg = self._UInt8MultiArray()
        msg.data = list(data)
        try:
            pub.publish(msg)
        except Exception as exc:
            logger.debug("ACK publish failed for %s: %s", rid, exc)

    def close(self) -> None:
        try:
            for _rid, pub in list(self._publishers.items()):
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            self._publishers.clear()
            self._node.destroy_node()
        except Exception:
            pass
        if self._should_shutdown:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
