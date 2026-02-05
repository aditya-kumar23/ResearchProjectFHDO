from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from collections import defaultdict, deque
import json
import logging
import os
import threading
import time

import numpy as np

from c_slam_common.models import Quaternion, Translation
from c_slam_ros2.impair import ImpairmentPolicy
from c_slam_ros2.interface_msg import decode_interface_message, encode_interface_message
from c_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("c_slam.decentral.communication")


@dataclass
class InterfaceMessage:
    sender: str
    receiver: str
    key: str
    rotation: Quaternion
    translation: Translation
    covariance: np.ndarray
    iteration: int
    stamp: float = 0.0
    sent_wall_time: float = None
    sent_mono_time: float = None

    def __post_init__(self):
        if self.sent_wall_time is None:
            self.sent_wall_time = time.time()
        if self.sent_mono_time is None:
            try:
                self.sent_mono_time = time.perf_counter()
            except Exception:
                self.sent_mono_time = None
        self.covariance = np.asarray(self.covariance, dtype=float)


class PeerBus:
    """In-process mailbox semantics shared by the ROS2-backed bus."""

    def __init__(self):
        self._mailboxes: Dict[str, deque] = defaultdict(deque)
        self._delivered: int = 0

    def post(self, message: InterfaceMessage) -> None:
        self._mailboxes[message.receiver].append(message)

    def drain(self, robot_id: str) -> List[InterfaceMessage]:
        mailbox = self._mailboxes.get(robot_id)
        if not mailbox:
            return []
        msgs = list(mailbox)
        self._delivered += len(msgs)
        mailbox.clear()
        return msgs

    @property
    def delivered(self) -> int:
        return self._delivered


class Ros2PeerBus(PeerBus):
    """ROS 2-backed peer bus: publish interface messages to `<prefix>/<rid>` topics."""

    def __init__(
        self,
        robot_ids: Iterable[str],
        *,
        local_robot_id: str,
        topic_prefix: str,
        qos_profile: Optional[Dict[str, object]] = None,
        spin_timeout: float = 0.05,
    ) -> None:
        super().__init__()
        import rclpy  # type: ignore
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
        from std_msgs.msg import UInt8MultiArray  # type: ignore

        self._topic_prefix = (topic_prefix or "/c_slam/iface").rstrip("/") or "/c_slam/iface"
        self._robot_ids = sorted({str(rid) for rid in robot_ids})
        self._local_robot_id = str(local_robot_id)
        self._qos_profile = dict(qos_profile or {})
        self._spin_timeout = max(0.01, float(spin_timeout))
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._should_shutdown = True
        else:
            self._should_shutdown = False

        reliability = str(self._qos_profile.get("reliability", "reliable")).lower()
        durability = str(self._qos_profile.get("durability", "volatile")).lower()
        qos = QoSProfile(
            depth=int(self._qos_profile.get("depth", 10)),
            reliability=ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE,
        )

        self._UInt8MultiArray = UInt8MultiArray
        self._node = rclpy.create_node("c_slam_ddf_peer_bus")
        configure_sim_time(self._node)
        self._publishers: Dict[str, object] = {}
        self._subscriptions: Dict[str, object] = {}
        self._impair = ImpairmentPolicy.from_env() or ImpairmentPolicy(None)
        self._delivery: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {"attempts": 0, "drops": 0, "delivered": 0, "bytes_attempted": 0, "bytes_delivered": 0, "drop_reasons": defaultdict(int)}
        )

        for rid in self._robot_ids:
            topic = f"{self._topic_prefix}/{rid}"
            self._publishers[rid] = self._node.create_publisher(UInt8MultiArray, topic, qos)

        local_topic = f"{self._topic_prefix}/{self._local_robot_id}"
        self._subscriptions[self._local_robot_id] = self._node.create_subscription(
            UInt8MultiArray,
            local_topic,
            lambda msg, _rid=self._local_robot_id: self._on_message(_rid, msg),
            qos,
        )

        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    def _spin_loop(self):
        while not self._closed.is_set():
            try:
                self._rclpy.spin_once(self._node, timeout_sec=self._spin_timeout)
            except Exception:
                time.sleep(self._spin_timeout)

    def _on_message(self, rid: str, msg) -> None:
        try:
            data = bytes(getattr(msg, "data", []) or [])
            iface = decode_interface_message(data)
        except Exception as exc:
            logger.debug("Failed to decode iface msg for %s: %s", rid, exc)
            return
        try:
            topic = f"{self._topic_prefix}/{rid}"
            stats = self._delivery[topic]
            stats["delivered"] = int(stats.get("delivered", 0)) + 1
            stats["bytes_delivered"] = int(stats.get("bytes_delivered", 0)) + int(len(data))
        except Exception:
            pass
        with self._lock:
            self._mailboxes[rid].append(iface)

    def post(self, message: InterfaceMessage) -> None:
        payload = encode_interface_message(message)
        topic = f"{self._topic_prefix}/{message.receiver}"
        try:
            stats = self._delivery[topic]
            stats["attempts"] = int(stats.get("attempts", 0)) + 1
            stats["bytes_attempted"] = int(stats.get("bytes_attempted", 0)) + int(len(payload))
        except Exception:
            pass
        if self._impair is not None:
            try:
                delay, reason = self._impair.on_send(
                    sender=str(message.sender),
                    receiver=str(message.receiver),
                    bytes_len=len(payload),
                    sent_wall_time=getattr(message, "sent_wall_time", None),
                    topic=topic,
                )
                if delay > 0.0:
                    time.sleep(delay)
                if reason is not None:
                    try:
                        stats = self._delivery[topic]
                        stats["drops"] = int(stats.get("drops", 0)) + 1
                        reasons = stats.get("drop_reasons")
                        if isinstance(reasons, dict):
                            reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
                    except Exception:
                        pass
                    return
            except Exception:
                pass
        msg = self._UInt8MultiArray()
        msg.data = list(payload)
        pub = self._publishers.get(str(message.receiver))
        if pub is None:
            raise RuntimeError(f"No publisher for receiver {message.receiver}")
        try:
            pub.publish(msg)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Failed to publish interface msg to %s: %s", topic, exc)
            return

    def drain(self, robot_id: str) -> List[InterfaceMessage]:
        with self._lock:
            mailbox = self._mailboxes.get(robot_id)
            if not mailbox:
                return []
            msgs = list(mailbox)
            self._delivered += len(msgs)
            mailbox.clear()
            return msgs

    def close(self) -> None:
        self._closed.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        try:
            for sub in self._subscriptions.values():
                try:
                    self._node.destroy_subscription(sub)
                except Exception:
                    pass
            for pub in self._publishers.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            self._node.destroy_node()
        except Exception:
            pass
        if self._should_shutdown:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

    def export_delivery_metrics(self, path: str) -> None:
        """Export per-topic attempts/drops/delivered counts for interface traffic.

        This is designed to be merged across processes by the runner.
        """
        try:
            topics_out: Dict[str, Dict[str, object]] = {}
            with self._lock:
                items = list(self._delivery.items())
            for topic, stats in items:
                attempts = int(stats.get("attempts", 0))
                drops = int(stats.get("drops", 0))
                delivered = int(stats.get("delivered", 0))
                topics_out[str(topic)] = {
                    "attempts": attempts,
                    "drops": drops,
                    "delivered": delivered,
                    "delivery_rate": (float(delivered) / float(attempts)) if attempts > 0 else None,
                    "bytes_attempted": int(stats.get("bytes_attempted", 0)),
                    "bytes_delivered": int(stats.get("bytes_delivered", 0)),
                    "drop_reasons": dict(stats.get("drop_reasons", {})),
                    "delivery_method": "delivered_vs_attempted",
                }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"stats": {"topics": topics_out, "generated_by": {"component": "ros2_peer_bus", "robot": self._local_robot_id}}}, f, indent=2)
        except Exception:
            return
