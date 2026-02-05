from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import AbstractContextManager
from functools import partial
from typing import Dict, Iterable, Iterator, List, Optional, Set, Union

from c_slam_common.models import BetweenFactorPose3, PriorFactorPose3
from c_slam_ros2.factor_batch import decode_factor_batch
from c_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("c_slam_ros2.factor_source")

Factor = Union[PriorFactorPose3, BetweenFactorPose3]


class FactorSourceError(RuntimeError):
    pass


class ROS2FactorSource(AbstractContextManager):
    """Subscribe to per-robot ROS 2 topics and decode factor batches on the fly."""

    def __init__(
        self,
        *,
        topic_prefix: str,
        robot_ids: Iterable[str],
        qos_profile: Dict[str, object],
        queue_size: int = 0,
        spin_timeout: float = 0.1,
        idle_timeout: float = 0.0,
    ) -> None:
        self._topic_prefix = (topic_prefix or "/c_slam/factor_batch").rstrip("/") or "/c_slam/factor_batch"
        self._robot_ids = sorted({str(rid) for rid in robot_ids}) or ["global"]
        self._active_publishers: Set[str] = set(self._robot_ids)
        self._done_publishers: Set[str] = set()
        self._qos_profile = dict(qos_profile)
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size if queue_size > 0 else 0)
        self._spin_timeout = max(spin_timeout, 0.01)
        self._idle_timeout = max(idle_timeout, 0.0)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._message_count = 0
        self._should_shutdown = False
        self._last_message_time: Optional[float] = None

        self._node = None
        self._subscriptions: List[object] = []
        self._control_subscription = None
        self._rclpy = None

        self._control_topic = f"{self._topic_prefix}/control"

    def _topic_for_robot(self, rid: str) -> str:
        return f"{self._topic_prefix}/{rid}"

    def _ensure_rclpy(self):
        try:
            import rclpy  # type: ignore
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
            from std_msgs.msg import UInt8MultiArray, String  # type: ignore
        except Exception as exc:
            raise FactorSourceError("ROS 2 runtime unavailable") from exc

        qos_kwargs = dict(depth=int(self._qos_profile.get("depth", 10)))
        reliability = str(self._qos_profile.get("reliability", "reliable")).lower()
        durability = str(self._qos_profile.get("durability", "volatile")).lower()
        qos_kwargs["reliability"] = (
            ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        )
        qos_kwargs["durability"] = (
            DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE
        )
        data_qos = QoSProfile(**qos_kwargs)
        control_qos = QoSProfile(depth=1, reliability=data_qos.reliability, durability=data_qos.durability)
        return rclpy, UInt8MultiArray, String, data_qos, control_qos

    def _rclpy_is_initialized(self, rclpy_module) -> bool:
        try:
            return bool(rclpy_module.ok())
        except Exception:
            return False

    def _rclpy_init(self, rclpy_module) -> None:
        try:
            rclpy_module.init(args=None)
        except Exception as exc:
            raise FactorSourceError(f"Failed to initialise rclpy: {exc}") from exc

    def _spin_loop(self, rclpy_module):
        while not self._closed.is_set():
            rclpy_module.spin_once(self._node, timeout_sec=self._spin_timeout)
        rclpy_module.spin_once(self._node, timeout_sec=0.0)

    def _on_message(self, rid: str, msg) -> None:
        try:
            data = bytes(getattr(msg, "data", []) or [])
        except Exception:
            return
        self._message_count += 1
        self._last_message_time = time.time()
        factors = decode_factor_batch(data)
        for f in factors:
            try:
                setattr(f, "__ros2_topic", self._topic_for_robot(rid))
                setattr(f, "__ros2_msg_bytes", len(data))
            except Exception:
                pass
            self._queue.put(f)

    def _on_control(self, msg) -> None:
        try:
            data = str(getattr(msg, "data", "") or "")
        except Exception:
            return
        if data.startswith("DONE:"):
            rid = data.split(":", 1)[1]
            self._done_publishers.add(str(rid))
            self._active_publishers.discard(str(rid))
        elif data.strip() == "DONE_ALL":
            self._active_publishers.clear()

    def __enter__(self):
        rclpy, UInt8MultiArray, StringMsg, data_qos, control_qos = self._ensure_rclpy()
        self._rclpy = rclpy
        if not self._rclpy_is_initialized(rclpy):
            self._rclpy_init(rclpy)
            self._should_shutdown = True
        self._node = rclpy.create_node("c_slam_factor_listener")
        configure_sim_time(self._node)
        for rid in self._robot_ids:
            topic = self._topic_for_robot(rid)
            subscription = self._node.create_subscription(
                UInt8MultiArray,
                topic,
                partial(self._on_message, rid),
                data_qos,
            )
            self._subscriptions.append(subscription)
        self._control_subscription = self._node.create_subscription(StringMsg, self._control_topic, self._on_control, control_qos)
        self._closed.clear()
        self._thread = threading.Thread(target=self._spin_loop, args=(rclpy,), daemon=True)
        self._thread.start()
        return self

    def iter_factors(self) -> Iterator[Factor]:
        while not self._closed.is_set():
            try:
                factor = self._queue.get(timeout=self._spin_timeout)
                yield factor
                continue
            except queue.Empty:
                pass
            if self._idle_timeout > 0.0 and self._last_message_time is not None:
                if (time.time() - self._last_message_time) >= self._idle_timeout:
                    break
            if not self._active_publishers:
                break

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._spin_timeout * 5)
        if self._node is not None and self._rclpy is not None:
            try:
                for sub in self._subscriptions:
                    try:
                        self._node.destroy_subscription(sub)
                    except Exception:
                        pass
                if self._control_subscription is not None:
                    try:
                        self._node.destroy_subscription(self._control_subscription)
                    except Exception:
                        pass
                self._node.destroy_node()
            except Exception:
                pass
        if self._should_shutdown and self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

