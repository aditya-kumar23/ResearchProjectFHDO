from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Optional, Union

import numpy as np

from .models import BetweenFactorPose3, PriorFactorPose3


def _flatten_cov(cov: np.ndarray):
    return np.asarray(cov, dtype=float).reshape(-1).tolist()


def _prior_payload(f: PriorFactorPose3) -> Dict[str, Union[str, float, list]]:
    return {
        "key": str(f.key),
        "stamp": float(getattr(f, "stamp", 0.0)),
        "rotation": [f.rotation.w, f.rotation.x, f.rotation.y, f.rotation.z],
        "translation": [f.translation.x, f.translation.y, f.translation.z],
        "covariance": _flatten_cov(f.covariance),
    }


def _between_payload(f: BetweenFactorPose3) -> Dict[str, Union[str, float, list]]:
    return {
        "key1": str(f.key1),
        "key2": str(f.key2),
        "stamp": float(getattr(f, "stamp", 0.0)),
        "rotation": [f.rotation.w, f.rotation.x, f.rotation.y, f.rotation.z],
        "translation": [f.translation.x, f.translation.y, f.translation.z],
        "covariance": _flatten_cov(f.covariance),
    }


def factor_bytes(factor: Union[PriorFactorPose3, BetweenFactorPose3]) -> int:
    if hasattr(factor, "key") and hasattr(factor, "translation") and not hasattr(factor, "key2"):
        payload = _prior_payload(factor)  # type: ignore[arg-type]
    elif hasattr(factor, "key1") and hasattr(factor, "key2"):
        payload = _between_payload(factor)  # type: ignore[arg-type]
    else:
        raise TypeError(f"Unsupported factor type {type(factor)}")
    serialized = json.dumps(payload, separators=(",", ":"))
    return len(serialized.encode("utf-8"))


def map_payload_bytes(pose_count: int, per_pose_bytes: int = 7 * 8) -> int:
    return int(pose_count * per_pose_bytes)


class BandwidthTracker:
    def __init__(self):
        self._uplink = defaultdict(lambda: {"messages": 0, "bytes": 0})
        self._downlink = defaultdict(lambda: {"messages": 0, "bytes": 0})

    def add_uplink(
        self,
        topic: str,
        size_bytes: int,
        *,
        payload_bytes: Optional[int] = None,
        serialize_ns: Optional[int] = None,
    ) -> None:
        bucket = self._uplink[topic]
        bucket["messages"] += 1
        bucket["bytes"] += int(size_bytes)
        if payload_bytes is not None:
            bucket["payload_bytes"] = int(bucket.get("payload_bytes", 0)) + int(payload_bytes)
        if serialize_ns is not None:
            bucket["serialize_ns"] = int(bucket.get("serialize_ns", 0)) + int(serialize_ns)

    def add_downlink(
        self,
        topic: str,
        size_bytes: int,
        *,
        payload_bytes: Optional[int] = None,
        deserialize_ns: Optional[int] = None,
    ) -> None:
        bucket = self._downlink[topic]
        bucket["messages"] += 1
        bucket["bytes"] += int(size_bytes)
        if payload_bytes is not None:
            bucket["payload_bytes"] = int(bucket.get("payload_bytes", 0)) + int(payload_bytes)
        if deserialize_ns is not None:
            bucket["deserialize_ns"] = int(bucket.get("deserialize_ns", 0)) + int(deserialize_ns)

    @property
    def uplink(self) -> Dict[str, Dict[str, int]]:
        return dict(self._uplink)

    @property
    def downlink(self) -> Dict[str, Dict[str, int]]:
        return dict(self._downlink)

    def summary(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        return {"uplink": self.uplink, "downlink": self.downlink}

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

