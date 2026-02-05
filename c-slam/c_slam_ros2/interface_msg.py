from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

from c_slam_common.models import Quaternion, Translation

logger = logging.getLogger("c_slam_ros2.interface_msg")


def _serialise_quaternion(q: Quaternion):
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _serialise_translation(t: Translation):
    return [float(t.x), float(t.y), float(t.z)]


def _serialise_covariance(covariance) -> Any:
    arr = np.asarray(covariance, dtype=float)
    if arr.shape == (6, 6):
        return arr.tolist()
    if arr.size == 36:
        return arr.reshape(6, 6).tolist()
    raise ValueError(f"Interface covariance must have 36 elements; got shape {arr.shape}")


def _deserialise_covariance(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.shape == (6, 6):
        return arr
    if arr.size == 36:
        return arr.reshape(6, 6)
    raise ValueError(f"Decoded covariance has unexpected shape {arr.shape}")


def _coerce_quaternion(values) -> Quaternion:
    if values is None:
        raise ValueError("Quaternion payload missing")
    vals = list(map(float, values))
    if len(vals) != 4:
        raise ValueError(f"Quaternion payload must have 4 elements, got {len(vals)}")
    return Quaternion(*vals)


def _coerce_translation(values) -> Translation:
    if values is None:
        raise ValueError("Translation payload missing")
    vals = list(map(float, values))
    if len(vals) != 3:
        raise ValueError(f"Translation payload must have 3 elements, got {len(vals)}")
    return Translation(*vals)


def encode_interface_message(msg, *, version: int = 1) -> bytes:
    payload: Dict[str, Any] = {
        "version": version,
        "message": {
            "sender": msg.sender,
            "receiver": msg.receiver,
            "key": str(msg.key),
            "stamp": float(getattr(msg, "stamp", 0.0)),
            "iteration": int(getattr(msg, "iteration", 0)),
            "rotation": _serialise_quaternion(msg.rotation),
            "translation": _serialise_translation(msg.translation),
            "covariance": _serialise_covariance(msg.covariance),
        },
    }
    sent_wall = getattr(msg, "sent_wall_time", None)
    sent_mono = getattr(msg, "sent_mono_time", None)
    if sent_mono is not None:
        payload["message"]["send_ts_mono"] = float(sent_mono)
    if sent_wall is not None:
        payload["message"]["send_ts_wall"] = float(sent_wall)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def decode_interface_message(data: Any):
    from c_slam_decentral.communication import InterfaceMessage

    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    doc = json.loads(data)
    version = doc.get("version", 1)
    if version != 1:
        logger.warning("Unknown interface message version %s; attempting fallback decode", version)
    payload = doc.get("message", {})
    rotation = _coerce_quaternion(payload.get("rotation"))
    translation = _coerce_translation(payload.get("translation"))
    covariance = _deserialise_covariance(payload.get("covariance"))
    sender = str(payload.get("sender"))
    receiver = str(payload.get("receiver"))
    key = payload.get("key")
    stamp = float(payload.get("stamp", 0.0))
    iteration = int(payload.get("iteration", 0))
    sent_wall_time = payload.get("send_ts_wall", payload.get("sent_wall_time"))
    sent_wall_time = float(sent_wall_time) if sent_wall_time is not None else None
    sent_mono_time = payload.get("send_ts_mono")
    sent_mono_time = float(sent_mono_time) if sent_mono_time is not None else None
    return InterfaceMessage(
        sender=sender,
        receiver=receiver,
        key=key,
        rotation=rotation,
        translation=translation,
        covariance=covariance,
        iteration=iteration,
        stamp=stamp,
        sent_wall_time=sent_wall_time,
        sent_mono_time=sent_mono_time,
    )
