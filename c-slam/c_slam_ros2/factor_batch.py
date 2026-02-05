from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Union

import numpy as np

from c_slam_common.models import BetweenFactorPose3, PriorFactorPose3, Quaternion, Translation

logger = logging.getLogger("c_slam_ros2.factor_batch")

FactorLike = Union[PriorFactorPose3, BetweenFactorPose3]


def _serialise_key(key: Union[str, int]) -> Dict[str, Any]:
    if isinstance(key, int):
        return {"type": "int", "value": key}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    return {"type": "repr", "value": repr(key)}


def _deserialise_key(payload: Dict[str, Any]) -> Union[str, int]:
    kind = payload.get("type")
    value = payload.get("value")
    if kind == "int":
        return int(value)
    if kind == "str":
        return str(value)
    if kind == "repr":
        return str(value)
    return value


def _serialise_quaternion(q: Quaternion) -> List[float]:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _serialise_translation(t: Translation) -> List[float]:
    return [float(t.x), float(t.y), float(t.z)]


def _serialise_covariance(cov) -> List[List[float]]:
    arr = np.asarray(cov, dtype=float)
    if arr.shape == (6, 6):
        return arr.tolist()
    if arr.size == 36:
        return arr.reshape(6, 6).tolist()
    raise ValueError(f"Covariance must have 36 elements; got shape {arr.shape}")


def _deserialise_covariance(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.shape == (6, 6):
        return arr
    if arr.size == 36:
        return arr.reshape(6, 6)
    raise ValueError(f"Decoded covariance has unexpected shape {arr.shape}")


def _serialise_factor(factor: FactorLike) -> Dict[str, Any]:
    if isinstance(factor, PriorFactorPose3):
        return {
            "type": "PriorFactorPose3",
            "key": _serialise_key(factor.key),
            "rotation": _serialise_quaternion(factor.rotation),
            "translation": _serialise_translation(factor.translation),
            "covariance": _serialise_covariance(factor.covariance),
            "stamp": float(getattr(factor, "stamp", 0.0)),
        }
    if isinstance(factor, BetweenFactorPose3):
        return {
            "type": "BetweenFactorPose3",
            "key1": _serialise_key(factor.key1),
            "key2": _serialise_key(factor.key2),
            "rotation": _serialise_quaternion(factor.rotation),
            "translation": _serialise_translation(factor.translation),
            "covariance": _serialise_covariance(factor.covariance),
            "stamp": float(getattr(factor, "stamp", 0.0)),
        }
    raise TypeError(f"Unsupported factor type {type(factor)!r}")


def _deserialise_factor(payload: Dict[str, Any]) -> FactorLike:
    kind = payload.get("type")
    if kind == "PriorFactorPose3":
        quaternion = Quaternion(*map(float, payload["rotation"]))
        t = Translation(*map(float, payload["translation"]))
        cov = _deserialise_covariance(payload.get("covariance"))
        key = _deserialise_key(payload.get("key"))
        stamp = float(payload.get("stamp", 0.0))
        return PriorFactorPose3(key=key, rotation=quaternion, translation=t, covariance=cov, stamp=stamp)
    if kind == "BetweenFactorPose3":
        quaternion = Quaternion(*map(float, payload["rotation"]))
        t = Translation(*map(float, payload["translation"]))
        cov = _deserialise_covariance(payload.get("covariance"))
        key1 = _deserialise_key(payload.get("key1"))
        key2 = _deserialise_key(payload.get("key2"))
        stamp = float(payload.get("stamp", 0.0))
        return BetweenFactorPose3(key1=key1, key2=key2, rotation=quaternion, translation=t, covariance=cov, stamp=stamp)
    raise ValueError(f"Unsupported factor payload type {kind!r}")


def encode_factor_batch(factors: Iterable[FactorLike], *, version: int = 1) -> bytes:
    payload, _meta = encode_factor_batch_with_meta(factors, version=version)
    return payload


def encode_factor_batch_with_meta(factors: Iterable[FactorLike], *, version: int = 1) -> tuple[bytes, Dict[str, Any]]:
    """Encode factors and return (payload_bytes, meta).

    The returned `meta` includes `message_id` and optional `send_ts_*` timestamps.
    """
    entries = [_serialise_factor(factor) for factor in factors]
    try:
        import time as _time

        send_ts_mono = float(_time.perf_counter())
        send_ts_wall = float(_time.time())
    except Exception:
        send_ts_mono = None
        send_ts_wall = None
    payload: Dict[str, Any] = {"version": version, "factors": entries}
    payload["message_id"] = uuid.uuid4().hex
    if send_ts_mono is not None:
        payload["send_ts_mono"] = send_ts_mono
    if send_ts_wall is not None:
        payload["send_ts_wall"] = send_ts_wall
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return data, {"message_id": payload.get("message_id"), "send_ts_mono": send_ts_mono, "send_ts_wall": send_ts_wall}


def decode_factor_batch(payload: Union[str, bytes, bytearray]) -> List[FactorLike]:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    doc = json.loads(payload)
    version = doc.get("version", 1)
    if version != 1:
        logger.warning("Unknown factor batch version %s; attempting fallback decode", version)
    factors = doc.get("factors", [])
    send_ts_mono = doc.get("send_ts_mono", None)
    send_ts_wall = doc.get("send_ts_wall", None)
    message_id = doc.get("message_id", None)
    out: List[FactorLike] = []
    for idx, item in enumerate(factors):
        try:
            f = _deserialise_factor(item)
            if send_ts_mono is not None:
                setattr(f, "send_ts_mono", float(send_ts_mono))
            if send_ts_wall is not None:
                setattr(f, "send_ts_wall", float(send_ts_wall))
            if message_id is not None:
                setattr(f, "__ros2_msg_id", str(message_id))
            out.append(f)
        except Exception as exc:
            logger.warning("Skipping factor[%d]: %s", idx, exc)
    return out
