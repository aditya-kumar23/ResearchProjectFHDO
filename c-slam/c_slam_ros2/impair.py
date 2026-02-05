from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict


def _now_mono() -> float:
    try:
        return float(time.perf_counter())
    except Exception:
        return float(time.time())


def _read_spec_from_env() -> Optional[Dict[str, Any]]:
    spec_raw = os.environ.get("C_SLAM_IMPAIR")
    path = os.environ.get("C_SLAM_IMPAIR_FILE")
    if spec_raw and spec_raw.strip():
        if os.path.exists(spec_raw):
            path = spec_raw
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    if spec_raw and spec_raw.strip():
        try:
            return json.loads(spec_raw)
        except Exception:
            return None
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


@dataclass
class _Bucket:
    rate_bps: float
    capacity_bytes: float
    tokens: float = 0.0
    last_mono: float = field(default_factory=_now_mono)

    def offer(self, bytes_len: int, now: Optional[float] = None) -> float:
        if now is None:
            now = _now_mono()
        dt = max(0.0, now - self.last_mono)
        self.tokens = min(self.capacity_bytes, self.tokens + dt * self.rate_bps)
        self.last_mono = now
        deficit = float(bytes_len) - self.tokens
        if deficit <= 0.0:
            self.tokens -= float(bytes_len)
            return 0.0
        wait = deficit / max(1.0, self.rate_bps)
        self.tokens = 0.0
        self.last_mono = now + wait
        return wait


class ImpairmentPolicy:
    def __init__(self, spec: Optional[Dict[str, Any]]) -> None:
        self.spec = dict(spec or {})
        self.anchor_mono = _now_mono()
        self.seed = int(self.spec.get("seed", 42))
        self.rng = random.Random(self.seed)
        self.p_loss = float(self.spec.get("random_loss_p", 0.0) or 0.0)
        self.random_warmup_messages = int(self.spec.get("random_warmup_messages", 0) or 0)
        self._sent_by_sender: Dict[str, int] = defaultdict(int)
        self.burst_period = _safe_float(self.spec.get("burst_period_s"), 0.0)
        self.burst_duration = _safe_float(self.spec.get("burst_duration_s"), 0.0)
        self.warmup_s = _safe_float(self.spec.get("warmup_s"), 0.0)
        self.blackouts = []
        for b in self.spec.get("blackouts", []) or []:
            try:
                self.blackouts.append(
                    {
                        "rid": str(b.get("rid")),
                        "start_s": _safe_float(b.get("start_s"), 0.0),
                        "end_s": _safe_float(b.get("end_s"), 0.0),
                        "mode": str(b.get("mode", "sender")),
                    }
                )
            except Exception:
                continue
        caps = self.spec.get("bw_caps_mbps", {}) or {}
        self.default_mbps = _safe_float(caps.get("default"), 0.0)
        self.caps_mbps: Dict[str, float] = {k: _safe_float(v) for k, v in caps.items() if k != "default"}
        self._buckets: Dict[str, _Bucket] = {}

    @classmethod
    def from_env(cls) -> Optional["ImpairmentPolicy"]:
        spec = _read_spec_from_env()
        if spec is None:
            return None
        return cls(spec)

    def _is_in_burst(self, now_mono: Optional[float] = None) -> bool:
        if self.burst_period <= 0.0 or self.burst_duration <= 0.0:
            return False
        if now_mono is None:
            now_mono = _now_mono()
        t = now_mono - self.anchor_mono
        return (t % self.burst_period) < self.burst_duration

    def _is_blackout(self, sender: str, receiver: str, now_mono: Optional[float] = None) -> bool:
        if not self.blackouts:
            return False
        if now_mono is None:
            now_mono = _now_mono()
        t = now_mono - self.anchor_mono
        for b in self.blackouts:
            mode = b.get("mode", "sender")
            rid = b.get("rid")
            if mode == "sender":
                match = sender == rid
            elif mode == "receiver":
                match = receiver == rid
            elif mode == "either":
                match = sender == rid or receiver == rid
            else:
                match = sender == rid
            if match and (t >= b.get("start_s", 0.0)) and (t <= b.get("end_s", 0.0)):
                return True
        return False

    def _bucket_for(self, sender: str) -> Optional[_Bucket]:
        mbps = self.caps_mbps.get(sender, self.default_mbps)
        if mbps <= 0.0:
            return None
        b = self._buckets.get(sender)
        if b is None:
            rate_bps = float(mbps) * 1e6 / 8.0
            capacity = max(1024.0, 0.1 * rate_bps)
            b = _Bucket(rate_bps=rate_bps, capacity_bytes=capacity)
            self._buckets[sender] = b
        return b

    def on_send(self, *, sender: str, receiver: str, bytes_len: int, sent_wall_time: Optional[float] = None, topic: Optional[str] = None) -> tuple[float, Optional[str]]:
        now = _now_mono()
        in_warmup = self.warmup_s > 0.0 and (now - self.anchor_mono) < self.warmup_s
        msg_count = self._sent_by_sender[str(sender)]
        self._sent_by_sender[str(sender)] = msg_count + 1
        if self._is_blackout(sender, receiver, now):
            return 0.0, "blackout"
        if (not in_warmup) and self._is_in_burst(now):
            return 0.0, "burst"
        if (not in_warmup) and msg_count >= self.random_warmup_messages and self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            return 0.0, "random"
        delay = 0.0
        bucket = self._bucket_for(sender)
        if bucket is not None:
            delay = float(bucket.offer(int(bytes_len), now))
        _ = sent_wall_time
        _ = topic
        return delay, None
