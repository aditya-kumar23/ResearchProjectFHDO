from __future__ import annotations

import json
import statistics
from typing import Dict, Iterable, List, Optional


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    rank = (pct / 100.0) * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = sorted(values)
    if not vals:
        return {"count": 0}
    out: Dict[str, Optional[float]] = {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p90": _percentile(vals, 90.0),
        "p95": _percentile(vals, 95.0),
        "p99": _percentile(vals, 99.0),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.pstdev(vals)
    return out


class LatencyTracker:
    def __init__(self):
        self._events: List[Dict] = []
        self._batches: Dict[int, List[int]] = {}

    def record_ingest(
        self,
        factor_type: str,
        dataset_stamp: float,
        robot: Optional[str],
        ingest_wall: float,
        ingest_mono: float,
        metadata: Optional[Dict] = None,
    ) -> int:
        event = {
            "factor_type": factor_type,
            "dataset_stamp": dataset_stamp,
            "robot": robot,
            "ingest_wall": ingest_wall,
            "ingest_mono": ingest_mono,
        }
        if metadata:
            event.update(metadata)
        event_id = len(self._events)
        self._events.append(event)
        return event_id

    def assign_batch(self, batch_id: int, event_ids: Iterable[int]) -> None:
        ids = [idx for idx in event_ids if idx is not None]
        if not ids:
            return
        self._batches[batch_id] = ids
        for idx in ids:
            self._events[idx]["batch_id"] = batch_id

    def mark_use(self, batch_id: int, use_wall: float, use_mono: float) -> None:
        ids = self._batches.get(batch_id, [])
        for idx in ids:
            ev = self._events[idx]
            ev["use_wall"] = use_wall
            ev["use_mono"] = use_mono
            ev["latency_ingest_to_use"] = use_mono - ev.get("ingest_mono", use_mono)
            send_ts_mono = ev.get("send_ts_mono")
            if isinstance(send_ts_mono, (int, float)):
                try:
                    ev["e2e_send_to_use"] = use_mono - float(send_ts_mono)
                except Exception:
                    pass

    def complete_batch(
        self,
        batch_id: int,
        opt_wall: float,
        opt_mono: float,
        broadcast_wall: float,
        broadcast_mono: float,
        payload_bytes: Optional[int] = None,
    ) -> None:
        ids = self._batches.pop(batch_id, [])
        for idx in ids:
            ev = self._events[idx]
            ev["opt_wall"] = opt_wall
            ev["opt_mono"] = opt_mono
            ev["broadcast_wall"] = broadcast_wall
            ev["broadcast_mono"] = broadcast_mono
            ev["latency_ingest_to_opt"] = opt_mono - ev["ingest_mono"]
            ev["latency_ingest_to_broadcast"] = broadcast_mono - ev["ingest_mono"]
            send_ts_mono = ev.get("send_ts_mono")
            if isinstance(send_ts_mono, (int, float)):
                try:
                    ev["e2e_send_to_ingest"] = ev["ingest_mono"] - float(send_ts_mono)
                    ev["e2e_send_to_opt"] = opt_mono - float(send_ts_mono)
                    ev["e2e_send_to_broadcast"] = broadcast_mono - float(send_ts_mono)
                except Exception:
                    pass
            if payload_bytes is not None:
                ev["broadcast_bytes"] = payload_bytes

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        ing_opt = [ev["latency_ingest_to_opt"] for ev in self._events if "latency_ingest_to_opt" in ev]
        ing_brd = [ev["latency_ingest_to_broadcast"] for ev in self._events if "latency_ingest_to_broadcast" in ev]
        e2e_ing = [ev["e2e_send_to_ingest"] for ev in self._events if "e2e_send_to_ingest" in ev]
        e2e_use = [ev["e2e_send_to_use"] for ev in self._events if "e2e_send_to_use" in ev]
        return {
            "events": len(ing_opt),
            "ingest_to_opt": _stats(ing_opt),
            "ingest_to_broadcast": _stats(ing_brd),
            "e2e_send_to_ingest": _stats(e2e_ing),
            "e2e_send_to_use": _stats(e2e_use),
        }

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"events": self._events, "summary": self.summary()}, f, indent=2)

    @property
    def events(self) -> List[Dict]:
        return list(self._events)

