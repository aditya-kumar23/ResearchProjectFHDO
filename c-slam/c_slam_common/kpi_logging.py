from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("c_slam.kpi")


class KPILogger:
    def __init__(
        self,
        *,
        enabled: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
        log_path: Optional[str] = None,
        emit_to_logger: bool = True,
    ):
        self.enabled = bool(enabled)
        self._extra = extra_fields.copy() if extra_fields else {}
        self._emit_to_logger = bool(emit_to_logger)
        self._fh = open(log_path, "w", encoding="utf-8") if log_path else None

    def _emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {"event": event, "ts": time.time()}
        payload.update(self._extra)
        payload.update({k: v for k, v in fields.items() if v is not None})
        if self._emit_to_logger:
            logger.info("KPI %s", json.dumps(payload, sort_keys=True))
        if self._fh:
            self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
            self._fh.flush()

    def emit(self, event: str, **fields: Any) -> None:
        """Emit a custom KPI event."""
        self._emit(event, **fields)

    def sensor_ingest(self, factor_type: str, stamp: float, **fields: Any) -> None:
        self._emit("sensor_ingest", factor_type=factor_type, stamp=stamp, **fields)

    def optimization_start(self, batch_id: int, factor_count: int, pending_factors: int) -> None:
        self._emit("optimization_start", batch_id=batch_id, factor_count=factor_count, pending_factors=pending_factors)

    def optimization_end(self, batch_id: int, duration_s: float, updated_keys: Optional[int] = None, **fields: Any) -> None:
        self._emit("optimization_end", batch_id=batch_id, duration_s=duration_s, updated_keys=updated_keys, **fields)

    def map_broadcast(self, batch_id: int, pose_count: Optional[int] = None, **fields: Any) -> None:
        self._emit("map_broadcast", batch_id=batch_id, pose_count=pose_count, **fields)

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None
