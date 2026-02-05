from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, Optional

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

logger = logging.getLogger("c_slam.isam2")


def update_translation_cache(cache: Dict[int, tuple], estimate: "gtsam.Values") -> float:
    if gtsam is None or estimate is None:
        return 0.0
    max_delta = 0.0
    try:
        keys = list(estimate.keys())
    except Exception:
        keys = []
    for key in keys:
        try:
            pose = estimate.atPose3(key)
        except Exception:
            continue
        trans = pose.translation()
        try:
            tx, ty, tz = float(trans.x()), float(trans.y()), float(trans.z())
        except Exception:
            tx = float(trans[0])
            ty = float(trans[1])
            tz = float(trans[2])
        prev = cache.get(int(key))
        if prev is not None:
            dx = tx - prev[0]
            dy = ty - prev[1]
            dz = tz - prev[2]
            delta = math.sqrt(dx * dx + dy * dy + dz * dz)
            if delta > max_delta:
                max_delta = delta
        cache[int(key)] = (tx, ty, tz)
    return max_delta


class ISAM2Manager:
    """Thin manager around GTSAM's iSAM2 with removeFactorIndices support."""

    def __init__(
        self,
        *,
        relinearize_threshold: float = 0.1,
        relinearize_skip: int = 10,
        cache_linearized: bool = True,
    ):
        if gtsam is None:
            raise RuntimeError("GTSAM not available; cannot run iSAM2")
        params = gtsam.ISAM2Params()

        def _set(obj, prop: str, value, setter: Optional[str] = None):
            if hasattr(obj, prop):
                try:
                    setattr(obj, prop, value)
                    return
                except Exception:
                    pass
            if setter and hasattr(obj, setter):
                getattr(obj, setter)(value)

        _set(params, "relinearizeThreshold", relinearize_threshold, "setRelinearizeThreshold")
        _set(params, "relinearizeSkip", relinearize_skip, "setRelinearizeSkip")
        _set(params, "cacheLinearizedFactors", cache_linearized, "setCacheLinearizedFactors")
        _set(params, "enableRelinearization", True, "setEnableRelinearization")

        self.isam = gtsam.ISAM2(params)
        self._estimate = gtsam.Values()
        self._warned_remove = False

    def update(self, graph: "gtsam.NonlinearFactorGraph", initial: "gtsam.Values", *,
               remove_factor_indices: Optional[Iterable[int]] = None):
        try:
            if remove_factor_indices:
                res = self.isam.update(graph, initial, removeFactorIndices=list(remove_factor_indices))
            else:
                res = self.isam.update(graph, initial)
        except TypeError:
            if remove_factor_indices and not self._warned_remove:
                logger.warning("ISAM2.update does not support removeFactorIndices in this wheel")
                self._warned_remove = True
            res = self.isam.update(graph, initial)
        self._estimate = self.isam.calculateEstimate()
        return res

    @property
    def estimate(self) -> "gtsam.Values":
        return self._estimate

