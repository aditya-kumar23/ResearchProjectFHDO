from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class Quaternion:
    """Quaternion in [w, x, y, z] order."""

    w: float
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)


@dataclass
class Translation:
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class InitEntry:
    key: Union[str, int]
    rotation: Quaternion
    translation: Translation
    type: str = "Pose3"


@dataclass
class PriorFactorPose3:
    key: Union[str, int]
    rotation: Quaternion
    translation: Translation
    covariance: np.ndarray  # 6x6
    stamp: float


@dataclass
class BetweenFactorPose3:
    key1: Union[str, int]
    key2: Union[str, int]
    rotation: Quaternion
    translation: Translation
    covariance: np.ndarray  # 6x6
    stamp: float


@dataclass
class JRLDocument:
    measurements: Dict[str, List[Dict[str, Any]]]
    outlier_factors: List[Dict[str, Any]]
    potential_outlier_factors: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    initialisation: List[Dict[str, Any]] | Dict[str, List[Dict[str, Any]]]


def to_covariance(cov_list: List[float]) -> np.ndarray:
    """Convert a flat list (36) or nested list (6x6) to a 6x6 ndarray."""

    arr = np.asarray(cov_list, dtype=float)
    if arr.size == 36 and arr.ndim == 1:
        return arr.reshape(6, 6)
    if arr.size == 36 and arr.ndim == 2 and arr.shape == (6, 6):
        return arr
    raise ValueError(f"Expected 36 elements for a 6x6 covariance, got shape {arr.shape} size {arr.size}")

