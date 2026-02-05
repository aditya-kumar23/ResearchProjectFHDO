from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore


def _point3_to_xyz_any(t):
    if hasattr(t, "x"):
        try:
            return float(t.x()), float(t.y()), float(t.z())
        except Exception:
            pass
    v = getattr(t, "vector", lambda: t)()
    return float(v[0]), float(v[1]), float(v[2])


def _pose_xyz(values, k):
    p = values.atPose3(k)
    return _point3_to_xyz_any(p.translation())


def _umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool = False):
    """Rigid (optionally similarity) alignment from A->B (Nx3). Returns R(3x3), t(3), s."""
    assert A.shape == B.shape and A.shape[1] == 3
    muA, muB = A.mean(0), B.mean(0)
    AA, BB = A - muA, B - muB
    C = AA.T @ BB / A.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if with_scale:
        varA = (AA**2).sum() / A.shape[0]
        s = (S.sum() / varA) if varA > 0 else 1.0
    else:
        s = 1.0
    t = muB - s * (R @ muA)
    return R, t, s


def align_and_ate_per_robot(
    estimate: "gtsam.Values",
    by_robot_keys: Dict[str, Iterable],
    gt_map: Dict[str, Dict[str, tuple]],
    key_label=lambda k: str(k),
):
    """Compute SE(3) alignment and ATE-RMSE per robot. Returns metrics dict."""
    metrics = {}
    for rid, keys in by_robot_keys.items():
        gt_for_r = gt_map.get(rid, {})
        common = [k for k in keys if key_label(k) in gt_for_r and estimate.exists(k)]
        if len(common) < 3:
            metrics[rid] = {"matches": len(common), "rmse": None}
            continue
        try:
            X_est = np.array([_pose_xyz(estimate, k) for k in common], dtype=float)
            X_gt = np.array([_translation_to_xyz(gt_for_r[key_label(k)][1]) for k in common], dtype=float)
        except Exception:
            metrics[rid] = {"matches": len(common), "rmse": None}
            continue
        if X_est.shape != X_gt.shape or X_est.shape[1] != 3:
            metrics[rid] = {"matches": len(common), "rmse": None}
            continue
        R, t, s = _umeyama(X_est, X_gt, with_scale=False)
        X_aligned = (X_est @ R.T) + t
        err = X_aligned - X_gt
        rmse = math.sqrt((err**2).sum(axis=1).mean())
        metrics[rid] = {
            "matches": len(common),
            "rmse": rmse,
            "R": R.tolist(),
            "t": t.tolist(),
            "s": s,
        }
    return metrics


def _translation_to_xyz(trans) -> Tuple[float, float, float]:
    if hasattr(trans, "x"):
        try:
            return float(trans.x), float(trans.y), float(trans.z)
        except Exception:
            pass
    if hasattr(trans, "vector"):
        vec = trans.vector()
        return float(vec[0]), float(vec[1]), float(vec[2])
    if isinstance(trans, (np.ndarray, list, tuple)) and len(trans) >= 3:
        return float(trans[0]), float(trans[1]), float(trans[2])
    raise TypeError(f"Unsupported translation type {type(trans)}")


def compute_rpe_per_robot(
    estimate: "gtsam.Values",
    by_robot_keys: Dict[str, Iterable],
    gt_map: Dict[str, Dict[str, tuple]],
    align_metrics: Dict[str, Dict[str, object]],
    *,
    key_label=lambda k: str(k),
    window_sizes: Sequence[int] = (1, 10, 50),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute Relative Pose Error (RPE) per robot for multiple window sizes."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rid, keys in by_robot_keys.items():
        align_info = align_metrics.get(rid, {}) if align_metrics else {}
        R = np.asarray(align_info.get("R", np.eye(3)), dtype=float)
        if R.shape != (3, 3):
            R = np.eye(3)
        t = np.asarray(align_info.get("t", np.zeros(3)), dtype=float)
        if t.shape != (3,):
            t = np.zeros(3)
        s = float(align_info.get("s", 1.0) or 1.0)

        gt_entries = gt_map.get(rid, {})
        if not gt_entries:
            continue

        seq = []
        for key in keys:
            label = key_label(key)
            if label not in gt_entries or not estimate.exists(key):
                continue
            est_xyz = np.asarray(_pose_xyz(estimate, key), dtype=float)
            gt_rot, gt_trans, stamp = gt_entries[label]
            gt_xyz = np.asarray(_translation_to_xyz(gt_trans), dtype=float)
            try:
                stamp_val = float(stamp)
            except Exception:
                stamp_val = float(len(seq))
            seq.append((label, est_xyz, gt_xyz, stamp_val))

        if len(seq) < 2:
            continue

        seq.sort(key=lambda item: (item[3], item[0]))
        est_aligned = []
        gt_series = []
        stamps = []
        for _, est_xyz, gt_xyz, stamp in seq:
            aligned = (est_xyz @ R.T) * s + t
            est_aligned.append(aligned)
            gt_series.append(gt_xyz)
            stamps.append(stamp)

        est_arr = np.stack(est_aligned, axis=0)
        gt_arr = np.stack(gt_series, axis=0)
        stamps_arr = np.asarray(stamps, dtype=float)

        window_stats: Dict[str, Dict[str, float]] = {}
        for window in window_sizes:
            window = int(window)
            if window <= 0:
                continue
            errors = []
            deltas = []
            for i in range(0, len(est_arr) - window):
                j = i + window
                rel_est = est_arr[j] - est_arr[i]
                rel_gt = gt_arr[j] - gt_arr[i]
                diff = rel_est - rel_gt
                errors.append(float(np.linalg.norm(diff)))
                deltas.append(float(stamps_arr[j] - stamps_arr[i]))
            if not errors:
                continue
            err_sq = [e * e for e in errors]
            rmse = math.sqrt(sum(err_sq) / len(err_sq))
            stats = {
                "count": len(errors),
                "rmse": rmse,
                "mean": mean(errors),
                "pstd": pstdev(errors) if len(errors) > 1 else 0.0,
                "mean_dt": mean(deltas),
            }
            window_stats[str(window)] = {k: float(v) for k, v in stats.items()}
        if window_stats:
            results[rid] = window_stats
    return results
