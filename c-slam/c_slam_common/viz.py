from __future__ import annotations

from typing import Dict
import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _point3_to_xyz_any(t):
    if hasattr(t, "x"):
        try:
            return float(t.x()), float(t.y()), float(t.z())
        except Exception:
            pass
    try:
        v = getattr(t, "vector", lambda: t)()
    except Exception:
        v = t
    return float(v[0]), float(v[1]), float(v[2])


def extract_xyz_per_robot(estimate, by_robot_keys: Dict[str, set]) -> Dict[str, np.ndarray]:
    out = {}
    for rid, keys in by_robot_keys.items():
        coords = []
        for k in sorted(keys, key=lambda x: str(x)):
            if estimate.exists(k):
                p = estimate.atPose3(k)
                t = p.translation()
                x, y, z = _point3_to_xyz_any(t)
                coords.append([x, y, z])
        if coords:
            out[rid] = np.asarray(coords)
    return out


def plot_trajectories_2d(estimate, by_robot_keys: Dict[str, set], path_png: str):
    traj = extract_xyz_per_robot(estimate, by_robot_keys)
    plt.figure(figsize=(8, 6))
    for rid, xyz in traj.items():
        plt.plot(xyz[:, 0], xyz[:, 1], label=rid)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.title("Trajectories (XY)")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def plot_trajectories_3d(estimate, by_robot_keys: Dict[str, set], path_png: str):
    traj = extract_xyz_per_robot(estimate, by_robot_keys)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for rid, xyz in traj.items():
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=rid)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_title("Trajectories (3D)")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def _apply_alignment(xyz: np.ndarray, align_info: dict | None) -> np.ndarray:
    if not isinstance(align_info, dict):
        return xyz
    R = np.asarray(align_info.get("R", np.eye(3)), dtype=float)
    t = np.asarray(align_info.get("t", np.zeros(3)), dtype=float)
    try:
        s = float(align_info.get("s", 1.0) or 1.0)
    except Exception:
        s = 1.0
    if R.shape != (3, 3):
        R = np.eye(3)
    if t.shape != (3,):
        t = np.zeros(3)
    return (xyz @ R.T) * s + t


def plot_est_vs_gt_xy_aligned(estimate, by_robot_keys, gt_map, align_metrics, out_path: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    for rid, keys in by_robot_keys.items():
        xs, ys = [], []
        gxs, gys = [], []
        gt_for_r = gt_map.get(rid, {})
        align_info = align_metrics.get(rid, {}) if align_metrics else {}
        raw = []
        for k in sorted(keys, key=lambda x: str(x)):
            if estimate.exists(k):
                p = estimate.atPose3(k)
                x, y, z = _point3_to_xyz_any(p.translation())
                raw.append([x, y, z])
            if str(k) in gt_for_r:
                gt_t = gt_for_r[str(k)][1]
                if hasattr(gt_t, "x"):
                    gx, gy, _ = float(gt_t.x), float(gt_t.y), float(gt_t.z)
                else:
                    gx, gy, _ = gt_t
                gxs.append(gx)
                gys.append(gy)
        if raw:
            xyz = np.asarray(raw, dtype=float)
            xyz_al = _apply_alignment(xyz, align_info)
            xs, ys = xyz_al[:, 0].tolist(), xyz_al[:, 1].tolist()
        if xs:
            plt.plot(xs, ys, label=f"{rid} est (aligned)")
        if gxs:
            plt.plot(gxs, gys, linestyle="--", label=f"{rid} gt")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectories (XY): est vs ground truth (aligned)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_est_vs_gt_3d_aligned(estimate, by_robot_keys, gt_map, align_metrics, out_path: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for rid, keys in by_robot_keys.items():
        xs, ys, zs = [], [], []
        gxs, gys, gzs = [], [], []
        gt_for_r = gt_map.get(rid, {})
        align_info = align_metrics.get(rid, {}) if align_metrics else {}
        raw = []
        for k in sorted(keys, key=lambda x: str(x)):
            if estimate.exists(k):
                p = estimate.atPose3(k)
                x, y, z = _point3_to_xyz_any(p.translation())
                raw.append([x, y, z])
            if str(k) in gt_for_r:
                gt_t = gt_for_r[str(k)][1]
                if hasattr(gt_t, "x"):
                    gx, gy, gz = float(gt_t.x), float(gt_t.y), float(gt_t.z)
                else:
                    gx, gy, gz = gt_t
                gxs.append(gx)
                gys.append(gy)
                gzs.append(gz)
        if raw:
            xyz = np.asarray(raw, dtype=float)
            xyz_al = _apply_alignment(xyz, align_info)
            xs, ys, zs = xyz_al[:, 0].tolist(), xyz_al[:, 1].tolist(), xyz_al[:, 2].tolist()
        if xs:
            ax.plot(xs, ys, zs, label=f"{rid} est (aligned)")
        if gxs:
            ax.plot(gxs, gys, gzs, linestyle="--", label=f"{rid} gt")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trajectories (3D): est vs ground truth (aligned)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
