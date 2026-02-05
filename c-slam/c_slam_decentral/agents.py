from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import logging
import math
import time

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

from c_slam_common.graph import GraphBuilder, default_robot_infer
from c_slam_common.isam2 import ISAM2Manager, update_translation_cache
from c_slam_common.kpi_logging import KPILogger
from c_slam_common.models import BetweenFactorPose3, InitEntry, PriorFactorPose3, Quaternion, Translation
from c_slam_decentral.communication import InterfaceMessage

logger = logging.getLogger("c_slam.decentral.agent")


@dataclass(frozen=True)
class _InterfaceEdge:
    factor: BetweenFactorPose3
    local_key: str
    remote_key: str
    remote_robot: str
    owns_factor: bool


def _rot3_to_quat_wxyz(rot: "gtsam.Rot3") -> Tuple[float, float, float, float]:
    if hasattr(rot, "quaternion"):
        q = rot.quaternion()
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])
    M = rot.matrix()
    m00, m01, m02 = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    m10, m11, m12 = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    m20, m21, m22 = float(M[2, 0]), float(M[2, 1]), float(M[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    return qw / n, qx / n, qy / n, qz / n


def _pose3_to_components(pose: "gtsam.Pose3") -> Tuple[Quaternion, Translation]:
    R = pose.rotation()
    qw, qx, qy, qz = _rot3_to_quat_wxyz(R)
    t = pose.translation()
    try:
        tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
    except AttributeError:
        tx, ty, tz = map(float, t)
    return Quaternion(qw, qx, qy, qz), Translation(tx, ty, tz)


def _is_indeterminant_system_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return ("Indeterminant linear system" in msg) or ("IndeterminantLinearSystemException" in msg)


def _is_duplicate_key_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return ("key already exists" in msg) or ("Attempting to add a key-value pair with key" in msg)


class DecentralizedRobotAgent:
    """Per-robot DDF agent with persistent iSAM2 local solver."""

    def __init__(
        self,
        robot_id: str,
        robot_map: Dict[str, str],
        init_lookup: Dict[str, InitEntry],
        *,
        robust_kind: Optional[str] = None,
        robust_k: Optional[float] = None,
        bootstrap_sigma: float = 1e1,
        relin_threshold: float = 0.1,
        relin_skip: int = 10,
    ) -> None:
        if gtsam is None:
            raise RuntimeError("gtsam bindings required")
        self.robot_id = str(robot_id)
        self._robot_map = {str(k): str(v) for k, v in robot_map.items()}
        self._base_init_lookup = {str(k): v for k, v in init_lookup.items()}
        self._robust_kind = robust_kind
        self._robust_k = robust_k
        self._bootstrap_sigma = float(bootstrap_sigma)
        self._relin_threshold = float(relin_threshold)
        self._relin_skip = int(relin_skip)

        self._local_priors: List[PriorFactorPose3] = []
        self._local_between: List[BetweenFactorPose3] = []
        self._interface_edges: List[_InterfaceEdge] = []

        self._remote_estimates: Dict[str, InterfaceMessage] = {}

        self._isam = ISAM2Manager(relinearize_threshold=self._relin_threshold, relinearize_skip=self._relin_skip)
        self._gb = GraphBuilder(robot_map=self._robot_map, robust_kind=self._robust_kind, robust_k=self._robust_k)
        self._init_lookup: Dict[str, InitEntry] = dict(self._base_init_lookup)
        self._static_added = False
        self._inserted_edge_ids: Set[Tuple[str, str, float]] = set()
        self._remote_prior_indices: Dict[str, List[int]] = {}
        self._translation_cache: Dict[int, tuple] = {}

        self._latest_estimate: Optional["gtsam.Values"] = None
        self._latest_graph: Optional[GraphBuilder] = None
        self._latest_marginals: Optional["gtsam.Marginals"] = None
        self._retry_pending_batch = False

    def _robot_of(self, key: Union[str, int]) -> str:
        skey = str(key)
        if skey in self._robot_map:
            return self._robot_map[skey]
        return default_robot_infer(skey)

    def ingest_factor(self, factor: Union[PriorFactorPose3, BetweenFactorPose3]) -> None:
        if isinstance(factor, PriorFactorPose3):
            if self._robot_of(factor.key) == self.robot_id:
                self._local_priors.append(factor)
            return
        if isinstance(factor, BetweenFactorPose3):
            rid1 = self._robot_of(factor.key1)
            rid2 = self._robot_of(factor.key2)
            if rid1 == rid2 == self.robot_id:
                self._local_between.append(factor)
                return
            if self.robot_id not in (rid1, rid2):
                return
            if rid1 == self.robot_id:
                local_key = str(factor.key1)
                remote_key = str(factor.key2)
                remote_robot = rid2
            else:
                local_key = str(factor.key2)
                remote_key = str(factor.key1)
                remote_robot = rid1
            owner_robot = min(rid1, rid2)
            self._interface_edges.append(
                _InterfaceEdge(
                    factor=factor,
                    local_key=local_key,
                    remote_key=remote_key,
                    remote_robot=remote_robot,
                    owns_factor=(self.robot_id == owner_robot),
                )
            )

    def receive_interface_messages(self, messages: Iterable[InterfaceMessage]) -> None:
        for msg in messages:
            key = str(msg.key)
            prev = self._remote_estimates.get(key)
            if prev is not None and int(getattr(msg, "iteration", 0)) <= int(getattr(prev, "iteration", -1)):
                continue
            self._remote_estimates[key] = msg

    def _bootstrap_prior(self) -> Optional[PriorFactorPose3]:
        # No synthetic bootstrap priors; rely solely on dataset-provided priors.
        return None

    def _static_priors(self) -> List[PriorFactorPose3]:
        priors = list(self._local_priors)
        return priors

    def _build_remote_prior(self, key: str) -> Optional[PriorFactorPose3]:
        key_str = str(key)
        if key_str in self._remote_estimates:
            msg = self._remote_estimates[key_str]
            self._init_lookup[key_str] = InitEntry(key=key_str, rotation=msg.rotation, translation=msg.translation)
            cov = np.asarray(msg.covariance, dtype=float)
            if cov.shape != (6, 6):
                cov = np.eye(6) * (self._bootstrap_sigma**2)
            return PriorFactorPose3(
                key=key_str,
                rotation=msg.rotation,
                translation=msg.translation,
                covariance=cov,
                stamp=float(getattr(msg, "stamp", 0.0) or 0.0),
            )
        entry = self._init_lookup.get(key_str) or self._base_init_lookup.get(key_str)
        if entry is None:
            return None
        cov = np.eye(6) * (self._bootstrap_sigma**2)
        return PriorFactorPose3(key=key_str, rotation=entry.rotation, translation=entry.translation, covariance=cov, stamp=0.0)

    def _edge_identifier(self, edge: _InterfaceEdge) -> Tuple[str, str, float]:
        return (edge.local_key, edge.remote_key, float(getattr(edge.factor, "stamp", 0.0)))

    def solve_round(self, iteration: int, kpi: Optional[KPILogger] = None) -> Optional["gtsam.Values"]:
        if not self._static_added:
            for prior in self._static_priors():
                self._gb.add_prior(prior, self._init_lookup)
            for f in self._local_between:
                self._gb.add_between(f, self._init_lookup)
            for edge in self._interface_edges:
                if not edge.owns_factor:
                    continue
                eid = self._edge_identifier(edge)
                if eid in self._inserted_edge_ids:
                    continue
                self._gb.add_between(edge.factor, self._init_lookup)
                self._inserted_edge_ids.add(eid)
            self._static_added = True

        # Remote priors (replace each round). If we are retrying a previously
        # failed iSAM2 update, do not append duplicate remote priors.
        remote_priors: List[PriorFactorPose3] = []
        if not self._retry_pending_batch:
            remote_keys = sorted({edge.remote_key for edge in self._interface_edges})
            for key in remote_keys:
                p = self._build_remote_prior(key)
                if p is not None:
                    remote_priors.append(p)
            for p in remote_priors:
                self._gb.add_prior(p, self._init_lookup, to_global=False)

        remove_indices: List[int] = []
        if self._remote_prior_indices:
            seen: Set[int] = set()
            for indices in self._remote_prior_indices.values():
                for idx in indices:
                    if idx in seen:
                        continue
                    remove_indices.append(int(idx))
                    seen.add(int(idx))
        remove_indices.sort()

        prev_factor_count = 0
        try:
            prev_factor_count = int(self._isam.isam.getFactorsUnsafe().size())  # type: ignore[attr-defined]
        except Exception:
            prev_factor_count = 0

        bg, bv = self._gb.pop_batch()
        do_update = bg.size() > 0 or bool(remove_indices)
        res = None
        start = time.perf_counter()
        if do_update and kpi:
            try:
                kpi.optimization_start(int(iteration), bg.size(), bg.size())
            except Exception:
                pass
        if do_update:
            try:
                res = self._isam.update(bg, bv, remove_factor_indices=remove_indices if remove_indices else None)
            except RuntimeError as exc:
                # Underconstrained updates are expected under packet loss: keep buffering until solvable.
                if _is_indeterminant_system_error(exc):
                    try:
                        self._gb.restore_batch(bg, bv)
                    except Exception:
                        pass
                    self._retry_pending_batch = True
                    if kpi:
                        try:
                            kpi.emit("optimization_failed", iteration=int(iteration), error=str(exc), solver="ddf_sam")
                        except Exception:
                            pass
                    return self._latest_estimate

                # Some GTSAM builds may partially mutate iSAM2 state before throwing, leading to
                # duplicate-key errors on subsequent retries. Recover by rebuilding from scratch.
                if _is_duplicate_key_error(exc):
                    if kpi:
                        try:
                            kpi.emit("optimization_failed", iteration=int(iteration), error=str(exc), solver="ddf_sam")
                        except Exception:
                            pass
                    try:
                        # Rebuild graph: static + owned interface factors + *fresh* remote priors.
                        remote_keys = sorted({edge.remote_key for edge in self._interface_edges})
                        rebuild_remote_priors: List[PriorFactorPose3] = []
                        for key in remote_keys:
                            p = self._build_remote_prior(key)
                            if p is not None:
                                rebuild_remote_priors.append(p)

                        rebuild_gb = GraphBuilder(robot_map=self._robot_map, robust_kind=self._robust_kind, robust_k=self._robust_k)
                        for prior in self._static_priors():
                            rebuild_gb.add_prior(prior, self._init_lookup)
                        for f in self._local_between:
                            rebuild_gb.add_between(f, self._init_lookup)
                        for edge in self._interface_edges:
                            if not edge.owns_factor:
                                continue
                            rebuild_gb.add_between(edge.factor, self._init_lookup)
                        for p in rebuild_remote_priors:
                            rebuild_gb.add_prior(p, self._init_lookup)

                        new_isam = ISAM2Manager(relinearize_threshold=self._relin_threshold, relinearize_skip=self._relin_skip)
                        rebuild_start = time.perf_counter()
                        rebuild_res = new_isam.update(rebuild_gb.graph, rebuild_gb.initial, remove_factor_indices=None)
                        rebuild_dt = time.perf_counter() - rebuild_start
                        self._isam = new_isam
                        self._translation_cache = {}
                        self._retry_pending_batch = False

                        # Refresh remote prior indices for removal in subsequent rounds.
                        try:
                            new_indices = []
                            if rebuild_res is not None and hasattr(rebuild_res, "getNewFactorsIndices"):
                                new_indices = list(rebuild_res.getNewFactorsIndices())
                            if not new_indices:
                                new_indices = list(range(int(rebuild_gb.graph.size())))
                            tail = new_indices[-len(rebuild_remote_priors) :] if rebuild_remote_priors else []
                            self._remote_prior_indices = {str(p.key): [int(idx)] for p, idx in zip(rebuild_remote_priors, tail)}
                        except Exception:
                            self._remote_prior_indices = {}

                        if kpi:
                            try:
                                kpi.emit("full_rebuild_ok", iteration=int(iteration), duration_s=float(rebuild_dt), solver="ddf_sam")
                            except Exception:
                                pass
                        estimate = self._isam.estimate
                        max_delta = update_translation_cache(self._translation_cache, estimate)
                        if kpi:
                            try:
                                kpi.optimization_end(
                                    int(iteration),
                                    rebuild_dt,
                                    updated_keys=estimate.size() if hasattr(estimate, "size") else None,
                                    max_translation_delta=max_delta,
                                    solver="ddf_sam",
                                )
                                kpi.emit("ddf_round_delta", iteration=int(iteration), max_translation_delta=max_delta, solver="ddf_sam")
                            except Exception:
                                pass
                        self._latest_estimate = estimate
                        self._latest_graph = rebuild_gb
                        try:
                            self._latest_marginals = gtsam.Marginals(rebuild_gb.graph, estimate)
                        except Exception:
                            self._latest_marginals = None
                        return estimate
                    except Exception as rebuild_exc:
                        if kpi:
                            try:
                                kpi.emit("full_rebuild_failed", iteration=int(iteration), error=str(rebuild_exc), solver="ddf_sam")
                            except Exception:
                                pass
                        return self._latest_estimate

                raise
        self._retry_pending_batch = False
        duration = time.perf_counter() - start
        estimate = self._isam.estimate
        max_delta = update_translation_cache(self._translation_cache, estimate)
        if do_update and kpi:
            try:
                kpi.optimization_end(
                    int(iteration),
                    duration,
                    updated_keys=estimate.size() if hasattr(estimate, "size") else None,
                    max_translation_delta=max_delta,
                    solver="ddf_sam",
                )
                kpi.emit("ddf_round_delta", iteration=int(iteration), max_translation_delta=max_delta, solver="ddf_sam")
            except Exception:
                pass

        new_indices: List[int] = []
        if res is not None and hasattr(res, "getNewFactorsIndices"):
            try:
                new_indices = list(res.getNewFactorsIndices())
            except Exception:
                new_indices = []
        if not new_indices and do_update:
            base_count = max(0, prev_factor_count - len(remove_indices))
            new_indices = list(range(base_count, base_count + bg.size()))

        # Assume remote priors are appended last in this update call.
        if remote_priors:
            tail = new_indices[-len(remote_priors) :] if new_indices else []
            self._remote_prior_indices = {str(p.key): [int(idx)] for p, idx in zip(remote_priors, tail)}
        else:
            self._remote_prior_indices = {}

        # Build a lightweight "view" graph (static + remote priors) for marginals/export.
        view_gb = GraphBuilder(robot_map=self._robot_map, robust_kind=self._robust_kind, robust_k=self._robust_k)
        for prior in self._static_priors():
            view_gb.add_prior(prior, self._init_lookup)
        for f in self._local_between:
            view_gb.add_between(f, self._init_lookup)
        for edge in self._interface_edges:
            if not edge.owns_factor:
                continue
            view_gb.add_between(edge.factor, self._init_lookup)
        for p in remote_priors:
            view_gb.add_prior(p, self._init_lookup)

        self._latest_estimate = estimate
        self._latest_graph = view_gb
        try:
            self._latest_marginals = gtsam.Marginals(view_gb.graph, estimate)
        except Exception:
            self._latest_marginals = None

        _ = duration
        return estimate

    def interface_messages(self, iteration: int) -> List[InterfaceMessage]:
        if self._latest_estimate is None or self._latest_graph is None:
            return []
        msgs: Dict[Tuple[str, str], InterfaceMessage] = {}
        marginals = self._latest_marginals
        gb = self._latest_graph
        for edge in self._interface_edges:
            norm_key = gb.normalize_key(edge.local_key)
            if not self._latest_estimate.exists(norm_key):
                continue
            pose = self._latest_estimate.atPose3(norm_key)
            rot, trans = _pose3_to_components(pose)
            covariance = np.eye(6) * (self._bootstrap_sigma**2)
            if marginals is not None:
                try:
                    covariance = np.asarray(marginals.marginalCovariance(norm_key))
                except Exception:
                    covariance = np.eye(6) * (self._bootstrap_sigma**2)
            key = (edge.remote_robot, edge.local_key)
            if key in msgs:
                continue
            msgs[key] = InterfaceMessage(
                sender=self.robot_id,
                receiver=edge.remote_robot,
                key=edge.local_key,
                rotation=rot,
                translation=trans,
                covariance=covariance,
                iteration=int(iteration),
                stamp=float(getattr(edge.factor, "stamp", 0.0)),
            )
        return list(msgs.values())

    def estimate_snapshot(self) -> Optional["gtsam.Values"]:
        return self._latest_estimate

    def graph_snapshot(self) -> Optional[GraphBuilder]:
        return self._latest_graph
