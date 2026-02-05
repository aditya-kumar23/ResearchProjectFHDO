#!/usr/bin/env python3
"""Derived KPI computations for c-slam outputs (centralised + decentralised)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from c_slam_common.stabilisation import DEFAULT_STABLE_EPSILON, DEFAULT_STABLE_REQUIRED


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not vals:
        return {"count": 0}
    import statistics

    def pct(p: float) -> Optional[float]:
        if not vals:
            return None
        if p <= 0:
            return vals[0]
        if p >= 100:
            return vals[-1]
        rank = (p / 100.0) * (len(vals) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(vals) - 1)
        weight = rank - lower
        return vals[lower] * (1 - weight) + vals[upper] * weight

    out: Dict[str, Optional[float]] = {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p90": pct(90.0),
        "p95": pct(95.0),
        "p99": pct(99.0),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.pstdev(vals)
    return out


def _infer_robot(key: str) -> str:
    if key is None:
        return "global"
    s = str(key)
    prefix = []
    for ch in s:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return "".join(prefix) or "global"


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def extract_loop_closure_correction_times(latency_json: Dict[str, Any], robot_map: Optional[Dict[str, str]] = None) -> List[float]:
    events = latency_json.get("events") if isinstance(latency_json, dict) else None
    if not isinstance(events, list):
        return []
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") != "BetweenFactorPose3":
            continue
        k1 = ev.get("key1")
        k2 = ev.get("key2")
        if not (isinstance(k1, (str, int)) and isinstance(k2, (str, int))):
            continue
        r1 = str(robot_map.get(str(k1))) if robot_map and str(k1) in robot_map else _infer_robot(str(k1))
        r2 = str(robot_map.get(str(k2))) if robot_map and str(k2) in robot_map else _infer_robot(str(k2))
        if r1 == r2:
            continue
        d = _safe_float(ev.get("latency_ingest_to_broadcast"))
        if d is not None and d >= 0.0:
            out.append(d)
    return out


def summarise_loop_closure_correction(latency_json: Dict[str, Any], robot_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    vals = extract_loop_closure_correction_times(latency_json, robot_map=robot_map)
    return {"loop_closure_correction": _stats(vals)}


def _stabilisation_time_after(
    events: List[Dict[str, Any]],
    *,
    start_ts: float,
    event_name: str,
    delta_field: str = "max_translation_delta",
    solver_allow: Optional[Iterable[str]] = None,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> Optional[float]:
    if not events:
        return None
    filtered: List[Tuple[float, float]] = []
    grouped: Dict[int, Tuple[float, float]] = {}
    has_iteration = False
    allow = set(solver_allow) if solver_allow else None
    for ev in events:
        if ev.get("event") != event_name:
            continue
        if allow is not None and ev.get("solver") not in allow:
            continue
        ts = _safe_float(ev.get("ts"))
        delta = _safe_float(ev.get(delta_field))
        if ts is None or delta is None or ts < start_ts or delta < 0:
            continue
        it = ev.get("iteration")
        if it is not None:
            try:
                idx = int(it)
                has_iteration = True
                prev = grouped.get(idx)
                if prev is None or delta > prev[1]:
                    grouped[idx] = (ts, delta)
            except Exception:
                filtered.append((ts, delta))
        else:
            filtered.append((ts, delta))
    if has_iteration:
        filtered.extend(grouped.values())
    if len(filtered) < required:
        return None
    filtered.sort(key=lambda t: t[0])
    for i in range(len(filtered) - required + 1):
        window = [filtered[i + j][1] for j in range(required)]
        if all(v <= epsilon for v in window):
            return filtered[i + required - 1][0]
    return None


def extract_loop_closure_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    robot_map: Optional[Dict[str, str]] = None,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> List[float]:
    if not isinstance(latency_json, dict):
        return []
    events = latency_json.get("events")
    if not isinstance(events, list):
        return []

    use_ddf = any(ev.get("event") == "ddf_round_delta" for ev in kpi_events or [])
    if use_ddf:
        stab_event = "ddf_round_delta"
        delta_field = "max_translation_delta"
        allow = None
    else:
        stab_event = "optimization_end"
        delta_field = "max_translation_delta"
        allow = {"isam2", "batch", "ddf_sam"}

    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") != "BetweenFactorPose3":
            continue
        k1 = ev.get("key1")
        k2 = ev.get("key2")
        if not (isinstance(k1, (int, str)) and isinstance(k2, (int, str))):
            continue
        r1 = str(robot_map.get(str(k1))) if robot_map and str(k1) in robot_map else _infer_robot(str(k1))
        r2 = str(robot_map.get(str(k2))) if robot_map and str(k2) in robot_map else _infer_robot(str(k2))
        if r1 == r2:
            continue
        t_det = _safe_float(ev.get("ingest_wall"))
        if t_det is None:
            continue
        t_stable = _stabilisation_time_after(
            kpi_events,
            start_ts=float(t_det),
            event_name=stab_event,
            delta_field=delta_field,
            solver_allow=allow,
            epsilon=float(epsilon),
            required=int(required),
        )
        if t_stable is None:
            continue
        dt = float(t_stable) - float(t_det)
        if dt >= 0.0:
            out.append(dt)
    return out


def summarise_loop_closure_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    robot_map: Optional[Dict[str, str]] = None,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> Dict[str, Any]:
    vals = extract_loop_closure_correction_stabilised(
        latency_json,
        kpi_events,
        robot_map=robot_map,
        epsilon=epsilon,
        required=required,
    )
    return {
        "loop_closure_correction_stabilised": _stats(vals),
        "loop_closure_correction_stabilised_params": {"epsilon": float(epsilon), "required": int(required)},
    }


@dataclass
class ConvergenceResult:
    seconds: Optional[float]
    start_ts: Optional[float]
    end_ts: Optional[float]
    method: str


def _read_kpi_events(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def _convergence_from_delta_events(
    events: List[Dict[str, Any]],
    *,
    event_name: str,
    delta_field: str = "max_translation_delta",
    solver_allow: Optional[Iterable[str]] = None,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> Optional[Tuple[float, float]]:
    if not events:
        return None
    filtered: List[Tuple[float, float]] = []
    grouped: Dict[int, Tuple[float, float]] = {}
    has_iteration = False
    allow = set(solver_allow) if solver_allow else None
    for ev in events:
        if ev.get("event") != event_name:
            continue
        if bool(ev.get("settle_only", False)):
            continue
        if allow is not None and ev.get("solver") not in allow:
            continue
        ts = _safe_float(ev.get("ts"))
        delta = _safe_float(ev.get(delta_field))
        if ts is None or delta is None or delta < 0:
            continue
        iteration_val = ev.get("iteration")
        if iteration_val is not None:
            try:
                iteration_idx = int(iteration_val)
                has_iteration = True
                prev = grouped.get(iteration_idx)
                if prev is None or delta > prev[1]:
                    grouped[iteration_idx] = (ts, delta)
            except Exception:
                filtered.append((ts, delta))
        else:
            filtered.append((ts, delta))
    if has_iteration:
        filtered.extend(grouped.values())
    if len(filtered) < required:
        return None
    filtered.sort(key=lambda item: item[0])
    start_ts = filtered[0][0]
    for idx in range(len(filtered) - required + 1):
        window = [filtered[idx + j][1] for j in range(required)]
        if all(v <= epsilon for v in window):
            end_ts = filtered[idx + required - 1][0]
            return start_ts, end_ts
    return None


def time_to_global_convergence(run_dir: str) -> ConvergenceResult:
    kpi_path = os.path.join(run_dir, "kpi_metrics", "kpi_events.jsonl")
    events = _read_kpi_events(kpi_path)
    if not events:
        return ConvergenceResult(seconds=None, start_ts=None, end_ts=None, method="no_events")
    first_ts = None
    for ev in events:
        ts_val = _safe_float(ev.get("ts"))
        if ts_val is None:
            continue
        if first_ts is None or ts_val < first_ts:
            first_ts = ts_val
    epsilon_result = _convergence_from_delta_events(
        events,
        event_name="optimization_end",
        solver_allow={"isam2", "batch", "ddf_sam"},
        epsilon=DEFAULT_STABLE_EPSILON,
        required=DEFAULT_STABLE_REQUIRED,
    )
    if epsilon_result is not None:
        start_ts, end_ts = epsilon_result
        if first_ts is not None:
            start_ts = first_ts
        return ConvergenceResult(
            seconds=max(0.0, end_ts - start_ts), start_ts=start_ts, end_ts=end_ts, method="delta_threshold"
        )
    ddf_result = _convergence_from_delta_events(
        events,
        event_name="ddf_round_delta",
        delta_field="max_translation_delta",
        solver_allow=None,
        epsilon=DEFAULT_STABLE_EPSILON,
        required=DEFAULT_STABLE_REQUIRED,
    )
    if ddf_result is not None:
        start_ts, end_ts = ddf_result
        if first_ts is not None:
            start_ts = first_ts
        return ConvergenceResult(
            seconds=max(0.0, end_ts - start_ts), start_ts=start_ts, end_ts=end_ts, method="delta_threshold_ddf"
        )
    ts0 = None
    ts_end = None
    last_brd = None
    last_opt = None
    for ev in events:
        ts = _safe_float(ev.get("ts"))
        if ts is None:
            continue
        if ts0 is None:
            ts0 = ts
        name = ev.get("event")
        if name == "map_broadcast":
            last_brd = ts
        elif name == "optimization_end":
            last_opt = ts
    if last_brd is not None:
        ts_end = last_brd
        method = "last_map_broadcast"
    elif last_opt is not None:
        ts_end = last_opt
        method = "last_optimization_end"
    else:
        ts_end = _safe_float(events[-1].get("ts"))
        method = "last_event_fallback"
    if ts0 is None or ts_end is None:
        return ConvergenceResult(seconds=None, start_ts=ts0, end_ts=ts_end, method=method)
    return ConvergenceResult(seconds=max(0.0, ts_end - ts0), start_ts=ts0, end_ts=ts_end, method=method)


def summarise_input_end(kpi_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise end-of-input (replay finished / last factor ingested) from KPI events."""
    if not kpi_events:
        return {"input_end_team_ts": None, "input_end_method": "no_events", "input_end_per_robot_ts": {}}
    per_robot: Dict[str, float] = {}
    for ev in kpi_events:
        if not isinstance(ev, dict) or ev.get("event") != "input_end":
            continue
        ts = _safe_float(ev.get("ts"))
        if ts is None:
            continue
        rid = ev.get("robot") or "team"
        try:
            rid_s = str(rid)
        except Exception:
            rid_s = "team"
        prev = per_robot.get(rid_s)
        if prev is None or float(ts) > prev:
            per_robot[rid_s] = float(ts)
    if per_robot:
        team_ts = max(per_robot.values())
        return {
            "input_end_team_ts": float(team_ts),
            "input_end_method": "input_end_event",
            "input_end_per_robot_ts": per_robot,
        }
    # Fallback: use last sensor_ingest timestamp if present.
    last_ingest = None
    per_robot_ingest: Dict[str, float] = {}
    for ev in kpi_events:
        if not isinstance(ev, dict) or ev.get("event") != "sensor_ingest":
            continue
        ts = _safe_float(ev.get("ts"))
        if ts is None:
            continue
        rid = ev.get("robot") or "team"
        rid_s = str(rid)
        prev = per_robot_ingest.get(rid_s)
        if prev is None or float(ts) > prev:
            per_robot_ingest[rid_s] = float(ts)
        if last_ingest is None or float(ts) > last_ingest:
            last_ingest = float(ts)
    return {
        "input_end_team_ts": float(last_ingest) if last_ingest is not None else None,
        "input_end_method": "sensor_ingest_fallback" if last_ingest is not None else "no_input_end",
        "input_end_per_robot_ts": per_robot_ingest,
    }


def time_from_input_end_to_team_convergence(
    kpi_events: List[Dict[str, Any]],
    *,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> Dict[str, Any]:
    """
    Shared convergence proxy anchored at end-of-input:

      T_conv(team) = max_r T_conv^(r), where T_conv^(r) is the time from t_input_end(team)
      until the first stable window (epsilon/required) in robot r's update stream.

    If the stable window is not observed for a robot within the run, the value is right-censored at run end.
    """
    if not kpi_events:
        return {
            "time_from_input_end_to_team_convergence_s": None,
            "time_from_input_end_to_team_convergence_method": "no_events",
            "time_from_input_end_to_team_convergence_per_robot_s": {},
            "time_from_input_end_to_team_convergence_params": {"epsilon": float(epsilon), "required": int(required)},
        }
    ts_all = []
    by_robot: Dict[str, List[Tuple[float, float]]] = {}
    for ev in kpi_events:
        if not isinstance(ev, dict):
            continue
        ts = _safe_float(ev.get("ts"))
        if ts is not None:
            ts_all.append(float(ts))
        if ev.get("event") != "optimization_end":
            continue
        if bool(ev.get("settle_only", False)):
            # settle-only updates are allowed for convergence measurement
            pass
        delta = _safe_float(ev.get("max_translation_delta"))
        if delta is None or delta < 0:
            continue
        rid = ev.get("robot")
        if rid is None:
            rid = "server"
        by_robot.setdefault(str(rid), []).append((float(ts or 0.0), float(delta)))
    if not ts_all:
        return {
            "time_from_input_end_to_team_convergence_s": None,
            "time_from_input_end_to_team_convergence_method": "no_timestamps",
            "time_from_input_end_to_team_convergence_per_robot_s": {},
            "time_from_input_end_to_team_convergence_params": {"epsilon": float(epsilon), "required": int(required)},
        }
    t_end = float(max(ts_all))
    input_summary = summarise_input_end(kpi_events)
    t_input_end = input_summary.get("input_end_team_ts")
    if t_input_end is None:
        # fall back to run start
        t_input_end = float(min(ts_all))
    t_input_end = float(t_input_end)
    horizon = max(0.0, t_end - t_input_end)

    per_robot: Dict[str, float] = {}
    censored = False
    for rid, seq in by_robot.items():
        seq = sorted([(t, d) for (t, d) in seq if t >= t_input_end], key=lambda x: x[0])
        if len(seq) < int(required):
            per_robot[str(rid)] = horizon
            censored = True
            continue
        deltas = [d for _t, d in seq]
        stable_dt = None
        for i in range(0, len(deltas) - int(required) + 1):
            window = deltas[i : i + int(required)]
            if all(v <= float(epsilon) for v in window):
                stable_dt = float(seq[i][0]) - t_input_end
                break
        if stable_dt is None:
            per_robot[str(rid)] = horizon
            censored = True
        else:
            per_robot[str(rid)] = max(0.0, float(stable_dt))

    team = max(per_robot.values()) if per_robot else None
    return {
        "time_from_input_end_to_team_convergence_s": team,
        "time_from_input_end_to_team_convergence_method": "censored_end" if censored else "stable_window",
        "time_from_input_end_to_team_convergence_per_robot_s": per_robot,
        "time_from_input_end_to_team_convergence_params": {"epsilon": float(epsilon), "required": int(required)},
    }


def time_from_input_end_to_ddf_stop(kpi_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decentralised-only termination timing: time from input_end(team) until each agent stops DDF."""
    if not kpi_events:
        return {
            "time_from_input_end_to_ddf_stop_team_s": None,
            "time_from_input_end_to_ddf_stop_method": "no_events",
            "time_from_input_end_to_ddf_stop_per_robot_s": {},
        }
    ts_all = [_safe_float(ev.get("ts")) for ev in kpi_events if isinstance(ev, dict)]
    ts_all = [float(t) for t in ts_all if t is not None]
    if not ts_all:
        return {
            "time_from_input_end_to_ddf_stop_team_s": None,
            "time_from_input_end_to_ddf_stop_method": "no_timestamps",
            "time_from_input_end_to_ddf_stop_per_robot_s": {},
        }
    t_end = float(max(ts_all))
    input_summary = summarise_input_end(kpi_events)
    t_input_end = input_summary.get("input_end_team_ts")
    if t_input_end is None:
        t_input_end = float(min(ts_all))
    t_input_end = float(t_input_end)
    horizon = max(0.0, t_end - t_input_end)

    stops: Dict[str, float] = {}
    for ev in kpi_events:
        if not isinstance(ev, dict) or ev.get("event") != "ddf_stop":
            continue
        ts = _safe_float(ev.get("ts"))
        if ts is None:
            continue
        rid = ev.get("robot")
        if rid is None:
            continue
        rid_s = str(rid)
        prev = stops.get(rid_s)
        if prev is None or float(ts) > prev:
            stops[rid_s] = float(ts)

    if not stops:
        return {
            "time_from_input_end_to_ddf_stop_team_s": None,
            "time_from_input_end_to_ddf_stop_method": "no_ddf_stop_event",
            "time_from_input_end_to_ddf_stop_per_robot_s": {},
        }

    # If we have stop events, compute deltas; if some robots are missing stop, right-censor them at horizon.
    per_robot: Dict[str, float] = {}
    censored = False
    robot_ids = sorted({str(ev.get("robot")) for ev in kpi_events if isinstance(ev, dict) and ev.get("robot") is not None})
    for rid in robot_ids:
        if rid in stops:
            per_robot[rid] = max(0.0, float(stops[rid]) - t_input_end)
        else:
            per_robot[rid] = horizon
            censored = True

    if not per_robot and stops:
        # fallback to whatever we have
        per_robot = {rid: max(0.0, ts - t_input_end) for rid, ts in stops.items()}
    team = max(per_robot.values()) if per_robot else None
    return {
        "time_from_input_end_to_ddf_stop_team_s": team,
        "time_from_input_end_to_ddf_stop_method": "censored_end" if censored else "ddf_stop_event",
        "time_from_input_end_to_ddf_stop_per_robot_s": per_robot,
    }


def add_kpi_name_aliases(out: Dict[str, Any]) -> None:
    """
    Add prefixed aliases so readers don't confuse different KPI families:

      - stable_* : stabilisation-window metrics (comparable)
      - event_*  : event-semantic proxies (useful, not strictly comparable)
      - term_*   : termination/stop-condition metrics (architecture-specific)

    Existing keys remain unchanged for backwards compatibility.
    """
    # Shared stabilisation proxy
    if "time_from_input_end_to_team_convergence_s" in out:
        out["stable_team_convergence_s"] = out.get("time_from_input_end_to_team_convergence_s")
        out["stable_team_convergence_method"] = out.get("time_from_input_end_to_team_convergence_method")
        out["stable_team_convergence_per_robot_s"] = out.get("time_from_input_end_to_team_convergence_per_robot_s")
        out["stable_team_convergence_params"] = out.get("time_from_input_end_to_team_convergence_params")

    # Stabilised correction metrics
    if "loop_closure_correction_stabilised" in out:
        out["stable_loop_closure_correction"] = out.get("loop_closure_correction_stabilised")
        out["stable_loop_closure_correction_params"] = out.get("loop_closure_correction_stabilised_params")
    if "interface_correction_stabilised" in out:
        out["stable_interface_correction"] = out.get("interface_correction_stabilised")
        out["stable_interface_correction_params"] = out.get("interface_correction_stabilised_params")

    # Event-semantic proxies
    if "time_to_global_convergence_s" in out:
        out["event_time_to_global_convergence_s"] = out.get("time_to_global_convergence_s")
        out["event_time_to_global_convergence_method"] = out.get("time_to_global_convergence_method")
    if "loop_closure_correction" in out:
        out["event_loop_closure_correction"] = out.get("loop_closure_correction")
    if "interface_correction" in out:
        out["event_interface_correction"] = out.get("interface_correction")

    # Termination/stop conditions
    if "time_from_input_end_to_ddf_stop_team_s" in out:
        out["term_ddf_stop_team_s"] = out.get("time_from_input_end_to_ddf_stop_team_s")
        out["term_ddf_stop_method"] = out.get("time_from_input_end_to_ddf_stop_method")
        out["term_ddf_stop_per_robot_s"] = out.get("time_from_input_end_to_ddf_stop_per_robot_s")


def derive_kpis_for_run(run_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    latency_path = os.path.join(run_dir, "kpi_metrics", "latency_metrics.json")
    lat = _read_json(latency_path)
    kpi_path = os.path.join(run_dir, "kpi_metrics", "kpi_events.jsonl")
    kpi_events = _read_kpi_events(kpi_path)
    robot_map = _load_robot_map_for_run(run_dir)
    if lat:
        out.update(summarise_loop_closure_correction(lat, robot_map=robot_map))
        out.update(summarise_interface_correction(lat))
        if kpi_events:
            out.update(
                summarise_loop_closure_correction_stabilised(
                    lat,
                    kpi_events,
                    robot_map=robot_map,
                    epsilon=DEFAULT_STABLE_EPSILON,
                    required=DEFAULT_STABLE_REQUIRED,
                )
            )
            out.update(
                summarise_interface_correction_stabilised(
                    lat, kpi_events, epsilon=DEFAULT_STABLE_EPSILON, required=DEFAULT_STABLE_REQUIRED
                )
            )
    conv = time_to_global_convergence(run_dir)
    out["time_to_global_convergence_s"] = conv.seconds
    out["time_to_global_convergence_method"] = conv.method
    out["timeline_start_ts"] = conv.start_ts
    out["timeline_end_ts"] = conv.end_ts

    # Per-batch/iteration optimisation time summary
    opt_durations: List[float] = []
    for ev in kpi_events:
        if not isinstance(ev, dict) or ev.get("event") != "optimization_end":
            continue
        if bool(ev.get("settle_only", False)):
            continue
        d = _safe_float(ev.get("duration_s"))
        if d is not None and d >= 0.0:
            opt_durations.append(d)
    out["optimization_duration_s"] = _stats(opt_durations)

    # Communication throughput summaries (using the KPI timeline window when available).
    bw = _read_json(os.path.join(run_dir, "kpi_metrics", "bandwidth_stats.json"))
    if isinstance(bw, dict):
        uplink = bw.get("uplink", {}) if isinstance(bw.get("uplink", {}), dict) else {}
        downlink = bw.get("downlink", {}) if isinstance(bw.get("downlink", {}), dict) else {}

        def _sum(bucket: Dict[str, Any]) -> Tuple[int, int]:
            msgs = 0
            byt = 0
            for _topic, v in bucket.items():
                if not isinstance(v, dict):
                    continue
                try:
                    msgs += int(v.get("messages", 0) or 0)
                except Exception:
                    pass
                try:
                    byt += int(v.get("bytes", 0) or 0)
                except Exception:
                    pass
            return msgs, byt

        up_msgs, up_bytes = _sum(uplink)
        dn_msgs, dn_bytes = _sum(downlink)
        out["communication_uplink_messages"] = int(up_msgs)
        out["communication_uplink_bytes"] = int(up_bytes)
        out["communication_downlink_messages"] = int(dn_msgs)
        out["communication_downlink_bytes"] = int(dn_bytes)
        if conv.start_ts is not None and conv.end_ts is not None and conv.end_ts >= conv.start_ts:
            dt = float(conv.end_ts - conv.start_ts) or 0.0
            if dt > 0.0:
                out["communication_uplink_bytes_per_s"] = float(up_bytes) / dt
                out["communication_downlink_bytes_per_s"] = float(dn_bytes) / dt
    robustness = _load_robustness_metrics(run_dir)
    if robustness:
        out.update(summarise_delivery_rates(robustness))
    est_path = os.path.join(run_dir, "kpi_metrics", "estimation_metrics.json")
    est = _read_json(est_path)
    if est:
        out["estimation_metrics"] = est
    if kpi_events:
        out.update(summarise_input_end(kpi_events))
        out.update(
            time_from_input_end_to_team_convergence(
                kpi_events, epsilon=DEFAULT_STABLE_EPSILON, required=DEFAULT_STABLE_REQUIRED
            )
        )
        out.update(time_from_input_end_to_ddf_stop(kpi_events))
    add_kpi_name_aliases(out)
    return out


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_robot_map_for_run(run_dir: str) -> Optional[Dict[str, str]]:
    res_path = os.path.join(run_dir, "kpi_metrics", "resource_profile.json")
    meta = _read_json(res_path).get("metadata", {})
    jrl = meta.get("jrl") if isinstance(meta, dict) else None
    if not jrl or not os.path.exists(jrl):
        return None
    try:
        from c_slam_common.loader import LoaderConfig, build_key_robot_map, load_jrl

        doc = load_jrl(jrl, LoaderConfig())
        return build_key_robot_map(doc)
    except Exception:
        return None


def _load_robustness_metrics(run_dir: str) -> Dict[str, Any]:
    primary = os.path.join(run_dir, "kpi_metrics", "robustness_metrics.json")
    if os.path.exists(primary):
        return _read_json(primary)
    fallback = os.path.join(run_dir, "kpi_metrics", "robustness_factors.json")
    return _read_json(fallback)


def summarise_delivery_rates(robustness_json: Dict[str, Any]) -> Dict[str, Any]:
    stats = robustness_json.get("stats", {}) if isinstance(robustness_json, dict) else {}
    topics = stats.get("topics", {}) if isinstance(stats, dict) else {}
    rates: List[float] = []
    per_topic: Dict[str, Dict[str, float | None]] = {}
    for topic, data in topics.items():
        if not isinstance(data, dict):
            continue
        attempts = _safe_float(data.get("attempts")) or 0.0
        drops = _safe_float(data.get("drops")) or 0.0
        delivered = _safe_float(data.get("delivered")) or (attempts - drops)
        eta = _safe_float(data.get("delivery_rate"))
        if eta is None and attempts > 0:
            eta = max(0.0, min(1.0, delivered / attempts))
        if eta is not None:
            rates.append(eta)
        per_topic[str(topic)] = {
            "attempts": float(attempts),
            "drops": float(drops),
            "delivered": float(delivered),
            "delivery_rate": eta,
        }
    summary: Dict[str, Any] = {"delivery_topics": per_topic}
    if rates:
        summary["delivery_rate"] = _stats(rates)
    return summary


def extract_interface_correction_times(latency_json: Dict[str, Any]) -> List[float]:
    events = latency_json.get("events") if isinstance(latency_json, dict) else None
    if not isinstance(events, list):
        return []
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") == "BetweenFactorPose3":
            continue
        d = _safe_float(ev.get("latency_ingest_to_broadcast"))
        if d is not None and d >= 0.0:
            out.append(d)
    return out


def summarise_interface_correction(latency_json: Dict[str, Any]) -> Dict[str, Any]:
    vals = extract_interface_correction_times(latency_json)
    return {"interface_correction": _stats(vals)}


def extract_interface_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> List[float]:
    if not isinstance(latency_json, dict):
        return []
    events = latency_json.get("events")
    if not isinstance(events, list):
        return []
    use_ddf = any(ev.get("event") == "ddf_round_delta" for ev in kpi_events or [])
    if use_ddf:
        stab_event = "ddf_round_delta"
        delta_field = "max_translation_delta"
        allow = None
    else:
        stab_event = "optimization_end"
        delta_field = "max_translation_delta"
        allow = {"isam2", "batch", "ddf_sam"}
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") == "BetweenFactorPose3":
            continue
        t_det = _safe_float(ev.get("ingest_wall"))
        if t_det is None:
            continue
        t_stable = _stabilisation_time_after(
            kpi_events,
            start_ts=float(t_det),
            event_name=stab_event,
            delta_field=delta_field,
            solver_allow=allow,
            epsilon=float(epsilon),
            required=int(required),
        )
        if t_stable is None:
            continue
        dt = float(t_stable) - float(t_det)
        if dt >= 0.0:
            out.append(dt)
    return out


def summarise_interface_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    epsilon: float = DEFAULT_STABLE_EPSILON,
    required: int = DEFAULT_STABLE_REQUIRED,
) -> Dict[str, Any]:
    vals = extract_interface_correction_stabilised(latency_json, kpi_events, epsilon=epsilon, required=required)
    return {
        "interface_correction_stabilised": _stats(vals),
        "interface_correction_stabilised_params": {"epsilon": float(epsilon), "required": int(required)},
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Compute derived KPIs for a run directory.")
    ap.add_argument("--run-dir", required=True, help="Path to run output (containing kpi_metrics)")
    ap.add_argument("--out", default=None, help="Where to write derived_kpis.json (default: run_dir/kpi_metrics)")
    args = ap.parse_args()

    derived = derive_kpis_for_run(args.run_dir)
    out_path = args.out or os.path.join(args.run_dir, "kpi_metrics", "derived_kpis.json")
    write_json(out_path, derived)
    print(f"Wrote derived KPIs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
