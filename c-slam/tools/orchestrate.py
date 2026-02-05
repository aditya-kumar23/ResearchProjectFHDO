#!/usr/bin/env python3
"""Minimal orchestrator: spawn solver (centralised/decentralised) + ROS2 factor publisher."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent


def _bool_flag(enabled: bool, flag: str) -> List[str]:
    return [flag] if enabled else []


def _resolve_path_like(path_like: object) -> str | None:
    if path_like is None:
        return None
    s = str(path_like).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return str(p.resolve())


def _apply_impair_env(base_env: Dict[str, str], cfg: Dict[str, object], *, role: str) -> Dict[str, str]:
    """Apply impairment env vars to a process environment.

    - role: "solver" or "publisher"
    - cfg expects optional:
        impair: {
          enabled: bool,
          apply_to: ["solver", "publisher"],
          spec_file: "tools/impair.example.json",
          spec_json: "{...}"
        }
    """
    env = dict(base_env)
    impair_cfg = cfg.get("impair", {}) or {}
    if not isinstance(impair_cfg, dict):
        impair_cfg = {}

    enabled = bool(impair_cfg.get("enabled", False))
    apply_to = impair_cfg.get("apply_to")
    if isinstance(apply_to, list):
        apply_to_set = {str(x) for x in apply_to}
    else:
        apply_to_set = {"solver", "publisher"}

    should_apply = enabled and (role in apply_to_set)
    if not should_apply:
        env.pop("C_SLAM_IMPAIR", None)
        env.pop("C_SLAM_IMPAIR_FILE", None)
        return env

    spec_file = _resolve_path_like(impair_cfg.get("spec_file"))
    spec_json = impair_cfg.get("spec_json")
    if spec_file:
        env["C_SLAM_IMPAIR_FILE"] = spec_file
        env.pop("C_SLAM_IMPAIR", None)
    elif spec_json is not None and str(spec_json).strip():
        env["C_SLAM_IMPAIR"] = str(spec_json)
        env.pop("C_SLAM_IMPAIR_FILE", None)
    else:
        # Enabled but no spec provided => keep off.
        env.pop("C_SLAM_IMPAIR", None)
        env.pop("C_SLAM_IMPAIR_FILE", None)
    return env


def _parse_cpu_affinity(obj: object) -> List[int] | None:
    """Parse a list of CPU indices from config; return None if not provided/invalid."""
    if obj is None:
        return None
    cores = []
    seq = obj if isinstance(obj, (list, tuple)) else [obj]
    for c in seq:
        try:
            cores.append(int(c))
        except Exception:
            continue
    return cores or None


def _apply_affinity(pid: int, cores: List[int] | None) -> None:
    """Set CPU affinity for a pid; best-effort, ignore if unsupported."""
    if cores is None:
        return
    try:
        os.sched_setaffinity(pid, set(cores))
    except Exception:
        pass


def build_commands(cfg: Dict[str, object]) -> tuple[List[str], List[str]]:
    dataset_raw = Path(cfg["dataset"]).expanduser()
    dataset = (dataset_raw if dataset_raw.is_absolute() else (ROOT / dataset_raw)).resolve()
    export_raw = Path(cfg.get("export_path", "out")).expanduser()
    export_path = (export_raw if export_raw.is_absolute() else (ROOT / export_raw)).resolve()
    backend = str(cfg.get("backend", "centralised"))
    factor_topic_prefix = str(cfg.get("factor_topic_prefix", "/c_slam/factor_batch"))
    iface_topic_prefix = str(cfg.get("iface_topic_prefix", "/c_slam/iface"))
    use_sim_time = bool(cfg.get("use_sim_time", False))
    enable_kpi = bool(cfg.get("kpi", True))
    quat_order = str(cfg.get("quat_order", "wxyz"))

    solver_cfg = cfg.get("solver", {}) or {}
    publisher_cfg = cfg.get("publisher", {}) or {}
    qos_cfg = cfg.get("qos", {}) or {}

    qos_reliability = str(
        solver_cfg.get(
            "qos_reliability",
            qos_cfg.get("reliability", publisher_cfg.get("qos_reliability", cfg.get("qos_reliability", "reliable"))),
        )
    )
    qos_durability = str(
        solver_cfg.get(
            "qos_durability",
            qos_cfg.get("durability", publisher_cfg.get("qos_durability", cfg.get("qos_durability", "volatile"))),
        )
    )
    qos_depth = int(
        solver_cfg.get(
            "qos_depth",
            qos_cfg.get("depth", publisher_cfg.get("qos_depth", cfg.get("qos_depth", 10))),
        )
    )

    solver_cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--backend",
        backend,
        "--jrl",
        str(dataset),
        "--export-path",
        str(export_path),
        "--factor-topic-prefix",
        factor_topic_prefix,
    ]
    solver_cmd += ["--iface-topic-prefix", iface_topic_prefix]
    if enable_kpi:
        solver_cmd += ["--kpi", "--kpi-dir", str(export_path / "kpi_metrics")]
    solver_cmd += ["--quat-order", str(solver_cfg.get("quat_order", cfg.get("quat_order", quat_order)))]
    solver_cmd += ["--robust", str(solver_cfg.get("robust", cfg.get("robust", "cauchy")))]
    solver_cmd += ["--robust-k", str(solver_cfg.get("robust_k", cfg.get("robust_k", 1.0)))]
    solver_cmd += ["--qos-reliability", qos_reliability, "--qos-durability", qos_durability, "--qos-depth", str(qos_depth)]
    solver_cmd += ["--relin-th", str(solver_cfg.get("relin_th", 0.05))]
    solver_cmd += ["--relin-skip", str(solver_cfg.get("relin_skip", 5))]
    solver_cmd += ["--ddf-rounds", str(solver_cfg.get("ddf_rounds", 8))]
    solver_cmd += ["--ddf-convergence", str(solver_cfg.get("ddf_convergence", 5e-3))]
    solver_cmd += ["--ddf-rot-convergence", str(solver_cfg.get("ddf_rot_convergence", 5e-3))]
    solver_cmd += ["--ddf-factor-idle-timeout", str(solver_cfg.get("ddf_factor_idle_timeout", 5.0))]
    solver_cmd += ["--ddf-stable-rounds", str(solver_cfg.get("ddf_stable_rounds", 3))]
    solver_cmd += ["--central-batch-size", str(solver_cfg.get("central_batch_size", 1))]
    solver_cmd += ["--central-idle-timeout", str(solver_cfg.get("central_idle_timeout", 5.0))]
    post_input_settle = solver_cfg.get("post_input_settle_updates", cfg.get("post_input_settle_updates", None))
    if post_input_settle is not None:
        try:
            solver_cmd += ["--post-input-settle-updates", str(int(post_input_settle))]
        except Exception:
            pass
    if bool(solver_cfg.get("include_potential_outliers", cfg.get("include_potential_outliers", False))):
        solver_cmd.append("--include-potential-outliers")
    emit_acks = bool(solver_cfg.get("emit_factor_acks", cfg.get("emit_factor_acks", enable_kpi)))
    solver_cmd += _bool_flag(emit_acks, "--emit-factor-acks")
    emit_map = bool(solver_cfg.get("emit_map_downlink", cfg.get("emit_map_downlink", False)))
    solver_cmd += _bool_flag(emit_map, "--emit-map-downlink")
    map_prefix = str(solver_cfg.get("map_topic_prefix", cfg.get("map_topic_prefix", "/c_slam/map")))
    solver_cmd += ["--map-topic-prefix", map_prefix]
    solver_cmd += _bool_flag(use_sim_time, "--use-sim-time")

    publisher_cmd = [
        sys.executable,
        str(ROOT / "tools" / "ros2_factor_publisher.py"),
        "--jrl",
        str(dataset),
        "--topic-prefix",
        factor_topic_prefix,
        "--batch-size",
        str(publisher_cfg.get("batch_size", 1)),
        "--time-scale",
        str(publisher_cfg.get("time_scale", 0.0)),
        "--max-sleep",
        str(publisher_cfg.get("max_sleep", 0.0)),
        "--idle-gap",
        str(publisher_cfg.get("idle_gap", 1.0)),
        "--loop",
        str(publisher_cfg.get("loop", 1)),
        "--wait-subscriber-timeout",
        str(publisher_cfg.get("wait_subscriber_timeout", 10.0)),
        "--qos-reliability",
        str(publisher_cfg.get("qos_reliability", qos_reliability)),
        "--qos-durability",
        str(publisher_cfg.get("qos_durability", qos_durability)),
        "--qos-depth",
        str(publisher_cfg.get("qos_depth", qos_depth)),
    ]
    if bool(publisher_cfg.get("include_potential_outliers", cfg.get("include_potential_outliers", False))):
        publisher_cmd.append("--include-potential-outliers")
    publisher_cmd += _bool_flag(use_sim_time, "--use-sim-time")
    publisher_cmd += ["--quat-order", str(publisher_cfg.get("quat_order", cfg.get("quat_order", quat_order)))]
    if enable_kpi:
        metrics_out = publisher_cfg.get("metrics_out") or str(export_path / "kpi_metrics" / "robustness_factors.json")
        publisher_cmd += ["--metrics-out", str(metrics_out), "--ack-wait", str(cfg.get("ack_wait", cfg.get("shutdown_grace", 2.0)))]

    return solver_cmd, publisher_cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal c-slam orchestrator (ROS2 factor publisher + solver).")
    ap.add_argument("--config", required=True, help="Path to JSON config (see tools/orchestrate.example.json)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    solver_cmd, publisher_cmd = build_commands(cfg)

    print("Solver cmd:", " ".join(map(str, solver_cmd)))
    print("Publisher cmd:", " ".join(map(str, publisher_cmd)))
    if args.dry_run:
        return 0
    try:
        export_raw = Path(cfg.get("export_path", "out")).expanduser()
        export_path = (export_raw if export_raw.is_absolute() else (ROOT / export_raw)).resolve()
        export_path.mkdir(parents=True, exist_ok=True)
        with open(export_path / "orchestrate_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        with open(export_path / "orchestrate_commands.json", "w", encoding="utf-8") as f:
            json.dump({"solver_cmd": solver_cmd, "publisher_cmd": publisher_cmd}, f, indent=2)
    except Exception:
        pass

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    if cfg.get("use_sim_time", False):
        env["C_SLAM_USE_SIM_TIME"] = "1"
    if cfg.get("kpi", True):
        env["C_SLAM_EXPECT_FACTOR_METRICS"] = "1"

    procs = []
    interrupted = False
    need_terminate = False
    solver_proc: subprocess.Popen | None = None
    publisher_proc: subprocess.Popen | None = None
    solver_log_fh = None
    publisher_log_fh = None
    try:
        # Start solver first so subscribers are ready
        solver_env = _apply_impair_env(env, cfg, role="solver")
        try:
            solver_log_fh = open(export_path / "solver.log", "w", encoding="utf-8", buffering=1)
        except Exception:
            solver_log_fh = None
        solver_proc = subprocess.Popen(
            solver_cmd,
            cwd=ROOT,
            env=solver_env,
            stdout=solver_log_fh or None,
            stderr=subprocess.STDOUT if solver_log_fh else None,
            text=bool(solver_log_fh),
        )
        solver_aff = _parse_cpu_affinity(cfg.get("solver", {}).get("cpu_affinity"))
        _apply_affinity(solver_proc.pid, solver_aff)
        procs.append(solver_proc)
        time.sleep(float(cfg.get("startup_delay", 1.0)))

        publisher_env = _apply_impair_env(env, cfg, role="publisher")
        try:
            publisher_log_fh = open(export_path / "publisher.log", "w", encoding="utf-8", buffering=1)
        except Exception:
            publisher_log_fh = None
        publisher_proc = subprocess.Popen(
            publisher_cmd,
            cwd=ROOT,
            env=publisher_env,
            stdout=publisher_log_fh or None,
            stderr=subprocess.STDOUT if publisher_log_fh else None,
            text=bool(publisher_log_fh),
        )
        publisher_aff = _parse_cpu_affinity(cfg.get("publisher", {}).get("cpu_affinity"))
        _apply_affinity(publisher_proc.pid, publisher_aff)
        procs.append(publisher_proc)

        # Wait for publisher; solver may exit after idle timeout
        publisher_proc.wait()
        # Give solver time to finish after DONE messages
        time.sleep(float(cfg.get("shutdown_grace", 2.0)))

        if solver_proc is not None and solver_proc.poll() is None:
            # Wait for solver to exit on its own; optional timeout to avoid hanging forever
            solver_wait = float(cfg.get("solver_wait_timeout", 0.0))
            try:
                if solver_wait > 0.0:
                    solver_proc.wait(timeout=solver_wait)
                else:
                    solver_proc.wait()
            except subprocess.TimeoutExpired:
                need_terminate = True
    except KeyboardInterrupt:
        interrupted = True
        need_terminate = True
    finally:
        for p in procs:
            if p.poll() is None and (interrupted or need_terminate):
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        try:
            rc_path = export_path / "orchestrate_returncodes.json"
            payload = {
                "solver_returncode": solver_proc.returncode if solver_proc is not None else None,
                "publisher_returncode": publisher_proc.returncode if publisher_proc is not None else None,
                "interrupted": bool(interrupted),
            }
            with open(rc_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass
        try:
            if solver_log_fh is not None:
                solver_log_fh.close()
        except Exception:
            pass
        try:
            if publisher_log_fh is not None:
                publisher_log_fh.close()
        except Exception:
            pass
    solver_rc = solver_proc.returncode if solver_proc is not None else None
    pub_rc = publisher_proc.returncode if publisher_proc is not None else None
    if interrupted:
        # Conventional shell code for Ctrl-C.
        return 130
    if isinstance(solver_rc, int) and solver_rc != 0:
        return int(solver_rc)
    if isinstance(pub_rc, int) and pub_rc != 0:
        return int(pub_rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
