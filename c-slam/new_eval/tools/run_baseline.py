#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_GLOB = ",".join(
    [
        "dataset/wifi/r3_*.jrl",
        "dataset/wifi/r4_*.jrl",
        "dataset/wifi/r5_*.jrl",
        "dataset/proradio/r3_*.jrl",
        "dataset/proradio/r4_*.jrl",
        "dataset/proradio/r5_*.jrl",
    ]
)


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    tag: str
    robot_ids: Tuple[str, ...]
    stamp_span: Optional[float]


@dataclass(frozen=True)
class Failure:
    dataset: str
    backend: str
    run_dir: str
    returncode: int


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object at {path}")
    return obj


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _resolve_repo_path(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def _dataset_tag(dataset_path: Path) -> str:
    modality = dataset_path.parent.name
    stem = dataset_path.stem
    return _slug(f"{modality}-{stem}")


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hash_dict(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _compute_ros_domain_id(config_hash: str, base: int, span: int) -> int:
    try:
        h = int(config_hash[:8], 16)
    except Exception:
        h = abs(hash(config_hash))
    return int(base) + (h % int(span))


def _restart_ros_daemon() -> None:
    try:
        subprocess.run(["ros2", "daemon", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass


def _deep_get(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _deep_set(obj: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur = obj
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _apply_overrides(out[key], value)
        else:
            out[key] = value
    return out


def _diff_paths(a: Any, b: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[Tuple[str, ...]] = []
        keys = set(a.keys()) | set(b.keys())
        for key in sorted(keys):
            if key not in a or key not in b:
                out.append(prefix + (str(key),))
                continue
            out.extend(_diff_paths(a[key], b[key], prefix + (str(key),)))
        return out
    if a != b:
        return [prefix]
    return []


def _require_locked_paths(template: Dict[str, Any], candidate: Dict[str, Any], allowed_prefixes: Sequence[Tuple[str, ...]]) -> None:
    changed = _diff_paths(template, candidate)
    for path in changed:
        if not any(path[: len(prefix)] == prefix for prefix in allowed_prefixes):
            raise SystemExit(f"Locked field changed: {'.'.join(path)}")


def _parse_globs(dataset_glob: str) -> List[str]:
    raw = str(dataset_glob).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _estimate_stamp_span(measurements: Any) -> Optional[float]:
    if not isinstance(measurements, dict):
        return None
    min_s: Optional[float] = None
    max_s: Optional[float] = None
    for entries in measurements.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            stamp = entry.get("stamp")
            if not isinstance(stamp, (int, float)):
                continue
            stamp_val = float(stamp)
            min_s = stamp_val if min_s is None else min(min_s, stamp_val)
            max_s = stamp_val if max_s is None else max(max_s, stamp_val)
    if min_s is None or max_s is None:
        return None
    return max(0.0, max_s - min_s)


def _discover_datasets(dataset_glob: str) -> List[DatasetInfo]:
    patterns = _parse_globs(dataset_glob)
    if not patterns:
        patterns = _parse_globs(DEFAULT_DATASET_GLOB)
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted((ROOT).glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    out: List[DatasetInfo] = []
    for path in unique:
        doc: Dict[str, Any] = {}
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {}
        measurements = doc.get("measurements", {}) if isinstance(doc, dict) else {}
        robot_ids = sorted(str(key) for key in measurements.keys()) if isinstance(measurements, dict) else []
        out.append(
            DatasetInfo(
                path=path,
                tag=_dataset_tag(path),
                robot_ids=tuple(robot_ids),
                stamp_span=_estimate_stamp_span(measurements),
            )
        )
    return out


def _parse_cores(value: str) -> Optional[List[int]]:
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "null", "off"}:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    cores = []
    for p in parts:
        try:
            cores.append(int(p))
        except Exception:
            continue
    return cores or None


def _qos_grid() -> List[Dict[str, Any]]:
    return [
        {"reliability": "best_effort", "durability": "transient_local", "depth": 10},
        {"reliability": "best_effort", "durability": "transient_local", "depth": 50},
    ]


def _apply_qos(cfg: Dict[str, Any], qos: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _apply_overrides(cfg, {"qos": dict(qos)})
    _deep_set(cfg, ["solver", "qos_reliability"], qos["reliability"])
    _deep_set(cfg, ["solver", "qos_durability"], qos["durability"])
    _deep_set(cfg, ["solver", "qos_depth"], int(qos["depth"]))
    _deep_set(cfg, ["publisher", "qos_reliability"], qos["reliability"])
    _deep_set(cfg, ["publisher", "qos_durability"], qos["durability"])
    _deep_set(cfg, ["publisher", "qos_depth"], int(qos["depth"]))
    return cfg


def _estimate_duration_s(dataset: DatasetInfo, template: Dict[str, Any]) -> Optional[float]:
    span = dataset.stamp_span
    if span is None:
        return None
    time_scale = _deep_get(template, ["publisher", "time_scale"])
    if not isinstance(time_scale, (int, float)) or time_scale <= 0:
        return None
    return float(span) / float(time_scale)


def _clamp(val: float, lo: float, hi: float) -> float:
    if hi < lo:
        return lo
    return max(lo, min(hi, val))


def _blackout_windows(duration_s: Optional[float]) -> List[Tuple[float, float]]:
    if duration_s is None or duration_s <= 0:
        return [(8.0, 12.0), (20.0, 24.0)]
    length = min(8.0, max(4.0, 0.05 * duration_s))
    if duration_s < (length * 2 + 4.0):
        length = max(2.0, (duration_s - 4.0) / 2.0) if duration_s > 4.0 else max(1.0, duration_s / 4.0)
    max_start = max(1.0, duration_s - length - 1.0)
    s1 = _clamp(0.15 * duration_s, 5.0, max_start)
    s2 = _clamp(0.60 * duration_s, s1 + length + 2.0, max_start)
    if s2 <= s1:
        s2 = min(max_start, s1 + length + 2.0)
    return [(s1, s1 + length), (s2, s2 + length)]


def _impair_bwcap_scenarios(caps_mbps: Sequence[float]) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for cap in caps_mbps:
        scenarios.append(
            {
                "name": f"bwcap_{str(cap).replace('.', 'p')}mbps",
                "spec": {"bw_caps_mbps": {"default": float(cap)}},
            }
        )
    return scenarios


def _impair_blackout_scenario(rid: str, duration_s: Optional[float]) -> Dict[str, Any]:
    windows = _blackout_windows(duration_s)
    blackouts = [
        {"rid": str(rid), "start_s": float(start), "end_s": float(end), "mode": "sender"}
        for start, end in windows
    ]
    return {"name": "blackout_2x", "spec": {"blackouts": blackouts}}


def _run_orchestrator(cmd: List[str], *, timeout_s: Optional[float], env: Dict[str, str]) -> int:
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout_s, env=env)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        return 124


def _parse_mode(argv: Optional[Sequence[str]]) -> Tuple[str, List[str]]:
    args = list(argv or [])
    if args and args[0] in {"baseline", "qos", "impair"}:
        return args[0], args[1:]
    return "baseline", args


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--baseline",
        default=str(ROOT / "new_eval" / "tools" / "orchestrate.baseline.json"),
        help="Path to baseline config JSON (template).",
    )
    ap.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help=f"Dataset glob(s), comma-separated. Default: {DEFAULT_DATASET_GLOB}",
    )
    ap.add_argument(
        "--export-root",
        default=str(ROOT / "new_eval" / "out"),
        help="Export root directory (default: new_eval/out).",
    )
    ap.add_argument(
        "--run-id",
        default="auto",
        help="Run id folder name. Use 'auto' for timestamp (YYYYmmdd_HHMMSS).",
    )
    ap.add_argument(
        "--backends",
        default="centralised,decentralised",
        help="Comma-separated backends to run (default: centralised,decentralised).",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs per dataset (default: 1).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only (no execution)")
    ap.add_argument(
        "--cores",
        default="2,3,4,5",
        help="CPU cores for solver/publisher affinity (comma-separated). Use 'none' to keep template.",
    )
    ap.add_argument("--nice", type=int, default=None, help="Run orchestrator with `nice -n` (optional).")
    ap.add_argument("--ros-domain-base", type=int, default=30, help="ROS_DOMAIN_ID base for isolation.")
    ap.add_argument("--ros-domain-span", type=int, default=200, help="ROS_DOMAIN_ID span for isolation.")
    ap.add_argument("--no-ros-daemon-restart", action="store_true", help="Do not stop ROS 2 daemon between runs.")
    return ap


def _prepare_run_root(export_root: Path, run_id_cfg: str, *, mode: str) -> Path:
    if run_id_cfg.lower() in {"", "auto"}:
        run_id = f"{_slug(mode)}_{_now_run_id()}"
    else:
        run_id = _slug(run_id_cfg)
    run_root = (export_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _run_plan_meta(
    *,
    sweep_type: str,
    dataset: DatasetInfo,
    backend: str,
    repeat: int,
    qos: Optional[Dict[str, Any]] = None,
    impair_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    meta = {
        "sweep_type": sweep_type,
        "dataset_tag": dataset.tag,
        "dataset_path": str(dataset.path),
        "backend": backend,
        "repeat": repeat,
    }
    if qos is not None:
        meta["qos_reliability"] = qos.get("reliability")
        meta["qos_durability"] = qos.get("durability")
        meta["qos_depth"] = qos.get("depth")
    if impair_name is not None:
        meta["impair_name"] = impair_name
    if run_dir is not None:
        meta["run_dir"] = str(run_dir.relative_to(ROOT))
    return meta


def _run(
    *,
    mode: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    ap = _build_common_parser(
        "Run orchestrator baseline/QoS/impair sweeps using new_eval/tools/orchestrate.baseline.json"
    )
    if mode == "impair":
        ap.add_argument(
            "--bw-caps-mbps",
            nargs="*",
            type=float,
            default=[3.0, 2.0, 1.0, 0.25],
            help="Bandwidth caps (Mbps) for impairment sweep (default: 3.0 2.0 1.0 0.25).",
        )
        ap.add_argument(
            "--blackout-rid",
            default="auto",
            help="Robot id to blackout (default: auto -> last robot id in dataset).",
        )
    args = ap.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    baseline_path = _resolve_repo_path(str(args.baseline))
    template = _read_json(baseline_path)

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    if not backends:
        raise SystemExit("--backends must be a non-empty comma-separated list")

    export_root = _resolve_repo_path(str(args.export_root))
    run_root = _prepare_run_root(export_root, str(args.run_id).strip(), mode=mode)

    datasets = _discover_datasets(str(args.dataset_glob))
    if not datasets:
        raise SystemExit(f"No datasets matched: {args.dataset_glob}")

    cores = _parse_cores(str(args.cores))
    if cores:
        _deep_set(template, ["solver", "cpu_affinity"], cores)
        _deep_set(template, ["publisher", "cpu_affinity"], cores)

    resolved = {
        "baseline": str(baseline_path),
        "dataset_glob": str(args.dataset_glob),
        "datasets": [str(p.path.relative_to(ROOT)) for p in datasets],
        "backends": backends,
        "repeats": int(args.repeats),
        "export_root": str(export_root),
        "run_id": str(run_root.name),
        "mode": mode,
        "cores": cores,
        "ros_domain_base": int(args.ros_domain_base),
        "ros_domain_span": int(args.ros_domain_span),
        "stop_ros_daemon": not bool(args.no_ros_daemon_restart),
    }
    _write_json(run_root / "orchestration.resolved.json", resolved)

    orchestrate_py = (ROOT / "new_eval" / "tools" / "orchestrate.py").resolve()
    if not orchestrate_py.exists():
        raise SystemExit(f"Expected wrapper at {orchestrate_py}")

    allowed_prefixes = [
        ("dataset",),
        ("backend",),
        ("export_path",),
        ("qos",),
        ("impair",),
        ("solver", "qos_reliability"),
        ("solver", "qos_durability"),
        ("solver", "qos_depth"),
        ("publisher", "qos_reliability"),
        ("publisher", "qos_durability"),
        ("publisher", "qos_depth"),
        ("solver", "cpu_affinity"),
        ("publisher", "cpu_affinity"),
    ]

    run_plan: List[Dict[str, Any]] = []
    failures: List[Failure] = []
    job_i = 0

    def iter_variants(dataset: DatasetInfo) -> Iterable[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        if mode == "baseline":
            yield ("baseline", None, None)
            return
        if mode == "qos":
            for qos in _qos_grid():
                name = f"{qos['reliability']}_tl_d{qos['depth']}"
                yield (name, qos, None)
            return
        if mode == "impair":
            duration_s = _estimate_duration_s(dataset, template)
            bw_caps = _impair_bwcap_scenarios(args.bw_caps_mbps)
            rid = str(dataset.robot_ids[-1]) if dataset.robot_ids else "a"
            if str(args.blackout_rid).strip().lower() not in {"", "auto"}:
                rid = str(args.blackout_rid).strip()
            scenarios = bw_caps + [_impair_blackout_scenario(rid, duration_s)]
            for scenario in scenarios:
                yield (scenario["name"], None, scenario["spec"])
            return

    total_jobs = len(datasets) * len(backends) * args.repeats
    if mode == "qos":
        total_jobs *= len(_qos_grid())
    if mode == "impair":
        total_jobs *= len(args.bw_caps_mbps) + 1

    for dataset in datasets:
        dataset_abs = dataset.path.resolve()
        dataset_rel = dataset_abs.relative_to(ROOT) if str(dataset_abs).startswith(str(ROOT)) else dataset_abs
        base_tag = dataset.tag
        for r in range(1, args.repeats + 1):
            scenario = base_tag if args.repeats == 1 else f"{base_tag}__r{r:02d}"
            for backend in backends:
                for variant_name, qos, impair_spec in iter_variants(dataset):
                    job_i += 1
                    if mode == "baseline":
                        run_dir = (run_root / scenario / _slug(backend)).resolve()
                    else:
                        run_dir = (run_root / scenario / _slug(backend) / mode / variant_name).resolve()
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_cfg: Dict[str, Any] = _apply_overrides(
                        template,
                        {
                            "dataset": str(dataset_abs),
                            "backend": str(backend),
                            "export_path": str(run_dir),
                        },
                    )

                    if qos is not None:
                        run_cfg = _apply_qos(run_cfg, qos)
                        run_cfg = _apply_overrides(run_cfg, {"impair": {"enabled": False}})

                    if impair_spec is not None:
                        impair_cfg = {
                            "enabled": True,
                            "apply_to": ["publisher"],
                            "spec_file": None,
                            "spec_json": json.dumps(impair_spec),
                        }
                        run_cfg = _apply_overrides(run_cfg, {"impair": impair_cfg})

                    _require_locked_paths(template, run_cfg, allowed_prefixes)

                    cfg_out = run_dir / "orchestrate_generated.json"
                    _write_json(cfg_out, run_cfg)

                    config_hash = _hash_dict(run_cfg)
                    ros_domain_id = _compute_ros_domain_id(config_hash, int(args.ros_domain_base), int(args.ros_domain_span))
                    base_env = os.environ.copy()
                    base_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
                    base_env["PYTHONPATH"] = str(ROOT) + (
                        ":" + base_env["PYTHONPATH"] if "PYTHONPATH" in base_env else ""
                    )
                    cmd = [sys.executable, str(orchestrate_py), "--config", str(cfg_out)]
                    if args.dry_run:
                        cmd.append("--dry-run")
                    if args.nice is not None:
                        cmd = ["nice", "-n", str(int(args.nice)), *cmd]

                    meta = _run_plan_meta(
                        sweep_type=mode,
                        dataset=dataset,
                        backend=str(backend),
                        repeat=r,
                        qos=qos,
                        impair_name=variant_name if mode == "impair" else None,
                        run_dir=run_dir,
                    )
                    meta["config_hash"] = config_hash
                    meta["ros_domain_id"] = ros_domain_id
                    meta["variant"] = variant_name
                    run_plan.append(meta)

                    print(
                        f"[{job_i}/{total_jobs}] dataset={dataset_rel} backend={backend} run={r}/{args.repeats} "
                        f"variant={variant_name}"
                    )
                    print("[cmd]", " ".join(map(str, cmd)))
                    if args.dry_run:
                        continue

                    rc = _run_orchestrator(cmd, timeout_s=None, env=base_env)
                    if not args.no_ros_daemon_restart:
                        _restart_ros_daemon()
                    if rc != 0:
                        failures.append(
                            Failure(
                                dataset=str(dataset_rel),
                                backend=str(backend),
                                run_dir=str(run_dir.relative_to(ROOT)),
                                returncode=int(rc),
                            )
                        )

    _write_json(run_root / "run_plan.json", {"count": len(run_plan), "runs": run_plan})

    if failures:
        print(f"[done] failures={len(failures)}")
        for f in failures:
            print(f"  - dataset={f.dataset} backend={f.backend} rc={f.returncode} dir={f.run_dir}")
        return 1

    print("[done] all runs completed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    mode, rest = _parse_mode(args)
    return _run(mode=mode, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
