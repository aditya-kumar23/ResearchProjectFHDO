from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from typing import Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

logger = logging.getLogger("c_slam.resource")


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    out: Dict[str, float] = {
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.pstdev(vals)
    return out


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


def _percentile_block(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    return {
        "p50": _percentile(vals, 50.0) or 0.0,
        "p90": _percentile(vals, 90.0) or 0.0,
        "p95": _percentile(vals, 95.0) or 0.0,
        "p99": _percentile(vals, 99.0) or 0.0,
    }


class ResourceMonitor:
    def __init__(self, interval: float | None = None, enable_gpu: bool = True):
        if interval is None:
            env_interval = os.environ.get("C_SLAM_RESOURCE_INTERVAL")
            if env_interval:
                try:
                    interval = float(env_interval)
                except Exception:
                    interval = None
        if interval is None or interval <= 0.0:
            interval = 0.5
        self.interval = float(interval)
        self.enable_gpu = enable_gpu
        self._samples: List[Dict[str, float]] = []
        self._metadata: Dict[str, object] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = None
        self._proc_by_pid: Dict[int, object] = {}
        self._aggregate_mode = False
        self._system_cpu_ready = False
        self._gpu_handle = None
        self._gpu_ready = False
        self._gpu_enabled = False
        self._started = False

    def start(self, metadata: Optional[Dict[str, object]] = None) -> None:
        if self._started:
            return
        if metadata:
            self._metadata.update(metadata)
        if psutil is None:
            logger.warning("psutil not available; resource monitoring disabled.")
            self._started = True
            return
        self._process = psutil.Process(os.getpid())
        try:
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
            self._system_cpu_ready = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cpu priming failed: %s", exc)
        if self.enable_gpu and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_ready = True
                self._gpu_enabled = True
            except Exception as exc:  # pragma: no cover - optional path
                logger.debug("GPU monitoring disabled: %s", exc)
                self._gpu_handle = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        self._started = True

    def set_pids(self, pids: List[int]) -> None:
        """Track resource usage across the provided process IDs (aggregate)."""
        if psutil is None:
            return
        self._proc_by_pid = {}
        for pid in pids or []:
            try:
                proc = psutil.Process(int(pid))
                try:
                    proc.cpu_percent(None)
                except Exception:
                    pass
                self._proc_by_pid[int(pid)] = proc
            except Exception:
                continue
        self._aggregate_mode = bool(self._proc_by_pid)

    def _run(self) -> None:
        if self._process is None:
            return
        while not self._stop.wait(self.interval):
            try:
                sample = self._collect_sample()
                if sample:
                    self._samples.append(sample)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("resource sample failed: %s", exc)

    def _collect_sample(self) -> Dict[str, float]:
        assert self._process is not None
        ts = time.time()
        sample: Dict[str, float] = {"ts": ts}
        if self._aggregate_mode and self._proc_by_pid:
            cpu_total = 0.0
            rss_total = 0.0
            vms_total = 0.0
            dead: List[int] = []
            for pid, proc in list(self._proc_by_pid.items()):
                try:
                    cpu_total += float(proc.cpu_percent(None))
                    mem = proc.memory_info()
                    rss_total += float(mem.rss)
                    vms_total += float(mem.vms)
                except Exception:
                    dead.append(pid)
            for pid in dead:
                self._proc_by_pid.pop(pid, None)
            sample["cpu_process_pct"] = cpu_total
            sample["rss_bytes"] = rss_total
            sample["vms_bytes"] = vms_total
        else:
            try:
                sample["cpu_process_pct"] = float(self._process.cpu_percent(None))
            except Exception as exc:  # pragma: no cover
                logger.debug("cpu sample failed: %s", exc)
            try:
                mem = self._process.memory_info()
                sample["rss_bytes"] = float(mem.rss)
                sample["vms_bytes"] = float(mem.vms)
            except Exception as exc:  # pragma: no cover
                logger.debug("memory sample failed: %s", exc)
        if self._system_cpu_ready:
            try:
                sample["cpu_system_pct"] = float(psutil.cpu_percent(None))
            except Exception as exc:  # pragma: no cover
                logger.debug("system cpu sample failed: %s", exc)
        try:
            sample["ram_percent"] = float(psutil.virtual_memory().percent)
        except Exception as exc:  # pragma: no cover
            logger.debug("ram sample failed: %s", exc)
        if self._gpu_ready and self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                sample["gpu_util_pct"] = float(util.gpu)
                sample["gpu_mem_used_bytes"] = float(mem.used)
                sample["gpu_mem_total_bytes"] = float(mem.total)
            except Exception as exc:  # pragma: no cover
                logger.debug("gpu sample failed: %s", exc)
        return sample

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval * 2)
        if self._gpu_enabled and pynvml is not None:  # pragma: no cover - optional
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._started = False

    def update_metadata(self, **entries: object) -> None:
        self._metadata.update(entries)

    @property
    def samples(self) -> List[Dict[str, float]]:
        return list(self._samples)

    def summary(self) -> Dict[str, object]:
        cpu = [s.get("cpu_process_pct", 0.0) for s in self._samples if "cpu_process_pct" in s]
        rss = [s.get("rss_bytes", 0.0) for s in self._samples if "rss_bytes" in s]
        sys_cpu = [s.get("cpu_system_pct", 0.0) for s in self._samples if "cpu_system_pct" in s]
        gpu_util = [s.get("gpu_util_pct", 0.0) for s in self._samples if "gpu_util_pct" in s]
        summary: Dict[str, object] = {
            "num_samples": len(self._samples),
            "cpu_process_pct": {**_stats(cpu), **_percentile_block(cpu)} if cpu else {},
            "cpu_system_pct": {**_stats(sys_cpu), **_percentile_block(sys_cpu)} if sys_cpu else {},
            "rss_bytes": {**_stats(rss)} if rss else {},
        }
        if gpu_util:
            summary["gpu_util_pct"] = {**_stats(gpu_util), **_percentile_block(gpu_util)}
        return summary

    def export_json(self, path: str) -> None:
        data = {
            "metadata": self._metadata,
            "summary": self.summary(),
            "samples": self._samples,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_summary(self, log: logging.Logger) -> None:
        summary = self.summary()
        if not summary.get("num_samples"):
            log.info("ResourceMonitor: no samples recorded.")
            return
        cpu = summary.get("cpu_process_pct", {})
        rss = summary.get("rss_bytes", {})
        log.info(
            "Resource usage: cpu mean=%.2f%% max=%.2f%% | rss max=%.2f MiB",
            cpu.get("mean", 0.0),
            cpu.get("max", 0.0),
            rss.get("max", 0.0) / (1024 * 1024) if rss else 0.0,
        )
        if "gpu_util_pct" in summary:
            gpu = summary["gpu_util_pct"]
            log.info(
                "GPU util: mean=%.2f%% max=%.2f%%",
                gpu.get("mean", 0.0),
                gpu.get("max", 0.0),
            )
