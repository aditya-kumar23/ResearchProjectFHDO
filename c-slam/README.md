# c-slam (minimal)

Minimal standalone extraction of:
- Centralised iSAM2 backend (ROS 2 factor streaming)
- Decentralised DDF-style backend (ROS 2 multiprocess + per-agent factor streaming)

ROS 2 is required in this repo.

## Install (editable)

From `c-slam/`:

`python3 -m pip install -e .`

You also need:
- ROS 2 Python (`rclpy`) available in your environment (source your ROS 2 setup).
- `gtsam` Python bindings available (your existing install).

## Demo run (ROS 2 factor streaming)

Terminal A (publish factors from a `.jrl`):

`python3 tools/ros2_factor_publisher.py --jrl /path/to/dataset.jrl --topic-prefix /c_slam/factor_batch`

Terminal B (centralised iSAM2):

`python3 main.py --backend centralised --jrl /path/to/dataset.jrl --export-path out --factor-topic-prefix /c_slam/factor_batch`

To place central map updates on the network (subject to QoS/impairment), add `--emit-map-downlink --map-topic-prefix /c_slam/map`.

Terminal C (decentralised iSAM2 + DDF multiprocess):

`python3 main.py --backend decentralised --jrl /path/to/dataset.jrl --export-path out --factor-topic-prefix /c_slam/factor_batch --iface-topic-prefix /c_slam/iface`

Outputs:
- `out/trajectories/trajectory_<robot>.csv`

## One-shot orchestrator (optional)

`python3 tools/orchestrate.py --config tools/orchestrate.example.json`

This spawns the solver (centralised/decentralised per config) and the ROS2 factor publisher concurrently. Edit the JSON to switch backend or dataset path.
- Optional CPU caps: set `solver.cpu_affinity` / `publisher.cpu_affinity` (list of cores) in the config to bound total compute used by each run.
- Optional map downlink on centralised runs: set `solver.emit_map_downlink=true` and `solver.map_topic_prefix`.

## KPI metrics (bandwidth/latency/resource/derived)

Add `--kpi` (optionally `--kpi-dir <path>`) to `main.py` or set `"kpi": true` in `tools/orchestrate.example.json`. Exports go to `<export-path>/kpi_metrics` by default:
- `kpi_events.jsonl` (KPILogger timeline)
- `latency_metrics.json`, `bandwidth_stats.json`, `resource_profile.json`
- `estimation_metrics.json` (ATE/RPE when ground truth is present)
- `derived_kpis.json` (post-processed KPIs)
- `map_delivery.json` (central map downlink send stats when `--emit-map-downlink` is used)

Also written to `<export-path>/run_manifest.json` (inputs + solver knobs + ROS2 QoS + environment snapshot).

## Fair comparison checklist (centralised vs decentralised)

To keep comparisons grounded, ensure:
- Same dataset + factor set: identical `.jrl`, same `--quat-order`, same `--include-potential-outliers` setting.
- Same robustification: same `--robust` and `--robust-k` for both runs.
- Same ROS2 stream configuration: same topic prefixes + QoS (`--qos-*`) + publisher batching/pacing. If you use impairment (`C_SLAM_IMPAIR*`), record and keep it identical.
- Same compute budget: pin to the same CPU cores (e.g., `cpu_affinity` in `tools/orchestrate.example.json`) or state the quota explicitly.
- If you want central downlink on the wire, use `--emit-map-downlink` (or `solver.emit_map_downlink` in the orchestrator) so comms costs are comparable.
- Same “run done” definition: factor publisher sends `DONE_ALL`; solvers stop ingest when DONE is observed. Centralised then finishes the final iSAM2 update; decentralised then runs up to `--ddf-rounds` (or earlier if stable under `--ddf-*` thresholds).
- Same compute budget criterion: pick one and state it explicitly (recommended: report both (1) fixed-data run: process full dataset to DONE, and (2) fixed-time run: cap wall-clock/runtime externally and report accuracy vs time).

## Metric definitions (what the repo exports)

- Accuracy: `kpi_metrics/estimation_metrics.json` exports per-robot ATE-RMSE (Umeyama alignment on matched keys) + RPE translation RMSE over fixed key windows.
- Performance: `kpi_metrics/derived_kpis.json` includes `time_to_global_convergence_s` and `optimization_duration_s` (per-batch/per-round).
- Communication: `kpi_metrics/bandwidth_stats.json` contains uplink/downlink message counts + bytes. Delivery rate is exported in `kpi_metrics/robustness_metrics.json` when using the orchestrator (publisher writes factor delivery stats; decentralised agents export interface delivery stats).
- Resource: `kpi_metrics/resource_profile.json` samples CPU/RAM (and GPU if available) at a fixed interval.

## Network impairment (packet loss / delays / “agent loss”)

Both the factor publisher and the decentralised interface bus support synthetic impairments via:
- `C_SLAM_IMPAIR` (inline JSON string) or `C_SLAM_IMPAIR_FILE` (path to JSON).

Implemented impairments (`c_slam_ros2/impair.py`):
- Packet loss: `random_loss_p`, periodic `burst_*`, and scheduled `blackouts`.
- Packet loss warm-up: `random_warmup_messages` (per-sender) disables *random* drops for the first N sends to avoid immediate underconstraint.
- Delays: token-bucket bandwidth caps `bw_caps_mbps` (per-sender), which add sleep before send.
- Agent loss (communication drop-out): `blackouts` with `mode: "sender"` (agent cannot send) and/or `mode: "receiver"` (agent cannot receive). `mode: "either"` affects both directions.
Note: `burst_duration_s` must be smaller than `burst_period_s` to avoid dropping every message in perpetuity.

Example: `C_SLAM_IMPAIR_FILE=c-slam/tools/impair.example.json`.

## Sweep evaluation (post-processing)

If you have a sweep directory that contains `results.csv` + `run_plan.json` (e.g. from `tools/targeted_sweep.py`), generate a compact evaluation report + tables/plots via:

`python3 tools/evaluate_sweep.py --sweep-dir out/sweep_small`

Outputs are written to `<sweep-dir>/eval/` (notably `eval/report.md` and `eval/eval_runs.csv`).
