# KPI Overview (new_eval)

This note documents **which KPIs are produced by the evaluation pipeline**, where they are stored per run, and how to interpret them at a high level. It is intended as a practical index while running `new_eval/tools/run_baseline.py`.

## Where KPIs live (per run)

Each run writes a `kpi_metrics/` folder under its run directory:

- Centralised: `.../<dataset_tag>/centralised/kpi_metrics/`
- Decentralised: `.../<dataset_tag>/decentralised/kpi_metrics/`

The primary “raw” source is:
- `kpi_events.jsonl` (event stream; used for stabilisation + convergence KPIs)

The primary “summary” source is:
- `derived_kpis.json` (computed by `tools/kpi_derive.py`)

## Categories

### Naming convention (recommended)

To avoid mixing up *comparable* stabilisation-based timing with *architecture-specific* event proxies, `derived_kpis.json` includes **prefixed alias keys**:

- `stable_*`: stabilisation-window metrics (intended to be comparable across architectures)
- `event_*`: event-semantic proxies (useful within an architecture; not strictly comparable)
- `term_*`: termination / stop-condition times (architecture-specific but valuable)

The pipeline keeps the original (legacy) keys as well; the prefixed keys are aliases.

### 1) Timing KPIs

**A) Shared (cross-architecture) timing anchors**

These exist in `kpi_metrics/derived_kpis.json`:

- `input_end_team_ts` / `input_end_per_robot_ts` / `input_end_method`
  - Timestamp(s) for “end of input” (replay finished / last factor ingested).
  - Used as the anchor for post-input convergence metrics.

- `time_from_input_end_to_team_convergence_s`
  - Shared stabilisation proxy \(T_{\mathrm{conv}}^{(\mathrm{team})}\) used for cross-architecture comparison.
  - Computed from `optimization_end.max_translation_delta` after `input_end`, using the stabilisation window \((\epsilon, S)\) defined in `Thesis_src/methodology.tex`.
  - Companion fields:
    - `time_from_input_end_to_team_convergence_method` (`stable_window` vs `censored_end`)
    - `time_from_input_end_to_team_convergence_per_robot_s`
    - `time_from_input_end_to_team_convergence_params` (`epsilon`, `required` == \(S\))

Aliases:
- `stable_team_convergence_s` (+ `stable_team_convergence_method`, `stable_team_convergence_per_robot_s`, `stable_team_convergence_params`)

**B) Decentralised-only termination timing**

These exist in `kpi_metrics/derived_kpis.json` for decentralised runs that include `ddf_stop` KPI events:

- `time_from_input_end_to_ddf_stop_team_s`
  - \(T_{\mathrm{stop}}^{(\mathrm{team})}\): time from `input_end(team)` until each agent terminates DDF; team statistic is `max_r`.
  - Companion fields:
    - `time_from_input_end_to_ddf_stop_method` (`ddf_stop_event` vs `censored_end` / `no_ddf_stop_event`)
    - `time_from_input_end_to_ddf_stop_per_robot_s`

Aliases:
- `term_ddf_stop_team_s` (+ `term_ddf_stop_method`, `term_ddf_stop_per_robot_s`)

**C) Architecture-specific timing indicators (reported side-by-side)**

These exist in `kpi_metrics/derived_kpis.json`:

- `time_to_global_convergence_s` / `time_to_global_convergence_method`
  - A run-level “time until stable / last activity” indicator based on the KPI timeline.
  - Method may use a delta-threshold window when available; otherwise it falls back to last map broadcast / last optimisation end.
  - Not guaranteed to be strictly cross-architecture comparable (event semantics differ).

Aliases:
- `event_time_to_global_convergence_s` (+ `event_time_to_global_convergence_method`)

- `loop_closure_correction` (stats dict: `count/min/max/mean/median/p90/p95/p99/stdev`)
  - Loop-closure “ingest → broadcast” latency proxy from `latency_metrics.json` (inter-robot loop closures only).

Aliases:
- `event_loop_closure_correction` (same stats dict)

- `loop_closure_correction_stabilised` + `loop_closure_correction_stabilised_params`
  - Loop-closure “ingest → stable window” time \(T_{\mathrm{LC}}\) using the stabilisation criterion.

Aliases:
- `stable_loop_closure_correction` (+ `stable_loop_closure_correction_params`)

- `interface_correction` (stats dict)
  - Interface message timing proxy from `latency_metrics.json` (decentralised correction propagation indicator).

Aliases:
- `event_interface_correction` (same stats dict)

- `interface_correction_stabilised` + `interface_correction_stabilised_params`
  - Interface “ingest → stable window” time using the stabilisation criterion.

Aliases:
- `stable_interface_correction` (+ `stable_interface_correction_params`)

**D) Optimisation compute timing**

These exist in `kpi_metrics/derived_kpis.json`:

- `optimization_duration_s` (stats dict)
  - Distribution of `optimization_end.duration_s` events.
  - Centralised “settle-only” post-input updates are excluded from this summary.

**E) Timeline window (used for throughput normalisation)**

These exist in `kpi_metrics/derived_kpis.json`:

- `timeline_start_ts`, `timeline_end_ts`
  - Start/end timestamps of the KPI timeline used by the derivation script.
  - `communication_*_bytes_per_s` uses `(timeline_end_ts - timeline_start_ts)` when available.

### 2) Accuracy KPIs

Accuracy KPIs are written into `kpi_metrics/estimation_metrics*.json`:

- Centralised: `kpi_metrics/estimation_metrics.json`
  - `ate.<rid>.rmse` (+ `matches`, and alignment parameters `R`, `t`, `s`)
  - `rpe.<rid>.<k>` where `<k>` are configured step sizes (e.g., `1`, `10`, `50`)

- Decentralised:
  - Per-agent: `kpi_metrics/estimation_metrics_<rid>.json`
  - Aggregated: `kpi_metrics/estimation_metrics.json` (dict keyed by rid)

The derived KPI file also embeds the estimation metrics under:
- `derived_kpis.json: estimation_metrics`

### 3) Communication / traffic KPIs

**A) Raw throughput counters**

These exist in `kpi_metrics/bandwidth_stats.json`:
- `uplink.<topic>.messages`, `uplink.<topic>.bytes`
- `downlink.<topic>.messages`, `downlink.<topic>.bytes` (centralised map downlink; decentralised monitor may not populate downlink)

**B) Summarised throughput KPIs**

These exist in `kpi_metrics/derived_kpis.json`:
- `communication_uplink_messages`, `communication_uplink_bytes`
- `communication_downlink_messages`, `communication_downlink_bytes`
- `communication_uplink_bytes_per_s`, `communication_downlink_bytes_per_s` (normalised by KPI timeline duration when available)

**C) Delivery / robustness KPIs**

These exist in `kpi_metrics/robustness_factors.json` (publisher-side) and `kpi_metrics/robustness_metrics.json` (merged):
- Per-topic: `stats.topics.<topic>.attempts/drops/delivered/delivery_rate/bytes_*`

Derived summary (from the robustness JSON) exists in `kpi_metrics/derived_kpis.json`:
- `delivery_topics.<topic>.(attempts|drops|delivered|delivery_rate)`
- `delivery_rate` (stats dict across topics; present when delivery_rate is computable)

**D) Centralised map delivery**

Centralised runs may also write:
- `kpi_metrics/map_delivery.json` (per-topic delivery counters for map downlink)

**E) Decentralised interface delivery**

Decentralised runs may write:
- `kpi_metrics/iface_delivery_<rid>.json` (delivery counters emitted by each peer-bus instance)

### 4) Resource / efficiency KPIs

Resource traces exist in:
- Centralised: `kpi_metrics/resource_profile.json`
- Decentralised: `kpi_metrics/resource_profile_<rid>.json` and merged `kpi_metrics/resource_profile.json`

Each profile contains:
- `samples[].ts`
- `samples[].cpu_process_pct`
- `samples[].rss_bytes` (RSS)
- `samples[].vms_bytes` (VMS)
- `samples[].cpu_system_pct`
- `samples[].ram_percent`

### 5) Failure-mode / debug artefacts (for “decentralised crashes” analysis)

These are not “KPIs” but are required evaluation artefacts:

- Orchestrator logs and return codes (per run dir):
  - `solver.log`
  - `publisher.log`
  - `orchestrate_returncodes.json`

- Decentralised per-agent logs (in `kpi_metrics/`):
  - `agent_<rid>.log`
  - `agent_error_<rid>.txt` (written on crash, includes traceback)
  - `agent_exitcodes.json` (written when any agent exits non-zero)

## Notes on stabilisation parameters

The stabilisation window parameters \((\epsilon, S)\) are documented in `Thesis_src/methodology.tex` and shared in code via `c_slam_common/stabilisation.py`.
