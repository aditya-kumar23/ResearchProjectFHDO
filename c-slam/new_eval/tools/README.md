# new_eval/tools

This folder mirrors the repo orchestrator and adds a baseline runner.

## One-shot orchestrator (wrapper)

Same behavior as `tools/orchestrate.py`:

`python3 new_eval/tools/orchestrate.py --config <path/to/config.json>`

## Baseline config (single template)

All sweeps use the single template:
- `new_eval/tools/orchestrate.baseline.json`

The runner generates per-run configs and only overrides:
- `dataset`
- `backend`
- `export_path`
- QoS fields (for QoS sweep)
- `impair` fields (for impairment sweep)
- CPU affinity (if `--cores` is provided)

## Unified runner (baseline / QoS / impair)

The runner keeps the original baseline behavior, and adds `qos` and `impair` modes:

```bash
python3 new_eval/tools/run_baseline.py --repeats 1
python3 new_eval/tools/run_baseline.py baseline --repeats 1
python3 new_eval/tools/run_baseline.py qos --repeats 1
python3 new_eval/tools/run_baseline.py impair --repeats 1
```

Defaults:
- Dataset filter: comma-separated globs covering `dataset/wifi/r{3,4,5}_*.jrl` + `dataset/proradio/r{3,4,5}_*.jrl`
- Backends: `centralised,decentralised`
- CPU affinity: `--cores 2,3,4,5`
- ROS isolation: unique `ROS_DOMAIN_ID` per run + stop ROS2 daemon between runs

Outputs (default run id = `<mode>_<timestamp>`):
- Baseline: `new_eval/out/<run_id>/<dataset_tag>/<backend>/...`
- QoS: `new_eval/out/<run_id>/<dataset_tag>/<backend>/qos/<variant>/...`
- Impair: `new_eval/out/<run_id>/<dataset_tag>/<backend>/impair/<variant>/...`

## Evaluate baseline outputs

`python3 tools/old_dev/evaluate_baseline.py --baseline-dir new_eval/out/<run_id>`
