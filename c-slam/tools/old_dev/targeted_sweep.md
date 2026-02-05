# Targeted Sweep Runner

This script runs the methodology-aligned targeted sweeps (QoS, impairments, scalability) using the baseline template.

## Quick start

```bash
python3 tools/targeted_sweep.py
```

Defaults:
- Uses `tools/sweep_baseline.template.json`
- Includes baseline runs for QoS/impair datasets
- Runs `qos`, `impair`, and `scale` sweeps
- Outputs to `out/sweep_<timestamp>/`

## Common options

```bash
python3 tools/targeted_sweep.py --dry-run
python3 tools/targeted_sweep.py --sweeps qos impair
python3 tools/targeted_sweep.py --no-baseline
python3 tools/targeted_sweep.py --modalities wifi
python3 tools/targeted_sweep.py --include-day-night
python3 tools/targeted_sweep.py --baseline-all
python3 tools/targeted_sweep.py --qos-impair-datasets dataset/proradio/r3_proradio.jrl dataset/wifi/r3_wifi.jrl
python3 tools/targeted_sweep.py --bw-caps-mbps 0.5 1.0 2.0
python3 tools/targeted_sweep.py --rep-seeds 0 1 2 3 4
python3 tools/targeted_sweep.py --out-root out/sweep_runs
python3 tools/targeted_sweep.py --no-resume
python3 tools/targeted_sweep.py --quiet
python3 tools/targeted_sweep.py --ros-domain-base 30 --ros-domain-span 200
python3 tools/targeted_sweep.py --no-ros-daemon-restart
```

## Outputs

- `out/sweep_<timestamp>/run_plan.json` — list of planned runs
- `out/sweep_<timestamp>/runs/<run_id>/` — per-run artifacts
- `out/sweep_<timestamp>/results.csv` — aggregated metrics table

## Baseline template

Edit `tools/sweep_baseline.template.json` to adjust the locked invariants:
- solver/publisher pacing
- solver iteration settings
- CPU affinity
- map downlink enabled for centralised runs

The sweep runner will reject any run that changes locked fields outside the template.
