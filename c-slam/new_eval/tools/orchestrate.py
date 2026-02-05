#!/usr/bin/env python3
"""
Wrapper around repo-level tools/orchestrate.py.

Keep the same CLI and behavior, but allow `new_eval/tools` to be self-contained.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    target = (ROOT / "tools" / "orchestrate.py").resolve()
    if not target.exists():
        raise SystemExit(f"Expected orchestrator at {target}")
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

