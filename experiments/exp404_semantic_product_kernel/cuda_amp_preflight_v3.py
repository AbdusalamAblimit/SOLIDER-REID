#!/usr/bin/env python3
"""Fresh v3 default-GradScaler steady-state wrapper for exp404 SPK."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


BASE = Path(__file__).with_name("cuda_amp_preflight.py")
spec = importlib.util.spec_from_file_location("exp404_cuda_preflight_v1_core", BASE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.__file__ = str(Path(__file__).resolve())
original_write_json = core.write_json


def write_v3_json(path, payload):
    payload["execution"] = "exp404_cuda_amp_preflight_v3"
    payload["production_contract"] = "joint_field_v3"
    payload["amp_contract"] = "default_gradscaler_natural_backoff_max8"
    original_write_json(path, payload)


core.write_json = write_v3_json


if __name__ == "__main__":
    if "--max-attempts" in sys.argv:
        raise RuntimeError("V3 max attempts are frozen by the wrapper")
    sys.argv.extend(["--max-attempts", "8"])
    core.main()
