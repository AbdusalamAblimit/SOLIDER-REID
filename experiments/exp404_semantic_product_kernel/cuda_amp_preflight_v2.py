#!/usr/bin/env python3
"""Fresh v2 execution wrapper after the exp404 joint-field production fix."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name("cuda_amp_preflight.py")
spec = importlib.util.spec_from_file_location("exp404_cuda_preflight_v1_core", BASE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.__file__ = str(Path(__file__).resolve())
original_write_json = core.write_json


def write_v2_json(path, payload):
    payload["execution"] = "exp404_cuda_amp_preflight_v2"
    payload["production_contract"] = "joint_field_v3"
    original_write_json(path, payload)


core.write_json = write_v2_json


if __name__ == "__main__":
    core.main()
