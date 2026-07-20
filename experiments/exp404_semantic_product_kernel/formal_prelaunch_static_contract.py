#!/usr/bin/env python3
"""Final static gate before the unique fresh exp404 e120 launch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch


DIRECTORY = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Prelaunch static output must be fresh")

    config = DIRECTORY / "swin_tiny_spk_formal.yml"
    preflight = DIRECTORY / "cuda_amp_preflight_v3_result.json"
    wrapper = DIRECTORY / "formal_once_wrapper.sh"
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    wrapper_source = wrapper.read_text(encoding="utf-8")
    config_source = config.read_text(encoding="utf-8")
    ast.parse(
        (DIRECTORY / "cuda_amp_preflight_v3.py").read_text(encoding="utf-8")
    )
    gates = {
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "config_sha_exact": sha256_file(config)
        == "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d",
        "preflight_sha_exact": sha256_file(preflight)
        == "70566973f0387d0b335040ff20fe2c1f091563cc18f4a65370b25aac303d58bf",
        "preflight_26_of_26_pass": preflight_payload["status"]
        == "CUDA_AMP_PREFLIGHT_PASS"
        and preflight_payload["gate_pass_count"]
        == preflight_payload["gate_count"]
        == 26,
        "preflight_authorizes_formal": preflight_payload[
            "formal_training_authorized"
        ],
        "preflight_execution_v3": preflight_payload["execution"]
        == "exp404_cuda_amp_preflight_v3",
        "formal_recipe_frozen": all(
            token in config_source
            for token in (
                "MAX_EPOCHS: 120",
                "IMS_PER_BATCH: 64",
                "CHECKPOINT_PERIOD: 120",
                "SEED: 1234",
                "SPK_ENABLED: True",
                "ELO_CUR_ENABLED: False",
            )
        ),
        "fresh_once_only_guards": all(
            token in wrapper_source
            for token in (
                'test ! -e "$OUTPUT"',
                'test ! -e "$RUNNER"',
                'test ! -e "$LAUNCH"',
                'test ! -e "$LOCK"',
                'mkdir "$LOCK"',
            )
        ),
        "exclusive_gpu_guard": "nvidia-smi --query-compute-apps=pid"
        in wrapper_source,
        "remote_clean_guard": 'test -z "$(git status --short)"'
        in wrapper_source,
        "source_hash_guards": all(
            sha in wrapper_source
            for sha in (
                "72ff5a609c7a080d848e96a2c12239795388441cc13b85519ef2cbf42f04bf2a",
                "44de28f34b675366606e4ae4734567f50c6ede755fd85280073c514543d61f76",
                "bc98121ab179e44f091ef6e7cabf9f75b6e2cfa3390ccba930d1324553a4beb1",
            )
        ),
        "fresh_runtime_guard": (
            "/home/afr/reid-clean/runtimes/exp404-spk-py310/bin/python"
            in wrapper_source
            and "exp394-openclip-reid-py310/bin/python" not in wrapper_source
        ),
        "no_resume": '"resume": false' in wrapper_source
        and "--resume" not in wrapper_source,
        "background_runner": "nohup" in wrapper_source
        and 'kill -0 "$PID"' in wrapper_source,
        "no_local_launch_artifacts": not any(
            (DIRECTORY / name).exists()
            for name in (
                "formal_train_v1.runner.log",
                "formal_train_v1.launch.json",
                "formal_train_v1.launch.lock",
            )
        ),
    }
    passed = all(gates.values())
    result = {
        "experiment": "exp404_semantic_product_kernel",
        "execution": "exp404_formal_train_v1",
        "status": "FORMAL_PRELAUNCH_STATIC_PASS" if passed else "FORMAL_PRELAUNCH_STATIC_FAIL",
        "formal_launch_authorized": passed,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "source_sha256": {
            "config": sha256_file(config),
            "preflight": sha256_file(preflight),
            "wrapper": sha256_file(wrapper),
            "contract": sha256_file(Path(__file__)),
        },
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
