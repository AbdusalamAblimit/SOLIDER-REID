"""Measure matched D0/H2-M model-only compute, memory, and throughput."""

import argparse
import importlib.util
import json
from pathlib import Path


D0_CONFIG = "configs/occluded_duke/swin_tiny_tapf_d0.yml"
H2M_CONFIG = "configs/occluded_duke/swin_tiny_tapf_h2m.yml"
BASE_SCRIPT = Path("experiments/exp387_clean_tapf_d0/preflight_efficiency.py")


def load_base_module():
    spec = importlib.util.spec_from_file_location("exp387_efficiency", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load matched efficiency base script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_arm(
    base,
    config_path,
    train_batch,
    eval_images,
    train_warmup,
    train_steps,
    eval_warmup,
    eval_steps,
):
    base.D0_CONFIG = config_path
    config = base.make_cfg(True)
    return {
        "config": config_path,
        "config_sha256": base.sha256_file(config_path),
        "train": base.train_benchmark(
            config, True, train_batch, train_warmup, train_steps
        ),
        "eval_rgb_only": base.eval_benchmark(
            config, True, eval_images, eval_warmup, eval_steps
        ),
    }


def numeric_delta(right, left, keys):
    return {key: right[key] - left[key] for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--d0-config", default=D0_CONFIG)
    parser.add_argument("--h2m-config", default=H2M_CONFIG)
    parser.add_argument("--dataset-root")
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--eval-warmup", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=20)
    args = parser.parse_args()

    base = load_base_module()
    base.DATASET_ROOT = args.dataset_root
    base.SEMANTIC_WEIGHT = args.semantic_weight
    base.D0_CONFIG = args.h2m_config
    train_batch, eval_images = base.load_fixed_real_batches()

    d0 = audit_arm(
        base,
        args.d0_config,
        train_batch,
        eval_images,
        args.train_warmup,
        args.train_steps,
        args.eval_warmup,
        args.eval_steps,
    )
    h2m = audit_arm(
        base,
        args.h2m_config,
        train_batch,
        eval_images,
        args.train_warmup,
        args.train_steps,
        args.eval_warmup,
        args.eval_steps,
    )

    d0_eval = d0["eval_rgb_only"]
    h2m_eval = h2m["eval_rgb_only"]
    if d0_eval["parameters"] != 28179484:
        raise RuntimeError("Unexpected D0 parameter count")
    if h2m_eval["parameters"] != 28287102:
        raise RuntimeError("Unexpected H2-M parameter count")
    parameter_delta = h2m_eval["parameters"] - d0_eval["parameters"]
    flop_delta = h2m_eval["supported_op_flops"] - d0_eval["supported_op_flops"]
    if parameter_delta != 107618:
        raise RuntimeError("Unexpected H2-M parameter delta")
    if flop_delta != 39351552:
        raise RuntimeError("Unexpected H2-M supported-op FLOP delta")

    result = {
        "status": "EXP391_H2M_MATCHED_EFFICIENCY_PASS",
        "torch_version": base.torch.__version__,
        "cuda_version": base.torch.version.cuda,
        "device": base.torch.cuda.get_device_name(0),
        "script_sha256": base.sha256_file(__file__),
        "base_script": str(BASE_SCRIPT),
        "base_script_sha256": base.sha256_file(BASE_SCRIPT),
        "measurement_scope": {
            "train": "one fixed real paired batch; forward+backward+SGD, AMP scale=1",
            "eval": "one fixed real RGB-only validation batch; forward, FP32",
            "flops": "MMEngine supported-operator trace; unsupported ops excluded",
        },
        "fixed_batch_sha256": {
            "train_rgb": base.tensor_sha256(train_batch[0]),
            "train_pid": base.tensor_sha256(train_batch[1]),
            "train_pose_keypoints": base.tensor_sha256(train_batch[4]["keypoints"]),
            "eval_rgb": base.tensor_sha256(eval_images),
        },
        "d0": d0,
        "h2m": h2m,
        "delta_h2m_minus_d0": {
            "parameters": parameter_delta,
            "parameter_percent": 100.0 * parameter_delta / d0_eval["parameters"],
            "supported_op_flops": flop_delta,
            "supported_op_flops_percent": (
                100.0 * flop_delta / d0_eval["supported_op_flops"]
            ),
            "train": numeric_delta(
                h2m["train"],
                d0["train"],
                ("mean_step_ms", "peak_allocated_bytes", "peak_reserved_bytes"),
            ),
            "eval_rgb_only": numeric_delta(
                h2m_eval,
                d0_eval,
                ("mean_step_ms", "peak_allocated_bytes", "peak_reserved_bytes"),
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print("parameter_delta={}".format(parameter_delta))
    print("supported_op_flops_delta={}".format(flop_delta))
    print(
        "train_delta={}".format(
            json.dumps(result["delta_h2m_minus_d0"]["train"], sort_keys=True)
        )
    )
    print(
        "eval_delta={}".format(
            json.dumps(
                result["delta_h2m_minus_d0"]["eval_rgb_only"], sort_keys=True
            )
        )
    )
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
