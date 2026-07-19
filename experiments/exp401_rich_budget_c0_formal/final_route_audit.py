#!/usr/bin/env python3
"""Read-only final checkpoint audit and serial full/all-bypass retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
CONFIG_SHA256 = "c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84"
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CODEBOOK_SHA256 = "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
CLIP_PATH = Path(
    "/home/afr/reid-clean/weights/"
    "exp401_clip_l14_openclip_9ce2e8a8.safetensors"
)
CODEBOOK_PATH = Path(
    "/home/afr/reid-clean/formal/exp401_rich_budget_c0/"
    "exp401_phase0e_full_codebook.json"
)
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml":
        "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor checkpoint entry: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        # reshape first because checkpoint buffers can be zero-dimensional.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_finite(value: torch.Tensor) -> bool:
    if value.is_floating_point() or value.is_complex():
        return bool(torch.isfinite(value).all())
    return True


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


class ExplodingPose(dict):
    accesses = 0

    def __getitem__(self, key):
        type(self).accesses += 1
        raise RuntimeError("RGB-only evaluation accessed external pose")

    def get(self, key, default=None):
        type(self).accesses += 1
        raise RuntimeError("RGB-only evaluation accessed external pose")

    def __iter__(self):
        type(self).accesses += 1
        raise RuntimeError("RGB-only evaluation iterated external pose")


def eval_descriptor(model, batch, device, pose_batch):
    image, _, _, camids, target_view, _ = batch
    image = image.to(device)
    camids = camids.to(device)
    target_view = target_view.to(device)
    model.eval()
    with torch.no_grad():
        descriptor, _ = model(
            image,
            cam_label=camids,
            view_label=target_view,
            pose_batch=pose_batch,
        )
    return descriptor.detach().clone()


def evaluate(model, loader, num_query, cfg, device):
    from utils.metrics import R1_mAP_eval

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
    )
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for image, pid, camid, camids, target_view, _ in loader:
            image = image.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            descriptor, _ = model(
                image,
                cam_label=camids,
                view_label=target_view,
            )
            if not bool(torch.isfinite(descriptor).all()):
                raise RuntimeError("Non-finite retrieval descriptor")
            evaluator.update((descriptor, pid, camid))
    cmc, mean_ap, *_ = evaluator.compute()
    return {
        "mAP": float(mean_ap),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
    }


def rounded_percent_matches(value: float, expected_percent: float) -> bool:
    return round(100.0 * value, 1) == float(expected_percent)


def run(args):
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    result_path = Path(args.result).resolve()
    output_dir = Path(args.output_dir).resolve()
    script_path = Path(__file__).resolve()
    sys.path.insert(0, str(repo_root))

    expected_checkpoint = output_dir / "transformer_120.pth"
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError("Checkpoint/output pairing contract failed")
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise RuntimeError("Final checkpoint must be one regular non-symlink file")
    checkpoint_files_before = sorted(
        str(path.resolve())
        for path in output_dir.glob("*")
        if path.is_file() and path.suffix in {".pth", ".pt", ".ckpt"}
    )
    if checkpoint_files_before != [str(checkpoint_path)]:
        raise RuntimeError("Final checkpoint uniqueness contract failed")

    source_sha_before = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    checkpoint_sha_before = sha256_file(checkpoint_path)
    config_sha_before = sha256_file(config_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("Checkpoint is not a nonempty model state dict")
    checkpoint_state_sha = tensor_state_sha256(payload)
    state_names = tuple(payload)
    forbidden = {"teacher", "clip", "codebook", "text", "pose_batch"}
    cpu_gates = {
        "repo_head_exact": run_text(["git", "rev-parse", "HEAD"], repo_root)
        == SOURCE_COMMIT,
        "repo_tracked_clean": not bool(
            run_text(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                repo_root,
            )
        ),
        "repo_all_clean": not bool(
            run_text(["git", "status", "--porcelain"], repo_root)
        ),
        "source_sha_exact": source_sha_before == EXPECTED_SOURCE_SHA256,
        "config_sha_exact": config_sha_before == CONFIG_SHA256,
        "checkpoint_unique": checkpoint_files_before == [str(checkpoint_path)],
        "checkpoint_state_nonempty": bool(payload),
        "checkpoint_state_all_tensors": all(
            torch.is_tensor(value) for value in payload.values()
        ),
        "checkpoint_state_finite": all(
            tensor_finite(value) for value in payload.values()
        ),
        "state_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden)
            for name in state_names
        ),
        "evidence_head_retained": any(
            "anchor.evidence_head" in name for name in state_names
        ),
        "two_routers_retained": all(
            any(f"psg_bank.{index}.evidence_projection" in name for name in state_names)
            for index in (0, 1)
        ),
        "rho_not_in_state": all("rho" not in name.lower() for name in state_names),
    }
    if not all(cpu_gates.values()):
        raise RuntimeError(f"CPU checkpoint gates failed: {cpu_gates}")

    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    clip_path = Path(cfg.MODEL.TAPF.CLIP_CHECKPOINT).resolve()
    codebook_path = Path(cfg.MODEL.TAPF.RICH_CODEBOOK).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID).strip("()'")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    (
        _,
        _,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    incompatible = model.load_state_dict(payload, strict=True)
    model_state_sha_loaded = tensor_state_sha256(model.state_dict())
    model = model.to(device)
    tapf = model.base.tapf
    strict_gates = {
        "strict_load_no_missing": not incompatible.missing_keys,
        "strict_load_no_unexpected": not incompatible.unexpected_keys,
        "strict_state_exact": model_state_sha_loaded == checkpoint_state_sha,
        "tapf_rich_exact": bool(tapf.rich_evidence and tapf.semantic),
        "rho_python_float": type(tapf.rho_star) is float,
        "rho_not_parameter": all(
            "rho" not in name.lower() for name, _ in model.named_parameters()
        ),
        "rho_not_buffer": all(
            "rho" not in name.lower() for name, _ in model.named_buffers()
        ),
        "two_router_modules": len(tapf.psg_bank) == 2,
        "evidence_head_module": hasattr(tapf.anchor, "evidence_head"),
        "config_output_dir_exact": Path(cfg.OUTPUT_DIR).resolve() == output_dir,
        "official_data_root_exact": str(cfg.DATASETS.ROOT_DIR).strip("()'")
        == "/mnt1/afrdata",
        "frozen_pose_root_exact": str(cfg.MODEL.TAPF.ARTIFACT_DIR)
        == "/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train",
        "model_object_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden)
            for name, _ in model.named_modules()
        ),
        "clip_asset_regular_exact": (
            clip_path == CLIP_PATH
            and clip_path.is_file()
            and not clip_path.is_symlink()
            and str(cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256) == CLIP_SHA256
            and sha256_file(clip_path) == CLIP_SHA256
        ),
        "codebook_asset_regular_exact": (
            codebook_path == CODEBOOK_PATH
            and codebook_path.is_file()
            and not codebook_path.is_symlink()
            and str(cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256) == CODEBOOK_SHA256
            and sha256_file(codebook_path) == CODEBOOK_SHA256
        ),
    }
    if not all(strict_gates.values()):
        raise RuntimeError(f"Strict model gates failed: {strict_gates}")

    first_batch = next(iter(val_loader))
    descriptor_none = eval_descriptor(model, first_batch, device, None)
    ExplodingPose.accesses = 0
    descriptor_exploding = eval_descriptor(
        model, first_batch, device, ExplodingPose()
    )
    rgb_gates = {
        "rgb_none_exploding_exact": torch.equal(
            descriptor_none, descriptor_exploding
        ),
        "exploding_pose_access_zero": ExplodingPose.accesses == 0,
        "rgb_descriptor_finite": bool(torch.isfinite(descriptor_none).all()),
    }
    if not all(rgb_gates.values()):
        raise RuntimeError(f"RGB-only gates failed: {rgb_gates}")

    model_state_sha_before_retrieval = tensor_state_sha256(model.state_dict())
    full = evaluate(model, val_loader, num_query, cfg, device)
    full_matches_logged = all(
        rounded_percent_matches(full[name], expected)
        for name, expected in {
            "mAP": args.expected_full_map,
            "rank1": args.expected_full_rank1,
            "rank5": args.expected_full_rank5,
            "rank10": args.expected_full_rank10,
        }.items()
    )
    if not full_matches_logged:
        raise RuntimeError(f"Full retrieval does not match logged e120: {full}")

    original_apply_gate = tapf.apply_gate
    had_override = "apply_gate" in tapf.__dict__
    override_before = tapf.__dict__.get("apply_gate")
    bypass_calls = [0, 0]

    def all_bypass(bank_index, tokens, hw_shape, state):
        if bank_index not in (0, 1):
            raise RuntimeError(f"Unexpected router bank: {bank_index}")
        bypass_calls[bank_index] += 1
        state["gate_deltas"].append(torch.zeros_like(tokens))
        return tokens

    try:
        tapf.apply_gate = all_bypass
        all_bypass_metrics = evaluate(
            model, val_loader, num_query, cfg, device
        )
    finally:
        if had_override:
            tapf.__dict__["apply_gate"] = override_before
        else:
            tapf.__dict__.pop("apply_gate", None)

    apply_gate_restored = (
        ("apply_gate" in tapf.__dict__) == had_override
        and (
            not had_override
            or tapf.__dict__.get("apply_gate") is override_before
        )
    )
    model_state_sha_after_retrieval = tensor_state_sha256(model.state_dict())
    checkpoint_sha_after = sha256_file(checkpoint_path)
    config_sha_after = sha256_file(config_path)
    source_sha_after = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    checkpoint_files_after = sorted(
        str(path.resolve())
        for path in output_dir.glob("*")
        if path.is_file() and path.suffix in {".pth", ".pt", ".ckpt"}
    )
    map_delta = full["mAP"] - all_bypass_metrics["mAP"]
    terminal_gates = {
        "full_recheck_matches_e120_logged": full_matches_logged,
        "both_router_banks_bypassed": bypass_calls
        == [len(val_loader), len(val_loader)],
        "apply_gate_restored_exact": apply_gate_restored,
        "model_state_before_after_exact": (
            model_state_sha_before_retrieval == model_state_sha_after_retrieval
        ),
        "checkpoint_before_after_exact": checkpoint_sha_before
        == checkpoint_sha_after,
        "config_before_after_exact": config_sha_before == config_sha_after,
        "source_before_after_exact": source_sha_before == source_sha_after,
        "checkpoint_list_before_after_exact": checkpoint_files_before
        == checkpoint_files_after,
        "full_metrics_finite": all(
            torch.isfinite(torch.tensor(value)) for value in full.values()
        ),
        "bypass_metrics_finite": all(
            torch.isfinite(torch.tensor(value))
            for value in all_bypass_metrics.values()
        ),
    }
    audit_pass = all(cpu_gates.values()) and all(strict_gates.values())
    audit_pass = audit_pass and all(rgb_gates.values())
    audit_pass = audit_pass and all(terminal_gates.values())
    route_alive = (
        audit_pass
        and full["mAP"] >= args.minimum_full_map / 100.0
        and map_delta >= args.minimum_delta_map / 100.0
    )
    result = {
        "status": "PASS" if audit_pass else "FAIL",
        "decision": (
            "RICH_BUDGET_ROUTE_ALIVE"
            if route_alive
            else "RICH_BUDGET_ROUTE_ALIVE_FAIL"
        ),
        "phase_b_interface_authorized": bool(route_alive),
        "cpu_gates": cpu_gates,
        "strict_gates": strict_gates,
        "rgb_only_gates": rgb_gates,
        "terminal_gates": terminal_gates,
        "metrics": {
            "full": full,
            "all_router_bypass": all_bypass_metrics,
            "full_minus_all_bypass_mAP": map_delta,
            "minimum_full_mAP": args.minimum_full_map / 100.0,
            "minimum_delta_mAP": args.minimum_delta_map / 100.0,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "state_sha256": checkpoint_state_sha,
            "state_names_count": len(state_names),
        },
        "execution": {
            "repo_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
            "config_sha256": config_sha_after,
            "script_sha256": sha256_file(script_path),
            "source_sha256": source_sha_after,
            "bypass_calls": bypass_calls,
            "checkpoint_files": checkpoint_files_after,
        },
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit_pass:
        raise RuntimeError("Final route audit failed")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-full-map", type=float, default=57.1)
    parser.add_argument("--expected-full-rank1", type=float, default=67.3)
    parser.add_argument("--expected-full-rank5", type=float, default=80.3)
    parser.add_argument("--expected-full-rank10", type=float, default=84.8)
    parser.add_argument("--minimum-full-map", type=float, default=56.7)
    parser.add_argument("--minimum-delta-map", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result).resolve()
    result_temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    if result_path.exists() or result_temporary.exists():
        raise FileExistsError("Final audit result path must be fresh")
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "status": "FAIL",
            "decision": "AUDIT_RUNTIME_FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(result_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
