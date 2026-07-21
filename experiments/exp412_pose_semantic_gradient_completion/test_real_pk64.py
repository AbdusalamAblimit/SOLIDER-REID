#!/usr/bin/env python3
"""Single real PK64 CUDA/AMP contract for exp412 PSGC."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg
from datasets import make_dataloader
from loss import make_loss
from loss.pose_clip_multi_positive_set import (
    PoseClipSetCache,
    build_pose_clip_training_state,
    pose_visibility_signature,
)
from model import make_model
from model.pose_semantic_gradient_completion import (
    PoseSemanticTextAxes,
    build_psgc_slot_weights,
)
from solver import make_optimizer


MAX_NATIVE_ATTEMPTS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--host-config",
        default=(
            "configs/occluded_duke/"
            "swin_tiny_tapf_pcmpsr_exp411_zero_owner.yml"
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def rng_snapshot():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [
            state.clone() for state in torch.cuda.get_rng_state_all()
        ],
    }


def assert_rng_exact(left, right, label):
    if left["python"] != right["python"]:
        raise RuntimeError(label + " Python RNG differs")
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    if (
        left_numpy[0] != right_numpy[0]
        or not np.array_equal(left_numpy[1], right_numpy[1])
        or left_numpy[2:] != right_numpy[2:]
    ):
        raise RuntimeError(label + " NumPy RNG differs")
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        raise RuntimeError(label + " Torch CPU RNG differs")
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        raise RuntimeError(label + " CUDA RNG device count differs")
    for index, (a, b) in enumerate(
        zip(left["torch_cuda"], right["torch_cuda"])
    ):
        if not torch.equal(a, b):
            raise RuntimeError(
                "{} CUDA RNG differs on device {}".format(label, index)
            )


def load_configs(method_path, host_path):
    method = default_cfg.clone()
    method.merge_from_file(method_path)
    method.freeze()
    host = default_cfg.clone()
    host.merge_from_file(host_path)
    host.freeze()
    disabled = method.clone()
    disabled.defrost()
    disabled.MODEL.TAPF.PSGC_ENABLED = False
    disabled.MODEL.TAPF.PSGC_CONTROL_MODE = "correct"
    disabled.MODEL.TAPF.PSGC_TEXT_AXES = ""
    disabled.MODEL.TAPF.PSGC_TEXT_AXES_SHA256 = ""
    disabled.OUTPUT_DIR = host.OUTPUT_DIR
    disabled.freeze()
    if disabled.dump() != host.dump():
        raise RuntimeError(
            "exp412 config differs from sealed zero-owner beyond PSGC"
        )
    return method, host, disabled


def make_one_model(local_cfg, num_classes, camera_num, view_num, device):
    return make_model(
        local_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=local_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)


def cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def assert_state_exact(left, right):
    if set(left) != set(right):
        raise RuntimeError("PSGC default-off model state keys differ")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise RuntimeError(
                "PSGC default-off model state differs at " + name
            )


def assert_tensor_exact(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError(label + " metadata differs")
    if not torch.equal(left, right):
        raise RuntimeError(label + " values differ")


def run_gradient_branch(
    local_cfg,
    num_classes,
    camera_num,
    view_num,
    image,
    identity,
    camera,
    view,
    pose_batch,
    pcmpsr_state,
    device,
):
    set_seed(local_cfg.SOLVER.SEED)
    model = make_one_model(
        local_cfg, num_classes, camera_num, view_num, device
    )
    loss_fn, center = make_loss(local_cfg, num_classes=num_classes)
    model.train()
    state = cpu_state(model)
    construction_rng = rng_snapshot()
    set_seed(4121234)
    forward_start_rng = rng_snapshot()
    with amp.autocast(enabled=True):
        score, feature, feature_maps, tapf_aux = model(
            image,
            label=identity,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
        reid = loss_fn(
            score,
            feature,
            identity,
            camera,
            pcmpsr_state=pcmpsr_state,
        )
        loss = reid + (
            local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
            * tapf_aux["pose_loss"]
        )
    forward_end_rng = rng_snapshot()
    output = {
        "score": score.detach().cpu().clone(),
        "feature": feature.detach().cpu().clone(),
        "final_map": feature_maps[-1].detach().cpu().clone(),
        "pose_loss": tapf_aux["pose_loss"].detach().cpu().clone(),
        "reid_loss": reid.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
    }
    route_diag = {
        key: tapf_aux[key].detach().cpu().clone()
        for key in (
            "psgc_gradient_min",
            "psgc_gradient_max",
            "psgc_gradient_mean",
            "psgc_body_fraction",
            "psgc_region_valid_fraction",
        )
        if key in tapf_aux
    }
    loss.backward()
    gradients = {}
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not (
            name.startswith("base.stages.3.")
            or name.startswith("base.norm3.")
        ):
            continue
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        gradients[name] = gradient.cpu().clone()
    if nonfinite or not gradients:
        raise RuntimeError(
            "PSGC Stage-3 gradient invalid: {}".format(nonfinite[:5])
        )
    del loss, reid, score, feature, feature_maps, tapf_aux
    del loss_fn, center, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "state": state,
        "construction_rng": construction_rng,
        "forward_start_rng": forward_start_rng,
        "forward_end_rng": forward_end_rng,
        "output": output,
        "route_diag": route_diag,
        "gradients": gradients,
    }


def native_amp_update(
    local_cfg,
    num_classes,
    camera_num,
    view_num,
    image,
    identity,
    camera,
    view,
    pose_batch,
    pcmpsr_state,
    device,
):
    set_seed(local_cfg.SOLVER.SEED)
    model = make_one_model(
        local_cfg, num_classes, camera_num, view_num, device
    )
    loss_fn, center = make_loss(local_cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(local_cfg, model, center)
    model.train()
    scaler = amp.GradScaler()
    attempts = []
    success = None
    for attempt in range(1, MAX_NATIVE_ATTEMPTS + 1):
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        with amp.autocast(enabled=True):
            score, feature, _, tapf_aux = model(
                image,
                label=identity,
                cam_label=camera,
                view_label=view,
                pose_batch=pose_batch,
                tapf_epoch=1,
            )
            reid = loss_fn(
                score,
                feature,
                identity,
                camera,
                pcmpsr_state=pcmpsr_state,
            )
            loss = reid + (
                local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
                * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("PSGC combined loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        active = []
        nonfinite = []
        for name, parameter in model.named_parameters():
            if not name.startswith("base.stages.3.") or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not bool(torch.isfinite(gradient).all()):
                nonfinite.append(name)
            elif bool((gradient != 0).any()):
                active.append((name, parameter))
        selected = (
            max(
                active,
                key=lambda item: float(
                    item[1].grad.detach().abs().max().item()
                ),
            )
            if active
            else None
        )
        before = selected[1].detach().clone() if selected is not None else None
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        overflow = scale_after < scale_before
        updated = (
            selected is not None
            and not torch.equal(before, selected[1].detach())
        )
        attempts.append(
            {
                "attempt": attempt,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "overflow": overflow,
                "nonfinite_gradient_tensors": len(nonfinite),
                "updated": updated,
            }
        )
        if overflow:
            if updated:
                raise RuntimeError("PSGC native overflow updated a parameter")
            continue
        if nonfinite:
            raise RuntimeError("PSGC non-finite gradients escaped GradScaler")
        if selected is None or not updated:
            raise RuntimeError("PSGC finite Stage-3 update is inactive")
        success = {
            "attempt": attempt,
            "parameter": selected[0],
            "stage3_count": len(active),
            "loss": float(loss.detach().item()),
            "reid_loss": float(reid.detach().item()),
            "pose_loss": float(tapf_aux["pose_loss"].detach().item()),
        }
        break
    if success is None:
        raise RuntimeError("PSGC default GradScaler did not update Stage-3")
    return attempts, success


def main():
    args = parse_args()
    method_cfg, _, disabled_cfg = load_configs(
        args.config, args.host_config
    )
    if not (
        method_cfg.MODEL.TAPF.ENABLED
        and method_cfg.MODEL.TAPF.PCMPSR_ENABLED
        and method_cfg.MODEL.TAPF.PSGC_ENABLED
        and str(method_cfg.MODEL.TAPF.PCMPSR_CONTROL_MODE).lower()
        == "zero_owner"
        and str(method_cfg.MODEL.TAPF.PSGC_CONTROL_MODE).lower()
        == "correct"
    ):
        raise RuntimeError("real PK64 check requires formal PSGC correct")
    if (
        method_cfg.SOLVER.IMS_PER_BATCH != 64
        or method_cfg.DATALOADER.NUM_INSTANCE != 4
    ):
        raise RuntimeError("PSGC contract requires frozen PK64/K4")
    set_seed(method_cfg.SOLVER.SEED)
    device = torch.device("cuda", 0)
    (
        train_loader,
        _,
        _,
        _,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(method_cfg)
    cache = PoseClipSetCache(
        method_cfg.MODEL.TAPF.PCMPSR_CACHE,
        method_cfg.MODEL.TAPF.PCMPSR_CACHE_SHA256,
        method_cfg.MODEL.TAPF.PCMPSR_CLIP_CHECKPOINT_SHA256,
        method_cfg.MODEL.TAPF.MANIFEST_SHA256,
    )
    axes = PoseSemanticTextAxes(
        method_cfg.MODEL.TAPF.PSGC_TEXT_AXES,
        method_cfg.MODEL.TAPF.PSGC_TEXT_AXES_SHA256,
        method_cfg.MODEL.TAPF.PCMPSR_CLIP_CHECKPOINT_SHA256,
    )
    batch = next(iter(train_loader))
    image, identity, camera, view, pose = batch
    if image.shape[0] != 64 or identity.unique().numel() != 16:
        raise RuntimeError("PSGC real batch is not 16x4")
    counts = torch.stack(
        [(identity == value).sum() for value in identity.unique()]
    )
    if not bool((counts == 4).all()):
        raise RuntimeError("PSGC real batch identity multiplicity drift")

    relative_paths = pose["relative_paths"]
    image_sha256 = pose["image_sha256"]
    base_pose = {
        "keypoints": pose["keypoints"].to(device),
        "scores": pose["scores"].to(device),
        "valid": pose["valid"].to(device),
    }
    image = image.to(device)
    identity = identity.to(device)
    camera = camera.to(device)
    view = view.to(device)
    clip_features, clip_valid = cache.lookup(relative_paths, image_sha256)
    clip_features = clip_features.to(device)
    clip_valid = clip_valid.to(device)
    text_prototypes = axes.to(device)
    with torch.no_grad(), amp.autocast(enabled=False):
        visibility = pose_visibility_signature(
            base_pose["scores"], base_pose["valid"]
        )
        pcmpsr_state = build_pose_clip_training_state(
            identity,
            visibility,
            clip_features,
            clip_valid,
            control_mode="zero_owner",
        )
        weights = {}
        diagnostics = {}
        for mode in ("correct", "pose_only", "q_only", "text_shuffle"):
            weights[mode], diagnostics[mode] = build_psgc_slot_weights(
                identity,
                visibility,
                clip_features,
                clip_valid,
                text_prototypes,
                mode=mode,
            )
    if pcmpsr_state["use_owner_multiplicity"]:
        raise RuntimeError("PSGC restored the forbidden PCMPSR owner term")
    control_change = {
        mode: float(
            weights[mode].ne(weights["correct"]).float().mean().item()
        )
        for mode in ("pose_only", "q_only", "text_shuffle")
    }
    if any(value <= 0 for value in control_change.values()):
        raise RuntimeError("PSGC real-batch control front is inactive")
    if float(diagnostics["correct"]["semantic_std"].item()) <= 0:
        raise RuntimeError("PSGC CLIP semantic margin is constant")
    if torch.equal(weights["correct"], torch.ones_like(weights["correct"])):
        raise RuntimeError("PSGC correct route collapsed to host")

    host_pose = dict(base_pose)
    method_pose = dict(base_pose)
    method_pose["psgc_slot_weights"] = weights["correct"]
    host = run_gradient_branch(
        disabled_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        host_pose,
        pcmpsr_state,
        device,
    )
    method = run_gradient_branch(
        method_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        method_pose,
        pcmpsr_state,
        device,
    )
    assert_state_exact(host["state"], method["state"])
    assert_rng_exact(
        host["construction_rng"],
        method["construction_rng"],
        "PSGC construction",
    )
    assert_rng_exact(
        host["forward_start_rng"],
        method["forward_start_rng"],
        "PSGC forward start",
    )
    assert_rng_exact(
        host["forward_end_rng"],
        method["forward_end_rng"],
        "PSGC forward end",
    )
    for key in host["output"]:
        assert_tensor_exact(
            host["output"][key], method["output"][key], "PSGC " + key
        )
    if len(method["route_diag"]) != 5 or host["route_diag"]:
        raise RuntimeError("PSGC route diagnostics scope mismatch")
    shared_gradients = sorted(
        set(host["gradients"]) & set(method["gradients"])
    )
    if not shared_gradients:
        raise RuntimeError("PSGC has no comparable Stage-3 gradients")
    changed_gradients = [
        name
        for name in shared_gradients
        if not torch.equal(
            host["gradients"][name], method["gradients"][name]
        )
    ]
    if not changed_gradients:
        raise RuntimeError("PSGC backward route is inactive")

    attempts, update = native_amp_update(
        method_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        method_pose,
        pcmpsr_state,
        device,
    )
    result = {
        "schema": "exp412-psgc-real-pk64-v1",
        "status": "PASS",
        "batch": 64,
        "identities": 16,
        "instances_per_identity": 4,
        "pcmpsr_owner_terms": 0,
        "default_off_state_forward_loss_rng_exact": True,
        "route_forward_dtype_exact": True,
        "changed_stage3_gradient_tensors": len(changed_gradients),
        "comparable_stage3_gradient_tensors": len(shared_gradients),
        "control_weight_change": control_change,
        "correct_front_size_mean": float(
            diagnostics["correct"]["front_size_mean"].item()
        ),
        "correct_front_fraction": float(
            diagnostics["correct"]["front_fraction"].item()
        ),
        "correct_fallback_fraction": float(
            diagnostics["correct"]["fallback_fraction"].item()
        ),
        "correct_weight_min": float(
            diagnostics["correct"]["weight_min"].item()
        ),
        "correct_weight_max": float(
            diagnostics["correct"]["weight_max"].item()
        ),
        "correct_semantic_mean": float(
            diagnostics["correct"]["semantic_mean"].item()
        ),
        "correct_semantic_std": float(
            diagnostics["correct"]["semantic_std"].item()
        ),
        "gradient_field": {
            key: float(value.item())
            for key, value in method["route_diag"].items()
        },
        "native_attempts": attempts,
        "first_successful_update": update["attempt"],
        "updated_parameter": update["parameter"],
        "stage3_nonzero_grad_tensors": update["stage3_count"],
        "loss": update["loss"],
        "reid_loss": update["reid_loss"],
        "pose_loss": update["pose_loss"],
        "cache_sha256": cache.sha256,
        "text_axes_sha256": axes.sha256,
        "text_prompt_spec_sha256": axes.prompt_spec_sha256,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
