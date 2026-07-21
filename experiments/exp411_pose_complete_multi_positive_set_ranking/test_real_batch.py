#!/usr/bin/env python3
"""One real PK64 CUDA/AMP contract for exp411 PCMPSR."""

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
    build_pose_clip_identity_sets,
    build_pose_clip_training_state,
    pose_clip_identity_set_ranking_loss,
    pose_visibility_signature,
)
from model import make_model
from solver import make_optimizer


MAX_NATIVE_ATTEMPTS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--d0-config",
        default="configs/occluded_duke/swin_tiny_tapf_d0.yml",
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
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def assert_rng_exact(left, right, label):
    if left["python"] != right["python"]:
        raise RuntimeError("{} Python RNG differs".format(label))
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    if (
        left_numpy[0] != right_numpy[0]
        or not np.array_equal(left_numpy[1], right_numpy[1])
        or left_numpy[2:] != right_numpy[2:]
    ):
        raise RuntimeError("{} NumPy RNG differs".format(label))
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        raise RuntimeError("{} Torch CPU RNG differs".format(label))
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        raise RuntimeError("{} CUDA RNG device count differs".format(label))
    for index, (a, b) in enumerate(
        zip(left["torch_cuda"], right["torch_cuda"])
    ):
        if not torch.equal(a, b):
            raise RuntimeError(
                "{} CUDA RNG differs on device {}".format(label, index)
            )


def load_configs(method_path, d0_path):
    method = default_cfg.clone()
    method.merge_from_file(method_path)
    method.freeze()
    baseline = default_cfg.clone()
    baseline.merge_from_file(d0_path)
    baseline.freeze()
    disabled = method.clone()
    disabled.defrost()
    disabled.MODEL.TAPF.PCMPSR_ENABLED = False
    disabled.MODEL.TAPF.PCMPSR_CONTROL_MODE = "correct"
    disabled.MODEL.TAPF.PCMPSR_CACHE = ""
    disabled.MODEL.TAPF.PCMPSR_CACHE_SHA256 = ""
    disabled.MODEL.TAPF.PCMPSR_CLIP_CHECKPOINT_SHA256 = ""
    disabled.freeze()
    return method, baseline, disabled


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value


def assert_tree_exact(left, right, path="output"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise RuntimeError("{} tensor structure mismatch".format(path))
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError("{} tensor metadata mismatch".format(path))
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise RuntimeError("{} tensor value mismatch".format(path))
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError("{} sequence mismatch".format(path))
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_exact(a, b, "{}[{}]".format(path, index))
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError("{} mapping mismatch".format(path))
        if set(left) != set(right):
            raise RuntimeError("{} mapping keys mismatch".format(path))
        for key in sorted(left):
            assert_tree_exact(left[key], right[key], "{}.{}".format(path, key))
        return
    if left != right:
        raise RuntimeError("{} scalar mismatch".format(path))


def gradient_report(named_parameters, prefix):
    active = []
    nonfinite = []
    for name, parameter in named_parameters:
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        elif bool((gradient != 0).any()):
            active.append((name, parameter))
    return active, nonfinite


def make_one_model(local_cfg, num_classes, camera_num, view_num, device):
    return make_model(
        local_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=local_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)


def default_off_contract(
    baseline_cfg,
    disabled_cfg,
    num_classes,
    camera_num,
    view_num,
    image,
    identity,
    camera,
    view,
    pose_batch,
    device,
):
    snapshots = {}
    outputs = {}
    losses = {}
    construction_rng = {}
    forward_start_rng = {}
    forward_end_rng = {}
    for name, local_cfg in (("d0", baseline_cfg), ("disabled", disabled_cfg)):
        set_seed(local_cfg.SOLVER.SEED)
        model = make_one_model(
            local_cfg, num_classes, camera_num, view_num, device
        )
        loss_fn, center = make_loss(local_cfg, num_classes=num_classes)
        model.train()
        snapshots[name] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        construction_rng[name] = rng_snapshot()
        set_seed(4111234)
        forward_start_rng[name] = rng_snapshot()
        with amp.autocast(enabled=True):
            output = model(
                image,
                label=identity,
                cam_label=camera,
                view_label=view,
                pose_batch=pose_batch,
                tapf_epoch=1,
            )
            score, feature, _, tapf_aux = output
            reid = loss_fn(score, feature, identity, camera)
            loss = reid + (
                local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
                * tapf_aux["pose_loss"]
            )
        outputs[name] = clone_tree(output)
        losses[name] = loss.detach().cpu().clone()
        forward_end_rng[name] = rng_snapshot()
        del loss, reid, output, loss_fn, center, model
        gc.collect()
        torch.cuda.empty_cache()

    if set(snapshots["d0"]) != set(snapshots["disabled"]):
        raise RuntimeError("default-off model state keys differ")
    for key in snapshots["d0"]:
        if not torch.equal(snapshots["d0"][key], snapshots["disabled"][key]):
            raise RuntimeError("default-off model state differs at {}".format(key))
    assert_tree_exact(outputs["d0"], outputs["disabled"])
    if not torch.equal(losses["d0"], losses["disabled"]):
        raise RuntimeError("default-off combined loss differs")
    assert_rng_exact(
        construction_rng["d0"],
        construction_rng["disabled"],
        "default-off construction",
    )
    assert_rng_exact(
        forward_start_rng["d0"],
        forward_start_rng["disabled"],
        "default-off forward start",
    )
    assert_rng_exact(
        forward_end_rng["d0"],
        forward_end_rng["disabled"],
        "default-off forward end",
    )
    return float(losses["d0"].item())


def main():
    args = parse_args()
    method_cfg, baseline_cfg, disabled_cfg = load_configs(
        args.config, args.d0_config
    )
    if not (
        method_cfg.MODEL.TAPF.ENABLED
        and method_cfg.MODEL.TAPF.PCMPSR_ENABLED
    ):
        raise RuntimeError("real-batch check requires PCMPSR")
    control_mode = str(method_cfg.MODEL.TAPF.PCMPSR_CONTROL_MODE).lower()
    if control_mode not in {"correct", "zero_owner", "wrong_rgb"}:
        raise RuntimeError("real-batch formal control mode is invalid")
    if method_cfg.SOLVER.IMS_PER_BATCH != 64 or method_cfg.DATALOADER.NUM_INSTANCE != 4:
        raise RuntimeError("real-batch check requires frozen PK64/K4")
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
    if len(cache) != len(train_loader.dataset):
        raise RuntimeError("real-batch cache coverage count mismatch")
    expected_paths = tuple(
        Path(record[0])
        .resolve()
        .relative_to(train_loader.dataset.pose_store.dataset_root)
        .as_posix()
        for record in train_loader.dataset.dataset
    )
    if len(set(expected_paths)) != len(expected_paths) or set(expected_paths) != set(
        cache.paths
    ):
        raise RuntimeError("real-batch cache path coverage mismatch")

    batch = next(iter(train_loader))
    image, identity, camera, view, pose = batch
    if image.shape[0] != 64 or identity.unique().numel() != 16:
        raise RuntimeError("real-batch sampler is not 16 identities x 4 images")
    counts = torch.stack([(identity == value).sum() for value in identity.unique()])
    if not bool((counts == 4).all()):
        raise RuntimeError("real-batch identity multiplicity is not four")
    relative_paths = pose["relative_paths"]
    image_sha256 = pose["image_sha256"]
    pose_batch = {
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
    with torch.no_grad(), amp.autocast(enabled=False):
        visibility = pose_visibility_signature(
            pose_batch["scores"], pose_batch["valid"]
        )
        states = {
            mode: build_pose_clip_identity_sets(
                identity,
                visibility,
                clip_features,
                clip_valid,
                mode=mode,
            )
            for mode in ("correct", "wrong_rgb", "generic", "pose_only")
        }
    correct = states["correct"]
    training_state = build_pose_clip_training_state(
        identity,
        visibility,
        clip_features,
        clip_valid,
        control_mode=control_mode,
    )
    if not torch.equal(
        training_state["support_indices"], correct["support_indices"]
    ):
        raise RuntimeError("formal control changed the frozen support set")
    if control_mode == "wrong_rgb":
        if not torch.equal(
            training_state["owner_indices"],
            states["wrong_rgb"]["owner_indices"],
        ):
            raise RuntimeError(
                "formal wrong-RGB is not the direct wrong owner"
            )
        if torch.equal(
            training_state["owner_indices"], correct["owner_indices"]
        ):
            raise RuntimeError("formal wrong-RGB owner collapsed to correct")
    elif not torch.equal(
        training_state["owner_indices"], correct["owner_indices"]
    ):
        raise RuntimeError("correct/zero-owner changed correct owners")
    control_change = {
        mode: float(
            states[mode]["owner_indices"]
            .ne(correct["owner_indices"])
            .float()
            .mean()
            .item()
        )
        for mode in ("wrong_rgb", "generic", "pose_only")
    }
    if float(correct["owner_unique_mean"].item()) <= 1.0:
        raise RuntimeError("real-batch PCMPSR owner multiplicity collapsed")
    if any(value <= 0.0 for value in control_change.values()):
        raise RuntimeError("real-batch PCMPSR control axis is inactive")

    default_loss = default_off_contract(
        baseline_cfg,
        disabled_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        pose_batch,
        device,
    )

    set_seed(method_cfg.SOLVER.SEED)
    model = make_one_model(
        method_cfg, num_classes, camera_num, view_num, device
    )
    loss_fn, center_criterion = make_loss(method_cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(method_cfg, model, center_criterion)
    model.train()
    scaler = amp.GradScaler()
    named_parameters = list(model.named_parameters())
    stage3_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("base.stages.3.")
    ]
    if not stage3_parameters:
        raise RuntimeError("could not identify Stage-3 parameters")
    attempts = []
    success = None
    set_diag = None
    isolated_stage3_count = None
    isolated_backbone_count = None
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
            if isinstance(feature, list):
                raise RuntimeError("PCMPSR requires one final global descriptor")
            reid_loss = loss_fn(
                score,
                feature,
                identity,
                camera,
                pcmpsr_state=training_state,
            )
            loss = reid_loss + (
                method_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
                * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("real-batch PCMPSR loss is non-finite")
        isolated_set_loss, _ = pose_clip_identity_set_ranking_loss(
            feature,
            identity,
            training_state,
            normalize_feature=method_cfg.SOLVER.TRP_L2,
        )
        descriptor_gradient = torch.autograd.grad(
            isolated_set_loss, feature, retain_graph=True
        )[0]
        if not bool(torch.isfinite(descriptor_gradient).all()) or not bool(
            (descriptor_gradient != 0).any()
        ):
            raise RuntimeError(
                "PCMPSR final descriptor gradient is inactive or non-finite"
            )
        isolated_named = [
            (name, parameter)
            for name, parameter in named_parameters
            if name.startswith("base.") and parameter.requires_grad
        ]
        isolated_gradients = torch.autograd.grad(
            isolated_set_loss,
            [parameter for _, parameter in isolated_named],
            retain_graph=True,
            allow_unused=True,
        )
        isolated_active = []
        isolated_nonfinite = []
        for (name, _), gradient in zip(isolated_named, isolated_gradients):
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                isolated_nonfinite.append(name)
            elif bool((gradient != 0).any()):
                isolated_active.append(name)
        if isolated_nonfinite:
            raise RuntimeError(
                "isolated PCMPSR gradient is non-finite: {}".format(
                    isolated_nonfinite[:5]
                )
            )
        isolated_stage3_count = sum(
            name.startswith("base.stages.3.") for name in isolated_active
        )
        isolated_backbone_count = len(isolated_active)
        if isolated_stage3_count <= 0 or isolated_backbone_count <= 0:
            raise RuntimeError(
                "isolated PCMPSR Stage-3/backbone gradient is inactive"
            )
        if attempt == 1:
            with torch.no_grad(), amp.autocast(enabled=False):
                _, set_diag = pose_clip_identity_set_ranking_loss(
                    feature.detach().float(),
                    identity,
                    training_state,
                    normalize_feature=method_cfg.SOLVER.TRP_L2,
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        stage3, stage3_nonfinite = gradient_report(
            named_parameters, "base.stages.3."
        )
        backbone, backbone_nonfinite = gradient_report(named_parameters, "base.")
        nonfinite = sorted(set(stage3_nonfinite + backbone_nonfinite))
        update_name, update_parameter = (
            max(
                stage3,
                key=lambda item: float(item[1].grad.detach().abs().max().item()),
            )
            if stage3
            else stage3_parameters[0]
        )
        before_update = update_parameter.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        updated = not torch.equal(before_update, update_parameter.detach())
        overflow = scale_after < scale_before
        attempts.append(
            {
                "attempt": attempt,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "nonfinite_gradient_tensors": len(nonfinite),
                "native_overflow": overflow,
                "updated": updated,
            }
        )
        if overflow:
            if updated:
                raise RuntimeError("native GradScaler overflow semantics failed")
            continue
        if nonfinite:
            raise RuntimeError("non-finite gradients did not trigger native backoff")
        if not stage3 or not backbone:
            raise RuntimeError("Stage-3/backbone gradient contract failed")
        if not updated:
            raise RuntimeError("finite optimizer step did not update Stage-3")
        success = {
            "attempt": attempt,
            "stage3": stage3,
            "backbone": backbone,
            "updated_parameter": update_name,
            "loss": loss,
            "reid_loss": reid_loss,
            "pose_loss": tapf_aux["pose_loss"],
        }
        break
    if success is None:
        raise RuntimeError("default GradScaler did not reach one native update")
    expected_owner_terms = 0 if control_mode == "zero_owner" else 5
    if set_diag["owner_term_count"] != expected_owner_terms:
        raise RuntimeError(
            "formal owner term count is not {}".format(expected_owner_terms)
        )
    result = {
        "schema": "exp411-pcmpsr-real-batch-v1",
        "status": "PASS",
        "control_mode": control_mode,
        "batch": 64,
        "identities": 16,
        "instances_per_identity": 4,
        "default_off_state_forward_loss_rng_exact": True,
        "default_off_loss": default_loss,
        "loss": float(success["loss"].detach().item()),
        "reid_loss": float(success["reid_loss"].detach().item()),
        "pose_loss": float(success["pose_loss"].detach().item()),
        "set_loss": float(set_diag["loss"].item()),
        "positive_set_distance": float(
            set_diag["positive_distance"].mean().item()
        ),
        "negative_set_distance": float(
            set_diag["negative_distance"].mean().item()
        ),
        "owner_unique_mean": float(correct["owner_unique_mean"].item()),
        "owner_fallback_fraction": float(
            training_state["owner_fallback_fraction"].item()
        ),
        "owner_term_count": set_diag["owner_term_count"],
        "control_owner_change": control_change,
        "native_attempts": attempts,
        "first_successful_update": success["attempt"],
        "stage3_nonzero_grad_tensors": len(success["stage3"]),
        "backbone_nonzero_grad_tensors": len(success["backbone"]),
        "isolated_set_stage3_nonzero_grad_tensors": isolated_stage3_count,
        "isolated_set_backbone_nonzero_grad_tensors": isolated_backbone_count,
        "updated_parameter": success["updated_parameter"],
        "cache_sha256": cache.sha256,
        "cache_source_head": cache.source_head,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
