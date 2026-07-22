#!/usr/bin/env python3
"""Single micro-oracle plus real PK64 CUDA/AMP contract for exp413."""

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
    pose_clip_identity_set_ranking_loss,
    pose_visibility_signature,
)
from loss.pose_semantic_coverage_chain import (
    build_coverage_chain_from_signals,
    build_pose_semantic_coverage_chain,
    greedy_coverage_permutation,
    pose_semantic_coverage_chain_ranking_loss,
    strict_support_reliability,
)
from model import make_model
from model.pose_semantic_gradient_completion import PoseSemanticTextAxes
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
    left_numpy, right_numpy = left["numpy"], right["numpy"]
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


def assert_tensor_exact(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError(label + " metadata differs")
    if not torch.equal(left, right):
        raise RuntimeError(label + " values differ")


def micro_oracle():
    visibility = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.4, 0.0],
            [0.3, 0.5, 0.1, 0.2, 0.0],
            [0.2, 0.1, 0.1, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    semantic = torch.tensor(
        [
            [-3.0, 1.0, -4.0, 9.0, -1.0],
            [-1.0, 1.0, -2.0, -2.0, -1.0],
            [-2.0, 0.0, -3.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [
            [True, True, False, False, True],
            [True, True, False, True, True],
            [False, True, False, True, True],
        ]
    )
    expected_rank_v = torch.tensor(
        [[0, 1, 2, 2, 0], [2, 1, 0, 0, 0], [1, 0, 0, 1, 0]]
    )
    expected_rank_q = torch.tensor(
        [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
    )
    expected_correct = torch.minimum(expected_rank_v, expected_rank_q)
    reliabilities = {}
    for mode, expected in (
        ("correct", expected_correct),
        ("pose_only", expected_rank_v),
        ("q_only", expected_rank_q),
    ):
        reliability, rank_v, rank_q = strict_support_reliability(
            visibility, semantic, valid, mode=mode
        )
        if not torch.equal(rank_v, expected_rank_v):
            raise RuntimeError("PSCCR micro-oracle rank_v direction failed")
        if not torch.equal(rank_q, expected_rank_q):
            raise RuntimeError("PSCCR micro-oracle rank_q/invalid failed")
        if not torch.equal(reliability, expected):
            raise RuntimeError("PSCCR micro-oracle reliability failed: " + mode)
        reliabilities[mode] = reliability

    support = torch.tensor([1, 8, 4], dtype=torch.long)
    expected_chains = {
        "correct": torch.tensor([8, 4, 1]),
        "pose_only": torch.tensor([1, 8, 4]),
        "q_only": torch.tensor([8, 4, 1]),
        "text_shuffle": torch.tensor([1, 4, 8]),
    }
    observed_chains = {}
    for mode in ("correct", "pose_only", "q_only"):
        observed_chains[mode] = greedy_coverage_permutation(
            support, reliabilities[mode]
        )[0]
    shuffled_semantic = -visibility
    shuffled_reliability = strict_support_reliability(
        visibility, shuffled_semantic, valid, mode="correct"
    )[0]
    observed_chains["text_shuffle"] = greedy_coverage_permutation(
        support, shuffled_reliability
    )[0]
    for mode, expected in expected_chains.items():
        if not torch.equal(observed_chains[mode], expected):
            raise RuntimeError("PSCCR micro-oracle chain failed: " + mode)
    all_invalid = torch.zeros_like(valid)
    all_invalid_q = strict_support_reliability(
        visibility, semantic, all_invalid, mode="q_only"
    )[0]
    if bool(all_invalid_q.any()):
        raise RuntimeError("PSCCR all-invalid q-only oracle failed")
    tie_chain = greedy_coverage_permutation(
        support, torch.zeros_like(expected_rank_v)
    )[0]
    if not torch.equal(tie_chain, torch.tensor([1, 4, 8])):
        raise RuntimeError("PSCCR batch-index tie-break oracle failed")
    return {
        mode: observed_chains[mode].tolist() for mode in expected_chains
    }


def load_configs(method_path, host_path):
    method = default_cfg.clone()
    method.merge_from_file(method_path)
    method.freeze()
    host = default_cfg.clone()
    host.merge_from_file(host_path)
    host.freeze()
    disabled = method.clone()
    disabled.defrost()
    disabled.MODEL.TAPF.PSCCR_ENABLED = False
    disabled.MODEL.TAPF.PSCCR_CONTROL_MODE = "correct"
    disabled.MODEL.TAPF.PSCCR_TEXT_AXES = ""
    disabled.MODEL.TAPF.PSCCR_TEXT_AXES_SHA256 = ""
    disabled.OUTPUT_DIR = host.OUTPUT_DIR
    disabled.freeze()
    if disabled.dump() != host.dump():
        raise RuntimeError(
            "exp413 config differs from sealed zero-owner beyond PSCCR"
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


def assert_state_exact(left, right, label):
    if set(left) != set(right):
        raise RuntimeError(label + " model state keys differ")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise RuntimeError(label + " model state differs at " + name)


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
    device,
    *,
    pcmpsr_state=None,
    psccr_state=None,
):
    set_seed(local_cfg.SOLVER.SEED)
    model = make_one_model(
        local_cfg, num_classes, camera_num, view_num, device
    )
    loss_fn, center = make_loss(local_cfg, num_classes=num_classes)
    model.train()
    state = cpu_state(model)
    construction_rng = rng_snapshot()
    set_seed(4131234)
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
            psccr_state=psccr_state,
        )
        loss = reid + (
            local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux["pose_loss"]
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
    prefix3_exact = None
    if psccr_state is not None:
        with amp.autocast(enabled=False):
            _, chain_diag = pose_semantic_coverage_chain_ranking_loss(
                feature.float(), identity, psccr_state
            )
            base_loss, base_diag = pose_clip_identity_set_ranking_loss(
                feature.float(),
                identity,
                {
                    **psccr_state,
                    "owner_indices": psccr_state["support_indices"][:, :, :1]
                    .expand(-1, -1, 5),
                    "use_owner_multiplicity": False,
                },
            )
        if not torch.equal(chain_diag["prefix_losses"][2], base_loss.detach()):
            raise RuntimeError("PSCCR prefix3 loss is not zero-owner exact")
        if not torch.equal(
            chain_diag["prefix3_set_distance"], base_diag["set_distance"]
        ):
            raise RuntimeError("PSCCR prefix3 distance is not zero-owner exact")
        prefix3_exact = True
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
            "PSCCR Stage-3 gradient invalid: {}".format(nonfinite[:5])
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
        "gradients": gradients,
        "prefix3_exact": prefix3_exact,
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
    psccr_state,
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
                psccr_state=psccr_state,
            )
            loss = reid + (
                local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
                * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("PSCCR combined loss is non-finite")
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
                raise RuntimeError("PSCCR native overflow updated a parameter")
            continue
        if nonfinite:
            raise RuntimeError("PSCCR non-finite gradients escaped GradScaler")
        if selected is None or not updated:
            raise RuntimeError("PSCCR finite Stage-3 update is inactive")
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
        raise RuntimeError("PSCCR default GradScaler did not update Stage-3")
    return attempts, success


def main():
    micro = micro_oracle()
    args = parse_args()
    method_cfg, host_cfg, disabled_cfg = load_configs(
        args.config, args.host_config
    )
    if not (
        method_cfg.MODEL.TAPF.ENABLED
        and method_cfg.MODEL.TAPF.PCMPSR_ENABLED
        and method_cfg.MODEL.TAPF.PSCCR_ENABLED
        and not method_cfg.MODEL.TAPF.PSGC_ENABLED
        and str(method_cfg.MODEL.TAPF.PCMPSR_CONTROL_MODE).lower()
        == "zero_owner"
        and str(method_cfg.MODEL.TAPF.PSCCR_CONTROL_MODE).lower()
        == "correct"
    ):
        raise RuntimeError("real PK64 check requires formal PSCCR correct")
    if (
        method_cfg.SOLVER.IMS_PER_BATCH != 64
        or method_cfg.DATALOADER.NUM_INSTANCE != 4
    ):
        raise RuntimeError("PSCCR contract requires frozen PK64/K4")
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
        method_cfg.MODEL.TAPF.PSCCR_TEXT_AXES,
        method_cfg.MODEL.TAPF.PSCCR_TEXT_AXES_SHA256,
        method_cfg.MODEL.TAPF.PCMPSR_CLIP_CHECKPOINT_SHA256,
    )
    image, identity, camera, view, pose = next(iter(train_loader))
    if image.shape[0] != 64 or identity.unique().numel() != 16:
        raise RuntimeError("PSCCR real batch is not 16x4")
    counts = torch.stack(
        [(identity == value).sum() for value in identity.unique()]
    )
    if not bool((counts == 4).all()):
        raise RuntimeError("PSCCR real batch identity multiplicity drift")

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
    text_prototypes = axes.to(device)
    with torch.no_grad(), amp.autocast(enabled=False):
        visibility = pose_visibility_signature(
            pose_batch["scores"], pose_batch["valid"]
        )
        base_state = build_pose_clip_training_state(
            identity,
            visibility,
            clip_features,
            clip_valid,
            control_mode="zero_owner",
        )
        states = {
            mode: build_pose_semantic_coverage_chain(
                identity,
                visibility,
                clip_features,
                clip_valid,
                text_prototypes,
                base_state,
                mode=mode,
            )
            for mode in ("correct", "pose_only", "q_only", "text_shuffle")
        }
    if base_state["use_owner_multiplicity"]:
        raise RuntimeError("PSCCR restored forbidden owner multiplicity")
    control_change = {
        mode: float(
            states[mode]["chain_indices"]
            .ne(states["correct"]["chain_indices"])
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
        for mode in ("pose_only", "q_only", "text_shuffle")
    }
    if any(value <= 0 for value in control_change.values()):
        raise RuntimeError("PSCCR real-batch control chain is inactive")

    anchor = 0
    class_index = int(states["correct"]["positive_class_indices"][anchor])
    excluded = anchor
    mutated_visibility = visibility.clone()
    mutated_semantic = torch.zeros_like(visibility)
    from loss.pose_semantic_coverage_chain import semantic_visibility_margin

    semantic = semantic_visibility_margin(clip_features, text_prototypes)
    mutated_semantic.copy_(semantic)
    mutated_valid = clip_valid.clone()
    mutated_visibility[excluded] = torch.tensor(
        [0.99, 0.01, 0.99, 0.01, 0.99], device=device
    )
    mutated_semantic[excluded] = torch.tensor(
        [-9.0, 9.0, -9.0, 9.0, -9.0], device=device
    )
    mutated_valid[excluded] = ~mutated_valid[excluded]
    mutated_state = build_coverage_chain_from_signals(
        identity,
        mutated_visibility,
        mutated_semantic,
        mutated_valid,
        base_state,
        mode="correct",
    )
    for key in ("chain_indices", "coverage", "rank_v", "rank_q"):
        if not torch.equal(
            mutated_state[key][anchor, class_index],
            states["correct"][key][anchor, class_index],
        ):
            raise RuntimeError("PSCCR excluded-image mutation leaked into " + key)

    host = run_gradient_branch(
        host_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        pose_batch,
        device,
        pcmpsr_state=base_state,
    )
    disabled = run_gradient_branch(
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
        pcmpsr_state=base_state,
    )
    assert_state_exact(host["state"], disabled["state"], "PSCCR default-off")
    for key in ("construction_rng", "forward_start_rng", "forward_end_rng"):
        assert_rng_exact(host[key], disabled[key], "PSCCR default-off " + key)
    for key in host["output"]:
        assert_tensor_exact(
            host["output"][key],
            disabled["output"][key],
            "PSCCR default-off " + key,
        )
    for name in host["gradients"]:
        if name not in disabled["gradients"] or not torch.equal(
            host["gradients"][name], disabled["gradients"][name]
        ):
            raise RuntimeError("PSCCR default-off gradient differs at " + name)

    method = run_gradient_branch(
        method_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        pose_batch,
        device,
        psccr_state=states["correct"],
    )
    assert_state_exact(host["state"], method["state"], "PSCCR method")
    for key in ("score", "feature", "final_map", "pose_loss"):
        assert_tensor_exact(
            host["output"][key], method["output"][key], "PSCCR " + key
        )
    if not method["prefix3_exact"]:
        raise RuntimeError("PSCCR prefix3 exact check was skipped")
    shared_gradients = sorted(
        set(host["gradients"]) & set(method["gradients"])
    )
    changed_gradients = [
        name
        for name in shared_gradients
        if not torch.equal(host["gradients"][name], method["gradients"][name])
    ]
    if not shared_gradients or not changed_gradients:
        raise RuntimeError("PSCCR isolated Stage-3 gradient is inactive")

    attempts, update = native_amp_update(
        method_cfg,
        num_classes,
        camera_num,
        view_num,
        image,
        identity,
        camera,
        view,
        pose_batch,
        states["correct"],
        device,
    )
    result = {
        "schema": "exp413-psccr-real-pk64-v1",
        "status": "PASS",
        "micro_oracle_chains": micro,
        "batch": 64,
        "identities": 16,
        "instances_per_identity": 4,
        "default_off_state_forward_loss_gradient_rng_exact": True,
        "excluded_image_mutation_invariant": True,
        "strict_support_permutation": True,
        "coverage_monotonic": True,
        "prefix3_zero_owner_exact": True,
        "control_chain_change": control_change,
        "changed_stage3_gradient_tensors": len(changed_gradients),
        "comparable_stage3_gradient_tensors": len(shared_gradients),
        "coverage_mean": [
            float(value)
            for value in states["correct"]["coverage"]
            .float()
            .mean(dim=(0, 1))
            .tolist()
        ],
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
