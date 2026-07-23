#!/usr/bin/env python3
"""Single micro-oracle plus real PK64 CUDA/AMP contract for exp414."""

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
from loss.pose_semantic_continuous_region import (
    build_continuous_region_from_signals,
    build_pose_semantic_continuous_region,
    continuous_region_distance,
    continuous_region_ranking_loss,
    pose_semantic_continuous_region_ranking_loss,
    semantic_visibility_margin,
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
    visibility_three = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.4, 0.0],
            [0.3, 0.5, 0.1, 0.2, 0.0],
            [0.2, 0.1, 0.1, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    semantic_three = torch.tensor(
        [
            [-3.0, 1.0, -4.0, 9.0, -1.0],
            [-1.0, 1.0, -2.0, -2.0, -1.0],
            [-2.0, 0.0, -3.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    valid_three = torch.tensor(
        [
            [True, True, False, False, True],
            [True, True, False, True, True],
            [False, True, False, True, True],
        ]
    )
    labels = torch.tensor([7, 7, 7, 7])
    support = torch.tensor(
        [
            [[1, 2, 3]],
            [[0, 2, 3]],
            [[0, 1, 3]],
            [[0, 1, 2]],
        ]
    )
    base_state = {
        "support_indices": support,
        "class_labels": torch.tensor([7]),
        "positive_class_indices": torch.zeros(4, dtype=torch.long),
        "use_owner_multiplicity": False,
    }
    visibility = torch.cat(
        (torch.full((1, 5), 99.0), visibility_three), dim=0
    )
    semantic = torch.cat(
        (torch.full((1, 5), 99.0), semantic_three), dim=0
    )
    valid = torch.cat(
        (torch.ones((1, 5), dtype=torch.bool), valid_three), dim=0
    )
    expected_rank_v = torch.tensor(
        [[0, 1, 2, 2, 0], [2, 1, 0, 0, 0], [1, 0, 0, 1, 0]]
    )
    expected_rank_q = torch.tensor(
        [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
    )
    expected = {
        "correct": ([2, 2, 3], [2, 0]),
        "pose_only": ([6, 5, 3], [0, 1]),
        "q_only": ([1, 2, 3], [2, 1]),
    }
    states = {}
    for mode in ("correct", "pose_only", "q_only"):
        state = build_continuous_region_from_signals(
            labels,
            visibility,
            semantic,
            valid,
            base_state,
            mode=mode,
        )
        if not torch.equal(state["rank_v"][0, 0], expected_rank_v):
            raise RuntimeError("PSCIR micro-oracle rank_v failed")
        if not torch.equal(state["rank_q"][0, 0], expected_rank_q):
            raise RuntimeError("PSCIR micro-oracle rank_q/invalid failed")
        if state["edge_weight"][0, 0].tolist() != expected[mode][0]:
            raise RuntimeError("PSCIR micro-oracle edge weight failed: " + mode)
        if state["selected_edge_ids"][0, 0].tolist() != expected[mode][1]:
            raise RuntimeError("PSCIR micro-oracle MST failed: " + mode)
        states[mode] = state
    shuffled_semantic = torch.cat(
        (torch.full((1, 5), 99.0), -visibility_three), dim=0
    )
    shuffled = build_continuous_region_from_signals(
        labels,
        visibility,
        shuffled_semantic,
        valid,
        base_state,
        mode="correct",
        reported_mode="text_shuffle",
    )
    if shuffled["edge_weight"][0, 0].tolist() != [4, 3, 3]:
        raise RuntimeError("PSCIR micro-oracle text-shuffle weight failed")
    if shuffled["selected_edge_ids"][0, 0].tolist() != [0, 1]:
        raise RuntimeError("PSCIR micro-oracle text-shuffle MST failed")
    all_invalid = build_continuous_region_from_signals(
        labels,
        visibility,
        semantic,
        torch.zeros_like(valid),
        base_state,
        mode="q_only",
    )
    if bool(all_invalid["edge_weight"].any()):
        raise RuntimeError("PSCIR all-invalid q-only oracle failed")
    if all_invalid["selected_edge_ids"][0, 0].tolist() != [0, 1]:
        raise RuntimeError("PSCIR batch-index tie-break oracle failed")

    internal_feature = torch.tensor(
        [[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    repeated_edge = torch.tensor(
        [[[[1, 2], [1, 2]]]] * 3, dtype=torch.long
    )
    internal_distance, _ = continuous_region_distance(
        internal_feature, repeated_edge
    )
    if not torch.allclose(internal_distance[0, 0], torch.tensor(1.0)):
        raise RuntimeError("PSCIR interior projection oracle failed")
    on_line_feature = torch.tensor(
        [[0.5, 0.0], [0.0, 0.0], [1.0, 0.0]],
        requires_grad=True,
    )
    on_line_distance, _ = continuous_region_distance(
        on_line_feature, repeated_edge
    )
    if not torch.equal(on_line_distance[0, 0], torch.tensor(0.0)):
        raise RuntimeError("PSCIR exact zero-distance oracle failed")
    endpoint_feature = torch.tensor(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        requires_grad=True,
    )
    endpoint_edge = torch.tensor(
        [[[[1, 2], [1, 2]]]] * 4, dtype=torch.long
    )
    endpoint_distance, _ = continuous_region_distance(
        endpoint_feature, endpoint_edge
    )
    if not torch.allclose(endpoint_distance[0, 0], torch.tensor(1.0)):
        raise RuntimeError("PSCIR lower-endpoint clamp oracle failed")
    if not torch.allclose(endpoint_distance[3, 0], torch.tensor(1.0)):
        raise RuntimeError("PSCIR upper-endpoint clamp oracle failed")
    zero_feature = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]], requires_grad=True
    )
    zero_edge = torch.tensor(
        [[[[1, 1], [1, 1]]]] * 2, dtype=torch.long
    )
    zero_distance, _ = continuous_region_distance(zero_feature, zero_edge)
    if not torch.allclose(
        zero_distance[0, 0], torch.sqrt(torch.tensor(2.0))
    ):
        raise RuntimeError("PSCIR zero-length edge oracle failed")
    oracle_loss = (
        internal_distance.sum()
        + on_line_distance.sum()
        + endpoint_distance.sum()
        + zero_distance.sum()
    )
    oracle_loss.backward()
    for name, feature in (
        ("interior", internal_feature),
        ("on-line", on_line_feature),
        ("endpoint", endpoint_feature),
        ("zero", zero_feature),
    ):
        if feature.grad is None or not bool(torch.isfinite(feature.grad).all()):
            raise RuntimeError("PSCIR {} projection backward failed".format(name))
    return {
        mode: states[mode]["selected_edge_ids"][0, 0].tolist()
        for mode in states
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
    disabled.MODEL.TAPF.PSCIR_ENABLED = False
    disabled.MODEL.TAPF.PSCIR_CONTROL_MODE = "correct"
    disabled.MODEL.TAPF.PSCIR_TEXT_AXES = ""
    disabled.MODEL.TAPF.PSCIR_TEXT_AXES_SHA256 = ""
    disabled.OUTPUT_DIR = host.OUTPUT_DIR
    disabled.freeze()
    if disabled.dump() != host.dump():
        raise RuntimeError(
            "exp414 config differs from sealed zero-owner beyond PSCIR"
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
    pscir_state=None,
):
    set_seed(local_cfg.SOLVER.SEED)
    model = make_one_model(
        local_cfg, num_classes, camera_num, view_num, device
    )
    loss_fn, center = make_loss(local_cfg, num_classes=num_classes)
    model.train()
    state = cpu_state(model)
    construction_rng = rng_snapshot()
    set_seed(4141234)
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
            pscir_state=pscir_state,
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
    zero_owner_exact = None
    if pscir_state is not None:
        with amp.autocast(enabled=False):
            _, region_diag = pose_semantic_continuous_region_ranking_loss(
                feature.float(),
                identity,
                pscir_state,
                normalize_feature=local_cfg.SOLVER.TRP_L2,
            )
            base_loss, base_diag = pose_clip_identity_set_ranking_loss(
                feature.float(),
                identity,
                {
                    **pscir_state,
                    "owner_indices": pscir_state["support_indices"][:, :, :1]
                    .expand(-1, -1, 5),
                    "use_owner_multiplicity": False,
                },
                normalize_feature=local_cfg.SOLVER.TRP_L2,
            )
        if not torch.equal(region_diag["zero_owner_loss"], base_loss.detach()):
            raise RuntimeError("PSCIR sealed zero-owner loss is not exact")
        if not torch.equal(
            region_diag["zero_owner_set_distance"],
            base_diag["set_distance"],
        ):
            raise RuntimeError("PSCIR sealed zero-owner distance is not exact")
        zero_owner_exact = True
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
            "PSCIR Stage-3 gradient invalid: {}".format(nonfinite[:5])
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
        "zero_owner_exact": zero_owner_exact,
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
    pscir_state,
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
                pscir_state=pscir_state,
            )
            loss = reid + (
                local_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT
                * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("PSCIR production loss is non-finite")
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
                raise RuntimeError("PSCIR native overflow updated a parameter")
            continue
        if nonfinite:
            raise RuntimeError("PSCIR non-finite gradients escaped GradScaler")
        if selected is None or not updated:
            raise RuntimeError("PSCIR finite Stage-3 update is inactive")
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
        raise RuntimeError("PSCIR default GradScaler did not update Stage-3")
    return attempts, success


def isolated_region_gradients(
    local_cfg,
    num_classes,
    camera_num,
    view_num,
    image,
    identity,
    camera,
    view,
    pose_batch,
    pscir_state,
    device,
):
    """Prove the continuous-region branch itself reaches Stage-3."""
    set_seed(local_cfg.SOLVER.SEED)
    model = make_one_model(
        local_cfg, num_classes, camera_num, view_num, device
    )
    model.train()
    set_seed(4141234)
    with amp.autocast(enabled=True):
        _, feature, _, _ = model(
            image,
            label=identity,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
    with amp.autocast(enabled=False):
        region_loss, _ = continuous_region_ranking_loss(
            feature.float(),
            identity,
            pscir_state,
            normalize_feature=local_cfg.SOLVER.TRP_L2,
        )
    if not bool(torch.isfinite(region_loss)):
        raise RuntimeError("PSCIR isolated region loss is non-finite")
    region_loss.backward()
    active = []
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
        elif bool((gradient != 0).any()):
            active.append(name)
    if nonfinite or not active:
        raise RuntimeError(
            "PSCIR isolated region Stage-3 gradient invalid: {}".format(
                nonfinite[:5]
            )
        )
    result = {
        "loss": float(region_loss.detach().item()),
        "stage3_nonzero_gradient_tensors": len(active),
    }
    del region_loss, feature, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    micro = micro_oracle()
    args = parse_args()
    method_cfg, host_cfg, disabled_cfg = load_configs(
        args.config, args.host_config
    )
    if not (
        method_cfg.MODEL.TAPF.ENABLED
        and method_cfg.MODEL.TAPF.PCMPSR_ENABLED
        and method_cfg.MODEL.TAPF.PSCIR_ENABLED
        and not method_cfg.MODEL.TAPF.PSGC_ENABLED
        and not method_cfg.MODEL.TAPF.PSCCR_ENABLED
        and str(method_cfg.MODEL.TAPF.PCMPSR_CONTROL_MODE).lower()
        == "zero_owner"
        and str(method_cfg.MODEL.TAPF.PSCIR_CONTROL_MODE).lower()
        == "correct"
    ):
        raise RuntimeError("real PK64 check requires formal PSCIR correct")
    if (
        method_cfg.SOLVER.IMS_PER_BATCH != 64
        or method_cfg.DATALOADER.NUM_INSTANCE != 4
    ):
        raise RuntimeError("PSCIR contract requires frozen PK64/K4")
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
        method_cfg.MODEL.TAPF.PSCIR_TEXT_AXES,
        method_cfg.MODEL.TAPF.PSCIR_TEXT_AXES_SHA256,
        method_cfg.MODEL.TAPF.PCMPSR_CLIP_CHECKPOINT_SHA256,
    )
    image, identity, camera, view, pose = next(iter(train_loader))
    if image.shape[0] != 64 or identity.unique().numel() != 16:
        raise RuntimeError("PSCIR real batch is not 16x4")
    counts = torch.stack(
        [(identity == value).sum() for value in identity.unique()]
    )
    if not bool((counts == 4).all()):
        raise RuntimeError("PSCIR real batch identity multiplicity drift")

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
            mode: build_pose_semantic_continuous_region(
                identity,
                visibility,
                clip_features,
                clip_valid,
                text_prototypes,
                base_state,
                mode=mode,
            )
            for mode in (
                "correct",
                "pose_only",
                "q_only",
                "text_shuffle",
                "all_edges",
            )
        }
    if base_state["use_owner_multiplicity"]:
        raise RuntimeError("PSCIR restored forbidden owner multiplicity")
    correct_selected = torch.sort(
        states["correct"]["selected_edge_ids"], dim=-1
    ).values
    topology_change = {}
    for mode in ("pose_only", "q_only", "text_shuffle"):
        selected = torch.sort(
            states[mode]["selected_edge_ids"], dim=-1
        ).values
        topology_change[mode] = float(
            selected.ne(correct_selected)
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
    topology_change["all_edges"] = 1.0
    if any(value <= 0 for value in topology_change.values()):
        raise RuntimeError("PSCIR real-batch control topology is inactive")

    semantic = semantic_visibility_margin(clip_features, text_prototypes)
    mutated_visibility = visibility.clone()
    mutated_semantic = semantic.clone()
    mutated_valid = clip_valid.clone()
    anchor = 0
    support = base_state["support_indices"][anchor]
    for class_index, label in enumerate(base_state["class_labels"]):
        class_rows = torch.nonzero(
            identity == label, as_tuple=False
        ).flatten()
        excluded = class_rows[
            ~class_rows[:, None].eq(support[class_index][None, :]).any(dim=1)
        ]
        if excluded.numel() != 1:
            raise RuntimeError("PSCIR excluded-image oracle cardinality drift")
        index = int(excluded.item())
        mutated_visibility[index] = torch.tensor(
            [0.99, 0.01, 0.99, 0.01, 0.99], device=device
        )
        mutated_semantic[index] = torch.tensor(
            [-9.0, 9.0, -9.0, 9.0, -9.0], device=device
        )
        mutated_valid[index] = ~mutated_valid[index]
    mutated_state = build_continuous_region_from_signals(
        identity,
        mutated_visibility,
        mutated_semantic,
        mutated_valid,
        base_state,
        mode="correct",
    )
    for key in (
        "edge_indices",
        "selected_edge_ids",
        "edge_weight",
        "rank_v",
        "rank_q",
    ):
        if not torch.equal(
            mutated_state[key][anchor], states["correct"][key][anchor]
        ):
            raise RuntimeError(
                "PSCIR excluded-image mutation leaked into " + key
            )

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
    assert_state_exact(host["state"], disabled["state"], "PSCIR default-off")
    for key in ("construction_rng", "forward_start_rng", "forward_end_rng"):
        assert_rng_exact(host[key], disabled[key], "PSCIR default-off " + key)
    for key in host["output"]:
        assert_tensor_exact(
            host["output"][key],
            disabled["output"][key],
            "PSCIR default-off " + key,
        )
    for name in host["gradients"]:
        if name not in disabled["gradients"] or not torch.equal(
            host["gradients"][name], disabled["gradients"][name]
        ):
            raise RuntimeError("PSCIR default-off gradient differs at " + name)

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
        pscir_state=states["correct"],
    )
    assert_state_exact(host["state"], method["state"], "PSCIR method")
    for key in ("score", "feature", "final_map", "pose_loss"):
        assert_tensor_exact(
            host["output"][key], method["output"][key], "PSCIR " + key
        )
    if not method["zero_owner_exact"]:
        raise RuntimeError("PSCIR zero-owner exact check was skipped")
    shared_gradients = sorted(
        set(host["gradients"]) & set(method["gradients"])
    )
    changed_gradients = [
        name
        for name in shared_gradients
        if not torch.equal(host["gradients"][name], method["gradients"][name])
    ]
    if not shared_gradients or not changed_gradients:
        raise RuntimeError("PSCIR combined-vs-host Stage-3 gradient is inactive")
    isolated = isolated_region_gradients(
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

    feature = method["output"]["feature"].float()
    correct_region, _ = continuous_region_distance(
        feature,
        states["correct"]["edge_indices"].cpu(),
        normalize_feature=method_cfg.SOLVER.TRP_L2,
    )
    distance_change = {}
    for mode in ("pose_only", "q_only", "text_shuffle", "all_edges"):
        control_region, _ = continuous_region_distance(
            feature,
            states[mode]["edge_indices"].cpu(),
            normalize_feature=method_cfg.SOLVER.TRP_L2,
        )
        distance_change[mode] = float(
            control_region.ne(correct_region).float().mean().item()
        )
    if any(value <= 0 for value in distance_change.values()):
        raise RuntimeError("PSCIR real-batch control distance is inactive")

    unused_mutation = {
        **states["correct"],
        "edge_weight": states["correct"]["edge_weight"].clone(),
    }
    selected_mask = torch.zeros_like(
        unused_mutation["edge_weight"], dtype=torch.bool
    )
    selected_mask.scatter_(
        -1, states["correct"]["selected_edge_ids"], True
    )
    unused_mutation["edge_weight"][~selected_mask] += 1000
    fixed_region, _ = continuous_region_distance(
        feature,
        unused_mutation["edge_indices"].cpu(),
        normalize_feature=method_cfg.SOLVER.TRP_L2,
    )
    if not torch.equal(fixed_region, correct_region):
        raise RuntimeError("PSCIR unused candidate record changed region")

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
        "schema": "exp414-pscir-real-pk64-v1",
        "status": "PASS",
        "micro_oracle_mst": micro,
        "batch": 64,
        "identities": 16,
        "instances_per_identity": 4,
        "default_off_state_forward_loss_gradient_rng_exact": True,
        "excluded_image_mutation_invariant": True,
        "strict_mst_two_edges_covers_three_supports": True,
        "line_projection_oracle_finite_backward": True,
        "zero_owner_loss_distance_exact": True,
        "unused_candidate_record_invariant": True,
        "control_topology_change": topology_change,
        "control_region_distance_change": distance_change,
        "changed_stage3_gradient_tensors": len(changed_gradients),
        "comparable_stage3_gradient_tensors": len(shared_gradients),
        "isolated_region_loss": isolated["loss"],
        "isolated_region_stage3_nonzero_gradient_tensors": isolated[
            "stage3_nonzero_gradient_tensors"
        ],
        "edge_weight_mean": [
            float(value)
            for value in states["correct"]["edge_weight"]
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
