#!/usr/bin/env python3
"""One real PK-batch CUDA/AMP optimizer-step check for exp409 PCHM."""

from __future__ import annotations

import argparse
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

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from loss.pose_clip_hard_mining import (
    PoseClipMiningCache,
    batch_hard_pair_indices,
    pose_visibility_signature,
    select_pose_clip_pairs,
)
from model import make_model
from solver import make_optimizer


MAX_NATIVE_ATTEMPTS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def gradient_report(named_parameters, predicate):
    active = []
    nonfinite = []
    for name, parameter in named_parameters:
        if not predicate(name) or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        elif bool((gradient != 0).any()):
            active.append((name, parameter))
    return active, nonfinite


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    if not (cfg.MODEL.TAPF.ENABLED and cfg.MODEL.TAPF.PCHM_ENABLED):
        raise RuntimeError("real-batch check requires PCHM")
    if cfg.MODEL.TAPF.PICRD_ENABLED:
        raise RuntimeError("real-batch check forbids PICRD")
    if cfg.SOLVER.IMS_PER_BATCH != 64 or cfg.DATALOADER.NUM_INSTANCE != 4:
        raise RuntimeError("real-batch check requires frozen PK batch64/K4")
    set_seed(cfg.SOLVER.SEED)
    device = torch.device("cuda", 0)

    (
        train_loader,
        _,
        _,
        _,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    cache = PoseClipMiningCache(
        cfg.MODEL.TAPF.PCHM_CACHE,
        cfg.MODEL.TAPF.PCHM_CACHE_SHA256,
        cfg.MODEL.TAPF.PCHM_CLIP_CHECKPOINT_SHA256,
        cfg.MODEL.TAPF.MANIFEST_SHA256,
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
            mode: select_pose_clip_pairs(
                identity,
                visibility,
                clip_features,
                clip_valid,
                mode=mode,
            )
            for mode in ("correct", "pose_shuffle", "clip_only")
        }
    correct = states["correct"]
    pair_indices = (
        correct["positive_indices"],
        correct["negative_indices"],
    )

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
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
    d0_positive_change = d0_negative_change = None
    control_change = None
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
                raise RuntimeError("PCHM requires one final global descriptor")
            reid_loss = loss_fn(
                score,
                feature,
                identity,
                camera,
                pair_indices=pair_indices,
            )
            loss = reid_loss + (
                cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("real-batch PCHM loss is non-finite")
        descriptor_gradient = torch.autograd.grad(
            loss, feature, retain_graph=True
        )[0]
        if not bool(torch.isfinite(descriptor_gradient).all()) or not bool(
            (descriptor_gradient != 0).any()
        ):
            raise RuntimeError(
                "unscaled final descriptor gradient is inactive or non-finite"
            )
        if attempt == 1:
            with torch.no_grad(), amp.autocast(enabled=False):
                d0_positive, d0_negative = batch_hard_pair_indices(
                    feature.detach().float(),
                    identity,
                    normalize_feature=cfg.SOLVER.TRP_L2,
                )
                d0_positive_change = (
                    d0_positive != correct["positive_indices"]
                ).float().mean()
                d0_negative_change = (
                    d0_negative != correct["negative_indices"]
                ).float().mean()
                control_change = {}
                for mode in ("pose_shuffle", "clip_only"):
                    changed = (
                        states[mode]["positive_indices"]
                        != correct["positive_indices"]
                    ) | (
                        states[mode]["negative_indices"]
                        != correct["negative_indices"]
                    )
                    control_change[mode] = float(
                        changed.float().mean().item()
                    )
            if not bool(d0_positive_change > 0) or not bool(
                d0_negative_change > 0
            ):
                raise RuntimeError("PCHM real-batch edges equal legacy batch-hard")
            if any(value <= 0 for value in control_change.values()):
                raise RuntimeError("PCHM real-batch pose or CLIP axis is inactive")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        stage3, _ = gradient_report(
            named_parameters, lambda name: name.startswith("base.stages.3.")
        )
        backbone, _ = gradient_report(
            named_parameters, lambda name: name.startswith("base.")
        )
        _, all_nonfinite = gradient_report(
            named_parameters, lambda name: True
        )
        nonfinite = sorted(set(all_nonfinite))
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
            raise RuntimeError("finite real-batch optimizer step did not update Stage-3")
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
    result = {
        "schema": "exp409-pchm-real-batch-v2",
        "status": "PASS",
        "batch": 64,
        "identities": 16,
        "instances_per_identity": 4,
        "loss": float(success["loss"].detach().item()),
        "reid_loss": float(success["reid_loss"].detach().item()),
        "pose_loss": float(success["pose_loss"].detach().item()),
        "positive_pose_distance": float(
            correct["positive_pose_distance"].mean().item()
        ),
        "positive_clip_similarity": float(
            correct["positive_clip_similarity"].mean().item()
        ),
        "negative_pose_distance": float(
            correct["negative_pose_distance"].mean().item()
        ),
        "negative_clip_similarity": float(
            correct["negative_clip_similarity"].mean().item()
        ),
        "d0_positive_index_change": float(d0_positive_change.item()),
        "d0_negative_index_change": float(d0_negative_change.item()),
        "control_index_change": control_change,
        "native_attempts": attempts,
        "first_successful_update": success["attempt"],
        "stage3_nonzero_grad_tensors": len(success["stage3"]),
        "backbone_nonzero_grad_tensors": len(success["backbone"]),
        "updated_parameter": success["updated_parameter"],
        "cache_sha256": cache.sha256,
        "cache_source_head": cache.source_head,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
