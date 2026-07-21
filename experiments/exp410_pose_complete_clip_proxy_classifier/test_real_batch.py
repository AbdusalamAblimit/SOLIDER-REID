#!/usr/bin/env python3
"""One real PK-batch CUDA/AMP contract for exp410 PC2P."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from model.pose_complete_clip_proxy import PoseCompleteClipProxyBank
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


def active_gradients(named_parameters, prefix):
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


def assert_default_off_exact(pc2p_cfg, num_classes, camera_num, view_num):
    off_cfg = pc2p_cfg.clone()
    off_cfg.defrost()
    off_cfg.MODEL.TAPF.PC2P_ENABLED = False
    off_cfg.MODEL.TAPF.PC2P_BANK = ""
    off_cfg.MODEL.TAPF.PC2P_BANK_SHA256 = ""
    off_cfg.freeze()
    set_seed(pc2p_cfg.SOLVER.SEED)
    enabled = make_model(
        pc2p_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=pc2p_cfg.MODEL.SEMANTIC_WEIGHT,
    )
    rng_enabled = torch.get_rng_state().clone()
    set_seed(pc2p_cfg.SOLVER.SEED)
    disabled = make_model(
        off_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=off_cfg.MODEL.SEMANTIC_WEIGHT,
    )
    rng_disabled = torch.get_rng_state().clone()
    if not torch.equal(rng_enabled, rng_disabled):
        raise RuntimeError("PC2P changed default model initialization RNG")
    enabled_state = enabled.state_dict()
    disabled_state = disabled.state_dict()
    if tuple(enabled_state) != tuple(disabled_state):
        raise RuntimeError("PC2P changed default model state keys")
    for name in enabled_state:
        if not torch.equal(enabled_state[name], disabled_state[name]):
            raise RuntimeError("PC2P changed default model state: " + name)
    return enabled, disabled


def assert_default_off_forward_exact(
    enabled,
    disabled,
    image,
    identity,
    camera,
    view,
    pose_batch,
    loss_fn,
    pose_loss_weight,
):
    enabled.pc2p_enabled = False
    enabled.train()
    disabled.train()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    with torch.no_grad(), amp.autocast(enabled=True):
        enabled_output = enabled(
            image,
            label=identity,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
        enabled_loss = loss_fn(
            enabled_output[0], enabled_output[1], identity, camera
        ) + pose_loss_weight * enabled_output[3]["pose_loss"]
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    with torch.no_grad(), amp.autocast(enabled=True):
        disabled_output = disabled(
            image,
            label=identity,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
        disabled_loss = loss_fn(
            disabled_output[0], disabled_output[1], identity, camera
        ) + pose_loss_weight * disabled_output[3]["pose_loss"]
    for enabled_value, disabled_value, name in (
        (enabled_output[0], disabled_output[0], "score"),
        (enabled_output[1], disabled_output[1], "global feature"),
        (
            enabled_output[3]["pose_loss"],
            disabled_output[3]["pose_loss"],
            "pose loss",
        ),
        (enabled_loss, disabled_loss, "combined loss"),
    ):
        if not torch.equal(enabled_value, disabled_value):
            raise RuntimeError("PC2P changed default-off " + name)
    enabled.pc2p_enabled = True


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    if not (cfg.MODEL.TAPF.ENABLED and cfg.MODEL.TAPF.PC2P_ENABLED):
        raise RuntimeError("real-batch check requires PC2P")
    if any(
        (
            cfg.MODEL.TAPF.PICRD_ENABLED,
            cfg.MODEL.TAPF.PCHM_ENABLED,
            cfg.MODEL.TAPF.SEMANTIC_ENABLED,
            cfg.MODEL.TAPF.HIERARCHICAL,
            cfg.MODEL.TAPF.SPK_ENABLED,
            cfg.MODEL.TAPF.ELO_CUR_ENABLED,
        )
    ):
        raise RuntimeError("real-batch check requires isolated clean D0 PC2P")
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
    asset = PoseCompleteClipProxyBank(
        cfg.MODEL.TAPF.PC2P_BANK,
        cfg.MODEL.TAPF.PC2P_BANK_SHA256,
        cfg.MODEL.TAPF.MANIFEST_SHA256,
        num_classes,
    )
    asset.validate_dataset(
        train_loader.dataset.dataset, train_loader.dataset.pose_store
    )
    proxy = asset.to(device)
    if proxy.requires_grad:
        raise RuntimeError("PC2P proxy unexpectedly requires gradients")

    model, default_model = assert_default_off_exact(
        cfg, num_classes, camera_num, view_num
    )
    model = model.to(device)
    default_model = default_model.to(device)
    batch = next(iter(train_loader))
    image, identity, camera, view, pose = batch
    if image.shape[0] != 64 or identity.unique().numel() != 16:
        raise RuntimeError("real-batch sampler is not 16 identities x 4 images")
    counts = torch.stack([(identity == value).sum() for value in identity.unique()])
    if not bool((counts == 4).all()):
        raise RuntimeError("real-batch identity multiplicity is not four")
    image = image.to(device)
    identity = identity.to(device)
    camera = camera.to(device)
    view = view.to(device)
    pose_batch = {
        "keypoints": pose["keypoints"].to(device),
        "scores": pose["scores"].to(device),
        "valid": pose["valid"].to(device),
    }

    default_loss_fn, _ = make_loss(cfg, num_classes=num_classes)
    assert_default_off_forward_exact(
        model,
        default_model,
        image,
        identity,
        camera,
        view,
        pose_batch,
        default_loss_fn,
        cfg.MODEL.TAPF.POSE_LOSS_WEIGHT,
    )
    del default_model
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_feature, _ = model(
            image,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
    if eval_feature.shape != (64, 768) or not bool(
        torch.isfinite(eval_feature).all()
    ):
        raise RuntimeError("PC2P eval descriptor contract failed")

    model.train()
    named_parameters = list(model.named_parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in make_optimizer(
            cfg,
            model,
            make_loss(cfg, num_classes=num_classes)[1],
        )[0].param_groups
        for parameter in group["params"]
    }
    state = model.state_dict()
    if any("pc2p" in name.lower() or "proxy" in name.lower() for name in state):
        raise RuntimeError("PC2P proxy entered model state")
    if any(tensor.data_ptr() == proxy.data_ptr() for tensor in state.values()):
        raise RuntimeError("PC2P proxy storage entered model state")
    if id(proxy) in optimizer_parameter_ids:
        raise RuntimeError("PC2P proxy entered optimizer")

    model.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score, feature, _, tapf_aux = model(
            image,
            label=identity,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
            identity_proxy_bank=proxy,
        )
        ce = F.cross_entropy(score, identity)
    if score.shape != (64, 702) or score.dtype != torch.float32:
        raise RuntimeError("PC2P logit shape/dtype contract failed")
    if not bool(torch.isfinite(score).all()) or float(score.std()) <= 0:
        raise RuntimeError("PC2P logits are non-finite or constant")
    descriptor_gradient = torch.autograd.grad(ce, feature, retain_graph=True)[0]
    if not bool(torch.isfinite(descriptor_gradient).all()) or not bool(
        (descriptor_gradient != 0).any()
    ):
        raise RuntimeError("PC2P CE does not reach the global descriptor")
    ce.backward()
    if proxy.grad is not None:
        raise RuntimeError("PC2P proxy acquired a gradient")
    if model.classifier.weight.grad is not None and bool(
        (model.classifier.weight.grad != 0).any()
    ):
        raise RuntimeError("PC2P invoked the learned classifier")
    bn_active, bn_nonfinite = active_gradients(named_parameters, "bottleneck.")
    norm3_active, norm3_nonfinite = active_gradients(named_parameters, "base.norm3.")
    stage3_active, stage3_nonfinite = active_gradients(
        named_parameters, "base.stages.3."
    )
    if bn_nonfinite or norm3_nonfinite or stage3_nonfinite:
        raise RuntimeError("PC2P CE produced non-finite gradients")
    if not bn_active or not norm3_active or not stage3_active:
        raise RuntimeError("PC2P CE gradient route is incomplete")

    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
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
                identity_proxy_bank=proxy,
            )
            reid_loss = loss_fn(score, feature, identity, camera)
            loss = reid_loss + (
                cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux["pose_loss"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("PC2P combined loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        stage3, stage3_nonfinite = active_gradients(
            named_parameters, "base.stages.3."
        )
        backbone, backbone_nonfinite = active_gradients(named_parameters, "base.")
        if model.classifier.weight.grad is not None and bool(
            (model.classifier.weight.grad != 0).any()
        ):
            raise RuntimeError("combined PC2P loss invoked learned classifier")
        update_target = (
            max(
                stage3,
                key=lambda item: float(item[1].grad.detach().abs().max()),
            )
            if stage3
            else next(
                (item for item in named_parameters if item[0].startswith("base.stages.3.")),
                None,
            )
        )
        if update_target is None:
            raise RuntimeError("could not identify a Stage-3 update parameter")
        update_name, update_parameter = update_target
        before = update_parameter.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        overflow = scale_after < scale_before
        updated = not torch.equal(before, update_parameter.detach())
        attempts.append(
            {
                "attempt": attempt,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "overflow": overflow,
                "updated": updated,
            }
        )
        if overflow:
            if updated:
                raise RuntimeError("native GradScaler overflow semantics failed")
            continue
        if stage3_nonfinite or backbone_nonfinite:
            raise RuntimeError("finite GradScaler step retained non-finite gradients")
        if not stage3 or not backbone or not updated:
            raise RuntimeError("PC2P combined optimizer-step contract failed")
        success = {
            "attempt": attempt,
            "loss": float(loss.detach()),
            "reid_loss": float(reid_loss.detach()),
            "pose_loss": float(tapf_aux["pose_loss"].detach()),
            "stage3": len(stage3),
            "backbone": len(backbone),
            "updated_parameter": update_name,
        }
        break
    if success is None:
        raise RuntimeError("default GradScaler never reached a native update")

    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_after, _ = model(
            image,
            cam_label=camera,
            view_label=view,
            pose_batch=pose_batch,
            tapf_epoch=1,
        )
    if eval_after.shape != (64, 768) or not bool(torch.isfinite(eval_after).all()):
        raise RuntimeError("post-update PC2P eval descriptor contract failed")

    result = {
        "schema": "exp410-pc2p-real-batch-v1",
        "status": "PASS",
        "batch": 64,
        "identities": 16,
        "classes": 702,
        "default_off_state_rng_exact": True,
        "bank_sha256": asset.sha256,
        "bank_source_head": asset.source_head,
        "bank_in_model_state": False,
        "bank_in_optimizer": False,
        "learned_classifier_gradient": None,
        "logit_dtype": str(score.dtype),
        "logit_mean": float(tapf_aux["pc2p_logit_mean"]),
        "logit_std": float(tapf_aux["pc2p_logit_std"]),
        "logit_abs_max": float(tapf_aux["pc2p_logit_abs_max"]),
        "bn_norm": float(tapf_aux["pc2p_bn_norm"]),
        "ce_only_bn_grad_tensors": len(bn_active),
        "ce_only_norm3_grad_tensors": len(norm3_active),
        "ce_only_stage3_grad_tensors": len(stage3_active),
        "native_attempts": attempts,
        "first_successful_update": success["attempt"],
        "combined_loss": success["loss"],
        "combined_reid_loss": success["reid_loss"],
        "combined_pose_loss": success["pose_loss"],
        "combined_stage3_grad_tensors": success["stage3"],
        "combined_backbone_grad_tensors": success["backbone"],
        "updated_parameter": success["updated_parameter"],
        "eval_without_bank_shape": list(eval_after.shape),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
