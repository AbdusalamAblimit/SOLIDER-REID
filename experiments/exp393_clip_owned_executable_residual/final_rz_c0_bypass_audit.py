#!/usr/bin/env python3
"""Strict final and all-router-bypass audit for exp393 Phase A RZ-C0."""

import argparse
import contextlib
import hashlib
import json
import random
import time
import types
from pathlib import Path

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.metrics import R1_mAP_eval


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bypass_forward(self, tokens, hw_shape, mask, support):
    del self, hw_shape, mask, support
    return tokens, torch.zeros_like(tokens)


@contextlib.contextmanager
def bypass_routers(routers):
    for router in routers:
        if "forward" in router.__dict__:
            raise RuntimeError("Router already has an instance forward override")
        router.forward = types.MethodType(bypass_forward, router)
    try:
        yield
    finally:
        for router in routers:
            del router.forward


def normalize_checkpoint(checkpoint):
    return {
        name.replace("module.", "", 1): tensor
        for name, tensor in checkpoint.items()
    }


def checkpoint_audit(checkpoint):
    floating = 0
    nonfinite = []
    for name, tensor in checkpoint.items():
        if tensor.is_floating_point() or tensor.is_complex():
            floating += 1
            if not torch.isfinite(tensor).all():
                nonfinite.append(name)
    banned_tokens = ("clip", "open_clip", "teacher", "text_encoder", "pose_model")
    banned = [
        name for name in checkpoint
        if any(token in name.lower() for token in banned_tokens)
    ]
    if nonfinite:
        raise RuntimeError("Non-finite checkpoint tensors: {}".format(nonfinite))
    if banned:
        raise RuntimeError("Teacher-side tensors leaked into checkpoint: {}".format(banned))
    return {
        "state_tensors": len(checkpoint),
        "floating_tensors": floating,
        "nonfinite": nonfinite,
        "teacher_side_keys": banned,
    }


def parameter_trajectory(model, initial):
    prefixes = {
        "anchor": "base.tapf.anchor.",
        "router0_token": "base.tapf.psg_bank.0.token_projection.",
        "router0_context": "base.tapf.psg_bank.0.context_projection.",
        "router0_expert": "base.tapf.psg_bank.0.expert",
        "router0_alpha": "base.tapf.psg_bank.0.alpha_logit",
        "router1_token": "base.tapf.psg_bank.1.token_projection.",
        "router1_context": "base.tapf.psg_bank.1.context_projection.",
        "router1_expert": "base.tapf.psg_bank.1.expert",
        "router1_alpha": "base.tapf.psg_bank.1.alpha_logit",
    }
    result = {name: {"changed": 0, "total": 0} for name in prefixes}
    for name, parameter in model.named_parameters():
        for group, prefix in prefixes.items():
            if name.startswith(prefix):
                result[group]["total"] += 1
                result[group]["changed"] += int(
                    not torch.equal(parameter.detach().cpu(), initial[name])
                )
                break
    failed = {
        name: stats for name, stats in result.items()
        if stats["total"] == 0 or stats["changed"] != stats["total"]
    }
    if failed:
        raise RuntimeError("Incomplete parameter trajectory: {}".format(failed))
    return result


def router_state(routers):
    records = []
    for index, router in enumerate(routers):
        alpha = float(router.alpha_logit.detach().cpu())
        record = {
            "index": index,
            "alpha_logit": alpha,
            "tanh_alpha": float(torch.tanh(router.alpha_logit.detach()).cpu()),
            "expert_norm": float(router.expert.detach().float().norm().cpu()),
            "token_projection_norm": float(
                router.token_projection.weight.detach().float().norm().cpu()
            ),
            "context_projection_norm": float(
                router.context_projection.weight.detach().float().norm().cpu()
            ),
        }
        if not all(np.isfinite(value) for key, value in record.items() if key != "index"):
            raise RuntimeError("Non-finite router state: {}".format(record))
        if alpha == 0.0:
            raise RuntimeError("Router alpha remained exactly zero: {}".format(index))
        records.append(record)
    equal = all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for left, right in zip(routers[0].parameters(), routers[1].parameters())
    )
    if equal:
        raise RuntimeError("The two trained routers are identical")
    return records, equal


def null_identity(routers):
    generator = torch.Generator().manual_seed(20260719)
    tokens = torch.randn(3, 24, routers[0].feature_channels, generator=generator).cuda()
    masks = torch.rand(3, 5, 6, 4, generator=generator).cuda()
    support = torch.zeros(3, 5, device="cuda")
    records = []
    with torch.no_grad():
        for index, router in enumerate(routers):
            routed, applied = router(tokens, (6, 4), masks, support)
            record = {
                "index": index,
                "tokens_exact": bool(torch.equal(routed, tokens)),
                "applied_exact_zero": bool(
                    torch.equal(applied, torch.zeros_like(applied))
                ),
                "finite": bool(
                    torch.isfinite(routed).all() and torch.isfinite(applied).all()
                ),
            }
            if not all(record[key] for key in ("tokens_exact", "applied_exact_zero", "finite")):
                raise RuntimeError("NULL identity failed: {}".format(record))
            records.append(record)
    return records


def descriptor_gap(model, routers):
    generator = torch.Generator().manual_seed(20260719)
    images = torch.randn(4, 3, 384, 128, generator=generator).cuda()
    model.eval()
    with torch.no_grad():
        full, _ = model(images)
        with bypass_routers(routers):
            bypass, _ = model(images)
    difference = full.float() - bypass.float()
    record = {
        "full_finite": bool(torch.isfinite(full).all()),
        "bypass_finite": bool(torch.isfinite(bypass).all()),
        "exact": bool(torch.equal(full, bypass)),
        "max_abs": float(difference.abs().max().cpu()),
        "mean_l2": float(difference.flatten(1).norm(dim=1).mean().cpu()),
    }
    if not record["full_finite"] or not record["bypass_finite"]:
        raise RuntimeError("Non-finite synthetic descriptor")
    if record["exact"] or record["max_abs"] <= 0.0:
        raise RuntimeError("Trained route remains descriptor-inert")
    return record


def evaluate(model, val_loader, num_query):
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    batches = 0
    samples = 0
    model.eval()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in val_loader:
            img = img.cuda(non_blocking=True)
            camids = camids.cuda(non_blocking=True)
            target_view = target_view.cuda(non_blocking=True)
            feat, _ = model(img, cam_label=camids, view_label=target_view)
            if not torch.isfinite(feat).all():
                raise RuntimeError("Non-finite evaluation descriptor")
            evaluator.update((feat, pid, camid))
            batches += 1
            samples += int(img.shape[0])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cmc, mAP, _, _, _, query, gallery = evaluator.compute()
    return {
        "mAP": float(mAP),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "batches": batches,
        "samples": samples,
        "query_shape": list(query.shape),
        "gallery_shape": list(gallery.shape),
        "seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def rounded_percent(value):
    return round(value * 100.0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    loaders = make_dataloader(cfg)
    _, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = loaders
    if hasattr(train_loader_normal.dataset, "pose_store") or hasattr(val_loader.dataset, "pose_store"):
        raise RuntimeError("RGB-only evaluator unexpectedly owns a pose store")

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    checkpoint = normalize_checkpoint(torch.load(args.checkpoint, map_location="cpu"))
    checkpoint_record = checkpoint_audit(checkpoint)
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load failed")
    trajectory = parameter_trajectory(model, initial)
    routers = tuple(model.base.tapf.psg_bank)
    router_records, routers_equal = router_state(routers)

    model = model.cuda().eval()
    null_records = null_identity(routers)
    synthetic_gap = descriptor_gap(model, routers)
    full = evaluate(model, val_loader, num_query)
    with bypass_routers(routers):
        bypass = evaluate(model, val_loader, num_query)

    expected_full = [56.8, 66.8, 79.6, 83.9]
    reproduced = [
        rounded_percent(full[key])
        for key in ("mAP", "rank1", "rank5", "rank10")
    ] == expected_full
    map_gap_points = (full["mAP"] - bypass["mAP"]) * 100.0
    full_floor_pass = rounded_percent(full["mAP"]) >= 56.7
    route_alive = map_gap_points >= 0.1
    verdict = (
        "EXP393_RZ_C0_ROUTE_ALIVE_PASS"
        if reproduced and full_floor_pass and route_alive
        else "EXP393_RZ_C0_ROUTE_ALIVE_FAIL"
    )
    result = {
        "status": verdict,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint": checkpoint_record,
        "strict_missing": list(incompatible.missing_keys),
        "strict_unexpected": list(incompatible.unexpected_keys),
        "parameter_trajectory": trajectory,
        "router_state": router_records,
        "routers_equal": routers_equal,
        "null_identity": null_records,
        "synthetic_full_bypass_gap": synthetic_gap,
        "dataset_boundary": {
            "num_query": num_query,
            "normal_train_has_pose_store": hasattr(train_loader_normal.dataset, "pose_store"),
            "validation_has_pose_store": hasattr(val_loader.dataset, "pose_store"),
        },
        "full": full,
        "all_router_bypass": bypass,
        "full_rounded_percent": reproduced and expected_full or [
            rounded_percent(full[key])
            for key in ("mAP", "rank1", "rank5", "rank10")
        ],
        "bypass_rounded_percent": [
            rounded_percent(bypass[key])
            for key in ("mAP", "rank1", "rank5", "rank10")
        ],
        "full_reproduced": reproduced,
        "full_floor_pass": full_floor_pass,
        "full_minus_bypass_map_points": map_gap_points,
        "route_alive": route_alive,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
