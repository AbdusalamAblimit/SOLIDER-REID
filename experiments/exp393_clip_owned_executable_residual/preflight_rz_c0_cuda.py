#!/usr/bin/env python3
"""CUDA/AMP preflight for exp393 Phase A RZ-C0.

The preflight keeps the sealed Semantic C0 teacher, loss, recipe, and router
locations.  It verifies that the only implementation delta is the identity-safe
ReZero parameterization, then checks alpha-first activation, subsequent branch
gradients, retrieval-path bypass growth, strict reload, and RGB-only inference.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_cfg(node, prefix=""):
    output = {}
    for key, value in node.items():
        name = key if not prefix else prefix + "." + key
        if hasattr(value, "items"):
            output.update(flatten_cfg(value, name))
        else:
            output[name] = value
    return output


def config_differences(baseline, candidate):
    left = flatten_cfg(baseline)
    right = flatten_cfg(candidate)
    keys = sorted(set(left).union(right))
    return {
        key: {"baseline": left.get(key), "candidate": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def bypass_forward(self, tokens, hw_shape, mask, support):
    del hw_shape, mask, support
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


def eval_full_and_bypass(model, images):
    routers = tuple(model.base.tapf.psg_bank)
    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        full, _ = model(images)
        with bypass_routers(routers):
            bypass, _ = model(images)
    difference = (full.float() - bypass.float()).detach()
    return {
        "exact": bool(torch.equal(full, bypass)),
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(bypass).all()),
        "max_abs": float(difference.abs().max()),
        "mean_l2": float(difference.flatten(1).norm(dim=1).mean()),
    }


def null_identity(routers, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260719)
    tokens = torch.randn(3, 24, routers[0].feature_channels, generator=generator)
    masks = torch.rand(3, 5, 6, 4, generator=generator)
    support = torch.zeros(3, 5)
    tokens = tokens.to(device)
    masks = masks.to(device)
    support = support.to(device)
    values = []
    with torch.no_grad(), amp.autocast(enabled=True):
        for router in routers:
            routed, applied = router(tokens, (6, 4), masks, support)
            values.append({
                "tokens_exact": bool(torch.equal(routed, tokens)),
                "applied_exact_zero": bool(
                    torch.equal(applied, torch.zeros_like(applied))
                ),
                "finite": bool(
                    torch.isfinite(routed).all() and torch.isfinite(applied).all()
                ),
            })
    return values


def gradient_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--base-preflight", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 10:
        raise ValueError("At least ten steps are required")
    repo_root = Path(args.repo_root).resolve()
    os.chdir(str(repo_root))
    import sys

    sys.path.insert(0, str(repo_root))
    import open_clip
    from config import cfg as default_cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model import make_model
    from model.clip_semantic_teacher import FrozenClipSlotTeacher
    from solver import make_optimizer

    base = load_module("exp393_rz_base_preflight", args.base_preflight)
    candidate = default_cfg.clone()
    candidate.merge_from_file(str(Path(args.config).resolve()))
    candidate.freeze()
    baseline = default_cfg.clone()
    baseline.merge_from_file(str(Path(args.baseline_config).resolve()))
    baseline.freeze()
    differences = config_differences(baseline, candidate)
    expected_differences = {
        "MODEL.TAPF.SEMANTIC_REZERO",
        "OUTPUT_DIR",
    }
    if set(differences) != expected_differences:
        raise RuntimeError("Config is not single-variable: %s" % differences)
    if not candidate.MODEL.TAPF.SEMANTIC_REZERO:
        raise RuntimeError("Candidate ReZero switch is not enabled")
    if baseline.MODEL.TAPF.SEMANTIC_REZERO:
        raise RuntimeError("Baseline unexpectedly enables ReZero")
    if candidate.SOLVER.IMS_PER_BATCH != 64:
        raise RuntimeError("Formal batch size changed")
    formal_output = (repo_root / candidate.OUTPUT_DIR).resolve()
    if formal_output.exists():
        raise RuntimeError("Formal output already exists: %s" % formal_output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.__version__.startswith("1.13.1"):
        raise RuntimeError("Unexpected Torch runtime: %s" % torch.__version__)
    if open_clip.__version__ != "2.32.0":
        raise RuntimeError("Unexpected OpenCLIP runtime: %s" % open_clip.__version__)

    base.set_seed(candidate.SOLVER.SEED)
    train_loader, _, val_loader, _, num_classes, camera_num, view_num = (
        make_dataloader(candidate)
    )
    if train_loader.batch_size != 64 or train_loader.num_workers != 8:
        raise RuntimeError("Loader does not preserve batch64/8 workers")

    base.set_seed(candidate.SOLVER.SEED)
    baseline_model = make_model(
        baseline,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=baseline.MODEL.SEMANTIC_WEIGHT,
    )
    baseline_state = baseline_model.state_dict()
    baseline_parameters = sum(p.numel() for p in baseline_model.parameters())
    base.set_seed(candidate.SOLVER.SEED)
    model = make_model(
        candidate,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=candidate.MODEL.SEMANTIC_WEIGHT,
    )
    candidate_parameters = sum(p.numel() for p in model.parameters())
    candidate_state = model.state_dict()
    nonvariable_mismatches = []
    for key, value in baseline_state.items():
        if key.endswith(".expert"):
            continue
        if key not in candidate_state or not torch.equal(value, candidate_state[key]):
            nonvariable_mismatches.append(key)
    extra_keys = sorted(set(candidate_state).difference(baseline_state))
    routers = tuple(model.base.tapf.psg_bank)
    parameter_contract = {
        "baseline_parameters": baseline_parameters,
        "candidate_parameters": candidate_parameters,
        "difference": candidate_parameters - baseline_parameters,
        "extra_state_keys": extra_keys,
        "nonvariable_mismatches": nonvariable_mismatches,
        "baseline_expert_exact_zero": all(
            bool(torch.equal(router.expert, torch.zeros_like(router.expert)))
            for router in baseline_model.base.tapf.psg_bank
        ),
        "candidate_expert_nonzero": all(
            bool((router.expert != 0).any()) for router in routers
        ),
        "candidate_expert_std": [
            float(router.expert.detach().double().std(unbiased=False))
            for router in routers
        ],
        "alpha_exact_zero": all(float(router.alpha_logit) == 0.0 for router in routers),
    }
    del baseline_model, baseline_state, candidate_state

    device = torch.device("cuda", 0)
    model = model.to(device)
    routers = tuple(model.base.tapf.psg_bank)
    teacher = FrozenClipSlotTeacher(
        checkpoint=candidate.MODEL.TAPF.CLIP_CHECKPOINT,
        checkpoint_sha256=candidate.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
        device=device,
        microbatch=candidate.MODEL.TAPF.CLIP_MICROBATCH,
    )
    loss_fn, center_criterion = make_loss(candidate, num_classes=num_classes)
    optimizer, _ = make_optimizer(candidate, model, center_criterion)
    teacher_parameter_ids = {id(p) for p in teacher.visual.parameters()}
    model_parameter_ids = {id(p) for p in model.parameters()}
    optimizer_parameter_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    state_keys = tuple(model.state_dict())
    teacher_isolated = (
        model_parameter_ids.isdisjoint(teacher_parameter_ids)
        and optimizer_parameter_ids.isdisjoint(teacher_parameter_ids)
        and not any("clip" in key.lower() or "teacher" in key.lower() for key in state_keys)
    )

    validation = next(iter(val_loader))[0][:32].to(device)
    initial_gap = eval_full_and_bypass(model, validation)
    initial_null = null_identity(routers, device)
    if not initial_gap["exact"]:
        raise RuntimeError("Initialization is not exact identity")
    if not all(item["tokens_exact"] for item in initial_null):
        raise RuntimeError("Initial NULL route is not exact identity")

    phase0b = load_module("exp393_rz_phase0b", args.phase0b_script)
    ontology = load_module("exp393_rz_ontology", args.ontology_script)
    pcmbcls = load_module("exp393_rz_pcmbcls", args.pcmbcls_script)
    scaler = amp.GradScaler()
    iterator = iter(train_loader)
    epoch_schedule = (1, 6, 10, 11)
    q_batches = []
    valid_batches = []
    records = []
    finite_updates = 0
    active_branch_updates = 0
    consecutive_active = 0
    longest_active = 0
    first_finite = None
    parity = None
    loss_history = []
    scale_history = []
    gap_history = [{"finite_updates": 0, **initial_gap}]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    model.train()
    for step in range(args.steps):
        batch, iterator = base.next_batch(iterator, train_loader)
        img, vid, target_cam, target_view, raw_pose = batch
        pose = base.move_pose_batch(raw_pose, device)
        targets = base.make_teacher_targets(teacher, pose)
        if parity is None:
            parity = base.reference_parity(
                teacher, targets, pose, phase0b, ontology, pcmbcls
            )
        q_batches.append(targets["q_visible"].detach().cpu())
        valid_batches.append(targets["valid"].detach().cpu())
        img = img.to(device)
        target = vid.to(device)
        target_cam = target_cam.to(device)
        target_view = target_view.to(device)
        optimizer.zero_grad(set_to_none=True)
        epoch = epoch_schedule[(step * len(epoch_schedule)) // args.steps]
        scale_before = float(scaler.get_scale())
        probes = {
            "q_head": model.base.tapf.anchor.support_head.weight,
            "backbone": model.base.patch_embed.projection.weight,
            "head": model.classifier.weight,
        }
        for index, router in enumerate(routers):
            probes.update({
                "alpha_%d" % index: router.alpha_logit,
                "token_%d" % index: router.token_projection.weight,
                "context_%d" % index: router.context_projection.weight,
                "expert_%d" % index: router.expert,
            })
        before = {name: p.detach().clone() for name, p in probes.items()}
        with amp.autocast(enabled=True):
            score, feature, _, aux = model(
                img,
                label=target,
                cam_label=target_cam,
                view_label=target_view,
                pose_batch=pose,
                tapf_epoch=epoch,
            )
            reid_loss = loss_fn(score, feature, target, target_cam)
            loss = reid_loss + candidate.MODEL.TAPF.POSE_LOSS_WEIGHT * aux["pose_loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss at step %d" % step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        found_inf = float(sum(
            value.detach().float().item()
            for value in scaler._per_optimizer_states[id(optimizer)][
                "found_inf_per_device"
            ].values()
        ))
        gradients = {name: gradient_norm(p) for name, p in probes.items()}
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        updated = {
            name: not torch.equal(p.detach(), before[name])
            for name, p in probes.items()
        }
        branch_names = [
            "%s_%d" % (kind, index)
            for index in range(2)
            for kind in ("token", "context", "expert")
        ]
        alpha_names = ["alpha_0", "alpha_1"]
        if found_inf > 0.0:
            if scale_after >= scale_before or any(updated.values()):
                raise RuntimeError("Overflow step failed exact skip")
            consecutive_active = 0
        else:
            if scale_after != scale_before:
                raise RuntimeError("Finite step changed GradScaler scale")
            if not all(gradients[name] > 0.0 for name in alpha_names):
                raise RuntimeError("Finite step missed alpha gradient")
            if not all(updated[name] for name in alpha_names + ["q_head", "backbone", "head"]):
                raise RuntimeError("Finite step missed mandatory update")
            finite_updates += 1
            if first_finite is None:
                first_finite = {
                    "step": step + 1,
                    "gradients": gradients,
                    "branch_gradients_exact_zero": all(
                        gradients[name] == 0.0 for name in branch_names
                    ),
                    "alpha_before_exact_zero": all(
                        float(before[name]) == 0.0 for name in alpha_names
                    ),
                }
            else:
                active = all(
                    gradients[name] > 0.0 and updated[name]
                    for name in branch_names
                )
                if not active:
                    raise RuntimeError("Post-alpha finite step missed branch activation")
                active_branch_updates += 1
                consecutive_active += 1
                longest_active = max(longest_active, consecutive_active)
            if finite_updates in (1, 4, 8, 12, 16, 20):
                gap_history.append({
                    "finite_updates": finite_updates,
                    **eval_full_and_bypass(model, validation),
                })
                model.train()
        loss_history.append(float(loss.detach()))
        scale_history.append(scale_after)
        records.append({
            "step": step + 1,
            "epoch_route": epoch,
            "loss": float(loss.detach()),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "gradients": gradients,
            "updated": updated,
            "alpha_values": [float(router.alpha_logit.detach()) for router in routers],
            "applied_delta_abs_mean": [
                float(delta.detach().float().abs().mean())
                for delta in aux["gate_deltas"]
            ],
        })
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    final_gap = eval_full_and_bypass(model, validation)
    if not gap_history or gap_history[-1].get("finite_updates") != finite_updates:
        gap_history.append({"finite_updates": finite_updates, **final_gap})
    final_null = null_identity(routers, device)

    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_feature, _ = model(validation)
    eval_rgb_only_finite = bool(torch.isfinite(eval_feature).all())
    state = model.state_dict()
    state_finite = all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in state.values()
    )
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        checkpoint_path = Path(handle.name)
    try:
        torch.save(state, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        reloaded = make_model(
            candidate,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            semantic_weight=candidate.MODEL.SEMANTIC_WEIGHT,
        )
        reloaded.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        strict_reload = True
    finally:
        checkpoint_path.unlink(missing_ok=True)

    q_summary = base.summarize_q(q_batches, valid_batches)
    gates = {
        "single_variable_config": set(differences) == expected_differences,
        "single_variable_state": (
            not nonvariable_mismatches
            and extra_keys == [
                "base.tapf.psg_bank.0.alpha_logit",
                "base.tapf.psg_bank.1.alpha_logit",
            ]
            and parameter_contract["difference"] == 2
            and parameter_contract["baseline_expert_exact_zero"]
            and parameter_contract["candidate_expert_nonzero"]
        ),
        "initial_identity_exact": initial_gap["exact"],
        "null_identity_exact": all(
            item["tokens_exact"] and item["applied_exact_zero"]
            for item in initial_null + final_null
        ),
        "first_finite_alpha_only": (
            first_finite is not None
            and first_finite["alpha_before_exact_zero"]
            and first_finite["branch_gradients_exact_zero"]
            and all(
                first_finite["gradients"][name] > 0.0
                for name in ("alpha_0", "alpha_1")
            )
        ),
        "eight_active_branch_updates": longest_active >= 8,
        "route_gap_grew": (
            initial_gap["max_abs"] == 0.0
            and final_gap["finite"]
            and final_gap["max_abs"] > 0.0
            and final_gap["mean_l2"] > 0.0
        ),
        "teacher_parity_isolation": bool(parity["pass"]) and teacher_isolated,
        "finite_amp": (
            finite_updates >= 9
            and all(np.isfinite(loss_history))
            and all(np.isfinite(scale_history))
            and state_finite
        ),
        "rgb_only_eval": eval_rgb_only_finite,
        "checkpoint_strict": strict_reload,
        "memory": peak_reserved < 24 * 1024 ** 3,
        "loader_recipe": train_loader.batch_size == 64 and train_loader.num_workers == 8,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "EXP393_PHASE_A_RZ_C0_CUDA_PREFLIGHT",
        "formal_training_authorized": all(gates.values()),
        "gates": gates,
        "config_differences": differences,
        "parameter_contract": parameter_contract,
        "initial_gap": initial_gap,
        "final_gap": final_gap,
        "gap_history": gap_history,
        "initial_null": initial_null,
        "final_null": final_null,
        "first_finite": first_finite,
        "finite_updates": finite_updates,
        "active_branch_updates": active_branch_updates,
        "longest_active_branch_updates": longest_active,
        "overflow_steps": args.steps - finite_updates,
        "records": records,
        "parity": parity,
        "q_slots": q_summary,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.steps * train_loader.batch_size / elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "state_tensor_count": len(state),
        "state_finite": state_finite,
        "teacher_isolated": teacher_isolated,
        "checkpoint_sha256": checkpoint_sha,
        "torch_version": torch.__version__,
        "open_clip_version": open_clip.__version__,
        "config_sha256": sha256_file(args.config),
        "baseline_config_sha256": sha256_file(args.baseline_config),
        "script_sha256": sha256_file(__file__),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
