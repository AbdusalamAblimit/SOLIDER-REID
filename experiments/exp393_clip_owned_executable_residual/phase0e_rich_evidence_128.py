#!/usr/bin/env python3
"""128-image stability audit for exp393 Phase 0E rich CLIP evidence.

This is a frozen teacher-only audit.  It reuses the sealed PC-MBCLS and C8
contracts, fits slot means/shared PCA on 64 PID-disjoint images, and adjudicates
variance, rank, and paired counterfactual margins on 64 held-out PIDs.  It never
builds a ReID model, optimizer, training config, or checkpoint.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGIONS = 5
IMAGES = 128
FIT_IMAGES = 64
AUDIT_IMAGES = IMAGES - FIT_IMAGES
COMPONENTS = 16
EXPECTED_PHASE0B_SHA256 = (
    "03b8f707bc6f189dd3de34505af82e63f7ee71bd23d70b6e9663aee318afcd70"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca"
)
EXPECTED_PCMBCLS_SHA256 = (
    "7206dc13bf69b5666b54169ae3333f838c48b16d0c963512e7c67d906354c2c7"
)
EXPECTED_STATIC_SHA256 = (
    "6c1b370912f5f668ce117d4320d62b68a032549ff06821f5bee1ae020acb3dab"
)
EXPECTED_C8_SHA256 = (
    "ab36357174fbf2f2181bcfbaefb71d5a47d0b55de901603c3d2e475a2bd32569"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_selection(records, recipient_dataset, static, seed):
    ordered = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            (
                str(seed) + "\0" + Path(records[index][0]).name
            ).encode("utf-8")
        ).hexdigest(),
    )
    unique_pids = torch.tensor(
        sorted({int(record[1]) for record in records}), dtype=torch.int64
    )
    pid_fit, _ = static.pid_disjoint_split(unique_pids, seed)
    fit_pid_set = set(unique_pids[pid_fit].tolist())
    selected = []
    items = []
    selected_pids = set()
    fit_count = 0
    audit_count = 0
    for index in ordered:
        pid = int(records[index][1])
        if pid in selected_pids:
            continue
        is_fit = pid in fit_pid_set
        if is_fit and fit_count >= FIT_IMAGES:
            continue
        if (not is_fit) and audit_count >= AUDIT_IMAGES:
            continue
        item = recipient_dataset[index]
        if not bool(item["valid"].all()):
            continue
        selected.append(index)
        items.append(item)
        selected_pids.add(pid)
        fit_count += int(is_fit)
        audit_count += int(not is_fit)
        if fit_count == FIT_IMAGES and audit_count == AUDIT_IMAGES:
            break
    if len(selected) != IMAGES:
        raise RuntimeError("Could not select fixed 128-image PID-disjoint audit")
    return selected, items


def clustered_bootstrap(values, pids, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    pids = np.asarray(pids, dtype=np.int64)
    if values.ndim != 1 or pids.shape != values.shape or len(values) == 0:
        raise ValueError("invalid clustered bootstrap inputs")
    unique = np.unique(pids)
    per_pid = np.asarray(
        [values[pids == pid].mean() for pid in unique], dtype=np.float64
    )
    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sample = rng.randint(0, len(per_pid), size=len(per_pid))
        draws[index] = per_pid[sample].mean()
    return {
        "mean": float(per_pid.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "pids": int(len(per_pid)),
        "observations": int(len(values)),
        "repeats": int(repeats),
    }


def cosine_margin(correct, positive, negative):
    correct = F.normalize(correct.double(), dim=-1)
    positive = F.normalize(positive.double(), dim=-1)
    negative = F.normalize(negative.double(), dim=-1)
    return (
        (correct * positive).sum(-1)
        - (correct * negative).sum(-1)
    )


def per_slot_margins(
    correct,
    positive,
    negative,
    valid,
    pids,
    repeats,
    seed,
):
    output = []
    for region in range(REGIONS):
        keep = valid[:, region]
        values = cosine_margin(
            correct[keep, region],
            positive[keep, region],
            negative[keep, region],
        )
        output.append(clustered_bootstrap(
            values.cpu().numpy(),
            pids[keep].cpu().numpy(),
            repeats,
            seed + region,
        ))
    return output


def rank_and_variance(code, audit, valid, static):
    per_slot = []
    for region in range(REGIONS):
        keep = audit & valid[:, region]
        selected = code[keep, region].double()
        standard_deviation = selected.std(dim=0, unbiased=False)
        rank = static.effective_rank(selected)
        per_slot.append({
            "slot": region,
            "images": int(keep.sum()),
            "std_min": float(standard_deviation.min()),
            "std_median": float(standard_deviation.median()),
            "std_max": float(standard_deviation.max()),
            "nonzero_dimensions": int((standard_deviation > 1e-8).sum()),
            **rank,
        })
    return per_slot


def transform_variant(static, encoded, valid, codebook):
    raw = static.raw_local_evidence(encoded["region"], encoded["global"])
    return static.transform_code(raw, valid, codebook)[0], raw


def random_codebook(codebook, dimensions, components, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        dimensions, components, generator=generator, dtype=torch.float64
    )
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return {
        "means": codebook["means"].clone(),
        "basis": q.T.contiguous(),
    }


def representation_summary(
    static,
    codebook,
    raw_correct,
    raw_positive,
    raw_wrong_rgb,
    raw_wrong_mask,
    valid,
    positive_valid,
    wrong_rgb_valid,
    wrong_mask_valid,
    audit,
):
    correct = static.transform_code(raw_correct, valid, codebook)[0]
    positive = static.transform_code(
        raw_positive, positive_valid, codebook
    )[0]
    wrong_rgb = static.transform_code(
        raw_wrong_rgb, wrong_rgb_valid, codebook
    )[0]
    wrong_mask = static.transform_code(
        raw_wrong_mask, wrong_mask_valid, codebook
    )[0]
    held = audit[:, None]
    rgb_keep = held & valid & positive_valid & wrong_rgb_valid
    mask_keep = held & valid & positive_valid & wrong_mask_valid
    rgb_means = []
    mask_means = []
    for region in range(REGIONS):
        rgb_values = cosine_margin(
            correct[rgb_keep[:, region], region],
            positive[rgb_keep[:, region], region],
            wrong_rgb[rgb_keep[:, region], region],
        )
        mask_values = cosine_margin(
            correct[mask_keep[:, region], region],
            positive[mask_keep[:, region], region],
            wrong_mask[mask_keep[:, region], region],
        )
        rgb_means.append(float(rgb_values.mean()))
        mask_means.append(float(mask_values.mean()))
    ranks = rank_and_variance(correct, audit, valid, static)
    return {
        "macro_effective_rank": float(np.mean([
            item["entropy_effective_rank"] for item in ranks
        ])),
        "wrong_rgb_margin_by_slot": rgb_means,
        "wrong_mask_margin_by_slot": mask_means,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--static-script", required=True)
    parser.add_argument("--c8-script", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    paths = {
        "phase0b": Path(args.phase0b_script).resolve(),
        "ontology": Path(args.ontology_script).resolve(),
        "pcmbcls": Path(args.pcmbcls_script).resolve(),
        "static": Path(args.static_script).resolve(),
        "c8": Path(args.c8_script).resolve(),
    }
    expected = {
        "phase0b": EXPECTED_PHASE0B_SHA256,
        "ontology": EXPECTED_ONTOLOGY_SHA256,
        "pcmbcls": EXPECTED_PCMBCLS_SHA256,
        "static": EXPECTED_STATIC_SHA256,
        "c8": EXPECTED_C8_SHA256,
    }
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise RuntimeError("Dependency SHA mismatch: %s" % key)
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.source_commit:
        raise RuntimeError("Source commit mismatch")

    base = load_module("exp393_128_base", paths["phase0b"])
    ontology = load_module("exp393_128_ontology", paths["ontology"])
    pcmbcls = load_module("exp393_128_pcmbcls", paths["pcmbcls"])
    static = load_module("exp393_128_static", paths["static"])
    c8 = load_module("exp393_128_c8", paths["c8"])
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    if sha256_file(pose_artifact / "manifest.json") != base.EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("Pose manifest SHA mismatch")
    if sha256_file(clip_checkpoint) != base.EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    runtime_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in base.EXPECTED_RUNTIME_SHA256
    }
    if runtime_sha != base.EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("Minimal runtime SHA mismatch")
    ontology_contract = ontology.ontology_static_contract(base, "hard-owner")
    if ontology_contract["status"] != "PASS":
        raise RuntimeError("Hard-owner ontology contract failed")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    base.set_seed(args.seed)
    dataset = OccludedDuke(root=str(Path(args.data_root).resolve()), verbose=False)
    records = list(dataset.train)
    if len(records) != 15618:
        raise RuntimeError("Unexpected official train size")
    pose_store = PoseTargetStore(
        pose_artifact, base.EXPECTED_POSE_MANIFEST_SHA256
    )
    recipient_dataset = ontology.RecipientDataset(
        base,
        records,
        pose_store,
        args.seed,
        verify_sha=True,
        partition_mode="hard-owner",
    )
    selected, selected_items = deterministic_selection(
        records, recipient_dataset, static, args.seed
    )
    recipients = ontology.collate_recipient(selected_items)
    donor_map, donor_summary = base.build_rgb_donor_map(records)
    donor_indices = [int(donor_map[index]) for index in selected]
    donor_items = [recipient_dataset[index] for index in donor_indices]
    donors = ontology.collate_recipient(donor_items)
    if bool((recipients["pid"] == donors["pid"]).any()):
        raise RuntimeError("Donor PID collision")

    device = torch.device(args.device)
    if device.type == "cuda":
        index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_device(index)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    teacher = c8.FrozenRichPCMBCLS(
        clip_checkpoint, device, pcmbcls, args.clip_batch
    )
    correct = teacher.encode(
        recipients["pre"], recipients["masks"],
        repeat_check=True, null_check=True,
    )
    positive = teacher.encode(
        recipients["pre"].flip(-1), recipients["masks"].flip(-1)
    )
    wrong_rgb = teacher.encode(donors["pre"], donors["masks"])
    donor_recipient = teacher.encode(donors["pre"], recipients["masks"])
    wrong_mask_masks = recipients["masks"].roll(shifts=-1, dims=1)
    wrong_mask = teacher.encode(recipients["pre"], wrong_mask_masks)
    peak_memory = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0
    )

    valid = recipients["valid"].bool() & correct["valid"].bool()
    positive_valid = valid & positive["valid"].bool()
    wrong_rgb_valid = valid & donors["valid"].bool() & wrong_rgb["valid"].bool()
    donor_recipient_valid = valid & donor_recipient["valid"].bool()
    wrong_mask_valid = valid & wrong_mask["valid"].bool()
    raw_correct = static.raw_local_evidence(
        correct["region"], correct["global"]
    )
    raw_positive = static.raw_local_evidence(
        positive["region"], positive["global"]
    )
    raw_wrong_rgb = static.raw_local_evidence(
        wrong_rgb["region"], wrong_rgb["global"]
    )
    raw_donor_recipient = static.raw_local_evidence(
        donor_recipient["region"], donor_recipient["global"]
    )
    raw_wrong_mask = static.raw_local_evidence(
        wrong_mask["region"], wrong_mask["global"]
    )
    codebook = static.fit_codebook(
        raw_correct,
        valid,
        recipients["pid"],
        args.seed,
        components=COMPONENTS,
    )
    correct_code = static.transform_code(
        raw_correct, valid, codebook
    )[0]
    positive_code = static.transform_code(
        raw_positive, positive_valid, codebook
    )[0]
    wrong_rgb_code = static.transform_code(
        raw_wrong_rgb, wrong_rgb_valid, codebook
    )[0]
    donor_recipient_code = static.transform_code(
        raw_donor_recipient, donor_recipient_valid, codebook
    )[0]
    wrong_mask_code = static.transform_code(
        raw_wrong_mask, wrong_mask_valid, codebook
    )[0]
    slot_cycle_code = correct_code.roll(shifts=-1, dims=1)
    audit = codebook["audit"]
    held = audit[:, None]

    rank_variance = rank_and_variance(
        correct_code, audit, valid, static
    )
    wrong_rgb_margins = per_slot_margins(
        correct_code,
        positive_code,
        wrong_rgb_code,
        held & valid & positive_valid & wrong_rgb_valid,
        recipients["pid"],
        args.bootstrap_repeats,
        args.seed + 100,
    )
    donor_recipient_margins = per_slot_margins(
        correct_code,
        positive_code,
        donor_recipient_code,
        held & valid & positive_valid & donor_recipient_valid,
        recipients["pid"],
        args.bootstrap_repeats,
        args.seed + 200,
    )
    wrong_mask_margins = per_slot_margins(
        correct_code,
        positive_code,
        wrong_mask_code,
        held & valid & positive_valid & wrong_mask_valid,
        recipients["pid"],
        args.bootstrap_repeats,
        args.seed + 300,
    )
    slot_cycle_margins = per_slot_margins(
        correct_code,
        positive_code,
        slot_cycle_code,
        held & valid & positive_valid,
        recipients["pid"],
        args.bootstrap_repeats,
        args.seed + 400,
    )

    uncentered_codebook = {
        "means": torch.zeros_like(codebook["means"]),
        "basis": codebook["basis"].clone(),
    }
    random_projection = random_codebook(
        codebook, raw_correct.shape[-1], COMPONENTS, args.seed + 999
    )
    controls = {
        "raw_uncentered": representation_summary(
            static,
            uncentered_codebook,
            raw_correct,
            raw_positive,
            raw_wrong_rgb,
            raw_wrong_mask,
            valid,
            positive_valid,
            wrong_rgb_valid,
            wrong_mask_valid,
            audit,
        ),
        "fixed_random_orthogonal": representation_summary(
            static,
            random_projection,
            raw_correct,
            raw_positive,
            raw_wrong_rgb,
            raw_wrong_mask,
            valid,
            positive_valid,
            wrong_rgb_valid,
            wrong_mask_valid,
            audit,
        ),
    }
    slot_mean_raw = codebook["means"][None].expand_as(raw_correct)
    slot_mean_code = static.transform_code(
        slot_mean_raw, valid, codebook
    )[0]
    global_only_raw = static.raw_local_evidence(
        correct["global"][:, None].expand_as(correct["region"]),
        correct["global"],
    )
    global_only_code = static.transform_code(
        global_only_raw, valid, {
            "means": torch.zeros_like(codebook["means"]),
            "basis": codebook["basis"],
        }
    )[0]
    controls.update({
        "slot_mean_code_abs_max": float(slot_mean_code.abs().max()),
        "slot_mean_code_variance_max": float(
            slot_mean_code.var(dim=0, unbiased=False).max()
        ),
        "global_only_raw_abs_max": float(global_only_raw.abs().max()),
        "global_only_code_abs_max": float(global_only_code.abs().max()),
    })

    selection_payload = {
        "seed": args.seed,
        "selected_indices": selected,
        "selected_paths": list(recipients["relative_path"]),
        "selected_pids": recipients["pid"].tolist(),
        "donor_indices": donor_indices,
        "donor_paths": list(donors["relative_path"]),
        "donor_pids": donors["pid"].tolist(),
        "fit_mask": codebook["fit"].tolist(),
        "audit_mask": codebook["audit"].tolist(),
    }
    codebook_payload = {
        "definition": (
            "normalize(region_cls)-normalize(global_cls); slot-center; "
            "shared PCA-16; L2 normalize"
        ),
        "selection": selection_payload,
        "selection_sha256": sha256_json(selection_payload),
        "slot_means": codebook["means"].tolist(),
        "shared_basis": codebook["basis"].tolist(),
        "fit_counts_by_slot": codebook["fit_counts"],
    }
    static.write_json(args.codebook_output, codebook_payload)
    codebook_sha = sha256_file(args.codebook_output)

    cycle_iou = c8.slot_cycle_iou(recipients["masks"])
    all_features_finite = all(
        bool(torch.isfinite(encoded[key]).all())
        for encoded in (
            correct, positive, wrong_rgb, donor_recipient, wrong_mask
        )
        for key in ("global", "region")
    )
    all_codes_finite = all(bool(torch.isfinite(value).all()) for value in (
        correct_code,
        positive_code,
        wrong_rgb_code,
        donor_recipient_code,
        wrong_mask_code,
        slot_cycle_code,
    ))
    macro_rank = float(np.mean([
        item["entropy_effective_rank"] for item in rank_variance
    ]))
    gates = {
        "fixed_pid_disjoint_selection": (
            len(selected) == IMAGES
            and len(set(recipients["pid"].tolist())) == IMAGES
            and int(codebook["fit"].sum()) == FIT_IMAGES
            and int(codebook["audit"].sum()) == AUDIT_IMAGES
            and not set(
                recipients["pid"][codebook["fit"]].tolist()
            ).intersection(
                recipients["pid"][codebook["audit"]].tolist()
            )
        ),
        "contract_parity_repeat_null": (
            correct["official_parity_max_abs"] <= 1e-6
            and correct["repeat_exact"]
            and correct["null_exact"]
            and correct["null_valid_exact"]
        ),
        "hard_owner_wrong_mask": (
            correct["pixel_product_max"] == 0.0
            and float(cycle_iou.max()) == 0.0
        ),
        "all_slot_all_dimension_nonzero_variance": all(
            item["nonzero_dimensions"] == COMPONENTS
            for item in rank_variance
        ),
        "macro_effective_rank_at_least_8": macro_rank >= 8.0,
        "wrong_rgb_ci_positive_all_slots": all(
            item["ci95_low"] > 0.0 for item in wrong_rgb_margins
        ),
        "wrong_mask_ci_positive_all_slots": all(
            item["ci95_low"] > 0.0 for item in wrong_mask_margins
        ),
        "static_global_controls_zero": (
            controls["slot_mean_code_abs_max"] == 0.0
            and controls["slot_mean_code_variance_max"] == 0.0
            and controls["global_only_raw_abs_max"] == 0.0
            and controls["global_only_code_abs_max"] == 0.0
        ),
        "frozen_no_grad": (
            teacher.all_parameters_frozen()
            and all(
                not encoded[key].requires_grad
                for encoded in (
                    correct, positive, wrong_rgb, donor_recipient, wrong_mask
                )
                for key in ("global", "region")
            )
        ),
        "finite": all_features_finite and all_codes_finite,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "EXP393_PHASE0E_128_STABILITY",
        "verdict": (
            "PHASE0E_128_PASS" if all(gates.values())
            else "PHASE0E_128_FAIL"
        ),
        "formal_training_authorized": False,
        "phase_b_authorized": False,
        "full_teacher_audit_authorized": all(gates.values()),
        "gates": gates,
        "rank_variance": rank_variance,
        "macro_effective_rank": macro_rank,
        "margins": {
            "correct_flip_vs_wrong_rgb_donor_donor": wrong_rgb_margins,
            "correct_flip_vs_donor_rgb_recipient_mask": donor_recipient_margins,
            "correct_flip_vs_same_rgb_wrong_mask": wrong_mask_margins,
            "correct_flip_vs_slot_cycle_binding": slot_cycle_margins,
        },
        "controls": controls,
        "checks": {
            "recipient_slots_all_valid": bool(valid.all()),
            "wrong_rgb_valid_by_slot": wrong_rgb_valid.sum(0).tolist(),
            "wrong_mask_cycle_iou_max": float(cycle_iou.max()),
            "hard_owner_pixel_product_max": correct["pixel_product_max"],
            "official_global_parity_max_abs": correct[
                "official_parity_max_abs"
            ],
            "repeat_exact": correct["repeat_exact"],
            "null_exact": correct["null_exact"],
            "null_valid_exact": correct["null_valid_exact"],
            "all_features_finite": all_features_finite,
            "all_codes_finite": all_codes_finite,
            "donor_different_pid": bool(
                (recipients["pid"] != donors["pid"]).all()
            ),
            "donor_no_fixed_point": all(
                left != right for left, right in zip(selected, donor_indices)
            ),
            "pcmbcls_seconds_total": float(sum(
                encoded["elapsed_seconds"]
                for encoded in (
                    correct, positive, wrong_rgb, donor_recipient, wrong_mask
                )
            )),
            "peak_memory_bytes": peak_memory,
            "codebook_sha256": codebook_sha,
            "selection_sha256": codebook_payload["selection_sha256"],
        },
        "execution": {
            "source_commit": actual_commit,
            "repo_root": str(repo_root),
            "device": str(device),
            "torch_version": torch.__version__,
            "open_clip_version": __import__("open_clip").__version__,
            "images": IMAGES,
            "fit_images": FIT_IMAGES,
            "audit_images": AUDIT_IMAGES,
            "clip_batch": args.clip_batch,
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "runtime_sha256": runtime_sha,
            "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint_sha256": base.EXPECTED_CLIP_SHA256,
            "dependency_sha256": {
                key: expected[key] for key in sorted(expected)
            },
            "script_sha256": sha256_file(__file__),
            "donor_summary": donor_summary,
        },
    }
    static.write_json(args.output, result)
    summary = {
        "status": result["status"],
        "verdict": result["verdict"],
        "gates": gates,
        "macro_effective_rank": macro_rank,
        "rank_variance": rank_variance,
        "margins": result["margins"],
        "controls": controls,
        "checks": result["checks"],
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
