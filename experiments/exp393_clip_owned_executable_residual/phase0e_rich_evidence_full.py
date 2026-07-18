#!/usr/bin/env python3
"""Full official-train teacher-only audit for exp393 Phase 0E.

The script streams all 15,618 clean Occluded-Duke train images.  Pass one
caches only correct raw CLIP evidence, fits slot means and a shared PCA-16 on
PID-disjoint fit identities with a covariance/eigh implementation, and never
keeps RGB tensors for the full dataset.  Pass two encodes counterfactuals only
for held-out identities and adjudicates PID-clustered margins.
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


REGIONS = 5
DIMENSIONS = 768
COMPONENTS = 16
OFFICIAL_IMAGES = 15618
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
EXPECTED_AUDIT128_SHA256 = (
    "deae5c9308650f9f9344ab19e0e78fa78b193a53244e41ccc24d9274fbd1526a"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--static-script", required=True)
    parser.add_argument("--c8-script", required=True)
    parser.add_argument("--audit128-script", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition-output", required=True)
    parser.add_argument("--codebook-output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch", type=int, default=4)
    parser.add_argument("--image-chunk", type=int, default=32)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--streaming-contract-only", action="store_true")
    return parser.parse_args()


def iter_chunks(indices, size):
    indices = list(indices)
    for start in range(0, len(indices), int(size)):
        yield indices[start:start + int(size)]


def collate_indices(dataset, indices, ontology):
    return ontology.collate_recipient([dataset[index] for index in indices])


def open_memmaps(cache_dir, mode):
    cache_dir = Path(cache_dir)
    return {
        "raw": np.memmap(
            cache_dir / "correct_raw.f32",
            dtype=np.float32,
            mode=mode,
            shape=(OFFICIAL_IMAGES, REGIONS, DIMENSIONS),
        ),
        "valid": np.memmap(
            cache_dir / "correct_valid.u8",
            dtype=np.uint8,
            mode=mode,
            shape=(OFFICIAL_IMAGES, REGIONS),
        ),
    }


def fit_streaming_codebook(raw_map, valid_map, fit, static, chunk_size):
    means = torch.zeros(REGIONS, DIMENSIONS, dtype=torch.float64)
    counts = torch.zeros(REGIONS, dtype=torch.int64)
    fit_indices = torch.nonzero(fit, as_tuple=False).flatten().tolist()
    for indices in iter_chunks(fit_indices, chunk_size):
        raw = torch.from_numpy(np.array(raw_map[indices], copy=True)).double()
        valid = torch.from_numpy(np.array(valid_map[indices], copy=True)).bool()
        for region in range(REGIONS):
            keep = valid[:, region]
            means[region] += raw[keep, region].sum(0)
            counts[region] += int(keep.sum())
    if bool((counts < 2).any()):
        raise RuntimeError("Insufficient fit samples for a slot")
    means /= counts[:, None]

    covariance = torch.zeros(
        DIMENSIONS, DIMENSIONS, dtype=torch.float64
    )
    centered_rows = 0
    center_sum_max = 0.0
    center_sums = torch.zeros_like(means)
    for indices in iter_chunks(fit_indices, chunk_size):
        raw = torch.from_numpy(np.array(raw_map[indices], copy=True)).double()
        valid = torch.from_numpy(np.array(valid_map[indices], copy=True)).bool()
        for region in range(REGIONS):
            keep = valid[:, region]
            centered = raw[keep, region] - means[region]
            covariance.addmm_(centered.T, centered)
            center_sums[region] += centered.sum(0)
            centered_rows += int(keep.sum())
    center_sum_max = float(
        (center_sums / counts[:, None]).abs().max()
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -COMPONENTS:].T.flip(0).contiguous()
    basis = static.canonicalize_basis_sign(basis)
    orthogonal_error = float(
        (basis @ basis.T - torch.eye(COMPONENTS)).abs().max()
    )
    return {
        "means": means,
        "basis": basis,
        "fit": fit,
        "audit": ~fit,
        "fit_counts": counts.tolist(),
        "fit_center_mean_abs_max": center_sum_max,
        "basis_orthogonal_max_abs": orthogonal_error,
        "covariance_rows": centered_rows,
        "top_eigenvalues": eigenvalues[-COMPONENTS:].flip(0).tolist(),
    }


def streaming_fit_contract(static, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    samples = 40
    raw = torch.randn(
        samples, REGIONS, DIMENSIONS,
        generator=generator, dtype=torch.float64,
    )
    valid = torch.ones(samples, REGIONS, dtype=torch.bool)
    pids = torch.arange(samples, dtype=torch.int64)
    fit, _ = static.pid_disjoint_split(pids, seed)
    codebook = fit_streaming_codebook(
        raw.numpy(), valid.numpy().astype(np.uint8), fit, static, 7
    )
    direct_rows = []
    direct_means = torch.zeros_like(codebook["means"])
    for region in range(REGIONS):
        direct_means[region] = raw[fit, region].mean(0)
        direct_rows.append(raw[fit, region] - direct_means[region])
    _, _, vh = torch.linalg.svd(
        torch.cat(direct_rows), full_matrices=False
    )
    direct_basis = vh[:COMPONENTS]
    subspace_error = float((
        codebook["basis"].T @ codebook["basis"]
        - direct_basis.T @ direct_basis
    ).abs().max())
    gates = {
        "means_match_direct": bool(torch.allclose(
            codebook["means"], direct_means, atol=1e-12, rtol=0.0
        )),
        "fit_centered": codebook["fit_center_mean_abs_max"] <= 1e-12,
        "basis_orthogonal": codebook["basis_orthogonal_max_abs"] <= 1e-10,
        "covariance_eigh_matches_direct_svd_subspace": subspace_error <= 1e-10,
        "finite": bool(
            torch.isfinite(codebook["means"]).all()
            and torch.isfinite(codebook["basis"]).all()
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "fit_images": int(fit.sum()),
        "fit_rows": codebook["covariance_rows"],
        "subspace_max_abs": subspace_error,
        "center_mean_abs_max": codebook["fit_center_mean_abs_max"],
        "orthogonal_max_abs": codebook["basis_orthogonal_max_abs"],
    }


def random_codebook(codebook, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        DIMENSIONS, COMPONENTS, generator=generator, dtype=torch.float64
    )
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return {"means": codebook["means"].clone(), "basis": q.T.contiguous()}


def transform(raw, valid, codebook, static):
    return static.transform_code(raw, valid, codebook)[0]


def effective_rank_by_slot(code, valid, static):
    output = []
    for region in range(REGIONS):
        selected = code[valid[:, region], region].double()
        standard_deviation = selected.std(dim=0, unbiased=False)
        output.append({
            "slot": region,
            "images": int(len(selected)),
            "std_min": float(standard_deviation.min()),
            "std_median": float(standard_deviation.median()),
            "std_max": float(standard_deviation.max()),
            "nonzero_dimensions": int((standard_deviation > 1e-8).sum()),
            **static.effective_rank(selected),
        })
    return output


def margins_by_slot(
    audit128, correct, positive, negative, valid, pids, repeats, seed
):
    output = []
    for region in range(REGIONS):
        keep = valid[:, region]
        values = audit128.cosine_margin(
            correct[keep, region],
            positive[keep, region],
            negative[keep, region],
        )
        output.append(audit128.clustered_bootstrap(
            values.numpy(), pids[keep].numpy(), repeats, seed + region
        ))
    return output


def representation_summary(
    static, audit128, codebook, raw, valid, pids, repeats, seed
):
    codes = {
        key: transform(raw[key], valid[key], codebook, static)
        for key in ("correct", "positive", "wrong_rgb", "wrong_mask")
    }
    common_rgb = (
        valid["correct"] & valid["positive"] & valid["wrong_rgb"]
    )
    common_mask = (
        valid["correct"] & valid["positive"] & valid["wrong_mask"]
    )
    ranks = effective_rank_by_slot(codes["correct"], valid["correct"], static)
    return {
        "macro_effective_rank": float(np.mean([
            item["entropy_effective_rank"] for item in ranks
        ])),
        "wrong_rgb": margins_by_slot(
            audit128, codes["correct"], codes["positive"],
            codes["wrong_rgb"], common_rgb, pids, repeats, seed
        ),
        "wrong_mask": margins_by_slot(
            audit128, codes["correct"], codes["positive"],
            codes["wrong_mask"], common_mask, pids, repeats, seed + 100
        ),
    }


def main():
    args = parse_args()
    if args.image_chunk < args.clip_batch or args.image_chunk <= 0:
        raise ValueError("image-chunk must be positive and >= clip-batch")
    repo_root = Path(args.repo_root).resolve()
    dependencies = {
        "phase0b": (Path(args.phase0b_script).resolve(), EXPECTED_PHASE0B_SHA256),
        "ontology": (Path(args.ontology_script).resolve(), EXPECTED_ONTOLOGY_SHA256),
        "pcmbcls": (Path(args.pcmbcls_script).resolve(), EXPECTED_PCMBCLS_SHA256),
        "static": (Path(args.static_script).resolve(), EXPECTED_STATIC_SHA256),
        "c8": (Path(args.c8_script).resolve(), EXPECTED_C8_SHA256),
        "audit128": (Path(args.audit128_script).resolve(), EXPECTED_AUDIT128_SHA256),
    }
    for key, (path, expected) in dependencies.items():
        if sha256_file(path) != expected:
            raise RuntimeError("Dependency SHA mismatch: %s" % key)
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.source_commit:
        raise RuntimeError("Source commit mismatch")

    static = load_module("exp393_full_static", dependencies["static"][0])
    if args.streaming_contract_only:
        contract = streaming_fit_contract(static, args.seed)
        print(json.dumps(contract, indent=2, sort_keys=True))
        raise SystemExit(0 if contract["status"] == "PASS" else 1)
    base = load_module("exp393_full_base", dependencies["phase0b"][0])
    ontology = load_module("exp393_full_ontology", dependencies["ontology"][0])
    pcmbcls = load_module("exp393_full_pcmbcls", dependencies["pcmbcls"][0])
    c8 = load_module("exp393_full_c8", dependencies["c8"][0])
    audit128 = load_module("exp393_full_128", dependencies["audit128"][0])
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
    if ontology.ontology_static_contract(base, "hard-owner")["status"] != "PASS":
        raise RuntimeError("Hard-owner ontology contract failed")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    base.set_seed(args.seed)
    dataset = OccludedDuke(root=str(Path(args.data_root).resolve()), verbose=False)
    records = list(dataset.train)
    if len(records) != OFFICIAL_IMAGES:
        raise RuntimeError("Unexpected official train size")
    pids = torch.tensor([int(record[1]) for record in records], dtype=torch.int64)
    fit, audit = static.pid_disjoint_split(pids, args.seed)
    fit_pids = sorted(set(pids[fit].tolist()))
    audit_pids = sorted(set(pids[audit].tolist()))
    if set(fit_pids).intersection(audit_pids):
        raise RuntimeError("PID split leakage")
    partition = {
        "seed": args.seed,
        "unit": "pid",
        "fit_pids": fit_pids,
        "audit_pids": audit_pids,
        "fit_images": int(fit.sum()),
        "audit_images": int(audit.sum()),
        "official_images": len(records),
        "record_order_sha256": sha256_json([
            [Path(record[0]).name, int(record[1]), int(record[2])]
            for record in records
        ]),
    }
    static.write_json(args.partition_output, partition)
    partition_sha = sha256_file(args.partition_output)

    pose_store = PoseTargetStore(
        pose_artifact, base.EXPECTED_POSE_MANIFEST_SHA256
    )
    recipient_dataset = ontology.RecipientDataset(
        base, records, pose_store, args.seed,
        verify_sha=True, partition_mode="hard-owner",
    )
    donor_map, donor_summary = base.build_rgb_donor_map(records)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(
            int(device.index) if device.index is not None else 0
        )
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    teacher = c8.FrozenRichPCMBCLS(
        clip_checkpoint, device, pcmbcls, args.clip_batch
    )

    cache_dir = Path(args.cache_dir).resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError("Cache directory must be absent or empty")
    cache_dir.mkdir(parents=True, exist_ok=True)
    maps = open_memmaps(cache_dir, "w+")
    first_pass = {
        "seconds": 0.0,
        "official_parity_max_abs": 0.0,
        "pixel_product_max": 0.0,
        "repeat_exact": True,
        "null_exact": True,
        "null_valid_exact": True,
        "finite": True,
    }
    for chunk_number, indices in enumerate(
        iter_chunks(range(OFFICIAL_IMAGES), args.image_chunk)
    ):
        batch = collate_indices(recipient_dataset, indices, ontology)
        encoded = teacher.encode(
            batch["pre"], batch["masks"],
            repeat_check=chunk_number == 0,
            null_check=chunk_number == 0,
        )
        raw = static.raw_local_evidence(encoded["region"], encoded["global"])
        valid = batch["valid"].bool() & encoded["valid"].bool()
        maps["raw"][indices] = raw.float().numpy()
        maps["valid"][indices] = valid.numpy().astype(np.uint8)
        first_pass["seconds"] += float(encoded["elapsed_seconds"])
        first_pass["official_parity_max_abs"] = max(
            first_pass["official_parity_max_abs"],
            float(encoded["official_parity_max_abs"]),
        )
        first_pass["pixel_product_max"] = max(
            first_pass["pixel_product_max"], float(encoded["pixel_product_max"])
        )
        first_pass["repeat_exact"] &= bool(encoded["repeat_exact"])
        first_pass["null_exact"] &= bool(encoded["null_exact"])
        first_pass["null_valid_exact"] &= bool(encoded["null_valid_exact"])
        first_pass["finite"] &= bool(torch.isfinite(raw).all())
        if chunk_number % 25 == 0:
            print(json.dumps({
                "stage": "correct-pass", "images": indices[-1] + 1,
                "total": OFFICIAL_IMAGES,
            }), flush=True)
    maps["raw"].flush()
    maps["valid"].flush()

    codebook = fit_streaming_codebook(
        maps["raw"], maps["valid"], fit, static, args.image_chunk * 4
    )
    codebook_payload = {
        "definition": (
            "normalize(region_cls)-normalize(global_cls); slot-center; "
            "shared covariance/eigh PCA-16; L2 normalize"
        ),
        "partition_sha256": partition_sha,
        "slot_means": codebook["means"].tolist(),
        "shared_basis": codebook["basis"].tolist(),
        "fit_counts_by_slot": codebook["fit_counts"],
        "fit_center_mean_abs_max": codebook["fit_center_mean_abs_max"],
        "basis_orthogonal_max_abs": codebook["basis_orthogonal_max_abs"],
        "covariance_rows": codebook["covariance_rows"],
        "top_eigenvalues": codebook["top_eigenvalues"],
    }
    static.write_json(args.codebook_output, codebook_payload)
    codebook_sha = sha256_file(args.codebook_output)
    random_projection = random_codebook(codebook, args.seed + 999)
    uncentered = {
        "means": torch.zeros_like(codebook["means"]),
        "basis": codebook["basis"].clone(),
    }

    audit_indices = torch.nonzero(audit, as_tuple=False).flatten().tolist()
    raw_parts = {key: [] for key in (
        "correct", "positive", "wrong_rgb", "donor_recipient", "wrong_mask"
    )}
    valid_parts = {key: [] for key in raw_parts}
    audit_pid_parts = []
    second_pass_seconds = 0.0
    cycle_iou_max = 0.0
    donor_pid_collision = False
    donor_fixed_point = False
    second_pass_finite = True
    for chunk_number, indices in enumerate(
        iter_chunks(audit_indices, args.image_chunk)
    ):
        batch = collate_indices(recipient_dataset, indices, ontology)
        donor_indices = [int(donor_map[index]) for index in indices]
        donors = collate_indices(recipient_dataset, donor_indices, ontology)
        donor_pid_collision |= bool((batch["pid"] == donors["pid"]).any())
        donor_fixed_point |= any(
            left == right for left, right in zip(indices, donor_indices)
        )
        variants = {
            "positive": teacher.encode(
                batch["pre"].flip(-1), batch["masks"].flip(-1)
            ),
            "wrong_rgb": teacher.encode(donors["pre"], donors["masks"]),
            "donor_recipient": teacher.encode(
                donors["pre"], batch["masks"]
            ),
            "wrong_mask": teacher.encode(
                batch["pre"], batch["masks"].roll(shifts=-1, dims=1)
            ),
        }
        correct_raw = torch.from_numpy(
            np.array(maps["raw"][indices], copy=True)
        ).double()
        correct_valid = torch.from_numpy(
            np.array(maps["valid"][indices], copy=True)
        ).bool()
        raw_parts["correct"].append(correct_raw)
        valid_parts["correct"].append(correct_valid)
        audit_pid_parts.append(batch["pid"])
        for key, encoded in variants.items():
            raw = static.raw_local_evidence(
                encoded["region"], encoded["global"]
            )
            raw_parts[key].append(raw)
            base_valid = donors["valid"] if key == "wrong_rgb" else batch["valid"]
            valid_parts[key].append(
                base_valid.bool() & encoded["valid"].bool()
            )
            second_pass_seconds += float(encoded["elapsed_seconds"])
            second_pass_finite &= bool(torch.isfinite(raw).all())
        cycle_iou_max = max(
            cycle_iou_max,
            float(c8.slot_cycle_iou(batch["masks"]).max()),
        )
        if chunk_number % 25 == 0:
            print(json.dumps({
                "stage": "counterfactual-pass",
                "images": min((chunk_number + 1) * args.image_chunk, len(audit_indices)),
                "total": len(audit_indices),
            }), flush=True)
    raw = {key: torch.cat(parts) for key, parts in raw_parts.items()}
    valid = {key: torch.cat(parts).bool() for key, parts in valid_parts.items()}
    audit_pids_tensor = torch.cat(audit_pid_parts).long()
    codes = {
        key: transform(raw[key], valid[key], codebook, static)
        for key in raw
    }
    common = {
        "wrong_rgb": valid["correct"] & valid["positive"] & valid["wrong_rgb"],
        "donor_recipient": (
            valid["correct"] & valid["positive"] & valid["donor_recipient"]
        ),
        "wrong_mask": valid["correct"] & valid["positive"] & valid["wrong_mask"],
        "slot_cycle": valid["correct"] & valid["positive"],
    }
    ranks = effective_rank_by_slot(codes["correct"], valid["correct"], static)
    margins = {
        "correct_flip_vs_wrong_rgb_donor_donor": margins_by_slot(
            audit128, codes["correct"], codes["positive"], codes["wrong_rgb"],
            common["wrong_rgb"], audit_pids_tensor,
            args.bootstrap_repeats, args.seed + 100,
        ),
        "correct_flip_vs_donor_rgb_recipient_mask": margins_by_slot(
            audit128, codes["correct"], codes["positive"],
            codes["donor_recipient"], common["donor_recipient"],
            audit_pids_tensor, args.bootstrap_repeats, args.seed + 200,
        ),
        "correct_flip_vs_same_rgb_wrong_mask": margins_by_slot(
            audit128, codes["correct"], codes["positive"], codes["wrong_mask"],
            common["wrong_mask"], audit_pids_tensor,
            args.bootstrap_repeats, args.seed + 300,
        ),
        "correct_flip_vs_slot_cycle_binding": margins_by_slot(
            audit128, codes["correct"], codes["positive"],
            codes["correct"].roll(shifts=-1, dims=1), common["slot_cycle"],
            audit_pids_tensor, args.bootstrap_repeats, args.seed + 400,
        ),
    }
    controls = {
        "raw_uncentered": representation_summary(
            static, audit128, uncentered, raw, valid, audit_pids_tensor,
            args.bootstrap_repeats, args.seed + 1000,
        ),
        "fixed_random_orthogonal": representation_summary(
            static, audit128, random_projection, raw, valid, audit_pids_tensor,
            args.bootstrap_repeats, args.seed + 2000,
        ),
    }
    slot_mean_raw = codebook["means"][None].expand(
        len(audit_indices), -1, -1
    )
    slot_mean_code = transform(
        slot_mean_raw, valid["correct"], codebook, static
    )
    global_only_raw = torch.zeros_like(slot_mean_raw)
    global_only_code = transform(
        global_only_raw, valid["correct"],
        {"means": torch.zeros_like(codebook["means"]), "basis": codebook["basis"]},
        static,
    )
    controls.update({
        "slot_mean_code_abs_max": float(slot_mean_code.abs().max()),
        "slot_mean_code_variance_max": float(
            slot_mean_code.var(dim=0, unbiased=False).max()
        ),
        "global_only_raw_abs_max": float(global_only_raw.abs().max()),
        "global_only_code_abs_max": float(global_only_code.abs().max()),
    })

    macro_rank = float(np.mean([
        item["entropy_effective_rank"] for item in ranks
    ]))
    all_codes_finite = all(bool(torch.isfinite(value).all()) for value in codes.values())
    gates = {
        "official_all_images_streamed": (
            len(records) == OFFICIAL_IMAGES
            and partition["fit_images"] + partition["audit_images"] == OFFICIAL_IMAGES
        ),
        "pid_disjoint_partition": not set(fit_pids).intersection(audit_pids),
        "contract_parity_repeat_null": (
            first_pass["official_parity_max_abs"] <= 1e-6
            and first_pass["repeat_exact"]
            and first_pass["null_exact"]
            and first_pass["null_valid_exact"]
        ),
        "hard_owner_wrong_mask": (
            first_pass["pixel_product_max"] == 0.0 and cycle_iou_max == 0.0
        ),
        "all_slot_all_dimension_nonzero_variance": all(
            item["nonzero_dimensions"] == COMPONENTS for item in ranks
        ),
        "macro_effective_rank_at_least_8": macro_rank >= 8.0,
        "wrong_rgb_ci_positive_all_slots": all(
            item["ci95_low"] > 0.0
            for item in margins["correct_flip_vs_wrong_rgb_donor_donor"]
        ),
        "wrong_mask_ci_positive_all_slots": all(
            item["ci95_low"] > 0.0
            for item in margins["correct_flip_vs_same_rgb_wrong_mask"]
        ),
        "static_global_controls_zero": (
            controls["slot_mean_code_abs_max"] == 0.0
            and controls["slot_mean_code_variance_max"] == 0.0
            and controls["global_only_raw_abs_max"] == 0.0
            and controls["global_only_code_abs_max"] == 0.0
        ),
        "frozen_no_grad": teacher.all_parameters_frozen(),
        "finite": (
            first_pass["finite"] and second_pass_finite and all_codes_finite
        ),
        "donor_contract": not donor_pid_collision and not donor_fixed_point,
        "streaming_codebook_contract": (
            codebook["fit_center_mean_abs_max"] <= 1e-6
            and codebook["basis_orthogonal_max_abs"] <= 1e-10
        ),
    }
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    )
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "EXP393_PHASE0E_FULL_OFFICIAL_TRAIN",
        "verdict": "PHASE0E_FULL_PASS" if all(gates.values()) else "PHASE0E_FULL_FAIL",
        "formal_training_authorized": False,
        "phase_b_teacher_interface_authorized": all(gates.values()),
        "semantic_multistage_authorized": False,
        "gates": gates,
        "rank_variance": ranks,
        "macro_effective_rank": macro_rank,
        "margins": margins,
        "controls": controls,
        "checks": {
            "first_pass": first_pass,
            "counterfactual_seconds": second_pass_seconds,
            "peak_memory_bytes": peak_memory,
            "cycle_iou_max": cycle_iou_max,
            "all_codes_finite": all_codes_finite,
            "donor_pid_collision": donor_pid_collision,
            "donor_fixed_point": donor_fixed_point,
            "partition_sha256": partition_sha,
            "codebook_sha256": codebook_sha,
        },
        "execution": {
            "source_commit": actual_commit,
            "repo_root": str(repo_root),
            "device": str(device),
            "torch_version": torch.__version__,
            "open_clip_version": __import__("open_clip").__version__,
            "official_images": OFFICIAL_IMAGES,
            "fit_images": partition["fit_images"],
            "audit_images": partition["audit_images"],
            "fit_pids": len(fit_pids),
            "audit_pids": len(audit_pids),
            "clip_batch": args.clip_batch,
            "image_chunk": args.image_chunk,
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "runtime_sha256": runtime_sha,
            "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint_sha256": base.EXPECTED_CLIP_SHA256,
            "dependency_sha256": {
                key: expected for key, (_, expected) in sorted(dependencies.items())
            },
            "script_sha256": sha256_file(__file__),
            "donor_summary": donor_summary,
        },
    }
    static.write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "verdict": result["verdict"],
        "gates": gates,
        "macro_effective_rank": macro_rank,
        "rank_variance": ranks,
        "margins": margins,
        "controls": controls,
        "checks": result["checks"],
        "output_sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
