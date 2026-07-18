#!/usr/bin/env python3
"""Synthetic contracts for exp393 Phase 0E rich CLIP evidence.

This script does not load CLIP, ReID data, a model, an optimizer, or CUDA.  It
freezes the deterministic PID split, slot centering, shared PCA, normalization,
effective-rank, counterfactual-margin, and hard-owner wrong-mask contracts used
by the later eight-image and full teacher-only audits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


REGIONS = 5
DEFAULT_SEED = 20260719


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pid_disjoint_split(pids, seed, fit_modulus=2):
    """Return a deterministic split whose unit is PID, never image."""
    pids = torch.as_tensor(pids, dtype=torch.int64).flatten()
    if pids.numel() == 0:
        raise ValueError("pids must be nonempty")
    if fit_modulus < 2:
        raise ValueError("fit_modulus must be at least two")
    unique = torch.unique(pids, sorted=True).tolist()
    assignment = {}
    for pid in unique:
        payload = (str(int(seed)) + "\0" + str(int(pid))).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        assignment[int(pid)] = (bucket % int(fit_modulus)) == 0
    fit = torch.tensor([assignment[int(pid)] for pid in pids], dtype=torch.bool)
    audit = ~fit
    if not bool(fit.any()) or not bool(audit.any()):
        raise RuntimeError("deterministic PID split produced an empty side")
    fit_pids = set(pids[fit].tolist())
    audit_pids = set(pids[audit].tolist())
    if fit_pids.intersection(audit_pids):
        raise RuntimeError("PID-disjoint split leaked an identity")
    return fit, audit


def canonicalize_basis_sign(basis):
    basis = torch.as_tensor(basis, dtype=torch.float64).clone()
    for row in range(basis.shape[0]):
        pivot = int(basis[row].abs().argmax())
        if float(basis[row, pivot]) < 0.0:
            basis[row].neg_()
    return basis


def raw_local_evidence(region_features, global_features):
    region_features = torch.as_tensor(region_features, dtype=torch.float64)
    global_features = torch.as_tensor(global_features, dtype=torch.float64)
    if region_features.ndim != 3 or region_features.shape[1] != REGIONS:
        raise ValueError("region_features must have shape [N,5,D]")
    if global_features.shape != (
        region_features.shape[0], region_features.shape[2]
    ):
        raise ValueError("global_features must have shape [N,D]")
    region = F.normalize(region_features, dim=-1)
    global_feature = F.normalize(global_features, dim=-1)
    return region - global_feature[:, None]


def fit_codebook(raw, valid, pids, seed, components):
    """Fit slot means and one shared PCA basis on the fit PID partition."""
    raw = torch.as_tensor(raw, dtype=torch.float64)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    pids = torch.as_tensor(pids, dtype=torch.int64)
    if raw.ndim != 3 or raw.shape[1] != REGIONS:
        raise ValueError("raw must have shape [N,5,D]")
    if valid.shape != raw.shape[:2] or pids.shape != raw.shape[:1]:
        raise ValueError("valid/pid shape mismatch")
    fit, audit = pid_disjoint_split(pids, seed)
    means = torch.zeros(REGIONS, raw.shape[-1], dtype=torch.float64)
    centered_fit = []
    fit_counts = []
    for region in range(REGIONS):
        keep = fit & valid[:, region]
        count = int(keep.sum())
        if count < 2:
            raise RuntimeError("insufficient fit samples for slot %d" % region)
        means[region] = raw[keep, region].mean(dim=0)
        centered = raw[keep, region] - means[region]
        centered_fit.append(centered)
        fit_counts.append(count)
    matrix = torch.cat(centered_fit, dim=0)
    if components > min(matrix.shape):
        raise RuntimeError("PCA component count exceeds fit matrix rank bound")
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = canonicalize_basis_sign(vh[:components])
    return {
        "means": means,
        "basis": basis,
        "fit": fit,
        "audit": audit,
        "fit_counts": fit_counts,
    }


def transform_code(raw, valid, codebook):
    raw = torch.as_tensor(raw, dtype=torch.float64)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    centered = raw - codebook["means"][None]
    projected = torch.einsum("nrd,kd->nrk", centered, codebook["basis"])
    code = F.normalize(projected, dim=-1)
    code = torch.where(valid[..., None], code, torch.zeros_like(code))
    return code, centered, projected


def effective_rank(code):
    code = torch.as_tensor(code, dtype=torch.float64)
    if code.ndim != 2 or code.shape[0] < 2:
        raise ValueError("effective_rank expects at least two rows")
    centered = code - code.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return {
        "entropy_effective_rank": float(entropy.exp()),
        "top_singular_energy_fraction": float(probability[0]),
        "singular_values": singular.tolist(),
    }


def paired_cosine_margin(correct, positive, negative, valid):
    correct = F.normalize(torch.as_tensor(correct, dtype=torch.float64), dim=-1)
    positive = F.normalize(torch.as_tensor(positive, dtype=torch.float64), dim=-1)
    negative = F.normalize(torch.as_tensor(negative, dtype=torch.float64), dim=-1)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if correct.shape != positive.shape or correct.shape != negative.shape:
        raise ValueError("paired code shape mismatch")
    margin = (correct * positive).sum(-1) - (correct * negative).sum(-1)
    return margin[valid]


def hard_owner_cycle_contract(height=20, width=10):
    masks = torch.zeros(REGIONS, height, width, dtype=torch.float64)
    boundaries = torch.linspace(0, height, REGIONS + 1).round().long()
    for region in range(REGIONS):
        masks[region, boundaries[region]:boundaries[region + 1]] = 1.0
    cycle = masks.roll(shifts=-1, dims=0)
    intersection = (masks * cycle).flatten(1).sum(1)
    union = ((masks + cycle) > 0).flatten(1).sum(1).double()
    iou = intersection / union.clamp_min(1.0)
    return {
        "pairwise_product_max": float(
            max(
                masks[left].mul(masks[right]).max()
                for left in range(REGIONS)
                for right in range(left + 1, REGIONS)
            )
        ),
        "cycle_iou": iou.tolist(),
        "cycle_iou_max": float(iou.max()),
    }


def synthetic_contract(seed=DEFAULT_SEED):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    identities = 40
    images_per_identity = 2
    samples = identities * images_per_identity
    dimensions = 32
    components = 16
    pids = torch.arange(identities).repeat_interleave(images_per_identity)
    global_features = torch.randn(
        samples, dimensions, generator=generator, dtype=torch.float64
    )
    identity_signal = torch.randn(
        identities, REGIONS, dimensions, generator=generator, dtype=torch.float64
    )
    slot_signal = torch.randn(
        REGIONS, dimensions, generator=generator, dtype=torch.float64
    )
    region_features = (
        global_features[:, None]
        + 0.8 * identity_signal.index_select(0, pids)
        + 0.4 * slot_signal[None]
        + 0.02
        * torch.randn(
            samples, REGIONS, dimensions,
            generator=generator,
            dtype=torch.float64,
        )
    )
    positive_features = region_features + 0.01 * torch.randn(
        region_features.shape, generator=generator, dtype=torch.float64
    )
    wrong_rgb_features = torch.randn(
        region_features.shape, generator=generator, dtype=torch.float64
    )
    wrong_mask_features = region_features.roll(shifts=-1, dims=1)
    valid = torch.ones(samples, REGIONS, dtype=torch.bool)
    valid[0, 0] = False

    raw = raw_local_evidence(region_features, global_features)
    positive_raw = raw_local_evidence(positive_features, global_features)
    wrong_rgb_raw = raw_local_evidence(wrong_rgb_features, global_features)
    wrong_mask_raw = raw_local_evidence(wrong_mask_features, global_features)
    codebook = fit_codebook(raw, valid, pids, seed, components)
    code, centered, _ = transform_code(raw, valid, codebook)
    positive, _, _ = transform_code(positive_raw, valid, codebook)
    wrong_rgb, _, _ = transform_code(wrong_rgb_raw, valid, codebook)
    wrong_mask, _, _ = transform_code(wrong_mask_raw, valid, codebook)

    fit_again, audit_again = pid_disjoint_split(pids, seed)
    fit_pids = set(pids[codebook["fit"]].tolist())
    audit_pids = set(pids[codebook["audit"]].tolist())
    fit_center_max = 0.0
    per_slot_rank = []
    for region in range(REGIONS):
        fit_keep = codebook["fit"] & valid[:, region]
        audit_keep = codebook["audit"] & valid[:, region]
        fit_center_max = max(
            fit_center_max,
            float(centered[fit_keep, region].mean(dim=0).abs().max()),
        )
        per_slot_rank.append(effective_rank(code[audit_keep, region]))
    rgb_margin = paired_cosine_margin(
        code, positive, wrong_rgb, codebook["audit"][:, None] & valid
    )
    mask_margin = paired_cosine_margin(
        code, positive, wrong_mask, codebook["audit"][:, None] & valid
    )
    basis_gram = codebook["basis"] @ codebook["basis"].T
    basis_error = float(
        (basis_gram - torch.eye(components, dtype=torch.float64)).abs().max()
    )
    hard_owner = hard_owner_cycle_contract()
    checks = {
        "split_repeat_exact": bool(
            torch.equal(codebook["fit"], fit_again)
            and torch.equal(codebook["audit"], audit_again)
        ),
        "fit_audit_pid_disjoint": not bool(fit_pids.intersection(audit_pids)),
        "fit_and_audit_nonempty": bool(
            codebook["fit"].any() and codebook["audit"].any()
        ),
        "fit_center_max_abs": fit_center_max,
        "basis_orthonormal_max_abs": basis_error,
        "invalid_code_exact_zero": bool(
            torch.equal(code[~valid], torch.zeros_like(code[~valid]))
        ),
        "valid_code_unit_norm_max_abs": float(
            (code[valid].norm(dim=-1) - 1.0).abs().max()
        ),
        "repeat_code_exact": bool(
            torch.equal(code, transform_code(raw, valid, codebook)[0])
        ),
        "wrong_rgb_margin_mean": float(rgb_margin.mean()),
        "wrong_mask_margin_mean": float(mask_margin.mean()),
        "finite": bool(
            torch.isfinite(code).all()
            and torch.isfinite(codebook["means"]).all()
            and torch.isfinite(codebook["basis"]).all()
        ),
        "hard_owner": hard_owner,
        "per_slot_rank": per_slot_rank,
    }
    gates = {
        "deterministic_pid_split": (
            checks["split_repeat_exact"]
            and checks["fit_audit_pid_disjoint"]
            and checks["fit_and_audit_nonempty"]
        ),
        "fit_center_exact": checks["fit_center_max_abs"] <= 1e-12,
        "shared_basis_orthonormal": (
            checks["basis_orthonormal_max_abs"] <= 1e-10
        ),
        "normalization_contract": (
            checks["invalid_code_exact_zero"]
            and checks["valid_code_unit_norm_max_abs"] <= 1e-12
            and checks["repeat_code_exact"]
        ),
        "synthetic_counterfactual_direction": (
            checks["wrong_rgb_margin_mean"] > 0.0
            and checks["wrong_mask_margin_mean"] > 0.0
        ),
        "hard_owner_wrong_mask": (
            hard_owner["pairwise_product_max"] == 0.0
            and hard_owner["cycle_iou_max"] == 0.0
        ),
        "finite": checks["finite"],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "checks": checks,
        "gates": gates,
        "seed": int(seed),
        "regions": REGIONS,
        "dimensions": dimensions,
        "components": components,
        "samples": samples,
        "identities": identities,
        "fit_images": int(codebook["fit"].sum()),
        "audit_images": int(codebook["audit"].sum()),
        "fit_counts_by_slot": codebook["fit_counts"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = synthetic_contract(args.seed)
    result["script_sha256"] = sha256_file(__file__)
    result["torch_version"] = torch.__version__
    if args.output:
        write_json(args.output, result)
        result["output_sha256"] = sha256_file(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
