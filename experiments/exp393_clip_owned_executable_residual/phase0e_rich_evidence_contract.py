#!/usr/bin/env python3
"""Eight-image real-data contract for exp393 Phase 0E rich evidence.

The script loads the frozen OpenCLIP PC-MBCLS teacher and eight deterministic
official-train images.  It does not build a ReID model, optimizer, training
config, or checkpoint.  Eight images validate execution contracts only; their
counterfactual margins are reported but never used as statistical gates.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


REGIONS = 5
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
EXPECTED_IMAGES = 8
FIT_IMAGES = 5
AUDIT_IMAGES = EXPECTED_IMAGES - FIT_IMAGES


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


def aspect_letterbox(images, masks, mean):
    images = torch.as_tensor(images, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.float32)
    resized_images = F.interpolate(
        images,
        size=(224, 75),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    clip_images = mean.cpu().expand(images.shape[0], 3, 224, 224).clone()
    clip_images[:, :, :, 74:149] = resized_images.clamp(0.0, 1.0)
    resized_masks = F.interpolate(masks, size=(224, 75), mode="nearest")
    clip_masks = torch.zeros(
        images.shape[0], REGIONS, 224, 224, dtype=torch.float32
    )
    clip_masks[:, :, :, 74:149] = resized_masks
    grid_masks = F.avg_pool2d(clip_masks, kernel_size=14, stride=14)
    if grid_masks.shape[-2:] != (16, 16):
        raise RuntimeError("CLIP mask grid mismatch")
    return clip_images, clip_masks, grid_masks


def pairwise_product_max(masks):
    return float(
        max(
            masks[:, left].mul(masks[:, right]).max()
            for left in range(REGIONS)
            for right in range(left + 1, REGIONS)
        )
    )


def slot_cycle_iou(masks):
    masks = torch.as_tensor(masks, dtype=torch.float64)
    cycle = masks.roll(shifts=-1, dims=1)
    intersection = (masks * cycle).flatten(2).sum(-1)
    union = ((masks + cycle) > 0).flatten(2).sum(-1).double()
    return intersection / union.clamp_min(1.0)


class FrozenRichPCMBCLS:
    def __init__(self, checkpoint, device, pcmbcls, clip_batch):
        import open_clip

        self.device = torch.device(device)
        self.pcmbcls = pcmbcls
        self.clip_batch = int(clip_batch)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(checkpoint)
        )
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.visual = model.visual
        del model
        if len(self.visual.transformer.resblocks) != 24:
            raise RuntimeError("Expected 24 ViT-L/14 blocks")
        if tuple(self.visual.grid_size) != (16, 16):
            raise RuntimeError("Expected 16x16 ViT-L/14 patch grid")
        normalizers = [
            transform
            for transform in preprocess.transforms
            if hasattr(transform, "mean") and hasattr(transform, "std")
        ]
        if len(normalizers) != 1:
            raise RuntimeError("Could not identify official CLIP Normalize")
        self.mean = torch.as_tensor(
            tuple(float(value) for value in normalizers[0].mean),
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.as_tensor(
            tuple(float(value) for value in normalizers[0].std),
            device=self.device,
        ).view(1, 3, 1, 1)
        self.preprocess_repr = repr(preprocess)
        if "BICUBIC" not in self.preprocess_repr.upper():
            raise RuntimeError("Official CLIP preprocess is not bicubic")

    def cuda_index(self):
        if self.device.type != "cuda":
            raise RuntimeError("cuda_index requested for a non-CUDA teacher")
        return (
            int(self.device.index)
            if self.device.index is not None
            else int(torch.cuda.current_device())
        )

    @torch.inference_mode()
    def _encode_chunk(self, images, masks, null_check=False):
        clip_images, pixel_masks, grid_masks = aspect_letterbox(
            images, masks, self.mean
        )
        normalized = (
            (clip_images.to(self.device) - self.mean) / self.std
        )
        grid_masks = grid_masks.to(self.device)
        shared = self.pcmbcls.forward_shared_trunk(self.visual, normalized)
        global_features = self.pcmbcls.forward_official_tail(
            self.visual, shared.clone()
        )
        official = F.normalize(self.visual(normalized).float(), dim=-1)
        region_features, region_valid = self.pcmbcls.forward_regions(
            self.visual, shared.clone(), grid_masks
        )
        null_exact = True
        null_valid_exact = True
        if null_check:
            null, null_valid = self.pcmbcls.forward_regions(
                self.visual,
                shared[:1].clone(),
                torch.zeros(1, REGIONS, 16, 16, device=self.device),
            )
            null_exact = bool(torch.equal(null, torch.zeros_like(null)))
            null_valid_exact = bool((~null_valid).all())
        return {
            "global": global_features.cpu(),
            "region": region_features.cpu(),
            "valid": region_valid.cpu(),
            "official_parity_max_abs": float(
                (official - global_features).abs().max()
            ),
            "pixel_product_max": pairwise_product_max(pixel_masks),
            "null_exact": null_exact,
            "null_valid_exact": null_valid_exact,
        }

    @torch.inference_mode()
    def encode(self, images, masks, repeat_check=False, null_check=False):
        global_parts = []
        region_parts = []
        valid_parts = []
        parity = 0.0
        overlap = 0.0
        null_exact = True
        null_valid_exact = True
        first = None
        elapsed = 0.0
        for start in range(0, len(images), self.clip_batch):
            stop = min(start + self.clip_batch, len(images))
            began = time.perf_counter()
            current = self._encode_chunk(
                images[start:stop],
                masks[start:stop],
                null_check=null_check and start == 0,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - began
            if start == 0:
                first = current
            global_parts.append(current["global"])
            region_parts.append(current["region"])
            valid_parts.append(current["valid"])
            parity = max(parity, current["official_parity_max_abs"])
            overlap = max(overlap, current["pixel_product_max"])
            null_exact = null_exact and current["null_exact"]
            null_valid_exact = null_valid_exact and current["null_valid_exact"]
        repeat_exact = True
        if repeat_check:
            stop = min(self.clip_batch, len(images))
            repeated = self._encode_chunk(
                images[:stop], masks[:stop], null_check=False
            )
            repeat_exact = bool(
                torch.equal(first["global"], repeated["global"])
                and torch.equal(first["region"], repeated["region"])
                and torch.equal(first["valid"], repeated["valid"])
            )
        return {
            "global": torch.cat(global_parts),
            "region": torch.cat(region_parts),
            "valid": torch.cat(valid_parts),
            "official_parity_max_abs": parity,
            "pixel_product_max": overlap,
            "null_exact": null_exact,
            "null_valid_exact": null_valid_exact,
            "repeat_exact": repeat_exact,
            "elapsed_seconds": elapsed,
        }

    def all_parameters_frozen(self):
        return all(
            not parameter.requires_grad
            for parameter in self.visual.parameters()
        )


def deterministic_contract_selection(records, recipient_dataset, static, seed):
    ordered = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            (
                str(seed)
                + "\0"
                + Path(records[index][0]).name
            ).encode("utf-8")
        ).hexdigest(),
    )
    unique_pids = torch.tensor(
        sorted({int(record[1]) for record in records}), dtype=torch.int64
    )
    pid_fit, _ = static.pid_disjoint_split(unique_pids, seed)
    fit_pid_set = set(unique_pids[pid_fit].tolist())
    selected = []
    selected_items = []
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
        selected_items.append(item)
        selected_pids.add(pid)
        fit_count += int(is_fit)
        audit_count += int(not is_fit)
        if fit_count == FIT_IMAGES and audit_count == AUDIT_IMAGES:
            break
    if len(selected) != EXPECTED_IMAGES:
        raise RuntimeError("Could not select the fixed eight-image contract")
    return selected, selected_items


def mean_margin_by_slot(static, correct, positive, negative, valid):
    output = []
    for region in range(REGIONS):
        keep = valid[:, region]
        values = static.paired_cosine_margin(
            correct[:, region],
            positive[:, region],
            negative[:, region],
            keep,
        )
        output.append(float(values.mean()) if len(values) else None)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--static-script", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    phase0b_script = Path(args.phase0b_script).resolve()
    ontology_script = Path(args.ontology_script).resolve()
    pcmbcls_script = Path(args.pcmbcls_script).resolve()
    static_script = Path(args.static_script).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    dependencies = {
        phase0b_script: EXPECTED_PHASE0B_SHA256,
        ontology_script: EXPECTED_ONTOLOGY_SHA256,
        pcmbcls_script: EXPECTED_PCMBCLS_SHA256,
        static_script: EXPECTED_STATIC_SHA256,
    }
    for path, expected in dependencies.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError("Dependency SHA mismatch: %s" % path)

    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.source_commit:
        raise RuntimeError("Source commit mismatch")
    base = load_module("exp393_phase0b_base", phase0b_script)
    ontology = load_module("exp393_hard_owner", ontology_script)
    pcmbcls = load_module("exp393_pcmbcls", pcmbcls_script)
    static = load_module("exp393_static", static_script)
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
    selected, selected_items = deterministic_contract_selection(
        records, recipient_dataset, static, args.seed
    )
    recipients = ontology.collate_recipient(selected_items)
    donor_map, donor_summary = base.build_rgb_donor_map(records)
    donor_indices = [int(donor_map[index]) for index in selected]
    donor_items = [recipient_dataset[index] for index in donor_indices]
    donors = ontology.collate_recipient(donor_items)
    if bool((recipients["pid"] == donors["pid"]).any()):
        raise RuntimeError("Donor PID collision")
    if any(left == right for left, right in zip(selected, donor_indices)):
        raise RuntimeError("Donor fixed point")

    device = torch.device(args.device)
    cuda_index = None
    if device.type == "cuda":
        cuda_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_device(cuda_index)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    teacher = FrozenRichPCMBCLS(
        clip_checkpoint, device, pcmbcls, args.clip_batch
    )
    correct = teacher.encode(
        recipients["pre"], recipients["masks"],
        repeat_check=True, null_check=True,
    )
    flipped = teacher.encode(
        recipients["pre"].flip(-1), recipients["masks"].flip(-1)
    )
    donor_donor = teacher.encode(donors["pre"], donors["masks"])
    donor_recipient = teacher.encode(donors["pre"], recipients["masks"])
    cycle_masks = recipients["masks"].roll(shifts=-1, dims=1)
    wrong_mask = teacher.encode(recipients["pre"], cycle_masks)
    peak_memory = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0
    )

    valid = recipients["valid"].bool() & correct["valid"].bool()
    donor_valid = donors["valid"].bool() & donor_donor["valid"].bool()
    raw = static.raw_local_evidence(correct["region"], correct["global"])
    codebook = static.fit_codebook(
        raw, valid, recipients["pid"], args.seed, components=16
    )
    correct_code, _, _ = static.transform_code(raw, valid, codebook)

    def code(encoded, code_valid):
        values = static.raw_local_evidence(
            encoded["region"], encoded["global"]
        )
        return static.transform_code(values, code_valid, codebook)[0]

    flip_valid = valid & flipped["valid"].bool()
    donor_pair_valid = valid & donor_valid
    donor_recipient_valid = valid & donor_recipient["valid"].bool()
    wrong_mask_valid = valid & wrong_mask["valid"].bool()
    flip_code = code(flipped, flip_valid)
    donor_code = code(donor_donor, donor_pair_valid)
    donor_recipient_code = code(donor_recipient, donor_recipient_valid)
    wrong_mask_code = code(wrong_mask, wrong_mask_valid)
    cycle_iou = slot_cycle_iou(recipients["masks"])
    output_shape = list(correct_code.shape)
    all_features_finite = all(
        bool(torch.isfinite(encoded[key]).all())
        for encoded in (
            correct, flipped, donor_donor, donor_recipient, wrong_mask
        )
        for key in ("global", "region")
    )
    no_grad = all(
        not encoded[key].requires_grad
        for encoded in (
            correct, flipped, donor_donor, donor_recipient, wrong_mask
        )
        for key in ("global", "region")
    )
    checks = {
        "selected_indices": selected,
        "donor_indices": donor_indices,
        "selected_pids": recipients["pid"].tolist(),
        "donor_pids": donors["pid"].tolist(),
        "selected_paths": list(recipients["relative_path"]),
        "fit_images": int(codebook["fit"].sum()),
        "audit_images": int(codebook["audit"].sum()),
        "all_recipient_slots_valid": bool(valid.all()),
        "official_global_parity_max_abs": correct["official_parity_max_abs"],
        "repeat_exact": correct["repeat_exact"],
        "null_exact": correct["null_exact"],
        "null_valid_exact": correct["null_valid_exact"],
        "hard_owner_pixel_product_max": correct["pixel_product_max"],
        "flip_rgb_exact": bool(
            torch.equal(
                recipients["pre"].flip(-1).flip(-1), recipients["pre"]
            )
        ),
        "flip_mask_exact": bool(
            torch.equal(
                recipients["masks"].flip(-1).flip(-1),
                recipients["masks"],
            )
        ),
        "wrong_mask_cycle_iou_max": float(cycle_iou.max()),
        "donor_different_pid": bool(
            (recipients["pid"] != donors["pid"]).all()
        ),
        "donor_no_fixed_point": all(
            left != right for left, right in zip(selected, donor_indices)
        ),
        "global_shape": list(correct["global"].shape),
        "region_shape": list(correct["region"].shape),
        "code_shape": output_shape,
        "all_features_finite": all_features_finite,
        "all_codes_finite": bool(
            torch.isfinite(correct_code).all()
            and torch.isfinite(flip_code).all()
            and torch.isfinite(donor_code).all()
            and torch.isfinite(donor_recipient_code).all()
            and torch.isfinite(wrong_mask_code).all()
        ),
        "teacher_parameters_frozen": teacher.all_parameters_frozen(),
        "teacher_outputs_no_grad": no_grad,
        "correct_vs_donor_donor_margin_by_slot": mean_margin_by_slot(
            static, correct_code, flip_code, donor_code, donor_pair_valid
        ),
        "correct_vs_donor_recipient_margin_by_slot": mean_margin_by_slot(
            static,
            correct_code,
            flip_code,
            donor_recipient_code,
            donor_recipient_valid,
        ),
        "correct_vs_wrong_mask_margin_by_slot": mean_margin_by_slot(
            static,
            correct_code,
            flip_code,
            wrong_mask_code,
            wrong_mask_valid,
        ),
        "pcmbcls_seconds_total": float(sum(
            encoded["elapsed_seconds"]
            for encoded in (
                correct, flipped, donor_donor, donor_recipient, wrong_mask
            )
        )),
        "peak_memory_bytes": peak_memory,
    }
    gates = {
        "fixed_sample_contract": (
            len(selected) == EXPECTED_IMAGES
            and len(set(checks["selected_pids"])) == EXPECTED_IMAGES
            and checks["fit_images"] == FIT_IMAGES
            and checks["audit_images"] == AUDIT_IMAGES
        ),
        "recipient_validity": checks["all_recipient_slots_valid"],
        "official_tail_parity": (
            checks["official_global_parity_max_abs"] <= 1e-6
        ),
        "repeat_and_null": (
            checks["repeat_exact"]
            and checks["null_exact"]
            and checks["null_valid_exact"]
        ),
        "geometry": (
            checks["hard_owner_pixel_product_max"] == 0.0
            and checks["flip_rgb_exact"]
            and checks["flip_mask_exact"]
            and checks["wrong_mask_cycle_iou_max"] == 0.0
        ),
        "donor_contract": (
            checks["donor_different_pid"]
            and checks["donor_no_fixed_point"]
        ),
        "shape": (
            checks["global_shape"] == [EXPECTED_IMAGES, 768]
            and checks["region_shape"] == [EXPECTED_IMAGES, REGIONS, 768]
            and checks["code_shape"] == [EXPECTED_IMAGES, REGIONS, 16]
        ),
        "frozen_no_grad": (
            checks["teacher_parameters_frozen"]
            and checks["teacher_outputs_no_grad"]
        ),
        "finite": (
            checks["all_features_finite"] and checks["all_codes_finite"]
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "EXP393_PHASE0E_C8_CONTRACT_ONLY",
        "statistical_teacher_verdict": "NOT_EVALUATED",
        "formal_training_authorized": False,
        "phase_b_authorized": False,
        "checks": checks,
        "gates": gates,
        "execution": {
            "source_commit": actual_commit,
            "repo_root": str(repo_root),
            "device": str(device),
            "torch_version": torch.__version__,
            "open_clip_version": __import__("open_clip").__version__,
            "clip_batch": args.clip_batch,
            "seed": args.seed,
            "runtime_sha256": runtime_sha,
            "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint_sha256": base.EXPECTED_CLIP_SHA256,
            "dependency_sha256": {
                str(path): digest for path, digest in dependencies.items()
            },
            "script_sha256": sha256_file(__file__),
            "donor_summary": donor_summary,
            "preprocess_repr": teacher.preprocess_repr,
        },
    }
    static.write_json(args.output, result)
    summary = {
        "status": result["status"],
        "scope": result["scope"],
        "gates": gates,
        "checks": checks,
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
