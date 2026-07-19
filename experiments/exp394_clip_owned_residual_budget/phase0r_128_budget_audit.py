#!/usr/bin/env python3
"""Read-only clean-D0 residual-budget audit for exp394 Phase 0R-128."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


EXPECTED_EXECUTION_HEAD = "0d1822a07dda8daac0210b68916035b1886d5d99"
EXPECTED_CONFIG_SHA256 = (
    "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069"
)
EXPECTED_CODEBOOK_SHA256 = (
    "4a671a70e0744edad88f911ce628d421650cb09453eb511a61e8d01c239269ef"
)
EXPECTED_SELECTION_SHA256 = (
    "7f3f7626c84553416f39c72be0c15ab430458aa7b201c4bf64461990bbdf15e3"
)
EXPECTED_TRAIN_IMAGES = 15618
EXPECTED_SELECTED_IMAGES = 128
EXPECTED_SELECTED_PIDS = 128
EXPECTED_FIT = 64
EXPECTED_AUDIT = 64
EXPECTED_BANKS = 2
EXPECTED_TOKENS = 48
EXPECTED_CHANNELS = 768
SEED = 20260719


class ExplodingPose(object):
    """Sentinel that fails if RGB-only evaluation touches external pose."""

    def __init__(self):
        self.accesses = 0

    def _fail(self, label):
        self.accesses += 1
        raise AssertionError("Evaluation accessed external pose via %s" % label)

    def __getitem__(self, key):
        return self._fail("getitem:%r" % (key,))

    def __iter__(self):
        return self._fail("iter")

    def items(self):
        return self._fail("items")

    def get(self, key, default=None):
        del default
        return self._fail("get:%r" % (key,))


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


def git_head(repo_root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
        universal_newlines=True,
    ).strip()


def git_tracked_status(repo_root):
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo_root), universal_newlines=True,
    ).strip()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_tensor_digest(digest, tensor):
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))


def tensor_sha256(tensor):
    digest = hashlib.sha256()
    update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        update_tensor_digest(digest, value)
    return digest.hexdigest()


def state_finite(model):
    count = 0
    for name, value in model.state_dict().items():
        count += 1
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise RuntimeError("Non-finite model state tensor: %s" % name)
    return count


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def distribution(values):
    array = values.detach().cpu().double().numpy().reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Invalid applied-delta RMS distribution")
    percentiles = np.percentile(array, [25.0, 50.0, 75.0, 95.0])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "max": float(array.max()),
        "nonzero_fraction": float(np.count_nonzero(array) / float(array.size)),
    }


class AppliedDeltaObserver(object):
    def __init__(self, banks):
        self.banks = list(banks)
        self.handles = []
        self.calls = [0 for _ in self.banks]
        self.deltas = [[] for _ in self.banks]

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Applied-delta observer entered twice")
        for index, bank in enumerate(self.banks):
            def hook(module, inputs, output, bank_index=index):
                del module
                if (
                    not inputs
                    or not torch.is_tensor(inputs[0])
                    or not isinstance(output, tuple)
                    or len(output) != 2
                    or not torch.is_tensor(output[0])
                ):
                    raise RuntimeError("Unexpected PoseSpatialGate seam")
                before = inputs[0]
                after = output[0]
                if before.shape != after.shape:
                    raise RuntimeError("PoseSpatialGate changed token shape")
                if tuple(before.shape[1:]) != (
                    EXPECTED_TOKENS, EXPECTED_CHANNELS
                ):
                    raise RuntimeError(
                        "Unexpected consumer token shape: %r" % (before.shape,)
                    )
                delta = (after - before).detach()
                if not bool(torch.isfinite(delta).all()):
                    raise RuntimeError("Non-finite applied delta")
                self.calls[bank_index] += 1
                self.deltas[bank_index].append(delta.cpu())
                return output

            self.handles.append(bank.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def stacked(self):
        result = []
        for chunks in self.deltas:
            if not chunks:
                raise RuntimeError("Consumer bank was never called")
            result.append(torch.cat(chunks, dim=0))
        return result


def load_fixed_inputs(repo_root, cfg, codebook_path):
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from datasets.bases import ImageDataset
    from datasets.make_dataloader import val_collate_fn
    from datasets.occluded_duke import OccludedDuke

    codebook = json.loads(Path(codebook_path).read_text(encoding="utf-8"))
    selection = codebook.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("Codebook selection is missing")
    selection_sha = sha256_json(selection)
    if codebook.get("selection_sha256") != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Stored selection SHA mismatch")
    if selection_sha != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Recomputed selection SHA mismatch")

    paths = list(selection.get("selected_paths", []))
    pids = [int(value) for value in selection.get("selected_pids", [])]
    indices = [int(value) for value in selection.get("selected_indices", [])]
    fit_mask = list(selection.get("fit_mask", []))
    audit_mask = list(selection.get("audit_mask", []))
    if int(selection.get("seed", -1)) != SEED:
        raise RuntimeError("Selection seed mismatch")

    dataset = OccludedDuke(root=str(cfg.DATASETS.ROOT_DIR), verbose=False)
    records = list(dataset.train)
    if len(records) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError("Unexpected official train size")
    if not (
        len(paths) == len(pids) == len(indices) == len(fit_mask)
        == len(audit_mask) == EXPECTED_SELECTED_IMAGES
    ):
        raise RuntimeError("Unexpected selection length")
    if len(set(paths)) != EXPECTED_SELECTED_IMAGES:
        raise RuntimeError("Selection paths are not unique")
    if len(set(pids)) != EXPECTED_SELECTED_PIDS:
        raise RuntimeError("Selection PIDs are not unique")
    if sum(bool(value) for value in fit_mask) != EXPECTED_FIT:
        raise RuntimeError("Fit selection count mismatch")
    if sum(bool(value) for value in audit_mask) != EXPECTED_AUDIT:
        raise RuntimeError("Audit selection count mismatch")
    if any(not path.startswith("bounding_box_train/") for path in paths):
        raise RuntimeError("Non-train path entered selection")

    data_dir = Path(cfg.DATASETS.ROOT_DIR) / "Occluded_Duke"
    selected_records = []
    rgb_digest = hashlib.sha256()
    for position, (index, relative, pid) in enumerate(zip(indices, paths, pids)):
        record = records[index]
        actual_relative = str(Path(record[0]).resolve().relative_to(data_dir))
        if actual_relative != relative:
            raise RuntimeError("Selection path mismatch at %d" % position)
        if int(record[1]) != pid:
            raise RuntimeError("Selection PID mismatch at %d" % position)
        image_path = Path(record[0]).resolve()
        if not image_path.is_file():
            raise RuntimeError("Selected train RGB is missing")
        rgb_digest.update(relative.encode("utf-8"))
        rgb_digest.update(b"\0")
        rgb_digest.update(sha256_file(image_path).encode("ascii"))
        rgb_digest.update(b"\n")
        selected_records.append(record)

    transform = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            ),
        ]
    )
    loader = DataLoader(
        ImageDataset(selected_records, transform),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=val_collate_fn,
    )
    batches = []
    loaded_paths = []
    input_digest = hashlib.sha256()
    for batch in loader:
        images = batch[0].contiguous()
        batch_paths = list(batch[-1])
        if not bool(torch.isfinite(images).all()):
            raise RuntimeError("Non-finite RGB input tensor")
        update_tensor_digest(input_digest, images)
        batches.append(images)
        loaded_paths.extend(
            str(Path(path).resolve().relative_to(data_dir)) for path in batch_paths
        )
    if loaded_paths != paths:
        raise RuntimeError("Loaded RGB order differs from sealed selection")
    if sum(batch.shape[0] for batch in batches) != EXPECTED_SELECTED_IMAGES:
        raise RuntimeError("Loaded RGB count mismatch")

    return {
        "batches": batches,
        "dataset": dataset,
        "selection": selection,
        "selection_sha256": selection_sha,
        "rgb_manifest_sha256": rgb_digest.hexdigest(),
        "input_tensor_sha256": input_digest.hexdigest(),
        "paths": paths,
        "pids": pids,
        "indices": indices,
    }


def run_pass(model, banks, batches, pose_mode, device):
    exploding = ExplodingPose() if pose_mode == "exploding" else None
    descriptors = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with AppliedDeltaObserver(banks) as observer:
        with torch.no_grad():
            for images in batches:
                images = images.to(device=device, non_blocking=False)
                output = model(
                    images,
                    pose_batch=exploding,
                    tapf_epoch=None,
                )
                if not isinstance(output, tuple) or not torch.is_tensor(output[0]):
                    raise RuntimeError("Unexpected RGB-only model output")
                descriptor = output[0].detach()
                if not bool(torch.isfinite(descriptor).all()):
                    raise RuntimeError("Non-finite descriptor")
                descriptors.append(descriptor.cpu())
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    deltas = observer.stacked()
    return {
        "descriptors": torch.cat(descriptors, dim=0),
        "deltas": deltas,
        "calls": list(observer.calls),
        "hooks_removed": not observer.handles,
        "seconds": float(seconds),
        "images_per_second": EXPECTED_SELECTED_IMAGES / float(seconds),
        "exploding_pose_accesses": 0 if exploding is None else exploding.accesses,
    }


def audit(args):
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    codebook_path = Path(args.codebook).resolve()

    if git_head(repo_root) != EXPECTED_EXECUTION_HEAD:
        raise RuntimeError("Execution HEAD mismatch")
    if git_tracked_status(repo_root):
        raise RuntimeError("Execution repo tracked source is dirty")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Config SHA mismatch")
    checkpoint_sha_before = sha256_file(checkpoint_path)
    if checkpoint_sha_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA mismatch")
    if sha256_file(codebook_path) != EXPECTED_CODEBOOK_SHA256:
        raise RuntimeError("Codebook SHA mismatch")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from config import cfg
    from model import make_model

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    set_seed(SEED)
    fixed = load_fixed_inputs(repo_root, cfg, codebook_path)
    dataset = fixed["dataset"]

    model = make_model(
        cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_train_cams,
        view_num=dataset.num_train_vids,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load failed")
    state_tensors = state_finite(model)
    state_sha_before = state_sha256(model)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = model.to(device).eval()
    tapf = model.base.tapf
    if len(tapf.psg_bank) != EXPECTED_BANKS:
        raise RuntimeError("Expected exactly two clean-D0 consumers")

    torch.cuda.reset_peak_memory_stats(device)
    first = run_pass(model, tapf.psg_bank, fixed["batches"], "none", device)
    second = run_pass(
        model, tapf.psg_bank, fixed["batches"], "exploding", device
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device))

    descriptor_exact = torch.equal(
        first["descriptors"], second["descriptors"]
    )
    delta_exact = [
        torch.equal(left, right)
        for left, right in zip(first["deltas"], second["deltas"])
    ]
    bank_rms = []
    bank_stats = []
    for index, delta in enumerate(first["deltas"]):
        if tuple(delta.shape) != (
            EXPECTED_SELECTED_IMAGES, EXPECTED_TOKENS, EXPECTED_CHANNELS
        ):
            raise RuntimeError("Bank %d delta shape mismatch" % index)
        rms = delta.float().square().mean(dim=-1).sqrt()
        if not bool(torch.isfinite(rms).all()):
            raise RuntimeError("Bank %d RMS is non-finite" % index)
        bank_rms.append(rms)
        record = distribution(rms)
        record.update(
            {
                "bank": index,
                "delta_shape": list(delta.shape),
                "delta_sha256": tensor_sha256(delta),
                "rms_sha256": tensor_sha256(rms),
            }
        )
        bank_stats.append(record)

    pooled_rms = torch.cat([value.reshape(-1) for value in bank_rms])
    pooled_stats = distribution(pooled_rms)
    rho_star = float(pooled_stats["median"])

    model = model.cpu()
    torch.cuda.empty_cache()
    state_sha_after = state_sha256(model)
    checkpoint_sha_after = sha256_file(checkpoint_path)
    tracked_after = git_tracked_status(repo_root)

    expected_calls = len(fixed["batches"])
    gates = {
        "execution_head_exact": git_head(repo_root) == EXPECTED_EXECUTION_HEAD,
        "tracked_source_clean": tracked_after == "",
        "config_checkpoint_codebook_sha_exact": (
            sha256_file(config_path) == EXPECTED_CONFIG_SHA256
            and checkpoint_sha_before == checkpoint_sha_after
            == EXPECTED_CHECKPOINT_SHA256
            and sha256_file(codebook_path) == EXPECTED_CODEBOOK_SHA256
        ),
        "selection_exact": (
            fixed["selection_sha256"] == EXPECTED_SELECTION_SHA256
            and len(fixed["paths"]) == EXPECTED_SELECTED_IMAGES
            and len(set(fixed["paths"])) == EXPECTED_SELECTED_IMAGES
            and len(set(fixed["pids"])) == EXPECTED_SELECTED_PIDS
            and all(
                path.startswith("bounding_box_train/")
                for path in fixed["paths"]
            )
        ),
        "official_train_exact": len(dataset.train) == EXPECTED_TRAIN_IMAGES,
        "strict_state_finite": state_tensors > 0,
        "both_consumers_called_exact": (
            first["calls"] == [expected_calls, expected_calls]
            and second["calls"] == [expected_calls, expected_calls]
        ),
        "hooks_removed": first["hooks_removed"] and second["hooks_removed"],
        "descriptor_repeat_pose_free_exact": descriptor_exact,
        "delta_repeat_pose_free_exact": all(delta_exact),
        "exploding_pose_unaccessed": second["exploding_pose_accesses"] == 0,
        "all_bank_rms_finite_nonzero": all(
            math.isfinite(record["median"])
            and record["nonzero_fraction"] > 0.0
            for record in bank_stats
        ),
        "rho_star_finite_positive": math.isfinite(rho_star) and rho_star > 0.0,
        "state_sha_before_after_exact": state_sha_before == state_sha_after,
        "checkpoint_sha_before_after_exact": (
            checkpoint_sha_before == checkpoint_sha_after
        ),
        "no_optimizer_constructed": True,
    }

    return {
        "scope": "EXP394_PHASE0R_128_TRAIN_ONLY_BUDGET",
        "verdict": (
            "PHASE0R_128_PASS" if all(gates.values()) else "PHASE0R_128_FAIL"
        ),
        "rho_star": rho_star,
        "rho_formula": (
            "median(concat(bank0_per_token_channel_rms,"
            "bank1_per_token_channel_rms))"
        ),
        "gates": gates,
        "assets": {
            "execution_head": EXPECTED_EXECUTION_HEAD,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "checkpoint_sha256": checkpoint_sha_before,
            "codebook_sha256": EXPECTED_CODEBOOK_SHA256,
            "selection_sha256": fixed["selection_sha256"],
            "rgb_manifest_sha256": fixed["rgb_manifest_sha256"],
            "input_tensor_sha256": fixed["input_tensor_sha256"],
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "coverage": {
            "official_train_images": len(dataset.train),
            "selected_images": len(fixed["paths"]),
            "selected_unique_pids": len(set(fixed["pids"])),
            "fit_images": sum(
                bool(value) for value in fixed["selection"]["fit_mask"]
            ),
            "audit_images": sum(
                bool(value) for value in fixed["selection"]["audit_mask"]
            ),
            "batches": expected_calls,
            "tokens_per_bank": EXPECTED_SELECTED_IMAGES * EXPECTED_TOKENS,
            "pooled_tokens": int(pooled_rms.numel()),
        },
        "bank_distributions": bank_stats,
        "pooled_distribution": pooled_stats,
        "repeat": {
            "descriptor_exact": descriptor_exact,
            "descriptor_sha256": tensor_sha256(first["descriptors"]),
            "delta_exact_by_bank": delta_exact,
            "calls_first": first["calls"],
            "calls_second": second["calls"],
            "exploding_pose_accesses": second["exploding_pose_accesses"],
        },
        "runtime": {
            "first_seconds": first["seconds"],
            "second_seconds": second["seconds"],
            "first_images_per_second": first["images_per_second"],
            "second_images_per_second": second["images_per_second"],
            "peak_memory_bytes": peak_memory,
            "torch_version": torch.__version__,
        },
        "state": {
            "tensors": state_tensors,
            "sha256_before": state_sha_before,
            "sha256_after": state_sha_after,
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "tracked_status_after": tracked_after,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    try:
        result = audit(args)
    except Exception as error:
        result = {
            "scope": "EXP394_PHASE0R_128_TRAIN_ONLY_BUDGET",
            "verdict": "PHASE0R_128_FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        write_json(output_path, result)
        traceback.print_exc()
        print("output_sha256=%s" % sha256_file(output_path), flush=True)
        return 1
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("output_sha256=%s" % sha256_file(output_path), flush=True)
    return 0 if result["verdict"] == "PHASE0R_128_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
