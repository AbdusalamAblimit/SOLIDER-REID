#!/usr/bin/env python3
"""Frozen clean-D0 internal-field counterfactual audit for exp392 Phase 0A.

The script is intentionally external to the sealed exp387 repository.  It
strictly loads one final checkpoint, never constructs an optimizer, and uses
short-lived hooks at the clean PoseAnchor / PoseSpatialGate seams.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


COCO17_LEFT_RIGHT = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9,
                     12, 11, 14, 13, 16, 15)
CHANNEL_CYCLE = tuple(range(1, 17)) + (0,)
EXPECTED_CONFIG_SHA256 = (
    "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069"
)
EXPECTED_EXECUTION_HEAD = "0d1822a07dda8daac0210b68916035b1886d5d99"
EXPECTED_CORRECT_PERCENT = (57.6, 67.7, 80.8, 84.6)

FULL_ARMS = (
    "correct_start",
    "external_correct",
    "external_shuffle",
    "external_none",
    "external_exploding",
    "channel_cycle",
    "left_right_channel_swap",
    "confidence_permutation",
    "matched_wrong_field",
    "spatial_constant",
    "zero_field",
    "psg_bypass_0",
    "psg_bypass_1",
    "psg_bypass_all",
    "correct_end",
)


class ExplodingPose:
    """Sentinel that fails if evaluation touches external pose."""

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


def git_head(repo_root):
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
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


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        update_tensor_digest(digest, value)
    return digest.hexdigest()


def validate_field(field):
    if not torch.is_tensor(field) or field.ndim != 4 or field.shape[1] != 17:
        raise RuntimeError("Unexpected clean field contract: %r" % (field,))
    if field.dtype != torch.float32:
        raise RuntimeError("Clean field must remain float32")
    if not field.is_contiguous():
        raise RuntimeError("Clean field must remain contiguous")
    if not bool(torch.isfinite(field).all()):
        raise RuntimeError("Non-finite clean field")
    if bool((field < 0).any()):
        raise RuntimeError("Clean field must be non-negative")
    return field


def transform_field(field, arm):
    field = validate_field(field)
    if arm == "channel_cycle":
        index = torch.as_tensor(CHANNEL_CYCLE, device=field.device)
        result = field.index_select(1, index)
    elif arm == "left_right_channel_swap":
        index = torch.as_tensor(COCO17_LEFT_RIGHT, device=field.device)
        result = field.index_select(1, index)
    elif arm == "confidence_permutation":
        index = torch.as_tensor(CHANNEL_CYCLE, device=field.device)
        peak = field.flatten(2).amax(dim=-1)
        shape = field / peak.clamp_min(1e-12)[:, :, None, None]
        shape = torch.where(
            (peak > 0)[:, :, None, None], shape, torch.zeros_like(shape)
        )
        result = shape * peak.index_select(1, index)[:, :, None, None]
    elif arm == "spatial_constant":
        result = field.mean(dim=(-2, -1), keepdim=True).expand_as(field)
    elif arm == "zero_field":
        result = torch.zeros_like(field)
    else:
        raise ValueError("Unknown field arm: %s" % arm)
    return validate_field(result.contiguous())


class TensorAccumulator:
    """Streaming exact digest plus finite norm/nonzero statistics."""

    def __init__(self):
        self.digest = hashlib.sha256()
        self.calls = 0
        self.elements = 0
        self.nonzero = 0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.max_abs = 0.0

    def update(self, tensor):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Accumulator received NaN/Inf")
        detached = tensor.detach()
        self.calls += 1
        self.elements += detached.numel()
        self.nonzero += int(torch.count_nonzero(detached).item())
        self.abs_sum += float(detached.abs().sum().item())
        self.square_sum += float(detached.float().square().sum().item())
        self.max_abs = max(self.max_abs, float(detached.abs().amax().item()))
        update_tensor_digest(self.digest, detached)

    def summary(self):
        denominator = max(self.elements, 1)
        return {
            "sha256": self.digest.hexdigest(),
            "calls": self.calls,
            "elements": self.elements,
            "nonzero_fraction": self.nonzero / float(denominator),
            "mean_abs": self.abs_sum / float(denominator),
            "rms": math.sqrt(self.square_sum / float(denominator)),
            "max_abs": self.max_abs,
        }


class AnchorFieldHook:
    """Observe or replace only PoseAnchor output['field']."""

    REQUIRED = (
        "heatmap_logits",
        "confidence_logits",
        "heatmaps",
        "confidence",
        "field",
    )

    def __init__(self, anchor, transform=None, collect_matching_stats=False):
        self.anchor = anchor
        self.transform = transform
        self.collect_matching_stats = bool(collect_matching_stats)
        self.handle = None
        self.calls = 0
        self.changed_calls = 0
        self.max_abs_delta = 0.0
        self.input_digest = hashlib.sha256()
        self.output = TensorAccumulator()
        self.matching_stats = []

    @staticmethod
    def _stats(field):
        batch, channels, height, width = field.shape
        h1 = max(1, int(round(height / 3.0)))
        h2 = min(height - 1, int(round(2.0 * height / 3.0)))
        peak = field.flatten(2).amax(-1)
        total = field.sum((1, 2, 3))
        active = (peak >= 0.10).float().sum(1)
        top = field[:, :, :h1].sum((1, 2, 3))
        middle = field[:, :, h1:h2].sum((1, 2, 3))
        bottom = field[:, :, h2:].sum((1, 2, 3))
        return torch.stack((total, active, top, middle, bottom), dim=1)

    def _hook(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, dict):
            raise RuntimeError("PoseAnchor output is not a dict")
        missing = [key for key in self.REQUIRED if key not in output]
        if missing:
            raise RuntimeError("PoseAnchor output missing keys: %s" % missing)
        before = validate_field(output["field"])
        update_tensor_digest(self.input_digest, before)
        after = before if self.transform is None else self.transform(before)
        after = validate_field(after)
        if after.shape != before.shape or after.device != before.device:
            raise RuntimeError("Field intervention changed shape/device")
        delta = float((after - before).abs().amax().item())
        self.calls += 1
        self.changed_calls += int(delta > 0.0)
        self.max_abs_delta = max(self.max_abs_delta, delta)
        self.output.update(after)
        if self.collect_matching_stats:
            self.matching_stats.append(self._stats(after).detach().cpu())
        if after.data_ptr() == before.data_ptr():
            return output
        replaced = dict(output)
        replaced["field"] = after
        for key in self.REQUIRED[:-1]:
            if replaced[key] is not output[key]:
                raise RuntimeError("Non-field anchor output was replaced")
        return replaced

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Anchor hook entered twice")
        self.handle = self.anchor.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        self.handle = None
        return False

    def summary(self):
        return {
            "calls": self.calls,
            "changed_calls": self.changed_calls,
            "max_abs_delta": self.max_abs_delta,
            "input_field_sha256": self.input_digest.hexdigest(),
            "consumed_field": self.output.summary(),
            "hook_removed": self.handle is None,
        }


class GateObserver:
    def __init__(self, banks):
        self.banks = list(banks)
        self.handles = []
        self.outputs = [TensorAccumulator() for _ in self.banks]

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Gate observer entered twice")
        for index, bank in enumerate(self.banks):
            def hook(module, inputs, output, bank_index=index):
                del module, inputs
                if (not isinstance(output, tuple) or len(output) != 2
                        or not torch.is_tensor(output[1])):
                    raise RuntimeError("Unexpected PoseSpatialGate output")
                self.outputs[bank_index].update(output[1])
                return output
            self.handles.append(bank.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def summary(self):
        return {
            "banks": [item.summary() for item in self.outputs],
            "hooks_removed": not self.handles,
        }


class PsgBypass:
    def __init__(self, banks, indices):
        self.banks = list(banks)
        self.indices = tuple(indices)
        self.handles = []
        self.calls = {str(index): 0 for index in self.indices}

    def __enter__(self):
        for index in self.indices:
            def hook(module, inputs, output, bank_index=index):
                del module
                if (not inputs or not torch.is_tensor(inputs[0])
                        or not isinstance(output, tuple) or len(output) != 2):
                    raise RuntimeError("Unexpected PSG bypass contract")
                self.calls[str(bank_index)] += 1
                return inputs[0], torch.zeros_like(output[1])
            self.handles.append(
                self.banks[index].register_forward_hook(hook)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def summary(self):
        return {
            "indices": list(self.indices),
            "calls": dict(self.calls),
            "all_called": all(value > 0 for value in self.calls.values()),
            "hooks_removed": not self.handles,
        }


@contextlib.contextmanager
def capture_one_field(anchor):
    captured = []

    def hook(module, inputs, output):
        del module, inputs
        captured.append(validate_field(output["field"]).detach().clone())
        return output

    handle = anchor.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def dummy_pose(batch_size, device):
    generator = torch.Generator(device="cpu").manual_seed(20260718 + batch_size)
    return {
        "keypoints": torch.rand(batch_size, 17, 2, generator=generator).to(device),
        "scores": torch.rand(batch_size, 17, generator=generator).to(device),
        "valid": torch.ones(batch_size, 17, dtype=torch.bool, device=device),
    }


def pose_for_arm(arm, batch_size, device, exploding):
    if arm == "external_none":
        return None
    if arm == "external_exploding":
        return exploding
    if arm in ("external_correct", "external_shuffle"):
        pose = dummy_pose(batch_size, device)
        if arm == "external_shuffle":
            return {key: value.roll(1, dims=0) for key, value in pose.items()}
        return pose
    return None


def forward_descriptor(model, images, camids_batch, viewids, pose_batch):
    descriptor, _ = model(
        images,
        cam_label=camids_batch,
        view_label=viewids,
        pose_batch=pose_batch,
        tapf_epoch=None,
    )
    if (not torch.is_tensor(descriptor) or descriptor.ndim != 2
            or not bool(torch.isfinite(descriptor).all())):
        raise RuntimeError("Invalid/non-finite descriptor")
    return descriptor


def descriptor_delta(actual, reference):
    actual = actual.float()
    reference = reference.float()
    cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=1)
    l2 = (actual - reference).norm(dim=1)
    max_abs = (actual - reference).abs().amax(dim=1)

    def summarize(value):
        value = value.double().numpy()
        return {
            "mean": float(value.mean()),
            "median": float(np.median(value)),
            "p95": float(np.quantile(value, 0.95)),
            "max": float(value.max()),
        }

    return {
        "cosine_to_correct": summarize(cosine),
        "l2_to_correct": summarize(l2),
        "max_abs_to_correct": summarize(max_abs),
        "elementwise_exact": bool(torch.equal(actual, reference)),
    }


def per_query_metrics(distmat, pids, camids, num_query, max_rank=50):
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int8)
    ap = []
    hits = []
    valid_indices = []
    for query_index in range(num_query):
        order = indices[query_index]
        keep = ~((g_pids[order] == q_pids[query_index])
                 & (g_camids[order] == q_camids[query_index]))
        binary = matches[query_index][keep]
        if not np.any(binary):
            continue
        cumulative = binary.cumsum()
        precision = cumulative / np.arange(1, len(binary) + 1, dtype=np.float64)
        ap.append(float((precision * binary).sum() / binary.sum()))
        row = []
        for rank in (1, 5, 10):
            row.append(float(np.any(binary[:rank])))
        hits.append(row)
        valid_indices.append(query_index)
    return {
        "ap": np.asarray(ap, dtype=np.float64),
        "hits": np.asarray(hits, dtype=np.float64),
        "valid_indices": np.asarray(valid_indices, dtype=np.int64),
    }


def retrieval_metrics(descriptors, pids, camids, num_query, feat_norm):
    from utils.metrics import euclidean_distance

    features = descriptors.float()
    if feat_norm:
        features = torch.nn.functional.normalize(features, dim=1, p=2)
    distmat = euclidean_distance(features[:num_query], features[num_query:])
    query = per_query_metrics(distmat, pids, camids, num_query)
    result = (
        float(query["ap"].mean()),
        float(query["hits"][:, 0].mean()),
        float(query["hits"][:, 1].mean()),
        float(query["hits"][:, 2].mean()),
    )
    return result, query


def paired_bootstrap(actual, reference, seed=20260718, repeats=1000):
    if not np.array_equal(actual["valid_indices"], reference["valid_indices"]):
        raise RuntimeError("Query validity changed across arms")
    values = np.column_stack((actual["ap"], actual["hits"]))
    baseline = np.column_stack((reference["ap"], reference["hits"]))
    delta = values - baseline
    rng = np.random.RandomState(seed)
    samples = np.empty((repeats, 4), dtype=np.float64)
    for index in range(repeats):
        draw = rng.randint(0, len(delta), size=len(delta))
        samples[index] = delta[draw].mean(axis=0)
    return {
        "delta_mean_percent": (100.0 * delta.mean(axis=0)).tolist(),
        "ci95_low_percent": (100.0 * np.quantile(samples, 0.025, axis=0)).tolist(),
        "ci95_high_percent": (100.0 * np.quantile(samples, 0.975, axis=0)).tolist(),
        "bootstrap_repeats": repeats,
    }


def build_donor_map(stats, pids, camids, num_query):
    """Deterministic near-neighbour map with same-split/camera preference."""
    stats = np.asarray(stats, dtype=np.float64)
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    count = len(stats)
    donor = np.full(count, -1, dtype=np.int64)
    same_camera = np.zeros(count, dtype=np.bool_)

    split_groups = [np.arange(0, num_query), np.arange(num_query, count)]
    for split_indices in split_groups:
        split_stats = stats[split_indices]
        mean = split_stats.mean(axis=0)
        std = split_stats.std(axis=0)
        std[std < 1e-8] = 1.0
        normalized = (stats - mean) / std
        for camera in np.unique(camids[split_indices]):
            group = split_indices[camids[split_indices] == camera]
            order = group[np.argsort(normalized[group, 0], kind="mergesort")]
            position = {int(value): index for index, value in enumerate(order.tolist())}
            for recipient in group:
                center = position[int(recipient)]
                left = max(0, center - 96)
                right = min(len(order), center + 97)
                candidates = order[left:right]
                candidates = candidates[pids[candidates] != pids[recipient]]
                if len(candidates):
                    distance = ((normalized[candidates] - normalized[recipient]) ** 2).sum(1)
                    best = candidates[np.argmin(distance)]
                    donor[recipient] = int(best)
                    same_camera[recipient] = True
        missing = split_indices[donor[split_indices] < 0]
        if len(missing):
            order = split_indices[np.argsort(normalized[split_indices, 0], kind="mergesort")]
            position = {int(value): index for index, value in enumerate(order.tolist())}
            for recipient in missing:
                center = position[int(recipient)]
                radius = min(max(256, len(order)), len(order))
                left = max(0, center - radius)
                right = min(len(order), center + radius + 1)
                candidates = order[left:right]
                candidates = candidates[pids[candidates] != pids[recipient]]
                if not len(candidates):
                    raise RuntimeError("Could not find different-PID donor")
                distance = ((normalized[candidates] - normalized[recipient]) ** 2).sum(1)
                donor[recipient] = int(candidates[np.argmin(distance)])

    if bool((donor < 0).any()) or bool((donor == np.arange(count)).any()):
        raise RuntimeError("Invalid donor map")
    if bool((pids[donor] == pids).any()):
        raise RuntimeError("Donor map contains same-PID pairs")
    if bool(((np.arange(count) < num_query) != (donor < num_query)).any()):
        raise RuntimeError("Donor map crosses query/gallery split")
    distances = np.sqrt(((stats[donor] - stats) ** 2).sum(1))
    return donor, {
        "count": int(count),
        "same_camera_fraction": float(same_camera.mean()),
        "different_pid_fraction": float((pids[donor] != pids).mean()),
        "no_fixed_points": bool(np.all(donor != np.arange(count))),
        "raw_stats_distance_mean": float(distances.mean()),
        "raw_stats_distance_p95": float(np.quantile(distances, 0.95)),
    }


def make_donor_loader(val_loader, donor_map):
    from torch.utils.data import DataLoader, Subset
    from datasets.make_dataloader import val_collate_fn

    dataset = Subset(val_loader.dataset, donor_map.tolist())
    return DataLoader(
        dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_collate_fn,
        pin_memory=bool(getattr(val_loader, "pin_memory", False)),
        drop_last=False,
    )


def arm_context(tapf, arm):
    if arm == "psg_bypass_0":
        return PsgBypass(tapf.psg_bank, (0,))
    if arm == "psg_bypass_1":
        return PsgBypass(tapf.psg_bank, (1,))
    if arm == "psg_bypass_all":
        return PsgBypass(tapf.psg_bank, (0, 1))
    return contextlib.nullcontext()


def run_regular_arm(model, tapf, val_loader, arm, device,
                    collect_matching_stats=False):
    transform = None
    if arm in (
        "channel_cycle",
        "left_right_channel_swap",
        "confidence_permutation",
        "spatial_constant",
        "zero_field",
    ):
        transform = lambda field: transform_field(field, arm)
    exploding = ExplodingPose()
    descriptors = []
    pids_all = []
    camids_all = []
    anchor_hook = AnchorFieldHook(
        tapf.anchor,
        transform=transform,
        collect_matching_stats=collect_matching_stats,
    )
    bypass = arm_context(tapf, arm)
    with anchor_hook, bypass as bypass_audit, GateObserver(tapf.psg_bank) as gates:
        for batch_index, batch in enumerate(val_loader):
            images, pids, camids, camids_batch, viewids, paths = batch
            del paths
            images = images.to(device, non_blocking=True)
            camids_batch = camids_batch.to(device, non_blocking=True)
            viewids = viewids.to(device, non_blocking=True)
            pose = pose_for_arm(arm, images.shape[0], device, exploding)
            with torch.no_grad():
                descriptor = forward_descriptor(
                    model, images, camids_batch, viewids, pose
                )
            descriptors.append(descriptor.detach().cpu())
            pids_all.extend(list(pids))
            camids_all.extend(list(camids))
            if (batch_index + 1) % 100 == 0:
                print("[%s] batches=%d" % (arm, batch_index + 1), flush=True)
    bypass_summary = None
    if hasattr(bypass_audit, "summary"):
        bypass_summary = bypass_audit.summary()
    if exploding.accesses:
        raise RuntimeError("Exploding pose was accessed")
    return {
        "descriptors": torch.cat(descriptors, dim=0),
        "pids": pids_all,
        "camids": camids_all,
        "anchor": anchor_hook.summary(),
        "gates": gates.summary(),
        "bypass": bypass_summary,
        "external_pose_accesses": exploding.accesses,
        "matching_stats": (
            torch.cat(anchor_hook.matching_stats, dim=0).numpy()
            if collect_matching_stats else None
        ),
    }


def run_matched_arm(model, tapf, val_loader, donor_loader, device):
    descriptors = []
    pids_all = []
    camids_all = []
    recipient_hook = AnchorFieldHook(tapf.anchor)
    donor_field = {"value": None}

    def replacement(field):
        value = donor_field["value"]
        if value is None:
            raise RuntimeError("Matched donor field was not captured")
        if value.shape != field.shape or value.device != field.device:
            raise RuntimeError("Matched donor field contract mismatch")
        return value

    recipient_hook.transform = replacement
    gates = GateObserver(tapf.psg_bank)
    for batch_index, (recipient, donor) in enumerate(zip(val_loader, donor_loader)):
        images, pids, camids, camids_batch, viewids, paths = recipient
        (donor_images, donor_pids, donor_camids, donor_camids_batch,
         donor_viewids, donor_paths) = donor
        del paths, donor_pids, donor_camids, donor_paths
        images = images.to(device, non_blocking=True)
        camids_batch = camids_batch.to(device, non_blocking=True)
        viewids = viewids.to(device, non_blocking=True)
        donor_images = donor_images.to(device, non_blocking=True)
        donor_camids_batch = donor_camids_batch.to(device, non_blocking=True)
        donor_viewids = donor_viewids.to(device, non_blocking=True)

        # Donor forward is outside both recipient replacement and gate observer.
        with capture_one_field(tapf.anchor) as captured:
            with torch.no_grad():
                forward_descriptor(
                    model, donor_images, donor_camids_batch, donor_viewids, None
                )
        if len(captured) != 1:
            raise RuntimeError("Matched donor anchor capture count mismatch")
        donor_field["value"] = captured[0]
        with recipient_hook, gates:
            with torch.no_grad():
                descriptor = forward_descriptor(
                    model, images, camids_batch, viewids, None
                )
        descriptors.append(descriptor.detach().cpu())
        pids_all.extend(list(pids))
        camids_all.extend(list(camids))
        donor_field["value"] = None
        if (batch_index + 1) % 100 == 0:
            print("[matched_wrong_field] batches=%d" % (batch_index + 1), flush=True)
    return {
        "descriptors": torch.cat(descriptors, dim=0),
        "pids": pids_all,
        "camids": camids_all,
        "anchor": recipient_hook.summary(),
        "gates": gates.summary(),
        "bypass": None,
        "external_pose_accesses": 0,
        "matching_stats": None,
    }


def roll_validation_batch(batch):
    images, pids, camids, camids_batch, viewids, paths = batch
    if images.shape[0] <= 1:
        raise RuntimeError("Route smoke requires batch size > 1")
    return (
        images.roll(1, dims=0),
        tuple(list(pids)[-1:] + list(pids)[:-1]),
        tuple(list(camids)[-1:] + list(camids)[:-1]),
        camids_batch.roll(1, dims=0),
        viewids.roll(1, dims=0),
        tuple(list(paths)[-1:] + list(paths)[:-1]),
    )


def route_smoke(model, tapf, val_loader, device):
    batch = next(iter(val_loader))
    loader = [batch]
    summaries = {}
    descriptors = {}
    smoke_arms = (
        "correct_start",
        "external_correct",
        "external_shuffle",
        "external_none",
        "external_exploding",
        "channel_cycle",
        "left_right_channel_swap",
        "confidence_permutation",
        "spatial_constant",
        "zero_field",
        "psg_bypass_0",
        "psg_bypass_1",
        "psg_bypass_all",
        "correct_end",
    )
    for arm in smoke_arms:
        run = run_regular_arm(model, tapf, loader, arm, device)
        descriptors[arm] = run.pop("descriptors")
        run.pop("pids")
        run.pop("camids")
        run.pop("matching_stats")
        summaries[arm] = run
    donor_batch = roll_validation_batch(batch)
    matched = run_matched_arm(model, tapf, loader, [donor_batch], device)
    descriptors["matched_wrong_field"] = matched.pop("descriptors")
    matched.pop("pids")
    matched.pop("camids")
    matched.pop("matching_stats")
    summaries["matched_wrong_field"] = matched

    reference = descriptors["correct_start"]
    exact_external = {}
    for arm in (
        "external_correct",
        "external_shuffle",
        "external_none",
        "external_exploding",
    ):
        exact_external[arm] = bool(torch.equal(descriptors[arm], reference))
        if not exact_external[arm]:
            raise RuntimeError("Route smoke external parity failed: %s" % arm)
    if not torch.equal(descriptors["correct_end"], reference):
        raise RuntimeError("Route smoke correct repeat failed")
    if not torch.equal(
        descriptors["zero_field"], descriptors["psg_bypass_all"]
    ):
        raise RuntimeError("Route smoke NULL identity failed")
    changed = {}
    for arm in (
        "channel_cycle",
        "left_right_channel_swap",
        "confidence_permutation",
        "matched_wrong_field",
        "spatial_constant",
        "zero_field",
        "psg_bypass_0",
        "psg_bypass_1",
        "psg_bypass_all",
    ):
        changed[arm] = float((descriptors[arm] - reference).abs().amax().item())
        if changed[arm] <= 0.0:
            raise RuntimeError("Route smoke dead intervention: %s" % arm)
    return {
        "status": "EXP392_PHASE0A_ROUTE_SMOKE_PASS",
        "batch_size": int(reference.shape[0]),
        "descriptor_shape": list(reference.shape),
        "external_pose_exact": exact_external,
        "correct_repeat_exact": True,
        "null_identity_exact": True,
        "descriptor_max_abs_delta": changed,
        "arms": summaries,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rounded_percent(metrics):
    return tuple(round(100.0 * value, 1) for value in metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--donor-map-output", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--route-smoke-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    donor_map_path = Path(args.donor_map_output).resolve()

    if git_head(repo_root) != EXPECTED_EXECUTION_HEAD:
        raise RuntimeError("Execution HEAD mismatch")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Config SHA mismatch")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA mismatch")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from config import cfg
    from datasets import make_dataloader
    from model import make_model

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    loaders = make_dataloader(cfg)
    _, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = loaders
    if hasattr(train_loader_normal.dataset, "pose_store") or hasattr(
        val_loader.dataset, "pose_store"
    ):
        raise RuntimeError("RGB-only audit loader unexpectedly owns pose_store")

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load failed")
    for name, value in model.state_dict().items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise RuntimeError("Non-finite state tensor: %s" % name)

    before_state_sha = state_sha256(model)
    model = model.cuda().eval()
    tapf = model.base.tapf
    if len(tapf.psg_bank) != 2:
        raise RuntimeError("Phase 0A requires two clean D0 PSG consumers")
    device = torch.device("cuda")

    if args.route_smoke_only:
        smoke = route_smoke(model, tapf, val_loader, device)
        model = model.cpu()
        after_state_sha = state_sha256(model)
        if after_state_sha != before_state_sha:
            raise RuntimeError("Model state changed during route smoke")
        smoke["state_sha256_before"] = before_state_sha
        smoke["state_sha256_after"] = after_state_sha
        write_json(output_path, smoke)
        print(smoke["status"], flush=True)
        print("output_sha256=%s" % sha256_file(output_path), flush=True)
        return

    results = {}
    correct_descriptors = None
    correct_query = None
    correct_metrics = None
    correct_pids = None
    correct_camids = None
    donor_loader = None
    donor_map_sha = None
    zero_descriptors = None
    bypass_all_descriptors = None

    for arm in FULL_ARMS:
        print("=== Phase0A arm: %s ===" % arm, flush=True)
        if arm == "matched_wrong_field":
            if donor_loader is None:
                raise RuntimeError("Donor map was not prepared by correct_start")
            run = run_matched_arm(model, tapf, val_loader, donor_loader, device)
        else:
            run = run_regular_arm(
                model,
                tapf,
                val_loader,
                arm,
                device,
                collect_matching_stats=(arm == "correct_start"),
            )
        descriptors = run.pop("descriptors")
        pids = run.pop("pids")
        camids = run.pop("camids")
        matching_stats = run.pop("matching_stats")

        if correct_descriptors is None:
            correct_descriptors = descriptors
            correct_pids = list(pids)
            correct_camids = list(camids)
            correct_metrics, correct_query = retrieval_metrics(
                descriptors,
                pids,
                camids,
                num_query,
                bool(cfg.TEST.FEAT_NORM),
            )
            if rounded_percent(correct_metrics) != EXPECTED_CORRECT_PERCENT:
                raise RuntimeError(
                    "Correct metrics mismatch: %r" % (rounded_percent(correct_metrics),)
                )
            donor_map, donor_summary = build_donor_map(
                matching_stats, pids, camids, num_query
            )
            donor_payload = {
                "execution_head": EXPECTED_EXECUTION_HEAD,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "num_query": num_query,
                "summary": donor_summary,
                "donor_indices": donor_map.tolist(),
            }
            write_json(donor_map_path, donor_payload)
            donor_map_sha = sha256_file(donor_map_path)
            donor_loader = make_donor_loader(val_loader, donor_map)
            query = correct_query
            metrics = correct_metrics
        else:
            if pids != correct_pids or camids != correct_camids:
                raise RuntimeError("Evaluation ordering changed for %s" % arm)
            exact = bool(torch.equal(descriptors, correct_descriptors))
            if exact:
                metrics, query = correct_metrics, correct_query
            else:
                metrics, query = retrieval_metrics(
                    descriptors,
                    pids,
                    camids,
                    num_query,
                    bool(cfg.TEST.FEAT_NORM),
                )

        bootstrap = paired_bootstrap(
            query,
            correct_query,
            repeats=args.bootstrap_repeats,
        )
        result = dict(run)
        result.update({
            "metrics_percent": [100.0 * value for value in metrics],
            "delta_percent": [
                100.0 * (value - baseline)
                for value, baseline in zip(metrics, correct_metrics)
            ],
            "descriptor_delta": descriptor_delta(
                descriptors, correct_descriptors
            ),
            "paired_bootstrap": bootstrap,
        })
        results[arm] = result
        if arm == "zero_field":
            zero_descriptors = descriptors.clone()
        if arm == "psg_bypass_all":
            bypass_all_descriptors = descriptors.clone()

        partial = {
            "status": "PHASE0A_RUNNING",
            "completed_arms": list(results),
            "results": results,
            "donor_map_sha256": donor_map_sha,
        }
        write_json(output_path, partial)
        print(
            "[%s] metrics=%s delta=%s exact=%s"
            % (
                arm,
                [round(value, 6) for value in result["metrics_percent"]],
                [round(value, 6) for value in result["delta_percent"]],
                result["descriptor_delta"]["elementwise_exact"],
            ),
            flush=True,
        )

    if zero_descriptors is None or bypass_all_descriptors is None:
        raise RuntimeError("NULL identity arms missing")
    if not torch.equal(zero_descriptors, bypass_all_descriptors):
        raise RuntimeError("zero_field != psg_bypass_all descriptor")
    if not results["correct_end"]["descriptor_delta"]["elementwise_exact"]:
        raise RuntimeError("correct_start != correct_end")
    for arm in (
        "external_correct",
        "external_shuffle",
        "external_none",
        "external_exploding",
    ):
        if not results[arm]["descriptor_delta"]["elementwise_exact"]:
            raise RuntimeError("External pose parity failed for %s" % arm)

    model = model.cpu()
    after_state_sha = state_sha256(model)
    if after_state_sha != before_state_sha:
        raise RuntimeError("Model state changed during frozen audit")

    channel_delta = results["channel_cycle"]["delta_percent"][0]
    matched_delta = results["matched_wrong_field"]["delta_percent"][0]
    bypass_delta = results["psg_bypass_all"]["delta_percent"][0]
    correct_minus_channel = -channel_delta
    correct_minus_matched = -matched_delta
    correct_minus_bypass = -bypass_delta
    if (correct_minus_channel < 0.3 and correct_minus_matched < 0.3
            and correct_minus_bypass >= 0.3):
        verdict = "CONSUMER_EFFECTIVE_JOINT_SEMANTICS_NOT_IDENTIFIED"
    elif (correct_minus_channel >= 0.3 or correct_minus_matched >= 0.3):
        verdict = "SEMANTIC_SENSITIVITY_CANDIDATE_REQUIRES_MULTI_ARM_CONFIRMATION"
    elif correct_minus_bypass < 0.3:
        verdict = "TAPF_TOTAL_EFFECT_TOO_SMALL_FOR_BINDING_CLAIM"
    else:
        verdict = "INCONCLUSIVE"

    final = {
        "status": "EXP392_PHASE0A_COMPLETE",
        "verdict": verdict,
        "protocol_threshold_percent": 0.3,
        "execution": {
            "repo_root": str(repo_root),
            "execution_head": EXPECTED_EXECUTION_HEAD,
            "config": str(config_path),
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "state_sha256_before": before_state_sha,
            "state_sha256_after": after_state_sha,
            "strict_state_tensors": len(payload),
            "num_query": num_query,
            "dataset_size": len(val_loader.dataset),
        },
        "donor_map": {
            "path": str(donor_map_path),
            "sha256": donor_map_sha,
        },
        "null_identity_exact": True,
        "correct_repeat_exact": True,
        "external_pose_exact": True,
        "results": results,
    }
    write_json(output_path, final)
    print("EXP392_PHASE0A_COMPLETE", flush=True)
    print("verdict=%s" % verdict, flush=True)
    print("output_sha256=%s" % sha256_file(output_path), flush=True)


if __name__ == "__main__":
    main()
