#!/usr/bin/env python3
"""One-shot RGB-only counterfactual audit for the sealed exp404 SPK checkpoint."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


CONFIG_SHA256 = "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d"
CHECKPOINT_SHA256 = "03dbebb341e9d085e3d697505b8793cca217fca4a3b8f2a1f28fc512336e7d23"
CLEAN_D0_REFERENCE = {
    "mAP": 0.575587756578,
    "rank1": 0.676923076923,
    "rank5": 0.807692307692,
    "rank10": 0.845701357466,
}
TRAIN_LOG_ROUNDED_REFERENCE = {
    "mAP_percent": 57.4,
    "rank1_percent": 67.5,
    "rank5_percent": 79.7,
    "rank10_percent": 85.0,
}
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "72ff5a609c7a080d848e96a2c12239795388441cc13b85519ef2cbf42f04bf2a",
    "model/make_model.py": "44de28f34b675366606e4ae4734567f50c6ede755fd85280073c514543d61f76",
    "model/backbones/swin_transformer.py": "45e020d20e42db3695a27b123ec9ad76c7c6d4498255c340537a75d6c3665036",
    "datasets/bases.py": "03d231558f46264e4cff0c251b9b728ab4971232ed6c4bb7324ce1964f139c2c",
    "datasets/occluded_duke.py": "f0e7b25e75251643430b699d9c9969fae207c0a85c48855cd0404d61a4228f8e",
    "utils/metrics.py": "8715f845a369688577773afbb974a660e2324961583e0f5ba066e2f93484b7f1",
    "config/defaults.py": "17df7121c3efa5ba967dcb41a185be9aba6b209fd36717faef59569024e62d46",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor state entry: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_finite(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(
        torch.isfinite(value).all()
    )


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_text(command, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp404_counterfactual_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_path_sha256(indices, paths) -> str:
    digest = hashlib.sha256()
    for index, path in zip(indices, paths):
        digest.update(str(int(index)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(Path(path).resolve()).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def finite_metrics(metrics) -> bool:
    return all(np.isfinite(float(value)) for value in metrics.values())


def gpu_compute_pids():
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def rng_snapshot():
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
    }


def rng_equal(left, right) -> bool:
    return (
        torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(
            torch.equal(first, second)
            for first, second in zip(left["torch_cuda"], right["torch_cuda"])
        )
        and left["python"] == right["python"]
        and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
    )


def rng_sha256(snapshot) -> str:
    digest = hashlib.sha256()
    digest.update(snapshot["torch_cpu"].numpy().tobytes())
    for state in snapshot["torch_cuda"]:
        digest.update(state.cpu().numpy().tobytes())
    digest.update(repr(snapshot["python"]).encode("utf-8"))
    digest.update(str(snapshot["numpy"][0]).encode("ascii"))
    digest.update(snapshot["numpy"][1].tobytes())
    digest.update(repr(snapshot["numpy"][2:]).encode("utf-8"))
    return digest.hexdigest()


class IndexedDataset(Dataset):
    def __init__(self, base, indices=None):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return (*self.base[index], index)


def indexed_collate(batch):
    images, pids, camids, viewids, paths, indices = zip(*batch)
    return (
        torch.stack(images, dim=0),
        pids,
        camids,
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        paths,
        torch.tensor(indices, dtype=torch.int64),
    )


def build_loader(base, indices, batch_size, workers, generator):
    return DataLoader(
        IndexedDataset(base, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=indexed_collate,
        generator=generator,
    )


class ExplodingPose(dict):
    accesses = 0

    def __getitem__(self, key):
        type(self).accesses += 1
        raise RuntimeError("RGB-only audit accessed external pose")

    def get(self, key, default=None):
        type(self).accesses += 1
        raise RuntimeError("RGB-only audit accessed external pose")

    def __iter__(self):
        type(self).accesses += 1
        raise RuntimeError("RGB-only audit iterated external pose")


class ForbiddenSemanticReadGuard:
    def __init__(self):
        self.accesses = []
        self._builtins_open = builtins.open
        self._io_open = io.open
        self._os_open = os.open
        self._derived_root = (Path("/mnt1") / "afrderived").resolve()

    def _forbidden(self, value):
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fsdecode(value)).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            return False
        return (
            path == self._derived_root
            or self._derived_root in path.parents
            or path.suffix == ".safetensors"
            or "codebook" in path.name.lower()
        )

    def _record(self, value):
        try:
            rendered = str(Path(os.fsdecode(value)).expanduser().resolve())
        except (TypeError, ValueError, OSError):
            rendered = repr(value)
        self.accesses.append(rendered)
        raise RuntimeError("Forbidden teacher/pose/codebook access")

    def __enter__(self):
        def guarded_builtin(value, *args, **kwargs):
            if self._forbidden(value):
                self._record(value)
            return self._builtins_open(value, *args, **kwargs)

        def guarded_io(value, *args, **kwargs):
            if self._forbidden(value):
                self._record(value)
            return self._io_open(value, *args, **kwargs)

        def guarded_os(value, *args, **kwargs):
            if self._forbidden(value):
                self._record(value)
            return self._os_open(value, *args, **kwargs)

        builtins.open = guarded_builtin
        io.open = guarded_io
        os.open = guarded_os
        return self

    def __exit__(self, exc_type, exc_value, traceback_object):
        builtins.open = self._builtins_open
        io.open = self._io_open
        os.open = self._os_open
        return False


@contextmanager
def capture_generic_mean(kernel, core):
    original = kernel.forward
    had_override = "forward" in kernel.__dict__
    override_before = kernel.__dict__.get("forward")
    report = {
        "calls": 0,
        "rows": 0,
        "zero_mass_rows": 0,
        "sum": torch.zeros(16, dtype=torch.float64),
        "restored_exact": False,
    }

    def patched(global_feature, evidence, presence):
        pooled = core.pooled_evidence(evidence, presence).detach().double().cpu()
        if not bool(torch.isfinite(pooled).all()):
            raise RuntimeError("Non-finite train generic evidence")
        report["calls"] += 1
        report["rows"] += int(pooled.shape[0])
        report["zero_mass_rows"] += int((presence.detach().sum(dim=1) <= 0).sum())
        report["sum"] += pooled.sum(dim=0)
        return original(global_feature, evidence, presence)

    kernel.forward = patched
    try:
        yield report
    finally:
        if had_override:
            kernel.__dict__["forward"] = override_before
        else:
            kernel.__dict__.pop("forward", None)
        report["restored_exact"] = (
            ("forward" in kernel.__dict__) == had_override
            and (not had_override or kernel.__dict__.get("forward") is override_before)
        )


@contextmanager
def patch_spk_inputs(
    kernel,
    arm,
    runtime,
    core,
    evidence_cache,
    presence_cache,
    evidence_seen,
    donor_map,
    generic_mean,
    random_permutations,
    random_signs,
    cluster_assignment,
    cluster_prototypes,
    capture=False,
):
    original = kernel.forward
    had_override = "forward" in kernel.__dict__
    override_before = kernel.__dict__.get("forward")
    report = {
        "calls": 0,
        "rows": 0,
        "captured_rows": 0,
        "duplicate_capture_rows": 0,
        "factor_finite": True,
        "descriptor_finite": True,
        "input_changed_rows": 0,
        "random_abs_multiset_exact": True,
        "random_norm_max_abs_error": 0.0,
        "restored_exact": False,
    }

    def patched(global_feature, evidence, presence):
        indices = runtime.get("indices")
        if indices is None:
            raise RuntimeError("SPK called without absolute validation indices")
        indices = torch.as_tensor(indices, dtype=torch.int64).flatten().cpu()
        if indices.numel() != evidence.shape[0]:
            raise RuntimeError("SPK/index batch mismatch")
        report["calls"] += 1
        report["rows"] += int(indices.numel())
        if capture:
            captured_evidence = evidence.detach().cpu()
            captured_presence = presence.detach().cpu()
            if not bool(
                torch.isfinite(captured_evidence).all()
                and torch.isfinite(captured_presence).all()
            ):
                raise RuntimeError("Non-finite cached SPK input")
            for position, index in enumerate(indices.tolist()):
                if int(evidence_seen[index]) > 0:
                    if not (
                        torch.equal(evidence_cache[index], captured_evidence[position])
                        and torch.equal(presence_cache[index], captured_presence[position])
                    ):
                        raise RuntimeError("Repeated SPK input capture changed")
                    report["duplicate_capture_rows"] += 1
                else:
                    evidence_cache[index].copy_(captured_evidence[position])
                    presence_cache[index].copy_(captured_presence[position])
                    report["captured_rows"] += 1
                evidence_seen[index] += 1

        changed_evidence = evidence
        changed_presence = presence
        if arm in core.INTERVENTION_ARMS:
            if arm == "wrong_rgb":
                selected = donor_map.index_select(0, indices)
                if not bool((evidence_seen[selected] > 0).all()):
                    raise RuntimeError("Wrong-RGB donor SPK inputs are not cached")
            changed_evidence, changed_presence = core.intervene_spk_inputs(
                evidence,
                presence,
                arm,
                absolute_indices=indices,
                donor_map=donor_map,
                evidence_cache=evidence_cache,
                presence_cache=presence_cache,
                generic_mean=generic_mean,
                random_permutations=random_permutations,
                random_signs=random_signs,
                cluster_assignment=cluster_assignment,
                cluster_prototypes=cluster_prototypes,
            )
            if changed_evidence.shape == evidence.shape:
                evidence_changed = (changed_evidence != evidence).reshape(evidence.shape[0], -1).any(1)
            else:
                evidence_changed = torch.ones(evidence.shape[0], dtype=torch.bool, device=evidence.device)
            if changed_presence.shape == presence.shape:
                presence_changed = (changed_presence != presence).reshape(presence.shape[0], -1).any(1)
            else:
                presence_changed = torch.ones(presence.shape[0], dtype=torch.bool, device=presence.device)
            report["input_changed_rows"] += int((evidence_changed | presence_changed).sum())
            if arm == "random_key":
                original_abs = evidence.detach().abs().sort(dim=-1).values
                changed_abs = changed_evidence.detach().abs().sort(dim=-1).values
                report["random_abs_multiset_exact"] = bool(
                    report["random_abs_multiset_exact"]
                    and torch.equal(original_abs, changed_abs)
                )
                error = (
                    evidence.detach().float().norm(dim=-1)
                    - changed_evidence.detach().float().norm(dim=-1)
                ).abs().max()
                report["random_norm_max_abs_error"] = max(
                    float(report["random_norm_max_abs_error"]), float(error)
                )

        if arm == "all_product_bypass":
            descriptor = global_feature
            factor = torch.ones(
                global_feature.shape[0],
                core.GROUPS,
                device=global_feature.device,
                dtype=torch.float32,
            )
        else:
            descriptor, factor = original(
                global_feature,
                changed_evidence,
                changed_presence,
            )
        report["factor_finite"] = bool(
            report["factor_finite"] and torch.isfinite(factor).all()
        )
        report["descriptor_finite"] = bool(
            report["descriptor_finite"] and torch.isfinite(descriptor).all()
        )
        return descriptor, factor

    kernel.forward = patched
    try:
        yield report
    finally:
        if had_override:
            kernel.__dict__["forward"] = override_before
        else:
            kernel.__dict__.pop("forward", None)
        report["restored_exact"] = (
            ("forward" in kernel.__dict__) == had_override
            and (not had_override or kernel.__dict__.get("forward") is override_before)
        )


@contextmanager
def count_psg_calls(tapf):
    original = tapf.apply_gate
    had_override = "apply_gate" in tapf.__dict__
    override_before = tapf.__dict__.get("apply_gate")
    report = {"calls": [0, 0], "restored_exact": False}

    def patched(bank_index, tokens, hw_shape, state):
        if bank_index not in (0, 1):
            raise RuntimeError(f"Unexpected PSG bank: {bank_index}")
        report["calls"][bank_index] += 1
        return original(bank_index, tokens, hw_shape, state)

    tapf.apply_gate = patched
    try:
        yield report
    finally:
        if had_override:
            tapf.__dict__["apply_gate"] = override_before
        else:
            tapf.__dict__.pop("apply_gate", None)
        report["restored_exact"] = (
            ("apply_gate" in tapf.__dict__) == had_override
            and (not had_override or tapf.__dict__.get("apply_gate") is override_before)
        )


def collect_descriptors(
    model,
    loader,
    num_query,
    cfg,
    device,
    runtime,
    canonical_records,
    compute_metrics,
):
    evaluator = None
    if compute_metrics:
        from utils.metrics import R1_mAP_eval

        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=cfg.TEST.RE_RANKING,
        )
        evaluator.reset()
    descriptors = []
    seen_indices = []
    seen_paths = []
    model.eval()
    with torch.no_grad():
        for image, pid, camid, camids, target_view, paths, indices in loader:
            expected_paths = tuple(canonical_records[int(index)][0] for index in indices.tolist())
            expected_pid = tuple(int(canonical_records[int(index)][1]) for index in indices.tolist())
            expected_camid = tuple(int(canonical_records[int(index)][2]) for index in indices.tolist())
            if tuple(paths) != expected_paths:
                raise RuntimeError("Validation absolute-index/path mismatch")
            if tuple(int(value) for value in pid) != expected_pid:
                raise RuntimeError("Validation PID/index mismatch")
            if tuple(int(value) for value in camid) != expected_camid:
                raise RuntimeError("Validation camera/index mismatch")
            image = image.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            runtime["indices"] = indices.clone()
            try:
                descriptor, _ = model(
                    image,
                    cam_label=camids,
                    view_label=target_view,
                )
            finally:
                runtime["indices"] = None
            if not bool(torch.isfinite(descriptor).all()):
                raise RuntimeError("Non-finite retrieval descriptor")
            detached = descriptor.detach().cpu()
            descriptors.append(detached)
            seen_indices.extend(int(index) for index in indices.tolist())
            seen_paths.extend(str(path) for path in paths)
            if evaluator is not None:
                evaluator.update((detached, pid, camid))
    descriptor_matrix = torch.cat(descriptors, dim=0)
    metrics = None
    if evaluator is not None:
        cmc, mean_ap, *_ = evaluator.compute()
        metrics = {
            "mAP": float(mean_ap),
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "rank10": float(cmc[9]),
        }
    return {
        "descriptors": descriptor_matrix,
        "metrics": metrics,
        "indices": seen_indices,
        "paths": seen_paths,
        "path_sha256": canonical_path_sha256(seen_indices, seen_paths),
    }


def collect_train_generic(
    model,
    kernel,
    loader,
    device,
    core,
    loader_generator,
    loader_generator_state,
    baseline_state_sha,
):
    loader_generator.set_state(loader_generator_state)
    rng_before = rng_snapshot()
    state_before = tensor_state_sha256(model.state_dict())
    if state_before != baseline_state_sha:
        raise RuntimeError("Model state drift before train generic collection")
    model.eval()
    with capture_generic_mean(kernel, core) as report:
        with torch.no_grad():
            for image, _, _, camids, target_view, _, _ in loader:
                descriptor, _ = model(
                    image.to(device),
                    cam_label=camids.to(device),
                    view_label=target_view.to(device),
                )
                if not bool(torch.isfinite(descriptor).all()):
                    raise RuntimeError("Non-finite train generic descriptor")
    loader_generator.set_state(loader_generator_state)
    rng_after = rng_snapshot()
    state_after = tensor_state_sha256(model.state_dict())
    if int(report["rows"]) <= 0:
        raise RuntimeError("Train generic collection is empty")
    generic_mean = report["sum"] / int(report["rows"])
    if not bool(torch.isfinite(generic_mean).all()):
        raise RuntimeError("Non-finite frozen generic mean")
    serializable = {
        "calls": int(report["calls"]),
        "rows": int(report["rows"]),
        "zero_mass_rows": int(report["zero_mass_rows"]),
        "restored_exact": bool(report["restored_exact"]),
        "state_restored_exact": state_after == state_before == baseline_state_sha,
        "rng_restored_exact": rng_equal(rng_before, rng_after),
        "loader_rng_restored_exact": torch.equal(
            loader_generator.get_state(), loader_generator_state
        ),
        "rng_sha256_before": rng_sha256(rng_before),
        "rng_sha256_after": rng_sha256(rng_after),
        "mean": [float(value) for value in generic_mean.tolist()],
    }
    return generic_mean.float(), serializable


def one_rgb_only_check(model, loader, device, loader_generator, generator_state):
    loader_generator.set_state(generator_state)
    iterator = iter(loader)
    image, _, _, camids, target_view, _, indices = next(iterator)
    del iterator
    image = image.to(device)
    camids = camids.to(device)
    target_view = target_view.to(device)
    model.eval()
    with torch.no_grad():
        descriptor_none, _ = model(
            image,
            cam_label=camids,
            view_label=target_view,
            pose_batch=None,
        )
        ExplodingPose.accesses = 0
        descriptor_exploding, _ = model(
            image,
            cam_label=camids,
            view_label=target_view,
            pose_batch=ExplodingPose(),
        )
    loader_generator.set_state(generator_state)
    return {
        "descriptor_exact": torch.equal(descriptor_none, descriptor_exploding),
        "descriptor_finite": bool(torch.isfinite(descriptor_none).all()),
        "pose_accesses": int(ExplodingPose.accesses),
        "rows": int(indices.numel()),
    }


def run_arm(
    arm,
    model,
    tapf,
    kernel,
    loader,
    num_query,
    cfg,
    device,
    runtime,
    records,
    core,
    assets,
    loader_generator,
    loader_generator_state,
    baseline_state_sha,
    compute_metrics,
):
    loader_generator.set_state(loader_generator_state)
    rng_before = rng_snapshot()
    state_before = tensor_state_sha256(model.state_dict())
    if state_before != baseline_state_sha:
        raise RuntimeError(f"Model state drift before arm {arm}")
    with patch_spk_inputs(
        kernel,
        arm,
        runtime,
        core,
        assets["evidence_cache"],
        assets["presence_cache"],
        assets["evidence_seen"],
        assets["donor_map"],
        assets["generic_mean"],
        assets["random_permutations"],
        assets["random_signs"],
        assets["cluster_assignment"],
        assets["cluster_prototypes"],
        capture=arm == "correct",
    ) as spk_report:
        with count_psg_calls(tapf) as psg_report:
            output = collect_descriptors(
                model,
                loader,
                num_query,
                cfg,
                device,
                runtime,
                records,
                compute_metrics,
            )
    loader_generator.set_state(loader_generator_state)
    rng_after = rng_snapshot()
    state_after = tensor_state_sha256(model.state_dict())
    expected_calls = [len(loader), len(loader)]
    report = {
        "spk": dict(spk_report),
        "psg": dict(psg_report),
        "rows": len(output["indices"]),
        "unique_indices": len(set(output["indices"])),
        "path_sha256": output["path_sha256"],
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_restored_exact": state_after == state_before == baseline_state_sha,
        "spk_restored_exact": bool(spk_report["restored_exact"]),
        "psg_restored_exact": bool(psg_report["restored_exact"]),
        "spk_calls_exact": int(spk_report["calls"]) == len(loader),
        "psg_calls_exact": list(psg_report["calls"]) == expected_calls,
        "rng_restored_exact": rng_equal(rng_before, rng_after),
        "loader_rng_restored_exact": torch.equal(
            loader_generator.get_state(), loader_generator_state
        ),
        "rng_sha256_before": rng_sha256(rng_before),
        "rng_sha256_after": rng_sha256(rng_after),
    }
    return output, report


def rounded_reference_match(metrics) -> bool:
    observed = {
        "mAP_percent": round(float(metrics["mAP"]) * 100.0, 1),
        "rank1_percent": round(float(metrics["rank1"]) * 100.0, 1),
        "rank5_percent": round(float(metrics["rank5"]) * 100.0, 1),
        "rank10_percent": round(float(metrics["rank10"]) * 100.0, 1),
    }
    return observed == TRAIN_LOG_ROUNDED_REFERENCE


def run(args):
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    core_path = Path(args.core).resolve()
    result_path = Path(args.result).resolve()
    script_path = Path(__file__).resolve()
    if Path("/home/afr") not in result_path.parents:
        raise RuntimeError("Remote result must remain under /home/afr")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo_root))

    expected_checkpoint = output_dir / "transformer_120.pth"
    checkpoint_files_before = sorted(
        str(path.resolve())
        for path in output_dir.glob("*")
        if path.is_file() and path.suffix in {".pth", ".pt", ".ckpt"}
    )
    source_sha_before = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    checkpoint_sha_before = sha256_file(checkpoint_path)
    config_sha_before = sha256_file(config_path)
    core_sha_before = sha256_file(core_path)
    gpu_pids_before = gpu_compute_pids()
    repo_status_before = run_text(["git", "status", "--porcelain"], repo_root)
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_state_sha = tensor_state_sha256(payload)
    state_names = tuple(payload)
    forbidden_state_tokens = {"teacher", "clip", "codebook", "text", "pose_batch"}
    cpu_gates = {
        "repo_tracked_clean": not bool(
            run_text(["git", "status", "--porcelain", "--untracked-files=no"], repo_root)
        ),
        "repo_untracked_exact_formal_output": repo_status_before == "?? log/",
        "source_sha_exact": source_sha_before == EXPECTED_SOURCE_SHA256,
        "config_sha_exact": config_sha_before == CONFIG_SHA256,
        "checkpoint_path_exact": checkpoint_path == expected_checkpoint,
        "checkpoint_regular": checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
        "checkpoint_sha_exact": checkpoint_sha_before == CHECKPOINT_SHA256,
        "checkpoint_unique": checkpoint_files_before == [str(checkpoint_path)],
        "checkpoint_state_all_tensors": all(torch.is_tensor(value) for value in payload.values()),
        "checkpoint_state_finite": all(tensor_finite(value) for value in payload.values()),
        "state_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden_state_tokens)
            for name in state_names
        ),
        "evidence_head_retained": any("anchor.evidence_head" in name for name in state_names),
        "two_psg_retained": all(
            any(f"psg_bank.{index}." in name for name in state_names)
            for index in (0, 1)
        ),
        "spk_parameter_free_in_state": not any("semantic_product_kernel" in name for name in state_names),
        "gpu_idle_before": not gpu_pids_before,
        "cuda_uninitialized_before": not torch.cuda.is_initialized(),
    }
    if not all(cpu_gates.values()):
        raise RuntimeError(f"CPU launch gates failed: {cpu_gates}")

    from config import cfg
    from datasets.bases import ImageDataset
    from datasets.occluded_duke import OccludedDuke
    from model import make_model

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    data_root = Path(str(cfg.DATASETS.ROOT_DIR).strip("()'"))
    dataset = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset.query) + list(dataset.gallery)
    train_records = list(dataset.train)
    num_query = len(dataset.query)
    pids = torch.tensor([int(record[1]) for record in records])
    camids = torch.tensor([int(record[2]) for record in records])
    core = load_module(core_path)
    donor_map = core.build_global_donor_map(pids, camids, num_query)
    donor_summary = core.validate_donor_map(donor_map, pids, camids, num_query)
    random_permutations, random_signs = core.build_signed_permutations(len(records))
    cluster_assignment = core.build_balanced_cluster_assignment(len(records))
    cluster_summary = core.validate_cluster_assignment(
        cluster_assignment, pids, camids
    )
    val_transform = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    base_dataset = ImageDataset(records, val_transform)
    train_dataset = ImageDataset(train_records, val_transform)
    all_indices = list(range(len(records)))
    preflight = int(args.preflight_samples) > 0
    selected_indices = (
        all_indices[: min(int(args.preflight_samples), len(all_indices))]
        if preflight
        else all_indices
    )
    train_indices = (
        list(range(min(int(args.generic_preflight_samples), len(train_records))))
        if preflight
        else list(range(len(train_records)))
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(1234)
    loader_generator_state = loader_generator.get_state().clone()
    workers = 0 if preflight else int(cfg.DATALOADER.NUM_WORKERS)
    loader = build_loader(
        base_dataset,
        selected_indices,
        min(int(cfg.TEST.IMS_PER_BATCH), len(selected_indices)),
        workers,
        loader_generator,
    )
    train_loader = build_loader(
        train_dataset,
        train_indices,
        min(int(cfg.TEST.IMS_PER_BATCH), len(train_indices)),
        workers,
        loader_generator,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID).strip("()'")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    with ForbiddenSemanticReadGuard() as read_guard:
        model = make_model(
            cfg,
            num_class=dataset.num_train_pids,
            camera_num=dataset.num_train_cams,
            view_num=dataset.num_train_vids,
            semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
        )
        incompatible = model.load_state_dict(payload, strict=True)
        loaded_state_sha = tensor_state_sha256(model.state_dict())
        model = model.to(device)
        tapf = model.base.tapf
        kernel = model.semantic_product_kernel
        baseline_state_sha = tensor_state_sha256(model.state_dict())
        strict_gates = {
            "strict_load_no_missing": not incompatible.missing_keys,
            "strict_load_no_unexpected": not incompatible.unexpected_keys,
            "strict_state_exact": loaded_state_sha == checkpoint_state_sha,
            "device_state_exact": baseline_state_sha == checkpoint_state_sha,
            "tapf_semantic_product_exact": bool(getattr(tapf, "semantic_product", False)),
            "tapf_rich_exact": bool(tapf.rich_evidence and tapf.semantic),
            "two_psg_modules": len(tapf.psg_bank) == 2,
            "spk_enabled_exact": bool(model.spk_enabled),
            "spk_parameter_count_zero": sum(p.numel() for p in kernel.parameters()) == 0,
            "spk_buffer_count_zero": sum(b.numel() for b in kernel.buffers()) == 0,
            "spk_groups_exact": int(kernel.groups) == 16,
            "spk_feature_dim_exact": int(kernel.feature_dim) == 768,
            "config_output_dir_exact": Path(cfg.OUTPUT_DIR).resolve() == output_dir,
            "official_data_root_exact": data_root == Path("/mnt1/afrdata"),
            "model_object_teacher_free": all(
                not (set(name.lower().split(".")) & forbidden_state_tokens)
                for name, _ in model.named_modules()
            ),
        }
        if not all(strict_gates.values()):
            raise RuntimeError(f"Strict model gates failed: {strict_gates}")

        rgb_only = one_rgb_only_check(
            model,
            loader,
            device,
            loader_generator,
            loader_generator_state,
        )
        if not (
            rgb_only["descriptor_exact"]
            and rgb_only["descriptor_finite"]
            and rgb_only["pose_accesses"] == 0
        ):
            raise RuntimeError(f"RGB-only gate failed: {rgb_only}")

        print(json.dumps({"progress": "train_generic_start", "rows": len(train_indices)}), flush=True)
        generic_mean, generic_report = collect_train_generic(
            model,
            kernel,
            train_loader,
            device,
            core,
            loader_generator,
            loader_generator_state,
            baseline_state_sha,
        )
        cluster_prototypes = core.build_cluster_prototypes(generic_mean)
        print(json.dumps({"progress": "train_generic_complete", "rows": generic_report["rows"]}), flush=True)

        evidence_cache = torch.zeros(len(records), 5, 16, dtype=torch.float32)
        presence_cache = torch.zeros(len(records), 5, dtype=torch.float32)
        evidence_seen = torch.zeros(len(records), dtype=torch.int64)
        runtime = {"indices": None}
        assets = {
            "evidence_cache": evidence_cache,
            "presence_cache": presence_cache,
            "evidence_seen": evidence_seen,
            "donor_map": donor_map,
            "generic_mean": generic_mean,
            "random_permutations": random_permutations,
            "random_signs": random_signs,
            "cluster_assignment": cluster_assignment,
            "cluster_prototypes": cluster_prototypes,
        }
        donor_warmup = None
        if preflight:
            selected_set = set(selected_indices)
            donor_indices = sorted(
                set(int(donor_map[index]) for index in selected_indices) - selected_set
            )
            if donor_indices:
                donor_loader = build_loader(
                    base_dataset,
                    donor_indices,
                    min(int(cfg.TEST.IMS_PER_BATCH), len(donor_indices)),
                    0,
                    loader_generator,
                )
                with patch_spk_inputs(
                    kernel,
                    "correct",
                    runtime,
                    core,
                    evidence_cache,
                    presence_cache,
                    evidence_seen,
                    donor_map,
                    generic_mean,
                    random_permutations,
                    random_signs,
                    cluster_assignment,
                    cluster_prototypes,
                    capture=True,
                ) as warmup_report:
                    warmup_output = collect_descriptors(
                        model,
                        donor_loader,
                        num_query,
                        cfg,
                        device,
                        runtime,
                        records,
                        False,
                    )
                donor_warmup = {
                    "indices": donor_indices,
                    "rows": len(warmup_output["indices"]),
                    "path_sha256": warmup_output["path_sha256"],
                    "spk": dict(warmup_report),
                }
            else:
                donor_warmup = {"indices": [], "rows": 0, "spk": {"restored_exact": True}}

        metrics = {}
        descriptors = {}
        arm_reports = {}
        expected_path_sha = canonical_path_sha256(
            selected_indices,
            [records[index][0] for index in selected_indices],
        )
        for arm in core.ARM_ORDER:
            print(json.dumps({"progress": "arm_start", "arm": arm}), flush=True)
            output, arm_report = run_arm(
                arm,
                model,
                tapf,
                kernel,
                loader,
                num_query,
                cfg,
                device,
                runtime,
                records,
                core,
                assets,
                loader_generator,
                loader_generator_state,
                baseline_state_sha,
                not preflight,
            )
            descriptors[arm] = output["descriptors"]
            arm_reports[arm] = arm_report
            if output["metrics"] is not None:
                metrics[arm] = output["metrics"]
            print(
                json.dumps(
                    {
                        "progress": "arm_complete",
                        "arm": arm,
                        "rows": arm_report["rows"],
                        "metrics": output["metrics"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        descriptor_deltas = {
            arm: core.descriptor_delta(descriptors["correct"], descriptors[arm])
            for arm in core.ARM_ORDER[1:]
        }
        read_accesses = list(read_guard.accesses)

    read_guard_restored = (
        builtins.open is read_guard._builtins_open
        and io.open is read_guard._io_open
        and os.open is read_guard._os_open
    )
    source_sha_after = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    checkpoint_sha_after = sha256_file(checkpoint_path)
    config_sha_after = sha256_file(config_path)
    core_sha_after = sha256_file(core_path)
    checkpoint_files_after = sorted(
        str(path.resolve())
        for path in output_dir.glob("*")
        if path.is_file() and path.suffix in {".pth", ".pt", ".ckpt"}
    )
    repo_status_after = run_text(["git", "status", "--porcelain"], repo_root)
    model_state_after = tensor_state_sha256(model.state_dict())
    all_arm_restore = all(
        report["state_restored_exact"]
        and report["spk_restored_exact"]
        and report["psg_restored_exact"]
        and report["spk_calls_exact"]
        and report["psg_calls_exact"]
        and report["rng_restored_exact"]
        and report["loader_rng_restored_exact"]
        for report in arm_reports.values()
    )
    all_arm_coverage = all(
        report["rows"] == len(selected_indices)
        and report["unique_indices"] == len(selected_indices)
        and report["path_sha256"] == expected_path_sha
        for report in arm_reports.values()
    )
    all_descriptor_active = all(
        report["finite"]
        and float(report["mean_l2"]) > 0.0
        and float(report["max_abs"]) > 0.0
        for report in descriptor_deltas.values()
    )
    if preflight:
        correct_evidence_coverage = all(
            int(evidence_seen[int(index)]) > 0
            and int(evidence_seen[int(donor_map[index])]) > 0
            for index in selected_indices
        )
    else:
        correct_evidence_coverage = bool((evidence_seen == 1).all())
    donor_warmup_valid = (
        not preflight
        or (
            donor_warmup is not None
            and donor_warmup["rows"] == len(donor_warmup["indices"])
            and donor_warmup["spk"]["restored_exact"]
        )
    )
    null_bypass_descriptor_exact = torch.equal(
        descriptors["null_zero"], descriptors["all_product_bypass"]
    )
    common_validity = {
        "cpu_gates_all_pass": all(cpu_gates.values()),
        "strict_gates_all_pass": all(strict_gates.values()),
        "rgb_only_exact": rgb_only["descriptor_exact"] and rgb_only["pose_accesses"] == 0,
        "forbidden_teacher_pose_reads_zero": not read_accesses,
        "read_guard_restored_exact": read_guard_restored,
        "generic_train_rows_exact": generic_report["rows"] == len(train_indices),
        "generic_collection_restore_exact": all(
            generic_report[key]
            for key in (
                "restored_exact",
                "state_restored_exact",
                "rng_restored_exact",
                "loader_rng_restored_exact",
            )
        ),
        "donor_warmup_valid": donor_warmup_valid,
        "all_arm_restore_exact": all_arm_restore,
        "all_arm_full_coverage": all_arm_coverage,
        "all_descriptors_finite_active": all_descriptor_active,
        "correct_evidence_coverage": correct_evidence_coverage,
        "null_bypass_descriptor_exact": null_bypass_descriptor_exact,
        "random_key_abs_multiset_exact": arm_reports["random_key"]["spk"]["random_abs_multiset_exact"],
        "random_key_norm_preserved": arm_reports["random_key"]["spk"]["random_norm_max_abs_error"] <= 1e-5,
        "random_cluster_balanced": cluster_summary["count_max_minus_min"] <= 1,
        "random_cluster_pid_coverage": cluster_summary["pid_coverage_min"] >= 40,
        "random_cluster_camera_coverage": cluster_summary["all_cameras_exact"],
        "model_state_terminal_exact": model_state_after == baseline_state_sha,
        "checkpoint_before_after_exact": checkpoint_sha_before == checkpoint_sha_after,
        "config_before_after_exact": config_sha_before == config_sha_after,
        "core_before_after_exact": core_sha_before == core_sha_after,
        "source_before_after_exact": source_sha_before == source_sha_after,
        "checkpoint_list_before_after_exact": checkpoint_files_before == checkpoint_files_after,
        "repo_status_before_after_exact": repo_status_after == repo_status_before,
    }
    if preflight:
        status = "PASS" if all(common_validity.values()) else "FAIL"
        decision = "EXP404_COUNTERFACTUAL_PREFLIGHT_PASS" if status == "PASS" else "EXP404_COUNTERFACTUAL_PREFLIGHT_FAIL"
        adjudication = None
        full_authorized = status == "PASS"
        paper_go = False
    else:
        reference_gates = {
            "correct_train_log_rounded_exact": rounded_reference_match(metrics["correct"]),
            "all_metrics_finite": all(finite_metrics(value) for value in metrics.values()),
            "null_bypass_metrics_exact": metrics["null_zero"] == metrics["all_product_bypass"],
        }
        common_validity.update(reference_gates)
        adjudication = core.adjudicate(metrics, descriptor_deltas, common_validity)
        status = "PASS" if all(common_validity.values()) else "FAIL"
        decision = adjudication["decision"] if status == "PASS" else "EXP404_COUNTERFACTUAL_SEALED_INVALID"
        full_authorized = False
        paper_go = bool(
            status == "PASS"
            and adjudication["mechanism_go"]
            and metrics["correct"]["mAP"] >= CLEAN_D0_REFERENCE["mAP"]
            and metrics["correct"]["rank1"] >= CLEAN_D0_REFERENCE["rank1"]
        )

    return {
        "status": status,
        "mode": "CUDA_PREFLIGHT" if preflight else "FORMAL_FULL",
        "decision": decision,
        "formal_full_authorized": bool(full_authorized),
        "c_track_paper_go": paper_go,
        "validity": common_validity,
        "cpu_gates": cpu_gates,
        "strict_gates": strict_gates,
        "rgb_only": rgb_only,
        "dataset": {
            "validation_count": len(records),
            "train_count": len(train_records),
            "num_query": num_query,
            "selected_count": len(selected_indices),
            "generic_selected_count": len(train_indices),
            "canonical_path_sha256": canonical_path_sha256(
                all_indices, [record[0] for record in records]
            ),
            "selected_path_sha256": expected_path_sha,
            "donor_summary": donor_summary,
            "donor_map_sha256": core.tensor_mapping_sha256({"donor": donor_map}),
            "donor_warmup": donor_warmup,
            "cluster_summary": cluster_summary,
        },
        "generic_mean": generic_report,
        "random_assets_sha256": core.tensor_mapping_sha256(
            {
                "permutations": random_permutations,
                "signs": random_signs,
                "cluster_assignment": cluster_assignment,
                "cluster_prototypes": cluster_prototypes,
            }
        ),
        "metrics": metrics,
        "references": {
            "clean_d0": CLEAN_D0_REFERENCE,
            "train_log_rounded": TRAIN_LOG_ROUNDED_REFERENCE,
        },
        "descriptor_deltas": descriptor_deltas,
        "arm_reports": arm_reports,
        "adjudication": adjudication,
        "assets": {
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "checkpoint_state_sha256": checkpoint_state_sha,
            "config_sha256_before": config_sha_before,
            "config_sha256_after": config_sha_after,
            "core_sha256_before": core_sha_before,
            "core_sha256_after": core_sha_after,
            "source_sha256_before": source_sha_before,
            "source_sha256_after": source_sha_after,
            "checkpoint_files_before": checkpoint_files_before,
            "checkpoint_files_after": checkpoint_files_after,
            "forbidden_accesses": read_accesses,
        },
        "execution": {
            "repo_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
            "repo_status_before": repo_status_before,
            "repo_status_after": repo_status_after,
            "script_sha256": sha256_file(script_path),
            "core_sha256": core_sha_after,
            "gpu_pids_before": gpu_pids_before,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--preflight-samples", type=int, default=0)
    parser.add_argument("--generic-preflight-samples", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result).resolve()
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    if result_path.exists() or temporary.exists():
        raise FileExistsError("Audit result path must be fresh")
    try:
        result = run(args)
    except Exception as error:
        result = {
            "status": "FAIL",
            "mode": "CUDA_PREFLIGHT" if int(args.preflight_samples) > 0 else "FORMAL_FULL",
            "decision": "AUDIT_RUNTIME_FAIL",
            "formal_full_authorized": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
