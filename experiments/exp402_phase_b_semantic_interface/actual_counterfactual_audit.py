#!/usr/bin/env python3
"""Read-only RGB counterfactual audit for the sealed exp401 checkpoint."""

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


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
CONFIG_SHA256 = "c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84"
CHECKPOINT_SHA256 = "fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc"
CHECKPOINT_STATE_SHA256 = (
    "3c2af267a4d15cc7f199c6b55fefcbb30c05f0c8f47940de13769e555c4d4035"
)
CORE_SHA256 = "6e9ac9cfc03d70606ee34f77af39accc6b66c89ee9974fee48ded1d6951dfb54"
REFERENCE_TOLERANCE = 5e-8
FULL_REFERENCE = {
    "mAP": 0.5712300755952,
    "rank1": 0.6728506684303284,
    "rank5": 0.8027149438858032,
    "rank10": 0.8475112915039062,
}
ALL_BYPASS_REFERENCE = {
    "mAP": 0.5700358607568072,
    "rank1": 0.6737556457519531,
    "rank5": 0.800452470779419,
    "rank10": 0.8461538553237915,
}
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "model/backbones/swin_transformer.py":
        "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/bases.py": "03d231558f46264e4cff0c251b9b728ab4971232ed6c4bb7324ce1964f139c2c",
    "datasets/occluded_duke.py":
        "f0e7b25e75251643430b699d9c9969fae207c0a85c48855cd0404d61a4228f8e",
    "utils/metrics.py": "8715f845a369688577773afbb974a660e2324961583e0f5ba066e2f93484b7f1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
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
    if value.is_floating_point() or value.is_complex():
        return bool(torch.isfinite(value).all())
    return True


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
    spec = importlib.util.spec_from_file_location("exp402_core", str(path))
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


def metrics_match(observed, reference) -> bool:
    return all(
        abs(float(observed[key]) - float(reference[key]))
        <= REFERENCE_TOLERANCE
        for key in reference
    )


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
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        return False
    if not all(
        torch.equal(first, second)
        for first, second in zip(left["torch_cuda"], right["torch_cuda"])
    ):
        return False
    if left["python"] != right["python"]:
        return False
    a_numpy = left["numpy"]
    b_numpy = right["numpy"]
    return (
        a_numpy[0] == b_numpy[0]
        and np.array_equal(a_numpy[1], b_numpy[1])
        and a_numpy[2:] == b_numpy[2:]
    )


def rng_sha256(snapshot) -> str:
    digest = hashlib.sha256()
    digest.update(snapshot["torch_cpu"].numpy().tobytes())
    for state in snapshot["torch_cuda"]:
        digest.update(state.cpu().numpy().tobytes())
    digest.update(repr(snapshot["python"]).encode("utf-8"))
    numpy_state = snapshot["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode("utf-8"))
    return digest.hexdigest()


class IndexedDataset(Dataset):
    def __init__(self, base, indices=None):
        self.base = base
        self.indices = (
            list(range(len(base)))
            if indices is None
            else [int(index) for index in indices]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        absolute_index = self.indices[position]
        return (*self.base[absolute_index], absolute_index)


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
        if path == self._derived_root or self._derived_root in path.parents:
            return True
        return path.suffix == ".safetensors" or "codebook" in path.name.lower()

    def _record(self, value):
        try:
            rendered = str(Path(os.fsdecode(value)).expanduser().resolve())
        except (TypeError, ValueError, OSError):
            rendered = repr(value)
        self.accesses.append(rendered)
        raise RuntimeError("Forbidden teacher/pose asset access")

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
def patch_prepare(
    tapf,
    arm,
    runtime,
    core,
    evidence_cache,
    evidence_seen,
    donor_map,
    orthogonal,
    capture=False,
):
    original = tapf.prepare
    had_override = "prepare" in tapf.__dict__
    override_before = tapf.__dict__.get("prepare")
    report = {
        "calls": 0,
        "rows": 0,
        "captured_rows": 0,
        "duplicate_capture_rows": 0,
        "restored_exact": False,
    }

    def patched(*args, **kwargs):
        state = original(*args, **kwargs)
        indices = runtime.get("indices")
        if indices is None:
            raise RuntimeError("TAPF prepare called without absolute indices")
        indices = torch.as_tensor(indices, dtype=torch.int64).flatten().cpu()
        if indices.numel() != state["consumer_evidence"].shape[0]:
            raise RuntimeError("TAPF prepare/index batch mismatch")
        report["calls"] += 1
        report["rows"] += int(indices.numel())
        if capture:
            evidence = state["consumer_evidence"].detach().cpu()
            if not bool(torch.isfinite(evidence).all()):
                raise RuntimeError("Non-finite cached student evidence")
            for position, index in enumerate(indices.tolist()):
                if int(evidence_seen[index]) > 0:
                    if not torch.equal(evidence_cache[index], evidence[position]):
                        raise RuntimeError("Repeated evidence capture changed")
                    report["duplicate_capture_rows"] += 1
                else:
                    evidence_cache[index].copy_(evidence[position])
                    report["captured_rows"] += 1
                evidence_seen[index] += 1
        if arm in core.STATE_INTERVENTIONS:
            if arm == "wrong_rgb_evidence":
                selected_donors = donor_map.index_select(0, indices)
                if not bool((evidence_seen[selected_donors] > 0).all()):
                    raise RuntimeError("Wrong-RGB donor evidence is not cached")
            state = core.apply_state_intervention(
                state,
                arm,
                absolute_indices=indices,
                donor_map=donor_map,
                evidence_cache=evidence_cache,
                orthogonal=orthogonal,
            )
        return state

    tapf.prepare = patched
    try:
        yield report
    finally:
        if had_override:
            tapf.__dict__["prepare"] = override_before
        else:
            tapf.__dict__.pop("prepare", None)
        report["restored_exact"] = (
            ("prepare" in tapf.__dict__) == had_override
            and (
                not had_override
                or tapf.__dict__.get("prepare") is override_before
            )
        )


@contextmanager
def count_router_calls(tapf):
    original = tapf.apply_gate
    had_override = "apply_gate" in tapf.__dict__
    override_before = tapf.__dict__.get("apply_gate")
    report = {"calls": [0, 0], "restored_exact": False}

    def patched(bank_index, tokens, hw_shape, state):
        if bank_index not in (0, 1):
            raise RuntimeError(f"Unexpected router bank: {bank_index}")
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
            and (
                not had_override
                or tapf.__dict__.get("apply_gate") is override_before
            )
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
            expected_paths = tuple(
                canonical_records[int(index)][0]
                for index in indices.tolist()
            )
            if tuple(paths) != expected_paths:
                raise RuntimeError("Validation absolute-index/path mismatch")
            expected_pid = tuple(
                int(canonical_records[int(index)][1])
                for index in indices.tolist()
            )
            expected_camid = tuple(
                int(canonical_records[int(index)][2])
                for index in indices.tolist()
            )
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
        "descriptor_exact": torch.equal(
            descriptor_none, descriptor_exploding
        ),
        "descriptor_finite": bool(torch.isfinite(descriptor_none).all()),
        "pose_accesses": int(ExplodingPose.accesses),
        "rows": int(indices.numel()),
    }


def build_loader(base, indices, batch_size, workers, generator):
    return DataLoader(
        IndexedDataset(base, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=indexed_collate,
        generator=generator,
    )


def run_arm(
    arm,
    model,
    tapf,
    loader,
    num_query,
    cfg,
    device,
    runtime,
    canonical_records,
    core,
    evidence_cache,
    evidence_seen,
    donor_map,
    orthogonal,
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

    generic_report = None
    router_report = None
    during_state_sha = state_before
    with patch_prepare(
        tapf,
        arm,
        runtime,
        core,
        evidence_cache,
        evidence_seen,
        donor_map,
        orthogonal,
        capture=(arm == "correct"),
    ) as prepare_report:
        if arm == "generic_expert_mean":
            with core.generic_expert_mean(tapf) as generic_report:
                during_state_sha = tensor_state_sha256(model.state_dict())
                with count_router_calls(tapf) as router_report:
                    output = collect_descriptors(
                        model,
                        loader,
                        num_query,
                        cfg,
                        device,
                        runtime,
                        canonical_records,
                        compute_metrics,
                    )
        elif arm in core.BYPASS_BANKS:
            with core.bypass_routers(
                tapf, core.BYPASS_BANKS[arm]
            ) as router_report:
                output = collect_descriptors(
                    model,
                    loader,
                    num_query,
                    cfg,
                    device,
                    runtime,
                    canonical_records,
                    compute_metrics,
                )
        else:
            with count_router_calls(tapf) as router_report:
                output = collect_descriptors(
                    model,
                    loader,
                    num_query,
                    cfg,
                    device,
                    runtime,
                    canonical_records,
                    compute_metrics,
                )

    loader_generator.set_state(loader_generator_state)
    rng_after = rng_snapshot()
    state_after = tensor_state_sha256(model.state_dict())
    expected_calls = [len(loader), len(loader)]
    report = {
        "prepare": dict(prepare_report),
        "router": {
            "calls": list(router_report["calls"]),
            "restored_exact": bool(router_report["restored_exact"]),
        },
        "generic": None if generic_report is None else dict(generic_report),
        "state_sha256_before": state_before,
        "state_sha256_during": during_state_sha,
        "state_sha256_after": state_after,
        "state_restored_exact": state_after == baseline_state_sha,
        "prepare_restored_exact": bool(prepare_report["restored_exact"]),
        "router_restored_exact": bool(router_report["restored_exact"]),
        "router_calls_exact": list(router_report["calls"]) == expected_calls,
        "rng_sha256_before": rng_sha256(rng_before),
        "rng_sha256_after": rng_sha256(rng_after),
        "rng_restored_exact": rng_equal(rng_before, rng_after),
        "loader_rng_restored_exact": torch.equal(
            loader_generator.get_state(), loader_generator_state
        ),
        "rows": len(output["indices"]),
        "unique_indices": len(set(output["indices"])),
        "path_sha256": output["path_sha256"],
    }
    if generic_report is not None:
        report["generic_state_changed_during"] = (
            during_state_sha != baseline_state_sha
        )
        report["generic_restored_exact"] = bool(
            generic_report["restored_exact"]
        )
    if arm in core.BYPASS_BANKS:
        report["bypassed"] = list(router_report["bypassed"])
    return output, report


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
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_state_sha = tensor_state_sha256(payload)
    state_names = tuple(payload)
    forbidden_state_tokens = {
        "teacher", "clip", "codebook", "text", "pose_batch"
    }
    cpu_gates = {
        "repo_head_exact": run_text(
            ["git", "rev-parse", "HEAD"], repo_root
        ) == SOURCE_COMMIT,
        "repo_tracked_clean": not bool(
            run_text(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                repo_root,
            )
        ),
        "repo_all_clean": not bool(
            run_text(["git", "status", "--porcelain"], repo_root)
        ),
        "source_sha_exact": source_sha_before == EXPECTED_SOURCE_SHA256,
        "config_sha_exact": config_sha_before == CONFIG_SHA256,
        "core_sha_exact": core_sha_before == CORE_SHA256,
        "checkpoint_path_exact": checkpoint_path == expected_checkpoint,
        "checkpoint_regular": checkpoint_path.is_file()
        and not checkpoint_path.is_symlink(),
        "checkpoint_sha_exact": checkpoint_sha_before == CHECKPOINT_SHA256,
        "checkpoint_unique": checkpoint_files_before == [str(checkpoint_path)],
        "checkpoint_state_exact": checkpoint_state_sha
        == CHECKPOINT_STATE_SHA256,
        "checkpoint_state_all_tensors": all(
            torch.is_tensor(value) for value in payload.values()
        ),
        "checkpoint_state_finite": all(
            tensor_finite(value) for value in payload.values()
        ),
        "checkpoint_state_count": len(payload) == 241,
        "state_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden_state_tokens)
            for name in state_names
        ),
        "evidence_head_retained": any(
            "anchor.evidence_head" in name for name in state_names
        ),
        "two_routers_retained": all(
            any(
                f"psg_bank.{index}.evidence_projection" in name
                for name in state_names
            )
            for index in (0, 1)
        ),
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
    num_query = len(dataset.query)
    pids = torch.tensor([int(record[1]) for record in records])
    camids = torch.tensor([int(record[2]) for record in records])
    core = load_module(core_path)
    donor_map = core.build_global_donor_map(pids, camids, num_query)
    donor_summary = core.validate_donor_map(
        donor_map, pids, camids, num_query
    )
    orthogonal = core.canonical_orthogonal(16, 1234, dtype=torch.float32)
    val_transform = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD,
            ),
        ]
    )
    base_dataset = ImageDataset(records, val_transform)
    all_indices = list(range(len(records)))
    preflight = int(args.preflight_samples) > 0
    selected_indices = (
        all_indices[: min(int(args.preflight_samples), len(all_indices))]
        if preflight
        else all_indices
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(1234)
    loader_generator_state = loader_generator.get_state().clone()
    loader = build_loader(
        base_dataset,
        selected_indices,
        min(int(cfg.TEST.IMS_PER_BATCH), len(selected_indices)),
        0 if preflight else int(cfg.DATALOADER.NUM_WORKERS),
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
        baseline_state_sha = tensor_state_sha256(model.state_dict())
        strict_gates = {
            "strict_load_no_missing": not incompatible.missing_keys,
            "strict_load_no_unexpected": not incompatible.unexpected_keys,
            "strict_state_exact": loaded_state_sha == checkpoint_state_sha,
            "device_state_exact": baseline_state_sha == checkpoint_state_sha,
            "tapf_rich_exact": bool(tapf.rich_evidence and tapf.semantic),
            "rho_python_float": type(tapf.rho_star) is float,
            "rho_not_parameter": all(
                "rho" not in name.lower() for name, _ in model.named_parameters()
            ),
            "rho_not_buffer": all(
                "rho" not in name.lower() for name, _ in model.named_buffers()
            ),
            "two_router_modules": len(tapf.psg_bank) == 2,
            "evidence_head_module": hasattr(tapf.anchor, "evidence_head"),
            "config_output_dir_exact": Path(cfg.OUTPUT_DIR).resolve()
            == output_dir,
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

        evidence_cache = torch.zeros(
            len(records), 5, 16, dtype=torch.float32
        )
        evidence_seen = torch.zeros(len(records), dtype=torch.int64)
        runtime = {"indices": None}
        donor_warmup = None
        if preflight:
            donor_indices = sorted(
                set(int(donor_map[index]) for index in selected_indices)
            )
            donor_loader = build_loader(
                base_dataset,
                donor_indices,
                min(int(cfg.TEST.IMS_PER_BATCH), len(donor_indices)),
                0,
                loader_generator,
            )
            loader_generator.set_state(loader_generator_state)
            with patch_prepare(
                tapf,
                "correct",
                runtime,
                core,
                evidence_cache,
                evidence_seen,
                donor_map,
                orthogonal,
                capture=True,
            ) as donor_prepare:
                donor_output = collect_descriptors(
                    model,
                    donor_loader,
                    num_query,
                    cfg,
                    device,
                    runtime,
                    records,
                    False,
                )
            loader_generator.set_state(loader_generator_state)
            donor_warmup = {
                "indices": donor_indices,
                "rows": len(donor_output["indices"]),
                "path_sha256": donor_output["path_sha256"],
                "prepare": dict(donor_prepare),
            }

        metrics = {}
        descriptors = {}
        arm_reports = {}
        expected_path_sha = canonical_path_sha256(
            selected_indices,
            [records[index][0] for index in selected_indices],
        )
        for arm in core.ARM_ORDER:
            output, arm_report = run_arm(
                arm,
                model,
                tapf,
                loader,
                num_query,
                cfg,
                device,
                runtime,
                records,
                core,
                evidence_cache,
                evidence_seen,
                donor_map,
                orthogonal,
                loader_generator,
                loader_generator_state,
                baseline_state_sha,
                not preflight,
            )
            descriptors[arm] = output["descriptors"]
            arm_reports[arm] = arm_report
            if output["metrics"] is not None:
                metrics[arm] = output["metrics"]

        descriptor_deltas = {
            arm: core.descriptor_delta(
                descriptors["correct"], descriptors[arm]
            )
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
    repo_status_after = run_text(
        ["git", "status", "--porcelain"], repo_root
    )
    model_state_after = tensor_state_sha256(model.state_dict())
    all_arm_restore = all(
        report["state_restored_exact"]
        and report["prepare_restored_exact"]
        and report["router_restored_exact"]
        and report["router_calls_exact"]
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
    correct_capture = arm_reports["correct"]["prepare"]
    if preflight:
        correct_evidence_coverage = all(
            int(evidence_seen[int(donor_map[index])]) > 0
            for index in selected_indices
        )
    else:
        correct_evidence_coverage = (
            correct_capture["captured_rows"] == len(records)
            and bool((evidence_seen == 1).all())
        )
    donor_warmup_valid = (
        not preflight
        or (
            donor_warmup is not None
            and donor_warmup["rows"] == len(donor_warmup["indices"])
            and donor_warmup["prepare"]["restored_exact"]
        )
    )
    common_validity = {
        "cpu_gates_all_pass": all(cpu_gates.values()),
        "strict_gates_all_pass": all(strict_gates.values()),
        "rgb_only_exact": rgb_only["descriptor_exact"]
        and rgb_only["pose_accesses"] == 0,
        "forbidden_teacher_pose_reads_zero": not read_accesses,
        "read_guard_restored_exact": read_guard_restored,
        "donor_warmup_valid": donor_warmup_valid,
        "all_arm_restore_exact": all_arm_restore,
        "all_arm_full_coverage": all_arm_coverage,
        "all_descriptors_finite_active": all_descriptor_active,
        "correct_evidence_coverage": correct_evidence_coverage,
        "model_state_terminal_exact": model_state_after == baseline_state_sha,
        "checkpoint_before_after_exact": checkpoint_sha_before
        == checkpoint_sha_after,
        "config_before_after_exact": config_sha_before == config_sha_after,
        "core_before_after_exact": core_sha_before == core_sha_after,
        "source_before_after_exact": source_sha_before == source_sha_after,
        "checkpoint_list_before_after_exact": checkpoint_files_before
        == checkpoint_files_after,
        "repo_clean_after": not bool(repo_status_after),
    }
    if preflight:
        status = "PASS" if all(common_validity.values()) else "FAIL"
        decision = (
            "EXP402_CUDA_PREFLIGHT_PASS"
            if status == "PASS"
            else "EXP402_CUDA_PREFLIGHT_FAIL"
        )
        adjudication = None
        authorized = False
    else:
        reference_gates = {
            "correct_reference_exact": metrics_match(
                metrics["correct"], FULL_REFERENCE
            ),
            "all_bypass_reference_exact": metrics_match(
                metrics["all_router_bypass"], ALL_BYPASS_REFERENCE
            ),
            "all_metrics_finite": all(
                finite_metrics(value) for value in metrics.values()
            ),
        }
        validity = dict(common_validity)
        validity.update(reference_gates)
        adjudication = core.adjudicate(metrics, descriptor_deltas, validity)
        status = "PASS" if all(validity.values()) else "FAIL"
        decision = (
            adjudication["decision"]
            if status == "PASS"
            else "EXP402_SEALED_INVALID"
        )
        authorized = bool(
            status == "PASS"
            and adjudication[
                "phase_b_formal_mechanism_design_authorized"
            ]
        )
        common_validity = validity

    return {
        "status": status,
        "mode": "CUDA_PREFLIGHT" if preflight else "FORMAL_FULL",
        "decision": decision,
        "phase_b_formal_mechanism_design_authorized": authorized,
        "validity": common_validity,
        "cpu_gates": cpu_gates,
        "strict_gates": strict_gates,
        "rgb_only": rgb_only,
        "dataset": {
            "count": len(records),
            "num_query": num_query,
            "selected_count": len(selected_indices),
            "canonical_path_sha256": canonical_path_sha256(
                all_indices,
                [record[0] for record in records],
            ),
            "selected_path_sha256": expected_path_sha,
            "donor_summary": donor_summary,
            "donor_map_sha256": core.tensor_mapping_sha256(
                {"donor": donor_map}
            ),
            "donor_warmup": donor_warmup,
        },
        "metrics": metrics,
        "references": {
            "correct": FULL_REFERENCE,
            "all_router_bypass": ALL_BYPASS_REFERENCE,
            "absolute_tolerance": REFERENCE_TOLERANCE,
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
            "mode": (
                "CUDA_PREFLIGHT"
                if int(args.preflight_samples) > 0
                else "FORMAL_FULL"
            ),
            "decision": "AUDIT_RUNTIME_FAIL",
            "phase_b_formal_mechanism_design_authorized": False,
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
