#!/usr/bin/env python3
"""Static/CPU contract for exp395 AMP gradient attribution design."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import torch
from torch import nn


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
SCALE = 65536.0
EXPECTED_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml": "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "configs/occluded_duke/swin_tiny_tapf_d0.yml": "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
    "experiments/exp394_clip_owned_residual_budget/cuda_amp_preflight.py": "bae2210bc606048371b4750f85919595c0b8fdbd1e11681abac59fe9727ea4f0",
}
GROUP_NAMES = (
    "backbone",
    "anchor_trunk",
    "pose_head",
    "mask_head",
    "presence_head",
    "evidence_head",
    "id_head",
    "router0_token_projection",
    "router0_context_projection",
    "router0_evidence_projection",
    "router0_experts",
    "router1_token_projection",
    "router1_context_projection",
    "router1_evidence_projection",
    "router1_experts",
)
LOSS_NAMES = (
    "reid",
    "heatmap",
    "confidence",
    "mask",
    "presence",
    "evidence_cosine",
    "evidence_relation",
    "exec_consumer0",
    "exec_consumer1",
    "pose",
    "total",
)
ROUTER0 = (
    "router0_token_projection",
    "router0_context_projection",
    "router0_evidence_projection",
    "router0_experts",
)
ROUTER1 = (
    "router1_token_projection",
    "router1_context_projection",
    "router1_evidence_projection",
    "router1_experts",
)
EXPECTED_OWNERSHIP = {
    "reid": {"backbone", "id_head", *ROUTER0, *ROUTER1},
    "heatmap": {"anchor_trunk", "pose_head"},
    "confidence": {"anchor_trunk", "pose_head"},
    "mask": {"anchor_trunk", "mask_head"},
    "presence": {"anchor_trunk", "presence_head"},
    "evidence_cosine": {"evidence_head"},
    "evidence_relation": {"evidence_head"},
    "exec_consumer0": {"evidence_head", *ROUTER0},
    "exec_consumer1": {"evidence_head", *ROUTER1},
    "pose": {
        "anchor_trunk",
        "pose_head",
        "mask_head",
        "presence_head",
        "evidence_head",
        *ROUTER0,
        *ROUTER1,
    },
    "total": set(GROUP_NAMES),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def rng_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    digest.update(repr(random.getstate()).encode("utf-8"))
    return digest.hexdigest()


def finite_statistics(values: torch.Tensor):
    values = values.detach().cpu().to(torch.float64).flatten()
    finite = torch.isfinite(values)
    selected = values[finite].abs()
    if selected.numel() == 0:
        return {
            "finite_abs_max": None,
            "finite_l2": None,
            "finite_abs_p50": None,
            "finite_abs_p95": None,
            "finite_abs_p99": None,
        }
    quantiles = torch.quantile(
        selected, torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64)
    )
    return {
        "finite_abs_max": float(selected.max()),
        "finite_l2": float(torch.linalg.vector_norm(selected)),
        "finite_abs_p50": float(quantiles[0]),
        "finite_abs_p95": float(quantiles[1]),
        "finite_abs_p99": float(quantiles[2]),
    }


def gradient_report(groups):
    report = {}
    for group_name, parameters in groups.items():
        present = [parameter.grad for parameter in parameters if parameter.grad is not None]
        flattened = (
            torch.cat([gradient.detach().cpu().to(torch.float64).flatten() for gradient in present])
            if present
            else torch.empty(0, dtype=torch.float64)
        )
        finite = torch.isfinite(flattened)
        nan = torch.isnan(flattened)
        posinf = torch.isposinf(flattened)
        neginf = torch.isneginf(flattened)
        item = {
            "parameter_tensors": len(parameters),
            "grad_present_tensors": len(present),
            "grad_absent_tensors": len(parameters) - len(present),
            "grad_nonzero_tensors": sum(
                int(bool(torch.count_nonzero(gradient.detach())))
                for gradient in present
            ),
            "grad_zero_tensors": sum(
                int(not bool(torch.count_nonzero(gradient.detach())))
                for gradient in present
            ),
            "elements": int(flattened.numel()),
            "finite_elements": int(finite.sum()),
            "nan_elements": int(nan.sum()),
            "posinf_elements": int(posinf.sum()),
            "neginf_elements": int(neginf.sum()),
            "all_finite": bool(finite.all()) if flattened.numel() else True,
        }
        item.update(finite_statistics(flattened))
        report[group_name] = item
    return report


class SyntheticAttributionGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_by_group = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.linspace(0.125, 1.0, 8, dtype=torch.float64)
                    + index * 0.03125
                )
                for index, name in enumerate(GROUP_NAMES)
            }
        )

    def groups(self):
        return {
            name: [self.parameters_by_group[name]] for name in GROUP_NAMES
        }

    def group_loss(self, *names):
        return sum(
            self.parameters_by_group[name].square().mean() for name in names
        )

    def losses(self):
        reid = self.group_loss("backbone", "id_head", *ROUTER0, *ROUTER1)
        heatmap = 1.1 * self.group_loss("anchor_trunk", "pose_head")
        confidence = 1.2 * self.group_loss("anchor_trunk", "pose_head")
        mask = 1.3 * self.group_loss("anchor_trunk", "mask_head")
        presence = 1.4 * self.group_loss("anchor_trunk", "presence_head")
        evidence_cosine = 1.5 * self.group_loss("evidence_head")
        evidence_relation = 1.6 * self.group_loss("evidence_head")
        exec_consumer0 = 1.7 * self.group_loss("evidence_head", *ROUTER0)
        exec_consumer1 = 1.8 * self.group_loss("evidence_head", *ROUTER1)
        exec_mean = torch.stack([exec_consumer0, exec_consumer1]).mean()
        semantic = torch.stack(
            [
                mask,
                presence,
                evidence_cosine,
                evidence_relation,
                exec_mean,
            ]
        ).mean()
        pose = heatmap + confidence + semantic
        total = reid + 0.1 * pose
        return {
            "reid": reid,
            "heatmap": heatmap,
            "confidence": confidence,
            "mask": mask,
            "presence": presence,
            "evidence_cosine": evidence_cosine,
            "evidence_relation": evidence_relation,
            "exec_consumer0": exec_consumer0,
            "exec_consumer1": exec_consumer1,
            "pose": pose,
            "total": total,
            "_semantic": semantic,
            "_exec_mean": exec_mean,
        }


def active_groups(report):
    return {
        name
        for name, item in report.items()
        if item["grad_present_tensors"] > 0 and item["grad_nonzero_tensors"] > 0
    }


def scaled_ratio_exact(scaled, unscaled):
    fields = (
        "finite_abs_max",
        "finite_l2",
        "finite_abs_p50",
        "finite_abs_p95",
        "finite_abs_p99",
    )
    for name in GROUP_NAMES:
        left = scaled[name]
        right = unscaled[name]
        for count in (
            "grad_present_tensors",
            "grad_absent_tensors",
            "grad_nonzero_tensors",
            "grad_zero_tensors",
            "elements",
            "finite_elements",
            "nan_elements",
            "posinf_elements",
            "neginf_elements",
        ):
            if left[count] != right[count]:
                return False
        for field in fields:
            scaled_value = left[field]
            unscaled_value = right[field]
            if scaled_value is None or unscaled_value is None:
                if scaled_value is not None or unscaled_value is not None:
                    return False
                continue
            if not math.isclose(
                scaled_value,
                unscaled_value * SCALE,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
    return True


def sentinel_contract():
    absent = nn.Parameter(torch.ones(1, dtype=torch.float64))
    zero = nn.Parameter(torch.ones(2, dtype=torch.float64))
    finite = nn.Parameter(torch.ones(3, dtype=torch.float64))
    nonfinite = nn.Parameter(torch.ones(4, dtype=torch.float64))
    zero.grad = torch.zeros(2, dtype=torch.float64)
    finite.grad = torch.tensor([-2.0, 0.5, 4.0], dtype=torch.float64)
    nonfinite.grad = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 3.0],
        dtype=torch.float64,
    )
    item = gradient_report(
        {"sentinel": [absent, zero, finite, nonfinite]}
    )["sentinel"]
    expected = {
        "parameter_tensors": 4,
        "grad_present_tensors": 3,
        "grad_absent_tensors": 1,
        "grad_nonzero_tensors": 2,
        "grad_zero_tensors": 1,
        "elements": 9,
        "finite_elements": 6,
        "nan_elements": 1,
        "posinf_elements": 1,
        "neginf_elements": 1,
        "all_finite": False,
        "finite_abs_max": 4.0,
        "finite_l2": math.sqrt(29.25),
        "finite_abs_p50": 1.25,
        "finite_abs_p95": 3.75,
        "finite_abs_p99": 3.95,
    }
    exact = all(
        math.isclose(item[key], value, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(value, float)
        else item[key] == value
        for key, value in expected.items()
    )
    return item, expected, exact


def static_source_contract(repo_root: Path):
    sources = {
        relative: (repo_root / relative).read_text(encoding="utf-8")
        for relative in (
            "model/tapf.py",
            "experiments/exp394_clip_owned_residual_budget/cuda_amp_preflight.py",
        )
    }
    for source in sources.values():
        ast.parse(source)
    tapf = sources["model/tapf.py"]
    preflight = sources[
        "experiments/exp394_clip_owned_residual_budget/cuda_amp_preflight.py"
    ]
    checks = {
        "rich_exposes_exec_losses": '"exec_losses": []' in tapf,
        "two_exec_losses_appended": 'state["exec_losses"].append(exec_loss)' in tapf,
        "exec_mean_formula_present": 'torch.stack(state["exec_losses"]).mean()' in tapf,
        "semantic_uses_exec_mean": 'state["exec_loss"],' in tapf,
        "rich_total_uses_pose_weight": 'reid + 0.1 * aux["pose_loss"]' in preflight,
        "sealed_scaled_backward_present": "scaler.scale(total_loss).backward()" in preflight,
        "sealed_unscale_present": "scaler.unscale_(optimizer)" in preflight,
        "sealed_nonfinite_stops_before_step": (
            preflight.index('if not gradient_finite:')
            < preflight.index('scaler.step(optimizer)')
        ),
        "sealed_script_has_no_loss_attribution_before_failure": (
            preflight.index('if not gradient_finite:')
            < preflight.index('isolated = isolated_gradients(')
        ),
    }
    return checks


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU contract requires CUDA_VISIBLE_DEVICES='' or '-1'")
    cuda_initialized_before = torch.cuda.is_initialized()
    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SHA256
    }
    source_checks = static_source_contract(repo_root)
    git_head = run_text(["git", "rev-parse", "HEAD"], repo_root)
    source_commit_exists = run_text(
        ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
    ) == "commit"

    torch.manual_seed(395)
    random.seed(395)
    model = SyntheticAttributionGraph()
    groups = model.groups()
    state_before = state_sha256(model)
    rng_before = rng_sha256()
    loss_records = {}
    ownership_exact = {}
    ratio_exact = {}
    formula_reference = model.losses()
    formula_gates = {
        "exec_mean_exact": torch.equal(
            formula_reference["_exec_mean"],
            torch.stack(
                [
                    formula_reference["exec_consumer0"],
                    formula_reference["exec_consumer1"],
                ]
            ).mean(),
        ),
        "semantic_exact": torch.equal(
            formula_reference["_semantic"],
            torch.stack(
                [
                    formula_reference["mask"],
                    formula_reference["presence"],
                    formula_reference["evidence_cosine"],
                    formula_reference["evidence_relation"],
                    formula_reference["_exec_mean"],
                ]
            ).mean(),
        ),
        "pose_exact": torch.equal(
            formula_reference["pose"],
            formula_reference["heatmap"]
            + formula_reference["confidence"]
            + formula_reference["_semantic"],
        ),
        "total_exact": torch.equal(
            formula_reference["total"],
            formula_reference["reid"] + 0.1 * formula_reference["pose"],
        ),
    }

    for loss_name in LOSS_NAMES:
        model.zero_grad(set_to_none=True)
        selected = model.losses()[loss_name]
        (selected * SCALE).backward()
        scaled = gradient_report(groups)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(SCALE)
        unscaled = gradient_report(groups)
        ownership_exact[loss_name] = (
            active_groups(unscaled) == EXPECTED_OWNERSHIP[loss_name]
        )
        ratio_exact[loss_name] = scaled_ratio_exact(scaled, unscaled)
        loss_records[loss_name] = {
            "loss": float(selected.detach()),
            "expected_active_groups": sorted(EXPECTED_OWNERSHIP[loss_name]),
            "actual_active_groups": sorted(active_groups(unscaled)),
            "scaled": scaled,
            "unscaled": unscaled,
        }
        model.zero_grad(set_to_none=True)

    state_after = state_sha256(model)
    rng_after = rng_sha256()
    sentinel, sentinel_expected, sentinel_exact = sentinel_contract()
    cuda_initialized_after = torch.cuda.is_initialized()
    gates = {
        "source_sha_exact": source_sha == EXPECTED_SHA256,
        "source_commit_exists": source_commit_exists,
        "source_static_checks": all(source_checks.values()),
        "group_count_15": tuple(groups) == GROUP_NAMES,
        "loss_count_11": tuple(loss_records) == LOSS_NAMES,
        "ownership_exact": all(ownership_exact.values()),
        "scaled_unscaled_ratio_exact": all(ratio_exact.values()),
        "formula_exact": all(formula_gates.values()),
        "sentinel_classification_exact": sentinel_exact,
        "state_exact_zero_update": state_before == state_after,
        "rng_exact": rng_before == rng_after,
        "cuda_never_initialized": (
            not cuda_initialized_before and not cuda_initialized_after
        ),
        "cpu_device_only": all(
            parameter.device.type == "cpu" for parameter in model.parameters()
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": git_head,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": source_sha,
        "scale": SCALE,
        "group_names": list(GROUP_NAMES),
        "loss_names": list(LOSS_NAMES),
        "source_checks": source_checks,
        "formula_gates": formula_gates,
        "ownership_exact": ownership_exact,
        "scaled_unscaled_ratio_exact": ratio_exact,
        "sentinel": sentinel,
        "sentinel_expected": sentinel_expected,
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "rng_sha256_before": rng_before,
        "rng_sha256_after": rng_after,
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "optimizer_updates": 0,
        "checkpoint_count": 0,
        "cpu_contract_does_not_claim_amp_root_cause": True,
        "loss_records": loss_records,
        "gates": gates,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite existing result/runner")
    result = run_contract(repo_root)
    result["script_sha256"] = sha256_file(Path(__file__).resolve())
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
