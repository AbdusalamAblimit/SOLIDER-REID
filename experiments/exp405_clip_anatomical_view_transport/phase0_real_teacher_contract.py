#!/usr/bin/env python3
"""Small CPU contract for the exp405 real-teacher implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaskedBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())

    def forward(self, value: torch.Tensor, mask: torch.Tensor | None):
        normalized = self.norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        value = value + attended
        return value + self.mlp(value)


def isolation_contract(teacher) -> dict:
    torch.manual_seed(405)
    batch = 2
    heads = 4
    width = 16
    selected = torch.zeros(batch, 256, dtype=torch.bool)
    selected[0, [0, 1, 16, 17, 32]] = True
    selected[1, [96, 97, 112, 113, 128, 129]] = True
    mask = teacher.isolated_attention_mask(selected, heads)
    value = torch.randn(batch, 257, width)
    changed = value.clone()
    non_target = ~torch.cat((torch.ones(batch, 1, dtype=torch.bool), selected), dim=1)
    changed[non_target] = torch.randn_like(changed[non_target]) * 100.0
    blocks = nn.ModuleList([MaskedBlock(width, heads), MaskedBlock(width, heads)])
    left = value
    right = changed
    for block in blocks:
        left = block(left, mask)
        right = block(right, mask)
    active = ~non_target
    isolated_gap = float((left[active] - right[active]).abs().max().detach())
    unmasked_left = value
    unmasked_right = changed
    for block in blocks:
        unmasked_left = block(unmasked_left, None)
        unmasked_right = block(unmasked_right, None)
    unmasked_gap = float(
        (unmasked_left[active] - unmasked_right[active]).abs().max().detach()
    )
    return {
        "selected_output_max_abs_after_non_target_mutation": isolated_gap,
        "unmasked_mutant_max_abs": unmasked_gap,
        "isolated": isolated_gap <= 1e-6,
        "unmasked_mutant_caught": unmasked_gap > 1e-4,
        "mask_shape": list(mask.shape),
    }


def geometry_contract(teacher) -> dict:
    keypoints = torch.tensor(
        [
            [64, 40], [58, 36], [70, 36], [52, 40], [76, 40],
            [44, 100], [84, 100], [30, 160], [98, 160], [20, 220], [108, 220],
            [50, 220], [78, 220], [50, 300], [78, 300], [50, 370], [78, 370],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones(17)
    valid = torch.ones(17, dtype=torch.bool)
    identity = {"flipped": False, "crop_top": 0, "crop_left": 0}
    transformed = teacher.transform_pose(
        keypoints, scores, valid, (teacher.WIDTH, teacher.HEIGHT), identity
    )
    masks, confidence, region_valid = teacher.render_anatomical_regions(
        transformed[0].unsqueeze(0),
        transformed[1].unsqueeze(0),
        transformed[2].unsqueeze(0),
    )
    empty_masks, _, empty_valid = teacher.render_anatomical_regions(
        transformed[0].unsqueeze(0),
        torch.zeros_like(transformed[1]).unsqueeze(0),
        torch.zeros_like(transformed[2]).unsqueeze(0),
    )
    zero_score_masks, zero_score_confidence, zero_score_valid = (
        teacher.render_anatomical_regions(
            transformed[0].unsqueeze(0),
            torch.zeros_like(transformed[1]).unsqueeze(0),
            transformed[2].unsqueeze(0),
        )
    )
    repeated = teacher.deterministic_geometry("bounding_box_train/a.jpg", 7)
    repeated_again = teacher.deterministic_geometry("bounding_box_train/a.jpg", 7)
    changed = teacher.deterministic_geometry("bounding_box_train/b.jpg", 7)
    return {
        "shape_exact": masks.shape == (1, 5, 96, 32),
        "all_regions_valid": bool(region_valid.all()),
        "all_confidence_positive": bool((confidence > 0).all()),
        "hard_owner_sum_at_most_one": bool((masks.sum(1) <= 1.000001).all()),
        "all_regions_have_mass": bool((masks.flatten(2).sum(-1) > 0).all()),
        "empty_exact_zero": bool(torch.equal(empty_masks, torch.zeros_like(empty_masks))),
        "empty_invalid": bool(~empty_valid.any()),
        "zero_score_keeps_geometry": bool(
            zero_score_valid.all() and torch.equal(zero_score_masks, masks)
        ),
        "zero_score_confidence_is_zero": bool(
            torch.equal(zero_score_confidence, torch.zeros_like(zero_score_confidence))
        ),
        "deterministic_geometry_repeat": repeated == repeated_again,
        "path_owns_geometry": repeated != changed,
    }


def selected_input_contract(teacher) -> dict:
    fake = SimpleNamespace(device=torch.device("cpu"))
    rgb = torch.zeros(2, 3, teacher.HEIGHT, teacher.WIDTH)
    masks = torch.zeros(2, teacher.REGIONS, teacher.MASK_HEIGHT, teacher.MASK_WIDTH)
    slots = torch.tensor([0, 4], dtype=torch.long)
    teacher.RegionIsolatedClipTeacher._validate_inputs(fake, rgb, masks, slots)
    nonfinite_caught = False
    changed = rgb.clone()
    changed[0, 0, 0, 0] = float("nan")
    try:
        teacher.RegionIsolatedClipTeacher._validate_inputs(fake, changed, masks, slots)
    except ValueError:
        nonfinite_caught = True
    range_caught = False
    changed = masks.clone()
    changed[0, 0, 0, 0] = 2.0
    try:
        teacher.RegionIsolatedClipTeacher._validate_inputs(fake, rgb, changed, slots)
    except ValueError:
        range_caught = True
    return {
        "selected_valid_input_passes": True,
        "selected_nonfinite_caught": nonfinite_caught,
        "selected_range_caught": range_caught,
    }


def measurement_statistics_contract(measurement, core) -> dict:
    pose_valid = torch.tensor([[True, False], [False, False]])
    readout_valid = torch.tensor([[False, True], [False, False]])
    analysis_valid = pose_valid & readout_valid
    targets = measurement.choose_targets(
        analysis_valid, torch.tensor([7, 11], dtype=torch.long)
    )
    effect = measurement.pid_effect_summary(
        torch.tensor([1.0, 3.0, 2.0, 4.0]),
        torch.tensor([1, 2, 1, 2]),
    )
    macro = measurement.non_torso_macro_effect(
        torch.tensor([1.0, 3.0, 2.0, 4.0, 3.0, 5.0]),
        torch.tensor([0, 0, 3, 3, 4, 4]),
        torch.tensor([1, 2, 1, 2, 1, 2]),
    )
    incomplete_macro = measurement.non_torso_macro_effect(
        torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0]),
        torch.tensor([0, 0, 3, 3, 3, 4, 4]),
        torch.tensor([1, 2, 1, 2, 3, 2, 3]),
    )
    correlated_macro = measurement.non_torso_macro_effect(
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        torch.tensor([0, 0, 3, 3, 4, 4]),
        torch.tensor([1, 2, 1, 2, 1, 2]),
    )
    torso_augmented_macro = measurement.non_torso_macro_effect(
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 999.0, -999.0]),
        torch.tensor([0, 0, 3, 3, 4, 4, 1, 2]),
        torch.tensor([1, 2, 1, 2, 1, 2, 99, 100]),
    )
    non_torso_two_way = measurement.non_torso_two_cluster_summary(
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        torch.tensor([0, 0, 3, 3, 4, 4]),
        torch.tensor([1, 2, 1, 2, 1, 2]),
        torch.tensor([11, 12, 11, 12, 11, 12]),
    )
    torso_augmented_two_way = measurement.non_torso_two_cluster_summary(
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 999.0, -999.0]),
        torch.tensor([0, 0, 3, 3, 4, 4, 1, 2]),
        torch.tensor([1, 2, 1, 2, 1, 2, 99, 100]),
        torch.tensor([11, 12, 11, 12, 11, 12, 101, 102]),
    )
    two_way_left = measurement.conservative_two_cluster_summary(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([1, 1, 2, 2]),
        torch.tensor([10, 11, 10, 11]),
    )
    two_way_right = measurement.conservative_two_cluster_summary(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([1, 1, 2, 2]),
        torch.tensor([10, 11, 10, 11]),
    )

    valid = torch.ones(4, 1, dtype=torch.bool)
    mass = torch.tensor([[10.0], [11.0], [10.0], [11.0]])
    centroid = torch.tensor([[0.2], [0.3], [0.2], [0.3]])
    confidence = torch.tensor([[0.5], [0.6], [0.5], [0.6]])
    support = torch.tensor([[0.4], [0.5], [0.4], [0.5]])
    global_feature = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        dim=-1,
    )
    pids = torch.tensor([0, 1, 2, 3])
    camids = torch.zeros(4, dtype=torch.long)
    keys = torch.tensor([4, 3, 2, 1], dtype=torch.long)
    donors, records = measurement.choose_wrong_masks(
        [0, 1], [0, 0], valid, mass, centroid, confidence, support,
        global_feature, pids, camids, keys, [0, 1],
    )
    augmenting_assignment = measurement.maximum_unique_assignment([[1, 2], [1]])
    expansion_preferences = [np.arange(65, dtype=np.int32)] + [
        np.arange(64, dtype=np.int32) for _ in range(64)
    ]
    expanded_assignment, expansion_limit = (
        measurement.maximum_unique_assignment_with_expansion(expansion_preferences)
    )
    preflight_adjudication = measurement.adjudicate_measurement(
        formal=False, validity_pass=True, scientific_pass=False
    )
    formal_quadrants = {
        (validity, scientific): measurement.adjudicate_measurement(
            formal=True, validity_pass=validity, scientific_pass=scientific
        )
        for validity in (False, True)
        for scientific in (False, True)
    }

    synthetic_records = [
        ("/unused/%04d.jpg" % index, index // 2, index % 4, -1)
        for index in range(620)
    ]
    synthetic_paths = ["bounding_box_train/%04d.jpg" % index for index in range(620)]
    selected = measurement.choose_preflight_indices(synthetic_records, synthetic_paths)
    reversed_records = list(reversed(synthetic_records))
    reversed_paths = list(reversed(synthetic_paths))
    selected_reversed = measurement.choose_preflight_indices(
        reversed_records, reversed_paths
    )
    selected_paths = [synthetic_paths[index] for index in selected]
    selected_reversed_paths = [reversed_paths[index] for index in selected_reversed]

    original_preflight_root = measurement.PREFLIGHT_OUTPUT_ROOT
    contract_root = Path(measurement.__file__).resolve().parent
    try:
        measurement.PREFLIGHT_OUTPUT_ROOT = contract_root
        preseal_args = SimpleNamespace(
            mode="preflight",
            output_dir=str(contract_root),
            _execution_seal_acquired=False,
        )
        postseal_args = SimpleNamespace(
            mode="preflight",
            output_dir=str(contract_root),
            _execution_seal_acquired=True,
        )
        preseal_failure_path = measurement.failure_receipt_path(preseal_args)
        postseal_failure_path = measurement.failure_receipt_path(postseal_args)
    finally:
        measurement.PREFLIGHT_OUTPUT_ROOT = original_preflight_root

    original_open = measurement.os.open
    original_write = measurement.os.write
    original_close = measurement.os.close
    seal_owner = SimpleNamespace(_execution_seal_acquired=False)
    seal_write_failure_caught = False
    try:
        measurement.os.open = lambda *args, **kwargs: 405
        measurement.os.write = lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected seal write failure")
        )
        measurement.os.close = lambda *args, **kwargs: None
        try:
            measurement.acquire_execution_seal(
                contract_root, "contract-only", seal_owner
            )
        except OSError:
            seal_write_failure_caught = True
    finally:
        measurement.os.open = original_open
        measurement.os.write = original_write
        measurement.os.close = original_close

    feature = torch.zeros(2, 1, 1, 5)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    ).unsqueeze(1)
    exact = True
    mutant_caught = True
    for fraction in measurement.DELETION_FRACTIONS:
        _, erased, realized = core.deterministic_slot_delete(
            feature,
            mask,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            fraction=fraction,
            ordering_seed=measurement.VIEW_SEED,
        )
        count = mask.flatten(1).sum(1)
        expected = torch.floor(count.double() * fraction + 0.5).long()
        actual = erased.flatten(1).sum(1)
        exact &= bool(torch.equal(actual, expected))
        exact &= bool(torch.equal(realized, actual.double() / count.double()))
        mutant_caught &= bool(not torch.equal(actual + 1, expected))
    return {
        "validity_axes_separate": bool(
            torch.equal(analysis_valid, torch.zeros_like(analysis_valid))
        ),
        "all_invalid_target_is_minus_one": bool(torch.equal(targets, torch.tensor([-1, -1]))),
        "pid_sign_counts_exact": effect["pid_sign"] == {
            "positive": 2, "zero": 0, "negative": 0
        },
        "non_torso_macro_equal_weight": abs(
            macro["pid_bootstrap"]["mean"] - 3.0
        ) < 1e-12,
        "incomplete_non_torso_still_equal_weight": abs(
            incomplete_macro["pid_bootstrap"]["mean"] - (172.0 / 3.0)
        ) < 1e-12,
        "non_torso_bootstrap_clusters_global_pid": bool(
            correlated_macro["global_pid_cluster"]
            and abs(correlated_macro["pid_bootstrap"]["lower_95"] + 1.0) < 1e-12
            and abs(correlated_macro["pid_bootstrap"]["upper_95"] - 1.0) < 1e-12
        ),
        "torso_only_pid_is_inert_for_non_torso_ci": bool(
            correlated_macro == torso_augmented_macro
            and non_torso_two_way == torso_augmented_two_way
        ),
        "two_way_bootstrap_deterministic": (
            two_way_left["two_way_pid_bootstrap"]
            == two_way_right["two_way_pid_bootstrap"]
        ),
        "wrong_mask_strict_and_unique": bool(
            len(set(donors)) == 2
            and set(donors).isdisjoint({0, 1})
            and all(record["recipient_pid"] != record["donor_pid"] for record in records)
            and all(record["camera"] == 0 for record in records)
            and all(
                record["primary_distance"] <= measurement.MATCH_PRIMARY_CALIPER
                for record in records
            )
        ),
        "augmenting_match_repairs_greedy_conflict": augmenting_assignment == [2, 1],
        "matcher_expands_beyond_top64": bool(
            expansion_limit == 65
            and len(expanded_assignment) == 65
            and len(set(expanded_assignment)) == 65
        ),
        "preflight_selection_permutation_invariant": (
            selected_paths == selected_reversed_paths
        ),
        "preseal_failure_is_read_only": preseal_failure_path is None,
        "postseal_failure_is_scoped": (
            postseal_failure_path == contract_root / "failure.json"
        ),
        "seal_owned_immediately_after_exclusive_create": bool(
            seal_write_failure_caught and seal_owner._execution_seal_acquired
        ),
        "preflight_ignores_scientific_fail": (
            preflight_adjudication["status"] == "PREFLIGHT_PASS"
            and preflight_adjudication["decision"] == "PREFLIGHT_ONLY_PASS"
            and not preflight_adjudication["transport_oracle_authorization_candidate"]
        ),
        "formal_adjudication_four_quadrants": (
            formal_quadrants[(False, False)]["status"] == "VALIDITY_FAIL"
            and formal_quadrants[(False, True)]["status"] == "VALIDITY_FAIL"
            and formal_quadrants[(True, False)]["status"] == "VALIDITY_PASS"
            and formal_quadrants[(True, True)]["status"] == "VALIDITY_PASS"
            and all(
                formal_quadrants[key]["decision"]
                == (
                    "P0B_REGION_ISOLATED_TEACHER_GO"
                    if key == (True, True)
                    else "P0B_REGION_ISOLATED_TEACHER_NO_GO"
                )
                for key in formal_quadrants
            )
            and all(
                formal_quadrants[key]["transport_oracle_authorization_candidate"]
                is (key == (True, True))
                for key in formal_quadrants
            )
        ),
        "deletion_counts_exact": exact,
        "deletion_plus_one_mutant_caught": mutant_caught,
    }


def semantic_contract(core) -> dict:
    torch.manual_seed(406)
    slots = 5
    text = torch.eye(slots)
    visual = text.unsqueeze(0).clone()
    visible = text.clone()
    occluded = -text.clone()
    valid = torch.ones(1, slots, dtype=torch.bool)
    state = core.clip_slot_state(
        visual,
        text,
        visible,
        occluded,
        valid,
        logit_scale=10.0,
    )
    invalid = valid.clone()
    invalid[:, 3] = False
    invalid_state = core.clip_slot_state(
        visual,
        text,
        visible,
        occluded,
        invalid,
        logit_scale=10.0,
    )
    return {
        "correct_text_top1": bool(
            torch.equal(state["distribution"].argmax(-1), torch.arange(slots).view(1, slots))
        ),
        "support_high": bool((state["support"] > 0.99).all()),
        "geometry_not_reused_as_support": bool(
            invalid_state["support"][0, 3] == 0
            and torch.equal(
                invalid_state["distribution"], state["distribution"]
            )
        ),
    }


def source_contract(measurement_path: Path, teacher_path: Path) -> dict:
    measurement = measurement_path.read_text(encoding="utf-8")
    teacher = teacher_path.read_text(encoding="utf-8")
    measurement_ast = ast.parse(measurement)
    forbidden_calls = []
    for node in ast.walk(measurement_ast):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node.func)
            if rendered.endswith(("backward", "step", "zero_grad")):
                forbidden_calls.append(rendered)
    return {
        "no_optimizer_or_training_calls": not forbidden_calls,
        "no_resume_argument": "--resume" not in measurement,
        "official_roots_literal": (
            'Path("/mnt1/afrdata")' in measurement
            and 'Path("/mnt1/afrderived")' in measurement
            and 'Path("/home/afr/reid-clean/audits")' in measurement
        ),
        "rgb_sha_verification_enabled": "verify_rgb=True" in measurement,
        "native_logit_scale_used": "model.logit_scale.detach().float().exp()" in teacher,
        "all_blocks_receive_isolation_mask": "block(active, attn_mask=attention)" in teacher,
        "token_pooling_is_pinned": 'pool_type", None) not in ("tok", "token")' in teacher,
        "formal_mode_is_explicit": 'choices=("preflight", "formal")' in measurement,
        "formal_manifest_required": "formal mode requires a frozen execution manifest" in measurement,
        "preflight_sample_count_frozen": "PREFLIGHT_SAMPLES = 512" in measurement,
        "complete_receipt_is_authoritative": (
            "PREFLIGHT_COMPLETE_PATH" in measurement
            and "authorization_requires_complete_receipt" in measurement
        ),
        "preseal_failure_handler_is_guarded": (
            "failure_path = failure_receipt_path(args)" in measurement
            and 'getattr(args, "_execution_seal_acquired", False)' in measurement
        ),
        "runtime_artifact_bytes_are_bound": (
            '"executable_sha256"' in measurement
            and '"module_origin_sha256"' in measurement
            and '"record_sha256"' in measurement
            and '"cuda_visible_devices"' in measurement
        ),
        "runtime_binding_fails_closed": (
            "required runtime package is not installed" in measurement
            and "runtime package lacks frozen origin/RECORD bytes" in measurement
        ),
        "complete_terminal_is_immutable": (
            'if (output_dir / "complete.json").is_file()' in measurement
        ),
        "failure_publication_preserves_primary_error": (
            "except BaseException as receipt_error" in measurement
            and "failure receipt publication also failed" in measurement
        ),
        "wrong_mask_metadata_is_selected_only": (
            "np.lexsort" in measurement
            and "pair_metadata" not in measurement
            and ".astype(np.int32, copy=False)" in measurement
        ),
        "result_embeds_four_source_hashes": all(
            token in measurement
            for token in (
                '"runner_sha256"',
                '"core_sha256"',
                '"teacher_sha256"',
                '"protocol_sha256"',
            )
        ),
        "target_coverage_thresholds_frozen": (
            "MAX_NO_TARGET_FRACTION = 0.01" in measurement
            and "MIN_TARGET_PID_FRACTION = 0.99" in measurement
        ),
        "frozen_taxonomy_exact": "upper_torso_arms" in teacher and "lower_torso" in teacher,
        "forbidden_calls": forbidden_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--measurement", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    teacher_path = Path(args.teacher).resolve()
    measurement_path = Path(args.measurement).resolve()
    core_path = Path(args.core).resolve()
    teacher = load_module("exp405_real_teacher", teacher_path)
    core = load_module("exp405_core", core_path)
    measurement = load_module("exp405_measurement", measurement_path)
    cuda_before = torch.cuda.is_initialized()
    isolation = isolation_contract(teacher)
    geometry = geometry_contract(teacher)
    selected_input = selected_input_contract(teacher)
    semantic = semantic_contract(core)
    measurement_statistics = measurement_statistics_contract(measurement, core)
    source = source_contract(measurement_path, teacher_path)
    cuda_after = torch.cuda.is_initialized()
    gates = {
        "attention_isolation": isolation["isolated"],
        "attention_negative_control": isolation["unmasked_mutant_caught"],
        "geometry": all(value for key, value in geometry.items() if isinstance(value, bool)),
        "selected_input": all(selected_input.values()),
        "semantic": all(semantic.values()),
        "measurement_statistics": all(measurement_statistics.values()),
        "source": all(
            value for key, value in source.items()
            if isinstance(value, bool)
        ),
        "cuda_not_initialized": not cuda_before and not cuda_after,
    }
    result = {
        "experiment": "exp405",
        "contract": "real_teacher_v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "attention": isolation,
        "geometry": geometry,
        "selected_input": selected_input,
        "semantic": semantic,
        "measurement_statistics": measurement_statistics,
        "source": source,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "provenance": {
            "contract_sha256": sha256_file(Path(__file__).resolve()),
            "measurement_sha256": sha256_file(measurement_path),
            "teacher_sha256": sha256_file(teacher_path),
            "core_sha256": sha256_file(core_path),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("output already exists")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(result["status"], sum(gates.values()), "/", len(gates))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
