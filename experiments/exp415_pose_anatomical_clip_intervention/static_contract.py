"""CPU-only synthetic and failure-injection contract for PACIT revision 3."""

from __future__ import annotations

import copy
import inspect
import random

import numpy as np
import torch

import asset_oracle_core as core
import clip_color_selector as clip_selector


def _expect_raises(callable_value):
    try:
        callable_value()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("expected contract failure")


def _fake_pose_fields(keypoints, valid):
    del keypoints, valid
    fields = torch.zeros(
        core.ANCHOR_COUNT, core.IMAGE_HEIGHT, core.IMAGE_WIDTH
    )
    bands = ((5, 75), (75, 165), (155, 235), (225, 315), (305, 384))
    for index, (top, bottom) in enumerate(bands):
        fields[index, top:bottom, 20:108] = 1.0
    return fields, torch.ones(core.ANCHOR_COUNT, dtype=torch.bool)


def _assert_sample_selection():
    rows = [{"relative_path": f"{index:05d}.jpg"} for index in range(700)]
    expected = core.select_oracle_rows(rows)
    shuffled = copy.deepcopy(rows)
    random.Random(1234).shuffle(shuffled)
    observed = core.select_oracle_rows(shuffled)
    assert [row["relative_path"] for row in observed] == [
        row["relative_path"] for row in expected
    ]
    assert len(observed) == core.ORACLE_COUNT
    assert len({row["relative_path"] for row in observed}) == core.ORACLE_COUNT


def _assert_proposals():
    assert tuple(inspect.signature(core.generate_fixed_proposals).parameters) == ()
    fixed = core.generate_fixed_proposals()
    original_renderer = core.render_full_pose_fields
    core.render_full_pose_fields = _fake_pose_fields
    try:
        pose, fields, valid = core.generate_pose_proposals(
            torch.zeros(17, 2), torch.ones(17, dtype=torch.bool)
        )
    finally:
        core.render_full_pose_fields = original_renderer
    assert len(fixed) == len(pose) == core.PROPOSALS_PER_POOL
    assert [row["candidate_index"] for row in fixed] == list(
        range(core.PROPOSALS_PER_POOL)
    )
    for oracle_index in range(core.ANCHOR_COUNT):
        fixed_active = core.active_proposals(fixed, oracle_index)
        pose_active = core.active_proposals(pose, oracle_index)
        assert len(fixed_active) == len(pose_active) == 7
        assert [row["aspect_index"] for row in fixed_active] == list(range(7))
        assert [row["aspect_index"] for row in pose_active] == list(range(7))
        for fixed_row, pose_row in zip(fixed_active, pose_active):
            assert fixed_row["area_pixels"] == pose_row["area_pixels"]
            assert fixed_row["aspect"] == pose_row["aspect"]
    return fixed, pose, fields, valid


def _assert_edit_and_colors(fixed):
    active = core.active_proposals(fixed, 1)
    mask = active[3]["mask"]
    formal_target = round(
        core.AREA_FRACTION * core.IMAGE_HEIGHT * core.IMAGE_WIDTH
    )
    assert abs(int(mask.sum()) - formal_target) / formal_target <= 0.01
    rgb = torch.linspace(
        0.0,
        1.0,
        3 * core.IMAGE_HEIGHT * core.IMAGE_WIDTH,
    ).reshape(3, core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
    fill = core.deterministic_fill("synthetic/0001.jpg")
    assert torch.equal(fill[0], fill[1]) and torch.equal(fill[1], fill[2])
    assert int((core.classify_color_bins(fill) >= 0).sum()) == 0
    edited = core.apply_candidate(rgb, mask, fill)
    assert torch.equal(edited[:, ~mask], rgb[:, ~mask])
    assert torch.equal(edited[:, mask], fill[:, mask])
    prototypes = torch.tensor(
        core.COLOR_PROTOTYPE_RGB, dtype=torch.float32
    ).transpose(0, 1).reshape(3, len(core.COLOR_NAMES), 1)
    labels = core.classify_color_bins(prototypes)
    assert labels[:, 0].tolist() == list(range(len(core.COLOR_NAMES)))


def _assert_scorer_and_caliper(fixed):
    assert (
        core.caliper_blind_key("synthetic/0001.jpg", 3)
        == "e24363c62d91fa5b61da3027736a3e64d17e6479a1db457a58e6af37f303e149"
    )
    assert tuple(
        inspect.signature(core.compute_centered_color_drop).parameters
    ) == (
        "original_image_feature",
        "edited_image_features",
        "color_text_features",
    )
    assert tuple(
        inspect.signature(
            clip_selector.FrozenWholeImageColorSelector.__call__
        ).parameters
    ) == ("self", "original_rgb", "edited_rgb")
    clip_source = "\n".join(
        (
            inspect.getsource(
                clip_selector.FrozenWholeImageColorSelector.__call__
            ),
            inspect.getsource(
                clip_selector.FrozenWholeImageColorSelector._encode_whole_rgb
            ),
            inspect.getsource(
                clip_selector.FrozenWholeImageColorSelector._normalize_whole_rgb
            ),
        )
    ).lower()
    for forbidden in ("pose", "keypoint", "slot", "region", "d0"):
        assert forbidden not in clip_source
    dimension = len(core.COLOR_NAMES)
    original = torch.zeros(dimension)
    original[2] = 1.0
    edited = torch.eye(core.ACTIVE_PROPOSALS_PER_IMAGE, dimension)
    text = torch.eye(dimension)
    drop = core.compute_centered_color_drop(original, edited, text)
    assert drop.shape == (
        core.ACTIVE_PROPOSALS_PER_IMAGE,
        len(core.COLOR_NAMES),
    )
    selection = core.select_clip_candidate(drop)
    assert 0 <= selection["aspect_index"] < 7
    assert 0 <= selection["selector_color_index"] < len(core.COLOR_NAMES)
    _expect_raises(
        lambda: core.select_clip_candidate(
            torch.zeros(core.PROPOSALS_PER_POOL, len(core.COLOR_NAMES))
        )
    )

    candidates = core.active_proposals(fixed, 1)
    reference = candidates[3]
    shifts = np.linspace(0.098, 0.102, 7)
    ce = np.linspace(0.0, 0.12, 7)
    top5 = np.ones(7, dtype=np.bool_)
    eligible = core.caliper_eligible(
        reference,
        candidates,
        0.10,
        shifts,
        0.06,
        ce,
        True,
        top5,
        require_centroid=True,
        allow_reference=False,
    )
    assert not eligible[3]
    chosen = core.select_caliper_hash_candidate(
        "synthetic/0001.jpg", candidates, eligible
    )
    assert chosen is not None and eligible[chosen]
    assert candidates[chosen]["mask_sha256"] != reference["mask_sha256"]
    return candidates, eligible


def _assert_blind_and_strong_controls(candidates, eligible, fields, valid):
    reference = candidates[3]
    mask = reference["mask"]
    target_slot = 1
    fields = torch.zeros_like(fields)
    fields[target_slot] = mask.float()
    # Add a small halo so the selected mask captures >25%, not exactly a
    # hand-written non-formal area.
    fields[target_slot] = torch.nn.functional.max_pool2d(
        fields[target_slot][None, None], kernel_size=9, stride=1, padding=4
    )[0, 0]
    valid = torch.zeros_like(valid)
    valid[target_slot] = True
    original = torch.full(
        (3, core.IMAGE_HEIGHT, core.IMAGE_WIDTH), 0.5
    )
    red = torch.tensor(core.COLOR_PROTOTYPE_RGB[2]).reshape(3, 1)
    original[:, mask] = red
    fill = core.deterministic_fill("synthetic/0002.jpg")
    edited = core.apply_candidate(original, mask, fill)
    result = core.blind_evaluate(
        original,
        edited,
        mask,
        fields,
        valid,
        expected_anchor_index=target_slot,
        identity_safe=True,
    )
    assert result["target_slot"] == target_slot
    assert result["blind_color_name"] == "red"
    assert result["component_pixels"] >= core.COLOR_COMPONENT_PIXELS_MIN
    assert result["coherent_color_removal"] and result["Y"] == 1

    evaluations = [dict(result) for _ in range(7)]
    # Strong controls retain the P+C reference itself. If the reference is the
    # raw-color/D0 optimum, equivalence must be reported rather than forcing a
    # second-best mask.
    reference = candidates[3]
    strong_base = core.caliper_eligible(
        reference,
        candidates,
        0.10,
        np.linspace(0.098, 0.102, 7),
        0.06,
        np.linspace(0.0, 0.12, 7),
        True,
        np.ones(7, dtype=np.bool_),
        require_centroid=True,
        allow_reference=True,
    )
    identity_safe = np.ones(7, dtype=np.bool_)
    identity_safe[0] = False
    strong_eligible = core.strong_control_eligible(
        strong_base, identity_safe
    )
    assert strong_eligible[3]
    evaluations[3]["blind_score"] = result["blind_score"] + 1.0
    raw = core.select_raw_color_candidate(evaluations, strong_eligible)
    hard = core.select_d0_hard_candidate(
        np.asarray([0.1, 0.101, 0.102, 0.106, 0.103, 0.104, 0.105]),
        strong_eligible,
    )
    assert raw == 3 and hard == 3
    identity = core.d0_identity_gate(
        True,
        True,
        0.105,
        np.linspace(0.09, 0.12, core.ROA_COUNT),
    )
    assert identity["identity_safe"]


def _assert_fixed_rows_and_statistics():
    ids = [f"{index:04d}.jpg" for index in range(core.ORACLE_COUNT)]
    arm_rows = {}
    all_edges = {name: True for name in core.QUARTET_EDGE_NAMES}
    for arm_name in core.FACTORIAL_ARM_NAMES:
        rows = [
            {
                "row_id": row_id,
                "arm_complete": True,
                "match_edges": dict(all_edges),
                "Y": 1,
            }
            for row_id in ids
        ]
        arm_rows[arm_name] = rows
    # A failure in only one arm must atomically zero all four outcomes.
    arm_rows["clip_only"][5].pop("Y")
    arm_rows["pose_only"][6]["match_edges"]["p_given_c0"] = False
    finalized = core.finalize_factorial_rows(arm_rows)
    for values in finalized["outcomes"].values():
        assert values.shape == (core.ORACLE_COUNT,)
        assert values[5] == 0 and values[6] == 0
    _expect_raises(
        lambda: core.finalize_factorial_rows(
            {
                name: rows[:4]
                for name, rows in arm_rows.items()
            }
        )
    )

    reference_rows = [
        {
            "row_id": row_id,
            "arm_complete": True,
            "pair_match_ok": True,
            "Y": 1,
        }
        for row_id in ids
    ]
    control_rows = copy.deepcopy(reference_rows)
    control_rows[9]["arm_complete"] = False
    pair = core.finalize_paired_control_rows(reference_rows, control_rows)
    assert pair["reference"][9] == pair["control"][9] == 0
    assert int(pair["pair_matched"].sum()) == core.ORACLE_COUNT - 1
    outcomes = finalized["outcomes"]
    interaction = core.paired_bootstrap_interaction(
        outcomes["pc"],
        outcomes["pose_only"],
        outcomes["clip_only"],
        outcomes["neither"],
        repetitions=1000,
    )
    assert interaction["estimate"] == 0.0
    difference = core.paired_bootstrap_difference(
        outcomes["pc"],
        outcomes["pose_only"],
        repetitions=1000,
        salt="c_given_p1",
    )
    assert difference["estimate"] == 0.0
    _expect_raises(
        lambda: core.factorial_interaction(
            [1, 0], [0, 0], [0, 0], [0, 0]
        )
    )


def _assert_common_noop():
    valid = {
        name: True for name in core.TRAINING_INTERVENTION_ARM_NAMES
    }
    passed = core.common_training_view_modes(valid)
    assert passed["quartet_valid"] and not passed["drop_sample"]
    valid["raw_color"] = False
    failed = core.common_training_view_modes(valid)
    assert not failed["quartet_valid"] and not failed["drop_sample"]
    assert set(failed["view_modes"].values()) == {"clean_noop"}


def main():
    _assert_sample_selection()
    fixed, _, fields, valid = _assert_proposals()
    _assert_edit_and_colors(fixed)
    candidates, eligible = _assert_scorer_and_caliper(fixed)
    _assert_blind_and_strong_controls(candidates, eligible, fields, valid)
    _assert_fixed_rows_and_statistics()
    _assert_common_noop()
    print("EXP415_STATIC_CONTRACT_V3=PASS")


if __name__ == "__main__":
    main()
