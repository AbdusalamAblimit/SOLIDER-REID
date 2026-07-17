"""Pure-CPU unit tests for the exp374 model audit seam.

These tests bind the production ``PoseBackboneModel.forward`` method directly
to a tiny receiver.  They do not construct a Swin model, load a checkpoint,
build a DataLoader, or use an accelerator.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.exp374_psg_fuel_gate.audit_gate_a import (
    PSGInputCapture,
    audit_override_context,
)
from experiments.exp374_psg_fuel_gate.protocol import GateProtocolError
from model.modules.pose_spatial_gate import PoseSpatialGate
from model.pose_backbone_model import (
    SCENE_HEATMAPS_UNSET,
    PoseBackboneModel,
)


BATCH = 2
SCENE_SHAPE = (BATCH, 17, 96, 32)


class TinyForwardHarness(nn.Module):
    """Minimum receiver for the production forward method."""

    forward = PoseBackboneModel.forward

    def __init__(self) -> None:
        super().__init__()
        self._audit_scene_override_enabled = False
        self.use_target_heatmap = False
        self.pose_dropout_p = 0.0
        self.reduce_feat_dim = False
        self.bottleneck = nn.Identity()
        self.neck_feat = "before"
        self.use_skeleton_gcn = False
        self.psg_modules_dict = nn.ModuleDict({
            "s3_b0": PoseSpatialGate(
                pose_channels=17, feat_channels=8, hidden_dim=4),
            "s3_b1": PoseSpatialGate(
                pose_channels=17, feat_channels=8, hidden_dim=4),
        })
        self.legacy_scene = torch.linspace(
            0.0, 1.0, 17 * 96 * 32, dtype=torch.float32,
        ).reshape(1, 17, 96, 32).repeat(BATCH, 1, 1, 1).contiguous()
        self.prepare_calls: list[object] = []
        self.run_scenes: list[torch.Tensor | None] = []
        self.eval()

    def _prepare_pose(self, pose_dict):
        self.prepare_calls.append(pose_dict)
        scores = torch.zeros((BATCH, 17), dtype=torch.float32)
        target = torch.zeros_like(self.legacy_scene)
        difference = torch.zeros_like(self.legacy_scene)
        return self.legacy_scene, scores, target, difference

    def _run_backbone_with_psg(self, x, scene_heatmaps, pose_dict=None):
        self.run_scenes.append(scene_heatmaps)
        if scene_heatmaps is not None:
            tokens = torch.zeros(
                (x.shape[0], 12 * 4, 8), dtype=torch.float32, device=x.device)
            for key in ("s3_b0", "s3_b1"):
                tokens = self.psg_modules_dict[key](
                    tokens, (12, 4), scene_heatmaps)
        global_feature = torch.zeros(
            (x.shape[0], 768), dtype=torch.float32, device=x.device)
        feature_map = torch.zeros(
            (x.shape[0], 8, 12, 4), dtype=torch.float32, device=x.device)
        return global_feature, [feature_map]


@pytest.fixture
def model() -> TinyForwardHarness:
    return TinyForwardHarness()


@pytest.fixture
def captured_model(model: TinyForwardHarness):
    capture = PSGInputCapture(model)
    try:
        yield model, capture
    finally:
        capture.close()


def _images() -> torch.Tensor:
    return torch.zeros((BATCH, 3, 8, 4), dtype=torch.float32)


def _valid_scene(maximum: float = 2.0) -> torch.Tensor:
    return torch.linspace(
        0.0, maximum, BATCH * 17 * 96 * 32, dtype=torch.float32,
    ).reshape(SCENE_SHAPE).contiguous()


def _signed_scene() -> torch.Tensor:
    return torch.linspace(
        -2.0, 2.0, BATCH * 17 * 96 * 32, dtype=torch.float32,
    ).reshape(SCENE_SHAPE).contiguous()


def _assert_eval_output(output) -> None:
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert torch.is_tensor(output[0])
    assert output[0].shape == (BATCH, 768)
    assert isinstance(output[1], list)


def _assert_gate_input(scene: torch.Tensor, actual: torch.Tensor) -> None:
    expected = torch.sigmoid(F.interpolate(
        scene, size=(12, 4), mode="bilinear", align_corners=False))
    assert torch.equal(actual, expected)


def test_signature_preserves_legacy_and_keyword_only_override(
    model: TinyForwardHarness,
) -> None:
    signature = inspect.signature(PoseBackboneModel.forward)
    parameters = signature.parameters
    assert list(parameters) == [
        "self", "x", "label", "cam_label", "view_label", "pose_dict",
        "scene_heatmaps_override",
    ]
    assert parameters["pose_dict"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["pose_dict"].default is None
    assert parameters["scene_heatmaps_override"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["scene_heatmaps_override"].default is SCENE_HEATMAPS_UNSET

    pose = object()
    output = model(_images(), None, None, None, pose)
    _assert_eval_output(output)
    assert model.prepare_calls == [pose]
    assert len(model.run_scenes) == 1
    assert model.run_scenes[0] is model.legacy_scene


def test_unset_uses_legacy_pose_path_and_real_gate_hook(captured_model) -> None:
    model, capture = captured_model
    pose = object()

    first = model(_images(), pose_dict=pose)
    _assert_eval_output(first)
    actual = capture.pop(expected_calls=1)
    assert actual is not None
    _assert_gate_input(model.legacy_scene, actual)

    second = model(
        _images(), pose_dict=pose,
        scene_heatmaps_override=SCENE_HEATMAPS_UNSET,
    )
    _assert_eval_output(second)
    actual = capture.pop(expected_calls=1)
    assert actual is not None
    _assert_gate_input(model.legacy_scene, actual)

    assert model.prepare_calls == [pose, pose]
    assert len(model.run_scenes) == 2
    assert all(scene is model.legacy_scene for scene in model.run_scenes)
    assert model._audit_scene_override_enabled is False


def test_unset_without_pose_keeps_legacy_no_pose_path(captured_model) -> None:
    model, capture = captured_model
    output = model(_images())
    _assert_eval_output(output)
    assert model.prepare_calls == []
    assert model.run_scenes == [None]
    assert capture.pop(expected_calls=0) is None


def test_explicit_none_is_true_bypass(captured_model) -> None:
    model, capture = captured_model
    with audit_override_context(model):
        output = model(
            _images(), pose_dict=None, scene_heatmaps_override=None)
    _assert_eval_output(output)
    assert model.prepare_calls == []
    assert model.run_scenes == [None]
    assert capture.pop(expected_calls=0) is None
    assert model._audit_scene_override_enabled is False


def test_tensor_override_bypasses_prepare_and_hits_both_real_gates(
    captured_model,
) -> None:
    model, capture = captured_model
    scene = _signed_scene()
    with audit_override_context(model):
        output = model(
            _images(), pose_dict=None, scene_heatmaps_override=scene)
    _assert_eval_output(output)
    assert model.prepare_calls == []
    assert len(model.run_scenes) == 1
    assert model.run_scenes[0] is scene
    actual = capture.pop(expected_calls=1)
    assert actual is not None
    _assert_gate_input(scene, actual)
    assert bool((scene < 0).any())
    assert float(actual.min()) < 0.5
    assert float(actual.max()) > 0.5
    assert all(not values for values in capture.values.values())


def test_signed_legacy_and_override_have_exact_gate_and_output_parity(
    captured_model,
) -> None:
    model, capture = captured_model
    scene = _signed_scene()
    model.legacy_scene = scene
    pose = object()

    legacy_output = model(_images(), pose_dict=pose)
    legacy_actual = capture.pop(expected_calls=1)
    assert legacy_actual is not None

    with audit_override_context(model):
        override_output = model(
            _images(), pose_dict=None, scene_heatmaps_override=scene)
    override_actual = capture.pop(expected_calls=1)
    assert override_actual is not None

    _assert_eval_output(legacy_output)
    _assert_eval_output(override_output)
    assert torch.equal(legacy_actual, override_actual)
    _assert_gate_input(scene, legacy_actual)
    assert torch.equal(legacy_output[0], override_output[0])
    assert len(legacy_output[1]) == len(override_output[1])
    assert all(
        torch.equal(legacy, override)
        for legacy, override in zip(legacy_output[1], override_output[1])
    )
    assert model.prepare_calls == [pose]
    assert len(model.run_scenes) == 2
    assert model.run_scenes[0] is scene
    assert model.run_scenes[1] is scene
    assert model._audit_scene_override_enabled is False


@pytest.mark.parametrize("kind", ["none", "tensor"])
def test_override_is_disabled_outside_audit_context(
    model: TinyForwardHarness,
    kind: str,
) -> None:
    override = None if kind == "none" else _valid_scene()
    with pytest.raises(RuntimeError, match="disabled outside an audit context"):
        model(_images(), scene_heatmaps_override=override)
    assert model.prepare_calls == []
    assert model.run_scenes == []


@pytest.mark.parametrize("kind", ["none", "tensor"])
def test_override_is_eval_only(model: TinyForwardHarness, kind: str) -> None:
    override = None if kind == "none" else _valid_scene()
    model.train()
    with audit_override_context(model):
        with pytest.raises(RuntimeError, match="requires eval mode"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []
    assert model._audit_scene_override_enabled is False


@pytest.mark.parametrize("kind", ["none", "tensor"])
def test_override_and_pose_dict_are_mutually_exclusive(
    model: TinyForwardHarness,
    kind: str,
) -> None:
    override = None if kind == "none" else _valid_scene()
    with audit_override_context(model):
        with pytest.raises(ValueError, match="mutually exclusive"):
            model(
                _images(), pose_dict=object(),
                scene_heatmaps_override=override,
            )
    assert model.prepare_calls == []
    assert model.run_scenes == []


def test_context_is_instance_local_and_restores_after_success() -> None:
    first = TinyForwardHarness()
    second = TinyForwardHarness()
    with audit_override_context(first):
        assert first._audit_scene_override_enabled is True
        assert second._audit_scene_override_enabled is False
        with pytest.raises(RuntimeError, match="disabled outside an audit context"):
            second(_images(), scene_heatmaps_override=_valid_scene())
    assert first._audit_scene_override_enabled is False
    assert second._audit_scene_override_enabled is False


def test_context_restores_after_exception_and_rejects_nesting(
    model: TinyForwardHarness,
) -> None:
    with pytest.raises(LookupError):
        with audit_override_context(model):
            assert model._audit_scene_override_enabled is True
            raise LookupError("synthetic")
    assert model._audit_scene_override_enabled is False

    with audit_override_context(model):
        with pytest.raises(GateProtocolError) as error:
            with audit_override_context(model):
                pass
        assert error.value.code == "E_OVERRIDE_CONTEXT"
        assert model._audit_scene_override_enabled is True
    assert model._audit_scene_override_enabled is False


@pytest.mark.parametrize("override", [object(), 1, []])
def test_override_rejects_non_tensor_values(
    model: TinyForwardHarness,
    override,
) -> None:
    with audit_override_context(model):
        with pytest.raises(TypeError, match="must be Tensor"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []


@pytest.mark.parametrize("shape", [
    (1, 17, 96, 32),
    (BATCH, 16, 96, 32),
    (BATCH, 17, 95, 32),
    (BATCH, 17, 96, 31),
    (BATCH, 17, 96, 32, 1),
])
def test_override_rejects_wrong_shape(
    model: TinyForwardHarness,
    shape: tuple[int, ...],
) -> None:
    override = torch.zeros(shape, dtype=torch.float32)
    with audit_override_context(model):
        with pytest.raises(ValueError, match="shape mismatch"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int64])
def test_override_rejects_wrong_dtype(
    model: TinyForwardHarness,
    dtype: torch.dtype,
) -> None:
    override = torch.zeros(SCENE_SHAPE, dtype=dtype)
    with audit_override_context(model):
        with pytest.raises(TypeError, match="must be float32"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []


def test_override_rejects_device_mismatch_without_accelerator(
    model: TinyForwardHarness,
) -> None:
    override = torch.empty(SCENE_SHAPE, dtype=torch.float32, device="meta")
    with audit_override_context(model):
        with pytest.raises(ValueError, match="RGB input device"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []


def test_override_rejects_noncontiguous_tensor(
    model: TinyForwardHarness,
) -> None:
    override = torch.zeros(
        (BATCH, 17, 32, 96), dtype=torch.float32).transpose(-1, -2)
    assert override.shape == SCENE_SHAPE
    assert not override.is_contiguous()
    with audit_override_context(model):
        with pytest.raises(ValueError, match="must be contiguous"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_override_rejects_nonfinite_values(
    captured_model,
    bad_value: float,
) -> None:
    model, capture = captured_model
    override = torch.zeros(SCENE_SHAPE, dtype=torch.float32)
    override[0, 0, 0, 0] = bad_value
    with audit_override_context(model):
        with pytest.raises(ValueError, match="contains NaN/Inf"):
            model(_images(), scene_heatmaps_override=override)
    assert model.run_scenes == []
    assert capture.pop(expected_calls=0) is None


def test_override_accepts_finite_signed_zero_and_values_above_one(
    model: TinyForwardHarness,
) -> None:
    for valid in (
        _signed_scene(),
        torch.zeros(SCENE_SHAPE, dtype=torch.float32),
        torch.full(SCENE_SHAPE, 3.0, dtype=torch.float32),
    ):
        with audit_override_context(model):
            output = model(_images(), scene_heatmaps_override=valid)
        _assert_eval_output(output)
    assert len(model.run_scenes) == 3
    assert torch.equal(model.run_scenes[0], _signed_scene())
    assert torch.equal(
        model.run_scenes[1], torch.zeros(SCENE_SHAPE, dtype=torch.float32))
    assert torch.equal(
        model.run_scenes[2], torch.full(SCENE_SHAPE, 3.0, dtype=torch.float32))


def test_validation_order_is_fail_closed(model: TinyForwardHarness) -> None:
    malformed = torch.zeros((1, 1), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="disabled outside an audit context"):
        model(_images(), scene_heatmaps_override=malformed)

    model.train()
    with audit_override_context(model):
        with pytest.raises(RuntimeError, match="requires eval mode"):
            model(_images(), scene_heatmaps_override=malformed)
    model.eval()

    with audit_override_context(model):
        with pytest.raises(ValueError, match="mutually exclusive"):
            model(
                _images(), pose_dict=object(),
                scene_heatmaps_override=malformed,
            )


def test_hook_rejects_missing_and_extra_calls(captured_model) -> None:
    model, capture = captured_model
    value = torch.zeros((BATCH, 17, 12, 4), dtype=torch.float32)

    model.psg_modules_dict["s3_b0"].encoder(value)
    with pytest.raises(GateProtocolError) as missing:
        capture.pop(expected_calls=1)
    assert missing.value.code == "E_HOOK_COUNT"
    capture.reset()

    model.psg_modules_dict["s3_b0"].encoder(value)
    model.psg_modules_dict["s3_b0"].encoder(value)
    model.psg_modules_dict["s3_b1"].encoder(value)
    with pytest.raises(GateProtocolError) as extra:
        capture.pop(expected_calls=1)
    assert extra.value.code == "E_HOOK_COUNT"
    capture.reset()


def test_hook_rejects_block_drift(captured_model) -> None:
    model, capture = captured_model
    first = torch.zeros((BATCH, 17, 12, 4), dtype=torch.float32)
    second = torch.ones((BATCH, 17, 12, 4), dtype=torch.float32)
    model.psg_modules_dict["s3_b0"].encoder(first)
    model.psg_modules_dict["s3_b1"].encoder(second)
    with pytest.raises(GateProtocolError) as error:
        capture.pop(expected_calls=1)
    assert error.value.code == "E_HOOK_BLOCK_DRIFT"
    capture.reset()


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (torch.zeros((BATCH, 17, 12, 4), dtype=torch.float64), "E_HOOK_DTYPE"),
        (torch.zeros((BATCH, 17, 11, 4), dtype=torch.float32), "E_HOOK_SHAPE"),
    ],
)
def test_hook_rejects_wrong_dtype_and_shape(captured_model, value, code) -> None:
    model, capture = captured_model
    with pytest.raises(GateProtocolError) as error:
        model.psg_modules_dict["s3_b0"].encoder(value)
    assert error.value.code == code
    capture.reset()


def test_hook_rejects_wrong_arity(captured_model) -> None:
    model, capture = captured_model
    value = torch.zeros((BATCH, 17, 12, 4), dtype=torch.float32)
    with pytest.raises(GateProtocolError) as error:
        model.psg_modules_dict["s3_b0"].encoder(value, value)
    assert error.value.code == "E_HOOK_INPUT"
    capture.reset()


def test_hook_clones_inputs_and_resets_after_pop(captured_model) -> None:
    model, capture = captured_model
    value = torch.linspace(
        0.0, 1.0, BATCH * 17 * 12 * 4, dtype=torch.float32,
    ).reshape(BATCH, 17, 12, 4)
    original = value.clone()
    model.psg_modules_dict["s3_b0"].encoder(value)
    model.psg_modules_dict["s3_b1"].encoder(value)
    value.zero_()

    actual = capture.pop(expected_calls=1)
    assert actual is not None
    assert torch.equal(actual, original)
    assert all(not values for values in capture.values.values())
    assert capture.pop(expected_calls=0) is None
