"""Formal CPU preflight for the exp374 override-to-real-Swin seam.

This module intentionally contains one comparatively heavyweight test.  It
constructs a randomly initialized production ``PoseBackboneModel`` with the
real Swin-Tiny backbone, then performs one tensor-override forward and one
explicit-None bypass forward at the frozen 384x128 evaluation resolution.

It must be run in its own process with an external 300-second timeout and a
small CPU thread budget (for example OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4).
The test does not use pytest-timeout, a checkpoint, a dataset, CUDA, or MPS.
Writing this file does not authorize running it or the formal Gate-A audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn

from experiments.exp374_psg_fuel_gate.audit_gate_a import (
    PSGInputCapture,
    assert_isolated_psg,
    audit_override_context,
    resolved_config,
)
from experiments.exp374_psg_fuel_gate.protocol import (
    actual_psg_input,
    canonical_json_bytes,
    sha256_bytes,
    sha256_tensor,
)
from model import make_model
from model.pose_backbone_model import PoseBackboneModel
import model.backbones.swin_transformer as swin_module


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/occluded_duke/pose_backbone_psg.yml"
EXTERNAL_TIMEOUT_SECONDS = 300
EXPECTED_OVERRIDE_EVENTS = [
    "b0_pre",
    "b0_post",
    "g0_pre",
    "e0_pre",
    "g0_post",
    "b1_pre",
    "b1_post",
    "g1_pre",
    "e1_pre",
    "g1_post",
]
EXPECTED_BYPASS_EVENTS = [
    "b0_pre",
    "b0_post",
    "b1_pre",
    "b1_post",
]


def _model_state_sha256(model: nn.Module) -> str:
    """Hash values, dtypes, shapes, and names without one giant copy."""

    entries = {
        name: sha256_tensor(value)
        for name, value in sorted(model.state_dict().items())
    }
    return sha256_bytes(canonical_json_bytes(entries))


def _assert_cpu_tensor(value: torch.Tensor) -> None:
    assert torch.is_tensor(value)
    assert value.device.type == "cpu"


def _assert_same_storage(first: torch.Tensor, second: torch.Tensor) -> None:
    """Prove that adjacent production modules received the same tensor."""

    _assert_cpu_tensor(first)
    _assert_cpu_tensor(second)
    assert first.data_ptr() == second.data_ptr()
    assert first.storage_offset() == second.storage_offset()
    assert first.shape == second.shape
    assert first.stride() == second.stride()
    assert first.dtype == second.dtype


def _assert_eval_output(output) -> None:
    assert isinstance(output, tuple)
    assert len(output) == 2
    descriptor, feature_maps = output
    _assert_cpu_tensor(descriptor)
    assert descriptor.shape == (1, 768)
    assert descriptor.dtype == torch.float32
    assert bool(torch.isfinite(descriptor).all())
    assert isinstance(feature_maps, list)
    assert len(feature_maps) == 4
    assert feature_maps[-1].shape == (1, 768, 12, 4)
    assert all(value.device.type == "cpu" for value in feature_maps)
    assert all(bool(torch.isfinite(value).all()) for value in feature_maps)


def _requested_device_type(value) -> str | None:
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, str):
        try:
            return torch.device(value).type
        except (RuntimeError, ValueError, TypeError):
            return None
    return None


def _install_no_io_no_accelerator_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[str]]:
    load_attempts: list[str] = []
    accelerator_attempts: list[str] = []

    def forbidden_load(*args, **kwargs):
        del kwargs
        source = str(args[0]) if args else "<missing>"
        load_attempts.append(source)
        raise AssertionError(f"formal CPU preflight attempted weight loading: {source}")

    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(swin_module, "_load_checkpoint", forbidden_load)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden_load)

    original_tensor_to = torch.Tensor.to

    def guarded_tensor_to(self, *args, **kwargs):
        requested = [_requested_device_type(value) for value in args]
        requested.append(_requested_device_type(kwargs.get("device")))
        forbidden = [value for value in requested if value in {"cuda", "mps"}]
        if forbidden:
            accelerator_attempts.append(forbidden[0])
            raise AssertionError(
                f"formal CPU preflight attempted accelerator transfer: {forbidden[0]}")
        return original_tensor_to(self, *args, **kwargs)

    def forbidden_cuda(self, *args, **kwargs):
        del self, args, kwargs
        accelerator_attempts.append("cuda")
        raise AssertionError("formal CPU preflight attempted .cuda()")

    monkeypatch.setattr(torch.Tensor, "to", guarded_tensor_to)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden_cuda)
    monkeypatch.setattr(nn.Module, "cuda", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if (hasattr(torch.backends, "mps")
            and hasattr(torch.backends.mps, "is_available")):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    return load_attempts, accelerator_attempts


def _register_stage3_trace_hooks(
    model: PoseBackboneModel,
    events: list[str],
    references: dict[str, object],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    stage = model.base.stages[3]
    gate0 = model.psg_modules_dict["s3_b0"]
    gate1 = model.psg_modules_dict["s3_b1"]

    def block_pre(name: str, key: str) -> Callable:
        def hook(_module, inputs):
            events.append(name)
            assert len(inputs) == 2
            references[key] = inputs[0]
            references[f"{key}_hw"] = tuple(inputs[1])
        return hook

    def block_post(name: str, key: str) -> Callable:
        def hook(_module, _inputs, output):
            events.append(name)
            references[key] = output
        return hook

    def gate_pre(name: str, key: str) -> Callable:
        def hook(_module, inputs):
            events.append(name)
            assert len(inputs) == 3
            references[key] = inputs[0]
            references[f"{key}_hw"] = tuple(inputs[1])
            references[f"{key}_scene"] = inputs[2]
        return hook

    def gate_post(name: str, key: str) -> Callable:
        def hook(_module, _inputs, output):
            events.append(name)
            references[key] = output
        return hook

    def encoder_pre(name: str, key: str) -> Callable:
        def hook(_module, inputs):
            events.append(name)
            assert len(inputs) == 1
            references[key] = inputs[0]
        return hook

    handles.extend([
        stage.blocks[0].register_forward_pre_hook(block_pre("b0_pre", "b0_in")),
        stage.blocks[0].register_forward_hook(block_post("b0_post", "b0_out")),
        gate0.register_forward_pre_hook(gate_pre("g0_pre", "g0_in")),
        gate0.encoder.register_forward_pre_hook(encoder_pre("e0_pre", "e0_in")),
        gate0.register_forward_hook(gate_post("g0_post", "g0_out")),
        stage.blocks[1].register_forward_pre_hook(block_pre("b1_pre", "b1_in")),
        stage.blocks[1].register_forward_hook(block_post("b1_post", "b1_out")),
        gate1.register_forward_pre_hook(gate_pre("g1_pre", "g1_in")),
        gate1.encoder.register_forward_pre_hook(encoder_pre("e1_pre", "e1_in")),
        gate1.register_forward_hook(gate_post("g1_post", "g1_out")),
    ])
    return handles


def _build_random_cpu_model() -> tuple[PoseBackboneModel, object]:
    local_cfg = resolved_config(CONFIG_PATH, []).clone()
    local_cfg.defrost()
    local_cfg.MODEL.PRETRAIN_PATH = ""
    local_cfg.MODEL.PRETRAIN_CHOICE = "none"
    local_cfg.MODEL.WITH_CP = False
    local_cfg.freeze()
    assert local_cfg.MODEL.PRETRAIN_PATH == ""
    assert str(local_cfg.MODEL.PRETRAIN_CHOICE) == "none"

    model = make_model(
        local_cfg,
        num_class=1,
        camera_num=0,
        view_num=0,
        semantic_weight=float(local_cfg.MODEL.SEMANTIC_WEIGHT),
    )
    assert type(model) is PoseBackboneModel
    model.eval()
    assert_isolated_psg(local_cfg, model)
    return model, local_cfg


def test_real_swin_full_forward_override_and_bypass_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove override -> real Stage-3 PSG order and true bypass call counts."""

    assert EXTERNAL_TIMEOUT_SECONDS == 300
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    load_attempts, accelerator_attempts = _install_no_io_no_accelerator_guards(
        monkeypatch)

    # Seed and restore only the CPU generator.  Do not call torch.manual_seed()
    # (which promises to seed accelerators) or fork_rng() (whose default device
    # module is CUDA even when the explicit device list is empty).
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(374)
        model, _local_cfg = _build_random_cpu_model()
        assert all(value.device.type == "cpu" for value in model.parameters())
        assert all(value.device.type == "cpu" for value in model.buffers())

        assert len(model.base.stages) == 4
        assert [len(stage.blocks) for stage in model.base.stages] == [2, 2, 6, 2]
        assert model.base.stages[3].downsample is None
        assert model.base.num_features[3] == 768
        assert model.psg_stage_indices == {3}
        assert set(model.psg_modules_dict.keys()) == {"s3_b0", "s3_b1"}
        assert len(model.psg_modules) == 2
        assert model.psg_modules[0] is model.psg_modules_dict["s3_b0"]
        assert model.psg_modules[1] is model.psg_modules_dict["s3_b1"]

        rgb = torch.zeros((1, 3, 384, 128), dtype=torch.float32)
        override = torch.linspace(
            -0.1,
            1.0,
            17 * 96 * 32,
            dtype=torch.float32,
        ).reshape(1, 17, 96, 32).contiguous()
        frozen_override = override.clone()
        _assert_cpu_tensor(rgb)
        _assert_cpu_tensor(override)

        events: list[str] = []
        references: dict[str, object] = {}
        capture = PSGInputCapture(model)
        handles = _register_stage3_trace_hooks(model, events, references)
        state_before = _model_state_sha256(model)
        try:
            with audit_override_context(model), torch.inference_mode():
                override_output = model(
                    rgb,
                    pose_dict=None,
                    scene_heatmaps_override=override,
                )
            _assert_eval_output(override_output)
            assert events == EXPECTED_OVERRIDE_EVENTS
            assert model._audit_scene_override_enabled is False

            assert references["b0_in_hw"] == (12, 4)
            assert references["g0_in_hw"] == (12, 4)
            assert references["b1_in_hw"] == (12, 4)
            assert references["g1_in_hw"] == (12, 4)
            _assert_same_storage(references["b0_out"], references["g0_in"])
            _assert_same_storage(references["g0_out"], references["b1_in"])
            _assert_same_storage(references["b1_out"], references["g1_in"])
            _assert_same_storage(override, references["g0_in_scene"])
            _assert_same_storage(override, references["g1_in_scene"])

            expected_encoder_input = actual_psg_input(override, (12, 4))
            assert torch.equal(override, frozen_override)
            assert float(expected_encoder_input.min()) < 0.5
            assert float(expected_encoder_input.max()) > 0.5
            assert len(capture.values["s3_b0"]) == 1
            assert len(capture.values["s3_b1"]) == 1
            for key, reference_key in (
                ("s3_b0", "e0_in"),
                ("s3_b1", "e1_in"),
            ):
                actual = capture.values[key][0]
                _assert_cpu_tensor(actual)
                assert actual.shape == (1, 17, 12, 4)
                assert actual.dtype == torch.float32
                assert torch.equal(actual, expected_encoder_input)
                assert torch.equal(references[reference_key], expected_encoder_input)
            captured = capture.pop(expected_calls=1)
            assert captured is not None
            assert torch.equal(captured, expected_encoder_input)

            events.clear()
            references.clear()
            with audit_override_context(model), torch.inference_mode():
                bypass_output = model(
                    rgb,
                    pose_dict=None,
                    scene_heatmaps_override=None,
                )
            _assert_eval_output(bypass_output)
            assert events == EXPECTED_BYPASS_EVENTS
            assert model._audit_scene_override_enabled is False
            assert capture.pop(expected_calls=0) is None
            assert all(not name.startswith(("g", "e")) for name in events)
            assert "g0_in" not in references and "g1_in" not in references
            _assert_same_storage(references["b0_out"], references["b1_in"])

            state_after = _model_state_sha256(model)
            assert state_after == state_before
            assert load_attempts == []
            assert accelerator_attempts == []
            assert all(value.device.type == "cpu" for value in model.parameters())
            assert all(value.device.type == "cpu" for value in model.buffers())
        finally:
            capture.close()
            for handle in handles:
                handle.remove()
            model._audit_scene_override_enabled = False
    finally:
        torch.random.set_rng_state(cpu_rng_state)
