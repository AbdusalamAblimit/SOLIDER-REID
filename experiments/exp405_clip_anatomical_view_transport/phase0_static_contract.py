#!/usr/bin/env python3
"""Synthetic positive/negative contract for exp405 Phase 0.

Passing this script authorizes only the real teacher measurement implementation.
It is not CLIP, ReID, or novelty evidence.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import sys
import types

try:
    _EXP405_BOOTSTRAP_CONTEXT
    _EXP405_EXECUTED_CONTRACT_BYTES
    _EXP405_EXECUTION_SENTINEL
except NameError as error:
    raise RuntimeError(
        "phase0_static_contract.py must be executed from the frozen exp405 bootstrap"
    ) from error

import torch
import torch.nn.functional as F


SEED = 20260720
SLOTS = 5
FLOAT32_EQUIVARIANCE_ATOL = 5e-7
OUTPUT_NAME = re.compile(r"phase0_static_[A-Za-z0-9][A-Za-z0-9_.-]*\.json\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CORE_SHA256 = "29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b"
EXPECTED_PYTHON_VERSION = "3.11.15"
EXPECTED_TORCH_VERSION = "2.13.0"
EXPECTED_TORCH_GIT_VERSION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
EXPECTED_TORCH_CONFIG_SHA256 = "1943041ad11240ec18f123ae0f5aa98ed16a9ebf53b479bb4f0c11ea4c6f12f1"
EXPECTED_PYVENV_SHA256 = "39fe3064980027fed1216d0b9ced4da9e270652270d84f2bcfb74d694711cf48"
EXPECTED_TORCH_INIT_SHA256 = "cf40c075c95864036e835795756d69b8cccfafa76f3bcde5eba9d06065ccd3d1"
EXPECTED_TORCH_RECORD_SHA256 = "b5e76f2212a8b17cac6bf771887c4d8a647502d3e33bf7e61d720bbab1f89367"
EXPECTED_TORCH_C_SHA256 = "06b303bc0e60a65552970fa2e2ca395a6f32a70ea167e3dbba7be82d2d6cbc4f"
EXPECTED_LIBTORCH_PYTHON_SHA256 = "eb3f31be95527c2d9ff816ddce5f282f2031f71586628d008855a315c5683bb1"
EXPECTED_LIBTORCH_CPU_SHA256 = "f1584a65a2a09b5ddbe90a4e195ba824087b430a9f21cb1df9b8894177b99987"
EXPECTED_SITE_TREE_SHA256 = "b3428d0e161d8bb6b2f98cc75301cf7ef4f67dac2cb3b5fea348b91b15faccf2"
EXPECTED_SITE_TREE_FILE_COUNT = 18763
EXPECTED_SITE_TREE_BYTE_COUNT = 585021614
EXPECTED_TORCH_RECORD_VERIFIED_FILES = 12713


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_core_from_frozen_source(path: Path, source_bytes: bytes):
    module = types.ModuleType("exp405_phase0_core")
    module.__file__ = str(path)
    code = compile(source_bytes, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def raises_value_error(callable_) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    return False


def masks(batch: int, height: int, width: int) -> torch.Tensor:
    result = torch.zeros(batch, SLOTS, height, width, dtype=torch.float64)
    boundaries = torch.linspace(0, height, SLOTS + 1).round().long()
    for slot in range(SLOTS):
        result[:, slot, boundaries[slot]:boundaries[slot + 1]] = 1.0
    return result


def build_ordering_fixture(core):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    batch, channels, height, width = 40, 12, 10, 6
    slot_masks = masks(batch, height, width)
    pids = torch.arange(batch) // 4
    identities = F.normalize(torch.randn(10, channels, generator=generator), dim=-1)
    anatomy = F.normalize(torch.randn(SLOTS, channels, generator=generator), dim=-1)
    identity_slot = F.normalize(
        torch.randn(10, SLOTS, channels, generator=generator), dim=-1
    )
    view_context = 0.5 * torch.randn(batch, channels, generator=generator)
    feature = torch.zeros(batch, channels, height, width, dtype=torch.float64)
    for row in range(batch):
        for slot in range(SLOTS):
            value = (
                2.0 * identities[pids[row]]
                + 0.8 * anatomy[slot]
                + 1.5 * identity_slot[pids[row], slot]
                + view_context[row]
            )
            feature[row] += slot_masks[row, slot].unsqueeze(0) * value[:, None, None]

    target_slot = torch.arange(batch) % SLOTS
    sample_paths = tuple(
        f"train/pid_{int(pids[row]):04d}/image_{row:05d}.jpg"
        for row in range(batch)
    )
    sample_key = core.stable_sample_keys(sample_paths)
    selected_mask = slot_masks[torch.arange(batch), target_slot]
    selected_mask_bool = selected_mask.bool()
    deleted, erase_mask, realized = core.deterministic_slot_delete(
        feature, selected_mask_bool, sample_key, target_slot,
        fraction=0.5, ordering_seed=SEED
    )
    deleted_repeat, erase_mask_repeat, realized_repeat = core.deterministic_slot_delete(
        feature, selected_mask_bool, sample_key, target_slot,
        fraction=0.5, ordering_seed=SEED
    )
    deleted_alt, erase_mask_alt, realized_alt = core.deterministic_slot_delete(
        feature, selected_mask_bool, sample_key, target_slot,
        fraction=0.5, ordering_seed=SEED + 1
    )
    batch_permutation = torch.arange(batch).roll(7)
    inverse_batch_permutation = torch.argsort(batch_permutation)
    permuted_paths = tuple(sample_paths[int(row)] for row in batch_permutation)
    regenerated_permuted_keys = core.stable_sample_keys(permuted_paths)
    deleted_permuted, erase_permuted, realized_permuted = core.deterministic_slot_delete(
        feature[batch_permutation], selected_mask_bool[batch_permutation],
        regenerated_permuted_keys, target_slot[batch_permutation],
        fraction=0.5, ordering_seed=SEED,
    )
    split_outputs = [
        core.deterministic_slot_delete(
            feature[part], selected_mask_bool[part],
            core.stable_sample_keys(sample_paths[part]), target_slot[part],
            fraction=0.5, ordering_seed=SEED,
        )
        for part in (slice(0, 17), slice(17, batch))
    ]
    deleted_split = torch.cat([value[0] for value in split_outputs])
    erase_split = torch.cat([value[1] for value in split_outputs])
    realized_split = torch.cat([value[2] for value in split_outputs])
    expected_deleted = feature.clone().permute(0, 2, 3, 1)
    expected_deleted[erase_mask] = 0.0
    expected_deleted = expected_deleted.permute(0, 3, 1, 2)
    changed_spatial = (deleted != feature).any(dim=1)
    pooled, geometry_valid, mass = core.mass_normalized_pool(feature, slot_masks)
    context_by_slot, _, _ = core.mass_normalized_pool(feature, 1.0 - slot_masks)

    same_pid_donor = torch.arange(batch).reshape(-1, 4).roll(1, dims=1).reshape(-1)
    wrong_pid_donor = (torch.arange(batch) + 4) % batch
    wrong_slot = (target_slot + 1) % SLOTS
    index = torch.arange(batch)
    recipient_context = context_by_slot[index, target_slot]
    donor_context = context_by_slot[same_pid_donor, target_slot]
    wrong_pid_context = context_by_slot[wrong_pid_donor, target_slot]
    correct_state = core.contextual_transport_state(
        pooled[same_pid_donor, target_slot], donor_context, recipient_context
    )
    same_id_wrong_slot = core.contextual_transport_state(
        pooled[same_pid_donor, wrong_slot],
        context_by_slot[same_pid_donor, wrong_slot],
        recipient_context,
    )
    wrong_id_same_slot = core.contextual_transport_state(
        pooled[wrong_pid_donor, target_slot], wrong_pid_context, recipient_context
    )
    generic = pooled.mean(dim=(0, 1), keepdim=True).expand(batch, -1, -1)[:, 0]
    generic_slot = pooled.mean(dim=0)[target_slot]
    generic_context = context_by_slot.mean(dim=0)[target_slot]
    generic_transport = core.contextual_transport_state(
        generic_slot, generic_context, recipient_context
    )
    random_key_source = (torch.arange(batch) + SLOTS) % batch
    random_key = correct_state[random_key_source]
    cluster_id = torch.arange(batch) % 8
    cluster_prototypes = F.normalize(
        torch.randn(8, channels, generator=generator), dim=-1
    ).double() * correct_state.norm(dim=-1).median()
    random_cluster = cluster_prototypes[cluster_id]

    arms = {
        "correct": correct_state,
        "same_id_wrong_slot": same_id_wrong_slot,
        "wrong_id_same_slot": wrong_id_same_slot,
        "generic": generic,
        "generic_transport": generic_transport,
        "random_key": random_key,
        "random_cluster": random_cluster,
    }
    outputs = {
        name: core.scatter_replace(deleted, erase_mask, state, budget=1.0)
        for name, state in arms.items()
    }
    outputs["NULL"] = core.scatter_replace(deleted, erase_mask, correct_state, budget=0.0)
    outputs["self_restore"] = core.scatter_replace(
        deleted, erase_mask, pooled[index, target_slot], budget=1.0
    )
    mutant_states = {
        "donor_slot_only": pooled[same_pid_donor, target_slot],
        "no_context_subtraction": pooled[same_pid_donor, target_slot] + recipient_context,
        "no_recipient_context_addition": pooled[same_pid_donor, target_slot] - donor_context,
        "wrong_recipient_context": core.contextual_transport_state(
            pooled[same_pid_donor, target_slot], donor_context,
            context_by_slot[wrong_pid_donor, target_slot],
        ),
    }
    mutant_outputs = {
        name: core.scatter_replace(deleted, erase_mask, state, budget=1.0)
        for name, state in mutant_states.items()
    }

    sentinel_state = pooled[index, target_slot] + 7.0
    erase_only_sentinel = core.scatter_replace(
        deleted, erase_mask, sentinel_state, budget=1.0
    )
    full_slot_sentinel_mutant = core.scatter_replace(
        deleted, selected_mask.bool(), sentinel_state, budget=1.0
    )
    observed_mask = selected_mask.bool() & ~erase_mask
    soft_budget = torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=feature.dtype).repeat(10)
    soft_scatter = core.scatter_replace(
        deleted, erase_mask, sentinel_state, budget=soft_budget
    )
    soft_alpha = erase_mask.to(feature.dtype).unsqueeze(1) * soft_budget[:, None, None, None]
    soft_expected = (
        deleted * (1.0 - soft_alpha)
        + sentinel_state[:, :, None, None].expand_as(deleted) * soft_alpha
    )
    binary_budget_mutant = core.scatter_replace(
        deleted, erase_mask, sentinel_state, budget=(soft_budget > 0).to(feature.dtype)
    )

    target = feature.flatten(1)
    utility = {
        name: float(-((value.flatten(1) - target) ** 2).mean())
        for name, value in outputs.items()
    }
    mutant_utility = {
        name: float(-((value.flatten(1) - target) ** 2).mean())
        for name, value in mutant_outputs.items()
    }
    return {
        "feature": feature,
        "slot_masks": slot_masks,
        "selected_mask": selected_mask,
        "selected_mask_bool": selected_mask_bool,
        "pooled": pooled,
        "geometry_valid": geometry_valid,
        "mass": mass,
        "deleted": deleted,
        "deleted_repeat": deleted_repeat,
        "realized": realized,
        "realized_repeat": realized_repeat,
        "realized_alt": realized_alt,
        "erase_mask": erase_mask,
        "erase_mask_repeat": erase_mask_repeat,
        "erase_mask_alt": erase_mask_alt,
        "batch_permutation": batch_permutation,
        "inverse_batch_permutation": inverse_batch_permutation,
        "deleted_permuted": deleted_permuted,
        "erase_permuted": erase_permuted,
        "realized_permuted": realized_permuted,
        "deleted_split": deleted_split,
        "erase_split": erase_split,
        "realized_split": realized_split,
        "expected_deleted": expected_deleted,
        "changed_spatial": changed_spatial,
        "deleted_alt": deleted_alt,
        "outputs": outputs,
        "utility": utility,
        "mutant_utility": mutant_utility,
        "pids": pids,
        "random_key_source": random_key_source,
        "cluster_id": cluster_id,
        "correct_state": correct_state,
        "random_key_state": random_key,
        "target_slot": target_slot,
        "sample_key": sample_key,
        "regenerated_permuted_keys": regenerated_permuted_keys,
        "observed_mask": observed_mask,
        "erase_only_sentinel": erase_only_sentinel,
        "full_slot_sentinel_mutant": full_slot_sentinel_mutant,
        "soft_budget": soft_budget,
        "soft_scatter": soft_scatter,
        "soft_expected": soft_expected,
        "binary_budget_mutant": binary_budget_mutant,
    }


def main(output_name: str) -> dict:
    context = _EXP405_BOOTSTRAP_CONTEXT
    required_context = {
        "bootstrap_path", "bootstrap_sha256", "contract_path", "contract_sha256",
        "contract_bytes", "core_bytes", "core_path", "core_sha256",
        "execution_sentinel", "import_path", "launcher_mode", "pyvenv_sha256",
        "site_packages_path", "torch_c_sha256", "torch_init_sha256",
        "torch_record_sha256", "libtorch_cpu_sha256", "libtorch_python_sha256",
        "site_tree_byte_count", "site_tree_file_count", "site_tree_sha256",
        "torch_record_verified_files",
    }
    if not isinstance(context, types.MappingProxyType) or set(context) != required_context:
        raise RuntimeError("invalid frozen bootstrap context")
    if context["execution_sentinel"] is not _EXP405_EXECUTION_SENTINEL:
        raise RuntimeError("bootstrap execution sentinel mismatch")
    if context["launcher_mode"] != "same-source-reexec-v4":
        raise RuntimeError("unrecognized frozen launcher mode")
    if not isinstance(output_name, str) or OUTPUT_NAME.fullmatch(output_name) is None:
        raise RuntimeError("output must be a non-empty phase0_static_*.json basename")
    if "/" in output_name or "\\" in output_name:
        raise RuntimeError("output must be a basename")
    contract_path = Path(context["contract_path"]).resolve()
    core_path = Path(context["core_path"]).resolve()
    bootstrap_path = Path(context["bootstrap_path"]).resolve()
    if contract_path != Path(__file__).resolve():
        raise RuntimeError("executed contract path does not match frozen context")
    if contract_path.name != "phase0_static_contract.py":
        raise RuntimeError("contract path is not canonical")
    if bootstrap_path != contract_path.with_name("run_phase0_static_frozen.py"):
        raise RuntimeError("bootstrap path is not the canonical sibling")
    canonical_core_path = contract_path.with_name("phase0_core.py")
    if core_path != canonical_core_path:
        raise RuntimeError("core path is not the canonical sibling")
    if any(
        not isinstance(context[name], str) or SHA256.fullmatch(context[name]) is None
        for name in ("bootstrap_sha256", "contract_sha256", "core_sha256")
    ):
        raise RuntimeError("invalid frozen digest provenance")
    contract_source_bytes = context["contract_bytes"]
    if (
        not isinstance(contract_source_bytes, bytes)
        or contract_source_bytes is not _EXP405_EXECUTED_CONTRACT_BYTES
        or sha256_bytes(contract_source_bytes) != context["contract_sha256"]
    ):
        raise RuntimeError("executed contract bytes do not match frozen provenance")
    core_source_bytes = context["core_bytes"]
    if not isinstance(core_source_bytes, bytes):
        raise RuntimeError("bootstrap did not inject immutable core bytes")
    actual_core_sha256 = sha256_bytes(core_source_bytes)
    if (
        actual_core_sha256 != context["core_sha256"]
        or actual_core_sha256 != EXPECTED_CORE_SHA256
    ):
        raise RuntimeError("injected core digest mismatch before execution")
    dependency_expectations = {
        "pyvenv_sha256": EXPECTED_PYVENV_SHA256,
        "torch_init_sha256": EXPECTED_TORCH_INIT_SHA256,
        "torch_record_sha256": EXPECTED_TORCH_RECORD_SHA256,
        "torch_c_sha256": EXPECTED_TORCH_C_SHA256,
        "libtorch_python_sha256": EXPECTED_LIBTORCH_PYTHON_SHA256,
        "libtorch_cpu_sha256": EXPECTED_LIBTORCH_CPU_SHA256,
    }
    if any(context[name] != expected for name, expected in dependency_expectations.items()):
        raise RuntimeError("frozen dependency artifact digest mismatch")
    if (
        context["site_tree_sha256"] != EXPECTED_SITE_TREE_SHA256
        or context["site_tree_file_count"] != EXPECTED_SITE_TREE_FILE_COUNT
        or context["site_tree_byte_count"] != EXPECTED_SITE_TREE_BYTE_COUNT
        or context["torch_record_verified_files"]
        != EXPECTED_TORCH_RECORD_VERIFIED_FILES
    ):
        raise RuntimeError("frozen dependency closure receipt mismatch")
    site_packages_path = Path(context["site_packages_path"]).resolve()
    if tuple(sys.path) != context["import_path"] or sys.path.count(
        str(site_packages_path)
    ) != 1:
        raise RuntimeError("explicit no-site import path mismatch")
    if Path(torch.__file__).resolve() != site_packages_path / "torch" / "__init__.py":
        raise RuntimeError("PyTorch was not imported from the frozen site-packages path")
    runtime_dependency_digests = {
        "torch_init_sha256": sha256_file(Path(torch.__file__).resolve()),
        "torch_record_sha256": sha256_file(
            site_packages_path / "torch-2.13.0.dist-info" / "RECORD"
        ),
        "torch_c_sha256": sha256_file(
            site_packages_path / "torch" / "_C.cpython-311-darwin.so"
        ),
        "libtorch_python_sha256": sha256_file(
            site_packages_path / "torch" / "lib" / "libtorch_python.dylib"
        ),
        "libtorch_cpu_sha256": sha256_file(
            site_packages_path / "torch" / "lib" / "libtorch_cpu.dylib"
        ),
    }
    if any(
        runtime_dependency_digests[name] != expected
        for name, expected in dependency_expectations.items()
        if name != "pyvenv_sha256"
    ):
        raise RuntimeError("loaded PyTorch artifact digest mismatch")
    if sys.version.split()[0] != EXPECTED_PYTHON_VERSION:
        raise RuntimeError("frozen Python version mismatch")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError("frozen PyTorch version mismatch")
    if (
        torch.version.git_version != EXPECTED_TORCH_GIT_VERSION
        or sha256_bytes(torch.__config__.show().encode("utf-8"))
        != EXPECTED_TORCH_CONFIG_SHA256
    ):
        raise RuntimeError("frozen PyTorch build mismatch")
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.safe_path
        and sys.flags.no_user_site
        and sys.dont_write_bytecode
    ):
        raise RuntimeError("isolated no-site Python import flags are required")
    provenance_validated = True
    core = load_core_from_frozen_source(core_path, core_source_bytes)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fixture = build_ordering_fixture(core)

    ownership_feature = torch.arange(1, 1 + 3 * 8 * 8, dtype=torch.float64).view(
        1, 3, 8, 8
    ).repeat(2, 1, 1, 1)
    ownership_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    ownership_slot = torch.zeros(2, dtype=torch.long)
    ownership_keys = core.stable_sample_keys((
        "train/pid_0001/camera_01_a.jpg",
        "train/pid_0001/camera_01_b.jpg",
    ))
    ownership_deleted, ownership_support, ownership_realized = (
        core.deterministic_slot_delete(
            ownership_feature, ownership_mask, ownership_keys, ownership_slot,
            fraction=0.5, ordering_seed=SEED,
        )
    )
    swapped_deleted, swapped_support, swapped_realized = (
        core.deterministic_slot_delete(
            ownership_feature, ownership_mask, ownership_keys.flip(0), ownership_slot,
            fraction=0.5, ordering_seed=SEED,
        )
    )
    multi_key_count = 8
    multi_key_feature = ownership_feature[:1].repeat(multi_key_count, 1, 1, 1)
    multi_key_mask = ownership_mask[:1].repeat(multi_key_count, 1, 1)
    multi_keys = core.stable_sample_keys(
        tuple(f"train/hash_owner_{value:02d}.jpg" for value in range(multi_key_count))
    )
    _, multi_key_support, _ = core.deterministic_slot_delete(
        multi_key_feature, multi_key_mask, multi_keys,
        torch.zeros(multi_key_count, dtype=torch.long),
        fraction=0.5, ordering_seed=SEED,
    )
    additive_support_mutant = torch.zeros_like(multi_key_support)
    flat_indices = torch.arange(8 * 8, dtype=torch.int64)
    for row in range(multi_key_count):
        additive_key = torch.remainder(
            flat_indices * 1103515245
            + multi_keys[row] * 12345
            + SEED * 2654435761,
            2147483647,
        )
        additive_order = torch.argsort(additive_key, stable=True)
        additive_support_mutant[row].reshape(-1)[additive_order[:32]] = True
    alias_keys = torch.tensor([1, 1 + 2147483647], dtype=torch.long)
    _, alias_support, _ = core.deterministic_slot_delete(
        ownership_feature, ownership_mask, alias_keys, ownership_slot,
        fraction=0.5, ordering_seed=SEED,
    )

    fill_feature = torch.arange(1, 1 + 2 * 3 * 4 * 5, dtype=torch.float64).view(
        2, 3, 4, 5
    )
    fill_mask = torch.ones(2, 4, 5, dtype=torch.bool)
    fill_keys = core.stable_sample_keys(("train/fill_a.jpg", "train/fill_b.jpg"))
    fill_value = -7.25
    fill_deleted, fill_support, _ = core.deterministic_slot_delete(
        fill_feature, fill_mask, fill_keys, torch.tensor([1, 1]),
        fraction=0.35, fill=fill_value, ordering_seed=SEED,
    )
    fill_expected = fill_feature.clone().permute(0, 2, 3, 1)
    fill_expected[fill_support] = fill_value
    fill_expected = fill_expected.permute(0, 3, 1, 2)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1)
    text = F.normalize(torch.randn(SLOTS, 12, generator=generator), dim=-1)
    visible = F.normalize(text + 0.1, dim=-1)
    occluded = F.normalize(-text + 0.1, dim=-1)
    donor_mix = torch.linspace(0.25, 0.95, 8)[:, None, None]
    recipient_mix = torch.linspace(0.85, 0.35, 8)[:, None, None]
    donor_visual = (
        donor_mix * visible.unsqueeze(0)
        + (1.0 - donor_mix) * occluded.unsqueeze(0)
        + 0.01 * torch.randn(8, SLOTS, 12, generator=generator)
    )
    recipient_visual = (
        recipient_mix * visible.unsqueeze(0)
        + (1.0 - recipient_mix) * occluded.unsqueeze(0)
        + 0.01 * torch.randn(8, SLOTS, 12, generator=generator)
    )
    geometry_valid = torch.ones(8, SLOTS, dtype=torch.bool)
    donor_state = core.clip_slot_state(
        donor_visual, text, visible, occluded, geometry_valid, logit_scale=10.0
    )
    recipient_state = core.clip_slot_state(
        recipient_visual, text, visible, occluded, geometry_valid, logit_scale=10.0
    )
    clip_batch_permutation = torch.arange(8).roll(3)
    clip_inverse_permutation = torch.argsort(clip_batch_permutation)
    donor_batch_permuted = core.clip_slot_state(
        donor_visual[clip_batch_permutation], text, visible, occluded,
        geometry_valid[clip_batch_permutation], logit_scale=10.0,
    )
    donor_single = core.clip_slot_state(
        donor_visual[2:3], text, visible, occluded, geometry_valid[2:3],
        logit_scale=10.0,
    )
    donor_visual_perturbed = donor_visual.clone()
    donor_visual_perturbed[2] = donor_visual_perturbed[2].roll(1, dims=-1)
    donor_perturbed = core.clip_slot_state(
        donor_visual_perturbed, text, visible, occluded, geometry_valid,
        logit_scale=10.0,
    )
    clip_batch_ownership = {
        "batch_permutation": all(torch.equal(
            donor_batch_permuted[name][clip_inverse_permutation], donor_state[name]
        ) for name in ("visual", "distribution", "support", "geometry_valid")),
        "single_sample_numerically_exact": bool(
            torch.equal(donor_single["geometry_valid"][0], donor_state["geometry_valid"][2])
            and all(torch.allclose(
                donor_single[name][0], donor_state[name][2],
                atol=FLOAT32_EQUIVARIANCE_ATOL, rtol=0.0,
            ) for name in ("visual", "distribution", "support"))
        ),
        "single_sample_perturbation_local": bool(
            all(torch.equal(
                donor_perturbed[name][torch.arange(8) != 2],
                donor_state[name][torch.arange(8) != 2],
            ) for name in ("visual", "distribution", "support", "geometry_valid"))
            and not torch.equal(
                donor_perturbed["distribution"][2], donor_state["distribution"][2]
            )
        ),
    }
    slot = torch.arange(8) % SLOTS
    budget = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"],
        donor_state["support"], geometry_valid, geometry_valid, slot,
    )
    recipient_p_cycle = core.semantic_transport_budget(
        recipient_state["distribution"].roll(1, dims=-1), donor_state["distribution"],
        donor_state["support"], geometry_valid, geometry_valid, slot,
    )
    donor_p_cycle = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"].roll(1, dims=-1),
        donor_state["support"], geometry_valid, geometry_valid, slot,
    )
    donor_q_counterfactual = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"],
        1.0 - donor_state["support"], geometry_valid, geometry_valid, slot,
    )
    wrong_rgb_state = {
        "distribution": donor_state["distribution"].flip(0),
        "support": donor_state["support"].flip(0),
    }
    wrong_rgb_budget = core.semantic_transport_budget(
        recipient_state["distribution"], wrong_rgb_state["distribution"],
        wrong_rgb_state["support"], geometry_valid, geometry_valid, slot,
    )
    recipient_support_mutant = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"],
        recipient_state["support"], geometry_valid, geometry_valid, slot,
    )
    ignore_agreement_mutant = donor_state["support"][torch.arange(8), slot]

    perm = torch.tensor([2, 0, 4, 1, 3])
    donor_permuted = core.clip_slot_state(
        donor_visual[:, perm], text[perm], visible[perm], occluded[perm],
        geometry_valid[:, perm], logit_scale=10.0
    )
    recipient_permuted = core.clip_slot_state(
        recipient_visual[:, perm], text[perm], visible[perm], occluded[perm],
        geometry_valid[:, perm], logit_scale=10.0
    )
    permutation_errors = {
        "visual": float(
            (donor_permuted["visual"] - donor_state["visual"][:, perm]).abs().max()
        ),
        "support": float(
            (donor_permuted["support"] - donor_state["support"][:, perm]).abs().max()
        ),
        "distribution": float((
            donor_permuted["distribution"]
            - donor_state["distribution"][:, perm][:, :, perm]
        ).abs().max()),
    }
    original_budget_matrix = torch.stack([
        core.semantic_transport_budget(
            recipient_state["distribution"], donor_state["distribution"],
            donor_state["support"], geometry_valid, geometry_valid,
            torch.full((8,), value, dtype=torch.long),
        ) for value in range(SLOTS)
    ], dim=1)
    permuted_budget_matrix = torch.stack([
        core.semantic_transport_budget(
            recipient_permuted["distribution"], donor_permuted["distribution"],
            donor_permuted["support"], geometry_valid[:, perm], geometry_valid[:, perm],
            torch.full((8,), value, dtype=torch.long),
        ) for value in range(SLOTS)
    ], dim=1)
    permutation_errors["budget"] = float(
        (permuted_budget_matrix - original_budget_matrix[:, perm]).abs().max()
    )
    permutation_error = max(permutation_errors.values())

    def all_slot_budgets(recipient_p, donor_p, support, recipient_valid, donor_valid):
        return torch.stack([
            core.semantic_transport_budget(
                recipient_p, donor_p, support, recipient_valid, donor_valid,
                torch.full((len(recipient_p),), value, dtype=torch.long),
            ) for value in range(SLOTS)
        ], dim=1)

    local_recipient = torch.full((3, SLOTS, SLOTS), 0.05)
    local_donor = torch.full((3, SLOTS, SLOTS), 0.06)
    for value in range(SLOTS):
        local_recipient[:, value, value] += 0.75
        local_donor[:, value, value] += 0.70
    local_support = torch.full((3, SLOTS), 0.8)
    local_valid = torch.ones(3, SLOTS, dtype=torch.bool)
    local_original = all_slot_budgets(
        local_recipient, local_donor, local_support, local_valid, local_valid
    )
    local_row, local_slot = 1, 2
    local_expected = torch.zeros(3, SLOTS, dtype=torch.bool)
    local_expected[local_row, local_slot] = True
    recipient_local = local_recipient.clone()
    recipient_local[local_row, local_slot] = F.one_hot(
        torch.tensor((local_slot + 1) % SLOTS), num_classes=SLOTS
    ).to(recipient_local.dtype)
    donor_local = local_donor.clone()
    donor_local[local_row, local_slot] = F.one_hot(
        torch.tensor((local_slot + 1) % SLOTS), num_classes=SLOTS
    ).to(donor_local.dtype)
    support_local = local_support.clone()
    support_local[local_row, local_slot] = 0.2
    locality_candidates = {
        "recipient_distribution": all_slot_budgets(
            recipient_local, local_donor, local_support, local_valid, local_valid,
        ),
        "donor_distribution": all_slot_budgets(
            local_recipient, donor_local, local_support, local_valid, local_valid,
        ),
        "donor_support": all_slot_budgets(
            local_recipient, local_donor, support_local, local_valid, local_valid,
        ),
    }
    locality_gates = {
        name: bool(
            torch.equal(candidate[~local_expected], local_original[~local_expected])
            and abs(float(candidate[local_row, local_slot] - local_original[local_row, local_slot])) > 1e-3
        )
        for name, candidate in locality_candidates.items()
    }
    locality_changed_gaps = {
        name: abs(float(
            candidate[local_row, local_slot]
            - local_original[local_row, local_slot]
        ))
        for name, candidate in locality_candidates.items()
    }

    invalid_geometry = torch.zeros_like(geometry_valid)
    zero_visual = torch.zeros_like(donor_visual)
    empty_state = core.clip_slot_state(
        zero_visual, text, visible, occluded, invalid_geometry, logit_scale=10.0
    )
    empty_budget = core.semantic_transport_budget(
        empty_state["distribution"], empty_state["distribution"], empty_state["support"],
        invalid_geometry, invalid_geometry, slot,
    )
    recipient_invalid_budget = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"], donor_state["support"],
        invalid_geometry, geometry_valid, slot,
    )
    donor_invalid_budget = core.semantic_transport_budget(
        recipient_state["distribution"], donor_state["distribution"], donor_state["support"],
        geometry_valid, invalid_geometry, slot,
    )
    invalid_scatter = core.scatter_replace(
        fixture["deleted"], fixture["erase_mask"], fixture["correct_state"],
        budget=torch.zeros(
            len(fixture["deleted"]), dtype=fixture["deleted"].dtype
        ),
    )

    end_feature = torch.arange(1, 1 + 12 * 2 * 5, dtype=torch.float32).view(1, 12, 2, 5)
    end_masks = torch.zeros(1, SLOTS, 2, 5)
    end_masks[:, 0, :, 0] = 1.0
    end_masks[:, 2, 0, 2] = 0.25
    end_masks[:, 3, :, 3] = 1.0
    end_masks[:, 4, :, 4] = 1.0
    end_pooled, end_valid, end_mass = core.mass_normalized_pool(
        end_feature, end_masks, min_mass=1.0
    )
    end_state = core.clip_slot_state(
        end_pooled, text, visible, occluded, end_valid, logit_scale=10.0
    )
    end_budgets = torch.stack([
        core.semantic_transport_budget(
            end_state["distribution"], end_state["distribution"], end_state["support"],
            end_valid, end_valid, torch.tensor([value]),
        ) for value in range(SLOTS)
    ], dim=1)
    end_invalid_scatter = core.scatter_replace(
        end_feature, torch.ones(1, 2, 5, dtype=torch.bool),
        end_pooled[:, 0], budget=end_budgets[:, 1],
    )
    empty_selected_mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    empty_deleted, empty_erase_support, empty_realized = (
        core.deterministic_slot_delete(
            end_feature, empty_selected_mask,
            core.stable_sample_keys(("train/empty_geometry.jpg",)),
            torch.tensor([1]), fraction=0.5, ordering_seed=SEED,
        )
    )
    empty_delete_budget_scatter = core.scatter_replace(
        empty_deleted, empty_erase_support, end_pooled[:, 0],
        budget=end_budgets[:, 1],
    )

    sample_count = 200
    images_per_pid = 10
    pids = torch.arange(sample_count) // images_per_pid
    fit, audit = core.pid_disjoint_split(pids, seed=SEED)
    target_identity_latent = torch.randn(SLOTS, 20, 4, generator=generator)
    selected_identity_latent = target_identity_latent[
        torch.arange(sample_count) % SLOTS, pids
    ]
    recipient_slot_maps = torch.randn(SLOTS, 4, 2, generator=generator)
    recipient_probe_slots = torch.einsum(
        "bd,kdc->bkc", selected_identity_latent, recipient_slot_maps
    ) + 0.001 * torch.randn(sample_count, SLOTS, 2, generator=generator)
    probe_target_slot = torch.arange(sample_count) % SLOTS
    probe_x = core.recipient_not_k_features(recipient_probe_slots, probe_target_slot)
    target_slot_mutant = recipient_probe_slots.clone()
    target_slot_mutant[torch.arange(sample_count), probe_target_slot] += 1000.0
    target_slot_mutant_x = core.recipient_not_k_features(
        target_slot_mutant, probe_target_slot
    )
    non_target_slot = (probe_target_slot + 1) % SLOTS
    non_target_mutant = recipient_probe_slots.clone()
    non_target_mutant[torch.arange(sample_count), non_target_slot] += 1000.0
    non_target_mutant_x = core.recipient_not_k_features(
        non_target_mutant, probe_target_slot
    )
    target_maps = torch.randn(SLOTS, 4, 6, generator=generator)
    probe_y = torch.einsum(
        "bd,bdc->bc", selected_identity_latent, target_maps[probe_target_slot]
    ) + 0.001 * torch.randn(sample_count, 6, generator=generator)
    audit_target_slot = probe_target_slot[audit]
    prediction = torch.zeros_like(probe_y[audit])
    per_slot_probe_scores = {}
    for target_value in range(SLOTS):
        fit_slot = fit & (probe_target_slot == target_value)
        audit_slot = audit & (probe_target_slot == target_value)
        audit_output_rows = audit_target_slot == target_value
        prediction[audit_output_rows] = core.ridge_fit_predict(
            probe_x[fit_slot], probe_y[fit_slot], probe_x[audit_slot]
        )
        per_slot_probe_scores[str(target_value)] = core.regression_scores(
            probe_y[audit_slot], prediction[audit_output_rows]
        )
    probe_scores = core.regression_scores(probe_y[audit], prediction)
    zero_scores = core.regression_scores(probe_y[audit], torch.zeros_like(probe_y[audit]))
    fit_pids = set(int(value) for value in pids[fit].tolist())
    audit_pids = set(int(value) for value in pids[audit].tolist())

    privileged_target = torch.randn(sample_count, 3, generator=generator)
    allowed_private_prediction = core.ridge_fit_predict(
        probe_x[fit], privileged_target[fit], probe_x[audit]
    )
    leaked_probe_x = torch.cat((probe_x, privileged_target), dim=1)
    leaked_private_prediction = core.ridge_fit_predict(
        leaked_probe_x[fit], privileged_target[fit], leaked_probe_x[audit]
    )
    allowed_private_scores = core.regression_scores(
        privileged_target[audit], allowed_private_prediction
    )
    leaked_private_scores = core.regression_scores(
        privileged_target[audit], leaked_private_prediction
    )
    column_fixture = torch.tensor(
        [[[10.0], [20.0], [30.0], [40.0], [50.0]]] * 2
    )
    column_fixture_output = core.recipient_not_k_features(
        column_fixture, torch.tensor([0, 1])
    )
    expected_column_fixture = torch.tensor([
        [0.0, 20.0, 30.0, 40.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 30.0, 40.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    recipient_only_schema = {
        "shape_exact": tuple(probe_x.shape) == (
            sample_count, SLOTS * 2 + SLOTS
        ),
        "target_slot_excluded_bitwise": torch.equal(probe_x, target_slot_mutant_x),
        "non_target_slot_retained": not torch.equal(probe_x, non_target_mutant_x),
        "fixed_anatomical_column_ownership": torch.equal(
            column_fixture_output, expected_column_fixture
        ),
        "all_target_slots_fit_and_audited": all(
            bool((fit & (probe_target_slot == value)).any())
            and bool((audit & (probe_target_slot == value)).any())
            for value in range(SLOTS)
        ),
        "explicit_privileged_column_mutant_detectable": (
            leaked_private_scores["r2"] > 0.99
            and allowed_private_scores["r2"] < leaked_private_scores["r2"] - 0.5
        ),
    }

    leakage_x = F.one_hot(pids, num_classes=20).float()
    pid_targets = torch.randn(20, 4, generator=generator)
    leakage_y = pid_targets[pids]
    row_fit = torch.arange(len(pids)) % images_per_pid < images_per_pid // 2
    row_prediction = core.ridge_fit_predict(
        leakage_x[row_fit], leakage_y[row_fit], leakage_x[~row_fit]
    )
    pid_prediction = core.ridge_fit_predict(
        leakage_x[fit], leakage_y[fit], leakage_x[audit]
    )
    row_leakage_scores = core.regression_scores(leakage_y[~row_fit], row_prediction)
    pid_leakage_scores = core.regression_scores(leakage_y[audit], pid_prediction)

    def ranked_vector(score: float, axis: int, orthogonal_axis: int) -> torch.Tensor:
        result = torch.zeros(6)
        result[axis] = float(score)
        result[orthogonal_axis] = math.sqrt(1.0 - float(score) ** 2)
        return result

    query = torch.stack((torch.eye(6)[0], torch.eye(6)[2], torch.eye(6)[4]))
    gallery = torch.stack((
        ranked_vector(1.00, 0, 1),  # PID0 same camera, removed
        ranked_vector(0.99, 0, 1),  # junk, removed
        ranked_vector(0.95, 0, 1),  # q0 negative rank 1
        ranked_vector(0.85, 0, 1),  # q0 positive rank 2
        ranked_vector(0.75, 0, 1),  # q0 negative rank 3
        ranked_vector(0.65, 0, 1),  # q0 positive rank 4
        ranked_vector(1.00, 2, 3),  # PID1 same camera, removed
        ranked_vector(0.96, 2, 3),  # q1 positive rank 1
        ranked_vector(0.86, 2, 3),  # q1 negative rank 2
        ranked_vector(0.76, 2, 3),  # q1 positive rank 3
        ranked_vector(0.66, 2, 3),  # q1 negative rank 4
        ranked_vector(1.00, 4, 5),  # PID6 same camera only: invalid query
    ))
    query_pids = [0, 1, 6]
    query_camids = [0, 0, 0]
    gallery_pids = [0, -1, 2, 0, 3, 0, 1, 1, 4, 1, 5, 6]
    gallery_camids = [0, 9, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0]
    gallery_sample_keys = list(range(100, 100 + len(gallery)))
    metric = core.reid_map_r1(
        query, gallery, query_pids, gallery_pids, query_camids, gallery_camids,
        gallery_sample_keys,
    )
    positive_r1_metric = core.reid_map_r1(
        query[:1],
        torch.stack((ranked_vector(0.9, 0, 1), ranked_vector(0.8, 0, 1))),
        [0], [0, 2], [0], [1, 2], [10, 20],
    )
    tied_query = torch.tensor([[1.0, 0.0]])
    tied_gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    tied_metric = core.reid_map_r1(
        tied_query, tied_gallery, [0], [1, 0], [0], [1, 1], [20, 10]
    )
    tied_permuted_metric = core.reid_map_r1(
        tied_query, tied_gallery.flip(0), [0], [0, 1], [0], [1, 1], [10, 20]
    )
    naive_scores = F.normalize(query[:1].double(), dim=-1) @ F.normalize(gallery.double(), dim=-1).T
    naive_order = torch.argsort(naive_scores[0], descending=True, stable=True)
    naive_matches = (torch.tensor(gallery_pids)[naive_order] == 0).double()
    naive_precision = naive_matches.cumsum(0) / torch.arange(1, len(gallery) + 1, dtype=torch.float64)
    naive_metric = {
        "mAP": float(100.0 * (naive_precision * naive_matches).sum() / naive_matches.sum()),
        "R1": float(100.0 * naive_matches[0]),
    }
    reciprocal_rank_map_mutant = 75.0

    finite_max = torch.finfo(torch.float32).max
    extreme_clip_state = core.clip_slot_state(
        torch.full_like(donor_visual, finite_max), text, visible, occluded,
        geometry_valid, logit_scale=10.0,
    )
    regression_max = torch.finfo(torch.float64).max
    extreme_regression_target = torch.tensor(
        [[regression_max, regression_max], [-regression_max, -regression_max]],
        dtype=torch.float64,
    )
    extreme_regression_scores = core.regression_scores(
        extreme_regression_target, extreme_regression_target.clone()
    )
    extreme_reid_metric = core.reid_map_r1(
        torch.tensor([[regression_max, regression_max]], dtype=torch.float64),
        torch.tensor(
            [[regression_max, regression_max], [regression_max, -regression_max]],
            dtype=torch.float64,
        ),
        [0], [0, 1], [0], [1, 1], [10, 20],
    )
    smallest32 = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    tiny_clip_visual = torch.full_like(donor_visual, smallest32)
    tiny_clip_state = core.clip_slot_state(
        tiny_clip_visual, text, visible, occluded, geometry_valid, logit_scale=10.0
    )
    smallest64 = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    tiny_reid_metric = core.reid_map_r1(
        torch.stack((smallest64.repeat(2),)),
        torch.stack((smallest64.repeat(2), torch.stack((smallest64, -smallest64)))),
        [0], [0, 1], [0], [1, 1], [10, 20],
    )

    utility = fixture["utility"]
    semantic_controls = [
        "same_id_wrong_slot", "wrong_id_same_slot", "generic",
        "NULL", "random_key", "random_cluster", "generic_transport",
    ]
    minimum_margin = min(utility["correct"] - utility[name] for name in semantic_controls)

    # Negative contracts: these deliberately wrong implementations must be caught.
    collapsed_pool = fixture["feature"].mean(dim=(-2, -1))[:, None].expand_as(fixture["pooled"])
    pool_mutant_caught = not torch.allclose(collapsed_pool, fixture["pooled"])
    support_from_geometry = fixture["geometry_valid"].to(donor_state["support"].dtype)
    support_mutant_caught = not torch.allclose(
        support_from_geometry[:8], donor_state["support"], atol=1e-8, rtol=0.0
    )
    budget_mutant_gaps = {
        "recipient_distribution": float((budget - recipient_p_cycle).abs().mean()),
        "donor_distribution": float((budget - donor_p_cycle).abs().mean()),
        "donor_support": float((budget - donor_q_counterfactual).abs().mean()),
        "wrong_rgb_same_slot": float((budget - wrong_rgb_budget).abs().mean()),
        "recipient_support_instead_of_donor": float(
            (budget - recipient_support_mutant).abs().mean()
        ),
        "ignore_distribution_agreement": float(
            (budget - ignore_agreement_mutant).abs().mean()
        ),
    }
    audit_count = int(audit.sum())
    derangement = torch.arange(audit_count).roll(1)
    random_target = probe_y[audit][derangement]
    random_probe = core.regression_scores(random_target, prediction)
    random_probe_mutant_caught = random_probe["r2"] < probe_scores["r2"] - 0.5
    random_probe_deranged = bool((derangement != torch.arange(audit_count)).all())
    formula_mutant_margin = min(
        utility["correct"] - value for value in fixture["mutant_utility"].values()
    )
    random_key_source = fixture["random_key_source"]
    random_key_clean = bool(
        (random_key_source != torch.arange(len(random_key_source))).all()
        and (fixture["pids"][random_key_source] != fixture["pids"]).all()
    )
    correct_norms = fixture["correct_state"].norm(dim=-1).sort().values
    random_key_norms = fixture["random_key_state"].norm(dim=-1).sort().values
    cluster_counts = torch.bincount(fixture["cluster_id"], minlength=8)
    input_guards = {
        "single_pid_split": raises_value_error(
            lambda: core.pid_disjoint_split(torch.zeros(4, dtype=torch.long), seed=SEED)
        ),
        "invalid_fit_fraction": raises_value_error(
            lambda: core.pid_disjoint_split(torch.arange(4), seed=SEED, fit_fraction=1.0)
        ),
        "non_integer_pid": raises_value_error(
            lambda: core.pid_disjoint_split(torch.arange(4).float(), seed=SEED)
        ),
        "invalid_min_mass": raises_value_error(
            lambda: core.mass_normalized_pool(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), min_mass=float("nan")
            )
        ),
        "pool_dtype_mismatch": raises_value_error(
            lambda: core.mass_normalized_pool(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2).double()
            )
        ),
        "invalid_sample_path": raises_value_error(
            lambda: core.stable_sample_keys(["../outside.jpg"])
        ),
        "duplicate_sample_path": raises_value_error(
            lambda: core.stable_sample_keys(["train/a.jpg", "train/a.jpg"])
        ),
        "non_nfc_sample_path": raises_value_error(
            lambda: core.stable_sample_keys(["train/e\u0301.jpg"])
        ),
        "control_character_sample_path": raises_value_error(
            lambda: core.stable_sample_keys(["train/a\n.jpg"])
        ),
        "non_binary_delete_mask": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2), torch.full((1, 2, 2), 0.5),
                torch.tensor([1]), torch.tensor([0]), fraction=0.5
            )
        ),
        "non_finite_delete_fill": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.tensor([1]), torch.tensor([0]),
                fraction=0.5, fill=float("nan")
            )
        ),
        "integer_delete_feature": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2, dtype=torch.long),
                torch.ones(1, 2, 2, dtype=torch.bool),
                torch.tensor([1]), torch.tensor([0]), fraction=0.5
            )
        ),
        "delete_mask_device_mismatch": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2),
                torch.ones(1, 2, 2, dtype=torch.bool, device="meta"),
                torch.tensor([1]), torch.tensor([0]), fraction=0.5
            )
        ),
        "boolean_delete_fraction": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.tensor([1]), torch.tensor([0]), fraction=True
            )
        ),
        "non_integer_delete_seed": raises_value_error(
            lambda: core.deterministic_slot_delete(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.tensor([1]), torch.tensor([0]), fraction=0.5,
                ordering_seed=1.5,
            )
        ),
        "invalid_scatter_mask": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(1, 1, 2, 2), torch.full((1, 2, 2), 2.0), torch.zeros(1, 1)
            )
        ),
        "non_finite_scatter_budget": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, 1),
                budget=torch.tensor([float("nan")])
            )
        ),
        "scatter_dtype_mismatch": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, 1).double(), budget=torch.tensor([0.5])
            )
        ),
        "scatter_budget_not_vector": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(4, 1, 2, 2), torch.ones(4, 2, 2, dtype=torch.bool),
                torch.zeros(4, 1), budget=torch.full((2, 2), 0.5)
            )
        ),
        "scatter_budget_dtype_mismatch": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, 1), budget=torch.tensor([0.5], dtype=torch.float64)
            )
        ),
        "scatter_budget_device_mismatch": raises_value_error(
            lambda: core.scatter_replace(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, 1), budget=torch.empty(1, device="meta")
            )
        ),
        "invalid_logit_scale": raises_value_error(
            lambda: core.clip_slot_state(
                donor_visual, text, visible, occluded, geometry_valid,
                logit_scale=float("nan")
            )
        ),
        "float32_clip_overflow": raises_value_error(
            lambda: core.clip_slot_state(
                torch.full_like(donor_visual.double(), 1e300), text.double(),
                visible.double(), occluded.double(), geometry_valid, logit_scale=10.0
            )
        ),
        "zero_visual_rejected_when_geometry_valid": raises_value_error(
            lambda: core.clip_slot_state(
                torch.zeros_like(donor_visual), text, visible, occluded,
                geometry_valid, logit_scale=10.0
            )
        ),
        "integer_clip_input": raises_value_error(
            lambda: core.clip_slot_state(
                torch.ones_like(donor_visual, dtype=torch.long), text, visible, occluded,
                geometry_valid, logit_scale=10.0
            )
        ),
        "out_of_range_slot": raises_value_error(
            lambda: core.semantic_transport_budget(
                recipient_state["distribution"], donor_state["distribution"],
                donor_state["support"], geometry_valid, geometry_valid,
                torch.full((8,), SLOTS, dtype=torch.long),
            )
        ),
        "invalid_distribution_simplex": raises_value_error(
            lambda: core.semantic_transport_budget(
                recipient_state["distribution"] * 2.0, donor_state["distribution"],
                donor_state["support"], geometry_valid, geometry_valid, slot,
            )
        ),
        "invalid_support_range": raises_value_error(
            lambda: core.semantic_transport_budget(
                recipient_state["distribution"], donor_state["distribution"],
                torch.full_like(donor_state["support"], 2.0),
                geometry_valid, geometry_valid, slot,
            )
        ),
        "integer_semantic_input": raises_value_error(
            lambda: core.semantic_transport_budget(
                torch.ones(1, SLOTS, SLOTS, dtype=torch.long),
                torch.ones(1, SLOTS, SLOTS, dtype=torch.long),
                torch.ones(1, SLOTS, dtype=torch.long),
                torch.ones(1, SLOTS, dtype=torch.bool),
                torch.ones(1, SLOTS, dtype=torch.bool), torch.tensor([0]),
            )
        ),
        "empty_semantic_batch": raises_value_error(
            lambda: core.semantic_transport_budget(
                torch.empty(0, SLOTS, SLOTS), torch.empty(0, SLOTS, SLOTS),
                torch.empty(0, SLOTS), torch.empty(0, SLOTS, dtype=torch.bool),
                torch.empty(0, SLOTS, dtype=torch.bool), torch.empty(0, dtype=torch.long),
            )
        ),
        "integer_context_input": raises_value_error(
            lambda: core.contextual_transport_state(
                torch.ones(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.long),
            )
        ),
        "empty_context_width": raises_value_error(
            lambda: core.contextual_transport_state(
                torch.empty(1, 0), torch.empty(1, 0), torch.empty(1, 0)
            )
        ),
        "zero_norm_reid_descriptor": raises_value_error(
            lambda: core.reid_map_r1(
                torch.zeros(1, 2), torch.ones(1, 2), [0], [0], [0], [1], [10]
            )
        ),
        "non_integer_reid_metadata": raises_value_error(
            lambda: core.reid_map_r1(
                torch.ones(1, 2), torch.ones(1, 2), [0.9], [0], [0], [1], [10]
            )
        ),
        "integer_reid_descriptor": raises_value_error(
            lambda: core.reid_map_r1(
                torch.ones(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.long), [0], [0], [0], [1], [10]
            )
        ),
        "empty_reid_width": raises_value_error(
            lambda: core.reid_map_r1(
                torch.empty(1, 0), torch.empty(1, 0), [0], [0], [0], [1], [10]
            )
        ),
        "duplicate_gallery_sample_key": raises_value_error(
            lambda: core.reid_map_r1(
                torch.ones(1, 2), torch.ones(2, 2), [0], [0, 1], [0], [1, 1],
                [10, 10],
            )
        ),
        "integer_ridge_input": raises_value_error(
            lambda: core.ridge_fit_predict(
                torch.ones(2, 2, dtype=torch.long),
                torch.ones(2, 1, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.long),
            )
        ),
        "empty_ridge_target_width": raises_value_error(
            lambda: core.ridge_fit_predict(
                torch.ones(2, 2), torch.empty(2, 0), torch.ones(1, 2)
            )
        ),
        "integer_regression_input": raises_value_error(
            lambda: core.regression_scores(
                torch.ones(2, 2, dtype=torch.long),
                torch.ones(2, 2, dtype=torch.long),
            )
        ),
        "empty_regression_width": raises_value_error(
            lambda: core.regression_scores(torch.empty(2, 0), torch.empty(2, 0))
        ),
        "boolean_pid_split_seed": raises_value_error(
            lambda: core.pid_disjoint_split(torch.arange(4), seed=True)
        ),
        "string_pid_fit_fraction": raises_value_error(
            lambda: core.pid_disjoint_split(
                torch.arange(4), seed=SEED, fit_fraction="0.6"
            )
        ),
        "boolean_clip_logit_scale": raises_value_error(
            lambda: core.clip_slot_state(
                donor_visual, text, visible, occluded, geometry_valid,
                logit_scale=True,
            )
        ),
        "boolean_pool_min_mass": raises_value_error(
            lambda: core.mass_normalized_pool(
                torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), min_mass=True
            )
        ),
        "boolean_ridge": raises_value_error(
            lambda: core.ridge_fit_predict(
                torch.ones(2, 2), torch.ones(2, 1), torch.ones(1, 2), ridge=True
            )
        ),
    }

    def tensors_are_cpu(value) -> bool:
        if torch.is_tensor(value):
            return value.device.type == "cpu"
        if isinstance(value, dict):
            return all(tensors_are_cpu(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return all(tensors_are_cpu(item) for item in value)
        return True

    all_runtime_tensors_cpu = tensors_are_cpu(dict(locals()))

    gates = {
        "cpu_only": (
            all_runtime_tensors_cpu
            and str(torch.get_default_device()) == "cpu"
            and not torch.cuda.is_initialized()
        ),
        "five_slot_pool_shape": tuple(fixture["pooled"].shape) == (40, SLOTS, 12),
        "all_geometry_valid": bool(fixture["geometry_valid"].all()),
        "slot_masses_positive": bool((fixture["mass"] > 0).all()),
        "primary_deletion_exact_50pct": bool(
            torch.allclose(fixture["realized"], torch.full_like(fixture["realized"], 0.5))
        ),
        "hashed_deletion_same_seed_bitwise_exact": bool(
            torch.equal(fixture["deleted"], fixture["deleted_repeat"])
            and torch.equal(fixture["erase_mask"], fixture["erase_mask_repeat"])
            and torch.equal(fixture["realized"], fixture["realized_repeat"])
        ),
        "hashed_deletion_seed_changes_pixels_not_budget": bool(
            torch.allclose(fixture["realized"], fixture["realized_alt"])
            and not torch.equal(fixture["erase_mask"], fixture["erase_mask_alt"])
            and not torch.equal(fixture["deleted"], fixture["deleted_alt"])
        ),
        "hashed_deletion_batch_permutation_invariant": bool(
            torch.equal(
                fixture["regenerated_permuted_keys"],
                fixture["sample_key"][fixture["batch_permutation"]],
            )
            and torch.equal(
                fixture["deleted_permuted"][fixture["inverse_batch_permutation"]],
                fixture["deleted"],
            )
            and torch.equal(
                fixture["erase_permuted"][fixture["inverse_batch_permutation"]],
                fixture["erase_mask"],
            )
            and torch.equal(
                fixture["realized_permuted"][fixture["inverse_batch_permutation"]],
                fixture["realized"],
            )
        ),
        "hashed_deletion_split_merge_invariant": bool(
            torch.equal(fixture["deleted_split"], fixture["deleted"])
            and torch.equal(fixture["erase_split"], fixture["erase_mask"])
            and torch.equal(fixture["realized_split"], fixture["realized"])
        ),
        "sample_key_owns_deletion_support_and_swaps_exactly": bool(
            not torch.equal(ownership_support[0], ownership_support[1])
            and torch.equal(swapped_support, ownership_support.flip(0))
            and torch.equal(swapped_deleted, ownership_deleted.flip(0))
            and torch.equal(swapped_realized, ownership_realized.flip(0))
        ),
        "joint_hash_avalanche_and_modulus_alias_mutants_caught": bool(
            torch.unique(multi_key_support.flatten(1), dim=0).shape[0]
            == multi_key_count
            and int(
                (multi_key_support != additive_support_mutant).any(dim=(1, 2)).sum()
            ) >= multi_key_count - 1
            and not torch.equal(alias_support[0], alias_support[1])
        ),
        "deletion_values_all_channels_exact": bool(
            torch.equal(fixture["deleted"], fixture["expected_deleted"])
            and torch.equal(fixture["changed_spatial"], fixture["erase_mask"])
            and ((fixture["feature"] - fixture["deleted"]).abs().sum() > 1.0)
        ),
        "nonzero_fill_all_channels_exact": bool(
            torch.equal(fill_deleted, fill_expected)
            and torch.equal(
                fill_deleted.permute(0, 2, 3, 1)[fill_support],
                torch.full((int(fill_support.sum()), fill_feature.shape[1]), fill_value,
                           dtype=fill_feature.dtype),
            )
        ),
        "erase_mask_is_subset_of_selected_slot": bool(
            (fixture["erase_mask"] <= fixture["selected_mask"].bool()).all()
        ),
        "erase_only_scatter_preserves_observed_pixels": torch.equal(
            fixture["erase_only_sentinel"].permute(0, 2, 3, 1)[fixture["observed_mask"]],
            fixture["deleted"].permute(0, 2, 3, 1)[fixture["observed_mask"]],
        ),
        "full_slot_scatter_mutant_changes_observed_pixels": not torch.equal(
            fixture["full_slot_sentinel_mutant"].permute(0, 2, 3, 1)[fixture["observed_mask"]],
            fixture["deleted"].permute(0, 2, 3, 1)[fixture["observed_mask"]],
        ),
        "soft_budget_scatter_matches_elementwise_oracle": bool(
            torch.equal(fixture["soft_scatter"], fixture["soft_expected"])
            and not torch.equal(
                fixture["soft_scatter"], fixture["binary_budget_mutant"]
            )
        ),
        "null_exact_identity": torch.equal(fixture["outputs"]["NULL"], fixture["deleted"]),
        "self_restore_exact_on_constant_slot_fixture": torch.equal(
            fixture["outputs"]["self_restore"], fixture["feature"]
        ),
        "correct_transport_exact_on_additive_fixture": abs(utility["correct"]) < 1e-12,
        "correct_beats_all_semantic_controls": minimum_margin > 1e-3,
        "context_formula_mutants_caught": formula_mutant_margin > 1e-3,
        "slot_permutation_equivariant": permutation_error <= FLOAT32_EQUIVARIANCE_ATOL,
        "clip_state_batch_sample_ownership": all(clip_batch_ownership.values()),
        "clip_budget_sample_specific": float(budget.std(unbiased=False)) > 1e-5,
        "clip_budget_axes_independently_owned": min(budget_mutant_gaps.values()) > 1e-3,
        "clip_budget_sample_slot_local_ownership": all(locality_gates.values()),
        "invalid_geometry_budget_exact_zero": bool(
            torch.equal(empty_budget, torch.zeros_like(empty_budget))
            and torch.equal(recipient_invalid_budget, torch.zeros_like(recipient_invalid_budget))
            and torch.equal(donor_invalid_budget, torch.zeros_like(donor_invalid_budget))
        ),
        "invalid_geometry_scatter_exact_identity": torch.equal(
            invalid_scatter, fixture["deleted"]
        ),
        "pool_to_budget_empty_geometry_end_to_end": bool(
            torch.equal(
                end_valid, torch.tensor([[True, False, False, True, True]])
            )
            and torch.allclose(
                end_mass, torch.tensor([[2.0, 0.0, 0.25, 2.0, 2.0]])
            )
            and torch.equal(
                end_pooled[:, 1:3], torch.zeros_like(end_pooled[:, 1:3])
            )
            and torch.equal(end_budgets[:, 1:3], torch.zeros_like(end_budgets[:, 1:3]))
            and torch.equal(end_invalid_scatter, end_feature)
        ),
        "empty_selected_mask_delete_budget_scatter_identity": bool(
            torch.equal(empty_deleted, end_feature)
            and not bool(empty_erase_support.any())
            and torch.equal(empty_realized, torch.zeros_like(empty_realized))
            and torch.equal(empty_delete_budget_scatter, end_feature)
        ),
        "geometry_not_reused_as_support": support_mutant_caught,
        "pid_split_disjoint": not (fit_pids & audit_pids),
        "pid_split_complete": bool((fit | audit).all() and not (fit & audit).any()),
        "donor_free_probe_beats_zero_cosine": probe_scores["cosine"] > zero_scores["cosine"] + 0.5,
        "donor_free_probe_positive_r2": probe_scores["r2"] > 0.99,
        "donor_free_probe_positive_r2_per_target_slot": min(
            score["r2"] for score in per_slot_probe_scores.values()
        ) > 0.99,
        "recipient_only_not_k_probe_schema": all(recipient_only_schema.values()),
        "pid_disjoint_split_blocks_row_level_leakage": (
            row_leakage_scores["r2"] > 0.99
            and pid_leakage_scores["r2"] < row_leakage_scores["r2"] - 0.5
        ),
        "metric_reports_map_r1_and_valid_query_count": set(metric) == {
            "mAP", "R1", "valid_queries"
        },
        "metric_multi_query_multi_positive_exact": (
            abs(metric["mAP"] - 100.0 * 2.0 / 3.0) < 1e-12
            and abs(metric["R1"] - 50.0) < 1e-12
            and metric["valid_queries"] == 2
        ),
        "metric_positive_r1_minimal_exact": (
            abs(positive_r1_metric["mAP"] - 100.0) < 1e-12
            and abs(positive_r1_metric["R1"] - 100.0) < 1e-12
            and positive_r1_metric["valid_queries"] == 1
        ),
        "metric_exact_ties_use_stable_sample_key": (
            tied_metric == tied_permuted_metric
            and abs(tied_metric["mAP"] - 100.0) < 1e-12
            and abs(tied_metric["R1"] - 100.0) < 1e-12
        ),
        "naive_reporter_mutant_caught": (
            abs(metric["mAP"] - naive_metric["mAP"]) > 1.0
            or abs(metric["R1"] - naive_metric["R1"]) > 1.0
        ),
        "reciprocal_rank_map_mutant_caught": abs(
            metric["mAP"] - reciprocal_rank_map_mutant
        ) > 1.0,
        "random_key_different_pid_no_fixed_point": random_key_clean,
        "random_key_preserves_anatomical_slot": torch.equal(
            fixture["target_slot"][random_key_source], fixture["target_slot"]
        ),
        "random_key_norm_multiset_preserved": torch.allclose(
            correct_norms, random_key_norms, atol=1e-10, rtol=0.0
        ),
        "generic_and_generic_transport_are_distinct": not torch.equal(
            fixture["outputs"]["generic"], fixture["outputs"]["generic_transport"]
        ),
        "random_cluster_frequency_balanced": int(cluster_counts.max() - cluster_counts.min()) <= 1,
        "global_gap_pool_mutant_caught": pool_mutant_caught,
        "random_residual_probe_mutant_caught": (
            random_probe_mutant_caught and random_probe_deranged
        ),
        "input_validation_mutants_caught": all(input_guards.values()),
        "finite_extreme_normalization_and_scores_remain_valid": bool(
            torch.allclose(
                extreme_clip_state["visual"].norm(dim=-1),
                torch.ones_like(extreme_clip_state["support"]),
                atol=1e-6, rtol=0.0,
            )
            and math.isfinite(extreme_regression_scores["cosine"])
            and math.isfinite(extreme_regression_scores["r2"])
            and abs(extreme_regression_scores["cosine"] - 1.0) < 1e-12
            and abs(extreme_regression_scores["r2"] - 1.0) < 1e-12
            and extreme_reid_metric == {
                "mAP": 100.0, "R1": 100.0, "valid_queries": 1
            }
            and torch.allclose(
                tiny_clip_state["visual"].norm(dim=-1),
                torch.ones_like(tiny_clip_state["support"]),
                atol=1e-6, rtol=0.0,
            )
            and tiny_reid_metric == {
                "mAP": 100.0, "R1": 100.0, "valid_queries": 1
            }
        ),
        "frozen_provenance_chain_validated_before_execution": provenance_validated,
        "deterministic_runtime_contract_active": (
            torch.are_deterministic_algorithms_enabled()
            and torch.get_num_threads() == 1
            and torch.get_num_interop_threads() == 1
            and os.environ.get("OMP_NUM_THREADS") == "1"
            and os.environ.get("MKL_NUM_THREADS") == "1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and sys.flags.isolated
            and sys.flags.no_site
            and sys.flags.safe_path
            and sys.flags.no_user_site
            and sys.dont_write_bytecode
            and sys.version.split()[0] == EXPECTED_PYTHON_VERSION
            and torch.__version__ == EXPECTED_TORCH_VERSION
        ),
    }
    payload = {
        "experiment": "exp405",
        "scope": "synthetic CPU/static contract only; no scientific evidence",
        "seed": SEED,
        "provenance": {
            "launcher_mode": context["launcher_mode"],
            "bootstrap_sha256": context["bootstrap_sha256"],
            "contract_sha256": context["contract_sha256"],
            "core_sha256": actual_core_sha256,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_config_sha256": sha256_bytes(
                torch.__config__.show().encode("utf-8")
            ),
            "dependency_artifact_sha256": {
                "pyvenv": context["pyvenv_sha256"],
                **runtime_dependency_digests,
            },
            "site_packages_tree_sha256": context["site_tree_sha256"],
            "site_packages_tree_file_count": context["site_tree_file_count"],
            "site_packages_tree_byte_count": context["site_tree_byte_count"],
            "torch_record_verified_files": context["torch_record_verified_files"],
            "python_isolated": bool(sys.flags.isolated),
            "python_no_site": bool(sys.flags.no_site),
            "python_safe_path": bool(sys.flags.safe_path),
            "python_no_user_site": bool(sys.flags.no_user_site),
            "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "gates": gates,
        "passed": sum(bool(value) for value in gates.values()),
        "total": len(gates),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "ordering_utility": utility,
        "minimum_correct_margin": minimum_margin,
        "formula_mutant_utility": fixture["mutant_utility"],
        "minimum_formula_mutant_margin": formula_mutant_margin,
        "clip_budget_mean": float(budget.mean()),
        "clip_budget_std": float(budget.std(unbiased=False)),
        "clip_budget_mutant_mean_abs_gaps": budget_mutant_gaps,
        "clip_budget_locality_gates": locality_gates,
        "clip_budget_locality_changed_gaps": locality_changed_gaps,
        "clip_batch_ownership": clip_batch_ownership,
        "invalid_geometry_budget": {
            "empty": empty_budget.tolist(),
            "recipient_invalid": recipient_invalid_budget.tolist(),
            "donor_invalid": donor_invalid_budget.tolist(),
        },
        "slot_permutation_max_abs": permutation_errors,
        "slot_permutation_float32_atol": FLOAT32_EQUIVARIANCE_ATOL,
        "donor_free_probe": probe_scores,
        "donor_free_probe_per_target_slot": per_slot_probe_scores,
        "donor_free_probe_target_generation": (
            "independent latent per target-slot/PID with target-specific map; not probe_x @ W"
        ),
        "recipient_only_probe_schema": recipient_only_schema,
        "allowed_private_probe": allowed_private_scores,
        "privileged_column_mutant_probe": leaked_private_scores,
        "zero_probe": zero_scores,
        "random_target_probe": random_probe,
        "row_level_leakage_probe": row_leakage_scores,
        "pid_disjoint_leakage_probe": pid_leakage_scores,
        "toy_retrieval": metric,
        "positive_r1_retrieval": positive_r1_metric,
        "tied_retrieval": tied_metric,
        "tied_permuted_retrieval": tied_permuted_metric,
        "finite_extreme_regression": extreme_regression_scores,
        "finite_extreme_retrieval": extreme_reid_metric,
        "finite_tiny_retrieval": tiny_reid_metric,
        "reciprocal_rank_map_mutant": reciprocal_rank_map_mutant,
        "naive_reporter_mutant": naive_metric,
        "input_guard_mutants": input_guards,
        "cluster_counts": cluster_counts.tolist(),
    }
    return payload
