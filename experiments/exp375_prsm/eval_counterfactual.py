#!/usr/bin/env python3
"""Offline counterfactual evaluation for a frozen exp375 PRSM checkpoint.

All arms reuse one model instance and one strictly loaded checkpoint.  The
only runtime changes are the pose batch supplied to the model and the PRSM
pose-routing control specified in ``ARM_SPECS``.  No optimizer or training
forward is constructed here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg  # noqa: E402
ARM_SPECS = {
    'correct_start': {
        'pose_source': 'input',
        'routing': 'parts',
        'donor_map': False,
    },
    'matched_shuffle': {
        'pose_source': 'input',
        'routing': 'parts',
        'donor_map': True,
    },
    'canonical': {
        'pose_source': 'canonical',
        'routing': 'parts',
        'donor_map': False,
    },
    'foreground_uniform': {
        'pose_source': 'input',
        'routing': 'foreground_uniform',
        'donor_map': False,
    },
    'zero_bypass': {
        'pose_source': 'zero',
        'routing': 'parts',
        'donor_map': False,
    },
    'correct_end': {
        'pose_source': 'input',
        'routing': 'parts',
        'donor_map': False,
    },
}

MATCH_ACCEPTANCE = {
    'mean_cost_over_random_median_max': 0.75,
    'max_dimension_median_abs_z_max': 0.65,
    'zero_write_concordance_min': 1.00,
}

WRITE_NUISANCE_NAMES = (
    'write_mass', 'support_area', 'vertical_centroid', 'vertical_span',
    'horizontal_centroid', 'horizontal_span', 'active_columns', 'zero_write',
    'visibility_max', 'visibility_l2', 'visibility_q25', 'visibility_q50',
    'visibility_q75',
    'head_mass', 'torso_mass', 'left_arm_mass', 'right_arm_mass',
    'left_leg_mass', 'right_leg_mass',
) + tuple('row_mass_%02d' % index for index in range(12)) \
    + tuple('column_mass_%02d' % index for index in range(4))


def _target_heatmaps(pose_dict: Mapping[str, Any]) -> torch.Tensor:
    heatmaps = pose_dict.get('heatmaps')
    if not torch.is_tensor(heatmaps):
        raise ValueError('pose_dict["heatmaps"] must be a tensor')
    if heatmaps.ndim == 5:
        if heatmaps.shape[1] < 1 or heatmaps.shape[2] != 17:
            raise ValueError(
                '5-D heatmaps must have shape [B,P,17,H,W], got %s'
                % (tuple(heatmaps.shape),))
        return heatmaps[:, 0]
    if heatmaps.ndim == 4 and heatmaps.shape[1] == 17:
        return heatmaps
    raise ValueError(
        'heatmaps must have shape [B,P,17,H,W] or [B,17,H,W], got %s'
        % (tuple(heatmaps.shape),))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_split_donor_map(
        query_path: Path, gallery_path: Path,
        query_metadata_path: Path, gallery_metadata_path: Path,
        query_audit_path: Path, gallery_audit_path: Path,
        records: Sequence[Sequence[Any]], num_query: int,
        total: int) -> tuple[np.ndarray, Dict[str, Any]]:
    """Load a frozen donor bijection bound to its original record order."""
    arrays = []
    audit: Dict[str, Any] = {'files': {}, 'record_order': {}}
    split_inputs = (
        ('query', Path(query_path), Path(query_metadata_path),
         Path(query_audit_path), 0, int(num_query)),
        ('gallery', Path(gallery_path), Path(gallery_metadata_path),
         Path(gallery_audit_path), int(num_query), int(total - num_query)),
    )
    for split, path, metadata_path, map_audit_path, offset, count in split_inputs:
        for required in (path, metadata_path, map_audit_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        raw = np.load(path, allow_pickle=False)
        if raw.ndim == 2:
            if raw.shape[0] != 1:
                raise ValueError('%s map must contain exactly one mapping' % split)
            raw = raw[0]
        mapping = np.asarray(raw, dtype=np.int64)
        expected = np.arange(count, dtype=np.int64)
        if mapping.shape != (count,):
            raise ValueError('%s map shape mismatch: %s' % (split, mapping.shape))
        if not np.array_equal(np.sort(mapping), expected):
            raise ValueError('%s donor map is not a bijection' % split)
        if np.any(mapping == expected):
            raise ValueError('%s donor map contains a fixed point' % split)

        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if not isinstance(metadata, list) or len(metadata) != count:
            raise ValueError('%s metadata count mismatch' % split)
        current = records[offset:offset + count]
        order_digest = hashlib.sha256()
        pair_digest = hashlib.sha256()
        for index, (record, frozen) in enumerate(zip(current, metadata)):
            basename = Path(str(record[0])).name
            frozen_basename = Path(str(frozen.get('path', ''))).name
            if (frozen.get('split') != split
                    or int(frozen.get('index', -1)) != index
                    or basename != frozen_basename
                    or int(record[1]) != int(frozen.get('pid', -1))
                    or int(record[2]) != int(frozen.get('camid', -1))):
                raise ValueError(
                    '%s frozen metadata does not bind current record %d'
                    % (split, index))
            row = (index, basename, int(record[1]), int(record[2]))
            order_digest.update(
                json.dumps(row, separators=(',', ':')).encode('utf-8'))
            donor = metadata[int(mapping[index])]
            pair = (
                index, basename, int(record[1]), int(mapping[index]),
                Path(str(donor['path'])).name, int(donor['pid']))
            pair_digest.update(
                json.dumps(pair, separators=(',', ':')).encode('utf-8'))

        frozen_audit = json.loads(
            map_audit_path.read_text(encoding='utf-8'))
        required_keys = {
            'cost_formula_version', 'mapping_audits', 'mapping_seeds',
            'solver', 'effective_unique_count'}
        if not isinstance(frozen_audit, dict) \
                or not required_keys.issubset(frozen_audit):
            raise ValueError('%s mapping audit provenance is incomplete' % split)
        if int(frozen_audit['effective_unique_count']) != 1 \
                or len(frozen_audit['mapping_seeds']) != 1 \
                or len(frozen_audit['mapping_audits']) != 1:
            raise ValueError('%s requires exactly one frozen mapping' % split)
        arrays.append(mapping if split == 'query' else mapping + num_query)
        audit['files'][split] = {
            'path': str(path.resolve()),
            'sha256': _sha256_file(path),
            'metadata_path': str(metadata_path.resolve()),
            'metadata_sha256': _sha256_file(metadata_path),
            'mapping_audit_path': str(map_audit_path.resolve()),
            'mapping_audit_sha256': _sha256_file(map_audit_path),
            'count': count,
        }
        audit['record_order'][split] = {
            'recipient_order_sha256': order_digest.hexdigest(),
            'recipient_to_donor_path_pair_sha256': pair_digest.hexdigest(),
            'matching_seed': frozen_audit['mapping_seeds'][0],
            'matching_cost_formula_version': frozen_audit[
                'cost_formula_version'],
            'solver': frozen_audit['solver'],
        }
    combined = np.concatenate(arrays)
    return combined, audit


class FrozenPoseDonorDataset(torch.utils.data.Dataset):
    """Replace only pose with a pre-frozen, sample-indexed donor package."""

    def __init__(self, base_dataset, donor_indices: np.ndarray,
                 num_query: int):
        self.base_dataset = base_dataset
        self.donor_indices = np.asarray(donor_indices, dtype=np.int64)
        self.num_query = int(num_query)
        if self.donor_indices.shape != (len(base_dataset),):
            raise ValueError('combined donor map length mismatch')
        records = getattr(base_dataset, 'dataset', None)
        if records is None or len(records) != len(base_dataset):
            raise ValueError('base validation dataset lacks stable records')
        self.records = records
        same_pid = []
        cross_split = []
        for index, donor in enumerate(self.donor_indices.tolist()):
            if not 0 <= donor < len(base_dataset):
                raise ValueError('donor index out of bounds')
            if (index < self.num_query) != (donor < self.num_query):
                cross_split.append(index)
            if int(records[index][1]) == int(records[donor][1]):
                same_pid.append(index)
        if cross_split:
            raise ValueError('donor map crosses query/gallery split')
        if same_pid:
            raise ValueError(
                'donor map must use a different PID (%d violations)'
                % len(same_pid))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        recipient = self.base_dataset[index]
        donor_index = int(self.donor_indices[index])
        donor = self.base_dataset[donor_index]
        recipient_pose = recipient[-1]
        donor_pose = dict(donor[-1])
        # These audit-only fields are collated but ignored by the model.
        donor_pose['_exp375_recipient_pose'] = recipient_pose
        donor_pose['_exp375_recipient_index'] = torch.tensor(index)
        donor_pose['_exp375_donor_index'] = torch.tensor(donor_index)
        return recipient[:-1] + (donor_pose,)


def make_frozen_donor_loader(val_loader, donor_indices, num_query):
    """Mirror the formal validation loader with a sample-indexed pose donor."""
    from torch.utils.data import DataLoader
    dataset = FrozenPoseDonorDataset(
        val_loader.dataset, donor_indices, num_query)
    return DataLoader(
        dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_loader.collate_fn,
        pin_memory=bool(getattr(val_loader, 'pin_memory', False)),
        drop_last=False,
    )


def pose_write_nuisance(
        prsm: torch.nn.Module, target_heatmaps: torch.Tensor) -> torch.Tensor:
    """Summarize the exact target-only write support seen by PRSM.

    The signature covers amplitude, support, both spatial axes, six part
    masses, and the full 12-row/4-column mass profiles of the production grid.
    """
    if prsm.routing != 'parts':
        raise ValueError('write nuisance audit requires parts routing')
    batch = int(target_heatmaps.shape[0])
    routes, visibility = prsm._pose_routes(
        target_heatmaps, (12, 4), batch, target_heatmaps.device,
        target_heatmaps.dtype)
    writes = routes.float() * visibility.float()
    total_mass = writes.sum(dim=(1, 2, 3))
    part_mass = writes.sum(dim=(2, 3))
    visible = visibility[:, 0].float()
    active = visible >= 0.05
    support = active.float().sum(dim=(1, 2))
    row_mass = visible.sum(dim=2)
    column_mass = visible.sum(dim=1)
    rows = torch.linspace(
        0.0, 1.0, visible.shape[1], device=visible.device).view(1, -1)
    row_total = row_mass.sum(dim=1).clamp_min(1e-8)
    centroid = (row_mass * rows).sum(dim=1) / row_total
    row_active = active.any(dim=2)
    valid = row_active.any(dim=1)
    first = row_active.float().argmax(dim=1)
    last = row_active.shape[1] - 1 - row_active.flip(1).float().argmax(dim=1)
    span = (last - first).float() / max(row_active.shape[1] - 1, 1)
    span = torch.where(valid, span, torch.zeros_like(span))

    columns = torch.linspace(
        0.0, 1.0, visible.shape[2], device=visible.device).view(1, -1)
    column_total = column_mass.sum(dim=1).clamp_min(1e-8)
    horizontal_centroid = (
        column_mass * columns).sum(dim=1) / column_total
    column_active = active.any(dim=1)
    column_valid = column_active.any(dim=1)
    first_column = column_active.float().argmax(dim=1)
    last_column = column_active.shape[1] - 1 \
        - column_active.flip(1).float().argmax(dim=1)
    horizontal_span = (last_column - first_column).float() \
        / max(column_active.shape[1] - 1, 1)
    horizontal_span = torch.where(
        column_valid, horizontal_span, torch.zeros_like(horizontal_span))
    active_columns = column_active.float().sum(dim=1)
    zero_write = (total_mass <= 1e-8).float()
    flat_visibility = visible.flatten(1)
    visibility_quantiles = torch.quantile(
        flat_visibility, torch.tensor(
            [0.25, 0.50, 0.75], device=visible.device), dim=1).transpose(0, 1)
    result = torch.cat([
        total_mass[:, None], support[:, None], centroid[:, None],
        span[:, None], horizontal_centroid[:, None], horizontal_span[:, None],
        active_columns[:, None], zero_write[:, None],
        flat_visibility.amax(dim=1, keepdim=True),
        flat_visibility.norm(dim=1, keepdim=True), visibility_quantiles,
        part_mass, row_mass, column_mass,
    ], dim=1)
    if result.shape != (batch, len(WRITE_NUISANCE_NAMES)) \
            or not bool(torch.isfinite(result).all()):
        raise RuntimeError('invalid target-only write nuisance audit')
    return result


def _update_digest(digest, value: Any) -> None:
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order='C'))
        return
    for item in value:
        digest.update(str(item).encode('utf-8'))
        digest.update(b'\0')


def constrained_random_bijection(
        pids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic random bijection with no self or same-PID donor."""
    pids = np.asarray(pids, dtype=np.int64)
    count = len(pids)
    rng = np.random.default_rng(int(seed))
    donor = rng.permutation(count)
    order = np.arange(count)
    for _ in range(max(count, 1)):
        conflicts = np.flatnonzero(
            (donor == order) | (pids[donor] == pids))
        if len(conflicts) == 0:
            break
        changed = False
        for left in conflicts.tolist():
            if donor[left] != left and pids[donor[left]] != pids[left]:
                continue
            candidates = rng.permutation(count)
            for right in candidates.tolist():
                if right == left:
                    continue
                left_donor = int(donor[left])
                right_donor = int(donor[right])
                if (right_donor != left
                        and pids[right_donor] != pids[left]
                        and left_donor != right
                        and pids[left_donor] != pids[right]):
                    donor[left], donor[right] = right_donor, left_donor
                    changed = True
                    break
        if not changed:
            raise RuntimeError('could not repair constrained random bijection')
    if (not np.array_equal(np.sort(donor), order)
            or np.any(donor == order)
            or np.any(pids[donor] == pids)):
        raise RuntimeError('invalid constrained random bijection')
    return donor


def matched_nuisance_audit(
        recipient: torch.Tensor, donor: torch.Tensor,
        donor_indices: np.ndarray, records: Sequence[Sequence[Any]],
        num_query: int) -> Dict[str, Any]:
    """Gate a frozen map against target-only PRSM nuisance and random maps."""
    recipient = recipient.float()
    donor = donor.float()
    if recipient.shape != donor.shape \
            or recipient.shape[1] != len(WRITE_NUISANCE_NAMES):
        raise ValueError('matched nuisance matrices have invalid shape')
    expected_donor = recipient[torch.as_tensor(donor_indices)]
    if not torch.allclose(donor, expected_donor, atol=1e-6, rtol=0.0):
        raise RuntimeError('donor pose order does not match frozen donor map')

    median = recipient.median(dim=0).values
    mad = (recipient - median).abs().median(dim=0).values
    scale = 1.4826 * mad
    active = scale >= 1e-8
    if not bool(active.any()):
        raise RuntimeError('all write nuisance dimensions are constant')
    standardized = torch.zeros_like(recipient)
    standardized[:, active] = (
        recipient[:, active] - median[active]) / scale[active]

    mapping = torch.as_tensor(donor_indices, dtype=torch.long)

    def costs_for(candidate):
        candidate = torch.as_tensor(candidate, dtype=torch.long)
        differences = (
            standardized - standardized[candidate]).abs().clamp(max=5.0)
        return differences[:, active].mean(dim=1), differences

    actual_costs, actual_differences = costs_for(mapping)
    random_means = []
    pids = np.asarray([int(record[1]) for record in records], dtype=np.int64)
    for seed in range(475000, 475020):
        split_maps = []
        for start, end in ((0, num_query), (num_query, len(records))):
            if end <= start:
                continue
            local = constrained_random_bijection(pids[start:end], seed)
            split_maps.append(local + start)
        random_map = np.concatenate(split_maps)
        random_costs, _ = costs_for(random_map)
        random_means.append(float(random_costs.mean()))
    random_median = float(np.median(random_means))

    dimension_medians = actual_differences.median(dim=0).values
    max_dimension_median = float(dimension_medians[active].max())
    zero_index = WRITE_NUISANCE_NAMES.index('zero_write')
    zero_concordance = float(
        (recipient[:, zero_index] == donor[:, zero_index]).float().mean())
    mean_cost = float(actual_costs.mean())
    ratio = mean_cost / max(random_median, 1e-12)
    accepted = (
        ratio <= MATCH_ACCEPTANCE['mean_cost_over_random_median_max']
        and max_dimension_median <= MATCH_ACCEPTANCE[
            'max_dimension_median_abs_z_max']
        and zero_concordance >= MATCH_ACCEPTANCE[
            'zero_write_concordance_min'])
    result = {
        'status': 'PASS' if accepted else 'FAIL',
        'acceptance': dict(MATCH_ACCEPTANCE),
        'active_dimensions': int(active.sum()),
        'mean_pair_cost': mean_cost,
        'p95_pair_cost': float(torch.quantile(actual_costs, 0.95)),
        'random_mean_costs': random_means,
        'random_median_mean_cost': random_median,
        'mean_cost_over_random_median': ratio,
        'max_dimension_median_abs_z': max_dimension_median,
        'zero_write_concordance': zero_concordance,
        'per_dimension_median_abs_z': {
            name: float(dimension_medians[index])
            for index, name in enumerate(WRITE_NUISANCE_NAMES)
        },
    }
    if not accepted:
        raise RuntimeError(
            'frozen donor map fails target-only write nuisance gate: %s'
            % json.dumps(result, sort_keys=True))
    return result


@contextlib.contextmanager
def prsm_arm(model: torch.nn.Module, arm: str):
    """Temporarily select an offline PRSM arm without touching parameters."""
    if arm not in ARM_SPECS:
        raise ValueError('unknown arm %r' % (arm,))
    if not getattr(model, 'use_prsm', False):
        raise ValueError('counterfactual evaluation requires POSE_PRSM=True')
    spec = ARM_SPECS[arm]
    old_source = model.prsm_pose_source
    old_routing = model.prsm.routing
    model.prsm_pose_source = str(spec['pose_source'])
    model.prsm.routing = str(spec['routing'])
    try:
        yield
    finally:
        model.prsm_pose_source = old_source
        model.prsm.routing = old_routing


def _unwrap_checkpoint(payload: Any) -> MutableMapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError('checkpoint must be a non-empty state_dict mapping')
    state = dict(payload)
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError('checkpoint state_dict must map strings to tensors')
    has_module = [key.startswith('module.') for key in state]
    if any(has_module) and not all(has_module):
        raise ValueError('checkpoint mixes module-prefixed and plain keys')
    if all(has_module):
        state = {key[len('module.'):]: value for key, value in state.items()}
    return state


def strict_load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> int:
    """Load every checkpoint tensor exactly; missing/unexpected keys are fatal."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = _unwrap_checkpoint(payload)
    model.load_state_dict(state, strict=True)
    return len(state)


def _pose_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _pose_to_device(item, device) for key, item in value.items()}
    return value


def _forward_descriptor(
        model: torch.nn.Module, images: torch.Tensor,
        pose_dict: Mapping[str, Any], camids: torch.Tensor,
        viewids: torch.Tensor, flip_test: bool) -> torch.Tensor:
    descriptor, _ = model(
        images, cam_label=camids, view_label=viewids, pose_dict=pose_dict)
    if not torch.is_tensor(descriptor) or descriptor.ndim != 2:
        raise RuntimeError('exp375 requires one tensor global descriptor')
    if flip_test:
        # Keep heavy project/dataset imports out of utility-only unit tests.
        from utils.flip_test import flip_batch
        flipped_images, flipped_pose = flip_batch(images, pose_dict)
        flipped_descriptor, _ = model(
            flipped_images, cam_label=camids, view_label=viewids,
            pose_dict=flipped_pose)
        descriptor = (descriptor + flipped_descriptor) / 2.0
    if not bool(torch.isfinite(descriptor).all()):
        raise RuntimeError('descriptor contains NaN/Inf')
    return descriptor


def _validate_isolation(model: torch.nn.Module, cfg) -> None:
    from model.pose_backbone_model import PoseBackboneModel
    if type(model) is not PoseBackboneModel:
        raise ValueError('exp375 requires the production PoseBackboneModel')
    if not model.use_prsm or not model.use_target_heatmap:
        raise ValueError('exp375 requires PRSM and target-person heatmaps')
    if model.psg_stage_indices:
        raise ValueError('exp375 counterfactual evaluation requires PSG stages=[]')
    if getattr(cfg.MODEL, 'POSE_TEST_FEAT', None) != 'global':
        raise ValueError('exp375 requires the standard global descriptor')
    incompatible = (
        'POSE_PBSR', 'POSE_LGPA', 'POSE_SKELETON_GCN', 'POSE_PPA',
        'POSE_VCSR', 'POSE_STRUCTURAL_ROUTING', 'POSE_CLIP_ID_PROMPT',
        'POSE_ADDITIVE_ADAPTER', 'POSE_PATCH_EMBED', 'POSE_PROMPT',
    )
    enabled = [name for name in incompatible if bool(getattr(cfg.MODEL, name, False))]
    if enabled:
        raise ValueError('exp375 isolation violated by: ' + ', '.join(enabled))


def evaluate_arm(model, cfg, val_loader, num_query: int, arm: str, device):
    from utils.metrics import R1_mAP_eval
    evaluator = R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
        reranking=False, cfg=cfg)
    evaluator.reset()
    total_images = 0
    path_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    descriptor_digest = hashlib.sha256()
    recipient_nuisance_batches = []
    donor_nuisance_batches = []
    zero_identity_forwards = 0
    zero_identity_failed = False
    flip_test = bool(getattr(cfg.TEST, 'FLIP_TEST', True))

    def zero_identity_hook(_module, inputs, output):
        nonlocal zero_identity_forwards, zero_identity_failed
        if not isinstance(output, tuple) or not torch.equal(inputs[0], output[0]):
            zero_identity_failed = True
        zero_identity_forwards += 1

    zero_handle = None
    if arm == 'zero_bypass':
        zero_handle = model.prsm.register_forward_hook(zero_identity_hook)
    with prsm_arm(model, arm), torch.inference_mode():
        try:
            for batch_data in val_loader:
                (images, pids, camids_eval, camids, viewids,
                 imgpaths, pose_dict) = batch_data
                images = images.to(device)
                camids = camids.to(device)
                viewids = viewids.to(device)
                pose_dict = _pose_to_device(pose_dict, device)
                if arm == 'matched_shuffle':
                    recipient_pose = pose_dict.pop('_exp375_recipient_pose')
                    recipient_indices = pose_dict.pop(
                        '_exp375_recipient_index')
                    donor_indices = pose_dict.pop('_exp375_donor_index')
                    if bool((recipient_indices == donor_indices).any()):
                        raise RuntimeError('matched donor batch contains fixed point')
                    recipient_nuisance = pose_write_nuisance(
                        model.prsm, _target_heatmaps(recipient_pose))
                    donor_nuisance = pose_write_nuisance(
                        model.prsm, _target_heatmaps(pose_dict))
                    recipient_nuisance_batches.append(
                        recipient_nuisance.cpu())
                    donor_nuisance_batches.append(donor_nuisance.cpu())
                descriptor = _forward_descriptor(
                    model, images, pose_dict, camids, viewids, flip_test)
                evaluator.update((descriptor, pids, camids_eval))
                batch = int(images.shape[0])
                total_images += batch
                _update_digest(path_digest, imgpaths)
                _update_digest(label_digest, list(pids) + list(camids_eval))
                _update_digest(descriptor_digest, descriptor)
        finally:
            if zero_handle is not None:
                zero_handle.remove()
    cmc, mean_ap, _, _, _, _, _ = evaluator.compute()
    result = {
        'mAP': float(mean_ap),
        'rank1': float(cmc[0]),
        'rank5': float(cmc[4]),
        'rank10': float(cmc[9]),
        'images': total_images,
        'flip_test': flip_test,
        'path_order_sha256': path_digest.hexdigest(),
        'pid_cam_order_sha256': label_digest.hexdigest(),
        'descriptor_sha256': descriptor_digest.hexdigest(),
    }
    if recipient_nuisance_batches:
        recipient_nuisance = torch.cat(recipient_nuisance_batches, dim=0)
        donor_nuisance = torch.cat(donor_nuisance_batches, dim=0)
        differences = (donor_nuisance - recipient_nuisance).abs()
        result['matched_target_write_nuisance_abs_diff'] = {
            name: {
                'mean': float(differences[:, index].mean()),
                'p95': float(torch.quantile(differences[:, index], 0.95)),
                'max': float(differences[:, index].max()),
            }
            for index, name in enumerate(WRITE_NUISANCE_NAMES)
        }
        dataset = val_loader.dataset
        combined_gate = matched_nuisance_audit(
            recipient_nuisance, donor_nuisance, dataset.donor_indices,
            dataset.records, num_query)
        query_gate = matched_nuisance_audit(
            recipient_nuisance[:num_query], donor_nuisance[:num_query],
            dataset.donor_indices[:num_query], dataset.records[:num_query],
            num_query)
        gallery_gate = matched_nuisance_audit(
            recipient_nuisance[num_query:], donor_nuisance[num_query:],
            dataset.donor_indices[num_query:] - num_query,
            dataset.records[num_query:], len(dataset.records) - num_query)
        result['matched_target_write_nuisance_gate'] = {
            'combined': combined_gate,
            'query': query_gate,
            'gallery': gallery_gate,
        }
    if arm == 'zero_bypass':
        if zero_identity_failed or zero_identity_forwards <= 0:
            raise RuntimeError('zero_bypass failed exact PRSM identity audit')
        result['zero_exact_identity_forwards'] = zero_identity_forwards
    return result


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Evaluate one frozen exp375 checkpoint under pose counterfactuals')
    parser.add_argument(
        '--config-file', type=Path,
        default=ROOT / 'configs/occluded_duke/exp375_p0_prsm.yml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--query-donor-map', type=Path,
        help='frozen query-local fixed-point-free donor bijection (.npy)')
    parser.add_argument(
        '--gallery-donor-map', type=Path,
        help='frozen gallery-local fixed-point-free donor bijection (.npy)')
    parser.add_argument('--query-donor-metadata', type=Path)
    parser.add_argument('--gallery-donor-metadata', type=Path)
    parser.add_argument('--query-mapping-audit', type=Path)
    parser.add_argument('--gallery-mapping-audit', type=Path)
    parser.add_argument(
        '--arms', nargs='+', choices=tuple(ARM_SPECS),
        default=list(ARM_SPECS))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # The evaluation CLI needs the complete training environment; pure pose
    # permutation/checkpoint unit tests intentionally do not.
    from datasets import make_dataloader
    from model import make_model

    args = _parse_args(argv)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(args.config_file))
    cfg.merge_from_list(args.opts)
    # A complete ReID checkpoint is loaded strictly below.  Do not perform an
    # unrelated pretrained-backbone read first (or require that asset to be
    # present on the offline evaluation worker).
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.TEST.WEIGHT = ''
    cfg.freeze()

    seed = int(cfg.SOLVER.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    (_, _, val_loader, num_query, num_classes,
     camera_num, view_num) = make_dataloader(cfg)
    donor_loader = None
    mapping_audit = None
    if 'matched_shuffle' in args.arms:
        required_mapping_inputs = (
            args.query_donor_map, args.gallery_donor_map,
            args.query_donor_metadata, args.gallery_donor_metadata,
            args.query_mapping_audit, args.gallery_mapping_audit)
        if any(value is None for value in required_mapping_inputs):
            raise ValueError(
                'matched_shuffle requires frozen query/gallery map, metadata '
                'and mapping-audit files; batch-local shuffle is forbidden')
        donor_indices, mapping_audit = load_split_donor_map(
            args.query_donor_map, args.gallery_donor_map,
            args.query_donor_metadata, args.gallery_donor_metadata,
            args.query_mapping_audit, args.gallery_mapping_audit,
            val_loader.dataset.dataset, num_query, len(val_loader.dataset))
        donor_loader = make_frozen_donor_loader(
            val_loader, donor_indices, num_query)
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=float(cfg.MODEL.SEMANTIC_WEIGHT))
    _validate_isolation(model, cfg)
    loaded_tensors = strict_load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    results = {}
    for arm in args.arms:
        arm_loader = donor_loader if ARM_SPECS[arm]['donor_map'] \
            else val_loader
        results[arm] = evaluate_arm(
            model, cfg, arm_loader, num_query, arm, device)
        row = results[arm]
        print('%-19s mAP %.4f R1 %.4f R5 %.4f R10 %.4f' % (
            arm, row['mAP'], row['rank1'], row['rank5'], row['rank10']),
              flush=True)

    reference_paths = {
        row['path_order_sha256'] for row in results.values()
    }
    reference_labels = {
        row['pid_cam_order_sha256'] for row in results.values()
    }
    if len(reference_paths) != 1 or len(reference_labels) != 1:
        raise RuntimeError('RGB/path/PID/camera order changed across arms')
    if 'correct_start' in results and 'correct_end' in results:
        start = results['correct_start']
        end = results['correct_end']
        exact_keys = (
            'mAP', 'rank1', 'rank5', 'rank10', 'descriptor_sha256')
        if any(start[key] != end[key] for key in exact_keys):
            raise RuntimeError('correct-start/end reproducibility audit failed')

    output = {
        'config': str(args.config_file.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': _sha256_file(args.checkpoint),
        'loaded_tensors': loaded_tensors,
        'mapping_audit': mapping_audit,
        'arms': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
