#!/usr/bin/env python3
"""Read-only semantic audit for the frozen exp378 D0 epoch-90 checkpoint.

The evaluator reuses one strictly loaded model for every arm.  It never
constructs an optimizer and never changes model parameters.  Counterfactuals
are applied with short-lived forward hooks at exactly two production seams:
the TAPF output field and the Stage-3 PoseSpatialGate modules.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg  # noqa: E402
from experiments.exp375_prsm.eval_counterfactual import (  # noqa: E402
    _pose_to_device,
    _sha256_file,
    _update_digest,
    load_split_donor_map,
)


EXPECTED_CONFIG_SHA256 = (
    'a4a184c178e69e8be4f91b4fa480d9eff7e5d5f19f0cadf55636bd8d4367497e')
EXPECTED_CHECKPOINT_SHA256 = (
    'c5407d30d145b92c1995b137ea917187bfb5c1e7c04cd662a44362ae68b4c253')
EXPECTED_CORRECT_PERCENT = (56.3, 67.6, 79.8, 83.5)
EXPECTED_MAPPING_SHA256 = {
    'query_map':
        '421e1a179fcf275e4225e6f72d7d10fff196134674e596842a6dc92569ed47e7',
    'gallery_map':
        '2403af852fe9c55340a6d265e1ef3c4a0215809e7f72fdc03399d8573e3353ad',
    'query_metadata':
        '15fc8de5d53a50274c64896c20bb734df342d26e5ec9f3586d2cc1ad09e5f433',
    'gallery_metadata':
        'c36f271cd4a577db1da3d0a4c07d4a5b57e6e42cba1633f507cfd2825d141b58',
    'query_audit':
        '12c63357b22eac134391c52aacba8b1e1a51d3988b371b6c9fe184dd60ac9461',
    'gallery_audit':
        '92839181b22f43966c4608184ddc8546dfb6b89cdeffd133bb6681cac6376d56',
}

JOINT_PERMUTATION = tuple(range(1, 17)) + (0,)
FLIP_JOINT_SWAP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9,
                   12, 11, 14, 13, 16, 15)

ARM_SPECS = {
    'correct_start': {'loader': 'recipient', 'kind': 'correct'},
    'external_correct': {'loader': 'recipient', 'kind': 'external'},
    'external_shuffle': {'loader': 'recipient', 'kind': 'external'},
    'external_none': {'loader': 'recipient', 'kind': 'external'},
    'external_unindexable': {'loader': 'recipient', 'kind': 'external'},
    'matched_wrong_field': {'loader': 'donor', 'kind': 'matched'},
    'joint_permutation': {'loader': 'recipient', 'kind': 'field'},
    'confidence_permutation': {'loader': 'recipient', 'kind': 'field'},
    'spatial_constant': {'loader': 'recipient', 'kind': 'field'},
    'zero_field': {'loader': 'recipient', 'kind': 'field'},
    'psg_bypass': {'loader': 'recipient', 'kind': 'psg'},
    'correct_end': {'loader': 'recipient', 'kind': 'correct'},
}

EXTERNAL_PARITY_ARMS = (
    'external_correct', 'external_shuffle',
    'external_none', 'external_unindexable')
FIELD_ARMS = (
    'joint_permutation', 'confidence_permutation',
    'spatial_constant', 'zero_field')


class UnindexablePose:
    """Sentinel that fails loudly if any external-pose operation is attempted."""

    def __getitem__(self, key):
        raise RuntimeError('external pose sentinel was indexed: %r' % (key,))

    def __iter__(self):
        raise RuntimeError('external pose sentinel was iterated')

    def items(self):
        raise RuntimeError('external pose sentinel .items() was called')

    def get(self, key, default=None):
        raise RuntimeError('external pose sentinel .get() was called: %r' % key)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        _update_digest(digest, value)
    return digest.hexdigest()


def _unwrap_checkpoint(payload: Any) -> MutableMapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError('checkpoint must contain a non-empty state_dict')
    state = dict(payload)
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError('checkpoint state_dict must map strings to tensors')
    prefixes = [key.startswith('module.') for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError('checkpoint mixes module-prefixed and plain keys')
    if all(prefixes):
        state = {key[len('module.'):]: value for key, value in state.items()}
    return state


def strict_load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> int:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(
            checkpoint, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch 1.13 compatibility.
        payload = torch.load(checkpoint, map_location='cpu')
    state = _unwrap_checkpoint(payload)
    model.load_state_dict(state, strict=True)
    return len(state)


def _batch_roll(value: Any, shift: int = 1) -> Any:
    """Deterministically roll every batch-aligned tensor without using RNG."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value.roll(shifts=shift, dims=0)
    if isinstance(value, Mapping):
        return {key: _batch_roll(item, shift) for key, item in value.items()}
    return value


def external_pose_for_arm(arm: str, pose_dict: Mapping[str, Any] | None):
    if arm in ('correct_start', 'correct_end', 'external_correct'):
        return pose_dict
    if arm == 'external_shuffle':
        if pose_dict is None:
            raise ValueError('external_shuffle requires a real pose batch')
        heatmaps = pose_dict.get('heatmaps')
        if not torch.is_tensor(heatmaps) or heatmaps.shape[0] <= 1:
            raise ValueError(
                'external_shuffle requires batch size > 1 for derangement')
        return _batch_roll(pose_dict)
    if arm == 'external_none':
        return None
    if arm == 'external_unindexable':
        return UnindexablePose()
    return pose_dict


def _validate_field(field: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(field):
        raise TypeError('TAPF field must be a tensor')
    if field.ndim != 4 or field.shape[1:] != (17, 96, 32):
        raise ValueError(
            'TAPF field must be Bx17x96x32, got %s' % (tuple(field.shape),))
    if field.dtype != torch.float32:
        raise TypeError('TAPF audit requires float32 field, got %s' % field.dtype)
    if not field.is_contiguous():
        raise ValueError('TAPF field must be contiguous')
    if not bool(torch.isfinite(field).all()):
        raise FloatingPointError('TAPF field contains NaN/Inf')
    return field


def transform_field(field: torch.Tensor, arm: str) -> torch.Tensor:
    """Apply one pre-registered field intervention."""
    field = _validate_field(field)
    permutation = torch.as_tensor(
        JOINT_PERMUTATION, device=field.device, dtype=torch.long)
    if arm == 'joint_permutation':
        result = field.index_select(1, permutation)
    elif arm == 'confidence_permutation':
        if bool((field < 0.0).any()):
            raise ValueError('confidence permutation requires nonnegative field')
        peak = field.flatten(2).amax(dim=-1)
        shape = field / peak.clamp_min(1e-12)[:, :, None, None]
        shape = torch.where(
            (peak > 0.0)[:, :, None, None], shape, torch.zeros_like(shape))
        result = shape * peak.index_select(1, permutation)[:, :, None, None]
    elif arm == 'spatial_constant':
        result = field.mean(dim=(-2, -1), keepdim=True).expand_as(field)
    elif arm == 'zero_field':
        result = torch.zeros_like(field)
    else:
        raise ValueError('unknown field intervention %r' % arm)
    return _validate_field(result.contiguous())


class TapfFieldHook:
    """Short-lived TAPF output hook with auditable field replacement."""

    def __init__(self, tapf: torch.nn.Module,
                 transform: Callable[[torch.Tensor], torch.Tensor]):
        self.tapf = tapf
        self.transform = transform
        self.handle = None
        self.calls = 0
        self.changed_calls = 0
        self.max_abs_delta = 0.0
        self.input_digest = hashlib.sha256()
        self.output_digest = hashlib.sha256()

    def _hook(self, _module, _inputs, output):
        if (not isinstance(output, tuple) or len(output) != 2
                or not torch.is_tensor(output[0])):
            raise RuntimeError('unexpected TAPF output contract')
        before = _validate_field(output[0])
        after = _validate_field(self.transform(before))
        if after.shape != before.shape or after.device != before.device:
            raise RuntimeError('field intervention changed shape/device')
        delta = float((after - before).abs().amax())
        self.calls += 1
        self.changed_calls += int(delta > 0.0)
        self.max_abs_delta = max(self.max_abs_delta, delta)
        _update_digest(self.input_digest, before)
        _update_digest(self.output_digest, after)
        return after, output[1]

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError('TAPF hook cannot be entered twice')
        self.handle = self.tapf.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        self.handle = None
        return False

    def summary(self):
        return {
            'calls': self.calls,
            'changed_calls': self.changed_calls,
            'max_abs_delta': self.max_abs_delta,
            'input_field_sha256': self.input_digest.hexdigest(),
            'output_field_sha256': self.output_digest.hexdigest(),
            'hook_removed': self.handle is None,
        }


@contextlib.contextmanager
def capture_tapf_field(tapf: torch.nn.Module):
    captured = []

    def hook(_module, _inputs, output):
        field = _validate_field(output[0])
        captured.append(field.detach().clone())
        return output

    handle = tapf.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


class PsgBypassHooks:
    """Temporarily turn every configured PSG block into exact identity."""

    def __init__(self, modules: Mapping[str, torch.nn.Module]):
        self.modules = modules
        self.handles = []
        self.calls = {str(key): 0 for key in modules}

    def __enter__(self):
        if self.handles:
            raise RuntimeError('PSG bypass cannot be entered twice')
        for key, module in self.modules.items():
            key = str(key)

            def bypass(_module, inputs, _output, name=key):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise RuntimeError('unexpected PSG input contract')
                self.calls[name] += 1
                return inputs[0]

            self.handles.append(module.register_forward_hook(bypass))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def summary(self):
        return {
            'calls': dict(self.calls),
            'all_called': bool(self.calls) and all(
                count > 0 for count in self.calls.values()),
            'hooks_removed': not self.handles,
        }


class SemanticAccumulator:
    """Streaming teacher agreement, flip equivariance and occupancy metrics."""

    def __init__(self):
        self.valid_joints = 0
        self.coord_error_sum = 0.0
        self.pck_hits = 0
        self.posterior_cos_sum = 0.0
        self.conf_count = 0
        self.conf_abs_sum = 0.0
        self.conf_sq_sum = 0.0
        self.corr = [0.0] * 5  # sum_x, sum_y, sum_x2, sum_y2, sum_xy
        self.flip_count = 0
        self.flip_posterior_cos_sum = 0.0
        self.flip_coord_error_sum = 0.0
        self.flip_field_abs_sum = 0.0
        self.flip_field_sq_sum = 0.0
        self.flip_field_values = 0
        self.channel_samples = 0
        self.channel_confidence_sum = torch.zeros(17, dtype=torch.float64)
        self.channel_peak_sum = torch.zeros(17, dtype=torch.float64)
        self.channel_mean_sum = torch.zeros(17, dtype=torch.float64)
        self.winner_counts = torch.zeros(17, dtype=torch.int64)

    @staticmethod
    def _coordinates(probability: torch.Tensor):
        height, width = probability.shape[-2:]
        ys = torch.linspace(
            0.0, 1.0, height, device=probability.device,
            dtype=probability.dtype).view(1, 1, height, 1)
        xs = torch.linspace(
            0.0, 1.0, width, device=probability.device,
            dtype=probability.dtype).view(1, 1, 1, width)
        return ((probability * xs).sum(dim=(-2, -1)),
                (probability * ys).sum(dim=(-2, -1)))

    def update_teacher(self, student_probability, student_confidence,
                       teacher_probability, teacher_confidence):
        valid = teacher_confidence >= 0.3
        sx, sy = self._coordinates(student_probability)
        tx, ty = self._coordinates(teacher_probability)
        error = torch.sqrt((sx - tx).square() + (sy - ty).square())
        cosine = torch.nn.functional.cosine_similarity(
            student_probability.flatten(2), teacher_probability.flatten(2),
            dim=-1, eps=1e-12)
        self.valid_joints += int(valid.sum())
        self.coord_error_sum += float(error[valid].sum())
        self.pck_hits += int((error[valid] <= 0.05).sum())
        self.posterior_cos_sum += float(cosine[valid].sum())

        x = student_confidence.double().flatten()
        y = teacher_confidence.double().flatten()
        self.conf_count += x.numel()
        self.conf_abs_sum += float((x - y).abs().sum())
        self.conf_sq_sum += float((x - y).square().sum())
        self.corr[0] += float(x.sum())
        self.corr[1] += float(y.sum())
        self.corr[2] += float(x.square().sum())
        self.corr[3] += float(y.square().sum())
        self.corr[4] += float((x * y).sum())

    def update_field(self, field, confidence):
        batch = field.shape[0]
        self.channel_samples += batch
        self.channel_confidence_sum += confidence.detach().double().sum(0).cpu()
        self.channel_peak_sum += field.detach().double().flatten(2).amax(-1).sum(0).cpu()
        self.channel_mean_sum += field.detach().double().mean((-2, -1)).sum(0).cpu()
        winners = field.detach().argmax(dim=1).flatten()
        self.winner_counts += torch.bincount(
            winners.cpu(), minlength=17).to(torch.int64)

    def update_flip(self, original_probability, flipped_probability,
                    original_field, flipped_field):
        swap = torch.as_tensor(
            FLIP_JOINT_SWAP, device=flipped_probability.device,
            dtype=torch.long)
        aligned_probability = flipped_probability.flip(-1).index_select(1, swap)
        aligned_field = flipped_field.flip(-1).index_select(1, swap)
        cosine = torch.nn.functional.cosine_similarity(
            original_probability.flatten(2), aligned_probability.flatten(2),
            dim=-1, eps=1e-12)
        ox, oy = self._coordinates(original_probability)
        fx, fy = self._coordinates(aligned_probability)
        error = torch.sqrt((ox - fx).square() + (oy - fy).square())
        delta = (original_field - aligned_field).double()
        self.flip_count += cosine.numel()
        self.flip_posterior_cos_sum += float(cosine.sum())
        self.flip_coord_error_sum += float(error.sum())
        self.flip_field_abs_sum += float(delta.abs().sum())
        self.flip_field_sq_sum += float(delta.square().sum())
        self.flip_field_values += delta.numel()

    def summary(self):
        corr = None
        n = self.conf_count
        if n:
            sx, sy, sx2, sy2, sxy = self.corr
            numerator = n * sxy - sx * sy
            denominator = math.sqrt(max(n * sx2 - sx * sx, 0.0)
                                    * max(n * sy2 - sy * sy, 0.0))
            if denominator > 0.0:
                corr = numerator / denominator
        winner_total = int(self.winner_counts.sum())
        occupancy = self.winner_counts.double() / max(winner_total, 1)
        nonzero = occupancy > 0
        entropy = float(-(occupancy[nonzero]
                          * occupancy[nonzero].log()).sum())
        samples = max(self.channel_samples, 1)
        return {
            'teacher_agreement': {
                'teacher_confidence_threshold': 0.3,
                'valid_joint_count': self.valid_joints,
                'normalized_coordinate_error_mean': (
                    self.coord_error_sum / max(self.valid_joints, 1)),
                'pseudo_pck_at_0_05': self.pck_hits / max(self.valid_joints, 1),
                'posterior_cosine_mean': (
                    self.posterior_cos_sum / max(self.valid_joints, 1)),
                'confidence_mae': self.conf_abs_sum / max(n, 1),
                'confidence_brier': self.conf_sq_sum / max(n, 1),
                'confidence_correlation': corr,
            },
            'flip_equivariance': {
                'joint_count': self.flip_count,
                'posterior_cosine_mean': (
                    self.flip_posterior_cos_sum / max(self.flip_count, 1)),
                'normalized_coordinate_error_mean': (
                    self.flip_coord_error_sum / max(self.flip_count, 1)),
                'field_mae': (
                    self.flip_field_abs_sum / max(self.flip_field_values, 1)),
                'field_rmse': math.sqrt(
                    self.flip_field_sq_sum / max(self.flip_field_values, 1)),
            },
            'channel_occupancy': {
                'sample_count': self.channel_samples,
                'confidence_mean': (
                    self.channel_confidence_sum / samples).tolist(),
                'field_peak_mean': (
                    self.channel_peak_sum / samples).tolist(),
                'field_spatial_mean': (
                    self.channel_mean_sum / samples).tolist(),
                'winner_counts': self.winner_counts.tolist(),
                'winner_fraction': occupancy.tolist(),
                'winner_entropy_nats': entropy,
                'winner_entropy_normalized': entropy / math.log(17.0),
                'effective_winner_channels': int(nonzero.sum()),
            },
        }


@contextlib.contextmanager
def tapf_semantic_probe(tapf: torch.nn.Module):
    """Capture one batch's student posterior/confidence and output field."""
    records = []

    def pre_hook(module, inputs):
        probability, confidence, _ = module._student_posterior(inputs[0])
        records.append({
            'probability': probability.detach(),
            'confidence': confidence.detach(),
        })

    def post_hook(_module, _inputs, output):
        if not records or 'field' in records[-1]:
            raise RuntimeError('TAPF semantic probe call ordering failed')
        records[-1]['field'] = _validate_field(output[0]).detach()
        return output

    pre_handle = tapf.register_forward_pre_hook(pre_hook)
    post_handle = tapf.register_forward_hook(post_hook)
    try:
        yield records
    finally:
        post_handle.remove()
        pre_handle.remove()


class FrozenRgbDonorDataset(torch.utils.data.Dataset):
    """Attach frozen donor RGB/camera/view to the unchanged recipient item."""

    def __init__(self, base_dataset, donor_indices: np.ndarray, num_query: int):
        self.base_dataset = base_dataset
        self.donor_indices = np.asarray(donor_indices, dtype=np.int64)
        self.num_query = int(num_query)
        self.records = getattr(base_dataset, 'dataset', None)
        if self.records is None or len(self.records) != len(base_dataset):
            raise ValueError('validation dataset lacks stable records')
        if self.donor_indices.shape != (len(base_dataset),):
            raise ValueError('combined donor map length mismatch')
        for index, donor in enumerate(self.donor_indices.tolist()):
            if donor == index:
                raise ValueError('donor map contains a fixed point')
            if not 0 <= donor < len(base_dataset):
                raise ValueError('donor index out of bounds')
            if (index < self.num_query) != (donor < self.num_query):
                raise ValueError('donor map crosses query/gallery split')
            if int(self.records[index][1]) == int(self.records[donor][1]):
                raise ValueError('donor map must use a different PID')

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _image(item):
        image = item[0]
        if isinstance(image, tuple):
            if len(image) != 1:
                raise ValueError('validation donor must contain one image view')
            image = image[0]
        if not torch.is_tensor(image):
            raise TypeError('donor image must be a tensor')
        return image

    def __getitem__(self, index):
        recipient = self.base_dataset[index]
        donor_index = int(self.donor_indices[index])
        donor = self.base_dataset[donor_index]
        pose = dict(recipient[-1])
        pose['_exp378_donor_image'] = self._image(donor)
        pose['_exp378_donor_camid'] = torch.tensor(int(donor[2]))
        pose['_exp378_donor_viewid'] = torch.tensor(int(donor[3]))
        pose['_exp378_recipient_index'] = torch.tensor(index)
        pose['_exp378_donor_index'] = torch.tensor(donor_index)
        return recipient[:-1] + (pose,)


def make_frozen_rgb_donor_loader(val_loader, donor_indices, num_query):
    from torch.utils.data import DataLoader
    dataset = FrozenRgbDonorDataset(
        val_loader.dataset, donor_indices, num_query)
    return DataLoader(
        dataset, batch_size=val_loader.batch_size, shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_loader.collate_fn,
        pin_memory=bool(getattr(val_loader, 'pin_memory', False)),
        drop_last=False)


def _pop_donor_payload(pose_dict: Mapping[str, Any]):
    pose = dict(pose_dict)
    required = (
        '_exp378_donor_image', '_exp378_donor_camid',
        '_exp378_donor_viewid', '_exp378_recipient_index',
        '_exp378_donor_index')
    missing = [key for key in required if key not in pose]
    if missing:
        raise ValueError('matched arm lacks donor payload: %s' % missing)
    payload = {key: pose.pop(key) for key in required}
    if bool((payload['_exp378_recipient_index']
             == payload['_exp378_donor_index']).any()):
        raise RuntimeError('matched donor batch contains a fixed point')
    return pose, payload


def _forward_once(model, images, pose_dict, camids, viewids):
    descriptor, _ = model(
        images, cam_label=camids, view_label=viewids, pose_dict=pose_dict)
    if (not torch.is_tensor(descriptor) or descriptor.ndim != 2
            or not bool(torch.isfinite(descriptor).all())):
        raise RuntimeError('invalid/non-finite global descriptor')
    return descriptor


def _forward_regular(model, images, pose_dict, camids, viewids, flip_test):
    descriptor = _forward_once(model, images, pose_dict, camids, viewids)
    if flip_test:
        flipped = images.flip(-1)
        flipped_descriptor = _forward_once(
            model, flipped, pose_dict, camids, viewids)
        descriptor = 0.5 * (descriptor + flipped_descriptor)
    return descriptor


def _forward_matched(model, images, pose_dict, camids, viewids,
                     donor_images, donor_camids, donor_viewids, flip_test):
    audits = []

    def one_orientation(recipient_rgb, donor_rgb):
        with capture_tapf_field(model.tapf) as captured:
            _forward_once(
                model, donor_rgb, None, donor_camids, donor_viewids)
        if len(captured) != 1:
            raise RuntimeError('donor TAPF capture count mismatch')
        hook = TapfFieldHook(model.tapf, lambda _field: captured[0])
        with hook:
            descriptor = _forward_once(
                model, recipient_rgb, pose_dict, camids, viewids)
        audits.append(hook.summary())
        return descriptor

    descriptor = one_orientation(images, donor_images)
    if flip_test:
        flipped = one_orientation(images.flip(-1), donor_images.flip(-1))
        descriptor = 0.5 * (descriptor + flipped)
    return descriptor, audits


def _target_teacher(pose_dict: Mapping[str, Any]):
    heatmaps = pose_dict['heatmaps']
    scores = pose_dict['scores']
    mask = pose_dict['person_mask'][:, 0].float()
    target_heatmaps = heatmaps[:, 0].float() * mask[:, None, None, None]
    target_scores = scores[:, 0].float() * mask[:, None]
    return target_heatmaps, target_scores


def evaluate_arm(model, cfg, loader, num_query: int, arm: str,
                 device: torch.device, max_batches: int | None = None):
    from utils.metrics import R1_mAP_eval
    formal_metrics = max_batches is None
    evaluator = None
    if formal_metrics:
        evaluator = R1_mAP_eval(
            num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
            reranking=False, cfg=cfg)
        evaluator.reset()
    descriptor_digest = hashlib.sha256()
    path_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    total = 0
    batches = 0
    flip_test = bool(getattr(cfg.TEST, 'FLIP_TEST', True))
    field_hook = None
    psg_hooks = None
    matched_audits = []
    semantics = SemanticAccumulator() if arm == 'correct_start' else None

    if arm in FIELD_ARMS:
        field_hook = TapfFieldHook(
            model.tapf, lambda field: transform_field(field, arm))
        arm_context = field_hook
    elif arm == 'psg_bypass':
        psg_hooks = PsgBypassHooks(model.psg_modules_dict)
        arm_context = psg_hooks
    else:
        arm_context = contextlib.nullcontext()

    with arm_context, torch.inference_mode():
        for batch_index, batch_data in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            (images, pids, camids_eval, camids, viewids,
             imgpaths, pose_dict) = batch_data
            images = images.to(device)
            camids = camids.to(device)
            viewids = viewids.to(device)
            pose_dict = _pose_to_device(pose_dict, device)

            if arm == 'matched_wrong_field':
                recipient_pose, donor = _pop_donor_payload(pose_dict)
                descriptor, batch_audits = _forward_matched(
                    model, images, recipient_pose, camids, viewids,
                    donor['_exp378_donor_image'],
                    donor['_exp378_donor_camid'],
                    donor['_exp378_donor_viewid'], flip_test)
                matched_audits.extend(batch_audits)
            elif semantics is not None:
                with tapf_semantic_probe(model.tapf) as records:
                    descriptor = _forward_regular(
                        model, images, pose_dict, camids, viewids, flip_test)
                expected = 2 if flip_test else 1
                if len(records) != expected:
                    raise RuntimeError('semantic TAPF probe count mismatch')
                teacher_heatmaps, teacher_scores = _target_teacher(pose_dict)
                teacher_probability, teacher_confidence = (
                    model.tapf._teacher_posterior(
                        teacher_heatmaps, teacher_scores))
                semantics.update_teacher(
                    records[0]['probability'], records[0]['confidence'],
                    teacher_probability, teacher_confidence)
                semantics.update_field(
                    records[0]['field'], records[0]['confidence'])
                if flip_test:
                    semantics.update_flip(
                        records[0]['probability'], records[1]['probability'],
                        records[0]['field'], records[1]['field'])
            else:
                external_pose = external_pose_for_arm(arm, pose_dict)
                descriptor = _forward_regular(
                    model, images, external_pose, camids, viewids, flip_test)

            if evaluator is not None:
                evaluator.update((descriptor, pids, camids_eval))
            total += int(images.shape[0])
            batches += 1
            _update_digest(descriptor_digest, descriptor)
            _update_digest(path_digest, imgpaths)
            _update_digest(label_digest, list(pids) + list(camids_eval))

    result = {
        'images': total,
        'batches': batches,
        'flip_test': flip_test,
        'descriptor_sha256': descriptor_digest.hexdigest(),
        'path_order_sha256': path_digest.hexdigest(),
        'pid_cam_order_sha256': label_digest.hexdigest(),
        'metrics_formal': formal_metrics,
    }
    if evaluator is not None:
        cmc, mean_ap, _, _, _, _, _ = evaluator.compute()
        result.update({
            'mAP': float(mean_ap),
            'rank1': float(cmc[0]),
            'rank5': float(cmc[4]),
            'rank10': float(cmc[9]),
        })
    if field_hook is not None:
        result['field_intervention'] = field_hook.summary()
    if psg_hooks is not None:
        result['psg_bypass'] = psg_hooks.summary()
    if matched_audits:
        result['matched_field_replacement'] = {
            'calls': len(matched_audits),
            'changed_calls': sum(
                row['changed_calls'] for row in matched_audits),
            'max_abs_delta': max(
                row['max_abs_delta'] for row in matched_audits),
            'all_hooks_removed': all(
                row['hook_removed'] for row in matched_audits),
        }
    if semantics is not None:
        result['semantic_diagnostics'] = semantics.summary()
    return result


def _validate_isolation(model, cfg):
    from model.pose_backbone_model import PoseBackboneModel
    if type(model) is not PoseBackboneModel:
        raise ValueError('semantic audit requires production PoseBackboneModel')
    if not model.use_tapf or not model.use_target_heatmap:
        raise ValueError('semantic audit requires TAPF target-person model')
    if model.tapf.mode != 'd0':
        raise ValueError('semantic audit is fixed to the D0 checkpoint')
    if model.tapf_source_stage != 2 or model.psg_stage_indices != {3}:
        raise ValueError('semantic audit requires Stage-2 TAPF and Stage-3 PSG')
    if not model.psg_modules_dict:
        raise ValueError('semantic audit requires non-empty Stage-3 PSG')
    if getattr(cfg.MODEL, 'POSE_TEST_FEAT', None) != 'global':
        raise ValueError('semantic audit requires global descriptor evaluation')
    if bool(cfg.TEST.RE_RANKING):
        raise ValueError('semantic audit forbids re-ranking')
    incompatible = (
        'POSE_PBSR', 'POSE_PRSM', 'POSE_SELECTIVE_SSM', 'POSE_HYPER_LORA',
        'POSE_LGPA', 'POSE_SKELETON_GCN', 'POSE_PPA', 'POSE_VCSR',
        'POSE_STRUCTURAL_ROUTING', 'POSE_ADDITIVE_ADAPTER',
        'POSE_PATCH_EMBED', 'POSE_PROMPT', 'POSE_CLIP_ID_PROMPT')
    enabled = [name for name in incompatible
               if bool(getattr(cfg.MODEL, name, False))]
    if enabled:
        raise ValueError('semantic audit isolation violated by: '
                         + ', '.join(enabled))


def _validate_frozen_assets(args):
    actual_config = _sha256_file(args.config_file)
    actual_checkpoint = _sha256_file(args.checkpoint)
    if actual_config != args.expected_config_sha256:
        raise RuntimeError('config SHA256 mismatch: %s' % actual_config)
    if actual_checkpoint != args.expected_checkpoint_sha256:
        raise RuntimeError('checkpoint SHA256 mismatch: %s' % actual_checkpoint)
    paths = {
        'query_map': args.query_donor_map,
        'gallery_map': args.gallery_donor_map,
        'query_metadata': args.query_donor_metadata,
        'gallery_metadata': args.gallery_donor_metadata,
        'query_audit': args.query_mapping_audit,
        'gallery_audit': args.gallery_mapping_audit,
    }
    actual = {name: _sha256_file(path) for name, path in paths.items()}
    mismatches = {name: value for name, value in actual.items()
                  if value != EXPECTED_MAPPING_SHA256[name]}
    if mismatches:
        raise RuntimeError('frozen donor asset SHA256 mismatch: %s' % mismatches)
    return {
        'config_sha256': actual_config,
        'checkpoint_sha256': actual_checkpoint,
        'mapping_sha256': actual,
    }


def _parse_args(argv: Sequence[str] | None = None):
    assets = ROOT / 'remote_artifacts/exp375/exp375_target_write_map_v1'
    parser = argparse.ArgumentParser(
        description='Read-only semantic audit of exp378 D0 epoch 90')
    parser.add_argument(
        '--config-file', type=Path,
        default=ROOT / 'configs/occluded_duke/exp378_d0_continued_pose.yml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-batches', type=int)
    parser.add_argument(
        '--expected-config-sha256', default=EXPECTED_CONFIG_SHA256)
    parser.add_argument(
        '--expected-checkpoint-sha256', default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument('--query-donor-map', type=Path,
                        default=assets / 'query_mappings.npy')
    parser.add_argument('--gallery-donor-map', type=Path,
                        default=assets / 'gallery_mappings.npy')
    parser.add_argument('--query-donor-metadata', type=Path,
                        default=assets / 'query_metadata.json')
    parser.add_argument('--gallery-donor-metadata', type=Path,
                        default=assets / 'gallery_metadata.json')
    parser.add_argument('--query-mapping-audit', type=Path,
                        default=assets / 'query_mapping_audit.json')
    parser.add_argument('--gallery-mapping-audit', type=Path,
                        default=assets / 'gallery_mapping_audit.json')
    parser.add_argument('--arms', nargs='+', choices=tuple(ARM_SPECS),
                        default=list(ARM_SPECS))
    args = parser.parse_args(argv)
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error('--max-batches must be positive')
    return args


def main(argv: Sequence[str] | None = None) -> int:
    # Parse first so utility-only imports and ``--help`` do not require the
    # complete torchvision/ReID runtime.
    args = _parse_args(argv)
    from datasets import make_dataloader
    from model import make_model

    frozen_assets = _validate_frozen_assets(args)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(args.config_file))
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
        raise RuntimeError('CUDA requested but unavailable')

    (_, _, val_loader, num_query, num_classes,
     camera_num, view_num) = make_dataloader(cfg)
    donor_indices, mapping_audit = load_split_donor_map(
        args.query_donor_map, args.gallery_donor_map,
        args.query_donor_metadata, args.gallery_donor_metadata,
        args.query_mapping_audit, args.gallery_mapping_audit,
        val_loader.dataset.dataset, num_query, len(val_loader.dataset))
    donor_loader = make_frozen_rgb_donor_loader(
        val_loader, donor_indices, num_query)

    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num,
        view_num=view_num,
        semantic_weight=float(cfg.MODEL.SEMANTIC_WEIGHT))
    _validate_isolation(model, cfg)
    loaded_tensors = strict_load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    state_before = _state_sha256(model)

    results = {}
    for arm in args.arms:
        loader = donor_loader if ARM_SPECS[arm]['loader'] == 'donor' \
            else val_loader
        results[arm] = evaluate_arm(
            model, cfg, loader, num_query, arm, device,
            max_batches=args.max_batches)
        row = results[arm]
        if row['metrics_formal']:
            print('%-24s mAP %.4f R1 %.4f R5 %.4f R10 %.4f' % (
                arm, row['mAP'], row['rank1'], row['rank5'], row['rank10']),
                  flush=True)
        else:
            print('%-24s images %d descriptor %s' % (
                arm, row['images'], row['descriptor_sha256'][:12]),
                  flush=True)

    state_after = _state_sha256(model)
    if state_after != state_before:
        raise RuntimeError('model state changed during read-only audit')
    if len({row['path_order_sha256'] for row in results.values()}) != 1 \
            or len({row['pid_cam_order_sha256']
                    for row in results.values()}) != 1:
        raise RuntimeError('RGB/path/PID/camera order changed across arms')

    reference = results.get('correct_start')
    if reference is not None:
        exact_keys = ('descriptor_sha256',)
        if reference['metrics_formal']:
            exact_keys += ('mAP', 'rank1', 'rank5', 'rank10')
            observed_percent = tuple(round(100.0 * reference[key], 1)
                                     for key in exact_keys[1:])
            if observed_percent != EXPECTED_CORRECT_PERCENT:
                raise RuntimeError(
                    'D0 e90 metric reproduction failed: observed %s, '
                    'expected %s' % (
                        observed_percent, EXPECTED_CORRECT_PERCENT))
        for arm in EXTERNAL_PARITY_ARMS + ('correct_end',):
            if arm in results and any(
                    results[arm][key] != reference[key] for key in exact_keys):
                raise RuntimeError('%s failed external/correct exact parity' % arm)
        for arm in FIELD_ARMS + ('matched_wrong_field', 'psg_bypass'):
            if (arm in results
                    and results[arm]['descriptor_sha256']
                    == reference['descriptor_sha256']):
                raise RuntimeError('%s failed to change descriptors' % arm)

    for arm in FIELD_ARMS:
        if arm in results:
            audit = results[arm]['field_intervention']
            if (audit['calls'] <= 0 or audit['changed_calls'] != audit['calls']
                    or not audit['hook_removed']):
                raise RuntimeError('%s field hook gate failed' % arm)
    if 'matched_wrong_field' in results:
        audit = results['matched_wrong_field']['matched_field_replacement']
        if (audit['calls'] <= 0 or audit['changed_calls'] != audit['calls']
                or not audit['all_hooks_removed']):
            raise RuntimeError('matched field replacement gate failed')
    if 'psg_bypass' in results:
        audit = results['psg_bypass']['psg_bypass']
        if not audit['all_called'] or not audit['hooks_removed']:
            raise RuntimeError('PSG bypass gate failed')

    if reference is not None and reference['metrics_formal']:
        for arm, row in results.items():
            row['delta_from_correct'] = {
                'mAP': row['mAP'] - reference['mAP'],
                'rank1': row['rank1'] - reference['rank1'],
                'rank5': row['rank5'] - reference['rank5'],
                'rank10': row['rank10'] - reference['rank10'],
            }

    output = {
        'schema': 'exp378-semantic-audit-v1',
        'git_head': _git_head(),
        'config': str(args.config_file.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'frozen_assets': frozen_assets,
        'loaded_tensors': loaded_tensors,
        'model_state_sha256_before': state_before,
        'model_state_sha256_after': state_after,
        'model_state_unchanged': state_before == state_after,
        'mapping_audit': mapping_audit,
        'max_batches': args.max_batches,
        'arms': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
