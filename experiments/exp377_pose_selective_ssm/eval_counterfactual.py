#!/usr/bin/env python3
"""Frozen-checkpoint counterfactual evaluator for exp377.

One strictly loaded P0 model is reused for every arm.  RGB order, labels and
camera IDs are held fixed; only the target-person pose or the module's
``pose_source`` mode changes.  Frozen query/gallery donor maps use the proven
exp375 dataset wrapper, but all exp377 pose transformations are defined here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg  # noqa: E402
from experiments.exp375_prsm.eval_counterfactual import (  # noqa: E402
    _pose_to_device,
    _sha256_file,
    _update_digest,
    load_split_donor_map,
    make_frozen_donor_loader,
)


ARM_SPECS = {
    'correct_start': {'loader': 'recipient', 'source': 'input'},
    'matched_shuffle': {'loader': 'donor', 'source': 'input'},
    'recipient_visibility_donor_composition': {
        'loader': 'donor', 'source': 'input'},
    'donor_visibility_recipient_composition': {
        'loader': 'donor', 'source': 'input'},
    'channel_permutation': {'loader': 'recipient', 'source': 'input'},
    'canonical': {'loader': 'recipient', 'source': 'canonical'},
    'pose_off': {'loader': 'recipient', 'source': 'zero'},
    'correct_end': {'loader': 'recipient', 'source': 'input'},
}

DONOR_ARMS = tuple(
    name for name, spec in ARM_SPECS.items()
    if spec['loader'] == 'donor')


def _target_heatmaps(pose_dict: Mapping[str, Any]) -> torch.Tensor:
    heatmaps = pose_dict.get('heatmaps')
    if not torch.is_tensor(heatmaps):
        raise ValueError('pose_dict["heatmaps"] must be a tensor')
    if heatmaps.ndim == 5 and heatmaps.shape[1] >= 1 \
            and heatmaps.shape[2] == 17:
        return heatmaps[:, 0]
    if heatmaps.ndim == 4 and heatmaps.shape[1] == 17:
        return heatmaps
    raise ValueError(
        'heatmaps must be [B,P,17,H,W] or [B,17,H,W], got %s'
        % (tuple(heatmaps.shape),))


def _replace_target_heatmaps(
        pose_dict: Mapping[str, Any], target: torch.Tensor) -> dict:
    """Clone a pose batch and replace only person-0 heatmaps."""
    result = dict(pose_dict)
    heatmaps = pose_dict['heatmaps']
    if heatmaps.ndim == 5:
        if heatmaps.shape[-2:] != target.shape[-2:]:
            batch, people, joints = heatmaps.shape[:3]
            replaced = F.interpolate(
                heatmaps.float().reshape(
                    batch * people, joints, *heatmaps.shape[-2:]),
                size=target.shape[-2:], mode='bilinear',
                align_corners=False).reshape(
                    batch, people, joints, *target.shape[-2:]).to(heatmaps)
        else:
            replaced = heatmaps.clone()
        replaced[:, 0] = target.to(replaced)
    else:
        replaced = target.to(heatmaps)
    result['heatmaps'] = replaced
    return result


def _composition_visibility(heatmaps: torch.Tensor):
    heatmaps = heatmaps.float().clamp(min=0.0, max=1.0)
    mass = heatmaps.sum(dim=1, keepdim=True)
    composition = heatmaps / mass.clamp_min(1e-6)
    visibility = heatmaps.amax(dim=1, keepdim=True)
    return composition, visibility


def _fill_empty_composition(composition: torch.Tensor,
                            heatmaps: torch.Tensor) -> torch.Tensor:
    """Fill source-empty pixels with that donor's global joint composition.

    A local composition is undefined where all source joints are zero.  The
    sample's own global joint-mass distribution is the only donor-derived
    fallback that lets the opposite arm preserve the requested support
    exactly.  The evaluator reports final-grid checks so fallback use cannot
    silently masquerade as an exact local swap.
    """
    global_mass = heatmaps.float().sum(dim=(-2, -1), keepdim=True)
    global_q = global_mass / global_mass.sum(
        dim=1, keepdim=True).clamp_min(1e-6)
    uniform = torch.full_like(global_q, 1.0 / global_q.shape[1])
    global_q = torch.where(
        global_mass.sum(dim=1, keepdim=True) > 1e-6,
        global_q, uniform)
    empty = composition.sum(dim=1, keepdim=True) <= 1e-6
    return torch.where(empty.expand_as(composition),
                       global_q.expand_as(composition), composition)


def _final_grid_pose(heatmaps: torch.Tensor,
                     size=(12, 4)) -> torch.Tensor:
    """Apply exactly the bilinear resize used by PoseSelectiveSSM."""
    if heatmaps.shape[-2:] == tuple(size):
        return heatmaps.float().clamp(min=0.0, max=1.0)
    return F.interpolate(
        heatmaps.float(), size=size, mode='bilinear',
        align_corners=False).clamp(min=0.0, max=1.0)


def _compose_pose(composition: torch.Tensor,
                  visibility: torch.Tensor) -> torch.Tensor:
    """Construct heatmaps with exactly the requested q and max visibility."""
    q_max = composition.amax(dim=1, keepdim=True)
    scale = visibility / q_max.clamp_min(1e-6)
    reconstructed = composition * scale
    empty = (q_max <= 1e-6) | (visibility <= 0)
    return torch.where(empty.expand_as(reconstructed),
                       torch.zeros_like(reconstructed), reconstructed)


def transform_pose_batch(arm: str, pose_dict: Mapping[str, Any]) -> dict:
    """Apply one exp377 intervention to a recipient/donor pose batch."""
    if arm not in ARM_SPECS:
        raise ValueError('unknown exp377 arm %r' % arm)
    result = dict(pose_dict)
    recipient_pose = result.pop('_exp375_recipient_pose', None)
    recipient_index = result.pop('_exp375_recipient_index', None)
    donor_index = result.pop('_exp375_donor_index', None)

    if arm in DONOR_ARMS:
        if recipient_pose is None:
            raise ValueError('%s requires the frozen donor loader' % arm)
        if recipient_index is not None and donor_index is not None \
                and bool((recipient_index == donor_index).any()):
            raise RuntimeError('frozen donor map contains a fixed point')

    if arm == 'recipient_visibility_donor_composition':
        donor = _final_grid_pose(_target_heatmaps(result))
        recipient = _final_grid_pose(_target_heatmaps(recipient_pose))
        donor_q, _ = _composition_visibility(donor)
        donor_q = _fill_empty_composition(donor_q, donor)
        _, recipient_visibility = _composition_visibility(recipient)
        result = _replace_target_heatmaps(
            result, _compose_pose(donor_q, recipient_visibility))
        result['_exp377_expected_composition'] = donor_q
        result['_exp377_expected_visibility'] = recipient_visibility
    elif arm == 'donor_visibility_recipient_composition':
        donor = _final_grid_pose(_target_heatmaps(result))
        recipient = _final_grid_pose(_target_heatmaps(recipient_pose))
        recipient_q, _ = _composition_visibility(recipient)
        recipient_q = _fill_empty_composition(recipient_q, recipient)
        _, donor_visibility = _composition_visibility(donor)
        result = _replace_target_heatmaps(
            result, _compose_pose(recipient_q, donor_visibility))
        result['_exp377_expected_composition'] = recipient_q
        result['_exp377_expected_visibility'] = donor_visibility
    elif arm == 'channel_permutation':
        target = _target_heatmaps(result)
        # Fixed derangement: preserves every pixel's total/max/support while
        # destroying COCO joint identity.  Applying it twice is not assumed.
        permutation = torch.tensor(
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 5,
             12, 13, 14, 15, 16, 11],
            device=target.device, dtype=torch.long)
        result = _replace_target_heatmaps(
            result, target.index_select(1, permutation))
    return result


@contextlib.contextmanager
def selective_ssm_arm(model: torch.nn.Module, arm: str):
    if arm not in ARM_SPECS:
        raise ValueError('unknown exp377 arm %r' % arm)
    if not getattr(model, 'use_pose_selective_ssm', False):
        raise ValueError('counterfactual requires POSE_SELECTIVE_SSM=True')
    old_source = model.pose_selective_ssm_pose_source
    model.pose_selective_ssm_pose_source = ARM_SPECS[arm]['source']
    try:
        yield
    finally:
        model.pose_selective_ssm_pose_source = old_source


def _unwrap_checkpoint(payload: Any) -> MutableMapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and 'state_dict' in payload:
        payload = payload['state_dict']
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError('checkpoint must contain a non-empty state_dict')
    state = dict(payload)
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError('state_dict must map strings to tensors')
    prefixes = [key.startswith('module.') for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError('checkpoint mixes module-prefixed and plain keys')
    if all(prefixes):
        state = {key[len('module.'):]: value for key, value in state.items()}
    return state


def strict_load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> int:
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location='cpu',
                             weights_only=False)
    except TypeError:  # PyTorch 1.13 compatibility.
        payload = torch.load(checkpoint, map_location='cpu')
    state = _unwrap_checkpoint(payload)
    model.load_state_dict(state, strict=True)
    return len(state)


def _forward_descriptor(model, images, pose_dict, camids, viewids,
                        flip_test):
    descriptor, _ = model(
        images, cam_label=camids, view_label=viewids, pose_dict=pose_dict)
    if flip_test:
        from utils.flip_test import flip_batch
        flipped_images, flipped_pose = flip_batch(images, pose_dict)
        flipped, _ = model(
            flipped_images, cam_label=camids, view_label=viewids,
            pose_dict=flipped_pose)
        descriptor = 0.5 * (descriptor + flipped)
    if descriptor.ndim != 2 or not bool(torch.isfinite(descriptor).all()):
        raise RuntimeError('invalid/non-finite global descriptor')
    return descriptor


def _pose_audit(model, pose_dict, source):
    target = _target_heatmaps(pose_dict)
    if source == 'canonical':
        target = model._canonical_heatmap(target.shape[0], target.device)
    elif source == 'zero':
        target = torch.zeros_like(target)
    q, visibility = model.pose_selective_ssm._local_pose(
        target, None, target.shape[0], 12, 4, target.device,
        target.dtype)
    digest = hashlib.sha256()
    _update_digest(digest, q)
    result = {
        'composition_sha256': digest.hexdigest(),
        'visibility_mean': float(visibility.float().mean()),
        'visibility_nonzero': float((visibility > 0).float().mean()),
    }
    expected_q = pose_dict.get('_exp377_expected_composition')
    expected_visibility = pose_dict.get('_exp377_expected_visibility')
    if expected_q is not None and expected_visibility is not None:
        expected_q_tokens = model.pose_selective_ssm._serpentine_grid(
            expected_q.permute(0, 2, 3, 1)).reshape_as(q).to(q)
        expected_v_tokens = model.pose_selective_ssm._serpentine_grid(
            expected_visibility.permute(0, 2, 3, 1)).reshape_as(
                visibility).to(visibility)
        active = expected_v_tokens > 1e-6
        result['composition_active_max_abs_error'] = float(
            ((q - expected_q_tokens).abs() * active).amax())
        result['visibility_max_abs_error'] = float(
            (visibility - expected_v_tokens).abs().amax())
    return result


def evaluate_arm(model, cfg, loader, num_query, arm, device):
    from utils.metrics import R1_mAP_eval
    evaluator = R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
        reranking=False, cfg=cfg)
    evaluator.reset()
    descriptor_digest = hashlib.sha256()
    path_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    pose_digests = hashlib.sha256()
    visibility_sum = 0.0
    visibility_nonzero_sum = 0.0
    composition_error_max = 0.0
    visibility_error_max = 0.0
    batches = 0
    total = 0
    flip_test = bool(getattr(cfg.TEST, 'FLIP_TEST', True))

    with selective_ssm_arm(model, arm), torch.inference_mode():
        for batch_data in loader:
            (images, pids, camids_eval, camids, viewids,
             imgpaths, pose_dict) = batch_data
            images = images.to(device)
            camids = camids.to(device)
            viewids = viewids.to(device)
            pose_dict = _pose_to_device(pose_dict, device)
            pose_dict = transform_pose_batch(arm, pose_dict)
            audit = _pose_audit(
                model, pose_dict, ARM_SPECS[arm]['source'])
            pose_digests.update(audit['composition_sha256'].encode('ascii'))
            visibility_sum += audit['visibility_mean']
            visibility_nonzero_sum += audit['visibility_nonzero']
            composition_error_max = max(
                composition_error_max,
                audit.get('composition_active_max_abs_error', 0.0))
            visibility_error_max = max(
                visibility_error_max,
                audit.get('visibility_max_abs_error', 0.0))
            batches += 1
            descriptor = _forward_descriptor(
                model, images, pose_dict, camids, viewids, flip_test)
            evaluator.update((descriptor, pids, camids_eval))
            total += int(images.shape[0])
            _update_digest(descriptor_digest, descriptor)
            _update_digest(path_digest, imgpaths)
            _update_digest(label_digest, list(pids) + list(camids_eval))

    cmc, mean_ap, _, _, _, _, _ = evaluator.compute()
    return {
        'mAP': float(mean_ap),
        'rank1': float(cmc[0]),
        'rank5': float(cmc[4]),
        'rank10': float(cmc[9]),
        'images': total,
        'flip_test': flip_test,
        'descriptor_sha256': descriptor_digest.hexdigest(),
        'path_order_sha256': path_digest.hexdigest(),
        'pid_cam_order_sha256': label_digest.hexdigest(),
        'pose_composition_sha256': pose_digests.hexdigest(),
        'pose_visibility_mean': visibility_sum / max(batches, 1),
        'pose_visibility_nonzero': visibility_nonzero_sum / max(batches, 1),
        'composition_active_max_abs_error': composition_error_max,
        'visibility_max_abs_error': visibility_error_max,
    }


def _validate_isolation(model, cfg):
    from model.pose_backbone_model import PoseBackboneModel
    if type(model) is not PoseBackboneModel:
        raise ValueError('exp377 requires production PoseBackboneModel')
    if not model.use_pose_selective_ssm or not model.use_target_heatmap:
        raise ValueError('exp377 requires selective SSM and target heatmaps')
    if model.psg_stage_indices or model.use_prsm \
            or getattr(model, 'use_pose_hyper_lora', False):
        raise ValueError('exp377 isolation violated by another backbone module')
    if getattr(cfg.MODEL, 'POSE_TEST_FEAT', None) != 'global':
        raise ValueError('exp377 requires global descriptor evaluation')


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Evaluate a frozen exp377 P0 checkpoint')
    parser.add_argument(
        '--config-file', type=Path,
        default=ROOT / 'configs/occluded_duke/exp377_p0_pose_ssm.yml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--query-donor-map', type=Path)
    parser.add_argument('--gallery-donor-map', type=Path)
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
    from datasets import make_dataloader
    from model import make_model

    args = _parse_args(argv)
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(args.config_file))
    cfg.merge_from_list(args.opts)
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
    donor_loader = None
    mapping_audit = None
    if any(arm in DONOR_ARMS for arm in args.arms):
        required = (
            args.query_donor_map, args.gallery_donor_map,
            args.query_donor_metadata, args.gallery_donor_metadata,
            args.query_mapping_audit, args.gallery_mapping_audit)
        if any(value is None for value in required):
            raise ValueError(
                'donor arms require frozen query/gallery maps, metadata and audits')
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
        loader = donor_loader if arm in DONOR_ARMS else val_loader
        results[arm] = evaluate_arm(
            model, cfg, loader, num_query, arm, device)
        row = results[arm]
        if arm in (
                'recipient_visibility_donor_composition',
                'donor_visibility_recipient_composition'):
            if row['composition_active_max_abs_error'] > 1e-5 \
                    or row['visibility_max_abs_error'] > 1e-5:
                raise RuntimeError(
                    '%s failed final-grid support/composition audit' % arm)
        print('%-43s mAP %.4f R1 %.4f R5 %.4f R10 %.4f' % (
            arm, row['mAP'], row['rank1'], row['rank5'], row['rank10']),
              flush=True)

    if len({row['path_order_sha256'] for row in results.values()}) != 1 \
            or len({row['pid_cam_order_sha256']
                    for row in results.values()}) != 1:
        raise RuntimeError('RGB/path/PID/camera order changed across arms')
    if 'correct_start' in results and 'correct_end' in results:
        start, end = results['correct_start'], results['correct_end']
        exact = ('mAP', 'rank1', 'rank5', 'rank10', 'descriptor_sha256')
        if any(start[key] != end[key] for key in exact):
            raise RuntimeError('correct-start/end reproducibility failed')
    if 'matched_shuffle' in results \
            and results['matched_shuffle']['descriptor_sha256'] == \
            results.get('correct_start', {}).get('descriptor_sha256'):
        raise RuntimeError('matched pose failed to change descriptors')
    if 'pose_off' in results and results['pose_off']['descriptor_sha256'] == \
            results.get('correct_start', {}).get('descriptor_sha256'):
        raise RuntimeError('pose-off failed to change descriptors')

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
