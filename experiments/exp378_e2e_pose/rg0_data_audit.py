"""Full real-data audit for the exp378 RG0 external Gaussian field.

This scans every target-person pose cache in train/query/gallery, then scans
the actual PoseImageDataset outputs for train (no augmentation), one complete
deterministic train augmentation pass, query, and gallery.  No model or GPU is
required and no training artifact is written.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as default_cfg
from datasets import make_dataloader
from datasets.pose_dataset import (
    PoseImageDataset, pose_train_collate_fn)
from model.modules.task_adaptive_pose_field import GaussianPoseFieldRenderer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class FieldAudit:
    def __init__(self, name, renderer):
        self.name = name
        self.renderer = renderer
        self.samples = 0
        self.joints = 0
        self.elements = 0
        self.raw_min = float('inf')
        self.raw_max = float('-inf')
        self.score_min = float('inf')
        self.score_max = float('-inf')
        self.field_min = float('inf')
        self.field_max = float('-inf')
        self.positive_mass_min = float('inf')
        self.active_positive_mass_min = float('inf')
        self.sigma_x_min = float('inf')
        self.sigma_x_max = float('-inf')
        self.sigma_y_min = float('inf')
        self.sigma_y_max = float('-inf')
        self.negative_elements = 0
        self.out_of_range_scores = 0
        self.near_zero_mass_joints = 0
        self.inconsistent_empty_joints = 0
        self.sigma_min_joints = 0
        self.sigma_max_joints = 0
        self.active_joints = 0
        self.active_sigma_min_joints = 0
        self.active_sigma_max_joints = 0
        self.missing_target_samples = 0
        self.nonfinite_values = 0
        self.peak_confidence_max_error = 0.0
        self.dtypes = set()
        self.shapes = set()

    def update(self, heatmaps, scores, person_mask):
        if heatmaps.ndim != 5 or heatmaps.shape[2:] != (17, 96, 32):
            raise RuntimeError(
                '%s unexpected pose shape %s' % (self.name, heatmaps.shape))
        if scores.shape[:2] != heatmaps.shape[:2] or scores.shape[2] != 17:
            raise RuntimeError('%s score shape mismatch' % self.name)
        if person_mask.shape != heatmaps.shape[:2]:
            raise RuntimeError('%s person-mask shape mismatch' % self.name)

        batch = heatmaps.shape[0]
        mask = person_mask[:, 0:1].float()
        target_heatmaps = heatmaps[:, 0].float() * mask[:, :, None, None]
        target_scores = scores[:, 0].float() * mask
        self.samples += batch
        self.joints += target_scores.numel()
        self.elements += target_heatmaps.numel()
        self.missing_target_samples += int((mask[:, 0] == 0).sum().item())
        self.dtypes.add(str(heatmaps.dtype))
        self.shapes.add(tuple(target_heatmaps.shape[1:]))

        finite_hm = torch.isfinite(target_heatmaps)
        finite_scores = torch.isfinite(target_scores)
        self.nonfinite_values += int((~finite_hm).sum().item())
        self.nonfinite_values += int((~finite_scores).sum().item())
        if not bool(finite_hm.all()) or not bool(finite_scores.all()):
            raise RuntimeError('%s contains non-finite pose data' % self.name)

        self.raw_min = min(self.raw_min, float(target_heatmaps.min().item()))
        self.raw_max = max(self.raw_max, float(target_heatmaps.max().item()))
        self.score_min = min(self.score_min, float(target_scores.min().item()))
        self.score_max = max(self.score_max, float(target_scores.max().item()))
        self.negative_elements += int((target_heatmaps < 0).sum().item())
        self.out_of_range_scores += int(((target_scores < 0)
                                        | (target_scores > 1)).sum().item())

        probability, confidence, mass = self.renderer.teacher_posterior(
            target_heatmaps, target_scores,
            reject_inconsistent_empty=False)
        active = confidence > 0
        near_zero = mass <= 1e-8
        inconsistent = near_zero & active
        self.positive_mass_min = min(
            self.positive_mass_min, float(mass.min().item()))
        if bool(active.any()):
            self.active_positive_mass_min = min(
                self.active_positive_mass_min,
                float(mass[active].min().item()))
        self.near_zero_mass_joints += int(near_zero.sum().item())
        self.inconsistent_empty_joints += int(inconsistent.sum().item())
        self.active_joints += int(active.sum().item())

        mu_x, mu_y, sigma_x, sigma_y = self.renderer.moments(probability)
        sigma_at_min = ((sigma_x <= self.renderer.sigma_min)
                        | (sigma_y <= self.renderer.sigma_min))
        sigma_at_max = ((sigma_x >= self.renderer.sigma_max)
                        | (sigma_y >= self.renderer.sigma_max))
        self.sigma_x_min = min(self.sigma_x_min, float(sigma_x.min().item()))
        self.sigma_x_max = max(self.sigma_x_max, float(sigma_x.max().item()))
        self.sigma_y_min = min(self.sigma_y_min, float(sigma_y.min().item()))
        self.sigma_y_max = max(self.sigma_y_max, float(sigma_y.max().item()))
        self.sigma_min_joints += int(sigma_at_min.sum().item())
        self.sigma_max_joints += int(sigma_at_max.sum().item())
        self.active_sigma_min_joints += int((sigma_at_min & active).sum().item())
        self.active_sigma_max_joints += int((sigma_at_max & active).sum().item())

        field = self.renderer.render(
            confidence, mu_x, mu_y, sigma_x, sigma_y)
        if field.dtype != torch.float32 or not bool(torch.isfinite(field).all()):
            raise RuntimeError('%s renderer output is invalid' % self.name)
        self.field_min = min(self.field_min, float(field.min().item()))
        self.field_max = max(self.field_max, float(field.max().item()))
        peak_error = (field.flatten(2).amax(dim=-1) - confidence).abs().max()
        self.peak_confidence_max_error = max(
            self.peak_confidence_max_error, float(peak_error.item()))

    @staticmethod
    def _finite_or_none(value):
        return value if math.isfinite(value) else None

    def result(self):
        return {
            'samples': self.samples,
            'joints': self.joints,
            'target_shape_set': [list(shape) for shape in sorted(self.shapes)],
            'input_dtype_set': sorted(self.dtypes),
            'raw_min': self._finite_or_none(self.raw_min),
            'raw_max': self._finite_or_none(self.raw_max),
            'raw_negative_fraction': (
                self.negative_elements / max(self.elements, 1)),
            'score_min': self._finite_or_none(self.score_min),
            'score_max': self._finite_or_none(self.score_max),
            'score_out_of_range_fraction': (
                self.out_of_range_scores / max(self.joints, 1)),
            'positive_mass_min': self._finite_or_none(
                self.positive_mass_min),
            'active_positive_mass_min': self._finite_or_none(
                self.active_positive_mass_min),
            'near_zero_mass_joints': self.near_zero_mass_joints,
            'inconsistent_empty_joints': self.inconsistent_empty_joints,
            'missing_target_samples': self.missing_target_samples,
            'nonfinite_values': self.nonfinite_values,
            'sigma_x_min': self._finite_or_none(self.sigma_x_min),
            'sigma_x_max': self._finite_or_none(self.sigma_x_max),
            'sigma_y_min': self._finite_or_none(self.sigma_y_min),
            'sigma_y_max': self._finite_or_none(self.sigma_y_max),
            'sigma_min_fraction': self.sigma_min_joints / max(self.joints, 1),
            'sigma_max_fraction': self.sigma_max_joints / max(self.joints, 1),
            'active_sigma_min_fraction': (
                self.active_sigma_min_joints / max(self.active_joints, 1)),
            'active_sigma_max_fraction': (
                self.active_sigma_max_joints / max(self.active_joints, 1)),
            'rendered_min': self._finite_or_none(self.field_min),
            'rendered_max': self._finite_or_none(self.field_max),
            'rendered_peak_confidence_max_error': (
                self.peak_confidence_max_error),
        }


class TargetCropDataset(torch.utils.data.Dataset):
    """Load only the production target person before random pad/crop.

    PoseImageDataset normally materializes all six persons and the RGB image.
    The exhaustive crop audit needs neither, so this adapter reproduces its
    target ordering, heatmap placement and resize while reading one NPZ and the
    image header per sample.
    """

    def __init__(self, source, target_size):
        self.dataset = source.dataset
        self.pose_dir = source.pose_dir
        self.index = source.index
        self.max_persons = source.max_persons
        self.target_size = tuple(int(value) for value in target_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        with Image.open(img_path) as image:
            orig_w, orig_h = image.size
        entry = self.index.get(os.path.basename(img_path))
        person_files = [] if entry is None else list(
            entry.get('persons', [])[:self.max_persons])
        if person_files:
            target_index = int(entry.get('target_person_idx', 0))
            if 0 < target_index < len(person_files):
                target_file = person_files.pop(target_index)
                person_files.insert(0, target_file)

        selected = None
        for npz_name in person_files:
            npz_path = (npz_name if os.path.isabs(npz_name)
                        else os.path.join(self.pose_dir, npz_name))
            if os.path.exists(npz_path):
                selected = npz_path
                break

        target_h, target_w = self.target_size
        if selected is None:
            return (torch.zeros(17, target_h, target_w),
                    torch.zeros(17, 2), torch.zeros(17),
                    torch.tensor(0.0))

        with np.load(selected) as data:
            heatmap = torch.from_numpy(
                data['heatmap'].astype(np.float32))
            keypoints = data['keypoints'].astype(np.float32)
            scores = data['scores'].astype(np.float32)
            crop_bounds = data['crop_bounds'].astype(np.float32)
        heatmap = PoseImageDataset._place_heatmap(
            heatmap, crop_bounds, orig_h, orig_w)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0), size=self.target_size,
            mode='bilinear', align_corners=False).squeeze(0)
        keypoints[:, 0] *= target_w / orig_w
        keypoints[:, 1] *= target_h / orig_h
        return (heatmap, torch.from_numpy(keypoints.copy()),
                torch.from_numpy(scores.copy()), torch.tensor(1.0))


def _rectangle_sum(prefix, y1, y2, x1, x2):
    return (prefix[..., y2, x2] - prefix[..., y1, x2]
            - prefix[..., y2, x1] + prefix[..., y1, x1])


def scan_all_train_crops(loader, pad, output_size, device):
    """Exhaustively audit all (2*pad+1)^2 production pad/crop offsets.

    For the fixed 384x128 -> 96x32 bilinear downsample, every output value is
    the average of a 2x2 block on a stride-4 lattice.  Sixteen residue-class
    average-pools plus integral images therefore recover the exact positive
    mass for all 441 crops without materializing 441 full-resolution crops.
    Horizontal flip need not be repeated: it only mirrors offsets and swaps
    left/right joint channels, both already covered by the exhaustive set.
    """
    target_h, target_w = loader.dataset.target_size
    output_h, output_w = tuple(int(value) for value in output_size)
    if (target_h, target_w) != (output_h * 4, output_w * 4):
        raise RuntimeError('crop audit requires exact 4x pose downsample')
    offsets = list(range(2 * pad + 1))
    result = {
        'samples': 0,
        'crop_offsets_per_sample': len(offsets) ** 2,
        'joint_crop_states': 0,
        'active_joint_crop_states': 0,
        'inconsistent_empty_joints': 0,
        'nonfinite_values': 0,
        'active_positive_mass_min': float('inf'),
        'analytic_interpolate_spotcheck_max_error': 0.0,
    }
    spotchecked = False
    for heatmaps, keypoints, scores, person_mask in loader:
        heatmaps = heatmaps.to(device).float()
        keypoints = keypoints.to(device).float()
        scores = scores.to(device).float()
        person_mask = person_mask.to(device).float()
        finite = (torch.isfinite(heatmaps).all()
                  & torch.isfinite(keypoints).all()
                  & torch.isfinite(scores).all())
        if not bool(finite):
            result['nonfinite_values'] += 1
            raise RuntimeError('exhaustive crop inputs contain NaN/Inf')
        positive = heatmaps.clamp_min(0.0) * person_mask[:, None, None, None]
        confidence = scores.clamp(0.0, 1.0) * person_mask[:, None]
        padded = F.pad(positive, (pad, pad, pad, pad), value=0.0)
        batch = heatmaps.shape[0]
        result['samples'] += batch
        result['joint_crop_states'] += batch * 17 * len(offsets) ** 2

        for residue_y in range(4):
            y_offsets = [value for value in offsets
                         if value % 4 == residue_y]
            for residue_x in range(4):
                x_offsets = [value for value in offsets
                             if value % 4 == residue_x]
                sampled = F.avg_pool2d(
                    padded[..., residue_y + 1:, residue_x + 1:],
                    kernel_size=2, stride=4)
                prefix = F.pad(
                    sampled.cumsum(dim=-2).cumsum(dim=-1),
                    (1, 0, 1, 0), value=0.0)
                for crop_y in y_offsets:
                    block_y = (crop_y - residue_y) // 4
                    for crop_x in x_offsets:
                        block_x = (crop_x - residue_x) // 4
                        mass = _rectangle_sum(
                            prefix, block_y, block_y + output_h,
                            block_x, block_x + output_w)
                        shifted_x = keypoints[..., 0] + pad - crop_x
                        shifted_y = keypoints[..., 1] + pad - crop_y
                        in_bounds = ((shifted_x >= 0)
                                     & (shifted_x < target_w)
                                     & (shifted_y >= 0)
                                     & (shifted_y < target_h))
                        active = (confidence > 0) & in_bounds
                        inconsistent = active & (mass <= 1e-8)
                        result['active_joint_crop_states'] += int(
                            active.sum().item())
                        result['inconsistent_empty_joints'] += int(
                            inconsistent.sum().item())
                        if bool(active.any()):
                            result['active_positive_mass_min'] = min(
                                result['active_positive_mass_min'],
                                float(mass[active].min().item()))

                        if (not spotchecked and crop_y in (0, pad, 2 * pad)
                                and crop_x in (0, pad, 2 * pad)):
                            crop = padded[:1, :,
                                          crop_y:crop_y + target_h,
                                          crop_x:crop_x + target_w]
                            reference = F.interpolate(
                                crop, size=(output_h, output_w),
                                mode='bilinear', align_corners=False
                            ).flatten(2).sum(dim=-1)
                            error = (reference - mass[:1]).abs().max()
                            result[
                                'analytic_interpolate_spotcheck_max_error'
                            ] = max(
                                result[
                                    'analytic_interpolate_spotcheck_max_error'],
                                float(error.item()))
        spotchecked = True
    if not math.isfinite(result['active_positive_mass_min']):
        result['active_positive_mass_min'] = None
    result['device'] = str(device)
    return result


def scan_loader(name, loader, renderer, split_paths=False):
    audits = {}
    for batch in loader:
        pose_dict = batch[-1]
        if split_paths:
            paths = batch[-2]
            groups = {
                'query': [i for i, path in enumerate(paths)
                          if Path(path).parent.name == 'query'],
                'gallery': [i for i, path in enumerate(paths)
                            if Path(path).parent.name != 'query'],
            }
        else:
            groups = {name: list(range(pose_dict['heatmaps'].shape[0]))}
        for group, indices in groups.items():
            if not indices:
                continue
            audit = audits.setdefault(group, FieldAudit(group, renderer))
            index = torch.tensor(indices, dtype=torch.long)
            audit.update(
                pose_dict['heatmaps'].index_select(0, index),
                pose_dict['scores'].index_select(0, index),
                pose_dict['person_mask'].index_select(0, index))
    return {key: value.result() for key, value in audits.items()}


def scan_pose_cache(pose_root):
    result = {}
    for split in ('train', 'query', 'gallery'):
        split_dir = pose_root / split
        with (split_dir / 'index.json').open() as handle:
            index = json.load(handle)
        stats = {
            'index_entries': len(index),
            'missing_person_entries': 0,
            'missing_npz_files': 0,
            'target_index_outside_loaded_persons': 0,
            'nonfinite_values': 0,
            'inconsistent_empty_joints': 0,
            'score_out_of_range_joints': 0,
            'joints': 0,
            'heatmap_shapes': set(),
            'heatmap_dtypes': set(),
            'raw_min': float('inf'),
            'raw_max': float('-inf'),
            'score_min': float('inf'),
            'score_max': float('-inf'),
            'active_positive_mass_min': float('inf'),
        }
        for entry in index.values():
            persons = list(entry.get('persons', [])[:6])
            if not persons:
                stats['missing_person_entries'] += 1
                continue
            target_index = int(entry.get('target_person_idx', 0))
            if target_index < 0 or target_index >= len(persons):
                stats['target_index_outside_loaded_persons'] += 1
                target_index = 0
            npz_path = split_dir / persons[target_index]
            if not npz_path.exists():
                stats['missing_npz_files'] += 1
                continue
            with np.load(str(npz_path)) as data:
                heatmap = data['heatmap']
                scores = data['scores'].astype(np.float32)
            stats['heatmap_shapes'].add(tuple(heatmap.shape))
            stats['heatmap_dtypes'].add(str(heatmap.dtype))
            finite_hm = np.isfinite(heatmap)
            finite_scores = np.isfinite(scores)
            stats['nonfinite_values'] += int((~finite_hm).sum())
            stats['nonfinite_values'] += int((~finite_scores).sum())
            if not finite_hm.all() or not finite_scores.all():
                continue
            raw = heatmap.astype(np.float32)
            positive_mass = np.maximum(raw, 0.0).reshape(17, -1).sum(axis=1)
            confidence = np.clip(scores, 0.0, 1.0)
            active = confidence > 0
            inconsistent = (positive_mass <= 1e-8) & active
            stats['inconsistent_empty_joints'] += int(inconsistent.sum())
            stats['score_out_of_range_joints'] += int(
                ((scores < 0) | (scores > 1)).sum())
            stats['joints'] += 17
            stats['raw_min'] = min(stats['raw_min'], float(raw.min()))
            stats['raw_max'] = max(stats['raw_max'], float(raw.max()))
            stats['score_min'] = min(stats['score_min'], float(scores.min()))
            stats['score_max'] = max(stats['score_max'], float(scores.max()))
            if active.any():
                stats['active_positive_mass_min'] = min(
                    stats['active_positive_mass_min'],
                    float(positive_mass[active].min()))
        stats['heatmap_shapes'] = [list(shape)
                                   for shape in sorted(stats['heatmap_shapes'])]
        stats['heatmap_dtypes'] = sorted(stats['heatmap_dtypes'])
        for key in ('raw_min', 'raw_max', 'score_min', 'score_max',
                    'active_positive_mass_min'):
            if not math.isfinite(stats[key]):
                stats[key] = None
        result[split] = stats
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/occluded_duke/exp378_rg0_external_gaussian.yml')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--crop-batch-size', type=int, default=16)
    parser.add_argument(
        '--crop-device',
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = default_cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / args.config))
    config.freeze()
    set_seed(args.seed)
    renderer = GaussianPoseFieldRenderer(
        output_size=tuple(config.MODEL.POSE_HEATMAP_SIZE),
        sigma_min=float(config.MODEL.POSE_EXTERNAL_SIGMA_MIN),
        sigma_max=float(config.MODEL.POSE_EXTERNAL_SIGMA_MAX))

    pose_root = ROOT / config.MODEL.POSE_DATA_DIR / 'pose_data'
    cache = scan_pose_cache(pose_root)
    (train_loader, train_normal, val_loader, num_query,
     _, _, _) = make_dataloader(config)
    actual = {}
    actual.update(scan_loader('train_base', train_normal, renderer))
    actual.update(scan_loader('val', val_loader, renderer, split_paths=True))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_augmented = DataLoader(
        train_loader.dataset,
        batch_size=int(config.SOLVER.IMS_PER_BATCH),
        shuffle=False,
        num_workers=int(config.DATALOADER.NUM_WORKERS),
        collate_fn=pose_train_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=False)
    actual.update(scan_loader(
        'train_augmented_seed%d' % args.seed,
        train_augmented, renderer))

    crop_dataset = TargetCropDataset(
        train_loader.dataset, target_size=tuple(config.INPUT.SIZE_TRAIN))
    crop_loader = DataLoader(
        crop_dataset, batch_size=args.crop_batch_size, shuffle=False,
        num_workers=int(config.DATALOADER.NUM_WORKERS), pin_memory=False)
    exhaustive_crops = scan_all_train_crops(
        crop_loader, pad=int(config.INPUT.PADDING),
        output_size=tuple(config.MODEL.POSE_HEATMAP_SIZE),
        device=torch.device(args.crop_device))

    report = {
        'schema': 'exp378-rg0-data-audit-v1',
        'seed': args.seed,
        'config': args.config,
        'cache': cache,
        'actual_dataset_outputs': actual,
        'exhaustive_train_pad_crop': exhaustive_crops,
    }
    failures = []
    for split, stats in cache.items():
        for key in ('missing_npz_files', 'nonfinite_values',
                    'inconsistent_empty_joints',
                    'target_index_outside_loaded_persons'):
            if stats[key] != 0:
                failures.append('cache.%s.%s=%s' % (split, key, stats[key]))
    expected_counts = {
        'train_base': len(train_loader.dataset),
        'train_augmented_seed%d' % args.seed: len(train_loader.dataset),
        'query': num_query,
        'gallery': len(val_loader.dataset) - num_query,
    }
    if set(actual) != set(expected_counts):
        failures.append('actual.keys=%s expected=%s' % (
            sorted(actual), sorted(expected_counts)))
    for split, expected_count in expected_counts.items():
        if split not in actual:
            continue
        stats = actual[split]
        if stats['samples'] != expected_count:
            failures.append('actual.%s.samples=%s expected=%s' % (
                split, stats['samples'], expected_count))
        for key in ('nonfinite_values', 'inconsistent_empty_joints'):
            if stats[key] != 0:
                failures.append('actual.%s.%s=%s' % (split, key, stats[key]))
        if stats['target_shape_set'] != [[17, 96, 32]]:
            failures.append('actual.%s.shape=%s'
                            % (split, stats['target_shape_set']))
        if stats['rendered_min'] < 0 or stats['rendered_max'] > 1:
            failures.append('actual.%s.rendered_range' % split)
        if stats['rendered_peak_confidence_max_error'] != 0:
            failures.append('actual.%s.peak_confidence_error=%s' % (
                split, stats['rendered_peak_confidence_max_error']))
    if exhaustive_crops['samples'] != len(train_loader.dataset):
        failures.append('exhaustive_crops.samples=%s expected=%s' % (
            exhaustive_crops['samples'], len(train_loader.dataset)))
    if exhaustive_crops['nonfinite_values'] != 0:
        failures.append('exhaustive_crops.nonfinite_values=%s' %
                        exhaustive_crops['nonfinite_values'])
    if exhaustive_crops['inconsistent_empty_joints'] != 0:
        failures.append('exhaustive_crops.inconsistent_empty_joints=%s' %
                        exhaustive_crops['inconsistent_empty_joints'])
    # The two paths sum the same nonnegative samples in different float32
    # orders (integral image versus F.interpolate reduction).  Zero/nonzero
    # decisions remain exact; allow only the observed reduction-scale noise.
    if exhaustive_crops['analytic_interpolate_spotcheck_max_error'] > 1e-4:
        failures.append('exhaustive_crops.spotcheck_error=%s' %
                        exhaustive_crops[
                            'analytic_interpolate_spotcheck_max_error'])
    report['status'] = 'PASS' if not failures else 'FAIL'
    report['failures'] = failures
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print('RG0_FULL_DATA_AUDIT_PASS')


if __name__ == '__main__':
    main()
