"""Read-only final checkpoint audit for exp379 HT0.

The audit verifies that every hierarchical component stays finite and moves
across all twelve saved checkpoints.  It then reloads the final checkpoint in
the production model and proves that evaluation descriptors are exactly
independent of external pose inputs.
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ.get(
    'EXP379_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))

from config import cfg
from datasets import make_dataloader
from model import make_model


GROUP_PREFIXES = {
    'projection_stage1': ('tapf.stage_projections.1.',),
    'projection_stage2': ('tapf.stage_projections.2.',),
    'shared_decoder': ('tapf.anchor.',),
    'psg_stage2': ('psg_modules_dict.s2_',),
    'psg_stage3': ('psg_modules_dict.s3_',),
}


class ExplodingPoseDict(dict):
    """Fail if the deployment path attempts to inspect external pose."""

    def __getitem__(self, key):
        raise RuntimeError('RGB-only eval touched external pose: ' + key)

    def get(self, key, default=None):
        del default
        raise RuntimeError('RGB-only eval touched external pose: ' + key)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_keys(state, prefixes):
    return tuple(key for key in state if key.startswith(prefixes))


def compare_group(reference, observed, keys):
    changed = 0
    maximum = 0.0
    squared_l2 = 0.0
    for key in keys:
        left = reference[key].detach().float()
        right = observed[key].detach().float()
        if not bool(torch.isfinite(left).all()) \
                or not bool(torch.isfinite(right).all()):
            raise RuntimeError('non-finite checkpoint tensor: ' + key)
        difference = right - left
        if bool(torch.count_nonzero(difference)):
            changed += 1
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max()))
        squared_l2 += float(difference.square().sum())
    return changed, maximum, math.sqrt(squared_l2)


def audit_trajectory(output_dir):
    checkpoints = [
        output_dir / ('transformer_%d.pth' % epoch)
        for epoch in range(10, 121, 10)
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise RuntimeError('missing checkpoints: %r' % missing)
    states = [torch.load(path, map_location='cpu') for path in checkpoints]
    reference_keys = tuple(states[0])
    for path, state in zip(checkpoints, states):
        if tuple(state) != reference_keys:
            raise RuntimeError('state-key drift at ' + str(path))
        for key, tensor in state.items():
            if not bool(torch.isfinite(tensor.detach().float()).all()):
                raise RuntimeError(
                    'non-finite checkpoint tensor at %s: %s'
                    % (path.name, key))

    group_keys = {}
    for group, prefixes in GROUP_PREFIXES.items():
        keys = selected_keys(states[0], prefixes)
        if not keys:
            raise RuntimeError('empty audit group: ' + group)
        group_keys[group] = keys

    for epoch, state in zip(range(20, 121, 10), states[1:]):
        parts = []
        for group, keys in group_keys.items():
            changed, maximum, l2 = compare_group(states[0], state, keys)
            if changed != len(keys) or maximum <= 0.0 or l2 <= 0.0:
                raise RuntimeError(
                    '%s did not fully move by e%d: %d/%d max=%g l2=%g'
                    % (group, epoch, changed, len(keys), maximum, l2))
            parts.append(
                '%s=%d/%d,max=%.9f,l2=%.9f'
                % (group, changed, len(keys), maximum, l2))
        print('trajectory e10->e%d %s' % (epoch, ' '.join(parts)))
    return states[-1]


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: to_device(item, device) for key, item in value.items()
        }
    return value


def subset_pose(pose_dict, count):
    return {
        key: value[:count] if isinstance(value, torch.Tensor) else value
        for key, value in pose_dict.items()
    }


def descriptor(model, image, camera, view, pose_dict):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        return model(
            image, cam_label=camera, view_label=view,
            pose_dict=pose_dict)[0].detach().clone()


def audit_final_eval(config, state):
    if not torch.cuda.is_available():
        raise RuntimeError('final eval parity requires CUDA')
    set_seed(int(config.SOLVER.SEED))
    loaders = make_dataloader(config)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    batch = next(iter(train_loader))
    image, _, camera, view, pose_dict = batch
    count = 2
    device = torch.device('cuda')
    image = image[:count].to(device, non_blocking=True)
    camera = camera[:count].to(device, non_blocking=True)
    view = view[:count].to(device, non_blocking=True)
    correct_pose = to_device(subset_pose(pose_dict, count), device)
    shuffled_pose = {
        key: (value.flip(0) if isinstance(value, torch.Tensor) else value)
        for key, value in correct_pose.items()
    }

    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    descriptors = {
        'correct': descriptor(
            model, image, camera, view, correct_pose),
        'shuffle': descriptor(
            model, image, camera, view, shuffled_pose),
        'none': descriptor(model, image, camera, view, None),
        'exploding': descriptor(
            model, image, camera, view, ExplodingPoseDict()),
    }
    expected = descriptors['correct']
    for name, observed in descriptors.items():
        if not bool(torch.isfinite(observed).all()):
            raise RuntimeError('non-finite descriptor: ' + name)
        if not torch.equal(observed, expected):
            error = float((observed - expected).abs().max())
            raise RuntimeError(
                'final external-pose parity failed for %s: %.9g'
                % (name, error))
    print(
        'final_eval_external_pose_exact_parity=PASS '
        'correct=shuffle=none=exploding descriptor_shape=%s'
        % (tuple(expected.shape),))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not cfg.MODEL.POSE_TAPF_HIERARCHICAL:
        raise RuntimeError('exp379 hierarchical mode is not enabled')
    if int(cfg.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('batch size drifted from 64')
    final_state = audit_trajectory(Path(args.output_dir))
    audit_final_eval(cfg, final_state)
    print('EXP379_FINAL_AUDIT_PASS')


if __name__ == '__main__':
    main()
