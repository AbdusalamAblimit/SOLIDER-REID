"""CUDA regression gate for the RG0 extraction of the shared TAPF renderer.

This compares the candidate TaskAdaptivePoseField against its exact parent
source under CUDA autocast, then strict-loads the completed MR-F0/MR-P0
production checkpoints into the candidate full model.  It never trains or
writes a checkpoint.
"""
import argparse
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as default_cfg
from datasets import make_dataloader
from model import make_model
from model.modules.task_adaptive_pose_field import TaskAdaptivePoseField


CONFIGS = {
    'mrf0': 'configs/occluded_duke/exp378_mrf0_sgd_relax.yml',
    'mrp0': 'configs/occluded_duke/exp378_mrp0_sgd_relax.yml',
}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head():
    return subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def require_provenance(expected_head):
    observed = git_head()
    if observed != expected_head:
        raise RuntimeError(
            'unexpected HEAD: expected %s observed %s'
            % (expected_head, observed))
    status = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain',
         '--untracked-files=no'], text=True).strip()
    if status:
        raise RuntimeError('tracked worktree is dirty:\n%s' % status)


def legacy_class(base_ref):
    source = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show',
         '%s:model/modules/task_adaptive_pose_field.py' % base_ref],
        text=True)
    module = types.ModuleType('exp378_legacy_task_adaptive_pose_field')
    module.__file__ = '<git:%s:task_adaptive_pose_field.py>' % base_ref
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module.TaskAdaptivePoseField


def module(mode, transition, cls):
    return cls(
        in_channels=32,
        hidden_dim=16,
        output_size=(96, 32),
        mode=mode,
        anchor_transition=transition,
        boot_epochs=10,
        handoff_start_epoch=6,
    )


def inputs(device):
    set_seed(378)
    feature = torch.randn(2, 32, 24, 8, device=device)
    heatmaps = torch.randn(2, 17, 96, 32, device=device) * 0.01
    scores = torch.linspace(
        0.1, 0.9, 17, device=device).repeat(2, 1)
    for sample in range(2):
        for joint in range(17):
            y = 3 + (sample * 7 + joint * 5) % 90
            x = 1 + (sample * 3 + joint * 2) % 30
            heatmaps[sample, joint, y, x] = (
                scores[sample, joint] + 0.2)
    return feature, heatmaps, scores


def assert_tree_equal(observed, expected, label):
    if isinstance(expected, torch.Tensor):
        if not isinstance(observed, torch.Tensor):
            raise RuntimeError('%s type mismatch' % label)
        if not torch.equal(observed, expected):
            difference = float(
                (observed.detach().float()
                 - expected.detach().float()).abs().max())
            raise RuntimeError(
                '%s tensor mismatch max_abs=%g' % (label, difference))
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or observed.keys() != expected.keys():
            raise RuntimeError('%s dict keys mismatch' % label)
        for key in expected:
            assert_tree_equal(
                observed[key], expected[key], label + '/' + str(key))
        return
    if isinstance(expected, (list, tuple)):
        if (not isinstance(observed, type(expected))
                or len(observed) != len(expected)):
            raise RuntimeError('%s sequence mismatch' % label)
        for index, (observed_item, expected_item) in enumerate(
                zip(observed, expected)):
            assert_tree_equal(
                observed_item, expected_item,
                label + '/' + str(index))
        return
    if observed != expected:
        raise RuntimeError(
            '%s value mismatch: %r != %r' % (label, observed, expected))


def compare_tapf_source(base_ref, device):
    legacy = legacy_class(base_ref)
    cases = (
        ('f0', 'hard'), ('p0', 'hard'),
        ('d0', 'hard'), ('j0', 'hard'),
        ('f0', 'sgd_relax'), ('p0', 'sgd_relax'),
    )
    epochs = (1, 6, 10, 11)
    comparisons = 0
    for mode, transition in cases:
        set_seed(1234)
        old = module(mode, transition, legacy).to(device)
        old_cpu_rng = torch.get_rng_state().clone()
        old_cuda_rng = torch.cuda.get_rng_state_all()
        set_seed(1234)
        new = module(mode, transition, TaskAdaptivePoseField).to(device)
        if not torch.equal(torch.get_rng_state(), old_cpu_rng):
            raise RuntimeError('%s/%s changed CPU construction RNG'
                               % (mode, transition))
        for observed, expected in zip(
                torch.cuda.get_rng_state_all(), old_cuda_rng):
            if not torch.equal(observed, expected):
                raise RuntimeError('%s/%s changed CUDA construction RNG'
                                   % (mode, transition))
        old_state = old.state_dict()
        new_state = new.state_dict()
        if old_state.keys() != new_state.keys():
            raise RuntimeError('%s/%s state keys changed'
                               % (mode, transition))
        for key in old_state:
            if not torch.equal(old_state[key], new_state[key]):
                raise RuntimeError('%s/%s init differs: %s'
                                   % (mode, transition, key))
        new.load_state_dict(old_state, strict=True)

        feature, heatmaps, scores = inputs(device)
        for training in (False, True):
            old.train(training)
            new.train(training)
            for epoch in epochs:
                old.set_epoch(epoch)
                new.set_epoch(epoch)
                with torch.no_grad(), amp.autocast(enabled=True):
                    old_output = old(feature, heatmaps, scores)
                    new_output = new(feature, heatmaps, scores)
                assert_tree_equal(
                    new_output, old_output,
                    '%s/%s/train%d/e%d'
                    % (mode, transition, int(training), epoch))
                comparisons += 1
        del old, new
        torch.cuda.empty_cache()
    return comparisons


def config_for(name):
    config = default_cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / CONFIGS[name]))
    config.MODEL.PRETRAIN_CHOICE = 'none'
    config.MODEL.WITH_CP = False
    config.freeze()
    return config


def strict_load_checkpoints(old_repo, device):
    config = config_for('mrf0')
    loaders = make_dataloader(config)
    _, _, _, _, num_classes, camera_num, view_num = loaders
    results = {}
    for name in ('mrf0', 'mrp0'):
        config = config_for(name)
        checkpoint = (
            old_repo / 'log' / 'occluded_duke'
            / ('exp378_%s_sgd_relax_s1234' % name)
            / 'transformer_120.pth')
        if not checkpoint.is_file():
            raise RuntimeError('missing production checkpoint: %s' % checkpoint)
        set_seed(1234)
        model = make_model(
            config, num_class=num_classes, camera_num=camera_num,
            view_num=view_num,
            semantic_weight=config.MODEL.SEMANTIC_WEIGHT).to(device)
        state = torch.load(
            str(checkpoint), map_location='cpu', weights_only=False)
        model.load_state_dict(state, strict=True)
        for key, value in model.state_dict().items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(
                    '%s checkpoint has non-finite tensor: %s' % (name, key))
        results[name] = {
            'path': str(checkpoint),
            'sha256': file_sha(checkpoint),
            'state_tensors': len(state),
        }
        del model, state
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-head', required=True)
    parser.add_argument(
        '--base-ref',
        default='ca62c475b43f17564bb09ede90de6eed53dd2d88')
    parser.add_argument('--old-repo', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('RG0 TAPF regression requires CUDA')
    require_provenance(args.expected_head)
    device = torch.device('cuda')
    comparisons = compare_tapf_source(args.base_ref, device)
    checkpoints = strict_load_checkpoints(
        Path(args.old_repo).resolve(), device)
    print('TAPF_PRE_REFACTOR_CUDA_EXACT_PARITY_PASS comparisons=%d'
          % comparisons)
    for name in sorted(checkpoints):
        evidence = checkpoints[name]
        print('OLD_CHECKPOINT_STRICT_LOAD_PASS arm=%s tensors=%d sha256=%s'
              % (name.upper(), evidence['state_tensors'],
                 evidence['sha256']))
    print('RG0_TAPF_REGRESSION_PASS head=%s torch=%s cuda=%s'
          % (git_head(), torch.__version__, torch.version.cuda))


if __name__ == '__main__':
    main()
