"""Production batch64 CUDA/AMP preflight and R0 parity snapshot for RG0.

Set ``RG0_REPO_ROOT`` to run the same R0 snapshot logic against an older exact
repository.  The script writes no checkpoints and performs exactly one
optimizer step in a fresh process.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('RG0_REPO_ROOT', str(SCRIPT_ROOT))).resolve()
sys.path.insert(0, str(ROOT))
# Configs intentionally keep production-relative data and pretrained paths.
# When snapshotting an older exact repository, resolve those paths against the
# selected repository too, rather than accidentally borrowing candidate files.
os.chdir(str(ROOT))

from config import cfg as default_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


CONFIGS = {
    'r0': 'configs/occluded_duke/exp378_r0_external_teacher.yml',
    'rg0': 'configs/occluded_duke/exp378_rg0_external_gaussian.yml',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def tensor_bytes(value):
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes(order='C')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(tensor_bytes(value))
    return digest.hexdigest()


def tensor_tree_sha(value):
    digest = hashlib.sha256()

    def update(item, prefix=''):
        if isinstance(item, torch.Tensor):
            digest.update(prefix.encode())
            digest.update(tensor_sha(item).encode())
        elif isinstance(item, dict):
            for key in sorted(item):
                update(item[key], prefix + '/' + str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                update(child, prefix + '/' + str(index))
        else:
            digest.update((prefix + '=' + repr(item)).encode())

    update(value)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(tensor_sha(state[key]).encode())
    return digest.hexdigest()


def optimizer_signature(optimizer, model):
    state = optimizer.state_dict()
    payload = {
        'param_groups': state['param_groups'],
        'state_keys': sorted(state['state']),
        'names': [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def grad_state_sha(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('non-finite gradient: %s' % name)
        digest.update(name.encode())
        digest.update(tensor_sha(parameter.grad).encode())
        count += 1
    return digest.hexdigest(), count


def parameter_delta(before, parameters):
    total = 0.0
    for expected, observed in zip(before, parameters):
        total += float((observed.detach().float() - expected).square().sum())
    return math.sqrt(total)


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def config_for(arm):
    config = default_cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / CONFIGS[arm]))
    config.MODEL.WITH_CP = False
    config.freeze()
    return config


def git_head():
    return subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def require_provenance(expected_head):
    observed_head = git_head()
    if observed_head != expected_head:
        raise RuntimeError(
            'unexpected HEAD: expected %s observed %s'
            % (expected_head, observed_head))
    tracked_status = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain',
         '--untracked-files=no'], text=True).strip()
    if tracked_status:
        raise RuntimeError(
            'preflight repository has tracked modifications:\n%s'
            % tracked_status)
    return observed_head


def compare_reference(report, reference_path):
    with Path(reference_path).open() as handle:
        reference = json.load(handle)
    if report['arm'] != 'r0' or reference.get('arm') != 'r0':
        raise RuntimeError('cross-commit exact comparison is R0-only')
    exact_keys = (
        'schema', 'torch', 'cuda', 'seed', 'batch_size',
        'preflight_script_sha256', 'cudnn_deterministic',
        'cudnn_benchmark',
        'batch_sha256', 'config_sha256', 'pretrain_sha256',
        'pose_train_index_sha256', 'initial_state_sha256',
        'optimizer_signature_sha256', 'descriptor_sha256',
        'featmaps_sha256', 'loss_sha256', 'gradient_sha256',
        'gradient_tensor_count', 'after_step_state_sha256',
        'target_heatmap_sha256', 'target_scores_sha256',
        'expected_field_sha256', 'amp_scale_before', 'amp_scale_after',
        'amp_step_skipped', 'psg',
    )
    mismatches = []
    for key in exact_keys:
        if report.get(key) != reference.get(key):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            'R0 cross-commit exact mismatch: %s' % ', '.join(mismatches))
    return reference.get('git_head')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('r0', 'rg0'), required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--reference-json')
    parser.add_argument('--report-json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('RG0 CUDA preflight requires CUDA')
    device = torch.device('cuda')
    observed_head = require_provenance(args.expected_head)
    config = config_for(args.arm)
    seed = int(config.SOLVER.SEED)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)

    loaders = make_dataloader(config)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    set_seed(seed)
    batch = next(iter(train_loader))
    image, target, camera, view, pose_dict = batch
    if not isinstance(image, torch.Tensor) or image.shape[0] != 64:
        raise RuntimeError('preflight requires a real production batch64')
    batch_sha = tensor_tree_sha(batch)
    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    camera = camera.to(device, non_blocking=True)
    view = view.to(device, non_blocking=True)
    pose_dict = to_device(pose_dict, device)

    set_seed(seed)
    model = make_model(
        config, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    model.train()
    initial_state_sha = state_sha(model.state_dict())
    loss_func, center_criterion = make_loss(config, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        config, model, center_criterion)
    optimizer_sha = optimizer_signature(optimizer, model)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE))

    captured = {key: {'field': [], 'encoder': []}
                for key in model.psg_modules_dict}
    handles = []
    for key, module in model.psg_modules_dict.items():
        handles.append(module.register_forward_pre_hook(
            lambda _module, inputs, key=key: captured[key]['field'].append(
                inputs[2].detach().clone())))
        handles.append(module.encoder.register_forward_pre_hook(
            lambda _module, inputs, key=key: captured[key]['encoder'].append(
                inputs[0].detach().clone())))

    final_parameters = []
    for module in model.psg_modules_dict.values():
        final_parameters.extend(module.encoder[-1].parameters())
    final_before = [parameter.detach().float().clone()
                    for parameter in final_parameters]
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with amp.autocast(enabled=True):
        output = model(
            image, label=target, cam_label=camera,
            view_label=view, pose_dict=pose_dict)
        if len(output) == 5:
            score, feature, featmaps, recon_loss, aux = output
        elif len(output) == 4:
            score, feature, featmaps, recon_loss = output
            aux = None
        else:
            raise RuntimeError('unexpected production model return length')
        if recon_loss is not None:
            raise RuntimeError('R0/RG0 must not return reconstruction loss')
        identity_loss = loss_func(score, feature, target, camera)
    for handle in handles:
        handle.remove()
    if not bool(torch.isfinite(identity_loss)):
        raise RuntimeError('non-finite identity loss')
    scaler.scale(identity_loss).backward()
    scaler.unscale_(optimizer)
    final_grad_norm = grad_norm(final_parameters)
    if not math.isfinite(final_grad_norm) or final_grad_norm <= 0:
        raise RuntimeError('PSG final projection gradient is not finite/nonzero')
    grad_sha, grad_count = grad_state_sha(model)
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    final_delta = parameter_delta(final_before, final_parameters)
    if not math.isfinite(final_delta) or final_delta <= 0:
        raise RuntimeError('PSG final projection did not update')
    amp_step_skipped = scale_after < scale_before
    if amp_step_skipped:
        raise RuntimeError(
            'AMP scaler skipped the production preflight step: %s -> %s'
            % (scale_before, scale_after))

    target_heatmaps = (pose_dict['heatmaps'][:, 0].float()
                       * pose_dict['person_mask'][:, 0:1, None, None].float())
    target_scores = (pose_dict['scores'][:, 0].float()
                     * pose_dict['person_mask'][:, 0:1].float())
    if args.arm == 'r0':
        expected_field = target_heatmaps
        external_stats = None
    else:
        if aux is None or 'external_field_stats' not in aux:
            raise RuntimeError('RG0 train forward did not return field stats')
        if list(model.external_field_renderer.parameters()):
            raise RuntimeError('RG0 renderer unexpectedly has parameters')
        if model.external_field_renderer.state_dict():
            raise RuntimeError('RG0 renderer unexpectedly has persistent state')
        expected_field, direct_stats = model.external_field_renderer(
            target_heatmaps, target_scores,
            reject_inconsistent_empty=True)
        external_stats = {
            key: float(value.item()) for key, value in direct_stats.items()
        }
        returned_stats = aux['external_field_stats']
        if returned_stats.keys() != direct_stats.keys():
            raise RuntimeError('RG0 returned/direct stats keys differ')
        for key in direct_stats:
            if not torch.equal(returned_stats[key], direct_stats[key]):
                raise RuntimeError('RG0 returned stat differs: %s' % key)

    psg_evidence = {}
    for key, values in captured.items():
        if len(values['field']) != 1 or len(values['encoder']) != 1:
            raise RuntimeError('PSG hook count mismatch: %s' % key)
        observed_field = values['field'][0]
        observed_encoder = values['encoder'][0]
        if observed_field.dtype != torch.float32:
            raise RuntimeError('PSG field must remain float32: %s' % key)
        if not torch.equal(observed_field, expected_field):
            raise RuntimeError('PSG did not receive exact expected field: %s'
                               % key)
        resized = F.interpolate(
            expected_field, size=observed_encoder.shape[-2:],
            mode='bilinear', align_corners=False)
        if not torch.equal(observed_encoder, torch.sigmoid(resized)):
            raise RuntimeError('PSG sigmoid boundary mismatch: %s' % key)
        psg_evidence[key] = {
            'field_sha256': tensor_sha(observed_field),
            'field_dtype': str(observed_field.dtype),
            'field_shape': list(observed_field.shape),
            'post_resize_sha256': tensor_sha(resized),
            'post_sigmoid_sha256': tensor_sha(observed_encoder),
            'post_resize_min': float(resized.min().item()),
            'post_resize_max': float(resized.max().item()),
            'post_sigmoid_min': float(observed_encoder.min().item()),
            'post_sigmoid_max': float(observed_encoder.max().item()),
        }

    after_state_sha = state_sha(model.state_dict())
    if after_state_sha == initial_state_sha:
        raise RuntimeError('one-step model state did not change')
    if args.arm == 'rg0':
        model.eval()
        try:
            with torch.no_grad(), amp.autocast(enabled=True):
                model(image[:2], cam_label=camera[:2], view_label=view[:2],
                      pose_dict=None)
        except ValueError as error:
            if 'requires target heatmaps/scores' not in str(error):
                raise
        else:
            raise RuntimeError('RG0 must retain external-pose dependence')

    config_path = ROOT / CONFIGS[args.arm]
    pretrain_path = ROOT / config.MODEL.PRETRAIN_PATH
    pose_train_index = (ROOT / config.MODEL.POSE_DATA_DIR / 'pose_data'
                        / 'train' / 'index.json')
    for label, path in (
            ('config', config_path), ('pretrain', pretrain_path),
            ('pose train index', pose_train_index)):
        if not path.is_file():
            raise RuntimeError('%s path is not a file: %s' % (label, path))

    report = {
        'schema': 'exp378-rg0-cuda-preflight-v1',
        'status': 'PASS',
        'arm': args.arm,
        'repo_root': str(ROOT),
        'git_head': observed_head,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'seed': seed,
        'preflight_script_sha256': file_sha(Path(__file__).resolve()),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'batch_size': int(image.shape[0]),
        'batch_sha256': batch_sha,
        'config_path': str(config_path),
        'config_sha256': file_sha(config_path),
        'pretrain_path': str(pretrain_path),
        'pretrain_sha256': file_sha(pretrain_path),
        'pose_train_index_path': str(pose_train_index),
        'pose_train_index_sha256': file_sha(pose_train_index),
        'initial_state_sha256': initial_state_sha,
        'optimizer_signature_sha256': optimizer_sha,
        'descriptor_sha256': tensor_sha(feature),
        'featmaps_sha256': tensor_tree_sha(featmaps),
        'loss': float(identity_loss.detach().float().item()),
        'loss_sha256': tensor_sha(identity_loss.detach().float()),
        'gradient_sha256': grad_sha,
        'gradient_tensor_count': grad_count,
        'psg_final_grad_norm': final_grad_norm,
        'psg_final_delta': final_delta,
        'amp_scale_before': scale_before,
        'amp_scale_after': scale_after,
        'amp_step_skipped': amp_step_skipped,
        'after_step_state_sha256': after_state_sha,
        'target_heatmap_sha256': tensor_sha(target_heatmaps),
        'target_scores_sha256': tensor_sha(target_scores),
        'expected_field_sha256': tensor_sha(expected_field),
        'external_field_stats': external_stats,
        'psg': psg_evidence,
        'peak_memory_mib': float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
    }
    if args.reference_json:
        reference_head = compare_reference(report, args.reference_json)
        report['reference_json'] = str(Path(args.reference_json).resolve())
        report['reference_git_head'] = reference_head
        report['reference_exact_match'] = True
    if args.report_json:
        output_path = Path(args.report_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    print('RG0_CUDA_PREFLIGHT_PASS arm=%s' % args.arm.upper())


if __name__ == '__main__':
    main()
