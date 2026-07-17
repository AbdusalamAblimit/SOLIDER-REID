"""Emit a deterministic 10-step production D0 snapshot.

Run this same file in independent parent/current repositories under the same
PyTorch 1.13.1 CUDA runtime.  D0's detached-field design permits an exact
factorization: the full CUDA run proves the ReID/PSG path exact, while the
isolated pose module consumes the captured real Stage-2 feature and performs
the same ten optimizer steps on deterministic CPU.  This avoids pretending
that CUDA bilinear-upsample backward is deterministic; its harmless ULP-scale
jitter is reported separately and excluded from the exact digest.  The script
reuses one real batch64 and writes no checkpoint.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device)
                for key, item in value.items()}
    return value


def update_tensor(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode('utf-8'))
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(tuple(value.shape)).encode('ascii'))
    digest.update(value.numpy().tobytes(order='C'))


def model_digest(model, include=None):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if include is not None and not include(name):
            continue
        update_tensor(digest, name, value)
    return digest.hexdigest()


def optimizer_digest(model, optimizer, include=None):
    digest = hashlib.sha256()
    parameter_names = {id(parameter): name
                       for name, parameter in model.named_parameters()}
    for parameter, state in sorted(
            optimizer.state.items(),
            key=lambda item: parameter_names[id(item[0])]):
        name = parameter_names[id(parameter)]
        if include is not None and not include(name):
            continue
        digest.update(name.encode('utf-8'))
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                update_tensor(digest, key, value)
            else:
                digest.update(('%s=%r' % (key, value)).encode('utf-8'))
    selected_group = 0
    for group in optimizer.param_groups:
        names = [parameter_names[id(parameter)]
                 for parameter in group['params']]
        if include is not None:
            names = [name for name in names if include(name)]
        if not names:
            continue
        digest.update(('group=%d' % selected_group).encode('ascii'))
        selected_group += 1
        for name in names:
            digest.update(name.encode('utf-8'))
        for key in sorted(key for key in group if key != 'params'):
            digest.update(('%s=%r' % (key, group[key])).encode('utf-8'))
    return digest.hexdigest()


def tensor_group_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        update_tensor(digest, name, value)
    return digest.hexdigest()


def make_named_sgd(module, config):
    groups = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        bias = 'bias' in name
        groups.append({
            'params': [parameter],
            'lr': (float(config.SOLVER.BASE_LR)
                   * (float(config.SOLVER.BIAS_LR_FACTOR)
                      if bias else 1.0)),
            'weight_decay': float(
                config.SOLVER.WEIGHT_DECAY_BIAS if bias
                else config.SOLVER.WEIGHT_DECAY),
        })
    return torch.optim.SGD(
        groups, momentum=float(config.SOLVER.MOMENTUM))


def run_exact_cpu_tapf(module, captured, config, steps):
    feature, teacher, scores = captured
    module = copy.deepcopy(module).cpu().train()
    module.set_epoch(1)
    optimizer = make_named_sgd(module, config)
    trace = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, data = module(feature, teacher, scores)
        loss = data['tapf_pose_loss']
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError(
                'non-finite CPU TAPF loss at step %d' % step)
        loss.backward()
        optimizer.step()
        trace.append(value.hex())
    return {
        'input': tensor_group_digest((
            ('feature', feature), ('teacher', teacher), ('scores', scores))),
        'trace': trace,
        'model': model_digest(module),
        'optimizer': optimizer_digest(module, optimizer),
    }


def batch_digest(batch):
    digest = hashlib.sha256()
    img, target, camera, view, pose_dict = batch
    for name, value in (
            ('img', img), ('target', target), ('camera', camera),
            ('view', view)):
        update_tensor(digest, name, value)
    for name, value in sorted(pose_dict.items()):
        if isinstance(value, torch.Tensor):
            update_tensor(digest, 'pose.' + name, value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))

    from config import cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model import make_model
    from solver import make_optimizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg.merge_from_file(str(root / args.config_file))
    cfg.freeze()
    if not cfg.MODEL.POSE_TAPF \
            or str(cfg.MODEL.POSE_TAPF_MODE).lower() != 'd0':
        raise RuntimeError('legacy parity requires exp378 D0')
    if bool(getattr(cfg.MODEL, 'POSE_TAPF_HIERARCHICAL', False)):
        raise RuntimeError('legacy parity must not enable hierarchical TAPF')
    if int(cfg.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('legacy parity batch must remain 64')
    if args.steps != 10:
        raise RuntimeError('registered legacy parity requires exactly 10 steps')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device('cuda')

    set_seed(int(cfg.SOLVER.SEED))
    loaders = make_dataloader(cfg)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        cfg, model, center_criterion)

    set_seed(int(cfg.SOLVER.SEED))
    raw_batch = next(iter(train_loader))
    if raw_batch[0].shape[0] != 64:
        raise RuntimeError('legacy parity did not receive batch64')
    batch_hash = batch_digest(raw_batch)
    img, target, camera, view, pose_dict = raw_batch
    batch = (
        img.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        camera.to(device, non_blocking=True),
        view.to(device, non_blocking=True),
        to_device(pose_dict, device),
    )
    img, target, camera, view, pose_dict = batch

    set_seed(int(cfg.SOLVER.SEED) + 379)
    model.train()
    model.set_tapf_epoch(1)
    initial_tapf = copy.deepcopy(model.tapf).cpu()
    captured_tapf = []

    def capture_tapf(_module, inputs):
        if captured_tapf:
            return
        captured_tapf.append(tuple(
            value.detach().float().cpu().clone() for value in inputs[:3]))

    tapf_handle = model.tapf.register_forward_pre_hook(capture_tapf)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(cfg.SOLVER.AMP_INIT_SCALE),
        growth_interval=100000)
    trace = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        with amp.autocast(enabled=True):
            score, feature, _, recon_loss, data = model(
                img, label=target, cam_label=camera,
                view_label=view, pose_dict=pose_dict)
            if recon_loss is not None:
                raise RuntimeError('D0 parity unexpectedly used recon_loss')
            identity_loss = loss_func(score, feature, target, camera)
            pose_loss = data['tapf_pose_loss']
            total_loss = (identity_loss + float(
                cfg.MODEL.POSE_TAPF_LOSS_WEIGHT) * pose_loss)
        values = [float(value.detach().float()) for value in (
            identity_loss, pose_loss, total_loss)]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('non-finite D0 parity loss at step %d' % step)
        scale_before = float(scaler.get_scale())
        scaler.scale(total_loss).backward()
        model.prepare_tapf_optimizer_step(optimizer)
        try:
            scaler.step(optimizer)
        finally:
            model.finish_tapf_optimizer_step()
        scaler.update()
        trace.append({
            'step': step,
            'identity': values[0].hex(),
            'pose': values[1].hex(),
            'total': values[2].hex(),
            'scale_before': scale_before.hex(),
            'scale_after': float(scaler.get_scale()).hex(),
        })
    tapf_handle.remove()
    if len(captured_tapf) != 1 or len(captured_tapf[0]) != 3:
        raise RuntimeError('failed to capture the real D0 TAPF input')
    cpu_tapf = run_exact_cpu_tapf(
        initial_tapf, captured_tapf[0], cfg, args.steps)

    shared = lambda name: not name.startswith('tapf.')
    tapf = lambda name: name.startswith('tapf.')

    payload = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'steps': args.steps,
        'batch': batch_hash,
        'trace': trace,
        'cuda_shared_model': model_digest(model, include=shared),
        'cuda_shared_optimizer': optimizer_digest(
            model, optimizer, include=shared),
        # These two diagnostics are intentionally not part of the exact
        # signature because upsample_bilinear2d_backward_cuda is explicitly
        # nondeterministic in torch 1.13.1.
        'cuda_tapf_model_diagnostic': model_digest(model, include=tapf),
        'cuda_tapf_optimizer_diagnostic': optimizer_digest(
            model, optimizer, include=tapf),
        'cpu_tapf': cpu_tapf,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    exact_payload = {
        'torch': payload['torch'],
        'cuda': payload['cuda'],
        'steps': payload['steps'],
        'batch': payload['batch'],
        'identity_trace': [
            (item['step'], item['identity'], item['scale_before'],
             item['scale_after']) for item in trace],
        'cuda_shared_model': payload['cuda_shared_model'],
        'cuda_shared_optimizer': payload['cuda_shared_optimizer'],
        'cpu_tapf': cpu_tapf,
    }
    exact_encoded = json.dumps(
        exact_payload, sort_keys=True, separators=(',', ':'))
    print('EXP379_D0_LEGACY_PARITY_SNAPSHOT=' + encoded)
    print('EXP379_D0_LEGACY_PARITY_SHA256='
          + hashlib.sha256(exact_encoded.encode('utf-8')).hexdigest())


if __name__ == '__main__':
    main()
