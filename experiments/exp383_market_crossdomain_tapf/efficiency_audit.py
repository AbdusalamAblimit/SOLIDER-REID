"""Matched B0/D0 efficiency audit for exp383.

The audit never enters the epoch runner or creates an output directory.  It
uses a real Market batch64 for AMP forward/backward/optimizer throughput and
peak memory, then measures the production FP32, pose-free, single-image eval
path.  FLOPs are the operations supported by ``torch.profiler`` and are only
compared under this identical profiler/runtime contract.
"""

import argparse
import copy
import gc
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as root_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


CONFIGS = {
    'b0': ROOT / 'configs' / 'market' / 'exp383_b0.yml',
    'd0': ROOT / 'configs' / 'market' / 'exp383_d0.yml',
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: to_cuda(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cuda(item) for item in value)
    return value


def load_config(arm):
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(CONFIGS[arm]))
    config.freeze()
    if int(config.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('efficiency audit requires batch64')
    if bool(config.MODEL.POSE_ENABLED) != (arm == 'd0'):
        raise RuntimeError('arm/config pose contract mismatch')
    return config


def unpack_batch(arm, batch):
    if arm == 'b0':
        if len(batch) != 4:
            raise RuntimeError('B0 batch contract changed')
        image, target, camera, view = batch
        pose_dict = None
    else:
        if len(batch) != 5:
            raise RuntimeError('D0 batch contract changed')
        image, target, camera, view, pose_dict = batch
    if image.shape != (64, 3, 384, 128):
        raise RuntimeError('unexpected production image batch: %s'
                           % (tuple(image.shape),))
    return tuple(to_cuda(value) for value in (
        image, target, camera, view, pose_dict))


def forward_train(model, arm, image, target, camera, view, pose_dict,
                  loss_func):
    if arm == 'b0':
        output = model(
            image, label=target, cam_label=camera, view_label=view)
        score, feature, _ = output
        aux = None
    else:
        output = model(
            image, label=target, cam_label=camera, view_label=view,
            pose_dict=pose_dict)
        score, feature, _, reconstruction_loss, aux = output
        if reconstruction_loss is not None:
            raise RuntimeError('D0 unexpectedly returned reconstruction loss')
    loss = loss_func(score, feature, target, camera)
    if aux is not None and aux.get('tapf_pose_loss') is not None:
        loss = loss + float(getattr(
            model, 'tapf_loss_weight', 1.0)) * aux['tapf_pose_loss']
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('non-finite efficiency-audit loss')
    return loss


def optimizer_step(model, arm, optimizer, optimizer_center, scaler, batch,
                   loss_func):
    image, target, camera, view, pose_dict = batch
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        loss = forward_train(
            model, arm, image, target, camera, view, pose_dict, loss_func)
    scaler.scale(loss).backward()
    if arm == 'd0':
        model.prepare_tapf_optimizer_step(optimizer, record_stats=False)
    try:
        scaler.step(optimizer)
    finally:
        if arm == 'd0':
            model.finish_tapf_optimizer_step()
    scaler.update()
    return loss.detach()


def forward_eval(model, arm, image, camera, view):
    if arm == 'd0':
        return model(
            image, cam_label=camera, view_label=view, pose_dict=None)
    return model(image, cam_label=camera, view_label=view)


def cuda_elapsed_ms(function, warmup, steps):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def audit_arm(arm, args):
    config = load_config(arm)
    set_seed(int(config.SOLVER.SEED))
    loaders = make_dataloader(config)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    batch = unpack_batch(arm, next(iter(train_loader)))

    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    if arm == 'd0':
        model.set_tapf_epoch(11)
    loss_func, center_criterion = make_loss(config, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        config, model, center_criterion)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE))

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)

    def step():
        return optimizer_step(
            model, arm, optimizer, optimizer_center, scaler, batch, loss_func)

    # Cost must not depend on an artificial parameter trajectory produced by
    # replaying one batch at a fresh epoch-11 state.  Restore the exact finite
    # snapshot outside every timed region, then time one complete AMP
    # forward/backward/optimizer microstep.  This includes optimizer kernels
    # but excludes state restoration.
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scaler_state = copy.deepcopy(scaler.state_dict())
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_state = torch.cuda.get_rng_state_all()

    def restore():
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        scaler.load_state_dict(scaler_state)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)

    for _ in range(args.train_warmup):
        restore()
        step()
    restore()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_ms = 0.0
    losses = []
    for _ in range(args.train_steps):
        restore()
        torch.cuda.synchronize()
        train_start = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)
        train_start.record()
        loss = step()
        train_end.record()
        torch.cuda.synchronize()
        train_ms += float(train_start.elapsed_time(train_end))
        losses.append(float(loss.cpu()))
    train_peak_allocated = float(
        torch.cuda.max_memory_allocated() / (1024 ** 2))
    train_peak_reserved = float(
        torch.cuda.max_memory_reserved() / (1024 ** 2))

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    image, _, camera, view, _ = batch
    image = image[:1]
    camera = camera[:1]
    view = view[:1]

    with torch.no_grad():
        latency_ms = cuda_elapsed_ms(
            lambda: forward_eval(model, arm, image, camera, view),
            args.infer_warmup, args.infer_steps) / args.infer_steps
        with profile(
                activities=[ProfilerActivity.CPU], with_flops=True) as trace:
            output = forward_eval(model, arm, image, camera, view)
            torch.cuda.synchronize()
    supported_flops = int(sum(
        event.flops for event in trace.key_averages()
        if event.flops is not None))
    feature = output[0] if isinstance(output, (tuple, list)) else output
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError('non-finite pose-free descriptor')

    report = {
        'arm': arm,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'torch_profiler_supported_flops': supported_flops,
        'torch_profiler_supported_gflops': supported_flops / 1e9,
        'train_batch': 64,
        'train_epoch_route': 11 if arm == 'd0' else None,
        'train_amp': True,
        'train_steps': args.train_steps,
        'train_step_ms': train_ms / args.train_steps,
        'train_images_per_second': (
            64.0 * args.train_steps / (train_ms / 1000.0)),
        'train_peak_allocated_mib': train_peak_allocated,
        'train_peak_reserved_mib': train_peak_reserved,
        'train_loss_first': losses[0],
        'train_loss_last': losses[-1],
        'amp_scale': float(scaler.get_scale()),
        'inference_batch': 1,
        'inference_fp32': True,
        'inference_pose_free': True,
        'inference_steps': args.infer_steps,
        'inference_ms_per_image': latency_ms,
        'config_sha256': file_sha(CONFIGS[arm]),
    }

    del output, trace, batch, model, model_state, loss_func, center_criterion
    del optimizer, optimizer_center, scaler, train_loader, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-warmup', type=int, default=3)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--infer-warmup', type=int, default=20)
    parser.add_argument('--infer-steps', type=int, default=100)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if min(args.train_warmup, args.train_steps,
           args.infer_warmup, args.infer_steps) <= 0:
        raise RuntimeError('all benchmark iteration counts must be positive')

    head = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    report = {
        'schema': 'exp383-efficiency-v1',
        'git_head': head,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
        'weight_sha256': file_sha(
            ROOT / 'pretrained' / 'swin_tiny.pth'),
        'flops_scope': 'torch.profiler supported FP32 pose-free eval ops',
        'training_scope': (
            'preloaded batch64 AMP full optimizer microstep; exact state '
            'restored outside each timed region'),
        'offline_pose_extraction_included': False,
        'arms': {arm: audit_arm(arm, args) for arm in ('b0', 'd0')},
    }
    b0 = report['arms']['b0']
    d0 = report['arms']['d0']
    report['delta'] = {
        'params': d0['total_params'] - b0['total_params'],
        'params_percent': 100.0 * (
            d0['total_params'] / b0['total_params'] - 1.0),
        'supported_flops': (
            d0['torch_profiler_supported_flops']
            - b0['torch_profiler_supported_flops']),
        'supported_flops_percent': 100.0 * (
            d0['torch_profiler_supported_flops']
            / b0['torch_profiler_supported_flops'] - 1.0),
        'train_peak_allocated_mib': (
            d0['train_peak_allocated_mib']
            - b0['train_peak_allocated_mib']),
        'train_throughput_percent': 100.0 * (
            d0['train_images_per_second']
            / b0['train_images_per_second'] - 1.0),
        'inference_latency_percent': 100.0 * (
            d0['inference_ms_per_image']
            / b0['inference_ms_per_image'] - 1.0),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    print('EXP383_EFFICIENCY_AUDIT_PASS')


if __name__ == '__main__':
    main()
