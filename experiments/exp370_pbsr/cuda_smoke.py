"""One-batch CUDA integration smoke test for exp370 PBSR.

This intentionally exercises the production dataloader, full model, standard
ID/triplet loss, PBSR route loss, CUDA AMP, and the production optimizer.  It
does not start an epoch runner or write checkpoints.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def finite_scalar(name, value):
    number = float(value.detach().float().item()) \
        if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError(f'{name} is not finite: {number}')
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        default='configs/occluded_duke/exp370_pbsr.yml',
    )
    parser.add_argument(
        '--amp-init-scale',
        type=float,
        default=None,
        help='Override the config GradScaler initial scale for diagnosis.',
    )
    parser.add_argument(
        '--baseline-diagnostic',
        action='store_true',
        help='Disable PBSR to isolate AMP overflow in the global baseline.',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this smoke test')

    cfg.merge_from_file(args.config_file)
    if args.baseline_diagnostic:
        cfg.MODEL.POSE_PBSR = False
    cfg.freeze()
    if not args.baseline_diagnostic and not cfg.MODEL.POSE_PBSR:
        raise RuntimeError('POSE_PBSR must be enabled')
    if int(cfg.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('exp370 smoke must preserve batch size 64')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device('cuda')
    set_seed(int(cfg.SOLVER.SEED))

    train_loader, _, _, _, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        cfg, model, center_criterion)

    # Freeze the paired stochastic stream after construction, as required by
    # the kill-switch manifest.  The first sampler iteration happens below.
    set_seed(int(cfg.SOLVER.SEED))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)

    img, target, target_cam, target_view, pose_dict = next(iter(train_loader))
    if not isinstance(img, torch.Tensor):
        raise RuntimeError('exp370 isolation must use a single image view')
    if img.shape[0] != int(cfg.SOLVER.IMS_PER_BATCH):
        raise RuntimeError(
            f'unexpected batch size {img.shape[0]} != {cfg.SOLVER.IMS_PER_BATCH}')
    img = img.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    target_cam = target_cam.to(device, non_blocking=True)
    target_view = target_view.to(device, non_blocking=True)
    pose_dict = to_device(pose_dict, device)

    amp_init_scale = float(cfg.SOLVER.AMP_INIT_SCALE) \
        if args.amp_init_scale is None else float(args.amp_init_scale)
    scaler = amp.GradScaler(enabled=True, init_scale=amp_init_scale)
    alpha_before = None if args.baseline_diagnostic else \
        float(model.pbsr.write_scale.detach().item())
    with amp.autocast(enabled=True):
        model_out = model(
            img,
            label=target,
            cam_label=target_cam,
            view_label=target_view,
            pose_dict=pose_dict,
        )
        expected_outputs = 4 if args.baseline_diagnostic else 5
        if len(model_out) != expected_outputs:
            raise RuntimeError(
                f'expected {expected_outputs} model outputs, got {len(model_out)}')
        if args.baseline_diagnostic:
            score, feat, _, recon_loss = model_out
            kp_data = {}
        else:
            score, feat, _, recon_loss, kp_data = model_out
        if recon_loss is not None:
            raise RuntimeError('exp370 isolation unexpectedly returned recon loss')
        if not args.baseline_diagnostic and (
                not isinstance(kp_data, dict)
                or 'pbsr_route_loss' not in kp_data):
            raise RuntimeError('PBSR route loss is missing from model output')
        identity_loss = loss_func(score, feat, target, target_cam)
        route_loss = identity_loss.new_zeros(()) if args.baseline_diagnostic \
            else kp_data['pbsr_route_loss']
        total_loss = identity_loss if args.baseline_diagnostic else \
            identity_loss + \
            float(cfg.MODEL.POSE_PBSR_ROUTE_WEIGHT) * route_loss

    identity_value = finite_scalar('identity_loss', identity_loss)
    route_value = finite_scalar('pbsr_route_loss', route_loss)
    total_value = finite_scalar('total_loss', total_loss)
    stats = kp_data.get('pbsr_stats', {})
    required_stats = (
        'write_scale', 'route_entropy', 'background_share', 'delta_norm',
        'input_norm', 'slot_norm',
    )
    stat_values = {} if args.baseline_diagnostic else {
        name: finite_scalar(name, stats[name]) for name in required_stats
    }

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)

    finite_grad_params = 0
    nonzero_grad_params = 0
    nonfinite_grad_params = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite_grad_params += 1
        if not torch.isfinite(parameter.grad).all():
            nonfinite_grad_params.append(name)
            continue
        if bool(torch.count_nonzero(parameter.grad)):
            nonzero_grad_params += 1
    if nonfinite_grad_params:
        preview = ', '.join(nonfinite_grad_params[:8])
        raise RuntimeError(
            f'non-finite gradients at AMP scale {scaler.get_scale():.1f} '
            f'({len(nonfinite_grad_params)} params): {preview}')

    required_grads = {
        'backbone.patch_embed': model.base.patch_embed.projection.weight,
    }
    if not args.baseline_diagnostic:
        required_grads.update({
            'pbsr.write_scale': model.pbsr.write_scale,
            'pbsr.slot_queries': model.pbsr.slot_queries,
            'pbsr.key_proj.weight': model.pbsr.key_proj.weight,
        })
    grad_norms = {}
    for name, parameter in required_grads.items():
        if parameter.grad is None:
            raise RuntimeError(f'missing required gradient: {name}')
        norm = float(parameter.grad.detach().float().norm().item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f'invalid required gradient {name}: {norm}')
        grad_norms[name] = norm

    scaler.step(optimizer)
    scaler.update()
    alpha_after = None if args.baseline_diagnostic else \
        float(model.pbsr.write_scale.detach().item())
    if not args.baseline_diagnostic and (
            not math.isfinite(alpha_after) or alpha_after == alpha_before):
        raise RuntimeError(
            f'write scale did not update: {alpha_before} -> {alpha_after}')

    mode = 'baseline AMP diagnostic' if args.baseline_diagnostic \
        else 'PBSR CUDA integration smoke'
    print(f'{mode}: PASS')
    print(f'batch={img.shape[0]} image={tuple(img.shape)}')
    print(
        f'loss identity={identity_value:.8f} route={route_value:.8f} '
        f'total={total_value:.8f}')
    if stat_values:
        print(
            'stats ' + ' '.join(
                f'{name}={value:.8f}' for name, value in stat_values.items()))
    print(
        f'grads finite_params={finite_grad_params} '
        f'nonzero_params={nonzero_grad_params} ' + ' '.join(
            f'{name}={value:.8e}' for name, value in grad_norms.items()))
    if not args.baseline_diagnostic:
        print(f'write_scale {alpha_before:.8e} -> {alpha_after:.8e}')


if __name__ == '__main__':
    main()
