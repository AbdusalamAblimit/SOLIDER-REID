"""Production batch64 CUDA/AMP gate for exp379 hierarchical TAPF.

The gate uses one real Occluded-Duke batch and the production model, loss,
optimizer, and AMP path.  It exercises both the exact-teacher endpoint (e1)
and the fully internal endpoint (e11), then proves a real GradScaler overflow
skips the complete optimizer step.  It never invokes the epoch runner or
writes a checkpoint.
"""
import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


class ExplodingPoseDict(dict):
    """Fail if an RGB-only evaluation tries to inspect external pose."""

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device)
                for key, item in value.items()}
    return value


def finite_scalar(name, value):
    number = float(value.detach().float().item()) \
        if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError('%s is non-finite: %r' % (name, number))
    return number


def assert_finite_tensors(name, tensors):
    for index, tensor in enumerate(tensors):
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise RuntimeError('%s[%d] is non-finite' % (name, index))


def grad_norm(parameters):
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('non-finite production gradient')
        total += float(parameter.grad.detach().float().square().sum().item())
        found = True
    return math.sqrt(total) if found else 0.0


def objective_grad_norm(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    if objective is None or not objective.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('non-finite independent objective gradient')
        total += float(gradient.detach().float().square().sum().item())
    return math.sqrt(total)


def parameter_snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def parameter_delta(before, parameters):
    total = 0.0
    for expected, parameter in zip(before, parameters):
        observed = parameter.detach().cpu()
        total += float((observed.float() - expected.float()).square().sum())
    return math.sqrt(total)


def optimizer_state_snapshot(optimizer):
    result = {}
    for parameter, state in optimizer.state.items():
        values = {}
        for key, value in state.items():
            values[key] = (value.detach().cpu().clone()
                           if isinstance(value, torch.Tensor)
                           else copy.deepcopy(value))
        result[id(parameter)] = values
    return result


def assert_optimizer_state_exact(before, optimizer):
    if set(before) != {id(parameter) for parameter in optimizer.state}:
        raise RuntimeError('overflow changed optimizer-state membership')
    for parameter, state in optimizer.state.items():
        expected = before[id(parameter)]
        if set(expected) != set(state):
            raise RuntimeError('overflow changed optimizer-state keys')
        for key, value in state.items():
            old = expected[key]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value.detach().cpu(), old):
                    raise RuntimeError(
                        'overflow changed optimizer tensor state: ' + key)
            elif value != old:
                raise RuntimeError(
                    'overflow changed optimizer scalar state: ' + key)


def subset_pose(pose_dict, count):
    return {
        key: value[:count] if isinstance(value, torch.Tensor) else value
        for key, value in pose_dict.items()
    }


def capture_eval_descriptor(model, img, camera, view, pose_dict):
    with torch.no_grad(), amp.autocast(enabled=True):
        return model(
            img, cam_label=camera, view_label=view,
            pose_dict=pose_dict)[0].detach().clone()


def validate_eval_pose_independence(model, img, camera, view, pose_dict):
    model.eval()
    count = 2
    correct_pose = subset_pose(pose_dict, count)
    shuffled_pose = {
        key: (value.flip(0) if isinstance(value, torch.Tensor) else value)
        for key, value in correct_pose.items()
    }
    descriptors = {
        'correct': capture_eval_descriptor(
            model, img[:count], camera[:count], view[:count], correct_pose),
        'shuffle': capture_eval_descriptor(
            model, img[:count], camera[:count], view[:count], shuffled_pose),
        'none': capture_eval_descriptor(
            model, img[:count], camera[:count], view[:count], None),
        'exploding': capture_eval_descriptor(
            model, img[:count], camera[:count], view[:count],
            ExplodingPoseDict()),
    }
    expected = descriptors['correct']
    assert_finite_tensors('eval_descriptor', descriptors.values())
    for name, observed in descriptors.items():
        if not torch.equal(observed, expected):
            raise RuntimeError(
                'eval external-pose parity failed for ' + name)


def validate_sigmoid_once(raw_field, encoded_input):
    resized = (F.interpolate(
        raw_field, size=encoded_input.shape[-2:], mode='bilinear',
        align_corners=False) if raw_field.shape[-2:]
        != encoded_input.shape[-2:] else raw_field)
    expected = torch.sigmoid(resized).float()
    if not torch.equal(encoded_input.float(), expected):
        error = float((encoded_input.float() - expected).abs().max())
        raise RuntimeError(
            'PSG did not resize and sigmoid its raw field exactly once: %g'
            % error)


def build_model(config, num_classes, camera_num, view_num, device):
    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    if not model.use_hierarchical_tapf:
        raise RuntimeError('exp379 preflight requires hierarchical TAPF')
    if model.tapf.source_stages != (1, 2):
        raise RuntimeError('unexpected hierarchical source stages')
    if model.psg_stage_indices != {2, 3}:
        raise RuntimeError('unexpected hierarchical PSG stages')
    if model.tapf.parameter_count >= 500000:
        raise RuntimeError('hierarchical TAPF exceeds the 0.5M gate')
    return model


def validate_successful_step(config, batch, dataset_meta, epoch, device):
    num_classes, camera_num, view_num = dataset_meta
    model = build_model(
        config, num_classes, camera_num, view_num, device)
    loss_func, center_criterion = make_loss(
        config, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        config, model, center_criterion)
    img, target, camera, view, pose_dict = batch

    projections = {
        stage: list(model.tapf.stage_projections[str(stage)].parameters())
        for stage in model.tapf.source_stages
    }
    decoder = list(model.tapf.anchor.parameters())
    psg_parameters = {
        stage: [parameter for name, parameter
                in model.psg_modules_dict.named_parameters()
                if name.startswith('s%d_' % stage)]
        for stage in (2, 3)
    }
    psg_final = {
        stage: [parameter for name, parameter
                in model.psg_modules_dict.named_parameters()
                if name.startswith('s%d_' % stage)
                and '.encoder.2.' in name]
        for stage in (2, 3)
    }
    backbone = [model.base.patch_embed.projection.weight]
    if any(not values for values in (
            list(projections.values()) + list(psg_parameters.values())
            + list(psg_final.values()))):
        raise RuntimeError('failed to enumerate hierarchical parameters')

    model.train()
    model.set_tapf_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    before = {
        'projection1': parameter_snapshot(projections[1]),
        'projection2': parameter_snapshot(projections[2]),
        'decoder': parameter_snapshot(decoder),
        'psg2': parameter_snapshot(psg_parameters[2]),
        'psg3': parameter_snapshot(psg_parameters[3]),
        'backbone': parameter_snapshot(backbone),
    }

    raw_fields = {}
    encoded_fields = {}
    handles = []
    for stage in (2, 3):
        key = 's%d_b0' % stage

        def raw_hook(_module, inputs, stage=stage):
            # Preserve the production dtype so the e1 teacher check also
            # proves the exact AMP cast at the PSG boundary.
            raw_fields[stage] = inputs[2].detach().clone()

        def encoded_hook(_module, inputs, stage=stage):
            encoded_fields[stage] = inputs[0].detach().float().clone()

        handles.append(model.psg_modules_dict[key].register_forward_pre_hook(
            raw_hook))
        handles.append(
            model.psg_modules_dict[key].encoder[0].register_forward_pre_hook(
                encoded_hook))

    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE),
        growth_interval=100000)
    try:
        with amp.autocast(enabled=True):
            output = model(
                img, label=target, cam_label=camera,
                view_label=view, pose_dict=pose_dict)
            if len(output) != 5:
                raise RuntimeError(
                    'hierarchical production forward must return five items')
            score, feature, _, recon_loss, data = output
            if recon_loss is not None:
                raise RuntimeError('hierarchical TAPF used recon_loss')
            identity_loss = loss_func(
                score, feature, target, camera)
            pose_loss = data.get('tapf_pose_loss')
            if pose_loss is None:
                raise RuntimeError(
                    'continuous hierarchical pose loss is missing')
            total_loss = (identity_loss + float(
                config.MODEL.POSE_TAPF_LOSS_WEIGHT) * pose_loss)
    finally:
        for handle in handles:
            handle.remove()

    if set(raw_fields) != {2, 3} or set(encoded_fields) != {2, 3}:
        raise RuntimeError('failed to capture both hierarchical PSG routes')
    assert_finite_tensors('raw_field', raw_fields.values())
    assert_finite_tensors('encoded_field', encoded_fields.values())
    validate_sigmoid_once(raw_fields[2], encoded_fields[2])
    validate_sigmoid_once(raw_fields[3], encoded_fields[3])

    teacher = (pose_dict['heatmaps'][:, 0].float()
               * pose_dict['person_mask'][
                   :, 0, None, None, None].float())
    if epoch == 1:
        for stage in (2, 3):
            expected = teacher.to(raw_fields[stage].dtype)
            if not torch.equal(raw_fields[stage], expected):
                error = float((raw_fields[stage] - expected).abs().max())
                raise RuntimeError(
                    'e1 Stage-%d PSG did not receive exact teacher: %g'
                    % (stage, error))
    elif epoch == 11:
        if torch.equal(raw_fields[2], raw_fields[3]):
            raise RuntimeError('e11 deeper field reused the shallower field')
        for stage in (2, 3):
            if torch.equal(raw_fields[stage], teacher.to(raw_fields[stage].dtype)):
                raise RuntimeError(
                    'e11 Stage-%d PSG still received external teacher' % stage)
    else:
        raise RuntimeError('exp379 preflight only accepts epoch 1 or 11')

    stats = data['tapf_stats']
    if finite_scalar(
            'hierarchical_stage_count',
            stats['hierarchical_stage_count']) != 2.0:
        raise RuntimeError('hierarchical loss did not aggregate two stages')
    stage_losses = []
    for source_stage in (1, 2):
        shape = stats['stage%d_shape_loss' % source_stage]
        confidence = stats['stage%d_confidence_loss' % source_stage]
        finite_scalar('stage%d_shape_loss' % source_stage, shape)
        finite_scalar('stage%d_confidence_loss' % source_stage, confidence)
        stage_losses.append(
            shape + float(config.MODEL.POSE_TAPF_CONF_LOSS_WEIGHT)
            * confidence)
    expected_pose_loss = torch.stack(stage_losses).mean()
    if not torch.equal(pose_loss.detach(), expected_pose_loss):
        error = float((pose_loss.detach() - expected_pose_loss).abs())
        raise RuntimeError(
            'total pose loss is not the strict two-stage mean: %g' % error)
    if finite_scalar(
            'stage1_refinement_active',
            stats['stage1_refinement_active']) != 0.0:
        raise RuntimeError('Stage-1 unexpectedly used a prior')
    if finite_scalar(
            'stage2_refinement_active',
            stats['stage2_refinement_active']) != 1.0:
        raise RuntimeError('Stage-2 did not refine the Stage-1 state')
    if finite_scalar(
            'stage2_posterior_refinement_l1',
            stats['stage2_posterior_refinement_l1']) <= 0.0:
        raise RuntimeError('Stage-2 posterior refinement is zero')

    pose_to_projection = {
        stage: objective_grad_norm(pose_loss, parameters)
        for stage, parameters in projections.items()
    }
    pose_to_decoder = objective_grad_norm(pose_loss, decoder)
    pose_to_psg = {
        stage: objective_grad_norm(pose_loss, parameters)
        for stage, parameters in psg_parameters.items()
    }
    pose_to_backbone = objective_grad_norm(pose_loss, backbone)
    reid_to_pose = objective_grad_norm(
        identity_loss, list(model.tapf.parameters()))
    reid_to_psg = {
        stage: objective_grad_norm(identity_loss, parameters)
        for stage, parameters in psg_final.items()
    }
    reid_to_backbone = objective_grad_norm(identity_loss, backbone)
    if any(value <= 0.0 for value in pose_to_projection.values()):
        raise RuntimeError(
            'pose loss missed a stage projection: %r' % pose_to_projection)
    if pose_to_decoder <= 0.0:
        raise RuntimeError('pose loss missed the shared decoder')
    if any(value != 0.0 for value in pose_to_psg.values()) \
            or pose_to_backbone != 0.0:
        raise RuntimeError(
            'pose loss escaped isolation: psg=%r backbone=%g'
            % (pose_to_psg, pose_to_backbone))
    if reid_to_pose != 0.0:
        raise RuntimeError('ReID loss rewrote the pose module: %g'
                           % reid_to_pose)
    if any(value <= 0.0 for value in reid_to_psg.values()):
        raise RuntimeError(
            'ReID loss missed a PSG consumer: %r' % reid_to_psg)
    if reid_to_backbone <= 0.0:
        raise RuntimeError('ReID loss missed the backbone')

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    gradient_norms = {
        'projection1': grad_norm(projections[1]),
        'projection2': grad_norm(projections[2]),
        'decoder': grad_norm(decoder),
        'psg2': grad_norm(psg_parameters[2]),
        'psg3': grad_norm(psg_parameters[3]),
        'backbone': grad_norm(backbone),
    }
    if any(value <= 0.0 for value in gradient_norms.values()):
        raise RuntimeError(
            'production gradient missed a component: %r' % gradient_norms)
    model.prepare_tapf_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        model.finish_tapf_optimizer_step()
    scaler.update()
    deltas = {
        'projection1': parameter_delta(before['projection1'], projections[1]),
        'projection2': parameter_delta(before['projection2'], projections[2]),
        'decoder': parameter_delta(before['decoder'], decoder),
        'psg2': parameter_delta(before['psg2'], psg_parameters[2]),
        'psg3': parameter_delta(before['psg3'], psg_parameters[3]),
        'backbone': parameter_delta(before['backbone'], backbone),
    }
    if any(value <= 0.0 for value in deltas.values()):
        raise RuntimeError(
            'production optimizer missed a component: %r' % deltas)

    validate_eval_pose_independence(
        model, img, camera, view, pose_dict)
    result = {
        'model': model,
        'optimizer': optimizer,
        'loss_func': loss_func,
        'identity_loss': finite_scalar('identity_loss', identity_loss),
        'pose_loss': finite_scalar('pose_loss', pose_loss),
        'total_loss': finite_scalar('total_loss', total_loss),
        'field_delta': float((raw_fields[2] - raw_fields[3]).abs().max()),
        'gradients': gradient_norms,
        'deltas': deltas,
    }
    return result


def validate_real_overflow(config, batch, successful, device):
    model = successful['model']
    optimizer = successful['optimizer']
    loss_func = successful['loss_func']
    img, target, camera, view, pose_dict = batch
    model.train()
    model.set_tapf_epoch(11)
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score, feature, _, _, data = model(
            img, label=target, cam_label=camera,
            view_label=view, pose_dict=pose_dict)
        identity_loss = loss_func(score, feature, target, camera)
        pose_loss = data['tapf_pose_loss']
        total_loss = (identity_loss + float(
            config.MODEL.POSE_TAPF_LOSS_WEIGHT) * pose_loss)
    scaler = amp.GradScaler(
        enabled=True, init_scale=128.0, growth_interval=100000)
    scaler.scale(total_loss).backward()
    injected = False
    for parameter in model.tapf.stage_projections['1'].parameters():
        if parameter.grad is not None:
            parameter.grad.view(-1)[0] = float('inf')
            injected = True
            break
    if not injected:
        raise RuntimeError('could not inject a real projection overflow')

    parameters = list(model.parameters())
    before_parameters = parameter_snapshot(parameters)
    before_optimizer = optimizer_state_snapshot(optimizer)
    scale_before = float(scaler.get_scale())
    model.prepare_tapf_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        model.finish_tapf_optimizer_step()
    scaler.update()
    scale_after = float(scaler.get_scale())
    if (scale_before, scale_after) != (128.0, 64.0):
        raise RuntimeError(
            'real GradScaler overflow was not 128->64: %g->%g'
            % (scale_before, scale_after))
    for index, (expected, parameter) in enumerate(
            zip(before_parameters, parameters)):
        if not torch.equal(parameter.detach().cpu(), expected):
            raise RuntimeError(
                'overflow changed model parameter %d' % index)
    assert_optimizer_state_exact(before_optimizer, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return scale_before, scale_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not cfg.MODEL.POSE_TAPF \
            or not cfg.MODEL.POSE_TAPF_HIERARCHICAL:
        raise RuntimeError('exp379 hierarchical TAPF must be enabled')
    if int(cfg.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('batch size must remain 64')
    if int(cfg.SOLVER.MAX_EPOCHS) != 120:
        raise RuntimeError('exp379 must remain 120 epochs')
    if float(cfg.INPUT.RE_PROB) != 0.0:
        raise RuntimeError('exp379 requires RE_PROB=0')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device('cuda')

    set_seed(int(cfg.SOLVER.SEED))
    loaders = make_dataloader(cfg)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    batch = next(iter(train_loader))
    img, target, camera, view, pose_dict = batch
    if not isinstance(img, torch.Tensor) or img.shape[0] != 64:
        raise RuntimeError('preflight did not receive a real batch64')
    batch = (
        img.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        camera.to(device, non_blocking=True),
        view.to(device, non_blocking=True),
        to_device(pose_dict, device),
    )
    dataset_meta = (num_classes, camera_num, view_num)

    epoch1 = validate_successful_step(
        cfg, batch, dataset_meta, 1, device)
    del epoch1['model'], epoch1['optimizer'], epoch1['loss_func']
    torch.cuda.empty_cache()
    epoch11 = validate_successful_step(
        cfg, batch, dataset_meta, 11, device)
    overflow = validate_real_overflow(cfg, batch, epoch11, device)

    print('EXP379_CUDA_PREFLIGHT_PASS')
    print('runtime=torch-%s cuda=%s batch=%d image=%s'
          % (torch.__version__, torch.version.cuda, batch[0].shape[0],
             tuple(batch[0].shape)))
    for epoch, result in ((1, epoch1), (11, epoch11)):
        print('epoch=%d loss identity=%.8f pose=%.8f total=%.8f '
              'field_delta=%.9f'
              % (epoch, result['identity_loss'], result['pose_loss'],
                 result['total_loss'], result['field_delta']))
        print('epoch=%d grad=%r delta=%r'
              % (epoch, result['gradients'], result['deltas']))
    print('eval_parity=correct/shuffle/None/exploding exact '
          'overflow_scale=%.1f->%.1f' % overflow)


if __name__ == '__main__':
    main()
