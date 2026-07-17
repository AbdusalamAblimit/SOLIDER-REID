"""PyTorch 1.13 production gate for exp380 ResNet B0/D0/HT0.

This script uses real Occluded-Duke batch64 data.  It validates matched model
initialization and optimizer membership, the legacy ResNet processor bridge,
the two TAPF routes at e1/e11, objective ownership, pose-free evaluation, a
paired ten-step legacy-control trajectory, and a real AMP overflow skip.  It
never invokes the epoch runner or writes a training checkpoint.
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

from config import cfg as root_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from processor.processor import (
    _eval_feature_from_output,
    _plain_train_score_feature,
)
from solver import make_optimizer


CONFIGS = {
    'b0': ROOT / 'configs/occluded_duke/exp380_r50_b0.yml',
    'd0': ROOT / 'configs/occluded_duke/exp380_r50_d0.yml',
    'ht0': ROOT / 'configs/occluded_duke/exp380_r50_ht0.yml',
}


class ExplodingPoseDict(dict):
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
    torch.backends.cudnn.benchmark = False


def load_config(name):
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(CONFIGS[name]))
    config.freeze()
    if config.MODEL.NAME != 'resnet50':
        raise RuntimeError(name + ' is not ResNet-50')
    if int(config.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError(name + ' changed batch64')
    if int(config.SOLVER.MAX_EPOCHS) != 120:
        raise RuntimeError(name + ' changed 120 epochs')
    if not Path(config.MODEL.PRETRAIN_PATH).is_file():
        raise RuntimeError(
            '%s pretrained checkpoint is missing: %s'
            % (name, config.MODEL.PRETRAIN_PATH))
    return config


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device)
                for key, item in value.items()}
    return value


def finite_scalar(name, value):
    number = (float(value.detach().float().item())
              if isinstance(value, torch.Tensor) else float(value))
    if not math.isfinite(number):
        raise RuntimeError('%s is non-finite: %r' % (name, number))
    return number


def grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('non-finite production gradient')
        total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def objective_grad_norm(objective, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('non-finite independent gradient')
        total += float(gradient.detach().float().square().sum())
    return math.sqrt(total)


def snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def delta(before, parameters):
    total = 0.0
    for expected, parameter in zip(before, parameters):
        total += float((parameter.detach().cpu().float()
                        - expected.float()).square().sum())
    return math.sqrt(total)


def optimizer_snapshot(optimizer):
    result = {}
    for parameter, state in optimizer.state.items():
        result[id(parameter)] = {
            key: (value.detach().cpu().clone()
                  if isinstance(value, torch.Tensor)
                  else copy.deepcopy(value))
            for key, value in state.items()
        }
    return result


def assert_optimizer_exact(expected, optimizer):
    observed_ids = {id(parameter) for parameter in optimizer.state}
    if set(expected) != observed_ids:
        raise RuntimeError('optimizer state membership changed')
    for parameter, state in optimizer.state.items():
        old = expected[id(parameter)]
        if set(old) != set(state):
            raise RuntimeError('optimizer state keys changed')
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.equal(value.detach().cpu(), old[key]):
                    raise RuntimeError('optimizer tensor state changed: ' + key)
            elif value != old[key]:
                raise RuntimeError('optimizer scalar state changed: ' + key)


def build(name, meta, device=None):
    config = load_config(name)
    set_seed(int(config.SOLVER.SEED))
    model = make_model(
        config, num_class=meta[0], camera_num=meta[1], view_num=meta[2],
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT)
    rng = torch.get_rng_state().clone()
    if device is not None:
        model = model.to(device)
    return config, model, rng


def optimizer_groups_by_name(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for group in optimizer.param_groups:
        if len(group['params']) != 1:
            raise RuntimeError('expected one parameter per optimizer group')
        parameter = group['params'][0]
        name = names.get(id(parameter))
        if name is None or name in result:
            raise RuntimeError('optimizer membership is invalid')
        result[name] = (float(group['lr']), float(group['weight_decay']))
    expected = {name for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    if set(result) != expected:
        raise RuntimeError('optimizer omitted or duplicated a parameter')
    return result


def validate_matched_invariants(meta):
    built = {name: build(name, meta) for name in CONFIGS}
    configs = {name: values[0] for name, values in built.items()}
    models = {name: values[1] for name, values in built.items()}
    rngs = {name: values[2] for name, values in built.items()}
    if not torch.equal(rngs['b0'], rngs['d0']) \
            or not torch.equal(rngs['b0'], rngs['ht0']):
        raise RuntimeError('matched model construction changed the RNG stream')

    shared = [key for key in models['b0'].state_dict()
              if key.startswith(('base.', 'bottleneck.', 'classifier.'))]
    if not shared:
        raise RuntimeError('shared ResNet state is empty')
    for key in shared:
        expected = models['b0'].state_dict()[key]
        for name in ('d0', 'ht0'):
            if not torch.equal(expected, models[name].state_dict()[key]):
                raise RuntimeError('shared initialization mismatch: ' + key)

    d0_psg = models['d0'].psg_modules_dict.state_dict()
    ht0_psg = models['ht0'].psg_modules_dict.state_dict()
    stage4 = [key for key in d0_psg if key.startswith('s3_')]
    if not stage4:
        raise RuntimeError('D0 Stage-4 PSG is empty')
    for key in stage4:
        if not torch.equal(d0_psg[key], ht0_psg[key]):
            raise RuntimeError('D0/HT0 Stage-4 PSG mismatch: ' + key)

    groups = {}
    for name, model in models.items():
        _, center = make_loss(configs[name], num_classes=meta[0])
        optimizer, _ = make_optimizer(configs[name], model, center)
        groups[name] = optimizer_groups_by_name(model, optimizer)
    for key in shared:
        if key not in groups['b0']:
            continue
        expected = groups['b0'][key]
        if groups['d0'][key] != expected or groups['ht0'][key] != expected:
            raise RuntimeError('shared optimizer mismatch: ' + key)
    counts = {name: sum(parameter.numel() for parameter in model.parameters())
              for name, model in models.items()}
    print('MATCHED_INVARIANTS_PASS counts=%r' % counts)
    del built, configs, models, groups


def validate_b0_step(batch, meta, device):
    config, model, _ = build('b0', meta, device)
    loss_func, center = make_loss(config, num_classes=meta[0])
    optimizer, optimizer_center = make_optimizer(config, model, center)
    image, target, camera, view = batch
    backbone = [model.base.conv1.weight]
    before = snapshot(backbone)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE),
        growth_interval=100000)
    with amp.autocast(enabled=True):
        output = model(
            image, label=target, cam_label=camera, view_label=view)
        if len(output) != 2:
            raise RuntimeError('legacy ResNet B0 did not return two train items')
        score, feature = _plain_train_score_feature(output)
        loss = loss_func(score, feature, target, camera)
    finite_scalar('b0_loss', loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if grad_norm(backbone) <= 0.0:
        raise RuntimeError('B0 loss missed the ResNet backbone')
    scaler.step(optimizer)
    scaler.update()
    if delta(before, backbone) <= 0.0:
        raise RuntimeError('B0 optimizer did not update the backbone')
    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        direct = model(image[:2], cam_label=camera[:2], view_label=view[:2])
        extracted = _eval_feature_from_output(direct)
    if not isinstance(direct, torch.Tensor) or not torch.equal(direct, extracted):
        raise RuntimeError('legacy ResNet eval bridge changed the descriptor')
    print('B0_BATCH64_PASS loss=%.8f scale=%.1f'
          % (finite_scalar('b0_loss', loss), scaler.get_scale()))
    del model, optimizer, optimizer_center, loss
    torch.cuda.empty_cache()


def capture_eval(model, image, camera, view, pose):
    with torch.no_grad(), amp.autocast(enabled=True):
        return _eval_feature_from_output(model(
            image, cam_label=camera, view_label=view,
            pose_dict=pose)).detach().clone()


def validate_eval_pose_independence(model, batch):
    image, _, camera, view, pose = batch
    count = 2
    correct = {key: (value[:count] if isinstance(value, torch.Tensor)
                     else value) for key, value in pose.items()}
    shuffled = {key: (value.flip(0) if isinstance(value, torch.Tensor)
                      else value) for key, value in correct.items()}
    model.eval()
    descriptors = {
        'correct': capture_eval(
            model, image[:count], camera[:count], view[:count], correct),
        'shuffle': capture_eval(
            model, image[:count], camera[:count], view[:count], shuffled),
        'none': capture_eval(
            model, image[:count], camera[:count], view[:count], None),
        'exploding': capture_eval(
            model, image[:count], camera[:count], view[:count],
            ExplodingPoseDict()),
    }
    expected = descriptors['correct']
    for name, observed in descriptors.items():
        if not bool(torch.isfinite(observed).all()) \
                or not torch.equal(observed, expected):
            raise RuntimeError('eval pose parity failed: ' + name)


def validate_pose_step(name, batch, meta, epoch, device):
    config, model, _ = build(name, meta, device)
    loss_func, center = make_loss(config, num_classes=meta[0])
    optimizer, optimizer_center = make_optimizer(config, model, center)
    image, target, camera, view, pose = batch
    stages = sorted(model.psg_stage_indices)
    tapf = list(model.tapf.parameters())
    psg = list(model.psg_modules_dict.parameters())
    backbone = [model.base.conv1.weight]
    before = {
        'tapf': snapshot(tapf),
        'psg': snapshot(psg),
        'backbone': snapshot(backbone),
    }
    raw_fields = {}
    handles = []
    for stage in stages:
        key = 's%d_b0' % stage

        def field_hook(_module, inputs, stage=stage):
            raw_fields[stage] = inputs[2].detach().clone()

        handles.append(model.psg_modules_dict[key].register_forward_pre_hook(
            field_hook))

    model.train()
    model.set_tapf_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE),
        growth_interval=100000)
    try:
        with amp.autocast(enabled=True):
            score, feature, _, recon, data = model(
                image, label=target, cam_label=camera, view_label=view,
                pose_dict=pose)
            if recon is not None:
                raise RuntimeError(name + ' unexpectedly returned recon loss')
            identity_loss = loss_func(score, feature, target, camera)
            pose_loss = data['tapf_pose_loss']
            total_loss = (identity_loss + float(
                config.MODEL.POSE_TAPF_LOSS_WEIGHT) * pose_loss)
    finally:
        for handle in handles:
            handle.remove()
    if set(raw_fields) != set(stages):
        raise RuntimeError(name + ' did not route every PSG bank')
    teacher = (pose['heatmaps'][:, 0]
               * pose['person_mask'][:, 0, None, None, None].float())
    if epoch == 1:
        for stage, field in raw_fields.items():
            if not torch.equal(field, teacher.to(field.dtype)):
                raise RuntimeError('%s e1 stage%d is not exact teacher'
                                   % (name, stage))
    elif epoch == 11:
        for stage, field in raw_fields.items():
            if torch.equal(field, teacher.to(field.dtype)):
                raise RuntimeError('%s e11 stage%d still reads teacher'
                                   % (name, stage))
        if name == 'ht0' and torch.equal(raw_fields[2], raw_fields[3]):
            raise RuntimeError('HT0 e11 reused one field for both anchors')
    else:
        raise RuntimeError('pose preflight only accepts e1/e11')
    finite_scalar(name + '_identity_loss', identity_loss)
    finite_scalar(name + '_pose_loss', pose_loss)
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    gradients = {
        'tapf': grad_norm(tapf),
        'psg': grad_norm(psg),
        'backbone': grad_norm(backbone),
    }
    if any(value <= 0.0 for value in gradients.values()):
        raise RuntimeError('%s missing production gradient: %r'
                           % (name, gradients))
    model.prepare_tapf_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        model.finish_tapf_optimizer_step()
    scaler.update()
    deltas = {
        'tapf': delta(before['tapf'], tapf),
        'psg': delta(before['psg'], psg),
        'backbone': delta(before['backbone'], backbone),
    }
    if any(value <= 0.0 for value in deltas.values()):
        raise RuntimeError('%s missing optimizer update: %r'
                           % (name, deltas))

    # Independent objective ownership on a small real-data slice, after the
    # batch64 production step has already proved the real memory/numerics path.
    model.train()
    model.set_tapf_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score_s, feature_s, _, _, data_s = model(
            image[:2], label=target[:2], cam_label=camera[:2],
            view_label=view[:2],
            pose_dict={key: (value[:2] if isinstance(value, torch.Tensor)
                            else value) for key, value in pose.items()})
        reid_objective = score_s.float().square().mean() \
            + feature_s.float().square().mean()
        pose_objective = data_s['tapf_pose_loss']
    ownership = {
        'pose_to_tapf': objective_grad_norm(pose_objective, tapf),
        'pose_to_psg': objective_grad_norm(pose_objective, psg),
        'pose_to_backbone': objective_grad_norm(pose_objective, backbone),
        'reid_to_tapf': objective_grad_norm(reid_objective, tapf),
        'reid_to_psg': objective_grad_norm(reid_objective, psg),
        'reid_to_backbone': objective_grad_norm(reid_objective, backbone),
    }
    if ownership['pose_to_tapf'] <= 0.0 \
            or ownership['pose_to_psg'] != 0.0 \
            or ownership['pose_to_backbone'] != 0.0 \
            or ownership['reid_to_tapf'] != 0.0 \
            or ownership['reid_to_psg'] <= 0.0 \
            or ownership['reid_to_backbone'] <= 0.0:
        raise RuntimeError('%s objective ownership failed: %r'
                           % (name, ownership))
    validate_eval_pose_independence(model, batch)
    result = {
        'model': model,
        'optimizer': optimizer,
        'loss_func': loss_func,
        'identity': finite_scalar(name + '_identity', identity_loss),
        'pose': finite_scalar(name + '_pose', pose_loss),
        'gradients': gradients,
        'deltas': deltas,
        'ownership': ownership,
        'scale': float(scaler.get_scale()),
    }
    print('%s_E%d_BATCH64_PASS identity=%.8f pose=%.8f scale=%.1f'
          % (name.upper(), epoch, result['identity'], result['pose'],
             result['scale']))
    return result


def validate_real_overflow(config, batch, successful):
    model = successful['model']
    optimizer = successful['optimizer']
    loss_func = successful['loss_func']
    image, target, camera, view, pose = batch
    model.train()
    model.set_tapf_epoch(11)
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score, feature, _, _, data = model(
            image, label=target, cam_label=camera, view_label=view,
            pose_dict=pose)
        total_loss = (loss_func(score, feature, target, camera)
                      + float(config.MODEL.POSE_TAPF_LOSS_WEIGHT)
                      * data['tapf_pose_loss'])
    scaler = amp.GradScaler(
        enabled=True, init_scale=128.0, growth_interval=100000)
    scaler.scale(total_loss).backward()
    injected = False
    for parameter in model.tapf.parameters():
        if parameter.grad is not None:
            parameter.grad.view(-1)[0] = float('inf')
            injected = True
            break
    if not injected:
        raise RuntimeError('failed to inject a real HT0 overflow')
    parameters = list(model.parameters())
    before_parameters = snapshot(parameters)
    before_optimizer = optimizer_snapshot(optimizer)
    before_scale = float(scaler.get_scale())
    model.prepare_tapf_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        model.finish_tapf_optimizer_step()
    scaler.update()
    after_scale = float(scaler.get_scale())
    if (before_scale, after_scale) != (128.0, 64.0):
        raise RuntimeError('overflow did not reduce scale 128->64')
    for index, (expected, parameter) in enumerate(
            zip(before_parameters, parameters)):
        if not torch.equal(expected, parameter.detach().cpu()):
            raise RuntimeError('overflow changed model parameter %d' % index)
    assert_optimizer_exact(before_optimizer, optimizer)
    print('HT0_REAL_OVERFLOW_PASS scale=128->64')


def validate_b0_ten_step_parity(meta, device):
    config_a, model_a, _ = build('b0', meta, device)
    config_b, model_b, _ = build('b0', meta, device)
    loss_a, center_a = make_loss(config_a, num_classes=meta[0])
    loss_b, center_b = make_loss(config_b, num_classes=meta[0])
    optimizer_a, _ = make_optimizer(config_a, model_a, center_a)
    optimizer_b, _ = make_optimizer(config_b, model_b, center_b)
    generator = torch.Generator().manual_seed(380)
    image = torch.randn(8, 3, 64, 32, generator=generator).to(device)
    target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    camera = torch.zeros(8, dtype=torch.long, device=device)
    for step in range(10):
        for model, optimizer, loss_func, bridge in (
                (model_a, optimizer_a, loss_a,
                 lambda output: (output[0], output[1])),
                (model_b, optimizer_b, loss_b,
                 _plain_train_score_feature)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            score, feature = bridge(model(image, label=target))
            loss = loss_func(score, feature, target, camera)
            loss.backward()
            optimizer.step()
        for key, value in model_a.state_dict().items():
            if not torch.equal(value, model_b.state_dict()[key]):
                raise RuntimeError(
                    'B0 ten-step processor parity diverged at step %d: %s'
                    % (step + 1, key))
    print('B0_TEN_STEP_LEGACY_PARITY_PASS')
    del model_a, model_b, optimizer_a, optimizer_b
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(args.device)
    configs = {name: load_config(name) for name in CONFIGS}
    if torch.__version__.split('+')[0] != '1.13.1':
        raise RuntimeError('production gate requires PyTorch 1.13.1')

    set_seed(int(configs['b0'].SOLVER.SEED))
    b0_loaders = make_dataloader(configs['b0'])
    b0_meta = b0_loaders[4:7]
    b0_batch_cpu = next(iter(b0_loaders[0]))
    if b0_batch_cpu[0].shape[0] != 64:
        raise RuntimeError('B0 loader did not return batch64')
    b0_batch = tuple(to_device(value, device) for value in b0_batch_cpu)

    set_seed(int(configs['ht0'].SOLVER.SEED))
    pose_loaders = make_dataloader(configs['ht0'])
    pose_meta = pose_loaders[4:7]
    pose_batch_cpu = next(iter(pose_loaders[0]))
    if pose_batch_cpu[0].shape[0] != 64:
        raise RuntimeError('pose loader did not return batch64')
    pose_batch = tuple(to_device(value, device) for value in pose_batch_cpu)
    if tuple(b0_meta) != tuple(pose_meta):
        raise RuntimeError('B0 and pose loaders disagree on dataset metadata')
    meta = tuple(b0_meta)

    validate_matched_invariants(meta)
    validate_b0_step(b0_batch, meta, device)
    validate_b0_ten_step_parity(meta, device)
    for name in ('d0', 'ht0'):
        for epoch in (1, 11):
            result = validate_pose_step(
                name, pose_batch, meta, epoch, device)
            if name == 'ht0' and epoch == 11:
                ht0_e11 = result
            else:
                del result['model'], result['optimizer'], result['loss_func']
                torch.cuda.empty_cache()
    validate_real_overflow(configs['ht0'], pose_batch, ht0_e11)
    del ht0_e11
    torch.cuda.empty_cache()
    print('EXP380_RESNET_CUDA_PREFLIGHT_PASS')
    print('runtime=torch-%s cuda=%s batch=64 image=%s'
          % (torch.__version__, torch.version.cuda,
             tuple(pose_batch[0].shape)))


if __name__ == '__main__':
    main()
