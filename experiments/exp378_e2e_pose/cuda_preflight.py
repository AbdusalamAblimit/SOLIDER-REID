"""One-batch production CUDA/AMP preflight for atomic TAPF.

This uses the configured real pose dataloader, production Swin/PSG model,
standard ID+triplet loss, optimizer, batch size 64, and the exact arm schedule.
It never enters the epoch runner and never writes a checkpoint. The historical
exp378 default requires RE_PROB=0; later matched datasets may explicitly pass
their preregistered value with ``--expected-re-prob``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from model.modules.task_adaptive_pose_field import TaskAdaptivePoseField
from solver import make_optimizer

N0_PERMUTATION = tuple(range(1, 17)) + (0,)


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


def finite(name, value):
    number = float(value.detach().float().item()) \
        if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError('%s is non-finite: %s' % (name, number))
    return number


def grad_norm(parameters):
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('non-finite TAPF gradient')
        total += float(parameter.grad.detach().float().square().sum().item())
        found = True
    return math.sqrt(total) if found else 0.0


def parameter_delta(before, parameters):
    delta = 0.0
    for old, parameter in zip(before, parameters):
        delta += float((parameter.detach().float() - old).square().sum().item())
    return math.sqrt(delta)


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


def _small_relaxation_optimizer(module, bias_weight_decay=1e-4):
    """SGD groups exercising production per-parameter LR/WD semantics."""
    groups = []
    for name, parameter in module.named_parameters():
        lr_factor = 2.0 if 'bias' in name else 1.0
        weight_decay = (float(bias_weight_decay)
                        if 'bias' in name else 1e-4)
        groups.append({
            'params': [parameter],
            'lr': 8e-4 * lr_factor,
            'weight_decay': weight_decay,
            '_tapf_test_lr_factor': lr_factor,
        })
    return torch.optim.SGD(groups, lr=8e-4, momentum=0.9)


def _small_relaxation_module(device, transition='sgd_relax'):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    return TaskAdaptivePoseField(
        in_channels=32, hidden_dim=16, output_size=(24, 8),
        mode='p0', boot_epochs=10, handoff_start_epoch=6,
        anchor_transition=transition,
    ).to(device).train()


def _bootstrap_small_relaxation(module, optimizer, scaler=None):
    device = next(module.parameters()).device
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    feature = torch.randn(2, 32, 6, 2, device=device)
    teacher = torch.rand(2, 17, 24, 8, device=device)
    scores = torch.rand(2, 17, device=device)
    module.set_epoch(10)
    _, data = module(feature, teacher, scores)
    loss = data['tapf_pose_loss']
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    anchor_parameters = list(module.anchor.parameters())
    if not all('momentum_buffer' in optimizer.state[parameter]
               for parameter in anchor_parameters):
        raise RuntimeError(
            'bootstrap did not create every anchor momentum buffer')


def validate_runtime_legacy_parity(device, bias_weight_decay=1e-4):
    """Prove exact legacy/explicit SGD parity in the frozen runtime."""
    explicit = _small_relaxation_module(device, 'sgd_relax')
    explicit_optimizer = _small_relaxation_optimizer(
        explicit, bias_weight_decay=bias_weight_decay)
    _bootstrap_small_relaxation(explicit, explicit_optimizer)

    legacy = _small_relaxation_module(device, 'hard')
    legacy.load_state_dict(explicit.state_dict(), strict=True)
    legacy_optimizer = _small_relaxation_optimizer(
        legacy, bias_weight_decay=bias_weight_decay)
    legacy_optimizer.load_state_dict(
        copy.deepcopy(explicit_optimizer.state_dict()))
    for legacy_parameter, explicit_parameter in zip(
            legacy.anchor.parameters(), explicit.anchor.parameters()):
        legacy_momentum = legacy_optimizer.state[legacy_parameter][
            'momentum_buffer']
        explicit_momentum = explicit_optimizer.state[explicit_parameter][
            'momentum_buffer']
        if not torch.equal(legacy_momentum, explicit_momentum):
            raise RuntimeError('cloned optimizer momentum values differ')
        if legacy_momentum.data_ptr() == explicit_momentum.data_ptr():
            raise RuntimeError('cloned optimizer momentum storage is aliased')
    explicit.set_epoch(11)

    learning_rates = (7e-4, 5e-4, 3e-4, 1e-4, 2e-5)
    for step_index, learning_rate in enumerate(learning_rates, 1):
        for group in legacy_optimizer.param_groups:
            group['lr'] = (learning_rate
                           * group['_tapf_test_lr_factor'])
        for group in explicit_optimizer.param_groups:
            group['lr'] = (learning_rate
                           * group['_tapf_test_lr_factor'])
        for parameter in legacy.anchor.parameters():
            parameter.grad = torch.zeros_like(parameter)

        stats = explicit.prepare_optimizer_step(
            explicit_optimizer, record_stats=True)
        legacy_optimizer.step()
        explicit_optimizer.step()
        explicit.finish_optimizer_step()
        if stats.get('anchor_relax_momentum_norm', 0.0) <= 0:
            raise RuntimeError('explicit relaxation momentum norm is zero')

        for legacy_parameter, explicit_parameter in zip(
                legacy.anchor.parameters(), explicit.anchor.parameters()):
            if not torch.equal(explicit_parameter, legacy_parameter):
                raise RuntimeError(
                    'legacy/explicit anchor parity failed at step %d'
                    % step_index)
            legacy_momentum = legacy_optimizer.state[legacy_parameter][
                'momentum_buffer']
            explicit_momentum = explicit_optimizer.state[explicit_parameter][
                'momentum_buffer']
            if not torch.equal(explicit_momentum, legacy_momentum):
                raise RuntimeError(
                    'legacy/explicit momentum parity failed at step %d'
                    % step_index)
    return len(learning_rates)


def validate_real_grad_scaler_overflow_skip(device):
    """Inject a real inf and prove GradScaler skips the whole SGD step."""
    module = _small_relaxation_module(device, 'sgd_relax')
    optimizer = _small_relaxation_optimizer(module)
    scaler = amp.GradScaler(
        enabled=True, init_scale=128.0, growth_interval=100000)
    _bootstrap_small_relaxation(module, optimizer, scaler=scaler)
    module.set_epoch(11)

    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    feature = torch.randn(2, 32, 6, 2, device=device)
    optimizer.zero_grad(set_to_none=True)
    field, data = module(feature, None)
    if data['tapf_pose_loss'] is not None:
        raise RuntimeError('overflow audit unexpectedly read pose objective')
    x_weight = torch.linspace(
        0, 1, field.shape[-1], device=device)[None, None, None]
    loss = (field.float() * x_weight).sum()
    scaler.scale(loss).backward()

    injected = False
    for parameter in module.geometry_adapter.parameters():
        if parameter.grad is not None:
            parameter.grad.view(-1)[0] = float('inf')
            injected = True
            break
    if not injected:
        raise RuntimeError('overflow audit could not inject an adapter inf')

    parameters = list(module.parameters())
    parameter_before = [parameter.detach().clone()
                        for parameter in parameters]
    anchor_parameters = list(module.anchor.parameters())
    momentum_before = [
        optimizer.state[parameter]['momentum_buffer'].detach().clone()
        for parameter in anchor_parameters
    ]
    scale_before = float(scaler.get_scale())
    module.prepare_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        module.finish_optimizer_step()
    scaler.update()
    scale_after = float(scaler.get_scale())

    if not scale_after < scale_before:
        raise RuntimeError(
            'real GradScaler overflow did not reduce scale: %g -> %g'
            % (scale_before, scale_after))
    for expected, observed in zip(parameter_before, parameters):
        if not torch.equal(observed, expected):
            raise RuntimeError(
                'GradScaler overflow changed a model parameter')
    for parameter, expected in zip(anchor_parameters, momentum_before):
        observed = optimizer.state[parameter]['momentum_buffer']
        if not torch.equal(observed, expected):
            raise RuntimeError(
                'GradScaler overflow changed anchor momentum')
        if parameter.grad is not None:
            raise RuntimeError(
                'overflow cleanup retained synthetic anchor gradient')
    return scale_before, scale_after


class _ExplodingPoseDict(dict):
    def __getitem__(self, key):
        raise RuntimeError('P0/F0 post-bootstrap touched external pose: ' + key)

    def get(self, key, default=None):
        raise RuntimeError('P0/F0 post-bootstrap touched external pose: ' + key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--expected-re-prob', default=0.0, type=float)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not cfg.MODEL.POSE_TAPF:
        raise RuntimeError('TAPF preflight requires POSE_TAPF=True')
    if int(cfg.SOLVER.IMS_PER_BATCH) != 64:
        raise RuntimeError('batch size must stay 64')
    if float(cfg.INPUT.RE_PROB) != float(args.expected_re_prob):
        raise RuntimeError(
            'unexpected RE_PROB: observed=%g expected=%g'
            % (float(cfg.INPUT.RE_PROB), float(args.expected_re_prob)))
    bootstrap_permutation = tuple(getattr(
        cfg.MODEL, 'POSE_TAPF_BOOTSTRAP_JOINT_PERMUTATION', []))
    if bootstrap_permutation:
        if bootstrap_permutation != N0_PERMUTATION:
            raise RuntimeError('N0 permutation differs from preregistered 17-cycle')
        if str(cfg.MODEL.POSE_TAPF_MODE).lower() != 'f0':
            raise RuntimeError('N0 permutation control must use residual-OFF F0')
        if str(getattr(
                cfg.MODEL, 'POSE_TAPF_ANCHOR_TRANSITION', 'hard')).lower() \
                != 'hard':
            raise RuntimeError('N0 permutation control must use hard transition')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device('cuda')
    # First reproduce the exact production config (both WD=1e-4), then
    # stress distinct per-group weight decay to prove the implementation does
    # not accidentally collapse parameter-group semantics.
    parity_steps = validate_runtime_legacy_parity(
        device, bias_weight_decay=1e-4)
    parity_steps += validate_runtime_legacy_parity(
        device, bias_weight_decay=2e-4)
    overflow_scale = validate_real_grad_scaler_overflow_skip(device)
    set_seed(int(cfg.SOLVER.SEED))
    loaders = make_dataloader(cfg)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    if model.tapf.parameter_count >= 500000:
        raise RuntimeError('TAPF exceeds the 0.5M parameter gate')
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    set_seed(int(cfg.SOLVER.SEED))
    batch = next(iter(train_loader))
    img, target, target_cam, target_view, pose_dict = batch
    if not isinstance(img, torch.Tensor) or img.shape[0] != 64:
        raise RuntimeError('production preflight did not receive one batch64 tensor')
    img = img.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    target_cam = target_cam.to(device, non_blocking=True)
    target_view = target_view.to(device, non_blocking=True)
    pose_dict = to_device(pose_dict, device)
    target_scores = (pose_dict['scores'][:, 0].float()
                     * pose_dict['person_mask'][:, 0:1].float())

    model.set_tapf_epoch(args.epoch)
    mode = model.tapf.mode
    bootstrap = args.epoch <= model.tapf.boot_epochs
    expect_pose = bootstrap or mode in ('d0', 'j0')
    expect_reid_adapter = (not bootstrap and mode in ('p0', 'j0'))
    expect_relaxation = (
        not bootstrap and model.tapf.anchor_transition == 'sgd_relax')

    anchor_parameters = list(model.tapf.anchor.parameters())
    adapter_parameters = list(model.tapf.geometry_adapter.parameters())
    adapter_output_parameters = list(
        model.tapf.geometry_adapter[-1].parameters())
    adapter_upstream_parameters = [
        parameter
        for module in list(model.tapf.geometry_adapter.children())[:-1]
        for parameter in module.parameters()
    ]
    backbone_parameters = [model.base.patch_embed.projection.weight]
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(cfg.SOLVER.AMP_INIT_SCALE))

    def optimizer_step_at(epoch):
        """One production step used only for minimal connectivity priming.

        Post-bootstrap training never begins from a zero PSG: epochs 1--10
        have already updated its zero-initialized final projection.  The first
        post-bootstrap step then opens the zero-initialized adapter output; the
        following step can exercise the complete adapter.  Reproduce only
        those zero-init dependencies before checking an epoch-11 arm from a
        fresh process; this does not approximate ten epochs of learned state.
        """
        model.train()
        model.set_tapf_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        psg_final_parameters = list(
            model.psg_modules_dict['s3_b0'].encoder[-1].parameters())
        psg_before = [parameter.detach().float().clone()
                      for parameter in psg_final_parameters]
        adapter_output_before = [parameter.detach().float().clone()
                                 for parameter in adapter_output_parameters]
        adapter_weight_before = [
            model.tapf.geometry_adapter[-1].weight.detach().float().clone()]
        adapter_bias_before = [
            model.tapf.geometry_adapter[-1].bias.detach().float().clone()]
        adapter_upstream_before = [parameter.detach().float().clone()
                                   for parameter in adapter_upstream_parameters]
        with amp.autocast(enabled=True):
            warm_output = model(
                img, label=target, cam_label=target_cam,
                view_label=target_view, pose_dict=pose_dict)
            warm_score, warm_feature, _, _, warm_data = warm_output
            warm_loss = loss_func(
                warm_score, warm_feature, target, target_cam)
            warm_pose = warm_data.get('tapf_pose_loss')
            if warm_pose is not None:
                warm_loss = warm_loss + float(
                    cfg.MODEL.POSE_TAPF_LOSS_WEIGHT) * warm_pose
        scaler.scale(warm_loss).backward()
        relaxation_stats = model.prepare_tapf_optimizer_step(optimizer)
        try:
            scaler.step(optimizer)
        finally:
            model.finish_tapf_optimizer_step()
        scaler.update()
        return {
            'psg_final': parameter_delta(
                psg_before, psg_final_parameters),
            'adapter_output': parameter_delta(
                adapter_output_before, adapter_output_parameters),
            'adapter_weight': parameter_delta(
                adapter_weight_before,
                [model.tapf.geometry_adapter[-1].weight]),
            'adapter_bias': parameter_delta(
                adapter_bias_before,
                [model.tapf.geometry_adapter[-1].bias]),
            'adapter_upstream': parameter_delta(
                adapter_upstream_before, adapter_upstream_parameters),
            'relaxation_stats': relaxation_stats,
        }

    if not bootstrap:
        bootstrap_delta = optimizer_step_at(model.tapf.boot_epochs)
        if bootstrap_delta['psg_final'] <= 0:
            raise RuntimeError(
                'bootstrap priming did not update zero-init PSG final layer')
        if expect_reid_adapter:
            reid_prime_delta = optimizer_step_at(args.epoch)
            if reid_prime_delta['adapter_weight'] <= 0:
                raise RuntimeError(
                    'first ReID-only step did not open adapter output weight: '
                    'output=%g weight=%g bias=%g' % (
                        reid_prime_delta['adapter_output'],
                        reid_prime_delta['adapter_weight'],
                        reid_prime_delta['adapter_bias']))
        model.set_tapf_epoch(args.epoch)

    # P0/F0 epoch 11+ must not even index an external pose dictionary.
    if not expect_pose and not bootstrap:
        model.train()
        with torch.no_grad(), amp.autocast(enabled=True):
            model(
                img[:2], label=target[:2], cam_label=target_cam[:2],
                view_label=target_view[:2], pose_dict=_ExplodingPoseDict())

    # Every TAPF eval is predicted-only and exactly independent of pose input.
    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_correct = model(
            img[:2], cam_label=target_cam[:2], view_label=target_view[:2],
            pose_dict={key: value[:2] if isinstance(value, torch.Tensor)
                       else value for key, value in pose_dict.items()})[0]
        eval_none = model(
            img[:2], cam_label=target_cam[:2], view_label=target_view[:2],
            pose_dict=None)[0]
        eval_exploding = model(
            img[:2], cam_label=target_cam[:2], view_label=target_view[:2],
            pose_dict=_ExplodingPoseDict())[0]
    if not torch.equal(eval_correct, eval_none):
        raise RuntimeError('eval descriptor depends on external pose')
    if not torch.equal(eval_correct, eval_exploding):
        raise RuntimeError('eval descriptor indexed external pose')

    captured_teacher_scores = []
    captured_psg_raw = []
    captured_psg_inputs = []
    tapf_hook = model.tapf.register_forward_pre_hook(
        lambda _module, inputs: captured_teacher_scores.append(
            None if len(inputs) < 3 or inputs[2] is None
            else inputs[2].detach().float()))
    first_psg = model.psg_modules_dict['s3_b0']
    raw_hook = first_psg.register_forward_pre_hook(
        lambda _module, inputs: captured_psg_raw.append(
            inputs[2].detach()))
    encoded_hook = first_psg.encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured_psg_inputs.append(
            inputs[0].detach().float()))
    model.train()
    model.set_tapf_epoch(args.epoch)
    optimizer.zero_grad(set_to_none=True)
    optimizer_center.zero_grad(set_to_none=True)
    anchor_before = [p.detach().float().clone() for p in anchor_parameters]
    adapter_before = [p.detach().float().clone() for p in adapter_parameters]
    with amp.autocast(enabled=True):
        output = model(
            img, label=target, cam_label=target_cam,
            view_label=target_view, pose_dict=pose_dict)
        if len(output) != 5:
            raise RuntimeError('TAPF production forward must return five items')
        score, feature, _, recon_loss, kp_data = output
        if recon_loss is not None:
            raise RuntimeError('TAPF unexpectedly used recon_loss return')
        pose_loss = kp_data.get('tapf_pose_loss')
        if (pose_loss is not None) != expect_pose:
            raise RuntimeError(
                'pose-loss schedule mismatch: mode=%s epoch=%d' %
                (mode, args.epoch))
        identity_loss = loss_func(score, feature, target, target_cam)
        total_loss = identity_loss
        if pose_loss is not None:
            total_loss = total_loss + float(
                cfg.MODEL.POSE_TAPF_LOSS_WEIGHT) * pose_loss

    tapf_hook.remove()
    raw_hook.remove()
    encoded_hook.remove()
    stats = kp_data['tapf_stats']
    permutation_active = finite(
        'teacher_permutation_active',
        stats['teacher_permutation_active'])
    permutation_fixed_points = finite(
        'teacher_permutation_fixed_points',
        stats['teacher_permutation_fixed_points'])
    if permutation_active != float(bool(bootstrap_permutation)):
        raise RuntimeError('bootstrap permutation active flag mismatch')
    expected_fixed_points = 0.0 if bootstrap_permutation else 17.0
    if permutation_fixed_points != expected_fixed_points:
        raise RuntimeError('bootstrap permutation fixed-point count mismatch')
    if expect_pose:
        if len(captured_teacher_scores) != 1 \
                or captured_teacher_scores[0] is None:
            raise RuntimeError('failed to capture TAPF teacher scores')
        score_tensor_error = float((
            captured_teacher_scores[0] - target_scores).abs().max().item())
        if score_tensor_error != 0:
            raise RuntimeError(
                'TAPF input is not exact person-0 scores: error=%g'
                % score_tensor_error)
        bounded_target_scores = target_scores.clamp(0.0, 1.0)
        score_error = abs(
            finite('reported_teacher_confidence', stats['teacher_confidence'])
            - float(bounded_target_scores.mean().item()))
        if score_error > 1e-6:
            raise RuntimeError(
                'confidence target mean mismatch: error=%g reported=%r/%s '
                'expected=%r/%s' % (
                    score_error, stats['teacher_confidence'].item(),
                    stats['teacher_confidence'].dtype,
                    bounded_target_scores.mean().item(),
                    bounded_target_scores.dtype))
    if len(captured_psg_raw) != 1 or len(captured_psg_inputs) != 1:
        raise RuntimeError('failed to capture exactly one first-block PSG path')
    psg_raw = captured_psg_raw[0]
    psg_input = captured_psg_inputs[0]
    if bootstrap and args.epoch == 1:
        target_heatmaps = (pose_dict['heatmaps'][:, 0].float()
                           * pose_dict['person_mask'][
                               :, 0, None, None, None].float())
        if bootstrap_permutation:
            index = torch.tensor(
                bootstrap_permutation, dtype=torch.long, device=device)
            target_heatmaps = target_heatmaps.index_select(1, index)
        expected_epoch1_field = target_heatmaps.to(psg_raw.dtype)
        if not torch.equal(psg_raw, expected_epoch1_field):
            error = float((psg_raw - expected_epoch1_field).abs().max().item())
            raise RuntimeError(
                'epoch-1 PSG field is not the exact once-permuted teacher: %g'
                % error)
    resized_raw = (F.interpolate(
        psg_raw, size=psg_input.shape[-2:], mode='bilinear',
        align_corners=False) if psg_raw.shape[-2:] != psg_input.shape[-2:]
        else psg_raw)
    expected_psg_input = torch.sigmoid(resized_raw).float()
    if not torch.equal(psg_input, expected_psg_input):
        error = float((psg_input - expected_psg_input).abs().max().item())
        raise RuntimeError(
            'raw heatmap was not resized then sigmoid exactly once: %g' % error)

    pose_to_anchor = objective_grad_norm(pose_loss, anchor_parameters)
    pose_to_adapter = objective_grad_norm(pose_loss, adapter_parameters)
    pose_to_backbone = objective_grad_norm(pose_loss, backbone_parameters)
    reid_to_anchor = objective_grad_norm(identity_loss, anchor_parameters)
    reid_to_adapter_output = objective_grad_norm(
        identity_loss, adapter_output_parameters)
    reid_to_adapter_upstream = objective_grad_norm(
        identity_loss, adapter_upstream_parameters)
    reid_to_adapter = math.sqrt(
        reid_to_adapter_output ** 2 + reid_to_adapter_upstream ** 2)
    reid_to_backbone = objective_grad_norm(identity_loss, backbone_parameters)
    if (pose_to_anchor > 0) != expect_pose:
        raise RuntimeError('independent pose->anchor mismatch: %g'
                           % pose_to_anchor)
    if pose_to_adapter > 0 or pose_to_backbone > 0:
        raise RuntimeError(
            'pose objective escaped isolation: adapter=%g backbone=%g'
            % (pose_to_adapter, pose_to_backbone))
    if reid_to_anchor > 0:
        raise RuntimeError('ReID objective rewrote anchor: %g' % reid_to_anchor)
    if ((reid_to_adapter_output > 0) != expect_reid_adapter
            or (reid_to_adapter_upstream > 0) != expect_reid_adapter):
        raise RuntimeError(
            'independent ReID->adapter mismatch: output=%g upstream=%g '
            'prime_output=%s prime_weight=%s prime_bias=%s'
            % (reid_to_adapter_output, reid_to_adapter_upstream,
               None if bootstrap or not expect_reid_adapter
               else reid_prime_delta['adapter_output'],
               None if bootstrap or not expect_reid_adapter
               else reid_prime_delta['adapter_weight'],
               None if bootstrap or not expect_reid_adapter
               else reid_prime_delta['adapter_bias']))
    if reid_to_backbone <= 0:
        raise RuntimeError('independent ReID gradient missed backbone')

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    backbone_norm = grad_norm(
        [model.base.patch_embed.projection.weight])
    anchor_norm = grad_norm(anchor_parameters)
    adapter_norm = grad_norm(adapter_parameters)
    if backbone_norm <= 0:
        raise RuntimeError('standard ReID gradient did not reach backbone')
    if (anchor_norm > 0) != expect_pose:
        raise RuntimeError('anchor gradient schedule mismatch: %g' % anchor_norm)
    if (adapter_norm > 0) != expect_reid_adapter:
        raise RuntimeError(
            'geometry adapter gradient schedule mismatch: %g' % adapter_norm)

    relaxation_stats = model.prepare_tapf_optimizer_step(optimizer)
    try:
        scaler.step(optimizer)
    finally:
        model.finish_tapf_optimizer_step()
    scaler.update()
    anchor_delta = parameter_delta(anchor_before, anchor_parameters)
    adapter_delta = parameter_delta(adapter_before, adapter_parameters)
    if (anchor_delta > 0) != (expect_pose or expect_relaxation):
        raise RuntimeError('anchor optimizer delta mismatch: %g' % anchor_delta)
    if (adapter_delta > 0) != expect_reid_adapter:
        raise RuntimeError('adapter optimizer delta mismatch: %g' % adapter_delta)

    print('TAPF_CUDA_PREFLIGHT_PASS')
    print('runtime_parity_steps=%d overflow_scale=%.1f->%.1f'
          % (parity_steps, overflow_scale[0], overflow_scale[1]))
    print('mode=%s epoch=%d params=%d batch=%d image=%s'
          % (mode, args.epoch, model.tapf.parameter_count, img.shape[0],
             tuple(img.shape)))
    print('loss identity=%.8f pose=%s total=%.8f scale=%.1f'
          % (finite('identity_loss', identity_loss),
             'None' if pose_loss is None else '%.8f' % finite(
                 'pose_loss', pose_loss),
             finite('total_loss', total_loss), scaler.get_scale()))
    print('grad backbone=%.8e anchor=%.8e adapter=%.8e'
          % (backbone_norm, anchor_norm, adapter_norm))
    print('cross pose=[anchor %.8e adapter %.8e backbone %.8e] '
          'reid=[anchor %.8e adapter_out %.8e adapter_up %.8e backbone %.8e]'
          % (pose_to_anchor, pose_to_adapter, pose_to_backbone,
             reid_to_anchor, reid_to_adapter_output,
             reid_to_adapter_upstream, reid_to_backbone))
    print('delta anchor=%.8e adapter=%.8e'
          % (anchor_delta, adapter_delta))
    print('transition=%s relaxation=%s'
          % (model.tapf.anchor_transition,
             relaxation_stats if relaxation_stats else 'inactive'))
    print('bootstrap_permutation=%s fixed_points=%d'
          % (bootstrap_permutation if bootstrap_permutation else 'off',
             int(permutation_fixed_points)))
    print('raw teacher=[%s,%s] field=[%.6f,%.6f] psg=[%.6f,%.6f]'
          % (stats.get('teacher_raw_min'), stats.get('teacher_raw_max'),
             finite('field_raw_min', stats['field_raw_min']),
             finite('field_raw_max', stats['field_raw_max']),
             finite('field_sigmoid_min', stats['field_sigmoid_min']),
             finite('field_sigmoid_max', stats['field_sigmoid_max'])))


if __name__ == '__main__':
    main()
