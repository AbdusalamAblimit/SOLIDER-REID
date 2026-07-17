import importlib.util
import io
import pickle
from pathlib import Path

import torch


_MODULE_PATH = (Path(__file__).resolve().parents[2] / 'model' / 'modules'
                / 'task_adaptive_pose_field.py')
_SPEC = importlib.util.spec_from_file_location('task_adaptive_pose_field',
                                               _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TaskAdaptivePoseField = _MODULE.TaskAdaptivePoseField
N0_PERMUTATION = tuple(range(1, 17)) + (0,)


def _inputs(batch=2, channels=32):
    torch.manual_seed(7)
    feature = torch.randn(batch, channels, 24, 8, requires_grad=True)
    teacher = torch.zeros(batch, 17, 96, 32)
    for joint in range(17):
        y = 4 + (joint * 5) % 88
        x = 2 + (joint * 3) % 28
        teacher[:, joint, y, x] = 0.4 + 0.03 * (joint % 10)
    scores = teacher.flatten(2).amax(dim=-1)
    return feature, teacher, scores


def _module(mode='p0', anchor_transition='hard',
            bootstrap_joint_permutation=None):
    return TaskAdaptivePoseField(
        in_channels=32,
        hidden_dim=16,
        output_size=(96, 32),
        mode=mode,
        anchor_transition=anchor_transition,
        bootstrap_joint_permutation=bootstrap_joint_permutation,
        boot_epochs=10,
        handoff_start_epoch=6,
    )


def _has_nonzero_grad(parameters):
    return any(parameter.grad is not None
               and bool(torch.isfinite(parameter.grad).all())
               and float(parameter.grad.abs().sum()) > 0
               for parameter in parameters)


def _has_nonzero_autograd(loss, parameters):
    parameters = [parameter for parameter in parameters
                  if parameter.requires_grad]
    if loss is None or not loss.requires_grad or not parameters:
        return False
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True)
    return any(gradient is not None
               and bool(torch.isfinite(gradient).all())
               and float(gradient.abs().sum()) > 0
               for gradient in gradients)


def test_bootstrap_schedule_is_fixed_and_reaches_student_at_epoch_10():
    module = _module()
    expected = {1: 0.0, 5: 0.0, 6: 0.2, 8: 0.6, 10: 1.0}
    for epoch, fraction in expected.items():
        module.set_epoch(epoch)
        assert abs(module._bootstrap_student_fraction() - fraction) < 1e-8


def test_bootstrap_pose_loss_updates_anchor_but_not_feature_or_adapter():
    feature, teacher, scores = _inputs()
    module = _module('p0').train()
    module.set_epoch(3)
    field, data = module(feature, teacher, scores)
    assert field.shape == teacher.shape
    assert data['tapf_pose_loss'] is not None
    assert not field.requires_grad
    data['tapf_pose_loss'].backward()
    assert _has_nonzero_grad(module.anchor.parameters())
    assert not _has_nonzero_grad(module.geometry_adapter.parameters())
    assert feature.grad is None


def test_n0_permutation_is_validated_as_a_full_derangement():
    invalid = (
        tuple(range(16)),
        tuple(range(16)) + (15,),
        (0,) + tuple(range(2, 17)) + (1,),
    )
    for permutation in invalid:
        try:
            _module('f0', bootstrap_joint_permutation=permutation)
        except ValueError:
            pass
        else:
            raise AssertionError(
                'invalid N0 bootstrap permutation was accepted: %s'
                % (permutation,))


def test_n0_internal_permutation_matches_exact_external_relabeling_once():
    feature, teacher, _ = _inputs()
    # Unique channel tags make a double permutation or an unpermuted score
    # impossible to hide behind equal confidence values.
    scores = torch.linspace(0.11, 0.91, 17).repeat(feature.shape[0], 1)
    index = torch.tensor(N0_PERMUTATION)
    permuted_teacher = teacher.index_select(1, index)
    permuted_scores = scores.index_select(1, index)

    torch.manual_seed(1234)
    internal = _module(
        'f0', bootstrap_joint_permutation=N0_PERMUTATION).train()
    torch.manual_seed(1234)
    external = _module('f0').train()
    teacher_before = teacher.clone()
    observed_teacher, observed_scores = internal._permute_bootstrap_teacher(
        teacher, scores)
    torch.testing.assert_close(
        observed_teacher, permuted_teacher, rtol=0, atol=0)
    torch.testing.assert_close(
        observed_scores, permuted_scores, rtol=0, atol=0)
    torch.testing.assert_close(teacher, teacher_before, rtol=0, atol=0)
    assert internal.state_dict().keys() == external.state_dict().keys()
    for key, value in internal.state_dict().items():
        torch.testing.assert_close(
            value, external.state_dict()[key], rtol=0, atol=0)

    for epoch in (1, 10):
        internal.set_epoch(epoch)
        external.set_epoch(epoch)
        internal_field, internal_data = internal(feature, teacher, scores)
        external_field, external_data = external(
            feature, permuted_teacher, permuted_scores)
        torch.testing.assert_close(
            internal_field, external_field, rtol=0, atol=0)
        torch.testing.assert_close(
            internal_data['tapf_pose_loss'],
            external_data['tapf_pose_loss'], rtol=0, atol=0)
        assert float(internal_data['tapf_stats'][
            'teacher_permutation_active']) == 1.0
        assert float(internal_data['tapf_stats'][
            'teacher_permutation_fixed_points']) == 0.0

    # At epoch 1 handoff is exactly the once-permuted teacher field.
    internal.set_epoch(1)
    field, data = internal(feature, teacher, scores)
    torch.testing.assert_close(field, permuted_teacher, rtol=0, atol=0)
    assert not torch.equal(field, teacher)
    data['tapf_pose_loss'].backward()
    assert _has_nonzero_grad(internal.anchor.parameters())
    assert not _has_nonzero_grad(internal.geometry_adapter.parameters())
    assert feature.grad is None


def test_n0_metadata_does_not_change_parameters_or_optimizer_groups():
    torch.manual_seed(1234)
    f0 = _module('f0')
    torch.manual_seed(1234)
    n0 = _module('f0', bootstrap_joint_permutation=N0_PERMUTATION)
    assert tuple(f0.named_parameters()) and tuple(n0.named_parameters())
    f0_named = dict(f0.named_parameters())
    n0_named = dict(n0.named_parameters())
    assert f0_named.keys() == n0_named.keys()
    for name, parameter in f0_named.items():
        torch.testing.assert_close(
            parameter, n0_named[name], rtol=0, atol=0)
    f0_optimizer = torch.optim.SGD(f0.parameters(), lr=8e-4, momentum=0.9)
    n0_optimizer = torch.optim.SGD(n0.parameters(), lr=8e-4, momentum=0.9)
    assert f0_optimizer.state_dict()['param_groups'] \
        == n0_optimizer.state_dict()['param_groups']


def test_p0_post_boot_is_teacher_independent_and_updates_only_geometry():
    feature, teacher, scores = _inputs()
    module = _module('p0').train()
    module.set_epoch(11)

    field_a, data_a = module(feature, teacher, scores)
    field_b, data_b = module(
        feature, torch.rand_like(teacher), torch.rand_like(scores))
    field_c, data_c = module(feature, None)
    assert data_a['tapf_pose_loss'] is None
    assert data_b['tapf_pose_loss'] is None
    assert data_c['tapf_pose_loss'] is None
    torch.testing.assert_close(field_a, field_b, rtol=0, atol=0)
    torch.testing.assert_close(field_a, field_c, rtol=0, atol=0)

    # A coordinate-weighted objective guarantees a non-symmetric shift signal.
    x_weight = torch.linspace(0, 1, field_a.shape[-1])[None, None, None]
    (field_a.float() * x_weight).sum().backward()
    assert _has_nonzero_grad(module.geometry_adapter.parameters())
    assert not _has_nonzero_grad(module.anchor.parameters())
    assert feature.grad is None


def test_post_boot_2x2_gradient_semantics():
    for mode, expect_pose, expect_reid in (
            ('f0', False, False), ('d0', True, False),
            ('p0', False, True), ('j0', True, True)):
        feature, teacher, scores = _inputs()
        module = _module(mode).train()
        module.set_epoch(11)
        field, data = module(feature, teacher, scores)
        assert (data['tapf_pose_loss'] is not None) is expect_pose
        pose_loss = data['tapf_pose_loss']
        assert (_has_nonzero_autograd(
            pose_loss, module.anchor.parameters()) is expect_pose)
        assert not _has_nonzero_autograd(
            pose_loss, module.geometry_adapter.parameters())
        assert not _has_nonzero_autograd(pose_loss, (feature,))

        x_weight = torch.linspace(0, 1, field.shape[-1])[None, None, None]
        reid_proxy = ((field.float() * x_weight).sum()
                      if field.requires_grad else None)
        assert not _has_nonzero_autograd(
            reid_proxy, module.anchor.parameters())
        assert (_has_nonzero_autograd(
            reid_proxy, module.geometry_adapter.parameters()) is expect_reid)
        assert not _has_nonzero_autograd(reid_proxy, (feature,))

        # autograd.grad must not leave accumulated .grad values that could
        # hide pose→adapter or ReID→anchor cross-contamination.
        assert not _has_nonzero_grad(module.anchor.parameters())
        assert not _has_nonzero_grad(module.geometry_adapter.parameters())
        assert feature.grad is None


def test_p0_f0_hard_freeze_survives_sgd_momentum_and_default_zero_grad():
    for mode in ('f0', 'p0'):
        feature, teacher, scores = _inputs()
        module = _module(mode).train()
        optimizer = torch.optim.SGD(
            module.parameters(), lr=1e-3, momentum=0.9,
            weight_decay=1e-4)

        # Build the exact stale-gradient/momentum state that exists at the end
        # of bootstrap under the production PyTorch 1.13 optimizer semantics.
        module.set_epoch(10)
        _, bootstrap_data = module(feature, teacher, scores)
        bootstrap_data['tapf_pose_loss'].backward()
        optimizer.step()
        anchor_parameters = list(module.anchor.parameters())
        assert any(parameter in optimizer.state for parameter in anchor_parameters)

        module.set_epoch(11)
        assert all(not parameter.requires_grad
                   for parameter in anchor_parameters)
        assert all(parameter.grad is None for parameter in anchor_parameters)
        frozen = [parameter.detach().clone()
                  for parameter in anchor_parameters]

        # Deliberately use the production default rather than
        # set_to_none=True; this is the behavior that exposed the regression.
        optimizer.zero_grad()
        post_feature, _, _ = _inputs()
        field, post_data = module(post_feature, None)
        assert post_data['tapf_pose_loss'] is None
        if mode == 'p0':
            x_weight = torch.linspace(
                0, 1, field.shape[-1])[None, None, None]
            (field.float() * x_weight).sum().backward()
        optimizer.step()

        for expected, observed in zip(frozen, anchor_parameters):
            torch.testing.assert_close(
                observed, expected, rtol=0, atol=0)


def _build_bootstrap_optimizer_state(module):
    feature, teacher, scores = _inputs()
    optimizer = torch.optim.SGD(
        module.parameters(), lr=8e-4, momentum=0.9,
        weight_decay=1e-4)
    module.train()
    module.set_epoch(10)
    _, data = module(feature, teacher, scores)
    data['tapf_pose_loss'].backward()
    optimizer.step()
    anchor_parameters = list(module.anchor.parameters())
    assert all('momentum_buffer' in optimizer.state[parameter]
               for parameter in anchor_parameters)
    return optimizer


def test_explicit_sgd_relaxation_matches_legacy_zero_gradient_steps():
    torch.manual_seed(1234)
    legacy = _module('f0', 'hard')
    torch.manual_seed(1234)
    explicit = _module('f0', 'sgd_relax')
    legacy_optimizer = _build_bootstrap_optimizer_state(legacy)
    explicit_optimizer = _build_bootstrap_optimizer_state(explicit)

    explicit.set_epoch(11)
    assert all(not parameter.requires_grad
               for parameter in explicit.anchor.parameters())
    assert all(parameter.grad is None
               for parameter in explicit.anchor.parameters())

    learning_rates = (7e-4, 5e-4, 3e-4, 1e-4, 2e-5)
    for learning_rate in learning_rates:
        for group in legacy_optimizer.param_groups:
            group['lr'] = learning_rate
        for group in explicit_optimizer.param_groups:
            group['lr'] = learning_rate

        # This is the old PyTorch 1.13 zero-objective-gradient behavior made
        # explicit only in the test oracle.
        for parameter in legacy.anchor.parameters():
            parameter.grad = torch.zeros_like(parameter)

        stats = explicit.prepare_optimizer_step(
            explicit_optimizer, record_stats=True)
        assert stats['anchor_relax_active'] == 1.0
        assert stats['anchor_relax_momentum_norm'] > 0
        legacy_optimizer.step()
        explicit_optimizer.step()
        explicit.finish_optimizer_step()

        for legacy_parameter, explicit_parameter in zip(
                legacy.anchor.parameters(), explicit.anchor.parameters()):
            torch.testing.assert_close(
                explicit_parameter, legacy_parameter, rtol=0, atol=0)
            torch.testing.assert_close(
                explicit_optimizer.state[explicit_parameter][
                    'momentum_buffer'],
                legacy_optimizer.state[legacy_parameter]['momentum_buffer'],
                rtol=0, atol=0)
            assert explicit_parameter.grad is None


def test_relaxed_f0_p0_share_anchor_trajectory_and_isolate_adapter():
    torch.manual_seed(1234)
    f0 = _module('f0', 'sgd_relax')
    torch.manual_seed(1234)
    p0 = _module('p0', 'sgd_relax')
    f0_optimizer = _build_bootstrap_optimizer_state(f0)
    p0_optimizer = _build_bootstrap_optimizer_state(p0)
    f0.set_epoch(11)
    p0.set_epoch(11)

    f0_adapter_before = [parameter.detach().clone()
                         for parameter in f0.geometry_adapter.parameters()]
    p0_adapter_before = [parameter.detach().clone()
                         for parameter in p0.geometry_adapter.parameters()]

    f0_optimizer.zero_grad(set_to_none=True)
    p0_optimizer.zero_grad(set_to_none=True)
    p0_feature, _, _ = _inputs()
    p0_field, p0_data = p0(p0_feature, None)
    assert p0_data['tapf_pose_loss'] is None
    x_weight = torch.linspace(
        0, 1, p0_field.shape[-1])[None, None, None]
    (p0_field.float() * x_weight).sum().backward()
    assert all(parameter.grad is None for parameter in p0.anchor.parameters())

    f0.prepare_optimizer_step(f0_optimizer)
    p0.prepare_optimizer_step(p0_optimizer)
    f0_optimizer.step()
    p0_optimizer.step()
    f0.finish_optimizer_step()
    p0.finish_optimizer_step()

    for f0_parameter, p0_parameter in zip(
            f0.anchor.parameters(), p0.anchor.parameters()):
        torch.testing.assert_close(
            p0_parameter, f0_parameter, rtol=0, atol=0)
    assert all(torch.equal(before, after)
               for before, after in zip(
                   f0_adapter_before, f0.geometry_adapter.parameters()))
    assert any(not torch.equal(before, after)
               for before, after in zip(
                   p0_adapter_before, p0.geometry_adapter.parameters()))


def test_relaxation_cleanup_preserves_anchor_when_optimizer_step_is_skipped():
    module = _module('f0', 'sgd_relax')
    optimizer = _build_bootstrap_optimizer_state(module)
    module.set_epoch(11)
    before = [parameter.detach().clone()
              for parameter in module.anchor.parameters()]
    momentum_before = [
        optimizer.state[parameter]['momentum_buffer'].detach().clone()
        for parameter in module.anchor.parameters()
    ]

    module.prepare_optimizer_step(optimizer)
    # GradScaler overflow semantics: optimizer.step() is skipped, but cleanup
    # still runs in the production finally block.
    module.finish_optimizer_step()

    for expected, observed in zip(before, module.anchor.parameters()):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert observed.grad is None
    for parameter, expected in zip(module.anchor.parameters(),
                                   momentum_before):
        torch.testing.assert_close(
            optimizer.state[parameter]['momentum_buffer'], expected,
            rtol=0, atol=0)


def test_sgd_relaxation_rejects_non_sgd_and_non_attribution_modes():
    try:
        _module('d0', 'sgd_relax')
    except ValueError as error:
        assert 'only for F0/P0' in str(error)
    else:
        raise AssertionError('D0 must reject TAPF sgd_relax')

    module = _module('p0', 'sgd_relax')
    _build_bootstrap_optimizer_state(module)
    module.set_epoch(11)
    adam = torch.optim.Adam(module.parameters(), lr=1e-3)
    try:
        module.prepare_optimizer_step(adam)
    except TypeError as error:
        assert 'requires torch.optim.SGD' in str(error)
    else:
        raise AssertionError('TAPF sgd_relax must reject Adam')


def test_eval_never_reads_external_teacher_and_uses_mode_geometry():
    feature, teacher, scores = _inputs()
    module = _module('p0').eval()
    module.set_epoch(1)  # eval must ignore the training schedule epoch.
    with torch.no_grad():
        module.geometry_adapter[-1].bias.copy_(
            torch.tensor([0.7, -0.4, 0.3, -0.2]))
        field_a, data_a = module(feature, teacher, scores)
        field_b, data_b = module(feature, None)
    assert data_a['tapf_pose_loss'] is None
    assert data_b['tapf_pose_loss'] is None
    torch.testing.assert_close(field_a, field_b, rtol=0, atol=0)
    assert bool(torch.isfinite(field_a).all())
    assert float(field_a.min()) >= 0.0
    assert float(field_a.max()) <= 1.0


def test_four_arms_have_exact_paired_initialization():
    reference = None
    for mode in ('f0', 'd0', 'p0', 'j0'):
        torch.manual_seed(1234)
        state = _module(mode).state_dict()
        if reference is None:
            reference = {key: value.clone() for key, value in state.items()}
            continue
        assert state.keys() == reference.keys()
        for key, value in state.items():
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0)


def test_strict_state_and_optimizer_roundtrip_is_exact():
    feature, teacher, scores = _inputs()
    torch.manual_seed(1234)
    module = _module('p0').train()
    module.set_epoch(11)
    optimizer = torch.optim.SGD(
        module.parameters(), lr=1e-3, momentum=0.9)
    field, _ = module(feature, teacher, scores)
    x_weight = torch.linspace(0, 1, field.shape[-1])[None, None, None]
    (field.float() * x_weight).sum().backward()
    optimizer.step()

    payload = io.BytesIO()
    torch.save({
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, payload)
    payload.seek(0)

    torch.manual_seed(9999)
    restored = _module('p0').train()
    restored_optimizer = torch.optim.SGD(
        restored.parameters(), lr=1e-3, momentum=0.9)
    try:
        checkpoint = torch.load(payload, weights_only=True)
    except (pickle.UnpicklingError, TypeError):
        # PyTorch 1.13's restricted loader cannot deserialize this test's
        # optimizer-state payload.  It was created in-memory immediately
        # above, so an explicit trusted fallback is safe and keeps the gate
        # compatible without weakening production checkpoint loading.
        payload.seek(0)
        try:
            checkpoint = torch.load(payload, weights_only=False)
        except TypeError:
            payload.seek(0)
            checkpoint = torch.load(payload)
    restored.load_state_dict(checkpoint['model'], strict=True)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    expected_optimizer = optimizer.state_dict()
    observed_optimizer = restored_optimizer.state_dict()
    assert observed_optimizer['param_groups'] == expected_optimizer['param_groups']
    assert observed_optimizer['state'].keys() == expected_optimizer['state'].keys()
    for parameter_id, expected_state in expected_optimizer['state'].items():
        observed_state = observed_optimizer['state'][parameter_id]
        assert observed_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            observed_value = observed_state[key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(
                    observed_value, expected_value, rtol=0, atol=0)
            else:
                assert observed_value == expected_value

    module.eval()
    restored.eval()
    with torch.no_grad():
        expected, _ = module(feature.detach(), None)
        observed, _ = restored(feature.detach(), None)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
