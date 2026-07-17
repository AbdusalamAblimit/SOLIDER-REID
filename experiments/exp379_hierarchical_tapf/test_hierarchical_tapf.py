import importlib.util
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_MODULES = _ROOT / 'model' / 'modules'

# Load only the two TAPF modules.  Importing ``model`` itself would eagerly
# import the full backbone stack (including optional cv2/mmcv dependencies),
# which is unrelated to this CPU unit test.
_model_package = types.ModuleType('model')
_model_package.__path__ = [str(_ROOT / 'model')]
_modules_package = types.ModuleType('model.modules')
_modules_package.__path__ = [str(_MODULES)]
sys.modules.setdefault('model', _model_package)
sys.modules.setdefault('model.modules', _modules_package)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_module(
    'model.modules.task_adaptive_pose_field',
    _MODULES / 'task_adaptive_pose_field.py')
_HIERARCHICAL = _load_module(
    'model.modules.hierarchical_task_adaptive_pose_field',
    _MODULES / 'hierarchical_task_adaptive_pose_field.py')
HierarchicalTaskAdaptivePoseField = (
    _HIERARCHICAL.HierarchicalTaskAdaptivePoseField)


def _module():
    return HierarchicalTaskAdaptivePoseField(
        source_channels={1: 16, 2: 32},
        hidden_dim=16,
        output_size=(24, 8),
        boot_epochs=10,
        handoff_start_epoch=6,
    )


def _inputs(batch=2):
    torch.manual_seed(17)
    feature1 = torch.randn(batch, 16, 12, 4, requires_grad=True)
    feature2 = torch.randn(batch, 32, 6, 2, requires_grad=True)
    teacher = torch.zeros(batch, 17, 24, 8)
    for joint in range(17):
        y = 2 + (joint * 3) % 20
        x = 1 + (joint * 2) % 6
        teacher[:, joint, y, x] = 0.25 + 0.04 * (joint % 10)
    scores = teacher.flatten(2).amax(dim=-1)
    return feature1, feature2, teacher, scores


def _run_two_stages(module, feature1, feature2, teacher=None, scores=None):
    field1, state1, record1 = module.forward_stage(
        1, feature1, teacher, scores)
    field2, state2, record2 = module.forward_stage(
        2, feature2, teacher, scores, prior_state=state1)
    data = module.aggregate_stage_data((record1, record2))
    return field1, field2, state1, state2, record1, record2, data


def _has_nonzero_grad(parameters):
    return any(parameter.grad is not None
               and bool(torch.isfinite(parameter.grad).all())
               and float(parameter.grad.abs().sum()) > 0.0
               for parameter in parameters)


def test_schedule_matches_single_point_d0_handoff():
    module = _module()
    expected = {1: 0.0, 5: 0.0, 6: 0.2, 8: 0.6, 10: 1.0,
                11: 1.0}
    for epoch, fraction in expected.items():
        module.set_epoch(epoch)
        assert abs(module._student_fraction() - fraction) < 1e-8


def test_stage_projections_feed_exactly_one_shared_decoder():
    module = _module()
    assert set(module.stage_projections.keys()) == {'1', '2'}
    assert not isinstance(module.anchor, torch.nn.ModuleList)
    assert not isinstance(module.anchor, torch.nn.ModuleDict)
    decoder_names = [name for name, _ in module.named_modules()
                     if name == 'anchor']
    assert decoder_names == ['anchor']


def test_epoch1_routes_teacher_to_both_consumers_and_averages_losses():
    feature1, feature2, teacher, scores = _inputs()
    module = _module().train()
    module.set_epoch(1)
    (field1, field2, _, _, record1, record2,
     data) = _run_two_stages(
         module, feature1, feature2, teacher, scores)
    torch.testing.assert_close(field1.float(), teacher, rtol=0, atol=0)
    torch.testing.assert_close(field2.float(), teacher, rtol=0, atol=0)
    expected_loss = 0.5 * (
        record1['pose_loss'] + record2['pose_loss'])
    torch.testing.assert_close(
        data['tapf_pose_loss'], expected_loss, rtol=0, atol=0)
    assert float(data['tapf_stats']['hierarchical_stage_count']) == 2.0
    assert float(data['tapf_stats']['stage1_refinement_active']) == 0.0
    assert float(data['tapf_stats']['stage2_refinement_active']) == 1.0


def test_continuous_pose_loss_updates_both_projections_and_shared_decoder_only():
    feature1, feature2, teacher, scores = _inputs()
    module = _module().train()
    module.set_epoch(11)
    field1, field2, _, _, _, _, data = _run_two_stages(
        module, feature1, feature2, teacher, scores)
    assert not field1.requires_grad
    assert not field2.requires_grad
    data['tapf_pose_loss'].backward()
    assert _has_nonzero_grad(module.stage_projections['1'].parameters())
    assert _has_nonzero_grad(module.stage_projections['2'].parameters())
    assert _has_nonzero_grad(module.anchor.parameters())
    assert feature1.grad is None
    assert feature2.grad is None


def test_reid_objective_cannot_update_pose_module():
    feature1, feature2, teacher, scores = _inputs()
    module = _module().train()
    module.set_epoch(11)
    field1, field2, _, _, _, _, _ = _run_two_stages(
        module, feature1, feature2, teacher, scores)
    module.zero_grad(set_to_none=True)
    # Mimic the shared ReID path: it depends on RGB features and consumes the
    # detached fields, but it must not backpropagate into the pose module.
    reid_loss = (feature1.square().mean() + feature2.square().mean()
                 + field1.float().mean() + field2.float().mean())
    reid_loss.backward()
    assert feature1.grad is not None
    assert feature2.grad is not None
    assert not _has_nonzero_grad(module.parameters())


def test_deeper_stage_is_a_real_refinement_of_the_previous_state():
    feature1, feature2, _, _ = _inputs()
    module = _module().eval()
    field1, state1, _ = module.forward_stage(1, feature1)
    field2, _, record2 = module.forward_stage(
        2, feature2, prior_state=state1)

    shifted_state = dict(state1)
    shifted_state['probability'] = torch.roll(
        state1['probability'], shifts=3, dims=-1)
    shifted_state['confidence'] = (
        0.85 * state1['confidence'] + 0.05).clamp(0.0, 1.0)
    shifted_field2, _, shifted_record2 = module.forward_stage(
        2, feature2, prior_state=shifted_state)

    assert field1.shape == field2.shape == shifted_field2.shape
    assert float((field2 - shifted_field2).abs().max()) > 0.0
    assert float(record2['stats']['posterior_refinement_l1']) > 0.0
    assert float(shifted_record2['stats'][
        'posterior_refinement_l1']) > 0.0


def test_eval_is_rgb_only_and_stage_order_is_strict():
    feature1, feature2, _, _ = _inputs()
    module = _module().eval()
    field1, state1, record1 = module.forward_stage(1, feature1)
    field2, _, record2 = module.forward_stage(
        2, feature2, prior_state=state1)
    data = module.aggregate_stage_data((record1, record2))
    assert field1.shape == field2.shape == (2, 17, 24, 8)
    assert data['tapf_pose_loss'] is None
    assert bool(torch.isfinite(field1).all())
    assert bool(torch.isfinite(field2).all())

    try:
        module.forward_stage(2, feature2)
    except ValueError as error:
        assert 'stage order mismatch' in str(error)
    else:
        raise AssertionError('out-of-order hierarchical stage was accepted')


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print('all %d hierarchical TAPF tests passed' % len(tests))
