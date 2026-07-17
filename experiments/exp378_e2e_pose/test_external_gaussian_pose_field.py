import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = (Path(__file__).resolve().parents[2] / 'model' / 'modules'
                / 'task_adaptive_pose_field.py')
_SPEC = importlib.util.spec_from_file_location('task_adaptive_pose_field_rg0',
                                               _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GaussianPoseFieldRenderer = _MODULE.GaussianPoseFieldRenderer


def _valid_external_inputs(batch=2):
    torch.manual_seed(378)
    heatmaps = torch.randn(batch, 17, 96, 32) * 0.02
    scores = torch.linspace(0.1, 0.9, 17).repeat(batch, 1)
    for sample in range(batch):
        for joint in range(17):
            y = 3 + (sample * 7 + joint * 5) % 90
            x = 1 + (sample * 3 + joint * 2) % 30
            heatmaps[sample, joint, y, x] = scores[sample, joint] + 0.2
    return heatmaps, scores


def _legacy_reference(heatmaps, scores, sigma_min=0.025,
                      sigma_max=0.25):
    positive = heatmaps.float().clamp_min(0.0)
    confidence = scores.float().clamp(0.0, 1.0)
    mass = positive.flatten(2).sum(dim=-1, keepdim=True)
    probability = (positive.flatten(2) / mass.clamp_min(1e-8)).reshape_as(
        positive)

    ys = torch.linspace(0.0, 1.0, heatmaps.shape[-2], dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, heatmaps.shape[-1], dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_x = grid_x[None, None]
    grid_y = grid_y[None, None]
    mu_x = (probability * grid_x).sum(dim=(-2, -1))
    mu_y = (probability * grid_y).sum(dim=(-2, -1))
    var_x = (probability
             * (grid_x - mu_x[:, :, None, None]).square()).sum(
                 dim=(-2, -1))
    var_y = (probability
             * (grid_y - mu_y[:, :, None, None]).square()).sum(
                 dim=(-2, -1))
    sigma_x = var_x.clamp_min(1e-8).sqrt().clamp(sigma_min, sigma_max)
    sigma_y = var_y.clamp_min(1e-8).sqrt().clamp(sigma_min, sigma_max)
    dx = ((grid_x - mu_x[:, :, None, None])
          / sigma_x[:, :, None, None])
    dy = ((grid_y - mu_y[:, :, None, None])
          / sigma_y[:, :, None, None])
    gaussian = torch.exp(-0.5 * (dx.square() + dy.square()))
    gaussian = gaussian / gaussian.flatten(2).amax(
        dim=-1)[:, :, None, None].clamp_min(1e-8)
    return confidence[:, :, None, None] * gaussian


def test_renderer_is_parameter_free_rng_neutral_and_state_empty():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    renderer = GaussianPoseFieldRenderer()
    after = torch.get_rng_state()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert list(renderer.parameters()) == []
    assert renderer.state_dict() == {}


def test_rg0_renderer_matches_frozen_legacy_equations_exactly():
    heatmaps, scores = _valid_external_inputs()
    renderer = GaussianPoseFieldRenderer(
        output_size=(96, 32), sigma_min=0.025, sigma_max=0.25)
    observed, stats = renderer(heatmaps, scores)
    expected = _legacy_reference(heatmaps, scores)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert observed.shape == (2, 17, 96, 32)
    assert observed.dtype == torch.float32
    assert bool(torch.isfinite(observed).all())
    assert float(observed.min()) >= 0.0
    assert float(observed.max()) <= 1.0
    torch.testing.assert_close(
        observed.flatten(2).amax(dim=-1), scores, rtol=0, atol=0)
    assert set(stats) == {
        'raw_min', 'raw_max', 'raw_peak', 'raw_mean',
        'raw_negative_fraction', 'score_out_of_range_fraction',
        'positive_mass_min', 'positive_mass_mean',
        'active_positive_mass_min', 'near_zero_mass_fraction',
        'near_zero_mass_count', 'confidence_mean', 'mu_x_mean',
        'mu_y_mean', 'sigma_x_min', 'sigma_x_max', 'sigma_x_mean',
        'sigma_y_min', 'sigma_y_max', 'sigma_y_mean',
        'sigma_min_fraction', 'sigma_max_fraction', 'rendered_min',
        'active_sigma_min_fraction', 'active_sigma_max_fraction',
        'rendered_max', 'rendered_peak', 'rendered_mean',
        'rendered_peak_confidence_max_error',
    }
    assert float(stats['rendered_peak_confidence_max_error']) == 0.0


def test_positive_confidence_zero_mass_fails_loudly():
    renderer = GaussianPoseFieldRenderer()
    heatmaps = torch.zeros(1, 17, 96, 32)
    scores = torch.zeros(1, 17)
    scores[0, 4] = 0.7
    try:
        renderer(heatmaps, scores)
    except ValueError as error:
        assert 'positive-confidence joints with zero heatmap mass: 1' in str(
            error)
    else:
        raise AssertionError('RG0 must reject a confident empty joint')


def test_masked_empty_target_renders_exact_zero_field():
    renderer = GaussianPoseFieldRenderer()
    heatmaps = torch.zeros(2, 17, 96, 32)
    scores = torch.zeros(2, 17)
    field, stats = renderer(heatmaps, scores)
    assert torch.equal(field, torch.zeros_like(field))
    assert float(stats['confidence_mean']) == 0.0
    assert float(stats['rendered_peak']) == 0.0


def test_renderer_rejects_nonfinite_and_shape_mismatch():
    renderer = GaussianPoseFieldRenderer()
    heatmaps, scores = _valid_external_inputs(batch=1)
    bad = heatmaps.clone()
    bad[0, 0, 0, 0] = float('nan')
    try:
        renderer(bad, scores)
    except ValueError as error:
        assert 'NaN/Inf' in str(error)
    else:
        raise AssertionError('non-finite heatmaps must fail')

    bad_scores = scores.clone()
    bad_scores[0, 0] = float('inf')
    try:
        renderer(heatmaps, bad_scores)
    except ValueError as error:
        assert 'teacher_scores contain NaN/Inf' in str(error)
    else:
        raise AssertionError('non-finite scores must fail')

    try:
        renderer(heatmaps[:, :, :-1], scores)
    except ValueError as error:
        assert 'spatial shape' in str(error)
    else:
        raise AssertionError('wrong external heatmap size must fail')
