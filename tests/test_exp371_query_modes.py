import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / 'model/modules/clip_part_head.py'
SPEC = importlib.util.spec_from_file_location('exp371_clip_part_head', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CLIPPartHead = MODULE.CLIPPartHead


def _build(mode):
    torch.manual_seed(123)
    return CLIPPartHead(
        feat_dim=8,
        num_classes=3,
        clip_dim=4,
        num_heads=2,
        query_mode=mode,
        query_seed=42,
    )


def test_random_query_modes_have_identical_initial_state():
    frozen = _build('random_frozen')
    learned = _build('random_learned')
    assert torch.equal(frozen.clip_text_features, learned.clip_text_features)
    assert frozen.state_dict().keys() == learned.state_dict().keys()
    for key in frozen.state_dict():
        assert torch.equal(frozen.state_dict()[key], learned.state_dict()[key])


def test_random_query_registration_and_trainable_delta():
    frozen = _build('random_frozen')
    learned = _build('random_learned')
    frozen_buffers = dict(frozen.named_buffers())
    learned_params = dict(learned.named_parameters())
    assert 'clip_text_features' in frozen_buffers
    assert 'clip_text_features' not in dict(frozen.named_parameters())
    assert 'clip_text_features' in learned_params
    assert 'clip_text_features' not in dict(learned.named_buffers())
    frozen_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    learned_trainable = sum(p.numel() for p in learned.parameters() if p.requires_grad)
    assert learned_trainable - frozen_trainable == 6 * 4


def test_random_query_initial_forward_is_identical_and_only_learned_gets_grad():
    frozen = _build('random_frozen').eval()
    learned = _build('random_learned').eval()
    feat = torch.randn(2, 8, 2, 1)
    heatmap = torch.rand(2, 17, 4, 2)
    frozen_out = frozen(feat, heatmap, return_cls=False)
    learned_out = learned(feat, heatmap, return_cls=False)
    for frozen_feat, learned_feat in zip(frozen_out[1], learned_out[1]):
        assert torch.equal(frozen_feat, learned_feat)

    learned.train()
    learned_out = learned(feat, heatmap, return_cls=False)
    learned_loss = sum(x.square().mean() for x in learned_out[1])
    learned_loss.backward()
    grad = learned.clip_text_features.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_unknown_query_mode_fails_closed():
    try:
        _build('not-a-mode')
    except ValueError as exc:
        assert 'query_mode' in str(exc)
    else:
        raise AssertionError('unknown query mode should fail closed')
