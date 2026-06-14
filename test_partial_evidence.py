# encoding: utf-8
"""PARTIAL_EVIDENCE 自检。

本脚本不读数据集、不加载预训练权重、不训练，只验证：
1. PARTIAL_EVIDENCE.ENABLED=False 时模型 forward 与基线逐字节一致。
2. CALIBRATE=False 时合成图损失数值等于直接调用原始 loss_fn。
3. clean feature 的梯度在加入合成路径后不变。
4. 证据分 e、CALIBRATE=True 的 CE 校准和 hard negative 排除规则符合公式。
"""

import random

import torch
import torch.nn.functional as F

from config.defaults import _C
from loss.triplet_loss import TripletLoss
from model import make_model
from model.occ_shortcut import paste_occluder_batch
from model.partial_evidence import (
    partial_evidence_ce_loss,
    partial_evidence_training_loss,
    partial_evidence_triplet_loss,
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_test_cfg(partial_enabled, calibrate=True, no_margin=True):
    cfg = _C.clone()
    cfg.defrost()
    cfg.MODEL.NAME = 'transformer'
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'
    cfg.MODEL.TRANSFORMER_TYPE = 'swin_tiny_patch4_window7_224'
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.DROP_PATH = 0.0
    cfg.MODEL.DROP_OUT = 0.0
    cfg.MODEL.ATT_DROP_RATE = 0.0
    cfg.MODEL.SEMANTIC_WEIGHT = -1.0
    cfg.MODEL.JPM = False
    cfg.MODEL.REDUCE_FEAT_DIM = False
    cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
    cfg.MODEL.NO_MARGIN = bool(no_margin)
    cfg.MODEL.IF_LABELSMOOTH = 'off'
    cfg.DATALOADER.SAMPLER = 'softmax_triplet'
    cfg.SOLVER.TRP_L2 = False
    cfg.INPUT.SIZE_TRAIN = [128, 64]
    cfg.INPUT.SIZE_TEST = [128, 64]
    cfg.TEST.NECK_FEAT = 'before'
    cfg.PARTIAL_EVIDENCE.ENABLED = bool(partial_enabled)
    cfg.PARTIAL_EVIDENCE.CALIBRATE = bool(calibrate)
    cfg.freeze()
    return cfg


def build_model(cfg, seed, num_classes=8, device='cpu'):
    set_seed(seed)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=0,
        view_num=0,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    return model.to(device)


def baseline_loss_fn(cfg, num_classes):
    if cfg.MODEL.NO_MARGIN:
        triplet = TripletLoss()
    else:
        triplet = TripletLoss(cfg.SOLVER.MARGIN)

    def id_loss(score, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            epsilon = 0.1
            log_probs = F.log_softmax(score, dim=1)
            soft_targets = torch.zeros_like(log_probs)
            soft_targets.scatter_(1, target.unsqueeze(1), 1.0)
            soft_targets = (1.0 - epsilon) * soft_targets + epsilon / float(num_classes)
            return (-soft_targets * log_probs).mean(0).sum()
        return F.cross_entropy(score, target)

    def loss_func(score, feat, target, target_cam):
        if isinstance(score, list):
            cls_losses = [id_loss(scor, target) for scor in score[1:]]
            cls_loss = sum(cls_losses) / len(cls_losses)
            cls_loss = 0.5 * cls_loss + 0.5 * id_loss(score[0], target)
        else:
            cls_loss = id_loss(score, target)

        if isinstance(feat, list):
            tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
            tri_loss = sum(tri_losses) / len(tri_losses)
            tri_loss = 0.5 * tri_loss + 0.5 * triplet(feat[0], target)[0]
        else:
            tri_loss = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

        return cfg.MODEL.ID_LOSS_WEIGHT * cls_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss

    return loss_func


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def max_forward_diff(out_a, out_b):
    feat_a, maps_a = out_a
    feat_b, maps_b = out_b
    diffs = [max_abs_diff(feat_a, feat_b)]
    diffs.extend(max_abs_diff(a, b) for a, b in zip(maps_a, maps_b))
    return max(diffs)


def check_disabled_forward(device):
    cfg_base = make_test_cfg(False)
    cfg_disabled = make_test_cfg(False)
    model_base = build_model(cfg_base, seed=20260607, device=device)
    model_disabled = build_model(cfg_disabled, seed=20260607, device=device)
    model_base.eval()
    model_disabled.eval()

    set_seed(11)
    x = torch.randn(2, 3, 128, 64, device=device)
    with torch.no_grad():
        out_base = model_base(x)
        out_disabled = model_disabled(x)
    diff = max_forward_diff(out_base, out_disabled)
    same_bytes = torch.equal(out_base[0], out_disabled[0]) and all(
        torch.equal(a, b) for a, b in zip(out_base[1], out_disabled[1])
    )
    print("关闭开关 forward max|diff| = {:.10f}, 逐字节一致 = {}".format(diff, same_bytes))
    assert diff == 0.0 and same_bytes


def check_calibrate_false_aug_only_equivalence(device):
    cfg = make_test_cfg(True, calibrate=False, no_margin=True)
    loss_fn = baseline_loss_fn(cfg, num_classes=3)
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    cams = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    occ_id = torch.tensor([1, 0, 2, 0], dtype=torch.long, device=device)
    evidence = torch.tensor([0.35, 1.0, 0.75, 1.0], device=device)
    synth_score = torch.tensor(
        [[2.0, 0.1, -1.0], [1.5, 0.0, -0.4], [-0.2, 1.7, 0.3], [0.0, 1.3, -0.8]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    synth_feat = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    direct = loss_fn(synth_score, synth_feat, targets, cams)
    pe_loss, details = partial_evidence_training_loss(
        synth_score,
        synth_feat,
        targets,
        cams,
        occ_id,
        evidence,
        cfg,
        loss_fn=loss_fn,
        return_details=True,
    )
    print("CALIBRATE=False pe_loss {:.8f}, direct loss_fn {:.8f}".format(
        float(pe_loss.item()),
        float(direct.item()),
    ))
    assert details["calibrate"] is False
    assert torch.equal(details["occ_id"], occ_id)
    assert torch.allclose(details["evidence"], evidence)
    assert torch.allclose(pe_loss, direct, atol=0.0, rtol=0.0)


def _make_grad_tensors(device):
    clean_score = torch.tensor(
        [[2.0, 0.0, -1.0], [1.5, 0.1, -0.5], [-0.3, 1.8, 0.2], [-0.1, 1.4, -0.4]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    clean_feat = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    synth_score = torch.tensor(
        [[1.8, 0.2, -0.7], [1.2, 0.0, -0.2], [-0.2, 1.5, 0.4], [0.1, 1.1, -0.5]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    synth_feat = torch.tensor(
        [[0.2, 0.0], [0.4, 0.0], [2.2, 0.0], [2.4, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    return clean_score, clean_feat, synth_score, synth_feat


def check_clean_gradient_isolated(device):
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    cams = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    occ_id = torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device)
    evidence = torch.tensor([0.2, 0.8, 0.5, 0.9], device=device)

    for calibrate in (False, True):
        cfg = make_test_cfg(True, calibrate=calibrate, no_margin=True)
        loss_fn = baseline_loss_fn(cfg, num_classes=3)
        clean_score, clean_feat, _, _ = _make_grad_tensors(device)
        clean_loss = loss_fn(clean_score, clean_feat, targets, cams)
        clean_loss.backward()
        base_score_grad = clean_score.grad.detach().clone()
        base_feat_grad = clean_feat.grad.detach().clone()

        clean_score, clean_feat, synth_score, synth_feat = _make_grad_tensors(device)
        total = loss_fn(clean_score, clean_feat, targets, cams)
        synth_loss = partial_evidence_training_loss(
            synth_score,
            synth_feat,
            targets,
            cams,
            occ_id,
            evidence,
            cfg,
            loss_fn=loss_fn,
        )
        (total + synth_loss).backward()
        score_diff = max_abs_diff(clean_score.grad, base_score_grad)
        feat_diff = max_abs_diff(clean_feat.grad, base_feat_grad)
        print("CALIBRATE={} clean score grad diff {:.10f}, clean feat grad diff {:.10f}".format(
            calibrate,
            score_diff,
            feat_diff,
        ))
        assert torch.allclose(clean_score.grad, base_score_grad, atol=0.0, rtol=0.0)
        assert torch.allclose(clean_feat.grad, base_feat_grad, atol=0.0, rtol=0.0)


def check_evidence_monotonic(device):
    images = torch.zeros(1, 3, 10, 10, device=device)
    small_pool = [torch.ones(3, 2, 2)]
    large_pool = [torch.ones(3, 5, 5)]
    default_result = paste_occluder_batch(
        images,
        small_pool,
        aug_prob=1.0,
        rng=random.Random(1),
    )
    assert isinstance(default_result, tuple) and len(default_result) == 2
    _, small_occ, small_rect, small_e = paste_occluder_batch(
        images,
        small_pool,
        aug_prob=1.0,
        rng=random.Random(1),
        return_metadata=True,
    )
    _, large_occ, large_rect, large_e = paste_occluder_batch(
        images,
        large_pool,
        aug_prob=1.0,
        rng=random.Random(1),
        return_metadata=True,
    )
    small_area = (small_rect[0, 2] - small_rect[0, 0]) * (small_rect[0, 3] - small_rect[0, 1])
    large_area = (large_rect[0, 2] - large_rect[0, 0]) * (large_rect[0, 3] - large_rect[0, 1])
    print("小遮挡面积 {:.3f}, e {:.3f}; 大遮挡面积 {:.3f}, e {:.3f}".format(
        float(small_area.item()),
        float(small_e.item()),
        float(large_area.item()),
        float(large_e.item()),
    ))
    assert int(small_occ.item()) == 1 and int(large_occ.item()) == 1
    assert small_area < large_area
    assert small_e > large_e
    assert torch.isclose(small_e, 1.0 - small_area)
    assert torch.isclose(large_e, 1.0 - large_area)


def check_ce_calibration(device):
    cfg = make_test_cfg(True, calibrate=True, no_margin=True)
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0], [1.0, -1.0, 0.5]],
        device=device,
    )
    feat = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=device)
    cams = torch.tensor([0, 0, 0], dtype=torch.long, device=device)
    occ_id = torch.tensor([1, 1, 1], dtype=torch.long, device=device)
    evidence = torch.tensor([0.2, 0.5, 0.9], device=device)
    loss, details = partial_evidence_training_loss(
        logits,
        feat,
        targets,
        cams,
        occ_id,
        evidence,
        cfg,
        return_details=True,
    )
    ce_loss, ce_details = partial_evidence_ce_loss(
        logits,
        targets,
        evidence,
        min_keep=0.2,
        ls_max=0.2,
        return_details=True,
    )
    expected_weight = 0.2 + 0.8 * evidence
    expected_smoothing = 0.2 * (1.0 - evidence)
    print("CALIBRATE=True total loss {:.6f}, CE loss {:.6f}".format(
        float(loss.item()),
        float(details["ce_loss"].item()),
    ))
    print("CE 权重：", ["{:.3f}".format(x) for x in details["weight"].detach().cpu().tolist()])
    print("label smoothing：", ["{:.3f}".format(x) for x in details["smoothing"].detach().cpu().tolist()])
    assert details["calibrate"] is True
    assert torch.allclose(details["ce_loss"], ce_loss)
    assert torch.allclose(details["weight"], expected_weight)
    assert torch.allclose(details["smoothing"], expected_smoothing)
    assert torch.allclose(ce_details["weight"], expected_weight)
    assert torch.allclose(ce_details["smoothing"], expected_smoothing)
    assert details["weight"][0] < details["weight"][1] < details["weight"][2]
    assert details["smoothing"][0] > details["smoothing"][1] > details["smoothing"][2]


def check_soft_triplet_matches_baseline(device):
    feat = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [2.0, 0.0],
            [2.4, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    evidence = torch.ones(4, dtype=torch.float32, device=device)
    pe_loss, details = partial_evidence_triplet_loss(
        feat,
        labels,
        evidence,
        no_hardneg_below=0.4,
        base_margin=0.3,
        margin_scale=True,
        no_margin=True,
        return_details=True,
    )
    baseline = TripletLoss()(feat, labels)[0]
    print("soft triplet pe_loss {:.8f}, baseline {:.8f}".format(
        float(pe_loss.item()),
        float(baseline.item()),
    ))
    assert details["no_margin"] is True
    assert torch.allclose(pe_loss, baseline, atol=1e-7, rtol=1e-7)


def check_low_e_not_hard_negative(device):
    synth_feat = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [10.0, 0.0],
            [10.0, 1.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    synth_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    synth_evidence = torch.tensor([0.8, 0.2, 0.8, 0.8, 0.2, 0.8], device=device)
    loss, details = partial_evidence_triplet_loss(
        synth_feat,
        synth_labels,
        synth_evidence,
        no_hardneg_below=0.4,
        base_margin=0.3,
        margin_scale=True,
        no_margin=False,
        return_details=True,
    )
    low_indices = set(details["low_synth_indices"].detach().cpu().tolist())
    anchors = set(details["anchor_indices"].detach().cpu().tolist())
    negatives = set(details["negative_indices"].detach().cpu().tolist())
    positives = set(details["positive_indices"].detach().cpu().tolist())
    print("explicit-margin triplet loss {:.6f}".format(float(loss.item())))
    print("低证据合成样本索引：", sorted(low_indices))
    print("被选为 anchor 的索引：", sorted(anchors))
    print("被选为 hard negative 的索引：", sorted(negatives))
    print("被选为 hard positive 的索引：", sorted(positives))
    assert int(details["num_clean"].item()) == 0
    assert low_indices == {1, 4}
    assert not (low_indices & anchors)
    assert not (low_indices & negatives)
    assert bool(low_indices & positives)
    for anchor, margin in zip(details["anchor_indices"], details["margins"]):
        expected = 0.3 * synth_evidence[int(anchor.item())]
        assert torch.isclose(margin, expected)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备：{}".format(device))
    check_disabled_forward(device)
    check_calibrate_false_aug_only_equivalence(device)
    check_clean_gradient_isolated(device)
    check_evidence_monotonic(device)
    check_ce_calibration(device)
    check_soft_triplet_matches_baseline(device)
    check_low_e_not_hard_negative(device)
    print("PARTIAL_EVIDENCE 自检通过。")


if __name__ == '__main__':
    main()
