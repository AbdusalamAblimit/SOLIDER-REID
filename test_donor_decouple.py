# encoding: utf-8
"""DONOR_DECOUPLE 自检。

本脚本不读数据集、不加载预训练权重、不训练，只验证三件事：
1. DONOR_DECOUPLE.ENABLED=False 时不构造 donor 模块，两个同 seed 模型 forward 输出逐字节一致。
2. ENABLED=True 的合成构造能产生 sameB-diffA 对，并打印 donor 复用统计。
3. ENABLED=True 时 P_A 的残差恒等初始化使初始主 embedding 与基线 embedding 一致。
"""
import random

import torch

from config.defaults import _C
from model import make_model
from model.donor_decouple import build_donor_synth_batch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_test_cfg(donor_enabled):
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
    cfg.INPUT.SIZE_TRAIN = [128, 64]
    cfg.INPUT.SIZE_TEST = [128, 64]
    cfg.TEST.NECK_FEAT = 'before'
    cfg.DONOR_DECOUPLE.ENABLED = bool(donor_enabled)
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


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def max_forward_diff(out_a, out_b):
    feat_a, maps_a = out_a
    feat_b, maps_b = out_b
    diffs = [max_abs_diff(feat_a, feat_b)]
    diffs.extend(max_abs_diff(a, b) for a, b in zip(maps_a, maps_b))
    return max(diffs)


def check_disabled_forward(device):
    cfg = make_test_cfg(False)
    model_a = build_model(cfg, seed=20260607, device=device)
    model_b = build_model(cfg, seed=20260607, device=device)
    model_a.eval()
    model_b.eval()
    assert getattr(model_a, 'donor_pa', None) is None
    assert getattr(model_a, 'donor_aux_head', None) is None

    set_seed(11)
    x = torch.randn(2, 3, 128, 64, device=device)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    diff = max_forward_diff(out_a, out_b)
    same_bytes = torch.equal(out_a[0], out_b[0]) and all(
        torch.equal(a, b) for a, b in zip(out_a[1], out_b[1])
    )
    print("关闭开关 forward max|diff| = {:.10f}, 逐字节一致 = {}".format(diff, same_bytes))
    assert diff == 0.0 and same_bytes


def check_synth_sameb(device):
    set_seed(22)
    images = torch.randn(12, 3, 128, 64, device=device)
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.long, device=device)
    synth, donor_label, donor_rect, donor_group, donor_source = build_donor_synth_batch(
        images,
        pids,
        paste_prob=1.0,
        donor_repeat=4,
        no_donor_label=6,
        rng=random.Random(20260607),
    )
    del synth
    group_ids = sorted(g for g in set(donor_group.cpu().tolist()) if g >= 0)
    print("donor 复用统计：")
    has_sameb_diffa = False
    for group_id in group_ids:
        idx = (donor_group == group_id).nonzero(as_tuple=False).flatten()
        target_pids = pids[idx].cpu().tolist()
        donor_pids = donor_label[idx].cpu().tolist()
        donor_sources = donor_source[idx].cpu().tolist()
        area = ((donor_rect[idx, 2] - donor_rect[idx, 0]) *
                (donor_rect[idx, 3] - donor_rect[idx, 1])).cpu().tolist()
        print("  group {}: donor_src={}, donor_pid={}, repeat={}, target_pids={}, area={}".format(
            group_id,
            donor_sources[0],
            donor_pids[0],
            len(idx),
            target_pids,
            ["{:.3f}".format(x) for x in area],
        ))
        if len(idx) >= 2 and len(set(target_pids)) >= 2 and len(set(donor_pids)) == 1:
            has_sameb_diffa = True
        assert all(donor_pid != target_pid for donor_pid, target_pid in zip(donor_pids, target_pids))
    no_donor_count = int((donor_label == 6).sum().item())
    print("无遮挡标签数量：{}".format(no_donor_count))
    assert has_sameb_diffa


def check_identity_pa(device):
    cfg_base = make_test_cfg(False)
    cfg_donor = make_test_cfg(True)
    model_base = build_model(cfg_base, seed=20260608, device=device)
    model_donor = build_model(cfg_donor, seed=20260608, device=device)
    model_base.eval()
    model_donor.eval()
    assert getattr(model_donor, 'donor_pa', None) is not None
    assert getattr(model_donor, 'donor_aux_head', None) is not None

    set_seed(33)
    x = torch.randn(2, 3, 128, 64, device=device)
    with torch.no_grad():
        feat_base, _ = model_base(x)
        feat_donor, _ = model_donor(x)
    diff = max_abs_diff(feat_base, feat_donor)
    print("P_A 初始恒等 embedding max|diff| = {:.10f}".format(diff))
    assert diff < 1e-6


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("使用设备：{}".format(device))
    check_disabled_forward(device)
    check_synth_sameb(device)
    check_identity_pa(device)
    print("DONOR_DECOUPLE 自检通过。")


if __name__ == '__main__':
    main()
