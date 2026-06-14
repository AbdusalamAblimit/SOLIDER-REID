#!/usr/bin/env python3
# encoding: utf-8
"""TARDIS 实现自检（CPU 即可，不占 GPU）。在 SOLIDER-REID 目录下运行：
   python tools/objgate_selfcheck.py
验证：A0 退化（门控 λ=0 时输出与无门控逐数值相等）、全开训练前向+反向、synth_mix 合成。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import cfg
from model import make_model

BASE_CFG = 'configs/occluded_posetrack/swin_tiny.yml'


def build(overrides):
    c = cfg.clone()
    c.merge_from_file(BASE_CFG)
    c.merge_from_list(overrides)
    c.freeze()
    m = make_model(c, num_class=100, camera_num=1, view_num=1, semantic_weight=c.MODEL.SEMANTIC_WEIGHT)
    return m, c


def main():
    torch.manual_seed(0)
    # 1) A0 退化自检：门控开启但 λ=0、α=γ=0，输出应等于把门控置 None
    m, _ = build(['MODEL.PRETRAIN_PATH', '', 'MODEL.PRETRAIN_CHOICE', 'self',
                  'OBJGATE.ENABLED', 'True', 'OBJGATE.LAMBDA_TARGET', '0.0',
                  'OBJGATE.SPLIT_W', '0.0', 'OBJGATE.ANTI_W', '0.0'])
    assert m.objgate is not None, '门控未构造'
    m.eval().cuda()
    x = torch.randn(2, 3, 384, 128).cuda()
    with torch.no_grad():
        f_on, _ = m(x)
        g = m.objgate
        m.objgate = None
        f_off, _ = m(x)
        m.objgate = g
    diff = (f_on - f_off).abs().max().item()
    print('A0 退化 max|diff| =', diff)
    assert diff < 1e-4, 'A0 退化失败：门控 λ=0 时改变了输出'

    # 2) 全开训练前向 + 反向
    torch.manual_seed(0)
    m2, _ = build(['MODEL.PRETRAIN_PATH', '', 'MODEL.PRETRAIN_CHOICE', 'self',
                   'OBJGATE.ENABLED', 'True', 'OBJGATE.LAMBDA_TARGET', '1.0',
                   'OBJGATE.SPLIT_W', '1.0', 'OBJGATE.ANTI_W', '0.1',
                   'OBJGATE.ENTROPY_MIN', '0.7', 'OBJGATE.ENTROPY_MAX', '3.0'])
    m2.train().cuda()
    B = 8
    x2 = torch.randn(B, 3, 384, 128).cuda()
    lbl = torch.randint(0, 100, (B,)).cuda()
    is_synth = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1], dtype=torch.float).cuda()
    side = torch.tensor([0, 1, 2, 0, 0, 1, 0, 2], dtype=torch.long).cuda()
    ratio = (torch.rand(B) * 0.4 + 0.3).cuda()
    out = m2(x2, label=lbl, obj_is_synth=is_synth, obj_side=side, obj_ratio=ratio, obj_lambda=1.0)
    assert len(out) == 4, '训练应返回 4 元组'
    score, feat, fmaps, reg = out
    print('train: score', tuple(score.shape), 'feat', tuple(feat.shape), 'reg_loss', float(reg))
    assert torch.isfinite(reg) and reg.item() >= 0, 'reg_loss 异常'
    loss = torch.nn.functional.cross_entropy(score, lbl) + reg
    loss.backward()
    gate_grad = sum(p.grad.abs().sum().item() for p in m2.objgate.parameters() if p.grad is not None)
    print('gate grad sum =', gate_grad)
    assert gate_grad > 0, '门控无梯度，L_split/L_anti 未接上'

    # 3) synth_mix 合成
    from datasets.synth_mix import mix_batch
    imgs = torch.randn(8, 3, 384, 128)
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    o, iss2, sd2, rt2 = mix_batch(imgs, pids, 1.0, 0.3, 0.7, 'both')
    print('mix: out', tuple(o.shape), 'n_synth', int(iss2.sum()), 'sides', sd2.tolist())
    assert o.shape == imgs.shape, '合成输出形状错'
    assert iss2.sum() > 0, '没有任何样本被合成'

    # 4) dataloader 合约：collate 字段数必须和 processor 解包统一按 OBJGATE.ENABLED 决定
    from datasets import make_dataloader
    import os.path as _osp
    root = '/root/reid-clean/data'
    for mp in ['0.5', '0.0']:
        c = cfg.clone()
        c.merge_from_file('configs/occluded_posetrack/swin_tiny_objgate.yml')
        c.merge_from_list(['INPUT.MIX_PROB', mp, 'DATALOADER.NUM_WORKERS', '2', 'DATASETS.ROOT_DIR', root])
        c.freeze()
        tl = make_dataloader(c)[0]
        b = next(iter(tl))
        print('OBJGATE on, MIX_PROB=%s: batch len=%d, is_synth_sum=%.0f' % (mp, len(b), float(b[4].sum())))
        assert len(b) == 7, 'collate 合约错：门控开启应返回 7 元组，得到 %d' % len(b)
    c2 = cfg.clone()
    c2.merge_from_file('configs/occluded_posetrack/swin_tiny.yml')
    c2.merge_from_list(['DATALOADER.NUM_WORKERS', '2', 'DATASETS.ROOT_DIR', root])
    c2.freeze()
    b2 = next(iter(make_dataloader(c2)[0]))
    print('OBJGATE off (baseline): batch len=%d' % len(b2))
    assert len(b2) == 4, 'baseline collate 应是 4 元组，得到 %d' % len(b2)

    print('ALL SELF-CHECKS PASSED')


if __name__ == '__main__':
    main()
