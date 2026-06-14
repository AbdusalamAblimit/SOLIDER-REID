# encoding: utf-8
"""DONOR_DECOUPLE：双出口反事实解耦的合成、头部和损失工具。

默认不开时本文件不会被 make_model 或 processor 引入。开启后，主路只多一个零初始化
残差线性层，初始等价于恒等映射；辅路只服务训练期 donor 分类和正交约束，测试期不参与。
"""
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DonorMainProjector(nn.Module):
    """P_A 的残差恒等实现：x + Linear(x)，Linear 零初始化。"""

    def __init__(self, in_dim):
        super(DonorMainProjector, self).__init__()
        self.delta = nn.Linear(in_dim, in_dim)
        nn.init.constant_(self.delta.weight, 0.0)
        nn.init.constant_(self.delta.bias, 0.0)

    def forward(self, x):
        return x + self.delta(x)


class DonorAuxHead(nn.Module):
    """P_B 小 MLP 加 donor 分类器。第 num_classes 类表示无遮挡。"""

    def __init__(self, in_dim, num_classes, hidden_dim=None):
        super(DonorAuxHead, self).__init__()
        hidden_dim = hidden_dim or max(128, min(512, in_dim // 2))
        self.num_classes = int(num_classes)
        self.no_donor_label = self.num_classes
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )
        self.classifier = nn.Linear(in_dim, self.num_classes + 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, feat_map, donor_rects, detach=True):
        x = feat_map.detach() if detach else feat_map
        pooled = donor_mask_pool(x, donor_rects)
        donor_feat = self.projector(pooled)
        donor_logits = self.classifier(donor_feat)
        return donor_logits, donor_feat


def donor_mask_pool(feat_map, donor_rects):
    """按输入图归一化矩形在最后一层 Swin 特征图上做平均池化。

    donor_rects 为 (B,4)，字段是 top、left、bottom、right，范围是输入图归一化坐标。
    未合成样本传整图矩形，等价于普通全局平均池化。
    """
    if feat_map.dim() != 4:
        raise ValueError("feat_map must have shape (B, C, H, W)")
    B, C, H, W = feat_map.shape
    if donor_rects is None:
        return feat_map.flatten(2).mean(dim=-1)
    if donor_rects.shape[0] != B or donor_rects.shape[1] != 4:
        raise ValueError("donor_rects must have shape (B, 4)")

    rects = donor_rects.to(device=feat_map.device, dtype=torch.float32)
    pooled = []
    for i in range(B):
        top, left, bottom, right = rects[i].detach().cpu().tolist()
        top = max(0.0, min(1.0, float(top)))
        left = max(0.0, min(1.0, float(left)))
        bottom = max(top, min(1.0, float(bottom)))
        right = max(left, min(1.0, float(right)))
        y0 = max(0, min(H - 1, int(top * H)))
        x0 = max(0, min(W - 1, int(left * W)))
        y1 = max(y0 + 1, min(H, int(math.ceil(bottom * H))))
        x1 = max(x0 + 1, min(W, int(math.ceil(right * W))))
        pooled.append(feat_map[i, :, y0:y1, x0:x1].mean(dim=(1, 2)))
    return torch.stack(pooled, dim=0)


def _sample_crop_shape(height, width, rng):
    for _ in range(100):
        h_ratio = rng.uniform(0.45, 0.90)
        w_ratio = rng.uniform(0.35, 0.80)
        area_ratio = h_ratio * w_ratio
        if 0.15 <= area_ratio <= 0.35:
            break
    else:
        h_ratio, w_ratio = 0.60, 0.45
    crop_h = max(1, min(height, int(round(height * h_ratio))))
    crop_w = max(1, min(width, int(round(width * w_ratio))))
    return crop_h, crop_w


def _select_targets_for_donor(remaining, pid_list, donor_pid, donor_repeat, rng, require_full=True):
    shuffled = list(remaining)
    rng.shuffle(shuffled)
    chosen = []
    seen_pids = set()
    for idx in shuffled:
        pid = pid_list[idx]
        if pid == donor_pid or pid in seen_pids:
            continue
        chosen.append(idx)
        seen_pids.add(pid)
        if len(chosen) == donor_repeat:
            break
    if require_full and len(chosen) < donor_repeat:
        return []
    if len(chosen) < 2:
        return []
    return chosen


def build_donor_synth_batch(images, pids, paste_prob=0.5, donor_repeat=4,
                            no_donor_label=None, rng=None):
    """构造 A+B 合成 batch，并返回 donor 标签、贴入矩形和 sameB 分组。

    images 是已增强且已归一化的 (B,C,H,W) 张量。每个 donor crop 优先复用到
    DONOR_REPEAT 个 target，且这些 target 的身份互不相同，从而形成 sameB-diffA 对。
    """
    if images.dim() != 4:
        raise ValueError("images must have shape (B, C, H, W)")
    rng = rng or random
    B, C, H, W = images.shape
    repeat = max(2, int(donor_repeat))
    pid_list = pids.detach().cpu().tolist() if torch.is_tensor(pids) else list(pids)
    pid_list = [int(x) for x in pid_list]
    if no_donor_label is None:
        no_donor_label = max(pid_list) + 1 if pid_list else 0

    out = images.clone()
    donor_labels = torch.full((B,), int(no_donor_label), dtype=torch.long, device=images.device)
    donor_rects = torch.zeros((B, 4), dtype=torch.float32, device=images.device)
    donor_rects[:, 2:] = 1.0
    donor_groups = torch.full((B,), -1, dtype=torch.long, device=images.device)
    donor_sources = torch.full((B,), -1, dtype=torch.long, device=images.device)
    if B < 3 or paste_prob <= 0:
        return out, donor_labels, donor_rects, donor_groups, donor_sources

    remaining = {i for i in range(B) if rng.random() < float(paste_prob)}
    donor_order = list(range(B))
    rng.shuffle(donor_order)
    group_id = 0

    def paste_group(donor_idx, targets):
        nonlocal group_id
        crop_h, crop_w = _sample_crop_shape(H, W, rng)
        src_top = rng.randint(0, H - crop_h)
        src_left = rng.randint(0, W - crop_w)
        patch = images[donor_idx, :, src_top:src_top + crop_h, src_left:src_left + crop_w].detach().clone()
        donor_pid = pid_list[donor_idx]
        for target_idx in targets:
            top = rng.randint(0, H - crop_h)
            left = rng.randint(0, W - crop_w)
            out[target_idx, :, top:top + crop_h, left:left + crop_w] = patch
            donor_labels[target_idx] = donor_pid
            donor_rects[target_idx] = torch.tensor(
                [top / H, left / W, (top + crop_h) / H, (left + crop_w) / W],
                dtype=torch.float32,
                device=images.device,
            )
            donor_groups[target_idx] = group_id
            donor_sources[target_idx] = donor_idx
        group_id += 1

    for donor_idx in donor_order:
        if len(remaining) < 2:
            break
        targets = _select_targets_for_donor(
            remaining, pid_list, pid_list[donor_idx], repeat, rng, require_full=True)
        if not targets:
            continue
        paste_group(donor_idx, targets)
        for target_idx in targets:
            remaining.discard(target_idx)

    if group_id == 0:
        all_targets = set(range(B))
        for donor_idx in donor_order:
            targets = _select_targets_for_donor(
                all_targets, pid_list, pid_list[donor_idx], repeat, rng, require_full=False)
            if targets:
                paste_group(donor_idx, targets)
                break

    return out, donor_labels, donor_rects, donor_groups, donor_sources


def donor_counterfactual_loss(synth_feat, clean_feat):
    synth = F.normalize(synth_feat, dim=1, p=2)
    clean = F.normalize(clean_feat.detach(), dim=1, p=2)
    return (1.0 - (synth * clean).sum(dim=1)).mean()


def donor_sameb_negative_loss(synth_feat, clean_feat, pids, donor_groups, margin=0.02):
    B = synth_feat.shape[0]
    if B < 2:
        return synth_feat.new_zeros(())
    synth = F.normalize(synth_feat, dim=1, p=2)
    clean = F.normalize(clean_feat.detach(), dim=1, p=2)
    sim_synth = synth @ synth.t()
    sim_clean = clean @ clean.t()
    same_group = donor_groups.view(-1, 1).eq(donor_groups.view(1, -1)) & donor_groups.view(-1, 1).ge(0)
    diff_target = pids.view(-1, 1).ne(pids.view(1, -1))
    upper = torch.triu(torch.ones((B, B), dtype=torch.bool, device=synth_feat.device), diagonal=1)
    mask = same_group & diff_target & upper
    if not mask.any():
        return synth_feat.new_zeros(())
    return F.relu(sim_synth[mask] - sim_clean[mask].detach() + float(margin)).mean()


def donor_orth_loss(main_feat, donor_feat):
    main = F.normalize(main_feat, dim=1, p=2)
    aux = F.normalize(donor_feat.detach(), dim=1, p=2)
    return (main * aux).sum(dim=1).pow(2).mean()
