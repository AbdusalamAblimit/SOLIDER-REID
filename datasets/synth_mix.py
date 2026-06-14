# encoding: utf-8
"""TARDIS 去偏合成归属监督：在 batch 内构造"多人同框"合成图。

两种合成：
- cross_id：把另一身份的人贴到目标人的旁边（横向，沿宽度拼）。目标侧/占比/左右随机且与
  几何先验统计独立（含目标占小半、在边的反先验样本），逼门控按身份而非按居中/占大锁定目标。
  合成图标签仍是目标人的身份。
- self_mix：用同一张图的两块拼接，制造合成接缝但两侧身份相同（整图都是目标），
  用来反"靠检测合成接缝来分目标"的捷径。

为弱化合成痕迹，贴入前把干扰侧的逐通道均值/方差对齐到目标侧，并对接缝做轻度横向模糊。
操作对象是已归一化的图像张量 (C,H,W)；宽度方向(W)被划分。
随机性由全局 set_seed 控制（普通 Python random / torch）。
"""
import random
import torch
import torch.nn.functional as F


def _match_channel_stats(src, ref, eps=1e-5):
    sm = src.mean(dim=(1, 2), keepdim=True)
    ss = src.std(dim=(1, 2), keepdim=True) + eps
    rm = ref.mean(dim=(1, 2), keepdim=True)
    rs = ref.std(dim=(1, 2), keepdim=True) + eps
    return (src - sm) / ss * rs + rm


def _resize_width(img, new_w):
    C, H, W = img.shape
    if new_w == W:
        return img
    return F.interpolate(img.unsqueeze(0), size=(H, new_w), mode='bilinear', align_corners=False).squeeze(0)


def _feather_seam(comp, col, band=2):
    """对接缝 col 附近 ±band 列做轻度横向均值模糊，弱化硬接缝痕迹。就地修改并返回。"""
    C, H, W = comp.shape
    lo = max(0, col - band)
    hi = min(W, col + band)
    if hi - lo >= 3:
        region = comp[:, :, lo:hi]
        # 沿宽度做 3 窗均值平滑
        smoothed = F.avg_pool1d(region.reshape(C * H, 1, hi - lo), kernel_size=3, stride=1, padding=1).reshape(C, H, hi - lo)
        comp[:, :, lo:hi] = smoothed
    return comp


def composite_cross(target, distr, r, side):
    """target 占宽度比例 r，side=0 目标在左、1 在右。返回合成图 (C,H,W)。"""
    C, H, W = target.shape
    distr = _match_channel_stats(distr, target)
    wt = max(1, min(W - 1, int(round(r * W))))
    wd = W - wt
    t = _resize_width(target, wt)
    d = _resize_width(distr, wd)
    if side == 0:
        comp = torch.cat([t, d], dim=2)
        seam = wt
    else:
        comp = torch.cat([d, t], dim=2)
        seam = wd
    return _feather_seam(comp, seam)


def composite_self(img, r):
    """同图自混合：左半与右半各取一块拼接，整图同身份。返回 (C,H,W)。"""
    C, H, W = img.shape
    wt = max(1, min(W - 1, int(round(r * W))))
    wd = W - wt
    half = max(1, W // 2)
    left = _resize_width(img[:, :, :half], wt)
    right = _resize_width(img[:, :, half:], wd)
    comp = torch.cat([left, right], dim=2)
    return _feather_seam(comp, wt)


def mix_batch(imgs, pids, mix_prob, ratio_lo, ratio_hi, mix_type):
    """在 batch 内做去偏合成。
    imgs: (B,C,H,W) 已归一化张量；pids: 长度 B 的 list/1D tensor。
    返回 out(B,C,H,W), is_synth(B,), split_side(B,) long, split_ratio(B,)。
    split_side: 0 目标在左 / 1 目标在右 / 2 self_mix 整图为目标。"""
    B = imgs.shape[0]
    out = imgs.clone()
    is_synth = torch.zeros(B)
    split_side = torch.zeros(B, dtype=torch.long)
    split_ratio = torch.zeros(B)
    pid_list = pids.tolist() if torch.is_tensor(pids) else list(pids)
    for i in range(B):
        if random.random() >= mix_prob:
            continue
        r = random.uniform(ratio_lo, ratio_hi)
        sd = random.randint(0, 1)
        use_self = (mix_type == 'self_mix') or (mix_type == 'both' and random.random() < 0.5)
        if use_self:
            out[i] = composite_self(imgs[i], r)
            split_side[i] = 2
            is_synth[i] = 1.0
            split_ratio[i] = r
        else:
            cand = [j for j in range(B) if pid_list[j] != pid_list[i]]
            if not cand:
                continue
            j = random.choice(cand)
            out[i] = composite_cross(imgs[i], imgs[j], r, sd)
            split_side[i] = sd
            is_synth[i] = 1.0
            split_ratio[i] = r
    return out, is_synth, split_side, split_ratio
