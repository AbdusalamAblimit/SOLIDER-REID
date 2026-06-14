# encoding: utf-8
"""TARDIS 核心模块：身份条件目标性门控（Objectness Gate）。

输入主干最后一层特征图 feat_map (B,C,H,W) 和全局平均特征 global_feat (B,C)。
预测逐位置目标性打分 s，softmax 归一化成门控权重 a，加权聚合得 f_obj，
再与 global_feat 按 lam 融合：f = (1-lam)*global_feat + lam*f_obj。

退化保证：lam=0 时 f 数值上精确等于 global_feat（即原全局平均池化特征），
所以 OBJGATE.ENABLED=True 但 LAMBDA=0 时输出与基线逐数值相等。

训练时若给定合成样本的目标侧信息，计算两个正则：
- L_split（仅合成样本）：把打分 s 与"目标侧网格 mask"做 BCE，并加"目标侧门控权重之和趋近 1"
  的质量约束，直接教门控激活目标侧、压制旁人侧。这是唯一真正"按身份归属分人"的方向性信号。
- L_anti（所有样本）：门控分布香农熵的双向约束，既不许塌成全图均匀（退化为平均池化、失去归属能力），
  也不许塌成单点。

注意：门控对每个空间位置打分，与位置先验无关；"目标"由 L_split 在去偏合成分布上的监督决定
（目标侧与位置/占比统计独立），而非靠居中/占大。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectnessGate(nn.Module):
    def __init__(self, in_chans, hidden=192, tau=1.0, entropy_min=0.0, entropy_max=1e9,
                 detach_score=False, mode='softmax', suppress_min=0.5):
        super(ObjectnessGate, self).__init__()
        self.tau = tau
        self.detach_score = detach_score
        self.mode = mode                 # 'softmax'(原尖注意力替换池化) | 'suppress'(宽池化软抑制)
        self.suppress_min = suppress_min  # suppress 模式权重下限；=1 时退化为全局平均池化
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        groups = 32 if hidden % 32 == 0 else 1
        self.score = nn.Sequential(
            nn.Conv2d(in_chans, hidden, kernel_size=1),
            nn.GroupNorm(num_groups=groups, num_channels=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat_map, global_feat, lam,
                is_synth=None, split_side=None, split_ratio=None,
                split_w=0.0, anti_w=0.0):
        """返回 (f, reg_loss)。
        feat_map: (B,C,H,W)；global_feat: (B,C)；lam: 标量。
        is_synth: (B,) 0/1；split_side: (B,) 0=目标在左,1=目标在右,2=整图为目标(self_mix)；
        split_ratio: (B,) 目标侧占宽度的比例（cross_id 用）。
        """
        B, C, H, W = feat_map.shape
        N = H * W
        # DETACH_SCORE=True：打分头吃 detach 后的特征，L_split 与注意力的梯度都不进主干；
        # 主干只经下面值路径 f_obj 的 fmap（注意力 a 作固定权重）被 L_id/L_tri 训练，保持干净。
        score_in = feat_map.detach() if self.detach_score else feat_map
        s = self.score(score_in).view(B, N)                 # (B,N) 逐位置打分
        if self.mode == 'suppress':
            # 宽池化软抑制：g∈[suppress_min,1]，只轻降疑似非目标区、保留全局池化主体（不塌成尖注意力，
            # 避免"尖门控→窄值路径→主干欠训退化"的病根）。suppress_min=1 时 g 恒1、a 均匀，
            # f_obj 精确等于全局平均池化（退化）。
            g = self.suppress_min + (1.0 - self.suppress_min) * torch.sigmoid(s)
            a = g / g.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B,N) 归一化的宽权重
        else:
            a = torch.softmax(s / self.tau, dim=1)          # (B,N) 门控权重（softmax 尖注意力）
        fmap = feat_map.view(B, C, N)                       # (B,C,N)
        f_obj = torch.einsum('bn,bcn->bc', a, fmap)         # (B,C)
        f = (1.0 - lam) * global_feat + lam * f_obj         # lam=0 时 f==global_feat

        reg_loss = feat_map.new_zeros(())
        if self.training and (split_w > 0.0 or anti_w > 0.0):
            reg_loss = self._regularizers(
                s, a, H, W, is_synth, split_side, split_ratio, split_w, anti_w)
        # 暴露最近一次门控权重，供分析脚本读取（不参与训练）
        self.last_a = a.detach()
        self.last_s = s.detach()
        return f, reg_loss

    def _regularizers(self, s, a, H, W, is_synth, split_side, split_ratio, split_w, anti_w):
        B, N = a.shape
        loss = a.new_zeros(())

        # L_anti：门控分布香农熵双向约束（所有样本）
        if anti_w > 0.0:
            ent = -(a * a.clamp_min(1e-8).log()).sum(dim=1)             # (B,)
            anti = torch.relu(a.new_tensor(self.entropy_min) - ent) + \
                   torch.relu(ent - a.new_tensor(self.entropy_max))
            loss = loss + anti_w * anti.mean()

        # L_split：仅合成样本，监督门控空间分布对准目标侧
        if split_w > 0.0 and is_synth is not None and split_side is not None:
            synth = is_synth.bool()
            if synth.any():
                M = self._target_mask(synth, split_side, split_ratio, H, W, a.device)  # (Bs,N) 1=目标侧
                ss = s[synth]                                                          # (Bs,N) logits
                bce = F.binary_cross_entropy_with_logits(ss, M)
                a_s = a[synth]
                tgt_mass = (a_s * M).sum(dim=1).clamp_min(1e-8)                         # 目标侧门控质量
                mass = -torch.log(tgt_mass).mean()
                loss = loss + split_w * (bce + mass)
        return loss

    @staticmethod
    def _target_mask(synth, split_side, split_ratio, H, W, device):
        """为合成样本构造目标侧网格 mask，(Bs,H*W)，行优先(h*W+w)。
        side=0 目标在左、side=1 目标在右、side=2 整图为目标(self_mix)。
        宽度方向(W)上按 ratio 划分目标列。"""
        idx = synth.nonzero(as_tuple=True)[0]
        sides = split_side[idx]
        ratios = split_ratio[idx] if split_ratio is not None else torch.full_like(sides, 0.5, dtype=torch.float)
        Bs = idx.numel()
        M = torch.zeros(Bs, H, W, device=device)
        for j in range(Bs):
            side = int(sides[j].item())
            if side == 2:                       # self_mix：整图都是目标
                M[j] = 1.0
                continue
            r = float(ratios[j].item())
            ncol = max(1, min(W, int(round(r * W))))
            if side == 0:                       # 目标在左
                M[j, :, :ncol] = 1.0
            else:                               # 目标在右
                M[j, :, W - ncol:] = 1.0
        return M.view(Bs, H * W)
