# encoding: utf-8
"""MULTIHYP（exp003）：全局锚定的多假设集合匹配。

把"目标归属"从单图决策改成检索时的跨图推理：每图除全局特征 g 外，再产出 K 个
"身份假设"槽特征 s_1..s_K（框内可能的多个人）。训练时用批内多实例约束监督槽
（同身份图至少一对槽相近，不同身份不能因一对槽像就整体拉近），槽不直接套目标身份
分类（避免把干扰者强标成目标）。检索时用保守公式：集合匹配只对全局已判可能相似、
且存在唯一一致假设的图像对做有上限的局部修正，α=0 或关闭时精确退化为基线。

退化保证：
- MULTIHYP.ENABLED=False：不构造本头，前向与基线逐数值相等。
- 评测时 α=0：检索距离矩阵 == 基线 euclidean 距离矩阵（逐数值相等，B0 自检）。

吸取 TARDIS 教训：第一版假设分支对主干 detach（只训槽 query/投影/小头），主干只由
原 ID/triplet 训练，保持干净。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHypHead(nn.Module):
    """K 个可学习假设 query，对主干最后一层特征图 token 做单头 cross-attention，
    产出 K 个槽特征（投影到 dim、L2 归一化）。"""

    def __init__(self, in_chans, dim, num_slots=3, detach=True):
        super(MultiHypHead, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.detach = detach
        self.query = nn.Parameter(torch.randn(num_slots, in_chans) * 0.02)  # (K, Cin)
        self.k_proj = nn.Linear(in_chans, in_chans, bias=False)
        self.v_proj = nn.Linear(in_chans, in_chans, bias=False)
        self.out_proj = nn.Linear(in_chans, dim)
        self.scale = in_chans ** -0.5

    def forward(self, feat_map):
        """feat_map: (B,C,H,W) -> slots: (B,K,dim)，已 L2 归一化。"""
        B, C, H, W = feat_map.shape
        x = feat_map
        if self.detach:
            x = x.detach()                          # 假设分支不污染主干（第一版）
        tokens = x.flatten(2).transpose(1, 2)        # (B,N,C)
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B,K,C)
        k = self.k_proj(tokens)                      # (B,N,C)
        v = self.v_proj(tokens)                      # (B,N,C)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)  # (B,K,N)
        slots = attn @ v                             # (B,K,C)
        slots = self.out_proj(slots)                 # (B,K,dim)
        slots = F.normalize(slots, dim=-1, p=2)      # 逐槽 L2 归一化
        return slots


def multihyp_set_loss(slots, pids, pos_margin=0.3, neg_margin=0.7, div_w=0.5, set_temp=0.1):
    """批内多实例集合损失。slots:(B,K,dim) 已归一化；pids:(B,) long。
    - 同身份图像对：K×K 槽距离的 soft-min 应小（至少一对假设相近）。
    - 不同身份图像对：soft-min 应 >= neg_margin（不能因一对槽像就整体拉近）。
    - 多样性：同图内不同槽两两相似度低（逼出"不同人的假设"而非塌成一个）。
    槽不直接做目标身份分类（避免把干扰者强标成目标）。返回标量损失。"""
    B, K, d = slots.shape
    device = slots.device
    # 成对 soft-min 槽距离 D[i,j] = softmin over (a,b) of (1 - cos(s_i^a, s_j^b))
    # 余弦相似 sim[i,j,a,b]
    sim = torch.einsum('iad,jbd->ijab', slots, slots)          # (B,B,K,K)
    dist = 1.0 - sim                                            # (B,B,K,K) ∈[0,2]
    softmin = -set_temp * torch.logsumexp(-dist.reshape(B, B, K * K) / set_temp, dim=-1)  # (B,B)

    same = (pids.unsqueeze(0) == pids.unsqueeze(1))            # (B,B) bool
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = same & (~eye)
    neg_mask = ~same

    loss = slots.new_zeros(())
    if pos_mask.any():
        pos = softmin[pos_mask]
        loss = loss + F.relu(pos - pos_margin).mean()          # 同身份：至少一对槽要近
    if neg_mask.any():
        neg = softmin[neg_mask]
        loss = loss + F.relu(neg_margin - neg).mean()          # 不同身份：最优槽对也要拉开
    # 槽多样性：同图内不同槽两两相似度尽量低
    if K > 1:
        self_sim = torch.einsum('iad,ibd->iab', slots, slots)  # (B,K,K)
        off = self_sim - torch.eye(K, device=device).unsqueeze(0)
        loss = loss + div_w * F.relu(off).pow(2).sum(dim=(1, 2)).mean() / (K * (K - 1))
    return loss


def multihyp_dshs_loss(slots, global_feat, pids, set_margin=0.3, n_hard=10,
                       set_temp=0.1, hard_by='global'):
    """C2 判别充分性对齐损失（DSHS）。把训练目标对齐到检索准则：检索时 d_set 取的是
    K×K 槽距离的 soft-min（最佳槽对），这里就用 soft-min 集合距离做一个硬负 triplet，
    硬负按"全局特征相似度"选（hard_by='global'：与 anchor 全局最像的不同身份图，最可能
    因共享遮挡/泛化外观而被全局特征搞混）——逼最佳槽对这些"全局已 confused"的负样本也
    拉开，使槽去判别恰恰是全局特征失败的那些对（正是检索修正 d_set<d_global 生效的regime）。
    诚实定位：不是"语义角色分配"（检索 soft-min 每对在全 K×K 重选、无持久判别槽），
    而是"用同遮挡硬负把集合匹配训练目标对齐到检索准则"。
    hard_by 控制对照：'global'=DSHS 正式版；'random'=随机负对照（分离"硬负构造"的作用）；
    'set'=按集合距离选最近负（消融硬负来源）。slots:(B,K,d) 已归一化；global_feat:(B,Cg)。
    """
    B, K, d = slots.shape
    device = slots.device
    sim = torch.einsum('iad,jbd->ijab', slots, slots)          # (B,B,K,K)
    dist = 1.0 - sim
    softmin = -set_temp * torch.logsumexp(-dist.reshape(B, B, K * K) / set_temp, dim=-1)  # (B,B)
    same = (pids.unsqueeze(0) == pids.unsqueeze(1))
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = same & (~eye)
    neg_mask = ~same
    if not pos_mask.any() or not neg_mask.any():
        return slots.new_zeros(())
    BIG = slots.new_tensor(1e4)
    # 每 anchor 最佳正样本集合距离
    pos_sd = torch.where(pos_mask, softmin, BIG).min(dim=1).values     # (B,)
    # 负样本硬度排序
    if hard_by == 'global':
        g = F.normalize(global_feat, dim=1)
        hardness = g @ g.t()                                          # 越像越硬
    elif hard_by == 'random':
        hardness = torch.rand(B, B, device=device)
    else:  # 'set'
        hardness = -softmin
    hardness = torch.where(neg_mask, hardness, slots.new_tensor(-1e4))
    nh = min(n_hard, B - 1)
    topk_vals, topk_idx = hardness.topk(nh, dim=1)                    # (B,nh)
    hardneg_sd = torch.gather(softmin, 1, topk_idx)                   # (B,nh)
    sel_valid = topk_vals > -1e3                                      # 选中项确实是真负样本（防 anchor 负样本数<nh 时选到被掩码项）
    anchor_valid = (pos_sd < BIG / 2).unsqueeze(1)                    # 该 anchor 有正样本
    keep = sel_valid & anchor_valid
    tri = F.relu(pos_sd.unsqueeze(1) - hardneg_sd + set_margin) * keep
    denom = keep.sum().clamp(min=1)
    return tri.sum() / denom


@torch.no_grad()
def multihyp_distmat(q_feat, g_feat, C, K, alpha, bonus_cap, gate_tau, gate_sigma,
                     unique_margin, set_temp, chunk_q=512):
    """保守检索距离矩阵（评测时）。q_feat:(Nq,C*(K+1))、g_feat:(Ng,C*(K+1))，
    前 C 为全局 g（已归一化），其后 K*C 为槽（每槽已归一化）。
    d = d_global + alpha * gate(d_global) * unique * clamp(d_set - d_global, -cap, 0)
    修正只取负（集合匹配只能把"有一致假设"的对拉近、且有上限），
    alpha=0 时 d == d_global == 基线 euclidean 距离矩阵（逐数值相等）。
    按 query 分块计算以限制显存（(Nq,Ng,K,K) 中间张量很大）。"""
    import numpy as np
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Nq, Ng = q_feat.shape[0], g_feat.shape[0]
    gg = g_feat[:, :C].contiguous().to(dev)
    gg_sq = gg.pow(2).sum(dim=1)                                # (Ng,)
    gs = g_feat[:, C:].reshape(Ng, K, C).to(dev) if alpha != 0.0 else None
    out = np.zeros((Nq, Ng), dtype=np.float32)
    for s in range(0, Nq, chunk_q):
        e = min(s + chunk_q, Nq)
        qg = q_feat[s:e, :C].contiguous().to(dev)              # (b,C)
        b = qg.shape[0]
        # d_global：逐字节复刻 metrics.euclidean_distance 的 addmm（g 已逐图 L2 归一化），
        # 保证 alpha=0 时与基线完全相同。
        d_global = (qg.pow(2).sum(dim=1, keepdim=True).expand(b, Ng)
                    + gg_sq.unsqueeze(0).expand(b, Ng))
        d_global = d_global.addmm(qg, gg.t(), beta=1, alpha=-2)  # (b,Ng)
        if alpha == 0.0:
            out[s:e] = d_global.cpu().numpy()
            continue
        qs = q_feat[s:e, C:].reshape(b, K, C).to(dev)
        sim = torch.einsum('bkc,gmc->bgkm', qs, gs)            # (b,Ng,K,K)
        dist = (1.0 - sim).reshape(b, Ng, K * K)              # ∈[0,2]
        d_set = -set_temp * torch.logsumexp(-dist / set_temp, dim=-1)   # (b,Ng) soft-min
        if dist.shape[-1] >= 2:                               # K*K>=2 才有次优可比
            d_sorted, _ = torch.sort(dist, dim=-1)
            margin = d_sorted[:, :, 1] - d_sorted[:, :, 0]    # 次小-最小
            unique = torch.sigmoid((margin - unique_margin) / 0.05)
        else:                                                 # K=1：无次优，不做唯一性门控
            unique = torch.ones_like(d_set)
        gate = torch.sigmoid((gate_tau - d_global) / gate_sigma)
        corr = torch.clamp(d_set - d_global, min=-bonus_cap, max=0.0)   # 只减距离、有上限
        d = d_global + alpha * gate * unique * corr
        out[s:e] = d.cpu().numpy()
    return out
