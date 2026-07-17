# 实验 exp052: KP-RPE (Keypoint Relative Position Encoding)

## 动机
- PSG 证明 backbone 级 pose injection 有效（+1.33% mAP, 3-seed confirmed）
- PAB（exp012）在 attention bias 中注入 pose 信息，效果弱于 PSG（+0.8% vs +1.7%），且 combo 失败（exp013 -0.7%）
- PAB 失败的根本原因：**additive decomposition** `bias(i,j) = val[i] + val[j]` 是一阶近似，丢失了 token-pair 之间的真正结构关系
- 5 个辅助 loss 全部失败 → 需要架构级改动，不是 loss 修改
- 动机链：PSG（空间门控成功）→ PAB（attention bias 有信号但不够强）→ KP-RPE（修复 PAB 的核心缺陷）

## 创新点 / 核心想法
- **将 Swin 的相对位置编码从欧氏空间推广到人体关键点定义的结构空间**
- 对于窗口内每对 token (i, j)，计算它们到 17 个身体关键点的距离差 r_ij = d_i - d_j（17 维向量）
- 通过小 MLP 将 r_ij 映射为 per-head attention bias，加入 attention score
- 与 Swin 原生 RPE（编码网格空间相对位置）互补：原生 RPE 知道"两个 patch 相距 3 个位置"，KP-RPE 知道"这两个 patch 一个靠近左肩、一个靠近右膝"

## 技术方案

### 核心模块: KeypointRPE
```python
class KeypointRPE(nn.Module):
    def __init__(self, num_keypoints=17, num_heads=24, hidden_dim=32):
        self.mlp = nn.Sequential(
            nn.Linear(num_keypoints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        # 零初始化确保安全退化
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token_kp_distances):
        # token_kp_distances: (B*nW, ws*ws, 17)
        # d_diff: (B*nW, ws*ws, ws*ws, 17) — pairwise distance differences
        d_diff = token_kp_distances.unsqueeze(2) - token_kp_distances.unsqueeze(1)
        # bias: (B*nW, ws*ws, ws*ws, num_heads)
        bias = self.mlp(d_diff)
        # -> (B*nW, num_heads, ws*ws, ws*ws)
        return bias.permute(0, 3, 1, 2)
```

### 距离计算
1. 将 person 0 的关键点坐标从像素空间映射到 Stage 3 feature map 坐标系（除以 patch_size*stride）
2. 对 feature map 上每个 token 位置 (h, w)，计算到 17 个关键点的 L2 距离
3. 不可靠关键点（score < 0.3）的距离设为 0（零化其贡献）
4. 距离归一化到 [0, 1] 范围（除以 feature map 对角线长度）

### 数据流
```
input image → Swin Stage 0-2 → Stage 3 feature map (12x4)
pose_dict → person 0 keypoints (17x2) → 映射到 feature map 坐标
  → compute_token_kp_distances(token_positions, kp_positions) → (B, 48, 17)
  → pad to (B, H_pad*W_pad, 17) where H_pad=14, W_pad=7
  → [block 1: cyclic shift distances]
  → window_partition → (B*nW, ws*ws, 17)
  → KeypointRPE.forward() → (B*nW, num_heads, ws*ws, ws*ws)
  → add to attention scores in WindowMSA
```

### 修改文件
1. `model/modules/keypoint_rpe.py` — 新增 KeypointRPE 模块
2. `model/pose_backbone_model.py` — 添加 KP-RPE 模式（类似 PAB 的 wiring）
3. `model/backbones/swin_transformer.py` — 可能需要微调 ShiftWindowMSA 的 distance 传递逻辑
4. `config/defaults.py` — 添加 POSE_KP_RPE 配置项
5. `configs/occluded_duke/pose_psg_gcn_kprpe.yml` — 实验配置

### 关键超参数
- `hidden_dim`: 32（MLP 隐藏维度）
- `num_keypoints`: 17（COCO）
- `num_heads`: 24（Swin Stage 3）
- `kp_score_threshold`: 0.3（低于此阈值的关键点距离置零）

### 参数量
- MLP: 17*32 + 32 + 32*24 + 24 = 1,368 per block
- 2 blocks × 1,368 = **2,736 total**

### 内存估算
- 距离差张量: (B*nW, 49, 49, 17) float32
  - B=64, nW=2: 64*2*49*49*17*4 bytes ≈ 21MB per block
- MLP hidden: 64*2*49*49*32*4 ≈ 39MB
- MLP output: 64*2*49*49*24*4 ≈ 29MB
- **Total: ~89MB per block, ~178MB for 2 blocks**
- 在 3090 24GB 上可行（当前 PSG+GCN 使用 ~10GB）

## 预期结果
- **如果成功**: mAP 提升 0.5-1.5%（类似 PSG 级别的增益）。证明 pairwise 结构编码优于 unary 编码（PAB），backbone 级 pose injection 仍有提升空间
- **如果中性**: 证明 attention bias 路线在 PSG 存在时确实冗余，12x4 分辨率下结构信息不够
- **如果失败**: PSG+KP-RPE combo 与 PSG+PAB 相同的冲突模式，需要彻底放弃 attention 级 pose 注入

## 对照组
- Baseline 对照: exp030a PSG+GCN equal_concat = 60.73% mAP (3-seed mean)
- 直接对照: exp012 PAB = 57.4% mAP（但 PAB 没有 GCN，需要注意口径差异）
- 消融变量: 相对于 exp030a，唯一改变是在 Stage 3 attention 中添加 KP-RPE bias

## 与 PAB (exp012) 的关键区别
| 维度 | PAB (exp012) | KP-RPE (exp052) |
|------|-------------|-----------------|
| 输入 | 热图强度 (heatmap intensity) | 关键点坐标距离 (kp distance) |
| 编码方式 | Unary: val[i]+val[j] | Pairwise: MLP(d_i - d_j) |
| 信息内容 | "token i 靠近某个关键点" | "token i 和 j 到各关键点的相对距离" |
| 表达力 | Rank-2 | Full rank (通过 MLP) |
| 参数量 | 5.4K | 2.7K |
| 是否包含 GCN | 否 | 是（在 exp030a 基础上添加） |
