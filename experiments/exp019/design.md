# 实验 exp019: Pose Cross-Attention (PXA)

## 动机
- PSG (exp007) 是目前最佳方法 (+1.7% mAP)，但采用逐位置独立的元素级门控 (element-wise gating)
- PSG 的局限：位置 (h,w) 的 gate 只由该位置的 heatmap 值决定，无法利用全局人体结构
- 例如：一个被遮挡位置无法知道身体其他可见部位的信息，只能被 PSG 抑制
- PAB (exp012, +0.8%) 尝试修改 attention，但限于分解形式 (bias[i,j] = val[i] + val[j]) 和窗口局部性
- 需要一种机制让每个特征位置可以看到**全局**的姿态信息

## 创新点 / 核心想法
- **核心假设**: 全局姿态上下文比逐位置门控更有效。通过 cross-attention，每个特征位置可以查询所有 pose 位置，获取完整的人体结构信息。
- 与 PSG 对比：PSG 是 FiLM 式元素级调制 (position-local)，PXA 是 cross-attention 式全局聚合 (position-global)
- 与 PAB 对比：PAB 在 Swin 现有 self-attention 中加 bias (窗口局部)，PXA 是独立的 cross-attention 层 (全局)

## 技术方案

### 新增模块: PoseCrossAttention (model/modules/pose_cross_attention.py)

```
输入: features (B, N, C=768), heatmaps (B, 17, hH, hW)
1. Pose 编码: pose_tokens = Conv2d(17, d, 1)(resize(sigmoid(heatmaps))) → (B, N, d)  [d=64]
2. Query 投影: Q = Linear(768, 64)(features)  → (B, N, 64)
3. Cross-attention: attn = softmax(Q @ K^T / sqrt(64)) @ V  [K=V=pose_tokens]  → (B, N, 64)
4. Output 投影: update = Linear(64, 768)(attn)  → (B, N, 768)  [零初始化]
5. 残差连接: output = features + update
```

### 数据流
- 图片 → Swin Stages 0-2 → Stage 3 Block 0 → **PXA** → Stage 3 Block 1 → **PXA** → GAP → BN → Classifier
- 姿态热图与 PXA 并行输入（与 PSG 相同位置，但不同机制）

### 关键超参数
- hidden_dim (d) = 64: Query/Key/Value 维度，与 PSG 保持一致
- pose_channels = 17: COCO 关键点数
- 零初始化 out_proj: 保证训练开始时 PXA 为恒等映射

### 参数量估算
每个 PXA 模块:
- pose_proj: Conv2d(17, 64, 1) = 17×64 + 64 = 1,152
- q_proj: Linear(768, 64) = 768×64 + 64 = 49,216
- out_proj: Linear(64, 768) = 64×768 + 768 = 49,920 (零初始化)
- 合计: ~100K per block × 2 blocks = ~200K (PSG 的 2 倍，仍然非常轻量)

### 修改文件
1. 新增: `model/modules/pose_cross_attention.py` — PXA 模块实现
2. 修改: `model/pose_backbone_model.py` — 添加 PXA 注入模式
3. 修改: `model/make_model.py` — 无需修改 (复用 POSE_BACKBONE_PSG 入口)
4. 新增: `configs/occluded_duke/pose_pxa.yml` — PXA 配置

### Config 开关
```yaml
POSE_CROSS_ATTN: True    # 使用 PXA 替代 PSG
POSE_BACKBONE_PSG: True   # 保持为 True 以使用 PoseBackboneModel
POSE_PFM_HIDDEN: 64       # hidden_dim 复用
```

## 预期结果
- **乐观**: mAP 59%+ — 全局姿态上下文带来显著提升
- **中性**: mAP 57-58% — PXA 有效但不优于 PSG，两种机制效果相当
- **悲观**: mAP < 57% — cross-attention 对 12×4 的小 feature map 不适合
- 如果失败，最可能的原因：12×4=48 tokens 的 cross-attention 中，信号太弱（heatmap 只有 17 通道有效信息）

## 对照组
- Baseline 对照: exp000 (mAP 56.6%)
- 核心对比: exp007 PSG (mAP 58.3%)
- 消融变量: 仅改变 pose 注入机制 (PSG element-wise gating → PXA cross-attention)，其余全部相同
