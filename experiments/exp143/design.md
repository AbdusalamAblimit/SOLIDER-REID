# 实验 exp143: SASA（Skeleton-Aware Self-Attention）

## 动机

- 141 个实验的核心教训：只有添加**强归纳偏置**且**几乎无新参数**的方法才能在 15K 数据集上生效
- PSG（乘法空间门控）和 PAA（加法适配器）都是这类成功案例
- 但当前所有方法只修改了 **feature values**，从未修改过 **attention routing**
- KP-RPE（exp052）尝试过基于关键点欧氏距离的注意力偏置，结果中性（+0.27%）
  - 但 KP-RPE 用的是 MLP 学习的偏置（需要学习），且只用了欧氏距离（不反映身体拓扑）
- SASA 提出一种**零参数**的注意力偏置：基于**骨架图的测地距离**，不需要学习

## 核心假设

如果让 Swin 的 window self-attention 知道"哪些 token 属于相同/相邻的身体部位"，那么：

1. 同一身体部位的 token 会更强地互相关注（提取更一致的部位特征）
2. 拓扑相邻部位的 token（如肩→肘→腕）会形成信息传播路径
3. 拓扑远距部位的 token（如头→脚）不会被错误关联（减少遮挡物干扰）

与 KP-RPE 的本质区别：
- KP-RPE：每个 token 到每个关键点的**欧氏距离** → MLP → bias（需要学习）
- SASA：每对 token 之间的**骨架测地距离** → 固定查找表 → bias（零参数）

## 技术方案

### 1. 预计算骨架测地距离矩阵

COCO 17 关键点的骨骼连接定义了一个无向图。预计算所有关键点对之间的最短路径长度（测地距离），得到一个 17×17 的固定矩阵 `G`。

```
G[i][j] = shortest_path_length(kp_i, kp_j) on COCO skeleton graph
```

例如：
- G[左肩, 左肘] = 1（直接相连）
- G[左肩, 左腕] = 2（经过左肘）
- G[左肩, 右踝] = 5（经过脊柱和右腿）

### 2. Token-to-Keypoint Assignment

对 feature map 上每个空间位置 (h, w)，利用 pose heatmap 确定其最近的关键点：

```python
# pose_heatmap: (B, 17, H_hm, W_hm)
# 双线性插值到 feature map 分辨率 (12, 4)
heatmap_resized = F.interpolate(pose_heatmap, (12, 4))  # (B, 17, 12, 4)
# 每个位置取响应最强的关键点
token_kp_assign = heatmap_resized.argmax(dim=1)  # (B, 12, 4) -> 值域 [0, 16]
```

### 3. 构造 Skeleton Attention Bias

对每个 window 内的 token 对 (i, j)：

```python
bias[i, j] = -alpha * G[token_kp_assign[i], token_kp_assign[j]]
```

- `alpha > 0` 是唯一的超参数（控制偏置强度）
- 负号表示测地距离越大，注意力偏置越负（减少远距部位的注意力）
- 测地距离为 0 时（同一关键点），无额外偏置（保持原有注意力）

### 4. 集成到 Swin 的 Window Attention

```python
# 原始 Swin attention:
# attn = Q @ K^T / sqrt(d) + relative_position_bias
#
# SASA 修改:
# attn = Q @ K^T / sqrt(d) + relative_position_bias + skeleton_bias
```

注意：
- `relative_position_bias` 是 Swin 原有的学习位置编码（保留）
- `skeleton_bias` 是新增的固定骨架距离偏置
- 两者叠加，不冲突

### 5. 实现位置

- 新模块：`model/modules/skeleton_attention.py`
  - `SkeletonAttentionBias` 类
  - 接收 `pose_heatmap`，输出 `skeleton_bias`
  - 注册 `G` 矩阵为 buffer（固定不学习）
- 修改：`model/pose_backbone_model.py`
  - 在 Stage 3 的 SwinBlock 前计算 skeleton_bias
  - 传入 window attention 模块
- 修改：Swin 的 `WindowAttention.forward`
  - 接收可选的 `extra_bias` 参数
  - 加到注意力分数上

### 6. 关键超参数

- `alpha = 0.1`（初始值，控制偏置强度）
- 仅在 Stage 3 应用（与 PSG 对齐）
- 可以和 PSG、PAA 共存

### 7. 为什么这次可能不同于 KP-RPE

| 维度 | KP-RPE (exp052) | SASA |
|------|-----------------|------|
| 距离定义 | 欧氏距离（空间） | 测地距离（拓扑） |
| 参数 | ~5K（MLP） | 0（固定表） |
| 信息粒度 | token-to-keypoint | token-to-token（经由 keypoint） |
| 过拟合风险 | 有（MLP 在 15K 数据上学习） | 无（零参数） |
| 结构先验强度 | 弱（欧氏距离不反映身体拓扑） | 强（测地距离直接编码骨架连接） |

## 预期结果

- 如果假设成立：mAP +0.3~1.0% over exp030a-eq（保守估计）
- 如果失败：最可能原因是 Swin window 大小限制了 token pair 覆盖，或者 Stage 3 特征已经足够 semantic

## 对照组

- 唯一基线：`exp030a`（PSG+GCN）
- 单变量：仅添加 SASA，不改变其他任何设置
- 报告模式：equal_concat、global

## 风险与失败解释

1. **KP-RPE 中性 → SASA 也中性**
   - 说明 attention routing 在 Swin window attention 下确实影响有限
   - Swin 的 window 太小，body structure prior 无法充分传播
2. **SASA 负面**
   - 说明固定的 topology prior 与数据实际不符（如遮挡下 token assignment 错误）
3. **SASA 中性但 alpha 扫参后有效**
   - 说明 alpha 的选择很关键，需要仔细调

## 日志需求

除标准 loss/mAP 外，记录：
- `sasa_alpha`：当前 alpha 值
- `sasa_bias_mean / sasa_bias_std`：skeleton bias 的统计
- `sasa_assign_entropy`：token-to-keypoint assignment 的熵（高熵说明 assignment 模糊）
