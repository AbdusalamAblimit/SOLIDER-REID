# 实验 exp017: PSG + Pose-Conditioned Channel Gate (PCG)

## 动机
- 16 个实验已证明 PSG 58.3% 是 backbone **空间级** injection 的上限
- 所有在空间维度扩展 PSG 的尝试（PAB combo、spatial conv、multi-stage、part pooling/supervision）均失败或中性
- PSG 在空间维度告诉 backbone "**哪些位置**重要"，但没有告诉匹配阶段"**哪些通道**更适合当前姿态下的匹配"
- 假设：不同姿态下，最具判别力的特征通道不同（如正面姿态→面部相关通道重要，侧面→轮廓通道重要）
- PCG 在 **通道维度** 操作，与 PSG 的空间维度完全正交，应不会互相干扰

## 创新点 / 核心想法
- **核心假设**: 在 PSG 的空间级 pose injection 之上，添加通道级 pose-conditioned gating (PCG)，可以进一步提升性能
- **与 exp007 PSG 的区别**: PSG 在 backbone Stage 3 blocks 之间注入（空间维度），PCG 在 GAP 之后、BN 之前注入（通道维度）
- **为什么正交**: PSG 改变 feature map 的空间分布（which positions），PCG 改变全局特征的通道权重（which channels）。两者在不同维度操作

## 技术方案

### 新增模块: PoseChannelGate
- 文件: `model/modules/pose_channel_gate.py`
- 输入:
  - `global_feat`: (B, 768) — GAP 后的全局特征
  - `scene_heatmaps`: (B, 17, hH, hW) — 场景级 pose 热图
- 输出: (B, 768) — 通道加权后的全局特征
- 架构:
  ```
  scene_heatmaps → sigmoid → GAP → pose_desc (B, 17)
  pose_desc → Linear(17, 64) → ReLU → Linear(64, 768) → gate (B, 768)
  global_feat * (1 + gate)  # 残差门控
  ```
- 关键设计:
  1. **Zero-init 最后一层**: 初始 gate=0, 输出 = global_feat × 1 = 不变，保护预训练特征
  2. **Sigmoid on heatmaps**: 与 PSG 一致，将 raw logits 转为概率
  3. **GAP on heatmaps**: 将空间热图压缩为 (B, 17) 的 pose 描述符，每个值表示该关键点在图中的整体存在程度
  4. **轻量 MLP**: 17 → 64 → 768，仅增加 ~50K 参数

### 修改文件
1. `model/modules/pose_channel_gate.py` — **新建**，PCG 模块定义
2. `model/pose_backbone_model.py` — 在 `forward()` 中 GAP 之后、`bottleneck` 之前插入 PCG
3. `config/defaults.py` — 新增 `POSE_CHANNEL_GATE` 和 `POSE_PCG_HIDDEN` 配置
4. `configs/occluded_duke/pose_psg_pcg.yml` — 实验 config

### 数据流
```
Input Image + Pose Heatmaps
  → Swin Backbone Stage 0-2 (unchanged)
  → Stage 3 with PSG (spatial gating, 同 exp007)
  → out_feat: (B, 768, 12, 4)
  → GAP → global_feat: (B, 768)
  → PCG: global_feat * (1 + channel_gate(pose_desc))  # NEW
  → BN (bottleneck) → Classifier
```

### 关键超参数
- `POSE_CHANNEL_GATE: True` — 开关
- `POSE_PCG_HIDDEN: 64` — MLP 隐藏层维度
- 其他所有配置与 exp007 完全一致

### 参数量估算
- Linear(17, 64): 17 × 64 + 64 = 1,152
- Linear(64, 768): 64 × 768 + 768 = 49,920
- 总计: ~51K 参数（PSG 102K 的一半）

## 预期结果
- **如果假设成立**: mAP 在 PSG 58.3% 基础上提升 0.5-1.0%，因为通道 gating 提供了 PSG 未覆盖的正交信息
- **如果失败，最可能原因**:
  1. GAP 后的 pose 描述符 (B, 17) 信息量不足——空间信息被 GAP 压缩后可能丢失关键位置信息
  2. 通道维度的 pose conditioning 在 BN 之前效果被归一化消除
  3. PSG 已经在空间维度做了足够的 pose conditioning，通道维度是冗余的

## 对照组
- Baseline 对照: exp007 PSG-only (mAP 58.3%, R1 67.9%)
- 消融变量: 在 GAP 后添加 PCG 模块，其他完全不变
