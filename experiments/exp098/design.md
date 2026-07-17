# 实验 exp098: PKP (Pose Keypoint Prompting)

## 动机
- exp095-096 证明: GCN 分支改进空间极其有限
- 10+ 辅助损失全部失败
- 多尺度/热图池化全部中性或负面
- **突破必须来自 BACKBONE**
- KPR (ECCV24) 核心机制: 在 patch embedding 层注入 pose heatmap 作为 prompt
  → backbone 从第一层就看到 pose 信息
  → 这是最深层的 pose 注入方式

## 核心创新
**Pose Keypoint Prompting: 在 Swin patch embedding 层注入 heatmap prompt**

当前: PSG 在 Stage 3 注入 → backbone Stages 0-2 完全不知道 pose
提议: PKP 在 patch embed 后注入 → backbone 从第一层就是 pose-aware

实现: 像 KPR 一样，用独立的 PatchEmbed 处理 heatmap，zero-init 加法融合到 image tokens。

关键区别于 KPR:
- KPR 用 ViT (global attention, stride-16)
- 我们用 Swin (window attention, stride-32) + SOLIDER 预训练
- KPR 的 prompt 替代了 pose branch
- 我们的 PKP 与 PSG+PAA+GCN 协同工作（全面 pose integration）

## 技术方案

### 数据流
```
Image (B, 3, 384, 128)     Heatmap (B, 17, 96, 32)
        ↓                           ↓
  PatchEmbed (原始)           PosePromptEmbed (新)
  Conv2d(3→96, 4×4)          Conv2d(17→96, 4×4), zero-init
        ↓                           ↓
  image_tokens (B, 96×32, 96)   pose_tokens (B, 96×32, 96)
        ↓                           ↓
        └──── additive fusion ────┘
                    ↓
        pose-prompted tokens (B, 96×32, 96)
                    ↓
              Stage 0 (96d)
                    ↓
              Stage 1 (192d)
                    ↓
              Stage 2 (384d)
                    ↓
              Stage 3 + PSG + PAA (768d)  [现有模块保持]
                    ↓
              GAP → Global feature + GCN branch
```

### 关键设计
1. **PosePromptEmbed**: Conv2d(17, 96, kernel_size=4, stride=4)
   - 与 Swin patch embed 相同的 kernel/stride
   - 输入: heatmap (17 channels) 而非 RGB (3 channels)
   - Output: (B, 96×32, 96) tokens — 与 image tokens 相同 shape

2. **Zero-init**: 初始化 PosePromptEmbed 的权重为 0
   - 训练开始时: pose_tokens = 0 → image_tokens + 0 = image_tokens
   - 保持 SOLIDER 预训练权重的初始行为
   - 随着训练，逐渐学习 pose prompt 的最优注入

3. **Heatmap 输入**: 需要 full-resolution heatmap (B, 17, 384, 128)
   - 当前 heatmap 存储为 (17, 64, 48) per person
   - 需要 resize 到 input size (384, 128) 再 patch embed
   - 或者直接 resize 到 patch embed 需要的分辨率

4. **与 PSG 的关系**: PKP + PSG 是互补的
   - PKP: early injection (patch level, all stages benefit)
   - PSG: late injection (Stage 3, 精细化调制)
   - 两者不冲突

### 参数估算
- PosePromptEmbed: Conv2d(17, 96, 4, 4) = 17 × 96 × 4 × 4 + 96 = 26K
- 极轻量！

### 修改文件
1. `model/pose_backbone_model.py`:
   - __init__: 创建 PosePromptEmbed (Conv2d)
   - _run_backbone_with_psg: 在 patch embed 后注入 pose tokens

2. `config/defaults.py`: POSE_PKP = False

3. `configs/occluded_duke/pose_psg_gcn_paa_pkp.yml`

## 预期结果
- PKP 让所有 4 个 Stage 都是 pose-aware → 应该比仅 Stage 3 PSG 更强
- 预期 mAP +1~3% over exp066
- 如果成功，这是一个与 KPR 同级别的 backbone-level pose injection

## 对照组
- exp066 (PSG+GCN+PAA): 61.6%/74.2%
- 消融: PKP only (no PSG), PKP+PSG, PKP+PSG+PAA

## 风险
- Swin window attention 可能限制 PKP 效果（vs KPR 的 global attention）
- 热图在 patch embed 后的分辨率 (96×32) 可能太高/太低
- Zero-init 可能收敛太慢
