# 实验 exp071: Pose-Conditioned LoRA (PCL)

## 动机
- PAA (exp066) 是当前最佳训练端创新: +0.87%/+1.63%
- PAA 的局限: `adapter = f(heatmap)` — 不依赖当前特征内容，对所有位置施加相同的 pose-derived 信号
- LoRA (Low-Rank Adaptation) 的核心思想: 用低秩分解做参数高效的特征适应
- 如果让 pose adapter 同时依赖 **特征内容** 和 **热图**，应该能产出更精确的适应

## 创新点 / 核心想法
- **PAA**: `x = x + adapter(heatmap)` — pose content is feature-independent
- **PCL**: `x = x + lora(x, heatmap)` — pose adaptation depends on both features AND heatmap
- PCL 使用低秩分解实现高效的 feature-conditioned pose adaptation:
  1. Down-project features: `x_down = W_down(x)` → (B, N, r)
  2. Encode heatmap: `hm_feat = conv(sigmoid(hm))` → (B, N, r)
  3. Element-wise modulation: `z = x_down * hm_feat`
  4. Up-project: `lora_out = W_up(z)` → (B, N, 768), zero-init
- 这让 adapter 能根据每个位置的特征内容来决定如何利用 pose 信息

## 技术方案
- **新文件**: `model/modules/pose_cond_lora.py`
  - PoseCondLoRA 模块: low-rank pose-conditioned adaptation
  - 输入: x (B, N, C), heatmap (B, 17, H, W)
  - 输出: x + lora_out (B, N, C)

- **修改文件**:
  1. `config/defaults.py`: 新增 `POSE_COND_LORA` (bool), `POSE_COND_LORA_RANK` (int)
  2. `model/pose_backbone_model.py`: 新增 PCL 模块创建和 forward 调用

- **关键超参数**:
  - rank r = 16 (默认), 参数量 ≈ (17×16 + 16 + 768×16 + 768×16) × 2 blocks ≈ 50K
  - Zero-init W_up for identity start

## 预期结果
- 如果 feature-dependent adaptation 优于 feature-independent addition:
  - mAP +0.3~1.0% over exp066 PAA
- 如果相当或略差: 说明 PAA 的简洁性已经足够，feature 依赖没有额外价值

## 对照组
- exp066 PAA (feature-independent): 61.6%/74.2%
- 本实验相对于 exp066 只改了一个变量: adapter 从 feature-independent → feature-dependent
