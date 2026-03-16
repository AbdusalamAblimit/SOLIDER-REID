# 实验 exp072: Part-Structured PAA (PS-PAA)

## 动机
- PAA (exp066) 使用 generic Conv2d(17→32→768)，所有 17 个关键点通道被平等对待
- 用户建议：让 encoder 知道身体结构——不同身体部位的 keypoint 应该在独立子空间中处理
- exp069 (增大 bottleneck) 和 exp071 (feature-dependent LoRA) 都不如 PAA
- 但 PAA 的 encoder 结构本身还未被探索——grouped convolution 是一个正交方向

## 创新点 / 核心想法
- **PAA**: Conv2d(17→32→768) — 所有通道混合处理
- **PS-PAA**: GroupedConv2d(20→40, groups=5) → Conv2d(40→768) — 按身体部位分组
  - 17 通道 padding 到 20（每组 4 通道）:
    - head(5) → pad to 4? NO — 分组需要均匀
    - 实际分组: head=5, shoulders=2, arms=4, hips=2, legs=4 → 总共 17
    - 选择: 将 17 pad 到 20 (每组 4 channels, 5 groups)
    - 或者: 使用 5 个独立小 Conv2d，每个处理一个 body part group
- **采用更简洁的方案**: 5 个独立的 1x1 Conv2d，每个只看对应 body part 的通道
  - 输出 concat 后共 hidden_per_part × 5 = bottleneck_dim 个通道
  - 再用第二个 Conv2d 映射到 768

## 技术方案
- **修改文件**: `model/modules/pose_additive_adapter.py` — 新增 `PosePartStructuredAdapter`
  - 5 个独立 encoder (每个 body part)
  - Part groups: head=[0,1,2,3,4], shoulders=[5,6], arms=[7,8,9,10], hips=[11,12], legs=[13,14,15,16]
  - 每个 encoder: Conv2d(n_kp, hidden_per_part=8, k=1) + ReLU
  - Concat 所有 part outputs: 8×5 = 40 channels
  - 最终 Conv2d(40, 768, k=1) — zero-init

- **关键超参数**:
  - hidden_per_part = 8 → total hidden = 40 → ~63K params (略多于 PAA b32 的 52K)
  - 通过 config 开关: `POSE_PAA_PART_STRUCTURED: True`

## 预期结果
- 如果 body-part-aware encoding 有效: mAP +0.3~1.0% over PAA
- 如果无效: 说明 17 通道的 generic mixing 已经足够，部位结构信息冗余

## 对照组
- exp066 PAA (generic, b=32): 61.6%/74.2%
- 本实验只改 PAA encoder 结构，其他不变
