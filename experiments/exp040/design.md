# 实验 exp040: 基于 exp030a 原始 checkpoint 的 CVK 检索复核

## 动机
- `exp039` 在 `exp035a` checkpoint 上给出正信号：`cvk_hybrid` 相比 `equal_concat` 出现 `+0.8% mAP`。
- 但 `exp035a` 属于 bundled sanity check，虽然整体接近 `exp030a`，仍不是最干净的主基线来源。
- 根据当前 `AGENTS.md`，所有新实验应以 `exp030a` 作为唯一实验基线，因此需要回到原始 `exp030a` checkpoint 做直接复核。

## 核心假设
- 如果 `exp039` 的正信号不是 bundled checkpoint 的偶然现象，那么在 `exp030a` 原始 checkpoint 上，`cvk_hybrid` 相对同 checkpoint 的 `equal_concat` 仍应保持接近或略优的 mAP。
- 如果 `cvk_hybrid` 在原始 checkpoint 上转负，则说明 `exp039` 的增益主要来自 `exp035a` 中额外 bundled 改动，不足以支撑后续主线。

## 技术方案

### checkpoint
- 固定使用：`log/occluded_duke/exp030a_psg_gcn/transformer_120.pth`

### 子实验
- `040a`: `equal_concat`
  - 目的：在当前代码版本下，对 `exp030a` checkpoint 生成同口径直接对照日志
- `040b`: `cvk_hybrid`
  - 目的：验证 `exp039` 的共同可见关键点补充是否能在原始主基线上复现

### 评测设定
- Backbone: `Swin-Tiny`
- batch size 不变
- 不改训练参数，只改测试模式
- 使用独立 `OUTPUT_DIR`

## 对照组
- 直接对照：`040a equal_concat`
- 待验证项：`040b cvk_hybrid`
- 参考背景：
  - `exp039b`（`exp035a` checkpoint）= `61.9% mAP / 73.2% R1`
  - `exp030a` 原始训练日志在默认 `concat_scaled` 下的 epoch120 = `60.5% mAP / 73.7% R1`

## 预期结果
- 理想结果：`040b` mAP >= `040a`
  - 说明共同可见关键点补充能在主基线 checkpoint 上复现
- 中性结果：`040b` mAP 与 `040a` 基本持平
  - 说明 `cvk_hybrid` 至少不是 bundled checkpoint 偶然产物，但正增益仍需更多验证
- 负结果：`040b` 明显低于 `040a`
  - 说明 `exp039` 的正结果不稳，当前 retrieval-time line 暂不足以上升为主叙事

## 风险与失败解释
1. `exp030a` 训练时默认监控模式不是 `equal_concat`，因此需要先补跑 `040a` 形成当前代码下的直接对照。
2. `cvk_hybrid` 仍属于测试时诊断机制，不能把轻微正增益直接写成训练端创新。
3. 若 `040b` 只提升 mAP 不提升 R1，依旧说明它更像整体排序修正项，而不是 top-1 决策主导项。
