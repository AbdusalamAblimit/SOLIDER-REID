# 实验 exp344 (Option B): pose-conditioned prompt

## 动机（用户路线，姿态融进 CLIP 机制 — 方式 B）
A 是「pose 引导对齐的图像特征」;B 是「pose 调制 prompt 本身」。每张图的 pose 调制可学习 ID context → ID 文本原型变 **pose-aware**,i2t/t2i 对齐时原型已含姿态信息。

## 核心假设
**per-image pose（GAP scene_heatmaps → (B,17) 每关键点激活）经 pose_encoder 产生 cls_ctx delta,加到 per-ID context → prototype = CLIP_text([prefix, cls_ctx+pose_delta, suffix]) pose-aware。对齐 pose-aware 原型比对齐固定 per-ID 原型塑造 backbone 更好 → global 涨。**

## 技术方案
- `CLIPIDPromptLearner` 加 `pose_cond`：`pose_encoder = Linear(17,768)→ReLU→Linear(768, 4×768)`,**末层零初始化**（起点 = exp341 的 0-delta,然后学）。
- forward(label, pose)：`cls_ctx = cls_ctx[label] + pose_encoder(pose).view(B,4,768)`。
- 模型 forward：`pose_vec = scene_heatmaps.mean(dim=(2,3))` (B,17) 传入;无 pose 时 pose=None → 不调制（退回 exp341）。
- = exp341 config + `POSE_CLIP_ID_POSE_PROMPT: True`。测试描述子 = global。

## 预期结果
exp344 global **> exp341 global(59.8)**。零初始化保证最差 == exp341（不会更糟）。

## 对照组
- **exp341(prompt 无 pose, 59.8)vs exp344(pose-conditioned prompt)**。单变量 = 仅 POSE_CLIP_ID_POSE_PROMPT。
- baseline 57.6。

## 审查重点
pose_encoder 零初始化正确(起点=exp341);forward(label,pose) pose 传递与 shape(B,17→B,4,768);dtype(pose_delta 转 CLIP dtype);pose_encoder 进优化器;scene_heatmaps None → pose=None 不崩;test 端 prompt train-only;单变量 vs exp341。

## 修复（codex B Medium-1）
pose_encoder 建在 clip_id_proj 前会消耗 RNG → 下游初始化偏离 exp341。已加 `torch.get/set_rng_state` 保存/恢复,下游模块初始化对齐 exp341。A/C 同样处理。
注：单 seed 比较下,init 差异≈seed 方差,真实增益需 >~0.3 才有意义（与是否修 RNG 无关，但修后更干净）。

## ★ 结果 (e120, test.py global) — NEGATIVE
**B (pose-conditioned prompt) global = 57.6%** = baseline, -2.2 vs exp341。
**结论**: pose 调制 prompt → prototype 变 pose-aware → 把 global 拉去编码姿态而非纯 ID,稀释了 exp341 的纯 ID 对齐增益。zero-init 没救回来(pose_encoder 学到的调制有害)。**Option B 失败。**
