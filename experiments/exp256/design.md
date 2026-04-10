# exp256: Pose Prompt Injection + GCN512 + 2-stage PSG (Small)

## 动机
KPR (ECCV 2024) 通过 pose prompt 注入获得 +1.8% mAP (73.3→75.1)。
我们仿制 KPR 的 prompt 机制：argmax part ID → learnable embedding → additive injection。
在 exp255 最强配置 (GCN512 + 2-stage PSG, 73.2/83.3) 上叠加 Pose Prompt。

## 核心假设
Pose Prompt 在 patch embedding 层注入离散 part identity 信号，与 PSG (stage 内乘法门控) 互补。

## 技术方案
- 新增: MODEL.POSE_PROMPT=True
- Pose Prompt: heatmap → clamp(min=0) → bg channel → argmax → Embedding(18, embed_dim) → scale * embed + patch tokens
- trunc_normal_(std=0.02) init, learnable sigmoid scale (init=-2.0 → 0.12)
- AMP-safe (.to(x.dtype)), .detach() before argmax
- 三轮 agent 审查通过 (含 critical sigmoid bug 修复)
- 其余与 exp255 相同: Small + GCN512 + 2-stage PSG + LGPA-D + OA-SD + PLBOA

## 代码修改
- config/defaults.py: 新增 POSE_PROMPT, POSE_PROMPT_NUM_PARTS, POSE_PROMPT_DROP
- model/pose_backbone_model.py: __init__ 创建 Embedding + scale, forward 注入

## 对照组
- exp255 (Small GCN512 + 2stage, 无 Prompt): 73.2/83.3, MaxSim 73.5/83.8
- exp249 (Small GCN256 + 1stage): 71.9/81.8, MaxSim 73.3/83.2

## 预期结果
- 成功: 74+ mAP (Prompt +0.8~1.0 over exp255)
- 中性: ≈ exp255 (Prompt 无额外收益)
- 失败: < exp255 (Prompt 干扰 PSG)

## 变体
- exp256: GCN512 + 2-stage PSG + Prompt (远程)
- exp256b: GCN256 + 1-stage PSG + Prompt (本地, 消融)
