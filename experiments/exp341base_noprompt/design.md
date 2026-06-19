# 实验 exp341base: 精确无-prompt 对照

## 动机
exp341（CLIP-ReID 可学习 ID prompt）的**精确单变量对照**：完全相同的 SOLIDER 配置（GLOBAL_LOSS_SCALE 1.0、无 LGPA/PSG），仅 `POSE_CLIP_ID_PROMPT: False`（关掉 prompt 分支）。
判据：**exp341 global > exp341base global = CLIP-ReID 机制真涨**。

## 技术方案
= exp341_clip_id_prompt.yml 但 `POSE_CLIP_ID_PROMPT: False`。无新代码（仅关一个已审查的 flag，prompt 分支不构建）。单变量 = 仅 prompt on/off。

## 审查说明
代码与 exp341 完全相同（已 Claude+Codex 双审通过，见 experiments/exp341_clip_id_prompt/）。本对照仅关闭 `POSE_CLIP_ID_PROMPT`，不引入任何新代码路径，风险低于 exp341 本身。
