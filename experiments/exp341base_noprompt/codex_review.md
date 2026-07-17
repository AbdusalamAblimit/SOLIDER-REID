# Codex Review — exp341base(= exp341 prompt off)

**Verdict**: approve（round 2，修复后）
**Date**: 2026-06-20

## Round 1（needs-attention）
- **High**: `GLOBAL_LOSS_SCALE: 0.5`（从 exp336 抄来）对「global 即描述子」不干净——砍半 CE+triplet 并相对放大 clip_id_loss。
- Medium: clip_id_loss 仅在 no-part 默认返回路径返回（exp341base(= exp341 prompt off) 无 part 分支 OK；exp342 加 pose 时需并入每个分支返回）。
- Medium: cls_ctx/proj 跟 backbone 同 LR（CoOp 可能要更高 LR）。
- Checks (a)-(d) PASS: prompt 构造正确、CLIP 冻结只训 cls_ctx+proj、supcon 正负确、test 端不用 prompt 无泄漏。

## 修复
- `GLOBAL_LOSS_SCALE` 0.5→**1.0**（exp341base(= exp341 prompt off)_clip_id_prompt.yml）。
- 新建**精确无-prompt 对照** `exp341base(= exp341 prompt off)base_noprompt.yml`（= exp341 但 `POSE_CLIP_ID_PROMPT: False`，同 1.0）。单变量 = 仅 prompt on/off。

## Round 2 结论
**Verdict: approve. Remaining findings: no blocking or High findings.** codex 审查通过。
exp341base(= exp341 prompt off) vs exp341base 单变量；GLOBAL_LOSS_SCALE 1.0 合理；CLIP-ReID prompt 集成（CoOp + i2t/t2i + model/processor wiring）正确。
clip_id-loss-only-on-no-part-path 与 prompt-LR 为已知非阻断项（exp341base(= exp341 prompt off) 无 pose 分支；训练时盯 clip_id loss 单调下降）。

## exp341base
= exp341 同代码仅关 prompt。codex 审查通过（复用，无新代码）。Verdict: approve。
