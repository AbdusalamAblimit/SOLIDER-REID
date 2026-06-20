# Codex Review — exp356 PC-MSC

**Verdict**: approve
**Date**: 2026-06-21
**Review round**: 2 (v1 needs-attention → 修复 → v2 approve)

## Round 1 Findings (needs-attention, 无 Critical/High)
- **Medium** (clip_id_prompt.py): `self.visual.eval()` 只在 __init__ 调用。训练循环每 epoch `model.train()` 会递归把 pcmsc_visual.visual 重置回 train 模式。@no_grad 防梯度, 但若 open_clip visual 有 dropout/patch-dropout → 补全目标随机化, patch-dropout 还可能破 16x16 reshape。
- **Low×3**(非阻断, exp356 不触发): POSE_PCMSC scoped under POSE_CLIP_ID_PROMPT(exp356 都开); in_planes vs featmap C under REDUCE_FEAT_DIM(exp356 False); frozen CLIP visual 进 state_dict → ckpt 变大(预期, 非 bug)。

## 修复
- Medium: CLIPVisualEncoder 加 `train(mode)` override(super().train 后立即 self.visual.eval())+ part_targets() 内 forward 前再 self.visual.eval() 双保险 → model.train() 每 epoch 无法把 visual 留在 train 模式, dropout/patch-dropout 对补全目标禁用。

## Round 2 (approve)
**Verdict: approve. 无 Medium/High 残留。** Medium 修复正确(train override + part_targets eval)。Re-confirm:
- AMP dtype 一致: target fp32/no-grad; mask token+query cast tokens.dtype; cos fp32。
- train/test 对称: PC-MSC 仅训练路径(self.training 守卫); eval 描述子=unmasked global。
- frozen CLIP: requires_grad False + 优化器排除 + @no_grad + 现 eval-stable。
- 单变量 vs exp341: config diff 仅 POSE_PCMSC/_W/_RANDOM_MASK + OUTPUT_DIR。
- 3 Low 均不触发(exp356 config)。

先例(Codex web search): PersonMAE(2311.04496)、MVP(2203.05175)、RILS(2301.06958)、RFCnet、MaskCLIP(2208.12262)。PC-MSC 差异化 = pose 部位选择 mask + 冻结 CLIP dense 区域语义目标 + 遮挡 ReID + 训练端正则的组合, 是 plausible distinct mechanism(非孤立"masked CLIP completion")。

## 结论
codex 审查通过。双审查(Claude PASS + Codex approve)全过, 可训练(L1: 在 lab-4090 启动)。
