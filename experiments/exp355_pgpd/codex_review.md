# Codex Review — exp355 PGPD

**Verdict**: approve
**Date**: 2026-06-21
**Review round**: 2 (v1 needs-attention → 修复 → v2 approve)

## Round 1 Findings (needs-attention)
- **Medium** (model/pose_backbone_model.py): PGPD teacher-completeness 用 scene_heatmaps(多人 max-merge)。多人遮挡图里干扰者可虚高完整度, 即使目标被遮也让场景"更完整"→ teacher 选择被非目标 pose 驱动。应用 target_heatmaps。
- **Low**: `uniq_protos[inv]=txt_proto` 在 pose_cond off 时正确(同 ID 原型相同), 但若日后 POSE_CLIP_ID_POSE_PROMPT=True 与 PGPD 同开则是 scatter 陷阱。

## 修复
- Medium: `_pgpd_loss` 加 `target_heatmaps` 参数, `comp_hm = target_heatmaps if not None else scene_heatmaps` 算完整度; forward 调用点传 target_heatmaps(_prepare_pose 产出, person-0=目标)。
- Low: __init__ 加 `assert not POSE_CLIP_ID_POSE_PROMPT` when use_pgpd。

## Round 2 (approve)
**Verdict: approve. Findings: none.** 两修复验证通过:
- target_heatmaps 确为 person-0 目标热图(dataset 显式 reorder target 到 index 0, collate 保序, _prepare_pose 取 heatmaps[:,0]); fallback scene 安全(PGPD 仅 scene_heatmaps not None 时调用, target 正常总产出)。
- pose_cond 互斥 assert 确认。
**Re-confirm**: PGPD 激活在 NOPARAM_POOL 外、PROMPT 内(exp355 prompt+PGPD 都开); POSE_PGPD False 复现 exp341; _pgpd_loss math(student行/teacher列轴、严格更完整、真ID硬负屏蔽、teacher detached、KD 方向、NaN/全零权重处理)全对; train/test 对称(PGPD 仅 self.training, eval 不碰); git diff --check 无空白错误。

先例(Codex web search): CLIP-ReID(2211.13977)、PCL-CLIP(2310.17218)、PGFL-KD(2108.00139, train-only pose-KD 近似先例, 论文须区分)、PromptSRC(2307.06948)。无"pose-completeness 选同ID teacher 蒸馏 CLIP ID-prompt 原型硬负暗分布"的精确先例。

## 结论
codex 审查通过。双审查(Claude PASS + Codex approve)全过, 可训练。
