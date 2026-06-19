# Codex Review — exp343 (Option A)

**Verdict**: approve
**Date**: 2026-06-20

## 结论
codex 审查通过。PoseGuidedPool shape 正确(featmap B,C,H,W → tokens → pooled B,C);pose 热图 amax 作 additive bias、softmax over spatial、einsum 池化正确;query+k_proj 可训且经 i2t/t2i 梯度更新、进优化器;scene_heatmaps None 时 fallback global 不崩;clip_id_loss 计算/返回正确;test 端描述子 = GAP global(pose_guided_pool 仅 train,无泄漏),exp343 vs exp341 同评 global 公平;单变量 vs exp341(仅 POSE_CLIP_ID_POSE_GUIDED)。Verdict: approve。
