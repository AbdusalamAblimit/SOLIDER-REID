# Codex Review — exp357 pose-shuffle kill-switch

**Verdict**: approve
**Date**: 2026-06-21
**Review round**: 2 (v1 needs-attention → 修复 → v2 none blocking)

## Round 1 (needs-attention, 无 Critical/High)
- Medium-1: randperm 留固定点(~1/64 图保留自己 pose), 非 derangement。
- Medium-2(判读 caveat): NO-DROP 侧被裁剪对齐混淆——别人 pose 仍带粗糙 canonical 头/躯干/腿先验; 不掉点只能说"精确图特定 pose 在对齐裁剪下非必需", 需补 cross-PART(通道)shuffle 二次确认。

## 修复
- Medium-1: 改 derangement(re-roll 至 (perm==arange).any() 为 False, 上限 8 次, fallback torch.roll(ar,1) 保证无固定点)→ 每张图都用别人 pose。
- Medium-2: 设计 note 已记录判读 caveat + cross-part follow-up + 最佳矩阵。

## Round 2 (none blocking = approve)
**Findings: none blocking. Medium-1 is fixed.** Verified(commit 21de414):
- derangement 逻辑正确((perm==ar).any() 触发 reroll, 非仅全恒等); 循环 tries<8 终止 + roll fallback; fallback 对 Bp>1 是有效 derangement(无自映射)。
- device/dtype OK(int64 on scene_heatmaps.device); Bp<=1 跳过(唯一合理); scene/target 同 perm 配对; 训练端守卫(eval 用真 pose); batch dim 正确; config 单变量(仅 POSE_SHUFFLE + OUTPUT_DIR)。
- Medium-2 仅判读 caveat, 设计 note 已 acknowledge。

## 结论
codex 审查通过。双审查(Claude PASS + Codex approve)全过, 可训练。判读: 掉点=pose 因果(故事稳); 不掉=需 cross-part 二次确认。
