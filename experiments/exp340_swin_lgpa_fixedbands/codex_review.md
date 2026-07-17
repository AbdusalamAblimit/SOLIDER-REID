# Codex Review — exp340

**Date**: 2026-06-19
**Review round**: 1（needs-attention）→ 2（修复后重跑）

## Findings（第一轮 codex --search exec）
- **Critical**: `.git_token` 在历史 commit（be92c5d/45b2e3e）含 `ghp_` token。
- **Medium**: `model/pose_backbone_model.py:635`(train) 与 `:833`(eval) 的 LGPA guard 仍 gate 在 `scene_heatmaps is not None`，FIXED_BANDS 在无 pose_dict 时会静默跳过 LGPA 回落 global。
- **Low**: exp340 yml `:37` 注释仍写 scene-merged 热图（fixed-bands 下被 canonical 覆盖）。
- **Low**: design.md 混用 59.0（trained/global）与 58.20（frozen）两套 baseline 口径。

### Checks Passed（第一轮）
Shape (`_canonical_heatmap` → (B,17,96,32) 匹配 CLIPPartHead bias/assign/visibility)、KL NaN（clamp+isfinite 守卫，低风险）、train/test 对称（两处 `_lgpa_heatmap`）、单变量 vs exp336、device/dtype/contiguous —— 全过。

### Novelty（第一轮）
固定空间分块（PCB/RPP 横条带）在 ReID 是老技术；未找到完全相同的"固定 COCO canonical 热图 + CLIP attention-bias"先例；defensible framing = 诊断/消融，除非 exp340 超 global 且 random-query 对照证明 CLIP 文本起作用。

## 修复（提交于 43d1e67 之后）
- **Critical（token）**：已 `git reset --soft 1cace4d` + 重新 commit，token 从 master 历史移除（`git log -- .git_token` 空）；且该 token **从未 push**（origin 落后本地 936 commit，无 be92c5d）→ GitHub 无泄露。
- **Medium（guard）**：两处 guard 改为 `(scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False))` → FIXED_BANDS 无 pose 也能跑。
- **Low**：exp340 yml 注释 + design.md baseline 口径均已标注修正。

## 结论（第二轮重跑）
**Verdict: approve。Remaining findings: none。** codex 审查通过。
verified against git show HEAD: guards 已允许 fixed-band 无 pose 运行(635/833)、_canonical_heatmap (B,17,96,32) 经 _lgpa_heatmap 路由、train/test 对称、CLIPPartHead 三路消费正确、exp340 vs exp336 运行时单变量(仅 POSE_LGPA_FIXED_BANDS:True + OUTPUT_DIR)。
