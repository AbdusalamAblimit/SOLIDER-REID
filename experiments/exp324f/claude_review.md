# Claude Broad Review — exp324f（eval-only fusion）

**Reviewer**: Claude (Opus 4.8) broad review 子代理
**Date**: 2026-06-16
**Round**: 1
**Verdict**: 审查通过 (PASS) — eval-only，无训练 hook 阻断

exp324f 与 exp325 在同一轮 broad review 中合并审查（两脚本共享 exp324b/exp324_dino helper）。完整审查结论见 `experiments/exp325/claude_review.md`。本文件为 exp324f 专项摘录。

## exp324f 专项结论

- **两阶段拆分合理**（lab-3090-d 无单一 env 同时具 mmengine + transformers）：Stage1 `exp324f_swin_distmat.py`（solider-reid env）dump npz；Stage2 `exp324f_fuse.py`（系统 python3）load + DINO + 融合。
- **文件名对齐正确**：Swin（make_dataloader, shuffle=False, val=query+gallery, OccludedDuke 内 sorted）与 DINO（sorted listdir）同 key 空间；join 后 pid 全等断言 + camid 偏移恒定断言（修正后），无法静默错位。
- **归一化/融合 soundness**：z-score/min-max 全矩阵单调仿射不改 per-query 排序；w=0 精确复现纯 Swin（实测 75.16）。
- **head 载入 / heavy mask** 正确，与 exp324b 口径一致。

## 运行时修正（已修，不改 eval 数值）
exp324f camid 断言因 SOLIDER（0-indexed）vs exp324（1-indexed）相机约定差异首跑报错；修正为"camid 偏移恒定"断言（仍捕获真实错位）。复跑通过，offset=1，w=0=75.16。

**审查通过**（eval-only）。
