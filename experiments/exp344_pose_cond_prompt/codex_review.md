# Codex Review — exp344 (Option B)

**Verdict**: approve（round 2，RNG fix 后）
**Date**: 2026-06-20

## Round 1 (needs-attention)
- Medium-1: pose_encoder 建在 clip_id_proj 前消耗 RNG → 下游 init 偏离 exp341。
- Medium-2: per-image 原型不破坏 SupCon（labels.eq 定正负，数学合法，可能有益）。
- Low: config POSE_TEST_FEAT equal_concat→global 文本非单变量（运行无影响，clip-only 退回 global）。

## 修复
- clip_id_prompt.py pose_encoder 构造包 `torch.get/set_rng_state`，下游 init 对齐 exp341；A/C 同样处理。

## Round 2 结论
**Verdict: approve.** RNG save/restore 正确，下游模块 init 与 exp341 一致;零初始化 pose_delta step0==exp341 prompt;shape/dtype/优化器/None fallback/test train-only/per-image 原型 SupCon 合法 全部 OK。codex 审查通过。
