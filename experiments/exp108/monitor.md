# exp108 DACCM 监控

## 实验信息
- 方法: DACCM（Duplicate-Aware Counterfactual Common-Support Matching）
- 类型: retrieval-time 诊断实验
- 主基线: `exp030a cvk_hybrid`
- 核心变量: per-keypoint common-support 层面的 duplicate-aware confuser penalty

## 启动记录

### [2026-03-19 12:55] 实验启动
- 来自 `exp107` 的直接教训：
  - pooled person embedding 粒度太粗，负面
  - 若继续 ambiguity 主线，必须回到 `per-keypoint / common-support`
- 当前执行内容：
  1. 新增 `scripts/eval_daccm.py`
  2. 在 `exp030a` 上跑 `base_cvk_hybrid / raw_daccm_penalty / daccm_penalty`
  3. 若仍负面，则停止 retrieval-time ambiguity 线，不再继续调参

### [2026-03-19 14:52] 实验完成
- 结果文件: `log/occluded_duke/exp108_daccm_exp030a/summary.json`
- 整体结果:
  - `base_cvk_hybrid = 61.88% mAP / 73.26% R1`
  - `raw_daccm_penalty = 61.35% / 72.85%`
  - `daccm_penalty = 61.39% / 72.94%`
- 关键子集:
  - `multi`: `64.07 / 76.51` → `63.16 / 75.87`
  - `clean multi`: `65.06 / 76.26` → `64.12 / 75.40`
  - `duplicate-suspect multi`: `62.31 / 76.96` → `61.47 / 76.71`
  - `n=2`: `65.76 / 78.35` → `64.99 / 77.82`
- 当前判断: 终止当前 retrieval-time ambiguity 线
- 原因:
  1. per-keypoint / common-support 粒度下的 penalty 仍然整体负面
  2. dedup 版仅比 raw 版略好，但依然低于 `base_cvk_hybrid`
  3. 说明 test-time confuser penalty 本身不构成稳定增益，不能继续作为主线调参
