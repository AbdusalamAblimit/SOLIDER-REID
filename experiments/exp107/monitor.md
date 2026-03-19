# exp107 DACHM 监控

## 实验信息
- 方法: DACHM（Duplicate-Aware Counterfactual Hypothesis Matching）
- 类型: retrieval-time 诊断实验
- 主基线: `exp030a equal_concat`
- 支持性复核: `exp066 equal_concat`
- 核心变量: duplicate-aware hypothesis pruning + counterfactual confuser margin rerank

## 启动记录

### [2026-03-19 09:xx] 实验启动
- 已完成本地前置诊断：
  - `exp066` 相对 `exp030a` 的增益主要来自多人 query
  - `cvk_hybrid` 相对 `equal_concat` 的 mAP 增益也主要偏向多人 query
  - query 多人图里存在明显重复检测伪多人，且在 `n>=4` 中占比很高
- 当前执行内容：
  1. 新增 `scripts/eval_dachm.py`
  2. 在 `exp030a` 上跑 base / raw counterfactual / DACHM
  3. 若有正信号，再在 `exp066` 上复核

## 待补充
- 具体命令
- 参数 sweep
- 总体结果
- 子集结果
- 是否进入训练端版本


## 正式结果

### [2026-03-19 12:40] exp030a 首轮完成
- 命令:
  - `/root/miniconda3/envs/solider-reid/bin/python scripts/eval_dachm.py --config configs/occluded_duke/pose_psg_gcn.yml --weight log/occluded_duke/exp030a_psg_gcn/transformer_120.pth --output_dir log/occluded_duke/exp107_dachm_exp030a_v2`
- 核心对照:
  - `base_equal_concat` = `61.14% mAP / 73.71% R1`
  - `raw_counterfactual_signed` = `60.32% / 72.76%`
  - `dachm_signed` = `60.27% / 72.81%`
  - `raw_counterfactual_penalty` = `60.70% / 73.17%`
  - `dachm_penalty` = `60.72% / 73.17%`
- 子集观察:
  - `clean multi`: base `63.99 / 77.27` → `dachm_penalty` `63.24 / 75.83`
  - `duplicate-suspect multi`: base `61.36 / 76.71` → `dachm_penalty` `60.64 / 76.46`
  - `n=2`: base `64.64 / 79.05` → `dachm_penalty` `63.99 / 77.82`
- 结论:
  1. 有符号 support-gap 重排明确负面，说明“奖励安全 pair + 惩罚危险 pair”的粗糙公式不成立。
  2. penalty-only 比 signed 稍好，但仍整体落后基线。
  3. duplicate-aware pruning 没有把该方向救回来，说明问题不只是重复检测噪声。
  4. 当前这版 **coarse pooled hypothesis rerank** 不进入 `exp066` 复核，也不进入训练端版本。

## 当前判断
- `exp107` 的价值不是结果本身，而是排除了一条看起来合理但实际无效的路线：
  **单图 pooled skeleton hypothesis + counterfactual margin，不足以形成有效的 retrieval-time ambiguity reasoning。**
- 若继续研究 target/distractor ambiguity，必须把推理粒度拉回 `per-keypoint / common-support`，而不是继续在 pooled person embedding 上做文章。
