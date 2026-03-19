# exp109 Oracle Support Bank 监控

## 实验信息
- 方法: Oracle Support Bank 上界诊断
- 类型: retrieval-time oracle 分析，不计入正式方法结果
- 主基线: `exp030a cvk_hybrid`
- 核心变量: GT same-ID per-keypoint prototype 是否提供明显 headroom

## 启动记录

### [2026-03-19 15:30] 实验启动
- 来自上一阶段的直接教训：
  - `DACHM/DACCM` 说明 confuser penalty 不是对的路
  - `SGCFR` 说明 cross-image support recovery 确有价值
- 当前执行内容：
  1. 新增 `scripts/eval_oracle_support_bank.py`
  2. 在 `exp030a` 上提取 `global_feat / kp_feats / kp_weights`
  3. 构造 GT same-ID oracle prototype，比较 `base / oracle_feat_only / oracle_feat_weight`
  4. 若 headroom 明显，再进入训练版 support-complete distillation 设计

### [2026-03-19 16:08] 实验完成
- 结果文件: `log/occluded_duke/exp109_oracle_support_bank_exp030a/summary.json`
- oracle 恢复统计:
  - `samples_recovered = 3385`
  - `keypoints_recovered = 10194`
  - `avg_support_count = 82.33`
- 整体结果:
  - `base_cvk_hybrid = 61.88% mAP / 73.26% R1`
  - `oracle_feat_only_cvk = 66.15% / 77.87%`
  - `oracle_feat_weight_cvk = 70.40% / 81.36%`
- 关键子集:
  - `clean multi`: `65.06 / 76.26` → `68.04 / 79.71` → `71.33 / 82.73`
  - `duplicate-suspect multi`: `62.31 / 76.96` → `65.21 / 78.73` → `68.34 / 81.27`
  - `target_vis<=8` (26 queries): `29.42 / 26.92` → `78.26 / 84.62` → `91.71 / 100.00`
  - `target_vis<=5` (7 queries): `16.85 / 14.29` → `78.43 / 85.71` → `86.28 / 100.00`
- 当前判断: 强阳性上界，继续
- 原因:
  1. 即使只替换低可见 keypoint feature、保留原始 weight，也有 `+4.27 mAP / +4.62 R1`
  2. 说明“support-complete latent representation”不是空想，而是存在巨大 headroom
  3. 下一步应做训练版 prototype distillation，而不是继续 retrieval-time penalty
