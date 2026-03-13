# exp045: 基于重建 `seed42` checkpoint 的 CVK 复核 — 监控日志

## 实验目标
- **目的**: 在 `exp044` 重建出的 `seed42` checkpoint 上，直接复核 `equal_concat` 与 `cvk_hybrid`
- **checkpoint**: `log/occluded_duke/exp044_exp030a_seed42_rebuild/transformer_120.pth`
- **子实验**:
  - `045a`: `equal_concat`
  - `045b`: `cvk_hybrid`

## 启动前检查
- [x] 与 `exp040` 相比，不改训练或模型结构，只更换 checkpoint
- [x] `TEST.WEIGHT` 已指向 `exp044` 产出的 `transformer_120.pth`
- [x] `045a/045b` 使用独立 `OUTPUT_DIR`
- [x] 当前结论会按“重建 seed42 复核”记录，不与历史原始 seed42 资产混写

---
## [启动] 运行计划

### 执行命令
- `045a`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp045_seed42_cvk_verify.yml MODEL.POSE_TEST_FEAT equal_concat OUTPUT_DIR ./log/occluded_duke/exp045a_seed42_eq`
- `045b`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp045_seed42_cvk_verify.yml OUTPUT_DIR ./log/occluded_duke/exp045b_seed42_cvk_hybrid`

### 当前判断
- **继续**

### 原因
1. `exp044` 已完成，当前最有价值的下一步就是把第二个 seed 的测试端证据补齐。
2. 这一步仍严格围绕 `exp030a` 主基线，不是偏题调参。
