# exp028: PDS + Part LR Boost (3x) 监控日志

**配置**: `configs/occluded_duke/pose_pds_stopgrad_partlr.yml`
**输出**: `./log/occluded_duke/exp028_pds_partlr`
**对照**: exp023 PDS+StopGrad (mAP 59.5%, R1 69.5%)
**关键变量**: POSE_PART_LR_FACTOR=3.0

**注意**: 日志中显示的 LR 是 Global 分支的 LR。Part 分支实际 LR = 显示值 * 3。Part bias LR = 显示值 * 6。

---
