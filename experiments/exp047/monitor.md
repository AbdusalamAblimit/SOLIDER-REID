# exp047: Common-Support-Guided Triplet (CSGT) — 监控日志

## 实验目标
- **目的**: 把 retrieval-time 的 common-support 信号迁到训练端 triplet mining
- **配置**: `configs/occluded_duke/exp047_csgt_triplet.yml`
- **输出目录**: `log/occluded_duke/exp047_csgt_triplet`

## 启动前检查
- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只新增 `POSE_CSGT` 相关开关
- [x] 默认代码路径保持不变，开关关闭时完全回退 baseline
- [x] 当前先完成代码接入与设计，不抢占正在运行的 `exp046`

---
## [准备完成] 当前状态

### 计划命令
- `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml`

### 当前判断
- **等待 GPU 空档后启动**

### 原因
1. `exp046` 正在重建 `seed2024` checkpoint，按当前优先级应继续保持。
2. `exp047` 的代码和配置可先准备好，等 `exp046` 进入更稳定阶段或结束后立即开跑。
