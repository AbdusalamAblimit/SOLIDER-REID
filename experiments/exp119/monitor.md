# exp119 监控

## 实验信息
- 方法: Common-Support Relational Distillation (CSRD)
- 类型: 训练端单变量改进
- 主配置: `exp030a`
- 核心变量: `POSE_CSRD = True`
- 输出目录: `log/occluded_duke/exp119_csrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只新增 `POSE_CSRD` 相关开关
- [x] 默认行为不变，开关关闭可完全回退 baseline
- [x] `CSRD` 使用 batch 内 `kp_feats / kp_weights` 作为 detached teacher，不新增 backbone 模块
- [x] `CSRD` 仅在 `epoch > 20` 后激活，避免早期 teacher 过噪

## 启动记录

### [2026-03-20 11:40] 实验准备
- 启动原因:
  1. `exp047` 失败的是 overlap mining，不是 `pair comparability` 问题本身
  2. `exp051` 只改了 part triplet 的距离定义，没有把 pairwise teacher 蒸馏到 global embedding
  3. `exp110-116` 说明 prototype bank 会丢失 pair-specific 细节
- 当前判断: 待启动
- 原因:
  - `CSRD` 是当前最直接的新机制验证：不用 prototype，而是直接蒸馏 common-support 关系
