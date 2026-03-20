# exp120 监控

## 实验信息
- 方法: Support-Complete Relational Distillation (SCRD)
- 类型: 训练端单变量改进
- 主配置: `exp119`
- 核心变量: `POSE_CSRD_SUPPORT_TEACHER = True`
- 输出目录: `log/occluded_duke/exp120_scrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp119` 只新增 `CSRD teacher` 的 support-complete enhancement
- [x] 默认行为不变，开关关闭可完全回退 `exp119`
- [x] bank 仅用于增强 `CSRD teacher`，不额外添加 pointwise cosine distillation
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 14:35] 实验准备

- 启动原因:
  1. `exp119` 已证明 relational teacher 有效，但 teacher 仍来自单图 `kp_feats`
  2. `exp109` 已证明 support-complete teacher headroom 很大
  3. `exp110-116` 只否定了 prototype-pointwise 蒸馏，不是否定 bank 作为 teacher enhancer
- 当前判断: 待启动
- 原因:
  - `exp120` 是当前最干净地把 `exp109` 和 `exp119` 接起来的单变量实验
