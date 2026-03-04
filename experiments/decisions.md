# 决策记录

### [2026-03-04 12:55] 决策 #1

**上下文**: 需要确定主攻创新方向
**选项**:
  A. Soft BPA + VGFC (OA-PAMS) — 在 PAMS 基础上增量改进，连续遮挡信息利用
  B. 层级化部件发现 (PSHPD) — 多粒度部件分割
  C. Part-Level NFC — 后处理方法
**选择**: A (OA-PAMS)
**理由**:
1. 建立在已有的 PAMS 实现上，技术风险最低
2. 故事完整：soft supervision + feature calibration + continuous distance，三个层面形成一个统一框架
3. 消融实验自然：每个组件可独立消融
4. 实现简单：核心改动在 loss 和 distance 计算，不需要大改架构
**执行结果**: 待跑实验

### [2026-03-04 12:55] 决策 #2

**上下文**: PAMS v8 只跑到 epoch 30，需要确认最终性能
**选项**:
  A. 直接在 v8/v9 基础上继续改进
  B. 先跑完 v9 120 epoch，确认 baseline 后再改进
**选择**: B
**理由**: 没有 baseline 数字就无法评估改进的有效性。必须先确认 PAMS 完整训练的性能。
**执行结果**: 待跑
