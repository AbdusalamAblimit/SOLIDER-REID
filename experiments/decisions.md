# 决策记录

### [2026-03-04 11:20] 决策 #1

**上下文**: 开始 Phase 1 论文学习，需要确定工作流程和预训练权重下载
**选项**:
  A. 串行学习每个仓库，逐个完成笔记
  B. 并行启动 7 个 agent，同时学习 12 个仓库
**选择**: B
**理由**: 并行化可以大幅缩短 Phase 1 耗时，12 个仓库互相独立
**执行结果**: 7 个 agent 并行工作，已完成 5/12 篇笔记

### [2026-03-04 11:20] 决策 #2

**上下文**: 预训练权重下载 — SOLIDER Swin-Tiny 和 ViTPose
**选项**:
  A. 自行下载所有权重
  B. SOLIDER 自行下载，ViTPose 由用户提供
**选择**: B
**理由**: SOLIDER Swin-Tiny 权重（774MB）已从 Google Drive 下载完成。ViTPose 权重 `best_coco_AP_epoch_210.pth` 是用户自定义训练的 VisPredictHead 模型，用户明确表示由他提供。
**执行结果**: SOLIDER 权重已下载到 `pretrained/swin_tiny.pth`，等待用户提供 ViTPose 权重
