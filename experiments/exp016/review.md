# exp016 审查记录

## 第一轮审查
- **结论**: FAIL（2 个 CRITICAL + 3 个 MODERATE 问题）
- **CRITICAL 1**: 身体部件分组不一致（代码 3 组 vs 设计 5 组）→ 已修复为 5 组
- **CRITICAL 2**: 实现方法不一致（设计描述 heatmap mask，代码用 keypoint bbox）→ 已更新设计文档
- **MODERATE 1**: "1-2 组"未实现（代码只选 1 组）→ 已更新设计文档
- **MODERATE 2**: 热图通道清零范围过大（整个通道 vs 空间区域）→ 已改为空间区域清零
- **MODERATE 3**: 擦除分布偏向大区域 → 5 组后更均衡

## 第二轮审查
- **结论**: PASS — 所有 5 个问题已正确修复，未引入新问题
- 单变量原则验证通过（仅添加 POSE_GUIDED_ERASING: True）
- 设计文档与代码完全一致
- 边界情况处理正确（关键点不足 fallback 到 RE）
