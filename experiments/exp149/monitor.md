# exp149 SCFA 监控

## 实验信息
- 方法: SCFA（Symmetry-Conditioned Feature Aggregation）
- 类型: skeleton branch 表示重构
- 主基线: `exp030a-eq`
- 当前状态: 仅完成设计，尚未实现/审查/启动

## 启动前检查清单
- [ ] 对称 pairs 定义清楚
- [ ] 默认 config 不受影响
- [ ] skeleton branch 聚合逻辑与旧逻辑单变量隔离
- [ ] `scfa_*` 行为日志接好
- [ ] Claude 广范围审查通过

## 当前判断
- 这条线不是 completion，不是 scorer，也不是 attention bias
- 若成立，可把 story 推到“pose-defined bilateral redundancy”
