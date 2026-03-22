# exp148 PCVT 监控

## 实验信息
- 方法: PCVT（Pose-Complementary View Training）
- 类型: 训练范式大改动
- 主基线: `exp030a-eq`
- 当前状态: 仅完成设计，尚未实现/审查/启动

## 启动前检查清单
- [ ] 数据管线支持生成 `full / view_a / view_b`
- [ ] complement partition 只依赖当前图 pose
- [ ] 三视图主损失与 `exp030a` 对齐
- [ ] `pcvt_*` 行为日志接好
- [ ] Claude 广范围审查通过

## 当前判断
- 这是对 `support incomplete` 的训练范式级重写，不是小调参
- 若实现失败，也应能明确排除“单图伪多 support”这条主线
