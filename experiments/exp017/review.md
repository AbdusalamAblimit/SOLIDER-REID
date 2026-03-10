# exp017 审查记录

## 第一轮审查
- **结论**: PASS — 所有 10 项检查通过，无问题
- [1] 单变量原则: PASS — 仅添加 POSE_CHANNEL_GATE + POSE_PCG_HIDDEN
- [2] 设计-代码一致性: PASS — sigmoid → GAP → MLP(17→64→768) + 残差门控
- [3] Zero-init: PASS — 最后一层零初始化，初始输出 = identity
- [4] 数据流: PASS — scene_heatmaps 在 PCG 插入点可用
- [5] Config 一致性: PASS — 与 exp007 完全一致，仅添加 PCG 选项
- [6] 输出目录: PASS — `./log/occluded_duke/exp017_psg_pcg` 唯一
- [7] 参数量: PASS — 51,072 ≈ 51K，与设计文档一致
- [8] 梯度流: PASS — PCG 和 PSG 在不同计算图节点，不干扰
- [9] 测试行为: PASS — PCG 在推理时正确应用
- [10] 边界情况: PASS — pose_dict=None 时优雅降级
