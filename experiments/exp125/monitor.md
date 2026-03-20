# exp125 监控

## 实验信息
- 方法: Sparse Pair-Delta SCRD
- 类型: 训练端单变量改进
- 主配置: `exp123_pair_delta_scrd`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta_top'`
- 输出目录: `log/occluded_duke/exp125_pair_top_scrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp123` 只修改 pair-level 路由机制
- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退 `exp123`
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 12:05] 实验准备

- 启动原因:
  1. `exp123` 正式评估表明 `alpha=1.0` 的平滑 delta focusing 只得到与 `exp119` 近乎等价的最终结果
  2. 远程 `exp124` 到 `ep40` 已说明“只增大 alpha”仍然不足以形成清晰中期领先
  3. 当前最合理的新假设是：teacher-change pairs 本来就稀疏，连续加权仍然过于平滑
- 当前判断: 待启动
- 原因:
  - `exp125` 是相对 `exp123` 的下一跳：不换 teacher、不换 bank，只把 pair focus 从连续加权改成稀疏 pair 选择
