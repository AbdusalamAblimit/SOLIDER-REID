# exp124 监控

## 实验信息
- 方法: Stronger Pair-Delta SCRD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp123_pair_delta_scrd`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_ALPHA = 4.0`
- 输出目录: `log/occluded_duke/exp124_pair_delta_scrd_a4`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp123` 只修改 `POSE_CSRD_PAIR_WEIGHT_ALPHA`
- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退 `exp123`
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 10:36] 实验准备

- 启动原因:
  1. 远程 `exp121` 已收敛，GPU 空出
  2. `exp123` 到 `ep60` 已给出“pair focus 方向对”的证据，但当前 `pair_focus` 长期只有 `1.06~1.08`
  3. 当前最有信息量的下一跳不是换题，而是验证 **focus 强度是否就是瓶颈**
- 当前判断: 待启动
- 原因:
  - `exp124` 是相对 `exp123` 的最小下一跳：只改 pair focus 的放大强度
