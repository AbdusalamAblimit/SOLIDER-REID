# exp126 监控

## 实验信息
- 方法: Exact Top-K Pair SCRD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp125_pair_top_scrd_v2`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_MODE = delta_top_exact`
- 输出目录: `log/occluded_duke/exp126_pair_top_exact_scrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp125` 只修改 pair 选择实现
- [x] `POSE_CSRD_PAIR_TOP_RATIO` 保持 `0.25`
- [x] `POSE_CSRD_PAIR_WEIGHT_ALPHA` 保持 `1.0`
- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 13:50] 实验准备

- 启动原因:
  1. `exp125` 到 `ep60` 已明确显示：结构化 pair focus 有效，但仍未形成清晰突破
  2. `exp125` 最关键的机制问题是 `csrd_psr` 长期高达 `0.90` 左右
  3. 当前最有信息量的下一跳不是再扫 `alpha`，而是验证 **tie 扩散是否就是“伪稀疏” 的根因**
- 当前判断: 待启动
- 原因:
  - `exp126` 是相对 `exp125` 的最小下一跳：只把阈值式 `delta_top` 改成 exact top-k mask

### [2026-03-20 13:58] 远程排队启动确认

- 远程状态:
  1. 5060 Ti 当前仍被 `exp124` 占用
  2. `exp124` 最新已跑到 `ep90 = 59.8 / 72.6`
  3. 该实验已进入后期有效区间，不适合为了抢卡而提前终止
- 处理:
  1. 已将 `exp126` 代码推送到 `origin/exp/pose_heatmap`
  2. 已在远程写入 `/tmp/run_exp126_after_exp124.sh`
  3. 该脚本会在检测到 `exp124` 主训练进程结束后自动：
     - `git pull origin exp/pose_heatmap`
     - 创建 `log/occluded_duke/exp126_pair_top_exact_scrd`
     - 后台启动 `train.py --config_file configs/occluded_duke/pose_psg_gcn_pair_top_exact_scrd.yml`
- 当前判断: 已排队，等待自动启动
- 原因:
  - 这样可以保留 `exp124` 的 late-gain 证据，同时不耽误 `exp126` 接续推进
