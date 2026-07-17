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

### [2026-03-20 22:53] 自动启动确认（远程 5060 Ti）

- 运行位置: 恒源云 5060 Ti
- 启动方式: `exp124` 结束后由排队脚本自动启动
- 输出目录: `log/occluded_duke/exp126_pair_top_exact_scrd`
- nohup 日志: `log/occluded_duke/exp126_pair_top_exact_scrd/remote_nohup.log`
- 关键确认:
  1. 远程当前主进程已切换为：
     - `python -u train.py --config_file configs/occluded_duke/pose_psg_gcn_pair_top_exact_scrd.yml`
  2. `delta_top_exact` 已成功接线
  3. 远程 GPU 已重新占用约 `6.7GB`
- 当前判断: 继续
- 原因:
  - 现在已经进入 `exp125 -> exp126` 的干净因果验证阶段

### [2026-03-20 23:28] 检查点 #1 — Epoch 20

- 结果:
  - `ep20 = 47.7% / 62.0% / 75.6% / 80.4%`
- 对照:
  - `exp124 ep20 = 47.7 / 62.0`
  - `exp125 ep20 = 47.0 / 60.7`
- `CSRD` 统计（epoch 21+）:
  - `csrd_pd = 0.001~0.002`
  - `csrd_pf = 1.16~1.18`
  - `csrd_psr = 0.292`
  - `csrd_sr = 0.144~0.148`
- 当前观察:
  1. `ep20` 仍处于 warmup 刚结束点，因此和 `exp124` 基本重合，只能说明启动健康
  2. 但机制上已经出现了最关键的新证据：
     - `csrd_psr` 直接降到 `0.292`
     - 说明 `delta_top_exact` 确实实现了我们想要的“真稀疏 pair 选择”
  3. 与 `exp125` 的 `0.90+` 相比，这不是小修小补，而是机制层面的根本变化
- 当前判断: 继续盯 `ep30/40`
- 原因:
  - 现在真正的问题不再是“有没有稀疏”，而是“真稀疏 routing 会提升还是伤害 late-stage 收益”
