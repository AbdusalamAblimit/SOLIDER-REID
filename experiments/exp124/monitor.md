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

### [2026-03-20 10:40] 首次远程启动失败并立即修正

- 异常:
  1. 第一次远程启动沿用了旧的解释器路径 `/root/miniconda3/envs/solider-reid/bin/python`
  2. 远程实际报错：
     - `nohup: failed to run command '/root/miniconda3/envs/solider-reid/bin/python': No such file or directory`
- 处理:
  1. 立即确认远程可用解释器路径
  2. 验证 `/usr/local/bin/python` 已正确安装 `torch / torchvision / yacs / cv2`
  3. 用 `python -u train.py ...` 重新后台启动
- 当前判断: 继续
- 原因:
  - 这是远程环境路径差异，不是实验机制问题；修正后即可继续按单变量方案执行

### [2026-03-20 10:40] 启动确认（远程 5060 Ti）

- 运行位置: 恒源云 5060 Ti
- 启动方式: 后台 `nohup`
- 输出目录: `log/occluded_duke/exp124_pair_delta_scrd_a4`
- nohup 日志: `log/occluded_duke/exp124_pair_delta_scrd_a4/remote_nohup.log`
- 关键确认:
  1. 远程仓库已同步到 `7fbcdd0`
  2. 配置已生效：`POSE_CSRD_PAIR_WEIGHT_MODE = delta`
  3. 放大系数已生效：
     - `[CSRD-PW] mode=delta, alpha=4.0`
  4. support-complete teacher 仍正常启用：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  5. GPU 已占用约 `6692 MiB`，利用率约 `69%`
- 当前判断: 继续
- 原因:
  - 现在已经形成了“本地 `exp123 alpha=1.0` + 远程 `exp124 alpha=4.0`”的干净并行对照

### [2026-03-20 11:59] 检查点 #1 — Epoch 20

- 结果:
  - `ep20 = 47.7% / 62.0% / 75.9% / 80.4%`
- 对照:
  - `exp123 ep20 = 47.0 / 60.7`
  - `exp119 ep20 = 47.6 / 61.5`
  - `exp120 ep20 = 47.6 / 61.5`
- `CSRD` 统计（epoch 21 前后）:
  - `csrd_pd = 0.002`
  - `csrd_pf = 1.24~1.28`
- 当前观察:
  1. `alpha=4.0` 已显著放大了 pair focus，`csrd_pf` 不再停留在 `1.06~1.08`
  2. 首个验证点相对 `exp123` 已出现明确早期增益，表现为 `+0.7 / +1.3`
  3. 但这还只是刚进入 `CSRD` 激活区后的早期信号，不能据此直接判成收敛正向
- 当前判断: 继续
- 原因:
  - 当前最重要的是看这份更强 focus 能否在 `ep30/40` 继续保留，而不是只抬高早期曲线

### [2026-03-20 11:59] 检查点 #2 — Epoch 30

- 结果:
  - `ep30 = 53.2% / 66.4% / 80.0% / 84.3%`
- 对照:
  - `exp123 ep30 = 52.3 / 66.5`
  - `exp119 ep30 = 53.4 / 66.7`
  - `exp120 ep30 = 53.2 / 66.5`
- `CSRD` 统计（epoch 30 前后）:
  - `csrd_pd = 0.001~0.002`
  - `csrd_pf = 1.25~1.29`
- 当前观察:
  1. 相对 `exp123`，更强的 pair focus 已带来 `mAP +0.9`，但 `R1` 没有同步提升
  2. 相对 `exp119/120`，当前仍只是近乎持平，说明单纯增大连续加权强度还没有把这条线直接拉成明显突破
  3. 这更像“强度方向有效，但收益形状仍不够稳”
- 当前判断: 继续
- 原因:
  - 下一关键点是 `ep40/50`；如果此时仍只是 `mAP` 小涨而 `R1` 不跟，说明问题可能不只在强度，还在 pair 路由的稀疏性

### [2026-03-20 11:59] 检查点 #3 — Epoch 40

- 结果:
  - `ep40 = 55.6% / 68.6% / 81.8% / 85.7%`
- 对照:
  - `exp123 ep40 = 55.5 / 68.9`
  - `exp119 ep40 = 55.9 / 68.7`
  - `exp120 ep40 = 55.5 / 67.8`
- `CSRD` 统计（epoch 40 前后）:
  - `csrd_pd = 0.002`
  - `csrd_pf = 1.27~1.29`
- 当前观察:
  1. 到 `ep40`，`alpha=4.0` 相对 `exp123` 表现为 `mAP +0.1 / R1 -0.3`
  2. 相对 `exp119` 也仍是近乎持平，尚未形成更强的中期领先
  3. 这说明“只把平滑 delta focus 变强”是有效干预，但当前还不像足够的主突破口
  4. 当前更合理的收束是：
     - teacher-change pairs 仍然值得聚焦
     - 但连续加权可能仍然太稀释，后续要考虑更稀疏的 pair 选择
- 当前判断: 远程继续跑，本地主线可准备更结构化的 pair routing
- 原因:
  - `exp124` 还没收敛，不宜过早停；但到 `ep40` 已经足够说明本地主线不该继续只做 `alpha` 扫点
