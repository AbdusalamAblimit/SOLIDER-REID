# exp125 监控

## 实验信息
- 方法: Sparse Pair-Delta SCRD
- 类型: 训练端单变量改进
- 主配置: `exp123_pair_delta_scrd`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta_top'`
- 输出目录: `log/occluded_duke/exp125_pair_top_scrd_v2`

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

### [2026-03-20 12:06] 首次后台启动未留住，立即转前台确认

- 异常:
  1. 和 `exp123` 首次启动时类似，第一次 `nohup` 后进程没有留住
  2. 输出日志为空，GPU 也没有真正占用
- 处理:
  1. 不把这次后台现象归因为机制失败
  2. 立即转为前台会话启动，直接确认真实配置与训练日志
- 当前判断: 继续
- 原因:
  - 当前更重要的是确认 `delta_top` 路由已正确接入，而不是纠结后台留驻方式

### [2026-03-20 12:06] 启动确认（正式训练）

- 运行位置: 本地 3090
- 配置: `configs/occluded_duke/pose_psg_gcn_pair_top_scrd.yml`
- 输出目录: `log/occluded_duke/exp125_pair_top_scrd`
- 关键确认:
  1. `POSE_CSRD_PAIR_WEIGHT_MODE = delta_top` 已生效
  2. `POSE_CSRD_PAIR_TOP_RATIO = 0.25` 已生效
  3. support-complete teacher 仍正常启用：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  4. 新机制日志已打印：
     - `[CSRD-PW] mode=delta_top, alpha=1.0, top_ratio=0.25`
- 当前判断: 继续
- 原因:
  - 当前已经确认 `exp125` 是相对 `exp123` 的干净单变量；下一关键点是 warmup 前段稳定性，以及 `epoch 21+` 后 `csrd_psr` 是否表明稀疏 pair routing 真在工作

### [2026-03-20 12:11] 检查点 #1 — Epoch 1-5 warmup 稳定

- 当前进度:
  - `Epoch 1 done`
  - `Epoch 2 done`
  - `Epoch 3 done`
  - `Epoch 4 done`
  - `Epoch 5 done`
- 当前局部训练状态:
  - `Epoch[1] Iter[140/227] Loss: 16.498, Acc: 0.001`
  - `Epoch 1 done`: `59.756s/epoch`
  - `Epoch 2 done`: `58.295s/epoch`
  - `Epoch 3 done`: `58.431s/epoch`
  - `Epoch 4 done`: `58.304s/epoch`
  - `Epoch 5 done`: `58.624s/epoch`
  - `Epoch[5] Iter[140/227] Loss: 7.793, Acc: 0.156`
- 当前观察:
  1. `exp125` 的 warmup 前段与 `exp123/124` 完全同型，没有新增不稳定
  2. 训练速度稳定在 `58~60s/epoch`，说明 `delta_top` 稀疏 pair 路由没有引入额外开销
  3. 当前仍未进入 `CSRD` 激活区，因此现在能下的结论仅限于：
     - 新机制没有污染 baseline 的早期收敛
     - 稀疏 pair 选择至少通过了最敏感的启动阶段
- 当前判断: 继续
- 原因:
  - 接下来真正有信息量的节点是 `ep10 / ep20` 与 `epoch 21+` 的 `csrd_psr / csrd_pf`

### [2026-03-20 12:17] 检查点 #2 — Epoch 10

- 结果:
  - `ep10 = 38.3% / 51.4% / 66.8% / 73.3%`
- 对照:
  - `exp123 ep10 = 38.3 / 51.4`
  - `exp122 ep10 = 38.3 / 51.4`
  - `exp030a ep10 = 38.2 / 51.3`
  - `exp119 ep10 = 39.8 / 52.9`
  - `exp120 ep10 = 39.8 / 52.9`
- 当前观察:
  1. `exp125` 到 `ep10` 与 `exp123/122` 完全一致，说明 `delta_top` 在 `epoch <= 20` 阶段没有引入任何额外副作用
  2. 同时它当前也没有回到 `exp119/120` 那条更高的 early 曲线，但这仍是合理现象，因为稀疏 pair routing 还未激活
  3. 因而 `ep10` 只能证明：
     - 稀疏 pair 选择没有伤害 warmup
     - 当前还不能据此判断方法正负
- 当前判断: 继续
- 原因:
  - `exp125` 真正有信息量的节点仍是 `ep20` 和 `epoch 21+`；届时 `csrd_psr` 才能回答稀疏 pair routing 是否真的接管了 `CSRD`

### [2026-03-20 12:28] 检查点 #3 — Epoch 20 与激活崩溃

- 结果:
  - `ep20 = 47.0% / 60.7% / 75.4% / 80.1%`
- 对照:
  - `exp123 ep20 = 47.0 / 60.7`
  - `exp030a ep20 = 46.8 / 60.9`
  - `exp119 ep20 = 47.6 / 61.5`
  - `exp120 ep20 = 47.6 / 61.5`
- 异常:
  1. `epoch 20` 验证仍正常完成，说明 warmup 与 eval 路径都没问题
  2. 但一进入 `epoch 21+` 的 `delta_top` 分支，训练立即在 `loss/make_loss.py` 崩溃：
     - `NameError: name 'math' is not defined`
  3. 触发位置是 `keep_num = max(1, int(math.ceil(...)))`
- 当前观察:
  1. 这不是机制负结果，而是一次实现错误；到崩溃前所有曲线都和 `exp123` 完全对齐
  2. 因而当前还没有得到任何关于 `delta_top` 有效性的真实证据
- 当前判断: 立即修复并重启
- 原因:
  - 错误位置明确、修复成本极低；最合理的处理是补上 `import math`，并用干净输出目录 `exp125_pair_top_scrd_v2` 重新运行正式版本
