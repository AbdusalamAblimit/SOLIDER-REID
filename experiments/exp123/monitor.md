# exp123 监控

## 实验信息
- 方法: Pair-Delta Focused SCRD
- 类型: 训练端单变量改进
- 主配置: `exp120`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_MODE = 'delta'`
- 输出目录: `log/occluded_duke/exp123_pair_delta_scrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp120` 只新增 pair-level `CSRD` focusing
- [x] support-complete teacher 构造、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退 `exp120`
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 17:20] 实验准备

- 启动原因:
  1. `exp120` 已证明 teacher 增强是真的，但监督没有自动兑现
  2. `exp122` 已明确否定 sample-level `replace_ratio` 作为 supervision routing
  3. 当前最合理的新假设是：真正需要被强调的不是样本，而是 **teacher-change pairs**
- 当前判断: 待启动
- 原因:
  - `exp123` 是相对 `exp120` 的最小下一跳：只改 `CSRD` 如何聚焦 pair-level relation

### [2026-03-20 17:21] 首次后台启动未留住，立即转前台确认

- 异常:
  1. 第一次 `nohup` 启动后，进程没有留住
  2. 这与 `exp122` 的首次后台现象相同，不先归因为机制失败
- 处理:
  1. 不保留该次后台启动结果
  2. 立即以前台会话启动，直接确认真实配置与训练日志
- 当前判断: 继续
- 原因:
  - 当前更重要的是确认 `pair-delta` 开关已正确接入，而不是纠结后台留驻方式

### [2026-03-20 17:24] 启动确认（正式训练）

- 运行位置: 本地 3090
- 配置: `configs/occluded_duke/pose_psg_gcn_pair_delta_scrd.yml`
- 输出目录: `log/occluded_duke/exp123_pair_delta_scrd`
- 关键确认:
  1. `POSE_CSRD_PAIR_WEIGHT_MODE = delta` 已生效
  2. `POSE_CSRD_PAIR_WEIGHT_ALPHA = 1.0` 已生效
  3. support-complete teacher 仍正常启用：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  4. 新机制日志已打印：
     - `[CSRD-PW] mode=delta, alpha=1.0`
- 当前判断: 继续
- 原因:
  - 当前已经确认 `exp123` 是相对 `exp120` 的干净单变量；下一关键点是 warmup 前段稳定性，以及 `epoch 21+` 后 `csrd_pd / csrd_pf` 是否给出有效信号

### [2026-03-20 17:25] 检查点 #1 — Epoch 1 Iter 140

- 当前局部训练状态:
  - `Epoch[1] Iter[20/227] Loss: 22.167, Acc: 0.001`
  - `Epoch[1] Iter[60/227] Loss: 18.873, Acc: 0.001`
  - `Epoch[1] Iter[140/227] Loss: 16.498, Acc: 0.001`
  - 分项:
    - `id_global: 6.554`
    - `id_part: 6.641`
    - `tri_global: 9.024`
    - `tri_part: 10.776`
- 当前观察:
  1. warmup 前段与 `exp120/122` 完全同型，没有新增不稳定
  2. 当前仍未出现 `csrd / csrd_pd / csrd_pf`，符合 `warmup=20` 的设计
  3. 说明 `pair-delta` 机制至少没有污染最敏感的早期收敛阶段
- 当前判断: 继续
- 原因:
  - 下一关键点仍是 `ep10 / ep20`，以及 `epoch 21+` 后 `csrd_pd / csrd_pf` 是否表明 pair-level focusing 真的介入了有效关系

### [2026-03-20 17:29] 检查点 #2 — Epoch 2-5 warmup 持续健康

- 当前进度:
  - `Epoch 2 done`
  - `Epoch 3 done`
  - `Epoch 4 done`
  - 当前已进入 `Epoch 5`
- 当前局部训练状态:
  - `Epoch 2 done`: `58.382s/epoch`
  - `Epoch 3 done`: `58.419s/epoch`
  - `Epoch 4 done`: `58.312s/epoch`
  - `Epoch[5] Iter[140/227] Loss: 7.793, Acc: 0.156`
- 当前观察:
  1. `exp123` 到 `ep5` 仍与 `exp120/122` 的 warmup 轨迹一致，没有新增不稳定
  2. 训练速度稳定在 `58~59s/epoch`，说明 pair-level focus 逻辑不会给 warmup 前段带来额外开销
  3. 当前仍未进入 `CSRD` 激活区，因此接下来真正有信息量的节点仍是 `ep10 / ep20` 与 `epoch 21+`
- 当前判断: 继续
- 原因:
  - 早期稳定性已经通过；下一步只需盯住首个验证点和 `csrd_pd / csrd_pf` 是否按设计出现

### [2026-03-20 17:30] 检查点 #3 — Epoch 10

- 结果:
  - `ep10 = 38.3% / 51.4% / 66.8% / 73.3%`
- 对照:
  - `exp122 ep10 = 38.3 / 51.4`
  - `exp030a ep10 = 38.2 / 51.3`
  - `exp119 ep10 = 39.8 / 52.9`
  - `exp120 ep10 = 39.8 / 52.9`
- 当前观察:
  1. `exp123` 的 warmup 首个验证点与 `exp122` 完全一致，说明 pair-level focus 在 `epoch <= 20` 阶段没有引入额外副作用
  2. 但它当前也没有回到 `exp119/120` 当时更高的 early 曲线，因此还不能把这条线判断为“早期更强”
  3. 由于 `pair-delta` 机制尚未激活，当前最合理的解释仍是：`ep10` 只能证明训练健康，不能证明方法正负
- 当前判断: 继续
- 原因:
  - 这条线真正有信息量的节点仍是 `ep20` 和 `epoch 21+`；届时 `csrd_pd / csrd_pf` 才会告诉我们 pair-level focusing 是否真在工作

### [2026-03-20 17:40] 检查点 #4 — Epoch 20 与 `CSRD` 激活起点

- 结果:
  - `ep20 = 47.0% / 60.7% / 75.4% / 80.1%`
- 对照:
  - `exp122 ep20 = 47.0 / 60.7`
  - `exp030a ep20 = 46.8 / 60.9`
  - `exp120 ep20 = 47.6 / 61.5`
- `CSRD` 激活后统计（epoch 21 前段）:
  - `csrd = 0.013`
  - `csrd_tgap = 0.329~0.337`
  - `csrd_sgap = 0.244~0.252`
  - `csrd_pd = 0.001~0.002`
  - `csrd_pf = 1.064~1.068`
  - `csrd_sr = 0.142~0.145`
  - `csrd_sn = 155~157`
- 当前观察:
  1. `exp123` 到 `ep20` 仍贴着 `exp122 / exp030a` 这条当前代码口径的基线，没有形成早期优势
  2. 但 `pair-delta` 机制已经真实生效：`csrd_pd > 0`、`csrd_pf > 1`
  3. 同时 `csrd_ar / csrd_aw = 1.0` 也符合设计，因为这一版不再做 sample-level anchor weighting
  4. 当前最值得警惕的新信息不是“没生效”，而是：
     - `pair_delta` 量级目前很小
     - 对应的 focus 只有约 `1.06`
     - 说明第一版 pair-level focusing 可能偏弱
- 当前判断: 继续，但重点关注 `ep30`
- 原因:
  - 这条线已经通过了“实现正确”检查；下一步要看的是这么弱的 pair focus 能不能仍然转成验证收益，如果不能，再考虑更强的 pair focus 形式

### [2026-03-20 17:53] 检查点 #5 — Epoch 30

- 结果:
  - `ep30 = 52.3% / 66.5% / 79.7% / 84.1%`
- 对照:
  - `exp030a ep30 = 52.2 / 66.0`
  - `exp122 ep30 = 52.5 / 65.5`
  - `exp120 ep30 = 53.2 / 66.5`
  - `exp119 ep30 = 53.4 / 66.7`
- `CSRD` 统计（epoch 21-30）:
  - `csrd = 0.012~0.015`
  - `csrd_tgap = 0.33 -> 0.44`
  - `csrd_sgap = 0.24 -> 0.38`
  - `csrd_pd = 0.001~0.002`
  - `csrd_pf = 1.06~1.08`
  - `csrd_sr = 0.14~0.15`
  - `csrd_sn = 154~158`
- 当前观察:
  1. `exp123` 到 `ep30` 仍然只是弱正向，基本形状是：
     - 相对 `exp030a`: `+0.1 / +0.5`
     - 相对 `exp119`: `-1.1 / -0.2`
     - 相对 `exp120`: `-0.9 / +0.0`
  2. 这说明 pair-level focusing 并没有把 `SCRD` 明显拉起来，至少第一版没有
  3. 但机制上它也不是无效接线：
     - `csrd_pd > 0`
     - `csrd_pf > 1`
     - 说明 focus 确实介入了 pairwise distillation
  4. 当前真正暴露的问题是：
     - `pair_delta` 量级一直很小
     - 对应 focus 只有 `1.06~1.08`
     - 因而这版更像“方向对，但力度不够”
- 当前判断: 继续到 `ep40` 再做停表决策
- 原因:
  - `ep30` 还不足以直接判死，但已经足够说明如果后续不转强，下一步不该再换问题，而应改更强的 pair focus 形式

### [2026-03-20 10:36] 检查点 #6 — Epoch 40

- 结果:
  - `ep40 = 55.5% / 68.9% / 81.3% / 85.1%`
- 对照:
  - `exp119 ep40 = 55.9 / 68.7`
  - `exp120 ep40 = 55.5 / 67.8`
  - `exp122 ep40 = 55.4 / 68.2`
- `CSRD` 统计（epoch 40 前后）:
  - `csrd = 0.013`
  - `csrd_tgap = 0.49~0.50`
  - `csrd_sgap = 0.45~0.46`
  - `csrd_pd = 0.002`
  - `csrd_pf = 1.06~1.07`
- 当前观察:
  1. `exp123` 到 `ep40` 已经不再像 `ep30` 那样明显落后：相对 `exp120` 形成 `+0.0 / +1.1`
  2. 相对 `exp119` 则表现为 `mAP -0.4 / R1 +0.2`，更像“开始追上”而不是已经被判负
  3. 但 `pair_delta` 量级依然极小，focus 仍只有 `1.06~1.07`，说明当前版本的 pair-level 强化仍然偏弱
- 当前判断: 继续
- 原因:
  - `ep40` 已经足够说明这条线没死，而且出现了比 `exp120` 更好的 `R1` 形态；现在更合理的是继续看它会不会在中后期兑现，而不是过早停表

### [2026-03-20 10:36] 检查点 #7 — Epoch 50

- 结果:
  - `ep50 = 56.2% / 69.9% / 82.1% / 85.9%`
- 对照:
  - `exp119 ep50 = 56.8 / 69.3`
  - `exp120 ep50 = 56.2 / 69.3`
- `CSRD` 统计（epoch 50 前后）:
  - `csrd = 0.012~0.013`
  - `csrd_tgap = 0.51~0.52`
  - `csrd_sgap = 0.48~0.49`
  - `csrd_pd = 0.002`
  - `csrd_pf = 1.06~1.08`
- 当前观察:
  1. `exp123` 到 `ep50` 已经在 `R1` 上稳定超过 `exp119/120`，表现为相对 `exp119` 的 `mAP -0.6 / R1 +0.6`
  2. 相对 `exp120` 则是 `mAP 持平 / R1 +0.6`
  3. 这说明 pair-level focusing 可能是 **延迟兑现**，不是早期就能直接拉起 `mAP`
  4. 与此同时，`csrd_pd / csrd_pf` 依然很小，进一步暴露出“方向对但放大不够”的问题
- 当前判断: 继续
- 原因:
  - 这条曲线已经从“可疑”转成“值得继续看”，下一关键点是 `ep60` 是否会把这份 delayed gain 扩展到 `mAP`

### [2026-03-20 10:36] 检查点 #8 — Epoch 60

- 结果:
  - `ep60 = 57.8% / 70.9% / 83.1% / 86.9%`
- 对照:
  - `exp119 ep60 = 57.7 / 70.5`
  - `exp120 ep60 = 57.5 / 69.7`
- `CSRD` 统计（epoch 60 前后）:
  - `csrd = 0.012~0.013`
  - `csrd_tgap = 0.53~0.54`
  - `csrd_sgap = 0.50~0.51`
  - `csrd_pd = 0.002~0.003`
  - `csrd_pf = 1.06~1.08`
  - `csrd_sr = 0.146~0.150`
  - `csrd_sn = 158~163`
- 当前观察:
  1. `exp123` 到 `ep60` 首次同时超过 `exp119` 与 `exp120` 同阶段，表现为：
     - 相对 `exp119`: `+0.1 / +0.4`
     - 相对 `exp120`: `+0.3 / +1.2`
  2. 这说明 pair-level teacher-change focusing 方向本身是成立的，至少已经不再只是“概念接线”
  3. 但当前版本的核心短板也更清楚了：`pair_delta` 依然只有 `0.002~0.003`，`pair_focus` 仍只有 `1.06~1.08`
  4. 因而现阶段最合理的解释不是“机制不对”，而是：
     **当前 alpha=1.0 的 focus 放大力度过弱，导致正向信号兑现得太慢、太浅。**
- 当前判断: 本地继续跑；并行准备更强的 pair focus 版本
- 原因:
  - `exp123` 已经给出“方向对”的证据，所以不该停；但远程资源也不该空着，下一步最自然的单变量就是放大 `pair-delta` 聚焦强度
