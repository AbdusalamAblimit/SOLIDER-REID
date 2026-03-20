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
