# exp120 监控

## 实验信息
- 方法: Support-Complete Relational Distillation (SCRD)
- 类型: 训练端单变量改进
- 主配置: `exp119`
- 核心变量: `POSE_CSRD_SUPPORT_TEACHER = True`
- 输出目录: `log/occluded_duke/exp120_scrd_clean`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp119` 只新增 `CSRD teacher` 的 support-complete enhancement
- [x] 默认行为不变，开关关闭可完全回退 `exp119`
- [x] bank 仅用于增强 `CSRD teacher`，不额外添加 pointwise cosine distillation
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 14:35] 实验准备

- 启动原因:
  1. `exp119` 已证明 relational teacher 有效，但 teacher 仍来自单图 `kp_feats`
  2. `exp109` 已证明 support-complete teacher headroom 很大
  3. `exp110-116` 只否定了 prototype-pointwise 蒸馏，不是否定 bank 作为 teacher enhancer
- 当前判断: 待启动
- 原因:
  - `exp120` 是当前最干净地把 `exp109` 和 `exp119` 接起来的单变量实验

### [2026-03-20 14:42] 启动即止损

- 异常:
  1. 首次启动时，新 config 没有完整继承 `exp119`，出现了默认值回退
  2. 具体表现为：`IF_LABELSMOOTH=on`、`NO_MARGIN=False`、`PRETRAIN_HW_RATIO=1`、`SEMANTIC_WEIGHT=1.0`
- 处理:
  1. 立即终止该进程，不记录任何训练结果
  2. 修正 config，确保重新锚定到 `exp119`
  3. 切换到新的干净输出目录 `log/occluded_duke/exp120_scrd_clean`
- 当前判断: 继续
- 原因:
  - 这是单变量原则下必须当场修正的问题；及时止损后，实验主线未受污染

### [2026-03-20 14:47] 重新启动确认

- 运行位置: 本地 3090
- 配置: `configs/occluded_duke/pose_psg_gcn_scrd.yml`
- 输出目录: `log/occluded_duke/exp120_scrd_clean`
- 关键确认:
  1. 已重新锚定到 `exp119` 设定：`IF_LABELSMOOTH=off`、`NO_MARGIN=True`、`PRETRAIN_HW_RATIO=2`、`SEMANTIC_WEIGHT=0.2`
  2. 新增变量生效：`POSE_CSRD_SUPPORT_TEACHER=True`
  3. teacher bank 配置生效：`low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1`
- 当前判断: 继续
- 原因:
  - 当前已经恢复到真正的单变量设置，可以开始观察 warmup 是否保持与 `exp119` 一致

### [2026-03-20 14:48] 检查点 #1 — Epoch 1 Iter 60

- 当前局部训练状态:
  - `Epoch[1] Iter[20/227] Loss: 22.167, Acc: 0.001`
  - `Epoch[1] Iter[40/227] Loss: 19.986, Acc: 0.001`
  - `Epoch[1] Iter[60/227] Loss: 18.873, Acc: 0.001`
  - 分项:
    - `id_global: 6.555`
    - `id_part: 6.667`
    - `tri_global: 11.438`
    - `tri_part: 13.087`
- 当前观察:
  1. warmup 前段与 `exp119` 完全同型，没有新增不稳定
  2. 当前还不会出现 `csrd` 或 `csrd_sr` 等分项，符合 `warmup=20` 设计
  3. support-complete teacher enhancement 至少没有破坏最早期收敛
- 当前判断: 继续
- 原因:
  - 下一关键点仍是 `ep10 / ep20`，以及 `epoch 21+` 后是否出现有效的 teacher replacement 统计

### [2026-03-20 14:50] 检查点 #2 — Epoch 1-2

- 当前进度:
  - `Epoch 1 done`
  - `Epoch 2 Iter 80/227`
- 当前局部训练状态:
  - `Epoch 1 done`: `Time per epoch = 59.931s`, `ETA = 1h58m`
  - `Epoch[2] Iter[20/227] Loss: 13.185, Acc: 0.005`
  - `Epoch[2] Iter[40/227] Loss: 12.715, Acc: 0.004`
  - `Epoch[2] Iter[60/227] Loss: 12.364, Acc: 0.003`
  - `Epoch[2] Iter[80/227] Loss: 12.062, Acc: 0.003`
- 当前观察:
  1. `Epoch 1 -> 2` 总 loss 持续正常下降，没有因为 teacher bank 接线引入异常抖动
  2. 速度约 `60s / epoch`，相比 `exp119` warmup 略慢，但仍在可接受范围
  3. 当前仍是 warmup 前段，核心目标仍是确认它不破坏基线轨迹
- 当前判断: 继续
- 原因:
  - 只有通过 `ep10 / ep20` 的 warmup 稳定性检查，这轮 `exp120` 才值得继续等 `epoch 21+` 的 teacher-enhanced `CSRD`
