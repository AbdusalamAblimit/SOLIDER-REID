# exp119 监控

## 实验信息
- 方法: Common-Support Relational Distillation (CSRD)
- 类型: 训练端单变量改进
- 主配置: `exp030a`
- 核心变量: `POSE_CSRD = True`
- 输出目录: `log/occluded_duke/exp119_csrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只新增 `POSE_CSRD` 相关开关
- [x] 默认行为不变，开关关闭可完全回退 baseline
- [x] `CSRD` 使用 batch 内 `kp_feats / kp_weights` 作为 detached teacher，不新增 backbone 模块
- [x] `CSRD` 仅在 `epoch > 20` 后激活，避免早期 teacher 过噪

## 启动记录

### [2026-03-20 11:20] 实验准备
- 启动原因:
  1. `exp047` 失败的是 overlap mining，不是 `pair comparability` 问题本身
  2. `exp051` 只改了 part triplet 的距离定义，没有把 pairwise teacher 蒸馏到 global embedding
  3. `exp110-116` 说明 prototype bank 会丢失 pair-specific 细节
- 当前判断: 待启动
- 原因:
  - `CSRD` 是当前最直接的新机制验证：不用 prototype，而是直接蒸馏 common-support 关系

### [2026-03-20 11:26] 启动确认
- 运行位置: 本地 3090
- 配置: `configs/occluded_duke/pose_psg_gcn_csrd.yml`
- 输出目录: `log/occluded_duke/exp119_csrd`
- 关键确认:
  1. 配置已生效：`POSE_CSRD=True, weight=0.5, warmup=20, tau=0.10`
  2. 模型与数据正常加载，已进入训练循环
  3. `CSRD` 设计为 `epoch > 20` 后才激活，因此 warmup 前日志里不会出现 `csrd` 分项
- 当前判断: 继续
- 原因:
  - 当前首要目标是确认 warmup 阶段与 `exp030a` 一样稳定，再看 `epoch 21+` 后 `csrd` 的数值形态

### [2026-03-20 11:27] 检查点 #1 — Epoch 1 Iter 60

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
  1. 训练正常启动，没有 `NaN / OOM / 爆 loss`
  2. warmup 前段曲线与 `exp030a` 的正常启动形态一致，没有出现额外不稳定
  3. 当前没有 `csrd` 分项是预期行为，因为 `warmup=20`
- 当前判断: 继续
- 原因:
  - 需要先看 `ep10/20` 是否保持基线轨迹，再判断 `epoch 21+` 的 `csrd` 是否会引入偏移

### [2026-03-20 11:29] 检查点 #2 — Epoch 1-3

- 当前进度:
  - `Epoch 1 done`
  - `Epoch 2 done`
  - `Epoch 3 Iter 60/227`
- 当前局部训练状态:
  - `Epoch 1 done`: `Time per epoch = 56.041s`, `ETA = 1h51m`
  - `Epoch 2 done`: `Time per epoch = 54.373s`, `ETA = 1h46m`
  - `Epoch[3] Iter[60/227] Loss: 9.427, Acc: 0.019`
  - 分项:
    - `id_global: 6.537`
    - `id_part: 6.263`
    - `tri_global: 2.066`
    - `tri_part: 3.988`
- 当前观察:
  1. `Epoch 1 -> 3` 总 loss 从 `18.873` 持续下降到 `9.427`，warmup 收敛稳定
  2. `tri_global / tri_part` 都在正常下降，没有出现新机制导致的异常抖动
  3. 当前仍未出现 `csrd` 分项，符合 `warmup=20` 的设计
  4. 单 epoch 速度约 `54~56s`，与 `exp030a` 同量级，没有额外训练开销问题
- 当前判断: 继续
- 原因:
  - 早期稳定性已确认，下一关键点是 `ep10` 首次验证

### [2026-03-20 11:31] 检查点 #3 — Epoch 4-6

- 当前进度:
  - `Epoch 4 done`
  - `Epoch 5 done`
  - `Epoch 6 Iter 60/227`
- 当前局部训练状态:
  - `Epoch 4 done`: `Time per epoch = 54.595s`, `ETA = 1h45m`
  - `Epoch 5 done`: `Time per epoch = 54.698s`, `ETA = 1h44m`
  - `Epoch[6] Iter[60/227] Loss: 7.472, Acc: 0.125`
  - 分项:
    - `id_global: 6.386`
    - `id_part: 5.437`
    - `tri_global: 0.835`
    - `tri_part: 2.286`
- 当前观察:
  1. `Epoch 4 -> 6` 继续稳定下降，没有出现早期停滞
  2. `Acc` 已从 `epoch 3` 的约 `0.02` 提升到 `epoch 6` 的 `0.12+`
  3. 单 epoch 速度稳定在 `54~55s`，训练开销与基线基本一致
  4. 仍未出现 `csrd` 分项，符合 `warmup=20` 设计
- 当前判断: 继续
- 原因:
  - 当前已经通过最敏感的早期稳定性检查，下一真正关键点是 `ep10` 首次验证

### [2026-03-20 11:37] 检查点 #4 — Epoch 10

- 结果:
  - `ep10 = 39.8% / 52.9% / 68.8% / 74.3%`
- 对照:
  - `exp030a ep10 = 38.2 / 51.3 / 66.8 / 73.1`
- 当前观察:
  1. `exp119` 的首次验证 **明确高于基线**：`+1.6 mAP / +1.6 R1`
  2. 这说明即使在 `CSRD` 还没激活的 warmup 阶段，本轮训练也没有被新接线拖坏
  3. 但当前还不能把这部分领先归因到 `CSRD`，因为 `warmup=20`
- 当前判断: 继续
- 原因:
  - 需要看 `ep20` 和 `epoch 21+` 后 `csrd` 激活后的形态是否继续保持领先

### [2026-03-20 11:47] 检查点 #5 — Epoch 20-21

- 结果:
  - `ep20 = 47.6% / 61.5% / 75.6% / 80.5%`
- 对照:
  - `exp030a ep20 = 46.8 / 60.9 / 75.2 / 80.2`
- `CSRD` 激活后首批统计（ep21）:
  - `csrd = 0.013`
  - `csrd_tgap = 0.330~0.338`
  - `csrd_sgap = 0.243~0.253`
  - `csrd_vr = 1.000`
- 当前观察:
  1. `ep20` 仍保持领先：`+0.8 mAP / +0.6 R1`
  2. `CSRD` 激活后 loss 量级很小（约 `0.013`，加权后约 `0.0065`），没有压过主训练
  3. `teacher_gap > student_gap`，说明 teacher 的 pairwise 几何比 global 空间更可分，这是机制成立的前提
  4. `valid_ratio=1.000`，说明 batch 内 pairwise teacher 覆盖是完整的，没有“信号过 sparse”的问题
- 当前判断: 继续
- 原因:
  - 当前最值得看的是 `ep30/40` 时 student 是否继续追向 teacher gap

### [2026-03-20 11:58] 检查点 #6 — Epoch 30

- 结果:
  - `ep30 = 53.4% / 66.7% / 80.4% / 84.5%`
- 对照:
  - `exp030a ep30 = 52.2 / 66.0 / 79.2 / 84.2`
- `CSRD` 统计（ep30 末尾）:
  - `csrd = 0.013`
  - `csrd_tgap = 0.444`
  - `csrd_sgap = 0.386`
  - `csrd_vr = 0.999`
- 当前观察:
  1. `exp119` 到 `ep30` 仍稳定领先：`+1.2 mAP / +0.7 R1`
  2. `student_gap` 已从 `ep21` 的约 `0.25` 提升到 `0.386`，在持续追近 teacher 几何
  3. 这和之前许多“中性实验”不同，本轮不是只在训练 loss 上有变化，而是已经对应到验证领先
- 当前判断: 继续
- 原因:
  - 需要看 `ep40` 后这份领先能否保持，而不是像噪声一样回落

### [2026-03-20 12:10] 检查点 #7 — Epoch 40-44

- 结果:
  - `ep40 = 55.9% / 68.7% / 81.5% / 85.5%`
  - 当前进度: `epoch 44`
- 对照:
  - `exp030a ep40 = 55.6 / 68.6 / 81.1 / 85.4`
- `CSRD` 当前统计（ep41-44）:
  - `csrd = 0.011~0.013`
  - `csrd_tgap = 0.489~0.504`
  - `csrd_sgap = 0.454~0.470`
  - `csrd_vr = 0.999~1.000`
- 当前观察:
  1. 到 `ep40` 为止，领先仍在，但幅度缩小到 `+0.3 mAP / +0.1 R1`
  2. 这说明当前更像“持续弱正向”，还不是“中期强突破”
  3. `student_gap` 继续逼近 `teacher_gap`，说明 `CSRD` 并没有空转
  4. epoch 时间从 warmup 前的 `54~55s` 增加到 `63~64s`，说明 relational distillation 有约 `15%` 左右额外开销，但仍可接受
- 当前判断: 继续
- 原因:
  - 当前曲线不像失败，更像是需要看 `ep50/60` 才能判断它到底是“稳定弱正向”还是“最终回归等价”

### [2026-03-20 13:45] 检查点 #8 — 训练完成（Epoch 120）

- 最终训练监控口径:
  - `ep120 = 60.4% / 73.4% / 85.0% / 88.6%`
- `CSRD` 末期统计（ep117-120）:
  - `csrd = 0.011~0.012`
  - `csrd_tgap = 0.556~0.562`
  - `csrd_sgap = 0.538~0.545`
  - `csrd_vr = 0.999~1.000`
- 当前观察:
  1. `CSRD` 全程保持数值稳定，没有出现后期发散或压坏主训练
  2. `student_gap` 已基本追近 `teacher_gap`，说明 relational distillation 确实在塑形 global 空间
  3. 但训练监控默认口径最终只到 `60.4 / 73.4`，说明它还不是“只看训练曲线就能宣布突破”的实验
- 当前判断: 继续补正式评估
- 原因:
  - `exp119` 的真正价值更可能体现在 `equal_concat / global / cvk_hybrid` 的正式测试口径，而不是默认训练监控模式

### [2026-03-20 14:12] 检查点 #9 — 正式评估完成

- 正式评估结果:
  - `exp119a equal_concat = 61.1% / 73.2% / 85.4% / 88.6%`
  - `exp119b global = 60.4% / 70.3% / 82.8% / 87.4%`
  - `exp119c cvk_hybrid = 62.0% / 73.2% / 85.5% / 88.8%`
- 直接对照:
  - `exp030a-eq seed1234 = 61.1% / 72.9% / 85.2% / 87.8%`
  - `exp030a-g seed1234 = 59.8% / 69.9%`
  - `exp040b cvk_hybrid = 61.9% / 73.2% / 85.2% / 88.6%`
- 当前观察:
  1. `equal_concat` 基本持平，但 `R1` 仍有 `+0.3`
  2. `global` 出现当前最清楚的正向：`+0.6 mAP / +0.4 R1`
  3. `cvk_hybrid` 也保留了 `+0.1 mAP`
  4. 这说明 `CSRD` 的作用更像是把 common-support pairwise 几何蒸进 backbone/global，而不是直接替代 fusion 或 test-time correction
  5. 同时它也暴露了当前版本的瓶颈：teacher 仍来自单图 `kp_feats`，还不够 support-complete
- 当前判断: `exp119` 成立为弱正向主线候选，但不足以单独支撑最终论文方法
- 原因:
  - 下一步最合理的单变量不是扫 `CSRD` 权重/温度，而是把 `exp109` 的 support-complete teacher headroom 引回 `CSRD`，做更强的 relational teacher
