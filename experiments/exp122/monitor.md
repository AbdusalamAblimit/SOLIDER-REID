# exp122 监控

## 实验信息
- 方法: SGW-SCRD（Support-Gap Weighted SCRD）
- 类型: 训练端单变量改进
- 主配置: `exp120`
- 核心变量: `POSE_CSRD_ANCHOR_WEIGHT_MODE = 'replace_ratio'`
- 输出目录: `log/occluded_duke/exp122_sgw_scrd`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp120` 只新增 `CSRD` anchor weighting
- [x] support-complete teacher 构造、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退 `exp120`
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 16:20] 实验准备

- 启动原因:
  1. `exp120` 机制上已证明 teacher 更强，但到 `ep90` 仍未优于 `exp119`
  2. `exp109` 的 oracle headroom 明显集中在低可见 / support-incomplete 样本
  3. 当前最合理的新假设不是继续增强 teacher，而是让 supervision 聚焦到真正 support-incomplete 的 anchor
- 当前判断: 待启动
- 原因:
  - `exp122` 是相对 `exp120` 最干净的单变量：teacher 不变，只改 `CSRD` 的 anchor 加权方式

### [2026-03-20 16:22] 首次后台启动未留住进程，立即改用前台确认

- 异常:
  1. 第一次 `nohup` 启动后，进程很快退出
  2. `nohup.log` 为空，未留下有效训练日志
- 处理:
  1. 不保留这次启动结果
  2. 立即改用前台短跑确认真实报错
  3. 确认并非代码错误，而是后台启动未留住；随后以前台持续会话正式启动
- 当前判断: 继续
- 原因:
  - 代码接线和 config 均已确认正常，异常只发生在首次后台留驻阶段，不属于机制失败

### [2026-03-20 16:23] 启动确认（正式训练）

- 运行位置: 本地 3090
- 配置: `configs/occluded_duke/pose_psg_gcn_sgw_scrd.yml`
- 输出目录: `log/occluded_duke/exp122_sgw_scrd`
- 关键确认:
  1. 新开关已生效：`POSE_CSRD_ANCHOR_WEIGHT_MODE=replace_ratio`
  2. support-complete teacher 仍正常启用：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  3. 训练已正常进入 loop，无 `NaN / OOM / 配置回退`
- 当前判断: 继续
- 原因:
  - 现在需要确认 warmup 前段仍与 `exp119/120` 同型，证明新变量只会在 `epoch > 20` 后介入

### [2026-03-20 16:23] 检查点 #1 — Epoch 1 Iter 60

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
  1. warmup 前段与 `exp119/120` 完全同型，没有新增不稳定
  2. 当前仍不会出现 `csrd / csrd_ar / csrd_aw`，符合 `warmup=20` 的设计
  3. 说明 anchor weighting 至少没有污染最敏感的早期收敛阶段
- 当前判断: 继续
- 原因:
  - 下一关键点仍是 `ep10 / ep20`，以及 `epoch 21+` 后 `csrd_ar / csrd_aw` 是否表明监督真的聚焦到了 support-gap anchor

### [2026-03-20 16:24] 检查点 #2 — Epoch 1-2

- 当前进度:
  - `Epoch 1 done`
  - `Epoch 2 Iter 20/227`
- 当前局部训练状态:
  - `Epoch 1 done`: `Time per epoch = 59.861s`, `ETA = 1h58m`
  - `Epoch[2] Iter[20/227] Loss: 13.185, Acc: 0.005`
- 当前观察:
  1. `Epoch 1 -> 2` 总 loss 正常下降，warmup 轨迹与 `exp119/120` 保持一致
  2. 单 epoch 速度约 `60s`，和 `exp120` 同量级，没有因 anchor weighting 带来额外开销
  3. 当前仍未出现 `csrd` 分项，符合设计预期
- 当前判断: 继续
- 原因:
  - 早期稳定性已确认；接下来只需要等 `ep10 / ep20` 首次验证，再看 `epoch 21+` 的新统计项是否合理

### [2026-03-20 17:06] 检查点 #3 — Epoch 10/20/30 与 selective `CSRD` 激活后形态

- 结果:
  - `ep10 = 38.3% / 51.4% / 67.1% / 73.3%`
  - `ep20 = 47.0% / 60.7% / 75.3% / 80.1%`
  - `ep30 = 52.5% / 65.5% / 80.0% / 84.5%`
- 对照:
  - `exp119 ep10 = 39.8 / 52.9`
  - `exp119 ep20 = 47.6 / 61.5`
  - `exp119 ep30 = 53.4 / 66.7`
  - `exp120 ep30 = 53.2 / 66.5`
  - `exp030a ep30 = 52.2 / 66.0`
- `SGW-SCRD` 统计（ep30-40 前段）:
  - `csrd = 0.014~0.017`
  - `csrd_tgap = 0.433~0.488`
  - `csrd_sgap = 0.367~0.443`
  - `csrd_vr = 0.999~1.000`
  - `csrd_ar = 0.53~0.57`
  - `csrd_aw = 0.132~0.149`
  - `csrd_sr = 0.132~0.149`
  - `csrd_sn = 144~161`
- 当前观察:
  1. `exp122` 的 selective supervision 已经按设计生效：`csrd_ar < 1`，只有约 `53-57%` 的 anchor 实际参与 `CSRD`
  2. 同时 `csrd_aw ≈ csrd_sr`，说明当前 anchor 权重本质上就是 sample-level support gap，没有接线错误
  3. 但到 `ep30` 为止，指标明显弱于 `exp119/120`，甚至只比 `exp030a` 略高 `+0.3 mAP / -0.5 R1`
  4. 这说明 “按 replace_ratio 做 sample-level selective weighting” 至少在第一版上 **没有把监督聚焦变成收益**
- 当前判断: 谨慎继续，但已出现负信号
- 原因:
  - 目前不是实现失败，而是机制负向；下一关键点是 `ep40`，如果仍明显落后于 `exp119/120`，就可以判定这条 sample-level weighting 方向不成立
