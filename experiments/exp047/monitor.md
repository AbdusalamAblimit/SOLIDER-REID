# exp047: Common-Support-Guided Triplet (CSGT) — 监控日志

## 实验目标
- **目的**: 把 retrieval-time 的 common-support 信号迁到训练端 triplet mining
- **配置**: `configs/occluded_duke/exp047_csgt_triplet.yml`
- **输出目录**: `log/occluded_duke/exp047_csgt_triplet`

## 启动前检查
- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp030a` 只新增 `POSE_CSGT` 相关开关
- [x] 默认代码路径保持不变，开关关闭时完全回退 baseline
- [x] `POSE_CSGT` 不依赖 `POSE_KP_TRIPLET` 才能生效
- [x] `CSGT` 作为独立损失项额外叠加，不并入已有 global/part 混合权重
- [x] 当前先完成代码接入与设计，不抢占正在运行的 `exp046`
- [x] 最终评测口径已固定：
  - `equal_concat` 主汇报
  - `global` 机制对照
  - 训练中的 `concat_scaled` 只作为监控口径

---
## [准备完成] 当前状态

### 计划命令
- `/root/miniconda3/envs/solider-reid/bin/python -u train.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml`

### 训练后必须补的评测
- `exp047a-eq`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml MODEL.POSE_TEST_FEAT equal_concat TEST.WEIGHT ./log/occluded_duke/exp047_csgt_triplet/transformer_120.pth OUTPUT_DIR ./log/occluded_duke/exp047a_csgt_eq`
- `exp047b-global`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp047_csgt_triplet.yml MODEL.POSE_TEST_FEAT global TEST.WEIGHT ./log/occluded_duke/exp047_csgt_triplet/transformer_120.pth OUTPUT_DIR ./log/occluded_duke/exp047b_csgt_global`

### 当前判断
- **等待 GPU 空档后启动**

### 原因
1. `exp046` 正在重建 `seed2024` checkpoint，按当前优先级应继续保持。
2. `exp047` 的代码和配置可先准备好，等 `exp046` 进入更稳定阶段或结束后立即开跑。

---
## [校正] 关键接线修复

### 修复内容
1. `POSE_CSGT` 现在可独立拿到 `kp_data`，不再依赖 `POSE_KP_TRIPLET=True`
2. `CSGT` 现在作为独立损失项额外叠加，不再被 `wt_g` 隐式打折
3. `POSE_KP_TRIPLET=False` 时，不会误走逐关键点 triplet 分支

### 最小验证
- `py_compile` 已通过：
  - `processor/processor.py`
  - `loss/make_loss.py`
  - `config/defaults.py`
- 随机张量 smoke test 已通过：
  - `POSE_CSGT=True`
  - `POSE_KP_TRIPLET=False`
  - `tri_csgt` 能正常出现在 `loss_details`
  - `tri_kp` 不会误触发

### 当前判断
- **可启动**

### 原因
1. 之前的“空跑 baseline”风险已排除。
2. 代码与 `design.md` 的损失定义现已一致。

---
## [19:03] 检查点 #1

**状态**: ▶️ 已启动

### 训练进度
- 已进入：`Epoch 1`
- 最新位置：`Iter 40/227`

### 当前局部训练状态
- `Epoch 1 Iter 20/227`：
  - `Loss`: `35.772`
  - `Acc`: `0.000`
  - `id_global`: `6.555`
  - `id_part`: `6.683`
  - `tri_global`: `14.249`
  - `tri_part`: `16.638`
  - `tri_csgt`: `13.710`
- `Epoch 1 Iter 40/227`：
  - `Loss`: `30.971`
  - `Acc`: `0.000`
  - `id_global`: `6.556`
  - `id_part`: `6.681`
  - `tri_global`: `11.848`
  - `tri_part`: `14.060`
  - `tri_csgt`: `11.399`
  - `csgt_pos_overlap`: `0.666`
  - `csgt_neg_overlap`: `0.708`
  - `csgt_pos_fallback`: `0.800`
  - `csgt_neg_fallback`: `0.100`

### 观察
1. 训练已正常启动，数据、模型、优化器和 pose 路径均完成加载。
2. `tri_csgt` 已在首轮日志中出现，说明这次不是“配置开了但实际没生效”的空跑 baseline。
3. `csgt_pos/neg_overlap` 与 fallback 统计也已打印，后续可以直接据此判断 common-support mining 是否真的在工作。

### 当前判断
- **继续**

### 原因
1. 首轮损失下降正常，暂未见 `NaN / OOM / 爆 loss`。
2. 关键新增训练项已经实际接线生效，具备继续观察价值。

---
## [19:06] 检查点 #2

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 3/120`
- 最新 ETA：约 `1h52m`

### 当前局部训练状态
- `Epoch 1` 末：
  - `Loss`: `20.170`
  - `Acc`: `0.001`
- `Epoch 2` 末：
  - `Loss`: `12.081`
  - `Acc`: `0.007`
- `Epoch 3` 末：
  - `Loss`: `9.475`
  - `Acc`: `0.067`
  - `id_global`: `6.521`
  - `id_part`: `5.874`
  - `tri_global`: `1.252`
  - `tri_part`: `3.008`
  - `tri_csgt`: `1.147`
  - `csgt_pos_overlap`: `0.650`
  - `csgt_neg_overlap`: `0.672`
  - `csgt_pos_fallback`: `0.860`
  - `csgt_neg_fallback`: `0.060`

### 观察
1. `Epoch 1 -> 3` 的总 loss 从 `20.170` 快速降到 `9.475`，起步收敛正常。
2. `tri_csgt` 从首轮 `13.710` 快速下降到 `1.147`，说明新增约束正在被模型吸收，而不是单独失控。
3. overlap 统计目前稳定在 `0.65-0.67`，说明 batch 内确实存在可用的 common-support pair。
4. `neg_fallback` 很低，但 `pos_fallback` 仍较高，后续需要继续观察阈值 `0.3` 下正样本可比对是否仍然偏紧。

### 当前判断
- **继续**

### 原因
1. 目前没有任何中断信号，训练稳定。
2. 新机制的关键内部统计已经开始提供可解释性信息，值得继续观察到 `Epoch 10` 首次评测。

---
## [19:08] 检查点 #3

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 5/120`
- 当前位于：`Epoch 6 Iter 120/227`
- 最新 ETA：约 `1h50m`

### 当前局部训练状态
- `Epoch 4` 末：
  - `Loss`: `8.538`
  - `Acc`: `0.167`
- `Epoch 5` 末：
  - `Loss`: `8.063`
  - `Acc`: `0.214`
  - `id_global`: `6.404`
  - `id_part`: `5.227`
  - `tri_global`: `0.792`
  - `tri_part`: `2.280`
  - `tri_csgt`: `0.711`
  - `csgt_pos_overlap`: `0.651`
  - `csgt_neg_overlap`: `0.666`
  - `csgt_pos_fallback`: `0.800`
  - `csgt_neg_fallback`: `0.090`
- `Epoch 6 Iter 120/227`：
  - `Loss`: `7.884`
  - `Acc`: `0.155`
  - `tri_csgt`: `0.647`
  - `tri_global`: `0.712`
  - `tri_part`: `2.196`

### 观察
1. `Epoch 4 -> 5` 仍在继续下降，但降幅已从起步阶段的快速下落切到更正常的 warmup 收敛节奏。
2. `tri_csgt` 已从首轮的双位数下降到 `0.6-0.7` 区间，并且与 `tri_global` 同量级，说明它没有压过原始 global triplet。
3. `tri_part` 仍明显高于 `tri_global / tri_csgt`，目前 branch 学习仍是前段主要难点，这和 `exp030a` 早期日志形态一致。
4. `csgt_pos_overlap / neg_overlap` 继续稳定在 `0.65-0.67`，没有出现“阈值过高导致几乎无可用 pair”的迹象。

### 当前判断
- **继续**

### 原因
1. 前 5 个 epoch 的 warmup 已平稳通过，没有异常信号。
2. 从现在开始切换到 `Epoch 6-30` 的约 3 分钟轮询，等待 `Epoch 10` 首次评测。

---
## [19:13] 检查点 #4

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 10/120`
- 当前位于：`Epoch 12 Iter 40/227`
- 最新 ETA：约 `1h46m`

### Epoch 10 验证结果
| Epoch | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| 10 | **38.7%** | **52.9%** | **67.2%** | **73.9%** |

### 当前局部训练状态
- `Epoch 10` 末：
  - `Loss`: `6.221`
  - `Acc`: `0.251`
  - `id_global`: `5.664`
  - `id_part`: `3.973`
  - `tri_global`: `0.482`
  - `tri_part`: `1.463`
  - `tri_csgt`: `0.431`
  - `csgt_pos_overlap`: `0.650`
  - `csgt_neg_overlap`: `0.661`
  - `csgt_pos_fallback`: `0.740`
  - `csgt_neg_fallback`: `0.080`
- `Epoch 11` 末：
  - `Loss`: `5.812`
  - `Acc`: `0.240`

### 观察
1. `Epoch 10` 已顺利完成首次验证，`38.7 / 52.9` 说明训练至少处在正常收敛轨道，没有出现早期崩坏。
2. `tri_csgt` 继续与 `tri_global` 保持接近量级，说明额外约束在参与训练，但没有压倒原始 global triplet。
3. overlap 统计仍稳定在 `0.65` 左右，说明 common-support pair 供给稳定。
4. `csgt_pos_fallback` 仍在 `0.74` 附近，后续如果中段仍维持高位，需要再判断阈值是否让正样本筛选过严。

### 当前判断
- **继续**

### 原因
1. 首次评测没有给出中断信号。
2. 当前最值得观察的是 `Epoch 20` 指标和 `csgt_pos_fallback` 是否进一步下降。

---
## [19:24] 检查点 #5

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 20/120`
- 当前位于：`Epoch 21`
- 最新 ETA：约 `1h38m`

### Epoch 20 验证结果
| Epoch | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| 20 | **42.3%** | **56.2%** | **71.0%** | **76.7%** |

### 相比 Epoch 10
- mAP: `38.7% -> 42.3%` (`+3.6%`)
- R1: `52.9% -> 56.2%` (`+3.3%`)

### 当前局部训练状态
- `Epoch 20` 末：
  - `Loss`: `3.364`
  - `Acc`: `0.562`
  - `id_global`: `3.095`
  - `id_part`: `1.939`
  - `tri_global`: `0.282`
  - `tri_part`: `0.929`
  - `tri_csgt`: `0.241`
  - `csgt_pos_overlap`: `0.650`
  - `csgt_neg_overlap`: `0.651`
  - `csgt_pos_fallback`: `0.745`
  - `csgt_neg_fallback`: `0.090`
- `Epoch 21 Iter 200/227`：
  - `Loss`: `3.175`
  - `Acc`: `0.616`
  - `tri_csgt`: `0.220`

### 观察
1. `Epoch 20` 相比 `Epoch 10` 有明确提升，说明训练曲线仍在正常抬升。
2. `tri_csgt` 已下降到 `0.2` 量级，但仍稳定存在，说明它在中段没有消失，也没有失控放大。
3. `csgt_pos_fallback` 仍约 `0.74`，比理想状态偏高，但没有继续恶化；现阶段更像“正样本筛选偏紧”，还不是异常终止理由。
4. `tri_part` 仍高于 `tri_global / tri_csgt`，当前瓶颈依旧更像 branch 端收敛，而不是 CSGT 额外扰乱训练。

### 当前判断
- **继续**

### 原因
1. 第二次验证仍在上升，没有出现训练端创新常见的中段崩坏。
2. 下一关键观察点应转到 `Epoch 30`，看中段评测能否继续沿基线量级上升。

---
## [19:35] 检查点 #6

**状态**: ▶️ 运行中

### 训练进度
- 已完成：`Epoch 30/120`
- 当前位于：`Epoch 31`
- 最新 ETA：约 `1h30m`

### Epoch 30 验证结果
| Epoch | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| 30 | **46.3%** | **60.5%** | **74.9%** | **80.2%** |

### 相比 Epoch 20
- mAP: `42.3% -> 46.3%` (`+4.0%`)
- R1: `56.2% -> 60.5%` (`+4.3%`)

### 当前局部训练状态
- `Epoch 30` 末：
  - `Loss`: `1.652`
  - `Acc`: `0.857`
  - `id_global`: `1.401`
  - `id_part`: `0.932`
  - `tri_global`: `0.133`
  - `tri_part`: `0.632`
  - `tri_csgt`: `0.103`
  - `csgt_pos_overlap`: `0.647`
  - `csgt_neg_overlap`: `0.649`
  - `csgt_pos_fallback`: `0.695`
  - `csgt_neg_fallback`: `0.065`
- `Epoch 31 Iter 20/227`：
  - `Loss`: `1.821`
  - `Acc`: `0.759`
  - `tri_csgt`: `0.073`

### 观察
1. `Epoch 30` 继续稳定上升，说明 `CSGT` 目前至少没有破坏主训练轨迹。
2. `tri_csgt` 已降到 `0.1` 量级，且仍与 `tri_global` 同步存在，说明它在中段仍被优化器利用。
3. `csgt_pos_fallback` 从 `Epoch 20` 的 `0.745` 降到 `0.695`，说明正样本筛选虽然仍偏紧，但没有恶化，反而在缓慢改善。
4. 到 `Epoch 30` 为止，最稳妥的结论是：**训练稳定，机制有效接线，值得继续跑完整训练。**

### 当前判断
- **继续**

### 原因
1. 中段曲线持续上升，没有看到提前终止的证据。
2. 从 `Epoch 30+` 开始切换到约 `5` 分钟轮询，下一关键点看 `Epoch 40` 验证。

---
## [终止] 实验失败分析

**状态**: ❌ 终止（Epoch 60 时进程中断，无 checkpoint）

### 根本失败原因

**CSGT 的核心假设不成立**：common-support overlap 无法区分正负 pair。

| 统计量 | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 50+ |
|--------|----------|----------|----------|-----------|
| pos_overlap | 0.650 | 0.650 | 0.647 | 0.640 |
| neg_overlap | 0.661 | 0.651 | 0.649 | 0.655 |
| **gap** | **0.011** | **0.001** | **0.002** | **0.015** |

正负 pair 的 overlap 差异始终 < 0.02，机制本质上无法区分"哪些 pair 更可比"。

### 为什么 overlap 不区分正负 pair

1. **keypoint visibility 是 image-level 属性**，不是 identity-level 属性
2. 一个 anchor 与正样本的关键点重叠，和与负样本的关键点重叠，取决于两张图各自的遮挡模式
3. 遮挡模式由相机角度和场景布局决定，与身份无关
4. 因此 overlap ≈ f(image_a, image_b)，不携带 identity 信息

### 另一个失败信号

- `csgt_pos_fallback ≈ 0.7-0.8`：70-80% 的正样本 pair 因 overlap 不足而回退到标准 mining
- 这意味着 CSGT 在大部分时间里就是标准 triplet

### 最终判断

**CSGT 作为训练端创新失败。** 不需要等最终 120 epoch 结果。

### 对 common-support 路线的影响

1. **retrieval-time CVK 仍然成立**：因为 CVK 做的不是 overlap 筛选，而是 pair-specific keypoint distance 替代
2. **但把 CVK 信号迁移到训练端的 simple approach 失败了**
3. 如果要继续推进训练端 common-support，需要完全不同的机制：
   - 不是用 overlap 做 mining filter
   - 而是改变 loss 本身的计算方式（如只在共同可见区域上计算距离）
