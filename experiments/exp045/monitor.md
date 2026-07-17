# exp045: 基于重建 `seed42` checkpoint 的 CVK 复核 — 监控日志

## 实验目标
- **目的**: 在 `exp044` 重建出的 `seed42` checkpoint 上，直接复核 `equal_concat` 与 `cvk_hybrid`
- **checkpoint**: `log/occluded_duke/exp044_exp030a_seed42_rebuild/transformer_120.pth`
- **子实验**:
  - `045a`: `equal_concat`
  - `045b`: `cvk_hybrid`

## 启动前检查
- [x] 与 `exp040` 相比，不改训练或模型结构，只更换 checkpoint
- [x] `TEST.WEIGHT` 已指向 `exp044` 产出的 `transformer_120.pth`
- [x] `045a/045b` 使用独立 `OUTPUT_DIR`
- [x] 当前结论会按“重建 seed42 复核”记录，不与历史原始 seed42 资产混写

---
## [启动] 运行计划

### 执行命令
- `045a`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp045_seed42_cvk_verify.yml MODEL.POSE_TEST_FEAT equal_concat OUTPUT_DIR ./log/occluded_duke/exp045a_seed42_eq`
- `045b`:
  - `/root/miniconda3/envs/solider-reid/bin/python -u test.py --config_file configs/occluded_duke/exp045_seed42_cvk_verify.yml OUTPUT_DIR ./log/occluded_duke/exp045b_seed42_cvk_hybrid`

### 当前判断
- **继续**

### 原因
1. `exp044` 已完成，当前最有价值的下一步就是把第二个 seed 的测试端证据补齐。
2. 这一步仍严格围绕 `exp030a` 主基线，不是偏题调参。

---
## [15:12] `045a` 完成

**状态**: ✅ 已完成，继续执行 `045b`

### 输出目录
- `log/occluded_duke/exp045a_seed42_eq`

### 评测结果
| 子实验 | 模式 | mAP | R1 | R5 | R10 |
|------|------|-----|----|----|-----|
| 045a | `equal_concat` | **60.2%** | **72.7%** | **84.4%** | **87.6%** |

### 观察
1. `045a` 与既有多 seed 文档中的 seed42 `equal_concat = 60.2%` 在 mAP 上完全一致，说明 `exp044` 的重建 checkpoint 具备可用的测试复核价值。
2. R1 为 `72.7%`，与旧表 `72.5%` 仅有很小差异，属于可接受波动。
3. 因此下一步可以直接把 `045b cvk_hybrid` 与 `045a` 做同 checkpoint 对照。

### 当前判断
- **继续**

### 原因
1. `045a` 已提供可靠直接对照。
2. 现在进入 `045b` 才能判断 `cvk_hybrid` 是否具备第二个 seed 的复核证据。

---
## [15:14] `045b` 完成

**状态**: ✅ 完成

### 输出目录
- `log/occluded_duke/exp045b_seed42_cvk_hybrid`

### 评测结果
| 子实验 | 模式 | mAP | R1 | R5 | R10 |
|------|------|-----|----|----|-----|
| 045b | `cvk_hybrid` | **61.1%** | **73.2%** | **84.2%** | **88.1%** |

### 对照差异
- vs `045a equal_concat`:
  - mAP: `60.2% -> 61.1%` (`+0.9%`)
  - R1: `72.7% -> 73.2%` (`+0.5%`)
  - R5: `84.4% -> 84.2%` (`-0.2%`)
  - R10: `87.6% -> 88.1%` (`+0.5%`)

### 观察
1. `cvk_hybrid` 在重建 `seed42` checkpoint 上继续给出稳定正 mAP，而且增幅与 `exp040` 的 `+0.8%` 高度接近。
2. 与 `exp040` 不同，这次 R1 也同步转正，说明“mAP 上升而 R1 小降”不是固定规律，R1 侧效应会随 checkpoint 而变。
3. 因而当前最稳妥的结论应收敛为：
   - **mAP 正增益具备跨 checkpoint 复核性**
   - R1 的具体变化方向暂不应写死

---
## exp045 阶段结论

1. `exp044` 重建出的 `seed42` checkpoint 已经通过 `045a` 证明可复用：
   - `equal_concat = 60.2% / 72.7%`
2. `cvk_hybrid` 在第二个 checkpoint 上继续成立：
   - `61.1% / 73.2%`
   - 相对 `equal_concat` 为 `+0.9% mAP / +0.5% R1`
3. 这使当前 CVK 主线从“单 checkpoint 正信号”进一步推进到“至少两个 checkpoint 上都能复核的正 mAP 信号”。
4. 下一步应继续补第三个 seed 资产，而不是回到细碎权重调参。
