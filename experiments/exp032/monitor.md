# exp032 训练监控日志

## 实验配置
- **Config**: `configs/occluded_duke/pose_psg_keypoint_pool.yml`
- **Output**: `./log/occluded_duke/exp032_psg_keypoint_pool`
- **核心变量**: PSG + Keypoint Pooling Only（保留 17 个关键点采样 + 置信度加权池化，不做图传播）
- **对照**:
  - exp007a (PSG + 0.5x loss, 无 keypoint head): mAP 59.5%, R1 69.8%
  - exp030a (PSG + keypoint pooling + GCN): eq 61.1%, gcn_only 58.2%
  - exp030b (PSG + GCN, w_p=0.01): eq 60.5%, gcn_only 56.9%
- **预期**:
  - 如果接近 exp030a，则主要增益来自 keypoint pooling
  - 如果明显低于 exp030a，则图传播仍有独立贡献

---

## 训练期间默认评估曲线

训练默认 `POSE_TEST_FEAT=concat_scaled`，因此下表反映的是训练过程中 `global + keypoint branch` 的融合特征表现。

| Epoch | mAP | R1 | R5 | R10 |
|------|-----|----|----|-----|
| 10 | 33.8% | 47.3% | 64.3% | 70.4% |
| 20 | 42.8% | 57.3% | 72.6% | 77.4% |
| 30 | 49.8% | 62.9% | 77.6% | 82.3% |
| 40 | 53.4% | 66.8% | 80.9% | 85.2% |
| 50 | 55.4% | 68.7% | 82.5% | 86.5% |
| 60 | 55.7% | 69.5% | 82.7% | 86.9% |
| 70 | 58.0% | 71.3% | 84.5% | 88.0% |
| 80 | 58.6% | 71.8% | 84.6% | 88.0% |
| 90 | 58.2% | 71.3% | 85.2% | 88.1% |
| 100 | 58.7% | 72.0% | 85.2% | 88.1% |
| 110 | 59.3% | 72.4% | 84.9% | 88.3% |
| 120 | **59.3%** | **72.4%** | **85.1%** | **88.4%** |

**观察**:
- 训练曲线整体平稳，没有出现 exp030a 那种“E50 后由图分支成熟带来的明显二次跳升”
- E70 后进入平台期，最终 `concat_scaled=59.3%`
- 这已经高于 `exp007` (58.3%)，说明仅靠 pose-guided sparse pooling 就能带来可观增益

---

## 训练末尾 Loss 状态

E120 末尾日志（iter 200/227）:

| 指标 | 数值 |
|------|------|
| Total Loss | 0.766 |
| ID Global | 0.178 |
| ID Part | 0.310 |
| Tri Global | 0.023 |
| Tri Part | 1.021 |
| Acc | 0.992 |

**观察**:
- `ID Part` 从早期的 ~6.0 降到 0.31，说明 keypoint branch 本身是能学起来的
- 但 `Tri Part` 始终明显高于 `Tri Global`，说明 branch ranking 质量仍弱于 global
- 这和最终测试现象一致：branch 单独可用，但最强结果仍然来自和 global 融合

---

## 四种测试模式最终结果

使用 `transformer_120.pth` 额外运行 `test.py` 四种模式：

| Test Mode | mAP | R1 | R5 | R10 |
|-----------|-----|----|----|-----|
| global | 59.8% | 70.0% | 81.7% | 85.4% |
| concat_scaled | 59.3% | 72.4% | 85.1% | 88.4% |
| **equal_concat** | **60.2%** | **72.5%** | **85.1%** | **88.3%** |
| gcn_only* | 54.7% | 69.9% | 82.4% | 86.0% |

\* `gcn_only` 在本实验里实际上是 **keypoint-pooling-only branch**，因为图传播被关闭了。

**直接结论**:
- `global` 仍然有 59.8%，和 exp030a 的 `global=59.8%` 几乎完全相同
- `equal_concat` 把 `global` 从 59.8% 提到 60.2%，说明 **keypoint pooling 本身就有补充信息**
- `gcn_only=54.7% / 69.9%` 并不低，解释了为什么此前在 exp030b 中即便图传播没学好，branch-only 和 R1 仍然偏高

---

## 对照分析

### vs exp007a

| 实验 | global | best fusion | branch-only |
|------|--------|-------------|-------------|
| exp007a | 59.5% / 69.8% | — | — |
| **exp032** | **59.8% / 70.0%** | **60.2% / 72.5%** | **54.7% / 69.9%** |

**解读**:
- 仅仅加一个 keypoint pooling branch，就已经能在 `equal_concat` 上比 exp007a 再高 `+0.7% mAP`
- 这证明 pose-guided sparse pooling 本身就是强基线，不需要图传播也不是“随机噪声”

### vs exp030a

| 模式 | exp032 (KPP only) | exp030a (KPP + GCN) | Δ |
|------|-------------------|---------------------|---|
| global | 59.8% / 70.0% | 59.8% / 69.5% | 0.0% / +0.5% |
| concat_scaled | 59.3% / 72.4% | 60.5% / 73.7% | -1.2% / -1.3% |
| equal_concat | 60.2% / 72.5% | 61.1% / 73.7% | -0.9% / -1.2% |
| branch-only | 54.7% / 69.9% | 58.2% / 72.9% | -3.5% / -3.0% |

**解读**:
- `global` 几乎完全不变，说明 **GCN 不解释 global 提升**
- 但三个包含 branch 的测试模式都明显低于 exp030a，尤其 branch-only 差了 `3.5% mAP`
- 这说明图传播不是没用，而是 **主要提升 branch 质量和 fusion 增益**

### vs exp030b

| 模式 | exp032 (KPP only) | exp030b (GCN w_p=0.01) | Δ |
|------|-------------------|------------------------|---|
| global | 59.8% / 70.0% | 60.6% / 71.0% | -0.8% / -1.0% |
| equal_concat | 60.2% / 72.5% | 60.5% / 73.0% | -0.3% / -0.5% |
| branch-only | 54.7% / 69.9% | 56.9% / 70.9% | -2.2% / -1.0% |

**解读**:
- exp030b 仍然整体更高，说明 single-seed 方差依旧存在
- 但 exp032 已经证明：exp030b 里高 `gcn_only` / 高 `R1` 的很大一部分，完全可以由 keypoint pooling 本身解释
- 因此不能把 exp030b 的现象解读为“图传播虽然 loss 很低但其实偷偷学得很好”

---

## 最终结论

1. **Keypoint pooling 本身很强**：
   - 无图传播时，branch-only 仍有 `54.7% / 69.9%`
   - `equal_concat` 仍能把 global 从 `59.8%` 提到 `60.2%`

2. **GCN 仍然有独立价值，但作用不在 global**：
   - exp032 明显低于 exp030a：
     - `equal_concat`: `60.2%` vs `61.1%`
     - `concat_scaled`: `59.3%` vs `60.5%`
     - `branch-only`: `54.7%` vs `58.2%`
   - 所以图传播主要提升的是 branch 特征质量和 fusion 增益

3. **global 提升不是 GCN 带来的**：
   - exp032 和 exp030a 的 `global` 都是 `59.8%`
   - 这和代码里的 `detach()` 机制一致：branch 不会反向影响 backbone/global

4. **对 exp030b 的重新解释**：
   - exp030b 中“GCN 没怎么学但 equal_concat / gcn_only 还是高”，并不奇怪
   - 现在更合理的解释是：**关键点采样 + 置信度池化本身已经很强，GCN 只是进一步把它做得更好**

---

## [2026-03-13] 事后修正：结合 exp030a 多种子后的最终定位

后续 4090 多种子结果表明：

- `exp030a-global` mean = **59.33%**
- `exp030a-equal_concat` mean = **60.73%**
- paired diffs = `(1.3, 1.1, 1.8)`, `p=0.0214`

### 因此这份实验现在应该这样使用

1. **它最可靠地支持了“keypoint pooling 本身就是强基线”**
   这一点仍然成立，而且很重要。

2. **它不应再被拿来单独量化 GCN 的精确贡献**
   因为 `exp032` 仍只有单 seed，而 `exp030a` 已经有了更可靠的 3-seed fusion 结果。

3. 更准确的综合表述是：
   - **KPP 提供了 branch 的大头信息量**；
   - **GCN 提供的是 refinement / relation modeling**，把 branch 做得更可融合、更稳定；
   - 因此论文里如果保留 GCN，最好把它写成“对强 KPP baseline 的进一步结构化增强”，而不是把全部 branch 增益都归功于图传播。
