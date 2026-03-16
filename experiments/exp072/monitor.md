# exp072 Part-Structured PAA (PS-PAA) 训练监控

## 实验信息
- **方法**: PSG + GCN + PS-PAA (body-part-aware encoder, ~63K params)
- **对照**: exp066 PAA (generic encoder, 52K params) = 61.6%/74.2%
- **核心变量**: PAA encoder 从 generic → body-part-structured

- **启动**: 2026-03-16 04:12, PID 2601540

---

### Ep10: 38.8%/52.4% (vs PAA: 38.4%/51.8%)
中性。与 PAA/S&C/PCL 的 ep10 几乎相同。

### Ep20: 47.3%

### Ep30: 52.6%

### Ep40: 56.2%

### Ep50: 57.6%

### Ep60: 58.6%

### Ep70: 58.5%

### Ep80: 60.4%/73.6%

### Ep90: 60.8%/73.8%

### Ep100: 60.7%/73.5%

### Ep110: 61.1%/73.9%

### Ep120（最终）: 61.1%/73.8%

---

## 最终结论

| 指标 | PAA (exp066) | PS-PAA (exp072) | 差异 |
|------|-------------|----------------|------|
| mAP | 61.6% | 61.1% | -0.5% |
| R1 | 74.2% | 73.8% | -0.4% |
| R5 | 85.4% | 84.8% | -0.6% |
| R10 | 88.4% | 88.4% | 0.0% |

**结论**: PS-PAA (body-part-aware) 略负于 generic PAA。body part 结构分组没有带来额外增益。说明 generic Conv2d 的 17→32 混合足以自动学到有效的通道组合，手工分组反而限制了信息流动。

**PAA 变体消融总结**:
| 变体 | vs PAA | 结论 |
|------|--------|------|
| PAA b128 (exp069) | -0.3%/+0.4% | 容量不是瓶颈 |
| S&C target-only (exp070) | -0.2%/-0.8% | scene 热图更好 |
| PCL LoRA (exp071) | -0.9%/-2.2% | feature-independent 更好 |
| PS-PAA parts (exp072) | -0.5%/-0.4% | generic 混合更好 |

**所有变体都不如原始 PAA。PAA 的简洁设计（generic Conv2d, scene heatmap, feature-independent）是最优的。**
