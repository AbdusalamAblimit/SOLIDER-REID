# exp099 POT-Match (Optimal Transport Matching) 监控

## 配置
- 测试时评估脚本 `scripts/eval_pot.py`
- 在 exp066 checkpoint 上评估 Sinkhorn OT 距离
- 无训练改动

## 结果
| 方法 | mAP | R1 |
|------|------|------|
| Global cosine (baseline) | 61.6% | 74.2% |
| OT-only (Sinkhorn) | 59.0% | 71.0% |
| Hybrid α=0.1 | 61.5% | 74.1% |
| Hybrid α=0.3 | 61.2% | 74.0% |
| Hybrid α=0.5 | 60.7% | 73.2% |
| Hybrid α=0.7 | 60.1% | 72.5% |

## 结论: ❌ OT 匹配不如 cosine (-2.6% mAP)
Per-keypoint 特征不够独立辨别力，OT 距离不如全局特征 cosine。
