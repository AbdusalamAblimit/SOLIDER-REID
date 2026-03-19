# exp102: SGMT with 50% masking ratio

## 动机
- exp101 SGMT (30% mask) 中性偏负 (-0.6%)
- 30% masking 可能太温和 — GCN 2-hop 传播轻松恢复
- 50% masking 更激进 → 更强的正则化效果 → 但也更难收敛

## 对照
- exp066 PAA baseline: 61.6%/74.2%
- exp101 SGMT-30%: 61.0%/73.8%
