# exp157c Gradient Occlusion 监控
- 方法: Bottom-heavy gradient occlusion (下半身高概率、上半身低概率)
- 基线: exp030a-eq (60.73%), ROA (61.8%), PLBOA (exp157, 进行中)
- 运行: 远程 5060 Ti
- p=0.7, mode=gradient

## 监控

### [2026-03-23 14:26] ep10 — 严重落后
- mAP 27.4% / R1 41.9%
- vs exp030a: -10.8 / -9.4 ← gradient 太激进
- tri_part=11.1 (baseline 5x)
- 需要看 ep30-50 能否追回
