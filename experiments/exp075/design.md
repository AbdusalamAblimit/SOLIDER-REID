# 实验 exp075: PAA Multi-Seed 验证

## 动机
- PAA (exp066) 单 seed 结果: 61.6%/74.2%
- PAA+ROA (exp067) 单 seed 结果: 62.0%/73.7%
- 3-seed baseline: 60.73%/72.57%
- 需要确认 PAA 的增益 (+0.87%/+1.63%) 是否跨 seed 稳定
- 远程 5060 Ti 已验证 seed1234: 61.2%/74.3% (一致)

## 实验计划
1. **本地**: PAA+ROA seed42 → 确认最高 mAP 配置
2. **远程**: PAA seed42 → 确认核心创新
3. 后续: seed2024 在有空时补充

## 对照
- exp030a 3-seed: 60.73%/72.57% (baseline)
- exp066 seed1234: 61.6%/74.2% (PAA)
- exp067 seed1234: 62.0%/73.7% (PAA+ROA)
