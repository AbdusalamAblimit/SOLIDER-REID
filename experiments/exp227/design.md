# 实验 exp227: GSPB(0.005) + PADPQ K=4 on Small

## 动机
- exp225 (GSPB+PADPQ on Tiny): 64.2/74.9 (+1.0/-0.5 vs OA-SD)
- GSPB scale=0.05 在 Small 上灾难 (exp222: 2.3% at ep10)
- 假设: 极低 scale (0.005) 可能在 Small 上安全
- PADPQ 在 Tiny 上对 mAP 正向，可能 Small 也如此

## 核心假设
GSPB scale=0.005 + PADPQ 在 Small 上不灾难且有正面效果。

## 风险
高风险。GSPB scale=0.01 在 Small 上也是灾难 (15.1% at ep10)。
scale=0.005 可能仍然太强。如果 ep10 < 30%，立即终止。

## 对照组
- exp206r (Small OA-SD): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
