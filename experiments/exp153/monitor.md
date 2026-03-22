# exp153 MaxSim Triplet Additive 监控

## 实验信息
- 方法: MaxSim Triplet 补充模式（pooled triplet + 0.25 × maxsim triplet）
- 对照: exp152b (替换模式, -3.3%), exp030a (baseline)
- 运行位置: 本地 3090
- CHECKPOINT_PERIOD: 20（每 20 epoch 保存）

## 止损条件
- ep40 equal_concat mAP < exp030a ep40 (55.6%) 1.0% 以上 → 止损
