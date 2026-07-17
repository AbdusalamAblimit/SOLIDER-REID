# 实验 exp216: Small GCN+PAA+CE+OA-SD 200 epochs

## 动机
- 所有 Small 实验跑 120 epochs
- 学习曲线显示 ep100-120 仍在增长 (+0.1-0.3%/10ep)
- 200 epochs 可能让模型充分收敛，获得更高 ceiling
- cosine schedule 的尾部衰减更慢 → 后期优化更充分

## 核心假设
200 epochs 给予模型更多优化时间，可能额外获得 +0.5-1% mAP。

## 技术方案
- 与 exp206r 完全相同，仅 MAX_EPOCHS=200

## 对照组
- exp206r (120ep): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
