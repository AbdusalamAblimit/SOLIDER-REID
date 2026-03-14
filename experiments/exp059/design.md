# 实验 exp059: ROA + PGAM 组合

## 动机
- exp054 PGAM: +0.37% mAP / +1.23% R1（微弱正向，边缘）
- exp058 ROA: +1.07% mAP / +0.23% R1（显著正向，历史最高 mAP）
- 两者作用于完全不同维度：ROA 改数据增强（输入），PGAM 改注意力（backbone 内部）
- 如果正交，组合应叠加增益

## 技术方案
- 组合 PGAM (Stage 3, threshold=0.3) + ROA (prob=0.5, VOC 2012)
- 配置文件合并 exp054 和 exp058 的设置

## 对照组
- exp054 (PGAM only), exp058 (ROA only), exp030a (baseline)
