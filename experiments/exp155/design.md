# 实验 exp155: Evidential Deep Learning for GCN Branch

## 动机

当前 GCN branch 用 CrossEntropy 分类。CE 对所有样本施加均匀梯度压力——不管样本是否被遮挡，都要求高置信预测。这对遮挡样本可能是有害的（强迫模型编码"虚假置信度"）。

Evidential DL（Sensoy et al., NeurIPS 2018）将分类头改为 Dirichlet 分布参数预测。模型不只预测"哪个类"，还预测"对这个预测有多确信"。对遮挡样本，Evidential loss 允许模型说"我不确定"而不是被强迫给出高置信错误预测。

**这从根本上改变了"模型输出什么"——从点估计概率到概率分布上的分布。**

## 文献定位

- Sensoy et al. (NeurIPS 2018): Evidential DL 原始论文，用于 MNIST/CIFAR
- 医学影像领域广泛使用（部分观测、缺失数据场景）
- **从未在 person ReID 中使用过**

## 技术方案

### 只改 GCN branch 的 ID loss

```
Global branch:  Linear(768, 702) → CrossEntropy  [不变]
GCN branch:     Linear(768, 702) → softplus → α = evidence + 1 → Evidential Loss  [改]
Triplet:        两个 branch 各自不变
```

### Evidential Loss = Bayes Risk + KL Regularization

```
Bayes Risk = Σ_k y_k * (log S - log α_k)     # S = Σα_k
KL = KL(Dir(α̃) || Dir(1,...,1))               # α̃ removes true-class evidence
Total = Bayes Risk + λ * anneal * KL
```

KL 退火：从 0 线性增长到 1.0，在 60% epochs (ep72) 达到满值。

### 关键日志

- `evid_br`: Bayes Risk（应类似 CE loss 的量级）
- `evid_kl`: KL 散度（应从 0 开始随 annealing 增长）
- `evid_unc`: 平均 uncertainty（K/S，0~1，越低越确信）
- `evid_ev`: 平均 evidence（Σα-K，越高越确信）
- `evid_ann`: 退火系数

## 对照组

- exp030a (baseline, 60.73% mAP 3-seed mean)
- exp153 (MaxSim additive, ~60.6% mAP, 中性)

## 预期结果

- 如果 Evidential 让 feature 空间"更诚实"：mAP 微正 (+0.3~1.0)
- 如果 KL 正则过强导致 GCN branch 学不好：mAP 负
- 关键看 `evid_unc`: 如果遮挡样本确实显示更高 uncertainty → 机制生效

## 风险

1. K=702 时 KL 绝对值很大，kl_reg=0.1 可能仍然太高
2. Evidential loss 梯度量级与 CE 不同，可能需要调 loss weight
3. AMP 下 lgamma/digamma 精度不够 → 已 force float32

## 止损

- ep30 mAP 低于 exp030a ep30 (52.2%) 2.0% 以上 → 止损
- evid_kl 爆炸（>100）→ 降低 kl_reg
