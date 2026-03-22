# 实验 exp155b: Evidential DL with Lower KL (kl_reg=0.01)

## 动机
exp155 中 Evidential 的 id_part=11.1（CE 同期 ~0.5），说明 Bayes Risk 梯度太弱。
kl_reg=0.1 时 KL 项（~54×0.1×anneal=~5.4）与 Bayes Risk（~5.8）量级相当，
但 KL 正则是惩罚 evidence 增长的——这限制了模型的判别力建设。

## 改动
- kl_reg: 0.1 → 0.01（降 10 倍）
- 其他完全不变

## 预期
- KL 惩罚更弱 → evidence 能更自由增长 → 分类梯度更强
- id_part 应比 exp155 收敛更快
- 如果 ep40-60 的 mAP 高于 exp155 → kl_reg 是瓶颈
