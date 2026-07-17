# 实验 exp153: MaxSim Triplet 补充模式（Additive, 不替换）

## 动机

exp152/152b 证明"替换 pooled triplet"是有害的（-3.3% mAP）。根本原因是 MaxSim triplet 与 pooled ID loss 梯度冲突。

但如果 MaxSim triplet 以低权重**补充**（而非替换）pooled triplet，可能避免冲突：
- Pooled triplet 仍是主度量学习信号（保护特征质量）
- MaxSim triplet 作为辅助信号，轻微推动 keypoint 独立判别力

## 技术方案

```
Loss = ID_global + ID_part + (pooled_triplet_global + pooled_triplet_part) + 0.25 * maxsim_triplet
```

关键区别：
- exp152: `part_triplet = maxsim_triplet`（替换）
- **exp153**: `part_triplet = pooled_triplet + 0.25 * maxsim_triplet`（补充）

### 实现

修改 loss/make_loss.py 中的 MaxSim triplet 集成方式：
- 当 `POSE_MAXSIM_TRIPLET=True` 且 `POSE_MAXSIM_TRIPLET_ADDITIVE=True` 时
- 先正常计算 pooled part triplet
- 再额外加 0.25 × maxsim triplet

## 对照组
- A: exp030a (baseline, equal_concat 60.73%)
- B: exp030a + maxsim test-only (62.2%)
- C: exp152b (hard MaxSim replace, 57.8% eq / 59.0% maxsim) — 已失败
- **D: exp153 (MaxSim additive) — 本实验**

## 预期结果
- equal_concat 应与 exp030a 持平或微正（不再有 -3% 退化）
- maxsim_hybrid test 可能微正（辅助 MaxSim 信号改善了 kp 判别力）
- 如果也失败，彻底放弃 MaxSim training

## 止损条件
- ep40 equal_concat mAP < exp030a ep40 1.0% → 止损
