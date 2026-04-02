# exp213 Claude Review: PKC(0.05) + MST(0.1) Combined

## 审查范围

a. `experiments/exp213/design.md` — 合理性、单变量原则、假设
b. `processor/processor.py` (PKC: lines 831-868, MST: lines 870-935) — 逐行审查
c. `config/defaults.py` (POSE_PKC lines 194-198, POSE_MST lines 188-192) — 默认值安全性
d. `loss/supcon_loss.py` — PKC 依赖的 SupConLoss
e. `loss/make_loss.py` — CE + triplet 对 GCN 参数的梯度流
f. `model/modules/skeleton_gcn.py` — kp_feats / kp_weights 产生路径
g. 与 exp206r, exp210b, exp211 的对照

## 审查结果

### 1. Design.md 合理性 — OK (带注意事项)

- 动机合理: PKC 和 MST 是正交优化信号 (SupCon 推全局分布 vs triplet 推 pair-wise MaxSim 距离)
- **注意: 这是组合实验** (PKC + MST)，不是单变量。但两者已分别在 exp210b 和 exp211 中验证，且 design.md 明确了对照组 (exp206r, exp210b, exp211)，属于合理的消融组合。
- PKC=0.05 和 MST=0.1 都是低权重，总额外 loss 贡献约为 (0.05 * ~3.0 + 0.1 * ~0.3) = ~0.18，相对 CE+triplet 总 loss ~10 仍很低。

### 2. PKC 代码审查 (lines 831-868) — OK

- **Lazy init** (line 843): `SupConLoss(temperature=pkc_temp)` 正确缓存在 `do_train._pkc_supcon`
- **Per-keypoint 循环** (lines 848-860): 对 17 个 keypoint 分别做 SupCon，仅选可见样本 (vis > 0.3)
- **最少样本检查** (line 852): n_vis < 4 跳过，(line 857) 至少 2 个不同 ID — 正确防御
- **特征提取** (line 854): `kp_f[vis_mask, k_idx, :]` — 正确索引 (n_vis, C)
- **Loss 聚合** (line 863): 平均 active keypoints 的 loss — 合理
- **_loss_details 传递** (lines 864-868): 正确使用 getattr 获取已有 details 再追加

**PKC 不修改 kp_data['kp_feats']** — 仅读取。

### 3. MST 代码审查 (lines 870-935) — OK (1 Low)

- **L2 normalize** (line 882): `F.normalize(kp_f, p=2, dim=2)` 创建新张量 `kp_fn`，不修改原始 kp_f
- **分块 einsum** (lines 894-908): chunk=32, 避免 OOM, 对 B=64 分 2 块
- **MaxSim 距离计算**: 与 test-time `_maxsim_distance` 公式一致 (query-side visibility weighting, 1-sim)
- **Hard triplet mining** (lines 912-926): 标准实现, pos_mask/neg_mask 正确
- **has_pos guard** (line 929): 正确防御无正样本情况
- **_loss_details 传递** (lines 932-935): 正确

**Low: 死代码** — line 891 `w_sum` 计算后未使用 (循环内重新计算 `w_s`)。无害。

**MST 不修改 kp_data['kp_feats']** — 仅读取。

### 4. PKC + MST 共存分析 — OK (无冲突)

**执行顺序**: PKC (lines 831-868) 先执行, MST (lines 870-935) 后执行。

**数据独立性**:
- 两者都从 `kp_data['kp_feats']` 读取同一个张量 (B, 17, C)
- 两者都不修改该张量 (PKC 做 indexing, MST 做 F.normalize 创建新张量)
- 无数据竞争

**Loss 累加正确性**:
- PKC: `loss = loss + pkc_weight * pkc_loss` (line 865)
- MST: `loss = loss + mst_weight * mst_loss` (line 933)
- 两者都用 `getattr(loss, '_loss_details', {})` 获取已有 details
- PKC 先执行时设置 `loss._loss_details`; MST 后执行时通过 `getattr` 获取包含 pkc key 的 dict 并追加 mst key
- **结论: 两个 loss 正确累加, details dict 正确保留两者的 key**

### 5. 梯度流分析 — OK (关键问题)

**GCN 参数收到的梯度来源** (当 PKC + MST 同时启用):
1. **CE loss** (via `score[1:]`): skeleton_feat → weighted pool of kp_feats_enhanced → GCN params
2. **Triplet loss** (via `feat[1:]`): 同上路径
3. **PKC** (SupCon): 直接在 kp_feats (B, 17, C) 上做 per-keypoint SupCon → GCN params
4. **MST** (MaxSim triplet): 直接在 kp_feats (B, 17, C) 上做 set-to-set triplet → GCN params
5. **OA-SD** (self-distillation): 通过 feat list → GCN params

**冲突风险评估**:
- CE 和 triplet 优化的是 **pooled** skeleton 特征的判别力
- PKC 优化的是 **per-keypoint** 特征的 SupCon 分布
- MST 优化的是 **per-keypoint** 特征在 MaxSim 距离度量下的判别力
- PKC 和 MST 虽然目标不同，但方向一致: 都让同 ID 的 keypoint features 更近，不同 ID 更远
- exp210 (PKC=0.5) 灾难性失败说明高权重 SupCon 会干扰 CE 收敛
- **PKC=0.05 已验证安全** (exp210b: mAP 不变, R1 -0.8)
- MST=0.1 的梯度贡献: margin loss 已收敛后 mst_loss ~0.1-0.3, weight=0.1, 贡献 ~0.01-0.03 — 极小
- **总额外梯度非常小，不太可能干扰 CE 收敛**

### 6. AMP 安全性 — OK

- 两个 loss 均在 `amp.autocast(enabled=True)` 块 (line 472) 内
- PKC: indexing + SupConLoss (matmul, exp, log, mean) — 全 AMP 安全
- MST: F.normalize, einsum, max, F.relu, mean — 全 AMP 安全
- 无 in-place 操作

### 7. OOM 风险 — 安全

- PKC 额外: 每 keypoint 做 (n_vis, n_vis) 相似度矩阵, n_vis <= 64, 最大 64*64*17 = ~280K floats = ~1MB
- MST 额外: 分块 (32, 17, 64, 17) = ~0.6M floats * fp16 = ~1.2MB
- 两者合计 < 5MB, 3090 24GB 完全安全

### 8. Config 默认值安全性 — OK

- POSE_PKC=False, POSE_MST=False (默认关闭)
- 不影响任何已有实验
- exp213 通过 CLI override 启用两者

### 9. POSE_MAXSIM_TRIPLET vs POSE_MST 潜在冲突 — OK

- `POSE_MAXSIM_TRIPLET` (make_loss.py line 224-240): 旧版 MaxSim triplet, 在 loss_fn 内部
- `POSE_MST` (processor.py line 870-935): 新版 MaxSim triplet, 在 processor 中
- 两者默认 False, exp213 只启用 POSE_MST
- 如果两者同时启用会导致双重 MaxSim triplet loss — 但 exp213 不会触发此情况

### 10. 与前序实验的对照 — OK

| 实验 | PKC | MST | 结果 |
|------|-----|-----|------|
| exp206r | - | - | 70.6/82.6 (eq), 72.3/82.9 (maxsim) |
| exp210b | 0.05 | - | 70.6/81.8 (eq), 72.4/83.1 (maxsim) |
| exp211 | - | 0.1? | TBD (仍在运行) |
| **exp213** | **0.05** | **0.1** | **预测: ~72.4/83 (maxsim)** |

**注意**: design.md 中 MST weight=0.1, 但 config 默认 POSE_MST_WEIGHT=0.5。确认 exp213 必须通过 CLI 显式设置 `MODEL.POSE_MST_WEIGHT 0.1`。

### 11. 实验价值评估

- PKC +0.1 mAP maxsim, MST 未知 → 两者组合的增量可能很小
- 但低成本 (无新代码, 只加 CLI 参数), 合理的消融组合
- 如果 PKC+MST > max(PKC, MST), 说明两者互补, 有论文价值

## 需确认的启动命令

exp213 启动时必须确保以下 CLI overrides (在 exp206r 配置基础上):
```
MODEL.POSE_PKC True
MODEL.POSE_PKC_WEIGHT 0.05
MODEL.POSE_MST True
MODEL.POSE_MST_WEIGHT 0.1
```

## 总结

| 级别 | 数量 | 详情 |
|------|------|------|
| Critical | 0 | |
| High | 0 | |
| Medium | 0 | |
| Low | 1 | 死代码 w_sum (MST line 891) 未使用 |

## 审查通过

PKC 和 MST 共存无数据竞争、无 loss 累加错误、无梯度冲突风险 (两者权重极低)。两者都只读取 kp_data['kp_feats'] 而不修改。_loss_details dict 正确传递两者的 key。AMP 安全、OOM 安全。可以启动训练。
