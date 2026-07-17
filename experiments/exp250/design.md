# exp250: Partial Optimal Transport Matching (POT)

## 动机

当前最强 test-time 匹配方法 MaxSim 是 heuristic（对每个 query part 取 gallery 最大相似度）。
Occluded ReID 本质上是 **partial-to-partial matching** 问题：
- query 被遮挡 → 部分 part 特征是噪声
- gallery 也可能被遮挡 → 部分 part 特征是噪声
- 需要找到"最佳部分匹配" —— 这正是 Partial Optimal Transport (POT) 的数学定义

**Prior art 检查**: 搜索文献未找到任何用 Partial OT 做 Person ReID 距离度量的论文。
最接近的工作 MoS (AAAI 2021) 用 Jaccard set matching，不是 OT。

## 核心假设

Partial OT 距离替代 cosine / MaxSim，能更准确地处理遮挡下的部分匹配，
因为 OT 数学框架天然建模了"未匹配 mass = 遮挡"。

## 技术方案

### Phase 1: Test-time POT (无需训练，快速验证)

1. 用现有 LGPA-D 模型提取 5 个语义 part features + visibility weights
2. 对每个 query-gallery pair：
   - Cost matrix: C[i,j] = 1 - cos(q_part_i, g_part_j)，5×5
   - Mass: a[i] = vis_q[i] / sum(vis_q), b[j] = vis_g[j] / sum(vis_g)
   - Partial mass fraction m: 用 min(sum_vis_q, sum_vis_g) / max(...)
   - 用 Sinkhorn 算法计算 partial Wasserstein distance
3. Hybrid: α * global_cosine + (1-α) * POT_distance
4. 效率：先用 global cosine 排序，只对 top-50 重算 POT

对照：
- Cosine on equal_concat (baseline)
- MaxSim hybrid (当前最佳)
- Visibility-weighted cosine (简单 baseline)
- POT hybrid (新方法)

### Phase 2: Training-time POT (如果 test-time 有效)

**核心创新: OT-Triplet Loss**

替换标准 triplet loss 中的 Euclidean distance 为 differentiable Sinkhorn distance：
```
L_OT-tri = [d_POT(anchor, positive) - d_POT(anchor, negative) + margin]+
```

Sinkhorn distance 可微分 (通过 unrolled Sinkhorn iterations)。

技术细节：
- 用 PyTorch 实现 differentiable Sinkhorn
- 只在 part features 上计算 OT，不影响 global branch
- 可在 detached features 上操作（与 LGPA-D 一致）

### Phase 3: Transport Plan 可视化与分析

OT 的 transport plan T[i,j] 揭示哪些 query parts 匹配到哪些 gallery parts。
这提供了：
1. 可解释的匹配决策
2. 遮挡检测（未匹配的 mass = 遮挡部位）
3. 论文可视化素材

## 代码修改

### Phase 1 (test-time):
- `scripts/eval_pot.py` — 已完成重写，支持 LGPA features

### Phase 2 (training-time, 如果需要):
- `model/modules/sinkhorn_distance.py` — 新文件：differentiable Sinkhorn
- `processor/processor.py` — 添加 OT-triplet loss
- `config/defaults.py` — 添加 OT 相关配置

## 预期结果

Phase 1:
- 成功: POT hybrid > MaxSim hybrid (+0.5% 以上)
- 中性: POT ≈ MaxSim（理论贡献仍有价值）
- 失败: POT < MaxSim (5 parts 太粗糙, OT 在小矩阵上不如 heuristic)

Phase 2:
- 成功: OT-triplet 提升 train-time +0.5% 以上
- 风险: Sinkhorn backward 数值不稳定

## 对照组

- exp244 (Tiny LGPA-D only): 65.3/75.7, MaxSim 66.0/76.4
- exp246b (Tiny LGPA-D+GCN): 65.5/77.2, MaxSim 66.3/77.7
- exp245h_v2 (Small LGPA-D): 71.6/81.6, MaxSim 73.0/82.7

## 论文价值

如果 POT + LGPA-D 形成完整方法：
1. **LGPA-D**: Language-Grounded Part Assembly (CLIP 语义 part assignment)
2. **POT matching**: Partial OT 距离做 partial-to-partial matching
3. 核心叙事: "occluded ReID = partial matching of language-grounded body parts"
4. OT 框架提供理论基础 + 可解释 transport plan
5. 联合 novelty: 7-8/10，可冲 CCF-B

## 执行计划

1. **先跑 Phase 1**：在 exp246b (Tiny) 和 exp245h_v2 (Small) 上测试 POT
   - 需要 GPU 但不需要训练
   - 用 `scripts/eval_pot.py --config_file ... --weight ...`
   - 预计 20-30 分钟 per checkpoint
2. 根据 Phase 1 结果决定是否进 Phase 2
3. Phase 2 需要大量代码修改 — 这正是 CLAUDE.md 说"不要逃避大改动"的情况
