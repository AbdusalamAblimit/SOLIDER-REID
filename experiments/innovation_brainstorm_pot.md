# Partial Optimal Transport for Occluded ReID — Innovation Analysis

## 核心创新 (调研 agent 推荐 #1)

**重新定义匹配距离**: Occluded ReID 本质上是 partial matching 问题 —— 
我们应该用一个天然处理 partial observations 的距离度量。

Partial Optimal Transport (POT) 的核心思想:
- 标准 OT: 要求把 query 的所有 mass 传输到 gallery 的所有 mass
- Partial OT: 只传输一部分 mass, 剩余的是 "unmatched" (occlusion!)
- 传输计划自动发现哪些部位可以匹配, 哪些被遮挡

## 为什么这是范式级创新

1. **问题重新定义**: ReID 不是 "相似度排序", 而是 "partial matching with transport"
2. **机制层面新**: 没有人用 partial OT 做 person ReID 的距离度量
3. **与 LGPA-D 完美互补**: LGPA-D 提供 per-part features, POT 在 part 层面做 optimal transport
4. **理论优美**: OT 是有坚实数学基础的框架, 不是 heuristic

## Prior Art 检查

- OT in computer vision: Sinkhorn for clustering (Cuturi 2013), Wasserstein GAN
- OT in metric learning: 有一些工作 (EMD), 但不在 ReID
- **Partial OT in ReID: 未找到任何论文!** 这是空白!
- 最接近: MoS (AAAI 2021) 做 set matching with Jaccard, 但不是 OT

## 技术方案

### Phase 1: Test-time POT (快速验证)
1. 提取 per-part features (LGPA-D 的 5 个 part features)
2. 将每个 part feature 视为 "mass" in a transport problem
3. 用 Sinkhorn algorithm 计算 partial OT distance
4. 用 POT distance 替代 cosine 做 ranking

### Phase 2: Training-time POT (如果 test-time 有效)
1. 用 partial OT distance 替代 triplet loss 中的 Euclidean distance
2. POT distance 可微分 (通过 Sinkhorn)
3. End-to-end 训练

## 与已有方法的关系

| 方法 | 距离度量 | 处理遮挡 |
|------|---------|---------|
| Cosine | holistic 相似度 | 不处理 |
| MaxSim | max per-part cosine | heuristic: 取最大匹配 |
| Equal Concat | 拼接后 cosine | 遮挡部位噪声 |
| **POT** | **partial optimal transport** | **理论最优的部分匹配** |

## 可行性评估

- Python POT 库已成熟 (pip install POT)
- Sinkhorn algorithm 可微分, GPU 加速
- 在 5 parts 上做 OT 计算量极小
- Test-time 验证无需训练, 几小时出结果

## 调研 Agent 评估
- Novelty: Very High
- Feasibility: Medium (需要理解 OT 数学)
- Paper Value: Very High

## 2026-04-08: 深度技术分析 (Opus 子代理)

### 核心结论: **POT 5/10, 值得快速验证但不能作为论文主线**

### 关键风险:
1. **5×5 OT 太小**: 5 个 LGPA parts 的 transport plan 几乎没有自由度 (5+5-1=9 constraints on 25 variables)。Partial OT 在如此小的矩阵上几乎退化为"丢弃最低可见性 part + cosine"
2. **训练集 95.8% 可见**: OT with uniform mass ≈ standard distance。只有 PLBOA(0.7) 创造的合成遮挡才让 OT 有意义
3. **可能无法超越 MaxSim**: MaxSim 已经隐式处理了 part-level matching + visibility

### 可能成立的场景:
- PLBOA 创造足够多样的 partial visibility patterns
- 17 keypoints (而非 5 parts) 让 OT 有更多自由度
- Transport plan 的可解释性有论文价值

### 建议:
- **快速测试 Phase 1** (test-time POT, 无训练需求): 2-3 小时
- 如果 POT > MaxSim +0.5%: 值得作为 secondary contribution
- 如果 POT ≈ MaxSim: 仅作为理论分析/消融
- **不要以 POT 为论文主线**, novelty 不够 CCF-B

### 备选更强方向:
1. 17-keypoint OT (需要更好的 per-keypoint features)
2. Cross-Instance Completion via OT (最新颖但最有风险)
3. 完全不同的方向: 需要跳出当前框架

## 2026-04-08: POT Test-time 实验结果 (exp246b Tiny LGPA-D+GCN)

| Method | mAP | R1 | Δ vs Global |
|--------|-----|----|-------------|
| Global cosine | 65.2 | 76.2 | — |
| Vis-weighted part | 65.7 | 77.5 | +0.5/+1.3 |
| POT m=0.6 | **66.4** | **78.7** | +1.2/+2.5 |
| POT m=0.8 | 66.1 | 77.7 | +1.0/+1.5 |
| POT m=auto | 66.0 | 77.6 | +0.8/+1.4 |
| POT m=1.0 | 66.0 | 77.6 | +0.8/+1.4 |
| MaxSim hybrid | **66.6** | 78.3 | +1.4/+2.1 |

**结论**: 
- POT m=0.6 最佳: mAP 66.4 (vs MaxSim 66.6 = -0.2), R1 78.7 (vs MaxSim 78.3 = **+0.4**)
- POT 在 R1 上超越 MaxSim，在 mAP 上略逊
- 验证了 agent 预测: 5-part POT ≈ MaxSim, 差异不够论文主线
- **POT 可作为消融实验/理论分析保留，不作为主创新**
- Vis-weighted cosine (65.7/77.5) 也是一个有意义的简单 baseline

## exp245h_v2 (Small LGPA-D, 最强 checkpoint) POT 结果

| Method | mAP | R1 | Δ vs Global |
|--------|-----|----|-------------|
| Global cosine | 71.8 | 81.1 | — |
| Vis-weighted part | 71.9 | 82.2 | +0.1/+1.1 |
| **POT m=0.6** | **73.0** | 83.1 | **+1.2/+2.0** |
| POT m=0.8 | 72.7 | 82.4 | +0.9/+1.3 |
| **MaxSim hybrid** | 72.8 | **83.7** | +1.0/+2.6 |
| MaxSim+POT 0.3 | 72.5 | 83.3 | -0.3 vs MaxSim |

**重要发现: POT m=0.6 mAP 73.0 > MaxSim mAP 72.8!**
- POT 在 mAP (整体排序质量) 上超越 MaxSim (+0.2)
- MaxSim 在 R1 (top-1 精度) 上超越 POT (+0.6)
- 两者互补: POT 更好排序, MaxSim 更好找 top-1
- MaxSim+POT 组合信号冲突, 反而不如单独的 MaxSim
- **在 Small backbone 上 POT 优势更明显** (Tiny: -0.2 mAP, Small: +0.2 mAP)

**修正评估**: POT 从 5/10 提升到 **6/10** — 在 Small 上确有 mAP 优势，
可作为论文的 complementary matching analysis (POT vs MaxSim 各有优势)
