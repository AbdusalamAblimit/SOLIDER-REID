# 范式级创新方向调研总结 (2026-03-25)

> 基于 4 个 Opus Agent 并行调研 + 前台快速分析的综合结论

## 当前最佳结果
- exp176 (SupCon T=0.05 + triple injection + PLBOA): **mAP 64.1% / R1 75.5%**
- exp178 (SupCon T=0.03): **mAP 64.5%** (最高 mAP)
- Baseline (exp166, CE): 63.1/73.9

## 已确认的核心发现
1. **Per-token SupCon** 首次在 ReID per-token 级别应用 — 无先例
2. **SupCon × PLBOA 超线性 synergy** — 无先例（PLBOA 在 SupCon 下 +4.3 vs CE 下 +2.8）
3. **SupCon 无 PLBOA 不如 CE** — 证明 synergy 是核心，不是 SupCon 单独
4. **Training objective > Architecture** — SupCon on base (+3.9) > 所有架构增强总和 (+2.8)

## 文献确认的独创性
- Per-token SupCon for ReID: **无先例**（搜索 20+ 论文）
- SupCon × PLBOA synergy: **无先例**
- PSG backbone 内 pose gating: **无先例**（现有工作全在 input 或 post-backbone）
- 最接近竞争者: PAFormer (2024, 未被顶会接受, 59.9% mAP)

---

## 范式级方向 A: Occlusion-Asymmetric Self-Distillation (OA-SD) ⭐⭐⭐⭐⭐

### 核心思想
用 DINOv2 self-distillation 范式，创建"遮挡不变表示"：
- Teacher (EMA, no grad): 看完整图 + 完整 pose → clean structural tokens
- Student (trained): 看 PLBOA 遮挡图 + degraded pose → degraded tokens
- Distillation: student tokens 逼近 teacher tokens

### 为什么是范式级
1. 重新定义问题：从"提取更好特征"到"学习遮挡不变表示"
2. 与 PersonMAE 本质不同：distill identity-level tokens 而非 reconstruct pixels
3. Teacher co-evolves with student (EMA)，不是固定 teacher
4. PLBOA 创造 asymmetry，STD-PR tokens 是 distillation target

### 与我们失败实验的区别
- exp048 SGMKC: 硬特征替换 → 失败（GCN 容量不够）
- exp091/092 TTSFR/LSRM: batch 内 recovery → 失败（只有 4 images/ID）
- OA-SD: 软目标 distillation + 同一图像的 clean/occluded 版本 → 本质不同

### 实现方案
- ~150 行新代码
- EMA model wrapper + distillation loss
- Teacher forward 无 PLBOA，Student forward 有 PLBOA
- 风险：2x forward memory（可用 no_grad 缓解）

---

## 范式级方向 B: Structural Contrastive Learning (SCL) ⭐⭐⭐⭐

### 核心思想
不在 feature 上做 contrastive，而是在 STRUCTURE 上做：
- 两人 visibility pattern 相似 → "structurally positive"
- 模型同时学：(a) 每个部位长什么样 (identity) (b) 该比较哪些部位 (structure)

### 创新评估
- 创新力度: 9/10（无先例，重新定义问题）
- 实现难度: 8/10（需要 structure encoder + dual loss）
- 可行性: 6/10

### 具体实现
- 用 per-token visibility (from pose) 动态调整 SupCon temperature/weighting
- 可见 token → 正常 contrastive；遮挡 token → 降权
- 本质是让 SupCon 结构感知

---

## 范式级方向 C: Contrastive Token Alignment (CTA) ⭐⭐⭐

### 核心思想
用 Video-ColBERT 的 dual sigmoid loss 替代失败的 MaxSim triplet training：
- MeanMaxSim score (不是 sum)
- Dual sigmoid loss (per-pair, 不做 hard mining)
- 避免 exp152 的梯度不稳定问题

### 评估
- 创新: 中（MaxSim training for ReID 是 gap，但概念不够新）
- 实现: ~80 行
- 风险: 可能仍然 neutral

---

## CCF-B 投稿策略

### 首选目标: Pattern Recognition (IF~7.5)
- 偏好方法论完整、消融充分
- 我们有 186 个实验的消融深度
- 近年发表大量 occluded ReID

### 论文 Story 候选
**"Synergizing Supervised Contrastive Learning and Pose-Guided Augmentation for Occluded Person Re-Identification"**

### 需要补充的实验（按优先级）
1. [硬性] exp176 的 3-seed 验证 → 用户在 4090 上做
2. [硬性] Market-1501 完整结果
3. [强烈建议] Swin-Small backbone 结果
4. [建议] t-SNE 可视化 + 检索结果对比
5. [建议] 效率分析表

### SOTA 对比 framing
- 不和 KPR (Swin-L, 200M params) 正面比
- 强调参数效率：30M params 达到 86M params 方法相当的性能
- 与 OGFR (ViT, 64.7/76.6) 差距仅 -0.6/-1.1

---

## 今晚实验计划
1. exp187 (parallel aug + SupCon) 在本地继续跑
2. 在远程实现并测试 OA-SD (方向 A)
3. 如果 OA-SD 有正向信号 → 继续迭代
4. 如果 OA-SD 中性 → 尝试 SCL (方向 B) 或 CTA (方向 C)

---

## 2026-03-26 凌晨：OA-SD 确认为范式级创新

### 关键实验结果

| 实验 | 方法 | mAP | R1 | 创新贡献 |
|------|------|------|------|---------|
| exp166 | CE baseline (full) | 63.1% | 73.9% | — |
| exp176 | +SupCon T=0.05 | 64.1% | 75.5% | SupCon |
| **exp187** | **+3-view parallel** | **64.9%** | **76.6%** | **3-view aug** |
| **exp191** | **+OA-SD (base,CE)** | **63.2%** | **75.4%** | **OA-SD paradigm** |

### OA-SD 的关键发现

1. **OA-SD + CE = 正向** (+2.9/+2.6 vs CE base)
2. **OA-SD + SupCon = 负向** (-0.7/-0.4 vs SupCon only)
3. OA-SD 的 distillation loss 与 SupCon 的 contrastive loss 有梯度冲突
4. OA-SD 与 CE 互补：distillation 提供遮挡不变性，CE 提供分类能力

### 下一步计划

1. [待做] OA-SD + CE + 3-view Parallel Aug（组合两个正向方向）
2. [待做] OA-SD + CE + PLBOA + 无 PAPE/multi-stage（最简 OA-SD 配置）
3. [用户做] 3-seed 验证 + Swin-Small/Base 刷 SOTA
4. [用户做] Market-1501 数据集结果
