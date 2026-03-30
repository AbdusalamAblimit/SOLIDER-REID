# 实验 exp199: Occlusion-Asymmetric Relational Distillation (OA-RD)

## 动机
- OA-SD (per-token feature distillation) 与 SupCon 互斥 — exp188/196 已证明
- 根本原因：OA-SD 在 token 级别 match 个体特征，与 SupCon 在 token 级别做对比学习梯度冲突
- **核心洞察**：如果不 distill 个体特征，而是 distill **样本间的关系结构**，就不会与 SupCon 冲突
- 来源：RKD (Relational Knowledge Distillation, Park et al., CVPR 2019) + CRD (Contrastive Representation Distillation, Tian et al., ICLR 2020)

## 核心假设
将 self-distillation 从 "feature-level matching" (OA-SD) 升级为 "relation-level matching" (OA-RD)，可以在不与 SupCon 冲突的情况下获得 occlusion invariance。

## 技术方案

### 核心机制
```
Teacher (EMA, no grad):
  clean image → backbone → global_feat_t → pairwise_sim_t[i,j] = cos(feat_t[i], feat_t[j])

Student (trained):
  occluded image → backbone → global_feat_s → pairwise_sim_s[i,j] = cos(feat_s[i], feat_s[j])

Relational Distillation Loss:
  L_rd = KL(softmax(pairwise_sim_t / τ) || softmax(pairwise_sim_s / τ))
```

### 为什么与 SupCon 不冲突
- SupCon 作用于：individual token features (feat[1:]) — 推拉 token 在 embedding space 中的位置
- OA-RD 作用于：pairwise similarity matrix of global features — 保持 batch 内样本间的关系结构
- 两者操作在不同的"对象"上（SupCon → individual tokens, OA-RD → pairwise relations）
- OA-RD 的梯度通过 similarity matrix 传播，不直接约束 individual feature 的方向

### 实现细节
1. EMA teacher 机制复用 OA-SD 的代码（PLBOA asymmetry、decay、EMA update）
2. 新增 `_compute_relational_distillation()` 函数
3. 使用 global feature (feat[0]) 计算 pairwise similarity
4. KL divergence on row-normalized softmax
5. Temperature τ = 0.1（比 SupCon 高，因为 relation matching 需要更平滑的分布）

### 修改文件
1. `config/defaults.py`: `POSE_OA_RD = False`, `POSE_OA_RD_TEMP = 0.1`, `POSE_OA_RD_WEIGHT = 1.0`
2. `processor/processor.py`: 在 OA-SD 代码块旁边新增 OA-RD 逻辑

### 梯度流分析
- Teacher: no_grad, 不参与优化
- Student: L_rd 的梯度 → pairwise_sim_s → cos(feat_s[i], feat_s[j]) → feat_s → model
- 关键：梯度不 "match" 个体 feature 的方向，而是 match 它们之间的 "距离关系"
- 这与 SupCon 不冲突：SupCon 优化 token-level similarity，OA-RD 优化 batch-level similarity pattern

## 预期结果
- 假设成立: OA-RD + SupCon + 3-view > exp187 (64.9/76.6)，即 65.0-65.5/77.0+
- 如果中性: = exp187，说明 relational distillation 信号太弱
- 如果失败: < exp187，说明 relation-level 同样与 SupCon 有某种冲突

## 对照组
- exp187 (3-view + SupCon, no distillation): 64.9/76.6
- exp196 (3-view + SupCon + OA-SD global-only): 62.4/75.2 (失败)
- exp193 (3-view + OA-SD + CE): 64.4/76.5

## 创新门槛评估
1. ✅ 问题层面：从 "feature-level distillation" 到 "relation-level distillation" — 重新定义 self-distillation 的目标
2. ✅ 机制层面：RKD 在 occluded ReID 的 EMA self-distillation 中无先例
3. ✅ 证据层面：OA-SD 失败 (exp188/196) → OA-RD 成功 = 清晰的对照链
