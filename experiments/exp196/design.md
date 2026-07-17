# 实验 exp196: 3-view + SupCon + OA-SD Global-Only（终极配置）

## 动机
- exp187 (3-view + SupCon): 64.9/76.6 — 当前整体最佳
- exp193 (3-view + OA-SD + CE): 64.4/76.5 — 3-view+OA-SD+CE 最佳
- exp195 (SupCon + OA-SD global-only): 验证了 global-only 解决梯度冲突
- **问题**: 能否把三大创新（3-view + SupCon + OA-SD）合为一体？
- exp188 证明 all-token OA-SD + SupCon 失败（梯度冲突）
- exp195 证明 global-only OA-SD + SupCon 成功（职责分离）
- **本实验验证**: 在 3-view 环境下，SupCon + OA-SD global-only 是否 additive

## 核心假设
通过 OA-SD global-only distillation 实现"职责分离"——global feature 学遮挡不变性（OA-SD），per-token features 学判别力（SupCon）——在 3-view parallel augmentation 下三者增益 additive。

## 技术方案
- 配置 = exp187 (3-view + SupCon) + OA-SD global-only
- 即: `POSE_PARALLEL_AUG=True + POSE_STR_SUPCON=True + POSE_OA_SD=True + POSE_OA_SD_GLOBAL_ONLY=True`
- 使用 full config (有 PAPE, multi-stage PSG)
- 代码已在 exp193/exp195 中实现完毕，无新代码修改
- 4-view tuple: 3 student views + 1 teacher clean view
- OA-SD distillation 仅在 `feat[0]` (global) 上计算

### 数据流
1. Dataset: PLBOA → 3-view augmentation → 4-view tuple (3 student + 1 teacher)
2. Forward: 3 student views 顺序 forward + 1 teacher forward (no_grad)
3. Loss:
   - 3 × (CE_global + SupCon_part + triplet) → averaged
   - 1 × OA-SD cosine distillation on global feat → added after averaging
4. Backward: OA-SD 梯度只流过 global feature，不影响 per-token SupCon

### GPU 显存估计
- exp193 (3-view + OA-SD all-token): 20.9GB — OK
- exp196 多了 SupCon loss 计算（但 SupCon 不增加 forward 开销）
- 预计 ~21-22GB，应在 24GB 内

## 预期结果
- 假设成立: mAP 65.0-65.5%, R1 77.0-77.5% (超过 exp187!)
- 如果中性: ~64.9/76.6 (= exp187)，说明 OA-SD global 在 SupCon 下冗余
- 如果失败: < 64.5，说明 3-view 下 OA-SD + SupCon 仍有某种干扰

## 对照组
- exp187 (3-view + SupCon, no OA-SD): 64.9/76.6 — 主对照
- exp193 (3-view + OA-SD + CE): 64.4/76.5
- exp195 (SupCon + OA-SD global-only, 1-view): 进行中
