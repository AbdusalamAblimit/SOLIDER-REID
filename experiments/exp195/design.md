# 实验 exp195: OA-SD Global-Only + SupCon

## 动机
- exp188 (OA-SD + SupCon, all-token distillation) = 负向 (-0.7/-0.4 vs SupCon only)
- 根本原因：per-token distillation 和 per-token SupCon 在同一特征上产生梯度冲突
  - SupCon 鼓励：同 ID 的 token 互相拉近，不同 ID 的推远
  - OA-SD distillation 鼓励：student token 逼近 teacher token（不管 ID）
  - 两者在 token 级别方向矛盾
- 假设：如果 OA-SD 只在 global feature（GAP 后）做 distillation，不碰 per-token 特征：
  - SupCon 专门负责 per-token 的判别力学习
  - OA-SD 专门负责 global 的遮挡不变性学习
  - 两者职责分离，不冲突

## 核心假设
OA-SD 与 SupCon 的梯度冲突来自 per-token 级别的 distillation。将 distillation 限制在 global feature 可以消除冲突，同时保留 OA-SD 的遮挡不变性学习信号。

## 技术方案
- 在 processor.py 中增加 `POSE_OA_SD_GLOBAL_ONLY` 配置开关
- 当 `GLOBAL_ONLY=True` 时，OA-SD distillation 只在 `feat[0]`（global pooled feature）上计算
- per-token features (`feat[1:]`) 完全由 SupCon + CE 负责
- EMA teacher、PLBOA asymmetry、decay 等机制不变

### 修改文件
1. `config/defaults.py`: 添加 `POSE_OA_SD_GLOBAL_ONLY = False`
2. `processor/processor.py`: 在 OA-SD distillation 逻辑中增加 global-only 分支

### 梯度流分析
- `feat[0]` (global): CE_global + triplet_global + OA-SD_distill → 三个梯度合力
- `feat[1:]` (tokens): CE_part + triplet_part + SupCon → 三个梯度合力（无 OA-SD 干扰）
- 关键：SupCon 和 OA-SD 不再在同一个特征上竞争

## 预期结果
- 假设成立: mAP 64.5-65.0% (SupCon 贡献 ~+1.0 + OA-SD global 贡献 ~+0.5-1.0)
- 如果失败: global-only distillation 信号太弱，不如 all-token
- 次优情况: 中性（与 SupCon only 持平），说明 OA-SD 的价值在 per-token 级别

## 对照组
- exp176 (SupCon T=0.05, no OA-SD): 64.1/75.5 — 主对照
- exp188 (OA-SD + SupCon, all-token): ~63.4/75.1 — 失败的组合
- exp191 (OA-SD + CE, all-token): 63.2/75.4 — OA-SD 在 CE 下的效果

## 远程服务器配置
- 1-view (无 parallel_aug, 16GB 限制)
- SupCon T=0.05
- OA-SD GLOBAL_ONLY=True, decay=0.999, weight=1.0
- PLBOA enabled
