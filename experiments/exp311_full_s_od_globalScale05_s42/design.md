# 实验 exp311: Small OD Full Scaffold + 真生效的 0.5× global loss scale

## 动机

`MODEL.GLOBAL_LOSS_SCALE = 0.5` 在 `prcv_best_*.yml` 配置中已经设了很久, 但 `loss/make_loss.py` 只在 **no-part 分支** (line 218/258) 应用它。Full Scaffold (有 part tokens) 走 line 213/253 的对称 `w_g * global + w_p * part`, GLOBAL_LOSS_SCALE 从未生效。

修复: 在 part-path 也乘上 `global_loss_scale`, 让 0.5 在 ID 和 Triplet 的 global 项实际生效。

## 核心假设

降低 global ID/Tri loss 权重到原来 1/2 (0.5 → 0.25 part-vs-global net split), 让 part tokens 学得更深, 可能提升 MaxSim mAP (MaxSim 主要靠 part 特征 late interaction)。

## 技术方案

### 修改文件
`loss/make_loss.py`:
- Line 213 (旧): `ID_LOSS = w_g * global_id + w_p * part_id_avg`
- Line 213 (新): `global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0); ID_LOSS = global_loss_scale * w_g * global_id + w_p * part_id_avg`
- Line 253 同样改, 应用于 TRI_LOSS

### 数据流
- `feat = [global, str_1..K, gcn]` (Full Scaffold)
- ID_LOSS = **0.5** * 0.5 * global_id + 0.5 * part_id_avg = **0.25** global + 0.5 part
- TRI_LOSS 同上
- Total = 1.0 * ID_LOSS + 1.0 * TRI_LOSS (cfg.MODEL.ID_LOSS_WEIGHT/TRIPLET_LOSS_WEIGHT 都 1.0)

### 关键超参
- `GLOBAL_LOSS_SCALE = 0.5` (来自 prcv_best_small.yml, 现在真生效)
- 其他全 default

## 预期结果

- 假设成立: MaxSim mAP +0.3-0.8 vs exp285b (74.7 → 75.0-75.5)
- 失败可能: global 监督不够, ID acc 收敛慢, e120 mAP 反而 -0.5

## 对照组

- **baseline**: exp285b Small s42 默认 (w_g=w_p=0.5, GLOBAL_LOSS_SCALE 实际未生效) → 73.8/83.8 eq, **74.7/84.8 MaxSim**
- 消融变量: 仅 global_loss_scale=0.5 (在 part-path 真生效)

## 兼容性

- `GLOBAL_LOSS_SCALE` 默认 1.0 (config/defaults.py), 老配置 (无显式设置) 不受影响
- 只有 prcv_best_*.yml (已设 0.5) 受影响 — 但这是用户预期的修复
