# 实验 exp340b: 固定 band LGPA 变体 — un-detach + 低 global loss

## 动机（用户假说）
- exp340a 是 **detached**(`POSE_LGPA_DETACH=True`)：LGPA part 分支的梯度**不回传 backbone**，只有 global loss 训 backbone。
- 用户洞察：**global 训 backbone(梯度回传)、part 不训(detach)→ backbone 被塑造成 global 导向 → part 分支只能在"为 global 优化过的特征"上池化，可能被拖累。**
- 历史上 detach +4.4 是在 **full global loss(0.5)** 下测的（detach 防 part 扰动 global 主线）。本变体**翻转 regime**：让 part loss 主导 backbone 塑形。

## 核心假设
**un-detach（part 梯度回传 backbone）+ 砍低 global loss → backbone 被 part loss 塑造成部位判别 → 固定 band part 分支 standalone（或 equal_concat）超 global，比 exp340a 更强。**

## 技术方案（无新代码，仅 config flag）
- = exp340a config，**仅改两项**：
  - `POSE_LGPA_DETACH: False`（part 梯度回传 backbone）
  - `GLOBAL_LOSS_SCALE: 0.1`（global 不再主导；保留小锚防 backbone 塌）
- 代码与 exp340a 完全相同（canonical 固定姿态 + LGPA），已双审查通过；本变体不引入新代码。

## 预期结果
- 理想：part_only 或 equal_concat **> global(59.0)**，且 **> exp340a**（un-detach 让 backbone 更适配 part）。
- 成功判据（采纳用户）：part_only > global **或** equal_concat > global = 固定语义涨点。
- 失败最可能原因：① un-detach 仍像历史那样扰动/退化（但 global 已降权，regime 不同）；② global 0.1 太低，backbone ID 监督不足 → 若塌，回调 global 0.2–0.3。

## 对照组
- **exp340a**（detached + global 0.5，4090 跑）vs **exp340b**（un-detach + global 0.1，3090 跑）。
- global baseline 59.0。
- 单变量：exp340b vs exp340a 仅差 DETACH(False) + GLOBAL_LOSS_SCALE(0.1)。

## 审查说明
代码与 exp340a 同（`experiments/exp340_swin_lgpa_fixedbands/` 已 Claude+Codex 双审通过）。本变体仅切换两个已存在、已审查的 config flag（`POSE_LGPA_DETACH`、`GLOBAL_LOSS_SCALE`），无新代码路径。下方 review 复核「config 变体是否安全 + 单变量隔离」。
