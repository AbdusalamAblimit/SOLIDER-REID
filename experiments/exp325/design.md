# 实验 exp325: 冻结 DINOv2-large + 轻量头 + 姿态部位匹配（天花板探测）

> **来源**：过夜创新探索。exp324b 证冻结 DINOv2-**base** 训轻量头 part-MaxSim 重遮挡 8.65 / 全部 14.61，但 e20 即到顶 → 冻结 base 特征天花板低。本实验换**更强冻结 backbone（DINOv2-large，hidden 1024，patch 14）**，问"**更强冻结模型重遮挡 mAP 能否抬过 exp324b 的 8.65 / 全部 14.61？**"。
> **性质**：**训练实验**（训轻量头）。开训前必须过 Claude broad review + Codex review（hook 阻断）。
> **机器**：lab-3090-d（完整 env + 3090 idle；原计划 hyy，但本 cycle 用 lab-3090-d 统一跑，env 现成）。需 `HF_ENDPOINT=https://hf-mirror.com` 下 `facebook/dinov2-large`。

## 动机

- exp324b 唯一变量是 backbone 容量/表征强度。DINOv2-large（300M vs base 86M，1024 vs 768 dim）在密集对应/分割等下游显著强于 base。
- 若 large 把冻结天花板抬高（重遮挡 8.65 → 显著更高），说明"冻结对应特征"这条线还有空间，值得继续（再上 v3 / LoRA 解冻）；若几乎不动，说明瓶颈不在 backbone 容量而在"冻结 + 轻量头"范式本身 → 这条线天花板低，止损。

## 核心假设

把 exp324b 的冻结 backbone 从 DINOv2-base 换成 DINOv2-large（其余 pipeline、损失、超参、采样、评测**完全不变**，单变量），part-MaxSim 重遮挡 mAP > 8.65、全部 > 14.61。

## 技术方案

- **唯一改动**：backbone `facebook/dinov2-base` → `facebook/dinov2-large`；hidden 768 → 1024；patch 16→14（DINOv2 都是 patch14，base 也是 patch14 —— 确认 exp324_dino 的 GRID 由 patch 推导，large 同 patch14，仅 hidden 变）。投影头输入维 1024→512（exp324b 是 768→512）。
- **复用 exp324b 脚本**：新增 `scripts/exp325_train_head.py`，import exp324b/exp324_dino 的损失/采样/eval/几何，仅覆盖：模型加载（large）、HIDDEN=1024、缓存目录（`experiments/exp325/_cache`，**独立缓存**，因部位特征维度/分布不同，不可复用 exp324b 缓存）、preprocess 的 grid 尺寸（按 large 的 patch/输入推导）。
- **冻结边界**：DINOv2-large 不反传，只训投影头 1024→512 + BNNeck + 全局 ID 分类头 + per-part 共享 ID 分类头（与 exp324b 同结构同 `part_weight=0.5`）。
- **超参**：与 exp324b **完全一致**（epoch 60、P16K4=BS64、Adam lr3.5e-4、wd5e-4、cosine、id_w/tri_w/part_w=1/1/0.5、margin soft、seed 1234、eval_period 10）——单变量隔离 backbone。
- **评测**：part-MaxSim + cos，全量 + 重遮挡（vis≤8），同 exp324b 口径。
- **风险（large 缓存）**：query 2210 + gallery 17661 + train 15618 张过 large dense token 抽部位特征，1024d × 5 part。缓存体积约 base 的 1.33×（521M→~700M），3090 716G 盘宽裕。抽取耗时 large 比 base 慢 ~2-3×，可接受（一次性缓存）。

## 预期结果

- 假设成立：重遮挡 > 8.65、全部 > 14.61（哪怕 +1~+3 即正向，说明 backbone 容量是有效杠杆 → 继续 v3/LoRA）。
- 失败：与 base 持平甚至略低（large 密集 token 未必更 ReID-判别）→ 瓶颈在范式不在容量，冻结线止损。

## 对照组

- **baseline**：exp324b（DINOv2-base，重遮挡 8.65 / 全部 14.61，cos 全部 13.51 / 重遮挡 7.32）。
- **单变量**：仅 backbone（base→large），其余全冻。
