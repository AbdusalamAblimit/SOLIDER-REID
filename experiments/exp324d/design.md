# 实验 exp324d: LoRA 解冻 DINOv2-base + 姿态部位匹配（破 14-mAP 天花板）

> **来源**：exp324b（冻结 DINO + 433K 轻量头）e20 即到顶：part 重遮挡 8.65 / 全部 14.61 mAP。机制对（姿态部位匹配 > 整图、可训 ×4.6），**但冻结特征天花板低**（vs exp255 Swin 75）。
> **性质**：训练实验。开训前必须过 Claude broad review + Codex review（hook 阻断）。
> **机器**：lab-3090-d（DINOv2-base 已下、RTX 3090 24G idle、peft 0.19.1 已装）。

## 动机

- exp324b 证明：把冻结 DINO 部位特征投到 ReID 空间能从 1.86 → 14.61（×7.9），机制（姿态锚定 + 只比可见部位 part-MaxSim）干净有效。
- **但天花板是冻结特征本身**：DINO 自监督预训练没见过 ReID 判别目标，dense token 对"同一人不同图"的判别力有上限，轻量头拉不动（e20 到顶）。
- 要 competitive，必须让 backbone 适应 ReID。全量微调 12 层 768d ×15K 数据会过拟合 + 成本大。**LoRA**：只在 attention q/v 注入低秩适配（rank 8/16），DINO 主权重冻结，参数量小、过拟合风险低，是"让冻结特征端动起来"的最小代价方案。

## 核心假设

给 DINOv2-base attention 加 LoRA（q/v，rank 8/16，alpha 16），DINO 主权重冻结，只训 LoRA + 轻量头（proj + BNNeck + 全局分类器 + part 分类器），**重遮挡 part-MaxSim mAP 能突破 14 往 competitive 走**。能破 → DINO 创新线可做成；破不动 → 冻结对应特征这条线天花板确实低，止损。

## 技术方案

### 关键变化 vs exp324b
- exp324b 在**缓存的部位特征**上训（DINO 不在循环里 forward，飞快）。
- exp324d 中 LoRA 改变 DINO 输出 → **不能再用缓存特征**。每 step 必须：图 → DINO(LoRA) forward → 姿态部位池化（**可微**）→ 轻量头 → 损失。慢但 3090 可承受。

### 数据流（每 step）
1. dataloader 喂 **图像 tensor** `(B,3,448,224)` + 预计算的**每图部位池化稀疏权重** `pool_w (B,NPARTS,GRID_H*GRID_W)` + part_vis `(B,NPARTS)` + label。
   - 池化权重在数据准备阶段一次性算好（cell 选择来自姿态 keypoint grid + 3×3 窗，不需要梯度），缓存到磁盘。池化本身 = `pool_w @ patch_flat`，**对 patch 可微** → 梯度回流 LoRA。这等价于 exp324 `build_part_pose` 的 mean-over-cells，但写成可微矩阵乘。
2. DINO(LoRA) forward → `last_hidden_state` → 去 CLS → `(B, GRID_H*GRID_W, 768)`。
3. `parts = bmm(pool_w, patch)` → `(B,NPARTS,768)`（visible part 是 cell 均值，invisible part 全 0，与 exp324b 一致）。
4. 轻量头（**与 exp324b PartHead 完全相同**）：shared Linear 768→512 + BNNeck → 全局 = 可见部位投影向量 masked mean。
5. 损失（**与 exp324b 完全相同**）：全局 ID CE（BNNeck）+ 全局 batch-hard soft-margin triplet（pre-BN 全局）+ part_weight×per-part ID CE（每个可见部位投影向量，共享 part 分类器，默认 0.5）。

### 有效 BS = 64（硬约束，不可改）
- DINO forward in loop + 448×224 输入，3090 24G 显存可能不够喂满 BS64。
- 策略：物理 micro-batch 可降（如 16 或 32），用**梯度累积**凑有效 BS=64。PK 采样仍按 P16×K4=64 一个完整逻辑 batch 出，micro-batch 是对这 64 个样本切片做累积。**triplet 必须在完整 64 个样本上算**（batch-hard 需要全 batch 的正负对）→ 所以策略：先 forward 全 64 个样本的 DINO（no_grad 分块？不行，要梯度）。
- **正确做法**：对 64 样本分 micro-chunk forward+保留计算图会爆显存。改用 **gradient checkpointing**（DINO encoder layer 级）让 64 样本一次 forward 显存可控；若仍不够，micro-batch 累积时 triplet 单独处理（见下）。
- **最终方案（见审查）**：优先 gradient checkpointing + 一次性 forward 全 64 → triplet 在完整 batch 上算，最干净。先小验证显存；不够再降级到 micro-batch + 在每 micro-chunk 内算 triplet（近似，记录在案）。

### LoRA 配置
- `peft.LoraConfig(target_modules=["query","value"], r=8/16, lora_alpha=16, lora_dropout=0.0)`，DINOv2 attention 是分离的 query/key/value Linear（已验证），suffix 匹配命中全 12 层 q/v。
- DINO 主权重 `requires_grad_(False)`；peft `get_peft_model` 自动只放开 LoRA。轻量头单独 requires_grad。
- 优化器两组：LoRA 参数 + 头参数（头 LR 可同或略高）。

### 修改文件
- 新增 `scripts/exp324d_lora.py`（不碰 model/ 核心、不碰 train.py、不碰 exp324b 脚本）。复用 `exp324_dino.py` 几何/池化/part-MaxSim/eval_func + `exp324b_train_head.py` 的 PartHead/triplet/PKSampler/eval（import 复用，不复制）。

### 超参
- rank 8（先），alpha 16，lora_dropout 0；LoRA LR 1e-4，head LR 3.5e-4（Adam）；cosine LR；epoch 30-40（带 DINO forward 慢，看趋势够）；part_weight 0.5；margin soft。eval period 5（epoch 少，密点看轨迹）。

## 预期结果

- 假设成立：重遮挡 part-MaxSim mAP 破 14 明显上升（进入 20-40+ → DINO 线有救、competitive 可期）。
- 失败最可能：(a) LoRA 容量不足拉不动（停在 ~14）→ 试 rank 16 / 加 MLP head；(b) 过拟合 15K（train acc 高但 eval 不涨）→ 加 dropout / 减 epoch；(c) 显存/速度不可承受 → 降 micro-batch + 累积。若 rank8/16 都破不动 14 → 冻结对应特征天花板确实低，止损这条线。

## 对照组

- **baseline**：exp324b 冻结头 e60（part 重遮挡 8.65 / 全部 14.61 / cos 全部 13.51 / cos 重遮挡 7.32）。**唯一变量 = LoRA 解冻 DINO**（头/损失/采样/eval 全同）。
- 上界参照：exp255 Swin 75（不正面比，定位"frontier-independent 新表征 + 范式角度"）。

## 协议待办（开训前硬性）
1. [x] exp324b 冻结头确认天花板低
2. [ ] 写 `scripts/exp324d_lora.py`（可微部位池化 + LoRA + 梯度累积/checkpointing）
3. [ ] Claude broad review → approve
4. [ ] Codex review → approve
5. [ ] 双审查 approve 后小验证（前几步 loss + 显存）→ 全量后台跑
