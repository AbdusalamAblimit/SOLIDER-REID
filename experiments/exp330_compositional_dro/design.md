# 实验 exp330: Compositional Occluder Generalization + group-DRO（范式：新问题定义 + 匹配目标）

## 动机
- 今晚数据驱动地证负了一整类 **in-domain 特征机制**（burstiness/backdoor/TopoFR/UCE/FM-import）：frozen 看着有戏，ReID 训练后被吸收，**连弱 baseline 也吸收**。frozen kill-switch 会骗人。
- 调研 agent（带着这个教训）唯一 Rank-1 过审 bet = **Compositional Occluder Generalization + group-DRO**，搬自 Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts* (group-DRO), ICLR 2020 (arXiv:1911.08731)。
- **为何这个能逃过"训练吸收"死法（结构性，非侥幸）**：保留的 (occluder类 × 身体部位) 组合**按构造从未进训练分布**，所以训练模型**没有可隐式吸收的结构**——这正是 burstiness 等死掉的失败模式（它们重新提取训练模型早见过的结构）。这个 bet 专门不犯这个错。

## 核心假设
ReID 模型在见过的遮挡组合上学到的是**occluder-specific 捷径**（"car 长这样→忽略"），而非**遮挡-不变的 part-identity 证据**。因此对**未见过的 (occluder类×部位) 组合**会崩。**group-DRO**（最小化 worst-group loss over occluder×part cells，而非平均 loss）逼模型学组合-不变的证据 → 在 held-out 组合上 > ERM。

## 技术方案（substrate = 弱 baseline TransReID/ViT @ hyy，纯 PyTorch）
### 合成组合遮挡
- **occluder 类轴（3）**：{car, bicycle, person}，从 Pascal VOC2012 分割对象裁 RGBA patch（复用 `occlusion_augmentation.py` 的 `load_occluders`，mmcv-free 纯 cv2/PIL）。
- **部位轴（3）**：{head, torso, legs} = 人框的上/中/下三分之一（**region-based 放置，免 pose**——hyy 无 Occ-Duke pose）。把 occluder 居中贴到对应三分区。
- **9 cells** = 3 occluder类 × 3 部位。每张训练图随机抽一个 cell，贴对应 (类,区) 的 occluder，记录 cell 标签。

### 组合 split（held-out 对角）
- **训练见 6 cells**，**hold out 3 对角**：car-legs / bicycle-torso / person-head。
- 保证每个 occluder 类、每个部位**单独都见过**，只是 3 个**组合**没见过 → 纯组合泛化（非"没见过某类/某区"）。

### group-DRO 目标（实现细节，Codex Medium 修正）
- **7 个 group** = 6 个 seen cell（idx 0-5）+ clean（idx 6，未遮挡样本，occlude_prob=0.5）。
- 在线 q 更新（Sagawa ICLR'20）：对**本 batch 出现的** group `q_g *= exp(η·L_g)`（fp32, no_grad），再对**全 7 组**归一。
- **loss 处对 present 组重归一**：`ce = Σ_{g∈present} (q_g/Σ_{present}q_g)·L_g`，权重和=1 → CE 尺度与 ERM 的 mean 一致（单变量隔离，非原始 Σq_g·L_g）。这是刻意的 CE-scale-matched 变体（非字面 Sagawa 全组 worst-group），为保证 ERM/DRO 只差聚合方式。triplet 不变。
- 单变量对照 = **ERM**（同 aug、同 7-group 流、同 seed/预算，CE = per_sample.mean()）。

### 评测协议（新 benchmark）
- **主判据**：对 3 个 held-out 组合，把该组合的合成遮挡施加到 test query → mAP（group-DRO vs ERM）。eval query 遮挡按 (cell,image) 确定性 seed，ERM/DRO 评在**同一**遮挡上（公平）。
- **副判据（脚本外，单独 test.py 验，不进自动 kill-switch）**：standard Occ-Duke 真实遮挡 mAP，确认不伤主任务。kill-switch 脚本只跑主判据（Market 组合 GAP）。

## Kill-switch（训练模型判据，非 frozen——frozen 会骗人）
- **GO**：group-DRO 在 3 个 held-out 组合 cell 上比 ERM **≥ +1.5 mAP**（主判据；副判据 standard Occ-Duke 不掉 >0.5 脚本外单独验）。→ 真组合泛化机制，升级（pose-anchored 放置 + 更细 cells + SOLIDER 强栈）。
- **前置 GAP 检查（先跑 ERM）**：ERM 自己 held-out mean mAP 是否显著 < seen mean（存在组合 GAP）。若 ERM 无 GAP（held≈seen）→ 无 occluder-class 捷径可利用 → 整 bet NO-GO（合理早 kill，省 DRO）。有 GAP 才看 DRO 能否合上。
- **NO-GO**：Δ < +0.7 mAP（训练已吸收 / DRO 只是拿平均换 worst-group 无净收益）。
- 两个训练（ERM / group-DRO）各 ~半天，单 24G GPU；**无 frozen 步骤**，不会像 burstiness 那样误导。

## 预期结果
- 若 GO：ReID 确实学 occluder-specific 捷径，group-DRO 逼出组合-不变证据 → 新问题定义（组合遮挡泛化）+ 新目标（occluder×part DRO）双贡献，是 paper-worthy 范式。
- 若 NO-GO（最可能失败原因）：3×3 cells 太粗，occluder 类间差异不足以构成"组合捷径"（模型已学通用遮挡鲁棒）；或合成遮挡的组合性不够强。则记录、转 BET 2 (DUL)。

## 对照组
- **ERM vs group-DRO**：同 backbone/aug/6-cells/预算，仅 loss 聚合方式不同（单变量）。
- backbone：TransReID vit_base 弱 baseline（hyy，有 headroom）。
- 不混入 NFC/re-ranking。

## 实现计划（分步）
1. [进行中] 下载 VOC2012 到 hyy → 抽 {car,bicycle,person} occluders。
2. 移植 `occlusion_augmentation.py` 到 TransReID dataloader（cell 标签 + region 放置）。
3. 实现 group-DRO loss wrapper（per-group running loss + 指数加权）。
4. held-out-cell 合成遮挡 eval 脚本。
5. design + Claude review + Codex review（hook 双门）。
6. 训练 ERM（GPU0）+ group-DRO（GPU1）并行。

## Novelty（agent 已核）
- 最近 cousin = OGFR (arXiv:2507.08520, 2025) 只做 occluder **类型** held-out（1-D），**无人做 2-D (类×部位) 组合 split + group-DRO**。`"compositional" occluder "body part"` / `group-DRO + reid + occlusion` web 搜零直接命中。
