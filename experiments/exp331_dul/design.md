# 实验 exp331: Identity-Conditioned Aleatoric Embedding (DUL) — 新监督/目标类

## 动机
- 今晚证负一整片：in-domain 特征机制(吸收)、组合重定义(已鲁棒)、跨域(不塌缩)。剩没测的类 = **新监督/目标**。
- 调研 agent Rank-2 bet = **DUL (Data Uncertainty Learning, Chang et al. CVPR 2020, arXiv:2003.11339)**，搬自人脸识别（遮挡/噪声下）。
- **为何可能逃过"训练吸收"（scout 论点）**：DUL 给模型一个**新自由度（per-image aleatoric 方差 σ²，联合学）**——是新容量而非可被重新提取的冗余结构。**非 PFE**（PFE 冻结模型外挂方差头=已死的后处理类）；DUL 的 σ² 在训练中**反向改 μ**（KL 项把难/遮挡样本拉向类心）。

## 核心假设
遮挡导致 aleatoric 不确定性。让 backbone 输出分布嵌入 (μ, σ²)，σ² **训练中联合学**预测自身不确定性；ID loss 作用在采样 s=μ+ε·σ 上（reparameterization），σ² 经 KL 反向正则 μ。遮挡样本应学到高 σ²、μ 被拉向类心 → 遮挡 mAP ↑。

## 技术方案（substrate = 弱 TransReID vit_base @ hyy）
- 复用 TransReID `make_dataloader`（标准 Occ-Duke train + RandomErasing + PK sampler）+ `make_model` + `make_optimizer` + `create_scheduler`（与 53.5 baseline 同 pipeline）。
- **variance head**：`Linear(768→768)` 输出 per-channel `log σ²`，**零初始化 → log σ²=0 → σ²=1**（起点 = σ²=1 单位高斯**先验**，**非确定性**——采样即注入单位噪声；早期 CE 会自然压 σ↓，据此解读塌缩，Codex Medium）。
- **DUL 前向**：μ=global_feat（model 返回）；logvar=var_head(μ)；σ=exp(0.5·logvar)；ε~N(0,I)；**s=μ+ε·σ**；score=model.classifier(model.bottleneck(s))。
- **loss**：ID CE 作用在 score（采样 s）；triplet 作用在 μ；**KL=0.5·mean(μ² + σ² − logvar − 1)**（real DUL KL(N(μ,σ²)‖**N(0,I)**)，正则 μ→0(拉向类心) + σ→1 防塌常数）；total = CE + triplet + λ_kl·KL（λ_kl=0.01）。**注**：μ² 项与 ID/triplet 判别性张力——若 μ→0 伤检索即 DUL 在 ReID 失败原因。
- **BN 单更新**（Claude H2 修）：sample 路 `model.bottleneck` 置 eval（不更新 running stats），BN 只在 clean μ 上更一次（=det），保单变量 + 防噪声样本污染 eval。
- **单变量对照 = deterministic**（无 var_head、无采样、无 KL，CE 直接作用在原 score；同 scaffold/seed/epoch）。

## Kill-switch（训练模型判据）
- **GO**：DUL 的 μ-matching Occ-Duke mAP **≥ deterministic +1.0** **且 σ² 非退化**（ρ(σ², occlusion) 显著：query(遮挡) 平均 σ² 明显 > gallery(整体)，diff>0 且统计显著）。
- **NO-GO**：mAP Δ<+0.5 **或** σ² 塌成常数（query≈gallery σ²）→ 退化成确定性 + no-op（正是 exp140 confidence 塌 0.99 的死法）。σ² 退化检查是关键 guard。
- **诚实风险**（scout 提）：σ² 极易塌成常数 → dead-on-arrival。kill-switch 的 σ²-occlusion 检查专防此。

## 评测
- Occ-Duke 标准 test（val_loader, R1_mAP_eval, μ-matching）mAP vs deterministic。
- σ² 诊断：抽 query/gallery 每图 σ².mean()，比均值（遮挡是否更高 σ²）。

## 对照组
- deterministic（同 scaffold 无 DUL）vs DUL，单变量=variance head + DUL loss + 采样。
- backbone TransReID vit_base 弱 baseline（~53.5）。不混 NFC/RR。

## 预期
- GO：σ² 学到遮挡不确定性、μ 被正则 → 遮挡 mAP +1~3。
- NO-GO（最可能）：σ² 塌常数（KL 太强/数据无信号）→ 退化确定性。则记录、DUL 死。

## 实现/审查
- self-contained `scripts/exp331_dul.py`（--mode {det,dul}）。Claude broad review + Codex review（hook 双门）→ smoke → 训练 det(GPU0 after ERM) + dul(GPU1) 120ep 单变量。
