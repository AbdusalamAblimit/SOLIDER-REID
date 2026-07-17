# 实验 exp359-LSRC: Lattice-Set Retrieval Contrastive（训练端机制）

> ⚠️ **流程补记**：本 design 是**事后补写**（2026-06-26）。LSRC 已在 smoke 通过后直接启动全量训练，违反了"训练前先写 design + 双审查"铁律。现补 design + codex 独立审查（codex_lsrc_review.md），若审出 Critical 立即 kill 重跑。教训记入 memory `pre-experiment-review-discipline`。

## 动机

- test-time decision-marginalization（对 K=9 lattice 变体边缘化检索决策）已成立（6/10），但纯 test-time。用户要训练端互补机制凑成完整方法。
- **frozen-feature 类训练端机制全部证负**：LS-MRT（set-retrieval 重投影 linear P）+0.028、LPA（query-side 加权）+0.075，两个冻结特征 probe 都 ≈0。
- **诊断**：no-LM-loss 的冻结特征对 test-time 边缘化已近最优——lattice union 固定，重投影/重加权没有 headroom（oracle +4.338 是 gallery 真 ID 上界，frozen 够不着）。
- → 训练端价值必须**改特征本身（backbone）**。codex（train2_backbone）收敛到 LSRC，7.5/10。

## 核心假设

一句话：**训 backbone 让 K 个 lattice 变体形成"可边缘化的证据集"**——让负样本不产生假高 lattice 相似（marginalization 的数学瓶颈），让正样本有多路可匹配证据——比 frozen 重投影有空间，因为 backbone 能改变 aliasing/bbox-shift 敏感的特征本身。

## 技术方案

- **改动文件**：`experiments/cargo_cvpb/cvpb_lm_reid_train.py`，新增 `--loss_mode lsrc` + `--lam_lsrc/--lam_negtail/--lsrc_tau/--lsrc_tau_c/--lsrc_margin/--lam_coverage/--cov_m`。
- **数据流**：B 张图各生成 M 个 lattice 变体 → backbone → `gf_bm [B,M,D]` → L2 norm → bag-to-bag `sim4 = einsum('ikd,jld->ijkl')` [B,B,M×M] cos → `S = tau_l·logsumexp(sim4/tau_l)` [B,B] set score。
- **loss**：`L_id (CE + per-slot batch-hard triplet) + lam_lsrc·(L_setsupcon + lam_negtail·L_negtail + lam_coverage·L_cov)`
  - `L_setsupcon`：S 做 batch-softmax supervised contrastive（对齐 test-time logsumexp 边缘化），对角 -1e9 + pos 对角 0 排除 self。
  - `L_negtail`：`softplus((max_neg_lattice_pair_cos - margin)/0.1)`，压低负样本对的最大 lattice-pair 相似度（核心：marg 最怕某错 lattice 假高分）。
  - `L_cov`（codex step2，默认关）：top-M 正样本 lattice-pair，正样本不靠单一固定 lattice 赢。
- **关键超参**：lam_lsrc 0.5(4090)/1.0(3090) hedge，lsrc_tau 0.1，lsrc_margin 0.4，lr 1e-3，40 epoch，`--lam_marg 0 --lam_cons 0`（隔离 LSRC）。

## 预期结果

- 假设成立：lattice mAP > no-LM-loss 79.90 **+0.3~0.5**，且 single_mean 不掉 >0.8（单变体判别力保住）。
- 失败最可能原因：(a) set-supcon 退化成普通 instance 判别，和 L_id 冗余（→ +0）；(b) neg-tail 压过头连正样本 lattice 也压（→ single 掉）；(c) 又一次"backbone 已最优、训练端无空间"。
- **训练观测（进行中）**：4090 lam0.5 ep23 acc 0.999 / 3090 lam1.0 ep13 acc 0.995 → single 判别力完全没塌（codex 最担心的 kill 信号未触发）。真判据仍是 eval。

## 对照组

- **baseline**：no-LM-loss（exp359_abl_noLMloss）= 训练端不动、只 test-time 边缘化 → lattice 79.90 / single ~76.9。
- **消融变量**：只开 LSRC（lam_marg=0, lam_cons=0），单变量隔离 LSRC 贡献。
- **后续消融档**（已备 code，未跑）：set-supcon only → +neg-tail → +coverage，证每项必要。

## 与已死路线的本质区别（codex 核验中）

- 非 **consistency**（exp359 -1.73）：不拉同图变体到均值，允许 residual 存在。
- 非 **L_marg**（frozen 无 headroom）：在**检索决策层**（q-g 相似度，denominator 有真负样本）边缘化，非 train-ID 分类头 posterior。
- 非 **Hard-Lattice ERM**（76.894 死）：把 K 变体当检索证据集，非独立硬样本逐个 ERM。
