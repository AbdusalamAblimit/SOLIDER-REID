# exp359 LM-ReID monitor

## 训练配置
- 脚本 `cvpb_lm_reid_train.py`，微调 exp260b backbone（pose-OFF global feat，pose_dict=None）
- Market 751 id / 12936 img，M=3 lattice variants/img（1 canonical + 2 随机轴），h∈{16,24,32} severe-biased（0.5/0.3/0.2）
- lr 1e-3（cosine + warmup5），40 epochs，BS=64（P16×K4），save_every 10，AMP
- `L = L_id + 1.0·L_marg + 0.2·L_cons`（L_adv off，lam_adv=0）
- lab-3090 PID 1335155，log `/tmp/exp359_lm_reid.log`

## 双审
- Claude 自审（主 Opus 循环）+ Codex 独立审 round-2 = approve（round-1 抓 5 findings 全修：High-1 RNG per-epoch / High-2 随机轴 / Medium-3 GRL 单 λ / Medium-4 per-slot triplet）。

## smoke 结论
- **pipeline 健康**（no crash，4 loss 计算，backward，save 工作）。
- 150-id smoke（lr 1.5e-3, 2 epoch）：iter 打印是**累积平均**（agg/n，注意别误读）；反推瞬时 loss epoch 末 6.5→5.1（在学），但 epoch 边界有 spike（每 epoch fresh 随机 variants 冲击 + warmup LR 升）。
- 判断：32/150-id 微子集 spike 偏大（模型过拟合那几十 iter 的 variants）；真实 751-id × ~202 iter/epoch 见足够多 variants → 应更稳。acc 是分类 acc 非检索 mAP，真判据=eval。

## 进度
- **[启动 2026-06-25]** 真实训练 40ep，lr 1e-3。
- **[epoch 0 健康 ✓]** GPU 100%/13GB（AMP，无 OOM），it200 L=3.30 / acc=0.845（累积平均），loss 降、acc 高、**完全无 spike**——**坐实 smoke 的 spike 是 32/150-id 微子集 artifact（classifier 不平衡+BN 偏移），full 751-id 训练稳定**（smoke L 飙 9-13，这里仅 3.3）。
- 速度: **~5min/epoch**（306-314s/epoch；上轮误判 18min）→ **epoch-10 ckpt ~50min**。
- **[epoch 1-3 健康]** L 稳定 ~3.3（warmup LR 升轻微波动），acc 0.846→0.888。
- **[epoch 5-7 收敛漂亮]** warmup 结束 cosine 段：**L 3.3→1.66，acc 0.888→0.979，cons loss 0.035→0.020（模型在变 lattice-invariant，机制在工作）**，marg 1.14→0.28。无崩溃。
- 触发器 b3e9ya110 等 epoch-10 ckpt → eval。训练 epoch 40 全部完成（transformer_40.pth），acc→1.000，cons→0.011。

## ★epoch-10 eval 结果（method-vs-trick 关键，2026-06-25）

fine-tuned ckpt transformer_10，h=16（SANITY HR mAP 83.31，frozen 94.43）：

| | mAP | 增益分解 |
|---|---|---|
| frozen single | 42.65 | baseline |
| frozen lattice ensemble | 46.87 | 零训练上界 |
| **LM-ReID single** | **69.96** | +27.3（"在 LR 上训"=标准 CR-ReID，**非我们创新**） |
| **LM-ReID lattice-mean** | **72.60** | +2.63 over single（lattice-marg，**真创新**） |
| LM-ReID ordinary-TTA | 69.09 | −0.87 |
| **LATTICE−TTA** | | **+3.504**（lattice-specific，干净 beat trivial TTA） |

h=24：single 78.67 → lattice 79.67（+1.00），LATTICE−TTA +0.978。

**诚实分解**：①+27 大头是 LR fine-tune（标准），②lattice-marg 真贡献 +2.63 但 robust beat TTA +3.504（frozen +3.04 / trained +3.504 都成立）。**HR-gallery 退化担心是多虑**（SANITY 掉但 LR mAP 高=模型学了 LR↔HR 对应，牺牲 HR-HR 换 LR-HR 正是 CR-ReID 目标，已纠正误判）。codex 过线被远超**但大头是 LR fine-tune 不是 lattice 机制** → 真判据靠 ablation。

## ★ablation + final eval（2026-06-25 在跑）

- **no-LM-loss ablation**（PID 1347672）：M=3 保留 LR 增强，关 L_marg/L_cons（lam_marg=0 lam_cons=0），只 L_id。隔离 consistency 训练独立价值。out exp359_abl_noLMloss，等待 bw6sm13th。
- **final eval**（PID 1347722）：epoch 40 ckpt，h=16/24/32 headline。log /tmp/exp359_eval_ep40.log，等待 bju02h54q。
- 判：LM-ReID single > ablation single → consistency 训练有独立价值（强 method）；≈ → 只是"LR 增强 + 测试时 lattice ensemble"（弱，test-time trick 级）。

## ★final eval（epoch 40 ckpt，2026-06-25，headline）

SANITY HR 86.09（epoch-10 83.31 回升，frozen 94.43）。

| h | single | lattice-mean | lattice-MaxSim | TTA-MaxSim | **LATTICE−TTA** |
|---|---|---|---|---|---|
| **16** | 75.71 | 78.01 (+2.30) | **78.04 (+2.33)** | 75.04 | **+3.006** |
| 24 | 81.99 | 82.74 | 83.00 (+1.01) | 81.82 | +1.174 |
| 32 | 84.10 | 84.60 | 84.81 (+0.71) | 84.11 | +0.697 |

- epoch 40 > epoch 10（h=16 single 69.96→75.71，+5.7），lattice-marg 增益随 h 递减（+2.33→+1.01→+0.71）= 符合"分辨率升高 lattice uncertainty 消退"机制，**三个 h 全 beat TTA**。
- **诚实**：大头是 LR fine-tune（single 75.71 vs frozen single 42.65=+33，标准 CR-ReID，非创新）；**lattice-marg 真贡献 +2.33（robust，frozen +3.04/trained +3.006 一致 beat TTA）= 方法的 lattice-specific 贡献，但是 test-time-trick 量级**。
- **method-vs-trick 终判仍靠 ablation（no-LM-loss）**：若 LM-ReID single > ablation single 则 consistency 训练有独立价值。

## ★ablation ladder（2026-06-25，method-vs-trick；OSS 打通后 ablation 移 4090 mmpose-abu，3090 跑 M=1）

h=16 single / lattice(best) / LATTICE−TTA：

| 配置 | single | lattice | LATTICE−TTA |
|---|---|---|---|
| **M=1 plain**（标准 CR-ReID，M=1，lam_marg/cons=0） | 73.62 | 75.95 | +3.727 |
| **M=3 no-LM-loss**（增强但关 L_marg/L_cons） | **77.44** | **79.90** | **+2.576**（SANITY 88.92） |
| **M=3 full LM-ReID** | 75.71 | 78.04 | +3.006 |

- **full − M=1 = +2.09**（single+lattice 都 +2.09）= M=3 增强 + consistency 合计。
- **★★★命门结果（2026-06-25）= 证伪 method**：no-LM-loss single **77.44 > full 75.71（+1.73）**！**consistency 训练（L_marg+L_cons）= HARMFUL 有害**，不是无用。分解：M=3 增强 +3.82（73.62→77.44，大）；consistency −1.73（77.44→75.71，过度正则压判别性）。SANITY 也证 no-LM HR 88.92 > full 86.09。**LM-ReID 作 method 死**（以 consistency 为创新，比简单 M=3 增强+测试时边缘化更差）。剩 test-time lattice-marg robust（五配置 beat TTA）但 trick（codex 5/10）。
- lattice-marg 三配置都 robust beat TTA（M=1 +3.727 > full +3.006：越不 lattice-invariant 的模型、测试时边缘化获益越多 = "训练内化测试时边缘化"干净机制故事）。
- cons-only（lam_cons=0.2 only）在 3090 跑（bsc27k3vn，补全消融）。

## ★salvage（2026-06-25 codex 后）：LM-ReID 没全死，是 consistency 版死了

salvage codex（`litreview2/lmreid_salvage.md`）：主救法 = **重构**，不卖 consistency，卖"**lattice-aware augmentation 增判别性 + test-time 对 lattice hidden variable 边缘化**（非压成同一 embedding）"。信心 6→7/10 after 命门对照，CCF-B 有条件可行。核心：no-LM-loss 77.44 已最强 = "invariance 死、lattice hidden variable + marginalization 没死"。

**★命门对照（4090 跑中 `b46h62df9`）= M=3 ordinary-aug no-LM**（同 3 views/h/步数，variants 换普通 random crop/flip/color，非 lattice）：
- ≈77+（=lattice no-LM）→ M=3 增益只是"更多增广"，training-side 创新缩水 → 纯 test-time。
- 明显低（−1.0+）→ **lattice-aug 是 lattice-specific** → 改名 **Lattice-Aware Augmentation + Lattice-Marginalized Retrieval** 重投。
- 脚本加 `--aug_mode ordinary`（make_ordinary_variants），smoke 过。

**★命门结果（2026-06-26）= lattice-aug 训练端不特殊（仅 +0.54）**：

| h=16 | single | lattice | LATTICE−TTA | SANITY |
|---|---|---|---|---|
| ordinary-aug（M=3 随机增广） | **76.90** | 79.28 | +1.624 | 88.96 |
| no-LM-loss（M=3 lattice） | **77.44** | 79.90 | +2.576 | 88.92 |

- lattice-aug 仅 **+0.54** over ordinary-aug（阈值 −1.0+ 才算 lattice-specific）；高 h 更平（h=24 84.00≈84.28，h=32 86.48≈86.44）→ **M=3 训练端增益主要是"更多增广"，lattice 训练不特殊**，salvage 主路 (a) 弱。
- **★不对称（干净）**：test-time lattice-marg 四配置全 beat TTA（M=1 +3.727 / ordinary-aug +1.624 / lattice no-LM +2.576 / full +3.006）→ **lattice 在 test-time 是真 hidden variable，在 training 不特殊**。
- 剩：(b) 纯 test-time（robust 但 codex 4.5/10）/ (c) Hard-Lattice ERM（被命门削弱）/ (d) pivot。**post-命门 codex（53466，bulrvjdg2）定夺**：含"test-time 特殊 / training 不特殊"不对称能否成更干净卖点。cons-only（3090）仍跑（诊断 L_cons vs L_marg，现次要）。

## ★LM-S2 inference 主实验（2026-06-26，重定位后第一个实验）= test-time 故事干净

no-LM-loss ckpt，h=12/16/20/24/32，single vs lattice-marg vs 普通TTA（MaxSim）：

| h | single | lattice | LATTICE−TTA | 普通TTA gain |
|---|---|---|---|---|
| 12 | ~66.7 | ~72.2 | **+6.534** | **−1.005** |
| 16 | 77.44 | 80.00 | +2.734 | −0.177 |
| 20 | 82.49 | 83.69 | +1.052 | +0.153 |
| 24 | 84.28 | 85.28 | +0.979 | +0.018 |
| 32 | 86.44 | 87.18 | +0.537 | +0.206 |

- ① lattice-marg **全 5 分辨率 beat 普通TTA**（robust）；② **优势随 h 单调递减**（+6.5→+0.5）= sampling-lattice 是 severe-LR nuisance 的干净证据；③ **severe LR（h=12/16）普通 TTA 反而有害**，lattice-marg 大涨 = clean contrast（防"普通 TTA"质疑）。
- 强化 test-time decision-marginalization 故事。下一步 LM-S3（logsumexp 聚合）/ LM-S4（phase/bbox/kernel 因子）+ 更强 TTA baseline（kill-switch make_tta_variants 加 color/resize）。
- **cons-only eval 结果（L_cons 诊断，2026-06-26）**：single **76.895**（h=16）vs no-LM-loss 77.44 / full 75.71 → **L_cons 单独只 −0.55，L_marg 才是大元凶（≈−1.18）**（codex 预测"L_cons 主杀"错）。**强化 asymmetry：训练时边缘化 L_marg 有害 / 测试时边缘化有益**——decision-level marginalization 故事更锐。
- **LM-S4 因子分解（h=16，single 77.44）**：phase +1.758 / **bbox +2.842（最大）** / zoom +1.702 / all +2.557 → **bbox 轴（±1 LR-pixel 检测框偏移）主导**，9 bbox > 3+3+3 混合。codex 预测 phase 不对，但 bbox 故事更直观（severe LR 检测框 ±1 LR 像素 = 几个 HR 像素 = 真实 crop 不确定性）。三轴全 beat TTA。→ nuisance 主要是 **LR 检测框/crop 不确定性**。
- LM-S2-strong（强 TTA 防御）还在跑；**LM-S5 Hard-Lattice ERM**（3090，bqd5skbum，loss_mode=hard 已实现+smoke 过）epoch 0，待 Hard-ordinary 对照（需 Hard-Lat ≥+0.8 over Hard-ord 才算训练端活）。

novelty：无 exact prior（sampling-lattice as LR-ReID hidden variable + marginalization）；BlurPool(aliasing)/FlipReID(flip consistency) 是相关先例；"consistency 有害"不单独成 paper 但强消融卖点。
training-side 转向备选（若 ordinary-aug 抹平 lattice 特异性）：Hard-Lattice ERM/CVaR（优化最难 variant 的 CE+triplet，非压 embedding）/ set-wise retrieval training。

### [2026-06-26] 训练端三大类全死（LSRC eval 确认）

| 指标 | no-LM-loss | LSRC lam0.5 | Δ |
|---|---|---|---|
| HR sanity | 88.92 | **85.84** | **−3.08** |
| h16 single | 77.44 | 75.70 | −1.74 |
| h16 lattice(MaxSim) | 79.90 | **77.98** | **−1.92** |
| h24 single | — | 82.31 | |
| h24 lattice(MaxSim) | — | 83.27 | |

- **LSRC（backbone set-loss，bag-to-bag set-supcon+neg-tail，4090 lam0.5 full fine-tune）死**：训练 acc 1.000 过拟合训练集，测试全掉，backbone 被训坏（HR sanity 掉 3 点）。marginalization 在受损 backbone 上仍 +2.288（证机制本身没问题，是 backbone 被训坏）。**asym 不用试**（对称 M×M 给 gallery-side oracle = 宽松上界，对称死→非对称必死）。3090 lam1.0 必死。
- **训练端三大类全死**：① frozen 重投影/重加权（LS-MRT +0.028 / LPA +0.075 — 无 headroom）；② backbone 改 loss（LSRC −1.9 / consistency −1.73 — 损判别力）；③ robust ERM（Hard-Lattice 76.9<77.44）。
- **强结论**：no-LM-loss backbone 已是 LR-ReID 好特征，**test-time decision marginalization 是唯一有效杠杆**。论文 = test-time 6/10 核心 + 训练端系统反例。备选 BLC（input canonicalize，design_blc.md）market 受限未验。启 codex（train3_{fourthclass,paperstrategy}）。审查纪律见 codex_review_lsrc.md（审出 Critical+High 已修）。
- **codex final（train4，8.5/10）判训练端确定无空间，别硬凑**：4 类全封（frozen/sidecar 无 headroom含LATS / backbone-loss 伤判别 / robust-ERM 没赢 / BLC 逻辑封住）。训练端定论穷尽（8 机制 + 4 codex）→ 转 test-time 论文 + "Why Training-Time Invariance Fails" 反例节。

### [2026-06-26] K-sweep compute-accuracy（no-LM-loss baseline，h=16）

| K | mAP | gain over single | 收益% |
|---|---|---|---|
| 1 (single) | 77.44 | — | 0% |
| 3 | 78.73 | +1.29 | 53% |
| 5 | 79.61 | +2.14 | 87% |
| 9 | 79.90 | +2.46 | 100% |

- **K=5 已达 87% 收益（79.61≈79.90）**，K=3 中等（53%）。compute-accuracy 边际递减 → 论文"防 compute"论点：K=5 性价比高（省 4/9 compute 保 87% 收益），K=9 完整。marginalization 的 compute 可调，K=5 是 sweet spot。

### [2026-06-26] LM-S3 聚合消融（no-LM-loss，K=9，5 分辨率）

| h | single | mean-feat(embed) | MaxSim(hard) | logsumexp(soft) |
|---|---|---|---|---|
| 12 | 66.72 | 72.25 | 72.03 | **73.01** |
| 16 | 77.44 | 79.84 | 80.00 | **80.28** |
| 20 | 82.49 | 83.41 | **83.69** | 83.62 |
| 24 | 84.28 | 85.02 | **85.28** | 85.16 |
| 32 | 86.44 | 86.87 | **87.18** | 87.02 |

- **soft decision marginalization（logsumexp，LM-ReID 公式 s=τlog[1/KΣexp(cos/τ)]）在 severe LR（h12/16）最优**；mild LR（h≥20）hard-max 略超。三种聚合都 >> single；decision-level（max/logsumexp）≥ embedding-mean（h≥16）。差距虽小（±0.3）但完整 controlled ablation，logsumexp 是理论 motivated 的 sweet spot。

### [2026-06-26] LM-S2-strong 5 分辨率（强 TTA 防御，no-LM-loss）

| h | phase-lattice best | strong-TTA best | LATTICE−TTA |
|---|---|---|---|
| 12 | +5.494 | −1.781 | **+7.275** |
| 16 | +2.811 | −0.526 | +3.337 |
| 20 | +1.195 | −0.035 | +1.230 |
| 24 | +0.991 | −0.144 | +1.135 |
| 32 | +0.759 | +0.003 | +0.757 |

- **lattice marg 全 5 分辨率 beat 强 TTA（pad-crop+resize-jitter+color），+0.76~7.28，h 越低优势越大**。severe LR（h12）强 TTA 本身 −1.781（有害），lattice +5.494 → 干净堵死"不就是多裁几次"质疑。论文核心防线。

### [2026-06-26] backbone 泛化（Swin-small baseline，market，LATTICE−strong-TTA）

| h | single | MaxSim | LATTICE−TTA |
|---|---|---|---|
| 12 | (弱) | — | +0.778 |
| 16 | 41.41 | 46.20(+4.80) | +3.061 |
| 20 | 61.92 | 66.43(+4.51) | +3.162 |
| 24 | 70.31 | 73.90(+3.60) | +2.370 |
| 32 | 81.97 | 83.93(+1.96) | +0.883 |

- **lattice marginalization 在 Swin-small backbone 上也 beat 强 TTA（h16/20 +3.06/+3.16）**，证机制不依赖 SOLIDER backbone（push-7.0 kill-switch②：两 backbone 成立 ✓ Swin+SOLIDER）。Swin baseline LR 更弱（h16 single 41.4 vs SOLIDER 77.4）→ lattice 敏感度更高。

### [2026-06-26] push-7.0 实验（codex push7 给冲 7.0 路径 6/10）

- **LM-S4 bbox 5-height**（完整）：bbox-only ≈ all-axis 跨所有分辨率（h12/16/20/24/32 = 72.01/80.34/83.94/85.53/87.33，all 72.03/80.00/83.69/85.28/87.18）→ **bbox 是主因子，跨分辨率确认**（好素材）。

**★★detector-jitter σ-sweep（冲 7.0 detector 腿确定失败）**：均匀 ±1 离散格点改成连续 Gaussian center+scale（模拟真实检测器 localization error），扫 σ：

| detector σ | h12 marg | h16 marg | h16 LATTICE−TTA | h20 LATTICE−TTA |
|---|---|---|---|---|
| 0（均匀离散） | +5.49 | +2.81 | +3.34 | +1.23 |
| 0.25（理想精确 detector） | +3.68 | +1.55 | +2.15 ✓ | +0.76 ✗ |
| 0.5（真实 COCO detector） | +2.18 | +0.86 | +1.46 ✗ | +0.37 ✗ |
| 1.0（大误差） | **−5.85** | **−3.11** | **−2.52** ✗ | **−1.39** ✗ |

- **marginalization 增益随 detector 误差 σ 单调衰减**。codex kill-switch③（h12/16/20 都 ≥+2）**确定不过**：σ=0.25（理想）h20 不过，σ=0.5（真实 detector localization error ~0.5-1.0 LR-px）h16/20 都不过。
- **→ 冲 7.0 的 detector 腿失败，LM-ReID 诚实定位 6.5。** 但 σ-sweep 是**有价值的诚实诊断**：精确界定机制范围 = **sub-pixel sampling-lattice 边缘化（小精确扰动），不是对大 detector 框误差的鲁棒性**。论文 Discussion 用这个诚实界定（比硬吹 7.0 扎实）。
- 6.5 后续（巩固泛化非冲 7.0）：跨数据集 MSMT17（kill-switch②）+ adaptive-K（compute）+ paper 6.5 定位写作。

### [2026-06-26] 跨数据集 MSMT17 = 止损（config 配对深坑）

参数化 cvpb 加 `--dataset msmt17`（msmt17_split 读 list 文件）+ `--semantic_weight` override。**msmt17_split 数据读取完全正确**（q 11659/g 82161 标准 count，pid 0-3059 同空间，img 存在，q⊂g）。但 **MSMT17 ckpt 的 model config 配对是深坑**：正确 config `swin_small_pose.yml` 已被删（缺失），现有 `pose_backbone_psg_small.yml`（sw=0.2）/swin_small.yml（baseline 无 pose_dict 报错）都不匹配 ckpt 训练（sw=0.6 + pose-mul-scale0.3 + llw0.7）；SANITY 持续 2.67~4.29（特征垃圾，非数据问题）。**止损留用户**（用正确 config 或重训），不无限 debug 缺失 config。

**→ 跨数据集（kill-switch②第二维度）做不了；但 backbone 泛化（Swin market +3）已是一个泛化维度，且冲 7.0 已失败（detector 腿），跨数据集对 6.5 是 nice-to-have 非必需。LM-ReID 诚实定位 6.5（核心齐 + 训练端反例 + backbone 泛化 + σ-sweep 机制范围界定）。**

### [2026-06-26] adaptive-K（最后 supporting，中性）

per-query phase volatility（median 阈值）选 K：高 vol marginalize K=9，低 vol K=1，avg_K=5（56% compute）。

| h | single | adaptive-K(56%) | fixed K=9 | 保留率 |
|---|---|---|---|---|
| 12 | 66.79 | 70.76 | 72.01 | 76% |
| 16 | 77.47 | 79.66 | 80.10 | 83% |
| 20 | 82.48 | 83.58 | 83.72 | 88% |

- **adaptive-K（per-query）≈ fixed K=5（uniform）at 56% compute**（h16 adaptive 79.66 ≈ K-sweep K=5 79.61）→ per-query volatility selection 没明显优势 over uniform K。和 LPA 死因一致（query-side 预测 lattice 受益度做不到）。compute 故事用 K-sweep（fixed K=5 87%）就够。

**★★→ LM-ReID 6.5 实验链完整**：LM-S2/S2-strong/S4/K-sweep/LM-S3/backbone/σ-sweep/adaptive-K + 训练端反例（4 类穷尽）。冲 7.0 失败（detector 腿 σ-sweep 单调衰减到负 + MSMT17 config 缺失止损）。**6.5 是诚实天花板，中等偏强 B 类候选。下一步=完善 paper 6.5 + 用户醒决定收尾投 vs 换方向（d17 备胎）。**
