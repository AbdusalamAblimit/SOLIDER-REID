# Paper 骨架（2026-06-26，test-time 核心已实，train 端待 A/B 验证）

## 暂定标题
**Test-Time Decision Marginalization over the Sampling Lattice for Low-Resolution Person Re-Identification**
（备选副题: Why Embedding-Level Invariance Fails and Decision-Level Marginalization Wins）

## 一句话
低分辨率 ReID 里，LR 观测只是 HR 场景在一个**未知采样格点**（sub-pixel phase / ±1 LR-pixel 检测框 crop / antialias kernel）下的一次采样；正确处理不是训练端逼特征不变，而是**检索时对格点隐变量做决策级边缘化**。

## Motivation / 问题重定义（核心新意）
- 现有 CR-ReID 主线 = SR / resolution-invariant / resolution-adaptive / feature-panning（LRAR/VPFA/PS-HRNet/MRJL/RFD），都把 LR 当"缺细节"。
- 我们重定义: LR = **采样格点不确定性**。同一 HR 人在 severe LR 下，检测框偏 ±1 LR 像素 = 偏几个 HR 像素 = 身份证据真实漂移（rank-flip 验证: 同图不同格点 retrieved ID 会翻）。
- **因子分解（LM-S4）**: bbox 检测框 crop 不确定性主导（+2.84）> sub-pixel phase（+1.76）> zoom（+1.70）。→ mechanism = LR detector crop lattice，直观且新。

## 评测协议重定义（Latent-Lattice CR-ReID，codex push7 核心 framing）

现有 CR-ReID 评测把每个 LR query 当**一个确定图像**（固定 bicubic/downsample/crop）。但 severe LR 下，同一目标对应**一族采样格点**（sub-pixel phase / ±1 LR-px bbox / antialias kernel）——固定评测漏掉了真实部署的采集不确定性。

**Latent-Lattice 协议**：把 LR query 建模为格点隐变量 z 上的观测 p(x_LR|HR,z)，检索决策对 z **边缘化** `s(q,g) ≈ logmeanexp_k sim(f(x_q^{z_k}), f(x_g))`。这把审稿人的"strong TTA/ensemble"质疑改成精确反驳：**普通 TTA 的 K 个增广没覆盖正确隐变量 z（acquisition lattice），我们的 K 是 lattice 积分的蒙特卡洛近似**——LM-S2-strong 实证（强 TTA 全 5 分辨率输给 lattice marginalization，severe LR 处强 TTA 本身有害）。

**诚实边界（σ-sweep 界定，关键不 over-claim）**：framing 严格成立于 **sub-pixel sampling lattice**（小精确格点扰动）；真实检测器大 localization error（σ~0.5-1.0 LR-px）下 marginalization 衰减甚至有害（σ-sweep h12 marg +5.49→+2.18→−5.85 单调衰减到负）。即机制是**采集格点边缘化、不是 detector 鲁棒性**。真实 detector-calibrated 验证需有原图数据集（CUHK-SYSU/PoseTrack），留未来——这也是 6.5（非 7.0）的诚实原因。

## 核心贡献（预计 3 点）
1. **问题重定义** + 诊断: sampling lattice 是 LR-ReID 此前未显式建模的 test-time hidden nuisance；rank-level 干预证明它造成身份翻转。
2. **方法（test-time）**: 决策级 lattice marginalization（K 变体 + mean/max/logsumexp 聚合），并证明 **decision-level marginalization > training-level invariance**。
3. **训练端系统反例（强论点，非"没做出来"）**: 系统验证 4 类"自然但错误"的学习式替代——embedding-invariance / frozen-adaptation / robust-ERM / input-canonicalization——全部无 headroom 或伤判别力（8 机制 + 4 codex 收敛，8.5/10）→ 证 sampling-lattice 该 test-time 边缘化，不该训练端消除/内化。论点：*Learning invariance is the wrong objective; marginalizing decisions over plausible observations is the right one.*

## 实验证据链（已实）
- **LM-S2 inference 主实验**: lattice-marg 在 h=12/16/20/24/32 **全 beat 普通 TTA**，优势随分辨率单调递减（h12 +6.5 → h32 +0.5）= sampling-lattice 是 severe-LR nuisance 的干净证据；severe LR（h12/16）普通 TTA 反而有害。
- **LM-S2-strong 防御**: 用更强 TTA（pad-crop + resize-jitter + color）当 baseline，lattice 仍赢且 gap 更大（h16 LATTICE−TTA +3.337）→ 堵死"不就是多裁几次"质疑。
- **LM-S4 因子消融**: bbox 轴主导（见上）。
- **训练端反例（强消融卖点）**: consistency 拉特征到均值有害（−1.73 还掉 HR sanity）；lattice-aug 训练 ≈ 普通 random 增广（只 +0.54 不 lattice-specific，命门对照）；marginal-likelihood L_marg 是训练有害大元凶（cons-only 诊断）。→ 坐实"训练端 invariance/collapse 压判别性"，反衬 test-time marginalization。

## 训练端穷尽（已完成，写成 "Why Training-Time Invariance Fails" 节，controlled alternatives）

训练端 4 类全负（详见 train_time_pipeline.md），codex final **8.5/10** 判无空间：
1. **embedding invariance hurts**: full LM 75.71 < no-LM 77.44，HR sanity 86.09 < 88.92，L_marg 主害。
2. **frozen adaptation no headroom**: LS-MRT +0.028 / LPA +0.075。
3. **backbone set/robust training damages base**: LSRC HR 88.92→85.84 / lattice 79.90→77.98，Hard-Lattice 76.9 < 77.44，train acc 1.0 但 test 掉。
4. **input canonicalization 被数据封住**: canonicalize bbox（主因子 +2.84）< marginalize 它，market 框已 canonical→退化 single。

★**2026-06-28 真 measure 升级（用户质疑"查代码正确性"后，把论点从"凭外推"变"真 measure"）**: codex 代码审查证实原"8 机制"部分凭外推——LCRS/LRFD/BLC 排队没跑、Hard-Lattice 没清零 lam_marg/lam_cons 不干净。补两个真 measure full run（Swin-Base/exp260b，frozen backbone + cached K=9 feats，streaming）：
- **LCRS**（Lattice-Complementary Residual Subspaces, z_k=norm(P_shared(g_k)+α·P_axis(g_k))+axis-decorr，每 variant 判别+残差互补非压一致）: full run **−4.964** mAP（K-cos 0.9047→0.9358 升=变体趋同塌缩）。
- **LRFD**（disentangle, z_id 纯身份 + r_lat 吸 lattice nuisance 推理丢）: full run **−4.993** mAP（lat_acc 0.540<0.6 = lattice axis 不可分 disentangle 前提就错 + z_id 塌缩）。

→ **六点定律（全真 measure，零外推）**: `consistency(−1.73) / LCRS(−4.96) / LATS(−5.15) / LSRC对称(−1.92)/非对称(−0.33) / LRFD(−4.99)` 塑造/对齐/分离 K 变体全 DEAD（K-cos 升=破坏多样性）+ frozen `LS-MRT(+0.028)/LPA(+0.075)` 无 headroom。**论点硬化**: *任何训练端塑造/对齐/分离 K 变体的机制都破坏 marginalization 依赖的变体多样性（K-cos 升），learning invariance/disentangle is the wrong objective; test-time decision marginalization is the only right one.* 审稿人"你没真做训练端"被"六点真 measure + K-cos 塌缩机理"挡死。

## 已补实验（test-time 补强，6.5 扎实）
- **K-sweep**（compute-accuracy）: K=1→3→5→9 = 77.44→78.73→79.61→79.90，**K=5 达 87% 收益≈K=9** → "防 compute" K=5 sweet spot。
- **LM-S3 聚合消融**: mean/max/logsumexp（h12 73.01/h16 80.28），**soft decision marginalization（logsumexp）severe LR 最优**，三者都 >> single；decision-level（max/logsumexp）≥ embedding-mean（h≥16）。
- **backbone 泛化**: Swin-small market，LATTICE−strong-TTA +0.78~3.16（h16/20 +3.06/+3.16）→ 机制不依赖 SOLIDER backbone。
- **跨数据集 MSMT17**: 参数化跑（in-domain Swin-small），验证机制不只 market（验证中）。
- **多 seed**: 留用户。

## 冲 7.0 尝试 → 诚实界定（codex push7 路径，detector 腿失败）
- codex push7 给冲 7.0 路径（6/10）：detector-calibrated bbox jitter 下机制成立 + 跨数据集 + adaptive-K + 协议化。
- **detector-jitter σ-sweep（关键负结果，诚实诊断）**: 均匀 ±1 离散格点 → 连续 Gaussian center+scale（模拟检测器 localization error），h12 marg gain **+5.49（σ=0）→ +3.68（σ=0.25）→ +2.18（σ=0.5 真实 detector）→ −5.85（σ=1.0）**。marginalization 增益随 detector 误差单调衰减、大误差下有害。kill-switch③（h12/16/20 ≥+2）在真实 detector σ~0.5-1.0 下不过。
- **→ 冲 7.0 失败，6.5 是天花板**。但 σ-sweep 是**有价值的诚实诊断**：机制是 **sub-pixel sampling-lattice 边缘化（小精确扰动），不是对大 detector 框误差的鲁棒性**。Discussion 诚实界定范围（market 图无原图无法真实 detector-calibrate；真实 detector 框需 CUHK-SYSU/PoseTrack 等有原图数据集，留未来）。

## 与 SOTA / 相关工作 narrative
- CR-ReID（SR/invariant）: 我们正交，不重建 HR，重新建模隐变量。
- TTA（k-reciprocal CVPR17 进 SOTA / FlipReID / BNTA）: test-time 后处理可发先例；我们是 lattice-specific marginalization 非通用 TTA。
- Aliasing（BlurPool）: shift-variance 已知，但我们不是"发现 aliasing"，是"把 sampling lattice 当可边缘化隐变量"。
- Alignment（STN/AlignedReID/STNReID/PAN/CDPM）: 若用 B LC-STN，卖点是"LR crop lattice 当隐变量 + 监督 canonicalization 非 invariance + 残差 marginalization"组合无先例。
- Uncertainty（PFE/DUL/UMTS）: 若用 A LPA，卖点是 supervised sampling-lattice posterior 非 generic data-quality。

## 风险 / 诚实定位（paperstrategy codex）
- test-time + 训练端反例: **5.5/10**（容易被打成 strong TTA/ensemble，靠 LM-S2-strong + LM-S4 + 反例三防线撑）。
- 补齐 K-sweep/跨数据集/MLR/真实 detector jitter: **6.5/10**（扎实问题重定义稿，B 类有机会，非稳）。
- **BLC 证伪 → 够不到 7.5**（原 7.5 靠 input canonicalization 过线，已被数据封住）。
- 数字非 SOTA-碾压，是"重定义 + 干净 test-time 机制 + 训练端系统反例"的中等偏强方法稿，目标 CCF-B。**训练端穷尽是论点不是短板。**
