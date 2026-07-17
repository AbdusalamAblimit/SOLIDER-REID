# LM-ReID 训练端创新候选（2026-06-26，3-codex 头脑风暴综合）

用户问："是不是多想一个 train-time 方法，两个结合更有把握？" → 启 3 codex 各从一个 lens 深挖（litreview2/train_lens{1,2,3}_*.md），收敛到两个机制。

## 硬约束（已试死的）
训练端做 invariance / feature-collapse 压判别性 = 死路：consistency −1.73、lattice-aug 不特殊 +0.54、L_marg 训练有害大元凶。新机制必须绕开"逼 K 变体特征塌成一个点"。

## 机制 A：学怎么加权边缘化（LPA / LUD）— codex 最高 7.5/10
- **LPA（Lattice Posterior Amortization，lens3 #1）**：冻结 backbone，只训小头预测 posterior qφ(z|x)（z=bbox/phase/kernel 格点），测试时 `score = logsumexp_z[log qφ(z|q) + cos(fθ(Tz q), fθ(g))/τ]`，uniform → posterior 加权。
- **LUD（Lattice Uncertainty Distillation，lens2）= 同族**：训头预测 lattice 风险 risk_k（stopgrad teacher 的 spread + margin），测试 per-variant 加权。
- **关键洞察（lens2）**：query 级 scalar confidence **不能改 gallery 排序**，必须 per-variant 权重 pi_k 才涨 mAP。
- **不塌缩**：主 embedding 完全不动（stopgrad），只训加权头。
- **kill-switch**：冻 no-LM-loss ckpt + 训 qφ 1-3 epoch；weighted 比 uniform 多 ≥+0.4 mAP(h12/16) + 格点预测 acc ≥35% + LATTICE−TTA 不降 → 活，否则杀。
- **不撞 PFE**：PFE=generic data-quality uncertainty；LPA=supervised sampling-lattice posterior（隐变量来源明确 = LR 采样格点，监督来自 K-hypothesis empirical spread）。

## 机制 B：学怎么对齐检测框（LC-STN / BLC）— 7/10（lens1）/ 6.5/10（lens3）
- 训小模块**估计 LR crop 格点偏移 (dx,dy) 并 grid_sample 重采样到 canonical**（类 STN，但监督几何参数非身份特征），残差留 test-time marginalization。
- 第一版 translation-only sub-pixel re-centering（max_shift=1.25 LR px，tanh bound），只打 bbox 主因子。
- **更有分量**：改进模型本身、能提 K=1 单图（不只测试加权）。**但风险高**：可能替代而非叠加边缘化；可能学 dataset center bias 致 HR sanity/h32 掉点。
- **不塌缩**：压的是已知几何扰动参数、不是身份特征；3 保险（HR gallery bypass / 硬边界 / 冻 backbone 先 probe）。
- **kill-switch**：冻 backbone 只训 canonicalizer 预测注入 offset；injected shift MAE<0.35 LR px + K-spread 降≥20% + K=1 mAP +1.0 OR K=9 再+0.3/0.5 + HR sanity 掉<0.2 → 活。看 θ 别 saturate 到边界（伪信号）。
- **novelty**：STN/AlignedReID/PAN/STNReID/CDPM 先例多 → 卖点不是"用 STN"，是"LR detector crop lattice 当可边缘化隐变量 + 监督 canonicalization 非 invariance + 残差 decision marginalization"组合无先例。

## codex 明确不建议（现在）
- raw Hard-Lattice ERM/CVaR（太像更狠 lattice-aug，续压判别性）→ 只做 cheap ablation（= 正在跑的 LM-S5，验证后即停）
- TTT（无标签自我确认错误）/ SR-auxiliary（撞车 + 拉回"LR=缺细节"削弱 re-frame）/ DEQ（不贴）/ EM-backbone（复现 L_marg 有害）

## 决策
- 两个 kill-switch 都冻结 backbone、廉价 → **两个都验，先 A（置信高）后 B（分量大）**。过的那个 = 训练端第二 contribution，和 test-time marginalization 组成完整 train+test 方法（推 7-8/10）。
- A、B 甚至可叠加（B 对齐降 spread + A 加权残差）。
- 当前 GPU 被 Hard-Lattice(3090)/Hard-ordinary(4090) 占 → 先实现两个 kill-switch 代码，GPU 一空即 cheap probe（不直接堆训练）。
