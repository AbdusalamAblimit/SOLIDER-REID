# 实验 exp359: LM-ReID (Lattice-Marginalized ReID)

> 后 PRCV「换问题」阶段第一个 GO 的 method 候选（9 个零训练 cheap-kill 全死后）。零训练 kill-switch 已 GO（d8 lattice, agent a68e6），method-design codex 判 **CCF-B 7/10**（训练版超 frozen ensemble → 8/10）。本实验做训练版，证明 lattice-marginalization 是 **method 不是 ensemble trick**。
> 设计来源：`experiments/cargo_cvpb/litreview2/pivot/clean/lattice_method_design.txt`（完整 codex 设计）+ `cvpb_lattice_result.md`（kill-switch 数据）。

## 动机

低分辨率 ReID 的传统视角：LR = 模糊/缺细节，解法 = SR / resolution-invariant feature。**我们重定义**：一部分 LR 失败不是"缺信息"，而是 **采样格点不确定性（sampling-lattice uncertainty）**——同一个 HR 身份在不同合法的 LR 采样格点（sub-pixel phase / bbox alignment / downsample kernel）下，落到不同 embedding 区域，导致 rank-1 身份翻转。

### 零训练 kill-switch 证据（GO）
frozen exp260b Market，K=9 lattice variants ensemble，HR gallery / LR query：

| h | rank-1 flip% | single LR | lat-MaxSim | **LATgain** | **LAT−TTA** |
|---|---|---|---|---|---|
| 16 | 74.9% | 42.65 | 46.87 | **+4.23** | **+3.04** |
| 24 | 31.3% | 69.31 | 72.98 | **+3.67** | **+2.68** |
| 32 | 9.7% | 81.93 | 83.98 | +2.05 | +1.44 |
| 48 | 1.2% | 90.44 | 91.02 | +0.58 | +0.41 |

两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。

### 诚实 caveat（写法要求）
phase-var 作 per-query 失败预测器**不干净**（控 LR-severity 后 partial 塌到 ≈0，与 per-image LR 失真共线）。**GO 靠的是 interventional 结果**（ensemble +4.2 / LAT−TTA +3.0 是直接测量）。故事写成 "lattice sensitivity 是 **mechanism-level nuisance**, 不是 standalone failure predictor"，方法是对所有 severe-LR query 做 marginalization（不是预测哪张失败）。

## 核心假设

训练一个 lattice-marginalized embedding（对 lattice variants 身份稳定）+ 推理 K-marginalization，在 h=16 上比 frozen lattice ensemble **再高 +0.8~2.0 mAP** → 证明它学到了 lattice-invariance（是 method），不是 ensemble trick。

## 技术方案

### 数据流
1. 正常 ReID baseline（Market，exp260b 同配置）。
2. fine-tune：HR train image 在线生成 LR lattice variants `x^l = U(D_l(x))`，l ∈ {sub-pixel phase, bbox jitter, downsample kernel}。每图每 iter 采样 M=2-4 variants，eval 用 K=9。
3. h 混合训练 h ∈ {16,24,32}，severe-biased（但不只训 h=16）。

### LM-ReID loss
```
z^l = norm(fθ(T_l(x)));  p^l = softmax(W z^l);  z^μ = norm(mean_l z^l);  p^μ = mean_l p^l
L_id   = mean_l [ CE(p^l, y) + Triplet(z^l, y) ]
L_marg = -log[ mean_l p^l[y] ] + Triplet(z^μ, y)                 # marginal likelihood（主贡献）
L_cons = mean_l (1 - cos(z^l, sg(z^μ))) + β·mean_l KL(p^l || sg(p^μ))  # consistency to mean
L_adv  = GRL-CE(Dφ(z^l), lattice_label_l)                        # 弱：去掉 embedding 中可预测 lattice label
L = L_id + λ_m·L_marg + λ_c·L_cons + λ_a·L_adv
```
默认 λ_m=1.0, λ_c=0.2, β=0.5, λ_a=0.02–0.05（warmup 后开）。**L_adv 弱辅助非主贡献**（太强会擦身份边缘细节，必须 ablation）。

### 推理 K-marginalization
```
s(q, g) = τ·log[ 1/K Σ_l exp( cos(f(T_l(q)), f(g)) / τ ) ]
```
τ→0 接近 lat-MaxSim（主推，因 lat-MaxSim 46.9 > mean），τ 大接近 mean（消融）。

## 预期结果

**过线（决定 method vs trick）**：
- h=16：训练版 > frozen ensemble **+0.8~2.0 mAP**；> single +5~7；> TTA +2~3.5。
- h=24：稳定收益。
- h=32：允许 marginal 不负。

失败最可能原因：训练版只 ≈ frozen ensemble（没学到额外 lattice-invariance）→ 沦为 test-time ensemble trick，不成方法稿。备选投稿角度：同等 mAP 下 K 从 9 降到 3 或 single inference 保留大部分收益。

## 对照组

- single LR（canonical bicubic，固定一个）。
- 普通 K-TTA（同 K，random crop/flip/color/resize）。
- **frozen lattice ensemble**（零训练 K=9，= kill-switch 的 +4.23，这是训练版必须超过的硬线）。
- （成稿）k-reciprocal / SR-based / VPFA。

消融：marg only / marg+cons / marg+cons+adv；τ sweep；K=1/3/5/9 曲线；phase-only vs +bbox+kernel。

## 协议 / benchmark

- 合成：Market/MSMT，gallery HR，query LR h=16/24/32，canonical LR single baseline，K=9（3×3 phase 主，bbox/kernel ablation，不无限扩 K）。所有 TTA 对照 **K-matched**。
- 标准 CR-ReID（成稿补）：MLR-Market / MLR-CUHK03 / CAVIAR（PS-HRNet 用过）。
- 新指标：PRF@1（phase rank-flip rate）、Flip Entropy、LEG（lattice ensemble gain）、LOTG（lattice-over-TTA gain）、query ΔAP。按 h 分报 + paired bootstrap 95% CI + K=1/3/5/9 曲线 + compute cost。PRF 随 h（74.9%→31%→10%→1.2%）是强故事线。

## 撞车边界（novelty，codex 5 路联网）

- **VPFA**（2510.00936，最近邻）：CR-ReID = feature-space resolution direction，Vector Panning LR→pseudo-HR。hidden variable 是 resolution gap/feature direction，**不是 sampling lattice，不做 lattice marginalization**。必须正面对比。
- LRAR（2207.13037）：resolution-adaptive representation，占"resolution adaptive"词 → 我们 novelty 写 sampling-lattice uncertainty。
- RFD（2109.07871）：multi-res gallery distillation，占"resolution-invariant distillation"，没占 lattice intervention。
- BlurPool（1904.11486）：anti-aliasing/shift-invariance 已老 → **不能说首次发现 aliasing 影响**。**能声称**：首次在 LR person ReID 把采样格点作隐藏变量 + rank-level intervention 证明 + lattice-specific marginalization 解决检索身份翻转。
- FlipReID（2105.05639）：ReID 常用 flip-mean TTA → **必须反复强调非 TTA 换名**（同 K vs-TTA 控制，lattice 多 +3.04）。

## 风险与定位

operating point 低（只在 h≤24 强 +3-4 mAP，h=32 marginal）→ 主动收窄定位 **"severe low-resolution / cross-resolution ReID under sampling-lattice uncertainty"**，不写成通用 ReID 鲁棒性稿。h=32 marginal 反而支持机制（分辨率升高 → lattice uncertainty 消退）。真正风险 = 训练版打不过 frozen ensemble，那就只是 ensemble trick。

## 审查 / 训练协议

1. 本 design.md（已写）。
2. 实现 LM-ReID（插件式，config 开关，lattice aug 复用 `cvpb_lattice_killswitch.py` 的 LR 生成）。
3. Claude 广审（Opus 子代理）→ `claude_review.md`；Codex 审（`codex --search exec`）→ `codex_review.md`。**两层通过才训练**。
4. smoke（几 iter 确认各 loss 分量下降无泄漏）。
5. 训练（lab-3090，Market，h 混合，BS=64 不改，TEST.IMS_PER_BATCH 64，PYTHONUNBUFFERED=1 nohup）。
6. eval（**test.py 不用 train.py**）：LM-ReID(K marg) vs single / TTA / frozen-ensemble，h=16/24/32 分报，LEG/LOTG/PRF。
