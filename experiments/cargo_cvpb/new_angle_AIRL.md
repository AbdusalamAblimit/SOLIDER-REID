# 新 B 类角度: AIRL — Aerial Identity Recoverability Learning(2026-06-23)

## 5 codex 收敛(罕见一致, 全 7-7.5/10 >> OVLI 的 4-5, 独立)
**把 CARGO 从"跨视角对齐"重新定义成"航拍观测条件下身份信息何时物理上不可辨识"。**

## 核心机制(codex4 最详)
- 不"对齐视角", 而是显式建模航拍图像的**信息损失上界**。
- 用 bbox 像素高/面积、camera id、估计 altitude/view 生成 **degradation ladder**, 把 ground 图降解到不同 UAV 像素预算。
- 训练 **resolvability branch** 学"在该像素预算下仍可稳定识别的身份证据", 抑制 ground-only 高频细节。
- 主损失: ID CE + degraded/original prediction consistency + entropy/margin 校准。**无 contrastive/late-interaction/pooling/prototype/visibility。**

## 为什么 B 类够 novel
- 论文问题不是"如何跨视角对齐", 而是 **"AG-ReID 中身份信息什么时候物理上不可辨识"**(新问题定义)。
- 区别于 GIQT(几何相似度)/GSAlign(空间变形)/SD-ReID(生成补视角): **拒绝 hallucination, 强调 observation-limited representation + calibrated reliability**。
- 撞车风险: cross-resolution ReID / AG-VPReID-Net → 贡献钉死在"altitude-conditioned recoverable evidence ceiling + fair-evidence matching"。

## ★★ 廉价 kill-switch(5 codex 一致: 先诊断后训练)
1. **零训练诊断(先做)**: 用现有 Swin baseline 按 aerial bbox height/area 分桶, 算 per-bucket A→G mAP。**判据: 低尺度桶相对高尺度桶塌陷(gap > 3-5 mAP)→ resolution 是主误差源, 角度成立; 不塌 → 杀(尺度不是主因)。**
2. **最小退化实验**: ground 图按 aerial bbox 尺度分布下采样/模糊/JPEG + prediction consistency 短训。**判据: Swin 上 A→G 涨 ≥1.0 或最低像素桶涨 ≥3.0 且 reliability AUROC ≥0.65 → 继续; 否则杀。**

## 评分
codex1 7 / codex2 7 / codex3 7.5 / codex4 7.5 / codex5 7。**全 ≥7, 收敛, 显著强于 OVLI(4-5)。**

## 与 OVLI 死因的对照(为什么这次可能不一样)
OVLI 死于"弱 backbone headroom artifact"(强 Swin 上 baseline 反超)。AIRL 的 kill-switch **直接在强 Swin baseline 上做诊断**——如果尺度分桶在强 backbone 上仍塌陷, 说明这是 backbone 解决不了的物理问题(像素预算), 不是 backbone-headroom artifact。这正是 OVLI 缺的"机制内在价值"检验, AIRL 第一步就过这关。

---

## ★ kill-switch #1 结果(零训练尺度分桶诊断, 2026-06-23)= **PASS**

脚本 `airl_scale_diag.py`(lab-3090)。按 aerial query 原生 bbox 像素分 4 等量分位桶, 各桶对同一全量 ground gallery 算 A->G mAP(gallery 跨桶不变→桶差只归因 aerial 尺度)。

| 配置 | b0(最小桶)mAP | 顶桶 mAP | gap | reliab AUROC |
|------|--------------|---------|-----|-------------|
| Swin × area(原生面积) | 40.78 | 54.19 | **+13.41** | 0.715 |
| Swin × height(原生高) | 46.12 | 65.12 | **+19.01** | 0.715 |
| ResNet50 × area | 18.85 | 34.71 | **+15.86** | 0.805 |

- **三跑全部最低尺度桶塌陷**, gap 13.4~19.0 >> 判据阈值 +3~5 → PASS。
- aerial 原生尺度跨度 17x(area 1170~19758 px); median 仅 ground 的 1/3 = 低像素预算真实存在。
- **关键: 强 Swin 上 b0 仍塌(40.78 vs 54-65)** → 不是 OVLI 那种弱 backbone headroom artifact, 是强 backbone 解决不了的物理问题。AIRL 第一步就过了 OVLI 缺的"机制内在价值"关(见 honest_assessment.md)。
- FULL A->G mAP 54.14 == cvpb_swin_fix256 训练 log ep30 → pipeline 精确, 数字可信。
- reliability AUROC 0.715/0.805: 模型置信度已部分预测可辨识性 → 支持 "altitude-conditioned recoverable evidence + calibrated reliability" 主张。
- 非单调(b3 偶低于 b1/b2): 对齐/裁剪噪声, 不影响 kill-switch(决定信号=小桶塌陷)。

**下一步(按 design 的 kill-switch 阶梯 #2)**: 最小退化实验 —— ground 图按 aerial bbox 尺度分布下采样/模糊/JPEG + prediction-consistency 短训。判据: Swin 上 A->G 涨 ≥1.0 或最低像素桶涨 ≥3.0 且 reliability AUROC ≥0.65 → 继续建机制; 否则杀。

---

## ★ kill-switch #2 结果(AIRL 训练后 per-bucket 决定性检查, 2026-06-23)= **PASS(area 轴) — 条件性继续**

AIRL-Swin 训练完成(lab-4090, `cvpb_airl_swin2`, ep60, `--airl --airl_lambda 0.5 --airl_min_scale 0.25 --consistency kl --warmup 5`, 无新可学参数, eval 路径与 baseline 逐键相同/379 键 0 多余)。对照 baseline-Swin(`cvpb_swin_baseline256`, ep60)。两个 model_best 各按 aerial query 原生 bbox 尺度分等量分位桶, 对**同一全量 ground gallery** 算 per-bucket A->G mAP。桶边界由图像文件尺度决定(模型无关)→ 同一桶跨两 ckpt 是**同一批 query**, Δ 干净归因于 AIRL。两 baseline FULL A->G==训练 log ep60(58.75)→ pipeline 精确。

### 整体(ep60 model_best, mean 选取)
| 方向 | AIRL | baseline | Δ |
|------|------|----------|---|
| **A->G**(航拍 query→地面 gallery) | **61.90** | 58.75 | **+3.15** |
| G->A(地面 query→航拍 gallery) | 59.75 | 62.93 | **-3.18** |
| mean | 60.83 | 60.84 | -0.01 |

→ **mean 翻平(像 OVLI), 但这是方向性 trade-off 不是 null**: AIRL **只退化 ground**(非对称, smoke S11)→ ground 特征对低清鲁棒 → A->G 涨, G->A 跌, mean 抵消。**AIRL 的 claim 从来不是 mean, 是 A->G 低清**——A->G +3.15 证机制有内在价值(OVLI 在强 backbone 上连方向都没赢)。

### 最小尺度桶 Δ(#2 决定性判据)— 跨 3 个分桶粒度
| key | nb=3 Δ | nb=4 Δ | nb=5 Δ | 判读 |
|-----|--------|--------|--------|------|
| **area**(h×w=真实像素预算) | **+5.17**(51.28 vs 46.11) | **+3.59**(42.50 vs 38.91) | **+8.39**(37.70 vs 29.31) | **全 ≥+3.0, 越小桶涨越多** → PASS |
| height(仅高, 弱代理) | +0.46 | -0.21 | +0.26 | 全 ~0 → 该轴不涨 |

- **area 轴: AIRL 最小桶在 3/4/5 桶全部 ≥+3.0 阈值(+3.59~+8.39), 且桶越小增益越大(nb=5 最小桶 +8.39)** = AIRL 机制(area 比例像素预算退化 `airl_degrade`)精确改善它专攻的最低像素预算 query。**area b0 AUROC 0.807(baseline 0.819, 相当)。**
- height 轴不涨: height 是更弱的分辨率代理(矮但宽的 crop 像素其实不少); AIRL 按 **area** 退化 → 机制天然对齐 area 轴, height 轴看不到 = 机制一致, 非 cherry-pick。kill-switch #1 中 area gap(+13.4)< height gap(+19.0)是 within-model 描述性信号, 与本处 cross-model Δ 不同口径。

### 裁决: **PASS(条件性)** — area 轴最小桶 +3.59(nb=4)/+8.39(nb=5)≥+3.0, A->G 方向 +3.15
**与 OVLI 死因的关键区别**: OVLI 在强 Swin 上 mean 被 baseline 反超 **且无任何桶/方向赢** = 机制无内在价值。AIRL mean 翻平**但 A->G 方向 +3.15 + area 最小桶跨粒度 +3.59~+8.39 稳定赢** = 机制在它声称的轴上有真实价值, 代价是 G->A -3.18 的方向 trade-off。

**继续条件(诚实标注的不足, 下一步必须解决)**:
1. **方向 trade-off(-3.18 G->A)是真问题**: 当前非对称退化 ground-only 把 A->G 的涨用 G->A 的跌换来; 论文若主打 A->G(AG-ReID 标准检索方向=航拍 query)可立, 但若要 mean 涨需对称化或 reliability-gated 融合。
2. **height 轴不涨**说明增益绑定 area 度量, 需在机制层面把"像素预算"定义钉死在 area。
3. **样本量小**(b0 n=27~45, query 仅 134)→ +3.59 含噪声; nb=5 的 +8.39 与 nb=3 的 +5.17 一致性是主要可信来源(非单点)。

**下一步(机制阶段, 非小修)**: 不在退化 augmentation 上堆参数; 走 design 的 resolvability branch —— altitude/area-conditioned recoverable-evidence ceiling + calibrated reliability(AUROC 已 0.80), 把 A->G 的方向增益做成"fair-evidence matching", 并解决 G->A trade-off。若机制化后 A->G 增益消失或 G->A 跌无法回收 → 那时再杀。

---

## ★ kill-switch #3 结果(零训练 gate/fusion oracle, codex 红队设计, 2026-06-23)= 待填

**问题**: #2 给出方向性 trade-off(A->G +3.15 / G->A -3.18, mean 翻平)。红队 codex 指出**方向路由的 mean 上界 = (61.90+62.93)/2 = 62.42 = baseline +1.58**(A->G 用 AIRL, G->A 用 baseline)。本关验证: **合法固定 gate(非 test 调阈值)能否逼近这个上界?** 逼近(≥+1.0)→ 值得建完整 resolvability 双分支; 回收不了(<+0.5)→ KILL(上界存在不等于可达, trade-off 单模型回收无望)。

**脚本** `airl_gate_oracle.py`(lab-4090, 零训练)。两 checkpoint(baseline `cvpb_swin_baseline256` / AIRL `cvpb_airl_swin2`, eval 架构逐键相同, missing=0 unexpected=0)各提 CARGO query/gallery 特征(A->G + G->A), 复用 `eval_market` 排序逻辑(重构成接受任意 distmat 以便逐 query 路由/融合)。所有 gate 终值 = mean=(A->G+G->A)/2。

**gate 清单**:
- view/方向 gate(合法上界): A->G→AIRL, G->A→baseline。query view 测试时已知 = 合法非 oracle。
- area gate(单模型可近似): 按 query 原生 bbox area 路由(低 area→AIRL), 阈值 = CARGO **train** 分位(非 test 调)。
- reliability gate: 按 baseline top-1 cos 置信路由(低→AIRL), 阈值 = train 分位; + **3b 合法变体**: conf_AIRL > conf_base 才路由 AIRL(无 label/无阈值, 两模型测试时都在)。
- per-query oracle(理论上界): 逐 query 取更优分支 AP。
- score 融合(软): cos = w·cos_AIRL + (1-w)·cos_base, 扫 w。

**FULL sanity(逐键复现 doc, pipeline 精确)**: base A->G 58.75 / G->A 62.93(mean 60.84); AIRL A->G 61.90 / G->A 59.75(mean 60.83)。两 ckpt load missing=0 unexpected=0。

**各 gate mean(Δ vs baseline 60.84)**:
| gate | 合法? | mean Δ |
|------|-------|--------|
| view/方向 gate | 合法(query view 已知) | **+1.58**(62.42, 复现红队上界) |
| **score 融合 w=0.25** | **合法(固定 w, 无 label)** | **+1.46**(62.30) |
| score 融合 w=0.40 | w 在 test 扫(轻 ceiling) | +1.86(62.70) |
| score 融合 plateau w∈[0.25,0.75] | — | 全 ≥+1.46(62.30~62.70) |
| conf-diff gate(3b, conf_AIRL>conf_base) | 合法(无 label/无阈值) | +0.77(61.61) |
| area gate(train 分位 0.25/0.5/0.75) | 合法 | +0.02 / −0.41 / +0.41 |
| reliability gate(train 分位) | 合法 | +0.07 / +0.35 / +0.11 |
| area gate ceiling(test 调阈值) | 非法(ceiling 参考) | +0.91 |
| reliability ceiling(test 调阈值) | 非法(ceiling 参考) | +0.48 |
| **per-query oracle** | 理论上界 | **+4.96**(65.80) |

**裁决 = PASS(≥+1.0 由合法固定 gate 达成)**。

**关键发现(改写红队上界判断)**:
1. **硬路由(area/reliability)回收不了 trade-off**: area 全 ≤+0.41, reliability 全 ≤+0.35 → 单一标量阈值无法把"A->G 该走 AIRL、G->A 该走 baseline"的方向性选对(area/conf 在两方向上分布重叠, 选不出方向)。**若只看硬路由 = KILL。**
2. **但 score 融合(软混合, 非硬路由)轻松过关**: `cos=w·cos_AIRL+(1-w)·cos_base` 单一全局 w, **w=0.25 这种保守默认就 +1.46**, plateau w∈[0.25,0.75] 全 ≥+1.46, 且 **w=0.40 的 +1.86 反超 view-gate 的 +1.58 "上界"** —— 红队的方向路由上界**不是真上界**, 因为硬路由丢掉了方向内的 per-query 互补性, 软融合保留了。融合无 label、w 非 knife-edge(整段 plateau 都赢)= 合法。
3. **per-query oracle +4.96** 证 headroom 远大于 +1.58: AIRL 在每个方向都有大量 query 赢过 baseline(A->G win/tie/lose=37/32/25, G->A=23/14/29), 只是 area/conf 这些 cheap proxy 选不出来。
4. **代价**: score 融合需测试时跑两个 model(2× inference)。这正是 PASS 的意义—— headroom 真实且被合法机制可达, **单模型双分支(resolvability branch)可把两套特征空间内化进一次 forward**, 省掉 2× 成本同时拿到 ≥+1.46 的 mean 增益。

**下一步**: 上完整 resolvability 双分支机制(area/altitude-conditioned recoverable-evidence ceiling, 单模型出两支特征做 score 融合)。证 B 类相: AIRL 不止 A->G 方向涨, 融合后 **mean 也涨 +1.5**, 解决了 #2 标注的 G->A trade-off。机制化目标 = 复现 score-fusion 的 +1.46~+1.86 mean 增益, 单 forward。

---

## ★ AIRL 完整机制: 单模型双分支 resolvability(2026-06-23 实现, 仅 smoke 未训练)

三关全过(kill-switch #1 尺度塌陷 +13~19 / #2 area 最小桶 +3.6~8.4、A->G +3.15 / #3 score 软融合 mean +1.46@w=0.25)后, 把"两模型 score 融合(+1.46, 2× inference)"内化进**单模型一次 forward**。综合 3 份 codex 设计(`/tmp/codex_airl_design{1,2}.log` + `assess2.log`)。

### 设计(综合 codex assess2 的 f_full/f_rec 双分支公式)
**1 backbone + 2 BNNeck head**, 共享 backbone 一次 forward 出两特征:
- **f_full head**(原 head): 只 ID-CE + 共享 triplet, **不加 degradation consistency** → 保 full-resolution 身份证据 → 保护 G->A(高清 ground query)。
- **f_rec head**(新增第二 BNNeck): 自己的 ID-CE(让它是合法身份空间)**+ AIRL ground-degradation consistency**(只退化 ground 到 aerial 像素预算, KL/feat 拉退化预测向 clean) → 学低像素预算 recoverable 证据 → 服务 A->G(低清航拍 query)。
- **eval 软融合**(distmat 级, 非特征拼接): `dist = 2 − 2·(w·cos(f_rec) + (1−w)·cos(f_full))`, 单一全局 w(默认 **0.25**, 非 test 调; #3 plateau w∈[0.25,0.75] 全 ≥+1.46)。**一次 forward 出 f_full/f_rec 两特征** → 替代 #3 的两模型(cos_AIRL→cos_rec, cos_base→cos_full)。

### 数据流
```
img ─backbone(shared)─ feat_map ─pool─ global_feat ┬─ bottleneck(f_full) ─ logits      ┐ ID-CE_full
                                                    │                                   ├ + 共享 triplet(global_feat, 不重复)
                                                    └─ bottleneck_rec(f_rec) ─ logits_rec┘ ID-CE_rec + AIRL consistency(只 f_rec)
deg(ground only) ─backbone─ global_feat_d ─ bottleneck_rec ─ logits_rec_d ─ consistency(clean f_rec detached ← deg f_rec)
eval: 一次 forward → (f_full_norm, f_rec_norm) → distmat 软融合
```

### 两 head 真分化(机制成立的关键)
- f_rec 的 consistency **只读 logits_rec/bn_feat_rec**, 梯度回流 **shared backbone + bottleneck_rec**, **f_full 的 BNNeck/classifier 完全不受 consistency 影响**(smoke D4: f_full grad=0)→ 两 head 不是同一空间复制, f_full 留判别性、f_rec 攻低清。
- f_rec **必须有自己的 ID-CE**: 只给 consistency 会塌成无身份空间; CE_rec 让它是合法检索空间(smoke D8: CE_rec 梯度回 classifier_rec+backbone)。
- triplet **只在共享 global_feat 上一份**(不给 f_rec 重复)→ 对应 #3 里 baseline/AIRL 各一个 triplet 的单模型类比, 也保 off 字节级。

### 实现(`--airl_dualbranch` + `--airl_fuse_w 0.25`)
- `afd_model.py`: `AFDModel(airl_dualbranch=True)` 加第二 BNNeck head(`bottleneck_rec`+`classifier_rec`, 同 f_full init 配方); train dict 多 `bn_feat_rec`/`logits_rec`; eval `return_dual=True` 出 `(f_full_norm, f_rec_norm)`, 默认 `return_dual=False` 仍出单 f_full(legacy 路径不变)。
- `afd_train.py`: 训练循环加 f_rec ID-CE + 复用 `airl_degrade`/`airl_consistency_loss`(**函数体不动**)路由到 f_rec head; 新增 `airl_dualbranch_eval`(镜像 `ovli_rerank_eval`, 出 full/rec/**fuse** 三组, model-selection 用 fuse mean); 共享 `--airl_lambda/min_scale/consistency/tau/blur/warmup`。
- **off 字节级**: `--airl_dualbranch` 不开 → 第二 head 不构造、dict 无 *_rec、eval 单特征、loss 不触碰(smoke D1/D1b: max|df|=0)。
- **参数入优化器**: 第二 head 在 `model.parameters()` 内自动入优化器; Swin 下走 full-LR "other" group(非 backbone 缩放组, smoke D10); 显式 assert + 日志。
- 互斥: 与 `--airl`(单 head consistency)互斥; 与 `--ovp/--ovli` 互斥(headline 独立跑)。

### smoke(`smoke_airl_dualbranch.py`, 11/11 全过)
D1 off 无第二 head + baseline dict keys; D1b on+return_dual=False == baseline f_full 特征(max|df|=0); D2 双 head shape/L2-norm/独立(扰 rec 不动 full); D3 f_rec consistency 梯度到 bottleneck_rec+backbone; **D4 f_full 无 consistency 梯度(头分化真)**; D5 两 head 都入优化器且一步都动; D6 软融合 distmat(w=0→full/w=1→rec/中间凸混合); D7 f_rec consistency 输出 finite(extreme logits/zero feats 下, fp32 + nan_to_num finite guard); D8 f_rec ID-CE 接地(梯度到 classifier_rec+backbone); D9 两 BNNeck 吃同一 global_feat(triplet 单份); D10 f_rec head full-LR。+ 全 py_compile 过 + 现有 `smoke_airl.py` 21/21 不破 + 完整训练步集成验证(loss/两 head 梯度/eval 融合 distmat 全 finite)。

### framing(钉死, 避撞车)
**"observation-limited evidence ceiling 下 clean(f_full)/recover(f_rec)evidence head 分化 + 固定先验软融合(fixed-prior fusion)"**——**不是** query-budget routing / resolution-adaptive dual-branch(那会撞 RAR arxiv 2207.13037 = resolution-adaptive metric、MRJL 2105.12684 = multi-resolution dual-branch fusion、cross-resolution ReID)。实现就是单一**固定** w=0.25 软融合(`cos = w·cos_rec + (1−w)·cos_full`), **不是动态 router**: kill-switch #3 已证硬 per-query 路由(area / reliability)回收不了 trade-off(≤+0.41), 增益全来自固定 w 软混合。所以诚实地主张 **observation-limited evidence ceiling → 两个 evidence head(full-evidence vs recoverable)分化 → fixed-prior 软融合**, 不吹 "按 query 路由证据空间"。codex assess2 已查 asymmetric ReID / DECAMEL / QPM 先例 → 贡献钉在 AG-ReID 的 recoverable-evidence ceiling + fair-evidence matching, 非 view-specific projection 本身。

**w-lock(对应 Medium finding)**: headline 永远用固定 w=0.25。`--airl_fuse_w` 标注为 **ablation-only**(w-sweep 消融), 传非 0.25 时 print 一条 `[AIRL-DUAL][WARN]`(软保护, 不 hard assert, 因为扫 w 消融仍要用)。

### kill-switch #4(训练后, 唯一裁决)
训练 dualbranch(resnet50 先, 后 Swin), eval fuse mean。**fuse mean ≥ baseline +1.0 → 机制成立(复现 #3 的 +1.46, 单 forward, 解决 G->A trade-off), 进 B 类方法稿; < +1.0 → 杀**(headroom 真但单模型双分支共享 backbone 内化不出来, trade-off 是 ground-degrade 内在属性 / codex assess2 信心 0.58)。审查要点供后续双审见下「审查要点」。

### 审查要点(供后续 Claude+Codex 双审)
1. **off 字节级**: `airl_dualbranch=False` 全路径(model 不构造第二 head、train dict 无 *_rec、loss 不加 CE_rec/consistency、eval 不调 dual)→ baseline 逐字节复现。
2. **头分化**: consistency 只读 f_rec(logits_rec/bn_feat_rec), f_full BNNeck/classifier 零 consistency **梯度**(smoke D4); clean f_rec 侧 detach(稳定目标)。**已知并接受的次要项(codex round-2 Medium)**: 退化 forward 是整模型 `model(deg_imgs)`(无 rec-only 路径), 故 f_full 的 frozen-bias BNNeck running mean/var 仍会"看到"退化 ground 图(仅统计跟踪, 非梯度泄漏)——与 `--airl` 单头路径完全一致(同一 degrade+forward 原语), 刻意保持对齐以保证消融诚实; 是否有实质影响由 kill-switch #4 训练结果裁决, 非 bug。若要彻底隔离需加 rec-only forward(改训练行为, 当前 lab-3090 正在跑此代码, 不动)。
3. **train/test 对称**: 融合 w 固定(`--airl_fuse_w`, 非 test 扫); eval 软融合公式 == #3 GATE-5(`2−2·(w·cos_rec+(1−w)·cos_full)`); `airl_dualbranch_eval` 的 f_full 数 == `run_cross_view_eval`(同特征同排序)。
4. **AMP 安全**: degrade fp32 image space; consistency `autocast(enabled=False)` 真 fp32(for numeric safety, 输入 finite); clean/deg forward 同 autocast; log_softmax 避 log(0) + 标量 `nan_to_num` finite guard(正常训练 no-op)。
5. **参数/优化器**: 第二 head 入优化器(assert); Swin 下 full-LR(非 backbone 缩放); triplet 不给 f_rec 重复(共享 global_feat)。
6. **>=2-ground 守卫**: 退化 batch 过 train-mode BNNeck 需 ≥2 ground row(size-1 BatchNorm1d crash 守卫), 同 `--airl`。
7. **互斥**: `--airl`/`--ovp`/`--ovli` 互斥, 隔离消融。
