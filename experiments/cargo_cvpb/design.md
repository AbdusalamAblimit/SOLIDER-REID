# 实验 CVPB: CARGO 航拍-地面 跨视角原型桥接 + 局部证据匹配

> 战略重思后选定(5 死角后, 3 codex strat 强收敛)。★★方法论转向: **放弃"验证 confound 是真依赖"(杀光一切), 改 empirical 涨点导向**——在 CARGO baseline 上试 +module 能否 beat = 真 kill-switch。这是 167 篇库真实配方(命名 gap + 2 模块 + 多数据集 SOTA 表 + framing)。

## 动机
- CARGO 航拍-地面 baseline(resnet50 BoT)mean A↔G mAP **32.48%**, VDT SOTA ~50% → **大 headroom**。AGPReID 活赛道(VDT CVPR24 / SeCap CVPR25 / GSAlign NeurIPS25 / ViSA CVPR26), 范式都是"命名视角/几何/语义 gap + 2 模块 + CARGO/AG-ReID 表", **不证 confound 依赖**。
- ★framing(避开那堵墙): **不是"模型被 view confound 骗"**(那已被 trained 模型 handle)——而是 **aerial-ground 单张图的身份证据在两视角下不一一对应**, 这是 ID loss/triplet 的全局监督约束不到的真实 gap(局部对应/跨视角证据补全)。

## 核心假设(empirical, 非 confound)
全局 cosine 之外加 (1) 跨视角原型对齐 + (2) 局部 token 集合匹配, 能补全单图缺失的对侧视角证据 → mAP 涨。**判据 = 涨点, 不是 confound 真伪**。

## 技术方案
- **模块1 OVP-Mem(Opposite-View Prototype Memory)**: 每 train pid 维护 aerial / ground 两个 EMA 原型。样本除 CE+triplet 外, aerial image 拉近同 pid 的 **ground** 原型(InfoNCE: sim(z, P[y, opp_view])/τ 分类到 y)。≠死掉的 CV-triplet(那是 batch 内跨视角正对稀缺假设, 已证 88% batch 有; 这里是**全局原型级跨视角监督, 覆盖 hard tail**)。
- **模块2 Local Token MaxSim**: feature map 切 K 个局部 evidence token, 测试 `global cosine + β·双向 MaxSim(tokens)`, 训练 token-level ID/SupCon + diversity 防坍缩。**不强制同部位对齐**(non-correspondence set matching)。≠GSAlign(无 TPS warp)/ ViSA(无 expert graph)/ SeCap(无 prompt)。

## ★ kill-switch(empirical, 分阶段, 现成 baseline)
1. **零训练 token-MaxSim probe**(最便宜, 先跑): CARGO baseline checkpoint 抽 feature map, 测 `global cosine` vs `global + 局部 MaxSim hybrid` 的 A↔G mAP。**+0.5~1.0 mAP** = 局部匹配有料 → 继续。
2. **OVP-Mem only 训练**: ep10/20 vs baseline 同 epoch, **≥+1.0** 继续。
3. **OVP+Local 训练**: final **≥35-36 mAP**(baseline 32.48)→ 进方法稿; 接近 38+ → 扩 Swin/VDT 级主表。
- **不涨 = 换 module**(empirical, 不纠结 confound 真伪)。这是新 approach 的核心: 直接试涨点。

## 对照 / 消融
baseline 32.48 → +OVP-Mem → +Local-MaxSim → +both; K(token 数) / β / τ 敏感性; EMA momentum。

## novelty 切开(查重)
- VDT(view decoupling + orthogonal): 我们不做 view-related/unrelated 减法解耦。
- SeCap(adaptive prompt + local refine): 不做 prompt recalibration。
- GSAlign(LTPS 几何 warp + visibility mask): 不做 TPS / 不叫 visibility mask。
- ViSA(view-aware expert graph): 不做 expert/graph。
- SD-ReID(diffusion 生成 all-view): 不用 diffusion。
- 我们的: **cross-view local evidence densification via opposite-view prototype + non-correspondence set matching**。

## 数据 / venue
- CARGO(lab-3090 现成)。补 AG-ReID.v2 多数据集表。
- venue: AGPReID 活, ICME/ACCV/BMVC 稳, 强则冲 CVPR-tier 子方向。

## ★ kill-switch 进度(2026-06-22)
- **#1 零训练 MaxSim probe: 弱 PASS +0.86**(8×4 grid β=1.0 → 33.34 vs baseline 32.48; pipeline 验证 global cosine 精确复现 32.48; 增益集中 aerial-query A→G 32.90→34.18 **+1.28**, G→A 平~32)。局部 token 集合匹配有料但弱。
- **双审**: Claude review 无 Critical, **H1 训练动力学已修**(λ warmup `ovp_warmup=10` + inited 列数日志, 防冷启动梯度尖峰), 审查通过。⚠️**H2/M5 novelty 撞车**: OVP-Mem(对侧视角原型+InfoNCE)≈ **PDPA(2025 同 CARGO)/ CMPC(CVPR22)/ MBCE(AAAI23 VI-ReID)** → **OVP 当组件不当 headline, 用 Local-MaxSim 集合匹配(non-correspondence)差异化**。Codex review 进行中。
- **#2 OVP-Mem 训练完成: 🎯 final 50.11 mAP(R1 54.93)@ep60, baseline 32.48 → +17.6, 达 VDT SOTA~50!** 轨迹 28.35→36.32→39.18→43.83→48.76→50.11。在 resnet50 BoT(弱于 VDT 的 ViT)上做到 SOTA 量级。★**方向铁证金矿**(代码双审无泄漏: 原型训练期/eval 测试 pid 不相交; +17.6 达 SOTA 待用户多 seed 复核)。但 OVP 撞 CMPC → **headline 走 OVLI**(见下), OVP 降为强 ablation。

## 备选(此线不涨则转)
- strat_3: Camera Residual TTA(source-free open-camera test-time adaptation, Market→MSMT/CARGO OOD, 贴 DART3/TEMP)——OOD 方向, 也绕开那堵墙。
- strat_1: CARGO View-Conditioned Token Evidence Matching(和本方案 module2 同源)。

---

# OVLI: Opposite-View Late-Interaction Evidence Alignment(★headline, 5-codex 收敛)

## 为什么从 OVP 转 OVLI
- **OVP-Mem 的硬伤是 novelty 撞车**(claude/codex 双审都标 High): per-view EMA prototype + opposite-view InfoNCE ≈ **CMPC(CVPR22)/MBCE(AAAI23)/PDPA(2025 同 CARGO)** 近乎逐字同构, 当 headline 过不了 novelty review。
- **但 OVP 训练验证了方向有料**: baseline 32.48 → OVP ep30 **39.18(+大)**, 证明"跨视角身份证据对齐"有 **BIG headroom**(不是 confound 噪声, 是真信号)。
- **OVLI 保留方向、换掉撞车的机制**: 把 per-pid prototype contrast 改成 **token-set late-interaction(ColBERT/MaxSim 风格)的 sample-to-sample 跨视角检索 loss**——无 prototype/memory/EMA, 直接局部 token 集合**部分匹配**。

## framing(差异化 headline)
**跨视角身份证据是"部分 token-set 匹配"问题, 不是"全局原型对齐"问题。** 航拍-地面单图在两视角下**无 1-1 部位对应**(俯视看头肩/平视看全身)。全局原型(OVP/CMPC)对"缺失区域"一律惩罚, 把不可对应的部分也算进相似度; **partial MaxSim 让"能对上的 token"承担相似度, 对不上的不拖累**——这正是 late-interaction 的 retrieval 语义, 契合跨视角证据稀疏/不对齐的本质。

## 核心假设
全局 cosine 之外, 加 **opposite-view sample-to-sample 的 (global cos + 双向 MaxSim) 混合检索 loss**(supervised-contrastive), 让 encoder 学到"对侧视角下能局部匹配上的身份证据" → mAP 涨。判据 = 涨点。

## 技术方案(已实现, `afd_train.py --ovli`)
1. **token 抽取(复用 maxsim_probe 配方)**: hook `model.layer4`(GeM 前 spatial map, 16×8)→ `adaptive_avg_pool2d` 到 grid(默认 8×4=32 token)→ flatten → **新增 1×1 conv proj 到 256d + 逐 token L2-norm**。
   - ★**proj 是新可学参数**, `OVLIHead.proj`(Conv2d 2048→256), **已加进 optimizer**(`list(model.parameters())+list(ovli.parameters())`, 且有 assert 自检 proj 在 optimizer.param_groups 里)。这是与 OVP(无新参数)的关键结构差异。
   - hook **不 detach**, 梯度回流 layer4→proj。
2. **opposite-view retrieval loss**(`OVLIHead.loss`): batch 内, anchor(view v)的 positive = 同 pid 的 **opposite-view(1-v)** 样本, negative = 其它 pid 的 opposite-view 样本; **同视角样本完全排除出候选**(纯跨视角目标)。
   - `score(i,j) = α·cos(g_i,g_j) + (1-α)·sym_MaxSim(tok_i,tok_j)`, sym_MaxSim = 双向 mean-max `0.5*(mean_u max_s + mean_s max_u)`(对称)。α=0.5。
   - 多正样本: **logsumexp supervised-contrastive**(`L_i = -logsumexp(score(i,pos)/τ) + logsumexp(score(i,cand)/τ)`, τ=0.05), 对"有 ≥1 opp 正 且 ≥1 opp 负"的 anchor 求均值。**无 memory/EMA/prototype。**
3. **★H1 教训(从 OVP 继承)**: `--ovli_warmup`(默认 10)对 λ 线性 warmup, 防随机 proj 早期梯度尖峰。epoch 日志 `OVLI[lam_eff loss pos neg gap]` 监控塌缩/过强。
4. **train/test 对称 + AMP 安全**:
   - OVLI 是训练期 loss; eval 默认 **global-only 不变**(精确 == baseline)。`--ovli_rerank` 可选额外报 `global + MaxSim` rerank(global 与 rerank 两个数都打印), 用与训练**同一套** proj token + 双向 MaxSim。
   - OVLI loss 在 `autocast(enabled=False)` 内走**真 fp32**(cos/MaxSim/logsumexp 在 τ=0.05 下要 fp32; proj 也在 fp32 跑), 修了 OVP 审查里"注释说 fp32 但其实在 autocast 内"的 M1。
   - `--ovli` off 精确复现 baseline(`ovli=None`, OVLIHead 不构造, optimizer 只含 model params)。
   - `--ovp` 与 `--ovli` 互斥(两个不同跨视角机制, 不混跑混淆消融)。

## kill-switch / 判据(empirical)
- **#2′ OVLI only 训练**(GPU 空出后): ep10/20/30 vs baseline 同 epoch。OVP ep30 到 39.18, OVLI 至少要进同量级才说明 late-interaction 机制不输 prototype。
- final **≥35-36 mAP**(baseline 32.48)→ 进方法稿; 接近 OVP 的 39+ 且 novelty 干净 → 扩 Swin/VDT 级主表 + AG-ReID.v2 多数据集。
- 消融: α(global vs MaxSim 权重)/ τ / grid(token 数 K)/ λ / proj_dim; OVLI vs OVP 同设置对比(证 late-interaction ≥ prototype 且 novelty 更干净)。
- **不涨 = 换 module**。

## novelty 切开(查重)
- vs OVP/CMPC/MBCE/PDPA: **无 prototype/memory/EMA**, sample-to-sample late-interaction, 这正是绕开 OVP 撞车的点。
- vs ColBERT/MaxSim(IR/ReID rerank): 我们不是把 MaxSim 当 **test-time rerank**(那是 maxsim_probe kill-switch #1 已做的弱 +0.86), 而是把 opposite-view late-interaction 当 **训练期跨视角监督 loss**(让 encoder 内化局部可匹配证据), eval 默认仍 global-only。
- vs GSAlign(TPS warp + visibility): **不强制部位对应**(non-correspondence partial set matching), 无几何 warp、无 visibility mask。
- 我们的: **cross-view identity evidence as opposite-view late-interaction (partial token-set) retrieval, learned as a training objective**。

## ★★ OVLI 训练结果(2026-06-22): 45.19 mAP, headline 成立
| 方法 | mean A↔G mAP | vs baseline | novelty |
|---|---|---|---|
| baseline(resnet50 BoT) | 32.48 | — | — |
| **OVLI(headline, late-interaction)** | **45.19**(A→G 49.21 / G→A 41.16, R1 51.06) | **+12.7** | ★无 exact prior |
| OVP(ablation, prototype) | 50.11 | +17.6 | 撞 CMPC |
| **OVP+OVLI(full model)** | **52.14**(rerank 52.71, R1 57.74) | **+19.7** | OVLI 互补 prototype **+2.0 over OVP** |
- **★组合 52.14 > OVP 50.11(+2.0)= OVLI(late-interaction)与 OVP(prototype)互补不冗余**。干净消融故事: 我们提出 OVLI(novel), 它给已知 prototype 对齐加东西, full model 达 52.14。组合轨迹 29.73→29.69(plateau)→41.51→49.15→50.92→52.14(LR 衰减后大涨)。
- 轨迹: 14.91(ep10)→24.24(ep20)→...→45.19(ep60)。rerank(global+MaxSim)≈global(45.17), 收敛后 global 已够, MaxSim rerank 早期 +3 后期收敛。
- **OVLI(novel)+12.7 显著 beat baseline = headline 成立**。< OVP(50.11)4.9 但 OVP 撞 CMPC。OVLI A→G(49.21)≈OVP, G→A(41.16)弱 → α 偏 aerial-query(与 kill-switch #1 一致, MaxSim 信号集中 A→G)。
- **方法稿骨架**: headline=OVLI(opposite-view late-interaction, novel, 45.19); 强 ablation=OVP(prototype, 50.11 但撞 CMPC); baseline 32.48; + Swin/SOLIDER port 冲 SOTA + AG-ReID.v2 多数据集主表 + 消融。
- **下一步**: ① **OVLI+OVP 组合**(改互斥→both, 看是否 >50, 组合当 headline+OVLI novel 成分); ② OVLI α/τ/grid 调(close G→A gap); ③ Swin port; ④ 多数据集。

## ★ OVLI novelty 评估(5 codex 深查, 2026-06-22)
**总评: B 类方法稿 headline 可行, novelty 中偏强(~3.5-4/5), 无 exact prior(ovlinov_1 确认), 非致命撞车, 但非"发明 late-interaction"式突破。**
- **立得住**: 完整组合无 exact prior(person ReID/aerial-ground + opposite-view-only sample-to-sample + token-set MaxSim + supervised contrastive + 无 prototype/memory + 测试 global-only)。真空白 = 训练期 opposite-view-only 晚交互 loss + 测试 global-only。
- **撞车风险(必切开)**: ★AlignedReID/Learning-by-Aligning(最大, ReID 早有"训练期局部对齐、测试 global"→ 强调 OVLI 无部位对应); ColBERT/FILIP/ColPali(写"inspired by", 绝不"we propose late interaction"); CM-EMD/G2DA/CVFT(OT, OVLI 无 transport plan); DTST(token selection vs pairwise 跨视角证据)。
- **★技术硬伤(必修)**: "dustbin/对不上不拖累"夸大——现 sym_MaxSim 对全 32 token 取 mean, 未匹配 token 仍拉低分。→ (a)改表述"减弱非对应惩罚"不说"discarded"; 或 **(b)真做 dustbin(null token/top-k/thresholded MaxSim)= AG-ReID 特有设计, 修硬伤+冲更高 venue+可能涨 G→A**。
- **framing 收窄**: 重心 "training-time MaxSim 新" → "**AGPReID 的 opposite-view partial evidence supervision 新**"。headline = "Opposite-View Partial Evidence Learning for Aerial-Ground Person ReID via training-only late interaction"。
- **SOTA 别乱说**: OVLI 45.19 非 CARGO SOTA(GSAlign 61.55/ViSA +10.06), 但超 VDT 42.76/DTST 43.39 这代。写"resnet50/global-only 设定下大幅超 baseline, 达/超 VDT/DTST 量级"。
- **必做消融**: global-oppview-SupCon vs +OVLI / oppview-only vs all-view / test global vs +MaxSim-rerank / MaxSim vs OT/top-k/avg / vs AlignedReID / α-τ-grid sweep / token-match 可视化(航拍头肩→地面上身/背包)/ AG-ReID.v2 跨数据集。
- **下一步**: ① dustbin/top-k MaxSim 变体(修硬伤+AG-ReID 特有设计); ② 必做消融; ③ 组合结果; ④ Swin port + 多数据集。

## 代码审查 / 验证(2026-06-22)
- 已写 `OVLIHead`(token proj + 双向 MaxSim + opp-view supcon loss)+ `ovli_rerank_eval`(eval 期 global vs global+MaxSim 双报)。
- **本地隔离 numeric smoke test(导入仓库真实 OVLIHead, 非副本)全过**: token shape (64,32,256) 逐 token L2-norm; sym_MaxSim 对称、self 对角≈1; 正常 batch loss 有限>0 且梯度回流 **proj.weight + global feat**; all-same-view batch loss=0 不崩; 某 pid 无 opp 正样本时该 anchor 被排除、loss 仍有限、grad 有限; fp16 cached map→fp32 token; AdamW 实际推动 proj 权重。
- ast.parse + py_compile 通过。待 codex 双审 → GPU 空出后训练(当前 GPU 被 OVP 占)。

## ★★★ Swin backbone eval mAP=0.03 诊断+修复(2026-06-23)

**问题**: `cvpb_swin_ovli`(Swin-Small + OVLI, 复用 resnet50 配方 AdamW lr=3.5e-4 均一)→ eval 跨视角 mAP=0.03(≈随机), 而 resnet50 同配方 eval 正常(52.37)。

**诊断结论(不是 eval-path bug, 是训练塌缩)**:
- 失败 log 显示 **ep1-7 训练健康**(Acc 0.003→0.472, CE 7.8→3.5, OVLI gap +0.32), **ep8 Iter50 一步塌缩**(LR 升过 2.46e-4): Loss 4.16→10.36, Acc→0.01, OVLI pos≈neg≈0.49。`model_best.pth` 是 ep10(唯一 eval, 已塌)。
- `diag_swin_eval.py`(fresh model): eval 特征**正常**(8 真实图 final off-diag cos +0.24, finite, unit-norm)→ forward / `.cuda()` semantic-weight / LayerNorm / 取 tensor 路径全对。
- `diag_swin_ckpt.py`(塌缩 ckpt): `outs[-1]` off-diag cos **+0.992**, batch-chan-std 0.038(健康 2.67)= backbone 对所有输入近常数; global_feat +0.9995; 权重全 finite 无 NaN = 表征塌缩非数值溢出。

**根因**: resnet50 调出的峰值 LR 3.5e-4 AdamW 均一施加到 ~50M 参 SOLIDER Swin transformer 过大, warmup 升过 ~2.5e-4 时几步大更新把 backbone 推进常数输出退化吸引子。仓库主 SOLIDER config 训 Swin 用 SGD 8e-4 + 20ep warmup(对 transformer 温和得多)。

**修复(只动 Swin 路径)**:
- `afd_train.py`: backbone=swin_small 时 Swin backbone 单独 param-group LR×`--swin_lr_factor`(默认 0.1), heads/BNNeck/OVLI proj 保持 full LR。resnet50 字节级不变(走 else)。
- `swin_transformer.py` L1400: `w.cuda()`→`w.to(x.device)`(鲁棒性, 非根因)。
- Claude broad review: APPROVE(参数不漏不重, 冻结 BNNeck.bias 正确排除, WarmupCosineLR per-group base_lr 正确, resnet50 路径不变)。
- 验证: `diag_swin_fix`(swin_lr_factor=0.1, 14ep)看是否平稳过 ep8。
