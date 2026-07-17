# OSAC (Open-Set Spectral Over-Collapse) kill-switch 设计
（2026-06-24, B+GOPL+Hubness 三死后, reassess2 x_2 主推; 第一个 training-side 候选）

## re-frame（破解前3死根因: 它们都是 retrieval/topology-side 被 test-time 工具碾压; OSAC 是 training-side）
> 大家以为 ReID 训练越把同 ID 压紧越好。其实 **open-set ReID 的隐藏变量是 seen-ID neural collapse 过强**——训练末期类内方差坍缩、特征过度对齐到 seen-ID classifier/prototype 几何，**unseen-ID 的可迁移身份证据被低秩化/各向异性化、挤到低能谱尾部**。测试是 unseen-ID 检索，k-reciprocal/camera-aware 只能**重排已有距离**，恢复不了被训练压没的维度。**Hubness(gallery 负向 in-degree)是这个过坍缩的检索症状, 不是根因。**

## novelty 切口（待 codex 核查）
- Neural Collapse(Papyan 2020)已定义训练末期坍缩, 但**没人把"过坍缩伤 open-set ReID 的 unseen 迁移"当可测失败变量 + 训练期抗坍缩**。
- vs k-reciprocal/CA-Jaccard: 它们 test-time 重排, **碰不到训练表征**(OSAC 核心优势)。
- vs ProNet(保留 classifier 到推理作 prototype projection): 别把 classifier prototype projection 当方法。
- vs VICReg/whitening/decorrelation/AMS group-whitening: 是通用泛化零件, **主贡献不能写"decorrelation loss"**, 必须写"open-set ReID 过坍缩诊断 + 训练期 spectral anti-collapse 保 unseen 证据"。
- vs hubness(HAL/NeighborRetr): hubness 只当 readout symptom, remedy 是 training-spectral 不是 retrieval-hub-fix。

## ★零训练 kill-switch（接近 0-GPU, 先只分析现有 checkpoint, 不训练）
团队有多 epoch checkpoint(CHECKPOINT_PERIOD=20 → ep20/40/60/80/100/120): occluded_duke exp255 / market exp260b/exp030a / CARGO strong ckpt。冻结提 BNNeck embedding。

**核心量(每个 ckpt epoch 算):**
- effective rank = exp(entropy(λ_i/Σλ)) 或 (Σλ)²/Σλ²; top-PC energy λ1/Σλ。
- NC1 = tr(S_w)/tr(S_b)(类内/类间方差比, 越小越坍缩)。
- classifier-feature alignment(若 ckpt 有分类器)。
- gallery hubness H_k + query hub mass M(q)(复用 hubness 脚本)。

**核心测试:**
1. **过坍缩轨迹**: 训练后期(ep80→120) loss 继续降, 但 effective rank **下降** / top-PC energy **上升** / NC1 下降。→ 证"过坍缩"存在。
2. **坍缩↔检索失败相关**: per-query AP error ~ (query 在 top-PC 上的投影能量 / prototype alignment) 相关, **控 camera/norm/margin 后仍显著**。hub mass M(q) ~ top-PC energy 相关(证 hubness 是坍缩症状)。
3. **去坍缩诊断干预(零训练)**: 对 embedding 做 ABTT(去 top-m PC) / whitening, 看 (a)hubness 是否降, (b)raw mAP 是否可见提升。

**破坏对照(决定生死):**
- D1 随机 PC / bottom-PC removal → 必须**不如** top-PC removal(否则不是坍缩信号是噪声)。
- D2 **去坍缩 vs k-reciprocal**: 若 ABTT/whitening 只涨 raw, 但 **k-reciprocal 之后完全无残余增益** → 判死(又被 re-ranking 吃掉, 同 Hubness 教训)。**这条最关键——必须证训练表征侧有 k-reciprocal 拿不到的东西。**
- D3 控 camera/norm/margin 后 top-PC↔AP 相关消失 → 判死(普通难度代理)。
- D4 不同 backbone(Swin/ViT/ResNet)/不同数据 都要看到过坍缩轨迹(否则架构偶然)。

**通过标准:** 过坍缩轨迹明确(ep后期 rank↓) + AP error ~ top-PC energy partial 显著 + ABTT 降 hubness 涨 raw + **D1 随机PC不如top-PC + D2 ABTT 在 k-reciprocal 之后仍有残余增益 + D3 控代理后仍在**。
→ 全过(尤其 D2 残余) = 过坍缩是 k-reciprocal 拿不到的真 training-side 失败变量 → 单训练 kill-switch(强 baseline 加 OSAC: spectral floor + Top-PC Dropout, 测试仍单 embedding; ep30/60 看 effective rank↑/top-PC hub correlation↓/raw +0.5 稳; 成功线 raw +0.8~1.0 且 k-reciprocal/camera 后仍 +0.3 残余)。
→ D2 不过(去坍缩被 k-reciprocal 吃掉) = OSAC 也死, 那就接受 x_1 终判: image-level frozen 彻底收, 转视频 AG-VPReID 或把 Hubness 写成 analysis short。

## 机制草案(过了 kill-switch 才做)
OSAC: ① warmup 后 BNNeck embedding 算 batch/memory covariance spectrum; ② 轻量 spectral floor(提 effective rank/限 top-PC energy); ③ Top-PC Dropout(stop-grad 估 top PCs, 一条 loss 分支随机减 top-m PC 再 CE/triplet, 迫使身份信息不只走 dominant axes); ④ clean branch 保原 CE/triplet 避免过度反坍缩伤主干。测试不变, 单 embedding。

## 资产
多 epoch ckpt 在 lab-3090 log/。复用 cvpb_hubness/gopl kill-switch 的 extract/per_query_ap/H_k 基建。谱分析纯 numpy。
