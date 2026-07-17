# 实验 exp368: Camera-Pose Transport Operators

## 动机
- Single-Support CVaR DEAD（exp367, worst 跨 camera/pose gap 不可训练改善, oracle-headroom 墙）→ 转 codex 训练侧 Top3。
- 现有 ReID 学 camera-invariant descriptor（抹平 camera nuisance）。但 invariance 可能丢判别信息。
- codex Top3（CCF-B 6.2）：不学 invariance, 学 low-rank transport 把 descriptor 从一 camera cell 映到另一 cell 后再比（comparability operator, not invariance）。

## 核心假设
- camera 间有系统性 feature shift（非随机），train ID 可拟合 transport map W_{a→b}，test query transport 后比 gallery 更准。

## cheap kill-switch（frozen, 零训练, 豁免审查）
- frozen global feat（/tmp/ae_feats.npz query/gallery + cam, exp260b SOLIDER）
- train ID 拟合 cam pair ID-mean ridge transport W_{a→b}（cam a ID-mean → cam b ID-mean, lam=1.0）
- test query(cam a) transport 到 gallery cam b 后 cosine
- 脚本 cvpb_camtransport_probe.py

## 成功线
- transport mAP Δ>+0.5 vs baseline cosine, 且明显 > camera-centering 对照（证不只是去 camera bias）
- 按 baseline #false@10 分桶 ΔAP（控 trivial: transport 增益应在难桶更显著, 非均匀偏移）
- 若 transport 抬 → camera invariance 不够 transport 有 headroom → codex 细查 novelty + 训练版; 不抬 → DEAD 转 Top2 Counterfactual Part-Contradiction

## 对照
- baseline: 直接 cosine（无 transport）
- camera-centering: per-cam mean 减（弱对照, 证 transport 不只是去 camera bias = 仅 first-order）

## 先例风险（codex Top5）
- CamStyle/camera-aware 先例密; 新意只在"transport not invariant"。GO 后 codex 细查 novelty 避先例。

## 诚实标注
- codex 明说训练侧天花板 6.2-6.8（没 8-9, 要换数据/task）。Top3 6.2 是 cheap 验先（有 headroom 再深入）。一方向证伪→不停→下一个。
