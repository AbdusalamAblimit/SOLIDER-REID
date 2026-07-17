# Claude Broad Review — Phase 1 Main Table (共 9 exp)

审查范围：configs/{occluded_duke,market,occluded_posetrack}/prcv_best_{tiny,small,base}.yml 共 9 个 + Phase 0 flip-test interaction。

**Verdict: NEEDS_ATTENTION** (两项操作性 High，非正确性问题)

## Critical
None.

## High

H1. Swin-Base + WITH_CP + POSE_PARALLEL_AUG (4 views) + OA-SD on 5060Ti 16G：Base 首 epoch OOM 风险。exp260b 历史 3090 运行 OK，但 5060Ti 16G 需要监控。若 OOM：关 OA-SD 或 PARALLEL_AUG，不改 BS。

H2. OUTPUT_DIR 默认 /hy-tmp/log/... 是为远程设计，local 3090 需在启动时 CLI override OUTPUT_DIR ./log/。

## Medium

M1. 16 项 scaffold 一致性检查全部通过（TRANSFORMER_TYPE, PRETRAIN_PATH, POSE_DATA_DIR, DATASETS.NAMES, PSG_STAGES=[-2,-1], GCN_HIDDEN=512, OA_SD=True, LGPA_DETACH=True, SEMANTIC_WEIGHT=0.2, STRIDE_SIZE=[16,16], WITH_CP=True, FLIP_TEST=True, SEED=42, BIAS_LR_FACTOR=2, WARMUP=20 等）。

M2. Occ-PTrack dataset class 解析 c{digits} 正则正确（多位 cam id）；train=17898, query=2581, gallery=10831。_val_merged 首次运行自动生成。

M3. Market 的 OA-SD 因为没 PLBOA，teacher/student view 1 几乎相同；蒸馏信号弱但不出错。可以接受作为一致性。

M4. flip-test + equal_concat 不做 per-block renorm，smoke test 证实 +0.9 mAP（exp255 ckpt 上），方向正确。

## Low

L1-L8. OA-SD 参数跨数据集统一（可接受）；PLBOA 在 Occ-PTrack 开（数据已 occluded 但一致性保持）；POSE_PFM_ENABLED 默认 False 即使 POSE_PFM_HIDDEN=64 存在（无 bug，命名小瑕疵）；DEVICE_ID=('0')、BIAS_LR_FACTOR=2、WARMUP_EPOCHS=20 都与 exp255 scaffold 一致。

## Phase 0 flip-test interaction

model.eval() 切到推理模式后再调 _extract_feat_flip，BN/Dropout 安全。flip_batch 对 heatmaps/keypoints/scores 做 L-R swap 正确（FLIP_PAIRS 定义在 datasets/pose_dataset.py）。dict feat (cvk/maxsim) 字段级 average，tensor feat 整体 average。

## Summary

9 个配置结构完全一致于 exp255 scaffold，变量只有 backbone + dataset + Market PLBOA 开关。Phase 0 改动在 .eval() 路径生效正确。两个 High 是运营监控项不阻塞训练。

```
审查通过
```
