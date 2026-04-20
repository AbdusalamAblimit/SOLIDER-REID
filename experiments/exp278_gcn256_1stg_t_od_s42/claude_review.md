# Claude Review — exp278_gcn256_1stg_t_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_tiny.yml`、`config/defaults.py`、`model/pose_backbone_model.py` (PSG stage resolution + GCN head init),以及与 Phase 1 exp261 (Tiny Full Scaffold FINAL 65.9/77.4) 的消融变量对照关系。重点检查"双变量同时改"在 Phase 3-B 里是否合法、CLI override 是否足够、`POSE_GCN_HIDDEN=256` 的 SkeletonGCNHead 初始化是否会崩。

## 变量隔离与 baseline

本 exp 是 Phase 3-B 矩阵中"最精简 full-scaffold"角，**同时**改两个变量:
- `POSE_GCN_HIDDEN`: 512 → 256
- `POSE_PSG_STAGES`: `[-2,-1]` → `[-1]`

**注意**: 这是 Phase 3-B 唯一双变量改的 Tiny cell。设计里明确指出它的意义不是"干净的单变量消融",而是"构成 2×2 GCN×PSG 矩阵的对角端点"。与 exp279 (GCN256+2stg Tiny) 和 exp280 (GCN512+1stg Tiny) 一起,它们三个 vs exp261 四格子组成 2×2 交互表。因此双变量改是刻意设计的,不违反 Phase 3-B 的整体单变量原则——后者指 Phase 3-B 矩阵内任意两个 cell 相互比较只差一变量。本 cell 的"baseline"不是 exp261,而是 exp279/exp280 各自组成的边缘。

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp278_gcn256_1stg_t_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

- `MODEL.POSE_GCN_HIDDEN` 对应 `config/defaults.py:140`,int 类型,256 合法
- `MODEL.POSE_PSG_STAGES` 对应 `config/defaults.py:102`,list[int] 类型,yacs merge_from_list 用 literal_eval 解析 `"[-1]"` → `[-1]`,与 exp271/273 同语法
- `SOLVER.SEED 42` 与 yml 默认一致(redundant 但防御性保留),OK
- 余下 5 个 full-scaffold 开关 (LGPA=True / GCN=True / OA-SD=True / PLBOA=True / ParAug=True / POSE_TEST_FEAT='equal_concat' / GLOBAL_LOSS_SCALE=0.5) 全部由 yml 继承,CLI 不 override → 符合 Phase 3-B 的"只改 GCN_HIDDEN + PSG_STAGES"scaffold 定义

## OOM 风险

srvB 5060 Ti 16G + Swin-Tiny + WITH_CP=True + GCN256 (比 GCN512 显存更小)。exp261 Tiny Full Scaffold 在 srvB 已跑通不 OOM,本 exp GCN hidden 降半、PSG stage 减半 (只剩 Stage 3 两 block gate,不含 Stage 2 两 block gate) → 显存和计算量都 ≤ exp261,OOM 概率 ≈ 0。

## 与 Phase 1 共享

- Swin-Tiny backbone: 与 exp261 同
- 数据、augmentation、sampler、optimizer、lr schedule、warmup、eval period: 同
- 仅 `POSE_GCN_HIDDEN` + `POSE_PSG_STAGES` 两项差异,单独一行在 config_log 里可以精确对照

## 边界检查

- `POSE_GCN_HIDDEN=256`: `SkeletonGCNHead.__init__` 只把 hidden 当中间层维度 (layer 0 out_dim = hidden, layer 1 out_dim = feat_dim=768),hidden=256 不需是 feat_dim 的倍数,linear 层参数量按 768→256 + 256→768 独立计算,**无维度约束**。与 `POSE_GCN_LAYERS=2` 组合合法。
- `POSE_PSG_STAGES=[-1]` + `num_backbone_stages=4`: 解析为 index 3 (last stage),合法;`psg_stage_indices={3}`;Stage 3 Swin 块 2 个,psg_modules_dict 注册 `s3_b0`, `s3_b1` 两模块,其他 stage 无 PSG → 与 pose_backbone_model.py L40-71 逻辑一致。
- flip-test per-block renorm fix (commit f69b61c) 已在 srvB/srvA 生效,equal_concat dict/tensor 两路径都正确,与 Phase 3-B 的 full scaffold + OA-SD 训练端对称性破坏组合兼容。

## 机器分配与 auto-chain

srvB 3 × Tiny × 3h20min ≈ 10h,在 exp273 (Tiny 3-stage PSG Phase 3-A) FINAL 后通过 queue_on_ckpt daemon 链 exp278 → exp279 → exp280。与 Phase 3-A 的 exp273 不冲突(后者先完,前者再起)。lab4090 只跑 Small Phase 3-A 尾部 exp275/276/277 + Phase 3-B Small 三格,不与 srvB Tiny 链相撞。

## 结论

**审查通过**。本 exp 是 Phase 3-B 2×2 交互表的"双变量最弱角",设计意图明确(与 exp279/280/261 共同构成矩阵),CLI 语法正确、scaffold 继承清晰、无 OOM / 维度 / config 错配风险。可以 auto-chain 启动。

注意事项:
- 若实际 mAP 大幅低于 exp279/exp280 的最小值 → 支持"GCN cap × PSG stage 交互非平凡"论文写作
- 若 exp278 ≈ exp261 → 意味着 Tiny 下容量/stage 都过剩,论文需重新措辞 "high-capacity structural branch 必要性"
