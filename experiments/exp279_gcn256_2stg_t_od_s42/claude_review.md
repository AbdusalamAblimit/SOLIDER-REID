# Claude Review — exp279_gcn256_2stg_t_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_tiny.yml`、`config/defaults.py`、`model/pose_backbone_model.py` 的 GCN 分支初始化路径、`model/modules/skeleton_gcn.py` 的 `hidden_dim` 参数合法性。与 Phase 1 exp261 (Tiny Full Scaffold GCN512 + 2-stage FINAL 65.9/77.4) 的单变量对照校验。重点评估"唯一改 GCN cap"的隔离是否干净、yacs CLI override 单项是否足够。

## 变量隔离与 baseline

本 exp 是 Phase 3-B 矩阵中**相对 exp261 的单变量消融**:
- `POSE_GCN_HIDDEN`: 512 → 256 (唯一变量)
- `POSE_PSG_STAGES`: 保持 yml 默认 `[-2,-1]` (无 CLI override)

隔离度极干净: exp279 - exp261 的 Δ 纯粹反映 "2-stage PSG + full scaffold 下 GCN cap 由 512 → 256 的边际影响"。这是 Phase 3-B 矩阵 2×2 交互表里最容易解读的格子,也是 exp283 在 Small 上的直接对照。

论文证据链价值:
- exp279 ≈ exp261 → GCN 容量冗余,论文写 "GCN256 足够 Tiny"
- exp279 << exp261 → GCN 512 是 2-stage 的必要配套,论文写 "scalable extension 对容量敏感"
- exp279 vs exp278 的 Δ = "GCN256 下多 1 stage PSG 的边际收益"

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp279_gcn256_2stg_t_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256
```

- 只 override `MODEL.POSE_GCN_HIDDEN=256` 一项,`POSE_PSG_STAGES` 让 yml 的 `[-2,-1]` 生效,其他 full-scaffold 开关由 yml 继承
- yacs `merge_from_list` 对 int 字段无需 quote,256 解析正确 (`config/defaults.py:140` 默认 256,但 yml override 为 512,CLI 再 override 回 256 合法)
- `SOLVER.SEED 42` 与 yml 同值,redundant 但防御性保留 (queue_on_ckpt daemon 可能需要统一复制 CLI),OK
- 相比 exp278,只 override 一项,scaffold 保留更多 yml 默认,变量数最少

## OOM 风险

相对 exp261 (GCN512+2stg Tiny,已在 srvB 跑通 FINAL 65.9/77.4),本 exp GCN hidden 降半 → 模型参数减少约 (512*512 + 512*768 - 256*768 - 256*256) × layers,但主内存占用仍由 Swin-Tiny backbone + PSG stage 2/3 + 3-view ParAug 决定。GCN hidden 降半在整体显存占用中是小数,不会触发 OOM,也不会让训练变显著慢 (GCN forward/backward 只占 < 5% FLOPs)。srvB 5060 Ti 16G + WITH_CP=True 足够。

## 与 Phase 1 共享

与 exp261 完全同: Swin-Tiny, Occ-Duke, SGD lr 8e-4, 120 epoch, seed 42, flip-test, equal_concat, GLOBAL_LOSS_SCALE=0.5, LGPA/OA-SD/PLBOA/ParAug 全开, POSE_PSG_STAGES=[-2,-1], POSE_TEST_FEAT='equal_concat'。唯一差: `POSE_GCN_HIDDEN 512→256`。可直接算 Δ。

## 边界检查

- `POSE_GCN_HIDDEN=256` 下 `SkeletonGCNHead` 初始化:
  - Layer 0: `Linear(768, 256)` + `LayerNorm(256)`
  - Layer 1: `Linear(256, 768)` + `LayerNorm(768)`,**zero-init** (identity start)
  - 无维度约束,256 无需是 feat_dim=768 的因子
- `POSE_PSG_STAGES=[-2,-1]` (yml 默认): 解析为 `{2, 3}` (Swin-Tiny num_stages=4),Stage 2 Swin 块 6 个、Stage 3 Swin 块 2 个 → 8 个 PSG 模块,与 exp261 一致
- flip-test per-block renorm fix (commit f69b61c) 已部署,equal_concat + GCN per_part=False 下 gcn_feats 只 1 元素,bug 影响为 0 (已由 exp262 re-eval 验证),本 exp 也免疫

## 机器分配与 auto-chain

srvB 第 2 个 Phase 3-B slot (exp278 → **exp279** → exp280),queue_on_ckpt daemon 监听 exp278 的 `transformer_120.pth` 出现后自动启动。3h20min × 3 runs 总 10h,在 4-30 deadline 前 (10 天) 完全来得及。与 lab4090 上的 Small Phase 3-A/B 不冲突。

## 结论

**审查通过**。本 exp 是 Phase 3-B 最干净的单变量消融 (vs exp261 仅改 GCN_HIDDEN),证据价值明确,CLI 语法最精简只 override 必要字段,scaffold 继承依赖 yml 默认 (已在 exp261/262 Phase 1 FINAL 验证正确)。可 auto-chain 启动。

注意事项:
- 若 exp279 mAP < exp261 且 exp278 mAP ≈ exp279,说明 GCN256 是瓶颈,不是 PSG stage 数
- 若 exp279 > exp278,说明 PSG 2-stage 下 GCN 容量瓶颈表现不同,是论文 interaction 的硬证据
