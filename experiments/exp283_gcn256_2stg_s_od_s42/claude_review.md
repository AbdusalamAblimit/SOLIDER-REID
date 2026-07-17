# Claude Review — exp283_gcn256_2stg_s_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_small.yml`、`config/defaults.py`、`model/pose_backbone_model.py` GCN 分支路径、`model/modules/skeleton_gcn.py` hidden_dim 实现。与 Phase 1 exp262 (Small Full Scaffold FINAL 73.8/83.1) 的单变量对照,以及与 Tiny exp279 的跨 backbone 对照 (exp283 ↔ exp279 结构相同,只换 backbone)。

## 变量隔离与 baseline

本 exp 是 Phase 3-B Small 侧**相对 exp262 的单变量消融**:
- `POSE_GCN_HIDDEN`: 512 → 256 (唯一变量)
- `POSE_PSG_STAGES`: 保持 yml 默认 `[-2,-1]` (无 CLI override)

隔离度最干净。Phase 3-B 矩阵里 exp283 是 Small 侧与 exp279 (Tiny) 对应的单变量格子,可直接 pair-wise 对比 "GCN256 vs GCN512 @ 2-stage PSG @ backbone ∈ {Tiny, Small}"。这一对 Δ 是论文 Table 3 最直接的"GCN capacity effect @ full scaffold"行。

论文证据链:
- exp283 vs exp262 = "GCN cap 从 512 降到 256 在 Small + 2-stage PSG 下的 Δ"
- exp283 vs exp282 = "Small + GCN256 下 2-stage 比 1-stage 多多少" (与 Tiny exp279 vs exp278 对比)
- exp283 Δ 与 exp279 Δ 同号同幅 → "GCN 容量瓶颈跨 backbone 稳定",结论稳健;反之则需论文解释 backbone-scale 差异

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp283_gcn256_2stg_s_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256
```

- 最简 CLI: 只 override `MODEL.POSE_GCN_HIDDEN=256` 一项,`POSE_PSG_STAGES=[-2,-1]` 从 yml 继承
- 其他 full-scaffold (LGPA=True / SKELETON_GCN=True / OA-SD=True / LOWER_BODY_OCC=True / PARALLEL_AUG=True / POSE_TEST_FEAT='equal_concat' / GLOBAL_LOSS_SCALE=0.5 / POSE_GCN_LAYERS=2) 全部 yml 默认
- yacs int 字段 256 直解,无 quote 问题
- `SOLVER.SEED 42` 与 yml 同,redundant 但 defensive,OK

## OOM 风险

lab4090 RTX 4090 24GB + Swin-Small + Full Scaffold。exp262 在原 srvA 5060 Ti 16GB 已跑通 (decisions.md L3615: 73.8/83.1 re-eval),4090 裕度极大。GCN hidden 降半实际影响 Linear(768, 256) 替代 Linear(768, 512) + Linear(512, 768) 替代 Linear(256, 768) 的 2 层结构,参数量约省 1.1M × GCN 分支占比 < 5%。OOM 概率 0。

## 与 Phase 1 共享

与 exp262 完全同除 `POSE_GCN_HIDDEN` 一字段。Swin-Small backbone, Occ-Duke, SGD lr 8e-4, 120 epoch, seed 42, flip-test, equal_concat, GLOBAL_LOSS_SCALE=0.5, LGPA/OA-SD/PLBOA/ParAug 全开, POSE_PSG_STAGES=[-2,-1], POSE_TEST_FEAT='equal_concat'。

## 边界检查

- `POSE_GCN_HIDDEN=256`: SkeletonGCNHead Layer 0 `Linear(768, 256)` + LayerNorm(256),Layer 1 `Linear(256, 768)` + LayerNorm(768) zero-init。与 defaults.py L140 默认 256 一致 (yml override 为 512 后 CLI override 回 256,等价默认)。无维度约束 → 合法
- `POSE_PSG_STAGES=[-2,-1]` 解析为 `{2, 3}` (Swin-Small 4 stages),Stage 2 Swin 共 18 个 block (swin_small config) → 18 个 PSG 模块 + Stage 3 2 个 PSG 模块 = 20 个。比 Swin-Tiny 的 `{Stage 2: 6 block, Stage 3: 2 block}` 多,PSG 参数量更大但仍远小于 backbone,不影响显存
- lab4090 pose_data 完整性: decisions.md L3745-3752 已确认 4 splits 全字段齐,visibility/visibility_binary 数值差 ~5e-5 ULP 级,与 srvB 等价,可直接使用
- flip-test per-block renorm fix (commit f69b61c): 适用 equal_concat 路径,与 Small + Full Scaffold + OA-SD 训练端对称性破坏的组合兼容

## 机器分配与 auto-chain

lab4090 第 2 个 Phase 3-B slot (exp277 → exp282 → **exp283** → exp284)。预计 1h42min (Small on 4090 比 5060 Ti 快约 2.5x)。queue_on_ckpt.sh PYTHON 环境变量已在 decisions.md L3769 fix 支持 conda python。

## 结论

**审查通过**。本 exp 是 Phase 3-B Small 侧最干净的单变量消融 (vs exp262 仅改 GCN_HIDDEN),与 Tiny exp279 构成直接跨 backbone 对照,论文 Table 3 的关键一行。CLI 语法最精简,scaffold 继承稳定,lab4090 已验证可跑 Full Scaffold Small。可 auto-chain 启动。

注意事项:
- 若 exp283 ≈ exp262 → GCN 容量在 Small 下冗余 (SRGB_best_small.yml 里 GCN_HIDDEN 默认可降为 256)
- 若 exp283 << exp262 → GCN 512 在 Small 下必要,保留 yml 当前设置
- 若 exp283 Δ 与 exp279 Δ 方向相同但幅度差异大 → 论文可写 "容量效应随 backbone scale 非线性",这是 nuanced 论文素材
