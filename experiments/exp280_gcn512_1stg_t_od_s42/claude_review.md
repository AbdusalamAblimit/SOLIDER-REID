# Claude Review — exp280_gcn512_1stg_t_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_tiny.yml`、`config/defaults.py`、`model/pose_backbone_model.py` 的 PSG stage resolution 路径。本 exp 是 phase3_design.md L153 明确标记的 "Phase 3-B 的核心最小闭环" 之一 (GCN512 1-stage vs 2-stage Tiny),即 "若时间不够跑完 8 runs 的 Phase 3-B,至少必须跑完 GCN512 1-stage vs 2-stage 两组 = 4 runs" 的关键 cell。审查重点是"仅改 PSG stage 数"的隔离度。

## 变量隔离与 baseline

本 exp 是 Phase 3-B 矩阵中**相对 exp261 的单变量消融**:
- `POSE_PSG_STAGES`: `[-2,-1]` → `[-1]` (唯一变量,Stage 2 PSG 模块整组移除,仅保留 Stage 3)
- `POSE_GCN_HIDDEN`: 保持 yml 默认 512 (无 CLI override)

隔离度极干净,是 Phase 3-B 最有论文价值的单格: exp261 - exp280 的 Δ 纯粹是 "高容量 GCN 下 Stage 2 PSG 注入是否不可或缺"。旧 exp255 (GCN512+2stg Small) vs exp255b (GCN512+1stg Small) 观察到 2-stage 显著高于 1-stage (详见 `results.md:1405`),本 exp 跨 backbone (Tiny) 直接验证此结论是否稳定。

论文价值:
- exp280 << exp261 → "2-stage PSG 是高容量结构分支的必要条件",论文保留 2-stage 作 scalable extension
- exp280 ≈ exp261 → 2-stage PSG 在 Tiny 上不显著,降级为 "default setting" (与 Small 上不同 → 提供 backbone-scale 分析)

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp280_gcn512_1stg_t_od_s42 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

- 只 override `MODEL.POSE_PSG_STAGES="[-1]"` 一项,`POSE_GCN_HIDDEN=512` 由 yml 继承,其他 full-scaffold 开关也由 yml 继承
- `"[-1]"` 语法正确: yacs merge_from_list 对 list 字段用 literal_eval 解析,`"[-1]"` → `[-1]` (与 exp271 `"[-1]"` / exp272 `"[-2,-1]"` / exp273 `"[-3,-2,-1]"` 同机制)
- 注意: quote 必须保留防 shell 吞 `-1` 成 flag,design.md 里已正确使用双引号 ✓

## OOM 风险

相对 exp261 (GCN512+2stg Tiny),本 exp 仅关 Stage 2 PSG (Swin-Tiny Stage 2 共 6 个 block → 6 个 PSG 模块不注册),显存和计算量都 < exp261,OOM 概率 0。srvB 5060 Ti 16G + WITH_CP=True 绰绰有余。

## 与 Phase 1 共享

与 exp261 完全同除一个字段: `POSE_PSG_STAGES [-2,-1] → [-1]`。可直接算 Δ,无任何其他混杂变量。equal_concat test_feat + GCN + LGPA + OA-SD + PLBOA + ParAug 全保留。

## 边界检查

- `POSE_PSG_STAGES=[-1]` 解析: `num_backbone_stages=4` → `idx = 4 + (-1) = 3` → `psg_stage_indices = {3}`。Stage 3 Swin 块 2 个 → 注册 `s3_b0`, `s3_b1` 两 PSG 模块 (参数共 ~2 × 17*64 bottleneck 级别,极小)。Stage 0/1/2 无 PSG,其 SwinBlock forward 走原始路径。
- `POSE_PFM_HIDDEN=64` yml 已设,`POSE_PSG_SPATIAL` defaults.py=False (未 enable 3×3 depthwise),符合 exp261 的 scaffold。
- flip-test per-block renorm fix 已部署,与本 scaffold 完全兼容。

## 机器分配与 auto-chain

srvB 第 3 个 Phase 3-B slot (exp278 → exp279 → **exp280**),queue_on_ckpt daemon 链接。预计 3h20min,总 Tiny 三 runs ~10h on srvB。在 Phase 3-A 的 exp273 FINAL (预计 ~2026-04-20 23:37 CST) 之后启动,无冲突。

## 结论

**审查通过**。本 exp 是 Phase 3-B 的关键"最小闭环"之一,设计 clean、对照干净、CLI 极简,scaffold 继承稳定。论文级别 Table 3 必须有的一行。可 auto-chain 启动。

注意事项:
- 本 exp 的 Δ vs exp261 是 Tiny 版 "1-stage vs 2-stage PSG @ GCN512",与 exp284 (Small) 的对应 Δ 构成跨 backbone 一致性证据
- 若两者同号且 Small 更显著 → "容量越大,Stage 2 PSG 贡献越明显",支持 scalable extension narrative
