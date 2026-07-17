# Claude Review — exp284_gcn512_1stg_s_od_s42

**Review round**: v1 (Phase 3-B 启动前广范围审查)
**Reviewer**: Opus 子代理 (Broad Review 制度)
**Date**: 2026-04-20

## 审查范围

覆盖 `design.md`、`configs/occluded_duke/prcv_best_small.yml`、`config/defaults.py`、`model/pose_backbone_model.py` PSG stage resolution 路径 (L40-71)。本 exp 是 phase3_design.md L153 标记的 "Phase 3-B 的 Small 核心最小闭环",与 Tiny exp280 构成 GCN512 下跨 backbone 的 "1-stage vs 2-stage PSG" 对照,直接回答旧 exp255 vs exp255b (`results.md:1405`) 观察到的"高容量 GCN 下 2-stage PSG 更优"是否稳定。

## 变量隔离与 baseline

本 exp 是 Phase 3-B Small 侧**相对 exp262 的单变量消融**:
- `POSE_PSG_STAGES`: `[-2,-1]` → `[-1]` (唯一变量)
- `POSE_GCN_HIDDEN`: 保持 yml 默认 512 (无 CLI override)

隔离度最干净,论文价值最高的 Small 侧格子。直接验证:
- exp255 vs exp255b 的差距是不是"2-stage PSG 是 GCN512 必要配套"(旧数据 `results.md:1405` "exp255b ≈ baseline — 2-stage PSG 是 GCN512 发挥的关键")
- 该结论在 Phase 1 新协议 (flip-fix, 新 OA-SD EMA) 下是否复现

论文证据链:
- exp284 << exp262 → "高容量 GCN 下 2-stage PSG 不可或缺",论文保留 2-stage 作主要 instantiation
- exp284 ≈ exp262 → "Small 下 2-stage 降级为 default",method section 重新措辞
- exp284 Δ 与 exp280 Δ (Tiny) 同号同幅 → 跨 backbone 一致,结论强
- Small Δ 比 Tiny 显著 → "高容量 backbone 上 PSG 多 stage 注入回报越大",论文 scale 分析素材

## CLI override 语法

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp284_gcn512_1stg_s_od_s42 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

- 最简 CLI: 只 override `POSE_PSG_STAGES="[-1]"` 一项
- yacs list 字段: quote 保留 `"[-1]"` 防 shell 吞 `-1`,literal_eval 解析为 Python list `[-1]`,与 exp271/273/280 同语法
- `POSE_GCN_HIDDEN=512` 继承 yml 默认 (Small yml L28),不需显式 override
- 其他 full-scaffold 开关全 yml 继承,scaffold 定义稳定

## OOM 风险

lab4090 RTX 4090 24GB + Swin-Small + GCN512 + 1-stage PSG。相对 exp262 (2-stage PSG) 仅关 Stage 2 PSG 注入 (Swin-Small Stage 2 共 18 个 block → 18 个 PSG 模块不注册,参数省 18 × 17*64 ≈ 20K),显存和计算都 ≤ exp262。OOM 概率 0。

## 与 Phase 1 共享

与 exp262 完全同除 `POSE_PSG_STAGES` 一字段 (从 `[-2,-1]` → `[-1]`)。Swin-Small Occ-Duke SGD lr 8e-4 120 epoch seed 42 flip-test equal_concat GLOBAL_LOSS_SCALE=0.5 GCN_HIDDEN=512 LGPA/OA-SD/PLBOA/ParAug 全开。Δ 可直接归因 PSG stage 2 注入。

## 边界检查

- `POSE_PSG_STAGES=[-1]` 解析: `num_backbone_stages=4` → `idx = 4 - 1 = 3` → `psg_stage_indices={3}`。Stage 3 Swin-Small 共 2 个 block → 注册 `s3_b0`, `s3_b1` 两 PSG 模块,其他 stage 0-2 无 PSG
- `POSE_GCN_HIDDEN=512`: SkeletonGCNHead Layer 0 `Linear(768, 512)` + LayerNorm,Layer 1 `Linear(512, 768)` + LayerNorm zero-init,与 exp262 一致
- POSE_PSG_SPATIAL=False (defaults.py L103),POSE_PFM_HIDDEN=64 (yml L19),POSE_PSG_PART=False → 纯 PSG 模式,不触发 PosePSGPartModel 路径
- flip-test per-block renorm fix (commit f69b61c) 已部署,equal_concat + GCN + LGPA 测试路径兼容,bug 影响 ≈ 0 (exp262 re-eval 已验证)

## 机器分配与 auto-chain

lab4090 第 3 个 (最后) Phase 3-B slot (exp277 → exp282 → exp283 → **exp284**)。预计 1h42min × 3 runs = ~5h total on lab4090。queue_on_ckpt daemon 链,exp283/transformer_120.pth 出现即启动 exp284。与 srvB 上的 Tiny Phase 3-B 链并行,互不干扰。

## 结论

**审查通过**。本 exp 是 Phase 3-B Small 侧的关键"最小闭环",直接回答 Phase 3-B 设计的核心问题 (`"2-stage PSG 在 GCN512 下是否不可或缺"`),与 Tiny exp280 构成跨 backbone 一致性证据。CLI 极简 (单 override),scaffold 继承干净,lab4090 容量裕度极大,无 OOM / 维度 / config 错配风险。可 auto-chain 启动。

注意事项:
- 本 exp 完成后,exp262 (73.8/83.1) vs exp284 的 Δ + exp261 (65.9/77.4) vs exp280 的 Δ 两对数字组成 Phase 3-B 论文 Table 3 的最核心 4 行
- 建议 FINAL 后立即用 `test.py` 跑 `maxsim` test 模式(与 Phase 1 exp262 做 head-to-head 对比),回答 "2-stage PSG 的收益是否会被 MaxSim 抹平"
- 若需要精确复现 exp255 vs exp255b 的历史差距 (约 Δ = +1.3/+1.5 FINAL),本 exp 的 Δ 应接近该值;若显著缩小,反映 Phase 1 新协议 (OA-SD + PLBOA + LGPA + ParAug) 改变了 GCN cap vs PSG stage 的互补关系
