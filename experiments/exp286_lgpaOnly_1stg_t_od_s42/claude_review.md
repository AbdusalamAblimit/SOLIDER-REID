# Claude Review — exp286_lgpaOnly_1stg_t_od_s42

**审查对象**: Phase 3-C 首个 run,LGPA-only + 1-stage PSG Tiny OD

## 审查范围

1. `design.md` — 双变量 (POSE_SKELETON_GCN False + POSE_PSG_STAGES [-1]) 相对 exp261 隔离是否干净
2. 代码改动: **无**
3. 对照矩阵位置: Phase 3-C 2×2 = (GCN on/off) × (PSG 1/2-stage) 的 "off + 1-stage" 角
4. CLI override 语法: `MODEL.POSE_SKELETON_GCN False` 与 `MODEL.POSE_PSG_STAGES "[-1]"` 的 yacs 解析,和 Phase 3-B 已验证

## 变量隔离与 baseline

- 相对 exp261 (Tiny Full Scaffold, GCN512 + 2-stage FINAL 65.9/77.4):
  - 变量 1: GCN True → False (关结构分支)
  - 变量 2: PSG stages `[-2,-1]` → `[-1]` (减少注入)
- 两变量同改,形成 Phase 3-C 2×2 矩阵的对角角点
- 直接单变量对照是 Phase 3-B exp280 (Tiny GCN512 + 1-stage) 或将来 exp287 (LGPA-only + 2-stage)

## CLI override 语法

- yacs `MODEL.POSE_SKELETON_GCN False` 可直接设置布尔开关
- `MODEL.POSE_PSG_STAGES "[-1]"` 在 Phase 3-A exp271 + Phase 3-B exp280 已充分验证
- 其他 scaffold 模块 (LGPA / OA-SD / ParAug / LOWER_BODY_OCC) 自动从 `prcv_best_tiny.yml` 继承 default True

## OOM 风险

- 相比 Phase 1 exp261 (Full Scaffold),关掉 GCN 显存减少 ~1GB (GCN head + 额外 aux loss 计算)
- PSG 从 2-stage 减到 1-stage 再省 ~0.5GB
- Tiny + Full Scaffold - GCN - 1 stage PSG 预估 < 7GB on 5060 Ti 16G (exp261 Tiny Full Scaffold 原本 ~8-10GB)
- eval 时 flip-test doubles activation,峰值 ~10GB,仍富余 > 5GB

## 与 Phase 1 共享

- 无共享,是 Phase 3-C 独立的 4 个 run (exp286/287/288/289) 中的第 1 个
- 未来可与 Phase 3-B exp280 做"semantic branch 是否吃 PSG stage 增益"对照

## 机器分配与 chain

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- srvC 原跑 exp266 在 21:27 silent exit,GPU 空闲
- srvC 上 Occ-Duke 数据完整 (4.9GB, train 22059 + query 4152 + gallery 24770)
- pretrained swin_tiny/small/base + clip_part_text_features 全齐
- exp286 完成后 auto-chain → exp287 (LGPA-only + 2-stage Tiny)

## 结论

**审查通过**。单变量 (双变量组合) ablation 干净,代码零改动,scaffold 继承 yml default 正确,风险极低。

Phase 3-C 科学目的: 通过 "关 GCN 保 LGPA" 的对照矩阵,隔离 PSG stage 数对 semantic branch 的独立贡献。和 Phase 3-B 共同构成 "PSG stage 收益归属" 的完整论证。

为 Table 4 (ablation optional) 提供 row "LGPA-only + 1-stage"。Phase 3-C Tiny 2 runs (exp286/287) 先在 srvC 完成,Small 2 runs (exp288/289) 等 Phase 3-A Small / Phase 3-B 完成后在 lab4090 继续。
