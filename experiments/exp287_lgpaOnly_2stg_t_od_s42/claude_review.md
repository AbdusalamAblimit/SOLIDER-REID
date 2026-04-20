# Claude Review — exp287_lgpaOnly_2stg_t_od_s42

**审查对象**: Phase 3-C 第 2 个 run,LGPA-only + 2-stage PSG Tiny OD

## 审查范围

1. `design.md` — 单变量 (POSE_SKELETON_GCN False) 相对 exp261 的隔离
2. 代码改动: **无**
3. 对照矩阵位置: Phase 3-C 2×2 = (GCN on/off) × (PSG 1/2-stage) 的 "off + 2-stage" 角
4. exp287 相对 exp286 = PSG stage 1→2 单变量

## 变量隔离与 baseline

- 相对 exp261 (Tiny Full Scaffold, GCN512 + 2-stage FINAL 65.9/77.4) 单变量: GCN True → False
  - 这是 Phase 3-C 矩阵里**最纯的单变量 ablation** (只改 GCN,PSG stage 保持 default)
- 相对 exp286 单变量: PSG_STAGES `[-1]` → `[-2,-1]` (同 semantic branch 下的 stage 数对照)

三角对照:
- exp287 vs exp261 = 移除 GCN 后的损失 (2-stage)
- exp287 vs exp286 = 同 semantic branch 下 PSG stage 增益
- exp287 vs exp280 (Phase 3-B Tiny GCN512 + 1-stage): "LGPA-only 2-stage" vs "GCN512 + 1-stage",对比两种"仅 1 个结构改变"的效果

## CLI override 语法

- `MODEL.POSE_SKELETON_GCN False` 单变量 override,scaffold 其他 (LGPA/OA-SD/ParAug/LOWER_BODY_OCC/PSG_STAGES default `[-2,-1]`) 自动继承
- 这是 Phase 3 里最简洁的 CLI override (1 行)
- yacs 直接接受布尔值

## OOM 风险

- 相比 exp261 (Full Scaffold),关 GCN 省 ~1GB 显存
- 保留 2-stage PSG,两个 spatial gate,显存同 exp261 PSG 部分
- Tiny + Full Scaffold - GCN 预估 ~7-9GB on 5060 Ti 16G,eval peak ~10GB,安全

## 与 Phase 1 共享

- 无共享,是 Phase 3-C 4 个 run 中的第 2 个 (exp286 的 2-stage 对照)
- exp287 vs exp286 差距直接回答 Phase 3 核心问题 3: "2-stage PSG 的收益是偏 structural 还是 semantic branch 也吃"

## 机器分配与 chain

- 机器: srvC (同 exp286)
- 启动方式: exp286 完成后 auto-chain via queue_on_ckpt daemon
- 数据: srvC 上 Occ-Duke 4.9GB + pose_data 完整
- 预计时长: ~3h30min (5060 Ti Tiny + Full Scaffold - GCN)

## 结论

**审查通过**。单变量 ablation 与 exp286 配对构成 Phase 3-C 最小闭环。代码零改动,yml default 继承正确,OOM 风险低。

对 Table 4 (ablation optional) 提供 row "LGPA-only + 2-stage",和 exp286 "LGPA-only + 1-stage" 组成 Tiny 2 行。
