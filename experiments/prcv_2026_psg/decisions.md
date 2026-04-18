# 当前有效决策（PRCV / PSG 线）

本文件只保留这轮重审后仍然有效的决策，不再混入更早阶段的旧判断。

## [2026-04-15 18:30] 过渡性重审结论

重审时先确认了两件事：
1. 不能继续把 `LGPA-D + MaxSim + flip` 当主创新来写
2. `exp109` 所代表的问题证据仍然强，说明旧故事并不牢

这一步的作用是**否掉旧主故事**，不是最终落地方案。

## [2026-04-15 19:20] 当前主线决定

用户确认：
1. 当前目标是先把一篇 PRCV 交出去
2. 不必强行切到全新路线
3. 所有实验都可以重跑

因此当前有效决定改为：
1. `PSG` = 主创新
2. `2-stage PSG` = 当前最终实现
3. `GCN` = structural pose branch
4. `LGPA-D / OA-SD / MaxSim / POT / flip` = supporting assets

## 当前写作决定

1. 标题 / 摘要 / 引言只讲 `PSG`
2. `2-stage PSG` 只作为最终 instantiation，不单独抬成主术语
3. `1-stage vs 2-stage` 放到消融里回答
4. `GCN` 必须明确写进方法，但不能与 `PSG` 并列成两个主创新
5. 默认测试协议包含 `flip-test`
6. `MaxSim` 必须在结果表里单独占一行，不能和默认主结果混写

## 当前实验决定

当前最重要的不是再想新故事，而是补三类实验：
1. 基础 `PSG` stage 消融
2. 高容量 `GCN` 分支依赖性消融
3. 视时间补 semantic branch 依赖性消融

## 当前 benchmark 决定

1. `Occluded-Duke` 仍是主 benchmark
2. `Occ-PTrack` 可以作为补充 benchmark 加入
3. `Occ-PTrack` 的最公平对标对象是 `KPR w/o prompt`
4. 不把 `KPR with prompt` 设为最低门槛
5. `Occ-PTrack` 只建议跑当前最强主配置，不为其展开完整新消融

## [2026-04-19 03:00] 决策 — 本地 3090 挂了，Phase 1 Base 3 run 推迟

**上下文**: 本地 3090（`phase1_design.md` 原定跑 Base 的 exp263/266/269 三个 run）已挂，不可用。另有 3 台 5060 Ti 16G 三台（`srvA/B/C`）在继续 Phase 1 前 6 个 run（Tiny + Small × 3 数据集）。

Small 在 5060 Ti 上训练时 GPU memory ~7.8 GB / 16 GB，Base 参数量约为 Small 的 1.5–1.7×，BS=64 @ 384×128 大概率会逼近或超 16 GB；铁律禁止下调 BS。

**选项**:
  A. 迁到 5060 Ti 强上 Base（风险 OOM 或要破 BS 铁律）
  B. PRCV 主表只上 Tiny + Small 两行，Base 留 rebuttal / camera-ready
  C. 先把 Tiny + Small × 3 数据集 6 个 run 打完，回头再看显存与时间

**选择**: C

**理由**:
1. 距离 2026-04-30 deadline 还有 11 天，Tiny/Small 6 run 估算总耗时 60–70h（三机并行），Phase 3 消融留约 30h，仍可在 deadline 前出全稿
2. Base 的先验数据已有 `exp260b = 73.9/83.2`（本地 3090 旧协议），即便新协议 Base 不跑，论文主表 Base 行也可引旧协议作为 reference
3. 若 Phase 3 消融跑完还剩时间，再评估是否把 Base 上 5060 Ti（届时可选择跑 192×64 输入或 Swin-Base@ 256×128 降分辨率，但这属于单独决策不在此处落）

**执行结果**:
- `todo.md` Phase 1 表中 Base 3 run（exp263/266/269）状态改为 DEFERRED
- Phase 4 multi-seed 三个配置也从"Base 排 srvC"改成"Small 排任意"（原计划 Base 的 multi-seed 改走 Small）
- 若时间允许再做 Base，单独开一条决策条目说明当时选项与约束
