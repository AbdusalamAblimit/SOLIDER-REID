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

Small 在 5060 Ti 上训练时 GPU memory ~7.8 GB / 16 GB。用户确认: **`MODEL.WITH_CP: True`（gradient checkpointing，已在 `configs/occluded_duke/prcv_best_base.yml:14` 打开）下，Base 显存占用也只 6–8 GB**，在 16 GB 上完全够。显存不是瓶颈。

**真正的瓶颈是时间预算**: 距 2026-04-30 deadline 11 天，Tiny/Small 6 个 run 三机并行约 22–28h；Base 3 个 run @ with_cp（额外 ~25% 前向开销）单 run ~17–18h，三机并行约 18h。Phase 3 消融留约 30h。Total ~66–76h 可达成，但必须按优先级排。

**选项**:
  A. 迁到 5060 Ti 立刻并排 Base（会挤占 Tiny/Small 进度，Phase 3 消融压缩）
  B. PRCV 主表只上 Tiny + Small 两行，Base 留 rebuttal / camera-ready
  C. 先把 Tiny + Small × 3 数据集 6 个 run 打完，再把 Base 3 个 run 排进同三台；Phase 3 按实际剩余时间调整

**选择**: C

**理由**:
1. Phase 1 主表目前 3 台都在跑 Tiny/Small，立刻并排 Base 会拖慢现有进度，反而风险更大
2. `exp260b Base = 73.9/83.2`（本地 3090 旧协议，含 `WITH_CP: True`）已存在，即便最终新协议 Base 没跑完，Base 行仍有 reference
3. Tiny/Small 6 run 预计 2026-04-20 晚至 2026-04-21 全部完成，届时再按剩余时间决策是否上 Base；三台 5060 Ti 每台都能容纳 Base

**执行结果**:
- `todo.md` Phase 1 表中 Base 3 run（exp263/266/269）状态标 DEFERRED，但机器列改为"srvA/B/C 任一"（不再绑定 local）
- Phase 4 multi-seed 三个配置短期改为 Small 优先；若 Phase 1 Base 跑完，Phase 4 可补 Base 的 multi-seed
- Tiny/Small 6 run 完成后立即评估：是否把 Base 3 run 并入 Phase 1，还是进 Phase 3 消融
