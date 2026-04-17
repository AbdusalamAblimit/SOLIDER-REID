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
