# exp246b Tiny + LGPA-D + GCN 双分支 + OA-SD (重跑)

之前 exp246 在 ep83 crash (GPU 竞争)。重新启动。

对照: exp244 (LGPA-D only): 65.3/75.7
对照: exp191 (GCN only): 63.2/75.4

## 检查点

### [03:04] 检查点 #1

ep1 iter120. 训练正常。lgpa_assign=7.20, oa_sd=0.35。
同时远程启动 exp245 Small LGPA-D + OA-SD 满血版。
**决策**: 等 ep10 eval

### [03:09] 检查点 #2

本地 ep2, ETA 5h8m. 远程 ep1 进行中.

### [03:19] 检查点 #3

本地 ep6 (ETA 5h19m), 远程 ep2 (ETA 15h, WITH_CP 慢)。
**决策**: 等本地 ep10 eval (~11min)

### [03:32] ep10 eval

**ep10: 41.9/55.5** — 与 exp246 (41.9/55.5) 完全一致! 复现成功。
vs exp244 ep10 (42.1/55.3): -0.2/+0.2 — GCN 中性。
**决策**: 继续等 ep20

### [03:40] Cron check — 两台都在跑

本地 exp246b: ep13, ETA 4h59m。
远程 exp245 Small: ep4, ETA 14h29m (WITH_CP 慢)。
**决策**: 等本地 ep20 (~18min), 远程 ep10 (~45min)

### [03:52] ep16

ep20 eval ~11min.

### [04:01] ep20 eval — 复现!

**ep20: 52.4/65.3** — 与 exp246 完全一致!
vs exp244 (51.0/63.9): +1.4/+1.4 — GCN 有贡献。
### [04:30] ep30 eval — 再次复现!

**ep30: 57.2/71.1** — 与 exp246 完全一致!

远程 exp245: ep10=7.6% — 远低于之前的 50.3%。原因待查（旧 OUTPUT_DIR 可能有残留 checkpoint 导致之前 50.3% 不可靠）。

### [04:41] ep34

ETA 4h13m.
**决策**: 继续等 ep60/120 final

### [04:59] ep40 eval — 再次复现!

**ep40: 60.8/73.9** — 与 exp246 完全一致! (exp244 = 60.8/73.5)

### [05:30] ep50 eval — 完美复现!

**ep50: 63.0/75.6** — 与 exp246 完全一致! 第3个完美复现的 checkpoint。
exp246b 至此验证: ep10/20/30/40/50 全部与 exp246 精确匹配。

### [05:41] ep54

ETA 3h7m. ep60 eval ~17min. 远程 ep5 (ETA 14h)。
Agent 审查结论: 7.6% 不是代码 bug, 是 OA-SD 训练不稳定性 + LGPA features 稀释。

### [05:59] ep60 eval

**ep60: 63.2/75.4** — 完美复现! (与 exp246 63.2/75.4 一致)

全部 6 个 checkpoint 完美匹配: ep10/20/30/40/50/60.
**决策**: 等 final (ep120), 同时等远程 ep10 eval

### [06:10] ep64

ETA 2h37m.

### [06:22] ep67 + 远程 ep10 done!

远程 ep10 完成, eval 进行中!

### [04:18] ep25

本地 ETA 4h27m. 远程 ep9 (ETA 14h1m)。
远程 ep10 eval 即将到来。
**决策**: 等远程 ep10 eval (~8min), 本地 ep30 (~14min)
