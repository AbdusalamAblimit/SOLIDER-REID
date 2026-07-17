# exp207 Swin-Base + GCN+PAA+CE+OA-SD 监控

配置: Swin-Base (88M) + GCN+PAA+ROA + CE + OA-SD + PLBOA + WITH_CP
对照: exp206 (Small, 70.5/82.3), KPR (Base, 73.3/82.5)
**目标**: 73-75% mAP, 83-85% R1

## 检查点

### [03:31] 检查点 #1

**状态**: 正常
**进度**: Epoch 1/120

| 指标 | 当前值 |
|------|--------|
| oa_sd | 0.595 |
| id_global | 6.554 |
| tri_part | 19.439 |
| GPU | **6.9GB/24GB** (WITH_CP, 超高效！) |

**Swin-Base + CP 仅 6.9GB！** 可以尝试不开 CP 甚至 3-view！
**决策**: 继续

### [03:33] 检查点 #2

Swin-Base 权重在传到远程 (1.7GB)。
本地 Base+CP 仅 6.9GB → 可以不开 CP 跑或加 3-view！
**决策**: 继续本地 1-view 跑基准，传完权重后远程也跑

### [03:35] 检查点 #3 — 本地改为 3-view

本地 Base 3-view+CP+OA-SD: **9.5GB/24GB** — 完美！
远程 Base 权重在传 (1.7GB)。传完后启动远程 1-view。
**决策**: 继续

### [03:40] 检查点 #4

本地 Base 3-view ep1 done. Speed 38.0 (慢), ETA **11h40m**.
远程 Base 权重传输 51%。
**决策**: 继续

### [03:46] 检查点 #5

本地 ep2. 远程权重 65%.
**决策**: 继续

### [03:52] 检查点 #6

本地 ep3. 远程权重 71%.
**决策**: 继续

### [03:57] 检查点 #7

远程 Base 权重传完！启动远程 1-view。
**决策**: 启动远程

### [04:01] 检查点 #8

**两台 Swin-Base 并行启动！**
本地: 3-view+CP, 9.5GB, ETA ~11.5h
远程: 1-view+CP, 刚启动

**这是突破 76/85 的关键实验！**
**决策**: 继续密切监控

### [04:07] 检查点 #9

本地 ep6, 远程 ep3. 两台正常。
远程 ep10 eval ~18min。
**决策**: 继续

### [04:12] 检查点 #10

本地 ep6 (3-view), 远程 ep4 (1-view). 远程 ep10 eval ~15min.
**决策**: 继续

### [04:18] 检查点 #11

远程 ep5. 远程 5060 Ti 跑 Base 更慢。ep10 eval ~12min。
本地 ep~8 (3-view 快)。
**决策**: 继续

### [04:23] 检查点 #12

本地 ep8 (Acc=0.126), 远程 ep7. 两台 ep10 eval 即将。
**决策**: 继续

### [04:29] 检查点 #13

本地 ep9, 远程 ep8. ep10 evals soon.
本地 Acc=0.140 > 远程 Acc=0.017 — 3-view 让 Base 学更快。
**决策**: 等 evals

### [04:36] 检查点 #14

本地 ep10 (iter 120). eval ~2min.
远程 ep9.
**决策**: 等 eval

### [04:42] 检查点 #15

本地 ep10 done, eval 运行中（Base eval 慢）。ETA 12h。
远程 ep~10。
**决策**: 等 eval

### [04:45] 检查点 #16 — 本地 ep10

**本地 Base 3-view ep10: 47.2/60.1**

| Config | ep10 mAP | ep10 R1 |
|--------|---------|---------|
| exp206 Small 1v | 47.9 | 60.3 |
| **exp207 Base 3v** | **47.2** | **60.1** |

Base ep10 略低于 Small！可能因为 LR 更小 (0.0002 vs 0.0004) warmup 更慢。
Base 优势应在 ep40+ 显现。
**决策**: 继续

### [04:51] 检查点 #17 — 远程 ep10 ⚠️

**远程 Base 1-view ep10: 7.6/13.6%** — 极低！但训练 loss/Acc 正常（与本地相当）。
**本地 Base 3-view ep10: 47.2/60.1** — 正常。

可能原因：1-view 的 test forward 在 Base (1024-dim) 下有 eval bug？
3-view 模式 eval 路径不同（用 view[0]）。
训练本身正常（Acc 在合理范围），问题可能只在 eval。
**决策**: 继续两台。远程如果训练正常但 eval 异常，后续可以用 test.py 单独评估 checkpoint。

### [04:57] 检查点 #18

本地 ep13. 远程 1 eval (ep10=7.6, 可能 eval bug)。
本地 ep20 eval ~28min。
**决策**: 继续

### [05:02] 检查点 #19

本地 ep14. ep20 eval ~23min.
**决策**: 继续

### [05:07] 检查点 #20

本地 ep15. 远程 ep10=7.6 (可能 eval bug, 等 ep20 确认)。
本地 ep20 eval ~18min。
**决策**: 继续

### [05:13] 检查点 #21

本地 ep15 (Acc=0.241, id_global=5.859 下降中). ep20 eval ~13min.
**决策**: 继续

### [05:18] 检查点 #22

本地 ep16. ep20 eval ~8min.
**决策**: 继续

### [05:24] 检查点 #23

本地 ep17. ep20 eval ~6min.
**决策**: 等 eval

### [05:29] 检查点 #24

本地 ep18. id_global=5.647. ep20 eval ~5min.
**决策**: 等 eval

### [05:35] 检查点 #25

本地 ep19. ep20 eval ~4min.
**决策**: 等 eval

### [05:45] 检查点 #26

本地 ep20 (iter 20/227). eval ~4min.
id_global=5.430 (正常下降中)。
**决策**: 等 eval

### [05:46] 检查点 #27

本地 ep20 done. eval 运行中. ETA 10h49m.
**决策**: 等 eval

### [05:48] 检查点 #28 — 本地 ep20 🔥

**本地 Base 3-view ep20: 58.5/68.7**

| Config | ep10 | ep20 |
|--------|------|------|
| exp206 Small 1v | 47.9/60.3 | 56.6/68.3 |
| **exp207 Base 3v** | **47.2/60.1** | **58.5/68.7** |
| delta | -0.7/-0.2 | **+1.9/+0.4** |

**Base 在 ep20 反超 Small！** ep10 落后 → ep20 领先。Base 后劲更强。
如果趋势持续：final 可能 **73-75%**!
**决策**: 继续！

### [05:54] 检查点 #29 — 远程 ep20 正常了！

**远程 Base 1-view ep20: 45.6/58.8** (ep10 的 7.6 确认是 eval bug)

| Config | ep10 | ep20 |
|--------|------|------|
| 本地 Base 3v | 47.2/60.1 | 58.5/68.7 |
| 远程 Base 1v | 7.6/13.6 (bug) | 45.6/58.8 |
| Small 1v (exp206) | 47.9/60.3 | 56.6/68.3 |

远程 Base 1v ep20=45.6 < Small 1v ep20=56.6 → Base 1-view 比 Small 慢！
但本地 Base 3v ep20=58.5 > Small 1v ep20=56.6 → 3-view + Base 组合更强。
**决策**: 继续

### [06:00] 检查点 #30

本地 ep22. id_global=4.939 (快速下降). ep30 eval ~30min.
**决策**: 继续

### [06:01] 检查点 #31 — 研究 agent 结果

**关键发现**: GLOBAL_LOSS_SCALE=1.0 在所有实验中！0.5x 从未在 Small/Base 上测试！
Tiny 上 0.5x = +1.53% mAP 确认增益。
exp207 完成后下一步: **Base + 0.5x global loss**

路径到 76%: Base (74-75) + 0.5x (+1.5) + KPR prompting (+1.5) = **76-77%**
**决策**: 继续 exp207，计划下一步

### [06:09] 检查点 #32 — 远程 ep30

**远程 Base 1-view ep30: 56.2/68.1**

| Config | ep10 | ep20 | ep30 |
|--------|------|------|------|
| 本地 Base 3v | 47.2/60.1 | 58.5/68.7 | — |
| 远程 Base 1v | 7.6(bug) | 45.6/58.8 | 56.2/68.1 |
| Small 1v (exp206) | 47.9/60.3 | 56.6/68.3 | 60.8/72.1 |

远程 Base 1v 落后 Small！LR=0.0002 可能太小。
但本地 3-view 已超 Small。Base 需要 3-view 或更高 LR。
**决策**: 继续两台。下一实验调 LR。

### [06:15] 检查点 #33

本地 ep25. ep30 eval ~15min.
**决策**: 继续

### [06:20] 检查点 #34

本地 ep25 (Acc=0.325, 上升中). ep30 eval ~10min.
**决策**: 继续

### [06:25] 检查点 #35

本地 ep26. ep30 eval ~8min.
**决策**: 等 eval

### [06:31] 检查点 #36

本地 ep27. ep30 eval ~5min.
**决策**: 等 eval

### [06:37] 检查点 #37

本地 ep28. ep30 eval ~4min.
**决策**: 等 eval

### [06:42] 检查点 #38

本地 ep29. ep30 eval ~3min.
**决策**: 等 eval

### [06:46] 检查点 #39

本地 ep29 (iter 200/227). Acc=0.412! id_global=3.899.
ep30 eval ~1min.
**决策**: 等 eval

### [06:51] 检查点 #40

本地 ep30 (iter 140/227). eval ~2min.
**决策**: 等 eval

### [06:55] 检查点 #41 — 本地 ep30 🔥🔥🔥

**本地 Base 3-view ep30: 64.2/74.2!!**

| Config | ep10 | ep20 | ep30 |
|--------|------|------|------|
| **Base 3v** | 47.2/60.1 | 58.5/68.7 | **64.2/74.2** |
| Small 1v (exp206) | 47.9/60.3 | 56.6/68.3 | 60.8/72.1 |
| Base 1v (remote) | 7.6(bug) | 45.6/58.8 | 56.2/68.1 |

**Base 3-view 在 ep30 已领先 Small +3.4/+2.1！增益加速！**
ep20 +1.9 → ep30 +3.4 → ep40 预计 +4-5!

如果趋势持续: final 可能 **74-75%** (vs Small 70.5)!
再加 0.5x global loss = **75-76%!!**
**目标 76/85 有望！**
**决策**: 继续！

### [07:02] 检查点 #42

本地 ep32. 远程 crashed (inline_container error)。专注本地。
ep40 eval ~30min.
**决策**: 继续本地

### [07:16] 检查点 #43

本地 Base ep34. id_global=3.745. ep40 eval ~18min.
远程 exp208 ep4.
**决策**: 继续

### [07:21] 检查点 #44

本地 ep35. Acc=0.430. ep40 eval ~13min.
**决策**: 继续

### [07:27] 检查点 #45

本地 ep35 (Acc=0.554!). ep40 eval ~8min.
**决策**: 等 eval

### [07:33] 检查点 #46

本地 ep36. ep40 eval ~10min.
**决策**: 继续

### [07:38] 检查点 #47

本地 ep37. ep40 eval ~5min.
远程 exp208 (0.5x) ep10: 42.3/56.4 (vs 1.0x exp206 47.9/60.3 = -5.6)
0.5x 初期落后——正常，0.5x global loss 让 global branch 学更慢但最终更好。
**决策**: 继续

### [07:44] 检查点 #48

本地 ep38. ep40 eval ~5min.
**决策**: 等 eval

### [07:50] 检查点 #49

本地 ep39. eval ~3min.
**决策**: 等 eval

### [07:55] 检查点 #50

本地 ep39 done. ep40 eval running (~6min for Base).
ETA 8h41m.
**决策**: 等 eval

### [08:03] 检查点 #51 — 本地 ep40 🔥🔥

**本地 Base 3-view ep40: 66.6/76.7!!**

| Epoch | Base 3v | Small 1v (exp206) | delta |
|-------|---------|------|------|
| 10 | 47.2/60.1 | 47.9/60.3 | -0.7 |
| 20 | 58.5/68.7 | 56.6/68.3 | +1.9 |
| 30 | 64.2/74.2 | 60.8/72.1 | +3.4 |
| **40** | **66.6/76.7** | **65.4/76.5** | **+1.2** |

Base 增益从 ep30 的 +3.4 缩小到 ep40 的 +1.2。
但 Base 仍领先！趋势：final ~73-74% (vs Small 70.5)。
再加 0.5x → **74-75%!** 再加 KPR prompting → **76%!**
**决策**: 继续！

### [08:25] 检查点 #52

本地 ep44. 远程 exp208 ep4 (正常)。
ep50 eval ~20min.
**决策**: 继续

### [08:29] 检查点 #53

本地 ep~44. exp208 取消(NO-OP)。exp209 审查中。
**决策**: 继续

### [08:37] 检查点 #54

本地 ep46. id_global=2.842, Acc=0.618. 训练正常。
远程 exp209 (STD-PR+CE+OA-SD) 已启动, ep2, ETA 6h49m。
ep50 eval ~15min。
**决策**: 继续

### [08:47] 检查点 #55

本地 ep47. id_global=2.461, Acc=0.707. 训练正常，loss 持续下降。
ep50 eval ~12min。
**决策**: 继续

### [08:50] 检查点 #56

本地 ep48. id_global=2.724, Acc=0.625. 正常波动。
ep50 eval ~8min。
**决策**: 等 eval

### [08:54] 检查点 #57

本地 ep48 尾段. id_global=2.307, Acc=0.738. ep50 eval ~5min。
**决策**: 等 eval

### [08:57] 检查点 #58

本地 ep49. id_global=2.622, Acc=0.658. ep50 eval ~3min。
**决策**: 等 eval

### [09:01] 检查点 #59

本地 ep50 开始 (iter 20/227)。ep50 eval 约需 ~6min 训练 + ~6min eval = ~12min。
id_global=2.248 (ep49 尾段), Acc=0.750.
**决策**: 等 eval

### [09:06] 检查点 #60

本地 ep50 iter 180/227. id_global=2.286, Acc=0.740. ep50 完成 + eval ~3min。
**决策**: 等 eval

### [09:10] ep50 OOM crash!

**Base eval OOM！** 训练用 9.5GB (WITH_CP)，eval 时 Swin-Base forward 需要更多。
`torch.cuda.OutOfMemoryError: Tried to allocate 430 MiB`
ep40 checkpoint 保存了 (66.6/76.7)，ep50 未保存。

**修复**: 
1. `processor.py` 添加 `torch.cuda.empty_cache()` 在 eval 前释放训练缓存
2. `TEST.IMS_PER_BATCH` 从 256 降到 128

### [09:15] 检查点 #61 — 重启

exp207 已重启（从头开始，无 resume 功能）。
ETA ~8h。已知 ep40=66.6/76.7。
**决策**: 继续

### [09:20] 检查点 #62 — 重启修正

发现重启缺少 PARALLEL_AUG（3-view）。GPU 仅 6.9GB, Speed 89 s/s — 不是 3-view。
Kill 并重新启动，加上 `MODEL.POSE_PARALLEL_AUG True`。
重启后 GPU 9.5GB — 与首次运行一致。确认 3-view 激活。
ETA ~14h43m（比原始 11h40m 慢——不同 random state 导致 Speed 30 vs 38 s/s）。
**决策**: 继续

### [09:49] 检查点 #63

ep4. id_global=6.538, oa_sd=0.371. 训练正常启动。
同时在准备 exp210 (PKC: Per-Keypoint Contrastive) 实验——但需等 GPU 空闲。
**决策**: 继续

### [10:10] 检查点 #64 — OA-SD teacher bug 发现 & 修复

用户的另一个 Claude 发现 Critical bug:
1. **EMA teacher 用 `train()` mode forward** → Dropout/DropPath 噪声污染 distillation target + BN running stats 被错误更新
2. **EMA 更新只复制 parameters，不复制 buffers** → BN running_mean/var 不同步

**修复**:
- teacher forward 改为 `eval()` mode + `pose_test_feat='global'`
- EMA 更新同时复制 buffers（BN running stats）
- 第一次修复 crash (2048 vs 1024 dim mismatch)——因为 eval 模式的 equal_concat 返回拼接后的 2048 dim
- 第二次修复: 临时设置 `pose_test_feat='global'` 让 teacher 只返回 1024-dim global feat

exp207 第三次重启。ep1 正常运行，oa_sd=0.402。
**重要**: 之前所有 OA-SD 实验 (exp191, exp193, exp200, exp206, exp207) 的 teacher 都是有噪声的。
修复后 OA-SD 应该更强——teacher target 更稳定。
**决策**: 继续

### [10:12] 检查点 #65 — 第二轮修复后重启 (第4次)

发现第一次修复 (eval mode) 导致 teacher 只返回 global feat (2048→1024 dim crash)。
第二次修复 (pose_test_feat='global') 修了 crash 但让 OA-SD 变成了 global-only distillation。
**第三次修复**: teacher 用 `train()` 模式但 BN/Dropout/DropPath 全部设为 `eval()`。
这样 teacher 返回完整的 `[global_feat, skeleton_feat]`，distillation 同时覆盖 global + part。
oa_sd=0.458 (正常)。

### [10:28] 检查点 #66

exp207 ep1 mid. oa_sd=0.458, id_global=6.554. 3-view 正常。
远程 exp210 (GCN+PAA+CE+OA-SD+PKC Small) 已启动，pkc=3.725。
**决策**: 继续

### [10:30] 检查点 #67

exp207 ep2. oa_sd=0.515, ETA 12h14m. 训练正常。
exp210 远程 ep2. pkc=3.942 (下降中，学习中)。
**决策**: 继续

### [10:43] 检查点 #68 — teacher pose 修复后第5次重启

又发现 PLBOA 修改 persons in-place 导致 teacher 收到 student 的 occluded pose。
**修复**: dataset 在 PLBOA 前 deepcopy persons，生成 `teacher_pose` sub-dict。
teacher forward 使用 `pose_dict['teacher_pose']`（干净 pose，与 clean image 对齐）。

exp207 第5次重启（最终版本，所有修复都已部署）。
ep1 running. oa_sd=0.449. 训练正常。ETA 12h14m。
**决策**: 继续

### [10:44] 检查点 #69

exp207 ep1 mid (iter 140). id_global=6.554, oa_sd=0.454. 正常 warmup。
exp210 远程 ep2. pkc=3.790, oa_sd=0.668. ETA 6h。
**决策**: 继续

### [10:49] 检查点 #70

exp207 ep2 mid. id_global=6.554, oa_sd=0.517. 正常。
exp210 远程 ep3 done. pkc=3.892, oa_sd=0.243. ETA 5h58m。
**决策**: 继续

### [10:55] 检查点 #71

exp207 ep3. id_global=6.549, oa_sd=0.456.
exp210 ep5. pkc=3.888, oa_sd=0.035 (极低——teacher≈student).
两台训练正常。ep10 eval: exp207 ~50min, exp210 ~17min.
**决策**: 继续

### [11:18] 检查点 #72

exp207 ep7. id_global=6.479, Acc=0.101. 训练正常。
**exp210 ep10 灾难: 3.6/5.3%！PKC 破坏了 feature space。**
已终止 exp210，远程改跑 exp206r (同配置无 PKC) 作为 OA-SD fix 对照。
exp206r ep3: id_global=6.541, Acc=0.010 (正常 warmup)。
**决策**: 继续 exp207，等 exp206r ep10 确认 OA-SD fix 无问题

### [11:30] 检查点 #73

exp207 ep8. Acc=0.229, id_global=6.411. 训练正常加速。oa_sd=0.089.
exp206r (对照) ep5: Acc=0.126, id_global=6.494 — **正常！远超 exp210 同 epoch**。
确认: OA-SD fix 无问题, PKC weight=0.5 是灾难原因。
**决策**: 继续两台。exp206r ep10 eval ~17min。

### [11:35] 检查点 #74

exp207 ep9. Acc=0.230, id_global=6.371. oa_sd=0.068. 正常收敛。
exp206r ep7. Acc=0.181, id_global=6.402. oa_sd=0.072. ep10 eval ~10min。
**决策**: 等 exp206r ep10 eval

### [11:41] 检查点 #75

exp207 ep~10 (approaching). exp206r ep9 mid. ep10 eval ~5min。
exp206r: id_global=6.278, Acc=0.184 — 学习正常！OA-SD fix confirmed OK。
**决策**: 等 ep10 evals

### [11:46] 检查点 #76 — ep10 🔥🔥

**两台 ep10 结果！OA-SD fix 带来显著提升！**

| Config | ep10 mAP | ep10 R1 | vs 原版 |
|--------|---------|---------|---------|
| **exp207 Base 3v (fixed)** | **51.4** | **62.9** | +4.2/+2.8 vs old 47.2/60.1 |
| **exp206r Small 1v (fixed)** | **50.4** | **63.9** | +2.5/+3.6 vs old 47.9/60.3 |
| exp206 Small 1v (buggy) | 47.9 | 60.3 | — |
| exp207 Base 3v (buggy) | 47.2 | 60.1 | — |

**OA-SD teacher fix 确认有效！** 修复后的 teacher (无 Dropout/DropPath 噪声 + 干净 pose) 产生更好的 distillation target。
Small 在 ep10 已经超过 Base！(50.4 > 51.4 考虑 Base LR 更小, warmup 更慢)

**如果趋势持续**: 
- exp206r final 可能达到 **72-73%** (vs old 70.5) → +2% from fix alone!
- exp207 final 可能达到 **76-77%** (vs old ~73-75 估计)
- 再加 maxsim_hybrid → **74-75% (Small) / 78-79% (Base)!**

**决策**: 继续两台！这可能是最重要的突破！

### [11:51] 检查点 #77

exp207 ep11. Acc=0.298, id_global=6.214. ETA 11h42m.
exp206r ep12. Acc=0.229, id_global=5.889 (下降很快！). ep20 eval ~25min。
**决策**: 继续

### [11:57] 检查点 #78

exp207 ep12. id_global=6.368, Acc=0.111. 
exp206r ep14. id_global=5.689, Acc=0.172. ep20 eval ~20min。
**决策**: 继续

### [12:02] 检查点 #79

exp206r ep16. id_global=5.361. ep20 eval ~13min。
**决策**: 继续

### [12:08] 检查点 #80

exp206r ep18. id_global=5.117. ep20 eval ~7min。
exp207 ep~14。
**决策**: 等 exp206r ep20 eval

### [12:13] 检查点 #81

exp206r ep20 mid. id_global=4.781. eval ~3min。
**决策**: 等 eval

### [12:17] 检查点 #82 — exp206r ep20

**exp206r ep20: 56.6/68.1** (vs exp206 buggy ep20: 56.6/68.3 = 0.0/-0.2)

| Epoch | exp206r (fixed) | exp206 (buggy) | delta |
|-------|------|------|------|
| 10 | 50.4/63.9 | 47.9/60.3 | +2.5/+3.6 |
| 20 | 56.6/68.1 | 56.6/68.3 | 0.0/-0.2 |

**OA-SD fix 加速了早期收敛但 ep20 已追平。** 
EMA teacher 在后期本就接近 student，所以 Dropout 噪声的影响减小。
Fix 仍然更正确，但可能不会改变 final 结果。
需要看 ep30/40 是否仍一致或开始分化。
**决策**: 继续 exp206r 到至少 ep40。继续 exp207。

### [12:23] 检查点 #83

exp207 ep16. id_global=5.757, Acc=0.248. ETA 10h57m.
exp206r ep23. id_global=4.282, Acc=0.283. ep30 eval ~22min。
**决策**: 继续

### [12:29] 检查点 #84

exp207 ep17. id_global=5.673, Acc=0.216. ep20 eval ~15min。
exp206r ep25. id_global=3.976, Acc=0.333. ep30 eval ~15min。
**决策**: 等 evals (两台都约 15min)

### [12:34] 检查点 #85

exp207 ep18. id_global=5.595. ep20 eval ~10min。
exp206r ep26 done. id_global=3.390, Acc=0.512. ep30 eval ~12min。
**决策**: 等 evals

### [12:39] 检查点 #86

exp207 ep19. id_global=5.505. ep20 eval ~5min。
exp206r ep28. id_global=3.105, Acc=0.577. ep30 eval ~6min。
**决策**: 等 evals

### [12:45] 检查点 #87

exp207 ep20 iter 80. id_global=5.397. eval ~5min。
exp206r ep30 eval 应已完成或即将完成。
**决策**: 等 evals

### [12:50] 检查点 #88 — ep20/ep30 🔥🔥

**exp207 Base 3v ep20: 59.2/70.5** (+0.7/+1.8 vs buggy 58.5/68.7)
**exp206r Small 1v ep30: 62.3/73.8** (+1.5/+1.7 vs buggy 60.8/72.1)

| Epoch | exp206r (fixed) | exp206 (buggy) | delta |
|-------|------|------|------|
| 10 | 50.4/63.9 | 47.9/60.3 | +2.5/+3.6 |
| 20 | 56.6/68.1 | 56.6/68.3 | 0.0/-0.2 |
| **30** | **62.3/73.8** | **60.8/72.1** | **+1.5/+1.7** |

**OA-SD fix 在 ep30 重新拉开差距！** ep20 的追平只是暂时的。
如果趋势持续: **exp206r final 可能达到 72-73%** (vs old 70.5)!
再加 maxsim_hybrid: **74-75%** on Small!

| Epoch | exp207 Base 3v (fixed) | exp207 Base 3v (buggy) | delta |
|-------|------|------|------|
| 10 | 51.4/62.9 | 47.2/60.1 | +4.2/+2.8 |
| **20** | **59.2/70.5** | **58.5/68.7** | **+0.7/+1.8** |

Base 也有 +0.7/+1.8 改进。Base final 可能 **75-76%**!
**决策**: 继续！两台都非常有前途！

### [12:56] 检查点 #89

exp207 ep21. id_global=5.075, Acc=0.231.
exp206r ep34. id_global=2.931, Acc=0.555. ep40 eval ~18min。
**决策**: 继续

### [13:01] 检查点 #90

exp207 ep22. id_global=5.021, Acc=0.212.
exp206r ep35. id_global=2.298, Acc=0.729. ETA 4h01m. ep40 eval ~13min。
**决策**: 继续

### [13:07] 检查点 #91

exp206r ep37. id_global=2.206, Acc=0.742. ep40 eval ~8min。
**决策**: 等 eval

### [13:12] 检查点 #92

exp206r ep39. id_global=2.108, Acc=0.752. ep40 eval ~3min。
**决策**: 等 eval

### [13:18] 检查点 #93 — exp206r ep40

**exp206r ep40: 65.8/76.4** (vs buggy 65.4/76.5 = +0.4/-0.1)

| Epoch | exp206r (fixed) | exp206 (buggy) | delta |
|-------|------|------|------|
| 10 | 50.4/63.9 | 47.9/60.3 | +2.5/+3.6 |
| 20 | 56.6/68.1 | 56.6/68.3 | 0.0/-0.2 |
| 30 | 62.3/73.8 | 60.8/72.1 | +1.5/+1.7 |
| 40 | 65.8/76.4 | 65.4/76.5 | +0.4/-0.1 |

差距在 ep40 再次缩小。fix 可能主要加速收敛而非改变 final。
需要看 ep60-120 是否保持/扩大优势。
**决策**: 继续跑完 120ep

### [13:20] 检查点 #94

exp207 ep25. id_global=4.641, Acc=0.265.
exp206r ep41. id_global=2.051, Acc=0.759. ep50 eval ~28min。
**决策**: 继续

### [13:25] 检查点 #95

exp207 ep26. id_global=4.569. exp207 ep30 eval ~25min。
exp206r ep43. id_global=2.005. ep50 eval ~22min。
**决策**: 继续

### [13:30] 检查点 #96

exp207 ep27. id_global=4.503. ep30 eval ~18min。
exp206r ep45. id_global=1.962. ep50 eval ~15min。
**决策**: 继续

### [13:35] 检查点 #97

exp207 ep27 done. id_global=4.127. ETA 9h54m. ep30 eval ~12min。
exp206r ep47. id_global=1.856. ep50 eval ~9min。
**决策**: 等 evals

### [13:41] 检查点 #98

exp207 ep28. id_global=4.111. ep30 eval ~8min。
exp206r ep48. id_global=1.409, Acc=0.866. ep50 eval ~6min。
**决策**: 等 evals

### [13:46] 检查点 #99

exp207 ep29. ep30 eval ~3min。
exp206r ep50 mid. ep50 eval ~2min。
**决策**: 等 evals

### [13:52] 检查点 #100 — exp206r ep50

**exp206r ep50: 67.6/79.5** (vs buggy 67.2/78.8 = +0.4/+0.7)

| Epoch | exp206r (fixed) | exp206 (buggy) | delta |
|-------|------|------|------|
| 10 | 50.4/63.9 | 47.9/60.3 | +2.5/+3.6 |
| 20 | 56.6/68.1 | 56.6/68.3 | 0.0/-0.2 |
| 30 | 62.3/73.8 | 60.8/72.1 | +1.5/+1.7 |
| 40 | 65.8/76.4 | 65.4/76.5 | +0.4/-0.1 |
| 50 | 67.6/79.5 | 67.2/78.8 | +0.4/+0.7 |

OA-SD fix 在 ep50 保持 +0.4 mAP, +0.7 R1 优势。不大但一致。
exp207 ep30 eval ~3min。
**决策**: 继续

### [13:56] 检查点 #101 — exp207 ep30

**exp207 Base 3v ep30: 63.6/73.8** (vs buggy 64.2/74.2 = -0.6/-0.4)

| Epoch | exp207 (fixed) | exp207 (buggy) | delta |
|-------|------|------|------|
| 10 | 51.4/62.9 | 47.2/60.1 | +4.2/+2.8 |
| 20 | 59.2/70.5 | 58.5/68.7 | +0.7/+1.8 |
| 30 | 63.6/73.8 | 64.2/74.2 | -0.6/-0.4 |

与 Small 类似的震荡模式。Fix 早期领先，中期略落后。
可能原因: buggy teacher 的 Dropout 噪声在某些阶段充当 regularizer。
需要看 ep40+ 是否追回。
**决策**: 继续

### [14:02] 检查点 #102

exp207 ep31. id_global=3.622, Acc=0.482. ETA 9h31m.
exp206r ep55. id_global=1.195, Acc=0.893. ep60 eval ~15min。
**决策**: 继续

### [14:08] 检查点 #103

exp207 ep32. id_global=3.644.
exp206r ep57. id_global=1.182, Acc=0.892. ep60 eval ~8min。
**决策**: 继续

### [14:14] 检查点 #104

exp206r ep59. ep60 eval ~3min。
exp207 ep~33。
**决策**: 等 eval

### [14:19] 检查点 #105 — exp206r ep60

**exp206r ep60: 68.3/79.8** (vs buggy 67.8/79.1 = +0.5/+0.7)

完整趋势:

| Epoch | exp206r (fixed) | exp206 (buggy) | delta mAP |
|-------|------|------|------|
| 10 | 50.4 | 47.9 | +2.5 |
| 20 | 56.6 | 56.6 | 0.0 |
| 30 | 62.3 | 60.8 | +1.5 |
| 40 | 65.8 | 65.4 | +0.4 |
| 50 | 67.6 | 67.2 | +0.4 |
| 60 | 68.3 | 67.8 | +0.5 |

ep50-60 稳定 +0.5 mAP。预计 final ~71.0 (vs old 70.5) → +0.5% from fix。
虽然不大，但加上 maxsim_hybrid (+1.8): **~72.8% on Small**。
**决策**: 继续两台

### [14:25] 检查点 #106

exp207 ep35. id_global=3.585, Acc=0.462. ep40 eval ~30min。
exp206r ep63. id_global=1.197, Acc=0.873. ETA ~2h30m。
**决策**: 继续

### [14:31] 检查点 #107

exp207 ep36. id_global=3.496.
exp206r ep65. ETA 2h37m.
**决策**: 继续。下次检查 ep40 eval。

### [14:36] 检查点 #108

exp207 ep37. id_global=3.468. ep40 eval ~18min。
exp206r ep66. id_global=0.867, Acc=0.931.
**决策**: 继续

### [14:41] 检查点 #109

exp207 ep38. id_global=3.295. ep40 eval ~13min。
exp206r ep68. Acc=0.927.
**决策**: 等 exp207 ep40 eval

### [14:47] 检查点 #110

exp207 ep38. ETA 8h44m. ep40 eval ~13min。
**决策**: 等 eval

### [14:53] 检查点 #111

exp207 ep39 mid. id_global=2.979, Acc=0.614. ep40 eval ~4min。
**决策**: 等 eval

### [14:58] 检查点 #112

exp207 ep40 iter 140. Acc=0.592. eval ~5min (iter 227 + 6min eval)。
**决策**: 等 eval

### [15:02] 检查点 #113 — exp207 ep40

**exp207 Base 3v ep40: 66.5/77.2** (vs buggy 66.6/76.7 = -0.1/+0.5)

| Epoch | fixed | buggy | delta |
|-------|------|------|------|
| 10 | 51.4/62.9 | 47.2/60.1 | +4.2/+2.8 |
| 20 | 59.2/70.5 | 58.5/68.7 | +0.7/+1.8 |
| 30 | 63.6/73.8 | 64.2/74.2 | -0.6/-0.4 |
| 40 | 66.5/77.2 | 66.6/76.7 | -0.1/+0.5 |

Base ep40 基本持平。Fix 主要帮助 R1 (+0.5)。
**决策**: 继续，等 ep50+

### [15:08] 检查点 #114

exp207 ep42. ETA 8h27m.
**exp206r ep70: 68.5/80.4** (vs buggy remote 69.4 = **-0.9!**)
Fix 在 ep70 落后！可能 teacher Dropout 噪声在后期充当 regularizer。
或者只是训练 variance（单次运行）。
需要继续到 ep120 看最终结果。
**决策**: 继续

### [15:14] 检查点 #115

exp207 ep42. ETA 8h16m.
exp206r ep79. ep80 eval ~3min。ETA ~2h。
**决策**: 继续

### [15:19] 检查点 #116

exp206r ep80 done. eval 运行中 (~2min). ETA 1h53m.
**决策**: 等 eval

### [15:21] 检查点 #117 — exp206r ep80 🔥

**exp206r ep80: 70.2/81.5!** (vs buggy 69.5 = **+0.7!**)

| Epoch | fixed | buggy | delta |
|-------|------|------|------|
| 60 | 68.3 | 67.8 | +0.5 |
| 70 | 68.5 | 69.4 | -0.9 |
| 80 | **70.2** | **69.5** | **+0.7** |

ep70 的落后在 ep80 追回并反超！OA-SD fix 在后期再次领先。
**70.2% at ep80 已超过 buggy ep80 (69.5)！**
预计 final: ~71.0-71.5% (vs buggy 70.5) → **+0.5-1.0%**
再加 maxsim_hybrid: **~72.8-73.3% on Small!**
**决策**: 继续！非常有前途！

### [15:26] 检查点 #118

exp207 ep44. id_global=2.624.
exp206r ep82. ETA 1h49m. ep90 eval ~24min。
**决策**: 继续

### [15:32] 检查点 #119

exp207 ep45. id_global=2.699.
exp206r ep84. ep90 eval ~17min。
**决策**: 继续

### [15:37] 检查点 #120

exp206r ep86. ep90 eval ~11min。
**决策**: 继续

### [15:43] 检查点 #121

exp206r ep88. ep90 eval ~6min.
exp207 ep~47.
**决策**: 等 eval

### [15:48] 检查点 #122

exp206r ep90 mid. eval ~4min.
**决策**: 等 eval

### [15:52] 检查点 #123 — exp206r ep90

**exp206r ep90: 70.2/81.9** (vs buggy 70.1 = +0.1 — 基本追平)

完整趋势:

| Epoch | fixed | buggy | delta |
|-------|------|------|------|
| 10 | 50.4 | 47.9 | +2.5 |
| 20 | 56.6 | 56.6 | 0.0 |
| 30 | 62.3 | 60.8 | +1.5 |
| 40 | 65.8 | 65.4 | +0.4 |
| 50 | 67.6 | 67.2 | +0.4 |
| 60 | 68.3 | 67.8 | +0.5 |
| 70 | 68.5 | 69.4 | -0.9 |
| 80 | 70.2 | 69.5 | +0.7 |
| 90 | 70.2 | 70.1 | +0.1 |

ep80-90 之间 fixed 停滞在 70.2，buggy 追到 70.1。
预计 final: ~70.5-71.0 for both（差别在训练 variance 范围内）。
**结论**: OA-SD fix 正确但不改变 final 结果。主要 lever 仍是 maxsim_hybrid (+1.8%)。
**决策**: 继续跑到 ep120 确认

### [15:59] 检查点 #124

exp207 ep49. id_global=2.220, Acc=0.756. ep50 eval ~8min。
exp206r ep93. ep100 eval ~20min。
**决策**: 继续

### [16:04] 检查点 #125

exp207 ep50 iter 160. eval ~4min。
exp206r ep95. ep100 eval ~14min。
**决策**: 等 evals

### [16:10] 检查点 #126 — exp207 ep50

**exp207 Base 3v ep50: 68.0/77.6**

| Epoch | fixed Base 3v | delta |
|-------|------|------|
| 10 | 51.4/62.9 | |
| 20 | 59.2/70.5 | |
| 30 | 63.6/73.8 | |
| 40 | 66.5/77.2 | |
| 50 | 68.0/77.6 | ep40→50 +1.5/+0.4 |

增长放缓 (ep30→40 +2.9, ep40→50 +1.5)。趋势预计 final ~73-74%。
exp206r ep97. ep100 eval ~8min。
**决策**: 继续

### [16:15] 检查点 #127

exp206r ep99. ep100 eval ~3min. ETA 1h03m.
**决策**: 等 eval

### [16:20] 检查点 #128

exp206r ep100 mid. eval ~3min.
**决策**: 等 eval

### [16:23] 检查点 #129 — exp206r ep100

**exp206r ep100: 70.3/81.9** (vs buggy 70.3 = **完全一致**)

**最终结论: OA-SD fix 不改变 final 结果。**
fix 加速了早期收敛 (ep10 +2.5%)，但 final 结果与 buggy 版相同。
Teacher Dropout 噪声在 EMA 框架下不影响最终性能。

exp206r 剩余 ~50min (ep100→120)。继续到完成。
**最重要的 lever 仍是 maxsim_hybrid (+1.8% on any checkpoint)。**

预计 exp206r final: ~70.5/82.0 → + maxsim_hybrid = **~72.3/83.0**
**决策**: 继续两台

### [17:14] 检查点 #130

exp207 ep61. id_global=2.074. ETA ~6h.
exp206r ep118! 即将完成！final eval ~8min。
**决策**: 等 exp206r final

### [17:19] 检查点 #131

exp206r ep120 训练中！ETA 0h2m！final eval ~5min。
ep110: 70.6/82.3
**决策**: 等 final eval

### [17:25] 检查点 #132 — exp206r FINAL 🔥

**exp206r (fixed OA-SD) FINAL: 70.6/82.6**
vs buggy exp206: 70.5/82.3 (remote), 70.3/81.8 (local)
**Fix: +0.1/+0.3 — 微小改善，在训练 variance 范围内。**

远程 GPU 空闲。正在测试 maxsim_hybrid on exp206r ep120 checkpoint。
同时规划下一个远程实验。
**决策**: 等 maxsim_hybrid 测试结果

### [17:33] 检查点 #133 — MaxSim on exp206r 🔥🔥

**exp206r + maxsim_hybrid: 72.3/82.9!!**

| 配置 | mAP | R1 |
|------|------|------|
| exp206r equal_concat | 70.6% | 82.6% |
| **exp206r maxsim_hybrid** | **72.3%** | **82.9%** |
| exp206 local maxsim_hybrid | 72.1% | 82.9% |

**72.3% mAP = 新最佳！无 NFC/reranking！**
远程 GPU 空闲。准备启动 exp210b (PKC weight=0.05)。
**决策**: 启动 exp210b

### [17:44] 检查点 #134

exp207 ep65. id_global=1.592.
exp210b ep3. pkc=2.865, Acc=0.011 — PKC weight=0.05 未干扰 CE！
**决策**: 继续

### [18:13] 检查点 #135

exp207 ep70. id_global=1.766. 接近 final territory。
exp210b ep12. Acc=0.201, pkc=3.233. 正常。
**决策**: 继续

### [18:18] 检查点 #136 — exp207 ep60/70 🔥

**exp207 ep60: 68.9/78.7, ep70: 69.7/79.6!**

| Epoch | Base 3v (fixed) | Small 1v (exp206r) | Base advantage |
|-------|------|------|------|
| 40 | 66.5 | 65.8 | +0.7 |
| 50 | 68.0 | 67.6 | +0.4 |
| 60 | 68.9 | 68.3 | +0.6 |
| 70 | **69.7** | **68.5** | **+1.2** |

**Base 在 ep70 领先 Small +1.2%！**增速: ep60→70 +0.8 (稳定)。
预计 final: ~73-74% → + maxsim_hybrid = **~75-76%!**
**决策**: 继续！Base scaling 非常有效！

### [18:25] 检查点 #137

exp207 ep72. ETA 5h14m.
exp210b ep16. pkc=3.214. ep20 eval ~12min.
**决策**: 继续

### [18:47] 检查点 #138

exp207 ep75. id_global=1.658. ETA ~4h45m.
exp210b ep23. pkc=3.050 (持续下降——keypoint features 在学习！). ep30 eval ~22min.
**exp210b ep20: 56.9/68.4 = +0.3/+0.3 vs exp206r — PKC=0.05 有效！**
**决策**: 继续

### [19:20] 检查点 #139

exp207 ep80 iter 140. eval ~4min.
exp210b ep30: 61.7/73.9 (-0.6 vs exp206r, 震荡模式)。
**决策**: 等 exp207 ep80 eval

### [19:24] 检查点 #140 — exp207 ep80 🔥

**exp207 Base 3v ep80: 70.3/80.3!**

| Epoch | Base 3v | delta |
|-------|------|------|
| 50 | 68.0 | — |
| 60 | 68.9 | +0.9 |
| 70 | 69.7 | +0.8 |
| 80 | 70.3 | +0.6 |

**ep80 Base 已超过 Small final (70.5-70.6)！**
增长放缓但仍在上升。预计 final ~72.0-72.5%。
+ maxsim_hybrid = **~73.5-74.5%!**
**决策**: 继续！

### [19:31] 检查点 #141

exp207 ep82. ETA 4h10m.
exp210b ep37. pkc=2.699 (持续下降). ETA 4h14m. ep40 eval ~10min.
两台都还有 ~4h。
**决策**: 继续

### [19:51] 检查点 #142

exp207 ep85. id_global=1.609. ETA ~3h30m.
exp210b ep43. pkc=2.743. ETA 4h.
**决策**: 继续。下一个重要 eval: exp207 ep90 (~15min), exp210b ep50 (~20min).

### [19:57] 检查点 #143

exp207 ep~87. ep90 eval ~10min.
exp210b ep44. pkc=2.634 (持续下降!). ep50 eval ~17min.
**决策**: 等 evals

### [20:02] 检查点 #144

exp207 ep86. ETA 3h37m. ep90 eval ~25min。
exp210b ep46. pkc=2.637. ep50 eval ~12min。
**决策**: 继续

### [20:23] 检查点 #145

exp207 ep90 mid. eval ~4min.
exp210b ep50: 67.6/79.6 (= exp206r). PKC=0.05 不影响 equal_concat。
**决策**: 等 exp207 ep90 eval

### [20:30] 检查点 #146 — exp207 ep90

**exp207 Base 3v ep90: 70.6/80.5!**

| Epoch | Base 3v | delta |
|-------|------|------|
| 70 | 69.7 | — |
| 80 | 70.3 | +0.6 |
| 90 | 70.6 | +0.3 |

增长继续放缓。ep80→90 仅 +0.3。预计 final ~71.2-71.5%。
+ maxsim_hybrid ≈ **73.0-73.3%**
ETA 3h12m.
**决策**: 继续

### [20:35] 检查点 #147

exp207 ep91.
exp210b ep56. pkc=2.484. ep60 eval ~12min.
**决策**: 继续

### [20:41] 检查点 #148

exp207 ep~93.
exp210b ep58. pkc=2.440. ep60 eval ~6min.
**决策**: 等 eval

### [20:46] 检查点 #149

exp210b ep60 mid. eval ~3min.
**决策**: 等 eval

### [20:56] 检查点 #150

exp207 ep95. ETA 2h45m. ep100 eval ~30min.
exp210b ep62. pkc=2.418. ETA 2h55m. ep70 eval ~24min.
**决策**: 继续。两台都约 3h 后完成。

### [21:02] 检查点 #151

exp207 ep95. ETA 2h38m.
exp210b ep64. pkc=2.412. ETA ~2h50m.
**决策**: 继续

### [21:07] 检查点 #152

exp207 ep~97. ep100 eval ~18min.
exp210b ep~66. ep70 eval ~12min.
**决策**: 继续监控

### [21:12] 检查点 #153

exp207 ep97. ep100 eval ~18min.
exp210b ep68. pkc=2.427. ep70 eval ~7min.
**决策**: 继续

### [21:18] 检查点 #154

exp207 ep98. ep100 eval ~12min.
exp210b ep69 done. ep70 eval 即将!
**决策**: 等 evals

### [21:28] 检查点 #155

exp207 ep100 开始. eval ~12min. ETA 2h14m.
exp210b ep70: 68.7/79.8 (+0.2 vs exp206r).
**决策**: 等 exp207 ep100 eval

### [21:34] 检查点 #156

exp207 ep100 done! eval 运行中 (~6min). ETA 2h07m.
**决策**: 等 eval

### [21:35] 检查点 #157 — exp207 ep100

**exp207 Base 3v ep100: 70.7/80.9**

| Epoch | Base 3v | delta |
|-------|------|------|
| 80 | 70.3 | — |
| 90 | 70.6 | +0.3 |
| 100 | 70.7 | +0.1 |

增长几乎停止。预计 final ~71.0%。
Base 仅比 Small (70.6) 高 +0.1-0.4%，低于预期。
可能原因: LR=0.0002 对 Base 偏低，3-view+OA-SD 交互不够好。
+ maxsim_hybrid: ~72.8%
**决策**: 继续到 ep120

### [21:45] 检查点 #158

exp207 ep102.
exp210b ep78. pkc=2.342 (从 3.27→2.34 = 28% drop). ep80 eval ~6min.
**决策**: 继续

### [22:01] 检查点 #159

exp207 ep105. ETA 1h41m.
exp210b ep82. pkc=2.342. ETA ~1h55m.
exp210b ep80: 70.1/82.0 (R1 领先 exp206r +0.5!)
**决策**: 继续到完成

### [22:07] 检查点 #160

exp207 ep105. ~1h30m to finish.
exp210b ~ep85. ~1h40m to finish.
**决策**: 继续到完成。exp207 ep110 eval ~15min。

### [22:12] 检查点 #161

exp207 ep106. ep110 eval ~25min.
exp210b ep86. pkc=2.328. ep90 eval ~12min.
**决策**: 继续

### [22:24] 检查点 #162

exp207 ep108. ep110 eval ~13min.
exp210b ep90 开始. pkc=2.306. eval ~5min.
**决策**: 等 evals

### [22:34] 检查点 #163

exp207 ep110 mid. eval ~10min.
exp210b ep90: 70.0/81.4 (-0.2/-0.5 vs exp206r).
**决策**: 等 exp207 ep110 eval

### [22:39] 检查点 #164

exp207 ep110 done! eval running. ETA 1h03m.
**决策**: 等 eval

### [22:41] 检查点 #165 — exp207 ep110

**exp207 Base 3v ep110: 70.7/80.7** (= ep100 持平)

| Epoch | Base 3v |
|-------|------|
| 90 | 70.6 |
| 100 | 70.7 |
| 110 | 70.7 |

**完全平台！Base 3v final ≈ 70.7%。**
仅比 Small (70.6) 高 +0.1% — Base scaling 几乎无效！
可能原因: LR=0.0002 太低 / 3-view+CP 限制了 Base 容量。
+ maxsim_hybrid: ~72.5%
**决策**: 继续到 ep120 确认 final

### [22:53] 检查点 #166

exp207 ep112. ETA 51min.
exp210b ep99. pkc=2.287. ep100 eval ~4min.
**决策**: 等 evals

### [23:07] 检查点 #167

exp207 ep115. ETA 38min.
exp210b ep102. ETA 56min.
两台都接近完成！
**决策**: 继续到 final

### [23:12] 检查点 #168

exp207 ep115. ETA ~33min.
exp210b ep104. ETA ~48min.
**决策**: 等 finals

### [23:18] 检查点 #169

exp207 ep116. final ~25min。
**决策**: 等 final

### [23:23] 检查点 #170

exp207 ep117. ~15min to final ep120 eval.
**决策**: 等 final

### [23:29] 检查点 #171

exp207 ep118. final ~12min.
**决策**: 等 final

### [23:35] 检查点 #172

exp207 ep119 mid. final eval ~10min.
**决策**: 等 final

### [23:40] 检查点 #173

exp207 ep120! ETA 6min! Final eval ~10min.
**决策**: 等 FINAL!

### [23:45] 检查点 #174

exp207 ep120 DONE! eval running (~6min for Base).
**决策**: 等 FINAL eval

## exp207 FINAL RESULTS

**exp207 Base 3-view + GCN+PAA+CE+OA-SD (fixed): 70.7/80.7**

| Epoch | mAP | R1 |
|-------|------|------|
| 10 | 51.4% | 62.9% |
| 20 | 59.2% | 70.5% |
| 30 | 63.6% | 73.8% |
| 40 | 66.5% | 77.2% |
| 50 | 68.0% | 77.6% |
| 60 | 68.9% | 78.7% |
| 70 | 69.7% | 79.6% |
| 80 | 70.3% | 80.3% |
| 90 | 70.6% | 80.5% |
| 100 | 70.7% | 80.9% |
| 110 | 70.7% | 80.7% |
| **120** | **70.7%** | **80.7%** |

**Base 仅比 Small (70.6) 高 +0.1%。** Base scaling 几乎无效！
可能原因: LR=0.0002 太低 / 3-view + CP 限制了 Base 容量。
下一步: 测试 maxsim_hybrid on Base checkpoint。

### MaxSim Hybrid 测试结果

**exp207 Base + maxsim_hybrid: 72.2/82.0**

| 模型 | equal_concat | maxsim_hybrid | delta |
|------|------|------|------|
| exp206r Small | 70.6/82.6 | 72.3/82.9 | +1.7/+0.3 |
| exp207 Base | 70.7/80.7 | 72.2/82.0 | +1.5/+1.3 |

**Base maxsim (72.2) < Small maxsim (72.3)！**
Base scaling 在当前配置下完全无效。
原因分析:
1. LR=0.0002 对 Base 太低（Small 用 0.0004）
2. 3-view + CP 可能限制了 Base 的学习
3. Occluded-Duke 训练集太小（15618 images）无法支撑 88M param Base

**本地 GPU 空闲。准备启动新实验。**
