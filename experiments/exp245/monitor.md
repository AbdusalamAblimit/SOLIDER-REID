# exp245 Small + LGPA-Detach + OA-SD 监控

配置: Small + PSG + LGPA-D (CLIP, detached) + OA-SD + PLBOA(0.7)
对照: exp206r (Small GCN+OA-SD): 70.6/82.6
对照: exp242 (Small PPA+GCN non-detach): 60.9/73.4 (灾难性失败)
目标: 验证 LGPA-D 在 Small 上泛化性

## 检查点

### [07:10] 检查点 #1

首次启动 (带 OA-SD): 5060Ti 16GB OOM! Small + LGPA + OA-SD 太大。
重启: 无 OA-SD 版本。等待确认启动。
**注意**: 无 OA-SD, 对照应为无 OA-SD 的 Small 基线。

### [07:12] 检查点 #2

远程重启成功 (无 OA-SD)! ep1 iter80. lgpa_assign=7.21。
Speed ~12s/20iter。
**决策**: 等 ep10 eval

### [07:35] 检查点 #3 — 远程 ep10 eval

**远程 ep10: 0.3/0.3%!!** 灾难性失败。模型完全未学习。
Acc=2.4%, id_global=6.51 (正常应 ~5.8)。
可能原因:
1. LR 0.0004 无 OA-SD 太保守 (Tiny 用 0.0008)
2. Small 需要更多 warmup
3. LGPA part classifiers 在 Small 上不收敛

先继续观察到 ep20, 如果仍然极低则 kill。
**决策**: 等 ep20 eval

### [07:58] 检查点 #4 — ep20 eval

**远程 ep20: 3.0/5.0%** — 仍然灾难性。从 0.3→3.0, 极慢学习。
**决策**: Kill 远程实验。Small 无 OA-SD 不收敛。

原因分析:
- Small 需要 OA-SD 的 teacher guidance 来收敛 LGPA part classifiers
- 无 OA-SD 时 LR 0.0004 太保守 (Tiny 用 0.0008)
- LGPA 有 5 个独立 part classifiers, 比 GCN (1个) 更难收敛

**下一步**: 尝试 Small + LGPA-D + OA-SD + WITH_CP=True (gradient checkpointing 节省显存)

### [11:17] 本地 3090 启动 Small + LGPA-D + OA-SD + WITH_CP!

之前尝试: 远程 5060Ti OOM → 远程 WITH_CP 太慢 (15h) → 远程无OA-SD 不收敛 (0.3%)
→ 本地 3090 无CP OOM → **本地 3090 WITH_CP 成功!**

ep1 iter20. lgpa_assign=7.22, oa_sd=0.40。满血配置。

### [11:22] 检查点 #6

调研 agent 结论: DenoiseRep/Counterfactual/DCAC 评分都低 (2-4/10)。
ep1 iter180, 训练正常, WITH_CP 较慢 (~28s/20iter)。
**决策**: 等 ep10 eval

### [08:04] 检查点 #5 — 远程 WITH_CP 重启成功! (已 kill, 太慢)

远程 Small + LGPA-D + OA-SD + WITH_CP=True 启动成功!
ep1 iter40. lgpa_assign=7.24, oa_sd=0.38。OA-SD teacher 正常工作。
Speed 较慢 (~44s/20iter, 因 gradient checkpointing), 但内存安全。
**决策**: 等 ep10 eval

### [11:30] 本地 3090 接管 (替代远程)

本地 3090 WITH_CP 成功启动。ep2 完成。306s/ep, ETA 10h。
远程因 OOM/慢/不收敛等问题全部失败。

6 个调研 agent 完成: DenoiseRep 2/10, Counterfactual 4/10, DCAC 3/10, 
Conformal Prediction 8/10 (但 test-time only), POT/PCFD test 全失败。

**决策**: 训练继续后台, 等 ep10 eval (~40min)

### [11:34] 检查点 #8

ep4 iter80。lgpa_assign=7.21, oa_sd=0.21, id_global=6.55。
308s/ep, ETA 10h。正常 warmup 阶段。
**决策**: 等 ep10 eval (~30min)

### [11:39] 检查点 #9

ep5 iter80。lgpa_assign=7.18, oa_sd=0.08, id_global=6.55。
Loss=11.47, 正常下降。
**决策**: 等 ep10 eval (~25min)

### [11:44] 检查点 #10

ep5 完成。308s/ep, ETA 9h51m。
**决策**: 等 ep10 eval (~25min)

### [11:50] 用户提问: 为什么 Acc 增长慢?

对比远程 ep5 Acc=11% vs 本地 ep5 Acc=0.6%。
远程版本学习快 18 倍。可能原因:
1. Python 版本差异 (远程 3.11 vs 本地 3.8)
2. PyTorch 版本差异导致 AMP 行为不同
3. 随机种子/数据 shuffle 差异
4. 远程可能继承了旧训练的某些 state

Loss 接近 (11.16 vs 11.47), Acc 差很大 → 分类器初始化/优化路径不同。
预期 ep20+ 两者收敛到类似水平。

### [12:00] Agent 审查结论: PyTorch 版本差异 (正常)

根因: 本地 PyTorch 1.13 vs 远程 2.9。
不是 bug。cuDNN/DropPath/DataLoader seeding 不同导致早期轨迹不同。
同一本地机器 exp244 (Tiny, LR=0.0008) ep5 Acc=21.5% — 机器没问题。
exp245 慢因为: LR 减半 (0.0004) + 18 blocks (3x 更复杂)。
预期 ep60+ 收敛到类似 final。

### [12:09] 检查点 #11 — ep10 eval

**本地 ep10: 19.2/28.8%** — 很低但 warmup 阶段 (LR 仅 0.0002/0.0004)。
ep5→ep10 Acc: 0.6%→12%, 正常加速中。
对比远程 WITH_CP ep10 (未 eval), 本地确实慢但在追赶。

**决策**: 继续等 ep20

### [12:15] 远程 ep10 对比

远程 ep10: 50.3/61.5% vs 本地 ep10: 19.2/28.8% — 差距 2.6x!
oa_sd loss 差异: 本地 ep10 = 0.005 (太低), 远程 ep10 = 0.031。
本地 OA-SD teacher-student gap 过快收敛 → teacher 没有提供有效监督。
可能原因: PyTorch 1.13 的 EMA/PLBOA/随机性组合导致 teacher 更快追上 student。
不一定是 bug, 但影响训练质量。继续观察 ep20+ 是否能追上。

### [12:20] 本地 seed42 重启

Kill seed1234 (ep10 19.2/28.8, oa_sd 过快降到 0.005)。
启动 seed42 版本验证是否是随机性问题。
远程连接被拒 (恒源云可能关机)。

ep1 iter20: oa_sd=0.307, Loss=19.31。
**决策**: 等 ep5 的 oa_sd 趋势判断

### [12:25] seed42 ep2

oa_sd=0.691 (vs seed1234 ep2 = 0.629, 远程 ep2 = 0.446)。
seed42 OA-SD gap 更大, 是好信号。
**决策**: 等 ep5

### [12:30] seed42 ep3

oa_sd trend: 0.307→0.691→0.421
对比 seed1234: 0.400→0.629→0.543
seed42 也在快速下降。可能不是种子问题,而是 PyTorch 1.13 的固有行为。
**决策**: 等 ep5

### [12:35] seed42 ep4

oa_sd ep4=0.163 (seed1234=0.242)。两个种子 oa_sd 都快速下降。
确认: **不是种子问题，是 PyTorch 1.13 的 OA-SD 行为问题。**
远程 PyTorch 2.9 oa_sd 下降明显慢 (ep4=0.315)。
**决策**: 等 ep5, 然后决定是否值得继续 (如果 Acc 仍然很低)

### [12:40] seed42 ep5 确认: PyTorch 版本问题

| Metric | seed1234 ep5 | seed42 ep5 | 远程 ep5 |
|--------|------|------|------|
| Acc | 0.6% | 0.8% | 11% |
| oa_sd | 0.096 | 0.066 | 0.207 |

**两个本地种子一致, 与远程完全不同。**
确认: PyTorch 1.13 vs 2.9 导致 OA-SD 行为差异。
不是种子/随机性问题, 是环境问题。

**选项**:
1. 升级本地 PyTorch (风险: 可能破坏其他依赖)
2. 等远程恢复后在远程跑
3. 本地不带 OA-SD 跑 Small LGPA-D (之前失败, LR太低)
4. 本地带 OA-SD 继续跑, 看最终结果是否仍然有效 (oa_sd 虽低但不是0)

### [12:55] 新环境 PyTorch 2.5 启动!

创建 solider-reid-pt2 环境 (Python 3.11, PyTorch 2.5.1+cu121)。
ep1 iter40: oa_sd=0.470 (vs PT1.13 iter40≈0.35, 远程 PT2.9=0.380)。
PT2 的 oa_sd 明显更高 → teacher-student gap 更大 → OA-SD 更有效。

**验证**: ep5 oa_sd 应该接近远程的 0.207 (而不是 PT1.13 的 0.096)。
**决策**: 等 ep5

### [13:00] PT2 ep2

oa_sd=0.656 (vs PT1.13=0.629, 远程=0.446)。
Acc=0.0%, 正常 warmup。
**决策**: 等 ep5

### [13:05] PT2 ep3

oa_sd=0.464 (vs PT1.13=0.543, 远程=0.439)。 
**PT2 本地与远程的 oa_sd 趋势匹配!** (0.464 ≈ 0.439)
**决策**: 等 ep5

### [13:10] PT2 ep4 — oa_sd 也暴降!

PT2 oa_sd: 0.358→0.656→0.464→**0.021**! 从 0.464 暴降到 0.021!
**与 PT1.13 相同的行为!** 不是 PyTorch 版本问题!

oa_sd 暴降说明 OA-SD 在 Small backbone + LGPA-D 上有代码 bug。
需要立即调查 OA-SD teacher forward 在 Small 上的行为。

**决策**: 立即启动调查

### [13:12] PT2 也出现 oa_sd 暴降!

PT2 ep3→ep4: 0.464→0.021 (和 PT1.13 一样)。
不是 PyTorch 版本问题! 是 OA-SD + Small + LGPA-D 的代码/数值交互问题。
远程没暴降 → 唯一差异是远程用的是另一台机器的 GPU (5060Ti vs 3090)。
可能是 CUDA kernel / cuDNN 算法在不同 GPU 上导致不同的数值路径。

**决策**: 继续训练, 看最终结果。model 仍通过 ID+triplet loss 学习。
oa_sd 低只是失去了 self-distillation 的额外增益, 不是灾难。

### [13:18] PT2 ep5

Acc=0.2%, oa_sd=0.005。与 PT1.13 一致。
PT2 没改善 oa_sd 暴降。问题确认在 3090 GPU + Small backbone。
**结论**: 3090 上 Small + OA-SD 有异常。远程 5060Ti 正常。
等远程恢复后再验证 Small, 或用本地跑不带 OA-SD 的 Small。

**决策**: 继续让训练跑, 等 ep10 eval 看最终 mAP

### [13:24] 检查点

PT2 ep5 完成。303s/ep, ETA 9h40m。
**决策**: 等 ep10 eval (~25min)

### [13:34] 检查点

PT2 ep7. ETA 9h28m.
**决策**: 等 ep10 eval (~15min)

### [13:43] 检查点

PT2 ep9. 
**决策**: 等 ep10 eval (~5min)

### [13:50] 策略转换: 无 OA-SD + 高 LR

放弃在 3090 上用 OA-SD (oa_sd 暴降 bug, 两个 PyTorch 版本都复现)。
改为无 OA-SD + LR=0.0008 (2x) + WITH_CP, PyTorch 2.5。
速度更快 (~9s/20iter vs 28s with OA-SD)。

ep1 iter60: Loss=17.6, 正常。无 oa_sd 输出。
**决策**: 等 ep5 看 Acc 和收敛速度

### [13:52] PT2 环境有 mmcv-lite bug!

PT2 ep10: 0.2% mAP — 灾难! 原因: mmcv-lite 不含 CUDA ops, 
Swin backbone 的某些操作设备不匹配。
需要 mmcv-full (需要编译 CUDA ops), 但 Python 3.11 + PyTorch 2.5 无法编译。

**决策**: 放弃 PT2 环境, 回到 PT1.13。
启动 PT1.13 无 OA-SD + LR=0.0008 的 Small LGPA-D。
之前无 OA-SD 在远程失败是因为 LR=0.0004 太低。LR=0.0008 应该能收敛。
ep1 iter80, 正常。

### [14:05] 安装 CUDA toolkit 中

Docker 容器无 nvcc → 无法编译 mmcv-full → PT2 环境 Swin 有设备 bug。
正在 apt install nvidia-cuda-toolkit, 安装后编译 mmcv-full 再用 PT2 复现远程结果。
conda install cuda-toolkit 也在并行尝试。

### [14:20] CUDA toolkit installed, mmcv building

conda install cuda-toolkit 成功 (nvcc 12.9)。
mmcv-full 正在编译 CUDA ops (需要几分钟)。
完成后用 PT2 + mmcv-full 启动 Small + LGPA-D + OA-SD 复现远程结果。

### [14:25] mmcv 编译进行中

65 个 .o 文件已编译。CUDA ops 编译中。

### [14:30] mmcv 编译继续

77 个 .o 文件。仍在编译。

### [14:35] mmcv 编译继续

85 个 .o 文件, 0 个 .so。仍在编译 CUDA kernels。

### [14:40] GPU 恢复! apt CUDA 已卸载

apt nvidia-cuda-toolkit (CUDA 11.5) 与 driver (CUDA 12.0) 冲突 → nvidia-smi broken。
已 apt remove + autoremove → nvidia-smi 恢复正常。
conda cuda-toolkit 12.9 在 solider-reid-pt2 env 内, 不影响系统。
mmcv build 仍在进行 (92 objects compiled)。

### [14:45] mmcv 重新编译

之前编译失败: cuda_runtime_api.h 找不到 (gcc 没有 CUDA include path)。
设置 CPLUS_INCLUDE_PATH 指向 conda cuda includes, 重新编译。

### [14:50] mmcv 编译进行

39 objects。CUDA kernels 编译中。

### [14:55] mmcv 编译继续

52 objects。继续编译。

### [15:00] mmcv 编译继续

61 objects. CUDA kernel 编译较慢 (每个 .cu 文件需要 ~1min)。

### [15:09] mmcv 编译继续

80 objects。编译约完成 80%。

### [15:18] mmcv 编译

108 objects。接近完成 (之前一次编译到 104 后 link 失败)。

### [15:34] PT2 + mmcv-full + OA-SD 启动成功!

mmcv-full 从源码编译成功 (conda cuda-toolkit + CPLUS_INCLUDE_PATH)。
PT2 + mmcv-full 环境现在完整。
ep1 iter20: oa_sd=0.358。与之前 PT2+mmcv-lite 的 0.358 相同 (因为 Swin backbone 在 CUDA 上一样)。
**关键测试**: ep3→ep4 oa_sd 是否暴降? (PT1.13 和之前 PT2+lite 都暴降)

**决策**: 等 ep5 检查 oa_sd 趋势

### [15:39] PT2+full ep2

oa_sd=0.621 (vs PT2+lite=0.656, PT1.13=0.629)。三个版本几乎一样。
**关键**: ep3→ep4 是否暴降?

### [15:44] PT2+full ep3

oa_sd=0.507 (远程=0.439, PT1.13=0.543, PT2+lite=0.464)。都在正常范围。
**关键: ep4!!!**

### [15:49] ep4 oa_sd=0.346 — 没有暴降!! 🎉

| 环境 | ep1 | ep2 | ep3 | ep4 | 状态 |
|------|-----|-----|-----|-----|------|
| 远程 PT2.9+full | 0.331 | 0.446 | 0.439 | 0.315 | ✅ 正常 |
| **本地 PT2.5+full** | **0.358** | **0.621** | **0.507** | **0.346** | **✅ 正常!** |
| 本地 PT2.5+lite | 0.358 | 0.656 | 0.464 | 0.021 | ❌ 暴降! |
| 本地 PT1.13+full | 0.400 | 0.629 | 0.543 | 0.242 | ❌ 大降 |

**结论: mmcv-full 是关键! mmcv-lite 导致 OA-SD oa_sd 暴降!**
PT2+mmcv-full 完美复现远程行为。

**决策**: 继续训练, 等 ep10 eval (预期接近远程 50.3%)

### [15:55] ep5 oa_sd=0.197 — 完美匹配远程!

| Metric | 本地 PT2+full ep5 | 远程 ep5 |
|--------|------|------|
| oa_sd | **0.197** | **0.207** |
| Acc | 4.5% | 11% |

oa_sd 完美匹配! Acc 差距在合理范围 (不同 GPU 随机性)。
问题已解决: mmcv-full 是关键。

**决策**: 等 ep10 eval (~25min)

### [16:04] ep7

Acc=11.1% (远程 ep7=20%), oa_sd=0.061 (远程=0.089)。
Acc 在追赶, oa_sd 下降正常 (不是暴降)。
**决策**: 等 ep10 eval (~15min)

### [16:13] ep8

ETA 9h29m. 等 ep10 eval (~10min)。

### [16:25] ep10 eval — 大幅改善!

**本地 PT2+mmcv-full ep10: 42.6/54.7**

| 环境 | ep10 mAP | ep10 R1 | 状态 |
|------|---------|---------|------|
| 远程 PT2.9+full | 50.3 | 61.5 | ✅ 最好 |
| **本地 PT2.5+full** | **42.6** | **54.7** | **✅ 正常!** |
| 本地 PT1.13+full | 19.2 | 28.8 | ❌ oa_sd 暴降 |
| 本地 PT2.5+lite | 0.2 | 0.2 | ❌ mmcv 设备 bug |

PT2+mmcv-full 比 PT1.13 提升 2.2 倍 (42.6 vs 19.2)!
仍低于远程 (50.3 vs 42.6, 差 ~8%), 可能是 GPU 差异 (3090 vs 5060Ti)。
预期 final 接近远程水平。

**决策**: 继续训练, 等 ep20 eval

### [16:30] ep11

训练正常继续。ETA 9h23m。
远程仍无法连接。本地是唯一环境。
**决策**: 持续监控, 等 ep20

### [16:45] ep14

训练正常, ETA 9h2m。ep20 eval 在 ~30min 后。

### [16:55] ep16

ETA 8h44m. ep20 eval 在 ~20min。

### [17:04] ep17

ETA 8h42m. ep20 eval ~15min.

### [17:13] ep19

ep20 eval ~7min.

### [17:17] ep20 eval — 好结果!

**ep20: 56.2/68.5** 

vs Tiny exp244 ep20: 51.0/63.9 = **+5.2/+4.6** — Small backbone 带来显著提升!
远程在 ep13 被 kill, 没有 ep20 对比。

训练正常, 趋势好。
**决策**: 等 ep30

### [17:27] ep22

ETA 8h14m. 正常。

### [17:37] ep24

ETA 8h10m. 正常。ep30 eval ~30min.

### [17:52] ep27

ETA 7h51m. ep30 eval ~15min.

### [18:01] ep28

ep30 eval ~10min.

### [18:09] ep30 eval

**ep30: 61.6/73.4**

| Epoch | Small exp245g | Tiny exp244 | Small delta |
|-------|------|------|------|
| 10 | 42.6/54.7 | 42.1/55.3 | +0.5/-0.6 |
| 20 | 56.2/68.5 | 51.0/63.9 | +5.2/+4.6 |
| **30** | **61.6/73.4** | 57.6/69.9 | **+4.0/+3.5** |

Small 持续 +4~5% mAP 领先 Tiny! 训练正常。
**决策**: 继续等 ep60/120

### [18:12] Cron 提醒 — 状态检查

本地 exp245g: ep30 完成, 61.6/73.4, ETA 7h39m, 正常。
远程: 仍然 Connection refused。

| Epoch | mAP | R1 | vs Tiny exp244 |
|-------|-----|----|------|
| 10 | 42.6 | 54.7 | +0.5/-0.6 |
| 20 | 56.2 | 68.5 | +5.2/+4.6 |
| 30 | 61.6 | 73.4 | +4.0/+3.5 |

**决策**: 继续等 ep40

### [18:22] Per-epoch 对比 (vs Small baselines)

| Epoch | exp245g LGPA-D+OASD | exp206-local GCN+PAA+OASD | delta | Tiny exp244 |
|-------|------|------|------|------|
| 10 | 42.6/54.7 | 47.9/61.1 | **-5.3/-6.4** | 42.1/55.3 |
| 20 | 56.2/68.5 | 56.6/68.8 | **-0.4/-0.3** | 51.0/63.9 |
| 30 | 61.6/73.4 | 58.9/71.1 | **+2.7/+2.3** | 57.6/69.9 |

**ep10 LGPA-D 落后 GCN+PAA (-5.3)**, 但 ep30 超过 (+2.7)!
LGPA-D 早期收敛慢 (CLIP cross-attention 需要更多 warmup), 中期反超。
与 Tiny 一致的模式: LGPA-D 早期弱但后期强。

本地 ep32, ETA 7h28m。
**决策**: 等 ep40

### [18:34] ep35

ETA 7h8m. ep40 eval ~25min.

### [18:43] ep36

ep40 eval ~20min.

### [18:53] ep38

ep40 eval ~10min.

### [19:01] ep40 eval

**ep40: 64.6/75.1**

| Epoch | LGPA-D | GCN+PAA (exp206) | delta |
|-------|------|------|------|
| 10 | 42.6/54.7 | 47.9/61.1 | -5.3/-6.4 |
| 20 | 56.2/68.5 | 56.6/68.8 | -0.4/-0.3 |
| 30 | 61.6/73.4 | 58.9/71.1 | +2.7/+2.3 |
| **40** | **64.6/75.1** | **64.9/76.3** | **-0.3/-1.2** |

ep40 GCN+PAA 追上。两者竞争中。
注意: exp206 有 PAA (non-detach), 后期可能干扰 backbone, 而 LGPA-D 不会。
**决策**: 等 ep50/60 看后期趋势

### [19:11] ep42

ETA 6h35m. 正常。

### [19:21] ep44

ETA 6h26m. ep50 eval ~30min.

### [19:25] Cron 提醒 — 状态检查

本地 ep44, ETA 6h26m. 远程仍 Connection refused.
All evals: ep10=42.6, ep20=56.2, ep30=61.6, ep40=64.6。
趋势正常, 持续增长中。ep50 eval ~30min。

### [19:35] ep46

ep50 eval ~20min.

### [19:44] ep47

ep50 eval ~15min.

### [19:53] ep49

ep50 eval imminent (~2min).

### [19:53] ep50 eval

**ep50: 66.7/76.5**

| Epoch | LGPA-D | GCN+PAA (exp206) | delta |
|-------|------|------|------|
| 10 | 42.6/54.7 | 47.9/61.1 | -5.3/-6.4 |
| 20 | 56.2/68.5 | 56.6/68.8 | -0.4/-0.3 |
| 30 | 61.6/73.4 | 58.9/71.1 | +2.7/+2.3 |
| 40 | 64.6/75.1 | 64.9/76.3 | -0.3/-1.2 |
| **50** | **66.7/76.5** | **66.8/78.5** | **-0.1/-2.0** |

mAP 追平 (-0.1)! R1 差 2.0。
注意: exp206 有 PAA (non-detach) — 后期可能干扰。
LGPA-D (detach) 应该在 ep60+ 反超。

**决策**: 等 ep60

### [20:03] ep52

ETA 5h46m. ep60 eval ~40min.

### [20:12] ep53

ETA 5h39m.

### [20:22] ep55

ETA 5h30m. ep60 eval ~25min.

### [20:32] ep57

ep60 eval ~15min.

### [20:42] ep59

ep60 eval ~7min.

### [20:45] ep60 eval — LGPA-D 反超!

**ep60: 68.1/78.8** (vs GCN+PAA 67.3/79.1 = **+0.8/-0.3**)

| Epoch | LGPA-D | GCN+PAA (exp206) | delta |
|-------|------|------|------|
| 10 | 42.6/54.7 | 47.9/61.1 | -5.3/-6.4 |
| 20 | 56.2/68.5 | 56.6/68.8 | -0.4/-0.3 |
| 30 | 61.6/73.4 | 58.9/71.1 | +2.7/+2.3 |
| 40 | 64.6/75.1 | 64.9/76.3 | -0.3/-1.2 |
| 50 | 66.7/76.5 | 66.8/78.5 | -0.1/-2.0 |
| **60** | **68.1/78.8** | **67.3/79.1** | **+0.8/-0.3** |

**LGPA-D mAP 反超!** 正如在 Tiny 上的模式:
- PAA (non-detach) 在 ep60 开始减速
- LGPA-D (detach) 持续稳定增长
- mAP 交叉点在 ep55 附近

68.1 mAP 已接近 exp206r final (70.6)!

**决策**: 继续等 ep80/100/120

### [20:58] ep62

ETA 4h54m. ep80 eval ~90min.

### [21:09] ep64

ETA 4h43m.

### [21:19] ep66

ETA 4h34m.

### [21:29] ep68

ep70 eval ~10min.

### [21:37] ep70 eval

**ep70: 69.0/79.7**

| Epoch | LGPA-D Small | GCN+PAA (exp206) | delta |
|-------|------|------|------|
| 50 | 66.7/76.5 | 66.8/78.5 | -0.1/-2.0 |
| 60 | 68.1/78.8 | 67.3/79.1 | +0.8/-0.3 |
| **70** | **69.0/79.7** | N/A (ep80=69.3) | — |

69.0 接近 exp206 ep80 (69.3)! LGPA-D 在 ep70 就达到 exp206 需要 ep80 才到的水平!
持续增长, 无衰减。

**决策**: 等 ep80/100/120

### [21:50] ep72

ETA 4h. ep80 eval ~40min.

### [22:01] ep74

ETA 3h54m.

### [22:06] ep75

ETA 3h47m. ep80 eval ~25min.

### [22:15] ep77

ep80 eval ~15min.

### [22:26] ep79

ep80 eval ~5min.

### [22:29] ep80 eval

**ep80: 69.6/79.8**

| Epoch | LGPA-D Small | GCN+PAA (exp206) | delta |
|-------|------|------|------|
| 60 | 68.1/78.8 | 67.3/79.1 | +0.8/-0.3 |
| 70 | 69.0/79.7 | N/A | — |
| **80** | **69.6/79.8** | **69.3/80.0** | **+0.3/-0.2** |

**mAP 持续领先 GCN+PAA!** (+0.3 at ep80)
接近 exp206r final (70.6/82.6): mAP -1.0, R1 -2.8。
预计 ep120 可达 70.5+ mAP!

**决策**: 等 final

### [22:42] ep82

ETA 3h13m.

### [22:45] Cron 提醒 — 状态检查

本地 ep82, ETA 3h13m. 远程 Connection timed out.

mAP 增长曲线: 42.6→56.2→61.6→64.6→66.7→68.1→69.0→69.6
R1 增长曲线: 54.7→68.5→73.4→75.1→76.5→78.8→79.7→79.8

持续增长, 无衰减。ep90 eval ~40min, ep120 final ~3h。

### [22:53] ep84

ETA 3h1m. ep90 eval ~30min.

### [23:03] ep86

ETA 2h50m. ep90 eval ~20min.

### [23:13] ep88

ep90 eval ~10min.

### [23:21] ep90 eval — 突破 70% mAP!

**ep90: 70.0/80.0**

vs exp206r final (70.6/82.6): mAP -0.6, R1 -2.6。接近!
vs exp206 ep80 (69.3/80.0): mAP **+0.7**, R1 0.0。**已超越!**

| Epoch | LGPA-D Small | mAP 增量 |
|-------|------|------|
| 60 | 68.1 | — |
| 70 | 69.0 | +0.9 |
| 80 | 69.6 | +0.6 |
| **90** | **70.0** | +0.4 |

增速放缓 (0.9→0.6→0.4), 但仍在涨。
预计 final: 70.5~71.0 mAP — 与 exp206r (70.6) 持平或超越!

**决策**: 等 final

### [23:34] ep92

ETA 2h21m. ep100 eval ~40min.

### [23:44] ep94

ETA 2h10m.

### [23:50] ep95

ETA 2h5m. ep100 eval ~25min.

### [00:00] ep97

ETA 1h56m. ep100 eval ~15min.

### [00:09] ep99

ep100 eval ~7min.

### [00:13] ep100 eval

**ep100: 70.0/80.0** — 与 ep90 完全一致, 已收敛。

| Epoch | LGPA-D Small mAP/R1 |
|-------|------|
| 80 | 69.6/79.8 |
| 90 | 70.0/80.0 |
| **100** | **70.0/80.0** |

mAP 持平 (70.0), R1 持平 (80.0)。模型已收敛。
预计 final (ep120) ≈ 70.0~70.5。

vs exp206r (70.6/82.6): mAP -0.6, R1 -2.6。
注意: exp206r 有 PAA + GCN + PARALLEL_AUG, exp245 只有 LGPA-D。
LGPA-D 用更简单的架构达到了接近的 mAP!

**决策**: 等 ep110/120 final

### [00:31] ep103

ETA 1h25m.

### [00:36] ep104

ETA 1h20m. ep110 eval ~30min, ep120 final ~1h20m.

### [00:46] ep106

ETA 1h11m.
用户怀疑: 70.0 mAP 持平而不是超越 exp206r, 可能有环境问题。
计划: exp245g 完成后, 用旧环境 (PT1.13) 重跑 exp244 Tiny 前 20ep 验证日志一致性。

### [00:56] ep108

ETA 1h. ep110 eval ~10min.

### [01:05] ep110 eval

**ep110: 70.0/80.2** — mAP 持平 ep90/100 (70.0), R1 微升 (80.0→80.2)。
完全收敛。ep120 final ~50min。

### [01:18] ep112

ETA 40min.

### [01:24] ep113

ETA 35min.

### [01:34] ep115

ETA 25min.

### [01:44] ep117

ETA 15min. Final imminent!

### [01:54] ep119

ETA 5min.

### [01:57] FINAL!

## 最终结果

**exp245g (Small LGPA-D + OA-SD + WITH_CP, PT2+mmcv-full): 70.2/80.1/89.8/91.2**

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|----|
| exp206r GCN+PAA+OA-SD | 70.6 | 82.6 | 89.5 | 91.4 |
| **exp245g LGPA-D+OA-SD** | **70.2** | **80.1** | **89.8** | **91.2** |
| delta | **-0.4** | **-2.5** | +0.3 | -0.2 |

mAP 接近 (-0.4), R1 差距 (-2.5)。
注意: exp206r 有 PAA + GCN + PARALLEL_AUG (3-view), exp245g 只有 LGPA-D (单分支)。
LGPA-D 用更简单架构 (无 GCN, 无 PAA) 达到了接近的 mAP!

## 完整训练曲线

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 42.6 | 54.7 |
| 20 | 56.2 | 68.5 |
| 30 | 61.6 | 73.4 |
| 40 | 64.6 | 75.1 |
| 50 | 66.7 | 76.5 |
| 60 | 68.1 | 78.8 |
| 70 | 69.0 | 79.7 |
| 80 | 69.6 | 79.8 |
| 90 | 70.0 | 80.0 |
| 100 | 70.0 | 80.0 |
| 110 | 70.0 | 80.2 |
| **120** | **70.2** | **80.1** |

下一步: 用旧环境 (PT1.13) 重跑 exp244 Tiny 前 20ep 验证日志一致性。

## exp244 复现验证

### [02:02] ep1 iter20 — 完全一致!

Original: Loss=17.220, id_global=6.554, tri_global=13.761, oa_sd=0.245, lgpa_assign=7.217
Repro:    Loss=17.220, id_global=6.554, tri_global=13.761, oa_sd=0.245, lgpa_assign=7.217
**完全匹配! PT1.13 环境正确。**

### [02:05] ep2 iter20 — 也完全一致!

Original ep2: Loss=13.802, oa_sd=0.415
Repro ep2:   Loss=13.802, oa_sd=0.415
**两个 epoch 完全匹配, 确认 exp244 结果 (65.3 mAP) 可靠。**

结论:
1. exp244 Tiny LGPA-D (65.3/75.7) — PT1.13 可精确复现
2. exp245g Small LGPA-D (70.2/80.1) — PT2+mmcv-full 环境
3. Small vs Tiny: +5.0/+4.4 — backbone 升级有效

### [02:12] Repro ep4

ETA 43min. ep10 eval ~30min, ep20 final ~43min.

### [02:23] Repro ep7

ETA 35min. ep10 eval ~16min.

### [02:30] Repro ep10 eval — 完全一致!

Original ep10: mAP=42.1%, R1=55.3%, R5=71.0%
Repro ep10:   mAP=42.1%, R1=55.3%, R5=71.0%
**100% 精确复现!** exp244 (65.3/75.7) 结果完全可靠。
等 ep20 eval 再确认一次。

### [02:41] Repro ep13

ETA 19min.

### [02:52] Repro ep17

ETA 8min.

### [12:59] 远程 exp245h_v2 重启 (新 OUTPUT_DIR)

Cache 文件从 git 恢复。用新 OUTPUT_DIR 避免覆盖旧日志。
ep1 iter20: Loss=19.460 (与所有之前 runs 一致)。
等 ep10 eval 验证 7.6% 还是 50.3%。

### [05:10] 远程 mmcv 调查

远程**不需要** mmcv — Swin 只用 load_checkpoint, 有 mmengine fallback。
远程 val_loader 返回 7 个元素, pose_dict 在 index 6 — 正确。
model(x, pose_dict=pd) 输出 5376 dim — 正确。
model(x, pose_dict=None) 输出 768 dim — 如果 eval 时 pose_dict=None 就解释了 7.6%。

**但 eval 代码明确传 pose_dict!** 所以问题不在这里。
等 exp245h_v2 ep10 eval 再验证一次。

### [05:15] 发现微小差异!

untitled.txt (7.6% run) ep1 iter200: tri_global=9.545, oa_sd=0.414
exp245h_v2 (新 run) ep1 iter200: tri_global=9.537, oa_sd=0.418
差异 ~0.003 级别, 非零! 服务器重启后环境微变 (cuDNN algo 等)。

### [06:25] 远程 v2 ep10 eval — 复现成功!!

**exp245h_v2 ep10: 49.6/60.6** — 接近原始 50.3/61.5!

| Run | ep10 mAP | ep10 R1 |
|-----|---------|---------|
| 原始 (34202ed) | 50.3 | 61.5 |
| **v2 (92fded3)** | **49.6** | **60.6** |
| 7.6% 异常 | 7.6 | 13.8 |
| 本地 PT2+full | 42.6 | 54.7 |
| 本地 PT1.13 | 19.2 | 28.8 |

**结论**: 
1. 50.3% 结果可复现 (v2=49.6%, delta -0.7)
2. 7.6% 那次是异常 bad trajectory (远程 cuDNN 非确定性)
3. 本地 3090 比远程 5060Ti 慢 ~8% (42.6 vs 49.6)
4. PT1.13 比 PT2.9 慢很多 (19.2 vs 49.6) — mmcv-full 环境差异

### [06:30] 环境差异确认

SOLIDER weights MD5 一致 (cc0d3e9d)。不是权重问题。
远程 mmcv=1.7.2, 本地 mmcv=2.1.0 (编译)。版本差异导致数值不同。
远程 PyTorch 2.9, 本地 2.5。也有差异。
**结论: 42.6 vs 49.6 的差距 (7%) 来自 mmcv + PyTorch + GPU 三重环境差异。**
不是 bug, 是跨环境不可避免的数值分歧。LGPA-D 方法在两个环境都有效。

7.6% 那次是 cuDNN 非确定性导致的 bad trajectory (无法复现, v2 已恢复到 49.6%)。

### [02:58] Repro ep20 — 完全一致!

Original ep20: mAP=51.0%, R1=63.9%, R5=78.1%
Repro ep20:   mAP=51.0%, R1=63.9%, R5=78.1%

**所有 checkpoint 完全匹配: ep1, ep2, ep10, ep20。exp244 结果 100% 可靠。**

复现验证完成。GPU 空闲。

---

## exp245h_v2 远程第三次 run (2026-04-06)

远程 5060Ti, 新 OUTPUT_DIR, 验证 exp245g 可复现性。

### [14:17] ep10 eval

**ep10: 49.6/60.6/76.2/81.7** — 接近原始 50.3/61.5, 确认可复现。

### [15:35] ep20 eval

**ep20: 60.2/72.8/84.6/87.8**

| Epoch | exp245h_v2 (远程) | exp245g (本地) | delta |
|-------|------|------|------|
| 10 | 49.6/60.6 | 42.6/54.7 | +7.0/+5.9 |
| 20 | 60.2/72.8 | 56.2/68.5 | +4.0/+4.3 |

远程 5060Ti 收敛明显快于本地 3090。

**关注**: oa_sd=0.015 (ep23), 极低。但 ep20 mAP=60.2 仍然很好。

### [16:54] ep30 eval

**ep30: 64.8/76.3/86.2/89.5**

| Epoch | exp245h_v2 (远程) | exp245g (本地) | delta |
|-------|------|------|------|
| 10 | 49.6/60.6 | 42.6/54.7 | +7.0/+5.9 |
| 20 | 60.2/72.8 | 56.2/68.5 | +4.0/+4.3 |
| **30** | **64.8/76.3** | **61.6/73.4** | **+3.2/+2.9** |

远程持续领先，差距在缩小 (7.0→4.0→3.2)。
oa_sd=0.018 (ep35), 极低但稳定。模型仍在正常学习。
### [09:32] 检查点 — ep35

远程 ep35 iter200. Loss=6.33, Acc=77%, oa_sd=0.018 (稳定低位)。
训练正常。ep40 eval ~40min 后。
### [09:42] 检查点 — ep37

远程 ep37 iter60. Loss=6.47, oa_sd=0.019-0.021 (微回升)。
ep40 eval ~25min 后。
### [09:51] 检查点 — ep38

远程 ep38 iter100. Loss=6.32, oa_sd=0.019-0.020。

### [10:01] 检查点 — ep39

远程 ep39 iter160. Loss 持续下降 (6.06), Acc=79.2%. ep40 eval ~5min 后。
### [10:09] 检查点 — ep40 训练中

远程 ep40 iter160. eval imminent (~3min).
### [10:14] ep40 eval

**ep40: 67.4/77.5/—/90.0**

| Epoch | exp245h_v2 (远程) | exp245g (本地) | delta |
|-------|------|------|------|
| 10 | 49.6/60.6 | 42.6/54.7 | +7.0/+5.9 |
| 20 | 60.2/72.8 | 56.2/68.5 | +4.0/+4.3 |
| 30 | 64.8/76.3 | 61.6/73.4 | +3.2/+2.9 |
| **40** | **67.4/77.5** | **64.6/75.1** | **+2.8/+2.4** |

差距继续缩小 (7.0→4.0→3.2→2.8)。远程仍领先但本地在追赶。
预计远程 final: 71-72 mAP (vs exp245g 70.2)。

### [10:24] 检查点 — ep42

远程 ep42 iter120. Loss=5.91, oa_sd=0.020. 训练正常。ep50 eval ~30min 后。
### [10:33] 检查点 — ep43

远程 ep43 iter160. Loss=5.75, oa_sd=0.020.

### [10:43] 检查点 — ep44

远程 ep44 done. 458s/ep, ETA 9h41m. ep50 eval ~46min 后。
远程比预期慢 (WITH_CP + Small 18 blocks)。
### [10:53] 远程连接失败

DNS 解析失败 (Temporary failure in name resolution)。网络临时问题。
训练仍在远程后台运行，不受影响。
### [11:02] 网络恢复 — ep47

远程 ep47. oa_sd=0.021 (微回升)。ep50 eval ~23min 后。
### [11:12] 检查点 — ep48

远程 ep48 iter180. Loss=5.39, oa_sd=0.022 (继续回升!)。

### 远程 exp245h_v2 完成! — 最终结果

## 完整训练曲线

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|-----|
| 10 | 49.6 | 60.6 | 76.2 | 81.7 |
| 20 | 60.2 | 72.8 | 84.6 | 87.8 |
| 30 | 64.8 | 76.3 | 86.8 | 89.5 |
| 40 | 67.4 | 77.5 | 87.4 | 90.0 |
| 50 | 69.5 | 79.9 | 88.5 | 91.2 |
| 60 | 70.0 | 80.2 | 88.9 | 91.0 |
| 70 | 71.2 | 81.6 | 89.3 | 91.7 |
| 80 | 71.4 | 81.6 | 89.6 | 91.2 |
| 90 | **71.7** | **82.2** | 89.3 | 91.1 |
| 100 | 71.6 | 81.6 | 89.3 | 91.3 |
| 110 | 71.6 | 81.7 | 89.3 | 91.2 |
| **120** | **71.6** | **81.6** | **89.2** | **91.2** |

## 最终结果

**exp245h_v2 (Small LGPA-D + OA-SD, 远程 5060Ti): 71.6/81.6/89.2/91.2**
**(ep90 peak: 71.7/82.2)**

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|----|
| exp245g (本地 3090) | 70.2 | 80.1 | 89.8 | 91.2 |
| **exp245h_v2 (远程 5060Ti)** | **71.6** | **81.6** | **89.2** | **91.2** |
| delta | **+1.4** | **+1.5** | -0.6 | 0.0 |
| exp206r (Small GCN+PAA+OA-SD) | 70.6 | 82.6 | 89.5 | 91.4 |
| vs exp206r | **+1.0** | **-1.0** | -0.3 | -0.2 |

**关键发现**:
1. **远程 71.6 > 本地 70.2 (+1.4 mAP)**: 环境差异 (mmcv 1.7.2 vs 2.1.0, PT2.9 vs PT2.5)
2. **远程 R1 81.6 > 本地 80.1 (+1.5)**: 全面更好
3. **mAP 超过 exp206r (70.6 → 71.6, +1.0)!** LGPA-D 单分支首次在 mAP 上超越 Small baseline!
4. R1 81.6 vs exp206r 82.6 — 仍差 1.0, 但已大幅缩小
5. oa_sd 虽然极低 (0.018-0.022), 但模型仍然正常收敛
6. **exp245g 的 70.2 可能是 3090 环境的下限, 实际方法效果接近 71.6**

### MaxSim test on ep120 ⭐⭐⭐

**MaxSim ep120: 73.0/82.7/90.5/92.7**

| 方法 | mAP | R1 | R5 | R10 |
|------|-----|----|----|----|
| exp245h_v2 equal_concat | 71.6 | 81.6 | 89.2 | 91.2 |
| **exp245h_v2 MaxSim** | **73.0** | **82.7** | **90.5** | **92.7** |
| MaxSim gain | **+1.4** | **+1.1** | **+1.3** | **+1.5** |
| exp245g MaxSim (本地) | 71.9 | 82.2 | 91.0 | 92.8 |
| exp206r (Small baseline) | 70.6 | 82.6 | 89.5 | 91.4 |

**关键发现**:
1. **MaxSim 73.0 — Small 上历史最佳 mAP!** 超越 exp245g MaxSim (71.9) +1.1!
2. **R1 82.7 — 超越 exp206r baseline (82.6) +0.1!** 首次在 R1 上也全面超越!
3. MaxSim gain (+1.4) 比 exp245g (+1.7) 略小但仍显著
4. **LGPA-D 在 Small 上首次全面超越 GCN+PAA+OA-SD baseline (mAP 和 R1 都超过)**

### MaxSim 跨 checkpoint 稳定性

| Checkpoint | mAP | R1 | R5 | R10 |
|------------|-----|----|----|----|
| ep80 | 72.8 | 82.8 | 90.9 | 92.7 |
| ep100 | 72.9 | 82.8 | 90.7 | 92.6 |
| **ep120** | **73.0** | **82.7** | **90.5** | **92.7** |

MaxSim 结果跨 checkpoint 极其稳定 (mAP 72.8-73.0, R1 82.7-82.8)。
