# Gallery Topology Failures in Strong Person ReID —— Hubness 诊断 analysis 素材

> 本 session 5 死角(B/GOPL/Hubness-method/OSAC/RMA)+ 视频 no-go 后, 唯一站得住的真发现。
> **定位: 诊断/分析贡献(analysis short), 不是训练方法稿**——remedy 被 k-reciprocal 占(D2)。诚实写清这条边界正是论文价值, 别包装成方法。
> 数据来源: `hubness_logs/hub_full_market.log`, `hub_full_occluded_duke.log`(全量, 冻结强 ckpt, 零训练)。

## 1. 核心叙事(motivation)
强 ReID(Swin/SOLIDER, Market 94.6 / Occluded-Duke 73)仍有残差失败。大家默认失败来自"某个 query 没匹配好"(pairwise similarity 病)。**重定义: 强 ReID 的残差失败是 gallery topology 的 many-to-one 吸附——少数 gallery 图像成为很多不同身份 query 的误吸附点(negative in-degree hub)。** ReID 不是独立 pair matching, 是 directed kNN graph retrieval; 隐藏变量 = gallery 的负向 in-degree, 不是 hard-negative distance(hard 对一个 anchor 近; hub 对很多不同身份都近)。

## 2. 可测隐藏变量
- gallery 负向 in-degree: `H_k(g) = #{ q | g ∈ top-k(q) 且 y_g ≠ y_q }`(被多少不同身份 query 误放进 top-k)。
- query hub mass: `M(q) = Σ_{g∈topk(q), y_g≠y_q} H_k(g)`。

## 3. 主结果(M(q) 解释 AP 误差, 控廉价代理后仍成立)
| 指标 | Market (mAP 94.61) | Occluded-Duke (mAP 73.05) |
|---|---|---|
| rho(AP-error, M(q)) | +0.2765 | **+0.6467** (perm-p 0.0010) |
| partial(AP-err, M \| norm+margin+camera+#pos) [D3] | +0.3314 | **+0.6035** |
| top-1% hub 吃 false-top1 (k=10) | 22.2% | **26.7%** (k=5: 30.7%) |
| 零训练 hub-fix 干预 (score'=cos−λlog(1+H_k)) | +0.31 mAP / +1.13 R1 | **+1.51 mAP / +5.07 R1** |

→ **越难/未饱和的 benchmark, hub 病越大**(Occluded-Duke 上 M 解释力是 Market 的 2.3×, 干预增益 5×)。

## 4. 破坏对照(证诊断真实, 非伪信号/非廉价代理)
- **D1 置换 H_k**: shuffle 后干预增益 → +0.002(Occluded-Duke), 真信号。
- **D3 控代理**: 控 feature-norm + top1-margin + camera-pair + #gallery-pos 后, M(q) 偏相关仍 +0.60(Occluded-Duke), 不是旧难度代理。
- **D4 负向 vs 全部 in-degree**: 干预增益 NEG/ALL/POS = +1.51 / +0.000 / +0.000; rho(AP-err, M_neg)=+0.65 而 M_all=+0.02——**关键是跨身份误吸附(负向), 不是单纯"热门样本"**。(注: Spearman(H_neg,H_all) Market −0.03 / Occluded-Duke +0.57, 难集上负向与热门开始重叠, novelty 在难集稍弱。)

## 5. ★诚实边界(为什么是诊断不是方法)——D2
zero-training hub-fix 的 mAP 增益**被现成 test-time 后处理完全盖过**:
| | Market | Occluded-Duke |
|---|---|---|
| hub-fix (我们) | +0.31 | +1.51 |
| 同相机降权 | +0.67 | +3.13 |
| k-reciprocal (plain) | +1.26 | **+10.98** |
| k-reciprocal (camera-aware) | — | +10.39 |

hub-fix 始终 ≤ 同相机降权 ≤ k-reciprocal, 三层都压它; 难集上 k-reciprocal 把差距拉到 7× **不是缩小**。训练端 anti-hub embedding 经 de-risk + 红蓝辩论(蓝队 8/10 胜)判定: 即使把增益榨进 embedding, 也是和已有 test-time 后处理抢一块它们做得更好的蛋糕。**故: 这是 where-strong-ReID-fails 的诊断, 不是 how-to-fix 的方法。**

## 6. 撞车边界(写作时引用切开)
- vs k-reciprocal/CA-Jaccard(test-time re-rank): 我们是诊断变量定义, 不声称更好的 re-rank。
- vs HAL/NeighborRetr(cross-modal hubness-aware training): 图文检索非 person ReID; 且我们坐实 person ReID 训练端 remedy 被 re-rank 吞, 不重复它们的 training-time claim。
- vs hard-negative mining: H_neg 是全局误吸附(对很多身份), 非 anchor-local difficulty。

## 7. 若要撑成 B 类 analysis short, 还需(留用户定夺)
- 多 backbone(ResNet/ViT/Swin)+ 多数据集(已有 Market/Occluded-Duke, 补 MSMT)证 hub 病普适。
- failure 样例可视化(高 H_k hub gallery 长什么样: 是否泛化/低质/多人/特定服饰)。
- 与 re-ranking 的互补性叙事(hub-fix R1 在 Market 赢 k-reciprocal +1.13 vs −0.12, 是 rank-repair 互补轴)。
- 诚实定位: "诊断 + 未来方向", 不强行训练方法。

## 7.5 Failure-case 表征 + 机制（零训练已做, 2026-06-24）
取 occluded_duke top-30 高 H_k hub + 30 相机匹配对照, frozen exp255 特征 + 取图（`hub_failure_grid_FINAL.png`, `hub_zoom_top18.png`, 脚本 `hub_failure_characterize.py`）。
- **hub = 非身份明亮场景过度编码**: ~7-8/30 hub 被大面积明亮橙车+砖广场占满（人小/被裁）, RANDOM 基率仅 0-1/30 → 该场景在 hub 真实过表征。量化（hub vs 相机匹配对照, 且 cam0 内复现排除相机混杂）: brightness 140 vs 119, bright_frac(>200) **0.245 vs 0.113**, colorfulness **18 vs 11**。
- **机制铁证**: 30 hub 平均两两余弦 **0.166**（对照 0.025 / 随机 0.051）, 跨 **24 个不同身份**却特征抱团; 每 hub 的 10-NN 里 **26.7% 也是 top-1% hub**（基率 1%, **27× 富集**）。→ 模型靠**非身份共同因子（明亮场景）**把不同身份 crop 聚到一起 = 跨身份吸附根因。这是诊断的 mechanism-level 直观确证。
- **method 种子 = 死（诚实）**: 唯一可操作方向 = 背景/非人区域抑制, 但 (a) 具体复现物是一辆特定橙车@cam1 不可泛化; (b) 泛化成背景抑制 = 团队已封板证负的 PSG/pose-mask 旧雷; (c) remedy 仍被 k-reciprocal 占。**故此图是高价值 analysis figure（直观坐实非身份场景吸附）, 非方法种子。**
- **相机控制**（必带）: hub 集中 cam1(20/30), 用相机匹配对照 + cam0-内复现两道控制确认亮度/色彩信号非相机伪信号。

## 7.6 ★诊断真伪复核（P0/P3/P4, 2026-06-24, 零训练 numpy + frozen exp255/exp260b）

> codex 红队指出 M(q) 含 query q 自己对 H_k 的贡献 → 潜在 circular。三道复核脚本（`hub_verify_p0_p4.py` / `hub_verify_p0c_deep.py` / `hub_verify_p3_mask.py`, log `/tmp/hub_p0p4_{oduke,market}.log`、`/tmp/hub_p3_oduke.log`）逐条验证。**结论：原 rho+0.60/+0.65 被 circular self-loop 高估, 且 M(q) 控住 trivial 代理 `#false-in-topk` 后无独立信号——诊断的"hub/拓扑"框架不成立, 退化为"top-k 里错的多"。诚实记录。**

### P0a leave-one-query-out（去 q 自身对 H_k 的 +1）
| | Occluded-Duke | Market |
|---|---|---|
| rho(AP-err, M_raw) | +0.6467 | +0.2765 |
| rho(AP-err, M_loo) | **+0.4821** | **+0.2272** |
| Δ(raw→loo) | **−0.1646** | −0.0493 |
LOO 后 Occluded-Duke 的 headline 从 +0.65 掉到 +0.48（掉 0.16, 即原值被 self-loop 高估约 25%）。仍 >0 且 perm-p≤0.001, 没归零, 但"+0.60"是高估值。

### P0b held-out split（A 半 query 估 H_k, B 半算 M 预测其 AP-err, 物理上无 self-loop）
| | Occluded-Duke | Market |
|---|---|---|
| primary split rho | +0.3383 [95%CI +0.27,+0.39] perm-p0.0005 | +0.2772 [95%CI +0.09,+0.32] perm-p0.001 |
| 20-split mean±std | **+0.3294 ± 0.029** | **+0.2467 ± 0.065** |
完全去 circular 后 rho 稳定在 **+0.33（OD）/ +0.25（Market）**。这是诊断"干净"版本的真实强度——比原报的 +0.60 低一半。

### P0c ★控 trivial 代理 `#false-in-topk`（= q 自己 top-k 里有几个异身份, 无需任何 hub/图概念）——决定性
| | Occluded-Duke | Market |
|---|---|---|
| rho(AP-err, #false-in-topk) | +0.6567 | +0.2837 |
| rho(AP-err, M_loo) | +0.4821 | +0.2272 |
| Spearman(M_loo, #false-in-topk) | +0.7772 | +0.8820 |
| **partial rho(AP-err, M_loo \| #false-in-topk)** | **−0.0595** | **−0.0510** |
| partial rho(AP-err, #false \| M_loo)（反向）| +0.5115 | +0.1816 |
| partial rho(AP-err, M_loo \| 全 6 强代理) | −0.0607 (n=741) | −0.1482 (n=842) |
**两数据集一致: 控住 `#false-in-topk` 后 M(q) 偏相关塌到 ≈0（甚至略负）, 而 `#false-in-topk` 控住 M 后仍 +0.51/+0.18。** 即 M(q) 完全可被"top-k 里错的多少个"这个 trivial 计数取代, 反之不行。原 §3/§4 的 D3"partial+0.60"只控了较弱的 norm/margin/camera/#pos, **从未控 `#false-in-topk`**——这是 D3 漏掉的最致命代理。→ **"gallery 负向 in-degree / many-to-one 拓扑"作为 AP-err 解释变量, 相对 trivial 代理无增量价值。**

### P3 机制因果（top-30 hub gallery, frozen 重提特征 + pose-heatmap 人体 mask; sanity: 无 mask 重提 vs 缓存 cos=0.999 OK）
| 条件 | HUB 平均 H_k | CTRL（相机匹配, H_k=0 池） |
|---|---|---|
| orig | 4.80 | 0.03 |
| bg_masked（留人, 灰填背景）| **2.53（−47%）** | 0.07 |
| person_masked（留背景, 灰填人）| **0.63（−87%）** | 0.10 |
- 去背景 → hub 吸附 **−47%**: 支持"场景/背景贡献了部分跨身份吸附"。
- **但** 去人 → **−87%**（掉更多）: **与严格"非身份场景因子"预测相反**（预测 person_masked 应 ≥ orig）。说明 hub 信号**过半在人体 crop 自身**, 不是纯背景场景。
- → P3 **只给"场景 over-encoding"部分支持**, 不能干净坐实"hub = 非身份场景因子"。§4.3/§7.5 的强机制叙事需降级为"背景贡献约一半, 但人体 crop 仍是吸附主体"。
- 注: 灰填本身是破坏性扰动（对照组 0.03→0.07/0.10 略升, 噪声级, 说明扰动非均匀破坏, 但 person_masked 大掉这一点仍真实）。

### P4 ★k-reciprocal 重定位（每 query AP_rerank − AP_base 增益 vs M(q)）——唯一正向 reframe
| | Occluded-Duke | Market |
|---|---|---|
| k-reciprocal mAP 增益（全局）| +10.98 | +1.26 |
| rho(M_raw, per-q k-recip 增益) | +0.244 [95%CI +0.19,+0.28] perm0.0005 | +0.251 [95%CI +0.14,+0.29] perm0.0005 |
| rho(M_loo, per-q 增益) | +0.191 | +0.273 |
| **分箱: 最低 M 五分位 → 最高 M 五分位 平均增益** | **+4.79 → +20.67（4.3×）** | **+0.65 → +2.93（4.5×）** |
连续 rho 只 ~+0.25（被大量 M=0 query 稀释, 5 箱里 3 箱 M≈0）, 但**分箱趋势干净**: 高 M(q) query 正是 k-reciprocal 修复增益最大的（OD 4.3×, Market 4.5×, 两集一致）。→ **P4 reframe 在分箱意义上成立**: "M(q) 标记了现成 re-rank 工具受益最大的 query"。这是验证后唯一仍站得住、可写的正向叙事（但注意 §P0c: M 此处可能也只是 `#false-in-topk` 的代理, reframe 同样可用 #false 复述, 严谨写作需在 P4 也加 `#false` 对照)。

## 8. 结论（2026-06-24 复核后修订）
~~M(q) 干净解释 AP 误差(rho+0.60, 控代理后仍在)~~ → **修订**: (1) rho+0.60/+0.65 含 circular self-loop, 去 circular（LOO/held-out）后降到 +0.33（OD）/+0.25（Market）; (2) **决定性**: 控 trivial 代理 `#false-in-topk` 后 M(q) 偏相关 ≈0（两集一致 −0.06/−0.05）, 即"gallery 负向 in-degree / 拓扑"框架相对"top-k 里错几个"无增量解释力——原 D3 漏控了这个最致命代理; (3) P3 机制只半支持（去背景 −47%, 但去人 −87% 反更大, 非纯场景因子）; (4) **唯一仍正向**: P4 高 M(q) query 正是 k-reciprocal 修复最多的（分箱 4.3×/4.5×, 两集一致）。→ **作为"诊断变量"的 headline 站不住**（被 trivial 代理吃掉）; 若要保留, 只能定位成"M(q)/负向 in-degree 标记 re-rank 高收益 query"的弱 reframe, 且必须诚实写明它相对 `#false-in-topk` 无增量。**当前形态不足以撑 analysis short 的核心 claim。**
