# Lattice-Marginalized ReID — 零训练 kill-switch 结果

> 脚本 `cvpb_lattice_killswitch.py`，设计 `cvpb_lattice_killswitch_DESIGN.md`，机会来源 `litreview2/explore20/clean/d_8.txt` 机会1。
> 机器 lab-3090-d（frozen exp260b market，pose-OFF 全局向量）。HR sanity mAP=94.43（=exp260b ref 94.4，提取忠实）。
> log：`/tmp/cvpb_lattice_market.log`（run1，h=16 完成后 h=24 OOM）、`/tmp/cvpb_lattice_market2.log`（run2，streaming 修内存，h=16 复现 run1）。
> **零训练，无 backward，frozen + numpy/PIL。**

## 判据回顾
- **GO**：h≤32 时 rank volatility 明显 **且** phase-ensemble ≥+2 mAP **且** 明显超普通 TTA（LAT-TTA>0）**且** phase-var 独立于 #false。
- **DEAD**：phase-var≈TTA-var / ensemble≈单图 / ensemble≈普通TTA / phase-var 被 #false 吃掉。

## ★核心结果 —— h=16（run2，与 run1 双跑一致）

| 测量 | 数值 | 解读 |
|---|---|---|
| **lattice phase-var**（K变体两两 1-cos 均值）| **0.1162** | 采样格点把 frozen 特征移动很大 |
| ordinary TTA var（同 K，参照）| 0.0156 | lattice 是 TTA 的 **7.5×** |
| single-LR→HR drift | 0.4319 | 单图 LR 绝对失真（16px 极大）|
| **top1 一致率**（跨相位 vs 标准 LR）| **0.506** | 半数变体连 rank-1 都不一致 |
| top10 Jaccard（标准 vs 各变体）| 0.464 | top10 集合跨相位重叠<一半 |
| **跨相位 rank-1 身份翻转**：distinct IDs / 翻转query占比 | **2.45 / 74.9%** | **74.9% query 的 rank-1 身份随采样相位翻转** |
| 单一 bicubic LR mAP / R1 | 42.645 / 44.745 | baseline |
| phase-lattice mean-feat | 43.339 / 44.121（d+0.69）| 均值融合一般 |
| **phase-lattice MaxSim** | **46.872 / 49.287（d+4.23）** | ★边缘化增益 |
| ordinary-TTA mean-feat | 43.150（d+0.51）| 控制 |
| **ordinary-TTA MaxSim** | **43.833（d+1.19）** | ★同 K 同融合的 TTA 控制 |
| **LATTICE-MINUS-TTA** | **+3.04**（4.23−1.19）| ★lattice 特有增益（非"多枪"）|

**生死对照（C）—— phase-var 解释失败 vs trivial 代理：**
| | rho(AP-err, ·) | partial |
|---|---|---|
| phase-var | +0.368 | — |
| #false-in-topk [trivial] | +0.911 | — |
| single-LR→HR drift [LR severity] | +0.777 | — |
| **phase-var \| #false** | — | **+0.189**（独立于 #false，过 Hubness 关）|
| **phase-var \| #false + drift** | — | **+0.027**（再控 LR severity→塌到≈0）|
| #false \| phase-var（反向）| — | +0.900 |

## 各 height（h=24/32/48 待 run2 完成补全；h=32 来自 150-query smoke）

| h | phase-var | TTAvar | top1stab | idFlip | flip% | single | lat-MaxSim | tta-MaxSim | LATgain | LAT-TTA | pv\|#false | pv\|#false+drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **16** | 0.116 | 0.016 | 0.506 | 2.45 | **74.9%** | 42.65 | **46.87** | 43.83 | **+4.23** | **+3.04** | +0.189 | +0.027 |
| **24** | 0.058 | 0.014 | 0.718 | 1.47 | **31.3%** | 69.31 | **72.98** | 70.30 | **+3.67** | **+2.68** | +0.211 | +0.062 |
| **32** | 0.033 | 0.012 | 0.837 | 1.12 | 9.7% | 81.93 | **83.98** | 82.54 | **+2.05** | **+1.44** | +0.209 | +0.084 |
| 48 | 0.016 | 0.012 | 0.935 | 1.01 | 1.2% | 90.44 | 91.02 | 90.61 | +0.58 | +0.41 | +0.180 | +0.109 |
| 32 (smoke150) | 0.033 | 0.013 | 0.807 | 1.15 | 12.0% | 77.24 | 78.92 | 77.67 | +1.68 | +1.25 | +0.247 | _NA_ |

**清晰的单调衰减（4 height 全跑完，full 3368-query，分辨率越高格点效应越弱）**：phase-var 0.116→0.058→0.033→**0.016**（h=48 已逼近 TTA-var 0.012 地板）；rank-1 翻转 74.9%→31.3%→9.7%→**1.2%**；lat-MaxSim 增益 +4.23→+3.67→+2.05→**+0.58**；LAT-TTA +3.04→+2.68→+1.44→**+0.41**。
- **h=16 / h=24 强 GO**（增益 +3.7~4.2，LAT-TTA +2.7~3.0，rank-1 翻转 31~75%）。
- **h=32 临界**（增益恰 +2.05 卡 +2 线，LAT-TTA +1.44 仍正；full +2.05 与 smoke +1.68 一致）。
- **h=48 机制消失**（增益 +0.58<2，rank-1 翻转仅 1.2%，单图已 90.4 接近 94.4 天花板）。
→ lattice 边缘化是**低分辨率现象**（h≤24 强、h=32 临界、h≥48 无），正是 d_8.txt 预测的 regime。**脚本自带 AUTO VERDICT: GO（h=16/24 过全部四关：volatility ✓ / 增益≥2 ✓ / LAT-TTA>0 ✓ / phase-var|#false>0 ✓）。**

## 结论与 VERDICT

**机制存在（强）**：在 h=16，同一人的 LR 采样格点族确实把 frozen 特征移动 0.116（**普通 TTA 的 7.5×**），**74.9% 的 query 其 rank-1 身份随亚像素相位 / ±1 LR-pixel bbox / antialias kernel 翻转**。这正坐实 d_8.txt 的重定义："低分辨率不是模糊缺细节，而是采样格点不确定性——同一人=一族 alias 观测，模型只见过一个"。

**边缘化有效且明显超普通 TTA（GO 的承重证据）**：K=9 phase-MaxSim 边缘化把 16px mAP 从 42.6→46.9（**+4.23**），而**同 K 同 MaxSim 融合的普通 TTA 只 +1.19**，**LAT-TTA=+3.04**。MaxSim 取 K 中最优本可能仅靠"多枪"涨点，但 TTA-MaxSim 同样多枪只拿 +1.19——**差值证明增益来自 lattice 结构本身，不是 TTA 换名**。双跑一致（run1 +4.25 / LAT-TTA +3.04，run2 +4.23 / LAT-TTA +3.04）。

**诚实边界（必须写清，Hubness §7.6 教训）**：
1. phase-var 作为"per-query 失败预测变量"**不干净**：控住 #false 后仍 +0.19（过了 Hubness 那道致命关），**但再控 LR-severity（single-LR→HR drift）后塌到 +0.03**。即 phase-var 解释"哪些 query 失败"的力量**大半与该图 LR 失真程度共线**，不是独立的"格点不确定性"诊断信号。→ **诊断/相关性 claim 弱**。
2. **但 GO 不依赖相关性**：承重证据是 **interventional**（ensemble +4.23、LAT-TTA +3.04 是直接测量，非相关）。边缘化确实回收精度且明显超 TTA，这与 (C) 的相关性弱无关。
3. 操作点很低（16px Market query vs HR gallery，mAP 42→47）。+4.23 是真实相对改善（~10% rel），但绝对精度远未可用——符合 frozen 零训练探针，支持"去训练 lattice-consistency / phase-marginalized embedding"的下一步。

**AUTO VERDICT：GO**（h=16 过全部四关：volatility 强 ✓、lat_gain +4.23≥2 ✓、LAT-TTA +3.04>0 ✓、phase-var|#false +0.19>0 ✓）。区分 trivial 的两道生死对照（vs 普通 TTA、vs #false）都干净通过；唯一诚实降级是 phase-var 的**相关性**被 LR-severity 共线（写作时机制叙事走 interventional，不靠 (C) 相关）。

## 坑（已踩/已记）
- **OOM（run1 h=24 死）**：原版一次性 materialize 全部 2·K·Nq=60k 张 384×128 PIL（~18GB RAM）→ 31GB 机器在 h=24 被 OOM kill（无 Traceback，静默死，dmesg 无权限）。**修法**：按 query chunk（256）流式生成+提特征+释放 PIL，峰值 RSS 从 ~25GB 降到 8.5GB。streaming 版慢但内存安全，h=16 数值与 run1 一致（phase-var 0.1164 vs 0.1162）。
- **LR 喂法**：统一 384×128 HR 画布 degrade（gallery 同画布），pose 关（pose_dict=None，否则 LR 与缓存 heatmap 错位）。
- **variant0=确定性 bicubic LR**：lattice 与 TTA 共享 variant0，保证单图 baseline 一致、对照干净。
- **MaxSim 对照陷阱**：必须用 TTA-MaxSim（同 K 同 max）当基线，否则把"多枪"误当 lattice 增益（OVLI 教训：我们提的 MaxSim 不能输给 trivial 基线——这里 lattice-MaxSim 46.9 > TTA-MaxSim 43.8 > 单图 42.6，干净）。
- **慢**：单线程 PIL 变体生成 + O(Nq·K) Jaccard python 循环是瓶颈，每 height ~15-18min，全跑 ~70min（非 GPU 瓶颈）。
