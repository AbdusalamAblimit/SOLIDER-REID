# Ambiguous Query-Bag ReID — 零训练 kill-switch 设计

> 机会来源: `litreview2/explore20/clean/d_20.txt` 机会 1。
> 脚本: `experiments/cargo_cvpb/cvpb_querybag_killswitch.py`（frozen 特征 + numpy, 无 backward）。
> 复用 hubness 缓存特征: market `/tmp/hub_market_feats.npz`(exp260b), occluded_duke `/tmp/hub_oduke_feats.npz`(exp255)。

## 1. 重定义（隐藏变量 = query-bag purity）
标准 ReID 默认 query 是单张正确目标; multi-query 默认多张全是同一 ID。真实部署里检索员/跟踪器给的是**一包候选 crop**（跟踪漂移 / 错框 / 相邻行人 / 操作失误），纯度未知。大家以为"query support 越多越稳"，隐藏变量其实是 **bag purity**: 多给错图可能比单张干净图更坏。把 single-query ReID 重定义为 **weak-target-evidence bag retrieval**。

## 2. 机制（零训练）Target-Consensus Query Aggregation
每张 bag 图独立检索其 top-L gallery，构建 bag×bag 一致性图（边 = top-L Jaccard 重叠，即两张 bag 图对"gallery 里谁是目标"是否一致）。取最大/最密一致子集（consensus set），只融合该子集（trimmed mean）。直觉: 真目标会对同一 gallery 身份一致投票; 污染图指向别处, 落在 consensus component 外。
- **medoid 模式**（更鲁棒）: 取 summed-agreement 最高的 bag 图作种子（最被"认同"=最可能真目标）, 迭代加入对当前集合平均一致性 ≥thr 的图。
- **component 模式**（草案字面版）: 取 (A≥thr) 一致性图的最大连通分量。

## 3. Bag 构造
对每个 query 身份建 1 个 bag:
- 1 张 **anchor** 真目标 query 图 → 定 pid/cam（junk 规则 + GT 用 anchor 的）。
- (m−1) 张额外真目标图（同 pid 的其他 query 图; 不够则补同 pid 跨相机 gallery; 再不够重采 anchor）。
- c 张 **hard contaminant** = 从 anchor 的 baseline 余弦 **top-20 里采异身份 gallery 图**（已经排名很高 → 主动把融合特征往错身份拉, 最对抗的污染）。
- 污染率 c/(m+c) 扫 {0, 25, 50, 75%}, 固定真目标数 m=4。

## 4. 对照策略（都是 bag → 单一 gallery 排名）
| 策略 | 说明 |
|---|---|
| `avg` | bag 特征均值 renorm 余弦 |
| `single-best` | score(g)=max_i cos(bag_i,g)（min-distance multi-query, 对"远离一切的污染"天然鲁棒, 但会被"强匹配错身份"的 hard contaminant 钓走） |
| `median` | 逐维中位数 |
| `trimmed` | 去离 medoid 最远的 trim 比例图后均值 |
| **`consensus`★** | Target-Consensus（**sweep mode×thr×topL, 取最优=oracle 上界**; 若 oracle-tuned 都过不了门槛即判死） |
| `k-recip` | 在 bag-average 距离上做 k-reciprocal re-rank（强 test-time baseline） |
| `camera` | bag-average 上对同相机(anchor)gallery 降权（sweep gamma 取最优） |

## 5. GO / NO-GO 门槛（铁律: consensus 必须真超过 trivial baseline）
- **GO**（某 cell 须全清）: 25-50% 污染下 `avg` 较 0% 掉 >10 mAP; 且 `consensus` 追回 ≥一半跌幅; 且 `consensus` 比 `single-best`/`k-recip`/`camera` 都 ≥+2 mAP。
- **NO-GO**: `single-best` 或 `k-recip` 已经把污染解决（≥半跌幅 → 不需要新方法, 吸取 HUBNESS §7.6 trivial-baseline 教训）; 或污染根本不造成明显 protocol gap。
- consensus 是 oracle-tuned 上界, 报告须显式标注每 cell 的 best config（暴露是否靠 cherry-pick 调参）。

## 6. 已知坑
- **contaminant 采样**: top-20 hard 是最对抗设定（忠实 d_20 草案）; 它让 single-best 被钓走但 average 部分稀释存活, 这个不对称是真实现象。
- **bag size m=4**: 太小则污染只 0/1 张粒度太粗; 太大则同 pid query 图不够要补 gallery。m=4 在两集都有足够真图。
- **consensus 图阈值**: 0.05-0.5 sweep。hard contaminant 与真目标 top-L 也重叠（它们本就被模型混淆）→ 低阈值会把污染纳入 consensus, 需高阈值才能切开; 高污染(75%)下纯度坍塌, consensus 失效。
- **RNG 复现**: bag 构造的 RNG 状态依赖前面跑了几个污染率, 单独 `--contam_rates 0.5` 与全扫的 0.5 cell 数字会有小差异（结论不变）。
- **k-recip 公平性**: 这里 k-recip 跑在 bag-average 上（已被污染稀释）; 更强的 k-recip-on-single-best 未做, 是潜在更高的 baseline（对 GO 更不利）。
