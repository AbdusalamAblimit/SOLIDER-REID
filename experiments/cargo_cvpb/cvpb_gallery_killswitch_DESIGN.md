# Gallery-Composition Kill-Switch — 设计 (cvpb_gallery_killswitch.py)

## 动机
三个独立 codex（终身 d3 / 开集 d9 / 长尾 d10）收敛到同一 re-framing：
**ReID 失败由 GALLERY 组成（规模/膨胀/分布形状）驱动，而非只看 query/模型。**
同一 frozen embedding，换不同 gallery → 不同失败。

## ★最高优先级教训（HUBNESS §7.6）
上一个诊断（Hubness M(q) 负向 in-degree）被**漏控 `#false-in-topk`** 证伪：
控住"top-k 里错几个"这个 trivial 计数后, M(q) 偏相关塌到 ≈0。
**本次铁律: 每个 per-query 相关必须控 #false-in-topk + k-reciprocal + camera + gallery-size。**
每个测试都内置一个 trivial-proxy 对照, 信号只有打赢 trivial 才算。

## 三个测试（全 frozen 零训练, numpy）

### 测试 A — Gallery-Growth Tax (d3)
- 固定 CORE 任务（query-ID 的一半作 core, 另一半作 distractor 池, 同域无域 shift）。
- 逐步把 held-out 同域 ID 的 gallery 图当纯 distractor 注入, gallery 1x→3x/5x/10x。
- 测 frozen 模型下 core query 的 ΔmAP/ΔR1 衰减曲线。
- **trivial 对照1**: per-query AP-drop vs #false-in-topk 增量的 Spearman（高→只是机械计数）;
  且看"#false-in-topk 没变"的 query 是否仍掉 AP（结构性 tax 的证据）。
- **trivial 对照2**: 用 row-shuffled 特征当 distractor（同 count, 破坏几何）→ 若 real≈shuffled, 纯 count。

### 测试 B — Gallery-Size Rejection (d9)
- watchlist size {10,50,100,250,500,full}, query = genuine(enrolled) + 同域 held-out impostor。
- 测 impostor max-cosine 是否随 size 系统上升（→ FPIR 漂移）。
- 比 GLOBAL 阈值 vs SIZE-CONDITIONED（按 size 分桶校准 impostor tail）: DIR@FPIR 1%/5%, FPIR@TPIR90%。
- **trivial 对照（max-of-N 陷阱）**: random-feature gallery（同 count）给纯"max over N 增长"基线;
  size-conditioning 只有比 random 上多救才有意义。

### 测试 C — Singleton Merge (d10)
- Zipf gallery: head ID 多图, tail ID singleton。tail query（held-out unknown tail）→
  是否错并入 head prototype, false-merge rate 是否随 head support 单调上升。
- 比 GLOBAL 阈值 vs SUPPORT-CALIBRATED（按 support 分层校准）在同 head-recall 下的 tail false-merge。
- **trivial 对照（"head 图多→NN 彩票多"）**: 报 per-ID rate（trivially 升）AND per-IMAGE rate;
  per-image rate 若 FLAT 则纯机械, 若仍随 support 升才是非平凡 over-attraction。

## 数据 / 复用
- frozen ckpt: market exp260b / occluded_duke exp255。
- 复用 hubness 缓存特征 `/tmp/hub_market_feats.npz` / `/tmp/hub_oduke_feats.npz`（含 q/g feat+pid+cam+name）。
- 复用 cvpb_hubness 的 extract/eval 约定（POSE_TEST_FEAT=global, NECK after, FEAT_NORM yes）。

## 预期 / 判定
- verdict: 三个里哪个有干净、非 trivial-proxy、非红海覆盖的信号 → 值得做方法。
- 被 #false-in-topk / max-of-N / per-image 吃掉的 → 诚实判死, 不重蹈 Hubness 覆辙。

## 对照变量隔离
- 单变量: 每个测试只改 gallery 组成（size/growth/shape）, 模型与 query 表示完全 frozen。
- A: 唯一变量=gallery size（注入 distractor）; B: 唯一变量=watchlist size; C: 唯一变量=head support。
