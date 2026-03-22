# Paper 21: 2026-03-22 方向重置记录（support incomplete 之后还能争什么）

**记录日期**: 2026-03-22  
**目的**: 在 `exp141-147` 之后重新收紧主线，回答两个问题  
1. 既然 cross-image completion、retrieval scorer 小修补、attention bias 都没有形成主突破，下一步还能争什么？  
2. 哪些方向足够大，值得占用本地和远程两台机器并行推进？

---

## 本轮重新对齐时参考的工作

### 仓库内已有学习/代码记录
- `paper_15_common_support_training.md`
- `paper_16_ProFD.md`
- `paper_18_SSSC_TransReID.md`
- `paper_19_20260316_weekly_review.md`
- `paper_20_PADE.md`
- `exp109` oracle support bank 结果
- `exp141-145` 的最终 monitor / results / decisions

### 当前必须承认的已知事实
1. `exp109` 仍然是最强问题证据  
   - `oracle_feat_only_cvk = 66.15 / 77.87`
   - `oracle_feat_weight_cvk = 70.40 / 81.36`
   - 尤其低可见 query headroom 巨大  
   这说明真正的问题依然是 **single-image support incomplete**

2. `exp141` 说明 retrieval-side context scorer 不是没价值，但目前所有变体都更像 supporting mechanism  
   - `query_ctx`、`comp_ctx`、confidence gate、hard rank、sparse routing 都没形成主突破

3. `exp142` 说明 feature-level completion 不是“完全没接上”，而是**接上了也不成立**  
   - same-ID support teacher / prototype / completion gate 这一路在 15K 数据上已经多次失败

4. `exp143-145` 说明再给 Swin 注意力塞 skeleton bias 也不是答案  
   - attention inductive bias 在这套 backbone 上基本冗余

---

## 这轮复盘真正排除了什么

### 1. 不能再继续把“更强 teacher / 更强 scorer / 更强 bias”当主线

这些方向不是完全无效，但都已经出现明确天花板：
- cross-image support bank / prototype completion
- retrieval-time pair correction 小变体
- skeleton attention bias / attention mask / prompt-like 小模块

如果继续在这些线上扫微调参数，得到的最多是 recipe，不是论文主创新。

### 2. 不能把“普通 augmentation”直接包装成创新

PADE、ROA、OIA、SSSC 一类工作都已经说明：
- 遮挡增强本身是有效 recipe
- 但普通随机增强、真实遮挡贴图、简单一致性训练都不够新

如果再做一条“更好的 random erase/ROA 组合”，大概率只能是 engineering trick。

### 3. 不能再假设“同 ID 跨图 support 一定比单图更好学”

`exp109` 的 oracle 证明跨图完整 support **存在**
但 `exp110-142` 的系列失败说明：
- “存在” 不等于 “能被当前训练机制学到”
- 问题不只是 teacher 不够强
- 更像是 15K 数据无法稳定学会复杂的 cross-image completion 函数

---

## 这轮复盘后仍然开放的两个 gap

### Gap A: 单图内部能否合成“伪多 support”？

`exp109` 暴露的是 single-image support incomplete。  
如果 cross-image support 太难学，一个更合理的问题是：

**能否用同一张图自己构造出互补的 partial views，把单图训练成“伪多 support 学习”？**

这个方向与 FCFormer / PADE / SSSC 的区别在于：
- 不是随机遮挡
- 不是普通 dual-view consistency
- 而是 **pose-defined complementary support**

也就是：两张增强视图不是随便挡，而是由姿态热图决定“谁挡左臂、谁挡右腿、谁保留 torso”，让两张视图在 body support 上形成可解释的互补关系。

如果这条线成立，它回答的是：
- 单图 support incomplete 能否在 **训练范式** 层面被改写
- 而不是再做 feature completion module

### Gap B: 单图内部的双侧冗余是否被浪费了？

当前 keypoint/skeleton 分支默认把左臂和右臂、左腿和右腿当作两个独立 token。  
但在遮挡场景里，更合理的问题是：

**当左侧不可见时，右侧的 homologous evidence 是否应显式进入表示？**

这不是普通 feature completion，因为：
- 它不依赖 same-ID memory bank
- 也不是跨图 teacher
- 而是利用 **人体双侧同源结构的单图先验**

如果这条线成立，它回答的是：
- pose-guided ReID 是否应该从“17 个独立关键点”升级成“同源体部 + 非对称残差信号”的表示

---

## 当前最值得开的两条大方向

### 方向 1: Pose-Complementary View Training（PCVT）

**一句话**: 用姿态热图生成两张“互补可见”的伪视图，不再做随机遮挡，而是训练模型从互补 body support 中学习稳定身份表示。

为什么值得做：
1. 问题层面有新意  
   - 不再是假设单图只能提供单一 support
   - 而是把单图改写成可训练的“伪多 support”样本
2. 机制层面有新意  
   - pose-defined complementary masking
   - full / view-A / view-B 三视图监督
3. 与已有失败线真不同  
   - 不是 cross-image completion
   - 不是 retrieval scorer
   - 不是简单 consistency loss

### 方向 2: Symmetry-Conditioned Feature Aggregation（SCFA）

**一句话**: 用双侧同源体部聚合替代“左/右完全独立”的 keypoint 表示，让单图内部的 homologous evidence 在遮挡时可被显式利用。

为什么值得做：
1. 问题层面有新意  
   - 把遮挡下的 support incomplete 进一步刻画成“同源体部利用不足”
2. 机制层面有新意  
   - 同源聚合 + 非对称残差分解
   - 不依赖外部 teacher / memory / test-time graph
3. 与已有失败线真不同  
   - 不是 completion
   - 不是 attention bias
   - 不是 retrieval scorer

---

## 当前结论

这轮方向重置后的结论不是“support incomplete 错了”，而是：

1. `support incomplete` 仍是最强问题定义  
2. cross-image completion 这条兑现方式基本已经做尽  
3. 下一步应转向两类更大的新机制：
   - **单图伪多 support 训练范式**
   - **单图同源结构表示重构**

也就是：

> 不是再问“怎么把 teacher 塞进 backbone”，  
> 而是问“单张图本身还能被重新组织成什么更合理的训练对象和结构对象”。
