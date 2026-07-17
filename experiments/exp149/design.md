# 实验 exp149: SCFA（Symmetry-Conditioned Feature Aggregation）

## 动机

`exp142` 之后，cross-image support completion 基本已经做尽。  
但这不代表单图没有额外可用结构信息。

当前 keypoint branch 有一个很强但尚未显式利用的先验：

**人体双侧同源结构。**

例如：
- 左臂被挡住时，右臂通常仍携带部分可迁移的服饰/纹理/形状证据
- 左腿缺失时，右腿至少还能提供“腿部类别与材质”这一层信息

当前 GCN/head 默认把左右侧当作两个独立 token，再做简单加权平均。  
这会浪费单图内部的同源冗余。

## 核心假设

1. 单图 support incomplete 不只体现在“缺少更多图”，也体现在“没有利用好同一张图里的 homologous evidence”
2. 若把左右同源关节先显式聚合成更稳的 body evidence，再把非对称差异作为残差信号保留，遮挡场景下的 skeleton branch 应更鲁棒
3. 这种做法与 prototype completion、本地/远程 scorer、attention bias 都不同，属于**表示层重构**

## 技术方案

### 1. 左右同源分解

定义对称 pairs：
- eyes `(1,2)`
- ears `(3,4)`
- shoulders `(5,6)`
- elbows `(7,8)`
- wrists `(9,10)`
- hips `(11,12)`
- knees `(13,14)`
- ankles `(15,16)`

midline:
- nose `(0)`

对每个 pair `(l, r)`，从 `kp_feats` 和 `kp_scores` 构造两种信号：

#### (a) 同源聚合 token
`h_pair = normalize(w_l * f_l + w_r * f_r)`

目的：
- 当一侧缺失时，另一侧还能独立提供该体部的稳定证据

#### (b) 非对称残差 token
`a_pair = min(w_l, w_r) * (f_l - f_r)`

目的：
- 只在两侧都可信时，保留左/右差异
- 避免把真正的非对称身份线索完全抹掉

### 2. 新的 skeleton aggregation

第一版不改 global branch，只改 skeleton branch 的表示聚合：

1. 用 `h_pair` 替换原本“左右完全独立”的一部分 token
2. 用 `a_pair` 作为附加残差信号
3. 对 `[nose, all h_pair, all a_pair]` 做新的 confidence-weighted aggregation
4. 输出新的 `skeleton_feat`

### 3. 训练方式

保持 `exp030a` 主干不变：
- global branch 不动
- 损失仍为现有 `ID + Triplet`

只改 `SkeletonGCNHead` 的 token 组织与 pooling 方式。

## 对照组

- 主基线: `exp030a-eq`
- 历史负面对照:
  - `exp142 SKC`: direct completion 失败
  - `exp143-145 SASA`: attention bias 中性

这条线的不同点在于：
- 不直接补 token 值
- 不改 attention bias
- 而是重写 body-structure representation

## 预期结果

若方向成立，应看到：

1. skeleton branch 的 late-stage 稳定性提升
2. 在单侧遮挡 case 上增益更明显
3. 相对 `exp030a` 至少出现可解释的弱正向
4. 若成立，它有机会成为“pose-defined bilateral redundancy”这条新 story 的核心机制

## 关键日志

训练期必须记录：

- `scfa_cov`: 参与同源聚合的 pair 比例
- `scfa_hm`: 同源聚合 token 的平均权重
- `scfa_hs`: 同源聚合 token 的权重方差
- `scfa_am`: 非对称残差 token 的平均权重
- `scfa_as`: 非对称残差 token 的权重方差
- `scfa_hn`: 同源 token 平均范数
- `scfa_an`: 非对称 token 平均范数
- `scfa_pg`: “单侧低置信 + 对侧高置信” pair 比例
- `scfa_eq`: “双侧都高置信” pair 比例

这些日志用于判断：
- 同源聚合是否真的只在遮挡时发挥作用
- 非对称残差是否在后期塌掉

## 风险与失败解释

1. 若 `scfa_hm` 很高但结果不涨  
   说明同源聚合抹掉了对身份有用的左右差异

2. 若 `scfa_am` 接近 0  
   说明非对称残差没真正进入表示

3. 若 `scfa_cov` 很低  
   说明当前数据里的可利用双侧同源信号不足

4. 若最终近似中性  
   说明“单图同源冗余”虽然概念合理，但不足以成为当前 benchmark 上的主突破口
