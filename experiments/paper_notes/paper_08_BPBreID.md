# Paper 8: BPBreID - Body Part-Based Representation Learning for Occluded Person Re-Identification
**来源**: WACV 2023  
**论文**: https://arxiv.org/abs/2211.03679  
**代码**: https://github.com/VlSomers/bpbreid  
**阅读日期**: 2026-03-13

## 这篇工作真正解决什么
- BPBreID 的核心不是“学 attention map”本身，而是对 **partial observation 下单一 global embedding 的理论局限** 做了明确表述。
- 论文与 README 都强调：
  - 同一人全身图 A
  - 同一人上半身图 B
  - 同一人下半身图 C
  在全局单向量空间里要求 `A≈B≈C`，但 `B` 与 `C` 实际没有共同可见证据，这个设定本身就有悖论。
- 因此它的问题定义是：**遮挡下应比较 mutually visible parts，而不是强迫所有图进入同一个无条件全局 embedding 空间。**

## 代码里真正怎么做

### 1. 输出是 part embeddings，而不是单一 embedding
- 文件: `torchreid/models/bpbreid.py`
- 输出包含：
  - `global`
  - `foreground`
  - `background`
  - `concat_parts`
  - `parts`
- 结构化 part embedding 被保留到检索阶段

### 2. visibility 由 part attention map 激活得到
- 文件: `torchreid/models/bpbreid.py`
- 连续模式: 直接取 part probability map 的最大激活
- 二值模式: argmax 之后再转 one-hot 判断 part 是否存在
- 这不是“额外小技巧”，而是 part-based distance 的必要输入

### 3. 训练核心是 GiLt，不是简单多头监督
- 文件: `torchreid/losses/GiLt_loss.py`
- 默认策略：
  - global / foreground / concat_parts 用 ID
  - `parts` 用 triplet
- 若开启 visibility，则 part ID / part triplet 可以只在可见 part 上生效

### 4. 检索核心是共同可见部位距离
- 文件: `torchreid/metrics/distance.py`
- `compute_distance_matrix_using_bp_features()` 按两类情况处理：
  - 二值可见性: 只在 query / gallery 共同可见 part 上求 masked mean
  - 连续可见性: 用 `sqrt(v_q * v_g)` 作为 pair-specific 权重
- 结论: **BPBreID 的主贡献落在距离定义，而不是单个 train-time 模块。**

## 对当前代码线的直接启发
1. 我们当前 `equal_concat` 的问题不是“权重还不够可学习”，而是 **branch 的局部结构信息在距离计算前就被压扁了**。
2. `exp030a` 已经告诉我们 branch 的价值主要在 fusion；BPBreID 则进一步说明：面对遮挡，fusion 最终应该落在 **pairwise common visible support** 上。
3. BPBreID 依赖 parsing pseudo labels；我们当前有 keypoint / skeleton branch，理论上可以走 **keypoint-level mutually-visible matching**，不必复刻 parsing 路线。

## 我们不能再把什么当创新
- “part visibility score”
- “quality-aware part weighting”
- “局部特征加权求距离”

这些点在 BPBreID 中都已经是主线组件，而不是新概念。

## 我们还能争什么
1. 不依赖 parsing mask，而是依赖更轻量的 keypoint/skeleton 表示
2. 把 `exp033 / exp034` 的 target-aware pose 数据与 retrieval-time matching 连起来
3. 在 `PSG + GCN` 这条已经验证过的代码线上做 **共同可见关键点检索**

## 当前判断
- BPBreID 把问题讲得很清楚：**partial observation 的本质是“比较谁和谁的哪一部分”，不是“再调一个融合系数”。**
- 这使得 `AFF` 更像工程候选，而不是足够强的主创新线。
