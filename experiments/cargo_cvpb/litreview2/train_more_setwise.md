Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f00b7-49b6-7a02-9f56-462fe7f5f58b
--------
user
ReID团队 LM-ReID(低分辨率person ReID 重定义为采样格点sampling-lattice隐变量): test-time decision-marginalization已成立(同图K=9 lattice变体[sub-pixel phase/±1 LR-pixel bbox检测框/antialias kernel]边缘化, Market h12-32全beat普通TTA优势随分辨率递减, bbox检测框crop不确定性主导+2.84)。用户要更强训练端互补机制(train+test结合6/10→7-8/10), **明确不许止步test-time-only**。**训练端已死/将死(别重复)**: ①consistency拉变体特征到均值=collapse压判别性-1.73还掉HR; ②marginal-likelihood L_marg=训练有害大元凶; ③lattice-aug训练≈普通random增广+0.54不lattice-specific; ④LPA query-side加权头FAIL(最佳变体query+gallery共定单看query预测不出来acc≈chance 12%); ⑤Hard-Lattice ERM大概率负。**铁律: 训练端invariance/feature-collapse死; query-side预测最佳变体死**。LIVE备选=LC-STN canonicalization(估计+重采样对齐LR crop监督几何非身份)。求新训练端机制和test-time marginalization互补, 避开上述死法, 廉价kill-switch(优先冻backbone probe)。给具体机制+为何不重蹈死法+kill-switch+联网查novelty+信心1-10, 务实中文。 ★lens=**训练目标对齐test-time(set-wise/amortize)**: 训练时直接用K-variant set与gallery做logsumexp/set相似度优化(不压特征不求均值), 把边缘化amortize进网络。和失败的L_marg(求softmax均值)区别在哪?
codex
我先按你给的“set-wise / amortize test-time marginalization”视角查一下近年 LR-ReID、TTA/marginalization、multi-instance/set loss 相关工作，重点确认有没有已经把 sampling lattice 当训练端 latent set 来优化的近似方案。
web search: 
web search: low resolution person re-identification sampling lattice marginalization test-time augmentation paper
web search: 
web search: test-time augmentation marginalization training logsumexp loss image retrieval
web search: 
web search: arxiv set based metric learning log-sum-exp retrieval loss multiple augmentations
web search: 
web search: "LM-ReID" low resolution person re-identification
web search: 
web search: 2024 low resolution person re-identification paper cross-resolution reid
web search: 
web search: "test-time augmentation" "image retrieval" "person re-identification"
web search: 
web search: "logsumexp" "person re-identification" loss
web search: 
web search: "set-to-set" "person re-identification"
web search: 
web search: https://arxiv.org/abs/2105.05639
web search: 
web search: 'policy' in https://arxiv.org/abs/2002.01642
codex
初步看下来，公开文献里有“拉近训练/推理 TTA gap”的 ReID 工作，但典型做法是 flip 特征取均值并加 MSE 一致性；这正好落在你们已经判死的 invariance/feature-collapse 类型。公开视频/集合 ReID也有 set-to-set，但不是把 LR sampling lattice 当隐变量并按检索决策边缘化来训练。
web search: 
web search: 'set to set' in https://arxiv.org/abs/1708.05512
codex
**结论**
最值得开的是 **LS-MRT: Lattice-Set Marginal Retrieval Training**。它不是再做 invariance，也不是预测 query 最佳 lattice，而是把测试时已经成立的 decision marginalization 直接写进训练检索损失里。

核心信心：**7/10**。  
原因：机制和已成立 test-time gain 对齐，且能用 frozen-backbone probe 很便宜地杀掉；但 logsumexp 温度不好会退化成 hard-lattice 或 soft mean，需要严控。

**具体机制**
给每个 LR crop 生成固定 K=9 lattice 变体：

```text
Q_i = {T_k(x_i)}_{k=1..K}
z_i,k = normalize(P(f(T_k(x_i))))
```

先做最小版：gallery 单图 embedding，query 用 K-set：

```text
S(Q_i, g_j) =
tau_l * logmeanexp_k( sim(z_i,k, z_j) / tau_l )
```

然后直接做 batch retrieval / supervised contrastive：

```text
L_set =
- log  sum_{j: y_j=y_i, j!=i} exp(S(Q_i,g_j)/tau_c)
      /sum_{m: m!=i}        exp(S(Q_i,g_m)/tau_c)
```

如果 gallery 也存在 LR/crop uncertainty，再扩展成：

```text
S(Q_i, G_j) =
tau_l * logmeanexp_{k,l}( sim(z_i,k, z_j,l) / tau_l )
```

训练组合建议：

```text
L = L_base(single/random view ID+triplet) + lambda * L_set
```

先 `lambda=0.2/0.5`，不要上来全量替换 baseline loss。

**和失败的 L_marg 的关键区别**
失败的 `L_marg` 本质大概率是：

```text
- log mean_k softmax(W z_k)[id]
```

它 marginalize 的是 **train-ID classifier posterior**。问题是：测试时 classifier 丢掉，优化目标和 retrieval ranking 不一致；`W` 是训练身份原型，不是 gallery；softmax 均值容易把低质变体也往同一身份中心拉，最后接近 invariance/collapse。

LS-MRT 是：

```text
softmax_over_gallery( logsumexp_over_lattice(sim(query_variant, gallery)) )
```

差别很实在：

1. marginalize 的对象是 **query-gallery 相似度证据**，不是 ID 分类概率。
2. denominator 里有真实负 gallery，负样本的 lattice 假匹配也会被惩罚。
3. 不要求 `z_k` 接近均值，也不要求每个变体单独可分类。
4. variant 权重由 **query-gallery pair** 决定，不是 query-side 预测最佳变体。
5. 训练分数形式和 test-time decision marginalization 是同一个接口。

一句话：`L_marg` 在训练分类头上求均值；LS-MRT 在检索决策层上做边缘化。

**为什么不重蹈死法**
- 不做 consistency：没有 `MSE(z_k, mean(z))`，不压 variant feature 到均值。
- 不做 query-side best lattice：最佳证据由 `(query variant, gallery)` 配对相似度产生。
- 不等于 lattice-aug：lattice-aug 每次只看一个随机变体；LS-MRT 同时看 K 个变体，并让 K-set 参与同一个 ranking decision。
- 不等于 hard-lattice ERM：`tau_l` 不能太低。太低就是 max/hard variant，必须用温度网格和 entropy 监控防退化。
- 不是 test-time-only：训练时 embedding/head 被检索式 marginal objective 反向更新，测试时继续用同一 set score。

**廉价 Kill-Switch**
第一阶段只做 frozen probe：

1. 冻结现有 backbone。
2. 缓存 train/query/gallery 的 K=9 features。
3. 只训练一个很小的 `P`：linear/BNNeck/diagonal scale，最多加温度 `tau_l`。
4. 用 `L_set` 训练 1-3 epoch 或直接 cached feature 训练。
5. 评估 h12/h16/h24/h32 的原 decision marginalization score。

杀掉条件：

- frozen probe 相比当前 test-time marginalization **没有 +0.2~0.3 mAP**；
- h12 涨但 h24/h32 明显掉；
- K 变体平均 cosine 明显上升，说明又在变相 collapse；
- positive lattice responsibility 总是集中到同一个固定 lattice，说明不是 pair-wise complementary；
- train retrieval loss 降，但 val mAP 不动，说明只是 batch scorer 过拟合。

通过条件：

- h12/h16 至少稳定 +0.3 mAP；
- h24/h32 不伤；
- positive pair 的 lattice posterior 有 pair-dependent 分布；
- negative logsumexp 分数下降，而不是只抬 positive top-1 variant。

**联网 Novelty 判断**
我查到的相近工作都没有精确覆盖这个点：

- FlipReID 明确讨论 ReID 中 TTA 训练/推理 gap，但做的是 flip 双图 mean feature 和 MSE flipping loss，属于你们已判死的 feature consistency 路线。来源：arXiv FlipReID 摘要和方法描述显示其用 mean feature 与 MSE flipping loss。([arxiv.org](https://arxiv.org/abs/2105.05639))
- Learning TTA for Image Retrieval 学的是 test-time augmentation policy，主要是 off-the-shelf feature 的测试聚合策略，不是训练 LR-ReID backbone 的 sampling-lattice latent set loss。([arxiv.org](https://arxiv.org/abs/2002.01642))
- Better Aggregation in TTA 说明简单平均 TTA 可次优，并学习 aggregation，但场景是分类 TTA，不是 ReID gallery ranking。([arxiv.org](https://arxiv.org/abs/2011.11156))
- Set-to-set ReID 早有，但处理的是视频/跨摄像头样本集合，目标还包含 intra-class compactness；不是单张 LR 图像的 sampling lattice 隐变量边缘化。([arxiv.org](https://arxiv.org/abs/1708.05512))
- 近年 CR-ReID 仍多围绕 SR、resolution-invariant 或 feature discrepancy alignment，例如 2025/2026 VPFA 把 resolution gap 看作 feature direction。([arxiv.org](https://arxiv.org/abs/2510.00936))

所以新意不是“logsumexp/MIL 本身新”，而是：**把 LR crop sampling lattice 定义为检索决策 latent variable，并用与测试边缘化同构的 set-wise retrieval objective 训练**。这个 novelty 我给 **7/10**，足够作为下一条主线实验，但论文表达必须强调 problem reformulation + train/test objective alignment。
tokens used
54,753
**结论**
最值得开的是 **LS-MRT: Lattice-Set Marginal Retrieval Training**。它不是再做 invariance，也不是预测 query 最佳 lattice，而是把测试时已经成立的 decision marginalization 直接写进训练检索损失里。

核心信心：**7/10**。  
原因：机制和已成立 test-time gain 对齐，且能用 frozen-backbone probe 很便宜地杀掉；但 logsumexp 温度不好会退化成 hard-lattice 或 soft mean，需要严控。

**具体机制**
给每个 LR crop 生成固定 K=9 lattice 变体：

```text
Q_i = {T_k(x_i)}_{k=1..K}
z_i,k = normalize(P(f(T_k(x_i))))
```

先做最小版：gallery 单图 embedding，query 用 K-set：

```text
S(Q_i, g_j) =
tau_l * logmeanexp_k( sim(z_i,k, z_j) / tau_l )
```

然后直接做 batch retrieval / supervised contrastive：

```text
L_set =
- log  sum_{j: y_j=y_i, j!=i} exp(S(Q_i,g_j)/tau_c)
      /sum_{m: m!=i}        exp(S(Q_i,g_m)/tau_c)
```

如果 gallery 也存在 LR/crop uncertainty，再扩展成：

```text
S(Q_i, G_j) =
tau_l * logmeanexp_{k,l}( sim(z_i,k, z_j,l) / tau_l )
```

训练组合建议：

```text
L = L_base(single/random view ID+triplet) + lambda * L_set
```

先 `lambda=0.2/0.5`，不要上来全量替换 baseline loss。

**和失败的 L_marg 的关键区别**
失败的 `L_marg` 本质大概率是：

```text
- log mean_k softmax(W z_k)[id]
```

它 marginalize 的是 **train-ID classifier posterior**。问题是：测试时 classifier 丢掉，优化目标和 retrieval ranking 不一致；`W` 是训练身份原型，不是 gallery；softmax 均值容易把低质变体也往同一身份中心拉，最后接近 invariance/collapse。

LS-MRT 是：

```text
softmax_over_gallery( logsumexp_over_lattice(sim(query_variant, gallery)) )
```

差别很实在：

1. marginalize 的对象是 **query-gallery 相似度证据**，不是 ID 分类概率。
2. denominator 里有真实负 gallery，负样本的 lattice 假匹配也会被惩罚。
3. 不要求 `z_k` 接近均值，也不要求每个变体单独可分类。
4. variant 权重由 **query-gallery pair** 决定，不是 query-side 预测最佳变体。
5. 训练分数形式和 test-time decision marginalization 是同一个接口。

一句话：`L_marg` 在训练分类头上求均值；LS-MRT 在检索决策层上做边缘化。

**为什么不重蹈死法**
- 不做 consistency：没有 `MSE(z_k, mean(z))`，不压 variant feature 到均值。
- 不做 query-side best lattice：最佳证据由 `(query variant, gallery)` 配对相似度产生。
- 不等于 lattice-aug：lattice-aug 每次只看一个随机变体；LS-MRT 同时看 K 个变体，并让 K-set 参与同一个 ranking decision。
- 不等于 hard-lattice ERM：`tau_l` 不能太低。太低就是 max/hard variant，必须用温度网格和 entropy 监控防退化。
- 不是 test-time-only：训练时 embedding/head 被检索式 marginal objective 反向更新，测试时继续用同一 set score。

**廉价 Kill-Switch**
第一阶段只做 frozen probe：

1. 冻结现有 backbone。
2. 缓存 train/query/gallery 的 K=9 features。
3. 只训练一个很小的 `P`：linear/BNNeck/diagonal scale，最多加温度 `tau_l`。
4. 用 `L_set` 训练 1-3 epoch 或直接 cached feature 训练。
5. 评估 h12/h16/h24/h32 的原 decision marginalization score。

杀掉条件：

- frozen probe 相比当前 test-time marginalization **没有 +0.2~0.3 mAP**；
- h12 涨但 h24/h32 明显掉；
- K 变体平均 cosine 明显上升，说明又在变相 collapse；
- positive lattice responsibility 总是集中到同一个固定 lattice，说明不是 pair-wise complementary；
- train retrieval loss 降，但 val mAP 不动，说明只是 batch scorer 过拟合。

通过条件：

- h12/h16 至少稳定 +0.3 mAP；
- h24/h32 不伤；
- positive pair 的 lattice posterior 有 pair-dependent 分布；
- negative logsumexp 分数下降，而不是只抬 positive top-1 variant。

**联网 Novelty 判断**
我查到的相近工作都没有精确覆盖这个点：

- FlipReID 明确讨论 ReID 中 TTA 训练/推理 gap，但做的是 flip 双图 mean feature 和 MSE flipping loss，属于你们已判死的 feature consistency 路线。来源：arXiv FlipReID 摘要和方法描述显示其用 mean feature 与 MSE flipping loss。([arxiv.org](https://arxiv.org/abs/2105.05639))
- Learning TTA for Image Retrieval 学的是 test-time augmentation policy，主要是 off-the-shelf feature 的测试聚合策略，不是训练 LR-ReID backbone 的 sampling-lattice latent set loss。([arxiv.org](https://arxiv.org/abs/2002.01642))
- Better Aggregation in TTA 说明简单平均 TTA 可次优，并学习 aggregation，但场景是分类 TTA，不是 ReID gallery ranking。([arxiv.org](https://arxiv.org/abs/2011.11156))
- Set-to-set ReID 早有，但处理的是视频/跨摄像头样本集合，目标还包含 intra-class compactness；不是单张 LR 图像的 sampling lattice 隐变量边缘化。([arxiv.org](https://arxiv.org/abs/1708.05512))
- 近年 CR-ReID 仍多围绕 SR、resolution-invariant 或 feature discrepancy alignment，例如 2025/2026 VPFA 把 resolution gap 看作 feature direction。([arxiv.org](https://arxiv.org/abs/2510.00936))

所以新意不是“logsumexp/MIL 本身新”，而是：**把 LR crop sampling lattice 定义为检索决策 latent variable，并用与测试边缘化同构的 set-wise retrieval objective 训练**。这个 novelty 我给 **7/10**，足够作为下一条主线实验，但论文表达必须强调 problem reformulation + train/test objective alignment。
