# Counterfactually Identifiable Anatomical Mediator Router：TAPF 的方法级重构方案

> 状态：**DESIGN-ONLY / RESEARCH-ONLY / NO-START**
>
> 日期：2026-07-18
>
> 约束：exp391已自然结束并封板；本方案不创建 config、不实现代码、不启动 GPU 实验。

## 零、独立反向审查后的收敛裁决

初始 SASR 版本把 unordered slots、Hungarian matching、CLIP semantic posterior、feature-dependent
router、multi-stage 和 relation KD 同时作为主方案。独立公开先例与机制审查表明，这样既容易被概括为
`Mask2Former/K-Net-style anatomical slots + conditional adapter`，又无法把增益归因给 anatomical
semantics。

因此当前主对象收敛为：

> **Counterfactually Identifiable Anatomical Mediator（CIAM）**：一个在 global ReID 前向路径中可被
> 冻结干预、满足 NULL identity、geometry-semantic congruence sensitivity 和真实 downstream
> mediation 的内部解剖状态。

首个核心版本不是 unordered slots，而是固定 coarse anatomical channels：pose teacher定义
`head/torso/arms/upper-legs/lower-legs` 的语义身份；冻结 CLIP 双编码 teacher不重复猜已知类别，只校准
该语义区域当前是否具有可信视觉支持、是否被背景/遮挡物污染。consumer则由 field-only PSG 改成
`gather local evidence -> semantic expert transform -> reliability-gated scatter` 的低秩 residual router。

所有 stage anchor 都直接从自己的 feature预测绝对状态；上一 stage不作为 offset base。每个 stage只放
一个主要 consumer，保持层级平衡。unordered slot/Hungarian、relation KD和显式 reject slot全部降为
核心版本成立后的可选扩展，不作为当前 novelty claim。

## 一、为什么“与近邻相似”不等于不能发表

RegionCLIP、π-VL、PAFormer、MUVA、ALADIN 之间本来就共享 CLIP、part、local alignment、
multi-level supervision 或 inference-free teacher 等构件。它们能成立，不是因为每个原子模块都从未
出现，而是因为各自建立了不同的中心对象：

- RegionCLIP 的对象是开放词汇 region representation；
- π-VL 的对象是 ReID feature map 的 pixel-level part-language consistency；
- PAFormer 的对象是 anatomically aware pose token 与 part-to-part retrieval；
- MUVA 的对象是 domain-generalized multi-grained visual-language mask injection；
- ALADIN 的对象是 attribute/local/global knowledge distillation。

论文 novelty 的实际判断更接近：

```text
一个清楚的新问题对象
+ 一个为该问题定制、不能被现有模块直接替换的机制
+ 一组证明该机制必要的证据
```

所以我们的目标不是找到完全没有近邻的“孤岛”，而是让审稿人无法把方法准确概括成
“RegionCLIP soft KD + PAFormer pose supervision + 普通 PSG”。

当前可以形成的一句话差分是：

> 现有工作让 feature 表达人体部位；我们把 anatomical state定义为 global ReID 路径中的可执行中介，
> 以受语义约束的 gather-transform-scatter 更新 backbone，并通过受控反事实证明正确 geometry-semantic
> binding相对容量匹配的 generic router确实改善检索。

## 二、为什么原 TAPF 很难获得更大增益

当前 PSG 的本质是：

```text
field --Conv(17→32)--> spatial delta
feature' = feature * (1 + bounded(delta))
```

它有三个性能上限：

1. gate 只看 field，不看当前区域的身份外观内容；
2. 自由 `17→32` 混合不要求区分具体关节语义；
3. 乘性 gate 只能放大/缩小现有通道，不能根据可见部位证据生成有条件的低秩 feature update。

CLIP 只给 anchor 增加 KD，并不会自动解除以上三个上限。要追求超过当前 `+0.2～+0.8 mAP` 的
效应，应替换 consumer 的数学对象，而不是继续给同一个 PSG 增加监督或复制更多 bank。

exp391进一步排除了“只因两层pose loss预算过大”的单一解释：把sum改为mean后，H2-M恢复到
`57.2/67.3/80.2/84.5`，early-bypass显示early route有`+0.141 mAP`独立贡献，但完整模型仍比D0低
`0.4 mAP`。因此纯结构Phase B/C封板；CIAM不是其续训，而是先重写consumer的数学对象和语义绑定，
随后才允许重新评估semantic multi-stage。

## 三、完整上界候选：Semantic Anatomical Slot Router（SASR）

本节保留完整 slot 版本作为上界设计空间，不授权把它作为首个实现。当前主实现顺序以第八节的
CIAM-core为准；只有 fixed-channel core已经证明 semantic mediation 和检索增益后，才允许评估本节的
unordered formulation。

### 3.1 可执行 anatomical state

第 `s` 个 stage 的 anchor 直接预测一组无序 slots：

```text
S_s = {M_s,k, q_s,k, r_s,k, z_s,k},  k=1...K
```

- `M_s,k`：slot 的空间 posterior/mask；
- `q_s,k`：slot 属于各 anatomical semantic prototype 的概率分布；
- `r_s,k`：slot 的可靠性；
- `z_s,k`：把 source mask resize到 downstream consumer后，从 consumer feature池化出的实例级局部证据；
  它不是 anchor额外预测的自由 identity code。

slot 顺序本身没有语义。真正的“head/torso/arms/legs”身份由 `q_s,k` 绑定，因此避免把固定 tensor
index 误当成已经学会的解剖语义。

训练期 teacher state 为：

```text
pose-derived region mask                -> M^T
frozen CLIP patch tokens pooled by M^T  -> v^T
v^T × frozen text prototypes            -> q^T
pose score + calibrated CLIP evidence    -> r^T
```

完整候选的语义粒度不是 COCO-17 文本类别，而是 coarse regions：

1. head/face；
2. torso/upper body；
3. arms/hands；
4. upper legs；
5. lower legs/feet；
6. background/occluder/unknown reject prototypes（核心首版移除，除非先得到可信空间 teacher）。

17-joint pose 仍用于几何 teacher，但先通过固定 incidence matrix 聚合成 region mask。固定映射已经
给出 region identity；CLIP distribution只能作为 visual-support/reject 校准，除非 teacher-only审计证明
它相对 pose one-hot具有额外 sample-specific 信息。不得把“用 CLIP重新识别已知 region标签”写成贡献。

空间状态必须拆成三个不可互相缩放吸收的量：

```text
M_s,k(p)  = bounded support mask，用于 scatter和空间监督
pi_s,k(p) = M_s,k(p) / (sum_p M_s,k(p) + eps)，只用于 gather pooling
r_s,k    = 独立、标定后的 visual-support/reliability amplitude
```

禁止直接用自由 `r*M` 而不区分 pooling posterior与调制 support；否则 `M` 放大、`r` 缩小会产生
同一输出，reliability解释不可辨识。

### 3.2 无序 slot matching

每个 stage 都从自己的 feature 直接预测 slots，不允许只有首 anchor 预测绝对状态、后续 anchor 只算
offset。训练时使用 Hungarian/optimal transport matching：

```text
cost(k,j) = lambda_sp * spatial_cost(M_s,k, M^T_j)
          + lambda_sem * KL(q^T_j || q_s,k)
          + lambda_rel * reliability_cost(r_s,k, r^T_j)
```

slot matching 的意义不是引入 DETR 形式，而是消除 label switching：模型可以改变 slot 枚举顺序，
但不能改变 slot 的空间证据与 semantic binding。

### 3.3 语义绑定 expert basis

使用 `q_s,k` 从 anatomical expert basis 中合成该 slot 的 feature-update operator：

```text
E_s,k = sum_c q_s,k,c * B_s,c
```

其中 `B_s,c` 是 semantic-class-specific 的低秩 expert basis。它不是文本 embedding；训练完成后由
student 参数独立存在，推理不需要 text prototype。

与固定“第 k 通道使用第 k 个卷积”相比，这个绑定有两个重要性质：

1. 同时置换 `(M,q,r,z)` 的 slot 顺序，router 输出应逐元素不变；
2. 只错配 `M` 与 `q`，空间证据会被路由到错误 semantic expert，router 输出必须改变。

第一项排除 tensor index 捷径，第二项定义真正要验证的 semantic mismatch sensitivity。

### 3.4 特征依赖的低秩 residual routing

每个 slot 不再只产生 field-only gate，而是根据 downstream consumer中的局部身份证据执行低秩更新。
必须显式区分 anchor source与router consumer：

```text
state_s      = Anchor_s(stopgrad(F_src_s))
M_cons_s,k   = Resize(M_s,k, spatial_size(F_cons_s))
pi_cons_s,k  = Normalize(M_cons_s,k);  if M_cons_s,k == 0, pi_cons_s,k = 0
h_s(p)       = V_s F_cons_s(p)
z_exec_s,k   = Wz_s Pool(F_cons_s, stopgrad(pi_cons_s,k))
context_s,k  = C_s z_exec_s,k
u_s,k(p)     = U(E_s,k) * sigma(h_s(p) + context_s,k)
DeltaF_s(p)  = sum_k r_s,k * M_cons_s,k(p) * u_s,k(p)
F'_cons_s(p) = F_cons_s(p) + alpha_s * tanh(DeltaF_s(p))
```

- `V_s/C_s` 跨 slots 共享；
- `Wz_s` 是推理保留的可执行 evidence projector，不是训练后删除的 auxiliary head；
- `U(E_s,k)` 由低秩 semantic expert basis合成；
- `alpha_s` 零初始化并有固定上界；
- `M_cons=0 => pi_cons=0`；显式 NULL state时严格identity；
- 不允许 bias、affine normalization 或静态常量产生 non-null update。

它比原 PSG 更可能带来实际增益，因为更新同时依赖：

- 人体区域在哪里；
- 该区域是哪一种人体语义；
- 该区域是否可靠；
- 当前图片在该区域实际包含什么身份外观证据。

### 3.5 balanced multi-stage routing

Swin-T 的 balanced multi-stage候选只允许三个 source/consumer interface，每个 state一个主要router：

```text
Stage-0 state -> Stage-1 内一个 router -> remaining Stage-1/downsample/后续网络
Stage-1 state -> Stage-2 内一个 router -> remaining Stage-2/downsample/后续网络
Stage-2 state -> Stage-3 内一个 router -> remaining Stage-3/GAP
```

每个 anchor 都从当前 stage feature 直接预测自己的 slots；上一层 state只用于 consistency，不作为
下一 anchor 的唯一输入或 offset base。禁止恢复 exp389 的 early 六次、late 两次累计调制不平衡。

跨层 consistency 分量化定义，不能对整个复合 state直接 Downsample：

```text
spatial     : Downsample(M_s) <-> M_s+1
semantic   : q_s <-> q_s+1 in matched semantic space
reliability: r_s <-> r_s+1
```

`z` 不做直接跨层一致性；不同通道数下最多经过受限共享投影后单独消融。unordered版本使用OT matching，
fixed-channel CIAM-core按已知semantic channel对齐，不能直接按slot tensor index做普通KL。

### 3.6 不蒸馏最终 descriptor 的身份关系 teacher

单纯 part-name distribution主要解决语义可辨识性，不一定提供足够身份增益。为了提高性能，又避免
把方法退化成普通 final-feature CLIP KD，增加一个只作用于内部 slot evidence 的 relational objective：

1. 对 frozen CLIP 局部 visual feature使用全训练集 running mean/diagonal variance或中心化+L2归一化；
2. 在同一 anatomical semantic class 内构造 teacher pairwise similarity matrix；
3. 让 router 必经且推理保留的 `z_exec_s,k=Wz_s(z_s,k)` 匹配 teacher relation；
4. 不对齐 final global descriptor，不输出 part descriptor，不做 language retrieval。

```text
L_relation = KL(softmax(Sim(v^T_region)/tau_T)
                || softmax(Sim(z_exec_student)/tau_S))
```

不使用 batch 内完整 covariance whitening：batch64下协方差秩不足且会随 identity sampler波动。相对
历史直接 cosine feature reconstruction，这个目标只保留相对实例关系，不强迫 SOLIDER feature
进入 CLIP 各向异性的绝对空间，更可能保留 ReID 判别性。`L_relation` 不允许接独立 projector；否则
它会退化成推理删除的 terminal auxiliary branch，不能被称为 router 的性能来源。

## 四、训练与推理边界

### 4.1 梯度所有权

- frozen CLIP image/text encoder：`eval + no_grad + requires_grad=False`；
- pose/CLIP teacher state：全部 detach；
- 每个 anchor 的输入明确为 `Anchor_s(stopgrad(F_s))`，防止 pose/semantic loss回流 backbone；
- `L_pose/L_semantic/L_slot_match/L_consistency`：更新 anchors/state heads；
- `z_exec=Wz_s(Pool(F_cons_s, stopgrad(pi_cons_s)))` 是 router 的必经输入；ReID loss可通过它更新
  backbone、`Wz_s` 和 router；
- `L_relation` 使用 `Wz_s(Pool(stopgrad(F_cons_s), stopgrad(pi_cons_s)))`，首版只更新同一个推理保留的 `Wz_s`，
  不更新 backbone或 anchor；是否放开 relation→backbone只能作为单独 ablation；
- ReID ID/triplet loss：更新 backbone、`Wz_s`、SASR router、BNNeck/classifier；
- `M/q/r` 进入 router前 detach，ReID loss不允许反向把 anchor变成任意 identity code；
- 每种 loss 分别 backward，逐组验证梯度所有权与相互梯度余弦。

### 4.2 推理

推理只保留：

```text
CIAM-core: RGB backbone + direct stage student M/r heads + fixed semantic binding
           + executable Wz_s/router + global GAP/BNNeck
optional SASR: 再保留 student q head和slot matching所需的执行状态
```

删除：

- 两个 CLIP encoder；
- text prototypes；
- external pose；
- teacher-side matching、whitening与纯 loss-only heads；
- 任何 part descriptor或language retrieval branch。

CIAM-core 的学生 `M/r` heads、固定semantic binding、`Wz_s` 与 router是RGB-only执行路径，推理必须
保留；optional SASR还保留student `q` head。它们不是auxiliary teacher head。逐模块删除审计应证明
删掉任一执行组件都会有限、非零地改变最终global descriptor。

最终检索仍只有单一 global descriptor。

## 五、为什么它比“CLIP 校准原 PSG”更像论文方法

### 5.1 新问题对象

不是“局部 feature 是否有语义”，而是“一个会执行视觉更新的 latent state 是否语义可辨识”。

### 5.2 新机制对象

不是把 CLIP loss接到现有 head，而是：

```text
counterfactually intervenable anatomical mediator
+ fixed geometry-semantic binding
+ feature-dependent low-rank gather-transform-scatter
+ calibrated visual-support gating
+ NULL identity / path-specific mediation
```

unordered slots、Hungarian matching和soft `q` 只属于optional SASR，不作为CIAM-core的新机制声明。

### 5.3 新证据对象

主消融不是只删 loss，而是验证：

- correct geometry-semantic binding vs 样本内/跨图错配；
- fixed pose-onehot vs CLIP-support calibration；
- anatomical ontology vs balanced CLIP visual clusters；
- matched wrong image state；
- static state；
- NULL state；
- semantic expert collapse；
- 每 stage router bypass；
- predicted RGB-only state、teacher-oracle state、wrong state的有序结果；
- optional SASR再验证paired `(M,q)`同步置换不变与只错配`M/q`的检索差异。

这组性质不是 π-VL、PAFormer、RegionCLIP 或普通 KD 的自然副产品。

## 六、为什么它更可能涨点

不能保证任何新结构一定涨点，但 CIAM 相对当前 D0 有四个真实增益来源：

1. **identity-aware update**：router读取局部实例证据 `z`，不再是 field-only static gate；
2. **visual-support gating**：CLIP与pose校准的 reliability抑制缺乏人体视觉支持的 update；
3. **多尺度互补**：每 stage直接预测、每 stage一次平衡路由，避免同一 field连续六次调制；
4. **semantic operator specialization**：固定pose semantics选择不同expert，并以expert-mean/random-ontology
   controls排除普通adapter解释。

显式 reject slot和relational teacher不是 CIAM-core 的预期增益来源；前者若后续加入，reject expert必须
固定为零，后者只能在 core已经成立后作为独立变量评估。

若目标是论文级效果，预注册目标不应再满足于 `+0.2 mAP`。探索性单 seed完整 e120 后，至少希望：

- CIAM-full 相对 current clean D0 `>= +1.0 mAP`；
- R1不下降；
- correct state相对 semantic mismatch/wrong state具有可辨别差异；
- 参数/FLOPs保持轻量，推理不运行 teacher。

该数值只能作为 final 后的描述性 GO 条件，不能用于中途早停或挑 best。

## 七、实现强度的三个版本

### A. Conservative：Semantic Expert PSG

- 保留固定 coarse body-part channels；
- 用 q-conditioned低秩 experts替换自由 `17→32` PSG；
- 单层先验证语义敏感性。

优点：实现风险最低。缺点：slot identifiability故事较弱，增益可能有限。

### B. Full candidate：SASR

- unordered slots；
- semantic binding；
- feature-dependent router；
- 三 stage balanced direct anchors；
- internal relational teacher。

它不是当前首个实现；只有 CIAM-core 相对 generic router和 pose-only control成立后才可评估。

### C. Aggressive：SASR + visible-to-uncertain evidence transfer

- 在 slot evidence之间加入 reliability-gated anatomy graph；
- 高可靠可见 slot为 uncertain slot提供 context；
- 仍只通过 router影响 global descriptor。

它可能进一步提高遮挡性能，但容易退化成普通 completion/GCN，应只在 B 版已证明 semantic routing
有效后考虑，不能和首版同时引入。

## 八、实验顺序与单变量归因

exp390/391全部封板后，固定顺序建议为：

1. clean D0 internal-field frozen audit；
2. coarse-region CLIP teacher-only audit；
3. `D0-PSG -> Router17`：保持同一17-channel field和single stage，只替换为feature-dependent
   low-rank residual consumer；
4. `Router17 -> Ordered-Coarse-CIAM`：固定pose incidence与one-hot semantic experts，不加CLIP；
5. `Ordered-Coarse-CIAM -> +CLIP support/reliability`：只增加双编码器visual-support校准；
6. `single-stage -> balanced multi-stage direct anchors`：三个anchor各自预测绝对状态，每层一个consumer；
7. 只有以上都成立，才分别评估 unordered slots和 executable relation teacher；
8. full 方法通过后再做multi-seed、Market与必要backbone验证。

主对照至少包括：

| arm | 要隔离的因素 |
|---|---|
| B0 | official RGB baseline |
| D0 | current clean anchor+PSG |
| Generic-LR-Adapter | 同参数/同深度，不读取 anatomical state |
| Router-static | 参数匹配但读取 canonical static state |
| Router17 | 原17-channel pose field + 新consumer数学 |
| Router-pose-onehot | fixed coarse pose semantics，无CLIP |
| Router-CLIP-support | 只加sample-specific visual support/reliability |
| Expert-mean | 所有semantic experts替换为均值，使语义绑定失效 |
| GSR-CLIPcluster | 同结构/参数/loss，但人体文本原型替换为train-only CLIP视觉balanced K-means原型 |
| SASR-single | 完整 single-stage semantic slot router |
| SASR-multi | balanced multi-stage direct anchors |

任何臂都必须 fresh、完整 e120、严格串行；本设计不授权现在创建这些 arm。

`GSR-CLIPcluster` 是最强generic语义对照：保留相同K、mask/reliability head、rank、expert、router、
stage、参数/FLOPs与训练目标，只移除 anatomical ontology。只有 anatomical版本优于它，且correct
binding相对样本内/跨图错配具有额外检索优势，才能把收益归给人体语义；若两者持平，收益只能归为
generic content routing或CLIP视觉监督。

## 九、硬 kill-switch

1. coarse-region CLIP teacher对 correct/random/shuffle mask不敏感，或不优于pose-one-hot：停止把CLIP
   作为核心校准；
2. q-conditioned expert发生 collapse，所有 expert近似相同：停止；
3. paired slot permutation不能在预注册 CPU/CUDA/AMP 容差内保持输出不变：实现错误；
4. mask/semantic mismatch不改变 router输出：semantic binding失败；
5. projector-only/static control复现 full：机制被吸收；
6. frozen correct-vs-wrong state仍 `<0.1 mAP`，但 router bypass很大：仍是容量模块；
7. SASR-full相对 D0没有清楚 final增益：不靠更多 stage、graph或loss小调参救场；
8. 只有 local auxiliary metric改善、global retrieval不升：不能进入主方法。
9. Generic-LR-Adapter、Expert-mean或GSR-CLIPcluster复现全部增益：不得把结果归因于 anatomical
   semantics；
10. 若后续reject slot的expert不是固定零，或没有可信空间teacher：不得称其为occluder suppression。

## 十、最终判断

如果只在原 PSG 上加 CLIP KD，方向容易成为现有方法拼装，也很难突破当前小效应。真正值得推进的
改造是把 TAPF 的基本对象从“17-channel field gate”重写成：

> **可冻结干预、语义绑定、可靠性有界、真实中介 global descriptor 的 anatomical state。**

pose one-hot首先定义 semantic identity；CLIP 的作用不是提供额外 part-name loss，而是校准视觉支持与
污染。性能主来源应是 feature-dependent low-rank routing；slot formulation和relation teacher都必须
作为后加变量单独证明。这个版本才同时回答：

1. 为什么 state具有解剖语义；
2. 为什么 consumer不能忽略语义；
3. 为什么它可能比原 PSG获得更大检索增益；
4. 为什么推理仍能删除 CLIP、text和external pose；
5. 为什么它不是普通 part feature或语言检索方法。

当前裁决：`CIAM-core = PRIMARY DESIGN CANDIDATE / DESIGN-ONLY / NO-START`；
`semantic balanced multi-stage = CONDITIONAL NEXT STAGE / NO-START`，只有single-stage语义门禁通过后
才能启动；`unordered SASR = OPTIONAL EXTENSION / NO-START`。exp391的NO-GO不永久否定这一新机制，
但禁止把它伪装成exp391 Phase B/C的恢复或续跑。
