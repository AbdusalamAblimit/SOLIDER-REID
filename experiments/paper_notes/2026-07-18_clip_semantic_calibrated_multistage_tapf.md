# CLIP 语义校准的多阶段 TAPF：文献、代码与机制审查

> 状态：**RESEARCH-ONLY / NO-START**
>
> 日期：2026-07-18
>
> 边界：本文只做公开文献、公开代码和当前 TAPF 机制的只读审查；不修改正式训练代码/config，
> 不创建实验 output，不启动训练，不占用 4090。任何未公开内部方法均不作为公开
> prior art、related work 或 novelty 冲突来源。

## 一、审查结论

这个方向值得继续研究，但当前文字方案还不能直接进入实现或正式实验。

最重要的判断不是“CLIP 能否提供人体部位语义”，而是：

> CLIP 校准得到的语义状态，是否被下游 consumer 以不可绕开的方式执行，并真实改变最终唯一
> global descriptor。

现有 TAPF 的主要缺口不在 anchor 是否能预测 pose-like field。clean anchor 已经逐通道接受
COCO-17 Gaussian 和 reliability 监督；旧 exp378 的 anchor 也有较好的 teacher agreement、flip
equivariance 和通道占用。缺口在于 PSG 可以把 17 通道当成任意条件扰动或静态重标定输入，而不必
使用“第 j 个通道是哪一个人体关节”的语义。

因此，若只增加：

```text
anchor field -> pool stage feature -> projector -> CLIP KD
```

但 PSG 仍只读取原来的自由 `Conv(17→hidden)` field，则 KD 可能被 projector 或 anchor 的诊断头
吸收，最终 consumer 仍然可以对 joint permutation、wrong field 和 spatial constant 不敏感。这样即使
最终 mAP 上涨，也只能解释为 generic local CLIP KD，不能解释为“语义可辨识的 TAPF”。

当前最合理的研究对象应收缩为：

> **CLIP 只在训练期校准多阶段、可执行的 latent anatomical control state；每个 state 都必须通过
> 受语义约束的 PSG/调制路径汇入同一个 global descriptor，并用内部 field 的因果干预证明通道语义
> 真正进入下游路径。**

公开查新没有发现上述完整链条的逐项同构实现，但几乎所有构件都已有强近邻。当前 B 类创新风险约
为 `8/10`；新意不能落在 CLIP、body-part prompt、局部 KD、多尺度对齐或 pose-free inference 本身，
只能落在“可执行 latent control state + 语义因果可辨识性”这个问题与证据对象上。

## 二、当前 TAPF 到底有用吗

### 2.1 检索层面的有限正证据

官方 clean 结果目前支持“完整 anchor+PSG 可能有小幅检索价值”：

| 数据/seed | B0 | D0 | D0−B0（mAP/R1/R5/R10） |
|---|---:|---:|---:|
| Occluded-Duke / 1234 | `57.4/67.4/80.6/85.2` | `57.6/67.7/80.8/84.6` | `+0.2/+0.3/+0.2/−0.6` |
| Occluded-Duke / 4321 | `56.0/66.2/79.4/83.8` | `56.8/66.5/79.9/84.3` | `+0.8/+0.3/+0.5/+0.5` |
| Occluded-Duke / 2025 | `57.5/67.9/81.1/85.7` | `57.9/67.0/80.4/85.2` | `+0.4/−0.9/−0.7/−0.5` |
| Market-1501 / 1234 | `91.6/96.3/98.7/99.2` | `92.0/96.5/98.8/99.3` | `+0.4/+0.2/+0.1/+0.1` |

Occluded-Duke 三 seed paired mean±sample std为
`+0.47±0.31/−0.10±0.69/+0.00±0.62/−0.20±0.61`，mAP方向`3/3`为正，rank方向混合。因此当前
只能写“小而可重复的mAP正差”，不能写稳定四项提升、统计显著或跨架构普适。

### 2.2 语义层面的证据仍未成立

clean exp387/390 的 `correct/shuffle/None/exploding exact` 干预的是 external pose 输入。eval 路径本来
就不读取 external pose，因此这些门禁严格证明的是：

1. 推理为 RGB-only；
2. query/gallery 不依赖 pose artifact；
3. PSG consumer 不是 terminal dead path。

它们没有替换内部 `student_field/consumer_field`，所以不能证明 PSG 使用了正确的 17 通道语义。

旧 exp378 对内部 field 的冻结干预给出了更直接的反证：

| 干预 | 相对 correct 的 mAP 变化 |
|---|---:|
| matched wrong field | `−0.0155` |
| joint-channel permutation | `+0.0024` |
| confidence permutation | `+0.0002` |
| spatial constant | `−0.0238` |
| zero field | `−0.0536` |
| bypass 全部 PSG | `−2.6829` |

旧 PSG 明确有用，但几乎不依赖正确实例、通道名称或空间结构，最符合静态/容量性重标定解释。
clean PSG 已改为无 bias、GN 无 affine、zero field 严格 identity，因此 clean checkpoint 更适合重新做
内部 field 因果审计；在该审计完成前，不能直接把旧结论外推到 clean D0，也不能假定问题已经消失。

### 2.3 一项相关的内部 CLIP 负证据

历史 exp356 曾使用冻结 CLIP dense patch tokens 和 pose-pooled coarse part feature。teacher-only 小样本
中，same-ID 相对 different-ID 的 cosine gap 只有：global `+0.022`、head `+0.011`、torso `+0.009`、
legs `+0.013`，局部 CLIP feature 各向异性明显。随后 pose-mask 与 random-mask 两臂近似相同，主臂
相对对照下降约 `2.7 mAP`。

这不是当前“视觉局部 feature × body-part text prototype 的 sample-specific distribution”机制的直接
反例，但它禁止预设“冻结 CLIP patch token 天然就是高质量局部人体 teacher”。新方向必须先做
teacher-only 可辨识性门禁，不能直接靠 CLIP 名称获得 GO。

## 三、“所有 anchor 都直接预测”是否已经试过

需要纠正一个实现认知：clean exp389 不是“第一个 anchor 预测姿态、第二个 anchor 只预测 offset”。

- Stage-1 的 `early_anchor` 和 Stage-2 的 late `anchor` 参数完全独立；
- 两者都直接输出完整的 `17-joint heatmap + confidence`；
- early field 驱动 Stage-2 六个独立 PSG；
- late field 驱动 Stage-3 两个独立 PSG；
- 八个 consumer 都已被逐一证明能有限、非零地改变最终 descriptor。

exp389 final=`56.9/65.9/80.0/84.1`，相对 clean D0=`−0.7/−1.8/−0.8/−0.5`，所以已经可以判定：

> `双独立 absolute anchor + early 六次连续调制 + late 两次连续调制` 为 NO-GO。

但不能把它扩大成“所有 direct-anchor 多阶段定义都失败”，原因是：

1. exp389 只有两个 anchor，没有覆盖所有合理 stage interface；
2. early/late consumer 数为 `6/2`，容量、累计调制次数和训练轨迹不平衡；
3. exp391把pose objective从sum改为mean后，frozen level bypass显示early六bank贡献
   `+0.141 mAP`、late两bank贡献`+1.546 mAP`；early并非dead，但仍明显弱于late；
4. exp389 没有 CLIP 语义校准、跨层语义一致性或 consumer 语义约束。

exp391 H2-M natural final=`57.2/67.3/80.2/84.5`，相对单层D0=
`−0.4/−0.4/−0.6/−0.1`，触发预注册NO-GO；这封板的是无语义校准的纯结构链，不是永久否定
多阶段。语义门禁通过后，“每个stage一个direct anchor、每个state一个consumer、consumer-balanced”
仍应作为新的semantic multi-stage对照，但不能因为拓扑看起来更完整就直接重启exp391。

## 四、公开最近邻与 novelty 边界

| 工作 | 已覆盖机制 | 与候选方向仍有差异 |
|---|---|---|
| π-VL, arXiv:2308.02738 | human parsing 生成 pixel-level part text prompt；融合 multi-stage feature 做细粒度 V-L 对齐；辅助头推理删除；最终直接用 global feature | 对齐的是 feature map，不是可执行 control state；无 PSG、无 internal field 因果干预 |
| ALADIN, arXiv:2603.21482v2 | frozen CLIP 双编码 teacher；实例级 attribute/local region；global/attribute/local 多层 KD；推理只用 student BN-neck global feature | 蒸馏 appearance attribute 和最终/局部 feature；无 pose joint field、PSG 或 latent control intervention |
| RegionCLIP, CVPR 2022 | frozen visual teacher 的 region feature 与 concept text embeddings 相乘，经 softmax 得到 sample/region-specific soft distribution，再 KL 蒸馏 student region scores | 目标为区域视觉表征/开放词汇检测；无人体 pose state、ReID global descriptor 或 executable PSG |
| PAFormer, arXiv:2408.05918 | 明确提出 part prototype 缺乏 anatomical awareness；pose heatmap 监督 pose-token↔patch attention；推理不需 pose | 保留 part feature、visibility 和 part-to-part retrieval；无 CLIP teacher、PSG control field、global-only 输出 |
| MUVA, IET Computer Vision 2026 | head/upper-body/legs 文本 prompt；视觉 grounding mask；每个 CLIP ViT block 预测 residual part mask并注入 attention | CLIP 本身是学生架构，最终保留 global+part；不是训练期 teacher 校准 TAPF state |
| ProFD, ACM MM 2024 | part-specific text prompt、外部 parsing mask、hybrid decoder、自蒸馏、part representation | 测试保留 part 路径；不是 pose-pooled CLIP soft distribution 或 global-only latent control |
| CLIP3DReID, CVPR 2024 | CLIP image global KD；body-shape text tokens 与 student local feature 做 OT 对齐 | 不是 pose-pooled patch teacher，不产生 body-part soft distribution，也不控制 backbone 中间更新 |
| DenseCLIP / MaskCLIP | dense patch/pixel-text score map与局部 CLIP 语义监督 | 不是 ReID，也不是可执行 latent anatomical state |
| CLIP-ReID, AAAI 2023 | identity-specific prompt、global image-text alignment、推理可只用视觉分支 | 无局部 pose teacher、无多阶段 state、无语义因果干预 |
| KPR / BPBreID | 显式 part embedding、visibility 与 mutually-visible part retrieval | 最终不是单一 global descriptor；结构信息保留到 retrieval |

RegionCLIP 的代码已经逐步实现：teacher region feature归一化、teacher concept embedding归一化、
相似度/温度 softmax、筛选可靠区域、student logits 与 teacher distribution 做 KL。因此以下表述均不
能作为 novelty：

1. 首次用冻结视觉 teacher 与文本原型产生局部软标签；
2. 首次用 CLIP 为人体部位提供语义；
3. 首次用 pose 让 part 表示具备解剖意识；
4. 首次做训练期局部 V-L 对齐、推理删除 teacher；
5. 首次做 multi-stage/multi-scale body-part language supervision；
6. 首次最终只保留 global descriptor 的 inference-free part guidance。

可以谨慎争取、但暂时不能写“首次”的差分是：

1. CLIP 只校准 backbone 内部的 executable anatomical control state，而不蒸馏 retrieval descriptor；
2. 每个 state 都经真实下游路径改变最终单一 global descriptor；
3. 把 internal channel permutation、matched wrong field、spatial constant、zero、bypass 作为主成功
   指标，直接解决 latent semantics 的可辨识性，而不是只看可视化或 KD loss；
4. CLIP soft distribution只是校准工具，不是贡献本体。

## 五、对双编码器 teacher 的机制审查

候选 teacher 定义为：

```text
geometrically aligned RGB -> frozen CLIP image encoder -> patch tokens V
body-part prompts         -> frozen CLIP text encoder  -> prototypes T
pose heatmap H_j pools V  -> local visual feature v_j
q_j = softmax(cos(v_j, all T) / tau)
```

这个定义比纯 text prototype 更好，因为 `q_j` 同时依赖当前 RGB、当前局部视觉内容和全部文本原型；
但它有三个必须先验证的问题。

### 5.1 循环监督风险

pose heatmap 已经告诉系统“当前池化的是第 j 个关节/部位”。如果 CLIP 分布几乎总是固定 one-hot j，
则 CLIP 没有提供超出当前 per-channel pose supervision 的新信息；如果它频繁输出其它类别，又可能是
CLIP patch 分辨率或人体部位语义不足造成的噪声。

只有当 `q_j` 同时满足以下条件时，sample-specific teacher 才有意义：

- 对期望部位具有可辨识 top-1/margin；
- 对 wrong RGB、wrong pose mask、wrong text label 都敏感；
- 同一部位跨样本不是完全相同的常量分布；
- 在遮挡/低置信样本上合理地提高 entropy 或降低 margin。

### 5.2 17 joints 的分辨率与左右歧义

按当前 `384×128` 输入，CLIP ViT-B/16 在做矩形 positional interpolation 后也只有约 `24×8`
patch；肘、腕、膝、踝常只覆盖一到两个 patch。若像历史实现一样先拉伸到 `224×224`，还会额外
引入人体比例失真。左右对称文本又存在 anatomical-left 与 observer-left 的歧义。即使换
ViT-L/14，局部 token 的人体解剖判别性也不能由分辨率自动保证。

因此，不建议让 CLIP 直接承担 17-joint channel identity 的主监督。17-joint geometry 应继续由 pose
teacher 直接监督；CLIP 首先只校准 head/torso/arms/legs 等 coarse regions。只有 teacher-only 审计
证明逐 joint，尤其左右六对关节的识别显著高于随机，才允许升级到 17-joint CLIP KD。

### 5.3 entropy 不自动等于 occlusion

CLIP teacher entropy 可能来自：

- 左右对称；
- patch grid过粗；
- prompt措辞；
- 衣物/背景/遮挡物占据局部 patch；
- 真正的不可见或遮挡。

所以未经 visible/low-confidence/invalid 与合成遮挡分组校准，late state 只能称 semantic uncertainty，
不能称 occlusion/reliability state。

## 六、当前方案的 projector/consumer 漏洞

原候选方案让每个 anchor field 池化对应 stage 的 ReID feature，再通过 projection 蒸馏 teacher
distribution。这里存在两个吸收路径：

1. 高容量 projector 可从 stage feature 中直接预测固定部位标签，anchor field 不必准确；
2. 即使 anchor 被校准，原 PSG 的自由通道混合仍可只用 field 总量、空间均值或静态模式。

### 6.1 最小梯度边界

- 两个 CLIP encoder：永久 `eval + requires_grad=False + no_grad`，不进 optimizer；
- CLIP patch tokens、text prototypes、teacher distribution：全部 detach；
- semantic KD 池化所读取的 stage ReID feature：stop-gradient；
- pose loss、CLIP KD、跨层一致性：只更新 anchor 和受限共享 projector；
- ReID loss：只更新 backbone、PSG、head；calibrated state 进入 PSG 前 detach；
- 各 joint、各 stage 不得拥有独立 classifier；
- projector 无 per-joint bias，最终 semantic classifier跨 stage/joint共享；
- 分别 backward `L_pose/L_clip/L_consistency/L_reid`，记录每组参数梯度所有权以及
  `L_pose`/`L_clip` 在 anchor 上的梯度余弦。

### 6.2 更严格的无 projector-bypass 版本

在正式采用“stage ReID feature + learnable projection”前，应先考虑一个更可辨识的 teacher loss：

```text
q_teacher = text_logits(pool(pose_heatmap, frozen_CLIP_patch_tokens))
q_anchor  = text_logits(pool(predicted_anchor_field, frozen_CLIP_patch_tokens))
L_sem     = KL(q_teacher || q_anchor)
```

这个版本没有 learnable projector，梯度只能通过 predicted anchor field 的池化权重回到 anchor，更适合
验证 CLIP 是否真的改善 state 的语义定位。它仍不能单独保证 PSG 使用语义，但可以作为原 projection
版本的必要 control。

### 6.3 consumer 必须消费校准后的语义状态

若 semantic distribution `Q_s` 只存在于训练期诊断 head，PSG 仍只读原 field，则语义校准与执行
路径没有闭环。至少需要满足以下之一：

1. PSG 显式读取 detached `(field, Q_s)`；
2. 使用 joint/region-specific、无 bias 的受限 expert bank，使通道置换会把空间支持路由到错误的
   semantic expert；
3. 使用可空分离的 state-innovation consumer，并对 correct/permuted state 施加 matched 因果约束。

一个更清楚的结构候选是：

```text
DeltaF_s(p) = sum_j A_s(j,p) * E_s,j(F_s(p))
F'_s        = F_s + bounded(DeltaF_s)
```

其中 `A_s` 是被 pose/CLIP 校准的 semantic state，`E_s,j` 是 joint/region-specific 轻量更新方向；
zero/null state必须严格 identity。这样 joint permutation 会把正确空间支持路由给错误 expert，语义
身份在结构上进入 consumer，而不是先被一个自由 `17→32` 卷积任意压缩。

该结构仍需 parameter-matched static control、expert-collapse 检查和冻结干预，不能因“结构上可解释”
就自动获得因果结论。

## 七、两种多阶段定义的裁决

### 7.1 各 stage 独立预测同一套 17 joints

优点：

- 最直接回答“所有 anchor 都自己算姿态”；
- 每层都有明确 COCO-17 监督，通道定义一致；
- cross-stage consistency容易定义；
- 可作为 progressive offset/refinement 的强对照。

风险：

- exp389 已证明双 direct-anchor 版本不会自动涨点；
- 各层重复同一 17-joint 任务，可能只学成同一模板；
- 普通 KL consistency 会抹掉 stage-specific 信息；
- CLIP 对细关节和左右语义很可能不过门禁。

裁决：保留为 teacher-only 17-joint 门禁通过后的直接机制对照，不作为首个正式方案。

### 7.2 分层粒度

更合理的第一候选是：

```text
early  : 17-joint geometry，由 pose teacher 直接监督
middle : head/torso/arms/legs 等 coarse region，由 CLIP 双编码 teacher 校准
late   : semantic uncertainty；只有通过遮挡校准后才改称 reliability/occlusion
```

不同粒度不能直接做普通 KL consistency。必须预先定义固定映射：

```text
17 joints --fixed aggregation matrix--> coarse regions
joint/region confidence --fixed reduction--> uncertainty/reliability
```

一致性只在映射后的公共空间计算。这个定义更符合 stage 的空间分辨率和语义带宽，也避免每层重复同一
任务；但它与 π-VL/MUVA 的 hierarchical/multi-layer body semantics 重叠更大，所以论文贡献仍必须
落在 executable latent state 和 causal identifiability，不能落在“分层粒度”四个字上。

### 7.3 stage 放置与 consumer-balanced

不应在最终 Stage-3 输出之后创建没有下游 block 的 anchor/consumer。Swin-T 首个可审计拓扑应满足：

- source 位于 Stage-0/1/2 或 final stage 内部 block 之间；
- 每个 state 只对应一个主要 consumer site，避免 exp389 的 `6/2` 累计调制不平衡；
- 每个 consumer 后仍有至少一个真实 block/downsample/GAP 路径；
- 逐 consumer bypass 必须有限、非零改变 descriptor；
- 参数量、consumer 数和累计 release 需做 matched control。

## 八、正式设计前的零训练门禁

### Gate 0A：clean D0 内部 field 冻结审计

必须在 GPU 空闲且不影响任何正式 arm 时，对一个已封板 clean D0 checkpoint 做：

1. correct-start/end；
2. matched wrong image-derived field；
3. 固定 17-cycle channel permutation；
4. COCO left/right-only permutation；
5. confidence permutation；
6. spatial constant；
7. zero field；
8. PSG bank0/bank1/all bypass。

每臂必须证明 field、gate 和 descriptor 的干预实际生效，hook完整恢复，checkpoint state不变。解释沿用
旧语义审计预注册区间：

- `|delta mAP| < 0.1`：当前 checkpoint 下无可辨贡献；
- `0.1–0.3`：弱/不确定；
- correct相对干预 `>= +0.3 mAP`：支持被破坏因素具有可辨贡献。

若 clean 仍出现 `permutation/wrong≈correct` 但 `zero/bypass` 明显下降，新方向必须明确写成
“修复 consumer semantics”，不能只强化 anchor。

### Gate 0B：CLIP teacher-only 可辨识性

只使用 train split；不训练 student，不启动正式 experiment。必须报告：

1. expected joint/region top-1、top-k、expected-vs-best-negative margin；
2. 每个 joint 单独结果，尤其左右六对的 pairwise accuracy 与 bootstrap 95% CI；
3. correct pose mask 对 channel-shuffle、uniform、random、matched-wrong mask 的差异；
4. 固定 pose mask替换异样本 RGB patch tokens；
5. 固定 RGB替换/置乱 pose masks；
6. 固定视觉 feature置乱 text prototype label；
7. 同一 joint跨样本 distribution 方差、JSD和有效秩，排除固定常量 teacher；
8. visible/low-confidence/invalid/合成遮挡的 entropy 与 margin；
9. 水平翻转后反翻空间并交换 COCO left/right 通道的 equivariance；
10. CLIP 权重、tokenizer、prompt bank、预处理、patch-token 提取位置与所有 artifact SHA。

硬 kill-switch：

- 左右关节辨识不显著高于随机：停止 17-joint CLIP KD；
- correct mask不优于 random/shuffle：停止该粒度；
- distribution近似常量：不得称 sample-specific teacher；
- entropy不随遮挡/可靠性变化：不得把它命名为 occlusion state；
- full teacher对 wrong RGB 或 wrong text不敏感：说明双编码器的一侧没有真实贡献。

## 九、最小实验顺序（仅预案，不授权启动）

1. exp390/391均已封板，不重启、不续训；
2. 做 clean D0 frozen internal-field audit；
3. 做 CLIP teacher-only audit；
4. 先在单层 D0 上加入 semantic calibration，隔离 CLIP 是否修复语义；
5. 单层必须同时满足：
   - correct field优于 joint permutation/wrong field；
   - PSG bypass仍有贡献；
   - projector-only/static control不能复现；
   - 推理严格 RGB-only；
6. 单层语义因果通过后，才以它为新基线比较：
   - consumer-balanced independent 17-joint anchors；
   - hierarchical granularity；
   - 各自 matched no-CLIP control；
7. 只有语义敏感性、final retrieval、多 seed和多数据集同时成立，才进入论文主线。

以下情况直接 kill 当前实现：

- KD梯度主要进入 projector，anchor梯度近零；
- calibrated state没有实际进入 PSG；
- 任一 anchor/consumer 无最终 descriptor 下游路径；
- 训练后 channel permutation/wrong field仍 `<0.1 mAP`，但 bypass很大；
- 性能上涨而语义因果失败：只能称 generic CLIP KD；
- 语义敏感性成立但 retrieval不升：说明诊断问题被修复，但暂不构成主方法性能证据。

## 十、B 类创新门槛裁决

当前方向满足“问题层面有新意”的潜力：它不再问“加 CLIP 是否涨点”，而是问一个训练期 latent
anatomical state 是否具有可辨识语义、是否被最终检索路径因果执行。

机制层面目前只部分满足。若沿用自由 PSG 加一个 projector KD，审稿人很容易将其概括为：

```text
RegionCLIP soft KD + PAFormer anatomy + π-VL/MUVA multi-level guidance + TAPF gate
```

这不足以支撑主贡献。只有 consumer 在结构上消费 calibrated state，并且 parameter-matched control 与
内部 field intervention 同时通过，机制贡献才可能成立。

证据层面可以设计得很强：clean internal-field counterfactual、teacher-only identifiability、projector
absorption control、consumer reachability、multi-seed/multi-dataset final 都是清楚、可证伪的证据链。

最终 verdict：

> **方向保留，状态 RESEARCH-ONLY / NO-START。优先做语义因果与 teacher-only 两个零训练门禁；
> 先在单层修复 consumer semantics，再重新比较 semantic single-stage 与 semantic multi-stage。
> 不得直接把 CLIP KD 叠到 exp389/391 多阶段结构上抢跑，但也不得把 exp391 的纯结构NO-GO扩大成
> 对CLIP语义校准多阶段TAPF的永久否定。**

## 参考来源

1. π-VL: [Exploring Part-Informed Visual-Language Learning for Person Re-Identification](https://arxiv.org/abs/2308.02738)
2. ALADIN: [Attribute-Language Distillation Network for Person Re-Identification](https://arxiv.org/abs/2603.21482)
3. RegionCLIP: [paper](https://arxiv.org/abs/2112.09106), [code at commit 4b8513b](https://github.com/microsoft/RegionCLIP/tree/4b8513b56e24827e3d6468e1f2105869f35c2d0b)
4. PAFormer: [Part Aware Transformer for Person Re-identification](https://arxiv.org/abs/2408.05918)
5. MUVA: [paper](https://arxiv.org/abs/2603.14012), [code at commit 8965263](https://github.com/RikoLi/MUVA/tree/896526309c3392abc01c4499b792606c3574d3b4)
6. ProFD: [paper](https://openreview.net/forum?id=o2axlPlXYY), [code at commit 14e47d3](https://github.com/Cuixxx/ProFD/tree/14e47d3b04f541d2a614482848bba2071bc90cda)
7. CLIP3DReID: [project](https://cvlab.cse.msu.edu/project-clip3dreid.html), [code at commit f2a0afa](https://github.com/liufeng2915/Clip3DReid/tree/f2a0afa2cc09f8d1b72f345d3a992cdcb3b7c6d8)
8. DenseCLIP: [paper](https://arxiv.org/abs/2112.01518), [code](https://github.com/raoyongming/DenseCLIP)
9. MaskCLIP: [paper](https://arxiv.org/abs/2112.01071), [code](https://github.com/chongzhou96/MaskCLIP)
10. CLIP-ReID: [code](https://github.com/Syliz517/CLIP-ReID)
11. KPR: [paper](https://arxiv.org/abs/2407.18112), [code](https://github.com/VlSomers/keypoint_promptable_reidentification)
12. BPBreID: [paper](https://arxiv.org/abs/2211.03679), [code](https://github.com/VlSomers/bpbreid)

本轮公开代码只读核验 commit：RegionCLIP `4b8513b`、DenseCLIP `5eda47f`、MaskCLIP `245d31f`、
CLIP-ReID `eb1898b`、CLIP3DReID `f2a0afa`、ProFD `14e47d3`、KPR `e3e6ee2`、BPBreID
`a2dc430`、MUVA `8965263`。公开仓库与 PDF 均只存放在临时审查目录，不进入正式 runtime。
