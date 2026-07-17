# exp378：姿态先验初始化、ReID-only 自适应空间控制器查新

> 查新日期：2026-07-16
> 范围：端到端/联合 pose–ReID、姿态先验初始化后仅用 ReID loss 微调、内部热图或
> attention map、训练期 heatmap teacher、无外部姿态模型推理，以及 PAFormer、PABR、
> PGFL-KD、PFD、KPR、ProFD、PeVL、PAAB、MUVA 等直接或机制近邻。

## 结论先行

需要把两个不同版本分开裁决：

1. **持续 pose loss 的多任务版本**：已有 2019 年 Visual Person Understanding 和
   2023 年 Pose Auxiliary VI-ReID 等直接先例，不能作为新方法；
2. **pose teacher 只作初始化/早期 bootstrap，随后唯一由 ReID loss 塑造内部空间图**：
   也已有一个更直接的先例——ECCV 2018 **PABR**。PABR 用预训练 OpenPose 子网络
   初始化内部 part-map stream，之后不使用 ReID 数据集上的 part/pose 标注，只优化
   ReID triplet loss，并明确证明 trainable pose stream 优于 frozen pose stream。

因此，当前朴素 exp378——“轻量高分辨率 pose head + 前期拟合 ViTPose heatmap + 后期
ReID-only 解冻 + 推理不调用外部 pose estimator”——**可以作为有价值的诊断实验，不能
单独承担论文主创新**。轻量 HRNet-style head、Stage-2 接头、前若干 epoch 冻结、逐步
切换 teacher/predicted map 都是架构或训练日程差异，尚不足以越过 PABR 和 PAFormer。

这并不等于该方向完全不能做。可以争取的方法对象应收紧为：

> **姿态先验如何被有控制地释放为 ReID 自适应空间控制器，同时保持可验证的解剖语义，
> 而不是漂移成任意 identity attention。**

只有加入一个不可归约的“受控释放/语义守恒”机制，并通过 pose drift、identity leakage、
正确/错配姿态干预和参数匹配 generic-mask 对照证明，才可能形成论文贡献。单靠 curriculum
或更轻的 heatmap head 不够。

---

## 一、被审计的精确定义

当前讨论过的实现可归纳为：

1. 从 Swin Stage-2 或相邻高分辨率视觉特征预测 17 个关节图/置信度；
2. 用离线 ViTPose heatmap 初始化权重或只在早期 epoch 做 bootstrap；
3. 随训练推进，从外部 teacher heatmap 切换到内部预测 map；
4. 后期去掉 pose loss，使内部 map 只接受 ReID loss；
5. map 在 backbone 中间控制视觉特征，最终仍输出 ReID descriptor；
6. 测试不调用独立 ViTPose/OpenPose，但内部 pose/map head 仍参与前向。

第 6 点应准确写成“**无外部 pose estimator 推理**”，不宜笼统写成“pose-free
inference”。只要内部 head 仍显式预测 17 个关节图，推理计算图就仍包含 pose estimation；
PAFormer/PGFL-KD 那种测试路径完全不读取 pose heatmap 或移除 pose branch，才更接近
严格 pose-free。

---

## 二、最强直接先例：PABR 已覆盖“pose 初始化后 ReID-only 自适应”

### 2.1 论文事实

Yumin Suh et al., *Part-Aligned Bilinear Representations for Person
Re-identification*, ECCV 2018：

- 论文：<https://arxiv.org/abs/1804.07094>
- 本地原文：`tmp/pdfs/joint_pose_reid/pabr.{pdf,txt}`

其结构与训练事实是：

1. 网络包含 appearance-map extractor `A` 和 part-map extractor `P`；
2. `P` 由 OpenPose 子网络 `P_pose` 构成，并用 COCO 预训练 pose 权重初始化；
3. appearance map 与 part map 在对应空间位置做 bilinear pooling，part map 因而直接控制
   哪些 appearance feature 被怎样写入最终 ReID 表征；
4. 在 ReID 数据上**没有 part annotation 或 pose loss**，总目标只有 triplet ReID loss
   （论文 Eq. 7）；
5. 作者明确比较 fixed `P_pose` 与 trainable `P_pose`，后者同样从 pose 权重初始化，
   然后只用 ReID loss fine-tune，结果明显更强；
6. 论文的原话结论就是：part maps 在预训练 pose model 的 guidance 下，被学习成
   “optimal/specially adapted for person re-identification”；
7. 推理运行网络内部 `P_pose` stream，不需要再调用一个网络外部的 pose estimator。

这不是宽泛相关，而是对用户补充版本的核心操作逐项覆盖：

| 当前 exp378 命题 | PABR 2018 |
|---|---|
| pose prior 仅作初始化/先验 | OpenPose/COCO 权重初始化 `P_pose` |
| 后期不用 pose 标注 | 全程在 ReID 数据上不使用 part/pose 标注 |
| 唯一由 ReID loss 塑造空间图 | 总损失只有 ReID triplet |
| 内部空间 map 控制 appearance | part map 与 appearance 做逐位置 bilinear pooling |
| trainable 比 frozen 更有用 | 论文直接做 fixed/trainable `P_pose` 对照 |
| 无网络外部 pose estimator | pose 子流已集成到 ReID 网络前向 |

### 2.2 仍有的字面差异为什么不够

exp378 与 PABR 仍有真实差别：

- Swin Stage-2 而不是双 CNN stream；
- 17 个显式 heatmap，而不是 latent part descriptor map；
- ViTPose teacher heatmap bootstrap，而不是直接复制 OpenPose 子网络权重；
- progressive handoff，而不是从第一个 iteration 就 ReID-only fine-tune；
- 可能用 PSG/其他中间控制器，而不是 bilinear pooling；
- 轻量 HRNet-style 高分辨率 decoder，而不是 PABR 的 OpenPose/CPM 子网。

这些差异足以构成一个新的工程实现，却没有改变上位机制：

> pose-pretrained spatial branch → ReID-only task adaptation → internal map controls
> appearance representation。

所以不能写“首次用 ReID loss 让 pose predictor 任务自适应”“首次在不持续 pose
supervision 下学习 ReID-oriented pose map”或“首次把 pose 网络集成进 ReID 推理”。

### 2.3 PABR 也是 exp378 最强必须对照

正式实验不一定要复刻 PABR 的大 bilinear descriptor，但必须复现它的核心因果比较：

- `FROZEN`：pose-prior/teacher 初始化后冻结 map head；
- `REID-FT`：相同初始化，仅用 ReID loss 微调；
- `RANDOM-FT`：相同结构随机初始化，仅用 ReID loss；
- `POSE-MTL`：持续 pose loss + ReID loss；
- `GENERIC-MASK`：参数匹配、没有关节语义的普通 spatial controller。

如果 exp378 只能复现 `REID-FT > FROZEN`，其结论仍属于 PABR 已有结论；必须证明新增
机制相对这五个对照提供了独立收益或新的可验证性质。

---

## 三、第二个最强近邻：PAFormer 已覆盖“heatmap 教内部定位，测试去外部姿态”

Hyeono Jung et al., *PAFormer: Part Aware Transformer for Person
Re-identification*, arXiv:2408.05918v1：

- 论文：<https://arxiv.org/abs/2408.05918>
- 本地原文：`tmp/pdfs/paformer.{pdf,txt}`

PAFormer 不是显式 17-joint pose decoder，但其功能路径与 exp378 非常接近：

1. PifPaf 离线 heatmap 只在训练阶段构造 part heatmap/visibility 监督；
2. 模型内部 pose token–patch token cross-attention 产生空间 attention map；
3. attention map 用 pose heatmap MSE 直接监督（§4.3, Eq. 2）；
4. 该 map 聚合视觉 token 得到 part feature，part feature 同时接受 ID 与 triplet loss
   （§4.4, Eq. 5），因此 ReID 梯度会反向塑造内部定位图；
5. 测试时不输入 pose heatmap，也不调用 localization module；内部 attention 自行定位；
6. 另有 learned visibility predictor 和 heatmap visibility teacher forcing。

因此 PAFormer 已封住：

- “训练时用 heatmap 教模型内部定位，测试时不需要 pose heatmap”；
- “由视觉特征自行产生 pose/part-aware attention map”；
- “该内部 map 同时受姿态监督和 ReID 目标影响”；
- “通过 learnable visibility 处理遮挡部位”。

exp378 的持续 pose-loss 版本尤其容易被评为“PAFormer attention map 换成显式 heatmap
decoder，再接回 Swin”。把 pose loss 在后期关闭会让它更接近 PABR，而不是自动获得新颖性。

---

## 四、其他直接 pose–ReID 联合训练先例

### 4.1 Visual Person Understanding：共享 backbone 联合预测 pose 与 ReID

Kilian Pfeiffer et al., *Visual Person Understanding through Multi-Task and
Multi-Dataset Learning*, 2019：

- 论文：<https://arxiv.org/abs/1906.03019>
- DOI：<https://doi.org/10.1007/978-3-030-33676-9_39>
- 本地原文：`tmp/pdfs/joint_pose_reid/vpu.{pdf,txt}`

该工作用共享 ResNet backbone 和轻量 pose head 同时做 ReID、pose、parsing、attribute，
已经覆盖“中间视觉特征接轻量 pose head并联合训练”。更重要的是，它专门比较：

- pose pretrain → 只 fine-tune ReID；
- pose + ReID joint training；
- shared/split/multi-branch backbone。

其 Table 7 显示，pose pretrain 后只做 ReID fine-tune 会严重擦除 pose：single-branch 的
pose PCKh 从 `86.4` 跌到 `9.0`（MPII）/`10.0`（LIP），作者明确解释为 pose-invariant
ReID training erases pose-related information。

这说明“后期只用 ReID loss”不是天然等于“任务自适应姿态”。它可能只是把 pose head
漂移成 identity attention。exp378 必须量化漂移，不能只展示 ReID 涨点和几张热图。

### 4.2 Pose Auxiliary VI-ReID：内部热图预测 + pose/ID/triplet 联合约束

Yunqi Miao et al., *On Exploring Pose Estimation as an Auxiliary Learning Task
for Visible-Infrared Person Re-identification*, Neurocomputing 2023：

- 预印本：<https://arxiv.org/abs/2201.03859>
- DOI：<https://doi.org/10.1016/j.neucom.2023.126652>
- 本地原文：`tmp/pdfs/joint_pose_reid/aux_vi.{pdf,txt}`

该方法：

- 在 ReID backbone 内集成 pose estimation branch；
- 预测 body keypoint heatmaps；
- 用 LIP 预训练 pose estimator 产生 pseudo heatmap；
- pose branch 同时接受 heatmap MSE、identity loss、hetero-center triplet；
- pose mask/pose feature 与 ReID feature 结合；
- 总损失明确为 `L_id + beta L_hctri + lambda L_pose + gamma L_KD`（Eq. 21）；
- 推理使用网络内部 pose/ReID 两支提取的特征，不再需要伪标签 teacher。

它与 exp378 的持续多任务版本近乎完整重合；visible-infrared 场景差异不能支持“首次在
person ReID 中”一类 claim。

### 4.3 PGFL-KD / PGDS / TSD：训练用结构 teacher，测试去结构模型

这些工作不等同于 exp378 的内部 map head，但共同封住“训练期 pose/structure、测试期
无外部结构模型”这一宽泛故事：

- [PGFL-KD, ACM MM 2021](https://arxiv.org/abs/2108.00139)：pose-guided branches
  在训练期教 main branch，测试只保留 main global branch；
- [PGDS, AVSS 2024](https://github.com/huyquoctrinh/PGDS)：冻结 OpenPose 特征以多层
  KL 深监督 Swin，测试丢弃 pose encoder；
- [TSD, ICASSP 2024](https://arxiv.org/abs/2312.09797)：parsing-aware teacher
  decoder 蒸馏 pose-free student decoder。

因此“推理速度不包含外部 ViTPose”“训练时 privileged pose，测试时 RGB-only”只能是
效率属性，不能单列为创新。

---

## 五、用户点名工作与 exp378 的真实关系

| 工作 | 已覆盖内容 | 与 exp378 的关系 | 风险等级 |
|---|---|---|---:|
| PABR, ECCV 2018 | pose 网络初始化；随后 ReID-only fine-tune；内部 part map 控制 appearance | **最直接撞车用户修订版** | 极高 |
| PAFormer, 2024 | heatmap 监督内部 attention；part ReID loss；测试无 heatmap | **最直接撞车 heatmap-bootstrap/内部定位版** | 极高 |
| Pose Auxiliary VI-ReID, 2023 | 内部 keypoint heatmap head；pose+ID+triplet；内部 pose branch 推理 | 直接撞车持续多任务版 | 极高 |
| PGFL-KD, MM 2021 | pose 训练期特权监督；测试只保留 global main branch | 封住 pose-free/无外部姿态故事 | 高 |
| PFD, AAAI 2022 | 在线 HRNet heatmap、pose-conditioned part/global decoder | 覆盖 pose heatmap 驱动 Transformer/ReID feature；但测试仍需 pose | 中高 |
| KPR, ECCV 2024 | keypoint prompt 直接进入 Swin token；target/distractor prompts；part matching | 覆盖 pose-conditioned Swin；但 pose 是测试输入，问题是多人歧义 | 中 |
| ProFD, MM 2024 | CLIP part proxy、双向 cross-attention、part decoder | 不做 pose prediction；若 exp378 接普通 part decoder，则构成强结构近邻 | 中 |
| PeVL, CVPR 2024 | pose mask 调制 CLIP visual attention | 非 ReID，但封住 pose-conditioned CLIP attention | 中 |
| PAAB/PAAT, 2023 | pose-pair mask 加入 ViT attention logits并更新 token | 非 ReID，但封住 pose mask→attention→residual update | 中 |
| MUVA, 2026 | ReID 中动态 body-part mask 逐层进入 CLIP ViT self-attention | 若 exp378 map 逐层改 attention，则是直接机制近邻 | 高 |
| ProFD/PFD/PAFormer 家族 | part query/pose token/decoder/visibility | part token、matching、visibility 不能拿来补第二贡献 | 高 |

这里的层级很重要：KPR、ProFD、PeVL、PAAB 并不是“端到端 pose predictor”直接先例，
不能因为它们沾边就宣布整条线不可能；真正导致朴素 exp378 失去 headline 资格的是
**PABR + PAFormer + Pose Auxiliary VI-ReID** 三篇的联合覆盖。

---

## 六、候选 claim 裁决

| 候选 claim | 裁决 | 原因 |
|---|---|---|
| 首次端到端联合 pose estimation 与 person ReID | 不可写 | VPU、Pose Auxiliary VI-ReID |
| 首次让 ReID loss 微调 pose estimator/pose branch | 不可写 | PABR 2018 直接先例 |
| 首次只用 pose 权重初始化，之后仅用 ReID loss | 不可写 | PABR 摘要、§3.3、§6.3 明确覆盖 |
| 首次学习 task-adaptive pose/part map | 不可写 | PABR 就以此作为核心结论 |
| 首次用 pseudo heatmap 监督内部 pose head | 不可写 | Pose Auxiliary VI-ReID |
| 首次用 heatmap 监督内部 attention，测试去 heatmap | 不可写 | PAFormer |
| 首次训练用 pose、测试不用外部 pose estimator | 不可写 | PGFL-KD、PAFormer、PGDS、TSD |
| 首次在 Swin Stage-2 接轻量 high-resolution pose head | 可能字面未见，但只是位置/实现差异 | 不足以当主贡献 |
| 首次 freeze N epoch 再 progressive handoff | 精确日程未见直接相同 | curriculum 本身不足以构成方法级新意 |
| HRNet-style head 更轻 | 可作为效率设计 | HRNet/高分辨率 head 是成熟架构，不是新意 |
| bootstrap 后只保留 ReID 单任务 | 区别于持续 pose 多任务，但不可单独写首次 | PAFormer 与多任务工作不同；PABR 已覆盖 ReID-only 自适应 |
| ReID-only adaptation 后热图仍保持解剖语义 | **可作为待验证性质，不能先声称** | PABR 有定性图；VPU 又显示可能彻底遗忘，需要更强定量证据 |
| 可靠性分组、低带宽、可测漂移边界下的受控释放 | **有条件可争** | 当前直接先例未覆盖完整联合机制，仍需专项查新与实验证明 |
| pose prior 比 random/generic spatial prior 提供独立收益 | **可作为证据贡献** | 必须参数匹配并跨 seed；单次涨点不够 |

---

## 七、如果继续 exp378，最小实验应回答什么

### 7.1 把实验定位成“诊断”，而不是先宣布新方法

最小问题应改成：

> 对当前 Swin/遮挡 ReID，pose-prior-initialized spatial map 在撤掉 pose loss 后，究竟
> 保留了解剖语义并产生额外身份效用，还是只是退化为一个普通 identity attention head？

这个问题仍值得跑，因为 PABR 在传统 CNN/holistic benchmarks 上成立，不代表它在当前
Swin、Occluded-Duke 和 PSG 证据边界下必然成立。

### 7.2 必做同构对照

所有对照必须使用相同 backbone、head 参数量、batch size、优化器和初始化 checkpoint：

1. `B0`：无 controller；
2. `FROZEN-POSE`：pose-bootstrap 后永久冻结；
3. `REID-ONLY`：pose-bootstrap 后仅由 ReID loss 更新；
4. `PERSISTENT-MTL`：全程 pose + ReID loss；
5. `RANDOM-REID`：随机初始化相同 head，仅由 ReID loss 更新；
6. `GENERIC-SPATIAL`：取消 17-joint 语义、保留相同容量的 spatial mask head；
7. `PAFORMER-LIKE`：heatmap 直接监督内部 attention map并保持到训练结束；
8. `TEACHER-INPUT`：始终使用离线 ViTPose map，作为可达到的 fixed-pose 上界。

最关键的 paired delta 不是 `REID-ONLY - B0`，而是：

- `REID-ONLY - FROZEN-POSE`：是否复现 PABR 的 task adaptation；
- `REID-ONLY - RANDOM-REID`：pose prior 是否有独立价值；
- `REID-ONLY - GENERIC-SPATIAL`：收益是否来自 anatomy，而不是多一个 attention head；
- `REID-ONLY - PERSISTENT-MTL`：撤掉 pose supervision 是否真的必要；
- `REID-ONLY - PAFORMER-LIKE`：新机制是否超出直接近邻。

### 7.3 不能只报 mAP/R1

至少同步报告：

1. 每个 epoch 的 mAP/R1/R5/R10；
2. bootstrap 结束与 e120 的 PCK/OKS、joint centroid error 或 teacher-map Wasserstein/KL；
3. 可靠关节/遮挡关节分开统计的 heatmap drift；
4. flip/crop/scale equivariance；
5. 从 heatmap 或低维 map latent 预测训练 ID 的 linear-probe 准确率；
6. correct teacher、matched-donor、canonical、channel permutation、zero/bypass 干预；
7. controller 对 final descriptor 的实际 delta，而不是只看 map 可视化；
8. 参数量、FLOPs、显存、吞吐和内部 pose head 的推理开销。

如果 ReID 指标上升，但 PCK/等变性崩溃且 heatmap ID probe 很高，只能写“pose-initialized
spatial attention”，不能继续称为“任务自适应姿态”。

---

## 八、什么机制才可能越过 PABR/PAFormer

### 8.1 不够的改动

以下都不能单独恢复主创新资格：

- 把 OpenPose 换成 ViTPose；
- 把 CNN pose stream 换成 HRNet-lite 或 Swin Stage-2 decoder；
- 17 joint map 换 5/6 body-group map；
- 前 5/10/20 epoch 冻结；
- teacher/student map 线性插值；
- 只解冻 adapter/LoRA；
- zero-init、较少参数、更快推理；
- 把 map 接到 PSG、PAA、FiLM 或普通 attention gate。

### 8.2 有条件可争的对象：语义守恒的受控释放

若要继续把这条线发展成方法，建议把新意落在**谁能偏离、偏离多少、偏离后仍如何被
认定为姿态**，而不是“是否解冻”。例如：

1. 每个关节只允许通过低维坐标/协方差 residual 改变，不允许任意像素图藏 identity；
2. teacher 置信度高的关节保持紧 trust region，低置信/遮挡关节才获得较大自由度；
3. 通过显式带宽或几何等变约束保证 controller 仍是 anatomical posterior；
4. ReID loss 只能决定受限 residual 的方向，不能任意重写整张 map；
5. 使用可测的 drift/leakage budget，而不是只调 `lambda_pose`；
6. map 必须控制一个相对 generic attention 不同、可被因果消融的 feature operation。

这仍不能直接宣称首次；但相对 PABR 的“无约束 ReID-only fine-tune”和 PAFormer 的“固定
加权 heatmap MSE”，至少形成了机制级差分。若没有这类约束，exp378 最多是 PABR 在
Swin/遮挡 ReID 上的现代化复现。

---

## 九、最终裁决

### 研究执行裁决

- **作为快速诊断：GO。** 它能回答当前 2D heatmap 线最后一个重要问题：内部 map 经
  ReID-only 自适应后是否比 fixed/canonical/generic map 更有用。
- **作为论文 headline 的朴素版本：NO-GO。** PABR 对用户修订版本构成直接先例，
  PAFormer 对 heatmap-supervised、无外部姿态推理版本构成直接先例。
- **作为“语义守恒受控释放”版本：CONDITIONAL GO。** 必须先把机制、漂移预算、
  identity leakage 和强对照写进 design，再进行实现与训练审查。

### 最安全的论文表述

若诊断成功，可以写：

> 我们在现代层次 Transformer 的遮挡 ReID 设置中系统研究了 pose-prior spatial maps 从
> 固定几何先验向 ReID-adaptive controller 的转变，并用冻结、随机初始化、持续 pose
> supervision、generic spatial controller 和语义漂移干预分离其真实来源。

这是一条**实证研究/机制分析**表述，不是首创 claim。只有新的受控释放机制稳定优于
上述强对照后，才可以把它升级为方法贡献。

---

## 十、可核验来源

- PABR：<https://arxiv.org/abs/1804.07094>
- Visual Person Understanding：<https://arxiv.org/abs/1906.03019>
- Pose Auxiliary VI-ReID：<https://arxiv.org/abs/2201.03859>
- PGFL-KD：<https://arxiv.org/abs/2108.00139>
- PFD：<https://arxiv.org/abs/2112.02466>
- PAFormer：<https://arxiv.org/abs/2408.05918>
- KPR：<https://arxiv.org/abs/2407.18112>
- ProFD：<https://arxiv.org/abs/2409.20081>
- TSD：<https://arxiv.org/abs/2312.09797>
- PeVL：<https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_PeVL_Pose-Enhanced_Vision-Language_Model_for_Fine-Grained_Human_Action_Recognition_CVPR_2024_paper.html>
- PAAB/PAAT：<https://arxiv.org/abs/2306.09331>
- MUVA：<https://arxiv.org/abs/2603.14012>

本地直接证据优先来自已下载全文，而不是只依赖搜索摘要：

- `tmp/pdfs/joint_pose_reid/pabr.txt`
- `tmp/pdfs/joint_pose_reid/vpu.txt`
- `tmp/pdfs/joint_pose_reid/aux_vi.txt`
- `tmp/pdfs/paformer.txt`
- `tmp/pdfs/pgfl_kd.txt`
- `experiments/exp370_pbsr/literature_novelty.md`
- `experiments/exp371_casd/literature_novelty.md`
- `experiments/exp372_pcar/literature_novelty.md`
- `experiments/paper_notes/end_to_end_pose_reid_novelty_audit_20260715.md`
