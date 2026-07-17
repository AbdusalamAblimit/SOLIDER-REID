# 端到端姿态预测与 ReID 联合训练：创新性审计（2026-07-15）

## 结论先行

把姿态预测器放进 ReID、让 ReID loss 反向更新姿态分支，本身不是新的方法对象；
“前若干 epoch 冻结、之后解冻”也只能作为优化策略，不能承担主创新。

已有工作至少覆盖了三种强先例：

1. 预训练 pose 子网络在 ReID 数据上由 ReID loss 端到端微调；
2. 共享 backbone 同时预测 pose 与 ReID，并研究两任务的梯度干扰；
3. pose 分支同时接受 pose loss 与 ID/ReID loss，使其既保持关键点预测能力，又适应
   ReID 目标。

因此，不能再使用“首次将 pose estimator 纳入 ReID 端到端训练”或“首次让姿态适应
ReID”一类表述。

这个想法仍有价值，但需要把问题重新定义为：

> **如何让姿态条件适应遮挡 ReID，同时防止身份梯度把语义姿态漂移成任意的身份注意力。**

当前首选候选是**不确定度有界的守恒特征传输**（暂记 `UBCFT`）：姿态后验只能在
语义信任域内适应 ReID，并驱动一个非对角、守恒的跨 token 证据传输算子。它不是
简单 joint training，也不是固定权重的 pose loss；其核心是从参数化上限制“哪些关节
允许改、允许改多少、能以什么形式改”，并让可学习姿态改变 feature routing，而不是
继续逐点缩放 feature。该候选仍需进一步查新和小规模可证伪验证，当前不能直接声称
新颖。

## 一、直接先例

### 1. PABR：pose 子网络由 ReID loss 直接微调

Suh et al., *Part-Aligned Bilinear Representations for Person Re-identification*, 2018：

- [arXiv](https://arxiv.org/abs/1804.07094)

其双流网络包含 appearance extractor 与由 OpenPose 子网络初始化的 part/pose map
extractor。论文明确说明：

- 整个网络只优化 ReID triplet loss；
- pose 子网络可以固定，也可以与其他参数一起 fine-tune；
- fine-tune pose 子网络优于固定 pose 子网络；
- pose/part 表示会适应 ReID，而不再只是原始关键点预测器的输出。

这已经直接封住“pose estimator 接入 ReID 并由身份目标端到端调整”的宽泛 claim。

### 2. Visual Person Understanding：共享网络联合做 pose 与 ReID，并明确发现语义遗忘

Pfeiffer et al., *Visual Person Understanding through Multi-Task and Multi-Dataset
Learning*, 2019：

- [arXiv](https://arxiv.org/abs/1906.03019)
- [DOI](https://doi.org/10.1007/978-3-030-33676-9_39)

该工作用同一个 ResNet backbone 联合进行 ReID、pose estimation、属性分类与人体
解析，并比较完全共享与末层分支化结构。与当前想法最相关的证据是：

- pose 与 ReID 可以端到端联合训练；
- ReID 往往从辅助任务获益，但 pose 质量会因 ReID 梯度显著下降；
- 仅用 pose 预训练后再 fine-tune ReID，会使 pose 估计几乎失效；
- 作者明确推测：pose-invariant 的 ReID 训练擦除了 backbone 中的 pose 信息；
- 其缓解方案包括复制末层 block、拆分 backbone 输出通道，降低梯度干扰。

因此，“姿态语义漂移/遗忘”不是尚未被观察的问题；若以此为新问题，必须提出比末层
分支化或固定多任务 loss 更明确的新约束。

### 3. Pose auxiliary VI-ReID：pose loss 与 ID loss 同时约束 pose 分支

Miao et al., *On Exploring Pose Estimation as an Auxiliary Learning Task for
Visible-Infrared Person Re-identification*, arXiv 2022 / Neurocomputing 2023：

- [arXiv](https://arxiv.org/abs/2201.03859)
- [DOI](https://doi.org/10.1016/j.neucom.2023.126652)

该工作已经把当前朴素方案推进得更远：

- 明确称 pose estimation 与 VI-ReID 为 end-to-end joint training；
- pose 分支预测关键点 heatmap；
- pseudo ground-truth heatmap 来自 LIP 上预训练的 pose estimator；
- pose 分支同时接受 heatmap pose loss、identity loss 和 hetero-center triplet loss；
- 预测的 pose mask 会逐元素回注/调制 ReID feature，pose feature 也参与最终检索；
- 总损失为 `L_id + beta L_hctri + lambda L_pose + gamma L_KD`；
- 论文动机就是避免盲信固定的 off-the-shelf pose estimator，使 pose feature 适应 ReID，
  同时用 pose loss 保持关键点质量。

这直接说明“joint pose+ID supervision 保留语义并适应 ReID”以及“可训练 pose heatmap
回注 ReID feature”都不能作为我们的方法级主创新。已核的三篇中没有发现“在同一次
训练中先冻结固定 N 个 epoch、再解冻”的完全相同日程；但仅增加这一
freeze-then-unfreeze schedule 仍不足以越过先例。

## 二、当前想法中哪些部分仍可保留

### 可以保留为实现策略

1. warm-up 阶段冻结 pose predictor；
2. 随后只解冻 pose decoder、adapter 或 LoRA，而不是全量 ViTPose-Huge；
3. pose loss 与 ReID loss 联合优化；
4. 用预训练 pose teacher 产生伪标签；
5. 训练时 pose supervision、推理时由网络内部预测 pose。

这些都可能提高稳定性或效率，但不能单独写成贡献。

### 真正需要争取的新对象

朴素 joint training 面临一个定义性矛盾：

- 若只用 ReID loss，heatmap 可以漂移成携带衣服、颜色或身份信息的任意空间编码，
  此时不能继续把它解释为 pose；
- 若用固定强 pose loss 完全钉住 teacher，模型只是复制离线 ViTPose，无法纠正其在遮挡
  和域偏移条件下的错误；
- 若 teacher 和 student 使用同一批伪标签，普通 self-distillation 不产生新的监督信息。

因此可研究的对象不是“是否解冻”，而是**受约束的任务适配自由度**。

## 三、首选候选：可靠性约束的姿态信任域适配

### 3.1 问题定义

冻结的 pose teacher 为第 `k` 个关键点产生 heatmap logits `l_k^T` 与可靠性 `r_k`。
student 不直接任意重写整张 heatmap，而是在受限空间内预测 residual：

\[
\tilde l_k = l_k^T + \rho_k \Delta l_k,
\qquad
\tilde H_k = \operatorname{softmax}(\tilde l_k),
\]

其中 `rho_k` 随 teacher 不可靠性增加：

- 可见且可靠的关键点：`rho_k` 接近 0，保持原始姿态语义；
- 遮挡或不可靠的关键点：`rho_k` 较大，允许 ReID 目标在有限范围内纠偏。

关键要求是 `Delta l_k` 不能成为任意像素图。至少需要采用以下一种结构瓶颈：

1. 只预测关键点均值位移与协方差变化，再解析生成 heatmap；
2. 只预测骨架图上的低维形变系数；
3. 只通过小型 decoder/LoRA 更新，并对位置、尺度、左右翻转等变性施加硬约束；
4. 对可靠关节施加输出空间 trust-region，而不是只靠全局固定权重的 MSE。

这种参数化的目的，是限制 heatmap 的信息带宽，使其无法轻易藏入细粒度身份纹理。

更严格的约束可以直接写成不确定度加权的后验信任域：

\[
\sum_k w_k W_2^2(p_k,q_k)\leq \epsilon,
\]

其中 `q_k` 是冻结 teacher 的关节后验，`p_k` 是可学习 student 后验；teacher 越可靠，
`w_k` 越大，允许的后验位移越小。`W_2`/Wasserstein 不是新意本身，姿态输出分布、
不确定性估计和多任务约束均已有成熟先例；它在这里的作用只是把 semantic preservation
写成可验证的输出空间边界，而不是继续调一个固定 `L_pose` 权重。

### 3.2 训练阶段

建议将 freeze/unfreeze 降级为配套策略：

1. **姿态 warm-up**：冻结 teacher 和 ReID 主干，只训练轻量 pose head 拟合 teacher；
2. **ReID warm-up**：固定 pose head，先让 ReID 主干在稳定 pose 条件下收敛；
3. **受约束适配**：仅释放 pose residual adapter，由 pose semantic constraint 与 ReID
   utility 共同更新；
4. teacher 始终冻结，只作为语义参照，不参与 ReID 反向传播。

不能把第 3 步简化成“到第 N 个 epoch 全量解冻 ViTPose”；否则方法会退化为常规
staged fine-tuning，创新性和稳定性都不足。

### 3.3 与新调制算子的关系

这个适配器不应只服务旧 PSG。旧 PSG 仍属于逐点条件仿射函数族，即使 heatmap 可学习，
也不会因此变成新调制方法。

更合理的组合是：让受约束姿态分布参数化一个非对角的 feature routing/transport
算子，使可见位置的真实视觉证据在姿态结构约束下传递到其他位置。候选形式为：

\[
Y_i=X_i+\lambda\sum_j T_{ij}(\tilde H,X)(VX_j-VX_i).
\]

这里 `T` 必须有明确的守恒、局部性或骨架结构约束；否则它只是普通 attention 或图
消息传递的改名。该 transport 候选尚未完成专项查新，不能直接立项或声称创新。

一个更具体、但仍待排雷的实例是：

\[
K_{ij}=\exp\!\left(
\frac{\langle QX_i,QX_j\rangle+
\sum_{a,b}p_a(i)R_{ab}p_b(j)}{\tau}
\right),
\]

再通过局部邻域 mask 与归一化得到 `T`，执行

\[
Y=X+\lambda(T-I)VX.
\]

它至少在函数上离开 PSG/PAA/FiLM 的逐点 `gamma X + beta`：`j` 位置的真实视觉内容
可以传到 `i`，且常量 feature 场的增量严格为 0。如果进一步使用双随机 `T`，还可
约束全局 feature mass；但 Sinkhorn、non-local attention、GAT 和图扩散均是成熟工具，
不能把“双随机/守恒/消息传递”中的任何一个词单独包装为创新。

### 3.4 先于训练的燃料门禁：检索条件姿态效用

姿态置信度回答“关键点定位是否可靠”，但不回答“该关键点是否对当前身份检索有
增量价值”。在写训练方案之前，应先对冻结 PSG checkpoint 做反事实审计：

\[
u_{l,k}(x)=\Delta\operatorname{margin}
\big(H_{\text{correct},k},H_{\text{counterfactual},k}\big),
\]

其中 counterfactual 至少包括 matched-shuffled、canonical 与 joint-drop。只有某个
stage/joint 的效用下置信界稳定大于 0，才有理由让其进入自适应 pose/transport。

建议的首轮 NO-GO 线是：correct pose 相对最强 counterfactual 的完整检索指标优势低于
`+0.3 mAP`，或 joint/stage utility 排序跨 seed 不稳定，则不启动端到端重训练。该门禁
不是最终论文结论，但能避免在“姿态本身几乎没有因果燃料”的情况下继续堆机制。

## 四、必须设置的判别实验

### 4.1 最小对照

1. 离线/固定 pose teacher + 现有 ReID；
2. pose predictor 从 epoch 0 全量 joint training；
3. 简单 freeze-then-unfreeze；
4. pose+ReID 固定加权多任务 loss（复现直接先例的核心形态）；
5. 只解冻 decoder/adapter；
6. 可靠性约束的姿态信任域适配；
7. 若 transport 通过独立查新，再增加旧 PSG 与新 transport 的同 pose 对照。

### 4.2 不能只看 ReID 指标

至少同时报告：

- ReID：mAP、Rank-1、三 seed paired delta；
- pose：PCK/OKS 或关键点坐标误差；
- semantic drift：适配前后可靠关节的位置/分布漂移；
- equivariance：水平翻转、裁剪、尺度变化下的关键点一致性；
- identity leakage：从 heatmap/pose latent 预测身份的可分性；
- intervention：正确 pose、打乱 pose、固定 canonical pose、完全 bypass pose；
- 参数量、FLOPs、显存与推理延迟。

若 ReID 上涨但 pose 质量严重下降、heatmap 可直接预测身份，结论只能是“学习了辅助
attention”，不能写成 task-adaptive pose。

## 五、工程可行性边界

当前仓库使用离线 ViTPose-Huge `.npz` heatmap，并在 dataset 中同步做 resize、flip、
crop 与遮挡增强。把完整 ViTPose-Huge 直接塞进 batch size 64 的训练图，显存与吞吐
风险很高，也会破坏现有离线 heatmap 的几何增强路径。

更可行的第一版不是全量端到端 ViTPose-Huge，而是：

1. 保留 frozen ViTPose-Huge 作为训练 teacher；
2. 在 ReID backbone 的中间 feature 上接轻量 pose student head；
3. 先做 teacher distillation 和等变性验证；
4. 只给 student 的低维 residual adapter 开放 ReID 梯度；
5. 不改变 batch size，先以单数据集、单 seed killswitch 判断是否值得展开。

这仍会与 2019 shared-backbone pose+ReID 先例相邻，所以最终论文贡献必须落在“可靠性
约束的适配自由度”及其可验证的 semantic-drift 防护，而不是 shared pose head 本身。

## 六、仍需排除的相邻先例

### 6.1 通用训练组件已经成熟

- [PCGrad](https://arxiv.org/abs/2001.06782)、
  [CAGrad](https://arxiv.org/abs/2110.14048)、
  [Nash-MTL](https://arxiv.org/abs/2202.01017) 已覆盖多任务梯度冲突处理；
- [Side-Tuning](https://arxiv.org/abs/1912.13503)、
  [AdaptFormer](https://arxiv.org/abs/2205.13535)、
  [LoRA](https://arxiv.org/abs/2106.09685) 已覆盖冻结主干与轻量适配；
- [RLE](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.html)
  与 [ProbPose](https://arxiv.org/abs/2412.02254) 已覆盖姿态后验/不确定性学习。

因此，gradient surgery、adapter/LoRA、confidence/entropy gate 与 teacher anchoring
都只能作为实现组件，不能单独列贡献。

### 6.2 2025 TIFS 的姿态感知 OT 工作

Lu et al., *Posture-Aware Robust Person Re-Identification via Optimal Transport
Calibration*, TIFS 2025：

- [DOI](https://doi.org/10.1109/TIFS.2025.3622067)

其摘要显示：训练阶段使用 prototype matching 与 posture-aware mixture of experts；
测试阶段把 gallery set 到 query credible set 的排序校准写成 optimal transport。它目前
看来是**检索集合分布之间的测试时 OT**，不是 backbone 内 spatial token 的守恒 feature
transport，也不是端到端 pose posterior 适配。

这意味着它暂未直接同构于 `UBCFT`，但已经封住宽泛的“首次在 pose-aware ReID 中
引入 OT”表述。后续命名和贡献必须明确是训练端、空间 token 级、语义受约束的证据
传输；在全文核清前仍不得声称精确首次。

## 七、当前裁决

| 候选 claim | 裁决 |
|---|---|
| 首次将 pose estimator 放进 ReID 端到端训练 | 不可写 |
| 首次让 ReID loss 微调 pose branch | 不可写 |
| 首次联合 pose loss 与 ID/ReID loss | 不可写 |
| 首次 freeze 若干 epoch 后再解冻 pose | 即使精确组合未检出，也只属于 schedule，不足以当主创新 |
| 观察到 joint training 会造成 pose semantic drift | 问题已有直接观察，不能单独声称新 |
| 对可靠/不可靠关节分配不同的输出空间适配自由度 | 有潜力，但尚需专项查新 |
| 用结构瓶颈防止 heatmap 退化为身份 attention | 有潜力，必须用 leakage/drift 实验证明 |
| 受约束自适应 pose 驱动非对角 feature transport | 当前最值得继续审查的组合候选，尚未通过创新性裁决 |

当前建议：**保留这个方向，但拒绝朴素端到端版本；先完成信任域适配和非对角
transport 的专项查新，再决定是否写实验 design。**
