# exp378 独立红队设计审查：bootstrap 后单任务 ReID 的自适应姿态预测

## 审查范围与结论

本审查只评估以下候选，不审查尚未形成的代码：从 Swin 第 3 个 stage（代码索引 2，
输出约为 `24×8×384`）接轻量高分辨率 refinement head，预测 target person 的 17 个
heatmap 与逐关节置信度；训练前段用离线 ViTPose teacher 做 bootstrap，随后 P0 完全关闭
teacher 与 pose loss，只由 ReID objective 更新姿态 head，并在推理时完全依赖内部预测。

**红队裁决：有条件 GO，但朴素版本当前不能直接开跑。** 它至少比“长期 pose+ReID
多任务”更清楚：teacher 只是初始化工具，正式方法阶段是单任务 ReID。不过，直接做
“D0 始终蒸馏且阻断 ReID 梯度”与“P0 关闭蒸馏且开放 ReID 梯度”仍同时改变两个变量；
仅凭这两个 arm 不能把差异归因于 ReID 对 pose head 的任务适配。正式结论至少需要一个
bootstrap 后同时关闭两类梯度的冻结 arm，最好形成 `2×2` 梯度因子对照。

此外，exp374 已证明现有 PSG 的正确实例 pose 与严格 matched 错 pose 几乎等价。exp378
只有在 learned pose **恢复 correct-image correspondence 的可测因果价值，同时没有退化成
隐藏身份纹理的任意 attention map** 时才有研究意义。只涨 mAP、不通过语义与泄漏门禁，
不能写成“task-adaptive pose”。

## 一、启动前的阻塞性科学问题

### 1. Stage-2 预测与多层 PSG 存在因果循环

若 heatmap 由 stage index 2 的输出预测，它在单次前向中只能影响后续 stage index 3；不能
再回头调制 stage 0/1/2。若为了复用“PSG 层数越多越好”而进行第二次 backbone forward，
会同时改变计算量、drop-path 随机性和优化路径，实验不再是最小单变量。

**硬要求**：exp378 Gate A 只能使用 stage-3-only PSG。stage 0/1/2 不读 external 或
student pose；head 在 stage index 2 完整结束后生成 pose，随后只供 stage index 3 的 PSG
使用。两遍 backbone 或预测后回灌早期 stage 均禁止在本实验加入。

### 2. D0 与 P0 不是单变量

拟议的两个 arm 实际同时改变：

1. bootstrap 后是否继续 teacher pose loss；
2. ReID loss 是否能进入 pose head。

因此 `P0-D0` 混合了“去掉语义锚”与“加入身份梯度”的作用。若 P0 更好，既可能是任务
适配，也可能只是停止了一个有害的 pseudo-label loss；若 P0 更差，也无法知道是身份梯度
有害还是失去 teacher anchor 有害。

最小正式因子设计应为：

| arm | bootstrap 后 pose loss | bootstrap 后 ReID→pose head | 作用 |
|---|---:|---:|---|
| `F0` | OFF | OFF | 相同 bootstrap 后冻结 head；隔离普通初始化收益 |
| `D0` | ON | OFF | 传统 teacher-distilled pose；用户指定的语义锚定对照 |
| `P0` | OFF | ON | 本方法：teacher 关闭，单任务 ReID 自适应 |
| `J0` | ON | ON | 常规长期 joint training；分离“有无 teacher anchor” |

两台 GPU 可以先同时筛查 P0/D0，但跨机结果只能看趋势，不能作上述因子归因。P0 出现燃料
后，必须在同一台 4090、同一运行时、相同初始化与 sampler 顺序补 F0/J0；否则最多写成
“两种训练配方比较”，不能写 pose adaptation 因果结论。

### 3. ReID-only pose 很容易变成身份 side channel

P0 在 teacher 关闭后没有任何约束要求 17 张图继续表示关节。stage-2 feature 已含衣服、颜色、
相机与身份信息，ReID loss 可以把这些图变成：

- 高频身份水印；
- 17 通道任意空间 code；
- 由 17 个置信度标量承载的身份 code；
- 与人体关键点无关的普通 foreground/attention map。

这不属于实现 bug，而是目标函数允许的最优解。仅看 heatmap 可视化或 teacher MSE 不足以排除。
如果最终 descriptor 上涨但语义重渲染后收益消失，结论只能是“bootstrap 初始化的可学习
spatial controller”，不能称为姿态。

### 4. ViTPose 输出是 pseudo-label，不是姿态真值

Occluded-Duke 没有当前流程可直接使用的 GT 关键点。teacher agreement 只能说明 student
是否复制 ViTPose，不能证明 student 的姿态更准确。低置信与遮挡关节恰好是 teacher 最可能
错误的位置，强制长期拟合会把错误固化；完全取消约束又会产生上一节的 semantic drift。

因此文档中必须使用“teacher agreement / pseudo-PCK”，不能把它写成 pose accuracy。
bootstrap loss只在 teacher 高置信关节上监督空间分布；低置信关节监督置信度为低，但不强迫
一个错误峰位置。teacher target 必须来自 `person index 0`，不能使用 scene max-merge。

### 5. target assignment 与增强链必须先审计

当前 `PoseImageDataset` 会同步 resize、flip、pad/crop，并在随机擦除或人工遮挡区域把对应
heatmap、score、visibility 清零，这是可复用的正确基础。但启动前仍必须逐 batch 证明：

- RGB 与 person-0 teacher heatmap 的 resize/flip/crop 坐标严格一致；
- horizontal flip 后同时完成空间翻转和 COCO 左右关节通道交换；
- erase/occlusion 后 image 与 teacher confidence/heatmap 同步更新；
- `target_person_idx` 的生成只依赖几何 targetness，不读 pid、camid、文件名身份 token；
- train/query/gallery 的 teacher cache 没有跨 split、同 ID prototype 或检索标签信息。

任一条不成立都可能把 target assignment 或缓存泄漏误当成端到端 pose 收益。

## 二、最小单变量实现建议

### 2.1 不使用完整 HRNet

完整 HRNet 会引入第二套高分辨率 backbone、显著参数/FLOPs/显存、独立预训练和更强的图像
特征路径；它既违反当前固定 Swin-Tiny 的约束，也使增益无法归因于“pose 适配”。**Gate A
禁止完整 HRNet。**

可使用一个轻量的“高分辨率 refinement decoder”，但不要把它宣称为 HRNet 创新：

```text
F2: B×384×24×8
 -> LayerNorm/1×1 Conv, 384→96
 -> bilinear ×2 + depthwise 3×3 + pointwise 1×1
 -> bilinear ×2 + depthwise 3×3 + pointwise 1×1
 -> 17 heatmap logits, 96×32
GAP(F2) -> Linear -> 17 confidence logits
```

第一版不接 stage-1 lateral、不加多尺度 fusion、deformable conv、HRNet branch、GCN、PAA、
LGPA、额外 keypoint token 或完整 pose backbone。建议参数量控制在约 `0.5M` 内，并强制
所有 ReID 路径只能读取最终 17 张图与 17 个置信度，不能读取 decoder hidden feature。

### 2.2 输出瓶颈

为降低身份 side channel，teacher 与 student 都应分解为逐关节空间形状与置信度：

```text
c_T[k] = clamp(max(H_T[k]), 0, 1)
p_T[k] = H_T[k] / max(c_T[k], eps)
p_S[k] = spatial_softmax(logit[k] / tau)
c_S[k] = sigmoid(conf_logit[k])
H_S[k] = c_S[k] * p_S[k] / max(p_S[k])
```

这样重构 teacher 时仍可得到其原始峰值量级，同时显式监督 shape 与 confidence。head 不得输出
额外 residual map、ID token 或未受审计的通道。为进一步隔离，Gate A 建议让 head 读取
`stop_gradient(F2)`：bootstrap 的 pose loss 与后续 P0 的 ReID loss只更新 head 参数，不经
pose 支路第二次更新共享 stage-2 backbone。Swin 主干仍由标准 ReID 主路径端到端更新，head
也确实由 ReID objective 更新；本 Gate 精确检验的是“pose predictor 参数是否值得由 ReID
自适应”。若成立，再在独立实验中开放 pose 支路到 F2，不能在 exp378 中混入。

### 2.3 固定 schedule

禁止复现 exp010 的 backbone 冻结—解冻。只改变 pose source 与 pose-head loss：

1. **epoch 1–5**：stage-3 PSG 使用 external teacher；head 在 `F2.detach()` 上拟合 teacher；
   ReID 正常训练整个主干，ReID 梯度不进入 head。
2. **epoch 6–10**：PSG 输入按预注册线性系数从 teacher 平滑切到 `H_S.detach()`；仍进行
   bootstrap pose loss，ReID 梯度仍不进入 head。epoch 10 结束时已经是 100% student，避免
   关闭 teacher 时硬切换。
3. **epoch 11–120**：所有 arm 的 PSG 都只使用 student；P0 的 pose loss精确为零且 teacher
   tensor不再参与任何 forward/loss，唯一 objective 是现有 ReID ID+triplet；F0/D0/J0按表中
   梯度因子执行。

schedule 必须由 epoch 固定，不能根据验证集 mAP临时延长 bootstrap。P0 在 epoch 11 后即使
dataloader 为并行对照仍携带 teacher tensor，也必须通过“任意改写 teacher tensor，P0 的
descriptor 与 loss逐元素不变”的测试证明 teacher 真正关闭。

## 三、必须报告的 loss 与梯度语义

bootstrap 可使用：

```text
L_boot = sum_k 1[c_T[k]>=thr] * KL(p_T[k] || p_S[k])
       + lambda_conf * BCE(c_S, c_T)
       + lambda_eq * L_flip_equivariance
```

这里 `L_flip_equivariance` 只在 bootstrap 使用；不能在 P0 epoch 11 后继续留一个“很小的
pose regularizer”，否则“后期唯一 ReID objective”不成立。P0 后期总 loss 必须与 direct
ReID control完全同构，只是 descriptor 的生成读取 `H_S`。

首次 CUDA preflight 必须逐 arm 验证：

- bootstrap：head 只有 `L_boot` 的有限非零梯度；
- P0 e11：`dL_reid/d(theta_head)` 有限非零，pose loss 未构建；
- D0 e11：ReID 到 head 的梯度精确为零，teacher loss到 head 的梯度有限非零；
- F0 e11：两类梯度均精确为零；
- J0 e11：两类梯度均有限非零；
- P0/D0/F0/J0 在 bootstrap 结束前 shared state dict、forward source与 blend 系数一致。

只打印 loss 权重不算通过，必须在同一 batch 做 backward、optimizer step 与参数 delta 审计。

## 四、必要对照与最小执行顺序

### 4.1 性能对照

1. `B0`：exact-commit clean Swin-Tiny global-only；
2. `R0`：external target teacher + stage-3-only PSG，全程固定；
3. `F0/D0/P0/J0`：上述 `2×2` 对照，完全相同 head、初始化、bootstrap、student forward；
4. P0 通过首 seed 后再加 `N0`：相同 head/参数/forward，但没有 teacher bootstrap，只从
   epoch 11 的 ReID-only状态开始，判断收益是否只是普通可学习 spatial head。

首轮两机可以只跑 P0/D0作趋势，但不得用跨机差值下正式 GO。若 P0 连趋势都不优于 D0/R0，
直接止损；若有燃料，再同机补 F0/J0/B0/R0。

### 4.2 冻结 checkpoint 干预

P0 checkpoint 至少评测：

- correct student heatmap；
- target-matched 跨图 student heatmap；
- joint-channel permutation；
- canonical pose；
- true bypass；
- 从 student heatmap 提取 `soft-argmax坐标+协方差+置信度` 后用固定 Gaussian **语义重渲染**；
- 重渲染后的 matched-shuffle。

correct-vs-shuffle 只证明图像对应信息有用，仍可能是身份泄漏；correct-rendered-vs-
rendered-shuffle 才能支持信息主要经过关节几何瓶颈。descriptor 必须确实变化，correct-start/end
必须复现，所有 donor 匹配需按 exp378 的 student heatmap重新建立，不能继承 exp374。

## 五、身份泄漏与语义漂移审计

以下不是可选可视化，而是方法归属门禁：

1. **teacher agreement（只审计、不训练）**：高置信关节 pseudo-PCK、normalized coordinate
   error、heatmap cosine、confidence calibration，从 e10 到 e120 记录漂移；不得称 GT pose。
2. **几何等变性**：同图水平翻转、resize/crop 后，将输出逆变换并做左右关节交换；比较 P0
   与 D0/F0。P0 若只对纹理敏感，等变误差会明显增大。
3. **heatmap-only ID probe**：冻结模型，用 train ID 的图像划分训练/验证 probe；分别用
   raw 17-map、坐标/协方差/置信度和 Gaussian 重渲染图训练相同容量 probe。P0 raw probe
   暴涨而 coordinate/rendered probe 不涨，是高频身份 code 的直接警报。
4. **检索描述符重渲染保真**：把 raw student map替换为 Gaussian re-render；若 P0 相对
   F0/D0 的主要 mAP 收益消失，不能称 pose adaptation。
5. **置信度 side channel**：分别评测 correct spatial shape + shuffled confidence、shuffled
   shape + correct confidence；若 17 个 confidence 单独携带主要 ID 增益，应冻结置信 head 或
   将方法降级为 generic controller。
6. **颜色/纹理干预**：保持几何结构的强 color/style 改变不应像几何翻转那样大幅改变关节
   后验。该审计只用于辨别泄漏，不计入主性能。

若没有真实 keypoint annotation，不能证明 P0 比 teacher “姿态更准”；可证明的最多是：
在可测 teacher agreement和几何等变性没有灾难性丢失的前提下，ReID objective 选择了更有
检索效用的低带宽关节后验。

## 六、预注册 kill gates

### Gate 0：实现与执行

任一项失败不得启动正式训练：

- 默认开关关闭时与 exact B0 的 state dict、descriptor、final featmap逐元素一致；
- head 生成后只影响 stage index 3，单次 backbone forward；
- P0/D0/F0/J0 初始化逐键一致，模块初始化不扰动 backbone/classifier RNG；
- batch64 AMP 下 forward/backward/step finite，不改 batch size；
- 上述 arm-specific 梯度与 teacher-off 干预测试全部 PASS；
- target-only 与增强几何审计 PASS；
- unique OUTPUT_DIR、唯一 main、完整 loss/gradient/heatmap统计日志。

### Gate 1：epoch 10 bootstrap 质量

阈值必须在训练前固定；建议首版至少满足：

- teacher confidence `>=0.3` 的关节，student-vs-teacher normalized coordinate error
  `<=0.05` 图像对角线；
-有效关节 heatmap cosine均值 `>=0.85`；
- confidence 对 teacher visibility 的 AUROC `>=0.75`；
-同 checkpoint 的 100% student-mode mAP 相对 teacher-mode 下降不超过 `1.0` 百分点；
-无全零/全一 confidence、单点/全均匀 heatmap collapse。

任一失败则正式 NO-GO；不得临时延长 teacher bootstrap、增大 head 或换完整 HRNet救场。

### Gate 2：epoch 60 燃料

同机 exact controls 下，P0 至少同时满足：

- `P0-F0 >= +0.5 mAP`，证明 ReID→head 适配相对相同 bootstrap/freeze 有效；
- `P0-D0 >= +0.5 mAP`，证明优于持续复制 teacher 的传统方案；
- 不低于 fixed external `R0` 超过 `0.3 mAP`；
- frozen checkpoint 的 `correct-rendered - rendered-matched >= +0.3 mAP`；
- 语义重渲染至少保留 raw P0相对 F0 的 `70%` mAP 增益；
- pseudo-PCK 相对 e10下降不超过 `10` 个百分点，flip-equivariance error 不比 D0恶化
  `25%` 以上。

不满足即 NO-GO，不跑更多 seed、完整 HRNet或参数小变体。跨机 P0/D0只能触发“明显无趋势”
的提前止损，不能触发正式 GO。

### Gate 3：e120 正式 GO

首 seed 至少要求：

- `P0-R0 >= +0.8 mAP`；
- `P0-F0 >= +0.5 mAP`、`P0-D0 >= +0.5 mAP`；
- J0 结果能说明收益来自“关闭 teacher 的单任务适配”，而不是普通 joint loss；
- correct/matched、语义重渲染、confidence split、ID probe、teacher agreement与等变性门禁
  全部通过。

随后才补同机三 seed paired delta、跨数据集与推理成本。最终必须同时报告内部 head 的延迟、
参数/FLOPs和 external ViTPose teacher 的离线/在线成本；不能把预计算 teacher 的时间隐去后
声称无代价端到端。

## 七、可写与不可写的结论

即使成功，也不能写：

- 首次端到端联合 pose 与 ReID；
- 首次让 ReID loss 更新 pose branch；
- freeze/unfreeze 本身是创新；
- pseudo-teacher agreement 表示真实 pose accuracy；
- 使用 HRNet-style decoder 就构成方法创新。

只有全部门禁通过后，候选贡献才可收敛为：

> 以外部姿态仅作短暂 bootstrap，随后移除 teacher 与 pose objective，让一个受低带宽关节
> 后验约束的内部 predictor 在单任务 ReID 下自适应；并通过冻结、持续蒸馏、长期 joint、
> generic head、matched-shuffle与语义重渲染证明收益来自可解释的图像对应关节控制，而非
> 多任务正则、额外容量或身份 side channel。

这仍需专项文献查新后才能声称新颖。若 raw map 有增益但 Gaussian 语义重渲染失败，应诚实
改称“pose-bootstrapped spatial controller”，并重新做 generic attention 强对照；不得继续
包装为 task-adaptive pose。

## 最终红队建议

允许主 agent 进入实现，但只能实现上述轻量 decoder、stage-3-only 单次 forward、固定
bootstrap schedule 与 P0/D0/F0/J0 梯度语义。完整 HRNet、更多 PSG stage、长期 P0 pose
loss、额外 transport/SSM/PAA/LGPA、动态延长 bootstrap 和运行中救场调参均应留到 exp378
之外。正式训练前必须完成 Gate 0；首轮两机 P0/D0 只是快速筛查，不能替代同机因子对照。
