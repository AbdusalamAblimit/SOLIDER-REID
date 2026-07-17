# Hierarchical TAPF 独立设计：可空分离的逐层姿态状态创新消费者

> 状态：仅设计，不实现、不训练、不触发旧H0。该文档建立在2026-07-17冻结语义审计之后，
> 不是对现有PSG做多层复制。

## 动机

exp378单锚点TAPF已经证明两件不同的事：

1. 训练期由teacher bootstrap、推理期完全不读取外部pose的内生场可以稳定工作；
2. 当前Stage-3 PSG的检索收益不依赖该场的图像对应、关节名称、confidence或空间结构。

D0 e90冻结审计中，matched wrong field、joint/confidence permutation、spatial constant与zero
field相对correct的mAP绝对差全部小于`0.1`个百分点，而PSG bypass下降`2.6829 mAP`。anchor本身
仍有teacher posterior cosine=`0.8276`、pseudo-PCK@0.05=`0.5539`和flip cosine=`0.9467`。
因此瓶颈不是“没有pose-like中间变量”，而是**consumer可以在不使用姿态语义的情况下靠自身容量
获得收益**。

现有PSG先计算`sigmoid(raw_field)`，所以raw zero变成常量`0.5`；随后有bias的encoder仍能产生
非零门控。zero field不是identity，模块容量与field因果被混在一起。若直接在多个stage复制PSG，
只会把同一混淆扩大，不能形成逐层姿态方法。

## 核心假设

真正可检验的Hierarchical TAPF不应消费每层“绝对热图”，而应消费跨层姿态状态的**创新量**：
当前视觉层相对上一层姿态先验新确认了什么、修正了什么。若姿态状态没有变化、被替换为null，或
consumer被置于静态输入，视觉更新必须具有可证明的相应退化。

核心假设分为两层：

1. 多尺度视觉层提供互补的姿态证据，跨层状态创新比单个Stage-2绝对场更能表达遮挡恢复；
2. 只有把null identity和容量匹配写进算子，才能区分“姿态状态被使用”与“多了一个可学习模块”。

若训练后correct state仍不优于matched/static state，假设即失败；不能再用热图可视化、teacher
agreement或模块整体bypass收益替代因果证据。

## 技术方案

### 1. 共享的概率姿态状态

第`l`个视觉层维护：

```text
S_l = {P_l, c_l, μ_l, σ_l}
```

- `P_l ∈ R^(B×17×H×W)`：每关节空间和为1的posterior；
- `c_l ∈ [0,1]^(B×17)`：可靠性；
- `μ_l, σ_l`：由posterior矩确定的有界坐标与尺度，只作状态摘要；
- 不允许额外raw residual、高频自由通道或identity loss进入状态。

所有stage共享主anchor decoder；每个stage只保留输入通道投影和轻量归一化。共享设计的目的不是
节省参数本身，而是防止每层独立head暗藏任意身份码。状态预测输入继续`detach`视觉feature，ReID
loss不得更新anchor；pose loss只更新共享anchor与stage projection，不回流共享backbone。

### 2. 逐层状态更新

将上一层状态重采样到当前统一网格，当前层只预测有界更新：

```text
Δ_l = U_l(stopgrad(F_l), stopgrad(S_{l-1}))
μ_l = clip(μ_{l-1} + r_l · tanh(Δμ_l), 0, 1)
log σ_l = clip(log σ_{l-1} + q_l · tanh(Δlogσ_l))
c_l = reliability_update(c_{l-1}, Δc_l)
P_l = Render(μ_l, σ_l)
```

`r_l/q_l`为预注册的小上限，不在运行中调参。与已失败的单锚点geometry residual不同，这里的更新
不直接接受ReID梯度，也不把最终`17×4`适配器当作主贡献；它只定义跨层状态递进，并接受持续teacher
监督。若专项查新显示该更新与已有multi-stage pose refinement重合，则状态更新只作为实现基础，
不能单独声称新颖。

### 3. 可空分离的创新消费者

consumer不读取绝对`P_l`，而读取同一编码器对“新状态”和“上一状态先验”的差：

```text
I_l = E_l(c_l ⊙ P_l) - E_l(c_{l-1} ⊙ Up(P_{l-1}))
ΔF_l = α_l · Q_l(F_l) ⊙ I_l
F'_l = F_l + ΔF_l
```

其中`E_l`两次调用严格共享参数。另定义显式null路径：

```text
Consumer(F_l, NULL, NULL) = F_l
```

实现必须直接短路或保证两项逐位相减；任何bias、normalization running state或AMP舍入都不能破坏
identity。`α_l`零初始化只用于稳定启动，不能作为因果保证。正式preflight必须在CPU fp32、CUDA
fp32和CUDA AMP下逐位验证null identity及RNG不变。

创新量设计带来一个可证伪性质：若跨层状态没有修正，`I_l=0`，consumer不能靠静态bias产生收益。
若真实状态产生收益，matched field、stage permutation或spatial constant应破坏`I_l`与当前RGB的
对应，并在冻结checkpoint上产生可分辨检索差。

### 4. 层级位置

Swin-T初始设计只允许三个状态节点：Stage-1输出、Stage-2输出、Stage-3输入前。Stage-0不插入
高分辨率consumer，避免计算量与浅层纹理混淆。每个节点共享状态定义，不能使用三套独立PSG。

ResNet-50和Video ReID映射只保留为后续：

- ResNet：`layer1→layer2→layer3→layer4`的相同状态接口；
- Video：每帧先做通过单图门禁的层级更新，再用关节可靠性与运动连续性定义跨帧状态创新；
- 在Swin单图因果门禁通过前，不实现这两项迁移。

## 对照组

所有训练arm固定Swin-T、Occluded-Duke、seed 1234、batch64、120epoch、同初始化/RNG、参数量、
optimizer groups和持续pose supervision。每个arm独立OUTPUT_DIR，禁止并行4090训练。

1. `HB0`：exact exp378 B0，无pose head、无consumer；
2. `H-POSE`：完整共享状态更新与创新consumer；
3. `H-STATIC`：保留并训练全部pose head/consumer参数，但consumer始终读取冻结、显式、非零的
   canonical/static状态序列；用于隔离活跃模块容量，不能用null代替；
4. `H-BYPASS`：保留相同pose监督、状态head和state dict，但视觉consumer逐位旁路；隔离aux课程；
5. `H-NO-PROGRESS`：每层独立从当前feature预测状态，禁止读取上一状态；只在H-POSE通过首轮因果
   门禁后运行，用于回答跨层传递而非多尺度监督是否必要。

冻结checkpoint的配对干预必须包含：

- correct-start/end与external pose sentinel exact parity；
- 对每个stage单独做matched-wrong-state、joint/confidence permutation、spatial constant、null；
- 固定stage order permutation与整段状态序列time/stage reversal；
- 单stage consumer bypass及全部consumer bypass；
- 每个干预记录state创新`I_l`、consumer输出`ΔF_l`和最终descriptor digest，证明干预确实传到视觉路径。

## 门禁

### Gate H-A：只验证实现，不启动训练

- config关闭时相对exact B0 state/init/RNG/descriptor逐位相同；
- `H-POSE/H-STATIC/H-BYPASS`参数key、数量、optimizer groups与初始共享权重逐位匹配；
- null consumer在CPU/CUDA/AMP下逐位identity，且不消费额外RNG；
- pose loss只更新anchor/stage projection，ReID loss只更新consumer与原ReID参数；
- batch64 e1/e11、10-step legacy parity、真实GradScaler overflow与hook恢复全部PASS；
- 默认行为不变，所有新功能由独立config开关控制。

Gate H-A完成前，不创建正式OUTPUT_DIR，不启动训练。

### Gate H-B：单seed训练与冻结因果同时成立

必须同时满足：

1. `H-POSE final − H-STATIC final >= +0.5 mAP`，且R1不低于`-0.2`；
2. `H-POSE final − H-BYPASS final >= +0.5 mAP`；
3. 同一H-POSE checkpoint下，correct相对matched-wrong-state或spatial-constant至少一项
   `>= +0.3 mAP`，且不是只靠单个CMC点；
4. 至少两个stage的独立干预各产生可分辨descriptor与检索差，排除只有一个stage在工作；
5. null identity、参数状态、异常与AMP审计全程PASS。

训练曲线任何单epoch都不能替代final；一个性能门槛失败也不能提前宣称整个TAPF问题永久失败，
但若冻结因果门禁失败，则当前Hierarchical consumer立即NO-GO，不补多seed/ResNet/Video。

### Gate H-C：仅在H-B通过后

1. 三seed Swin-T复现；
2. ResNet-50同机制、同强控制复现；
3. 再设计Video ReID时序姿态状态，与逐帧H-POSE、普通temporal pooling、外部pose smoothing、
   RGB-only video backbone做强对照。

## 预期结果

理想结果不是仅让`H-POSE`高于HB0，而是同时满足：`H-POSE > H-STATIC/H-BYPASS`，并且冻结
correct state明显优于matched/constant。只有这组证据才能把收益归因于逐层状态创新，而不是辅助
姿态课程或consumer容量。

若成立，论文对象才有资格描述为“可靠性有界、跨层递进、可空分离的内生姿态状态”，并进一步
检验其backbone与时序迁移性。B类潜力仍需专项查新和多seed支撑，不能由单seed门禁直接宣布。

## 风险与失败解释

1. **H-POSE≈H-STATIC但都优于HB0**：仍是活跃consumer容量，不是姿态贡献；停止；
2. **H-POSE≈H-BYPASS但都优于HB0**：收益来自pose auxiliary课程或随机波动；停止；
3. **训练涨点但冻结干预不敏感**：重复当前单锚点问题；停止，不能迁移；
4. **只有单stage敏感**：说明hierarchical对象不成立，最多退化为单位置模块；不得包装成逐层方法；
5. **状态像pose但检索不敏感**：teacher agreement/可视化只作诊断，不救场；
6. **null路径不严格identity**：实现阻塞，必须先修复，禁止通过训练观察“是否自己学会不用bias”；
7. **与已有hierarchical pose refinement高度重合**：即使有效也需重新定位问题/证据贡献，不能声称
   机制首创；
8. **Video有效但单图门禁失败**：优先解释为普通时序容量，不得归因于姿态状态。

## 当前执行边界

- 本文档只完成设计预注册；不创建config、不改生产代码、不启动H0/H-POSE训练；
- 下一步是针对hierarchical pose estimation、multi-stage pose-guided ReID、recurrent pose
  refinement、null-separable modulation与video pose-ReID做独立查新，再做代码可行性审查；
- 明确禁止Claude；审查只使用Codex与可复核的本地/远端证据；
- 现有D0/J0/R0/RG0/N0与语义审计资产只读，禁止重启、续训或改写。
