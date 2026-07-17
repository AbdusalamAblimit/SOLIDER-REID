# exp378 TAPF 第二轮科学与新颖性审查

> 审查日期：2026-07-16
> 审查对象：`design.md` 中最终收紧的 `17×4` bounded joint geometry residual、
> confidence-conditioned release、Gaussian 正式重渲染、bootstrap 后 anchor 停止接收 loss、
> P0 epoch 11 起仅用标准 ReID loss，以及 F0/D0/P0/J0 因子对照。
> 本审查不批准训练执行，不替代代码/AMP/数据链审查。

## 一、最终裁决

**裁决：方法级差分已经形成，但仍是“有条件 GO”，不是无条件创新通过。**

相对朴素 exp378，TAPF 不再只是：

```text
pose 初始化 -> ReID-only 自由微调一个内部 pose/map head
```

而是：

```text
短期 pose bootstrap
-> anchor 参数停止接收 loss
-> ReID 只能修改每关节 (mu_x, mu_y, sigma_x, sigma_y)
-> 修改半径由冻结梯度的 confidence 决定
-> 最终控制场必须经过 Gaussian 重渲染
```

这确实把 ReID 可写函数空间从任意高维 map 限制到每图 `17×4` 的显式几何变量。它与
PABR 的自由微调整个 pose/part-map extractor、PAFormer 的持续 heatmap MSE + 自由
cross-attention map 均有机制级差异。因此不能再简单裁决为“PABR 现代复现”。

但当前设计仍有五个必须写清或补进门禁的科学问题：

1. **冻结的是 anchor 参数，不是 anchor 函数。** anchor 读取仍在继续训练的 Stage-2
   feature，因此其输出、置信度和 release 半径在 e10 后仍可能随 backbone 漂移；
2. **低置信释放与低置信抑制相互抵消。** 位移上界随 `(1-c)^gamma` 增大，但最终场又乘
   `c`；接近零置信的关节即使位移很大，也几乎不影响 PSG；
3. **`17×4` 是低带宽，不是零泄漏。** 68 个连续实数足以编码 ID，Gaussian 只排除了
   高频像素水印，没有排除“用坐标/尺度编码身份”；
4. **当前不是严格数学意义的 trust-region optimization，也不是质量守恒 transport。**
   它是 confidence-conditioned bounded parameterization；Gaussian 采用峰值归一化时，改变
   sigma 会改变总质量；
5. F0/D0/P0/J0 足以分离“持续 pose loss”和“ReID residual”，但还不足以证明收益来自
   pose bootstrap、confidence-conditioned radius 和 Gaussian geometry 三者，而非相同容量
   的 generic controller。

在这些边界被改成准确 claim，并把下文所列控制预注册后，TAPF 可以进入 Gate 0 与首轮
killswitch；否则即便上涨，也只能叫 pose-bootstrapped bounded spatial controller，不能叫
可信域任务自适应姿态。

---

## 二、是否真正区别 PABR

### 2.1 PABR 已覆盖的上位故事

PABR（ECCV 2018）已经覆盖：

- OpenPose/COCO 预训练 pose 子流初始化；
- ReID 数据上不使用 pose/part 标注；
- 只用 ReID triplet fine-tune pose/part map extractor；
- map 直接控制 appearance representation；
- trainable pose stream 优于 frozen pose stream。

所以 TAPF 仍然不能声称：

- 首次 pose-prior initialization 后 ReID-only adaptation；
- 首次 task-adaptive pose/part map；
- 首次让 ReID loss 改变内部姿态分支；
- 首次无网络外部 pose estimator 的 joint ReID model。

### 2.2 TAPF 的不可归约差分

TAPF 与 PABR 的实质差分不是 Swin、ViTPose 或训练日程，而是**ReID 可写子空间**：

| 维度 | PABR | TAPF P0 |
|---|---|---|
| pose prior | OpenPose 子网权重初始化 | ViTPose heatmap 短期 bootstrap |
| ReID 可更新对象 | 整个 pose/part-map extractor | 仅 geometry adapter |
| 每图可变输出 | 高维 latent part map | 17 个关节的 2-D mean + diagonal scale |
| map 约束 | 无输出空间硬瓶颈 | bounded residual + Gaussian renderer |
| 可靠性作用 | 未定义 joint-wise adaptation radius | `release=(1-c)^gamma` |
| pose anchor | 与 ReID 一起自由更新 | anchor 参数不接 ReID/pose loss（e11+） |
| 归因对照 | fixed vs trainable pose stream | 2×2 pose-loss × ReID-residual + U0/G0 |

只要正式实现严格保持上述边界，TAPF 就不是 PABR 的同构实现。特别是：

> PABR 允许 ReID 任意重写 part-map network；TAPF 只允许 ReID 在一个显式、有限范围的
> 关节几何邻域中移动均值和尺度。

这是可以进入论文方法段的机制差分。但不能说“完全防止身份泄漏”，也不能把 `17×4`、
Gaussian 或 confidence gate 中任何单个组件单独声明为首创。

### 2.3 仍需 PABR-like U0

`U0` 是不可省略的最近邻内部复现，但必须定义得更精确：

- 与 P0 共享 e1–10 bootstrap、sampler、initial state；
- e11 后允许 ReID 自由更新 anchor map，而不是另换一个更宽 decoder；
- 不额外增加 ID head 或 descriptor；
- 使用同一 Stage-3 PSG 和同一 global retrieval path。

如果 U0 更强，说明 bounded geometry 主要是解释性约束而非性能方法；TAPF 仍可讨论稳健性，
但不能声称约束同时带来性能和语义优势。正式门槛 `P0-U0 >= +0.3 mAP` 是合理的，不过应
同时比较语义漂移与 ID leakage，而不是只比较 mAP。

---

## 三、是否真正区别 PAFormer

PAFormer 已覆盖：

- 离线 pose heatmap 监督模型内部空间 attention；
- attention map 聚合 part feature；
- part feature 的 ID/triplet 反向塑造定位 map；
- learned visibility predictor；
- 测试不再输入 pose heatmap。

TAPF 与之有四个实质差分：

1. PAFormer 的 pose MSE 在正式训练中持续存在；P0 e11 后不构建 pose objective；
2. PAFormer attention map 是高维可学习关联；TAPF 的 ReID 可写量只有显式 joint geometry；
3. PAFormer 的定位与 part descriptor/matching绑定；TAPF 只修改标准 global descriptor 的
   backbone 路径；
4. TAPF 将 frozen-parameter anchor 和 ReID residual 分离，并以 F0/D0/P0/J0 解耦
   pose supervision 与 identity adaptation。

因此 TAPF 可以安全说“区别于持续 heatmap-supervised part attention”，不能说“首次训练
heatmap、测试不需要 heatmap”。J0 是 PAFormer/auxiliary-pose 式持续监督的内部控制；只有
P0 同时优于 D0/J0/F0，才有证据说明“关闭 teacher 后的 bounded single-task adaptation”有
独立价值。

---

## 四、相邻通用机制进一步压缩哪些 claim

即使 PABR/PAFormer 没有 TAPF 的完整组合，下列组件已有成熟先例：

- [PoseFix, CVPR 2019](https://arxiv.org/abs/1812.03595)：从初始姿态出发做受图像条件的
  pose refinement；“初始 pose + residual correction”不是新问题；
- [RLE, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.html)：
  pose coordinate distribution/uncertainty modeling 已成熟；
- [ProbPose](https://arxiv.org/abs/2412.02254)：姿态后验与不确定性学习进一步说明
  uncertainty-aware posterior 不能单独当贡献；
- Gaussian heatmap、soft-argmax/integral regression、bounded `tanh` residual、冻结 teacher、
  curriculum 与 confidence gating 都是通用组件。

所以 TAPF 的新颖性只能落在完整联合机制：

> **pose-bootstrap anchor 与 ReID-only residual 的梯度隔离 + confidence-conditioned bounded
> joint geometry + 强制 Gaussian field + 遮挡 ReID 的因果归因。**

当前已核范围内没有发现与该完整联合机制同构的 ReID 先例；这支持“未发现直接先例”，
不支持“世界首次 trust-region pose adaptation”这种绝对表述。

---

## 五、关键科学问题一：anchor 并未在函数意义上冻结

设计写“冻结姿态锚点”，但实际拓扑是：

```text
RGB -> trainable Swin Stage-2 F2_t -> frozen-weight anchor A -> posterior
```

`stop_gradient(F2)` 只阻断 TAPF loss/gradient 回到 backbone，不会冻结 `F2` 的数值。Swin
仍由标准 ReID 主路径持续更新，所以：

```text
A_theta_frozen(F2_e10) != A_theta_frozen(F2_e60)
```

完全可能成立。随之变化的还有：

- anchor mean/scale；
- confidence `c_j`；
- release radius `(1-c_j)^gamma`；
- controller 的 anchor-weighted Stage-2 input。

因此当前可准确声称的是“**anchor 参数冻结、TAPF 梯度与 backbone 隔离**”，不能先声称
“冻结的解剖语义”或“固定姿态后验”。

### 必加门禁

F0 与 P0 都要从 e10 到每个 eval 记录：

1. 同一 frozen audit subset 的 anchor-only mean/scale/confidence；
2. anchor 对 teacher 的 normalized coordinate error、heatmap cosine、pseudo-PCK；
3. e10 anchor 与当前 anchor 的 self-drift；
4. F2 feature norm/distribution drift；
5. adapted field drift减去 anchor drift后的净 residual effect。

若 e60 anchor 自身已严重失去 teacher agreement，就不能把 P0 描述为“在冻结姿态语义附近
适配”；此时最多是“frozen-head over evolving ReID features”。Gate 应明确：即使 P0 mAP
上涨，anchor drift 失败也会触发语义 claim NO-GO。

这不是要求冻结 Swin Stage 0–2；冻结 backbone 会改变 ReID 基线。首轮可以保留当前设计，
但必须准确命名并用 F0/漂移审计量化其局限。

---

## 六、关键科学问题二：低置信 release 与幅值抑制的乘积

当前几何位移上界近似为：

\[
|\Delta\mu_j| \le (1-c_j)^\gamma M,
\]

而正式场为：

\[
H_j=c_j\,G(\mu_j+\Delta\mu_j,\sigma'_j).
\]

因此“位移自由度”随低置信增加，但“对 PSG 的有效影响”又随 `c_j` 减小。粗略的一阶有效
量级与 `c_j(1-c_j)^gamma` 有关：

- `c→1`：release 接近 0；
- `c→0`：field amplitude 也接近 0；
- 真正可能产生作用的是中等置信而非完全不可见关节。

这并非逻辑 bug：完全不可见关节不应凭空获得高影响。但论文不能笼统说“越不可靠的关节
适配越强”或“低置信关节贡献主要收益”。更准确的描述是：

> 高置信关节被锁紧；完全无证据关节被幅值抑制；中等置信但可能错位的关节获得最大可用
> 几何修正空间。

### 当前机制门禁不够

“低置信组 residual RMS 高于高置信组”只证明 delta 较大，不能证明 delta 对正式 field 或
descriptor 有作用。还必须记录：

- confidence-binned `||H_adapt-H_anchor||`；
- confidence-binned descriptor sensitivity / joint residual-off delta；
- near-zero、medium、high 三组，而不是只做 `<0.3` 与 `>=0.7`；
- shift RMS 与 **effective field delta** 同时报告。

若低置信 residual 很大但 field delta≈0，不能用它支持机制故事。

---

## 七、关键科学问题三：低带宽不等于无身份 side channel

`17×4=68` 个有界连续变量足以编码训练 identity。Gaussian renderer 能排除任意
`17×96×32` 高频水印，这是重要优势；但它不能排除：

- 用微小 joint offset 编码衣服/ID；
- 用 34 个 sigma 编码 ID；
- 通过 confidence-dependent allowed range形成相机/身份 code；
- controller 读取 identity-rich `F2` 后把外观信息压缩到几何参数。

所以安全表述是“显著降低并结构化 side-channel capacity”，不是“防止 identity leakage”。

### 必须升级为正式 Gate 的审计

当前 `design.md` 的 Gate B 未把 ID probe 写成硬门禁，应补入：

1. 对 `mu/sigma/confidence` 做 train-ID probe；
2. 相同容量 probe 比较 F0、D0、P0、J0 和 generic G0；
3. P0 的 probe 若相对 F0/D0 暴涨，而 teacher agreement不改善，语义 claim NO-GO；
4. style/color intervention后 geometry 应比 generic spatial controller更稳定；
5. internal matched-donor field replacement必须使用 exp378-specific nuisance matching；
6. correct-vs-matched 的收益还要在 joint geometry/channel controls 下保持，不能只依赖
   confidence scalars。

ID probe 没有唯一普适阈值，不能凭空规定准确率上限；应预注册相对门槛，例如 P0 相对 F0
的 probe 增量不能显著大于其 geometry agreement/检索因果收益所能解释的程度，并同时报告
随机标签和 teacher geometry probe。

---

## 八、“trust region”“守恒”与 Gaussian 的准确措辞

### 8.1 不是严格 trust-region optimization

经典 trust-region method 通常含显式距离预算、局部模型与步长接受/拒绝或投影。TAPF 当前是：

```text
confidence-conditioned tanh bound
```

它没有样本级 KL/Wasserstein budget，也没有优化步接受准则。论文可使用：

- 可信邻域；
- reliability-conditioned bounded geometry；
- confidence-bounded residual field。

不宜把“首次 trust-region optimization for ReID pose”作为 claim。若保留 TAPF 名称中的
“可信域”，方法段必须明确它是参数化邻域，不冒充通用优化算法。

### 8.2 当前 Gaussian 不是质量守恒传输

正式 renderer 把 Gaussian 峰值归一为 1，再乘 confidence。改变 sigma 时空间积分会变化，
因此它不是 mass-preserving/守恒 transport。设计中的“守恒式空间迁移”应删除或改为：

> 受界的均值迁移与尺度重整。

如果改成单位质量 Gaussian，会同时改变 PSG 输入幅值与 teacher handoff，属于新设计，不能
在训练中途临时修改。当前峰值归一化版本本身可接受，但不能使用守恒 claim。

### 8.3 Gaussian 的真正价值

可安全主张：

- 正式 forward 强制所有适配场经过可解释的 mean/diagonal-scale renderer；
- raw map residual 不进入 PSG；
- 因而高频空间 residual 被结构上排除；
- mean/scale/confidence 可逐项干预和审计。

不能主张 Gaussian rendering 自身新颖或彻底消除身份编码。

---

## 九、F0/D0/P0/J0 是否足够

### 9.1 2×2 因子设计本身是强项

四臂可形成清楚的因子解释：

| 对比 | 可回答的问题 |
|---|---|
| `P0-F0` | pose loss OFF 时，ReID geometry residual 是否有用 |
| `J0-D0` | pose loss ON 时，ReID geometry residual 是否有用 |
| `D0-F0` | residual OFF 时，持续 pose supervision 是否有用 |
| `J0-P0` | residual ON 时，持续 pose supervision 是否有用/有害 |

这比 P0/D0 两臂比较科学得多。四臂 e1–10 相同、e11 才分叉也能降低初始化和课程混淆。
设计中的同机 exact comparison、单 seed探索、多 seed后置是合理的。

### 9.2 对“完整 TAPF 优于旧范式”仍不够

要支持 pose bootstrap、confidence radius 和几何瓶颈三项联合归因，至少还应预注册：

1. `N0`：**相同 TAPF architecture**，无正确 pose bootstrap或使用 joint-permuted teacher
   bootstrap；用于隔离 pose prior，而不是换成不同 generic head；
2. `C0`：相同 frozen-parameter anchor + `17×4` Gaussian residual，但训练与推理均使用
   uniform release；用于证明 confidence-conditioned radius，而不是仅做 P0 checkpoint 的
   分布外后处理干预；
3. `RG0`：外部 teacher 先经过与 TAPF 相同的 moments + Gaussian renderer，再进入 PSG；
   用于分离“内部预测/适配”与“Gaussian 化本身”。R0 raw teacher 与 P0 Gaussian field
   数值域不同，不能单独隔离 renderer 贡献；
4. `U0`：前述精确定义的 PABR-like free adaptation；
5. `G0`：参数匹配 generic controller，且必须说明是否仍输出同样 `17×4` 变量、是否共享
   Gaussian renderer。否则“参数匹配”不等于“函数类匹配”。

不要求首轮同时跑满全部控制。可以按预注册止损：P0/D0 无燃料立即停止；P0 有燃料后，
在正式新颖性 GO 前补齐。关键是必须在看到结果前锁定定义与门槛。

### 9.3 frozen checkpoint 干预必须补进主设计

`design_review.md` 已提到、但 `design.md` Gate B 还不够明确的控制包括：

- correct internal adapted field；
- exp378-specific matched-donor adapted field；
- matched-donor anchor + correct residual、correct anchor + donor residual；
- joint-channel permutation；
- confidence correct/shuffle 与 geometry correct/shuffle 分离；
- residual-off；
- canonical/bypass；
- correct-start/end 复现。

只有这些干预能判断收益来自当前图关节几何、anchor、residual还是 confidence side channel。
外部 `pose_dict=correct/shuffle/None` exact parity只证明推理不读外部 pose，不证明内部场有
姿态因果价值。

---

## 十、性能门禁审查

### 10.1 合理部分

- `<e60` 不作负裁决；
- e60 P0 相对 matched B0 低 `>=0.5 mAP` 即止损；
- 正式要求 `P0-B0 >= +0.8`、`P0-F0/D0 >= +0.5`、`P0-U0 >= +0.3`；
- 单 seed 只作探索，正式结论后补同机多 seed；
- 跨 4090/3090只看趋势；
- 不允许用层数、宽度、loss weight 临场救场。

这些门槛与近期实验方差和机制目标相称。

### 10.2 必须修正的基线口径

exp378 所有 arm 固定 `RE_PROB=0`，因此 e60/e120 正式 B0 必须是**同一 exp378 config、同一
运行时、同样关闭 RE 的 B0**。不能直接拿 exp375/377 的历史 B0 数值作正式差值；历史结果
只能作 sanity reference。

R0 使用 raw external teacher，而 TAPF 使用 Gaussianized field。应增加前述 RG0，或明确
`P0-R0` 是系统级比较而非单变量机制比较。

### 10.3 机制门槛需要从 residual 数值升级为有效作用

现有门槛：

- residual-off / release-permuted 退化；
- low-confidence residual RMS > high-confidence；
- anchor 保持语义；
- external pose parity。

还需增加：

- confidence-binned effective field delta；
- matched internal field 因果差；
- anchor drift 门禁；
- ID geometry probe；
- N0/C0/RG0 结果；
- P0 相对 F0 的增益不能只由 17 confidence scalars解释。

否则门禁可能出现“delta 很大但被 `c` 乘没了”或“68 个几何数编码 ID但仍满足 Gaussian”
这样的假阳性。

---

## 十一、安全 claim 与禁止 claim

### 11.1 训练前即可安全描述的方法事实

- TAPF 使用短期外部 pose bootstrap，正式推理不读取外部 pose；
- P0 e11 后不构建 pose objective，唯一训练目标是标准 ReID loss；
- anchor 参数与 ReID residual 的梯度路径分离；
- ReID 只能修改每关节 mean 与 diagonal scale；
- confidence 定义每关节的有界几何邻域；
- 正式控制场必须由 Gaussian renderer生成，raw residual map 不进入 PSG；
- F0/D0/P0/J0 形成 pose-supervision × ReID-residual 的因子对照。

建议统一使用“**无外部姿态推理**”，而不是绝对“pose-free inference”，因为内部 anchor
head 与 geometry controller 仍在推理计算图中。

### 11.2 只有全部 Gate 通过后才能写

- bounded single-task adaptation 优于 frozen pose、持续 distillation、长期 joint、自由
  PABR-like adaptation 与 generic controller；
- 低/中可靠关节的受限几何调整对遮挡 ReID 有独立收益；
- 改进主要经过可解释 joint geometry，而不是额外容量或 confidence/ID side channel；
- 在保持 teacher agreement 与几何等变性不灾难性退化的条件下，ReID utility 提升。

### 11.3 无论结果如何都不能写

- 首次 end-to-end pose + ReID；
- 首次 ReID-only task-adaptive pose；
- 首次训练用 pose、测试不用 pose；
- 首次 pose residual refinement、uncertainty pose、Gaussian heatmap或 trust region；
- `17×4` 完全防止身份泄漏；
- anchor 在函数意义上固定不变；
- teacher agreement 等于真实 pose accuracy；
- 越低 confidence 的关节一定贡献越大；
- 当前 renderer 是质量守恒 feature transport；
- 单 seed或跨机趋势已证明方法成立。

### 11.4 若实验成功，推荐的窄贡献表述

> 我们将姿态先验与身份适配拆成冻结参数的解剖 anchor 和 ReID-only 几何 residual，
> 并把身份目标的可写空间限制为可靠性条件下的关节均值与尺度邻域。通过 Gaussian 正式
> 重渲染、因子化训练对照和内部场干预，验证收益来自受界的图像对应几何控制，而非持续
> pose 多任务监督、自由 part-map 微调或通用空间注意力。

其中“解剖 anchor”“图像对应几何控制”必须由 anchor drift、matched field、ID probe 和
等变性结果支持；若任一失败，应降级为“pose-bootstrapped bounded spatial controller”。

---

## 十二、启动前要求与最终建议

### 必须在设计/门禁中补齐

1. 把“冻结 anchor”统一改成“冻结 anchor 参数”，并加入 e10→e60/e120 anchor drift gate；
2. 把“低置信适配更强”改成“高置信锁紧、零置信抑制、中等置信获得最大有效修正”，并记录
   confidence-binned effective field delta；
3. 删除“守恒式空间迁移”，把 trust region准确写成 bounded geometry neighborhood；
4. 把 `N0/C0/RG0`、ID geometry probe 和 internal matched-field intervention预注册到
   Gate B；
5. 明确 U0/G0 的函数类、参数量、bootstrap 与 schedule，不能只给名称；
6. 正式 B0 固定为 exp378 同机、同 commit、`RE_PROB=0` 的 matched B0；
7. 数值门槛除 shift RMS 外必须包含正式 field/descriptor effect；
8. 预注册实际 `max_shift/max_log_scale/gamma/sigma bounds` 及归一化坐标单位；当前 config
   `0.12/0.5/1.0/[0.025,0.25]` 可以作为首版，但训练中不得改。

### 可以保留

- Stage-2 detached input、Stage-3-only 单次 forward；
- e1–5 teacher、e6–10 handoff、e11 predicted-only；
- P0 e11 后 pose loss精确关闭；
- F0/D0/P0/J0 2×2；
- Gaussian official forward；
- external pose exact parity；
- e60 killswitch 和首 seed 后置 controls；
- RE 全臂关闭及独立 OUTPUT_DIR。

### 最终建议

TAPF 已经有足够明确的机制差分，**允许在上述预注册修订和 Gate 0 全部 PASS 后进入 P0/D0
首轮 killswitch**。不建议因为通用组件有先例而放弃；也不允许仅凭结构看起来合理就宣布
新颖性完成。

真正的论文 GO 条件是一个联合事实：

```text
P0 性能过门
+ 优于 F0/D0/U0/G0/N0/C0
+ anchor 漂移可控
+ internal correct > matched
+ effective geometry 而非 confidence side channel解释收益
+ ID probe/等变性不支持身份编码退化
```

只有这个联合证据成立，TAPF 才能诚实地成为“我们的创新”；否则它仍然是一项执行干净、
有价值但不进入论文主贡献的负结果或诊断结果。
