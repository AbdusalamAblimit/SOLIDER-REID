# 实验 exp378：TAPF 可靠性有界任务自适应姿态场

## 动机

exp374–377 反复得到同一组边界：外部姿态驱动的 PSG 相对同 checkpoint bypass 有稳定收益，
但正确实例姿态与 matched-shuffle 几乎等价；把同一固定 heatmap 改成路由、动态低秩算子或
selective SSM 也没有产生可报告增益。一个尚未被验证的共同瓶颈是：ReID 始终被迫消费为
通用人体姿态估计设计的固定 ViTPose heatmap，而不能决定遮挡检索真正需要保留或移动的结构场。

朴素的“pose estimator 初始化后用 ReID loss 自由微调”不能作为本实验的创新点。ECCV 2018
PABR 已训练过面向 ReID 的内部 part-map extractor，PAFormer 等工作也覆盖了训练期 pose
heatmap 监督、测试期不依赖外部 pose 的路线。因此 exp378 不采用自由微调 pose head，而把
问题收缩为：**在保持可见关节解剖语义的有界几何域内，只释放不确定关节的有限几何量给 ReID 单任务
目标适配，能否产生超出普通内部 pose predictor 的检索收益？**

## 核心假设

通用 pose teacher 对清晰关节通常可靠，此时自由微调容易把 17 个通道退化成无语义 attention；
对被遮挡、截断或低置信关节，固定 teacher 又可能不是 ReID 的最佳结构先验。若把内部预测场
分成冻结的姿态锚点与受约束的身份残差，则应同时满足：

1. 最终推理完全不读取外部 pose；
2. 高可靠关节基本保留 teacher 解剖位置，中等置信/不确定关节允许更强的 ReID 适配；
3. 适配残差本身而不是普通内部 pose predictor 带来收益；
4. 关节通道、可靠性或残差对应被破坏时产生可测退化；
5. residual-off 后仍得到可识别的 pose-like anchor，而不是任意身份 attention。

## 技术方案

### 1. 轻量高分辨率姿态锚点

在 Swin Stage-2 的 `24×8×384` 特征上接一个 LiteHR-style head，而不是再运行完整 HRNet：

- `1×1 Conv 384→64 + GroupNorm + SiLU`；
- 一个高分辨率 depthwise residual 分支；
- 一个 `12×4` context 分支，上采样后与高分辨率分支融合；
- `1×1 Conv 64→17`，再双线性上采样到 `96×32`。

这里把“轻量 HRNet”分成两个层级。Gate A 先使用上述约 `8.6 万` 参数的最小
LiteHR-style anchor，避免把更强姿态网络容量与 TAPF 机制混为一谈；若其 bootstrap 质量门禁
明确失败，可另立预注册 `H0`，把 anchor 单变量替换为从同一 Stage-2 特征出发的轻量
Lite-HRNet 多分辨率 decoder。`H0` 不得读取原图建立第二套 backbone，输出仍只能是
`17` 个关节场与置信度，后续 `17×4` 几何写入空间、课程和 ReID loss 必须完全不变。
因此轻量 HRNet 是可检验的 anchor 容量对照，不是 TAPF 的创新 claim，也不能在 P0 机制已经
判负后被当作临场调参救场。

记输出为姿态锚点 `A(F2)`。它只在前 `E_boot=10` 个 epoch 用 target-person ViTPose
heatmap 做 bootstrap；pose loss 的输入特征 detach，不让辅助监督重写 Swin backbone。
空间 shape target 取 person-0 heatmap；置信度 target 必须取同一 person-0 的
`scores * person_mask`，禁止用 heatmap peak 偷换 reliability 定义。
epoch 10 后锚点输出对 ReID 路径 detach，锚点 head 参数不再接收任何 loss。需要诚实区分：
Stage-2 主干仍由正常 ReID 主路径训练，因此固定 head 在变化的 `F2` 上输出的锚点函数仍可能
间接漂移；e10→e120 的坐标、方差、置信度漂移必须报告，不能写成“锚点场绝对冻结”。后期不是长期
pose+ReID 多任务，也不是 PABR 式自由微调整个 pose extractor。

### 2. 低带宽可靠性有界几何残差

不允许 ReID head 直接输出任意 `17×96×32` residual map。先把锚点的每个关节通道归一化成
空间分布，提取 `mu_x/mu_y/sigma_x/sigma_y` 与独立置信度 `c_j`；另一个共享轻量 MLP 只从
锚点加权的 Stage-2 关节特征预测四个有界几何残差：

```text
release_j = (1 - stopgrad(c_j))^gamma
mu'_j     = mu_j + release_j * max_shift * tanh(delta_mu_j)
logsigma'_j = logsigma_j
              + release_j * max_log_scale * tanh(delta_logsigma_j)
H_adapt_j = c_j * Gaussian(mu'_j, sigma'_j)
```

- 最终 ReID 控制器每图只有 `17×4` 个可变几何量，不能在高频像素图中藏任意身份 code；
- `release_j` 使高可信关节的几何残差更小；但最终 raw field 还乘 `c_j`，实际有效调制约为
  `c_j(1-c_j)^gamma`，零置信关节同样几乎不起作用，通常由中等置信/中等不确定关节获得最大修正；
- 位移和尺度都有预注册上限，坐标、方差和置信度均可直接审计；
- Stage-2 controller 输入全程 `stop_gradient`。Swin 主干仍由标准 ReID 主路径训练，首版只隔离
  “controller 参数是否值得由 ReID 适配”；若成功，开放 controller→Stage-2 梯度必须另立实验；
- epoch 10 后 P0 的几何 residual 只通过标准 ReID ID/triplet loss 和 PSG 路径学习，不加
  pose、part、matching、GCN、PAA、Mamba 或额外身份分类头。

最终 `H_adapt` 只注入 Stage-3 两个 Swin block 的现有 PSG。Stage 0–2 不注入姿态，避免在
内部预测场形成之前引入先后依赖。Gaussian 重渲染不是事后可视化，而是正式 forward 的信息
瓶颈；任意 raw map residual 不进入 exp378。

现有 PSG 会对输入 raw field 统一执行一次 sigmoid。离线 ViTPose teacher（允许小负值）与
student Gaussian 都在该 sigmoid **之前**交接；空间 softmax 只用于 bootstrap shape loss 和
几何矩提取，绝不直接作为 PSG 输入。启动审计必须同时记录 teacher/anchor/adapted 的 raw
min/max/mean/peak 与 PSG-sigmoid 后统计，禁止出现二次 sigmoid。

### 3. 单任务交接课程

- epoch 1–5：Stage-3 PSG 使用外部 teacher；只训练 detached-input anchor 的重建；
- epoch 6–10：teacher 到内部 Gaussian anchor 线性 handoff，继续 bootstrap，但 ReID 仍不进入
  anchor 或几何 residual；epoch 10 已是 100% student，禁止后续硬切换；
- epoch 11–120：Stage-3 PSG 只使用内部场；P0 的 teacher tensor 与 pose loss 精确关闭，唯一
  目标是标准 ReID loss；
- eval：从第一次评测起始终强制内部 predicted-only，禁止因 epoch 较早而偷用 teacher；
- forward 在 eval 时传入真实、shuffle 或 `None` 的 `pose_dict` 必须产生逐元素相同 descriptor。

bootstrap loss 只是一段初始化过程，不能作为论文贡献；论文方法对象是后期的可靠性有界单任务适配。
“后期只有 ReID 单任务”确实区别于 PAFormer 式持续 pose 多任务，但 PABR 已经覆盖 pose 初始化后
ReID-only 自适应，所以单任务设定本身仍不是新颖性。可争的新意必须落在：冻结姿态锚点参数、
可靠性决定释放量、ReID 只能写入有界 `17×4` 几何子空间，以及这些约束的成套反事实证据。

## 对照组

### Gate A：首轮筛查

1. `B0`：clean Swin-Tiny global-only，无 PSG、无内部 pose head；
2. `R0`：原始 Stage-3 PSG，始终使用外部 target-person ViTPose heatmap；
3. `F0`：bootstrap 后 pose loss OFF、ReID residual OFF，冻结内部 Gaussian anchor；
4. `D0`：bootstrap 后 pose loss ON、ReID residual OFF，持续复制 teacher；
5. `P0`：bootstrap 后 pose loss OFF、ReID residual ON，完整可靠性有界几何适配；
6. `J0`：bootstrap 后 pose loss ON、ReID residual ON，常规长期 joint control。

首轮 4090 跑 P0，3090 跑 D0，仅作趋势筛查；跨机差值不作正式结论。P0 有燃料后在同一 4090、
同一 exact commit、同一运行时补 B0/R0/F0/D0/J0，形成 pose-loss × ReID-residual 的 `2×2`。

### Gate B：不可省略的归因控制

若 P0 通过性能筛查，追加：

- `U0`：PABR-like 自由微调 anchor，去掉冻结/可靠性/低带宽几何约束；
- `G0`：参数匹配 generic 17-channel spatial controller，不做 pose bootstrap；
- residual-off、release-uniform、release-permuted、joint-channel permutation；
- predicted anchor / adapted field 与 teacher 的峰值距离、flip equivariance、熵、通道占用；
- 高/低 teacher confidence 分组的 residual RMS，要求低置信组显著更大；
- 外部 `pose_dict=correct/shuffle/None` 的 descriptor exact parity。

另预注册三个正式归因 control，在 P0 首轮有燃料后实现：

- `N0`：以 residual-OFF hard F0 为唯一直接对照，bootstrap期间把target-person teacher heatmap
  与joint confidence按同一个固定17通道无不动点置换重标号；保留每张图的姿态空间支持、置信度
  多重集、anchor/renderer/PSG、课程与优化配方，只破坏正确解剖通道语义。N0不启用geometry
  residual，避免重新引入已被2×2判为无mAP贡献的变量；完整固定定义见
  `n0_permutation_design.md`；
- `C0`：训练时统一 release，不使用 confidence gating，隔离可靠性函数的贡献；
- `RG0`：固定 external teacher 先提取同样的 `mu/sigma/confidence` 并经同一 Gaussian renderer，
  隔离 R0 raw ViTPose 与 student Gaussian 的 renderer/数值域差异。
- `H0`：仅在最小 anchor 未通过 bootstrap 质量门禁时启用的轻量 Lite-HRNet decoder 容量对照；
  不改变 TAPF 写入空间、课程或 loss，不用于挽救已经明确为负的几何适配机制。

只有 P0 同时优于 F0、D0、U0、G0/N0、C0，并与 RG0 公平比较，且上述机制审计成立，才能把
收益归给可靠性有界的几何适配。

### RG0 固定实现与验收边界（R0 完成后冻结）

RG0 与 R0 使用相同的 external target-person ViTPose heatmap、person-0 score/mask、Stage-3 PSG、
训练配方、seed 与 120-epoch 日程。唯一核心变量是送入 PSG 前的 field 表示：

- R0：target-person raw heatmap 直接进入现有 PSG，PSG 内执行一次 sigmoid；
- RG0：同一 raw heatmap 先做 `positive clamp → spatial mass normalization → diagonal moments →
  sigma clamp → confidence × peak-normalized Gaussian`，再进入同一 PSG，仍只由 PSG 执行一次
  sigmoid；
- RG0 固定 `sigma_min=0.025`、`sigma_max=0.25`，与 TAPF renderer 相同；不实例化 anchor、
  geometry adapter、handoff 或 pose loss，不能把 RG0 解释为 internal/external source 对照；
- raw heatmap 空间正质量为零但 target confidence 非零时必须显式失败，禁止静默在左上角产生
  伪 Gaussian；person mask 为零且 confidence 为零的空 target 合法并渲染为全零场。
- RG0 只接受真实 `pose_dict` 中成对的 target heatmap/score；现有 audit-only
  `scene_heatmaps_override` 没有 score 语义，因此在 RG0 明确禁用，不能把一个 raw override
  误当成已 Gaussian 化的 final field。

启动 RG0 前必须同时通过：默认 raw R0 的 state/init/RNG/forward 路径逐位不变；R0 与 RG0
初始模型参数及 optimizer parameter groups 逐位一致；RG0 renderer 与 TAPF 的 teacher posterior、
moments 和 renderer 在有效输入上 exact 或严格浮点容差一致；person-0 heatmap 与
`scores[:,0] × person_mask[:,0]` 对齐；输出固定为 float32 `B×17×96×32`、全有限且在 `[0,1]`；
hook 证明 PSG 只执行一次 sigmoid；batch64 CUDA AMP forward/backward、PSG gradient/update 与显存
门禁通过。训练日志必须记录 raw/rendered min、max、joint peak、mean、负值比例、score 越界比例、
positive mass、confidence、mu、sigma 范围/上下界命中率与 rendered peak-confidence 误差；全量
数据扫描另记录真正的 global min/max 和 inconsistent-empty 计数，不能用 batch meter 代替。

## 预注册门禁

- batch 固定 64，Swin-Tiny，seed 1234，120 epochs，标准 768-d global descriptor；
- exp378 全部 arm 的普通 Random Erasing 固定关闭（`RE_PROB=0`）：现有 erase 链不能保证同步
  清除 teacher heatmap，不能让 RGB/teacher 几何错位进入首轮；B0/R0/F0/D0/P0/J0 必须一致；
- 完整记录每次 eval 的 mAP/R1/R5/R10；`<e60` 不作性能负裁决；
- 启动前必须通过 CPU 单元、CUDA batch64 AMP、梯度归属与 predicted-only parity；
- bootstrap 期必须证明 anchor loss 下降，且 pose loss 不向 backbone 传播；
- handoff 后必须证明 anchor 无梯度、residual 与 Stage-2 有有限非零 ReID 梯度；
- e60 若 P0 比预注册 clean B0 低 `>=0.5 mAP`，或 P0 相对 D0 没有任何正趋势，判首轮
  NO-GO；不得用层数、宽度、loss weight 小调参救场；
- 正式性能门槛：同机 `P0-B0 >= +0.8 mAP`、`P0-F0 >= +0.5 mAP`、
  `P0-D0 >= +0.5 mAP`、`P0-U0 >= +0.3 mAP`，且不弱于 R0 超过常见单 seed 波动；
- 正式机制门槛：residual-off 与 release-permuted 均退化 `>=0.3 mAP`；报告低/中/高 confidence
  三组的几何 residual 与乘 `c_j` 后有效调制，不预设零置信组最大；predicted anchor 保持关节
  语义与 flip equivariance；外部 pose 三臂 exact parity；对 68 维几何量做同容量 ID probe，
  若身份可分性异常升高则不得称为 pose adaptation；
- 首 seed 通过后再补同机多 seed。单 seed 只作探索证据。

## 预期结果

成功时，exp378 支撑的不是“端到端 pose ReID 首创”，而是：冻结的通用姿态语义与 ReID
自适应不应自由混合；通过关节可靠性定义硬边界，只让有限关节均值与对角尺度发生受限变化，
可以得到测试期无外部 pose 输入、仍具解剖可解释性的身份表征。

失败时，结论只否定当前 Stage-2 LiteHR anchor + Stage-3 PSG + 可靠性低带宽几何 residual 的实现；
它不能否定所有端到端 pose-ReID。只有失败被明确定位为 anchor bootstrap 质量不足时，才允许按
预注册 `H0` 测轻量 Lite-HRNet；若 anchor 已合格而 P0 几何适配无效，则不得用更深 head 或更长
bootstrap 临场救场。

## 风险与失败解释

1. teacher heatmap 数值域与 PSG 内部 sigmoid 不一致：实现必须监督 PSG 实际消费前的同一数值域，
   并记录 teacher/anchor/adapted 的 min/max/peak；
2. anchor 在 10 epoch 内学不稳：先看重建、峰值距离与 eval predicted-only，而不是盲目延长课程；
3. 68 个连续几何量仍可能编码身份：以 Gaussian 正式重渲染、可靠性分组、geometry-only ID
   probe、generic controller 和通道置换共同审计；低带宽只降低风险，不等于自动排除泄漏；
4. 早期 teacher 训练、eval predicted-only 可能造成曲线偏低：这是有意的部署一致性压力，不能在
   eval 偷用 teacher；
5. 两机运行时不同：跨机只看趋势，正式差值必须同机 exact control。
