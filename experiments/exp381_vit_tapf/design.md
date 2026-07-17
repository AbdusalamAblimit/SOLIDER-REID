# 实验 exp381：ViT-B/16 上的 TAPF 原子方法与逐层判别

## 动机

exp378/379 在 Swin-T 上得到完整 `anchor+PSG` 相对 B0 的正结果，但逐层 HT0 相对 D0
近中性；exp380 在 ResNet-50 上得到 D0−B0=`+3.1 mAP`、HT0−D0=`+0.8 mAP`。
因此已经可以把 `anchor+PSG` 作为一个完整训练期姿态监督、测试期 pose-free 的原子方法讨论，
但“逐层 refinement 是否是架构无关贡献”仍缺少第三种骨干证据。

本实验选择标准 TransReID ViT-B/16。它既不同于分层 Swin，也不同于卷积 ResNet；仓库已有
Occluded-Duke 配置与 ImageNet 权重来源。实验只比较同一 ViT 内部的 B0→D0→HT0，不把
ViT、Swin、ResNet 的绝对指标横向解释成方法增益。

## 核心假设

1. **完整方法假设**：ViT-D0 相对 ViT-B0 提升，说明训练期 target-person pose 监督形成的
   内部 anchor 与后继 PSG 在第三种骨干上仍有价值，且测试期不需要姿态模型。
2. **逐层假设**：若 ViT-HT0 相对 ViT-D0 也稳定为正，则可把 progressive hierarchical
   pose distillation 提升为核心贡献候选；若近中性或为负，则逐层版只能作为
   backbone-conditional 扩展，论文中心回到完整单 anchor+PSG。
3. **解释边界**：本实验不重新拆开 anchor 与 PSG 归因，也不声称精确关节通道语义是收益来源；
   exp378 的置换与语义审计已经限制了这类主张。

## 骨干与分组定义

- 骨干：`vit_base_patch16_224_TransReID`，12 个等宽 Transformer block，hidden dim=768。
- 输入：`256×128`，patch=`16×16`，stride=`16×16`，得到 `16×8=128` 个 patch token，另有
  1 个 CLS token。
- 预训练权重：`jx_vit_base_p16_224-80ecf9dd.pth`，4090 已找到只读源
  `/home/afr/reid-clean/weights/jx_vit_base_p16_224-80ecf9dd.pth`，大小 `346292833` bytes，
  SHA256=`80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`。
- ViT 没有天然 stage。为避免按实现方便任意切层，固定按深度等分为四个连续 block group：
  `G0=[0,1,2]`、`G1=[3,4,5]`、`G2=[6,7,8]`、`G3=[9,10,11]`。
- anchor 只读取 patch token reshape 后的 `B×768×16×8` 特征；CLS token 不进入 pose decoder。
- PSG 只调制 patch token，CLS token逐位原样旁路，再与 patch token拼回。这样不会把空间场错误地
  广播到非空间 CLS token。

## 技术方案

### ViT-B0

- 使用与 D0/HT0 相同的 exp381 专用 ViT wrapper、预训练加载、BNNeck、classifier、数据、loss、
  optimizer 和训练日程。
- 不创建、不读取 pose 数据，不启用 anchor、PSG 或 hierarchical 模块。
- 该 arm 是同骨干锚点，不复用旧 exp333 数字作为直接对照。

### ViT-D0：完整单 anchor+PSG 原子方法

- block 0–8 正常前向。
- 在 block 8 后，从 patch map 预测单一 TAPF anchor/field。
- 同一个 field 进入后继 PSG bank：分别在 block 9、10、11 后调制 patch token。
- 训练期使用 target-person ViTPose heatmap/confidence 监督 anchor；hard transition、geometry
  residual OFF；测试期不索引 external pose。

### ViT-HT0：每个 anchor 对应一个后继 PSG bank

- anchor-1：block 5 后读取 patch map；field-1 进入 block 6、7、8 的 PSG bank。
- anchor-2：block 8 后读取已经由 field-1 调制过的 patch map，显式 refinement 前一层 state；
  field-2 进入 block 9、10、11 的 PSG bank。
- 两层使用 stage-specific projection 与 shared decoder；每个 anchor 只控制自己的后继 bank，
  不跨组复用 PSG 参数。
- HT0 的 `G3` PSG bank 必须先构造，使其初始化与 D0 的 `G3` bank exact matched；新增 `G2`
  bank 是 HT0 唯一额外 consumer。

## 实现边界

1. 新增独立、默认关闭的 exp381 ViT TAPF 开关；默认 config 和现有 Swin/ResNet 路径不变。
2. 不能直接复用 `PoseBackboneModel`：该类手工迭代 `self.base.stages`，是 Swin 专用。
3. 当前 `build_transformer` 也按 Swin 的 `init_weights/num_features[-1]` 接口构造，不能把旧 ViT
   config 视为已经可运行。exp381 wrapper 必须显式调用 TransReID factory 与 `load_param`，并
   通过测试证明加载数量、positional embedding resize 和输出形态正确。
4. 三臂先构造完全相同的 ViT/BNNeck/classifier 公共部分，再构造额外模块；构造结束恢复公共部分
   后的 RNG state，保证 state/init/RNG/data 顺序 matched。
5. 不启用 JPM、SIE、pose loss 以外辅助 loss、geometry residual、adapter、GCN、LGPA、PAA、
   pose dropout、joint permutation、re-ranking 或其他 test-time trick。

## 对照与固定设置

| 项目 | ViT-B0 | ViT-D0 | ViT-HT0 |
|---|---|---|---|
| ViT-B/16 预训练、输入与 stride | 相同 | 相同 | 相同 |
| seed / batch / epoch | 1234 / 64 / 120 | 1234 / 64 / 120 | 1234 / 64 / 120 |
| anchor source | 无 | block 8 | block 5、8 |
| 后继 PSG bank | 无 | blocks 9–11 | blocks 6–8、9–11 |
| 每 anchor 独立后继 bank | 不适用 | 是 | 是 |
| geometry residual | OFF | OFF | OFF |
| 测试期 external pose | 不读 | 不读 | 不读 |

三臂必须 fresh 串行 B0→D0→HT0，使用独立 repo/config/output；前一 arm 跑满并完成进程、GPU、
12 checkpoints、SHA、参数轨迹、异常和 pose-free parity 终审后，才允许启动下一 arm。

## 启动前门禁

1. CPU unit：block 分组、CLS 旁路、patch reshape、每 anchor→PSG bank 路由、默认行为不变。
2. 权重门禁：4090 原生 PyTorch1.13.1 下 ViT 权重严格统计，所有预期 backbone key 成功加载；
   不接受静默大面积漏载。
3. matched 门禁：B0/D0/HT0 公共 ViT、BNNeck、classifier state exact；构造后 RNG exact；
   optimizer 中公共参数的 group/order/hyperparameter exact。
4. D0/HT0 shared `G3` PSG bank init exact；HT0 的两个 anchor 与两个后继 bank 路由确实不同且
   不能被二次覆盖。
5. 真实数据 batch64 e1/e11 CUDA/AMP：loss、field、confidence 全部有限；e11 继续 student field；
   梯度归属符合设计，各 anchor、projection、PSG bank 与 ViT 都有预期更新。
6. 两次 legacy ViT-B0 多步 parity；真实 AMP overflow 必须整步跳过，model/optimizer state 不变、
   scale 正确下降。
7. eval `correct/shuffle/None/exploding` external pose descriptor exact parity；任何 pose 索引即失败。
8. exact execution commit、full-history bundle、weight/config SHA 和 output 不存在门禁全部固化；
   GPU 非空闲或任一门禁未通过时不得启动训练。

## 记录与裁决

- 每 10 epoch 记录 `mAP/R1/R5/R10`。
- D0 每个 eval 现场计算相对同 epoch B0 的四项显式差值；HT0 同理相对 D0。
- 不因单一 epoch、早期负值或一个门槛提前结束已启动 arm，固定使用 e120 final，不挑 best
  checkpoint。
- final 后才更新 `results.md / decisions.md / innovation_brainstorm.md / story.md`。

## 风险与失败解释

1. **ViT-B 显存风险**：batch64 可能高于现有卡容量。先做真实 batch64 forward/backward/AMP
   门禁；不能通过时不得私自改 batch，需缩减实现临时内存而非改变实验协议。
2. **平坦 ViT 的层级人为性**：四等分是基于 12 层深度的预注册分组，不把它描述成 ViT 天然
   stage；若 HT0 失败，只能说明这一定义未支持逐层贡献。
3. **预训练加载风险**：当前通用 transformer wrapper 偏向 Swin。若加载统计或 B0 legacy parity
   不通过，先修复专用 wrapper 与门禁，不启动训练。
4. **结果边界**：单 seed ViT 正结果仍是跨架构描述性证据，不自动等于统计显著；多 seed 是否补充
   在三骨干排序闭合后另行决定。
5. **后续边界**：ViT 三臂闭合前不进入 Video ReID；闭合后才设计时序姿态可靠性、运动连续性和
   遮挡恢复对照。

## 跑后实现边界与裁决补记

- 预注册方案写为post-block9/10/11 PSG；跑后梯度与checkpoint审计确认post-block11位于最后一次
  CLS–patch交互之后，其final zero projection全轨迹保持`0/2 changed`，对最终CLS descriptor无
  下游路径。实际有效G3 consumer为post-block9/10，论文不得把block11记作有效层。
- 该terminal冗余在D0/HT0间共享，因此不混淆HT0新增G2 bank的直接对照；已完成arm不修改、不重跑。
- final B0/D0/HT0=`52.9/59.5/77.1/82.0`、`54.9/61.4/78.9/84.0`、
  `54.6/60.6/78.4/84.1`。完整原子方法D0−B0为四项正差；逐层HT0−D0未提供稳定增益，按预注册
  风险解释降级为backbone-conditional扩展。
