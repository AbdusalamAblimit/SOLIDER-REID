# 实验 exp379：Progressive Hierarchical TAPF（逐层内生姿态场）

> 状态：设计预注册。本文以用户在2026-07-17确认的论文口径为准：anchor与PSG是一个不可拆的
> 完整方法单元；核心价值是把原始PSG对测试期外部姿态热图的依赖，改为训练期姿态监督、测试期
> RGB-only。exp378冻结语义审计继续约束机理措辞，但不把完整模块相对B0的`+1.1 mAP`判无效。

## 动机

原始PSG本质上是`RGB feature + external heatmap → spatial gate`，训练和测试都需要额外姿态模型。
exp378的fresh同机D0把它改成了完整的内生模块：Swin Stage-2生成pose-like场，Stage-3 PSG消费该场，
训练期持续使用ViTPose监督，测试期完全不读取外部pose。D0 final=`56.2/67.6/79.8/83.4`，相对
clean B0=`55.1/66.7/79.5/83.8`为`+1.1/+0.9/+0.3/-0.4`；它也基本复现了测试期依赖外部
ViTPose的R0=`56.1/67.4/79.5/83.7`。

这一结果足以把`anchor+PSG`作为原子方法讨论，不要求anchor和PSG分别贡献独立mAP。冻结语义审计
只说明当前Stage-3消费者对精确关节名称、confidence和空间结构不敏感，因此论文不能声称“每个关节
通道在推理时具有独立因果贡献”；但“训练期姿态监督、测试期姿态模型可移除”的部署事实与整体收益
仍然成立。

下一步不再只在单个Stage-2→Stage-3接口生成一次场，而是检验一条真正的逐层闭环：较浅视觉层先
产生初始结构场并调制下一层，较深视觉层在上一状态上继续修正，再调制更深层。历史exp009已经表明
“把同一个外部热图复制到多个PSG stage”没有额外收益，因此exp379不能退化成旧multi-stage PSG；
不同层必须消费由对应视觉层产生、并通过共享状态递进得到的内部场。

## 核心假设

1. Stage-1保留较高空间分辨率，适合建立粗粒度人体结构支持；Stage-2具有更强语义，适合在上一姿态
   状态上修正遮挡和不确定关节；
2. `Stage-1 field → Stage-2视觉更新 → Stage-2 refined field → Stage-3视觉更新`比单次
   `Stage-2 field → Stage-3`更充分地把训练期姿态监督写入身份特征；
3. 共享姿态解码器加stage-specific轻量投影，比每层独立pose head更接近一个可迁移的模块接口，
   也便于映射到ResNet层级和分组ViT blocks；
4. 在Swin-T上增益可能受强SOLIDER backbone饱和限制，因此Swin首轮必须跑满并诚实报告，但中性
   结果不自动取消后续一次预注册的ResNet-50/ViT迁移检验。

## 技术方案

### 1. 两个内生状态节点

首版固定source stages=`[1,2]`，consumer stages=`[2,3]`：

```text
Stage-1 feature F1 ─→ state S1 ─→ Stage-2 PSG
                                  ↓
Stage-2 feature F2 + stopgrad(S1) ─→ refined state S2 ─→ Stage-3 PSG
```

- `F1`和`F2`分别是Swin Stage-1/2的归一化pre-downsample输出；
- 每个stage先用独立`1×1 Conv + GroupNorm + SiLU`投影到共同64维；
- 两个stage严格共享同一个LiteHR-style spatial/confidence decoder；
- 不实例化两套完整anchor，不增加geometry residual，不启用exp378已判无独立收益的adapter。

### 2. 递进姿态状态

共享decoder在Stage-1输出`L1`和`q1`：

```text
P1 = softmax(L1)
c1 = sigmoid(q1)
S1 = {P1, c1}
```

Stage-2 decoder输出的是对上一状态的修正证据`R2/r2`，而不是第二套互不相关的绝对场：

```text
P2 = softmax(log(stopgrad(P1) + eps) + R2)
c2 = sigmoid(logit(stopgrad(c1)) + r2)
S2 = {P2, c2}
```

`P1/P2`经与exp378相同的diagonal moments与Gaussian renderer形成`17×96×32` raw field，仍只由
PSG内部执行一次sigmoid。上一状态进入refinement时detach，Stage-2 pose loss不会经prior反向改写
Stage-1实例路径；两层通过共享decoder参数发生结构共享。

### 3. 逐层消费者

- `field1`只进入Stage-2每个block后的现有非spatial PSG；
- `field2`只进入Stage-3每个block后的同类PSG；
- Stage-2完成后必须替换当前field，Stage-3不得继续误用`field1`；
- eval时不读取、索引或验证任何external `pose_dict`；correct/shuffle/None/unindexable必须descriptor
  exact parity；
- exp379把anchor、递进状态和两级PSG视为一个原子模块。冻结field干预只用于理解机制，不作为
  “每个子部件必须单独涨点”的论文成立条件。

### 4. 训练课程与损失

沿用fresh D0的部署一致课程与持续pose监督：

- epoch 1–5：两个consumer均用external target-person teacher field；两个内部节点各自接受pose监督；
- epoch 6–10：两个节点分别按同一student fraction从teacher平滑交接到对应student field；
- epoch 11–120：Stage-2/3只消费内部`field1/field2`，但两个节点继续接受teacher pose loss；
- eval：从第一次评测起始终predicted-only；
- 总pose loss固定为两个stage pose loss的算术平均，避免仅因节点数量把辅助loss权重翻倍；
- pose输入feature detach，ReID loss不得更新projection/shared decoder；pose loss只更新projection和
  shared decoder，不回流Swin；PSG与Swin只接受标准ID/triplet ReID目标。

## 对照组

### 首轮必须比较

1. `B0`：exp378同机clean baseline，`55.1/66.7/79.5/83.8`；
2. `D0`：exp378同机单点完整模块，`56.2/67.6/79.8/83.4`，是exp379唯一直接升级对照；
3. `HT0`：本文两节点逐层版本，source=`[1,2]`、consumer=`[2,3]`、持续pose监督、无geometry residual。

B0/D0已经完整结束并审计，禁止重复训练。通过默认关闭路径的state/init/RNG/optimizer/forward exact
parity证明新commit没有改变旧D0后，HT0可直接复用其同机逐epoch轨迹作为历史matched参考。

### 结果为正后再补

- `HT-NOPRIOR`：Stage-2独立预测field2，不读取S1，隔离“逐层传递”与“双层监督/双层PSG”；
- `HT-S3ONLY`：保留共享两节点pose监督，但只让field2进入Stage-3，隔离Stage-2视觉consumer；
- parameter/FLOPs与训练、测试期外部姿态模型开销表。

历史exp009 external multi-stage PSG只作背景证据，不能替代上述matched消融，也不能与exp379直接做
数字差值。

## 预注册门禁

### Gate A：实现与单变量门禁

正式训练前必须全部通过：

1. `POSE_TAPF_HIERARCHICAL=False`时，旧B0/D0的state keys、参数、构造后RNG、optimizer groups、
   train/eval descriptor与loss逐位不变；
2. HT0只新增stage projections、一个共享decoder和Stage-2 PSG；共享decoder在object/state层面只有
   一份，不能为两个stage复制权重；
3. field1/field2均为有限`B×17×96×32`，Stage-2与Stage-3 hook证明分别只收到对应field；
4. 两个pose loss均有效，总loss严格等于其均值；Stage-2 refinement实际读取一次且仅一次S1；
5. pose loss梯度只到projections/shared decoder；ReID loss只到Swin/PSG/classifier，不到pose head；
6. eval external correct/shuffle/None/unindexable exact parity；
7. CPU单元、PyTorch1.13.1 CUDA batch64 AMP e1/e11、10-step legacy parity、真实overflow缩放与
   NaN/Inf/Traceback/RuntimeError/OOM门禁全部PASS；
8. batch固定64、seed1234、120epoch、`RE_PROB=0`、独立OUTPUT_DIR，禁止续训、并行或手工跳epoch。

### Gate B：Swin-T完整训练判断

- 不因任何单个epoch或门槛提前停止，必须跑到final并记录每10 epoch的mAP/R1/R5/R10；
- primary升级差值是`HT0−D0`，总方法差值`HT0−B0`只作辅助；
- `HT0−D0 >= +0.5 mAP`：逐层升级有清晰单seed燃料，进入matched消融和多seed；
- `+0.2～+0.4 mAP`：只记为弱正pilot，先看checkpoint波动与ResNet/ViT迁移，不宣称升级成立；
- `<=0 mAP`：Swin上逐层升级无收益，但考虑SOLIDER Swin-T饱和，仍允许按预注册方案完成一次较弱
  backbone迁移；不能临场调层数、loss weight或hidden width救Swin结果。

## Backbone迁移顺序

exp379 final闭合后另立独立实验，不在HT0运行中改config：

1. 优先ResNet-50：映射`layer2→layer3`两个source与`layer3→layer4`两个consumer；同backbone内必须
   同时训练RGB-only B0、单点D0和逐层HT0，不能把Swin数字跨backbone比较；
2. 再选一个ViT基线：按block depth分组为中层/深层两个source-consumer接口，保持同一共享状态定义；
3. 迁移目标不仅是“弱backbone涨得更多”，还要检验方法排序`HT0 ≥ D0 > B0`是否跨结构保持；
4. backbone迁移完成前不宣称backbone-agnostic。

## 预期结果

成功时，方法可描述为：用训练期姿态监督建立跨视觉层递进的内部结构场，使原本测试期需要ViTPose
热图的PSG变成RGB-only部署模块。贡献对象是完整的Progressive Hierarchical TAPF，而不是anchor、
PSG或某个关节通道各自的独立涨点。

若Swin增益有限而ResNet/ViT更明显，最合理解释是强SOLIDER Swin-T对人体结构先验已有较高吸收度，
而逐层模块在较弱/不同归纳偏置backbone上更有空间；该解释必须由同backbone三臂对照支持，不能只靠
跨backbone绝对数字。

## 风险与失败解释

1. **HT0≈D0**：逐层状态没有超出单点模块；仍保留D0的部署价值，不把HT0写成贡献；
2. **HT0<D0**：浅层PSG干扰特征形成，和历史multi-stage PSG一致；停止Swin小变体，转预注册backbone
   迁移判断是否为Swin饱和/结构特例；
3. **pose loss下降但检索不涨**：层级姿态预测学会，不等于身份特征受益；不能用可视化替代指标；
4. **只有更多参数带来收益**：HT0可以作为完整工程模块，但主论文需要HT-NOPRIOR/HT-S3ONLY或
   parameter-matched control收紧机制；
5. **冻结field干预仍不敏感**：继续限制精确姿态因果措辞，但只要整体方法、多seed和跨backbone稳定，
   不因此删除“训练期姿态监督、推理期RGB-only”的模块级结论；
6. **ViT接口不自然**：先完成ResNet；不为了凑迁移强行修改ViT token结构。

## Video ReID后续边界

单图与backbone迁移闭合后，再独立设计Video ReID：把每帧的层级内部场扩展为轨迹内状态，利用关节
可靠性、运动连续性和跨帧遮挡恢复；强对照至少包括普通temporal pooling、逐帧D0/HT0、外部pose
smoothing与RGB-only video backbone。当前exp379不实现时序模块，不抢跑视频训练。
