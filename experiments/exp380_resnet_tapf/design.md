# 实验 exp380：ResNet-50 上的 TAPF 跨骨干验证

## 动机

exp378/379在SOLIDER Swin-T上的matched单seed结果为：B0=`55.1/66.7/79.5/83.8`、
单点D0=`56.2/67.6/79.8/83.4`、逐层HT0=`56.1/67.6/79.9/83.4`。完整
`anchor+PSG`相对B0保留约`+1 mAP`，但HT0相对D0基本中性。Swin-T的行人域预训练与强层级表征
可能压缩了结构监督的增量空间，因此需要在不同归纳偏置、仅ImageNet预训练的ResNet-50上做同骨干
三臂验证。

本实验不比较Swin与ResNet的绝对指标，也不把弱backbone上的更大headroom自动写成方法更强。
只回答：同一个ResNet-50内部，D0是否优于B0，以及每个anchor配置一个PSG的HT0是否优于D0。

## 核心假设

1. **完整模块可迁移**：训练期pose监督学习的内部anchor与PSG组成原子方法，在ResNet-50上仍能
   相对纯RGB B0产生正向检索差值，同时推理期不读取外部pose。
2. **逐层空间可能在ResNet释放**：ResNet的卷积层级归纳偏置不同于SOLIDER Swin-T；浅层anchor→
   中层PSG、深层refined anchor→末层PSG可能比单点D0更有价值。
3. **解释边界不变**：即使D0或HT0上涨，也只首先支持完整pose-supervised、pose-free模块；不据此
   宣称17个精确关节通道在冻结推理时各自具有独立因果贡献。

## 技术方案

### ResNet层级映射

固定ImageNet预训练ResNet-50、`LAST_STRIDE=1`，把四个残差stage映射为：

| stage索引 | ResNet模块 | 通道 | 384×128输入下空间尺寸 | block数 |
|---:|---|---:|---:|---:|
| 0 | layer1 | 256 | 96×32 | 3 |
| 1 | layer2 | 512 | 48×16 | 4 |
| 2 | layer3 | 1024 | 24×8 | 6 |
| 3 | layer4 | 2048 | 24×8 | 3 |

PSG继续复用同一个`PoseSpatialGate`。每个Bottleneck输出的NCHW feature临时展平为`B×HW×C`，
经过PSG后原样还原为NCHW；不改变PSG数学定义、初始化或参数化。

### 三个arm

1. **R50-B0**：标准ResNet-50 global baseline，无pose loader消费、无anchor、无PSG。
2. **R50-D0**：layer3输出进入单点TAPF anchor，生成内部field并供layer4的PSG bank使用；训练期
   持续pose supervision，推理期RGB-only。source=`[2]`，consumer=`[3]`。
3. **R50-HT0**：
   - layer2输出→Stage-1 projection/shared decoder→field-1→layer3 PSG bank；
   - layer3输出（已由field-1更新）+ detached prior state→Stage-2 projection/shared decoder→
     refined field-2→layer4 PSG bank；
   - 两个anchor节点分别对应一个后继PSG bank；projection与PSG独立，decoder严格共享一份；
   - 两节点pose loss取算术平均，训练期持续监督，推理期RGB-only。

### 训练配方

三臂固定单变量matched：

- dataset=`Occluded-Duke`，输入=`384×128`，batch=`64`，seed=`1234`；
- ImageNet ResNet-50同一权重文件与SHA；
- ImageNet normalization，flip=`0.5`，padding=`10`，random erasing=`0.0`；
- optimizer=`Adam`，base LR=`3e-4`，warmup=`5`，cosine，weight decay=`5e-4`；
- `120` epochs，每10 epoch保存并评估，test batch=`64`，no re-ranking/NFC；
- softmax+triplet、BNNeck、global descriptor，三个arm的classifier、sampler和loss路径一致；
- 独立OUTPUT_DIR：
  - `log/occluded_duke/exp380_r50_b0_s1234`
  - `log/occluded_duke/exp380_r50_d0_s1234`
  - `log/occluded_duke/exp380_r50_ht0_s1234`

采用ResNet代码线的标准Adam配方，而不是把Swin专用SGD配方机械移植过来；这一选择对三个arm完全
一致。实验结论只在该ResNet配方内部成立。

## 对照组与结果判读

必须完整跑完并按顺序解释：

1. `D0−B0`：完整单anchor+PSG是否跨backbone保留；
2. `HT0−D0`：每anchor一PSG与递进state是否产生额外价值；
3. `HT0−B0`：只作总模块差值，不能代替第2项；
4. 所有final都报告mAP/R1/R5/R10显式正负差值，不因单epoch或单门槛提前停止整个链路。

描述性判读线：`D0−B0 >= +0.5 mAP`视为跨骨干正信号；`HT0−D0 >= +0.3 mAP`视为逐层正信号。
这两条不是提前停训门槛，所有已启动arm仍跑满120 epoch。单seed只用于方向筛选；最终论文显著性
需要在最佳结构上补multi-seed。

## 启动前Gate A

1. 默认config与旧Swin B0/D0/HT0路径逐位不变；
2. R50-B0新wrapper关闭时与现有`Backbone`的state keys、共享参数初始化、forward、loss与optimizer
   groups exact parity；
3. R50-D0/HT0的共享ResNet、BNNeck、classifier初始化exact matched，HT0的layer4 PSG与D0 exact；
4. layer2/3 feature shape、source→consumer路由、每anchor一PSG bank与single shared decoder通过hook；
5. e1 teacher exact；e11全部consumer只读各自内部field，两个field非复制；
6. pose loss只更新projection/shared decoder，不回流ResNet/PSG；ReID loss只更新ResNet/PSG/
   classifier，不更新anchor；
7. correct/shuffle/None/exploding external pose在eval descriptor exact parity；
8. full-model strict reload、构造RNG、optimizer membership与参数量审计；
9. PyTorch1.13.1真实Occluded-Duke batch64 CUDA/AMP、10-step parity与真实overflow整步跳过；
10. ImageNet权重SHA、exact execution commit/full-history bundle/config SHA、独立remote repo、GPU空闲、
    output不存在均固化后才允许fresh启动。

用户明确禁止Claude，本实验不生成或调用Claude审查；用独立design、Codex代码审查、unit、
full-model invariants及4090原生CUDA/AMP门禁替代。任何Gate未完成都不得启动训练。

## 风险与失败解释

1. **D0≈B0**：完整模块在ResNet上不具可迁移性；不能只保留Swin正点声称backbone-agnostic。
2. **D0>B0但HT0≈D0**：最终方法优先收敛为单anchor+PSG；逐层版只作中性扩展或消融。
3. **HT0>D0但仅ResNet成立**：可提出强backbone吸收结构先验的解释，但在ViT验证前不能声称普适。
4. **弱B0导致大涨**：必须同时报告B0绝对性能与同backbone差值，避免把弱baseline headroom包装成
   创新；不与Swin绝对值横比。
5. **BN/AMP不稳定**：先修实现与数值路径，不改batch、不降输入尺寸、不手工跳epoch。
6. **参数量混淆**：HT0比D0多一个projection与一组PSG是方法定义的一部分；若HT0形成正信号，再
   补parameter-matched静态/RGB control，不在首轮三臂前抢跑。

## 后续顺序

ResNet三臂闭合后，再选择合适ViT做同样的B0/D0/HT0内部排序。只有单图跨backbone证据明确后，
才进入Video ReID：把内部姿态state扩展为跨帧可靠性、运动连续性和遮挡恢复，并与逐帧TAPF、
普通temporal pooling、外部pose smoothing及RGB-only video backbone对照。
