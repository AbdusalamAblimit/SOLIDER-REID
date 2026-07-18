# 实验 exp391：官方 clean 全阶段独立直预测 TAPF

> 状态：PHASE-A PREFLIGHT / NO-START。exp390 四臂已全部自然 e120、终审并完成多 seed 汇总；
> 当前只允许实现和审计 H2-M，全部门禁通过前不得创建正式 output 或启动训练。

## 动机

clean 单层 D0 在 Occluded-Duke seed1234 相对 B0 为 `+0.2 mAP`，seed4321 新完成结果为
`+0.8 mAP`；因此完整 `anchor+PSG` 尚不能判为无效，但需要 exp390 的第三个 seed确认稳定性。

层级证据需要重新划清：

1. 旧 exp379 使用 Stage-1/2 stage-specific projection、唯一 shared decoder；Stage-1直接建状态，
   Stage-2显式读取上一状态并预测 posterior/confidence residual。其 HT0−D0 在 Swin/ResNet/ViT
   分别为 `−0.1/+0.8/−0.3 mAP`，只支持backbone-conditional结果。
2. clean exp389 已经不是上述offset/refinement：它有参数独立的Stage-1 early anchor和Stage-2 late
   anchor，两个都直接从各自feature预测absolute field，不显式读取上一pose state。它只覆盖两个
   source stage，early同一field经过六个Stage-2 PSG，late经过两个Stage-3 PSG；final HT0−D0=
   `−0.7 mAP`。
3. exp389冻结旁路显示early六bank只贡献约`+0.0723 mAP/0 R1`，late两bank贡献约
   `+1.3565 mAP/+1.9005 R1`。early不是dead，但几乎没有独立价值；旁路early也不能恢复D0，说明
   主要差异已经发生在联合训练轨迹中。
4. 独立复核还发现一个不能带入全stage实验的尺度混淆：exp389的Stage-1/2 native field分别为
   `48×16/24×8`，却都用grid-space `GAUSSIAN_SIGMA=1.5`。同一个grid sigma映射回原图后物理尺度
   相差2倍；若再直接加入`96×32`的Stage-0，三层物理sigma会成为`1:2:4`。旧exp379的两个节点
   统一输出`96×32`，没有这个混淆。

用户提出让所有stage各自直接预测pose。该三source全直预测版本尚未在official clean runtime中测试，
但不能直接把Stage-0层叠到已失败的六/二不平衡拓扑上；应先隔离loss budget敏感性与consumer site
不平衡，再做参数/loss/AMP matched的Stage-0 route off/on。

## 核心假设

Swin不同stage的feature各自包含可直接恢复的尺度特定姿态证据。若每个source stage独立预测absolute
pose field，并只调制紧邻的下一视觉stage，则Stage-0/1/2三层可以形成：

```text
Stage-0 F0 → direct anchor A0 → Stage-1 PSG
Stage-1 F1 → direct anchor A1 → Stage-2 PSG
Stage-2 F2 → direct anchor A2 → Stage-3 PSG → GAP descriptor
```

与旧exp379不同，`A1/A2`不读取`A0/A1`的pose state，不预测offset，也不共享decoder；各层是否有效
由独立route-off/on和冻结旁路检验。方法强度必须来自多个stage真实增加检索性能，而不是“插入位置
更多”或参数更多。

## 分阶段单变量顺序

### Phase A：H2-M，先诊断exp389的loss aggregation

完全保持exp389的两个direct anchors、early六bank、late两bank、初始化、路径和recipe，只把总pose
objective从：

```text
0.1 × (L_early + L_late)
```

改为：

```text
0.1 × mean(L_early, L_late)
```

这只解释loss-budget/anchor-step敏感性。mean会把两个anchor各自的pose gradient减半，不预设为比sum
“更公平”；两个pose source都detach，pose loss不进入Swin/PSG/head，所以也不得先验声称sum直接
干扰backbone。必须记录默认GradScaler每步scale/skip，检验唯一全局数值耦合。

H2-M必须fresh完整e120。若final低于exp387 D0超过`0.2 mAP`，或冻结early-bypass显示early贡献仍
小于`+0.1 mAP`，Phase A判NO-GO，不增加Stage-0。

### Phase B：H2-B，平衡六/二consumer site

仅在Phase A通过后运行。保持两个direct anchors与mean pose loss，只把early Stage-2 consumer从六个
block后全部注入改为预注册的两个位置：block0与block5；late仍为Stage-3 block0与block1，因此两个
pose level都是2个独立consumer。该arm回答exp389的主要结构不平衡是否令early field被重复消费、
容量/累计调制过强。

H2-B必须相对H2-M达到`>=+0.3 mAP`且R1不低于`−0.2`，冻结early route贡献达到`>=+0.1 mAP`，
才进入全stage。失败则保留“单层D0优于当前多层直预测”的结论。

### Phase C：H3-OFF/H3-ON，全stage direct的严格pair

仅在Phase B通过后运行。两臂都实例化并训练相同的A0/A1/A2、三组pose loss、2/2/2 PSG bank、参数、
optimizer成员和AMP路径；三个pose loss统一取mean，唯一差异为Stage-0 route是否写入Stage-1视觉流：

- `H3-OFF`：A0与对应PSG参数存在、接受相同pose/ReID优化边界，但Stage-0→Stage-1 consumer显式
  identity bypass；
- `H3-ON`：开启Stage-0→Stage-1两个consumer；其余逐元素与H3-OFF一致。

source/consumer固定为：

| source anchor | 输入feature | absolute field | target sigma | consumer |
|---|---|---|---:|---|
| A0 | Stage-0 pre-downsample，`96×96×32` | 原生grid `17×96×32` | `6.0` | Stage-1 block0/1后 |
| A1 | Stage-1 pre-downsample，`192×48×16` | 原生grid `17×48×16` | `3.0` | Stage-2 block0/5后 |
| A2 | Stage-2 pre-downsample，`384×24×8` | 原生grid `17×24×8` | `1.5` | Stage-3 block0/1后 |

sigma以A2的`1.5`为基准，按source stride反比缩放，使三层都对应原图约`24 px`的物理标准差；
H3-OFF/H3-ON必须使用完全相同的renderer与三层target。另一可接受实现是三层统一渲染到共同field
resolution，但不得继续在不同native grid上机械复用`1.5`。这项归一只消除监督尺度混淆，不作为
额外可调超参数。

三个anchor沿用clean direct结构并参数独立；A2与现有D0 late anchor构造顺序、参数名和初始化必须exact。
所有field边界detach：pose loss只更新对应anchor，ReID loss只更新Swin/PSG/head。测试期三个anchor只读
RGB feature，external correct/shuffle/None/exploding均不得被访问。

## 对照与报告

1. 原子参考：exp390三seed paired B0/D0；
2. clean失败参考：exp389 sum-loss 6/2 HT0；
3. Phase A：H2-M相对D0与exp389；
4. Phase B：H2-B相对H2-M与D0；
5. Phase C：H3-ON相对严格matched H3-OFF，并同时报告H3-ON相对D0。

每臂都只报告自然e120 final，不挑best。H3主张至少要求：

- H3-ON−H3-OFF `>=+0.3 mAP`且R1不低于`−0.2`；
- H3-ON−D0 `>=+0.5 mAP`；
- 冻结checkpoint逐stage旁路中至少两个stage各自贡献`>=+0.1 mAP`；
- correct/field干预必须实际改变对应stage gate与descriptor；若correct与matched/static仍不可分辨，只能
  称training-privileged multiscale modulation，不能声称精确关节语义因果。

## 启动前全套门禁

1. config-off与D0-off的state/init/RNG/optimizer/10-step CUDA-AMP逐字节exact；
2. A2 late path相对exp387/390 D0参数名、初始化、路由exact；
3. 每个direct anchor的absolute预测不读取prior，hook调用次数严格为1；
4. H2为6/2或2/2、H3为2/2/2 consumer route exact，所有bank参数独立；H3三层target的原图
   物理sigma exact matched，并以renderer单元测试证明；
5. pose/ReID双向gradient ownership、strict state、overflow整步skip；
6. 真实batch64/8-worker 24-step CUDA/AMP，记录GradScaler scale与skip；
7. correct/shuffle/None/exploding pose-free descriptor/三field/六gate exact；
8. 每个consumer逐一旁路均使最终descriptor出现有限非零变化；
9. H3-OFF/H3-ON除route开关外state keys、参数、optimizer、RNG和loss逐项matched；
10. 参数、supported FLOPs、训练/eval显存与速度相对D0完整报告；
11. fresh repo、exact commit/full-history bundle/config SHA、output不存在、GPU空闲；严格串行、batch64、
    seed1234、120epoch、SGD/lr0.0008，不续训、不重复、不运行中改代码/config。

## 风险与失败解释

1. H2-M仍负：不是“mean没调好”，而是当前两层direct topology本身无燃料；停止。
2. H2-M恢复、H2-B不升：loss budget敏感但consumer balance假设失败；不做H3。
3. H2-B升、H3-ON≈H3-OFF：Stage-0 direct anchor没有检索贡献；保留两层版本，不包装全stage。
4. H3-ON升但只有late stage旁路敏感：仍退化为单层D0，不称hierarchical。
5. 多stage相对D0升但correct/matched/static不可分：只支持多尺度训练调制，不支持姿态语义因果。
6. 参数/FLOPs增幅过大：即使小涨也不能作为主方法，必须与轻量D0并列报告。
7. H3继续对三个native grid固定同一sigma：实验把层级与pose target sharpness/field support混在一起，
   属于门禁失败，不得启动或据此解释多阶段有效性。

## 审查与执行边界

本设计由主审与只读独立子agent交叉检查；禁止Claude。独立审查指出exp389已经是两层direct、sum
loss不能直接归因为backbone干扰、6/2 consumer不平衡是更强结构差异，并进一步发现native-grid
固定sigma会混淆跨stage监督尺度，因此采用上述A→B→C顺序，并在H3固定物理尺度归一。
exp390已封板；当前按预注册顺序进入Phase A H2-M实现与全套门禁。Phase B/C继续保持NO-START，
只有前一阶段自然e120与终审满足design阈值后才允许推进。
