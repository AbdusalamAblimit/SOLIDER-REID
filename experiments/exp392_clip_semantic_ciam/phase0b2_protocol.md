# exp392 Phase 0B2：CLIP 解剖 slot 条件读出协议

> 状态：`PROTOCOL-FROZEN / STATIC-PASS / OPENCLIP-CPU-PASS / GPU-NO-START`
>
> 本协议不撤销 Phase 0B 的 `CURRENT_CLIP_TEACHER_NO_GO`。它只定义一个新的 teacher
> 接口及其零训练证伪顺序；在 CPU/static、独立代码审查和小样本 smoke 全部通过前，不占用 4090。

## 一、Phase 0B 真正排除了什么

Phase 0B 已经通过 tight-crop反证、head→legs纵向顺序、OpenCLIP token parity、`ln_post+proj`、label
顺序和插值对照，实质性排除了能单独解释`2.7%~4.6%`的常见大错。独立复核同时指出两个仍需在
B2补齐的审计缺口：原 synthetic RGB/mask 检查使用同一个signal走同构resize，只证明函数内部一致，
没有覆盖`original pose -> resize/flip/pad/crop -> rendered mask peak`全链；matched-mask donor也只在增强前
pose descriptor上匹配，未报告增强后的realized match。它们不推翻tight-crop/global CLS与dense路径
之间的强差异，但在B2前必须补齐，不能继续写成“所有几何可能均已严格排除”。失败对象是：

```text
last-block 普通 patch token
-> pose mask 加权平均
-> 与五个 body-part name prototype 直接分类
```

该接口的 macro top-1 只有 `2.692%/4.637%`，但同一 mask 的 tight-crop global CLS 可达
`44.688%`，image-only patch cluster 可达 `52.8%~60.0%`。因此 patch 中有空间结构，CLIP 也能理解
部分局部人体语义；错误在于普通 patch residual 没有被 CLIP 的 global image-text objective 校准到
body-part 文本轴。

另一个更根本的设计错误是：pose mask 已经给出 region identity，上一版却让 CLIP 再猜一次
“这是 head 还是 arm”。这会把 arms/upper-leg 的 prompt 重叠当成 teacher 主任务，并不能提供
TAPF 真正缺少的 sample-specific 信息。

## 二、重定义 teacher 的职责

Phase 0B2 固定以下分工：

1. **pose 决定不可置换的 anatomical slot identity 与 geometry**；
2. **frozen CLIP image encoder 读取该 slot 当前 RGB 中的实例视觉证据**；
3. **frozen CLIP text encoder只把证据投影到 slot 内的 support/appearance 语义轴**；
4. CLIP 不再负责重复预测已知的 body-part class；
5. student 预测的 slot state 后续必须进入 TAPF 的 gather-transform-scatter consumer，不能成为
   terminal auxiliary head。

这仍是双编码器 teacher，但问题从跨 slot 分类改为 slot-conditioned visual calibration：

```text
pose-fixed slot c + aligned RGB
        -> frozen region-aware CLIP visual readout v_c
v_c x frozen slot-conditioned text bank
        -> q_support_c, q_appearance_c
        -> executable student state (M_c, r_c, a_c)
        -> semantic expert router
        -> final single global descriptor
```

推理删除 CLIP image/text encoder、prompt bank和 external pose；保留 RGB student anchor、固定 slot
binding、router 和单一 global descriptor。

## 三、互斥解剖 ontology

仍使用五个 coarse slots：`head/torso/arms/upper-legs/lower-legs`，但不再允许 shoulder、hip、knee
在多个 region 中以完整强度重复出现。

1. 先按 COCO-17 joints/limb segments渲染五个 nonnegative raw supports；shoulder/hip/knee边界joint
   各有唯一owner，arms/upper-leg/lower-leg的segment两端各trim `15%`，torso保留完整central
   chest/abdomen/waist segments；
2. 对每个像素先得到composition `P_c=raw_c/(sum_c raw_c+eps)`，再保留总证据幅度
   `a=min(sum_c raw_c,1)`，最终`M_c=a*P_c=raw_c/max(sum_c raw_c,1)`；不能把低置信微弱tail重新
   放大成sum为1的强support；
3. 所有 raw support 为零的像素保持全零，不分给任一 slot，且始终`sum_c M_c<=1`；
4. amplitude-preserving soft partition作为主定义；crop bbox用hard argmax owner后的`M_c`，再按该slot
   自身max的5%取support域，避免Gaussian非零tail扩张到整图；5%只影响参考crop bbox，不影响slot
   valid、teacher loss coverage或student target；
5. left/right 在本阶段合并，避免把 CLIP 不可靠的细粒度左右语义提前引入；
6. slot channel 顺序由固定 incidence table定义，不允许学习 permutation 或 Hungarian matching。

必须先做 synthetic one-hot joint/segment、左右翻转、边界空 mask 和 `sum_c M_c<=1` 的 CPU exact
测试，并在真实增强后报告pairwise normalized intersection：median `<0.10`、P95 `<0.25`。该 ontology
一旦进入teacher-only审计即冻结，不能按结果改region边界。

## 四、候选 visual readout

### 4.1 主候选：PC-MBCLS

`PC-MBCLS`（pose-conditioned multi-block CLS）复用 frozen CLIP ViT-L/14 的前20个 block，只在
最后4个 block为五个 region复制 CLS readout。具体实现必须把block20前的完整官方token sequence沿
batch维复制五份，每份仍只有一个CLS；禁止把五个region CLS塞进同一sequence，否则patch会看到额外
CLS、region之间也会互相attention，all-one parity不可能成立。patch token来自同一个几何对齐的行人
RGB；每个region的CLS query在每个后段block加入无可学习参数的mask log-prior：

```text
delta_c = rho * Sum(M_c) / N_patch, rho=0.01
b_c(p) = log((M_c(p) + delta_c) / (Max(M_c) + delta_c))
attn_logit_c(CLS -> p) += b_c(p)
```

- bias 只作用于 `CLS query -> patch key`，不修改 patch-to-patch 或 patch-to-CLS logits；
- region CLS和共享 patch tokens一起经过后4个 frozen residual blocks；
- additive background leakage总质量约为region mass的1%，避免逐patch固定`eps`在稀疏region上累积成
  与前景相当的总质量；
- all-one mask产生严格零 bias，FP32与官方global CLS max error必须`<=1e-6`；
- zero mask不执行 readout，返回显式 NULL/invalid，不得借 CLS residual产生伪证据；
- 不增加可训练 adapter、projector、prompt或attention参数；
- 理论 block计算量约为`20 + 5x4 = 40`，即单次 ViT-L/14 的`1.67x`，显著低于五次完整 crop
  forward 的`5x`；真实速度和显存必须实测。

该接口沿用 CLIP 真正受监督的 CLS projection，而不是假定普通 patch token天然与文本对齐。后4层
不是可调搜索项；若该固定定义失败，先封板归因，不按结果试`1/2/6/8`层。

### 4.2 受监督global路径参考：region-crop global CLS

对每个 region 的 hard support bbox外扩15%上下文，保持原宽高比、letterbox到224，并在 bbox 内
保留原 RGB。五个 crop批量送入完整 frozen CLIP global image encoder。它只承担三个作用：

1. 验证当前 prompt/support task 在真正受监督的 global CLS路径上是否可解；
2. 给 PC-MBCLS 提供受监督global路径参考；crop会放大局部并改变上下文，不是同信息条件下的理论上界；
3. 评估离线缓存的数值一致性和容量边界。

由于它需要约`5x` CLIP forward，默认不授权作为120-epoch online teacher。只有 PC-MBCLS未过门禁、
crop teacher明显通过，且离线缓存能对 official flip/pad/crop 几何给出一致监督时，才另写协议讨论；
不得直接把昂贵上界塞进正式训练。

### 4.3 Phase 0B naive dense readout

原 last-block patch pooling 只作为冻结负对照，不重跑全量、不修 prompt救场。历史 exp354 的
MaskCLIP value-only 小样本归属失败也保留为负证据；任何 dense segmentation读出若要复活，必须是
独立单变量接口，并先证明它在本协议的 slot-conditioned task 上优于 PC-MBCLS，而不能只引用论文。

## 五、双编码器输出定义

每个固定 slot使用两组冻结文本轴；不再把五个 slot name放进同一个互斥分类器。

### 5.1 visual-support bank

首个定义只使用二分类、多模板平均原型：

1. `clearly visible c with reliable visual evidence`；
2. `occluded or obstructed c with weak visual evidence`。

得到 `q_support_c`，执行可靠度首版固定为`r_c=q_visible`，不得学习标量重映射。只有二分类在三种
合成遮挡上均表现单调，才允许另开B2-S2把第二类拆成partial/heavy/none；不得一次引入四级后按结果
合并。zero/invalid pose slot直接返回NULL且不计算teacher loss，不能把它偷换成可见slot的`q_none`。

### 5.2 appearance bank

appearance不与support首验捆绑。support通过后，先把外观拆成两个互不替代的分布：

1. dominant color：`black/white/gray/red/blue/green/brown/yellow/other-or-mixed`；
2. texture：`solid/patterned/textured`。

两者都使用带slot name上下文的prompt，不能把`patterned`与颜色塞进同一个互斥softmax。它们不是
最终descriptor KD；未来只允许作为对应anatomical expert内部的mixture coefficient或同slot relation
teacher。若teacher-only证据显示颜色/纹理轴不稳定，整组删除，不能用可学习prompt掩盖失败。

### 5.3 local visual feature

同时保留归一化 `v_c` 作为审计量。未来如使用 visual cosine/relation loss，必须接到推理保留的
slot executor `z_exec_c`，不能另建训练后删除的 projector。该变量必须晚于 support-only CIAM，作为
独立实验，不与首个正式 arm捆绑。

## 六、严格单变量审计顺序

### B2-O：只修 ontology

固定region-crop global CLS、Phase 0B原始五个part-name prototypes/prompt、geometry和指标，只把旧
重叠mask替换为
第三节互斥ontology。该步不得引入PC-MBCLS或support/appearance bank。使用PID-cluster bootstrap，要求：

- macro top-1 lower bound `>=35%`；
- 五类各自lower bound均`>20%`；
- 每类raw expected cosine margin lower bound均`>0`；
- post-transform pairwise mask overlap满足第三节门槛。

若mask ontology通过overlap/static却仍因原`torso and upper body`文本与arms混淆，必须另开`B2-P`
prompt-only步骤；不得把prompt修改塞回B2-O、删除arms/upper-leg或按结果同时改mask与prompt。

#### B2-O 封板边界与 B2-O2 预注册

128图CPU smoke显示，上述amplitude-preserving **soft** partition的median overlap虽仅
`0.000322`，但P95=`0.438040`、max=`0.997638`，未过`P95<0.25`。这不是阈值边缘：
soft composition仍允许多个slot在同一像素上共存，因而不能实现本阶段要求的像素级不可置换
identity。B2-O因ontology gate失败封板，不改阈值、trim比例或Gaussian sigma救场。

`B2-O2`只修正这一个定义错误：保持raw joint/segment support、唯一joint owner、
`15%`segment trim、证据幅度`a=min(sum raw,1)`、crop、RGB、CLIP、prompt、geometry和指标全部
不变，用固定region顺序打破tie，定义

~~~text
owner(p) = argmax_c raw_c(p)
M_c(p) = min(sum_j raw_j(p), 1) * 1[owner(p)=c]
~~~

zero-support仍全零；这是teacher ontology的离线渲染，不向pose回传，因此不需要用soft
overlap交换可微性。B2-O2必须首先达到overlap的median/P95/max严格`0`、coverage exact与全数值
finite。该步仍使用Phase 0B原prompt，所以arms/upper-leg文本分类失败只用来触发后续
`B2-P prompt-only`，不得与B2-O2同时修。

### B2-I：只改 readout 接口

固定B2-O ontology、RGB、五个part-name prototypes、geometry和指标，只比较：

1. 同ontology的naive patch负对照；
2. PC-MBCLS；
3. crop global CLS参考。

目的只回答“沿CLS受监督路径能否恢复local-text alignment”。PC-MBCLS至少需满足：

- all-one FP32 max error `<=1e-6`，repeat exact，finite；
- macro top-1 PID-bootstrap lower bound `>=max(30%, crop_reference-10pp)`；
- 五类各自lower bound均`>20%`，raw expected cosine margin lower bound均`>0`；
- correct相对四个non-identity channel cycles的平均top-1优势lower bound `>=10pp`；
- correct优于matched-wrong mask、mass-matched uniform/fixed bands和真正的non-human wrong text bank；
- 两套互不重叠但语义正确的paraphrase bank top-1 agreement `>=90%`。

### B2-S：只改 teacher 语义对象

固定B2-I胜出的readout与B2-O ontology，只把五类part-name诊断改为support/appearance bank。主指标
不再是“猜中slot name”，而是teacher是否依赖当前RGB、当前region和文本语义：

1. 同图同 slot 的两次轻几何/颜色视图 `q` 一致性；
2. wrong RGB拆成“不同PID同slot有效局部”和“donor RGB+recipient mask错配”两臂；
3. fixed RGB换增强后realized geometry仍匹配、但低IoU的wrong mask；
4. 四个non-identity slot cycles，以及同步置换mask/slot/text后的inverse-map equivariance；
5. uniform、fixed horizontal bands、text-only constant、image-only feature/cluster；
6. region-overlap synthetic erasing与non-overlap erasing；
7. horizontal flip后空间反映射的一致性；
8. high/low pose confidence、valid/invalid和真实遮挡分组；confidence只分层报告，不独立kill；
9. 同PID跨相机同slot相似度与matched不同PID对照，仅作appearance可用性指标，不当身份teacher；
10. q的跨样本方差、JSD、centered effective rank、entropy和每slot confusion。

预注册核心门槛：

- same-image same-slot的相似度显著高于不同PID同slot及同图wrong-slot，PID-cluster paired bootstrap
  95% CI均不跨0，标准化效应期望`>=0.2`；
- same-image same-slot的q-JSD显著小于RGB-mask mismatch与低IoU wrong-mask；
- region-overlap erasing分别使用CLIP mean、随机纹理和different-PID CutMix，`q_visible`随overlap
  单调下降的Spearman lower bound `>=0.2`，且效应显著强于non-overlap erasing；本任务明确声称
  support，因此该项是kill-switch；
- wrong text显著改变support分布，而text-only constant不能复现图像/遮挡分组；
- wrong mask/channel cycle不能与correct近似等价；
- valid region的flip raw feature cosine `>=0.95`、median q-JSD `<=0.02`；support-state top-1 consistency
  `>=95%`只在原始raw margin足够大的subset上要求；
- 每个有效slot centered effective rank `>=2`，不存在单一常量输出；
- PC-MBCLS吞吐不低于单CLIP的50%，teacher峰值显存与ReID sequential forward预算可共存。

任一核心输入不敏感、support对合成遮挡方向相反，或text-only/static强对照复现sample binding，
当前B2 teacher即`NO-GO`。不得把只通过appearance同PID指标当作语义门禁通过。

## 七、代码与数值门禁

实现前必须独立核对 OpenCLIP 当前版本中：

1. residual block的pre-LN顺序、`MultiheadAttention` q/k/v布局和attention mask广播；
2. `attn_mask`只改变指定CLS query行，其他logits逐元素不变；
   需要单patch定向测试，不能只靠all-one parity，因为float mask所有logit同加常数也会保持softmax；
3. AMP下`log-prior`dtype、`-inf`避免、zero/all-one mask边界；
4. 五region必须沿batch复制独立sequence，batch展开和还原顺序exact；
5. positional embedding保持原patch位置，不对region crop式重排；
6. 最终每region使用官方`ln_post + visual.proj + L2Norm`；
7. teacher永久`eval/no_grad/requires_grad=False`且不进入optimizer/state dict；
8. 手动前20+后4无mask路径与官方24层一致；all-one分支使用与官方相同的`attn_mask=None` fast path，
   CPU fp32 max error `<=1e-6`，CUDA AMP误差另行冻结；
9. 原图尺寸必须与pose artifact记录exact；用可解析synthetic坐标覆盖
   `pose->resize->flip->pad->crop->render->CLIP grid`全链，峰值/质心与解析结果一致；
10. wrong RGB/mask/text donor均无fixed point、different-PID；wrong mask按实际增强后面积/y/conf重新
   匹配并报告IoU，median IoU `<=0.30`、P95 `<=0.50`；
11. artifact、checkpoint、tokenizer、prompt、runtime、脚本和runner SHA完整记录。

CLIP normalization、bicubic/antialias与checkpoint metadata必须从OpenCLIP官方preprocess config显式读取
并写入artifact；为保持RGB/mask同几何，可复现其参数，不能直接调用会独立center-crop的黑盒transform。

Phase 0B现有`spherical_kmeans()`已核实确实执行20轮assignment/center update，并非只做初始化；但
其fit/eval使用同一全集。B2的image-only cluster控制必须改为PID-disjoint fit/eval、固定seed，并把
raw cosine margin设为主指标，temperature-dependent probability margin/JSD只作补充。

region-crop若使用缓存，key至少包含relative path、image/pose manifest SHA、实际geometry record、
ontology、CLIP checkpoint/preprocess、prompt和interface SHA。同key online/cache要求FP32 max error
`<=5e-6`；FP16 cache cosine `>=0.9999`、q-JSD `<=1e-5`。geometry/seed变化必须cache miss，禁止把
original-image cache默认为random student geometry的aligned teacher。

CPU/static和128图smoke通过只授权全train teacher-only审计，不授权Phase 0C或正式训练。

## 八、未来与 TAPF 的深耦合边界

若B2 teacher通过，Phase 0C只实现single-stage `Semantic TAPF`：

```text
student anchor predicts fixed slots (M_c, q_support_c, q_appearance_c)
        -> detach state
        -> slot c selects semantic low-rank expert
        -> q_support gives execution reliability
        -> q_appearance mixes slot-internal expert atoms
        -> gather local ReID evidence
        -> transform and scatter to backbone
        -> final global descriptor
```

CLIP loss只更新anchor/state head；ReID loss通过执行router更新backbone与router，但不能改写slot state。
必须有NULL identity、correct/wrong binding、expert-mean、generic low-rank adapter、static state和bypass
强对照。只有single-stage同时满足语义反事实与final性能门槛，才重新授权balanced semantic multi-stage；
这正是CLIP与TAPF的深耦合，不是给final descriptor附加普通CLIP KD。

## 九、当前裁决

`PHASE 0B2 PROTOCOL FROZEN / STATIC-PASS / OPENCLIP-CPU-PASS / GPU NO-START`。

三路独立审查和两层CPU契约已经通过。下一步顺序固定为：B2-O ontology/reference小样本CPU smoke →
B2-I readout小样本smoke → 若全部通过才允许唯一4090执行teacher-only全量门禁。任何teacher-only
通过都只授权Phase 0C，不直接授权120 epoch训练。
