# exp392 Phase 0B2：CLIP 解剖 slot 条件读出协议

> 状态：`B2-Sv1 SEALED-FAIL / B2-Sv2 DESIGN-FROZEN / IMPLEMENTATION-NO-START / GPU-NO-START`
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

#### B2-P：一次性 prompt-only 消歧

B2-O2的128图结果在exact-zero ontology下仍为head/torso/lower-leg强、arms/upper-leg弱，证明
mask修正与文本消歧是两个变量。B2-P固定B2-O2 hard-owner、region crop、RGB、CLIP、四个template、
geometry、样本与指标，只把五个region phrase一次性冻结为：

1. `head, face, and hair`；
2. `chest, abdomen, waist, and torso`；
3. `upper limbs, arms, elbows, forearms, wrists, and hands`；
4. `thighs between the hips and knees`；
5. `lower legs below the knees, including shins, calves, ankles, and feet`。

该词表不用否定词，不用`upper body`或`upper legs`这类跨slot umbrella term，不增加template、
可学习prompt、同义词搜索或按类别单独调参。128图CPU smoke要求ontology overlap exact-zero、
coverage exact、finite、macro top-1 point `>=35%`、五类top-1 point均`>20%`、五类raw cosine
margin point均`>0`；通过只授权B2-I CPU readout比较。未来全train teacher-only仍使用原PID-cluster
lower-bound门禁，不能用128图点估计替代。

#### B2-P 失败后的实际语义对象审计：B2-SC -> B2-SI

B2-P的128图结果显示disjoint phrase能把upper-leg从约`18%`提高到`30%`，但arms仍严格`0%`，
因此五类part-name诊断不能作为全slot readout gate继续使用，也不得再试第二组同义词。该失败不授权
忽略arms，也不否定原始teacher假设：pose已经固定slot identity，最终teacher需要回答的是该slot当前
是否有可执行视觉证据，而不是再次猜slot name。

因此后续顺序改为两个仍然单变量的零训练审计：

1. `B2-SC crop-support feasibility`：固定B2-O2 hard-owner、region-crop global CLS、RGB、
   geometry、CLIP和每slot crop，只把五类part-name目标替换为原第五节预注册的slot-conditioned
   visible/occluded二分类；readout不变；
2. `B2-SI support-readout`：只有B2-SC通过后，固定同一support任务、RGB、mask和文本，唯一把
   crop-global readout替换为PC-MBCLS，并保留crop结果为reference。

B2-SC使用原始crop bbox固定几何，不因遮挡重算bbox；在每个有效slot support内以固定坐标顺序构造
`25%/50%/75%`三档嵌套遮挡，首验只用CLIP mean replacement，避免把材质变量一起带入。128图前先做
8图CPU contract。smoke必须满足：

- 0/25/50/75四档coverage exact、finite、repeat exact；
- 每slot实际support overlap严格递增，目标误差`<=1/support_pixels`；
- macro以及每slot的`Spearman(overlap, -q_visible) > 0`；
- 每slot `mean(q_visible@0 - q_visible@75) > 0`；
- 至少三档相邻宏平均响应严格单调。

通过只授权B2-SI；完整B2-S仍需CLIP-mean、随机纹理和different-PID wrong-slot occluder三类
遮挡、non-overlap control、wrong RGB/mask/text、flip与PID-cluster bootstrap全部门禁。同slot
donor只用于appearance/binding对照，永不进入visibility下降kill-switch。B2-SC失败才是当前
slot-support语义对象的直接NO-GO，不得用appearance任务救场。

#### B2-SI：同一 support 任务的 PC-MBCLS readout

B2-SC的128图结果通过后，B2-SI固定hard-owner ontology、support二分类prompt、0/25/50/75嵌套
CLIP-mean遮挡、样本、CLIP checkpoint和温度。唯一主候选变化是把crop-global CLS换成PC-MBCLS。
为避免crop缩放差异混进主比较，B2-SI在384x128 full RGB上先施加同一target-slot遮挡，再固定使用
aspect-letterbox到224；hard owner像素mask用nearest映射到content box，随后14x14 average pooling为
patch coverage。这样跨slot混合只可能来自同一patch覆盖多个解剖区域，不会由bilinear在同一像素
重新制造soft identity overlap。

每个target-slot/level的full image只共享运行前20个CLIP blocks一次；同一个shared token同时得到：

1. PC-MBCLS五个slot feature；
2. 同图official global CLS control；
3. target slot的`q_visible`；
4. 四个non-target slot各自support bank的`q_visible`。

8图CPU contract先要求既有OpenCLIP parity继续PASS，并冻结以下smoke门禁：

- target slot五类`Spearman(overlap,-q_visible)>0`且0→75%下降均为正；
- target response的macro三档相邻差值均为正；
- 每个target class的0→75%下降大于同图四个non-target slot平均下降；
- macro target下降大于official global CLS对同一target文本的下降；
- repeat exact、finite、hard mask映射后像素级互斥、level overlap误差不超过一个support pixel；
- PC-MBCLS每个valid target均执行，zero/invalid仍显式NULL。

8图通过后才跑128图；128图通过只授权完整B2-S teacher-only反事实审计，不授权训练。若PC-MBCLS只
对全图遮挡敏感而不能优于non-target/global control，则当前readout `NO-GO`，不得用crop-global
`5x` teacher直接进入120-epoch训练。

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
- region-overlap erasing分别使用CLIP mean、随机纹理和different-PID wrong-slot occluder，
  `q_visible`随overlap单调下降的Spearman lower bound `>=0.2`，且效应显著强于non-overlap
  erasing；本任务明确声称support，因此该项是kill-switch；different-PID same-slot donor只进入
  appearance/binding对照，不进入该门禁；
- wrong text显著改变support分布，而text-only constant不能复现图像/遮挡分组；
- wrong mask/channel cycle不能与correct近似等价；
- valid region的flip raw feature cosine `>=0.95`、median q-JSD `<=0.02`；support-state top-1 consistency
  `>=95%`只在原始raw margin足够大的subset上要求；
- 每个有效slot centered effective rank `>=2`，不存在单一常量输出；
- PC-MBCLS吞吐不低于单CLIP的50%，teacher峰值显存与ReID sequential forward预算可共存。

任一核心输入不敏感、support对合成遮挡方向相反，或text-only/static强对照复现sample binding，
当前B2 teacher即`NO-GO`。不得把只通过appearance同PID指标当作语义门禁通过。

#### B2-SI 128图封板与B2-S全量审计冻结

B2-SI的128图CPU复核已在同一冻结代码上完成：639个valid target slots、2,556个full-image variants，
全部12项gate PASS。macro target `q_visible`在0/25/50/75%遮挡下为
`0.51104/0.48925/0.47765/0.46990`；五slot target Spearman均为正且最低为torso的`0.482`。
五slot的target 0→75%下降、target−non-target下降、target−global下降均逐类为正且PID-cluster
95% CI不跨0，macro target−global=`+0.03570`。因此当前PC-MBCLS读出通过小样本语义局部化门禁，
授权B2-S teacher-only审计，但不授权训练。

B2-S执行定义冻结如下，后续不能按结果改target分配、材质、阈值或donor规则：

1. **数据、target与donor先验封板**：使用official Occ-Duke全部15,618张train图；先按
   `SHA256(relative_path || seed)`对图像排序，再依次从该图valid hard-owner slots中选择当前全局计数
   最小者，tie由hash对应的cyclic slot order打破。该full-split target map必须在任何8图smoke之前
   生成并记录SHA；8图只能从该map中选覆盖五slot的样本，禁止在8图子集上重算target。分配只读取
   path与pose valid，不读取CLIP、ReID或结果。全部五slot的base feature仍由同一次PC-MBCLS前向产出，
   高成本反事实只作用于唯一target。
2. **连通嵌套遮挡几何**：禁止hash-randperm散点腐蚀。所有坐标定义在aspect-letterbox前的384×128
   RGB/mask上；以target hard support bbox为范围，由`SHA256(path || slot || seed)`一次性选择
   `top/bottom/left/right` sweep方向，使用与bbox同宽或同高的轴对齐矩形从该侧连续增长。25/50/75%
   cut position按矩形与target support的实际交集首次达到对应比例确定，三个矩形严格嵌套；realized
   overlap误差上限为新增最后一行/列所含support比例，不再声称one-pixel exact。该定义测的是连通
   occlusion，而不是salt-and-pepper corruption。
   full pose-only阶段还必须逐图逐level计算矩形与target及其余四slot hard support的交集，并报告
   `target-support covered / rectangle area`。每个样本、每个level都必须满足target交集严格大于任一
   单个non-target slot交集；否则记为occluder localization construction failure，不得进入CLIP审计。
   上述交集比例按slot和sweep方向分别报告P50/P95，禁止用macro掩盖arms bbox跨torso等几何泄漏。
3. **non-overlap位置控制**：一个14×14 CLIP patch反映射为
   `ceil(14*384/224)=ceil(14*128/75)=24`个pre-image像素；对最大75%遮挡矩形，穷举content box内
   可平移位置，要求与target的24-pixel dilation pixel product严格0，优先选择normalized y-center
   最接近者，tie按坐标字典序。较低level保持同一anchor和sweep方向，只改变矩形长边，因而control也
   严格嵌套。overlap/control使用完全相同的矩形尺寸与材质tensor，只改变粘贴位置；必须报告可行率、
   mean/P95 normalized-y error，正式门槛为insufficient fraction=`0`、mean`<=1/8`、P95`<=2/8`。
4. **三种冻结occluder材质**：`CLIP-mean`使用反归一化前固定RGB mean；`random-texture`只由
   `path/slot/material-seed`生成一次覆盖最大矩形的随机场，三个level读取该同一tensor的嵌套crop，
   按target bbox逐通道mean/std匹配并以固定`sigma=1.5 px`低通；`different-PID wrong-slot CutMix`
   使用不同PID、与target不同且valid的固定cyclic donor slot，将其bbox保持宽高比letterbox到最大
   occluder矩形。不同PID同slot可见内容不能作为support kill-switch，它只保留在sample-specific
   appearance/binding对照。
5. **donor冻结**：appearance/binding的same-slot donor与support的wrong-slot occluder donor分别建图。
   二者均优先同camera，并按增强后area ratio、normalized y-center、pose confidence的L1距离选最近者；
   wrong-slot donor依固定cyclic slot顺序选择第一个valid非target slot。必须different-PID、不同path、
   无fixed point，tie按relative path；matching只用pose/geometry，不用CLIP/ReID。full target/donor map、
   area/y/conf差及增强后mask IoU必须在8图前pose-only封板并记录SHA。
6. **wrong RGB/mask/text**：wrong RGB包含`donor RGB+donor mask`与`donor RGB+recipient mask`两臂；
   wrong mask固定recipient RGB，使用different-PID且area/y/conf matched的donor mask，要求增强后IoU
   median`<=0.30`、P95`<=0.50`；wrong text同时包含slot phrase cycle与visible/occluded state-order
   inversion；两者都必须固定PC-MBCLS visual feature后重新与替换后的text prototype计算logits，禁止
   roll已有`[slot,state]` logits冒充wrong text。另保留四个非identity slot cycles、同步置换
   mask/slot/text后的inverse-map equivariance、text-only constant、mass-matched uniform/fixed bands及
   PID-disjoint image-only cluster强对照。
7. **配对视图与flip**：base以aspect-letterbox为主几何；同图轻视图只使用冻结的brightness/contrast
   扰动，不改变pose geometry。flip单独做RGB水平翻转、mask空间反映射和slot固定映射；left/right
   已在coarse slot内合并。所有paired view seed由path固定，禁止重复抽样挑结果。
8. **统计单位**：所有均值同时报告image-level point estimate和PID-cluster bootstrap 95% CI，主裁决
   只用PID-cluster CI；bootstrap固定seed=`20260718`、repeats=`1000`。每slot单独报告，不允许macro
   掩盖单slot失败。centered effective rank明确计算归一化PC-MBCLS visual feature `v_c`，不对总和为1
   的二分类`q`误算rank；`q`另报variance、JSD、entropy与margin。
9. **support kill-switch**：三种真正occluder材质都必须逐slot满足overlap Spearman lower bound`>=0.2`，且0→75%
   target下降相对non-overlap的paired PID-bootstrap CI严格`>0`；target下降还必须逐slot高于同图
   non-target及official global CLS。任一材质或任一slot方向反转即当前B2-S `NO-GO`，不得用另外两种
   材质平均救场。same-slot donor不要求`q_visible`下降，只进入下一条sample binding比较。
10. **binding门禁**：same-image/same-slot的raw feature cosine必须显著高于different-PID/same-slot和
   same-image/wrong-slot，两个paired PID-bootstrap CI均`>0`且标准化效应`>=0.2`；correct-view q-JSD
   必须显著低于RGB-mask mismatch与low-IoU wrong-mask。wrong slot/text不能与correct近似等价，
   text-only/static不能复现遮挡响应。
11. **equivariance与非退化**：valid slot flip feature cosine`>=0.95`、median q-JSD`<=0.02`；仅在
    base `abs(q_visible-0.5)>=0.10`的样本上要求support-state consistency`>=95%`。每slot visual
    feature centered effective rank`>=2`，并报告q跨样本方差，防止常量teacher过门。
12. **GPU效率边界**：只在static、full pose-only feasibility与8图CPU contract全部PASS后启用唯一
    4090。official CLIP与
    PC-MBCLS使用同一FP16/no-grad/common microbatch，20次warmup后计100次并同步；报告images/s、
    相对吞吐、峰值allocated/reserved显存和CLIP常驻权重。PC-MBCLS吞吐必须`>=50%` official CLIP；
    以已测clean D0峰值加CLIP常驻权重和2 GiB安全余量核算sequential teacher→student预算，超过
    24 GiB或发生OOM即效率NO-GO，不通过改batch64救场。
13. **执行边界**：先做connected-rectangle/material/donor static exact；再对全部15,618张图做纯
    pose-only target/donor/non-overlap feasibility并冻结map SHA；两者PASS后才从full map取8图CPU
    CLIP contract。三步全部PASS后才首次启动全train teacher-only GPU审计。全量PASS仍只授权Phase 0C
    single-stage Semantic TAPF的实现与preflight，不直接授权120 epoch；任何训练前仍需新design、
    独立代码审查和全部门禁。

#### B2-Sv1封板边界与B2-Sv2重新预注册

B2-Sv1 full pose-only已按上述冻结门禁自然结束并裁决FAIL。该结果只关闭“bbox connected rectangle +
24px y-matched non-overlap + geometry-nearest low-IoU mask”这一组反事实构造，不关闭B2-SI已经通过的
PC-MBCLS slot-support readout，也不关闭CLIP校准TAPF。v1不得改阈值、换方向、重复运行或直接进入
8图CLIP；其FAIL map不得作为正式teacher map复用。

B2-Sv2把任务明确改名为**slot-evidence deletion**，不再声称physical occlusion。它是独立新协议，
执行前必须完成static与full pose-only feasibility；当前只冻结定义，不授权实现、CLIP、GPU或训练：

1. **只继承正交通过的target/augmentation先验**：从v1 FAIL map中只提取
   `relative_path/image_sha256/pid/camid/valid/target_slot/augmentation`形成独立submap，并在任何v2
   实现前记录SHA与15,618 record count。v2必须从official data和exp386 artifact重新生成同一submap并
   逐字节exact；v1的bbox/control/donor/IoU字段一律不得进入v2。
2. **support-clipped方向前缀**：对target hard support按原path/slot/seed选定
   `top/bottom/left/right`。top/bottom以y升/降序、left/right以x升/降序，tie依次用正交坐标与
   `(y,x)`字典序打破；level `l∈{0.25,0.50,0.75}`固定取前
   `ceil(l*N)`个384×128 pre-image support pixels，并要求每个target的`N>=4`。三个集合必须严格
   嵌套、selected始终是target support子集、与全部
   non-target hard support pixel product严格0、count误差`<=1/N`，全15,618图禁止construction skip。
   该定义测的是“删除slot内部视觉证据”，不是模拟矩形遮挡。
3. **材质角色拆分**：CLIP-mean与同一max-level deterministic blurred random texture是support-evidence
   deletion主臂，继续承担`q_visible`单调下降门禁。random texture对每个
   `relative_path/target_slot/material-seed`只在384×128坐标生成一次冻结的full/max-level field，固定
   `sigma=1.5 px`低通；各level必须按第2条冻结的ordered support coordinates读取同一位置的值，不能
   重新采样、缩放或按level重做统计。lower levels因此严格复用同一texture的冻结前缀。
   different-PID wrong-slot人体纹理改列semantic-mismatch/binding臂，不要求`q_visible`单调下降，
   不得用它误判一个仍能看到清晰人体纹理的正确support teacher。若以后加入第三种deletion材质，
   必须是预先验证的non-person/background texture，并另开单变量协议。
4. **删除不可实现的背景平移control**：不再要求24px dilation外的同尺寸/y-matched矩形。主定位仍报告
   target readout相对同图四个non-target readout与official global CLS的下降；所有readout共享完全相同
   base/corrupted RGB tensor，不能各自重做扰动。
5. **共享RGB的2×2 DID**：对每个target和四个nonidentity slot cycles同时计算；固定slot编号
   `0..4`，第`k`个cycle明确定义为`w=(t+k) mod 5, k=1..4`，不允许按结果另选wrong slot。
   `base/corrupted × correct/cyclic-wrong mask`。令
   `D=[q_v(base,M_t)-q_v(corrupt,M_t)]-[q_v(base,M_w)-q_v(corrupt,M_w)]`；四个cycle、五slot的
   PID-cluster 95% CI都必须严格`>0`。这直接检验“正确anatomical binding比错误binding更依赖被删除的
   target证据”，取代不可实现的背景位置control。wrong mask主定义因此是same-image hard-owner
   nonidentity cycle；DID始终固定target slot的support text prototypes，只替换spatial mask，禁止把
   mask与text一起cycle后冒充wrong binding。same-slot donor只作appearance hard negative，高IoU只报告、
   不再设置low-IoU kill。
6. **patch-space严格对照**：pixel hard-owner exact0不能代替CLIP patch检查。自然coverage路径逐slot/cycle
   报16×16 token-grid（patch size=14px）的overlap、mass、nonzero patch count。对任一coverage
   `c`，先定义`P_c`为mask经`16×16`average pooling后的patch coverage，
   `K_c=count(P_c>0)`；correct target与cyclic-wrong分别得到`K_t/K_w`，再固定
   `K=min(K_t,K_w)`。各自按coverage值降序排列，coverage相同按patch坐标`(row,col)`升序打破tie，
   选中的K个patch置binary 1、其余置0。由此两臂具有相同K与
   uniform log-prior。只乘coverage标量不能叫mass matching，因为PC-MBCLS的归一化log-prior对全局scale
   不变。自然coverage DID和top-K DID必须分开报告，不可混合挑优。
7. **eligibility与不静默跳过**：四cycle分别要求target及wrong slot均valid且patch K>=1；不满足者只在
   对应cycle记ineligible，逐slot报告images、unique PID及每种skip reason。不得更换target或只保留有利
   cycle；full pose-only必须逐slot/cycle至少保留`500` images和`100` unique PID，才授权8图CPU
   CLIP contract。
8. **support kill-switch**：CLIP-mean与random-texture分别逐slot要求
   `Spearman(overlap,-q_visible)`的PID-cluster bootstrap 95% CI lower bound `>=0.2`；0→75%
   correct-mask drop的PID CI严格`>0`，且correct drop减去同图non-target drop、correct drop减去同图
   official-global drop这两个paired PID CI也都严格`>0`。自然coverage DID与top-K DID都作为主门禁，
   必须对每个`material × target-slot × cycle(k=1..4)`分别满足PID-cluster 95% CI严格`>0`；两条DID
   分开报告，不允许互相替代。任一主材质、slot或cycle方向反转即B2-Sv2当前teacher定义NO-GO，
   不得跨材质/cycle平均救场；但该裁决仍只关闭B2-Sv2这个teacher定义，不扩张成CLIP–TAPF总路线
   NO-GO。wrong-slot semantic-mismatch只报告feature cosine/JSD与correct binding差，不进入visibility
   monotonic kill-switch。
9. **执行顺序**：先冻结target/augmentation submap SHA，再做纯synthetic static exact；随后只做一次
   全15,618 pose-only feasibility，PASS后才从full map确定覆盖五slot/四cycle的8图CPU CLIP contract。
   三者全部PASS才可请求唯一4090做full teacher-only审计。任何teacher PASS仍只授权Phase 0C
   single-stage Semantic TAPF实现/preflight，不直接授权120 epoch或semantic multi-stage。

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
10. B2-Sv1历史路径的wrong RGB/mask/text donor均无fixed point、different-PID；其wrong mask按实际
   增强后面积/y/conf重新匹配并报告IoU，median IoU `<=0.30`、P95 `<=0.50`。B2-Sv2明确覆盖该旧门禁：
   wrong mask改为same-image四个hard-owner nonidentity cycles并执行16×16自然coverage/top-K DID；
   same-slot donor只需different-index/path/PID、same-camera priority与geometry matching exact，IoU只报告，
   不再要求low-IoU；
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

`PHASE 0B2 B2-SI SEALED-PASS / B2-Sv1 SEALED-FAIL / B2-Sv2 DESIGN-FROZEN NO-START /
FORMAL TRAINING NO-START`。

hard-owner ontology、slot-conditioned support task与PC-MBCLS readout已经分别通过128图CPU门禁；
这排除了“PC-MBCLS只响应全图扰动”的解释，但尚未证明teacher对真实RGB/mask/text binding和
slot-evidence deletion稳健。B2-Sv1的FAIL只封板其connected-bbox/non-overlap/nearest-low-IoU
反事实构造，不能覆盖上述已通过证据，也不能否定CLIP语义校准TAPF。下一步只允许提取并双源复算
target/augmentation submap，然后实现B2-Sv2 pure synthetic static；二者完成独立审查前不得启动
full pose-only、8图CLIP或4090。未来B2-Sv2 full pose-only PASS后才允许8图CPU CLIP contract，
其后所有teacher门禁PASS才可请求唯一4090全train teacher-only审计；即使该审计PASS，也只授权
Phase 0C single-stage机制实现，不直接授权120 epoch训练。
