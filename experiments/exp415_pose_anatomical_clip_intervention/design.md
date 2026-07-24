# 实验 exp415：PACIT（Pose-Anatomical CLIP Intervention Training）

## 当前状态

`REVISION-3 / PREFORMAL RUNNERS REVIEW PASS / REMOTE MECHANICAL GATES NEXT /
ORACLE NO-START / E120 NO-START`

第一版因CLIP自证、缺真正CLIP-only和训练合同不完整被`BLOCK`；revision-2虽切断CLIP→verifier依赖，但复审又发现
arm-specific validity、短数组也能计算interaction、逐图难度未匹配、canonical→training view断裂与强control
定义性偏弱。本版覆盖前两版；前两版均未执行oracle、未训练、未占用CUDA。

exp415不复活exp408--414的owner、prefix、MST、continuous-region、proxy、relation KD、hard-pair、
gradient-routing或donor transport。

## 研究对象与主张边界

PACIT构造同图可观察像素干预：

> 实例pose与canonical人体anchor分别产生位置proposal；同一个冻结CLIP颜色选择器挑选遮挡形状；完全不读取CLIP
> 的CIELAB连通颜色、pose anatomy与sealed D0裁决器验证mask是否形成身份安全的anatomical color-evidence
> removal。

像素操作仍是structured erasing；blind evaluator不能判断“服饰归属”，因此只声称
`anatomical coherent color evidence`，不把红/棕像素自动写成衣物，也不声称新遮挡原语。C类资格只可能来自
instance-pose proposal与CLIP selector在独立裁决上的matched factorial交互，以及最终ReID增量。

## 严格变量与每图平衡层

oracle的第`i`个hash样本固定`anchor_index=i mod 5`，因此512图的五层频率为102或103，不由pose、CLIP、RGB、
PID、camera或结果决定。五层为：

`head / upper_torso_arms / lower_torso / upper_legs / lower_legs_feet`。

全量15,618资产使用
`SHA256("exp415-pacit-train-v3\0"+relative_path) mod 5`，规则在512结果前冻结。

两个因素：

- `P=1`：active anchor为该图对应pose slot的soft-field质量中心；
- `P=0`：active anchor为同层fixed normalized canvas中心
  `[(.10,.50),(.30,.50),(.48,.50),(.67,.50),(.86,.50)]`；
- `C=1`：同一个冻结CLIP centered-color-margin-drop scorer在该anchor的7个shape中选`(shape,color)`；
- `C=0`：在逐图冻结caliper内由relative-path hash盲选shape，不读取RGB、pose、CLIP score/label或blind颜色。

P=0准确名称是`canonical-anchor`，不是“无任何人体先验”；它不读取实例pose，检验的是instance-pose相对
canonical anatomical prior的必要性。

每图P+/P-各生成完整`5 anchors × 7 aspects=35`proposal，但selector只能看到预定active anchor的7个proposal。
七个aspect为`[.40,.60,.80,1.00,1.25,1.67,2.50]`。面积固定为`384×128` canvas的`6%`，integer误差
`<=1%`，alpha=`1.0`。proposal不得按pose coverage、CLIP、RGB或D0过滤、重排、截top-k。

无效pose anchor只记录诊断并回退同层fixed坐标；不得向Y注入arm-specific validity bit。四臂统一由选择后的pose
coverage裁决，所以相同mask不可能只因来源arm不同而一边强制失败。

四臂：

| arm | active proposal center | selector |
|---|---|---|
| `P+C` | instance pose | CLIP |
| `P-only` | instance pose | caliper-hash blind |
| `C-only` | canonical anchor | 同一CLIP |
| `neither` | canonical anchor | 同一caliper-hash blind |

P只改变active center；C只改变该center内的shape selector。全部臂共享area、aspect候选、fill、alpha、样本、
失败分母和后续双forward。

## 固定资产与宿主

- official Occluded-Duke train、frozen COCO-17 pose、OpenCLIP ViT-L/14、sealed D0 checkpoint均只读；
- canonical view固定bicubic `384×128`；
- asset oracle无训练宿主，只调用D0 `eval+no_grad`；
- 若asset GO，production四臂共同继承sealed zero-owner的普通等支持、全身份listwise强宿主；zero-owner不读取
  pose/CLIP，不是新增贡献。owner/prefix/MST/continuous-region不回来；
- Swin-Tiny、64张唯一图、P×K=`16×4`、seed和RGB-only eval保持。

## 唯一512图asset oracle

### 1. 固定样本与固定分母

对全部train relative path计算
`SHA256("exp415-pacit-oracle-v3\0"+relative_path)`，按`(digest,path)`升序取512。四臂必须有相同有序
row id；decode、pose、proposal、selector、match、颜色或D0失败均保留row并写`Y=0`。统计入口硬断言
`shape==(512,)`和四臂row id逐元素相同，禁止complete-case或短数组统计。

`arm_complete`只表示该臂的机械字段已经完整、finite并可审计；anatomy/color/identity科学门失败只令
`raw_y=0`，不得把`arm_complete`改为false。`match_edges`只表达四条预注册caliper边，最终四臂Y只能由统一
row accumulator生成；任何单臂或match失败共同置0，科学失败不能通过改变有效子集被过滤。

### 2. 编辑与fill

四臂使用同一relative-path hash播种的achromatic fill：

`gray=clip(.50+.08*N(0,1),.38,.62)`，RGB三通道相等。

十色bank不含gray，fill必须全部落入`unclassified`。mask外RGB byte-exact。

### 3. 冻结CLIP scorer

颜色bank：

`black/white/red/orange/yellow/green/cyan/blue/purple/brown`。

prompt ensemble与SHA冻结。标准whole-image CLIP encoder分别编码一张original和7张edited canonical RGB；
不使用pose crop、slot名、region feature或D0。模板平均后：

`margin(v,k)=cos(v,t_k)-mean_{j!=k}cos(v,t_j)`

`drop(c,k)=margin(v_original,k)-margin(v_edited_c,k)`。

C=1在`7×10`上取argmax，按`(aspect_index,color_index)`破并列。score function源码与static test必须在formal
前冻结；builder不得在外部重新定义margin。

### 4. 逐图caliper与C=0

先独立冻结P+C与C-only的C=1选择。C=0不读CLIP score，只在各自同center的另外六个shape中筛：

- mask面积差`<=1 pixel`；
- centroid normalized L∞差`<=0.01`（同center只允许integer rounding）；
- `abs(log(aspect_control/aspect_C1))<=log(1.25)`；
- clean→edited D0 cosine displacement差`<=0.010`；
- edited-vs-clean D0 true-PID CE变化差`<=0.25`；
- clean与两张edited的true PID均在D0 classifier top-5；
- control mask SHA必须不同。

在eligible中取
`SHA256("exp415-v3-caliper-blind\0"+path+"\0"+candidate_index)`最小者；没有eligible则该图该P-level
match失败，禁止放宽、跨层或reselect C=1。

跨P匹配比较P+C与C-only：

- area差`<=1 pixel`；
- aspect log差`<=log(1.25)`；
- D0 displacement差`<=0.010`；
- D0 CE变化差`<=0.25`；
- clean与两臂edited top-5保留。

跨P不匹配centroid，因为instance center与canonical center正是P处理。还必须直接对P-only与neither执行同一
跨P caliper，不能用三角链替代。四条直接边`C|P1 / C|P0 / P|C1 / P|C0`全部通过时才有
`quartet_matched=1`；否则该图四个factorial Y统一置0。matched quartet仍按512分母，且完整率必须`>=90%`。

### 5. 完全独立blind evaluator

blind evaluator不得import CLIP，不接收checkpoint、embedding、prompt、score或selector label。

1. 固定sRGB→CIELAB与十个prototype/radius，其他像素为`unclassified`；
2. 用选择后的mask在五槽soft field上找覆盖最大的target slot；
3. 对每个颜色计算：
   - `presence`：颜色占target-slot soft mass；
   - `capture`：mask捕获该颜色总soft mass；
   - `purity`：mask内target-slot soft mass中的该颜色比例；
   - `component`：mask∩target hard-support中该色最大4连通分量占被捕获该色像素比例；
   - original→edited的absolute/relative drop；
4. blind evaluator内部取归一化最弱项最大的颜色，四臂同规则。

门：

- `anatomy_valid`：覆盖最大槽必须等于结果前固定的active anchor；target coverage`>=.25`，其余有效槽
  coverage mean`<=.10`、max`<=.25`；
- `coherent_color`：presence`>=.10`、capture`>=.25`、purity`>=.20`、最大连通分量
  `>=32 pixels`且component ratio`>=.60`、absolute drop`>=.15`、relative drop`>=.80`；
- `identity_safe`：clean与edited true PID均在D0 top-5，且D0 displacement位于该图8个同面积ROA的linear
  P50--P90。

唯一主结果：

`Y=anatomy_valid AND coherent_color AND identity_safe AND quartet_matched`。

不存在额外`d0_nontrivial`隐藏变量。

### 6. CLIP颜色agreement

blind-max Y保持四臂可比；另外独立检验C=1 selector颜色：

- P+C的selector color必须等于blind color，固定512分母agreement rate`>=60%`；任何selector、blind或match
  失败都按不agreement，禁止complete-case；
- 固定`(color_index+1) mod 10`的text-shuffle agreement至少比correct低10 percentage points。

该门不使用CLIP score，只检查CLIP输出标签是否与独立像素事实一致；不通过时只能称hard selector，asset NO-GO。

### 7. 强control

在P+同一active 7-shape pool内，先用P+C为reference应用完全相同的area/aspect/D0 displacement/D0 CE/top-5
caliper和ROA P50--P90 identity-safe门，再选择：

- `D0-severity-matched hard`：eligible中D0 displacement最大；
- `raw-color+D0-safe`：eligible中blind coherent-color score最大。

P+C自身mask必须保留在两个strong-control的eligible集合中。若raw-color或D0-hard选择同一mask，必须如实记录
机制等价；P+C不能强迫control取次优。P+C-vs-raw与P+C-vs-D0各自原子地产生pair-match bitmap，任一侧失败时
该pair两侧共同置0，固定512分母；两条strong pair的matched完整率都必须`>=90%`。这样P+C胜strong control
不能靠无alternative、P90外或identity-unsafe。

### 8. factorial统计与唯一GO

在固定512图计算：

- `ΔC|P1=mean(Y_PC-Y_Ponly)`，要求`>=.08`；
- `ΔP|C1=mean(Y_PC-Y_Conly)`，要求`>=.08`；
- `ΔC|P0=mean(Y_Conly-Y_neither)`，要求`>=.04`；
- `ΔP|C0=mean(Y_Ponly-Y_neither)`，要求`>=.04`；
- `I=mean(Y_PC-Y_Ponly-Y_Conly+Y_neither)`，要求`>=.04`。

salt固定10,000次paired image bootstrap；同一重采样索引四臂。前两项与I的单侧95% lower bound必须`>0`。

`JOINT ASSET ORACLE GO`还同时要求：

1. 512/512完整row、四臂各7 active proposals、8 ROA、provenance与失败码；
2. matched quartet至少461/512；
3. P+C-vs-raw与P+C-vs-D0 pair-match各至少461/512；
4. P+C Y至少359/512，且五个预定slot的P+C success各至少64；
5. factorial P+C Y严格高于另三factorial臂；对两个strong control分别使用其pair accumulator产生的
   pair-specific P+C reference数组，reference Y率严格高于对应control；
6. clean top-5、P+C edited top-5及两者交集均报告；edited相对clean下降`<=5 points`且交集至少90%；
7. selector-color agreement与text-shuffle差通过；
8. 全部finite，SHA一致，optimizer update=`0`、checkpoint=`0`。

任一失败即`EXP415 PACIT ASSET NO-GO / FORMAL E120 NO-START`。正式oracle结果后不得改门或重跑。

## oracle前执行门

1. 本地pure-CPU core/scorer/row-accumulator/failure-injection static contract；
2. 只读全15,618 pose geometry census，只报告anchor-valid、proposal count/area，不加载CLIP、不计算Y；
3. 独立`preflight-smoke` namespace固定8图，只测真实decode/shape/finite/CLIP/D0/cache/result回读，不计算
   selector优劣或oracle rate；
4. 三路子agent复审只在致命bug、变量混淆、旧机制同构=`0/0/0`时放行；
5. fresh formal source、独占4090、正式oracle output不存在，才写once-only seal。

smoke不得触碰正式oracle started/result/output。

## 512 GO后的全量资产

用同一frozen builder对15,618图一次性生成四臂、两个强control与共同manifest：

- 每图保留全部arm记录；
- 只有四factorial臂与两个strong control全部完整、逐图caliper和identity-safe均通过才设
  `common_intervention_valid=1`；
- 任一失败时所有训练臂该图都使用clean second-view NOOP，禁止drop、换图或改变P×K；
- 全量common-intervention-valid率仍须`>=70%`，否则`FULL-ASSET NO-GO / E120 NO-START`；
- manifest在任何训练前封存SHA。

## actual-view与batch64训练合同

为避免canonical severity在训练view中漂移，所有双forward臂关闭Random Erasing、padding与random crop。只保留
horizontal flip，并在全量资产阶段预先对`canonical`与`hflip`两种orientation分别重跑六臂的blind
anatomy/color、D0 displacement、D0 CE、top-5、ROA P50--P90与四条/strong caliper。两种orientation全部通过
才设该图`common_intervention_valid=1`；否则六臂永久clean-NOOP。

训练时只按`(seed,epoch,iteration,ordered path)`在这两个已封存orientation中选择，使不同arm得到同一方向与
active bitmap；不在线重跑pose、CLIP或D0。

共同宿主为zero-owner普通listwise。每iter仍为64张唯一源图、16×4：

1. clean forward，反传`0.5*L_host_clean`，保存stop-gradient normalized global；
2. optimizer尚未step时second-view顺序forward，反传
   `0.5*L_host_second + 0.5*mean(1-cos(z_second,sg(z_clean)))`；
3. 一次GradScaler step/update、一次optimizer/scheduler step。

先自然e120运行`double-view clean-pair` seed1234，再运行P+C seed1234。P+C必须：

- 相对double-view clean-pair至少`+0.5 mAP point/+0.5 R1 point`；
- 同时超过sealed clean D0、sealed zero-owner和当前历史显示best
  `exp413 correct=59.3 mAP/70.8 R1`至少`+0.5/+0.5`（按日志一位小数算术，不接受round持平）。

否则`PERFORMANCE NO-GO`，其余controls不启动。通过后串行运行P-only、C-only、neither、
D0-severity-matched、raw-color+D0-safe；P+C相对每个都须`>=+.5/+.5`。text-shuffle只保留为asset
颜色agreement诊断：颜色label循环不会改变`7×10`全局argmax的mask，因此不得伪装成formal训练control。

seed1234全部通过后，只为P+C与最强matched control补seed`5678/91011`。最终要求：

- 三seed每个mAP与R1差都严格为正；
- 3-seed mean差至少`+.5 mAP/+.5 R1`；
- seed固定10,000次per-query paired bootstrap的mAP贡献与R1 indicator差95% lower bound均`>0`。

满足后才记`PERFORMANCE + POSE×CLIP ATTRIBUTION GO`。

## 失败解释

1. 不胜raw-color：CLIP不如简单像素启发式；
2. 不胜D0 matched-hard：只是generic hard occlusion；
3. agreement失败：CLIP只是黑盒difficulty selector，不是颜色语义selector；
4. interaction失败：instance pose与CLIP无联合必要性；
5. full-asset或actual-view有效率失败：512资产不可外推到生产；
6. 训练不胜强宿主：资产可识别但无ReID价值；
7. 任一结果都不得通过调prompt、caliper、fill、loss或跨臂拼点救回。
