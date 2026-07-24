# exp416 PC-NEC：Pose-Conditioned CLIP Negative-Evidence Certification

## 当前阶段

`D0 SIGNAL DIAGNOSTIC IMPLEMENTATION PASS / EFFECTIVENESS RED TEAM BLOCKED 2B/2H /
CONSUMER-ALIGNED GATE DESIGN NEXT / REMOTE FORMAL NO-START / TRAINING NO-START`

现有runner只实现“sealed D0固定bank上是否存在OpenCLIP region增量”的无训练signal diagnostic。
代码/统计实现回归未发现致命bug，但独立效能红队发现两个训练因果BLOCKER：future宿主是zero-owner而非D0，
且top-20 image bank没有覆盖future PK64全部负身份pair。consumer-aligned residual gate与全PK64 pair域
闭合之前，不创建formal asset、不运行真实fuel、不进入PK64或e120。

## 动机

exp395--415已经排除两类反复失败的对象：

1. 在同PID合法support内部加权、排序、prefix或连边，语义可被PID loss绕过；
2. 用外部CLIP坐标、proxy、teacher residual或当前PACIT像素selector替代不可观察的缺失身份。

single-image support incomplete并不只意味着“需要补全”。它还给出一个不对称事实：

> 部分观测通常不足以证明同一身份，但双方被pose估计为共同可用的解剖槽中，一个可靠语义矛盾可能排除错误身份。

PC-NEC因此不预测缺失区域、不生成完整feature，也不让CLIP定义identity。它只研究pose对齐的共同可用槽中，
CLIP是否相对raw pixel与student part baseline提供不可替代的跨身份负证据。

## 核心假设

在一个完全由sealed D0预先冻结的候选bank中，instance-pose对齐的region CLIP矛盾分数能够同时：

1. 区分真匹配与hard impostor，并在AUROC/AUPRC上显著胜最强matched control；
2. 在不改变candidate bank的诊断排序中改善D0的候选内mAP/R1；
3. 其收益不能由canonical-location crop、raw color、student part、slot shuffle或wrong RGB解释；
4. 覆盖大多数query，而不是依靠稀疏post-hoc caliper。

该假设只构成必要的signal kill-switch。即使全部成立，也不能直接说明zero-owner宿主还存在可利用残差，
更不能授权训练期negative-evidence certificate。

本审计中的“CLIP”严格指冻结OpenCLIP image encoder的region visual representation；不读取text encoder、
prompt或语言相似度。正结果只能支持`OpenCLIP visual region evidence`相对当前matched controls具有增量，
不能声称语言对齐语义不可替代，也不能外推为胜过所有非语言visual foundation encoder。

## 与旧机制的边界

- 不是exp409 hard-pair mining：所有arm共享相同完整bank，语义分数不能选pair、删pair或改变bank；
- 不是exp411--414：不在同PID support内加权、排序、prefix或连边；
- 不是exp401--404/410：CLIP不输出identity坐标、proxy、hidden operator或final feature；
- 不是exp408：不蒸馏中层relation；
- 不是exp412：不路由或转移梯度预算；
- 不是exp415：不编辑像素、不选择mask、不使用颜色prompt；
- 不是test-time part matching：fuel audit的诊断重排只用于kill-switch，未来正式eval仍必须global RGB-only。

KPR/BPBreID共同可见part matching、PAT-CSL跨ID视觉邻居、Instruct-ReID语义margin和普通part-aware metric
learning是强近邻。当前只能主张
`fixed full candidate bank + pose-estimated common-available slots + train-only existential negative certificate +
global-only retrieval`的整体窄差分。

## Fuel audit技术方案

### 1. 固定样本与PID折

- 数据只使用official train与冻结pose资产，不读取official query/gallery测试标签；
- query先按固定path hash排序；只有存在至少一个跨相机同PID图的query进入固定审计分母；
- 原始PID按固定salt hash分成5个互斥折，所有同PID图必须在同一折；
- 折只用于cross-fitting诊断组合系数与PID-bootstrap，不能改变candidate bank或arm分数。

sealed D0已经在official train上训练，因此本审计只声称“冻结CLIP变量在固定hard bank中的相对燃料”，不把D0
候选内绝对指标写成未见PID泛化或论文性能结果。若需要论文级泛化证据，必须先通过consumer-aligned残差门，
再另立不接触测试集的held-out训练协议，不能把本审计放大。

### 2. 先冻结共同candidate bank

必须在加载pose、CLIP或任何semantic score之前：

1. 用sealed D0 global RGB descriptor计算距离；
2. 每个query保留全部跨相机同PID真匹配；
3. 按真匹配candidate camera频数，先给每个出现的camera stratum预留至少1个impostor，再对剩余
   `20-|camera support|`用largest-remainder分配quota；每个stratum内只保留D0最近的quota个跨PID impostor；
4. 写入有序`query/candidate/PID/camera/D0 distance` manifest与SHA；
5. 所有arm逐pair读取同一manifest；不允许semantic score改变候选、完整率或分母。

若query缺真匹配或任一camera quota没有足够跨PID候选，必须在读取pose/CLIP前使bank INVALID；禁止arm-specific
drop或退回camera-confounded候选。由此每个query内genuine/impostor的candidate-camera分布按预注册quota匹配，
不能把camera-pair shortcut解释成身份负证据。

### 3. 共同可见槽

五个解剖槽沿用冻结pose定义。correct只使用query与candidate双方在同一实际RGB view上都机械有效的槽；每个槽的
RGB crop/mask、pose field与CLIP预处理必须来自同一原图，不允许raw cache与增强view混用。

在读取任何CLIP score前，CPU geometry census必须只根据pose/RGB尺寸为五槽各冻结一个固定归一化crop高宽，
并为canonical control冻结五个canvas center。correct与canonical-location CLIP使用完全相同的逐槽高宽、
resize、插值、归一化和输出分辨率；两者唯一位置差为`instance-pose center / frozen canonical center`。
crop center统一clamp到保证完整矩形位于图内的最近位置，不padding、不改变面积。

所有pose-location与canonical-location arm共享由correct pose机械validity得到的同一slot availability bitmap；
canonical arm可以读取该bitmap以匹配分母，但其crop函数不得接收pose坐标、heatmap或实例center。由此P因素只表示
实例位置，不混入crop面积、slot数或UNDECIDED率。CPU census必须在CLIP加载前证明两位置arm逐row的crop面积、
shape、availability与输出tensor shape exact。

共同可见集合为空时，该pair统一输出`UNDECIDED`，所有依赖该集合的arm保持同一pair记录，禁止重新选candidate。
正式metric中`UNDECIDED`固定记反证能量`0`并保留在原pair/query/PID分母，同时单独报告coverage；不得删除该pair
或只在complete-case上计算AUROC/AUPRC/mAP/R1。
visibility的具体机械定义、crop边界和最小像素门必须在实现前冻结并由static failure-injection覆盖；本设计阶段
不事后发明confidence阈值。

### 4. Correct负证据

对每个共同可见槽`s`，从同一冻结OpenCLIP image encoder得到region embedding：

`e_s(q), e_s(g)`。

槽矛盾为非负量：

`c_s(q,g) = 1 - cosine(e_s(q), e_s(g))`。

图对的存在性负证据为：

`E(q,g) = max_{s in common(q,g)} c_s(q,g)`。

fuel audit不训练certificate head；它只检验该固定分数是否在真匹配与跨ID impostor之间形成增量。若后续训练获
授权，`max`可由冻结的logsumexp/MIL实现，但不得在看到audit结果后临时选择聚合式。

### 5. Matched controls

所有control使用同一query/candidate bank和固定pair分母：

1. `pose-only/raw-color`：相同instance-pose槽，以CIELAB histogram/连通颜色距离形成存在性矛盾；
2. `pose-only/student-part`：相同槽，固定使用sealed D0 eval输出的最后一个Swin normalized feature map
   `featmaps[-1]`；把同一输入像素slot rectangle以area interpolation缩到feature-map尺寸，按fractional
   mask作加权均值并L2 normalize，槽矛盾固定为`1-cosine`。禁止选择其他stage、hook、BN feature或层组合；
3. `canonical-location CLIP`：它不是严格pose-free/CLIP-only，因为仍共享correct的pose-estimated
   slot availability；它只消融instance pose center。crop函数不读instance pose坐标，只使用五个
   frozen canonical center、相同固定crop高宽和同一CLIP距离；
4. `neither`：与`canonical-location CLIP`共享同一availability bitmap、逐槽固定高宽和frozen canonical
   center，只把CLIP距离替换为raw-color距离；不得退回不同面积的horizontal stripe；
5. `slot-shuffle`：保持region与分数分布，固定循环错配槽索引；
6. `wrong-RGB`：candidate region CLIP来自固定donor自己的同名pose槽。donor pool在读取CLIP score前只包含
   五槽全机械valid、camera与candidate相同、PID同时不同于query和candidate的official-train图；
   对`query path || candidate path || slot`的固定SHA256取模，在relative-path排序pool中选唯一donor。
   禁止fallback到query/candidate PID、按CLIP分数选donor或改变原pair分母；pool为空则该pair固定E=0并记
   donor-invalid；
7. `global-CLIP`：不分槽的whole-image CLIP距离；
8. `D0-only`：原冻结global距离。

任何control若实际读取了correct专属pose对应或改变bank，静态合同必须失败。最终比较使用最强control，不挑软对照。

### 6. 诊断指标

所有指标固定pair/query/PID分母并同时报告覆盖率：

1. 每个query在自己的固定bank内计算same/different PID AUROC与AUPRC，其中跨PID impostor固定为
   positive class；
2. 每个query的AUROC/AUPRC先在同一query PID内等权平均，再对全部query PID等权平均；
   该`PID-macro AUROC/AUPRC`是唯一GO主指标，pair-global与query-macro只作诊断，不参与裁决；
3. 只在相同固定bank内计算每query诊断AP与R1，再按“query内→query PID间”相同两级等权规则得到
   `PID-macro mAP/R1`；它们是唯一排序GO主指标；
4. 五槽各自pair数、共同可见槽数分布与`UNDECIDED`率；
5. correct相对每个control的paired差；
6. 以PID为单位固定10,000次bootstrap的95%下界。

诊断排序只用于测试燃料。不得把它用于official test、最终部署、论文主结果或后续checkpoint选择。

每个query bank内，D0 distance和各arm反证能量分别转换为`[0,1]`经验mid-rank；数值越大均表示越像impostor。
相同数值使用平均rank，不用path打破数值tie。固定组合为：

`S_lambda = (1-lambda) * rank(D0 distance) + lambda * rank(E)`。

诊断检索按`S_lambda`从小到大排序；若`S_lambda`相同，依次用原D0 distance与candidate relative path升序
tie-break。`lambda`候选集合唯一冻结为`[0.0, 0.25, 0.5, 0.75, 1.0]`。

每个arm独立但使用完全相同的五折cross-fitting规则：对一个held-out PID折，只在其余4折的PID上选择使
`PID-macro mAP`最大的lambda；若mAP并列，选R1更高者；仍并列则选数值最小的lambda。选定lambda无改动应用到
held-out折。最终只汇总五个held-out折的out-of-fold预测；禁止为correct改候选集合、目标或tie-break。
D0-only固定`lambda=0`，不参与选择。

“最强control”按每个主指标分别在完整out-of-fold输出上取数值最大的非D0 control；并列时按第5节control顺序
取最前者。AUROC、AUPRC、mAP、R1可对应不同最强control，必须分别记录arm名称。point-estimate gate 1和
gate 3等价于要求correct超过每一个非D0 control的相应门，而不是事后挑一个较弱control。bootstrap不把这个
sample-selected arm固定后冒充同时推断，而在每个replicate上取correct对全部七个control paired差的最小值。

### 7. PID-bootstrap唯一合同

- bootstrap单位固定为query的原始PID，而不是pair两端PID、query或pair；
- 设固定审计分母内有`P`个不同query PID；每次用PCG64有放回抽取恰好`P`个PID，某PID被抽中多次时，
  其全部query级metric按出现次数重复贡献；
- 每个PID内部先对其全部固定query等权平均，再对抽中的`P`个PID occurrence等权平均；
- out-of-fold arm score、lambda和每query AUROC/AUPRC/AP/R1在bootstrap前已经冻结；
  bootstrap不重选lambda、不重算candidate bank，也不把pair-global非线性metric冒充主estimand；
- 重复次数固定10,000；base seed固定`4161234`，每个`metric/control`名称按UTF-8字节SHA256前8字节取
  无符号整数并与base seed异或，作为独立PCG64 seed；
- one-sided 95% lower固定为10,000个paired delta的线性经验5%分位数；
- 不按fold、camera、slot或arm complete率分层，不丢`UNDECIDED`；
- 六个核心下界分别对应：
  1. 每个bootstrap replicate内，correct AUROC减七个非D0 control AUROC paired差的最小值；
  2. 每个replicate内，correct AUPRC减七个非D0 control AUPRC paired差的最小值；
  3. correct combined mAP减D0-only；
  4. correct combined R1减D0-only；
  5. 每个replicate内，correct combined mAP减七个非D0 control mAP paired差的最小值；
  6. 每个replicate内，correct combined R1减七个非D0 control R1 paired差的最小值。

每个bootstrap replicate都从同一抽样PID occurrence计算全部paired差；四个非D0下界使用
`min_k(M_correct-M_control_k)`，两个D0下界使用单独paired差。arm完整率或metric不可用必须使整个audit
INVALID，不能改为complete-case。

## 预注册GO门

必须同时满足：

1. correct AUROC与AUPRC各自相对最强非D0 control至少`+0.03`；
2. correct同bank诊断mAP/R1相对D0-only至少`+1.0/+1.0`；
3. correct同bank诊断mAP/R1相对最强control至少`+0.5/+0.5`；
4. 上述六个核心paired差的PID-bootstrap 95% lower bound均`>0`；
5. 至少80%的固定query存在可用共同可见槽；
6. 五个槽各自均达到预注册的最小pair/PID覆盖；具体绝对门必须由不读取CLIP分数的CPU geometry census先冻结；
7. correct的PID-macro AUROC、AUPRC、combined mAP与combined R1分别严格胜
   pose-only/raw-color、pose-only/student-part、canonical-location CLIP、neither、slot-shuffle、wrong-RGB与
   global-CLIP的对应主指标；
8. candidate manifest SHA、row order、pair count和每个arm完整率exact。

任一失败即：

`PC-NEC FUEL NO-GO / TRAINING NO-START / NO CANDIDATE`

禁止调slot、crop、visibility、CLIP layer、聚合式、系数、top-K或门限重跑。

## D0 signal GO后的consumer-aligned残差否决门

本节尚未实现，不构成当前formal或训练授权。现有D0 signal fuel即使GO，也只能允许实现并复审下面这一个
consumer-aligned无训练门；不能直接进入threshold、PK64或e120。

### 1. 消费者必须与future宿主一致

固定读取sealed zero-owner checkpoint的global RGB descriptor；D0只继续定义原candidate bank，不再代理
future consumer。所有correct/control共享同一bank、同一zero-owner descriptor和同一分母。

在同一bank中，对每个anchor与candidate identity计算zero-owner身份聚合相似度，并以负身份是否违反
genuine identity的全排序约束定义zero-owner residual error。correct certificate对这些真实残余误排序身份的
富集必须严格胜全部七个matched controls，且raw-PID cluster bootstrap单侧95%下界均`>0`。否则说明证书只在
排斥zero-owner已经容易区分的负身份，直接`PC-NEC TRAINING NO-START / NO CANDIDATE`。

### 2. 全PK64 pair域必须闭合

预先冻结与正式recipe一致的确定性`P×K=16×4` batch序列，但不更新optimizer、不写checkpoint。对batch内全部
`64×64`有序图对，都必须能从同一份按图冻结的pose availability与OpenCLIP region embedding动态计算`E/v`；
不允许只查top-20 bank、missing、fallback、重新选pair或改变batch。

此处`P`唯一指batch内16个身份，每个anchor恰有15个负身份。raw-PID五折cross-fitting冻结`theta`后，完整OOF
必须同时满足：

1. genuine identity被任一same-PID support误证的anchor-level family-wise比例不超过`1%`；
2. batch内负身份coverage `|C_q|/15`的PID-macro mean至少`30%`；
3. `C_q`和`U_q`同时非空的anchor比例至少`80%`；
4. 全部实际pair证书来源完整率`100%`，fallback/missing为`0`。

任一失败即`THRESHOLD NO-GO / PK64 NO-START / NO CANDIDATE`。原top-20 bank上的coverage不得冒充该门。

### 3. exact future梯度必须与独立全排序下降方向一致

在sealed zero-owner descriptor上实例化下文唯一`L_cert`，只做一次无更新backward。令`g_cert`为其对global
descriptor的梯度，`g_rank`为独立zero-owner全身份排序目标对同一descriptor的梯度，预注册归一化对齐：

`A = <g_rank, g_cert> / (||g_rank|| ||g_cert||)`。

correct相对每个certificate control的PID-bootstrap单侧95%下界都必须`>0`，并保留所有不利anchor/PID。
禁止扫描epsilon、loss权重、temperature、margin或threshold。梯度nonzero本身不算通过；只有残差富集、
identity安全、全pair覆盖和梯度对齐同时通过，才允许再次复审真实训练合同。

## consumer-aligned门通过后的条件训练对象

本节不构成当前授权。只有consumer-aligned门全部通过后才允许另立实现与真实训练合同：

只有D0 signal、zero-owner残差富集、全PK64 pair coverage、identity安全与梯度对齐全部通过，才允许下面这一个
最小数学对象进入下一轮实现复审；不能替换成triplet调margin、pair mining或relation KD。

连续AUROC/AUPRC通过不自动证明存在可训练的二值certificate阈值。consumer-aligned门必须冻结`theta`并同时
检查同PID false-certificate、真实PK64负身份certificate coverage及`C/U`非空率；未过门仍为
`TRAINING NO-START`。

该门必须以raw PID五折cross-fitting运行：每个held-out折的`theta`只在其余四折选择，固定选择满足
anchor-level genuine-identity family-wise误证率`P(V_q,y_q=1)<=1%`的最低阈值，再无改动应用到held-out折。
完整OOF必须同时满足：

1. genuine identity被任一same-PID support误证的anchor比例不超过`1%`；
2. 每anchor被证实的batch内负身份比例`|C_q|/15`的PID-macro mean至少`30%`；
3. `C_q`和`U_q`同时非空的anchor比例至少`80%`。

任一失败即`THRESHOLD NO-GO / PK64 NO-START`，禁止把pair-level false-certificate冒充identity-level安全性。

对PK batch中的anchor `q`及每个身份`j`，令`G_j`为该身份全部K个图。为避免重现exp411--414的pixel-view
混淆，certificate loss的student descriptor必须来自与离线`v_qg`相同的deterministic official RGB view；
宿主可继续使用随机增强view，但二者共享student参数且certificate branch不得读取另一套像素。student global
身份集合分数为归一化logmeanexp：

`z_qj = tau_g * logmeanexp_{g in G_j}(cosine(f(x_q), f(x_g)) / tau_g)`。

pose/CLIP cache只产生detached pair证书：

`v_qg = 1[E(q,g) >= theta and common_available(q,g)]`。

其中`E`就是D0 signal audit已冻结的存在性region CLIP矛盾；`theta`只能由consumer-aligned raw-PID OOF协议
在不接触正式结果的校准折冻结。对每个负身份：

`V_qj = max_{g in G_j} v_qg`。

全部负身份无删除地分成`C_q={j != y_q: V_qj=1}`与
`U_q={j != y_q: V_qj=0}`。对两个集合使用归一化logmeanexp：

`a_q = tau_c * logmeanexp_{j in C_q}(z_qj / tau_c)`，

`b_q = stopgrad[tau_c * logmeanexp_{j in U_q}(z_qj / tau_c)]`。

最小certificate loss唯一为：

`L_cert(q) = softplus((m + a_q - b_q) / tau_l)`。

若`C_q`或`U_q`为空，`L_cert(q)=0`并计入coverage，不重新选pair/identity。`b_q`完全detach，因而该损失只直接
降低被pose-estimated共同可用槽矛盾证实的整个负身份集合相似度，不会主动吸引未决负身份。它不指定identity
prototype、局部target feature或单个hard pair。

genuine身份`y_q`永不进入`C_q/U_q`，始终只由共同宿主的原CE/positive/listwise项吸引；同PID上的`v_qg`
只统计false-certificate rate，禁止产生排斥梯度。`v/V`、pose、CLIP与所有cache都stop-gradient且无trainable
certificate head。`L_cert`的梯度只通过`a_q`进入anchor和certified-wrong负身份support的student global descriptor；
不进入slot feature、pose/CLIP或候选选择。

未来冻结总损失只能是：

`L_total = L_zero_owner_host + lambda_cert * mean_q L_cert(q)`，

其中`tau_g/tau_c/tau_l/m/lambda_cert/theta`必须在下一设计的static/PK64前一次冻结。所有负身份仍在宿主与
`C/U`并集中，
不得删pair、只取top-k、Borda选边、调triplet margin或把CLIP距离蒸馏为student pair distance。eval删除
pose/CLIP/cache与certificate loss，仍为原Swin-T global descriptor。

正式训练是否具有C类贡献，仍须自然e120、实际效应门、完整controls与3 seed证明；D0 signal或
consumer-aligned门本身都不进入论文性能表。

## 风险与失败解释

1. region CLIP可能只重现raw color/纹理，correct不会胜raw与global-CLIP；
2. instance-pose槽可能不比canonical-location crop更可靠；
3. 跨ID矛盾可能主要来自相机/背景，而非人体语义；
4. `max`可能被单个噪声槽支配；
5. D0已见official train，候选内绝对重排不是泛化证据；
6. PC-NEC可能与共同可见part matching同构，只有global-only训练证书整体能保留窄差分；
7. 若coverage不足，再放宽visibility会复现exp415的事后门修补，必须直接NO-GO。
8. 当前availability来自pose估计而非遮挡ground truth；fuel最多证明固定bank上的判别增量，不能直接声称逻辑
   意义上的错误身份证书或真实可见性。

## 对照组

当前阶段只有已实现的D0 signal controls与未实现的consumer-aligned gate，没有训练臂。后者完成实现、复审与
本地合同前，禁止创建任何formal namespace、OUTPUT_DIR、optimizer、checkpoint或e120 config。

## 预期结果

唯一可接受的正结果必须同时包含：correct在高覆盖共同bank中胜全部controls、在zero-owner上富集残余误排序、
全PK64 pair证书域闭合、identity安全且梯度与独立全排序下降方向一致。任何较弱结果都不足以建立
pose×CLIP训练方法。
