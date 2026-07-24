# exp395--414失败复盘与下一方向

日期：2026-07-24

状态：`RESEARCH RESET / NO TRAINING AUTHORIZED`

## 1. 复盘范围与口径

本复盘覆盖exp395--414共20个编号，但它们不能被误写为“20个pose+CLIP方法都失败”：

- exp395--400共6项是AMP、测量器与production contract；
- exp405--407共3项是CAVT teacher/cache/caliper测量合同；
- 真正提供方法科学证据的是exp401--404与exp408--414，共11项。

因此，本轮首先区分工程合同、未回答的科学问题、性能失败与归因失败。所有数字只来自已封存日志与文档；
exp414 text-shuffle因用户终止没有自然e120，不能进入终点比较。

## 2. 二十项证据表

| 编号 | 对象 | 结果或状态 | 科学结论 |
|---|---|---|---|
| exp395 | loss×参数组AMP归因矩阵 | 真实大参数组reporter失效 | 测量器失败，不是方法结果 |
| exp396 | chunk-safe AMP reporter | D0/rich非有限支持与计数相同 | 排除rich-specific数值失败 |
| exp397 | 12步native GradScaler matched门 | D0/rich轨迹exact，绝对门失败 | relative parity成立，生产门设计过严 |
| exp398 | 32步稳态group reporter | 首个forward前容器类型错误 | 测量器失败 |
| exp399 | named-parameter稳态合同 | D0/rich均26/32更新，extra skip=0 | rich AMP稳态不劣于D0 |
| exp400 | final production contract | reload、teacher-free、RGB-only、双consumer PASS | 只证明可训练/可部署 |
| exp401 | rich-budget C0 route | e120约`57.1/67.3`，低于D0；route gap仅约`+0.119 mAP` | route数值存活，语义所有权未成立 |
| exp402 | 10臂semantic-interface反事实 | wrong-RGB、zero不弱于correct | 当前semantic mediator NO-GO |
| exp403 | evidence-owned low-rank operator | correct/wrong/generic/NULL/bypass mAP最大差约`0.001` | operator active不等于retrieval ownership |
| exp404 | final Semantic Product Kernel | NULL/bypass比correct高约`0.181 mAP/0.588 R1` | terminal semantic product有害 |
| exp405 | CAVT region-isolated teacher | wrong-mask无合格donor | 科学问题未评估，测量合同失败 |
| exp406 | donor reserve与cache | 全量编码后cache自检失败 | 科学问题未评估 |
| exp407 | trusted-cache CAVT formal | wrong-mask caliper validity失败 | 科学问题仍未评估；不再救测量器 |
| exp408 | pose-indexed CLIP relation KD | relation order PASS；e120低D0约`0.5 mAP` | 局部关系可学但不改善最终检索 |
| exp409 | pose×CLIP hard-pair miner | 相对D0约`-0.6 mAP/+0.9 R1` | 单hard edge改善首位而伤全排序 |
| exp410 | frozen CLIP identity proxy | 相对D0约`-12.6 mAP/-11.3 R1` | CLIP identity坐标与ReID几何严重错配 |
| exp411 | pose-complete set ranking | correct=`58.8/70.1`；zero-owner=`58.9/70.3`；wrong-RGB=`59.1/70.7` | set/listwise宿主有效，pose×CLIP owner失败 |
| exp412 | pose×CLIP梯度机会路由 | `56.9/69.7`，低zero-owner`2.0/0.6` | 估计为非支配的视图获得更高路由预算，该冻结路由相对zero-owner性能下降 |
| exp413 | coverage-chain prefix ranking | correct=`59.3/70.8`，pose-only=`59.3/70.1` | 性能GO，联合归因失败 |
| exp414 | continuous identity region + MST | correct=`59.2/70.7`，pose-only=`59.2/70.8`，q-only=`59.2/70.8` | 性能GO，pose轴与联合归因失败 |

exp414 text-shuffle只保留e10/e20/e30=`29.2/38.5/54.1/61.0`、
`46.7/56.4/71.5/76.8`、`49.6/61.0/74.5/79.8`；用户在e35 iter20终止，整臂
`VOID / NO RESUME`，all-edges=`NO-START`。

exp413 q-only只到e80，未自然e120；exp413的q-only/text-shuffle与exp414的text-shuffle/all-edges均不得用于
终点反事实比较。已有pose-only/q-only足以阻断联合机制的必要条件，但不能进一步声称文本绑定或all-edges方向已被
单独证伪。exp414 q-only仍使用pose-defined region CLIP cache，因此“POSE AXIS FAILED”只应收窄为
“online pose-visibility轴未证明必要”，不是严格pose-free结论。

## 3. 反复失败的结构性根因

### 3.1 同PID交换对称性使pose/CLIP不可识别

exp411、413、414虽然名称不同，本质都在同PID合法正样本上改变索引或系数：

- exp411复制或加权owner；
- exp413排列同三张support并优化prefix；
- exp414在同三个顶点的三条边中删除一条chord。

它们都可写成：

`phi_p({D(anchor,i): y_i=p}, z_pose, z_clip)`。

PID loss对同类样本交换不变，并持续把同PID descriptor压向同一簇。训练后若`f_i≈mu_p`，owner、顺序、
prefix或MST自然趋同。correct与wrong/control仍然都是标签合法的监督，网络可以完全绕过pose/CLIP而不付出身份
损失。因此出现稳定两难：

1. pose/CLIP强行进入身份几何时，性能下降；
2. pose/CLIP只改变同PID索引时，普通宿主涨点，但correct语义无法归因。

exp409不能归入上述同PID交换对称性：它还改变different-PID negative。它的负结果更直接支持“单个hard edge/R1
目标与全候选mAP排序错位”，而不是同PID owner不可识别。

### 3.2 “机制active”不是“最终检索拥有语义”

route nonzero、topology change、gradient nonzero、relation order正确、descriptor发生变化与AMP真实更新，都只是
接线证据。exp401、403、404、408已分别证明：这些门全部通过后，correct仍可不优于wrong、NULL、bypass或D0。
以后不得再用接线门替代全验证集matched counterfactual。

### 3.3 当前CLIP信号不是可靠的ReID身份充分变量

- wrong-RGB、zero、pose-only、q-only反复不弱于correct；
- frozen CLIP proxy接管identity axis导致exp410灾难性退化；
- visible-vs-occluded文本轴当前只有很小的未校准标量变化，机械上非零不等于真实遮挡语义；
- “prompt不含身份”不代表视觉投影结果不含身份、相机、背景或mask artifact。

CLIP可以作为训练输入干预的审核器，但不应规定最终身份坐标、prototype、router或同PID owner。

### 3.4 pose、CLIP与student没有观察同一个事实view

exp411--414的CLIP cache来自`raw-rgb-pose-resize-384x128-no-augmentation`，student却读取随机flip/crop和
`RE_PROB=0.5`后的图像；在线pose visibility又来自增强后坐标，Random Erasing没有同步更新pose或cache。
这混合了当前增强几何、原图CLIP appearance与未反映擦除的keypoint confidence，属于直接变量混淆。

下一方法若student看增强图，pose和CLIP必须审核完全同一张增强后像素图；若clean原图作为target，则必须明确写成
`masked/edited view -> original view`的可观察反事实。

### 3.5 局部目标与mAP全排序错位

exp408关系KD能学却不涨点；exp409只涨R1而降mAP；exp412把梯度集中到pose×CLIP估计非支配视图后性能下降。
遮挡ReID需要保持所有
正样本与全部身份的排序，而不是只优化单pair、中层关系或“可靠样本”的训练预算。

### 3.6 zero-owner宿主掩盖了新增机制

zero-owner相对D0同时改变：

1. batch-hard变为all-identity logmeanexp；
2. 单正样本变为三support均值；
3. 负身份也执行K-1 support平滑；
4. occurrence-wise LOO形成jackknife式正则。

它是当前最强的普通metric-learning宿主，但不是pose completion证据。exp413/414相对D0的主要增益继承自该宿主，
新增机制只剩单seed `0.2--0.5`点波动。未来不能只拿D0当基线，也不能把zero-owner涨点写成pose+CLIP贡献。

### 3.7 单seed微小严格差产生winner's curse

连续候选只要求单seed `>0`便进入matched controls，`0.2--0.5`点很可能处于训练噪声分辨率内。后续必须设置
实际效应下界，并在探索臂通过后使用多seed与per-query paired bootstrap；不能让“严格大于0”承担论文因果主张。

### 3.8 工程复杂度吞噬科学信息

20个编号中9个主要用于AMP、cache、donor与caliper。CAVT连续三个编号没有得到一次科学结果。以后每个新方法只允许
一次小规模、围绕可观察target的合同；测量器需要多轮修补时直接判当前证据路径成本过高，不再以新编号救援。

## 4. 永久关闭的机制家族

1. pose/CLIP只负责从同PID样本中选择、排序、加权、复制或连边；
2. 三support上的owner、prefix、MST、polyline、convex hull、cone/box变体；
3. hidden bias、low-rank operator、terminal product、channel/group scaling式CLIP注入；
4. per-slot cosine/KL/relation KD及只证明hidden head学到teacher的辅助loss；
5. pose×CLIP hard-pair miner的margin、top-k或rank fusion微调；
6. backward-only梯度路由，尤其把recipient预算转给pose×CLIP估计非支配视图；
7. frozen CLIP visual/text proxy直接接管ReID classifier或identity coordinate；
8. 继续修CAVT donor/caliper/cache测量器；
9. 调prompt、temperature、MST edge、support数或loss比例救exp411--414；
10. 只报告correct相对D0涨点而隐藏zero-owner、wrong-RGB、pose-only或q-only。

## 5. 下一候选排序

| 排名 | 候选 | 新训练对象 | pose职责 | CLIP职责 | 主要风险 |
|---:|---|---|---|---|---|
| 1 | PACIT | 同一真实图的clean→受控遮挡像素反事实 | 定义合法人体槽与遮挡几何 | 验证目标身份语义被移除、非目标语义保持 | 可能退化成普通structured augmentation |
| 2 | CPOS | train-only同身份pose/occlusion orbit生成 | 给目标姿态轨道 | 局部认证衣着/携物语义未漂移 | Pose2ID/IPG近邻强、生成身份漂移 |
| 3 | PSE-IRM | 真实图上的pose×context环境泛化 | 几何/遮挡环境 | 身份无关context环境 | 易退化成GroupDRO/exp412重权 |
| 4 | APCO | pose/CLIP约束下的最坏像素遮挡 | 限制可实现mask | 约束非目标语义保持 | adversarial erasing近邻多、训练不稳 |

查新已核对Pose2ID/IPG与AAAI 2025 DiVE等近邻，但网络检索不稳定，不能声称穷尽或绝对首次。四者均不应被包装为
普通attention、part、fusion或可见性小改。

## 6. 唯一首选：PACIT

### 6.1 问题定义

训练集缺的不是另一个feature proxy，而是可观察的配对事实：

> 同一真实人物图像在一个明确的身份语义人体区域被遮挡后，最终global descriptor是否仍保持正确身份邻域？

原图`x`本身就是clean target；反事实图`x_cf`在像素空间真实存在。wrong pose/CLIP会生成任务上真正错误的干预，
不再只是给合法同PID positive换一个索引。

### 6.2 机制

1. 只从sealed clean D0/TAPF开始，不复用zero-owner代码路径；
2. pose heatmap在当前实际输入像素上定义人体槽与合法遮挡形状，即`where`；
3. frozen CLIP在同一像素view上测量该槽的衣着、颜色、纹理或携物等身份相关语义，并审核编辑后非目标槽保持，
   即`what/validity`；
4. 用图内或数据内非人物背景patch进行alpha composite，不用生成器猜脸或衣物；
5. clean与counterfactual都走相同Swin-T/TAPF、CE与原metric loss；只加stop-gradient
   `clean descriptor -> counterfactual descriptor`一致性，CLIP不进入identity coordinate；
6. 测试完全保持原RGB global descriptor。

这条路线不做feature completion、donor transport、owner、prefix、MST、hard-pair、proxy或gradient routing。
它第一次把pose+CLIP变成“有原图target的输入分布干预”，直接回应single-image support incomplete。

### 6.3 首个asset oracle

先做一次`512图 × 每图8个固定候选 × frozen D0只推理`的oracle，不训练、不改模型。所有阈值预注册：

1. mask占人物框前景`12%--20%`；各arm对每张图的mask area、alpha、slot频率逐项exact matched；
2. 目标槽归一化pose coverage相对原图下降至少`0.35`；
3. 非目标槽pose coverage mean绝对变化不超过`0.03`、max不超过`0.08`；mask外像素byte-exact；
4. 目标槽“原图最高身份属性prompt相对generic person prompt”的CLIP margin下降，同时超过该图8候选的P75和
   全oracle分布`median + 0.5*MAD`；
5. 非目标四槽region-CLIP cosine mean至少`0.98`、minimum至少`0.95`；
6. frozen D0的clean→counterfactual cosine位移落在同面积ordinary ROA分布P50--P90；
7. 原PID classifier top-5保留率至少`90%`；若无classifier接口，则同PID真实prototype margin仍高于batch内
   异PID P95；
8. 满足全部门的图至少占512图的`70%`，且五槽各至少`64`例；
9. correct相对pose-only、text-shuffle、random-key的candidate choice change rate均至少`25%`；
10. 各arm最终像素难度的面积、alpha、D0位移P50/P90差不超过`5%`；
11. random-key或frequency-matched random-cluster若也通过CLIP有效性门，oracle直接NO-GO。

只有correct在等面积、等难条件下同时通过pose单槽干预、CLIP目标语义变化、非目标语义保持、身份保留、覆盖率与
random-key排除，才记`JOINT ASSET ORACLE GO`。任一失败即
`PACIT ASSET NO-GO / E120 NO-START`，不在同一oracle内放宽阈值。

### 6.4 必须保留的正式对照

- sealed clean D0；
- ordinary ROA；
- pose-only：相同pose mask但随机等难patch/语义选择；
- CLIP-only：相同CLIP分数但不使用人体槽约束；
- text-shuffle；
- semantic-blind random-key；
- frequency-matched random-cluster。

所有训练臂必须匹配图数、双视图数量、mask area、alpha、slot频率、计算预算与recipe。只有PACIT-correct自然e120的
mAP与R1同时严格高于sealed clean D0、sealed zero-owner历史性能线与全部matched controls，随后多seed与
paired bootstrap也通过预注册效应门，才记`PERFORMANCE + POSE×CLIP ATTRIBUTION GO`。zero-owner不属于
PACIT matched control，也不进入新代码，只作为当前最强简单宿主的性能下限。只胜D0但不胜zero-owner或任一
matched control，最多记
`STRUCTURED AUGMENTATION PERFORMANCE GO / JOINT ATTRIBUTION FAILED`。

## 7. 执行顺序

当前用户已明确停止训练，因此本文件只完成复盘与预注册方向，不启动PACIT或任何GPU任务。

若未来用户明确授权恢复：

1. 先冻结PACIT asset design与查新边界；
2. 只做一次512图oracle；
3. oracle失败即停，不修改阈值救臂；
4. oracle通过后先做唯一真实PK64接线合同；
5. 先运行correct；过D0与ordinary ROA实际效应门后，再串行运行全部matched controls；
6. 单seed归因通过后才允许多seed；
7. zero-owner只作为必须超过的强历史性能线，不进入PACIT代码路径或正面贡献。

## 8. 最终研究判断

当前证据不是“pose+CLIP必然无效”，而是：

> pose/CLIP至今没有获得一个与student当前像素view对齐、错误信号在任务上真正错误、且PID-only loss无法绕过的
> 可观察训练对象。

只要外部信号仍然给同PID exchangeable positives换权、换序或换边，就应直接`NO-START`。PACIT值得成为下一唯一
首选，原因不是机制更复杂，而是它第一次同时满足可观察target、pose/CLIP职责不可交换、输入支持真实扩充、测试路径
不变和一刀可证伪。

## 9. exp415 oracle回填：PACIT当前实例已被一刀否定

用户授权后，PACIT经三轮设计复审、全15,618 geometry census、固定8图runtime smoke和唯一512正式oracle。
oracle执行与provenance完整，不是INVALID：CLIP/D0/row均`512/512`，严格异常0，top-5保持`512/512`，
optimizer/checkpoint写入0，formal与GPU终态洁净。

科学结果却明确NO-GO：

- factorial Y `neither/clip-only/pose-only/P+C = 2/1/4/2`；
- complete quartet=`21/512=4.10%`，两条strong pair各`199/512=38.87%`；
- 四个条件边际为`-0.001953/-0.003906/+0.003906/+0.001953`，interaction=`-0.001953`，
  10,000次paired bootstrap没有正向下界；
- selector与独立blind颜色事实agreement=`1/512`，text-shuffle=`0/512`；
- raw-color与D0-hard两条strong control中，P+C reference与control均为`80/512`。

因此第6节首选只在当时设计阶段成立，现已由预注册oracle正式覆盖。当前PACIT实例同时失败于：

1. 四条post-hoc severity/identity/geometry direct caliper求交过于稀疏，无法形成高覆盖estimand；
2. `7 shape × 10 color`的CLIP选择最终只执行7种灰色矩形，color label不改变像素动作，CLIP的`what`
   退化为geometry selector；
3. whole-image颜色margin与6%局部解剖槽的CIELAB连通颜色事实不一致；
4. 即使进入较宽的strong matched子集，P+C仍与raw/D0-hard exact tie。

这不证伪所有clean→edited输入干预，也不证明CLIP普遍没有局部语义；它只证伪当前
`pose center + gray rectangle + whole-image color-to-shape selector + sparse caliper`对象。exp415不得通过
调prompt、阈值、caliper、候选数或blind门复活，也不得启动e120。

对下一方向的约束更新为：

- CLIP选择必须对应实际不同、局部可ground并能由独立事实验证的像素动作，不能只改变metadata/index/weight；
- pose、CLIP与student必须观察同一实际像素view；
- 所有臂从同一完整候选bank按构造匹配，不能再依赖事后四边筛选；
- 先做无训练signal audit，CLIP局部变量相对raw/geometry baseline必须有paired增量；
- 目标必须作用于最终global descriptor或全身份全排序，并严格超过zero-owner/listwise强宿主；
- 科学Y、complete、match、identity-safe继续分开，固定分母与不利结果完整保留。

当前最终状态=`EXP395--415 POSTMORTEM UPDATED / EXP415 ASSET NO-GO /
NO FORMAL TRAINING AUTHORIZED`。

## 10. exp415后的候选重排：唯一首选PC-NEC

三路独立复盘只保留两个条件候选：

| 排名 | 候选 | 新训练对象 | 关键风险 |
|---:|---|---|---|
| 1 | PC-NEC | pose估计共同可用槽对错误身份的存在性负证据 | 易与part matching/metric learning同构，必须先证CLIP增量 |
| 2 | MH-PSO | 多假设pose-semantic orbit上的最坏情形身份一致性 | 生成身份漂移，Pose2ID/IPG/DiVE近邻强 |

唯一首选为`PC-NEC`（Pose-Conditioned CLIP Negative-Evidence Certification）。它不再从不完整单图证明
“这是同一个人”，而使用一个更可识别的非对称命题：

> 缺失区域不足以证明同一身份，但双方被pose估计为共同可用的解剖槽中，一个可靠语义矛盾可能排除错误身份。

pose只建立两张真实图之间的解剖槽对应与共同可用集合；CLIP只给这些槽提供“矛盾/未决”证据，不生成
identity prototype、完整feature或support顺序。未来student若获授权，应对全部候选负身份输出非负反证能量，
用存在性MIL/listwise目标要求每个错误身份至少被一个可见槽否决；测试删除pose、CLIP与证书头，仍为原RGB
global descriptor。

它与exp409的差分必须保持清楚：候选bank先由sealed D0一次冻结，correct/control共享全部真匹配与top-20
跨ID impostor，语义分数不能选pair、删pair或改变bank；审计与未来训练都作用于全部候选身份，不做hard-pair
mining。与KPR/BPBreID类part matching的差分则是局部匹配只作训练期负证据证书，正式检索不保留part分支。

当前只允许建立无训练fuel audit设计。启动代码/CUDA前必须证明：

- correct的candidate-level AUROC/AUPRC相对最强pose-only、canonical-location CLIP、neither、slot-shuffle、
  wrong-RGB、raw-color/student-part control均至少`+0.03`；
- 在同一固定bank的诊断重排中相对D0至少`+1.0 mAP/+1.0 R1`，相对最强control至少
  `+0.5/+0.5`；
- PID-bootstrap 95% lower bound大于0，query覆盖至少80%，五槽均有足够样本；
- bank manifest必须在读取pose/CLIP前封存，所有arm固定相同pair与分母；
- 诊断重排不得进入论文方法或最终测试路径。

任一门失败即`PC-NEC FUEL NO-GO / TRAINING NO-START`，并把当前搜索结论更新为
`NO CANDIDATE`；禁止回到PACIT、owner/prefix/MST、proxy/router或临时再开第三个弱组合。

## 11. 独立复审后的候选边界与执行判断

三名只读子agent分别从20次证据链、公开近邻/本地167篇笔记和PC-NEC数学对象复核，结论一致：

1. `PC-NEC`是唯一其关键前提仍值得用一次固定fuel audit证伪的候选；这不是性能概率承诺，创新也只具C类条件资格；
2. `PS-ODM`（pose-semantic occluder distribution matching）可作为第二备选：pose决定解剖落点，CLIP选择
   实际不同的真实occluder RGB，所有control exact匹配area/alpha/槽/类别边际；风险是与realistic occlusion
   augmentation近邻拥挤；
3. `APCO`可作为第三备选：D0全身份ranking adversary选最坏真实mask，pose/CLIP只限定可实现域和未遮语义保持；
   风险是与adversarial erasing邻近且训练不稳；
4. `MH-PSO`只保留为远期候选：多假设pose/occlusion orbit加CVaR全身份ranking，但生成身份漂移和Pose2ID/IPG/
   DiVE近邻风险最高。

以下路线继续关闭：donor identity source separation、PSC-JEPA/EMA support completion、PSE-IRM/GroupDRO、
attention/part fusion/prompt/proxy/router，以及任何同PID owner/prefix/MST改名变体。

本轮复审还阻断了PC-NEC原future训练式：`softplus(m+LSE_C-LSE_U)`会在梯度下降时主动提高未决负身份`U`的
相似度，raw LSE又混入集合大小。无训练fuel audit不受影响；fuel GO后也不得直接训练。下一设计只能使用
归一化logmeanexp，并把`U`作为stop-gradient参考，使梯度只排斥certified-wrong集合；certificate descriptor
还必须与离线证书观察同一deterministic RGB view。连续AUROC/AUPRC通过后，PK64前仍需held-out-PID阈值可行性门，
验证anchor-level genuine-identity family-wise误证率、负身份coverage与`C/U`非空率。

实现回归还发现并修复了camera-label捷径：真匹配原本只来自跨相机，而任意top-20跨PID impostor会造成
真假标签与candidate camera分布不匹配。冻结bank现按每个query的genuine candidate-camera频数分层，
每个出现的camera先保留至少1个impostor，再以largest-remainder分配剩余quota，并只在同camera stratum内按D0
距离取最近跨PID候选；任一quota不足在pose/CLIP读取前直接INVALID。当前“CLIP”也只指冻结OpenCLIP image
encoder的region visual evidence，不使用text encoder，因此不得写成语言语义不可替代。threshold-feasibility
唯一门冻结为：genuine identity的anchor-level family-wise误证率不超过`1%`、负身份coverage的PID-macro mean
至少`30%`、`C/U`同时非空anchor至少`80%`。

因此当前执行顺序固定为：

`PC-NEC唯一fuel audit → 任一门失败则NO CANDIDATE → 全门通过才重审threshold与same-view训练对象`。

不因fuel audit通过而自动授权PK64或e120。

## 12. 最终效能红队：唯一候选仍需consumer-aligned重构

实现正确不等于方法有效。最终两名独立效能红队指出，现有D0 signal fuel仍跨越四个没有验证的对象转换：

1. diagnostic consumer是sealed D0，future宿主却是zero-owner；
2. 连续E与OOF lambda变成二值theta与C/U；
3. 每query top-20 image bank变成随机PK64全部负身份；
4. 诊断重排直接使用E，正式eval却完全删除E。

最强旧反证是exp409和exp412：前者让pose×CLIP选择的异PID negative直接进入triplet，得到约
`-0.6 mAP/+0.9 R1`；后者的真实PK64路由active却比zero-owner低`2.0 mAP/0.6 R1`。因此PC-NEC虽不受同PID
交换对称性约束，仍与“外部语义选择额外负梯度”家族高度近邻。

四候选最终只保留修订PC-NEC，而且只授权下一次无训练consumer-aligned否决门：

- sealed zero-owner descriptor上，correct证书必须富集真正残余误排序负身份并严格胜七controls；
- 确定性`16×4` batch的全部`64×64` pair必须从按图pose/OpenCLIP cache获得证书，missing/fallback=`0`；
- `P=16`、15个负身份，genuine误证`<=1%`、负身份coverage`>=30%`、`C/U`同时非空`>=80%`；
- exact `L_cert`与独立zero-owner全排序梯度的归一化对齐，correct相对每个control的PID-bootstrap下界均`>0`。

PS-ODM不选，因为最可能得到的是ordinary realistic occlusion augmentation收益而非CLIP增量；APCO不选，
因为与adversarial erasing几乎同构且“攻击更难”会构成自证。修订PC-NEC任一门失败即
`PC-NEC TRAINING NO-START / NO CANDIDATE`，不自动递补其他候选。
