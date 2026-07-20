# 实验 exp405：CAVT — CLIP 校准的反事实解剖视图运输

## 当前状态

`BROAD NOVELTY NO-GO / NARROW PROBLEM+EVIDENCE CONDITIONAL-GO /
REAL-TEACHER STATIC V8 PASS / CUDA PREFLIGHT NEXT / FORMAL P0B NO-START / STUDENT NO-START`

exp405 是 exp404 封板后的新训练对象，不是 SPK 修补、续训或换 seed。exp392–404 的 checkpoint、
output、runner 与反事实结果保持只读。用户明确禁止 Claude，本实验不调用 Claude；正式执行前采用两路
独立 Codex 代码/机制红队与一路近期论文/开源实现审计，输出单独落盘。

## 动机

exp404 终审已经证明当前 `rich evidence -> slot mean -> descriptor product` 有效执行但无检索所有权：
correct 与 wrong-RGB 只差约 `0.0019 mAP`，NULL/bypass 反而高约 `0.1809 mAP`。后续根因审计又确认：

1. rich teacher 实际为 image-only PCA residual，不是双编码器 teacher；
2. student evidence 由全图 GAP 一次预测五槽，未做pose-defined局部池化；
3. `hidden.detach()` 使 evidence loss 只能训练最后一层线性头；
4. e120 `EvidenceCos=0.973`，student基本没有学会teacher；
5. in-bounds keypoint被误当presence，五槽几乎全1；
6. pre-RE teacher与post-RE student的局部appearance/support目标不一致；
7. SPK先对槽求均值，数学上消除slot identity。

因此下一步不能再问“怎样把同一个code更强地乘到descriptor上”，而要给CLIP–TAPF一个可观测的训练目标：

> 对同一身份的跨相机图像，正确身份、正确解剖槽的真实互补证据，能否在中间stage恢复一次明确删除的
> 局部状态；单图student能否在同一TAPF执行路径上学到该状态转移？

## 核心假设

对训练图 `x_i`、目标槽 `k`、同身份不同相机 donor `x_j`，构造 recipient 的受控槽证据删除
`x_tilde(i,k)`。若 CLIP image encoder 的实例级局部视觉读出、CLIP text encoder 的解剖原型和pose
slot correspondence共同提供有效中介，则 teacher-forced 运输必须同时依赖：

- `identity(j) = identity(i)`；
- `slot(j) = k`。

正确臂必须分别击败 same-ID/wrong-slot 与 wrong-ID/same-slot；否则系统只利用身份捷径或部位捷径，
不能称为可辨识解剖运输。

正式student不直接蒸馏CLIP final descriptor。它在每个被启用的backbone stage内预测同一个可执行
gather-transform-scatter状态；推理删除donor、两个CLIP encoder、文本原型和external pose，只保留RGB
student与一个global descriptor。

## 与旧实验的关键差异

| 对象 | exp392–404 | exp405 CAVT |
|---|---|---|
| teacher | raw/PC-MBCLS后转image-only PCA residual | frozen image+text双编码器共同定义slot state |
| presence | in-bounds keypoint flag | `geometry_valid`与连续`semantic_support`分离 |
| student读出 | full-map GAP -> 5x16 | 每slot mask-normalized stage-feature pooling |
| 可观测target | 单图任意latent code | original / slot-deleted / real same-ID donor三元组 |
| slot identity | 求均值前未固定 | 在运输、反事实和scatter完成前不得聚合 |
| CLIP作用 | 辅助loss或任意product factor | teacher selection + slot state + transport中介 |
| TAPF作用 | terminal/weak route | 中间stage gather-transform-scatter，最终descriptor可达 |

## 数据可行性边界

train split只读统计：702个PID全部具有至少两个camera，15,618张图全部可找到同ID不同camera donor；
510个PID覆盖至少3个camera。该统计只说明身份配对可行，不说明每个slot都具有高CLIP support。

Phase 0必须从train split冻结pair map，并逐槽报告高support donor覆盖率。不得读取query/gallery来选pair、
threshold、stage、prompt、loss或transport budget。

## 术语与状态拆分

### Geometry valid

`g_i,k`只表示pose坐标足以构造slot mask。它由finite/in-bounds坐标与最小mask mass决定，不能解释为可见性。

### Semantic support

`q_i,k in [0,1]`由实际student-view RGB上的frozen CLIP局部视觉读出与visible/occluded text prototype
产生。它必须保留连续幅值，不允许用 `score>0` 或固定 `0.5` 先二值化。

### Semantic slot distribution

`p_i,k`由同一个局部视觉feature与全部冻结body-part text prototypes的相似度形成，用于验证slot绑定；
`p`不能由text-only常量或pose label直接填充。

### Local visual content

`v_i,k`为pose-conditioned CLIP局部视觉feature；student不直接把它拼入最终descriptor，只在训练期作为
slot state/transport teacher。外观幅值与方向分别保存，不再统一L2后丢掉support强度。

## Phase 0：train-only teacher-forced oracle

### 0A. 实现与测量合同

在不训练TAPF前，先冻结以下检查：

1. RGB、pose、slot mask经过resize/flip/pad/crop后的marker误差与hash；
2. 左右joint翻转、patch row-major布局、CLIP normalization与checkpoint SHA；
3. 每图每slot的mask mass、质心、patch index集合和hash；
4. correct、slot-cycle、wrong-field、matched-context与NULL的patch集合真实不同；
5. `geometry_valid`与`semantic_support`分别记录，禁止字段复用；
6. slot permutation输入必须使teacher输出同样permutation，而非保持不变；
7. body-part crop、dense/region readout与matched wrong-mask三层诊断分别报告，不能用impure bbox crop冒充上界。

任一机械门失败只封板该measurement execution，不能解释为CLIP科学结论。若修正contract，必须另开
新实验编号，不得在exp405内以`v2`覆盖、补跑或改判。

### 0B. 双编码teacher门

固定prompt、checkpoint原生`logit_scale`与readout后，逐槽报告。不得继续沿用exp392–404的固定
`T=0.07`或image-only PCA轴。teacher state明确写成：

```text
v_i,k = FrozenClipImageReadout(x_i, M_i,k)
p_i,k = softmax(native_logit_scale * cosine(v_i,k, T_all_parts))
q_i,k = calibrated support from the same actual RGB view
```

其中`T_all_parts`由冻结text encoder一次编码并缓存，`p_i,k`必须是sample-specific全原型分布，不能把正确
pose label直接填入。`q_i,k`的可见/遮挡标定只用train上的受控slot deletion，不把in-bounds keypoint或
ViTPose score当作真值。

exp392的raw patch-pooling teacher已经封板失败，exp405不得同编号重测或补跑。exp405唯一primary readout冻结为
`P0B-ISO`：从第1层开始的region-isolated frozen image readout。目标slot的CLS/patch在全部block中不得读取
非目标patch，不能复用旧PC-MBCLS“前20层全局、末4层局部”的泄漏路径。五槽taxonomy一次冻结为
`head / upper torso+arms / lower torso / upper legs / lower legs+feet`，不得在结果后切17-joint或更换粒度。

逐槽报告：

- body-part distribution macro accuracy、MRR、confusion matrix；
- correct pose slot相对same-area/matched-centroid wrong slot的text margin；
- support AUROC/AUPRC/ECE及每slot动态范围；
- correct RGB、wrong RGB、wrong mask、text cycle、image-only、text-only、generic与NULL；
- crop-global参考与部署候选readout之间的差距。

若crop-level成立而region-isolated readout失败，只封板exp405 readout；后续新readout必须另开实验编号。

必须额外检查：同一slot在original、deleted-25/50/75%上的`q`单调性；**50% deletion是唯一primary**，
25/75%只作预注册诊断，禁止挑best。 同一图slot permutation后`p/v/q`
必须等变；wrong RGB、matched wrong mask、文本原型循环分别破坏对应轴。任何一个execution若只在torso有效，
不得以macro均值授权transport。

### 0C. 受控槽证据删除

第一版不把内部操作包装为自然物理遮挡。对目标stage的recipient slot `k`执行精确定义的feature-slot
deletion，并保留original recipient作为可观察target。donor为同PID、不同camera、该slot高support样本。

冻结表示：

```text
z_i,k^s = MaskPool(F_i^s, M_i,k^s)
z_j,k^s = MaskPool(F_j^s, M_j,k^s)
c_i,not-k^s = MaskPool(F_i^s, 1-M_i,k^s)
c_j,not-k^s = MaskPool(F_j^s, 1-M_j,k^s)

z_transport = z_j,k^s - stopgrad(c_j,not-k^s) + stopgrad(c_i,not-k^s)
F_transport = ScatterReplace(F_tilde_i^s, M_i,k^s, z_transport)
```

这是teacher-only闭式oracle，不是最终可学习operator，也不声称公式本身是贡献。它只回答same-ID/same-slot
证据是否存在可利用上界。remaining blocks必须真实重算，不能只比较局部cosine。

CLIP校准以 `q_j,k` 与 `p_i,k/p_j,k`控制donor有效性；pose-only uniform support是强对照。若CLIP校准
不能优于pose-only，不能声称CLIP增量，Phase A保持NO-START。

该闭式式子只作为零参数上界探针；它不能被直接包装为最终operator。还必须同时执行`original`、`deleted`
和`self-slot restore`三条参照：若连同图原槽回写都不能恢复remaining-block descriptor，则失败属于
deletion/scatter测量器；若self可恢复而same-ID/same-slot不能优于错误donor，才属于跨视角transport失败。

### 0D. 身份轴 x 解剖轴二维反事实

每个recipient必须配齐：

1. `same-ID / same-slot`：correct；
2. `same-ID / wrong-slot`：破坏slot轴；
3. `wrong-ID / same-slot`：破坏identity轴；
4. `pose-only / same-ID / same-slot`：去除CLIP校准；
5. `image-only`：去除text encoder；
6. `text-only/static`：去除image encoder；
7. `generic mean`；
8. `NULL`；
9. `random-key`：同维度、同范数、同分布。
10. `self-slot restore`：只作测量上界，不进入方法收益比较；
11. `frequency-matched random-cluster`：排除共享伪语义类别再次形成authentication假阳性。
12. `MVI2P-full`：matched donor/budget的full-feature多视图综合近邻；
13. `pose-part`：pose-only same-ID/same-slot，不使用CLIP；
14. `attribute-relation`：pose slot + frozen text关系，但不执行counterfactual transport；
15. `generic-transport`：同scatter位置、同范数与同预算的普通feature recovery。

wrong-ID donor须匹配camera、support、mask mass，并尽量匹配global CLIP similarity，避免easy negative。
不预注册两个单轴破坏臂之间的顺序，只要求correct分别高于每一个对照。

### 0E. Oracle指标与kill-switch

主指标同时包括：

- 运输后descriptor相对original recipient descriptor的cosine恢复；
- slot deletion造成的same-ID similarity损失被恢复的比例；
- correct相对每个control的paired mean/median gap与PID-cluster bootstrap CI；
- train-only冻结检索的rank improvement；
- 每slot、每删除强度、每PID正/零/负方向计数；
- CLIP校准相对pose-only的独立差值。
- correct相对`MVI2P-full / pose-part / attribute-relation / generic-transport`的matched近邻差值。

以下任一成立即 `CAVT TEACHER ORACLE NO-GO`，不实现student、不跑e120：

- correct不高于same-ID/wrong-slot；
- correct不高于wrong-ID/same-slot；
- correct不高于generic、NULL或random-key；
- correct不高于frequency-matched random-cluster；
- correct不高于generic-transport；
- pose-only复现correct，CLIP无独立作用；
- `MVI2P-full`或`pose-part`在matched设置下取得同等或更好的identity/rank恢复，且CAVT没有额外二维偏序；
- 仅torso成立，其他槽方向不成立；
- rank与identity margin未随descriptor恢复同步改善。

### 0F. donor-free可预测性门

teacher oracle通过后仍不得直接实现Phase A。必须在缓存state上按PID隔离fit/validation，用recipient的
`not-k`状态预测teacher transport residual，并同时报告held-out PID的residual cosine、R2和恢复rank。
它必须优于zero、identity-mean和generic linear predictor。若不优于这些control，说明oracle依赖donor私有且
单图不可观察的信息，直接判`CAVT DONOR-FREE TARGET NO-GO`。

## Phase A：semantic single-stage CAVT

只有Phase 0A--0F全部门PASS才创建正式config。以sealed clean D0 seed1234为直接recipe对照，第一臂只在原D0
late anchor/consumer位置加入CAVT，不同时启用多阶段。

### Slot-local student anchor

```text
H_s        = AnchorConv(stopgrad(F_source^s))
M_hat_s,k  = direct pose/region field
z_hat_s,k  = MaskPool(H_s, M_hat_s,k)
[q_hat,p_hat,v_hat]_s,k = SlotHead(z_hat_s,k)
```

相对旧实现必须满足：

- 不使用full-map GAP一次拆五槽；
- teacher loss可更新AnchorConv与SlotHead，但不回流teacher；
- ReID梯度允许通过一个预注册有界接口告诉semantic trunk哪些状态有检索价值；
- 不把evidence整体detach后再要求router凭空理解其作用；
- NULL/min-mass路径逐tensor identity。

### Donor-free state transition

训练teacher branch用correct donor得到slot transport residual `r_teacher`；student branch只看recipient当前
RGB和其余可见slot，预测 `r_student`。两者在同一production gather-transform-scatter operator上执行，
禁止另接训练后删除的projector或只蒸馏最终descriptor。

CLIP不是只做donor过滤器。冻结teacher的`p/q/v`必须共同定义production residual的semantic address、执行预算与
局部内容target；分别循环text prototype、image evidence或slot address时，teacher residual和最终descriptor都
必须按预注册方向变化。若去掉CLIP后相同donor与pose即可复现residual，则CAVT退化为普通multi-view feature
completion，不能声称CLIP–TAPF深耦合。

损失只监督：

- slot distribution/support/content teacher；
- production residual direction/relative geometry；
- 原有ReID ID+triplet与D0 pose loss。

不直接做CLIP global descriptor KD、不建language retrieval、不保留part-token matcher到测试。

### Phase A终审

训练轨迹固定在e10/20/.../120评估。每个评估点必须同时抄录CAVT与sealed clean D0**同一epoch**的
`mAP/R1`及两者差值；不同epoch或不同训练预算只作背景，不得写成涨点证据。唯一正式判定使用自然完成的e120，
不得按中间性能早停或挑选best epoch。

必须同时满足：

1. final mAP与R1均不低于clean D0；
2. correct相对wrong-slot、wrong-ID、pose-only、image-only、text-only、generic、NULL、random-key均有
   预注册正margin；
3. all-CAVT-bypass至少下降`0.1 mAP`，且不是只靠一个slot；
4. teacher residual与student residual的cosine/R2达到预注册门；
5. query/gallery执行图无CLIP、text、pose、donor或train cache；
6. config-off相对clean D0 state/RNG/optimizer/output exact。
7. CLIP image-only、text-only、pose-only任一单轴都不能复现full teacher；
8. teacher与student的identity轴 x slot轴二维反事实偏序均成立。

若只学会teacher但不超过D0，判“语义蒸馏成立、ReID机制NO-GO”；不得直接进入多阶段。

## Phase B：所有stage独立direct CAVT

只有Phase A通过后才实现。Phase B相对semantic single-stage只改一个变量：把同一冻结CAVT对象复制到
预注册的多个backbone stage。

- 每个stage独立预测direct pose/region fields；
- anchor不共享、不用offset链；
- 每stage用自己的field池化自己的ReID feature；
- 每stage的transport residual在进入下一stage前真实执行；
- 每stage consumer对最终global descriptor必须有独立bypass贡献；
- 跨层一致性只约束同一slot的distribution/support，不把所有stage强制成相同feature；
- 总语义loss按consumer数mean，保持与single-stage相同总budget。

最终须报告single-stage与all-stage的matched参数/FLOPs/速度/显存、逐stage bypass、slot-cycle、wrong identity、
wrong slot与完整mAP/R1/R5/R10。exp391纯结构NO-GO只作为无语义直接背景，不禁止本Phase B；但若
semantic multi-stage不超过semantic single-stage，则如实判层级增益不成立。

## 新颖性边界

CAVT不能声称首次：

- pose+CLIP；
- part-language alignment；
- same-ID multi-view teacher；
- token transfer/feature completion；
- inference-free KD；
- multi-stage pose conditioning。

可争对象仅是：

> 在单RGB固定descriptor ReID中，以完整图/受控槽删除/真实跨相机donor构成可观察target，使用frozen
> CLIP双编码器校准pose-defined解剖槽，并以identity轴 x slot轴二维反事实证明同一中间TAPF状态转移
> 同时拥有正确身份与正确解剖语义；推理删除全部特权输入。

若公开近邻已覆盖这一完整对象，或最终证据不能同时击败两个单轴破坏臂，则降级为组合工程，不包装主贡献。

## 实现纪律

1. Phase 0前不创建训练config/output/runner；
2. 先写pair-map/teacher-oracle protocol与static/CPU正反合同，再允许GPU teacher-only执行；
3. 所有数据来自train日志/冻结结果，不凭记忆；
4. Swin-Tiny、batch64、official clean数据与D0 recipe固定；
5. fresh、串行、不续训、不挑best、不按中间性能改设计；
6. 同一时刻只允许一条4090工作；
7. 只显式暂存exp405目标文件，禁止`git add -A`。
8. exp405只允许一次unique Phase 0 measurement；runtime/contract错误封板该execution，修正须使用新实验编号。
