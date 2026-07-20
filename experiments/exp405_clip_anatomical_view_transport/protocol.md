# exp405 CAVT 最小Phase 0协议

## 当前授权边界

本协议只授权：文献/代码审计、synthetic CPU合同、train-only pair map、region-isolated双编码teacher测量、
teacher-forced oracle和held-out PID donor-free probe。正式训练config、output、runner、e120均为`NO-CREATE /
NO-START`。

## 冻结对象

- 数据：official Occluded-Duke train只读；不读取query/gallery选择任何设计。
- pose：只读冻结artifact，且只把in-bounds/finite解释为`geometry_valid`。
- taxonomy：五槽`head / upper torso+arms / lower torso / upper legs / lower legs+feet`。
- CLIP：冻结ViT-L/14 image+text encoder、官方normalization和checkpoint原生`logit_scale`。
- readout：从第1个block起region-isolated；旧raw patch和PC-MBCLS不得复用为primary。
- deletion：50%为唯一primary；25/75%只作预注册单调性诊断。
- ReID对象：sealed clean D0 seed1234/e120 checkpoint只读，remaining blocks真实重算。
- 目录：本机只写本仓库；远端只写`/home/afr`；official数据和pose资产始终只读。

五槽的COCO-17绑定在真实执行前固定为：`head=(0..4)`；`upper torso+arms=(5..10)`；
`lower torso=(11,12)`并使用左右shoulder-to-hip与hip-to-hip段；`upper legs=(11,13)/(12,14)`；
`lower legs+feet=(13,15)/(14,16)`。重叠响应使用hard-owner分配，任何像素不得同时属于两个槽；CLIP
16x16 patch只要具有正mask mass即属于该槽。该taxonomy、prompt或patch选择规则不得按结果更换。

真实teacher测量按信息增益串行：先完成`exp405-p0b-iso-teacher-v1`，全train缓存original五槽与global
image/text state，并在按PID/槽分层冻结的2,000图上测25/50/75%删除和wrong-mask；只有P0B通过才创建
`exp405-p0c-transport-oracle-v1`。这两个execution id各自once-only，前者失败不得用后者补救。RGB删除位置
由v14联合hash固定，填充值为逐通道CLIP mean；50%是唯一primary。

## Fresh与once-only

P0B CUDA wiring preflight id固定为`exp405-p0b-preflight-v1`，唯一输出根为
`/home/afr/reid-clean/audits/exp405-p0b-preflight-v1`；P0B formal id固定为
`exp405-p0b-iso-teacher-v1`，唯一输出根为
`/home/afr/reid-clean/audits/exp405-p0b-iso-teacher-v1`。两者使用显式互斥的`preflight/formal`模式，默认不进入
formal；formal必须验证固定preflight PASS receipt、冻结manifest及其SHA。每个execution在输出根外使用排他
STARTED seal，异常也写FAILED receipt并永久封板同一id。P0C只有在P0B scientific GO后才允许创建独立的
`exp405-p0c-transport-oracle-v1`根，不得与P0B共用cache或execution id。

preflight样本数固定`512`，由全train的`PID/camera/path`联合hash分层冻结；每槽只取4个recipient，其余样本作为
donor reserve。preflight只裁决shape/finite、region readout、精确删除、same-camera/different-PID wrong-mask接线
与状态恢复，不计算PID CI、non-torso macro或科学GO，结果必须写`scientific_evaluated=false`。formal与P0C只认
最终`complete.json`；单独存在的result/cache或同时存在FAILED receipt均无授权语义。

不得读取旧cache/path mapping作为运行输入；需要的checkpoint必须复制到fresh asset目录并记录SHA。

只有成功取得本次fixed execution seal才算execution开始；在此之前的参数、旧终态、receipt、manifest、source、runtime
或数据合同拒绝均保持目标目录只读，不产生FAILED，也不消耗once-only execution。取得seal后的任何runtime/measurement错误
都必须写FAILED并封板；修正必须使用新实验编号，不得在exp405下v2重跑。
formal执行前manifest必须绑定repo HEAD、core、runner、protocol、运行参数、CLIP checkpoint、pose artifact、
有序official train manifest、preflight receipt和runtime freeze的SHA256；执行前后都要复核绑定不变。P0C未来
另行绑定D0 checkpoint、stage replay adapter、wrapper/postflight与正式counterfactual config。

formal有效性门预注册：无可用target图像比例不得超过`1%`，具有至少一个可用target的PID比例不得低于`99%`；
每槽仍须冻结足400个诊断样本。wrong-mask硬约束same-camera/different-PID/analysis-valid/图像不复用，四个主匹配量
为log mask mass、y-centroid、pose confidence与CLIP support；按每槽全候选MAD标准化，主距离caliper固定`8.0`，
先取每个recipient排序前64个候选并用确定性增广匹配求一对一解；若Hall约束下无完整解，按
`64 -> 128 -> 256 -> 全部caliper内候选`扩展，只有完整caliper图仍无解才判validity FAIL。全部诊断recipient
都从donor集合中排除，donor之间也不得复用；任一槽balance、caliper或完整一对一匹配失败均为validity FAIL。

## 串行门

1. 文献与三路独立机制审查完成；
2. design/protocol/monitor冻结；
3. synthetic CPU正合同与mutant反合同连续两次byte-exact；
4. 新代码盲审`BLOCKER=0`；
5. 远端static、数据路径只读、独占4090和fresh output门通过；
6. 必要CUDA/AMP preflight通过；
7. 才允许唯一teacher-only execution。

任何时刻只允许一个4090任务。Phase 0不得并行，不按中间数值停掉已启动的有效measurement。

截至2026-07-20，串行门1--4已经完成：v14两次fresh CPU结果byte-exact、`56/56 PASS`，最终独立盲审
`0B/0H/0M/0L`。当前只进入真实teacher measurement实现；门5--7仍未通过，GPU与真实执行保持NO-START。
静态启动器不再扩展新的供应链/receipt威胁模型；后续阻塞项必须能直接改变真实科学结论。

真实teacher measurement的最终v8已两次fresh `8/8 PASS`且byte-exact，固定结果SHA256为
`45413c3323f7af7636e1e2f9e581b4a9c5fe15c44d4b0a6e47aa987c0ef9f8ca`；代码、复现/once-only、统计/matching
三路固定快照盲审均为`0 BLOCKER / 0 HIGH`。这只闭合本地门1--4，下一步仍是门5的远端环境/资产/独占GPU
检查和唯一512图CUDA preflight；formal P0B与任何student训练继续NO-START。

## 机械有效性

必须记录RGB/pose同步变换、左右翻转、mask mass/centroid/index/hash、CLIP normalization、16x16 patch布局、
geometry/support字段分离、slot permutation等变、NULL identity和self-slot restore上界。self restore失败即
measurement FAIL，科学结果不成立。

## Teacher反事实臂

每个recipient固定配齐：

1. same-ID/same-slot correct；
2. same-ID/wrong-slot；
3. wrong-ID/same-slot；
4. pose-only；
5. image-only；
6. text-only/static；
7. generic mean；
8. NULL；
9. random-key；
10. frequency-matched random-cluster；
11. self-slot restore；
12. MVI2P-full；
13. attribute-relation；
14. generic-transport。

wrong-ID donor匹配camera、support、mask mass并尽量匹配global CLIP similarity。任何control不得被删除或弱化。

## 主指标与判定

同时报告descriptor恢复、same-ID identity margin恢复、冻结train-only mAP/R1、每槽结果、非torso macro、
PID-cluster bootstrap CI和PID正/零/负计数。只恢复cosine、不恢复identity margin与mAP/R1不能通过。

correct必须分别高于same-ID/wrong-slot、wrong-ID/same-slot、pose-only、image-only、text-only、generic、NULL、
random-key、random-cluster和generic-transport；CLIP还必须相对pose-only有独立增量。matched MVI2P-full或
pose-part若同等/更好且CAVT无额外二维偏序，判近邻已解释。

teacher全部通过后，按PID隔离fit/validation做donor-free probe。recipient `not-k`状态预测teacher residual的
cosine/R2/rank必须优于zero、identity-mean和generic linear predictor，否则不实现student。

## 后续正式训练的epoch比较合同

只有Phase 0全部通过后，后继正式实验才可创建。评估点固定e10/20/.../120，每点必须同时记录方法与sealed
clean D0同一epoch的`mAP/R1`和差值。不同epoch、best epoch或不同训练预算不得作为涨点证据；最终只以自然完成
e120裁决，并保留所有强反事实与all-bypass。
