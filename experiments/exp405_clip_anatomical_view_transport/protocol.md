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

## Fresh与once-only

唯一Phase 0 execution id为`exp405-p0-iso-oracle-v1`，fresh远端根计划为
`/home/afr/reid-clean/audits/exp405-p0-iso-oracle-v1`。不得读取旧cache/path mapping作为运行输入；需要的
checkpoint必须复制到fresh asset目录并记录SHA。

若runtime/measurement contract错误，本execution仍封板；修正必须使用新实验编号，不得在exp405下v2重跑。
执行前manifest必须绑定repo HEAD、core、runner、wrapper、postflight、config、CLIP checkpoint、D0 checkpoint、
pose artifact和runtime freeze的SHA256。

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
