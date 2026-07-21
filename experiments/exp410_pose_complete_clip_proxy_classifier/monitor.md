# exp410 PC²P 监控

## 2026-07-21：对象重置与设计冻结

exp409已永久封板为`SEALED NO-GO / RANK-1 PASS / mAP FAIL`；4090恢复`2 MiB/0%/0 compute PID`。
三路独立审计比较了pose×CLIP语义缺失协方差与pose-complete classifier对象，最终选择PC²P：它把作用对象从
单个hard pair改为全部702类分类几何，更直接回应exp409只改善R1、不改善mAP的结果。

初稿中的可学习`Q:768→768`被独立审计判为科学HIGH并删除：由于类别数702小于特征维768，`P @ Q`几乎可表达
任意learned classifier，既会让proxy机制退化，也增加推理成本。冻结方案无Q/adapter/projection：
`BN(global_feat) @ frozen_pose_complete_proxy.T`直接替换learned classifier，原triplet和eval global descriptor不变。

近期近邻审计确认CLIP-ReID已有冻结身份text feature的I2T CE，ProFD已有part prompt/centroid/memory，普通固定proxy
分类本身不能声称新颖。PC²P只保留C类窄主张：pose五槽跨同PID多图补全的visual identity-set proxy、无adapter
替换learned classifier、测试期恢复原global descriptor。问题门PASS、证据门PASS、机制门CONDITIONAL PASS。

当前状态=`DESIGN/PROTOCOL FROZEN / IMPLEMENTATION NEXT / GPU IDLE`。下一步实现fresh bank builder、严格loader和
最小model接线；必要合同及一次独立智能体盲审`0B/0H`后立即fresh运行，不增加无穷preflight。

## 2026-07-21：实现与独立盲审通过

已完成默认关闭的三个config开关、fresh bank builder、严格train-only loader、processor外部bank持有、model内
FP32 `BN(global_feat) @ proxy.T`接线和独立exp410 config。PC²P不创建Q/projection/adapter或任何新参数；开启时
原classifier不被调用，triplet仍读取原global descriptor，eval不接收bank。

固定MMPOSE-ABU已通过语法、模块导入和config解析。对冻结exp409 source cache的只读覆盖统计为15,618图、702 PID，
每个PID五槽支持最小计数均为6、空槽为0；GPU保持空闲。唯一独立智能体代码盲审最终`0 BLOCKER / 0 HIGH`，确认
default-off state/RNG/真实forward/loss exact、builder-loader字段、PID/path/RGB/provenance、互斥、proxy冻结、
FP32 logits/梯度路线和无bank eval合同完整。

当前状态=`IMPLEMENTATION REVIEW 0B/0H / FRESH BANK NEXT / GPU IDLE`。下一步显式提交目标文件并在fresh clean远端
repo构建唯一bank；写回真实SHA后只执行一次真实PK batch64 CUDA/AMP合同，通过即启动e120。

## 2026-07-21：fresh proxy bank完成

在fresh clean远端repo `/home/afr/SOLIDER-REID-exp410-pc2p-cb3486b-v2`（build source HEAD
`a3be68f7ae368ba3700767b88a11e8cf3f61c36a`）以固定MMPOSE-ABU完成唯一fresh bank构建；过程为CPU聚合，GPU
未被占用。bank覆盖15,618图/702 PID，shape=`[702,768]` FP32，每行单位范数且无重复；五槽每PID支持最小计数
均为6，最大计数=`[425,426,426,426,425]`。

- bank SHA=`8f435036d56b2a5a1a8e63466b383314f07d706a7465a499aef88a5f7435dc8c`；
- manifest SHA=`d31b0c1e4b1ee211979f6dda05ff4977925149719eafebf788a99578891a829b`；
- builder SHA=`71a41fe85a76f919b5d987c9f960250d4a49779093199f7d60ba42ac6b155a69`；
- loader SHA=`f6fe1930f81e44fd7eaf624ae6808f56e891ebbd3ed74866a192b628b6048b72`；
- official path/RGB/PID mapping SHA分别为
  `e53ef9189f12737d6621ae152979cf2d12f8bb24cc823466a6ef11928bd99f4e`、
  `10176dd5dd3e54f7139a43abca61fdf147766c06e5f58c04b8cf28795fb9ea5a`、
  `56d53771bf0fbb4978ee51d118f921c61763ce4d2aeede7611740ec705c630d4`。

真实bank SHA已写回冻结config。当前状态=`FRESH BANK PASS / REAL PK64 CUDA-AMP NEXT / GPU IDLE`。

## 2026-07-21：唯一真实PK batch64 CUDA/AMP合同通过

固定MMPOSE-ABU合同自然完成并正常退出，runner SHA=
`51ee59d9f9bf701daad94b1af9b93e4bcaa653464fea509ab4d3506cdb481d1c`。结果为：

- default-off state/RNG/同RNG真实forward和combined loss exact；
- logits=`[64,702]` FP32，mean/std/abs-max=`1.69e-08/1.0203568/2.7546368`，BN norm=`27.4120655`；
- CE-only对BNNeck/norm3/Stage-3的非零梯度tensor=`1/2/26`；
- 原classifier梯度为None，bank不进model state/optimizer且无梯度；
- combined loss/reid/pose=`19.3192997/19.2277889/0.9151036`；default GradScaler由65536自然backoff，
  第6个native attempt在scale 2048取得真实update；Stage-3/backbone非零梯度tensor=`26/181`；
- 无bank/CLIP/外部pose的eval返回finite `[64,768]`原global descriptor；
- 合同退出后GPU=`2 MiB/0%/0 compute PID`。

config SHA=`f099b7f778e376f9ff12787d1bd6bb21de3ea37be50cfac7866f155f77ad6cba`。当前状态=
`IMPLEMENTATION/BANK/REAL-PK64 PASS / E120 AUTHORIZED / GPU IDLE`。不再追加测试，下一步从fresh clean repo启动
唯一seed1234/e120 correct arm。

## 2026-07-21：唯一fresh correct student已启动

formal fresh gates全部PASS：repo=`/home/afr/SOLIDER-REID-exp410-pc2p-formal-d38a3d4-v1`，source HEAD=
`d38a3d415fa97e2ada5ba9157dfb5600adcb75e9`；config/bank SHA分别为
`f099b7f778e376f9ff12787d1bd6bb21de3ea37be50cfac7866f155f77ad6cba`/
`8f435036d56b2a5a1a8e63466b383314f07d706a7465a499aef88a5f7435dc8c`。启动前repo clean、GPU无compute
PID，output和runner均fresh。

唯一训练主PID=`539255`，output=`/home/afr/reid-clean/logs/exp410-pc2p-s1234-v1`，runner=
`/home/afr/reid-clean/train-logs/exp410-pc2p-s1234-v1.runner.log`。首batch BN norm/logit mean/std/abs-max=
`27.435509/-0.000000/0.961771/3.391294`，预测类别unique=`48`，全部finite。首次观测e1 iter160/227，
loss=`13.530`，GPU=`6964 MiB/94%`且只有该compute PID，无Traceback/RuntimeError/OOM/NaN/Inf。

当前状态=`UNIQUE FRESH E120 RUNNING / SOURCE+CONFIG+BANK FROZEN`。运行中不修改formal repo、config、bank或参数；
只在e10/20/.../120记录与sealed clean D0的同epochmAP/R1，不按中间点早停。

## 正式同epoch轨迹

| Epoch | PC²P mAP/R1 | sealed clean D0 mAP/R1 | rounded ΔmAP/ΔR1 |
|---:|---:|---:|---:|
| 10 | 31.7/42.0 | 33.4/42.7 | -1.7/-0.7 |
| 20 | 36.8/48.2 | 42.2/52.4 | -5.4/-4.2 |
| 30 | 41.2/51.7 | 46.6/56.2 | -5.4/-4.5 |
| 40 | 41.7/51.5 | 50.0/60.7 | -8.3/-9.2 |
| 50 | 42.7/54.0 | 52.1/62.8 | -9.4/-8.8 |
| 60 | 43.1/55.1 | 55.1/66.1 | -12.0/-11.0 |
| 70 | 43.9/54.9 | 55.4/65.2 | -11.5/-10.3 |

e10训练自然完成后评测得到R5/R10=`57.1/63.1`；进程随后自然进入e11。此时主PID仍为`539255`，GPU只有该
compute PID，runner无Traceback/RuntimeError/OOM/NaN/Inf。e10双指标落后只是中间轨迹，不触发早停、续训或
任何运行中修改。

e20/30/40完整R5/R10依次为`64.2/71.1`、`67.3/73.6`、`66.0/72.0`。PC²P在e20--e40持续双指标落后，且e40
差距扩大；这是明确的不利中间证据，但冻结协议仍只以自然e120裁决。最新检查已进入e44，主PID仍为唯一compute
PID，GPU约`7.1 GiB`，异常计数为0；继续运行，不修改source/config/bank/参数。

e50/60/70完整R5/R10依次为`69.0/74.8`、`71.2/76.2`、`69.8/76.2`。双指标差距在后段维持于约
`-9`至`-12` point，固定proxy没有显示追平D0的迹象，但仍不能把中间轨迹改写为最终e120。最新已进入e79，
主PID和GPU独占正常，异常计数0；继续自然训练。
