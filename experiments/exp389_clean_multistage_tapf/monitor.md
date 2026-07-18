# exp389 监控记录

## 当前状态

- 状态：`RUNNING`，fresh e120 HT0 已正式启动；
- 4090：唯一训练进程占用；
- 正式 output：`log/occluded_duke/exp389_clean_swin_tiny_ht0_s1234`；
- 直接对照：exp387 clean Occ-Duke D0=`57.6/67.7/80.8/84.6`；
- 当前阶段：只读监控唯一的 e120 HT0，自然跑满后做 final 终审。

## 初始设计审计

- 禁止复用旧 HT0 runtime、旧 pose_data/cache/path mapping；输入继续只使用 exp386 fresh ViTPose-H train-only artifact；
- 既有 late Stage-2 anchor → Stage-3 两 PSG 必须逐参数、构造顺序和数值路径保持 D0 exact；
- 新增 early anchor 从 Stage-1 pre-downsample feature 产生场，六个独立 consumer 分别位于 Stage-2 每个 block 后；
- 八个 consumer 都位于最终 GAP 使用的 spatial feature 上游，不接受 terminal dead consumer；
- 两层均使用同一 teacher/handoff/student 日程，eval 均只读 RGB internal field；
- config-off、D0-off、双层 route/gradient、真实 batch64 CUDA/AMP/overflow、strict state、pose-free parity、consumer path 与效率任一未通过前，保持 `NO-START` 和 GPU 空闲。

## 首版实现与基础门禁

- 本地实现提交=`e6c49c4`；远端独立 preflight repo=`/home/afr/SOLIDER-REID-exp389-preflight-e6c49c4`，执行提交=`bbf272c49c9aa1159b61b16919665255c3a76a7b`；
- HT0 config SHA256=`f4b6cfde243de97634eef9320a7a96e2d58f6cd0fc747ee4a747997da455675b`；正式 output 尚未创建，GPU 在门禁前后均空闲；
- `CleanTapfHt0` 先构造完整 `CleanTapfD0`，随后追加独立 early anchor 与六成员 early PSG bank；`MODEL.TAPF.HIERARCHICAL=False` 为默认值；
- Swin 仅在 hierarchical 分支手动展开 Stage-2 六个 block，并保持原生 block→gate→下一 block、最终 downsample 顺序；D0 分支不进入该代码；
- unit 由既有 5 项扩展为 6 项，Gaussian/reliability/空 valid/schedule/zero-field identity/D0 公共初始化/early 三 bank route/early-late 参数独立/eval exploding pose 全部 PASS；
- 真实 Swin-T batch2 CUDA/AMP train/backward smoke PASS：early/late field shape=`17×48×16` / `17×24×8`，e6 student fraction=`0.2/0.2`，early/late gate route=`6/2`，两个 anchor 与两组 PSG 均获得有限梯度，最终 feature=`2×768` 且全量有限；
- 可执行结论：`EXP389_FULL_MODEL_CUDA_SMOKE_PASS`。

该结果只通过基础实现门禁，不替代 config-off/D0-off exact、真实 paired batch64/24-step、严格 gradient ownership、overflow、strict state、pose-free、逐 consumer path 与效率门禁；状态继续为 `NO-START`。

## 完整正式训练前门禁

### config-off 与 D0-off 精确等价

- pre-TAPF 官方代码与新代码 `TAPF.ENABLED=False` 的三步完整输出、梯度、state 与 optimizer 指纹逐项 exact，fingerprint=`e12e187da94e2b599fe3c986d38dee76310742526f9498a0d6c6fb8aa8c39d77`；
- exp387 D0 与新代码 `TAPF.ENABLED=True,HIERARCHICAL=False` 的三步完整指纹 exact，fingerprint=`3e0415c796b2cd2fa908b1fb4cfeafe5a9c0a2cdcb093c992e197d8ce0764b3e`；
- 另用独立脚本 `preflight_d0_exact.py` 做默认 GradScaler 十步受控复测。旧 exp387 commit=`0d1822a07dda8daac0210b68916035b1886d5d99` 与新代码 commit=`04f1086ed3daddce9c3c4c83fe26550d96d1a206` 使用同一固定 RGB、pose、seed、loss 和 step RNG，两个完整 JSON 逐字节 exact，JSON SHA256=`b2f19d3f97e4d2d6c4d60241364876be0562605680058031779897d3e4499d16`；
- 十步受控复测 state tensor=`223`、momentum buffer=`185`，scale 轨迹=`32768/16384/8192/4096/2048/1024/512/512/512/512`，initial/final/momentum SHA256 分别为 `017e9870bbbba1f2d460d62ea1fd81ba2cfdb13f92cf1dd5a02efa90a35777ad`、`134c34beb9422298e2ad61a9b07a68531103ff08872c0614e2e88c44e5ef5f47`、`3c453e997e74aa63e36157079bcbbfd16b5575a6064ddb6de48d15d346d253ca`；
- 可复现脚本 SHA256=`7882a631559b8d8b6c4c751fa79edb8201e329a275ebcc407d3f66929434f01b`。此前一次临时检查因两次独立进程没有共享固定输入而给出不同聚合指纹，不构成代码差异；受控逐字节比较取代该无效临时结果。

### 构造、参数与优化器边界

- full-model common state=`223`，HT0 extra state=`20`；D0/HT0 参数量=`28,179,484/28,287,102`，唯一新增=`107,618`；
- common optimizer parameters=`191`、extra optimizer parameters=`20`，公共 state、构造后 CPU/CUDA RNG、公共 optimizer 成员/顺序/超参数全部 exact；
- HT0 先构造完整 D0 late path，再追加 early anchor 和六个独立 PSG，D0 公共参数名、初始化值与随机数消耗不变。

### 真实数据 CUDA/AMP、路由与梯度

- exp386 strict paired loader、batch64、8 workers 连续 24 step 原生 CUDA/AMP PASS；默认 GradScaler 从 `65536` 回退至 `1024` 后连续有限更新，共 18 次真实 optimizer update；
- 参数更新覆盖 early anchor=`8/8`、early PSG=`12/12`、late anchor=`8/8`、late PSG=`4/4`、Swin=`171/193`、head=`2/3`，optimizer state tensor=`205`；peak allocated/reserved=`7,018,148,352/7,488,929,792 B`；
- e1/e6/e10/e11 两层 student fraction 均为 `0/0.2/1/1`，每次 forward early/late consumer route=`6/2`；
- pose supervision ownership：early pose loss 只进入 early anchor `8/8`，late pose loss 只进入 late anchor `8/8`；ReID loss 进入 early 六个 PSG、late 两个 PSG、Swin 与 head，两个 anchor 均无 ReID 梯度。零初始化 output projection 在首步仍保留预期的下游梯度边界；
- 真实 overflow 门禁：scale=`1→0.5`，model state=`243`、optimizer state=`205` 在 nonfinite 整步前后逐张量 exact，证明 GradScaler 确实整步跳过。

### strict state、pose-free 与 consumer 可达性

- strict save/load state tensor=`243`；correct/shuffle/None/exploding 四种外部 pose 输入的 descriptor、early/late student field 与八个 gate delta 全部 exact，exploding pose access=`0`；
- 八个 consumer 逐一旁路均使最终 descriptor 产生有限非零变化，max absolute delta 依次为 `0.0245/0.0444/0.0648/0.1450/0.2691/0.3392/0.3721/0.3834`；六个 early 与两个 late consumer 全部存在到最终 GAP descriptor 的真实下游路径，无 terminal dead consumer。

### matched 效率

- 参数增量=`107,618`，相对 D0 为 `+0.381902%`；
- analyzer 支持算子口径 FLOPs=`5,548,787,520→5,588,139,072`，增量=`39,351,552 / +0.709192%`；不将 analyzer 未支持算子包装成完整 FLOPs；
- train batch64 latency=`100.061→105.356 ms`，peak allocated=`6,080,495,104→6,621,345,792 B`；
- eval batch256 latency=`228.036→243.524 ms`，peak allocated=`4,843,570,176→4,844,131,328 B`。

以上 unit、config-off、D0-off、state/RNG/optimizer、真实 paired CUDA/AMP、route、gradient、overflow、strict state、pose-free、八 consumer 路径与效率门禁均 PASS。正式变量只剩新增 early hierarchy，允许进入 fresh e120 HT0；不得并行、续训、提前停止或修改运行中的代码/config。

## 正式启动

- fresh formal repo=`/home/afr/SOLIDER-REID-exp389-ht0-0d8436b`，detached exact HEAD=`0d8436b7271cb4f0cce44f3b655f821abc867d92`，tracked 与 staged source clean；
- full-history bundle=`/home/afr/reid-clean/bundles/exp389_ht0_0d8436b.bundle`，`git bundle verify` 确认记录完整历史，SHA256=`44f818dec9f50bf87bda046925f351fab48321b5f827cc645af723330eac978d`；
- formal config SHA256=`f4b6cfde243de97634eef9320a7a96e2d58f6cd0fc747ee4a747997da455675b`；output 与 runner 启动前均不存在，GPU 空闲，fresh repo unit=`6/6 PASS`；
- 启动时间=`2026-07-18 00:47:47`，main PID=`1091044`，runner=`/home/afr/train-logs/exp389_clean_ht0_s1234.runner.log`，环境=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- recipe 与 exp387 D0 matched：Occluded-Duke、batch64、seed1234、120 epoch、SGD、lr=`0.0008`、semantic weight=`0.2`、eval10、唯一 e120 checkpoint；唯一变量为 early hierarchy；
- 初检唯一 main+8 workers，GPU=`7.35–7.53 GiB`，e1/e2 自然完成并进入 e3；e1 time=`25.733 s`、speed=`522.3 samples/s`，e2 time=`25.281 s`、speed=`536.7 samples/s`；
- e1 teacher route：StudentEarly/StudentLate=`0/0`，PoseEarly/PoseLate=`0.939/0.915`（epoch 末累计），GateEarlyAbs/GateLateAbs=`9.436e-05/1.092e-04`，两层均已产生有限非零 gate；截至 e3 iter100，NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`。

当前只允许继续自然训练；每次 e10 eval 现场相对 exp387 D0 同 epoch计算 mAP/R1/R5/R10，禁止依据中间单点提前停止。

## e10–e40 完整评测轨迹

### e10

- HT0 mAP / R1 / R5 / R10=`34.2 / 44.4 / 59.7 / 65.8`；
- 同 epoch exp387 D0=`33.4 / 42.7 / 59.8 / 65.2`；
- HT0−D0=`+0.8 / +1.7 / −0.1 / +0.6`；
- e10 末尾 PoseEarly/PoseLate=`0.785/0.786`、StudentEarly/StudentLate=`1/1`、ReliabilityEarly/ReliabilityLate=`0.852/0.852`、GateEarlyAbs/GateLateAbs=`3.823e-03/6.206e-03`。

### e20

- HT0 mAP / R1 / R5 / R10=`42.8 / 53.1 / 68.9 / 74.4`；
- 同 epoch exp387 D0=`42.2 / 52.4 / 67.6 / 74.0`；
- HT0−D0=`+0.6 / +0.7 / +1.3 / +0.4`；
- e20 末尾 PoseEarly/PoseLate=`0.546/0.564`、StudentEarly/StudentLate=`1/1`、ReliabilityEarly/ReliabilityLate=`0.839/0.839`、GateEarlyAbs/GateLateAbs=`1.007e-02/1.762e-02`。

### e30

- HT0 mAP / R1 / R5 / R10=`47.7 / 58.3 / 72.0 / 77.1`；
- 同 epoch exp387 D0=`46.6 / 56.2 / 71.3 / 76.4`；
- HT0−D0=`+1.1 / +2.1 / +0.7 / +0.7`；
- e30 末尾 PoseEarly/PoseLate=`0.474/0.492`、StudentEarly/StudentLate=`1/1`、ReliabilityEarly/ReliabilityLate=`0.836/0.836`、GateEarlyAbs/GateLateAbs=`1.477e-02/2.194e-02`。

### e40

- HT0 mAP / R1 / R5 / R10=`49.0 / 59.3 / 74.0 / 79.0`；
- 同 epoch exp387 D0=`50.0 / 60.7 / 76.2 / 81.0`；
- HT0−D0=`−1.0 / −1.4 / −2.2 / −2.0`；
- e40 末尾 PoseEarly/PoseLate=`0.455/0.473`、StudentEarly/StudentLate=`1/1`、ReliabilityEarly/ReliabilityLate=`0.833/0.833`、GateEarlyAbs/GateLateAbs=`1.659e-02/2.262e-02`。

四次 eval 均为完整 query/gallery 评测。e10–e30 的正差与 e40 的负差共同说明中间轨迹存在波动，不能选择局部节点或用单一 epoch 裁决。评测后训练自然进入 e41；exact HEAD/config/tracked source clean、唯一 main+8 workers、GPU 约 `13.34 GiB`、checkpoint 尚未生成，NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow=`0`，继续运行至 e120。

## e50–e70 完整评测轨迹

### e50

- HT0 mAP / R1 / R5 / R10=`52.7 / 62.1 / 76.4 / 81.7`；
- 同 epoch exp387 D0=`52.1 / 62.8 / 77.0 / 81.9`；
- HT0−D0=`+0.6 / −0.7 / −0.6 / −0.2`；
- e50 末尾 PoseEarly/PoseLate=`0.450/0.468`、ReliabilityEarly/ReliabilityLate=`0.830/0.830`、GateEarlyAbs/GateLateAbs=`1.775e-02/2.294e-02`。

### e60

- HT0 mAP / R1 / R5 / R10=`54.5 / 64.8 / 78.6 / 83.9`；
- 同 epoch exp387 D0=`55.1 / 66.1 / 79.0 / 83.3`；
- HT0−D0=`−0.6 / −1.3 / −0.4 / +0.6`；
- e60 末尾 PoseEarly/PoseLate=`0.448/0.466`、ReliabilityEarly/ReliabilityLate=`0.847/0.847`、GateEarlyAbs/GateLateAbs=`1.845e-02/2.313e-02`。

### e70

- HT0 mAP / R1 / R5 / R10=`55.1 / 64.6 / 78.8 / 83.1`；
- 同 epoch exp387 D0=`55.4 / 65.2 / 79.5 / 83.6`；
- HT0−D0=`−0.3 / −0.6 / −0.7 / −0.5`；
- e70 末尾 PoseEarly/PoseLate=`0.445/0.463`、ReliabilityEarly/ReliabilityLate=`0.855/0.855`、GateEarlyAbs/GateLateAbs=`1.876e-02/2.288e-02`。

三次 eval 均为完整 query/gallery 评测。两层 StudentEarly/StudentLate 持续为 `1/1`，pose supervision 与 early/late PSG 均保持有限 active；e50–e70 仍是混合正负波动，不改变自然跑满协议。评测后训练已进入 e73；exact HEAD/config/tracked source clean、唯一 main+8 workers、GPU 约 `7.44 GiB`、checkpoint 尚未生成，严格异常=`0`。

## e80–e100 完整评测轨迹

### e80

- HT0 mAP / R1 / R5 / R10=`55.4 / 65.4 / 78.9 / 82.9`；
- 同 epoch exp387 D0=`56.1 / 66.3 / 79.5 / 84.0`；
- HT0−D0=`−0.7 / −0.9 / −0.6 / −1.1`；
- e80 末尾 PoseEarly/PoseLate=`0.442/0.462`、ReliabilityEarly/ReliabilityLate=`0.833/0.833`、GateEarlyAbs/GateLateAbs=`1.913e-02/2.282e-02`。

### e90

- HT0 mAP / R1 / R5 / R10=`56.5 / 66.1 / 79.8 / 84.4`；
- 同 epoch exp387 D0=`57.5 / 67.9 / 81.2 / 85.3`；
- HT0−D0=`−1.0 / −1.8 / −1.4 / −0.9`；
- e90 末尾 PoseEarly/PoseLate=`0.443/0.463`、ReliabilityEarly/ReliabilityLate=`0.831/0.831`、GateEarlyAbs/GateLateAbs=`1.934e-02/2.318e-02`。

### e100

- HT0 mAP / R1 / R5 / R10=`56.4 / 65.9 / 79.2 / 84.3`；
- 同 epoch exp387 D0=`56.9 / 67.1 / 79.6 / 83.8`；
- HT0−D0=`−0.5 / −1.2 / −0.4 / +0.5`；
- e100 末尾 PoseEarly/PoseLate=`0.443/0.462`、ReliabilityEarly/ReliabilityLate=`0.851/0.851`、GateEarlyAbs/GateLateAbs=`1.934e-02/2.315e-02`。

三次均为完整 query/gallery 评测；e80–e100 大部分指标低于同 epoch D0，但 R10 在 e100 为正，仍不得以中间节点裁决。训练已自然进入 e107；两层 Student=`1/1`、pose supervision 与八个 PSG 持续 finite/active，exact HEAD/config/tracked source clean、唯一 main+8 workers、GPU 约 `7.33 GiB`、e120 前无 checkpoint、严格异常=`0`，继续至 e120。

## e110 与 e120 final

### e110

- HT0 mAP / R1 / R5 / R10=`56.6 / 65.9 / 79.5 / 83.9`；
- 同 epoch exp387 D0=`57.4 / 67.4 / 80.5 / 84.6`；
- HT0−D0=`−0.8 / −1.5 / −1.0 / −0.7`；
- e110 末尾 PoseEarly/PoseLate=`0.442/0.461`、ReliabilityEarly/ReliabilityLate=`0.857/0.857`、GateEarlyAbs/GateLateAbs=`1.939e-02/2.335e-02`。

### e120 final

- HT0 mAP / R1 / R5 / R10=`56.9 / 65.9 / 80.0 / 84.1`；
- exp387 D0 e120=`57.6 / 67.7 / 80.8 / 84.6`；
- **HT0−D0=`−0.7 / −1.8 / −0.8 / −0.5`**；
- exp385 official B0 e120=`57.4 / 67.4 / 80.6 / 85.2`，HT0−B0=`−0.5 / −1.5 / −0.6 / −1.1`；
- e120 末尾 PoseEarly/PoseLate=`0.443/0.462`、StudentEarly/StudentLate=`1/1`、ReliabilityEarly/ReliabilityLate=`0.828/0.828`、GateEarlyAbs/GateLateAbs=`1.938e-02/2.302e-02`；
- 全程自然运行至 e120，报告唯一 final checkpoint；没有挑选 e30 等局部正节点，也没有按阈值或单点提前停止。

## e120 终审

- 原 main PID=`1091044` 与全部 8 workers 自然退出，GPU 空闲；output 仅 `train_log.txt` 和唯一 `transformer_120.pth`；
- checkpoint SHA256=`b78e4a62258cd16d5181b6e224880fb521e53af893f7eaa522f07dde9ac15d61`，runner SHA256=`19e5dd625a33c56114491e10173bfe7ef5ad7941f8dbd887abe265f71c6c96b0`，train log SHA256=`6806702b9f740dfeb9af6ec52c39ab7ba772b393f1dd0780e0d197b02b8c6095`；
- exact HEAD/config 与 tracked/staged source clean；runner/train log 的严格边界词 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow 命中=`0`；
- checkpoint state tensor=`243`，其中浮点或复数 tensor=`230`，全部有限；strict load missing/unexpected=`0/0`；
- 相对同 seed、同 dataloader→model 构造顺序的 fresh 初始化，early anchor=`8/8`、early PSG=`12/12`、late anchor=`8/8`、late PSG=`4/4`、Swin=`171/193`、head=`2/3` 参数 tensor changed；六个 early 与两个 late bank 均各自 `2/2 changed`，组内任意 bank pair 均不相等；
- final checkpoint 上 correct/shuffle/None/exploding external pose 的 descriptor、early/late field 与八个 gate delta 逐元素 exact，exploding pose access=`0`；normal-train evaluator 与 query/gallery 均为无 pose store 的 `ImageDataset`；
- final 逐 consumer 旁路的 descriptor max absolute delta：early0–5=`0.1450/0.0679/0.0767/0.1382/0.2606/0.6103`，late0–1=`1.0834/2.1360`；八个 learned consumer 均保持到最终 descriptor 的有限非零下游路径，无 terminal dead consumer；
- final audit 脚本 SHA256=`80f9c8234849a5dbd2ee495ff1c6aa2b499dc09bac3bbbb2e61964b05418122a`，canonical JSON SHA256=`2546fbdbcbb420b2555ccc9277af3aef6ebdb6f29a53be628e6f42ab6baa6cc8`；
- 可执行终审结论：`EXP389_FINAL_AUDIT_PASS`。

最终结论边界：在官方最后代码、fresh ViTPose-H train-only target、同一 seed/batch/120-epoch recipe 下，新增 clean early hierarchy 相对单层 D0 的 final 四项全部下降，且低于 official B0。实现、参数学习、RGB-only 推理与八条 consumer 路径均已排除失效，因此这是有效负结果。exp389 至此封板，禁止重启、续训、重复或用中途 best 替代 e120；hierarchical 继续只作为 backbone-conditional 历史扩展，不进入 clean 方法 headline。

## 2026-07-18 冻结层级旁路诊断

用户追问多阶段是否可能因实现方式不理想而失败。为区分“early consumer在推理时直接有害”和
“新增层级在训练期干扰late/backbone优化”，对封板 e120 checkpoint 做只读冻结评测；不修改权重、
config或训练，不读取external pose。顺序固定为full→early六bank全旁路→late两bank全旁路→八bank
全旁路→full-repeat，最后一次full逐位复现首轮结果，排除hook恢复漂移。

| arm | mAP/R1/R5/R10 | 相对full |
|---|---:|---:|
| full | `56.8605/65.9276/79.9547/84.1176` | `0/0/0/0` |
| early-bypass | `56.7882/65.9276/79.4118/83.8914` | `−0.0723/+0.0000/−0.5430/−0.2262` |
| late-bypass | `55.5040/64.0271/78.2353/82.5339` | `−1.3565/−1.9005/−1.7195/−1.5837` |
| all-bypass | `55.4369/64.1629/77.5113/82.5339` | `−1.4236/−1.7647/−2.4434/−1.5837` |

结论：early层不是dead path，但冻结checkpoint上的独立mAP贡献只有约`+0.07`、R1为`0`；late层仍
提供约`+1.36 mAP`。旁路early不能把HT0恢复到exp387 D0=`57.6`，所以exp389相对D0的主要退化
不是推理时early gate本身直接减分，而是训练期early六个PSG已通过ReID路径改写Stage-2/backbone/
late输入的联合轨迹。

独立复核同时收紧了一个容易过度归因的点：exp389把early+late pose loss求和后乘`0.1`，确实令
日志中的总辅助标量相对D0增加，但两个source均detach，pose loss只更新各自anchor；late anchor仍
保持与D0相同的`0.1`系数，Swin/PSG/head不接收pose梯度。因此不能直接写“pose总强度翻倍干扰
backbone/late”，明确的全局耦合只剩GradScaler overflow/整步skip，而当前没有生产skip差异能作
因果证据。更强的结构差异是early同一field连续经过六个独立Stage-2 PSG，late只有两个，新增层级
同时带入sites、容量与累计调制。因而本诊断收紧原裁决为：**clean exp389这套独立双anchor+
六/二consumer+pose-loss求和方案NO-GO，不等于所有consumer-balanced或loss-budget变体永久
NO-GO；但mean loss只应作为单变量敏感性诊断，不能预设为天然更公平。**

- 脚本：`frozen_level_ablation.py`，SHA256=
  `71fea0c3781eff2449c6c7cd3460a2fb6e80ebe9a416a3e7ee1f20db9b6d0cb8`；
- JSON SHA256=`dcb65d23fbbfb54780b1fd522439348864e2389fb4e319919f62b8fc2fb05b7a`；
- 状态=`EXP389_FROZEN_LEVEL_ABLATION_PASS`，运行后GPU=`2 MiB/0%`。
