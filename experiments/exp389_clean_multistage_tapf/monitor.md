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
