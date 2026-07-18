# exp390 监控：官方干净 TAPF matched 多 seed

## 当前状态

- 状态：`RUNNING`；B0/D0-s4321 与 B0-s2025 均已封板，当前且唯一 arm=`D0-s2025`；
- GPU：D0-s2025 main PID=`1254315`，约 `7,098 MiB`；已完成 e100 并自然进入 e101；
- 已有 seed1234 pair：B0=`57.4/67.4/80.6/85.2`，D0=`57.6/67.7/80.8/84.6`，
  D0−B0=`+0.2/+0.3/+0.2/−0.6`；
- 预注册新增顺序：B0-s4321→D0-s4321→B0-s2025→D0-s2025；
- 任一新增 arm 启动前必须完成 design 中的 config、继承代码门禁与 fresh execution 检查。

## 不变量

- 禁止复用旧 runtime、旧 pose data/cache/path mapping；
- B0 只读原始 RGB；D0 只读 exp386 fresh train-only artifact；query/gallery 始终 RGB-only；
- batch64/120 epoch/SGD/lr0.0008/semantic weight0.2/增强/sampler/eval10/checkpoint120 固定；
- 不并行、不续训、不重复、不挑 best、不按中间性能提前停止；
- 每次只提交本实验目标文件，保护用户工作树，禁止 `git add -A`。

## Config 静态门禁

- B0-s4321 SHA256=`5cbf30f2129c4b55a9677f5025d96c2bddb75dffffd7d2d9ff3802097fc282ab`；
- D0-s4321 SHA256=`979c897da79327bd8ecc04fcc4b370f0f5ad6b318170fe3afec5594a5c769711`；
- B0-s2025 SHA256=`53a7c895c39174fc288655bbb35206597c6d681c2f9bc89adb1a283e82521605`；
- D0-s2025 SHA256=`56b2a30fd1856d2dc1077df013cc5d4bab9312be5488f25e1fe7dfa882263116`；
- D0 config 相对 seed1234 canonical 的文本 diff 严格只有 `SOLVER.SEED` 与 `OUTPUT_DIR`；B0
  config 将 exp385 正式命令中的固定 teacher choice/path 收进自包含 YAML，归一化这两个既有
  official 覆盖后也只改 seed/output。dataset、teacher、pose artifact、batch、epoch、optimizer、
  LR、增强、sampler、eval/checkpoint 周期均未改变；
- 四个 output 名称唯一且互不重叠。静态门禁完成时均未创建，并保持 `NO-START`，等待远端继承
  门禁与 fresh execution 审计；其后的状态更新见下文。

## 远端继承与 exact 门禁

- SSH 转发恢复后发现此前的 1.85 GiB 本地 bundle 只传入 `84,864,000` bytes，远端 SHA 不符，
  且失败的 preflight 仓库为空；两者均移入独立 quarantine，未作为任何门禁或执行来源；
- 以已封板 exp389 的 clean full-history bundle 为只读基底重建 fresh preflight repo，只显式加入
  exp390 四个 config 与 design/monitor；preflight HEAD=`de604a643b9aaa0c0d885d42a41142c7065670b0`；
- 新 full-history execution bundle=`/home/afr/reid-clean/bundles/exp390_multiseed_de604a6.bundle`，
  SHA256=`beb37224bdc7d6cb4dd7880468afab03fc9b563f1f1f405d6f144eb9aee9cd1f`，bundle HEAD 与 repo
  exact；
- clean TAPF unit=`6/6 PASS`，clean pose data unit=`5/5 PASS`；
- seed4321 B0 相对 pre-TAPF clean `d4fa227` 的同 seed/teacher/SGD/10-step CUDA-AMP 完整 JSON
  逐字节 exact，双侧 JSON SHA256 均为
  `44033069cd094961f5c3082864d66b47d5130dc317808a7bf09152a15f5c3467`；
- seed4321 `HIERARCHICAL=False` D0 相对 exp387 execution `0d1822a` 的 10-step CUDA-AMP
  完整 JSON 逐字节 exact，双侧 SHA256 均为
  `124c58de1142752f1e6f46973bba50bb6c6d70efadf030a688c23e89ae2e2fa8`；
- B0/D0 公共 state=`211`、公共 optimizer parameter=`179`，state/RNG/optimizer exact；D0 新增
  `105,442` 参数、overhead=`0.375585%`，12 个 TAPF parameter tensor 全部且只出现一次于
  optimizer。

## seed4321 真实训练稳定性门禁

旧 `preflight_cuda.py` 本身未创建 cosine scheduler，因而用满额 `8e-4` 而非正式 warmup 初始化
的 `8e-6` 做 stress。seed4321 在这个高 100 倍 LR 的非正式条件下，step9/10/12 出现可恢复
forward nonfinite，GradScaler 最终降至 `128`，随后连续 12 步有限更新；该结果只作为 stress 诊断，
不得冒充正式 recipe，也没有用于放宽正式训练异常标准。

随后在完全相同代码、数据、seed、batch64/8-worker 下显式创建正式 cosine scheduler，再跑 24 step：

- 结果=`EXP387_REAL_BATCH64_CUDA_AMP_PASS`；正式初始 LR=`8e-6`；
- GradScaler=`65536→512`，overflow=`7`，有限更新=`17`，最长连续有限更新=`13`；
- 全 24 step forward loss/feature/pose/gate finite；overflow step 的参数 probe 不变；
- 最终 Swin/anchor/PSG/head 均有有限更新，全部 model parameter 与 optimizer state finite；
- peak allocated/reserved=`6,484,122,624/6,796,869,632` bytes；
- JSON SHA256=`b1f04dbec9a6da66ebb3ef681436c68d4db66bac43a6bfb730ade6e1f0ea39db`；
- gate script SHA256=`98f60a80740ecbcb9407566fcde2495906e620bf8ff68580dc1827af29bb5059`。

B0 另以正式 scheduler/warmup、真实 batch64/8-worker 跑完整 e1=`227` step smoke，无 eval、无
checkpoint、输出仅位于 audit 目录：所有周期日志 loss finite，严格异常 `0`，train log SHA256=
`9a924954cd22d3398c6f219fa006f835b27d6aecbec990e4ff5b0b6190bcf666`。

## D0 full semantics

- e1/e6/e10/e11 teacher/handoff/student route 与两个独立 PSG consumer 调用 exact；
- pose loss 只到 anchor，ReID loss 只到 Swin/PSG/head；
- 人为 overflow：`208` model parameter 与 `185` optimizer state 整步 exact skip；
- strict state=`223` tensors，missing/unexpected=`0/0`；
- correct/shuffle/None/exploding pose 的 descriptor/student field/two gates 全部 exact，测试期不读取
  pose；
- 语义 JSON SHA256=`2577d337a98e13e16acb2bb452bab035179edcb6db2ee19c36960daf446dedf9`，
  strict roundtrip checkpoint SHA256=
  `26ed693405e7e7f9118d2059b0966643167be26772697d405aca0c365ebcba46`。

截至本记录：preflight repo tracked source clean，GPU=`2 MiB/0%`，无训练/preflight 进程；四个正式
output 与首臂 runner 均不存在。门禁由 `NO-START` 更新为 `GO`，下一步只能从上述 exact bundle
建立 fresh formal repo，并按预注册顺序首次启动 B0-s4321。

## B0-s4321 正式启动

- fresh formal repo=`/home/afr/SOLIDER-REID-exp390-b0s4321-de604a6`；
- exact detached HEAD=`de604a643b9aaa0c0d885d42a41142c7065670b0`；
- bundle SHA256=`beb37224bdc7d6cb4dd7880468afab03fc9b563f1f1f405d6f144eb9aee9cd1f`；
- config SHA256=`8fd054b528608b524212170962f30274b3185c3ee22304720f305f81816a9cfa`；
- output=`log/occluded_duke/exp390_clean_swin_tiny_b0_s4321`；
- runner=`/home/afr/train-logs/exp390_clean_b0_s4321.runner.log`；
- main PID=`1133345`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- 启动前 exact HEAD/config/bundle/teacher SHA、tracked source clean、output/runner 不存在、无其它训练、
  GPU=`2 MiB/0%` 全部 PASS；这是该 arm 的首次且唯一正式启动。

首次健康检查：e1/e2 自然完成并进入 e3；唯一 main+8 workers，GPU 约 `6,846 MiB`，训练 loss
持续 finite，runner 严格异常 `0`，e120 前无 checkpoint。必须继续自然跑满 e120，不因任何中间
eval 或阈值停止。

## B0-s4321 首次启动失效：遗漏官方 teacher 启动覆盖

e10/e20/e30/e40 现场结果分别为：

- e10=`0.6/0.9/2.4/3.7`；
- e20=`1.4/2.3/5.9/8.6`；
- e30=`3.3/5.9/13.0/16.2`；
- e40=`5.0/8.3/16.7/20.8`。

这些数值只触发配置审计，不用于性能早停。审计确认：exp385 已封板 official B0 的 YAML
SHA256 虽同为 `90d715...`，但正式启动通过 CLI 额外固定
`MODEL.PRETRAIN_CHOICE=self` 与
`MODEL.PRETRAIN_PATH=/home/afr/reid-clean/weights/solider_swin_tiny_tea.pth`；其正式 train log 明确
记录这两个有效值并 exact 加载 teacher。exp390 新 B0 config 只复制了 YAML、没有复制这两个启动
覆盖，当前运行日志因此为 `PRETRAIN_CHOICE=imagenet`、`PRETRAIN_PATH=''`，也没有 teacher exact
load；这与 design 中“同一 official Swin-T teacher”直接冲突。

先前 B0 config-off parity/model-invariant 脚本会在脚本内部强制 teacher，因此掩盖了正式 config
自包含性缺口；不带 eval 的 e1 smoke 只验证 finite，无法发现表征起点错误。这两点均须在重启前
补强。当前 PID=`1133345` 的执行被判定为变量边界无效，不属于按指标/阈值早停；必须只终止该
主 PID、确认 workers/GPU 退出，并将 output/runner 移入独立 invalid quarantine。无效轨迹不得写入
结果表、不得计作 B0-s4321、不得从其续训。

修复边界：只在 exp390 的两个 B0 config 中显式加入 exp385 正式启动使用的 teacher choice/path，
其余 recipe 不变；更新 design/config SHA，重建 exact execution bundle。随后必须重跑自包含 config
审计、B0/D0 common state/RNG/optimizer 与带 e1 eval/checkpoint 的 B0 smoke，确认 teacher exact load
及指标回到 official-clean 合理量级，全部 PASS 后才能以新 output/runner fresh 首次有效启动
B0-s4321。

无效 main PID=`1133345` 已单独终止；8 个 workers 自然退出，GPU 回到 `2 MiB/0%`，无 checkpoint。
终止时进度为 e50 iter80。无效 runner/train SHA256 分别为
`03145bbbc8997be57a9f24f97993bb16ce7a2c3dabea5ba84d17e51cf55dbd18` 与
`82460328e213c0ec250d2bba862196f9e735b27e09dd2cb9ab50374df6f5ffea`；output/runner 已原样移入
`/home/afr/reid-clean/quarantine/exp390_b0_s4321_invalid_no_teacher_de604a6/`。原 formal output 与
runner 路径重新为空，但必须使用新 execution commit/bundle 与新的唯一命名，不能覆盖或续用旧
路径。

## B0 teacher 修复后的重新门禁

- 新 remote execution HEAD=`6fc558f44ecdd4cc4bd2352349855dbc6b3288eb`；
- 新 full-history bundle=`/home/afr/reid-clean/bundles/exp390_multiseed_6fc558f.bundle`，SHA256=
  `3295f49ee21fabaa528d1f23b556dc67569c0cf7ff4c220d1f02d3389e85b8ec`，bundle HEAD exact；
- 相对失效 execution `de604a6`，运行边界只改变两个 B0 config 中的 official teacher choice/path；
  D0 config、模型、数据、loss、processor、optimizer、增强与测试路径均未改变；
- 修正并为有效重启分配独立 output 后，B0-s4321/B0-s2025 config SHA256 分别为
  `5cbf30f2129c4b55a9677f5025d96c2bddb75dffffd7d2d9ff3802097fc282ab` 与
  `53a7c895c39174fc288655bbb35206597c6d681c2f9bc89adb1a283e82521605`；
- B0-s4321 相对 pre-TAPF `d4fa227` 的同 seed 10-step CUDA/AMP JSON 再次逐字节 exact，双侧
  SHA256=`44033069cd094961f5c3082864d66b47d5130dc317808a7bf09152a15f5c3467`；
- B0/D0 common state=`211`、state SHA=`c2ccfdd2...`、构造 RNG、179 个公共 optimizer parameter
  与超参数再次 exact，12 个 TAPF parameter 全部纳入 optimizer；
- unit、D0 parity/paired CUDA/full semantics 可由代码与 D0 config 零变化严格继承；未以失效 B0
  运行的任何 state、log 或数值作为门禁输入。

补强后的 B0-s4321 e1 全链路 smoke 使用自包含 config，stdout 明确记录
`PRETRAIN_CHOICE=self`、official teacher path 与 `All keys matched successfully`。完整 227-step
train + e1 eval + checkpoint 自然结束：

- e1 mAP/R1/R5/R10=`10.0/17.4/28.5/33.7`，与 exp385 seed1234 smoke
  `9.2/15.1/25.7/31.1` 同量级，排除无 teacher 时的近随机轨迹；
- train log 严格异常=`0`，SHA256=
  `8cfbd0bed57922d2c87574436520b2c07e0b7ed8f6412b4ba44b80e34c8d76dc`；
- checkpoint=`211` tensors、全 finite，SHA256=
  `6fcfdfae731f697b1bd7f9f9af884426686c0de2060570be22e9f21fc390bcd9`；
- smoke PID/workers 自然退出，GPU=`2 MiB/0%`。

重新门禁全部 PASS。下一步必须从新 bundle 建立全新 formal repo，使用与 quarantine 不同的新
output/runner 名称；启动前再次核对 exact HEAD/config/teacher、tracked source、GPU 与路径不存在。
这才计作 B0-s4321 的首次有效正式启动。

为保证无效 quarantine 与有效运行路径永不重叠，B0-s4321 的自包含 config output 已进一步固定为
`log/occluded_duke/exp390_clean_swin_tiny_b0_s4321_valid`；此处只改 output 名称，不改变上述已通过
smoke 的任何训练/eval 字段。必须以包含该最终 config 的新 execution commit/bundle 启动。

## B0-s4321 首次有效正式启动

- fresh formal repo=`/home/afr/SOLIDER-REID-exp390-b0s4321-valid-ec46d50`；
- exact detached HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`；
- full-history bundle=`/home/afr/reid-clean/bundles/exp390_multiseed_ec46d50.bundle`，SHA256=
  `a1c5329b52a3119fef3d070eb0430f02b21b02fe891d8add16849888db97bfb7`；
- config SHA256=`5cbf30f2129c4b55a9677f5025d96c2bddb75dffffd7d2d9ff3802097fc282ab`；
- output=`log/occluded_duke/exp390_clean_swin_tiny_b0_s4321_valid`；
- runner=`/home/afr/train-logs/exp390_clean_b0_s4321_valid.runner.log`；
- main PID=`1149542`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- 启动前 exact HEAD/config/bundle/teacher SHA、tracked source clean、独立 output/runner 不存在、无其它
  GPU 任务、GPU=`2 MiB/0%` 全部 PASS；
- stdout 明确记录 official teacher `All keys matched successfully`，首次检查 e1 iter160/227，唯一
  main+8 workers，GPU 约 `6,846 MiB`，loss finite，严格异常 `0`，无 checkpoint。

这是 B0-s4321 唯一计入 exp390 的正式运行。必须自然跑满 e120；无效 de604a6 运行继续保持
quarantine，不能与本运行合并、续接或作数值比较。

## B0-s4321-valid 运行监控

### 2026-07-18 03:10 UTC：e10 完整评测

- e1--e10 自然完成，e10 mAP/R1/R5/R10=`36.1/46.8/63.0/69.4`；评测后自然进入 e11，
  现场 e11 iter40/227 loss=`5.400`，finite；
- exact detached HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`，config SHA256=
  `5cbf30f2129c4b55a9677f5025d96c2bddb75dffffd7d2d9ff3802097fc282ab`，tracked source clean；
- 唯一 main PID=`1149542` 与 8 个 DataLoader workers；GPU 只有该 main，约 `6,944 MiB`；
- stdout 中 official teacher 仍为 `All keys matched successfully`；runner/train log 的
  `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/GradScaler/autocast` 严格扫描命中 `0`；
- e120 前无 checkpoint，继续自然训练。该中间评测只记录轨迹，不作早停、best 选择或跨 seed
  结论。

### 2026-07-18 03:23 UTC：e20/e30/e40 完整评测

- e20 mAP/R1/R5/R10=`38.5/48.1/64.6/69.8`；
- e30 mAP/R1/R5/R10=`48.3/59.8/75.1/80.6`；
- e40 mAP/R1/R5/R10=`50.9/62.9/76.7/81.0`；
- e40 评测后自然进入 e41，现场 e41 iter160/227 loss=`0.268`、Acc=`0.986`，finite；
- exact HEAD/config SHA 与上次记录一致，tracked source clean；仍为唯一 main PID=`1149542` 与
  8 workers。完整 eval 时 GPU 短暂约 `13,748 MiB`，返回训练后约 `6,940 MiB`，无第二个 GPU
  进程；
- runner/train log 严格异常与 AMP 警告扫描命中 `0`，仍无 checkpoint；继续自然跑满 e120，
  不按上述中间性能选择或停止。

### 2026-07-18 03:38 UTC：e50/e60/e70 完整评测

- e50 mAP/R1/R5/R10=`50.4/61.7/74.6/80.5`；
- e60 mAP/R1/R5/R10=`53.2/64.0/77.3/82.0`；
- e70 mAP/R1/R5/R10=`55.0/66.6/79.0/83.7`；
- 训练自然进入 e75，现场 e75 iter100/227 loss=`0.108`、Acc=`0.997`，finite；
- exact HEAD/config SHA、tracked clean、唯一 main PID=`1149542`+8 workers 均保持；GPU 只有该
  main，训练现场约 `7,018 MiB`；
- runner/train log 严格异常与 AMP 警告扫描继续命中 `0`，e120 前仍无 checkpoint。所有中间点
  只作轨迹证据，继续完整 e120。

### 2026-07-18 03:54 UTC：e80/e90/e100/e110 完整评测

- e80 mAP/R1/R5/R10=`55.5/66.2/79.4/83.5`；
- e90 mAP/R1/R5/R10=`55.3/65.4/78.5/83.3`；
- e100 mAP/R1/R5/R10=`55.9/66.1/79.5/83.8`；
- e110 mAP/R1/R5/R10=`56.0/66.2/79.5/83.8`；
- e110 评测后自然进入 e111，现场 e111 iter120/227 loss=`0.080`、Acc=`0.997`，finite；
- exact HEAD/config、tracked clean、唯一 main PID=`1149542`+8 workers 继续 PASS，训练 GPU 约
  `6,950 MiB`；
- runner/train log 严格异常与 AMP 警告扫描命中 `0`，e120 前仍无 checkpoint。下一检查必须读取
  final e120，并完成 PID/workers、GPU、唯一 checkpoint、SHA 与 strict finite 终审。

### 2026-07-18 03:58 UTC：B0-s4321-valid e120 封板

- final mAP/R1/R5/R10=`56.0/66.2/79.4/83.8`；这是自然 e120 final，不是 best；
- main PID=`1149542` 与全部 workers 自然退出，GPU=`2 MiB/0%`，无其它训练、pose 或 preflight
  进程；
- output 中唯一 checkpoint=`transformer_120.pth`，size=`112,619,971` bytes；
- runner SHA256=`44612bc74f88f926db65040222fdfa75afdfc523429a38b020134d50dcc98023`；
- train log SHA256=`2033287441ea75ee4937868f9fcaecc63d57769f971c0dd1bd4442f6c951b186`；
- checkpoint SHA256=`b86a337679f44f6da790b952bfbe2aa895f495861b18a525052620650235dae2`；
- exact HEAD/config/tracked clean 终审 PASS；checkpoint 共 `211` tensors，全量 finite，构造 official
  B0 后 `strict=True` load 的 missing/unexpected=`0/0`，model state 仍全量 finite；
- runner/train log 的严格异常与 AMP 警告终审命中 `0`。B0-s4321-valid 正式封板，禁止重启、
  续训或重复；下一步只能 fresh 启动预注册的 D0-s4321。

## D0-s4321 正式启动

- fresh formal repo=`/home/afr/SOLIDER-REID-exp390-d0s4321-ec46d50`；
- exact detached HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`；
- full-history bundle=`/home/afr/reid-clean/bundles/exp390_multiseed_ec46d50.bundle`，SHA256=
  `a1c5329b52a3119fef3d070eb0430f02b21b02fe891d8add16849888db97bfb7`；
- config=`configs/occluded_duke/swin_tiny_tapf_d0_s4321.yml`，SHA256=
  `979c897da79327bd8ecc04fcc4b370f0f5ad6b318170fe3afec5594a5c769711`；
- output=`log/occluded_duke/exp390_clean_swin_tiny_d0_s4321`；runner=
  `/home/afr/train-logs/exp390_clean_d0_s4321.runner.log`；
- main PID=`1183898`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- 启动前 repo/output/runner fresh，exact HEAD/config/bundle、official teacher SHA256=
  `8bf35b39e6042929383782e0190884ef69fa68abae8437c78c885ade584b404b`、exp386 manifest SHA256=
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`、tracked clean、GPU=
  `2 MiB/0%`、无其它任务全部 PASS；
- stdout 显示 `HIERARCHICAL=False`、official teacher `All keys matched successfully`。e1 自然完成并
  进入 e2；唯一 main+8 workers，GPU 约 `6,994 MiB`，Loss/Pose/Reliability/GateAbs 均 finite，
  Student=`0` 符合 e1--5 teacher route；严格异常与 AMP 警告扫描命中 `0`，无 checkpoint。

这是 D0-s4321 的首次且唯一正式启动。必须自然跑满 e120，query/gallery 继续严格 RGB-only；不得
从 B0 或任何 D0 checkpoint 续训，也不得按中间性能停止。

## D0-s4321 运行监控

### 2026-07-18 04:08 UTC：e10 完整评测

- D0 e10 mAP/R1/R5/R10=`37.9/48.6/64.2/70.5`；同 epoch B0-s4321=
  `36.1/46.8/63.0/69.4`，D0−B0=`+1.8/+1.8/+1.2/+1.1`；
- 评测后自然进入 e15，现场 e15 iter160/227 Loss=`3.543`、Pose=`0.650`、Acc=`0.535`、
  Student=`1.00`、Reliability=`0.844`、GateAbs=`1.205e-02`，全部 finite；
- exact HEAD/config、tracked clean、唯一 main PID=`1183898`+8 workers 继续 PASS；GPU 只有该
  main，约 `7,070 MiB`；
- runner/train log 严格异常与 AMP 警告扫描命中 `0`，e120 前无 checkpoint。e10 差值只作同
  epoch 轨迹，不用于早停、best 选择或最终 paired 判断。

### 2026-07-18 04:24 UTC：e20/e30/e40/e50 完整评测

| epoch | D0 mAP/R1/R5/R10 | B0-s4321 同 epoch | D0−B0 |
|---:|---:|---:|---:|
| 20 | `40.3/50.3/64.3/69.7` | `38.5/48.1/64.6/69.8` | `+1.8/+2.2/−0.3/−0.1` |
| 30 | `47.7/58.7/73.9/79.5` | `48.3/59.8/75.1/80.6` | `−0.6/−1.1/−1.2/−1.1` |
| 40 | `50.3/61.2/75.1/80.4` | `50.9/62.9/76.7/81.0` | `−0.6/−1.7/−1.6/−0.6` |
| 50 | `51.8/61.6/75.8/81.2` | `50.4/61.7/74.6/80.5` | `+1.4/−0.1/+1.2/+0.7` |

- e50 评测后自然进入 e51，现场 e51 iter20/227 Loss=`0.232`、Pose=`0.463`、Acc=`0.994`、
  Student=`1.00`、Reliability=`0.848`、GateAbs=`2.298e-02`，全部 finite；
- exact HEAD/config、tracked clean、唯一 main PID=`1183898`+8 workers 保持；完整 eval 时 GPU
  短暂约 `13,798 MiB`，回到训练后约 `7,064 MiB`，无第二个 GPU 进程；
- 严格异常与 AMP 警告扫描命中 `0`，仍无 checkpoint。正负中期差值均不改变完整 e120 计划。

### 2026-07-18 04:39 UTC：e60/e70/e80 完整评测

| epoch | D0 mAP/R1/R5/R10 | B0-s4321 同 epoch | D0−B0 |
|---:|---:|---:|---:|
| 60 | `54.5/64.7/78.3/83.4` | `53.2/64.0/77.3/82.0` | `+1.3/+0.7/+1.0/+1.4` |
| 70 | `55.8/66.0/79.8/83.9` | `55.0/66.6/79.0/83.7` | `+0.8/−0.6/+0.8/+0.2` |
| 80 | `55.8/66.3/79.4/84.0` | `55.5/66.2/79.4/83.5` | `+0.3/+0.1/+0.0/+0.5` |

- 训练自然进入 e84，现场 e84 iter180/227 Loss=`0.130`、Pose=`0.462`、Acc=`0.997`、
  Student=`1.00`、Reliability=`0.836`、GateAbs=`2.343e-02`，全部 finite；
- exact HEAD/config、tracked clean、唯一 main PID=`1183898`+8 workers 继续 PASS；训练 GPU 约
  `7,092 MiB`；
- 严格异常与 AMP 警告扫描命中 `0`，e120 前仍无 checkpoint；继续自然完成剩余 epochs。

### 2026-07-18 04:46 UTC：e90/e100 完整评测

| epoch | D0 mAP/R1/R5/R10 | B0-s4321 同 epoch | D0−B0 |
|---:|---:|---:|---:|
| 90 | `56.8/66.6/80.2/85.0` | `55.3/65.4/78.5/83.3` | `+1.5/+1.2/+1.7/+1.7` |
| 100 | `56.5/66.5/79.8/83.8` | `55.9/66.1/79.5/83.8` | `+0.6/+0.4/+0.3/+0.0` |

- e100 评测后自然进入 e101，现场 e101 iter60/227 Loss=`0.127`、Pose=`0.461`、Acc=`0.999`、
  Student=`1.00`、Reliability=`0.858`、GateAbs=`2.340e-02`，全部 finite；
- exact HEAD/config SHA、tracked source clean、唯一 main PID=`1183898`+8 workers 保持；GPU 只有该
  main，约 `7,058 MiB`；
- runner/train log 未出现 AMP warning、NaN、Inf 数值、Traceback、RuntimeError、OOM、nonfinite 或
  overflow，e120 前仍无 checkpoint。e90/e100 均只作同 epoch 轨迹，不用于早停或挑选 best。

### 2026-07-18 04:55 UTC：e110/e120 与 D0-s4321 封板

| epoch | D0 mAP/R1/R5/R10 | B0-s4321 同 epoch | D0−B0 |
|---:|---:|---:|---:|
| 110 | `56.8/66.5/79.9/84.3` | `56.0/66.2/79.5/83.8` | `+0.8/+0.3/+0.4/+0.5` |
| 120 | `56.8/66.5/79.9/84.3` | `56.0/66.2/79.4/83.8` | `+0.8/+0.3/+0.5/+0.5` |

- e120 自然完成，未挑 best；main PID=`1183898`与8 workers自然退出，GPU=`2 MiB/0%`，无其它
  train/pose/preflight 进程；
- output 中唯一 checkpoint=`transformer_120.pth`，SHA256=
  `c8bd663c8a03022b649e4c38970f5b19017f921c588ac413daed471c11630678`；runner/train log SHA256=
  `c9324c20e50132dfb83d213cbbcd6a8f2b421b86edb4c964d329c2bba0702ccc`/
  `6b2335c0597b5d4ac41e4bee2e1c2ce399c77558408d3dac3f32f2f230e5c2a0`；
- exact detached HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`，config SHA256=
  `979c897da79327bd8ecc04fcc4b370f0f5ad6b318170fe3afec5594a5c769711`，tracked source clean；
- `final_audit_d0.py` 原生终审 `EXP390_D0_FINAL_AUDIT_PASS`：checkpoint `223` state tensors/
  `210` floating tensors 全部 finite，strict missing/unexpected=`0/0`；anchor=`8/8`、两 PSG=
  `4/4` 参数相对 fresh 初始化改变，每个独立 bank=`2/2`，两 bank 不相同；
- correct/shuffle/None/exploding external pose 的 descriptor、`17×24×8` student field和两个
  `48×768` gate逐元素 exact，exploding accesses=`0`；normal-train/validation均为无pose store的
  `ImageDataset`；
- 分别旁路 PSG bank0/bank1，最终 descriptor 最大绝对变化=`1.3385458/1.9894147`，两个 consumer
  均有真实下游路径；final audit JSON SHA256=
  `a426c315977406cc15153ccaadb2a36f59aeacb7cc14912ffca22a3c45d8b32f`；
- runner/train log 的 AMP warning、NaN、Inf 数值、Traceback、RuntimeError、OOM、nonfinite、
  overflow严格边界扫描均命中 `0`。D0-s4321 正式封板，禁止重启、续训或重复。

等待期间另完成 `old_new_implementation_audit.md`，并由只读独立子 agent 交叉复核；它不使用 GPU、
不修改训练代码/config，也不影响本 arm。主结论是旧 exp378 的较大相对增量建立在更弱 B0、不同
RE/loader/TTA protocol与约2倍 PSG consumer容量上，不能解释为旧 D0 绝对更强或更会利用正确姿态。

## B0-s2025 首次正式启动

### 2026-07-18 05:08 UTC：fresh execution 与首次健康检查

- fresh repo=`/home/afr/SOLIDER-REID-exp390-b0s2025-ec46d50`，exact detached HEAD=
  `ec46d50486d645da0872d5549e1071f2a8072b24`；
- full-history bundle=`/home/afr/reid-clean/bundles/exp390_multiseed_ec46d50.bundle`，SHA256=
  `a1c5329b52a3119fef3d070eb0430f02b21b02fe891d8add16849888db97bfb7`，bundle verify完整历史且
  HEAD exact；
- config=`configs/occluded_duke/swin_tiny_s2025.yml`，SHA256=
  `53a7c895c39174fc288655bbb35206597c6d681c2f9bc89adb1a283e82521605`；official teacher SHA256=
  `8bf35b39e6042929383782e0190884ef69fa68abae8437c78c885ade584b404b`；
- output=`log/occluded_duke/exp390_clean_swin_tiny_b0_s2025`，runner=
  `/home/afr/train-logs/exp390_clean_b0_s2025.runner.log`，启动前两者均不存在；exact HEAD/config/
  bundle/teacher、tracked source clean、GPU=`2 MiB/0%`、无其它train/pose/preflight进程全PASS；
- 首次且唯一main PID=`1220240`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；stdout
  明确记录TAPF disabled、official teacher `All keys matched successfully`；
- e1/e2自然完成并进入e3/e4；唯一main+8 workers，GPU约`6,846 MiB`，loss/accuracy finite，严格
  异常与AMP warning为`0`，e120前无checkpoint。

B0-s2025必须自然跑满e120；不得按中间点停止或挑best。终审PASS后才能fresh启动D0-s2025。

### 2026-07-18 05:19 UTC：e10/e20完整评测

- e10 mAP/R1/R5/R10=`34.7/44.6/59.6/65.8`；
- e20 mAP/R1/R5/R10=`38.8/50.0/63.4/69.3`；
- e20评测后自然进入e21，现场e21 iter200/227 loss=`1.543`、Acc=`0.863`，finite；
- exact HEAD/config SHA保持，tracked source clean；唯一main PID=`1220240`与8 workers，GPU只有该
  main，约`6,930 MiB`；output仍只有train log，e120前无checkpoint；
- runner/train log的严格数值异常、Traceback/RuntimeError/OOM与AMP warning边界扫描命中`0`。
  两个中间点只记录seed2025轨迹，不用于早停、挑best或提前判断TAPF跨seed效应。

### 2026-07-18 05:23 UTC：e30完整评测

- e30 mAP/R1/R5/R10=`48.0/58.8/74.0/79.2`；评测后自然进入e35，现场e35 iter20/227
  loss=`0.461`、Acc=`0.977`，finite；
- exact HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`、config SHA256=
  `53a7c895c39174fc288655bbb35206597c6d681c2f9bc89adb1a283e82521605`，tracked source clean；
- 唯一main PID=`1220240`与8 workers，GPU只有该main，约`6,942 MiB`；output仍无checkpoint；
- 严格异常与AMP warning扫描保持`0`。e30只记录轨迹，继续自然跑满e120。

### 2026-07-18 06:01 UTC：e40--e120 与 B0-s2025 封板

| epoch | B0-s2025 mAP/R1/R5/R10 |
|---:|---:|
| 40 | `48.5/59.8/73.4/78.8` |
| 50 | `52.9/63.1/77.8/82.9` |
| 60 | `53.4/64.5/78.8/83.3` |
| 70 | `55.6/66.3/79.8/84.8` |
| 80 | `55.9/66.4/79.1/83.9` |
| 90 | `57.0/67.3/80.5/85.2` |
| 100 | `57.2/67.5/80.8/85.6` |
| 110 | `57.4/67.8/80.8/85.5` |
| 120 | `57.5/67.9/81.1/85.7` |

- e120自然完成，未挑best；main PID=`1220240`和8 workers自然退出，GPU=`2 MiB/0%`，无其它
  python训练或GPU进程；
- output中唯一checkpoint=`transformer_120.pth`，size=`112,619,971` bytes；runner/train/checkpoint
  SHA256分别为`42e983abe8759625b55f2ad6feda056cb6be4e6ed20ffc8a39452abb4a31c8a2`/
  `7736b3d8a91ff14a879e09d9f32e5e33bee30b256a9e55bf3b76501fe390afb0`/
  `97560b039fb1a7adade514c301e8f5fb46aa1d43efbf067729ba6dd6f6881afe`；
- exact HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`、config SHA256=
  `53a7c895c39174fc288655bbb35206597c6d681c2f9bc89adb1a283e82521605`、tracked source clean；
- 原生构造official B0后checkpoint共`211` state tensors/`198` floating tensors，全量finite，
  `strict=True` missing/unexpected=`0/0`，load后model state仍全量finite；normal-train/validation均为
  无pose store的`ImageDataset`；
- runner/train log的AMP warning、NaN、Inf数值、Traceback、RuntimeError、OOM、nonfinite、overflow
  严格终审命中`0`。B0-s2025正式封板，禁止重启、续训或重复。

## D0-s2025 首次正式启动

### 2026-07-18 06:57 UTC：fresh execution 与首次健康检查

- fresh repo=`/home/afr/SOLIDER-REID-exp390-d0s2025-ec46d50`，exact detached HEAD=
  `ec46d50486d645da0872d5549e1071f2a8072b24`；full-history bundle=
  `/home/afr/reid-clean/bundles/exp390_multiseed_ec46d50.bundle`，SHA256=
  `a1c5329b52a3119fef3d070eb0430f02b21b02fe891d8add16849888db97bfb7`，bundle verify为完整历史且
  HEAD exact；
- config=`configs/occluded_duke/swin_tiny_tapf_d0_s2025.yml`，SHA256=
  `56b2a30fd1856d2dc1077df013cc5d4bab9312be5488f25e1fe7dfa882263116`；official teacher SHA256=
  `8bf35b39e6042929383782e0190884ef69fa68abae8437c78c885ade584b404b`；exp386 manifest SHA256=
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；
- output=`log/occluded_duke/exp390_clean_swin_tiny_d0_s2025`，runner=
  `/home/afr/train-logs/exp390_clean_d0_s2025.runner.log`；启动前repo/output/runner fresh，HEAD/config/
  bundle/teacher/manifest exact，tracked source clean，GPU=`2 MiB/0%`且无其它python/GPU任务，全PASS；
- 首次且唯一main PID=`1254315`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；stdout
  明确记录TAPF enabled、`HIERARCHICAL=False`、official teacher `All keys matched successfully`；
- e1自然完成并进入e2；唯一main+8 workers，GPU约`6,994 MiB`，Loss/Pose/Reliability/GateAbs全部
  finite，Student=`0`符合e1--5 teacher route，严格异常与AMP warning扫描命中`0`，无checkpoint。

这是D0-s2025首次且唯一正式启动。必须自然跑满e120；D0只在训练期读取exp386 artifact，
query/gallery保持RGB-only，不得按任何中间点停止或挑best。

### 2026-07-18 07:02 UTC：e10完整评测

- D0 e10 mAP/R1/R5/R10=`35.0/44.7/60.4/66.3`；同epoch B0-s2025=
  `34.7/44.6/59.6/65.8`，D0−B0=`+0.3/+0.1/+0.8/+0.5`；
- e1--5 Student=`0`，e6/e7/e8/e9/e10依次=`0.2/0.4/0.6/0.8/1.0`，handoff路由完整；
  评测后自然进入e11，现场e11 iter200/227 Loss=`4.879`、Pose=`0.783`、Acc=`0.309`、
  Student=`1.00`、Reliability=`0.848`、GateAbs=`7.055e-03`，全部finite；
- exact HEAD/config、tracked source clean、唯一main PID=`1254315`+8 workers保持；GPU只有该main，
  约`6,994 MiB`；runner/train log严格异常与AMP warning扫描命中`0`，无checkpoint；
- e10只作同seed、同epoch轨迹记录，不用于提前裁决、早停或挑best。

### 2026-07-18 07:08 UTC：e20完整评测

- D0 e20 mAP/R1/R5/R10=`41.9/52.6/66.7/72.8`；同epoch B0-s2025=
  `38.8/50.0/63.4/69.3`，D0−B0=`+3.1/+2.6/+3.3/+3.5`；
- 评测后训练自然进入e26，现场e26 iter140/227 Loss=`0.879`、Pose=`0.518`、Acc=`0.939`、
  Student=`1.00`、Reliability=`0.846`、GateAbs=`2.031e-02`，全部finite；
- exact HEAD/config、tracked source clean，唯一main PID=`1254315`+8 workers；GPU只有该main，约
  `7,074 MiB`；严格异常与AMP warning扫描命中`0`，仍无checkpoint；
- e20较大正差只作中间轨迹，不改变必须自然完成e120的预注册计划，也不用于跨seed结论。

### 2026-07-18 07:24 UTC：e30/e40/e50/e60完整评测

| epoch | D0 mAP/R1/R5/R10 | B0-s2025 同epoch | D0−B0 |
|---:|---:|---:|---:|
| 30 | `48.0/58.9/72.6/78.2` | `48.0/58.8/74.0/79.2` | `+0.0/+0.1/−1.4/−1.0` |
| 40 | `51.7/63.0/77.1/81.1` | `48.5/59.8/73.4/78.8` | `+3.2/+3.2/+3.7/+2.3` |
| 50 | `53.6/64.8/78.0/82.6` | `52.9/63.1/77.8/82.9` | `+0.7/+1.7/+0.2/−0.3` |
| 60 | `53.3/64.5/78.2/82.9` | `53.4/64.5/78.8/83.3` | `−0.1/+0.0/−0.6/−0.4` |

- e60评测后自然进入e61，现场e61 iter200/227 Loss=`0.174`、Pose=`0.467`、Acc=`0.994`、
  Student=`1.00`、Reliability=`0.862`、GateAbs=`2.336e-02`，全部finite；
- exact HEAD/config、tracked source clean；唯一main PID=`1254315`+8 workers，GPU只有该main，约
  `7,068 MiB`；严格异常与AMP warning扫描命中`0`，e120前仍无checkpoint；
- e30--e60正负波动均只作同epoch轨迹，不作早停、best选择或跨seed结论。

### 2026-07-18 07:42 UTC：e70/e80/e90/e100完整评测

| epoch | D0 mAP/R1/R5/R10 | B0-s2025 同epoch | D0−B0 |
|---:|---:|---:|---:|
| 70 | `54.6/64.3/77.7/82.1` | `55.6/66.3/79.8/84.8` | `−1.0/−2.0/−2.1/−2.7` |
| 80 | `56.7/66.6/79.8/84.0` | `55.9/66.4/79.1/83.9` | `+0.8/+0.2/+0.7/+0.1` |
| 90 | `57.4/66.9/79.8/85.0` | `57.0/67.3/80.5/85.2` | `+0.4/−0.4/−0.7/−0.2` |
| 100 | `57.5/66.7/80.5/84.8` | `57.2/67.5/80.8/85.6` | `+0.3/−0.8/−0.3/−0.8` |

- e100评测后自然进入e101，现场e101 iter200/227 Loss=`0.113`、Pose=`0.463`、Acc=`0.998`、
  Student=`1.00`、Reliability=`0.854`、GateAbs=`2.339e-02`，全部finite；
- exact HEAD=`ec46d50486d645da0872d5549e1071f2a8072b24`、config SHA256=
  `56b2a30fd1856d2dc1077df013cc5d4bab9312be5488f25e1fe7dfa882263116`、tracked source clean；唯一
  main PID=`1254315`+8 workers，GPU只有该main，约`7,098 MiB`；
- runner/train log的严格异常与AMP warning扫描命中`0`，e120前仍无checkpoint。e70--e100的
  正负波动均只作同epoch轨迹；继续自然跑满e120，不作早停、best选择或提前裁决。
