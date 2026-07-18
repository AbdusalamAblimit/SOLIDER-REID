# exp390 监控：官方干净 TAPF matched 多 seed

## 当前状态

- 状态：`RUNNING`；当前且唯一有效 arm=`B0-s4321-valid`；
- GPU：main PID=`1149542`，约 `6,846 MiB`；
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
