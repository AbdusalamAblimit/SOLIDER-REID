# exp390 监控：官方干净 TAPF matched 多 seed

## 当前状态

- 状态：`RUNNING`；当前且唯一 arm=`B0-s4321`；
- GPU：main PID=`1133345`，约 `6,846 MiB`；
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

- B0-s4321 SHA256=`8fd054b528608b524212170962f30274b3185c3ee22304720f305f81816a9cfa`；
- D0-s4321 SHA256=`979c897da79327bd8ecc04fcc4b370f0f5ad6b318170fe3afec5594a5c769711`；
- B0-s2025 SHA256=`30c2000dff8e9fa1d554a2873cf16c98a5d8e7d62182c2f95501e4fb8be20a33`；
- D0-s2025 SHA256=`56b2a30fd1856d2dc1077df013cc5d4bab9312be5488f25e1fe7dfa882263116`；
- 四个 config 相对各自 seed1234 canonical 文件的文本 diff 均严格只有 `SOLVER.SEED` 与
  `OUTPUT_DIR`；dataset、teacher、pose artifact、batch、epoch、optimizer、LR、增强、sampler、
  eval/checkpoint 周期均未改变；
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
