# exp391 监控：官方 clean 全阶段独立直预测 TAPF

## 当前状态

- 状态：`PHASE-A H2-M FORMAL RUNNING`；全部preflight与fresh启动边界已PASS，e1自然完成并进入e2；
- H2-M唯一变量：在保持exp389两个direct anchor、early/late=`6/2` consumer和全部recipe不变时，
  将总pose objective从`0.1×sum(L_early,L_late)`改为`0.1×mean(L_early,L_late)`；
- 正式output=`log/occluded_duke/exp391_clean_swin_tiny_h2m_s1234`；必须自然跑满e120，不挑best；
- Phase B/C继续保持`NO-START`，只有前一阶段按design.md通过才允许实现或训练。

## 不变量

- official clean runtime、teacher、exp386 train-only artifact、batch64、seed1234、120 epoch、SGD、
  lr0.0008、semantic weight0.2、增强/sampler/eval10/checkpoint120固定；
- 不并行、不续训、不重复、不挑best、不按中间性能停止；不修改运行中代码/config；
- query/gallery严格RGB-only；correct/shuffle/None/exploding external pose必须exact；
- 每个阶段先完成state/init/RNG/optimizer、route/gradient、AMP overflow、strict load、consumer path与
  参数/效率门禁，再决定是否fresh正式启动；
- 保护用户工作树，只显式暂存目标文件，禁止`git add -A`。

## Phase A 本地实现边界

- 默认新增`MODEL.TAPF.POSE_LOSS_REDUCTION='sum'`，因此既有D0、HT0和config-off路径保持默认语义；
  只有H2-M config显式设为`mean`；
- `CleanTapfHt0`只在合并early/late pose loss时执行`sum`或乘`0.5`，两个单独pose loss、state keys、
  anchor/PSG构造顺序、forward route和optimizer参数均不改变；非法reduction严格抛错；
- H2-M config相对exp389 formal HT0 config的文本diff只有
  `POSE_LOSS_REDUCTION: mean`与独立`OUTPUT_DIR`；
- 本地`uv`环境的四个修改Python文件`py_compile` PASS；该环境未安装`cv2/torchvision`，因此首次
  unittest在导入阶段退出、未执行任何case。它不计作单元测试结果；必须在远端canonical
  `mmpose-abu`环境完整重跑后才可继续门禁。

## Phase A canonical单元门禁

- 从已验证exp390 full-history bundle建立fresh preflight repo；七个目标文件的本地/远端SHA256
  逐文件exact；
- 远端canonical `mmpose-abu`环境、显式formal repo `PYTHONPATH`下：clean TAPF unit=`7/7 PASS`，
  clean pose data unit=`5/5 PASS`；新增case确认sum/mean两模型state逐项exact、mean总loss严格为
  sum的`0.5×`、early/late单项loss不变，非法reduction严格异常；
- 最初两种未显式设置repo import path的调用均在test discovery/import阶段退出，未执行case、未
  构造模型或占用GPU；它们不计作测试结果，也未修改任何代码/config。后续canonical门禁统一显式
  使用fresh formal repo作为`PYTHONPATH`。

## Phase A 既有路径精确等价

- config-off相对pre-TAPF官方路径的10-step CUDA/AMP完整JSON逐字节exact，JSON SHA256=
  `44033069cd094961f5c3082864d66b47d5130dc317808a7bf09152a15f5c3467`；
- `HIERARCHICAL=False`的D0相对exp387正式代码10-step CUDA/AMP完整JSON逐字节exact，JSON
  SHA256=`b2f19d3f97e4d2d6c4d60241364876be0562605680058031779897d3e4499d16`；
- exp389 legacy HT0显式`sum`与新默认`sum`使用同一输入、RNG、loss和默认GradScaler的10-step完整
  JSON逐字节exact，JSON SHA256=
  `4b2dc99db94d74a519ae83c040627e35daf1b4a832f76cfdedead77cae1eeb0a`；
- 因此默认新增项没有改变B0、D0或legacy HT0；H2-M只由config显式选择`mean`。

## Phase A state、loss、语义与consumer门禁

- H2-M invariant JSON=`/home/afr/reid-clean/audits/exp391/h2m_invariants.json`，SHA256=
  `3bb9fda1d8fb5f73b7589b7d5c322c5ead8f49aa7f21d20d3b988fa6e9f2e682`；
- sum/mean两臂state/init、CPU/CUDA RNG、optimizer成员/顺序/超参数逐项exact；initial state=`243`
  tensors，optimizer parameter groups=`211`；config差异仅为reduction与独立output；
- e1/e6/e10/e11的early/late student route完整；mean combined pose loss严格为sum的`0.5×`，两个
  anchor共`16`个gradient tensors严格为`0.5×`，其余`189`个非anchor gradient tensors逐元素
  exact；strict load=`243` tensors；
- correct/shuffle/None/exploding四种external pose下descriptor、两层student field和八个gate全部
  exact，证明测试路径不读取external pose；
- 人工建立非零gate后逐一旁路consumer，early0–5的descriptor max delta=
  `0.043998/0.037091/0.026748/0.020784/0.020161/0.033009`，late0–1=
  `0.024820/0.039862`；八条路径均有限非零，无terminal dead consumer。

## Phase A gradient ownership与overflow

- ownership/overflow JSON=`/home/afr/reid-clean/audits/exp391/h2m_overflow.json`，SHA256=
  `8f9fd956b771faf09cb3faac227e2009d4fe4507dc89b9b3a31ac6c17ac89f3a`；
- early/late pose loss分别只进入对应anchor各`8/8` tensors；ReID loss不进入两个anchor，且真实进入
  Swin、early/late PSG与head；
- 人工nonfinite后found_inf=`1`，model state=`243`与optimizer state=`205`整步逐张量exact skip，
  GradScaler=`65536→32768`；没有用部分更新或清空state掩盖overflow。

## Phase A 真实paired batch64 CUDA/AMP

- 审计脚本SHA256=`480ed11a0021bbe301bba6c2dc186d8c6cd6cae310975bcdd0cae798ce4816b0`；
  JSON=`/home/afr/reid-clean/audits/exp391/h2m_real_batch64_cuda.json`，SHA256=
  `d64b69bc33a6710dcc3fe24812d236eda64ab11bf5d2af667786c23e670eb00c`；
- formal cosine scheduler初始LR=`8.000000000000001e-6`（与`8e-6`在`1e-15`绝对容差内），真实
  batch64/8-worker连续24 step PASS：finite update=`18`、recoverable overflow=`6`、最长连续finite
  update=`15`、GradScaler=`65536→1024`；
- 参数更新覆盖early anchor=`8/8`、late anchor=`8/8`、early PSG=`12/12`、late PSG=`4/4`、
  Swin=`171/193`、head=`1/3`，optimizer=`205` state tensors且nonfinite=`0`；全部forward、model
  parameter与optimizer state finite；
- peak allocated/reserved=`7,031,979,008/7,342,129,152 B`，24-step mean/median=
  `155.429/124.843 ms`；这些时间包含默认GradScaler恢复过程，不代替下方matched steady-state效率；
- 首次脚本在任何训练step前因`8e-6`与其浮点表示做字面量相等比较而退出；修正为`1e-15`严格容差后
  重跑通过。该退出没有构造formal output、checkpoint或训练轨迹，不计作正式运行或训练异常。

## Phase A matched效率

- 同进程、同一真实paired train batch与RGB-only eval batch的D0/H2-M审计PASS；脚本SHA256=
  `73c712e81e5a354ade15d503a9830ec8eaf792bed7592874ece6bb8d3846e0db`，JSON SHA256=
  `79e4d3ac9e249f8f6d352e86c111eee88a388328028bb2bfb500c8bde80c9a7f`；
- D0/H2-M参数=`28,179,484/28,287,102`，增量=`107,618 / +0.381902%`；analyzer支持算子
  FLOPs=`5,548,787,520/5,588,139,072`，增量=`39,351,552 / +0.709192%`，与exp389同结构
  HT0逐项复现；未支持的elementwise/normalization算子不包装为完整理论FLOPs；
- train batch64 mean latency=`102.851→108.332 ms`，peak allocated=
  `6,188,034,048→6,652,672,000 B`；eval batch256 RGB-only mean latency=
  `228.796→244.041 ms`，peak allocated=`4,725,158,400→4,724,674,048 B`；
- reserved memory受allocator缓存顺序影响，尤其eval出现负差，不作为容量收益解释；报告matched
  allocated与latency原始值，不把测量噪声包装成方法优势。

## Phase A preflight最终裁决

- unit、config-off/D0-off/legacy-sum exact、state/init/RNG/optimizer、route/loss/gradient、overflow、
  strict state、pose-free、八consumer路径、真实paired CUDA/AMP与matched效率门禁均PASS；
- 最终审计文件已逐SHA同步并显式提交到execution repo，fresh formal repo的HEAD/config/bundle/
  teacher/exp386 manifest、tracked source clean、正式output/runner不存在及GPU空闲全部PASS；Phase A
  preflight至此关闭，允许且只允许下述首次正式启动。

## Phase A H2-M正式启动

### 2026-07-18 08:46 UTC：fresh execution与首次健康检查

- fresh formal repo=`/home/afr/SOLIDER-REID-exp391-h2m-0b976f5`，exact detached HEAD=
  `0b976f53496ae4b0fa195f2f30ddc91539c0c8e9`；
- full-history bundle=`/home/afr/reid-clean/bundles/exp391_h2m_0b976f5.bundle`，SHA256=
  `590449a1ee57f640df3f5d96152b5d92e950af9271265e5a4a9a53602c293145`，bundle verify为完整历史且
  HEAD exact；
- config=`configs/occluded_duke/swin_tiny_tapf_h2m.yml`，SHA256=
  `3d172a70b1ea926ed040dc467a6e14d67a27b4cf5c8f0aa4d10674f0e83bc21a`；official teacher SHA256=
  `8bf35b39e6042929383782e0190884ef69fa68abae8437c78c885ade584b404b`；exp386 manifest SHA256=
  `cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`；
- output=`log/occluded_duke/exp391_clean_swin_tiny_h2m_s1234`，runner=
  `/home/afr/train-logs/exp391_clean_h2m_s1234.runner.log`；启动前两者不存在，tracked/staged source
  clean、GPU无计算进程；
- 首次且唯一main PID=`1293570`，python=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；stdout
  明确记录`HIERARCHICAL=True`、`POSE_LOSS_REDUCTION=mean`、official teacher
  `All keys matched successfully`；
- e1自然完成并进入e2：e1 time=`25.705 s`、speed=`522.8 samples/s`；唯一main+8 workers，GPU=
  `7512 MiB`，e1的PoseEarly/PoseLate、ReliabilityEarly/Late、GateEarly/Late全部finite，
  StudentEarly/StudentLate=`0/0`符合teacher-only route；严格异常、AMP warning与checkpoint均为`0`。

这是H2-M的首次且唯一正式启动。必须自然跑满e120，不续训、不重复、不按任何中间epoch/best/阈值
停止，不修改运行中代码/config；Phase B/C继续`NO-START`。

## e10完整评测轨迹

- H2-M mAP / R1 / R5 / R10=`34.7 / 43.9 / 60.0 / 65.9`；
- 同epoch exp387 D0=`33.4 / 42.7 / 59.8 / 65.2`，H2-M−D0=
  `+1.3 / +1.2 / +0.2 / +0.7`；
- 同epoch exp389 HT0=`34.2 / 44.4 / 59.7 / 65.8`，H2-M−HT0=
  `+0.5 / −0.5 / +0.3 / +0.1`；
- e10末尾PoseEarly/PoseLate=`0.856/0.850`、StudentEarly/StudentLate=`1/1`、
  ReliabilityEarly/ReliabilityLate=`0.852/0.852`、GateEarlyAbs/GateLateAbs=
  `3.938e-03/6.378e-03`，全部finite；
- 完整query/gallery评测后自然进入e11；exact HEAD/config/tracked source clean，唯一main+8 workers，
  GPU约`7764 MiB`，严格异常与AMP warning=`0`，e120前checkpoint=`0`。

e10相对两个参考的正负混合只记录为轨迹，不作Phase A裁决，不用于早停或挑best；继续自然运行至e120。
