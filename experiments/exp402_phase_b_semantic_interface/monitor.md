# exp402 Phase-B semantic-interface监控

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / ACTUAL-AUDIT STATIC-CPU SEALED-PASS / CUDA PREFLIGHT NO-START / FORMAL GPU NO-START`

## 2026-07-20 接手与设计冻结

- 上游exp401=`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`，full−all-bypass=
  `+0.1194214838 mAP point`，但R1差为`−0.0904977322 point`；
- exp402只做同checkpoint RGB-only只读语义接口kill-switch，不修改sealed repo/config/checkpoint，不训练；
- 冻结10个串行arm：correct、五类evidence/slot/binding破坏、generic expert mean、router0/1/all bypass；
- wrong RGB使用same-split/same-camera/different-PID的dataset-global absolute-index donor，不允许batch roll；
- scientific GO要求所有六个semantic controls相对correct至少低`0.1 mAP`，并复核route gap与两个consumer；
- 当前远端无GPU任务，正式脚本尚未实现或传输；下一步仅实现CPU/static正反contract。

## initial static CPU run1：SEALED-FAIL

- fresh run1完成38项CPU/static contract，GPU始终未初始化；36项PASS，仅
  `orthogonal_norm=false`、`orthogonal_cosine=false`；
- canonical正交矩阵本身Gram max-abs=`9.9920072216e-16`，已通过`<=1e-12`门；但double
  synthetic evidence在干预实现中被强制降为float32，导致norm max-abs=`6.1642145166e-08`、
  cosine max-abs=`1.2328428656e-07`；
- run1 core/static/result/runner SHA256=
  `8b536ec1e03d6ac17253900462af182cc11f65757fb35ef134b038644e71dfd2`/
  `039f0b9cf3ed45b7416772be0acaafbc6c7ecb8bbbcb8be59d8f7d75e1963778`/
  `41f2e84b5cc2ea2f165417ecb4822442472b4b6f993afb03dad651a7657b16dd`/
  `41f2e84b5cc2ea2f165417ecb4822442472b4b6f993afb03dad651a7657b16dd`；
- 判定=`EXP402_STATIC_CPU_FAIL`。这是通用contract的dtype语义错误，不是正式模型数值失败；正式
  evidence原本为float32。run1永久保留，不覆盖、不补跑；修复仅让右乘矩阵跟随evidence dtype，随后必须
  用fresh run2/run3路径重新完成两遍38/38与byte-exact门，GPU继续`NO-START`。

## dtype修复与core static CPU封板

- 修复只移除`orthogonal_evidence`分支对double synthetic evidence的强制float32降精度，使正交矩阵
  跟随输入evidence dtype；实际模型evidence本来是float32，因此不改变正式推理精度或干预定义；
- fresh run2因登录shell不存在裸`python`命令，未进入contract、未生成result，只留下0-byte runner，
  SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；该执行器选择记录为
  `STATIC_EXECUTOR_INVALID`并永久保留，不覆盖、不复用；
- 改用exp394–exp401冻结链的canonical runtime后，fresh run3/run4均为`38/38 PASS`；每次
  result/runner逐字节一致，两遍result也逐字节一致，统一SHA256=
  `98d2c50ce8abd69ff0123c597d8079779b1d0107d47342d65489ec2a087e897a`；
- 修正版core/static SHA256=
  `6e9ac9cfc03d70606ee34f77af39accc6b66c89ee9974fee48ded1d6951dfb54`/
  `039f0b9cf3ed45b7416772be0acaafbc6c7ecb8bbbcb8be59d8f7d75e1963778`；
- run3/run4 orthogonal Gram/norm/cosine max-abs=
  `9.9920072216e-16/2.2204460493e-16/6.6613381478e-16`；donor global-index与chunk
  invariant、五种state intervention、generic expert mean、三个bypass、正反adjudicator及source AST
  全部PASS；CUDA initialized before/after=`false/false`；
- 退出后GPU=`2 MiB/0%`且无compute process。当前判定=
  `CORE_STATIC_CPU_SEALED_PASS / ACTUAL_AUDIT_IMPLEMENTATION_AND_STATIC_REQUIRED / GPU NO-START`；
  core PASS不授权正式GPU执行，下一步必须实现actual只读脚本，并把该脚本纳入两遍static/AST contract。

## actual audit实现与两遍static/AST封板

- actual脚本只用`OccludedDuke + ImageDataset + val transforms`构造query+gallery loader，不调用训练
  dataloader，不构造pose target；correct全局缓存按absolute index/path绑定，wrong-RGB严格使用冻结的
  same-split/same-camera/different-PID donor map；
- 10个arm严格串行；五个state干预patch `prepare`，generic arm临时求两个router各自五expert均值，三个
  bypass按bank exact identity；每臂均检查两个router call count、prepare/apply-gate patch恢复、模型state
  SHA、global与loader RNG、full index/path覆盖；
- teacher/pose读取guard覆盖正式model构造与eval，拒绝derived pose、safetensors和codebook读取；正式脚本
  无训练、更新、保存checkpoint或旧path mapping路径；correct与all-bypass还必须以`5e-8`绝对误差复现
  exp401 raw reference；
- actual/core/static SHA256=
  `dcde68ecf7f25a6d802bd34c0950524af4834023e73c53d24df13e9c2ca7104d`/
  `6e9ac9cfc03d70606ee34f77af39accc6b66c89ee9974fee48ded1d6951dfb54`/
  `039f0b9cf3ed45b7416772be0acaafbc6c7ecb8bbbcb8be59d8f7d75e1963778`；
- 纳入actual AST后的fresh run5/run6均为`38/38 PASS`，每次result/runner逐字节一致，两遍result也
  逐字节一致，统一SHA256=
  `2c7f19b81b618245e0a1e2d148836e7a3099d076ffc2c7b293e90f3313d15b78`；
- AST确认无backward、parameter update、checkpoint write、train loader/pose target、derived/teacher/codebook
  literal；两遍CUDA initialized before/after=`false/false`，退出后GPU=`2 MiB/0%`且无compute process；
- 当前判定=`ACTUAL_AUDIT_STATIC_CPU_SEALED_PASS / CUDA PREFLIGHT GO / FORMAL GPU NO-START`。
  下一步只允许fresh小批CUDA preflight；其终审全部通过后，才允许唯一一次exp402 formal full执行。

## CUDA preflight run1：SEALED-INVALID

- run1使用32个recipient及其global donor进入只读CUDA preflight；model strict load、RGB-only前置路径均已
  通过，但donor warmup与recipient集合存在少量索引重叠；同一RGB在不同batch形状下重算evidence产生
  非逐元素相等的浮点差，触发`Repeated evidence capture changed`并退出；
- run1未进入10臂完整preflight，更未进入formal full；result/runner SHA256=
  `f63514b5dde9dc5ec84028b8c8c881c5330a987a8558b78229ceaad0d877c55d`/
  `675fcfe7980f649d09916582eb62f0acde0cb1338816d7c890a8d467a57e52bb`；退出后GPU=
  `2 MiB/0%`且无compute process；
- 判定=`CUDA_PREFLIGHT_RUN1_SEALED_INVALID / DUPLICATE_CACHE_REPORTER_TOO_STRICT`。该错误不回答
  semantic-interface科学问题；run1永久保留、不覆盖。修复只让warmup排除recipient集合内donor，这些donor
  改由correct recipient pass一次性缓存；global donor定义、arm、模型、checkpoint与门槛全部不变。修正版
  必须先用fresh source和两遍static/AST重新授权，再执行fresh CUDA preflight run2。
