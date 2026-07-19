# exp402 Phase-B semantic-interface监控

## 当前状态

`SEALED / FORMAL VALIDITY PASS / CURRENT_SEMANTIC_INTERFACE_NO-GO`

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

## 修正版static与CUDA preflight run2：SEALED-PASS

- 修正版actual source SHA256=
  `f450c5e3801a61d20a7ba831d6cefb1ab1d9d5f5549824fc2fa55943059cc479`；fresh static
  run7/run8均为`38/38 PASS`，每次result/runner与跨遍result逐字节一致，统一SHA256=
  `1f556f036dca8064fad1ffabeffd565417c24e72119a565b04e21c0bca8176f6`；CUDA未初始化；
- fresh CUDA preflight run2使用32个recipient与6个recipient集合外donor，status=
  `EXP402_CUDA_PREFLIGHT_PASS`；17项validity全部PASS：strict reload、RGB None/ExplodingPose exact且访问0、
  external teacher/pose/codebook访问0、全局19,871条donor same-split/same-camera/different-PID、10臂完整覆盖、
  每臂state/RNG/loader RNG/prepare/apply-gate恢复、checkpoint/config/source/core/repo终审全部exact；
- 九个破坏臂相对correct的descriptor mean L2全部有限非零：wrong-RGB=`0.0942934`、zero=
  `1.7235049`、orthogonal=`2.1975737`、slot-cycle=`2.2633276`、wrong-binding=`0.9477000`、
  generic-mean=`3.1618040`、bypass0/1/all=`2.0468879/2.2085962/2.8598566`；每臂32行
  exact-equal=`0`，两个router各调用1次，single bypass均有独立影响；
- preflight result/runner SHA256=
  `7bd25af541d55d1bc5619fe5932c27215893b26fec9c11d03c8c16b25b54a52b`/
  `3500ad54d6318c9830e4d706edf563ac2ce544508e0cd1a440dad8a375d955ed`；异常扫描0，
  checkpoint SHA执行前后均=`fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc`；
  进程退出后GPU=`2 MiB/0%`且无compute process；
- 判定=`CUDA_PREFLIGHT_SEALED_PASS / FORMAL_FULL ONE-TIME GO`。这只授权唯一一次full执行，不包含任何
  retrieval科学结果；formal仍必须fresh result/runner/manifest，严格串行，不得同编号补跑。

## formal full唯一执行启动

- postflight manifest脚本SHA256=
  `b1e9e68602768e6cc1708c0d7be8b1454729eb8d310a431b9c94f9e7caf4aa92`；纳入AST后的fresh
  static run9/run10均为`38/38 PASS`且result/runner/跨遍逐字节一致，统一SHA256=
  `ed4f1f3b1c8f30960779311130086e5454f8df88bb834c0dbea90103018207d8`；
- once-only wrapper SHA256=
  `6b8c8cfc77e4a0bd1a3a3969632aba4ab2f533ad49a44f0c6a7adeb104eb1e98`，bash static syntax
  PASS；fresh formal result/runner/manifest/launcher路径启动前均不存在；
- 唯一formal full已后台启动：wrapper PID=`418043`，actual main PID=`418044`；初检main唯一占用GPU，
  显存约`7,926 MiB`、利用率`93%`，无并行GPU任务；
- runner已出现两次完整distance-matrix计算，说明前两个串行arm已完成；目前仅有backbone `pretrained`
  与`addmm_` API deprecation warning，不是AMP或数值异常；NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/
  overflow=`0`；
- 当前状态=`FORMAL FULL RUNNING`。不得修改source/config/checkpoint，不得重跑或提前按中间arm裁决；自然
  结束后由wrapper自动生成post-exit manifest，再按冻结的六semantic-control最大值与route门一次性裁决。

## formal full自然结束与终审封板

- 唯一formal依冻结顺序完成10/10个全validation arm，每臂`78`个batch、`19,871`行、unique index=
  `19,871`；correct缓存evidence `19,871/19,871`且无duplicate；两个router每臂各调用78次，single/all
  bypass覆盖分别为`[78,0]/[0,78]/[78,78]`；
- 20项validity全部PASS：correct与all-bypass四项对exp401 reference absolute error=`0`，所有metrics/
  descriptor finite，九臂descriptor mean L2均非零且exact-equal rows=`0`；逐臂prepare/apply-gate、
  generic weights、model state、global/loader RNG全部恢复，official path/donor全覆盖；external teacher/pose/
  codebook访问0，RGB None/ExplodingPose exact且pose访问0；
- full arm raw mAP/R1/R5/R10：correct=`57.1230075595/67.2850668430/80.2714943886/
  84.7511291504`；wrong-RGB=`57.1296975953/67.2398209572/80.3167402744/84.7963809967`；
  zero=`57.1237039863/67.3303186893/80.0904989243/84.8416268826`；orthogonal=
  `57.0932184448/67.3303186893/80.0452470779/84.8416268826`；slot-cycle=
  `57.0596836822/67.3755645752/80.0452470779/84.5701336861`；wrong-mask=
  `57.1094812717/67.3303186893/80.3619921207/84.7963809967`；generic-mean=
  `56.9989891041/67.0135736465/80.2714943886/84.7511291504`；
- bypass0/1/all raw=
  `56.9922507039/67.4660623074/80.0904989243/84.5248878002`、
  `57.1390635559/67.4208164215/80.4524898529/84.7058832645`、
  `57.0035860757/67.3755645752/80.0452470779/84.6153855324`；correct−arm mAP point分别为
  `+0.1307568556/−0.0160559964/+0.1194214838`；router0有正贡献，router1单独旁路反而略升；
- semantic controls的correct−arm mAP point：wrong-RGB=`−0.0066900358`、zero=`−0.0006964267`、
  orthogonal=`+0.0297891147`、slot-cycle=`+0.0633238773`、wrong-mask=`+0.0135262878`、
  generic-mean=`+0.1240184555`。只有generic-mean跨过`+0.1`；最高control是wrong-RGB，冻结semantic
  margin=`−0.0066900358 point`，因此`semantic_all_controls_margin=false`；
- route gap=`+0.1194214838 point`、correct floor=`57.1230075595>=56.7`均复现；九臂descriptor mean L2=
  wrong-RGB/zero/orthogonal/slot/wrong-mask/generic/bypass0/1/all=
  `0.0992314/1.7037808/2.1863244/2.2443638/0.9062496/3.0557334/1.9942849/2.1490941/
  2.7376952`，证明干预触达接口但不构成semantic retrieval优势；
- actual PID=`418044`自然退出，wrapper退出；postflight 5/5 gate PASS，GPU=`2 MiB/0%`且无compute
  process；NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow/AMP数值warning全部0。repo exact HEAD=
  `11d7a35788c4645c355d96d76a2a4ff20a9801ac`且clean；唯一checkpoint/config SHA执行前后exact；
- formal result/runner/manifest SHA256=
  `e52af8971f11a950e082bd0e5233e9a9679321fce8865f7e5999c52082642c4e`/
  `68df35d43a322815c759756c6f158353304abcfbcb1b36fbda209e0b774a37af`/
  `bcb8fe1a62cc90b5469ac3a09dac7b91acf8307d53b3b5c8a9ffa319289e03d7`；
- 最终判定=`CURRENT_SEMANTIC_INTERFACE_NO-GO / PHASE-B FORMAL MECHANISM DESIGN NO-START`。
  exp401的`RICH_BUDGET_ROUTE_ALIVE`保留，但不能把其弱mAP贡献归因给图像特定、slot特定的student rich
  evidence。exp402禁止重跑、补跑、删control或调rho/loss/batch；该NO-GO只关闭当前C0 student-evidence/
  expert语义解释，不永久否定Phase0E、Phase0R或CLIP–TAPF。
