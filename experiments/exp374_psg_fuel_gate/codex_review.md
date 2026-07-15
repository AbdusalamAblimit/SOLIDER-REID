# exp374 训练/评测前 Codex 交叉审查

## 审查方式

用户要求任何训练或正式测试开始前完成充分审查，并明确禁止 Claude。本实验以多路
独立 Codex 红队替代旧流程中的 Claude 审查，覆盖：

1. PSG 与端到端 pose 新颖性；
2. transport/graph/attention 直接先例；
3. 数学可归约性与计算可行性；
4. 反事实设计与统计门槛；
5. checkpoint provenance、实现入口与资源安全。

## 当前总裁决

- 设计：`PASS_FOR_AUDIT_SCRIPT_DESIGN`
- unit/synthetic tests：`PASS_LOCAL_AND_REMOTE`（87/87）
- formal preflight：`PASS`（20/20）
- 4090 checkpoint/data/disk/GPU 资源：`PASS`
- 4090 isolated deployment：`PASS_AT_BFFB8BE`
- 4090 remote CPU preflight：`PASS`（20/20 formal + 87/87 regression）
- Gate A prepare：`NO_GO_AFTER_A01_E_SCENE_NEGATIVE`
- C+ signed-raw 修复设计：`PASS_FOR_CODE_DESIGN`（两路独立红队，同一 SHA）
- C+ 本地实现/全量复验：`PASS`（exact `8ca57ed…`，pytest 9 JUnit 126/126）
- C+ 远端历史环境全量复验：`PASS`（pytest 8 JUnit 121/121 + 5 个源码内 subTest row）
- A02 prepare-only：`FAILED_NONREPORTABLE_OFFICIAL_MIRROR_PROTOCOL_BUG`，attempt 永久烧毁
- A02 official-mirror 全资产只读证据：`PASS`（两路独立审查）
- production relation-v2 修复：`PASS_LOCAL_IMPLEMENTATION_AND_TEST_EVIDENCE`
  （audit `345056f4…`，protocol `99c72a8a…`，本地 207 direct + 146 subtests）
- relation-v2 历史 Python 3.8/torch 1.13.1 远端复验：`PASS`
  （exact `2b1b17f…`，JUnit 207/207 + runner 146 subtests）
- 当前授权：`PASS_FOR_A03_PREPARE_COMMAND_REVIEW_ONLY`
- A03 `prepare`、Gate A `run`/`summarize` 与训练：`NO_GO_FOR_EXECUTION`

统计、工程与独立总红队已分别完成第三轮只读签字；该 PASS 只授权编写 audit-only
脚本。当前 audit-only runner、协议层、模型三态 seam，以及对应的纯 CPU/synthetic
测试文件均已完成多路静态审查；冻结 SHA 对应的 85 个纯 CPU/synthetic tests 已全部
PASS。三项 formal preflight 也已分别完成自审与独立交叉静态审查，并在独立 CPU
进程中按冻结 SHA全部 PASS；production runner 重构后，原 85 个 regression tests 亦
原样重跑全 PASS。checkpoint provenance 仍只能支持 legacy screen；在完成 4090 真实
资产与资源只读复审、并把总裁决显式升级为 `PASS_FOR_LEGACY_GATE_A` 前，仍禁止
prepare、读取真实 checkpoint/data 或指标、GPU 推理、训练、正式评测或用旧 flag
拼出近似干预。

4090 资源复审后，三枚冻结 checkpoint、六份日志、Occluded-Duke RGB/pose 资产、磁盘和
GPU 均 PASS；但基础仓库是旧 commit 且有大量用户工作树，不能原地修改。部署计划红队
初审对原方案判 `FAIL_AS_WRITTEN`，指出 bundle 双端校验、Git provenance、环境冻结、
data symlink 口径、prepare 外层锁、历史指标读取口径和前后磁盘门槛必须修正。当前已
逐项接受这些条件，只授权建立新隔离 clone 并运行远端 CPU preflight；prepare 尚未授权。

首次远端 CPU 执行中，formal 20/20、protocol regression 14/14、model seam regression
37/37 均 PASS；runner regression 为 29/34，五项统一因 Python 3.8 缺少
`str.removeprefix` 失败。该失败发生在 synthetic checkpoint key/prepare resume 单测，
未加载真实资产、未运行 prepare。修复只把两处 `removeprefix` 改为严格等价的
`startswith + slice`；独立静态审查已签 `PASS_FOR_FULL_RETEST`，本机完整 20+85 又全部
PASS。兼容修复 commit=`f053a43cd520ff6f93ffff2df7ece8b358b62150`，runner
SHA=`fc002330a4bb25711fc8caf977b30a3f29ec1f35c1be650cbf80d9a797db5b4d`。
正式 Gate A 仍未授权，必须先在远端历史 Python 3.8/torch 1.13.1 环境从头全量复验。

兼容修复已在新隔离 clone 的历史 Python 3.8.20/torch 1.13.1+cu117 环境从头复验：
formal 20/20、regression 85/85 全部 PASS。执行后 exact HEAD、Git clean、三份 production
与六份 tests SHA、environment freeze、GPU 空闲、无旧 marker 均 PASS。当前只升级为
`PASS_FOR_PREPARE_ONLY`：prepare 必须由外层 `flock -n` 覆盖 staging 全生命周期，
前置磁盘至少 100 GiB，命令不得链接 run；`PREPARED_ONLY` 后必须退出并重新审查。

首次 prepare 随后按上述边界安全失败：默认 manifest 把三份 checkpoint 的 flat parity
日志错绑为旧 nested `test_default/test_log.txt`，而当前 checkpoint 对应的是同 seed 目录
下唯一一组指标的 `test_default.txt`。进程在任何 Gate-A forward 和 canonical output root
创建前退出，失败日志、锁和 PID 证据保留。修复 code commit
`1b3e155aadd63f429f5652f88692d4437ddfd0de` 只替换三条路径，并新增默认
seed→flat-path→expected mAP/R1 精确断言与 manifest 指标不符的 fail-closed 测试；
两轮独立静态红队 PASS，本机从头重跑 formal 20/20 与 regression 87/87（另 5
subtests）全部 PASS。旧
`PASS_FOR_PREPARE_ONLY` 因 execution source 已变化而自动撤销；必须把新 exact commit
部署到新隔离 clone，并在历史环境从头重跑 20+87 后，才能重新申请 prepare-only 授权。

该重验已完成：完整 bundle、新 clone、detached exact HEAD、production/tests SHA、历史
Python 3.8 环境冻结值以及 formal 20/20 + regression 87/87 均通过双端与独立红队核验。
新 execution source 固定为 `bffb8be252b4a155ce404618362ee42f2a76b1cc`，当前只重新授权
一次使用全新 output/lock/PID/log 的 `prepare`，仍不授权 Gate A `run` 或 `summarize`。

该一次性 attempt 随后在训练 split cache 阶段触发 `E_SCENE_NEGATIVE` 并 fail closed；
没有 canonical `gate_a_*` execution、PREPARED marker、arm/result 或 GPU 推理，wrapper
退出码为 1，PID 已退出。该授权已消耗且 attempt 禁止复用；当前回退为 diagnosis-only，
任何修复必须先证明不会改变 correct-arm 实际 PSG 输入或伪造非负性，再重新走静态与
双端全量测试授权链。

两路独立只读红队已对修复空间作出一致裁决：全量 clamp、阈值式 tiny-negative clamp
都会改变历史 raw cache/actual input，直接 FAIL；把 signed raw 直接塞入原 L1/entropy/
centroid 公式也因负质量而 FAIL。唯一可进入代码设计审查的是 C+：raw 因果输入与
positive-part 统计视图分离，centroid 只用 Hpos 求几何但对 Hraw 做同位移，实际 PSG
input 与 correct/shuffle/group provenance 不变。两路独立红队已对相同 design SHA
`87fcff641a893ecbb35dc0316334566f7a2378023237710b732de916d5b80df9` 给出
`PASS_FOR_CODE_DESIGN`。该 PASS 只授权实现代码与测试；不授权写完即执行测试，更不
授权 A02。

C+ 实现随后经过三轮双路静态红队：先修正 actual-space CPU/CUDA provenance、独立
little-endian 字节 oracle 与 sample-order 覆盖，再把 correct hook SHA 门禁前移到任何
ReID 指标计算之前，并把 centroid runtime 降级限制为精确几何/能量错误白名单；TOCTOU、
hook、模型与 provenance 错误仍全局失败。最终 production commit=`4b3a07f…`，test-only
fixture commit=`8ca57ed…`。干净 detached worktree 对最终 commit 从头重跑六组：formal
`18+1+1`、regression `19+38+49`，共 126 个 JUnit case 全部 PASS，0 error/failure/skip。
两路证据红队均签 `PASS_FOR_REMOTE_RETEST_ONLY`；当前只允许在历史 Python 3.8 / torch
1.13.1 环境完整重跑六组，仍不授权 A02。

远端完整 bundle 随后以 SHA
`773b4a46f0d7db6a4d8c6f9d894f8961ac2bbf10caa9e4f94c8fa13bc2672696`
部署到全新 v4 clone。三份更早的 Git 外执行证据分别保留 SHA 转录失败、环境版本字段
转录失败和 post-audit 规则误报；前两者没有启动测试，第三者六组 RC 均为 0，但不作为
最终成功目录。v4 从 bundle 验签、clone、uv 环境开始完整重跑，exact detached HEAD、
Git clean、9 项源码 SHA、环境 freeze 和 37 项历史 bytecode pre/post manifest 全部通过。

远端 pytest 8 的六份 JUnit 为 `18+1+1+19+38+44=121`，全部零
error/failure/skip；冻结 runner 源码另含 5 个同步 `unittest.subTest` row。pytest 8 不把
这些 row 展开成 JUnit testcase，而本地 pytest 9 会把它们加入 testsuite header，因此
本地 126 与远端 121 是版本计数语义差异，不是少跑。六份 JUnit、每组 rc/out、前后
源码/bytecode、环境和 post-audit 已逐文件回传到 Git 外
`remote_artifacts/exp374_remote_retest_8ca57ed_v4/`，传输前后 SHA 全一致。主线与独立
红队均重算通过，裁决为 `PASS_FOR_A02_PREPARE_REVIEW_ONLY`；该裁决只允许下一步
审查 A02 prepare-only 命令，尚不授权 A02，更不授权 `run`、`summarize` 或训练。

## 分项裁决

| 审查项 | 状态 | 裁决 |
|---|---|---|
| 原 PSG 新颖性 | 完成 | FAIL：WACV 2020 + SFT/FiLM 直接覆盖 |
| joint pose+ReID 新颖性 | 完成 | FAIL：PABR/VPU/VI-ReID 已覆盖 |
| 原 UBCFT 数学新颖性 | 完成 | FAIL：可归约 residual attention/GAT/Laplacian |
| 原 heatmap W2 trust region | 完成 | FAIL：计算量与约束漏洞阻塞 |
| source/demand + skeleton + Sinkhorn 改写 | 完成 | FAIL：差分不足，易被解释为 HOReID/FRT/RFC + UNITE/SOT/Sinkformers 拼接 |
| Gate A 对应依赖协议 | 完成 | 三路第三轮均签 `PASS_FOR_AUDIT_SCRIPT_DESIGN` |
| checkpoint 文件完整性 | 完成 | PASS |
| exact execution provenance | 完成 | FAIL：目录复用、文档与当前 checkpoint 日志错代、无 Git SHA |
| 4090 数据/磁盘/GPU | 完成 | PASS：RGB/pose 三 split 齐全，可用约 216.6 GiB，4090 无计算进程 |
| 4090 execution source | 完成 | exact `bffb8be…` 完整 bundle 双端验签，新隔离 clone detached clean；基础 dirty repo 未修改 |
| legacy Python 3.8 compatibility | 完成 | 两处 `removeprefix` 等价改写后，远端 Python 3.8 完整 20+85 PASS |
| Gate A prepare | A01 安全失败，授权消耗 | `E_SCENE_NEGATIVE`；只允许诊断与修复审查，禁止重试 |
| C+ signed-raw 设计 | 两路独立复审完成 | `PASS_FOR_CODE_DESIGN`；Hraw 因果输入、Hpos 统计视图、signed manifest 与 centroid 语义已唯一化 |
| C+ signed-raw 实现 | 本地与远端完整复验完成 | exact `8ca57ed…`；本地 pytest 9 JUnit 126/126，远端 pytest 8 JUnit 121/121 + 5 个源码 subTest row；`PASS_FOR_A02_PREPARE_REVIEW_ONLY` |
| true bypass 语义 | 完成 | PASS：同模型传 `pose_dict=None` |
| matched donor/centroid 实现 | formal preflight PASS | runner/protocol 静态复审、synthetic regression 与完整 N=128 matching preflight 均 PASS |
| per-query/层级 bootstrap | 单元测试 PASS | 两 primary contrasts 的 synthetic test PASS；正式输入 preflight 未做 |
| publication state machine | formal preflight PASS | 18/18：execution lock、arm/run/results 原子发布、哈希恢复与语义漂移拒绝均通过 |
| 真实 Swin seam | formal preflight PASS | 1/1：随机初始化真实 Swin-Tiny 的插入顺序、encoder 输入、bypass 与状态不变性均通过 |
| 资源安全 | 设计完成 | 492 passes、4.25–4.5h、矩阵 hash 后释放、80GB 门槛；正式资源 preflight 未做 |
| 不确定度约束联合 transport 数学对象 | 未完成 | BLOCKED：当前只有问题描述，没有清晰联合目标与可行域 |
| 2026 TTPM / Pose-Guided Feature Restoration 全文边界 | 部分完成 | TTPM 已核；后一篇仍 BLOCKED，阻止新机制训练 |

## 已否决的执行捷径

1. 不用 `heatmap=0` 冒充 no-pose；
2. 不用 `POSE_ENABLED=False` 做同 checkpoint bypass；
3. 不用 training-only `POSE_SHUFFLE` 做正式评测；
4. 不用 LGPA 的 fixed-bands flag 干预 PSG；
5. 不把 exp371 的 LGPA correct/shuffle 结果外推到 PSG；
6. 不在全量 gallery 上构造稠密 Hungarian cost；
7. 不在看到指标后修改 donor、centroid、阈值或解剖组；
8. 不因旧 checkpoint provenance 不足就把 legacy screen 包装成正式复现。

## 代码实现前置要求

### 第三轮签字摘要

- 统计红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`；
- 工程红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`；
- 独立总红队：`PASS_FOR_AUDIT_SCRIPT_DESIGN`。

三路均明确：该签字不授权测试或正式评测。

设计交叉审查通过后，audit-only 脚本至少必须实现并静态证明：

- checkpoint/config/path/content SHA manifest；
- no-flip correct parity 与历史 flat 日志逐 seed 对齐；
- query/gallery split-local、exact-person-count 的 sparse bijective donor matching；
- PID 不同、person count 相同、无 fixed point；
- 冻结 nuisance/cost/k/Gumbel/solver/tie-break/20 mapping 与 1,000 baseline seeds；
- 在两个 PSG block 实际消费的 sigmoid-resized tensor 上通过弱干预门禁；
- train-only final-scene centroid fitting、zero-padded translation 与输入能量保持；
- flip 只作 secondary，并对同一受控 scene bundle 同步执行；
- audit-only 三态 override、strict state load、active-module inventory 与同模型 true bypass；
- correct、20 shuffle、centroid secondary、bypass、七组 group-bundle corruption sensitivity；
- per-query AP/R1/margin；
- 三 seed 固定 paired blocks + 同步 PID-cluster bootstrap；
- shuffle/bypass 两个 primary contrasts 的 one-sided simultaneous max-deviation intervals；
- 七组只作 secondary sensitivity，不触发 GO；若额外报告 seven-group simultaneous
  interval，必须使用冻结设计中的同一 PID replicate，不能事后选择性添加或删组；
- create-exclusive output、atomic publish、逐项 hash resume 和异常 fail closed；
- 逐臂释放 feature/distance matrix，启动前目标卷至少 80 GB。

## 下一轮审查顺序

1. 先显式提交本轮三份审查文档，取得包含 `1b3e155…` code ancestry 的最终 exact
   execution commit；只从该最终 commit 创建 bundle，本地与远端分别核 bundle SHA、
   `git bundle verify/list-heads`；
2. clone 目标必须事先不存在并按新 exact SHA 命名；新 clone detached checkout 与
   bundle 相同的 exact full SHA，逐项核 production/tests SHA；data 仅称
   策略式只读 symlink，并由 runner 的 path/content hash 做 pre/post 审计；
3. clone 内用 `uv` 创建 `.venv`，只复用已核候选 base Python 的 system packages；
   `.git/info/exclude` 排除 `.venv/`，pytest 禁用 cacheprovider，所有 JUnit/log 放仓库外；
4. 以含 `1b3e155aadd63f429f5652f88692d4437ddfd0de` code ancestry 的最终 execution
   commit，远端独立重跑 20 formal + 87 regression；前后核 exact HEAD、clean status、
   源码 SHA、environment freeze SHA、GPU/进程/磁盘；当前已全 PASS；
5. prepare-only 已重新授权；只运行一次 prepare：outer
   `flock -n` 覆盖 staging 全生命周期，唯一 wrapper/PID/log
   位于 execution root 外，前置磁盘至少 100 GiB，命令绝不链接 run；
6. prepare 会加载冻结 checkpoint 并解析历史 flat parity 指标，正确口径不是“无指标
   读取”，而是“matching 不读取当前 arm/per-query 结果”；prepare 后至少 80 GiB；
7. 审计 PREPARED manifest/cache/mapping/hash/schedule 且单独升级授权后才允许 run；
   RUN_COMPLETE 后再次单独授权 summarize。Gate A 即使 GO，也不授权新机制训练。

## 2026-07-15 当前覆盖顺序

上面的部署顺序保留为历史记录。C+ exact `8ca57ed…` 的本地与远端全量复验现在均已
闭合，当前只允许：

1. 显式提交 `monitor.md` 与本文件的远端证据记录；
2. 为全新 A02 output/lock/PID/log/status 设计 prepare-only 命令并做只读红队；
3. 只有命令、GPU/进程、磁盘、clone/source SHA 和旧 A01/A02 目标隔离全部 PASS，才可
   单独签发一次 A02 prepare-only；
4. A02 只能执行 `prepare`，看到 `PREPARED_ONLY` 后退出；随后再次独立审计 cache、
   mapping、signed manifest、energy、schedule 和 hash；
5. 未取得下一轮明确签字前，Gate A `run`、`summarize` 与任何训练继续 `NO_GO`。

## 2026-07-15 — A02 prepare-only 执行与官方 mirror 根因复审

两路命令红队先后对最终 wrapper
`7c79818bdecdeb9546939707461d8412cf68ff774e57e721914fb20b9f2feb61`
和 launcher
`806c1a69ebc7b9ef39544c46f80ebaec6f5ad28486a15e5ef7558360494f8fb8`
从头复签。锁、PID/status 握手、A01 永久指纹、v4 exact source/env、data/exclude、
资产、唯一 9-token prepare 命令、失败证据保留与禁止 run/summarize/resume 边界均
PASS；只授权一次全新 A02 `prepare`。

A02 在 metadata/cache 物化后以
`E_SPLIT_CONTENT_OVERLAP: query/gallery/rgb_sha256` 安全失败，exit code `1`；没有
published execution、arm、指标或训练。主线只读重算与三路独立诊断均得到完全相同的
关系结构：train/eval 四类标识 0 overlap；query/gallery 恰有 1870 组一对一 RGB 与
pose-content mirror，全部同 basename/PID/camera/view/person-count/frame，Hraw/score/
nuisance bitwise 相同，path 与 pose-path 仍分离，0 forbidden。exp371 的已冻结 content
sidecar/dry-run、Occluded-Duke 官方 lists 和仓库标准 same-PID+same-camera junk removal
共同证明这是官方 split mirror，不是 retrieval 泄漏。

独立裁决统一为：

- A02 fail-closed 行为：PASS；
- blanket cross-split content-disjoint 假设：FAIL；
- 数据污染：未发现；
- A02 resume/复用/同名重试：NO-GO；
- Gate A run/summarize/训练：NO-GO；
- 下一步：只允许 design-first 的 strict official-mirror relation gate 及静态复审。

修订设计要求 train/eval 仍绝对不交，query/gallery 只允许 group size=2、1Q+1G、同
basename/PID/camera 等全部关系一致的 official mirror；pose-content pair 集必须与 RGB
pair 集逐对相同，cached PSG 因果输入也必须 bitwise 相同。official list、basename
relation、RGB bytes relation 与每对内容均进入 canonical premetric payload；matching
继续 split-local，`eligible_pair` 不放宽。合成负例矩阵、A02 metadata 真实资产 preflight、
本地和历史环境全量复验全部 PASS 前，不得申请全新 A03 prepare-only。

## 2026-07-15 — official-mirror 只读诊断脚本与实资产证据签字

最终诊断脚本 `42218 B` / SHA256
`88db86bb09a8d7d6fde7394ba2d12f8b115d517f9609bffe3c40ffd8836c7348`，design SHA256
`39513400ce65652ca787cafbc69e2fe247716cb6e5e8091e8a9322650b568d68`。脚本把 official raw/
canonical lists、三份 A02 metadata、三份 active pose index、全部 RGB/constituent NPZ、
六个 q/g cache、loader full/effective 人员顺序、source PID、standard evaluator junk
predicate、RGB/pose endpoint pair、Hraw/score/nuisance endpoint hashes 和 full joint pair
投影绑定在同一只读证据链中；两路静态红队均签
`PASS_FOR_READONLY_A02_DIAGNOSTIC`。

实资产 stdout 报告 `9290 B` / SHA256
`7b070824f86304e9ce4a4fd24e69b0b1c2bda6bea1f24c049c5b844b79553fa2`，两路独立复算签
`PASS_FOR_A02_ASSET_EVIDENCE`。报告确认：1870 official mirror 全部 standard junk，0
false/forbidden；RGB/pose endpoint pair digest 均为
`4135cdc4bb3cecd52dcf79423cf24d53595ce695a8b91544e2732be4bf3ebdfc`；full joint pair 为
`3542413 B` / `b82fd6aa1a81faf85e80b876a62bd892d259e3c7e1e9bb9d9a381641dbb3df93`；
三 split `target_outside_effective_count=0`；六 cache 前后整文件 SHA/identity 与全局
finite 均闭合。

本轮 PASS 只证明 A02 数据关系可由 strict official-mirror gate 安全白名单化，并授权
production relation 修复的代码设计审查。它不授权改完即测试，不授权 A03 prepare、
Gate A run/summarize、任何 arm/per-query 指标或训练。

## 第四轮 production relation code design 红队（2026-07-15）

三路独立审查只读复核 exact design
`fcaecc2cf9b78ae883514194ed32c2b538e07336d3ac6210b10434a748c6ba6f`：prepare/runtime
工程路与独立 overlap/protocol 路签
`PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN`；matching 专项红队拒绝签字。专项阻塞为：
token 虽绑定 full report/record SHA 和每个 global slot 的 record SHA，却没有绑定每个
person-count 的完整 global membership。仅凭传入 local records 与 indices，底层可以接受
合法 stratum 的 proper subset 或拆分调用，违背“错 subset/未审计调用必须拒绝”的设计
声明。两路旧 PASS 因 exact design 后续改变，只保留为审查历史，不作为最终实现授权。

修订后的 exact design 为 `60679 B` / `970` 行 / SHA256
`c5d43998c8dea7cb76a6163c096d96a8b690fc8a346c89e41f86fef0cab42406`。冻结契约新增：

1. private factory 从已验证的 full-split records 唯一生成
   `strata_global_indices: person_count -> exact ordered tuple[global_index]`；
2. tuple 必须覆盖该 stratum 全部且仅这些 rows，顺序固定为 full record 顺序，调用者无权
   提供、删减或重排；
3. `exact_sparse_candidates` 先要求传入 tuple 逐项完全相等，再逐 row 核 canonical SHA；
4. omission、superset、reorder、split-call、重复、越界、跨 stratum 均由
   `E_MATCH_RELATION_TOKEN` 拒绝，并进入 synthetic negative matrix。

三路现正针对新 exact SHA 从头复审。裁决收齐前仍为
`BLOCKED_PENDING_FINAL_DESIGN_REVIEW`；不授权实现、测试、A03 prepare、run/summarize、
指标或训练。

### 最终裁决

三路均已从头复审 exact design
`c5d43998c8dea7cb76a6163c096d96a8b690fc8a346c89e41f86fef0cab42406`，并分别签署：

- matching/full-stratum 红队：`PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN`；
- prepare/runtime 工程红队：`PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN`；
- official-mirror/overlap protocol 红队：`PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN`。

最终状态升级为 `PASS_FOR_PRODUCTION_RELATION_IMPLEMENTATION_ONLY`。该状态只允许实现
strict relation v2、完整 token/stratum binding、manifest/runtime gates 与冻结负例；实现后
必须先做多路静态代码审查并另行取得测试授权。当前仍禁止执行任何测试、远端操作、A03
prepare、Gate A run/summarize、arm/per-query 指标或训练。

## 2026-07-15 — relation-v2 production 实现与本地证据裁决

production 实现完成后，三路 Codex 分别针对 official mirror/overlap、完整 stratum token、
prepare/runtime state machine 与测试夹具做了多轮只读复审。最终冻结源码为：

- `audit_gate_a.py`：
  `345056f499567ea4f2c9e7cad3daa7a4d9e723939123eb38ebd7334d6a875b39`；
- `protocol.py`：
  `99c72a8a0bb2d26f2173cb2b8d50de281edbb801e33397725f1f20bd6f7af409`；
- `test_audit_runner.py`：
  `a8d26e0d4379647d209a4e88abc2f66b7b9fdef8368156c9e806fa4489407aca`；
- `test_formal_state_machine_cpu.py`：
  `ee709efc3722e455c28e47fd49b315594e6acf6557035a2c9c6d72ec881aca7b`。

关键闭包包括：strict-v2 完整 record projection/self-hash/immutable token；完整
person-count stratum membership；within/train-eval/qg full/effective constituent
谓词；1Q+1G official mirror 的 endpoint、metadata、Hraw/score/nuisance 等价；pairs 对
basename/RGB legacy+canonical/endpoint/joint metadata/joint pairs/count 的反向重算；
report/object/artifact/prepared SHA 三重绑定；prepare/run/summarize 五次 full relation
audit、seed/arm/COMPLETE 的 identity gates；A02 root/descendant 在入口与 failure writer
中均永久拒绝；稳定 FD/identity/TOCTOU 与文件 I/O 异常全部归一为协议错误。

第一次本地全量测试保留在 Git 外
`remote_artifacts/exp374_local_relation_tests_898156b_20260715/`。前五套 PASS，runner
暴露 15 个失败；其中唯一 production 缺陷是 `BURNED_A02_ROOT` 未与候选路径使用同一
`resolve()` 口径，macOS `/var`→`/private/var` 可绕过拒写。该缺陷以单行对称
canonicalization 修复并经独立复签。其余失败均来自严格谓词和 19-file quick identity
接口升级后测试夹具未同步；夹具改为真实构造目标违规，没有降低 production 断言。

最终六套均从头全量重跑，而非只补失败项：

| suite | pytest 口径 | JUnit tests | errors/failures/skipped |
|---|---:|---:|---:|
| formal state machine | 41 | 41 | 0/0/0 |
| formal protocol preflight | 1 | 1 | 0/0/0 |
| formal Swin preflight | 1 | 1 | 0/0/0 |
| protocol | 31 | 31 | 0/0/0 |
| model audit seam | 38 | 38 | 0/0/0 |
| audit runner | 95 + 146 subtests | 241 | 0/0/0 |
| 合计 | 207 + 146 subtests | 353 | 0/0/0 |

最终 Git 外证据目录为
`remote_artifacts/exp374_local_relation_tests_345056f_rerun1_20260715/`；其
`evidence_manifest.md` SHA256 为
`0179eedfd98833e042fba6ac95f37ac26601badf6564a7617606f3ecfbb767d3`，冻结命令、9 份
源码 SHA、12 份 log/JUnit SHA、环境与首轮失败边界。两路独立证据复审均签
`PASS_FOR_LOCAL_RELATION_TEST_EVIDENCE`。

当前授权只升级到：显式小步提交上述 exact source/tests/docs，构建并验签 bundle，再在
历史 Python 3.8/torch 1.13.1 隔离 clone 中全量复验。尚未授权远端真实资产 preflight、
A03 prepare、Gate A `run`/`summarize`、arm/per-query 指标或任何训练。

提交边界补充：本地 formal Swin 证据同时绑定工作树中的
`model/backbones/swin_transformer.py` SHA256
`e0223a1d0fbf1bd6fc9c46a55a35081fd570eab82743577feea425ce31d08c4d`；其唯一 diff 是把
默认 semantic weight 的硬编码 `.cuda()` 改为 `.to(x.device)`。该文件在本轮接手前已是
未提交脏改，当前不混入 relation-v2 小步提交。因为 HEAD 版本仍会在 formal CPU preflight
触发 `.cuda()`，在其提交归属明确前，不能声称仅凭新 relation commit 的 isolated bundle
可以复现 353/353，也不得启动远端全量复验。

## 2026-07-15 — Swin compatibility seam 独立提交裁决

主线已在用户授权技术取舍后，把既有 `.cuda()`→`.to(x.device)` 兼容修复作为独立提交
`75605b7592785e5e1f043f148b624e75807ba010` 固化；该提交只包含
`model/backbones/swin_transformer.py`，文件 SHA256 仍为
`e0223a1d0fbf1bd6fc9c46a55a35081fd570eab82743577feea425ce31d08c4d`。因此 relation-v2
源码提交与 backbone 兼容提交的归属相互独立，而后续 exact HEAD 同时包含本地 353/353
证据绑定的两部分源码。

裁决升级为 `PASS_FOR_EXACT_BUNDLE_AND_ISOLATED_FULL_RETEST`。该 PASS 只允许：显式提交
本记录、从最终 exact HEAD 构建/双端验签 bundle，以及在全新 isolated clone 中重跑六套
测试；它不继承工作树结果为 exact-clone 结果。isolated 复验和历史 Python 3.8/torch
1.13.1 远端复验尚未通过，因此 A03 `prepare`、Gate A `run`/`summarize`、真实 arm/
per-query 指标和训练仍为 `NO_GO`。

## 2026-07-15 — relation-v2 exact bundle 与历史环境复验最终裁决

Swin seam 归属闭合后，最终 execution source 固定为
`2b1b17f096aab11ec73f0d1534eb22535ff45412`。完整 bundle 大小为
`22,775,101 B`，SHA256 为
`07e2d8ceba224747a471b848b7b40bc525bb2f89080fccb789480d390521538b`；
双端 `verify/list-heads` 均确认完整历史、唯一
`refs/heads/exp/pose_heatmap` 指向该 exact HEAD。

本地全新 detached clone 先重跑六套：pytest 9 JUnit `353/353`
（`207 direct + 146 subtests`），0 error/failure/skip。Git 外证据为
`remote_artifacts/exp374_local_exactclone_2b1b17f_20260715/`，
`evidence_manifest.md` SHA256 为
`2d72d4e1c36702becaffde384832ffc9acc9ee35a9de452c3e87244c2b8f00a8`。

三路静态红队随后对远端 launcher exact SHA
`aa885e90a3e110a7a9dba6fb79d45b3e4c39fdda643249bb7eb6cce2d3581f5f`
复签 PASS。launcher 只允许在全新 clone 中创建历史环境、隐藏 CUDA，并运行六套
CPU/synthetic 测试；没有 data symlink、`prepare/run/summarize`、arm/per-query 指标或
训练入口。首个 scp 在链路中断后留下的 `2,611,200 B` 错误 partial 已按 SHA 单独归档；
完整 bundle 通过前缀续传、全文件大小/SHA 与原子替换后才进入正式路径。

4090 历史环境复验结果：

| suite | pytest 8 JUnit | errors/failures/skipped |
|---|---:|---:|
| formal state machine | 41 | 0/0/0 |
| formal protocol preflight | 1 | 0/0/0 |
| formal Swin preflight | 1 | 0/0/0 |
| protocol | 31 | 0/0/0 |
| model audit seam | 38 | 0/0/0 |
| audit runner | 95 | 0/0/0 |
| 合计 | 207 | 0/0/0 |

pytest 8 不把 successful `unittest.subTest` 展开为独立 JUnit case，因此又用原生
`unittest` 在同一 exact source 上独立执行，得到 `95 methods + 146 subtests` 全 PASS。
环境冻结为 Python 3.8.20、NumPy 1.24.4、pytest 8.3.5、torch 1.13.1、CUDA 11.7、
torchvision 0.14.1、timm 1.0.22 与 cuDNN 8500。

回传目录
`remote_artifacts/exp374_remote_retest_2b1b17f_relation_v2/` 的
`evidence_sha256.txt` SHA256 为
`5a326f3f2f13fdc4d58316f82eb6e60de2d299ce49fda38ad22d126d276128a5`，
30/30 项逐一通过；`final_status.txt` SHA256 为
`9aa1d54192702a116e062233947b5a394ca213d9bbdabd8a01c05abe7fe5f222`。
10 项源码、37 项 tracked bytecode 前后完全相同，post audit 为 detached exact HEAD、
Git clean、无 data/cache/Gate-A marker、无 GPU compute process。三路独立证据红队均复签
PASS；本地复核记录 SHA256 为
`97ec7b9a0109307e784125c74ebe69d7e109a151dfee9ffc22e94ce816c72fed`。

最终权限只升级为 `PASS_FOR_A03_PREPARE_COMMAND_REVIEW_ONLY`：可以设计和红队全新
A03 prepare-only wrapper/launcher。A01/A02 永久烧毁且不得复用；A03 `prepare` 本身、
Gate A `run`/`summarize`、arm/per-query 指标及任何训练仍未授权。

## 2026-07-15 — 收缩版 A06 最终审查与执行后裁决

针对用户要求“直接完成科学实验、减少形式门禁”，正式矩阵从 492 臂收缩为三 seed、
每 seed 四臂的 primary screen；每维匹配阈值固定为 `0.65`，随机 full-matching 基线固定
20 次，单 mapping 明确不伪造 MCSE/LOO。执行前六套测试为 `207 direct + 146 subtests`
全 PASS；执行、科学解释与 shell 边界三路 Codex 最终审查均为 PASS。

A06 12/12 完成后，原始结果给出 correct−shuffle mAP=`+0.001163 pp`
（区间 `[-0.363577,+0.377887]`），而 correct−bypass mAP=`+3.857684 pp`
（区间 `[+3.492944,+4.234408]`）。因此裁决是 `COMPLETE / NO_GO`，不是执行失败：
PSG 有效，但正确实例姿态没有提供可分辨的额外因果价值。不得继续 PSG 权重小变体；下一
方向必须更换状态交互机制，而不是把 PSG 搬到其他 backbone 后改名。
