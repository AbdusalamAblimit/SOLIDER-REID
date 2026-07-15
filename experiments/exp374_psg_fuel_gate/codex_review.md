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
- A02 prepare-only：`NO_GO_PENDING_COMMAND_REVIEW`
- 正式 Gate A/训练：`NO_GO_FOR_EXECUTION`

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
