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
- unit/synthetic tests：`PASS`（85/85）
- formal preflight：`PASS`（20/20）
- 4090 checkpoint/data/disk/GPU 资源：`PASS`
- 4090 isolated deployment：`PASS_AT_F053A43`
- 4090 remote CPU preflight：`PASS`（20/20 formal + 85/85 regression）
- Gate A prepare：`PASS_FOR_PREPARE_ONLY`
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
| 4090 execution source | 完成 | exact `f053a43…` 新隔离 clone，bundle 双端验签、tracked SHA 与 clean status 全 PASS；基础 dirty repo 未修改 |
| legacy Python 3.8 compatibility | 完成 | 两处 `removeprefix` 等价改写后，远端 Python 3.8 完整 20+85 PASS |
| Gate A prepare | 已授权未启动 | 必须 outer lock、单 wrapper、≥100 GiB；PREPARED_ONLY 后停，不得链 run |
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

1. 从 full commit `a02feff714f235e8985fa354fe1e31be42e2c87d` 创建 bundle；本地与
   远端分别核 bundle SHA、`git bundle verify/list-heads`，clone 目标必须事先不存在；
2. 新 clone detached checkout exact full SHA，逐项核 production/tests SHA；data 仅称
   策略式只读 symlink，并由 runner 的 path/content hash 做 pre/post 审计；
3. clone 内用 `uv` 创建 `.venv`，只复用已核候选 base Python 的 system packages；
   `.git/info/exclude` 排除 `.venv/`，pytest 禁用 cacheprovider，所有 JUnit/log 放仓库外；
4. 远端独立重跑 20 formal + 85 regression；前后核 exact HEAD、clean status、源码 SHA、
   environment freeze SHA、GPU/进程/磁盘；当前已全 PASS；
5. 只运行一次 prepare：outer `flock -n` 覆盖 staging 全生命周期，唯一 wrapper/PID/log
   位于 execution root 外，前置磁盘至少 100 GiB，命令绝不链接 run；
6. prepare 会加载冻结 checkpoint 并解析历史 flat parity 指标，正确口径不是“无指标
   读取”，而是“matching 不读取当前 arm/per-query 结果”；prepare 后至少 80 GiB；
7. 审计 PREPARED manifest/cache/mapping/hash/schedule 且单独升级授权后才允许 run；
   RUN_COMPLETE 后再次单独授权 summarize。Gate A 即使 GO，也不授权新机制训练。
