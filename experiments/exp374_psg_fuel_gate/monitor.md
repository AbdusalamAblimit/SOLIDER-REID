# exp374 PSG 图像—姿态对应依赖门禁 — 监控记录

## 2026-07-15 — A03/A04 fail-closed 后收缩为 12-arm primary screen

- A03 在任何 checkpoint forward 前以 `E_PAIR_DIM max=0.618080...` 失败；原上限
  `0.50` 调整为 `0.65`，其余 hard matching/P95/baseline-cost 门禁不变；
- A04 在任何 checkpoint forward 前以 `E_MAPPING_HAMMING minimum=0.641176...,
  effective_unique=1` 失败，说明 20 个扰动 seed 最终只有一个有效匹配解；
- 决策：不再调 matching 优化器，也不运行 492 个大量重复的前向臂。改为三 seed、每
  seed 四臂，共 12 臂的 primary-only screen；secondary 延后；
- A05 证明 gallery 上 1,000 次随机 full matching 仍会让辅助准备压过实验本身；已安全
  终止并降为固定 20 次，该 baseline 不进入正式指标或决策；
- 代码与测试：affected protocol/runner suites 合计 `207 passed + 146 subtests passed`；
  训练仍未启动，尚无正式 Gate A 指标。

## 2026-07-15 — Goal 激活

- 状态：`PRE_EXECUTION_REVIEW_ACTIVE`
- 训练进程：无
- 推理/正式评测进程：无
- GPU：本实验未占用 3090/4090
- 当前动作：只读文献、数学、资产、因果协议与资源审查
- 约束：禁止 Claude；改用多路 Codex 红队交叉审查

## 2026-07-15 — PSG 与端到端 pose 查新边界

已完成两份专项审计：

- `experiments/paper_notes/psg_novelty_audit_20260715.md`；
- `experiments/paper_notes/end_to_end_pose_reid_novelty_audit_20260715.md`。

结论：

1. 原 PSG 是 WACV 2020 gated fusion 和 SFT/FiLM 类条件调制的受限实例；
2. PABR、VPU 与 VI-ReID pose auxiliary 已覆盖 pose branch 由 ReID loss 微调、
   联合 pose/ReID 训练和 pose loss 保语义；
3. freeze-then-unfreeze 只能是训练策略，不能承担主创新。

## 2026-07-15 — 数学/因果红队

原 UBCFT 公式与 trust region 触发两个阻塞项：

1. `Y=X+lambda(T-I)VX` 可归约为 residual attention/GAT 或图拉普拉斯扩散；
2. 全分辨率 heatmap 离散 W2 在计算和约束上均不可接受；batch 平均预算允许单个
   样本/关节严重漂移。

这一版 Gate 0 口径后来被正式复审替代，不能继续作为当前执行条件。当前 Gate A 只把
matched-shuffle 与 true-bypass 作为两个 primary contrasts：三 seed 均同号、最小均值
至少 `+0.30 mAP`、两个 simultaneous mAP LCB 均大于 0，且两个 R1 LCB 均不低于
`-0.50 pp`。scene-centroid 与七个解剖组均为 secondary，不触发 GO/NO-GO。

## 2026-07-15 — 文献红队

以下宽泛 claim 均已被封住：

- pose-aware ReID + OT；
- pose-guided graph/message passing；
- multi-stage pose attention/modulation；
- 从可见证据向遮挡区域传播；
- joint pose estimator + ReID。

第三路 source-to-sink 专项查新又确认，HOReID/FRT/PIRT/HUPOR/RFC 等工作分别并
在组合上覆盖 source reliability、visible-to-occluded recovery、pose topology 与
confidence-conditioned mixing；UNITE 已覆盖 keypoint-map 条件下对缺失身体部位
做 learned-mass UOT feature transport，SOT 与 Sinkformers 已覆盖 ReID OT feature
transform 和双随机 attention。因此，完整四元组合“未逐式同构”也不足以支撑主创新，
当前新机制正式为 FAIL/NO-GO。只有先形成差分足够清楚的不确定度约束联合优化问题，
才允许重新审查。TTPM 全文现已核为 pose-patch matching、confidence filtering 与
texture decoder 的强邻居；Pose-Guided Feature Restoration Transformer 正文仍是
额外阻塞项。

## 2026-07-15 — 资产与实现预审

4090 上三枚 exp007 checkpoint 的文件和 SHA 完整，GPU 空闲；但发现：

1. 同一输出目录存在两代测试日志；
2. 当前 checkpoint 对应 flat 日志，而文档引用旧 nested 日志；
3. 训练日志无 Git SHA；
4. 现有 shuffle/fixed-bands/zero-input 均不能直接实现 PSG Gate A；
5. 没有现成脚本同时满足 matched shuffle、scene-centroid、true bypass、
   per-query statistics 和哈希门禁。

当前裁决：

- 资产完整性：PASS；
- exact execution provenance：FAIL；
- 正式 Gate A 立即执行：NO-GO；
- 下一步：冻结设计，编写 audit-only 脚本，再做静态、合成、统计和资源复审。

## 2026-07-15 — 第一轮设计复审 FAIL，协议重写

统计红队否决了初稿中的 composite max-control percentile bootstrap、seed bootstrap、
未定义的 20-shuffle 聚合和把解剖组破坏作为 GO 条件。工程红队又证明
`camera × person_count` 硬分层存在 singleton，完美匹配数学上不可行；person-level
canonical 也无法保证 PSG 最终接收的 max-merged scene 输入只改变几何。

设计已按下列方向重写，并完成第二轮与独立总红队复审：

1. 两个 primary controls 分别定义 paired contrast，使用同步 one-sided simultaneous interval；
2. 三个 seed 固定为 paired blocks，只对 query PID cluster 重采样；
3. 20 mappings 先在 per-query AP 上等权平均，不平均 descriptor/distance；
4. 解剖组降为 group-bundle corruption secondary，不触发 GO；
5. donor 硬约束缩为 split/exact-person-count/different-PID/bijection，其余全进 soft cost；
6. scene-centroid 直接在 final scene heatmap 上做 zero-padded integer translation，
   但因多人峰间结构仍保留而降为 secondary；
7. 历史主口径固定 no-flip，correct 必须先复现 flat 日志；
8. Occluded-Duke 明确锁为 development，Partial-REID 锁为方法冻结后一次性确认。

第二轮统计主体 PASS；工程与独立总红队仍判 FAIL，指出 matching cost、Gumbel、弱干预、
centroid 边界、final-scene override、R1 聚合、反向 NO-GO、manifest/恢复和资源预算尚未
冻结。设计已逐项补齐，目前等待第三轮只读签字；签字前仍禁止实现脚本。

## 2026-07-15 — 第二轮/独立总红队后的设计修订

- matched-shuffle 的 95 维 nuisance、robust scaling、cost、soft weights、稀疏 k 序列、
  20 mapping seeds、Gumbel scale、solver tie-break 与 1,000 baseline seeds 已冻结；
- exact person-count 的 Hall/稀疏完美匹配失败仍 fail closed，不允许事后分箱；
- 在 PSG 真正消费的 `sigmoid(bilinear_resize(H))` 上新增干预强度门禁；
- final-scene override 三态入口、strict state load、UNSET/override seam parity、active
  module inventory 与 correct 前后 parity 已写成硬约束；
- primary 只剩 shuffle/bypass；R1 的 20-mapping query-level 聚合、bootstrap RNG、
  quantile 方法和反向 NO-GO 已唯一化；
- 492 passes 约 4.25–4.5 小时，大矩阵只留 SHA 后释放，目标卷启动前至少 80 GB；
- 三路第三轮只读签字均为 `PASS_FOR_AUDIT_SCRIPT_DESIGN`；只授权编写 audit-only
  脚本。unit/synthetic tests 仍 `NO_GO_FOR_TESTS`，正式 Gate A/训练仍
  `NO_GO_FOR_EXECUTION`。

## 下一检查点

1. 使用仓库内隔离 `uv` 环境运行已审签 SHA 对应的三组单元/合成测试；
2. 若失败，保持 `NO_GO_FOR_EXECUTION`，只修复并重新静态审查失败路径；
3. 测试 PASS 后补 execution lock、真实 solver 20-map、完整 arm hash/resume 的 preflight；
4. 所有单元测试和 preflight PASS 后，才允许 4090 顺序执行 Gate A。

## 2026-07-15 — Audit 实现已落盘，等待实现后静态签字

- 已编写、未运行：`protocol.py`、`audit_gate_a.py`、模型 final-scene 三态 override seam；
- 已编写、未运行：`test_protocol.py`、`test_audit_runner.py`、
  `test_model_audit_seam_cpu.py`；
- 当前没有测试、推理、正式评测或训练进程，3090/4090 均未被本实验占用；
- 本检查点只允许静态阅读和修复审查发现，`NO_GO_FOR_TESTS` 与
  `NO_GO_FOR_EXECUTION` 当时保持不变；后续状态见下一节。

## 2026-07-15 — 实现与测试静态审查 PASS

- 生产实现冻结 SHA：
  - `model/pose_backbone_model.py`：`b76a2e103a507bac0686b0b5562a6a7c5bcdbe8916a6afb59adb81343ca6ce37`；
  - `protocol.py`：`82427920ce2f1c11d4e0bb6efb7b6f76a0df5f7c12a78976ce04693f2920eb3c`；
  - `audit_gate_a.py`：`5df37ed1dd6e6aba34b19379a7efaa85e7e054405fe9fcf9445201859822352b`；
- 测试冻结 SHA：
  - `test_protocol.py`：`8e3ca5801930f253d0733bf19b5e253f1bf4a8f4f6e592666cb4faeff568bf4e`；
  - `test_audit_runner.py`：`c215928833b4801055e86da2f8655ddf55e84c5811268bfc0cd94f41f3ee966f`；
  - `test_model_audit_seam_cpu.py`：`cfad6b41d61fafabf3aad04ed2a8eb1b4c111a32a5128f1509de6c324592d363`；
- 三路交叉静态审查、AST parse 与 focused Ruff 均 PASS；未运行任何测试、推理或训练；
- 当前门禁升级为 `PASS_FOR_SYNTHETIC_TEST_EXECUTION`，只授权上述 SHA 的纯
  CPU/synthetic tests；正式 Gate A 继续 `NO_GO_FOR_EXECUTION`；
- 正式 preflight 仍需覆盖真实 solver 的 20-map/Hamming、execution lock、完整
  `RUN_ARM_MANIFEST`/arm marker/hash/resume 链，以及真实 Swin stage PSG 插入位置。

## 2026-07-15 — 纯 CPU/synthetic tests 85/85 PASS

- 隔离环境：仓库内 `.venv-exp374`，由 `uv` 创建并安装测试依赖；未污染系统 Python；
- `test_protocol.py`：14/14 PASS；
- `test_model_audit_seam_cpu.py`：37/37 PASS；
- `test_audit_runner.py`：34/34 PASS，另 5 个 subtests PASS；
- 首轮 protocol 执行中的两个失败均为隔离环境依赖缺失：依次补入 `torchvision` 与
  `timm` 后，冻结测试文件原样重跑 PASS；没有协议或断言失败；
- 仅出现 67 条 PyTorch JIT deprecation warnings，不影响数值、异常门禁或结果；
- JUnit XML 保存在 Git 外
  `remote_artifacts/exp374_local_unit_tests_20260715/`，三份 SHA 分别为：
  - protocol：`6ce8e00c23f820fe72606fcb4fbfd4a54e7503c36f3c48947945241279b3fa7b`；
  - runner：`8839f254f05b892e478140e2cd22feb2f4ac25d984fce011300f103892b9b631`；
  - seam：`a8981c07bb2257b97a9a015bb96bc3f8c23c176c2019db5407e7e2b24a1e4373`；
- 测试后六个实现/测试文件 SHA 与审签值完全一致；没有 prepare、真实数据读取、
  checkpoint 加载、GPU 推理、正式评测或训练；
- 当前状态升级为 `UNIT_TESTS_PASS_PREFLIGHT_REVIEW_REQUIRED`：允许编写和静态审查
  formal preflight，但 Gate A 继续 `NO_GO_FOR_EXECUTION`。

## 2026-07-15 — Formal preflight 静态审查 PASS

- production publication state machine 已从 runner 主流程中提取为可直接审计的
  `publish_run_completion`、`verify_run_completion` 与 `publish_or_verify_results`；
- production runner 冻结 SHA：
  - `audit_gate_a.py`：`6882e96886d4319c3760e7efc846a1d1a90c0b05615231862572cbcfc1192fda`；
- 三项 formal preflight 冻结 SHA：
  - state machine：`1240df696a93a0cf875fa2341da8d5979a859755357a6635b2a2f966fcc010cf`；
  - complete matching protocol：`4a80835b09163d2fb05c707ad374d70fa4ada4c34faf8ba81b28ced6787db7f6`；
  - real Swin seam：`c96e6b41661a640132e435d35a2db7030f33b5abeebc5d39eb91a1c4f98ede98`；
- protocol、state-machine 与 real-Swin 三路均完成自审和独立交叉静态审查，结论均为
  `PASS_FOR_PREFLIGHT_EXECUTION`；AST、focused Ruff、whitespace 检查均 PASS；
- 当前仅授权在独立 CPU 进程中顺序执行上述三项 preflight。protocol 与 Swin 使用
  外部 300 秒 timeout；Swin 进程启动前固定 OMP/MKL 各 4 线程；
- 正式 Gate A 仍为 `NO_GO_FOR_EXECUTION`：禁止读取真实 checkpoint/data、GPU 推理、
  正式评测与训练。任一 preflight 失败只修该路径并重新静态审查。

## 2026-07-15 — Formal preflight 20/20 与 regression 85/85 PASS

三项 formal preflight 均在独立 CPU 进程、外部 300 秒 timeout 下按冻结 SHA 原样执行：

- publication state machine：18/18 PASS；
- complete matching protocol：1/1 PASS，固定 N=128、Hall `selected_k=127`、1,000
  baseline 与 20 份 Gumbel min-cost mapping 完整执行；
- real Swin seam：1/1 PASS，CPU batch 1、384×128、OMP/MKL 各 4 线程；未加载
  checkpoint/data，未访问 CUDA/MPS。

production runner 重构后又原样重跑三组 regression：

- `test_protocol.py`：14/14 PASS；
- `test_model_audit_seam_cpu.py`：37/37 PASS；
- `test_audit_runner.py`：34/34 PASS，另 5 个 subtests PASS。

六份 JUnit 均位于 Git 外 `remote_artifacts/exp374_formal_preflight_20260715/`：

- `state_machine.xml`：`b90b6045b7aa1e323a22ae096af315ba2d64627a35637afc9b87b3e80f4a2c5e`；
- `protocol.xml`：`f3dca226e49fa523b2e29fdf0c0c9690e4facff597707dd71ec203140c9f427a`；
- `swin.xml`：`61436312bbea33ee75d6270ef953ef97016423816060da2b3b3efa7075f0c9d3`；
- `regression_protocol.xml`：`b104dfcc9a2fe0d1d34dcdc44ed3f64c6f688b01948ac2408bb29fdbb25d2c3e`；
- `regression_seam.xml`：`1a23bf03a592a1ebdb9e7d67455a47f9236f3b67d6d5ee1882e13545f9012786`；
- `regression_runner.xml`：`eb1a459088e2fdb0f0e7725330f024c4ac89edb3aeefd7cfe5d4ef08b5da647f`。

执行后 production/test SHA 与审签值完全一致。当前仅升级为
`FORMAL_PREFLIGHT_PASS_RESOURCE_REVIEW_REQUIRED`；正式 Gate A 继续
`NO_GO_FOR_EXECUTION`。下一步只做 4090 真实资产、磁盘、进程、GPU 与预计资源的
只读复审，禁止 prepare、读取 checkpoint 内容/指标、GPU 推理、正式评测或训练。

## 2026-07-15 — 4090 真实资产/资源 PASS，旧源码环境阻塞

独立只读资源红队未创建文件、未 prepare、未加载 checkpoint、未解析日志指标：

- 三枚 checkpoint 均为 `113,033,331 B`，SHA 与设计冻结值逐一一致；
- 六份日志只做文件 hash，不读取内容；事后确认前三个当时被误标为 flat，实际是旧
  nested `test_default/test_log.txt`，这里保留为历史资源审计记录：
  - seed 1234 nested/train：`2c42dd62c48119cacbb7979944762a51d3f0d85d117e6fc8a3f50359c386a4ac` /
    `f29a870b3d8edfadba64bc57abccfacd6ae351ee5a68c084b5604313da22472b`；
  - seed 42 nested/train：`9794757c8009f540950dd665fa7c9560f84315b920761fcfaeb126bb8c2631ca` /
    `091eb57cd0aea9c4b080bd0ba6699a814c771150fa58356d0de91a18d8c1cf14`；
  - seed 2024 nested/train：`57362690072021d75206e3be247b606eb534436f3b065524de0c772936744255` /
    `7c6000152f57508a788bdf6bc77c089c3cc68392b2622c2621608a145b61e392`；
- `/home/afr` 可用 `232,580,870,144 B`（约 216.6 GiB），超过 80 GiB 门槛；
- RTX 4090 D 仅 2 MiB 显存、0% 利用率，无 compute/exp374 进程；
- RGB train/query/gallery 数量为 `15,618 / 2,210 / 17,661`；pose_data 为
  `22,050 / 4,155 / 24,779`，六目录均存在；
- legacy Python 环境及所需核心包存在；`uv`、git、sha256sum、flock、timeout 均可用；
- 没有旧 exp374 输出根、锁或 completion/failure marker。

资源本身 PASS，但旧基础 repo HEAD=`715c020e…` 且 dirty，缺 `audit_gate_a.py`、
`protocol.py`，远端 `pose_backbone_model.py` SHA=`634939…`，不能原地覆盖，故资源红队
当时裁决 `KEEP_NO_GO_FOR_EXECUTION`。

## 2026-07-15 — 隔离部署计划红队与修订

原部署草案被判 `FAIL_AS_WRITTEN`，所有阻塞均属于部署流程而非 runner 代码：

1. bundle 必须绑定 full commit `a02feff714f235e8985fa354fe1e31be42e2c87d`，双端核
   SHA、bundle integrity 与 heads；
2. clone 目标必须不存在，detached checkout exact commit，逐项核源码/测试 SHA；
3. `.venv/` 写入 clone 私有 `.git/info/exclude`，pytest 禁 cacheprovider，禁止
   editable/setup install；环境冻结后不再升级包；
4. data symlink 只称策略式只读，真正的不变性由 runner 的 path/content hash 复核；
5. prepare 必须加覆盖 staging 全生命周期的外层独占锁，不能链 run；
6. prepare 会加载 checkpoint 并解析历史 flat parity 指标；冻结的因果隔离要求是
   matching 不接收任何当前 Gate-A arm/per-query 结果，而不是字面“零指标读取”；
7. prepare 前至少 100 GiB，PREPARED_ONLY 后再要求至少 80 GiB。

当前只授权双端验签 bundle、新隔离 clone、工作目录内 uv 环境和远端纯 CPU 20+85
preflight。正式 Gate A 与 prepare 继续 `NO_GO_FOR_EXECUTION`；remote preflight 全 PASS
后才可另行升级为 `PASS_FOR_PREPARE_ONLY`。

## 2026-07-15 — 首次远端 CPU preflight 的 Python 3.8 兼容失败

exact `a02feff714f235e8985fa354fe1e31be42e2c87d` 已经双端验签 bundle 部署到
新 clone `/home/afr/SOLIDER-REID-exp374-a02feff`；bundle SHA 为
`84afe11107053b2822402f56b92160bf4d68f01de286a4ff296ce901a4bf8a4d`，clone detached
HEAD 与全部 production/tests SHA 正确，Git status clean。`data` 只作策略式只读
symlink，并在 clone 私有 `.git/info/exclude` 中排除，不修改基础 dirty repo。

远端工作目录内 uv 环境固定为 Python 3.8.20、torch 1.13.1+cu117、torchvision 0.14.1、
timm 1.0.22、NumPy 1.24.4；只额外安装 pytest 8.3.5，freeze SHA=
`7699815505136173aa3f398ac43a0c82fabfa8af9aad2e769b3badaab32cd6c6`。执行结果：

- formal state-machine/protocol/Swin：18/18、1/1、1/1 PASS；
- regression protocol/model seam：14/14、37/37 PASS；
- regression runner：29/34 PASS，五项统一以 Python 3.8
  `AttributeError: 'str' object has no attribute 'removeprefix'` 失败。

失败只触及 synthetic checkpoint key normalization 和 synthetic prepared-resume tests；
没有运行 prepare、没有加载真实 checkpoint/data/指标、没有 GPU forward。原 Python 3.8
JUnit/log 和失败环境全部保留。曾尝试创建 Python 3.10 + legacy torch 的新 uv 环境，但
1.7 GiB wheel 下载/解压过慢；确认任务仍健康但尚未完成后主动 TERM，只保留为失败环境
证据，未改 base env，未切换到变量更大的 torch 2.6。

## 2026-07-15 — Python 3.8 等价兼容修复与本机全量复验 PASS

只修复上述失败路径：

- `str(key).removeprefix("module.")` 改为先 `startswith` 再单次定长切片；
- 已通过 `gate_a_` 前缀 require 的 execution dirname 同样改为定长切片。

独立静态红队核对 collision、重复前缀、空后缀、Unicode 与 fail-closed 行为完全等价，
结论 `PASS_FOR_FULL_RETEST`；AST、focused Ruff、whitespace 均 PASS。新冻结值：

- exact commit：`f053a43cd520ff6f93ffff2df7ece8b358b62150`；
- `audit_gate_a.py`：`fc002330a4bb25711fc8caf977b30a3f29ec1f35c1be650cbf80d9a797db5b4d`。

本机又从头重跑 formal 20/20 与 regression 85/85，全部 PASS；六份 JUnit 位于 Git 外
`remote_artifacts/exp374_compat_retest_20260715/`，SHA 为：

- formal state-machine/protocol/Swin：
  `51df064b0a50fae6420b59411bbf459cb7e506cd2e3994e53dbcf1e938aba27d` /
  `732949154ae83b26fa327217232c852a2a58c85ea977787ceed5d9559a81833e` /
  `b7f1522b57725e5aef43de6f9434826cd0d1d0715e158ca8123d1a23231d8d55`；
- regression protocol/seam/runner：
  `21c8d94b4e381f2a23392d76026ef108f2009bc9bae168b940554fdba176f674` /
  `8f577fe8ba0e384a383e3dce1e8335eeda662e3de4e17c7b525a1238326abd5f` /
  `91537b350cceeb04a543378306b2acf56f4bfe2a7952dafa792661f460abde0e`。

当前仍为 `NO_GO_FOR_EXECUTION`；只授权把新 exact commit 部署到新的隔离 clone，并在
历史 Python 3.8/torch 1.13.1 环境从头重跑 20+85。不得只补跑五个失败项。

## 2026-07-15 — 远端 Python 3.8 全量复验 20+85 PASS

兼容修复 commit 已从双端验签 bundle 部署到新的隔离 clone：

- exact HEAD：`f053a43cd520ff6f93ffff2df7ece8b358b62150`；
- clone：`/home/afr/SOLIDER-REID-exp374-f053a43`；
- bundle SHA：`2e1ba75d00aa5ace623962936fa065917dce9df405f87e077d6a4f2c64656466`；
- runtime：Python 3.8.20、torch 1.13.1+cu117、torchvision 0.14.1、timm 1.0.22、
  cuDNN 8500；uv freeze SHA 仍为
  `7699815505136173aa3f398ac43a0c82fabfa8af9aad2e769b3badaab32cd6c6`；
- 执行前后 Git status clean，production/tests SHA 与冻结值逐一一致。

同一环境中从头执行，不是只补五项：

- formal state-machine/protocol/Swin：18/18、1/1、1/1 PASS；
- regression protocol/seam/runner：14/14、37/37、34/34 PASS，另 5 subtests PASS；
- 六份 JUnit 位于 `/home/afr/exp374_remote_preflight_f053a43/`，SHA 为：
  - state-machine/protocol/Swin：
    `ff42c222c9e72e17771a505581a8013ae7c6d39edb84b21409cc7a910f59e59d` /
    `26e785709b436fd01d09807f9ead7edc9c73be41476976a7da2281da87a06338` /
    `9ff630082338415445cdddcab248e2e88b203dfa7c23b6f9a429cfb5565b0a38`；
  - regression protocol/seam/runner：
    `9d11a3aeca8a5a8c43ac93fb69e21090c6016a72bcf9fc375ee722a80f87545b` /
    `f54fcfb3b604fb6a08d361e1eaf8b7ba07b69e7399629c4f68828d7868e0f6b9` /
    `8c90a76ea43a3cd6e7ae79eeef4049d4ec8616058eb54a009bea5321f0fd0749`。

执行后无 exp374 正式 marker，GPU 0%/2 MiB，无 compute process；可用磁盘
`230,264,311,808 B`。当前裁决升级为 `PASS_FOR_PREPARE_ONLY`，仍不是正式 Gate A
run 许可。prepare 约束：

1. 输出根、outer lock、PID/log 必须事先不存在；
2. outer `flock -n` 覆盖 staging 全生命周期，只允许唯一 wrapper；
3. 前置同卷可用空间至少 100 GiB；
4. prepare 可加载冻结 checkpoint 并解析历史 flat parity 指标，但 matching 数据流不得
   接收任何当前 arm/per-query 结果；
5. 只运行 `prepare`，绝不链 `run`；`PREPARED_ONLY` 后 wrapper 退出；
6. 之后核 canonical execution dir、PREPARED marker、cache/mapping/energy/hash/schedule、
   无 arms/results/current metrics 且剩余至少 80 GiB，再另行决定是否授权 production smoke/run。

## 2026-07-15 — 首次 prepare 因 flat 日志错绑安全失败

唯一 prepare wrapper 使用 exact execution source
`f053a43cd520ff6f93ffff2df7ece8b358b62150`，在远端历史环境中按 outer `flock -n`
和 prepare-only 边界启动，但 `checkpoint_specs()` 对 seed 1234 旧 nested 日志触发
`E_FLAT_LOG_AMBIGUOUS`，在创建 canonical output root、读取 dataset、构造 matching 或
执行任何 Gate-A/GPU forward 前 fail closed。保留证据：

- `/home/afr/exp374_gate_a_f053a43.prepare.log`；
- `/home/afr/exp374_gate_a_f053a43.prepare.lock`；
- `/home/afr/exp374_gate_a_f053a43.prepare.pid`；
- PID `4185900` 已退出；
- `/home/afr/exp374_gate_a_f053a43` 不存在。

根因不是 checkpoint、数据或模型失败，而是默认 checkpoint manifest 把三份 flat parity
日志指向旧 nested `test_default/test_log.txt`。当前三枚 checkpoint 对应的正确 flat 日志
均为同 seed 目录下的 `test_default.txt`，且各自只有一组 mAP/R1：

- seed 1234：`58.3 / 68.1`，SHA
  `d8d724d10e5de8ad536dfb49bc74a250bafa9df96dfa780664efa98e5595d41d`；
- seed 42：`57.5 / 66.7`，SHA
  `200d18c80e279b1689bd31aff5948b5ee088d76116a07633aef3f313a345426b`；
- seed 2024：`58.0 / 68.4`，SHA
  `ede3c4c3bca332e6c5c2c53edb375aeb1d50cdb033ef67295dca5b340317eb3c`。

旧 nested 日志与当前 checkpoint provenance 不一致：seed 1234 还含两组重复指标，严格
parser 正确拒绝；seed 42/2024 分别是 `57.9/66.7`、`57.3/66.6`，也不允许拿来替代。
因此本次失败验证了 parser 与 prepare phase ordering 的 fail-closed 行为，不产生
任何可报告科学结果。

## 2026-07-15 — flat 日志修复与本机 20+87 全量复验 PASS

修复仅包含三条默认 `flat_log` 路径替换，并新增 seed→flat-path 精确回归断言，防止任何
seed 再退回 nested 日志；第二个 synthetic 用例还验证实际日志指标与 manifest 不符时
必须触发 `E_FLAT_LOG_MANIFEST`。两轮独立 Codex 静态红队核对路径、指标、checkpoint
provenance、Python 3.8 兼容性和 diff scope，结论均为 `PASS`。冻结值：

- exact code commit：`1b3e155aadd63f429f5652f88692d4437ddfd0de`；
- `audit_gate_a.py`：
  `b759e897962a50c75e946ae54cea6628594061fe8239b40d16035ab03cb99e4c`；
- `test_audit_runner.py`：
  `b8653cbf2db5786d91dd0322185cbe8d2dff4d2d7cdbea88f0155e27e90b672c`。

本机工作目录内 `.venv-exp374` 从头顺序执行：

- formal state-machine/protocol/Swin：18/18、1/1、1/1 PASS；
- regression protocol/seam/runner：14/14、37/37、36/36 PASS，另 5 subtests PASS；
- 六份 JUnit 位于 Git 外
  `remote_artifacts/exp374_flatlog_retest_20260715_local_v2/`，SHA 为：
  - state-machine/protocol/Swin：
    `0c7a36579de36f9338dd3b1fc66c6f2067dc734eebb59d2a76e5b60084f6284b` /
    `358db8b1806edcc1b1794ecb20a44cf529ba70edd3e7a9dabdfd713905038f4c` /
    `1e6a179d9c4d94f97092247aa44582bea0084d3d4035b0fea6ba8159042b58a6`；
  - regression protocol/seam/runner：
    `eb4e70f5a6107330b9c146d61b01b59eaad954949dfc100f316d3577682644a5` /
    `d96a2ac23662dad917b780b7a4b8c9bcf27622dc7b879bb9be669901f7137233` /
    `200b711ec415ff93afa6d8e13008625f083916ae216dc8b2137ff6bfe19ef5e3`。

当前状态回退为 `NO_GO_FOR_PREPARE`：旧远端 20+85 PASS 和 prepare-only 授权只绑定
`f053a43…`，不能跨 commit 继承。下一步只允许构建绑定新 exact commit 的增量 bundle、
双端验签、部署到新隔离 clone，并在同一 Python 3.8/torch 1.13.1 环境从头重跑
formal 20/20 与 regression 87/87；全 PASS 后才可重新申请 prepare-only 授权。新的
prepare 必须使用全新 output root、lock、PID 和 log，禁止 resume 本次失败尝试。

## 2026-07-15 — 新 execution source 远端 20+87 PASS

本地最终 execution commit、完整 bundle 与远端隔离部署已逐层验签：

- exact commit：`bffb8be252b4a155ce404618362ee42f2a76b1cc`；
- bundle：`remote_artifacts/exp374_execution_bffb8be.bundle`；
- bundle SHA256：
  `79cad8c71d9a120504d56c91069bdc5c3de95199fae5387a085b08b8def8bcac`；
- 远端 clone：`/home/afr/SOLIDER-REID-exp374-bffb8be`；
- bundle 双端 `verify/list-heads` 均指向同一 exact HEAD，完整历史无 prerequisite；
- clone detached exact HEAD，测试前后 Git status 均为空；基础 dirty repo 未修改；
- `audit_gate_a.py`、`protocol.py`、`pose_backbone_model.py` 与六份测试文件的远端 SHA
  均与本地逐项一致；
- runtime 仍为 Python 3.8.20、torch 1.13.1+cu117、torchvision 0.14.1、
  timm 1.0.22、NumPy 1.24.4、cuDNN 8500；freeze SHA 仍为
  `7699815505136173aa3f398ac43a0c82fabfa8af9aad2e769b3badaab32cd6c6`。

远端同一环境从头顺序执行，不是补跑新增测试：

- formal state-machine/protocol/Swin：18/18、1/1、1/1 PASS；
- regression protocol/seam/runner：14/14、37/37、36/36 PASS；runner 内 5 个
  `unittest.subTest` 属于同一 pytest item，JUnit 计 36 正确；
- 六个 suite 的 errors/failures/skipped 均为 0；
- JUnit 已回传到 Git 外
  `remote_artifacts/exp374_remote_retest_bffb8be/`，SHA 为：
  - state-machine/protocol/Swin：
    `48e910511c3c34c8d7b439dbedefb170e895b41f5c8273b4eefb0b617e92ac87` /
    `a3601e4145b85e983869459083db3737e78f861c4aa5feb7af67388e9b78c7e1` /
    `c6f8f4d5da907514c7a59ae763fc78096a183cbdefafe905c6d2b4729c468b36`；
  - regression protocol/seam/runner：
    `f33b1a70913921315d733312f897788c4d4c74e12f80d262fc6745ad058526c2` /
    `f5207c909c819548881cd23f1a554f15065dab5d379dc959283832071808ff7b` /
    `8af3326cef76e0e44907035aa17a0e9832c7fcc66a7a6fb1e3ca73275d6698b2`。

测试前后 GPU 均为 2 MiB/0%，compute process 为空；可用空间从
`230,169,919,488 B` 到 `230,167,740,416 B`，远高于 100 GiB；没有
`exp374_gate_a_bffb8be*`、PREPARED、RUN_COMPLETE 或正式 Gate-A 进程。独立远端证据
红队裁决 `PASS_FOR_NEW_PREPARE_ONLY`。

当前重新授权 exact `bffb8be…` clone 的一次全新 prepare，仍不授权 run/summarize：

1. output root、outer lock、PID、log 必须全新且不得含/复用 `f053a43`；
2. 禁止 `--resume`，outer `flock -n` 必须覆盖 staging 全生命周期；
3. 启动前重新核 exact HEAD、clean、无旧进程、目标不存在、同卷空间至少 100 GiB；
4. 命令只含 `prepare`，看到 `PREPARED_ONLY` 后 wrapper 退出；
5. 随后只读审计 20 mappings、Hamming、artifact hash、schedule=492、剩余空间至少
   80 GiB，并确认无 arms/results/current metrics；单独签字后才讨论 Gate A run。

## 2026-07-15 — prepare A01 在 scene 非负门禁安全失败

经最终 command 红队 `PASS_FOR_PREPARE_COMMAND` 后，唯一 attempt
`bffb8be_a01_20260715` 启动：

- output root：`/home/afr/exp374_gate_a_bffb8be_a01_20260715`；
- wrapper/launcher SHA：
  `c73425540c88d924ffd68bd170d5eda96ae96b771c6b15691d3750d79f15e809` /
  `9d704ba13de47ce9fa93b3c6d4b26ce3ca47486b954297cd6d59405d00149c93`；
- wrapper PID：`4188179`；
- 启动前可用空间：`230,159,323,136 B`；
- outer lock、PID、log、status 全部为全新 create-exclusive 证据。

wrapper 退出码为 1，结束时间 `2026-07-15T01:57:10Z`。日志显示三 split pose index
成功加载后，在训练 split `cache_split()` 调用 `summarize_scene()` 时触发：

```text
E_SCENE_NEGATIVE: heatmap
```

失败发生在 canonical execution 目录与 `PREPARED.json` 创建前：

- output root 内只有 `.exp374_prepare_zo18jmvf` staging，物理占用约 205 MiB；
- staging 含 partial train cache 与 `FAILED.json`；
- 没有 `gate_a_*` canonical dir、PREPARED、arms、results 或 COMPLETE；
- wrapper/Python 均已退出，GPU 回到 2 MiB/0%。

因此 A01 是不可报告的 prepare 失败，不产生 Gate-A 科学结果。该 attempt、lock、PID、
log、status 和 partial staging 必须保留，禁止删除后同名重跑。当前正式回退为
`NO_GO_FOR_PREPARE`：只允许量化负值大小/频率、追溯 raw pose 与 resize/merge 来源，
并审查 nonnegative 假设是否与实际 PSG 输入相容；在修复、回归测试、双端全量复验和
新的独立授权前，禁止任何 A02 或 Gate A run。

## 2026-07-15 — signed raw 只读溯源完成

A01 partial train memmap 全阵列只读扫描结果：

- shape=`(15618,17,96,32)`、float32；tensor data 逻辑字节数为
  `3,262,537,728 B`，`.npy` 文件另含 128 B header；
- A01 在前三批完成后预写当前 batch，只有 rows `0..1023` 已 materialize，其余 mmap
  页是未写零值；已物化 `53,477,376` 个元素中有 10 个负值，占约 `1.86995e-7`，
  **不得外推为完整训练集频率**；
- 全部集中在 row 904、0-based channel 6/10；min=`-7.576643110951409e-05`，负值中位数
  `-1.9857e-05`，最接近 0 为 `-3.3474e-06`；
- row 904 对应 `data/occluded_duke/bounding_box_train/0093_c2_f0068896.jpg`，有 6 人。

六份 raw pose NPZ 均 finite，但各自已有 5,692–20,692 个负响应，raw min 约
`-0.00466` 至 `-0.01169`；channel 6/10 也各有数百至数千负值。提取脚本保存的是
ViTPose-Huge hook 捕获的 MSE heatmap head raw output，并以 float16 落盘。因此负值是
真实历史输入的有限低响应；bilinear 只传播/平滑，6 人 max-merge 只在 6/6 同负时暴露，
不是插值凭空产生，也不是数据损坏。

当前修复裁决冻结为 C+：`Hraw` 保留为 PSG 因果输入，`Hpos=clamp_min(Hraw,0)` 仅作
nuisance/support/entropy/bbox/centroid 几何视图。全量或阈值 clamp raw、直接用 signed
质量算 entropy/centroid、以 Hpos 替代 correct arm 均禁止。完整语义与测试门禁已写入
`design.md`；下一步只做独立静态设计审查，仍为 `NO_GO_FOR_PREPARE`。

## 2026-07-15 — C+ signed-raw 设计双重复审 PASS

两路独立只读红队均核对同一份 `design.md`：

- SHA256：`87fcff641a893ecbb35dc0316334566f7a2378023237710b732de916d5b80df9`；
- 裁决：两路均为 `PASS_FOR_CODE_DESIGN`；
- 已闭合：partial mmap 正确分母、Hpos 全零时逐值保留 Hraw、split-level sorted unique
  负 channel payload、canonical JSON、Sraw/Spos/Delta streaming SHA、Rplus intervention
  质心、Hpos 几何/Hraw 位移与 Mneg 零质量规则；
- 全文未残留旧“raw 全非负”或 signed `R=S-0.5` 质量公式。

该 PASS 仅授权实现代码与测试，再进入静态代码审查；不授权执行测试、A02 prepare、
Gate A `run`、`summarize` 或训练。当前继续 `NO_GO_FOR_PREPARE`。

## 2026-07-15 — C+ 实现、静态红队与本地全量复验 PASS

实现严格保持 `Hraw` 为 correct/shuffle/group/override 的唯一 raw tensor，`Hpos` 只进入
nuisance 与 centroid 几何；新增 signed manifest、CUDA actual-space
Sraw/Spos/Delta、correct hook premetric 门禁、Rplus intervention、Hpos 几何/Hraw 位移、
Mneg 与 centroid runtime `INVALID_SECONDARY` 白名单。冻结提交：

- production + design：`4b3a07f4f01ef099b5b69677698234a7ea3ead76`；
- test-only fixture：`8ca57edc2bf7b5db66a0913dad2be2b4078a38d7`；
- production SHA（protocol/audit/model）：
  `b6536d2e8d1e6d9e48ce1d2a0e534dc9cc99fb1604c5e1c398dcfa6d80e6d0f7` /
  `94c36a363bc007abd7b286009272072fa009b8af9eb42022c35217e8d488c693` /
  `f60980f28bef0ded7b71b94bcd64d3f99373b42f8aecdc4bd7e558b2bb0a7100`；
- regression tests SHA（protocol/runner/model seam）：
  `c56adaf7e07db7915406a00dbe597cca71e55b948da825a53d58c7ea52d8d1b2` /
  `a678fda1190613bb652fd92f893fd3756c00c9b9e6a1f131d7e96f763c5635f9` /
  `cfa4451b103f8e5c9c3dacf78d978957f9c7cc68ba5d9e2ed87bf95984f5b2f3`。

第一轮完整本地执行中 formal 20/20、protocol 19/19、model seam 38/38 PASS；runner
三个新 fixture 失败：单像素负值未落在 12×4 bilinear 采样网格，以及 resume 用例先命中
execution-dir SHA。production 没有失败。失败 JUnit 永久保留于 Git 外
`remote_artifacts/exp374_signed_retest_4b3a07f_local/`；不作有效 PASS 证据。

test-only fixture 经双路静态复审后，在最终 exact commit 的全新 detached clean worktree
从头重跑全部六组，不是只补失败项：

- formal state/protocol/Swin：`18/18 + 1/1 + 1/1`；
- regression protocol/model seam/runner：`19/19 + 38/38 + 49/49`；
- 总计 126/126 JUnit case，0 error、0 failure、0 skip；runner 命令口径为
  44 passed + 5 subtests；
- 证据目录：Git 外 `remote_artifacts/exp374_signed_retest_8ca57ed_local_v2/`；
- JUnit SHA（formal protocol/state/Swin）：
  `d158662ff90f40b7c1dcde96df72992d5b117a2c5fa738fa99fb82979b4b5b18` /
  `e944f1d29395e98414b0d4c05d6b89f5f4a4922736493523dd39d5fe52639f19` /
  `f97f011f09d4cbe0f094d21e077027fd69eadc90a11289aa082fd7aa1d09e920`；
- JUnit SHA（regression model/protocol/runner）：
  `3037484056b4a0f75ea02a899c37fec94b72c48821b959b0157932100aacaa4a` /
  `8a7366be1ca20c68a2028cb73a60ffa3789f0d0ff6aca916004e228a03141bd9` /
  `8ece82c9f862d13e5ecb4b78f74df39010f675be40f4e1aec4fff4f184853769`。

两路独立证据红队均签 `PASS_FOR_REMOTE_RETEST_ONLY`。完整 bundle
`remote_artifacts/exp374_execution_8ca57ed.bundle` 的 SHA256 为
`773b4a46f0d7db6a4d8c6f9d894f8961ac2bbf10caa9e4f94c8fa13bc2672696`，记录完整历史且
head 精确指向 `8ca57ed…`。当前只允许新隔离 clone 的历史 Python 3.8/torch 1.13.1
六组全量复验；A02 prepare、Gate A run/summarize 和训练继续 `NO_GO`。

## 2026-07-15 — C+ 远端历史环境全量复验与证据核验 PASS

完整 bundle 已经唯一传输到 4090 并完成双端验签：

- bundle：`/home/afr/exp374_execution_8ca57ed.bundle`；
- 大小：`22,734,688 B`；
- SHA256：
  `773b4a46f0d7db6a4d8c6f9d894f8961ac2bbf10caa9e4f94c8fa13bc2672696`；
- `git bundle verify/list-heads`：完整历史，唯一目标 head 为
  `8ca57edc2bf7b5db66a0913dad2be2b4078a38d7`。

正式成功前保留了三份 Git 外执行层失败证据，均没有改变 exact 代码：

1. 首个新 clone 在创建 venv/运行测试前被外部 expected SHA 多写两位拦下，标记
   `PRE_SOURCE_SHA_TRANSCRIPTION_FAIL`；
2. v2 的七包 freeze 已通过，但把 `torch.__version__` 与 CUDA 构建信息误拼成
   `1.13.1+cu117`，在测试前标记
   `ENVIRONMENT_VERSION_ASSERT_TRANSCRIPTION_FAIL`。真实字段为
   `torch.__version__=1.13.1`、`torch.version.cuda=11.7`；
3. v3 六组 pytest 均返回 0，但旧 post-audit 误要求源码树不存在 `__pycache__`，并把
   本地 pytest 9 的 subTest 计数硬套到远端 pytest 8。exact commit 本来就跟踪 37 个
   Python 3.7 bytecode；该次标记 `POST_AUDIT_RULE_MISMATCH`，不作为最终证据目录。

三次失败 clone、证据、脚本、PID/log/status 均保留，禁止删除或 resume。v4 使用全新
clone `/home/afr/SOLIDER-REID-exp374-8ca57ed-v4` 和证据目录
`/home/afr/exp374_remote_retest_8ca57ed_v4`，从 bundle 验签开始完整重做全部流程：

- detached exact HEAD、前后 Git clean、9 项 production/tests SHA 全部一致；
- Python 3.8.20、torch 1.13.1、CUDA 11.7、torchvision 0.14.1、timm 1.0.22、
  NumPy 1.24.4、cuDNN 8500；七包 freeze SHA 为
  `7699815505136173aa3f398ac43a0c82fabfa8af9aad2e769b3badaab32cd6c6`；
- formal state/protocol/Swin：`18/18 + 1/1 + 1/1`；
- regression protocol/model seam/runner：`19/19 + 38/38 + 44/44`；
- 远端 pytest 8 的 JUnit 权威总数为 121/121，errors/failures/skipped 均为 0；runner
  另有冻结源码内 5 个同步 `unittest.subTest` row，AST 核验为 44 个 test method、
  1 个 subTest site、5 个 row。它们在远端 JUnit 中不展开，不能误写成 49 个 JUnit
  testcase；本地 pytest 9 的 49 是版本相关的 header 口径，不代表远端少跑；
- exact commit 跟踪的 37 个 `cpython-37.pyc` 在 pre/post 的 path、regular type、mode、
  size、SHA256 与 HEAD blob 全部一致；bytecode manifest SHA 前后均为
  `477199f65d43035dd3d37709968374df7345f4d1fbaa8c6faa1c79bc14115806`；
- 六份远端 JUnit SHA（state/protocol/Swin/protocol/model/runner）：
  `c9da3515a6c4d55c0ef74d79bb610c1ebd5243106dbbf635fa29a1760930a622` /
  `0afae60053c41f23dfeece2ee4534a8508464c436bd1b9739e3bb22017b56145` /
  `3fa2e71c8c26990b2aefb91637df52d3d5f113ead117c2234a3323b773bd1bce` /
  `c3f5c7b661c868d646fee1ed74fb5e706189d34e2dfc6cd820f233c03aa300a1` /
  `9bc7c0de4a07ed678e26e490d3b5e92778f569ca476cf7c19f02c20961cb10dc` /
  `a4db42dcf494184962b1b607a4fb1e34c5a176690659a087fa0b0e54de41eca6`；
- GPU 2 MiB/0%、compute process 为空、无 pytest/Gate-A 进程、无 PREPARED/
  RUN_COMPLETE/FAILED/arms/results，剩余空间约 229.62 GB；
- 远端证据已逐文件回传到 Git 外
  `remote_artifacts/exp374_remote_retest_8ca57ed_v4/`，本地/远端全目录 SHA 对账一致。

主线与独立证据红队均完成重算，独立裁决为
`PASS_FOR_A02_PREPARE_REVIEW_ONLY`。这只允许进入一次全新 A02 prepare-only 的
命令、资源和边界审查；A02 尚未授权或启动，Gate A `run`/`summarize` 与训练仍为
`NO_GO`。

## 2026-07-15 — A02 prepare-only 安全失败：官方 mirror 协议假设过强

A02 wrapper 与 launcher 经两路独立静态复签后，只取得一次 prepare-only 授权：

- wrapper：`16275 B`，SHA
  `7c79818bdecdeb9546939707461d8412cf68ff774e57e721914fb20b9f2feb61`；
- launcher：`3747 B`，SHA
  `806c1a69ebc7b9ef39544c46f80ebaec6f5ad28486a15e5ef7558360494f8fb8`；
- 远端 exact detached HEAD：`8ca57edc2bf7b5db66a0913dad2be2b4078a38d7`；
- 启动前 A02 output/lock/PID/log/status 与两份远端脚本均不存在，GPU `2 MiB/0%`，
  可用空间 `229624778752 B`；
- 唯一 launcher 成功握手 wrapper PID `3138`，命令只有 `prepare`，没有
  `run`、`summarize`、`--resume` 或训练。

prepare 在三个 split 的 deterministic metadata/cache 物化后、任何 matching、checkpoint
forward、arm 或指标生成前，因
`E_SPLIT_CONTENT_OVERLAP: query/gallery/rgb_sha256` 返回 exit code `1`。status finished
UTC 为 `2026-07-15T04:17:14Z`。没有发布 `gate_a_<sha>` execution，也没有
`PREPARED.json`、arms、results、RUN_COMPLETE、COMPLETE 或正式指标。失败后 PID/Gate-A
进程为空、lock 可重新取得、GPU 回到 `2 MiB/0%`。A02 output/staging、外围 lock/PID/
log/status 和部署脚本全部永久保留，禁止 resume、复用或同名重试。

外围失败证据 SHA（lock/PID/log/status）分别为：

- `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；
- `a3fe40281a02fd330fdfd6fdf129ac48c6410f207f3ca3de3024e6e0286e2e39`；
- `22bc27cf2028e5d31682641d48499685d25ead19322c437e8f73d63d47511d0f`；
- `647c5fa40b462864e114d1427fee5e2c14fdc68f8267292cc54025c77883b715`。

staging `FAILED.json` SHA 为
`740c4cabfb3475f35ab4086b5912749e427b4b59dbcca29e6582615db1352bfe`。
三份 metadata 已完整回传到 Git 外
`remote_artifacts/exp374_gate_a_8ca57ed_a02_20260715_failure/`；train/query/gallery 的
大小与 SHA 分别为：

- `8755781 B` / `6e94380939d1ceb2b0f2cb9f3c3cfa0f244b5ebda6afaa3f8233b3b681a189fc`；
- `1209118 B` / `236f7967a5f7c1a28b3dbc0b773268802c75028e67b91f732eaffb8d983b0fb6`；
- `9931008 B` / `5fdcfd64ca55b7ec2c8def03195a24ed89f19917e1b07f9fc3faf145ded6e959`。

主线与三路独立诊断一致确认根因是 `PROTOCOL_ASSUMPTION_BUG`，不是数据污染：

1. 三 split 内 path/RGB SHA/pose-path SHA/pose-content SHA 各自全部唯一；
2. train↔query 与 train↔gallery 的四类交集全部为 0；
3. query↔gallery path 与 pose-path 交集为 0，RGB SHA 与 pose-content SHA 均恰有
   1870 个一对一交集；
4. 1870 对全部同 basename、PID、camid、viewid、person_count、frame，cached Hraw、
   scene score、continuous nuisance 逐值相同，0 forbidden/额外碰撞；
5. 该关系与 exp371 永久证据以及 Occluded-Duke 官方 lists 完全一致；标准 evaluator
   会删除同 PID、同相机的 gallery endpoint。

当前裁决为
`A02_FAILED_NONREPORTABLE / PROTOCOL_ASSUMPTION_BUG / NO_GO_FOR_RUN`。不能简单删除
disjoint 断言，更不能删 gallery 样本或改 evaluator；已先在 `design.md` 冻结结构化、
可哈希、fail-closed 的 official-mirror relation gate，正在做多路静态设计复审。设计
复审通过前禁止实现或测试；随后仍需新 exact commit 的本地/历史环境全量复验与全新
A03 prepare-only 审批。

## 2026-07-15 — A02 official-mirror 全资产只读证据闭合

只读诊断脚本经过多轮移动目标复审后，最终冻结为 `42218 B` / SHA256
`88db86bb09a8d7d6fde7394ba2d12f8b115d517f9609bffe3c40ffd8836c7348`；对应 design SHA 为
`39513400ce65652ca787cafbc69e2fe247716cb6e5e8091e8a9322650b568d68`。两路独立 Codex
均从头签 `PASS_FOR_READONLY_A02_DIAGNOSTIC` 后，脚本才部署到 A02 失败树之外的
`/home/afr/exp374_a02_readonly_mirror_88db86bb.py`，remote size/SHA 与本地逐项一致，权限
`0400`。执行固定使用 Python `-B -s`、`PYTHONDONTWRITEBYTECODE=1`、
`PYTHONNOUSERSITE=1`，只读取 A02 staging 与 resolved data root，不加载 checkpoint、
不运行 GPU forward、不读取/生成 ReID 指标。

同一冻结脚本只读执行两次均 exit `0`，第二次 stdout 原样保存在 Git 外
`remote_artifacts/exp374_a02_readonly_mirror_report_88db86bb.json`，为 `9290 B` / SHA256
`7b070824f86304e9ce4a4fd24e69b0b1c2bda6bea1f24c049c5b844b79553fa2`。两路独立证据
复审均签 `PASS_FOR_A02_ASSET_EVIDENCE`。核心结果：

- status=`PASS_READONLY_A02_DIAGNOSTIC`，共绑定并尾部复核 `86477` 个文件 identity；
- train/query/gallery 的 bundle 与 full/effective constituent split 内 duplicate 全为 `0`，
  `target_outside_effective_count` 也全部为 `0`；
- train↔query/train↔gallery 所有 overlap 均为 `0`；
- q/g 只有 `1870` 组 official RGB/pose-content mirror 和 `3486` 个同 endpoint/同 position
  constituent content mirror，path overlap 与 forbidden overlap 均为 `0`；
- `junk_true/junk_false/forbidden_pair=1870/0/0`；
- RGB/pose endpoint list 均为 `1870` 对、`21666 B`、SHA256
  `4135cdc4bb3cecd52dcf79423cf24d53595ce695a8b91544e2732be4bf3ebdfc`，逐项相等；
- full joint-pair canonical payload 为 `3542413 B` / SHA256
  `b82fd6aa1a81faf85e80b876a62bd892d259e3c7e1e9bb9d9a381641dbb3df93`；
- 六个 q/g cache 全局 shape/dtype/finite、whole-file pre/post SHA 与 identity 全部闭合。

执行后 remote 诊断脚本 SHA 未变；A02 root 仍只含原 staging，没有 PREPARED、
RUN_COMPLETE、COMPLETE、arms/results 或 `__pycache__`；没有 GPU compute process。当前只把
许可升级为 `PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN_REVIEW`：先冻结 production report、
manifest nesting、negative regression matrix 与 runtime 成本，再做多路代码设计审查。
仍禁止实现后直接测试、A03 prepare、Gate A run/summarize、指标生成或训练。

## 2026-07-15 — production relation 设计第四轮：完整 stratum token 契约

对 `fcaecc2cf9b78ae883514194ed32c2b538e07336d3ac6210b10434a748c6ba6f`
的三路从头复审得到两路 PASS、一路精确阻塞。阻塞不是 official mirror 或 metadata
语义，而是 matching 底层契约仍不能识别“同一 person-count stratum 漏掉部分 row”或
“把完整 stratum 拆成多次调用”：旧 token 只有按 global index 排列的 per-record SHA，
无法从未传入的 slot 推回它属于哪个 person-count，因此设计声称的错 subset 必拒绝尚未
闭合。旧版 PASS 不沿用为最终授权。

设计已改为由验证 full records 的 module-private factory 唯一派生并冻结
`strata_global_indices: person_count -> exact ordered tuple[global_index]`。底层
`exact_sparse_candidates` 必须要求传入 indices 与对应完整 tuple 逐项相等，再逐 row
核 local record SHA；omission、superset、reorder、split-call、重复、越界和跨 stratum
全部 fail closed。回归矩阵同步加入这些负例。新候选为 `60679 B` / `970` 行 / SHA256
`c5d43998c8dea7cb76a6163c096d96a8b690fc8a346c89e41f86fef0cab42406`，已重新送三路
从头只读复审。裁决收齐前继续禁止实现、测试、远端操作、A03 prepare、Gate A
run/summarize、arm/per-query 指标和训练。

三路最终复审现已全部收齐，且都只对 exact design
`c5d43998c8dea7cb76a6163c096d96a8b690fc8a346c89e41f86fef0cab42406`
签署 `PASS_FOR_PRODUCTION_RELATION_CODE_DESIGN`。matching 专项确认完整 stratum tuple
能拒绝 omission/superset/reorder/split-call；工程专项确认实现拓扑、调用顺序与 I/O
预算可落地；独立 protocol 专项确认 official mirror、strict-v2、report 三重闭环和六个
runtime callpoint 保持闭合。许可只升级为“可以实现 production relation v2，并在实现后
做独立静态代码审查”；仍不授权执行测试、远端操作、A03 prepare、Gate A run/summarize、
指标或训练。

## 2026-07-15 — relation-v2 实现静态审查与本地全量复验

relation-v2 production、protocol、state machine 与 synthetic negative matrix 已实现。
多轮静态红队先后发现并闭合：pairs 未反向约束 relations、nested schema 可增键、cache
dtype/shape 混码、full/effective 错误码死分支、非 junk 谓词死分支、坏输入原生异常、
A02 入口拒绝后 failure writer 反写、稳定文件 I/O/TOCTOU 分类等问题。最终 production
冻结为：

- audit SHA256：
  `345056f499567ea4f2c9e7cad3daa7a4d9e723939123eb38ebd7334d6a875b39`；
- protocol SHA256：
  `99c72a8a0bb2d26f2173cb2b8d50de281edbb801e33397725f1f20bd6f7af409`；
- runner tests SHA256：
  `a8d26e0d4379647d209a4e88abc2f66b7b9fdef8368156c9e806fa4489407aca`；
- formal state tests SHA256：
  `ee709efc3722e455c28e47fd49b315594e6acf6557035a2c9c6d72ec881aca7b`。

首轮本地执行前五套 PASS，runner 为 `15 failed / 91 passed / 135 subtests passed`。
唯一 production 缺陷是 burned root 常量未先 `resolve()`，macOS tempfile 的
`/var`→`/private/var` 规范化使 exact root/descendant 拒写测试失败；修复后经独立红队
复签。其余失败是测试夹具未同步严格谓词前置顺序或 19-file quick identity 参数，不是
放宽 production；夹具均改为真实违规构造。

最终六套从头重跑结果：formal state `41`、formal protocol `1`、formal Swin `1`、
protocol `31`、model seam `38`、runner `95 + 146 subtests`，即 `207 direct + 146
subtests` 全 PASS；JUnit 总计 `353`，`errors/failures/skipped=0/0/0`。Git 外证据目录：

`remote_artifacts/exp374_local_relation_tests_345056f_rerun1_20260715/`

证据 manifest SHA256：
`0179eedfd98833e042fba6ac95f37ac26601badf6564a7617606f3ecfbb767d3`。两路独立复核确认
命令、9 份源码 SHA、12 份 log/JUnit SHA、环境和计数全部一致，且正式 rerun 后没有新增
pytest cache/bytecode。

当前状态升级为 `PASS_FOR_LOCAL_RELATION_TEST_EVIDENCE`。下一步只允许显式小步提交、
bundle 双端验签及历史 Python 3.8/torch 1.13.1 隔离 clone 全量复验；远端真实资产、A03
prepare、Gate A `run`/`summarize`、arm/per-query 指标和训练继续 `NO_GO`。

提交边界：formal Swin 本地 PASS 绑定了接手前已脏的
`model/backbones/swin_transformer.py`（SHA256
`e0223a1d0fbf1bd6fc9c46a55a35081fd570eab82743577feea425ce31d08c4d`），唯一 diff 为
`.cuda()`→`.to(x.device)`。该文件不混入本轮 9 个 exp374 文件的小步提交；但 HEAD 版本
无法复现 formal CPU seam。因此在这条既有改动的提交归属明确前，bundle/历史环境全量
复验保持 `BLOCKED_BY_SWINT_SEAM_PROVENANCE`，不能用工作树 PASS 冒充 isolated exact
commit PASS。

## 2026-07-15 — Swin seam provenance 已独立闭合

用户将后续技术取舍交由主线决定。为避免把既有工作树改动混入 relation-v2 提交，同时让
本地 formal Swin 证据可以由精确提交复现，已将该兼容修复单独提交：

- commit：`75605b7592785e5e1f043f148b624e75807ba010`；
- 目标文件：仅 `model/backbones/swin_transformer.py`；
- 文件 SHA256：
  `e0223a1d0fbf1bd6fc9c46a55a35081fd570eab82743577feea425ce31d08c4d`；
- 语义：默认 semantic weight 从硬编码 `.cuda()` 改为 `.to(x.device)`；正常单卡 CUDA
  路径保持等价，同时支持 CPU、非默认 CUDA device 与隔离 preflight；
- 其他 `.claude/*`、`CLAUDE.md`、`experiments/decisions.md` 和未跟踪资产均未暂存或提交。

`BLOCKED_BY_SWINT_SEAM_PROVENANCE` 已解除。下一步只授权：把本记录显式小步提交，从新的
exact HEAD 构建/验签 bundle，并在全新隔离 clone 中从头重跑六套测试。isolated clone
353/353 与历史 Python 3.8/torch 1.13.1 复验完成前，远端真实资产、A03 `prepare`、Gate A
`run`/`summarize`、arm/per-query 指标和训练继续 `NO_GO`。

## 2026-07-15 — exact-clone 与远端历史环境 relation-v2 全量复验 PASS

最终 execution source 为
`2b1b17f096aab11ec73f0d1534eb22535ff45412`。完整 bundle：

- 路径：`remote_artifacts/exp374_execution_2b1b17f.bundle`；
- 大小：`22,775,101 B`；
- SHA256：
  `07e2d8ceba224747a471b848b7b40bc525bb2f89080fccb789480d390521538b`；
- 双端 `git bundle verify/list-heads`：完整历史、唯一 head 精确指向
  `2b1b17f…`。

本地全新 detached clone 六套全量重跑为 pytest 9 JUnit `353/353`
（`207 direct + 146 subtests`），0 error/failure/skip；证据 manifest SHA256：
`2d72d4e1c36702becaffde384832ffc9acc9ee35a9de452c3e87244c2b8f00a8`。

远端 launcher 经三路静态红队冻结为 SHA256
`aa885e90a3e110a7a9dba6fb79d45b3e4c39fdda643249bb7eb6cce2d3581f5f`。
4090 的 Python 3.8.20 / torch 1.13.1 / CUDA 11.7 历史环境从头重跑六套：
`41+1+1+31+38+95=207` 个 pytest 8 JUnit case 全 PASS，
`errors/failures/skipped=0/0/0`；原生 unittest 另核
`95 methods + 146 subtests` 全 PASS。六个 rc 均为 0。

回传证据目录：
`remote_artifacts/exp374_remote_retest_2b1b17f_relation_v2/`。其中：

- `evidence_sha256.txt` SHA256：
  `5a326f3f2f13fdc4d58316f82eb6e60de2d299ce49fda38ad22d126d276128a5`，
  30/30 项 PASS；
- `final_status.txt` SHA256：
  `9aa1d54192702a116e062233947b5a394ca213d9bbdabd8a01c05abe7fe5f222`；
- environment freeze SHA256：
  `7699815505136173aa3f398ac43a0c82fabfa8af9aad2e769b3badaab32cd6c6`；
- 10 项源码和 37 项 tracked bytecode pre/post 完全相同；
- post audit：detached exact HEAD、Git clean、无 data/pytest cache/Gate A marker、
  无 GPU compute process；
- 三路独立证据红队均 PASS；本地复核记录 SHA256：
  `97ec7b9a0109307e784125c74ebe69d7e109a151dfee9ffc22e94ce816c72fed`。

链路异常边界：首次 scp 中断形成的 `2,611,200 B` 错误 partial 以 SHA
`1bbc7afaa78985c484bdb9ff854cbeeb1d6fc1f899aa782fda837529434d0968`
单独归档；正式路径只在 incoming 完整大小与全文件 SHA 均通过后原子替换，故未污染复验。

当前状态为 `PASS_FOR_A03_PREPARE_COMMAND_REVIEW_ONLY`。只允许设计、冻结并多路红队
全新 A03 prepare-only 命令；A01/A02 永久烧毁，A03 `prepare`、Gate A
`run`/`summarize`、arm/per-query 指标和训练仍为 `NO_GO`。
