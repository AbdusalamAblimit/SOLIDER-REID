# exp374 PSG 图像—姿态对应依赖门禁 — 监控记录

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
