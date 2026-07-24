# exp416 PC-NEC监控

## 2026-07-24：设计建立

- exp415唯一512 oracle已永久封板为`ASSET NO-GO / FORMAL E120 NO-START`；
- 三路独立复盘将科学失败、测量失败、归因失败与用户终止严格分开；
- 唯一首选切换为PC-NEC，但当前只允许设计一次固定共同candidate bank的无训练fuel audit；
- 未实现runner、未连接远端GPU、未创建asset namespace、未启动PK64或e120；
- 下一步只做子agent致命bug、变量混淆与exp395--415旧机制同构复审。

当前=`DESIGN ACTIVE / SUBAGENT REVIEW NEXT / FUEL AUDIT NO-START / TRAINING NO-START`。

## 2026-07-24：三路设计复审完成

- 最终design SHA256=
  `d118fe038c96b067ea139c727980bc26529fe48c2131ebeabe3f817869619e67`；
- 统计/证据路首轮发现GO指标层级/组合器与PID-bootstrap不唯一；已冻结PID-macro唯一主指标、
  mid-rank组合、lambda grid、OOF五折选择/tie-break、逐指标最强control、query-PID cluster bootstrap、
  PCG64 seed/salt、10,000次与线性5%单侧下界，回归=`PASS / 0B / 0H`；
- 机制/变量路首轮阻断future certificate可退化为hard-pair/part metric/relation KD，并指出wrong donor、
  D0 student-part、canonical-location nuisance和neither别名不完整；现已冻结全负身份C/U set-LSE数学合同、
  genuine/gradient owner、donor排除、`featmaps[-1]` pooling与exact area/availability，最终=
  `PASS / 0B / 0H / 0 old-isomorphism`；
- 公开近邻路确认KPR/BPBreID、PAT-CSL、Instruct-ReID和普通part-aware metric均为强近邻，但当前整体窄差分
  未与其或exp395--415旧机制同构，结论=`PASS / 0B / 0H / 0 old-isomorphism`；
- 所有复审均只读，未修改文件、未连接远端或GPU。

该PASS只授权实现并复审无训练fuel audit；仍不授权任何certificate训练、PK64或e120。

当前=`DESIGN REVIEW PASS / FUEL RUNNER IMPLEMENTATION NEXT / FUEL AUDIT NO-START /
TRAINING NO-START`。

## 2026-07-24：fuel runner实现进行中

- 用户明确要求开始，授权范围仍严格限定为一次无训练fuel audit；没有启动PK64、optimizer或e120；
- 本地首先复核真实git状态，受保护未跟踪文件
  `experiments/exp411_pose_complete_multi_positive_set_ranking/创新性判断.md`
  保持未修改、未删除、未暂存；
- 远端只读核验GPU无compute进程、无exp416 formal目录、无训练进程；
- `fuel_io.py`首轮known-answer SHA自测按预期拦截三个错误预填常量；已从实际canonical JSON/array/order字节重算，
  修正后`EXP416_FUEL_IO_SELF_TEST=PASS`；
- 子agent完成纯NumPy `fuel_core.py`，覆盖train-only枚举、candidate bank、wrong donor、九臂能量、
  tie-aware AUROC/AUPRC、mid-rank、五折OOF、PID-macro与10,000次PID-bootstrap，主进程复测
  `exp416-pcnec-fuel-core-v1=PASS`；
- 子agent完成新的真实rectangle OpenCLIP encoder和sealed D0 global/`featmaps[-1]` extractor；两者CPU mock
  self-test与主进程语法复测均PASS，不复用exp411 region-isolated cache或exp415 selector；
- 主进程实现pose/RGB-only `geometry_census.py`，在任何CLIP前冻结：
  score `>=0.30`、每槽至少2 joint、span q90、每侧5% padding、最小`16×16`、canonical center中位数、
  query coverage 80%、每槽100,000 pair与300 query PID；self-test PASS；
- 主进程实现CPU-only `fuel_audit.py`初版，固定四指标、逐指标最强control、六个bootstrap、全部GO门与原子封存；
  failure injection先发现“最强arm可伪造为correct”未被拒绝，已加入control membership/value exact校验，
  回归`EXP416_FUEL_AUDIT_SELF_TEST=PASS`；
- 新增`protocol.md`冻结四个物理阶段、只读资产、fresh namespace、几何门和唯一裁决；
- 当前本地提交`5585e442`只包含五个已复测核心/提取/几何文件；candidate builder与stage3 cache builder仍由两个
  子agent独立实现，尚未创建formal或运行任何真实资产。

当前=`FUEL IMPLEMENTATION ACTIVE / LOCAL SELF-TESTS PASS / FINAL IMPLEMENTATION REVIEW PENDING /
FUEL AUDIT NO-START / TRAINING NO-START`。

## 2026-07-24：20次独立复盘与实现阻断修复

- 用户明确停止exp414；correct/pose-only/q-only保持永久封存，text-shuffle为
  `USER-DIRECTED STOP / VOID / NO RESUME`，all-edges=`NO-START`；
- 三名新子agent独立完成20次失败谱系、近邻/候选搜索与PC-NEC数学复审；一致结论为：
  PC-NEC是唯一值得先做固定fuel kill-switch的候选，PS-ODM/APCO/MH-PSO只保留条件备选；
- 证据复核收紧：exp409不是同PID交换对称实验；exp413 q-only未e120；exp414 q-only仍使用pose-defined
  region cache；CAVT只能记科学未评估/工程路线关闭；
- 实现首轮复审发现`2B/3H`，包括geometry门未接stage3、audit namespace过晚、artifact未交叉绑定、
  microbatch链式判断与provenance/物理封存不足；
- 已完成全部修复，并把非D0 bootstrap改为replicate内七control最小paired差；
- future certificate原式被阻断：`-LSE_U`会吸引未决负身份；现只在设计中修为stop-gradient U、
  normalized logmeanexp与same-view deterministic certificate branch，训练仍未授权；
- 当前design SHA256=
  `b0c4301c3bf34b2a59f329d0b31c261e23f58cb7b12b1624c7ffc96c6a03eed0`；
- 本地综合static contract、candidate、geometry、D0/CLIP串行mock、energy、OOF/bootstrap及failure
  injection全部PASS；`git diff --check`为0；
- 最终回归发现并修复candidate camera-label捷径：bank现按genuine candidate-camera频数分层匹配impostor
  quota，极端`1:100`频数self-test已覆盖；任一stratum不足在pose/CLIP前直接INVALID；
- 科学主张已收窄为`OpenCLIP image encoder region visual evidence`，不读取text encoder，不声称语言语义
  不可替代；future threshold门冻结为identity-level genuine误证`<=1%`、负身份PID-macro coverage
  `>=30%`、`C/U`同时非空anchor `>=80%`；
- 远端首次复核SSH返回255且无输出，按基础设施连接异常分类；随后连接恢复，旧exp414 PIDs均不存在、
  `nvidia-smi`无compute进程、exp416 formal/asset namespace均不存在；未启动、恢复或重跑任何训练。

当前=`LOCAL IMPLEMENTATION PASS / FINAL SUBAGENT REGRESSION ACTIVE / REMOTE FORMAL NO-START /
TRAINING NO-START`。

## 2026-07-24：效能红队阻断D0 fuel到训练的外推

三名最终独立裁决收敛为：

- 现有D0 signal runner代码/统计=`DIAGNOSTIC PASS / 0 implementation blocker`；
- future训练因果链=`BLOCKED / 2B / 2H`；
- 唯一保留候选仍为修订PC-NEC，但只允许先设计consumer-aligned无训练否决门；
- PS-ODM/APCO不推进：前者大概率只能证明普通真实遮挡增强，后者近乎adversarial erasing且“攻击更难”会自证。

两个BLOCKER是：

1. 现有diagnostic由sealed D0消费，future宿主却是zero-owner；过去最稳定事实正是zero-owner吞掉新增残差，
   因而D0候选内`+1 mAP`不能预测训练收益；
2. top-20 image bank没有定义真实PK64全部`64×64` pair的`E/v`，原`|C|/(P-1)`中的`P`也未闭合。

design现冻结下一唯一前置门：sealed zero-owner残余误排序富集、raw-PID OOF identity误证/coverage、
确定性PK64全pair `100%`证书来源，以及exact `L_cert`梯度对独立全排序梯度的归一化对齐。禁止扫epsilon、
loss权重、temperature、margin或threshold。任一失败即
`PC-NEC TRAINING NO-START / NO CANDIDATE`。

为了防止旧D0 runner被误启动，`static_contract.py --formal`现显式硬阻断，直到consumer-aligned gate完成实现与
复审。最新design SHA256=
`6d062b2782abb1bd5c9fa36a1a7500ff3105c203d70e5e9e10c6d8e03622ef77`。

当前=`D0 SIGNAL DIAGNOSTIC IMPLEMENTATION PASS / CONSUMER-ALIGNED GATE NOT IMPLEMENTED /
FORMAL NO-START / TRAINING NO-START`。
