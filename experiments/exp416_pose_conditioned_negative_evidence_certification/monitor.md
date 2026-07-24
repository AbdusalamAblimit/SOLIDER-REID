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
