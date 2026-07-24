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
