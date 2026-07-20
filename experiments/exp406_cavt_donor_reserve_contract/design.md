# 实验 exp406：CAVT P0B donor-reserve合同修复

## 动机

exp405唯一`exp405-p0b-preflight-v1`已取得started seal并永久封板为
`MEASUREMENT CANDIDATE-POOL CONTRACT FAILURE / SCIENCE NOT EVALUATED`。它完成512图真实MMPOSE-ABU
CLIP original编码，但在同一512子池内选择20个recipient后，剩余候选不足以为所有recipient找到满足正式
四变量MAD、caliper `8.0`、same-camera/different-PID和全局一对一约束的wrong-mask donor。

该失败不能外推到formal的15,618图donor universe，也不能解释为MMPOSE-ABU、CLIP或CAVT科学NO-GO。
exp406不是重跑或修改exp405，而是独立新编号的测量合同修复；exp405源码、输出、seal、failure receipt和结论
全部只读保留。

## 核心假设

失败根因是preflight把固定512图同时当recipient基池和完整donor universe，而不是匹配门本身错误。保持原512
基池、20 recipients、四变量MAD尺度、caliper `8.0`及全部身份/相机/唯一性门不变，仅按预注册顺序单调扩展
donor-only前缀，应能在不降低任何门槛的前提下裁决真实wrong-mask接线。

若扩展到full train仍不存在完整匹配，exp406必须自然FAIL；不得fallback、放宽caliper或改写尺度。

## 技术方案

### 1. 全新命名空间与fresh边界

- 实验目录：`experiments/exp406_cavt_donor_reserve_contract/`
- preflight execution：`exp406-p0b-preflight-v1`
- preflight output：`/home/afr/reid-clean/audits/exp406-p0b-preflight-v1`
- formal execution：`exp406-p0b-iso-teacher-v1`
- formal output：`/home/afr/reid-clean/audits/exp406-p0b-iso-teacher-v1`
- fresh asset root：`/home/afr/reid-clean/assets/exp406-p0b-preflight-v1`

不得读取exp405 output、cache、pair map、MAD、receipt或path mapping。official RGB仍只读
`/mnt1/afrdata`，pose仍只读冻结`/mnt1/afrderived` artifact；这两者是共享只读数据，不伪装为fresh资产。

### 2. 冻结core基池与recipient

从official有序train manifest重新计算原512图hash基池，不读取exp405缓存。五槽仍各取4个recipient，共20个；
recipient选择、sample key、pose/CLIP readout和analysis-valid定义与exp405冻结代码一致。

四个primary matching量保持：

1. `log(mask mass)`；
2. `y-centroid`；
3. pose confidence；
4. CLIP support。

每槽median absolute deviation尺度只用原512 core中该槽analysis-valid样本计算一次并冻结；扩展donor不得进入
recipient选择或重算尺度。global CLIP cosine继续只参与原有排序，不替代primary caliper。

### 3. 固定尺度、单调扩池

core外的official train样本只作为donor。先按camera分组；每组用
`SHA256("exp406-donor" || camera || PID || relative_path)`排序，再按camera升序round-robin形成唯一全局顺序。
冻结pool总规模阶段：

`512 -> 1024 -> 2048 -> 4096 -> 8192 -> 15618(full train)`。

每个新增样本最多做一次original pose/CLIP编码，结果只留在本次preflight内；不得写成formal可复用cache。每个阶段
都从当前累积pool重新执行完整匹配，且必须同时满足：

- target slot `analysis-valid`；
- same-camera；
- different-PID；
- 20个recipient全局排除；
- donor全局唯一、不复用；
- 固定四变量MAD与caliper `8.0`；
- 原`64 -> 128 -> 256 -> full-caliper`偏好扩展；
- 确定性一对一增广匹配。

只有当前阶段存在完整匹配才执行20对wrong-mask CUDA重编码并自然完成；否则进入下一冻结前缀。由于尺度、阈值、
recipient和旧donor都不变，feasible donor集合只能单调增加。full train仍无完整解即immutable FAIL，零fallback。

每阶段必须在结果或failure receipt中记录每槽四轴median/MAD、candidate与caliper degree分布、零边recipient、
每个recipient最近primary distance、Hall/assignment失败类型、pool规模和最终preference扩展层级。不得只留下笼统
“no donor”错误，也不得据这些诊断事后改变前缀、尺度或recipient。

### 4. Formal保持科学合同不变

exp406 formal不得复用preflight scale、feature、pair map或cache。它仍独立编码全15,618 train图，重新计算full-train
MAD，五槽各400 recipient，caliper固定`8.0`，`64 -> 128 -> 256 -> full`一对一、recipient全排除、donor零复用，
全部teacher反事实、coverage、bootstrap、non-torso和科学门保持exp405冻结投影。唯一允许变化是execution namespace
和preflight donor-pool plumbing。

## 对照组

### CPU/static正合同

构造512 core中无合格donor、扩展前缀中存在合格donor的synthetic case；exp406必须在预期阶段完成匹配，同时
保持recipient、尺度、caliper和旧候选排序exact。

### 强反合同

必须抓住以下mutant：

1. 仍只用512 subset的旧合同；
2. caliper大于`8.0`或跳过caliper；
3. 扩池后重算MAD；
4. 扩展样本参与recipient选择；
5. same-camera或different-PID失效；
6. recipient成为donor；
7. donor复用；
8. 非冻结前缀、结果驱动改序或无限扩展；
9. preflight cache进入formal；
10. exp405 receipt/cache/output能够授权exp406；
11. formal科学常量、controls或阈值相对exp405冻结投影发生变化；
12. fresh asset缺失、字节变化、旧路径或runtime前后不一致。
13. 某轴MAD为零/近零时scale floor或距离语义漂移；
14. 每个recipient都有edge但全局Hall条件无解；
15. failure receipt缺少slot/尺度/最近距离/stage诊断。

## 预期结果

- 正常路径：在有限前缀内取得完整匹配，512 core与20 recipients不变，20对wrong-mask机械接线全部finite/active，
  `scientific_evaluated=false`，生成唯一COMPLETE PASS。
- 最坏路径：增量编码full train后仍无完整匹配，写权威failure receipt并永久FAIL。
- 成本上界：original编码15,618图，约为exp405 512图的`30.5x`；diagnostic仍仅20对，不执行formal 2,000对科学测量。

preflight PASS只授权创建fresh formal manifest，不直接授权formal执行。formal必须另行复核源码/runtime/assets/GPU、
COMPLETE provenance和全部fresh门。

## 风险与失败解释

1. **full train仍无完整匹配**：说明冻结core尺度与正式约束不相容，exp406机械合同FAIL；不得放宽门。
2. **成本接近formal original pass**：接受该上界，因为它换取有限终止与不降门；不得因耗时改成抽样挑池。
3. **扩池改变recipient或尺度**：属于实现BLOCKER，不得启动GPU。
4. **formal分支漂移**：任何科学常量或control变化均BLOCKER；exp406只修preflight候选池。
5. **只过preflight但formal科学失败**：如实判CAVT P0B NO-GO，不启动transport/student。
6. **未来进入student**：仍须e10/20/.../120逐点记录方法与clean D0同epochmAP/R1和差值，最终只看自然e120。

## 启动门

1. design/protocol/monitor冻结并提交；
2. 新代码默认路径与formal冻结投影静态exact；
3. CPU/static正反合同连续两次byte-exact PASS；
4. 代码、复现/once-only、统计/matching三路盲审均`BLOCKER=0 / HIGH=0`；
5. fresh exp406资产、远端隔离仓库、只读数据/pose、唯一4090和fresh output复核；
6. 才允许唯一exp406 preflight。

任何门未过时，GPU、formal、transport和student均`NO-START`。
