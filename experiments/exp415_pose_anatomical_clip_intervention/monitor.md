# exp415 PACIT监控

## 2026-07-24：设计建立

- exp414 text-shuffle已按用户指令终止并作废；遗留8个PPID 1 DataLoader worker已在用户重新授权开始工作后
  逐一TERM，进程与CUDA context全部消失，4090=`2 MiB / 0% / 0 compute PID`；
- exp415冻结为PACIT输入空间反事实方向；不继承owner、prefix、MST、continuous-region、proxy或gradient
  routing。若asset GO，production所有臂共同使用zero-owner的普通等支持全身份listwise强宿主，但它不参与
  pose×CLIP归因；
- 当前只授权设计、实现、静态合同与子agent复审；512图oracle尚未启动，正式训练与PK64均`NO-START`；
- 受保护文件`experiments/exp411_pose_complete_multi_positive_set_ranking/创新性判断.md`保持未跟踪且不修改、
  不删除、不暂存。

当前=`DESIGN ACTIVE / ORACLE NO-START / FORMAL TRAINING NO-START`。

## 2026-07-24：第一版复审BLOCK与revision-2静态实现

- 两个独立子agent均判第一版`BLOCK`：
  1. CLIP选择属性/候选后又用同一margin验证，构成自证；
  2. control全部继承pose候选池，没有真正CLIP-only；
  3. arm难度、固定分母、same-view与batch64双forward合同未闭合；
- 第一版从未运行oracle、未连接CUDA、未生成checkpoint；
- revision-2改为`pose/fixed proposal × CLIP/hash selector`严格2×2，P+/P-各35个proposal且index/面积/aspect
  一一对应；
- fixed proposal generator已拆成不接受pose参数的独立调用图；CLIP selector函数不接受pose/slot/D0；
- blind evaluator只接收RGB、mask、pose field与D0 boolean gate，不接收CLIP分数或标签；
- 使用工作区`.venv`和`uv run`完成py_compile与纯CPU synthetic contract：
  `EXP415_STATIC_CONTRACT=PASS`；
- synthetic检查覆盖：512 hash顺序不变性、35 proposal索引、mask外byte-exact、achromatic fill不落颜色桶、
  CLIP/hash tie-break、frequency histogram exact、CIELAB blind正例、factorial bootstrap与难度统计；
- 复审已再次并行提交给三个子agent。复审清零前不冻结formal、不做geometry census/runtime smoke、不启动唯一oracle。

当前=`REVISION-2 RE-REVIEW PENDING / GPU IDLE / ORACLE NO-START / E120 NO-START`。

## 2026-07-24：revision-2复审BLOCK，revision-3重构

三路复审一致确认revision-2已打破exp401--414旧机制同构，但仍`BLOCK`。关键问题：

1. P+ invalid anchor bit进入Y，而P-相同fallback mask没有该bit，污染P变量；
2. scorer只接收外部score矩阵，无法证明外部没混入pose；prompt margin也未定义；
3. 统计函数允许4条toy数组，不硬断言512；
4. blind颜色缺连通性，raw-color/D0-hard未受同一severity/identity caliper；
5. 只做总体难度统计而非逐图匹配；
6. canonical资产经过crop/Random Erasing后不保证actual-view事实；
7. 512到全15,618失败图与四臂共同NOOP未定义；
8. zero-owner宿主、double-clean顺序和多seed门存在文字矛盾。

revision-3已完成当前本地静态重构：

- oracle row index固定`mod 5`平衡解剖层；P+/P-均只在同层7个shape选择；
- 删除arm-specific anchor-valid Y门；
- 增加源码内centered-color-margin scorer，签名不接受pose/slot/D0；
- C=0改为同图area/centroid/aspect/D0 displacement/D0 CE/top5 caliper内hash；
- quartet任一match失败时四臂共同Y=0，固定分母仍为512；
- blind evaluator加入最大4连通颜色分量，不再声称服饰归属；
- D0-hard与raw-color改为同一identity/severity-caliper强control；
- 统计与row accumulator硬断言四臂各512且row id同序；故障注入row保留Y=0；
- full-asset任一arm失败时四臂共同clean-NOOP、不drop sample；
- production关闭Random Erasing，crop有四mask共同80% survival门；
- production共同宿主统一为zero-owner；先跑double-view clean-pair，再跑correct；
- 性能与seed/paired-bootstrap阈值已量化。

本地`uv run`重新执行py_compile与扩展static contract，结果：
`EXP415_STATIC_CONTRACT_V3=PASS`。覆盖真实6%面积、synthetic pose/fixed pools、7×10 scorer、caliper、连通颜色、
identity gate、strong controls、512 row故障注入、短数组拒绝与全臂NOOP。

当前=`REVISION-3 STATIC PASS / SUBAGENT RE-REVIEW REQUIRED / GPU 2 MiB 0% / ORACLE NO-START / E120 NO-START`。

### revision-3第二轮聚焦修复

聚焦复审继续发现：factorial少`P-only↔neither`直接边、strong control排除correct同mask会强迫次优、实际
whole-image CLIP encode调用图未落盘、共同NOOP未覆盖strong controls，且crop后只看mask面积。

当前已修复：

- quartet要求四条直接caliper边，任何一边失败四臂共同0；
- strong eligible允许P+C自身mask；若raw/D0选择同mask即如实机制等价；
- 增加P+C-vs-raw与P+C-vs-D0原子pair accumulator，任一侧失败两侧共同0；
- full common bitmap覆盖四factorial+两strong control；任一strong失败六臂共同clean-NOOP；
- 删除formal text-shuffle训练臂，因label循环不改变7×10全局argmax mask；只保留固定512 agreement诊断；
- 新增`clip_color_selector.py`：checkpoint SHA、标准whole-image letterbox、OpenCLIP image/text encode、template
  ensemble和centered margin调用图全部冻结，public call仅接original RGB与7 edited RGB；
- crop后同步变换RGB/mask/pose field并重跑非学习blind anatomy+color；任一六臂失败共同NOOP；
- static新增actual selector源码禁用词检查、四边match、单臂故障传播、strong同mask等价、strong identity交集与
  单strong失败六臂NOOP。再次返回`EXP415_STATIC_CONTRACT_V3=PASS`。

仍等待三路新复审，未连接远端CUDA。

### final pre-formal contract correction

- 复审发现design冻结hash salt与实现拼接不同；现统一为
  `exp415-v3-caliper-blind\0path\0candidate_index`，并加入固定known-vector SHA断言；
- 为彻底关闭canonical→actual severity漂移，production不再允许crop/padding/Random Erasing；全量builder必须
  预验证canonical与hflip两种方向上六臂的blind、D0、ROA与全部caliper，训练只在两个已封存方向中选择；
- blind anatomy新增“覆盖最大槽必须等于预定active anchor”，防止事后换槽通过；
- 当前已有两路聚焦复审给出`PASS / 0B / 0H / 0 old-isomorphism`；第三路上述两项BLOCK已按原文修复，
  需其回归确认后才进入远端机械门。

第三路回归现已确认hash known-vector与canonical/hflip双方向合同，最终三路一致：
`PASS / 0B / 0H / 0 old-isomorphism`。设计审查阶段完成，只授权formal前机械门；oracle仍NO-START。

## 2026-07-24：formal前runner实现与复审

- 新增全15,618 `geometry_census.py`：严格复用`PoseTargetStore.valid`，没有score confidence threshold；
  逐图核验official/pose path coverage、RGB SHA与尺寸，只统计5槽anchor-valid、P+/P-各35 proposal、
  active7及canonical/hflip面积可达性；不加载CLIP/D0、不算blind/Y；
- 新增固定8图`runtime_smoke.py`：唯一输出namespace锁死为
  `/home/afr/reid-clean/assets/exp415-pacit-smoke-v3`，代码不接收或读取oracle路径；
- smoke以同一whole-image OpenCLIP实例检查P+与canonical-anchor各`[7,10]`输出；D0固定检查
  `clean+pose7+canonical-anchor7+ROA8=23`个变体的descriptor/logit/true-PID CE/top5/displacement；
  只判shape/dtype/finite，不选proposal、不写raw score/Y/rate/GO；
- smoke cache/result只保存路径、mask/tensor SHA与机械shape，原子写入并exact回读；失败保留
  `failure.json`且禁止续跑；
- 三路实现复审最终一致：
  `PASS / 0B / 0H / 0 variable-confusion / 0 old-isomorphism`；
- 本地`.venv`下两个`py_compile`与两个`--self-test`均PASS；未连接远端GPU；
- 远端固定D0/CLIP/pose manifest SHA已先验精确核验，formal tracked=`0/0`、GPU=`2 MiB/0%`、
  smoke/oracle namespace均不存在。

当前=`PREFORMAL RUNNERS REVIEW PASS / REMOTE CPU CENSUS NEXT / ORACLE NO-START / E120 NO-START`。

## 2026-07-24：全15,618 geometry census完成

- frozen formal执行HEAD=`3dcbe378303c68c1639bd60fa794e35835b461ca`；
- 固定解释器、`CUDA_VISIBLE_DEVICES=""`，自然完成`15618/15618`，`GEOMETRY_CENSUS_EXIT=0`；
- official/pose path-set、pose manifest/shard/records digest、逐图RGB SHA与尺寸全部exact，严格异常0；
- `pose_confidence_threshold=None`，没有新增score阈值；`clip_loaded=false`、`scientific_y_computed=false`、
  `cuda_used=false`；
- 五槽全局valid：
  `head=15616/15618`、`upper_torso_arms=15618/15618`、`lower_torso=15618/15618`、
  `upper_legs=15618/15618`、`lower_legs_feet=15586/15618`；
- 预定active assignment valid：
  `3216/3217, 3062/3062, 3106/3106, 3117/3117, 3109/3116`；
  `15584/15618`图五槽全valid，`15618/15618`至少一槽valid；无效anchor全部使用fixed fallback；
- P+与canonical-anchor在canonical/hflip四组均为`546630/546630` proposal面积可达，
  area=`2950--2952` pixels、最大相对误差`0.001018 < 0.01`；
- result SHA256=`82dd0f72af71ad03bda3cb11f471ad6651fcba10a35d26e0c036a61fd5352e8f`；
  runner SHA256=`9e66c63c5450ff2600c0a5ad82aa5c7651e3d182bb0f7ed33ca0f073dc5f4bf5`；
- 结束后formal tracked/index/full status均清理回`0/0/0`，GPU=`2 MiB/0%/0 CUDA PID`；
  smoke namespace仍fresh，oracle namespace未创建。

当前=`GEOMETRY CENSUS PASS / RUNTIME SMOKE NEXT / ORACLE NO-START / E120 NO-START`。

## 2026-07-24：固定8图runtime smoke完成

- frozen formal执行HEAD=`bbc23fe096d73bd203c455150a96975e52558e7e`，formal
  tracked/index/full status=`0/0/0`；
- smoke唯一namespace
  `/home/afr/reid-clean/assets/exp415-pacit-smoke-v3`自然完成，`RUNTIME_SMOKE_EXIT=0`，
  严格异常0；该namespace已永久封存，禁止删除、覆盖或重跑；
- 固定8图逐图得到P+与canonical-anchor各7个CLIP编辑，CLIP输出均为finite
  `[7,10]`；没有选择winner、没有计算Y/agreement/rate/GO；
- D0变体严格为`clean + pose7 + fixed7 + ROA8 = 23/图`，合计184：
  descriptor=`[184,768]`、logits=`[184,702]`、CE=`[184]`、top5=`[8,23]`、
  displacement与CE-change均为`[8,22]`，全部finite；
- official train、pose/RGB SHA、active7、mask外byte-exact、cache exact-readback均PASS；
  optimizer update=`0`、checkpoint write=`0`，smoke代码未触碰formal oracle namespace；
- result SHA256=
  `154809de3ddceafc85efb5db7d8403fb463f75ba1c35e9f50c0416192ad155a8`；
  smoke cache SHA256=
  `17c05f3a4741750e7e469f3ba56561e1c5dbb5fded09c969f80353280db9e725`；
  runner SHA256=
  `d5900ff8ba7ec359a0d43745c9265650f6e3bc1329c67138dae85240cc63d674`；
- 结束后4090=`2 MiB / 0% / 0 CUDA PID`，正式oracle namespace
  `/home/afr/reid-clean/assets/exp415-pacit-oracle-v3`仍不存在。

当前=`RUNTIME SMOKE PASS / FULL 512 ORACLE RUNNER IMPLEMENTATION NEXT /
ORACLE NO-START / E120 NO-START`。

## 2026-07-24：oracle前统计与变量合同加固

完整512 runner实现前的两路独立只读复审发现并阻断了三个会改变正式Y或匹配分母的问题；均发生在尚未创建
oracle namespace、尚未计算任何oracle科学结果时：

1. blind evaluator的颜色排序只把presence/capture/purity/component纳入“最弱项最大化”，漏掉
   absolute/relative drop；现已把七个冻结颜色门全部显式纳入rank，并补“空间五项更强但drop失败不得胜过
   七项全过颜色”的反例；
2. caliper helper只显式检查clean与candidate top-5，漏掉reference edited top-5；现已把
   `reference_top5`作为独立参数接入候选内匹配和四条direct pair，禁止runner暗中合并变量；
3. strong-control helper只检查candidate identity-safe，未显式要求P+C reference自身位于同图ROA
   P50--P90；现已加入`reference_identity_safe`，reference失败时整条eligible为false。

同时完成：

- 新增冻结`direct_pair_caliper`，统一area/aspect/centroid/D0 displacement/D0 CE、clean/reference/candidate
  top-5与mask-difference语义；
- `match_edges / arm_complete / pair_match_ok`只接受严格bool，拒绝字符串与NaN truthiness；
- formal paired bootstrap硬锁10,000次，短数组与非固定次数均拒绝；
- 补跨P不比较centroid、同mask允许、任一top-5失败、非finite、reference identity失败与错误bool的
  failure-injection；`.venv`下`py_compile`与`EXP415_STATIC_CONTRACT_V3=PASS`。

已封存runtime smoke不调用blind evaluator、caliper、strong selector、Y或bootstrap，因此上述修复不改变其
机械PASS与SHA。正式oracle namespace仍不存在。

当前=`PREFORMAL CORE FIXED / FULL 512 ORACLE RUNNER IMPLEMENTATION ACTIVE /
ORACLE NO-START / E120 NO-START`。

## 2026-07-24：完整512 oracle runner封板

- 新增完整正式runner `asset_oracle.py`，固定512行、每行
  `clean + pose7 + fixed7 + ROA8 = 23`个D0变体，并完整保存P+/P-各35 proposal、active各7、
  两个`[7,10]` CLIP池和14个blind候选；
- 四factorial臂、两条pair-specific strong control、四条direct caliper边、五个factorial effect、
  10,000次paired bootstrap、agreement与top-5门均进入同一固定分母裁决；科学Y=0保留为matched row，
  不再污染`arm_complete`；
- runner只允许固定解释器、`seed=1234`、`PYTHONDONTWRITEBYTECODE=1`、
  `PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并强制deterministic algorithms、
  cuDNN deterministic、TF32关闭及状态回读；
- runner/core/selector/prompt SHA256分别为：
  - `b9083a6dd4923e0eec6c1b4f29e67813fc352b937d9a46363ff9b8583f7d836a`；
  - `15ad21a7a79dc59819cee61a2971bf334f2683bb1ea77d71e0d6b155c3020311`；
  - `4b10a9899c203e51e67fed9dbe119d8f52150c60b8e41ffef9c68fc366bf78a9`；
  - `1fb55c6ca451e132084293c9c583cbcab4ee3e45b993a6f7fbaf672dd99e60bd`；
- 本地工作区`.venv`下重新执行`asset_oracle.py --self-test`与`static_contract.py`，分别得到
  `EXP415_ASSET_ORACLE_SELF_TEST=PASS`和`EXP415_STATIC_CONTRACT_V3=PASS`；`git diff --check`通过；
- 三路最终只读复审分别覆盖确定性/provenance、机制/变量旧同构、固定分母/统计/schema，全部返回：
  `PASS / 0B / 0H`，其中机制路额外为`0 variable-confusion / 0 old-isomorphism`；
- 启动前第一次远端只读门：formal HEAD=`d1c9f7e603bae1f38e3456ae80ce8859ca0dfc3b`，
  tracked/index/full=`0/0/0`，oracle namespace=`ABSENT`，4090=`2 MiB/0%/0 CUDA PID`；
  该HEAD尚未包含最终runner，下一步仅允许显式同步并提交本轮目标文件，再执行第二次preseal门。

当前=`FINAL RUNNER REVIEW PASS / FORMAL SYNC NEXT / ORACLE NO-START / E120 NO-START`。
