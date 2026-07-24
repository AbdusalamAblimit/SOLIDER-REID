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
