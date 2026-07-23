# exp413 PSCCR 监控记录

## 2026-07-22：设计与查新完成，GPU保持空闲

exp412已自然封板为`PERFORMANCE NO-GO`，远端GPU=`0% / 2 MiB`且无compute进程。exp413当前仅完成聚焦论文/代码
审计与设计冻结，尚未实现、未运行CPU/CUDA合同、未创建formal/output，也没有mAP/R1。

在线Google、arXiv与Semantic Scholar在当前网络环境均超时，已在审计中明确限制；本地截至2026年的已审计语料未
发现与“pose×identity-free CLIP无丢弃coverage chain + 逐前缀all-identity ranking”完整同构的方法，但generic
greedy coverage、prefix curriculum、listwise与multi-positive均不得作为新颖原子。当前创新状态=
`C-CLASS CONDITIONAL / DESIGN ALLOWED`。

冻结执行顺序：独立盲审只拦截致命bug/变量混淆；随后实现默认关闭机制并做静态检查；只执行一次真实PK64 CUDA/AMP
合同。合同PASS后立即启动唯一fresh correct seed1234/e120，中间点只记录不早停。当前判断=`CONTINUE DESIGN REVIEW`，
原因是GPU空闲且新对象尚未越过一次性机械门。

首轮独立盲审=`1B/2H`，已在实现前全部修正文档：序数改为每个LOO三support内重算，禁止被排除图/anchor evidence
泄漏；invalid与三control规则改为严格单变量；同一个唯一runner加入手算micro-oracle与excluded-image mutation
invariance。同一审查者复核=`0B/0H`，当前=`DESIGN REVIEW PASS / IMPLEMENTATION AUTHORIZED / CUDA NO-START`。

默认关闭实现、formal config与唯一合同runner已完成静态AST/config exact检查；独立代码盲审=`0B/0H`，确认生产接线与
合同无致命bug/变量混淆。当前只授权把冻结提交传到fresh remote formal并执行一次真实PK64合同；尚未传输、未使用GPU。

## 2026-07-22：唯一真实PK64合同PASS

fresh formal=`/home/afr/SOLIDER-REID-exp413-psccr-formal-v1`，HEAD=
`add6adae4d192da4c44bf44120dd571f0dfe14e1`。唯一runner自然退出，GPU回到`0% / 2 MiB`且无compute进程，严格
异常计数0，runner SHA256=`24a64a1a9db5dec24ee8c7a3765a51d05945a253df2cd64a85133953c9180623`。

合同结果：

- 手算correct/pose-only/q-only/text-shuffle链=`[8,4,1]/[1,8,4]/[8,4,1]/[1,4,8]`；
- default-off state/forward/loss/gradient/RNG exact；excluded-image mutation invariant；三support严格排列；
- correct相对pose-only/q-only/text-shuffle的真实链改变率=`0.750000/0.656250/0.468750`；
- prefix coverage均值=`4.078125/5.765625/6.203125`，单调不降；prefix3 distance/loss与zero-owner exact；
- Stage-3可比梯度tensor=`28`，改变=`28`；combined AMP第5次native attempt取得真实update，Stage-3非零梯度
  tensor=`26`，更新参数=`base.stages.3.blocks.0.ffn.layers.1.weight`。

最终=`UNIQUE REAL PK64 PASS / FORMAL E120 AUTHORIZED`。不得重跑合同或追加preflight；下一步只允许fresh
correct seed1234自然训练到e120。

## 2026-07-22：唯一fresh correct e120已启动

从冻结contract formal建立fresh训练formal=`/home/afr/SOLIDER-REID-exp413-psccr-train-v1`，HEAD仍为
`add6adae4d192da4c44bf44120dd571f0dfe14e1`且tracked worktree/index diff均为0。fresh output=
`/home/afr/reid-clean/logs/exp413-psccr-s1234-v1`，runner=
`/home/afr/reid-clean/train-logs/exp413-psccr-s1234-v1.runner.log`，主PID=`754511`。

首批真实训练诊断：zero-owner宿主set loss=`1.970790`；PSCCR总set loss=`2.952360`，prefix1/2/3=
`4.396325/2.489964/1.970790`；coverage=`4.156250/5.796875/6.234375`；correct相对
pose-only/q-only/text-shuffle链改变率=`0.703125/0.687500/0.468750`。已进入e1 iter20/227，GPU约
`7,000 MiB/43%`，修正后的严格异常计数0。

当前=`FORMAL RUNNING / CONTINUE`。source/config/cache/text asset冻结，不续训；e10/20/.../120记录PSCCR、sealed
zero-owner与clean D0同epoch mAP/R1/R5/R10及差值，中间点不早停。

## 2026-07-22：PSCCR e10正式评测

PSCCR自然完成e10=`29.9 mAP / 39.8 R1 / 55.1 R5 / 60.9 R10`；同epoch sealed zero-owner=
`28.4/38.1/53.9/60.8`，sealed clean D0=`33.4/42.7/59.8/65.2`。当前轨迹为：

| epoch | PSCCR mAP/R1/R5/R10 | zero-owner同epoch | clean D0同epoch | PSCCR−zero | PSCCR−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 29.9/39.8/55.1/60.9 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +1.5/+1.7/+1.2/+0.1 | -3.5/-2.9/-4.7/-4.3 |

首个预注册点四项均严格高于关键宿主zero-owner，但四项均低于clean D0，说明早期相对宿主轨迹有利、绝对优化仍慢。
单个早期点不能代替e120性能门，也不能据此宣称pose×CLIP归因；继续冻结运行，不早停、不改机制。读取时已进入e13，
主PID=`754511`仍为唯一compute PID，GPU约`7,072 MiB/43%`，formal HEAD与tracked worktree/index保持不变，
runner/train严格异常计数0。

## 2026-07-22：PSCCR e20正式评测

PSCCR自然完成e20=`46.8 mAP / 56.7 R1 / 71.4 R5 / 77.6 R10`；同epoch sealed zero-owner=
`45.6/55.0/70.6/75.8`，sealed clean D0=`42.2/52.4/67.6/74.0`。截至当前轨迹为：

| epoch | PSCCR mAP/R1/R5/R10 | zero-owner同epoch | clean D0同epoch | PSCCR−zero | PSCCR−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 29.9/39.8/55.1/60.9 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +1.5/+1.7/+1.2/+0.1 | -3.5/-2.9/-4.7/-4.3 |
| 20 | 46.8/56.7/71.4/77.6 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.2/+1.7/+0.8/+1.8 | +4.6/+4.3/+3.8/+3.6 |

连续两个预注册点四项均严格高于zero-owner，且e20四项也已高于clean D0；这是有利的早期优化轨迹，但不能替代
e120性能门，也不能在未运行matched controls前归因于pose×CLIP联合排序。继续冻结运行，不早停、不改机制。
读取时已进入e23，主PID=`754511`仍为唯一compute PID，GPU约`7,078 MiB/47%`，formal HEAD与tracked
worktree/index保持不变，runner/train严格异常计数0。

## 2026-07-22：PSCCR e30正式评测

PSCCR自然完成e30=`52.0 mAP / 64.2 R1 / 76.9 R5 / 81.5 R10`；同epoch sealed zero-owner=
`49.2/60.3/75.0/80.0`，sealed clean D0=`46.6/56.2/71.3/76.4`。截至当前轨迹为：

| epoch | PSCCR mAP/R1/R5/R10 | zero-owner同epoch | clean D0同epoch | PSCCR−zero | PSCCR−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 29.9/39.8/55.1/60.9 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +1.5/+1.7/+1.2/+0.1 | -3.5/-2.9/-4.7/-4.3 |
| 20 | 46.8/56.7/71.4/77.6 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.2/+1.7/+0.8/+1.8 | +4.6/+4.3/+3.8/+3.6 |
| 30 | 52.0/64.2/76.9/81.5 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +2.8/+3.9/+1.9/+1.5 | +5.4/+8.0/+5.6/+5.1 |

e30相对zero-owner的四项优势扩大，且继续全面高于clean D0；三点轨迹支持coverage-prefix训练对象值得保留关注，
但不改变e120性能门，也不能替代pose-only/q-only/text-shuffle归因。继续冻结运行，不早停、不改机制。读取时已进入
e34，主PID=`754511`仍为唯一compute PID，GPU约`7,078 MiB/42%`，formal HEAD与tracked worktree/index保持
不变，runner/train严格异常计数0。

## 2026-07-22：PSCCR e40正式评测

PSCCR自然完成e40=`54.4 mAP / 65.5 R1 / 79.0 R5 / 83.2 R10`；同epoch sealed zero-owner=
`55.0/66.2/79.8/84.4`，sealed clean D0=`50.0/60.7/76.2/81.0`。截至当前轨迹为：

| epoch | PSCCR mAP/R1/R5/R10 | zero-owner同epoch | clean D0同epoch | PSCCR−zero | PSCCR−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 29.9/39.8/55.1/60.9 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +1.5/+1.7/+1.2/+0.1 | -3.5/-2.9/-4.7/-4.3 |
| 20 | 46.8/56.7/71.4/77.6 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.2/+1.7/+0.8/+1.8 | +4.6/+4.3/+3.8/+3.6 |
| 30 | 52.0/64.2/76.9/81.5 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +2.8/+3.9/+1.9/+1.5 | +5.4/+8.0/+5.6/+5.1 |
| 40 | 54.4/65.5/79.0/83.2 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | -0.6/-0.7/-0.8/-1.2 | +4.4/+4.8/+2.8/+2.2 |

e40首次四项低于关键宿主zero-owner，打断前三点的全面优势，但仍四项高于clean D0。该反转既不能被遗漏，也不能
单点早停或解释为最终失败；继续冻结运行到自然e120。读取时已进入e41，主PID=`754511`仍为唯一compute PID，
GPU约`7,074 MiB/43%`，formal HEAD与tracked worktree/index保持不变，runner/train严格异常计数0。

## 2026-07-22：PSCCR e50正式评测

PSCCR自然完成e50=`55.5 mAP / 66.6 R1 / 79.5 R5 / 84.1 R10`；同epoch sealed zero-owner=
`55.1/66.1/80.0/83.7`，sealed clean D0=`52.1/62.8/77.0/81.9`。截至当前轨迹为：

| epoch | PSCCR mAP/R1/R5/R10 | zero-owner同epoch | clean D0同epoch | PSCCR−zero | PSCCR−D0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 29.9/39.8/55.1/60.9 | 28.4/38.1/53.9/60.8 | 33.4/42.7/59.8/65.2 | +1.5/+1.7/+1.2/+0.1 | -3.5/-2.9/-4.7/-4.3 |
| 20 | 46.8/56.7/71.4/77.6 | 45.6/55.0/70.6/75.8 | 42.2/52.4/67.6/74.0 | +1.2/+1.7/+0.8/+1.8 | +4.6/+4.3/+3.8/+3.6 |
| 30 | 52.0/64.2/76.9/81.5 | 49.2/60.3/75.0/80.0 | 46.6/56.2/71.3/76.4 | +2.8/+3.9/+1.9/+1.5 | +5.4/+8.0/+5.6/+5.1 |
| 40 | 54.4/65.5/79.0/83.2 | 55.0/66.2/79.8/84.4 | 50.0/60.7/76.2/81.0 | -0.6/-0.7/-0.8/-1.2 | +4.4/+4.8/+2.8/+2.2 |
| 50 | 55.5/66.6/79.5/84.1 | 55.1/66.1/80.0/83.7 | 52.1/62.8/77.0/81.9 | +0.4/+0.5/-0.5/+0.4 | +3.4/+3.8/+2.5/+2.2 |

e50核心mAP/R1及R10重新严格高于zero-owner，但R5仍低`0.5`；四项均继续高于clean D0。该点说明e40反转并非
单调恶化，却仍不能替代e120核心门或matched-control归因。继续冻结运行，不早停、不改机制。读取时已进入e52，
主PID=`754511`仍为唯一compute PID，GPU约`7,072 MiB/43%`，formal HEAD与tracked worktree/index保持不变，
runner/train严格异常计数0。

## 2026-07-22：v1因主机硬重启中断，禁止续训

08:11心跳经中继连续两次SSH banner超时；08:26恢复后确认主PID=`754511`已消失、GPU=`0% / 2 MiB`，日志停在
e56完整结束后的e57首批诊断。主机`uptime -s=2026-07-22 08:16:54`，`last -x`只有08:16 reboot而没有对应
shutdown记录，说明训练被外部主机硬重启终止，不是自然e120完成，也不是训练代码异常。runner/train严格异常计数仍为0。

v1完整性封存：完整epoch数=`56`，正式评测点数=`5`（e10--e50）；train-log SHA256=
`a39e3355bd678d6662d6f820087998d8014f8fb469dc12342588aede8e4b8376`，runner SHA256=
`acb170024d55860e3d5454c7f1de77fd0eda19e1b09406fe15ac9e9f76c286cb`。v1状态=
`INFRASTRUCTURE INTERRUPTED / E120 VOID / NO RESUME`：已有五点轨迹全部保留，但不能代替e120性能裁决，禁止从
checkpoint续训。

同一冻结formal HEAD仍为`add6adae4d192da4c44bf44120dd571f0dfe14e1`，tracked worktree/index=`0/0`；合同已
PASS且机制/参数/资产未变，故不重跑合同或盲审。GPU恢复空闲，新fresh路径=
`/home/afr/reid-clean/logs/exp413-psccr-s1234-v2`与runner=
`/home/afr/reid-clean/train-logs/exp413-psccr-s1234-v2.runner.log`均确认不存在。下一步只允许从epoch 1、seed1234
在该全新OUTPUT_DIR重启完整正式臂；这不是续训，v1不得覆盖或删除。

## 2026-07-22：v2 fresh正式臂从epoch 1启动

同一冻结formal与seed1234通过命令行仅覆盖输出路径，fresh output=
`/home/afr/reid-clean/logs/exp413-psccr-s1234-v2`，runner=
`/home/afr/reid-clean/train-logs/exp413-psccr-s1234-v2.runner.log`；没有checkpoint参数或旧权重恢复，主PID=`2316`。

v2首批诊断与v1完全一致：zero-owner loss=`1.970790`；PSCCR prefix1/2/3=
`4.396325/2.489964/1.970790`；coverage=`4.156250/5.796875/6.234375`；correct相对pose-only/q-only/
text-shuffle链改变率=`0.703125/0.687500/0.468750`。启动后PID为唯一compute，GPU约`7,000 MiB/48%`，
formal HEAD=`add6adae4d192da4c44bf44120dd571f0dfe14e1`且tracked worktree/index=`0/0`，严格异常0。

当前=`V2 FRESH FORMAL RUNNING / CONTINUE`。v1保持封存，禁止续训或覆盖；v2自然运行到e120，e10/20/.../120
重新记录完整指标，不能用v1中间点填补v2。

## 2026-07-22：v2按用户指令人工终止，禁止续训

用户先暂停自动推进，随后明确要求训练也停止。远端仅终止v2主PID=`2316`及其同命令DataLoader子进程；终止后
目标进程=`0`，GPU=`2 MiB / 0%`。日志最后一行为e2 iter200，完整epoch数=`1`，正式评测点数=`0`，输出目录
仅有`train_log.txt`且没有checkpoint。runner/train严格异常计数均为`0`，说明这是用户授权的人工终止，不是训练
代码异常或基础设施中断。

v2完整性封存：train-log SHA256=
`ee53e0adff01a8ee332c4bb9c7e986b13b49b017f8ab84eec5fbdb8ce4fe83f9`，runner SHA256=
`04a9e08c113aa6b099f7a0bd46fef4eca686c93a24175aefcbd2df9a341726c1`。v2状态=
`USER STOPPED / E120 VOID / NO RESUME`；不得从该目录或任何状态续训，也不得把v1评测点补入v2。

用户现已明确恢复。恢复前核验唯一4090 compute为空、GPU=`2 MiB / 0%`，冻结formal HEAD仍为
`add6adae4d192da4c44bf44120dd571f0dfe14e1`且tracked worktree/index=`0/0`，fresh v3 output/runner均不存在。
下一步仅允许使用相同代码、config、资产与seed1234，在全新v3 OUTPUT_DIR从epoch 1启动；不重跑唯一PK64合同或
盲审，不覆盖v1/v2。

## 2026-07-22：v3 fresh正式臂从epoch 1启动

同一冻结formal与seed1234仅通过命令行覆盖fresh output=
`/home/afr/reid-clean/logs/exp413-psccr-s1234-v3`，runner=
`/home/afr/reid-clean/train-logs/exp413-psccr-s1234-v3.runner.log`；启动命令不含checkpoint参数，主PID=`3668`。

v3首批诊断再次与v1/v2完全一致：zero-owner loss=`1.970790`；PSCCR prefix1/2/3=
`4.396325/2.489964/1.970790`；coverage=`4.156250/5.796875/6.234375`；correct相对pose-only/q-only/
text-shuffle链改变率=`0.703125/0.687500/0.468750`。主PID是唯一CUDA compute，GPU约
`7,000 MiB / 42%`，formal tracked worktree/index=`0/0`，runner/train严格异常计数=`0/0`；读取时已到
e1 iter60。

当前=`V3 FRESH FORMAL RUNNING / CONTINUE`。v1/v2均保持封存且禁止续训或覆盖；v3须自然运行到e120，独立记录
e10/20/.../120全部12个正式评测点，禁止使用v1轨迹或v2日志补点。

## 2026-07-22：v3 e10正式评测

v3原始train log读取结果：PSCCR=`29.9/39.2/55.0/60.7`（mAP/R1/R5/R10）。同epoch sealed
zero-owner=`28.4/38.1/53.9/60.8`，sealed clean D0=`33.4/42.7/59.8/65.2`；因此
PSCCR−zero-owner=`+1.5/+1.1/+1.1/-0.1`，PSCCR−D0=`-3.5/-3.5/-4.8/-4.5`。

该点核心mAP/R1及R5严格胜宿主，但R10低`0.1`，且四项均低于D0。v3首批诊断虽与v1 exact，e10排序指标并不
与作废v1完全相同；后续裁决只使用v3自身12点评测，不引用v1补点或选择性替换。读取时主PID=`3668`仍为唯一
CUDA compute，运行至e18 iter140，GPU约`7,072 MiB / 42%`，runner/train严格异常=`0/0`，formal HEAD与
tracked worktree/index保持不变。继续冻结运行，不早停、不启动controls、不作pose×CLIP归因。

## 2026-07-22：v3 e20正式评测

v3原始train log读取结果：PSCCR=`46.1/56.0/71.3/76.6`。同epoch sealed zero-owner=
`45.6/55.0/70.6/75.8`，sealed clean D0=`42.2/52.4/67.6/74.0`；因此PSCCR−zero-owner=
`+0.5/+1.0/+0.7/+0.8`，PSCCR−D0=`+3.9/+3.6/+3.7/+2.6`。

e20四项同时严格胜宿主与D0，和e10的混合排序相比转为全项正差；仍只是第二个中间点，不能替代e120门或支持
pose×CLIP归因。读取时已进入e23，主PID=`3668`仍为唯一CUDA compute，GPU约`7,078 MiB / 47%`，
runner/train严格异常=`0/0`，formal HEAD与tracked worktree/index不变。继续冻结运行，不早停、不启动controls。

## 2026-07-22：v3 e30正式评测

v3原始train log读取结果：PSCCR=`51.1/61.7/74.9/80.4`。同epoch sealed zero-owner=
`49.2/60.3/75.0/80.0`，sealed clean D0=`46.6/56.2/71.3/76.4`；因此PSCCR−zero-owner=
`+1.9/+1.4/-0.1/+0.4`，PSCCR−D0=`+4.5/+5.5/+3.6/+4.0`。

e30核心mAP/R1及R10胜宿主，但R5低`0.1`；四项均继续胜D0。必须保留该混合证据，不能用作提前GO或选择性
忽略R5。读取时已进入e32 iter40，主PID=`3668`仍为唯一CUDA compute，GPU约`7,076 MiB / 42%`，
runner/train严格异常=`0/0`，formal HEAD与tracked worktree/index保持不变。继续冻结运行至e120。

## 2026-07-22：v3 e50正式评测

v3原始train log读取结果：PSCCR=`55.2/65.4/78.9/83.7`。同epoch sealed zero-owner=
`55.1/66.1/80.0/83.7`，sealed clean D0=`52.1/62.8/77.0/81.9`；因此PSCCR−zero-owner=
`+0.1/-0.7/-1.1/+0.0`，PSCCR−D0=`+3.1/+2.6/+1.9/+1.8`。

e50相对宿主仅mAP微正，R1与R5转负，R10持平；四项仍胜D0。该点削弱连续核心优势，但不能替代e120裁决，
必须保留且不得早停或救臂。读取时已进入e53 iter100，主PID=`3668`仍为唯一CUDA compute，GPU约
`7,090 MiB / 42%`，runner/train严格异常=`0/0`，formal HEAD与tracked worktree/index保持不变。继续冻结运行。

## 2026-07-22：v3 e60正式评测

v3原始train log读取结果：PSCCR=`58.2/70.0/82.4/85.8`。从sealed exp411原始记录读取同epoch
zero-owner=`57.6/70.3/81.0/85.2`，clean D0=`55.1/66.1/79.0/83.3`；因此PSCCR−zero-owner=
`+0.6/-0.3/+1.4/+0.6`，PSCCR−D0=`+3.1/+3.9/+3.4/+2.5`。

e60相对宿主mAP/R5/R10为正，但核心R1仍低`0.3`；四项继续胜D0。该点不能满足最终双核心门，也不能替代e120
裁决。读取时已进入e62，主PID=`3668`仍为唯一CUDA compute，GPU约`7,066 MiB / 42%`，runner/train严格
异常=`0/0`，formal HEAD与tracked worktree/index保持不变。继续冻结运行，不早停、不启动controls。

## 2026-07-22：v3 e70正式评测

v3原始train log读取结果：PSCCR=`58.1/69.5/82.2/85.7`。从sealed exp411原始记录读取同epoch
zero-owner=`57.8/70.2/81.7/85.5`，clean D0=`55.4/65.2/79.5/83.6`；因此PSCCR−zero-owner=
`+0.3/-0.7/+0.5/+0.2`，PSCCR−D0=`+2.7/+4.3/+2.7/+2.1`。

e70相对宿主mAP/R5/R10仍为正，但核心R1负差由e60的`0.3`扩大到`0.7`；四项仍胜D0。该点进一步增加
最终双核心门风险，但中间点不得早停或启动controls。读取时已进入e73 iter140，主PID=`3668`仍为唯一CUDA
compute，GPU约`7,088 MiB / 47%`，runner/train严格异常=`0/0`，formal状态保持冻结。继续自然运行。

## 2026-07-22：v3 e80正式评测

v3原始train log读取结果：PSCCR=`58.8/70.7/82.2/86.3`。从sealed exp411原始记录读取同epoch
zero-owner=`58.6/71.6/82.4/86.3`，clean D0=`56.1/66.3/79.5/84.0`；因此PSCCR−zero-owner=
`+0.2/-0.9/-0.2/+0.0`，PSCCR−D0=`+2.7/+4.4/+2.7/+2.3`。

e80相对宿主仅mAP微正，核心R1负差扩大至`0.9`，R5略负、R10持平；四项仍胜D0。这是连续第三个R1低于
宿主的正式点，必须保留但不能提前判NO-GO。读取时已进入e84 iter40，主PID=`3668`仍为唯一CUDA compute，
GPU约`7,070 MiB / 42%`，runner/train严格异常=`0/0`，formal状态保持冻结。继续自然运行至e120。

## 2026-07-22：v3 e90正式评测

v3原始train log读取结果：PSCCR=`59.6/71.4/83.1/87.0`。从sealed exp411原始记录读取同epoch
zero-owner=`59.1/71.2/82.6/86.8`，clean D0=`57.5/67.9/81.2/85.3`；因此PSCCR−zero-owner=
`+0.5/+0.2/+0.5/+0.2`，PSCCR−D0=`+2.1/+3.5/+1.9/+1.7`。

e90四项重新同时严格胜宿主与D0，核心R1从e60--e80连续负差恢复为`+0.2`。该恢复点不能删除此前不利证据，
也不能提前触发性能GO或controls。读取时已进入e91 iter80，主PID=`3668`仍为唯一CUDA compute，GPU约
`7,066 MiB / 42%`，runner/train严格异常=`0/0`，formal状态保持冻结。继续自然运行至e120。

## 2026-07-22：v3 e40正式评测

v3原始train log读取结果：PSCCR=`55.6/66.7/80.2/84.3`。同epoch sealed zero-owner=
`55.0/66.2/79.8/84.4`，sealed clean D0=`50.0/60.7/76.2/81.0`；因此PSCCR−zero-owner=
`+0.6/+0.5/+0.4/-0.1`，PSCCR−D0=`+5.6/+6.0/+4.0/+3.3`。

e40核心mAP/R1及R5胜宿主，但R10低`0.1`；四项均胜D0。该点仍是混合宿主证据，不能表述为全项胜或提前触发
controls。读取时已进入e44 iter80，主PID=`3668`为唯一CUDA compute，GPU约`7,076 MiB / 46%`，
runner/train严格异常=`0/0`，formal HEAD与tracked worktree/index保持不变。继续冻结运行至e120。

## 2026-07-22：v3 e100正式评测

v3原始train log读取结果：PSCCR=`59.2/70.4/82.2/86.2`。从sealed exp411原始记录读取同epoch
zero-owner=`58.8/70.5/82.2/86.1`，clean D0=`56.9/67.1/79.6/83.8`；因此PSCCR−zero-owner=
`+0.4/-0.1/+0.0/+0.1`，PSCCR−D0=`+2.3/+3.3/+2.6/+2.4`。

e100相对宿主仅mAP和R10微正，核心R1低`0.1`、R5持平；四项仍胜D0。该点使e90的四项正差再次转为混合
证据，必须保留但不能提前判NO-GO。读取时已进入e103 iter140，主PID=`3668`仍为唯一CUDA compute，GPU约
`7,076 MiB / 42%`，runner/train严格异常=`0/0`，formal tracked worktree/index保持`0/0`。继续冻结运行至
e120，不启动controls；下一正式记录点为e110。

## 2026-07-22：v3 e110正式评测

v3原始train log读取结果：PSCCR=`59.1/70.5/82.3/85.8`。从sealed exp411原始记录读取同epoch
zero-owner=`58.8/70.4/81.8/86.1`，clean D0=`57.4/67.4/80.5/84.6`；因此PSCCR−zero-owner=
`+0.3/+0.1/+0.5/-0.3`，PSCCR−D0=`+1.7/+3.1/+1.8/+1.2`。

e110相对宿主核心mAP/R1与R5严格为正，但R10低`0.3`；四项仍胜D0。该点重新满足中间双核心正差，但不能
替代e120预注册裁决或触发controls。读取时已进入e111 iter160，主PID=`3668`仍为唯一CUDA compute，GPU约
`7,088 MiB / 42%`，runner/train严格异常=`0/0`，formal tracked worktree/index保持`0/0`。继续冻结运行至
自然e120，不改参、不续训、不提前归因。

## 2026-07-22：v3 correct自然e120完成，性能GO

v3原始train log最终结果：PSCCR=`59.3/70.8/82.6/86.0`。sealed zero-owner e120=
`58.9/70.3/81.9/86.2`，sealed clean D0 e120=`57.6/67.7/80.8/84.6`；因此PSCCR−zero-owner=
`+0.4/+0.5/+0.7/-0.2`，PSCCR−D0=`+1.7/+3.1/+1.8/+1.4`。mAP与R1同时严格胜预注册宿主，故判定
`EXP413 PERFORMANCE GO / ATTRIBUTION PENDING`；R10低宿主`0.2`仍完整保留。

训练自然完成全部120 epoch与`[10,20,...,120]`共12个正式评测点后主PID=`3668`消失，目标训练进程为0，GPU=
`2 MiB / 0% / 0 compute`，runner/train严格异常=`0/0`，formal HEAD=
`add6adae4d192da4c44bf44120dd571f0dfe14e1`且tracked worktree/index=`0/0`。唯一checkpoint、train log、runner
SHA256依次为：

- `b49ae00246d07bf014e43eaeb5c0c76d6c95071f52d6034a3fcc7b6f687c4af4`；
- `6cec0de7bba41277f400ed86fa8f30268be5a549ba250c164b38b5a92f4f0b30`；
- `d156614f836131051bca99d167d68998953b698173df32a9b6f7ec7bbd839a60`。

correct臂现永久封存，禁止重跑、续训或修改产物。按预注册顺序只授权同一冻结formal、recipe与seed1234上的fresh
`pose-only` matched control；q-only与text-shuffle继续`NO-START`。只有correct在mAP/R1同时严格胜全部三control，
才能判`POSE+CLIP SCIENTIFIC GO`。

## 2026-07-22：fresh pose-only matched control启动

GPU空闲、correct封存和formal tracked 0/0核验后，以同一formal HEAD、config、cache、text asset、student、batch64、
`P×K=16×4`、seed1234及e120 recipe启动唯一pose-only；命令行只覆盖
`MODEL.TAPF.PSCCR_CONTROL_MODE=pose_only`与fresh OUTPUT_DIR。output/runner为：

- `/home/afr/reid-clean/logs/exp413-psccr-pose-only-s1234-v1`；
- `/home/afr/reid-clean/train-logs/exp413-psccr-pose-only-s1234-v1.runner.log`。

命令不含checkpoint恢复，主PID=`40519`。config dump确认control=`pose_only`、seed=`1234`、batch=`64`、epochs=`120`；
首批zero-owner宿主loss=`1.970790`，PSCCR pose-only loss/prefix1/2/3=
`2.682307/[3.459568,2.616564,1.970790]`，prefix3继续与zero-owner exact，coverage=
`7.812500/9.593750/10.000000`。e1 iter20已正常，主PID为唯一CUDA compute，GPU约`7,000 MiB / 57%`，
runner/train严格异常=`0/0`，formal tracked worktree/index=`0/0`。q-only与text-shuffle保持`NO-START`；pose-only
自然运行到e120并逐10 epoch完整记录，不以中间点早停。

## 2026-07-22：pose-only e10正式评测

pose-only原始train log读取结果=`29.9/39.7/54.5/60.9`。同epoch sealed v3 correct=
`29.9/39.2/55.0/60.7`，sealed zero-owner=`28.4/38.1/53.9/60.8`，sealed clean D0=
`33.4/42.7/59.8/65.2`；因此pose-only−correct=`+0.0/+0.5/-0.5/+0.2`，pose-only−zero-owner=
`+1.5/+1.6/+0.6/+0.1`，pose-only−D0=`-3.5/-3.0/-5.3/-4.3`。

e10时pose-only与correct mAP持平，R1/R10分别高`0.5/0.2`，R5低`0.5`；这对correct最终严格胜pose-only的联合
归因门是暂时不利的早期证据，但不能替代e120裁决。读取时已进入e11 iter20，主PID=`40519`仍为唯一CUDA
compute，GPU约`7,072 MiB / 72%`，runner/train严格异常=`0/0`，formal tracked worktree/index=`0/0`。继续
冻结运行，不早停、不修改，q-only与text-shuffle保持`NO-START`。

## 2026-07-22：pose-only e20正式评测

pose-only原始train log读取结果=`46.2/55.7/70.8/77.1`。同epoch sealed v3 correct=
`46.1/56.0/71.3/76.6`，sealed zero-owner=`45.6/55.0/70.6/75.8`，sealed clean D0=
`42.2/52.4/67.6/74.0`；因此pose-only−correct=`+0.1/-0.3/-0.5/+0.5`，pose-only−zero-owner=
`+0.6/+0.7/+0.2/+1.3`，pose-only−D0=`+4.0/+3.3/+3.2/+3.1`。

e20时pose-only的mAP/R10高于correct，R1/R5低于correct；相较e10，R1关系由pose-only领先转为correct领先，但
mAP由持平转为pose-only领先`0.1`，仍是混合归因证据。读取时已进入e21 iter120，主PID=`40519`仍为唯一CUDA
compute，GPU约`7,078 MiB / 48%`，runner/train严格异常=`0/0`，formal tracked worktree/index=`0/0`。继续
自然训练，不早停，不提前运行q-only或text-shuffle。

## 2026-07-22：pose-only e30正式评测

pose-only原始train log读取结果=`51.0/61.8/75.2/80.8`。同epoch sealed v3 correct=
`51.1/61.7/74.9/80.4`，sealed zero-owner=`49.2/60.3/75.0/80.0`，sealed clean D0=
`46.6/56.2/71.3/76.4`；因此pose-only−correct=`-0.1/+0.1/+0.3/+0.4`，pose-only−zero-owner=
`+1.8/+1.5/+0.2/+0.8`，pose-only−D0=`+4.4/+5.6/+3.9/+4.4`。

e30时correct仅mAP高pose-only `0.1`，而pose-only的R1/R5/R10分别高`0.1/0.3/0.4`；前三个中间点均未形成
correct双核心同时严格领先，当前联合归因风险持续。读取时已进入e32 iter20，主PID=`40519`仍为唯一CUDA
compute，GPU约`7,078 MiB / 47%`，runner/train严格异常=`0/0`，formal tracked worktree/index=`0/0`。该轨迹
不能替代e120裁决，继续冻结运行。

## 2026-07-22：pose-only e40正式评测

pose-only原始runner读取结果=`54.9/66.4/79.7/83.4`。同epoch sealed v3 correct=
`55.6/66.7/80.2/84.3`，sealed zero-owner=`55.0/66.2/79.8/84.4`，sealed clean D0=
`50.0/60.7/76.2/81.0`；因此pose-only−correct=`-0.7/-0.3/-0.5/-0.9`，pose-only−zero-owner=
`-0.1/+0.2/-0.1/-1.0`，pose-only−D0=`+4.9/+5.7/+3.5/+2.4`。

e40是首个correct在核心mAP/R1上同时严格领先pose-only的正式点，且correct四项均领先；但pose-only的R1仍比
zero-owner高`0.2`，其余三项低于宿主，证据仍须按完整轨迹解释。读取时已进入e42 iter140，主PID=`40519`
仍为唯一CUDA compute，GPU约`7,076 MiB / 48%`，runner/train严格异常=`0/0`，formal tracked
worktree/index=`0/0`。该中间点不触发科学GO或早停，继续冻结运行至自然e120。

## 2026-07-22：pose-only e50正式评测

pose-only原始runner读取结果=`55.6/66.1/79.2/83.8`。同epoch sealed v3 correct=
`55.2/65.4/78.9/83.7`，sealed zero-owner=`55.1/66.1/80.0/83.7`，sealed clean D0=
`52.1/62.8/77.0/81.9`；因此pose-only−correct=`+0.4/+0.7/+0.3/+0.1`，pose-only−zero-owner=
`+0.5/+0.0/-0.8/+0.1`，pose-only−D0=`+3.5/+3.3/+2.2/+1.9`。

e50时pose-only四项均严格高于correct，mAP/R1双核心关系从e40的correct领先再次反转，是当前最明确的不利联合
归因中间证据；相对zero-owner则mAP/R10为正、R1持平、R5为负。读取时已进入e53 iter60，主PID=`40519`
仍为唯一CUDA compute，GPU约`7,072 MiB / 45%`，runner/train严格异常=`0/0`，formal tracked
worktree/index=`0/0`。不得据此早停或删点，继续冻结运行至自然e120。

## 2026-07-23：pose-only e60正式评测

pose-only原始runner读取结果=`57.7/69.1/81.2/84.7`。同epoch sealed v3 correct=
`58.2/70.0/82.4/85.8`，sealed zero-owner=`57.6/70.3/81.0/85.2`，sealed clean D0=
`55.1/66.1/79.0/83.3`；因此pose-only−correct=`-0.5/-0.9/-1.2/-1.1`，pose-only−zero-owner=
`+0.1/-1.2/+0.2/-0.5`，pose-only−D0=`+2.6/+3.0/+2.2/+1.4`。

e60时correct再次在mAP/R1/R5/R10四项严格领先pose-only，e50的四项反超没有连续保持；但pose-only相对宿主仍
呈mAP/R5略正、R1/R10为负的混合关系，且e50不利点不得删除。读取时已进入e64 iter20，主PID=`40519`仍为
唯一CUDA compute，GPU约`7,084 MiB / 42%`，runner/train严格异常=`0/0`，formal tracked
worktree/index=`0/0`。继续冻结运行，不提前裁决或启动后续controls。
