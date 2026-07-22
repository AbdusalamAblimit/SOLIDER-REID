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
