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
