# exp413 PSCCR 独立盲审

## 2026-07-22：首轮 1B/2H

独立审查只检查致命bug与变量混淆，首轮结论=`1 BLOCKER / 2 HIGH`：

1. BLOCKER：原设计先在K=4内计算序数再做leave-one-position-out，导致support chain泄漏被排除图；正类还会读取
   anchor自身pose/CLIP。已改为先得到每个`S[p,a]`，再只在三support内部重算严格序数，并预注册excluded-image
   mutation invariance。
2. HIGH：原invalid规则与pose-only定义冲突，且未说明valid图是否与invalid peer比较。已冻结q-dependent臂只在valid
   peer中计算`rank_q`、invalid自身q-rank/u为0；pose-only完全忽略CLIP valid/q，q-only完全忽略visibility。
3. HIGH：原真实PK64门无法发现比较方向写反、invalid混入或K=4泄漏。已在同一个唯一runner前半增加手算micro-oracle，
   后半仍只执行一次真实PK64 CUDA/AMP/update，不增加第二次preflight。

其余结构通过：逐步删除候选可保证三support严格排列且无丢弃；coverage单调；prefix3显式复用原support路径可保持
bit-exact；修正后四臂具备归因能力。当前等待同一独立审查者复核，复核`0B/0H`前不实现、不运行合同。

## 2026-07-22：复核 0B/0H

同一独立审查者确认三项首轮问题均已闭环，且没有引入新的BLOCKER/HIGH：

- 每个`(anchor, identity)`先构造LOO三support，再只在三图内排名；
- q-dependent valid规则、pose-only与q-only证据轴边界无歧义；
- 唯一runner同时覆盖手算micro-oracle、mutation invariance与真实PK64/AMP。

最终=`0 BLOCKER / 0 HIGH / IMPLEMENTATION AUTHORIZED`。该结论只授权默认关闭的最小实现和一次性合同，不是性能或
创新GO。

## 2026-07-22：实现盲审 0B/0H

独立代码审查完整核对新loss/state、defaults、make_loss、processor、formal config及唯一runner，结论=
`0 BLOCKER / 0 HIGH`：

- 生产路径先读取sealed zero-owner `support_indices`完成LOO，再只在三support内部计算rank；
- q-valid语义、四control证据轴、边际coverage与绝对batch-index tie均与设计一致；
- prefix3直接调用未改zero-owner原support/mean/loss路径，processor不叠加两种set loss；
- default-off与eval不加载PSCCR asset；
- runner的micro-oracle、excluded mutation、真实16×4、control活性、Stage-3差异梯度与原生GradScaler update不存在
  HIGH级伪PASS。

最终=`IMPLEMENTATION REVIEW PASS / UNIQUE REAL PK64 CONTRACT AUTHORIZED`，仍无CUDA或性能结果。
