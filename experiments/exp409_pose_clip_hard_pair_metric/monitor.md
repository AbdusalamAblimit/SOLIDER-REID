# exp409 PCHM 监控

## 2026-07-21：设计冻结前审计

exp408 已按 `MECHANISM ORDER PASS / PERFORMANCE FAIL` 封板，4090 当前空闲。三个独立视角提出 PCOIR、PCHM、
PCSIC 与 PSRD；近期近邻审计认为继续 part KD/attention/router 会重复 PICRD、ProFD、PAFormer、MUVA 等对象。

当前选择 PCHM：它不再回归 CLIP embedding/relation，而是让 pose×CLIP 共同决定 final descriptor 原 triplet 的
离散 pair index，最直接回应 exp408“语义可学但未改变最终检索几何”的失败。PCOIR 暂不执行，因为 foreign-part
paste 容易退化成 pose-guided CutMix并引入 recipient 标签噪声；不得把 PCOIR 与 PCHM 组合。

代码映射智能体独立确认：exp408 的 supervision 位于 Stage-2 局部 Gram，而最终检索走 Stage-3→GAP
`global_feat`；PCHM 可在 `processor.py` 构造pair index，并由 `make_loss.py` 透传到 `TripletLoss` 对最终
distance matrix gather。默认 `None` 可保持 D0 exact。另一个智能体从实现侧否决 PCOIR：现有RGB已经
normalize/随机擦除，pose hard-owner是稀疏Gaussian区域，异PID part copy同时存在patch质量与partial-chimera
标签污染。

状态：`DESIGN FROZEN / IMPLEMENTATION NO-START / GPU IDLE`。下一步只做必要实现检查与一次独立代码盲审；
`0B/0H` 后立即构建 fresh exp409 cache并运行，不进行无穷 preflight。

首轮实现盲审发现`0B/1H`：cache整体SHA已绑定，但NPZ未内嵌逐图RGB SHA、CLIP checkpoint、pose manifest、
preprocess和source provenance，且builder未验证实际RGB SHA。已在任何cache/GPU执行前补为逐图image SHA验证、
完整metadata写入和loader严格校验；等待同一盲审者聚焦闭环。

同一独立审查者聚焦复审确认原HIGH闭环，最终=`0 BLOCKER / 0 HIGH`。rank方向、候选mask、Borda/tie-break、
pair gather梯度、default-off exact及AMP接线均未发现B/H；紧凑contract还覆盖cache provenance roundtrip和错误
image SHA拒绝。当前=`IMPLEMENTED / BLIND REVIEW 0B/0H / REMOTE MMPOSE CONTRACT NEXT / GPU IDLE`。

远端fresh隔离repo=`/home/afr/SOLIDER-REID-exp409-pchm-d2cc18f-v1`，固定source HEAD=
`07170a22f1a6fbef2f0f140106e7350c223deb29`且clean。MMPOSE-ABU contract在CUDA上PASS：batch64，
default loss/gradient bit-exact，错误positive与cache image-SHA mutant均被拒；correct相对control的pair change rate=
wrong-RGB `0.9375`、generic `0.96875`、zero/CLIP-only `0.875`、pose-shuffle `0.90625`。这证明联合miner与
五个control均active，不是D0路径漂移；下一步直接构建唯一fresh exp409 cache。
