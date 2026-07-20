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
