# exp392 监控记录

## 当前状态

- `DESIGN-ONLY / RESEARCH-ONLY / NO-START`；
- exp390与exp391均已封板，禁止重启、续训或把本实验记为exp391 Phase B/C；
- 当前GPU训练任务：无；
- 当前仅完成文献、公开代码、当前TAPF执行路径与机制审查；
- 未创建config、output、runner、checkpoint，未启动任何GPU任务。

## 已确认边界

1. current PSG是field-only自由`17→32`空间门控，结构上不要求joint-channel语义可辨识；
2. exp391 H2-M相对D0为`−0.4 mAP`，但early-bypass仍有`+0.141 mAP`，说明route可达而纯结构
   topology不足；
3. RegionCLIP、π-VL、PAFormer、MUVA、ProFD、ALADIN等已覆盖CLIP局部语义、pose-aware part、
   multi-level guidance或inference-free KD的主要构件；
4. 可争对象必须是counterfactually identifiable executable anatomical mediator，而不是“首次CLIP+pose”；
5. exp391只封板无语义校准的纯结构链；semantic single-stage通过后允许重新验证semantic multi-stage。

## 下一步

继续只读复核Phase 0A/0B指标、公开近邻与当前代码接口；所有门禁冻结前保持`NO-START`。
