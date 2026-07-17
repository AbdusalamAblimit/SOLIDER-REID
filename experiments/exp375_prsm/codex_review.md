# exp375 训练前 Codex 审查

当前状态：`PASS_FOR_TRAINING_AND_FROZEN_COUNTERFACTUAL_EVAL`。

审查只关注会改变科学结论的阻塞项：pose 是否只作用于 recurrent write/retain、默认关闭
是否保持 baseline、三臂是否单变量、correct/shuffle/canonical/foreground/zero 是否可解释、
梯度与 AMP 是否健康。非阻塞的格式、额外哈希或大矩阵建议不延迟训练。

## 2026-07-15 — 第一轮静态审查完成

三路 Codex 独立审查分别覆盖模块代码、科学归因和生产集成。审查共发现并修复四类真实
阻塞：三份 YAML 中的字面换行、LGPA 关闭时 canonical cache 未初始化、同 token
post-write read 可能退化为 memoryless pose gate，以及孤立模块测试不能证明 target-only
生产数据流。

最终实现改为 pre-write read：当前 token 先读取 carried state，再写入自己的 RGB
candidate，因此 pose 只能通过 recurrence 影响其他 token。新增真实 Swin 集成 smoke：按
B0/M0/P0 config 构造 production `PoseBackboneModel`，验证 M0/P0 参数完全配对、有效
distractor 条件下 PRSM 实际输入 bitwise 等于 target person 而非 scene merge、标准 768-d
descriptor 与 `12×4` feature map、关键梯度、strict checkpoint reload。

最终三路裁决均为 `PASS_FOR_TESTS_AND_SINGLE_BATCH_GPU_SMOKE`。下一步直接执行模块单测与
真实模型 smoke；若通过，立即启动正式训练，不再增加审查层级。

三路审查留痕如下；这里记录的是各路最终检查范围与裁决，不把综合摘要冒充逐字审查原文：

| 审查路 | 主要检查对象 | 阻塞项处置后裁决 |
|---|---|---|
| 模块/状态动力学 | write/retain 边界、pre-write read、zero identity、梯度与 AMP | PASS |
| 科学归因 | B0/M0/P0 单变量、姿态仅控制写入、反事实可解释性 | PASS_WITH_EVAL_INVARIANTS |
| 生产集成 | YAML、默认关闭、target-only 数据流、Swin shape、checkpoint reload | PASS_FOR_GPU_SMOKE |

随后模块测试与 production Swin 单 batch GPU smoke 均通过，故状态由最初的
`PENDING_IMPLEMENTATION_REVIEW` 更新为当前 PASS；上表不新增审查层级，只纠正原文件首部
未随实际执行结果同步的问题。

## 2026-07-15 — 同 checkpoint 反事实协议复审

训练启动后，科学审查进一步收紧了反事实的可解释边界，训练实现无需回退，评测入口必须
满足以下不变量后才允许解释：

1. 所有 arm 使用同一 P0 checkpoint、同一模型实例、同一 RGB/path/PID/camera 顺序与
   evaluator；correct-start/end 必须精确复现；
2. shuffle donor map 必须在 query/gallery 内分别冻结，与 batch/worker/order 无关，且为
   无 fixed point、异 PID 的双射；只替换 target-person pose，并审计 write mass、support、
   纵向中心与跨度；
3. foreground-uniform 必须沿用 correct visibility，使逐像素与全图总写入量严格相等，
   只删除 anatomical slot assignment；
4. zero 必须对 PRSM 输入形成 exact identity；它不替代另训 B0；
5. full canonical 因同时改变 route/support/write mass，只能作诊断；只有 mass-matched
   canonical 才能进入 `correct−canonical` 硬门禁。

裁决：`PASS_WITH_EVAL_INVARIANTS`。训练继续；评测代码和结果须逐项证明上述不变量。按用户
新增纪律，epoch 60 之前即使反事实差值为负，也只能用于管线 smoke，不能作 NO-GO。
