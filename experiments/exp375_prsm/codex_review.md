# exp375 训练前 Codex 审查

当前状态：`PENDING_IMPLEMENTATION_REVIEW`。

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
