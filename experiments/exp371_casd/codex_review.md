# exp371 Phase 0 Codex 实现审查

## 审查范围

- `design.md`
- exp336/exp340/exp340c 原始 config、checkpoint 与日志
- `clip_part_head.py` / `pose_backbone_model.py`
- `eval_pose_interventions.py` / `intervention_utils.py`
- `packed_descriptor_oracle.py`
- query-mode 与 intervention/packing 单元测试

## 审查结论

**Phase 0 可执行；完整 CASD 训练仍不允许启动。**

## 已发现并修正的问题

1. 初版 shuffled 只做全数据有放回 donor，会跨 query/gallery 且不保持 pose multiset。已改为 query、gallery 各自的异 PID、无 fixed point、严格双射。
2. 初版 uniform 使用空间常数，会在 attention softmax 中近似抵消。已改为保留当前图 common body foreground、复制到全部 keypoint 通道，只删除 anatomical assignment。
3. learned query 若直接与 CLIP 比较会混淆初始化来源和可学习性。已锁定 fixed-random 与 learned-random 逐 bit 相同，仅改变 buffer/Parameter。
4. 5376→768 若在 query/gallery 上 fit PCA 会产生 transductive leakage。已锁定只用 `train_loader_normal` fit，并硬检查 train/eval path 不相交。
5. 只比较绝对 mAP 不能回答“保留多少 LGPA 涨点”。已改为相对 matched global/full 的 paired-gain retention。

## 执行前硬断言

1. checkpoint SHA、config、Git execution 和 dataset 顺序必须落 manifest；
2. exp336 必须是空 PSG、LGPA detach、equal-concat、无 NFC/re-ranking/MaxSim；
3. correct 全量必须复现原始 mAP/R1，允许误差不超过 `0.1 pp`；
4. 五种 intervention 的 global feature SHA 必须完全相同；
5. shuffled 每个 split donor PID 冲突为 0、unique donor=N、max reuse=1；
6. 输出必须严格为 `7×768=5376-D` 且全部 finite；
7. Gate D 的 train/val checkpoint SHA 必须相同，路径交集为 0，输出严格 768-D。

## Query mode smoke 门禁

- random-frozen/random-learned 初值和除注册类型外的 state dict 逐 tensor 相同；
- fixed query 是 buffer，learned query 是 Parameter；
- trainable parameter 差恰为 3072；
- 初始 eval forward 一致；
- part loss backward 后 backbone 因 detach 无梯度，learned query 梯度 finite 且非零；
- checkpoint round-trip 后注册类型、query SHA 和输出保持。

## 停止规则

- 任一 parity/invariant 失败：结果无效，只修脚本，不解释指标；
- correct pose 不优于 uniform/shuffled：pose-specific IPER claim 判负；
- packed retention 不足 80%：同维化判负，不用测试集调投影；
- 所有 Phase 0 结果只决定是否进入冻结-backbone kill-switch，不得提前写成论文正结果。
