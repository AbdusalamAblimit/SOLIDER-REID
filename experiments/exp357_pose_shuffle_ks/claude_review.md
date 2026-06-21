# Claude Review — exp357 pose-shuffle kill-switch

**Verdict: PASS（审查通过）** 无 Critical/High。

## 代码正确性(全过)
- 置换非恒等(randperm + identity 时 roll), scene/target 同一 perm 保持配对, batch dim 0 正确; 图像 x 不置换 → 图 i 收到 perm[i] 的 pose(intended cross-image mismatch)。
- 训练端守卫(self.training); 测试用真 pose(与 exp353 test 对称公平)。
- 单变量: config diff 仅 POSE_SHUFFLE True + OUTPUT_DIR; POSE_SHUFFLE False 真 no-op(向后兼容)。
- 干净隔离: exp353 POSE_PSG_STAGES 空 → backbone pose-free, GCN/PPA/VCSR 全 off → pose 只进 LGPA, shuffle 只扰 LGPA。assign loss 内部自洽(pose_bias 与 assign target 同源 shuffled hm)。
- 边界: B=1 跳过; device/dtype/AMP 安全。

## 判读注意(Medium, 非 bug)
**NO-DROP 侧有混淆**: ReID 裁剪空间对齐(人居中、头上脚下), cross-image pose 的 blob 部分重叠真实布局 → 若 exp357≈60.5 有两解: (1)pose 正确性无关, 或 (2)裁剪对齐让乱 pose 还是大致对。**DROP 侧干净**(掉点 = pose 因果, 故事稳)。
**缓解**: 若 exp357 不掉点, 必须补 cross-PART shuffle(打乱 17 关键点通道, 破坏正确性无裁剪对齐 rescue)二次确认, 才能下"pose 正确性无关"结论。
Low: 固定点泄漏(~1/64 保留自身 pose)、PK 同 ID 污染(~5%)——均可忽略。

## 结论
审查通过。kill-switch 设计 sound, 结果可判读但不对称: 掉点=干净铁证(pose 因果); 不掉=需 cross-part shuffle 二次确认。可进 Codex。
