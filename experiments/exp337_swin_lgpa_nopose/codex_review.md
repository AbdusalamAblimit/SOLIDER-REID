# Codex Review — exp337 no-pose LGPA ablation

**Verdict: approve**（无 Critical/High/Medium，1 Low 措辞）

## Checks passed
1. **heatmaps=None 安全**:CLIPPartHead.forward 里 pose_bias→None、visibility→uniform 1/K（feature device 上）、assign_loss=torch.tensor(0.0, device=feat_map.device)。无除零/CPU-GPU 不匹配（clip_part_head.py:242,261）。
2. **flag 不禁用 LGPA**:scene_heatmaps 仍由 _prepare_pose 正常构建;只是 `lgpa_hm` 在调 clip_part_head 前置 None（pose_backbone_model.py:483,605,802）。门 `scene_heatmaps is not None` 仍过。
3. **单变量干净**:vs exp336 仅 `POSE_LGPA_NO_POSE: True`（+ OUTPUT_DIR/注释）；default False（defaults.py:223）。
4. **eval 比较有效**:POSE_TEST_FEAT=global 跳过 LGPA 组装;equal_concat 用 normed global + LGPA [pooled, p1..p5]（pose_backbone_model.py:800,914）。
5. **lgpa_assign 应 log 0.000**:processor 在 assign_loss present 时总加且 log;no-pose CLIPPartHead 仍返回该 key（processor.py:1025）。

## Low（措辞,非阻断）
- "no pose" = "LGPA 无 pose 引导",非"pose 管线禁用"。LGPA 分支仍需 scene_heatmaps is not None 才进入。design 措辞已正确,论文写作保持精确（CLIP 原型仍 cross-attend 全部 token,只是无 pose-bias）。

## 结论
codex 审查通过。配合 Claude review PASS（41 行）+ build 验证（NO-POSE ablation 生效）,可训练。测试只 override POSE_TEST_FEAT,不设 POSE_LGPA=False。
