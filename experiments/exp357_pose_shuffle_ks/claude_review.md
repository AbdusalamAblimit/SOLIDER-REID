# Claude Broad Review — exp357 pose-shuffle kill-switch

**Verdict: PASS（审查通过）** 无 Critical/High。Claude Opus 子代理逐行审查, 追踪了完整 forward 路径、所有 pose-consumer flag、config diff、LGPA train/test 路径。findings 均为判读注意, 非代码 bug。

## 逐项 focus-area 核对

### a. 置换正确性 — OK
- 非恒等: `model/pose_backbone_model.py:715` `torch.randperm(Bp)`, 等于 arange 时 roll(后续 Codex 升级为 derangement)。
- scene/target 同 perm 配对: `scene_heatmaps[perm]` 与 `target_heatmaps[perm]` 用同一 `perm`。
- 正确维度: `_prepare_pose` 返回 (B,17,H,W); `Bp = scene_heatmaps.shape[0]` 索引 batch dim 0。图像 x 不置换 → 图 i 收到 perm[i] 的 pose, 正是 intended cross-image mismatch。

### b. 训练端 — OK
- `self.training` 守卫。eval 时 self.training=False → shuffle 跳过; LGPA test 路径喂真 scene_heatmaps。与 exp353 test 对称公平。

### c. 单变量 — OK
- config diff 仅 `POSE_SHUFFLE: True` + `OUTPUT_DIR`(diff 验证)。
- `POSE_SHUFFLE False` 真 no-op: forward 守卫短路; `__init__` 仅设 attr + print; `defaults.py` 默认 False。向后兼容, 不影响已有实验。

### d. 无附带损伤 — OK
- shuffle 到达 LGPA: shuffled scene_heatmaps → `_lgpa_heatmap` → `clip_part_head`。
- LGPA 是唯一活跃 pose consumer: `POSE_PSG_STAGES: []` → psg_modules_dict 空 ModuleDict, `_run_backbone_with_psg` 完全忽略 scene_heatmaps(patch_embed/prompt/PAA/PSG 全 off); GCN/PPA/VCSR/structural-routing 全默认 False 且 config 未设。backbone pose-free, shuffle 只扰 LGPA — 理想隔离。
- 顺序: shuffle 在 use_target_heatmap 替换前, 一致(且此处 use_target_heatmap False)。assign loss 内部自洽(clip_part_head 的 pose_bias attn 与 assign target 同源 shuffled lgpa_hm, 无 keypoint 泄漏)。

### e. kill-switch 有效性 — OK(含判读 caveat)
- cross-image pose = 有效 pose 但对错人 → 部位误定位, 正确测"pose-spatial causality"。
- **Medium(判读, 非 bug)**: ReID 裁剪空间对齐(人居中、头上脚下), cross-image pose 非随机噪声、blob 部分重叠真实布局 → NO-DROP 侧有两解(pose 无关 OR 裁剪对齐让乱 pose 还大致对)。DROP 侧干净。缓解: 不掉点则补 cross-PART(17关键点通道)shuffle 二次确认。
- Low: 固定点泄漏(~1/64 保留自身 pose, 后续 Codex derangement 修复); PK 同 ID 污染(~5%, 仍是不同实例 pose, 不软化)。

### f. 边界 — OK
- B=1 跳过(Bp>1); perm/arange 同 device; int64 比较有效; AMP 安全(纯索引)。

## 结论
审查通过。kill-switch 设计 sound, 结果可判读但不对称: 掉点(exp357<<60.5)=干净铁证(pose 空间正确性因果, 故事稳); 不掉(≈60.5)=被裁剪对齐混淆, 须 cross-part shuffle 二次确认才能下"pose 正确性无关"结论。可进 Codex 审查。
