# 实验 exp337: Swin 纯 LGPA-D + 不注入 pose（ablation vs exp336）

## 动机
- exp336 证：纯 LGPA-D 在 Swin 上(无 PSG/OASD/aug)给 global +1.7 mAP(e60)→ CLIP 模块有 standalone 价值。
- 但 LGPA-D 里 pose 通过三条注入：① pose-bias(attention bias 引导部位 attend 对应区) ② assign 损失(KL 监督注意力匹配 pose) ③ visibility(pose 响应加权池化)。
- **问题**：+1.7 是来自 **CLIP 文本部位语义本身**，还是 **pose 注入**？用户要的 ablation。

## 核心假设
- 若 exp337(无 pose)仍 +X(≈+1.7)→ 增益来自 **CLIP 文本语义本身**，pose 不必要 → step2 可纯做 CLIP 文本/视觉，不依赖 pose。
- 若 exp337 ~0(equalcat≈global)→ 增益靠 **pose 注入**(CLIP 文本只是占位，pose-bias 才是 localizer)→ step2 的 CLIP 接法需保留 pose 引导。

## 技术方案（小代码改动 + config）
- 新 flag `POSE_LGPA_NO_POSE`（config/defaults.py:223，默认 False）。
- pose_backbone_model.py：__init__ 读 `self._lgpa_no_pose`；train(603) + eval(800) 两处调用 `lgpa_hm = None if self._lgpa_no_pose else scene_heatmaps`，传 `lgpa_hm`。
- CLIPPartHead 收 heatmaps=None → pose_bias=None(无 attention bias)、visibility=uniform(1/5 均匀池化)、assign_loss=0。= 纯 CLIP 文本部位原型 cross-attend tokens(无 pose)。
- 已 build 验证：`[LGPA] NO-POSE ablation: heatmaps=None` 生效，no_pose=True，psg=0。
- config exp337 = exp336 完全相同 + `POSE_LGPA_NO_POSE: True`（单变量）。

## 判据（同 exp336，同 ckpt 两描述子）
- `test.py POSE_TEST_FEAT=equal_concat`(LGPA-no-pose) vs `=global`(baseline)。只改 POSE_TEST_FEAT，不设 POSE_LGPA=False。
- **训练 log `lgpa_assign` 应 = 0**（无 pose → 无 assign）——这是 no-pose 的 sanity（对照 exp336 的 ~7）。
- equalcat − global：≈+1.7→CLIP 语义本身；~0→靠 pose 注入。

## 对照
- exp336（有 pose,+1.7）vs exp337（无 pose,?）。单变量 = POSE_LGPA_NO_POSE。

## 机器
4090（Clash 修复 PROCESS-NAME,tailscaled,DIRECT 后复活）并行跑,mmpose-abu env,29s/epoch（快于 3090）。

## ✅ 结果（决定性）
| | equalcat | global | 增益 |
|---|---|---|---|
| exp336 **有 pose** | 59.6 | 58.5 | **+1.1** |
| exp337 **无 pose** | 58.7 | 58.8 | **≈0（−0.1）** |
- sanity：exp337 全程 `lgpa_assign=0`（无 pose 注入）✓。
- **结论：+1.1 增益完全来自 pose 注入，不是 CLIP 文本语义本身。** 纯 CLIP 文本原型 cross-attend token（无 pose-bias）→ 部位对 global 零贡献。CLIP 文本"head/torso/legs"自己定位不出判别性部位；是 pose-bias 引导注意力到对的身体区域才让部位有用。**CLIP 文本只是 query 壳，pose 是增益驱动。**
- **step2 启示**：纯 CLIP 文本路线无效（语义冗余于 global）。新 CLIP 接法要么保留 pose 引导，要么换带 global 没有的新信息的 CLIP 信号（CLIP 视觉特征 / 遮挡推理 / ID 原型），不能靠 CLIP 文本部位语义本身。
- 待补 ablation（可选）：pose-biased + **无 CLIP 文本**（学习 query）→ 测 CLIP 文本是否比学习 query 有任何增量。
