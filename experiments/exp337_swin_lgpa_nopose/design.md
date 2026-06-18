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
4090 下线(relay 挂)/ hyy 不解析 → 待 3090 空(exp336 e120 后)或 4090 复活。Swin-Tiny 384 ~2-2.5h。
