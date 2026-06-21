# 实验 exp357: pose-shuffle kill-switch (LGPA-pose 论文地基验证)

## 动机
8-路 codex 讨论结论: "语义无关"洞察单独发不了 B 类, 唯一翻盘路 = #1 query-invariance 机制。但**整个故事赌一个前提: "pose 的空间内容是因果的"**。codex red-team(file00/07)最强警告: **若 pose-shuffle 训练不掉点, "pose spatial prior 是价值"的故事当场塌**——那 LGPA 增益其实只是"有个部位结构 + 部位监督", 与 pose 正确定位无关。

**先花 ~2h 排这个雷, 再决定要不要全建 #1。** 学 PC-MSC 教训(弱 kill-switch 早警示)。

## 核心假设
LGPA 的增益依赖 pose 把部位**正确定位**到该人体区域。若用**错误的 pose**(别人的 pose)训练, 部位被误定位 → 应该掉点。

## 方法
基于 exp353(un-detach LGPA, 无 CLIP, random query, = 60.5 mAP, 当前最干净的纯 pose 部位机制)。
新增 flag `POSE_SHUFFLE`: 训练时把每个 batch 的 pose(scene_heatmaps + target_heatmaps)沿 batch 维**随机置换**(cross-image: A 图用 B 图的真实 pose——有效 pose 分布, 但对错人)。部位监督结构不变, 只破坏 pose 与图像的对应。

## 判读(决定地基)
- exp357(shuffle)**明显掉点**(< 60.5, 比如 < 59) → **pose 空间内容是因果的** → 地基稳 → 全建 #1 query-invariance。
- exp357 ≈ exp353 60.5 → **pose 正确定位无所谓**, 只是"部位结构 + 监督"在涨 → "Pose, Not Prompt"故事塌(但揭示更极简机制: 部位监督才是价值) → 撤或重新 framing。

## 对照
exp353(真 pose, 60.5)vs exp357(shuffle pose)。单变量 = POSE_SHUFFLE。

## 实现
- `config/defaults.py`: POSE_SHUFFLE(bool, default False)。
- `model/pose_backbone_model.py`: forward 里 _prepare_pose 后, 若 self.training and use_pose_shuffle, 用一个非恒等随机置换 perm 重排 scene_heatmaps + target_heatmaps(同一 perm 保持配对)。仅训练端; 测试不 shuffle。
- config: exp357_pose_shuffle.yml = exp353 + POSE_SHUFFLE True。

## 审查重点
- 置换非恒等(perm != arange)且 scene/target 用同一 perm。
- 仅训练端 shuffle(self.training 守卫); 测试用真 pose(否则不可比)。
- 单变量 vs exp353(仅 POSE_SHUFFLE)。
- 不影响其他 pose 用途(exp353 无 PSG, pose 只进 LGPA)。

## 状态
设计完成 → 实现 → 双审 → 训练(3090/4090 空闲)→ 判地基。

## Codex 修复 (2026-06-21)
- Medium-1: randperm 留固定点(~1/64 图保留自己 pose)→ 改 **derangement**(re-roll 至无固定点, fallback cyclic shift), 每张图都用别人 pose。
- Medium-2(判读): NO-DROP 侧被裁剪对齐混淆(别人 pose 仍带粗糙 canonical 头/躯干/腿先验)。Codex/Claude 一致: 掉点=干净铁证(图特定 pose correspondence 重要); 不掉=只能说"精确图特定 pose 在对齐裁剪下非必需", 需补 **cross-PART(17关键点通道)shuffle** 二次确认(测解剖通道身份是否重要, 同图空间 support 不变)。最佳矩阵: cross-image + per-image channel-shuffle + no-pose/fixed-canonical control。
