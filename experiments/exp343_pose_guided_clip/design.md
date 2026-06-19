# 实验 exp343 (Option A): 姿态引导 CLIP 机制对齐的图像特征

## 动机（用户澄清）
用户要的不是「CLIP 机制 + 旁边一个独立 LGPA 分支」(exp342 的错)，而是**把姿态引导融进 CLIP 机制本身**：CLIP-ID-prompt 对齐的那个图像特征,从 raw GAP global 换成 **pose-guided 池化特征**。pose 引导「CLIP 机制对齐什么」。

## 核心假设
**i2t/t2i 对齐 ID 文本原型时,用的图像特征是 LGPA 式 pose-bias 池化的 pose-guided 特征(去遮挡器/背景),比对齐 raw global 更干净 → backbone 被塑造得更判别 → global 涨更多。**

## 技术方案
- 新模块 `PoseGuidedPool`(clip_id_prompt.py)：可学习 query attend backbone token,加 person pose 热图 additive bias → pose-guided 池化特征。
- forward：`POSE_CLIP_ID_POSE_GUIDED` 开时,`img_proj = clip_id_proj(pose_guided_pool(featmaps[-1], scene_heatmaps))`(替代 raw global);i2t/t2i 对齐它。pose_guided_pool 的参数(query, k_proj)经 i2t/t2i 梯度训练。
- = exp341 config + `POSE_CLIP_ID_POSE_GUIDED: True`。无 LGPA 独立分支。
- 测试描述子 = global(POSE_TEST_FEAT global);pose-guided 对齐塑造 backbone → 改善 global。

## 预期结果
exp343 global **> exp341 global(59.8)** → pose-guided 对齐比 raw-global 对齐塑造更好。哪怕 +0.3 就证明「姿态融进 CLIP 机制能涨」。

## 对照组
- **exp341(prompt 对齐 raw global, 59.8)vs exp343(prompt 对齐 pose-guided 特征)**。单变量 = 仅 POSE_CLIP_ID_POSE_GUIDED。
- baseline exp341base 57.6。

## 审查重点
PoseGuidedPool 正确(shape、pose bias、softmax、可训)；img_proj 用 pose-guided 特征时 clip_id_loss 正确；pose_guided_pool 参数进优化器；scene_heatmaps None 时 fallback global;test 端 prompt train-only。
