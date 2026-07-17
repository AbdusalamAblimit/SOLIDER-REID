# 实验 exp342: Step2 — 姿态(LGPA 式)注入能涨的 CLIP 机制,再涨

## 动机
- Step1（exp341）成功：CLIP-ReID 可学习 ID prompt 在 SOLIDER 上 **+2.2**（59.8 vs 57.6 matched）。找到了能涨的 CLIP 机制。
- Step2（用户路线）：把姿态像 LGPA 那样注入这个能涨的机制 → 进一步涨。**理想叙事**：baseline 57.6 → +CLIP prompt 59.8 → +pose ~60.6。

## 核心假设
**在 exp341（CLIP-ID-prompt 改善的 global）之上，加回 LGPA 姿态部位分支（detached, pose-guided part pooling，已证 +0.8~0.9 的真姿态机制）→ equal_concat（prompt-global + pose-parts）> exp341 global。**

## 技术方案
- exp342 = exp341 config **仅多**：`POSE_LGPA: True`（重开姿态部位分支，detached）+ `POSE_TEST_FEAT: equal_concat`（描述子 = global + 部位）。
- **模型修复（codex Medium）**：`pose_backbone_model.py` LGPA 分支 `kp_data = lgpa_data` 后注入 `kp_data['clip_id_loss'] = clip_id_loss`，使 CLIP-ID-prompt 损失在开 LGPA 时仍回传（之前只在 no-part 默认路径返回）。
- 两机制：CLIP-ID-prompt（ID 级，i2t/t2i 正则 global）+ LGPA（部位级，pose 池化）。prompt 改善 backbone-global，LGPA 在 detached 特征上做部位（不扰 backbone）。

## 预期结果
- exp342 global（prompt）≈ 59.8；exp342 equal_concat（+pose parts）> 59.8（pose 在 prompt 之上再加 +0.8~0.9）。
- 完整链：baseline 57.6 → +CLIP 59.8（+2.2）→ +pose ~60.6（再 +0.8）。
- 失败可能：两机制干扰（prompt 改了 backbone，LGPA 的 pose 池化在新 backbone 上失效）；或 equal_concat 的 global 部分被 1.0-scale 略降。

## 对照组
- **exp341（prompt only, global 59.8）vs exp342（prompt + pose, equal_concat）**。单变量 = 仅 POSE_LGPA on/off + 描述子。
- 也可看 exp342 自身：global（prompt）vs equal_concat（+pose）隔离 pose 贡献。

## 审查重点
clip_id_loss 走 LGPA 路径是否正确、不重复计；两机制损失（i2t/t2i + LGPA assign + ID + triplet）是否都正确累加；equal_concat 描述子组装；单变量隔离 vs exp341。

## 审查提醒 M1（Claude，重要）
开 LGPA 后模型返回 list → loss 走 list-path，global 有效权重变 **0.5x**（w_g=1/(1+POSE_PART_WEIGHT)=0.5），非 exp341 的 1.0x。这是标准 LGPA 训练 regime（+0.8~0.9 就在此测）。
**正确解读 Step2 pose 增益**：用 **exp342 自己的 global（prompt, 0.5x）vs equal_concat（+pose, 0.5x）**，同尺度干净隔离 pose 贡献。Step1 的 prompt +2.2 是 1.0x（exp341），两步尺度不同但各自内部干净。记两个数。

## 变体 (打破冗余尝试, 配置级 ablation, 代码已审)
- **exp342b** POSE_LGPA_DETACH False: 姿态去塑造 backbone (部位有独立 id_part/tri_part 监督, 不进 CLIP 对齐, 区别于 A/B/C)。假设: 姿态塑造 backbone 的部位判别 = CLIP 全局判别的互补, 打破"只当外挂"的冗余。
- **exp342c** GLOBAL_LOSS_SCALE 2.0: 抵消 LGPA list-path 的 0.5x, 让 global 实际 1.0x (M1 修正), 看干净 global + pose 是否清噪声。

## 变体 exp350: exp342b(un-detach) + clean global 2.0x
exp342b global 掉到 58.8(un-detach 竞争纯ID)。exp350 = exp342b + GLOBAL_LOSS_SCALE 2.0(让 global 实际 1.0x,保护纯ID对齐)→ 期望 global 回到 ~59.8 + 部位 → equal_concat 更高(>60.7)。

## 本变体: exp342b un-detach + GLOBAL_LOSS_SCALE 2.0 (clean global)
配置级 ablation, 代码与父实验 exp342_clip_id_pose 相同(已双审查通过), 仅改 config flag。
