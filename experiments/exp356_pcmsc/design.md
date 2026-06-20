# 实验 exp356: PC-MSC (Pose-Conditioned Masked Semantic Completion)

## 动机
20-codex #3, 机制上离今晚所有失败最远: **不"融合"、是"补全"**。pose mask 掉**可见**部位的 backbone token, 让模型从可见证据**重建**那块的 CLIP 语义——被删的 token 吸不走(从输入删了), 是 CLIP(单图全局)和 LGPA(部位判别)都没有的补全能力。

## 核心机制
1. **Frozen CLIP ViT**(保留 clip_model.visual)在(重预处理的)输入图上 → dense patch tokens → 按 pose 部位池化 → 每部位 CLIP 语义 `C_part (B,K,768)` = **补全目标(冻结, 无梯度)**。
2. SOLIDER backbone → feature map F + global。
3. pose 选一个**可见**部位 p*(高 pose 激活), 在 F 里 mask 掉它的 token(置可学习 mask token)→ F_masked。
4. **轻量 decoder**(部位 query cross-attend F_masked 的可见 token, 1 层)→ 重建 p* 的特征 R (B,768)。
5. **Loss**: `L_msc = 1 − cos(R, C_part[p*])`(从可见证据重建被删部位的 CLIP 语义)。
6. 总 loss = L_clipreid(exp341 i2t/t2i, 不动)+ λ_msc · L_msc。**测试端描述子还是 global**(L_msc 是训练端正则)。

## 为什么避开今晚所有坑
- **不被吸收**: 目标是**被删 token 的 clean CLIP 语义**, global 参数通路看不到(从输入删了), 吸不走(A/C 吸收是因目标在梯度可达通路上)。
- **不冗余**: 遮挡补全是 CLIP 和 LGPA 都没有的能力(不是再学一遍 ID)。
- **非禁忌**: 不是 visibility gate(禁忌#1), 不是 retrieval scorer, 不进 CLIP 对齐。

## ★ 廉价 kill-switch(全建前必做, 学 PC-SOR 教训)
**冻结 CLIP ViT 的每部位特征 C_part 到底带不带 ID 信息?** 取多 ID 多实例 Occluded-Duke 图, 算每部位 CLIP 特征, 看 same-ID 部位特征是否比 diff-ID 近(ID-判别)。
- 若 C_part 是垃圾(same-ID ≈ diff-ID, 不带 ID)→ 重建它对 ReID 无意义 → **直接 kill**。
- 若 C_part 带 ID(same-ID 明显近)→ 重建目标有意义 → 全建训练。

## 复杂点(诚实标注)
- CLIP 预处理: SOLIDER 输入(384×128, 0.5 norm)→ un-norm → CLIP norm → resize 224(畸变, 但 pose 热图同步 resize 一致)。
- 网格对齐: pose 热图(96×32)interp 到 CLIP 16×16 网格池化 CLIP token。
- CLIP ViT frozen forward 每 iter(加计算, 但冻结无梯度)。

## 对照/消融
- baseline **exp341 59.8**。单变量 = PC-MSC 开关。
- **★必做控制 exp356r: random-block mask**(不按 pose 随机 mask 部位)→ 隔离 pose-guided mask 的价值。若 exp356 ≈ exp356r 则 pose 无贡献(只是 generic MIM 正则)。
- ablation: 重建 CLIP 特征 vs 重建 SOLIDER 自身特征(隔离"CLIP 语义目标"价值, 后者=PersonMAE 类已知)。

## 实现文件(全建阶段)
- `model/modules/clip_id_prompt.py`: 保留 clip_model.visual + `dense_tokens(img)` 方法。
- `model/pose_backbone_model.py`: __init__ 加 mask token + decoder + flags; forward 加 PC-MSC 块(CLIP 目标/mask/重建/loss)。
- `config/defaults.py`: POSE_PCMSC(bool), POSE_PCMSC_W, POSE_PCMSC_RANDOM_MASK(控制)。
- config: exp356_pcmsc.yml = exp341 + POSE_PCMSC True。

## 审查重点(全建阶段)
- CLIP 预处理 un-norm/re-norm/resize 正确; CLIP ViT 真冻结(无梯度, 不进优化器)。
- mask 只 mask 可见部位(重建有意义); mask token 可学习。
- 重建目标 C_part detach(冻结)。
- 网格对齐(pose 热图 interp 到 CLIP grid)正确。
- L_msc 不进测试; 描述子仍 global; 单变量 vs exp341。
- AMP 安全(CLIP fp16 forward / cos 数值)。

## 状态
设计完成 → 先跑 kill-switch(CLIP 每部位特征带不带 ID)→ 通过则全建+双审+训练。

## ★ Kill-switch 结果 (2026-06-21): 弱通过
CLIP 每部位特征(8 ID×4 实例)same-ID vs diff-ID gap: GLOBAL +0.022, head +0.011, torso +0.009, legs +0.013。
**带 ID(gap 全正, 非 PC-SOR 式垃圾), 但偏弱(部位 gap≈global 一半; 绝对 sim~0.93 = CLIP 各向异性)。** 目标有意义但弱 → 预期温和信号, 非清晰涨/死。决定: 继续全建(用户要全协议 + 非清晰 kill), 经验训练定论。
