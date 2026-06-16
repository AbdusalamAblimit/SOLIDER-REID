# Claude Broad Review — exp324d (LoRA-unfrozen DINOv2 + pose part-MaxSim)

**Reviewer**: Opus broad pre-training review (strict). **Date**: 2026-06-16.
**Scope**: `scripts/exp324d_lora.py`（新）, cross-checked against `scripts/exp324b_train_head.py`（对照）, `scripts/exp324_dino.py`（复用 geometry/eval）, `experiments/exp324d/design.md`。
**结论：审查通过（approve）。无 Critical/High/Medium，仅 4 条 Low 信息项，无需改动。**

## A. 可微池化正确性 — PASS
- `build_pool_matrix`(exp324d) cell-for-cell 复刻 `build_part_pose`(exp324_dino)：同 PART_GROUPS 循环、同 vb 跳过、同 (0,0) sentinel、同 int(round())、同 POOL_RADIUS=1 3×3 窗、同越界守卫。唯一变化：存 `1/len(cells)` 到 flat index `r*GRID_W+c` 而非 gather+mean。
- 审查者跑了 **200-trial 数值等价测试**：`bmm(pool_w, patch)` vs `build_part_pose` gather-mean → **max abs diff 1.79e-7**（float32 舍入），visibility flag 永远一致，invisible 行永远精确为 0。等价确认。
- flat-index/reshape 顺序正确：exp324 reshape 为 `(GRID_H,GRID_W,HIDDEN)` row-major，flat = `r*GRID_W+c`，exp324d 用同一映射。PASS。
- **梯度到达 LoRA**：train path 无 detach / 无 no_grad / 无 .numpy()；dry-run 经验确认（LoRA 294,912 参数更新，loss↓，acc 0.016→0.766）。PASS。

## B. 训练/测试对称 — PASS
eval `encode_split` 与 exp324b 完全相同的 pooling+head+normalization，唯一差异是 parts 来自 live DINO+pool 而非缓存（即预期单变量）。distmats/eval_func/heavy mask 全是同一 import 函数。PASS。

## C. 有效 BS=64 — PASS
`assert bs==64` 硬守卫。micro_bs 只切 DINO forward 再 `torch.cat` 拼回完整 64 → batch-hard triplet 见全 64 样本。每逻辑 batch 恰好一次 zero_grad/backward/step；micro-batch 是激活内存分块非梯度累积。PASS。

## D. 梯度检查点 — PASS
`use_reentrant=False` 是 frozen base+LoRA+非梯度图像输入下的正确设置（reentrant=True 会静默清零 LoRA 梯度）；dry-run 已验证。train/eval 模式正确（step 里 train，encode_split 里 eval+no_grad）。PASS。

## E. 优化器 — PASS
LoRA 参数（294,912，仅 LoRA require grad）+ head（decay/no-decay 分组，BN 排除 WD，冻结 bottleneck.bias 经 requires_grad 守卫跳过）。三组不相交无重复无遗漏。LoRA lr=1e-4 WD=0，head lr=3.5e-4。PASS。

## F. peft 使用 — PASS
target_modules=['query','value'] suffix 命中全 12 层 q/v；base 在 get_peft_model 前显式冻结，之后仅 LoRA require grad；无任何代码 re-freeze LoRA 或解冻 base；无 merge/disable_adapter。PASS。

## G. eval 正确性 — PASS
heavy-occ mask（vis≤8）、same-cam 排除（eval_func 内）、distmat（cosine + part-MaxSim）全不变。无 autocast，DINO float32，pool_w float32 → bmm dtype 一致；bn/pp `.float()` 后转 numpy。无 dtype mismatch。PASS。

## H. 单变量隔离 — PASS
唯一偏离 exp324b = LoRA 解冻 DINO（+ 机制必需的"缓存→live forward + 可微池化"）。head/损失/权重/采样/eval/distmat/heavy mask 全 import 复用。缓存目录(`exp324d/_cache`)+tag前缀(`_pool_`)与 exp324b 不冲突。干净单变量。

## I. 高度/创新 — ACCEPTABLE
确认机制（exp324b 证 pose part-MaxSim 可训但冻结特征受限于 14.61）上的 sanctioned 破天花板实验，非调参/镀金。设计有清晰 go/no-go。

## Low 信息项（不阻断，无需改）
- L1 d_ap/d_an 跨 step 标量平均（同 exp324b）。
- L2 save_pretrained 只存 LoRA adapter（peft 行为，小 ckpt，预期）。
- L3 默认 micro_bs16 + grad-ckpt 同时开（冗余省内存，不错；dry-run 2.35G）。
- L4 emb 变量与预分配一致，无 bug。

## VERDICT
**审查通过**（approve）。所有硬门通过：可微池化数值等价（1.79e-7）、梯度证实到达 LoRA、train/eval 对称、有效 BS=64 + 完整 batch-hard triplet + 单次 step、use_reentrant=False 正确且 dry-run 验证、优化器分组完整不相交 BN 排除 WD、peft 仅 LoRA 可训、eval 复用 exp324b/exp324 函数。LoRA 解冻 DINO 为相对 exp324b 唯一变量。
