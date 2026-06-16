# Codex Review — exp324i

**Verdict**: approve
**Date**: 2026-06-16 03:1x
**Review round**: 1（一次通过）
**Tool**: `codex --search exec -s read-only`（GPT-5.5，联网核查新颖性）

## Findings（codex 原文，均 Low，非阻断）

- **Low** — `scripts/exp324i_lora_decorr.py:247` decorr_loss：前向 `+eps` 已防 0/0，但 `torch.std()` 在某维 batch 内恒定（var=0）时**反向**仍有 NaN 边缘风险；更稳的写法是 variance + clamp/rsqrt。
- **Low** — `scripts/exp324i_swin_cache.py:79` 缓存维度由 config 推断（`base`→1024 否则 768），未断言必为 exp255-small/768；给定命令下正确，但错配 config/weight 会**静默**改变实验对象（trainer 接受任意 Dswin）。

## Checks passed（codex 确认）
- diff 是 surgical：仅加 cache loader + decorr_loss + 2 args + gated cache load + train-step term + logging；**eval 路径未动**。
- `decorr_weight==0` 经 `use_decorr` 完全跳过 cache 加载（line 365/384）→ 数值等价 exp324d。
- 名字对齐按 basename，缺键 `raise KeyError` fail-loud（line 160/166）。
- `s` detached、float32、移到 CUDA、与训练 idx 同序索引；`decorr_loss(glob, s)` 梯度经 `glob` 到 head + LoRA。
- cache 脚本用 val-style train loader、shuffle=False、no flip/aug、model.eval、单次前向。

## Novelty（联网）
Barlow Twins（网络内冗余消除）、VI-ReID 的 Shape-Erased orthogonal feature learning 是相邻先例；**未找到** "adapted DINOv2/LoRA 与 frozen Swin SOTA 之间做跨网络跨协方差解相关以强制互补 for occluded ReID" 的直接先例。

## 已修（两个 Low 都修了，虽非阻断）
1. decorr_loss 改 `(x-mean)*rsqrt(var+eps)`（population var + rsqrt，恒定维反向不再 NaN）。
2. swin_cache 加 `--expect_dim`（默认 768）并断言 `D==expect_dim`，错配 config 立即 fail-loud。

## 结论
codex 审查通过（verdict approve）。两个 Low 已加固修复。
