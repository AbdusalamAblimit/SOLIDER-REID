# 实验 exp324d_r32: LoRA-解冻 DINOv2-base rank32 + pose-anchored part-MaxSim

## 动机
- 同 exp324d_large：FM 方向 LoRA 解冻彻底测试，看能否突破冻结天花板（重遮挡 8.65 / 全部 14.61）。
- 本变体测**更大 adaptation 容量**：DINOv2-base（hidden 768）+ LoRA **rank32/alpha32**
  （base-rank16 在 lab-3090-d 跑着，本次翻倍 LoRA 容量，隔离"adaptation 容量"变量）。

## 核心假设
- 若 base-rank16 不够突破天花板，更大 LoRA 容量（rank32）能否让 base backbone 自适应更充分？
  rank16→32 LoRA 可训练参数从 ~0.59M 翻到 1.18M。

## 技术方案
- 同 exp324d_large 脚本（`scripts/exp324d_lora.py`），唯一区别：
  - `--dino_model facebook/dinov2-base`（hidden 768）
  - `--lora_rank 32 --lora_alpha 32`
  - micro_bs 32（峰值 2.81G，~2.6s/step，比 large 快 3x）
- 其余（损失/采样/eval/有效 BS=64/pool 缓存复用/30 epoch）与 large 变体完全相同。

## 超参
- DINOv2-base, hidden 768, LoRA rank32/alpha32, lora_lr 1e-4, head_lr 3.5e-4。
- micro_bs 32, grad_ckpt on, 有效 BS=64, 30 epoch, eval_period 5, cosine LR。

## 预期结果
- 假设成立：重遮挡 mAP 明显超过 base-rank16 与冻结 8.65。
- 失败最可能原因：同 large——5-part 表征上限低 / LoRA 解冻 q/v 不足以让 frozen DINO 判别化。

## 对照组
- 冻结基线 exp324b：重遮挡 8.65 / 全部 14.61。
- base-rank16（lab-3090-d，在跑）：隔离 LoRA rank 容量（16 vs 32）。
- 平行 exp324d_large（GPU0）：隔离 backbone 容量（base vs large）。
- 消融变量：LoRA rank 容量（32 vs 16），其余全同。
