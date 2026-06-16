# exp324d 监控 — LoRA 解冻 DINOv2-base + 姿态部位匹配（破 14-mAP 天花板）

机器：lab-3090-d（RTX 3090 24G, py3.10.12, torch 2.7.1+cu118, peft 0.19.1）
脚本：`scripts/exp324d_lora.py`
对照：exp324b 冻结头 e60 = part 全部 14.61 / part 重遮挡 8.65 / cos 全部 13.51 / cos 重遮挡 7.32
变量：唯一 = LoRA 解冻 DINO（q/v rank8 alpha16），头/损失/采样/eval 全同 exp324b。

## 配置（全量，2026-06-16 按用户严格设计修正）
- LoRA: q/v, **rank 16, alpha 16, dropout 0.05**, lr 1e-4（用户设计明确 r=16/dropout 0.05）
- head: shared Linear 768→512 + BNNeck + 全局分类器 + part 分类器, lr 3.5e-4 (Adam)
- 损失: 全局 ID CE + soft-margin batch-hard triplet + part_weight 0.5 per-part ID CE
- 采样: PK P16×K4=64（有效 BS=64 硬约束）
- **micro_bs 16 + gradient_checkpointing ON**（设计要求开 ckpt；峰值 2.39G/24G）
- **epochs 30**, cosine LR(T_max=30), eval_period 5, ckpt_period 10
- LoRA params=**589,824**（r16，≈ r8 的 2 倍，确认 r=16 生效）；head params=1,112,576（702 类，classifier+part_classifier 随类数缩放）

> 注：早先一个 prior-session 启动的是 rank8/dropout0/no-ckpt/35ep 的旧配置（与用户本次严格设计不符），
> 已 kill 并按设计重启 rank16 配置。重启时发现一个 setsid 残留的 rank8 进程在抢同一 GPU+同一
> `/tmp/exp324d.log`（一次性 leftover，非 cron/loop），已一并清除；现单进程跑，日志改用专用
> `/tmp/exp324d_r16.log` 避免冲突。

## Dry-run 验证（设计配置，limit_train 400, 5 steps）
- rank16: LoRA 589,824 params + head（19 类时 413,184）均训练
- loss 12.14→9.63 ↓, id 2.95→2.68, tri 7.72→5.62, part 2.94→2.65, acc 0.047→0.797 ↑ → 梯度确实回流 LoRA + 头
- 显存：micro_bs16 + grad-ckpt **2.39G**（24G 富余），~2.0s/step
- 可微池化数值等价已独立复验：bmm(pool_w,patch) vs build_part_pose max abs diff **1.9e-6**，vis 掩码完全一致
- 预计：~244 step/epoch × ~2s ≈ 8min/epoch（带 ckpt）→ 30ep ≈ 4h + eval

## 审查
- Claude broad review：审查通过（含 200-trial 可微池化等价测试 max diff 1.79e-7）。见 claude_review.md
- Codex review：approve（141,797 tokens，独立确认池化等价/梯度到 LoRA/full-batch triplet/use_reentrant=False/eval 对称/dtype；组合 plausibly new，未找直接先例）。见 codex_review.md

## 启动（2026-06-16，rank16 设计配置）
- lab-3090-d PID 309591，日志 `/tmp/exp324d_r16.log`，OUTPUT_DIR `log/occluded_duke/exp324d`
- prep 阶段：train/query/gallery 姿态池化矩阵已全部缓存（`experiments/exp324d/_cache/*_pool_*.npz`），重启秒加载
- 命令：`python3 scripts/exp324d_lora.py --epochs 30 --lora_rank 16 --lora_alpha 16 --lora_dropout 0.05 --lora_lr 1e-4 --head_lr 3.5e-4 --part_weight 0.5 --micro_bs 16 --eval_period 5 --ckpt_period 10 --output_dir log/occluded_duke/exp324d`
- 启动后 GPU 4154 MiB / 100% util，单进程，正常训练中

## 训练轨迹
（每 5 epoch eval 事件追加）

| epoch | loss | id | tri | part | acc | cos ALL mAP/R1 | part ALL mAP/R1 | cos HEAVY | part HEAVY |
|-------|------|-----|-----|------|-----|----------------|-----------------|-----------|------------|
| 1 | 10.36 | 5.48 | 1.93 | 5.89 | 0.409 | — | — | — | — |
| 2 | 4.63 | 2.70 | 0.71 | 2.43 | 0.831 | — | — | — | — |

### [Epoch 1-2] 检查点
- e1 健康：acc 0.409（702 类，单 epoch 强起步），id 5.48 / part 5.89 主导，tri 1.93（soft-margin，d_ap≈d_an=23.2 早期未分开正常）。
- **e2 强正信号**：loss 10.36→4.63，acc 0.409→**0.831**；**d_ap=30.9 / d_an=36.5**（正负已分开 Δ5.6，e1 时几乎相等）→ embedding 快速变 ReID-判别。LoRA 让 DINO 适应 ReID 的迹象明确。
- 502-513s/epoch（grad-ckpt + micro_bs16）→ 30ep ≈ 4.2h，首 eval @ e5（~42min）。
- **关键里程碑：清过 Epoch[1]**（前两次启动均在 Epoch[1] 前因 session teardown 被 kill；本次 setsid 独立 session 存活）。
