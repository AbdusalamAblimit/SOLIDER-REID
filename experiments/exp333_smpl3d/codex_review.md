# Codex Review — exp333 SMPL-3D 辅助分支

**Date**: 2026-06-18
**Tool**: `codex exec -s read-only`（聚焦提示，输出到文件；长 prompt + tail 会卡，故短 prompt + 重定向文件）

## Round 1 — Verdict: needs-attention
findings（已全部修复）：
- **High1（A/B 非同 RNG 流）**：`set_seed` 在建模前调用；`--use_smpl` 多初始化几层 → 提前推进 Torch RNG → 训练随机流（数据序/dropout/drop-path）与 baseline 不同。**修复**：建模后**再 set_seed 一次**，两臂训练随机流完全一致，唯一差异=3D 分支。
- **High2（缓存目标人选择）**：`smpl_cache_occduke.py` 只按 ROMP 置信度 argmax 选人，多人遮挡 crop 可能选到遮挡人。**处理（不改缓存，附理由）**：Stage-A 用同一 argmax 选择已测得 β 跨摄像头 NN 身份 **4.6×** chance——若频繁选到遮挡人，β 不可能带目标身份信号。**经验上证伪该担忧**（ROMP 单人模式多数只返回 1 人，且 ReID crop 以目标为中心→置信最高即目标）。作为 limitation 记录；若结果边际再 re-cache 加 center-preference。
- **Med1（无效样本污染 3D 损失）**：valid=0 样本共享一个 `missing` 向量却有不同 pid → CE/triplet 矛盾梯度。**修复**：3D 损失只在 valid 检测样本上算（`_valid_triplet_mask`：只留 ≥2 valid 成员的 id，≥2 个这样的 id；保证 batch-hard 每个 anchor 有正样本、有负样本，否则跳过该 batch 的 3D 损失）。
- **Med2（BN 被零输入污染）**：`smpl_mlp` 的 BatchNorm 在替换 missing 前先吃了 valid=0 的零输入 → 污染 batch/running stats。**修复**：`smpl_mlp` 改 **LayerNorm（逐样本）**，删除 `bn3d`，eval 嵌入直接用 `_smpl_embed(smpl)`。
- **Low（stats 空）**：`compute_train_stats` 零检测会崩。**修复**：加 assert 守卫（train valid 93.3%，不会触发）。
- **No issue**：z-norm 仅用 train-valid 统计、eval 调 model.eval()、AMP 顺序正确、control 不建 3D 参数、融合归一+alpha 加权 honest。

## Round 2 — Verdict: approve（codex v2）
复核 4 项修复全部确认：reseed-after-model（RNG 流对齐）、3D 损失只在 valid 上算、LayerNorm 无 BN 污染、stats assert 守卫；并 reconfirm 单变量 A/B / AMP / 融合 honest / 无测试泄漏。
**但 v2 漏判一个真 bug**：`_valid_triplet_mask`（变长 valid 子集）与仓库 batch-hard `hard_example_mining` 的 `view(N,-1)` 不兼容（要求每 anchor 同样多正样本）→ smoke 实测崩 `shape '[61,-1]' invalid for input of size 235`。**我经 smoke 实测抓到并修复**：改 `_balanced_valid_mask`——只保留**全部实例都 valid 的 id**（RandomIdentitySampler K=4 → 保留子集仍是平衡 P×K），既排除无检测样本又满足 hard-mining 平衡。

## Round 3 — Verdict: approve（codex v3，复核 _balanced_valid_mask 最终代码）
确认：(a) 保留整 id → 子集平衡 P×K → hard_example_mining `view(N,-1)` 安全；(b) valid=0 样本完全排除出 3D 损失；(c) <2 id 存活则跳过 3D 损失、外观损失照常；(d) 无回归。并 reconfirm 单变量 A/B / reseed / LayerNorm / 融合 honest。
**smoke3 实测**：Epoch[1] app+3d 双损失正常、5 个 alpha eval 全跑通、done.（1-epoch 数字无意义，仅验流程）。

## 结论
codex 审查通过（v2+v3 approve），且 smoke 实测发现并修复了 codex 漏判的 batch-balance 崩溃。最终代码经三轮审查 + 端到端 smoke 验证，可训练。
