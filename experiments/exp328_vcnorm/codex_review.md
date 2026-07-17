# Codex Review — exp328 VC-Norm

**Verdict (前一轮 codex)**: needs-attention
**Date**: 2026-06-17
**Review round**: 第 1 轮 codex 提出 → 本文件记录修复（待第 2 轮 codex 复审）

> 本文件记录 codex `--search exec` 第 1 轮提出的 findings 以及每条的修复方式与验证。
> 修复后已在 lab-3090-d 带 CUDA 重跑 dryrun（含新增的 High-1 梯度路由单测）全部 PASS。
> 等用户重跑 codex 复审改 verdict 为 approve 后训练。

---

## Findings（codex 第 1 轮）+ 修复

### High-1 [核心机制 bug] — student 矩用 visibility score 加权，被遮挡 token 不进统计

**原问题**（`loss/vcnorm_loss.py:87`）：student 矩 `s_w = s_sc`（visibility score）加权。
PLBOA 把被遮挡 keypoint 的 score 置 0（`datasets/pose_dataset.py:871-874`、
`model/modules/skeleton_gcn.py:667-668` kp_weight_mode='score' 直接返回 scores），于是
被遮挡 student token 权重≈0、几乎不进 batch mean/var、拿不到 VCA 梯度。VCA 实际只对齐
本来就可见的 token（无 domain shift），机制空转、与目标相反。

**我对 PLBOA token 处理的判断（关键）**：
- PLBOA 在**数据层**对被遮挡 keypoint 做三件事：① 图像区域 gray-fill / 贴真实 occluder
  （`pose_dataset.py:824-862`）；② score/visibility/visibility_binary 置 0（871-873）；
  ③ **heatmap** 置 0（874）。
- 但 **GCN token 本身不是 0-mask**：**keypoint 坐标 `kp[:,0/1]` 被保留**（PLBOA 只用坐标
  判 `in_occ`，从不改坐标）。本 config 走标准 bilinear 采样
  `_sample_keypoint_features`（`skeleton_gcn.py:553-557`，非 DPF heatmap-pool 路），仍在被遮挡
  像素位置 `grid_sample` 采到一个**退化但真实的"遮挡特征"**——backbone 对 occluder/gray 像素的响应。
- 这个退化 token **正是探针指认、要被拉向 clean teacher 的 domain-shifted 统计**。所以
  **不需要换接入点到 pre-mask**：就地对齐即可，被遮挡 token 的特征是有意义的。
- （注：若日后切到 DPF heatmap-pool 路，heatmap=0 会让那条路的 token 退化为采样 fallback；
  本 config 用点采样，不受影响。已在 design.md 注明。）

**修复**（`loss/vcnorm_loss.py` 重写 `vcnorm_align_loss`）：
- student 矩改为只在 **"被遮挡且 teacher 可见"子集**上估计：
  `s_w = (s_occ & t_vis) * t_sc`，其中 `s_occ = (s_sc < vis_thr)`、`t_vis = (t_sc >= vis_thr)`。
  权重用 **teacher score**（student score≈0 不能当权重）。
- teacher 矩仍在 teacher-visible 上估计（clean 目标分布），detach。
- 这样**被遮挡的 student token（teacher 可见、student score 低/0）真正进入统计并被拉向 clean teacher**，
  与机制目标一致。可见 student token 故意排除（它们没漂移，对齐是 no-op）。

**验证**（dryrun 单测 [5]，新加）：构造 k0..5 student 遮挡(s_sc=0)+teacher 可见(0.9)、k6..16 双可见。
反传后 `occluded-token grad sum=0.96 > 0`、`both-visible grad sum=0.0`、`valid_k=6`（恰好遮挡 cohort）。
→ **被遮挡 token 确实进了统计并拿到梯度，可见 token 不拿——High-1 修复成立**。

### High-2 [单变量对照基线]

**原问题**：当前 yml（Swin-Base + PLBOA=True + VCNORM=True）不能直接对 exp260b 88.0
（那是 PLBOA=False）。

**修复**：新建 `configs/market/pose_vcnorm_base_control.yml`——与实验组逐行相同，
**唯一差异 `POSE_VCNORM=False`**（PLBOA 仍 True），OUTPUT_DIR=`exp328_vcnorm_control`。
远程实测两 config 加载：exp(VCNORM=True/OA_SD=True/PLBOA=True)、ctrl(VCNORM=False/OA_SD=True/PLBOA=True)，
仅 VCNORM 不同。design.md §对照组已写明：VC-Norm 净增益**只能 vs 这条对照，不能 vs 88.0**。

### Medium-a [config guard] — VCNORM=True 但 OA_SD=False 静默无对齐

**原问题**：VCA 嵌在 OA-SD 分支内（`processor.py:783` 的 `if oa_sd_enabled ...`），
POSE_VCNORM=True 但 OA_SD=False 时 VCA 块根本不进入、静默跳过（伪装成"VC-Norm 跑了"）。

**修复**（`processor/processor.py`，EMA teacher 创建段后）：加启动期 assert——
`POSE_VCNORM=True` 必须 `POSE_OA_SD=True`，否则报错并指明原因（VCA 消费 OA-SD EMA teacher 的
per-keypoint token）。另加 PLBOA 关闭时的 warning（遮挡 cohort 会塌成 ~0，VCA 变 no-op、valid_k=0）。

### Medium-b [student-side min-count gate]

**原问题**：`valid_k` 只看 teacher weight；某 keypoint 全 batch 被遮挡时 student 矩退化为噪声，
仍可能因 teacher 可见被对齐 → 噪声梯度。

**修复**：`_weighted_moments` 同时返回 `s_wsum`；`valid_k = (t_wsum>=min_weight) & (s_wsum>=min_weight)`
——双侧 min-count gate，student-occluded cohort 在本 batch 没足够样本的 keypoint 跳过。
dryrun [3] 全遮挡 teacher → valid_k=0、零损失保图、grad finite。

### Medium-c [backbone 不更新，文档校正]

**原问题**：GCN 输入 `featmaps[-1].detach()`（`pose_backbone_model.py:608`），VCA 只训
GCN/VCN/head，**不更新 backbone**；design.md 原文却写"反传更新 backbone/GCN"。

**修复**：design.md §数据流第 5 步改为"**不更新 backbone**：GCN 分支输入 detach，VCA/GCN 只更新
GCN+VCN+skeleton head；backbone 经此路径无梯度（PSG/主 ID·triplet 仍更新 backbone，但那不是 VCA 贡献）"。

### Low [文档一致性] — design.md §config 漏 GAIN_SCALE

**修复**：design.md §config 代码块补 `POSE_VCNORM_GAIN_SCALE`（共 7 key，与 defaults.py:315-321 一致）。

---

## dryrun 输出（lab-3090-d, CUDA, torch 1.13.1+cu117）

```
[dryrun] device=cuda, B=8 K=17 C=1024
[1] zero-init identity: max|out-in|=0.000e+00  gain_abs=0.000e+00 shift_abs=0.000e+00
    PASS: VCN zero-init == identity (VCNORM_MODULE=True untrained == baseline)
[2] align loss=0.6233 stats={'vca_loss': 0.623, 'vca_valid_k': 14.0, 'vca_mean_dist': 0.412,
    'vca_std_dist': 0.211, 'vca_occ_ratio': 0.544}
    PASS: finite loss, valid_k>0, grad to student only, teacher detached
    student==teacher align loss=0.000e+00 valid_k=6.0
    PASS: align loss ~0 when student==teacher on the occluded cohort
[3] all-occluded-teacher loss=0.000e+00 valid_k=0.0 grad_finite=True
    PASS: graceful zero-loss + finite grad when no valid teacher keypoint
[4] AMP: vcn out dtype=torch.float32, align loss dtype=torch.float32, finite=True
    PASS: AMP-safe (dtype preserved, finite loss)
[5] occluded-token grad sum=9.6335e-01  visible-token grad sum=0.0000e+00
    valid_k=6.0  occ_ratio=0.353
    PASS: occluded student tokens get gradient, both-visible tokens do NOT (High-1 fixed)

[dryrun] ALL UNIT CHECKS PASSED
```

- zero-init 恒等仍成立（baseline-safe）✓
- 新 occluded-subset 对齐让被遮挡 token 拿到梯度、可见 token 不拿 ✓（单测 [5] 显式验证）
- align loss finite ✓；AMP（CUDA float16 autocast）实跑、float32 矩、finite ✓（本轮在远程实跑，不再 SKIP）
- 配置加载：exp / control 两 config 仅 VCNORM 差异 ✓

## 结论

codex 第 1 轮 needs-attention 的 High-1/High-2/Medium-a/b/c/Low 全部已修，dryrun（含新增 High-1
梯度路由单测）在 lab-3090-d CUDA 全 PASS。本地 + lab-3090-d 文件 checksum 一致。
**待用户重跑 codex `--search exec` 复审改 verdict→approve 后方可训练**（本任务只到 dry-run，不正式训练）。
