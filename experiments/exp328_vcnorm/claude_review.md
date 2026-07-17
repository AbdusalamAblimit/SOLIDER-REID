# Claude Broad Review — exp328 VC-Norm

**审查者**: Claude (Opus 4.8, 1M ctx) 主 agent 直接逐行审
**日期**: 2026-06-17
**轮次**: 第 1 轮（这是另一个 agent 中途 API 崩溃留下的未审代码，按"宁可误报"逐行审）
**范围**: design.md / vcnorm.py / vcnorm_loss.py / skeleton_gcn.py wiring / pose_backbone_model.py wiring /
processor.py wiring / config(defaults.py + pose_vcnorm_base.yml) / dryrun_unit.py / 优化器 / train-test 对称 / AMP / 与 exp260b 复现性交互

---

## 0. 实跑验证（不只读代码）

- 本地（Mac, CPU）实际跑了 `experiments/exp328_vcnorm/dryrun_unit.py`：`[dryrun] ALL UNIT CHECKS PASSED`。
  zero-init identity = 0.000e+00（恒等真成立）；align loss 有限、grad 只到 student、teacher.grad is None；
  student==teacher → loss=0；全遮挡 teacher → valid_k=0、loss=0、grad finite。这些 load-bearing 正确性主张**真实成立**。
- ⚠️ **AMP 分支在本地被 SKIP（无 CUDA）**，所以 float16 autocast 下的矩估计稳定性**本轮未实跑验证**。代码层面 loss 内部
  `.float()` 强转、VCN 输出 `.to(kp_feats.dtype)`，逻辑上 AMP-safe，但建议正式训练前在 lab-3090-d 上补跑一次带 CUDA 的 dryrun。
- ⚠️ **远程 `lab-3090-d` 上的 dryrun 是旧版**（`from model.modules.vcnorm import ...`，会触发包 `__init__` → `mmcv`/`mmengine`
  缺失 Traceback），与本地 importlib 旁路版本不一致 → **远程仓库与本地不同步**，"远程已跑过 ALL PASSED"无法在当前远程状态复现。
  正式训练前必须先把本地这套文件 rsync/commit-push 到 lab-3090-d，并在远程重跑 dryrun。

## 1. design.md — 合理性 / 单变量 / 是否换皮

- 假设清晰：把遮挡当未对齐 domain factor，对 GCN 17 per-keypoint token 的"可见性条件归一化统计(一阶/二阶矩)"做 batch 级对齐。有探针(commit e3a709b, KL~288/AUC~0.97)支撑前提，是 problem-level 机制，不是小调参逃避创新。
- **新于 OA-SD/BT-PKD 确认**：OA-SD 蒸 pooled global cosine（instance 级），BT-PKD 蒸 per-keypoint 方向 cosine（instance 级）。VCA 对齐的是 **batch 级 per-keypoint per-channel mean/var**，不监督"谁是谁"——机制上确实正交，不是换皮。代码里也确实如此（`_weighted_moments` 在 dim=0/batch 上聚合）。
- 单变量对照设计正确：design.md §对照组明确"严格单变量 = 同 PLBOA、VC-Norm 开/关"，并诚实说明**不是** vs 原 exp260b 88.0（原 exp260b PLBOA=False），88.0 仅作量级参照。这一点处理得对（见下方 High-1 的隐患）。

## 2. vcnorm.py — 逐行

- zero-init 恒等：`cond[-1]` weight/bias 全 0 → gain=shift=0 → `out = kp_feats + (0*(x_ln-kp_feats)+0)` = 恒等。dryrun 实测 0.000e+00。✅
- tanh 包裹防爆：`gb = gain_scale * tanh(gb)`，gain/shift ∈ [-1,1]，gain_scale=1.0。残差 `gain*(x_ln-kp_feats)` 受 (x_ln-kp_feats) 量级影响，但有界 tanh + LayerNorm 参考，不会 runaway。✅
- AMP dtype：`(... ).to(kp_feats.dtype)` 残差强转回输入 dtype，autocast 下不破 dtype。✅（dryrun AMP 分支未在本地跑，见 §0）
- train/test 对称：模块单点应用在 skeleton_gcn forward §3.5，train/test 同一条路径，无 `self.training` 分支。✅
- stats 在 `torch.no_grad()` 内、`.item()` 取标量，不挂图。✅
- **device/shape**：reshape(B*K,1)→MLP→view(B,K,C)，clamp(0,1)。形状一致。✅

## 3. vcnorm_loss.py — 逐行

- 矩估计 `_weighted_moments`：weighted mean/var per (K,C)，`var=clamp(E[x^2]-E[x]^2, eps)` 防负方差。✅
- teacher detach：`t = teacher_kp.float().detach()`，且 `t_mean.detach()/t_var.detach()` 二次保险，单向对齐。dryrun 实测 teacher.grad is None。✅
- vis_thr 跳过：teacher 用 `(t_sc>=vis_thr)*t_sc` 加权，valid_k 由 `t_wsum>=min_weight` 门控，全遮挡→valid_k=0→零损失保图(`(student_kp*0).sum()`)。✅
- AMP：内部 `.float()` 强转做统计，稳。✅
- **与探针"同一把尺子"**：对角高斯一阶/二阶矩匹配（mean L2 + std L2），与 design 声称一致。✅

## 4. wiring — 逐行

- VCN 插点：skeleton_gcn.py §3.5，**post-GCN/pre-pool**，只动 GCN 17 token，不碰 LGPA/global/PSG。✅ 单点应用 → 同时流入 pooled ReID 特征与导出的 `aux_data['kp_feats']`，train/test 对称。✅
- dual-forward 接线：student 走 `model(v_img)`（LGPA+GCN 分支，line 599-622）→ `kp_data['gcn_kp_feats']`；teacher 走 `ema_teacher(img_teacher, pose_dict=teacher_pose)`（train mode）→ `teacher_kp_data['gcn_kp_feats']`。parallel_oa_sd 下 `kp_data=all_kpdata[0]`（student view-0），teacher 是 clean 第 4 视图。配对正确。✅
- 复用现成前向、不新增 backbone pass：teacher 前向是 OA-SD 既有的，VCA 只读它的 kp_data，无额外前向。✅
- loss 加进总 loss：`loss = loss + vcn_weight*vca_loss`，权重 0.5，warmup>20（epoch 1-indexed，epoch 21 起生效）。✅
- 优化器：`make_optimizer` 遍历 `model.named_parameters()`，VCN 参数由 `skeleton_head.vcnorm` 持有 → 自动入 optimizer。✅
- **关 POSE_VCNORM 复现 baseline**：`POSE_VCNORM=False` → 模型不建 vcnorm（None，no-op）+ processor VCA 分支不进入。第二个 caller `pose_dual_stream_model.py` 全 kw 传参、vcnorm 默认 False，不破坏。✅ 代码层面复现性成立。
- PLBOA 确实把 student 被遮挡 keypoint 的 score/visibility/heatmap 全部清 0（pose_dataset.py:871-874），occluded-vs-clean 非对称信号真实存在，VCA 前提在数据侧成立。✅

## 5. config — 逐行

- defaults.py：6 个新键全部默认安全（POSE_VCNORM=False，其余仅在开时生效）。✅
- pose_vcnorm_base.yml：POSE_VCNORM_MODULE=True 在 yml 里有，design.md §config 代码块漏列了 GAIN_SCALE / MODULE 两键（design 写的是 6 键，实际 defaults 是 7 键含 GAIN_SCALE）——**文档与代码轻微不一致**（Low）。yml 本身完整正确。

---

## Findings（分级）

### Critical
- 无。

### High
- **High-1 [实验设计 / 对照基线]**：yml 与 exp260b baseline 的差异不止 VC-Norm 一项——本 yml `POSE_LOWER_BODY_OCC=True`，而原 exp260b 88.0 是 `POSE_LOWER_BODY_OCC=False` 跑出来的。VC-Norm 需要 PLBOA 提供 occluded-vs-clean 信号，所以开 PLBOA 是必须的，但这意味着 **"88.0"不能直接当对照**。design.md 已经诚实指出这点，但**必须实跑 VC-Norm-OFF + PLBOA-ON 的对照实验**才能算 VC-Norm 净增益，否则结论无效。
  修法：正式训练时务必同时排一条 `POSE_VCNORM=False`（其余完全相同含 PLBOA=True）的对照；results.md 里 VC-Norm 增益只能 vs 这条对照，不能 vs 88.0。**这是实验有效性的硬前提，不是代码 bug。**

### Medium
- **Medium-1 [对齐目标的语义]**：VCA 的 teacher target `t_kp` = teacher 路 GCN token **经过 teacher 自己的 VCN 之后**的输出（VCN 在 skeleton_gcn forward 内单点应用，teacher 前向也走这条路）。但探针测的是**raw GCN token**的统计 shift。teacher VCN 起步是恒等(zero-init)，前期 t_kp≈raw，但训练后 teacher VCN（EMA 跟随 student）会改写 t_kp，于是"对齐目标"本身在动。这不是错（EMA self-distill 范式本就如此），但要意识到：**对齐的是 VCN 后的统计，不是探针 raw 统计**。建议日志同时记录 `vca_md/vca_sd` 是否单调下降以确认对齐在收敛；若不收敛，考虑把 VCA 接到 pre-VCN 的 raw GCN token（design §预期失败原因 3 已提到 pre-GCN 备选，可一并考虑 pre-VCN）。
- **Medium-2 [全 batch 遮挡某 keypoint 的 student 矩]**：当某 keypoint 在整个 batch 都被 PLBOA 遮挡时，student `s_w[k]≈0`，`_weighted_moments` 的 `wsum_safe=clamp(eps)` 使 `s_mean[k]` 退化为 `(≈0)/(1e-5)` 的噪声值。该 keypoint 仍可能因 teacher 可见而 `valid_k=True`，于是 student 噪声矩被强行对齐到 teacher → 噪声梯度。lower-body keypoint(脚踝/膝)被 PLBOA 高概率遮挡，batch=64 下全遮挡虽罕见但非零。
  修法（建议，非阻断）：valid_k 同时加 student 侧门控，例如 `valid_k = (t_wsum>=min_weight) & (s_wsum>=min_weight_student)`，跳过 student 自己几乎没样本的 keypoint。当前不会 NaN（有 eps/clamp 保护），只是引入少量噪声梯度，可先跑、观察 vca_vk 是否异常偏低。

### Low
- **Low-1 [文档一致性]**：design.md §config 代码块只列 6 键，漏 `POSE_VCNORM_GAIN_SCALE`、把 MODULE 顺序也写得和 defaults 不完全一致；defaults.py 实为 7 键。补 design.md 即可。
- **Low-2 [AMP 未实跑]**：见 §0，本地无 CUDA 跳过了 AMP 分支。代码 AMP-safe，但正式训练前在 lab-3090-d 补跑 CUDA dryrun 验证 float16 矩估计有限性。
- **Low-3 [远程不同步]**：见 §0，lab-3090-d 上 dryrun 是旧版会崩。push 本地新文件到远程后再训。
- **Low-4 [import 位置]**：`from loss.vcnorm_loss import vcnorm_align_loss` 写在 processor 训练循环内（line 900）。功能无碍（首次后被 cache），但每 iter 走一次 import 查找，可提到 do_train 顶部或文件头，微优化。

---

## 结论

代码质量高：插件式、zero-init 恒等、teacher detach、train/test 单点对称、AMP dtype 处理、优化器自动纳管、baseline 复现性（关开关=no-op）全部正确，dryrun 的 load-bearing 正确性主张本地实跑通过。机制确实新于 OA-SD/BT-PKD（batch 级统计矩对齐 vs instance 级特征蒸馏），有探针支撑，是 problem-level 创新而非小调参。

**无 Critical / 无 High 级代码 bug。** 唯一 High 是实验设计前提（必须跑 PLBOA-ON 的 VC-Norm-OFF 对照，不能拿 88.0 当对照）——design.md 已意识到，只需在执行时落实。Medium/Low 均为可观察、非阻断的改进项（建议训练时盯 `vca_md/vca_sd` 单调性、`vca_vk` 数量、`vcn_gain_std` 不塌缩）。

**审查通过**（前提：正式训练前 ① 把本地这套文件同步到 lab-3090-d 并在远程补跑一次带 CUDA 的 dryrun；② 务必同排 `POSE_VCNORM=False`+PLBOA 的单变量对照）。代码层面可以进入第二轮 Codex 审查。
