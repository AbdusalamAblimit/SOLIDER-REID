# 实验 exp328: VC-Norm — 把"遮挡"当 domain factor，对 per-part token 做 visibility-conditioned normalization 对齐

> **来源**：post-PRCV「搬范式 / 重定义问题」路线。前置探针 `scripts/vcnorm_probe.py`（commit e3a709b）已证前提成立：在 Market 全可见训练的 exp260b 模型上，Occluded-ReID 被遮挡部位的 per-part GCN token 归一化统计（per-channel mean/var）漂到与可见 token 明显不同、近完美线性可分的区（pre-GCN KL≈288 / LDA-AUC≈0.97，post-GCN 仍 KL 数十 / AUC≈0.8），且 3 个对照证明不是采样伪影。**遮挡 = 未对齐的 domain factor**。
> **性质**：训练端实验（**本任务只到 dry-run，不正式训练**；正式训练前须过双审查）。**机器**：lab-3090-d，repo `/root/work/SOLIDER-REID`。
> **baseline**：exp260b（Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA），Market 训练 → Occluded-ReID 跨域 = **88.0 mAP（MaxSim + flip）**。目标 **> 88.0**。

## 动机

exp260b 已有 OA-SD（occluded student ↔ clean EMA teacher 双前向）+ PLBOA（下肢遮挡增广），但其蒸馏只作用在 **pooled global 特征**（cosine）。BT-PKD 进一步蒸馏 per-keypoint 特征**方向**（cosine），仍是「让被遮挡 token 的特征向 clean token 看齐」。两者都没有触碰探针指认的真正病灶：**被遮挡 per-part token 的归一化统计（一阶/二阶矩）整体漂移到一个可分离的子空间**——这是一条「有/无遮挡」的 domain 轴，叠加在身份信号上。OA-SD/BT-PKD 把个体特征拉近，但没有显式消掉这条 domain 轴；探针显示这条轴在跨域时大量残留（pre-GCN AUC 0.97），GCN 只部分修复。

VC-Norm 的洞察：**把可见性当条件变量，对 per-part token 显式做 visibility-conditioned 归一化，并用一个对齐目标把 occluded 路 token 的"可见性条件归一化统计"拉向 clean 路的统计，从而消掉这条 domain 轴**——但只动归一化统计（per-keypoint per-channel 的 mean/std 漂移），不动可分辨身份的判别方向（关键 caveat：别把对齐做成「抹身份」）。

## 核心假设

在 exp260b 的 dual-forward（occluded student / clean teacher）之上，对 GCN 的 17 个 per-keypoint token 加一个 **VisibilityConditionedNorm（VCN）模块** + **可见性条件统计对齐 loss（VCA）**：训练时把 student 路被遮挡 token 的 per-keypoint per-channel 归一化统计对齐到 teacher（clean）路同 keypoint 的统计，能消掉「遮挡 domain 轴」、让 visible-part 表示对有/无遮挡鲁棒，从而把 Market→Occluded-ReID 跨域 mAP 抬过 88.0。

## 技术方案

### 改了哪些文件 / 新增模块

1. **新模块** `model/modules/vcnorm.py` → `class VisibilityConditionedNorm(nn.Module)`（插件式）。
   - 输入：`kp_feats (B,17,C)`、`kp_scores (B,17)`。
   - 结构：一个**可见性条件的仿射归一化**——对每个 token 先做 channel-wise LayerNorm（去掉 instance 内尺度漂移），再乘一个由可见性标量预测的 per-channel `gamma(v), beta(v)`（小 MLP：`score → (C_gamma, C_beta)`，**zero-init 输出 → 起步恒等**，sigmoid/tanh 包裹防爆）。残差形式 `out = feat + g(v) ⊙ (LN(feat) - feat_detach_mean...)`，**zero-init 保证开关关或训练起点 = baseline**。
   - 该模块只插在 **GCN per-keypoint token**（探针测的就是它），不碰 LGPA part token、不碰 global、不碰 PSG。
   - 返回 normalized token + 一个 `vcn_stats`（gamma/beta 均值，用于日志看模块是否在工作、是否塌缩）。

2. **对齐 loss** `loss/vcnorm_loss.py` → `def vcnorm_align_loss(student_kp, teacher_kp, student_scores, teacher_scores, vis_thr, ...)`：
   - **机制（核心新意）**：不是 per-instance 拉特征（那是 OA-SD/BT-PKD），而是**对齐 per-keypoint 的"可见性条件归一化统计"**。对每个 keypoint k，在 batch 内取 student 路该 keypoint 的 token 群，估计其 per-channel mean/var（一阶/二阶矩），与 teacher 路同 keypoint 的 mean/var 做对齐（Gaussian/统计距离，**与探针同一把尺子**：对角高斯一阶/二阶矩匹配）。
   - **★ student 统计取自"被遮挡子集"（codex High-1 修复，核心机制点）**：PLBOA 把被遮挡 keypoint 的 score 置 0，但 **GCN token 本身不是 0-mask**——keypoint 坐标保留，标准 bilinear 采样仍在被遮挡像素位置采到一个退化但真实的"遮挡特征"（backbone 对 occluder 像素的响应），这正是探针指认、要被拉向 clean 的 domain-shifted token。陷阱：若像旧实现那样用 student score 给 student 矩加权，被遮挡 token（score≈0）几乎不进 batch 统计、拿不到对齐梯度，VCA 退化成"对齐本来就可见的 token"=空转、与目标相反。**修复**：student 矩只在 **被遮挡且 teacher 可见的 keypoint 子集（`s_sc < vis_thr` 且 `t_sc >= vis_thr`）** 上估计，权重用 **teacher score**（student score≈0 不能当权重）。这样被遮挡 student token 真正进入统计并被拉向 clean teacher。`valid_k` 同时要求 teacher 侧和 student-occluded 侧都有足够权重（双 min-count gate，codex Medium-b）。
   - **只对齐统计、不对齐身份**：loss 作用在 **batch 级 per-keypoint 矩**（mean/var），不是 instance 级特征；身份判别来自 token 在该统计下的相对位置，不被这个对齐项约束 → 防「抹身份」。
   - **可见性条件化**：teacher（clean）路 keypoint 几乎全可见，其统计 = 「无遮挡」目标分布；student 路含被 PLBOA 遮挡的 keypoint，把它们的统计拉向 teacher → 消 domain 轴。对低于 `vis_thr` 的 teacher keypoint 跳过（teacher 自己都看不见就没有可靠目标）。
   - **detach teacher**：teacher 矩 `.detach()`，单向对齐。
   - 返回 `(loss, stats)`，stats 含对齐前/后矩距离、生效 keypoint 数、low-vis 比例。

3. **config 开关** `config/defaults.py`（插在 LPCS 段后、INPUT 段前，约 line 312）：
   ```python
   # VC-Norm: 把遮挡当 domain factor，对 per-part token 做 visibility-conditioned normalization 对齐
   _C.MODEL.POSE_VCNORM = False              # 总开关（默认 OFF，必须复现 baseline）
   _C.MODEL.POSE_VCNORM_WEIGHT = 0.5         # 对齐 loss 权重
   _C.MODEL.POSE_VCNORM_WARMUP = 20          # warmup epoch 后才上对齐（前期统计不稳）
   _C.MODEL.POSE_VCNORM_VIS_THR = 0.3        # 可见性阈值：teacher 低于则跳过；student 低于则视为"被遮挡"进对齐子集
   _C.MODEL.POSE_VCNORM_HIDDEN = 64          # VCN 条件 MLP 隐藏维
   _C.MODEL.POSE_VCNORM_GAIN_SCALE = 1.0     # VCN 仿射 gain/shift 的 tanh 幅度上限（codex Low-1 补列）
   _C.MODEL.POSE_VCNORM_MODULE = True        # 是否插 VCN 仿射模块（False = 只用对齐 loss，纯正则）
   ```
   共 **7 个 key**（与 `config/defaults.py:315-321` 完全一致）。

4. **模型接线** `model/pose_backbone_model.py`：
   - `__init__`：`if POSE_VCNORM and POSE_VCNORM_MODULE` → `self.vcnorm = VisibilityConditionedNorm(feat_dim=in_planes, hidden=...)`。
   - LGPA+GCN dual 分支（train/test 两处）：拿到 GCN `gcn_data['kp_feats']`（17 token）后，过 `self.vcnorm(kp_feats, scores)`，再写回 `kp_data['gcn_kp_feats']`（保证 train/test 对称）。**zero-init → 起点恒等**。

5. **processor 接线** `processor/processor.py`：在 OA-SD 段（已算好 `teacher_kp_data`）后追加 VCA loss：
   ```python
   if vcnorm_enabled and epoch > vcnorm_warmup and kp_data is not None and teacher_kp_data is not None:
       s_kp = kp_data.get('gcn_kp_feats'); s_sc = kp_data.get('gcn_kp_weights')
       t_kp = teacher_kp_data.get('gcn_kp_feats'); t_sc = teacher_kp_data.get('gcn_kp_weights')
       if all is not None:
           vcn_loss, vcn_stats = vcnorm_align_loss(s_kp, t_kp, s_sc, t_sc, vis_thr)
           loss = loss + vcnorm_weight * vcn_loss; details['vcn'] = ...
   ```
   - **复用现成 dual-forward**：exp260b 已是 `parallel_oa_sd`（3 occluded student + 1 clean teacher），teacher 前向已在 OA-SD 段算出 `teacher_kp_data`，**VCA 不新增前向、不增显存峰值的额外 backbone pass**。

### 数据流（输入 → 输出）

1. dataloader 出 4 视图：`img[0..2]`=PLBOA 遮挡 student，`img[3]`=clean teacher；`pose_dict` 含 student（被遮挡 keypoint score=0）与 `teacher_pose`（全可见）。
2. student 前向：backbone(PSG) → featmap → GCN 采 17 token → **VCN（可见性条件仿射，zero-init 恒等起步）** → `kp_data['gcn_kp_feats'] (B,17,C)`, `gcn_kp_weights (B,17)`。
3. EMA teacher 前向（已存在，no_grad）：clean 图 + `teacher_pose` → `teacher_kp_data['gcn_kp_feats']`（全可见目标统计）。
4. VCA loss：per-keypoint 在 batch 内估 student/teacher 的 per-channel mean/var → 对齐距离（teacher detach）→ `vcnorm_weight * vcn_loss` 加进总 loss。
5. 反传**不更新 backbone**：GCN 分支输入是 `featmaps[-1].detach()`（`pose_backbone_model.py:608`），所以 VCA / GCN 分支只更新 **GCN + VCN + skeleton head**，backbone 经此路径拿不到梯度（codex Medium-c）。teacher 不更新（EMA 跟随 student）。注意 PSG / 主 ID·triplet 仍走非 detach 路更新 backbone，但那不是 VCA 的贡献。
6. 测试：student-only 前向，GCN token 过同一个 VCN（train/test 对称），equal_concat + MaxSim + flip 评测。

### 关键超参及依据

- `WEIGHT=0.5`：与 PLBOA 体系内 part 分支、FSDC、PACI 等辅助 loss 同量级，避免压过 ID/triplet 主 loss。
- `WARMUP=20`：与 SOLVER.WARMUP_EPOCHS 对齐；前 20ep backbone warmup、统计噪声大，过早对齐会把噪声当 domain 轴。
- `VIS_THR=0.3`：与 PKC/MST/SKC 全项目一致的可见性阈值。
- VCN `zero-init` 输出：保证 `POSE_VCNORM_MODULE=True` 但未训练时输出严格 = 输入，不破坏 baseline 复现。
- 只插 GCN 17 token：探针测的就是这 17 个，单变量最干净；不碰 LGPA/global/PSG。

## 预期结果

- **成立**：Occluded-ReID（Market→跨域）mAP/R1 > 88.0/baseline；日志里 VCA 的「对齐前矩距离 > 对齐后矩距离」单调改善，gamma/beta 不塌缩（std>0）；Market 同分布性能不掉（对齐不抹身份的证据）。
- **失败最可能原因**：
  1. **抹身份**：对齐项把 per-keypoint 统计拉太狠，连身份信号一起压平 → Market 同分布 mAP 掉、跨域也不涨。30ep kill-switch：看 `id_part` CE 是否异常升高、Market e30 eval 是否低于 baseline e30。
  2. VCN 仿射 zero-init 后训不动（gamma/beta 一直 ≈0）→ 退化成纯对齐正则（`POSE_VCNORM_MODULE=False` 对照可隔离）。
  3. domain 轴在 GCN **之后**已被 GCN 部分修复（post-GCN AUC 0.8 < pre-GCN 0.97），对齐 GCN 输出 token 收益被 GCN 吃掉 → 备选：对齐 **pre-GCN** 采样 token（探针 pre-GCN shift 更大）。

## 对照组

- **直接对照（单变量，codex High-2 已落实为独立 config）**：
  - 实验组：`configs/market/pose_vcnorm_base.yml`（`POSE_VCNORM=True`，PLBOA=True）。
  - 对照组：`configs/market/pose_vcnorm_base_control.yml`——**与实验组逐行相同，唯一差异 `POSE_VCNORM=False`**（PLBOA 仍 True）。单变量 = VC-Norm 开/关。
  - VC-Norm 净增益**只能 vs 这条对照**，**不能 vs 原 exp260b 88.0**（后者 PLBOA=False，是另一条 baseline）。
- **PLBOA 说明（重要）**：exp260b 原 config `POSE_LOWER_BODY_OCC=False`，此时 OA-SD 的 teacher≈student（代码已 warning「near-identical images」），OA-SD 近退化。VC-Norm 要 occluded-vs-clean 的对比信号**必须开 PLBOA**，故本实验与对照都设 `POSE_LOWER_BODY_OCC=True`。因此严格单变量对照是「同 PLBOA、VC-Norm 开/关」，**不是** vs 原 exp260b 88.0（后者 PLBOA 关）。88.0 仅作量级参照；若开 PLBOA 的 VC-Norm-OFF 对照已偏离 88.0，以对照为准。
- **内部消融**（后续，不在 dry-run）：
  - `POSE_VCNORM_MODULE=False`（只对齐 loss，无仿射模块）→ 隔离「对齐正则 vs 条件归一化模块」各自贡献。
  - 对齐 pre-GCN vs post-GCN token → 隔离 domain 轴位置。

## Kill-switch / 下一步

- 30ep：Market 同分布 mAP 不掉 + 跨域趋势 ≥ baseline + VCA 矩距离单调下降 + gamma/beta 不塌缩 → 跑满 120ep。
- 反之（Market 掉 / 跨域不涨 / 对齐塌缩）→ 记录止损，最可能结论是「per-part 统计对齐在判别性-互补性张力下抹身份」，回 `research_directions.md`。

## 风险：对齐会不会抹身份？怎么在 30ep kill-switch 看出来

- **机制层面已防**：VCA 对齐的是 **batch 级 per-keypoint per-channel 矩（mean/var）**，不是 instance 级特征。身份来自单个 token 在该统计坐标下的**相对位置**，矩对齐只搬动整群分布的中心/尺度，不直接监督「谁是谁」。这与 OA-SD/BT-PKD（instance 级拉近）正交。
- **30ep 可观测信号**：
  1. **Market 同分布 e30 eval**：若 VC-Norm 抹身份，Market（全可见、无 domain shift）mAP 会**先掉**——这是最灵敏的探针，因为 Market 没有要对齐的 domain 轴，任何掉分都是判别性损伤。
  2. **`id_part` / `tri_part` 分量**：日志看 GCN 分支 CE/triplet 是否随对齐项升高（判别性被压的直接证据）。
  3. **gamma/beta std**：VCN 仿射若塌缩到全 0（恒等）说明模块没学到东西；若 |gamma|/|beta| 暴涨说明在过度改写特征（潜在抹身份）。
  4. **VCA stats**：对齐前/后矩距离应缓降而非骤降到 0（骤降到 0 = 把 student/teacher 统计强行压成一点 = 抹身份）。
- 任一信号异常（Market 掉 >0.5 mAP / id_part 持续升 / gamma 爆 / 矩距离骤塌）→ 30ep 止损。
