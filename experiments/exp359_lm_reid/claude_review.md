# Claude Broad Review — exp359 LM-ReID 训练脚本

**审查者**: 主 Opus 循环（用户指示「关闭子 agent，自己推进」→ 主循环自审 + Codex 独立审 = 双审）
**日期**: 2026-06-25
**范围**: `cvpb_lm_reid_train.py` 全文 + `design.md` + 模型接口源码验证 + 与 frozen baseline 的对照隔离

## a. design.md 合理性 + 单变量原则
- **不是小调参**：LM-ReID 是新机制（lattice-marginalized embedding），重定义低分辨率 ReID = 采样格点不确定性；有零训练 kill-switch GO（独立轴、双 trivial 控制 vs-TTA/vs-#false 都过、GO 靠 interventional 直接测量）做前置，codex 判 CCF-B 7/10。
- **单变量**：唯一变量 = "训练 lattice-invariance" vs "frozen 模型"。eval 复用 kill-switch（byte-identical degradation），与 frozen baseline 只差"是否 fine-tune"。
- **过线明确**：训练版 h=16 > frozen ensemble 46.87 **+0.8~2.0** 才算 method（否则 ensemble trick，诚实判死，design 已写）。

## b. 模型接口验证（逐行核对源码 pose_backbone_model.py）
- exp260b = **PoseBackboneModel**（config `POSE_BACKBONE_PSG: True`，无 `POSE_PSG_PART`）。
- train forward(pose_dict=None)：`scene_heatmaps=None` → 所有 pose 分支（VCSR/structural/LGPA/PPA/GCN）条件 `scene_heatmaps/pose_dict is not None` 全 False，跳过。`_lgpa_fixed_bands = POSE_LGPA_FIXED_BANDS`（config 无）= **False** → LGPA 分支条件 `scene_heatmaps is not None or _lgpa_fixed_bands` 也 False，跳过。
- 默认 return（line 1088）：`cls_score, global_feat, featmaps, None` → **out[0]=cls_score[B*M,C], out[1]=global_feat[B*M,D]**（pre-BN triplet 特征）。✓
- 稳健：`out[0][0]/out[1][0] if isinstance list`（防 pose 分支意外触发；所有分支 return 的 global 永远是 list 第 0 个）。
- `global_feat/cls_score` 在 pose 分支**之前**算好（`_run_backbone_with_psg(x,None)` → fcneck → bottleneck → classifier），关 pose 分支不影响主特征，与 frozen baseline 的 PSG-off global feat 一致。

## c. 损失正确性（逐项核对）
- **L_id** = CE(cls, y_rep) + batch_hard_triplet(gf, y_rep)：per-variant ReID。triplet = margin_ranking_loss(d_an, d_ap, y=1) = relu(d_ap−d_an+margin)；d_ap=最难正(max same-id，自身 dist=0 不会赢 max)，d_an=最难负(min diff-id，正样本 +1e9 排除)。✓
- **L_marg** = −log[mean_l softmax(cls^l)[y]] + triplet(mean_l gf)：边缘似然 + 均值特征 triplet。`p_mean.gather(1,y)` 取正确类概率。✓
- **L_cons** = mean(1−cos(z^l, sg(z_mu))) + β·KL(p^l‖sg(p_mu))：cos 项拉每 variant 到均值(z_mu detach)；KL 已修为 **forward 方向** `p_bm·(logp_l − log_pmu)`（p_mu detach）匹配 design。✓
- **L_adv** = GRL：disc 预测 variant slot，梯度反转使 z 不含可预测 lattice label。`slot=arange(M).repeat(B)` 与 image-major reshape 对齐。warmup-gated（adv_start），默认 lam_adv=0（弱辅助先不开，可后续 ablation）。✓

## d. reshape 一致性（image-major）
- `x=xb.view(B*M,…)`，xb[B,M] → x[i*M+m]=img i var m。`y_rep=y.repeat_interleave(M)` → y_rep[i*M+m]=y[i]。`cls.view(B,M,−1)/gf.view(B,M,D)` 还原 [i,m]=img i var m。三者一致 ✓。

## e. 数据 / AMP / 优化器
- lattice 生成函数（make_lr/make_lattice_variants/pil_to_tensor_np/_to_target_aspect）**逐行复制自 cvpb_lattice_killswitch.py** → 训练降质与 GO kill-switch eval **byte-identical**（单变量隔离的关键）。
- PK sampler（P=16×Kins=4=BS64），M=2 variants/img → forward 128。DataLoader num_workers 并行 PIL 生成（避免 CPU 瓶颈）。
- AMP：autocast 仅包 forward，损失 `.float()` 在 fp32（log/KL 防 fp16 下溢）。GradScaler 标准 scale/step/update。✓
- 优化器 SGD nesterov + cosine + warmup，fine-tune lr=3.5e-3（低于从头 0.008，保护已学特征）。**BS=64 不改** ✓，M 是方法固有增广非 BS。
- 保存 `model.state_dict()` → kill-switch 的 load_param 兼容（标准 SOLIDER 格式）。

## f. 对照隔离 + 风险
- 对照 = frozen 同模型同 eval。eval **复用 kill-switch** `--ckpt <微调>`，不重写 eval 逻辑 → apples-to-apples。
- cam=zeros：SIE 配置无，kill-switch eval 用 zeros 已得 sanity 94.43（=ref 94.4）→ benign，训练用 zeros 与 eval 一致无 train/test mismatch。
- 风险1：训练版若只 ≈ frozen ensemble → ensemble trick，诚实判死。
- 风险2：M=2 forward 128 显存（AMP 缓解，24G 应够，smoke 验）。
- 风险3：fine-tune 可能掉 HR mAP（预期，专攻 LR；报 LR 增益为主，HR 作 trade-off 记录）。

## 结论
逐行审查 + 源码接口验证完成，4 处问题已修（KL 方向 / AMP / 稳健 out 取值 / scaler backward）。代码正确、对照隔离干净（lattice 生成 byte-identical 复用、eval 复用 kill-switch）、过线判据明确、诚实判死路径清楚。**审查通过**，可进 smoke + 训练（仍需 Codex 独立审通过）。
