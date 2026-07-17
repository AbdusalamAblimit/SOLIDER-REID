# Claude Broad Review — exp330 Compositional Occluder Generalization + group-DRO

**Date**: 2026-06-17
**Scope**: design.md + scripts/compositional_occlusion.py + scripts/exp330_train.py（全范围）

## 第一轮（Opus 子代理广审）发现 + 修复

子代理逐行审 + 与项目参考 `datasets/occlusion_augmentation.py` 交叉核对。发现按严重度：

### Critical
- **C1 eval_func 参数序/camid dtype**：调用 `eval_func(distmat, q_pids, g_pids, q_cams, g_cams)` 序与验证签名一致；camid 由 `np.asarray(tensor)` 拼接（CPU tensor 可行）。→ smoke test 验证（待）。**保留为 smoke 验证项。**
- **C2 手搓 optimizer 把 weight-decay 施加到所有参数（含 norm/bias）**，削弱 ViT substrate（两臂同等受损但拉低质量）。**已修**：改用 TransReID `solver.make_optimizer`（bias 用 BIAS_LR_FACTOR=2、WEIGHT_DECAY_BIAS，norm/bias 处理与 53.5-baseline 一致）。
- **C3 无 warmup 的裸 cosine**，ViT 无 warmup 易劣化。**已修**：改用 `solver.scheduler_factory.create_scheduler`（linear warmup + 主调度），并 `cfg.SOLVER.MAX_EPOCHS=args.epochs` 对齐调度 horizon，`scheduler.step(epoch)`。

### High
- **H1 SIE train/eval 不对称**：若 SIE 开，train 注入 Duke cam emb、eval 无 → 跨域污染。**已查**：config 未设 SIE_CAMERA/SIE_VIEW → 默认 OFF。**已修**：train 与 eval 一律 `cam_label=None, view_label=None`，对称。
- **H2（最关键的单变量隔离缺陷）DRO 的 CE 尺度 ≠ ERM 的 mean**：原 `ce=Σ_{present} q_g·L_g`，q 在全 7 组归一但只对 present 组求和 → DRO 的 CE 系统性偏小 → CE-vs-triplet 平衡两臂不同 → 违反单变量。**已修**：loss 处 q 对 present 组重归一（`q_g/Σ_present q_g`，权重和=1），与 ERM mean 同尺度。
- **H3 q 更新在 autocast 内用 fp16 loss**。**已修**：q 更新用 `lv.float().detach()`（fp32），no_grad。
- **H4 torch.cuda.amp 在 torch2.9 弃用**。**已修**：`torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`。

### Medium
- **M2 occluder pool 加载两次（train/eval 池不同 + 慢）**。**已修**：eval 复用 `occ.occluders_by_class`。
- **M3 occlusion stream 两臂不保证一致**（worker 种子）。**已修**：加 `worker_init_fn`（seed+wid）+ `generator`，同 --seed → 同流。
- **M5 design 的 Occ-Duke 副判据脚本未实现**。**决定**：副判据从 kill-switch 脚本剔除；主判据 = Market 上 held-out vs seen 组合 GAP（跨域 GAP 无偏）。结果文档注明副判据未跑。
- **M1 occluder 缩放可能过大被 clip**。self-test 已验 9 cell 全 pixel-change>0、量级合理。

### NO_MARGIN
- baseline NO_MARGIN=True（soft triplet）。**已修**：`TripletLoss()` soft 当 NO_MARGIN，否则带 margin。

## 第二轮自查（修复后全范围复核）
- optimizer/scheduler：复用 baseline 同款，substrate 对齐 ✓。
- ERM vs DRO 单变量：现仅 loss 聚合不同（CE 同尺度 H2 已修）、occlusion stream 同（M3）、同 seed/数据/aug/optim/sched ✓。
- AMP：autocast 内前向+loss，scaler 反传；q 更新 fp32 no_grad ✓。
- 模型调用：train 返回 (cls_score, global_feat)，eval 返回单 feat tensor，cam/view=None（SIE off）✓。
- eval：gallery 每次 eval 重抽（权重变）`_gallery_cache.clear()` ✓；occlusion 只施 query ✓；L2-norm + euclidean distmat 与 cosine 单调 ✓。
- 概念风险（子代理提的，非 bug）：region 放置可能"组合塌缩成 region-only"→ 若 ERM 已无 GAP 则机制无意义。**应对**：先跑 ERM 测 compositional GAP；GAP 不存在→早 kill（合理 NO-GO）；GAP 存在→才比 DRO。

## 待 smoke 验证项（reviews 通过后 --smoke 跑 3 iter + subset eval）
- C1 eval_func 实跑通 + mAP 合理（非 0/nan）。
- 模型前向 shape、DRO q 演化、AMP 无 inf-skip、group 标签 0-6。

## 结论
Critical/High/Medium 全部修复或转 smoke 验证项；单变量隔离缺陷（H2/M3）已修；substrate 对齐 baseline（C2/C3）。**审查通过**（pending：Codex 独立审 + --smoke 运行时验证；任一暴露问题则修复并复审）。
