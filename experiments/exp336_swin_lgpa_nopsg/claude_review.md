# Claude Broad Review — exp336 Swin 纯 LGPA-D 隔离（config-only, Opus 子代理）

**Verdict: PASS（审查通过）** — 无 Critical/High。零新代码,纯 config 跑原 pipeline(PoseBackboneModel + 真 LGPA,已审过)。

审查文件:design.md、exp336 config vs 原版 pose_psg_lgpa_detach.yml,交叉核对 pose_backbone_model.py(PSG 构建 38-72 / forward 419-466 / _prepare_pose 478-495 / test-feat 782-912)、make_model.py、pose_dataset.py、processor.py、make_loss.py、clip_part_head.py、pose_utils.py、defaults.py、test.py。

## 七项核查（全 PASS）
1. **POSE_PSG_STAGES:[] 安全禁 PSG**:psg_stage_indices 空 → psg_modules_dict size 0(实测);backward-compat(66-71)被 `if last_stage_idx in psg_stage_indices`(False)跳过,不创建 psg_modules;forward 每 stage 走 `if i in psg_stage_indices`(False)→ plain Swin stage,无 pose 门控。无崩。
2. **POSE_BACKBONE_PSG:True 仅选模型**(make_model:467),不强制 PSG。
3. **OASD/parallel-aug/PLBOA=False 干净禁用**:dataset 出 1-tuple(274)→collate n_views==1 单 tensor(1093);processor 非 list img → parallel_aug/oa_sd False(550)→ 标准单视图;OASD loss block gated(801)跳过;PLBOA 仅 augmentation。
4. **不设 POSE_USE_TARGET_HEATMAP → scene-merged → assign~7**:default False(110)→ use_target False(135)→ target-swap(489)跳过 → scene_heatmaps 保 merge_person_heatmaps 值。GT assignment 良构 → KL 非平凡 → **lgpa_assign≈7**(对 exp244 7.218 / exp335 修后 7.02)。关键 sanity(对照 exp335 bug=0)。
5. **eval 比较有效**:detach(601)→ backbone/global 与有无 LGPA 完全相同。POSE_TEST_FEAT=equal_concat → [g_norm⊕part_norm](909);=global → LGPA test 分支 `!= 'global'` False(796)→ 跳过组装(906)→ test_feat=global_feat(782)。两描述子同权重,global 块字节相同。**只 override POSE_TEST_FEAT,绝不设 POSE_LGPA=False**(会破 ckpt/架构 parity)。within-checkpoint ablation 有效。
6. **无崩溃交互、单变量隔离正确**:GCN/PPA/VCSR/STR/FSDC 全默认 off;LGPA train 返回干净 2+5 list(622);GLOBAL_LOSS_SCALE 0.5 在 list/non-list 损失分支一致(214,255);list triplet(222-261)正确处理 [pooled,p1..p5]。
7. **设计确实回答"CLIP 模块能否 standalone"**:within-ckpt equalcat-vs-global 隔离 LGPA 部位描述子在 plain(未门控)Swin 上的边际价值。GLOBAL_LOSS_SCALE 0.5 / 384 / scene 热图在两描述子间恒定,不偏倚比较。非正结果→佐证"PSG/系统驱动";正结果→定位 exp335 失败为 ViT-specific。两种结果都 informative。

## Medium（操作,非代码）
- **M1 远程资产**:swin_tiny.pth + clip_part_text_features.pt 须在 lab-3090-d(否则加载错/触发 open_clip 联网下载)。**已满足**:model-build 检查中两者均成功加载(CLIP cache loaded、swin "All keys matched")。
- **M2 结论 scoping**:相对 exp244 翻了 4 项(PSG/OASD/aug/PLBOA),故**别**把 equalcat 与 exp244 70.2 的 gap 归因于 PSG 单独;结论 scope 为"LGPA-D standalone on Swin(无 PSG/OASD/aug)",判据用 equalcat-vs-global(有效单变量轴)。

## Low
- lgpa_assign 走独立日志路径(processor 1027-1033,不经 kp_data=None 的 loss_fn gate),会正常打~7。
- clip_part_head.forward 形参名 target_heatmaps 实收 scene 热图——cosmetic 命名,行为正确。

## 训练时必看
**首次 eval 的 lgpa_assign 必须 ≈7(非~0)**。若~0 = scene/target 热图接线回归 → 立即 kill 重查,任何 mAP 不可信。

## 预期结果与判读（两种都 informative）
- **若 equalcat > global**(Swin 上纯 LGPA 涨):CLIP 模块**能 standalone** → exp335(ViT)的失败是 **ViT-specific**(ViT 末层 token 全局抽象,池不出强部位;Swin 多尺度 stage 特征部位友好)→ step2 可在 ViT 上换更适配的 CLIP 接法,或直接用 Swin。
- **若 equalcat ≈ global / < global**(Swin 上也不涨):确认 **PSG(pose 门控 backbone)才是增益来源,CLIP 部位本身冗余于 global** → step2 的新 CLIP 接法必须带 **global 没有的新信息**(CLIP 视觉特征 / 遮挡推理 / ID 文本原型),而非纯部位池化。
- 量级参考:exp244 纯-Swin baseline 约 56-60;若 equalcat 显著高于该区间且 > 本 ckpt global,则 standalone 成立。

## 训练监控清单
1. **首 eval(e10)lgpa_assign ≈7**(非~0)——scene 热图接线 sanity,~0 立即 kill。
2. equalcat mAP 曲线(EVAL_PERIOD 10);e60/e120 checkpoint 用 test.py POSE_TEST_FEAT=global 取 baseline 对照。
3. loss 分量:id_global / id_part / tri_global / tri_part / lgpa_assign 都应出现(单视图,无 oa_sd)。
4. Swin-Tiny 384 显存:TEST.IMS_PER_BATCH 64(flip-test 防 OOM)。
5. assign 应随训练下降(~7→~2,注意力学定位),与 exp335 修复后轨迹一致。

## 结论
config 正确隔离纯 LGPA-D(实测 PSG off + LGPA on + detach);within-ckpt equalcat-vs-global 判据干净有效;预期 assign~7;M1 资产已满足;M2 结论 scoping 已记。**审查通过**。
