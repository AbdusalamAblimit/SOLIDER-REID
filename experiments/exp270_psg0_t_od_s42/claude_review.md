# Claude Review — exp270_psg0_t_od_s42

**审查对象**: Phase 3-A 第一个消融 run,纯 baseline 对照(无 PSG)

## 审查范围

1. `design.md` — 单变量 ablation,关所有 pose 模块,等同 SOLIDER Tiny baseline
2. 代码改动: **无**(纯 CLI override,不新增/修改任何文件)
3. 配置: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override 5 个 POSE_* 键
4. `config/defaults.py`: 影响的字段都有对应默认值(False/None),不影响其他实验
5. processor / forward / loss 路径: `POSE_ENABLED=True` 下会构造 PSG/LGPA/GCN/OA-SD,但 `POSE_BACKBONE_PSG=False` 时 PSG gate 退化 identity,`POSE_LGPA=False` 时 LGPA head 不构造,`POSE_SKELETON_GCN=False` 时 GCN head 不构造,`POSE_OA_SD=False` 时 EMA teacher 不启用,`POSE_LOWER_BODY_OCC=False` 时 PLBOA augmentation 跳过
6. 与前序对照: exp000 (旧 baseline,120 epoch SOLIDER-Tiny, SW=0.2) = 56.6/66.5;本次新协议加默认 flip-test 预期 +0.5-0.9

## 单变量原则检查

相对 exp261 (Tiny + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA) 本 run 关了 5 个模块。**多于单变量**,但 Phase 3-A 的目的不是单变量对照 exp261,而是建立一个**纯 baseline**用以对比后续 exp271-273 的 PSG 变体(单变量: PSG stages 数)。这个 baseline 本身必须没有其他模块才能隔离 PSG 的贡献。Phase 3-A 内部的 4 格(exp270/271/272/273)之间才是严格单变量。

## 代码安全性检查

1. 所有 `POSE_*=False` 的分支在模型构造时都有对应的 `if` 保护,不会空指针
2. `POSE_TEST_FEAT='global'` 在 processor 的 eval 路径里已实现(见 `processor.py` 的 `_extract_feat_flip`),无 branch 时直接返回 global feat
3. 无代码修改 → 与 exp261 共享同一套 codebase,commit `c6f391d`,代码上已通过前序 8 个 Phase 1 run 的实战验证
4. SEED=42 与 Phase 1 Tiny exp261 一致,可直接纵向对照 PSG 贡献

## 风险评估

- **OOM 风险**: 低。Tiny + 无 pose 模块训练 GPU mem ~4-5GB,eval 含 default flip 也只 ~7-8GB,远低于 16GB。srvB exp263 OOM 是 Base 特有(88M params + full scaffold 推到了 13.2GB 边缘)
- **数据一致性**: 训练集 Occluded-Duke,与 exp261 同 split
- **收敛性**: 已在 exp000 验证过 SOLIDER-Tiny baseline 稳定收敛到 56-57 mAP
- **评估协议**: `TEST_FEAT='global' + FLIP_TEST=True`,即默认 flip-test 但不用 branch fusion(因为无 branch)

## 结论

**审查通过**。本 run 是 Phase 3-A 的 baseline 对照,代码零改动,风险极低,预期 mAP 56-58。可启动。

## 备注

Phase 3-A 后续 exp271-273 的 review 可复用本次审查的框架,只需补充"单变量改动是 PSG_STAGES"说明。
