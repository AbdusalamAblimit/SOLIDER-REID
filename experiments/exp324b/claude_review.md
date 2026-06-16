# Claude Broad Review — exp324b

实验：冻结 DINOv2-base + 轻量共享线性投影头 + 姿态部位匹配，训练把 exp324 frozen 的 part-MaxSim 信号拉上去。
审查对象：`scripts/exp324b_train_head.py`（全文逐行）+ `experiments/exp324b/design.md` + helper `scripts/exp324_dino.py` + repo `loss/triplet_loss.py` / `datasets/sampler.py` / `loss/make_loss.py` + 远程 pose_data 实地核对。
审查者：Claude (Opus 4.8 1M)。日期：2026-06-16。

## 第一轮：需修改

无 Critical / 无运行时 bug。数据流、PKSampler、batch-hard soft-margin triplet、eval、cache 隔离、NaN/零部位边界均逐行 + 离线复现验证正确。

**唯一阻断项 H1（High）— train/test 表征不一致**：投影头只通过"全局 masked-mean 特征"收梯度（ID CE on BNNeck + triplet on pre-BN 全局），但测试的主指标 part-MaxSim 用"逐部位 L2-归一化向量的余弦"。只优化平均向量 → 逐部位向量可能仍弱判别 → part-MaxSim 训不上甚至低于 frozen baseline → **会被误判成"冻结特征天花板低"而错误止损**。（Codex 第一轮独立收敛到同一结论。）
修复要求：加 part-level 监督（per-part ID/triplet，做 global-only vs +part 消融），或改口径让全局 cosine 当主指标并在 design 标注 part-MaxSim 为诊断。

非阻断：H2（cos 走 BNNeck、part 不走，分开报告）；M2（dead CACHE_QG，q/g 重抽 ~5min）；M3/M4（无 LR scheduler；WD 命中 BN gamma）。

## 修复（已应用）

- 加 **per-part 共享 ID 分类头** `part_classifier`，对每个**可见**部位的投影向量直接 CE，权重 `--part_weight`（默认 0.5，=0 即 global-only 消融臂）。loss = 全局 ID CE + 全局 soft-margin triplet + part_weight×per-part ID CE。
- 顺带：cosine LR scheduler；BN 参数（ndim≤1）排除 weight decay（param group）；PKSampler `num_batches=max(1,…)` 守卫；移除 dead CACHE_QG；design.md 同步。
- dry-run（800 img / 5 step）：part loss 初始 ≈ ln(39) 正常，全损失有限、acc 上升、params 433,664。

## 第二轮全范围复核：审查通过

- **H1 已正确解决**：per-part CE 监督打在与 part-MaxSim **同源的 `proj` 空间**（part-MaxSim 也用 `project_parts` 输出再归一化）。无 bias 线性 CE 推动部位向量**方向**朝类原型靠拢、跨类分散——方向判别性正是归一化余弦 part-MaxSim 所依赖的量，对齐成立，符合 ReID 标准范式（CE on un-normalized feature + cosine retrieval）。`vmask` 仅监督可见部位正确。残余 gap（CE 约束"向量 vs 类原型"而非"同部位两向量余弦"，per-part triplet 更直接）属可接受近似，不阻断；**part-MaxSim 若仍不涨可归因冻结特征天花板而非监督未到位——首轮判据满足。**
- **逐行**：step 返回 7 值 ↔ meters np.zeros(7) ↔ epoch/dry 两处解包均 7，一致无 off-by-one；part_loss dtype/device 正确；`vmask.any()` 守卫防空 CE NaN；part_loss=0 时 backward 安全；part_classifier（2D）正确入 decay 组优化器、无遗漏参数。
- **scheduler**：CosineAnnealingLR 每 epoch 末 step，位置正确（非 per-iter）；dry-run 不触发。
- **param group**：proj/classifier/part_classifier（2D）带 WD；BN gamma（1D）no_decay；BN bias（frozen）经 requires_grad 守卫跳过。正确。
- **triplet / 采样 / eval / 同cam排除 / heavy-occ 切片 / pose split 映射**：本轮改动未触及，复核未被破坏（soft-margin `SoftMarginLoss(dan-dap,+1)`、`remove=(g_pids==q_pid)&(g_camids==q_camid)`、HEAVY_OCC_THR=8、p0 映射均正确）。
- **design.md 与代码逐项一致**（loss 描述、顺带修项、成功口径 part-MaxSim 主+cos 对照、part_weight 默认 0.5）。
- 无新引入问题（part_classifier 与全局 classifier 独立头，分别作用于 proj / BNNeck，有意设计；part CE 与 part-MaxSim 表征同源）。

**Findings**：Critical 无 · High 无 · Medium M-1（part-loss 重复一次 `project_parts` 前向，纯效率，结果正确，可复用 forward 的 proj 优化，不阻断）· Low（dry double-limit 无害、末轮 LR≈0 仅评测）。

## 结论

**审查通过。** 首轮唯一阻断项 H1 已正确修复，四项非阻断顺带修实现正确，无 Critical/High/运行时 bug，design 与代码一致。可进入 Codex 第二轮 / 开训。
