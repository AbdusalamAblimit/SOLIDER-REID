# Claude Broad Review — exp342 (CLIP-ID-prompt + LGPA 姿态部位分支, Step 2)

**审查类型**: Claude Broad Review（Opus，全范围逐行）
**日期**: 2026-06-20
**Commit**: working tree (exp/pose_heatmap)
**审查轮次**: v1
**变更范围**: model/pose_backbone_model.py (+2 行, LGPA 分支注入 clip_id_loss) + configs/occluded_duke/exp342_clip_id_pose.yml (新, = exp341 但 POSE_LGPA True)

---

## 验证方法说明

- git diff HEAD：本次代码改动仅 2 行（pose_backbone_model.py:675-676），逐行读。
- 逐行读了 clip_id_loss 的全链路：computed（pose_backbone_model.py:573-580）→ carried（676 LGPA / 850 plain）→ consumed（processor.py:1297-1302）。
- grep 全仓 `clip_id_loss` / `POSE_CLIP_ID_*` 确认计算 1 处、消费 1 处、注入 2 处（互斥分支）。
- diff exp341↔exp342 config：净改动 = POSE_LGPA False→True + OUTPUT_DIR + 注释，单变量。
- 读 make_loss.py:128-261 list-loss 路径，核对 GLOBAL_LOSS_SCALE × POSE_PART_WEIGHT 实际权重（这是本实验最大的"叙事"风险点）。
- 读 eval 分支 852-1004（LGPA test path + equal_concat 组装），确认 prompt learner train-only、test 端不碰。
- exp341 claude_review.md 已审过上游 CLIP wiring（approve），本轮聚焦 exp342 的 2 行 fix + LGPA 交互。

---

## Findings by Severity

### Critical
无。

### High
无阻断级问题。2 行 fix 正确，clip_id_loss 不重复计、test 端不受影响。

### Medium

**M1 — 全局损失权重在 exp341↔exp342 之间不一致（叙事陷阱，非代码 bug，必须知情）**
这是本实验最重要的发现。`GLOBAL_LOSS_SCALE: 1.0` 在两个实验里值相同，但**实际作用不同**：
- **exp341**（POSE_LGPA=False）→ 模型走 plain 元组 return（pose_backbone_model.py:850-851），loss 命中 make_loss.py:219/260 的 `else` 分支：`ID_LOSS = 1.0 * ce_fn`、`TRI_LOSS = 1.0 * triplet` → global **全权重 1.0** 训练。这正是 exp341 codex-High 要的（"无 part 分支，global 即描述子，须全权重"）。
- **exp342**（POSE_LGPA=True）→ 模型走 **list** return（pose_backbone_model.py:690/694），loss 命中 make_loss.py:128 的 `isinstance(score,list)` 分支：`w_g = 1/(1+POSE_PART_WEIGHT) = 1/(1+1.0) = 0.5`，`ID_LOSS = GLOBAL_LOSS_SCALE * w_g * global_id + w_p * part = 1.0*0.5*global + 0.5*part`，triplet 同理 → global **有效权重 0.5**。
- POSE_PART_WEIGHT 用默认 1.0（defaults.py:88，exp342 未 override），因此 50/50 split。

**后果**：exp342 的 **global 部分是在 0.5x 权重下训练的**，而 exp341 的 global 是 1.0x。所以 design.md:16 的"exp342 global（prompt）≈ 59.8"**很可能不成立**——不是 prompt 机制失效，而是 global 被半权重训练（恰好回到 exp341 当初想避免的 0.5x 局面）。同理"57.6→59.8→60.6"这条干净链条的**中间台阶口径变了**。

**这不是 bug**：50/50 global/part 正是标准 LGPA 训练范式（exp244/exp255 同款），也是当初 +0.8~0.9 LGPA 增益的测量口径，**对 LGPA 实验本身完全正确**。本实验真正关心的对照是 **exp342-equal_concat vs exp341-global**，这个对照成立。
**要求（知情，非阻断）**：(a) 结果分析时，若 exp342-global < 59.8，先归因到 0.5x 权重而非 prompt 失效；用 exp342 自身 global vs equal_concat 隔离 pose 贡献才是干净的。(b) 建议 monitor/results 里同时记 exp342-global 与 equal_concat 两个数，并标注 global 是 0.5x 口径。(c) pose_backbone_model.py:846 那句注释"implicit 0.5x global"措辞误导（实际是 w_g=0.5 由 POSE_PART_WEIGHT 决定，不是硬编码 0.5，且受 GLOBAL_LOSS_SCALE 再乘），建议改注释。

**M2 — clip_id_loss 张量复用同一对象，已确认在图上且无副作用（核对通过，非问题）**
`clip_id_loss`（573-580）用 `global_feat`（fcneck 后/bottleneck 前，与 ID/triplet 同源）经 `clip_id_proj` 投影 + `clip_id_prompt(label)` 文本原型 + SupCon 双向算出，全在 autocast 内（training forward），挂在 autograd 图上。676 行只是把这个**已存在的张量引用**塞进 kp_data dict（不 detach、不 clone），processor 1300 行 `loss + w*kp_data['clip_id_loss']` 直接用它 backward → 梯度正确回流到 clip_id_proj/cls_ctx/backbone。dict 引用不破坏图。✓

### Low

**L1 — config 顶部注释与实际配置部分对不上（文档陈旧，非 bug）**
exp342_clip_id_pose.yml 第 2-4 行注释写"问题：CLIP 模块(LGPA-D)本身能否 standalone 涨点，还是增益来自 PSG?"、"基于 pose_psg_lgpa_detach.yml(exp244 原版)"、"判据：POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline)"——这是从 exp336/LGPA-standalone 模板复制来的，与 exp342 真实意图（CLIP-ID-prompt 之上加 pose）不符。不影响运行（注释），但会误导接手。建议改成 Step 2 的实际叙事。

**L2 — POSE_LGPA_CLIP_DIM 512 vs CLIP-ID-prompt 用 ViT-L-14(768)，两套 CLIP 维度并存（已确认互不干扰）**
config 里 `POSE_LGPA_CLIP_DIM: 512`（LGPA 的 clip_part_head 内部 CLIP-text 维度）与 `POSE_CLIP_ID_PRETRAINED` 的 ViT-L-14（clip_id_prompt 投影到 768）是**两个独立 CLIP 用途**：LGPA 用自己的 clip_part_head（512），CLIP-ID-prompt 用 clip_id_prompt（768）。两者各自维度自洽，不共享权重，不冲突。✓ 仅提示：日志里会看到两组 CLIP 初始化打印，正常。

**L3 — pose_backbone_model.py:575 函数体内 import supcon_i2t（继承自 exp341，非问题）**
每 step 触发一次 sys.modules 缓存查找，开销可忽略。exp341 已记录为 L 级，沿用。

---

## 关键正确性核对（逐项 PASS）

1. **clip_id_loss 在作用域内**：573 行在 `if self.training:`（564）块内、LGPA 分支（668）之前定义，default None；674-676 行 LGPA 分支能读到它。✓
2. **carried 进两个 LGPA return**：676 行注入 `kp_data['clip_id_loss']`，之后 dual-GCN return（690-692）与 LGPA-only return（694）**都用同一个 kp_data 对象**（674 行 `kp_data = lgpa_data`，676 注入，两 return 均 `..., kp_data`）→ 两条路径都带 clip_id_loss。✓
3. **无重复计**：clip_id_loss 全仓**计算仅 1 处**（573-580），**消费仅 1 处**（processor.py:1297-1302，`kp_data.get('clip_id_loss')`，加 1 次）。LGPA 分支自身**不**算/不加 clip_id_loss（grep 确认）。loss_fn（646）收的是 kp_aux_data（616 行仅当 maxsim/evid/supcon on 才建，exp342 全 off → None），loss_fn 也不处理 clip_id_loss → 无第二次累加。✓
4. **张量在图上**：见 M2，用 global_feat（与 ID/triplet 同源）算，autocast 内，dict 引用不 detach。✓
5. **plain return 与 LGPA return 互斥**：exp342 use_lgpa=True → 命中 668 的 elif，从 690/694 return，**永不到达** 849-851 的 plain-path（那是所有 part 分支都关时的兜底）。两处注入不会同时发生。✓
6. **processor 解包 5-tuple**：use_pose 路径 598-599 行 `len==5 → score,feat,feat_maps,recon_loss,kp_data = model_out`，LGPA return 正是 5-tuple，kp_data（含 clip_id_loss）正确落到局部 kp_data。✓
7. **list-loss 路径正常**：score/feat 是 list（[global]+lgpa...）→ loss_fn（646）走 make_loss.py:128 list 分支，global ID+triplet + LGPA part CE/triplet 全部累加（M1 说明权重）。LGPA assign-KL 经 processor.py:1026-1032（`lgpa_enabled and not ppa_enabled and 'assign_loss' in kp_data`）加 `POSE_LGPA_ASSIGN_WEIGHT*assign_loss`。✓ 四类损失（global ID/tri + clip_id i2t/t2i + LGPA part CE/tri + LGPA assign）各自独立累加，无冲突。
8. **LGPA detached 不扰 backbone**：670 行 `lgpa_input = featmaps[-1].detach()`（_lgpa_detach=True，config POSE_LGPA_DETACH:True）→ LGPA 部位分支梯度不回 backbone；backbone 只被 global ID/triplet + clip_id_loss 训练。prompt-improved backbone 不被 LGPA 扰动。✓
9. **eval forward（equal_concat）**：test 分支 852+ 完全不引用 use_clip_id_prompt/clip_id_prompt（grep 确认 0 处）→ prompt learner train-only 被忽略。LGPA test path（870-875）use_lgpa=True 且 pose_test_feat='equal_concat'≠'global' → clip_part_head(featmaps[-1], lgpa_hm) 出 [pooled,part1..K]=gcn_feats；equal_concat 组装（984-987）：global L2-norm + 各 part L2-norm → cat。描述子 = norm(global)⊕norm(parts)。✓ 注：test 端 LGPA 用 **非 detached** featmaps[-1]（873 行，detach 只是 train 优化），正确。
10. **config 单变量**：diff exp341↔exp342 净 = POSE_LGPA False→True（+ OUTPUT_DIR + 注释）。POSE_TEST_FEAT='equal_concat' 在两份 config 里都有（exp341 因 LGPA=False 而 inert，eval 落 global；exp342 LGPA=True 激活 equal_concat）→ 实际单变量比 brief 说的更干净（只 POSE_LGPA 翻转）。✓
11. **优化器**：exp341 已核对 cls_ctx+clip_id_proj 被优化（make_optimizer 遍历 requires_grad）。exp342 新增 clip_part_head 参数同样被遍历加入。LGPA detach 只断 backbone 梯度，不影响 clip_part_head 自身参数更新。✓
12. **维度**：Swin-Tiny in_planes=768；clip_id_proj 768→768（ViT-L-14）；LGPA clip_part_head 内部 512，独立。equal_concat 把 768(global)+ LGPA part 向量 cat，维度自洽（test.py 端同 config 复现）。✓

---

## 设计层面质疑（审查协议要求）

- **这是不是小调参/逃避创新？** —— 本实验是已批准 2-step 计划的 Step 2（Step 1 exp341 真涨 +2.2）。代码改动仅 2 行是**因为机制（LGPA + CLIP-ID-prompt）都已存在**，exp342 是把已证有效的 pose 部位分支叠到已证能涨的 CLIP 机制上，验证"姿态能否在 prompt-improved backbone 上再加 +0.8~0.9"。这是**组合验证而非新机制**——属于计划内的增益叠加测试，可消融（global vs equal_concat 隔离 pose），不是在旧 branch 上堆全新小模块。符合"既定计划执行"，**不算逃避创新**（Step 1 才是机制创新点）。✓
- **单变量**：对 exp341 仅翻 POSE_LGPA；对 exp342 自身 global vs equal_concat 隔离 pose。✓ 但 M1 提醒：global 口径在 exp341↔342 间因 list/plain 路径而变（0.5x），分析时要小心。

---

## 日志充分性

- `details['clip_id']`（processor.py:1301）打印 clip_id_loss（观察 prompt 是否仍在学，exp341 实测 8.7→2.83）。
- `details['lgpa_assign']`（processor.py:1032）打印 LGPA assign-KL（观察部位分配是否塌缩，历史诊断信号 assign~7）。
- make_loss `id_global/id_part/tri_global/tri_part`（215-216/256-257）打印 global vs part 分量 → 可直接看到 M1 的 0.5x 效果（id_global 会比 exp341 同期略小是预期）。
- 足够判断两机制是否同时工作、是否互相塌缩。✓

---

## 调参风险（flag，非阻断）

- **R1 — 两损失 + LGPA 早期竞争**：clip_id_loss(1.0) + LGPA assign(0.5) + part CE/tri(0.5) 同时叠在 global(0.5) 上，早期 prompt/LGPA 都随机时多 loss 竞争。exp341 已实测 prompt 能收敛；LGPA detached 不扰 backbone，风险可控。建议盯前 10 epoch id_global/clip_id/lgpa_assign 三者趋势。
- **R2 — global 0.5x 可能拉低 prompt 增益**：见 M1，若 exp342-global 掉到 ~57-58（被半权重），说明 prompt 在 0.5x 下边际缩水；这时 equal_concat 的 pose 增益是否还能把总分推过 exp341-global(59.8) 是本实验真正看点。非阻断，是结果解读重点。

---

## 结论

2 行 fix（pose_backbone_model.py:675-676）经全链路逐行核对**正确**：clip_id_loss 在作用域内、carried 进 LGPA 的两个 return、计算 1 处/消费 1 处/无重复计、张量在图上、与 plain-path 互斥、processor 5-tuple 解包正确、test 端 prompt learner train-only 被忽略、equal_concat 组装正确、LGPA detached 不扰 prompt-improved backbone、维度自洽。config 对 exp341 单变量（仅 POSE_LGPA True）。无 Critical/High 阻断项。

**M1 是必须知情的叙事陷阱**（非代码 bug）：exp342 的 global 因 list-return 路径走 50/50 global/part split（POSE_PART_WEIGHT=1.0），有效权重 0.5x，而 exp341 的 global 是 1.0x。这是标准 LGPA 训练范式、对 LGPA 实验完全正确，但意味着"exp342 global ≈ 59.8"的中间台阶口径已变，结果分析须用 exp342 自身 global vs equal_concat 隔离 pose，且别把 global 可能的下降误判为 prompt 失效。M2/L1-L3 为核对通过/文档陈旧/双 CLIP 维度并存（均确认不冲突）。R1/R2 为多损失竞争与 0.5x global 的调参/解读 flag，不阻断训练。

**审查通过**（强烈建议：results/monitor 同记 exp342-global 与 equal_concat 两数并标注 global 为 0.5x 口径；修 pose_backbone_model.py:846 注释；可选清理 L1 config 注释）。

## 变体
exp353 = exp342b 去CLIP, 配置级, 代码已审. 审查通过.
