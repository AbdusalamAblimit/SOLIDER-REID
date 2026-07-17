# Claude Broad Review — exp340b（固定 band LGPA · un-detach + 低 global loss）

**审查范围**：`experiments/exp340b_fixedbands_undetach/design.md` / `configs/occluded_duke/exp340b_fixedbands_undetach.yml`（逐行 diff vs exp340）/ 两个 flag 的代码消费路径（`model/pose_backbone_model.py` 的 `_lgpa_detach`、`loss/make_loss.py` 的 `GLOBAL_LOSS_SCALE`、`processor/processor.py` 的 LGPA assign-loss 接线）。
**审查方式**：`diff` 精确比对配置；grep + 逐行读取确认 exp340b 不引入任何新代码，仅切两个已存在、exp340 已双审查通过的 flag；据 loss 代码定量评估「global 降到 0.1 是否欠监督」与「un-detach 把固定 canonical assign 梯度引入 backbone」的风险。
**对照基线**：exp340（`exp340_swin_lgpa_fixedbands.yml`，detached + global 0.5，已 Claude+Codex 双审通过）。global baseline = 59.0。

---

## 核心结论先行

exp340b 是 exp340 的**纯配置变体**，`diff` 证实只改三处：`POSE_LGPA_DETACH True→False`、`GLOBAL_LOSS_SCALE 0.5→0.1`、`OUTPUT_DIR`（指向自己的独立目录，无 exp340 当年的 C1 覆盖问题）。两个 flag 均为既有、已审查路径，**无新代码、无新 config key、无 NaN/crash 风险、eval 前向与 exp340 完全相同**。设计假说（降权 global 后翻转 detach regime）成立且非冗余。**无 Critical / High 问题**。两个 Medium 是「科学归因 + 监控提示」而非阻断项。**结论：审查通过**。

---

## 配置 diff 复核（精确）

`diff exp340_swin_lgpa_fixedbands.yml exp340b_fixedbands_undetach.yml` 仅 4 行不同：
- L1 注释（文档，无功能）
- **L31** `POSE_LGPA_DETACH: True → False` ✓ 设计意图（part 梯度回传 backbone）
- **L33** `GLOBAL_LOSS_SCALE: 0.5 → 0.1` ✓ 设计意图（global 降权）
- **L80** `OUTPUT_DIR` → `./log/occluded_duke/exp340b_fixedbands_undetach` ✓ 独立目录，不覆盖 exp340

L5–30、32、34–79 **逐字相同**：detach 以外的 LGPA 超参（CLIP_DIM 512 / NUM_HEADS 8 / POSE_TEMP 1.0 / ASSIGN_WEIGHT 0.5 / FIXED_BANDS True / TEST_FEAT equal_concat）、384×128、SGD/LR 0.0008/120ep、PLBOA/OA-SD/parallel-aug 全关、Swin-Tiny。**无任何意外差异**，确认是「只翻两个相关旋钮」的受控变体。

## 代码「无新代码」声明复核（通过）

- `POSE_LGPA_DETACH`：`pose_backbone_model.py:183` 读取（default False），`:637` 消费 `lgpa_input = featmaps[-1].detach() if _lgpa_detach else featmaps[-1]`。设 False → `lgpa_input=featmaps[-1]`（不 detach）→ LGPA 的 CE/triplet/assign 梯度全部回传 backbone。**既有路径**，exp340（True）走的是同一行的另一分支。
- `GLOBAL_LOSS_SCALE`：`config/defaults.py:141` 默认 1.0；`make_loss.py:213/218/254/259` 消费。exp340 已用此 flag（0.5），exp340b 仅改值为 0.1。**既有路径**。
- assign-loss 接线：`processor/processor.py:1027-1032`，`lgpa_assign_w=POSE_LGPA_ASSIGN_WEIGHT(0.5)`，`loss = loss + 0.5*assign_loss`，**不受 GLOBAL_LOSS_SCALE 影响**。
- 结论：exp340b 不动任何 .py，「无新代码」**属实**。

---

## Critical

无。（exp340 当年的 C1 OUTPUT_DIR 覆盖、C2 `.git_token` 均已修复，且不适用于本变体——OUTPUT_DIR 正确且唯一，无新 token。）

## High

无。代码/配置/数据流层面无 High 级问题。

---

## Medium

### M1. 同时改两个 flag → 若 exp340b 赢，增益无法单独归因
- exp340b 相对 exp340 同时翻转 `DETACH` 与 `GLOBAL_LOSS_SCALE`，严格意义上**不是单变量**。design.md「单变量：仅差 DETACH + GLOBAL_LOSS_SCALE」措辞偏松。
- 二者是设计上**耦合的 regime 翻转**（"un-detach 必须配 global 降权才有意义"），作为假说验证变体可接受；但若 part_only/equal_concat 超过 global，**无法判断**增益来自 un-detach 还是 global 降权。
- 建议（非阻断）：exp340b 若 win，补一个单旋钮消融（un-detach @ global 0.5，或 detached @ global 0.1）做归因；design.md 把「单变量」改写为「联合 regime 翻转」更准确（正文 L9 已隐含此意）。

### M2. un-detach 把「固定 canonical」assign 梯度引入 backbone —— 内容无关的强空间先验
- exp340（detached）下 assign-loss（KL，w=0.5）只训 `clip_part_head`，**不碰 backbone**。exp340b（un-detach）下，因 assign 注意力建在非 detach 的 `featmaps[-1]` 上，**assign 梯度现在直接塑造 backbone**。
- 关键：FIXED_BANDS 的 GT 是**对所有图相同的固定竖直 band 布局**（与图像内容/真实姿态无关）。于是该项在 w=0.5 + 全量 backbone 梯度下，逼 backbone 产出「CLIP 跨注意力匹配固定 canonical 分块」的空间特征——在遮挡数据上（真实部位常缺失/位移），可能逼 backbone 幻想出与实际遮挡内容矛盾的 canonical 结构，**有损判别性**。这是本变体最大的科学风险，design.md 仅泛泛写「un-detach 扰动/退化」，未点名这条「固定先验正则化 backbone」的具体路径。
- 稳定性不受影响：exp340 审查已验证 canonical GT 不触发 NaN（峰值 1.0、无全零通道、bg `clamp(min=0)`、KL `clamp(min=-30)`+`isfinite` 双守卫），这些守卫在 detach 上游，un-detach 下依旧生效——**非 crash 风险**。
- 建议（非阻断）：训练时盯 `loss_details` 的 `lgpa_assign`（processor:1032）与 `id_global`/`id_part`（make_loss:215-216）；若 `lgpa_assign` 压过 ID 项或 `id_global` 持续恶化，按 design 失败预案回调 global 到 0.2–0.3，或降 ASSIGN_WEIGHT。日志已覆盖这些量，**监控充分**。

---

## Low

### L1. design.md 称基线为「exp340a」，但实际基线 config/目录是 `exp340_swin_lgpa_fixedbands`（无 exp340a）
- 仓库无 `exp340a` 配置或目录；对照实际上是 exp340b vs **exp340**（detached 基线）。建议在 design.md 注明「exp340a ≡ exp340 detached 基线」，避免读者找不到 exp340a。

### L2. 对照跨设备（exp340/exp340a 在 4090，exp340b 在 3090）
- design L23 标注两机。按 `remote_server.md`，跨设备 Δ<0.5% 可互信，且最干净的控制是同 config 仅切两 flag——可接受，仅提示主表数字标清机器。

---

## 澄清性核查（全部通过）

- **backbone 是否欠 ID 监督（global 0.1）？** 否，风险 LOW。`make_loss.py:214` `ID_LOSS = 0.1*w_g*global_id + w_p*part_id_avg`：part 项 = LGPA part 分类器（`score[1:]`，输出 num_classes、对 target=人物 ID 做 CE）的均值，**不被 global_loss_scale 缩放**（全量 w_p）；un-detach 下这份全量 part-ID CE 回传 backbone。triplet 同构（`:255` part_tri_avg 全量 wt_p，un-detach）。POSE_PART_WEIGHT 未覆盖（default 1.0 → w_g=w_p=0.5），故 backbone 的 ID 信号 = 0.05·global_id + 0.5·part_id_avg，**由 part-ID 主导**——这正是设计意图（part loss 塑形 backbone），backbone 不欠 ID 监督。唯一偏弱的是 global **特征头**（feat[0]）本身只剩 0.1 监督，而 eval 用 equal_concat=global⊕part，global 半边若退化可能拖累 concat——这是 design 失败预案 #2 已覆盖的真实下行情形，有回调方案。
- **global 降权是否真的改变了「un-detach 历史 +4.4 受损」的 regime？** 是，合理。历史 +4.4 是在 global 0.5（强 global 主线）下测的，un-detach 让 part 梯度扰动强 global 目标；exp340b 把 global 砍到 0.1，竞争动力学确实不同（part 主导塑形）。假说非「重测已知死配置」，**有新意、可证伪**。
- **train/test 对称**：DETACH 仅影响训练期梯度路由，eval 无梯度 → True/False 对 eval 前向特征**零影响**；GLOBAL_LOSS_SCALE 是 train-only loss 缩放。故 exp340b 的 eval 路径与 exp340 在同权重下**完全一致**，equal_concat 门控（`pose_test_feat != 'global'`）不变。**对称成立**。
- **分支隔离**：exp340b 配置下 GCN（POSE_PSG_STAGES 空、无 GCN flag）、OA-SD、PLBOA、parallel-aug 全关，前向走纯 LGPA 分支（`:635-659`），无 dual-branch 交互。**隔离干净**。
- **AMP 安全**：detach 与否不改 dtype；exp340 审查已验证 fp16 featmap + fp32 pose_bias 路径安全，un-detach 只改梯度流不改 dtype，**AMP 安全**。
- **center loss**：`IF_WITH_CENTER:'no'`，center 分支关闭，无交互。
- **日志充分性**：id_global / id_part / tri_global / tri_part / lgpa_assign 均入 loss_details，足以观察 part 是否真在塑形、global 是否塌、assign 是否过强。**满足「日志够重」铁律**。

---

## 结论

**Verdict：审查通过。**

exp340b 为 exp340（已双审通过）的纯配置变体，`diff` 证实只切两个既有、已审查的 flag（`POSE_LGPA_DETACH:False`、`GLOBAL_LOSS_SCALE:0.1`）+ 独立 OUTPUT_DIR，**无新代码、无 NaN/crash 风险、eval 前向不变、分支隔离干净、日志充分**。假说（降权 global 翻转 detach regime）成立且可证伪。两个 Medium（M1 双旋钮归因、M2 固定 canonical assign 梯度入 backbone 的判别性风险）是科学归因与训练监控提示，**非阻断项**，靠现有 loss_details 日志即可观测、design 失败预案已含回调方案。两个 Low 为文档措辞澄清（exp340a≡exp340、跨设备标注）。可进入 Codex 第二轮审查 / 启动训练。
