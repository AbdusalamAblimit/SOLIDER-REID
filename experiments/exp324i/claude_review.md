# Claude Broad Review — exp324i（解相关感知的 DINO-LoRA）

**审查类型**：开训前全范围 Claude Broad Review（Opus 4.8）
**日期**：2026-06-16
**审查轮次**：第 1 轮
**范围**：design.md / exp324i_swin_cache.py / exp324i_lora_decorr.py（对照 exp324d_lora.py）/ 复用 helper（exp324b、exp324_dino、exp324f 参考）/ make_dataloader / eval_fliptest_maxsim / occluded_duke 数据类

逐行读完上述全部文件。下面按协议 a–h 逐项给结论，再汇总 findings。

---

## a. design.md — 假设是否成立 + 是否真机制还是小调参

**结论：是真机制（distinct objective），值得一个 GPU-night；即使为负也是把"判别性-互补性张力"从观察升级为强结论的关键对照实验。**

- 假设链清晰：exp324h 已量化出 ID-adaptation 让 DINO 与 Swin 趋同（top-10 Jaccard 0.062→0.253），融合只 +0.37。exp324i 显式加一个跨网络解相关项把 `d` 推进 Swin 的线性正交补，赌融合 > Swin 单独。这是直接攻击 headline 张力的因果干预，不是在已投 PSG branch 上堆模块。
- 解相关形式**正确且维度安全**：原稿 `cos(d,s)^2` 因 d=512、s=768 不同空间确实无定义；改成 Barlow-Twins 风格跨协方差 `C=(d̂ᵀŝ)/B`（Dd×Ds，**维度无关**），`L=||C||_F²` 的 mean。维度问题已在设计阶段修掉，代码与设计一致。
- 诚实判断：该方向属于 FM-import（先前已大范围证负，见 MEMORY）。但 decorr 目标本身是一个**先前没试过的明确机制**（不是 visibility/completion/scorer 那批雷区），且实验自带"成→method / 败→强诊断结论"的双向价值。**符合"开实验"的门槛**，不属于逃避创新的小调参。
- 设计自报的失败模式（正交补里 ID 信号更弱 / 全局线性解相关不针对遮挡盲点 / 95.8% 全可见墙 / decorr 与 ID 冲突）都列出了，预期合理。

→ 无阻断问题。**Low**：见 d 的 scope mismatch（设计应显式承认）。

## b. exp324i vs exp324d — diff 是否 surgical，λ=0 是否精确复现 exp324d

逐行 diff（去空行后）确认：**新增内容仅为 decorr 相关**——
docstring、`load_swin_cache()`、`decorr_loss()`、`--decorr_weight`/`--swin_cache` 两个 arg、`output_dir` 默认改 exp324i、`use_decorr` 门、cache 加载（`if use_decorr` 包裹）、step 内 `dec_loss`（`if use_decorr` 包裹）、loss 加 `decorr_weight*dec_loss`、meters 7→8 与打印加 `decorr=`。**没有任何与 decorr 无关的改动。**

**λ=0 精确复现 exp324d，验证如下**：
- `use_decorr = args.decorr_weight != 0.0`，默认 0.0 → False。
- cache **不加载**（`if use_decorr` 分支跳过 `load_swin_cache`，`swin_feat_t=None`）→ 无 npz 依赖、无额外 RNG 消耗。
- step 内 `dec_loss = torch.zeros((), device=device)`，loss 项 `args.decorr_weight(0.0) * dec_loss(0)` = 常数 0，对 forward 值和 backward 梯度都是 no-op；不进计算图、不扰动 LoRA/head 梯度。
- seed/sampler/optimizer/scheduler/数据全未变；meters/打印的 reorder 只影响日志，不影响权重。
→ **λ=0 与 exp324d 数值等价（同种子逐 step 等价）**，单变量隔离成立。

→ 无阻断问题。

## c. decorr_loss() 数学正确性

逐行核对（行 238–250）：
- `s = s.detach()` → 只 `d` 回传梯度。✔（且调用处传入的 `s` 本就来自 no-grad 缓存张量，detach 是双保险。）
- z-score per-dim across batch，`unbiased=False`（population std），eps=1e-5 加在 std **外**：`/(std+eps)`。✔ eps 位置正确（分母永不为 0）。
- `C=(d̂.t()@ŝ)/B`，shape (Dd,Ds)=(512,768)。✔ 与设计一致。
- `L=(C**2).mean()` ∈ ~[0,1]，scalar。✔
- 梯度路径：`glob`（head masked-mean 后、BN 前，未 L2）→ `decorr_loss` → 反传进 head.proj 与 LoRA（`glob` 由 `pool_parts_diff(bmm)` 可微连到 patch 再到 LoRA）。✔ 梯度确实流入 LoRA。

**退化/NaN 风险评估**：
- 某维 std=0（batch 内该维恒定）：分母=eps=1e-5，不除零；该维 z-score 会被放大但 `d̂`/`ŝ` 仍有限，C 有限，L 有限。对 `d` 侧恒定维 → 对应 `d̂` 列趋 0（分子 d-mean=0），不放大、贡献近 0；对 `s` 侧恒定维同理。**无 NaN/Inf 风险。**
- B=64 固定，population std 对 B≥2 良定义；不存在 degenerate-batch。
- `glob` 是 fp32（全程无 AMP，见 g），`s` fp32，matmul 安全。

→ 无阻断问题。**Low**：B=64 下 1/B 标度与 z-score 已使 L 量级稳定，λ 选 1/2 合理（设计已计划 λ sweep）。

## d. Swin 特征语义 — 解相关的是 GLOBAL-only，张力/eval 是 FULL MaxSim（scope mismatch）

**这是本实验最该写进文档的局限，严重度 Medium（不阻断，但影响 null 结果的可解释性）：**
- 缓存的 `s` 是 Swin **holistic global**（`feat[:,:768]` L2-normed），与 eval 全局 cos 项一致。
- 但 headline 张力（Jaccard 0.062→0.253）与最终 fusion eval 用的是 **FULL MaxSim = global + 5 part 子空间**。decorr 只把 DINO-global 推离 Swin-**global**，**没有**推离 Swin 的 part 子空间。
- 后果：即便 decorr 成功降低 global 线性相关，融合仍可能因 part 子空间趋同而不涨 → **null 结果会出现"到底是张力不可破，还是只解相关了 global 这一路"的归因含混**。
- 但这不是 bug，是 scope。设计已把"解相关是全局线性的、不针对 Swin 盲点"列为失败模式之一，方向正确，只是没显式点出"只对 global 不对 part"这一精确口径。

→ **建议（非阻断）**：在 design.md 预期结果里显式写明"decorr 仅作用于 global 子空间，part 子空间趋同未被约束；若 fusion 仍不涨，需区分'张力不可破'与'仅 global 被解相关'"。判 method 成/败时以此为前提解读。已记为 Medium finding，不阻断训练（实验仍能给出"对 global 施压"这一干净因果）。

## e. exp324i_swin_cache.py — 是否抽取 eval 同款 global + 确定性 + 命名对齐

- **同款 global**：行 79 `C = 1024 if "base" in TRANSFORMER_TYPE else 768`，行 95–99 `g=F.normalize(feat[:,:C])`，与 `eval_fliptest_maxsim.extract_features_flip`（行 100–102）**逐字一致**。swin_small → C=768。✔ 抽的就是 eval 全局项。
- **确定性**：用 `train_loader_normal`（make_dataloader 行 199–202：`train_set_normal` is_train=False=val transform、`shuffle=False`、`pose_val_collate_fn`），**no random aug、no flip、单次 forward**（脚本注释与代码一致，`@torch.no_grad`，无 flip 分支）。✔
- **train_loader_normal 确实是 val-transform 的 TRAIN split**：行 129–136 `train_set_normal=PoseImageDataset(dataset.train, is_train=False, img_size=SIZE_TEST,...)`。✔ index 1 = train_loader_normal。✔
- **命名无碰撞**：缓存写 `names=os.path.basename(p)`；末尾断言 `n_unique==N` 且 `N==15618`。Occluded-Duke train 15618 已被 exp324d_large 的 `train_pool_n15618` 缓存与 decisions.md 双重佐证。✔ 15618 basename 唯一。
- 健壮性：行 83–88 强制 7-tuple（POSE_ENABLED），行 129 assert POSE_ENABLED，norm/finite/unique 全 assert。✔ fail-loud。

→ 无阻断问题。

## f. cache↔train 对齐 — aligned[i] 是否对应 train_names[i]

**对齐是按名查表（name→row），不是按位置，这正是健壮性的来源：**
- `load_swin_cache`（行 147–169）：建 `name2row`，对训练脚本的 `train_names`（`list_imgs(TRAIN_DIR)`=sorted disk basenames）逐个 `name2row.get(nm)` → `aligned[i]=feats[r]`。所以 `aligned[i]` **严格对应 train_names[i]**，与缓存运行时 loader 的迭代顺序**无关**。✔
- 缓存运行（dataloader 顺序，源自 `train.list` 过滤 pid==-1）与训练运行（`list_imgs` 全盘 disk 顺序）即使**顺序不同**也不影响——按名查表自愈。✔
- 名字集合一致性：Occluded-Duke `bounding_box_train` 无 pid==-1 junk（Duke junk 仅在 gallery），故 `train.list`（15618）= disk jpg（15618）= `list_imgs` 输出。两侧 basename 集合相等。
- 万一不等：`if missing: raise KeyError` **fail-loud 阻断**，绝不静默错位。✔ duplicate 也有 assert。

→ 无阻断问题（名字集合一致已由 15618 旁证；即便不一致也是 loud-fail）。**Low**：dry 阶段建议先 `--decorr_weight 1` 跑 `--dry_run` 确认 cache 加载 0 missing。

## g. train/test 对称 / AMP / optimizer / dtype-device

- **decorr 只在训练**：`step()` 内、`if use_decorr` 包裹；`encode_split`/`run_eval` 完全没碰 decorr。eval 与 exp324d 逐字相同。✔ eval 不受影响。
- **新参数**：decorr **不引入任何可训练参数**（纯 loss 项）。optimizer 三组（lora / head_decay / head_nodecay）与 exp324d 完全一致，无需新增 param group。✔
- **AMP**：grep 确认 exp324i/exp324d **均无 autocast/GradScaler/half**，全程 fp32。`glob` fp32、`swin_feat_t` fp32 on cuda，`s=swin_feat_t[idx]` fp32 → decorr 全 fp32，无 dtype/device 不匹配。✔
- **device**：`idx` numpy→`torch.from_numpy(idx).to(device)` 索引 cuda 张量 `swin_feat_t`，`s` 在 cuda；`glob` 在 cuda。✔
- 显存：decorr 只多一个 (512×768) C 矩阵 + 两个 z-score，峰值增量可忽略；不改 BS、不改 micro_bs。设计已计划 `--dry_run` 验 peak-mem。✔

→ 无阻断问题。

## h. config/defaults 泄漏

- 两个脚本均为 standalone（`scripts/`），cache 脚本只 `cfg.merge_from_file` 读 exp255 config 不写；训练脚本完全不碰 repo `config/defaults.py`。
- 未改任何 repo config / defaults，**不影响任何已有实验可复现性**。✔

→ 无阻断问题。

---

## Findings 汇总

| # | 级别 | 位置 | 问题 | 处置 |
|---|------|------|------|------|
| 1 | Medium | design.md / 概念 | scope mismatch：decorr 仅作用于 Swin-**global**，而张力/fusion eval 是 FULL MaxSim（含 part 子空间）。null 结果归因会含混（张力不可破 vs 仅 global 被解相关）。 | **非阻断**。建议在 design.md 预期/解读处显式写明该口径；判 method 成败时以此为前提。不需改代码。 |
| 2 | Low | 概念/解读 | decorr 在正交补里编码 ID 可能更弱（设计已列），属预期风险非 bug。 | 观察 λ sweep 下 mAP vs Jaccard 即可量化。 |
| 3 | Low | 运行流程 | cache↔train 名字集合虽有 15618 旁证，仍建议先 `--decorr_weight 1 --dry_run` 确认 0 missing + peak-mem。 | 协议本就要求 dry-run，照做即可。 |
| 4 | Low | decorr_loss | std=0 维由 eps 兜底，无 NaN；λ 量级稳定，λ∈{1,2} 合理。 | 无需改动。 |

**无 Critical / 无 High。** Medium #1 是诚实的范围局限（影响结果解读，不影响正确性/可运行性/单变量隔离），按协议不阻断训练，仅要求文档显式承认（建议性，不作为硬 gate）。

## 关键正确性确认（复述供 hook 计数）
1. λ=0 逐 step 数值等价 exp324d（cache 不加载、0×0 no-op、RNG 不变）—— 单变量隔离成立。
2. decorr 数学正确、维度安全（512×768 跨协方差）、`s` detach、梯度入 LoRA、无除零/NaN。
3. 缓存抽取 = eval 同款 global（`feat[:,:768]` L2-normed），确定性（val transform / shuffle=False / no-flip / no-aug）。
4. 对齐按名查表，顺序无关，缺失 fail-loud；15618 唯一名旁证集合一致。
5. 全程 fp32 无 AMP，eval 路径未被 decorr 触碰，无新增 optimizer 参数，无 repo config 泄漏。
6. diff surgical：仅 decorr 相关新增，无附带改动。

## 结论
代码与设计自洽，单变量隔离严格，数学与维度正确，无 Critical/High 阻断项；唯一 Medium 为范围局限（建议补文档，非硬 gate）。实验作为打破/证伪"判别性-互补性张力"的因果干预是 sound 的，可进入 Codex 审查并训练。

**审查通过**
