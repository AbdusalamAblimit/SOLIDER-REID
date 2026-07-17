# Claude Broad Review — exp345 (Option C: pose-localized part features)

**审查范围**：`model/modules/clip_id_prompt.py`（PoseGuidedPartPool）、`model/pose_backbone_model.py`（forward C 分支 + 构造）、`configs/occluded_duke/exp345_pose_part_clip.yml`、`processor/processor.py`（clip_id_loss 消费）、`config/defaults.py`（新默认）、`design.md`。
**代码来源**：git HEAD `0562b46`（exp345 代码已 commit，working tree 无 diff；exp345 目录 untracked）。
**结论先行**：审查通过。逐行 + 数值复现，未发现 Critical/High 问题。点 2(c) 与点 6 已专门核实。

---

## 逐项核实（对照审查清单）

### 1. PoseGuidedPartPool（clip_id_prompt.py L139-168）

- **(a) shapes**：`featmap (B,C,H,W) → flatten(2).transpose(1,2) = (B,N,C)`；`k_proj` 保持 `(B,N,C)`；
  `pose = interpolate(heatmap, (H,W)) = (B,17,H,W)`；每 part `bias = pose[:,grp].amax(1).flatten(1) = (B,N)`；
  `attn = (k @ queries[i])/√C + temp*bias = (B,N)`；einsum `bn,bnc->bc = (B,C)`；
  `stack(dim=1) = (B,3,C)`。**数值复现确认**（B=2,C=8,H=12,W=4 → 输出 (2,3,8)）。✔ 正确。
  注意 pose 的空间分辨率（heatmap 96×32）与 featmap 网格（12×4）不一致，靠 `interpolate` 对齐——已正确处理（A 同款）。

- **(b) 各 part 用其自身关键点**：`pose[:, grp]` 用 `grp` 索引 17 通道热图的对应子集，`amax(dim=1)` 在该 part 的关键点通道上取 max → part-i 可见性。✔ head/torso/legs 各自 bias 正确。

- **(c) queries[i] per-part**：`self.queries = nn.Parameter(n_parts, dim)`，循环里 `queries[i]` 取第 i 行 → 每 part 独立 query 向量。✔

- **(d) softmax over spatial dim=1**：`F.softmax(attn, dim=1)`，attn 形状 (B,N)，dim=1 = N（空间）。数值复现 `attn.sum(dim=1)=1.0000`。✔

- **(e) COCO-17 全覆盖无空隙/重叠**：`PART_GROUPS = [[0..4],[5..10],[11..16]]`，数值复现 `cov == range(17)` 且无重叠。✔ 解剖学分组合理（头/躯干+臂/腿）。

- **(f) trainable + RNG 保护**：`queries`、`k_proj` 均 `nn.Parameter`/`nn.Linear`，可训练；构造点（model L231-234）用 `_rng = get_rng_state() … set_rng_state(_rng)` 包裹，保证不扰动下游模块 init（与 exp341 对齐，codex B Medium-1 修复已落地）。✔
  数值复现确认梯度同时流向 backbone featmap 与 queries。

### 2. forward C 分支（pose_backbone_model.py L600-608）

- **(a) per-part 循环正确**：`part_feats[:, kp]` 取 (B,C)，经**共享** `self.clip_id_proj`（`Linear(in_planes → clip_id_prompt.clip_dim)`，model L215）投到同一 clip 空间，对齐同一 `txt_proto`。✔ in/out 维度自洽（part_feats[:,kp] 是 (B, in_planes)，与 A 的 pose_guided_pool 输出、exp341 的 global_feat 同维，复用同一 proj 安全）。
  注意：`clip_id_proj` 输出维度来自 `self.clip_id_prompt.clip_dim`（= CLIP text_projection 维度，ViT-L-14 → 768），**不是** config 里的 `POSE_LGPA_CLIP_DIM=512`（那是 LGPA 头独立参数，C 分支未用）。两者无冲突，proj 与 txt_proto 都走 768，✔ 维度一致。

- **(b) loss 对 K 平均**：`clip_id_loss = clip_id_loss / part_feats.shape[1]`（÷3）。✔

- **(c) `0.0 + tensor` 累加**：`clip_id_loss = 0.0`（python float）起步，首次 `0.0 + supcon(...)` → Python `float.__add__` 返回 NotImplemented，转而走 `Tensor.__radd__` → 结果是 graph 上的 tensor。**数值复现确认**：最终 `clip_id_loss` 为 Tensor、`requires_grad=True`、`backward()` 成功且梯度到达 backbone。✔ 安全。
  （前置守卫 `use_clip_id_part_guided AND scene_heatmaps is not None` 保证循环至少进入一次，不会出现“0.0 始终是 float”的退化；scene_heatmaps None 时走 else 分支，见 2f。）

- **(d) txt_proto 一次性计算**：`txt_proto = self.clip_id_prompt(label, pose_vec)`（L598）在分支判断之前，K 个 part 共享。✔

- **(e) C 优先于 A**：`if use_clip_id_part_guided and scene_heatmaps … :` 先判 C，`else` 内再判 A/exp341。exp345 仅开 C（POSE_CLIP_ID_POSE_GUIDED 默认 False），逻辑正确，C/A 不会同时执行。✔

- **(f) scene_heatmaps None → graceful**：C 分支条件含 `scene_heatmaps is not None`，None 时跳过 → 落入 else → A 条件同样 None 失败 → `feat_for_clip = global_feat`（exp341 行为）。✔ 优雅退化。
  返回路径：C 分支下 `score=cls_score`（单 tensor，非 list）、`feat=global_feat`（单 tensor）、L886-887 `return cls_score, global_feat, featmaps, None, {'clip_id_loss': clip_id_loss}`。loss_fn 的 `isinstance(score,list)` 守卫（make_loss L128）对单 tensor 走纯 CE+triplet，正确；clip_id_loss 经 5-tuple（processor L598-599 解包）在 L1297-1302 加权累加。✔ 端到端贯通。

### 3. config 单变量

`diff exp341 vs exp345` 仅 3 处：`POSE_CLIP_ID_PART_GUIDED: True`（新增，唯一方法变量）、`POSE_TEST_FEAT: global`（vs exp341 的 equal_concat）、`OUTPUT_DIR`。
TEST_FEAT 差异**是预期且正确的**：exp345 测试描述子 = global（part_pool 仅训练端，见点 4），exp341 因带 LGPA 分支才用 equal_concat。这不破坏单变量——两者训练端方法的唯一差异仍是 PART_GUIDED；TEST_FEAT 只是测试端读取方式，与“C 是否帮 backbone”的假设正交。✔
其余继承 exp341（Swin-Tiny、384×128、ViT-L-14 frozen、temp 0.07、weight 1.0、PSG_STAGES=[] 无 PSG、LGPA/PLBOA/OASD/aug 全关）。✔ 干净隔离。

### 4. test-time 无泄漏

prompt learner（CLIPIDPromptLearner）与 part_pool 仅在 `self.training` 分支内调用；eval 分支（L889+）descriptor = global（TEST_FEAT=global → 不进 gcn_feats 组装，直接 `test_feat = feat/global_feat`）。✔ 无 prompt/part 泄漏到推理。

### 5. clip_id_proj 维度一致性

`clip_id_proj` 在 A（pose_guided_pool 输出 (B,in_planes)）、exp341（global (B,in_planes)）、C（part_feats[:,k] (B,in_planes)）三者间共享，输入恒为 in_planes、输出恒为 clip_dim。✔ 已数值确认 part_feats[:,k] = (B,C=in_planes)。

---

## 点 6 深度分析：K 个 pose-localized 部位特征全部对齐同一 ID 原型——合理吗？

**结论：机制上自洽、不是 bug，但存在“过约束”风险，这正是本实验要实证回答的问题——审查放行，把判断交给数据。**

- **不会塌缩成“强迫 head==legs==原型 逐元素相等”**：对齐用的是 `supcon_i2t`（L100），它在 L2-归一化后算 batch 内 cosine logits + log-softmax，正样本 = 同 ID。优化的是“proj(part_k) 与本 ID 原型的相对相似度高于其它 ID”，**不是** L2 回归到原型。所以 head/legs 各自的 proj 只需把**本 ID 的判别方向**对齐到原型，不要求三个 part 的原始特征相同。投影矩阵 `clip_id_proj` 共享，但 part_feats 本身不同 → proj 后可落在原型邻域的不同点，只要类间可分即可。机制上**不强制 part 间相等**。

- **“有益”的可能机制（假设成立）**：每个身体区独立被推向“本 ID 在 CLIP 空间可判别”，梯度经 part_pool 的 attention 流回 backbone（已验证），逼 backbone 在 head/torso/legs 各区都编码 ID 判别信息 → 对遮挡（某区缺失时其余区仍判别）更鲁棒 → global 可能涨。这是 design 的核心赌注，合理。

- **“过约束”的真实风险（失败模式）**：把 legs 特征对齐到一个主要由全身外观主导的 ID 原型，对腿部被遮挡/信息少的样本，可能注入与该区不符的梯度，干扰 backbone。design.md 已正确预判失败信号 = `global < exp341 的 59.8`。三个 part loss 等权平均（÷3），未对可见性加权——若某 part 整体不可见（bias≈0，attention 退化为纯 query-driven 全局软池化），该 part 仍被强行对齐原型，是潜在噪声源。**但这属于方法层面的“想法对不对”，不是代码 bug**；属于实验该证伪/证实的范畴，符合本仓库“先实证再下结论”的原则，不构成阻断。

- **与 A 的关系**：A 已测（exp343），C 是 A 的细粒度版。C 比 A 多的“新意”仅在“K 个区各自对齐”这一点——较增量，但作为 A→C 的消融阶梯成立，不算逃避创新（属同一 CLIP-机制探索线的合理细化）。

---

## Findings by severity

- **Critical**：无。
- **High**：无。
- **Medium**：无（代码层面）。
- **Low / 观察（非阻断）**：
  1. **part loss 等权、无可见性加权**：某 part 完全不可见时（bias≈0）其特征退化为纯 query 驱动的全局软池化，仍被强行对齐 ID 原型，可能引入噪声梯度。这是方法设计的开放问题，design.md 已预判失败信号；不改也可跑，结果若 global<59.8 即印证此风险。
  2. **`design.md` / `config` 头部注释残留 exp244/LGPA 语境**（“基于 pose_psg_lgpa_detach.yml”“POSE_TEST_FEAT=equal_concat(LGPA) vs global”），与实际 C 分支（CLIP-ID part-guided、TEST_FEAT=global）措辞不完全贴合，纯文档表述问题，不影响运行。建议跑前顺手澄清，但不阻断。
  3. **多人场景**：C 用 `scene_heatmaps`（max-merge 全场景），多人时 part bias 可能混入干扰人；Occluded-Duke 主基准影响小，记录备查。

## 结论

PoseGuidedPartPool 形状/覆盖/per-part bias/softmax 轴/可训练性/RNG 保护全部正确（含数值复现）。forward C 分支 per-part 循环、共享 proj、÷K 平均、`0.0+tensor` 图连接（已数值确认 requires_grad+backward）、txt_proto 共享、C>A 优先级、None 退化、5-tuple→processor 贯通，均正确。config 单变量隔离干净（TEST_FEAT 差异为预期、与假设正交）。test 端无泄漏。点 6 的“K-to-one-prototype”机制自洽、非 bug，过约束风险属实验待证范畴。

**审查通过。** 可进入第二轮 Codex Review。
