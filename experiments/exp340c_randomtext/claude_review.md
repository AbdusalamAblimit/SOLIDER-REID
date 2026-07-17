# Claude Broad Review — exp340c (RANDOM-TEXT 归因对照)

**审查范围**: design.md / 新代码(pose_backbone_model.py L195-204) / config/defaults.py / exp340c YAML / 科学公平性
**审查方式**: 逐行阅读 + 经验性验证(系统 python3 + torch 2.8.0 实跑 random 向量构造逻辑)
**结论**: 审查通过

---

## 一、实验目的与单变量隔离

exp340c 是 exp340a 的归因决定性对照。exp340a(固定 CLIP 文本 + 固定 canonical 姿态)= part_only 59.4(+0.6 over global 58.8)。本实验把 CLIP 文本部位原型替换成**固定 random 单位向量**，其余完全不变，以判定 +0.6 来自 CLIP 词义还是 canonical 姿态先验。

YAML diff(exp340 vs exp340c)确认**严格单变量**：
- 仅注释、新增 `POSE_LGPA_RANDOM_TEXT: True`、`OUTPUT_DIR` 三处不同。
- `POSE_LGPA_FIXED_BANDS: True`、`POSE_LGPA_DETACH: True`、`GLOBAL_LOSS_SCALE: 0.5`、384×128、assign_weight 0.5、关闭 PLBOA/PARALLEL_AUG/OA-SD —— 全部一致。
- canonical 姿态路径(`_canonical_heatmap`, L493-510)与 `_lgpa_heatmap`(L512-519)不受 RANDOM_TEXT 影响，两实验喂入 LGPA 的 heatmap 完全相同。

唯一变量 = 文本来源(CLIP 语义 vs random)。✓

## 二、新代码正确性(pose_backbone_model.py L195-204)

逐行核对，并用 torch 实跑验证：

1. **Shape**: `torch.randn(self.clip_part_head.num_labels, self.clip_part_head.clip_dim)` = (6, 512)。
   - `num_labels = NUM_PARTS+1 = 6`(clip_part_head.py L58)，`clip_dim = 512`(config L218)。
   - buffer `clip_text_features` 注册形状 = (num_labels, clip_dim)(clip_part_head.py L106-108, L120)，shape assert 也是 (6,512)。**完全匹配**。✓ (实跑确认 shape=(6,512))

2. **frozen / buffer 不训**: `clip_text_features` 经 `register_buffer` 注册(L108/L120)，**非 nn.Parameter**(全文件无对该名的 Parameter)。`copy_` 在 `with torch.no_grad()` 内、写入已存在 buffer，不破坏注册、不引入梯度。random 向量与 CLIP 文本同为 frozen buffer —— 两边都是固定 frozen query，公平。✓ (实跑确认 copy_ 后张量 requires_grad=False)

3. **单位向量**: `F.normalize(..., p=2, dim=-1)` 使每行 L2=1，与 CLIP 文本在 clip_part_head.py L119 的 L2-norm 一致。✓ (实跑确认 6 行范数均=1.0)

4. **确定性 / 可复现**: `torch.Generator().manual_seed(42)` 显式喂给 `randn(generator=_g)`，**不依赖全局 RNG 状态**，跨进程/跨运行完全可复现。✓ (实跑两次构造 torch.equal=True)

5. **device / dtype**: 在 CPU 上构造，`.float()` 后 `copy_` 进 buffer；buffer 随后由外层 `.to(device)` 统一搬运。dtype=float32 与 CLIP buffer 一致。✓

6. **构造顺序**: 整段在 `if self.use_lgpa:`(L174)块内，`self.clip_part_head = CLIPPartHead(...)` 在 L176 先构造，random 覆盖在 L195-204 **之后**执行，`self.clip_part_head` 必然存在。✓

7. **随机性真实**: 实跑 6 行间最大 off-diag |cos|=0.059，近似正交，确为真随机原型(非塌缩/共线)。✓

## 三、test-time 一致性(关键，超出原始审查清单但已核查)

`clip_text_features` 是 buffer → **会写入 checkpoint**。test.py 经 `load_param`(make_model.py L435-438)把 checkpoint 每个 key `copy_` 进 state_dict，包含该 buffer。由于训练全程 buffer frozen，checkpoint 里存的就是 seed-42 random 向量，load 后与 build 时 override 完全一致 —— **无 stale-value 覆盖、无前后不一致**。✓

## 四、config/defaults.py

`_C.MODEL.POSE_LGPA_RANDOM_TEXT = False`(L225)默认 False，安全，不影响任何已有实验。✓

---

## Findings by severity

### Critical
- 无。

### High
- 无。

### Medium
- 无。

### Low（不阻断，记录）
- **defaults.py L225 注释复制粘贴残留**: 该行尾部跟了一段来自 FIXED_BANDS 的旧注释("# Fixed-semantics: replace per-image pose ...")，与 RANDOM_TEXT 语义无关。纯注释噪声，不影响功能。建议清理但非必须。
- **YAML 行内注释错位(2 处)**: exp340c L27 `POSE_LGPA_RANDOM_TEXT: True` 后注释写成"# 固定 canonical pose 替代 per-image pose"(那是 FIXED_BANDS 的描述，被误粘到 RANDOM_TEXT 行)；exp340c L38 沿用 exp340 的"FIXED_BANDS 下 pose 被忽略"注释（本实验仍成立，因 FIXED_BANDS 仍 True）。仅注释文字误导，不改变任何行为。建议改为"# 用固定 random 向量替代 CLIP 文本原型"。
- **科学解读提示(非代码问题)**: 此对照精确隔离的是"CLIP 文本语义结构 vs 随机原型"。若 random≈CLIP，结论是"在 frozen-band+canonical 配置下 CLIP 词义零增量贡献"，与历史 5 次"CLIP 是壳"证据一致；这是设计预期，不是 bug。结论表述时应限定在"此固定 frozen + canonical-pose 配置下"。

---

## 验证清单（全部通过）
- [x] design.md 合理、单变量假设清晰、可证伪
- [x] 新代码 shape 匹配 buffer (6,512)（实跑确认）
- [x] copy_ 在 no_grad 内、写入已存在 frozen buffer、不破坏注册
- [x] F.normalize 产生单位向量，与 CLIP 文本 L2-norm 对齐
- [x] generator-seeded randn 确定性可复现、CPU 构造、不污染全局 RNG（实跑确认）
- [x] 构造顺序正确（clip_part_head 先建，override 在后）
- [x] buffer 非 Parameter → 不被训练（实跑确认 requires_grad=False）
- [x] YAML 严格单变量（仅 RANDOM_TEXT + OUTPUT_DIR + 注释）
- [x] defaults 默认 False 安全
- [x] test-time load_param 与 build-time override 一致，无 stale 覆盖
- [x] assign loss / visibility / canonical pose / detach / global 0.5 均未改动

## 结论

**审查通过。** exp340c 是一个干净、公平、单变量的归因对照：random 与 CLIP 文本同为 (6,512) frozen buffer，唯一差异是"CLIP 语义结构 vs 固定随机原型"。新代码 7 行经逐行 + 实跑双重验证，无 Critical/High/Medium 问题。仅 3 处 Low 级注释残留（defaults 行尾旧注释、2 处 YAML 行内注释错位），均为文字噪声，不影响功能与科学有效性，可选清理。
