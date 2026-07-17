# 实验 exp340c: 归因决定性对照 — CLIP 文本 vs RANDOM 文本（同固定 canonical 姿态）

## 动机（一锤定音）
- exp340a（固定 CLIP 文本 + 固定 canonical 姿态）= part_only 59.4（+0.6 over global 58.8）。
- **关键未解问题**：这 +0.6 是 CLIP 文本词义给的，还是固定 canonical 姿态先验给的？
- 全部历史证据指向「CLIP 文本是壳，定位才是价值」。本实验**直接证伪/证实**。

## 核心假设（可证伪）
**把 CLIP 文本部位原型换成固定 random 单位向量（同 shape、frozen），其余完全不变。**
- random part_only ≈ 59.4（≈ CLIP）→ **CLIP 词义零贡献**，+0.6 全是 canonical 姿态先验（那 ≈ PCB，不是 CLIP 创新）。
- random part_only **< 59.4**（明显低于 CLIP）→ **CLIP 词义真添了判别信息**，这才配叫「新 CLIP 用法」，用户的理想成立。

## 技术方案（单变量）
- `MODEL.POSE_LGPA_RANDOM_TEXT`（新 flag）：构造 clip_part_head 后，用 `torch.Generator().manual_seed(42)` 生成 `(num_labels, clip_dim)` 的 random 单位向量，`copy_` 覆盖 `clip_text_features` buffer。frozen（buffer 不训）。
- = exp340a config + `POSE_LGPA_RANDOM_TEXT: True`。**唯一变量 = 文本来源（CLIP 语义 vs random）。**

## 预期结果
- 预判（基于 5 次「CLIP 是壳」证据）：random ≈ CLIP（≈59.4），CLIP 词义零贡献。
- 若意外 CLIP > random：CLIP 真有用，翻案。

## 对照组
- **exp340a（CLIP 文本 + canonical，part_only 59.4）vs exp340c（random 文本 + canonical）**。
- 单变量：仅 `POSE_LGPA_RANDOM_TEXT` True/False；canonical 姿态、detach、global 0.5、384×128 全同。
- 评测同口径 test.py（global / part_only / equal_concat）。

## 审查说明
新代码仅 7 行（random 向量覆盖 buffer），其余复用 exp340a（已双审通过）。下方 review 复核 override 正确性 + 单变量隔离。
