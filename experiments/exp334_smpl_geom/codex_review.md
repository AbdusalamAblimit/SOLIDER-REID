# Codex Review — exp334 SMPL 几何空间先验

**Tool**: `codex exec -s read-only`（聚焦提示 + 输出到文件）

## Round 1 — Verdict: needs-attention
findings（已全部修复）：
- **High（翻转不同步）**：train transform 有 `RandomHorizontalFlip(p=0.5)`，但热图仍用原始 pj2d 坐标 → ~50% 样本的身体先验相对图像 token **镜像错位**，会污染训练。比已记录的 crop jitter 严重得多。**修复**：把翻转移到 `GeomDataset` 里**手动做**（图像 + pj2d 一起翻，pj2d x→W-1-x），从 transform 删除 `RandomHorizontalFlip`。两臂一致（geom off 也手动翻图、无热图）。
- **Medium（bn_body 污染）**：`bn_body` 对全 batch 做（含 valid=0 的 missing-token 行）→ 污染 BatchNorm running stats。**修复**：`bn_body` 改 **LayerNorm**（逐样本，无 batch 污染），与 exp333 同理。
- **Low（注释口径）**：脚本头说"identical to exp333 baseline / appearance path untouched" 不准（body loss 回传进共享 backbone）。**修复**：改注释——geom 臂 alpha=0 是诊断量、非 appearance baseline；headline = geom-on best-alpha vs geom-off 自训对照。
- **额外（eval drag）**：顺手加了 **eval valid-gate**（缺检测 item 的 body 特征置 0 → 只用 cls 比），避开 missing-token 聚类拖累（同 rawbeta）。
- **Checked OK**：_tokens 复制 forward_features 正确、body-pool 形状/AMP 无 NaN、优化器覆盖新参、scheduler/eval/R1_mAP_eval、融合 cls(before-BN)+body(after-LN) L2 归一 concat 无泄漏。

## Round 2 — Verdict: approve（代码可训练）
复审确认：运行正确性 PASS（无 Critical/runtime/NaN）、单变量隔离（on vs off）成立、AMP/scheduler/eval/optimizer 全对、`_tokens` 返回 (cls, patches=(B,128,768))。翻转同步修复确认（未再 flag flip-High）、bn_body→LayerNorm 无污染、eval valid-gate 正确。**唯一保留 = 报告口径（3 文档必修项），已全部补入 design.md**：① alpha=0 是 backbone 正则诊断量非 baseline；② crop 错位记录在案（≤0.6 patch 零均值软噪声，test 对齐）；③ 重遮挡子集单列。smoke 实测端到端跑通（Epoch1 app+body 双损失、5 alpha eval、done）。

## 结论
codex 审查通过（Round2 approve，代码可训练；3 文档口径已补入 design.md）。配合 Claude review（代码 PASS）+ smoke 实测，可训练。
