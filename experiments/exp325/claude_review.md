# Claude Broad Review — exp325 (+ exp324f）

**Reviewer**: Claude (Opus 4.8) broad review 子代理
**Date**: 2026-06-16
**Round**: 1
**Verdict**: 审查通过 (PASS)

审查范围：`scripts/exp325_train_head.py`、`scripts/exp324f_swin_distmat.py`、`scripts/exp324f_fuse.py`、对照基线 `scripts/exp324b_train_head.py`、`scripts/exp324_dino.py` 全部 helper、`scripts/eval_fliptest_maxsim.py`、`datasets/make_dataloader.py` + `datasets/pose_dataset.py` 的 val 顺序、两份 `design.md`。逐行阅读 + 远程实证（dump head_60 键/形状、确认缓存 tag、AutoConfig 确认 dinov2-base/large 均 patch14 / 768·1024 / 无 register token、3090 idle）。

## 关键校验结论

### exp325 monkeypatch 正确性（核心）
- `e324.HIDDEN=1024` 与 `e324.load_model=large` 在 `import exp324b_train_head` **之前**设置 → exp324b 的 `from exp324_dino import (...,HIDDEN,...)` 拿到 1024（`assert b.HIDDEN==1024` 守卫）。
- `build_part_pose`/`build_part_grid`/`extract_split` 定义在 exp324_dino，调用时读模块全局 `HIDDEN` → 用 1024（实证：三函数均引用模块全局，非捕获局部）。
- `extract_part_features`（exp324b）分配 `(N,5,1024)` 并 reshape `out[:,1:]→(B,32,16,1024)`，对 dinov2-large @224×448（patch14→512 patch，last_hidden_state (B,1+512,1024)）成立。
- `PartHead` 默认 `in_dim=HIDDEN` 在 exp324b import 时定义即 1024；`main()` 亦显式 `in_dim=HIDDEN`。
- `b.CACHE_TRAIN` 在 `b.main()` 前覆写为 `experiments/exp325/_cache`，main() 引用模块全局 `CACHE_TRAIN`（非冻结局部）→ 覆写生效，独立缓存防维度不匹配复用 base 缓存。

### exp325 单变量隔离
仅 backbone（base→large）→ HIDDEN 768→1024 → 头输入维。其余超参（epochs 60、P16K4=BS64、Adam lr3.5e-4、wd5e-4、cosine、id/tri/part=1/1/0.5、soft margin、seed 1234、eval_period 10、ckpt 20）全继承 exp324b 默认未改。干净单变量。

### exp324f（eval-only）
- 文件名对齐：Swin（make_dataloader, PoseImageDataset, shuffle=False, val=query+gallery, OccludedDuke 内部 sorted）vs DINO（sorted listdir）→ 同 key 空间；`align_dino_to_swin` 按文件名 join + pid 全等断言 + camid 偏移恒定断言（修正后，见下），无法静默错位。
- 归一化：z-score/min-max 全矩阵单调仿射 → 不改 per-query 排序；w=0 精确复现纯 Swin（实测 75.16 = 单独）。
- head 载入：`num_classes,embed_dim = classifier.weight.shape` 正确（Linear(embed,classes) weight=(classes,embed)）；strict load 通过。
- heavy mask：DINO 侧 find_pose vis≤8，与 exp324b 口径一致。

## Findings（仅 Low，全部非阻断）
| # | Severity | 位置 | 说明 | 处置 |
|---|----------|------|------|------|
| 1 | Low | exp324f_fuse mmnorm | min-max 对离群敏感，仅作稳健性臂 | z-score 主报，保留 |
| 2 | Low | exp325 header | dinov2-large config image_size=518，输入 224×448 走插值 pos-emb（base 已如此） | 标准 ViT 行为，无需改 |
| 3 | Low | exp324f_fuse | 载入完整 BN buffer 但 encode_parts 不过 BNNeck | 无害 |

## 复审修正（运行时发现，已修，不改变结论）
- **exp324f camid 断言**：首跑触发 `query camid mismatch`。根因：SOLIDER loader camid 0-indexed（c1→0），exp324 `parse_pid_cam` 1-indexed（c1→1），**同相机不同约定**（文件名 cK 相同，pid 已全等证明行对应正确）。修正：断言改为"camid 偏移恒定"（捕获真实错位，容忍约定偏移），eval 全程用 Swin（0-indexed）camid。复跑通过，offset=1，w=0=75.16 sanity 通过。该修正只在对齐校验层，**不影响任何 eval 数值**。

## 结论
exp324f / exp325 代码对其声明目的均正确：exp324f 的文件名 join + pid/camid 断言杜绝静默错位、w=0 可证复现纯 Swin、heavy mask 与 exp324b 一致；exp325 monkeypatch 在 exp324b main() 读取前正确改绑 HIDDEN/load_model/CACHE_TRAIN（按实际模块全局引用点验证），其余超参全保持 exp324b 默认（干净单变量），large 的 patch14/512-token/1024-dim 几何已对 HF config 实证。仅 3 个 Low 非阻断项。

**审查通过。** exp324f 为 eval-only（无训练 hook 阻断，Codex 可选）；exp325 为训练实验，开训前仍需 Codex `--search exec` 审查（hook 强制 codex_review.md approve）。
