# CODEX 调查 brief — 为什么 exp335 的 LGPA-D 复现不涨点

## 结论先行（用户判断）
**用户确信：这是复现代码的 BUG，不是 backbone(ViT vs Swin) 问题。** 请带着"一定有 bug"的怀疑去深挖，不要轻易归因于 backbone 差异。

## 现象
- 复现 LGPA-D（CLIP 部位文本原型 cross-attend backbone token，pose 热图当 attention bias，detached 部位分支）到 ViT-base baseline。
- 原版（Swin）LGPA-D 有效：exp245g=70.2、纯 LGPA-D Swin-Tiny≈63.6（baseline 56.6，+7 真涨）。
- **我的复现负**：测试描述子 equal_concat `[global_norm‖p1..p5_norm]`，**equalcat 全程 < global-only ~4.5 mAP**（部位拖累，不互补）。纯部位 maxsim 仅 35.5（global 53.5）。

## 🔑 最强烟枪：lgpa_assign 损失
- **原版 exp245g train log: `lgpa_assign: 7.218`（大，主动训练部位定位）。**
- **我的复现: `assign≈0.000`（全程）。**
- assign = KL(pose-GT 部位分配 ‖ 预测的 cross-attention)。我的≈0 = 预测注意力已经匹配 pose-GT = **pose-bias 主导/饱和了注意力 → 部位退化成被动 pose-pooled 冻结特征 → 弱**。
- `_compute_pose_bias` 和 `_compute_gt_assignment`（clip_part_head.py）都从同一热图同一 PART_KPS 派生 → 若 pose-bias 尺度 >> QK 分数，注意力 trivially = GT → KL=0。**为什么原版 Swin 不这样（assign=7.2）？是我的热图尺度/feat_map 构造/特征幅度的 bug 吗？**

## 关键差异（原版 vs 我的）
| | 原版 exp245g | 我的 exp335 |
|---|---|---|
| backbone | Swin-Small | ViT-base |
| 输入 | 384×128 | 256×128 |
| 系统 | PSG+LGPA-D+OA-SD+GCN | 纯 LGPA-D |
| feat_map 来源 | Swin 某 stage | ViT 末层 token reshape (B,768,16,8) |
| config | pose_psg_lgpa_detach.yml | exp335_vit_lgpa.yml |
| lgpa_assign | **7.218** | **≈0** |

## 要看的文件
- `scripts/exp335_train_vit_lgpa.py` — 我的自包含 trainer（主嫌疑）
- `scripts/exp335_maxsim_eval.py` — 我的 post-hoc eval
- `model/modules/clip_part_head.py` — LGPA 模块（CLIPPartHead，原版我都用它）
- `model/pose_backbone_model.py` — 原版 LGPA 集成（对照）
- `processor/processor.py` — 原版 eval（equal_concat 在 `_extract_feat_flip`）
- `datasets/pose_dataset.py` — pose 数据 + 联合增强
- `experiments/exp335_vit_lgpa/{design,monitor,claude_review,codex_review}.md` — 设计 + 进度 + 前两轮审查
- `experiments/exp335_vit_lgpa/logs_for_codex/` — exp245g_original.txt（原版 log + config）、exp335_mine_1p0.txt、exp335_mine_0p5.txt

## 你的任务
深挖你被分配的角度，找出**为什么我的部位特征弱 / 为什么 assign=0 / 复现哪里错了**。给出：根因（具体到文件:行）、证据、修复建议、信心分。**逐行读代码，对照原版，别只看表面。**
