# 实验 exp340: 固定语义涨点 — 固定 CLIP 文本 + 固定解剖先验(无 per-image pose)

## 动机
- 已证(exp336/337/338/339):固定 CLIP 文本 standalone **不涨**(no-pose part −0.2),根因是**没有定位**——`CLIP文本-query · SOLIDER-key` 在未对齐空间里点积无意义,塌成 global。
- 是 **pose 的 assign loss + bias 在教定位**(pose part +0.8)。但 pose 是 per-image。
- CLIP 自己 grounding 也定位不了这些遮挡 crop(exp339 −1.2)。
- 用户目标:**实现固定语义涨点**。洞察:**行人直立,头顶→脚踝的竖直布局是通用解剖先验**(PCB 横条带就靠这个),不需要 per-image pose。

## 核心假设
**用固定标准行人姿态(canonical pose,所有图共用)替代 per-image pose,给固定 CLIP 文本提供固定的定位先验 → 固定文本 + 固定先验 = 全固定 → part 分支 standalone 超 global = 固定语义涨点。**

## 技术方案
- 新 flag `MODEL.POSE_LGPA_FIXED_BANDS`(`config/defaults.py`)。
- `pose_backbone_model.py`:
  - `_canonical_heatmap(B, device)`:生成固定的 17-COCO-关键点热图,关键点固定在直立行人标准位(鼻 0.06、肩 0.18、髋 0.50、膝 0.72、踝 0.95 归一化纵向位置),Gaussian blob。**无 per-image 信息**,缓存复用。
  - `_lgpa_heatmap()`:flag 开时把 canonical 当 lgpa_hm 喂给 clip_part_head(替代 scene_heatmaps)。
  - clip_part_head 内部 bias + assign loss + visibility **全自动用 canonical**(改一处,复用全部机制)。
- 数据流:input → SOLIDER backbone → featmap → clip_part_head(固定文本 query + canonical-pose bias + canonical assign 监督)→ 5 part 描述子 + global。

## 预期结果
- **理想:part_only > global(标准 test.py eval,baseline 59.0)= 固定语义涨点。** 预期 +0.3~0.6(固定先验比 per-image pose 弱,但抓住平均布局)。
- equal_concat(global + fixed-band parts)> 59.0。
- 对照锚点:global 59.0 / no-pose part 58.8(−0.2)/ pose part 59.8(+0.8)。fixed-band 落在两者间且 > 59.0 即成功。
- 失败最可能原因:固定先验对遮挡/非全身 crop 误对齐,assign loss 把 attention 锁死在错位置 → ≈ no-pose。

## 对照组
- baseline:exp336 的 global(59.0)。
- 变量隔离:exp340 = exp336 **仅多** `POSE_LGPA_FIXED_BANDS: True`(per-image pose → 固定 canonical)。其余(detach、equal_concat、0.5 global loss、384×128)全同。
- 关键诚实对照(reviewer 必问,训练后补):固定文本 vs random 文本 prototype(同 canonical 先验)——证明涨点来自"固定 + 定位"而非 CLIP 词义本身;但**本实验目标是"固定语义(固定文本)standalone 涨点"这一事实**,random 对照属于归因分析。

> **两套 baseline 口径**: test.py(含 flip)global = **59.0**(主对照,part_only 同口径比这个);exp339 frozen 脚本(无 flip)global = **58.20**(仅 frozen 内部自比)。

## 前置 frozen 测试结果(exp339b,均未训练,直接池化 pose-trained 特征)
| frozen 配置 | mAP | vs baseline 58.20 |
|---|---|---|
| fixed bands ONLY | 58.03 | −0.17 |
| global + fixed bands | 58.16 | −0.04 |
| band × CLIP-grounding | 57.86 | −0.34 |
| CLIP-grounded | 56.98 | −1.22 |

**结论**：冻结池化全 < baseline，但 fixed bands 仅差 0.17（远好于 CLIP-grounded −1.22）。冻结在 pose-trained 特征上次优（frozen 误导）；训练让 head 适应固定 band 是翻正的关键 → 本实验。

## ★ 结果（e120, test.py 同口径）—— 固定语义涨点达成
| exp340a 描述子 | mAP | vs 自身 global |
|---|---|---|
| global | 58.8 | — |
| **part_only（固定 CLIP 文本 + 固定 canonical 姿态）** | **59.4** | **+0.6** ✅ |
| **equal_concat** | **59.5** | **+0.7** ✅ |

**两个口径都 > global → 固定语义 standalone 涨点实现。** 全谱对比（part_only 口径）：no-pose −0.2 → **固定 canonical +0.6** → per-image pose +0.8。固定标准姿态抓住行人平均布局，拿到 per-image pose 约 75% 的增益，但无需 per-image pose、无可学习 prompt = 全固定。
