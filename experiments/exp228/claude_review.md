# exp228 Claude Review — GSPB(0.05) + PADPQ K=8 on Tiny

## 审查范围

配置变更实验，无新代码。在 Swin-Tiny 上将 PADPQ 的 K 从 4 增加到 8，组合 GSPB(scale=0.05)。

---

## a. design.md 审查

**合理性**: 合理。
- 动机清晰: PADPQ K=8 单独比 K=4 好 (+0.2 mAP in exp223b vs exp223)，GSPB+K=4 已验证 (exp225=64.2)。
- 假设明确: K=8 更广 receptive field + GSPB 可能叠加。
- 对照组充分: exp225 (GSPB+K=4) 和 exp223b (K=8 only)。

**单变量原则**: 满足。相对于 exp225 只改了 `POSE_DEFORMABLE_K` 从 4 到 8。

**创新门槛审查**: 纯超参调优，不声称创新。作为消融/对照可以接受。

---

## b. 代码审查 (无新代码)

### PADPQ K=8 代码路径验证 (`skeleton_gcn.py`)

**构造函数** (L380-396):
- `self._deform_k = 8`
- `deform_offset_head`: Linear(770, 128) -> ReLU -> Linear(128, **16**) — 输出 K*2=16 维
- `deform_attn_head`: Linear(770, 128) -> ReLU -> Linear(128, **8**) — 输出 K=8 维
- Zero-init on offset_head 最后一层: 初始行为 = 8 个重复的 identity 采样点
- 参数量变化: offset_head 最后层 128*16+16=2064 (vs K=4: 128*8+8=1032), attn_head 最后层 128*8+8=1032 (vs 128*4+4=516)。总增量约 +1548 参数，可忽略。

**Forward** (L480-512):
- L493: `K = self._deform_k` → K=8
- L494: `offsets = offsets.view(B, 17, K, 2)` → `(B, 17, 8, 2)` — 正确
- L497: `sample_pts = grid_base.unsqueeze(2) + offsets` → `(B, 17, 8, 2)` — 正确
- L502: `pts_flat = sample_pts.view(B, 17 * K, 1, 2)` → `(B, 136, 1, 2)` — 正确
- L503-506: `grid_sample` → `(B, C, 136, 1)` → squeeze → permute → `(B, 136, C)` — 正确
- L507: `sampled_k = sampled_flat.view(B, 17, K, C)` → `(B, 17, 8, C)` — 正确
- L510: `attn_logits = self.deform_attn_head(context)` → `(B, 17, 8)` — 正确
- L511: `F.softmax(attn_logits, dim=-1)` → `(B, 17, 8)` — 正确
- L512: `(sampled_k * attn_w.unsqueeze(-1)).sum(dim=2)` → `(B, 17, C)` — 正确

**所有 shape 在 K=8 时完全正确。无 bug。**

### GSPB 交互 (`pose_backbone_model.py` L447-454)
- scale=0.05 on Tiny: 已在 exp220 (单独) 和 exp225 (with K=4) 中验证安全。
- K=8 不改变 gradient flow 结构，仅在 deformable heads 中增加少量参数。
- 无额外风险。

### 内存估算
- K=8 vs K=4: grid_sample 处理 `(B, 17*8, 1, 2)` vs `(B, 17*4, 1, 2)`。
- 增量约 `B * 17 * 4 * C` = `64 * 17 * 4 * 768` = ~3.4M float = ~13MB。可忽略。

---

## c. 配置审查

实验通过命令行 override:
- `MODEL.POSE_PART_GRAD_SCALE 0.05` (same as exp225)
- `MODEL.POSE_DEFORMABLE_SAMPLE True` (same as exp225)
- `MODEL.POSE_DEFORMABLE_K 8` (changed from default 4)

所有其他配置继承 exp225 的 Tiny GCN+PAA+OA-SD+PLBOA baseline。

无配置冲突。

---

## d. defaults.py 审查

- `POSE_DEFORMABLE_K = 4` (默认值): exp228 override 为 8。不影响其他实验。
- 其余默认值无变化。

---

## e. Processor 审查

PADPQ K 值变化完全是 model-internal。模型输出结构 (cls_scores list, feats list) 不受影响。Processor 无需修改。

---

## f. 与前序实验对照

| 实验 | GSPB scale | PADPQ K | mAP (eq) | mAP (maxsim) |
|------|-----------|---------|----------|-------------|
| exp191 OA-SD baseline | 0 | No | 63.2 | 64.2 |
| exp220 GSPB only | 0.05 | No | 62.9 | 64.6 |
| exp223 PADPQ K=4 only | 0 | 4 | 63.7 | 63.9 |
| exp223b PADPQ K=8 only | 0 | 8 | 63.9 | — |
| exp225 GSPB + PADPQ K=4 | 0.05 | 4 | 64.2 | — |
| **exp228 GSPB + PADPQ K=8** | **0.05** | **8** | **?** | **?** |

消融隔离性: 完美。相对于 exp225 只改了 K=4→8。

---

## 审查结论

| 级别 | 问题 | 状态 |
|------|------|------|
| 无 | 无任何 bug 或风险 | 全部通过 |

这是一个低风险、单变量的配置调优实验。所有代码路径已在前序实验中验证。K=8 的 shape 推导完全正确。

---

## 审查通过

无 code bug、config 错误或风险。批准运行。
