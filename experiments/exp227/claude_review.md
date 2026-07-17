# exp227 Claude Review — GSPB(0.005) + PADPQ K=4 on Small

## 审查范围

配置变更实验，无新代码。在 Swin-Small 上组合 GSPB(scale=0.005) 和 PADPQ(K=4)。

---

## a. design.md 审查

**合理性**: 可以接受，但这是一个高风险探索实验。
- 目标明确：验证 GSPB 在 Small 上是否存在一个安全的 scale。
- 对照组清晰 (exp206r)。
- 早停条件合理 (ep10 < 30% 终止)。

**单变量原则**: 违反。同时引入两个变量 vs exp206r:
  1. GSPB scale=0.005
  2. PADPQ K=4 (deformable sampling)
  
  如果实验成功或失败，无法区分是哪个组件的贡献/问题。不过考虑到这是 Small 上的探索性实验（不是消融实验），可以接受。

**创新门槛审查**: 这是配置组合实验，不声称创新。作为 supporting evidence 可以接受。

---

## b. 代码审查 (无新代码)

所有代码路径已在 exp220 (GSPB)、exp223 (PADPQ)、exp225 (GSPB+PADPQ Tiny) 中审查通过。

### GSPB 代码路径验证 (`pose_backbone_model.py` L447-454)
```python
_gs = getattr(self, '_part_grad_scale', 0.0)  # 0.005
if _gs > 0:
    feat_map_detached = featmaps[-1].detach() + _gs * (featmaps[-1] - featmaps[-1].detach())
```
- scale=0.005 是合法值，无边界问题。
- 效果: Part branch 反向传播时，backbone 收到 0.5% 的 Part loss gradient。

### PADPQ 代码路径验证 (`skeleton_gcn.py` L380-396, L480-512)
- K=4 是默认值，已在多个实验中验证。
- `deform_offset_head`: Linear(770, 128) -> ReLU -> Linear(128, 8), zero-init。
- `deform_attn_head`: Linear(770, 128) -> ReLU -> Linear(128, 4)。
- 与 GSPB 的交互: PADPQ 在 gradient-scaled feature map 上采样。当 scale=0.005 时，offset/attn heads 也通过 0.5% 的 backbone gradient 学习。在 Tiny 上 (scale=0.05) 已验证安全。

### Small 特殊考量
- Swin-Small Stage 3 有 **18 blocks** (vs Tiny 6 blocks)。PSG 在每个 block 后注入，产生 18 个 PSG gate。
- GSPB 的 gradient 通过 18 个 PSG gate 累积回传，梯度规模约为 Tiny 的 3x。
- 历史数据:
  - scale=0.05 on Small: **2.3%** (catastrophic, exp222)
  - scale=0.01 on Small: **15.1%** (still catastrophic, exp222c)
  - scale=0.005 on Small: 已尝试 (exp222d) 但在首个 eval 前因"早期训练明显异常"而止损
- **关键风险**: exp222d (scale=0.005, 无 PADPQ) 已经显示异常迹象。exp227 在此基础上还加了 PADPQ，deformable offset 的额外参数也接收 scaled gradient，可能进一步恶化。

---

## c. 配置审查

实验通过命令行 override 配置。需要设置:
- `MODEL.POSE_PART_GRAD_SCALE 0.005`
- `MODEL.POSE_DEFORMABLE_SAMPLE True`
- `MODEL.POSE_DEFORMABLE_K 4`
- `MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224`
- `MODEL.PRETRAIN_PATH pretrained/swin_small.pth`
- 其余继承 GCN+PAA+OA-SD 的 Small baseline (exp206r)

无配置冲突。`POSE_DEFORMABLE_K=4` 是默认值。

---

## d. defaults.py 审查

- `POSE_PART_GRAD_SCALE = 0.0` (默认 detach): 安全，不影响其他实验。
- `POSE_DEFORMABLE_SAMPLE = False`: 安全，不影响其他实验。
- `POSE_DEFORMABLE_K = 4`: 安全，仅在 deformable_sample=True 时生效。

无安全隐患。

---

## e. Processor 审查

GSPB 和 PADPQ 均为 model-internal，processor 不需要任何修改。模型输出结构 (cls_scores list, feats list) 不受影响。

---

## f. 与前序实验对照

| 实验 | backbone | GSPB scale | PADPQ | ep10 结果 |
|------|----------|-----------|-------|----------|
| exp206r | Small | 0 (detach) | No | 50.4/63.9 → 最终 70.6/82.6 |
| exp222 | Small | 0.05 | No | 2.3/3.9 (catastrophic) |
| exp222c | Small | 0.01 | No | 15.1/23.8 (catastrophic) |
| exp222d | Small | 0.005 | No | 未完成 eval (异常终止) |
| exp225 | Tiny | 0.05 | K=4 | 正常 → 最终 64.2/74.9 |
| **exp227** | **Small** | **0.005** | **K=4** | **?** |

消融隔离性: 不完美（双变量），但作为探索实验可以接受。

---

## 审查结论

### 风险评估

| 级别 | 问题 | 状态 |
|------|------|------|
| High | GSPB scale=0.005 在 Small 上 (exp222d) 已显示异常迹象，加 PADPQ 可能更差 | 已知风险，设有早停条件 |
| Medium | 双变量违反单变量原则 | 探索实验可接受 |
| Low | PADPQ deformable heads 接收 scaled gradient 可能增加不稳定性 | 在 Tiny 上已验证安全 (exp225) |

### 建议

1. **严格执行早停**: ep10 < 30% 立即终止。
2. 如果成功 (ep10 > 40%)，后续需要分离消融: 单独 GSPB(0.005) on Small (无 PADPQ) 和单独 PADPQ on Small (无 GSPB)。

---

## 审查通过

无 code bug 或 config 错误。风险已知且有早停机制。批准运行。
