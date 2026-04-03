# exp230 Claude Review — BT-PKD on Small

## 审查范围

配置变更实验，无新代码。在 Swin-Small 上运行 exp229 的 BT-PKD 创新。

---

## a. design.md 审查

**合理性**: 合理。exp229 在 Tiny 上测试 BT-PKD，exp230 在 Small 上测试。这是 cross-backbone 验证，不是组合实验。

**单变量原则**: 满足。相对于 exp206r (Small OA-SD) 只添加了 BT-PKD。

**创新门槛审查**: 与 exp229 共享同一创新 (BT-PKD)。作为 cross-backbone 消融可以接受。

---

## b. 代码审查 (无新代码)

所有代码在 exp229 审查中已通过。BT-PKD 的代码路径在 Small 上的行为：
- `featmaps[-1]` 为 Swin-Small Stage 3 输出 (B, 768, 12, 4)，与 Tiny 相同形状
- `grid_sample` 在同样大小的 feature map 上采样，行为一致
- Cosine distillation loss 计算不依赖 backbone 大小

### Small 特殊考量
- Small Stage 3 有 **18 blocks** (vs Tiny 6 blocks)
- BT-PKD 的 distillation 梯度通过 `featmaps[-1]` → 18 blocks 反向传播
- 梯度累积量约为 Tiny 的 3x
- **关键区别**: BT-PKD 梯度是 cosine distillation (smooth)，不是 CE/SupCon (sharp)
- GSPB scale=0.005 在 Small 上勉强存活 (exp227); BT-PKD weight=0.01 的有效梯度更小
- **风险**: 中等。需要监控 ep10 是否正常

---

## c. 配置审查

命令行 override:
- `MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224`
- `MODEL.PRETRAIN_PATH pretrained/swin_small.pth`
- `SOLVER.BASE_LR 0.0004` (Small 用 0.0004, not 0.0008)
- `MODEL.POSE_OA_SD True`
- `MODEL.POSE_LOWER_BODY_OCC True`
- `MODEL.POSE_BT_PKD True`
- `MODEL.POSE_BT_PKD_WEIGHT 0.01`

无配置冲突。

---

## d. defaults.py 审查

BT-PKD 默认值在 exp229 审查中已验证安全。不影响其他实验。

---

## e. Processor 审查

BT-PKD loss 计算在 exp229 审查中已验证。Small 上无额外 processor 考虑。

---

## f. 与前序实验对照

| 实验 | backbone | BT-PKD | GSPB | 预期 |
|------|----------|--------|------|------|
| exp206r | Small | No | No | 70.6/82.6 (baseline) |
| exp227 | Small | No | 0.005 + PADPQ | 进行中 (~71.3/80.7 at ep100) |
| exp229 | Tiny | Yes (w=0.01) | No | 进行中 |
| **exp230** | **Small** | **Yes (w=0.01)** | **No** | **?** |

---

## 审查结论

| 级别 | 问题 | 状态 |
|------|------|------|
| Medium | Small 18-block 可能放大 BT-PKD 梯度 | 有早停条件 |

低风险配置变更实验。代码已在 exp229 审查中通过。

---

## 审查通过

无 code bug 或 config 错误。风险已知且有早停机制。批准运行。
