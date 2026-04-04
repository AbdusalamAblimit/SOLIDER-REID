# exp239 审查: PPA + GiLt-style loss (Part triplet only, no Part CE)

## 审查范围

a. `experiments/exp239/design.md` — 合理性
b. `loss/make_loss.py` — GiLt elif 分支
c. `config/defaults.py` — 新默认值安全性
d. 单变量隔离: vs exp237 仅加 POSE_PPA_GILT=True

## a. design.md 审查

**通过**。动机清晰: exp237 PPA 使用 5 个 part CE + triplet，CE 可能与 global CE 竞争梯度。
KPR (ECCV 2024) 证明 global CE + part triplet-only (GiLt) 有效。
这是一个有理论依据的消融实验，符合单变量原则。

对照组明确: exp237 (PPA w=0.5, full CE): 63.7/75.0。

## b. 代码审查: `loss/make_loss.py`

**修改点 (L202-205)**:
```python
elif getattr(cfg.MODEL, 'POSE_PPA_GILT', False):
    # GiLt mode: Part branch uses triplet-only, no Part CE
    # This prevents part CE from competing with global CE
    part_id_avg = torch.tensor(0.0, device=score[0].device)
```

检查项:
1. **elif 位置正确**: 在 `POSE_STR_SUPCON` 之后、默认 CE 分支之前。PPA 和 STR 是互斥架构，不存在优先级冲突。
2. **`torch.tensor(0.0, device=...)`**: 无 requires_grad, 不会在计算图中引入多余节点。`w_p * 0.0` 结果为 0，ID_LOSS 完全由 `w_g * global_id` 决定。正确。
3. **`.item()` 调用 (L215)**: `part_id_avg.item()` 对 `torch.tensor(0.0)` 返回 `0.0`，不会报错。日志中 `id_part` 始终为 0.0，可用于监控确认 GiLt 生效。
4. **Part triplet 不受影响**: triplet 计算在 L221-255，独立于 CE 路径，`feat[1:]` 仍然参与 triplet loss。正确。
5. **无 AMP 风险**: `torch.tensor(0.0)` 为 float32 标量，与 mixed precision 无冲突。

**注意 (非阻塞)**:
- PPA 的 `part_classifiers` 和 `part_bns` 在 GiLt 模式下仍然执行 forward（在 `PartAssignmentHead.forward` L150-153），但其输出 `score[1:]` 不参与 CE loss。这些参数变成"死权重"——无 CE 梯度，但 BN running stats 仍更新。
  - 影响: 约 5 * (768 * 702 + 768 * 2) ≈ 2.7M 参数无用，增加少量内存和计算。
  - 风险: 极低。如果 GiLt 有效且需要进一步优化，可后续移除这些参数。
- 这些 dead classifiers 在 test 时不影响结果（test 只用 feat，不用 score）。

## c. `config/defaults.py` 审查

**通过**。L203: `_C.MODEL.POSE_PPA_GILT = False`，默认关闭，不影响已有实验。

## d. 单变量隔离

| 项目 | exp237 | exp239 |
|------|--------|--------|
| PPA | True | True |
| PPA_NUM_PARTS | 5 | 5 |
| PPA_ASSIGN_WEIGHT | 0.5 | 0.5 |
| PPA_GILT | False | **True** |
| OA-SD | True | True |
| PLBOA | True (0.7) | True (0.7) |
| PSG | True | True |

仅 POSE_PPA_GILT 一个变量变化。**通过**。

## 汇总

| 级别 | 问题 | 状态 |
|------|------|------|
| Low | Dead part classifiers (2.7M 无用参数) | 已知，非阻塞 |

无 Critical / High / Medium 问题。

## 监控建议

- 观察 `id_part` 是否始终为 0.0（确认 GiLt 生效）
- 对比 `tri_part` 与 exp237 的趋势差异
- 关注 `ppa_assign` 收敛速度是否因缺少 part CE 的辅助训练信号而变慢

---

**审查通过**。代码修改最小且正确，单变量隔离完好，可启动训练。
