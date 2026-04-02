# exp220 GSPB 审查报告

**审查时间**: 2026-04-01
**审查范围**: design.md, model/pose_backbone_model.py (GSPB相关), config/defaults.py

---

## a. design.md 审查

**合理性**: 通过。动机清晰 — detach=0 vs detach=1 的二元选择是真实存在的问题，gradient scaling 是一个合理的中间方案。类比 multi-task learning 的 task-specific learning rate 是恰当的。

**单变量原则**: 通过。相对 exp191 (OA-SD, scale=0.0) 只改一个变量：`POSE_PART_GRAD_SCALE` 从 0.0 改为 0.05。

**创新门槛评估**: 这是一个"改一行代码"的实验。按照 CLAUDE.md 的标准，这不属于大创新，但 design.md 中对问题的定义（detach 的二元困境）有一定的问题层面新意，且可以设计清晰的消融（scale=0, 0.01, 0.05, 0.1, 1.0 的对比曲线）。作为一个快速验证实验是合理的。

---

## b. 代码审查 — model/pose_backbone_model.py

### 初始化 (line 115-118)

```python
self._part_grad_scale = float(getattr(cfg.MODEL, 'POSE_PART_GRAD_SCALE', 0.0))
if self._part_grad_scale > 0:
    print(f'[GSPB] Part branch gradient scale: {self._part_grad_scale}')
```

**通过**。正确从 config 读取，类型转换为 float，默认值 0.0 与 defaults.py 一致。使用 `getattr` 带默认值确保向后兼容。

### Forward (line 443-450)

```python
_gs = getattr(self, '_part_grad_scale', 0.0)
if _gs > 0:
    feat_map_detached = featmaps[-1].detach() + _gs * (featmaps[-1] - featmaps[-1].detach())
else:
    feat_map_detached = featmaps[-1].detach()
```

**数学正确性验证**:

设 `x = featmaps[-1]`，`d = x.detach()`。

- Forward 值: `d + _gs * (x - d)` = `d + _gs * x - _gs * d` = `(1 - _gs) * d + _gs * x`
  - 因为 `d` 和 `x` 的**值**完全相同（detach 只断开梯度），所以 forward 值 = `(1 - _gs) * x_val + _gs * x_val` = `x_val`。**正确**，forward 值不变。

- Backward 梯度: `d` 没有梯度，`x` 有梯度。
  - `d/dx [d + _gs * (x - d)]` = `0 + _gs * (1 - 0)` = `_gs`
  - 所以 backward 梯度 = `_gs * upstream_grad`。**正确**，梯度被缩放到 `_gs` 倍。

- `_gs = 0.0` → 纯 detach（等价于现状）。**正确**。
- `_gs = 1.0` → `d + 1*(x-d) = x`，完全 non-detach。**正确**。
- `_gs = 0.05` → 5% 梯度。**正确**。

### 影响范围

- **GCN 分支 (line 443+, `use_skeleton_gcn`)**: 受 GSPB 影响。`feat_map_detached` 变量被 GCN head 和 dual-branch STD-PR 同时使用。
- **STD-PR 独立分支 (line 366-368)**: 不受影响。该路径条件是 `not self.use_skeleton_gcn`，硬编码 `detach()`。这是正确的 — 当 GCN 不启用时，GSPB 不应生效。
- **BA-PKC (line 532-535)**: 不受影响。BA-PKC 使用 `featmaps[-1]`（非 detached），与 GSPB 的 `feat_map_detached` 无关。
- **Global 分支**: 不受影响。Global 分支在 line 443 的 `elif` 之前已经完成了 GAP→BN→classifier。

**Dual-branch 注意**: 当 GCN + STD-PR dual-branch 模式启用时，STD-PR 也使用同一个 `feat_map_detached`（line 456-457），所以 STD-PR 也会收到 5% 梯度。这在 exp220 的配置下是否启用取决于具体 config（如果只启用 GCN 不启用 STD-PR，则无影响）。设计文档未提及这一点，但如果 exp220 只用 GCN，则不构成问题。

### AMP 安全性

训练使用 `amp.autocast(enabled=True)` (line 487) 和 `amp.GradScaler` (line 377)。gradient scaling trick 的算术操作（加法、乘法、减法）在 float16 下没有精度问题。`detach()` 操作与 autocast 兼容。**通过**。

### `getattr` 冗余

Line 446 使用 `getattr(self, '_part_grad_scale', 0.0)` 而不是直接 `self._part_grad_scale`。这是因为 `__init__` 已经设置了该属性，`getattr` 是冗余但安全的。**无问题**，但可以简化为 `self._part_grad_scale`。

**级别**: Low（风格建议，不影响正确性）

---

## c. 配置文件审查 — config/defaults.py

```python
_C.MODEL.POSE_PART_GRAD_SCALE = 0.0  # 0.0 = detach (default), 1.0 = non-detach, 0.05 = scaled
```

**通过**。默认值 0.0 意味着现有所有实验不受影响（等价于纯 detach）。注释清晰。

**注意**: 未找到 exp220 的独立 config yaml 文件。实验应通过命令行参数 `MODEL.POSE_PART_GRAD_SCALE 0.05` 来覆盖默认值。需确认启动命令包含此参数。

---

## d. 对照实验隔离性

| 维度 | exp191 (对照) | exp220 (GSPB) |
|------|-------------|---------------|
| POSE_PART_GRAD_SCALE | 0.0 (detach) | 0.05 |
| 其他所有参数 | 不变 | 不变 |

**通过**。单变量隔离。

---

## e. 新参数是否被优化器正确处理

GSPB 不引入任何新的可学习参数。它只修改已有 tensor 的梯度流。不需要检查优化器配置。**通过**。

---

## f. 日志充分性

初始化时 print `[GSPB] Part branch gradient scale: 0.05`，足以确认生效。但 training loop 中没有 GSPB 特定的日志（不需要，因为效果通过 loss 趋势和最终 mAP 可观察）。**通过**。

---

## 汇总

| # | 级别 | 问题 | 状态 |
|---|------|------|------|
| 1 | Low | Line 446 `getattr` 冗余（已在 `__init__` 中设置属性） | 可忽略 |
| 2 | Info | 需确认启动命令包含 `MODEL.POSE_PART_GRAD_SCALE 0.05` | 操作检查项 |
| 3 | Info | Dual-branch 模式下 STD-PR 也会收到 5% 梯度，需确认 exp220 config 不启用 STD-PR | 操作检查项 |

无 Critical / High / Medium 级别问题。

---

## 结论

**审查通过**。

GSPB 的实现正确：
1. Forward 值不变，backward 梯度被精确缩放到 `_gs` 倍
2. 默认值 0.0 保证向后兼容
3. 只影响 GCN 分支路径（当 `use_skeleton_gcn=True`）
4. AMP 安全
5. 无新参数，不影响优化器

scale=0.05 是一个合理的起点，介于已验证的两个极端（0=有效, 1=灾难）之间。
