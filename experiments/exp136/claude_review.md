Now I have all the information needed. Here's the comprehensive review report:

---

# exp136 严格代码审查报告

## 1. 审查结论：**允许启动**

---

## 2. Critical

**无 Critical 问题。**

---

## 3. High

**无 High 问题。**

---

## 4. Medium

### M1: 正样本极度稀疏可能导致训练不稳定

`softmax_triplet` sampler 配合 `NUM_INSTANCE=4` 意味着每个 anchor 只有 **3 个正样本对**。`top_ratio=0.25` 时：

```
keep = max(1, ceil(3 * 0.25)) = 1
```

即每个 anchor 只保留 **1 个正样本对**。这比负样本侧（约 60 对保留 15 对）稀疏得多。如果这个唯一正对恰好是 noise pair，会导致单次梯度方向极端。

**风险等级**：可接受——这是 `delta_top` 的 intended behavior，且 `pair_weight` 选择的是 teacher-change 最大的正对（信噪比理论上最高的那个）。但需要在训练日志中关注 `lpcs_psr` 是否出现异常波动。

### M2: 旧（buggy）config 文件仍然存在

目录中同时存在 4 个 LPCS config：
- `pose_psg_gcn_lpcs.yml` → exp133（buggy，OUTPUT 指向 `exp133_lpcs`）
- `pose_psg_gcn_lpcs_delta_top.yml` → exp134（buggy，OUTPUT 指向 `exp134_lpcs_delta_top`）
- `pose_psg_gcn_lpcs_fix.yml` → exp135（fixed）
- `pose_psg_gcn_lpcs_delta_top_fix.yml` → exp136（fixed）

远程启动时如果手误用了不带 `_fix` 后缀的旧 config，会把日志写入已失效实验的目录。不会造成功能错误（代码已修好），但会造成日志混淆。

**建议**：远程启动命令中严格使用全路径 `configs/occluded_duke/pose_psg_gcn_lpcs_delta_top_fix.yml`。

---

## 5. Low

### L1: `pair_focus` 指标在 `selected_pair_count = 0` 时返回 `1.0`

`processor.py` L430-434：

```python
pair_focus = 1.0
if selected_pair_count > 0.0 and total_pair_weight_sum > 0.0:
    pair_focus = ...
```

当某个 batch 所有 anchor 都被 `continue` 跳过时（极端罕见），`lpcs_pf` 静默报告 `1.0` 而不是 `NaN` 或明确的 sentinel 值。不影响训练，但日志中单次 `lpcs_pf=1.0` 可能被误读为"没有聚焦效果"。

### L2: design.md 中未标注 `GLOBAL_LOSS_SCALE` 的一致性

exp135 和 exp136 均未在 config 中显式设置 `GLOBAL_LOSS_SCALE`，均 fallback 到默认值 `1.0`。这对单变量对照没有问题，但 design.md 中未提及这一点。建议在 design.md 的"其余全部保持一致"部分加一行确认 `GLOBAL_LOSS_SCALE` 也一致。

---

## 6. 逐项检查表

| 检查项 | 结论 | 详情 |
|--------|------|------|
| **共享接线 bug 是否已真正修复** | ✅ 已修复 | `processor.py:602` 条件现为 `kp_triplet_enabled or csgt_enabled or csrd_enabled or ltcs_enabled or lpcs_enabled or paml_enabled or kdl_enabled or lku_enabled or pke_enabled`，`lpcs_enabled` 已包含在内。当 `POSE_LPCS=True` 时，`kp_aux_data` 会被正确构建，后续 `lpcs_teacher_feats` 会被注入（L639-646），`_compute_lpcs_loss` 会被调用（L682-705）。 |
| **单变量性** | ✅ 通过 | `diff` 确认 exp136 config 相对 exp135 config 仅新增 `POSE_LPCS_PAIR_MODE: 'delta_top'` 和 `POSE_LPCS_PAIR_TOP_RATIO: 0.25`，以及 `OUTPUT_DIR` 不同。所有其他参数（scorer/descriptor/teacher bank/cvk_residual/优化器/训练设置）完全一致。 |
| **默认行为是否被破坏** | ✅ 未破坏 | `defaults.py:250-251` 默认 `POSE_LPCS_PAIR_MODE='all'`、`POSE_LPCS_PAIR_TOP_RATIO=1.0`。当 `pair_mode='all'` 时（L400-402），`pos_sel/neg_sel` 均为全 True，等价于修复前的无筛选行为。exp135 和所有不设此参数的旧实验不受影响。 |
| **delta_top 是否真的能形成 sparse routing** | ✅ 能 | `_select_top`（L344-351）使用 `torch.topk` 选取 `pair_weight` 前 25% 的 pair。`pair_weight`（L380-381）由 `\|teacher_dist - base_dist\|` 归一化后得到，代表 teacher bank 对该 pair 的改变幅度。Top-25% 选择后，仅高 teacher-change pair 参与 ranking loss。梯度仅流经 `lpcs_head(desc)` 产出的 `delta`，`pos_sel/neg_sel` 是 detach bool mask，不阻断梯度链。 |
| **日志是否足以验证 lpcs_psr / lpcs_pf** | ✅ 足够 | `lpcs_psr`（pair_selected_ratio，L703）在 `delta_top` 模式下预期约 `0.25`（上下浮动因正样本很少时 `ceil` 效应）。`lpcs_pf`（pair_focus，L704）预期 `>1.0`，代表被选中 pair 的平均 weight 高于全体。两者在 `epoch > 20` 后每个 log interval 都会输出。如果 `lpcs_psr ≈ 1.0`，说明 sparse routing 未生效（需排查）。 |
| **远程启动风险** | ✅ 可控 | 代码已 push 到 `origin/exp/pose_heatmap`；远程 pull 后即可获得修复后的 processor.py 和新 config。唯一风险是手误使用旧 config（见 M2），通过启动命令中使用完整 `_fix` 路径可避免。`import math` 已在 L1（exp134 审查时修复），不会出现 delayed crash。 |
| **teacher bank 更新逻辑** | ✅ 正确 | `lpcs_teacher_bank.update()`（L1073-1078）使用 `kp_data`（模型 forward 直接输出），不依赖 `kp_aux_data`。bank 从 epoch 1 开始更新，warmup 20 epoch 后 bank 已有充分数据。fix 前后 bank 更新行为不变；fix 仅影响 loss 计算路径。 |
| **`import math` 缺失** | ✅ 已修复 | `processor.py:1` 为 `import math`。`math.ceil` 在 `_select_top`（L347）中使用，仅在 `epoch > 20` 且 `pair_mode='delta_top'` 时触发。 |
| **参数校验** | ✅ 充分 | L194-195: `delta_top` 模式下 `top_ratio` 必须在 `(0, 1]`，否则 `raise ValueError`。L247-248: `top_ratio >= 1.0` 时输出 warning。 |
| **优化器覆盖** | ✅ 正确 | `lpcs_head` 是 `PoseBackboneModel` 的 `nn.Module` 子模块（`model/pose_backbone_model.py:510`），自动进入 `model.named_parameters()` → `make_optimizer` 正确纳入。`delta_top` 改动不新增任何参数。 |

---

## 7. 最终结论

**允许启动。**

exp136 相对 exp135 的变更干净且隔离：仅新增 `POSE_LPCS_PAIR_MODE='delta_top'` + `POSE_LPCS_PAIR_TOP_RATIO=0.25`。共享接线 bug 已在 `processor.py:602` 修复到位。`delta_top` 的 sparse routing 逻辑（`_select_top` + `pair_weight`）正确，梯度流安全，日志指标（`lpcs_psr`/`lpcs_pf`）足以验证机制是否真实生效。远程启动时注意使用带 `_fix` 后缀的 config 文件即可。
