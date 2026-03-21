I now have all the information needed. Here is the second-round review report.

---

# exp134 第二轮代码审查报告：Changed-Pair Sparse LPCS

## 1. 审查结论：✅ 允许启动

第一轮发现的 3 个问题（Critical ×1, Medium ×1, Low ×1）均已正确修复。未发现新的 Critical/High/Medium 级别问题。

---

## 2. Critical

无。

**第一轮 C1 修复验证**：`import math` 已出现在 `processor/processor.py:1`，`_select_top` 函数（L344-351）中的 `math.ceil` 调用在 delta_top 路径下不会再崩溃。

---

## 3. High

无。

---

## 4. Medium

无。

**第一轮 M1 修复验证**：训练端 `base_dist` 的权重计算已改为从 `cfg.TEST.CVK_GLOBAL_WEIGHT` / `cfg.TEST.CVK_KP_WEIGHT` 读取（L187-188），与测试端 `metrics.py:284-286` 使用相同的权重来源。具体实现：

| 路径 | 代码位置 | 权重来源 | 计算公式 |
|------|----------|----------|----------|
| 训练（LPCS loss） | processor.py:364-366 | `cfg.TEST.CVK_GLOBAL_WEIGHT/KP_WEIGHT` | `(gw * global + kw * kp) / (gw + kw)` |
| 测试（cvk_residual） | metrics.py:284-286 | `cfg.TEST.CVK_GLOBAL_WEIGHT/KP_WEIGHT` | `(gw * global + kw * kp) / (gw + kw)` |

两侧完全一致。当前两个 config 中均未显式设置 CVK 权重，使用默认值 1.0:1.0（等价于简单平均），train-test 一致。✅

---

## 5. Low

### L1: `POSE_LPCS_PAIR_MODE` 无枚举校验

- L397 仅检查 `== 'delta_top'`，其余值（包括拼写错误如 `'deltatop'`）都静默走 `else`（等价于 `'all'`）
- **影响**：exp134 显式设置为 `'delta_top'`，无拼写风险；若未来新增其他 mode 才需要扩展校验
- **本次判定**：不阻塞，属于防御性编码建议

### L2: `torch.topk` 在并列值时选择不确定

- `_select_top`（L348）使用 `torch.topk`，当多个 pair 的 `pair_change` 值完全相等时，被选中的元素是非确定性的
- **影响**：`pair_change` 是 float 连续值，完全并列的概率极低，实际不构成问题

---

## 6. 逐项检查表

| 检查维度 | 结论 | 详细说明 |
|---------|------|----------|
| **单变量性** | ✅ 通过 | `diff` 确认 exp134 yml 与 exp133 yml 仅 3 处差异：`PAIR_MODE='delta_top'`、`TOP_RATIO=0.25`、`OUTPUT_DIR`。所有其他超参（backbone、PSG、GCN、LPCS scorer、teacher bank、优化器、数据）完全一致 |
| **默认行为是否被破坏** | ✅ 通过 | `defaults.py:250-251` 新增 `PAIR_MODE='all'` + `TOP_RATIO=1.0`，使所有未设置这两项的已有实验走 `else` 全选分支，行为不变。exp133 config 中未设置这两项，使用默认值 |
| **config 接线** | ✅ 通过 | `processor.py:185-186` 正确读取 `PAIR_MODE` 和 `TOP_RATIO`；L194-195 有范围校验；L240-248 logging 完整打印两个新参数并含 degenerate 情况 warning |
| **train loss 接线** | ✅ 通过 | `_select_top`（L344-351）正确实现 top-k 选择：`import math` 已在 L1 引入；`max(1, ...)` 保证至少选 1 个；L408-409 有空选后 guard；L397-402 正确分支 `delta_top` vs fallback |
| **统计量是否足以验证机制** | ✅ 通过 | `lpcs_psr`（L429, L700）= 实际保留 pair 比例（预期 ~0.25）；`lpcs_pf`（L430-434, L701）= 选中 pair 的平均 weight / 全 pair 平均 weight（预期 >1.0 表示确实选中了 teacher-change 更大的 pair）。两个统计量可清晰验证 sparse routing 是否生效 |
| **test 路径** | ✅ 通过 | `cvk_residual`（metrics.py:277-307）仅使用 `pair_residual_head.forward(desc)`，无任何 `pair_mode` / `top_ratio` 参数参与。PairResidualScorer（pair_adaptive_fusion.py:76-102）未被修改 |
| **PairResidualScorer 模型** | ✅ 通过 | model 定义（pose_backbone_model.py:510）和 PairResidualScorer（pair_adaptive_fusion.py:76-102）均未改动，与 exp133 完全一致 |
| **teacher bank** | ✅ 通过 | SupportCompleteBank 初始化（L197-205）和更新（L1070-1075）逻辑未改动，与 exp133 一致。bank 更新独立于 pair routing |
| **pair descriptor** | ✅ 通过 | `build_pair_descriptors`（L372-373）未改动，6 维 descriptor 与 exp133 一致 |
| **梯度流** | ✅ 通过 | `base_dist` 由全 detach 输入计算（feat_g=global_feat.detach, kp_base/kp_teacher=detach）；仅 `delta`（L374, 由 lpcs_head 产出）携带梯度；`pos_sel`/`neg_sel` 由 detach 的 `pair_weight` 派生的 bool mask，不影响梯度链 |
| **warmup 门控** | ✅ 通过 | L636 `epoch > lpcs_warmup`（warmup=20），即 epoch 21 开始进入 LPCS loss。`_select_top` 在此时才被调用，与预期一致 |
| **远程启动风险** | ⚠️ 低风险 | 需 `git push` + 远程 `git pull` 同步最新修复。远程 Python 环境应与本地一致。建议远程启动后首先确认 log 中出现 `[LPCS] ... pair_mode=delta_top, top_ratio=0.25` |
| **design.md 与实现一致性** | ✅ 通过 | design.md 描述的所有改动（pair_mode config、top_ratio config、per-anchor top-k selection、psr/pf logging、测试路径不变）均在代码中对应 |
| **第一轮 Low 修复** | ✅ 通过 | L247-248 新增 `delta_top + top_ratio >= 1.0` 的 logger.warning；L194-195 新增 `0 < ratio <= 1` 的 ValueError 校验 |

---

## 7. 最终结论

**允许启动。**

第一轮发现的所有 3 个问题均已正确修复：
1. `import math` 已补上（L1），`_select_top` 不会再 NameError
2. 训练端 `base_dist` 已与测试端 CVK 权重统一（L187-188, L364-366）
3. `delta_top + ratio >= 1.0` 的 warning 和非法 ratio 的 ValueError 校验均已就位（L194-195, L247-248）

代码改动严格遵守单变量原则，不影响 exp133 或任何已有实验的行为。

**启动建议**：远程同步代码后，确认 log 首行出现 `pair_mode=delta_top, top_ratio=0.25`，并在 epoch 21 后首次 log 中确认 `lpcs_psr` ≈ 0.25、`lpcs_pf` > 1.0。
