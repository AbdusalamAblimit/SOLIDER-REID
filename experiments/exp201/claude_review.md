# exp201 Claude Review: Global SupCon

## 审查范围

1. `experiments/exp201/design.md` -- 实验设计合理性
2. `config/defaults.py` -- 新默认值 `POSE_STR_SUPCON_GLOBAL = False`
3. `loss/make_loss.py` -- `supcon_global_loss` 的计算与使用
4. `loss/supcon_loss.py` -- SupConLoss 对 global feature 的兼容性
5. 已有配置文件 -- 默认值是否破坏已有实验

## 审查结论

### 审查通过

代码改动极小（3 处：defaults.py 1 行，make_loss.py 约 8 行），逻辑清晰，不影响默认行为。

---

## 逐项检查

### 1. design.md 评估

- 动机明确：当前 SupCon 只在 per-token 上做，global 只有 CE+triplet，尝试增加 global SupCon 合理
- 单变量原则：满足，唯一变量是 `POSE_STR_SUPCON_GLOBAL: True`
- 创新门槛：设计文档自认是消融实验，非独立创新 -- 诚实，可接受
- 预期分析合理（中性可能性高，因为 CE+triplet 对 global 已较充分）

**判定: PASS**

### 2. config/defaults.py (line 166)

```python
_C.MODEL.POSE_STR_SUPCON_GLOBAL = False
```

- 默认 False -- 不影响任何已有实验
- 位置正确，紧跟其他 SUPCON 相关参数
- 无已有 yml 文件设置此项 -- 已验证所有 yml 无 `SUPCON_GLOBAL`

**判定: PASS**

### 3. loss/make_loss.py -- supcon_global_loss 的定义与使用

**定义点 (line 166-170):**
```python
supcon_global_enabled = getattr(cfg.MODEL, 'POSE_STR_SUPCON_GLOBAL', False)
if supcon_global_enabled:
    g_norm = F.normalize(feat[0], p=2, dim=1)
    supcon_global_loss = supcon_fn(g_norm, target)
    loss_details['supcon_g'] = supcon_global_loss.item()
```

- `feat[0]` 是 global pooled feature，shape (B, D)
- L2 normalize 后传入 SupConLoss -- 正确，SupConLoss.forward 接受 (B, D) + (B,)
- `supcon_fn` 在上方 line 164 已创建 -- 正确

**使用点 (line 202-204):**
```python
if getattr(cfg.MODEL, 'POSE_STR_SUPCON', False) and getattr(cfg.MODEL, 'POSE_STR_SUPCON_GLOBAL', False):
    supcon_g_w = float(getattr(cfg.MODEL, 'POSE_STR_SUPCON_WEIGHT', 0.5))
    global_id = global_id + supcon_g_w * supcon_global_loss
```

**关键问题: `supcon_global_loss` 是否总是已定义？**

line 202 的条件要求 `POSE_STR_SUPCON=True` 且 `POSE_STR_SUPCON_GLOBAL=True`。
当两者均为 True 时，line 160 的 `elif` 会进入（除非 `POSE_EVIDENTIAL=True` 的 `if` 先匹配）。
进入 elif 后，line 166-169 定义 `supcon_global_loss` -- 路径正确。

**Low -- 理论边界情况:** 如果 `POSE_EVIDENTIAL=True` 且 `POSE_STR_SUPCON=True` 且 `POSE_STR_SUPCON_GLOBAL=True` 同时启用，line 137 的 `if` 会先进入，跳过 `elif`，`supcon_global_loss` 不会被定义，line 204 会触发 `NameError`。但当前无任何配置同时启用这两者，且逻辑上 Evidential 和 SupCon 是互斥替代方案。实际运行无风险。如果希望防御性编程，可在 line 202 前加 `supcon_global_loss = None` 的 fallback，但不阻塞本次实验。

**判定: PASS (Low issue noted)**

### 4. SupConLoss 对 global feature 的兼容性

SupConLoss 接受 (B, D) normalized features + (B,) labels，返回 scalar loss。
global feature `feat[0]` shape 正好是 (B, D)，L2 normalize 后传入 -- 完全兼容。

**判定: PASS**

### 5. 权重复用

`supcon_g_w` 复用 `POSE_STR_SUPCON_WEIGHT`（默认 0.5）。这意味着 global SupCon 和 part SupCon 共享同一权重。对于消融实验这是合理的简化。

**判定: PASS**

### 6. 内存影响

增加一次 SupConLoss 计算：(B, D) matmul 产生 (B, B) similarity matrix，B=64 时 64x64 float32 = 16KB。可忽略。

**判定: PASS**

### 7. 日志充分性

`loss_details['supcon_g']` 记录了 global SupCon loss 值 -- 可以监控其数值是否合理、是否收敛。

**判定: PASS**

### 8. 默认行为安全性

- `POSE_STR_SUPCON_GLOBAL = False` 时：line 166-170 跳过，line 202-204 跳过 -- 零影响
- 所有已有 yml 无此项 -- 不破坏任何已有实验的可复现性

**判定: PASS**

## 问题汇总

| 级别 | 问题 | 状态 |
|------|------|------|
| Low | Evidential + SupCon + SupCon_Global 同时启用时理论 NameError | 无实际风险，可后续防御性修复 |

## 最终结论

代码改动正确、最小化、默认安全。唯一标记的 Low 级问题为理论边界情况，不影响本次实验运行。

**审查通过**
