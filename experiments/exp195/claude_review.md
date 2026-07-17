# exp195 审查报告: OA-SD Global-Only + SupCon

## 审查范围

- experiments/exp195/design.md — 设计合理性
- config/defaults.py (line 180) — 新配置安全性
- processor/processor.py (lines 635-665) — OA-SD global-only 分支实现
- loss/make_loss.py (lines 160-195) — SupCon 在 per-token features 上的操作
- loss/supcon_loss.py — SupCon 实现正确性
- EMA teacher 初始化与更新逻辑

---

## a. 设计审查

**假设清晰且可验证**: design.md 明确指出 exp188 失败的原因是 per-token 级别 OA-SD 与 SupCon 的梯度冲突,
并提出通过将 OA-SD 限制到 global feature 来消除冲突。这是可以通过与 exp176 (SupCon only) 和 exp188 (all-token OA-SD + SupCon) 对比来验证的。

**单变量实验**: 对比 exp176 (SupCon T=0.05, no OA-SD) 增加一个变量: OA-SD global-only。
对比 exp188 (OA-SD + SupCon, all-token) 改变一个变量: distillation 范围从 all-token → global-only。
单变量原则满足。

**创新性评估**: 这个实验本身不是独立创新,而是组合验证 — 验证 OA-SD 与 SupCon 能否共存。
它的价值在于如果成功,可以将两个独立有效的组件合并,为更强的系统提供基础。
不属于"小调参",因为它测试的是梯度流分离这一机制层面的假设,且对 OA-SD 的理解有意义。

**判定: 通过**

---

## b. 代码正确性审查

### processor.py (lines 637-645) — global-only 分支

```python
oa_sd_global_only = getattr(cfg.MODEL, 'POSE_OA_SD_GLOBAL_ONLY', False)
if oa_sd_global_only:
    sf = feat[0] if isinstance(feat, list) else feat
    tf = teacher_feat[0] if isinstance(teacher_feat, list) else teacher_feat
    sf_norm = F.normalize(sf, p=2, dim=1)
    tf_norm = F.normalize(tf.detach(), p=2, dim=1)
    oa_sd_loss = (1.0 - (sf_norm * tf_norm).sum(dim=1)).mean()
```

逐行检查:

1. **`feat[0]` 提取正确**: model 输出 `feat` 为 list 时, `feat[0]` 是 global pooled feature (B, D)。确认。
2. **`teacher_feat[0]` 提取正确**: teacher model 同架构, 同样 `teacher_feat[0]` = global feature。确认。
3. **`isinstance` 保护**: 如果 feat 不是 list (单 feature 模式), 直接用 feat 本身。安全。
4. **`.detach()` 在 teacher 端**: `tf_norm = F.normalize(tf.detach(), ...)` — teacher feature 正确 detach,
   不会有梯度回流到 teacher。确认。但注意: teacher feature 已经是在 `torch.no_grad()` 下生成的,
   所以 `.detach()` 是双重保险,不是必需的。无害。
5. **L2 normalize**: 两端都做了 L2 normalize。cosine distance = 1 - cos_sim。正确。
6. **loss 计算**: `(1.0 - (sf_norm * tf_norm).sum(dim=1)).mean()` = mean cosine distance per sample。正确。
   范围 [0, 2],当完全对齐时为 0,完全反向时为 2。

**无 bug 发现。**

### all-token 分支 (lines 646-654) 未被修改

`elif` 分支在 `oa_sd_global_only=False` 时走,与原有代码完全一致。
默认值 `POSE_OA_SD_GLOBAL_ONLY = False`,所以现有实验不受影响。确认。

### fallback 分支 (lines 655-661)

当 feat 或 teacher_feat 不是 list 时的后备,逻辑与 global-only 分支相同。无变更。

**判定: 通过**

---

## c. 配置安全审查

### config/defaults.py (line 180)

```python
_C.MODEL.POSE_OA_SD_GLOBAL_ONLY = False
```

- 默认值 `False`: 不改变任何现有实验的行为。确认。
- 位置在 OA-SD 相关配置块中,紧跟 `POSE_OA_SD_EMA_DECAY` 之后。组织合理。
- processor.py 中使用 `getattr(cfg.MODEL, 'POSE_OA_SD_GLOBAL_ONLY', False)`,
  即使配置未加载也有 fallback。双重安全。

**判定: 通过**

---

## d. 交互检查

### 对 exp193 (OA-SD + 3-view + CE) 的影响

exp193 不设置 `POSE_OA_SD_GLOBAL_ONLY`,默认为 `False`,走 all-token 分支。
代码路径完全不变。**无影响。**

### 对 exp191/192/194 (标准 OA-SD) 的影响

同上,这些实验不设置 `GLOBAL_ONLY` 配置,默认走 all-token 分支。**无影响。**

### 对 baseline 的影响

OA-SD 整体由 `POSE_OA_SD = False` 控制,OA-SD 块根本不执行。**无影响。**

**判定: 通过**

---

## e. 梯度流分析

### global-only 模式下的梯度路径

OA-SD loss 通过 `sf = feat[0]` 获得 student global feature。

`feat[0]` 的计算链:
- backbone output → GAP (global average pooling) → BN → feat[0]
- 这是一个有梯度的操作链,OA-SD loss 的梯度可以回传到 backbone。确认。

OA-SD loss **不**接触 `feat[1:]` (per-token features):
- `feat[1:]` 在 global-only 分支中没有被引用,因此不会收到 OA-SD 的梯度。确认。

### SupCon 在 per-token features 上的操作 (make_loss.py line 160-191)

```python
for f in feat[1:]:
    f_norm = F.normalize(f, p=2, dim=1)
    sc_loss = supcon_fn(f_norm, target)
```

SupCon 只操作 `feat[1:]` (per-token features), **不**操作 `feat[0]` (global)。
这与 global CE 是分开的: `global_id = ce_fn(score[0], target)` 在 line 133。

### 梯度汇总

| 特征 | CE | Triplet | SupCon | OA-SD (global-only) |
|------|-----|---------|--------|---------------------|
| feat[0] (global) | global CE | global triplet | 无 | **cosine distill** |
| feat[1:] (tokens) | 无 (replaced by SupCon) | part triplet | **per-token SupCon** | 无 |

**关键确认**: OA-SD 和 SupCon 作用在完全不同的特征子集上,梯度不冲突。

但需要注意: feat[0] 和 feat[1:] 都来自同一个 backbone,所以梯度最终都会合流到 backbone 参数。
但这不是 "冲突" — 这是正常的多任务学习 (global branch 和 part branch 本来就是不同 loss 合力)。
exp188 的问题是 OA-SD 和 SupCon 在**完全相同的 feature tensor** 上方向相反,
而 exp195 将它们分离到不同的 feature tensors 上。

### 一个微妙点: triplet loss 在 feat[1:] 上

Per-token triplet (make_loss.py line 231-234) 也作用在 `feat[1:]` 上。
所以 `feat[1:]` 实际上同时接收 SupCon + triplet 的梯度。
这在 exp176 (SupCon only) 中已经验证过有效 (64.1/75.5),所以这不是新的问题。

**判定: 通过**

---

## f. 其他检查

### AMP 安全性

OA-SD loss 中的操作 (F.normalize, dot product, mean) 全部是 AMP 安全的。
没有 log/exp/softmax 在小值上的风险。

### 日志充分性

OA-SD loss 通过 `details['oa_sd'] = oa_sd_loss.item()` 记录在训练日志中。
可以监控 OA-SD loss 的收敛趋势。但无法区分 global-only 和 all-token 模式在日志中的区别。
**建议 (Low)**: 可以在日志中加一行标识 `oa_sd_mode: global_only`,但不影响功能。

### 配置文件缺失

实验目录中尚无 .yml 配置文件 — 预计通过命令行 override 传参 (与远程服务器启动方式一致)。
design.md 中列出了所有配置: SupCon T=0.05, OA-SD GLOBAL_ONLY=True, decay=0.999, weight=1.0, PLBOA enabled。

---

## 问题汇总

| 级别 | 问题 | 状态 |
|------|------|------|
| Low | 日志中无法区分 OA-SD 是 global-only 还是 all-token | 不影响实验,可后续改进 |

无 Critical / High / Medium 级别问题。

---

## 结论

代码修改量极小 (config 1 行, processor ~8 行), 逻辑清晰, 默认值安全,
不影响任何现有实验。梯度流分析确认 OA-SD 和 SupCon 在不同特征子集上操作,
解决了 exp188 中的直接梯度冲突。设计假设明确,对照组合理。

**审查通过**
