# exp232 Claude Review — BT-PKD Cosine Decay on Small

## 审查范围

a. `experiments/exp232/design.md` — 合理性、单变量原则、假设清晰度
b. 代码变更 — 配置实验，无新代码。审查已有实现：
   - `config/defaults.py`: `POSE_BT_PKD_DECAY_EPOCH` 默认值
   - `processor/processor.py`: cosine decay 逻辑 (lines 786-794)
   - `model/pose_backbone_model.py`: bt_pkd 非 detached sampling (lines 546-561)
c. 配置 — Small backbone overrides
d. `config/defaults.py` — 默认值安全性
e. Processor — loss 计算、梯度流
f. 与 exp230/exp231 对照 — 消融隔离

## a. Design.md 审查

设计清晰。动机合理：exp231 在 Tiny 上测试 cosine decay，exp232 在 Small 上验证同一创新。
核心假设三条均有实验依据（exp229 的双阶段模式、exp230 的 OOM 问题）。

**单变量**: vs exp230 仅添加 `POSE_BT_PKD_DECAY_EPOCH 60`。满足单变量原则。

**一个文档误导**: design.md 声称 "ep60 后 weight=0, non-detached graph 不再需要 retain → 减少内存"。
实际上：`bt_pkd=True` 时，model forward 仍然对 `featmaps[-1]` 做 non-detached `grid_sample`，
computation graph 仍然被构建。`0.0 * bt_pkd_loss` 不会让 PyTorch 跳过 backward graph traversal。
内存节省微小（17 点 grid_sample 的中间激活本身极小），不应作为核心论点。
**严重度: Low** — 不影响实验正确性，但文档描述不准确。

## b. 代码审查

### config/defaults.py (line 221)
```python
_C.MODEL.POSE_BT_PKD_DECAY_EPOCH = 0     # 0=no decay
```
默认值 0 → 不影响已有实验。安全。

### processor/processor.py (lines 786-794)
```python
bt_pkd_weight = float(getattr(cfg.MODEL, 'POSE_BT_PKD_WEIGHT', 0.01))
bt_pkd_decay_ep = int(getattr(cfg.MODEL, 'POSE_BT_PKD_DECAY_EPOCH', 0))
if bt_pkd_decay_ep > 0 and epoch > 0:
    import math
    if epoch >= bt_pkd_decay_ep:
        bt_pkd_weight = 0.0
    else:
        bt_pkd_weight *= 0.5 * (1 + math.cos(math.pi * epoch / bt_pkd_decay_ep))
```

逐行验证：
- `epoch` 从 1 开始 (`for epoch in range(1, epochs + 1)`)，`epoch > 0` 总为 True → 无害冗余
- `epoch >= bt_pkd_decay_ep` (60 >= 60) → weight=0.0。ep60 及之后完全关闭。正确
- ep1: `0.01 * 0.5 * (1 + cos(pi/60))` ≈ `0.01 * 0.999` ≈ 0.00999。几乎全权重。正确
- ep30: `0.01 * 0.5 * (1 + cos(pi/2))` = `0.01 * 0.5 * 1.0` = 0.005。半权重。正确
- ep59: `0.01 * 0.5 * (1 + cos(59pi/60))` ≈ `0.01 * 0.0007` ≈ 0.000007。接近零。正确
- `math.cos` 是 CPU 标量运算，AMP 安全
- `import math` 在循环内：微小性能开销但无 bug（Python 缓存 import）

**无 bug。**

### model/pose_backbone_model.py (lines 546-561)
`bt_pkd=True` 时从 `featmaps[-1]` 做 non-detached grid_sample。
此逻辑与 exp229/230/231 完全相同，已在 exp229 审查中通过。

**需确认**: teacher forward 是否也提供 `kp_feats`/`kp_weights` — 是的，OA-SD teacher forward
在 processor.py 中完成，通过 `teacher_kp_data` 传递。已验证。

## c. 配置审查

exp232 使用 `configs/occluded_duke/swin_small.yml` 基础配置 + 命令行 overrides:
- `MODEL.POSE_BT_PKD True`
- `MODEL.POSE_BT_PKD_WEIGHT 0.01`
- `MODEL.POSE_BT_PKD_DECAY_EPOCH 60`
- `TEST.IMS_PER_BATCH 128` (防 eval OOM, 从 exp230 经验)
- 无 `POSE_PARALLEL_AUG` (OOM with BT-PKD on Small)

Small 配置使用 `swin_small_patch4_window7_224`, SGD, MAX_EPOCHS=120, WITH_CP=False。
`CHECKPOINT_PERIOD=120` — 仅保存 final checkpoint。
**注意**: 建议改为 `CHECKPOINT_PERIOD=20`，按 CLAUDE.md 规范每 20ep 保存，以防再次 OOM crash 丢失中间结果（exp230 教训）。
**严重度: Medium** — 实验可运行但风险高（若 ep100-120 再次 OOM，丢失所有 checkpoint）。

## d. 默认值安全

`POSE_BT_PKD_DECAY_EPOCH = 0` 表示不 decay。所有不设此参数的实验行为不变。安全。

## e. Processor — Loss 计算与梯度流

- 当 `bt_pkd_weight > 0`: cosine distillation loss 通过 non-detached `bt_kp_feats` 反传到 backbone。梯度温和（cosine distillation + 低权重 0.01）。已在 exp229/230 验证安全
- 当 `bt_pkd_weight = 0.0` (ep60+): `loss + 0.0 * bt_pkd_loss` → 该项梯度为零。backbone 不受 BT-PKD 影响。但 computation graph 仍构建（前述 Low 问题）
- OA-SD distillation 不受 BT-PKD decay 影响，始终以 `POSE_OA_SD_WEIGHT=1.0` 运行。正确
- 其余 loss 路径（CE, triplet, GCN）不受影响

## f. 与 exp230/exp231 对照

| 对比项 | exp230 | exp231 | exp232 |
|--------|--------|--------|--------|
| Backbone | Small | Tiny | Small |
| BT-PKD weight | 0.01 (constant) | 0.01 (cosine→0@ep60) | 0.01 (cosine→0@ep60) |
| PARALLEL_AUG | No | Yes | No |
| TEST.IMS_PER_BATCH | 128 | 256 | 128 |

- **exp232 vs exp230**: 仅多了 decay_epoch=60。单变量。正确
- **exp232 vs exp231**: backbone 不同 (Small vs Tiny)。是否有其他差异？
  - exp231 有 PARALLEL_AUG (Tiny 内存够)，exp232 没有。这是因为 Small + BT-PKD + PAUG OOM
  - 这意味着 exp232 和 exp231 不能直接比较 — 需要各自对比各自的 no-decay baseline (exp230 for Small, exp229 for Tiny)

**消融隔离**: exp232 vs exp230 是正确的消融（constant vs decay on Small）。满足要求。

## Bug/Risk 表

| # | 类型 | 严重度 | 描述 | 影响 |
|---|------|--------|------|------|
| 1 | 文档 | Low | design.md 声称 decay 减少内存，但 model forward 仍构建 non-detached graph | 不影响正确性 |
| 2 | 配置 | Medium | CHECKPOINT_PERIOD=120 仅保存 final，若 ep100-120 OOM 则丢失所有中间 checkpoint | 建议改为 20 |

## 综合判断

- 实验设计合理，单变量明确
- 代码无 bug，cosine decay 逻辑正确
- 默认值安全，不影响已有实验
- 主要风险：CHECKPOINT_PERIOD=120 可能导致 OOM 后丢失所有 checkpoint（exp230 前车之鉴）

**建议**: 在启动命令中添加 `SOLVER.CHECKPOINT_PERIOD 20`。这不影响实验变量（只是保存频率），但可以防止 crash 后完全丢失结果。

如果操作者已知悉 checkpoint 风险并接受（或在启动命令中手动添加了 CHECKPOINT_PERIOD 20），则实验可以启动。

## 审查通过

代码和配置无阻塞问题。上述 Medium 风险（CHECKPOINT_PERIOD）建议修复但不阻塞启动。
