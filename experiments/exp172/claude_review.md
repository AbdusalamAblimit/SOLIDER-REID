# Claude Review (Opus 4.6): exp172 PAPE 3x3

## 审查范围

a. design.md 合理性与单变量原则
b. 所有修改代码逐行审查
c. 配置文件正确性
d. defaults.py 新默认值安全性
e. Processor/loss 影响
f. 消融变量隔离

---

## a. design.md

- 动机清晰：exp171 PAPE 1x1 产生正增益 (+0.1/+0.4)，3x3 kernel 可捕捉局部 pose spatial pattern
- 单变量：仅改 POSE_PATCH_EMBED_KS: 1 -> 3，其余完全相同
- 对照组正确：exp171 (PAPE 1x1)
- 创新性注意：这本身是调参（kernel size），但作为 PAPE 消融实验合理。不是主线创新。

## b. 代码审查

### pose_backbone_model.py (lines 96-108) -- PAPE 初始化

```python
pape_ks = int(getattr(cfg.MODEL, 'POSE_PATCH_EMBED_KS', 1))
pape_pad = pape_ks // 2  # same-padding for odd kernels
self.pose_patch_embed = nn.Conv2d(
    17, embed_dim, kernel_size=pape_ks, padding=pape_pad, bias=True)
nn.init.zeros_(self.pose_patch_embed.weight)
nn.init.zeros_(self.pose_patch_embed.bias)
```

逐项检查：

1. **Conv2d(17, 96, 3, padding=1) 空间维度**: 输入 (96, 32)，kernel=3, padding=1, stride=1(默认), dilation=1(默认)。输出 = floor((96 + 2*1 - 3)/1 + 1) = 96, floor((32 + 2*1 - 3)/1 + 1) = 32。**正确，空间维度保持 (96, 32)**。

2. **padding 公式 `pape_ks // 2`**: 对 ks=1 -> pad=0 (正确), ks=3 -> pad=1 (正确), ks=5 -> pad=2 (正确)。对所有奇数 kernel 成立。

3. **零初始化**: weight 和 bias 都 zeros_。对 3x3 kernel 仍然正确 -- 初始时 conv 输出全零，`x = x + pose_tokens` 退化为 `x = x`，保持预训练行为。

4. **参数量**: 17 * 96 * 3 * 3 + 96(bias) = 14,688 + 96 = 14,784。design.md 中声称 14,784，正确。

### _run_backbone_with_psg (lines 247-255) -- PAPE forward

```python
hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw), ...)
pose_tokens = self.pose_patch_embed(hm)       # (B, 96, 96, 32)
pose_tokens = pose_tokens.flatten(2).transpose(1, 2)  # (B, 3072, 96)
x = x + pose_tokens
```

- `scene_heatmaps` shape: (B, 17, H, W) -> interpolated to (B, 17, 96, 32)
- Conv2d(17, 96, 3, pad=1) on (B, 17, 96, 32) -> (B, 96, 96, 32)。空间维度正确。
- flatten(2): (B, 96, 3072) -> transpose: (B, 3072, 96)
- `x` from patch_embed: (B, 3072, 96)。**维度对齐正确。**
- 加法操作安全。

### dtype / AMP 安全性
- Conv2d 是标准操作，AMP autocast 安全。F.interpolate 也是 AMP 兼容的。无风险。

## c. 配置文件对比

exp171 config vs exp172 config 逐行比较：
- **唯一差异**: exp172 增加了 `POSE_PATCH_EMBED_KS: 3`（exp171 使用默认值 1）
- **OUTPUT_DIR**: exp171 -> `exp171_stdpr_pertoken_plboa_pape`, exp172 -> `exp172_pape3x3`。正确区分。
- 所有其他参数完全一致。**单变量隔离确认。**

## d. defaults.py 审查

```python
_C.MODEL.POSE_PATCH_EMBED = False
_C.MODEL.POSE_PATCH_EMBED_KS = 1
```

- POSE_PATCH_EMBED 默认 False：PAPE 整体关闭，不影响任何不使用 PAPE 的实验。
- POSE_PATCH_EMBED_KS 默认 1：与 exp171 行为一致。**后向兼容，安全。**
- 不影响 baseline 或其他实验的可复现性。

## e. Processor / loss 影响

- PAPE 仅在 forward 的 backbone 阶段注入 pose tokens，不改变 loss 计算逻辑。
- loss 路径仍通过 STD-PR per-token classification + PLBOA，与 exp171 完全相同。
- **无 processor 变更，无 loss 变更。**

## f. 消融变量隔离

| 参数 | exp171 | exp172 |
|------|--------|--------|
| POSE_PATCH_EMBED | True | True |
| POSE_PATCH_EMBED_KS | 1 (default) | **3** |
| 其余所有参数 | 相同 | 相同 |

**严格单变量。**

## 优化器检查

`solver/make_optimizer.py` 遍历 `model.named_parameters()`，`self.pose_patch_embed` 作为 `nn.Conv2d` 挂在模型上，其参数会被自动收录。weight 使用 BASE_LR, bias 使用 BASE_LR * BIAS_LR_FACTOR=2。与 exp171 处理方式完全一致。

## 问题清单

无 Critical / High / Medium / Low 问题。

---

## 审查通过

所有审查维度均通过：
1. Conv2d(17, 96, 3, padding=1) 正确保持空间维度 (96, 32) -> (96, 32)
2. 零初始化对 3x3 kernel 正确，确保从预训练行为平滑启动
3. 后向兼容：POSE_PATCH_EMBED_KS=1 默认不影响 exp171 或其他实验
4. 严格单变量：仅 kernel size 1 -> 3
5. 无 processor/loss/optimizer 逻辑变更
6. AMP 安全，无边界问题
