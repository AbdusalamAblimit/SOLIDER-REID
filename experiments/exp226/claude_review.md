# exp226 Claude Review — 2-Stage Fusion with Zero-Init Projection

## 审查范围
a. design.md — 合理性
b. 代码修改 — 零初始化 kamp_s2_proj
c. 与 exp224 对照

## a. design.md 审查

### 假设合理性: OK
exp224 的审查 (检查点 #Medium) 明确指出 kamp_s2_proj 随机初始化是问题。
零初始化使初始行为 = identity (只用 Stage 3)，是直接修复。

### 单变量原则: OK
vs exp224: 仅改 kamp_s2_proj 初始化。其余完全相同。

## b. 代码审查

### skeleton_gcn.py 修改

新增两行:
```python
nn.init.zeros_(self.kamp_s2_proj.weight)
nn.init.zeros_(self.kamp_s2_proj.bias)
```

- 零初始化 weight 和 bias → `kamp_s2_proj(x) = 0` for all x
- 与 kamp_scale_attn 零初始化配合: softmax([0,0]) = [0.5, 0.5]
- 初始: 0.5 * zeros(stage2) + 0.5 * stage3_feats = 0.5 * stage3_feats
- **注意**: 这意味着初始输出是 Stage 3 的 50%，不是 100%！
- 但由于 GCN 后面还有 BN，50% 缩放会被 BN 归一化掉
- 所以有效初始行为 ≈ 只用 Stage 3 (after BN normalization)

### 风险: Low
零初始化是标准 identity start 做法。GCN BN 会补偿幅度变化。

## c. 对照

| 参数 | exp224 | exp226 |
|------|--------|--------|
| kamp_s2_proj init | Kaiming (random) | **Zeros** |
| 其余 | 相同 | 相同 |

## 审查通过
