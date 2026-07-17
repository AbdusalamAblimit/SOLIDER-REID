# Claude Broad Review: exp176 SupCon T=0.05 Ablation (Opus 4.6)

## 审查范围

a. design.md — 合理性、单变量原则
b. supcon_loss.py — temperature 参数使用、数值稳定性
c. config/defaults.py — POSE_STR_SUPCON_TEMP 默认值安全性
d. loss integration — temperature 传递链路
e. 单变量隔离 vs exp174

---

## a. design.md

单参数消融（T=0.07 到 T=0.05）对照 exp174。动机清晰：更低 T = 更尖锐分布 = 更关注 hard pairs。适合作为 SupCon 消融实验。

## b. 代码审查 — supcon_loss.py

### 数值稳定性 (T=0.05)

1. `sim = torch.matmul(features, features.T) / self.temperature` (line 42)
   - L2-normalized features: max cosine similarity = 1.0
   - max sim = 1.0 / 0.05 = 20.0
   - `exp(20) ≈ 4.85e8` — 在 fp32 范围内 (~3.4e38)

2. **稳定性减法** `sim = sim - sim_max.detach()` (line 56-57)
   - 减法后所有值在 `[-40, 0]` 范围
   - 每行最大值为 0
   - `exp(0) = 1.0`, `exp(-40) ≈ 4e-18`: fp32 完全安全

3. `log(... + 1e-8)` (line 62): epsilon 保护防止 log(0)

**结论**: T=0.05 数值安全。即使没有 sim_max 减法，exp(20) 也在 fp32 范围内。有了减法，最大指数始终为 0。

## c. defaults.py

`POSE_STR_SUPCON_TEMP = 0.07` (line 162)。本实验通过 CLI 覆盖为 0.05。默认值不变，不影响其他实验。

## d. Loss integration

Temperature 传递链路: config → make_loss.py:163 `supcon_temp = float(getattr(...))` → SupConLoss(temperature=supcon_temp) → forward line 42 `/ self.temperature`。正确。

## e. 单变量隔离

vs exp174: 唯一差异是 `POSE_STR_SUPCON_TEMP: 0.05` vs `0.07`。所有其他设置（triple injection, PLBOA, PAPE, STD-PR, eval 模式等）完全一致。严格单变量。

## 问题清单

无 Critical / High / Medium / Low 问题。

---

## 审查通过

T=0.05 消融实验代码正确、数值安全、单变量隔离。无需任何代码修改，仅 CLI 覆盖 temperature 参数。
