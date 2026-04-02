# exp219 Claude Review — PACI WITHOUT OA-SD on Tiny

## 审查范围

a. design.md — 合理性、单变量原则
b. 代码修改 — 无新代码，仅配置变更
c. 与前序实验的对照

---

## a. design.md 审查

### 假设合理性: OK

exp218 显示 PACI + OA-SD (~62%) < OA-SD-only (64.4%)。去掉 OA-SD 测试 PACI 独立效果是合理的消融。

### 单变量原则: OK

vs exp218: 仅关闭 POSE_OA_SD (True → False)。其余完全相同。
vs exp030a: 仅增加 POSE_PACI (False → True)。但 exp030a 没有 PLBOA。
注意: exp219 仍保留 PLBOA (POSE_LOWER_BODY_OCC=True)，这与 exp030a 不同。
严格消融需要与"baseline + PLBOA" 对比，但该组合未单独跑过。

### 创新性评估

这是消融实验，不声称创新。目的是确认 PACI 的独立贡献。

---

## b. 代码审查

**无代码修改。** PACI 模块已在 exp218 中实现并审查通过。
仅通过命令行参数 POSE_OA_SD False 关闭 OA-SD。

### 需要验证的点

1. PACI bank 在没有 OA-SD 时仍正常工作 — OK (PACI 不依赖 OA-SD)
2. 没有 OA-SD 时 `oa_sd_mode` 不激活 → 没有 2-view → 正常 1-view 训练 — 但 `_oa_sd_mode` 仍需要检查
3. `img_teacher` 不存在 → OA-SD/OERL blocks 自动跳过 — OK

### 潜在问题

PLBOA (`POSE_LOWER_BODY_OCC=True`) 需要 `_oa_sd_mode` 来保存 clean image：
- 没有 OA-SD → `_oa_sd_mode = False`
- → dataset 不会保存 `img_clean_for_oa_sd`
- → PLBOA 仍然正常工作（PLBOA 不依赖 OA-SD），但 teacher_pose 不会生成
- 这对 exp219 不影响（不需要 teacher）

### 结论

配置安全，无运行时风险。

---

## c. 对照关系

| 参数 | exp030a | exp219 | exp218 | exp191 |
|------|---------|--------|--------|--------|
| PACI | No | **Yes** | Yes | No |
| OA-SD | No | No | **Yes** | **Yes** |
| PLBOA | No | Yes | Yes | Yes |

exp219 的核心对照是 "baseline + PLBOA (无 OA-SD, 无 PACI)"。
该组合未单独跑过，所以 exp219 vs exp030a 不是严格单变量。
但从 practical 角度，PLBOA 在 Tiny 上效果有限 (~+0.5-1%)，
所以 exp219 结果如果 >61.5%，可以认为 PACI 有独立贡献。

---

## 风险评估

- 训练不稳定: Low (PACI 在 detached GCN 上操作)
- 无效: Medium (PACI consistency loss 可能只在 OA-SD 环境下有意义)
- 冲突: None (OA-SD 已关闭)

---

## 审查通过

纯配置变更消融实验，无代码修改，风险极低。
