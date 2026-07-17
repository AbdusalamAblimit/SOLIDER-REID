# exp020 审查记录

## 第一轮审查 — FAIL (1 issue)

**问题**: processor.py 中 `loss = loss + recon_loss` 创建新 tensor，丢失 `_loss_details` 属性，导致训练日志中 per-component loss 消失。

**修复**: 在 loss 相加前保存 details，相加后恢复并添加 recon loss 项。

## 第二轮审查 — PASS

**修复已应用**: processor.py 中正确保留 `_loss_details` 并添加 `recon` 条目。

### 关键验证点:
1. PRA head 接收正确的 feature map (PSG-enhanced Stage 3 output) ✅
2. MSE loss 正确计算 (resize GT → sigmoid → MSE × λ) ✅
3. 4-value return 对 PRA-enabled 和 PRA-disabled configs 都正确 ✅
4. Processor 正确处理 recon_loss (len check, None check, details 保留) ✅
5. AMP 下 MSE loss 数值稳定 ✅
6. Config 正确启用 PSG + PRA ✅
7. 单变量实验确认 — 与 exp007 仅 PRA head + MSE loss 不同 ✅
8. 参数量 ~887K 准确 ✅
