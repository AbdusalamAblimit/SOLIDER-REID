# exp235 Tiny + FSDC (Feature-Space Diffusion Completion) + OA-SD 监控

配置: Tiny + GCN+PAA+OA-SD+PLBOA+ROA + FSDC(w=0.5, 2-layer denoiser)
对照: exp191 (Tiny OA-SD): 63.2/75.4
**创新**: 首次在 ReID 特征空间引入 denoising-based feature completion

## 检查点

### [16:34] 检查点 #1

本地启动成功。ep1.
fsdc_recon=14.988 (高, 但应该快速下降)
fsdc_mask_ratio=0.220 (~22% tokens masked per sample)
**决策**: 等 ep10 eval

### [16:35] 检查点 #2

ep1 done. Speed=148 s/s (vs baseline ~160 — denoiser 开销极小)。
fsdc_recon=14.94 (下降中)。ETA ~3h。
**决策**: 继续

### [16:39] 检查点 #3

ep4. fsdc_recon=8.92 (**从 14.99 快速下降到 8.92!** denoiser 在学习)。
ep10 eval ~12min。
**决策**: 继续

### [16:41] 检查点 #4

ep5. Speed=151 s/s. ETA ~2h52m. ep10 eval ~9min.
**决策**: 等 ep10 eval

### [16:44] 检查点 #5

ep7. fsdc_recon=4.43 (**14.99→8.92→4.43 — 快速收敛!**). ep10 eval ~5min.
**决策**: 等 ep10 eval

### [16:47] 检查点 #6

ep9. fsdc_recon=3.45. ep10 eval ~1min.
fsdc_recon trajectory: 14.99→8.92→4.43→3.45 — denoiser 快速学习特征空间结构。
**决策**: 等 ep10 eval

### [16:50] 检查点 #7 — ep10

**ep10: 36.3/49.0** (vs exp191 34.3/46.8 = **+2.0/+2.2**)

**FSDC 正向！** mAP +2.0, R1 +2.2 超过 baseline。
与 BT-PKD ep10 (+3.2/+3.2) 类似的早期加速，但 FSDC 的机制完全不同。
fsdc_recon=3.45 (快速收敛中)。

远高于 25% 早停线。
ETA ~2h45m。
**决策**: 继续！观察 FSDC 的加速是否像 BT-PKD 一样后期消退，还是能持续

### [16:52] 检查点 #8

ep12. fsdc_recon=2.60 (继续下降: 14.99→8.92→4.43→3.45→2.60)。
ep20 eval ~16min。
**决策**: 等 ep20 eval

### [16:55] 检查点 #9

ep14. fsdc_recon=2.34. ep20 eval ~12min。
**决策**: 等 ep20 eval

### [16:58] 检查点 #10

ep17-19. fsdc_recon~1.9. ep20 eval imminent。
**决策**: 等 ep20 eval

### [17:06] 检查点 #11 — ep20

**ep20: 46.6/59.1** (vs exp191 46.0/58.0 = **+0.6/+1.1**)

| Epoch | exp235 mAP/R1 | exp191 mAP/R1 | delta |
|-------|------|------|------|
| 10 | 36.3/49.0 | 34.3/46.8 | +2.0/+2.2 |
| **20** | **46.6/59.1** | **46.0/58.0** | **+0.6/+1.1** |

FSDC 仍正向！优势从 +2.0 缩小到 +0.6 (正常收敛)。
**关键对比**: BT-PKD ep20 = +1.5/+0.6, FSDC ep20 = +0.6/+1.1。
FSDC 的 R1 优势 (+1.1) 比 BT-PKD (+0.6) 更强。
**FSDC 在 detached features 上操作 → 不会有后期 backbone 干扰！**
ETA ~2h37m。
**决策**: 继续！关键在 ep50-70 是否维持正向
