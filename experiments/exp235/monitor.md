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
