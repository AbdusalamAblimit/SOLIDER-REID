# exp236 Tiny + FSDC (正确增强配置) 监控

配置: Tiny + GCN+PAA+OA-SD+PLBOA(0.7)+**无ROA** + FSDC(w=0.5)
**与 exp191 唯一差异: FSDC**
对照: exp191 (ROA=False, PLBOA=0.7): 63.2/75.4

## 检查点

### [01:39] 检查点 #1

远程启动。ep1。fsdc_recon=15.02。ROA=False confirmed。
ETA ~4h51m。
**决策**: 等 ep10 eval

### [01:43] 检查点 #2

ep3. fsdc_recon=11.17 (15.0→11.2, 快速下降). ETA ~4h40m。
**决策**: 继续

### [01:50] 检查点 #3

ep6. fsdc_recon=5.47 (15.0→11.2→5.5 — denoiser 快速学习). ep10 eval ~8min。
**决策**: 等 ep10 eval

### [01:59] 检查点 #4

ep10 iter80. fsdc_recon=3.14. eval ~2min。
**决策**: 等 ep10 eval

### [02:02] 检查点 #5 — ep10

**ep10: 38.9/53.1** (vs exp191 34.3/46.8 = **+4.6/+6.3**)

**FSDC 正确配置比错误配置 (exp235 +2.0/+2.2) 更强！**
+4.6 mAP, +6.3 R1 — 这是所有实验中最强的 ep10 结果！

| 实验 | ep10 delta vs baseline |
|------|------|
| BT-PKD (exp229) | +3.2/+3.2 |
| FSDC wrong config (exp235) | +2.0/+2.2 |
| **FSDC correct config (exp236)** | **+4.6/+6.3** |

ETA ~4h20m。
**决策**: 继续！密切监控 — 这是最有前途的结果

### [02:05] 检查点 #6

ep12. fsdc_recon=2.58. ep20 eval ~16min。
**决策**: 等 ep20 eval
