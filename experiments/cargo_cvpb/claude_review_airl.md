# Claude Broad Review — AIRL(--airl,退化一致性,kill-switch #2)

**审查对象**: afd_train.py 的 airl_degrade(L1065)+ airl_consistency_loss(L1116)+ 训练循环 AIRL 块 + argparse(L1358)+ 验证(L1437);smoke_airl.py
**日期**: 2026-06-23
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 设计合理性(AIRL=新问题定义,kill-switch #1 PASS 后验机制能否涨点)
b. airl_degrade 逐行(退化正确 + NaN-safe)
c. airl_consistency_loss(clean detach + KL/feat 数值)
d. --airl off 字节级复现
e. 梯度回流 backbone + 无新参数
f. AMP fp32 + train/test 对称

## 逐项
### 1. 设计合理性
AIRL = resolution-degradation consistency: ground 图退化到 aerial 像素预算 + 强迫退化版预测==原版 → 学低像素预算下稳定的身份证据。**无 contrastive/late-interaction/pooling/prototype/visibility(避开 OVLI 死区)。** kill-switch #1 PASS(小尺度桶强 Swin 上仍塌 +13~19)后验机制利用。

### 2. airl_degrade 逐行(L1065-1113)
- per-image s~U[min_scale,1](L1089-1092)= 像素预算。✓
- 下采样 (s·H,s·W) **antialias=True**(L1103-1104)清洁低通 + 上采样回 (H,W)(L1105-1106)。✓
- s 圆整到全尺寸跳过(L1098-1101)恒等(min_scale=1 → no-op,smoke 验)。✓
- 可选 3x3 avg-pool blur reflect pad(L1108-1112)NaN-safe。纯 F.interpolate/avg_pool2d,无 PIL/cv2。✓

### 3. airl_consistency_loss(L1116-1148)
- **clean 侧 detach**(L1135/1142/1146)= 稳定目标,梯度只走退化分支(模型被拉去让退化预测匹配 clean,非反向)。✓
- kl: 蒸馏方向 `KL(softmax(o/τ).detach ‖ softmax(d/τ))·τ²`,log_softmax NaN-safe。✓
- feat: `1-cos(normalize(o).detach, normalize(d))`,eps 防零向量,[0,2] 有界。✓

### 4. --airl off 字节级复现
`loss += airl_lambda_eff*loss_airl` 在 `if args.airl:` 内,off 时退化/loss 函数从不调用,loss 不触碰 → baseline 逐字节复现。smoke S8 验。✓

### 5. 梯度回流 + 无新参数
退化版 forward 复用同一 model(共享权重),consistency 梯度回流 backbone conv(smoke grad_sum=2.88)。退化=augmentation,consistency=loss → **无新可学习参数,optimizer 未动**。✓

### 6. AMP/NaN + train/test 对称
consistency 全程 fp32(autocast disabled),log_softmax/normalize(eps)防爆。smoke S9 极端 logits/零向量 finite。AIRL 纯训练期,eval 路径不变。✓

## Findings
- **Critical/High/Medium: 无。Low: 无实质问题。**

## 结论
审查通过。AIRL = airl_degrade(退化正确 antialias)+ consistency(clean detach KL/feat)+ off 字节级 + 梯度回流 backbone + 无新参数 + NaN-safe + train/test 对称。smoke 16/16 全过。codex 审 + GPU 空即跑 kill-switch #2:**Swin 上 A→G +≥1.0 或最小尺度桶 +≥3.0 且 reliability AUROC ≥0.65 → 继续建机制;否则杀。**
