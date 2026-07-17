# 实验 exp369: Wardrobe-Causal 换装 ReID（unlock 突破方向 #1）

## 背景
- 训练侧旧约束（SOLIDER+Market+死区）探透, codex 天花板 6.2-6.8。用户 unlock: 不限 backbone + 换数据规模 + 换 task。
- codex unlock Top5（8.0-8.6）, 自主选 #1 Wardrobe-Causal Clothes-Changing/Long-Term ReID（8.6）。

## Novelty 边界（codex 细查, 必守）
- ✗ 纯 disentangle / causal intervention / adversarial 已被占满: AIM(CVPR23 causal)/CCFA/CLIP3DReID/FIRe/SCNet(shape-aware)/2024 counterfactual。
- ★ 收窄到 novel 的角: **反事实换装一致性 + clothing-residual 当显式建模、可度量、可消融的"泄漏量"**, 证明压制泄漏量直接降 cross-clothes top-k false。卖点从"分解"→"**残差泄漏可度量+可压制**"（AIM/SCNet 没做死的角）。

## 核心假设
- frozen foundation（DINOv3）下, cross-clothes 时 identity-core 证据被 clothing 淹没 → 去衣服（parsing mask）有 oracle headroom → 训练端可压 clothing-residual 泄漏。

## cheap kill-switch（frozen DINOv3, 零训练, 豁免审查）
codex spec:
1. frozen DINOv3-L 抽 PRCC query/gallery global feat, L2 norm（不 fine-tune）
2. PRCC cross-clothes(C) vs same-clothes(A/B) 两协议分别 mAP/R1
3. oracle headroom: SCHP human parsing mask 掉上衣+下装, 只留 head+shape, 重抽 DINOv3 masked feat, 看 cross-clothes mAP 是否 ≥+5
4. #false-in-top10 控制: 按 baseline cross-clothes 每 query #false-in-top10 分桶, 高 false 桶看 masked/oracle 是否仍有 ΔmAP（排除 trivial）

## GO / NO-GO
- **GO**: 去衣服 oracle 在 cross-clothes 高 false 桶仍 +5 mAP / +10 R1 → identity-core 被衣服淹没, evidence factorization 有训练空间。
- **NO-GO**: masked 不升甚至降 → DINOv3 已把身份证据编码够好, 衣服没淹没 → 弃 #1 转 #4 RGB-Event。

## 数据 / 模型
- 数据: PRCC（221 ID, isee-ai.cn/~yangqize/clothing.html）先用; LTCC（152 ID）备。
- backbone: frozen DINOv3-L/16（HF facebook/dinov3, 本地代理下→rsync 3090）。
- parsing: SCHP 预训练（别自训）。

## fallback
- #1 太挤 → #4 RGB-Event（EvReID 新基准, foundation 化没人做, novelty 更干净, 风险=数据小能否撑满一篇）。

## 诚实标注
- CC-ReID crowded, novelty 全押在"残差泄漏可度量+可压制"收窄。kill-switch 先验 headroom, GO 才开训练工程。
