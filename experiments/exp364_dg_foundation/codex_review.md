# Codex Review — exp364 DG direct-FT（纯 codex 审查）

## Round 1
**Verdict**: needs-attention（不否决，思路值得开 30ep）
**Date**: 2026-06-27

## Findings
codex 审 DG direct-FT config（configs/market/swin_tiny_dg_directft.yml）+ frozen probe（solider_frozen_probe.py）。config 无 MSMT 训练泄漏（MSMT 只训后 eval，Market source 训练标准），SOLIDER Swin-Tiny/SEMANTIC_WEIGHT 0.2/256x128/SGD 0.0004/30ep/triplet/IMS64 都能做 cheap direct-FT。但 3 个口径修：

1. **预处理口径不一致（关键）**：solider_frozen_probe.py 用 ImageNet norm，config 用 PIXEL_MEAN/STD [0.5,0.5,0.5]。frozen 和 FT 必须同口径 → 改 probe 0.5 norm + 重测 frozen baseline。
2. **ckpt 加载**：probe 加载训练 ckpt（base. prefix）需 strip → backbone load_state_dict。
3. **CHECKPOINT_PERIOD 30→10**（看 epoch U-shape，eval ep10/20/30 MSMT）。
+ head-only 必须补（不挡 direct-FT，但 U-shape 锚点要它）。

判定口径（codex 详细）：主看 MSMT mAP，R1 辅助。先重测同口径 F0_Market/F0_MSMT，direct-FT 后测 FT_Market/FT_MSMT。
- Go/有燃料：FT_Market 大涨 + FT_MSMT ≤ F0_MSMT−3；更强信号 head-only_MSMT ≥ FT_MSMT+2~3。
- Kill/降级：FT_MSMT ≥ F0_MSMT+2 且 source 明显成功（full FT 没破坏 held-out）。
- Ambiguous：差距 ±1~2 mAP 或 FT_Market 没训起来，此时不能讲 harm/U-shape。

**泄漏检查（codex）**：训练只用 DATASETS.NAMES market1501，MSMT 只训后 eval，不算泄漏；Market source eval 用 query/gallery 标准 source-domain test，不算泄漏；训练用 bounding_box_train。

**单源 vs 多源（codex）**：单源 Market→MSMT 够 cheap kill-switch（若源域大涨、MSMT held-out 反低于 frozen/head-only，足够证明 fine-tune 可能破坏跨域 prior，值得进入 preservation）；但单源没 harm 不能立刻强杀 DG（最多降信心），多源 Market+Duke→MSMT 才是更标准、更强的 DG 证据。务实顺序：先单源，信号强再补多源，不为多源 dataloader 阻塞。

**epoch U-shape（codex）**：CHECKPOINT_PERIOD=10 拿 ep10/20/30 的 MSMT，看 direct-FT 过程中 held-out 是否单调下降（破坏）还是先升后降（中间 sweet spot）。head-only 是低成本关键锚点，最好别省（只 direct-FT 答"有无 held-out harm"，不答"U-shape"）。

## Round 2
**Verdict**: approve
**Date**: 2026-06-27

## Findings
3 个修复都对：
- `Normalize((0.5)*3, (0.5)*3)` 与 direct-FT config 的 PIXEL_MEAN/STD 对齐。
- `CHECKPOINT_PERIOD: 10` 能拿到 10/20/30ep 中间判据。
- `--ckpt` 的 base. strip 逻辑对：make_model.build_transformer 里 backbone 是 self.base，训练端保存裸 model.state_dict()，base.xxx → xxx 正好喂给单独的 Swin backbone。

没有阻塞遗漏。非阻断增强：以后跑 DDP/包装 ckpt 可兼容 module.base.（当前单卡 direct-FT/raw state_dict 不需要）。

## 结论
codex 审查通过（Round1 needs-attention 3 修 → Round2 approve）。
**F0 baseline 确定**（0.5 norm + camid fix）：F0_Market 15.56 / F0_MSMT 4.18。
开 direct-FT 30ep → solider_frozen_probe --ckpt transformer_10/20/30.pth eval MSMT held-out 比 F0_MSMT 4.18 判 harm：
- FT_MSMT ≤ 1.18（F0−3）= Go 有燃料（fine-tune 破坏跨域 prior，DG 有戏 U-shape）
- FT_MSMT ≥ 6.18（F0+2）= Kill 降级（full FT 没破坏 held-out，foundation-preserving 没燃料 → 转 open-set/gallery-growth）

## 实测 F0 baseline（0.5 norm + MSMT camid fix）
- F0_Market 15.56 / R1 40.50（source frozen）
- F0_MSMT 4.18 / R1 17.98（held-out frozen；camid fix 前 nan，根因 MSMT 文件名 camid 是第 3 段非 _c 前缀）
- Occ-Duke 3.27（难域 frozen 更弱）
- 0.5 norm vs ImageNet norm 影响极小（15.56 vs 15.62），确认 SOLIDER frozen 行人 ReID prior 真实弱、非口径问题。

## Round2 非阻断增强（codex）
以后跑 DDP / 包装 ckpt（含 module.base. 前缀或 'model' 字段）可在 probe 加载逻辑兼容；当前单卡 direct-FT 保存裸 model.state_dict()，base.xxx→xxx strip 已足够，不需要。

## 下一步执行
1. direct-FT 30ep（本 config）→ ckpt transformer_10/20/30.pth。
2. solider_frozen_probe --ckpt 各 ckpt eval Market(source)+MSMT(held-out)。
3. 按判定线判 Go/Kill；direct-FT 后补 head-only（freeze backbone）做 U-shape 锚点。
