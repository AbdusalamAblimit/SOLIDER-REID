# exp364 DG Foundation-Preserving Adaptation — monitor

## DG cheap kill-switch 第一步：frozen cross-domain probe（零训练，2026-06-27）

验证前提：frozen DINOv2-reg 有无跨域 ReID prior（DG 方法的核心假设是"frozen foundation 有 prior、fine-tune 破坏它"）。

### debug 历程（多轮排除假设）
1. patch14 倍数 bug（256x128 → AssertionError）→ 改 252x126
2. 252x126 mAP 1.35（反常）→ 怀疑非方形 dynamic interpolate 退化 → 改 224x224
3. 224x224 CLS mAP 1.55（还低）→ 排除尺寸；debug 确认 camid(1-6)/dim(768)/pid(750/751 Market 标准)都对
4. CLS → patch tokens GAP（skip CLS+4register）→ mAP 2.71（略好还低）

### ★结果：frozen DINOv2-reg 对行人 instance ReID 弱
| domain | CLS mAP | patch GAP mAP |
|---|---|---|
| Market | 1.55 | 2.71 (R1 8.88) |
| Occ-Duke | 0.68 | 0.89 |

- 排除尺寸/camid/dim/pooling 后，一致低（接近随机，Market frozen ReID 文献该 20-40）→ 指向 **DINOv2-reg（通用自监督）对行人 instance ReID 弱**。
- AG（exp363）frozen DINOv2 也低（8.62）一致。
- codex 查文献判中（codex_dinov2_verdict.md）：2.71 是否正常 / DG 用 DINOv2 前提是否存疑（frozen prior 弱谈何破坏）/ 换 CLIP-ReID/SOLIDER / DG 降级转 open-set。

### cheap probe 价值
零训练几轮 debug 就暴露"DINOv2 不是好 ReID foundation"，没闷头训 direct-FT 才发现 base 弱（同 AG frozen baseline 验死视频证据积累）。

## SOLIDER frozen base bounded retest（codex 首选，2026-06-27）
| foundation | Market mAP | Market R1 | Occ-Duke mAP |
|---|---|---|---|
| DINOv2-reg（通用） | 2.71 | 8.88 | 0.89 |
| **SOLIDER（行人 LUPerson 预训练）** | **15.62** | **39.96** | 3.30 |

- SOLIDER frozen 5-6x DINOv2 → 行人预训练 foundation frozen 邻域确实有身份结构。
- codex 门槛 frozen base Market >10 → SOLIDER **15.62 勉强过线**（DINOv2 2.71 死）。
- 但 15.62 不强（fine-tuned 91.6；R1 40 但 mAP 15 = top1 有信号、ranking 弱）。DG 前提靠 SOLIDER 勉强成立、prior 不强。
- `.eval()` 返回 None 坑（PSC-JEPA memory 记过）又踩一次，已 fix（分开 .to/.eval）。
## ★codex 判第二步（2026-06-27，DG 完整方向降 5.5）
- DG 完整方向 **5.5**（SOLIDER 15.62 再降）；bounded second-step kill-switch **7/10 值得做**；直接写 preservation 方法**不值得**。
- **务实：再花一次 30ep 验证，不再为 DG 追加无条件投入**。"不因 15.62 直接弃，也不因 15.62 写方法——15.62 只买一次廉价判决机会"。
- **第二步（暂不写 preservation 代码）**：head-only + direct-FT 30ep，多源 Market+Duke → held-out MSMT，SOLIDER swin。
- **判定**：direct-FT 赢 head-only/frozen（无 held-out harm）→ DG 降 3/10，转 open-set/gallery-growth；direct-FT source overfit + held-out 输 head-only → 再 λ sweep（preservation 有戏）；干净 U-shape → DG 回 6.5。
- 项目 datasets/ 有 market1501/msmt17/occluded_duke 单源 dataloader。config swin_tiny。

## ★★DG 判死（Kill，2026-06-27）：direct-FT held-out 反涨，foundation-preserving 没燃料

direct-FT SOLIDER swin_tiny on Market 30ep → probe --ckpt transformer_30 eval（同口径 0.5 norm）：

| domain | F0(frozen) | FT(direct-FT) | Δ |
|---|---|---|---|
| Market(source) | 15.56 | **88.70** | +73.1 |
| MSMT(held-out) | 4.18 | **11.37** | **+7.19** |
| Occ-Duke | 3.27 | 14.47 | +11.2 |

**codex 判定 Kill**：FT_MSMT 11.37 >> F0_MSMT+2（6.18）。full FT 不但没破坏 held-out 反而大幅提升（+7.19）→ **foundation-preserving 没燃料**（fine-tune 让所有行人域都涨，保护弱 frozen topology 15.56 无意义，没 U-shape sweet spot = DG no-op，PSC-JEPA 同质死另一形式）。

★**DG 降 3/10，转 open-set/gallery-growth/distractor-aware lifelong**（Market/MSMT/Duke 构造协议，零训练先验，不依赖 frozen prior 强）。

**今天 AG（exp363 杀）+ DG（exp364 杀）两个跳出盒子的范式方向都诚实证伪**，但全程 cheap 先验前提（frozen probe 零训练 + 一次 30ep bounded）、codex 审、诚实定位每个 bug、不被沉没成本绑架。DG 这个负结果干净（direct-FT held-out 反涨，因果清楚）。
