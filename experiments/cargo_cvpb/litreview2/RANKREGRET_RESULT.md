# Rank-Regret (Rank-Instability) 效率路由 — 零训练 kill-switch 结果

> **判定: DEAD（4/4 配置，两数据集 × 两 stage 全 NO-GO）。** 撞 CFPER（Test B 死）+ 无效率收益（Test D 死）。
> 数据: 冻结强 ckpt（Market exp260b 94.6 / Occluded-Duke exp255 73.1）, 零训练 + no_grad + numpy。
> 脚本: `cvpb_rankregret_killswitch.py`；日志/JSON: `rr_logs/`。两层审查（Claude + Codex）通过，含 Codex 三个 High 公平性修复。

## 0. re-frame
ReID 默认所有 query 跑 full 网络。重定义: 用 cheap（早 Swin stage GAP）vs full（BNNeck）的**检索排名不一致度 RI（Rank-Regret）** 路由——低 RI 早退 cheap、高 RI 走 full。生死线 = RI 是**关系级/检索结果级**变量, 必须比静态难度代理（margin/entropy/norm/top1-gap）更预测 (AP_full−AP_cheap), 否则退化成 CFPER（ICME2025, query-difficulty adaptive ReID）撞车。

## 1. ★架构墙（这是核心发现, 也是死因）
Swin depths (2,2,18,2) 把 ~75% FLOPs 压在 stage-2。各 stage GAP 单独检索 mAP vs 跑完该 stage 的算力占比:

| stage GAP | dim(small/base) | 算力占比 (cum) | Occluded-Duke mAP | Market mAP |
|---|---|---|---|---|
| stage0 | 96 / 128 | **0.083** | 0.27 | 1.19 |
| stage1 | 192 / 256 | **0.167** | 0.23 | 1.21 |
| stage2 | 384 / 512 | **0.917** | 47.70 | 91.01 |
| stage3 | 768 / 1024 | 1.000 | 72.75 | 94.34 |
| FULL(BNNeck) | 768 / 1024 | 1.000 | **73.05** | **94.61** |

→ **没有一个 stage 既便宜又有用**: stage0/1（省 83-92% 算力）特征近乎随机（mAP 0.2-1.2）; stage2（第一个可用特征 48/91）只省 8%。任何 cascade 都被这堵墙卡死。（FULL mAP 与训练端 exp255/exp260b 一致, 抽取干净。）

## 2. 四配置完整结果

| 数据集 | cheap stage | cheap/full mAP | Test A rho(RI,APgap) | **Test B: RI rho vs 最强静态** | **Test B partial RI\|8静态** | Test C cheap→RI | **Test D deploy(@target)** | OVERALL |
|---|---|---|---|---|---|---|---|---|
| Occ-Duke | stage1 | 0.23 / 73.05 | +0.123 | +0.123 vs **+0.403**(full_margin) | +0.140 | +0.459 | RIhat 37.9 **< random 38.1** < APgap 40.1 (@60%) | **DEAD** |
| Occ-Duke | stage2 | 47.70 / 73.05 | +0.393 | +0.393 vs **+0.420**(cheap_ent) | **+0.045**(塌) | +0.781 | RIhat 58.2 < APgap 58.8, 仅省8% (@94%) | **DEAD** |
| Market | stage1 | 1.21 / 94.61 | +0.102 | +0.102 vs **−0.335**(full_ent) | +0.123 | +0.606 | RIhat 49.4 **< random 49.8** (@60%) | **DEAD** |
| Market | stage2 | 91.01 / 94.61 | +0.114 | +0.114 vs **+0.425**(cheap_ent) | **−0.057**(塌) | +0.373 | RIhat 92.97 < APgap 93.16 (@94%) | **DEAD** |

## 3. Test B（★CFPER 生死）——撞车坐实
**RI 在所有 4 配置都被最强静态难度代理碾过（3-4×）**, RI 从不更强:
- stage0/1（cheap 随机）: RI marginal 仅 +0.10~0.12, 而 full-side margin/entropy +0.33~0.40。RI 的预测力低且非独立。
- **stage2（cheap 可用, 这才是关键反例）**: RI marginal 看似不错（+0.39/+0.11）, 但**控住 8 个静态代理后 partial 塌到 +0.045 / −0.057** —— 即当 cheap 特征足够有信息量时, RI 几乎**完全**由静态难度解释。这正是 CFPER：RI ≡ 静态难度。
- 结论: RI 不是独立于难度的「关系级」信号; 它就是难度的一种含噪测量。**Test B FAIL ×4 → 撞 CFPER。**

## 4. Test D（效率）——无收益, 被 CFPER 与 random 双杀
公平对照（全 cheap-only, cross-fit 5-fold OOF, 与 RI_hat 同输入）:
- **deployable RI 路由（RIhat-DEPLOY）在 4/4 配置都 ≤ deployable cheap-static AP-gap 路由（CFPER）。** 在省算力的 stage1, RIhat 甚至 < random。
- **compute@99% full mAP**: stage1 全 = 1.000（cheap 太弱, 必须对几乎所有 query 跑 full → 零节省）; stage2 RIhat 能到 0.95-0.988, 但 APgap(CFPER) 同样 0.95-0.988, 且绝对节省 ≤5-8%。
- ORACLE 上界（用 full 信息的 RI 排名 / full-side 静态）也输给 full-side 静态难度, 且不可部署。
- 结论: **没有「~50-60% 算力拿 ≥99% full mAP」的点存在**（架构墙 + cheap 无用）, 且即便有, RI 也不比 CFPER 强。**Test D FAIL ×4。**

## 5. Test C（可行性）——唯一 PASS, 但无意义
cheap-only 多元能估 RI（spearman +0.30~0.61, R² 0.14~0.38）。但这恰恰反证撞车: cheap-only 能估出 RI ⟺ RI 主要是 cheap difficulty 的函数。Test C PASS 不救命, 因 B 和 D 已死。

## 6. 诚实判定
**判死。** 两个生死测试都失败:
1. **Test B（撞 CFPER）FAIL**: RI 从不beat 静态难度; cheap 可用时 partial 塌到 0 = RI≡难度。先前 novelty 分析（`novelty_rankinstab.txt`, 7/10）明确说「若退化成 cheap margin/entropy/top1-gap 就是 CFPER, 新意很弱」——数据正落在此。
2. **Test D（效率）FAIL**: deployable RI 4/4 ≤ deployable CFPER, 省算力 stage 连 random 都不如; 架构墙（Swin 75% FLOPs 在 stage2）使「便宜且有用」的 cheap 出口不存在。

不粉饰: RI ≈ 静态难度代理（B 死）, 且 cheap-only 估出 RI 反而坐实它是难度（C 的 PASS 是反向证据）, Test D 无 Pareto 收益。**不立项。**

## 7. 边界 / 可复用
- 这套基建（单 forward 抽全 stage GAP + BNNeck; RI@K 三度量 RBO/overlap/kendall; cross-fit 路由; 精确 O(Nq) cascade）可复用于任何「cheap-vs-full 路由」想法。
- **架构墙是普适教训**: 在 FLOPs 高度集中于单 stage 的 backbone（Swin/多数层级 ViT）上, early-exit-by-feature 的效率上限被 backbone 的 FLOPs 分布钉死, 与路由信号好坏无关。换均匀 FLOPs 的 backbone（ResNet 各 stage 较均匀）或多分辨率输入或许才有 early-exit 空间——但那是另一个问题, 且 remedy 已被 CtF/DaReNet 占。
