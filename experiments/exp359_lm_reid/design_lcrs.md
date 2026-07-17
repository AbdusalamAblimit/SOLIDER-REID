# exp359-LCRS: Lattice-Complementary Residual Subspaces（用户想法挖出，2026-06-27）

## 动机

用户 2026-06-27 二次质疑链挖出此方向：
1. 用户提想法："LM-ReID test-time 对一个人多种糊法综合判断，训练时能不能也合成多糊法综合判断 / 多糊法特征尽可能相同？"
2. 我回"训练端 8 机制穷尽"（consistency −1.73 / Hard-Lattice 76.9）。
3. 用户："查的同时去查查当时代码的正确性"（别凭"当时负"打发）。
4. **codex 代码正确性审查（codex_train_correctness.md）证实**：`cvpb_lm_reid_train.py` 代码无明显 bug，但 **"训练端穷尽"结论不可信**——`LCRS/LRFD/LC-STN/DeepSets` 4 个 cheap probe 排队**没跑**，Hard-Lattice 没强制清零 `lam_marg/lam_cons` 不干净。

用户直觉"多糊法**不压成一样**而是**互补**"恰是 LCRS（train_time_pipeline.md 7/10 候选，从未跑）。区别于已死的 consistency（invariance-collapse −1.73）。

## 核心假设

test-time K=9 lattice marginalization 有效（LM-ReID 6.5）。训练端的正确做法不是压多糊法特征一致（已死），而是**每个糊法 variant 都保持身份判别力 + 残差子空间互补（decorrelation）**→ K=9 marginalization 拿到更富的 union 证据。

## 技术方案（cvpb_lcrs_probe.py）

```
z_k = norm( P_shared(g_k) + alpha * P_axis[k % n_axis](g_k) )
  P_shared : 共有身份证据（D→D linear, init eye）
  P_axis   : phase/bbox/kernel 各一残差子空间 head（D→D, init 0）
身份 loss : per-variant set-retrieval SupCon（每个 z_k 单独检索 zg，避 L_marg train-ID classifier 塌缩死因）
decorr    : 同样本不同 axis 残差互补（只 correct 样本，axis-level 非全局 Gram）
test      : K=9 z_k logmeanexp marginalization
```
frozen backbone + cached K=9 feats（复用 `cvpb_lattice_killswitch` / `cvpb_lsmrt_probe` framework）= cheap。

## codex 审查（两轮）

- **代码正确性审**（codex_train_correctness.md）：训练端穷尽不可信，LCRS 没跑，建议先做 LCRS cheap probe。
- **LCRS probe 审**（codex_lcrs_review.md）：verdict **needs-attention**，5 bug：
  1. 身份 loss 不是 per-variant（logmeanexp 后 SupCon = set-level）→ 修：每 z_k 单独 SupCon
  2. correct-only decorr 失效（S 含对角线 argmax 是自己）→ 修：S 去对角线 + top1 同 pid 才 correct
  3. residual decorr 不是 axis 互补（全局 Gram 含身份/样本）→ 修：同样本 axis-a vs axis-b cos
  4. full run OOM（一次 materialize 11 万 PIL）→ smoke 用小 train_cap 避
  5. K-cos 含对角线 + 缺 individual variant mAP（诊断，次要）
- **★codex (d) 高风险警告**：LS-MRT frozen 重投影 +0.028 / LPA +0.075 已证 frozen 后处理无 headroom，LCRS axis-conditioned residual 概念稍不同但仍是 frozen final-feature 后处理，**更可能是 residual 噪声/过拟合**。修完小 smoke 验，不主线 full run。

## 预期结果

- **PASS**：K=9 gain ≥ +0.5 over uniform(no-P) + individual variant mAP 不掉 >0.8 + K-cos 不升 → LM-ReID 从 6.5（test-time only）冲 7-8（train+test 完整方法）
- **DEAD**：撞 frozen 墙（gain ≈ 0/负，residual 噪声，codex 预期）→ 诚实杀，但干净验过（修了 5 bug，不像第一版）

## 对照组

uniform-lattice-marg（no-P，同 K=9 同 logmeanexp，只差 LCRS heads）。codex 确认内部对照公平（但 protocol 是 LR canonical gallery，不能直接和 prior LM-ReID/LS-MRT 数字比）。

## 状态 / 结果

### smoke（train_cap=2000, 10ep, h=16, K=9）2026-06-27 lab-3090-d

| | mAP | K-cos |
|---|---|---|
| uniform-lattice-marg (no-P) | 74.462 | 0.9048 |
| LCRS heads marg | 69.391 | 0.9165 |
| **gain** | **−5.071** | **+0.0117（升=塌缩）** |

**verdict DEAD**。机理：LCRS heads 训练后让 K 变体特征趋同（K-cos 升 0.9048→0.9165），破坏 test-time marginalization 依赖的变体多样性 → −5.071。**和 LATS(−5.147)/LSRC 同死法**（训练端塑造/对齐变体即损害 marginalization 多样性，monitor "Why Training-Time Invariance Fails"）。codex 高风险预警（frozen 后处理无 headroom + residual 噪声/塌缩）对。

（首版 smoke 训练完到 eval 被 OOM killer 杀 = codex bug4 警告的全量 materialize；修 streaming 后重跑得此结果。codex 5 bug 全修过才得干净 −5.071。）

### full run（全 train, 30ep）2026-06-27 坐实 DEAD

| | mAP | K-cos |
|---|---|---|
| uniform-lattice-marg (no-P) | 74.365 | 0.9047 |
| LCRS heads marg | 69.401 | 0.9229 |
| **gain** | **−4.964** | **+0.0182（升=塌缩更明显）** |

**verdict DEAD 坐实**（full run −4.964 ≈ smoke −5.071，不是 smoke artifact）。loss 30ep 1.12-1.17 震荡几乎没降 = frozen 特征上无可学梯度。K-cos 升比 smoke 更明显（0.9047→0.9229）→ 更训练更塌缩。

**★结论**：用户"多糊法互补"想法（LCRS）真 measure 证负。机理坐实——**训练端塑造/对齐 K 变体（无论压一致 invariance / 残差互补 LCRS / token sidecar LATS / set-loss LSRC）都破坏 test-time marginalization 依赖的变体多样性**。`LCRS(−4.964) + LATS(−5.147) + LSRC对称(−1.92)/非对称(−0.33) + consistency(−1.73)` 五点连成此定律。LM-ReID 训练端这格真 measure 填实（不再凭外推）。

### 剩 LRFD / LC-STN / DeepSets（codex 代码审查发现也"排队没跑"）判定

- **DeepSets Marginalizer**（frozen embedding + 学 pairwise scorer）≈ LS-MRT(+0.028 已证 frozen scorer 无 headroom) → 机理覆盖。
- **LC-STN**（input-level canonicalize 重采样）= BLC 已数据证伪（bbox 主因子不适合 canonicalize，不像 phase 可重采样回 canonical）。
- **LRFD**（disentangle：z_id 纯身份 + r_lat 吸 lattice nuisance，推理丢 r_lat）机理稍不同（不塑造变体而是分离）——但 z_id 也是 frozen 特征 + head。**唯一值得 cheap measure 的（不 extrapolate）**，复用 LCRS 框架改 head/loss。
