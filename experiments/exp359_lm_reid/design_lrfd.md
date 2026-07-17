# exp359-LRFD: Lattice-Residual Feature Disentangle（2026-06-27）

## 动机

LCRS（残差互补）真 measure DEAD −4.964（塑造变体 → K-cos 升塌缩）。LRFD 机理**不同**：不塑造变体，而是 **disentangle**——z_id 纯身份子空间（推理用） + r_lat 吸 lattice nuisance（必须能预测 lattice axis，推理丢）。codex 代码审查判：剩 3 probe 里（LRFD/LC-STN/DeepSets）唯一值得 cheap measure 不 extrapolate 的（DeepSets≈LS-MRT scorer，LC-STN=BLC 数据证伪）。

## 核心假设

lattice 是可分离的 nuisance。z_id 学纯身份（去 lattice），r_lat 吸 lattice（预测 axis）。推理用 z_id K=9 marg（已去 lattice noise）拿更干净身份。

## 技术方案（cvpb_lrfd_probe.py）

```
z_id  = norm(P_id(g_k))      推理用, per-variant set-retrieval SupCon 身份
r_lat = P_lat(g_k)           lattice nuisance sink, lat_cls 预测 lattice axis(CE)
orth  : z_id ⊥ r_lat         disentangle (身份子空间不含 lattice)
test  : z_id K=9 logmeanexp marg(丢 r_lat) vs uniform(no-P)
```
frozen backbone + cached K=9 feats streaming（复用 LCRS framework，含 bug4 streaming fix）。

## codex 审（codex_lrfd_review.md）

verdict **needs-attention**（核心代码基本对，可作 cheap probe 跑）：
1. gallery protocol：LR-gallery 内部对照公平（判 gain），但不能和 LM-ReID/LS-MRT 数字直接比（HR gallery 才能比）。
2. disentangle 证据：lat_acc>0.6 是必要不充分，PASS 后追加冻结 z_id→axis probe（应近 chance）。
3. 其余（z_id/r_lat/orth/SupCon/split/接口）都对。axis2 是 zoom/scale+kernel 混（不纯）。

## 结果（smoke train_cap=2000, 10ep, h=16, K=9）2026-06-27 lab-3090-d

| | mAP | K-cos |
|---|---|---|
| uniform-lattice-marg (no-P) | 74.462 | 0.9048 |
| z_id (drop r_lat) marg | 69.451 | 0.9208 |
| **gain** | **−5.011** | **+0.0160（升=塌缩）** |
| r_lat lattice-axis pred acc | **0.461**（chance 0.333）| |

**verdict DEAD**。**双重死**：
1. **lat_acc 0.461 ≈ chance** → r_lat 没真学到 lattice axis → **disentangle 前提就不成立**（codex 预警的 axis 混 zoom/kernel 本身不可分）。
2. **K-cos 升塌缩** → z_id 还是趋同，同 LCRS 死法。

★**根本 kill**：不只塌缩，是"lattice axis 可分离"这个 disentangle 前提本身错（lat_acc≈chance 实测）。比 LCRS 更根本——LCRS 是塑造变体塌缩，LRFD 连 disentangle 都没发生。

## full run（全 train, 30ep）2026-06-27 坐实 DEAD

| | mAP | K-cos | lat_acc |
|---|---|---|---|
| uniform (no-P) | 74.365 | 0.9047 | |
| z_id (drop r_lat) | 69.371 | 0.9358 | r_lat axis **0.540**（chance 0.333, <0.6 disentangle 不够）|
| **gain** | **−4.993** | **+0.0311（升塌缩更明显）** | |

**DEAD 坐实**（full run −4.993 ≈ smoke −5.011，不是 smoke artifact）。lat_acc 0.540 仍<0.6（disentangle 部分但不够，lattice axis 不可分），K-cos 升更多（0.9358）→ 更训练更塌缩。

★**LM-ReID 训练端真 measure 六点穷尽坐实**：
- 塑造变体：consistency −1.73 / LCRS −4.964 / LATS −5.147 / LSRC −1.92/−0.33（破坏 marginalization 多样性）
- disentangle：LRFD −4.993（前提错 lattice axis 不可分 + z_id 塌缩）
- frozen scorer：LS-MRT +0.028 / LPA +0.075（无 headroom）

test-time marginalization(6.5) 是唯一活的。**用户二次质疑（"查代码正确性"）把"训练端穷尽"从凭外推虚账 → 真 measure 诚实结论**（LCRS/LRFD 两个真没跑的 probe 真 measure DEAD）。
