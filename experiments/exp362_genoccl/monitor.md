# exp362 gap-measured occlusion engine — gap 审计（negative control）

## 由来
PSC-JEPA continued-pretrain（exp361）判死后，codex pivot 决策选生成数据引擎（6/10，窄缝=gap-measured occlusion distribution engine，必须赢 PLBOA）。第一步 cheap kill-switch：先 gap 审计验证前提（train-test 遮挡分布 gap 是否存在）。

## gap 审计（2026-06-27，cheap，no training，no diffusion）

**pose visibility threshold sweep（occluded_duke train N=15618 vs query N=2210）**：

| VIS_THR | legs gap | arm gap | heavy-occ(可见组≤2) gap |
|---|---|---|---|
| 0.3 | +9.2% (q9.5/t0.4) | rarm +0.5 | +0.1% |
| 0.5 | +16.4% (q17.2/t0.8) | larm +2.0 / rarm +2.1 | +1.0% |
| 0.7 | +26.1% (q28.1/t2.0) | larm +8.9 / rarm +5.9 | +5.1% |

**gap 形态确认：各 threshold 下始终主导 lower-body（legs），arm 次之（高 thr 才显现），heavy-occ 始终少。**

## ★结论：生成引擎窄缝被 PLBOA 占 → 转 LM-ReID（codex 2.5-3/10）

- gap 几乎全在 legs（lower-body），**正是 PLBOA（Pose-guided Lower-Body Occlusion，3-seed +1.37 mAP）已经在补的**。
- 生成引擎要成立 = 在 PLBOA 已覆盖的 lower-body gap 上**净增益**（否则只是"更贵的 PLBOA"）。窄缝太窄。
- caveat：pose visibility ≠ 真遮挡（codex），但 occluded_duke **无官方 mask**（真 mask audit 要跑 human parsing，中等成本）；且最强相对信号始终是 legs（非全身 heavy occlusion），caveat 不足以救生成线。
- **codex 综合判：转 LM-ReID**（exp359，诚实 6.5 B 类候选，表链全：强 TTA 对照/聚合消融/因子消融/K-sweep/backbone 泛化/σ-sweep/训练端反例）。exp362 保留为 audit/negative control。

## ★★换量级在 occluded ReID 内部的诚实困境（重大节点）

范式转向（换量级）在 occluded ReID 内部探索了多个 build，**都接近墙**：
- Intruder（exp360）DEAD：donor 可读但压它不救排序。
- PSC-JEPA continued-pretrain（exp361）DEAD：partial-view JEPA 与 ReID 判别性本质冲突（41/50 << 58.5）。
- 生成引擎（exp362）2.5-3/10：gap 被 PLBOA 占。
- support-bank aux 4/10 死：撞 SCKD 穷尽（+0.1）/FGEU 16.3%。

cheap kill-switch（Stage0.5 frozen 因果 / gap 审计纯 numpy）省了多次大投入——这是分工（先验证再大 build）的价值。**用户拍板节点**：接受"occluded ReID 内部换量级接近墙"+ 回 LM-ReID 6.5 收尾投 B 类（最务实）/ 生成引擎 human parsing 最后一搏 / 转非遮挡 reframe。
