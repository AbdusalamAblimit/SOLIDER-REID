# exp215 Small + GCN+PAA+CE+OA-SD + BA-PKC (weight=0.1) 监控

配置: exp206r + BA-PKC weight=0.1 (backbone-aware, NON-detached keypoint features!)
对照: exp206r (72.3 maxsim), exp210b PKC-detached (72.4 maxsim)

**关键创新**: BA-PKC 的 SupCon 梯度直通 backbone (50M params)，
而旧 PKC 只更新 GCN (200K params, detached)。

## 检查点

### [01:50] 检查点 #1

**状态**: 正常
**进度**: Epoch 2/120

| 指标 | 当前值 |
|------|--------|
| ba_pkc | 2.783 (快速下降中!) |
| ba_nk | 17.0 |
| oa_sd | 0.475 |
| id_global | 6.552 |

BA-PKC 正常工作。17 个 keypoints 全部参与。
远程 exp215b (weight=0.05) 也已启动。
**决策**: 继续。ep10 eval 将是关键——看是否与 exp206r 不同。

### [01:55] 检查点 #2

ep5. ba_pkc=3.117, id_global=6.516.
**vs exp206r ep5 id_global=6.497 — 不同！BA-PKC 确实在影响 backbone！**
id_global 下降更慢（SupCon 梯度与 CE 竞争），但这可能在后期带来更好的 metric space。
exp215b (remote) ep3. ba_pkc=4.416.
**决策**: 继续。ep10 eval ~15min。

### [02:01] 检查点 #3 ⚠️

ep8. id_global=6.545, Acc=0.007!
**vs exp206r ep8 id_global=~6.41, Acc=~0.17 — 严重落后！**
BA-PKC weight=0.1 的 SupCon 梯度与 CE 在 backbone 上冲突。
与 PKC=0.5 (exp210) 的灾难类似，但程度轻一些。
exp215b (weight=0.05) ep5: id_global=6.543 — 也很慢。
**决策**: 等 ep10 eval 确认

### [02:07] ep10 — 终止！

**ep10: 0.5/0.8% — 比 PKC=0.5 (3.6%) 更惨！**

BA-PKC weight=0.1 的 SupCon 梯度直接打到 backbone，完全摧毁 CE 收敛。
这证实了 detach() 的必要性——Part 梯度（无论是 CE 还是 SupCon）都不能回传到 backbone。

**exp215 已终止。exp215b (weight=0.05) 可能稍好但也不太可能有效。**

### 重要结论

**per-keypoint loss 的梯度不能到达 backbone。** 这是架构约束：
1. 有 detach: 梯度不影响 backbone → PKC/MST 无效
2. 无 detach: 梯度摧毁 backbone CE 收敛 → 灾难

**只有 test-time 方法（MaxSim hybrid）可以利用 per-keypoint features 的判别力。**
训练端改善 per-keypoint features 需要完全不同的方法。
