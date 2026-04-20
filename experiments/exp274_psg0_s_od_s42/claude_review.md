# Claude Review — exp274_psg0_s_od_s42

**审查对象**: Phase 3-A Small baseline,对标 exp270 Tiny baseline

## 审查范围

1. `design.md` — 结构同 exp270,只换 backbone Small
2. 代码改动: 无(纯 CLI override,commit `f69b61c` 代码路径成熟)
3. 与 exp270 单变量差异: backbone Tiny → Small
4. Phase 3-A 矩阵内: exp274 (本) 为 Small no-PSG baseline,exp275-277 为 Small 加 PSG stage 变体
5. `POSE_ENABLED=False` 路径已在 exp270 验证可通(绕开 dead import),本 run 同路径

## 风险

- OOM: Small pure Swin 显存 ~7GB,flip-test eval peak ~10-11GB,**远低于 16GB 上限,无 OOM 风险**(只有 full scaffold + Base 才贴边)
- 收敛: SOLIDER-Small 标准训练,historical 有数字参考

## 结论

**审查通过**。
