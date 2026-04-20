# Claude Review — exp273_psg3_t_od_s42

**审查对象**: Phase 3-A Tiny 3-stage PSG,对标 exp271 (1-stage) / exp272 (2-stage)

## 审查范围

1. `design.md` — 结构同 exp271/272,只改 `POSE_PSG_STAGES=[-3,-2,-1]`
2. 代码改动: 无(纯 CLI override,同 exp271/272 代码路径)
3. 与 exp272 单变量差异: PSG stages `[-2,-1]` → `[-3,-2,-1]`(新增 stage 1 注入)
4. Phase 3-A 矩阵内: exp270 baseline → exp271 1-stage → exp272 2-stage → **exp273 (本) 3-stage**
5. `POSE_BACKBONE_PSG=True` 路径已在 exp271 跑通,PSG 三个 stage 的注入路径已在 exp254a/b 验证存在

## 风险

- OOM: Tiny + 3-stage PSG 额外显存开销 ~0.3GB,远低于 16GB 上限,**无 OOM 风险**
- 收敛: 3-stage PSG 在浅层可能引入过量 pose 先验,历史 exp007 (stage3-only) 58.3/67.9 > exp009 (stage2+3) 58.3/67.2,3-stage 可能更差
- PSG 模块本身已在 exp254a (2-stage + full scaffold) 训练到 74.0/84.0 稳定,代码成熟

## 结论

**审查通过**。
