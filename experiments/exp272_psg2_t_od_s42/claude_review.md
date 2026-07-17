# Claude Review — exp272_psg2_t_od_s42

**审查对象**: Phase 3-A 第 3 个 run,2-stage PSG Tiny OD

## 审查范围

1. `design.md` — 单变量相对 exp271: `POSE_PSG_STAGES=[-2,-1]`(增加 stage 2)
2. 代码改动: **无**
3. 单变量隔离: exp271 vs exp272 差异 = 是否在 stage 2 也注入 PSG
4. 代码路径与 exp271 相同(`PoseBackboneModel`,`make_model.py:467` 分支),只是 PSG gate 多 1 处注入
5. 历史 exp009 (Tiny Stage2+3): 58.3/67.2 ≈ exp007 (Stage3): 58.3/67.9 → Tiny 上多 stage 不一定有增益
6. 本 run 新协议预期: ~59-60,与 exp271 大致持平

## OOM 风险

低。Tiny + 2-stage PSG 显存 < 8GB,eval 含 flip ~10GB,余 6GB 富余。

## 结论

**审查通过**。单变量 ablation,代码零改动,风险极低。Phase 3-A 科学目的: 量化 multi-stage PSG 在 pure scaffold 下的增益(即便可能中性),为论文 ablation 提供"2-stage 是否 Tiny 上必要"的回答。
