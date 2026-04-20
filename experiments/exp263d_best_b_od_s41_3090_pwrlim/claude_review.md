# Claude Review — exp263d_best_b_od_s41_3090_pwrlim

**审查对象**: exp263 系列变体, seed 42 → seed 41 切换重跑

## 审查范围

1. `design.md` — 单变量 (SEED 42→41) 相对 exp263c 的隔离
2. 代码改动: **无**
3. Base OD Full Scaffold 环境与 exp263c 完全一致 (docker 容器 + pwrlim 280W + solider-reid env)
4. 启动命令与 exp263c 唯一差异: SEED + OUTPUT_DIR

## 变量隔离

- 相对 exp263c (seed 42) 严格单变量: `SOLVER.SEED 42 → 41`
- 其他所有参数 (backbone, config, scaffold, pwrlim, env) 保持不变
- 按用户指示,exp263 系列代表 FINAL 数字用 seed 41 (seed 42 疑似 trajectory 异常)

## 启动命令合法性

- Python path `/root/miniconda3/envs/solider-reid/bin/python` 是 lab3090 docker 容器内 env,已验证 (exp263/exp263c 都用此 path)
- config `configs/occluded_duke/prcv_best_base.yml` 默认开 LGPA + GCN + OA-SD + ParAug + LOWER_BODY_OCC (Full Scaffold)
- OUTPUT_DIR 新路径,不会覆盖 exp263c 历史

## 风险

1. **Seed 41 是否复现 exp263 原 trajectory**? 未知,但 seed 41 是常用种子,预期合理 (mAP e10 应 > 20)
2. **pwrlim 280W 是否稳定**? 已在 exp263c 跑到 e31 无 GPU hang,稳定性验证 OK
3. **docker 容器状态**: exp263c 运行 3h46min 后仍 Rl (running),kill 后需启动 exp263d 替代。正常操作

## OOM 风险

- exp263 原跑到 e100 之后 OOM kill (eval 内存 13.2G→16G)。pwrlim 280W 下 exp263c 跑到 e31 无 OOM,应该能跑完
- 若 e100 eval 仍 OOM,同 exp263 处理:effective FINAL from e100 ckpt

## 时间预算

- 3090 pwrlim 280W: Base backbone 421s/epoch × 120 = 14h05min
- 启动 23:30 CST → FINAL tmr ~13:35 CST
- 晚于 exp273/275 chain 但独立,不影响 Phase 3-A/B

## 与 PRCV 论文用途

- exp263 系列是 Phase 1 Base OD Full Scaffold 表示列
- 主表用 exp263d seed 41 的数字 (按用户指示)
- exp263c abandoned (seed 42 异常) + exp263d 是 seed 41 的完整 replacement

## 结论

**审查通过**。单变量 SEED ablation,代码零改动,环境和启动 pattern 与 exp263c 完全一致。

建议:
- 命名 `exp263d_best_b_od_s41_3090_pwrlim` 清晰区分 seed 41 (文件名含 _s41 而非 _s42)
- monitor.md 记录 seed 42 (exp263c) abandoned 原因和 seed 41 开始时间
- 若 exp263d e10-20 轨迹 normal (mAP 20+) 则证明 seed 42 异常,论文用 seed 41;若 seed 41 也低,则可能是 pwrlim 280W 对 Base 的 warmup 影响 (需进一步调查)
