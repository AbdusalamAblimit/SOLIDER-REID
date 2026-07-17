# Claude Review — exp285b_gcn512_2stg_s_od_s42

**审查对象**: exp262 (srvA 5060Ti) 的 lab4090 同设备复制验证

## 审查范围

1. `design.md` — 零变量 rerun, 跨设备复现验证
2. 代码改动: **无**
3. Config 完全复用 `prcv_best_small.yml` default (GCN512 + 2-stage PSG + Full Scaffold + seed 42)
4. CLI override 仅 OUTPUT_DIR, 无其他参数修改 (yml 默认值即 exp262 config)

## 动机再审

exp262 原值 73.8/83.1 @ srvA (2026-04-19 09:59) 用 pre-fix code:
- 2026-04-20 14:13 用 ckpt + fix code test.py re-eval: 73.8/83.1 (完全一致,bug 是 no-op)
- decisions.md L3619-3632 分析: Phase 1 `POSE_GCN_PER_PART=False`,equal_concat 只 2 blocks, flip bug 无效应

**但残余疑虑**:
1. **跨设备 cudnn 非确定性**: srvA 5060Ti vs lab4090 4090,float32 ops 顺序可能有 ULP 级差异 → 累积到训练端 e120 ckpt 可能 0.1-0.3 mAP 差
2. **OA_SD=True 训练时打破 flip 对称性**: 虽然 eval 端的 flip bug no-op (因 block_count=2), 但训练梯度轨迹可能因 OA_SD 对数据方向性的敏感略有 device 差
3. **论文审稿**: reviewer 若质疑 "exp282 (lab4090) vs exp262 (srvA) 对照不严谨"难回答

## 解决方案

exp285b on lab4090 rerun exp262 config,提供:
- Phase 3-B Small 2×2 矩阵同设备完整对照 (exp282/283/284 + exp285b 都在 lab4090)
- exp262 数字可信度验证 (若 exp285b ≈ 73.8/83.1,则论文可任选)

## CLI override 语法

- `prcv_best_small.yml` default: GCN512 + PSG_STAGES `[-2,-1]` + POSE_ENABLED True + LGPA True + GCN True + OA-SD True + ParAug True + LOWER_BODY_OCC True + SEED 42
- **无需任何 override**, daemon 只传 hardcode `SOLVER.SEED 42 OUTPUT_DIR ...`,yml default 自动 apply
- EXTRA_OVERRIDES 空 array, shift 5 后 `"$@"` 为空, yacs 只吃 hardcode override,和 exp262 完全等价

## OOM 风险

- exp262 在 5060Ti 16G 跑过没 OOM, lab4090 24G 更宽松
- Small Full Scaffold flip eval ~10GB, 安全

## chain 位置与时间预算

- lab4090 chain: exp282 (running) → exp283 → exp284 → exp277b (seed 41 重跑 exp277) → **exp285b** (本)
- Small Full Scaffold lab4090 速度 165s/epoch × 120 = 5.5h/run
- 启动预计: exp284 tmr ~15:00 → exp277b tmr ~16:50 (若 seed 41 健康) → exp285b tmr ~22:30
- FINAL tmr ~04:00 后天 CST,在 PRCV deadline (04-30) 前非常充裕

## 预期结果

若 exp285b FINAL 73.6-73.9 / 83.0-83.3 (±0.2 from exp262): **exp262 可信**
若 exp285b FINAL < 73.5 或 > 74.1: **设备 / code 版本影响 > 0.3**, 论文主表用 exp285b

不论哪种,**exp285b 都给 Phase 3-B 矩阵提供同设备 gold-standard 数据点**。

## 结论

**审查通过**。零代码零变量 rerun,风险极低,论文价值明确(严谨性 + 审稿鲁棒性)。

daemon 挂在 exp277b → exp285b,chain 已有 5-stage (exp282→283→284→277b→285b)。
