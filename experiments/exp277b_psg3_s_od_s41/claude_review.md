# Claude Review — exp277b_psg3_s_od_s41

**审查对象**: exp277 (seed 42 塌缩) 的 seed 41 重跑变体

## 审查范围

1. `design.md` — 单变量 (SEED 42→41) 相对 exp277 的隔离
2. 代码改动: **无**
3. exp277 塌缩已在 exp277/monitor.md 诊断 (e2 id_global 卡 3.277 = 0.5×ln(702) uniform classifier output)
4. 用户指示: "之前类似情况出现过, 偶发随机性问题, 换 seed 41 重跑"

## 变量隔离

- 相对 exp277 (seed 42) 严格单变量: `SOLVER.SEED` 42 → 41
- 其他参数完全保持: Small backbone + PSG 3-stage [-3,-2,-1] + pure scaffold (LGPA/GCN/OA-SD/ParAug/LOWER_BODY_OCC 全关)
- Config path 和 yml default 继承一致 (prcv_best_small.yml)

## 塌缩原因再审视

exp277 e2 id_global 恒为 3.277 是 **classifier output uniform** 的 signature:
- CE = ln(702) = 6.554, 乘以 GLOBAL_LOSS_SCALE=0.5 = 3.277 ✓
- 仅 triplet loss 在学,id loss 训不动
- 可能原因:
  1. 随机初始化 + 3-stage gate 早期压缩 feature → BNNeck 输出 degenerate → CE stuck
  2. 梯度被 AMP scaler 或其他机制归零 (但没有 NaN/Inf 触发)
  3. **早期 seed 特定 bad state**, 换 seed 后可能 avoid

历史: 用户提到 "之前类似情况出现过" — seed 42 在 Small 大 backbone + 多 gate 配置下偏不稳,换 seed 可复现。

## CLI override 语法

`SOLVER.SEED 41` 在 queue_on_ckpt.sh 的 EXTRA_OVERRIDES 里放置在最后,yacs merge_from_list 线性覆盖默认的 `SOLVER.SEED 42`,最终 SEED=41。已在 exp263d 验证同模式。

## OOM 风险

与 exp277 同配置,显存占用相同 (Small + 3-stage PSG < 10GB 在 lab4090 4090 24G),安全。

## 预期结果

若 seed 41 训练正常:
- e10 mAP 45+ (接近 exp274/275/276 同期 43-45 区间)
- e120 FINAL 在 68-69 / 76-77 范围
- 验证 "Tiny 3-stage ≈ 2-stage" pattern 在 Small 上复现

若 seed 41 再次塌缩:
- 说明不是偶发 seed 问题,而是 "Small + PSG 3-stage pure scaffold" 配置的系统性不稳定
- 需进一步调查 (可能尝试 LR warmup 延长 / gradient clipping / 不同 norm 初始化)

## 机器分配

- lab4090 auto-chain from exp284 via daemon (exp284 预计 tmr 10:00 FINAL)
- daemon tag: `exp284_to_277b_s41`
- PYTHON=mmpose-abu env (与 Phase 3-A/B Small runs 一致)

## 结论

**审查通过**。单变量 SEED ablation, 代码零改动, 复用 exp277 验证过的 pure scaffold 配置。

若验证 seed 41 正常 → exp277b 替代 exp277 作为 PRCV Table 2 Small 3-stage 行, exp277 (seed 42) 降级为 decisions.md 里的 "偶发 seed 塌缩" 记录。
