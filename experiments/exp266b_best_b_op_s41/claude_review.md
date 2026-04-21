# Claude Review — exp266b_best_b_op_s41

**审查对象**: Base OP seed 41 重跑 + 5060Ti TEST BATCH 降

## 审查范围

1. `design.md` — 双变量 (SEED + TEST BATCH) 相对 exp266 的隔离
2. 代码改动: **无**
3. 用户指示: 5060Ti 上 Base 必须降 TEST BATCH, exp266b 是首次应用该约束
4. daemon 挂 srvA: exp265b → exp266b

## 变量隔离

- 相对 exp266 双变量:
  - `SOLVER.SEED 42 → 41`: 和 exp263d 同策略 (seed 42 可能 lucky/unlucky draw)
  - `TEST.IMS_PER_BATCH 256 → 128`: 防 5060Ti + Base eval OOM (历史 exp263/exp269 eval OOM 13.2G→16G)
- 训练 IMS_PER_BATCH 不改 (保持 64), 只降 eval batch
- 其他参数保持 Phase 1 exp266 config (Base Full Scaffold WITH_CP)

## OOM 风险评估

历史证据:
- exp263 Base OD e100 eval OOM (default TEST=256) on srvB 5060Ti
- exp269 Base Market e80 eval OOM on srvA 5060Ti
- exp266 Base OP silent exit 不确定 (可能 OOM 可能外部 kill)
- 训练阶段 (batch 64) 没 OOM, 仅 eval (batch 256) OOM

**TEST.IMS_PER_BATCH 128 预期效果**:
- eval 显存消耗减半 (~6.6G → 3.3G), 总 eval peak < 11GB 留 5GB margin
- eval 时间翻倍但可接受 (Base eval 8min → 16min 相对 20h 训练可忽略)

## TEST BATCH 降是否影响数字

**不影响**: TEST.IMS_PER_BATCH 仅控制 eval 时 batch 大小 (前向传播的并发度), **不影响数值结果** (gallery/query 特征一样, cosine distance 计算不变)。
- 验证: test.py 测不同 TEST BATCH 相同 ckpt → 完全一致 mAP/R1

## CLI override 合法性

- `TEST.IMS_PER_BATCH 128` 是 yacs key, 合法
- queue_on_ckpt.sh EXTRA_OVERRIDES 传入 `SOLVER.SEED 41 TEST.IMS_PER_BATCH 128` 两个 key-value pair, 经 python parse 后均生效

## 时间预算

- Base WITH_CP on 5060Ti: ~15min/epoch × 120 = 30h
- 实际 exp266 was 836s/epoch × 120 = 27.9h, 但 exp266 在 e70 挂了只跑 14h
- 预计 exp266b 完整 28h → 后天中午 FINAL (取决于 srvA 稳定性)

## 结论

**审查通过**。

关键点:
1. SEED ablation 复制 exp263d 策略 ✓
2. TEST BATCH 降是必要 OOM 防护 ✓
3. 不改训练, 仅改 eval batch ✓
4. 论文价值: 修复 exp266 e60 eff FINAL 瑕疵, OP 主表完整性 ✓

若 exp266 其实是外部 kill 而非 OOM, TEST BATCH 128 也无副作用 (仅 eval 慢 8min)。
