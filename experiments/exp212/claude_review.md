# exp212 Claude Review — Small GCN+PAA+CE+OA-SD LR=0.0008

## 审查范围

a. design.md 合理性、单变量原则
b. 代码修改（无）
c. 配置文件
d. defaults.py
e. scheduler 逻辑
f. 与 exp206r 的对照

---

## a. design.md 审查

### 假设合理性: OK

- Swin-Tiny (28M params) 用 LR=0.0008，Small (50M params) 用 LR=0.0004 是当前惯例
- 但这个惯例从未做过消融验证。尝试 LR=0.0008 在 Small 上是合理的单变量探索
- exp206r 结果为 70.6/82.6 (eq), 72.3/82.9 (maxsim)，提供了清晰的对照基准

### 单变量原则: OK

仅改 `SOLVER.BASE_LR` 从 0.0004 到 0.0008，其余完全相同。

### 创新性质疑（按协议要求）

这是纯超参数调整，不是创新实验。但 design.md 并未声称这是创新，而是定位为"LR 消融"。作为 supporting experiment 是可接受的。只要不作为主线创新方向即可。

---

## b. 代码审查

**无代码修改。** 仅通过命令行参数 `SOLVER.BASE_LR 0.0008` 覆盖。

---

## c. 配置文件审查

基础配置 `pose_psg_gcn_paa_roa.yml` 的 `SOLVER.BASE_LR` 已经是 0.0008（第 52 行）。
这是 Tiny 的默认值。exp206r 通过命令行覆盖为 0.0004（Small 惯例）。

**关键确认**: exp212 如果直接用此 config 而不加 LR 覆盖，BASE_LR 自然就是 0.0008。
或者如果显式传 `SOLVER.BASE_LR 0.0008`，效果相同。两种方式都正确。

需要注意：启动命令必须确保除 LR 外的其余参数（特别是 TRANSFORMER_TYPE=swin_small, PRETRAIN_PATH=swin_small.pth 等）与 exp206r 一致。

---

## d. defaults.py 审查

`SOLVER.BASE_LR` 默认值为 `3e-4`（第 290 行），但被 yml 配置覆盖。不影响本实验。
无新增默认值，不影响已有实验。

---

## e. scheduler 逻辑审查

`scheduler_factory.py` 的 cosine schedule 参数：

| 参数 | LR=0.0004 (exp206r) | LR=0.0008 (exp212) |
|------|---------------------|---------------------|
| warmup_lr_init | 0.01 * 0.0004 = 4e-6 | 0.01 * 0.0008 = 8e-6 |
| lr_min | 0.002 * 0.0004 = 8e-7 | 0.002 * 0.0008 = 1.6e-6 |
| warmup_t | 20 epochs | 20 epochs |
| peak LR | 0.0004 | 0.0008 |

- warmup 从 8e-6 线性升到 0.0008，20 个 epoch 内完成。坡度合理。
- 之后 cosine 从 0.0008 衰减到 1.6e-6 over 100 epochs (ep20→ep120)。

**所有数值在 float32 精度内安全。** 没有数值下溢/溢出风险。

### SGD 与 LR=0.0008

SGD optimizer, momentum=0.9, weight_decay=1e-4。
LR=0.0008 * bias_lr_factor=2 → bias 参数 LR=0.0016。
这些值在正常范围内（典型 SGD LR 范围 0.0001 ~ 0.01）。

---

## f. 与 exp206r 的对照

| 参数 | exp206r | exp212 |
|------|---------|--------|
| Backbone | Swin-Small | Swin-Small |
| BASE_LR | 0.0004 | **0.0008** |
| Optimizer | SGD | SGD |
| GCN+PAA | Yes | Yes |
| OA-SD | Yes | Yes |
| ROA | Yes | Yes |
| MAX_EPOCHS | 120 | 120 |
| 其余所有参数 | 相同 | 相同 |

**确认: 严格单变量。**

---

## 风险评估

| 风险 | 级别 | 说明 |
|------|------|------|
| 训练发散 | Low | cosine warmup 20ep 平滑升温；Tiny 用同 LR 稳定 |
| loss spike | Low-Med | Small 参数多 → 梯度方差略大，初期可能有波动 |
| 收敛变差 | Med | Small 可能确实需要更小 LR，ep10 eval 可以快速判断 |

**早停建议**: 如果 ep10 eval mAP < 40% 或 loss 出现 NaN，立即终止。
正常情况 exp206r ep10 = ~47.9%，exp212 在 44~52% 范围都算正常（LR 不同会有波动）。

---

## 结论

纯超参数实验，无代码改动，cosine scheduler 正确处理 LR=0.0008，
严格单变量 vs exp206r，风险低。

## 审查通过
