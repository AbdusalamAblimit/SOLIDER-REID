# exp293_best_b_m_plboa_s42 — Swin-Base Market 满血 Full Scaffold + PLBOA

## 动机

Phase 1 Market 实验 (exp267/268/269) 都 **关闭了 PLBOA** (`configs/market/prcv_best_*.yml:32  POSE_LOWER_BODY_OCC: False`)。结果是 **Market 上 OA-SD 失效** — 训练 log 显式 WARN:
```
[OA-SD/RD] WARNING: PLBOA is disabled. Teacher and student see near-identical images.
```
OA-SD 依赖 PLBOA 创造 teacher (clean) vs student (lower-body occluded) 差异, PLBOA 关 → 蒸馏信号 ~ 0, OA-SD 等价 no-op。

**问题**: exp269 Base Market e80 eff FINAL **94.4/97.0** (仅 8 epoch partial, silent exit), 在关 PLBOA 状态下达到。如果开 PLBOA, OA-SD 真正生效, 是否能进一步提升?

**论文价值**: 完整的 Base Market 主数字填 main_results Table 1 第 3 列 (之前 TBD)。并回答 "PLBOA on Market 是助力还是干扰" 的问题。

## 核心假设

**PLBOA 在 Market 上 net 收益 ≥ 0**:
- 正面: 激活 OA-SD 蒸馏 + 增加 occlusion 鲁棒训练样本
- 反面: Market 本是 full-body benchmark, PLBOA 引入 train-test 分布偏差

预期三种结果:
1. **>94.5/97.2**: OA-SD 增益 > 分布偏差 → PLBOA 应在所有 benchmark 默认开
2. **持平 94.3-94.5 / 97.0-97.2**: 两力相抵, OA-SD 帮 Market 不明显
3. **<94.1**: PLBOA 分布偏差太大, Market 应保持关闭

## 技术方案

**代码零改动**。只修 CLI override `MODEL.POSE_LOWER_BODY_OCC True`。

### 实验配置

- Backbone: **Swin-Base** (88M params)
- Dataset: Market-1501
- Base config: `configs/market/prcv_best_base.yml`
- **CLI override**: `MODEL.POSE_LOWER_BODY_OCC True` (其他保留 default)
- 完整 scaffold: LGPA + GCN512 + OA-SD + ParAug + **PLBOA (本实验启用)** + 2-stage PSG `[-2,-1]`
- Seed: 42 (对齐 exp269)
- Epochs: 120

### 机器 / 数据

- **lab4090** (24GB 4090, mmpose-abu env, idle after exp291 FINAL ~18:30 CST)
- Market data ready: `/home/afr/SOLIDER-REID/data/market1501` → `/mnt1/afrdata/Market-1501-v15.09.15` symlink
- pose_data 16864 npz (train) + gallery + query 齐全 (legacy 字段缺 visibility/target_person_idx, pose_dataset.py backward compat 处理)
- Speed 预期: Base Market ~3 min/epoch on 4090 → ~6h
- Pretrained: `swin_base.pth` (128MB, 已存在)

### auto-chain 启动

等 exp291 transformer_120.pth 出现:
```bash
nohup bash tools/queue_on_ckpt.sh \
  /home/afr/SOLIDER-REID/log/occluded_duke/exp291_target_s_od_s42/transformer_120.pth \
  configs/market/prcv_best_base.yml \
  /home/afr/SOLIDER-REID/log/market1501/exp293_best_b_m_plboa_s42 \
  /tmp/exp293.log \
  exp291_to_exp293 \
  MODEL.POSE_LOWER_BODY_OCC True \
  &
```

或等 exp291 Monitor 事件 FINAL 后手动启动。

## 对照组

- **主对照 exp269 Base Market** e80 eff FINAL 94.4/97.0 (PLBOA OFF)
- exp268 Small Market FINAL 94.3/97.3 (PLBOA OFF)
- 其他: Small Market target-heatmap exp292 (running on lab3090, PLBOA OFF)

## 预期结果

| 情景 | mAP / R1 | 含义 |
|------|----------|------|
| > 94.5 / 97.2 | PLBOA+OA-SD 正向增益 | 主 contribution, 论文 narrative 扩展 |
| 94.2-94.5 / 97.0-97.3 | 持平 | OA-SD 边际价值, Base 已饱和 Market |
| < 94.0 / 97.0 | 分布偏差 PLBOA 不适用 Market | 保留 exp269 94.4 作主数字 |

## 风险

1. **lab4090 GPU 容量**: Base 88M + Full Scaffold, 显存峰值预计 14-18GB, 24GB 4090 充足
2. **OOM on flip-test eval**: Market eval 3368 query, 4090 应不 OOM (vs lab3090 5060Ti 的 OOM 是 3090 共享机器 zombie 问题)
3. **auto-chain**: lab4090 exp291 daemon 挂靠 transformer_120.pth 触发 exp293 (tools/queue_on_ckpt.sh 已验证)

## 论文定位

- main_results Table 1 第 3 列 "Market" Base 行更新为 exp293 (若 ≥ 94.4)
- supplementary discussion: PLBOA on/off × dataset = 2×3 ablation (OD on/off, OP on/off, Market on/off)
- 若 exp293 > 94.4: 建议回 Small Market + PLBOA (exp294, 另起) 验证 Small Market 也受益
