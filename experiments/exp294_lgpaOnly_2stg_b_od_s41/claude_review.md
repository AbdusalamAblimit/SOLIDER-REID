# Claude Broad Review — exp294_lgpaOnly_2stg_b_od_s41

**审查对象**: Base backbone + Full Scaffold 去 GCN (LGPA-only) + 2-stage PSG + seed 41 on Occ-Duke
**审查日期**: 2026-04-23
**审查轮次**: 第 1 轮
**机器**: lab4090 (24GB RTX 4090, mmpose-abu env)

## 审查范围

按照 `.claude/rules/experiment_protocol.md` 第 a-f 项完整覆盖:

a. `design.md` — 动机/假设/方案/对照/预期
b. 新增/修改代码 — **无** (零代码改动, 仅 CLI override)
c. 配置文件 — `configs/occluded_duke/prcv_best_base.yml` 对照
d. `config/defaults.py` — 本次不涉及默认值改动
e. processor / forward / eval — 复用 exp286-289 已验证代码路径
f. 前序实验对照 — Phase 3-C (exp286-289) + exp263d Base Full 对照

## 变量隔离性审查 (PASS)

### 单变量断言

相对 baseline exp263d (Base Full Scaffold GCN512+2stg+s41 = 74.1/83.3, MaxSim 75.2/84.8):
- **唯一改动**: `MODEL.POSE_SKELETON_GCN True → False`
- 其他所有变量保持 exp263d 完全一致:
  - Backbone: Swin-Base (`swin_base_patch4_window7_224`)
  - PSG stages: `[-2, -1]` (2-stage)
  - LGPA: True, CLIP_DIM=512, 8 heads, TEMP=1.0, ASSIGN_WEIGHT=0.5, DETACH=True
  - OA-SD: True, WEIGHT=1.0, EMA=0.999
  - ParAug: True
  - PLBOA: True, PROB=0.7
  - Seed: 41 (已证明 Base 上最佳 seed, 对齐 exp263d)
  - Dataset: Occ-Duke, 384×128, BS=64
- 符合实验协议"单变量"铁律

### 跨 backbone 对照矩阵一致

`design.md` 表格正确列出:
- Tiny LGPA-only: exp286 (1-stg) / exp287 (2-stg) 已 FINAL
- Small LGPA-only: exp288 (1-stg) / exp289 (2-stg) 已 FINAL
- **Base LGPA-only**: exp294 (2-stg) 本 run, 补齐 3 backbone × 2-stg 的 Full-GCN 格子

## 代码路径安全性 (PASS)

- `POSE_SKELETON_GCN=False` 是 Phase 3-C (exp286/287/288/289) 已批准 + 已 FINAL 的相同 flag
- 无 dead-branch / None-deref 风险: forward pass 在 exp286-289 上已跑满 120 epoch × 4 次无 crash
- eval 路径 `eq_concat` 天然兼容无 GCN 分支 (LGPA part features + global 拼接, 不依赖 GCN output)
- 优化器参数注册: GCN 模块未构建时 `named_parameters` 不会包含 GCN 相关权重, 无 zero-grad 风险
- AMP / dtype / device: 复用 Full Scaffold 同一套, 无新增 mixed precision 边界

## CLI 配置审查 (PASS)

```bash
--config_file configs/occluded_duke/prcv_best_base.yml   ← 正确引用 Base 配置
SOLVER.SEED 41                                            ← 匹配 exp263d best seed
MODEL.POSE_SKELETON_GCN False                            ← 唯一变量
TEST.IMS_PER_BATCH 64                                    ← 遵循硬性 override 默认
OUTPUT_DIR /home/afr/SOLIDER-REID/log/.../exp294_...     ← 独立输出目录
```

交叉验证 `prcv_best_base.yml`:
- 默认 `SEED=42`, CLI 正确 override 到 41
- 默认 `POSE_SKELETON_GCN=True`, CLI 正确 override 到 False
- 默认 `TEST.IMS_PER_BATCH=256`, CLI 正确 override 到 64 (防 eval OOM)
- 其他所有 POSE_* flags 在 yml 与 design.md 预期完全一致

## OOM 风险评估 (PASS)

- lab4090 24GB, 去 GCN512 head 后参数更少 (Full-GCN 比 Full 轻 ~3-5M GCN 参数)
- 训练峰值: Base + 2-stg PSG + LGPA ~ 17-19GB (exp263d 已验证可跑)
- eval 峰值: `TEST.IMS_PER_BATCH=64` + flip-test TTA, 估计 12-14GB (从 256 降到 64 留足余量)
- 遵循 2026-04-22 OOM 教训 (exp292/exp293), 新默认值硬性执行
- 4090 24GB 对 Base 8h 训练完全安全, 无 fragmentation 累积风险

## 对照组完整性 (PASS)

**同 seed 41 Base Full-GCN 对照**: exp263d (74.1/83.3, MaxSim 75.2/84.8) — 严格单变量对照
**跨 backbone Full-GCN 2-stg 对照**: exp287 Tiny (65.9/77.0), exp289 Small (73.8/83.3) — 容量递进
**Phase 3-C 结论一致性检验**: 若 exp294 ≈ exp263d, 巩固"GCN 全容量冗余"叙事; 若显著低, 暴露"Base 容量大能吸收 GCN"分界

## 预期结果合理性 (PASS)

- 成功带 (73.8-74.2 / 82.5-83.5): 与 Small 结果 (Δ=-0.2/-0.8) 外推到 Base 合理
- 失败带 (72.5-73.5 / 81-82): 若 Base 对 GCN 敏感, 论文叙事调整为 "Small 可简化, Base 保留" 也 coherent
- 论文价值清晰: 3 个 backbone × GCN on/off 完整消融, 为 ablation 主表加一格

## 时间预算 (PASS)

- lab4090 4090 24GB, Base 4.2 min/epoch (对齐 exp263b 历史) × 120 = ~8h 30min
- ETA tmr ~02:30 CST 合理, 不冲突其他机器队列

## 发现问题

无 Critical / High / Medium / Low 级别问题。零代码改动, 零配置新增, 完全复用 Phase 3-C 已 4/4 批准 + FINAL 代码路径。唯一变量 (POSE_SKELETON_GCN=False) 已在 exp286-289 上累计跑满 480 epoch 无任何异常。CLI override 3 项 (SEED/POSE_SKELETON_GCN/TEST.IMS_PER_BATCH) 均符合硬性规则。

## 结论

**审查通过**。单变量消融, 零代码改动, 严格对照 exp263d (Base Full s41) + Phase 3-C 跨 backbone 结论. OOM 风险已用 TEST.IMS_PER_BATCH=64 规避, 符合 2026-04-22 OOM 教训. CLI 正确引用 Base 配置文件, 对照组定义完整 (exp263d 同 seed + exp287/289 跨 backbone). 可直接启动, 无需 codex review 前置修复 (但仍需走 codex 审查流程通过 hook).
