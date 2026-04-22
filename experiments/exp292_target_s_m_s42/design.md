# exp292_target_s_m_s42 — target-heatmap on Market-1501 (lab3090)

## 动机

完成 target-heatmap 机制三数据集对照 (exp290 OP, exp291 OD, **exp292 Market**)。Market 几乎全 single-person, 预期 target-heatmap = scene-heatmap (严格 no-op), 用于 **证明机制对 single-person 数据集零回归**。

## 核心假设

Market 上 target-heatmap 严格等价 scene-heatmap:
- Market `num_persons = 1` 占 >99% (刚 Index.json 检查确认)
- `heatmaps[:, 0]` == `max(heatmaps, dim=1)` when only person 0 exists
- 训练轨迹应与 exp268 (94.3/97.3) 基本一致 (方差 ≤ 0.2 mAP)

## 技术方案

代码零改动, reuse exp290 合入的 `POSE_USE_TARGET_HEATMAP` flag。详见 `experiments/exp290_target_s_op_s42/design.md + claude_review.md + codex_review.md` (all 审查通过)。

### 数据兼容性验证 (已确认)

- Market pose_data on lab3090 (`/root/work/SOLIDER-REID/data/market1501/pose_data`, 4.3GB, 46635 npz):
  - ✓ npz has: heatmap, keypoints, scores, bbox, crop_bounds
  - ⚠️ 缺 visibility → `pose_dataset.py` fallback 到 clipped scores (backward compat)
  - ⚠️ 缺 target_person_idx → `pose_dataset.py` default 0 (Market 单人场景 target 就是 person 0, 正确)
- **不需要重新 extract**, legacy data 兼容 target-heatmap 机制

### 实验配置

- Backbone: Swin-Small (Full Scaffold)
- Dataset: Market-1501
- Config: `configs/market/prcv_best_small.yml` + `MODEL.POSE_USE_TARGET_HEATMAP True`
- Seed: 42 (对齐 exp268 FINAL 94.3/97.3)
- Epochs: 120
- 机器: **lab3090 (24GB 3090, solider-reid env, Market 数据本地)**
- FINAL ETA: ~5-6h (Small Market ~2-3 min/epoch on 3090)

## 对照组

- exp268 FINAL 94.3/97.3 (scene-heatmap default, Swin-Small Market)
- 预期 exp292 ≈ exp268 (方差 ≤ 0.2 mAP)
- Supplementary: exp267 Tiny 92.5/96.4 (不同 backbone 对照)

## 预期结果

| 情景 | mAP / R1 |
|------|----------|
| 严格持平 exp268 (预期, Market 单人) | 94.1-94.5 / 97.1-97.5 |
| 意外回归 (-0.5+) | < 93.8 → 机制有 bug, 回退 |
| 意外提升 (+0.3+) | > 94.6 → Market 部分多人样本, 机制也有效 |

## 风险

- 代码 scp 到 lab3090 (GitHub 连接超时), 不经 git push → 需确认 scp 后文件无损
- lab3090 之前跑 exp266b_3090 + exp263d + 早期 exp260b, 使用 solider-reid env, 兼容性已验证 (Phase 1 同 env)
- 数据兼容性 (legacy npz 缺 visibility) 已通过代码层 fallback 处理
