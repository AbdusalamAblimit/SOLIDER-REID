# exp277b_psg3_s_od_s41 — Phase 3-A Small 3-stage PSG 重跑 seed 41

## 变体说明

**exp277 (seed 42) 塌缩**: e2 `id_global=3.277=0.5×ln(702)` classifier 均匀输出,e120 FINAL 49.0/57.7 (vs exp274 no-PSG 68.1/76.8 Δ=-19.1)。

用户判断: **偶发随机性问题**(之前类似情况出现过), 换 seed 41 重跑验证。exp277b 替代 exp277 作为 Phase 3-A Small 3-stage 的 PRCV 主表数字。

## 本 exp 变量

- 相对 exp277 (seed 42) 严格单变量: `SOLVER.SEED` 42 → 41
- 其他参数不变: Small backbone, PSG 3-stage `[-3,-2,-1]`, LGPA/GCN/OA-SD/ParAug/LOWER_BODY_OCC 全关 (pure PSG)

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 41 \
  OUTPUT_DIR /home/afr/SOLIDER-REID/log/occluded_duke/exp277b_psg3_s_od_s41 \
  MODEL.POSE_BACKBONE_PSG True MODEL.POSE_PSG_STAGES "[-3,-2,-1]" \
  MODEL.POSE_LGPA False MODEL.POSE_SKELETON_GCN False MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT global
```

## 输出

- 机器: lab4090 (auto-chain from exp284 via daemon)
- 预计时长: ~1h50min (Small pure PSG 54s/epoch × 120)
- ETA: 待 exp284 FINAL (预计 tmr 10:00) 后启动 → tmr 11:50 CST FINAL

## 对照

| Exp | PSG stages | seed | FINAL mAP/R1 |
|-----|-----------|------|--------------|
| exp274 | 无 | 42 | 68.1 / 76.8 |
| exp275 | `[-1]` | 42 | 68.8 / 76.8 |
| exp276 | `[-2,-1]` | 42 | 68.3 / 77.2 |
| exp277 | `[-3,-2,-1]` | 42 | **49.0 / 57.7 (塌缩, seed 问题)** |
| **exp277b (本)** | `[-3,-2,-1]` | **41** | pending |

**预期**: 若 seed 41 正常,FINAL 在 68-69 / 76-77 范围(接近 exp275/276),进一步验证 "Tiny 3-stage ≈ 2-stage" pattern 在 Small 上复现。
