# exp270_psg0_t_od_s42 — Phase 3-A: no-PSG baseline (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Tiny
- Dataset: Occluded-Duke
- Seed: 42
- PSG: **关闭** (本 run 的核心变量)
- LGPA / GCN / OA-SD / PLBOA / Parallel-Aug: **全部关闭**(纯 baseline,排除其他模块干扰)
- TEST_FEAT: `global` (因为无 branch 融合)
- BS=64, LR=8e-4, 120 epoch, WARMUP=20 cosine

## 核心假设

去除所有 pose 模块,等同 SOLIDER baseline + Swin-Tiny。期望 mAP ~56/67 (对照 exp000 历史 baseline 56.6/66.5,新协议 default flip-test 期望 +0.5-0.9 → ~57/67-68)。

## CLI 配置

初版配置(逐个关 POSE_* 开关)触发了 codebase 里的 dead import bug:  
`model/pose_model.py:10` 仍 `from .modules.pose_feature_modulation import PoseFeatureModulation`,但 `pose_feature_modulation.py` 在 commit `252eaa3` 代码清理时已删。Phase 1 所有 run `POSE_BACKBONE_PSG=True` 走 `pose_backbone_model.py` 路径,绕过死 import。exp270 `POSE_BACKBONE_PSG=False` 走 fallback `from .pose_model import PoseReIDModel` → `ModuleNotFoundError`。

**改用 `POSE_ENABLED=False`** 直接走 pure `build_transformer`(纯 Swin-Tiny + ID + triplet loss,零 pose branch,等同 SOLIDER baseline),彻底绕开 bug。最终命令:

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  MODEL.POSE_ENABLED False \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp270_psg0_t_od_s42
```

Phase 3-A 后续 exp271/272/273 `POSE_BACKBONE_PSG=True` 仅改 `POSE_PSG_STAGES`,不会触发该 bug。若将来需要其他无 PSG 变体,应先修 `pose_model.py` 的死 import 再开。

## 输出

- 机器: srvB (exp263 OOM 后空闲)
- Log: /hy-tmp/log/occluded_duke/exp270_psg0_t_od_s42/train_log.txt
- 预计时长: ~8h (Tiny 无 pose 模块,单 epoch ~4min)

## 对照 Phase 3-A 矩阵

| Exp | PSG stages | 期望 mAP |
|-----|-----------|---------|
| **exp270 (本)** | 无 | ~56/67 (baseline) |
| exp271 | `[-1]` | ~57-58 (+PSG stage 3) |
| exp272 | `[-2,-1]` | ~58-59 |
| exp273 | `[-3,-2,-1]` | 待定 |

用于回答 Phase 3-A 核心问题: **PSG 本体是否稳定有效**。
