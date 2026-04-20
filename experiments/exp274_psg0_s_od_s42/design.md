# exp274_psg0_s_od_s42 — Phase 3-A: no-PSG Small baseline (Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Small
- Dataset: Occluded-Duke
- Seed: 42
- `POSE_ENABLED=False` → pure Swin-Small build_transformer path(绕开 dead import bug;等同 SOLIDER Small baseline)
- 其他 pose 模块自动不构造

## 核心假设

对标 Phase 3-A Tiny baseline (exp270 = 59.2/68.4)。Small 参数 ~2x 于 Tiny,预期 mAP ~65-67(historical 4090-OD Small baseline = 65.8,+ default flip 应 +1-2)。

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  MODEL.POSE_ENABLED False \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp274_psg0_s_od_s42
```

## 输出

- 机器: **lab4090** (原 srvA 挂后,2026-04-20 19:30 重新在 lab4090 启动)
- 预计时长: ~2.5-3h (4090 比 srvA 5060Ti 快 2x;Small 无 pose 模块)
- 120 epoch 预计 2026-04-20 ~22:30 完成
- Auto-chain: 完成后 → exp275 1-stage PSG
- 环境: `/usr/local/anaconda3/envs/mmpose-abu/bin/python`
