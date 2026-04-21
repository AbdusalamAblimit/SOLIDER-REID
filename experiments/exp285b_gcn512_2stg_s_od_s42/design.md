# exp285b_gcn512_2stg_s_od_s42 — Phase 3-B exp262 同设备复制 (lab4090 rerun)

## 变体说明

- **exp262** (≡ Phase 3-B exp285 cell "GCN512+2stg Small"): **73.8/83.1** FINAL @ 2026-04-19 09:59 srvA (5060Ti + pre flip-fix code)
- 2026-04-20 14:13 用 ckpt_120.pth + 新 fix code re-eval → **73.8/83.1 完全一致** (bug 是 no-op, 因 POSE_GCN_PER_PART=False)
- **但跨设备 (srvA 5060Ti → lab4090 4090) cudnn 非确定性可能 ~0.1-0.3 mAP**, 同设备 vs exp282/283/284 对照更严谨
- 用户指示: **论文审稿可能纠结这点, 加 exp285b 在 lab4090 上同设备 rerun**

## 本 exp 变量

- 相对 exp262 零变量: 完全相同 config (`prcv_best_small.yml` default: Swin-Small + GCN512 + 2-stage PSG + Full Scaffold + seed 42)
- 唯一差异: **跑在 lab4090 (4090 24GB) + 当前修好的 flip-test code** (vs exp262 srvA 5060Ti 16GB + 原 code)

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /home/afr/SOLIDER-REID/log/occluded_duke/exp285b_gcn512_2stg_s_od_s42
```

无其他 override (yml 默认已是 Full Scaffold GCN512 + 2-stage PSG)。

## 输出

- 机器: lab4090 (auto-chain from exp277b via daemon)
- 预计时长: ~3h20min (Small Full Scaffold lab4090 165s/epoch × 120)
- ETA: exp277b FINAL 后启动 → 预计 tmr **22:00+ CST** FINAL

## chain 位置

`exp282 (running) → exp283 → exp284 → exp277b seed 41 → exp285b (本)`

## 对照

| Exp ID | GCN_HIDDEN | PSG_STAGES | backbone | seed | 机器 | mAP/R1 |
|--------|-----------|-----------|----------|------|------|--------|
| exp282 | 256 | `[-1]` | Small | 42 | lab4090 | e100 73.6/83.9 (running) |
| exp283 | 256 | `[-2,-1]` | Small | 42 | lab4090 | pending |
| exp284 | 512 | `[-1]` | Small | 42 | lab4090 | pending |
| exp285 ≡ exp262 | 512 | `[-2,-1]` | Small | 42 | **srvA 5060Ti (old)** | 73.8/83.1 |
| **exp285b (本)** | **512** | **`[-2,-1]`** | **Small** | **42** | **lab4090 (same-device)** | **pending** |

## 预期

若 exp285b FINAL 和 exp262 差异 <0.3 mAP → 确认 exp262 数字可信,论文主表用 exp262 数字或平均
若差异 >0.5 mAP → 说明设备确实影响,论文主表用 exp285b 作可信对照

**关键**: exp285b 提供的是 **同设备公平对照**, exp282/283/284 vs exp285b 的 Δ 才是真正反映 GCN cap × PSG stage 效应。
