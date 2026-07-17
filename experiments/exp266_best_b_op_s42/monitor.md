# exp266 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA @ Occluded-PoseTrack-ReID

**第 3 (最后 1) 个 Base run** (原计划 local 3090,3090 挂,改 5060 Ti with_cp)

- 机器: srvC (i-2.gpushare.com:25551, 5060 Ti 16G)
- 启动: 2026-04-20 04:46:14 (auto-chained by queue_on_ckpt.sh daemon 34381 from exp265 → exp266,新 main PID 49593)
- Log: /hy-tmp/log/occluded_posetrack/exp266_best_b_op_s42/train_log.txt
- Config: configs/occluded_posetrack/prcv_best_base.yml (WITH_CP=True, PLBOA ON)

## 对照

- 同一 scaffold: exp264 Tiny OP = 76.7/85.1, exp265 Small OP = 78.4/86.2
- 期望 Base > Small,目标 ≥79/87 on Occ-PTrack
- KPR 在 Occ-PTrack 的 baseline 数字需从 KPR paper Table 补上用于对比

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R-1 | R-5 | R-10 | 备注 |
|-------|-----|-----|-----|------|------|
| 10 | 74.3 | 83.3 | 93.6 | 96.4 | 7h 完成 |
| 20 | 76.1 | 84.6 | 94.0 | 96.6 | ckpt saved |
| 30 | 77.4 | 85.5 | 94.5 | 96.8 | |
| 40 | 78.1 | 86.3 | 94.4 | 96.9 | ckpt saved |
| 50 | 78.5 | 86.3 | 94.7 | 97.1 | **peak mAP** |
| **60** | **78.4** | **86.2** | **94.5** | **97.0** | ckpt saved, **effective FINAL** |

## e60 → silent exit (2026-04-20 ~21:27 CST)

在 e70 完成 (21:27) 后 **进程 49593 silent 退出**,没有 Traceback/OOM/NaN/CUDA error。诊断:
- Memory: 458G free/515G total — 充足,**非 OOM**
- GPU: 16G free, 0% util — 空闲
- dmesg 权限拒绝 (无法查 OOM killer)
- log 结尾正常结束在 `Epoch 70 done. Time per epoch: 836.925`

**推测原因**: hy-tmp 算力平台维护/reboot 或外部因素 (无法确认)。

## 决策: 不重训,用 e60 作 effective FINAL

**理由**:
1. e60 **78.4 / 86.2** 和 exp265 Small FINAL 78.4 / 86.2 **完全持平** → Base 对 Small 在 Occ-PTrack 上 0 增益
2. 训练从 e50 (peak 78.5) 已开始 plateau,剩余 60 epoch 期望涨幅 ~0.1-0.3 mAP,不值 14h 重训
3. PRCV 主表用 Small (exp265 78.4/86.2) 已足够,Base 补充 benchmark 无实际超越
4. Deadline 2026-04-30 紧张,14h 重训挤占 Phase 3-B 宝贵 GPU 时间

## effective FINAL (to results.md)

- **mAP 78.4 / R-1 86.2 / R-5 94.5 / R-10 97.0** (e60 eval,用最后 1 次 eval 数字,同 exp263/exp269 OOM 处理模式)
- Ckpt: `transformer_60.pth` (406MB,完整)
- Peak mAP @ e50: 78.5/86.3 (可选备份引用)

## srvC 状态

- silent exit 后 GPU 空闲 21:27 至今
- 因 Occ-Duke 数据未同步到 srvC (exp266 用的是 Occ-PTrack),启动 Phase 3-B 需先 rsync Occ-Duke + pose_data 5.5GB 从 srvB
- 暂**作 failover 备用**,如 srvB/lab4090 chain 出故障可承接
