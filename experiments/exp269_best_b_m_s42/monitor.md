# exp269 monitor — Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) @ Market-1501

**第 2 个 Base run** (原计划 local 3090,3090 挂,改 5060 Ti with_cp)

- 机器: srvA (i-2.gpushare.com:29162, 5060 Ti 16G)
- 启动: 2026-04-20 00:40:16 (auto-chained by queue_on_ckpt.sh daemon 3901347 from exp268 → exp269,新 main PID 4170236)
- Log: /hy-tmp/log/market/exp269_best_b_m_s42/train_log.txt
- Config: configs/market/prcv_best_base.yml (WITH_CP=True, PLBOA OFF)

## 对照

- 旧协议 exp260b Base Market = 94.4 / 97.1 (本地 3090,无默认 flip-test)
- 新协议加 default flip-test 期望 +0.2~0.5,目标 ≥94.6
- 同期 Small exp268 (刚完成) = 94.3 / 97.3 → Base 应该 ≥ Small,目标 ≥94.5

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | loss 14.7 acc 0 | warmup |

## 自动化状态

- srvA 后续无 daemon 排队 (L0 queue_next + L1 queue_on_ckpt 均已完成使命)
- 下一个 chain 要人工补: Phase 3-A exp274/275/276/277 Small PSG stage 消融

## 预期 ETA

- Base with_cp ~10.7min/epoch,120 epoch ≈ 21h
- 预计 2026-04-20 ~22:00 完成 → srvA 空闲,届时起 Phase 3-A Small 4 runs

## 异常: e80 eval 期间 OOM-killed (2026-04-20 ~13:38)

- **事件**: e80 训练完成 + transformer_80.pth 成功保存(13:37:08),随后 e80 eval 启动时 GPU mem 飙到 ~13-14GB 触天花板,内核 SIGKILL,log 末尾干净无 Traceback(同 exp263 OOM 模式)
- **根因**: Base (88M params) + PLBOA OFF(少一项) 但仍有 default flip-test @ BS=256 的 eval peak,Market 数据集 3368 samples 理论够低,实际仍触 16GB 边缘
- **transformer_80.pth 完整**,e90/100/110/120 无 ckpt
- **决策**: 不重训,accept e80 as effective FINAL

## FINAL (effective e80) — 2026-04-20 13:xx srvA

通过 `scripts/eval_fliptest_maxsim.py` + fix (commit `f69b61c`) 独立 eval:

- **Global cosine + flip: 94.4 / 97.0**
- **MaxSim hybrid + flip: 94.5 / 97.1** ← 主结果
- ckpt: `/hy-tmp/log/market/exp269_best_b_m_s42/transformer_80.pth`

### 训练内部 eq_concat+flip 轨迹(broken flip,仅参考)

| Epoch | mAP | R-1 |
|-------|-----|-----|
| 30 | 92.7 | 96.5 |
| 40 | 93.5 | 96.8 |
| 50 | 93.9 | 97.1 |
| 60 | 94.1 | 97.0 |
| 70 | 94.4 | 97.1 |
| 80 | — (OOM) | — |

e70 已接近饱和(94.4),e80 fixed eval 显示基本持平。剩余 40 epoch 增益预计 ≤0.1-0.3 mAP。

### 对照

- **exp260b Base Market FINAL (旧协议,eq_concat,no flip)**: 94.4 / 97.1 → 新协议 e80 持平
- **exp260b + MaxSim+flip**: 94.7 / 97.2 → 新协议 e80 MaxSim 94.5/97.1,差 -0.2 (不完整训,差可接受)
- **exp268 Small Market FINAL (新协议)**: 94.3 / 97.3 → Base e80 +0.2 mAP,-0.2 R1 → 规模优势 marginal(Market 已饱和)
- **4090-M-PSG-Small-lr4 historical**: 93.9/96.9 → +0.6/+0.2 improvement

### 结论

- Swin-Base + 2-stage PSG + LGPA-D + GCN512 + OA-SD (PLBOA OFF) + default flip-test @ Market-1501 (e80 eff FINAL) = **94.4 / 97.0 (Global+flip)**, **94.5 / 97.1 (MaxSim+flip)**
- Market 在 Swin backbone 上已近性能上限,Base 相对 Small 优势有限

### 后续

- srvA 空闲 → 排 Phase 3-A Small (exp274-277) 或其他
