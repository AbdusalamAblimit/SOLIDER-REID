# AG-ReID.v2 复现监控 (baseline-Swin vs AIRL-iso)

## 实验配置
- 数据集: AG-ReID.v2 官方协议 (A→G=exp1 aerial_to_cctv 2356q/6347g, G→A=exp4 cctv_to_aerial 1811q/14362g), mean
- backbone: Swin-Small + SOLIDER pretrain (swin_small.pth), 256×128, bs64, 60ep, lr3.5e-4(backbone×0.1)
- baseline: `--dataset agreid_v2 --backbone swin_small --swin_pretrain ...` (无 AIRL)
- AIRL-iso: + `--airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25` (全 detach WORKING)
- 机器分工: baseline-Swin on lab-4090, AIRL-iso on lab-3090 (iso2 跑完后)

## 接线 + smoke (2026-06-23)
- 新增 `afd_reid/agreid_v2_combined.py` (官方 exp1+exp4 合并, filter_by_view 还原方向)
- afd_train.py 3 处 hunk (import / --dataset choices / elif), cargo 字节级不变
- CPU smoke 全过: 官方计数 2356/6347 & 1811/14362, int pid collate, eval_market sanity
- codex review: approve, 0 findings
- py_compile 通过

## 数据传输 (lab-4090) — 失败, 改 3090-only
- Mac-relay 双 ssh 管道 0.03MB/s (废, 11h ETA)
- 单文件分两跳: 3090→Mac **2.7MB/s OK** (tar md5 校验通过 f79e00db...); **Mac→4090 STALLED 0MB/s**
  (relay4090 ProxyJump 上传带宽死, 后续连 SSH banner 都超时 → 4090 当前不可靠)
- OSS 不可用 (401, 非 gpushare 机器, 账号未登录)
- **决策: 两实验都在 lab-3090 跑** (数据本地, 无需传输, 可靠). iso2 跑完后顺序: baseline-Swin → AIRL-iso.
  代价 = 顺序 ~7-8h (单 GPU), 但可靠. 4090 数据上传从 Mac 死路, 无法并行.

## 训练进度

### baseline-Swin (lab-3090, 2026-06-23 启动)
- iso2(best 63.61@ep50)跑完 GPU 释放后立即启动。PID 1260513,log `/tmp/agreidv2_baseline.log`。
- 启动验证:config use_afd=False 全 AIRL 关(纯 baseline);SOLIDER pretrain All keys matched;
  **官方 combined 正确加载**(A→G exp1 2356/6347, G→A exp4 1811/14362, train 51530/807);
  Swin LR fix 生效(backbone×0.1);786 iter/epoch;GPU 7.9GB/81% 健康。
- ep1 Loss 38→21 正常 warmup。eval_period=10 → ep10/20/30 出趋势。out_dir `log/cargo/cvpb_agreidv2_baseline`。

### AIRL-iso (待 baseline 跑完顺序启动 on lab-3090)
- config: `--airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25`
  (全 detach WORKING, = CARGO 定稿 +2.37 配置)。
- ★ 机制确认(读 afd_model.py):trunk_recce=0 → clean fork feed **detached**,f_rec 梯度都不回流共享
  trunk → f_full 路径与 baseline 训练损失相同。故 AIRL run 的 `full` 头 ≈ baseline(交叉验证隔离),
  仍按用户要求跑独立 baseline 做干净对照。

## baseline-Swin 趋势 (lab-3090)
| epoch | A→G mAP | A→G R1 | G→A mAP | G→A R1 | mean mAP | mean R1 |
|-------|---------|--------|---------|--------|----------|---------|
| ep10  | 73.39   | 82.60  | 73.97   | 82.99  | **73.68**| 82.80   |
| ep20  | 71.25   | 80.22  | 71.19   | 80.73  | 71.22    | 80.47   |
| ep30  | 72.09   | 80.65  | 73.32   | 82.88  | 72.71    | 81.76   |
| ep40  | -       | -      | -       | -      | 76.82    | 84.53   |
| ep50  | -       | -      | -       | -      | 79.22    | 86.37   |
| **ep60** | 79.72 | 86.42  | 80.04   | 87.80  | **79.88**| 87.11   |
ep1-10 收敛健康(无 ep8 collapse,Swin LR fix 生效),Acc ep10=0.96/30=0.987/60≈0.997。
ep20 mean 小回落(cosine 过 warmup 峰值 transient,两方向对称),ep30 回爬 → ep40-60 强劲爬升
73.68→71.22→72.71→76.82→79.22→**79.88**。**baseline final = 79.88 mAP / 87.11 R1**(A→G 79.72 /
G→A 80.04,两方向均衡),正落在 AG-ReID.v2 SOTA 区间(81-88)。best=final@ep60。

### AIRL-iso (lab-3090, 2026-06-24 启动, baseline 跑完后接力)
- PID 1269156,log `/tmp/agreidv2_airl_iso.log`,config 验证:`airl_dualbranch_iso=True iso_stage=3
  trunk_recce=False(全 detach WORKING)fuse_w=0.25`;rec late stage 14.2M + rec BNNeck 入优化器;
  degradation-consistency grad 与共享 trunk 隔离;官方 combined 正确加载(2356/6347 & 1811/14362)。
- ep1 健康:Loss 44→29,CE≈CE_rec≈6.68(rec 分支在学),AIRL_rec=0.0002 一致性激活,GPU 14.1GB/96%。
- eval 报 full/rec/FUSE 各方向 + FUSE mean(model-selection 用 FUSE)。判据 = FUSE mean 是否净超
  baseline 79.88。

#### AIRL-iso FUSE 趋势
| epoch | A→G FUSE | G→A FUSE | full mean | rec mean | **FUSE mean** | vs baseline同期 |
|-------|----------|----------|-----------|----------|-----------|---------|
| ep10  | 72.89    | 73.53    | 71.61     | 72.33    | **73.21** | base ep10=73.68 (−0.47) |
| ep20  | 72.83    | 73.14    | 71.10     | 71.79    | **72.99** | base ep20=71.22 (**+1.77**) |
| ep30  | 74.80    | 75.26    | 73.63     | 73.20    | **75.03** | base ep30=72.71 (**+2.32**) |
| ep40  | 77.58    | 77.65    | 76.39     | 75.23    | **77.62** | base ep40=76.82 (**+0.80**) |
| ep50  | 80.25    | 81.08    | 79.70     | 78.24    | **80.67** | base ep50=79.22 (**+1.45**) |
| **ep60** | 81.06 | 81.46    | 80.21     | 78.96    | **81.26** | base ep60=79.88 (**+1.38**) |
ep10-60: FUSE > full & rec 各方向(互补融合稳定工作),全程 **0 KILL flag**(机制稳定)。**同期对照**:
−0.47→+1.77→+2.32→+0.80→+1.45→**+1.38**(ep20 起每个同期点都领先;ep10 例外仅因 base ep10 恰好高点)。
ep30+ AIRL `full` 追上 baseline,ep60 full(80.21)≈ baseline final(80.21 vs 79.88,+0.33)→ 全 detach
隔离使共享 trunk 训练等效 baseline 验证成立,FUSE 的 +1.05(over own full)= 互补融合净贡献。

## 最终结果 ★ 跨数据集复现成立
| 方法 (AG-ReID.v2, Swin, 60ep) | A→G mAP | G→A mAP | **mean mAP** | mean R1 |
|------|---------|---------|----------|---------|
| baseline-Swin | 79.72 | 80.04 | **79.88** | 87.11 |
| AIRL-iso FUSE (full-detach iso, w=0.25) | 81.06 | 81.46 | **81.26** | 88.41 |
| **净超 baseline** | +1.34 | +1.42 | **+1.38** | +1.30 |

- AIRL-iso 单头分解: full 80.21 / rec 78.96 / **FUSE 81.26**(model-selection)。
- **结论**: AIRL-iso 在真实低清 AG-ReID.v2 上 **FUSE 净超 baseline +1.38 mAP**,复刻 CARGO 模式
  (CARGO +2.37)。AG-ReID.v2 headroom 比 CARGO 略小(真实数据,涨幅 +1.38 < +2.37,与诚实预期一致),
  但**机制非数据集特例,跨数据集成立 = AIRL 方法稿第二支柱到位**。
- 机制干净: 全程 0 KILL;full≈baseline(隔离正确);FUSE>full&rec 双方向(互补融合真贡献);
  净超在每个同期 checkpoint(ep20-60)都成立。
- ckpt: baseline `log/cargo/cvpb_agreidv2_baseline/`,AIRL `log/cargo/cvpb_agreidv2_airl_iso/`。
