# exp370 PBSR kill-switch manifest

## 冻结公共设置

- dataset：Occluded-Duke
- backbone：Swin-Tiny
- input：384×128
- batch size：64
- optimizer：SGD，LR 8e-4，weight decay 1e-4
- schedule：120 epoch，20 epoch warmup，cosine
- seed：1234
- AMP 初始 scale：1024（B0/P0/P1/P4 公共；历史默认 65536 不适用于本矩阵）
- baseline descriptor：768-d global，before BN，L2 normalize by evaluator
- 禁用：PSG、LGPA、GCN、PPA、VCSR、CLIP、OA-SD、PLBOA、parallel aug、re-ranking、MaxSim、part concat
- pose：target-person heatmap，仅作为 PBSR route loss target
- 公共 config：`configs/occluded_duke/exp370_pbsr.yml`

manifest 状态：**FROZEN / 允许启动第一批 B0、P0**。任何参数变更都必须新建 execution，不得覆盖本矩阵。

## 第一批四臂

### B0：global baseline

```bash
python train.py --config_file configs/occluded_duke/exp370_pbsr.yml \
  MODEL.POSE_PBSR False \
  OUTPUT_DIR ./log/occluded_duke/exp370_b0_global_s1234
```

### P0：PBSR full

```bash
python train.py --config_file configs/occluded_duke/exp370_pbsr.yml \
  OUTPUT_DIR ./log/occluded_duke/exp370_p0_coupled_s1234
```

### P1：read-only / writeback off

```bash
python train.py --config_file configs/occluded_duke/exp370_pbsr.yml \
  MODEL.POSE_PBSR_WRITEBACK False \
  OUTPUT_DIR ./log/occluded_duke/exp370_p1_readonly_s1234
```

### P4：shuffled pose supervision

```bash
python train.py --config_file configs/occluded_duke/exp370_pbsr.yml \
  MODEL.POSE_PBSR_SUPERVISION shuffled \
  OUTPUT_DIR ./log/occluded_duke/exp370_p4_shuffled_s1234
```

## 启动顺序与早停

1. 先在单机做 P0 一批次 forward/backward smoke；
2. 3090/4090 并行 B0/P0；
3. 若 P0 到 ep30 仍落后 B0 超过 1.0 mAP，记录后停止 P0，不跑 P1/P4；
4. 若 P0 到 ep30 与 B0 差距在 ±1.0 内，继续到 ep60；
5. P0 最终未超过 B0，则 NO-GO；
6. P0 有明确正向，才并行 P1/P4；
7. P0>P1 且 P0>P4 后再补 independent-write 和 uniform。

## 结果记录要求

每次 eval 从原始 log 抽取：epoch、mAP、R1、R5、R10。每个 epoch 同时记录：

- `pbsr_route`
- `pbsr_alpha`
- `pbsr_entropy`
- `pbsr_bg`
- `pbsr_delta`

任何 NaN/Inf、route loss 不降、alpha 长期为 0、background share 塌缩，都先按机制失败处理，不通过改 batch size 或叠加增强挽救。
