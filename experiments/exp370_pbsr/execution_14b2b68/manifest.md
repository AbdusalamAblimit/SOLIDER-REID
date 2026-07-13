# exp370 execution `14b2b68` 证据清单

## 源码与环境

- exact execution commit：`14b2b68`
- source archive SHA256：`188df5e7049ed5b9bc877bdbe7965e5853e4e653711da09496707d16970664ee`
- config SHA256：`d91ad58ffe527eb90d6aec7a5d9e29ff10860040b31ff69ba82f721084da1277`
- PBSR module SHA256：`e4f9f9afef839dc91db5a30a4d6628ddaf15ec41401ef02d3a44fb4062724da9`
- CUDA smoke SHA256：`3e9d3a37cf8fbee7665e7b868c0138fc5ddb0a290f1122852975139ee08237d6`
- batch size：64
- seed：1234
- AMP initial scale：1024
- torch / torchvision：两端均为 `2.4.1+cu121 / 0.19.1+cu121`

首轮 B0 在 RTX 3090、P0 在 RTX 4090 D。核心框架版本已对齐，但 Python 与部分非核心依赖版本并非逐项相同，因此首轮跨机差值只作为 screening；最终因果裁决必须补同机同 runtime B0 对照。

## 原始日志 SHA256

| 文件 | SHA256 |
|---|---|
| `b0_runner_stdout.log` | `45fe625e3ddf1ea58a0ca83fcff8c9d0fb3beafb3ba563fd74fb7073b0e39f1d` |
| `b0_train_log.txt` | `689ff9f9441c62bb12e4b5ef02019d143298d2c5968a953a6b1618a52d85b29e` |
| `p0_runner_stdout.log` | `16ad029e5aa5d22e5cd96a23cc1344834b455726f3ba59e7b558ddf4d2e8ff63` |
| `p0_train_log.txt` | `1801cc886cd31ccee5833a8ed669d7d3598c1d58544bae4985c2efd7fed13508` |

## 首轮 matched-epoch screening

| Epoch | B0 mAP / R1 / R5 / R10 | P0 mAP / R1 / R5 / R10 | P0-B0 mAP / R1 |
|---:|---:|---:|---:|
| 10 | 36.0 / 44.8 / 61.0 / 67.4 | 34.2 / 43.0 / 59.4 / 65.7 | -1.8 / -1.8 |
| 20 | 43.3 / 53.5 / 69.8 / 75.3 | 38.4 / 48.1 / 64.8 / 71.4 | -4.9 / -5.4 |
| 30 | 48.9 / 58.6 / 73.3 / 78.6 | 48.5 / 57.8 / 72.9 / 78.5 | -0.4 / -0.8 |
| 40 | 51.8 / 61.7 / 76.6 / 81.4 | 51.4 / 61.0 / 75.6 / 80.5 | -0.4 / -0.7 |
| 50 | 52.4 / 62.6 / 76.5 / 80.9 | 53.2 / 63.5 / 78.0 / 82.9 | +0.8 / +0.9 |
| 60 | 55.3 / 64.7 / 78.1 / 83.2 | 54.4 / 63.7 / 76.9 / 81.6 | -0.9 / -1.0 |

P0 等待 B0 时额外产生的非匹配诊断：epoch 70 `56.9/66.6/80.5/85.1`，epoch 80 `57.9/67.6/81.1/85.9`。不得拿它们与 B0 epoch 60 比较。

## 远程 checkpoint SHA256

### B0 / 3090

- epoch 20：`1807bf3ec8eb0d819186ee92b74c6ecca683f548fc8e80196d3bcb7ba672f9a3`
- epoch 40：`ae34df0f5ae4989021f83d338f6703b9adc7c878b71e63764a8599fc8156c77f`
- epoch 60：`1d4e65f9d467fea20a7a1c821c734dd50e9cbd700d97cd80ef4d079e1e781125`

### P0 / 4090

- epoch 20：`e8c0e866dad433c52a3a49faaa3e069b62c6d5051645e495e0f92b086a848d2a`
- epoch 40：`78359f0a0523d231bf5347ebe1e5b2226ebc2ca6a1fbe1b9b771bf2968448ca5`
- epoch 60：`47226c0098e3e399be745a95bd6a25d12694736bb2451b5f2788aa8f976856a4`
- epoch 80：`453e93914d660778f3223d0c5d0e7f01dad824107b3944aa68f5aed31f3d5843`

checkpoint 保留在各自隔离执行目录，不回传 Git。
