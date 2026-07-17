# exp384 监控记录

## 2026-07-18 干净起点

- 旧工作树完整快照提交：`529c35e`
- 新分支基于官方提交 `8c08e1c3255e8e1e51e006bf189e52cc57b009ed`
- 研究治理与实验档案同步提交：`f0793cc`
- 工作树隔离提交：`28503da`
- 远端 fresh repo HEAD：`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`
- 4090：空闲

## 数据门禁

原始 Market1501 位于 `/mnt1/afrdata/market1501`。官方 loader 现场统计：

- train：12936 张，751 IDs
- query：3368 张，750 IDs
- gallery：15913 张有效图像，751 IDs

磁盘上的 gallery 共 19732 张 JPG，差额为官方 loader 按 `pid == -1` 排除的 junk，统计一致。旧 `/home/afr/SOLIDER-REID/data/market/pose_data` 不属于原始数据，明确禁止使用。

## 运行时门禁

正式环境固定为 `/usr/local/anaconda3/envs/mmpose-abu/bin/python`：

- Python：3.8.20
- PyTorch：1.13.1 / CUDA 11.7
- torchvision：0.14.1
- MMCV：2.1.0
- MMPose：1.3.2
- MMEngine：0.10.7
- timm：1.0.22
- GPU：NVIDIA GeForce RTX 4090 D

官方代码首次 import 在 `torch._six.container_abcs` 处失败。该接口已在新 PyTorch 删除；已加入兼容 fallback，未改变模型计算。完成权重逐张量校验和 CUDA forward 后再进入短程训练。

第二次 import 在未使用的 `mmcv.runner.load_checkpoint` 处失败。现场静态搜索确认 `_load_checkpoint` 除导入外没有任何引用；已删除该死 import，避免为无效依赖降级或覆盖 MMCV，模型计算仍不变。

## 权重与 CUDA forward 门禁

- 完整官方 Swin-T checkpoint：`/home/afr/SOLIDER-REID/pretrained/swin_tiny.pth`
  - SHA256：`c4a3cbd1eaf9ae2bb9ed7e01c628400e689314be628a3c6fe8ed45ccd9f3b71b`
- converted teacher：`/home/afr/reid-clean/weights/solider_swin_tiny_tea.pth`
  - SHA256：`8bf35b39e6042929383782e0190884ef69fa68abae8437c78c885ade584b404b`
- 两者 teacher state dict：213/213 键一致，逐张量 exact equal，最大差值 0；其中 205 个键带 `backbone.` 前缀。
- 官方 `init_weights`：`All keys matched successfully`
- 总参数/可训练参数：28,111,674 / 28,098,234
- CUDA eval forward：descriptor `[1, 768]`，全部有限；峰值 allocated 130.78 MiB

在正式 mmpose-abu 环境中重新验证：

- batch-2 CUDA eval：descriptor `[2, 768]`，全部有限，峰值 allocated 144.13 MiB
- batch-4 CUDA AMP train step：loss 6.625，梯度全部有限，GradScaler scale 65536

结论：数据、官方权重和模型前向门禁通过。下一步运行官方 `train.py` 的单 epoch 全链路 smoke；该 smoke 仅检查训练/评测链路，不作为性能结果。

## 官方全链路 smoke

- 远端兼容执行提交：`b72ebf17b7731d52313effc96ed44b8055a76ecb`
- 官方 Swin-Tiny config SHA256：`8f810e0c62bae9a6bed0d4d471b39f91eb5a2bc500015cd01035358c8957ff0f`
- output：`log/market1501/official_swin_tiny_smoke_e1`
- 唯一训练进程 + 8 workers，batch64，186 iterations
- e1 训练时间：20.770 秒；速度：563.9 samples/s
- e1 eval：47.0 mAP / 71.5 R1 / 87.1 R5 / 91.4 R10
- 原进程与 workers 自然退出，GPU 释放，严格异常 0

产物 SHA256：

- runner stdout：`6cefcff27f3bbad9ba4a3d973cd948905177084b3ac14ed2f5c62ee4c64eb0e9`
- train log：`6f494ffaae68172dffc3673b895e52abfafd82e5298ae9e5d927f77311dc9b6b`
- e1 checkpoint：`5c55ed270061fa100ea30e716f9c8615ac245cdec68e8833f475ed8a262797fa`

该 smoke 把 `MAX_EPOCHS/EVAL_PERIOD/CHECKPOINT_PERIOD` 临时设为 1，只用于验证官方 train/eval/checkpoint 链路，不能与正式 120-epoch 性能比较。正式 B0 使用官方 config 原值与独立 output。

## Market1501 正式 B0

- output：`log/market1501/official_swin_tiny_b0_s1234`
- main PID：924146
- execution commit：`b72ebf17b7731d52313effc96ed44b8055a76ecb`
- 环境：mmpose-abu
- batch/seed/epoch：64 / 1234 / 120
- optimizer/base LR：SGD / 0.0008
- semantic weight：0.2
- eval/checkpoint period：10 / 120
- 启动前 GPU 空闲、output 不存在、tracked source clean

启动后检查：唯一 main + 8 workers；e1-e5 自然完成，e6 训练中；GPU 约 6.8 GiB；严格异常 0。e5 epoch 平均 loss 6.978，训练精度 0.256，未用于性能裁决。

### e10

- mAP / R1 / R5 / R10：78.4 / 90.8 / 96.9 / 97.9
- 首次正式完整 eval；仅作训练轨迹记录，不与官方 e120 指标直接裁决
- eval 后训练继续，未改代码、config 或进程

### e20

- mAP / R1 / R5 / R10：82.2 / 92.4 / 97.4 / 98.3
- 训练继续，严格异常 0

### e30

- mAP / R1 / R5 / R10：87.0 / 94.3 / 98.0 / 98.8
- e32 已自然完成；唯一 main + 8 workers，GPU 约 6.8 GiB，tracked source clean

### e40

- mAP / R1 / R5 / R10：88.9 / 95.4 / 98.5 / 99.0
- 评测后训练继续；唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean
- `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow` 严格异常 0

### e50

- mAP / R1 / R5 / R10：89.8 / 95.5 / 98.8 / 99.3
- 评测后自然进入 e51；未改代码、config 或进程，不作中途性能裁决
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean，严格异常 0

### e60

- mAP / R1 / R5 / R10：90.2 / 95.8 / 98.7 / 99.2
- 评测后自然进入 e61；未改代码、config 或进程，不以该中途点替代 e120
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean，严格异常 0

### e70

- mAP / R1 / R5 / R10：90.8 / 96.1 / 98.6 / 99.2
- 评测后自然进入 e71；R1 到达官方报告值不构成提前结束条件，继续运行至 e120
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean，严格异常 0

### e80

- mAP / R1 / R5 / R10：91.3 / 96.1 / 98.8 / 99.2
- 评测后自然进入 e81；mAP 距官方报告值 0.3 不构成提前结束条件，继续运行至 e120
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean，严格异常 0

### e90

- mAP / R1 / R5 / R10：91.4 / 96.3 / 98.7 / 99.2
- 评测后自然进入 e91；R1 略高于官方报告值、mAP 相差 0.2，均不改变跑满 e120 的门禁
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD 与 tracked source clean，严格异常 0

### e100

- mAP / R1 / R5 / R10：91.4 / 96.3 / 98.7 / 99.3
- 评测后自然进入 e101；相对 e90 仅 R10 增加 0.1，不选择中途 best，继续运行至 e120
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD/config SHA 与 tracked source clean，严格异常 0

### e110

- mAP / R1 / R5 / R10：91.6 / 96.4 / 98.7 / 99.2
- mAP 达到官方报告值、R1 高 0.3；评测后自然进入 e111，仍以 e120 final/checkpoint 封板
- 唯一 main + 8 workers，GPU 约 6.8 GiB，exact HEAD/config SHA 与 tracked source clean，严格异常 0

### e120 final

- mAP / R1 / R5 / R10：91.6 / 96.3 / 98.7 / 99.2
- 相对官方报告的 91.6 mAP / 96.1 R1：mAP exact，R1 +0.2；以 e120 而非中途 best 封板
- 原 main PID 924146 与 8 workers 自然退出；GPU 回到 2 MiB / 0%，未续训或重启
- 唯一 checkpoint：`transformer_120.pth`，112,770,499 bytes

产物 SHA256：

- e120 checkpoint：`7c968e729b20560143e8779186ced19ecd6f89a9e8edc169ef0b2ed60b37acf4`
- runner stdout：`ddc539b20d5d00fbdb9a66f32c9258777bbae3e8285197643e385a0170d98a69`
- train log：`ad660d4fc7a5a5e7b360edcd4ceaba084ac75dd6cdce4a8a39ad0b1732087ee2`

终审：

- execution HEAD：`b72ebf17b7731d52313effc96ed44b8055a76ecb`
- config SHA256：`8f810e0c62bae9a6bed0d4d471b39f91eb5a2bc500015cd01035358c8957ff0f`
- tracked source clean；严格 `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow` 与 AMP warning 扫描均为 0
- checkpoint strict load：0 missing / 0 unexpected；211 个 state tensor 全部有限
- CUDA AMP descriptor：`[1, 768]`，descriptor 与全部 featmaps 有限，峰值 allocated 168.5 MiB；检查后 GPU 释放

结论：官方干净 Market1501 Swin-Tiny B0 全量复现通过，禁止重启、续训或重复该 arm。下一步仅在同一干净代码上审计并建立 Occluded-Duke B0。

## 官方 `sw` / `with_cp` 并行审计

CPU 单测确认：`sw` 存在设备硬编码和 terminal controller 死路径；`with_cp` 内核前向/梯度 exact parity，但官方 config/make_model 无法开启。证据与后续修复边界见 `official_sw_withcp_audit.md`。当前 B0 不改代码、不重启。
