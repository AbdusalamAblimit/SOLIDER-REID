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
