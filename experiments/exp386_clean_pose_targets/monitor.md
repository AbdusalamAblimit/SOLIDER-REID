# exp386 监控记录

## 起点

- exp385 clean Occluded-Duke B0 已封板：`57.4/67.4/80.6/85.2`
- 4090 空闲：2 MiB / 0%
- mmpose-abu API 可用：`init_model`、`inference_topdown`
- installed ViTPose-Huge config SHA256：`72fcd88a4483742869867a1da2aa6e2af533155950185e524bf4ed24e7c15d36`
- 官方权重 URL 已从 MMPose 1.3.2 model index 现场核对
- 旧仓库同名模型文件与旧 `pose_data` 均不复用

下一步：fresh 下载官方 checkpoint 到新目录并校验，再实现最小提取器；尚未启动 pose 提取或 ReID 训练。

## Fresh 官方 teacher

- 新目录：`/home/afr/pose-models/exp386_vitpose_huge_coco`
- 权重大小：2,548,954,167 bytes
- 权重 SHA256：`e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b`
- fresh dumped config SHA256：`c4fee8723dc3ec74d9d57e75d9b22138480fe556c1f5278f319e9ae5b65b6e16`
- 下载工具：MIM 0.3.9；来源为 MMPose 1.3.2 官方 model index URL
- 加载时唯一 unexpected key：`backbone.cls_token`；MMPose 官方模型可正常推理，其余 state 加载完成

未复制、链接或读取旧仓库中的同名 teacher 文件。

## API 抽样门禁

4 张分布抽样原始 crop 均返回：

- 恰好 1 个 pose；
- keypoints shape=`[1,17,2]`；scores shape=`[1,17]`；
- 坐标与 confidence 全有限；
- 单模型峰值 allocated 2,540.15 MiB；检查后 GPU 回到 2 MiB / 0%。

## 提取器实现与 smoke

- 本地实现提交：`0aca14c`
- 远端执行提交：`b08c2c2`
- 远端脚本 SHA256：`e57ae6fc21df7ac594774490fade69884e879b0d6324574b59cedddc24b83045`
- 特性：严格输入边界、config/weight/dataset SHA 门禁、原子 incomplete→final 目录切换、分片 NPZ、逐图 RGB SHA、全链 records SHA 与统计 manifest

16 张 fresh smoke：

- output：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_smoke16`
- 两个 shard，各 8 张；schema/count/shape/finite 全部 PASS
- manifest SHA256：`c4856f8a4ad1d5d4b17ec9b7ca2d8e807a3ae2dde272899b5e54fec5c4ba841e`
- records SHA256：`3be2e2faf60b9f994377e2248d379de9fa223aa887bf57671483922493d9214e`
- score min/mean/max=`0.0741/0.8907/0.9836`；低于 0.5 的 joint 为 3/272
- out-of-bounds joint：1/272；保留原始预测，后续 paired transform 决定有效 mask
- 峰值 allocated 2,540.15 MiB；退出后 GPU 空闲

结论：teacher provenance、API 与提取器 smoke 门禁通过。下一步 fresh 启动全量 15,618 张 train-only 提取；仍不处理 query/gallery，不启动 ReID 训练。
