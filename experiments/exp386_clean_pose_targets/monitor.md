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

## 全量 train-only 提取启动

- 启动前远端 exact HEAD：`b08c2c27c0527a414369520f537f35ad813fbfe5`
- tracked source 与 index：clean；GPU：无计算进程
- final 与 `.incomplete` 输出目录：启动前均不存在
- main PID：`1005655`
- runner：`/home/afr/pose-logs/exp386_vitpose_huge_train.runner.log`
- incomplete：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train.incomplete`
- final：`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`
- 输入严格限定为 `bounding_box_train`；expected count=`15618`
- dataset/config/checkpoint SHA 门禁均启用；shard size=`256`
- 启动后首检：单一 PID，GPU 约 3,102 MiB、利用率 91%；已处理 200/15,618，约 33.39 image/s
- 唯一模型加载提示仍为官方已知的 unexpected `backbone.cls_token`；首检无 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite

全量任务保持后台运行；依赖 15 分钟 heartbeat 监控，不阻塞等待，不并行 ReID 训练或第二条 pose 提取。

## 全量完成与终审

- 完成：15,618/15,618；438.4 s，35.62 image/s
- 原 PID `1005655` 已退出；`.incomplete` 已原子切换为 final
- GPU 回到 2 MiB / 0%；pose/ReID 训练进程均为 0
- runner SHA256：`669a92e858dde55a8ae8d1f2bed5270e1ed055037faae556561d7624046ff472`
- manifest SHA256：`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`
- records manifest SHA256：`cd3dc28661a06daa4cc7b2a30dc5d5ddac1475f144330fba75931365a68a43c7`
- 62 shards：前 61 个各 256 条，末 shard 2 条
- shard SHA/count/schema 全部 PASS；schema 固定为 relative path、RGB SHA、原图尺寸、17×2 keypoints、17 scores
- 15,618 条 relative path 与原始 train JPG 排序后一一 exact，unique=15,618
- 逐图 RGB SHA 与尺寸全量重算 exact；dataset manifest 重算仍为 `9be350a47c848844053c86a7f58e7f7a98b92c4940aaad9c18b80386e276f304`
- records manifest 从 NPZ 内容独立重算 exact；keypoints/scores 全量有限
- score min/mean/max=`0.01334639/0.85300736/1.06461477`；低于 0.5 为 10,013/265,506
- 原图范围外 joint 为 1,129/265,506；保留 teacher 原始输出，交给 paired transform 的显式有效 mask
- teacher score 不保证严格落在 `[0,1]`；不静默裁剪，后续可靠性变换必须在设计中显式定义
- runner 严格检索 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite：0

固定 seed=386 从全量记录抽取 8 张，重新加载同一 fresh teacher 并在线推理：

- 8/8 keypoints 逐 bit exact；全局 max abs diff=0
- 8/8 scores 逐 bit exact；全局 max abs diff=0
- 在线校验退出后 GPU 仍为 2 MiB / 0%，无遗留进程

结论：fresh train-only pose target 提取、可追溯性、全量结构与在线—离线等价门禁全部 PASS。query/gallery 从未生成 pose；下一步只实现 paired augmentation 与 pose target loader 门禁，暂不启动 D0。

## Clean pose data path 与 paired augmentation

- 本地实现提交：`749850d`
- 远端门禁提交：`d4fa227b30f7ea9a7c97973854323637d9fc8126`
- `pose_targets.py` SHA256：`42f6e35eff2ad572445143cb3ecc5b6a22d856facc4453b989411300dec22624`
- `paired_pose_transform.py` SHA256：`5a88021e80acc3e0a0ff45571cedcd13b7acf02f5e07570ec65be972eda191dd`
- `pose_dataset.py` SHA256：`876065939acf278265ab5a99572b7d148c723c26ec557f6da2f47f64e91aed6f`
- unit SHA256：`72a7ad6a30cf1a08e22bd7f18ceca33713e3e79e3555f36c0cfc7bfdd57114f2`

实现边界：

- loader 强制 manifest SHA、每个 shard SHA、records digest、COCO-17 schema/count/float32/finite/unique path；拒绝路径逃逸和任何 `pose_data` 路径；
- 每次 lookup 返回 tensor 副本，并可逐图重验 RGB SHA；query 路径没有 fallback，现场确认抛出 `KeyError`；
- resize/flip/COCO 左右交换/pad/crop 共用一组几何随机量；原始越界 joint 不会因 pad 重新变为有效；
- score 保持 teacher 原始数值，不裁剪；Random Erasing 位于归一化后且只改变 RGB；
- 新数据集与 collate 尚未接入官方默认 dataloader，B0 路径不受影响。

原生 torch1.13.1/torchvision unit：5/5 PASS，覆盖 manifest 与 shard 篡改拒绝、lookup 副本、resize、左右交换、pad/crop、越界 mask、Random Erasing pose 不变。pose-disabled RGB 路径在 32 个固定随机种子下与官方 Compose 逐 tensor bit exact。

真实 Occluded-Duke 门禁：

- final artifact 严格加载 15,618 条；query pose absent；
- 8 workers、batch64，连续 4 batches/256 张逐图 RGB SHA 校验；RGB=`64×3×384×128`、pose=`64×17×2`、scores=`64×17`；
- RGB/keypoints/scores CPU→CUDA 全有限，crop 后 valid 与输出边界一致；4 batches 有效 joints=4,265/4,352；
- 最终代码复检 batch64/8 workers/CUDA PASS，有效 joints=1,050/1,088；退出后 workers=0、GPU=2 MiB/1%；
- 远端 tracked source/index clean，无 ReID 或 pose 进程。

结论：exp386 的提取、严格 loader、paired augmentation、RGB parity 与真实 DataLoader/CUDA 门禁全部 PASS。下一步另写 TAPF D0 单变量设计和 config；在模型 route/gradient/AMP/overflow/state-RNG-optimizer/eval pose-free parity 全部通过前不启动正式训练。
