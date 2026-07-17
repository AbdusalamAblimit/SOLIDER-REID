# exp388 监控：Market-1501 官方干净 TAPF D0

## 当前状态

- 状态：POSE ARTIFACT SEALED / D0 IMPLEMENTATION PREFLIGHT
- 直接对照：exp384 official clean Market B0 e120=`91.6/96.3/98.7/99.2`
- exp387 clean Occ-Duke D0 已封板：`57.6/67.7/80.8/84.6`，相对 B0=`+0.2/+0.3/+0.2/−0.6`
- 4090：Market pose extraction 已退出并终审通过，当前 GPU 空闲
- 正式 Market D0：尚未创建 output 或启动

## Market 原始数据审计

- `/mnt1/afrdata/market1501` 是指向 `/mnt1/afrdata/Market-1501-v15.09.15` 的 symlink；
- train/query/gallery=`12,936/3,368/19,732`，非 junk ID=`751/750/751`，camera=`1–6`；
- gallery 含 3,819 张标准 junk/distractor；三 split 文件名均合法、无重复，全部图像可解码且尺寸为 `64×128`；
- train manifest SHA256=`9e372e8ffd6f3e45ee8a0216defd185f5d57250f02cb150944ba499272c5466d`；
- query manifest SHA256=`c7b071922ca6b05f6e29ceb7ead76067adf5b6a3b58ea24ed7c1fc58e342b7e0`；
- gallery manifest SHA256=`8b45d37a44f0de151158413840220f2945ebb93c6aa81926b13aa34f834269e4`；
- `readme.txt` 的标准计数与现场一致，并明确 research only、禁止分发及商业用途；
- 数据树中的历史 `pose_data` 已发现但严格排除，不作为输入、fallback 或校验源。

结论：真实 RGB 数据与研究用途许可门禁通过。下一步只从 `bounding_box_train` 用 fresh ViTPose-H 提取新的 COCO-17 target；query/gallery 不生成 pose。

## Fresh teacher 固定项

- 环境：`/usr/local/anaconda3/envs/mmpose-abu`；
- config SHA256=`c4fee8723dc3ec74d9d57e75d9b22138480fe556c1f5278f319e9ae5b65b6e16`；
- weight SHA256=`e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b`；
- 预期 train dataset manifest SHA256=`9e372e8ffd6f3e45ee8a0216defd185f5d57250f02cb150944ba499272c5466d`；
- planned incomplete=`/mnt1/afrderived/exp388_market_vitpose_huge_train.incomplete`；
- planned final=`/mnt1/afrderived/exp388_market_vitpose_huge_train`。

## 提取器与 smoke 门禁

- 执行 repo：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`，exact HEAD=`0d1822a07dda8daac0210b68916035b1886d5d99`，tracked source clean；
- 通用 fresh 提取器 SHA256=`e57ae6fc21df7ac594774490fade69884e879b0d6324574b59cedddc24b83045`；
- full 与 `.incomplete` 输出在 smoke 前均不存在，GPU 无计算进程；
- smoke output=`/mnt1/afrderived/exp388_market_vitpose_huge_smoke16`；
- 16 records / 2 shards，keypoints=`16×17×2`、scores=`16×17`，全量 finite；
- manifest SHA256=`79beebf79e14feb08254702441d15318710c18018002bf73be58115e905570ff`；
- score min/mean/max=`0.177427/0.863472/1.006203`，保留 teacher 原值，不静默裁剪；
- 固定样本 `bounding_box_train/0002_c1s2_050846_02.jpg` 重新在线推理，keypoints/scores 与离线记录逐 bit exact；
- 唯一模型加载提示为 MMPose 官方 checkpoint 已知的 unexpected `backbone.cls_token`；无 NaN/Inf/Traceback/RuntimeError/OOM/nonfinite，退出后 GPU=`2 MiB/0%`。

结论：teacher provenance、Market 低分辨率输入 API、artifact schema/finite 与在线—离线等价 smoke 全部 PASS。允许 fresh 启动 12,936 张 train-only 全量提取；仍禁止处理 query/gallery 或并行 ReID 训练。

## 全量 train-only 提取启动

- 启动时间：exp387 终审封板、GPU 空闲后串行启动；
- 执行 repo/exact HEAD：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a` / `0d1822a07dda8daac0210b68916035b1886d5d99`；
- main PID=`1048801`；
- runner=`/home/afr/pose-logs/exp388_market_vitpose_huge_train.runner.log`；
- incomplete=`/mnt1/afrderived/exp388_market_vitpose_huge_train.incomplete`；
- final=`/mnt1/afrderived/exp388_market_vitpose_huge_train`；
- 输入严格限定 `bounding_box_train`，expected count=`12,936`，dataset/config/weight SHA 门禁全部启用，shard size=`256`；
- 启动前 final/incomplete/runner 均不存在，tracked source clean，GPU 无计算进程；
- 首检为唯一 pose 进程，GPU 约 `3,102 MiB / 93%`；已处理 300/12,936，约 33.90 image/s，已有 1 个完整 shard；
- 唯一加载提示仍为官方已知的 unexpected `backbone.cls_token`；NaN/Inf/Traceback/RuntimeError/OOM/nonfinite 严格命中为 0。

提取期间 4090 唯一工作为该 full extraction，未并行 ReID 训练或第二个 pose 进程；完成后先做了 12,936 一一覆盖、shard/records/manifest/RGB SHA/尺寸/finite 与随机在线重跑 exact 终审。

## 全量 pose artifact 完成与终审

- 完成：12,936/12,936；364.04 秒，35.53 image/s；原 PID=`1048801` 已退出，`.incomplete` 已原子切换为 final；
- final manifest SHA256=`cc297ce97325b042a18e0b20512f9fd24322b72300ad8d66c88c0773239f3134`；
- records manifest SHA256=`ddefe09234dbb7a51a5d3927ae26a1d675543e5814bb32da3b9f584010796798`；
- runner SHA256=`ae9e864bd3b5a0d18d79237c63191a0220abd7e5518d555f43c7da4405866195`；
- 51 shards：前 50 个各 256 条，末 shard 136 条；全部 shard SHA/count/schema、float32 shape 与 finite 独立重算 PASS；
- 12,936 条 relative path 唯一且与 train JPG 排序后一一 exact；逐图 RGB SHA、尺寸与 dataset manifest 全量重算 exact；
- query/gallery relative path 为 0，artifact 只覆盖 `bounding_box_train`；
- score min/mean/max=`0.004411/0.850207/1.075122`，低于 0.5 为 9,226/219,912；原图范围外 joint 为 1,954/219,912；均保留原始值并由 paired transform/renderer 显式处理；
- seed=388 随机抽取 8 张，用同一 fresh teacher 重跑，keypoints 与 scores 全部逐 bit exact；
- runner 严格异常命中为 0；终审后 pose/ReID 进程为 0，GPU=`2 MiB/0%`；
- 可执行结论：`EXP388_FULL_POSE_AUDIT_PASS`。

pose artifact 数据门禁已封板。下一步只允许把现有 clean strict loader 最小泛化到 `market1501` 并复跑全部数据、数值、因果与 pose-free 门禁；在全部 PASS 前仍不创建正式 D0 output。
