# exp388 监控：Market-1501 官方干净 TAPF D0

## 当前状态

- 状态：FORMAL MARKET D0 RUNNING
- 直接对照：exp384 official clean Market B0 e120=`91.6/96.3/98.7/99.2`
- exp387 clean Occ-Duke D0 已封板：`57.6/67.7/80.8/84.6`，相对 B0=`+0.2/+0.3/+0.2/−0.6`
- 4090：唯一正式 Market D0 运行中
- 正式 Market D0：fresh 启动，main PID=`1051663`

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

## Market 最小泛化实现

- 本地提交：`879760e`；远端执行提交：`5bbbe4e64815a1b10b469ccfd4a20cac4675da67`；
- 代码变量只有：clean TAPF dataloader 白名单加入 `market1501`，并新增 matched Market D0 config；anchor、renderer、PSG、handoff、loss 与 exp387 完全相同；
- D0 config SHA256=`81abd0d4247c26bdb306f54be0e9c9d1c8a595a64e85c30e40bd606a86b2cc80`；
- patch SHA256=`b11407209a1e438a67ee18584d38e24edafdf32f4655209abfce42229f246146`；
- clean data unit 5/5 PASS；真实 artifact strict load 12,936 条，query/gallery lookup 明确拒绝；Market 原始 RGB 在 32 个固定 seed 下与官方 transform 逐 tensor bit exact。

## D0 启动前全门禁

### config-off 与构造不变量

- pre-TAPF clean commit=`d4fa227b30f7ea9a7c97973854323637d9fc8126` 与新提交在 Market B0 config 下运行同一 10-step CUDA/AMP 指纹脚本；
- 两份完整 JSON SHA256 均为 `dbc7c5964cd118d1a3469346ec8af01fa54ee39956b0b6f45bab5b718f420c4a`，211 state、构造后 CPU/CUDA RNG、optimizer groups、10 次 loss/output/173 gradients、最终 state、173 momentum 与 GradScaler scale=512 全部 exact；
- B0/D0 公共 state、构造后 CPU/CUDA RNG 与 179 个公共 optimizer 参数顺序/超参数 exact；12 个 TAPF parameter tensor 全部且只出现一次；
- 参数：B0=`28,111,674`，D0=`28,217,116`，新增 `105,442 / 0.375083%`；两个 PSG bank 参数独立、末投影 zero-init。

### 数据、CUDA/AMP 与因果路径

- 真实 Market batch64/8 workers 连续 24 step：默认 GradScaler `65536→4096`，4 次可恢复 overflow，随后连续 20 次有限更新；最终 optimizer 185 state 全有限；
- changed parameter tensor：Swin `171/193`、anchor `8/8`、PSG `4/4`、head `2/3`；峰值 allocated/reserved=`6,489,576,960/6,796,869,632` bytes；
- e1/e6/e10/e11 student fraction=`0/0.2/1/1`，两个独立 PSG bank 每次各消费一次；
- pose loss 只更新 anchor `8/8`，Swin/PSG/head 精确为零或无梯度；ReID loss 更新 Swin/PSG/head，anchor 精确为零或无梯度；
- 人为 nonfinite 后 `found_inf=1`、scale=`1→0.5`，208 个 model parameter tensor 与 185 项 optimizer state 整步逐元素 exact skip；
- strict roundtrip 223 state，missing/unexpected=`0/0`，correct/shuffle/None/exploding pose 的 descriptor/student field/两个 gate delta 逐元素 exact；query/gallery 仍为 RGB `ImageDataset`。

关键证据 SHA256：CUDA24 JSON=`c6d2414c884e88613d86924a81ac244b6406c00da04357280e6628aa470aa606`；semantics JSON=`4d021675505dd923dd41d1bdb7cc633b6f299fa360e4334cb60928a90f0797c7`；roundtrip checkpoint=`76a31c52aebb1956b85ec248a358fe5f356b4745c1eae57e2a9bde361f1c6662`；data/RGB parity runner=`21aadecaee9e8cfc5a48d6108bc9a57a038dff7be07505d36abb8d3ded00dedd`。

### Matched efficiency

| 项目 | Market B0 | Market D0 | D0−B0 |
|---|---:|---:|---:|
| 参数 | 28,111,674 | 28,217,116 | +105,442 / +0.375083% |
| supported-op FLOPs / image | 5,535,368,448 | 5,548,787,520 | +13,419,072 / +0.242424% |
| train batch64 mean step | 101.003 ms | 102.775 ms | +1.771 ms |
| train peak allocated | 6,044,115,968 B | 6,186,716,160 B | +142,600,192 B |
| eval batch256 mean step | 225.232 ms | 228.983 ms | +3.751 ms |
| eval peak allocated | 4,725,344,768 B | 4,725,308,928 B | −35,840 B（分配测量噪声） |

效率 JSON SHA256=`54113a19da0a8e5157b76a8ad2df336819bd5adf8c534ba64d51100bf04464d6`。FLOPs 只报告 analyzer 支持算子，两臂未支持的 elementwise/normalization 不伪装为完整理论 FLOPs；eval 两臂均显式 RGB-only。

## Fresh 正式执行门禁

- full-history bundle=`/home/afr/reid-clean/bundles/exp388_market_d0_5bbbe4e.bundle`；SHA256=`23df2f88afd0d1defb4b6b4aed1dfa5c53a5e45c81b3e69b7033d05d8cdec0fa`；
- planned formal repo=`/home/afr/SOLIDER-REID-exp388-d0-5bbbe4e`，detached exact HEAD=`5bbbe4e64815a1b10b469ccfd4a20cac4675da67`；
- fresh repo tracked clean，formal output 不存在，GPU=`2 MiB/0%`，fresh unit 5/5 PASS；
- 全部 Gate 结论由 `NO-START` 更新为 `GO`。下一步只允许以该 exact commit/config fresh 启动唯一 Market D0，并自然运行至 e120。

## 正式 D0 启动

- formal repo=`/home/afr/SOLIDER-REID-exp388-d0-5bbbe4e`；
- exact HEAD=`5bbbe4e64815a1b10b469ccfd4a20cac4675da67`；
- config SHA256=`81abd0d4247c26bdb306f54be0e9c9d1c8a595a64e85c30e40bd606a86b2cc80`；
- output=`log/market1501/exp388_clean_swin_tiny_d0_s1234`；
- runner=`/home/afr/train-logs/exp388_clean_market_d0_s1234.runner.log`；
- main PID=`1051663`；环境=`/usr/local/anaconda3/envs/mmpose-abu/bin/python`；
- 启动前 detached exact HEAD、tracked source clean、formal output/runner 不存在、GPU=`2 MiB/0%`；fresh unit 5/5 PASS；
- recipe 固定为 Market official B0 matched 的 batch64/seed1234/120epoch/SGD/lr0.0008/semantic weight0.2/eval10/checkpoint120；
- 首检唯一 main+8 workers，GPU 约 `6,994 MiB/92%`；e1 已到 iter160/186，`Loss=11.948`、`Pose=0.904`、`Student=0`、`Reliability=0.837`、`GateAbs=8.784e-05`；
- exact HEAD/config 与 tracked source clean，严格异常命中为 0，当前仅有 train log、无 checkpoint。

该 arm 必须自然跑满 e120，不改运行中代码/config，不续训、不重复；每次完整 eval 相对 exp384 Market B0 同 epoch 显式计算 mAP/R1/R5/R10 四项差值，只更新并提交本 monitor。

## 正式训练阶段评测（e10–e80）

| Epoch | exp388 D0 mAP / R1 / R5 / R10 | exp384 B0 同 epoch | D0−B0 |
|---:|---:|---:|---:|
| 10 | 80.2 / 91.6 / 97.1 / 98.3 | 78.4 / 90.8 / 96.9 / 97.9 | +1.8 / +0.8 / +0.2 / +0.4 |
| 20 | 83.3 / 93.4 / 97.8 / 98.6 | 82.2 / 92.4 / 97.4 / 98.3 | +1.1 / +1.0 / +0.4 / +0.3 |
| 30 | 88.1 / 94.8 / 98.4 / 99.1 | 87.0 / 94.3 / 98.0 / 98.8 | +1.1 / +0.5 / +0.4 / +0.3 |
| 40 | 89.4 / 95.4 / 98.3 / 98.8 | 88.9 / 95.4 / 98.5 / 99.0 | +0.5 / +0.0 / −0.2 / −0.2 |
| 50 | 90.4 / 96.3 / 98.8 / 99.1 | 89.8 / 95.5 / 98.8 / 99.3 | +0.6 / +0.8 / +0.0 / −0.2 |
| 60 | 90.7 / 95.9 / 98.5 / 99.2 | 90.2 / 95.8 / 98.7 / 99.2 | +0.5 / +0.1 / −0.2 / +0.0 |
| 70 | 91.2 / 96.2 / 98.7 / 99.3 | 90.8 / 96.1 / 98.6 / 99.2 | +0.4 / +0.1 / +0.1 / +0.1 |
| 80 | 91.7 / 96.5 / 98.8 / 99.3 | 91.3 / 96.1 / 98.8 / 99.2 | +0.4 / +0.4 / +0.0 / +0.1 |

- e10 末：`Pose=0.784`、`Student=1.00`、`GateAbs=5.300e-03`；
- e20 末：`Pose=0.602`、`Student=1.00`、`GateAbs=1.516e-02`；
- e30 末：`Pose=0.521`、`Student=1.00`、`GateAbs=2.066e-02`；
- e40 末：`Pose=0.499`、`Student=1.00`、`GateAbs=2.241e-02`；
- e50 末：`Pose=0.490`、`Student=1.00`、`Reliability=0.854`、`GateAbs=2.332e-02`；
- e60 末：`Pose=0.486`、`Student=1.00`、`Reliability=0.855`、`GateAbs=2.378e-02`；
- e70 末：`Pose=0.484`、`Student=1.00`、`Reliability=0.836`、`GateAbs=2.402e-02`；
- e80 末：`Pose=0.484`、`Student=1.00`、`Reliability=0.835`、`GateAbs=2.420e-02`。

截至 e80，八次完整评测均正常完成；e60/e70/e80 的同 epoch mAP 差值依次为 `+0.5/+0.4/+0.4`，e80 的 R1/R5/R10 差值为 `+0.4/+0.0/+0.1`。这些仅是中途轨迹，不作单点裁决，正式 arm 继续自然运行到 e120。期间 exact HEAD/config 保持不变，唯一 main+8 workers，未生成早期 checkpoint，AMP/NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow 严格异常命中为 0。
