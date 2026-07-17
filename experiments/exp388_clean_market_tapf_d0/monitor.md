# exp388 监控：Market-1501 官方干净 TAPF D0

## 当前状态

- 状态：POSE EXTRACTION PREFLIGHT
- 直接对照：exp384 official clean Market B0 e120=`91.6/96.3/98.7/99.2`
- exp387 clean Occ-Duke D0 已封板：`57.6/67.7/80.8/84.6`，相对 B0=`+0.2/+0.3/+0.2/−0.6`
- 4090：exp387 已退出并终审通过，当前 GPU 空闲
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

正式提取前先执行独立 smoke，并再次确认输出不存在、唯一 GPU 工作与严格异常为 0。
