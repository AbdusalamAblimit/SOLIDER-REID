# 实验 exp388：Market-1501 官方干净 TAPF D0 外部验证

## 动机

exp384 已在官方最后代码上复现 Market-1501 B0，e120=`91.6/96.3/98.7/99.2`。exp387 在 Occluded-Duke 的 fresh ViTPose-H target 与干净最小 TAPF 上得到 e120=`57.6/67.7/80.8/84.6`，相对 matched B0 为 `+0.2/+0.3/+0.2/−0.6`。该结果只显示小幅、混合的单 seed 信号，不能单独支持稳定普适提升；因此需要在同一官方骨干和训练 recipe 下进行跨数据集 matched B0/D0 验证。

## 核心假设

训练期 target-person pose 可在 Market-1501 上监督 Stage-2 RGB pose anchor，并由 Stage-3 两个独立 PSG consumer 调制身份特征；测试 query/gallery 不提供也不构造 pose，descriptor 严格 RGB-only。主比较只解释完整 `anchor+PSG` 原子方法的 `Market D0−Market B0`，不拆分包装 anchor、pose loss 或 PSG。

## 数据与许可边界

- 数据入口：`/mnt1/afrdata/market1501`，解析到 `/mnt1/afrdata/Market-1501-v15.09.15`；
- train/query/gallery：`12,936 / 3,368 / 19,732` 张 JPG；
- 非 junk ID：`751 / 750 / 751`，gallery 另含 3,819 张 junk/distractor；
- 六个 camera，全部图像尺寸为 `64×128`；文件名均符合标准协议且无重复；
- train/query/gallery 内容 manifest SHA256 分别为：
  - `9e372e8ffd6f3e45ee8a0216defd185f5d57250f02cb150944ba499272c5466d`
  - `c7b071922ca6b05f6e29ceb7ead76067adf5b6a3b58ea24ed7c1fc58e342b7e0`
  - `8b45d37a44f0de151158413840220f2945ebb93c6aa81926b13aa34f834269e4`
- 数据包 `readme.txt` 明确限定 research only，禁止分发和商业用途；本实验只在已有远端副本上做研究复现，不复制或发布原图。
- 原始数据树虽有历史 `pose_data`，本实验禁止读取、校验、复制或建立 fallback；它的存在不算 pose 可用性证据。

## Fresh pose targets

- 唯一 pose 输入图像：`bounding_box_train/*.jpg` 共 12,936 张；query/gallery 绝不提取 pose；
- 环境：`/usr/local/anaconda3/envs/mmpose-abu`；
- teacher：MMPose 1.3.2 官方 ViTPose-Huge COCO-17 top-down；
- fresh config SHA256=`c4fee8723dc3ec74d9d57e75d9b22138480fe556c1f5278f319e9ae5b65b6e16`；
- fresh weight SHA256=`e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b`；
- 提取器沿用已审计的通用 MMPose 公共 API 工具，不读取任何旧 pose 路径；输出采用 `.incomplete→final` 原子切换、256 条/分片、逐图 RGB SHA/尺寸、records manifest 与 provenance；
- final 计划为 `/mnt1/afrderived/exp388_market_vitpose_huge_train`。

## 单变量对照

| 项目 | exp384 Market B0 | exp388 Market D0 |
|---|---|---|
| 官方 Swin-Tiny、预训练 teacher | 相同 | 相同 |
| train/query/gallery、sampler | 相同 | 相同 |
| batch / seed / epoch | 64 / 1234 / 120 | 完全相同 |
| optimizer / LR / semantic weight | SGD / 0.0008 / 0.2 | 完全相同 |
| RGB 增强、ID/triplet、BNNeck | 相同 | RGB 随机状态 exact；仅同步 pose 几何 |
| 额外变量 | 无 | exp387 同款完整 Stage-2 anchor + Stage-3 两 PSG + pose loss |
| 测试输入 | RGB | 严格 RGB-only |

D0 保持 exp387 的 e1–5 teacher、e6–10 平滑 handoff、e11–120 student consumer 和持续 pose supervision，不针对 Market 临时调宽、调权或改 schedule。

## 启动前门禁

1. 全量 pose artifact 对 12,936 张 train 一一覆盖，shard/records/manifest/RGB SHA/尺寸/finite 全部通过；固定随机样本在线重跑与离线记录数值 exact。
2. strict pose loader 只接受本实验 final manifest；query/gallery lookup 必须失败且评测数据集不拥有 pose store。
3. paired resize、COCO 左右交换、pad/crop、越界 mask；Random Erasing 只改 RGB；pose-disabled Market RGB 路径做多 seed 官方 transform bit-exact parity。
4. config-off 与 Market B0 的 state/init/RNG/forward/loss/optimizer 多步 exact；D0 公共 state/RNG/optimizer exact。
5. 真实 batch64/8 workers CUDA/AMP、GradScaler 动态回退与连续有限更新；人为 nonfinite overflow 整步 exact skip。
6. e1/e6/e10/e11 full-model route、pose/ReID 梯度隔离、两个 PSG bank 独立且各消费一次。
7. strict save/load；correct/shuffle/None/exploding external pose 的 eval descriptor exact；参数/FLOPs/训练与评测效率记录完整。
8. fresh 独立正式 repo、exact commit/full bundle/config SHA、output 不存在、GPU 空闲；任一门禁失败不得启动 D0。

## 运行与裁决

- 正式 D0 必须自然跑满 e120；每 10 epoch 固定评测并相对 Market B0 同 epoch计算 mAP/R1/R5/R10 四项差值；
- 不因单 epoch、阈值或 best checkpoint 提前停止，不挑 best 代替 e120；
- final 只报告 e120 matched 差值；单 seed 结果只作为跨数据集外部验证，不自动上升为多 seed 稳定性结论。

## 风险与失败解释

1. Market 遮挡弱于 Occluded-Duke，pose privileged 信号可能接近饱和或无增益；负结果仍是跨数据集边界证据。
2. 低分辨率 64×128 crop 可能降低 ViTPose 局部关节质量；保留原始 soft confidence，不手工补点或静默换 teacher。
3. 若 pose loss 可学但 D0≤B0，说明 anchor 可预测不等于 PSG 改善检索，不允许复活旧运行代码救点。
4. 只有全部数据、因果、数值与 pose-free 门禁通过，D0−B0 才能解释为完整原子方法的 matched 变化。
