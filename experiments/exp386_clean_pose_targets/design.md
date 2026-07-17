# 实验 exp386：从原始 RGB 重建 Occluded-Duke 姿态 targets

## 动机

exp385 已在官方 SOLIDER 干净代码上完成 Occluded-Duke B0，final=`57.4/67.4/80.6/85.2`。后续 TAPF 不能读取旧 `pose_data`、旧 cache 或旧路径映射，因此先把姿态提取独立成可核验的数据构建实验，再进入模型实现。

## 核心假设

使用 MMPose 1.3.2 的官方 ViTPose-Huge COCO-17 top-down 模型，对 Occluded-Duke 原始训练图像重新推理，可以得到与原图内容、官方 config 和官方 checkpoint 一一绑定的紧凑 pose targets。测试 query/gallery 不提取 pose，从数据层保证后续 D0 推理只能读取 RGB。

## Pose teacher

- 环境：`/usr/local/anaconda3/envs/mmpose-abu`
- MMPose/MMCV/MMEngine：1.3.2 / 2.1.0 / 0.10.7
- 模型：`td-hm_ViTPose-huge_8xb64-210e_coco-256x192`
- 关键点：COCO 17 joints
- 官方模型索引 COCO AP：0.788
- installed config SHA256：`72fcd88a4483742869867a1da2aa6e2af533155950185e524bf4ed24e7c15d36`
- 官方权重 URL：`https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth`

虽然旧仓库目录中存在同名 ViTPose 文件，本实验明确不复制、不链接、不读取它。config 与 checkpoint 均从当前 mmpose-abu 官方包/官方 URL 重新取得，并记录 fresh 文件 SHA256。

## 输入边界

- 唯一图像输入：`/mnt1/afrdata/Occluded_Duke/bounding_box_train/*.jpg`
- 样本数：15,618
- 输入清单 SHA256：`9be350a47c848844053c86a7f58e7f7a98b92c4940aaad9c18b80386e276f304`
- 每张 ReID crop 视为单人 full-image bounding box，不运行额外 detector。
- 不读取 `/mnt1/afrdata/Occluded_Duke/pose_data`。
- 不处理 query 或 gallery；后续评测接口没有 pose artifact 可读。

## 产物方案

新增独立提取脚本，只依赖 MMPose 公共 API。按文件名排序处理每张训练图像，保存：

- relative path；
- 原图 SHA256 与原始宽高；
- 17×2 原图像素坐标；
- 17 个 keypoint confidence；
- teacher/config/checkpoint/version 元数据。

关键点以 float32 保存，不预先生成大体积 heatmap。后续 paired augmentation 在训练时把坐标同步应用 resize、flip、pad 与 crop，再按目标 feature resolution 现场生成监督 heatmap。Random Erasing 只作用 RGB，pose target 作为 privileged clean target 保持不变。

产物放在全新的 derived-data 目录，不写入原始数据集，也不使用旧 `pose_data` 命名。最终 manifest 记录样本数、输入清单、config/weight SHA、target 文件 SHA 和统计摘要。

## 关键语义

水平翻转时必须同时：

1. 镜像 x 坐标；
2. 交换 COCO 左右关节：eyes、ears、shoulders、elbows、wrists、hips、knees、ankles。

crop 后落在视野外的关节在该增强样本中置为无效；不得把未同步增强的原图坐标直接监督增强后的 RGB。

## 门禁

1. fresh 官方 checkpoint 下载完成，URL/config/weight SHA 固定；不依赖旧仓库文件。
2. 16 张抽样推理检查 shape、坐标、confidence、有限值与原图尺寸映射。
3. 全量 15,618 样本一一覆盖，无缺失、重复、非有限值或无法解码图像。
4. 随机抽样重新在线推理，与离线记录做数值等价检查。
5. paired augmentation unit 覆盖 resize、左右交换、pad/crop、出界 mask；pose-disabled RGB 路径与官方 transform 的随机语义一致。
6. 只有 pose targets 与 paired augmentation 门禁通过后，才进入 TAPF 模块实现；本实验不启动 ReID 训练。

## 风险与失败解释

- ReID crop 中严重遮挡会降低部分 joint confidence；保留原始 soft confidence，不用人工补点。
- full-image bbox 可能包含背景，但避免引入 detector 版本和阈值作为额外变量。
- 若 ViTPose-Huge 资源或显存不可用，可在独立设计修订后改用官方 RTMPose-M；不得静默切 teacher。
- 离线 targets 只是训练期 privileged supervision，不构成测试期输入。
