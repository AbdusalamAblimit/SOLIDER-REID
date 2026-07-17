# 实验 exp384：官方 SOLIDER 干净复现与 TAPF 重建起点

## 动机

此前研究分支经历了三百余次实验，运行时代码已经混入多代数据接口、姿态缓存和模型变体。exp384 不复用这些实现，而是回到 SOLIDER-REID 官方最后提交，在可核验的干净基线上先复现官方结果，再从零实现 TAPF。

## 基线来源

- 官方最后提交：`8c08e1c3255e8e1e51e006bf189e52cc57b009ed`（2023-05-04）
- 本地工作分支：`codex/solider-official-tapf-clean`
- 远端 fresh repo：`/home/afr/SOLIDER-REID-official-repro-8c08e1c`
- 官方历史 bundle SHA256：`690877fb8277b4f278f0b4432f0538816b76c1bfc2686b299f84c603c4242478`

治理规则、论文 Markdown 和 `experiments/` 作为研究档案同步，但官方 `config/`、`datasets/`、`loss/`、`model/`、`processor/`、`solver/`、`train.py` 与 `test.py` 在兼容补丁前均与官方提交一致。旧实验脚本不得进入新训练 import 路径。

## 第一阶段：官方复现

使用官方 Swin-Tiny / Market1501 配方：

- 数据：`/mnt1/afrdata/market1501` 中的原始 JPG；不读取任何旧 `pose_data`
- 训练/测试尺寸：`384 × 128`
- batch：64
- seed：1234
- epoch：120
- optimizer：SGD
- base LR：0.0008
- semantic weight：0.2
- 测试：不使用 re-ranking
- 官方报告目标：`91.6 mAP / 96.1 Rank-1`

RTX 4090 无法使用官方声明的 CUDA 10.1 / PyTorch 1.7.1。允许的唯一代码偏差是使官方代码运行在 PyTorch 1.13.1 + CUDA 11.7 上的兼容性修复；不得改变模型结构、数据增强、采样、loss、optimizer、scheduler 或评测协议。

## 第二阶段：姿态数据重建

不得读取旧 pose cache、旧路径映射或旧姿态融合代码。只允许两种输入方式：

1. 从原始 ReID 图像重新离线提取姿态，并生成带版本、模型、输入尺寸和图像哈希的 manifest；
2. 训练时通过独立的 ViTPose 或 RTMPose 适配器在线推理。

第一版优先采用离线提取，保证 B0/D0 读取相同 RGB 样本、训练可重复且不把 pose detector 的波动混入 ReID 对照。在线模式只作为等价接口和抽样校验，不默认与 ReID 联合训练。

## 第三阶段：从零实现 TAPF

TAPF 必须作为独立、小型、可关闭的模块重新实现。完整方法仍定义为 anchor 与后继 PSG 的原子组合；训练期使用姿态监督，推理期模型接口只接受 RGB。旧 `pose_backbone_model.py`、旧 `model/modules/` 和旧数据管线不得复制或导入。

## 启动门禁

1. Market1501 官方 loader 数量与 ID 统计一致；
2. 官方权重加载键与未加载参数清单可解释；
3. CUDA forward/backward、AMP、overflow 与有限值检查通过；
4. 先完成官方 B0 的短程确定性检查，再自然训练 120 epoch；
5. B0 复现结束后才实现 pose adapter 和 TAPF；
6. TAPF D0 必须与 B0 保持单变量、同 batch、seed、epoch 和评测协议。

## 风险与失败解释

- 新运行时可能造成数值差异，必须记录版本和兼容补丁，不能把差异误写成方法收益。
- 若官方指标无法复现，先定位权重、数据、scheduler 和 runtime 差异，不得直接开始 TAPF。
- 若 pose 重新提取失败或许可不清晰，停止 D0，不得回退到旧 pose cache。
