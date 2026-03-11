# 实验 exp022: Market-1501 Baseline

## 动机
- 前 21 个实验全部在 Occluded-Duke 上完成，需要在 Market-1501 上验证 PSG 的跨数据集泛化性
- Market-1501 是 ReID 领域最标准的 benchmark，论文必须报告
- 先跑 baseline 确定 Market-1501 上的基准性能

## 创新点 / 核心想法
- 本实验不涉及创新，仅是建立 Market-1501 baseline
- 使用与 Occluded-Duke baseline 完全相同的配置（SOLIDER pretrained Swin-Tiny, SW=0.2）

## 技术方案
- Config: `configs/market/swin_tiny.yml`
- 与 Occluded-Duke baseline 相同：SOLIDER 预训练 + SGD + cosine warmup + 120 epochs
- 无 pose 模块，纯 SOLIDER-Swin-Tiny
- Output: `./log/market1501/exp022_baseline/`

## 预期结果
- 参考 SOLIDER 论文，Market-1501 + Swin-Tiny 应在 mAP ~88-90%, R1 ~95% 左右
- 这是一个非遮挡数据集，baseline 本身就很高

## 对照组
- 这是 Market-1501 上的第一个实验，无对照
- 后续 exp023 (PSG) 将以本实验为对照
