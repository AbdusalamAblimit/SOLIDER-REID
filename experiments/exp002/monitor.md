# exp002: Spatial Softmax Part Pooling

## 实验配置
- **分支**: `exp/pose_heatmap`
- **Config**: `configs/occluded_duke/pose_spatial_softmax.yml`
- **输出目录**: `./log/occluded_duke/exp002_spatial_softmax`
- **GPU**: RTX 3090 24GB

## 与 exp001 的唯一区别
- 热图归一化：sigmoid → spatial_softmax (temperature=1.0)
- Spatial softmax 在每个 part 的 12×4=48 个位置上做 softmax，产生更尖锐的注意力

## Baseline 参考（来源: log 文件）
- exp000 Baseline: mAP 56.6%, R1 66.5%
- exp001 Sigmoid: mAP 57.1%, R1 66.7%

## 监控日志

