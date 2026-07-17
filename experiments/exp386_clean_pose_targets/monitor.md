# exp386 监控记录

## 起点

- exp385 clean Occluded-Duke B0 已封板：`57.4/67.4/80.6/85.2`
- 4090 空闲：2 MiB / 0%
- mmpose-abu API 可用：`init_model`、`inference_topdown`
- installed ViTPose-Huge config SHA256：`72fcd88a4483742869867a1da2aa6e2af533155950185e524bf4ed24e7c15d36`
- 官方权重 URL 已从 MMPose 1.3.2 model index 现场核对
- 旧仓库同名模型文件与旧 `pose_data` 均不复用

下一步：fresh 下载官方 checkpoint 到新目录并校验，再实现最小提取器；尚未启动 pose 提取或 ReID 训练。
