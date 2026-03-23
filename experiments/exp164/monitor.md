# exp164 STD-PR V2 (Anchor Queries) 监控
- 本地: V2+PLBOA, 远程: V2 no PLBOA
- V2 改动: keypoint-sampled anchors 初始化 queries
- V2 无 PLBOA 的 R1 持续改善 (+2~4 vs V1)
- V2+PLBOA 目前弱于 V1+PLBOA (anchor 在遮挡位置采噪声)
- 训练速度慢 (113s/ep on local)
