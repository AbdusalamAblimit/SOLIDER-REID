# exp163 PNIS 监控
- 方法: Pose-Normalized Identity Space
- 基线: exp030a-eq (60.73%)
- 运行: 本地 3090
- CHECKPOINT_PERIOD: 20

## 结果：ep10 止损
- pn_alpha=0.047（从未变化），pn_off=0.487（从未变化）
- PoseNormalizer 没有学到任何东西
- 原因：triplet-only 梯度信号太弱，0.047*0.487=0.023 的减法对 ~70 范数的 feature 完全不可见
- **结论：PNIS 失败。"减法去 pose"在当前实现下不可行——梯度死区。**
