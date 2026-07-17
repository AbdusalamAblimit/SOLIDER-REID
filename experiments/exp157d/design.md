# 实验 exp157d: 人体范围内随机遮挡 (Body-Bbox Random Occlusion)

## 动机
用户要求的消融：在人体 bbox 范围内随机位置贴 VOC 物体。
与 ROA（画面任意位置）对比，限制遮挡只出现在人体区域内。

## 技术方案
- 用 pose keypoints 计算人体 bbox (所有可见 keypoint 的范围)
- 在 bbox 内随机位置贴 VOC 物体
- 概率 p=0.7
