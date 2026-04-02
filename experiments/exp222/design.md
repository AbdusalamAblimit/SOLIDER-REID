# 实验 exp222: GSPB on Small — GCN+PAA+CE+OA-SD + gradient scale=0.05

## 动机
- exp220 (GSPB on Tiny): equal_concat 62.9 (-0.3 vs OA-SD), **maxsim_hybrid 64.6 (+0.4 vs OA-SD!)**
- GSPB 的 5% Part gradient 改善了 per-keypoint features → 只在 MaxSim 体现
- **需要验证 GSPB 在 Small 上的效果**
- 当前 Small 最佳: 72.4% maxsim (exp210b)

## 核心假设
GSPB (5% gradient scale) 在 Small 上也能改善 per-keypoint features，
让 MaxSim 达到 72.8-73%+。

## 技术方案
- 与 exp206r 完全相同 + MODEL.POSE_PART_GRAD_SCALE 0.05

## 对照组
- exp206r (scale=0): 70.6 eq, 72.3 maxsim
- exp210b (scale=0, PKC=0.05): 70.6 eq, 72.4 maxsim
- exp222 (scale=0.05): 目标 73%+ maxsim
