# 实验 exp228: GSPB(0.05) + PADPQ K=8 on Tiny

## 动机
- exp225 (GSPB+PADPQ K=4): 64.2/74.9 (+1.0 vs OA-SD)
- PADPQ K=8 单独比 K=4 好 (+0.2 mAP in exp223b vs exp223)
- 假设: GSPB + K=8 可能比 GSPB + K=4 更好

## 核心假设
K=8 提供更广的 deformable receptive field, 与 GSPB 组合可能达到 64.5%+。

## 对照组
- exp225 (GSPB+PADPQ K=4): 64.2/74.9
- exp223b (PADPQ K=8 only): 63.9/74.3
