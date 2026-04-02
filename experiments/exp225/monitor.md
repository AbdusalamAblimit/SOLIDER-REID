# exp225 Tiny + GSPB + PADPQ Combined 监控

配置: 基于 pose_psg_gcn_paa_roa.yml + OA-SD + PLBOA + GSPB(0.05) + PADPQ(K=4)
对照: exp220 GSPB(maxsim 64.6), exp223 PADPQ(eq 63.7), exp191 OA-SD(63.2/75.4)

## 检查点

### [18:33] 检查点 #1

远程刚启动。等 ep10 eval。

### [02:59] 检查点 #2 — ep10

**ep10: 38.3%** (vs GSPB 40.1, PADPQ 37.5, OA-SD 34.3)
介于 GSPB 和 PADPQ 之间——没有明显的叠加效果。
**决策**: 继续到 final

### [03:20] 检查点 #3 — ep20

**ep20: 49.4%** (vs GSPB 49.1, PADPQ 47.7, OA-SD 46.0)
微幅超过 GSPB (+0.3)。组合效果不明显。
**决策**: 继续
