# exp401 rich-budget C0正式训练协议

## 状态

`PROTOCOL-FROZEN / FORMAL SEALED-PASS / RICH_BUDGET_ROUTE_ALIVE /
PHASE-B INTERFACE GO`

## static封板

初次run的17项科学门中仅`official_data_read_only_paths`因YAML保留`('/mnt1/afrdata')`字面括号而FAIL；
该结果保留，未改config。static只把路径判定修正为同一字面内包含exact `/mnt1/afrdata`。正式两遍
18/18 PASS且result/runner逐字节一致，CUDA未初始化。static/config/result SHA=
`90c95b4ac1be32a8d4917882be1c407d17945511205446ede7ddaefb847f319d`/
`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`/
`85cab0e0a8391b3470f0e11acbd634d3dce2fee638432679a2ef9dc49cae020d`。

## 上游授权

- exp400=`FINAL_PRODUCTION_PREFLIGHT_PASS / FORMAL E120 GO`；
- exp400 result SHA=`3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f`；
- result内`formal_training_authorized=true`；exp394–400保持sealed且不得重跑。

## fresh与唯一GPU

fresh execution必须是source exact working tree的新实体，无alternate、tracked clean；CLIP/codebook必须是
新regular非symlink实体且SHA分别为`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`/
`fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`。启动前后任意时刻只允许一个
4090 main及其8个DataLoader worker。

## 训练边界

- config与冻结rich-budget C0只允许CLIP路径、codebook路径、OUTPUT_DIR三项fresh差异；
- `MAX_EPOCHS=120 / IMS_PER_BATCH=64 / SEED=1234 / CHECKPOINT_PERIOD=120 / EVAL_PERIOD=10`；
- 从随机/冻结pretrained构造自然开始，不读任何训练checkpoint；
- 不续训、不挑best、不按中间mAP/loss/GateAbs早停、不并行评测；
- e120前checkpoint必须为0，e120自然结束后必须唯一checkpoint。

## 监控

每轮核查HEAD/config/source、唯一main+8 workers、GPU唯一进程、runner/train log、epoch/iter、Loss、Pose、
Semantic、Mask、Presence、Q、EvidenceCos/Relation、Exec、Student、Reliability、GateAbs，以及NaN/Inf/
Traceback/RuntimeError/OOM/nonfinite/overflow/AMP warning。每个完整e10 eval记录mAP/R1/R5/R10但不裁决。

## final与反事实

训练自然退出后验证PID/workers退出、GPU空闲、唯一checkpoint与SHA、strict finite、teacher不在state、
两个router/evidence head retained、RGB-only。随后在同一checkpoint上串行执行full与all-router-bypass exact
retrieval，不改state、不保存新checkpoint。route alive门=`full-bypass >= +0.1 mAP`且`full >=56.7 mAP`。

## final封板

e120 full raw mAP=`57.1230075595`，all-router-bypass raw mAP=`57.0035860757`，差值=
`+0.1194214838 point`；两项预注册门均PASS。终审41项gate全PASS，执行前后checkpoint/config/source/
model state exact，两个router各在78个validation batch全部旁路并精确恢复。协议最终状态=
`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`；当前编号禁止重跑、补跑、续训或换seed。
