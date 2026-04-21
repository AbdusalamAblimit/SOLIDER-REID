# Claude Review — exp266b_best_b_op_s41_3090

**审查对象**: Base OP seed 41 on lab3090 (3090 24G), 和 srvA exp266b 双机并行

## 审查范围

1. `design.md` — 单变量 SEED 42→41, 复制 exp263d 策略到 OP
2. 代码改动: **无**
3. 机器切换: srvC 5060Ti → lab3090 3090 (24G)
4. docker env: `/root/miniconda3/envs/solider-reid/bin/python` (已验证 exp263d)
5. lab3090 GPU 刚空 (exp263d 14:27 FINAL), 立即利用

## 变量隔离

- 相对 exp266 (srvC seed 42 silent exit): 
  - SEED 42→41 (和 exp263d 策略一致)
  - **机器 5060Ti → 3090** (设备差,但 3090 24G 显存足, 无 OOM 风险)
- 同 config: Base Full Scaffold + PSG [-2,-1] + GCN512 + Full LGPA/OA-SD/ParAug/LOWER_BODY_OCC + WITH_CP

## 和 srvA exp266b 的协调

- srvA daemon 992 仍活着, 等 exp265b ckpt 后触发 exp266b (seed 41, TEST BATCH 128)
- OUTPUT_DIR 不同 (srvA 是 `_s41`, 本是 `_s41_3090`), **不冲突**
- 双 run 后形成 seed 41 multi-device 验证, 论文主表可用更稳的 (3090 版 FINAL 更早出)

## OOM 风险

- lab3090 3090 24G 显存, exp263d Base OD Full Scaffold 同配置训练 + eval 一路稳定无 OOM
- Base OP 数据集规模类似 Occ-Duke, eval 显存消耗类似
- 无需降 TEST BATCH (TEST default 256 安全)

## 时间预算

- exp263d Base OD 14h50min (pwrlim 280W + WITH_CP)
- exp266b_3090 预计 12-14h (Base OP 可能略快, query/gallery ID 数少)
- ETA: 14:29 启动 → 后天 03-04:00 CST FINAL

## 与论文影响

- 修 exp266 "e60 eff FINAL" 主表瑕疵, 给 e120 完整 FINAL
- 和 exp265b (Small seed 41) 在 srvA 的 FINAL 形成 "Base vs Small" on OP seed 41 对照
- Phase 1 OP 主表可由 {exp264 Tiny, exp265 Small, exp266b_3090 Base} 三行, 全 e120 FINAL, 无瑕疵

## 结论

**审查通过**。

lab3090 空闲即启,符合 CLAUDE.md 持续执行铁律。零代码改动, 单变量 SEED,  3090 24G 无 OOM 风险。
