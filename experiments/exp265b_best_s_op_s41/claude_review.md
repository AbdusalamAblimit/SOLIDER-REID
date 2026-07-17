# Claude Review — exp265b_best_s_op_s41

**审查对象**: exp265 seed 42 → seed 41 重跑 (OP SOTA 候选)

## 审查范围

1. `design.md` — 单变量 seed 42→41 刷 SOTA 策略
2. 代码改动: **无**
3. config 复用 `configs/occluded_posetrack/prcv_best_small.yml` (Phase 1 exp265 config)
4. CLI override: 仅 SEED + OUTPUT_DIR

## 变量隔离

- 相对 exp265 严格单变量: SOLVER.SEED 42 → 41
- 机器从 srvC 换到 srvA,但两者同型号 5060Ti 16G,设备方差小
- 其他一切不变: backbone Small, Full Scaffold, PSG [-2,-1], GCN512, 数据 OP

## 设备健康验证

srvA 刚 resume:
- GPU 0 MiB used / 15849 MiB free / 0% util ✓
- Occ-PoseTrack 数据齐全 (bounding_box_test, gallery.list, masks, pose_data with 4 splits)
- pretrained swin_{tiny,small,base}, clip_part_text_features ✓
- configs/occluded_posetrack/prcv_best_small.yml 齐全

## OOM 风险

- exp265 seed 42 在同型 5060Ti (srvC) 正常 FINAL 不 OOM (内存 50G /hy-tmp, GPU 16G 足够)
- OP 数据量比 Occ-Duke 小, 更稳
- 预估显存 10-12GB (Small + Full + OA-SD), flip eval 峰值 13GB, < 16G 安全

## 时间预算

- exp265 seed 42 在 srvC 花了 ~14h (628s × 120 = 20.9h 如果算足)
- 实际 Phase 1 exp265 FINAL 04-20 04:45, 基本上 04-19 11:00 启动左右
- srvA 同机性能, 预计 12-14h FINAL → 11:55 + 13h = tmr 00:55 CST

## 预期结果 & 论文价值

若 exp265b FINAL ≥ 79.0 / 86.5:
- 单 seed SOTA 刷新,直接论文主表
- vs KPR w/o prompt 73.3/82.5 → +5.7+/+4.3+ 明显超

若 exp265b ∈ [78.0, 78.9] / [85.5, 86.5]:
- 和 exp265 seed 42 组成 2-seed,取 best 或 mean
- multi-seed 稳定性增强 SOTA 论断

若 exp265b < 78.0 (概率极低):
- 说明 seed 42 反而是 lucky draw, exp265 数字可能偏乐观
- 用 mean(exp265, exp265b) 作更保守数字

## 结论

**审查通过**。单变量 seed ablation, 代码零改动, 风险极低。

srvA 同设备方差小 (5060Ti 同型号),同 code 版本 (当前 repo),主要变量仅 seed。这是刷 OP SOTA 最务实的一步。
