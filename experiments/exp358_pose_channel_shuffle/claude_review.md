# Claude Broad Review — exp358 cross-PART (channel) shuffle

**Verdict: PASS（审查通过）** 无 Critical/High。Claude Opus 子代理逐行审查 forward gather、clip_part_head PART_KPS/_compute_pose_bias/_compute_gt_assignment、config diff。

## 逐项 focus 核对
### a. gather 正确性 — PASS
`cperm = argsort(rand(B,Kc))` 每行独立置换 [0..Kc-1]; `idx = cperm[:,:,None,None].expand(-1,-1,H,W)` (B,Kc,H,W); gather dim1: out[b,k,h,w]=scene[b,cperm[b,k],h,w] → 输出通道 k = 输入通道 cperm[b,k]。正确 per-image 通道置换, 每图自己的 cperm。

### b. scene/target 配对 — PASS
同一 idx 用于 scene + target 两个 gather, 通道一致。

### c. 训练端 — PASS
self.training 守卫; eval(training=False)跳过, LGPA test 路径用真未 shuffle 热图。shuffle 在 _run_backbone_with_psg + target-swap 前执行, shuffled 张量到达 LGPA。

### d. 单变量 — PASS
diff exp353 仅 POSE_CHANNEL_SHUFFLE True + OUTPUT_DIR; 默认 False 严格 no-op; pose_dropout_p 0.0 无隐藏交互。

### e. 与 exp357 独立 — PASS
exp358 yml 无 POSE_SHUFFLE → 默认 False → exp357 cross-image block 跳过, 只 exp358 block 触发。两 flag 独立, 不同时触发。

### f. 测的是不是声称的 — PASS(含 subtlety)
shuffle 后 PART_KPS[k] 选随机关键点通道子集 → 每部位 query 定位到本图随机关键点位置(本图空间 support 保留, 解剖映射打乱)。bias 与 KL assign GT 同源 shuffled 热图, 迭代内自洽。组基数保留(head 仍 5 通道, lower-leg 仍 2), 只随机化身份 = 干净 identity-only ablation。
**★subtlety: 背景通道(1-body_max)对置换不变**(body_max 是全 17 通道 max)→ shuffle 只打乱 5 前景部位身份, 保留前景/背景分离(fg/bg 非"解剖部位身份", scope 正确, 但 bg 通道不被扰)。

### g. 边界 — PASS
device/dtype/AMP 安全(rand→float32, argsort→int64 idx, gather 保 dtype, _compute_pose_bias 再 .float()); B=1 well-defined 无需 guard; idx stride-0 view 不 materialize, gather 输出同尺寸 ~13MB trivial; Kc 动态读 shape[1]。

## Findings
- Medium(判读非bug): 每 forward 重采样置换 = 随机正则; NO-DROP 时分不清"部位身份无关"vs"随机 shuffle 当补偿正则"。同 exp357 confound 类。记录结果时注明。
- Low-1: bg 通道置换不变(fg/bg 保留, 只 5 前景部位身份被打乱)——设计/注释加一句防误读。
- Low-2: target_heatmaps 被 shuffle 但本 config 不用(harmless, 防御性正确)。

## 结论
审查通过。通道 shuffle 正确、训练端、单变量, 破坏解剖部位身份保留同图空间 support, test 用真 pose。结果可与 exp357 联合判读(exp357 隔离图-pose 对应, exp358 隔离解剖部位身份, 都对 exp353=60.5)。共享 caveat: NO-DROP 可能部分反映随机 shuffle 正则 + 保留的 fg/bg 分离, 非纯"部位结构无关"。可进 Codex。
