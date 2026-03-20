# exp124 监控

## 实验信息
- 方法: Stronger Pair-Delta SCRD
- 类型: 训练端单变量改进
- 运行位置: 远程 5060 Ti
- 主配置: `exp123_pair_delta_scrd`
- 核心变量: `POSE_CSRD_PAIR_WEIGHT_ALPHA = 4.0`
- 输出目录: `log/occluded_duke/exp124_pair_delta_scrd_a4`

## 启动前检查

- [x] Backbone 固定为 `Swin-Tiny`
- [x] batch size 保持 `64`
- [x] 相对 `exp123` 只修改 `POSE_CSRD_PAIR_WEIGHT_ALPHA`
- [x] support-complete teacher、bank 更新、主 loss 配比全部保持不变
- [x] 默认行为不变，开关关闭可完全回退 `exp123`
- [x] `OUTPUT_DIR` 独立

## 启动记录

### [2026-03-20 10:36] 实验准备

- 启动原因:
  1. 远程 `exp121` 已收敛，GPU 空出
  2. `exp123` 到 `ep60` 已给出“pair focus 方向对”的证据，但当前 `pair_focus` 长期只有 `1.06~1.08`
  3. 当前最有信息量的下一跳不是换题，而是验证 **focus 强度是否就是瓶颈**
- 当前判断: 待启动
- 原因:
  - `exp124` 是相对 `exp123` 的最小下一跳：只改 pair focus 的放大强度

### [2026-03-20 10:40] 首次远程启动失败并立即修正

- 异常:
  1. 第一次远程启动沿用了旧的解释器路径 `/root/miniconda3/envs/solider-reid/bin/python`
  2. 远程实际报错：
     - `nohup: failed to run command '/root/miniconda3/envs/solider-reid/bin/python': No such file or directory`
- 处理:
  1. 立即确认远程可用解释器路径
  2. 验证 `/usr/local/bin/python` 已正确安装 `torch / torchvision / yacs / cv2`
  3. 用 `python -u train.py ...` 重新后台启动
- 当前判断: 继续
- 原因:
  - 这是远程环境路径差异，不是实验机制问题；修正后即可继续按单变量方案执行

### [2026-03-20 10:40] 启动确认（远程 5060 Ti）

- 运行位置: 恒源云 5060 Ti
- 启动方式: 后台 `nohup`
- 输出目录: `log/occluded_duke/exp124_pair_delta_scrd_a4`
- nohup 日志: `log/occluded_duke/exp124_pair_delta_scrd_a4/remote_nohup.log`
- 关键确认:
  1. 远程仓库已同步到 `7fbcdd0`
  2. 配置已生效：`POSE_CSRD_PAIR_WEIGHT_MODE = delta`
  3. 放大系数已生效：
     - `[CSRD-PW] mode=delta, alpha=4.0`
  4. support-complete teacher 仍正常启用：
     - `[CSRD-ST] enabled: low_thr=0.3, update_thr=0.7, mom=0.9, min_count=1, stop_epoch=-1`
  5. GPU 已占用约 `6692 MiB`，利用率约 `69%`
- 当前判断: 继续
- 原因:
  - 现在已经形成了“本地 `exp123 alpha=1.0` + 远程 `exp124 alpha=4.0`”的干净并行对照
