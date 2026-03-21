# exp134 监控

## 实验信息
- 方法: `Changed-Pair Sparse LPCS`
- 类型: `exp133` 的单变量并行远程实验
- 运行位置: 未启动
- 当前状态: 已完成设计，待代码接线自检与 Claude 审查
- 直接对照:
  - `exp133 LPCS`
  - `exp125 Sparse Pair-Delta SCRD`

## 启动记录

### [2026-03-21 08:15] 设计建档
- 启动原因:
  1. 本地 `exp133` 已进入正式训练，但当前训练机制仍对所有 pair 做连续 teacher-change 加权
  2. `exp125` 已证明，在 `SCRD` 线上“稀疏 changed-pair 路由”比连续平滑加权更值得探索
  3. 需要一个与 `exp133` 同主线、但不只是调 hidden/scale 的远程并行实验
- 当前判断: 待实现
- 原因:
  - 先把单变量设计和代码接线落稳，再按规则交给 Claude 审查

### [2026-03-21 08:18] 代码接线与自检完成
- 已完成:
  1. `defaults.py` 新增：
     - `POSE_LPCS_PAIR_MODE`
     - `POSE_LPCS_PAIR_TOP_RATIO`
  2. `processor.py` 中 `LPCS` loss 新增 `delta_top` 路由
  3. 记录统计新增：
     - `lpcs_psr`
     - `lpcs_pf`
  4. 配置已建：
     - `configs/occluded_duke/pose_psg_gcn_lpcs_delta_top.yml`
- 单变量自查:
  1. 相对 `exp133` 仅新增两项 MODEL config：
     - `POSE_LPCS_PAIR_MODE: 'delta_top'`
     - `POSE_LPCS_PAIR_TOP_RATIO: 0.25`
  2. `OUTPUT_DIR` 已独立为：
     - `log/occluded_duke/exp134_lpcs_delta_top`
- 已通过自检:
  1. `python -m py_compile processor.py config/defaults.py` 通过
  2. config 差分确认仅有目标变量变化
- 当前判断: 等待 Claude 审查
- 原因:
  - 代码接线已完成，但按用户规则必须先通过 Claude 审查才能启动训练

### [2026-03-21 08:20] Claude 审查已启动
- 审查文件:
  - `experiments/exp134/claude_review.md`
- 审查重点:
  1. 单变量性
  2. 默认行为不破坏
  3. `delta_top` 是否真正形成 changed-pair sparse routing
  4. 统计量是否足以解释机制
  5. 远程启动风险
- 当前判断: 等待审查结果
- 原因:
  - 审查未通过前，不允许启动远程训练

### [2026-03-21 08:23] 第一轮 Claude 审查未通过，已进入修复
- 审查结论:
  - `experiments/exp134/claude_review.md`
  - **不允许启动**
- 审查抓到的问题:
  1. Critical:
     - `processor.py` 缺少 `import math`，会在 `epoch 21+` 首次进入 `_select_top` 时崩溃
  2. Medium:
     - 训练端 `base_dist` 仍硬编码 `1:1`，与测试端 `CVK_GLOBAL_WEIGHT / CVK_KP_WEIGHT` 机制不完全统一
  3. Low:
     - `delta_top + top_ratio >= 1.0` 缺少显式 warning
- 已完成修复:
  1. 补上 `import math`
  2. `LPCS` 训练端 `base_dist / teacher_dist` 改为与测试端一致的 `CVK` 权重驱动
  3. 新增 `delta_top + top_ratio >= 1.0` warning
  4. 新增 `delta_top` 下非法 `top_ratio` 的显式校验
  5. 修复后 `py_compile` 已重新通过
- 当前判断: 等待二次审查
- 原因:
  - 第一轮发现的是实际阻塞项，必须二次审查确认修复生效后才能启动
