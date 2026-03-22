# exp149 SCFA 监控

## 实验信息
- 方法: SCFA（Symmetry-Conditioned Feature Aggregation）
- 类型: skeleton branch 表示重构
- 主基线: `exp030a-eq`
- 当前状态: 已完成实现与工程自检，待 Claude 广范围审查

## 启动前检查清单
- [ ] 对称 pairs 定义清楚
- [ ] 默认 config 不受影响
- [ ] skeleton branch 聚合逻辑与旧逻辑单变量隔离
- [ ] `scfa_*` 行为日志接好
- [ ] Claude 广范围审查通过

## 当前判断
- 这条线不是 completion，不是 scorer，也不是 attention bias
- 若成立，可把 story 推到“pose-defined bilateral redundancy”

## 启动记录

### [2026-03-22 06:42] 代码接入完成，保持对 `exp030a` 的单变量关系
- 已修改:
  - `config/defaults.py`
  - `model/modules/skeleton_gcn.py`
  - `model/pose_backbone_model.py`
  - `processor/processor.py`
  - `configs/occluded_duke/pose_psg_gcn_scfa.yml`
- 实现要点:
  1. 只在 `SkeletonGCNHead` 里重写 bilateral token aggregation
  2. 不改 backbone、不改损失、不改 test-time trick
  3. 新表示由 `nose + homologous tokens + asymmetry tokens` 组成
  4. `scfa_*` 统计已接到 trainer 日志出口

### [2026-03-22 06:45] 自检通过，`scfa_*` 已可真实产生
- 语法检查:
  - `python -m py_compile ...` 通过
- 轻量单元前向:
  - `feat_shape = (4, 768)`
  - `cls_shape = (4, 702)`
  - `scfa_stats` 示例:
    - `cov = 1.0`
    - `hm = 0.617`
    - `hs = 0.222`
    - `am = 0.343`
    - `as = 0.261`
    - `hn = 1.000`
    - `an = 8.702`
    - `pg = 0.250`
    - `eq = 0.219`
- 额外说明:
  - 尝试做整模型 GPU probe 时，因为本地主卡正在跑 `exp148`，触发了显存竞争 OOM
  - 这不是 `SCFA` 本身的错误，因此改用 `SkeletonGCNHead` 级别单元前向验证
- 当前判断:
  - 继续
  - 原因:
    1. 这次不是纸上设计，表示层与日志链路都已打通
    2. 下一步应按规则送 Claude 做广范围审查，而不是直接开远程训练

### [2026-03-22 06:48] Claude 广范围审查已启动
- 审查请求:
  - `experiments/exp149/claude_review_request.txt`
- 输出目标:
  - `experiments/exp149/claude_review.md`
- 当前判断:
  - 等待审查
  - 原因:
    1. 远程机器现在空闲，但不能越过审查规则抢跑
    2. 这次要优先确认这条线是不是“真新方向”，而不只是能跑通

### [2026-03-22 06:50] 审查进程重挂为稳定模式
- 问题:
  - 第一种 `stdin 重定向 -> 文件` 的调用方式长时间不落内容，无法判断是 Claude 慢还是壳进程异常
- 处理:
  - 已中断旧会话
  - 改为直接把请求文本传给 `claude -p`，并保留运行中的 PTY 会话监控
- 当前状态:
  - Claude 进程仍在运行，CPU 正常占用
- 当前判断:
  - 等待审查
  - 原因:
    1. 现在已经不是“空等空壳”
    2. 若稍后审查仍无结果，再考虑进一步收紧 prompt 或拆分材料
