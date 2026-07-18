# exp389 监控记录

## 当前状态

- 状态：`NO-START`；
- 4090：空闲；
- 正式 output：尚未创建；
- 直接对照：exp387 clean Occ-Duke D0=`57.6/67.7/80.8/84.6`；
- 当前阶段：只允许从零实现 early Stage-1 anchor → Stage-2 六个有效 PSG，并完成全部门禁。

## 初始设计审计

- 禁止复用旧 HT0 runtime、旧 pose_data/cache/path mapping；输入继续只使用 exp386 fresh ViTPose-H train-only artifact；
- 既有 late Stage-2 anchor → Stage-3 两 PSG 必须逐参数、构造顺序和数值路径保持 D0 exact；
- 新增 early anchor 从 Stage-1 pre-downsample feature 产生场，六个独立 consumer 分别位于 Stage-2 每个 block 后；
- 八个 consumer 都位于最终 GAP 使用的 spatial feature 上游，不接受 terminal dead consumer；
- 两层均使用同一 teacher/handoff/student 日程，eval 均只读 RGB internal field；
- config-off、D0-off、双层 route/gradient、真实 batch64 CUDA/AMP/overflow、strict state、pose-free parity、consumer path 与效率任一未通过前，保持 `NO-START` 和 GPU 空闲。
