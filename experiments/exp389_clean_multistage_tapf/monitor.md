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

## 首版实现与基础门禁

- 本地实现提交=`e6c49c4`；远端独立 preflight repo=`/home/afr/SOLIDER-REID-exp389-preflight-e6c49c4`，执行提交=`bbf272c49c9aa1159b61b16919665255c3a76a7b`；
- HT0 config SHA256=`f4b6cfde243de97634eef9320a7a96e2d58f6cd0fc747ee4a747997da455675b`；正式 output 尚未创建，GPU 在门禁前后均空闲；
- `CleanTapfHt0` 先构造完整 `CleanTapfD0`，随后追加独立 early anchor 与六成员 early PSG bank；`MODEL.TAPF.HIERARCHICAL=False` 为默认值；
- Swin 仅在 hierarchical 分支手动展开 Stage-2 六个 block，并保持原生 block→gate→下一 block、最终 downsample 顺序；D0 分支不进入该代码；
- unit 由既有 5 项扩展为 6 项，Gaussian/reliability/空 valid/schedule/zero-field identity/D0 公共初始化/early 三 bank route/early-late 参数独立/eval exploding pose 全部 PASS；
- 真实 Swin-T batch2 CUDA/AMP train/backward smoke PASS：early/late field shape=`17×48×16` / `17×24×8`，e6 student fraction=`0.2/0.2`，early/late gate route=`6/2`，两个 anchor 与两组 PSG 均获得有限梯度，最终 feature=`2×768` 且全量有限；
- 可执行结论：`EXP389_FULL_MODEL_CUDA_SMOKE_PASS`。

该结果只通过基础实现门禁，不替代 config-off/D0-off exact、真实 paired batch64/24-step、严格 gradient ownership、overflow、strict state、pose-free、逐 consumer path 与效率门禁；状态继续为 `NO-START`。
