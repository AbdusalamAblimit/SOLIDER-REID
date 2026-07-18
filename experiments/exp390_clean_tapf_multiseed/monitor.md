# exp390 监控：官方干净 TAPF matched 多 seed

## 当前状态

- 状态：`NO-START`；
- GPU：空闲；
- 已有 seed1234 pair：B0=`57.4/67.4/80.6/85.2`，D0=`57.6/67.7/80.8/84.6`，
  D0−B0=`+0.2/+0.3/+0.2/−0.6`；
- 预注册新增顺序：B0-s4321→D0-s4321→B0-s2025→D0-s2025；
- 任一新增 arm 启动前必须完成 design 中的 config、继承代码门禁与 fresh execution 检查。

## 不变量

- 禁止复用旧 runtime、旧 pose data/cache/path mapping；
- B0 只读原始 RGB；D0 只读 exp386 fresh train-only artifact；query/gallery 始终 RGB-only；
- batch64/120 epoch/SGD/lr0.0008/semantic weight0.2/增强/sampler/eval10/checkpoint120 固定；
- 不并行、不续训、不重复、不挑 best、不按中间性能提前停止；
- 每次只提交本实验目标文件，保护用户工作树，禁止 `git add -A`。

## Config 静态门禁

- B0-s4321 SHA256=`8fd054b528608b524212170962f30274b3185c3ee22304720f305f81816a9cfa`；
- D0-s4321 SHA256=`979c897da79327bd8ecc04fcc4b370f0f5ad6b318170fe3afec5594a5c769711`；
- B0-s2025 SHA256=`30c2000dff8e9fa1d554a2873cf16c98a5d8e7d62182c2f95501e4fb8be20a33`；
- D0-s2025 SHA256=`56b2a30fd1856d2dc1077df013cc5d4bab9312be5488f25e1fe7dfa882263116`；
- 四个 config 相对各自 seed1234 canonical 文件的文本 diff 均严格只有 `SOLVER.SEED` 与
  `OUTPUT_DIR`；dataset、teacher、pose artifact、batch、epoch、optimizer、LR、增强、sampler、
  eval/checkpoint 周期均未改变；
- 四个 output 名称唯一且互不重叠。当前均未创建，状态继续为 `NO-START`，等待远端继承门禁与
  fresh execution 审计。
