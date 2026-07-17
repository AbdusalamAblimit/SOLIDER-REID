# 实验 exp385：干净 Occluded-Duke Swin-Tiny B0

## 动机

exp384 已在官方最后提交的干净代码上完整复现 Market1501：e120 为 `91.6 mAP / 96.3 R1`。下一步不是复用旧实验实现，而是在同一官方代码和运行环境中，从 Occluded-Duke 原始 RGB 建立可核验的 B0，作为后续从零重写 TAPF D0 的唯一直接对照。

## 核心假设

只新增一个最小 Occluded-Duke 数据集 loader，并把官方 Market Swin-Tiny 配方中的数据集与 output 替换为 Occluded-Duke，即可在不读取任何旧 pose/cache/配置/checkpoint 的条件下形成稳定、可复现的 RGB-only baseline。

## 原始数据边界

- 根目录：`/mnt1/afrdata/Occluded_Duke`
- train：`bounding_box_train`，15,618 张，702 IDs，8 cameras
- query：`query`，2,210 张，519 IDs，8 cameras
- gallery：`bounding_box_test`，17,661 张，1,110 IDs，8 cameras
- 三个 split 均无非法文件名、空文件、不可解码 JPG 或 split 内内容重复。
- train 与 test IDs 完全不相交；全部 query IDs 都存在于 gallery。
- query 与 gallery 有 1,870 个同名、同内容副本。标准 Market-style evaluator会剔除同 PID、同 camera 的 gallery 样本，因此必须保留标准 17,661 张 gallery，不能自行删除或改写 split。
- 根目录现有 `pose_data/` 属于旧实验产物。新 loader 只构造上述三个显式 RGB 目录，绝不扫描、导入或读取 `pose_data/`。

原始 RGB 内容清单 SHA256（按排序后的文件名、大小和逐文件 SHA256 聚合）：

- train：`9be350a47c848844053c86a7f58e7f7a98b92c4940aaad9c18b80386e276f304`
- query：`e7de2acbfebee35177dd3aeb176298ec940066ff1b794d7cd9d777b5b1f01a4d`
- gallery：`0a4d1f3aa0d736ae6faebd3a5d1e2d6940252e8409955f30fae45c2139c44351`

## 技术方案

1. 新写 `datasets/occluded_duke.py`：
   - 只接受完整文件名语法 `PID_cCAM_fFRAME.jpg`；
   - train PID 确定性重标为 `0..701`；
   - camera 从磁盘的 `1..8` 映射为模型接口的 `0..7`；
   - 严格检查三个 RGB split 存在，不对 `pose_data` 提供任何入口。
2. 在官方 dataset registry 中仅增加 `occluded_duke` 条目。
3. 新写独立 config，保持官方 Market 配方的模型、增强、采样、batch、optimizer、scheduler、epoch、semantic weight 和评测协议不变，只替换 dataset 与 output。
4. 使用与 exp384 相同且已逐张量核验的官方 converted Swin-T teacher 初始化，不使用任何旧 Occluded-Duke 训练 checkpoint。

## 对照组

本实验是后续 matched D0 的 B0。D0 必须使用同一数据、初始权重、batch 64、seed 1234、120 epochs、SGD、LR 0.0008、增强、sampler 和评测协议；届时唯一方法变量才是重新实现的完整 anchor+PSG。

Market B0 只验证官方代码与运行环境，不与 Occluded-Duke 绝对指标横比。

## 启动门禁

1. loader 单元检查精确复现 split/ID/camera 数量、train relabel 与 query-gallery ID 约束；所有返回路径必须位于三个 RGB split。
2. DataLoader 首批 batch 64、标签范围、图像尺寸与有限值通过。
3. 官方 teacher 加载、CUDA forward/backward、AMP/GradScaler、optimizer state 与严格有限值通过。
4. 独立一 epoch smoke 完成 train/eval/checkpoint，自然退出且无 `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow`。
5. smoke 结束、GPU 空闲、正式 output 不存在后，才 fresh 启动 120-epoch B0。

## 预期结果

获得一个仅依赖原始 RGB、官方代码和公开初始化的 Occluded-Duke e120 baseline。性能只在完整训练和终审后记录，不使用单个早期 epoch 或 best checkpoint 代替 final。

## 风险与失败解释

- query/gallery 的 1,870 个副本是标准数据结构，不是 loader 泄漏；若 evaluator 未执行同 PID/同 camera 剔除，则门禁失败，禁止训练。
- 若新 loader 数量或标签不一致，优先修复数据解析，不得回退到旧 `occluded_duke.py`。
- 若 CUDA/AMP 或 smoke 异常，先定位运行时与数据，不得通过减 batch、续训或旧 checkpoint 绕过。
- B0 跑满并终审前，不实现或启动 TAPF D0。
