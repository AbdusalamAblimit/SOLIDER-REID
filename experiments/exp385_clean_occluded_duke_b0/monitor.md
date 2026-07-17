# exp385 监控记录

## 数据审计

- 数据根：`/mnt1/afrdata/Occluded_Duke`
- 原始 RGB：train/query/gallery=`15618/2210/17661`
- IDs：train/query/gallery=`702/519/1110`
- cameras：三个 split 均为 `1..8`
- 非法文件名/空文件/解码失败/单 split 内容重复：均为 0
- train-test ID 交集：0；query 缺失 gallery ID：0
- query-gallery 同名且同内容副本：1,870；标准 evaluator 按同 PID/同 camera 排除
- 旧 `pose_data/` 共 50,987 个文件，明确排除，不属于本实验输入

原始 RGB 内容清单 SHA256：

- train：`9be350a47c848844053c86a7f58e7f7a98b92c4940aaad9c18b80386e276f304`
- query：`e7de2acbfebee35177dd3aeb176298ec940066ff1b794d7cd9d777b5b1f01a4d`
- gallery：`0a4d1f3aa0d736ae6faebd3a5d1e2d6940252e8409955f30fae45c2139c44351`

结论：标准 split、文件名与原始图像门禁通过；下一步新写最小 loader/config，不复用旧运行时代码。

## Loader 与配置实现

- 本地提交：`35e1fd1`（仅新增最小 loader、registry 接线与独立 config）
- 远端执行提交：`3ce6506768ffe5cdf5453640a3ef0e1b721e6641`
- config SHA256：`90d715d29324a069a0245e27a06d79fda8f261427a4f46b7e202c3e0351ad867`
- loader 只拼接 `bounding_box_train/query/bounding_box_test`，没有 `pose_data` 参数、扫描或 import
- 相对官方 Market 配方的有效变化仅为 dataset/output；semantic weight 0.2 与 seed 1234 从命令行/默认值改为 config 显式固定

## Unit / DataLoader 门禁

真实原始数据检查全部通过：

- split=`15618/2210/17661`，IDs=`702/519/1110`，cameras=`8/8/8`
- train labels 精确连续为 `0..701`，camera 精确为 `0..7`
- 全部返回路径只位于三个允许的 RGB split，`pose_data` 命中为 0
- query IDs 全部存在于 gallery；1,870 个 query/gallery 同名副本的 PID/camera 完全一致
- evaluator 人工例验证：距离最近的同 PID/同 camera 副本被排除，跨 camera 正样本成为有效 Rank-1
- 首个真实训练 batch：`[64,3,384,128]`，图像与标签范围全部有限/合法
- DataLoader 元数据：num_query/classes/cameras/views=`2210/702/8/1`

## CUDA / AMP / overflow 门禁

使用正式 batch64、官方 teacher、SGD 与真实 Occluded-Duke sampler 做 24-step CUDA/AMP 检查：

- teacher：`All keys matched successfully`
- 默认 GradScaler 首步 scale=65536；step1-step5 梯度 overflow 并按官方机制依次回退到 2048，均正确跳过 optimizer update
- step6 首次有限更新；step7 再回退一次到 1024；step8-step24 连续 17 次有限更新
- 24 步共 18 次真实 optimizer update；最终 scale=1024
- 每步 loss、score、feature、featmaps 与完整 model state 始终有限；最终 optimizer state 全部有限
- CUDA 峰值 allocated 5,993.5 MiB；检查退出后 GPU 回到 2 MiB / 0%

该 overflow 是 PyTorch GradScaler 可恢复的动态 scale 探测，不产生模型非有限值，也未修改官方训练逻辑。正式 B0 必须保留同一路径，不能通过改初始 scale 或跳 batch 改变配方。

## 一 epoch全链路 smoke

- output：`log/occluded_duke/official_swin_tiny_smoke_e1`
- train：227 iterations，23.782 秒，565.1 samples/s
- e1 eval mAP/R1/R5/R10：`9.2/15.1/25.7/31.1`；仅作链路门禁，不作性能结论
- train/eval/checkpoint 完整结束，进程与 workers 自然退出，GPU 回到 2 MiB / 0%
- e1 checkpoint：211/211 tensor 有限
- 严格日志 `NaN/Inf/Traceback/RuntimeError/OOM/nonfinite/overflow` 与 AMP warning 扫描为 0
- remote exact HEAD/config/tracked source clean

smoke 产物 SHA256：

- checkpoint：`b73e87bf232d69a768df87d981043f9e5633214e8228c2f89aff2dce140d72cc`
- runner stdout：`ba6bec26c67612579af790a8b8c7e8f1df90adde48e959fd55ac7c857d7ef8f2`
- train log：`9a5d2414c167e337ddab454cda94c67f72f08433e03d80311dc3fec31f3819ae`

结论：loader/unit/CUDA/AMP/overflow/eval/checkpoint 门禁全部通过。确认正式 output 不存在、GPU 空闲后，可 fresh 启动 120-epoch Occluded-Duke B0。
