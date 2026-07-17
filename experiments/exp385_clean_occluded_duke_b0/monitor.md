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
