# exp387 监控：官方干净代码上的最小 TAPF D0

## 当前状态

- 状态：DESIGN
- 直接对照：exp385 official clean B0 e120=`57.4/67.4/80.6/85.2`
- pose provenance：exp386 final manifest SHA256=`cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8`
- exp386 extraction/loader/paired augmentation/RGB parity/DataLoader CUDA：PASS
- 4090：空闲
- 正式 D0：未实现、未启动

下一步严格按 design 实现独立 TAPF 模块、默认关闭接线与可执行门禁。所有 Gate PASS 前不得创建正式 output 或启动训练。
