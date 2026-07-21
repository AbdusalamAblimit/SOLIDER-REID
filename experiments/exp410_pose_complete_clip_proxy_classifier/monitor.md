# exp410 PC²P 监控

## 2026-07-21：对象重置与设计冻结

exp409已永久封板为`SEALED NO-GO / RANK-1 PASS / mAP FAIL`；4090恢复`2 MiB/0%/0 compute PID`。
三路独立审计比较了pose×CLIP语义缺失协方差与pose-complete classifier对象，最终选择PC²P：它把作用对象从
单个hard pair改为全部702类分类几何，更直接回应exp409只改善R1、不改善mAP的结果。

初稿中的可学习`Q:768→768`被独立审计判为科学HIGH并删除：由于类别数702小于特征维768，`P @ Q`几乎可表达
任意learned classifier，既会让proxy机制退化，也增加推理成本。冻结方案无Q/adapter/projection：
`BN(global_feat) @ frozen_pose_complete_proxy.T`直接替换learned classifier，原triplet和eval global descriptor不变。

近期近邻审计确认CLIP-ReID已有冻结身份text feature的I2T CE，ProFD已有part prompt/centroid/memory，普通固定proxy
分类本身不能声称新颖。PC²P只保留C类窄主张：pose五槽跨同PID多图补全的visual identity-set proxy、无adapter
替换learned classifier、测试期恢复原global descriptor。问题门PASS、证据门PASS、机制门CONDITIONAL PASS。

当前状态=`DESIGN/PROTOCOL FROZEN / IMPLEMENTATION NEXT / GPU IDLE`。下一步实现fresh bank builder、严格loader和
最小model接线；必要合同及一次独立智能体盲审`0B/0H`后立即fresh运行，不增加无穷preflight。
