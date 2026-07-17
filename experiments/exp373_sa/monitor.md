# exp373 SA Gate 0 监控

## [2026-07-15] Goal 激活与 Gate 0 启动

- 状态：`GATE0_ACTIVE_INCOMPLETE_NONREPORTABLE`
- 训练进程：无
- GPU 占用：本实验未启动任何训练或推理任务
- 代码改动：仅新增实验设计与监控文档
- 当前判断：继续只读审计

### 已确认事实

1. 现有 PSG/PAA 已在每个启用 stage 的每个 block 后串联，不是只在最终输出层；
2. 多层同步 PSG+PAA 已由 `exp073` 跑过，Stage 2+3 比 Stage-3-only
   低 `0.5 mAP`；
3. matched `exp251/exp254` 中，两阶段 PAA 边际为 `-0.3 mAP / -0.6 R1`；
4. clean PSG stage sweep 不支持“层数越多越好”的普遍规律；
5. 当前 PSG/PAA 对 `[0,1]` heatmap 重复 sigmoid，zero-input 不能当 no-pose；
6. 普通 scale–shift 可归约为 FiLM/SPADE，不能作为新贡献。

### 已定位的可用资产

- 机器：lab-4090（只读检查）
- repo：`/home/afr/SOLIDER-REID`
- checkpoint：
  `/home/afr/SOLIDER-REID/log/occluded_duke/4090_gcn_paa_oa_sd_small/transformer_120.pth`
- checkpoint 大小：约 194 MiB
- config：`Swin-Small / seed1234 / PSG Stage3 / PAA / GCN / OA-SD / equal_concat`
- 原始日志 FINAL：`70.6 mAP / 81.4 R1`
- 限制：不是 pure-global，也不是独立第二 seed；当前只能用于 Gate 0 探索审计。

### 下一检查点

1. 完成 checkpoint/config/commit/SHA provenance；
2. 确认是否存在第二个独立 PAA seed checkpoint；
3. 完成正交残差专项查新；
4. 冻结只读 hook 统计与经验 null 的实现审查；
5. 在上述四项完成前不启动 forward audit。

## [2026-07-15] 新颖性 Gate FAIL，Gate 0 封板

- 状态：`NO_GO_COMPLETE`
- 训练进程：无
- 推理进程：无
- GPU 占用：本实验从未占用 3090/4090
- 实现：未修改训练代码

### 资产核对

在 `lab-3090-d:/root/work/SOLIDER-REID` 找到 exact `exp066`：

- checkpoint：`log/occluded_duke/exp066_paa/transformer_120.pth`
- SHA256：`a084d84995f8fcfd53eea19d8c674d1cdce07d954d9cafbd78e73a211a8903ad`
- execution commit：`8eacaf16dcd797ab8090fe19aca49f80f86bec6a`
- 原始结果：`61.6 mAP / 74.2 R1`
- Stage3 两个 block 均含 PSG/PAA 参数，数据与 pose_data 齐全。

因此停止并非资产或执行阻塞。

### 门禁结果

1. 普通 PSG+PAA 可归约为 FiLM/SPADE；
2. pose-only gate 正交化只是受约束的 FiLM 子集；
3. actual-displacement 正交化的关键 operator 与 arXiv 2025 Orthogonal
   Residual Update 直接重合；
4. CVPR 2023 Shape-Erased VI-ReID、ICML 2026 CoLoRAI Workshop Ortho-ReID
   已覆盖 ReID 中结构/外观子空间与正交补身份表征叙事；
5. 剩余差异只是把已有 operator 放到 PSG/PAA 两支之间，不足以承担主贡献。

### 决策

按 `design.md` 预注册规则正式 NO-GO。不运行 forward fuel audit，不实现、不训练，
不转 transport/routing/adaptive/content-LoRA/普通 FiLM/层数或阈值小变体。
