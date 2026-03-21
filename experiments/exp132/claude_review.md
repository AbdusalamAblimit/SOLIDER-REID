## exp132 代码接线审查

### 1. 审查结论：**允许启动**

### 2. HIGH（阻塞项）：无

### 3. MEDIUM（非阻塞项）：

- **L273 梯度路径**：`mixed_dist = (1-alpha)*global_dist + alpha*base_dist`，其中 `global_dist`/`base_dist` 均由 detached 特征计算，本身无梯度。梯度仅通过 `alpha→ltcs_head` 流动。这是设计意图，但意味着 **LTCS head 只学"融合权重"，不反向影响 backbone/GCN**。如果期望 LTCS 信号回传到特征提取器，当前设计做不到。确认这是有意为之即可。

- **`pair_adaptive_fusion.py` 设备问题**：`_compute_ltcs_loss` 在 GPU 上运行（特征已在 GPU），`build_pair_descriptors` 全部 tensor 操作，无显式 `.to(device)`。但因为输入全在同设备上，无风险。test-time `_compute_structured` 中有显式 `.to(head_device)`（L269），安全。

### 4. 结论

| 检查项 | 结果 |
|--------|------|
| head 是否进 checkpoint | **是**，`ltcs_head` 是 model 子模块，`model.state_dict()` 自动包含 |
| test 是否真调用 head | **是**，`POSE_TEST_FEAT=cvk_adaptive` → `_compute_structured` → `pair_fusion_head`（L249-275），且 L964/991/1036 三处均赋值 `evaluator.pair_fusion_head` |
| 设备/内存风险 | **低**，head 仅 3 层 MLP（6→32→32→1），参数 ~1.2K；test-time 按 chunk=256 处理，无大矩阵风险 |
