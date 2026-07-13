# exp371 Gate C：target-only support cache 设计

## 目的与证据边界

本工具只提取冻结 `exp336` LGPA checkpoint 的 target-only 描述子与 pose-response 元数据，不构造 support、不计算 oracle，也不启动训练。它解决 Gate C 的输入归因问题：后续所有 support-routing arm 必须复用同一份 teacher features，只允许改变 donor/slot 的组织方式。

缓存不能证明 CASD 有效，也不能把 pose response 称为绝对可见性。`CLIPPartHead.kp_weights` 是五个 slot 在单图内归一化后的相对分配；目标整体很弱时仍会和为 1。

## 强制协议

1. `POSE_LGPA=True`、`POSE_LGPA_DETACH=True`；
2. `POSE_USE_TARGET_HEATMAP=True`，LGPA 只读取 person-0 target heatmap；
3. 关闭 GCN/PPA/VCSR/structural routing，输出必须恰为 `7×768`；
4. 同一 batch、同一 checkpoint、同一图像/pose 输入依次提取：
   - `equal_concat`：`global + pooled + 5 slots`；
   - `maxsim_hybrid`：`global + 5 kp_feats + kp_weights`；
5. `maxsim_hybrid.kp_feats` 必须逐元素匹配 `equal_concat[:, 2:7]`，global 也必须匹配；不匹配即中止；
6. 不加载 classifier，不拟合投影，不改模型权重。

## Flip 处理

当 `TEST.FLIP_TEST=True` 时，原图和水平翻转图分别前向：

- `equal_concat` 的七个 block 先逐视图输出，再平均并逐 block L2 归一化；
- `maxsim_hybrid` 的 global/五个 slot 先平均，再分别 L2 归一化；`kp_weights` 做算术平均；
- raw pose response 在原图和 flip 后的 target heatmap上分别计算，再做算术平均。

`flip_batch` 会翻转 heatmap 宽度并交换 COCO 左右关键点。LGPA 的五个 `PART_KPS` 都同时包含左右侧，因此理论上 raw response 在 flip 前后不变。工具仍显式计算两次、记录 `raw_flip_max_abs_diff`，并以平均值落盘；超出容差即中止，不静默选择一侧。

## Pose-response 定义

对 target person heatmap `H∈R^(17×h×w)`，先 resize 到最终 LGPA feature-map 分辨率，再按 `PART_KPS` 聚合：

```text
raw_response[k] = mean_hw(max_{joint in PART_KPS[k]} H[joint])
relative_allocation = raw_response / sum_k(raw_response[k])
```

缓存同时保留：

- `raw_pose_response`：未跨 slot 归一化的五维响应；
- `relative_allocation`：模型 `maxsim_hybrid.kp_weights` 的 flip 平均；
- `raw_response_relative_allocation`：由 raw response 重新归一化的审计副本。

后两项应数值一致，但都只能称为 **relative pose-response allocation**，不能称 absolute visibility。

## Cache schema

每个 split 输出一个原子写入的 `.pt`：

- `features`: `[N, 7×768]`；
- `kp_feats`: `[N, 5, 768]`；
- `relative_allocation`: `[N, 5]`；
- `raw_pose_response`: `[N, 5]`；
- `raw_response_relative_allocation`: `[N, 5]`；
- `target_person_valid`: `[N] bool`，只来自 `person_mask[:,0]`；
- `person_count`: `[N] int64`；
- `pids/camids/paths`；
- `split/num_query/block_dim/schema_version/pose_source`；
- checkpoint、脚本、`PART_KPS` 和各 tensor 的 SHA256；
- flip、一致性误差和样本计数审计。

`target_person_valid=False` 的样本保留在缓存中，raw response 必须为零；后续 Gate C 必须显式报告并决定是否排除，不能在提取阶段静默丢弃。

## 后续使用限制

1. Gate C 必须先做 strict-path / near-duplicate 审计；本缓存不负责 LOO；
2. extraction factor 与 routing factor 必须分开，不能重新提取 teacher features来制造 control；
3. `relative_allocation` 可以用于 `R=pose-response/equal/permuted`，但不能宣称是真实遮挡 visibility；
4. 只有 pose-response routing 明确超过 identity mean、equal routing 与 permutation，才允许保留 pose-organized claim。
