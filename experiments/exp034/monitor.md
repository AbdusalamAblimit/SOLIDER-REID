# exp034: Target-Aware Dataloader — 监控日志

## 实现

修改了 `datasets/pose_dataset.py` 的 `_load_persons()` 方法：
- 读取 `target_person_idx`（exp033 已写入 index.json）
- 如果 target 不是 person 0，将其移到列表第一位
- 向后兼容：无 `target_person_idx` 字段时默认使用原始顺序

## 改动量
- 仅修改 `datasets/pose_dataset.py`，增加 6 行代码
- GCN 分支代码无需修改（已使用 `keypoints[:, 0, :, :]`）
- PSG 使用 scene-level merged heatmaps（MAX over all persons），不受 person 顺序影响

## 影响范围分析

| Split | 受影响图片 | 占总图片比例 |
|-------|-----------|-------------|
| train | 76 / 15618 | 0.49% |
| query | 38 / 2210 | 1.72% |
| gallery | 67 / 17661 | 0.38% |

## 是否需要训练验证

**决定不单独跑训练验证**，理由：
1. 受影响图片仅占训练集 0.49%（76 张）
2. 这些图片的 target margin 很小（mean=0.052），两个选择接近等价
3. 预期性能变化远小于训练方差（3-seed std = 0.47%）
4. 此改动是代码正确性修正，不是模型改进
5. exp035 的训练会自动使用 target-aware 代码，其结果自然包含此改动

## 验证

- [x] 代码修改后 dataset 可正常加载（返回正确形状的 tensor）
- [x] 单人图不受影响（target_person_idx = 0）
- [x] 多人图 target 正确移到 index 0（通过 disagreement case 验证）
- [x] 向后兼容：无 target_person_idx 字段时行为不变

## 结论

这是一个纯代码修正，不改变模型或训练配置。性能影响预计在噪声范围内。真正的验证将在 exp035 中体现。
