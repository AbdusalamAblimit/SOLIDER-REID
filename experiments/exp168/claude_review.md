# Claude Review: exp168 (17-token per-token + PLBOA)

## 审查通过

### 检查项

1. **代码支持**: `POSE_STR_NUM_PARTS: 17` 已在 structural_routing.py 中完整支持
   - `_compute_part_centroids` 有 `num_parts == 17` 分支（每个关键点即 centroid）
   - `pose_to_part_bias`: Conv2d(17, 17, 1) — 正确

2. **Per-token 路径**: 17 tokens 时 for 循环正确迭代 17 次
   - CE: 18 个 score (1 global + 17 per-token)
   - Triplet: 18 个 feat (1 global + 17 per-token)
   - `use_norm = len(feat) > 3` → True → L2 normalize before triplet ✓

3. **Test feature**: K_str=17 → else 分支 → `structural_tokens.mean(dim=1)` → 简单 mean pooling
   - 注意：17-token 没有 confidence-weighted pooling（只有 K==6 才有）
   - 这是合理的：17 个单关键点没有 body-group 的热图聚合概念

4. **配置**: 单变量变化，与 exp166 只差 POSE_STR_NUM_PARTS
5. **内存**: 17 × per-token CE + triplet 略增，但应在 3090 24GB 内
6. **兼容性**: 不影响任何其他实验

### 注意

- Test feature 是 simple mean pool (1536-d)，不是 confidence-weighted pool
- 这与 exp166 的 confidence-weighted pool 略有差异（K==6 vs K==17 代码路径不同）
- 但两者都产生 1536-d test feature，可以比较
