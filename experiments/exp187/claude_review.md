# Claude Broad Review: exp187 Parallel Aug + SupCon (Opus 4.6)

## 审查通过（第二轮，修复 Critical 后）

### 第一轮发现的 Critical Bug（已修复）
PLBOA/PGMPOA 在 parallel_aug 代码路径中被绕过——3 个 view 都不会有 PLBOA。
修复：将 PLBOA/PGMPOA 移到 parallel_aug 分支之前，让所有 view 基于已 PLBOA 处理的图像。

### 修复验证
1. PLBOA 现在在 `if self.pcvt ... elif self.parallel_aug ... else:` 之前执行
2. 所有 3 个 view 共享 PLBOA 处理后的 img
3. standard path 中的重复 PLBOA 代码已删除（避免双重应用）
4. PGMPOA 同样在分支前执行

### OOM 风险
3x forward graph 同时在显存。AMP fp16 下预估：
- 每个 forward ~5-6GB activations
- 3x = ~15-18GB + model params ~2GB + optimizer ~2GB = ~19-22GB
- 3090 24GB 可能刚好够，如果 OOM 需改 gradient accumulation

### SupCon 在 parallel mode 的行为
- 每个 view 独立调用 loss_fn → 独立计算 SupCon
- 3 个 SupCon 损失平均 → 更多样的 contrastive signal（不同增强下的同 ID pairs）
- 审查建议将来可优化为 cross-view SupCon（3B 合并），但当前独立模式也有效

### 配置
CLI: 在 exp176 配置上增加 MODEL.POSE_PARALLEL_AUG True + MODEL.POSE_STR_SUPCON_TEMP 0.05
OUTPUT_DIR: exp187_parallel_supcon

### 单变量
vs exp176: 仅增加 POSE_PARALLEL_AUG=True

### 后向兼容
PLBOA 移位不影响 single-view 路径（standard else 分支仍正确执行，PLBOA 只是提前了）。
