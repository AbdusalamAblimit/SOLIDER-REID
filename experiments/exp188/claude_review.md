# Claude Broad Review: exp188 OA-SD with EMA Teacher (Opus 4.6)

## 审查通过

### 第一轮（无 EMA）发现的 issues
- [High] OA-SD 无 PLBOA 无 warning → **已修复**（line 405-406 添加 warning）
- [Medium] Teacher/student 独立 RE → 可接受（增加多样性）
- [Medium] pose_dict 共享 → 可接受（visibility 用于 PSG 不是 distillation）
- [Low] OA-SD + PCVT/parallel_aug 静默禁用 → 已记录

### 第二轮（EMA）审查
1. **deepcopy** (line 399): 正确。解包 model.module 后 deepcopy，独立参数副本。
2. **eval mode** (line 401): 正确。BN 用 running stats，dropout 禁用。
3. **EMA update** (line 672): `t.mul_(0.999).add_(s, alpha=0.001)` 标准公式，in-place，torch.no_grad。
4. **Memory**: student forward ~18GB + teacher params ~112MB + teacher forward ~2GB = ~20GB。3090 24GB 够。
5. **PLBOA warning**: 已添加（line 405-406）。
6. **DDP 隔离**: ema_teacher 不参与 DataParallel。EMA update 正确解包 model.module。

### Medium 观察（不 blocking）
- BN buffer (running_mean/var) 未 EMA 更新（teacher 用初始 deepcopy 的 frozen stats）
- 标准 EMA 实现通常跳过 buffers，可接受
- 如果后期 distillation 质量下降，可考虑添加 buffer EMA

### 后向兼容
OA_SD=False 时：ema_teacher=None，所有 OA-SD 代码短路，行为不变。

### 显存估算
- Student: model weights ~112MB + forward activations ~8GB + backward ~8GB = ~16GB
- EMA Teacher: model weights ~112MB + forward activations ~2GB (no_grad) = ~2.1GB
- Optimizer states: ~500MB
- Total: ~18.7GB — 3090 24GB 充裕
