# Claude Broad Review — exp326 (DIFT) + exp327 (DINOv3)

**审查范围**：`scripts/exp326_dift.py`、`scripts/exp327_dinov3.py`、两个 `design.md`，与参考脚本 `exp324_dino.py` 逐行对照。性质：training-free eval 脚本（纯推理，不训练，不碰 train.py / 核心 model 代码）。审查由 Opus 子代理在 hyy 远程箱上**实时核验** transformers 5.12.0 / diffusers 0.38.0 源码与 live probe 完成。

## 结论：审查通过（修复后）

子代理对两处最易在远程崩的"新部分"做了 live 核验，均**正确**：
- exp327 register-token 切片 `out[:, 1+nreg : 1+nreg+n_tok]`：DINOv2-with-registers / DINOv3-ViT 的 `last_hidden_state` 布局确为 `[CLS, registers, patches]`，DINOv3-ViT-B = 1 CLS + 4 reg + 196 patch，切片正确；`assert` 正确；`dinov3_vit` 在 transformers 5.12 的 `MODEL_MAPPING_NAMES` 中，`AutoModel.from_pretrained` 可解析。
- exp326 diffusers 用法：up_blocks forward 返回单个 tensor（hook 的 `out` 是 `torch.Tensor`）；`DDIMScheduler.add_noise(original_samples, noise, timesteps)` 签名正确；UNet `get_time_embed` 接受长度 B 的 1-D int64 timestep；VAE `latent_dist.mean * scaling_factor` 确定性编码正确；空文本 embedding expand 正确；probe-then-allocate 几何逻辑可行；patch token 在缓存前 `.float()`，下游 cosine 全 fp32。

## 发现与处置（Critical/High/Medium/Low）

**C1 (Critical) — pose data 缺失会静默给 0.00**：原脚本若 pose_data 不在，`find_pose` 全 None → pose-part mAP 静默 0.00（无报错），且 heavy_mask 全错。
→ **已修**：(1) main() 启动 `assert POSE_DIR/{query,gallery}` 存在；(2) build_reps 中 `n_nopose > 0.5*N` 抛 RuntimeError。并已先把 slim pose_data（4150 query + 24768 gallery）rsync 到 hyy `data/occluded_duke/pose_data/`，启动前会打印 `heavy-occ=X/2210` 供与 exp324 对核。

**H1 (High) — /hy-tmp 28G 盘 disk cache 溢出**：exp326 up_block1 gallery 缓存 ~23GB、exp327 gallery ~10-14GB，强制 `np.save` 会撑爆。
→ **已修**：缓存改为 `--cache` 可选，**默认关闭**，特征全部留 503G 主存（float16，host RAM 480G free，绰绰有余）。日志打印 in-RAM GB 量。

**H2 (High) — DINOv3 是 gated 模型**：`facebook/dinov3-*` 需接受 license + token，匿名 `from_pretrained` 可能 401/404。
→ **处置（运行策略）**：exp327 先跑 `--model dinov2reg-b`（ungated，registers 干净，正是 apples-to-apples 升级），再试 `dinov3-b`（失败则回退，design.md 已写）。

**M1 (Medium) — exp327 fp32 backbone**：exp327 未传 torch_dtype（fp32），exp326 fp16。fp32 更安全且 16G 装得下，patch token 显式 `.float()` 后下游全 fp32 → 非 bug，仅精度记录。先用 `--model dinov2-b` 复现 exp324 1.86 验证 slim-pose pipeline 一致性。

**M2 (Medium) — exp326 holistic 基准是 mean-pool 非 CLS**：SD 无 CLS，exp326 正确改用 mean-pool。决定性的 pose-part 绝对 mAP vs 1.86 不受影响（独立绝对数）；只是 exp326 的"相对 holistic 增益"不可跨脚本与 exp324/327 直接比 → results.md 注明，无需改码。

**M3 (Medium) — DIFT hook 可能读到 stale 特征**：原无 `assert 'x' in self._feat`。
→ **已修**：每次 forward 前 `self._feat.clear()`，forward 后 `assert 'x' in self._feat`。

**Low**：未用 `import sys`（无害）；`find_pose` p0 lexicographic 排序与 exp324 一致；`(0,0)` sentinel `cx<=0 and cy<=0` 与 exp324 一致；grid 行序 y-major 与 scale_kp 一致（无转置 bug）。均与 exp324 逐字节一致，comparable。

## Comparability 判定

PART_GROUPS / POOL_RADIUS=1 / HEAVY_OCC_THR=8 / find_pose p0 / part_maxsim（mutually-visible、per-part L2-norm cosine、no-common→2.0）/ eval_func（Market metric）/ dinov2-b 几何（224×448, grid 32×16）**全部与 exp324 一致**。`--model dinov2-b` 应复现 exp324 1.86（modulo M1 dtype）→ **先跑此 sanity 验证 hyy 上的 slim-pose pipeline，再信 dinov3/DIFT 数字**。

## "是否只是 config swap" 质疑

两者是**刻意的特征源替换探针**，整个下游 pipeline 与 exp324 逐字节保持一致，唯一隔离变量 = 特征源，kill-switch 明确（>1.86 才上头），负结果本身有信息量（天花板瓶颈在 frozen 而非模型新旧）。属合法探索，非逃避创新。审查通过。
