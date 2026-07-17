# 实验 exp327: DINOv3 / DINOv2-with-registers pose-anchored part-MaxSim 训练-free 天花板 check

> **来源**：post-PRCV「搬范式」路线，exp324（frozen DINOv2-base）的更强/更新特征源探针。
> **性质**：training-free 廉价首验，零训练。**机器**：hyy GPU1（5060 Ti 16G）。
> **决定性问题**：更新/更强的冻结对应模型（DINOv3-B / DINOv2-with-registers-B）训练-free 重遮挡 pose-part mAP 能否抬过 exp324 DINOv2-base 的 **1.86**？

## 动机

- exp324 用 frozen DINOv2-base 验证了机制（pose 锚定 dense token + mutually-visible MaxSim），但天花板低（重遮挡 1.86，训头 e20 到顶 14）。
- DINOv3（2025）与 DINOv2-with-registers 都改进了 dense feature 质量：registers 去掉了 ViT 的 high-norm artifact token（更干净的 patch 特征），DINOv3 用更大数据 + Gram anchoring 进一步提纯 dense 特征。**若更干净的 dense 对应能直接抬训练-free 重遮挡 mAP，则更强冻结源的天花板更高，值得优先上头。**
- 与 exp326（SD-DIFT）并行，从「更新 DINO 系」与「换范式（SD）」两条腿同时探天花板。

## 核心假设

把 exp324 的特征源从 DINOv2-base 换成 DINOv3-vitb16 或 DINOv2-with-registers-base（其余 pipeline 完全不变），训练-free 重遮挡 pose-part mAP **超过 1.86**。

## 技术方案（training-free）

1. **特征源**：`AutoModel.from_pretrained`，frozen，`--model ∈ {dinov3-b, dinov2reg-b, dinov2-b}`。
   - dinov3-b: `facebook/dinov3-vitb16-pretrain-lvd1689m`（patch 16, hidden 768）
   - dinov2reg-b: `facebook/dinov2-with-registers-base`（patch 14, hidden 768, +4 register tokens）
   - dinov2-b: `facebook/dinov2-base`（patch 14，复现 exp324 sanity）
2. **token 切片**：patch token 在 `1(CLS)+nreg(registers)` 之后；nreg 从 config 读取（dinov2-base=0，registers/v3 有值），**assert token 数 == grid_h×grid_w** 防 off-by-register 错位。
3. **几何自适应**：输入宽 = (224//patch)*patch，高 = 宽×2；grid = 输入/patch（patch16→14×28，patch14→16×32）。keypoints 缩放到该 grid。
4. **pose 锚定 5-part + MaxSim**：与 exp324 **逐行相同**（同 PART_GROUPS、POOL_RADIUS=1、mutually-visible part 均值 cosine）。
5. **对照**：(a) CLS / mean-pool 全局 cosine；(b) pose part-MaxSim；(c) grid part-MaxSim。
6. **eval**：ALL query + 重遮挡子集（vis<=8），同 exp324 口径。

## 关键超参及依据

- 输入 ~224×448（patch 整除）：与 exp324 同量级分辨率，保 1:2 行人比，控显存。
- POOL_RADIUS=1（3×3 窗）：与 exp324 一致，单变量只换特征源。
- 主跑 dinov3-b（最新最强）；如 hf-mirror / transformers 版本对 DINOv3 不兼容，回退 dinov2reg-b。

## 预期结果

- 成立：DINOv3/registers 重遮挡 pose-part mAP > 1.86，pose>grid → 更强冻结源天花板更高 → 优先上头。
- 失败最可能：(1) 更干净 dense 特征在 ReID 零样本判别性上仍弱（PCK 强 ≠ ReID 强）；(2) DINOv3 patch16 在 224 输入 grid 更粗，空间分辨率降低抵消特征提纯；(3) transformers 对 DINOv3 输出格式不一致导致 token 切片错位（已加 assert 拦截）。

## 对照组

- baseline = exp324 frozen DINOv2-base 重遮挡 pose-part 1.86。唯一变量 = **特征源**（DINOv2-base → DINOv3-B / registers-B）。
- 用 `--model dinov2-b` 可在本机复现 exp324（验证 slim pose data + 本机 pipeline 与 lab-3090-d 一致）。
- 内部消融：pose vs grid（隔离锚定）。

## Kill-switch / 下一步

- 重遮挡 > 1.86 且 pose>grid → exp327b：该更强冻结源上轻量 part 头，天花板 check vs DINOv2-base 头的 14。
- 否则 → 更强冻结源不优于 base，记录止损（说明天花板瓶颈不在 SSL 模型新旧，而在 frozen 本身）。

## 备注（数据来源一致性）

- pose_data 为 slim npz（keypoints+visibility_binary），由 lab-3090-d 原 exp324 npz 剥 heatmap 生成（仅加速传输，数值与 exp324 一致）。find_pose p0 约定与 exp324 相同。
