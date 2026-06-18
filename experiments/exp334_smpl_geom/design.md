# 实验 exp334: SMPL 几何当空间先验（occluded ReID）

## 背景：为何不再用 β 当特征
exp333 证 SMPL 体型 β 当**全局身份特征** = 随机（0.18% mAP），且任何 shape-派生特征都是 β 的线性函数 → 都随机。用户洞察（LGPA→LGPA-D）：**换用法可能翻盘**。SMPL 真正不同、不受 β=随机 影响的价值 = **几何空间先验**（"身体在哪/什么结构"），不是身份描述子。

## 已解锁基础设施
ROMP（去掉 `--calc_smpl` store_false 陷阱 + 手装 SMPL_NEUTRAL.pth）现输出 `pj2d_org(71,2)` 2D 投影关节、`joints(71,3)`、`verts`。缓存 `cache/smpl_geom/{train,gallery,query}.npz`（pj2d + j3d + valid，目标人=最居中检测）。详见 [[smpl-infra-on-lab3090]]。

## 核心假设（一句话）
SMPL 给出**完整身体的 2D 关节**（遮挡处由人体先验补全，ViTPose 在遮挡处会缺失/低置信）；用这套**完整骨架**生成 patch 网格上的身体-存在热图，对 backbone token 做空间门控（强调身体区、压低背景），能帮弱 baseline——因为它提供了一个**遮挡下仍完整**的身体结构先验。

## 诚实的机制风险（写在前面）
1. **遮挡处 location≠visibility**：SMPL 说"身体应在这"，但遮挡处那块 patch 其实是遮挡物像素。所以"往完整身体处 attend"在遮挡部位会 attend 到遮挡物。→ 第一版用**软**门控（+bias 而非 hard mask）缓解；真正解需 visibility（见下）。
2. **与 PSG 概念重叠**：项目 PSG 已用 ViTPose 做 pose-spatial-gate（强栈 73 mAP）。本实验在**弱 TransReID** 上验"SMPL 几何是否帮得上"，是干净的隔离测试，不是抢 PSG。
3. **SMPL 独特价值的真正测法**（follow-up）：同 baseline 上 **SMPL-gate vs ViTPose-gate**——SMPL 完整（遮挡下不缺）vs ViTPose 退化。若 SMPL>ViTPose → 完整几何的价值被证。

## 技术方案（exp334a，第一版）
- backbone = 弱 TransReID ViT-base（同 exp333，有 headroom）。
- SMPL 空间门控：pj2d(71,2) 归一到 [0,1] → 在 16×8 patch 网格上 Gaussian-splat 71 个关节 → 每 patch 身体-存在分 `h∈R^128`（valid=0 图 h=0，退化为 baseline）。
- 注入：backbone 出 patch tokens `(B,128,768)` → 身体-加权池化 `f_body = Σ_p softmax(h)_p · token_p` → 与 cls 特征 concat 或加权和。**软门控**（不 hard-mask）。
- 单变量 vs baseline；valid=0 时模块自动退化 → 只在有 SMPL 的图起作用。
- 损失：同 baseline（ID+triplet），body 特征加独立 ID+triplet。融合在测试期 concat（alpha 扫）。

## 预期 / 判据
- 成立：弱 baseline +X mAP（X>0），重遮挡子集涨更多。
- 失败最可能：① 遮挡处 attend 到遮挡物（location≠visibility）抵消增益；② 弱 baseline 已靠 bbox crop 聚焦人体，body 先验冗余。
- **若 exp334a 中性/负** → 上 exp334b：SMPL 完整骨架 + ViTPose visibility 融合的 visibility-aware 门控（SMPL 补全 + ViTPose 判可见）。

## 对照组（Claude review 修正）
- **必须训练 exp334 自己的 `--use_geom off` 臂**当对照（≈53.09 sanity）——**不能直接复用 exp333_baseline**：exp334 的 body 分支**回传进共享 backbone**（exp333 的 β 分支是 detached，故 exp333 alpha=0 == baseline；exp334 不成立）。
- **alpha=0 是诊断量**（body loss 对 backbone 的正则效应），**不是 appearance-only baseline**，results 里单列。
- 二级对照（后续）= ViTPose-gate（验 SMPL 完整性的增量）。

## 口径修正（Claude review）
1. **A/B headline = geom-on best-alpha vs geom-off 自训对照**（两臂同 seed 同脚本）。
2. **crop 错位**：train transform 有 Resize→Pad(10)→RandomCrop，热图用未裁剪归一坐标 → 训练期 ≤~0.6 patch 零均值随机错位；测试无 crop 对齐。属可接受软先验噪声（Resize 均匀，按原图 W,H 归一正确），但记录在案防误归因。
3. **重遮挡子集单列**：location≠visibility 风险（遮挡 patch 被 body-pool 进遮挡物 token）是判据核心，必须分层报 mAP。
4. 代码经 Claude review：_tokens 正确复制 forward_features、body-pool fp16 无 NaN、单变量隔离成立、优化器覆盖新参数。

## 机器
lab-3090-d（ControlMaster 持久连接，conda env solider-reid 训练）。
