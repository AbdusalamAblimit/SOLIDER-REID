# exp253: Multi-Stage PSG (Stage 1+2+3, 无 PAA) on Tiny LGPA-D+GCN

## 动机
exp251 测试了 Stage 2+3 PSG + PAA，与 baseline 持平 (65.2 vs 65.5)。
但无法分离 multi-stage PSG 和 PAA 的各自贡献。
本实验：纯 multi-stage PSG（无 PAA），并扩展到 3 个 stage (Stage 1+2+3)。
目标：验证 "更深层级的 hierarchical pose injection" 是否有额外收益。

## 核心假设
Stage 1 的低层特征 (192 dim, 48x16 spatial) 也能受益于 pose 空间门控。
3-stage PSG 比 2-stage 或 1-stage 提供更全面的 pose 引导。

## 技术方案
- Config: POSE_PSG_STAGES=[-3,-2,-1] (Stage 1+2+3)
- 无 PAA (POSE_ADDITIVE_ADAPTER=False)
- 其余与 exp246b 相同：Tiny LGPA-D+GCN+OA-SD+PLBOA

## Swin-Tiny stage 结构
- Stage 0: 96 dim, 2 blocks, 96x32 spatial (跳过)
- Stage 1: 192 dim, 2 blocks, 48x16 spatial → 2 PSG modules
- Stage 2: 384 dim, 6 blocks, 24x8 spatial → 6 PSG modules  
- Stage 3: 768 dim, 2 blocks, 12x4 spatial → 2 PSG modules
- 共 10 PSG modules (vs baseline 2 modules)

## 代码修改
无。仅 config 参数。

## 对照组
- exp246b (Tiny, Stage 3-only PSG): 65.5/77.2
- exp251 (Tiny, Stage 2+3 PSG + PAA): ~65.2/?? (进行中)
- exp000 baseline (no pose): 56.6/66.5

## 预期结果
- 成功: ≥ exp246b (65.5) — 3-stage PSG 有额外收益
- 中性: ≈ exp246b — 更多 stage 不帮忙但不伤害
- 失败: < 64 — 过早注入 pose 干扰低层特征学习
