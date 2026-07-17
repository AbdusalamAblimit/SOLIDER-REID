# 实验 exp359-BLC: Bbox-Lattice Canonicalizer（训练端 input-level 备选）

> **状态**：备选 design（2026-06-26 预写）。仅当 LSRC（对称 + 非对称）都 go/no-go 失败时启动。预写以备无缝接 + 体现"实验前先写 design"纪律。

## 动机

- LSRC 走 backbone-loss 路线。codex train2_input 收敛到 **BLC（input-level，8/10→market 受限 6.5）**作互补/备选。
- **最对症的证据**：LM-S4 因子消融显示 **bbox 检测框不确定性主导（+2.84）**，远超 phase/kernel。→ 直接 canonicalize bbox 的 refiner 比改 resize kernel 更打中主因。
- 改输入（像素分布）不是 frozen-feature tweak，backbone 参与重学。

## 核心假设

训一个 tiny crop refiner，把带 bbox/phase 扰动的 crop 重采样成 canonical crop，**削弱 bbox 主因子**，剩下的 phase/kernel/残差留给 test-time K=9 decision marginalization。目标**不是**让 K 个 lattice 特征一致（那是已证伪的 consistency），是把错误 crop 校正到更可靠的 person support。

## 技术方案

- **数据流**：margin-expanded / padded crop → tiny localization net 预测 bounded `(dx, dy, ds)` → `grid_sample` 得 canonical crop → backbone → ReID。
- **loss**：`L = CE + Triplet on f(grid_sample(x_z, θ̂)) + λ_geo·SmoothL1(θ̂, -δ_z) + λ_reg·||θ̂||²`
  - `δ_z` 来自自生成的 bbox/phase lattice → 几何监督干净（压偏移参数，非身份 embedding）。
  - 第一版只做 translation，不 affine/TPS（防裁掉身体边缘）。
  - K-spread 只 monitor，不进 loss。
- **⚠️ market 约束（codex 警告）**：market 图是已裁好的人框，无原图上下文 → 要先 **pad crops 人工制造 bbox 不确定性**，否则 refiner 没有可校正的 support 偏移。这是 BLC 在 market 上先天降到 6.5 的原因。

## 预期结果

- 假设成立：offset MAE `<0.35` LR px，bbox-axis feature spread 降 ≥20%（identity margin 不降），K=1 mAP `+0.8` 或 K=9 在 lattice-marg 上再 `+0.3~0.5`。
- 失败最可能原因：(a) STN 学偏 / 裁掉身体边缘 / 过拟合检测框（→ 加 offset 范围 + identity loss + grid reg）；(b) market pad-crop 制造的不确定性不真实 → spread 降但 mAP 不涨（直接杀）。

## 对照组

- baseline：no-LM-loss（lattice 79.90 / single ~76.9）。
- 强对照：固定 margin crop（不学 refiner）→ 证 learned canonicalization 必要。
- 消融变量：只加 canonicalizer（冻 backbone 先 probe → 活则 unfreeze）。

## kill-switch（冻 backbone 先 probe）

冻 no-LM-loss backbone 只训 canonicalizer，h=12/16 bbox-only lattice：
- offset MAE <0.35 LR px；
- K=1 mAP +0.8 或 K=9 +0.3~0.5；
- h=32/HR sanity 掉 <0.3；
- θ̂ 不大量贴边（saturation <10%）；
- **只降 spread 不涨 mAP → 直接杀**（别陷入"指标好看但无用"）。

## 与已死路线区别
压几何偏移参数（非身份 embedding）→ 非 consistency；几何监督来自自生成 lattice（干净）→ 非弱监督瞎学；test 仍保留 K=9 marginalization → canonicalizer 只削 bbox 主因子，不替代边缘化。
