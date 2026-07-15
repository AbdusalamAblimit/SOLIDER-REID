# exp376 Codex 启动前审查

## 审查方式

按用户明确要求，本实验禁止使用 Claude，改由两个独立 Codex 审查代理分别进行代码审查与
科学/归因审查。首轮审查发现的问题全部在正式训练前修复，修复后两路均复审。

## 首轮代码审查

初始结论：`BLOCK`。

阻塞问题：A/B 的三维 basis bank 被整体执行 Kaiming 初始化，PyTorch 会把额外维度计入
fan-in，使 B basis 尤其被缩小约 `sqrt(C)`。真实 canonical 输入下，raw delta RMS 仅约
`3.96e-4 / 1.85e-4`（C=384/768）；再乘 `α=1e-3` 后 FP16 几乎完全等于 identity。

同时指出：训练器没有读取 8 层运行统计；缺真实 CUDA AMP/GradScaler/optimizer/batch64 smoke；
heatmap batch 可能静默广播。

修复：

1. 每个二维 `A_m/B_m` 独立初始化；
2. 增加 coefficient/visibility 两路 batch fail-fast；
3. 每 50 iter 写入 alpha、visibility、coefficient abs mean、delta RMS；
4. 新增 production CUDA AMP、GradScaler、8 层关键梯度、optimizer update、changed fraction、
   strict reload 与 peak-memory preflight；
5. 本地修复后 canonical float32 校准：C=384/768 raw delta RMS 提升为
   `6.74e-3 / 1.01e-2`，applied residual RMS 为 `6.74e-6 / 1.01e-5`，分别约 61.8% / 60.1%
   元素改变，不再是原初始化死模块。

复审结论：静态 `GO`；仍须在训练机显式以 batch64 跑 GPU preflight。

## 首轮科学审查

初始结论：`NO-GO AS WRITTEN`。

阻塞问题：

1. exp071 的有效矩阵本来就是 `W_up·diag(f(P))·W_down`，不能声称 exp376 首次产生动态算子；
2. 显式 visibility gate 会混淆“正确前景门控”与“动态 A/B factorization”贡献。

修复：

1. 将创新边界收紧为 factor-wise A/B basis mixture，是 exp071 diagonal modulation 的泛化；
2. 新增 D0：相同 8 blocks、相同 visibility、A/B projection 参数精确匹配的 exp071-style control；
3. 冻结反事实拆成 correct/matched coefficient 与 correct/matched visibility 两个正交变量；
4. exact-commit B0、同机 M0、exp376 专用 donor nuisance preflight 和多 seed 被列为正式结论前置。

复审结论：`GO`。首轮允许 4090 P0 与 3090 D0 并行筛查。

## 启动裁决

`CONDITIONAL GO`：代码与科学审查均通过；只有两机 production model smoke 和 batch64 CUDA AMP
preflight 完整通过后，才可正式启动训练。
