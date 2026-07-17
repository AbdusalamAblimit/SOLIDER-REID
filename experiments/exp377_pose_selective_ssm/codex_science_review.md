# exp377 科学与实验设计审查

## 审查结论

**结论：`GO_FOR_IMPLEMENTATION_AND_SMOKE`。**

本轮放行的主假设是：在固定的双向 serpentine 视觉序列上，当前图像的 target-person pose
同时条件化原生 selective SSM 的 `Delta / B / C`，从而改变视觉证据的状态时间尺度、写入映射
与读取映射。它是一个完整的 bundled mechanism；首轮不要求先拆成 `Delta-only` 才能训练。

该 GO 只允许实现、单测、真实模型 smoke 和预注册 Gate A。它不等于论文创新已经成立。
只有 P0 同时通过训练性能、参数匹配控制和冻结反事实门禁后，才能进入多 seed 与论文归因。

## 当前最小机制

建议在 Swin 最终空间 token 与 GAP 之间只插入一个 selective SSM residual block。扫描顺序固定为
serpentine 及其严格逆序，两向使用同一组参数并对输出做固定融合；本实验不学习扫描顺序，不加入
骨架图、身体状态槽、额外 loss、PAA、PSG、PRSM、LGPA 或 GCN。

对 RGB token `x_t`，基础 selective 参数仍由视觉内容产生：

```text
u_t       = W_in LN(x_t)
delta_x,t = W_delta u_t
B_x,t     = W_B u_t
C_x,t     = W_C u_t
```

pose 分支只对三个原生 selective 参数提供有界残差：

```text
Delta_t = softplus(delta_x,t + rho_delta * g_delta(P_t))
B_t     = B_x,t + rho_B * g_B(P_t)
C_t     = C_x,t + rho_C * g_C(P_t)
A       = -exp(A_log)

h_t = exp(Delta_t A) * h_(t-1) + discretize_B(Delta_t, B_t, u_t)
y_t = C_t * h_t + D * u_t
```

`A`、`D`、scan permutation 和输出融合不得读取 pose。实现必须真实经过状态递推及 `Delta`
离散化；不能用逐 token 仿射、普通 MLP、GRU gate 或 exp375 的 slot update 冒充 Mamba selective
update。pose residual 建议零初始化或小初始化，保证 P0/M0/D0 能从同一视觉 SSM 起点比较。

## 与 exp375 / exp376 的边界

exp375 PRSM 使用 6 个手工身体槽，pose 直接决定 RGB candidate 写到哪个槽以及写入多少；它没有
RGB-selective `B/C`，也没有连续 SSM 的 `exp(Delta A)` 与一致离散化。exp377 不建立身体槽，
RGB 内容先产生基础 `Delta/B/C`，pose 只作为外生条件修改原生 selective dynamics。因此它不是
把 PRSM 类名换成 Mamba。

但二者都涉及状态保留与写入，不能宣称完全没有机制邻接。特别是 pose-conditioned `B` 与 PRSM
的 pose-guided write 在高层语义上相近；这正是 D0、M0、matched pose 和 support/composition
拆分控制不可省略的原因。exp375 的负结果应作为 exp377 的强失败先验，而不是从论文中隐藏。

exp376 验证的是不跨 token 传播的 post-block dynamic low-rank residual；exp377 验证的是跨序列
递推的状态转移。两者问题对象不同，但 pose-conditioned `B/C` 仍可能退化为普通动态算子。
如果 full P0 成功，`Delta-only` 必须作为后续关键消融，判断收益来自状态时间尺度还是再次来自
pose-conditioned 写入/读取矩阵。**`Delta-only` 是成功后的归因要求，不是当前训练阻塞项；full
P0 若失败，也不得用 `Delta-only`、`B-only` 或 `C-only` 作为救场 sweep。**

## 训练对照

1. **B0：clean Swin**
   - 无 SSM；
   - exact execution commit、同数据、seed、batch、优化器、运行时；
   - 回答加入任何状态模块是否优于原始 global baseline。
2. **D0：RGB-only selective SSM**
   - 与 P0 相同 state dimension、scan、双向融合、残差路径和视觉 `Delta/B/C`；
   - pose residual 强制 bypass；
   - 尽量保留相同 state dict，使 P0 checkpoint 可在同一模型实例中切换 D0 mode；
   - 回答普通 Mamba 容量，而不是 pose 的价值。
3. **M0：canonical-pose selective SSM**
   - 与 P0 参数、初始化和训练配方完全相同；
   - 所有样本使用同一个固定 canonical pose 同时条件化 `Delta/B/C`；
   - 回答固定人体布局或静态空间先验。
4. **P0：instance-pose selective SSM**
   - 只使用当前 target person 的 heatmap；
   - pose 同时条件化真实 `Delta/B/C`；
   - 其余路径与 M0 完全一致。

跨 3090/4090 的结果只能筛查趋势。e60 早停可使用预注册的 exp375 同一 4090 clean B0 曲线，
因为 production integration 已证明 exp377 B0 与其 state dict、descriptor 和最终 featmap
逐元素相同；该旧 B0 不得用于最终论文差值。任何正式 GO 仍必须补同一机器、同一解释器、
同一 commit 的 B0/D0/M0/P0。D0 是最重要的直接容量控制。

## 冻结 checkpoint 反事实

P0 checkpoint 至少需要以下 arm，并在同一模型实例、同一 RGB/path/PID/camera 顺序下运行：

1. `correct-start`；
2. target-matched shuffle；
3. recipient support/amplitude + donor joint composition；
4. donor support/amplitude + recipient joint composition；
5. joint-channel permutation，保持逐像素总量和空间 support；
6. canonical pose，仅作诊断；
7. pose residual off，即 D0 mode；
8. `correct-end`。

matched donor map 必须针对 exp377 实际进入 `g_delta/g_B/g_C` 的输入与输出做专用 nuisance
preflight，至少覆盖多尺度 support、幅度、关节组成、三个 pose residual 的均值/范数和 Delta
分布；不能直接继承 exp375 的 PRSM write-profile PASS。correct-start/end 必须逐指标及 descriptor
精确复现，各有效 intervention 的 descriptor 必须确实改变。

本实验不应在 SSM 输出外再乘 pose visibility gate。如果实现保留任何显式 visibility/support
乘子，只有在 `recipient support + donor composition` 仍明显退化时，才可把收益归因给 pose 对
selective 参数的条件化；否则只能称正确前景/support 对齐有效。

`pose residual off` 的正确不变量是：同一 P0 checkpoint 下与 D0 mode 的 selective block 输出
逐 forward 一致。它不应被错误要求等于 clean Swin identity，因为 RGB-only SSM 本身仍在工作。

## 启动前阻塞检查

以下项目全部 PASS 后才能正式启动：

- 小张量 sequential reference 与正式 selective scan 输出/梯度一致；
- serpentine permutation 与 inverse permutation 完整互逆，正反向没有 token 错位；
- P0 的 pose 确实同时改变 `Delta/B/C`，D0 mode 三个 pose residual 均严格为零；
- P0/M0 除 pose source 与 OUTPUT_DIR 外配置一致，初始化逐键一致；
- target-person heatmap 而非 scene-merged heatmap 实际进入模块；
- batch 64、真实 GPU AMP + GradScaler 下，`Delta/B/C`、state、loss 均 finite；
- `rho_delta/rho_B/rho_C`、pose encoder、RGB selective projections 与输出投影均有 finite 非零梯度，
  一个 optimizer step 后参数实际变化；
- B0 默认路径 bitwise/严格数值兼容，输出仍是标准 768-d global descriptor；
- 无重复 controller，OUTPUT_DIR 独立。

## e60 Gate 与最终 GO / NO-GO

`epoch < 60` 的评测只记录轨迹，不作负裁决。e60 时先确认模块不是 dead path：三个 pose
residual、RGB `Delta/B/C`、state norm、applied residual、关键梯度和参数更新均为 finite/non-zero。

### e60 继续条件

- P0 没有稳定落后上述预注册同机 clean B0 `0.5 mAP` 以上；并且
- 冻结 checkpoint 的 `correct - matched` 或 support-preserved composition control 至少出现
  `+0.1 mAP` 的对应性燃料。

若 P0 相对该 clean B0 已稳定 `<= -0.5 mAP`，或 `correct - matched` 与
`correct - (recipient support + donor composition)` 都 `< +0.1 mAP`，则在完整 e60 评测与
checkpoint 落盘后正式 NO-GO，停止 M0、多 seed 和组件拆分。

### e120 正式 GO 条件

- `P0 - B0 >= +0.8 mAP`；
- `P0 - D0 >= +0.5 mAP`；
- `P0 - M0 >= +0.4 mAP`；
- 同 checkpoint `correct - matched >= +0.3 mAP`；
- `correct - (recipient support + donor composition) >= +0.3 mAP`；
- `correct - pose-off >= +0.3 mAP`；
- correct-start/end 精确复现，无 NaN/Inf、状态爆炸或 clean-query 明显崩塌。

首 seed 全部通过后，至少补同机多 seed。若训练差值为正但 support-preserved composition 门禁失败，
只能把收益归因于 foreground/support；若 P0 不优于 D0，只能说普通 Mamba 容量有效；若 P0 不优于
M0，只能说固定人体布局有效。任何一种情况都不能声称 instance-pose selective dynamics 成立。

若 full P0 通过全部门禁，再运行 `Delta-only` 关键消融，并与 full `Delta/B/C` 比较：它用于回答
时间尺度是否足以解释收益，以及 pose-conditioned `B/C` 是否必要。该消融不得反过来改变首轮
成功阈值，也不得在 full P0 失败后启动为救场。

## 文献与声明边界

现有工作已经覆盖多个相邻声明：TM-Mamba 已展示外部条件共同生成 `Delta/B/C`；Hamba、
PoseMamba、PS-Mamba、SasMamba、MeshMamba 与 SAMA 已覆盖人体姿态/骨架和 Mamba 的输入、
扫描、状态或 Delta 条件化；ReIDMamba、MambaReID、MambaPro、ReMamba 与 Tac-Mamba 已覆盖
Person ReID 或 pose-guided cross-modal state-space 的相邻空间。

因此不得声称：

- 首个 Mamba ReID；
- 首个 pose x Mamba；
- 首个条件 selective SSM 或首个条件化 `Delta/B/C`；
- 首个双向/serpentine/人体扫描；
- 首个姿态控制遗忘、写入或读取。

当前最多可争的窄边界是：**在单图遮挡 ReID 中，将当前 target-person 2D pose 作为外生观测，
同时校准 RGB-native selective `Delta/B/C`，并用 RGB-only Mamba、canonical pose、matched pose
及 support/composition 拆分证明收益依赖正确 image-pose correspondence。** 这一边界仍须在论文
声明前完成专门查新，不能因为暂未发现完全同构标题就直接写“首次”。固定 serpentine scan、
双向融合、标准 Mamba core 和 global descriptor 都是实现约束，不是独立贡献。

## 最终裁决

当前设计具备可证伪性，且与 exp375 的 body-slot routing、exp376 的瞬时低秩算子存在明确实现
边界，故科学审查放行。正式训练前必须完成上述 reference scan、生产接线、AMP 更新与配置隔离
检查；正式论文 GO 仍完全取决于 B0/D0/M0、冻结对应性和 support/composition 三层证据。

## 启动前终审补充（2026-07-16）

**最终结论：`GO_FOR_TRAINING`，无剩余启动级 BLOCKER。**

终审曾发现并已修复三项执行问题：e60 的 `AND/OR` 判负冲突已统一为严格 OR；e60 clean
早停参考已冻结为 exp375 同一 4090 B0 曲线，且 production test 证明新旧 B0 的 state dict、
descriptor、final featmap 逐元素相同；冻结 evaluator 已覆盖 8 个 arm。

support/composition 交换现在先进入与模块一致的 `12×4` final grid，再直接构造混合 pose，
并由生产 `PoseSelectiveSSM._local_pose` 逐 batch 审计。独立数值复核的最大误差为：

- recipient visibility + donor composition：composition `2.98e-8`，visibility `0`；
- donor visibility + recipient composition：composition `4.47e-8`，visibility `5.96e-8`。

任一正式 batch 的误差超过 `1e-5`，evaluator 会直接失败。exp377-specific donor nuisance
builder 不构成训练启动阻塞，因为训练后的 pose MLP/gain、`Δ/B/C` residual 与 Delta 分布必须
等 e60 checkpoint 才存在；但 builder 特征、匹配算法和接受阈值必须在读取 e60 反事实指标前
冻结，先通过 nuisance audit，再读取 matched/composition mAP。audit 未通过时暂停 e60 裁决，
不得用未匹配 donor 的结果下结论。
