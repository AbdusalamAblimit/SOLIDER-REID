# exp370 PBSR：设计与实现前审查

## 审查结论

**设计、隔离实现与远程单批次 CUDA smoke 全部通过；允许按冻结 manifest 启动第一批 B0/P0。**

方法边界已经从“姿态监督部位 query”收紧为“共享路由的结构分解—重组”，并且最终只输出 baseline 同维 global descriptor。该方向避开了 PAFormer、ProFD、TSD 的直接主张，但仍属于高近邻密度区域，实验必须证明 coupled write-back 的独立价值。

## 已发现并写回设计的风险

### H1：新增模块会破坏 paired RNG

若只在 P0 构造 PBSR，参数初始化会推进 CPU RNG；若 shuffled supervision 调用全局 `randperm`，还会推进训练 RNG。这样 sampler、drop-path 或后续随机增强可能与 B0 不一致。

处理要求：

- PBSR 构造前后保存/恢复 CPU RNG；
- PBSR 首验内部 dropout 固定为 0；
- shuffled target 用确定性 batch roll 或独立 generator；
- runner 在模型构造后再次设定 seed。

### H2：零门可能让 router 永远收不到 identity gradient

`alpha=0` 时，identity loss 对 router/write message 的梯度为 0，只对 alpha 有梯度；如果 alpha 的梯度也因 message 恰好为零而消失，模块不会打开。

处理要求：

- `W_message/W_up` 使用有限的非零小初始化；
- 单元测试必须验证 `alpha.grad` 在第一步非零且有限；
- route loss 同时更新 router，使门打开前的结构路由不是随机静止状态；
- 不同时把 alpha 和 W_up 都零初始化。

### H3：所谓 independent-write 容易混入参数量变量

若增加第二套 Q/K projection，P2 会同时改变耦合方式和参数量，不能归因。

处理要求：P2 使用现有 `K(X)` 与 slot states 重新算 token→slot attention，不新增投影。P0 与 P2 参数量必须逐项相同。

### H4：pose 可能通过前向或日志支路意外进入 eval

处理要求：

- PBSR forward 在 eval 模式不接收/不读取 heatmap；
- 测试 `pose=None/correct/random` 输出逐元素一致；
- evaluator 只接收 tensor global feature，不返回 part dictionary；
- config 不允许 `POSE_TEST_FEAT=equal_concat/maxsim` 成为 PBSR 主结果。

### H5：梯度防火墙容易“只在注释里成立”

同一 `A` 若直接用于 KL，route loss 仍会进入 backbone。不能靠 `A.detach()`，因为那也会截断 router 参数梯度。

处理要求：必须用 `X.detach()` 重新执行 router 得到 `A_supervised`。测试分别检查：

- 仅 `L_route.backward()`：backbone/input grad 为 0，router grad 非 0；
- identity loss：backbone/global/write gate grad 有限；
- pose heatmap 永远不 requires-grad，也不参与表征前向。

### M1：target-only 与 scene heatmap 会改变问题定义

scene max-merge 可能监督旁人结构。主 kill-switch 必须固定使用 target-person heatmap；若数据或 loader 无 target-only 信号，monitor 必须明确降级，不能静默回退。

### M2：background slot 可能让写回退化

background 参与列归一，但其 message 固定为 0。必须记录每槽质量、background mass、dead-slot ratio 和 write residual norm；若 background 吸收全部 token，视为机制失败，不允许只看最终 mAP。

### H6：线性 GAP 会让空间写回代数退化

若 `X_refined=X+A^T U` 后立即 GAP，空间写回很可能等价于 slot messages 的一次加权求和，无法证明主空间表征真的被重组。

处理要求：写回消息先与对应位置的原始 token 经过 token-conditioned gate 相互作用，再进入零门残差和 GAP。`writeback off` 与错误监督对照必须保留，以排除它只是通用 token MLP 的增益。

## 实现门禁

实现完成后的审计结果：

1. [x] default-off 不实例化；新增 config 默认为 False；
2. [x] alpha=0 输出与输入逐元素 `torch.equal`；
3. [x] 仅 route loss backward：input/backbone-side grad 为 0，router grad 非 0；
4. [x] alpha 首步 gradient 非 0 且 finite；打开门后 identity loss 到达 router/out projection；
5. [x] coupled/independent 参数量逐项相同；
6. [x] eval 对 `None/correct/random` heatmap 输出 bitwise 相同；
7. [x] fp32 与 CPU bfloat16 autocast 的 shape、finite、backward 通过；
8. [x] config 可由 YACS 合并/冻结，processor 静态编译通过；
9. [x] 目标 diff 未触碰用户无关文件；
10. [x] 远程真实 dataloader + CUDA AMP + optimizer 单批次 smoke；
11. [x] smoke 后检查 route/alpha/entropy/bg/delta 日志可观测性。

本地命令：

```bash
.venv/bin/python -m py_compile model/modules/pose_bidirectional_router.py \
  model/pose_backbone_model.py config/defaults.py processor/processor.py
.venv/bin/python tests/test_pose_bidirectional_router.py
```

结果：`PBSR mechanism checks: PASS`。

## 远程 CUDA 审查结果

执行环境：RTX 3090，真实 Occluded-Duke dataloader，batch size 64，Swin-Tiny 预训练权重，标准 ID/triplet loss，CUDA AMP 和生产 optimizer。

首先复现到历史默认 AMP 初始 scale `65536` 会使首批 backbone 梯度溢出；关闭 PBSR 的纯 global baseline 同样出现 156 个非有限梯度参数，因此排除 PBSR 特有数值错误。PBSR 在 scale `4096` 仍有 2 个 backbone 参数溢出，在 `2048` 已通过。为覆盖批次波动，冻结矩阵统一使用更保守的 `1024`；该值只由 exp370 config 显式启用，其他实验默认行为不变。

最终 P0 单批次结果：

```text
PBSR CUDA integration smoke: PASS
batch=64 image=(64, 3, 384, 128)
loss identity=21.99402428 route=1.52670991 total=22.75737953
write_scale=0.00000000 route_entropy=3.87105513
background_share=0.14284959 delta_norm=2.41153574
finite_grad_params=203 nonzero_grad_params=177
backbone.patch_embed grad=1.83466599e+02
pbsr.write_scale grad=1.87530518e-02
pbsr.slot_queries grad=3.92610133e-02
pbsr.key_proj.weight grad=3.94807458e-02
write_scale 0.00000000e+00 -> 1.50024407e-05
```

相同 `1024` 设置下 B0 也通过：identity loss `21.99402428`，173/173 个现有梯度参数均 finite/nonzero，patch-embed grad 与 P0 首步一致。P0 在零门初始化时不改变 identity 前向，数值与 B0 完全一致；新增 route loss 只提供 router 监督，符合设计。
