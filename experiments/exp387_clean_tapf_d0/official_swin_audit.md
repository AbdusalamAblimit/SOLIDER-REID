# SOLIDER 官方 Swin `semantic_weight` / `with_cp` 审计

## 审计边界

本审计只针对官方最后提交 `8c08e1c3255e8e1e51e006bf189e52cc57b009ed`，不修改正在运行的 exp387 repo、代码或 config。可执行审计脚本为 `audit_official_swin.py`，在远端 CPU、`CUDA_VISIBLE_DEVICES=""`、单线程下运行，避免与正式 4090 训练争用。

结论先行：用户关于两处问题的判断成立，但要区分“核心算子坏了”和“官方接线/API 有问题”。`with_cp` 的 checkpoint block 核心是正常的，坏在 config 与 wrapper 没接；`semantic_weight` 不是整体无效，而是前三次跨 stage 调制有效、最后一次调制对最终 descriptor 是 dead path。

## 1. `with_cp`：核心可用，官方配置不可达

官方 `SwinBlock` 与 `SwinTransformer` 都声明了 `with_cp`，并在 block 内调用 `cp.checkpoint`；构造 stage 时也会继续传递该参数。问题出在上层：

1. 官方 `config/defaults.py` 没有 `MODEL.WITH_CP`；向 YACS 合并 `MODEL.WITH_CP=True` 现场得到 `AssertionError: Non-existent key: MODEL.WITH_CP`。
2. `build_transformer` 调用 Swin factory 时没有传 `with_cp`，所以即使外部私自扩展 config，该值仍不会进入 backbone。
3. 小型四 stage Swin 直接以 `with_cp=True` 构造后，4/4 block flag 正确；与 `with_cp=False` 的同权重模型相比，forward、输入梯度、67 个 parameter gradient 的最大绝对差均为 `0`。

因此，不能说官方 checkpoint 核心实现数值错误；准确说法是：**官方 release 无法通过其 config/model builder 真正启用 gradient checkpointing**。任何只在 YAML 写 `WITH_CP: True`、但没有同时补 defaults 与 builder 接线的历史运行，都不能据此声称已经启用。

exp385 B0 与 exp387 D0 的当前 config 都没有该键，二者共同使用 `with_cp=False`，所以这不是本轮 D0−B0 的变量混淆。当前训练不得为此修改。

## 2. `semantic_weight`：前三次有效，terminal 调制无 consumer

官方 forward 的顺序是：stage 先返回 `x/out`，随后 semantic branch 只重绑定 `x = x * softplus(sw) + sb`，再从旧的 `out` 构造 `outs`，最终 descriptor 则池化 `outs[-1]`。

对有 downsample/后继 stage 的 stage0–2，调制后的 `x` 会成为下一 stage 输入，所以它们能影响最终 descriptor。对最后的 stage3，调制后的 `x` 没有后继 consumer；最终池化仍读取调制前的 `out`。可执行反事实结果：

- 将 `semantic_embed_w/b[3]` 全部改成跨度约 `[-50, 50]` 的极端值，最终 descriptor 仍逐元素 exact；
- 修改 `semantic_embed_w/b[2]` 后，descriptor 最大绝对变化=`2.918170928955078`。

所以不能写“SW 四个 stage 都有效”；应写成：**stage0–2 的跨 stage semantic modulation 有效，terminal stage3 semantic branch 对当前 descriptor 是 dead path**。exp385 B0 与 exp387 D0 共享这个官方边界；D0 在 final stage 内加入 PSG，但 terminal semantic modulation 仍发生在最后 descriptor consumer 之后，因此不混淆 matched D0−B0。若未来修复 terminal SW，必须重跑 matched B0/D0，不能把修复混进当前运行。

另有两个独立 API 问题：

1. 默认 semantic tensor 使用硬编码 `w.cuda()`，不跟随 `x.device`，CPU 与非默认 device 路径不安全；当前单卡 cuda:0 正式运行不受影响。
2. `SwinTransformer.train()` 没有 `return self`，因此直接调用 backbone 的 `.train()` / `.eval()` 返回 `None`；父级 ReID wrapper 调用子模块 train 时忽略返回值，所以当前训练/eval 状态切换仍生效，但 backbone API 不符合 PyTorch 常规合同。

## 3. 证据与后续处理

- 状态：`EXP387_OFFICIAL_SWIN_AUDIT_PASS`；
- 脚本 SHA256：`b0080584fd183b1141a6c1e53654c571c82076550ddc679618086fab65fe8e0c`；
- JSON SHA256：`684f70fe5e3d3e31ef170e85dfe94a4f1ed0d013ed99123d4340b477c6b32d71`；
- runner SHA256：`02fc2f87b496abeeeea64d083711b8644548cd3d441942bfb30cbe6785fae96f`。

处理决策：本轮只记录，不改 exp387。后续若需要显存 checkpoint，最小修复是新增默认 `MODEL.WITH_CP=False` 并在 factory 调用显式传参，再做 off exact parity 与 on/off gradient/throughput 门禁；若修复 terminal SW，则属于新的 baseline recipe，必须建立独立 matched 对照。
