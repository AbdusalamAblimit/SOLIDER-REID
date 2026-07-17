# SOLIDER 官方 `sw` / `with_cp` 审计

## 范围

审计对象是官方最后提交 `8c08e1c` 的 `model/backbones/swin_transformer.py`，兼容执行提交仅删除未使用的 MMCV import，不改变以下逻辑。审计在独立 CPU 进程中完成，没有修改或占用正在运行的 Market1501 B0。

本文中的 `sw` 指 forward 中由 `semantic_embed_w` 产生的 semantic scale，而不是 shifted-window attention。

## 结论

1. `sw` 有两个确定 bug：设备硬编码，以及最后一组 controller 为死路径。
2. `with_cp` 的 checkpoint 内核在当前 PyTorch 1.13.1 上数值正确；真正的 bug 是官方训练入口没有任何配置或参数把它打开。
3. 正在进行的官方 B0 必须保留这些行为，才能复现官方结果。修复只能进入后续 fresh commit，并需要行为保持/单变量测试。

## `sw`：设备硬编码

官方 forward 在未显式传入 conditioning tensor 时执行：

```python
w = torch.ones(x.shape[0], 1) * self.semantic_weight
w = torch.cat([w, 1-w], axis=-1)
semantic_weight = w.cuda()
```

设备被硬编码为 CUDA 当前卡，且每个 forward 新建 CPU tensor 再复制到 GPU。

后果：

- CPU forward 直接失败；现场错误为 `RuntimeError: No CUDA GPUs are available`。
- 显式传入的 conditioning tensor 不会自动迁移到 `x.device` 或转换为 `x.dtype`。
- 多设备、CPU 单测、导出和编译路径依赖当前 CUDA device 的隐式状态。

行为保持修复应使用 `x.new_full`/`x.new_tensor` 构造 conditioning，并验证默认标量和显式 `[B,2]` tensor 的形状、设备与 dtype。

## `sw`：terminal controller 为死路径

每个 stage 返回：

```python
x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
```

随后 semantic controller 只修改 `x`，而输出特征追加的是修改前的 `out`：

```python
x = x * self.softplus(sw) + sb
out = norm_layer(out)
outs.append(out)
...
x = self.avgpool(outs[-1])
```

stage 0–2 的 `x` 会进入后续 stage，因此 controller 有效；最后 stage 后没有后继计算，修改后的 `x` 被丢弃，descriptor 始终来自未调制的 `outs[-1]`。

CPU 参数扰动实证：

- 最后一组 `semantic_embed_w[-1]` 的 weight/bias 同时加 50：descriptor 最大差值 `0.0`
- 倒数第二组同样扰动：descriptor 最大差值 `1.055903434753418`

官方 teacher checkpoint 也呈现相同边界：stage 0–2 controller bias 非零，而 stage 3 的 `semantic_embed_w.3.bias` 与 `semantic_embed_b.3.bias` 范数均精确为 0；stage 3 weight 仍为初始化量级。这与 terminal controller 从未获得有效梯度一致。

建议的干净实现是只保留三个“stage 间 transition controller”，删除无效的 terminal bank，从而保持官方 descriptor exact parity。若希望最终 descriptor 也受 semantic modulation，应作为独立机制重新设计和对照，不能伪装成无影响的兼容修复。

## `with_cp`：内核正确但训练入口不可达

官方 `SwinTransformer`、`SwinBlockSequence` 和 `SwinBlock` 都接收 `with_cp`，block 内部也正确调用：

```python
if self.with_cp and x.requires_grad:
    x = cp.checkpoint(_inner_forward, x)
```

但是：

- `config/defaults.py` 没有 `MODEL.WITH_CP`；
- 六个官方 YAML 均没有该字段；
- `build_transformer` 调用 Swin factory 时没有传递 `with_cp`。

因此通过官方 `train.py` 不可能开启 gradient checkpointing，构造函数中的默认值始终为 `False`。

在一个 8-block 小型 Swin 上，用相同 state dict、输入和 loss 比较 `with_cp=False/True`：

- forward 最大差值：`0.0`
- 所有参数梯度最大差值：`0.0`
- 梯度缺失集合：空
- 8/8 blocks 的 `with_cp` 标志为真

所以当前证据不支持“checkpoint 数值实现错误”；支持的是“官方配置接线缺失”。后续应新增 `MODEL.WITH_CP=False`，透传到 factory，并在 4090 空闲后补 AMP 下的输出/梯度、峰值显存和吞吐测试。B0/D0 必须使用相同开关。

## 对当前实验的处理

- Market1501 官方 B0：保持 `with_cp=False` 和官方 `sw` 行为，继续到 e120。
- Occluded-Duke B0：先以同一官方行为建立可比 baseline，不静默引入 final semantic modulation。
- TAPF 重建：可采用设备安全、仅三组 transition controller 的行为保持清理；是否启用 `with_cp` 由 matched B0/D0 的共同 config 决定。
