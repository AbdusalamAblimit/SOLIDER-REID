# 实验 exp069: Part-Structured PAA (PS-PAA)

## 动机
- exp066 PAA (generic conv): +0.87%/+1.63% — 突破性结果
- PAA 的 encoder 是一个 generic Conv2d(17→32→768)，不区分身体部位
- 如果让 encoder 知道身体结构（17 个关键点按部位分组），应该能产出更精确的 additive content

## 创新点
- 将 17 通道按 5 个身体部位分组
- 每个部位组有独立的 encoder（而非共享）
- 各部位的 adapter output 按空间位置叠加
- 这让 PAA 从 "generic pose injection" 升级为 "part-structured semantic completion"

## 技术方案
- 简化版：在 PAA encoder 的第一层用 **5-group grouped convolution**
  - Conv2d(17→40, groups=1) → 改为 Conv2d(17→40, kernel=1, groups=5)
  - 需要 pad heatmap 到 20 channels (17→20)，使每组 4 channels
  - 这让 head channels、torso channels 等在独立的 subspace 中处理
  - 第二层仍是正常 Conv2d(40→768)
- 或更简单：直接用 5 channel attention → 不需要 grouped conv
  - 先对 17 channels 做 channel attention（17→5→17，sigmoid）
  - 然后正常 conv

实际上最简单的方式是直接增大 PAA 的 bottleneck_dim。当前 bottleneck=32 可能太小。

## 修正方案: PAA 大 bottleneck (exp069)
- 将 bottleneck_dim 从 32 增大到 128
- 参数量从 51.8K 增加到 ~200K
- 测试 PAA 是否受限于 bottleneck 容量

## 对照组
- exp066 PAA (bottleneck=32, 51.8K): 61.6%/74.2%
