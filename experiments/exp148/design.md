# 实验 exp148: PCVT（Pose-Complementary View Training）

## 动机

`exp109` 证明真正的 headroom 在 `single-image support incomplete`。  
但 `exp110-142` 也反复证明：把 same-ID 跨图 support bank 直接蒸到单图特征里，很难在 15K 数据上学成。

因此这次不再做 cross-image completion，也不再做 retrieval scorer 小修补，而是直接改训练对象：

**让一张图在训练时变成两张“互补可见”的伪视图。**

核心问题不是“如何从别的图补我”，而是：
**能否让同一张图自己提供两份互补 support，从而逼迫编码器学会对 partial support 稳定？**

## 核心假设

1. 随机遮挡或普通 ROA 之所以只能当 recipe，是因为它们没有显式构造“互补 body support”
2. 如果两张增强视图是由 pose 定义的互补遮挡，那么：
   - 单独看每张视图都不完整
   - 合起来却接近完整 support
3. 在这种设置下训练得到的 backbone，应比普通 `exp030a` 更擅长处理真实遮挡

## 技术方案

### 1. 数据视图构造：从随机增强改成 pose-defined complementary views

对每张训练图：

1. 用 scene heatmap / person-0 keypoint response 找到可见 body groups  
   默认 groups:
   - head
   - left_arm
   - right_arm
   - torso
   - left_leg
   - right_leg

2. 将“当前可见”的 groups 做一次平衡划分，得到互补两组 `A/B`
   - 尽量让两组总响应面积接近
   - 保证 `A` 与 `B` 不重叠

3. 生成三张训练视图
   - `view_full`: 原图标准增强
   - `view_a`: 遮掉 `A`
   - `view_b`: 遮掉 `B`

这样 `view_a` 与 `view_b` 的 body support 是互补的，而不是两个随机遮挡版本。

### 2. 三视图共享 backbone

三张图都走同一个 `exp030a` 主干：
- full view
- complement view A
- complement view B

### 3. 训练目标

总损失分为两部分：

#### (a) ReID 主损失
对三张视图都计算标准 `ID + Triplet`

目的：
- 不是只让 masked view 对 full 做对齐
- 而是要求每个 partial view 本身也能保留身份判别性

#### (b) Complement-Union Consistency
定义：
- `f_full`
- `f_a`
- `f_b`
- `f_union = 0.5 * (f_a + f_b)`（第一版先不用新参数）

约束：
- `f_union` 应比 `f_a`、`f_b` 单独更接近 `f_full`

实现第一版用：
- cosine consistency / MSE consistency（二选一，优先 cosine）

这一步的核心不是“重建像素”，而是：
**让互补 partial supports 在表示空间里重新合成接近完整 support 的身份特征。**

## 对照组

- 主基线：`exp030a-eq`
- 间接历史参考：
  - `exp050 PAMC`：body-aware masking consistency，中性
  - `exp067 ROA`：遮挡增强 recipe
  - `exp142 SKC`：feature-level completion，负

本实验的价值在于它与这三条都不同：
- 不只是 consistency
- 不只是增强
- 不只是 completion module

## 预期结果

若方向成立，应该出现：

1. `view_a/view_b` 单独性能较低，但 full 主分支最终更强
2. `union consistency` 的统计明显优于单视图 consistency
3. 最终 `exp148` 应优于 `exp030a`
4. 若成立，它会比 retrieval-side 小 scorer 更像论文主贡献

## 关键日志

训练期必须额外记录：

- `pcvt_lc`: complement consistency loss
- `pcvt_cov_a`: A 视图保留的可见 support 比例
- `pcvt_cov_b`: B 视图保留的可见 support 比例
- `pcvt_cov_u`: A/B 联合 support 比例
- `pcvt_ovr`: A/B overlap 比例
- `pcvt_mga`: A 视图平均 mask 面积
- `pcvt_mgb`: B 视图平均 mask 面积
- `pcvt_cos_fa`: `cos(f_full, f_a)`
- `pcvt_cos_fb`: `cos(f_full, f_b)`
- `pcvt_cos_fu`: `cos(f_full, f_union)`
- `pcvt_gap`: `cos(f_full, f_union) - 0.5*(cos_fa + cos_fb)`

如果这些日志缺失，这次实验直接算不可解释 run。

## 风险与失败解释

1. 若 `pcvt_cov_u` 明显高于单视图，但结果仍无提升  
   说明“伪多 support”这个训练对象本身不够有用

2. 若 `pcvt_gap <= 0`  
   说明简单 `avg(f_a, f_b)` 并没有真的形成 union support

3. 若 masked views 太难，主损失明显恶化  
   说明 complementary masking 过重，需要收紧 partition / mask 策略

4. 若结果只是和 `PAMC` 一样中性  
   说明单图伪多 support 这条训练范式也很可能不足以支撑主创新
