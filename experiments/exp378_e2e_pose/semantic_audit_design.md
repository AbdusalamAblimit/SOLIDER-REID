# exp378 residual-OFF 冻结 checkpoint 语义审计

## 审计对象

本审计不训练新模型，只对一个已完成的 residual-OFF checkpoint 做只读、同 checkpoint、同 RGB
顺序的反事实评测。审计对象固定为同机 fresh D0 的 epoch 90：

- execution commit：`ca62c475b43f17564bb09ede90de6eed53dd2d88`；
- config：`configs/occluded_duke/exp378_d0_continued_pose.yml`，SHA256=
  `a4a184c178e69e8be4f91b4fa480d9eff7e5d5f19f0cadf55636bd8d4367497e`；
- checkpoint：`exp378_d0_continued_pose_s1234/transformer_90.pth`，SHA256=
  `c5407d30d145b92c1995b137ea917187bfb5c1e7c04cd662a44362ae68b4c253`；
- 原始 e90：mAP/R1/R5/R10=`56.3/67.6/79.8/83.5`。

D0 e90 是当前全部有效同机 residual-OFF checkpoint 中最高 mAP 点：hard F0 final=`55.9`、
MR-F0最佳/最终=`56.0`、N0最佳=`56.2`、D0 e90=`56.3`。D0还在全程接受正确teacher监督，适合
检查内生场是否保留teacher相关语义。该 checkpoint 是看到曲线后选出的审计资产，只允许做同一
checkpoint内的配对干预；不得把其`56.3`写成预注册final或替代D0 final=`56.2`。

## 要回答的问题

1. 部署态 descriptor 是否确实完全不读取外部`pose_dict`？
2. Stage-3 PSG是否真正消费内生场，而不是训练课程或附加参数带来的无关容量收益？
3. 内生场的图像对应空间结构、17通道排列和confidence是否对检索结果有因果影响？
4. 内生anchor与teacher的agreement、水平翻转等变性和通道占用是否足以支持“pose-like field”，
   而不是任意17通道身份码？

teacher agreement只称pseudo-PCK/agreement，不称真实pose accuracy；Occluded-Duke没有当前流程可用
的GT关键点。

## 冻结反事实

所有arm复用同一严格加载模型、同一query/gallery RGB与标签顺序、同一flip-test设置。只允许通过
短生命周期forward hook替换TAPF输出或PSG输出；hook退出后必须恢复，模型参数不得变化。

1. `correct_start / correct_end`
   - 不做干预；首尾descriptor SHA和四项指标必须逐位复现。
2. `external_correct / external_shuffle / external_none / external_unindexable`
   - 分别传真实pose、batch内固定循环错配pose、`None`和任何索引都会报错的sentinel；
   - 四臂全数据descriptor必须与`correct_start`逐位相同，证明部署态不读取外部pose。
3. `matched_wrong_field`
   - 使用exp375已冻结、query/gallery分离、无fixed point、异PID的双射donor map；
   - 先由donor RGB产生其内生场，再把该场写入recipient RGB的Stage-3 PSG；RGB、标签和顺序不变。
4. `joint_permutation`
   - 对recipient内生场执行与N0一致的固定17-cycle
     `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0]`；
   - 只改通道到解剖名称的对应，不改每张图17张空间图的多重集。
5. `confidence_permutation`
   - 保留每个通道的peak-normalized空间shape，只按上述17-cycle重排17个peak confidence；
   - 用于隔离confidence side channel。
6. `spatial_constant`
   - 每个样本、每个通道用其正确场的空间均值铺满，保留17个通道均值但移除空间几何。
7. `zero_field`
   - 送入17通道全零raw field；PSG内部仍执行其既定sigmoid/encoder。
8. `psg_bypass`
   - TAPF照常计算，但每个Stage-3 PSG模块直接返回输入token，是真正的Stage-3 PSG关闭。

`matched_wrong_field`固定复用以下映射资产，禁止现场重抽：

- query/gallery mapping SHA256：
  `421e1a179fcf275e4225e6f72d7d10fff196134674e596842a6dc92569ed47e7` / 
  `2403af852fe9c55340a6d265e1ef3c4a0215809e7f72fdc03399d8573e3353ad`；
- query/gallery metadata SHA256：
  `15fc8de5d53a50274c64896c20bb734df342d26e5ec9f3586d2cc1ad09e5f433` / 
  `c36f271cd4a577db1da3d0a4c07d4a5b57e6e42cba1633f507cfd2825d141b58`；
- query/gallery mapping-audit SHA256：
  `12c63357b22eac134391c52aacba8b1e1a51d3988b371b6c9fe184dd60ac9461` / 
  `92839181b22f43966c4608184ddc8546dfb6b89cdeffd133bb6681cac6376d56`。

## teacher agreement、flip equivariance与通道占用

在`correct_start`同一遍数据上，从D0 anchor的spatial posterior/confidence计算：

- teacher confidence `>=0.3`关节的student/teacher normalized coordinate error与pseudo-PCK@0.05；
- student/teacher posterior cosine；confidence MAE、Brier与相关系数；
- 每通道confidence、空间峰值、均值，以及每像素winner的17通道占用、占用熵和有效通道数。

水平翻转时只翻RGB，由内生anchor再次预测；将输出水平逆翻并按COCO左右关节
`[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]`换回，再计算原图/翻转图posterior cosine、
normalized coordinate error与field误差。该结果只衡量几何等变性，不等于姿态准确率。

## 门禁与解释边界

正式运行前必须通过CPU单元与4090 batch64 CUDA小样本门禁：严格checkpoint加载、无参数变化、
hook计数/恢复、correct-start/end exact、external四臂exact、donor无fixed point/异PID/不跨split、
每个干预field shape=`B×17×96×32`/float32/contiguous/finite，以及每个非correct arm确实改变PSG输入
和descriptor。

配对性能只作机制证据，预先使用以下描述性区间：

- correct相对某干预`>=+0.3 mAP`：支持该被破坏因素具有可分辨贡献；
- 绝对差`<0.1 mAP`：当前benchmark/single checkpoint下无可分辨贡献；
- `0.1–0.3 mAP`或四项明显混合：证据弱/不确定，不强行归因。

这些区间不产生新的训练GO，不把单checkpoint干预当作多seed显著性。N0≈F0已说明正确关节名称在
训练bootstrap中不是必要条件；若`joint_permutation`也近似不变，只能进一步收紧为“当前PSG更依赖
pose-like空间支持而非固定解剖通道名”。若`matched_wrong_field`、`spatial_constant`与
`psg_bypass`也均近似不变，则单锚点姿态因果叙事不成立，后续Hierarchical TAPF必须重新定义机制，
不得把普通容量/课程收益包装成姿态语义贡献。

本审计不启动训练，不修改任何已完成arm，不触发H0。
