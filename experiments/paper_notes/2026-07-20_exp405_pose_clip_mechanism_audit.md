# exp405 pose+CLIP 近期论文与开源实现审计

## 审计结论

裸的“CLIP选槽 + 同ID跨视角donor + TAPF搬运”不满足机制创新门。它可以被现有的局部CLIP对齐、
pose/part分解、多视图LUPI/KD和patch transfer解释。exp405只条件授权一个最小Phase 0，检验更窄的问题：

> 对明确删除的recipient内部槽，correct donor的恢复是否同时因果依赖身份轴和解剖槽轴；该运输是否具有
> CLIP相对pose-only的独立增量，并且能由held-out PID的单图剩余状态预测？

当前创新门判定：宽claim仅证据设计`1/3`；收窄后“问题+证据”条件性`2/3`；机制项仍需Phase 0后重新定义。
因此不创建formal config，不启动e120。

## 已核对的直接近邻

### MVI²P：同ID多视图teacher到单图student已存在

- 论文：*Multi-View Information Integration and Propagation for Occluded Person Re-Identification*，
  Information Fusion，DOI `10.1016/j.inffus.2023.102201`，arXiv `2311.03828`。
- 官方仓库：`nengdong96/MVIIP`，审计commit `4efd9fc920d2b3b5a8e9329059d81a6573f19b13`。
- 代码事实：sampler构造同ID K-instance；`network/processing.py`综合同IDfeature map；`core/train.py`
  与`tools/loss.py`把integrated feature传播给单图student。
- 边界：已覆盖same-ID multi-view teacher、可靠性综合、测试单图；未覆盖original/deleted/donor可观察target、
  identity×slot二维干预和同一中间operator上的donor-free transition。

### RegionCLIP / ProFD / KPR：局部CLIP软teacher和part表示不是新机制

- RegionCLIP：arXiv `2112.09106`，官方commit
  `4b8513b56e24827e3d6468e1f2105869f35c2d0b`。已覆盖region visual feature与text prototype形成soft
  distribution并蒸馏，不能声称首次局部CLIP双编码teacher。
- ProFD：ACM MM 2024，arXiv `2409.20081`，官方commit
  `14e47d3b04f541d2a614482848bba2071bc90cda`。代码中已有part prompts、外部parsing、hybrid
  cross-attention、自蒸馏和part prototype/representation；推理保留part路径。
- KPR：ECCV 2024，arXiv `2407.18112`，官方commit `e3e6ee2f`。已有keypoint prompt、part/global
  descriptor和共同可见匹配；未覆盖single-global内部运输。

### 2025--2026近邻进一步压缩宽claim

- FLaN-Net，IJCAI 2025：细粒度subject/attribute/occluder prompt、patch-text cross-attention和动态可靠性
  融合均已存在。未找到可核验官方commit，因此不编造代码版本。
- Composite-Attribute Person ReID，CVPR 2026：pose把patch分到`id/head/top/bottom/feet/other`槽，冻结
  CLIP文本向量调制对应part，并用身份/属性关系解耦。它直接关闭“pose slot + CLIP文本 + 身份/属性二维关系”
  的宽新颖性表述；该工作使用文本查询的新任务，未覆盖本项目的训练期donor transport。
- MUVA，2026，arXiv `2603.14012`，官方commit
  `896526309c3392abc01c4499b792606c3574d3b4`：visual grounding mask与局部prompt进入CLIP ViT各block；
  CLIP为保留的student backbone，不是train-only teacher。
- VLCDC，Pattern Recognition 2026，官方commit
  `bb589178de468b44bbedc9f1245edbe2db181dec`：已有key-part attention、局部/全局text centroid和
  saliency-guided structured occlusion；其对象为无监督ReID。
- DPM++，arXiv `2605.06637`：已覆盖CLIP identity anchor、dynamic masked metric和saliency-guided patch
  transfer；无核实源码commit。
- 未决危险近邻MVCD，Neurocomputing 2026，DOI `10.1016/j.neucom.2026.133015`：摘要已覆盖multi-view
  LUPI teacher、cross-view patch alignment和reliability aggregation；正文/官方代码未公开。

## 对exp405的约束

以下内容全部降级为实现或背景，不能写成贡献：pose+CLIP、part-language alignment、同ID多视图teacher、
测试删除CLIP/pose、gather-transform-scatter名称、多stage和普通feature recovery。

唯一可争差分必须同时包含：

1. original/deleted/donor形成可观察target；
2. identity轴与slot轴两个正交破坏干预；
3. CLIP image+text相对pose-only、image-only、text-only有独立增量；
4. teacher-forced transport与donor-free student使用同一production operator；
5. random-key与frequency-matched random-cluster排除伪语义authentication；
6. held-out PID上transport residual可由单图`not-k`状态预测；
7. 最终只和clean D0同epoch、同训练预算的mAP/R1比较。

任何一项失败即NO-GO，不通过换taxonomy、prompt、temperature、删除强度、loss、batch或stage补救。
