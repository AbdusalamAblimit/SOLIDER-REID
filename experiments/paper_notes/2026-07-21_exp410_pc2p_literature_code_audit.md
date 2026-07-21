# exp410 PC²P 近期文献与代码审计

## 根问题

exp409 PCHM自然e120提高R1却降低mAP，说明单个hard positive/negative不足以改善完整身份排序。PC²P因此不再
选择pair，而是用pose-complete CLIP identity support定义全部训练类别的固定分类几何。

## 最近邻边界

| 最近邻 | 已有对象 | PC²P只能主张的差分 |
|---|---|---|
| CLIP-ReID, AAAI 2023 | Stage-2以冻结身份text feature计算I2T logits，同时保留普通learned ID classifier | 不能声称冻结CLIP原型分类新；PC²P使用pose五槽跨同PID多图补全的visual proxy，并直接替换learned classifier |
| ProFD, ACM MM 2024 | part-specific text prompt、part decoder、多分类头、momentum centroid/memory；测试保留part/visibility路径 | PC²P proxy是冻结外部CLIP visual identity set；测试无part/visibility/bank |
| PCL-CLIP / CCAFL | 当前模型feature形成pseudo-cluster或memory prototype，并用momentum/temperature更新 | 不是监督PID的冻结external CLIP proxy，也无pose-complete五槽支持 |
| PFD-Net / PIRT | pose query、part decomposition与多个learned classifier | 无CLIP identity-set proxy或固定类别几何 |
| TF-CLIP / CLIMB / P-CLIP | CLIP backbone、prompt/cross-modal loss和普通learned linear classifier | 未发现pose-complete frozen visual classifier |
| uncertainty ReID / UFFM / multi-shot KD | 不确定性融合、多视图teacher或推理期邻域聚合 | 它们压缩了另一个候选PC-COVE的新意；不构成PC²P同构实现 |

浏览器查重还确认UFFM/AMC属于无训练的推理期多视图feature/metric组合，不是当前train-only fixed classifier；
但这也说明不能把“多视图聚合”本身当贡献。

## 代码审计结论

1. 初稿 `frozen_proxy @ learnable_Q` 被否决：702类、768维下会近似任意learned classifier。
2. 最终必须无Q/无projection：BNNeck后feature直接与`[702,768]`单位proxy做FP32点积。
3. triplet继续读取同一原始global feature；eval也返回该global feature，因此proxy CE不能被auxiliary head吸收。
4. generic必须是身份正确的full-image global CLIP proxy；所有PID共享同一均值row会与zero一样产生零feature梯度，
   是无效重复control。
5. wrong-RGB用无不动点PID row置换，保留row集合与范数，只破坏PID–CLIP绑定；random-code用于排除任意source key。

## 创新裁决

- 问题门：PASS；
- 机制门：CONDITIONAL PASS；
- 证据门：PASS。

允许按C类候选进入实现，但只有自然e120性能双门GO且correct胜过wrong-RGB/generic，才能进入正面story。
