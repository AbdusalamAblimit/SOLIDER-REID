# exp411 PCMPSR 近期论文与开源实现审计

## 审计问题

exp409表明单个pose×CLIP hard pair可提高R1但不提高mAP，exp410又表明冻结CLIP identity axes会破坏student分类
空间。审计目标不是寻找另一个metric loss名字，而是确认：是否已有方法在标准单图ReID中，用pose覆盖与同PID
region CLIP共识离散组织完整多视图支持集，并在student自身global空间对全部身份集合做等支持listwise排序。

## 近邻边界

| 近邻 | 已覆盖对象 | 与PCMPSR差异 |
|---|---|---|
| SupCon / supervised metric learning | 同标签多正样本对比，通常把所有正样本同权 | 不按解剖槽选择owner multiplicity，也没有等支持身份集合距离 |
| Lifted Structured / Ranked List / Smooth-AP / ROADMAP | all-pair、listwise或AP surrogate | relevance来自标签/student距离，不由pose缺失与region CLIP同PID共识定义集合结构 |
| episodic/prototypical set learning | query到类别prototype或support set距离 | PCMPSR不产生外部/固定prototype；每个身份等支持且五槽owner可重复计数 |
| pose-aware sampling / part ReID | pose选择样本、part或可见区域 | 未同时使用frozen region CLIP槽共识组织all-identity final ranking |
| CLIP-ReID | identity text prompt与image-text alignment | 不构造pose-complete多正身份集合，且CLIP轴直接参与跨模态监督 |
| ProFD / part prompt-memory | part prompt、centroid/memory及局部表征 | 测试保留part机制；不是global student空间的leave-one-position-out set ranking |
| exp408 PICRD | Stage-2逐槽CLIP relation | 中层teacher relation可学但不改善最终mAP |
| exp409 PCHM | pose×CLIP选一个正负pair | 单边改善R1、mAP失败；PCMPSR让所有正支持与所有负身份集合进入梯度 |
| exp410 PC²P | 冻结pose-complete CLIP classifier | 外部坐标轴错配；PCMPSR只让CLIP决定离散support multiplicity |

## 开源代码映射

当前仓库的原`TripletLoss`在每个anchor上只取一个最远同ID与一个最近异ID；PCHM也只把这两个index替换为外部
owner。PCMPSR只需新增独立set-loss/strict cache loader，在`make_loss.py`的互斥分支替换triplet，并由processor
沿现有relative-path/cache/pose metadata传入离散owner state；model forward与eval无需修改。这一接点避免新head、
projection、temperature和测试分支。

## 创新判断

- 问题门：PASS。把遮挡ReID从单pair判别改写为等支持的跨视图身份集合排序，直接对应AP全列表质量；
- 机制门：CONDITIONAL PASS。已有原子很多，只有leave-one-position-out等支持、五槽owner multiplicity与final
  all-identity set ranking三者整体成立；
- 证据门：PASS。D0、zero-owner、generic、wrong-RGB、pose-only以及owner unique/change可明确证伪。

因此只定位为C类候选，不声称首次listwise、首次multi-positive或首次pose+CLIP。自然e120不过clean D0双门，或
correct不胜zero-owner/wrong-RGB，均立即关闭正面创新主张。
