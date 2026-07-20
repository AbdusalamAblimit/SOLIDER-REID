# exp405 CAVT 监控记录

> 当前：`BROAD NOVELTY NO-GO / NARROW PHASE-0 CONDITIONAL GO / STATIC V14 PASS /
> REAL TEACHER MEASUREMENT IMPLEMENTING / GPU IDLE`

## 2026-07-20：exp404后根因边界

exp404已封板为`VALIDITY PASS / SPK MECHANISM NO-GO`。本实验不修改或重跑SPK。根因审计确认旧
student evidence到e120仍基本未学会teacher，且presence定义、slot-local readout、双编码teacher与SPK聚合
均偏离用户原始候选。

两路独立审查均反对把失败扩大为CLIP–TAPF总否定，同时也反对直接把“CLIP选槽 + same-ID donor + TAPF
搬运”包装为新方法。新设计因此冻结identity轴 x slot轴二维反事实、可观察original/deleted/donor target与
donor-free同路径student三项必要条件。

## 2026-07-20：train pair可行性只读统计

远端official Occluded-Duke train目录只读统计：

- train images=`15,618`；PID=`702`；
- `702/702` PID具有至少两个camera；
- `15,618/15,618`图可找到same-ID/different-camera donor；
- `510` PID覆盖至少3个camera。

exp386 artifact全量score只读统计：五region的`max(score)>0`比例均为`100%`；阈值`0.7`时分别为
`98.56/98.26/98.26/96.12/94.31%`。这进一步证明旧`semantic_valid`不是visibility。

这些统计只授权继续设计pair map与teacher oracle，不授权CLIP/GPU执行或正式训练。下一步等待近期公开代码
审计闭合后修订新颖性边界，再实现Phase 0 static合同。

## 2026-07-20：根因驱动的teacher/readout拆分

设计现已把CLIP失败拆成两个独立对象：`P0B-PATCH`严格验证原始pose-pooled patch-token teacher；若其失败，
只封板该readout。`P0B-ISO`作为预注册的独立后继，使用从第一层开始的region-isolated frozen image readout，
专门排除旧PC-MBCLS前20层全局CLS残差泄漏。二者均使用frozen image+text双编码器、checkpoint原生
`logit_scale`、actual-view RGB和全body-part prototype分布，不再使用image-only PCA target。

transport oracle新增self-slot restore测量上界和frequency-matched random-cluster强控制；CLIP必须相对
pose-only产生独立收益，且correct必须同时击败wrong-slot与wrong-ID。该修订仍是设计，不创建config、output、
runner，不占用GPU。

## 2026-07-20：近期近邻与第二轮红队收窄授权

MVI²P已覆盖同ID多视图teacher到单图student；RegionCLIP/ProFD/KPR/MUVA已覆盖局部CLIP、part prompt、
可见性和局部表示；FLaN-Net、Composite-Attribute ReID、DPM++与VLCDC进一步覆盖细粒度语言引导、pose槽、
身份/属性关系、动态masked metric与patch transfer。因此宽CAVT机制创新判NO-GO。

只保留最小Phase 0：唯一五槽taxonomy、region-isolated双编码readout、50% primary deletion、identity×slot
双干预、MVI²P/pose-part/attribute-relation/generic-transport近邻，以及held-out PID donor-free可预测性门。
正式config与e120继续NO-CREATE/NO-START。远端只读检查为RTX4090 `2 MiB / 0% / 0 compute PID`。

## 2026-07-20：Phase 0 static v14最终授权

v11--v13均保留为历史、不授权结果，不覆盖、不重判。v14只修复盲审指出的科研正确性合同：完整Torch
RECORD/site-packages闭包、payload/receipt统一异常清理、逐target-slot/PID独立latent、极大/次正规有限值、
feature/mask设备一致性、sample/pixel/slot/seed联合avalanche删除以及camera-aware mAP/R1稳定tie。

唯一v14源码SHA为core=`29ddd00c...f8b`、contract=`13aff524...f60`、bootstrap=`1b9cacf0...17da`。
两次fresh CPU execution均为`56/56 PASS`，canonical payload逐字节相同，SHA256=
`6d073b72894c65236a53ee52d8e1d868e8492c60cc69ade139d57d0560130ee3`；receipt分别绑定各自output名。

最终独立七文件盲审结论为`BLOCKER=0 / HIGH=0 / MEDIUM=0 / LOW=0`。这只授权立即实现真实
region-isolated frozen image+text teacher measurement；不授权直接运行、CUDA、正式训练或e120。静态启动器与
供应链威胁模型到此封口，不再扩审；下一次盲审只针对真实CLIP/pose/data/metric实现是否污染科学结论。
