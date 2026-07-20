# exp408 PICRD 文献与代码差分审计

## 裁决

没有发现与PICRD完整同构的公开方法；问题门与证据门可过，机制门条件通过，只够C类候选。不能声称
“首次pose+CLIP”或“首次part distillation”。保留资格依赖四件事同时存在：逐解剖槽跨batch CLIP relation、
correct与wrong/generic/zero训练内排序、未detach Stage-2直传、测试期标准global RGB descriptor。

## 最接近工作

1. **π-VL**（arXiv:2308.02738）：human parsing产生part prompt并做局部视觉语言对齐；无external-pose
   binding反事实、无逐槽跨batchrelation、无Stage-2直传标准global descriptor。
2. **ProFD**（ACM MM 2024，`Cuixxx/ProFD`）：part text proxy与CLIP visual token双向cross-attention并保留
   part representation；无pose intervention排序或relation teacher。
3. **PAFormer**（arXiv:2408.05918）：pose heatmap监督pose-token与patch attention，pose-free inference；
   无CLIP local relation，最终是part-to-part retrieval。
4. **KPR**（ECCV 2024，`VlSomers/keypoint_promptable_reidentification`）：keypoint prompt产生part mask、
   visibility和part embedding；无CLIP或correct/wrong/zero关系蒸馏。
5. **MUVA**（arXiv:2603.14012，`RikoLi/MUVA`）：局部CLIP token、文本prompt与mask注入CLIP attention；
   CLIP本身是student，无external pose反事实或逐槽跨样本teacher relation。
6. **Composite-Attribute Person ReID via Pose-Guided Disentanglement**（CVPR 2026）：最危险近邻，已覆盖
   pose分槽、CLIP part slot和batch关系约束；但任务是图文组合检索，不是标准单图ReID，也无same-image
   pose-CLIP binding干预或Stage-2执行中介。

## 仓库根因

- `CleanRichEvidenceBudgetTapf.prepare()` 对Stage-2 source detach；
- `RichEvidencePoseAnchor.forward()` 对hidden再次detach；
- 旧五槽evidence来自全图GAP而非slot-local pooling；
- 因此旧relation loss不能训练backbone，也不能证明sample-specific局部语义进入descriptor。

PICRD只复用clean D0的PSG与标准ReID头，新loss直接作用于未detach Stage-2 source；不复用CAVT cache、
donor、matcher或运行结果。
