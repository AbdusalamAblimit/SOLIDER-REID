# exp408 PICRD 独立盲审

## 审查范围

独立智能体只读审查design/protocol/config、cache builder、cache loader、Stage-2接入、relation loss、processor
数据路径与相对HEAD的完整diff。按当前执行约束只裁决BLOCKER/HIGH，不使用Claude。

## 首轮

- `0 BLOCKER / 1 HIGH`。
- HIGH：processor外层AMP可能让已显式`.float()`的einsum与Gram matmul重新按FP16执行，违反FP32 relation合同。

## 修复

`PoseIndexedClipRelationalTapf.prepare()`中的mask渲染、slot pooling与四臂relation整体进入
`torch.cuda.amp.autocast(enabled=False)`，source/teacher显式FP32；D0 super路径保持不变。

## 闭环结论

`0 BLOCKER / 0 HIGH / GO`。

审查确认：四臂共同valid/pair支持、control stop-gradient、offset4 different-PID、cache ontology/完整路径、
eval forward不读cache、config-off D0 exact以及Stage-2到Stage0--2的梯度路径均闭合。授权进入既定最小
CUDA-AMP检查，随后fresh cache builder与训练。
