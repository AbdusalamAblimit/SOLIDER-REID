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

## e120冻结diagnostic执行器复审

首轮=`2 BLOCKER / 1 HIGH`：执行器允许任意strict-compatible checkpoint冒充e120，config/manifest/cache可由
调用者共同替换，source HEAD仅原样写入而未验证。修复后CLI只保留fresh output；硬绑定并实际核验训练HEAD、
四个关键源码SHA、config路径/SHA、e120 checkpoint路径/SHA、runner终态receipt、cache路径/SHA、64图manifest
路径/SHA及其preprocessing/cache字段。模型先递归eval，只临时设置`base.training=True`进入PICRD计算，所有真实
children继续eval，finally恢复；两遍全部科学标量须bit-exact。

同一独立审查者复审结论：前述`2B/1H`全部闭环，未引入新问题，最终=`0 BLOCKER / 0 HIGH`；授权唯一fresh
64图diagnostic。
