# 实验 exp399：named-parameter state reporter exact门

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / IMPLEMENTATION STATIC SEALED-PASS /
STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`

## 动机与唯一变量

exp398没有进入科学测试：exp396 reporter的`parameter_groups()`返回`(name, parameter)`tuple，exp398新增
state hasher却按裸parameter处理，actual在首个forward前INVALID。exp399保持exp398的source、32-step
schedule、default GradScaler、loss、batch、baseline-relative裁决和全部门不变，唯一变量是把group-state
哈希定义为有序`[{name, tensor_sha256}]`，显式支持真实named-parameter tuple。

## 冻结实现

1. 每个group item必须exact为长度2 tuple：`(non-empty unique str, torch.nn.Parameter)`；
2. 空group允许并得到稳定空列表SHA；重复名、空名、裸tensor、错误tuple长度或非Parameter必须FAIL；
3. 哈希同时绑定parameter name、顺序与tensor bytes；同值换名、调序、改tensor都必须改变SHA；
4. static直接调用exp396真实`parameter_groups()`处理synthetic model+optimizer，验证返回容器可被hasher读取、
   coverage exact、重复调用SHA exact、参数变化只改变所属group；
5. 其余32-step/e1×16→e6×16/tail8/default scaler/11组active和state变化门逐字保持exp398。

## 裁决

static连续两遍PASS后直接执行唯一fresh actual。任何FAIL封板停止，不重跑。actual PASS仍只授权新编号
final production preflight，不直接授权e120；formal训练`NO-START`。

## static/CPU封板

初次两遍35项逻辑gate均PASS但tiny model未固定seed，result byte-exact FAIL；失败证据保留。只在static
contract固定CPU seed后，正式v1/repeat连续35/35 PASS且四份result/runner逐字节一致。真实exp396
`parameter_groups()`的15个named-tuple group、coverage、name/order/value绑定、空组和五类类型反例全部
PASS；CUDA initialized=`false/false`。

implementation/static/result SHA=
`b9da4346b0d74d13b537bd7fa3f5eff1e65b0b6e512014026800506807723907`/
`7948845f1600141302285cee12c025cbf0ba50faa1af01d1fb298bd3aa558810`/
`32adc18d2b6dc06c0d3ea37ca6003d749a2ff2540efefdbca0e35e1fba2f0d98`。裁决=
`STATIC_CPU_SEALED_PASS / CUDA FRESH-EXECUTION GO`。
