# 实验 exp400：rich-budget final production CUDA preflight

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / IMPLEMENTATION STATIC SEALED-PASS /
STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`

## 动机

exp399已证明default GradScaler自然适应后，rich在32-step production-shaped窗口不劣于D0，11个新增组
也真实更新。但它没有回答训练后state能否strict reload、推理是否真正RGB-only、rho=0是否exact identity、
两个consumer在nonzero rho下是否都产生有限非零执行差。exp400只补齐这些terminal production contract；
不改变模型、loss、scale、batch、rho或训练schedule。

## 固定执行

- source/config/runtime/assets与exp399 exact；
- fresh D0/rich，各32 batch64，e1×16→e6×16，default GradScaler，tail8门与exp399完全相同；
- 所有更新只在内存，checkpoint load/save=`0`；
- rich final state以CPU tensor clone封存，之后所有diagnostic每次strict restore同一state与RNG；
- 不运行retrieval，不写可续训权重。

## terminal PASS门

在exp399全部trajectory/validity门PASS之外，必须同时满足：

1. final state全部finite，state names不含teacher/clip/codebook/text/pose_batch；
2. fresh rich model `load_state_dict(strict=True)`无missing/unexpected，RGB-only descriptor与原model exact；
3. eval传正确pose、batch错配pose、`None`、访问即抛错的pose，descriptor逐元素exact且爆炸pose访问数=0；
4. train-mode epoch1 `rho=0` full与all-bypass逐元素exact；
5. epoch6 full相对all-bypass、只bypass consumer0、只bypass consumer1均非exact，descriptor finite；
6. full/all-bypass mean L2和两个单consumer max-abs均严格`>0`；
7. 两router retained，evidence head retained；11个rich组active/updated；
8. diagnostic前后model state/RNG exact，teacher/codebook exact，source/assets/tracked exact；
9. checkpoint=`0`、scratch=`0`、进程退出、GPU空闲、SHA封板。

## 裁决与授权

任一FAIL只关闭当前final production interface，禁止补跑或调参。全部PASS时=
`FINAL_PRODUCTION_PREFLIGHT_PASS / FORMAL E120 GO`，按用户持续授权立即启动唯一fresh rich-budget C0
seed1234 e120；不续训、不挑best、自然跑满、final-only裁决。
