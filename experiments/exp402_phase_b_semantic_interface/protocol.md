# exp402 Phase-B semantic-interface协议

## 状态

`PROTOCOL-FROZEN / STATIC-CPU NO-START / GPU NO-START`

## once-only边界

- exp402只允许一个fresh正式diagnostic execution；result/runner/manifest路径必须事前不存在；
- static/CPU可重复两遍验证byte-exact，但不读取official RGB、checkpoint或初始化CUDA；
- 正式脚本必须先通过syntax、AST、toy、真实checkpoint CPU state-name/hash与必要小批CUDA preflight；
- 正式execution出现脚本runtime错误即封板`SEALED-INVALID`，不得修补重跑；新编号继续。

## 资产边界

- 所有写入仅本地experiment目录和远端`/home/afr`的fresh exp402目录；
- official `/mnt1/afrdata`只读；eval禁止访问`/mnt1/afrderived`、CLIP checkpoint/codebook内容或teacher；
- exp401 repo/config/checkpoint只读且SHA exact；脚本置于repo外，不创建checkpoint/scratch/cache；
- `PYTHONDONTWRITEBYTECODE=1`，不让诊断在sealed repo生成执行资产。

## static/CPU最低contract

1. donor map same split/camera、different PID、no fixed point与batch/chunk invariance；
2. zero、orthogonal、evidence-cycle、binding-cycle的shape/value/non-target字段exact；
3. orthogonal matrixrepeat exact、`QᵀQ`误差`<=1e-12`、norm/cosine误差`<=1e-12`；
4. generic expert mean应用与restore exact；
5. router0/1/all bypass覆盖、identity与restore exact；
6. reporter对PASS、单control margin FAIL、route gap FAIL、inactive descriptor FAIL、patch/state FAIL均正确；
7. AST阻断optimizer/backward/train/checkpoint write、external pose/teacher eval与batch-local wrong donor；
8. 两遍正式static result和runner逐字节一致，CUDA initialized始终false。

## 正式执行顺序

1. CPU前门：HEAD/config/source/checkpoint/state/assets/结果路径fresh；
2. GPU唯一性前门；
3. correct完整pass，缓存global evidence与descriptor并复核exp401 raw reference；
4. 按design固定顺序串行执行九个counterfactual arms；
5. 每arm finally恢复patch/parameter并检查model state；
6. 统一计算retrieval、descriptor差、coverage与scientific verdict；
7. 原子写result，退出后独立核验PID/GPU/checkpoint/source并生成manifest。

不得看到中间arm后停止、删arm、改阈值或重排为有利结果；不得并行跑arm。

## 最终门

validity全部PASS后才读scientific verdict。semantic controls中最高mAP也必须比correct低至少`0.1 point`；
route gap必须复现`>=+0.1 point`且correct `>=56.7`。PASS只授权下一编号Phase-B formal mechanism设计；
FAIL只关闭当前semantic interface解释。
