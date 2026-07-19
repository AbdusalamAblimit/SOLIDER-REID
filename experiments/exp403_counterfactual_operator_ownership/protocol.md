# exp403 ELO-CUR 冻结协议

## 0. 当前授权边界

当前只执行 targeted audit、设计与 CPU/static contract。GPU=`NO-START`。后续只有在本协议的 CPU、
source、real-batch CUDA/AMP 门全部通过后，才允许建立 fresh formal execution；不得修改 sealed
exp394–exp402 代码、config、checkpoint 或结果。

## 1. 不变量

- backbone=`Swin-Tiny`；batch=`64`；seed=`1234`；epoch=`120`；workers=`8`；
- SGD/cosine/default GradScaler/final checkpoint only；
- rank=`16`，evidence dim=`16`，rho=`0.08075544983148575`；
- teacher/handoff=`5/5`，outer pose loss weight=`0.1`；
- official data只读 `/mnt1/afrdata`，pose只读 `/mnt1/afrderived`；
- train/eval后部署均 RGB-only；
- 不续训、不换 seed、不挑 best、不按中间性能早停。

## 2. donor contract

对 batch 中样本 `i`，按 cyclic absolute batch offset 选择第一个同时满足：

```text
camera[j] == camera[i]
pid[j]    != pid[i]
j         != i
```

的 donor。没有合法 donor 时记 `-1` 并从 compatibility/CUR mask 掉；禁止 global random、跨 camera 回退、
batch-local roll 冒充 matched donor。每次日志记录 eligible count、same-camera、different-PID 和重复确定性。

## 3. CPU/static 正反 contract

必须连续两次 byte-exact，并至少验证：

1. CUDA 未初始化；
2. router 无 slot-specific expert、所有 linear 无 bias；
3. evidence 为逐 rank 系数，zero evidence逐元素 exact identity；
4. correct/wrong/generic/NULL compatibility 按冻结诊断 margin排序；生产训练只允许
   `correct-max(stopgrad(reference))`单边 hinge，禁止无梯度的reference-reference hinge或reference sabotage；
5. 四个 production descriptor finite且干预 active；
6. donor same-camera/different-PID/重复 exact；
7. CUR reference 在 autograd 中 detach；
8. correct evidence、compatibility、H/U/V/C 获得非零有限梯度；
9. wrong/generic/NULL evidence reference 梯度为 0/None；
10. evidence-ignored、aux-only、reference-not-detached 三个 mutant 必须被抓住。

任何失败均保持 GPU `NO-START`。

## 4. CUDA preflight（CPU PASS 后才创建）

### 4.1 static/source

- exact fresh repo HEAD/config/source SHA；
- 只新增 ELO-CUR 开关，默认 config 与 D0/C0行为不变；
- optimizer覆盖全部且仅覆盖应训练参数；
- teacher-free strict state；
- no resume、fresh output/assets、checkpoint=0；
- D0 off-parity逐元素 exact。

### 4.2 actual batch64

- 同一 first batch 对 D0 与 ELO-CUR做 default GradScaler baseline-relative矩阵；
- e1/e6各冻结32 attempts，沿用 exp399 的 tail8与rich-extra-skip门；
- correct/reference Stage-3 RNG exact；
- reference无grad、correct所有生产组finite/active/updated；
- peak memory不 OOM；batch不得改变；
- rho0 full/bypass exact，rho>0 correct/wrong/zero/generic均finite/active；
- eval correct/shuffle/None/ExplodingPose exact且pose访问0；
- preflight不得产生checkpoint。

## 5. formal once-only

preflight result必须显式 `formal_training_authorized=true`。随后唯一 fresh seed1234 e120自然跑满；每10 epoch
eval只记录，不裁决。e120前checkpoint=0，结束后唯一checkpoint。

训练日志额外记录：eligible donor ratio、compat correct/wrong/generic/NULL、correct-max ownership hinge、两个
reference-reference诊断 gap、CUR wrong/generic/NULL、correct/ref utility、operator coefficient
std/effective rank、BudgetAbs、NaN/Inf/AMP warning。

## 6. final frozen counterfactual

同一 checkpoint 串行完整执行：

```text
correct
wrong-RGB (same split/camera, different PID)
generic train-split mean
NULL zero
slot-cycle
wrong-mask binding
all-router-bypass
```

所有 arm 必须覆盖完整 query/gallery；state/RNG/patch/source/config/checkpoint 在每臂后 exact恢复。正式科学门
按 design.md：D0 mAP/R1 floor、correct-vs-controls `+0.1 mAP`、route gap `+0.1 mAP`，三者缺一不可。

## 7. 封板纪律

- reporter/runtime error只封板该 execution，新编号修 contract；
- scientific FAIL不删 control、不调 rho/loss/batch/stage救场；
- PASS/FAIL 后更新 monitor/results/decisions/innovation/story并显式暂存目标文件；
- exp403 once-only编号一旦进入 CUDA/formal 不重跑、不补跑、不续训。
