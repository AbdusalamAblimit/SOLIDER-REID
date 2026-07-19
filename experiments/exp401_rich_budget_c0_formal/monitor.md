# exp401 rich-budget C0正式训练监控

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
FORMAL RUNNING`

## 2026-07-19 接手

- exp400唯一actual=`FINAL_PRODUCTION_PREFLIGHT_PASS`，D0/rich各`27/32`更新，31项terminal全PASS；
- result显式`formal_training_authorized=true`，exp400进程已退出，GPU=`2 MiB/0%`；
- exp401只把已冻结production graph自然跑满e120，不修改任何科学变量。

## 2026-07-19 static launch contract

- candidate config相对冻结rich-budget C0仅改变fresh CLIP路径、fresh codebook路径和OUTPUT_DIR；
- 初次17项中仅数据路径门FAIL：YAML parser保留`('/mnt1/afrdata')`字面括号，而不是数据路径错误；
  初始result/runner保留，config未改；
- static判定修正为字面包含exact `/mnt1/afrdata`后，正式两遍18/18 PASS且逐字节一致；
- exp400 formal授权、source SHA、Swin-Tiny、rich route、rho schedule、120/batch64/seed1234、optimizer/
  warmup、loss权重、checkpoint/eval周期、fresh output、无resume全部PASS；CUDA未初始化；
- static/config/result SHA=
  `90c95b4ac1be32a8d4917882be1c407d17945511205446ede7ddaefb847f319d`/
  `c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`/
  `85cab0e0a8391b3470f0e11acbd634d3dce2fee638432679a2ef9dc49cae020d`。

裁决：`STATIC-CPU SEALED-PASS / FORMAL FRESH-EXECUTION GO`；直接建立fresh远端repo/assets/config并
启动唯一e120，不等待确认。

## 2026-07-19 formal启动

- repo=`/home/afr/SOLIDER-REID-exp401-rich-budget-c0-formal-11d7a35`，exact HEAD=
  `11d7a35788c4645c355d96d76a2a4ff20a9801ac`，tracked/all status clean、无alternate；
- formal config SHA=`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`；
  source八项SHA exact；fresh CLIP/codebook均regular、非symlink、新inode且SHA exact；
- output=`/home/afr/SOLIDER-REID-exp401-rich-budget-c0-formal-11d7a35/log/occluded_duke/exp401_clean_swin_tiny_rich_budget_c0_s1234`；
  runner=`/home/afr/train-logs/exp401_rich_budget_c0_s1234.runner.log`；
- main PID=`404782`，parent=`1`，8 workers；唯一GPU compute PID=`404782`；启动后约
  `8,492 MiB`，无并行任务；
- official数据统计train/query/gallery=`15,618/2,210/17,661`；Frozen rich teacher checkpoint/codebook
  SHA在日志中exact；首批valid=`320`，evidence norm mean/min/max=`1/1/1`，basis orthogonal max-abs=
  `2.554e-15`；
- e1 Iter20 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `20.215/1.496/0.580/0.657/0.734/0.989/0.324/0.196`，Student=`0.00`，Reliability=`1.000`，
  rho=`0`，BudgetAbs=`0`，finite；
- 仅有已知PyTorch AMP API deprecation FutureWarning，无NaN/Inf/Traceback/RuntimeError/OOM/nonfinite；
  checkpoint=`0`。

当前裁决=`FORMAL RUNNING`；自然跑满e120，不按中间指标或GateAbs早停。

## 2026-07-19 e2完成 / e3进行中

- exact HEAD/config/source tracked clean；main PID=`404782`、8 workers，唯一GPU process，约
  `8,492 MiB`、util约`99%`；checkpoint=`0`、异常扫描=`0`；
- e2 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `9.289/1.492/0.577/0.653/0.709/0.994/0.319/0.207`，Acc=`0.020`，Student=`0`，
  Reliability=`1.000`，rho=`0`，BudgetAbs=`0`；
- e2自然完成，time=`127.106 s`；已进入e3 Iter40，Loss=`8.238`，所有记录finite；
- 当前判断=`继续`；原因：进程、数值、schedule与零checkpoint边界全部正常，中间结果不裁决。

## 2026-07-19 e9完成 / e10进行中

- exact HEAD/config/source tracked clean；main PID=`404782`、8 workers，唯一GPU process，约
  `8,492 MiB`，checkpoint=`0`；异常与AMP数值warning扫描均=`0`；
- e9 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `5.797/1.328/0.523/0.531/0.530/0.995/0.297/0.259`，Student=`0.80`，Reliability=`0.998`，
  rho=`0.064604360`，BudgetAbs=`3.669e-02`；e9自然完成；
- 已进入e10 Iter60，Student=`1.00`，rho达到冻结上限`0.080755450`，BudgetAbs=`5.618e-02`；
  Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `5.816/1.303/0.515/0.518/0.510/0.991/0.294/0.262`，全部finite；
- e10完整eval尚未发生；当前判断=`继续`，中间训练值不裁决。

## 2026-07-19 e10评测 / e17进行中

- e10完整评测mAP/R1/R5/R10=`34.4/43.3/59.7/65.7`，只记录不裁决；
- 已完成e16并进入e17；e16 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `3.050/1.063/0.444/0.358/0.311/0.994/0.283/0.272`，Acc=`0.648`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`3.858e-02`，全部finite；
- exact HEAD/config/source tracked clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,602 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：完整e10轨迹已记录，正式裁决仍只在e120 final进行。

## 2026-07-19 e20评测 / e24进行中

- e20完整评测mAP/R1/R5/R10=`42.4/54.7/69.1/75.4`，只记录不裁决；
- e23 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `1.332/0.905/0.392/0.251/0.169/0.991/0.278/0.268`，Acc=`0.897`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`2.538e-02`，全部finite；e23自然完成；
- 已进入e24 Iter20，Loss=`1.437`，Student=`1.00`，Reliability=`1.000`，rho保持冻结上限，
  BudgetAbs=`2.530e-02`；
- exact HEAD/config/source tracked clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,608 MiB`、util约`99%`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：训练轨迹、rich route schedule、数值与执行边界正常，中间评测不触发裁决。

## 2026-07-19 e30评测 / e31进行中

- e30完整评测mAP/R1/R5/R10=`45.6/55.9/70.2/75.8`，只记录不裁决；
- e30 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.634/0.851/0.368/0.209/0.110/0.987/0.278/0.255`，Acc=`0.970`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`1.913e-02`，全部finite；e30自然完成；
- 已进入e31 Iter140，Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.630/0.844/0.365/0.204/0.103/0.985/0.278/0.255`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`1.836e-02`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,634 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e30评测完整，训练、rich route与所有执行边界正常，中间性能不用于裁决。

## 2026-07-19 e40评测 / e45进行中

- e40完整评测mAP/R1/R5/R10=`48.6/59.2/73.2/79.0`，只记录不裁决；
- e40 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.362/0.818/0.350/0.182/0.070/0.983/0.278/0.236`，Acc=`0.986`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`1.418e-02`，全部finite；e40自然完成；
- 已完成e44并进入e45 Iter40；当前Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.338/0.807/0.345/0.177/0.061/0.973/0.280/0.231`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`1.337e-02`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,642 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e40评测完整，训练、数值及所有冻结执行边界正常，中间性能不用于裁决。

## 2026-07-19 e50评测 / e52进行中

- e50完整评测mAP/R1/R5/R10=`53.5/65.0/77.7/83.1`，只记录不裁决；
- e50 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.265/0.808/0.343/0.173/0.055/0.981/0.279/0.228`，Acc=`0.991`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`1.165e-02`，全部finite；e50自然完成；
- e51自然完成并已进入e52 Iter40；当前Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.294/0.804/0.342/0.172/0.056/0.980/0.277/0.225`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`1.154e-02`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,608 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e50评测完整，训练、数值及冻结执行边界正常，中间性能不用于裁决。

## 2026-07-19 e60评测 / e61进行中

- e60完整评测mAP/R1/R5/R10=`53.5/64.8/78.1/82.9`，只记录不裁决；
- e60 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.211/0.803/0.339/0.168/0.048/0.977/0.279/0.223`，Acc=`0.994`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`1.083e-02`，全部finite；e60自然完成；
- 已进入e61 Iter160；当前Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.211/0.799/0.338/0.167/0.046/0.974/0.279/0.223`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`1.056e-02`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,614 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e60评测完整，训练、数值及冻结执行边界正常，中间性能不用于裁决。

## 2026-07-19 e70评测 / e76进行中

- e70完整评测mAP/R1/R5/R10=`55.2/66.1/79.4/83.8`，只记录不裁决；
- e70 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.185/0.798/0.336/0.165/0.041/0.975/0.279/0.221`，Acc=`0.996`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`9.781e-03`，全部finite；e70自然完成；
- 已完成e75并进入e76 Iter100；当前Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.183/0.792/0.333/0.164/0.038/0.968/0.278/0.219`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`9.551e-03`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,614 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e70评测完整，训练、数值及冻结执行边界正常，中间性能不用于裁决。

## 2026-07-19 e80评测 / e82进行中

- e80完整评测mAP/R1/R5/R10=`56.0/66.8/80.2/84.5`，只记录不裁决；
- e80 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.166/0.795/0.335/0.164/0.039/0.973/0.280/0.219`，Acc=`0.997`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`9.686e-03`，全部finite；e80自然完成；
- 已进入e82 Iter200；当前Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.162/0.795/0.335/0.164/0.039/0.974/0.279/0.218`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`9.300e-03`；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,620 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e80评测完整，训练、数值及冻结执行边界正常，中间性能不用于裁决。

## 2026-07-19 e90评测 / e96完成

- e90完整评测mAP/R1/R5/R10=`56.5/66.8/80.2/85.0`，只记录不裁决；
- e90 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.156/0.795/0.335/0.163/0.037/0.976/0.279/0.218`，Acc=`0.997`，Student=`1.00`，
  Reliability=`1.000`，rho=`0.080755450`，BudgetAbs=`8.953e-03`，全部finite；e90自然完成；
- e96 Iter200 Loss/Pose/Semantic/Mask/Presence/EvidenceCos/EvidenceRel/Exec=
  `0.150/0.794/0.334/0.163/0.036/0.973/0.279/0.218`，Student=`1.00`，
  Reliability=`1.000`，rho保持冻结上限，BudgetAbs=`9.192e-03`；e96自然完成；
- exact HEAD/config/source tracked及all status clean；main PID=`404782`、8 workers、唯一GPU process，约
  `8,626 MiB`；checkpoint=`0`，异常与AMP数值warning扫描均=`0`；
- 当前判断=`继续`；原因：e90评测完整，训练、数值及冻结执行边界正常，中间性能不用于裁决。
