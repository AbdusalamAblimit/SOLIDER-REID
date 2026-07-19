# exp401 rich-budget C0正式训练监控

## 当前状态

`DESIGN-FROZEN / PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS /
FORMAL FRESH-EXECUTION GO`

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
