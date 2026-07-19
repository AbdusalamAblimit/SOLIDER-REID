# exp394 Phase 0R-128：train-only执行预算冻结协议

## 状态与目的

`PROTOCOL-FROZEN / AUDIT SEALED-PASS / rho_star=0.08075544983148575 /
PRODUCTION NO-START / FORMAL NO-START`。

本阶段只测exp387 clean D0两个已训练活跃consumer在固定train RGB上的真实production applied delta，
为exp394的无grad预算冻结唯一`rho_star`。它不实现exp394 production model，不训练、不构建optimizer，
也不读取query/gallery或任何检索指标。

## 冻结输入

- execution repo：`/home/afr/SOLIDER-REID-exp387-d0-0d1822a`；
- exact HEAD：`0d1822a07dda8daac0210b68916035b1886d5d99`；
- config：`configs/occluded_duke/swin_tiny_tapf_d0.yml`，SHA256=
  `510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b`；
- checkpoint：`log/occluded_duke/exp387_clean_swin_tiny_d0_s1234/transformer_120.pth`，SHA256=
  `59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069`；
- fixed codebook：
  `/home/afr/reid-clean/audits/exp393_phase0e/phase0e_128_codebook.json`，SHA256=
  `4a671a70e0744edad88f911ce628d421650cb09453eb511a61e8d01c239269ef`；
- selection SHA256：
  `7f3f7626c84553416f39c72be0c15ab430458aa7b201c4bf64461990bbdf15e3`；
- data root：`/mnt1/afrdata/Occluded_Duke`。

selection必须保持seed=`20260719`、128 path/128 PID唯一、fit/audit=`64/64`、全部
`bounding_box_train/`且RGB存在。不得因delta分布重新抽样、去零、截尾或只取某个PID/slot。

## Production seam与冻结公式

clean D0的两个consumer是Stage-3两个Swin block之后串行执行的`PoseSpatialGate`。对bank
`k in {0,1}`，hook只读取真实forward的input token `x_k`与output gated token `y_k`：

```text
a_k[b,p,c] = y_k[b,p,c] - x_k[b,p,c]
r_k[b,p]   = sqrt(mean_c(a_k[b,p,c]^2))
```

`a_k`已包含D0的`release * tanh(delta)`和输入token幅度，不能用日志中的pre-tanh `GateAbs`替代。
两个bank各自报告全`128×48` token的count/min/mean/std/P25/median/P75/P95/max及nonzero fraction。

唯一预算定义在看结果前冻结为：

```text
rho_star = median(concat(r_0.flatten(), r_1.flatten()))
```

两个bank token数相同，因此pooled median给两个consumer等权；包含所有token，不按图、PID、bank或
幅度过滤。exp394的`bhat`按token做channel RMS=1，故`rho_star`直接使用与D0 feature delta相同的
单位。该值不按final性能搜索，也不使用尚未实现的exp394中间输出修正。

## 执行方式

1. exact HEAD、tracked source、config/checkpoint/codebook/selection SHA全部先验校验；
2. 从official `OccludedDuke.train`重建selection index/path/PID，验证15,618记录与128项逐项一致；
3. 使用D0原生RGB-only eval transform，一次性冻结128张CPU input tensor及RGB manifest SHA；
4. strict load唯一checkpoint，全部state finite，`eval+no_grad`且不创建optimizer；
5. 第一遍`pose_batch=None`，第二遍传入exploding dummy pose；descriptor、两个bank applied delta与
   input tensor必须逐tensorexact，证明repeat与RGB-only；
6. 只在唯一空闲4090串行执行，记录model-only两遍耗时、images/s与peak allocated memory；
7. checkpoint文件SHA和model state SHA前后exact，hook全部移除，进程自然退出后GPU回空闲。

## 正式门禁

以下全部PASS才允许封板`rho_star`：

1. official train=`15,618`，selection count/path/PID/prefix/file=`128/128`且fit/audit=`64/64`；
2. execution HEAD、tracked source、config/checkpoint/codebook/selection SHA exact；
3. strict load missing/unexpected=`0/0`，全部checkpoint/model/output finite；
4. 两个bank每遍都exact覆盖`128×48`个token，applied delta shape一致且nonzero；
5. None与exploding-pose两遍descriptor及两个bank delta逐tensorexact；
6. `rho_star` finite且`>0`，不做任何性能或样本选择；
7. state/checkpoint SHA前后exact、hook removed、无optimizer、异常词0；
8. 审计进程自然退出且GPU恢复空闲。

任一FAIL先保留result/runner并归因，只阻断当前预算接口，不修改阈值或换样本。PASS只冻结
`rho_star`并授权下一步production实现前的static设计审查；不直接授权production model/config、
CUDA训练preflight、正式训练或semantic multi-stage。

## 封板结果

正式审计严格覆盖official train=`15,618`与sealed 128图/128 PID，fit/audit=`64/64`。两个bank
各覆盖`6,144=128×48`个token，全部RMS非零且finite：

| bank | median | P95 | mean | min | max |
|---|---:|---:|---:|---:|---:|
| 0 | `0.0376448072` | `0.1426717915` | `0.0480008594` | `0.0031023663` | `0.1795158237` |
| 1 | `0.1204396486` | `0.3115715981` | `0.1398641857` | `0.0117391217` | `0.3967587054` |
| pooled | `0.0807554498` | `0.2651471898` | `0.0939325225` | `0.0031023663` | `0.3967587054` |

按冻结公式得到唯一`rho_star=0.08075544983148575`。None/exploding-pose两遍descriptor和两个bank
applied delta逐tensor exact，exploding pose访问数=`0`；223-state SHA前后均为
`c75e9d2e26f83255ae122a6c84b1717bc9474493453c7e04d95163da3cea96a3`，checkpoint SHA前后exact，
16项gate全部PASS。script/result/runner SHA256分别为
`628ce2f88a868ccb2a14f5c0a3204099332253e392bf8c271dd53301057222a3`、
`4f20bef4539129d0e2a9250262b7a09ee7feee03a80fbd2c5491e3450e0d1715`、
`7142cdb1cfd194262ef7daf6c4e3e9823bf561080ec8227143e55408d464887d`。

裁决：`PHASE0R_128_PASS / RHO FROZEN`。该PASS只授权production实现前的static设计审查，不把固定
预算本身写成贡献，也不授权CUDA训练preflight、正式训练或semantic multi-stage。
