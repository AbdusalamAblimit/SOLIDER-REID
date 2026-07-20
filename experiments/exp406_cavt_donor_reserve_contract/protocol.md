# exp406 CAVT P0B donor-reserve合同协议

## 目的与边界

本实验只修复exp405 preflight把512子集误作完整donor universe的候选池合同，不修改CAVT teacher、正式科学门、
pose/CLIP语义、删除比例、反事实臂、统计阈值或后继student。exp405的started seal、failure receipt、源码和输出
永久只读；`exp405-p0b-preflight-v1`禁止重跑、补跑或授权formal。

## 冻结身份

- execution：`exp406-p0b-preflight-v1`
- output：`/home/afr/reid-clean/audits/exp406-p0b-preflight-v1`
- formal execution：`exp406-p0b-iso-teacher-v1`
- formal output：`/home/afr/reid-clean/audits/exp406-p0b-iso-teacher-v1`
- asset root：`/home/afr/reid-clean/assets/exp406-p0b-preflight-v1`
- runtime：Conda `MMPOSE-ABU`，固定Python路径
  `/usr/local/anaconda3/envs/mmpose-abu/bin/python`
- official data：只读`/mnt1/afrdata`
- pose：只读`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`

所有execution/output/started/FAILED/COMPLETE/cache/asset manifest与exp405完全不相交。exp406不得接收任何
exp405 receipt、cache、pair map、MAD或path mapping作为输入。

runner、core、teacher必须是当前exp406隔离仓库中本目录的固定文件；pose路径和manifest SHA、CLIP SHA及
MMPOSE-ABU解释器路径均在runner中硬绑定。CLIP/pose regular-file与字节SHA必须在取得started seal和初始化CUDA
之前通过。COMPLETE必须绑定started.json与排他started seal，formal必须复核该完整provenance，不能仅凭
result/cache/COMPLETE三件套授权。

## Core与尺度

按exp405冻结hash从official有序manifest重新得到512 core；五槽各4 recipient。四变量`mass_log / centroid_y /
confidence / support`的per-slot MAD仅由core中analysis-valid样本计算一次。global cosine保持原排序字段。recipient、
MAD、caliper `8.0`在所有扩池阶段恒定。

## Donor-only冻结顺序

排除core后，按camera分组；组内按
`SHA256("exp406-donor" || camera || PID || relative_path)`升序，camera升序round-robin合并。累积pool总规模严格为
`512 / 1024 / 2048 / 4096 / 8192 / 15618`。preflight先按official顺序一次性original编码全train，每图恰好
一次，再在内存中按上述前缀显现donor；扩展图不进入recipient、core MAD、semantic summary或科学统计。

每阶段保持：same-camera、different-PID、analysis-valid、全recipient排除、donor唯一、四变量固定MAD、caliper
`8.0`、`64 -> 128 -> 256 -> full-caliper`一对一增广匹配。完整匹配后不再扩池；full train仍失败则本execution
FAIL。禁止阈值fallback、改序、重算尺度或结果驱动增加新阶段。

每阶段必须持久化四轴median/MAD、candidate/caliper degree、零边recipient、最近primary distance、匹配失败
类别、pool规模和preference扩展层级；这些字段只用于复现和失败归因，不参与后续阶段选择。

若进入Hall失败，每一层`64/128/256/full-caliper`必须记录status与冲突见证；failure diagnostics缺任一全局、
尺度、stage或recipient必需字段即fail closed。public matcher必须自行验证donor plan的stage、prefix、hash和full
universe，不能只信调用方先验。

## Preflight裁决

preflight只评估：

- 数据/pose同步、flip/crop与image SHA；
- region-isolated CLIP image+text readout；
- 25/50/75%删除count/hash；
- same-camera/different-PID wrong-mask实际重编码；
- shape/finite、slot/readout、state/runtime/source恢复；
- fresh/once-only/receipt完整性。

结果必须写`scientific_evaluated=false`。不得计算或解释PID CI、non-torso macro、teacher scientific GO或ReID
mAP/R1。PASS只形成formal manifest候选，不自动启动formal。

## Formal不可变投影

formal重新编码全train、独立计算full-train MAD、五槽各400 recipient，并保留exp405全部coverage、caliper、
一对一、二维反事实、pose/image/text、generic/NULL/random、MVI2P/pose-part/attribute/generic-transport、bootstrap
和non-torso门。preflight cache/scale/pair不得进入formal。formal只接受同source/runtime/assets绑定的exp406
COMPLETE；单独result/cache、exp405 receipt或FAILED共存均拒绝。

## Static/CPU合同

两次fresh执行必须byte-exact，至少覆盖：

1. core无donor、外部前缀有donor的正例；
2. subset-only mutant失败；
3. caliper放宽/删除、MAD重算mutant失败；
4. recipient或donor复用mutant失败；
5. 非冻结前缀/阶段mutant失败；
6. exp405 namespace输入拒绝；
7. formal冻结投影exact；
8. preflight到formal cache泄漏拒绝；
9. asset/runtime/source start/end exact；
10. Python3.8/MMPOSE-ABU下CUDA未初始化。
11. 零/近零MAD轴的`1e-6` floor语义exact；
12. 单点均有edge但Hall全局无解的mutant失败；
13. failure诊断字段缺失的mutant失败。

## 串行授权

`design -> static/CPU x2 -> 三路盲审0B/0H -> fresh资产与远端static -> 独占4090 -> 唯一preflight`。

preflight COMPLETE PASS后也必须先记录结果、提交本地文档、创建并审计fresh formal manifest，才可另行判断formal
是否启动。任何失败写权威receipt并封板当前execution，不调caliper、pool阶段、batch、prompt或loss救同一编号。
