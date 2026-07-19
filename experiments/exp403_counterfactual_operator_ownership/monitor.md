# exp403 ELO-CUR 监控记录

## 2026-07-20 接手与封板复核

- local HEAD=`7d880d2e843d3bb431f87515ba245eea3526b344`，tracked clean；
- remote exp401 HEAD=`11d7a35788c4645c355d96d76a2a4ff20a9801ac`，tracked clean；
- formal config SHA=`c2992bdf4321f906b19eb22dc7ec69a5678498ea0f93bf55a45a15a2e47cea84`；
- 唯一checkpoint=`transformer_120.pth`，SHA=`fe00d08a9a0f651c2c0852c0661e720995a65292459aec9797a359895aa52efc`；
- train/eval process=`0`，GPU=`2 MiB / 0%`；
- exp401/402均不重跑、不补跑。

判断：封板状态与 heartbeat 一致，允许只读研究与本地 CPU 设计，不允许 GPU。

## 2026-07-20 targeted literature/code audit

已审计 CAL、AIM、UCT、Instruct-ReID legacy dual causality，并对照 PGMAN/CIFT、SGFNet 及通用
dynamic-filter/hypernetwork/LoRA先例。结论不是原子首创，而是当前未发现
`evidence-owned operator + matched complete-execution ordering + frozen retrieval controls`同构闭环。

创新门槛：问题/机制/证据=`3/3`，条件允许进入 exp403 design与CPU contract；novelty风险=`6/10`。

当前状态：`GPU NO-START`。

## 2026-07-20 standalone CPU/static contract

`static_cpu_contract.py` 在本地 `.venv`/uv 环境连续执行两次，均为 `26/26 PASS`，两个 result逐字节一致：

- result SHA256=`041cd6d26f1e3469478c902d443f0f211fd329e9f6a89ffc2e85fcef818b4df5`；
- source SHA256=`b0f40b015150942f12b099e54de406faf63baf787e2fd74cc80cca4706a8eefe`；
- compatibility correct/wrong/generic/NULL=`0.9877629876/0.5/0/−1`；
- ordinal gaps=`0.4877629876/0.5/1.0`，hinge=`0`；
- correct evidence grad norm=`1.8288789e-02`；
- wrong/generic/NULL reference evidence grad norm=`0/0/0`；
- `H/U/V/C/Q/K` 六组参数梯度全部finite且非零；
- NULL逐元素exact identity，correct/wrong/generic descriptor均active/distinct；
- same-camera/different-PID donor映射完整、确定且重复exact；
- evidence-ignored、aux-only、reference-not-detached三个mutant全部被抓住；
- CUDA在执行前后均未初始化。

判断：`STANDALONE STATIC-CPU PASS / PRODUCTION IMPLEMENTATION GO / CUDA NO-START`。这只证明数学和
autograd contract可执行，不证明真实 Swin/AMP/检索有效。下一步先在新 config开关下实现生产图及 off-parity/
source CPU contract；未通过前不启动4090。

## 2026-07-20 生产实现前的梯度合同澄清

复核发现，三个reference全部stop-gradient时，`wrong>generic>NULL`两个reference-reference hinge不可能提供
训练梯度；若开放它们的梯度，又会违反“不靠破坏control制造margin”。因此生产目标冻结为
`correct-max(stopgrad(wrong,generic,NULL))`单边compatibility hinge，后两段顺序只记录为诊断。该修订不
改变standalone的26项结果、final retrieval门或ELO结构，只消除一个无梯度伪目标。

判断：允许继续生产实现；GPU仍为`NO-START`。

## 2026-07-20 生产 CPU/source 门

按用户要求不再扩张重复 CPU 矩阵，只执行一次必要生产合同。结果`34/34 PASS`：默认关闭时 D0/C0相对
实现前commit=`0722176`的state、初始化RNG与输出逐tensor exact；ELO无slot expert、六组linear无bias、
NULL exact identity；mini-Swin三个Stage-3 reference完整no-grad重放，correct输出和全局RNG相对correct-only
exact；student evidence与12组共享生产参数梯度均finite/nonzero；strict reload、optimizer覆盖、teacher/generic-
free state和generic资产SHA/metadata正反校验全部通过。CUDA未初始化。

result=`production_cpu_result.json`，当前判定：`PRODUCTION CPU PASS / FRESH ASSET + CUDA PREFLIGHT GO / FORMAL NO-START`。
