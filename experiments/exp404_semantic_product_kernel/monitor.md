# exp404 SPK 监控记录

> 当前：`C-TRACK PRODUCTION CPU PASS / CUDA PREFLIGHT AUTHORIZED / GPU NO-START`

## 2026-07-20：目标降为C类后的机制准入

用户明确要求不要继续按B类主贡献标准过度筛选，目标为C类会议。研究底线与sealed纪律不变；创新准入调整为
“问题/证据明确 + 一个可执行的适度结构贡献”，不再要求张量积或对比学习原子本身首创。

针对性代码审计：

- ICLR 2023 *Identifiability Results for Multimodal Contrastive Learning*官方commit=
  `c1b361f277deeff645c15aa3c3002de8d275003c`。代码用图像/文本独立encoder和双向对称InfoNCE恢复paired
  modalities的共享factor；它没有open-set ReID final-descriptor ownership与random-key controls。
- CITRIS/iCITRIS官方commit=`95f6c90b9ff769ef0250d3a5434b9352853f4302`。代码与README明确依赖temporal
  sequence、已知intervention target及合成可观测causal variables；当前official ReID数据不满足其可识别性前提。

因此exp404只借用“paired shared factor”作为训练直觉，不声称理论可识别；机制对象冻结为无参数SPK final
descriptor，证据对象冻结为wrong/generic/NULL/bypass/random-key/random-cluster完整终审。按C类门槛，
问题/机制/证据=`PASS/PASS-moderate/PASS`，允许进入static CPU，不授权CUDA/GPU。

## 2026-07-20：standalone static CPU正反合同

冻结source SHA=`9739ad1d8388b45922f2ccdb3fec91ffa77c12d6a2e333e75e43c144c10e9e05`，通过仓库
uv环境、`CUDA_VISIBLE_DEVICES=''`连续执行两次。两次均exit `0`，stdout SHA均为
`6b2ca7d88669238cc9f7bebd04ff21567fe5b7f61a0a3f1dbfa549a909b19a64`且byte-exact。

结果为`17/17 PASS`：

- SPK参数数=`0`，固定`12 -> 4 x 3`toy映射exact；
- NULL factor逐元素为1，NULL输出逐元素exact等于global feature；
- correct相对wrong/generic/NULL/random-key/random-cluster的最小positive utility margin=`0.7641115299`；
- global feature/correct evidence梯度范数=`0.6707680955/0.4743046689`，均finite/nonzero；
- unique random-key保持逐样本norm与绝对值多重集；
- random-cluster为8簇、每簇48 sample、48 PID、双camera；
- donor same-camera/different-PID/no-fixed-point；
- evidence-ignored、auxiliary-only、additive-bypass三个mutant全部被抓；
- CUDA前后均未初始化，official data/pose/cache/checkpoint访问0。

result SHA=`6b2ca7d88669238cc9f7bebd04ff21567fe5b7f61a0a3f1dbfa549a909b19a64`。判定：
`STATIC CPU PASS / PRODUCTION IMPLEMENTATION AUTHORIZED / CUDA NO-START / GPU NO-START`。toy utility不作为
ReID性能或semantic ownership结果。

## 2026-07-20：production实现与CPU/source正反合同

生产实现只修改默认关闭的四个目标文件：config增加`SPK_ENABLED=False/SPK_GROUPS=16`；TAPF图使用rich RGB
student evidence与原D0 `PoseSpatialGate`，不含C0 static expert或ELO-CUR router；无参数SPK在BNNeck、
classifier、triplet返回与eval descriptor之前绑定final global feature。构建期新增`SPK_GROUPS==16`显式门，
SPK与ELO-CUR互斥。

production CPU v1真实动态路径及40项门均通过，但源码顺序reporter用全文件`string.index`，误命中构造函数中更早
出现的BNNeck文本，得到`40/41`与`PRODUCTION_CPU_FAIL`。这不是模型失败，但按测量纪律保留v1记录，不覆盖：

- v1 contract SHA=`0716dd5db1521d0b4ecf2ea072c7970aa4e3bb89d06344fa8ce43e053a59a26c`；
- v1 result SHA=`086b627d89052ff21e68878f3636e2fc8c1f96fc0b0d051df605e08365ea1f0c`。

fresh v2仅把该reporter改为AST限定`build_transformer.forward`；绑定行`355`严格早于BNNeck行`364`。连续两次
均为`41/41 PASS`且result byte-exact：

- D0/C0相对preimplementation commit=`07ca01c`的state、初始化RNG、output逐tensor exact；
- SPK参数/缓存数=`0/0`，NULL factor全1且float16 descriptor逐元素identity；
- train classifier、triplet返回、eval before/after BNNeck全部读取同一个bound descriptor；
- direct global/evidence梯度范数=`0.1973796189/0.1360884756`，真实forward shell为
  `0.4505997896/0.2918346226`，均finite/nonzero；
- 两个consumer均为D0 `PoseSpatialGate`，旧C0/ELO router type与state key为0；
- strict reload、optimizer覆盖、teacher/generic-free state与evidence-ignored/aux-only/additive-bypass mutant全部PASS；
- CUDA前后均未初始化。

v2 contract/result SHA=
`766ef5ad65e0ee8cbc2643e320fb5c1f4b247664ce459a2ea834a818a3fe78dd`/
`829fcaad9b9aa88f596b4b3ca51180e6e42ce50d488542ae0f8ebdcc27a4f6c8`。

判定：`PRODUCTION CPU PASS / CUDA PREFLIGHT AUTHORIZED / FORMAL TRAINING NO-START / GPU NO-START`。下一步只
允许创建fresh config与必要CUDA/AMP preflight；尚无ReID性能或semantic ownership结论。
