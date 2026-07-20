# exp404 SPK 监控记录

> 当前：`C-TRACK DESIGN / STATIC CPU PENDING / GPU NO-START`

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
