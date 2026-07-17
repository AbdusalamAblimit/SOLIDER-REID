# exp371 CASD：support sampler 可兑现性审计

日期：2026-07-14

状态：**历史审计。Gate C formal frozen oracle 已正式 NO-GO，CASD student 不再实现；本文件只保留 batch-local donor 不可兑现的协议证据。**

## 结论

Gate C 的 metric-free dry-run 已证明：在完整身份 support pool 中，五折 query/PID coverage 最低分别为 `93.48%/92.87%`，每个 eligible query 均可固定取得三名 cross-camera donors，禁止的 path/content 泄漏为 `0`。

但这不等于现有普通 `RandomIdentitySampler(P×K,K=4)` 能在训练 batch 内兑现相同机制。对 Occluded-Duke 真实 `train.list` 做 1000 epoch Monte Carlo 后，anchor 在同组另外三张图中恰好拥有三名 cross-camera donors 的覆盖仅为：

```text
min  = 30.3758%
mean = 31.4848%
max  = 32.6611%
```

该结果显著低于预注册 `70%` 门槛。因此，即使 formal frozen oracle 数值通过，也**不得直接启动 batch-local CASD student**。

## 输入与协议

- 数据：`/home/afr/SOLIDER-REID-exp371-146bc44/data/occluded_duke/train.list`
- 数据软链接实际根目录：`/home/afr/SOLIDER-REID/data`
- `train.list` SHA256：`dadffee79d8601545ca2217a38406c1cb6dab39d0b4b0c6370c8486738dee059`
- 规模：`702 PID / 15,618 images`
- 每 PID 图像数：`min=6 / max=426`
- sampler 实现：`datasets/sampler.py::RandomIdentitySampler`
- 固定条件：`K=4`，batch size 不变
- Monte Carlo：1000 epoch，Ruby `Random` seed=`371000..371999`
- 每个 PID：camera-agnostic shuffle 后连续四张切组，不足四张的尾部丢弃
- 对每个 anchor：统计组内其他三张图中 `cam_donor != cam_anchor` 的数量

Ruby RNG 与 Python `random.shuffle` 不逐 bit 相同，因此这里是复现相同分组规则的预注册 Monte Carlo，不冒充某次真实 Python epoch replay。结果另用不依赖 RNG 的超几何闭式期望核对。

可复现脚本：`sampler_realizability_sim.rb`

原始聚合结果：`sampler_realizability_result.json`

## 1000 epoch 原始统计

每个 epoch 使用 `14,528` 个 anchors，因每 PID 尾部不足四张而合计丢弃 `1,090` 张；1000 epoch 总计 `14,528,000` 个 used anchors。

| cross-camera donor 数 | anchor 数 | 比例 |
|---:|---:|---:|
| 0 | 1,247,348 | 8.5858% |
| 1 | 3,052,299 | 21.0098% |
| 2 | 5,654,238 | 38.9196% |
| 3 | 4,574,115 | 31.4848% |

strict-3 每 epoch：

| 统计 | eligible / 14,528 | coverage |
|---|---:|---:|
| min（epoch index 763） | 4,413 | 30.3758% |
| mean | 4,574.115 | 31.4848% |
| max（epoch index 518） | 4,745 | 32.6611% |

超几何闭式 sample-weighted 期望为：

```text
donor=0/1/2/3: 8.3713% / 21.1140% / 39.1167% / 31.3979%
```

仿真与闭式结果一致。若再要求三名 donor 彼此来自三个不同 camera，闭式概率只有 `4.5013%`；当前 Gate C 并未提出这一更强条件，只要求每名 donor 相对 anchor 跨 camera。

## 全局池与 batch-local 的差异

完整训练身份池中，`15,502/15,618 = 99.2573%` 的 anchors 所属 PID 至少存在三张异 camera 图；仅 `116` 张不满足。因此困难不是数据集中没有 support，而是普通 sampler 不按 camera 组织每个四元组。

必须区分：

```text
完整身份池静态可用率：99.26%
普通 K=4 batch 内实际 strict-3：31.48%
Gate C 验证协议五折最低 query coverage：93.48%
```

Gate C 与 student 若使用不同 support pool，必须显式解决协议落差，不能把 oracle coverage 写成普通训练 batch coverage。

## 正式 student 前的唯一可接受方案

优先采用**离线冻结 donor index**，不改变 batch size，也不把 sampler 主效应混入方法：

1. student 主 batch 仍用普通 `P=16,K=4`；
2. teacher support 只来自 train-only frozen cache；
3. 对每个 `(anchor, relation endpoint)`，按预注册 path/content hash 从同 PID 全训练池固定选择三名 donor；
4. donor 必须同时满足：排除 anchor、排除 relation endpoint、path/content disjoint、相对 anchor cross-camera；
5. 选择不得读取 feature、pose response、训练 loss 或验证指标；
6. 所有 CASD 与强对照共享完全相同的 donor paths、eligibility、active-slot mask 和 loss denominator；
7. 测试时删除该 index/cache，student 仍只能输入单张 RGB。

离线 index 在任何实现前仍要补四类 coverage：anchor、PID、正 relation、负 relation。每类必须 `>=70%`，且每个 eligible relation 恰好三名 donor；不足时该 relation 的 CASD loss 必须为 `0`，不得回退到：

- same-camera donor；
- self/current；
- relation endpoint；
- 重复 path/content；
- 重复抽同一 donor；
- 一名或两名 donor；
- queue/memory 或动态 feature-selected donor。

若不采用离线 index 而改用 camera-aware sampler，则必须额外执行 sampler×method 2×2：

| sampler | 无 CASD | CASD |
|---|---|---|
| standard | `B0-standard` | `CASD-standard(masked)` |
| camera-aware | `B0-camera-aware` | `CASD-camera-aware` |

方法主效应只能写 `CASD-camera-aware - B0-camera-aware`；同时报告 sampler-only 主效应和交互。camera-aware sampler 不得通过排除低 camera PID、重复图、改变 PID 频率或改变 steps 来虚增 coverage。

## Fail-closed 记录项

正式 student 每个 epoch 必须记录：

- anchor/PID/positive-relation/negative-relation coverage；
- 0/1/2/3 donor 直方图；
- eligible 与 active-slot mask SHA；
- forbidden self/endpoint/path/content overlap 计数；
- 每个 arm 的实际正负 loss mass、有效样本数和 denominator；
- 每 PID inclusion probability 与 camera 分布。

若离线 strict-3 relation coverage 仍低于 `70%`，CASD student 直接 NO-GO，不扫 donor 数、queue、temperature、slot 数或 loss 权重。
