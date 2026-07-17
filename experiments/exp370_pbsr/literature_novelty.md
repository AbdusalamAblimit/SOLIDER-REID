# exp370 PBSR：代码级查新与新颖性边界

## 结论先行

当前可继续的不是“LGPA 换掉 CLIP query”或“再做一个 pose token decoder”，而是把部位结构从**终端检索分支**改成**主表征内部的分解—重组算子**。

暂定方法名为 **PBSR（Pose-Supervised Bidirectional Structural Routing）**。其可主张的新机制必须同时满足：

1. 姿态只监督路由矩阵，不作为前向 attention bias，也不在推理时输入；
2. 同一个结构路由矩阵完成 `spatial → slots` 的分解和 `slots → spatial` 的重组，而不是两套互不约束的 cross-attention；
3. 重组结果改变标准 global descriptor，最终检索不使用 part matching、MaxSim 或额外 pose；
4. 姿态监督对 backbone 建立梯度防火墙，identity loss 则可以通过重组路径训练主表征；
5. 写回为零初始化残差，初始化时逐元素退化为原 baseline。

对下表最近邻论文和官方代码的核对中，尚未发现同时满足以上五点的直接先例。因此 PBSR 获得的是**有条件查新通过**，不是“所有组件均首次出现”的过强结论。若后续检索发现共享路由重组主表征的直接先例，必须在训练前重新裁决。

## 核对范围与版本

| 工作 | 论文/代码版本 | 本地证据 |
|---|---|---|
| PAFormer | arXiv:2408.05918v1 | `tmp/pdfs/paformer.{pdf,txt}` |
| ProFD | ACM MM 2024；官方代码 `14e47d3b04f541d2a614482848bba2071bc90cda` | `tmp/third_party/ProFD` |
| KPR | 官方代码 `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` | `tmp/third_party/KPR` |
| BPBreID | 官方代码 `a2dc4304284784d8b2c061764512f0698aa38c21` | `tmp/third_party/BPBreID` |
| PFD | 官方代码 `a999c3a4ad6b1b8e3513cbb9145847668e7930fc` | `tmp/third_party/PFD` |
| PAT | 官方代码 `104d42e8292f7e5d534689ded15e4afafb453785` | `tmp/third_party/PAT` |
| TSD | ICASSP 2024；官方代码 `7d5dd416939e922ec893469497853a7d67c3b3df` | `tmp/third_party/TSD` |
| PGDS | AVSS 2024；官方代码 `d3567d9e90dfc16fa314b80c4965fbdc67546b54` | `tmp/third_party/PGDS` |
| PGFL-KD | ACM MM 2021 | `tmp/pdfs/pgfl_kd.{pdf,txt}` |
| 项目历史 STD-PR | exp161 | `model/modules/structural_routing.py`、`experiments/exp161/` |

此外，使用 arXiv API 对 `person re-identification + pose / slot / bidirectional / recomposition / write-back` 做了补充检索，快照保存在 `tmp/pdfs/arxiv_*_reid.xml`。检索只能支持“在已查范围未发现”，不能替代正式审稿阶段的完整相关工作检索。

## 最近邻逐项差异

| 工作 | 已覆盖内容 | 是否改变 global 主表征 | 推理是否需要外部姿态/解析 | 与 PBSR 的实质差异 |
|---|---|---:|---:|---|
| PAFormer | learnable pose tokens；热图监督 token-to-patch attention；部位特征和可见性预测 | 否；CLS 走独立 global ReID 路径 | 否 | 已直接覆盖“训练期姿态监督部位 query”，所以该点不能再主张；PBSR 必须靠共享路由的反向重组和单 global 输出区分 |
| ProFD | text proxy→visual 聚合；先做 visual token→proxy 的 reverse cross-attention | 否；更新的 visual token 副本只在 decoder 内作 key/value，`global_embeddings` 仍取原 CLIP CLS | 否（但输出 part descriptors） | 已覆盖“泛化的双向 cross-attention”措辞；PBSR 不能以“双向”本身为新意，必须证明更新后的主空间/全局表征被检索损失直接使用 |
| PAT | part tokens 与 CLS、patch tokens 共同进入多层 self-attention | 是，存在隐式双向 token 交互 | 否 | 已覆盖“part token 可影响 CLS”；但无姿态监督、无显式共享分解—重组算子、无梯度防火墙 |
| KPR | 将 keypoint prompt 编码后直接加到 image tokens，随后形成 global/part features | 是 | 是，prompt 是任务输入 | 已覆盖“姿态信息改写主空间 token”；PBSR 的边界是 pose 为训练期特权监督，推理前向不读取 pose |
| BPBreID | parsing 监督 pixel-to-part classifier，并做 foreground/part pooling | 否；global 是原 spatial feature 的 GAP | 其主设置使用外部分割监督/标签 | 已覆盖解析监督的 part assignment；PBSR 不把 assignment/matching 当创新，而把它用作内部重组算子的监督 |
| PFD | pose heatmap 经 decoder 形成局部特征，并对 global/part 路径加权 | 是 | 是，forward 内调用 pose estimator | 已覆盖 pose-conditioned global；PBSR 要求 train/test 前向一致，eval 即便传入 pose 也必须忽略 |
| TSD | parsing-aware teacher decoder 蒸馏 pose-free student decoder | 否；最终仍是 global + part matching | 否 | 已覆盖“训练期特权人体解析 → 推理期无解析的 part query”；PBSR 不蒸馏终端 part descriptor，而重组标准 global 表征 |
| PGFL-KD / PGDS | 训练期姿态 teacher/encoder 将结构知识迁移到主 ReID encoder | 是 | 否 | 已覆盖“训练用 pose、测试丢 pose”和“pose 影响 global”；PBSR 的候选新意只能是耦合路由的显式中间结构与可验证的分解—重组机制 |
| STD-PR | pose-guided spatial→structural tokens，加 slot self-attention | 否；只把 structural tokens 作为额外终端 part branch | 是 | 已覆盖本项目自己的单向 read 路径；exp161 单独为 58.7%，比 baseline 低 2.4 mAP。PBSR 必须通过 write-back 独立消融证明不是 STD-PR 改名 |

## ProFD 风险的代码级裁决

ProFD 的 `SemiAttentionDecoderLayer` 先执行：

```text
keys = visual tokens
keys <- keys + MHA(query=keys, key=proxy, value=proxy)
proxy <- proxy + MHA(query=proxy, key=visual, value=visual)
proxy <- proxy + MHA(query=proxy, key=updated keys, value=updated keys)
```

但是 `PartFeatureDecoder.forward()` 最终只返回更新后的 proxy；`visual`/`keys` 不返回。主模型的 `global_embeddings` 仍然直接取 `image_features_proj[:, 0]`。因此：

- “视觉 token 与 proxy 双向交互”已有直接先例；
- “把结构重组结果写回实际用于检索的主空间/global 表征”在该实现中没有发生；
- PBSR 必须在论文和消融中把这两个概念分开，不能笼统声称 ProFD 是单向方法。

## 历史证据对设计的约束

1. `exp320`：完整系统中 `POSE_LGPA_DETACH=False` 相对对照约 `-6.4 mAP`。这否定“让 pose/part 辅助损失直接塑形 backbone”的简单做法。
2. `exp161`：单向 structural routing 为 58.7%，相对 baseline `-2.4 mAP`。这否定“只换成 learnable part decoder”足以成立。
3. `exp336/337`：Swin 上 LGPA 的增益来自 pose 空间信息，CLIP 文本语义不是必要来源。PBSR 应删除 CLIP 文本依赖。
4. `exp353` 等隔离结果说明 pose 分支在弱设置中可以有信号，但额外分支梯度容易与强系统冲突。因此 pose assignment loss 只能更新 router，不能更新 backbone；主表征只接受 identity loss 经过零初始化重组路径的梯度。

## 可以主张与不能主张

### 若实验成立，可以主张

- 将人体结构从终端 part descriptor 改为主表征内部的低秩路由基；
- 用同一 pose-supervised routing operator 耦合结构分解与空间重组；
- 利用梯度防火墙把 noisy pose supervision 与 identity-driven representation update 解耦；
- 推理只输出标准 global descriptor，不需要姿态、解析或 part matching；
- 跨 CNN、普通 ViT、层次 Transformer 的通用插拔性（必须有实验后才能写）。

### 无论结果如何，都不能主张

- learnable pose/part query 是首次提出；
- cross-attention part pooling 是首次提出；
- 双向 attention 本身是首次提出；
- 训练期使用 pose、推理期丢弃 pose 是首次提出；
- matching、GCN、CLIP text semantics 是本文创新；
- 仅凭正确 pose 优于 shuffled pose 就证明完整方法新颖。

## 查新门禁

当前状态：**有条件通过，可进入设计和无训练实现审查**。

正式训练前仍需满足：

1. 代码实现必须使用共享路由矩阵做 read/write；若改成独立第二个 cross-attention，则退化为 ProFD/PAT 邻域，查新失效；
2. eval 路径不得读取 heatmap；
3. retrieval descriptor 只能是与 baseline 同维度、同评测路径的 global feature；
4. 必须有 `write-back off`、`independent write` 或等价对照，证明有效点来自耦合重组而非加参数；
5. 必须有 correct/uniform/shuffled supervision，证明 pose 监督提供了结构因果信号。
