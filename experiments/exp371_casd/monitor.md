# exp371 CASD 监控

## 当前状态

- 阶段：Gate C 唯一正式 frozen routing screen 已完成；**CASD 正式 NO-GO**；未启动训练
- 主方案：CASD（Cross-instance Allocation Support Distillation）已停止，不再进入 student
- IPER 位置：Gate B 的 correct-effect 门禁失败，已正式停止，不再作为主方案或辅助权重
- 当前训练进程：无
- 当前 GPU 占用：formal oracle 已自然结束；未启动后续训练

## 已完成

- [x] 审计 `exp109/148/335/336/337/340/353/357/358/370` 的真实证据边界
- [x] 核对 LGPA descriptor 真实维度和 test-time pose 依赖
- [x] 核对 PAFormer、PGFL-KD、TSD、PFD、BPBreID、KPR、PAT、SAP、ProFD、DROP、PASS、SPT、PGMAN
- [x] 核对 PDiscoNet、Invariant Slot Attention、SoftMoE、OT/privileged KD、residual/exclusive KD 邻域
- [x] 二次查新发现 AAAI 2020 UMTS 已覆盖 multi-shot teacher → single-shot student，并据此把 CASD 收紧为 part-wise leave-one-view-out support advantage
- [x] 发现 2022 `Pose-guided counterfactual inference` 精确撞名并降低 IPER 优先级
- [x] 排除 CLIP query 换皮、普通 pose KD、slot/write-back、OT/MoE、masking、matching 路线
- [x] 冻结唯一主方案与停止规则
- [x] 创建长期 Goal；其总目标仍是保留 LGPA 涨点并改造成自有创新
- [ ] 替换 Goal 的旧 IPER 主方案正文：工具不支持原地编辑 active Goal，需清空/结束旧 Goal 后按 CASD 正文重建；在此之前以本设计为执行真值
- [x] 在 4090 找回 exp340/340c 的原始 checkpoint、train/test logs 与 SHA；canonical fixed-random `59.9/68.7` 高于 CLIP `59.5/68.1`，共同 global `58.8/67.8`
- [x] 实现 query mode 枚举；random-frozen/random-learned 初值逐 bit 相同，仅差 3072 个可训练参数
- [x] 实现 Gate B 五臂评测与缓存脚本；shuffled 为 query/gallery 内异 PID 双射，uniform 为 common-body-support
- [x] 实现 Gate D train-only JL/PCA-768 oracle 与 paired-gain retention
- [x] 本地 uv 环境 11 项单元测试通过，Python compile 与 `git diff --check` 通过
- [x] 3090 完整模型 query 接线 smoke 通过；4090 execution 的 11 项测试通过
- [x] Gate B correct parity 通过：`59.8357 / 67.6018`，复现 exp336 s0 `59.9 / 67.6`
- [x] Gate B 五臂完成；五臂 global SHA 完全一致，descriptor 均为 `7×768=5376-D`
- [x] Gate D 单 seed 完成：train-only PCA-768 为 `59.9336 / 67.8733`，paired-gain retention=`1.1158`；固定 JL-768 失败
- [x] Gate T target-only 推理干预完成：`59.8121 / 67.5113`，仅比 scene-merged correct 低 `0.0236 mAP`
- [x] 外部系统查新完成：MVI²P 是 CASD 第一直接邻居；AERC 被 NNCL 机制级覆盖并独立 NO-GO
- [x] Gate C 正式 frozen oracle 完成：POSE-RESP 相对最强 `PART-EQUAL` 为 `-0.0766` mAP pp，五折全负，正式 NO-GO

## Gate B / Gate D 单 seed 结果

| arm | mAP | R1 | 相对 global mAP | 解释边界 |
|---|---:|---:|---:|---|
| global | 58.9908 | 67.3756 | — | 同一 checkpoint 的共同 global |
| correct | 59.8357 | 67.6018 | +0.8449 | 原 exp336 scene-merged pose |
| target-only | 59.8121 | 67.5113 | +0.8213 | 同一权重，仅把 person-0 目标人物 heatmap 送入 LGPA |
| canonical | 59.7374 | 67.6471 | +0.7465 | 固定 canonical 只比 correct 低 0.0984 |
| shuffled | 59.8037 | 67.7376 | +0.8129 | 异 PID 双射 donor pose 只比 correct 低 0.0320 |
| uniform | 59.3689 | 66.8326 | +0.3781 | 删除通道特异结构但保留 foreground support |
| no-pose | 59.4014 | 66.6063 | +0.4106 | 同一 pose-trained head 的推理干预，不等于 exp337 重训 |

五臂的共同 `global_sha256` 为：

```text
e5c3a041d6fe930c4c074ee3d7bdec1bea984503ff1c184f8f5cbf7ddfc0d310
```

单 seed packing：

| method | dim | mAP | R1 | retention | 判断 |
|---|---:|---:|---:|---:|---|
| full equal-concat | 5376 | 59.8357 | 67.6018 | 1.0000 | reference |
| fixed JL | 768 | 58.8011 | 67.5566 | -0.2245 | NO-GO |
| train-only PCA | 768 | 59.9336 | 67.8733 | 1.1158 | provisional GO |

PCA 只在 `train_loader_normal` 上拟合，train/eval path overlap=`0`；该结果只说明“线性 learned packing 可行”，不说明任意随机压缩可行。最终同维 claim 仍需三 seed paired 验证。

Gate B 的机制结论必须收紧：LGPA 的局部融合增益真实存在，但当前图精确姿态只解释很小部分；更可靠的资产是**结构化局部分解**，不是实例级精确姿态对齐。后续在 target-only / support-routing 门禁通过前，不把 `anatomical pose support` 当作已成立事实。

小型原始日志、manifest 与 results 已回传至 Git 外：

```text
remote_artifacts/exp371_30aca94/
```

## 执行裁决

- [ ] Gate A：canonical CLIP/random 已闭合；correct-pose random-frozen/random-learned 不再占用资源
- [x] Gate B：exp336 checkpoint inference intervention 矩阵（s0）
- [x] Gate C：三 cache dry-run与唯一正式 frozen routing screen 均完成；`all_pass=false`
- [x] Gate D：5376-D→768-D frozen oracle（s0 provisional）；因主机制判负，不补三 seed
- [ ] Phase 1：**禁止执行**；Gate C 失败不能由 student 补救

## 当前判断

**CASD 正式 NO-GO；停止 LGPA→CASD 自有化，不实现或训练 student。**

内部 `exp120/123/125/129/130` 可以作为动机历史，但“过去失败、现在成功”只有在新机制越过强 control 时才成立。本次 formal oracle 没有成功：same-ID、part-aligned support 很强，但 POSE-RESP 低于 `PART-EQUAL`、`POSE-SCALAR` 与 `RESP-PERM`，scene 协议也五折全负。因此不能把 generic multi-view/part support 重命名为 CASD 创新。

严格停止范围：不转 AERC、OT、MoE、slot 数、temperature、queue 或 loss-weight 小变体；不进入三 seed、ResNet/ViT 或多数据集。该裁决不否定 LGPA 的 `global+parts` 性能资产，只否定“逐图 pose response 组织跨实例 support”作为其可投稿的新机制。

## 2026-07-13 23:27 target-only cache 首批安全退出

- 唯一提取进程 PID=`4168018`；未启动训练，首批断言失败后已退出。
- `equal_concat` 与 `maxsim_hybrid` 的 part block 最大差为 `2.37673521e-06`，通过 `2e-05` 门限。
- global block 最大差为 `2.58302316e-05`，超过 `2e-05` 门限，未生成任何 `.pt` cache。
- 根因不是 target-only pose 接线漂移，而是 orig/flip 融合的运算顺序不同：`equal_concat` 是“各视图先归一化，再平均并归一化”；原 cache MaxSim 路径是“先平均 raw，再归一化”。当 orig/flip norm 略有差异时两者不严格相等。
- 修复不放宽阈值，只让 MaxSim 元数据路径复现 `equal_concat` 的既有归一化顺序，并新增不等 norm 的回归测试。本地 uv 全套相关测试 `12 passed`；远端通过后才允许重新启动唯一提取任务。

## 2026-07-13 23:44 target-only cache 完成与独立审计

- 修复提交：`f5b8b61`；远端脚本 SHA=`9e61cd36bbdc7180df3dc0f4a54ac8bde4fa5b7e15217a96c6e1a2acd23875bb`。
- 唯一重启进程 PID=`4168962` 已自然退出并打印 `COMPLETE`；未启动训练。
- train cache：`15,618` 张，文件 SHA=`1046a6df1036fad5bd6865c920150ff09106116bfab1bd4c8dc453db7c5a2a4f`。
- val cache：`19,871` 张，其中 query=`2,210`，文件 SHA=`83b170efc31e6a81bb35fad428d04a27287183defa9f2edacc37a47e552d95e4`。
- 两个 cache 的 manifest SHA、全部七类 tensor SHA、样本数、有限值、target-person validity 均独立复算通过；train/val path overlap=`0`。
- equal/maxsim global、part、allocation 与 orig/flip raw-response 的最大差均为 `0`；train/val target valid 分别为 `15,618/15,618`、`19,871/19,871`。
- 多人物图数量：train=`4,124`，val=`5,625`。

## 2026-07-13 23:48 逐图内容 SHA 回填

- 回填提交：`94df093`；只生成与 source-cache SHA 和 ordered-path SHA 双重绑定的 JSON sidecar，不重写 cache、不重算 feature。
- train：`15,618/15,618` 个唯一内容，无重复。
- val：`19,871` 条记录、`18,001` 个唯一内容，存在 `1,870` 组二次出现。
- 逐组拆分确认：全部 `1,870` 组均恰好是 `1 query + 1 gallery`、同 PID、同 CAM、相同帧文件名；query-query、gallery-gallery、异 PID/CAM 与成员数大于 2 的重复均为 `0`。
- 这是 Occluded-Duke 标准 query 目录与 `bounding_box_test` 的同图拷贝。后续协议只白名单这一种形式：标准 evaluator 会删除该 gallery endpoint，support donor 还必须按 content SHA 排除；其余任何内容重复继续 fail-closed。

## 2026-07-14 Gate C v2 配对 cache 与 metric-free dry-run

- fixed-three-donor 与 paired extraction 实现提交：`c3129b3 / 1406a03 / 3947c04 / 005ab74`；正式 oracle 脚本 SHA=`9eae0cfaa3c58f03b56cc0395ac57d23f412b06b087904ad57888237cca4ef95`。
- 本地 uv 相关测试最终为 `31 passed`，`py_compile` 与 `git diff --check` 通过；两轮只读红队审计无剩余阻塞。
- canonical val cache：`19,871` 张，文件 SHA=`4de51adf795ad21037250ba626a27a435105f951543480114920045ee13fdfc2`。
- scene val cache：`19,871` 张，文件 SHA=`0517eac367d4f7831cf79c5dacd73b85d6acd26ba1a6304b25899f1cc50fe220`。
- target/canonical/scene 的 target raw-response SHA 均为 `8393eec6190fa54d53eb2f668d3f7e5ae58c7529fd08a69a7322b5aaf143ec7c`；target validity、person count、PID/CAM/path/checkpoint/num-query/block-dim 逐项配对通过。
- canonical/scene 各自 content sidecar 均记录 `19,871` 样本、`18,001` 唯一内容、`1,870` 组标准 query↔gallery 同 PID/CAM 拷贝；source-cache SHA 与 ordered-path SHA 绑定通过。
- 三 cache dry-run 状态为 `DRY_RUN_COMPLETE`、`metrics_computed=false`、`coverage_hard_gate=true`、`max_queries=0`、`cross-camera`；canonical/scene 均已实际加载。
- 五 fold query coverage=`0.9615/0.9543/0.9403/0.9348/0.9348`，PID coverage=`0.9595/0.9557/0.9538/0.9461/0.9287`。
- 每个 eligible query 的 selected donor 恒为 `3`；support/reference path overlap=`0`、content overlap=`0`；五个 active slots 全部有效；forbidden duplicate=`0`。
- 数据 cache 没有 tracklet/frame metadata，`near_duplicate_tracklet_answerable=false`。因此正式结论必须保留“同轨近重复泄漏不可回答”的限制，不能写成已完成 strict tracklet sensitivity。
- 红队新增 `POSE-SCALAR` donor-quality control、canonical `2×4` E×R、student routing×transfer `2×2` 与 matched anchor-inclusive control；这些都在看正式指标前冻结。
- 小型 manifest、dry-run、sidecar 与 stdout 已回传到 Git 外 `remote_artifacts/exp371_gate_c_paired_005ab74/`，未下载 700MB 级 cache。
- 2026 第二轮查新新增 MVCD/MHSF 两个 unresolved critical priors；即使 frozen screen 通过，也只能记“内部机制可行、外部新颖性未决”。

## 2026-07-14 Gate C formal frozen oracle：COMPLETE / NO-GO

- 唯一进程 PID=`4175845` 自然结束并打印 `COMPLETE`；未启动 controller 或训练。
- 正式参数：`max_queries=0`、五折、`cross-camera`、每个 eligible query 固定三名 donor、`2000` 次 PID-grouped bootstrap、CUDA distance。
- target oracle mAP：`PART-EQUAL=94.3121`、`POSE-SCALAR=94.2517`、`POSE-RESP=94.2355`、`RESP-PERM=94.2727`、`SLOT-PERM=93.0774`、`ID-MEAN=93.9357`。
- POSE-RESP 相对最强 `PART-EQUAL=-0.0766` pp；五折差=`-0.1504/-0.0139/-0.0623/-0.0936/-0.1238` pp。
- PID-grouped bootstrap（POSE-RESP−PART-EQUAL）：point=`-0.0765` pp，95% CI=`[-0.1561,+0.0022]` pp。
- POSE-RESP−POSE-SCALAR=`-0.0162` pp；POSE-RESP−RESP-PERM=`-0.0372` pp。
- scene-merged POSE-RESP−本协议最强 routing control=`-0.0868` pp，五折全部为负；canonical `2×4` 矩阵完整但未提供反向证据。
- 唯一通过的核心结构信号是 `PART-EQUAL−SLOT-PERM=+1.2347` pp；wrong-ID 为 `1.2525` mAP，说明同 ID 与固定 slot 对应重要，但不支持逐图 pose-response routing。
- 安全门禁有效：五折 query/PID coverage 均高于 `70%`、selected donor 恒为 `3`、五个 slot 全 active、path/content overlap=`0`、forbidden duplicate=`0`。`near_duplicate_tracklet_answerable=false` 限制继续保留。
- 总门禁：12 项中 5 项通过、7 项失败，`routing_screen_all_pass=false`、`all_pass=false`。
- SHA：raw `results.json=2213d91fdf4594409d38e4ce2ab7c03dccdef8e1390cd9bcb3837f92006b429f`；manifest=`bb977bdc5b80d05370306be77c874bcaa160406acb7b946895160cb85e9a797d`；stdout=`2565890f4d9f05dd791fb0f6a0d972d4068c0ba83e0cae6cec4006d92adf1483`。
- 本地 Git 外结果包：`remote_artifacts/exp371_gate_c_formal_005ab74/`；`results.json.gz` 完整性通过，SHA=`98fbbbaa4584185b9d2f17dbc68d245fa9735f9d428d3cdedfc11e7c7d7a882b`。三份约 700MB cache 未下载。
- 15 分钟 heartbeat automation `monitor-exp371-lgpa-ownership` 已删除；无需继续监控。

## 保护事项

- `experiments/decisions.md` 当前包含用户未提交的 #99/#100 改动；本轮只在末尾追加 exp371 裁决，并在暂存时隔离用户内容。
- 不修改现有 tracked 模型/config，不启动 3090/4090 训练。
- 后续若使用 Python，必须先在工作目录通过 `uv` 建立环境。
- 禁止 Claude；正式训练前的审查改由 Codex 与可复现机制测试完成。
