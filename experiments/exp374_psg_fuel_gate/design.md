# 实验 exp374：PSG 图像—姿态对应依赖门禁

## 当前状态

- 阶段：`PREPARE_A02_FAIL_OFFICIAL_MIRROR_PROTOCOL_REVISION_ONLY`
- 训练：未启动
- 正式评测：未启动
- 代码实现：audit-only runner、协议层、模型三态 seam 已编写；首次 prepare 因历史
  flat 日志路径错绑而在 output 创建前 fail closed，修复后本机 formal 20/20 与
  regression 87/87 已从头 PASS；远端历史环境 formal 20/20 与 regression 87/87
  也已绑定 exact execution commit 全量 PASS
- GPU：未占用
- 当前执行许可：A01 已因 signed scene 假设安全失败并永久保留；C+ signed-raw 修复在
  exact `8ca57ed…` 上完成本地/历史环境全量复验后，唯一 A02 prepare-only 又因
  blanket query/gallery RGB content-disjoint 假设与 Occluded-Duke 官方 mirror 冲突而
  fail closed。A02 也已烧毁并保留。当前只允许冻结、审查 official-mirror 关系协议；
  禁止 resume/复用 A02、全新 prepare、Gate A `run`、arm/per-query 指标生成、
  `summarize` 和训练

## 动机

当前 PSG 的性能价值与方法新颖性必须分开判断。

旧协议中，三组 seed 的 no-PSG 与 PSG 均值分别为 `56.50` 和 `57.83 mAP`，
说明 PSG 在当时设置下具有稳定正向趋势；但是 WACV 2020 的 pose-guided gated
fusion、SFT 和 FiLM 已覆盖 backbone 中层的 pose-conditioned spatial/channel
调制。因此，原始

\[
Y=X\odot(1+E(H))
\]

不能继续承担论文主创新。

在设计更复杂的新算子前，必须先回答一个更基础的问题：PSG 的收益是否真的依赖
**当前图像与其正确实例姿态的对应关系**，还是来自额外参数、固定人体位置模板、
训练正则或测试分布外干预造成的假象。exp371 在 LGPA 上得到的
`correct-shuffled=+0.0320 mAP` 不能替代这个检查，因为它干预的是 LGPA，而不是
纯 PSG checkpoint。

## 核心假设

若 PSG 确实利用实例姿态，固定同一 RGB、同一 checkpoint 和同一评测流程后，
正确姿态应稳定优于：

1. 协变量匹配但身份错误的完整姿态；
2. 真正跳过 PSG 的 bypass。

训练集导出的 scene-channel centroid control 和七个解剖组局部破坏只作 secondary
敏感性分析。特别是，多人 scene 的 max-merged channel 可能含多个峰，整体平移不能
消除峰间相对结构，所以它不能被称为固定 canonical 姿态，也不能触发 GO/NO-GO。

只有该假设在三组 seed 上通过预注册门槛，才允许为 PSG 设计下一阶段干净训练门禁。
冻结 checkpoint 的反事实只能证明“已训练模型依赖对应姿态”，不能单独证明 PSG
相对 no-pose 的训练因果增益，更不能证明未来新机制有效。本文中的 Gate A 结论只称
`matched counterfactual dependency/fuel screen`，不称总体因果效应。

## 两级门禁

### Gate A：历史 checkpoint 冻结燃料筛查

Gate A 不训练，只使用 4090 上现存的三枚 PSG-only checkpoint：

| seed | checkpoint SHA256 | 当前 flat 日志 mAP/R1 |
|---:|---|---:|
| 1234 | `51c37c49537119deb38bce08702fb5a3ea7fc2b4bc251f1b8f4eebd9ddf6ec69` | `58.3 / 68.1` |
| 42 | `174e8f9316f60219cbeca292457bf976e73cc88df6fddf9d83f94a89280d2a75` | `57.5 / 66.7` |
| 2024 | `c525e9c1ba90d896b703f6eca9a117ba1a97cd08fbab02618021bf20efd09f3d` | `58.0 / 68.4` |

这些目录被复用过：nested 日志是另一代结果，现有训练日志没有 execution commit。
因此 Gate A 无论结果多好都只能叫 `LEGACY_FROZEN_FUEL_SCREEN`，不能直接进入论文
主表或作为最终可复现因果证据。

### Gate B：干净配对训练门禁

只有 Gate A 明确 GO 后，才另写训练设计并重新进行完整审查。Gate B 至少要求：

1. exact commit、archive SHA、run manifest 和独立 `OUTPUT_DIR`；
2. 同一 `PoseBackboneModel`、同一初始化、同一 sampler、同一 batch size；
3. `correct-pose train`、`matched-shuffle train` 与 `true-bypass train`；
4. 三组 paired seed；
5. 训练前验证共享参数逐项 hash 一致，避免 PSG 模块初始化消耗 RNG 后改变
   classifier/backbone 初始化；
6. 形成 train intervention × eval intervention 的配对矩阵。

Gate B 的代码、配置和资源方案必须另行 PASS，Gate A 通过本身不授权训练。

## Signed raw heatmap 修订：因果输入与统计视图分层

prepare A01 证明“scene heatmap 必须逐元素非负”的原假设与真实历史数据不一致。只读
追溯定位到训练 row 904、图像 `0093_c2_f0068896.jpg`：该图有 6 个有效人，六份 pose
NPZ 保存的是 ViTPose-Huge MSE head 的 raw output activation，经 float16 落盘；每个人
原始 heatmap 都含有限负响应。`_place_heatmap` 和两次 bilinear resize 只传播这些值，
6 人 max-merge 仅在六人同一位置均为负时留下 scene 负值。A01 partial train cache 只有
rows `0..1023` 已 materialize；这 `53,477,376` 个元素中有 10 个负值，均在 row 904
的 0-based channel 6/10，最小值 `-7.5766431e-05`。mmap 未写尾部是零页，因此该比例
不得外推到完整 train split。这不是文件损坏，也不能被事后称为概率或强制截断。

修订冻结为 C+ 双视图，禁止混淆：

1. `Hraw` 是唯一因果输入。cache、manifest、correct、shuffle、group corruption 与
   model override 必须逐值保存/传递原始 merged scene；PSG 继续消费
   `sigmoid(bilinear_resize(Hraw))`。禁止在 model 前 clamp，禁止改变历史 descriptor。
2. `Hpos = clamp_min(Hraw, 0)` 只是统计/几何解释视图。95-D nuisance 中的 L1、peak、
   entropy、support、bbox 与 matching cost 只在局部只读 `Hpos` 上计算；不得原地修改
   `Hraw`。
3. 每个 split 的 premetric manifest 必须冻结：raw min、负元素数、负 sample 数、负
   sample-channel 数、`negative_channel_indices_0based`，以及负绝对质量
   `Mneg(Hraw)=sum(clamp(-Hraw,0))`。这些计数只能在 split 全部 materialize 后按数据集
   index `0..N-1` 聚合，不得把 mmap 未写零页计入分母。transform 名固定为
   `positive_part_v1`。其中 `negative_channel_indices_0based` 唯一定义为该 split 所有
   sample 中出现过负元素的 channel index 的 sorted unique union，按严格升序编码为
   JSON integer array；禁止写成逐 sample 列表、集合字符串或未排序遍历结果。整个
   premetric payload 继续使用 `protocol.canonical_json_bytes`（`sort_keys=True`、
   `separators=(",", ":")`）序列化并参与 execution SHA，resume 必须逐项复算一致。
4. audit override 接受 finite signed float32 raw tensor，仍严格保留 shape/device/
   contiguous 与 audit-only 三态门禁；signed legacy 与 signed override 的 descriptor、
   hook tensor 必须逐值相等。
5. intervention 的 relative-L1 继续直接比较真实 actual PSG input；质心位移的质量改为
   `(actual - 0.5).clamp_min(0)`。在旧 nonnegative raw 假设下这与原定义完全等价，在
   signed raw 下则避免负质量质心。
6. centroid secondary 在 `Hpos` 上拟合 bbox/target/质心与 entropy，但输出必须是同一
   `(dx,dy)` 对 `Hraw` 的 zero-padded 纯平移。必须验证
   `clamp(translate(Hraw),0) == translate(Hpos)`，positive L1/peak/entropy 仍过原门槛；
   若输入 `Mneg>0`，输出/输入负绝对质量比也必须在 `[0.95,1.05]`；若输入 `Mneg=0`，
   输出必须仍为 0，不定义 0/0 ratio。任一失败只令 centroid
   `INVALID_SECONDARY`，不得影响 primary GO/NO-GO。
7. bypass 仍为显式 `None`；不得以 clamp、zero 或 Hpos 冒充 bypass/correct/control。

actual-space provenance 对 active inventory `s3_b0,s3_b1` 分别冻结；每个 block 使用其
真实 `(H,W)`（当前二者均为 `(12,4)`），按上述 split sample order 依次计算
`Sraw=sigmoid(bilinear(Hraw))`、`Spos=sigmoid(bilinear(Hpos))` 与
`Delta=Sraw-Spos`。每个 sample tensor 必须转为 C-contiguous little-endian float32，按
sample 顺序把 bytes streaming 输入 SHA256；manifest 分别保存 `Sraw/Spos/Delta` SHA、
shape、dtype、element count，以及 `Delta` 的 max-abs、sum-abs、mean-abs。相同 shape 的
两个 PSG block 三组 SHA 与差异统计必须完全一致。上述字段属于 canonical premetric
payload，resume 时任何字段漂移都由 execution SHA 与逐项复算共同拒绝。为避免把 CPU
reference digest 冒充真实 hook provenance，这三组 tensor 必须在 prepare 的冻结
`cuda:0` 环境由同一个 `actual_psg_input` 算子生成，manifest 明示 compute device/backend/
operator；正式 `correct_start` 又按 query/gallery、相同 sample order 与 `<f4` bytes 从
真实 PSG hook 重算 `Sraw` SHA，并对两个 active block 逐项 bitwise 对齐。任一不一致
fail closed，不产生指标。

该修订是对审计协议与真实历史输入契约的纠错，不是性能方法改动。A01 及 partial cache
永久保留且不可 resume；只有修订通过独立静态审查、formal 与全部 regression 在本地和
历史远端环境从头 PASS 后，才可为新 exact commit 申请全新 A02 prepare-only 授权。

修订最少测试：

1. person count `<6` 与 `=6` 的 synthetic signed merge，证明历史 merge 未改；
2. signed raw nuisance 等于显式 Hpos nuisance，且 raw tensor/SHA 不变；全负无 positive
   support 仍 fail closed；
3. sign manifest 字段、actual-space SHA/差异与 resume drift；
4. signed override 接受、nonfinite 等旧门禁拒绝，legacy/override exact parity；
5. real Swin formal seam 使用含负 raw，验证事件顺序、actual sigmoid input、state SHA
   与 bypass；
6. signed actual 的 positive-mass centroid：双边无质量为 0、单边有质量为 1；
7. centroid raw translation/Hpos commutation 与正负质量保存，失败只作 secondary INVALID；
8. correct start/end、flat parity、raw cache provenance、shuffle/group raw donor provenance。

## Gate A 干预臂

所有臂固定相同 RGB、checkpoint、resize、descriptor 定义/归一化和距离计算，只改变
送入 PSG 的最终 `scene_heatmaps`；descriptor 数值本身正是干预结果，不可能固定。
历史主口径固定为 no-flip；flip-test 只允许作为 secondary。

### A0 correct

使用数据集中原始 pose bundle，经原 `_prepare_pose` 得到最终 scene heatmap。

在运行任何 control 前，correct-only 必须按以下历史口径做 parity：

- `TEST.FLIP_TEST=False`；
- `TEST.RE_RANKING=False`；
- `TEST.NFC=False`；
- `TEST.POWER_NORM=0`；
- `TEST.NECK_FEAT='before'`；
- `MODEL.POSE_USE_TARGET_HEATMAP=False`；
- `MODEL.POSE_PSG_STAGES=[-1]`。

三个 checkpoint 的 correct mAP/R1 都必须复现各自 flat 日志到打印精度，即四舍五入
到 `0.1 pp` 后完全一致；否则 Gate A 整体 `INVALID`。若补 flip secondary，必须先
固定受控 scene bundle，再同步翻转 RGB 与该 bundle，禁止重新抽 donor。

### Occluded-Duke 官方 query/gallery mirror 关系门禁

A02 在任何 matching、checkpoint forward 或指标读取前安全失败：train/query/gallery
分别已完成 `15618/2210/17661` 条 metadata 物化，但旧
`assert_disjoint_records` 要求任意两个 split 的 RGB/pose 内容 SHA 完全不交。只读审计
证明该假设过强，而不是数据污染：train 与 query/gallery 的 path、RGB content、
pose path、pose content 四类交集全部为 0；三个 split 内四类标识也分别唯一。唯一交集
是 Occluded-Duke 官方 query/gallery mirror：恰有 1870 个一对一 RGB content group，
同时也是恰有 1870 个一对一 pose-content group。每对均同 basename、PID、camid、
viewid、person_count 和 frame，cached `Hraw`、scene score 与 95 维 nuisance 逐值相同；
query/gallery 的真实 path 与 pose-path 标识仍不同。本仓库标准 evaluator 会删除同
PID、同相机的 gallery endpoint；这些 mirror 对其对应 query 是 official junk，但对
其它 query 仍可能是合法 gallery，禁止删除、合并或修改 evaluator。

修订分为结构层 `audit_split_relations_v2` 与 exact-asset 层
`assert_occluded_duke_official_v1`。结构层可以接受 query/gallery 无重复；exact 层才要求
当前官方数据必须精确复现 1870 对。prepare 必须先完成二者，再生成 donor candidate、
读取 checkpoint 或计算任何指标。

#### 路径、标签与 identity split

1. 每个 split 内的 RGB absolute path、RGB content SHA、pose bundle path SHA 和 pose
   bundle content SHA 必须分别唯一。basename 唯一匹配
   `^(?P<pid>\\d{4})_c(?P<cam>[1-8])_f(?P<frame>\\d{7})\\.jpg$`。query/gallery 的
   metadata `pid` 必须等于 basename PID，`camid` 必须等于 `cam-1`；train PID 已
   relabel，只能另存 basename 解析的 `source_pid`。train source-PID 集与 query/gallery
   source-PID 集必须无交集。
2. official list 的每个 stripped nonempty entry 必须本身就是 basename：禁止 `/`、`\\`、
   `.`、`..`、absolute path 或重复。每个 RGB resolved path 必须是预期 split root 的直接
   child：train=`bounding_box_train`、query=`query`、gallery=`bounding_box_test`；不能只
   比 basename 后放行 traversal、symlink escape 或错误 split root。
3. 每个 pose index entry 必须同时冻结原始 `persons` 全列表和 loader 实际使用的
   effective 列表：先取 `persons[:max_persons]`；仅当 `target_person_idx` 仍落在该截断列表
   内且大于 0 时才把目标移到首位，否则保持截断顺序。每个 split 必须另报
   `target_outside_effective_count`，禁止把条件语义写成无条件 target-first。
   每个 entry 必须是 pose split root 的直接 child basename，resolved 后仍位于该 root。
   metadata 新增两套 constituent path/content SHA list；旧聚合 bundle SHA 只能作便捷摘要，
   不能替代 constituent 审计。每个 split 内 constituent path/content 必须唯一；
   train↔evaluation 的任一 full/effective constituent path/content alias 都 fail closed。
4. train↔query、train↔gallery 的四类 bundle 标识与 source PID 均必须零交集。错误码按
   failure predicate 唯一，不再使用宽泛 `INVALID_DATA_RELATION`。

#### query/gallery official junk 的唯一白名单

1. query/gallery RGB path、pose bundle path 和 constituent pose path 必须无交集。
2. 合并后的 RGB-content duplicate group 唯一允许：group size=2、恰有 1Q+1G、各 split
   内 multiplicity=1、basename/PID/camid/viewid/person_count/frame/report 完全相同，且
   仓库标准 junk predicate 对该 endpoint 返回 true。Q/Q、G/G、三成员、异名、标签漂移
   或未被 evaluator 删除均 fail closed。
3. RGB duplicate endpoint set、pose-bundle-content duplicate endpoint set 唯一编码为按
   `(query_index,gallery_index)` 排序的 JSON pair list，二者必须逐项完全相等。所有跨 q/g
   constituent pose-content alias 还必须落在同一 allowed endpoint 且 person position
   相同；full/effective content list 在 mirror 两端逐项相等，禁止部分 alias、跨 mirror
   alias、位置置换或 orphan alias。
4. 两端 cached `Hraw`、scene score、95-D nuisance、frame 与 report 必须 bitwise/逐字段
   相同；同一原图形成不同 PSG 因果输入时禁止白名单化。

#### official provenance 与 canonical digests

官方 source 固定为 `lightas/Occluded-DukeMTMC-Dataset` commit
`dcba185bb20cbd53d3da2c8a4bfc25aa6971ce1d`。raw list SHA256 固定为 train/query/gallery：
`dadffee79d8601545ca2217a38406c1cb6dab39d0b4b0c6370c8486738dee059` /
`fb5e1b1a749a0ab8602414bc9159e7a03216c2bdc519b5a4e513e05e3f612333` /
`0393fa86344ef4c220a5589aaad409f3adda1e14e39fc8425c80e90196065fca`，count 固定为
`15618/2210/17661`。

当前 active pose index 也作为 exact asset 冻结，train/query/gallery 的 byte length / SHA256
分别为：`4713227` /
`63dc1f5db9bab90717a484dfc5033a197ee8b95f9c94a92f2082dc18a588103b`、
`719516` / `6b60745066f9b921d347558db3ad8ee7021ad103182db8afe7fffd510bc5f7c4`、
`5320783` / `d5f2e14f8665ce045dfa8085dbdff031a1c9de7a7c258a594802e2a63ccefabc`。
metadata、official list 与 index 必须从同一个已 `fstat` 前后稳定的文件描述符读取 bytes、
计算 SHA 后直接解析，禁止“先哈希、再另开文件读取”。六个 q/g cache `.npy` 必须在
mmap 前整文件 SHA+identity 绑定，按首维分块验证全局 finite，读取完成关闭 mmap 后再做
整文件 SHA+identity 复核；所有已读取 RGB/NPZ/list/index/metadata 的 identity 在输出前
统一复核。只读诊断必须以 `PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1` 和 Python
`-B -s` 启动，并在导入 NumPy 前 fail closed 验证两项环境与 `sys.flags.no_user_site`。
最终 launcher 必须在执行前后分别复算诊断脚本 SHA，命令行只接受冻结的 absolute
staging/data-root，且 stdout `resolved_inputs` 必须逐字等于这两个 resolved 路径。

所有列表字符串按 Unicode/codepoint 升序。`canonical_json_bytes` 唯一定义为当前
protocol 的 UTF-8、`ensure_ascii=False`、`sort_keys=True`、
`separators=(",",":")`、`allow_nan=False`，并尾随一个 LF。字段固定如下：

1. `official_lists.canonical_basename_sha256.{train,query,gallery}` =
   `96aa7aa80a3bb09cb48e16089f04d13ef51575442ba1aab162721add27c07189` /
   `e7bff615f1722a10be3d108341d0e9ceb2934ebbfcfb7506957100201cdd887b` /
   `81d57ac6b5015497133d9771b18ded6fdbb60c430341ffd589da291f3a799271`。
2. `relations.query_gallery_shared_basenames`：count=`1870`、canonical byte length=`43012`、
   SHA=`e940491d5471d3b976095335d1472e734fe8e6a76c192a3e98d5d0e9dbb7567f`。
3. `relations.query_gallery_shared_rgb_sha256_legacy` 唯一使用
   `json.dumps(sorted_list,ensure_ascii=False,sort_keys=True,separators=(",",":"),
   allow_nan=False).encode("utf-8")`，无尾随 LF；count=`1870`、byte length=`125291`、
   SHA=`e02e1be9b04d1428691809d81627235a3c4bb489e794d9b125c1f6c9c55b2e0c`。
4. `relations.query_gallery_shared_rgb_sha256` 对同一 list 使用 `canonical_json_bytes`；
   count=`1870`、byte length=`125292`、SHA=
   `54a624b1490cecfa77677ae275229d59b68714d5216bd7ab5bf749d66b9a552d`。
5. `relations.query_gallery_joint_metadata_pairs` 使用下一小节的 joint metadata 投影，
   count=`1870`、canonical byte length=`566372`、A02-derived SHA=
   `e59e8e935c9aa1cb19888ad23ab4f23a052cdda0d35fbb84928ae0d1ea1c3f51`。

joint pair projections 不能互相替代：按 pair sort 后的 `[pair.basename]` 必须逐项等于
official shared-basename list，`[pair.rgb_sha256]` 再独立排序后必须逐项等于 shared-RGB
list；RGB 与 pose duplicate endpoint pair list 则按 `(query_index,gallery_index)` 逐项
相等。这样既拒绝子/超集，也拒绝只保持两个边缘投影的成对置换。

A02 只读诊断脚本固定为 `42218 B` / SHA256
`88db86bb09a8d7d6fde7394ba2d12f8b115d517f9609bffe3c40ffd8836c7348`；stdout 原样证据位于
Git 外 `remote_artifacts/exp374_a02_readonly_mirror_report_88db86bb.json`，固定为
`9290 B` / SHA256
`7b070824f86304e9ce4a4fd24e69b0b1c2bda6bea1f24c049c5b844b79553fa2`。两次独立只读执行均
exit `0`，resolved data/staging 分别为 `/mnt1/afrdata/Occluded_Duke` 与
`/home/afr/exp374_gate_a_8ca57ed_a02_20260715/.exp374_prepare_qdm_o793`。A02-derived
新增冻结常量为：

1. RGB 与 pose endpoint pair list 各 count=`1870`、canonical bytes=`21666`、SHA256=
   `4135cdc4bb3cecd52dcf79423cf24d53595ce695a8b91544e2732be4bf3ebdfc`，且逐项相等；
2. 完整 joint pair payload count=`1870`、canonical bytes=`3542413`、SHA256=
   `b82fd6aa1a81faf85e80b876a62bd892d259e3c7e1e9bb9d9a381641dbb3df93`；
3. train/query/gallery 的 `target_outside_effective_count` 均为 `0`；
4. `junk_true/junk_false/forbidden_pair`=`1870/0/0`，train↔eval 全部 overlap=`0`，q/g
   forbidden overlap=`0`；full/effective constituent content mirror 均为 `3486`，path
   overlap 均为 `0`；
5. 六个 A02 cache 的 bytes/SHA256 固定为：query Hraw
   `461660288/ce908ee4e57a602f03e66340ab66c16097d0ff9f26a678f5d249a4ba10f7b45f`、
   score `150408/30a40c5d4c349d38b6527b8ad13b4b3f2b5e4dbfdbd0cf285d484a0be1116ce4`、
   nuisance `1679728/bdf7c98729b369904187d8711a39f9441013c4a0b423e22cb4e468aea6b90cfb`；
   gallery Hraw
   `3689312384/645c352137680dcde416a33b0abe37fc32109000243a1c97b6310c9172c90d3d`、
   score `1201076/8385ea376e03b460d3b5e7c3084712b2fac70b81e74b1a71b19e9d3c6096b09d`、
   nuisance `13422488/b3127a3bfb388a8ce5542386bde6f715be78096e9e5da44b6408054f635d2c65`。

#### frozen report schema 与数组 SHA

premetric manifest 的 `dataset.split_relations` schema 固定为
`occluded_duke_official_mirror_v2`，只含以下顶层 key：`schema`、`official_source`、
`official_lists`、`split_counts`、`within_split`、`cross_split`、`relations`、`pairs`、
`relation_report_sha256`。所有 count 为 JSON integer，SHA/dtype/schema 为 JSON string，
shape 为 JSON integer list，禁止 set/tuple/repr。`official_source` 精确含
`repository/commit/filename_regex`；`official_lists.{split}` 精确含
`rgb_root/list/count/raw_bytes/raw_sha256/canonical_bytes/canonical_sha256/
pose_index_bytes/pose_index_sha256`；`split_counts` 精确含 `train/query/gallery`。

- `within_split.{split}` 精确含 `path_duplicate_count/rgb_sha256_duplicate_count/
  pose_path_sha256_duplicate_count/pose_content_sha256_duplicate_count/
  full_pose_person_path_duplicate_count/full_pose_person_content_duplicate_count/
  effective_pose_person_path_duplicate_count/effective_pose_person_content_duplicate_count/
  source_pid_count/target_outside_effective_count`；
- `cross_split.{train_query,train_gallery,query_gallery}` 统一使用精确 key：
  `path_overlap_count/rgb_sha256_overlap_count/pose_path_sha256_overlap_count/
  pose_content_sha256_overlap_count/full_pose_person_path_overlap_count/
  full_pose_person_content_overlap_count/effective_pose_person_path_overlap_count/
  effective_pose_person_content_overlap_count/source_pid_overlap_count/
  rgb_content_forbidden_group_count/pose_content_forbidden_group_count/
  full_pose_person_content_forbidden_count/effective_pose_person_content_forbidden_count/
  forbidden_overlap_count`。train↔eval 的四个 content-forbidden count 分别等于对应
  content overlap，q/g 则只把 official mirror 关系之外的 group/orphan 计为 forbidden；
- `relations` 精确含 `query_gallery_shared_basenames/
  query_gallery_shared_rgb_sha256_legacy/query_gallery_shared_rgb_sha256/
  query_gallery_endpoint_pairs/query_gallery_joint_metadata_pairs/
  query_gallery_joint_pairs/split_record_sets/allowed_pair_count/junk_true_count/
  junk_false_count/forbidden_pair_count`。所有摘要对象精确含
  `count/canonical_bytes/sha256`；endpoint 对象另含 `equal` 与 `rgb/pose` 两个摘要；
  `split_record_sets` 精确含 `train/query/gallery` 三个摘要；
- `pairs` 按 `(rgb_sha256,basename,query_index,gallery_index)` 排序。`query_index` 与
  `gallery_index` 唯一定义为冻结 metadata 的 `index` 字段，不是当前迭代位置。每项的
  shared key 精确为：`basename/camid/effective_pose_person_sha256/frame/
  full_pose_person_sha256/hraw_sha256/nuisance_sha256/person_count/pid/
  pose_content_sha256/report/rgb_sha256/score_sha256/source_camid/source_frame_id/
  source_pid/viewid`；endpoint-specific key 精确为：
  `query_index/gallery_index/query_rgb_relpath/gallery_rgb_relpath/
  query_pose_path_sha256/gallery_pose_path_sha256/query_target_person_idx/
  gallery_target_person_idx/query_full_pose_person_relpaths/
  gallery_full_pose_person_relpaths/query_effective_pose_person_relpaths/
  gallery_effective_pose_person_relpaths`。RGB relpath 必须含 `query/` 或
  `bounding_box_test/` split prefix；pose relpath 必须含 `pose_data/query/` 或
  `pose_data/gallery/` prefix。report 先验证两端 canonical JSON 相同，再只保存 shared
  一份；Hraw/score/nuisance 两端各自哈希相等后也只保存 shared 一枚。

`array_sha256_v1(array, expected_dtype, expected_shape)` 唯一执行：验证 array finite 且
shape 精确；转换成 little-endian `expected_dtype` 的 C-contiguous array；header 精确为
`{"schema":"array_sha256_v1","dtype":np.dtype(expected_dtype).str,
"shape":[int...],"order":"C"}`；返回
`SHA256(canonical_json_bytes(header)+array.tobytes(order="C"))`。Hraw/score/nuisance 分别
固定 `expected_dtype=<f4/<f4/<f8`、shape=`[17,96,32]/[17]/[95]`。q/g 两端必须各自重算
后逐值相等。`relation_report_sha256` 对移除自身后的整个 report 使用
`canonical_json_bytes` 计算；execution SHA 覆盖含该字段的 report。prepare 同时原子写
`prepared/split_relations.json`，run/summarize 在 RGB/pose TOCTOU 验证后完整复算并逐字节
相等。

manifest 嵌套固定如下，不允许把摘要移到 execution SHA 之外：

1. `premetric_manifest.dataset.split_relations` 保存含
   `relation_report_sha256` 的完整 report object；
2. `premetric_manifest.dataset.split_relations_artifact` 精确含
   `relpath="split_relations.json"/bytes/sha256`，其中 bytes/SHA 对
   `canonical_json_bytes(dataset.split_relations)` 计算；
3. `premetric_manifest.prepared_artifact_sha256["split_relations.json"]` 必须等于上项
   SHA；`prepared/split_relations.json` 必须由 `atomic_write_json` 原子写入同一 canonical
   bytes；
4. prepare-resume、run 入口、`RUN_COMPLETE` 发布前、summarize 入口及 results 发布前，
   都必须从真实 RGB/official lists/pose index/NPZ/metadata/cache 重算完整 report，验证
   object、artifact bytes/SHA 与 prepared artifact manifest 三者同时相等；不得只信
   frozen JSON 或只重查 RGB。`verify_prepared_artifacts` 只负责 manifest/prepared file
   hashes；新增 `verify_relation_runtime(manifest, split_datasets, prepared, phase)` 负责
   active dataset 全资产重算并返回 entry-bound identity set。summarize 在
   `publish_or_verify_results` 前做 full tail audit，再重验 RUN_COMPLETE/全部 arm marker；
   results 发布后、COMPLETE 前再次做 quick identity、prepared triple 与 result hash 复核；
   任一漂移全局失败且不得发布 COMPLETE；
5. execution `premetric_manifest.schema` 仍为 `exp374-gate-a-v1`，dataset 原有
   `name/num_train_pids/num_train_cams/num_train_vids/num_query/num_gallery/cache` 不删除，
   只新增上述两个 key；matching、centroid、schedule 与结果均不得进入 split-relation
   构造输入。

#### matching、回归与 attempt 边界

query/gallery donor matching 仍各自在 split 内独立生成；`eligible_pair` 的 different
PID/path/RGB content/pose bundle path/content/constituent path/content、exact person count、
bijection 与 no-fixed-point 均不得放宽。official mirror 白名单只描述 cross-split dataset
relation，不授权跨 split donor，也不改变 evaluator。

production seam 固定为：

1. 以 `_pose_asset_manifest_v2` 替换当前只返回两个聚合 SHA 的
   `_pose_asset_identity`；它从 active `PoseImageDataset.index/max_persons` 同时返回 full 与
   conditional-target-first effective 的 relpath/path/content tuple、两个旧聚合 SHA、
   `target_person_idx` 和 source filename 标签；cache metadata 每行必须写
   `schema="exp374-scene-metadata-v2"` 及全部非空 full/effective tuple。
   v2 metadata row required 字段精确覆盖现有
   `schema/index/split/path/rgb_sha256/pose_path_sha256/pose_content_sha256/pid/camid/
   viewid/person_count/frame/report`，再追加
   `source_pid/source_camid/source_frame_id/target_person_idx/
   full_pose_person_relpaths/full_pose_person_paths/full_pose_person_sha256/
   effective_pose_person_relpaths/effective_pose_person_paths/
   effective_pose_person_sha256`。row 不重复保存 95-D `continuous`；它的唯一权威源仍是
   `{split}_continuous.npy`。`SceneRecord` required 字段为上述 row 去掉 `schema` 后加
   `metadata_schema` 与 `continuous`；`load_scene_records` 必须同时验证 metadata count/order
   和 continuous cache 的 exact `<f8`、`[N,95]`、finite，再逐 index 合成 record，错误码
   `E_METADATA_SCHEMA_V2/E_CONTINUOUS_CACHE_V2`；
   `load_scene_records` 只接受 v2，缺字段、空 tuple、长度错位或 v1 一律
   `E_METADATA_SCHEMA_V2`。`SceneRecord` 新字段全部 required、无空默认；现有 synthetic
   fixture 必须机械补齐 v2 字段，禁止以 legacy default 让 hard gate vacuous。A02 old
   metadata 只由已冻结的 Git 外只读 evidence adapter 读取，production/A03 resume 不得
   转换、补字段或接受；
2. 纯 CPU `eligible_pair` 逐项拒绝 constituent path/content overlap。report 的
   `relations.split_record_sets.{train,query,gallery}` 另存用于 matching 的完整 canonical
   `SceneRecord` projection 摘要，精确含 `count/canonical_bytes/sha256`；projection 覆盖
   上述每一个 required dataclass 字段，包括 base-cost 使用的 `camid/frame/continuous`、
   eligible predicate 使用的 path/content/PID/person-count/constituents、`viewid`、source/
   target 标签和 canonical report；不得用字段子集冒充 complete record digest。
   public `prepare_split_mappings` 必须新增无默认的 keyword-only `relation_report` 与
   `split`；入口以 full-split records 重算 record-set digest、relation self-hash，并要求
   对应 within-split full/effective path/content duplicate 全为 0，否则
   `E_MATCH_RELATION_TOKEN`。验证成功后只能由 module-private factory 构造 immutable
   `_ValidatedRelationToken`，精确绑定 split、full record-set/report SHA、按 global index
   排列的 per-record canonical SHA，以及由已验证 full records 唯一派生的
   `strata_global_indices: person_count -> exact ordered tuple[global_index]`。tuple 顺序唯一为
   full-split record 顺序，必须覆盖该 person-count 的全部且仅这些 global rows；不得由调用者
   提供、裁剪或重排。底层 GPU `exact_sparse_candidates` 不再重验 full digest，而是强制接收
   无默认的 token、`global_indices` 与 stratum `local_records`；先要求 local records 的唯一
   person-count 存在于 token，且传入 `global_indices` 与 token 对应的完整 ordered tuple
   逐项完全相等，再逐项重算 local record SHA 并与 token 对应 global slot 相等。任何遗漏、
   超集、重排、把同一 stratum 拆成多次调用、indices 重复/越界或跨 person-count 都必须以
   `E_MATCH_RELATION_TOKEN` 失败。这样当前按 person-count 分层实现可落地，又无法独立、
   错 subset、拆分 subset 或未审计调用；GPU 才能在已证明 global uniqueness 后用 distinct
   index 省略 variable-length 比较。
   新增 synthetic equivalence test 必须证明 GPU candidate edge 与纯 CPU predicate 完全
   一致，并覆盖缺 token、伪造/错 split/token drift、partial/full/effective overlap；
3. `audit_split_relations_v2` 是 dataset-agnostic 结构层，只接受 metadata/cache/constituent
   显式输入并允许 q/g 零 overlap；`assert_occluded_duke_official_v1` 只在结构层 PASS 后
   叠加 official source/count/digest/1870 exact gate；两者都不得读取 checkpoint、历史或
   当前 ReID 指标、matching output、arm/per-query 结果；
4. prepare 调用顺序唯一为：cache 三 split → 构造 full report → exact official gate →
   原子写 `split_relations.json` → 生成 split-local donor candidate/mappings → 调用
   `checkpoint_specs` 解析 checkpoint/log/parity。当前 prepare 顶部的 `checkpoint_specs`
   必须后移，任何 relation 失败都必须发生在 mapping、checkpoint 与指标之前；
5. 当前 `verify_frozen_runtime` 必须拆成
   `verify_frozen_config_environment(manifest,device)` 与
   `verify_frozen_checkpoint_specs(manifest,device)`。run/summarize 唯一顺序为：
   `verify_prepared_artifacts` → config/environment-only → `direct_datasets` →
   `verify_relation_runtime` full entry → checkpoint/log verification → schedule/arm/result。
   relation runtime 不得因拿不到 active index 而退化；prepared metadata/cache 加 active
   RGB/list/index/NPZ 必须复用同一 report builder，不得维护第二套“较宽松”的 resume/run
   validator。A02 的旧
   `assert_disjoint_records` 删除后只能由这两个更强 gate 替代，禁止简单跳过 q/g 检查。

I/O 边界固定如下：full audit 包含全部 9 个三 split scene cache（q/g 六份另做 mirror
endpoint audit）、三 split RGB/NPZ/list/index/metadata 的 SHA/identity；A02 的 q/g 六 cache+
86477 identities 实测约 30 秒，production 单次上限 90 秒。单个 lifecycle 正常最多
prepare、run entry/tail、summarize entry/tail 共 5 次，连一次 resume 预算上限 6 次/
10 分钟；超限必须记录性能异常并停止授权链，不能删门禁。每个 arm publication 前的
quick set 只含 19 个高扇出
文件 identity：3 official lists、3 pose indices、3 metadata、9 scene cache 和
`split_relations.json`，不得扫描 86477 个资产；每个 seed 开始/结束各复核 full identity
registry，run 尾再整文件重哈希。这样 492 arms 只增加 `492×19` 次 stat；任何 seed/tail
漂移都令整个 execution `FAILED_NONREPORTABLE`，即使已有原子 arm 也不得汇总。

新增回归至少覆盖：结构层 q/g 无 overlap PASS；合法 1Q+1G mirror PASS；四类 split 内
duplicate；四类 train/eval alias 与 source-PID overlap；q/g RGB/pose/constituent path
alias；Q/Q、G/G、三成员、pair-count drift；basename PID/cam、view/person-count/frame/report
drift；list count/raw/canonical digest、traversal/prefix/symlink-root drift；pose constituent
部分/跨 mirror/位置置换 alias；RGB joint pair 置换；协调 pose/cache 漂移；Hraw/score/
nuisance dtype/shape/byte-order drift；mirror 零距离仍被 junk mask 删除；canonical order/hash
不依赖输入遍历顺序；manifest/prepared/resume/runtime drift；以及 mirror 白名单不能放宽
split-local `eligible_pair`。另必须显式覆盖：v1/缺字段/空 tuple metadata 拒绝；
`target_person_idx` 在 full 合法但落于 `persons[:6]` 外时只计数且不重排，以及 full index
非法；matching 缺/伪造/错 split relation report，以及 token 后 `camid/frame/continuous` 或
任一 constituent 字段漂移、global_indices 重复/越界、完整 stratum 的 omission/superset/
reorder/split-call、错 subset/跨 person-count stratum；
metadata 与 continuous cache count/order/dtype/shape/nonfinite drift；
relation-before-checkpoint/log/metric/mapping
的 spy；prepare-resume、run entry、run tail、summarize entry、summarize pre-results、
pre-COMPLETE 六调用点逐一 drift；quick-set 与 full-registry drift；A02 path/resume 永久拒绝；
已有 A02 old metadata 不能经 production adapter 升级。错误码必须按 predicate 唯一覆盖
上述分支。

A02 永久保留的三份 metadata、六个 q/g cache arrays 与 pose index/NPZ constituent 只读
preflight 必须复现全部 fixed digest、1870 allowed 与 0 forbidden。A02 禁止 resume 或
同名重试；design、实现、真实资产 preflight、本地/历史 Python 3.8 全量复验和多路证据
审查全部 PASS 后，才允许为新 exact commit/bundle/clone 设计全新 A03 prepare-only；
仍不授权 Gate A `run`、`summarize` 或训练。

### A1 matched-shuffled（primary）

query 与 gallery 分别生成 20 份固定、无 fixed point、严格一一的 donor map：

- donor PID 必须不同；
- 禁止 query/gallery 跨 split；
- different path/content 同时约束 RGB path/content SHA 与 pose path/content SHA；
- 完整 final scene heatmap 一起替换，不逐关节拼接；
- person count 必须完全相同；
- camera、framing 与连续 pose summary 只进入 soft cost，不作硬 strata；
- nuisance 只允许使用 pose score、heatmap L1/peak/entropy/support、skeleton bbox
  中心/尺度/aspect、border-touch/crop 程度；
- 禁止使用 ReID embedding、feature norm、AP、排序或任何评测结果挑 donor；
- 所有 mapping 和 RNG seed 必须在读取指标前落盘并哈希。

全量 gallery 不允许构造稠密 `N×N` Hungarian cost。实现应使用分层的稀疏
候选图和 minimum-weight full bipartite matching；若任一分层不存在完美匹配，
协议标记 `INVALID`，不得静默退化成随机乱序。

#### 唯一 nuisance 与 cost 定义

对每个 final scene bundle 唯一定义连续向量 `u_i`：

1. 17 维 `scene_scores`；
2. 每个 joint channel 的 `log1p(L1)`、peak、normalized spatial entropy 和 support
   fraction，共 `17 × 4` 维；support 定义为大于该 channel `0.10 × peak` 的像素比例；
3. scene support-union bbox 的 normalized `cx, cy, log(w/W_hm), log(h/H_hm),
   log(w/h)`、四边 border-touch indicator 和 crop degree，共 10 维；这里 `H_hm,W_hm`
   是 heatmap 高宽，crop degree 是四个 border-touch indicator 的均值。

`Hraw` 只要求 finite，可以为 signed；本小节所有统计中的 `H` 唯一指
`Hpos=clamp_min(Hraw,0)`，所以必须非负。零 Hpos channel 的 L1、peak、entropy、support
均定义为 0；
normalized entropy 唯一定义为

\[
-\sum_p \pi_p\log(\pi_p)/\log(H_{hm}W_{hm}),\qquad
\pi_p=H_p/\sum_pH_p.
\]

任一非有限值使对应 split 整体 `INVALID`。每一连续维在 query/gallery 各自 split 内用
`z=(u-median)/(1.4826×MAD)` 标准化，再 winsorize 到 `[-5,5]`；`MAD<1e-8` 的维置 0
并在 manifest 中记录为 constant。定义

\[
d_{cont}(i,j)=\frac{1}{d}\sum_{m=1}^{d}\min(|z_{im}-z_{jm}|,5),
\]

\[
C_{base}(i,j)=d_{cont}(i,j)+0.25\,\mathbb{1}[camera_i\ne camera_j]
+0.25\,\mathbb{1}[frame_i\ne frame_j].
\]

`frame` 是四个 border-touch bit 按 `top,bottom,left,right` 顺序组成的 4-bit 类别；不再
存在其它人工 framing bin 或可调权重。person count 仍是硬约束，不进入 cost。

#### 稀疏 matching 与固定随机化

匹配实现与门槛固定如下：

1. continuous nuisance 在每个 split 内用 median/MAD robust z-score；
2. 硬约束只有 split-local、exact person count、different PID、different path/content、
   bijection 与 no fixed point；
3. 每个 `split × person_count` 二部图的候选数序列唯一为
   `K(n)=sorted(unique({min(n-1,k): k in [8,16,32,64,128,256]}))`；每个 anchor 实际
   保留按 `C_base` 排序的 `min(k, eligible_count)` 条边。每次扩图先用 Hopcroft-Karp
   检查 `matched=N, unmatched=0`，取第一个存在完美匹配的 k；k=256 仍失败即
   `INVALID`，禁止构造或退化成稠密 `N×N` 图；
4. 第 `r=1..20` 份 mapping 使用固定 seed `374000+r`。在选定稀疏图上定义
   `C_r=C_base+eta_g×Gumbel(seed,edge)`；`eta_g` 在每个已选定的
   `split × person_count` sparse edge set 上单独固定为 `0.25×IQR(C_base)`，若 IQR
   小于 `1e-8` 则固定为 `0.01`。这称 seeded edge perturbation，不称 tie-only
   perturbation。20 个 seed、候选边、`C_base/C_r` 与 mapping 必须先落盘并哈希；
5. 任意两份 donor map 的 Hamming distance 必须在 query maps 和 gallery maps **分别**
   `>=0.90`；否则报告 effective unique count 并将 Gate A 标记 `INVALID`；
6. 每份 mapping 的全体 donor 必须是同 split 的完整排列，因此每个 nuisance 的
   marginal SMD/KS 理论上应为 0；这里仅作 permutation sanity，不当作 pair-quality
   证据，浮点容差固定为 `1e-10`；
7. pair quality 唯一用未加 Gumbel 的 `C_base` 审计：每个连续维的 paired
   `median(|z_i-z_j|)` 必须 `<=0.50`，每份 mapping 的 `P95(C_base)<=1.25`；mapping
   mean cost 还必须不高于同一稀疏候选图和硬约束下 1,000 个随机化 full matching
   的 median mean cost 的 `0.75` 倍。baseline seed 唯一定义为 `475000+b`，
   `b=0..999`。baseline 不优化 cost：在同一稀疏候选图上，以 PCG64DXSM(seed)
   分别打乱 anchor 顺序和各 anchor adjacency，再用固定 Hopcroft-Karp augmenting
   order 求 full matching；任一 baseline 非 full matching 即实现错误并 fail closed。
   任一 pair-quality 门槛失败即 `INVALID`；
8. final scene-merged heatmap 另保存四个 report-only summaries：
   `total_L1=sum_j L1_j`、`mean_confidence=mean_j scene_score_j`、
   `visible_joint_count=sum_j 1[L1_j>1e-8 and peak_j>1e-6]`、
   `scene_entropy=mean_j channel_entropy_j`。它们只用于透明报告，不扩展 95 维 `u_i`，
   不进入 `d_cont/C_base` 或额外 `INVALID` threshold；禁止实现者二次选择。

若任一 scene 的 support union 为空，95 维 bbox 项没有定义，对应 matching split 必须
直接 `INVALID`，禁止把 bbox 项零填或换用整图 bbox。

所有 arms 必须使用同一 query 集合；禁止删除难匹配样本或在结果后放宽门槛。

20 份 query/gallery mappings 在三枚 checkpoint 间完全复用，mapping index `r` 唯一
绑定 `(query_map_r, gallery_map_r)`，不得交叉重配。候选 edge 先按
`(C_base, donor_path)` 排序；Gumbel 使用 NumPy `Generator(PCG64DXSM(seed))`，按
`(anchor_path, donor_path)` 词典序消费随机数；assignment cost 最后加
`1e-12 × edge_lexicographic_rank/(E+1)` 作确定性 tie-break。solver 名称、版本和输出
mapping SHA 必须进 manifest。

#### 实际 PSG 输入上的干预强度门禁

禁止仅审计原始 heatmap。只读 forward hook 必须捕获两个 PSG block 在 encoder 前实际
消费的 float32、同-device tensor；其语义必须与
`S(H)=sigmoid(F.interpolate(H, block_hw, mode='bilinear', align_corners=False))` bitwise
一致。在读取 ReID 指标前计算：

\[
D_{rel}(i,j)=\frac{\|S(H_i)-S(H_j)\|_1}
{0.5(\|S(H_i)-0.5\|_1+\|S(H_j)-0.5\|_1)+10^{-12}}.
\]

另对 `Rplus(H)=clamp_min(S(H)-0.5,0)` 的 17 个 channel 计算 centroid displacement，
并除以 block 网格对角线。每个 block/channel 唯一以 `mass=sum Rplus >1e-8` 且
`peak(Rplus)>1e-6` 判为有效，centroid 权重为 `Rplus/mass`；两边有效则用 centroid
欧氏距离，仅一边有效则记为 1，
两边均无响应则记为 0，最后对 17 channels 等权平均。每份 query map 与 gallery map
分别要求：全部 sample
tensor content SHA 不同、`median(D_rel)>=0.10`、`P10(D_rel)>=0.01`、median normalized
centroid displacement `>=0.03`。两个 block 若 shape 相同，审计值必须逐值一致；若不同
则分别过门槛。失败称 `INVALID_WEAK_INTERVENTION`，禁止据此作 NO-GO。

### A2 train-derived scene-channel centroid control（secondary）

该 control 直接作用于 PSG 真正接收的 final scene heatmap，而不在 person-level
处理后重新 max-merge，从而避免把 nonlinear owner/overlap 变化混进 treatment。
但它只平移每个 max-merged channel 的整体质心；多人 scene 的峰间相对结构仍被保留。
因此它只称 centroid control，不称 canonical pose，且不进入 primary contrast、
`theta_min` 或 GO/NO-GO。

唯一算法固定如下：

1. 用训练集 deterministic、无增强的原 `_prepare_pose` 得到 `Hraw`，仅以其派生的
   `Hpos` 拟合几何；
   训练集与 query/gallery 的 RGB path、RGB content SHA、pose path 和 pose content SHA
   必须完全无交集，否则该 secondary arm `INVALID`；
2. joint channel 有效条件始终在 `Hpos` 上统一为 L1 `>1e-8` 且 peak `>1e-6`；scene
   bbox 定义为所有
   有效 channels 中大于各自 `0.10 × peak` 的 support union；有效 channel 少于 2 个
   时：0 个有效 channel 必须逐值输出原 `Hraw` scene（其 `Hpos` 虽全零，但 `Hraw`
   可能含有限负值，禁止造出 all-zero tensor）；恰有 1 个有效 channel 则以该
   channel support 作为 scene bbox，并按该 joint 的训练集 median 平移；任何非零但
   不满足有效谓词的 weak channel、空 union、或缺失训练 median 都使整个 arm `INVALID`；
3. 对每个有效 Hpos channel 计算 heatmap centroid，在 scene bbox 中归一化；训练集
   centroid target 是该 joint 有效样本 normalized centroid 的逐坐标 median；
4. 测试时以测试 Hpos scene bbox 恢复目标 centroid；用 Hpos 得到唯一整数位移后，只对
   对应 `Hraw` 原 channel 做同一 zero-padded integer translation，输出仍为 signed raw。
   整数取整固定为 half-away-from-zero。禁止 wrap-around、插值
   变形或重新生成 Gaussian；target 与实际输出 centroid 的误差在平移和裁剪后计算，
   必须 `<=0.75` heatmap pixel；
5. 必须逐值验证 `clamp_min(translate(Hraw),0)==translate(Hpos)`；positive L1、peak、
   entropy 与 shape 应保持。负绝对质量按上文零质量规则审计；任何
   边界裁剪都由下述门禁 fail closed，不能逐样本修补或删除。

对 correct Hpos 中不满足统一有效谓词的 channel，只有 Hpos all-zero channel 可保持
对应 Hraw 原样；
其它情况已按上文 fail closed。所有有效 channel 要求：

- 100% 数值 finite；
- 100% 的 sample-channel positive L1 ratio 位于 `[0.95,1.05]`；
- 100% 的 sample-channel positive peak ratio 位于 `[0.95,1.05]`；
- Hpos normalized spatial entropy 绝对差 100% 不超过 `0.01`；
- 100% 的有负质量 sample-channel 满足 negative absolute mass ratio `[0.95,1.05]`；
  输入负质量为 0 的 channel 输出负质量必须也为 0；
- 不允许删除违规样本；任一比例门槛失败，整个 centroid arm `INVALID`。

几何/能量审计发生在 Hpos，实际 arm tensor 是 translated Hraw；runner 还必须 hash
translated Hraw 经过真实 resize+sigmoid 的 actual PSG input。若边界导致任何非零
channel 超门槛，整个
secondary centroid arm `INVALID`，但不使 primary Gate A 失效；只能报告该 secondary
control 不可用，禁止据此修改 primary 结论。

### A3 true bypass

保持同一个 `PoseBackboneModel` 和同一 checkpoint，向 forward 传
`pose_dict=None`。此时 `scene_heatmaps=None`，两个 PSG block 的 guard 均跳过。

以下方式明确禁止解释为 bypass：

- `MODEL.POSE_ENABLED=False`：会切换成另一个模型类；
- `heatmap=0`：PSG 内部 `sigmoid(0)=0.5`；
- 现有 `POSE_DROPOUT_P`：同样只是 zero-response 输入。

### Audit-only final-scene override 入口

实现必须给 `PoseBackboneModel.forward` 增加默认关闭、显式三态的 audit-only keyword：

- `UNSET`：保持现有 `pose_dict -> _prepare_pose` 完整路径；
- tensor：跳过 `_prepare_pose`，把该 tensor 作为唯一 `scene_heatmaps` 输入；
- explicit `None`：true bypass。

A0 correct 也必须先由原 `_prepare_pose` 离线生成 final scene tensor，再通过 tensor
override 进入模型；A1/A2/A4–A10 只改同一类 tensor；A3 使用 explicit `None`。禁止把
final scene 伪装成单人 `pose_dict`。运行时必须断言：eval mode、恰有两个 PSG block、
`POSE_PSG_STAGES=[-1]`、descriptor 为 768 维、PAA/LGPA/GCN/PPA/VCSR/PBSR/part branch/
pose prompt/pose patch embedding 等其它 pose 机制全部关闭。A0 必须在全部 arms 前后各
运行一次，两次 descriptor SHA 和 mAP/R1 必须完全一致。每枚 checkpoint 必须 strict
state-dict load，任何 missing/unexpected key 均 `INVALID`；同一 batch 的 normal UNSET
forward 与离线 correct tensor override 的 descriptor 必须 bitwise identical，先过该
seam parity 才允许生成 control。

### A4–A10 解剖组局部破坏敏感性（secondary）

预先固定七个左右对称解剖组：

1. head：nose/eyes/ears；
2. shoulder；
3. elbow；
4. wrist；
5. hip；
6. knee；
7. ankle。

每次只把一个组的 heatmap channels 替换为 matched donor 的对应 channels，其余
channels 保持 correct。每组使用与 primary shuffle 完全相同的 20 个 mapping index，
先对 mapping 等权平均，再报告全部七组。

该操作会破坏 recipient 相邻关节之间的骨长和拓扑一致性，因此只能称
`matched-donor local-channel corruption sensitivity`，不称 joint drop、关节效用或
因果移除。它不进入 GO/NO-GO 硬条件；即使显著下降，也只能说明 checkpoint 对该类
局部 OOD 破坏敏感。禁止用零通道表示“移除”，也禁止看完结果后只挑最好的一组。

### Secondary controls

- 项目现有的无协变量 wrong-PID bijection 只作 secondary；
- zero-response 只作 sigmoid 语义诊断；
- query-only 干预作敏感性分析，主结果仍对 query/gallery 都施加固定干预；
- exp007 只有 Stage 3 的两个 block，block leave-one-out 只能作 secondary，不能
  宣称跨 stage 规律。

## 指标与统计

### 原始输出

每个 seed、每个臂必须保存：

- 完整 mAP、R1、R5、R10；
- 每 query AP；
- 每 query R1 indicator；
- 每 query retrieval margin（最近负样本距离减最近正样本距离）；
- descriptor、distance matrix、donor map、centroid 参数与运行 manifest 的 SHA256；
- split relation canonical audit：train/eval 完全无交叉，以及 1870 组严格
  official query/gallery mirror、0 forbidden duplicate 的断言结果。

retrieval margin 必须在 official junk removal 后定义为
`min(valid negative distance) - min(valid positive distance)`；不存在 valid positive 或
negative 的 query 使该 arm `INVALID`。descriptor 与 distance matrix 逐臂生成、校验、
计算 per-query 结果并记录 SHA/shape/dtype 后删除，不作长期持久化；长期保存的是输入、
mapping、per-query 输出、汇总和上述内容哈希。

### Fail-closed manifest、恢复与资源

`execution_sha` 唯一定义为不含输出路径和任何结果的 frozen pre-metric manifest 的
SHA256。新 execution 必须以 `mkdir(exist_ok=False)` create-exclusive 地创建
`/home/afr/exp374_artifacts/gate_a_<execution_sha>`；普通启动遇到已存在目录即拒绝，
禁止覆盖。只有显式 `--resume <exact_execution_dir>` 可打开已有目录，并且必须满足：
存在 frozen manifest、没有 execution `COMPLETE` marker、所有输入 SHA 完全一致；已
atomic-published 的 arm 只读复用，残留临时 arm 目录必须删除后从该 arm 重做，禁止
覆盖已发布 arm。manifest 在读取任何 ReID 指标前冻结并至少包含：

1. audit code commit、dirty diff SHA 或 clean archive SHA；
2. 三枚 checkpoint、config、flat log、train log 的路径与 SHA；
3. RGB 顺序与内容 SHA、pose index/NPZ SHA、query/gallery 顺序、`num_query`；
4. nuisance scaler、constant dims、cost 公式版本、k、Gumbel scale、20 mapping seeds、
   1,000 baseline seeds、candidate edges、mappings 和 centroid 参数 SHA；
5. Python/PyTorch/CUDA/cuDNN/GPU、determinism flags 与 package lock；
6. checkpoint 中 PSG canonical keys 与兼容 alias keys 的 shape、内容 SHA 和逐值一致性。
7. 每个 split 的 signed raw audit、`positive_part_v1`、两个 active PSG block 的
   Sraw/Spos/Delta streaming SHA、shape/dtype/count 与差异统计；它们全部进入 canonical
   premetric payload 和 resume drift 复算。

每个 arm 只可写临时目录，全部文件 fsync、SHA 与断言 PASS 后 atomic rename；失败保留
失败 manifest 但不得发布半成品。恢复前必须逐项重算上述 SHA，任何路径、顺序、shape、
NaN/Inf、版本或 hash 不一致都拒绝恢复并把 execution 标成 `INVALID`。每个 arm 后必须
释放 descriptor/distance/GPU cache，且任一时刻只运行一个 arm、一个 seed、一个评测
进程。

按 `2 correct（全臂前后各一次）+ 20 shuffle + 1 centroid + 1 bypass + 7×20 group`
估算为每 seed 164 passes、三 seed 492 passes，历史 no-flip 速度约需 4.25–4.5 小时。
Secondary controls 中的 unmatched wrong-PID、query-only、zero、flip 和 block-LOO 不在
这 492 次核心执行内；若以后执行，必须另做设计与资源复审。4090 根卷当前
只读预审约有 217 GB 可用；由于大矩阵只保留 hash，启动前仍要求目标卷可用空间至少
80 GB，低于门槛直接拒绝。任何实现若改为持久化全部 descriptor/distmat，门槛必须
提高到 150 GB 并重新做资源审查。

### Primary estimands

所有 AP/R1 差值以 percentage point（pp）报告。对 seed `s`、query `q`、第 `r`
份 mapping，先唯一地定义：

\[
AP_{shuffle}(s,q)=\frac{1}{20}\sum_{r=1}^{20} AP(s,q,r).
\]

禁止先平均 descriptor 或 distance matrix 再计算 AP。20 份 mapping 是预注册的固定
Monte-Carlo nuisance ensemble，不作为 20 个独立数据集扩充样本量，也不在 primary
bootstrap 中重采样。另报告 leave-one-mapping-out sensitivity 与 Monte-Carlo SE。
R1 同样先按 query 定义
`R1_shuffle(s,q)=mean_r R1_indicator(s,q,r)`，再进入 seed contrast；禁止先对 20 次
整体 R1 求均值后伪装成 query-level 数据。per-query Monte-Carlo SE 定义为 20 个 mapping
值的 sample SD（`ddof=1`）除以 `sqrt(20)`，并汇总其 median/P95。leave-one-mapping-out
必须报告删去每个 mapping 后两个 primary `theta_c` 的 min/max，不据此删除 mapping。

对两个 primary controls

\[
c\in\{shuffle,bypass\}
\]

分别定义配对 contrast：

\[
\theta_{s,c}=100\times\operatorname{mean}_{q}
[AP_{correct}(s,q)-AP_c(s,q)],
\]

\[
\theta_c=\frac{1}{3}\sum_s\theta_{s,c},\qquad
\theta_{min}=\min_c\theta_c,
\]

以及每 seed 的

\[
\theta_{min,s}=\min_c\theta_{s,c}.
\]

`theta_min` 只作为保守点估计；不对 `correct-max(control)` 做普通 percentile CI。
两个 `theta_c` 必须始终单列报告。centroid control 只作为 secondary 单列，不进入
`theta_min`。

Gate B 另定义：

\[
\Delta_{train}=mAP(correct\text{-}pose\ train)-mAP(true\text{-}bypass\ train).
\]

### 固定重采样协议

三个 seed 只有三个，固定为 paired blocks，不对 seed 做 bootstrap，也不声称泛化到
随机 seed 总体。执行 10,000 次 one-sided 95% PID-cluster bootstrap。唯一 bootstrap
RNG 为 NumPy `Generator(PCG64DXSM(374900))`；PID 按数值升序形成抽样 universe，
quantile 固定使用 NumPy `method='higher'`：

1. 每个 replicate 只对 query PID clusters 有放回抽样；抽中 PID 时保留其全部 query；
2. 同一组 PID multiplicities 同步应用到三个 seed、全部 arms 和全部 20 mappings；
3. 每个 replicate 内先计算各 seed contrast，再对三个固定 seed 等权平均；
4. gallery、checkpoint 与 20 份 donor mapping 固定。

对两个 mAP contrasts 使用同一 bootstrap replicate 构造 one-sided simultaneous
max-deviation intervals：

\[
q_L=\max\left(0,Q_{0.95}\left(\max_c[\theta_c-\theta_c^{(b)}]\right)\right),\quad
LCB_c=\theta_c-q_L,
\]

\[
q_U=\max\left(0,Q_{0.95}\left(\max_c[\theta_c^{(b)}-\theta_c]\right)\right),\quad
UCB_c=\theta_c+q_U.
\]

R1 indicator 对两个 controls 单独定义 `theta^R1_c`，用同样方法构造另一个两对照
simultaneous interval family。不得先按 mAP 选择 control 再检查 R1。

七个解剖组只作 secondary sensitivity；使用固定 20 mappings 和相同 PID replicate，
可报告 one-sided seven-group simultaneous max-deviation intervals，但不触发 Gate A GO。

上述区间只表示：给定 official gallery、checkpoint 和固定 donor mappings 时，对 query
identity 采样的条件不确定性。gallery 干预会共同影响全部 query，当前 bootstrap 不支持
gallery 总体 ATE 或一般 SUTVA 因果解释。

## 预注册决策

### Gate A GO

必须全部满足：

1. 三个固定 seed 的 `theta_min,s` 均大于 0；
2. `theta_min >= +0.30 pp`；
3. 两个 mAP contrasts 的 one-sided 95% simultaneous `LCB_c` 全部大于 0；
4. 两个 R1 contrasts 的 one-sided 95% simultaneous LCB 全部大于 `-0.50 pp`；
5. donor、路径哈希、协变量匹配和能量审计全部 PASS。secondary centroid arm 即使
   `INVALID` 也必须原样报告，但不改变 primary GO/NO-GO。

Gate A GO 只授权 Gate B 的干净配对训练设计与审查，不授权新机制训练。

### Gate A NO-GO

满足任一项即停止 PSG 自有化：

- 至少两个 seed 的 `theta_min,s <= 0`；
- 任一 primary mAP contrast 的 simultaneous `UCB_c < +0.30 pp`，使 GO 的最小
  实用效应不可能成立；
- 任一 primary `theta_c <= -0.30 pp`，且同一 simultaneous family 的 `UCB_c < 0`。
  这唯一表示 control 显著优于 correct；不再另造未定义的“反向 LCB”。

### INVALID / INCONCLUSIVE

- donor map 非双射、跨 split、PID 碰撞、split 内 donor 路径/内容碰撞、official
  mirror 关系之外的任意数据 alias、匹配门槛或能量门槛失败：`INVALID`，只允许修协议
  后以全新 execution 重跑；
- 任一 matching 数值门槛失败均使 primary Gate A `INVALID`；secondary centroid arm
  数值门槛失败只使该 arm `INVALID`。禁止逐样本删除，所有可比较 arms 使用同一
  query 集合；
- 其余未达到 GO/NO-GO 的灰区：`INCONCLUSIVE`，不得直接开大训练或放宽门槛。

Gate A NO-GO 是本项目停止复杂化 PSG 的管理决策，不构成“所有 pose 方法均无因果
价值”的科学证明。

## Development / confirmation 数据锁

Occluded-Duke official test 已在本项目中被长期用于方法、阈值和 stop-rule 选择，
因此 Gate A/B 的全部结果从一开始就是 development evidence，不能作为最终
confirmatory claim。

在读取 Gate A 指标前固定 `Partial-REID` 为一次性独立确认数据集：只允许在 Gate A
GO、Gate B 完成且方法/超参数/epoch 全部冻结后评测一次。在此之前只允许核对数据
许可、文件完整性和 pose 资产可生成性，不读取任何方法指标。若 Partial-REID 资产或
许可不可获得，则本项目放弃 confirmatory claim，不在看到结果后改选其他数据集。

## 新机制边界

原候选 `Y=X+lambda(T-I)VX` 已被数学红队判 FAIL：它可精确归约为 residual
attention/GAT 或图拉普拉斯扩散；若双随机更新位于末 block 后并立刻 GAP，空间和
保持还会使其对 global descriptor 严格零作用。全分辨率 heatmap 离散 W2 与
batch-average trust region 也因计算量和单关节漂移漏洞判 FAIL。

第三路专项查新进一步确认：即便改成显式 source/demand，已有工作也已分别并在组合
上覆盖 source reliability、visible-to-occluded recovery、pose topology、
confidence-conditioned mixing、UOT 处理缺失部位、ReID 中双随机 OT feature
transform 和 Sinkhorn 双随机 attention。直接邻居包括 HOReID、FRT、PIRT、
HUPOR、RFC、UNITE、SOT 与 Sinkformers。因此，“把 GCN 换成 Sinkhorn”或“把
confidence 换成 posterior entropy”的差分不足，容易被解释为已有模块拼接，仍然
直接 NO-GO。

当前只保留一个待查候选**问题对象**，不保留方法 claim：

> 校准 pose posterior 是否能定义一个不确定度约束可行域，并与 ReID utility 共同决定
> source/demand 边际，在骨架支持上求解显式 source depletion + sink receipt。

即使 Gate GO，该候选仍必须先形成不能被解释为 confidence-weighted Sinkhorn
拼接的联合优化问题，再正面对照 HOReID、FRT、RFC、PGGANet、RTGAT、UNITE、
SOT、普通 GAT、row-stochastic attention、2026 TTPM 与 2026 Pose-Guided
Feature Restoration Transformer。TTPM 正文已核，属于 pose-patch matching +
confidence filtering + texture decoder 的强邻居；没有完成联合公式并排除后一篇
restoration 正文前，
不得启动该机制训练；
不得使用“首次 pose-aware OT”“首次姿态图传播”或“首次从可见区域补偿遮挡区域”
等宽泛表述。

## 风险与失败解释

1. 历史 checkpoint 缺 exact execution provenance，Gate A 只能筛燃料；
2. correct/shuffle/bypass 都是训练后反事实，其中后两者可能 OOD；centroid 与解剖组
   corruption 的 OOD 更强，所以只作 secondary；
3. official Occluded-Duke test 已被本项目长期用于选择机制，当前就是 development
   set；Gate A/B 只能做开发决策，最终 claim 必须遵守上面的 Partial-REID 一次性锁；
4. PSG 当前对 `[0,1]` ViTPose heatmap 重复 sigmoid，限制了可解释性；
5. Gate A 成功不证明任何候选新模块成功，失败则足以停止继续复杂化 PSG。
