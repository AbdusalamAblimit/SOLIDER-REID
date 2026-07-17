Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef71f-9f38-7212-b517-9899470a5158
--------
user
你是 ReID 论文创新挖掘员 12/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

== 团队资产(新创新点要能挂上其中之一)==
- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
- aerial-ground 跨视角 ReID(CARGO / AG-ReID.v2,航拍↔地面极端视角+低清)
- pose 热图门控(PSG / LGPA-D,姿态引导空间 gating)
- SMPL 3D 几何(mesh/joints/2D投影,团队已打通基建)

== 目标 ==
找能投 B 类(Pattern Recognition / TMM / TCSVT / AAAI 级)的**强创新点**。不要中等工程组合,要问题层面或机制层面有真新意的。

== 每篇提取(简洁)==
(1) 创新类型: 问题重定义 / 新机制 / 新数据-设定 / 工程组合(标注哪种)
(2) 为什么能发: 填了什么 gap + 证据链怎么搭的
(3) story 套路: 怎么把卖点讲成 headline 的

== 然后综合这批,产出 2-4 个针对团队资产的强创新点候选 ==
每个候选要: a) 一句话 headline; b) 挂哪个团队资产; c) 和这批里最像的工作的区别(切开点); d) cheap kill-switch(怎么花最小代价首验真假)。

这批论文(12/20):
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则核对 `GOAL.md`、`CLAUDE.md` 和实验状态，然后按你的约束只抽每篇 PDF 前 3 页。这个任务命中 PDF 处理技能，但实际读取会严格使用你指定的 `pdftotext -l 3`。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# CLAUDE.md · ReID 工作区（reid-paper）

## 0. 这个工作区是什么

这是一个全新的 ReID 研究工作区。目标是做一篇 CCF-B 级别、真正新颖、有效、又能讲清楚的行人重识别工作。具体做哪个子方向（遮挡、标准、跨域、长期等等），自己调研之后再定。

目录说明：
- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
- `experiments/` 是实验记录，从 exp000 开始编号。
- `.claude/rules/` 是详细规则，`.claude/hooks/` 是用来强制执行纪律的检查脚本。

## 1. 接手要先做什么（最高优先级）

第一步不是开实验，而是先对齐目标，再放开去调研、把方向定下来。

1. 先读 `GOAL.md`。这是当前目标的唯一来源，由用户来写。用户随时会改它来调整方向，所以每次接手、每个大的步骤开始之前都要重新读一遍，以它为准。它说的是"现在要做什么"，你写的 `experiments/STATUS.md` 记的是"进度到了哪里"，两者不要混。
2. 读本文件，了解铁律、三条研究纪律、对新方向的要求、以及训练前的审查规矩。
3. 读 `experiments/STATUS.md`，看现在到了哪一步。
4. 按 `GOAL.md` 当前的目标，自己读文献、做差距分析：ReID 现在还有哪些没解决好的真问题？最新的工作都在改进什么、又漏掉了什么？形成你自己对"哪里能做出真正新东西"的判断，提出几个有野心、又确实能做出来的候选方向。

方向没想清楚之前，不要开实验。

## 2. 铁律（违反了基本等于白做）

1. 数字只认日志。所有指标都要用代码从日志文件里解析出来，不能凭记忆、凭印象写。
2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
4. 每涨一次点，都要换一个挑刺的角度重新核对一遍，看它是不是噪声、是不是数据泄漏、是不是评测口径前后不一致。
5. 评测口径是冻结的。要改评测口径，必须先问用户。
6. 正式训练之前先用很小的规模快跑一遍，确认不会崩、模块确实在起作用。
7. 做好实验记录（`experiments/decisions.md` 和 `results.md`）。同样的配置加同样的种子，不要重复跑。

永远不要挑随机种子，也不要挑表现最好的那个 epoch，那等于评测作弊。一律上报最后一个 epoch 的结果，不要用 best_model。

## 3. 三条研究纪律（这个项目最容易栽跟头的地方，必须遵守）

第一，判定一个方向"死了"之前，要先定好标准、并且有足够的证据。开始跑之前就把"什么样的结果才算这条路走死了"写进 `design.md`，比如三个种子配上两种配置都落在噪声范围内或者为负。只有一两个负结果的时候，只能写"还需要再试"，不能判定整条方向死掉，更不能据此就推翻方向。一个活跃的方向往往要反复试很多次才出东西，掉几个点很正常，不要一受挫就放弃。

第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。

第三，"贡献"是个有门槛的词，不要夸大。一个结果，只有同时满足下面几条，才能叫做贡献、才能说可以投稿：通过了第 4 节对新方向的要求；跑了三个种子、报了均值和标准差；涨幅超过了噪声范围；并且和最接近的已有工作区分得清清楚楚。在那之前，一律只叫它"信号"或"探索"。复现别人的方法、公开别人没公开的基准、做一个分析，这些都不算贡献。

## 4. 一个新方向值不值得做（先过这一关）

ReID 是个活跃领域，每年都在出 B 类甚至 A 类的工作。不要一上来就觉得"能做的都被做完了"，那是错觉。你的任务是放开去找一个真正新颖、有效、又讲得清楚的角度。

一个新方向至少要满足下面三条里的两条，否则不作为主线：
1. 问题上有新意：不是"加一个模块"，而是重新定义、或者更准确地刻画一个真实存在的问题。
2. 机制上有新意：是过去的工作没有清楚写出来、而且代码上能实现的机制。
3. 证据上讲得清：能设计出干净的对照和消融，能回答"它为什么有效"。

另外几条硬要求：
- 要和最接近的已有工作区分得清清楚楚，不能是换个名字的同一个东西。
- 方向定下来之前，自己和 codex 或者子代理讨论核实，确认它确实是新的、和最接近的工作区分得清楚。
- 不能拿测试时的小技巧（重排序、特征归一化、翻转测试这类）当作主要贡献。
- 不能用"比基线高了零点几"来定义创新。

先把文献读够、把方向选准，再去花算力。要保持不轻易放弃的劲头，但动手之前多花时间读论文、做差距分析。

## 5. 正式训练前的两轮独立审查（改了方法的实验，必须做）

任何改了模型、或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查：
- 一轮由 Claude 做：用 Agent 工具起一个 opus 子代理来审。
- 一轮由 codex 做：用 `codex exec`，内联在 Bash 里跑。

两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。

规矩是这样：
- 只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。
- 操作上的、文档上的小问题，记成待办，不拦。
- 实质问题修好之后必须再审一轮，不能修完就放行。某一轮里两个审查都没有实质问题，才算放行。
- 纯复现实验（只改随机种子）不用审查，在 `design.md` 里写一行"需要训练前审查：否"就行，检查脚本会放行。
- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。

## 6. 一直往下做（只要用户没说停）

默认的工作节奏是：先把方向定下来（读文献），写好 `design.md`，过两轮独立审查，先小规模快跑一遍，再正式后台训练，用 Monitor 跟着日志看，跑完立刻补好文档（results 和 decisions），然后接着做下一个。

- 每个大步骤开始之前都重新读一遍 `GOAL.md`。它变了就马上按新的来，用户是靠改这个文件来调方向的，不一定会打断你。如果 `GOAL.md` 的主目标被清空、或者写成了"暂停"，就停下来等用户，不要自己找活干。
- GPU 不要空着：要么排下一个实验，要么补文档、读文献、做消融表。
- 不要频繁问用户。长期自己往下做、自己拿主意；拿不定的先找子代理或者 codex 讨论再定。只有真正只能用户决定的事（改评测口径、大方向的取舍），才打断用户。
- 用 Monitor 或者后台的 Bash 等待器来跟日志，不要用 sleep 反复轮询。
- Claude 的额度紧张时，能独立完成的子任务（独立审查、讨论、探索）多交给 codex，省额度。

## 7. 机器和网络

- 你在 Mac 上跑，能联网（GitHub、pip 镜像、HuggingFace 镜像都通）。
- 服务器只有国内网，装包用清华源，下模型用 hf-mirror。
- 三台 GPU（详细连接方式见 `.claude/rules/remote_server.md`，连接信息在 `~/.ssh/config`）：
  - `hyy-5060ti-double`：恒源云，两块 5060Ti 16G，环境已经配好，`/hy-tmp` 只有 50G。
  - `lab-3090-d`：实验室的 RTX3090 24G，在一个 docker 容器里，经 `lab-3090` 跳板连；容器一重启就会丢掉 sshd 和 IP。
  - `lab-4090`：实验室的 RTX4090D 24G，是共享机器，只能用 `afr` 自己的空间，绝对不要碰 `/root`、`/hy-tmp` 和共享的 conda。
- 磁盘纪律：`/hy-tmp` 只有 50G，每次训练只保留最后一个 epoch 的 checkpoint，中间的和 best_model 都删掉。
- 训练在服务器后台跑（用 `setsid nohup ... </dev/null &`），Mac 这边通过 ssh 监控、解析日志。

## 8. 代码底子（上游 SOLIDER-REID）

- 上游只带了 Market-1501 和 MSMT17 的配置（Swin 的 Tiny、Small、Base）。要用别的 benchmark（比如 Occluded-Duke、Occluded-ReID），得自己加配置和数据集读取代码。
- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
- `SEMANTIC_WEIGHT` 默认是 0.2。
- 主干的顺序：先用 Swin-Tiny 快速验证，有信号了再放大到 Small、Base。创新点先在 Tiny 上看有没有效果，确实有效再放大，不要在没效果的点上用大主干硬磨、白白浪费算力。

## 9. 怎么写中文（这一条务必照做）

所有给人看的文字，包括文档、注释、提交信息、以及给用户的回复，都要用平实、完整的中文句子，像正常人说话写字那样。

下面这些不要出现：
- 不要把判断标准、检查步骤之类的东西生造成"某某门"（比如"创新门""双审门""新颖性门"）。中文不这么说话。要表达就直接说清楚，比如"判断方向值不值得做的标准""训练前的两轮审查"。
- 不要用箭头（→）把几个词串起来代替句子，过程要一步一步用完整的句子写出来。
- 不要用破折号（——）来连接或停顿，改用逗号、句号，或者干脆把话说完整。
- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
- 不要堆砌小标题、加粗、表格来充字数，能用一段话讲清楚就用一段话。
- 不要用"收口""罗盘""锚点""铺路""掺进去"这类比喻性的简写和网络口头禅。

这一条对你自己想问题的时候同样适用：思考的时候也用完整的中文句子，不要用上面这些写法。

## 10. 详细规则

放在 `.claude/rules/` 里：
- `result_discipline.md`：数字纪律和三条研究纪律（最重要，先读）。
- `experiment_protocol.md`：实验怎么命名、`design.md` 的格式、两轮审查的流程。
- `innovation_and_decisions.md`：决策怎么记录、怎么让两个子代理对着辩论、对创新方向的把关。
- `documentation.md`：文档结构和数据一致性。
- `monitoring.md`：怎么用 Monitor 和后台 Bash。
- `remote_server.md`：三台机器的详细连接方式和磁盘策略。

exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# 活动状态（每步更新）

## ★★★ 当前最新（2026-06-14，转"补标准数据+公平对标"执行路线 决策#54，正获取标准LReID数据+搭LSTKC标准协议台子）

**方向转折(决策#54)**: 大方向取舍抛用户后用户表达失望("算了我对你绝望了")。复盘: 我一直在查先例/做归因/把候选判负/再把方向推回用户,优化"别做通用/已有的东西"而非"做出完整的东西",耗算力却没落地。**改打法,自己拿决定不再punt**: 不再"找新机制→判负"循环,直接解决卡脖子的"缺公平对标台子":
- 用标准 LReID 协议代码库 LSTKC(AAAI2024, 已 clone 到 lreid_lstkc/): 标准5域顺序(market→cuhk_sysu→duke→msmt→cuhk03)+遗忘矩阵+未见域泛化+PTKP/PatchKD/LSTKC强基线全现成,是 C2R/AKT 共用的台子。
- 补标准数据(已有Market/MSMT;取CUHK03/CUHK-SYSU/Duke[学界通用,论文不展示图]/7个unseen,子代理 a98cae3 正查获取路径)→复现LSTKC/PTKP基线对上其论文数字(纪律2)→在公平台子做方法对标C2R/AKT,三种子出稿。
- 候选二"多版本混库塌21mAP"作为干净 motivation,标准协议上做成多版本兼容/免全量重建索引的真方法。
hyy 候选二复现序列(v2 OccDuke)仍在跑,给第二条序列的多版本不兼容佐证。

**执行进展(2026-06-14 晚)**: 子代理给出 link 核验的数据获取计划(无预处理包,各自装;Drive 必须 Mac 下再 rsync;CUHK-SYSU 自裁剪;MSMT 加载器只支持 V2)。已完成:
- LSTKC 环境在 lab-4090 装好(独立 .venv_lstkc, torch2.x, continual_train.py --help 通过; uv 超时调 UV_HTTP_TIMEOUT=600 才装上)。
- Mac 下好 3 个训练域: CUHK03 cuhk-03.mat(1.1G)+新协议 mats、DukeMTMC-reID(154M,版图对)、CUHK-SYSU(160M)。脚本在 experiments/exp029_lifelong/{lreid_download_mac,setup_lstkc_env}.sh。
- **5 训练域数据已全部到手**: ①Market+MSMT17_V1 已符号链接进 PRID/(版图校验OK); ②MSMT 加载器已 patch 支持 V1(VERSION_DICT 加 V1 条目; V1≈V2,仅人脸模糊+目录名不同,身体ReID数字几乎无差); ③CUHK03 cuhk-03.mat(1.1G)+新协议mats 已下; ④Duke 已下(版图对); ⑤CUHK-SYSU **原始** person-search 已下(1.2G tar.gz,18184 SSM图+annotation),排成 cuhk-sysu/CUHK-SYSU/ 加载器自裁剪版图(之前那个160M是预裁剪版,弃用)。
- 关键发现: 服务器够不到 Google Drive/Google,但能到 Github+hf-mirror;故 Drive 数据(CUHK03/Duke/CUHK-SYSU)Mac 下再 rsync 到 lab-4090。relay4090 跳板不能并发 ssh,传输要串行。
- 剩(纯工程): rsync CUHK03+Duke(在跑 bfj7oxudl)+CUHK-SYSU raw 到 lab-4090 /mnt1/afrdata/PRID/ → 部署 msmt17.py patch(scp,等 relay 空)→ 跑 continual_train.py 复现 LSTKC 标准5域基线对上其论文数字(纪律2)→ 在公平台子做方法。
---

**全局裁决(经验,非先验; 详 results.md exp029 + 决策#51/52/53)**: 候选一(跨阶段桥)无效、候选二(多版本陈旧索引)前提判负=经典BCT不兼容悬崖非新结构、候选三(阶段顺序桥选择)前提弱=遗忘在差异极大目标域上高度一致(−20.7~−21.9,无域距离结构,LTCC偏小由数据集大小解释)。task-free无边界(4.5/10)未测且协议风险高。codex两次独立判终身方法稿在现约束下 3-4/10,binding=缺标准数据(Duke/CUHK-SYSU/CUHK03,无法公平对标)+主流子问题全拥挤。机制候选已便宜先导穷尽→剩下的路被用户专属决定主导(补敏感数据/换论文形态/离开终身)→已用 AskUserQuestion 抛大方向。hyy 候选二复现序列仍在跑确认,不空等。
---

**候选二裁决(主序列, 详 results.md exp029 + 决策#52 执行结果)**: 判负。复现校验过(v0 Market 87.01=训练日志)。Market 陈旧索引: mAP 阶梯非单调(g=v0 51.74>g=v2 50.85,是 v0 作为 Market 专家的质量混淆,非版本年龄递减); 真正大效应是"混版本图库" −21mAP,而同质旧版本单独可用(g=v0 51.74)→ 这是经典 BCT/C2R/Hot-refresh 不兼容悬崖(那条线的立项动机),不是新的可利用版本年龄结构。候选二作为新方法线判负。hyy 独立序列(Market→OccPoseTrack→OccDuke)跑完确认稳健性。
**已转候选三便宜先导(lab-4090, run_exp029_cand3.sh, 等待器 bn1jz0w7n)**: 从同一 v0 单步 fine-tune 30ep 到 OccDuke/PRCC/LTCC/MSMT,量各目标造成的 Market 遗忘,看是否随目标域/域距离有规律(校正数据规模混淆)。有强顺序效应→候选三(4/10)前提成立可继续; 遗忘与目标域无关→候选三前提弱。
---

**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。

**codex option-B 战略复审**：冻结协议内基本无可撑 CCF-B 的方法贡献；要做方法稿须放松一个约束。**用户拍板：转终身/增量 ReID(工程深度)**——真正不同的问题类(训练过程+数据流上定义新问题),离死线远,中等拥挤但活跃。

**缺口分析已出**(lifelong_reid_gap-codex.md):候选一(跨阶段同身份桥)零训练先导无效(PRCC 换衣太轻,原型检索已 100%,无空间,决策#51);候选二(多版本陈旧索引)其次,且能用手上 Market/MSMT/Occ-Duke 标准不相交身份序列直接做。候选三(阶段顺序敏感桥选择)备选,最接近 AKT/DASA/DKAR 先例压力大。

**exp029 终身最小基建已搭+先导在跑(2026-06-14 08:06)**:
- 加 MODEL.CONTINUE_FROM 配置 + train.py 注入(顺序微调:加载旧域主干+瓶颈,分类头按域重建) + eval_lifelong_stale.py(多版本陈旧索引评测,复用仓库标准 eval_func,单版本退化即标准评测做复现校验)。本地语法过、lab-4090 烟测过(CONTINUE_FROM 加载 PRCC ckpt 进 Market 训练,分类头重建,loss 9.55→正常,13.4s/epoch)。
- lab-4090 后台跑顺序微调代理序列(seed 0):v0=Market(120ep,jx_vit 初始化)→v1=+MSMT(60ep,接 v0)→v2=+Occ-Duke(60ep,接 v1)。约 70min,EXP029_SEQ_DONE 标志。这条序列同时产出朴素基线+遗忘矩阵对角。
- 训练后:先用 v0 在 Market 的训练日志 mAP 校验 eval_lifelong_stale.py(复现校验),再跑候选二探针(陈旧阶梯 g=v0/v1/v2 + 随机混版本)。判负线见 design:混版本掉点<1mAP 或阶梯非单调(悬崖式=通用不兼容非可利用结构)→候选二判负转候选三。
- 代理序列只做机制先导+工程自检,不冒充标准 SOTA(缺 Duke/CUHK-SYSU/CUHK03);若候选二过线,再补标准数据做公平对标。

**候选二先例核查(codex 3/10, candidate2_vs_c2r-codex.md, 决策#52)**: 免重建索引+持续兼容转移被 C2R(CVPR24)/Bi-C2R(TPAMI26)/URCPD(AAAI26) 占; 新旧兼容+在线部分回填+预算曲线被 BCT 系/Hot-refresh(ICLR22,已做回填比例+预算曲线)/DGR 占。窄空白=多版本原始特征同库+相机/时间非均匀陈旧+预算调度,且增益须可证来自版本年龄结构。前提门=两条独立序列在 Market 上看混版本是否明显掉点+随版本年龄单调(对照各版本 native 自评排"质量阶梯"混淆)。不过→转候选三(阶段顺序敏感桥选择,未撞免重建/BCT 工业线); 过→仍须全套破坏性归因(打乱版本/随机陈旧/全重提/年龄反转吃掉70%增益)+超 C2R 全量转移基线+三种子才算贡献(同杀模块线那道归因门)。

**两条独立先导(都在跑)**: lab-4090=Market→MSMT→OccDuke(主, ~13s/epoch); hyy GPU0=Market→OccPoseTrack→OccDuke(独立复现,不同中间域,防单序列偶然)。两条都在 Market 上做陈旧索引评测(eval_lifelong_stale.py 已用 Occ-Duke ViT-base 53.70 复现校验通过)。完成等待器 b2adeq7sh(lab-4090)/b8pv682at(hyy)。
GPU 分配: hyy GPU1 + lab-3090-d 暂空——候选二仅 3/10 的前提先导,2 个工作位已足,不在低先验方向上铺满算力; 前提过线再扩种子/破坏性对照/标准数据。

**候选三+拓宽先例核查(codex, candidate3_and_broaden-codex.md)**: 候选三(阶段顺序敏感稀疏旧域桥选择) 4/10,被 AKT/DASA/DKAR/DKP/PKA/LSTKC 从"动态迁移+旧知识过滤"两侧挤,只有当便宜先导证明域距离能预测遗忘/迁移(Spearman≥0.4、近远旧域遗忘差>1mAP)且随机相似度/最远桥/全旧域均匀对照吃掉增益才值得做。拓宽最不坏两条: task-free/无边界终身(按相机时间 shard 漂移触发 micro-domain,4.5/10,协议风险高,最接近 CIPR/CVS) > 候选三收窄版(4/10)。整体判断: 当前数据(缺 Duke/CUHK-SYSU/CUHK03)+算力下终身方法稿 3-4/10,codex 诚实建议若前提都不成立则别硬做、把 exp029 基建当协议分析素材退回更稳形态。**策略: 先用 exp029 一条序列同时验候选二(陈旧阶梯)和候选三(域距离→遗忘)两个前提的经验证据,有强信号再投;两个前提都经验证伪才把"是否离开终身方向"作为有证据的大方向问题抛用户。**

资产:PRCC/LTCC/Market/Occ-Duke 标准 ViT-base 3 种子基线 + 各探针/归因基建。lab-4090 venv=/home/afr/reid-clean/.venv、hyy venv=/hy-tmp/reid-clean/.venv(python 不在 PATH,必须用 venv 全路径)。MSMT 已 symlink MSMT17→MSMT17_V1。jx_vit 两机都在(lab-4090 /home/afr/reid-clean/weights/、hyy /hy-tmp/reid-clean/weights/)。

---

## 旧（2026-06-14，公平基线纠正——已被模块线全关+终身转向取代）

**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。

**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。

重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。

**已落到真训练（exp028 CCS 跨衣稳定证据保全，主注）**：
- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
- hyy: MGPARTS 占位消融因 protos/bank 路径依赖重建复杂、信号弱(flat mAP)已放弃; 改后台把 PRCC(217M)拷到 hyy，为 exp028 出信号后并行消融/LTCC 做前置。lab-3090-d 空闲（PRCC 瓶颈在 lab-4090）。

下一步取决于 exp028: 过判负线则补消融(采样器对照已=基线,需±衣服破坏视图)+LTCC+对标CAL/CCIL; 不过则按"还需再试/判负"处置,转 PARTIAL_EVIDENCE 或 exp024 结构线坐实。

注：仍真死的（机制被证伪或新颖性被占，换基线也救不了）：OC4 超加性是数据属性（决策#47）、exp022 贴图检测器机制错、与 CAL/CCIL/instance-wise invariance 同质的部分。

---

## 旧（2026-06-14 夜，自适应不变性判负——此条"整簇都死"的泛化已被上面的纠正收窄）

自适应不变性方向判负（决策#48）。Level-1+flip噪声地板: PRCC换衣 blur 改善15.5%真异质,但 Market 标准 ViT 仅8%(噪声地板6.8%)、强主干 Swin 7.5%<8.7%。Level-2 冻结选择器: PRCC oracle+6 但选择器预测线索28%≈随机25%,应用-0.94mAP。注:Level-2 是【冻结探针】非真训练,按新原则这属"探针假象"待真训确认。

正在做: 起多路 codex 先例优先搜下一方向(明确禁用鲁棒性/不变性/去偏/线索手术死簇),我用 WebSearch 逐个核实先例。资产复用: PRCC/LTCC/Market 标准 ViT-base 基线 + 三个零训练探针(指纹/oracle/冻结选择器)在 lab-4090/hyy。三机当前空闲(探针类零训练已完成),搜出方向+探针过线再占 GPU。

---

## 旧（2026-06-14，OC4 冲突角度门0 判负）

---

## 旧（exp027 VCR 实现+双审，已被门0 判负取代）

方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。

进度：
- VCR 插件已实现（model/vcr.py + make_model/processor/make_dataloader/defaults 接线 + configs/{prcc,ltcc}/vit_base.yml + test_vcr.py + probe_superadd.py），默认关闭逐字节退化，本地 CPU 自检 6/6 过。
- 第一轮三路独立 codex 审查全部不放行，但都是可修问题：新颖性 5.0/10（窄但开放，命门是门0 超加性和交互项消融）；正确性两处（衣服关系损失 detach 切断路由器、跨衣损失用了 PRCC/LTCC 不可达的异身份同衣对）；协议两处（LTCC 探针把 general 当同衣 SC、合成贴图不能与 OC4 官方数字同台）。
- 已全部修复并验证：梯度流改进后用梯度检查确认衣服关系损失只训练路由门与关系头、不灌主干；跨衣损失改可达三元组、关系损失改身份内同衣换衣；LTCC 探针改 mode=SC、单位统一个百分点；design 收窄叙事、命名 Synth-Occ、补 CAL 式全局对照与先例核查清单。提交 0adc745、584a5dc。
- 第二轮 codex 审查：正确性 approve（无实质问题）；协议放行门0 探针与小规模冒烟，门1 三种子论文级裁决暂不放行，要求补一个与完整 VCR 同双前向口径的"只增广"控制臂加显式消融开关（这是公平性要求，不挡门0）。
- 正在跑（lab-4090，已恢复）：PRCC 标准 ViT-base 基线 seed0（约 20 秒一个 epoch，120 轮约 40 分钟）→ 门0 超加性零训练探针（两个遮挡种子）→ LTCC 基线 → LTCC 探针，远端 orchestrate_oc4.sh 串起来，Mac 侧等门0 结果通知。

下一步取决于门0：超加性 > 0.4 个百分点则方向成立，补公平控制臂与消融开关后三种子训练 VCR；否则问题定义偏弱，判负转向。hyy 两卡、lab-3090-d 空闲（都缺换衣数据，PRCC/LTCC 只在 lab-4090）。

---

## 旧（2026-06-13 20:37，用户拍板"新信息源"，3路codex深读PDF全文中，A路和C路已完成）

部位方向(exp020/024/025/026)与换衣旋钮(决策#45)全判负归档,三机空闲。用户拍板大方向="调查新信息源,仔细看168篇B类论文"。
3路codex深读全文(experiments/paper_materials/newinfo_*.md):
- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
- B 生成/合成数据: CCUP/InfinitePerson/Identity Diffuser/扩散增广,找适中成本+新颖的生成机制(换衣战场PRCC 49 vs SOTA 55-66,CCUP用百万合成预训练)。
- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
出齐后综合排序+零训练探针,交用户定具体押哪条。纪律不变:探针先行,强主干验证,三种子。

---

## 旧（2026-06-13 15:30，新方向调研收敛：换衣衣服捷径为第一推荐，待零训练探针）

6 路 codex 调研全出（paper_materials/newdir_*.md）。候选清单第一推荐：**基于 SEMANTIC_WEIGHT 的衣服捷径可控蒸馏**。
- 为什么排第一：换衣是强主干仍明确失败的具体干扰（PRCC 换衣远低于同衣）；我们有独有观察（PRCC 换衣对 w 有约 7 分响应、Occ-Duke/LTCC 不响应）；已发表换衣方法都靠衣服标签/解析/CLIP/生成，"用预训练自带语义旋钮控衣服捷径、无衣服标签、测试单 embedding"是干净区分点。
- 命门（红队 codex 正在核准）：笔记里"w=0.2 换衣好"与"语义权重升换衣涨"文字自相矛盾，必须从代码核准 w 到底控制什么；w 差异方向若只是通用 detuning 而非衣服捷径，方向即塌。
- 执行铁律（血换的教训）：先做零训练探针（现有 PRCC/LTCC/Occ-Duke checkpoint 上提不同 w 特征、衣服方向线性探针、闭式衣服方向擦除对随机方向），探针过线才写 design、才开训练。预注册判负线已在候选报告里。
- 后备候选：换衣状态原型(2)、状态扰动结构保持蒸馏(3)、模型敏感遮挡课程(4)、相机白化(5)。

部位/掩码方向确认种子(exp026 nd_s1/s2)在 hyy 收尾(~15min)、PoseTrack 在 3090 收尾(~1.5h)，纯清账。新方向探针不需要训练 GPU。

---

## 旧（2026-06-13 14:45，方向重置：部位/掩码方向终审判负，回语料找方法级新方向）

用户拍板转向（原话）："回去找新方向，好好看别人怎么发的，主要是别人的思路，作出能发b会的方法级成果，codex任你使用token管够"。决策#44 已记。

**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。

**新方向调研进行中**：4路并行codex（思路解剖/幸存性筛选/相邻领域扫描/强论文结构逆向），写入 paper_materials/newdir_*.md。出齐后第二波（候选生成+新颖性红队），再交用户拍板。重活全交codex，Claude只做全局把控。

**机器**：hyy双卡=exp026确认种子s1/s2(~1h)；3090=PoseTrack对照s2重跑→部位s1链；4090=跳板断，恢复循环守着。新方向定下来前不开新训练。

---

## 旧（部位/掩码方向攻坚期，2026-06-11-13，已判负归档）

## ★★★ 当前最新（2026-06-11 午后 14:10，exp022 判负在途、exp023 形状池审查中）

用户硬要求不变：方法稿、双指标。主线推进到第三层迭代：

**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。

**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。

**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
- 审查放行后顺序：lab-4090 跑冒烟 → 方法臂 s0/s1/s2 + aug s2 链；hyy 跑 aug s0/s1（库已在传）。全部今天 ~17:30 出齐。
- Swin-Small 移植已就绪（827e023，双自检过），exp023 有信号立即上（生死线）。

**锚点收尾**：Swin-Base s2 在 lab-3090-d（~14:30 完）；s0 64.4/72.9、s1 64.5/73.3 已档。

更新（15:00）：exp022 已封档（方法 53.97±0.58/60.50±1.06，对纯增广 -0.63/-0.97，判负；α 轨迹三种子一致、死因=贴图分布错配，诊断三件套进论文）。Swin-Base 锚点封档 63.63±1.15/72.50±0.86。exp023 冒烟过闸、方法 s0 在跑。库 md5 四处字节一致（f9370729…），跨机库混杂顾虑解除。

机器：lab-4090=exp023 六臂链（方法s0→aug s0→…，~18:45 收齐，第一对同种子读数 ~16:20）；hyy=wocc0 双臂收尾（~15:00）→自动接力 Swin-Small 形状库 aug 对照 s0/s1（带冒烟闸门）；lab-3090-d=Swin-Small 形状库 aug 对照 s2（65.6s/epoch，~17:10 完）。

---

## 旧（2026-06-11 午后早些，方法稿主线：exp022 三种子链在跑）

用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。

**主线 exp022（构造精确遮挡掩码监督，方法稿核心）**：
- 机制：贴图增广免费产生的像素级遮挡位置 → token 网格 0/1 掩码 → 线性头在第 8 块后预测、只在合成样本上 BCE 监督 → 零初始化 α 可学重加权喂回主干。测试零外部输入。
- 两轮独立审查同轮通过（069249a），七项自检两台机器全过，冒烟通过：occ_bce 1.01→0.26（头在学）、α 在动（注意走的是负方向，收敛后符号待看）、评测路径正常。
- **三种子链在 lab-4090 跑**（4090 速度 15.7 秒/epoch，全链约 2 小时，监视器 biefwf3ui 盯 done 标记）。
- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
- 可视化脚本 viz_occaware.py 已部署（真实遮挡热图 + 合成三联图 + 留出 IoU），s0 一完就出图。

**对照臂 exp021（纯增广数据轴）**：s0 完成 54.4/61.1（对同种子控制 +1.8/+1.2）；s1（第 49 轮）/s2（第 32 轮）在 hyy 双卡，等待器 bncb0fpt4。
**锚点收尾**：Swin-Base 无 cp，s0 64.4/72.9、s1 64.5/73.3 已档，s2 在 lab-3090-d（第 34 轮，等待器 bv5tdiume；注意容器时钟 UTC+8 才是北京时间）。

机器：lab-4090=exp022 链；hyy 双卡=exp021 s1/s2（完了接 exp022 的 W_OCC=0 归因臂或重试臂）；lab-3090-d=Base s2。

---

## 旧（2026-06-11 清晨，第二夜战役：方法三连测+锚点全家桶）

**方法线（候选 A 的三种形态，全部走完预注册流程）**：
- exp018 蒸馏损失版：三种子 +0.33（无信号，归档为消融臂）。
- exp019 OSG 门控版：三种子 +0.20（无信号；门控诊断 g2≈0.001 门没开）；W_SEM 0.3 预注册重试在跑（hyy GPU0）。
- **exp020 LGP（无姿态语言部位分支，主注）**：保留 LGPA-D 被验证的两个成分（部位级 ID 监督+部位特征进表征）去掉全部姿态依赖；两轮审查同轮通过（裁决程序：对照双杆同报、不用配对差措辞），seed0 在跑（hyy GPU1）。信号线 ≥+0.8 对 53.27/53.63 双对照。
- 关键认知：探针证明"信息存在"，三连测在证明"哪种注入形态才有用"——弱监督门控不行，强监督部位分支是 PSG 验证过的形态。

**锚点全家桶（全部 w=0.2 发表级配方，数字在 results.md）**：
- Occ-Duke：Tiny 56.87±0.29（exp018 对照三种子）；Small w=1.0 66.53±0.12 + w=0.2 链（66.9/66.6/s2在跑）；Base 64.4（无cp，seed1/2 排队）；ViT-base 53.27±0.49（exp019 对照三种子）。
- Occ-PoseTrack：Tiny 76.27±0.09（exp001h，已核实 w=0.2）；Small 77.40±0.08 三种子。
- Occluded-REID 跨域容量曲线：Tiny 71.0→Small 84.2→Base 86.4 mAP（官方检查点零训练直评；裸 Base 86.4/89.2 超 FED/KPR带提示/BPBreID，距 ProFD 约 2 点）。
- 换衣：PRCC Tiny w02 三种子 46.17±1.13（w 旋钮效应 +6 mAP 坐实）、Small w02 49.4；LTCC 全档低于发表（seeds 在补）。
- **两个工程毒点已揪出并落档**：WITH_CP（重入式检查点+AMP）毁训练（41.7→64.4）禁用；SEMANTIC_WEIGHT 默认 1.0 污染（全线统一 0.2 烤进配置）。

**强主干审计三支柱（独立于方法成败的分析主线）**：同域遮挡白送（裸 Small 超全部已发表 B 类）、跨域遮挡近顶（裸 Base 第二梯队）、换衣失灵但有 w 旋钮。加 14 个方法判负的谱系和可靠检索保底稿。

在跑：hyy=OSG重试+LGP s0；lab-3090-d=Duke Small w02 s2；lab-4090=LTCC tiny w02 s1/s2。等待器全挂。

---

## 旧（2026-06-10 凌晨，用户完全放权，夜间自主战役进行中）

用户指令（原话要点）："不要把自己限定在遮挡这个领域，大胆创新，大胆做所有可能让我们稳发 B 类的实验，三台 GPU 全用，不依赖 codex（额度尽，06-11 13:01 恢复），别用 Workflow（会话额度也耗尽过一次），你自己做全部决定，我去睡觉。" 方向部署见决策 #42：锚点先行的双战场，主纲领"强人本主干 + 训练期新信息源/训练组织，测试单 embedding 冻结"。

到 09:10 为止的战果（数字都在 results.md，全部提交）：
- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
- **exp018（语义蒸馏 v1）实现完成**：插件式 SEMDISTILL（defaults/make_model/dataloader/processor），自检五项全过（ENABLED=False 字节级退化、标签 15618 全齐、蒸馏在学、eval 前向 max|diff|=0）。CLIP 标签和代码已部署 hyy。等 10:40（Claude 子代理配额）做第一轮审查、13:01（codex 额度）做第二轮，过审即 smoke+三种子。
- Swin-Base：下载+转换+加载烟雾全过（373 键 All matched）。

正在跑：
- lab-3090-d：Occ-Duke **Swin-Base seed0**（with_cp，114 秒/epoch，约 4 小时完，首个 Base 遮挡锚点）。
- hyy 双卡：**Occ-PoseTrack Small seed1/seed2**（Epoch 29/27；注意该配置 EVAL_PERIOD=20 会打中途 mAP，等待器盯 transformer_120.pth）。
- lab-4090：Small 三连链 **Occ-PoseTrack Small seed0 → PRCC Small seed0 → LTCC Small seed0**（链 pid 1719268；后两个是换衣捷径的容量趋势分析）。
- 各完成等待器已挂；check_monitor.sh 钩子按用户指示已删（保留 check_design.sh）。

---

## 旧（2026-06-10 早，通读 167 篇 2025/26 CCF-B ReID 论文找新角度，已完成）

用户醒后对可靠检索应用稿的定位有疑虑（它不是"模块带来大提升"的故事，而且最稳的去处 PRCV 是 C 类不是 B 类），下载了 167 篇 2025/2026 年 CCF-B venue 的 ReID 论文（paper_materials/pdfs_by_title/），指令是和 codex 配合全部读完、提炼经验、找出一个能发的角度。用户确信一定有角度。范围约束不变：只做遮挡+普通 ReID。

阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。

---

## ★★★ 旧（2026-06-09，转可靠检索应用稿，覆盖以下所有旧状态）★★★

**方向(决策#39，用户拍板)**：选项1 加机制涨标准 mAP 累计判负 13+(MULTIHYP/DONOR/想法2/找新机制三候选[成对匹配/episode soft-AP/框所属者蒸馏]全栽 train→test 泛化/容量律/分割失效)。转**遮挡 ReID label-free 可靠检索应用诊断稿**(非新方法稿，定位见 paper_materials/reliable_retrieval_redteam-codex.md，命中 PRCV 0.5-0.65/ICME ICPR 0.35-0.55)。

exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# 当前目标（唯一来源，随时改这里来调方向）

这是 Claude 当前要追的目标。要调整方向，直接改这个文件就行，不用 `/goal clear` 那一套。Claude 每次接手、每个大步骤开始前都会重新读它，以它为准（见 `CLAUDE.md` 第 1 节和第 6 节）。这个文件由用户来写，定的是"做什么"；`experiments/STATUS.md` 由 Claude 来写，记的是"进度到哪"。两者不要混。

## 主目标

做一篇 CCF-B 级别、真正新颖、有效、又讲得清楚的行人重识别工作。

## 现在这一步

自己读文献、做差距分析，提出几个有野心、又确实能做出来的候选方向。每个方向写清楚：它解决的是什么真问题，新机制是什么，打算怎么证明它有效，和最接近的已有工作差在哪里。提完拿给用户挑。先不要开实验。

## 这一步什么时候算完成

提出三到五个候选方向，每个都满足"问题、机制、证据这三条里至少占两条"，并且和最接近的工作区分清楚。和codex或者子代理去讨论之后选择其中一个，才进入写设计文档、做两轮审查、正式训练的环节。

## 不要做的事

- 方向还没定、还没通过对新方向的要求之前，不要开实验。
- 不要夸大。一个结果，没有同时满足"对新方向的要求、三个种子的均值和标准差、涨幅超过噪声范围、和已有工作区分清楚"这几条，就只能叫"信号"。

## 暂停信号

- 用户说"停"就停。
- 这个文件的主目标被清空、或者写成"暂停"，就表示当前没有目标，等用户的指示，不要自己找活干。

exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.622.11653/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
---
name: "pdf"
description: "Read, create, inspect, render, and verify PDF files where visual layout matters. Use Poppler rendering plus Python tools such as reportlab, pdfplumber, and pypdf for generation and extraction."
---

# PDF Skill

## When To Use

- Read or review PDF content where layout and visuals matter.
- Create PDFs programmatically with reliable formatting.
- Validate final rendering before delivery.

## Workflow

1. Prefer visual review: render PDF pages to PNGs and inspect them.
   - Use `pdftoppm` from the bundled runtime or system Poppler when available.
   - If unavailable, install Poppler or ask the user to review the output locally.
2. Use `reportlab` to generate PDFs when creating new documents.
3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
4. After each meaningful update, re-render pages and verify alignment, spacing, and legibility.

## Temp And Output Conventions

- Use `tmp/pdfs/` for intermediate files; delete them when done.
- Write final artifacts under `output/pdf/` when working in this repo.
- Keep filenames stable and descriptive.

## Dependencies

Prefer the Codex bundled workspace/runtime dependencies when available. The primary runtime is expected to include:

- Python packages: `reportlab`, `pdfplumber`, `pypdf`
- Rendering tools: `pdftoppm` and `pdfinfo` from Poppler

If a dependency is missing, install only what is needed.

Python packages:

```bash
uv pip install reportlab pdfplumber pypdf
```

If `uv` is unavailable:

```bash
python3 -m pip install reportlab pdfplumber pypdf
```

System tools for rendering:

```bash
# macOS (Homebrew)
brew install poppler

# Ubuntu/Debian
sudo apt-get install -y poppler-utils
```

If installation is not possible in this environment, tell the user which dependency is missing and how to install it locally.

## Environment

No required environment variables.

## Rendering Command

```bash
pdftoppm -png "$INPUT_PDF" "$OUTPUT_PREFIX"
```

## Quality Expectations

- Maintain polished visual design: consistent typography, spacing, margins, and section hierarchy.
- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
- Charts, tables, and images must be sharp, aligned, and clearly labeled.
- Use ASCII hyphens only. Avoid U+2011 and other Unicode dashes.
- Citations and references must be human-readable; never leave tool tokens or placeholder strings.

## Final Checks

- Do not deliver until the latest PNG inspection shows zero visual or formatting defects.
- Confirm headers, footers, page numbering, and section transitions look polished.
- Keep intermediate files organized or remove them after final approval.

codex
目标文件和项目方向已经对齐：现在只做文献差距分析，不开实验。下一步我会逐篇抽前三页，重点看它把问题怎么定义、机制怎么卖、证据怎么组织，而不是复述全文。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'ls -1' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
'Knowledge and experience' for visible-infrared person re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Occluded person Re-Identification with noise injection.pdf
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
manifest.csv

exec
/bin/zsh -lc 'command -v pdftotext && pdftotext -v 2>&1 | head -n 2' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/opt/homebrew/bin/pdftotext
pdftotext version 26.04.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org

codex
我已经确认 `pdftotext` 可用。现在分批读，避免一次性输出太乱；每篇只取前三页，后面的实验细节和全文方法不作为依据。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Lifelong Visible-Infrared Person Re-Identification with Prompt
Pool and Instance-level Prompt Generator
Zhenxi Luo

Guoqiang Xiao

Southwest University
College of Computer and Information Science
Chongqing, China
1908249431@qq.com

Southwest University
College of Computer and Information Science
Chongqing, China
gqxiao@swu.edu.cn

Michael S. Lew

Song Wu∗

Leiden University
LIACS Media Lab
Leiden, Netherlands
m.s.lew@liacs.leidenuniv.nl

Southwest University
College of Computer and Information Science
Chongqing, China
songwuswu@swu.edu.cn

Abstract

CCS Concepts

Most existing Visible-Infrared Person Re-Identification (VI-ReID)
methods primarily rely on fixed datasets for training, which struggle to accommodate continuously evolving cross-domain data, thus
significantly limiting the adaptation in real-world dynamic scenarios. The task of Lifelong Visible-Infrared Person Re-Identification
(LVI-ReID) emerged and is required to overcome the challenge of
the semantic gap caused by both cross-modality and cross-domain
data. Drawing inspiration from complementary learning systems,
we propose a prompt-based dynamic learning framework to address the challenges inherent in LVI-ReID. Specifically, we design a
Prompt Pool (PP) module to encapsulate shared knowledge across
tasks or domains. In addition, we propose an instance-level prompt
generator (IPG) to further enhance the model’s ability to capture
domain-specific knowledge, overcoming the limitations of a fixedsize prompt pool. For task-agnostic inference during the LVI-ReID
phase, we develop a query-key mechanism that adaptively selects
the most relevant prompt by evaluating the similarity between
query tokens and keys, thereby addressing the nuanced requirements of varying tasks. Extensive experimental evaluations demonstrate the superiority of our proposed prompt learning-based PPIPG framework over state-of-the-art methods in both lifelong learnings, lifelong person re-identification (LReID), and LVI-ReID settings. These results underscore the efficacy and practicality of our
framework for advancing LVI-ReID across dynamic cross-modality
and cross-domains. The source code of our designed PP-IPG method
is at https://github.com/SWU-CSMediaLab/PP-IPG.

• Information systems → Information retrieval; • Computing
methodologies → Computer vision.

Keywords
Lifelong learning, Visible-infrared person re-identification, Prompt
learning
ACM Reference Format:
Zhenxi Luo, Guoqiang Xiao, Michael S. Lew, and Song Wu. 2025. Lifelong
Visible-Infrared Person Re-Identification with Prompt Pool and Instancelevel Prompt Generator. In Proceedings of the 2025 International Conference on
Multimedia Retrieval (ICMR ’25), June 30-July 3, 2025, Chicago, IL, USA. ACM,
New York, NY, USA, 10 pages. https://doi.org/10.1145/3731715.3733373

1

Introduction

Person Re-identification (ReID) and lifelong learning are two prominent research areas in the field of information retrieval. Person
re-identification primarily aims to associate the same individual
across non-overlapping camera views, while lifelong learning focuses on continuously acquiring knowledge from new tasks or
domains without forgetting previously learned knowledge. Given
the dynamic adaptation requirements of real-world applications,
models need to transcend the limitations of traditional training on
fixed datasets and progressively learn from the ever-evolving data,
thereby providing practical solutions for modern dynamic surveillance systems. The intersection of these two domains is emerging
as a promising research direction, offering a practical path to address the challenges of catastrophic forgetting in lifelong person
re-identification (LReID).
Relevant studies [21, 58] have proposed an online learning method
for one-shot person re-identification and a continuous representation learning framework for biometric recognition, respectively.
The relatively narrow domain gap between the training and testing
datasets in these approaches presents fewer challenges in maintaining learned knowledge while enhancing generalization capabilities.
Notably, AKA [32] pioneered the introduction of cross-domain
LReID tasks, drawing inspiration from human cognitive processes

∗ Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-1877-9/2025/06
https://doi.org/10.1145/3731715.3733373

953

ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA

Zhenxi Luo, Guoqiang Xiao, Michael S. Lew, Song Wu

to propose an adaptive knowledge accumulation strategy, encompassing both knowledge representation and knowledge manipulation aspects. However, existing methods and tasks primarily focus
on single-modality lifelong person re-identification, specifically in
daylight visible-light settings, which limits their applicability to
cross-modal tasks and may cause surveillance systems to fail in
nighttime conditions. Notably, modern camera systems automatically switch to infrared mode when confronted with insufficient
lighting during the night. In recent years, the Visible-Infrared Person Re-identification (VI-ReID) technology has been continuously
developed. The core task of VI-ReID is to use the visible-light images of pedestrians to retrieve their corresponding infrared images,
or vice versa for retrieving similar pedestrians. Wu et al. first clearly
proposed the VI-ReID task and released a large-scale visible-infrared
dataset (SYSU-MM01) [43]. Most VI-ReID studies [13, 29] adopted
a contrastive loss-based strategy to learn a shared feature space
by directly minimizing the distance between visible and infrared
features. Existing approaches, while capable of learning aligned
representations between visible and infrared modalities in a latent
space, typically rely on fixed datasets for training and lack mechanisms to continually accumulate knowledge across tasks or domains.
This limitation renders them impractical for real-world surveillance
systems, which must handle extensive, dynamic, cross-modal, and
multi-domain data streams.
To address these challenges, the task of Lifelong Visible-Infrared
Person Re-Identification (LVI-ReID) emerged, with its overall process illustrated in Fig. 1. The LVI-ReID task presents two significant
challenges. First, similar to LReID, it is a fine-grained open-set
problem, where identities in the training and testing sets are entirely disjoint. Second, unlike LReID, LVI-ReID must tackle not only
the semantic gaps across domains but also the significant crossmodal semantic discrepancies between visible and infrared data.
To effectively overcome these challenges in the LVI-ReID, we propose a novel prompt learning-based framework. Specifically, our
framework incorporates two distinct prompt generation modules
to address the challenges of LVI-ReID. The first is a Prompt Pool
(PP) module, which allows each task or domain to flexibly retrieve
grouped prompts as model inputs. The PP module effectively captures shared knowledge across tasks or domains while maintaining
the independence of task-specific or domain-specific knowledge in
diverse scenarios. The second is an Instance-level Prompt Generation (IPG) module, which dynamically generates instance-specific
prompts from input tokens without relying on a fixed-size prompt
pool. The IPG module also utilizes task identifiers to dynamically
encode domain-relevant knowledge associated with the target task.
The combination of PP and IPG modules significantly enhances
the ability to extract cross-domain features and fine-grained intradomain representations for LVI-ReID. To ensure an efficient and
accurate selection of task- or domain-specific classifiers, we introduce an additional query token. This token, processed through a
dedicated query module, learns the unique features of each task or
domain, guiding the model to interact adaptively with the appropriate keys during inference, removing the need for explicit task
identifiers. During training, a cosine similarity loss is employed to
align the query token with its corresponding key. In the inference
phase, the model identifies the key most similar to the current query

Figure 1: The Pipeline of Lifelong Visible-Infrared Person
Re-identification. The model undergoes iterative training
within streaming domains. During the training phase, the
model must both reduce the modality gap and prevent catastrophic forgetting of previous tasks. Some samples from
previous tasks will be stored in a memory bank for replay
purposes. It is worth noting that the models dealing with
visible modality data and infrared modality data can either
be shared or distinct.

token and selects the associated prompt and classifier for effective
task processing.
The main contributions of our prompt learning-based PP-IPG
framework for LVI-ReID are summarised as follows:
• We introduce a prompt learning-based approach to tackle
the LVI-ReID problem. A Prompt Pool (PP) module is designed to enable flexible retrieval of grouped prompts for
diverse tasks or domains. This module dynamically routes
task-specific prompts through parameter-isolated groups,
effectively balancing cross-task knowledge sharing while
preserving domain-specific discriminative features.
• An Instance-Level Prompt Generator (IPG) is proposed to
overcome the fixed size of proposed pool constraints by
decomposing input token semantics and dynamically generating instance-level prompts. It Integrates task identifierguided domain encoding to achieve fine-grained prompt
customization at the instance level. Furthermore, we design
a query-key mechanism using query tokens to adaptively
select the most suitable prompts and classifiers during the
LVI-ReID inference phase.
• Extensive experiments demonstrate that our PP-IPG approach
outperforms state-of-the-art lifelong learning methods, LReID
methods, and existing LVI-ReID solutions on the task of LVIReID. To the best of our knowledge, our work is the first
attempt to integrate prompt learning with lifelong learning in the context of visible-infrared ReID, addressing the
practical challenges of real-world ReID tasks.

2

RELATED WORK

Lifelong Learning. Lifelong learning sequentially delivers multitask data to algorithms for dynamic knowledge acquisition under
the constraint of accessing only current task data, aiming to learn
a globally effective model while addressing challenges of evolving
data distributions and catastrophic forgetting. Existing approaches

954

Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator

ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA

Figure 2: Overview of Our Proposed LVI-ReID Method. Initially, the input image (visible or infrared) is processed into image
patches with positional embeddings, which are then fed into the IPG (Instance-specific Prompt Generator) module and the
pre-trained model. The class token output by the pre-trained model retrieves highly relevant prompts from the prompt pool,
while the image tokens are concatenated with an additional query token and passed to the query attention block to compute a
task identifier, which guides the IPG module in generating instance-level prompts. Subsequently, the instance-level prompts,
selected prompts, class token, and image tokens are concatenated and fed into the self-attention module. Notably, each layer of
the attention module is independently equipped with an IPG module to compute distinct instance-level prompts.
are categorized into distillation-based methods [7, 11, 22, 25, 34, 46,
59], regularization-based methods [15, 18, 19, 55], and structurebased methods [8, 14, 23, 33, 35, 49, 54]. Distillation-based methods preserve cross-task performance via knowledge distillation:
iCaRL [34] combines limited historical data replay with current
task training, while LwF [25] generates pseudo-labels from old
models for storage-free optimization; Regularization-based methods protect critical knowledge by constraining parameter updates,
such as EWC [18] slowing learning rates for task-critical parameters and Lee et al. [19] fusing old and new parameters through
Bayesian integration; Structure-based methods explicitly construct
task-specific parameters, exemplified by DyTox [8] dynamically
expanding task-specific tokens with a shared backbone for crosstask feature generalization. Compared to task interference risks in
shared parameter spaces, structure-based methods achieve superior
knowledge isolation through parameter decoupling.
Visible-Infrared Person Re-identification (ReID). Person
re-identification is a task that aims to ascertain whether images
of individuals captured from different camera viewpoints or at

different times by the same camera correspond to the same individual. Early neural networks, primarily designed for classification tasks, naturally extracted global features from an input image when applied to ReID tasks [62]. Traditional ReID methods
were insufficient for addressing the complexities of real-world
scenarios. Based on deep neural networks, numerous methods
[3, 28, 36, 44, 45, 56, 61, 63] have incorporated advanced feature
learning strategies, including local features (e.g., pedestrian body
parts or simple vertical region partitioning) and viewpoint information. These methods are tailored specifically for visible-spectrum
image retrieval tasks, making them effective only in daytime scenarios and limiting their applicability in nighttime conditions. Thus,
the Visible-Infrared Person Re-identification (VI-ReID) methods
have been proposed. Wu et al. [43] designed a deep zero-padding
framework to adaptively learn modality-sharable features for VIReID. Typical methods[9, 12, 50, 51, 64] adopt a two-stream network,
where each modality has its own parameters in the shallow layer to
extract modality-variant features and shares parameters in the deep
layer to extract modality-invariant features, thus resolving the differences within and across modalities. Recently, VI-ReID methods

955


exec
/bin/zsh -lc "pdftotext -l 3 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Local-Aware Residual Attention Vision Transformer for
Visible-Infrared Person Re-Identification
XUECHENG HUA, KE CHENG, and GEGE ZHU, Jiangsu University of Science and
Technology, Zhenjiang, China
HU LU, Jiangsu University, Zhenjiang, China
YUANQUAN WANG, Hebei University of Technology, Tianjin, China
SHITONG WANG, Wenzhou-Kean University, Wenzhou, China and Jiangnan University, Wuxi, China
Visible-infrared person re-identification (VI-ReID) task is to retrieve the same pedestrian across the visible
and infrared modalities. The existing transformer-based works are constrained by the inherent structure of
the ViT that feature collapse in deeper layers and the over-globalization of extracted features, resulting in
incomplete learning of local and low-level features. However, these features are instrumental in representing
and identifying elements within visible-infrared images more comprehensively, which increases the accuracy
and robustness of cross-modal pedestrian matching. To solve the above problem, we propose the LocalAware Residual Attention Vision Transformer (LAReViT) to enhance the learning of fine-grained local and
shallow-level information to reinforce the feature discrimination and comprehensiveness in ViT. Specifically,
the Local-Aware Residual (LAR) Module, which uses a novel Local Residual Attention (LRA) mechanism, is
proposed to increase the fine-grained local information contained in feature extraction. In order to exploit
fine-grained local information lost in lower-level visual features, the LRA in the LAR module adopts novel
attention residual connections. Additionally, we propose a Positional Channel Reconstruction (PCR) Module
that takes advantage of the local receptive field benefits of convolution. PCR reweights features within patches
at the channel level, further facilitating the network emphasis on effective fine-grained local information.
Finally, the novel Center Aggregation Loss (CAL) is designed to reduce modality discrepancies moderately
and promote comprehensive feature extraction. Extensive experiments conducted on the SYSU-MM01, RegDB,
and LLCM datasets demonstrate the state-of-the-art performance achieved by our proposed method. The code
is available at https://github.com/Hua-XC/LAReViT.
CCS Concepts: • Computing methodologies → Object identification;
Additional Key Words and Phrases: Visible-infrared person re-identification, person re-identification, vision
transformer, residual attention
Xuecheng Hua and Ke Cheng contributed equally to this research.
This work was supported in part by the National Science Foundation Program of China (NSFC) (grant number: 61976241),
the Postgraduate Research and Practice Innovation Program of Jiangsu Province (grant number: KYCX24_4129), and the
International Science and technology cooperation plan project of Zhenjiang (grant number: GJ2021008).
Authors’ Contact Information: Xuecheng Hua, Jiangsu University of Science and Technology, Zhenjiang, China;
e-mail: huaxuecheng8888@gmail.com; Ke Cheng, Jiangsu University of Science and Technology, Zhenjiang, China;
e-mail: chengke1972@just.edu.cn; Gege Zhu, Jiangsu University of Science and Technology, Zhenjiang, China; e-mail:
zhugege2677@gmail.com; Hu Lu (corresponding author), Jiangsu University, Zhenjiang, China; e-mail: luhu@ujs.edu.cn;
Yuanquan Wang, Hebei University of Technology, Tianjin, China; e-mail: wangyuanquan@scse.hebut.edu.cn; Shitong
Wang, Wenzhou-Kean University, Wenzhou, China and Jiangnan University, Wuxi, China; e-mail: wxwangst@aliyun.com.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/5-ART146
https://doi.org/10.1145/3723358
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.

146:2

X. Hua et al.

ACM Reference format:
Xuecheng Hua, Ke Cheng, Gege Zhu, Hu Lu, Yuanquan Wang, and Shitong Wang. 2025. Local-Aware Residual
Attention Vision Transformer for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput.
Commun. Appl. 21, 5, Article 146 (May 2025), 24 pages.
https://doi.org/10.1145/3723358

1

Introduction

Person Re-Identification (ReID) and Visible-Infrared Person Re-Identification (VI-ReID)
are crucial techniques aimed at retrieving and identifying the same person from images captured
by non-overlapping cameras at different times. With the increasing emphasis on public safety,
all-weather video surveillance and retrieval systems have garnered significant attention within the
domain of computer vision. Existing ReID methods primarily focus on RGB image datasets captured
under well-lit conditions, while numerous surveillance contexts require the utilization of Infrared
(IR) cameras to capture scenes within low-light environments. Therefore, the task performed by
ReID involves the retrieval between RGB-RGB images, whereas the task undertaken by VI-ReID
focuses on the retrieval between RGB-IR images due to significant differences in color and texture
across images captured in various spectra. For instance, shorter wavelengths of IR images tend to
lose the color and texture information present in RGB images. In ReID, the focus is primarily on
extracting and discriminating features related to the identity of the person. In contrast, VI-ReID
not only needs to address the difficulty of extracting person identity information but also must
consider the misalignment of person identity information contained in images of visible and IR
modalities and the resulting difficulties in feature extraction. Additionally, the VI-ReID dataset is
significantly smaller than traditional ReID datasets, and there is a disparity in the number of visible
and IR images within the VI-ReID dataset. This presents challenges for the model generalization
ability and the balance of modality information. Summarily, the inadequacy of models designed for
the visible light modality in performing image retrieval under low-light conditions, it is typically
necessary to develop more advanced models for VI-ReID to address the differences in cross-modal
features and reduce modality discrepancies.
Existing outstanding works proposed in VI-ReID mainly concentrated on convolutional neural
networks (CNNs) [20, 61, 64]. However, as shown in Figure 1(c), due to the inherent local receptive
field properties of CNNs, the network emphasizes certain local features of images while failing to
recognize long-range dependencies, resulting in the loss of some global features. To address this
limitation, transformers have been widely adopted in computer vision. This attention mechanism
can model extensive dependencies between tokens in data sequences, thereby enhancing the
ability of the network to capture global features. However, as the token proceeds deeper into
the network, the attention maps become increasingly globalized. Vision Transformer (ViT)
aggregates features between patches at a global scale, resulting in lost focus on local information
and capturing extra irrelevant information [36], as shown in Figure 1(b). Therefore, one of our aims
is to enable the network to learn comprehensive global features, while emphasizing certain effective
discriminative local features. In addition, the self-attention mechanism in ViT is characterized by
feature collapse in deeper layers leading to the vanishing of low-level visual features [46]. To provide
evidence for the phenomenon of feature collapse that occurs as the network deepens in VI-ReID,
Figure 2 presents a visual representation of the feature similarity matrices for ViT and LocalAware Residual Attention Vision Transformer (LAReViT) across different modalities. The
matrices were constructed by calculating the cosine similarity between distinct feature patches
extracted by models. We observe that as the features progress from the shallower to the deeper
layers of the model, the similarity between distinct patches gradually increases. The increased
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.

Local-Aware Residual Attention Vision Transformer for VI-ReID

146:3

Fig. 1. Motivation: Analyze feature maps learned from GradCAM [42] algorithm. CNN-based networks often
emphasize local information, which results in the loss of global features. ViT is devoted to capturing global
features but faces the challenge of over-globalization, leading to the redundancy of irrelevant information
such as redundant background. Although global features are crucial, it is equally important to emphasize
discriminative local features, which assist in guiding the network to express comprehensive discriminative
features. Therefore, exploring valuable local information on transformer is crucial, and our proposed LAReViT
precisely captures effective cross-modal person feature. CNN, convolutional neural network; LAReViT, LocalAware Residual Attention Vision Transformer; ViT, vision transformer.

similarity among distinct patches means the loss of feature diversity, resulting in feature collapsing.
The proposed LAReViT, through an innovative residual connection between attention layers,
effectively alleviated this phenomenon. The excessive similarity among patches is caused by
feature collapse, resulting in the loss of low-level fine-grained feature information [6, 23, 75]. In
VI-ReID, low-level fine-grained cues typically represent details such as color and texture [71].
Significantly, these low-level cues, typically found in shallower network layers, are crucial for
cross-modal matching. Therefore, another of our aims is to optimal the utilization of those shallow
low-level cues.
In this article, we leverage the strengths of transformer in modeling global information, while
enhancing the advantages of convolution operations in patch embedding regarding the local
receptive field. Additionally, we thoroughly consider the utilization of shallow-layer low-level
features. Consequently, the LAReViT is proposed to capture and preserve local low-level features,
addressing the issue of transformer features over-globalization and collapse in VI-ReID. This
approach aims to bridge the modal gap within a semantically richer high-level feature space,
thereby enhancing the comprehensiveness and discriminativeness of person features across the
visible and IR modals. Different from the existing pure transformer-based VI-ReID method, we
designed a Local-Aware Residual (LAR) Module with a Local Residual Attention (LRA)
mechanism therein to enhance the transmission and accumulation of attention-related information
from shallow to deep layers. LAR allows the network to provide more low-level detailed information
about local elements of pedestrians and capture and retain crucial low-level pedestrian features
across different modalities. Through utilizing those detailed crucial elements, we guide the model in
learning more comprehensive feature representations. Moreover, we addressed the issue of the overglobalization of transformer features by leveraging fine-grained local information from different
parts of the human body, which enables transformer to extract local features more effectively. Finally,
taking into account the advantages of convolution kernel in local receptive fields, we innovatively
proposed a Positional Channel Reconstruction (PCR) Module that reweights each patch block
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

5821

Mask-Aware Hierarchical Aggregation Transformer
for Occluded Person Re-Identification
Guoqing Zhang , Member, IEEE, Yan Yang, Yuhui Zheng , Member, IEEE, Gaven Martin, and Ruili Wang

Abstract— Occluded person re-identification (Re-ID) is a challenging problem due to the absence of notable discriminative
features resulting from incomplete body part images and interference from occluded regions. Recently, some transformer-based
methods have demonstrated excellent capabilities in resolving this
problem, however these methods are not able to precisely focus
on the non-occluded body parts and cannot capture fine-grained
local features. To achieve these we propose a Mask-Aware
Hierarchical Aggregation TrAnsforMer (MAHATMA) method
to enhance occluded person Re-ID. Specifically, we propose a
Mask Information Embedding (MIE) module, which directs the
model to focus on non-occluded body parts by incorporating
the mask semantic information of a human body. Furthermore,
to effectively capture fine-grained local features, we propose
a Hierarchical Feature Aggregation (HFA) module that mines
more exploitable high-quality detail information by aggregating
hierarchical image patch representations. To further alleviate the
feature loss problem, we propose a Diverse Feature Completion
(DFC) module, which is able to complete global features through
multi-path feature integration. Extensive experimental evaluations demonstrate that our method exhibits superior performance
in dealing with occluded and holistic person datasets.
Index Terms— Occluded person re-identification, vision transformer, feature aggregation, feature completion.

I. I NTRODUCTION
ERSON Re-Identification (Re-ID) aims to locate targeted
individuals across multiple non-overlapping camera perspectives and is a critical topic in the realm of computer vision

P

Received 29 May 2024; revised 18 October 2024; accepted 8 January
2025. Date of publication 17 January 2025; date of current version 6 June
2025. This work was supported in part by the National Natural Science
Foundation of China under Grant 62172231, Grant 92470202, and Grant
U22B2056, in part by the Natural Science Foundation of Jiangsu Province
under Grant BK20220107, in part by the Preliminary Research Project on
Leading Technologies by Wuxi Industrial Innovation Research Institute-Visual
Intelligent Analysis of Worker Behavior and Anomaly Warning, Wenzhou
Key Scientific and Technological Projects under Grant ZG2024012, and in
part by the 2020 Catalyst: Strategic – New Zealand (NZ)-Singapore Data
Science Research Programme (funded by Ministry of Business, Innovation
and Employment (MBIE), NZ). This article was recommended by Associate
Editor A. Chetouani. (Corresponding author: Ruili Wang.)
Guoqing Zhang is with the School of Computer Science, Nanjing University
of Information Science and Technology, Nanjing 210044, China, and also with
the School of Mathematical and Computational Sciences, Massey University,
Auckland 0632, New Zealand (e-mail: guoqingzhang@nuist.edu.cn).
Yan Yang and Yuhui Zheng are with the School of Computer Science,
Nanjing University of Information Science and Technology, Nanjing 210044,
China (e-mail: yangyan@nuist.edu.cn; zheng_yuhui@nuist.edu.cn).
Gaven Martin is with the Institute for Advanced Study, Massey University,
Auckland 0632, New Zealand (e-mail: G.J.Martin@massey.ac.nz).
Ruili Wang is with the School of Mathematical and Computational Sciences,
Massey University, Auckland 0632, New Zealand, also with the School
of Computer Science, University of Nottingham Ningbo China, Ningbo
315104, China, and also with the School of Data Science and Artificial
Intelligence, Wenzhou University of Technology, Wenzhou, China (e-mail:
ruili.wang@massey.ac.nz).
Digital Object Identifier 10.1109/TCSVT.2025.3531142

with wide ranging application in video surveillance, including
intelligent security and smart city initiatives. Over the past
few years, a wide array of solutions have been proposed [1],
[2], [3], [4], leading to substantial progress in holistic person
Re-ID. Nevertheless, in practical situations, as illustrated in
Fig. 1(a), individuals are frequently prone to occlusion by
obstacles (e.g., pedestrians, cars, trees, roadblocks), making
the accurate matching of individuals with incomplete and
obscured body parts a challenging task. Consequently, the
research on occluded person Re-ID holds substantial practical
significance.
Recently, benefiting from the multi-head self-attention
(MSA) mechanism that drives a model to focus on different
body parts and captures long-range dependencies, transformerbased methods [5], [6], [7], [8], [9] have shown promising
advancements in the occluded person Re-ID task. However,
these methods still have following limitations: (i) Due to the
diversity of occlusion types, existing methods cannot precisely
focus on non-occluded body parts [7], [10]. As shown in
Fig 1(b), MSA can guide model (TransReID) to focus on a
discriminative part, but it may also introduce background and
occlusion information in the feature embedding (as indicated
by the red box). To address this issue, some studies [11], [12],
[13] have utilized additional semantic information to enhance
feature robustness and achieved some effective results, but
these methods still fail to completely avoid the limitation that
MSA is highly sensitive to background and occlusion information, which in turn limits the ability of accurate positioning.
(ii) Current methods fail to effectively capture fine-grained
features in images [9], [14], [15]. The main reason is that
most of these methods rely on MSA to assist the model in
capturing global dependencies, while ignoring fine modeling
of details between pixels in different patches, resulting in the
loss of fine-grained features and contextual information.
We design an architecture tailored specifically for the
occluded person Re-ID task, which can not only precisely focus on non-occluded body parts, but also extract
fine-grained feature information for pedestrian identity matching. Fig 2 illustrates our proposed framework, which we
call Mask-Aware Hierarchical Aggregation TrAnsforMer
(MAHATMA). Specifically, MAHATMA consists of a mask
extractor, a Mask Information Embeddings (MIE) module,
a Hierarchical Feature Aggregation (HFA) module, and a
Diverse Feature Completion (DFC) module. Firstly, we direct
the model to specifically focus on non-occluded body parts in
occlusion scenarios by designing the MIE module to encode
the mask semantic information of body parts obtained through
a mask extractor, mitigating the impact of occlusion on the

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.

5822

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

Fig. 1. Visualization of images with different occlusion types. The first, second and third columns represent the occluded input images, attention heatmaps
of TransReID and MAHATMA, respectively. Compared to TransReID, the
proposed MAHATMA can effectively mitigate the impact of occlusions and
focus on non-occluded body parts, as indicated by the red box, thus extracting
discriminative feature representations.

precise positioning of the model. Secondly, our proposed
HFA module is to mine more exploitable high-quality detail
information by aggregating hierarchical image patch representations to capture fine-grained local features. To further
alleviate the problem of feature loss under occlusion conditions and enhance the model’s global perception capability,
we propose the DFC module that is able to compensate
for the possible loss of global features through a multi-path
feature integration approach. We perform experiments on
diverse datasets, encompassing both holistic (Market-1501,
DukeMTMC-reID, MSMT17) and occluded (Occlued-Duke,
Occlued-REID) scenarios. A wealth of experimental results
corroborate the validity of our approach.
The new contributions in this paper are summarised as
follows:
• A novel framework for occluded person Re-ID, called
Mask-Aware Hierarchical Aggregation TrAnsforMer
(MAHATMA), is designed to tackle the challenge of
accurately focusing on non-occluded body parts and
capturing distinctive fine-grained features.
• We propose a mask information embeddings (MIE) module that encodes mask semantic information of body parts
through learnable embeddings, which is demonstrated to
effectively alleviate the impact of occlusion and improve
the model’s localization of non-occluded body parts.
• We propose a hierarchical feature aggregation (HFA)
module, which can capture more fine-grained local features by aggregating hierarchical image patch representations, thus enabling the extraction of more exploitable
high-quality detail information.
• A diverse feature completion (DFC) module that can
integrate multi-path features is proposed to complete for
possible global feature loss.
II. R ELATED W ORK
A. Visual Transformer
Transformer [16] is a revolutionary neural network architecture, initially introduced in the study of Natural Language

Processing (NLP), and has become a fundamental tool for
many NLP tasks. Inspired by the self-attention mechanism,
transformer has recently been transplanted into diverse computer vision tasks. Vision Transformer has demonstrated
powerful capabilities and enormous potential in handling
sequential data. For example, Vision Transformer (ViT) [17]
applied the original transformer architecture to the image
classification task, dividing images into different patches
that serve as tokens similar to words in NLP. Swin Transformer [18] designed a hierarchical transformer with shifting
windows, providing flexibility for images of different scales
while retaining linear computational complexity concerning
image dimensions. Carion et al. [19] performed cross-attention
between object queries and feature maps, converting object
detection into a one-to-one correspondence challenge, thereby
eliminating the necessity for manually engineered modules in
object detection. Han et al. [20] presented a survey of the
transformer applications in computer vision, showcasing its
advancements in this field.
B. Occluded Person Re-Identification
The pioneering work on occluded person Re-ID was initiated by [21]. Due to the absence of discriminative features
resulting from invisible or incomplete body parts and interference from occluded regions, occluded person Re-ID presents
a greater challenge in contrast to holistic person Re-ID.
Early CNN-based methods typically fall into three categories:
(i) methods based hand-craft splitting [22], [23], [24], (ii)
methods based on attention [25], [26], [27], and (iii) methods
based additional clues [28], [29], [30]. These methods have
achieved good results. However, due to the limited receptive
field and single-head self-attention mechanism of CNN, they
fail to capture integral contextual information from the image,
and fine-grained features can be easily lost due to the presence
of extensive occlusion noise.
Recently, transformer-based methods have made advancements in person Re-ID, and two types of methods have been
proposed for these occlusion scenarios. One group of methods
is based around additional semantic information, and the other
groups of methods are based on multi-scale feature aggregation. Some methods have leveraged additional semantic
information to strengthen the robustness of feature representation. Hou et al. [31] presented a Spatial and Temporal
Region Feature Completion (RFC) module which jointly captured the distant spatial contextual information and long-term
temporal contextual information to restore occluded regions.
Xu et al. [11] proposed a Feature Recovery Transformer
(FRT) that employs person information from k-nearest neighbour features for occlusion feature recovery. Wang et al. [6]
designed a transformer-based Pose-guided Feature Disentangling (PFD) method, which effectively disentangles semantic
components by leveraging pose information, and then discriminately matches non-occluded parts accordingly. Despite
the promising results achieved by these methods, they still
cannot avoid the issue of MSA being susceptible to noise
interference, leading to the model’s inability to precisely locate
non-occluded body parts of a person. Recently, some methods
for aggregating different feature representations have emerged.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: MAHATMA FOR OCCLUDED PERSON RE-IDENTIFICATION

Zhang et al. [32] introduced a Hierarchical Aggregation
Transformers (HAT) framework, which proposed a feature calibration module based on transformer to integrate multi-scale
features by leveraging global perspectives and enhancing local
information. Fan et al. [9] proposed a novel Skip Connection Aggregated Transformer (SCAT) network, incorporating
elements from different layers to compose pedestrian feature
representations. Tan et al. [33] designed a Multi-level Feature
Aggregated Transformer (MFAT) network, which constructs
an aggregation framework from global and local perspectives
to obtain more comprehensive and discriminative attention
regions. However, these multi-scale feature aggregation methods do not effectively capture significant variations in pattern
information among pixels within a patch and cannot effectively
model similar pattern information between pixels in different
patches when processing patch feature representations in the
transformer, resulting in the loss of many fine-grained features
and context information.
In contrast to the methods mentioned above, our proposed MAHATMA can not only realizes the perception of
non-occluded body parts in a variety of occlusion scenarios,
but also has the ability to capture more fine-grained information. This advantage improves the feature robustness of the
model under occlusion conditions and significantly enhances
the matching accuracy.
III. P ROPOSED M ETHOD
A. Overall Framework
Fig 2 presents the framework of our approach, which
employs a pre-trained ViT [17] as the feature extractor to capture feature embeddings from input images. Let x ∈ R H ×W ×C
be an image input, with H, W, C representing its height, width,
and channel dimensions. Initially,
n we partition the oinput image
x into N fixed-size patches x ip |i = 1, 2, . . . , N through a
sliding window approach. The stride and dimension of each
image patch are denoted as S and P. The resultant count of
generated patches N can be depicted as follows:
 


W +S−P
H+S−P
×
,
(1)
N=
S
S
where ⌊.⌋ is the floor function. Since the input to the
transformer encoder must be in the form of a sequence,
a trainable linear transformation function f (.) is applied to
flatten patches, mapping them to D dimensions after which
patch embeddings E p ∈ R N ×D are obtained. A learnable
classification token E g is added before patch embeddings,
with output classification token serving as the global feature
representations for encoder. To retain the positional information, we use learnable position encodings PE . Considering
that feature representations are highly sensitive to camera
variations, we adopt the approach proposed in [7] for acquiring
camera viewpoint information Cid . To solve the problem that
MSA in transformer is susceptible to occlusion interference,
we design a mask information embeddings (MIE) module
to obtain the foreground semantic embeddings E m through
encoding the mask semantic information of body parts, and
then integrate E m and E p to direct the model’s attention

5823

towards non-occluded body parts of the person. In the end,
the sequence forwarded as input to the transformer encoder is
represented as:
E input = {E g ; E p ; E m } + PE + λcm Cid ,

(2)

where PE represents position embeddings, Cid ∈ R (N +1)×D
represents the camera embeddings and remains unchanged for
the same image, λcm is a hyper-parameter used to adjust
the weights of the camera embeddings. Then, the input
embedding E input undergoes processing by L transformer
layers to generate the ultimate feature outputs. To better deal
with occlusion scenes, inspired by [34], the representation
for image patches at the 2nd , 4th , 10th , and 12th stages are
concatenated and sent to the hierarchical feature aggregation
(HFA) module to capture fine-grained local features. To further
strengthen the robustness of feature learning with transformer
framework and improve the model’s global perception ability,
we design a diverse feature completion (DFC) module, which
makes full use of dilated convolutional networks to obtain
discriminating global features by learning visual cues from
different receptive fields. The current methods [9], [12] of
directly using global features for identity recognition often
fail to produce satisfactory results, and so we combine the
above-mentioned global features with local features for person
matching to achieve more precise classification outcomes.
B. Mask Information Embeddings Module
It is known that precise focusing on non-occluded body
parts is a necessary step for occluded person Re-ID. Despite
the remarkable performance achieved by transformer-based
strong baseline in occluded person Re-ID, as indicated by
the visualization of attention maps of TransReID in Fig 1(b),
the MSA mechanism may be susceptible to interference
from background information and occlusion noise, making it
challenging to precisely focus on non-occluded body parts.
Because of this problem, we design a MIE module that incorporates semantic information of body parts into the embedding
representation to mitigate the impact of occlusion interference
on the precise positioning of the model.
Inspired by position and side information embeddings,
which encode both positional and camera information using
learnable embeddings, we introduce the foreground semantic
embeddings to retain the semantic information of body parts.
This design constitutes the core principle of the MIE module,
which significantly mitigates the negative impact of occlusion on feature extraction by integrating foreground semantic
information. Specifically, we first generate the semantic mask
of pedestrians through a mask extractor. Considering the
excellent performance of HRNet [35] in human semantic
parsing, we use it as mask extractor backbone network to
extract pedestrian features, and these features are processed
through a 1 × 1 convolutional layer followed by a softmax
to generate human semantic mask m. In addition, we use the
parsing labels generated by the human body parsing model
PifPaf [36] to supervise m to further optimize its quality.
Subsequently, we follow the methods provided in [37] to learn
five categories of semantic information for body parts and one

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 191 (2025) 107812

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Memory-augmented shuffled meta learning for visible–infrared person
re-identification
Hanxiao Wu a,b , Yutao Chen c , Yi Xie d , Jianqing Zhu c

,∗, Liu Liu e ,∗, Huanqiang Zeng c,f

a

College of Information Science and Engineering, Huaqiao University, Xiamen, 361021, Fujian, China
School of Computer Science and Artificial intelligence, Wuhan University of Technology, Wuhan, 430070, Hubei, China
c
College of Engineering, Huaqiao University, Quanzhou, 362021, Fujian, China
d
School of Future Technology, South China University of Technology, Guangzhou, 511442, Guangdong, China
e
School of Artificial Intelligence and Hangzhou International Innovation Institute, Beihang University, Beihang University, Beijing, 100191, China
f School of Optoelectronic and Communication Engineering, Xiamen University of Technology, Xiamen, 361024, Fujian, China
b

ARTICLE

INFO

Keywords:
Meta learning
Memory-augmentation
Visible–infrared person re-identification
Video surveillance system

ABSTRACT
Visible–infrared person re-identification (VIPR) poses significant challenges due to the inherent differences
between visible and infrared images. These differences result in lower similarity among individuals of the
same identity across modalities and higher similarity among different identities within the same modality.
Existing methods often struggle to effectively address this issue, as they fail to capture global similarity metrics
with limited training data, which hinders the model’s ability to learn discriminative features. To address these
challenges, we introduce a novel approach called memory-augmented shuffled meta (MASM) learning. Our
approach is distinguished by two key components: shuffled meta learning (SML) and memory meta learning
(MML). SML constructs diverse query and support sets in each training cycle, allowing the model to learn
from a wide range of data inputs. Meanwhile, MML leverages historical information stored in memory banks to
capture long-term dependencies. This strategic combination of SML and MML not only enhances data utilization
but also empowers the model to learn comprehensive global meta metrics, significantly improving its ability
to distinguish individuals across modalities. Extensive experiments on the RegDB and SYSU-MM01 datasets
validate the effectiveness of our MASM method, demonstrating its superiority over several state-of-the-art
approaches.

1. Introduction
Recently, person re-identification (Wang, Liu, Zhang et al., 2024;
Ye, Chen, Shen and Shao, 2022; Zhu, Zeng et al., 2020) has gained significant attention due to its immense potential in intelligent transportation systems and intelligent video surveillance systems. Traditional
person re-identification methods primarily rely on visible cameras,
which encounter low identification rates due to bad visible person
image qualities in the environments with inadequate lighting. However,
modern cameras are often equipped with both visible and infrared
modes (Liu, Kuang et al., 2025; Wang et al., 2025; Wang, Liu, Zheng
and Zhang, 2024; Zhang, Wang, Liu, Tu, & Lu, 2024), enabling them
to capture visible images during the day and infrared images under
low light conditions. Thus, visible–infrared person re-identification
(VIPR) (Chen et al., 2022; Ye, Shen et al., 2022; Zheng et al., 2024)
is a natural extension of traditional person re-identification in allday video analysis scenarios. But VIPR is more challenging than the
traditional person re-identification that only handles visible images,

due to significant modal discrepancies between visible and infrared
images.
To address modal discrepancies in VIPR, metric learning is a promising approach, which does not require overly complex networks. Existing metric learning methods in VIPR include sample-based and centerbased approaches. The sample-based approaches, such as the triplet
loss function (Dai, Ji, Wang, Wu, & Huang, 2018; Hermans, Beyer, &
Leibe, 2017; Ye, Shen et al., 2022), pull samples of the same identity
together and push samples with different identities apart, as shown in
Fig. 1(a). Taking into account the modal discrepancies in VIPR, some
methods (Dai et al., 2018; Feng, Xu, Ji, & Wu, 2021; Liu, Ma, Xia, &
Li, 2023) use improved triplet loss functions that specifically restrict
the distance between samples of different modalities. However, most
sample-based methods involve sample mining or weighting techniques,
which are sensitive to outlier samples, as illustrated in Fig. 1(a), causing
biased metrics.

∗ Corresponding authors.

E-mail addresses: jqzhu@hqu.edu.cn (J. Zhu), liuliubh@gmail.com (L. Liu).
https://doi.org/10.1016/j.neunet.2025.107812
Received 23 November 2024; Received in revised form 29 April 2025; Accepted 24 June 2025
Available online 9 July 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 191 (2025) 107812

H. Wu et al.

In this paper, we introduce a memory-augmented shuffled meta
(MASM) framework. As illustrated in Fig. 1(c), the motivation for
adopting a meta-learning perspective lies in the way query-support
set partitioning simulates the retrieval process of gallery libraries
from query images in real-world person re-identification scenarios.
To fully leverage the advantages of the query-support paradigm in
meta-learning, we propose a shuffling strategy that maximizes data
utility by dynamically constructing diverse query-support pairs in
each training cycle. Furthermore, our memory-augmented mechanism
utilizes historical feature representations stored in memory banks,
allowing the model to capture long-term dependencies and learn more
comprehensive cross-modality metrics. The proposed MASM integrates
shuffled meta learning (SML) and memory meta learning (MML).
Specifically, the SML method enhances data utilization by constructing
diverse query and support sets during training, enabling the model to
learn from a wide range of data inputs. SML is implemented through
the designed shuffling strategy. The shuffling strategy illustrated in
Fig. 1(c) operates by cyclically rotating the selection of query and
support sets among the samples of each identity. Specifically, in each
iteration, one sample is designated as the query, while the remaining
samples form the support set. This ensures that every sample is utilized
equally across iterations. The loop structure is intentionally designed
to systematically explore all possible combinations of query-support
pairs. By doing so, it maximizes the diversity of data presented during
training, which is crucial for enhancing the model’s robustness and
generalization capabilities. This process is visually represented in Fig. 1,
where each iteration corresponds to a unique configuration of query
and support sets. Concurrently, the MML method captures long-term
dependencies by leveraging historical data stored in memory banks,
thereby enhancing the model’s ability to retain crucial information
over time and facilitating the learning of comprehensive global meta
metrics. Consequently, the MASM framework not only enhances data
usage but also empowers the model to learn comprehensive global meta
metrics, resulting in improved performance in visible–infrared person
re-identification tasks.
This contribution of this paper can be summarized as follows.

Fig. 1. The illustration of metric learning methods. (a) The sample-based approach
directly constraints the metric between the samples. (b) The center-based approach
emphasizes intra-class compactness. (c) The proposed meta-based approach shuffles to
obtain different queries and constraints the distance between query and corresponding
meta cells.

Center-based approaches (Can, Hong, Wei, & Mang, 2020; Liu, Tan,
& Zhou, 2021; Zhu, Yang et al., 2020) could be more robust to outliers
than sample-based methods through centralization technologies, as illustrated in Fig. 1(b). For example, improved center loss functions (Can
et al., 2020; Zhu, Yang et al., 2020) restrict the distance between
visible and infrared centers of the same identity to eliminate modal
discrepancies. The centralized triplet loss function (Liu et al., 2021)
pulls the distance between the samples and the positive centers and
pushes the distance between the samples and the negative centers.
However, center-based approaches underestimate complex intra-class
variations by simply using a center to represent a class of samples,
which may not be sufficient for learning a metric that is suitable for
handling different modalities.
In addition, both sample-based and center-based methods only perform metric learning on a fixed batch size of data in each calculation,
making it difficult to capture more global similarity metrics. The limited dataset further exacerbates the challenge for the model to learn
more discriminative features. To tackle data insufficiency, some works
adopt image augmentation methods (Jia, Zhong, Ye, Liu, & Huang,
2022; Qian & Tang, 2024; Wang, Wang, Zheng, Chuang, & Satoh,
2019) using generative models. For example, Qian and Tang (2024)
designed a GAN-based network to generate cross-modal paired images
that maintain shape and appearance consistency with real images.
Similarly, Jia et al. (2022) employed an encoder–decoder structure to
blend the appearance and posture of different individuals for image
enhancement. However, these methods may introduce artifacts in the
VIPR task, such as generating unrealistic textures in visible images or
incorrect heat distributions in infrared images. Other approaches rely
on feature augmentation techniques (Hua et al., 2025; Josi, Alehdaghi,
Cruz, & Granger, 2025; Qiu et al., 2025; Zhang & Wang, 2023). For
example, Qiu et al. (2025) developed a relationship-reinforced fusion
mechanism that dynamically combines inter-modality global features
with intra-modality global–local features through adaptive weighting.
While these feature enhancement methods mitigate data insufficiency
by obtaining richer composite features, they fundamentally depend on
cross-modal priors to learn optimal fusion weights or to train enhancement models, creating a paradoxical reliance on adequate training data.
Consequently, rather than using image enhancement and feature fusion,
it is crucial to fully utilize the limited available data to learn global
similarity measures, as this enhances the model’s ability to distinguish
individuals across modalities.

• We propose the shuffled meta learning (SML) method to enhance
data utilization, thereby increasing the model training intensity.
• We propose the memory meta learning (MML) method leverages
historical data to enhance the model’s ability to learn global meta
metrics.
• Extensive experiments on two public datasets, RegDB (Nguyen,
Hong, Kim, & Park, 2017) and SYSU-MM01 (Wu, Zheng, Yu,
Gong, & Lai, 2017), demonstrate that our approach outperforms
several state-of-the-art VIPR methods.
The remainder of this paper is organized as follows. Section 2
surveys recent work related to our paper. Section 3 describes our
method in detail. Section 4 presents experimental results and analysis to
show the superiority of our method. Section 5 discusses the limitations
and future work. Section 6 concludes this paper.
2. Related work
2.1. Metric learning
In visible–infrared person re-identification task, sample-based metric learning methods (Hu, Liu, Zeng, & Hu, 2022; Seokeon, Lee, Kim,
& Kim, 2020; Wang, Zhang, Yang et al., 2020; Ye, Chen et al., 2022;
Ye, Shen, J. Crandall, Shao, & Luo, 2020) directly apply triplet losses
(Hermans et al., 2017; Ye, Shen et al., 2022) that were originally
designed for single-modal person re-identification, which do not fully
meet the requirements of cross-modal retrieval. Specifically, when it
comes to pushing close positive pairs, most optimizations focus on the
processing of intra-modal positive pairs because intra-modal pairs are
2

Neural Networks 191 (2025) 107812

H. Wu et al.

usually closer than cross-modal pairs, as demonstrated in Zhu et al.
(2023). Some methods (Alehdaghi, Josi, Cruz, & Granger, 2022; Li,
Qi, Chen, & Zhou, 2021; Ye, Wang, Lan, & Yuen, 2018; Zhang, Zhang
et al., 2022; Zhao, Lin, Xuan, & Xi, 2019) limit the distance between
samples in different modalities, which allows a more specific processing
of cross-modal samples. For example, Zhao et al. (2019) designed a
hard pentaplet loss that pulls the furthest cross-modal positive pairs
and pushes the nearest cross-modal negative pairs. However, in the
realm of sample-based methods, it is important to note that many of
them employ hard sample mining (i.e., optimizing the most challenge
negative and positive pairs) or reweighting (i.e., assigning greater
weights to challenging negative and positive pairs) techniques that
can be quite sensitive to outlier samples. This sensitivity could result
in the generation of biased metrics, which in turn leads to a weak
discrimination.
Center-based method (Cai, Zhu, & Zhang, 2021; Feng, Wu, & Zheng,
2023; Hao, Zhao, Ye, & Shen, 2021; Liu et al., 2021; Wei, Yang, Wang,
& Gao, 2021; Zhang, Kang, Zhao and Shen, 2023; Zhang, Yan, Li and
Wang, 2023; Zhu et al., 2024; Zhu, Yang et al., 2020) is another type
of metric learning for VIPR, which would be more robust to outliers via
centralization. For instance, Zhu, Yang et al. (2020) designed a heterocenter loss that obtains a visible feature center and an infrared feature
center for each class and constrains the intra-class center distance
between two different modalities. Can et al. (2020) designed a marginal
exponential center (MeCen) loss to reduce acceptable variances among
easy examples and imposed strong exponential constraints on hard
positive examples. Similarly, Liu et al. (2021) calculated the average
of visible features and the average of infrared features in each class as
centers and closed positive centers as well as pushed negative centers in
different modalities. In addition to constraining clustered centers, Liu,
Sun et al. (2022) sought to learn the visible and infrared proxies in each
identity as parameterized centers and restrict the distances between
cross-modal proxies and samples. However, center-based approaches
tend to overlook the intricate and diverse variations that exist within
a class of samples because they rely solely on a single center point to
represent the entire class, which does not fully capture the complexity
and nuances present within each class. Therefore, center-based methods
may not be sufficient for learning metrics that can be generalized
across different modalities. Compared with center-based methods, the
proposed shuffled meta learning (SML) method is designed to capture
complex intra-class variations effectively. By decomposing the training
task into multiple subtasks and randomly shuffling samples within each
subtask, SML constructs diverse query and support sets during each
training cycle. This approach enhances data utilization and allows the
model to learn from a broader range of intra-class variations, addressing some limitations of center-based methods that may overlook these
complexities. Additionally, by leveraging historical information stored
in memory banks, SML is expected to improve the model’s ability to
recognize and differentiate between subtle variations within the same
class, which could enhance overall performance in visible–infrared
person re-identification tasks

results for general visual recognition tasks under the few-shot limitation
that requires a model of good generalization ability. In addition, there
are model based meta learning (Munkhdalai & Yu, 2017; Santoro,
Bartunov, Botvinick, Wierstra, & Lillicrap, 2016) and optimization
based meta learning (Finn, Abbeel, & Levine, 2017; Nichol, Achiam,
& Schulman, 2018). The classical model based meta learning method,
namely, meta network (Munkhdalai & Yu, 2017), contains a meta
learner for fast weight generation by operating across tasks and a base
learner to perform within each task by capturing the task objective. The
well-known optimization based meta learning is the model-agnostic
meta-learning (MAML) (Finn et al., 2017) method, which learns good
initial parameters of the learner for a fast adaptation. Meta-learning
is a promising way to train deep networks in scenarios requiring good
generalization ability, using few-shot samples, and fast adaption by few
training steps.
There are also some meta-based methods to handle re-identification
tasks. For example, Ni et al. (2022) designed an optimization-based
meta-learning strategy to simulate real train–test domain shifts. Zhang,
Liu, Zhang and Zhang (2023) developed a self-paced meta-learning
method that extends conventional one-stage meta-learning to a multistage training process, simulating human learning. These methods involve alternating meta-train and meta-test processes: the model is
trained on source data during meta-training and validated on unseen
data during meta-testing. In contrast, our method directly optimizes a
metric space for VIPR by introducing a shuffling strategy that dynamically constructs query-support pairs while leveraging memory banks
to preserve long-term feature relationships. Our mate-based method
eliminates the need for complex alternating training phases while
establishing a more effective metric space for cross-modality retrieval.
2.3. Memory bank designs
Memory-based designing is an effective method to learn useful
knowledge from historical information in metric learning.
Wang, Zhang, Huang and Scott (2020) proposed a cross-batch memory
(XBM) mechanism to collect sufficient hard negative pairs across multiple mini-batches, acquiring a comprehensive use of past features. Jiang
et al. (2022) constructed a memory dictionary to take advantage of the
meta learning validation set, which provides rich information in the
training stage. Zhao et al. (2021) built memories to store the center of
each identity in different domains and optimized the distances between
features and memorized centers for domain generalization. Liu, Sun
et al. (2022) stored proxies in memories after every iteration and
optimized cross-modal proxies for VIPR. Inspired by the effectiveness
of memory-based designs, we enhance our approach by integrating
memory banks and meta learning techniques. This integration allows
us to learn a global meta-metric that can effectively capture long-term
dependencies in the data, leading to improve the discrimination of the
learned metrics.
In addition to memory mechanisms, existing works introduce temporal modeling to handle challenges in person re-identification. For
example, MITML (Lin et al., 2022) employs long short-term memory (LSTM) layers to learn temporal representations from fixed-length
frame sequences to capture modal-invariant features over time. SSN3D
(Jiang et al., 2021) extends this by combining part-specific 3D convolutions with LSTM-based modeling to better align spatial parts across
frames, mitigating temporal misalignments. Similarly, STMN (Eom,
Lee, Lee, & Ham, 2021) integrates LSTM units to encode temporal
context for a dedicated temporal memory module, which aggregates
frame-level features through learned temporal attention patterns. However, one inherent limitation of recurrent units such as LSTMs is their
sequential processing nature, which can challenge the effective capture
of very long-range dependencies across training episodes. In contrast,
our MML method provides a more flexible and robust mechanism for
retaining critical information over extended temporal horizons, thereby
enhancing the extraction of cross-modality invariant features even in
the presence of temporal discontinuities.

2.2. Meta learning
Meta learning, also known as learning to learn, is a learning
paradigm that focuses on training models to acquire the ability to
learn new tasks quickly and effectively. This approach enables models
to generalize well to unseen tasks or domains, thereby exhibiting
strong generalization capabilities. There are some meta learning methods (Chen, Zhang, Lu, & Zhou, 2019; Vinyals, Blundell, Lillicrap,
Kavukcuoglu, & Wierstra, 2016; Zhou, Zhang, & Wei, 2022) designed
based on metric learning. For example, Vinyals et al. (2016) explicitly
learned to learn from a given support set to minimize loss over a batch
during the training process. Later, Chen et al. (2019) improved the setbased distance of meta learning by incorporating a hard sample mining
strategy to accelerate the learning process. These works show good
3


codex
前四篇已经读完，形态很清楚：一个是把固定 VI-ReID 改成终身跨域跨模态问题，两个是 Transformer 局部细节或遮挡掩码增强，一个是把 VI-ReID 的度量学习改成带记忆的元学习。现在读剩余几篇，包括重复的 Mix-Modality 两个版本。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Mix-Modality Person Re-Identification: A New and
Practical Paradigm
WEI LIU, School of Computer Science and Technology, Wuhan University of Science and Technology,
Wuhan, China
XIN XU, Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial
System, Wuhan, China and School of Computer Science and Technology, Wuhan University of Science and
Technology, Wuhan, China
HUA CHANG and XIN YUAN, School of Computer Science and Technology, Wuhan University
of Science and Technology, Wuhan, China
ZHENG WANG, National Engineering Research Center for Multimedia Software, Institute of Artificial
Intelligence, School of Computer Science, Wuhan University, Wuhan, China
Current visible-infrared cross-modality person re-identification research has only focused on exploring the
bi-modality mutual retrieval paradigm, and we propose a new and more practical mix-modality retrieval
paradigm. Existing Visible-Infrared Person Re-Identification (VI-ReID) methods have achieved some
results in the bi-modality mutual retrieval paradigm by learning the correspondence between visible and
infrared modalities. However, significant performance degradation occurs due to the modality confusion
problem when these methods are applied to the new mix-modality paradigm. Therefore, this article proposes a
Mix-Modality Person Re-Identification (MM-ReID) task, explores the influence of modality mixing ratio
on performance, and constructs mix-modality test sets for existing datasets according to the new mix-modality
testing paradigm. To solve the modality confusion problem in MM-ReID, we propose a Cross-Identity
Discrimination Harmonization Loss (CIDHL) adjusting the distribution of samples in the hyperspherical
feature space, pulling the centers of samples with the same identity closer, and pushing away the centers of
samples with different identities while aggregating samples with the same modality and the same identity.
Furthermore, we propose a Modality Bridge Similarity Optimization Strategy (MBSOS) to optimize
the cross-modality similarity between the query and queried samples with the help of the similar bridge
This work was supported by the National Nature Science Foundation of China (No. 62376201). This research was financially
supported by funds from Key Laboratory of Social Computing and Cognitive Intelligence (Dalian University of Technology),
Ministry of Education (No. SCCI2024YB02), Hubei Province Key Laboratory of Intelligent Information Processing and
Real-Time Industrial System (Wuhan University of Science and Technology; No. ZNXX2023QNO3), Fund of Hubei Key
Laboratory of Inland Shipping Technology and Innovation (No. NHHY2023004), and Entrepreneurship Fund for Graduate
Students of Wuhan University of Science and Technology (No. JCX2023049).
Authors’ Contact Information: Wei Liu, School of Computer Science and Technology, Wuhan University of Science and
Technology, Wuhan, China; e-mail: liuwei@wust.edu.cn; Xin Xu (corresponding author), Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial System, Wuhan, China and School of Computer
Science and Technology, Wuhan University of Science and Technology, Wuhan, China; e-mail: xuxin@wust.edu.cn; Hua
Chang, School of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan, China; email: changhua@wust.edu.cn; Xin Yuan, School of Computer Science and Technology, Wuhan University of Science and
Technology, Wuhan, China; e-mail: xinyuan@wust.edu.cn; Zheng Wang, National Engineering Research Center for Multimedia Software, Institute of Artificial Intelligence, School of Computer Science, Wuhan University, Wuhan, China; e-mail:
wangzwhu@whu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/3-ART112
https://doi.org/10.1145/3715142
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.

112:2

W. Liu et al.

sample in the gallery. Extensive experiments demonstrate that compared to the original performance of
existing cross-modality methods on MM-ReID, the addition of our CIDHL and MBSOS demonstrates a general
improvement.
CCS Concepts: • Computing methodologies → Artificial intelligence; Computer vision;
Additional Key Words and Phrases: Cross-Modality Person Re-identification, Mix-Modality Paradigm, Metric
learning, Post-processing
ACM Reference format:
Wei Liu, Xin Xu, Hua Chang, Xin Yuan, and Zheng Wang. 2025. Mix-Modality Person Re-Identification: A
New and Practical Paradigm. ACM Trans. Multimedia Comput. Commun. Appl. 21, 4, Article 112 (March 2025),
21 pages.
https://doi.org/10.1145/3715142

1

Introduction

Person Re-Identification (ReID) is a critical task in intelligent video security systems, aiming
to retrieve specific pedestrians across a network of non-overlapping cameras [11, 35, 39]. While
Single-Modality ReID (SM-ReID) methods focusing on visible image retrieval have achieved
significant advancements, they often fall short in low-light conditions where crime rates are notably
higher. These illumination conditions lead to substantial information loss in RGB camera captures.
To overcome this limitation, increasingly sophisticated cameras capable of automatically switching
to infrared mode in low-light conditions are being integrated into video surveillance systems. This
shift has spurred a growing research interest in Visible-Infrared ReID (VI-ReID), which aims to
tackle the challenge of cross-modality image matching [9, 14, 16, 38].
VI-ReID not only grapples with the common challenges of viewpoint, background, pose, and
occlusion typical of SM-ReID task [2, 3, 10, 28, 34, 45, 46] but also faces significant difficulties
arising from modality differences. Despite considerable progress achieved in the bi-modality mutual
retrieval paradigm, as illustrated in Figure 1(a) through learning potential correspondences between
visible and infrared images, practical ReID scenarios pose additional complexities. As shown in
Figure 1(b), pedestrian images in real applications may need to be identified across both day
and night, necessitating a database that integrates both visible and infrared images—not merely
a collection of one modality type. This integration often leads to a mix of what seems to be
straightforward SM-ReID tasks into the existing cross-modality framework. However, as indicated
in Figure 1(c), this approach results in a marked performance degradation, primarily due to the
“Modality Confusion” problem. This issue stems from identity-independent features such as colors
being more similar within the same modality, which confuses the matching of cross-modality
identity information. More specifically, the impact of modality confusion on the current approach
of learning only visible-infrared cross-modality correspondences is huge, due to the fact that there
exists not only one correspondence from visible to infrared in the retrieval process, but also two
same-modality correspondences, which is not taken into account by the existing approaches of
VI-ReID. Under this influence, the natural similarity between samples of the same modality will
disturb the perception of the algorithm, resulting in the distance between samples of different
identities of the same modality being smaller than the distance between samples of the same
identity of different modalities, and ultimately giving wrong retrieval results. To address these
challenges, we introduce a new and practical Mix-Modality Re-ID (MM-ReID) task, creating
mixed modality test sets for existing datasets and examining how the ratio of modality mixing
affects retrieval performance.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.

Mix-Modality Person Re-Identification: A New and Practical Paradigm

112:3

Fig. 1. (a) Existing bi-modality mutual retrieval test paradigms for VI-ReID use a visible probe image queried
in an infrared image gallery or an infrared probe image queried in a visible image gallery. (b) Our proposed
mix-modality testing paradigm for MM-ReID uses mix-modality probes to query in a mixed-modality gallery.
(c) The unique challenge in MM-ReID is the interference of sample points with different identities in the same
modality. In the figure, blue/gray represents visible/infrared modality samples, respectively, and different
shapes represent different identities. It can be seen that due to more similar color and other identity-irrelevant
information, the distance between samples of the same modality but different identities is closer than that
between samples of the same identity with different modalities, which eventually leads to confusion during
retrieval and reduces the accuracy. MM-ReID, mix-modality person re-identification; VI-ReID, visible-infrared
person re-identification.

To address the modality confusion challenge in MM-ReID, we introduce the Cross-Identity
Discrimination Harmonization Loss (CIDHL). This novel approach optimizes the sample
distribution within a hyperspherical feature space. Specifically, CIDHL aggregates distances between
samples of the same identity and modality towards their respective centers, while simultaneously
drawing together centers of samples sharing identities across different modalities and distancing
those of different identities, irrespective of modality. This strategy mitigates the effects of modality
confusion. Additionally, we propose a Modality Bridge Similarity Optimization Strategy
(MBSOS), which optimizes the cross-modality distance metric by identifying a similar bridge
sample in the gallery to serve as an intermediary, thus refining the distance measures between
the query and the queried samples. Extensive testing confirms that our methods—CIDHL and
MBSOS—significantly enhance the model performance of existing cross-modality approaches,
demonstrating their efficacy and adaptability in resolving modality confusion in MM-ReID.
The contributions of this article are three aspects:
— Paradigm Contribution: We pioneer the MM-ReID task, introducing a novel testing paradigm
that incorporates both visible and infrared modalities within a single framework.
— Empirical Contribution: We investigate the impact of varying modality mixing ratios on crossmodality model performance and conclude the unique challenges of “Modality Confusion”
specific to MM-ReID.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 4, Article 112. Publication date: March 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
MoDA: Mixture of Domain Adapters for Parameter-efficient
Generalizable Person Re-identification
YANG WANG, School of Electronic Information, Huzhou College, Huzhou, Zhejiang, China and Software
Engineering Institute, East China Normal University, Shanghai, China
YIXING ZHANG, Software Engineering Institute, East China Normal University, Shanghai, China
XUDIE REN, Informatization Office, Tongji University, Shanghai, China
YUXIN DENG, Software Engineering Institute, East China Normal University, Shanghai, China
The Domain Generalizable Re-identification (DG ReID) task has attracted significant attention in recent
years, as a challenging task but closely aligned with practical applications. Mixture-of-experts (MoE)-based
methods have been studied for DG ReID to exploit the discrepancies and inherent correlations between diverse
domains. However, most of DG ReID methods, especially MoE-based methods, have to fully fine-tune a large
amount of parameters, which are not always practical in real-world scenarios. Considering this problem, we
propose a novel MoE-based DG ReID method, named Mixture of Domain Adapters (MoDA), which utilizes
many expert adapters and a global adapter to help MoE-based method scale to a much larger model but in a
more parameter-efficient way. Furthermore, we conduct our approach with the large-scale vision-language
pre-trained model CLIP, which exploits both visual and text encoders, to learn more robust representations
based on multimodal information. Extensive experiments verify the effectiveness of our method and show that
MoDA achieves competitiveness with state-of-the-art DG ReID methods with much fewer tunable parameters.
CCS Concepts: • Computing methodologies → Computer vision; Visual content-based indexing and
retrieval; Image representations;
Additional Key Words and Phrases: Generalizable Person Re-Identification, Domain Generalization, Parameterefficient Fine-tuning
ACM Reference format:
Yang Wang, Yixing Zhang, Xudie Ren, and Yuxin Deng. 2025. MoDA: Mixture of Domain Adapters for
Parameter-efficient Generalizable Person Re-identification. ACM Trans. Multimedia Comput. Commun. Appl.
21, 5, Article 139 (May 2025), 19 pages.
https://doi.org/10.1145/3712595

1

Introduction

Person Re-identification (ReID) has emerged as a pivotal research area in computer vision. ReID
aims to identify and retrieve individuals of the same identity across non-overlapping cameras.
The work is partially supported by Shanghai Artificial Intelligence Innovation and Development Fund (No. 2020-RGZN02026).
Authors’ Contact Information: Yang Wang, School of Electronic Information, Huzhou College, Huzhou, Zhejiang, China
and Software Engineering Institute, East China Normal University, Shanghai, China; e-mail: wangyang@zjhzu.edu.cn;
Yixing Zhang (corresponding author), Software Engineering Institute, East China Normal University, Shanghai, China;
e-mail: yixingzhang_ecnu@outlook.com; Xudie Ren, Informatization Office, Tongji University, Shanghai, China; e-mail:
23666509@tongji.edu.cn; Yuxin Deng, Software Engineering Institute, East China Normal University, Shanghai, China;
e-mail: yxdeng@sei.ecnu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/5-ART139
https://doi.org/10.1145/3712595
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.

139:2

Y. Wang et al.

There have been many works that can improve great performance on ReID benchmarks in the conventional scenario [8, 29, 32, 48]. However, when these methods are confronted with a completely
unseen domain, the performance drops significantly. This phenomenon is commonly attributed
to domain shift and domain conflict. When ReID comes to Domain Generalization (DG), the
task becomes even more difficult. The model can only use images from source domains to optimize and is not allowed to access any target domain image during the training period. This task
definition is intended to augment the generalization capability of model and enhance the model
robustness when confronted with out-of-distribution data. To tackle this problem, some Domain
Adaptive (DA) and Domain Generalizable ReID (DG ReID) methods are proposed. DA methods
[6, 16, 42] can access a part of target domain data and then try to adapt the model which is already
trained with source domain data to the target domain [36].
Compared to the conventional ReID, the DG ReID task has attracted much attention in recent
years, as a task which is more challenging but more closely aligned with practical applications.
In the context of DG ReID, the training and testing of models are performed on diverse domains,
which are respectively referred to as the source domain and target domain. Many prior DG ReID
methods [28, 44] only utilize one individual model and train the model on a hybrid dataset that
consists of samples from different source domains. And then the model is directly tested on unseen target domains. These methods achieve good performance by extracting domain-invariant
features. However, they disregard the discrepancies and inherent correlations between diverse
domains, which may provide more discriminative and complementary information to help generalize better. For this reason, Mixture-of-experts (MoE)-based methods [2, 36] have been studied
for DG ReID. A common framework of MoE-based methods is to train domain-specific expert
networks on each source domain and then these methods integrate multiple experts by calculating
the relevance of the test sample and source domains to get one aggregated feature. The existing
MoE-based DG ReID methods get better performance but have the common issue that the number
of model parameters scales linearly with the number of source domains due to the increase of the
number of experts as shown in Figure 1. Although there are some methods [36, 39] that try to
minimize the number of expert parameters, there are still a large amount of trainable parameters
need to be updated. All these existing methods have to fully fine-tune the entire model including the backbone and experts. And it is not practical and efficient enough in many real-world
scenarios. Especially under the DG ReID setting where the number of person IDs is particularly
large, the number of parameters of classifier heads significantly increases. These MoE-based methods always need at least one classifier for each source domain and one global classifier to learn
domain-invariant feature and to optimize the model. And it leads to a certain degree of parameter
redundancy.
For the motivation to solve the issues mentioned above, we propose a novel DG ReID method
which is parameter-efficient and can be scaled to a much larger backbone model, named Mixture
of Domain Adapters (MoDA). We apply the large-scale vision-language model CLIP [27] and
Adapter [10] in our method. CLIP is a powerful pretrained model which aligns texts and images
in one feature space. It can provide cross-modality information, which may help to learn robust
representations against the challenge of domain shift. Moreover, the contrastive losses that CLIP
uses can provide similar discriminative guidance of optimization in comparison with cross-entropy
loss, which means that it may substitute the ID loss of ReID to reduce the classifier head parameters.
Some works [30] with the image encoder of CLIP have already shown good performance in other
computer vision DG tasks. In addition, Adapter is a bottleneck module used for Parameterefficient Fine-tuning (PEFT) and will be used as domain-specific experts in MoDA. As Figure 2
shows, different from prior CNN-based DG ReID methods, the adapters (experts) are inserted in
each block of MoDA. Due to this architecture, we propose a Block-aware Voting Network to make
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.

MoDA for Parameter-efficient Generalizable Person Re-ID

139:3

Fig. 1. Difference between prior MoE-based DG ReID methods and our more parameter-efficient method.
Upper: Prior MoE-based DG methods need to optimize the whole model including the backbone and experts,
parameters of which usually increase linearly with the increasing of the number of person IDs. Below: In our
method, the backbone is frozen and only adapters, the parameters of which are not related with the person
IDs, need to be optimized during training.

Fig. 2. We show the architecture of the adapter, ViT block with adapter, and our method MoDA block. The
standard architectures of adapter and ViT block are shown within (a) and (b), respectively. In our proposed
MoDA block, we set Adapters after the multi-head attention layer to implement an MoE architecture as (c)
shows. An additional block-aware voting network is set to mix the adapters. Only the parameters of adapters
and the voting network need to be updated during training.

MoE possible to be used for this kind of model which consists of blocks. It enables the model to
integrate expert adapters in a more fine-grained way by generating aggregation weight for each
block, as shown in Figure 2(d). As be seen, this framework greatly reduces the number of trainable
parameters in the model network.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 139. Publication date: May 2025.


 succeeded in 0ms:
2021 IEEE/CVF International Conference on Computer Vision (ICCV)

2021 IEEE/CVF International Conference on Computer Vision (ICCV) | 978-1-6654-2812-5/21/$31.00 ©2021 IEEE | DOI: 10.1109/ICCV48922.2021.00364

Meta Pairwise Relationship Distillation for Unsupervised
Person Re-identification
Haoxuanye Ji1 Le Wang1∗ Sanping Zhou1 Wei Tang2 Nanning Zheng1 Gang Hua3
1
Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University
2
University of Illinois at Chicago 3 Wormpex AI Research
Ranking list

Abstract

Pairwise Similarity
(q,i1)

q

i2

i1

Pairwise Pseudo
Labels

if K<2

(q,i3)

(q,i2)

if K=2

i3

if K>2

Pairs

Unsupervised person re-identification (Re-ID) remains
challenging due to the lack of ground-truth labels. Existing
methods often rely on estimated pseudo labels via iterative clustering and classification, and they are unfortunately
highly susceptible to performance penalties incurred by the
inaccurate estimated number of clusters. Alternatively, we
propose the Meta Pairwise Relationship Distillation (MPRD)
method to estimate the pseudo labels of sample pairs for
unsupervised person Re-ID. Specifically, it consists of a Convolutional Neural Network (CNN) and Graph Convolutional
Network (GCN), in which the GCN estimates the pseudo labels of sample pairs based on the current features extracted
by CNN, and the CNN learns better features by involving
high-fidelity positive and negative sample pairs imposed
by GCN. To achieve this goal, a small amount of labeled
samples are used to guide GCN training, which can distill
meta knowledge to judge the difference in the neighborhood
structure between positive and negative sample pairs. Extensive experiments on Market-1501, DukeMTMC-reID and
MSMT17 datasets show that our method outperforms the
state-of-the-art approaches.

1. Introduction
Given a query pedestrian image, person re-identification
(Re-ID) aims to match it with target pedestrian images of
the same identity. It remains challenging due to the large
appearance variations caused by different viewing angles,
light conditions and background clutters in disjoint scenes.
Existing methods usually learn discriminative features in a
supervised manner [39, 35, 2, 1, 25], which requires extensive manual labeling efforts. Due to the prohibitively high
cost of such annotation, training person Re-ID systems in
the unsupervised manner has become a popular and practical
research topic.
* Corresponding author.

978-1-6654-2812-5/21/$31.00 ©2021 IEEE
DOI 10.1109/ICCV48922.2021.00364

KNN

(1,0,0)
(1,1,0)
(1,1,1)

(a) Only Similarity
q

Baseline
Network

(q,i1)

i1

Pairwise
Pseudo

(q,i2)

q i2

Labels

GCN

Pairwise Label
Estimation

(1,0,1)

(q,i3)
i3
Pairwise Neighborhoods
Structure
q

Pairs

i3

i2

i1

Ranking list

(b) Our method

Ranking list

Ranking list

Figure 1. Illustrations of two pseudo label estimation methods,
in which (a) the traditional method directly take the pairwise similarity to estimate pseudo labels, while (b) our method takes the
pairwise neighborhood structures to estimate pseudo labels. Each
circle denotes an individual image. The green circles represent the
same identity as the query image, dark color indicates high visual
similarity, while red circles represent other identities.

Recent unsupervised person Re-ID methods [13, 14, 6]
attempted to learn discriminative feature embeddings from
unlabeled training data based on iterative clustering and
classification. However, it is often nontrivial to determine the
number of clusters, and mishaps that wrongly estimate the
cluster numbers often incurs excessive noise in the pseudo
labels.
To address these issues, we reformulate the unsupervised
discriminative feature learning as a pairwise relationship
estimation problem. In this paper, we use the term positive
pair to denote a pair of the pedestrian images of the same
perceived identity; and conversely, negative pair to denote
images with different perceived identities. In the embedding specified by a GCN, positive pairs are pulled closer;
while negative pairs are pushed away from one another. With
this soft semantic preserving rule replacing the clustering

3641

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.

algorithm, the dilemma of determining cluster numbers are
circumvented. In the unsupervised learning paradigm, we
will need to differentiate such positive pairs and negative
pairs without relying on human annotations. One intuitive
solution is thresholding visual similarity scores as the criterion, i.e., considering two images with high visual similarity
as a positive pair, and vice versa. However, as with many
other thresholding based techniques, this criterion is unreliable in practice. For example, as shown in Figure 1 (a), pair
(q, i2 ) is higher in visual similarity score than pair (q, i3 ),
contradicting the ground-truth. Alternatively, we argue that
a graph structure is more suitable to estimate pairwise labels, as shown in Figure 1 (b), which exploits contextual
information to deduce the correct pairwise pseudo label for
(q, i2 ).
In this paper, we propose the Meta Pairwise Relationship
Distillation (MPRD) method for unsupervised person ReID. It comprises a Convolutional Neural Network (CNN)
and Graph Convolutional Network (GCN), where the GCN
estimates the pseudo labels of sample pairs via the meta
knowledge learned from small amount of labeled samples,
and the CNN learns the discriminative features from input
images according to the estimated pseudo labels.
Specifically, the CNN and GCN are trained in an alternating manner, which iteratively and respectively refines its
per-image feature and pairwise pseudo labels. At each iteration, the CNN extracts the current per-image feature, and
updates the feature memory by a linear combination of it
and the previous features. Afterwards, the pairwise neighborhood structure is estimated by connecting every image with
its neighbors, according to the visual similarity metric. The
resulting graph structure is then fed into the GCN to infer
the pseudo label for sample pairs. Empirically, we found
that it is very hard to train the GCN without any supervision,
therefore, we exploit a small amount of labeled metadata to
explicitly supervise GCN, which greatly helps its robustness.
The GCN is only leveraged to provide pseudo supervision
to the CNN training, and it is excluded in the testing stage.
We evaluate our proposed method on Market-1501 [34],
DukeMTMC-reID [20], and MSMT17 [26] datasets.
In summary, the contributions of this paper are summarized as follows.
1. We reformulate the unsupervised discriminative feature
learning task as a pairwise relationship estimation problem, which avoids the error-prone step of estimating the
number of clusters in most existing methods.
2. We propose the MPRD method for unsupervised person
Re-ID, which incorporates a dedicated GCN as the pairwise pseudo label generator in the training stage and it
iteratively refines its estimated labels with better CNN
features.
3. We design an effective GCN that generates high-fidelity
pseudo labels based on the pairwise neighborhood struc-

tures.

2. Related Work
Supervised Person Re-identification methods require
labor-intensive labeled images during their training process.
Early methods usually extract a global feature representation per image for image retrieval [28, 18, 10]. In PersonNet [28], a small-scale convolutional filter captures the
fine-grained cues. By combining such cues and automatically determined scale weights, multi-scale discriminative
features are learned in [18]. SPRe-ID [10] employs a human semantic parsing technique to capture the pixel-level
discriminative clues. When the background is cluttered or
the pedestrian is occluded, part-level features are shown to
boost performance with the mining of discriminative body
regions [22, 19, 42, 5, 41]. Attention and multi-loss are also
used to enhance representation learning from a multi-view
perspective [29, 33, 4, 21, 40].
Unsupervised Person Re-identification methods relieve the requirement for the cost-prohibitive annotations,
which include hand-crafted feature based methods [12, 34],
unsupervised domain adaptation methods [7, 36, 37, 16, 9,
11, 3, 31, 43] and fully unsupervised methods [13, 6, 27, 15,
24, 14]. It is very challenging to hand-craft robust features
to handle the appearance variations incurred by different
camera models, varying illuminations and viewpoints.
Methods based on unsupervised domain adaptation utilize prior knowledge on a source dataset with labels, and
attempt to generalize on another unlabeled target dataset.
HHL [36] enforces camera invariance and domain connectedness to improve the generalization. ECN [37] introduces
an exemplar memory to store features of the target domain
and accommodate examplar-invariance, camera-invariance,
and neighborhood-invariance of the target domain properties. SSG [7] exploits the potential similarity (from the
global body and local parts) of unlabeled samples to automatically build multiple clusters from different views. Mekhazni
et.al. [16] design the Dissimilarity-based Maximum Mean
Discrepancy loss to bridge the domain gap. ADTC [9] uses
an unsupervised voxel attention and a two-stage clustering
strategy to to account for the variations in images.
Some fully unsupervised methods are guided by pseudo
supervision obtained from clustering results on the embeddings [13, 6, 14]. SSLR [15] replaces the hard one-hot label
with soft labels to alleviate the error caused by unsupervised clustering. MLCR [24] predicts a “multi-label” for
each training sample through Memory-based Positive Label Prediction (MPLP) and learns discriminative features
via the Memory-based Multi-label classification loss. With
the intrinsic “tracklet” structure and appearance, TSSL [27]
eliminates the necessity of both pedestrian identity and camera labels.

3642
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.

1

2

3

Multi-Layer Perception

Gqi
Gqi
Gqi

AGGREGATION

Feature

i1
i2
i3

GRAPH CONV-LAYER

Pairwiase
Neighborhood
Structure

GRAPH CONV-LAYER

q

CNN model

q

Pairwise
Relationship

Pq

NNk(q)

N×D Features

Figure 2. Overview of MPRD. An initialized backbone network extracts the feature of the training image. Then GCN infers the pairwise
relationship between the features and their neighbors, which is used to train the CNN model.

The most relevant existing method is MLCR [24], which
reformulates the unsupervised person Re-ID task as a multilabel classification problem. However, we argue that our
MPRD differs from MLCR in two aspects. First, we reformulate the task as a pairwise relationship estimation problem; second, we design an effective GCN model to provide
high-fidelity pseudo labels. The ablation study in Section 5.3
verifies the MPRD’s performance advantage over MLCR.

where γ (t) denotes an iteration-dependent updating rate.
This feature memory mechanism practically implements a
smoothing operation over the iterations, potentially reducing
violent oscillations in features.
Loss function. Suppose the pairwise pseudo labels are
provided by GCN, we introduce the Binomial Deviance (BD)
loss [30] function LF to train the CNN, which aims to minimize the distance in positive pairs and to maximize the
distance in negative pairs.

3. Meta Pairwise Relationship Distillation
Given an unlabeled dataset X = {xi }N
i=1 , where xi denotes the ith input image, and N denotes the number of
training samples, the MPRD estimates the pairwise pseudo
labels for feature learning. As illustrated in Figure 2, the
CNN learns discriminative features supervised by the pairwise pseudo labels generated by GCN; while the GCN estimates the pairwise pseudo labels based on CNN features.
This interdependency is practically solved via alternating
optimization of the GCN and the CNN.

3.1. CNN
Network backbone. The CNN module extracts discriminative features, which allows nearest neighbor search in the
feature space. For simplicity, we adopt the backbone network
in [8] as our CNN choice* , which consists of a feature extraction module and a feature memory module. In practice, the
feature extraction module F extracts a d-dimensional feature
F(xi ) from each input image xi , and then `2 -normalized by
F̃(xi ) ← F(xi )/kF(xi )k2 , kF(xi )k2 indicates the norm
of F(xi ), the feature memory M stores all the features of
training images. The feature memory is updated at the tth
iteration as follows.
M(t) [i] ← γ (t) F̃(xi ) + (1 − γ (t) )M̃(t−1) [i],
M̃(t) [i] ← M(t) [i]/kM(t) [i]k2 ,
* Our method is compatible with various network backbones.

(1)

LF =

n
X
1X δ
ζ(α(λ1 − hF̃(xi ), M̃[j]i))
n i=1 | Pi |
xj ∈Pi

(2)
X
1
+
ζ(α(hF̃(xi ), M̃[j]i − λ2 )),
| N∗i |
∗
xj ∈Ni

where h·, ·i represent inner product, since both F̃ (xi ) and
M̃[j] are `2 -normalized unit vector, hF̃(xi ), M̃[j]i denote
the cosine similarity between them, ζ(x) = log(1 + ex ), n
is the batch size, | · | denotes the cardinality (number of elements), δ indicates the importance of positive pairs against
negative pairs, λ1 and λ2 denote two margin parameters, and
α is an amplification factor. Besides, Pi and Ni represent
the positive pair list and the negative pair list, respectively.
As |Ni ||Pi | in practice, we further focus on the hard negative pair list N∗i with fixed size r as follows.
N∗i = {xj | xj ∈ top(hF̃(xi ), M̃[j]i, r), xj ∈ Ni }, (3)
where top(·, r) represent the r largest samples. Therefore,
N∗i contains the r closest negative samples to the query xi
in the embedding feature space.
After training the CNN, all positive pairs concentrate
within a radius of λ1 ; while all negative pairs locate elsewhere with a distance of at least λ2 . Afterwards, a nearest
neighbor searching algorithm can be applied to solve the
person Re-ID problem.

3643
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:11:04 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
arXiv:2412.04719v1 [cs.CV] 6 Dec 2024

Mix-Modality Person Re-Identification: A New and Practical
Paradigm
WEI LIU, School of Computer Science and Technology, Wuhan University of Science and Technology, China
XIN XU, School of Computer Science and Technology, Wuhan University of Science and Technology, China
HUA CHANG, School of Computer Science and Technology, Wuhan University of Science and Technology,
China

XIN YUAN, School of Computer Science and Technology, Wuhan University of Science and Technology,
China

ZHENG WANG, National Engineering Research Center for Multimedia Software, Institute of Artificial
Intelligence, School of Computer Science, Wuhan University, China
Current visible-infrared cross-modality person re-identification research has only focused on exploring the
bi-modality mutual retrieval paradigm, and we propose a new and more practical mix-modality retrieval
paradigm. Existing Visible-Infrared person re-identification (VI-ReID) methods have achieved some results
in the bi-modality mutual retrieval paradigm by learning the correspondence between visible and infrared
modalities. However, significant performance degradation occurs due to the modality confusion problem when
these methods are applied to the new mix-modality paradigm. Therefore, this paper proposes a Mix-Modality
person re-identification (MM-ReID) task, explores the influence of modality mixing ratio on performance, and
constructs mix-modality test sets for existing datasets according to the new mix-modality testing paradigm. To
solve the modality confusion problem in MM-ReID, we propose a Cross-Identity Discrimination Harmonization
Loss (CIDHL) adjusting the distribution of samples in the hyperspherical feature space, pulling the centers of
samples with the same identity closer, and pushing away the centers of samples with different identities while
aggregating samples with the same modality and the same identity. Furthermore, we propose a Modality
Bridge Similarity Optimization Strategy (MBSOS) to optimize the cross-modality similarity between the
query and queried samples with the help of the similar bridge sample in the gallery. Extensive experiments
demonstrate that compared to the original performance of existing cross-modality methods on MM-ReID, the
addition of our CIDHL and MBSOS demonstrates a general improvement.
CCS Concepts: • Computing methodologies → Artificial intelligence; Computer vision;
Additional Key Words and Phrases: Cross-Modality Person Re-identification, Mix-Modality Paradigm, Metric
learning, Post-processing
ACM Reference Format:
WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG. 2024. Mix-Modality Person Re-Identification:
A New and Practical Paradigm. J. ACM , (May 2024), 21 pages. https://doi.org/XXXXXXX.XXXXXXX
Authors’ addresses: WEI LIU, liuwei@wust.edu.cn, School of Computer Science and Technology, Wuhan University of
Science and Technology, Wuhan, China; XIN XU, xuxin@wust.edu.cn, School of Computer Science and Technology, Wuhan
University of Science and Technology, Wuhan, China; HUA CHANG, changhua@wust.edu.cn, School of Computer Science
and Technology, Wuhan University of Science and Technology, Wuhan, China; XIN YUAN, yuanxin@wust.edu.cn, School
of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan, China; ZHENG WANG,
wangzwhu@whu.edu.cn, National Engineering Research Center for Multimedia Software, Institute of Artificial Intelligence,
School of Computer Science, Wuhan University, Wuhan, China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 0004-5411/2024/5-ART
https://doi.org/XXXXXXX.XXXXXXX
J. ACM, Vol. , No. , Article . Publication date: May 2024.

2

WEI LIU, XIN XU, HUA CHANG, XIN YUAN, and ZHENG WANG
Visible Probe

Infrared Probe

Infrared Gallery

Visible Gallery

Mix Modality Probe

Mix Modality Gallery

or

(a)
Probe

Gallery

(b)
Probe

Ranklist
Shapes denote Identities

Distance

(c)

Visible modality sample
Infrared modality sample
Incorrect match distance

Distance

Correct match distance

Fig. 1. (a) Existing bi-modality mutual retrieval test paradigms for VI-ReID use a visible probe image queried
in an infrared image gallery or an infrared probe image queried in a visible image gallery. (b) Our proposed
mix-modality testing paradigm for MM-ReID uses mix-modality probes to query in a mixed-modality gallery.
(c) The unique challenge in MM-ReID is the interference of sample points with different identities in the same
modality. In the figure, blue/gray represents visible/infrared modality samples, respectively, and different
shapes represent different identities. It can be seen that due to more similar color and other identity-irrelevant
information, the distance between samples of the same modality but different identities is closer than that
between samples of the same identity with different modalities, which eventually leads to confusion during
retrieval and reduces the accuracy.

1

INTRODUCTION

Person Re-IDentification (ReID) is a critical task in intelligent video security systems, aiming to
retrieve specific pedestrians across a network of non-overlapping cameras [11, 33, 37]. While SingleModality ReID (SM-ReID) methods focusing on visible image retrieval have achieved significant
advancements, they often fall short in low-light conditions where crime rates are notably higher.
These illumination conditions lead to substantial information loss in RGB camera captures. To
overcome this limitation, increasingly sophisticated cameras capable of automatically switching to
infrared mode in low-light conditions are being integrated into video surveillance systems. This
shift has spurred a growing research interest in Visible-Infrared person re-identification (VI-ReID),
which aims to tackle the challenge of cross-modality image matching [9, 14, 16, 36].
VI-ReID not only grapples with the common challenges of viewpoint, background, pose, and
occlusion typical of SM-ReID task [2, 3, 10, 28, 32, 43, 44] but also faces significant difficulties
arising from modality differences. Despite considerable progress achieved in the bi-modality mutual
retrieval paradigm, as illustrated in Fig 1 (a) through learning potential correspondences between
visible and infrared images, practical ReID scenarios pose additional complexities. As shown in
Fig 1 (b), pedestrian images in real applications may need to be identified across both day and night,
necessitating a database that integrates both visible and infrared images—not merely a collection
of one modality type. This integration often leads to a mix of what seems to be straightforward
SM-ReID tasks into the existing cross-modality framework. However, as indicated in Fig 1 (c), this
approach results in a marked performance degradation, primarily due to the ‘Modality Confusion’
J. ACM, Vol. , No. , Article . Publication date: May 2024.

Mix-Modality Person Re-Identification: A New and Practical Paradigm

3

problem. This issue stems from identity-independent features such as colors being more similar
within the same modality, which confuses the matching of cross-modality identity information.
More specifically, the impact of modality confusion on the current approach of learning only
visible-infrared cross-modality correspondences is huge, due to the fact that there exists not only
one correspondence from visible to infrared in the retrieval process, but also two same-modality
correspondences, which is not taken into account by the existing approaches of VI-ReID. Under this
influence, the natural similarity between samples of the same modality will disturb the perception of
the algorithm, resulting in the distance between samples of different identities of the same modality
being smaller than the distance between samples of the same identity of different modalities, and
ultimately giving wrong retrieval results. To address these challenges, we introduce a new and
practical Mix-Modality person re-identification (MM-ReID) task, creating mixed modality test sets
for existing datasets and examining how the ratio of modality mixing affects retrieval performance.
To address the modality confusion challenge in MM-ReID, we introduce the Cross-Identity
Discrimination Harmonization Loss (CIDHL). This novel approach optimizes the sample distribution within a hyperspherical feature space. Specifically, CIDHL aggregates distances between
samples of the same identity and modality towards their respective centers, while simultaneously
drawing together centers of samples sharing identities across different modalities and distancing
those of different identities, irrespective of modality. This strategy mitigates the effects of modality
confusion. Additionally, we propose a Modality Bridge Similarity Optimization Strategy (MBSOS),
which optimizes the cross-modality distance metric by identifying a similar bridge sample in the
gallery to serve as an intermediary, thus refining the distance measures between the query and the
queried samples. Extensive testing confirms that our methods—CIDHL and MBSOS—significantly
enhance the model performance of existing cross-modality approaches, demonstrating their efficacy
and adaptability in resolving modality confusion in MM-ReID.
The contributions of this paper are three aspects:
• Paradigm Contribution: We pioneer the Mix-Modality person re-identification (MMReID) task, introducing a novel testing paradigm that incorporates both visible and infrared
modalities within a single framework.
• Empirical Contribution: We investigate the impact of varying modality mixing ratios
on cross-modality model performance and conclude the unique challenges of ‘Modality
Confusion’ specific to MM-ReID.
• Technical Contribution: We introduce the Cross-Identity Discrimination Harmonization
Loss (CIDHL) and the Modality Bridge Similarity Optimization Strategy (MBSOS). These
transferable methodologies effectively address modality confusion, as validated by extensive
empirical experiments.
2
2.1

RELATED WORK
Single-Modality Person Re-Identification

Single-Modality person re-identification (SM-ReID) seeks to identify and retrieve all images of a
specific pedestrian across a network of non-overlapping cameras using visible light images. With
the advancement of deep learning technologies, SM-ReID has achieved substantial progress in
recent years. However, the predominant focus of existing approaches has been on addressing
changes in pose, viewpoint, and occlusion [4, 38], with less attention given to the degradation of
image quality in low-light conditions [19, 21], which are prevalent in high-crime areas. Some studies
have explored low-light person re-identification, focusing on learning illumination-independent
features and employing image enhancement techniques [7, 8, 15, 39, 42]. Despite these efforts, the
loss of pedestrian identity information due to the inherent limitations of RGB cameras in dimly lit
J. ACM, Vol. , No. , Article . Publication date: May 2024.


exec
/bin/zsh -lc "pdftotext -l 3 'MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
MSP-REID: HAIRSTYLE-ROBUST CLOTH-CHANGING PERSON RE-IDENTIFICATION
Xiangyang He1
1

Lin Wan1,*

School of Computer Science, China University of Geosciences, Wuhan, China

arXiv:2603.01640v2 [cs.CV] 7 Mar 2026

ABSTRACT
Cloth-Changing Person Re-Identification (CC-ReID) aims to
match the same individual across cameras under varying clothing
conditions. Existing approaches often remove apparel and focus on
the head region to reduce clothing bias. However, treating the head
holistically without distinguishing between face and hair leads to
over-reliance on volatile hairstyle cues, causing performance degradation under hairstyle changes. To address this issue, we propose the
Mitigating Hairstyle Distraction and Structural Preservation (MSP)
framework. Specifically, MSP introduces Hairstyle-Oriented Augmentation (HSOA), which generates intra-identity hairstyle diversity
to reduce hairstyle dependence and enhance attention to stable facial and body cues. To prevent the loss of structural information,
we design Cloth-Preserved Random Erasing (CPRE), which performs ratio-controlled erasing within clothing regions to suppress
texture bias while retaining body shape and context. Furthermore,
we employ Region-based Parsing Attention (RPA) to incorporate
parsing-guided priors that highlight face and limb regions while suppressing hair features. Extensive experiments on multiple CC-ReID
benchmarks demonstrate that MSP achieves state-of-the-art performance, providing a robust and practical solution for long-term person
re-identification.
Index Terms— Cloth-Changing Person Re-Identification(CCReID), hairstyle augmentation, clothing-structure preservation,
parsing-guided regional priors.
1. INTRODUCTION
Person re-identification (Re-ID) aims to recognize the same
individual across different cameras and time intervals [1, 2]. With
the increasing demand for large-scale intelligent surveillance, shortterm Re-ID research has become relatively mature, where a person’s
clothing typically remains unchanged [3]. However, models trained
under such settings tend to overfit clothing appearance, causing
significant performance degradation when individuals change outfits
or different people wear visually similar clothes. This limitation
has motivated research on cloth-changing person re-identification
(CC-ReID), which aims to ensure robust identity matching under
clothing variations and better satisfies the requirements of long-term,
real-world applications [4, 5, 6].
Beyond clothing changes, factors such as hairstyle variations
and aging also significantly affect appearance, yet both are visually
salient but identity-irrelevant features. Existing CC-ReID methods
typically reduce clothing dependence through semantic mining, feature
attention, and regional erasing [4, 7, 8, 9, 10], often without relying
on auxiliary modalities [11, 12, 13]. However, a critical challenge
remains largely overlooked: the impact of hairstyle variations on
recognition has not been adequately addressed. Standard parsing
techniques generally mark the entire head (including both face and hair)
*Corresponding author (wanlin@cug.edu.cn).

(b) Feature map

(a) Human parsing

Erase

query

top-1

top-2

top-3

(c) Retrieval result

top-4

Raw image

Erase image

(d) Clothing erasure

Fig. 1. The Hairstyle Shortcut problem in CC-ReID. (a) Standard parsing merges face and hair together as "head"; (b) attention
consequently focuses excessively on this region; (c) resulting models
are robust to clothing changes but brittle to hairstyle variations; (d)
conventional clothing erasure further removes structural cues.
as identity-related [9, 14] (Fig. 1(a)), causing models to overemphasize
head regions and become highly sensitive to hairstyle cues (Fig. 1(b)).
This introduces a "hairstyle shortcut" whereby models mistakenly rely
on hairstyle as the primary identity cue, degrading their generalization
ability when hairstyles change (Fig. 1(c)).Furthermore, parsing-based
clothing removal methods are often too aggressive [14], eliminating
not only clothing pixels but also crucial structural cues—such as body
silhouette, proportions, and pose—further weakening the model"s
generalizability (Fig. 1(d)).
Motivated by these challenges, we propose the Mitigating
Hairstyle Distraction and Structural Preservation (MSP) framework
to simultaneously address these limitations of existing CC-ReID
methods: the over-reliance on hairstyle cues and the loss of structural
information caused by complete clothing removal. Specifically, we
introduce Hairstyle-Oriented Augmentation (HSOA) to generate
"same-identity, different-hairstyle" samples and align their features in
the embedding space, explicitly decoupling hairstyle from identity representation. To preserve body geometry, we design Cloth-Preserved
Random Erasing (CPRE), which retains a controllable portion of
clothing pixels, suppressing texture bias while maintaining body
contours, posture, and proportions. Finally, we propose Region-based
Parsing Attention (RPA) to leverage human parsing priors, strengthening identity-relevant regions (e.g., face, limbs) and suppressing
non-identity features such as hair.
Contributions. The contributions are summarized as follows.
(1) We pioneer MSP-ReID, a framework that explicitly addresses
hairstyle-induced bias for the first time, leading to consistent improvement in robustness and performance in CC-ReID.
(2) We present Hairstyle-Oriented Augmentation (HSOA) to

HairStyle-Oriented Augmentation(HSOA)

Region-based Parsing Attention(RPA)

Mask
Human Parsing
Parsing result

Raw image

HairFastGan

1

1

0

0

1

1

0

0

1

1

0

0

0

0

0

1

0

0

1

1

0

0

1

0

1

1

0

0

1

1

0

0

0

0

0

SCHP

Semantic Parsing

middle

0

0
0
0
0
0
0
0
0
0

1
0
0
0
0
0
0
0
0

1
0
0
0
0
0
0
0
0

0
0
0
0
0
0
0
0
0

0
0
0
0
1
1
0
0
0

0
1
1
0
0
0
1
1
0

0
1
1
0
0
0
1
1
0

0
0
0
0
1
1
0
0
0

𝓛𝒂𝐭𝐭

Mix Decision

long

Raw image

1

:

short

Erase image

1x1 Conv
+Softmax

Input Image

1

Different Hairstyle

Clothing-Preserved Random Erasing (CPRE)
Clothing
mask

Mix
Decision

CPRE
Clothes
mask

ID
Classifier
𝑭𝐈𝐃

Erase
image

Random erase

𝑭𝐈𝐃'

ID
Encoder

Raw image

Input image

𝓛𝐢𝐝 𝓛𝐭𝐫𝐢

Clothes
Encoder

𝓛𝐜𝐚𝐥
Clothes
Classifier

𝑭𝐂
Inference
Clothing-Preserved image

Erase image

Element-wise Addition

SCHP

Human Parsing tool

Element-wise Multiplication

Fig. 2. Overview of MSP-ReID. HSOA (blue dashed, top-right) synthesizes same-ID different-hairstyle images. CPRE (pink dashed,
bottom-right) creates raw/erased pairs with a clothing keep ratio. RPA (purple, center-top) uses parsing masks to boost face/limbs and suppress
hair. Green denotes the ID branch, pink denotes the clothes branch for adversarial regularization. Inference is RGB-only using the ID branch.
decouple hairstyle cues from identity learning and propose Regionbased Parsing Attention (RPA) to focus representation on stable facial
and body regions while suppressing hair-induced noise.
(3) We propose Cloth-Preserved Random Erasing (CPRE), which
differs from conventional random erasing by maintaining a controllable
ratio of clothing pixels, preserving geometric information (body shape,
pose) and suppressing texture dependency.
(4) Extensive experiments confirm MSP’s robustness and effectiveness, achieving a new state-of-the-art for CC-ReID.
2. METHODOLOGY

image x, we utilize the human parser SCHP [16] to obtain a pixelwise semantic map P = SCHP(x) ∈ {1, . . . , K}H×W . From P
we derive binary masks for face and hair and define the head mask as
their union:
Mface = 1[P ∈ Sface ] , Mhair = 1[P ∈ Shair ] ,
Mhead = Mface ∨ Mhair = 1[P ∈ Sface ∪ Shair ] .

(2)
(3)

The cropped head region x ⊙ Mhead and its mask are fed to HairFastGAN to synthesize three target hairstyles—short, medium,
long—yielding heads ĥS , ĥM , ĥL that preserve facial structure while
altering hair. Each synthesized head is seamlessly composited back:

2.1. Problem Formulation
Let x ∈ RH×W ×3 be a pedestrian image with identity label y.
Besides identity, the appearance also contains time-varying nuisance
attributes such as clothing c and hairstyle h. We posit an underlying
data-generating distribution p(x, y, c, h) but only observe (x, y) during training, (c, h) are unannotated. Our goal is to learn an encoder
fθ that maps x to an embedding z = fθ (x) which is (i) highly
discriminative for identity and (ii) insensitive to (c, h). Formally, we
seek a representation z that maximizes identity information while
minimizing dependence on nuisance factors:



max {I fθ (X), Y − λc I fθ (X), C − λh I fθ (X), H }, (1)
θ

where I(·, ·) denotes mutual information, i.e., the amount of information one variable reveals about the other (I(Z, C) = 0 implies Z and
C are statistically independent). Here, Y denotes the identity label, C
the clothing state, and H the hairstyle state. Equivalently, we aim for
z to preserve identity-discriminative cues (large inter-identity margins,
low intra-identity variance) while suppressing clothing/hairstyle cues.
2.2. Hairstyle-Oriented Augmentation (HSOA)
To explicitly break the shortcut "hair ≈ identity", we perform a
hairstyle augmentation with HairFastGAN [15]. For each training

x̃ℓ = Mhair ⊙ ĥℓ + (1 − Mhair ) ⊙ x,

ℓ ∈ {S, M, L},

(4)

where ⊙ denotes element-wise multiplication. Each x̃ℓ inherits
the identity label y and the clothing label c. The augmented set
{x̃S , x̃M , x̃L } is sampled together with originals, creating abundant
positive pairs of the same identity under different hairstyles. Notably,
we leverage triplet loss to optimize the feature space, pulling closer
representations of the same identity across varying hairstyles and
clothing while pushing apart those from different identities.
2.3. Cloth-Preserved Random Erasing (CPRE)
Although clothing contains strong visual cues, they are considered
identity-unrelated features in our task. A common practice in existing
methods is to remove this information entirely to compel the model to
learn identity-related features. However, removing the entire clothing
region also discards useful information about body structure and
spatial context. To address this limitation, we propose Cloth-Preserved
Random Erasing (CPRE), we design Cloth-Preserved Random Erasing
(CPRE) to erase only within the clothing region, retaining a random
proportion of clothing pixels, forcing the model to rely more on
identity-related cues (face, limbs, shape). Let Mcloth ∈ {0, 1}H×W be
the clothing mask (slightly dilated to cover boundary errors). Sample
a keep ratio r ∈ [rmin , rmax ] and draw Kr ∈ {0, 1}H×W inside the

clothing
a proportion r is preserved
 region such that approximately

(i.e., E Kr (i, j) Mcloth (i, j) = 1 = r) . The erased image is

Table 1. Results on PRCC and LTCC. The best result is in bold, and
the second-best result is underlined.
Methods



xerase = (1−Mcloth )+Mcloth ⊙Kr





⊙x+ Mcloth ⊙(1−Kr ) ⊙ϵ,
(5)

where ϵ is a constant fill (zero).
Pixel-wise image. Equivalently, for each pixel (i, j),


xi,j ,
xerase
xi,j ,
i,j =

ϵ,

Mcloth (i, j) = 0,
Mcloth (i, j) = 1 ∧ Kr (i, j) = 1,
Mcloth (i, j) = 1 ∧ Kr (i, j) = 0 .

(6)

image averaging. When CPRE is enabled, the Mix Decision module
constructs input batches by the raw image x and erased image xerase
at a 1:1 ratio.
2.4. Region-based Parsing Attention (RPA)
While CPRE reduces reliance on clothing features, hairstyle
remains a prominent, identity-unrelated distractor. To mitigate this
specific problem, we propose Region-based Parsing Attention (RPA),
a lightweight attention mechanism that uses human parsing priors to guide the model’s focus.It generates a spatial attention map
that emphasizes identity-related regions and minimizes attention to
identity-unrelated features, such as hair, helping the model learn more
robust identity representations.
Backbone and ID head. A backbone B(·) produces a feature map
F ∈ RC×H×W , a shallow ID head yields FID = ϕid (F ).
Attention prediction and gating. Given FID , a 1 × 1 convolution
predicts attention logits S ∈ R1×H×W :
S = W ∗ FID + b,

exp(Sij )
∈ (0, 1),
u,v exp(Suv )

Âij = P

(7)

where W ∈ R1×C×1×1 . The gated ID features are
FID ” = FID ⊙ Â,

(8)

with Â broadcast along channels, global average pooling (GAP) of
FID ” is used for downstream losses. At test time, the RPA gate is
disabled and the model uses the ungated FID .
Parsing-guided attention loss. Given the parsing masks Mface , Mlimbs
, Mhair , with
Mface , Mlimbs , Mhair ∈ {0, 1}H×W ,
we define the normalized positive target
T+ =

Mface + Mlimbs
.
⟨1, Mface + Mlimbs ⟩ + ε

(9)

We supervise Â toward T+ and penalize mass on hair:
Latt = −⟨T+ , log Â⟩ + λneg

⟨Â, Mhair ⟩
.
⟨1, Mhair ⟩ + ε

When parsing masks are absent, this term is omitted.

(10)

Year

PRCC
Cloth-Changing
Standard

LTCC
Cloth-Changing Standard

R1

mAP

R1

R1

mAP

R1

mAP

HACNN [20]
CVPR ’18 21.8
PCB [21]
ECCV ’18 41.8
IANet [22]
CVPR ’19 46.3
FSAM [7]
CVPR ’21 54.5
AIM [12]
CVPR ’23 57.9
CCFA [13]
CVPR ’23 61.2
Instruct-ReID [23] CVPR’24 54.2
LIFTCAP [10]
TVT’24
54.3
JIMGP [24]
TMM’24 57.3
CISupNet [25]
ICASSP’25 58.3
FAIM [26]
TMM’25 59.8
RLQ [27]
arXiv’25 64.0

38.7
46.9
58.3
58.4
52.3
55.6
65.8
58.2
62.5
63.2

82.5
21.6
99.8 97.0 23.5
99.4 98.3 25.0
98.8
38.5
100.0 99.9 40.6
99.6 98.7 45.3
100.0 99.8 37.0
99.7 99.8 43.4
100.0 99.8 41.5
100.0 100.0 48.2
100.0 99.8 46.4

9.3
10.0
12.6
16.2
19.4
22.1
39.7
18.2
19.2
27.5
21.5

60.2
65.1
63.7
73.2
76.3
75.8
75.8
76.0
79.5
76.9

26.7
30.6
31.0
35.4
41.1
42.5
52.0
41.6
53.4
41.8

CAL [11]
ours

55.8
63.4

100.0
100.0

18.0
19.3

74.2
78.7

40.8
60.1

CVPR ’22
-

55.2
65.1

mAP

99.8
99.1

40.1
41.6

2.5. Objective
We optimize a weighted sum of four terms. Here, Lid and Ltri
are well-known identity classification and triplet losses, Latt is the
parsing-guided attention loss defined in Sec. 2.4, and Lcal denotes the
clothes-adversarial loss adopted from our baseline CAL[11].
Ltotal = Lid + λtri Ltri + λatt Latt + λcal Lcal ,

(11)

where λtri , λatt , λcal balance the contributions.
Inference. At test time, we only use the ungated FID branch, followed
by ℓ2 normalization and cosine similarity for retrieval.
3. EXPERIMENTS
3.1. Datasets and Protocols
Datasets. We evaluate the proposed method on four mainstream
Cloth-changing person Re-ID benchmarks: PRCC [17], LTCC [4],
VC-Clothes [18], and LaST [19]. The first three are medium-scale
datasets, while LaST is a large-scale dataset.
Implementation details. We use CAL[11] as the baseline CC-ReID
model in our experiments as it generally performed the best with our
generated data across all datasets. We use ResNet-50 pre-trained on
ImageNet with a maxavg global pooling head and BNNeck. Inputs
are resized to 384 × 192. For random erasing, the optimal erase
range is [0.1, 0.3] on PRCC, LaST and VC-Clothes, and [0.2, 0.5] on
LTCC. We train for 60 epochs using Adam (lr 3.5 × 10−4 , weight
decay 5 × 10−4 ) with a step scheduler (decay ×0.1 at epochs 20 and
40). All experiments run on two NVIDIA RTX 3080 Ti GPUs. For
PRCC/LTCC/VC-Clothes we sample 4 identities × 16 images per
identity per GPU, for LaST we sample 2 identities × 16 images per
identity per GPU. λcal performs best at 0.5 on PRCC/LaST/LTCC,
1.0 on VC-Clothes, and λatt = 1.0 performs best across all datasets.
Evaluation is performed every 5 epochs.
Evaluation metrics. We report Rank-1 (R1) and mean Average
Precision (mAP), the two common metrics in Cloth-changing Re-ID.
For all dataset, we evaluate our method under the standard setting and
the cloth-changing setting following[8].
3.2. Comparison with State-of-the-Art Methods
Comparative results on PRCC and LTCC. Tables 1 reports comparisons on PRCC and LTCC with classic Re-ID methods (e.g.,
HACNN [20], PCB [21], IANet [22]) and recent CC-ReID methods
(e.g., AIM [12], RLQ [27], CAL [11]). On PRCC (cloth-changing),
MSP-ReID shows a clear margin over the CAL baseline (about ten
points in Rank-1) and is on par with, or slightly better than, recent
strong competitors such as RLQ. On PRCC (standard), performance


codex
我按 `pdftotext -l 3` 只读了每篇前三页。先给结论：这批里真正值得学的是“改问题口径”和“暴露隐藏捷径”，不是再加一个局部注意力模块。对我们最有价值的是 Mix-Modality、MSP-ReID、Lifelong VI-ReID 这三类写法。

**逐篇提取**

1. `Lifelong Visible-Infrared...PP-IPG`
创新类型：问题重定义加新机制。  
能发原因：固定训练集的 VI-ReID 不符合真实昼夜监控，作者把问题改成跨域、跨模态、持续到来的 LVI-ReID，再用 prompt pool 保存共享知识，用实例级 prompt 适配样本差异。证据链是新任务设定、终身对比、遗忘控制、prompt 选择消融。  
story 套路：真实系统不是一次训练完，模型必须一边跨可见光和红外，一边持续吸收新域。

2. `Local-Aware Residual Attention ViT`
创新类型：新机制，偏工程组合。  
能发原因：抓住 ViT 在 VI-ReID 中“过度全局化”和深层特征坍缩的问题，用局部残差注意力和卷积式通道重构把浅层局部细节带回去。证据链靠 GradCAM、patch 相似矩阵、模块消融和 SYSU、RegDB、LLCM。  
story 套路：ViT 不是天然适合 VI-ReID，它会丢掉跨模态匹配需要的局部低层细节。

3. `Mask-Aware Hierarchical Aggregation Transformer`
创新类型：新机制，接近工程组合。  
能发原因：遮挡 ReID 的注意力会看向遮挡物和背景，作者用人体 mask embedding 引导非遮挡区域，再加层级 patch 聚合和多路径补全。证据链是遮挡热图、遮挡和完整数据集双验证、MIE/HFA/DFC 消融。  
story 套路：遮挡下不是更强注意力就够了，模型必须知道哪些身体区域是可靠证据。

4. `Memory-augmented shuffled meta learning`
创新类型：新机制。  
能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
story 套路：训练过程要像真实检索，而不是只像分类或 batch 内度量学习。

5. `Meta Pairwise Relationship Distillation`
创新类型：问题重定义加新机制。  
能发原因：无监督 ReID 的聚类伪标签依赖 cluster 数，容易错。作者把问题改成“估计样本对关系”，用少量有标注元数据训练 GCN 判断邻域结构，再反过来监督 CNN。证据链是避免 cluster 数、pairwise label 质量、无监督主流数据集对比。  
story 套路：不要先猜有多少身份，先学会判断两张图在邻域结构里是不是同一人。

6. `Mix-Modality...TOMM`
创新类型：新数据设定和问题重定义，机制是补充。  
能发原因：传统 VI-ReID 只测 visible query 到 infrared gallery 或反向，但真实图库会混有可见光和红外。混合图库会产生 modality confusion，同模态不同人可能比跨模态同人更近。证据链是构造混合测试集、改变混合比例、证明旧方法掉点，再用 CIDHL 和桥样本后处理改善。  
story 套路：旧 benchmark 问错了问题，真实系统要在混合模态图库里检索。

7. `Mix-Modality...arXiv`
创新类型和上面相同，是同一工作的早版。  
能发原因和 story 也相同。这个重复版本反而说明一件事：只要新设定足够贴近真实部署，即使机制不复杂，也能支撑 TOMM 级故事。

8. `MoDA`
创新类型：新机制，偏参数高效工程。  
能发原因：DG ReID 的 MoE 方法有效但参数随域数和身份数膨胀，作者用冻结 CLIP 主干加 domain adapters 和 block-aware voting，做到少量可训练参数的多域专家。证据链是参数量、DG 性能、adapter/voting 消融。  
story 套路：泛化 ReID 不只是更大模型，而是可扩展、可部署的域专家混合。

9. `MSP-ReID`
创新类型：问题重定义加新机制。  
能发原因：换衣 ReID 去掉衣服后，模型会转而依赖发型。作者把这个定义成 hairstyle shortcut，用发型生成增强、保留结构的衣服擦除、解析注意力压制头发。证据链是 parsing 把脸和头发混在一起、注意力和检索失败可视化、PRCC/LTCC 等数据集对比。  
story 套路：消除一个捷径后，模型会抓住下一个捷径；真正鲁棒要识别并切断捷径链。

**对我们有用的共同经验**

这批 B 类故事的强点不在“模块复杂”，而在三件事：第一，指出旧协议不真实，比如 Mix-Modality 和 Lifelong VI；第二，揭出模型隐藏捷径，比如 MSP 的发型捷径；第三，用训练期外部先验解决测试期真实困难，比如 mask、parsing、prompt、memory，但测试时尽量保持普通 ReID 输入。我们不该再做通用局部注意力、通用 adapter、通用 mask aggregation，除非它们被一个更强的新问题带着走。

**强创新点候选**

1. **混合视角航拍地面 ReID：真实图库不是航拍查地面，而是航拍、地面、不同高度和低清图混在一起检索。**  
挂团队资产：CARGO / AG-ReID.v2，SOLIDER-Swin，SMPL。  
和最像工作的区别：最像 Mix-Modality，但它处理可见光和红外的二元模态混合；我们处理航拍和地面、多高度、尺度、俯仰角造成的连续几何混合。机制不能照搬中心损失，要用视角、尺度、人体可见性定义 view confusion。  
cheap kill-switch：不训练，直接用现有 SOLIDER 或 AG-ReID 模型构造 mixed-view query/gallery，按航拍比例、地面比例、低清比例扫一遍。如果混合协议比纯跨视角掉超过 2 mAP，并且错误主要来自同视角不同人压过跨视角同人，就成立；否则杀掉。

2. **3D 可见性门控 ReID：先判断极端视角下哪些身体证据几何上可比，再让模型只相信这些证据。**  
挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
和最像工作的区别：最像 MAHATMA 和 pose-guided occlusion，但它们用 2D mask 或 pose 说明“哪里没遮挡”；我们用 SMPL 投影、关节可见性、相机视角估计“哪个身体部位在航拍和地面之间可比较”。这是跨视角几何可靠性，不是普通遮挡注意力。  
cheap kill-switch：先不训练，只在一小批 AG-ReID 样本上生成 2D pose 或 SMPL 可见部位，把全局特征或部位特征按可见性重加权做检索。如果航拍 query 的 AP 明显提升，或者错误样本从“姿态/视角相似”中被拉开，再进入训练；如果 train-free 重加权都没信号，杀掉。

3. **跨视角捷径链消毒：航拍地面模型不只会学衣服，还会依赖视角、背景、尺度和发型，必须用反事实样本逐个切断。**  
挂团队资产：aerial-ground 数据，SMPL，pose gating，SOLIDER-Swin。  
和最像工作的区别：最像 MSP-ReID，但 MSP 只讲换衣后的发型捷径；我们讲航拍地面里的多级捷径链。先证明 SOLIDER 特征能预测相机高度、视角、背景块或尺度，再用结构保留的擦除、尺度反事实、背景替换、SMPL 轮廓保持来压掉这些非身份线索。  
cheap kill-switch：冻结 SOLIDER 特征训练几个线性探针，预测视角、相机、高度、背景类别。如果这些 nuisance 可预测性很高，并且错误检索按这些因素聚类，方向成立。若探针预测不出来，说明捷径链证据不足，杀掉。

4. **几何提示池的持续航拍地面 ReID：新无人机高度、新地面相机和新天气持续接入时，用实例级几何 prompt 选择旧知识。**  
挂团队资产：SOLIDER-Swin，CARGO / AG-ReID.v2，SMPL。  
和最像工作的区别：最像 PP-IPG 和 MoDA，但它们是通用 prompt 或 domain adapter；我们的 prompt key 来自几何因素，比如相机俯仰、人体朝向、投影可见部位、分辨率，而不是只按数据集域编号。重点是 task-agnostic、mixed-gallery、old-new camera 同时检索。  
cheap kill-switch：把 AG-ReID 按相机、高度或清晰度切成顺序域，跑一个顺序微调先导，测旧域遗忘和混合图库掉点。如果没有明显遗忘或混合掉点，这条不做；如果掉点明显，再做几何 prompt，而不是普通 prompt pool。

我会把优先级排成：先杀候选 1，再杀候选 2。候选 1 是最像 B 类问题重定义的，成本最低；候选 2 最能吃到团队已有 SMPL 和 pose 门控资产。候选 3 可以作为候选 1 或 2 的诊断章节和机制补强。候选 4 有价值，但终身 ReID 已经拥挤，只有在 AG-ReID 顺序域确实出现强遗忘时才值得继续。
tokens used
86,418
我按 `pdftotext -l 3` 只读了每篇前三页。先给结论：这批里真正值得学的是“改问题口径”和“暴露隐藏捷径”，不是再加一个局部注意力模块。对我们最有价值的是 Mix-Modality、MSP-ReID、Lifelong VI-ReID 这三类写法。

**逐篇提取**

1. `Lifelong Visible-Infrared...PP-IPG`
创新类型：问题重定义加新机制。  
能发原因：固定训练集的 VI-ReID 不符合真实昼夜监控，作者把问题改成跨域、跨模态、持续到来的 LVI-ReID，再用 prompt pool 保存共享知识，用实例级 prompt 适配样本差异。证据链是新任务设定、终身对比、遗忘控制、prompt 选择消融。  
story 套路：真实系统不是一次训练完，模型必须一边跨可见光和红外，一边持续吸收新域。

2. `Local-Aware Residual Attention ViT`
创新类型：新机制，偏工程组合。  
能发原因：抓住 ViT 在 VI-ReID 中“过度全局化”和深层特征坍缩的问题，用局部残差注意力和卷积式通道重构把浅层局部细节带回去。证据链靠 GradCAM、patch 相似矩阵、模块消融和 SYSU、RegDB、LLCM。  
story 套路：ViT 不是天然适合 VI-ReID，它会丢掉跨模态匹配需要的局部低层细节。

3. `Mask-Aware Hierarchical Aggregation Transformer`
创新类型：新机制，接近工程组合。  
能发原因：遮挡 ReID 的注意力会看向遮挡物和背景，作者用人体 mask embedding 引导非遮挡区域，再加层级 patch 聚合和多路径补全。证据链是遮挡热图、遮挡和完整数据集双验证、MIE/HFA/DFC 消融。  
story 套路：遮挡下不是更强注意力就够了，模型必须知道哪些身体区域是可靠证据。

4. `Memory-augmented shuffled meta learning`
创新类型：新机制。  
能发原因：VI-ReID 的普通 triplet 或 center loss 只看 batch 内关系，数据少时学不到全局度量。作者把训练改成 query-support 检索小任务，并用 memory bank 引入历史特征。证据链是 sample-based、center-based、meta-based 三类对比，加 RegDB/SYSU 实验。  
story 套路：训练过程要像真实检索，而不是只像分类或 batch 内度量学习。

5. `Meta Pairwise Relationship Distillation`
创新类型：问题重定义加新机制。  
能发原因：无监督 ReID 的聚类伪标签依赖 cluster 数，容易错。作者把问题改成“估计样本对关系”，用少量有标注元数据训练 GCN 判断邻域结构，再反过来监督 CNN。证据链是避免 cluster 数、pairwise label 质量、无监督主流数据集对比。  
story 套路：不要先猜有多少身份，先学会判断两张图在邻域结构里是不是同一人。

6. `Mix-Modality...TOMM`
创新类型：新数据设定和问题重定义，机制是补充。  
能发原因：传统 VI-ReID 只测 visible query 到 infrared gallery 或反向，但真实图库会混有可见光和红外。混合图库会产生 modality confusion，同模态不同人可能比跨模态同人更近。证据链是构造混合测试集、改变混合比例、证明旧方法掉点，再用 CIDHL 和桥样本后处理改善。  
story 套路：旧 benchmark 问错了问题，真实系统要在混合模态图库里检索。

7. `Mix-Modality...arXiv`
创新类型和上面相同，是同一工作的早版。  
能发原因和 story 也相同。这个重复版本反而说明一件事：只要新设定足够贴近真实部署，即使机制不复杂，也能支撑 TOMM 级故事。

8. `MoDA`
创新类型：新机制，偏参数高效工程。  
能发原因：DG ReID 的 MoE 方法有效但参数随域数和身份数膨胀，作者用冻结 CLIP 主干加 domain adapters 和 block-aware voting，做到少量可训练参数的多域专家。证据链是参数量、DG 性能、adapter/voting 消融。  
story 套路：泛化 ReID 不只是更大模型，而是可扩展、可部署的域专家混合。

9. `MSP-ReID`
创新类型：问题重定义加新机制。  
能发原因：换衣 ReID 去掉衣服后，模型会转而依赖发型。作者把这个定义成 hairstyle shortcut，用发型生成增强、保留结构的衣服擦除、解析注意力压制头发。证据链是 parsing 把脸和头发混在一起、注意力和检索失败可视化、PRCC/LTCC 等数据集对比。  
story 套路：消除一个捷径后，模型会抓住下一个捷径；真正鲁棒要识别并切断捷径链。

**对我们有用的共同经验**

这批 B 类故事的强点不在“模块复杂”，而在三件事：第一，指出旧协议不真实，比如 Mix-Modality 和 Lifelong VI；第二，揭出模型隐藏捷径，比如 MSP 的发型捷径；第三，用训练期外部先验解决测试期真实困难，比如 mask、parsing、prompt、memory，但测试时尽量保持普通 ReID 输入。我们不该再做通用局部注意力、通用 adapter、通用 mask aggregation，除非它们被一个更强的新问题带着走。

**强创新点候选**

1. **混合视角航拍地面 ReID：真实图库不是航拍查地面，而是航拍、地面、不同高度和低清图混在一起检索。**  
挂团队资产：CARGO / AG-ReID.v2，SOLIDER-Swin，SMPL。  
和最像工作的区别：最像 Mix-Modality，但它处理可见光和红外的二元模态混合；我们处理航拍和地面、多高度、尺度、俯仰角造成的连续几何混合。机制不能照搬中心损失，要用视角、尺度、人体可见性定义 view confusion。  
cheap kill-switch：不训练，直接用现有 SOLIDER 或 AG-ReID 模型构造 mixed-view query/gallery，按航拍比例、地面比例、低清比例扫一遍。如果混合协议比纯跨视角掉超过 2 mAP，并且错误主要来自同视角不同人压过跨视角同人，就成立；否则杀掉。

2. **3D 可见性门控 ReID：先判断极端视角下哪些身体证据几何上可比，再让模型只相信这些证据。**  
挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
和最像工作的区别：最像 MAHATMA 和 pose-guided occlusion，但它们用 2D mask 或 pose 说明“哪里没遮挡”；我们用 SMPL 投影、关节可见性、相机视角估计“哪个身体部位在航拍和地面之间可比较”。这是跨视角几何可靠性，不是普通遮挡注意力。  
cheap kill-switch：先不训练，只在一小批 AG-ReID 样本上生成 2D pose 或 SMPL 可见部位，把全局特征或部位特征按可见性重加权做检索。如果航拍 query 的 AP 明显提升，或者错误样本从“姿态/视角相似”中被拉开，再进入训练；如果 train-free 重加权都没信号，杀掉。

3. **跨视角捷径链消毒：航拍地面模型不只会学衣服，还会依赖视角、背景、尺度和发型，必须用反事实样本逐个切断。**  
挂团队资产：aerial-ground 数据，SMPL，pose gating，SOLIDER-Swin。  
和最像工作的区别：最像 MSP-ReID，但 MSP 只讲换衣后的发型捷径；我们讲航拍地面里的多级捷径链。先证明 SOLIDER 特征能预测相机高度、视角、背景块或尺度，再用结构保留的擦除、尺度反事实、背景替换、SMPL 轮廓保持来压掉这些非身份线索。  
cheap kill-switch：冻结 SOLIDER 特征训练几个线性探针，预测视角、相机、高度、背景类别。如果这些 nuisance 可预测性很高，并且错误检索按这些因素聚类，方向成立。若探针预测不出来，说明捷径链证据不足，杀掉。

4. **几何提示池的持续航拍地面 ReID：新无人机高度、新地面相机和新天气持续接入时，用实例级几何 prompt 选择旧知识。**  
挂团队资产：SOLIDER-Swin，CARGO / AG-ReID.v2，SMPL。  
和最像工作的区别：最像 PP-IPG 和 MoDA，但它们是通用 prompt 或 domain adapter；我们的 prompt key 来自几何因素，比如相机俯仰、人体朝向、投影可见部位、分辨率，而不是只按数据集域编号。重点是 task-agnostic、mixed-gallery、old-new camera 同时检索。  
cheap kill-switch：把 AG-ReID 按相机、高度或清晰度切成顺序域，跑一个顺序微调先导，测旧域遗忘和混合图库掉点。如果没有明显遗忘或混合掉点，这条不做；如果掉点明显，再做几何 prompt，而不是普通 prompt pool。

我会把优先级排成：先杀候选 1，再杀候选 2。候选 1 是最像 B 类问题重定义的，成本最低；候选 2 最能吃到团队已有 SMPL 和 pose 门控资产。候选 3 可以作为候选 1 或 2 的诊断章节和机制补强。候选 4 有价值，但终身 ReID 已经拥挤，只有在 AG-ReID 顺序域确实出现强遗忘时才值得继续。
