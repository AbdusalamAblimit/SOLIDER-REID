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
session id: 019ef71f-77a2-7500-beb0-1441643570e5
--------
user
你是 ReID 论文创新挖掘员 7/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(7/20):
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则确认当前目标和状态，然后只用 `pdftotext -l 3` 读取这 9 篇的前三页。当前环境是只读的，所以这轮只做分析，不会改实验记录。
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
/bin/zsh -lc "sed -n '1,260p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Privacy preserving person re-identification via anonymizing diffusion model.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Occluded person Re-Identification with noise injection.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf

codex
目标和状态已对齐：现在确实处在“读文献、做差距分析、先不要开实验”的阶段。我接下来按你给的名单逐篇读前三页，重点抓它们怎么定义问题、怎么构造证据，而不是展开全文复述。
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
/bin/zsh -lc "pdftotext -l 3 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5494

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Disentangling Modality and Posture Factors:
Memory-Attention and Orthogonal Decomposition
for Visible-Infrared Person Re-Identification
Zefeng Lu , Ronghao Lin , and Haifeng Hu , Member, IEEE
Abstract— Striving to match the person identities between
visible (VIS) and near-infrared (NIR) images, VIS-NIR reidentification (Re-ID) has attracted increasing attention due to
its wide applications in low-light scenes. However, owing to
the modality and pose discrepancies exhibited in heterogeneous
images, the extracted representations inevitably comprise various modality and posture factors, impacting the matching of
cross-modality person identity. To solve the problem, we propose a disentangling modality and posture factors (DMPFs)
model to disentangle modality and posture factors by fusing
the information of features memory and pedestrian skeleton.
Specifically, the DMPF comprises three modules: three-stream
features extraction network (TFENet), modality factor disentanglement (MFD), and posture factor disentanglement (PFD). First,
aiming to provide memory and skeleton information for modality
and posture factors disentanglement, the TFENet is designed
as a three-stream network to extract VIS-NIR image features
and skeleton features. Second, to eliminate modality discrepancy
across different batches, we maintain memory queues of previous
batch features through the momentum updating mechanism and
propose MFD to integrate features in the whole training set by
memory-attention layers. These layers explore intramodality and
intermodality relationships between features from the current
batch and memory queues under the optimization of the optimal
transport (OT) method, which encourages the heterogeneous
features with the same identity to present higher similarity. Third,
to decouple the posture factors from representations, we introduce the PFD module to learn posture-unrelated features with the
assistance of the skeleton features. Besides, we perform subspace
orthogonal decomposition on both image and skeleton features
to separate the posture-related and identity-related information.
The posture-related features are adopted to disentangle the posture factors from representations by a designed posture-features
consistency (PfC) loss, while the identity-related features are concatenated to obtain more discriminative identity representations.
The effectiveness of DMPF is validated through comprehensive
experiments on two VIS-NIR pedestrian Re-ID datasets.
Index Terms— Modality-unrelated features learning, posture
factor disentangling, visible (VIS)-near-infrared (NIR) person
reidentification (Re-ID).

P

I. I NTRODUCTION
ERSON reidentification (Re-ID) seeks to match pedestrian identities across images captured by nonoverlapping

Manuscript received 7 August 2023; revised 18 November
2023 and 30 January 2024; accepted 28 March 2024. Date of publication
15 April 2024; date of current version 1 March 2025. This work was
supported in part by the National Natural Science Foundation of China under
Grant 62076262, Grant 61673402, Grant 61273270, and Grant 60802069;
and in part by the China Scholarship Council under Grant 202306380182.
(Corresponding author: Haifeng Hu.)
The authors are with the School of Electronics and Information
Technology, Sun Yat-sen University, Guangzhou 510006, China (e-mail:
huhaif@mail.sysu.edu.cn).
This article has supplementary downloadable material available at
https://doi.org/10.1109/TNNLS.2024.3384023, provided by the authors.
Digital Object Identifier 10.1109/TNNLS.2024.3384023

Fig. 1. Illustration of motivation and methodology regarding the proposed
DMPF. (a) VIS-NIR dataset presents challenges for learning identity-invariant
feature representations due to the modality and posture factors. (b) and (c) Proposed DMPF successfully disentangles the modality and posture factors from
the identity features by the proposed MFD and PFD modules, respectively.
Specifically, to eliminate modality discrepancies across the entire training set,
the MFD is proposed to integrate features from the current batch and memory
queues by adopting the designed memory-attention layers. To disentangle the
postures factor, the PFD is designed to use subspace orthogonal decomposition
between posture-related and identity-related information in the image and
skeleton features.

cameras [1]. This field has attracted substantial attention in
the past decade, primarily owing to its extensive applications
in security surveillance [2], criminal investigation [3], and
intelligent transportation [4], [5], [6]. Previous Re-ID methods
based on deep learning [1], [7], [8], [9] have achieved remarkable success and even surpassed human performance in certain
scenarios, benefiting from large-scale annotated datasets. However, these approaches assume that the pedestrians are captured
by visible (VIS) spectrum cameras in well-lit conditions,
which limits their applicability in realistic nighttime scenarios
owing to the VIS camera’s failure to capture target appearance
in low-light scenes [10]. Fortunately, most modern surveillance
systems are equipped with the capability to switch seamlessly
from VIS to near-infrared (NIR) spectrum cameras in lowlight conditions, since NIR cameras can explore information
on pedestrian appearance in low-light environments [11],
[12]. Nevertheless, the dramatic modality discrepancy between
VIS and NIR images remains unresolved by previous ReID methods [1], [7], [8], [9]. Therefore, striving to establish
correspondences of pedestrian identities across heterogeneous
images acquired from distinct spectra, VIS-NIR Re-ID has
emerged as a relevant and challenging task.
Compared to conventional Re-ID tasks, VIS-NIR Re-ID
presents more difficulty due to the substantial intraclass
disparities exhibited by heterogeneous images [13]. As illustrated in Fig. 1(a), the intraclass differences primarily stem
from two factors: modality discrepancy and posture variation.

2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION

5495

For the former factor, the NIR camera captures pedestrian
appearance details in low-light environments by utilizing NIR
spectral information, which inevitably sacrifices the color,
contrast, and other VIS spectral attributes in the captured
NIR images [14]. Consequently, the pronounced modality
discrepancy arises between heterogeneous images. Moreover,
pedestrians undergo significant pose variations over time,
leading to noticeable differences in the poses of the same
pedestrian captured at different moments. As a result, the
disparity in postures is notable for both NIR and VIS
images. Therefore, the modality and posture discrepancies
in heterogeneous images bring challenges in extracting the
identity-invariant features for Re-ID [15].
To learn identity-discriminative features, various VIS-NIR
Re-ID methods have focused on eliminating modality discrepancies by aligning VIS-NIR image styles from channel [16],
[17], [18] or spatial [19], [20], [21] dimension, known
as image alignment methods. However, both channel- and
spatial-based image alignment methods face challenges in
eliminating stylistic differences while preserving the appearance information of the original image [22]. On the one hand,
due to the complexity and nonlinearity of modality factors
in raw images, considerable stylistic differences still persist
between the NIR-like images generated by channel operations
and the real NIR images. On the other hand, the pixel
misalignment problem disrupts the appearance information
of pedestrians in the image generated by spatial operations, which hinders the learning of identity-invariant features.
To address these issues, recent GAN-based methods [23], [24],
[25] utilize a generative adversarial framework supervised
by identity-invariant and style-consistent losses to generate
images that exhibit consistent style and invariant appearance.
However, the introduction of the generator and discriminator in
the generative adversarial framework brings massive trainable
parameters, resulting in increasing training costs and computation complexity. Besides, the image alignment approaches
primarily focus on eliminating modality differences at the
image level, which neglects the isolation of heterogeneous
samples in the feature space. This limitation impacts the
extraction of modality-independent features and subsequently
affects the performance of the model in VIS-NIR Re-ID.
To bridge the modality gap between heterogeneous samples in the feature space, feature learning methods have
been developed to enhance the identity discriminability of
heterogeneous features by introducing discriminative local
features [26], [27]. These approaches effectively reduce the
influence of modality factors on identity matching. Differently,
to learn modality-independent features for heterogeneous
samples, feature alignment methods incorporate various heterogeneous losses to restrict the distribution or distance
between VIS and NIR features in each batch [28], [29], [30].
In addition, considering that modality factors embedded in
identity representations affect cross-modal identity matching,
disentangling-based approaches decouple modality information from identity representations by imposing orthogonal
constraints on identity-related and modality-related features
[13], [22], [31]. These feature learning methods have demonstrated their effectiveness in mitigating modality gaps in the
feature space. However, they primarily focus on eliminating
modality discrepancies at the batch-level, ignoring the modality disparities present in heterogeneous features outside the
batch, which leads to modality factors remaining in these

features. Furthermore, the aforementioned methods predominantly focus on the modality differences, while ignoring
another crucial difference between heterogeneous samples, i.e.,
posture discrepancy.
To eliminate the posture discrepancy, several approaches
generally utilize landmark [32] or parsing [33] information
of pedestrians to guide the learning of pose-invariant features. However, the extracted skeleton features encompass
both pose-related and identity-related information, which may
impact posture factor decoupling and undermine the discriminability of the identity features. Therefore, it is necessary to
disentangle them before introducing the skeleton features into
the representations. Moreover, landmark and parsing represent the information of keypoints location and shape profile,
respectively. Considering the complementary between them,
fusing the two types of information is beneficial for exploring
the pedestrian’s pose features. In addition, some methods
leverage keypoint information to extract local features, fostering the model’s ability to discern nuances among distinct
pedestrians. For instance, Liang et al. [34] proposed keypointguided modality-invariant discriminative learning (KMDL)
to extract discriminative fine-grained local representations
by adopting human keypoint information and an additional
graph convolutional network. Nevertheless, the concatenation
of local features unavoidably amplifies the dimensions of
feature representation, consequently elevating the time and
memory demands during model training. Moreover, during
the inference phase, the introduced graph-convolutional network extends the time required for feature extraction, thereby
impacting the real-time performance of the model in practical applications. Distinctly, the proposed method utilizes
keypoint information to learn pose-related features, aiding in
pose-independent identity feature extraction. Consequently, the
dimension of the feature representation is not significantly
increased, ensuring it does not impact efficiency during both
training and inference phases.
To learn identity-discriminative representations for crossmodality identity matching, we propose a disentangling
modality and posture factors (DMPFs) model to disentangle
modality and posture factors by fusing the information
of features memory and pedestrian skeleton. The former
preserves intraclass features for the entire training set,
facilitating the comprehensive elimination of modality
differences across the training set, while the latter fuses
landmark and parsing information of pedestrians, providing
pose-related information to effectively decouple pose factors
from identity features. Specifically, the DMPF comprises three
modules: three-stream features extraction network (TFENet),
modality factor disentanglement (MFD), and posture factor
disentanglement (PFD). First, to provide memory and skeleton
information for modality and posture factors disentanglement,
we designed a three-stream network named TFENet to extract
intraclass feature memories and skeleton features. Second,
as illustrated in Fig. 1(b), to eliminate modality discrepancy
across the training set, we propose the MFD to integrate batch
and memory features using designed memory-attention layers.
Optimizing by the optimal transport (OT) method, these layers
can effectively explore intramodal and intermodal relationships
by multihead self-attention [35], exposing augmented batch
features with the same identity that present high similarity.
As a result, we successfully decouple the modality factors
from identity features by the memory-attention layers.
Furthermore, to transfer the ability of modality difference

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

5496

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

elimination from the memory-attention layers to the backbone
network, we propose the modal-independent distillation
(MiD) loss to measure the similarity between the original
batch features and the augmented batch features. This loss
encourages the backbone network to learn from the memoryattention layers, thereby extracting modality-independent
feature representations effectively. Third, as illustrated in
Fig. 1(c), considering the complementary between landmark
and parsing information, we fuse both information to extract
skeleton features of pedestrians. Moreover, to handle the
entanglement between posture-related and identity-related
information in the images and skeleton features, we propose
the PFD module to extract posture-related and identity-related
features through subspace orthogonal decomposition [36].
The posture-related features are extracted to effectively
separate the posture factors by the proposed posture-features
consistency (PfC) loss, while the identity-related features
are utilized to improve the discriminability of image identity
representations by concatenating operation.
In brief, the main contributions are as follows.
1) Aiming at learning identity-invariant feature representations, the DMPF model integrates TFENet, MFD,
and PFD modules into a unified framework to
effectively disentangle modality and posture factors
from identity representations. In contrast to previous
approaches, the DMPF considers both modality and
posture discrepancies between VIS and NIR data, which
consequently learn identity-discriminative feature representations more efficiently.
2) To achieve the elimination of modality discrepancies
across the entire training set, we propose the MFD
module to integrate features from the current batch
and memory queues by adopting the designed memoryattention layers. Different from the previous method,
these layers progressively explore intramodality and
intermodality relationships between batch and memory
features under the optimization of the OT method,
thereby encouraging the extracted heterogeneous features with the same identity to present higher similarity.
Hence, we effectively decouple the modality factors
from identity features by the memory-attention layers.
3) To disentangle the postures factor from identity representation, we propose the PFD module to handle
the entanglement between posture-related and identityrelated information in the image and skeleton features
through subspace orthogonal decomposition. Distinctive
from existing methods, the PFD module decomposes
both image and skeleton features into pose-related and
identity-related features. The former is adopted to effectively separate the pose factors by the proposed PfC loss,
while the latter is utilized to improve the discriminability
of identity representations by concatenating operations.
The article proceeds as follows. Section II provides an
overview and analysis of related works. In Section III, the
proposed DMPF is presented in detail. Section IV showcases
the experimental results, demonstrating the superiority of
DMPF. Finally, Section V concludes this article.
II. R ELATED W ORK
A. Single-Modality Re-ID
Single-modality Re-ID aims to match identities across
images captured by nonoverlapping cameras in the VIS spectrum [37]. Existing methods have made remarkable progress in

learning identity-discriminative features using annotated data,
achieving human-level performance in specific scenarios [38],
[39], [40]. Some approaches focus on enhancing the network
structure and incorporating metric losses to learn global features that are discriminative for identity [39], [40]. Others
emphasize local feature learning by employing a uniform
division approach to extract features [41], [42]. For instance,
Sun et al. [41] proposed the part-based convolutional baseline
(PCB) to learn distinctive local features. However, uniform
division-based methods often encounter challenges related to
parts misalignment [43]. To address this challenge, Li et al.
[44] introduced an approach centered on local feature extraction through parts localization, which necessitates additional
annotations. In contrast, Zhu et al. [45] took a different
route, devising an adaptive division method to align the parts.
However, these approaches assume that the pedestrians are
captured by VIS spectrum cameras in well-lit conditions,
which substantially hampers their applicability in realistic
nighttime and indoor scenes.
B. VIS-NIR Re-ID
In contrast to single-modality Re-ID approaches [38], [39],
[40], [41], [42], [43], [44], [45], VIS-NIR Re-ID aims to establish identity matches between VIS and NIR images, which is
challenging due to significant modality discrepancy. Hence, the
primary challenge in VIS-NIR Re-ID is eliminating modality
discrepancy. To achieve the goal, existing VIS-NIR methods
can be categorized into image alignment methods [16], [17],
[20], [21], [23], [25], [46] and feature learning methods [26],
[28], [29], [30], [31], [47].
1) Image Alignment Methods: To address the dissimilarities
between heterogeneous images, image alignment techniques
strive to align the styles of the two modality images, thereby
simplifying the cross-modal Re-ID task into a single-modal
Re-ID task. These techniques can be categorized into three
groups: channel augmentation, image mixing, and GAN-based
methods.
First, to generate NIR-like images, channel augmentation
methods apply various operations, such as channel selection,
swapping, and weighting, to the RGB channels of VIS images.
For instance, some studies [16], [48] converted VIS images to
grayscale as an auxiliary modality, aiming to reduce modality
discrepancy. Others, like Ye et al., used channel exchanged
augmentation [17] and channel selection [49] techniques to
generate color-irrelevant images. However, the complexity and
nonlinearity of modality factors in raw images cause notable
stylistic differences between the generated NIR-like images
and real NIR images, affecting the learning of modalityunrelated features.
Differently, image mixing methods aim to bridge the stylistic gap between the two modality images by weighting
VIS-NIR image pairs in the spatial dimension, resulting in
style-aligned images. For instance, Hua et al. [20] proposed
a cross-modality channel mixup (CCM) augmentation that
combines a random RGB channel with an infrared image to
create a new mixed image while preserving identity information. Similarly, Kim et al. [19] introduced the PartMix
method, which synthesizes augmented samples by mixing part
descriptors across modalities to enhance the performance of
Re-ID models. Nevertheless, the pixel misalignment between
VIS and NIR images disrupts the appearance information of
the spatially generated images.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Distribution Aligned Semantics Adaption for
Lifelong Person Re-Identification
Qizao Wang1 , Xuelin Qian2 , Bin Li1 , Xiangyang Xue1*
1 Fudan University, Shanghai, China.

arXiv:2405.19695v2 [cs.CV] 8 Feb 2025

2 Northwestern Polytechnical University, Xi’an, China.

*Corresponding author(s). E-mail(s): xyxue@fudan.edu.cn;
Contributing authors: qzwang22@m.fudan.edu.cn; xlqian@nwpu.edu.cn;
libin@fudan.edu.cn;
Abstract
In real-world scenarios, person Re-IDentification (Re-ID) systems need to be
adaptable to changes in space and time. Therefore, the adaptation of Re-ID models to new domains while preserving previously acquired knowledge is crucial,
known as Lifelong person Re-IDentification (LReID). Advanced LReID methods
rely on replaying exemplars from old domains and applying knowledge distillation
in logits with old models. However, due to privacy concerns, retaining previous
data is inappropriate. Additionally, the fine-grained and open-set characteristics of Re-ID limit the effectiveness of the distillation paradigm for accumulating
knowledge. We argue that a Re-ID model trained on diverse and challenging
pedestrian images at a large scale can acquire robust and general human semantic knowledge. These semantics can be readily utilized as shared knowledge for
lifelong applications. In this paper, we identify the challenges and discrepancies associated with adapting a pre-trained model to each application domain
and introduce the Distribution Aligned Semantics Adaption (DASA) framework.
It efficiently adjusts Batch Normalization (BN) to mitigate interference from
data distribution discrepancy and freezes the pre-trained convolutional layers to
preserve shared knowledge. Additionally, we propose the lightweight Semantics
Adaption (SA) module, which effectively adapts learned semantics to enhance
pedestrian representations. Extensive experiments demonstrate the remarkable
superiority of our proposed framework over advanced LReID methods, and it
exhibits significantly reduced storage consumption. DASA presents a novel and
cost-effective perspective on effectively adapting pre-trained models for LReID.
The code is available at https://github.com/QizaoWang/DASA-LReID.

1

Samples

…

New
Model

(b) Comparison of using the
same pre-trained weights

(a) Knowledge distillation paradigm

…

Old
Model
Exemplars

New
Samples

Knowledge
Distillation
Loss

…

New
Model

(b) Knowledge distillation paradigm

(a) Results of using pre-trained weights

Fig. 1: (a) Comparison of average incremental performance for different methods
using the same pre-trained weights. (b) Conventional knowledge distillation pipeline
with exemplars. Our proposed DASA paradigm shows great superiority in LReID.

Keywords: Person re-identification, Lifelong learning, Distribution alignment,
Semantics adaption

1 Introduction
Person Re-IDentification (Re-ID) aims at recognizing the same pedestrian across disjoint cameras. With significant advancements in deep learning algorithms over the
past decade, it has demonstrated remarkable performance [1–7]. However, advanced
methods usually assume the pedestrian data is provided at once. In real-world scenarios, surveillance data accumulates continuously due to successive deployments and
ongoing applications. Re-ID data expands discretely, both spatially and temporally,
necessitating the adaptation of Re-ID models to new environments. Consequently, Lifelong person Re-IDentification (LReID) seeks to continuously adapt Re-ID models to
novel domains while preserving previously acquired knowledge during the incremental
learning process. Unfortunately, due to substantial variations between different Re-ID
domains, merely updating the model with new surveillance data can lead to a loss of
discriminative ability in previously learned domains. This phenomenon is commonly
referred to as the catastrophic forgetting problem.
Recently, we have witnessed the remarkable success of pre-training in computer
vision [8, 9], which also revolutionizes the field of person Re-ID. For example, Fu et al.
[10, 11] adopt unsupervised pre-training to prevailing Convolutional Neural Network
(CNN) architectures like ResNet [12] and achieve significant performance improvement. It is promising to leverage pre-trained models that can be obtained effortlessly
for real-world applications. Nevertheless, as observed in Fig. 1 (a), directly using the
pre-trained weights for model initialization yields limited benefits in the lifelong evolution scenario. Advanced LReID methods [13–16] follow the prevailing knowledge
distillation paradigm [17]. Thus we are motivated to think whether the performance
degradation stems from the inherent drawbacks of the knowledge distillation paradigm.

2

As depicted in Fig. 1 (b), existing LReID methods follow the distillation baseline
to distill knowledge at the logit level. This paradigm heavily relies on the learned
classifier. However, unlike conventional lifelong learning [17–20], LReID is inherently
a fine-grained open-set problem. The distillation paradigm would restrict the model’s
ability to incorporate valuable knowledge for recognizing unseen identities during inference. It also limits person Re-ID in real-world applications in two aspects. Firstly,
relying on replaying data of old tasks, namely exemplars [21–24], is impractical due
to strict privacy constraints associated with pedestrian data. Secondly, the number
of classes in the LReID datasets significantly exceeds those in conventional lifelong
learning tasks, such as ImageNet [25]. The high dimensional classifier would consume
great resources when saving it for knowledge distillation.
Therefore, to achieve a balance between knowledge preservation and updating with
minimal source consumption, and without using exemplars, we introduce the Distribution Aligned Semantics Adaption (DASA) framework, which presents a novel
LReID paradigm based on pre-trained models. In this paradigm, we do not rely on
knowledge distillation that uses saved old data and models for knowledge retention.
Instead, we exert the advantages of the pre-trained model in the lifelong learning process. Specifically, we carefully handle and leverage the two crucial techniques in CNN,
i.e., Batch Normalization (BN) and Convolutional (Conv) blocks. Firstly, BN effectively regularizes the model and captures data distributions, contributing to improved
learning [26, 27]. In our proposed paradigm, we tune BN effectively to deal with
domain distribution differences between pre-training and application. Secondly, Conv
layers specialize in encoding data patterns and semantics at different depths of the
network. It is expected that the Conv layers trained on a substantial amount of pedestrian data can acquire robust and generalizable human semantic knowledge [10, 11],
thereby serving as a valuable source of shared knowledge across domains. Therefore,
we propose freezing the pre-trained Conv layers to preserve shared person semantic
knowledge. However, general semantics are not capable of distinguishing pedestrians
in complex application scenarios. To adapt the acquired general knowledge to each
application domain effectively, we introduce the lightweight Semantics Adaption (SA)
module, which can efficiently aggregate, refine, and generate improved pedestrian representations. Adapting the general human semantics to more representative identity
semantics represents a process from coarse to fine, showing great ability to evolve the
model for lifelong application. Our contributions can be summarized as follows:
(1) We advocate taking advantage of the robust and general human semantic
knowledge acquired through large-scale pre-training to effectively adapt models in
application domains, presenting a new paradigm free of exemplars for LReID.
(2) We propose the Distribution Aligned Semantics Adaption (DASA) framework
for LReID. It eliminates the interference of domain distribution discrepancy between
pre-training and application, and incorporates the lightweight semantics adaption
module to aggregate and refine learned semantics for better pedestrian representations.
(3) Extensive experiments demonstrate the superiority of the proposed framework for LReID, achieving state-of-the-art results with significantly reduced resource
consumption compared to other advanced LReID methods.

3


exec
/bin/zsh -lc "pdftotext -l 3 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1

Diverse Representations Embedding for Lifelong
Person Re-Identification

arXiv:2403.16003v2 [cs.CV] 2 Apr 2024

Shiben Liu

, Huijie Fan*

, Qiang Wang

Abstract—Lifelong Person Re-Identification (LReID) aims to
continuously learn from successive data streams, matching individuals across multiple cameras. The key challenge for LReID
is how to effectively preserve old knowledge while incrementally
learning new information, which is caused by task-level domain
gaps and limited old task datasets. Existing methods based on
CNN backbone are insufficient to explore the representation
of each instance from different perspectives, limiting model
performance on limited old task datasets and new task datasets.
Unlike these methods, we propose a Diverse Representations Embedding (DRE) framework that first explores a pure transformer
for LReID. The proposed DRE preserves old knowledge while
adapting to new information based on instance-level and tasklevel layout. Concretely, an Adaptive Constraint Module (ACM)
is proposed to implement integration and push away operations between multiple overlapping representations generated by
transformer-based backbone, obtaining rich and discriminative
representations for each instance to improve adaptive ability
of LReID. Based on the processed diverse representations, we
propose Knowledge Update (KU) and Knowledge Preservation
(KP) strategies at the task-level layout by introducing the adjustment model and the learner model. KU strategy enhances the
adaptive learning ability of learner models for new information
under the adjustment model prior, and KP strategy preserves old
knowledge operated by representation-level alignment and logitlevel supervision in limited old task datasets while guaranteeing
the adaptive learning information capacity of the LReID model.
Extensive experiments were conducted on eleven Re-ID datasets,
including five seen datasets for training in order-1 and order-2
orders and six unseen datasets for inference. Compared to stateof-the-art methods, our method achieves significantly improved
performance in holistic, large-scale, and occluded datasets. Our
code will be available soon.

, Xiai Chen

, Yandong Tang

Fig. 1. An illustration of our DRE. Each shape denotes an instance, and
color indicates a different representation of the same instance. The blue circle
indicates the primary embedding representation after integration. At the top,
the generated diverse representation is implemented for integration and push
operation. Integrated primary embedding representation (blue circle) presents
rich body information. Multiple separate auxiliary embedding representations
(orange and purple) maintain discrimination of each instance. At the bottom,
based on instance-level, we introduce Knowledge Update (KU) and Knowledge Preservation (KP) strategies to improve model performance in preserving
old knowledge and adapting to new information in sequential tasks.

Index Terms—Lifelong learning, diverse representation, adaptive constraint learning, person re-identification.

This work is supported by the National Natural Science Foundation of
China (62273339, 61991413, U20A20200), and the Youth Innovation Promotion Association of Chinese Academy of Sciences (2019203). (Corresponding
author: Huijie Fan)
Shiben Liu is with the State Key Laboratory of Robotics, Shenyang
Institute of Automation, Chinese Academy of Sciences, Shenyang 110016,
China, and with the Institutes for Robotics and Intelligent Manufacturing,
Chinese Academy of Sciences, Shenyang 110169, China, and also with the
University of Chinese Academy of Sciences, Beijing 100049, China (e-mail:
liushiben@sia.cn).
Huijie Fan, Xiai Chen, Zhi Han, and Yandong Tang are with the State Key
Laboratory of Robotics, Shenyang Institute of Automation, Chinese Academy
of Sciences, Shenyang, 110016, China, and with the Institutes for Robotics
and Intelligent Manufacturing, Chinese Academy of Sciences, Shenyang,
110016, China (e-mail: fanhuiie@sia.cn; chenxiai@sia.cn; hanzhi@sia.cn;
ytang@sia.cn).
Qiang Wang is with the Key Laboratory of Manufacturing Industrial Integrated Automation, Shenyang University, and with the State Key Laboratory
of Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences,
Shenyang, 110016, China (e-mail: wangqiang@sia.cn).

, Zhi Han

I. I NTRODUCTION

P

ERSON re-identification (ReID) aims to retrieve the same
individuals across different camera views [1–4], which is
crucial in applications, such as intelligent surveillance [5–7]
and multi-camera collaborative tracking[8–10]. Most ReID
methods assume that the training datasets can be accessed all
at once, whereas ReID datasets are continuously incoming
in practical applications. Lifelong learning is necessary for
person re-identification in continuous datasets across the
scene.
Most ReID methods [11, 12] learn robust features on
specific distribution datasets through pre-trained networks.
Unlike ReID, the main purpose of LReID is to facilitate model
to efficiently preserve old knowledge while adapt to new
information in sequential tasks. The rehearsal-based [13, 14]
methods reserve a memory buffer for storing limited instances
from old tasks, and recall these instances when learning
new tasks, which ignore model representation capabilities

2

on limited old task datasets. A framework of learnable
and consistent features [15] is proposed that provides
complementary knowledge for diverse representations
generation, improving the discrimination and adaptability of
the LReID model. Inspired by knowledge distillation [16],
several methods [17, 18] in LReID incorporate distillation
loss between teacher networks with prior knowledge of
old tasks and student networks with adaptive learning of
current tasks to mitigate task-level domain gaps. These
methods all employ fixed-weight teacher networks, forcing
the model to reduce its adaptability to new tasks. Yu et al.
[19] introduced a knowledge refresh scheme that updates the
memory model with a smaller learning rate, which turns the
memory model into a dynamic teacher. The above CNN-based
methods insufficiently explore the discriminative and rich
representations of each instance, limiting the performance
of the LReID model in preserving old knowledge while
adapting to new information. We consider that the CNN
backbone (ResNet50, ResNet101) forces the introduction of
multiple branches to obtain diverse representations that are
independent and uncontrollable from each other in the LReID
model.
Recently, Vision Transformer (ViT) [20] and Data-efficient
Image Transformers (DeiT) [21] have achieved significant
success in image recognition [22, 23], detection [24, 25], and
restoration [26, 27]. In addition, TransReID [28] indicates
that pure transformer is effective in feature extraction for
ReID, because multi-head self-attention of the transformer
frame captures long-range dependencies and drives the
model to attend diverse human body parts. DC-Former[29]
gets multiple diverse and compact embedding subspaces.
Each embedding of these compact subspaces is more robust
and discriminative to identify similar classes. Although the
above methods have gained significant advantages in ReID,
transformers still need to be specifically designed for LReID
to preserve old knowledge while adapting new information.
Side information embeddings [28] (such as camera and
viewpoint-specific information) constantly change as the
number of tasks increases in LReID, leading to bias in the
LReID model. We introduce Maximum Embedding (ME)
to replace side information embeddings, focusing on critical
regions of the input sequence and reducing background
interference. Meanwhile, the class token of transformer is
transformed into a global representation of each instance in
the ReID task. We introduce multiple class tokens embedding
to learn multiple overlopping representation of each instance
in the LReID task. Multiple class tokens and patches are
concatenated in the first dimension and sent to the transformer
for training on the seen dataset. Therefore, we construct a
strong backbone based on a pure transformer in LReID, as
an adjustment model and learner model structure.
Based on the above improved transformer backbone,
we propose a diverse representation embedding that first
explores the pure transformer for LReID, named DRE.
Specifically, multiple overlopping representations are
generated by multiple class tokens of transformer, as
shown in Figure 1 (instance-level). In this paper, we set
up multiple overlopping representations. One is the primary

embedding representation for learning rich body information,
the other is multiple auxiliary embedding representations
for providing discrimination of each instance. We propose
an Adaptive Constraint Module (ACM) to implement
integration and push away operations between multiple
overlopping representations. Multiple auxiliary embedding
representations minimize overlapping elements by orthogonal
operation. We adaptively integrate multiple discriminative
auxiliary embedding representations into primary embedding
representations to enhance the representational capacity of
the LReID model, which provides rich body information for
classification. At the instance level, diverse representation
generation presents richness and discrimination of each
instance, remarkably improving feature learning and label
matching processes for models with the same ID. Based
on instance-level, we propose Knowledge Update (KU) and
Knowledge Preservation (KP) strategies at the task-level
layout by introducing the adjustment model and the learner
model. KU strategy enhances the adaptive learning ability of
learner models for new information under adjustment model
prior. Because the knowledge update strategy is prone to adapt
to new tasks, the knowledge preservation scheme is proposed
to preserve old knowledge operated by representation-level
alignment and logit-level supervision in limited old task
datasets while guaranteeing the adaptive learning information
capacity of the LReID model. The main contributions of this
paper are summarized as follows:
• We propose a diverse representation embedding framework that first exploits the transformer-based backbone
for LReID tasks. Maximum embedding Multiple class
tokens are introduced to reduce bias and generate multiple
representations of each instance.
• We design an adaptive constraint module to maintain rich
primary embedding and discriminative auxiliary embedding representations in all seen domains, improving the
adaptive capability of the LReID model.
• Based on diverse representations, knowledge update
facilitate learning new information capability of the
learner model, and knowledge preservation leverages
representation-level alignment and logit-level supervision
to preserve old knowledge while guaranteeing the learning information ability of the LReID model.
• Extensive experiments demonstrate that the proposed
DRE outperforms state-of-the-art methods on both seen
and unseen datasets. In general, our DRE outperforms
performance with CNN-based methods.
II. R ELATED WORK
A. Person Re-Identification
Research in person re-identification (ReID) make significant
achievements over the years. These methods can be broadly
divited into four kategories. First, CNN-bsed approaches
[30–33] extract discriminative features from pedestrian
images to effectively address the spatial complexities. Crossentropy loss [34] and triplet loss [35] are commonly applied
for training CNN to learn better representation. Second,

3

transformer-based methods [28, 36–38] make further progress
in ReID for their ability to capture long-range dependencies
in feature maps. In these methods, self-attention mechnism
is integrated into ReID to enhance the extraction of relevant
information. Third, domain adaptation and transfer learning
techniques [39–42] are also explored to improve model
generalization, particularly when dealing with variations
in lighting, viewpoints, and camera sources. Fourth, textto-image methods [4, 43, 44] leverage textual descriptions
and image data to improve recognition accuracy and model
performance, enabling more effective matching of individuals
in different contexts by combining text descriptions with
image-based features.
B. Lifelong Person Re-Identification
Lifelong Person Re-Identification (LReID) faces a
formidable challenge, aiming to address the evolving nature
of person identification across various scenarios and domains.
Some works [14, 45, 45, 46] are proposed to tackle the
issue of adapting ReID models over time while retaining
knowledge gained from previous experiences. Generally,
Pu et al. [14] proposed learnable knowledge graphs that
adaptively facilitate the mutual exchange of new and old
knowledge, thus achieving knowledge accumulation. Some
works [15, 17, 18, 47] aim to extract rich and discriminative
representation, mitigating the risk of knowledge forgetting.
Pu et al. [47] proposed meta-reconciliation normalization
(MRN) for mining meta-knowledge shared across different
domains. Meanwhile, ConRFL [15] maintains learnable and
consistent features across all seen domains, which improves
the discrimination and adaptation ability of the LReID
model. In addition, some methods [14, 17, 19] mitigate
catastrophic forgetting and enhance model accuracy by using
rehearsal-based strategies with images stored from previous
tasks.
C. Diverse representation learning
Representation learning methods employ specific modules
or branches to mine multiple discriminative features of each
instance, which minimize intra-class distance and maximize
inter-class distance. It effectively improves model discrimination for identifying similar classes. DEEN [32] can effectively
learn the informative representations by generating diverse
embeddings subspace of each instance. DC-Former [29] employs multiple class tokens in vision transformer to generate
multiple representations, which increases the identity density
of embedding space to help model improve its discrimination
for identifying similar classes. Sun et al. [18] employed
adaptively-chosen patches to exploit body local information.
Ge et al. [17] generated pseudo-task features by a pseudo-task
transformation module to complement the limited exemplars.
These LReID methods employ CNN as a feature extractor,
which insufficiently exploits rich and discriminative representations of each instance. We consider that diverse representations generated by multiple class tokens with transformerspecific structure play an important role in maintaining a

trade-off between preserving old knowledge and adapting to
new information. Thus, we first propose a transformer-based
diversity representation embedding for lifelong person reidentification.
III. P ROPOSED M ETHOD
A. Overview
To preserve old knowledge while adapting new information
in the face of limiting old task datasets and task-level domain gaps, we propose a Diverse Representation Embedding
(DRE) framework based on improved transformer backbone
for LReID. The input images are mapped into three representations using a transformer-based backbone. Three representations, including a primary embedding representation
and multiple auxiliary embedding representations, are operated
by Adaptive Constraint Module (ACM) to maintain richness
and discrimination of each instance. Knowledge Update (KU)
and Knowledge Preservation (KP) are proposed at the tasklevel layout by introducing the adjustment model and the
learner model. We define the problem of lifelong person reidentification in this section. Specifically, continuous person
re-identification datasets E = {E t }Tt=1 are collected in dift
t
ferent environments. E t = {Etrain
and Etest
} indicates the
training datasets and corresponding test datasets in t-th step,
respectively. The training datasets Etrain are fed into the
model in sequence for training. To store a limited number of
instances from each old task dataset, we establish a memory
buffer M t .
The DRE consists of an adjustment model Φta and a learner
model Φtl to extract diverse representations of each instance.
The adjustment model maintains the priority of old tasks,
and the learner model is responsible for adaptive learning of
information for new instances. An adjustment model and a
learner model both use improved transformer backbones. ϕta
and ϕtl serve as classifier heads for the adjustment and learner
models, providing logits of each instance for recognition, as
shown in Figure 2. The outputs of old instances xo and new
instances xn through adjustment model Φta and learner model
Φtl are as follows:
Pln , (Anl )S = Φtl (xn ); Pan , (Ana )S = Φta (xn )
Plo , (Aol )S = Φtl (xo );

Pao , (Aoa )S = Φta (xo )

(1)

Where Pa and Pl ∈ RB×D denote the primary embedding
representations from the adjustment and learner models. (Aa )S
and (Al )S ∈ RB×D are auxiliary embedding representations
from the adjustment and learner models. S is the number of
auxiliary embedding representations.
B. Transformer-based structure
Currently, LReID methods based on CNN backbone force
the introduction of multiple branches to obtain diverse representations that are independent and uncontrollable from each
other in sequential task datasets. Multiple class tokens of
transformer better fit our idea of diverse representations. Thus,
we first propose a transformer-based backbone to generate
diverse representations of each instance in LReID, as shown


exec
/bin/zsh -lc "pdftotext -l 3 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 189 (2025) 107602

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

DiverseReID: Towards generalizable person re-identification via Dynamic
Style Hallucination and decoupled domain experts
Jieru Jia ∗, Huidi Xie, Qin Huang , Yantao Song , Peng Wu
Institute of Big Data Science and Industry, Shanxi University, Taiyuan 030006, China
School of Computer and Information Technology, Shanxi University, Taiyuan, China
Engineering Research Center for Machine Vision and Data Mining of Shanxi Province, Shanxi University, Taiyuan 030006, China

ARTICLE

INFO

Keywords:
Person re-identification
Domain generalization
Data augmentation
Mixture of experts

ABSTRACT
Person re-identification (re-ID) models often fail to generalize well when deployed to other camera networks
with domain shift. A classical domain generalization (DG) solution is to enhance the diversity of source
data so that a model can learn more domain-invariant, and hence generalizable representations. Existing
methods typically mix images from different domains in a mini-batch to generate novel styles, but the mixing
coefficient sampled from predefined Beta distribution requires careful manual tuning and may render suboptimal performance. To this end, we propose a plug-and-play Dynamic Style Hallucination (DSH) module that
adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
dynamically measured with the reciprocal of Wasserstein distances. This approach not only reduces the tedious
manual tuning of parameters but also significantly enriches style diversity by expanding the perturbation space
to the utmost. In addition, to promote inter-domain diversity, we devise a Domain Experts Decoupling (DED)
loss, which constrains features from one domain to go towards the orthogonal direction against features from
other domains. The proposed approach, dubbed DiverseReID, is parameter-free and computationally efficient.
Without bells and whistles, it outperforms the state-of-the-art on various DG re-ID benchmarks. Experiments
verify that style diversity, not just the size of the training data, is crucial for enhancing generalization.

1. Introduction
Person re-identification (re-ID) aims to match pedestrian images of
the same identity in a non-overlapping camera system. Despite their
almost saturated performance in the closed-world supervised learning
testbed, most current methods suffer from drastic performance degradation if applied to a different camera network with distribution shifts.
This prevents a single re-ID system from being successfully applied to
real-world scenarios in an off-the-shelf manner.
To overcome this limitation, domain generalization (DG) is a
promising solution with the goal of training a model that generalizes
well to arbitrary unseen domains out-of-the-box, without any model
re-training or fine-tuning. This is a more challenging but practical task
compared with unsupervised domain adaptation (UDA) (Ge, Zhu, Chen,
Zhao, et al., 2020; Zhang, Li et al., 2022), as it does not require data
collection or any prior knowledge about the unseen target domain for
model training purposes. Moreover, unlike most DG methods (Huang
et al., 2023; Sun, Liu, Liu, Li, & Chu, 2023; Wang, Zhang, Lei, & Zhang,
2023) which assume a homogeneous environment where different
domains share the same label space, DG re-ID is a more challenging

heterogeneous problem, in that there is no identity overlap between
source and target domains.
To enhance the robustness to domain shifts, existing DG methods intend to learn domain-invariant representations across source
domains, mainly via data augmentation (Yang et al., 2024; Zhang,
Cisse, Dauphin, & Lopez-Paz, 2018; Zhou, Yang, Qiao & Xiang, 2020),
normalization techniques (Choi, Kim, Jeong, Park, & Kim, 2021; Jia,
Ruan, & Hospedales, 2019; Jin, Lan, Zeng, Chen, & Zhang, 2020), novel
network architecture designs (Dou, Wang, Li, & Wang, 2023; Liao &
Shao, 2020; Zhang, Dou, Yu & Li, 2022), etc. Another popular way
to address the DG problem is to develop novel learning strategies,
which can be roughly divided into but not limited to: (a) adversarial
learning (Lin, Li, & Kot, 2020); (b) mixture of experts (Dai, Li, Liu,
Tong, & Duan, 2021; Xu, Liang, He, & Sun, 2022) that learn a system
composed of many separated networks (experts); (c) meta-learning (Ni
et al., 2022; Zhao et al., 2021) that learns general knowledge by
constructing meta-learning tasks to simulate domain shift.
Among recent DG methods, feature-based data augmentation approaches like MixStyle (Zhou, Yang, Qiao et al., 2020), pAdaIN (Nuriel,

∗ Correspondence to: 030006, #92 Wucheng Street, Taiyuan, Shanxi Province, China.

E-mail address: jierujia@sxu.edu.cn (J. Jia).
https://doi.org/10.1016/j.neunet.2025.107602
Received 5 December 2024; Received in revised form 24 March 2025; Accepted 7 May 2025
Available online 24 May 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 189 (2025) 107602

J. Jia et al.

Benaim, & Wolf, 2021), and DSU (Li et al., 2022) et al. have exhibited
astonishing performance, which substantially stems from the ability to
avoid overfitting to source data by introducing channel-level style perturbations to emulate potential domain shifts. Despite their popularity
and superiority in domain generalization tasks, existing style augmentation methods are limited in: (1) when synthesizing novel styles, the
weights for two instances are obtained by sampling from the manually
designed distributions, which requires careful hyper-parameter tuning
and may be inadequate to explore a broader perturbation space. More
importantly, any fixed value of the parameters would become suboptimal at some iteration during training; (2) their success relies on the
enriched diversity of newly produced samples, which could not always
be guaranteed in practice since the generated samples may remain
closely to existing ones in the feature space. The desirable superiority
disappears when synthesized sample diversity is insufficient.
This paper attempts to solve the above two crucial issues and
proposes a novel framework dubbed DiverseReID, which expands both
inter-domain and intra-domain style space to maximize the source
diversity for generalized re-ID. Particularly, to resolve the first issue
and achieve intra-domain style diversification, we propose a Dynamic
Style Hallucination (DSH) module that dynamically generates new
styles by mining the relations across sample pairs in an adaptive
manner. Specifically, we leverage the Wasserstein distance to estimate
the style distribution discrepancy between the input image pairs, and
the reciprocal of similarity distance is utilized as the mixing coefficients. The DSH module could alleviate the tedious manual tuning of
parameters. Moreover, instead of a constant coefficient among all the
sample pairs, the DSH module adaptively adjusts the weighting factors
according to the features of different instances, which can expand a
broader perturbation space and significantly enrich the intra-domain
style diversity.
For the second issue, to achieve inter-domain variety, we maintain
a domain-specific memory for each domain (also termed as domain experts) and devise a Domain Experts Decoupling (DED) loss to constrain
the features from one domain to go towards the orthogonal direction
against other domains. The DED loss is implemented by minimizing the
squared Frobenius norm of the diagonal elements in the correlation
matrix. In this way, the domain-specific features are guaranteed to
be dissimilar from each other, which provides a substantial degree
of artificial domain shift and thereby remarkably reduces the risk of
overfitting to source training data. In summary, our main contributions
can be summarized in three aspects:

methods focus on diversifying the source data by artificially inducing
style perturbations in the training data. For instance, MixStyle (Zhou,
Yang, Qiao et al., 2020) combines the statistics of two instances to
synthesize novel samples. DomainMix (Wang, Liao, Zhao, Kang, & Shao,
2021) combines a labeled synthetic dataset with unlabeled real-world
data to learn domain-invariant features. PartMix (Kim, Kim, Park, Park,
& Sohn, 2023) synthesizes the augmented samples by mixing the part
descriptors across modalities. DACS (Yang et al., 2024) designs a style
transformation model with diversity and authenticity constraints to
generate diverse and authentic data.
Domain-invariant Representation learning approaches exploit
normalization techniques or novel network architecture designs to reduce the domain gap in the feature space. For example, DualNorm (Jia
et al., 2019) exploits instance normalization (IN) to filter out variations
in style statistics in earlier layers. SNR (Jin et al., 2020) filters out
identity-irrelevant interference and alleviates the loss of discriminative
information with an attention mechanism. MetaBIN (Choi et al., 2021)
combines batch normalization (BN) and IN with learnable balancing
parameters. DTIN (Jiao et al., 2022) introduces a novel normalization
module by employing dynamic convolutions. As for network architecture designs, QAConv (Liao & Shao, 2020) constructs query-adaptive
convolution kernels to find local correspondences in feature maps.
ACL (Zhang, Dou et al., 2022) proposes a Cross-Domain Embedding
Block to capture the domain-invariant and domain-specific features.
ISR (Dou et al., 2023) puts forward a reliability-guided contrastive
learning method to enable positive pairs to dominate the representation
learning process. MLRAT (Lin et al., 2024) proposes a multi-level
relation-aware Transformer to extract fine-grained features by learning
structural relations among key patches. ReNorm (Nie, Ding, Zhou, & Li,
2024) resolves statistical mismatch in normalization layers by mixing
statistics and freezing affine parameters, helping the model adapt to
cross-domain distribution shifts.
Lastly, some works focus on designing novel learning strategies to
learn domain-agnostic representations. For instance, MMFA-AAE (Lin
et al., 2020) employs the adversarial auto-encoder module to extract
domain-invariant features. ADNMT (Zhang, Li, Ma, & Wang, 2024)
introduces an asymmetric dual-network mutual teaching framework
to optimize pseudo-labels’ quality. Mixture of Experts (MoE) provides
another promising solution to deal with the domain gap. For instance,
META (Xu et al., 2022) functions as an ensemble of multiple experts
sharing all the parameters except for the domain-specific BN layers.
RaMoE (Dai et al., 2021) puts forward a voting network to integrate
all the experts’ features with domain relevance. Meta-learning methods
focus on simulating the training/testing discrepancy. For instance,
M3L (Zhao et al., 2021) proposes a meta BN module to integrate
domain information from the meta-train datasets into the meta-test
features. MDA (Ni et al., 2022) designs a meta-learning strategy to
simulate train–test domain shifts with prior distribution alignment
and distribution-guided refining. As an interesting alternative to metalearning, Tan, Ding, Wang, Gong, and Jia (2023) propose an efficient
Interleaved Learning (IL) technique, which simulates the domain gap
by introducing a shift between the feature extractor and classifier.
Despite the exciting progress in recent years, existing methods
often overlooked the importance of data diversity in enhancing generalization. Different from previous data augmentation methods that
mainly focus on enlarging the size of the training data, our research
underscores the effect of data diversity in DG, and expands both interdomain and intra-domain style space to maximize the source diversity.
The proposed DiverseReID is efficient but not sophisticated, and its lowcost and easy-to-use merit will be of broad interest and use for driving
future research.

• We propose a new Dynamic Style Hallucination module that
dynamically generates novel styles by mining the relations across
sample pairs in an adaptive manner, which significantly enriches
intra-domain diversity and spares the requirement of manual
parameter tuning.
• We develop a novel Domain Experts Decoupling loss for domain
generalization that guarantees the disparity between individual
source domains’ styles, which is highly efficient in disentangling
inter-domain representations and exploiting the corresponding
domain’s powerful characteristics sufficiently.
• We perform extensive experiments on multiple DG re-ID benchmarks, which show that our method consistently outperforms
state-of-the-art methods by significant margins.
2. Related work
2.1. Domain generalizable person re-ID
The goal of generalizable person re-identification is to learn a model
from one or multiple related source domains that generalizes well
to unseen target domains. Existing methods realize this goal from
various perspectives, including data augmentation, domain-invariant
representation learning, training strategy, etc. Data augmentation

2.2. Style manipulation for domain generalization
The proposed work is mainly dedicated to diversifying the styles
of the source data, i.e. data augmentation, which aids in improving
2

Neural Networks 189 (2025) 107602

J. Jia et al.

the model’s generalization ability by simulating a broader range of
scenarios during training. For instance, MixStyle (Zhou, Yang, Qiao
et al., 2020) combines the statistics of two instances to synthesize novel
samples. pAdaIN (Nuriel et al., 2021) swaps feature statistics between
samples in one batch. DSU (Li et al., 2022) imposes disturbance on the
original feature statistics by adding Gaussian noise. MOEX (Li, Wu, Lim,
Belongie, & Weinberger, 2021) encourages the model to extract training
signals from the moment information. SagNet (Nam, Lee, Park, Yoon,
& Yoo, 2021) introduces a style randomization module by interpolating
the feature statistics between different examples. SHADE (Zhao, Zhong,
Zhao, Sebe, & Lee, 2022) presents a style hallucination module that
selects basis styles from the source distribution, and simulates potential
style variations through combination and perturbation. SPC (Huang
et al., 2023) projects unseen styles into the style representation space
built on known domains. SIRL (Chang, Yang, Luo, Ji, & Wang, 2023)
devises an Adversarial Style Synthesis module that consists of a multibranch style generation and fusion strategy. DSM (Li, Song, Ni, & Shen,
2023) maintains a single-domain style queue for each source domain to
enable style mixing. DACS (Yang et al., 2024) designs a style transformation model with diversity and authenticity constraints, enabling it
to generate styles that closely resemble the true data distribution.
While delivering superior performance, there are several limitations
with existing style augmentation works: (1) their efficacy actually heavily depends on delicately designed weight parameters, whose optimal
combinations are time-consuming to acquire; (2) existing methods produce new styles via merely mixing the existing styles from few observed
source domains with the linear interpolation mechanism, leading to
insufficient diversity which may hamper the model optimization. In
this paper, we explicitly consider and explore the style diversity in
improving the model’s generalization ability. Unlike previous methods that generate new data with weights sampled from pre-defined
distributions with a fixed parameter, we devise a novel strategy to
ensure sample diversity by formulating the weights in an adaptive
manner, which is capable of broadening the perturbation space and
providing a substantial degree of style diversity. The proposed DSH
module remarkably boosts the DG re-ID performance and can be easily
integrated with other methods in a plug-and-play manner.

previous works in that the DED loss is proposed to achieve inter-domain
diversification, which is equally crucial as intra-domain style diversity
for enhancing DG performance. Experiments and ablation studies show
that DED loss is orthogonal to the intra-domain DSH module and they
can be combined to potentially further boost the re-ID performance.
3. Methodology
3.1. Preliminaries
In the typical DG re-ID setting, we assume 𝐾 labeled source domains
{
}
{
}𝐾
 = 1 , 2 , … 𝐾 , where each source domain 𝑘 = 𝑘 , 𝑘 𝑘=1
consists of 𝑁𝑘 Image–label pairs. The label spaces for 𝐾 source sets
are disjoint. The core objective of DG re-ID is to learn a model parameterized by 𝜃 that is capable of predicting accurate identity labels on
unseen target domain  .
3.2. Overview
The overall process of the proposed framework is illustrated in Fig.
1, which includes a feature extractor 𝑓𝜃 and constructs an individual
memory-based classifier for each source domain. This paper employs
a domain-specific sampling strategy where each mini-batch contains
samples from the same domain. For a source domain 𝑘 with 𝑃𝑘
identities, the domain-specific memory 𝑘 has 𝑃𝑘 slots, where each
slot saves the feature centroid of the corresponding domain. 𝑘 is
initialized with features averaged of the corresponding identity. In
the forward propagation, the similarities between features produced
by 𝑓𝜃 and centroids of 𝑘 are used to compute the identification
loss. To maximize the dissimilarity of representations, the DED loss is
proposed to push apart features from one domain from those of other
domains. In the backward propagation, the network is optimized by
the combination of identification loss and DED loss. At the end of each
training iteration, the memory modules are updated with the features
in the current mini-batch.
3.3. Domain experts decoupling

2.3. Diversity via feature decorrelation

During the forward propagation, any image 𝑥𝑘𝑖 (where 𝑥𝑘𝑖 denotes
the 𝑖𝑡ℎ image sampled from the 𝑘𝑡ℎ dataset 𝑘 ) is passed through the
feature extractor to obtain the L2-normalized feature 𝑓𝑖𝑘 = 𝑓𝜃 (𝑥𝑘𝑖 ).
We calculate the non-parametric memory-based identification loss as
follows:
(
)
𝑁𝑘
∑
exp 𝑓𝑖𝑘 𝑐+𝑘 ∕𝜏
𝑘 = −
log ∑
(1)
(
)
𝑃𝑘
𝑘 𝑐 𝑘 ∕𝜏
𝑖=1
exp
𝑓
𝑖
𝑗
𝑗=1

Emerging evidence suggests that the correlations between features
affect or even impair the model prediction. Feature decorrelation is
a promising way to separate the explanatory factors of variations
and learn representations that are more resilient to complex variants.
Recent works realize this goal from various perspectives. For instance,
DecAug (Bai et al., 2021) decomposes the high-level features into
category-related and context-related features to deal with the diversity shift. RaMoE (Dai et al., 2021) proposes a decorrelation loss by
minimizing the L2-norm of point-wise product between features from
various domains. ABD-Net (Chen et al., 2019) incorporates a spectral
value difference orthogonality (SVDO) regularization to reduce the
correlation between learned features by constraining the condition
number of the weight Gram matrix. DACS (Yang et al., 2024) achieves
decorrelation by enlarging the distributional discrepancies between the
original and generated data with the Wasserstein distance.
Different from the above methods that require complex architectures or high computational costs, we investigate the memory-based
feature decorrelation with a novel Domain Experts Decoupling (DED)
loss, which provides a flexible and low-cost solution to enrich diversity by minimizing the correlation among the source domain experts.
Unlike DecAug (Bai et al., 2021) and ABD-Net (Chen et al., 2019),
which rely on explicit feature decomposition or matrix regularization,
our method achieves feature decorrelation by enforcing orthogonality
among the features stored in memories. This in vivo regularization not
only reduces computational overhead but also enhances adaptability
to diverse domain shifts. Moreover, our motivation is different from

where 𝑐+𝑘 refers to each centroid of the positive class prototype corresponding to 𝑓𝑖𝑘 , and 𝑐𝑗𝑘 denotes the centroid of one class prototype,
which could be either positive or negative. Here, 𝜏 is the temperature
factor that controls the scale of similarities. The memory-based identification loss computes the average loss across 𝐾 source domains, where
𝑘 is the loss for domain 𝑘, aiming to classify 𝑓𝑖𝑘 into its own identity
and the total identification loss is a combination on all source domains:
1 ∑

(2)
𝐾 𝑘=1 𝑘
Next, in order to impose inter-domain diversity and keep source
domains’ discriminative power, we propose the Domain Experts Decoupling loss to enable the feature disentanglement among various source
domains. Let 𝑓 𝑘 and 𝑓 𝑗 (𝑗 ≠ 𝑘) be representations from any two source
domains (the subscript of the image index is omitted for simplicity), we
first compute their cross-correlation matrix along the batch dimension:
∑ 𝑘 𝑗
𝑏𝑓 𝑓
𝑘𝑗 = √
(3)
√
∑ ( 𝑘 )2 ∑ ( 𝑗 )2
𝑏 𝑓
𝑏 𝑓
𝐾

𝑀 (𝜃) =

3


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 590ms:
1

Domain Consistency Representation Learning for Lifelong Person
Re-Identification

arXiv:2409.19954v4 [cs.CV] 3 Sep 2025

Shiben Liu

, Huijie Fan

, Qiang Wang

, Weihong Ren
Member, IEEE

Abstract—Lifelong person re-identification (LReID) exhibits a
contradictory relationship between intra-domain discrimination
and inter-domain gaps when learning from continuous data.
Intra-domain discrimination focuses on individual nuances (i.e.,
clothing type, accessories, etc.), while inter-domain gaps emphasize domain consistency. Achieving a trade-off between maximizing intra-domain discrimination and minimizing inter-domain
gaps is a crucial challenge for improving LReID performance.
Most existing methods strive to reduce inter-domain gaps through
knowledge distillation to maintain domain consistency. However,
they often ignore intra-domain discrimination. To address this
challenge, we propose a novel domain consistency representation
learning (DCR) model that explores global and attribute-wise representations as a bridge to balance intra-domain discrimination
and inter-domain gaps. At the intra-domain level, we explore
the complementary relationship between global and attributewise representations to improve discrimination among similar
identities. Excessive learning intra-domain discrimination can
lead to catastrophic forgetting. We further develop an attributeoriented anti-forgetting (AF) strategy that explores attribute-wise
representations to enhance inter-domain consistency, and propose
a knowledge consolidation (KC) strategy to facilitate knowledge
transfer. Extensive experiments show that our DCR achieves superior performance compared to state-of-the-art LReID methods.
Our code is available at https://github.com/LiuShiBen/DCR.

P

ERSON re-identification (ReID) aims to retrieve the
same individual across multiple cameras in a largescale database by using uni-modal architectures such as
convolutional neural networks (CNN) [1]–[3] or vision
transformers (ViT) [4]–[6]. However, when ReID models

, Yang Cong

Global
representations center
attribute category
ID 2

ID 1

ID 3

ID 4

Step t-1

, Senior

class boundary
domain boundary
ID 6

ID 5

Step t
Knowledge
Distllation
Minimize
inter-domain gaps
Step t

Step t-1

(a) Existing Methods
Man/woman
Shorts
Trousers
Backpack
Shoulder bag
Long sleeve
Short sleeve

Index Terms—Lifelong person re-identification, attribute-text
generator, text-image aggregation, domain consistency representation.

I. I NTRODUCTION

, Yandong Tang

Step t-1
Step t
Minimize inter-domain gaps
Maximize intra-domain discrimination
(b) Ours
Fig. 1. Comparison between our method and existing methods. (a) Existing
methods [9], [10] leverage knowledge distillation to minimize inter-domain
gaps but ignore intra-domain discrimination, which limits the LReID model’s
ability to learn new knowledge. (b) Our method explores domain consistency
representations as a bridge to achieve a trade-off between maximizing intradomain discrimination and minimizing inter-domain gaps, enhancing the
LReID model’s anti-forgetting and generalization capabilities.

Text
Encoder

This work is supported by the National Natural Science Foundation
of China (62273339, U24A201397), the Key Research and Development
are applied to continuous datasets collected by video-based
Program of Liaoning (2024JH2/102400022)
and the
LiaoNing Revitalization
Attribute-text
generator
Text-guided
aggregation
network
(TGA)
Globa
monitoring systems
[7], [8], they
exhibit notable
performance
Talents Program (XLYC2403128). (Corresponding author: Huijie Fan)
Shiben Liu is with the State KeyALaboratory
Shenyang Institute
limitations. As a result, recent works have focused on the representa
photo ofof aRobotics,
woman
of Automation, Chinese Academy
of Sciences,
110016,
(LReID),
wearing
a longShenyang
sleeved
top, China, and practical problem of lifelong person identification[CLS]
also with the University of Chinese Academy of Sciences, Beijing 100049,
which
maintains
strong
performance
with
continuously
and
a
trousers,
while
China (e-mail: liushiben@sia.cn).
Huijie Fan, and Yandong Tang
are withathe
State Key Laboratory of
updated data streams.
carrying
backpack.
Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences,
[CLS] At present, lifelong person re-identification (LReID) suffers
Shenyang, 110016, China (e-mail: fanhuijie@sia.cn; ytang@sia.cn).
from the challenge of balancing the
PFManti-forgetting of old
Qiang Wang is with the Key Laboratory of Manufacturing Industrial Integrated Automation, Shenyang University, and with the State Key Laboratory
knowledge and learning new knowledge. Specifically, there
of Robotics, Shenyang Institute Attribute
of Automation, Chinese Academy of Sciences,
are two main issues to solve this challenge. 1) Intra-domain
Shenyang, 110016, China (e-mail: wangqiang@sia.cn).
1
Recognition
discrimination. Each identity may exhibit subtle nuances of
Weihong Ren is with the Harbin Institute of Technology, Shenzhen 518055,
2
China (e-mail: renweihong@hit.edu.cn).
individual information (i.e., clothing type, accessories, haircut,
Yang Cong is with the College of Automation Science and Engineering,3
etc.) and lead to severe distribution overlapping. Learning
South China University of Technology, Guangzhou, 510640, China (e-mail:
discriminative representations of individuals are effective for
congyang81@gmail.com).

Image
Encoder

distinguish identity information. 2) Inter-domain gaps. Each

Linear

T_

T_B

T_B

AN

GN

AN＋GN

2

Domain is collected in different illumination and background,
leading to inter-domain gaps. Bridging intra-domain gaps are
significant for mitigating catastrophic forgetting in LReID.
To address these issues, we aim to learn domain
consistency representations that capture individual nuances
in intra-domain and inter-domain consistency in LReID.
Knowledge distillation-based approaches [10]–[12] ensure
distribution consistency between the previous and current
domain to alleviate catastrophic forgetting. However, these
approaches impose strict constraints and ignore intradomain discrimination, [13]–[15], as outlined in Fig. 1(a).
While LReID models significantly improve intra-domain
discrimination for the current step, they inevitably damage
inter-domain consistency, leading to catastrophic forgetting.
Thus, we explore global and attribute-wise representations
to strike a trade-off between maximizing intra-domain
discrimination and minimizing inter-domain gaps, improving
the anti-forgetting and generalization capabilities of the
LReID model, as illustrated in Fig. 1(b).
Specifically, we propose a novel domain consistency
representation learning (DCR) model that first explores
attribute and text information to enhance LReID performance.
Unlike methods [16]–[18], we develop domain consistency
representations including global and attribute-wise
representations to capture individual nuances in intradomain and inter-domain consistency in LReID. We design
an attribute-text generator (ATG) to dynamically generate
text-image pairs for each instance, which are then fed into
a text-guided aggregation (TGA) network to enhance the
global representation capability, effectively distinguishing
identities in LReID. Furthermore, the attributes of each
instance guide an attribute compensation (ACN) network to
generate attribute-wise representations that focus on specific
regional information about identities. We consider that
attributes can enhance reliability by setting higher thresholds
across domains. Therefore, the generated attribute-wise
representations and text for each instance are considered
reliable in our model.
In summary, we aim to strike a balance between
maximizing intra-domain identity-discriminative information
and minimizing inter-domain gaps by exploring global and
attribute-wise representations. At the intra-domain level,
global representations capture whole-body information, while
attribute-wise representations focus on specific regional
information. When whole-body appearances or attributerelated information are similar across identities, we combine
global and attribute-wise representations to distinguish among
similar identities, maximizing intra-domain discrimination.
While the model adapts well to new information, it often
struggles with catastrophic forgetting. To preserve old
knowledge, we develop an attribute-oriented anti-forgetting
(AF) strategy that explores attribute-wise representations
to bridge inter-domain gaps across continuous datasets.
Meanwhile, striking a trade-off between maximizing intradomain discrimination and minimizing inter-domain gaps is
crucial for improving the performance of LReID models.
Knowledge consolidation (KC) is proposed to facilitate
knowledge transfer and enhance generalization capabilities,

which consists of alignment and logit-level distillation
mechanisms. The alignment mechanism explores global
representations of knowledge transfer from the current
dataset. The logit-level distillation mechanism enhances the
extraction of identity information, thereby improving the
model’s ability to consolidate knowledge. Our contributions
are as follows:
We propose a novel domain consistency representation
learning (DCR) model that explores global and attributewise representations to capture individual nuances in
intra-domain and inter-domain consistency, achieving a
trade-off between maximizing intra-domain discrimination and minimizing inter-domain gaps.
• In the intra-domain context, we explore the complementary relationship between global and attribute-wise
representations to enhance the discrimination of each
identity and adapt to new knowledge.
• In the inter-domain context, we design an attributeoriented anti-forgetting (AF) and a knowledge consolidation (KC) strategy to minimize inter-domain gaps
and facilitate knowledge transfer, improving the LReID
model’s generalization and anti-forgetting capabilities.
•

II. R ELATED WORK
A. Lifelong Person Re-Identification
Lifelong Person Re-Identification (LReID) aims to balance
intra-domain discrimination with minimizing inter-domain
gaps in continuously updated datasets across scenarios, improving the model’s anti-forgetting and generalization capabilities. LReID methods can be divided into two categories.
1) Knowledge distillation-based methods [12], [19]–[21] utilize metric strategies to achieve domain-consistent alignment
between the old model with learned knowledge distribution
and the new model that adaptively learns new knowledge. 2)
Exemplar-based methods [9], [10], [22] achieve a distribution
balance between old and new samples to prevent catastrophic
forgetting by forming a memory buffer to select the limited
samples from some identities. These methods strive to reduce
inter-domain gaps and ensure consistency across domains to
prevent catastrophic forgetting. However, they ignore intradomain identity discrimination and lack consistency optimization within the inter-domain, limiting the LReID model’s performance in learning new knowledge. In this paper, we explore
domain consistency representations as a bridge to achieve
a trade-off between maximizing intra-domain discrimination
and minimizing inter-domain gaps for enhancing the antiforgetting and generalization capabilities of the LReID model.
B. Vision-Language for Person Re-Identification
Vision-language learning paradigms [23], [24] have gained
widespread popularity in recent years. Contrastive LanguageImage Pre-training (CLIP) [25], establishes a connection between natural language and visual content through the similarity constraint of image-text pairs. CLIP has been applied
to multiple person re-identification tasks [26]–[28], including text-to-image, text-based single-modality, and text-based

3

Attribute-text generator

Text-guided aggregation network (TGA)

Text
Encoder

A photo of a woman
wearing a long sleeved top,
and a trousers, while
carrying a backpack.

[CLS]

G1

C

G2

[CLS]

PFM

1
2

LayerNorm

LayerNorm

Drop

Drop

CA

CA

GN

Image
Encoder

Attribute
Recognition

Parallel Fusion Module (PFM)
Parallel Fusion Module (PFM)
MLP

Global
representations

3

Linear

Q

AG1

T_Block

T_Block

T_Block
Attribute
semantic information

Attribute
features

Attribute Decoder

G1

A1

G2

AN＋GN

G_1
G_2
G_N

Step t+1

Old model 

AG_1
AG_2
AG_N
t−1

AG_1
AG_2

Step t

Backbone
New model 

t

Step t-1

C
Multiplication
Patch embedding

Position embedding

Hat
Shorts
Trousers
Backpack
Short sleeve
Long sleeve

Old model
distribution

Q

[CLS] embedding
Concatenation

AGN

Attribute-wise
representations

Attribute matching

K,V

K,V

AG2

Attribute compensation network (ACN)

Backbone

Step t-1

GN

AN
A2

N

 G_i

Classifier

Logits

i=1

Step t

Minimize
intra-domain gaps

AG_N

New model
distribution

G_1
G_2

Attribute-oriented anti-forgetting

Push
away
N

 G_i

LAL
Classifier

Push
away

LLD
Maximize
inter-domain
discrimination

Logits

i=1

Knowledge consolidation

G_N

Fig. 2. Overview of the proposed DCR for LReID. First, the attribute-text generator (ATG) dynamically generates text-image pairs for each instance. Then, the
text-guided aggregation network (TGA) captures global representations for each identity, while the attribute compensation network (ACN) generates attributewise representations. We explore the complementary relationship between global and attribute-wise representations to maximize intra-domain discrimination.
Meanwhile, we design attribute-oriented anti-forgetting (AF) and knowledge consolidation (KC) strategies to minimize inter-domain gaps and facilitate
knowledge transfer.

cross-modality. Text-to-image methods [28]–[30] aim to retrieve the target person based on a textual query. Text-based
single-modality works [5], [27], [31] leverage text descriptions
to generate robust visual features or integrate the beneficial
features of text and images for the person category. TextQuery
Proposed
DCR
based
cross-modality methods [32]
utilize text
descriptions to
reduce visible-infrared modality gaps. Providing insufficient
text descriptions of each identity, due to prompt learning and
text inversion. In this paper, we dynamically generate textimage pairs from single images to capture fine-grained global
representations based on the CLIP model for improving model
performance capability in terms of inter-domain and intradomain.
C. Pedestrian Attribute Recognition
Pedestrian attribute recognition aims to assign a set of
attributes (Gender, Bag, Short/Long sleeve, and etc.) to a
visual representation of a pedestrian based on their attributes.
Deep learning-based research [33] automatically learns
hierarchical features from raw images, improving recognition
accuracy. Multi-task learning methods [34]–[36] leverage
additional contextual information from multiple domains, such

as pedestrian detection or pose estimation, to significantly
improve attribute recognition. Part-based methods [37], [38]
divide the pedestrian image into several parts or regions,
providing more accurate localization. Currently, these methods
have achieved significant success in improving the accuracy
of attribute recognition. WeDKP
are Method
the first to explore the
application of attributes to LReID from two perspectives.
1) Attributes are converted into text descriptions for each
image to enhance global representation capabilities. 2) The
attributes are transformed into attribute-wise representations
by specific networks to maximize intra-domain discrimination
and minimize intra-domain gaps.

III. P ROPOSED M ETHOD
A. Preliminary: Overview of Method
The overview of our DCR to achieve a trade-off between
maximizing intra-domain discrimination and inter-domain
gaps is shown in Fig. 2. The DCR model learns the old
model Φt−1 and new model Φt from (t-1)-th and t-th steps,
where Φt is inherited from Φt−1 . Φt−1 and Φt with three
parts of attribute-text generator (ATG), text-guided aggregation


codex
前 5 篇已读完。初步看，这批不是在拼单点模块，而是在把干扰源改写成“可分解因素”“持续适配范式”“多域风格空间”这类更大的问题表述，后面我会把这种写法反推出到我们自己的资产上。
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
/bin/zsh -lc "pdftotext -l 3 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Dual-level Modality Debiasing Learning for Unsupervised
Visible-Infrared Person Re-Identification⋆
Jiaze Lia,1 , Yan Lub,1 , Bin Liua,∗, Guojun Yina , Mang Yec
a University of Science and Technology of China, Hefei, 230026, China
b Shanghai Artificial Intelligence Laboratory, Shanghai, 200233, China

arXiv:2512.03745v2 [cs.CV] 9 Apr 2026

c the School of Computer Science, Wuhan University, Wuhan, 430072, China

Abstract
Two-stage learning pipeline has achieved promising results in unsupervised visibleinfrared person re-identification (USL-VI-ReID). It first performs single-modality learning and then operates cross-modality learning to tackle the modality discrepancy. Although promising, this pipeline inevitably introduces modality bias: modality-specific
cues learned in the single-modality training naturally propagate into the following
cross-modality learning, impairing identity discrimination and generalization. To address this issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework that implements debiasing at both the model and optimization levels. At the
model level, we propose a Causality-inspired Adjustment Intervention (CAI) module
that replaces likelihood-based modeling with causal modeling, preventing modalityinduced spurious patterns from being introduced, leading to a low-biased model. At the
optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to
interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive
experiments on benchmark datasets demonstrate that DMDL could enable modalityinvariant feature learning and a more generalized model. The code is available at
https://github.com/priester3/DMDL.
⋆ This work is supported by the National Natural Science Foundation of China (Grant No. 62272430).
∗ Corresponding author

Email addresses: jz_li@mail.ustc.edu.cn (Jiaze Li), luyan@pjlab.org.cn (Yan Lu),
flowice@ustc.edu.cn (Bin Liu), gjyin@mail.ustc.edu.cn (Guojun Yin), yemang@whu.edu.cn
(Mang Ye)
1 These authors contributed equally to this work.

Keywords: Visible-infrared person re-identification, Unsupervised learning, Causal
intervention, Modality-invariant feature

1. Introduction
Visible-infrared person re-identification (VI-ReID) focuses on the identification
and matching of individuals across distinct modalities, visible and infrared. Remarkable progress has been made in this field, as evidenced by the success of existing works
[1, 2]. However, the collection of extensive cross-modality annotations is a costly and
time-consuming process, which poses limitations on its broader application. As a solution, Unsupervised Visible-infrared Person Re-identification (USL-VI-ReID) [3, 4, 5]
has emerged to facilitate VI-ReID without the reliance on human identity labels.
The main challenge in the USL-VI-ReID is the modality discrepancy, which limits the direct application of standard unsupervised learning of traditional unsupervised
ReID. Therefore, the mainstream methods for USL-VI-ReID typically follow a twostage learning pipeline [4, 5, 6, 7]: 1) In the first stage, the model is trained by operating unsupervised learning techniques [8] on each modality separately to have the
single-modality discriminative ability. 2) In the second cross-modality unsupervised
process, the model alternately establishes relationships across modalities and fits these
relationships to achieve cross-modality discrimination capabilities. Although promising, it also suffers from a modality bias issue that restricts the overall results. The first
single-modality learning process naturally captures modality-specific cues from visible/infrared data, resulting in a biased model. Initializing the second stage with this
model inevitably introduces modality bias into the cross-modality learning, leading to
biased cross-modality relationships, e.g., similar clothing color cues may result in incorrect matches across modalities, as illustrated in Fig. 1 (a). Since cross-modality
relationships (i.e., pseudo labels) are the model-fitting target in the second stage, the
biased knowledge (i.e., modality-specific cues) is gradually enhanced in the learned
patterns, leading to modality-related features. In summary, modality bias originating
from data propagates into labels and features throughout the learning pipeline, leading
the model to rely on modality-specific cues for identification and thereby significantly

2

Color

Identity

biased likelihood
model

causal model

(a) Existing biased USL-VI-ReID

(b) Our Modality Debiasing

Figure 1: Existing USL-VI-ReID methods suffer from modality bias, leading to modality-related features.
In contrast, our approach achieves modality-invariant feature learning through causal modeling and unbiased
optimization. Green, yellow, and blue circles represent visible-specific, infrared-specific, and modalityshared information, respectively.

limiting its generalization.
To address the aforementioned modality bias issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework. DMDL performs modality debiasing at
both the model and optimization levels, where the former prevents the model from
learning modality bias in incorrect cross-modality relationships, and the latter aims to
disrupt the propagation of biased knowledge from data to labels and features directly.
To this end, a Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Bias-free Training (CBT) strategy are proposed. Specifically, CAI facilitates
causal intervention under cross-modality unsupervised learning with backdoor adjustment, making the model only capture the causal patterns. Compared with the traditional
likelihood method, the causal modeling in CAI is theoretically unaffected by modality bias, thereby achieving a low-biased model. To further prevent biased knowledge
from deepening during optimization, we propose the CBT strategy, integrating data
augmentation, label refinement, and feature alignment. CBT first introduces a pseudomodality augmentation scheme to modify modality-specific cues in images. Based on
the augmented images, a cross-modality label smoothing scheme and a feature alignment loss are proposed to refine the biased relationships and learn shared knowledge
across pseudo-modalities, respectively. By jointly leveraging these components, CBT
explicitly interrupts the propagation of modality bias across data, labels, and features.
Ultimately, the overall DMDL keeps an effective modality debiasing implementation,

3


exec
/bin/zsh -lc "pdftotext -l 3 'Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Dual-Modality-Shared Learning and Label Refinement for
Unsupervised Visible-Infrared Person ReID
LICUN DAI and ZHIMING LUO, Department of Artificial Intelligence, Xiamen University,
Xiamen, China

YONGGUO LING, The School of Computer, Electronics and Information, Guangxi University,
Nanning, China

JIAXING CHAI, Department of Artificial Intelligence, Xiamen University, Xiamen, China
SHAOZI LI, Department of Artificial Intelligence, Xiamen University, Xiamen, China and Fujian Key
Laboratory of Big Data Application and Intellectualization for Tea Industry, Wuyi University,
Wuyishan, China
Unsupervised visible-infrared person re-identification (USVI-ReID) aims to match a person across two modalities without annotations. Current research primarily addresses the modality gap by establishing cross-modality
correspondences through matching algorithms and utilizing memory banks for contrastive learning. However,
the inherent noise in pseudo labels and neglect of hard samples often limit the efficacy of cross-modality
learning. In this article, we propose a dual-modality-shared learning and label refinement (DLLR) algorithm for
USVI-ReID. First, we leverage a cluster similarity matching (CSM) module and a cluster relationship-based label
refinement (CRLR) algorithm to create and refine pseudo labels. Then, we adopt a weighted modality-shared
memory (WMM) to construct memory banks by jointly considering sample distribution and feature differences,
thereby enhancing the effectiveness of cross-modality learning. Extensive experiments on three publicly
available datasets validate the effectiveness of our proposed method, which outperforms state-of-the-art
methods. The code is available at https://github.com/CharRic/DLLR.
CCS Concepts: • Computing methodologies → Computer vision; • Information systems → Information retrieval;
Additional Key Words and Phrases: Visible-Infrared Person Re-Identification, Contrastive Learning, Unsupervised Learning

This work is supported by the National Natural Science Foundation of China (No. 62276221, No. 62376232, No. 62466003);
the Fujian Provincial Natural Science Foundation of China (No. 2022J01002); the Natural Science Foundation of Guangxi
Province of China (No. 2023JJB170012, No. 2024JJA170003); the Open Project Program of Fujian Key Laboratory of Big Data
Application and Intellectualization for Tea Industry, Wuyi University (No. FKLBDAITI202304).
Authors’ Contact Information: Licun Dai, Department of Artificial Intelligence, Xiamen University, Xiamen, China; e-mail:
dailicun@stu.xmu.edu.cn; Zhiming Luo (corresponding author), Department of Artificial Intelligence, Xiamen University,
Xiamen, China; e-mail: zhiming.luo@xmu.edu.cn; Yongguo Ling, The School of Computer, Electronics and Information,
Guangxi University, Nanning, China; e-mail: yongguo_ling@sina.com; Jiaxing Chai, Department of Artificial Intelligence,
Xiamen University, Xiamen, China; e-mail: 31520221154193@stu.xmu.edu.cn; Shaozi Li, Department of Artificial Intelligence,
Xiamen University, Xiamen, China and Fujian Key Laboratory of Big Data Application and Intellectualization for Tea
Industry, Wuyi University, Wuyishan, China; e-mail: szlig@xmu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/11-ART325
https://doi.org/10.1145/3724397
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.

325:2

L. Dai et al.

ACM Reference format:
Licun Dai, Zhiming Luo, Yongguo Ling, Jiaxing Chai, and Shaozi Li. 2025. Dual-Modality-Shared Learning and
Label Refinement for Unsupervised Visible-Infrared Person ReID. ACM Trans. Multimedia Comput. Commun.
Appl. 21, 11, Article 325 (November 2025), 24 pages.
https://doi.org/10.1145/3724397

1

Introduction

Visible-infrared person re-identification (VI-ReID) aims at matching pedestrian images of the
same identity captured by visible and infrared cameras when provided with a query image from
a different modality. The advancements in 24-hour security surveillance systems have led to the
widespread usage of multi-modal cameras that automatically switch between visible and infrared
modes based on lighting conditions. Consequently, VI-ReID has garnered significant attention
[32, 65, 69, 71]. In contrast to traditional person re-identification [14, 43, 51, 61], VI-ReID not only
addresses the challenges like appearance and viewpoint variations in ReID, but further confronts
the significant modality gap between visible and infrared images. To address the modality gap,
some studies designed network architectures and loss functions [13, 24, 31, 53, 64, 71], while some
explored auxiliary features to compensate for modality-shared information [35, 48, 49, 63, 69].
However, these methods rely on manually annotated pedestrian datasets for training, which is
costly and time-intensive.
To advance the practical application of VI-ReID, we investigate unsupervised VI-ReID (USVIReID) in this study. The main goal of USVI-ReID is to accurately associate samples from different
modalities while effectively guiding the model in handling the significant modality difference.
Many USVI-ReID methods [5, 28, 50] employ a two-stage framework to learn modality-invariant
representations. However, these approaches require an additional annotated visible dataset for
model training, limiting the generalizability of the methods. Recently, pseudo labels generated by
advanced clustering algorithms [11, 54] have shown effectiveness in unsupervised learning (USL).
Besides, memory bank-based approaches have garnered increasing attention for unsupervised
single-modality person re-identification [8, 18]. These methods iteratively assign pseudo labels
to samples and train the model with a contrastive learning framework. To successfully associate
samples from different modalities for USVI-ReID, various attempts [4, 55, 59] have been made
utilizing different matching algorithms, achieving impressive performance. However, most existing
methods overlook the noise present in pseudo labels. Furthermore, many methods initialize memory
banks by averaging the features of samples within a cluster, neglecting the hard samples in the
dataset and hindering the effectiveness of cross-modality learning.
To address the aforementioned challenges, we propose a novel method for USVI-ReID called
dual-modality-shared learning and label refinement (DLLR). As shown in Figure 1, the
framework consists of a cluster similarity matching (CSM) module, a cluster relationshipbased label refinement (CRLR) algorithm, and a weighted modality-shared memory (WMM)
module. Initially, intra-modality relationships among samples are established through clustering.
Subsequently, the CSM associates samples from different modalities by assigning modality-shared
pseudo labels to clusters based on cluster similarity. Next, the CRLR is used to enhance the quality
of modality-shared pseudo labels by separating clusters that potentially contain samples with
different identities and merging those that may belong to the same identity. This approach proves
effective by using readily available information from matched cross-modality clusters, providing
high-quality homogeneous instance-level associations that aid the refinement of pseudo labels.
Moreover, considering the diverse difficulties of samples, we introduce a WMM for dual-modalityshared learning, in which a weighted strategy based on the feature distribution is used to assign
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.

DLLR for Unsupervised Visible-Infrared Person ReID

325:3

Fig. 1. The overall pipeline of our proposed method. Samples from each modality are clustered into clusters,
and the cluster similarity matching (CSM) module matches clusters from different modalities. The cluster
relationship-based label refinement (CRLR) algorithm is then used to refine modality-shared pseudo labels.
Next, dual-modality-shared learning is conducted using the weighted modality-shared memory (WMM) to
narrow the modality gap while considering sample difficulties.

different weights to samples for initializing memory banks. Consequently, the proposed model
excels in learning from two modalities by a dual-modality-shared learning framework, while
eliminating the need for labels or additional labeled single-modality datasets.
In summary, our main contributions are listed as follows:
— We propose a DLLR framework for USVI-ReID. By incorporating the CSM and CRLR algorithms, our framework can effectively establish associations between unlabeled samples
across modalities, and then generate high-quality pseudo labels for model training.
— We design the WMM to assign different weights to samples for constructing memory banks,
which can enhance the model’s capacity to learn modality-invariant features by considering
hard samples.
— Extensive experiments on three public benchmarks demonstrate the superiority of our proposed method, outperforming state-of-the-art USVI-ReID methods and even surpassing many
supervised VI-ReID methods.
2

Related Work

2.1

VI-ReID

VI-ReID is a challenging task due to the substantial modality gap between visible and infrared
images [7, 16, 19, 29, 46, 71]. Existing approaches for VI-ReID can be categorized into the following
two categories. (1) The first category focuses on developing novel network architectures and loss
functions to extract modality-shared features [12, 13, 30, 33, 52, 53, 62]. For instance, Wu et al. [52]
conducted pioneering research in VI-ReID, introducing a zero-padding single-stream network,
generating multiple representations of each image, and then fusing multi-level features to enhance
feature discrimination. Lu et al. [34] proposed a progressive learning strategy with modality-shared
enhancement loss and discriminative center loss to improve the discrimination of modality-shared
features. Hua et al. [22] designed a multi-scale semantic correlation mining network to exploit
semantic features at multiple scales, which improves the issue of losing valuable information during
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 11, Article 325. Publication date: November 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 203 (2026) 109083

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Dynamic adaptive multi-view contrastive learning for unsupervised person
re-identiﬁcation
Zhi-Hua Li

a , Xue-Yan Wang a , Si-Bao Chen a,∗, Chris H.Q. Ding b , Bin Luo a

a MOE Key Laboratory of ICSP, IMIS Laboratory of Anhui Province, Anhui Provincial Key Laboratory of Multimodal Cognitive Computation, and Zenmorn-AHU AI Joint

Laboratory, School of Computer Science and Technology, Anhui University, Hefei, 230601, China
b
Department of Computer Science and Engineering, The Chinese University of Hong Kong, Shenzhen, 518172, China

a r t i c l e

i n f o

Keywords:
Unsupervised person re-identiﬁcation
Contrastive learning
Pseudo-label
Cluster proxies

a b s t r a c t
Recent unsupervised person re-identiﬁcation (Re-ID) methods leverage clustering to generate pseudo-labels for
contrastive learning with a memory bank. However, camera variations introduce noise into these clusteringbased pseudo-labels, and contrastive learning is hindered by inaccurate proxy construction, with hard pseudolabels exhibiting inherent sensitivity to noise. This paper proposes a novel framework, Dynamic Adaptive Multiview Contrastive Learning (DAMCL), to address these challenges. We introduce a Dynamic Adaptive Camera
Jaccard (DACJ) distance to dynamically estimate and mitigate camera variations during each training epoch.
Additionally, a Dynamic Adaptive Proxies (DAP) module, comprising Dynamic Optimal Cluster Proxies (DOCP)
and Dynamic Instance Proxies (DIP), is proposed. Building on DACJ, DOCP forms the cluster proxy using the
medoid of all cluster instances as its optimal feature representation. It aligns samples closely with their designated
cluster proxy while distancing them from foreign proxies, using pseudo-labels generated by DBSCAN. Meanwhile,
the DIP enhances clustering by leveraging global sample relationships. Finally, a Dynamic Adaptive Knowledge
Distillation (DAKD) module is introduced to generate reﬁned soft labels, improving robustness and accuracy.
Comprehensive experiments conﬁrm the eﬃciency of our approach.

1. Introduction
Person re-identiﬁcation (Re-ID) focuses on recognizing and matching
the same individual across diﬀerent camera views. It has gained considerable interest in intelligent surveillance systems because of its significant potential in practical applications. Most supervised Re-ID methods (He et al., 2021; Li et al., 2024; Zheng et al., 2019) leverage humanannotated labels to learn discriminative features and have achieved signiﬁcant progress. However, these methods rely heavily on human annotation, which severely limits the application. Hence, recent studies have
shifted focus to unsupervised Re-ID. Most unsupervised methods (Chen
et al., 2021; Dai et al., 2022; Ge et al., 2020b; Lan et al., 2023; Wang
et al., 2021; Yin et al., 2023; Zou et al., 2023) employ a two-stage alternating training scheme: 1) generating pseudo-labels using Jaccard
distance (Zhong et al., 2017) to measure similarity between all training
samples, followed by a clustering algorithm (Ester et al., 1996; Lloyd,
1982); 2) training the model with a memory bank constructed from
these pseudo-labels. Leveraging this scheme, memory-based contrastive
learning approaches have demonstrated superior performance by utilizing proxies stored within the memory bank.

Despite signiﬁcant advancements achieved in previous methods,
memory-based contrastive learning framework still faces three inherent
and challenging issues.
Firstly, noise within clustering-generated pseudo-labels propagates
erroneous supervisory signals during training. This arises from substantial cross-camera variations, including diﬀerences in backgrounds,
illumination, perspectives, and other factors. As shown in Fig. 1, in
some cases, camera variations can cause within-camera neighbors of
a query image (from diﬀerent identities) to exhibit higher similarity
than its cross-camera counterparts. Consequently, clustering algorithms
frequently incorrectly group within-camera samples from distinct
identities due to their inﬂated similarity, yet neglect to associate
cross-camera samples sharing the same identity. This leads to numerous
pseudo-labels noise. Secondly, inaccurate cluster proxies negatively impact the contrastive learning system. Previous studies have adopted the
mean feature as the proxy for the cluster centroid, operating under the
assumption that data within each cluster adheres to a high-dimensional
spherical distribution (Chen et al., 2021; Dai et al., 2022; Ge et al.,
2020b; Lan et al., 2023). However, in practice, the feature clusters
formed in the Re-ID feature space are often complex, non-spherical

∗ Corresponding author.

E-mail address: sbchen@ahu.edu.cn (S.-B. Chen).
https://doi.org/10.1016/j.neunet.2026.109083
Received 4 August 2025; Received in revised form 20 January 2026; Accepted 8 May 2026
Available online 10 May 2026
0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 203 (2026) 109083

Z.-H. Li et al.

after computing DACJ, performing clustering can yield cleaner pseudolabels. Secondly, we propose a Dynamic Adaptive Proxies (DAP) module
integrating two synergistic submodules: the Dynamic Optimal Cluster
Proxies (DOCP) and Dynamic Instance Proxies (DIP). The DOCP constructs the cluster proxy by selecting the medoid—the instance within
each cluster whose sum of distances to all other members is minimal.
By serving as the optimal feature representation, the medoid preserves
the intrinsic data manifold of DBSCAN clustering, diverging from prior
spherical distribution assumptions. At the same time, the DIP treats
each sample as an anchor, performing contrastive learning with all positive and negative instances stored in the instance-level memory bank.
This enables comprehensive exploitation of global information to mitigate within-cluster variations and adaptively reﬁne clustering outcomes.
Thirdly, we introduce a novel dynamic adaptive knowledge distillation
(DAKD) approach that generates high-quality soft labels through knowledge distillation and DAP, using class prediction consistency to improve
robustness against pseudo-label noise. Finally, we integrate two local
features extracted from distinct views (upper and lower body regions)
to capture more discriminative information that may be overlooked by
the global feature.
The key contributions of this work are summarized as follows:

Fig. 1. Illustration of the camera variations problem. Shapes represent camera
sources, with identical shapes denoting images captured by the same camera,
while colors represent identity labels, with identical colors indicating images of
the same individual.

due to signiﬁcant variations in pose, viewpoint, and illumination. This
inherent complexity indicates that the data points are actually distributed on a manifold rather than strictly conforming to the traditional
spherical cluster structure (Bengio et al., 2013; Yin et al., 2023).
Speciﬁcally, as illustrated in Fig. 2, real-world unsupervised person
Re-ID data exhibits manifold distributions, which consequently leads to
proxy deviations from the true cluster centers and potential intrusion
into foreign clusters. Concurrently, the momentum update strategy,
by compelling features to converge to historical averages, distorts the
data’s intrinsic manifold structures. This distortion is fundamentally
at odds with the local Euclidean properties upon which DBSCAN
clustering is predicated. This theoretical incompatibility between proxy
modeling and update mechanisms collectively induces feature representation distortion and misguided optimization, severely limiting model
performance ceilings. Thirdly, typical single hard targets (i.e., one-hot
vectors representing predicted labels) are not robust to pseudo-label
noise and can mislead model optimization, amplifying errors during
training.

•

We propose a novel dynamic adaptive camera Jaccard distance
(DACJ), which explicitly incorporates camera-speciﬁc distribution
characteristics into similarity measurement. By adaptively adjusting
the weight of sample correlations within the same camera and across
diﬀerent cameras according to the variation in camera conditions,
this approach decreases pseudo-label noise and improves the robustness of feature representation learning.
• We propose a novel dynamic adaptive proxies (DAP) module that
mitigates the negative impact of inaccurate cluster proxies on contrastive learning. Unlike conventional static proxy methods, DAP addresses the error accumulation caused by ﬁxed cluster centroids in
scenarios with imbalanced data distributions.
• We propose a dynamic adaptive knowledge distillation (DAKD) module that generates high-quality soft labels. This design enables adaptive adjustment of knowledge transfer intensity across diﬀerent samples, eﬀectively suppressing error propagation from noisy pseudolabels.

Therefore, to address the existing challenges, we introduce a new
contrastive learning framework called dynamic adaptive multi-view
contrastive learning (DAMCL). We ﬁrst propose a Dynamic Adaptive
Camera Jaccard (DACJ) distance to dynamically estimate camera variations per training epoch and perform adaptive variation reduction. Crucially, when camera variations for a sample exceed a threshold, DACJ
proportionally adjusts the within-/cross-camera neighbor ratio in its
feature space, ensuring highly informative positive samples from crosscamera sources are eﬀectively included in the relevant neighbors. Thus,

2. Releted works
This section discusses the most relevant studies from three angles: 1)
unsupervised person Re-ID, encompassing unsupervised domain adaptation (UDA) and fully unsupervised learning (USL) approaches for person
Re-ID; 2) contrastive learning, which develops robust feature representations by leveraging positive and negative sample pairs; and 3) knowledge distillation, which involves techniques to improve model perfor-

Fig. 2. DBSCAN clustering on two no-spherical distributed datasets. Diﬀerent colors of points represent distinct clusters, and the red pentagram indicates the mean
feature proxy point for each cluster.
2

Neural Networks 203 (2026) 109083

Z.-H. Li et al.

mance by transferring knowledge from a larger model to a more compact one.

incorporating camera information, eﬀectively mitigating camera variations to improve pseudo-label generation and re-ranking accuracy. However, this method rely on static parameter settings, which constrain
their adaptability to dynamic feature distributions and complex, nonspherical data across training epochs. Our DACJ introduces a dynamic
camera variations degree vector Δ, which continuously estimates and
adapts to camera variations at each epoch. This approach enables DACJ
to produce cleaner pseudo-labels and achieve superior performance, delivering substantial improvements in mean Average Precision (mAP).
Additionally, methods employing memory banks for model training,
which depend on mean proxies, often neglect the challenges posed by
data stream distribution. These limitations can hinder the eﬀectiveness
of model training.

2.1. Unsupervised person Re-ID
Existing unsupervised person Re-ID methods can be roughly divided
into unsupervised domain adaptation (UDA) methods (Dai et al., 2021;
Ge et al., 2020b; Han et al., 2022a; He et al., 2022; Lee et al., 2023;
Li et al., 2023; Yin et al., 2022) and purely unsupervised learning
(USL) (Cho et al., 2022; Dai et al., 2022; Ji et al., 2024a; Lan et al.,
2023; Ran et al., 2025; Wang et al., 2021; Yin et al., 2023; Yu et al.,
2023; Zhang et al., 2024, 2022; Zou et al., 2023) methods. UDA methods transfer knowledge from a labeled source domain to an unlabeled
target domain, while USL methods train directly on unlabeled target
datasets. The core challenge in UDA Re-ID methods lies in eﬀectively
utilizing a labeled source dataset to enhance model performance on an
unlabeled target dataset. These methods typically rely on the premise
that the diﬀerences between the source and target domains are minimal,
employing transfer learning techniques to address this challenge.
Our approach addresses the more demanding requirements of the
USL framework. In unsupervised person Re-ID frameworks, a selfsupervision paradigm is typically used to leverage unlabeled target data:
clustering algorithms (i.e., K-means Lloyd, 1982, DBSCAN Ester et al.,
1996) partition target domain samples into pseudo-label clusters, followed by joint training strategies that combine these pseudo-labels with
the unlabeled data through iterative optimization cycles (Dai et al.,
2022; Ge et al., 2020b). This cyclical process enables progressive reﬁnement of feature embeddings while mitigating error accumulation from
imperfect cluster assignments.
To enhance the reliability of pseudo-labels, several advanced methods for pseudo-label generation have been developed. MMT (Ge et al.,
2020a) enhances pseudo-labels by training two neural networks with
combined supervision from oﬄine reﬁned hard pseudo-labels and online reﬁned soft pseudo-labels. SPCL (Ge et al., 2020b) proposed a selfpaced approach that incrementally builds more reliable clusters to reﬁne
hybrid memory and learning objectives.
To tackle inconsistencies in memory updates, CCL (Dai et al.,
2022) developed a novel cluster contrast learning framework using a
cluster-level memory dictionary, achieving remarkable performance.
ISE (Zhang et al., 2022) employs a progressive linear interpolation strategy to create support samples from real samples and adjacent clusters in
the embedding space, enabling more precise cluster representations by
capturing underlying information. In Li et al. (2022), a self-guided hard
negative generation approach is introduced, utilizing adversarial training between a hard negative generation network and a re-ID network
to mutually improve their performance. In Han et al. (2022b), a group
sampling approach is introduced to mitigate overﬁtting by minimizing
the negative inﬂuence of individual samples on statistical stability. In
GMT (Cai et al., 2025), a balancing multi-task oriented fusing method is
introduced to explicitly enhance Re-ID sub-task learning through additional Re-ID training and target-guided multi-model parameters fusion.
Recent works have achieved signiﬁcant progress in multi-view clustering tasks. Among them, BWIC-TIMC (Yao et al., 2025) jointly exploits
both inter-view and intra-view information and employs an adaptive
fused graph learning strategy, demonstrating remarkable eﬀectiveness.
IMVC-TGR (Wang et al., 2025) separates redundant information in each
aﬃnity graph through graph reﬁnement operations, stacks the feature
graphs into a tensor, and introduces semantic consistency constraints
along with a self-weighted fusion strategy, which not only maintains
inter-view consistency but also signiﬁcantly enhances the quality of
the feature graphs. Recent approaches have tackled camera variations
in unsupervised person Re-ID with innovative methods. CGMAL (Ran
et al., 2025) constructing a heterogeneous graph to model semantic
relationships, including positive pairs across cameras (same identity)
and negative pairs within cameras (diﬀerent identities). Similarly, CAJaccard (Chen et al., 2024) enhances the traditional Jaccard distance by

2.2. Contrastive learning
Contrastive learning (Chen et al., 2020; He et al., 2020; Wang & Qi,
2022), a leading self-supervised learning approach, focuses on maximizing the similarity between representations derived from various augmented or distorted versions of a sample while minimizing similarity
with representations of other samples in the latent space. CLSA (Wang
& Qi, 2022) introduces a contrastive learning framework that combines
weakly and strongly augmented images using distribution divergence
minimization, enhancing the performance of self-supervised learning.
In SimCLR (Chen et al., 2020), it directly extract representations from
a large batch. In MoCo (He et al., 2020), a novel contrastive learning framework is introduced, featuring a momentum-based encoder
and a queue-structured memory bank to dynamically update negative
samples, enhancing the eﬀectiveness of contrastive learning. Distinct
from MoCo’s approach, our proposed method simultaneously performs
cluster-level and instance-level contrastive learning through our innovative DAP.
2.3. Knowledge distillation
Knowledge distillation aims to transfer knowledge from a teacher
network to a smaller student network. Recently, there has been a focus on self-knowledge distillation, where both networks share the same
structure (Lan et al., 2023; Tian et al., 2020; Ye et al., 2020, 2021; Yun
et al., 2020). These methods leveraged outputs from a structurally identical teacher. In Tian et al. (2020), a straightforward yet eﬀective baseline for few-shot learning was proposed, focusing on minimizing the
loss tied to the class probability distribution generated by the teacher
model. CS-KD (Yun et al., 2020) introduced an innovative regularization
approach that aligns distribution predictions across samples within the
same class. Meanwhile, HDCPD (Cheng et al., 2022) aims to harmonize
the probability distribution between the network and the teacher network, which is updated using the Exponential Moving Average (EMA)
method. CORE (Ye et al., 2021) jointly reﬁnes networks and label predictions by leveraging knowledge distillation from peer networks, relying
on minimal and possibly imprecise annotations. Additionally, a collaborative ensemble learning scheme in Ye et al. (2020) utilizes relationships
among classiﬁers for cross-modality person re-identiﬁcation, enhancing
discriminability through consistent ensemble outputs. Although knowledge distillation has led to signiﬁcant progress, these methods may not
entirely mitigate the susceptibility of traditional hard labels to pseudolabel noise.
3. Method
3.1. Overview
Given a person Re-ID dataset 𝑃 = {𝑥𝑖 }𝑁
, where 𝑁 represents the
𝑖=1
total number of images and 𝑥𝑖 denotes the 𝑖th image, the goal is to train
a robust network 𝑓𝜃 that maps each sample 𝑥𝑖 to a corresponding feature
vector 𝑿 𝑖 = 𝑓𝜃 (𝑥𝑖 ) in the embedding space  . Additionally, each image
3


exec
/bin/zsh -lc "pdftotext -l 3 'Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020

1

Dynamic Modality-Camera Invariant Clustering for
Unsupervised Visible-Infrared Person
Re-identification

arXiv:2412.08231v1 [cs.CV] 11 Dec 2024

Yiming Yang, Weipeng Hu, Haifeng Hu

Abstract—Unsupervised learning visible-infrared person reidentification (USL-VI-ReID) offers a more flexible and costeffective alternative compared to supervised methods. This field
has gained increasing attention due to its promising potential.
Existing methods simply cluster modality-specific samples and
employ strong association techniques to achieve instance-tocluster or cluster-to-cluster cross-modality associations. However,
they ignore cross-camera differences, leading to noticeable issues
with excessive splitting of identities. Consequently, this undermines the accuracy and reliability of cross-modal associations.
To address these issues, we propose a novel Dynamic ModalityCamera Invariant Clustering (DMIC) framework for USL-VIReID. Specifically, our DMIC naturally integrates ModalityCamera Invariant Expansion (MIE), Dynamic Neighborhood
Clustering (DNC) and Hybrid Modality Contrastive Learning
(HMCL) into a unified framework, which eliminates both the
cross-modality and cross-camera discrepancies in clustering. MIE
fuses inter-modal and inter-camera distance coding to bridge
the gaps between modalities and cameras at the clustering
level. DNC employs two dynamic search strategies to refine the
network’s optimization objective, transitioning from improving
discriminability to enhancing cross-modal and cross-camera generalizability. Moreover, HMCL is designed to optimize instancelevel and cluster-level distributions. Memories for intra-modality
and inter-modality training are updated using randomly selected
samples, facilitating real-time exploration of modality-invariant
representations. Extensive experiments have demonstrated that
our DMIC addresses the limitations present in current clustering
approaches and achieve competitive performance, which significantly reduces the performance gap with supervised methods.
Index Terms—Person re-identification (Re-ID), cross-modality,
unsupervised learning, clustering.

I. I NTRODUCTION

P

ERSON re-identification is employed to identify and
locate specific individuals among pedestrians captured in
multiple camera surveillance scenarios [1], [16]. This technology have been significantly developed during the last decade
due to its important role in the fields of multimedia data
retrieval and criminal investigation [3], [4], [5]. Earlier works
on person Re-ID focus on the retrieval of pedestrian images
captured by RGB cameras. However, such methods falter in
low-light conditions. This is primarily due to the inherent
limitation of RGB cameras in acquiring high-definition images
Yiming Yang and Haifeng Hu are with the School of Electronics and
Information Technology, Sun Yat-sen University, Guangzhou 510006, China.
(e-mail: yangym53@mail2.sysu.edu.cn; huhaif@mail.sysu.edu.cn).
Weipeng Hu is with the School of Electrical and Electronic Engineering (EEE), Nanyang Technological University, Singapore 639798. (e-mail:
weipeng.hu@ntu.edu.sg).

Fig. 1. Illustration of cross-modality and cross-camera discrepancies in
clustering. Large variations caused by these discrepancies lead to identities
splitting. Fine-tuning the network using these inaccurate labels obtains suboptimal results.

in darkness. Therefore, visible-infrared person re-identification
(VI-ReID) is proposed and employed to form 24-hour surveillance system, which aims to match infrared images under poor
illumination with visible images under good illumination.
The current VI-ReID methods focus on generation and subspace mapping technique to learn modality-invariant representations, achieving notable success [6], [7], [8], [9]. However,
their dependence on manually annotated associations between
visible and infrared modalities can hinder the scalability and
deployment of the VI-ReID model. Unsupervised Learning
Visible-Infrared Person Re-Identification (USL-VI-ReID) is
introduced to eliminate this reliance on annotations and gains
increasing attention due to its promising potential.
The USL-VI-ReID method explores cross-modal associations, eliminating the requirement for manual identity labeling [10], [11]. Although DBSCAN [12] is an effective
clustering algorithm for pseudo-labeling unlabeled data, challenges still arise due to substantial cross-modality and crosscamera variations within the images from the same ground
truth ID [6], [13], [14]. Fig. 1 illustrates the challenge in
clustering cross-modal person data. Variations across cameras
and modalities lead to excessive identity splitting and hinder
accurate label assignment. Fine-tuning the network using these

JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020

labels may amplify the distances within the same class more
than those between different classes. In this case, additional
noise can be potentially introduced to adversely affect the
performance of the model. Existing methods [15], [16], [17],
[18], [19] primarily focus on eliminating differences between
different modalities while neglecting the challenge of crosscamera discrepancy. For instance, previous approaches utilize
graph matching [16] and optimal transmission [20] techniques
to facilitate cross-modal cluster association. However, these
methods are impeded by the severe issue of excessive identity
splitting, which may affect the accuracy of the association.
To address the above problems, we propose a novel Dynamic Modality-Camera Invariant Clustering (DMIC) framework to eliminate cross-modality and cross-camera discrepancies at the clustering level. The flowchart of DMIC is displayed
in Fig. 2. To be specific, DMIC naturally combines ModalityCamera Invariant Expansion (MIE), Dynamic Neighborhood
Clustering (DNC) and Hybrid Modality Contrastive Learning
(HMCL) into a joint framework. The MIE integrates intermodal and inter-camera distance coding, generating robust
embeddings for the clustering algorithm. This eradicates the
implicit inclusion of modality and camera information in the
distance coding, resulting in modality-camera invariant embeddings. Consequently, modality-camera invariant associations
can be established. To tackle the issue of intra-class distances surpassing inter-class distances due to excessive identity
splitting, the DNC employs two dynamic search strategies.
Specifically, for the first stragegy, we dynamically narrow the
search radius to include the reliable positive samples within
clusters, thereby enhancing the model’s ability to distinguish
between relevant and irrelevant samples. Subsequently, we
dynamically broaden the search radius, using the model’s
refined discrimination to effectively incorporate reliable crosscamera and cross-modality samples into the cluster. The
second strategy involves recalibrating the expanded distance
coding. This refinement facilitates a broaden affinities of more
cross-camera instances, which can be leveraged to contribute
to cross-camera invariant learning. Importantly, our strategies
do not require additional parameters as a cost, effectively
enhancing the model’s performance. Taking inspiration from
[21], we design HMCL to optimize instance-level and clusterlevel distributions. We randomly select instance from different
modalities as cluster’s centroid and update the representations
of clusters in a real-time manner, which effectively reduces
modal gap.
To sum up, the main contributions of this paper are list as
follows:
• We propose a novel DMIC network for USL-VI-ReID
that simultaneously eliminates both the cross-modality
and cross-camera discrepancies in clustering.
• The MIE fuses distance coding between inter-modal and
inter-camera instances, which bridges cross-modality and
cross-camera gaps.
• The DNC consists of two dynamic search strategies
that do not require additional parameters, facilitating
early optimization of model discriminability and gradually extending generalization to different cameras and
modalities.

2

The HMCL performs cluster-level and instance-level contrastive learning for intra-modality and inter-modality
training. The cluster representations are updated using
randomly selected cross-modal samples to obtain compact cross-modal distribution.
• Extensive experiments on the SYSU-MM01 and RegDB
datasets showcase the effectiveness and competitive performance of our DMIC framework.
•

II. R ELATED WORK
In this section, we provide a brief overview of the following areas: Supervised Visible-Infrared person ReID (SVIReID), Unsupervised Single-Modality Person ReID (USLReID), and Unsupervised Learning Visible-Infrared Person
Re-Identification (USL-VI-ReID).
A. Supervised Visible-Infrared person ReID
Supervised Visible-Infrared Person ReID (SVI-ReID) can
be roughly divided into feature-level and image-level modality
alignment methods.
The feature-level modality alignment methods aim to project
cross-modal features into a shared subspace and employs metric learning techniques to narrow the distribution. In pursuit of
this goal, Ye et al. [6] introduce a modality-aware collaborative
ensemble learning approach to eliminate modal discrepancies
at both the instance and classifier levels. Lu et al. [7] propose a
two-step Wasserstein loss to align modality-unrelated information, which includes viewpoint, background, and posture. To
fully leverage a wide array of cross-modality cues, a diverse
embedding expansion network [22] is introduced to expand
the feature set and reduce modal differences through triplelevel constraints. Hao et al. [23] design a camera-aware and
modality-aware framework to enhance the discriminability and
generalization of cross-modal representations.
Image-level modality alignment methods use generative
networks or image enhancement techniques to eliminate modal
differences between pixels. Ye et al. [24] introduce a Channel
exchangeable Augmentation (CA) method to enhance the generalization of visible stream. Li et al. [25] design a lightweight
network to transform visible images into X -modality in selfsupervised manner. Zhang et al. [26] propose a non-linear
simple generator to synthesize cross-modal images to middle
modality. Furthermore, several works [8], [9], [27], [28] successfully employ Generative Adversarial Networks (GANs) to
achieve the transitions between visible and infrared modalities
while preserving identity information. However, it is noted that
modality alignment methods may inevitably introduce noise
and may not be suitable for scenarios demanding high realtime performance.
B. Unsupervised Single-Modality person ReID
Unsupervised Single-Modality Person ReID (USL-ReID)
tries to tackle the demanding and time-intensive task of assigning labels to visible modality images. Recently, mainstream
methods fine-tune the network by assigning labels to the
data through clustering algorithms. Memory-based learning

JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020

3

Fig. 2. The flowchart of Dynamic Modality-Camera Invariant Clustering (DMIC) model. Our DMIC model is composed of three key modules: ModalityCamera Invariant Expansion (MIE), Dynamic Neighborhood Clustering (DNC), and Hybrid Modality Contrastive Learning (HMCL). MIE fuses the distance
encoding from multiple cameras to generate modality-camera invariant embeddings for clustering. DNC employs two dynamic search strategies that optimize
the network’s performance, transitioning from improving discriminability to enhancing generalization. The estimated pseudo labels from MIE and DNC are
used to initialize instance-level and cluster-level memories. HMCL includes intra-modality and inter-modality contrastive learning to learn modality-camera
invariant representations. During the testing phase, our framework only utilizes the backbone for testing purposes.

methods are then used to optimize the relationships between
instances and clusters. SPCL [29] design unified contrastive
learning to distinguish inter-cluster distributions. ClusterContrast [30] refines InfoNCE [31] and presents ClusterNCE,
effectively enhancing the optimization of unsupervised clusters
and improving the performance of unsupervised methods.
Lan et al. [32] introduce multi-view features to contrastive
framework, which enables efficient mining of partial cues and
the refinement of pseudo labels. To solve camera discrepancy
problem, Xuan et al. [33] divide unsupervised learning into
inter-camera training and intra-camera training to generate
reliable pseudo labels for cross-camera data. Zhang et al. [34]
propose time-based camera contrastive learning to select the
hardest camera centroid as a proxy for each cluster. ICE [35]
design cross-camera proxy contrastive loss to mitigate camera
discrepancy.
C. Unsupervised Visible-Infrared person ReID
The existing Unsupervised Learning Visible-Infrared Person
Re-Identification (USL-VI-ReID) methods mainly focus on
establishing cross-modal associations. ADCA [15] adopt count
priority selection method to facilitate cross-modal fusion.
Wu et al. [16] design two cross-modal graphs to discover
correspondences between different modalities. Pang et al.
[17] extract three channels from visible images, conduct
clustering with infrared images, and subsequently utilize IoU
for label refinement. Liang et al. [18] pretrain model with

labeled single-modality dataset and introduce a homogeneousto-heterogeneous training method. Wang et al. [20] utilize
optimal transport techniques to transfer label knowledge from
the visible modality to the infrared modality. However, the
above-mentioned methods do not take into account the impact
of camera differences in clustering phase. Especially, the
susceptibility of visible images to camera discrepancy will
lead to excessive identity splitting. In this case, using one-toone or one-to-many association methods can exacerbate the
impact of noisy labels. Unlike the above-mentioned methods,
we fully utilize camera information throughout the clustering
process to solve the identity splitting problem. Like ICE [35]
and CAP [55], GUR [37] and DCCL [36] cluster the samples
within camera and employ cross-camera proxy contrastive
loss to solve camera discrepancy. Unlike these methods, we
do not need to perform clustering within individual cameras.
Instead, we integrate camera information in a global clustering
approach and introduce two dynamic search strategies to
address camera differences.

III. T HE P ROPOSED M ODEL
In this section, we present a Dynamic Modality-Camera
Invariant Clustering (DMIC) framework to simultaneously
reduce cross-modality and cross-camera discrepancies. Our
framework is illustrated in Fig. 2.


codex
我按 `pdftotext -l 3` 只读了前三页。下面重点不是复述，而是提炼可迁移的发文套路。

**逐篇提取**

1. `Disentangling Modality and Posture Factors`
(1) 类型：新机制。  
(2) 能发的点：把 VI-ReID 的误差拆成模态差异和姿态差异两类因素，指出已有方法只处理模态，且只在 batch 内对齐。证据链是 memory 全训练集对齐、骨架辅助姿态解耦、正交分解消融。  
(3) story：跨模态 ReID 不是单一模态 gap，而是模态和姿态双因素纠缠，必须同时剥离。

2. `Distribution aligned semantics adaption`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：把 LReID 从“旧模型蒸馏加旧样本回放”改写成“预训练人本语义的低成本持续适配”。它抓住隐私、开放集、分类器维度过大三个现实 gap。  
(3) story：不用保存旧数据，也不用蒸馏旧分类器，保存大规模预训练得到的通用人体语义，再用 BN 和轻量语义适配到新域。

3. `Diverse Representations Embedding`
(1) 类型：新机制，偏结构设计。  
(2) 能发的点：用 transformer 多个 class token 给同一实例生成多种表示，解决 LReID 中“记旧”和“学新”的冲突。证据链是多表示、ACM、KU/KP 和多 seen/unseen 数据集。  
(3) story：一个 embedding 不够承载持续学习，多表示分别承担丰富性和判别性。

4. `DiverseReID`
(1) 类型：新机制。  
(2) 能发的点：DG ReID 里不是数据量不够，而是源域风格空间没有被充分展开。DSH 用 Wasserstein 距离自适应混合风格，DED 拉开域专家。  
(3) story：泛化靠最大化风格多样性，而不是手调 MixStyle 或堆更多源域。

5. `Domain Consistency Representation Learning`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：把 LReID 的矛盾表述成“域内判别”和“域间一致”的冲突。用属性文本和属性表示当桥，既学细粒度身份差异，又减轻遗忘。  
(3) story：旧方法只顾减少域间 gap，牺牲了新域身份判别；属性级表示能同时服务两边。

6. `Dual-level modality debiasing`
(1) 类型：问题重定义加因果机制。  
(2) 能发的点：指出无监督 VI-ReID 两阶段流程会把第一阶段学到的模态偏置传到伪标签和特征里。证据链是模型层 CAI 因果干预，优化层 CBT 打断数据、标签、特征传播。  
(3) story：无监督跨模态的失败不是普通噪声，而是模态偏置传播链。

7. `Dual-Modality-Shared Learning and Label Refinement`
(1) 类型：工程组合加机制。  
(2) 能发的点：伪标签噪声和 hard sample 被忽略。用跨模态 cluster 匹配、关系修标签、加权共享 memory。  
(3) story：跨模态伪标签不是一次匹配完事，要匹配、修正、再按样本难度建记忆。

8. `Dynamic adaptive multi-view contrastive learning`
(1) 类型：新机制。  
(2) 能发的点：无监督 ReID 中相机变化导致伪标签噪声，均值 proxy 又不适合非球形 cluster。证据链是动态相机 Jaccard、medoid proxy、instance proxy、软标签蒸馏。  
(3) story：聚类、proxy 和标签都不能是静态的，要随训练过程动态适配。

9. `Dynamic Modality-Camera-Invariant Clustering`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：USL-VI 只管模态差异，忽略相机差异，导致同一身份过切分。证据链是 modality-camera distance coding、动态邻域搜索、混合模态对比学习。  
(3) story：无监督 VI-ReID 的伪标签失败来自模态和相机共同作用，不是单一模态 gap。

**这批给我们的启发**

这批能发的套路很明确：强论文不是说“我加了一个模块”，而是先把失败原因改写成一个更准确的问题，例如偏置传播链、域内判别和域间一致的冲突、相机加模态共同过切分、非球形 cluster proxy 失真。机制再围绕这个问题做动态化、因果化、语义化或几何化。

**候选方向**

1. Headline：跨视角 ReID 的核心 gap 不是风格差异，而是 3D 投影造成的身份证据重排。  
挂靠资产：aerial-ground ReID 加 SMPL 3D 几何，加 SOLIDER-Swin。  
区别：最像 DMPF 和 DMIC，但它们处理的是 VI-ReID 的模态、姿态、相机差异。我们切的是航拍到地面的 3D 投影差异，同一个人体部位在图像中尺度、可见性和形变全部变了。机制可以是把 2D token 投到 SMPL canonical body surface，再在 canonical 空间做身份表示，view-specific residual 单独建模。  
cheap kill-switch：不用训练新模型，先在 CARGO 或 AG-ReID.v2 上用现有 SMPL/pose 估计，把 SOLIDER token 按身体区域重聚合，比较原始全局特征和 canonical part 特征的跨视角同 ID 相似度。如果同 ID 排名或正负间隔没有明显改善，这条先停。

2. Headline：无监督 aerial-ground ReID 的伪标签噪声是视角可见性噪声，不是普通聚类噪声。  
挂靠资产：aerial-ground 数据和 pose 热图门控，必要时加 SMPL 可见性。  
区别：最像 DAMCL 和 DMIC。它们用相机 ID、Jaccard、动态邻域修 cluster，但仍然把相机差异当特征空间距离问题。我们把过切分和误合并解释为身体可见性不一致，例如航拍只稳定看到头肩轮廓，地面看到躯干衣着。机制是 visibility-aware clustering，先用 pose/SMPL 估计每张图可靠身体区域，再让伪标签只在可比区域上扩张。  
cheap kill-switch：拿已有有标签数据，模拟无监督聚类后统计 false split 和 false merge 是否被 pose visibility mismatch 显著解释。如果可见性指标不能预测聚类错误，或者简单几何重加权不能提升伪标签纯度，就不继续。

3. Headline：终身 ReID 不该只保存旧外观知识，而应保存可跨域复用的“人体几何语义坐标”。  
挂靠资产：SOLIDER-Swin 加 SMPL 3D 几何。  
区别：最像 DASA 和 DCR。DASA 用预训练卷积语义和 BN 适配，DCR 用属性文本做一致性桥。我们的切开点是 3D 几何坐标，它不是属性词，也不是旧分类器，而是跨域、跨视角、跨衣着都相对稳定的中间表示。机制可以是冻结 SOLIDER 主干，只训练 geometry-conditioned adapter，让旧域知识以 SMPL body surface token 或 joint-token 原型保存。  
cheap kill-switch：在现有顺序数据上做一个轻量探针，比较普通 SOLIDER 特征和 SMPL 分区后的特征在旧域遗忘样本上的互补性。如果 SMPL 分区特征不能挽回旧域近邻，或者和原特征高度冗余，这条不押。

4. Headline：泛化 ReID 的风格增强应该遵守物理视角约束，而不是任意混合 feature statistics。  
挂靠资产：aerial-ground ReID，加 SMPL 和 SOLIDER。  
区别：最像 DiverseReID。它强调动态风格多样性，但风格是统计意义上的。我们的切开点是 aerial-ground 的风格变化有物理原因，包括俯视角、低清、人体部位尺度压缩和遮挡模式变化。机制是 geometry-conditioned style hallucination，只生成符合视角和可见性的特征扰动。  
cheap kill-switch：先做零训练分析，按估计视角高度或 SMPL 投影误差分桶，看跨域检索错误是否随这些物理量单调变化。没有单调关系，就说明物理视角约束不是主因，别硬做。

我更建议优先首验第 1 和第 2 个。它们和这批论文的套路最接近，但又能用我们的 aerial-ground 和 SMPL 资产切出清楚差异。第 3 个适合接当前 LReID 线，但要小心别落成 DASA/DCR 的 3D 版工程组合。
tokens used
77,121
我按 `pdftotext -l 3` 只读了前三页。下面重点不是复述，而是提炼可迁移的发文套路。

**逐篇提取**

1. `Disentangling Modality and Posture Factors`
(1) 类型：新机制。  
(2) 能发的点：把 VI-ReID 的误差拆成模态差异和姿态差异两类因素，指出已有方法只处理模态，且只在 batch 内对齐。证据链是 memory 全训练集对齐、骨架辅助姿态解耦、正交分解消融。  
(3) story：跨模态 ReID 不是单一模态 gap，而是模态和姿态双因素纠缠，必须同时剥离。

2. `Distribution aligned semantics adaption`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：把 LReID 从“旧模型蒸馏加旧样本回放”改写成“预训练人本语义的低成本持续适配”。它抓住隐私、开放集、分类器维度过大三个现实 gap。  
(3) story：不用保存旧数据，也不用蒸馏旧分类器，保存大规模预训练得到的通用人体语义，再用 BN 和轻量语义适配到新域。

3. `Diverse Representations Embedding`
(1) 类型：新机制，偏结构设计。  
(2) 能发的点：用 transformer 多个 class token 给同一实例生成多种表示，解决 LReID 中“记旧”和“学新”的冲突。证据链是多表示、ACM、KU/KP 和多 seen/unseen 数据集。  
(3) story：一个 embedding 不够承载持续学习，多表示分别承担丰富性和判别性。

4. `DiverseReID`
(1) 类型：新机制。  
(2) 能发的点：DG ReID 里不是数据量不够，而是源域风格空间没有被充分展开。DSH 用 Wasserstein 距离自适应混合风格，DED 拉开域专家。  
(3) story：泛化靠最大化风格多样性，而不是手调 MixStyle 或堆更多源域。

5. `Domain Consistency Representation Learning`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：把 LReID 的矛盾表述成“域内判别”和“域间一致”的冲突。用属性文本和属性表示当桥，既学细粒度身份差异，又减轻遗忘。  
(3) story：旧方法只顾减少域间 gap，牺牲了新域身份判别；属性级表示能同时服务两边。

6. `Dual-level modality debiasing`
(1) 类型：问题重定义加因果机制。  
(2) 能发的点：指出无监督 VI-ReID 两阶段流程会把第一阶段学到的模态偏置传到伪标签和特征里。证据链是模型层 CAI 因果干预，优化层 CBT 打断数据、标签、特征传播。  
(3) story：无监督跨模态的失败不是普通噪声，而是模态偏置传播链。

7. `Dual-Modality-Shared Learning and Label Refinement`
(1) 类型：工程组合加机制。  
(2) 能发的点：伪标签噪声和 hard sample 被忽略。用跨模态 cluster 匹配、关系修标签、加权共享 memory。  
(3) story：跨模态伪标签不是一次匹配完事，要匹配、修正、再按样本难度建记忆。

8. `Dynamic adaptive multi-view contrastive learning`
(1) 类型：新机制。  
(2) 能发的点：无监督 ReID 中相机变化导致伪标签噪声，均值 proxy 又不适合非球形 cluster。证据链是动态相机 Jaccard、medoid proxy、instance proxy、软标签蒸馏。  
(3) story：聚类、proxy 和标签都不能是静态的，要随训练过程动态适配。

9. `Dynamic Modality-Camera-Invariant Clustering`
(1) 类型：问题重定义加新机制。  
(2) 能发的点：USL-VI 只管模态差异，忽略相机差异，导致同一身份过切分。证据链是 modality-camera distance coding、动态邻域搜索、混合模态对比学习。  
(3) story：无监督 VI-ReID 的伪标签失败来自模态和相机共同作用，不是单一模态 gap。

**这批给我们的启发**

这批能发的套路很明确：强论文不是说“我加了一个模块”，而是先把失败原因改写成一个更准确的问题，例如偏置传播链、域内判别和域间一致的冲突、相机加模态共同过切分、非球形 cluster proxy 失真。机制再围绕这个问题做动态化、因果化、语义化或几何化。

**候选方向**

1. Headline：跨视角 ReID 的核心 gap 不是风格差异，而是 3D 投影造成的身份证据重排。  
挂靠资产：aerial-ground ReID 加 SMPL 3D 几何，加 SOLIDER-Swin。  
区别：最像 DMPF 和 DMIC，但它们处理的是 VI-ReID 的模态、姿态、相机差异。我们切的是航拍到地面的 3D 投影差异，同一个人体部位在图像中尺度、可见性和形变全部变了。机制可以是把 2D token 投到 SMPL canonical body surface，再在 canonical 空间做身份表示，view-specific residual 单独建模。  
cheap kill-switch：不用训练新模型，先在 CARGO 或 AG-ReID.v2 上用现有 SMPL/pose 估计，把 SOLIDER token 按身体区域重聚合，比较原始全局特征和 canonical part 特征的跨视角同 ID 相似度。如果同 ID 排名或正负间隔没有明显改善，这条先停。

2. Headline：无监督 aerial-ground ReID 的伪标签噪声是视角可见性噪声，不是普通聚类噪声。  
挂靠资产：aerial-ground 数据和 pose 热图门控，必要时加 SMPL 可见性。  
区别：最像 DAMCL 和 DMIC。它们用相机 ID、Jaccard、动态邻域修 cluster，但仍然把相机差异当特征空间距离问题。我们把过切分和误合并解释为身体可见性不一致，例如航拍只稳定看到头肩轮廓，地面看到躯干衣着。机制是 visibility-aware clustering，先用 pose/SMPL 估计每张图可靠身体区域，再让伪标签只在可比区域上扩张。  
cheap kill-switch：拿已有有标签数据，模拟无监督聚类后统计 false split 和 false merge 是否被 pose visibility mismatch 显著解释。如果可见性指标不能预测聚类错误，或者简单几何重加权不能提升伪标签纯度，就不继续。

3. Headline：终身 ReID 不该只保存旧外观知识，而应保存可跨域复用的“人体几何语义坐标”。  
挂靠资产：SOLIDER-Swin 加 SMPL 3D 几何。  
区别：最像 DASA 和 DCR。DASA 用预训练卷积语义和 BN 适配，DCR 用属性文本做一致性桥。我们的切开点是 3D 几何坐标，它不是属性词，也不是旧分类器，而是跨域、跨视角、跨衣着都相对稳定的中间表示。机制可以是冻结 SOLIDER 主干，只训练 geometry-conditioned adapter，让旧域知识以 SMPL body surface token 或 joint-token 原型保存。  
cheap kill-switch：在现有顺序数据上做一个轻量探针，比较普通 SOLIDER 特征和 SMPL 分区后的特征在旧域遗忘样本上的互补性。如果 SMPL 分区特征不能挽回旧域近邻，或者和原特征高度冗余，这条不押。

4. Headline：泛化 ReID 的风格增强应该遵守物理视角约束，而不是任意混合 feature statistics。  
挂靠资产：aerial-ground ReID，加 SMPL 和 SOLIDER。  
区别：最像 DiverseReID。它强调动态风格多样性，但风格是统计意义上的。我们的切开点是 aerial-ground 的风格变化有物理原因，包括俯视角、低清、人体部位尺度压缩和遮挡模式变化。机制是 geometry-conditioned style hallucination，只生成符合视角和可见性的特征扰动。  
cheap kill-switch：先做零训练分析，按估计视角高度或 SMPL 投影误差分桶，看跨域检索错误是否随这些物理量单调变化。没有单调关系，就说明物理视角约束不是主因，别硬做。

我更建议优先首验第 1 和第 2 个。它们和这批论文的套路最接近，但又能用我们的 aerial-ground 和 SMPL 资产切出清楚差异。第 3 个适合接当前 LReID 线，但要小心别落成 DASA/DCR 的 3D 版工程组合。
