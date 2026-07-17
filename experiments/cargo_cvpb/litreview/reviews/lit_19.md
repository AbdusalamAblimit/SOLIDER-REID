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
session id: 019ef71f-d64e-71e3-b435-61765a4e7913
--------
user
你是 ReID 论文创新挖掘员 19/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(19/20):
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会按项目规则先补齐当前目标和项目纪律，再只用 `pdftotext -l 3` 读这 5 篇的前三页。这个任务只需要抽取论文套路和反推创新点，不会改文件。
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
我已读完当前目标和状态，任务仍然是读文献做差距分析并提出候选方向。下面我先核对文件名，然后并行抽取每篇前三页文本，只看摘要和引言范围。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
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

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

1

UP-Person: Unified Parameter-Efficient Transfer
Learning for Text-based Person Retrieval

Abstract—Text-based Person Retrieval (TPR) as a multimodal task, which aims to retrieve the target person from a
pool of candidate images given a text description, has recently
garnered considerable attention due to the progress of contrastive
visual-language pre-trained model. Prior works leverage pretrained CLIP to extract person visual and textual features
and fully fine-tune the entire network, which have shown notable performance improvements compared to uni-modal pretraining models. However, full-tuning a large model is prone
to overfitting and hinders the generalization ability. In this
paper, we propose a novel Unified Parameter-Efficient Transfer Learning (PETL) method for Text-based Person Retrieval
(UP-Person) to thoroughly transfer the multi-modal knowledge
from CLIP. Specifically, UP-Person simultaneously integrates
three lightweight PETL components including Prefix, LoRA and
Adapter, where Prefix and LoRA are devised together to mine
local information with task-specific information prompts, and
Adapter is designed to adjust global feature representations.
Additionally, two vanilla submodules are optimized to adapt
to the unified architecture of TPR. For one thing, S-Prefix is
proposed to boost attention of prefix and enhance the gradient
propagation of prefix tokens, which improves the flexibility and
performance of the vanilla prefix. For another thing, L-Adapter is
designed in parallel with layer normalization to adjust the overall
distribution, which can resolve conflicts caused by overlap and
interaction among multiple submodules. Extensive experimental
results demonstrate that our UP-Person achieves state-of-the-art
results across various person retrieval datasets, including CUHKPEDES, ICFG-PEDES and RSTPReid while merely fine-tuning
4.7% parameters. Code is available at https://github.com/LiuYating/UP-Person.
Index Terms—Text-based Person Retrieval, parameter-efficient
transfer learning, unified architecture, cross-modal retrieval.

I. I NTRODUCTION

T

EXT-based Person Retrieval (TPR) [1] aims to locate the
person of interest from a large pool of candidates given
a pedestrian description, which is a cross-task that integrates
Manuscript initially received on May 2, 2024. Revised version received on
October 16, 2024, January 31, 2025. This work was supported by the National
Natural Science Foundation of China NSFC under U23B2030, and the Major
Key Project of Peng Cheng Laboratory under PCL2023A08. (Corresponding
author: Zimo Liu and Qingmin Liao.)
Yating Liu is with Shenzhen International Graduate School, Tsinghua
University, Shenzhen 518071, China and Peng Cheng Laboratory, Shenzhen
518071, China (e-mail: liuyatin21@mails.tsinghua.edu.cn).
Yaowei Li is with School of ECE, Peking University, Shenzhen 518071,
China and Peng Cheng Laboratory, Shenzhen 518071, China (e-mail:
ywl@stu.pku.edu.cn).
Xiangyuan Lan and Zimo Liu are with Peng Cheng Laboratory, Shenzhen
518071, China (e-mail: lanxy@pcl.ac.cn; liuzm@pcl.ac.cn).
Wenming Yang and Qingmin Liao are with Shenzhen International
Graduate School, Tsinghua University, Shenzhen 518071, China (e-mail:
yang.wenming@sz.tsinghua.edu.cn; liaoqm@tsinghua.edu.cn).

Knowledge of CLIP (KC)
TPR-specific Knowledge of CLIP (PKC)
TPR-specific Knowledge from Training datasets (PKT)

(a)

Fine-tuned
Frozen

PKC + PKT

…

Image
Encoder

Text
Encoder

PETL
module

Parameter-Efficient Transfer Learning

>

arXiv:2504.10084v1 [cs.CV] 14 Apr 2025

Yating Liu, Yaowei Li, Xiangyuan Lan,
Wenming Yang, Senior Member, IEEE, Zimo Liu, and Qingmin Liao, Senior Member, IEEE

PKT

Image
Encoder

Text
Encoder

Full-tuning

Model

Knowledge

TPR-specific Knowledge

(b)

R@1

Ours
Full Tuning

Ours
74.17%

+5.98%

Full Tuning

68.19%

7.4M

-95.1%

Fine-tuned Params
150M

Fig. 1. The motivation for our proposed method. (a) shows how PETLbased methods can transfer TPR-specific knowledge from both CLIP and
training data, whereas full-tuning relies solely on the training data as its
knowledge source. Full-tuning (lower) only utilizes the knowledge of the
pre-trained CLIP at initialization and almost loses the original knowledge,
which thus only retains the knowledge of TPR from training datasets (PKT).
PETL (upper) fine-tunes a small parameters and keeps CLIP backbone frozen,
which means that it can integrate both TPR-specific within CLIP (PKC) due
to the retained parameters of original CLIP and TPR-specific knowledge
from training data (PKT). Therefore, PETL methods can incorporate more
knowledge compared to full-tuning if rationally designed. (b) On CUHKPEDES, our approach reduces 95.1% training parameters and gains an
improvement by 5.98% on R@1 compared to the full-tuning CLIP.

person re-identification (Re-ID) [2] with cross-modal retrieval
[3]. The core of TPR is to establish the matching relationship
between person images and texts. Compared to conventional
image-based person retrieval (Re-ID) [2], [4]–[6] and attributebased person retrieval [7], text-based person retrieval [8] provides a more intuitive and convenient way by forming queries
with natural language descriptions, thus attracts increasing
attention from both academia and industry, benefiting a variety
of applications, such as security surveillance and intelligent
transportation.
As large foundation models expand, the embedded knowledge becomes increasingly abundant. Consequently, it is crucial to explore how to effectively transfer these pre-trained
models to downstream tasks in order to maximize knowledge
utilization. The great success in Vision-Language Pre-training
(VLP) has recently shown a strong cross-modal transfer capability in various vision-language understanding tasks [9] ,
where the most representative work is Contrastive Language-

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Image Pre-training (CLIP) [10]. With 400M web-crawled
trainable image-text pairs, CLIP contains abundant generic
knowledge learned from the large-scale dataset. Besides, the
pre-trained encoders of CLIP have greater cross-modal matching potential due to its two-branch contrastive architecture
than uni-modal encoders, i.e., ViT [11] for vision, BERT [12]
for language. Therefore, several works [13], [14] adopt CLIP
as the backbone, and propose multi-level matching modules
to achieve CLIP-based cross-modal interactions from coarse
to fine with fully fine-tuning, which leads to a significant
improvement compared to many previous uni-modal retrieval
frameworks [15]–[18]. However, this full-tuning paradigm
faces two fatal issues: (1) it has a high risk of overfitting
on limited task-specific training datasets as the scale of large
pre-trained models continues to increase, and (2) training
and storing a completely new large model for each dataset
independently is expensive in practice.
An elegant solution to the above-mentioned problems
is Parameter-Efficient Transfer Learning (PETL) [19], i.e.,
Adapter [20], LoRA [21], Prefix [22], Prompt [23] and other
related variants [24]–[26], which can achieve comparable or
superior performance only with fine-tuning a few parameters of large models. As shown in Figure 1(a), full-tuning
(lower) only utilizes the knowledge of the pre-trained CLIP
at initialization stage and almost loses the original knowledge
embedded in network. On the contrary, the CLIP backbone
is frozen and preserved in PETL paradigm (upper). Guided
by PETL-related components, TPR-specific knowledge within
CLIP is effectively transferred as the parameters of the original
CLIP remain unchanged. The rich knowledge of CLIP about
visual objects and textual descriptions can serve as knowledge
complementarity for fine-grained TPR task. Consequently,
PETL facilitates learning both task-specific knowledge from
general CLIP and knowledge from the training dataset, making
it superior to full-tuning in terms of knowledge retention,
particularly in our scenarios with limited training data.
However, PETL paradigm is not well explored in TPR.
CSKT [27] makes the first attempt to explore CLIP with
PETL-related methods on TPR and proposes a novel bidirectional multi-modal prompt-tuning, which attains superior
performance only with fine-tuning 7.4% parameters of CLIP.
This existing PETL-based method only focuses on the global
feature representation, and has not sufficiently transferred both
global and local information from multiple views of CLIP. The
depth of exploration by PETL on TPR still remains limited.
This prompts us to consider whether it is feasible to design a
unified PETL framework for TPR from a more comprehensive
perspective.
In this paper, our target is to investigate how to design a simple, effective and parameter-efficient unified transfer learning architecture based on multiple lightweight PETL methods.
A serious concern on unifying various PETL methods is
component conflict, where the performance drops significantly
in practice when a single PETL component is incorporated into
a unified framework. This occurs as the original structures of
multiple PETL components overlap and interact, potentially
disrupting the intended optimization direction. Consequently,
it is crucial to design a unified framework where different

2

components do not interfere with each other and can work
cooperatively. Another issue we observe is that when the
vanilla prefix component is adapted to TPR, it shows a poor
performance compared to the approximate PETL submodule
such as prompt-tuning in CSKT [27]. Thus, we consider
whether we could optimize the vanilla PETL components for
better synergy in the unified framework.
To address the aforementioned issues, we propose a novel
Unified Parameter-Efficient Transfer Learning (PETL) method
for Text-based Person Retrieval (UP-Person). UP-Person implements a comprehensive PETL-based method to enhance
knowledge transferring without requiring any additional complex cross-modal interaction modules. Specifically, as shown
in Figure 2, in a transformer block of CLIP, we design and
unify three submodules based on Prefix, LoRA and Adapter
to adapt to TPR task. LoRA is incorporated to modify the
weights of Multi-Head Attention (MHA), allowing the model
to capture more nuanced and local features and relationships
in TPR that the original CLIP cannot fully uncover. Prefix
is prepended to the keys and values of MHA to enrich taskspecific information for TPR. Adapter is designed in layer normalization (layernorm) to adjust the overall distribution from
a global perspective. Meanwhile, we propose two improved
submodules: Salable Prefix (S-Prefix) and Layernorm Adapter
(L-Adapter). S-Prefix is introduced to enhance gradient backward propagation of prefix embeddings, which improves both
the flexibility and performance of the vanilla prefix. L-Adapter
is designed in parallel with layernorm, alongside the residual
connection, to get rid of component conflicts. Our method
demonstrates greater advantages in data-scarce scenarios especially in RSTPReid dataset. It achieves better performance
and parameter-efficiency with negligible sacrifice in inference
efficiency, and reduces computation and storage costs. Our
contributions can be summarized as follows:
• We propose a unified parameter-efficient transfer learning
method for text-based person retrieval. To the best of our
knowledge, our study is the first attempt to investigate
the unified PETL framework for TPR, which effectively
transfers both global and local knowledge, along with
task-specific knowledge, to TPR task with very fewer
computation and storage costs.
• To optimize the PETL components for better synergy,
we further develop two improved PETL components,
respectively: (1) a scalable prefix in attention named SPrefix, and (2) a novel adapter L-Adapter designed in
parallel with layernorm of blocks.
• Extensive experiments show that UP-Person achieves
superior performance compared with the prior state-ofthe art on three public datasets while merely fine-tuning
4.7% parameters.

II. R ELATED W ORK
In this section, we will briefly review the most relevant study including vision-language pre-training, parameterefficient transfer learning, and text-based person retrieval.

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

A. Vision-Language Pre-training.
Vision-Language Pre-training (VLP), incorporating both
image-text and video-text pre-training, focuses on learning
semantic correspondence between heterogeneous modalities
through pre-training on large datasets. Generally, VLP models
include five modules: vision encoder, text encoder, multimodal
fusion, decoder (optional) and pre-training objective.
For vision encoder, VLP models in recent works [10],
[28], [29] adopt a pre-trained vision transformer to encode
patch embeddings, such as ViT [11] or swin transformer [30].
For language, textual pre-trained transformer like BERT [12]
can be utilized to encode word embeddings. Furthermore,
models can be categorized into two types from the multimodal fusion perspective: dual-stream and single-stream. For
former, text and visual features are independently encoded by
two transformer branches [10], [28], [29]. The latter singlestream models [31], [32] concatenate the text and visual
features and then input them into a single shared transformer
structure, which have less parameters compared with dualstream models. In addition to VLP models that solely employ
an encoder, the encoder-decoder architecture [32] incorporates
a decoder structure, which feeds the representations into a
encoder first and then sends outputs to an decoder. Decoder
is extremely helpful when combining other generative tasks
such as image captioning and Visual Question Answering
(VQA). The VLP models are trained in an end-to-end manner
under the guidance of the pre-training objectives [9], such
as Masked Language Modeling (MLM), Next Sequence Prediction (NSP), Masked Vision Modeling (MVM), Image-Text
Matching (ITM), Image-Text Contrastive (ITC), Word-Region
Alignment (WGA) and other task-specific objectives.
In addition to model architecture, the huge training datasets
are crucial to performance improvement of VLP models. The
common pre-training datasets like LAION-400M [33] and
LAION-5B [34] are mainly composed of numerous public
image-caption pairs, visual questions and answers, or larger
crawled data from internet. The abundant knowledge embedded in VLP can be applied to a broad range of downstream tasks including cross-modal classification, regression,
generation and retrieval. In this work, we focus on text-based
person retrieval based on vision-language pre-training model
CLIP (400M trainable image-text pairs) due to its general
knowledge, and aim to bridge the gap between pre-training
large models and downstream tasks by transfer learning.

B. Parameter-Efficient Transfer Learning.
With the rapid advancement of large models, parameterefficient transfer learning (PETL) [19] has gained substantial
attention from researchers . In this work, we focus on leveraging vision-language models to effectively transfer and enhance
knowledge for the downstream TPR task, with a particular
emphasis on PETL techniques.
As the size of pre-trained models continues to increase, from
language models like BERT [12] with 110 million parameters
to Yi-VL-34B [35] with 34 billion parameters and llama 3 [36]
with 70 billion parameters, full fine-tuning will be more and

3

more time-consuming, computationally expensive and storageinefficient, especially for multiple domains. PETL has emerged
as a viable strategy to compensate for the above disadvantages
of full-tuning, which can be broadly categorized into three
types [19]: additive fine-tuning, reparameterized fine-tuning
and selective fine-tuning. Additive fine-tuning methods such
as Adapter [37], Prompt [23] and Prefix [22] were initially
introduced to facilitate the transfer of large language models to
specific downstream tasks by inserting additional parameters to
models. LoRA [21] as a representative method for reparameterized fine-tuning methods, utilizes low-rank decomposition
to reconstruct the weight matrices. Selective fine-tuning aims
to reduce the number of fine-tuned parameters by selecting a
subset of pre-trained parameters, such as Bitfit [38].
Subsequently, inspired by PETL in NLP, solutions like
VPT [39] and AdapterFormer [40] have emerged to address
challenges in vision transfer learning. With the success of
VLP [9], PETL on VLP becomes a novel trend. Cross-modal
prompt called MaPLe [41] and cross-modal adapter [42] are
proposed in both vision and language branches and further
achieve cross-modal interactions. In CSKT [27], PETL is first
successfully incorporated in CLIP for TPR, which provides
an effective solution by reducing the number of fine-tuning
parameters and training time while achieving comparable
performance to full fine-tuning. In this work, our aim is to
develop a more effective and parameter-efficient unified PETL
method based on CLIP for TPR, which can transfer more
comprehensive information to achieve better performance
while fine-tuning fewer parameters. We focus on three most
representative PETL approaches, including Adapter, LoRA
and Prefix.
C. Text-based Person Retrieval.
Text-based Person Retrieval (TPR) was first proposed by Li
et al. [1] to solve the problem that the target query images are
not always available in real-world scenario, which is a trending
topic in intelligent surveillance research [2], [43]–[46], which
also includes crowd counting, object detection and tracking,
person re-identification, anomaly detection, etc. The central
challenge of TPR lies in aligning the person image and text
from different modals efficiently.
An early trend in TPR is adopting different uni-modal
backbones [15]–[18], [47], [48] such as ResNet, ViT, LSTM
or BERT to extract vision and language features, and then
two types of representations are aligned by global or local
matching methods. Global matching methods [1], [16], [49]
align images and texts into a joint embedding space by designing cross-modal matching loss functions. Although global
matching is simple and efficient, it struggles to comprehend
more localized information. This limitation often leads to poor
retrieval performance, even when a synthetic loss function is
employed. Therefore, local matching is proposed to explicitly
explore visual-textual salient part pairs for semantic alignment,
e.g., human body parts, person strips or regions for image
[47], [50]–[52], phrases or words for text [15], [52]–[54].
Afterwards, implicit matching mechanisms are adopted to
extract subtle visual-textual cues such as hairstyle and logo


exec
/bin/zsh -lc "pdftotext -l 3 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Visible-Infrared Person Re-Identification Based on Feature
Decoupling and Refinement
HAO DING, South-Central Minzu University, Wuhan, China
JING SUN, Shenzhen Institute of Advanced Technology Chinese Academy of Sciences, Shenzhen,
China

RUI LONG, XIAOPING JIANG, HONGLING SHI, and YUTING QIN, South-Central Minzu
University, Wuhan, China
ZONGZE LI, Beidou Intelligent Technology (Shenzhen) Company Limited, Shenzhen, China
JIAN-JIN LI, Université Clermont Auvergne, Clermont-Ferrand, France
The objective of visible-infrared person re-identification is to accurately match pedestrian images captured
in different modalities. Since these images are taken from varying viewpoints by different cameras, the
cross-modal detection task must address both modality discrepancies and camera variations. Many existing
approaches primarily focus on minimizing inter-modality differences to enhance retrieval accuracy, often
overlooking the impact of camera viewpoint differences. To tackle these challenges, this article introduces
a hierarchical feature decoupling network. First, the network decouples and extracts camera-related and
camera-irrelated features separately to mitigate the effects of camera variations. Second, it addresses modality
differences by extracting modality-independent features. Additionally, an adversarial decoupling loss is
employed to further disentangle identity-irrelevant information from identity-relevant features, thereby
boosting the system’s accuracy and robustness. Extensive experiments conducted on the SYSU-MM01 and
RegDB datasets validate the effectiveness of the proposed method.
CCS Concepts: • Computing methodologies → Visual content-based indexing and retrieval;
Additional Key Words and Phrases: Person re-identification, modality discrepancies, feature decoupling

Hao Ding and Jing Sun contributed equally to this work.
This work is supported by the National Key R&D Program of China (Grant No. 2023YFC3321600) and funds of South Central
Minzu University (Grant No. CZY23026).
Authors’ Contact Information: Hao Ding, South-Central Minzu University, Wuhan, China; e-mail: dingh_scu@mail.
scuec.edu.cn; Jing Sun, Shenzhen Institute of Advanced Technology Chinese Academy of Sciences, Shenzhen, China; e-mail: jing.sun1@siat.ac.cn; Rui Long (corresponding author), South-Central Minzu University, Wuhan, China; e-mail: 2022110192@mail.scuec.edu.cn; Xiaoping Jiang, South-Central Minzu University,
Wuhan, China; e-mail: jiangxp@mail.scuec.edu.cn; Hongling Shi, South-Central Minzu University, Wuhan,
China; e-mail: hongling.shi@scuec.edu.cn; Yuting Qin, South-Central Minzu University, Wuhan, China; e-mail:
202021111218@mail.scuec.edu.cn; Zongze Li, Beidou Intelligent Technology (Shenzhen) Company Limited, Shenzhen,
China; e-mail: lizongze@szbit.cn; JianJin Li, Université Clermont Auvergne, Clermont-Ferrand, France; e-mail: jianjin.li@isima.fr.

This work is licensed under Creative Commons Attribution International 4.0.
© 2025 Copyright held by the owner/author(s).
ACM 1551-6865/2025/9-ART265
https://doi.org/10.1145/3749843
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.

265:2

H. Ding et al.

ACM Reference format:
Hao Ding, Jing Sun, Rui Long, Xiaoping Jiang, Hongling Shi, Yuting Qin, Zongze Li, and Jian-Jin Li. 2025.
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement. ACM Trans. Multimedia
Comput. Commun. Appl. 21, 9, Article 265 (September 2025), 16 pages.
https://doi.org/10.1145/3749843

1

Introduction

Person re-identification aims to match a query image of an individual with gallery images captured
by multiple non-overlapping cameras. This technology has significant practical value in surveillance
tasks, making it a research hotspot in the field of computer vision. Although most existing methods
primarily focus on visible light scenarios, utilizing color information as an important cue for
identification, these methods are limited in low-light conditions as visible light cameras struggle to
capture effective color information, thus restricting their applicability.
With the advancement of surveillance technology, an increasing number of new cameras are
equipped with both visible light and infrared dual modalities, allowing them to automatically
switch modes between day and night, thereby achieving around-the-clock monitoring. To address
this emerging technological demand, Wu et al. [40] pioneer the study of cross-modality person
re-identification between visible and infrared images, thereby establishing this research field and
significantly expanding the application scenarios of person re-identification. However, despite
the progress made, several challenges remain unresolved. One of the key issues is the significant
modality gap between visible light and infrared images, which makes it difficult to effectively
match person identities between these different modalities. The appearance differences caused
by differences in lighting conditions, background, and the inherent properties of visible light and
infrared imaging complicate feature extraction and alignment. Therefore, how to effectively bridge
the modality gap between visible light and infrared images becomes the key to improving the
accuracy and robustness of cross-modality person re-identification.
In modal-shared feature spaces, identity information is always inevitably coupled with other
identity-irrelated information. Camera-related features, induced by differences in imaging mechanisms, viewpoints, lighting, and backgrounds, lack identity consistency and often interfere with
identity learning. In contrast, camera-irrelated features robustly capture identity information and
remain stable in cross cameras and modalities. Additionally, due to the differences in imaging principles between visible and infrared spectra, cross-modality person re-identification must contend
not only with feature variations within a single modality but also with cross-modality feature
heterogeneity arising from differences in reflectance in the visible spectrum and emissivity [27, 35]
in the infrared spectrum.
Most existing works [1, 38, 41, 46] focused on extracting identity-related features or rely on a
single decoupling strategy to address the effects of modal differences, to achieve identity-based
matching and similarity ranking. However, in the modality-shared feature space, identity information is often intertwined with camera noise and modality-sensitive redundancy, resulting in the
weakening of discriminative features. This means that a single-stage decoupling strategy may find
it difficult to simultaneously mitigate the impact of modality and camera differences, resulting in a
decrease in the discriminative ability of modality-shared features, thereby affecting the performance
of cross-modal person re-identification.
To address the above challenges, this study proposes an innovative cross-modality person
reidentification framework—the Hierarchical Decoupling Network (HDNet), which includes
two main modules: Camera Decoupling Module (CDM) and Modality-Invariant Identity
Decoupling Module (MIDM). This framework aims to progressively decouple camera difference
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.

Visible-Infrared Person Re-Identification

265:3

features, modality difference features, and identity-irrelevant redundant features, and significantly
enhance the discriminative performance of cross-modality person re-identification through this
systematic feature purification mechanism. Specifically, HDNet introduces several innovative
aspects to address the challenges in visible-infrared person re-identification. First, it adopts a
hierarchical feature decoupling approach. CDM decouples camera-related features and camerairrelated features from modality-shared features to mitigate the impact of camera variations. This
is achieved by adopting an adversarial approach with Gradient Reversal Layers (GRL) [8] to
minimize the identity information in camera-related features, allowing the network to focus more
on identity-specific features. Then, the MIDM further decouples identity-dependent and identityindependent features from camera-irrelated features, using residual features to recover the lost
identity information. In MIDM, an adversarial decoupling loss is designed to optimize the feature
decoupling process and promote a better separation of identity features and redundant features. This
loss function encourages identity-dependent features to be clustered more tightly and redundant
features to be clustered more dispersedly, enhancing the discriminative power of identity features.
And, a Feature Cross-Modal Recombination Strategy (FCRS) is introduced to further reduce
the modality differences of identity features, which guides the network to focus on the shared
identity information across modalities, improving the robustness and accuracy of the system.
The research work presented in this article can be summarized in the following three aspects:
— HDNet, which includes CDM and MIDM, is an innovative end-to-end feature purification
framework that utilizes a multi-level decoupling strategy (camera-modality-identity), ensuring
maximal preservation of identity information at each level, while effectively eliminating
interference.
— CDM decouples camera-related and camera-irrelated features from modality-shared features
by adopting an adversarial approach with the GRL and minimizes the identity information
in camera-related features, thereby allowing the network to focus more on identity-specific
features.
— MIDM utilizes adversarial decoupling loss to optimize identity disentanglement information and introduces the FCRS to focus on the shared identity information by pairing and
recombining residual identity features from different modalities.
2

Related Work

Person Re-Identification. Person re-identification refers to the problem of pedestrian retrieval across
multiple surveillance cameras [57]. The main challenges arise from variations in camera environments, pose changes, and viewpoint differences. Traditional person re-identification methods
are based on visible light cameras and can be divided into two categories [15, 29, 45, 56]. One
is based on feature extraction methods [25, 31], which usually introduce the ID-discriminative
Embedding model [58] and regard each identity as a different category to obtain fine-grained
global features; or combining local features with full-body representation to make it robust to
misalignment changes. Cheng et al. [5] design a multi-channel parts-aggregated deep convolutional
network by integrating local body part features and global full-body features. The other is the
metric learning-based method [2, 12, 30], which aims to learn discriminative features by reducing
the feature distance between pedestrian images. Hermans et al. [12] demonstrate that using a
variant of the triplet loss is beneficial for discriminative Re-ID model learning. Zhou et al. [60]
present a point to set similarity for deep metric learning, which can jointly minimize the intra-class
distance and maximize the inter-class distance. Although these methods have achieved effective
performance in traditional person re-identification, they are not fully applicable to cross-modal
person re-identification, thus requiring additional methods to handle modality-specific information.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Visible-Infrared Person Re-Identification via Mutual
Reinforcement of Prompts and Image Encoders
Hongde Zhang

Bingpeng Ma∗

University of Chinese Academy of Sciences
Beijing, China
zhanghongde22@mails.ucas.ac.cn

University of Chinese Academy of Sciences
Beijing, China
bpma@ucas.ac.cn

Abstract
Contrastive Language-Image Pre-training (CLIP) has achieved good
results in Visible-Infrared Person Re-IDentification (VI-ReID) task.
However, CLIP does not focus on person-related information, so
prompts generated by original CLIP can not accurately describe
identity information of a person. We argue that compared to original CLIP, encoders familiar with person-related information can
generate prompts which are more suitable for VI-ReID. Based on
such idea, we design a novel network that helps prompts focus
on person-related information through alternately optimizing the
prompts and image encoders. Specifically, when optimizing prompts,
we introduce modality knowledge propagation loss. The loss aligns
the predicted class probability of text and image features, so that
the knowledge in image encoders is transferred to prompts. When
optimizing encoders, we design modality alignment loss. The loss
considers text features as a bridge between two modalities, aligning features from two modalities with text features. In this way,
modality discrepancies are effectively reduced. Finally, through
the mutual reinforcement of two parts, the quality of both prompts
and image encoders is improved in a positive feedback manner.
Experiments on two widely used datasets show that the proposed
network outperforms state-of-the-art methods.

Expect answer: A boy wearing a green shortsleeved shirt, black trousers and grey shoes.
He is relatively strong and has a short haircut.
User

Please describe the image.

The image shows a person walking
outdoors, surrounded by green foliage.

GPT

Fig. 1: Text descriptions generated directly by multimodal large language models often do not focus on person-related information.
Therefore, these descriptions have limited value for VI-ReID.

to their identities. As this task is crucial for achieving 24-hour
monitoring, it has received increasing attention and research. The
biggest challenge VI-ReID faces is the significant discrepancies between two modalities, so current studies focus on how to alleviate
the huge inter-modality gap.
With the rise of large models, some approaches [2, 21] use text
prompts to help alleviate inter-modality discrepancies. These works
utilize the Contrastive Language-Image Pre-training (CLIP) model
by two-stage training strategy [5]. They fix image encoders to optimize text prompts in the first stage and fix text prompts to optimize
CCS Concepts
image encoders in the second stage. The generated text prompts
• Information systems → Information retrieval.
contain high-level semantic information shared across modalities,
so they are used to eliminate differences between modalities.
Keywords
However, the text prompts generated by multimodal large language models can not effectively describe identity information of
visible-infrared person re-identification, text prompts, knowledge
a person. Taking Generative Pre-trained Transformer (GPT) as an
distillation
example, the generated descriptions often focus on background,
ACM Reference Format:
Hongde Zhang and Bingpeng Ma. 2025. Visible-Infrared Person Re-Identification poses, and other identity-irrelevant information of an image, as
shown in Fig. 1. And such descriptions can provide little assistance
via Mutual Reinforcement of Prompts and Image Encoders. In Proceedings
for VI-ReID. As for CLIP, since it does not focus on person informaof the 2025 International Conference on Multimedia Retrieval (ICMR ’25),
June 30-July 3, 2025, Chicago, IL, USA. ACM, New York, NY, USA, 5 pages.
tion, prompts generated directly by the original CLIP also can not
https://doi.org/10.1145/3731715.3733496
capture useful information for VI-ReID. Therefore, it is necessary
to improve the quality of text prompts.
1 Introduction
In this paper, we propose the Mutual Promotion Training (MPT)
network for this issue. Generally, prompts that focus on person
The Visible-Infrared person Re-IDentification (VI-ReID) task aims
information can better assist in optimizing the image encoders.
to match the visible images with the infrared images according
Conversely, prompts optimized with the constraints of trained en∗
Corresponding Author.
coders will pay more attention to person information. Based on
this idea, we attempt to introduce more person-related information into prompts. Specifically, we adopt an alternating training
This work is licensed under a Creative Commons Attribution 4.0 International License.
strategy which alternately optimizes the text prompts and image
ICMR ’25, Chicago, IL, USA
encoders in different epochs. When optimizing text prompts, we
© 2025 Copyright held by the owner/author(s).
design the modality knowledge propagation loss. By aligning preACM ISBN 979-8-4007-1877-9/2025/06
https://doi.org/10.1145/3731715.3733496
dicted class probability of features, the loss can enable prompts to

2118

ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA

Hongde Zhang and Bingpeng Ma

focus on person-related information during the training process
and encoders can benefit from continuously optimized prompts.
However, because the optimization objectives of prompts and
encoders are different, optimizing prompts and image encoders simultaneously may cause mutual interference. For example, when
prompts learn person-related information from image encoders,
the parameters of encoders are supposed not to be affected. To
avoid the mutual interference, we propose the alternating training
strategy to implement the mutual reinforcement between prompts
and encoders. Specifically, we fix image encoders (prompts) and
optimize the prompts (image encoders) when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 (𝑒𝑝𝑜𝑐ℎ =
2𝑖 + 1). Using 𝑊𝑡𝑒𝑥𝑡 (𝑊𝑖𝑚𝑔 ) to represent the parameters of text
prompts (image encoders) and 𝐹𝑡𝑒𝑥𝑡 (𝐹𝑖𝑚𝑔 ) to represent the optimization process of text prompts (image encoders), the optimization process can be formalized as:

MPT Module
Infrared
Image
Encoder

I1
I2

Infrared
Image
Encoder

IB
A photo of [X]1 [X]2
[X]3 [X]4 person.

Text
Encoder

T1 T2

prompts

Visible
Image
Encoder

IB
Modality
T B Knowledge
Propagation
Loss

A photo of [X]1 [X]2
[X]3 [X]4 person.

prompts

V1
V2

Text
Encoder

Visible
Image
Encoder

VB

2i-th epoch

I1
I2

T1 T2

Modality
T B Alignment
Loss

V1
V2
VB

(2i+1)-th epoch

Fig. 2: The MPT network introduces an alternating training strategy,
in which the modality knowledge propagation loss is designed for
optimizing text prompts and the modality alignment loss is used
to optimize image encoders. The lock represents fixed parameters,
while the star indicates parameters being optimized.

𝑓𝑖𝑥
𝑊𝑡𝑒𝑥𝑡 = 𝐹𝑡𝑒𝑥𝑡 (𝑊𝑡𝑒𝑥𝑡 , 𝑊𝑖𝑚𝑔
),

learn more person-related information from the encoders. This significantly alleviates the problem that prompts lack person-related
information. When training the image encoders, we propose the
modality alignment loss. The loss treats the text features as a special intermediate modality, and aligns both visible and infrared
modalities to the text modality. In this way, prompts can serve
as the bridge for eliminating the discrepancies between the two
modalities, helping the encoders generate better features. With the
help of MPT, text prompts can help image encoders become better,
and better image encoders can in turn promote prompts to learn
more person-related information. Under such a positive feedback,
both text prompts and image encoders will continue to improve.
We evaluate the MPT network on two widely used datasets to
demonstrate its effectiveness. Extensive experimental results show
that the MPT network indeed solves the problems existing in the
current methods and outperforms state-of-the-art methods.

2

𝑓𝑖𝑥
𝑊𝑖𝑚𝑔 = 𝐹𝑖𝑚𝑔 (𝑊𝑖𝑚𝑔 , 𝑊𝑡𝑒𝑥𝑡
),

(1)

(𝐸𝑝𝑜𝑐ℎ 2𝑖 + 1)

(2)

where the superscript 𝑓𝑖𝑥 indicates that the weight is fixed.

2.2

Modality Knowledge Propagation Loss

The image encoders are trained on person images with labels, so
they can capture details related to person identity. And such ability
is exactly what prompts need most. Therefore, when optimizing
prompts, our purpose is to propagate the knowledge contained in
image encoders to prompts.
The class probability distribution of a sample contains dark knowledge about the inter-class relationships. For example, classes with
similar predicted probabilities for the same sample may be close to
each other. Under such a premise, we propose the modality knowledge propagation loss. The loss constrains the predicted class probability of text features to be similar to that of image features. In
this way, person-related knowledge is propagated from image encoders into text prompts. The loss function can be expressed as:

The proposed method

Some studies follow the two-stage training strategy in CLIP ReID
to generate text prompts, but such strategy can only generate prompts
which do not focus on person-related information. And the reason
is that the CLIP model is unfamiliar to the ReID task, so it does not
pay attention to details related to person identity. Therefore, how
to generate prompts that better focus on person-related information remains an urgent problem to be solved.
In this case, we propose the MPT network, which improves both
prompts and image encoders by alternately optimizing them. The
structure of MPT is shown in the Fig. 2. Specifically, the text encoder takes prompts as input and outputs corresponding text features, while image encoders take images as input and output corresponding image features. When 𝑒𝑝𝑜𝑐ℎ = 2𝑖, we fix image encoders and optimize prompts with modality knowledge propagation loss. And when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 + 1, we fix prompts and optimize
image encoders with modality alignment loss.

2.1

(𝐸𝑝𝑜𝑐ℎ 2𝑖)

Alternating Training Strategy

Generally, encoders trained on VI-ReID datasets can help prompts
focus on person information, while prompts which focus on person
information can also help encoders extract better features. Therefore, the target of our network is that prompts can increasingly

2119

𝐿𝑚𝑘𝑝 =

1
∑ 𝐾𝐿(𝐹 𝐶(𝐼𝑖𝑘 ), 𝐹 𝐶(𝑇𝑖 ))
|𝑃 (𝑦𝑖 )| 𝑘∈𝑃 (𝑦 )

(3)

𝑖

where 𝐼𝑖𝑘 is the 𝑘-th image feature with the 𝑖-th identity and 𝑇𝑖
is the text feature with the 𝑖-th identity (each identity only has
one text prompt, so there is one text feature for each identity). 𝐹 𝐶
denotes the shared classifier, and 𝐾𝐿(⋅) is the Kullback-Leibler divergence between two probabilities. 𝑃 (𝑦𝑖 ) is the set of indices of
all positives image features for 𝑇𝑖 in the mini-batch and | ⋅ | is its
cardinality.

2.3

Modality Alignment Loss

Due to the fact that the high-level semantics contained in the text
are often modality independent (such as body shape, hairstyle, etc.),
we can consider optimized prompts as a special intermediate modality. When optimizing encoders, we treat the text features generated by prompts as the “bridge” and use them to connect the two
modalities. In this way, the modality discrepancies can be effectively reduced.
The proposed modality alignment loss first calculates the similarity between text features and image features, then utilizes contrastive learning loss to align the image features of both modalities

Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders

ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA

Table 1: Comparisons with SOTA methods on SYSU-MM01 and RegDB datasets. We report Rank at 𝑟 accuracy (%) and mAP (%).
Dataset
Setting
Methods
Hi-CMD [3]
JSIA [14]
AlignGAN [13]
SPOT [1]
PMWGCN [12]
WF-CAMReViT [11]
DART [17]
MSSFCS [18]
MAUM [7]
CAL [16]
DEEN [23]
SEFL [4]
FMCNet [22]
PMT [8]
ACD [10]
CSDN [21]
MPT (ours)

SYSU-MM01
All-search
Indoor-search
r=1
mAP
r=1
mAP
34.9
35.9
38.1
36.9
43.8
52.9
42.4
40.7
45.9
54.3
65.34
62.25
69.42
74.63
66.82
64.88
72.64
76.19
68.05
65.17
72.43
77.58
68.72
66.29
72.52
78.17
70.59
67.49
75.98
80.24
71.68
68.79
76.97
81.94
74.66
71.73
79.69
83.68
74.7
71.8
80.3
83.3
75.18
78.40
70.12
81.82
66.34
62.51
68.15
74.09
67.53
64.98
71.66
76.52
74.44
71.17
78.98
82.75
75.2
71.8
82.0
85.0
78.64
83.12
74.22
85.03

Venue
CVPR-20
AAAI-20
ICCV-19
TIP-22
TIFS-24
PR-24
CVPR-22
TMM-24
CVPR-22
ICCV-23
CVPR-23
CVPR-23
CVPR-22
AAAI-23
TIFS-24
arxiv-24
-

RegDB
VIS to IR
r=1
mAP
70.93
66.04
48.1
48.9
57.9
53.6
80.35
72.46
90.61
84.53
91.66
85.96
83.60
75.67
85.34
76.39
87.87
85.09
94.51
88.67
91.1
85.1
91.07
85.23
89.12
84.43
84.83
76.55
84.71
83.28
89.0
84.7
95.14
89.62

IR to VIS
r=1
mAP
48.5
49.3
56.3
53.4
79.37
72.26
88.77
81.61
92.97
86.08
81.97
73.78
83.88
75.16
86.95
84.34
93.64
87.61
89.5
83.4
92.18
86.59
88.38
83.86
84.16
75.13
87.18
84.72
88.2
82.8
94.33
89.01

Table 2: Ablation studies of each component in MPT network.
All-search
Rank-1
mAP
73.19
68.34
75.4
71.03
77.09
73.26
78.64
74.22

Methods
Baseline
two-stage [21]
MPT w/ 𝐿𝑀𝑃 𝑇
MPT

Indoor-search
Rank-1
mAP
78.58
80.12
80.14
82.37
82.15
84.32
83.12
85.03

Baseline

to the text features. Specifically, before calculating similarity, we
use linear layers to project them into a common space. And the
similarity between text and image features can be expressed as:

Fig. 3: Illustration of visualization results using t-SNE. Features
in visible (infrared) modality are represented by circles (triangles),
while the various colors are used to distinguish different identities.

(4)

𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑖 ) = (𝑊𝐼 ⋅ 𝐼𝑖 ) ⋅ (𝑊𝑇 ⋅ 𝑇𝑖 ),

where 𝑊𝐼 and 𝑊𝑇 are linear layers. Then the modality alignment
loss can be expressed as:
𝑁

𝐿𝑚𝑎 = ∑ −𝑞𝑘 𝑙𝑜𝑔
𝑘=1

𝑒𝑥𝑝(𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑘 ))
𝑁

∑𝑎 𝑒𝑥𝑝(𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑎 ))

,

(5)

where 𝑁 is both the number of identities in the dataset and the
number of text prompts. 𝑞𝑘 is the value in the target distribution,
and it can be expressed as:
1 − 𝜖,
𝑞𝑘 = { 𝜖
𝑁,

𝑦𝑘 = 𝑦 𝑖
,
𝑦𝑘 ≠ 𝑦 𝑖

Initialization

(6)

where 𝑦𝑘 is the identity of the 𝑘-th image, and 𝜖 is a hyper-parameter
used for the smoothing operation and is set to 0.1 as [6].
Therefore, the mutual promotion loss is expressed as:
𝐿𝑀𝑃 𝑇 = 𝐿𝑚𝑘𝑝 + 𝐿𝑚𝑎

(7)

With the help of the above loss functions, text prompts and image encoders can mutually promote each other, ultimately improving the performance of cross-modality retrieval.

2.4

Baseline

Objective Functions

MPT

Fig. 4: The distribution of three models (i.e. Initialization, Baseline,
and MPT network) between cross-modality features. Blue and green
color are used to represent the intra-class distances and inter-class
distances, respectively. Red lines represent the mean of inter-class
distances and the mean of inter-class distances, and 𝛿𝑖 is the difference between the two type of means.

𝐿𝑡𝑟𝑖 = [𝑚𝑎𝑥(𝑑𝑝 − 𝑑𝑛 + 𝜌)]+ ,

(9)

where 𝑝𝑘 is the prediction logits of 𝑘-th class, and 𝑑𝑝 (𝑑𝑛 ) denotes
feature distance of positive (negative) pair. 𝜌 is the margin which
is set to 0.3 following [6].
Therefore, the total loss can be formulated as follow:

To ensure the performance of the model, we employ the commonly
used triplet loss and identity loss, which can be calculated as:
𝑁

𝐿𝑖𝑑 = ∑ −𝑞𝑘 𝑙𝑜𝑔(𝑝𝑘 ),

MPT

(8)

𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑖𝑑 + 𝐿𝑡𝑟𝑖 + 𝐿𝑀𝑃 𝑇

𝑘=1

2120

(10)


exec
/bin/zsh -lc "pdftotext -l 3 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

4857

Visible-Infrared Person Re-Identification With
Real-World Label Noise
Ruiheng Zhang , Member, IEEE, Zhe Cao , Graduate Student Member, IEEE, Yan Huang , Shuo Yang,
Lixin Xu , and Min Xu , Member, IEEE

Abstract— In recent years, growing needs for advanced security
and traffic management have significantly heightened the prominence of the visible-infrared person re-identification community
(VI-ReID), garnering considerable attention. A critical challenge
in VI-ReID is the performance degradation attributable to label
noise, an issue that becomes even more pronounced in crossmodal scenarios due to an increased likelihood of data confusion.
While previous methods have achieved notable successes, they
often overlook the complexities of instance-dependent and realworld noise, creating a disconnect from the practical applications
of person re-identification. To bridge this gap, our research
analyzes the primary sources of label noise in real-world settings,
which include a) instantiated identities, b) blurry infrared images,
and c) annotators’ errors. In response to these challenges,
we develop a Robust Hybrid Loss function (RHL) that enables
targeted recognition and retrieval optimization through a more
fine-grained division of the noisy dataset. The proposed method
categorises data into three sets: clean, obviously noisy, and
indistinguishably noisy, with bespoke loss calculations for each
category. The identification loss is structured to address the
varied nature of these sets specifically. For the retrieval subtask, we utilize an enhanced triplet loss, adept at handling
noisy correspondences. Furthermore, to empirically validate our
method, we have re-annotated a real-world dataset, SYSU-Real.
Our experiments on SYSU-MM01 and RegDB, conducted under
various noise ratios of random and instance-dependent label
noise, demonstrate the generalized robustness and effectiveness
of our proposed approach.
Index Terms— Person re-identification, visible infrared, crossmodality, instance-dependent label noise, robust deep learning.

Received
24
October
2024;
revised
5
December
2024;
accepted 30 December 2024. Date of publication 6 January 2025; date
of current version 7 May 2025. This work was supported in part by the
Scientific and Technological Innovation (STI) 2030–Major Projects under
Grant 2022ZD0209600; in part by the National Natural Science Foundation
of China under Grant 62201058, Grant 62475016, and Grant 62306311;
and in part by the Science and Technology on Electromechanical Dynamic
Control Laboratory Funding under Grant 6142601012402. This article was
recommended by Associate Editor J. Gui. (Corresponding authors: Lixin Xu;
Yan Huang.)
Ruiheng Zhang, Zhe Cao, and Lixin Xu are with the State Key
Laboratory of Electromechanical Dynamic Control, School of Mechatronical
Engineering, Beijing Institute of Technology, Beijing 100081, China (e-mail:
ruiheng.zhang@bit.edu.cn; zhe.cao@bit.edu.cn; lxxu@bit.edu.cn).
Yan Huang is with the New Laboratory of Pattern Recognition, State Key
Laboratory of Multimodal Artificial Intelligence Systems, Institute of
Automation, Chinese Academy of Sciences, Beijing 100190, China (e-mail:
huangyan.750@outlook.com).
Shuo Yang is with the School of Computer Science and Technology, Harbin
Institute of Technology (Shenzhen), Shenzhen 518055, China.
Min Xu is with the School of Electrical and Data Engineering, Faculty of
Engineering and Information Technology, University of Technology Sydney,
Ultimo, NSW 2007, Australia.
Digital Object Identifier 10.1109/TCSVT.2025.3526449

I. I NTRODUCTION

P

EDESTRIAN re-identification utilizes extracted image
features to facilitate identity localization across different
cameras, playing a crucial role in public safety and traffic
detection [1], [2], [3], [4], [5], [6]. The scope of person
re-identification has recently expanded to include visibleinfrared cross-modal scenarios [7], [8], [9], [10], [11],
[12]. Incorporating infrared images enables effective reidentification in poorly lit environments, enhancing its
practicality in real-world scenarios. However, visible-infrared
person identification (VI-ReID) encounters notable challenges
due to the modality gap [13] and the quality of infrared images
[14]. To overcome these challenges, previous methods have
focused on uncovering shared features across modalities [15],
[16], [17], [18], employing data augmentations [19], [20],
[21], and leveraging auxiliary information [22], [23], [24].
Despite these advancements, most VI-ReID models, have not
adequately addressed the adverse effects of noisy labels on
their performance.
In the real world, label noise is a common issue [25], [26],
[27], [28], and this is particularly true in the VI-ReID task.
There are three primary causes of noisy labels in this context:
• Instantiated Identity within Single Modality: The
classification of identities often involves transferring
categories from semantically distant entities, leading to
confusion between identities. The main reason for this
issue is the small gap between the characteristics of the
same entity in different identities.
• Instantiated Identity between Two Modalities: Infrared
imaging often loses key features such as color and texture,
making it challenging to distinguish between different
identities. Consequently, it is hard to match infrared
images with the same identity and to correlate infrared
images with their visible counterparts.
• Annotator Errors: While primarily stemming from
the first two reasons, errors due to the sloppiness or
carelessness of annotators can also contribute to noisy
labels. These errors, which can sometimes be random,
are often exacerbated by difficult annotation tasks.
Currently, only one method [19] has discussed the issue
of noisy labels in VI-ReID. Although the existing study
DART [19] has considered the impact of noisy labels on
VI-ReID, it primarily focuses on random noise, such as
arbitrarily switching image labels to incorrect ones. This noise
generation approach does not fully represent the complex

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.

4858

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Fig. 1. The processing of visible-infrared re-identification dataset annotation. From left to right, these are random assignment, instance-dependent algorithmic
generation, and real-world annotation. Circles represent visible images, and triangles represent infrared images. The outline of each instance is its potential
real label (identity), and the center is filled with annotation labels, with different colors indicating different identities.

forms of noise encountered in real-world scenarios, as outlined
in the previously discussed points. To better simulate realworld conditions, we propose using instance-dependent label
noise (IDN), which varies with each instance and reflects
specific characteristics, offering a more realistic representation
of label noise than the random noise used in the existing
method [19]. As depicted in the middle part of Fig. 1, our
approach follows existing studies [29], [30], employing the
IDN generation algorithm to create synthetic datasets with
noisy labels. However, it is important to acknowledge that even
this advanced method has limitations. That is, it cannot fully
capture the unpredictable and complex nature of real-world
label noise in practical applications of VI-ReID.
Considering the aforementioned analysis, we identify two
primary reasons for the discrepancy between IDN and realworld noise: Firstly, as an algorithmically synthesized noise,
IDN is unable to accurately mimic annotator errors, as shown
in the real-world scenarios depicted in Fig. 1. Specifically,
it does not account for instances of annotator carelessness or
inattention. Secondly, the robustness of VI-ReID algorithms
still requires real-world labelling for corroboration and cannot
rely on simulated security challenges alone. Indeed, relying
on synthetic algorithms for noise simulation might lead
to a distorted evaluation of the methods’ effectiveness.
To establish a more robust foundation for validating VIReID methods against real-world label noise, we have
re-labeled the SYSU-MM01 dataset, creating an updated
version called SYSU-Real that includes genuine real-world
annotations.
To tackle the challenges of increasingly complex label
noise above, we introduce a robust hybrid loss (RHL)
framework that categorizes data into three segments: clean,
indistinguishably noisy, and obviously noisy. For the clean
data, we apply the standard cross-entropy (CE) loss, given
that its labels are reliable. The second segment contains
samples with ambiguous label quality, posing a significant

challenge to label distinction. We note a bias in the CE loss
function, particularly its tendency to assign disproportionately
negative values to samples with noisy labels, leading to skewed
optimization. To counteract this, we utilize the Generalized
Cross-Entropy (GCE) loss function [31], which aims to
reduce this bias and ensure fair treatment of these samples.
In the third segment, where noise is evident, we discard the
noisy labels altogether. Instead, we leverage the preheating
model’s generalization capability, using the confidence level
as the basis for loss calculation. Additionally, recognizing the
inadequacy of general retrieval loss, we adopt the pair division
method and triplet loss from DART to enhance the retrieval
process’s robustness.
In summary, our contributions are shown as follows:
• We present, to our knowledge, the first consideration
of real-world label noise in VI-ReID. To overcome
the challenges of obtaining real-world labeled datasets,
we have implemented two strategies. Firstly, we synthesized datasets with an instance-dependent label
noise generation algorithm. Secondly, we manually
re-labeled the SYSU-MM01 dataset, creating the
human-annotated SYSU-Real dataset. This approach
significantly enhances the realism of label noise, yet it
also adds to the complexity and challenge of VI-ReID
studies.
• We introduce a Robust Hybrid Loss (RHL) function
that effectively manages instance-dependent label noise
and real-world noise. This function utilizes a threecomponent mixture, allowing distinct loss calculation
strategies for different noise subsets, thereby enhancing
the method’s robustness and adaptability.
• We conducted comparative experiments with six stateof-the-art methods on the SYSU-MM01 and RegDB
datasets, incorporating both random and instancedependent label noise. Additionally, we evaluated
real-world noise handling on the relabeled SYSU-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE

Real dataset. Our results demonstrate the superior
effectiveness and robustness of our proposed method.
II. R ELATED W ORK
In this section, we will provide a concise overview of the
research advancements in cross-modal person re-identification
and noise label learning that are pertinent to this paper.
A. Visible-Infrared Person Re-Identification (VI-ReID)
VI-ReID, an evolution from traditional single-modality
person re-identification, aims to match individuals across different modality images. This field has experienced significant
advancements recently, with a plethora of innovative methods
being proposed [1], [3], [5], [32], [33], [34], [35].
The primary focus of current mainstream VI-ReID methods
is to identify co-modality features using strategies such as
feature projection, disentanglement, and loss function design.
These shared features encapsulate essential identification
information, including the subject’s outline, semantics, and
morphology. For feature projection, prior methods have
explored extracting multi-level features [36], [37], utilizing
graph neural networks [17], [38], [39], and examining both
global and local features [15], [40], [41]. All these techniques
aim to design networks that extract more effective features for
identification.
Feature disentanglement, another critical aspect, focuses
on differentiating between identity-relevant and identityindependent features. Identity-related features constitute
entangled information across both modalities, while the rest
are disentangled. For instance, Pu et al. [42] employed
a Gaussian mixture model to isolate identity-discriminable
information. Hu et al. [43] achieved disentanglement through
a min-max adversarial process. Additionally, Tian et al. [44]
analyzed the decoupling issue from an information-theoretic
perspective, employing variational reasoning to separate taskirrelevant information.
The design of the loss function is the most prevalent and
effective approach, encompassing five main types: contrastive,
triplet, center, adversarial, and identity loss. Contrast loss,
based on metric learning, works to mitigate cross-modal
differences [45]. Triplet loss and its variations, popularly used,
enable the comparison of sample-to-sample relationships,
thereby facilitating retrieval capabilities [46], [47], [48].
Center loss emphasizes the connections between identities
internally, using the feature’s center as an anchor point for
computing distance-based loss [49], [50]. Adversarial loss,
employed for cross-modal feature extraction, involves the
adversarial interaction of generators and discriminators [51],
[52], and is known for mining deep semantic features,
albeit with challenges in training stabilization [53]. Identity
loss, fundamental to identification, typically utilizes crossentropy loss for classification, with some studies using it
to constrain feature extraction for capturing more identityrelevant information [54]. This paper also considers this
perspective, emphasizing that the essence of retrieval lies
predominantly in identity recognition.

4859

B. Learning With Noisy Label (LNL)
Label noise has emerged as a significant challenge in the
field of machine learning, particularly with the continuous
expansion of dataset sizes. This growth has led to an increased
reliance on various annotation methods, such as manual
annotation, crowdsourcing, and automatic annotation, each of
which can inadvertently introduce label noise.
To combat this issue, researchers have developed several
strategies. These include sample selection methods [55], [56],
[57], [58], [59], [60] that focus on identifying and utilizing
reliable samples, and label correction techniques [30], [61],
[62], [63] that aim to rectify erroneous labels. Additionally,
some approaches incorporate normalization [64], [65], [66],
[67], [68] to mitigate the impact of label noise, and others
design robust loss functions [31], [69], [70], [71], [72] to
enhance model resilience against noisy data. While these
methods have shown promising results in suppressing label
noise, they have predominantly been applied to classification
tasks and single-modality data (visible images).
The study of label noise in multi-modality domains,
however, is a more recent development and presents
unique challenges [19], [73], [74], [75], [76], [77]. For
instance, Huang et al. [73] introduced the concept of noise
correspondence and developed the Noisy Correspondence
Rectifier (NCR) for image-text matching. Qin et al. [74]
proposed a Cross-modal Evidential Learning (CEL) approach
and a Robust Dynamic Hinge loss (RDH) for matching
tasks. Furthermore, Yang et al. [75] analyzed the impact
of noise correspondence between modalities, leading to
the development of the Bidirectional Cross-modal similarity
consistency (Bicro) framework, which significantly enhances
noise-robustness.
However, none of these methods specifically address crossmodal person re-identification. In this paper, we extend the
discussion of label noise to encompass both infrared and
visible images in the context of VI-ReID. Our approach
integrates elements of sample selection, a robust framework,
and a robust loss function, tailored to the unique challenges
of this domain.

C. Label Noise in Person Re-ID
Label noise significantly impacts person re-identification
in real-life scenarios, often leading to misidentification
and potential conflicts. Recent research in this area has
explored various strategies to mitigate these issues, including
optimizing labels [78], modifying loss functions [79], and
designing new paradigms [80]. In the cross-modal domain,
emerging studies have focused on label noise in text-to-image
person re-identification [81] and visible-infrared person reidentification [19]. However, these methods predominantly
focus on symmetric or random noise, overlooking the
complexity of real-world noise. Recognizing this gap, our
paper concentrates on real-world noisy labels. We undertake
the task of relabeling a cross-modality person re-identification
dataset to better reflect the challenges posed by real-world
conditions.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.


codex
前四篇的前三页已经抽出来了，内容集中在参数高效迁移、可见红外解耦、提示词与图像编码器互促、真实标签噪声。还剩 LVLM-ReID 这一篇，我继续按同样口径抽前三页。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'When Large Vision-Language Models Meet Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
WHEN LARGE VISION-LANGUAGE MODELS MEET PERSON RE-IDENTIFICATION
Qizao Wang

Bin Li

Xiangyang Xue†

School of Computer Science, Fudan University, Shanghai, China
qzwang22@m.fudan.edu.cn, libin@fudan.edu.cn, xyxue@fudan.edu.cn

arXiv:2411.18111v2 [cs.CV] 11 May 2026

ABSTRACT
Large Vision-Language Models (LVLMs) that incorporate visual
models and large language models have achieved impressive results
across cross-modal understanding and reasoning tasks. In recent
years, person re-identification (ReID) has also started to explore
cross-modal semantics to improve the accuracy of identity recognition. However, effectively utilizing LVLMs for ReID remains an
open challenge. While LVLMs operate under a generative paradigm
by predicting the next output word, ReID requires the extraction
of discriminative identity features to match pedestrians across cameras. In this paper, we propose LVLM-ReID, a novel framework
that harnesses the strengths of LVLMs to promote ReID. Specifically, we employ instructions to guide the LVLM in generating
one semantic token that encapsulates key appearance semantics
from the person image. This token is further refined through our
Semantic-Guided Interaction (SGI) module, establishing a reciprocal interaction between the semantic token and visual tokens.
Ultimately, the reinforced semantic token serves as the representation of pedestrian identity. Our framework integrates the semantic
understanding and generation capabilities of LVLM into end-to-end
ReID training, allowing LVLM to capture rich semantic cues during both training and inference. LVLM-ReID achieves competitive
results on multiple benchmarks without additional image-text annotations, demonstrating the potential of LVLM-generated semantics
to advance person ReID.
Index Terms— Person Re-Identification, Large Vision-Language
Model, Semantic Token
1. INTRODUCTION
Person re-identification (ReID) is a crucial task in computer vision,
aimed at accurately matching pedestrians across different camera
views [1]. With the continuous advancements in deep learning techniques, person ReID methods have evolved significantly [2, 3]. In
the past decade, many research has significantly improved ReID
accuracy by optimizing the distances between features [4] and designing refined modules [5, 6, 7], following the paradigm shown in
Fig. 1 (a). Due to the difficulty of learning rich pedestrian semantics
from a single modality, cross-modal learning has received close attention recently. For example, in the context of the development of
pre-trained Vision-Language Models (VLMs), CLIP-ReID [8] based
on the representative VLM model CLIP [9] to leverage text semantics. As shown in Fig. 1 (b), it enhances visual features through
cross-modal contrastive learning with image-text pairs. Meanwhile,
Large Language Models (LLMs) [10, 11] have attracted widespread
This work was supported by the National Natural Science Foundation of
China under Grant No. 62576110.
† Corresponding author.

ReID Loss

ReID Loss

Visual
Encoder

Visual
Encoder

Contrastive Loss

Text
Encoder

A photo of [X]1 [X]2
… [X]M person.

(a) Conventional ReID

(b) CLIP-ReID

Generated
Semantic Token
Summarize the
person image into
one word, …

ReID
Loss

Instruction

LVLM
Visual Tokens

(c) LVLM-ReID

Fig. 1. Comparison of different person ReID frameworks.
attention due to their powerful capabilities in text generation and
comprehension. Large Vision-Language Models (LVLMs) [12, 13,
14] enhance LLMs by incorporating visual perception and understanding, demonstrating considerable potential in multi-modal learning tasks. However, integrating LVLMs with person ReID remains
an underexplored challenge.
LVLMs typically operate on a generative paradigm, training and
functioning by predicting the next word in a sequence. Thanks to
pre-training and instruction tuning, LVLMs can follow instructions
and converse with humans. As a result, a direct approach might be
to have the model to identify the input person images. However,
ReID gallery databases are usually very large (comprising tens of
thousands of pedestrian images) [15, 16]. The time and cost of comparing identities one by one with LVLMs are substantial. Processing
multiple images simultaneously would also lead to an unacceptable
increase in visual tokens. Therefore, we are motivated to leverage
the reasoning and understanding capabilities of LVLMs, while adhering to the mainstream ReID paradigm of feature extraction combined with feature similarity-based retrieval [1]. A potential solution involves using LVLMs to describe pedestrian images and finetuning the visual encoder via tasks such as image-text matching or
image caption prediction. However, it presents several limitations:
(1) High-quality and diverse text annotations are expensive to obtain.
(2) The goals of image-text matching or image caption prediction
tasks may not align well with those of image-based ReID. (3) During
the inference phase, the potential of LVLMs is often underutilized,
as they are not effectively integrated with the visual features.

<REID>

FC

Camera Semantic Supplementation
4x

LLM

Visual Encoder
V

Pedestrian Semantic Token Generation

Feed-Forward Network

K

Multi-Head Self-Attention

Summarize the person image
into one word, focusing on
age, gender, clothing, and
biometric features.

Q

Semantic-Guided Interaction

Fig. 2. Framework of our LVLM-ReID. It leverages instructions to guide the frozen LLM toward focusing on particular visual semantics
within pedestrian images, resulting in the generation of one semantic token that encapsulates the pedestrian’s appearance information. Subsequently, an efficient interaction module is designed to facilitate refinement between the generated token and the visual tokens. Finally, the
reinforced semantic token is optimized and employed for person retrieval.
To address these issues, we propose a new ReID framework
called LVLM-ReID to leverage the superior semantic understanding
and generation ability of LVLMs. Specifically, as shown in Fig. 1 (c),
we use instruction to guide the LVLM to focus on specific visual
semantics in pedestrian images, generating a semantic token representing the pedestrian’s appearance information. We then design
an effective interaction module between the generated token and visual tokens, refining the visual representations of pedestrians while
reinforcing the semantic token as a discriminative identity representation. Ultimately, the reinforced semantic token is optimized and
used during inference to achieve person retrieval. Our framework
integrates the generative process of LVLMs into the ReID model,
eliminating the need for additional image caption annotations and
enabling end-to-end effective learning. More importantly, during
the inference phase, we continue to leverage the generative power
of LVLMs to enhance visual features adaptively. Our experiments
show that one LVLM-generated semantic token can effectively facilitate the learning of pedestrian representations. Our contributions
are summarized as follows: (1) We propose a novel framework
that incorporates LVLMs into the person ReID task, offering a new
perspective on using generative language models to assist discriminative visual models. (2) We propose to utilize the generative capability of LVLMs to produce a semantic token for pedestrians and design a semantic-guided interaction module leveraging the generated
semantic token to enhance identity representations. (3) Experimental results show that, without requiring additional annotations, our
method effectively improves the discriminability of identity features
and achieves competitive results across multiple datasets.
2. METHODOLOGY
2.1. Overview of LVLM
Overall framework. A typical LVLM consists of three key components: a visual encoder, a vision-language connector, and an LLM.
The visual encoder extracts rich visual representations from images,
which are then processed by the vision-language connector that converts visual features into the word embedding space. The LLM,
trained for next-word prediction, generates text based on the encoded
visual content. In this work, we leverage Qwen2-VL [14], one of

the most advanced LVLMs, known for its superior capabilities in
instruction-following, semantic understanding, and text generation
across diverse tasks. Qwen2-VL combines a Vision Transformer
(ViT) [17] as the visual encoder and the Qwen2 [11] as the LLM.
The vision-language connector between the two components is one
MLP layer that also compresses the extracted visual tokens.
Visual token extraction. Before inputting a pedestrian image into
the LLM, the image is first encoded and compressed by the visual
encoder. Specifically, each input RGB image x ∈ RH×W ×3 , where
H and W are its height and width, is first divided into patches of size
P ×P . These patches are then embedded and flattened into a feature
vector xp ∈ RN ×d , where N = H × W/P 2 represents the number of patches, and d is the embedding dimension. The resulting
patch embeddings are processed through multiple layers of Transformer self-attention blocks [18], producing visual representations
f ∈ RN ×d . To enhance the model’s ability to capture spatial dependencies, Multimodal Rotary Position Embedding (M-RoPE) [14] is
used in the process. Afterward, a simple MLP layer compresses adjacent 2 × 2 tokens into a single token, producing the final visual
tokens V , which is formulated as:
V = MLP(f ) = [v1 ; v2 ; · · · ; vn ] ∈ Rn×d ,

(1)

N
. Notably, instead of using the traditional [class]
4

where n =
token [17], the image is transformed into a set of visual tokens.
These visual tokens will then be passed to the LLM for further processing and interaction.
2.2. Pedestrian Semantic Token Generation
We aim to integrate the advanced visual semantic understanding and
generation capabilities of LVLM into the feature extraction pipeline,
by guiding the ReID model to generate one semantic token that encapsulates instructive information of the pedestrian. To achieve this,
we use instructions to direct the LVLM to summarize the pedestrian’s visual appearance:
<|vision start|> V <|vision end|> Summarize
the person image into one word, focusing on age, gender,
clothing, and biometric features.

where V represents the extracted visual tokens, while the special
tokens <|vision start|> and <|vision end|> are used to
mark the beginning and end of the visual token sequence. With this
instruction, the LVLM is guided to focus on the appearance-related
semantics in the image, and then generate a semantic token that summarizes the relevant identity features. We denote this generated token as <REID>, which serves as a compact representation of the
pedestrian’s visual appearance.
Camera semantic supplementation. The semantic token generation process overlooks the influence of camera variations. To improve pedestrian semantic consistency across cameras, we explicitly model and account for these camera-induced feature variations.
Specifically, we assign a unique learnable embedding vector to each
camera, which allows the model to learn the inherent feature shifts
caused by cameras. We denote the set of learnable camera embedi
dings as Vcam = {vcam
|i = 1, 2, · · · , N c }, where N c is the total
number of cameras. One direct implementation is to supplement the
generated pedestrian semantic token with the camera semantics as
follows:
yc
v̄reid = vreid + vcam
,
(2)
where vreid is the encoding of the <REID> token, y c is the camera
ID corresponding to the image x. However, this late supplementation
strategy may affect the visual model weakly. We thus try to transfer
the usage of camera embeddings to the input of the visual model,
where the camera embeddings are added to the patch embeddings
xp . We evaluate and discuss the two variants in Sec. 3.3.
2.3. Semantic-Guided Interaction
We design the Semantic-Guided Interaction (SGI) module to facilitate bidirectional interaction between the generated semantic token
and the visual tokens. Specifically, the generated semantic token is
first concatenated with the visual tokens. Formally,
z = [vreid ; v1 ; v2 ; · · · ; vn ] ∈ R(n+1)×d .

(3)

This concatenated token sequence is then passed through 4
layers of Transformer blocks, each consisting of a multi-head selfattention layer and a feed-forward network. The module refines
the visual features to capture identity-relevant information under
the guidance of the semantic token. Meanwhile, the semantic token, serving as the pivot for information aggregation, distills more
discriminative features from the visual representations, enhancing
the overall understanding of the pedestrian’s identity. Through
the semantic-guided interaction module, the model produces the
reinforced representation as:
ẑ = [v̂reid ; v̂1 ; v̂2 ; · · · ; v̂n ] = SGI(z).

(4)

Then, the reinforced semantic token representation v̂reid is used
to compute the Re-ID losses, i.e., identity classification loss [2] and
triplet loss [4].
2.4. Optimization and Inference
During training, we optimize the parameters of both the visual model
and the SGI module. The parameters of LLM are frozen, but gradients backpropagate through it to update other learnable components.
By leveraging the generated <REID> token in conjunction with the
SGI module, we achieve a joint end-to-end training that harnesses
the strengths of LVLM in instruction-following and visual semantic understanding. This process allows for the integration of rich

Table 1. Comparison with the state-of-the-art methods. The results of our method and the best results of comparison methods are
shown in bold.
DukeMTMC-reID

Market-1501

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

CNN

MGN [19]
DG-Net [20]
SAN [5]
Pyramid [21]
Relation-Net [22]
RGA-SC [23]
CDNet [6]
CAL [7]

78.4
74.8
75.5
79.0
78.6
76.8
76.4

88.7
86.6
87.9
89.0
89.7
88.6
87.2

86.9
86.0
88.0
88.2
88.9
88.4
86.0
87.0

95.7
94.8
96.1
95.7
95.2
96.1
95.1
94.5

67.4
76.4
76.9
75.6
77.4
-

68.0
80.1
78.9
77.9
81.1
-

-

-

ViT

TransReID [24]
PAT [25]
DCAL [26]
AAformer [27]
CLIP-ReID [8]

80.6
78.2
80.1
80.0
82.5

89.6
88.8
89.0
90.1
90.0

88.2
88.0
87.5
88.0
89.6

95.0
95.4
94.7
95.4
95.5

79.0
80.3

80.3
81.6

55.7
53.6
58.2
59.5

64.2
64.5
67.1
67.1

LVLM

LVLM-ReID (Ours)

82.8

92.2

89.2

95.6

82.3

84.6

59.8

68.1

Backbone

Methods

CUHK03

Occluded-Duke

semantic cues into the visual representations, improving the accuracy of pedestrian identity recognition. The overall training loss is a
weighted combination of the identity classification loss Lid [2] and
the triplet loss Ltri [4], which is expressed as follows:
L = α1 Lid + α2 Ltri ,

(5)

where α1 and α2 are balancing factors that control the contribution
of each loss term.
During inference, the LVLM is also used to generate the
<REID> token for each input image. Then, the reinforced semantic token representation, v̂reid , is used to compute the cosine
similarity between different person images. These similarity scores
are employed for identity matching, allowing the model to identify
pedestrians. Note that the identity representations of persons in the
large gallery databases need be extracted only once in applications.
3. EXPERIMENTS
3.1. Experimental Settings
Datasets. We evaluate our methods on four person ReID datasets:
DukeMTMC-reID [16], Market-1501 [15], CUHK03 [28], and
Occluded-Duke [29].
Evaluation metrics. We follow the common practices to adopt Cumulative Matching Characteristics (CMC) at Rank-1 and mean Average Precision (mAP) for performance evaluation.
Implementation details. We employ Qwen2-VL-2B [14] considering its efficiency with limited resources, while larger model sizes
such as 7B and 72B have better LLM capabilities. The model adopts
BFloat16 mixed precision. H, W , and P are set to 280, 140, and
14, respectively, resulting in n = 50. In other words, 50 visual
tokens are included in the input of LLM and our SGI module. Following [2], random horizontal flipping, padding, random cropping,
and random erasing [30] are used for data augmentation. 16 identities and 4 images per person are randomly sampled to constitute a
training batch. Adam optimizer with weight decay of 3 × 10−4 is
adopted, with the warmup strategy that linearly increases the learning rate from 3 × 10−5 to 3 × 10−4 in the first 10 epochs. We train
the model for 60 epochs, with a learning rate decay factor of 0.1 at
the 30th epoch. α1 and α2 are set to 0.25 and 1 following [8]. The
margin m of triplet loss is set to 0.3. Our method is implemented
using PyTorch and on one NVIDIA A800 GPU.
3.2. Comparison with State-of-the-Art Methods
As shown in Tab. 1, methods based on CNNs achieve solid performance by designing elaborate modules for person ReID, while


codex
我只读了每篇前三页，重点看摘要、引言和方法开头。

**逐篇提取**

1. `UP-Person`
- 创新类型：工程组合偏新机制。统一 Prefix、LoRA、Adapter，并改 S-Prefix 和 L-Adapter 解决组件冲突。
- 为什么能发：TPR 里 CLIP 全量微调容易过拟合，已有 PETL 只做局部尝试。它把“保留 CLIP 知识”和“少量参数适配”讲成主问题，并用三数据集加参数量下降支撑。
- story 套路：全量微调会破坏大模型原有知识，参数高效适配能同时保留预训练知识和学习 ReID 任务知识。

2. `Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement`
- 创新类型：问题分解加工程组合。把 VI-ReID 的干扰拆成相机、模态、身份无关冗余三层。
- 为什么能发：多数 VI-ReID 只讲模态差异，忽略相机视角和背景差异。它用层级解耦、对抗损失、跨模态残差信息重组形成完整证据链。
- story 套路：跨模态差异不是单一差异，必须逐层净化，先去相机扰动，再提模态不变身份特征。

3. `Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders`
- 创新类型：新机制。提示词和图像编码器交替优化，互相强化。
- 为什么能发：原始 CLIP 或大模型生成的描述不关注身份细节，两阶段训练又是单向利用文本。它把提示词从静态辅助变成可被 ReID 编码器反向教育的身份桥梁。
- story 套路：好 prompt 不是先验给定的，而是在 ReID 训练中从图像编码器学出来，再反过来对齐可见光和红外特征。

4. `Visible-Infrared Person Re-Identification With Real-World Label Noise`
- 创新类型：问题重定义加新数据设定。把 VI-ReID 从干净标签扩展到真实标签噪声。
- 为什么能发：已有噪声工作多用随机噪声，和真实跨模态标注错误不一致。它分析真实噪声来源，重标 SYSU-Real，并设计 clean、obvious noisy、indistinguishable noisy 三类样本的混合损失。
- story 套路：真实系统里最先坏掉的不是模型结构，而是标签可信度。先把真实噪声定义清楚，再用针对性损失处理不同噪声层级。

5. `When Large Vision-Language Models Meet Person Re-Identification`
- 创新类型：新机制。把 LVLM 的生成能力变成一个用于检索的身份语义 token。
- 为什么能发：LVLM 逐词生成和 ReID 特征检索范式冲突，直接描述图片太慢，也不一定服务身份匹配。它用指令生成一个 `<REID>` token，再和视觉 token 交互，最后用该 token 做身份表征。
- story 套路：不要让 LVLM 逐张对话式比人，而是把它压缩成可检索的身份语义 token，接入标准 ReID 训练和推理流程。

**综合反推**

这批的共同启发是：能发的点不是“再加一个分支”，而是把一个旧问题改成更具体、更真实、更可验证的问题。最有价值的套路有三个：把干扰源重新分层，把静态语义提示改成训练闭环，把真实噪声或真实约束引入设定。

**候选 1：3D 可见表面一致性 ReID**

一句话 headline：航拍到地面 ReID 不应匹配整张图，而应匹配同一个人在不同视角下仍可见的 3D 人体表面证据。

挂团队资产：SMPL 3D 几何、aerial-ground、SOLIDER-Swin。

和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。

cheap kill-switch：先不训练。用现有 SMPL 或 2D pose 估计，把图像切成可见身体区域，测试 CARGO 或 AG-ReID.v2 上“几何可比区域相似度”是否比全局 SOLIDER 相似度更能区分正负样本。若同身份跨视角失败样本主要集中在低可见表面重叠区域，并且简单几何重加权能带来超过噪声线的提升，再立项。

**候选 2：跨视角身份证据可靠性门控**

一句话 headline：极端航拍低清场景里，关键不是提更多特征，而是让模型知道哪些身体区域在当前视角下不可信。

挂团队资产：pose 热图门控、aerial-ground、SOLIDER-Swin。

和这批最像工作的区别：它最像真实标签噪声那篇的“分清干净、模糊、错误样本”，但我们把噪声从标签层下沉到 token 层。每个身体区域都有可靠性，低清、遮挡、俯视压缩造成的区域不应该同等参与身份学习。

cheap kill-switch：冻结 SOLIDER，用 pose heatmap 置信度或 SMPL 可见性做零训练 token reweighting。看航拍 query 对地面 gallery 的错误检索是否被低置信区域主导。若只用高置信身体区域能稳定改善，说明“区域可靠性”是真问题。

**候选 3：航拍地面 ReID 的真实身份歧义学习**

一句话 headline：aerial-ground ReID 的核心噪声不是随机错标，而是低清航拍下多个身份在视觉上不可区分造成的真实歧义。

挂团队资产：aerial-ground、SMPL、SOLIDER-Swin。

和这批最像工作的区别：它最像 real-world label noise，但那篇处理 VI-ReID 的类别标签噪声。我们的切开点是跨视角低清导致的“身份歧义集合”，一个航拍人可能在视觉证据上对应多个地面候选，训练不应强行把所有非真值都当负样本。

cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。

**候选 4：训练期 3D 语义 token，测试期纯 ReID**

一句话 headline：用 SMPL 生成训练期身份几何 token，把 LVLM 式“语义 token”改成不依赖语言、不增加测试成本的 3D 人体 token。

挂团队资产：SMPL 3D 几何、SOLIDER-Swin、pose 热图门控。

和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。

cheap kill-switch：先做训练很轻的辅助头。冻结或半冻结 SOLIDER，在现有数据上预测 SMPL joint visibility、身体表面区域或姿态几何 token，看辅助监督是否能改善跨视角检索。若只学几何 token 不改善，说明几何语义没有进入身份判别，及时停。

我最建议优先试候选 1。它同时占“问题新”和“机制新”，和这批论文的切开点清楚，也最能发挥你们已经打通的 SMPL 基建。候选 3 是问题层面最像 B 类故事的备选，但需要一点真实歧义证据支撑，否则容易变成普通鲁棒损失。
tokens used
55,992
我只读了每篇前三页，重点看摘要、引言和方法开头。

**逐篇提取**

1. `UP-Person`
- 创新类型：工程组合偏新机制。统一 Prefix、LoRA、Adapter，并改 S-Prefix 和 L-Adapter 解决组件冲突。
- 为什么能发：TPR 里 CLIP 全量微调容易过拟合，已有 PETL 只做局部尝试。它把“保留 CLIP 知识”和“少量参数适配”讲成主问题，并用三数据集加参数量下降支撑。
- story 套路：全量微调会破坏大模型原有知识，参数高效适配能同时保留预训练知识和学习 ReID 任务知识。

2. `Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement`
- 创新类型：问题分解加工程组合。把 VI-ReID 的干扰拆成相机、模态、身份无关冗余三层。
- 为什么能发：多数 VI-ReID 只讲模态差异，忽略相机视角和背景差异。它用层级解耦、对抗损失、跨模态残差信息重组形成完整证据链。
- story 套路：跨模态差异不是单一差异，必须逐层净化，先去相机扰动，再提模态不变身份特征。

3. `Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders`
- 创新类型：新机制。提示词和图像编码器交替优化，互相强化。
- 为什么能发：原始 CLIP 或大模型生成的描述不关注身份细节，两阶段训练又是单向利用文本。它把提示词从静态辅助变成可被 ReID 编码器反向教育的身份桥梁。
- story 套路：好 prompt 不是先验给定的，而是在 ReID 训练中从图像编码器学出来，再反过来对齐可见光和红外特征。

4. `Visible-Infrared Person Re-Identification With Real-World Label Noise`
- 创新类型：问题重定义加新数据设定。把 VI-ReID 从干净标签扩展到真实标签噪声。
- 为什么能发：已有噪声工作多用随机噪声，和真实跨模态标注错误不一致。它分析真实噪声来源，重标 SYSU-Real，并设计 clean、obvious noisy、indistinguishable noisy 三类样本的混合损失。
- story 套路：真实系统里最先坏掉的不是模型结构，而是标签可信度。先把真实噪声定义清楚，再用针对性损失处理不同噪声层级。

5. `When Large Vision-Language Models Meet Person Re-Identification`
- 创新类型：新机制。把 LVLM 的生成能力变成一个用于检索的身份语义 token。
- 为什么能发：LVLM 逐词生成和 ReID 特征检索范式冲突，直接描述图片太慢，也不一定服务身份匹配。它用指令生成一个 `<REID>` token，再和视觉 token 交互，最后用该 token 做身份表征。
- story 套路：不要让 LVLM 逐张对话式比人，而是把它压缩成可检索的身份语义 token，接入标准 ReID 训练和推理流程。

**综合反推**

这批的共同启发是：能发的点不是“再加一个分支”，而是把一个旧问题改成更具体、更真实、更可验证的问题。最有价值的套路有三个：把干扰源重新分层，把静态语义提示改成训练闭环，把真实噪声或真实约束引入设定。

**候选 1：3D 可见表面一致性 ReID**

一句话 headline：航拍到地面 ReID 不应匹配整张图，而应匹配同一个人在不同视角下仍可见的 3D 人体表面证据。

挂团队资产：SMPL 3D 几何、aerial-ground、SOLIDER-Swin。

和这批最像工作的区别：它最像 HDNet 的“层级解耦”，但 HDNet 用对抗学习去掉相机和模态噪声，我们用 SMPL 投影显式定义哪些身体表面在航拍和地面视角中可比。它也不同于 LVLM-ReID 的语义 token，我们的 token 是几何可见性和人体表面，不是语言生成语义。

cheap kill-switch：先不训练。用现有 SMPL 或 2D pose 估计，把图像切成可见身体区域，测试 CARGO 或 AG-ReID.v2 上“几何可比区域相似度”是否比全局 SOLIDER 相似度更能区分正负样本。若同身份跨视角失败样本主要集中在低可见表面重叠区域，并且简单几何重加权能带来超过噪声线的提升，再立项。

**候选 2：跨视角身份证据可靠性门控**

一句话 headline：极端航拍低清场景里，关键不是提更多特征，而是让模型知道哪些身体区域在当前视角下不可信。

挂团队资产：pose 热图门控、aerial-ground、SOLIDER-Swin。

和这批最像工作的区别：它最像真实标签噪声那篇的“分清干净、模糊、错误样本”，但我们把噪声从标签层下沉到 token 层。每个身体区域都有可靠性，低清、遮挡、俯视压缩造成的区域不应该同等参与身份学习。

cheap kill-switch：冻结 SOLIDER，用 pose heatmap 置信度或 SMPL 可见性做零训练 token reweighting。看航拍 query 对地面 gallery 的错误检索是否被低置信区域主导。若只用高置信身体区域能稳定改善，说明“区域可靠性”是真问题。

**候选 3：航拍地面 ReID 的真实身份歧义学习**

一句话 headline：aerial-ground ReID 的核心噪声不是随机错标，而是低清航拍下多个身份在视觉上不可区分造成的真实歧义。

挂团队资产：aerial-ground、SMPL、SOLIDER-Swin。

和这批最像工作的区别：它最像 real-world label noise，但那篇处理 VI-ReID 的类别标签噪声。我们的切开点是跨视角低清导致的“身份歧义集合”，一个航拍人可能在视觉证据上对应多个地面候选，训练不应强行把所有非真值都当负样本。

cheap kill-switch：在 CARGO 或 AG-ReID.v2 上取 SOLIDER top-k 错误但高度相似的负样本，人工快速看一小批，判断是否存在真实不可分或标注不确定。再做一个极简 soft negative 训练或只在 triplet 里屏蔽疑似歧义负样本。若 rank1 或 mAP 有明显提升，说明这个设定值得扩成方法稿。

**候选 4：训练期 3D 语义 token，测试期纯 ReID**

一句话 headline：用 SMPL 生成训练期身份几何 token，把 LVLM 式“语义 token”改成不依赖语言、不增加测试成本的 3D 人体 token。

挂团队资产：SMPL 3D 几何、SOLIDER-Swin、pose 热图门控。

和这批最像工作的区别：它最像 LVLM-ReID，但 LVLM-ReID 推理时仍用 LVLM 生成 `<REID>` token，成本高且语义偏外观。我们的 token 来自 SMPL joints、mesh visibility、2D projection 和 pose confidence，只在训练期蒸馏到 SOLIDER 特征，测试仍是单 embedding。

cheap kill-switch：先做训练很轻的辅助头。冻结或半冻结 SOLIDER，在现有数据上预测 SMPL joint visibility、身体表面区域或姿态几何 token，看辅助监督是否能改善跨视角检索。若只学几何 token 不改善，说明几何语义没有进入身份判别，及时停。

我最建议优先试候选 1。它同时占“问题新”和“机制新”，和这批论文的切开点清楚，也最能发挥你们已经打通的 SMPL 基建。候选 3 是问题层面最像 B 类故事的备选，但需要一点真实歧义证据支撑，否则容易变成普通鲁棒损失。
