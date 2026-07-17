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
session id: 019ef71f-c690-71a3-956a-46bd1cec2034
--------
user
你是 ReID 论文创新挖掘员 17/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(17/20):
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则读当前目标和实验状态，再只抽每篇 PDF 的前三页。后面会把重点放在能挂团队资产、能形成 B 类论文故事的候选点上。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../../experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse
hook: PreToolUse
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

codex
我已经确认当前目标仍是“读文献做差距分析，先不要开实验”。现在开始逐篇抽前三页，先处理前四篇，避免一次输出太大把关键信息挤掉。
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
/bin/zsh -lc "pdftotext -l 3 'Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Text-Guided Cross-Modal Alignment with Attribute and
Contour Prototypes for Visible-Infrared Person
Re-Identification
YONG TAO and XINMING ZHANG, School of Computer Science and Technology, University of
Science and Technology of China, Hefei, China
Visible-infrared person re-identification (VI-ReID) aims to match pedestrian images captured under visible
and infrared modalities, which suffer from significant domain discrepancies. Existing approaches either
synthesize cross-modal images or learn modality-invariant representations, yet often encounter semantic
degradation or limited alignment capacity. Recent vision-language models leverage textual semantics for
modality bridging; however, CLIP-based frameworks typically rely on learnable token proxies with limited
expressiveness. In this article, we propose a novel semantic-driven framework that explicitly generates rich,
modality-agnostic textual descriptions from images as alignment cues. Specifically, we design a dual-branch
Text Semantic Generation Module that includes: (1) an Attribute-Aware text description Generation module
using prompt-based templates to capture modality-invariant identity cues, and (2) a Contour-Aware text
prompt Module that provides complementary structural information often missing in textual descriptions.
To reconcile semantic heterogeneity, a Text Re-definition Module (TRM) fuses instance-level and class-level
semantics into unified representations, enabling fine-grained alignment with image features. Furthermore, we
construct category-level textual prototypes as global semantic anchors to enhance cross-modal consistency.
Extensive experiments on two standard VI-ReID benchmarks demonstrate that our method achieves superior
performance, validating its effectiveness in semantic-guided modality alignment.
CCS Concepts: • Computing methodologies → Matching; Visual content-based indexing and retrieval;
Additional Key Words and Phrases: Person re-identification, Vision-Language Learning, infrared person
re-identification
ACM Reference format:
Yong Tao and Xinming Zhang. 2026. Text-Guided Cross-Modal Alignment with Attribute and Contour
Prototypes for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 22,
4, Article 113 (March 2026), 18 pages.
https://doi.org/10.1145/3796720

1

Introduction

Person re-identification (ReID) plays a crucial role in public safety and urban surveillance, as
it enables the tracking and localization of suspects by analyzing images captured from city-wide
surveillance cameras. Early ReID studies [23] mainly focused on pedestrian retrieval between
Authors’ Contact Information: Yong Tao, School of Computer Science and Technology, University of Science and Technology
of China, Hefei, China; e-mail: taoguandiu@163.com; Xinming Zhang (corresponding author), School of Computer Science
and Technology, University of Science and Technology of China, Hefei, China; e-mail: xinming@ustc.edu.cn.

This work is licensed under Creative Commons Attribution International 4.0.
© 2026 Copyright held by the owner/author(s).
ACM 1551-6865/2026/3-ART113
https://doi.org/10.1145/3796720
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

113:2

Y. Tao and X. Zhang

RGB images captured by daytime cameras. However, such settings are inadequate for crossmodal retrieval tasks involving RGB images from daytime cameras and infrared (IR) images
from nighttime cameras. The visible-infrared person re-identification (VI-ReID) task aims
to retrieve all images containing the same identity from a gallery of images captured under one
modality (e.g., nighttime IR), given a probe image captured under the opposite modality (e.g.,
daytime RGB). In recent years, numerous works [2, 3, 14, 25] have attempted to tackle this problem
by narrowing the modality gap between RGB and IR images, which is widely regarded as one of
the key challenges limiting the performance of cross-modal retrieval.
Mitigating the modality discrepancy between daytime RGB images and nighttime IR images
remains a fundamental challenge in visual identity recognition (VI-ReID). Existing approaches
can be broadly categorized into two paradigms. The first line of work leverages generation-based
methods [4, 18, 19], which learn a mapping between RGB and IR domains via image translation,
aiming to align different modalities within a shared distribution space. While effective to some
extent, these methods inevitably introduce generation artifacts. Specifically, generated images often
suffer from distributional shifts compared to real data, and their perceptual quality is generally
suboptimal, thereby limiting overall retrieval performance. The second category includes modalityinvariant representation learning approaches [5–7, 10], which aim to extract shared, identitydiscriminative features across modalities solely through visual supervision. Despite improvements
over generation-based methods, these approaches tend to overlook high-level semantic cues, as
they rely purely on image-level features. As a result, their ability to bridge the intrinsic modality
gap remains constrained [12]. Recent efforts have explored the integration of vision-language
models (VLMs), such as CLIP [15], into VI-ReID. These methods [1, 9, 20] attempt to leverage
implicit textual semantics derived from image features as auxiliary alignment signals. However,
most of these models rely on a limited number of learnable semantic tokens to represent latent
textual information. Such token representations lack sufficient semantic richness and generalization
capacity, thereby hindering their effectiveness in bridging modality gaps.
To overcome the aforementioned limitations, we propose a novel framework (Figure 1) that
incorporates an image-to-text generation paradigm to construct discriminative and modalityinvariant textual descriptions, thereby replacing traditional learnable token vectors. The proposed
framework consists of two complementary modules: (1) an attribute-aware text description
generation (ATG), which synthesizes identity-specific textual descriptions based on pre-defined
attribute templates (e.g., gender, clothing style) while deliberately omitting modality-sensitive
information such as color, and (2) a contour-aware text prompt (CTP) module, which captures
structural semantics (e.g., body shape, posture) to compensate for the lack of fine-grained contour
information typically absent in large-scale captioning models. These two modules jointly generate
semantic descriptions that are intentionally free from modality bias, serving as reliable anchors for
cross-modal alignment.
However, the ATG generates instance-level descriptions, while the CTP yields class-level contour
representations in the form of learnable textual vectors, making their direct integration nontrivial. To
bridge this gap, we introduce a text re-definition module (TRM) that serves two key functions:
(1) fusing attribute-aware and contour-aware representations into a unified and semantically
enriched textual prototype and (2) further narrowing the modality gap between generated textual
and visual features via adaptive refinement. The final class-level textual prototypes derived from
TRM are then adopted as shared semantic anchors to align both RGB and IR features within a
unified embedding space. This design enables effective modality-agnostic alignment, substantially
reducing cross-modal discrepancies and significantly enhancing retrieval performance on VI-ReID
benchmarks. Our main contributions are summarized as follows:

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.

Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID

113:3

Fig. 1. Comparison of different person VI-ReID frameworks. (a) While visual encoders primarily extract
modality-shared identity features, they often overlook valuable semantic cues provided by other modalities.
(b) “CSDN” [25], a CLIP-based VI-ReID variant, uses learnable feature vectors to represent textual semantics,
providing alignment cues for cross-modal matching. (c) Our proposed method introduces an image-to-text
generator to produce rich, modality-invariant textual descriptions for images from different modalities, which
serve as semantic cues for cross-modal alignment.

— Modality-invariant text semantic generation: We introduce a dual-branch text generation
framework that produces attribute-aware and contour-aware descriptions from pedestrian
images. The attribute branch employs modality-agnostic templates to encode identity-related
semantics, while the contour branch captures structural cues such as body shape and pose,
which are often overlooked in existing approaches.
— Unified text representation and fine-grained alignment: A TRM is proposed to integrate heterogeneous semantic descriptions into a unified representation. This unified text is further aligned
with visual features through a fine-grained interaction mechanism, effectively narrowing the
modality gap between RGB and IR modalities.
— Prototype-guided semantic alignment: We derive category-level textual prototypes from the
unified descriptions, which act as modality-shared semantic anchors to guide the alignment
of heterogeneous visual features in a prototype-centered manner.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 113. Publication date: March 2026.


 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

1

Structured Distilled 3D Gait Fields for
Clothes-Changing Person Re-Identification in Videos
Yubin Wang , Huimin Yu , Member, IEEE, Haoji Hu , Member, IEEE, Yongsheng Gao , Senior Member, IEEE

Abstract—3D gait representation, intrinsically tied to identity,
is of great importance in tackling the challenge of video-based
clothes-changing person re-identification (VCCRe-ID), where individuals’ clothing appearances vary across video clips captured
over an extended duration. Most existing methods derive 3D
gait features from sequential 3D skeletons or parameters of a
predefined 3D human model (e.g. SMPL). However, the sparsity
of 3D skeletons and the low-dimensional nature of 3D human
model parameters not only limit their expressive power and
discriminative ability, but also impede alignment with higherdimensional appearance features. To address these issues, we
propose Structured Distilled 3D Gait Fields (SD-3DGF) to learn
better aligned, more expressive, and discriminative 3D gait
features for VCCRe-ID. SD-3DGF defines distilled feature fields
that evolve over time, allowing the simulation of 3D gait dynamics
from temporal 2D descriptors. Specifically, it densely transfers
frame-level shape knowledge from structured 3D space (i.e., 3D
mesh surface) to the 2D image domain via 3D-2D knowledge
distillation, and models local spatiotemporal correlations among
the distilled features to extract 3D gait features. This learning
paradigm makes 3D gait features more compatible with appearance features, as they are naturally aligned in the spatiotemporal
domain. To fully exploit this potential, we develop a novel
Modality-Representative Interactive Encoding (MRIE) module,
which facilitates efficient gait-appearance interaction and joint
encoding via global and local representative tokens. Experiments
on both VCCRe-ID and standard video-based Re-ID benchmarks
demonstrate the effectiveness of our method. The code is available
at https://github.com/yubinwang2021/SD-3DGF.
Index Terms—Video-based clothes-changing person reidentification, structured distilled 3D gait fields, 3D gait features,
modality-representative interactive encoding.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) involves re-identifying
a particular person across a series of disjoint cameras,
which plays a pivotal role in numerous practical applications
including intelligent surveillance, the retail industry, and security authentication [1], [2], [3]. After years of focused research,
current short-term Re-ID methods [4], [5], [6] have achieved
remarkable performance on public benchmarks, surpassing
human-level recognition, and have also exhibited promising
performance when extended to cross-modal retrieval, such
as leveraging additional text modalities [7], [8] or infrared

This work was supported in part by the Research Project of ZJULeague Research and Development Center, Zhejiang Laboratory under Grant
2019KD0AB01. (Corresponding authors: Huimin Yu; Yongsheng Gao.)
Yubin Wang, Huimin Yu and Haoji Hu are with the College of Information
Science and Electronic Engineering, Zhejiang University, Hangzhou, China (email: zjuwyb2021@zju.edu.cn; yhm2005@zju.edu.cn; haoji hu@zju.edu.cn).
Yongsheng Gao is with the Institute for Integrated and Intelligent Systems, Griffith University, Nathan, QLD4111, Australia (e-mail:
yongsheng.gao@griffith.edu.au).

3D vertex
embeddings

(a)

(b)

(c)

(d)
Distill

(e)
Fig. 1. Examples of different auxiliary modalities for gait modeling in
VCCRe-ID. (a) the input video tracklet, sampled from the CCVID dataset;
(b) 2D keypoints and silhouettes; (c) part segmentation maps; (d) SMPL
parameters; (e) dense correspondence embeddings, from the proposed SD3DGF framework.

modalities [9], [10]. However, these methods assume that
pedestrians appear briefly within the camera network, with
their clothing remaining unchanged. In real-world scenarios,
pedestrians may be re-captured by the camera after a long
period (a day, a month, or even a year). In such cases,
their clothing is likely altered due to various factors such as
temperature changes and shifts in personal preferences [11],
[12]. For this reason, short-term Re-ID methods, which overrely on clothing appearances, often fail in long-term settings.
To tackle this issue, researchers place increasing emphasis on clothes-changing person re-identification (CCRe-ID),
where the key challenge lies in extracting clothing-invariant
identity features [13]. A prominent line of work [14], [15],
[16] explores 2D and 3D body shape as soft biometrics, as
they provide personalized information [17]. However, both 2D
and 3D shape representations remain susceptible to clothing
variations. For example, an individual’s perceived body proportions can differ drastically when wearing a T-shirt versus
a thick down jacket.
To address this limitation, gait emerges as a promising
biometric cue, owing to its spatiotemporal dynamics and robustness to clothing variations [18], [19]. Using gait as an auxiliary modality for extracting clothing-invariant features fosters
growing interest in video-based clothes-changing person re-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

identification (VCCRe-ID) [20], [21], [22], [23]. As shown
in Fig. 1(b) and (c), some VCCRe-ID approaches [20], [21]
learn 2D gait representations using sequential 2D keypoints,
silhouette sequences and part segmentation maps. However,
the learned 2D gait features may suffer from appearance ambiguity caused by viewpoint variations due to the lack of spatial
alignment. Another line of work focuses on training the gait
stream using 3D auxiliary modalities [22], [23]. For instance,
ASGL [22] extracts gait embeddings through spatiotemporal
graphs of 3D skeletons. In contrast, CMCL [23], as illustrated
in Fig. 1(d), models 3D gait dynamics using sequential SMPL
parameters [24]. However, both 3D skeletons and 3D human
model parameters are inherently low-dimensional and sparse,
exhibiting limited identity discriminability and poor distribution alignment with the high-dimensional appearance features.
Attempts to expand their dimensionality, such as mapping
them to the RGB feature space through a spatial transformation
network [18], still produce highly sparse features that are at
risk of losing key semantic information, thereby compromising
the reliability of gait-appearance cross-modal interaction.
In this paper, we propose Structured Distilled 3D Gait Fields
(SD-3DGF), a new 3D gait representation learning paradigm
for VCCRe-ID. As shown in Fig. 1(e), different from prior
3D gait-based methods, SD-3DGF performs dense, frameby-frame transfer of 3D shape knowledge from structured
surface meshes to the image domain via 3D-2D distillation,
with the distilled frame-level features forming a structured
3D gait field. More specifically, based on the surface-based
geodesic distances, we pre-train vertex embeddings for the
canonical SMPL mesh, and attach them to the reconstructed
sequential body meshes. Then, using the estimated camera
parameters, we project both the 3D vertices and randomly
sampled surface points onto the image plane frame by frame,
generating dense 2D-3D paired data to guide the training
of a lightweight frame-level feature extractor. The features
extracted by this extractor are spatially aligned with the
input frame, i.e., each pixel is mapped to a dedicated pixel
embedding. Theoretically, pixel embeddings associated with
the same 3D vertex across frames are inherently consistent,
and those corresponding to the neighboring 3D points should
also be similar. Thus, local spatiotemporal correlations can be
naturally modeled via similarity learning [25], facilitating the
extraction of 3D gait features that encode discriminative local
patterns. The proposed 3D gait learning paradigm represents
gait as a spatiotemporal field aligned with the video sequence,
enabling seamless interaction with clothing-invariant appearance features. Furthermore, the learned 3D gait features are
robust to viewpoint variations, as the extracted frame-level
features capture rich alignment information with the canonical
3D space.
To facilitate complementary information sharing between
gait and appearance modalities, we propose a novel ModalityRepresentative Interactive Encoding (MRIE) module. MRIE
first employs the Dual-Granularity Representative Token Generation (DGRT) strategy to produce, for each modality, a
global representative token summarizing holistic identity information and a set of local representative tokens preserving
region-level discriminative details. This process also reduces

2

Canonical SMPL Mesh

Canonical SMPL Mesh

3D vertex
coordinates

...

3D vertex
embeddings

𝓛𝒗𝒆𝒑

Graph Convolutions

Fig. 2. 3D vertex embeddings pre-training pipeline. The graph convolutional
network takes the canonical vertex coordinates as input and produces the 3D
vertex embeddings, constrained by geodesic distances and tangent directions
on the mesh surface.

the spatial dimensions of tokens, which significantly reduces
the computational cost. The Gait-Appearance Joint Encoding
(GAJE) submodule then models cross-modality interactions
between the dual-granularity tokens from the two modalities,
and outputs the final identity feature via adaptive integration.
Our contributions are summarized as follows:
• We propose Structured Distilled 3D Gait Fields (SD3DGF), a new 3D gait learning scheme that enables the
simulation of 3D gait dynamics from distilled feature
fields. To the best of our knowledge, this is the first
study to mine 3D gait features for VCCRe-ID from a
perspective of 3D-2D distillation.
• For the training of the proposed 3D gait fields, we develop
a comprehensive pipeline for generating dense paired 2D3D data, comprising 3D vertex embedding pretraining,
3D body motion capture, and dense 3D-2D projection
with both vertices and sampled surface points.
• We propose a novel Modality-Representative Interactive
Encoding (MRIE) module, which introduces global and
local representative tokens to enable efficient interaction
and joint encoding of 3D gait and clothing-invariant
appearance features.
• Extensive experiments on four video-based clotheschanging Re-ID benchmarks (VCCR, CCVID, CCVR, and CCV-S) and three standard video-based Re-ID
benchmarks (MARS, LS-VID, and iLIDS-VID) show that
our method not only delivers strong performance under
clothes-changing settings, but also maintains robustness
under standard video-based Re-ID settings.
II. R ELATED W ORKS
A. Video-Based Person Re-ID
Video-based person Re-ID (VRe-ID) has been extensively
investigated in recent years. The mainstream video-based ReID methods [26], [27], [28], [29], [30], [31], [32], [33], [34]
can be broadly categorized into two types: frame-level feature
aggregation methods and sequence-level spatiotemporal modeling methods. Frame-level feature aggregation methods [26],
[27], [28], [29], [30] first extract frame-level features, and
then aggregate them along the temporal dimension to obtain
a video-level representation. Zheng et al. [26] adopt a pooling
strategy to fuse frame-level features. Hou et al. [27] and Bai

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3691145

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

et al. [28] leverage attention mechanisms to focus on salient
regions in frame-level features and capture temporal relations.
Yu et al. [29] and Su et al. [30] design a CLIP-based video
Re-ID framework to enhance the multi-modal understanding of
frame-level features. Sequence-level spatiotemporal modeling
methods [31], [32], [33] extract unified representations from
video sequences by jointly modeling spatial and temporal
information. Gu et al. [31] employ 3D Convolutional Neural
Networks (3D-CNNs) to uniformly encode spatiotemporal
information. Li et al. [33] leverage a dual-stream network
with I3D [35] as the backbone to learn spatiotemporal motion
and appearance features. Tang et al. [32] propose a multistage Transformer network to learn attribute- and identityrelevant features from raw video sequences. Leng et al. [34]
mine structured spatiotemporal relations in both Euclidean and
hyperbolic spaces using dynamic prompt graphs.
B. Clothes-Changing Person Re-ID from a Single Image
The key challenge of image-based clothes-changing person
Re-ID (CCRe-ID) is to learn clothing-invariant identity features from a single image. The existing literature mainly falls
into two categories: disentanglement-based methods [36], [37],
[38], [39], [40] and biometrics-based methods [15], [41], [42],
[43]. Disentanglement-based methods [36], [37], [38], [39],
[40] emphasize disentangling identity-related appearance features from clothing-related ones, without relying on additional
modalities. Xu et al. [36] and Xiong et al. [37] decouple
identity and clothing features through reconstructing person
images or parsing maps. Wang et al. [38] decouple attribute
features through clustering and recompose them for identity
modeling. Li et al. [39] and Liang et al. [40] use CLIP [44]
to disentangle identity-related and identity-irrelevant feature
subspaces guided by text descriptions.
Biometrics-based methods [15], [41], [42], [43] use auxiliary modalities to mine discriminative biometric cues. Shu
et al. [41] randomly replace clothing pixels guided by part
segmentation to learn clothing-invariant features. Jin et al. [42]
predict 2D gait dynamics from a single image to regulate the
appearance features. Chen [15] treat the 10-dimensional SMPL
shape parameters as 3D shape features and fuse them with
appearance features. Liu et al. [43] perform 3D clothed human
reconstruction to learn discriminative shape latent codes.
C. Clothes-Changing Person Re-ID in Videos
Video-based clothes-changing person Re-ID (VCCRe-ID)
exploits rich spatiotemporal cues in videos to mine clothinginvariant identity features. Gu et al. [45] introduce a reliable
and simple VCCRe-ID baseline, which achieves competitive
benchmark performance using only the RGB modality. The
mainstream VCCRe-ID approaches [20], [21], [23], [46], [47],
[48] place great emphasis on learning video-level canonical
shape or gait features as soft biometric cues. Chen et al. [20]
and Lu et al. [21] present a 2D gait and appearance fusion
network using keypoint heatmaps or silhouettes as additional
input. Wang et al. [46] propose a graph-based pipeline for
joint encoding of 2D gait and appearance features. Nguyen et
al. [22] adopt sequential 3D skeletons for gait modeling. Han

3

et al. [47] and Nguyen et al. [48] derive canonical 3D shape
features at the video level through temporal aggregation of
10-dimensional SMPL shape parameters. However, the lowdimensional SMPL shape parameters carry very limited information and exhibit weak discriminative capability. Moreover,
their feature space differs substantially from that of RGB
features. Nguyen et al. [23] take a further step by utilizing
SMPL parameters to guide the regulation of silhouette feature
maps via a 3D transformation network. However, directly mapping SMPL parameters into a high-dimensional visual space
produces sparse features and loses key 3D prior knowledge.
D. Dense Correspondence Embeddings
A core technique of our method is the learning of dense correspondence embeddings. Güler et al. [49] design a cascaded
network to estimate per-pixel UV coordinates and generate
IUV maps, which bridges the 2D and 3D domain. Wang
et al. [50] introduce a per-pixel vertex classification loss
for extracting pixel-wise surface embeddings. Liu et al. [51]
facilitate indoor parsing by distilling 3D scene embeddings
onto the image plane. Ianina et al. [52] explore scalable
correspondence embeddings across different 3D meshes. Hou
et al. [53] devise a Transformer-based refinement network
to generate fine-grained full-body correspondence embedding
maps. Hönig et al. [54] leverage diffusion models to learn
2D-3D correspondence mappings between images and point
clouds. Yang et al. [55] propose to learn video-level correspondence embeddings for animating deformable 3D objects.
Different from the prior works, we propose a novel method
using pre-trained 3D vertex embeddings and 3D-2D distillation
for learning dense correspondence embeddings.
III. M ETHODOLOGY
This section presents the overall architecture of the proposed method, detailing the SD-3DGF learning pipeline, the
MRIE module for efficient gait-appearance interaction, and the
formulation of training losses.
A. Overview
Fig. 3(a) depicts the overall architecture of our method and
illustrates the relationships among its modules. It adopts a
dual-stream structure comprising an appearance stream SA , a
gait stream SG , and the MRIE module. SA and SG operate in
parallel, while the MRIE module receives features from both
streams and yields the final identity representation. Given a
video tracklet T = {It | t = 1, 2, . . . , T } of a pedestrian
with T sampled frames, the appearance stream SA seeks
to extract clothing-invariant appearance features from T . It
employs a ResNet-50 backbone [56], using the first three
residual blocks to extract frame-level appearance features
A = {At | t = 1, 2, . . . , T }. The resulting feature set A is then
concatenated along the temporal dimension to form the initial
video-level appearance feature XA ∈ RT ×h×w×c , which is
further regularized by an orthogonal loss Lorg to achieve
clothing desensitization. The 3D gait stream SG , on the other
hand, is designed to learn expressive and discriminative 3D

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:09:18 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -l 3 'Spectrum-guided feature enhancement network for event person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112705

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Spectrum-guided feature enhancement network for event person
re-identiﬁcation
Hongchen Tan
a

a,∗, Yi Zhang

b , Xiuping Liu

b

Institute of Future Technology, Dalian University of Technology, 116024, Dalian, Dalian, China

b School of Mathematical Sciences, Dalian University of Technology, 116024, Dalian, China

a r t i c l e

i n f o

Keywords:
Event person re-ID
Spectrum attention mechanism
Consecutive patch dropout module
Event contextual semantics.

a b s t r a c t
As a novel biosensor, the event camera holds signiﬁcant potential, particularly regarding privacy preservation.
However, compared to traditional cameras, event streams often contain noise and possess extremely sparse semantics, posing a formidable challenge for event-based person re-identiﬁcation (event Re-ID). To address this, we
introduce a novel event person re-identiﬁcation network: the Spectrum-guided Feature Enhancement Network
(SFE-Net). The SFE-Net consists of two innovative components: the Multi-grain Spectrum Attention Mechanism
(MSAM) and the Consecutive Patch Dropout Module (CPDM). MSAM employs a fourier spectrum transform strategy to ﬁlter event noise, while also utilizing an event-guided multi-granularity attention strategy to enhance and
capture discriminative person semantics. CPDM employs a consecutive patch dropout strategy to generate multiple incomplete feature maps, encouraging the deep Re-ID model to equally perceive each eﬀective region of the
person’s body and capture robust person descriptors. Extensive experiments on Event Re-ID datasets demonstrate
that our SFE-Net achieves the best performance.

1. Introduction
Person re-identiﬁcation (Re-ID) is a technique that identiﬁes a speciﬁc person across non-overlapping camera systems, and it is widely
utilized in various surveillance applications. Currently, many outstanding RGB-based methods [1–3] have achieved good performance. However, traditional RGB cameras struggle to capture eﬀective person
semantics in high-exposure and low-light environments. Moreover, capturing too many detailed semantics also poses a risk of privacy leakage. As a novel biosensor, event cameras produce asynchronous data
streams that record the timestamp, location, and polarity of intensity
changes. They can still perceive and capture eﬀective person semantics
in the harsh lighting conditions mentioned above. Besides, the imaging
principle of event cameras means that they do not capture too many semantic details of persons, which also reduces the probability of privacy
leakage. Therefore, conducting person re-identiﬁcation tasks based on
event streams is a highly promising research endeavor.
From the perspective of data representation, in contrast to conventional RGB data, event data exhibit notably sparse semantic content and
incorporate substantial noise, as depicted in Fig. 1. This makes it challenging to directly adapt Re-ID technology from RGB-based to eventbased modalities. However, rich contextual semantics from RGB modalities remain critical for object perception and recognition. To address

this, Cao et al. [4] proposed combining RGB and event modalities for
long-sequence person re-identiﬁcation tasks. By leveraging the complementary strengths of both modalities, the performance of RE-ID was
signiﬁcantly improved. Nevertheless, such hybrid approaches inadvertently reintroduce facial portrait exposure risks due to their reliance
on RGB data. Additionally, long-sequence Re-ID requires substantially
more computational resources and imposes stricter data quality requirements. In this context, Ahmad et al. [5] pioneered person matching using
only event data. Their work focuses on preventing grayscale image reconstruction from event streams, thereby enhancing privacy protection.
However, their approach overlooks two key challenges: (1) extracting
high-quality person semantics from sparse event data, and (2) mitigating the impact of event noise. Similarly, Ahmad et al. [6] attempted to
enhance discriminative features by unfolding person contours in polar
coordinates. However, event-captured data inherently contains signiﬁcant noise and exhibits extremely sparse semantics. Consequently, the
resulting person contours remain fragmented, and the discriminative
power of the extracted descriptors remains limited. So, we try to alleviate two issues to capture high-quality person descriptors.
Event cameras capture changes in light intensity (or electromagnetic
waves) within a scene, generating event-based data streams. However,
during this process, these event streams often contain signiﬁcant noise
due to factors such as photon instability, the sensitivity threshold of

∗ Corresponding author.

E-mail addresses: thc123@dlut.edu.cn, tanhongchenphd@bjut.edu.cn (H. Tan), dlutzy@mail.dlut.edu.cn (Y. Zhang), xpliu@dlut.edu.cn (X. Liu).
https://doi.org/10.1016/j.patcog.2025.112705
Received 19 October 2024; Received in revised form 16 September 2025; Accepted 30 October 2025
Available online 2 November 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112705

H. Tan et al.

sensor hardware, and pixel-level defects (as shown in the right panel
of Fig. 1). Moreover, owing to the highly discrete nature of event semantics, eﬀective semantic information and noise exhibit strong coupling, making them diﬃcult to distinguish. As illustrated in Fig. 1,
a notable characteristic of this noise is that it predominantly manifests as isolated points. Consequently, in the frequency domain, such
noise components are highly concentrated in high-frequency regions.
Thus, we can utilize the spectral convolution theorem [7,8] to separate low-frequency components (representing general semantic information) from high-frequency components (corresponding to noise) in
the spectral domain. In addition to most noise manifesting as isolated
points, the sparsity of event semantics causes a substantial portion
of noise to also appear as semantic isolated points. Simply removing
high-frequency information at a single scale under these conditions inevitably leads to the loss of valuable semantic content. Multi-scale perception further enhances semantic interactions and fusion across regions of varying sizes, thereby stabilizing eﬀective semantics as much
as possible. Therefore, we propose integrating frequency-domain denoising with multi-scale receptive ﬁelds, enabling the model to simultaneously perform denoising and strengthen general semantics across
diﬀerent scales. However, spectral-domain denoising strategies are
prone to oversmoothing artifacts. Moreover, the inherent discreteness
of event semantics hinders the representation of high-quality semantic
information. To address this, we further leverage attention mechanisms
to enhance interactions and fusion among event semantics, thereby
strengthening semantic representation. Therefore, we propose a joint
spatial-spectral denoising and semantic interaction strategy that integrates semantic modeling across both domains.
In addition, deep models tend to concentrate solely on speciﬁc local key semantics and overlook the overall eﬀective semantics. This is
often attributed to the convolutional model being “lazy”. To address
this, the PCB strategy [9] was designed to guide deep models in extracting eﬀective semantics from small image patches. However, due
to the sparse nature of event semantics, accurately determining person
identities based solely on semantics from a single small patch remains
challenging. Moreover, semantic sparsity results in a very low proportion of eﬀective semantics within feature maps, which frequently causes
Dropout-based strategies [10–13] to focus on non-informative semantic
regions, thereby reducing their eﬀectiveness. To mitigate this issue, we
propose an inverse strategy to the PCB [9], which involves systematically erasing features to generate multiple incomplete feature representations. This approach encourages the deep model to balance its perception of eﬀective semantics across diﬀerent spatial regions.
To achieve the goal, we propose a novel event person reidentiﬁcation network, Spectrum-guided Feature Enhancement Network
(SFE-Net), to reduce the interference of noise on the event-streams and
fully explore the discriminative person semantics from the sparse eventstreams. The SFE Network incorporates two novel designs: the Multigrain Spectrum Attention Mechanism (MSAM) and the Consecutive
Patch Dropout Module (CPDM). In MSAM, we ﬁrst convert multi-scale
event semantics into the spectral domain and ﬁlter out high-frequency
noise information. Then, we attempt to enhance the interaction between
event semantics and strengthen the representation of person semantics,
which, to some extent, alleviates the shortcomings of smoothing caused
by the previous ﬁltering process. The second novel design, CPDM, involves a two-step process. First, we conduct a uniform partition on the
feature map, dividing it into smaller patches. Then, we independently
and continuously drop each patch from top to bottom on the feature
map. This operation results in multiple incomplete person feature maps.
By using these multiple incomplete feature maps, the Re-ID model is
driven to equally focus on every discriminative area. Since then, the ReID model captures discriminative semantics from a wider area rather
than getting trapped by a single feature area. This work makes the following contributions:
(i) We introduce the MSAM to eﬀectively ﬁlter noise from event
streams and enhance the representation of person semantics.

Fig. 1. Display of traditional RGB person samples versus event-based person
samples. Unlike RGB images, the semantic information encoded in event tensors is highly sparse and discrete; event cameras intentionally avoid capturing
redundant facial feature details. This characteristic makes them advantageous
for protecting person portraits.

(ii) We design the CPDM to encourage the event Re-ID model to
capture discriminative semantics from a broader area, rather than being
limited to a single feature area.
(iii) We propose the SFE-Net for the event person Re-ID task. Extensive experiments on many Event Re-ID datasets demonstrate that our
SFE-Net achieves the best performance.
2. Related work
2.1. Various person re-ID task
Person Re-ID, a pivotal technology for intelligent video surveillance,
has attracted substantial research attention. Early image-based Re-ID
methods, as demonstrated in studies such as [14], captured the holistic
semantic information of individuals by extracting global features and
employed specialized metric learning losses to enhance performance.
To enable the model to distinguish ﬁne-grained appearance details, recent approaches have adopted local patch partitioning strategies [15],
in which person images are divided into localized regions for detailed
semantic extraction.
Person re-identiﬁcation (Re-ID) faces diverse challenges in realworld scenarios, including occlusion, viewpoint variations, resolution
discrepancies, missing query images, and illumination changes; to address these issues, researchers have developed specialized methods such
as mask-aware mechanisms in [16,17] to mitigate partial occlusion effects, contrastive learning frameworks in [18] for viewpoint adaptation, and dual-resolution feature fusion strategies in [19,20] to bridge
resolution gaps, while alternative paradigms like text-aided retrieval
in [21,22] leverage textual descriptions for zero-shot person retrieval,
infrared modality fusion methods demonstrated in [23,24] signiﬁcantly
improve nighttime Re-ID performance, and disguise-robust approaches
including [25,26] extract gait patterns and appearance-invariant cues
to combat deliberate disguises.
With the growing prominence of Person Re-ID research, privacy
concerns have garnered increasing scrutiny, yet few methods [27–29]
speciﬁcally address this issue. Existing approaches include: 1) face blurring techniques [30] for identity anonymization; 2) person identity shift
(PIS) [31] that preserves image relationships while removing absolute
identity; and 3) cloud-based encryption schemes [27] enabling secure
matching on encrypted data. However, these methods fail to ensure
end-to-end privacy protection, as unauthorized camera access remains
a critical vulnerability. Event cameras, which capture essential motion
information without redundant details, oﬀer a promising privacypreserving alternative to RGB sensors. Their superior performance under
extreme lighting conditions [32,33] has spurred research on event-based
person vision. While some studies focus on reconstructing RGB images
from event data [34], innovative work [5,6] seeks to prevent such reconstruction to enhance privacy. Nevertheless, the sparse and discrete
2

Pattern Recognition 172 (2026) 112705

H. Tan et al.

3.1. Basic information for SFE-Net

nature of event data poses challenges for semantic perception-a limitation overlooked by current methods. To address this gap, we propose a
framework that extracts robust person descriptors from event semantics
while maintaining privacy protection.

The SFE-Net comprises four key components: a Backbone, an EventStream Anonymization Module (EAM), a Multi-grain Spectrum Attention Mechanism (MSAM), and a Consecutive Patch Dropout Module
(CPDM). Its architectural design, including the implementations of
MSAM and CPDM along with the data ﬂow, is illustrated in Fig. 2. This
section provides a concise overview of three fundamental components:
Input Event Representation, the Backbone, and EAM.
Input Event Representation. The output of an event camera is
an asynchronous event stream that encodes the location, time, and
polarity of the intensity changes (increase or decrease in intensity).
For fairness, we also adopt voxel grids of event data, following [5,6].
Spatiotemporal coordinates, 𝑥𝑘 , 𝑦𝑘 , 𝑡𝑘 , lie on a voxel grid such that
𝑥𝑘 ∈ {1, 2, … , 𝑊 }, 𝑦𝑘 ∈ {1, 2, … , 𝐻}, and 𝑡𝑘 ∈ {𝑡0 , 𝑡0 + △𝑡, … , 𝑡0 + 𝐵 △
𝑡}, where 𝑡0 is the ﬁrst time stamp, △𝑡 is the bin size, and 𝐵 is the number of temporal bins and 𝑊 , 𝐻 are the sensor width and height. So, like
many outstanding methods, the input of our SFE-Net is 𝑋𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 .
Backbone. Our SFE-Net adopted the ResNet-50 pre-trained on ImageNet as the backbone network, to encode a person event-stream 𝑋𝑒 ∈
ℝ𝐵×𝑊 ×𝐻 . To get a larger size high-level feature tensor and enhance the
event information representation, we modify the basic structure of the
ResNet-50. We adopted two “ResBlock” to replace the whole “ResNet-50
Stage 4”. Therefore, we can get a larger feature tensor 𝑇 (𝑥) ∈ ℝ24×8×2048 .
Event-Stream Anonymization Module (EAM). Semantics in event
streams are sparse, which can protect the person’s privacy to a certain extent. However, deep image reconstruction methods [34] have
demonstrated impressive abilities in recovering gray images from event
streams, representing a potential threat to the privacy of event-based
vision applications. To this, [5] proposed the Event-Stream Anonymization Module (EAM) to modify the event streams to prevent the image
reconstruction techniques from converting events into intensity images
that can reveal privacy-sensitive information. At the same time, EAM
preserves useful spatial information for performing person Re-ID successfully. For more information about EAM refer to [5] or Appendix A.1.
The EAM takes a raw event-voxel 𝑋𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 and output anonymized
event-voxel 𝑋̂ 𝑒 ∈ ℝ𝐵×𝑊 ×𝐻 . In our SFE-Net, we also adopt such privacy
protection strategy [5]. Unlike them, our SFE-Net focuses on how to
perceive and capture high-quality person semantics from event streams.

2.2. Event-based vision task
As an innovative sensor modality, event cameras distinguish themselves through their high dynamic range (HDR), microsecond-level temporal resolution, and inherent privacy-preserving characteristics. Leveraging these unique advantages, recent years have witnessed a surge in
research exploring event-based vision applications across diverse domains.
Recent advances in event-based vision systems have demonstrated
signiﬁcant potential for object perception tasks [35]. propose a groundbreaking MoE (Mixture of Experts)-based heat conduction algorithm
that achieves an optimal trade-oﬀ between detection accuracy and
computational eﬃciency [36]. pioneer the integration of asynchronous
event cameras into 3D object detection frameworks, leveraging their
high temporal resolution and low data bandwidth to enable real-time
3D perception. For temporal-aware segmentation tasks, [37] introduce
a continuous-time processing paradigm that recurrently updates temporal embeddings by jointly modeling historical and current event streams,
thereby supporting persistent object segmentation. In the domain of motion tracking, [38] develop a salient event blob detector that identiﬁes regions with consistent optical ﬂow through a novel Field of Active Flow Directions (FAFD) representation constructed from the Surface
of Active Events (SAE), achieving robust object tracking performance.
Multi-modal fusion approaches further enhance perception capabilities by combining event data with complementary modalities [36,39].
demonstrate performance improvements in perception tasks through
cross-modal calibration between event cameras and RGB/Radar sensors.
Speciﬁcally, [40] address the spatio-temporal resolution mismatch between RGB and event modalities by designing a dedicated fusion mechanism, which they validate on object tracking benchmarks. Similarly,
[41] propose a hierarchical alignment strategy for event-RGB fusion,
achieving state-of-the-art results in monocular facial expression recognition.
The imaging mechanism of event cameras inherently couples noise
with event data, posing challenges for semantic interpretation. Consequently, extensive research [7,42] has explored denoising methodologies to enhance event-based semantic representations. Notably,
frequency-domain approaches [8,43] ﬁlter high-frequency noise while
preserving low-frequency components, integrating denoising into perception pipelines. Inspired by these advances, we propose extending frequency-domain denoising to event-based Re-ID (RE-ID) tasks to
obtain high-ﬁdelity semantic embeddings. However, three limitations
arise: (1) event semantics exhibit extreme sparsity, making single-scale
ﬁltering prone to over-smoothing and detail loss; (2) asynchronous event
generation results in insuﬃcient contextual aggregation; (3) existing
methods [8,43] rely on ﬁxed receptive ﬁelds, limiting adaptation to
noise variations. Unlike prior works, our approach employs multi-scale
receptive ﬁelds for noise perception and enhances semantic interactions
through cross-scale attention mechanisms.

3.2. MSAM
As described in Intro. 1, the imaging principle of event cameras
results in a signiﬁcant amount of noise in the event streams. Although
[5,6] also adopt event-based models for Re-ID tasks, they neglect to
address the critical challenges of event semantic denoising and enhancement, which are essential for improving feature discriminability
in sparse event data scenarios. Therefore, denoising becomes an important prerequisite for obtaining high-quality semantic representations of
events. Consequently, extensive research [7,42,44] has explored denoising methodologies to enhance event-based semantic representations.
Notably, frequency-domain approaches [8,43] ﬁlter high-frequency
noise while preserving low-frequency components, integrating
denoising into perception pipelines. Inspired by these advances, we
propose extending frequency-domain denoising to event-based Re-ID
(RE-ID) tasks to obtain high-ﬁdelity semantic embeddings. However, the
sparsity of event semantics causes a substantial portion of noise to also
appear as semantic isolated points. Simply removing high-frequency
information at a single scale under these conditions inevitably leads to
the loss of valuable semantic content. Multi-scale perception further
enhances semantic interactions and fusion across regions of varying
sizes, thereby stabilizing eﬀective semantics as much as possible.
Therefore, we employ convolutions with diﬀerent scales to perceive
event semantics and attempt to ﬁlter high-frequency information of
event semantics from the perspective of diﬀerent receptive ﬁelds.
Furthermore, low-frequency information from diﬀerent receptive ﬁelds

3. Method
Structure of SFE-Net is shown in Fig. 2. The proposed SFE-Net is designed to mitigate noise interference in event-based data while comprehensively extracting discriminative person semantics from sparse event
streams. It integrates two innovative mechanisms: (1) the Multi-grain
Spectrum Attention Mechanism (MSAM), which ﬁlters noise and enhances semantic representations of person identities, and (2) the Consecutive Patch Dropout Module (CPDM), which forces the model to focus
uniformly on all discriminative regions, thereby improving the robustness of person descriptors.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 195 (2026) 108206

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Stochastic style perturbation modelling for visible-Infrared person
re-Identiﬁcation with severely modality imbalance
Haojie Liu a,b , Zhiyong Li a , Jianyang Gu a , Mingyu Wang a , Q. M Jonathan Wu c , Wei Jiang

b,∗

a The College of Control Science and Engineering, Zhejiang University, Hangzhou, 310027, China
b

The School of Computer Science and Technology, Zhejiang University of Water Resources and Electric Power, Hangzhou, 310018, China

c The department of Electrical and Computer Engineering, University of Windsor, Canada

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Data imbalance problems
Cross-modality learning

a b s t r a c t
In this paper, we address the challenge of data imbalance in visible-infrared person re-identiﬁcation (VI-ReID).
Previous studies often presuppose a uniform distribution of training data across various modalities, however,
due to constraints such as device limitations, privacy concerns, and operational conditions, gathering nightime
infrared modality data can be prohibitively expensive or even impossible. Consequently, the limited infrared (IR)
modality data tend to be overshadowed by the more plentiful visible (RGB) modality data during the training,
particularly in scenarios marked by signiﬁcant imbalance. To counter this issue, we introduce the Co-Modality
Balance Learning (CMBL) framework, designed to recalibrate the balance in cross-modality learning and enhance
the extraction of discriminative features. Initially, we design a Stochastic Style Perturbation (SSP) module that
dynamically generates IR modality samples within the deep feature space to emulate the characteristics of a
balanced dataset. Subsequently, we develop a cross-distribution alignment loss, which enables a reﬁned optimization of sparse modality features to improve their accuracy and robustness. Additionally, we propose the
novel Class-Aware Contrast Similarity Learning (CACS) strategy, which capitalizes on latent feature consistency
to boost intra-class compactness and inter-class separation. Our extensive empirical evaluations and ablation
studies on two publicly available cross-modality datasets under imbalanced conditions underscore the eﬃcacy
of our approach, showcasing its ability to adeptly navigate the complexities of data imbalance in VI-ReID.

1. Introduction
Person Re-identiﬁcation (ReID) aims to identify a query person from
non-overlapping camera networks (Leng et al., 2020; Liu et al., 2021a;
Ye et al., 2021b). Existing methods mainly exploit the appearance information captured by visible light cameras to associate the person
identities (Cheng et al., 2024; Huang et al., 2024). However, under
poor illumination, visible cameras cannot provide suﬃcient information, resulting in ineﬀective feature recognition. Wu et al. (2017) propose to employ infrared cameras alongside visible cameras to supplement the low-light information, thus giving rise to the Visible-Infrared
Re-Identiﬁcation (VI-ReID) task.
For VI-ReID, previous cross-modality methods have endeavored to
bridge the gap between modalities through the use of image generation techniques (Liu et al., 2025, 2023, 2022c; Pang et al., 2024; Wang
et al., 2019b; Zhong et al., 2021) and invariant feature learning strategies (Chen et al., 2021; Hu et al., 2022; Huang et al., 2022; Liu et al.,
2020), achieving remarkable results on public benchmarks. However, a
fundamental assumption underpinning the success of these networks is

the completeness of modality data. In practice, obtaining infrared images of people poses signiﬁcant challenges due to the limited number
of pedestrians active at night and the high cost associated with infrared
cameras. As a result, it is common to encounter scenarios where RGB
modality data is signiﬁcantly more abundant than in infrared modality data, challenging the eﬀectiveness of cross-modality approaches in
practical applications. To this end, we propose to address the data imbalance problem in this paper where certain infrared (IR) modality samples
are absent, whereas visible (RGB) modality data remains accessible. In
this case, the dataset contains signiﬁcantly fewer IR samples compared
to the number of RGB samples. This discrepancy in sample sizes not
only hampers the network’s ability to learn from IR data eﬀectively but
also reﬂects a broader issue in balancing the representation of diﬀerent
modalities within a dataset. A common strategy to address data imbalance involves adjusting the dataset’s balance through resampling at the
data level. This approach generally employs two classical resampling
techniques: undersampling (Gu et al., 2021; Yang & Ismail, 2022; Yulong et al., 2022) and oversampling (Chawla et al., 2002; Wang et al.,
2013; Yang et al., 2020). Undersampling is designed to restore balance

∗ Corresponding author.

E-mail address: jiangwei_zju@zju.edu.cn (W. Jiang).
https://doi.org/10.1016/j.neunet.2025.108206
Received 28 December 2024; Received in revised form 20 May 2025; Accepted 9 October 2025
Available online 14 October 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 195 (2026) 108206

H. Liu et al.

∙ We conduct comprehensive experiments to evaluate the eﬀectiveness of our method in the problems of imbalance VI-ReID. The experimental results demonstrate that our method surpasses the state-of-thearts by a large margin.
2. Related works
Visible-Infrared Re-ID Methods. Visible-infrared person reidentiﬁcation (VI-ReID) is proposed to improve the performance of
surveillance systems in low-light conditions during nighttime (Kim et al.,
2023; Wu et al., 2017; Ye et al., 2021b). Early works attempted to mitigate the cross-modality discrepancy at image level via generative adversarial networks (GANs) (Liu et al., 2023; Qi et al., 2024; Wang et al.,
2019a; Zhong et al., 2021). However, due to a lack of pair cross-modality
samples, GAN-based methods always suﬀer from uncontrollable noise
and ambiguity in identity information. Another line of shared feature
learning approaches (Du & Zhang, 2023; Feng et al., 2023; Liu et al.,
2021b; Ye et al., 2020a, 2018; Zhang et al., 2022a; Zhao et al., 2022)
aim to design various network structures or loss functions to bridge the
modality gap. One typical design choice is the utilization of two-stream
structures (Liu et al., 2025, 2021b; Ye et al., 2020a; Yu et al., 2025),
which involves the incorporation of modality-speciﬁc shallow layers and
shared deeper layers to learn rich feature representation. However, regardless of whether it is feature learning methods or GAN-based methods, their primary focus is on addressing the issue of modality gap, overlooking the challenge of data imbalance modality problem in VI-ReID.
In recent years, researchers have focused on alleviating the modality gap between infrared and visible light in video sequences. The pioneering work by Lin et al. (2022) introduced HITSZ-VCM, the ﬁrst
video-based visible-infrared person re-identiﬁcation dataset, along with
an adversarial learning framework for extracting modality-invariant features. While eﬀective in certain scenarios, this approach was found to
inadvertently discard potentially valuable modality-speciﬁc characteristics. To overcome this limitation, Li et al. (2023) developed the innovative IBAN (Intermediary-Based Alignment Network) model, which
employs anaglyph representations of pedestrian images as an intermediate modality. This architecture not only preserves distinctive modalityindependent features but also incorporates a bidirectional spatiotemporal aggregation module to eﬀectively exploit the rich spatial and temporal information inherent in video sequences.
Imbalanced Modality Learning. There exists two types of imbalanced scenario in multi-modal learning realm, eg. data imbalance and
optimization imbalance. To address the data imbalance problem, previous researchers introduce the re-sampling (Cui et al., 2019; Huang et al.,
2016; Wang et al., 2017) and re-weighting (Buda et al., 2018; Byrd &
Lipton, 2019; Cao et al., 2019) strategies to balance optimization proportion during the training process. More recently, MAUM (Liu et al.,
2022a) proposes to learn unidirectional metrics to adjust the modalityspeciﬁc augmentation against modality imbalance, which achieves the
satisfactory results. For another issue of optimization imbalance, Wang
et al. (2020) introduced gradient blending strategy to achieve optimal
blending of modalities. Du et al. (2021) exploited knowledge distillation
technique to distill knowledge from a well-trained uni-modal model.
These methods have potential to improve performance, however, they
necessitate the introduction of additional modules, complicating the
training process. In this paper, we tackle this issue from the feature normalization standpoint, addressing the imbalance problem by adaptively
perturbing features without the need for extra modules.

Fig. 1. Previous data-level oversampling methods face several issues: (a) The
presence of noise samples in the minority class. (b) The minority class spans
multiple subspaces. (c) The minority class exhibits high distribution density.

by eliminating some samples from the majority class, whereas oversampling aims to increase the dataset’s diversity by replicating existing minority class samples or creating new synthetic ones. Typically, synthetic
samples are produced by selecting a minority class sample at random
and performing an interpolation among its neighbors. However, this
straightforward application of local linear interpolation could lead to
generated samples deviating from the original data distribution. Additionally, as depicted in Fig. 1, the eﬀectiveness of basic data-level oversampling methods is signiﬁcantly aﬀected by the sample distribution,
constraining their utility in novel scenarios.
In this research, we extend the use of oversampling techniques into
cross-modality data imbalance scenario. Unlike previous approaches
that focus on data-level oversampling, we propose Stochastic Style Perturbation (SSP) to generate novel sparse modality samples at the feature
level. Our main idea is treating the infrared modality feature statistic as
a undeterministic point measured from the feature to synthetize novel
sparse modality instances. Speciﬁcally, we estimate the new style information range based on the variances of the mini-batch statistics in
an eﬃcient non-parametric manner. Subsequently, variants of feature
statistics are randomly sampled from this estimated range and utilized
to replace the original deterministic values. Due to the generated feature
statistics with diverse possibilities, the models can be trained on a more
varied distribution of infrared modality data and encode more robust
modality-balanced features.
In addition, to minimize modality discrepancies across both
modality-shared and SSP-enhanced features, we propose a crossdistribution alignment loss to optimize the distribution distance relationships in the modality level. Furthermore, we introduce the ClassAware Contrast Similarity (CACS) learning approach to regulate feature
relationships in the instance level. CACS calculates the aggregated class
centers for both visible and infrared modalities to reﬁne the similarity relationships among diﬀerent identities, which signiﬁcantly enhances the
angular discriminability of features and promote the learning of underlying interaction between diﬀerent modalities. Extensive experiments
were conducted to validate the eﬀectiveness of the proposed method
(Fig. 2).
The main contributions can be summarized as:
∙ We reveal a new problem for VI-ReID, termed imbalance VI-ReID.
In contrast to existing VI-ReID studies which only consider the modality
discrepancy issue, we consdier to tackle the scenario where data from
one modality is severely deﬁcient.
∙ To allow the neural network to work with highly imbalanced data,
we propose the SSP module that can dynamically estimate the potential
of instances within the deep feature space without any modiﬁcation on
its architecture, thus are applicable to a wide range of deep learning
models.
∙ To promote the balanced representation learning, we introduce a
cross-distribution alignment loss to optimize the distance relationships
in the global distribution level. In addition, we propose a novel CACS
learning paradigm in the class-instance level to explore the discriminative feature relationships.

3. Method
3.1. Stochastic style perturbation
3.1.1. Background
The conceptually approach to addressing imbalanced data through
oversampling involves randomly duplicating existing instances until
2

Neural Networks 195 (2026) 108206

H. Liu et al.

Fig. 2. This is a high-level overview of our CMBL framework, which uses a three-stream network for multi-modal feature encoding. SSP features are produced
from the IR features following conv layer1 in the IR stream subnetwork. Parameters of conv layers 2-4 in the three-stream network are shared between RGB and IR
streams, while being independent from the oversampling stream. The cross-distribution alignment module aligns heterogeneous features using MMD distance at the
global distribution level. The Class-Aware Contrast Similarity Learning (CACS) module enhances feature discrimination through intra- and inter-class similarity at a
nuanced identity level. Best viewed in color.

balanced class distributions are achieved. However, this method often
results in the minority class distribution being overly concentrated in
the areas where the original observations were located. Consequently,
models trained on data altered in this manner are at a higher risk of
overﬁtting. To mitigate this issue, data-level neighborhood-based methods (Chawla et al., 2002; Cui et al., 2019; Huang et al., 2016; Wang
et al., 2017) have been developed, among which the Synthetic Minority
Over-sampling Technique (SMOTE) (Chawla et al., 2002) is one of the
most prominent. The basic idea of SMOTE (Chawla et al., 2002) is to
identify the nearest neighbors within the same class for a given minority instance and then synthesizes new samples in the space between the
target instance and one of its neighbors. Speciﬁcally, suppose we have
a minority class sample (𝑥) and its (𝑘) nearest neighbors (𝑥1 , 𝑥2 , … , 𝑥𝑘 ).
The formula for generating a new synthetic sample by SMOTE (Chawla
et al., 2002) is as follows:
𝑥𝑛𝑒𝑤 = (𝑥 + 𝑟𝑎𝑛𝑑(0, 1) × (𝑥𝑖 − 𝑥)),

Fig. 3. Visualization in multi-modal scenarios for SMOTE (Chawla et al., 2002):
pre-synthesis (left) vs. post-synthesis (right). Due to the large modality discrepancy, the generated new samples always deviate from the original modality distribution (green dots). Best view in color.

age by mean and standard deviation 𝜇(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , 𝜎(𝑓 𝑡 ) ∈ ℝ𝐵×𝐶 , as:

(1)

1 ∑∑ 𝑡
𝑓
,
𝐻𝑊 ℎ=1 𝑤=1 𝑏,𝑐,ℎ,𝑤
√
√
)2
√ 1 𝐻
∑ 𝑊
∑(
𝑡
𝑡
𝜎(𝑓 ) = √
𝑓𝑏,𝑐,ℎ,𝑤
− 𝜇(𝑓 𝑡 ) .
𝐻𝑊 ℎ=1 𝑤=1
𝐻

𝑊

𝜇(𝑓 𝑡 ) =

where 𝑟𝑎𝑛𝑑(0, 1) ranges between [0, 1] and is used to control the position
of the synthetic sample. Repeating the steps allows for the creation of
synthetic samples in quantities tailored to achieve dataset balance.
The foundational premise of SMOTE (Chawla et al., 2002) is that the
spaces between the nearest minority neighbors are viable for generating new instances. However, this assumption encounters challenges in
multi-modal scene. As illustrated in Fig. 3, due to large modality discrepancy, the nearest minority neighborhood is spread apart, leading to
the generation of synthetic samples that deviate from the original distribution. Moreover, given the varying degrees of modality discrepancy, it
is unclear what the optimal size of the neighborhood 𝑘 should be.

(2)

(3)

Here, 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) encapsulate the statistical characteristics of the
image, capturing its inherent style.
The normalized features that encode the semantic information are
represented as:
𝑓 =𝛾⊙

𝑓 𝑡 − 𝜇(𝑓 𝑡 )
+ 𝛽,
𝜎(𝑓 𝑡 )

(4)

where 𝛾 and 𝛽 represent the channel-wise aﬃne parameters. According
to the work Huang and Belongie (2017), Tang et al. (2021), modifying
the feature statistics (𝛾, 𝛽) can change the feature style of an image.
Then, to obtain reasonable values of 𝛾 and 𝛽, we ﬁrstly utilize the variance of 𝜇(𝑓 𝑡 ) and 𝜎(𝑓 𝑡 ) to estimate meaningful style vectors, denoted
as:

3.1.2. Feature-level stochastic style perturbation
Rather than depending on nearest neighbors for generating new samples at the data level, in this paper we explores the potential of a featurebased approach. Our objective diverges from SMOTE techniques: we
aim to directly generate IR features within the deep feature space. By
undertaking this approach, we aspire to transform the style of minority
modality samples while preserving their semantic information, ensuring
that the generated features do not stray from the original distribution.
Speciﬁcally, as shown in Fig. 4, given infrared features 𝑓 𝑡 ∈
ℝ𝐵×𝐶×𝐻×𝑊 where 𝐵, 𝐶, 𝐻, 𝑊 denoting the batchsize, dimension of
channel, height and width, respectively, we represents the style of im-

𝜎(𝜇(𝑓 𝑡 ))2 =

𝐵
𝐵
1 ∑
1 ∑
(𝜇(𝑓 𝑡 ) −
𝜇(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1

𝐵
𝐵
1 ∑
1 ∑
𝜎(𝜎(𝑓 )) =
(𝜎(𝑓 𝑡 ) −
𝜎(𝑓 𝑡 ))2 ,
𝐵 𝑏=1
𝐵 𝑏=1
𝑡 2

3

(5)


codex
前四篇看完了，已经能看到几个反复出现的套路：不是单纯加模块，而是先把“数据形态不完整、语义稀疏、模态不平衡、3D 表征太稀疏”定义成新问题，再给一个可检验的机制。现在继续抽后五篇。
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
/bin/zsh -lc "pdftotext -l 3 'Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
arXiv:2509.16635v2 [cs.CV] 30 May 2026

Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification
Xulin Li1,2 , Yan Lu3 , Bin Liu1,2 ,* Jiaze Li1,2 , Qinhong Yang1,2 ,
Tao Gong1,2 , Qi Chu1,2 , Mang Ye4 , Nenghai Yu1,2
1
School of Cyber Science and Technology, University of Science and Technology of China
2
Anhui Province Key Laboratory of Digital Security
3
The Chinese University of Hong Kong
4
School of Computer Science, Wuhan University, China
lxlkw@mail.ustc.edu.cn, yanlu@cuhk.edu.hk, flowice@ustc.edu.cn, jz li@mail.ustc.edu.cn,
qhyang233@mail.ustc.edu.cn, {tgong,qchu}@ustc.edu.cn, yemang@whu.edu.cn, ynh@ustc.edu.cn

Abstract
In real applications, person re-identification (ReID) is
expected to retrieve the target person at any time, including both daytime and nighttime, ranging from short-term to
long-term. However, existing ReID tasks and datasets cannot meet this requirement, as they are constrained by available time and only provide training and evaluation for specific scenarios. Therefore, we investigate a new task called
Anytime Person Re-identification (AT-ReID), which aims to
achieve effective retrieval in multiple scenarios based on
variations in time. To address the AT-ReID problem, we collect the first large-scale dataset, AT-USTC, which contains
403k images of individuals wearing multiple clothes captured by RGB and IR cameras. Our data collection spans
21 months, and 270 volunteers were photographed on average 29.1 times across different dates or scenes, 4-15 times
more than current datasets, providing conditions for followup investigations in AT-ReID. Further, to tackle the new
challenge of multi-scenario retrieval, we propose a unified
model named Uni-AT, which comprises a multi-scenario
ReID (MS-ReID) framework for scenario-specific features
learning, a Mixture-of-Attribute-Experts (MoAE) module
to alleviate inter-scenario interference, and a Hierarchical Dynamic Weighting (HDW) strategy to ensure balanced
training across all scenarios. Extensive experiments show
that our model leads to satisfactory results and exhibits excellent generalization to all scenarios. Our dataset and
code are available at https://github.com/kw66/AT-ReID.

1. Introduction
Person re-identification (ReID) aims to retrieve specific
pedestrians with given query images. As illustrated in
* Corresponding author.

Scenario 1
Day

Scenario 2

Scenario 5

Scenario 6

Spring

Winter

Scenario 4
Night

Scenario 3

Figure 1. (a) AT-ReID aims to perform retrieval at any time, including both daytime and nighttime, ranging from short-term to
long-term.

Fig. 1, a robust ReID system is expected to retrieve a person
at any time, including daytime and nighttime, ranging from
short-term to long-term, thereby satisfying the requirements
of different surveillance scenarios. This puts more challenges on the ReID system because the capturing time of
the query image and the target image makes the task more
variable. For instance, if two images are captured during
daytime and nighttime, respectively, they will have different
modalities, and when there is a long time interval between
their capturing, the person’s appearance may change due to
alterations in clothing. Consequently, traditional ReID (TrReID) [58] may not perform effectively.
The researchers acknowledged this challenge and attempted to address these problems separately. They in-

troduced the Visible-Infrared Cross-Modality ReID (CMReID) [47] to address the issue of searching between daytime RGB images and nighttime infrared (IR) images, and
the Long-Term Cloth-Changing ReID (CC-ReID) [50] was
proposed to handle long-term retrieval in which pedestrians
change their clothes. However, existing methods designed
for these specific tasks were only able to achieve success
in one of them and incapable of retrieving targets at any
time simultaneously. This situation primarily arises from
the absence of a long-term visible-infrared dataset covering all scenarios in Fig. 2 (a), which should encompass
diverse variations in clothing and modality for each individual. The deficiency in intra-identity diversity of modalities and clothing in current ReID datasets has led to research gaps, especially in Nighttime Long-term (NT-LT)
and All-day Long-term (AD-LT) scenarios. Another issue
arises from the poor generalization of task-specific methods
in non-target scenarios. This is attributed to the differing
learning objectives across different scenarios. For instance,
prior research [29, 35] has indicated that RGB-specific cues
and clothing information are harmful to the All-day Shortterm (AD-ST, CM-ReID) and Daytime Long-term (DT-LT,
CC-ReID) scenarios, respectively, while they are crucial for
the Daytime Short-term (DT-ST, Tr-ReID) scenario.
To meet the requirements of retrieving persons at any
time, we investigate a new task called Anytime Person Reidentification (AT-ReID) and propose to focus on its exploration from dataset to model level, as depicted in Fig. 2 (b).
We collect the first corresponding large-scale dataset named
AT-USTC, which contains 403k images of 270 volunteers
and covers all six scenarios in AT-ReID. Our data collection spans 21 months, covering both day and night periods
across the seasons of spring, summer, and winter. We focus
on simultaneously providing a greater variety of clothing
and more RGB and IR cameras for each person. Through
efforts to expand in terms of capture dates, time periods, and
scene variations, our AT-USTC provides a broader intraidentity diversity and more comprehensive AT-ReID cases
than previous datasets.
To tackle the new challenge of multi-scenario retrieval
in AT-ReID, we further propose a unified model named
Uni-AT to effectively handle all scenarios. Given that the
AT-ReID encompasses six different scenarios, the information shared among all scenarios becomes limited, and learning a unified representation for all scenarios is sub-optimal.
Therefore, we propose a novel Multi-Scenario ReID (MSReID) framework with multiple classification tokens and
a scenario-aware identity loss to facilitate effective learning of specific features for each scenario. To achieve better discriminative feature extraction for different scenarios,
we improve MS-ReID at both the model structure and optimization levels. Specifically, we propose a Mixture-ofAttribute-Experts (MoAE) module, which builds the ex-

(a)

Daytime
RGB camera
Scenario : DT-ST

All-day
RGB
IR

Nighttime
IR camera

Scenario : AD-ST

Tr-ReID

CM-ReID

Scenario : NT-ST
New

Scenario : DT-LT

Scenario : AD-LT
New

Scenario : NT-LT
New

Shortterm
Same
clothes
CC-ReID

Longterm
Clothes
changes

(b)

Uni-AT method

AT-USTC dataset
DT-ST

Multi-scenario learning
Cover all scenarios

DT-LT AD-ST AD-LT NT-ST

AT-ReID task

NT-LT

Anytime retrieval

Tr-ReIDCC-ReIDCM-ReID absent absent absent

Figure 2. (a) Six non-overlapping scenarios based on variations in
time. AT-ReID aims to perform retrieval in all of these scenarios.
(b) Our solution of AT-ReID from the dataset to the model level.

pert network and assigns different experts to address distinct scenarios, thus enabling the model to alleviate interference between scenarios. Additionally, we define the attribute layer as the basic cell shared among experts with
similar scenario attributes, e.g., DT-related attribute layers
are shared among DT-LT and DT-ST experts. With this,
the model can benefit from multiple interrelated scenarios.
And we propose a Hierarchical Dynamic Weighting (HDW)
strategy, that tackles the AT-ReID training from the multitask learning view. It establishes all scenarios into several
tasks and balances the training for different tasks with a loss
weighting scheme. This method considers multiple relevant
tasks when computing weights, implicitly modeling the relationships between tasks and leading to better optimization
of the overall multi-scenario learning framework.
Our main contributions can be summarized as follows:
• We investigate a new task called AT-ReID, which aims
at enabling retrieval at any time moment and across different time intervals. We contribute for the first time a largescale dataset named AT-USTC to support the study of ATReID. Compared to existing datasets, AT-USTC stands out
for its long data collection period and the inclusion of both
RGB and IR camera footages, meeting the requirement of
AT-ReID. Importantly, our data collection has obtained the
consent of each volunteer.
• We propose a Uni-AT model to effectively handle all scenarios of AT-ReID. In Uni-AT, three components, a new
multi-scenario ReID framework, a Mixture-of-AttributeExperts module, and a Hierarchical Dynamic Weighting

training strategy are proposed to tackle the new challenges
of multi-scenario retrieval in AT-ReID tasks. Extensive experiments show that our model leads to satisfactory results
and exhibits excellent generalization to all scenarios.

Cam 15
Cam 10

Cam 16

Cam 9
Cam 1

Cam 11

Cam 12

2. Related Work

Cam 2

Cam 11
Cam 8

Cam 3
Cam 7

Person Re-Identification. Traditional ReID (Tr-ReID)
aims to achieve short-term pedestrian retrieval in the RGB
modality. The corresponding datasets, such as Market1501 [58], CUHK03 [21], and MSMT17 [46], focused
on providing more identities as well as more camera variations. Tr-ReID methods involve general pedestrian retrieval
techniques, such as the design of more robust backbone networks [13, 30, 54], effective ReID loss functions [40], and
the utilization of part-level features [39] to achieve discriminative representations of pedestrians.
Visible-Infrared Cross-Modality ReID (CM-ReID) aims
to achieve short-term pedestrian retrieval between the RGB
and the infrared (IR) modalities. The corresponding
datasets, such as SYSU-MM01 [47], RegDB [34], and
LLCM [57], focused on providing more RGB and IR cameras. Some CM-ReID methods [8, 49] aimed to project features from different modalities into the same feature space,
while others [7, 22, 29, 52] aimed to learn cross-modality
relationships.
Long-term Cloth-changing ReID (CC-ReID) aims to
achieve long-term pedestrian retrieval in the RGB modality. The corresponding datasets, such as PRCC [50],
LTCC [35], and DeepChange [48], focused on providing
clothing variations for each person. Some CC-ReID methods [4, 11, 17, 26] introduced additional data such as contour, key points, human parsing, and 3D shape for model
training, while others [9, 12, 23, 51] utilized RGB images
only to learn robust clothing-irrelevant feature.
The aforementioned tasks and datasets can only cover a
portion of the AT-ReID scenarios. In addition, some unified methods [3, 5, 14, 19, 42, 59] focus on multiple ReID
tasks, such as text/sketch-to-RGB ReID, clothes template
based CC-ReID, and occlusion ReID, as well as humancentric tasks, such as human parsing, pose estimation, and
pedestrian detection. Our research is distinct from previous
methods as it is the first to focus on the availability of ReID
at any time and proposes a relevant dataset and method to
bridge the gap between existing research and AT-ReID.
Multi-Task Learning. Multi-task learning (MTL) refers
to building a model that can handle multiple distinct
tasks [2, 33]. By sharing parameters between tasks, MTL
methods achieve efficient memory and data utilization and
expect to derive benefits from multiple related tasks. In ATReID, various input modalities and learning objectives are
present in different scenarios. Retrieval in each scenario can
be considered an individual ReID task, and it is promising

Cam 4
Cam 5

Cam 6

Cam 10
RGB
Camera
IR
Camera

Figure 3. The plan of the camera layout for collecting data.

that employing MTL methods can improve the overall efficacy of the model across all scenarios.
Some MTL methods focused on network architecture [28, 32, 45, 61] to achieve more effective parameter
sharing. Recently, some effective approaches [1, 31, 36, 41,
61] are to utilize the Mixture-of-Experts (MoE) [16] model
that employs multiple expert sub-networks to tackle multitask learning. Compared to these MoE methods, our MoAE
constructs scenario experts in a more flexible sharing manner, making the model benefit from multiple interrelated
scenarios. Other methods focused on MTL optimization,
such as manipulating gradient [25, 27, 56] and adjusting
the loss weight by task difficulty, training speed, and priority [3, 10, 18, 28]. Our HDW method groups tasks based
on their attributes and applies hierarchical dynamic weighting to the loss of each task, achieving a more effective task
balance.

3. AT-USTC Dataset
Dataset Description. AT-USTC is the first AT-ReID
benchmark that includes 403,599 (199,803 RGB and
203,796 IR) images of 270 identities and 710 sets of different clothing captured by 16 cameras. As shown in Fig. 3,
we deployed 8 RGB and 8 IR cameras across 16 nonoverlapping locations, comprising 5 indoor and 11 outdoor
scenes. We filmed videos spans 21 months including spring,
summer, and winter, with temperatures ranging from -3◦ C
to 33◦ C to cover a wider range of clothing types. Each individual in our training set has 2-14 outfits with an average of 3.6, which facilitates retrieval in long-term scenarios.
Due to the variations in both modality and clothing in ATUSTC, the process of capturing and annotating the data is
more time-consuming compared to other datasets. We made
considerable effort to provide annotations, including labels
for person, camera, and clothing.
Privacy Protection. Following the established ReID
datasets [47, 50], we made efforts for privacy protection in
five aspects: 1) Data collection was authorized by the relevant authorities, involving the deployment of cameras and
image capture. 2) The individuals we photographed did not


 succeeded in 0ms:
9852

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

TIENet: A Tri-Interaction Enhancement Network
for Multimodal Person Reidentification
Xi Yang , Senior Member, IEEE, Wenjiao Dong, De Cheng , Nannan Wang , Senior Member, IEEE,
and Xinbo Gao , Fellow, IEEE

Abstract—Multimodal person reidentification (ReID), which
aims to learn modality-complementary information by utilizing
multimodal images simultaneously for person retrieval, is crucial for achieving all-time and all-weather monitoring. Existing
methods try to address this issue through modality fusion to
absorb complementary information. However, most of these
methods are limited to the spatial domain only and usually
overlook the intra-/intermodal interactions during feature fusion,
resulting in insufficient learning of modality-specific and complementary information. To address these issues, we propose a
tri-interaction enhancement network (TIENet), which contains
three modules: spatial-frequency interaction (SFI), intermodal
mask interaction (IMMI), and intramodal feature fusion (IMFF).
Specifically, the SFI boosts the modality-specific representation
by integrating the amplitude-guided attention mechanism into
the phase space, combined with spatial-domain convolution to
achieve fine-grained information learning. Meanwhile, the IMMI
enhances the richness of the feature descriptors by embedding
the intermodal relationships to preserve complementary information. Finally, the IMFF module considers the structure of the
human body and integrates intramodal contextual information.
Extensive experimental results demonstrate the effectiveness of
our method, achieving superior performances on RGBNT201 and
MARKET1501 RGBNT datasets.
Index Terms—Inter-/intramodal interaction, multimodal person reidentification (ReID), spatial-frequency interaction (SFI).

I. I NTRODUCTION

P

ERSON reidentification (ReID) is a technique for retrieving and identifying the same person under different
cameras, which has a wide range of applications in the fields
of security, surveillance, and intelligent transport. Different
Received 28 July 2024; revised 28 October 2024; accepted 18 February 2025. Date of publication 19 March 2025; date of current version
4 June 2025. This work was supported in part by the National Natural
Science Foundation of China under Grant 62372348, Grant 62176198, Grant
U22A2096, and Grant 62036007; in part by the Key Research and Development Program of Shaanxi under Grant 2024GX-ZDCYL-02-10; in part by
Shaanxi Outstanding Youth Science Fund Project under Grant 2023-JC-JQ53; in part by Shaanxi Province Core Technology Research and Development
Project under Grant 2024QY2-GJHX-11; and in part by the Fundamental Research Funds for the Central Universities under Grant QTZX23042.
(Corresponding author: De Cheng.)
Xi Yang, Wenjiao Dong, De Cheng, and Nannan Wang are with
the State Key Laboratory of Integrated Services Networks, School of
Telecommunications Engineering, Xidian University, Xi’an 710071, China
(e-mail: yangx@xidian.edu.cn; dwj@stu.xidian.edu.cn; dcheng@xidian.edu.
cn; nnwang@xidian.edu.cn).
Xinbo Gao is with Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing
400065, China, and also with the School of Electronic Engineering,
Xidian University, Xi’an 710071, China (e-mail: gaoxb@cqupt.edu.cn;
xbgao@mail.xidian.edu.cn).
Digital Object Identifier 10.1109/TNNLS.2025.3544679

from the traditional person ReID task, i.e., ReID based on
single visible image [1], [2], [3], [4], [5], [6], the multimodal
person ReID aims to perform robust ReID by introducing
multiple complementary modality images for each person
example. Such multimodal person ReID helps to deal with
more complex illumination scenarios, which greatly assists the
traditional ReID task and addresses its application limitations.
Besides, the popularity of diverse kinds of cameras (i.e.,
various infrared and RGB cameras) makes multimodal person
ReID possible and has attracted more interest recently. Therefore, the multimodal person ReID will have great potential
applications in the field of intelligent surveillance system, due
to its strong complementary benefits from different modalities.
As shown in Fig. 1(a), it illustrates examples from
RGBNT201 dataset [7], which includes three-modality images
for each example, i.e., visible (RGB), near-infrared (NI), and
thermal-infrared (TI) modalities. Obviously, the RGB images
can provide rich color and texture information under favorable
lighting condition, the NI images can provide clear edge
information without being affected by the light, while the TI
images can discriminate persons from surroundings by using
their temperatures without being affected by the complicated
environment [7]. Therefore, how to make full use of the
complementary information of different modalities is the key
to multimodal person ReID.
It is very different from the traditional cross-modality
person ReID [8], [9], [10], [11], [12], [13], which focuses
on reducing the modality discrepancy and learning modalityshared features. In contrast, the multimodal person ReID
focuses on effective modality fusion to absorb complementary
information from the multimodality data, and further improves
the person discriminability.
Existing fusion-based multimodal feature learning can be
roughly divided into the following three categories: imagelevel early fusion [14], [15], feature-level late fusion [16],
[17], which merges different modality features to obtain
enhanced representations, and their combinations with different strategies [18], [19]. However, these methods are limited
to the spatial domain only, and usually ignore the intermodal
interactions during feature fusion procedure. It will cause
insufficient modality-specific and complementary information
learning, and result in some redundancy or biased information
in the fused features. To address these issues, we propose
a tri-interaction enhancement network (TIENet) to boost the
modality-specific information through the spatial-frequency
interaction (SFI) module, while absorbing complementary

2162-237X © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

Fig. 1. Frequency information for multimodal person ReID task. (a) Key
to multimodal person ReID task. (b) Fourier transforms of different modal
images. (I) Spatial-domain form of different modal pedestrian images. (II)
Different modal amplitude components. (III) Different modal phase components. (IV) Reconstructed image with phase only (phase-r). (V) Reconstructed
image with amplitude only (amplitude-r).

information between intra-/intermodalities by intermodal mask
interaction (IMMI) and intramodal feature fusion (IMFF)
module.
The frequency relationship between RGB and NI images
has been explored in [20]; in order to further explore the
relationship between RGB, NI, and TI frequencies, as shown
in Fig. 1(b), we recover the pedestrian images using only phase
(IV) and amplitude (V), respectively. It can be visualized that
the phase mainly conveys global structural information, while
amplitude could convey specific content information. To show
the difference between phase and amplitude of different modal
images more intuitively at the feature level, we take the conv1,
bn1, relu, and maxpool layers of the pretrained ResNet50 on
ImageNet as the backbone and take seven random ID image
sets to extract shallow features and draw t-SNE maps. The
following should be noted.
1) Shallow features express the basic information of the
multimodal image, while deep features reflect the
advanced semantic information of the image. Thus, we
only use shallow features to show the direct relationship
between the phase and amplitude of different modal
images.
2) The feature extraction stage is without any loss function
guidance and training process. Therefore, we can mine
the most essential properties of multimodal images in
the frequency domain. From Fig. 2(a-1) and (a-2), we
can see that the amplitude of different modalities is more
discriminative than the phase, which is consistent with
our image visualization results.
Therefore, a straightforward approach is to use an attention mechanism on amplitude in the frequency domain to
enhance the phase, as illustrated in Fig. 2(b). Compared
to Fig. 2(a), the enhanced feature modalities exhibit clearer
boundaries, indicating that amplitude-guided phase can capture
more modality-specific information in feature representation.
Additionally, features from the same ID become more compact, suggesting that enhanced features can learn ID-specific
distinguishing features.

9853

In conclusion, we deem that different modalities of the same
pedestrian can be more differentiated by amplitude than phase
in the frequency domain, as the same person usually shares
the same global structure in the phase domain, and that the
amplitudes of different pedestrians are more discriminative
because the content information of the amplitude response is
more valuable than the structural information in the pedestrian
classification task. Based on this, we propose the SFI module,
which consists of two data flows. The frequency-domain
flow implements the amplitude-guided attention mechanism
to instruct phase learning, ultimately resulting in enhanced
feature representations. The spatial-domain flow utilizes traditional convolutional neural network (CNN)-based spatial
feature learning to preserve local detailed information, and the
interaction between the two data flows achieves fine-grained
feature enhancement.
Moreover, different modalities can convey different information that complements each other in pedestrian retrieval.
In order to absorb complementary information from different
modalities, intermodal interaction is essential. Due to the existence of different relationships between different modalities,
fixed interactions will destroy the intermodal relationships
and cause inconsistent information. Therefore, we introduce
the IMMI module in the middle of modality-specific and
shared networks to guide the process of feature interaction
by learning the intermodal correlation matrix. Specifically, the
IMMI module forms a mask that adaptively reflects intermodal
correlations and complementarities by computing intermodal
similarities. Therefore, this approach ensures the consistency
of information after interaction and reduces information redundancy compared to traditional intermodal interaction.
Meanwhile, the human body structure provides rich spatial
information that can enhance the robustness of the features.
By exploiting the structural relationships of the human body,
the effects of lighting conditions, pose, and occlusion can
be alleviated and more effective pedestrian features can be
extracted under unfavorable conditions. In addition, the human
body structure is independent of the modality since the pose of
each pedestrian remains consistent across modalities. Based on
this, at the end of the network, we introduce the IMFF module
based on location attention, which takes into account the connections between human body parts. Thereby, the local feature
representation contains not only its own attribute information,
but also the contextual information of the neighboring local
features. Finally, it fully integrates intermodal features while
fusing intramodal contextual information. Also considering
the distribution of intra- and intermodal features, we utilize a
modal aggregation-separation (MAS) loss function to enhance
feature complementarity and robustness.
In summary, our contributions are as follows.
1) In the designed TIENet, we propose an SFI module, which enhances modality-specific features through
amplitude-guided phase learning to capture global information in the frequency domain, and simultaneously
preserves detailed information through convolution in
the spatial domain.
2) We propose to enhance the richness of the feature
descriptors, by taking into account the intra-/intermodal

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

9854

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

Fig. 2. Distribution of features in different states. (a) Spatial-domain original image feature distribution. (a-1) Frequency-domain amplitude feature distribution.
(a-2) Frequency-domain phase feature distribution. (b) Spatial-domain enhanced feature distribution. The amplitude in (a-1) shows a clearer modal difference
compared to the phase in (a-2) and the enhanced features (b) contain more modality-specific information compared to the original features (a).

structure relationships to achieve the fusion of complementary information in different modalities.
3) Extensive experimental results demonstrate the effectiveness of the proposed method, which achieves superior
performances to the state-of-the-art methods by a large
margin on RGBNT201 and MARKET1501 RGBNT
datasets.
II. R ELATED W ORK
A. Single-Modal Person ReID
Single-modal person ReID refers to conducting a person
search within the same modality, with a key focus on extracting distinctive pedestrian features, predominantly utilizing
RGB images due to their widespread availability. In the deep
learning-based approach, Chang et al. [21] proposed a multilevel factorization net (MLFN) architecture for factorizing the
visual appearance of individuals into latent discriminative factors at multiple semantic levels. HACNN [22] integrated soft
pixel and hard regional attention mechanisms within CNNs to
enhance feature representation. Despite this, factors such as
occlusion, clothes changes, and background information limit
the further upgrading of model performance, generating the
special focus of some recent studies [23], [24], [25], [26].
However, due to the limited information conveyed by singlemodal images, it is not feasible to achieve all-time monitoring,
greatly restricting the application of person ReID.
B. Cross-Modal Person ReID
Cross-modal person ReID involves searching for pedestrians across two modalities; hence, the key is to narrow
the gap between modalities. Current major methods focus
on feature representation, metric learning, and generation
based on intermediate modalities. In terms of feature representation, the pioneering work by Wu et al. [8] addressed
this issue by proposing a single-stream network structure
with deep zero-padding to extract intermodal shared features.
Ye et al. [9] employed a twin network to extract features

and utilized a loss function to mitigate modal heterogeneity.
Wei et al. [12] and Zhang et al. [27] used local features
to capture fine-grained information and reduce intermodal
distance. Moreover, attention mechanisms are used to extract
discriminative modality-shared features from those personrelated regions in [28]. In metric learning, Feng et al. [29]
utilized Euclidean constraint to reduce the cross-modality
gaps. Zhu et al. [30] proposed a heterogeneous center loss that
extracts multimodal features through the twin-stream network
and uses the distance between the center of these modalities
as the loss to bring the feature of the same person closer
together. Liu et al. [31] designed an intercenter triplet loss
to ensure compact within-class features and discriminative
between-class features. Finally, in modal generation, Wang
et al. [32] introduced AlignGAN, converting RGB images
to infrared images with joint pixel-level and feature-level
alignment constraints to reduce modality gaps. To fully exploit
the complementary nature of the two modalities, Liu et al. [33]
utilized generative adversarial network (GAN) to generate a
high-quality missing image.
Despite the significant progress in cross-modal ReID techniques, the gap between modalities still exists, thus limiting
their accuracy and robustness. Therefore, it becomes crucial to
collaborate multimodal information simultaneously for pedestrian retrieval rather than cross-modal retrieval.
C. Multimodal Object ReID
Unlike cross-modal solutions to heterogeneous intermodal
issues, multimodal person ReID aims to fully utilize multisource information to improve pedestrian recognition accuracy.
Li et al. [35] proposed a weighted fusion approach using class
activation map (CAM) to combine different modalities. H-ViT,
presented by Pan et al. [36], integrates modal information as
tokens using Transformer architecture. They investigated the
impact of different fusion stages on model performance. Building upon H-ViT, Pan et al. [18] introduced a random hybrid
enhancement module for the image-level fusion of different modalities. However, they simply concatenate or linearly

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112341

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Texture-aware transformer with pose-patch mapping for occluded person
re-identiﬁcation
Dengwen Wang a , Guanyu Xing
a

b , Yanli Liu a,∗

College of Computer Science, Sichuan University, Chengdu, 610017, China

b School of Cyber Science and Engineering, Sichuan University, Chengdu, 610017, China

a r t i c l e

i n f o

Keywords:
Occluded person re-identiﬁcation
Pose-patch mapping
Texture aware decoder

a b s t r a c t
Occluded person re-identiﬁcation (re-ID) aims to retrieve the target person from occluded images captured by
diﬀerent cameras, where the challenges lie in identity loss caused by diﬀerent types of occlusion. To alleviate the
occlusion interference, some methods rely on external clues or generate more occlusion samples. However, these
methods fail to address the issues of pose misalignment under extreme occlusion and identity confusion caused
by non-target pedestrian occlusion. To solve these problems, we design a novel Texture-Aware Transformer
with Pose-Patch Mapping (TTPM), which does not require generating any occlusion samples. Speciﬁcally, a
Multi-patch Feature Encoder is proposed to encode discriminative features from inter patches and intra patches.
Afterwards, the Pose-Patch Mapping is designed to construct a positional mapping between poses and patches,
which highlights human patches and weakens the impact of occluded patches. Finally, to mitigate the non-target
pedestrian occlusion, a Texture-Aware Decoder is introduced to perceive texture features and leverage their
distinctiveness to enhance the representation of important regions. Extensive experiments show that our method
achieves state-of-the-art results on Occluded-Duke and Occluded-REID datasets.

1. Introduction
Person re-identiﬁcation (re-ID) aims to retrieve a target person from
multiple non-overlapping camera views. It provides critical technical
support for tracking suspects and ﬁnding lost people. Due to the impact
of imaging environments, the target person is often occluded by uncontrollable factors, including object occlusion (e.g., cars, billboards, and
trees) and non-target pedestrian occlusion, which often lead to significant loss of body information and increase the complexity of person
re-identiﬁcation. Therefore, re-identifying persons under complex occlusions in real-world scenarios poses a critical challenge for advancing
the practical applications of person re-identiﬁcation [1].
In recent years, researchers have designed various approaches to occluded person re-identiﬁcation from diﬀerent perspectives. These methods primarily fall into the following two categories: (1) Utilize extra
pre-trained pose estimators or human parsing models [2,3] to locate the
unoccluded regions of the human body. With additional semantic guidance, non-human features can be removed; (2) Generate more occlusion
samples through occlusion augmentation to enhance the perception of
occlusion, including randomly generating occlusion noise and manually
cropping occluding objects to paste onto the person images [4,5].

While the above methods perform well on holistic and objectoccluded person re-ID, it is still diﬃcult to maintain the same performance in cases of extreme object occlusion and non-target pedestrian
occlusion, as shown in Fig. 1(a). We attribute this performance degradation primarily to the following two reasons: (1) Under extreme object
occlusion, for some pose-based methods, owing to their mapping from
poses to global features, errors in local pose estimation can directly affect the eﬀectiveness of global features. In such cases, features highlighted by pose information usually introduce additional foreground
noise. (2) Under non-target pedestrian occlusion, the aforementioned
methods focus on separating occluded and non-occluded features, while
failing to fully capture distinctive and useful individual appearances
(e.g., backpacks, umbrellas, and clothes). As we know, when our human
eye re-identiﬁes a target person occluded by non-target pedestrians, the
perception of discriminative appearance often plays a decisive role in
the re-identiﬁcation process.
However, few studies simultaneously focuse on these two issues.
Some methods [6,7] directly use the output of pose estimation models as local masks without adaptive adjustment according to occlusion
conditions. For instance, Gao et al. [2] proposed a pose-guided visible part matching method, which estimates whether parts are occluded

∗ Corresponding author.

E-mail address: yanliliu@scu.edu.cn (Y. Liu).
https://doi.org/10.1016/j.patcog.2025.112341
Received 8 January 2025; Received in revised form 31 July 2025; Accepted 21 August 2025
Available online 24 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112341

D. Wang et al.

To implement these ideas, we propose a Texture-Aware Transformer
with Pose-Patch Mapping (TTPM) that relies solely on real data. It
is composed of three parts: a Multi-patch Feature Encoder (MFE), a
Pose-Patch Mapping (PPM), and a Texture-Aware Decoder (TAD). To
ensure that the patch features are rich in semantics while remaining
compact, MFE is proposed to encode features at two diﬀerent levels:
inter-patches and intra-patches. To eliminate occlusion semantics from
human features, PPM is developed to enhance pose-semantic awareness by calculating a similarity correlation matrix between patches and
poses, thereby weakening the representation of occluded patches. However, due to the presence of similar postures in non-target pedestrian
occlusions, PPM still cannot thoroughly eliminate the impact of pedestrian occlusion. To further address this issue, TAD uses the pose features
ﬁltered by PPM as constraints, and learns discriminative textures hierarchically through multi-head attention and cross-attention mechanisms.
Then, the discriminative textures are leveraged to enhance the eﬀective semantic representation of relevant regions, serving as a signiﬁcant
distinction for diﬀerent identities. As shown in Fig. 1(b), even under extreme object occlusion and non-target pedestrian occlusion, meaningful
regions still receive the appropriate attention.
In summary, the main contributions of this work are three-fold:

Fig. 1. Comparison between diﬀerent re-ID methods for two challenging cases:
extreme object occlusion and pedestrian occlusion. In Fig. (a), the (i) column
represents the original query image, the (ii) column shows the pose estimation
image, and the (iii) column denotes the human parsing image. In scenarios of
extreme object occlusion and non-target pedestrian occlusion, methods relying
solely on pose and human parsing inherently introduce occlusion noise, leading
to inaccurate feature representations. In contrast, Fig. (b) presents the attention
heatmaps of the proposed TTPM (on the right). Visualization results demonstrate
that TTPM is able to consistently focus on semantically discriminative regions,
even under these challenging occlusion cases.

•

A pose-patch mapping is proposed that constructs a direct mapping
from pose to patch, ensuring accurate alignment between the pose
and occluded pedestrians without any additional occlusion augmentation.
• A texture-aware decoder is designed that enhances the semantic representation of the eﬀective regions by hierarchically learning discriminative textures, serving as an important distinction for diﬀerent
identities.
• We conduct extensive experiments on six benchmark datasets across
three categories (i.e., occluded, holistic, and partial re-ID datasets)
and demonstrate that the proposed method outperforms 31 re-ID
methods. Especially compared to the latest method, MTIPE [11], it
shows improvements of mAP (+5.6 %) and Rank-1 (+7.0 %) on the
Occluded-Duke dataset.

through pose-guided attention and trains a visibility prediction model
in a self-supervised manner. Alternatively, some methods [8,9] utilize
customized attention mechanisms to extract specialized discriminative
semantics without pre-introducing human topological structures. For example, Li et al. [9] introduced a second-order attention mechanism to
model high-order semantic relationships among diﬀerent class tokens,
thereby enabling a more comprehensive capture of structural dependencies between human body parts. Despite their eﬀectiveness in speciﬁc
scenarios, pose-based methods struggle to generalize under extreme occlusions, while local attention-based methods often fail to guarantee the
extraction of reliable and holistic human appearance features.
In this paper, we attempt to provide a spatial alignment framework
for texture features through pose features, while texture features supplement pose features with ﬁne-grained discriminative clues. The combination of the two forms dual constraints of structure and detail. Speciﬁcally, instead of performing pose-to-global mapping, observing that the
re-ID based on Vision Transformer (Vit) [10] usually encodes the input
image into a set of patch sequences where diﬀerent patches vary in occlusion situations, we seek to establish a direct mapping from pose to
patch. Specially, we utilize smaller-scale patch mappings and pose-patch
correlation matrices to highlight human semantics in diﬀerent patches.
By calculating the similarity score between poses and patches, the negative impact caused by inaccurate pose estimation regions (i.e. regions
with lower similarity scores) can be eﬀectively reduced. Additionally,
since the occlusion situation of each patch is diﬀerent, constraining the
pose mapping of patches separately is beneﬁcial for guiding the model
to focus more on less-occluded areas. However, as non-target pedestrians can also be detected with pose skeletons, relying solely on pose
information is insuﬃcient to separate pedestrian occlusion from the target person. To tackle this issue, we attempt to leverage distinctive appearance as the primary basis for distinguishing target pedestrians from
non-targets. As the appearance of target and non-target pedestrians often exhibits notable texture variations, it is more feasible to leverage
these texture variations to enhance the representation of target regions
rather than directly identifying discriminative features from the overall
appearance. Therefore, we employ textures to augment the representation of discriminative appearance, thereby increasing the inter-class
distance between diﬀerent identities.

The remainder of this paper is structured as follows: Section 2 provides a brief review of related work on occluded re-ID. Section 3 introduces the pipeline of the proposed TTPM. Section 4 presents the experimental details, results, and analyses. Section 5 concludes the paper with
a summary.
2. Related work
2.1. Occluded person Re-ID
Existing occluded person re-ID methods can be roughly divided into
three streams, including occlusion augmentation-based methods, partbased methods, and external clue-based methods.
Occlusion augmentation-based methods enhance the adaptability of
the model to occlusion by simulating and generating more occlusion
samples. Dong et al. [4] improve the resistance of model to occlusion
noise by generating adversarial representations through random erasing
and feature perturbation. Bian et al. [5] develop a prior knowledgebased occlusion simulation approach, designing a novel OcclusionAware Feature Recovery (OAFR) network. This network leverages unoccluded gallery features to recover occluded query features. Chen et
al. [12] enhance occlusion robustness by simulating realistic, labeled occlusions during training and unifying occlusion types during inference
to reduce ambiguity and improve attention to visible body parts. The
aforementioned methods improve the re-ID performance of the model.
However, due to the overly simplistic and random strategy for generating occlusion, they may fail to maintain stable performance.
Part-based methods use a part-to-part region matching strategy to
solve the occlusion problem. For example, Somers et al. [13] propose
2

Pattern Recognition 172 (2026) 112341

D. Wang et al.

3.1. Overview

Body Part-Based Representation Learning ReID (BPBReID), which predicts part attention maps and generates body part features to leverage
identity and prior human topology information. Yan et al. [14] fully
leverage partial correlations to aggregate discriminative information between part features and enhance their signiﬁcance by suppressing occluded category features. Wang et al. [15] introduce jointly learning
part features and predicting part quality scores, automatically assigning
low scores to occluded body parts. Nevertheless, part-based methods
may not ensure the consistency of image matching in cases of severe
occlusion.
External clue-based methods mainly locate the non-occluded areas
of the human body through pose estimation or human parsing. PVPM
[2] utilizes pose information to achieve visibility matching of person
images and jointly learned discriminative features through attention to
mine the visibility of certain regions. Li et al. [3] propose a part-guided
representation method consisting of pose-invariant features and local
description features to infer pose-invariant representation through pose
estimation and normalization. Yuan et al. [11] employed a pose aggregation module to guide the learning of visible body parts and suppress occlusion noise by matching non-occluded regions based on pose-aligned
features. These methods can more accurately locate human body regions with the help of extra clues, but they cannot solve the problem of
occlusion caused by multiple pedestrians.

The overall network architecture of the proposed TTPM is outlined
in Fig. 2. Given a person image 𝑋 ∈ ℝ𝐻×𝑊 ×𝐶 , where 𝐻, 𝑊 , and 𝐶 are
the height, width, and channel dimensions of 𝑋, respectively, we feed
it into both the pose estimator HRnet [20] and the Vit [10] branches
to localize the target person accurately. In the pose estimation branch,
the estimator extracts 𝑀 landmarks from the input image 𝑋, each of
which corresponds to a keypoint heatmap {𝑦𝑖 |𝑖 = 1, 2, … , 𝑀 }. In the
Vit branch, the image 𝑋 is partitioned into 𝑁 non-overlapping patches
{𝑥𝑖 |𝑖 = 1, 2, … , 𝑁 }, which are mapped to 𝐷 dimensions through the linear projection function  (⋅) to generate a sequence of inputs 0 . Additionally, a learnable embedding token 𝑥𝑐𝑙𝑠 is added to the input sequence
to capture global representations of each patch. Similarly, following He
[17], we also incorporate learnable position embeddings  ∈ ℝ(𝑁+1)×𝐷
and camera viewpoint  ∈ ℝ(𝑁+1)×𝐷 as the crucial weights for the input sequence. The initial input sequence for the transformer encoder is
represented as:
[
( ) ( )
( )]
0 = 𝑥cls ;  𝑥1 ;  𝑥2 ; ⋯ ;  𝑥𝑁 +  + 𝜆
(1)
where 0 is the initial sequence input to transformer layer, and 𝜆 is a
hyperparameter used to balance the camera embedding.
Subsequently, the patch sequence 0 is fed into the MFE, which
encodes features at two levels: intra-patch and inter-patch, to explore
the relationships between diﬀerent body parts. The pose features 𝑦𝑖
and patch sequences 0 are input separately into PPM, constructing a
similarity mapping between poses and patches to avoid coarse-grained
interference under global mapping. Further details are available in
Sections 3.2 and 3.3.
Since transformer-based encoding and pose mapping focus more on
deep feature extraction and precise matching, they fail to address the
retrieval challenges caused by occlusions from non-target pedestrians.
To address this issue, we introduce a texture-aware decoder consisting
of a texture-extraction layer and a context-extraction layer. The former
is designed to capture the contextual feature of the original sequence
0 , while the latter applies cross-attention between the contextual features and patch features to extract distinctive textures. The structure of
TAD is presented in Section 3.4. To improve the re-ID accuracy of our
method, we employ encoder loss, decoder loss, and pose loss as supervision, which are introduced in Section 3.5.

2.2. Transformer-based person Re-ID
Transformer is a deep learning network widely known for its use of
attention mechanisms. Inspired by its excellent performance in natural
language processing, several works have attempted to apply it to computer vision. Carion et al. [16] ﬁrst introduce transformer into vision
tasks, proposing a Detection Transformer (DETR) model, which regards
object detection as a direct set prediction problem. Importantly, Dosovit
et al. [10] creatively present Vision Transformer (Vit) to process images
into patch sequences and demonstrate superior performance in traditional vision tasks.
Recently, some researchers have started to explore the application
of transformer in occluded person re-ID. For example, TransReID [17]
applies a pure transformer architecture based on Vit for re-identiﬁcation
and uses side information embedding to learn camera view clues, helping to mitigate feature bias caused by diﬀerent camera perspectives.
Wang et al. [18] employ transformer to occlusion augmentation and
propose a feature completion decoder that aggregates potential information to generate occluded features. Li et al. [9] introduced a secondorder attention mechanism to model high-order semantic relationships
among diﬀerent class tokens, thereby enabling a more comprehensive
capture of structural dependencies between human body parts. Wang et
al. [19] present a transformer-based Pose-guided Feature Disentangling
(PFD) approach, which selectively matches non-occluded parts using
pose information and automatically separates occluded semantics from
non-occluded semantics. In PFD, pose information is directly embedded into both global contextual features and local features. Although a
learnable semantic view is introduced to implicitly enhance discriminative features, it still fails to eﬀectively map pose positions to human
semantics.
In the paper, we present a transformer architecture with explicit
pose-patch mapping and introduce a texture-aware decoder to realize
eﬀective separation of occluded and non-occluded human semantics.

3.2. Multi-patch feature encoder
Under the transformer architecture, patches often have suﬃcient
global perception information but lack exploration of multi-level features within the patches. Therefore, we construct a multi-level patch
encoding to obtain semantics within patches and the interrelationships
between patches. As shown in Step 1 of Fig. 2, it includes two levels:
the intra-patch feature encoding and the inter-patch feature encoding.
First, the input embeddings 0 are split into 𝑘 groups:
⌊ ( )⌋
𝑙𝑒𝑛 0
(
)
𝑧 = 𝑠𝑝𝑙𝑖𝑡 0 [1 ∶], 𝑙 , 𝑙 =
(2)
𝑘
where 𝑠𝑝𝑙𝑖𝑡(⋅, ⋅) denotes the segmentation function, 𝑙 is the split length,
0 [1 ∶] is the input sequence 0 with 𝑥cls removed, and 𝑙𝑒𝑛(⋅) represents
the length of the current feature.
At the intra-patch level, patch features from the sequence 𝑧 =
[
]
𝑧0 , 𝑧1 , … , 𝑧𝑖 , … , 𝑧𝑘 and the embedding token 𝑥cls are separately input
into independent transformer layers. After the patch features and embedding token are concatenated, they are ﬂattened into three vectors
𝑄, 𝐾, 𝑉 ∈ ℝ𝑁×𝑑 , where 𝑑 denotes the embedding dimension. The attention weight calculation process for intra-patch feature encoding is as
follows:
(
)
𝐶𝑖 = 𝑐𝑎𝑡 𝑥𝑐𝑙𝑠 , 𝑧𝑖 ,
(3)
(
(
( )))
′
𝐼𝑖 = 𝜌 𝑓 𝑙𝑎𝑡𝑡𝑒𝑛 𝑛𝑜𝑟𝑚 𝐶𝑖
+ 𝐶𝑖 ,
(4)

3. Proposed method
This section details the proposed Texture-Aware Transformer with
Pose-Patch Mapping (TTPM). We ﬁrst present an overview of TTPM in
Section 3.1 and then introduce the motivation, objectives, and implementation of Multi-patch Feature Encoder (MFE), Pose-Patch Mapping
(PPM), and Texture-Aware Decoder (TAD) from Sections 3.2 to 3.4.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Text-to-image Person Search based on Semantic Reorganization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 267 (2026) 104733

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

Text-to-image Person Search based on Semantic Reorganization
Jielong He a , Feng Chen a,b ,∗, Xiwen Qu a , Yang Liu a
a
b

School of Computer Science and Technology, Anhui University of Technology, Anhui, Ma’anshan, 243032, China
Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education, Jilin University, Changchun, 130012, China

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
Cross-modal retrieval
Person re-identification
Data generation
Cross-modal re-ranking
Semantic alignment

ABSTRACT
Despite significant advancements in text-based person search (TBPS), challenges remain in acquiring large-scale
pedestrian data due to privacy concerns and the high cost of manual annotation. Additionally, substantial
modality gaps between textual and visual semantics lead to a degradation in retrieval performance. To
address these challenges, we propose a novel framework, i.e., text-to-image Person Search based on Semantic
Reorganization (PSSR). Specifically, PSSR consists of two key modules: a semantic reconstruction-based data
augmentation module, which enhances images through pedestrian structure and semantic information, and
utilizes a large multimodal language model to generate corresponding textual descriptions, thereby increasing
the diversity of the training data. Furthermore, during the testing phase, we introduce a cross-modal similarity
re-ranking mechanism that improves the feature representation by obtaining reliable image features for the
given text feature and fusing them into a whole. This approach alleviates the negative impact of modality
differences. Extensive experimental results demonstrate the significant effectiveness of our method across three
text-based person search datasets. Source code is at https://github.com/JielongHe/PSSR.

1. Introduction
Text-Based Person Search (TBPS) (Li et al., 2017; Wang et al., 2021)
is a cross-modal understanding task (Wu et al., 2024; Wang et al., 2023;
Yu et al., 2024; Gong et al., 2023) that retrieves pedestrian images
based on textual descriptions. Its goal is to establish a semantic link
between natural language and visual content, enabling the localization of fine-grained information in complex environments. Thanks to
breakthroughs in deep learning, object detection (Cheng et al., 2024),
and attention mechanisms (Zhou et al., 2022), traditional image-based
pedestrian retrieval (Feng et al., 2025b) has achieved great success,
but the performance of TBPS tasks has relatively declined. Unlike
other text–image retrieval tasks (Lülf et al., 2024), TBPS requires more
detailed textual descriptions of pedestrian images. As a result, data
annotation for TBPS is more complex than for text–image retrieval. And
due to privacy concerns, acquiring large-scale personnel image datasets
is challenging, leading to typically smaller TBPS datasets. Furthermore,
compared to traditional person re-identification (Chen et al., 2023; R.
Wang et al., 2022; Chen et al., 2021), TBPS faces significant modal
differences, making text-to-image retrieval more difficult than imageto-image retrieval. Therefore, effective measures must be taken to
augment TBPS datasets and mitigate the negative impact of modal
differences on retrieval performance.
In the TBPS task, due to data scarcity, some methods (Tan et al.,
2024) utilize extra large datasets such as LUPerson (Fu et al., 2021)

as image sources and employ Multi-modal Large Language Models
(MLLM) to generate textual descriptions. The generated image–text
pairs are used to train a pre-trained model, which is then fine-tuned
using transfer learning in the target domain. However, this approach
typically requires substantial computational resources for pre-training
and demands high-quality textual descriptions. To effectively augment
target domain data without relying on external datasets, Z. Li et al.
(2024) used large language models to diversify and rewrite text in the
existing TBPS datasets, achieving data augmentation. Z. Song et al.
(2024) leveraged publicly available clothing and accessory images from
the internet as references, as shown in Fig. 1(a), and used a diffusion
model to edit images from the original dataset, generating diverse
new images. They also modified the original textual descriptions using
the LLM model. While this approach achieved some success, it relies
on diffusion models, which have a complex training process, and the
generated images may differ from real images. To avoid excessive
computational costs, we propose a Semantic Restructuring-based Data
Augmentation (SRDA). This method divides the original images into
two groups according to gender information provided by the MLLM.
As shown in Fig. 1(b), horizontally splitting pedestrian images with
different IDs but the same gender and stitching the top and bottom parts
together to create a new image with a new ID. Unlike enhancement
strategies that rely on large-scale generative models to edit attributes
or reconstruct images (Z. Song et al., 2024), our SRDA does not attempt

∗ Corresponding author at: School of Computer Science and Technology, Anhui University of Technology, Anhui, Ma’anshan, 243032, China.

E-mail address: chenfeng@ahut.edu.cn (F. Chen).
https://doi.org/10.1016/j.cviu.2026.104733
Received 11 October 2025; Received in revised form 4 January 2026; Accepted 9 March 2026
Available online 10 March 2026
1077-3142/© 2026 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

Fig. 1. Comparison of different data augmentation methods. (a) Generate images using diffusion models based on attributes, while generating text based on the
LLM. (b) Image–text pair augmentation based on semantic information and pedestrian image structure information.

to generate visually natural new images. Instead, it focuses on the core
requirements of TBPS by explicitly reorganizing the structure of pedestrians (upper-body/lower-body semantic units) to create as diverse
and previously unseen images as possible from the original dataset,
thereby enhancing the model’s robustness. Diffusion models typically
perform high-cost iterative sampling in pixel space, with a focus on
the visual realism of the image. In contrast, our SRDA directly utilizes
semantic information and leverages the local structure of real images to
achieve combinatorial expansion. Its generation process requires no additional training and does not suffer from the cross-domain generation
quality fluctuations. Therefore, conceptually, SRDA is a ‘‘semanticbased structural combinatorial enhancement’’ rather than a generative
enhancement based on image reconstruction. Computationally, its cost
is much lower than that of diffusion models, and it does not rely on
large CLIP models for high-dimensional feature-driven image editing,
making it lightweight, stable, and controllable. Due to the lack of
consistency between the token positions in the text description and the
spatial information of the image, it is impossible to obtain an accurate
description of the generated image by splitting and reassembling the
two texts. Accordingly, we utilize the MLLM model to produce textual
descriptions for the reconstructed images, significantly improving the
accuracy of the textual descriptions.
To address the modality differences in the TBPS task, most methods
design improved cross-modal matching approaches. These approaches
align images and texts in a joint embedding space and use crossmodal matching loss functions to align image–text pairs. Common loss
functions include ranking loss and cross-modal projection matching
(CMPM) loss (Zhang and Lu, 2018). Compared to ranking loss, CMPM
loss does not require selecting specific triplets or adjusting margin parameters and demonstrates better stability across different batch sizes,
making it widely used in text-to-image person retrieval. However, due
to the variability in projection lengths, CMPM cannot precisely control
the projection probability distribution and struggles to focus on hard
negative samples when the model is updated. To address this, Jiang and
Ye (2023) proposed the SDM loss, which improves model performance
by minimizing the KL divergence between the normalized image–text
similarity score distribution and the normalized label distribution. To
better achieve semantic alignment across different modalities, several
works have made targeted improvements to the contrastive loss. Z. Lu

et al. (2025) introduced the anchor-based contrastive loss, which uses
features extracted from MLLM as anchors to align visual and text features more closely. Lu et al. (2024) proposed multimodality contrastive
learning, which facilitates the convergence of intra- and inter-modal
positive samples. Despite these advancements, these loss functions and
matching methods still do not fully address the modality differences
between text and visual data. On the other hand, some work attempts
to improve performance during the testing phase by re-ranking search
results (Zhong et al., 2017). Specifically, this method improves retrieval
performance by treating the target images (query) and the images to
be retrieved (gallery) as a whole through a metric based on reciprocal
nearest neighbors. However, the core assumption of traditional reranking relies on the ‘‘query and candidate samples sharing a stable
k-reciprocal neighborhood structure’’. This assumption holds true in
the same modality (e.g., traditional Re-ID image retrieval) but is inherently not satisfied in cross-modal (text–image) scenarios. Meanwhile,
comparisons of distance distributions across different modalities reveal
that the average cosine distance between image–image pairs is significantly smaller than that between text–image pairs, with the latter
even exhibiting an unbalanced distribution. This phenomenon suggests
that text lacks stable local structural support in feature space. These
empirical results clearly indicate that traditional re-ranking cannot
effectively utilize textual semantics in cross-modal retrieval. Based on
these observations, we propose a Cross-modal Similarity Re-ranking
Mechanism (CSRM) during the testing phase to build a more semantically reliable cross-modal neighborhood structure for text. Through
cross-modal mutual neighborhood selection and feature fusion (J. Song
et al., 2024), we retain only those neighbors that are consistent with
the text semantics and also representative in image space. We then use
the Jaccard distance to suppress noisy samples, generating more stable
new text query features. Compared to methods that improve model
architectures or loss functions, our CSRM module is activated during
the testing phase, improving the initial ranking results by finding crossmodal neighbors. The CSRM can be seen as a plug-and-play module,
which can be directly applied to the aforementioned methods to further
enhance performance.
It should be stressed that the SRDA and CSRM module are not two
independent, but rather complementary solutions proposed to address
two core issues of TBPS. Specifically, SRDA aims to alleviate the
2

J. He, F. Chen, X. Qu et al.

Computer Vision and Image Understanding 267 (2026) 104733

problems of data scarcity and insufficient semantic coverage during the
training phase, while CSRM corrects cross-modal distribution discrepancies during the testing phase, thus forming an integrated framework
of ‘‘training enhancement + testing calibration’’. More specifically,
SRDA significantly enriches the attire combinations and upper–lower
body semantic variations required for cross-modal alignment through
structural-level semantic reconstruction, enabling the model to learn a
more comprehensive diversity during training. However, even with the
expanded training data, the alignment between the textual and visual
modalities remains challenging in feature space, which affects the ranking process during the testing phase due to modal discrepancies. CSRM
addresses this residual issue by leveraging cross-modal neighborhood
selection and Jaccard weight fusion, enabling the query text to adjust
its position in feature space with the help of the most similar image
features, thereby enhancing the stability and accuracy of the final
ranking. Therefore, the two modules are logically interconnected in
their design.
Our contribution can be summarized as follows:

2.2. Cross-modal alignment-based person search
In the field of cross-modal person search (Wang et al., 2021; Duan
et al., 2024; Y. Li et al., 2024), bridging the semantic gap between
text and visual representations remains a core challenge. Li et al.
(2017) pioneered this field by constructing a shared embedding space
and introducing a cross-modal alignment framework. Later studies implemented deep learning frameworks, incorporating VGG-LSTM dualstream networks to capture global visual representations and sequential
textual features, thereby establishing the groundwork for advanced
semantic alignment. Zheng et al. (2025) proposed a graph aggregation
alignment network that consists of two modules. The graph interaction fusion module leveraged graph representation learning to process
single-modality features, while guiding low-level multimodal features
for global fusion. And the information alignment module then refined
the results through secondary calibration. To tackle modal heterogeneity, several works have explored multi-granularity feature fusion. Feng
et al. (2025a) proposed a cross-modality multi-granularity transformer
framework that extracts coarse-to-fine feature representations and integrates discriminative information across multiple granularities. Shao
et al. (2022) introduced a BERT-Transformer collaborative architecture,
achieving phrase-level visual localization via multi-level attention networks. Yan et al. (2024) suggested that combining global and local
alignment modules enhances the semantic correspondence between
different modalities. Liu et al. (2023) utilized visual attention within
the global feature extraction network, enabling the model to selectively
concentrate on key local features in both images and text descriptions,
enhancing its focus on local details.
More recent studies have concentrated on resolving modal differences at a deeper level. S. Li et al. (2024) proposed an uncertaintyaware progressive learning framework to progressively eliminate
semantic biases between modalities. Liu et al. (2024) integrated causal
reasoning by separating visual features into causally connected elements and scene perturbations, establishing domain-invariant crossmodal links via counterfactual learning. Visual–language pre-training
models also show great promise, Jiang and Ye (2023) designed a
multi-granularity attention module based on the CLIP architecture to
reinforce intra-modal consistency and inter-modal topological correspondence. You et al. (2025) designed an effective framework that
extracts co-saliency features between images and text, enhancing the
variation of cross-modal discriminative features while reducing noise
interference. To tackle the semantic inconsistency in weak positive
pairs caused by viewpoint differences, Lu et al. (2024) extracted
cross-modal identity-invariant features using semantic-aligning optimal
transport and multimodal contrastive learning techniques. Moreover,
Z. Lu et al. (2025) utilized learnable prompts and MLLM interactive
learning to address the semantic variations in positive pairs and the appearance similarity in negative pairs, respectively. Beyond conventional
retrieval settings, Y. Lu et al. (2025) explored an interactive text-based
person search paradigm, where multimodal large language models are
leveraged to refine ambiguous queries through dialogue.
Despite progress, challenges in handling modal differences and
redundant information remain. We propose a cross-modal similarity
re-ranking mechanism in the testing phase. Specifically, we select
appropriate image features to represent the counterpart text, thereby
narrowing the modality gap. Unlike prior approaches, our method
refines cross-modal similarity estimation during the inference phase in
a training-free and model-agnostic manner, making it complementary
to existing cross-modal alignment frameworks.

• We utilize the MLLM model based on pedestrian image structural information to perform semantic-based data augmentation,
effectively enriching the diversity of the data.
• We propose a cross-modal similarity re-ranking mechanism,
which selects the most suitable image features to fuse with
the corresponding text feature, thereby mitigating the negative
impact of modality differences on retrieval performance.
• Our method outperforms existing state-of-the-art methods, achieving the mean Average Precision (mAP) of 73.34%, 48.11%
and 51.77% on the CUHK-PEDES, ICFG-PEDES and RSTPReid,
respectively.
2. Related works
Text-based person retrieval in this paper is closely related to data
augmentation and cross-modal alignment. We briefly review the literature in these two areas.
2.1. Data augmentation for person search
In Text-Based Person Search (TBPS), data augmentation is a key
technique for enhancing model robustness and generalization, gaining
significant attention in recent years. Traditional person re-identification
tasks typically employ image augmentation methods such as cropping,
flipping, and rotation to improve model generalization. However, generative augmentation strategies, including those based on traditional
methods and GANs (Zheng et al., 2017), often incur high training costs
with limited performance gains. Chen et al. (2020) proposed a selfsupervised learning-based data augmentation strategy that combines ID
and instance augmentation, but it lacks fine-grained semantic control.
In TBPS, both image augmentation and generation of semantically consistent image–text pairs are crucial. Z. Li et al. (2024) leveraged large language models to rewrite textual descriptions in existing
datasets, generating diverse text, though only addressing textual augmentation. The MALS dataset (Yang et al., 2023) employs diffusion
models and text calibration to generate large-scale synthetic data. Tan
et al. (2024) used the LUPerson dataset (Fu et al., 2021) and Multimodal Large Language Models (MLLM) to generate textual descriptions
for images, while Z. Song et al. (2024) proposed the Diverse Person framework, using a diffusion model to edit person attributes and
large language models for textual annotations, mitigating privacy and
annotation cost issues.
However, these methods depend on external attributes or images
and have limitations in fine-grained semantic control. In this paper, we
propose a data augmentation method based on semantic reorganization. Specifically, we generate high-quality images based on pedestrian
image structure and produce corresponding text by fine-tuning the
MLLM. During this process, we utilized MLLM to ensure cross-modal
semantic consistency of the generated data.

3. Method
To address the issue of insufficient diversity of training data and significant modal differences in TBPS tasks, as shown in Fig. 2, we propose
the PSSR framework. Firstly, we introduce semantic restructuring-based
data augmentation to get more diverse data. Secondly, all original and
3


exec
/bin/zsh -lc "pdftotext -l 3 'TP-LReID - Lifelong person re-identification using text prompts.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112326

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

TP-LReID: Lifelong person re-identiﬁcation using text prompts
Zhaoshuo Liu a,1 , Zhiwei Guo a,1 , Chaolu Feng
Jinzhu Yang a,b
a

a,b,∗, Wei Li a,b , Kun Yu c , Jun Hu

d,

School of Computer Science and Engineering, Northeastern University, 110819, Shenyang Liaoning, China

b Key Laboratory of Intelligent Computing in Medical Image, Ministry of Education, 110169, Shenyang Liaoning, China
c

College of Medicine and Biological Information Engineering, Northeastern University, 110016, Shenyang Liaoning, China

d Neusoft Reach Automotive Technology Company, 110179, Shenyang Liaoning, China

a r t i c l e

i n f o

Keywords:
Lifelong person re-identiﬁcation
Model generalization
Feature diversity

a b s t r a c t
Lifelong person re-identiﬁcation (LReID) aims to develop a single model that is capable of continuously learning
from new domain (present) while retaining knowledge from previously encountered ones (past) and generalizing
to unseen domains (future). However, distribution shifts across these domains pose a signiﬁcant challenge in
maintaining performance across past, present, and future domains, that is, causing the catastrophic forgetting on
previously seen domains and limited generalization to unseen ones. To address the above issues, we propose to
guide consistent feature extraction to bridge distribution shifts using text prompts designed to remain invariant
across domains. First, identity-consistent text prompts capturing high-level image semantics are extracted and
aligned with image features throughout the lifelong learning pipeline. Moreover, to enhance generalization to
unseen domains, we introduce an adversarial training that text features are contrastively aligned with both
original and future-style image features, the latter generated by applying gradient-based perturbations in the
feature space. Compared with 21 representative models on 11 benchmark datasets, our proposed model, trained
without access to historical data, achieves performance comparable to the model trained using a joint training
approach, and it performs well on all of the past, present, and future domains. We further explored the forgetting
of the ﬁrst historical domain and the generalization to all unseen domains under all 24 orders, and the results
conﬁrmed the superiority of our model. Codes will be released if this paper is accepted.

1. Introduction
Person re-identiﬁcation (ReID) [1,2] aims to match the same person
captured by cameras at diﬀerent times and/or locations. The model has
to perform well on both historical (seen) and future (unseen) domains
while ensuring its person identiﬁcation ability on the present domain.
While image-based ReID methods have made signiﬁcant progress, recent studies reveal that hybridizing images with text can further enhance performance [3]. Text, as a high-level semantic representation of
images [4], exhibits stronger domain consistency compared to images,
as illustrated in Fig. 1(a). The advent of pre-trained vision-language
models like CLIP [5] has enabled hybridized ReID models to achieve
strong performance when trained on a single present domain. These
methods, which focus only on ﬁtting a single present-domain distribution and are referred as Common ReID in this paper, fail to learn dynamic distributions in a continual manner or generalize to unseen distributions. Continual person ReID (CReID) strives to dynamically learn

domain-consistent features and overcome catastrophic forgetting caused
by distribution shifts, using regularization [6], knowledge accumulation [7–9] and so on. CKP [7] is one of the latest model in the ﬁeld
of CReID, which performs better than Common ReID on past domains
as shown in Fig. 1(b). Note that the present domain in CReID in our
work refers to the most recently encountered domain in the training sequence and those before the present domain are past domains. However,
there is a lack of dedicated mechanisms in CReID speciﬁcally aimed at
improving generalization to future unseen domains. Generalized person ReID endeavors to adapt to the distribution of future unseen domains when trained on one or several present domains, using data augmentation, meta learning, adversarial learning and so on. As shown in
Fig. 1(b), the generalized ReID model QAConv [12] performs better than
the CReID CKP on future domains. Nevertheless, generalized ReID fails
to dynamically ﬁt the distribution in a continual fashion, which is how
the practical data stream arrives. In this paper, we propose a person ReID
model that leverages text prompts to guide domain-consistent feature

∗ Corresponding author.
1

E-mail address: fengchaolu@cse.neu.edu.cn (C. Feng).
These authors contributed equally to this work.

https://doi.org/10.1016/j.patcog.2025.112326
Received 10 March 2025; Received in revised form 18 June 2025; Accepted 21 August 2025
Available online 26 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

learning by aligning them with image features throughout lifelong learning. To further enhance generalization to unseen domains, we apply
style adversarial training, contrastively aligning text with both original
and future-style image features generated via gradient-based perturbations. Our method combines the advantages of common ReID, continual ReID and generalized ReID to achieve performance improvement,
forgetting prevention, and generalization promotion on domains corresponding to present, past, and future. The overall training paradigm is
illustrated in Fig. 1(c). The main contributions are summarized as follows,

and domain-consistent knowledge representation [33,34]. Additionally,
continual compatible representation and transfer mechanisms enhance
feature compatibility across domains [35], while knowledge adapters
dynamically learn and assign domain-speciﬁc knowledge for LReID tasks
[36]. To improve robustness to real-world clothing variations in CReID,
some methods incorporate personalized clothing-state tokens alongside
identity-speciﬁc tokens in CLIP-ReID, addressing challenges posed by
clothing changes in lifelong learning [37]. The aforementioned works
primarily focus on addressing catastrophic forgetting, with relatively
less emphasis on mechanisms speciﬁcally designed to enhance generalization to future domains.

– To the best of our knowledge, we are the ﬁrst to propose the mechanism that smooths distribution diﬀerences with the guidance of text
prompts in the lifelong person re-identiﬁcation across past, present,
and future domains.
– Text prompts and image encoding update alternately and sequentially with the arrival order of datasets to promote each other continually to ensure performance of the model on the past and present.
– Future style generation is introduced to simulate unseen samples in
the image embedding space, adopted as an additional regularization
together with the corresponding text embeddings on training the image encoder and therefore the text prompt generator, to ensure the
generalization on the future.

2.3. Generalized person re-identiﬁcation
The aim of generalized person ReID is to learn a generalized model
using samples from a single domain or multiple domains and then to apply it to samples coming from unseen domains. Depending on whether
the unseen domains can be accessed in advance, generalized person
ReID can be classiﬁed into methods based on domain adaptation and
methods based on domain generalization [2]. Typically, the domain
adaptation-based methods acquire samples of target domain prematurely and transfers the style of samples from training domain to target
domain for improving the performance on target domain [38]. One line
of domain generalization methods focuses on learning domain-invariant
features by ﬁltering domain-speciﬁc information, using techniques such
as instance normalization [25] and MetaBN [24]. Another direction
seeks to reduce overﬁtting to the source domain through methods like
query-adaptive convolution [12] and self-distillation [39]. In addition,
cross-camera generalization has been explored by exploiting unpaired
samples from diﬀerent cameras and introducing identity-guided perturbation prompts to enhance robustness against camera variations [40].
Recent progress in vision-language models, notably CLIP, has also been
utilized to boost generalizability in person ReID [41].

Overall structure of our paper. We begin with the motivation and
key contributions in Section 1, followed by a review of related works in
Section 2. Section 3 details our proposed TP-LReID model, while Section 4 presents the experimental setup, results, and analysis. Further
discussions on implications and limitations are in Section 5, and we conclude in Section 6 with a summary and future directions.
2. Related work
2.1. Person re-identiﬁcation

3. Method

Person ReID is a sub-problem of image retrieval, aiming to identify a
given person in an image from a gallery of images captured by cameras
at diﬀerent times and/or locations. According to diﬀerent perspectives
of methodology, person ReID can be classiﬁed into representative learning based [13] and metric learning based [14], global based [15] and
local based [16]. From the data modality perspective, ReID methods include image-based [17], video-based [18], and multimodal approaches
that incorporate complementary modalities, such as text-image pairs
[3,19,20] or visible-infrared data [21–23]. According to the perspectives
of model adaptability, it can be classiﬁed into continual [7–9] and generalized [12,24,25]. All of the above methods require annotated data
during training, i.e., they belong to supervised learning. Accordingly,
unsupervised learning-based re-identiﬁcation does not require labeled
data and focuses on learning from a large amount of unlabeled data
[26,27].

Problem formulation. As described above, the proposed TP-LReID
aims to learn a model that generalizes across both seen (i.e., past and
present) and unseen (i.e., future) domains. Let 𝐷 = {𝐷1 , 𝐷2 , … , 𝐷𝑍 } represent a sequence of datasets (domains), where training is performed sequentially from 𝐷1 to 𝐷𝑖 . At any stage 𝑖, the training process involves all
data up to and including 𝐷𝑖 , denoted as 𝐷1∶𝑖 . We deﬁne the past domains
as 𝐷past = {𝐷1 , 𝐷2 , … , 𝐷𝑖−1 }, the present domain as 𝐷present = {𝐷𝑖 }, and
the future domains as 𝐷future = {𝐷𝑖+1 , … , 𝐷𝑍 }. Each domain 𝐷𝑡 , for
|𝐷 |
𝑡 = 1, 2, … , 𝑍, consists of a set of image-label pairs 𝐷𝑡 = {(𝑥𝑘 , 𝑦𝑘 )}𝑘=1𝑡 ,
where |𝐷𝑡 | denotes the number of samples in 𝐷𝑡 . During training, two
mini-batch sets are constructed by randomly sampling from 𝐷𝑡 : one for
text prompt generation and one for image encoding, which are denoted
txt } and 𝐷img = {𝐵 img , … , 𝐵 img }. Each mini-batch
by 𝐷text = {𝐵1txt , … , 𝐵𝐻
𝑅
1
img

(𝐵ℎtxt or 𝐵𝑟 ) contains 𝑁 identities with 𝐾 samples per identity. Each
domain 𝐷𝑡 is split into disjoint training and testing subsets, denoted as
𝐷𝑡train and 𝐷𝑡test , respectively, with no class overlap between them.
Overview. The framework of the proposed model TP-LReID is given in
Fig. 2. Generally, the proposed model consists of three main modules,
namely text prompt generator (⋅), image feature encoder (⋅), and future sample simulator (⋅). To extract and preserving domain invariant
knowledge from already seen domains during training, (⋅) and (⋅) are
updated interactively by turn for each domain arrival. In addition, we
adopt the text encoder  existing in CLIP as a frozen module to extract
features from texts generated by the text prompt generator (⋅). The future sample simulator (⋅) is introduced to simulate unseen samples in
the image embedding space to ensure generalization of the proposed
model on future domains. During testing, the image encoder is used
to construct a metric space where inter-identities are far away while
intra-identities are close. Embeddings of the query and gallery images
are extracted by the image encoder and similarly measured by euclidean

2.2. Continual person re-identiﬁcation
Due to data-driven characteristics, performance of person ReID often degrades on historical domains when trained solely on current domain, leading to the issue of catastrophic forgetting. Continual person
re-identiﬁcation (CReID), also known as lifelong person re-identiﬁcation
(LReID), aims to address this issue and can be broadly categorized into
replay-based and exemplar-free methods. Replay-based methods mitigate catastrophic forgetting by storing and replaying samples from
past domains during training [28,29]. Exemplar-free methods rely on
knowledge representation and transfer to retain learned knowledge.
Some methods use graph neural networks to model and accumulate
knowledge in CReID [7–9,30]. Others employ reﬁned knowledge transfer and ﬁne-grained knowledge acquisition to further mitigate forgetting [31,32]. To address domain shifts in CReID, various strategies have been proposed, including patch-based knowledge distillation
2

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Fig. 1. Depiction of (a) domain distribution consistency, (b) comprehensive robustness across all domains, and (c) overall setting.

3


codex
只读了每篇前三页，判断基于摘要和引言，不当作全文审查。

**逐篇提取**
1. **Spectrum-guided feature enhancement network for event person re-identification**
创新类型：新数据设定加新机制。把 event camera ReID 定义成“隐私友好但语义稀疏、噪声强”的问题。  
为什么能发：gap 是 RGB ReID 不能直接搬到 event，已有 event ReID 又没系统处理稀疏语义和噪声。证据链是频域去噪 MSAM 加连续 patch dropout，在 event ReID 数据集上做 SOTA。  
story 套路：隐私和极端光照需要 event 传感器，但 event 丢语义，所以要从频域和区域鲁棒性两条线把身份信息救回来。

2. **Stochastic style perturbation modelling for VI-ReID with severely modality imbalance**
创新类型：问题重定义加新机制。不是再讲 RGB-IR gap，而是讲 IR 样本严重不足。  
为什么能发：现有 VI-ReID 默认 RGB/IR 平衡，现实里 IR 难采。机制是特征空间的随机 style perturbation，模拟少数模态分布，再用分布对齐和类感知对比学习稳住。  
story 套路：真实部署不是两种模态都够，而是少数模态被多数模态淹没，因此要做“模态均衡学习”。

3. **Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-ID in Videos**
创新类型：强新机制。把 3D gait 从稀疏 skeleton/SMPL 参数升级成 dense 3D gait field。  
为什么能发：gap 是 skeleton 太稀疏、SMPL 参数太低维，和 RGB appearance feature 不好交互。证据链是 3D mesh surface 到 2D 图像的稠密蒸馏，再和 appearance token 交互，多视频换衣数据集验证。  
story 套路：换衣场景不能信衣服，要信动态人体几何，但几何不能只用几个关节或十维 SMPL 参数，要变成和图像特征同空间的稠密场。

4. **Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID**
创新类型：新机制，语义原型对齐。  
为什么能发：gap 是 CLIP 类方法只靠 learnable token，语义太薄。它用属性文本加轮廓文本，构造 modality-agnostic prototype 来桥接 RGB/IR。  
story 套路：文本不是装饰，而是跨模态共享的语义锚点，属性管身份线索，轮廓补结构线索。

5. **Text-to-image Person Search based on Semantic Reorganization**
创新类型：偏工程组合，但有一个清楚的数据增强设定。  
为什么能发：gap 是 TBPS 数据少、标注贵，扩散生成成本高且不稳。它用上半身和下半身结构重组生成新图，再用 MLLM 生成文本，测试时再做跨模态重排序。  
story 套路：不追求生成逼真图，而是低成本扩大“语义组合空间”，训练增强加测试校准。

6. **Texture-aware transformer with pose-patch mapping for occluded person re-identification**
创新类型：问题细化加新机制。把遮挡进一步拆成极端物体遮挡和非目标行人遮挡。  
为什么能发：gap 是全局 pose guidance 在极端遮挡下会引噪声，且 pose 本身分不清目标人和遮挡行人。机制是 pose-patch mapping 加 texture-aware decoder。  
story 套路：遮挡不是只找可见人体区域，还要在“结构对齐”和“纹理判别”之间互相约束。

7. **TIENet**
创新类型：新机制。多模态 ReID 里做空间、频率、模态间、模态内三类交互。  
为什么能发：gap 是 RGB/NI/TI 多模态融合多停留在空间拼接，没利用频域和互补关系。证据链是 amplitude/phase 分析加三交互模块。  
story 套路：多模态不是简单融合，而是每个模态先保留自身频域特性，再按关系交互。

8. **Towards Anytime Retrieval**
创新类型：强问题重定义加新数据设定。提出 AT-ReID，覆盖白天、夜晚、短期、长期、换衣、RGB/IR。  
为什么能发：gap 是现有 Tr-ReID、VI-ReID、CC-ReID 各管一块，真实系统要 anytime retrieval。证据链是 21 个月采集的 AT-USTC 数据集和六场景统一模型。  
story 套路：真正应用不是某一个 benchmark，而是任何时间都能检索，所以先定义新任务，再给统一模型。

9. **TP-LReID**
创新类型：新机制。用 text prompt 统一 lifelong ReID 的过去域、当前域、未来域。  
为什么能发：gap 是 CReID 重防遗忘，DG 重泛化，没人同时处理 past、present、future。机制是文本 prompt 域一致约束加 future-style feature perturbation。  
story 套路：终身 ReID 不能只记旧域，还要为未来域留空间，文本语义比图像风格更稳定。

**综合规律**
这批能发的工作，大多不是“模块更复杂”，而是先把一个默认假设打掉：模态一定平衡、数据一定同场景、3D 表征够用、pose 一定可靠、ReID 只发生在单一时间尺度。我们的强点也应该这么讲，优先用团队已有的 aerial-ground、SMPL、pose gating 和 SOLIDER，而不是再做普通注意力模块。

**候选创新点**
1. **把 aerial-ground ReID 从图像匹配改成人体表面匹配。**  
挂靠资产：aerial-ground 数据和代码，SMPL mesh 投影，SOLIDER-Swin 特征。  
核心想法：用 SMPL 投影建立跨视角一致的 body-surface token，让航拍俯视低清图和地面图都对齐到同一 3D 人体表面坐标。  
区别：最像 SD-3DGF 和 TTPM，但 SD-3DGF 是视频换衣 gait field，TTPM 是 2D pose-patch 遮挡；我们是极端 aerial-ground 视角下的 3D surface correspondence。  
cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。

2. **把 aerial-ground 训练定义成严重不平衡的视角模态学习。**  
挂靠资产：aerial-ground，SOLIDER-Swin。  
核心想法：现实里地面图多、航拍图少且低清，不能假设双视角平衡。用少量航拍样本估计 view-style 分布，在 SOLIDER 特征空间做航拍式扰动，而不是图像生成。  
区别：最像 CMBL，但 CMBL 是 RGB/IR 样本不平衡；我们处理的是视角、高度、尺度和清晰度共同造成的 aerial-ground 不平衡。  
cheap kill-switch：人工把航拍训练样本降到 10% 或 20%，比较特征扰动、重采样、普通增强三者。若扰动不能稳定超过重采样，或只在人工极端不平衡下有效，就不作为主线。

3. **用 3D 几何 prompt 替代 text prompt 做 lifelong ReID 的稳定锚。**  
挂靠资产：SMPL 3D 几何，pose 热图门控，当前 lifelong ReID 台子，SOLIDER-Swin。  
核心想法：旧域保留的是人体几何对应关系和可见表面结构，新域主要更新外观判别，避免纯外观特征在顺序训练中漂移。  
区别：最像 TP-LReID，但它用文本 prompt 和 future-style perturbation；我们用显式 3D joints、mesh surface、2D projection 做 prompt，证据更可解释，也更贴合换衣和跨视角。  
cheap kill-switch：先不完整训练，只在 Market、Duke、MSMT、CUHK03 上抽 SMPL/pose token，测它们是更按身份聚类还是按数据集聚类。再做一个两域短训，若不能减少第一域遗忘，止损。

4. **把遮挡 ReID 收窄成目标人和干扰人的竞争问题。**  
挂靠资产：pose 热图门控，SOLIDER-Swin。  
核心想法：专攻非目标行人遮挡，不把车、树、包和另一个人混成一种遮挡。多人体 pose heatmap 负责分人，SOLIDER 纹理负责保留身份细节。  
区别：最像 TTPM，但 TTPM 还是通用遮挡框架；我们的切开点是“多人遮挡下目标人归属”，需要专门的多人 pose 实例和目标一致性证据。  
cheap kill-switch：先从 Occluded-Duke 或 aerial-ground 里筛出多人遮挡子集，做冻结特征门控对照。若多人遮挡样本太少，或 target pose gate 相对 all-pose gate 没有明显优势，就不要推进。

我会优先押候选 1 和候选 3。候选 1 更像真正的问题加机制创新，和团队资产贴得最紧；候选 3 能接上当前 lifelong 台子，但要先证明 3D 几何 prompt 在普通 ReID 图像上够稳定。候选 2 可以做低成本辅线，候选 4 风险最大，因为和 TTPM 太近，必须靠“多人遮挡归属”这个窄切口拉开。
tokens used
79,287
只读了每篇前三页，判断基于摘要和引言，不当作全文审查。

**逐篇提取**
1. **Spectrum-guided feature enhancement network for event person re-identification**
创新类型：新数据设定加新机制。把 event camera ReID 定义成“隐私友好但语义稀疏、噪声强”的问题。  
为什么能发：gap 是 RGB ReID 不能直接搬到 event，已有 event ReID 又没系统处理稀疏语义和噪声。证据链是频域去噪 MSAM 加连续 patch dropout，在 event ReID 数据集上做 SOTA。  
story 套路：隐私和极端光照需要 event 传感器，但 event 丢语义，所以要从频域和区域鲁棒性两条线把身份信息救回来。

2. **Stochastic style perturbation modelling for VI-ReID with severely modality imbalance**
创新类型：问题重定义加新机制。不是再讲 RGB-IR gap，而是讲 IR 样本严重不足。  
为什么能发：现有 VI-ReID 默认 RGB/IR 平衡，现实里 IR 难采。机制是特征空间的随机 style perturbation，模拟少数模态分布，再用分布对齐和类感知对比学习稳住。  
story 套路：真实部署不是两种模态都够，而是少数模态被多数模态淹没，因此要做“模态均衡学习”。

3. **Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-ID in Videos**
创新类型：强新机制。把 3D gait 从稀疏 skeleton/SMPL 参数升级成 dense 3D gait field。  
为什么能发：gap 是 skeleton 太稀疏、SMPL 参数太低维，和 RGB appearance feature 不好交互。证据链是 3D mesh surface 到 2D 图像的稠密蒸馏，再和 appearance token 交互，多视频换衣数据集验证。  
story 套路：换衣场景不能信衣服，要信动态人体几何，但几何不能只用几个关节或十维 SMPL 参数，要变成和图像特征同空间的稠密场。

4. **Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for VI-ReID**
创新类型：新机制，语义原型对齐。  
为什么能发：gap 是 CLIP 类方法只靠 learnable token，语义太薄。它用属性文本加轮廓文本，构造 modality-agnostic prototype 来桥接 RGB/IR。  
story 套路：文本不是装饰，而是跨模态共享的语义锚点，属性管身份线索，轮廓补结构线索。

5. **Text-to-image Person Search based on Semantic Reorganization**
创新类型：偏工程组合，但有一个清楚的数据增强设定。  
为什么能发：gap 是 TBPS 数据少、标注贵，扩散生成成本高且不稳。它用上半身和下半身结构重组生成新图，再用 MLLM 生成文本，测试时再做跨模态重排序。  
story 套路：不追求生成逼真图，而是低成本扩大“语义组合空间”，训练增强加测试校准。

6. **Texture-aware transformer with pose-patch mapping for occluded person re-identification**
创新类型：问题细化加新机制。把遮挡进一步拆成极端物体遮挡和非目标行人遮挡。  
为什么能发：gap 是全局 pose guidance 在极端遮挡下会引噪声，且 pose 本身分不清目标人和遮挡行人。机制是 pose-patch mapping 加 texture-aware decoder。  
story 套路：遮挡不是只找可见人体区域，还要在“结构对齐”和“纹理判别”之间互相约束。

7. **TIENet**
创新类型：新机制。多模态 ReID 里做空间、频率、模态间、模态内三类交互。  
为什么能发：gap 是 RGB/NI/TI 多模态融合多停留在空间拼接，没利用频域和互补关系。证据链是 amplitude/phase 分析加三交互模块。  
story 套路：多模态不是简单融合，而是每个模态先保留自身频域特性，再按关系交互。

8. **Towards Anytime Retrieval**
创新类型：强问题重定义加新数据设定。提出 AT-ReID，覆盖白天、夜晚、短期、长期、换衣、RGB/IR。  
为什么能发：gap 是现有 Tr-ReID、VI-ReID、CC-ReID 各管一块，真实系统要 anytime retrieval。证据链是 21 个月采集的 AT-USTC 数据集和六场景统一模型。  
story 套路：真正应用不是某一个 benchmark，而是任何时间都能检索，所以先定义新任务，再给统一模型。

9. **TP-LReID**
创新类型：新机制。用 text prompt 统一 lifelong ReID 的过去域、当前域、未来域。  
为什么能发：gap 是 CReID 重防遗忘，DG 重泛化，没人同时处理 past、present、future。机制是文本 prompt 域一致约束加 future-style feature perturbation。  
story 套路：终身 ReID 不能只记旧域，还要为未来域留空间，文本语义比图像风格更稳定。

**综合规律**
这批能发的工作，大多不是“模块更复杂”，而是先把一个默认假设打掉：模态一定平衡、数据一定同场景、3D 表征够用、pose 一定可靠、ReID 只发生在单一时间尺度。我们的强点也应该这么讲，优先用团队已有的 aerial-ground、SMPL、pose gating 和 SOLIDER，而不是再做普通注意力模块。

**候选创新点**
1. **把 aerial-ground ReID 从图像匹配改成人体表面匹配。**  
挂靠资产：aerial-ground 数据和代码，SMPL mesh 投影，SOLIDER-Swin 特征。  
核心想法：用 SMPL 投影建立跨视角一致的 body-surface token，让航拍俯视低清图和地面图都对齐到同一 3D 人体表面坐标。  
区别：最像 SD-3DGF 和 TTPM，但 SD-3DGF 是视频换衣 gait field，TTPM 是 2D pose-patch 遮挡；我们是极端 aerial-ground 视角下的 3D surface correspondence。  
cheap kill-switch：先在少量 AG-ReID/CARGO 样本上跑现有 SMPL/pose 管线，冻结 SOLIDER，只做 mesh surface pooling。若 SMPL 投影在航拍上失败率高，或同一表面区域的跨视角相似度不优于普通 patch/pose part，直接止损。

2. **把 aerial-ground 训练定义成严重不平衡的视角模态学习。**  
挂靠资产：aerial-ground，SOLIDER-Swin。  
核心想法：现实里地面图多、航拍图少且低清，不能假设双视角平衡。用少量航拍样本估计 view-style 分布，在 SOLIDER 特征空间做航拍式扰动，而不是图像生成。  
区别：最像 CMBL，但 CMBL 是 RGB/IR 样本不平衡；我们处理的是视角、高度、尺度和清晰度共同造成的 aerial-ground 不平衡。  
cheap kill-switch：人工把航拍训练样本降到 10% 或 20%，比较特征扰动、重采样、普通增强三者。若扰动不能稳定超过重采样，或只在人工极端不平衡下有效，就不作为主线。

3. **用 3D 几何 prompt 替代 text prompt 做 lifelong ReID 的稳定锚。**  
挂靠资产：SMPL 3D 几何，pose 热图门控，当前 lifelong ReID 台子，SOLIDER-Swin。  
核心想法：旧域保留的是人体几何对应关系和可见表面结构，新域主要更新外观判别，避免纯外观特征在顺序训练中漂移。  
区别：最像 TP-LReID，但它用文本 prompt 和 future-style perturbation；我们用显式 3D joints、mesh surface、2D projection 做 prompt，证据更可解释，也更贴合换衣和跨视角。  
cheap kill-switch：先不完整训练，只在 Market、Duke、MSMT、CUHK03 上抽 SMPL/pose token，测它们是更按身份聚类还是按数据集聚类。再做一个两域短训，若不能减少第一域遗忘，止损。

4. **把遮挡 ReID 收窄成目标人和干扰人的竞争问题。**  
挂靠资产：pose 热图门控，SOLIDER-Swin。  
核心想法：专攻非目标行人遮挡，不把车、树、包和另一个人混成一种遮挡。多人体 pose heatmap 负责分人，SOLIDER 纹理负责保留身份细节。  
区别：最像 TTPM，但 TTPM 还是通用遮挡框架；我们的切开点是“多人遮挡下目标人归属”，需要专门的多人 pose 实例和目标一致性证据。  
cheap kill-switch：先从 Occluded-Duke 或 aerial-ground 里筛出多人遮挡子集，做冻结特征门控对照。若多人遮挡样本太少，或 target pose gate 相对 all-pose gate 没有明显优势，就不要推进。

我会优先押候选 1 和候选 3。候选 1 更像真正的问题加机制创新，和团队资产贴得最紧；候选 3 能接上当前 lifelong 台子，但要先证明 3D 几何 prompt 在普通 ReID 图像上够稳定。候选 2 可以做低成本辅线，候选 4 风险最大，因为和 TTPM 太近，必须靠“多人遮挡归属”这个窄切口拉开。
