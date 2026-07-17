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
session id: 019ef71f-5026-7192-ba2b-acaf09a0530b
--------
user
你是 ReID 论文创新挖掘员 2/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(2/20):
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则核对目标和状态文件，再只用 `pdftotext -l 3` 抽取这批论文的前 3 页。当前环境是只读，我不会改动任何实验记录或源码。
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

exec
/bin/zsh -lc "sed -n '1,260p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
状态文件确认了当前只做文献差距分析和候选方向，不启动实验。下面我会按你限定的范围逐篇抽前 3 页，重点看摘要、引言里的问题定义、证据组织和卖点写法。
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
/bin/zsh -lc "pdftotext -l 3 'Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

5067

Adaptive Occlusion-Aware Network for Occluded
Person Re-Identification
Xiangzeng Liu , Member, IEEE, Jianfeng Guo , Hao Chen , Qiguang Miao , Senior Member, IEEE,
Yue Xi , and Ruyi Liu

Abstract— Occluded person re-identification (ReID) is a challenging task due to some of the essential features are interfered
by obstacles or other pedestrians. Multi-granularity local feature
extraction and recognition can effectively improve the accuracy
of ReID under occlusion. However, manual segmentation methods
for local features can lead to feature misalignment. Feature alignment based on pose estimation often ignores non-body details
(e.g., handbags, backpacks, etc.) while increasing the complexity
of the model. To address the above challenges, we propose a novel
Adaptive Occlusion-Aware Network (AOANet), which mainly
consists of two modules, the Adaptive Position Extractor (APE)
and the Occlusion Awareness Module (OAM). In order to
adaptively extract distinguishing features of body parts, APE
optimizes the representation of multi-granularity features by
the guidance of attention mechanism and keypoint features.
To further perceive the occluded region, the OAM is developed
by adaptively calculating the occlusion weights for body parts.
These weights can lead to highlighting the non-occluded parts
and suppressing the occluded parts, which in turn improves
the accuracy in the occluded situation. Extensive experimental
results confirm the advantages of our method on the MSMT17,
DukeMTMC-reID, Market-1501, Occluded-Duke and OccludedReID datasets. The comparative results demonstrate that our
method outperforms comparable methods. Especially on the
Occluded-Duke dataset, our method achieved 70.6% mAP and
81.2% Rank-1 accuracy.
Index Terms— Occluded person re-identification, body positions, transformer, local features.

I. I NTRODUCTION

A

S AN important direction of research in the field of
intelligent monitoring, person re-identification (ReID) is
a key technology to realize long-time pedestrian object tracking and cross-camera tracking. Its principle is to recognize
a specific pedestrian object by comparing the similarity of
appearance features of pedestrians in different scenes, and thus
Received 7 July 2024; revised 1 November 2024; accepted 28 December
2024. Date of publication 31 December 2024; date of current version
7 May 2025. This work was supported in part by the Natural Science Basic
Research Program of Shaanxi under Grant 2024JC-YBMS-467, in part by
the Aeronautical Science Foundation of China under Grant D023030002,
and in part by the Fundamental Research Funds for the Central Universities
under Grant QTZX24067. This article was recommended by Associate Editor
Z. Tao. (Corresponding authors: Xiangzeng Liu; Qiguang Miao.)
Xiangzeng Liu, Qiguang Miao, and Ruyi Liu are with the School
of Computer Science and Technology, Xidian University, Xi’an 710071,
China (e-mail: xzliu@xidian.edu.cn; qgmiao@xidian.edu.cn; ruyiliu@xidian.
edu.cn).
Jianfeng Guo, Hao Chen, and Yue Xi are with Guangzhou Institute
of Technology, Xidian University, Xi’an 710071, China (e-mail: jianfengguo@stu.xidian.edu.cn; haochenxd@stu.xidian.edu.cn; xiyue@xidian.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3524555

realizing the continuous cross-scene tracking of the object.
It is mainly applied in the fields of public security criminal
investigation, intelligent security, intelligent campus, intelligent shopping malls, and so on. However, object occlusion,
pose uncertainty, appearance changes, and scene complexity
make person ReID still face many challenges. To address
these challenges, several methods [1], [2], [3] have recently
been proposed for ReID. For example, Luo et al. [1] used
ResNet50 [4] to extract global features of people and combined a number of ReID tricks to achieve good performance.
The occlusion phenomenon occurs frequently in different scenarios, which seriously affects the accuracy of re-recognition.
Therefore, methods relying only on global features cannot
achieve good performance in occluded environments. To cope
with occluded person ReID, some local feature-based methods [5], [6], [7] have demonstrated promising results. However,
the local features are susceptible to noise, such as background
interference, and their robustness against occlusion remains
insufficient. Therefore, the development of a robust occluded
person re-identification method is imperative.
The part-based methods show great potential in addressing the challenge of occluded person ReID, and are mainly
divided into manual splitting methods and pose estimation
based methods. Manual splitting methods are prone to feature
misalignment problems. As show in Fig. 1a, manual splitting is
employed to determine the positions of human bodies, which
in turn generates local features. However, due to differences
in object scales, this method may incorrectly compare the
head region of one image with the background region of
another image, leading to matching failures. The uniform
splitting of each person contributes to the issue of local feature
misalignment. One approach to rectify this challenge involves
leveraging pose estimation [8] to aid in position generation as
depicted in Fig. 1b. However, the local features generated by
pose estimation may become unstable due to the sensitivity
of environmental noise. In addition, the algorithm prioritizes
human features, which can lead to the omission of certain
important non-human human features, such as backpacks and
handbags. Furthermore, the implementation of this method
requires the incorporation of an auxiliary pose estimation
network, which increases the overall complexity of the model.
To address the aforementioned challenges, we introduce an
Adaptive Occlusion-Aware Network (AOANet) in this paper.
First, we employ Swin-Transformer [9] to acquire multi-scale
features and utilize them according to the semantic hierarchy of different scale features. Motivated by the concept of

1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5068

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

•

To demonstrate the effectiveness of our method, we conducted experiments on five related ReID datasets. The
experimental results demonstrate that our proposed
method achieves state-of-the-art performance.
II. R ELATED W ORK

Fig. 1.

Three different methods for human body positions extraction.

Deep Supervision [10], we adopt the Identity Loss [11] for
supervised training during this stage. Secondly, we obtain
local features for occlusion adaptation in two steps: body
part localization and local feature representation. Notably,
we constructed the Adaptive Position Extractor (APE) by
combining self-attention and cross-attention mechanisms with
keypoints for body part localization. As described in Fig. 1c,
our APE extracts human body parts with strong occlusion
adaptation. Thirdly, taking into account the effect of occlusion on the representation of local features, we design the
Occlusion Awareness Module (OAM). Since APE can extract
occlusion-adaptive local features that accurately reflect the
degree to which the object parts is occluded, we can use OAM
to compute occlusion weights for these local features. Overall,
the contributions of the paper are summarized as follows:
• We propose a novel Adaptive Occlusion-Aware Network (AOANet) for occluded person re-identification.
The AOANet enables feature adaptive representation of
occluded objects through multiple attention mechanisms
with supervised learning of key parts of the human body.
• We design an Adaptive Position Extractor (APE) that
obtains body regions that are adaptive to occlusion by
incorporating self-attention, cross-attention, and human
keypoints. Compared to the methods for pose estimation,
our model does not require pose estimation and thus has
a lower complexity.
• We develop the Occlusion Awareness Module (OAM),
which can adaptively calculate occlusion weights. These
weights have the ability to suppress the occluded part of
the feature and highlight the non-occluded part.

The challenges in ReID primarily stem from factors such
as variations in lighting, viewpoints, attitude, and occlusion.
Current approaches can be broadly categorized into two main
groups: feature representation learning [12], [13], [14] and
deep metric learning [15], [16], [17]. Within the realm of
feature representation learning, methods aim to extract discriminative features for person. On the other hand, deep
metric learning focus on learning similarity metrics capable of
measuring distances between person representations, thereby
enabling accurate identification across diverse surveillance
cameras. Nevertheless, these methods exhibit limited performance in intricate scenes, particularly occlusion changes.
In the context of occluded images, occluded person ReID
aims to match person exhibiting either holistic or occluded
appearances across different cameras. This task becomes
notably challenging owing to the presence of incomplete
information and spatial misalignment. To address these challenges, several approaches focus on information alignment.
Zhuo et al. [5] employed an occluded/non-occluded binary
classification loss to differentiate between occluded and nonoccluded images, leveraging this information to enhance
performance. Miao et al. [6] introduced a Pose-Guided Feature Alignment (PGFA), which utilizes pose landmarks to
mitigate the effects of noisy information from the occluded
regions of the target person. Wang et al. [18] applied graph
convolution to enhance the message-passing of semantic
features while suppressing that of meaningless and noisy elements. Yan et al. [19] introduced a model capable of acquiring
single-scale discriminative global features through the utilization of occlusion-based augmented data. In comparison
with these methodologies, our approach demonstrates superior
adaptability and robustness in acquiring local features. Another
class of approaches emphasizes spatial alignment, typically
seen in Part-based ReID methods. These methods extract
aggregated features from different body parts, placing emphasis on localized features and fine-grained information. Where
body parts can be generated either through specific predefined
semantic parts or with the assistance of pose estimation [11].
As a representative method for the specific predefined
semantic parts, PCB [20] divides the human body horizontally into multiple parts and then trains multiple part-level
classifiers. Moreover, some similar methods [21], [22], [23]
achieved the extraction of more discriminative features than
the original global features. However, these methods rely on
predefined parts and lack of adaptability prone to feature
misalignment.
Pose estimation based method adopted pose estimation
to integrate full-body features with local features, resulting
in commendable performance [24]. To leverage higher-order
information for feature learning and alignment, HOReID [18]
utilized high-order relations and human topology information
to achieve higher performance. BPBreID [25] incorporated

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: ADAPTIVE OCCLUSION-AWARE NETWORK FOR OCCLUDED PERSON ReID

5069

Fig. 2. Overview of our framework. It mainly consists of the Adaptive Position Extractor (APE) and the Occlusion Awareness Module (OAM). After using
Swin-Transformer as a backbone for feature extraction, human body positions are generated by APE. Weighted Average Pooling (WAP) is then used to process
the human body positions to obtain local features, and these local features are aggregated to generate global feature. OAM is used in the inference stage to
further improve recognition accuracy.

a body part focus module to improve the effectiveness of
localized features through external human semantic information. While the methods mentioned above have demonstrated
impressive performance, their vulnerability to noise in pose
estimation algorithm remains a significant concern. In addition, these methods usually require the addition of a pose
estimation model, which often generates larger feature maps
and is more complex than the ReID model itself. Our method
is more tolerant to pose estimation prediction errors, and
the extracted local features show strong adaptability under
occlusion.

body is obtained from four local features through a fully
connected layer:
f g2 = Fcat W,
(2)
h
i
where Fcat = f p1 , f p2 , f p3 , f p4 ∈ R1×(4D) represents the
feature obtained by concatenating four local features, and
W ∈ R(4D)×D is the weight matrix. Then, we present the
procedure for obtaining the four adaptive local features. Adaptive Position Extractor (APE) is developed to determine the
H
W
regions {Pi , i = 1, 2, 3, 4} ⊆ [0, 1]1× 4 × 4 of the four parts
of the human body, which can be calculated:

III. M ETHODOLOGY

P = A P E ( f 4 , F P N ( f 1 , f 2 )) ,

In this section, we first introduce the overall framework of
the proposed method, then present the structure of APE and
OAM in detail, and finally give the design of the loss function.

where P = [P1 ; P2 ; P3 ; P4 ] ∈ [0, 1]4× 4 × 4 . In particular,
f 1 and f 2 , which represent shallow features, contain more
low-level semantic information, such as clothing color and
body shape, and thus they are the best choices for extracting
part locations. We further fused these two layers of features
via the feature pyramid network (FPN) [27]. We adopt the
Weighted Averageh Pooling (WAP)
i method to extract local
features Flocal = f p1 ; f p2 ; f p3 ; f p4 ∈ R4×D , which is defined
as follows:

A. Overall Framework
Our model mainly consists of the Adaptive Position Extractor (APE) and the Occlusion Awareness Module (OAM), and
uses Swin-Transformer as the backbone network. As described
in Fig. 2, the proposed model
ultimately extractso two global
n
and four local features f g1 , f g2 , f p1 , f p2 , f p3 , f p4 ⊆ R1×D
from each person image. Firstly, we utilize the SwinTransformer [9] to obtain four distinct scale feature maps
D
H
W
D
H
W
D
H
W
{ f 1 ∈ R 8 × 4 × 4 , f 2 ∈ R 4 × 8 × 8 , f 3 ∈ R 2 × 16 × 16 , f 4 ∈
H
W
R D× 32 × 32 }. It is well known that f 4 contains a wealth of
high-level semantic information, making it a pivotal source
for both global and local features in our model. The feature
map f g1 for deep supervision can be obtained as follows:
f g1 = G A P ( f 4 ) ,

(1)

where G A P (·) represents Global Average Pooling [26].
Futhermore, global feature map f g2 for representing the human

H

(3)

W

Flocal = G A P (inter p ( f 4 ) ⊙ P) ,

(4)

where inter p (·) means using interpolation for upsampling and
⊙ represents element-wise product. After the training phase,
we introduced the Occlusion Awareness Module (OAM)
during the inference phase to further perceive occlusion information in local features.
B. Adaptive Position Extractor
The overall structure of the Adaptive Position Extractor is
illustrated in Fig. 3(a). In the APE, we define four learnable
part catchers, each representing a specific regions of the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

10571

Adaptive Pseudo-Label Purification and Debiasing
for Unsupervised Visible-Infrared Person
Re-Identification
Xiangbo Yin , Jiangming Shi , Zhizhong Zhang , Member, IEEE, Yuan Xie , Member, IEEE,
and Yanyun Qu , Senior Member, IEEE
Abstract—Unsupervised
Visible-Infrared
Person
ReIdentification (USVI-ReID) aims to match visible and infrared
person images without relying on prior annotations. Recently,
unsupervised contrastive learning methods have become the
mainstream approach for USVI-ReID, leveraging clustering
algorithms to generate pseudo-labels. However, these methods
often suffer from inherent noisy pseudo-labels, which significantly
hinders their performance. To address this challenge, we propose
a Adaptive Pseudo-label Purification and Debiasing (APPD)
framework for USVI-ReID, which is designed to calibrate noisy
pseudo-labels and dynamically detects clean pseudo-labels,
thereby enhancing the model’s performance and reliability.
Specifically, we propose an Adaptive Pseudo-label Calibration
and Division (APCD) module, which calibrates noisy pseudolabels by assessing their reliability and divides pseudo-labels into
clean and noisy subsets, ensuring a more focused and accurate
learning process. Based on the calibrated pseudo-labels, we
develop an Optimal Transport Prototype Matching (OTPM)
module to establish robust cross-modality correspondences.
For clean pseudo-labels, we propose a Debiased Memory
Hybrid Learning (DMHL) module, which jointly captures
modality-specific and modality-invariant information while
addressing sampling bias to enhance feature representation.
To effectively utilize noisy pseudo-labels, we introduce a
Neighbor Relation Learning (NRL) module that mitigates
intra-class variations by exploring neighbor relationships in
the feature space. Comprehensive experiments conducted on
two widely recognized USVI-ReID benchmarks demonstrate
that APPD achieves state-of-the-art performance, significantly
outperforming existing methods. The source code will be made
available at https://github.com/XiangboYin/RPNR
Received 17 January 2025; revised 15 April 2025; accepted 13 May 2025.
Date of publication 20 May 2025; date of current version 6 October 2025.
This work was supported by in part by the National Natural Science Foundation of China under Grant 62176224, Grant 62222602, Grant 62176092,
Grant U23A20343, and Grant 62476090; in part by the Natural Science
Foundation of Shanghai under Grant 23ZR1420400; in part by Shanghai
Sailing Program under Grant 23YF1410500; in part by CCF-Tencent under
Grant RAGR20240122; and in part by the Science and Technology on Sonar
Laboratory under Grant 2024-JCJQ-LB-32/07. This article was recommended
by Associate Editor Z. Mao. (Jiangming Shi contributed equally to this work.)
(Corresponding author: Yanyun Qu.)
Xiangbo Yin and Yanyun Qu are with the School of Informatics, Xiamen University, Xiamen 361005, China (e-mail: xiangboyin@stu.xmu.edu.cn;
yyqu@xmu.edu.cn).
Jiangming Shi is with the Institute of Artificial Intelligence, Xiamen
University, Xiamen 361005, China (e-mail: jiangming.shi@outlook.com).
Zhizhong Zhang is with the School of Computer Science and Technology, East China Normal University, Shanghai 200062, China (e-mail:
zzzhang@cs.ecnu.edu.cn).
Yuan Xie is with the School of Computer Science and Technology, East
China Normal University, Shanghai 200062, China, and also with Chongqing
Institute, East China Normal University, Chongqing 401120, China (e-mail:
yxie@cs.ecnu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3571976

Index Terms—USVI-ReID, noisy labels, optimal transport,
debiased contrastive learning, neighbor relation learning.

I. I NTRODUCTION
HE increasing demand for intelligent security has led
to the widespread adoption of smart monitoring sensor devices designed for 24-hour surveillance. The distinct
imaging principles of sensor devices during daytime and
nighttime result in significant differences between visible
and infrared images. This discrepancy has fueled growing
interest in research on visible-infrared person re-identification
(VI-ReID), which aims to accurately match visible and
infrared pedestrian images, allowing for the retrieval of a
pedestrian image from one modality based on a query from
another [1], [2], [3], [4], [5]. However, the substantial disparity
between these two modalities poses a significant challenge
for this task. Recently, numerous VI-ReID methods [6], [7],
[8], [9], [10] have focused on reducing cross-modality discrepancies by aligning visible and infrared images at both
the image and feature levels, achieving notable performance
gains. However, these approaches are highly dependent on
well-annotated cross-modality datasets, which are costly and
labor-intensive to obtain in real-world applications. As a
result, unsupervised visible-infrared person re-identification
(USVI-ReID) has garnered increasing attention.
The primary challenges of USVI-ReID lie in generating
robust pseudo-labels. Existing USVI-ReID methods [11], [12],
[13], [14] predominantly adopt the DCL [15] framework,
which utilizes DBSCAN for pseudo-label generation and
establishes cross-modality correspondences based on these
pseudo-labels. Pseudo-labels, being the result of clustering,
are inherently prone to noise. These noisy pseudo-labels can
misdirect the model, resulting in distorted learning of data
distributions and suboptimal feature representations, thereby
undermining overall performance. To address the impact of
noisy pseudo-labels, MMM [14] computes confidence scores
based on the classifier loss, using these scores to reduce
the influence of noisy labels. PGM [11], on the other hand,
minimizes the effect of noisy labels by alternately applying
two unidirectional metric losses, which helps prevent the
rapid emergence of inaccurate pseudo-labels. However, neither
method explicitly refines noisy pseudo-labels into cleaner
ones, limiting the model’s ability to fully exploit hard-todiscriminate features.
As shown in Fig. 1, noisy pseudo-labels can lead to three
negative influences for USVI-ReID: a) The centroid memory

T

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10572

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

Fig. 1. The negative influence of the noisy pseudo-labels. (a) Noisy pseudo-labels may initialize an unreliable memory bank, misguiding the model to train
in the wrong direction. (b) Noisy pseudo-labels can lead to unreliable cross-modality correspondences, which hinder the learning of robust cross-modality
representations. (c) “Sampling bias” problem caused by noisy pseudo-labels, i.e., the negative samples from negative sets may not be true negative samples,
which interferes with the performance improvement of the model.

bank is initialized and continuously updated by pseudo-labels.
However, since pseudo-labels inherently contain noise, the
memory bank fails to reflect the true feature distribution,
making it unreliable and leading the model to deviate from
the correct training. b) Reliable cross-modality correspondences are crucial for USVI-ReID. However, the existence
of noisy pseudo-labels may lead to unreliable cross-modality
correspondences, which hinder the learning of robust crossmodality representation. c) The sampling bias phenomenon
caused by noisy pseudo-labels, having not been explored by
previous methods, refers to negative samples from negative
sets that may not be true negative samples, which can empirically lead to a significant performance drop [16]. Therefore,
calibrating and identifying cleaner pseudo-labels can help
mitigate the aforementioned issues to a certain extent.
In this paper, we propose the Adaptive Pseudo-label
Purification and Debiasing (APPD) framework to address
the above three negative influences of noisy pseudo-labels.
Specifically, to obtain robust pseudo-labels, we propose an
Adaptive Pseudo-label Calibration and Division (APCD) module. Unlike traditional methods that directly use pseudo-labels
produced by clustering algorithms to train the model, APCD
first calibrates noisy pseudo-labels into more robust ones and
subsequently isolates clean and noisy pseudo-labels. These
calibrated pseudo-labels are then used in the Optimal Transport
Prototype Matching (OTPM) module to establish reliable
cross-modality alignments. To reduce cross-modality gaps, the
Debiased Memory Hybrid Learning (DMHL) module captures
both modality-specific and modality-invariant information,
while addressing sampling biases caused by noisy labels.
To fully utilize noisy pseudo-labels, the Neighbor Relation
Learning (NRL) module models pair-wise relationships in the

feature space, encouraging closer clustering of similar samples
and mitigating intra-class variations.
In conclusion, the main contributions of our method can be
summarized as follows:
• We propose the Adaptive Pseudo-label Purification and
Debiasing (APPD) framework to address the negative
influences of noisy pseudo-labels in USVI-ReID. Compared to the conference version, this paper has several
critical improvements: adaptive pseudo-label division,
debiased contrastive learning, and more detailed experiments.
• We design the Adaptive Pseudo-label Calibration and
Division (APCD) module to calibrate noisy pseudo-labels
into more robust ones and isolate clean and noisy pseudolabels for effective learning.
• We propose the Optimal Transport Prototype Matching
(OTPM) module to establish reliable cross-modality correspondences based on calibrated noisy pseudo-labels.
• We propose Debiased Memory Hybrid Learning (DMHL)
and Neighbor Relation Learning (NRL) modules to alleviate inter- and intra-modality gaps.
• Experiments on mainstream datasets demonstrate the
superiority of our method compared with existing methods, and APPD generates higher-quality pseudo-labels
than other methods. In addition, the performance of
APPD for SVI-ReID and SSVI-ReID is promising.
II. R ELATED W ORK
A. Unsupervised Single-Modality Person ReID
Unsupervised single-modality person ReID aims to extract
discriminative identity features from unlabeled person ReID

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

datasets. Many current unsupervised methods heavily rely on
pseudo-labels, employing an iterative process that alternates
between generating pseudo-labels and representation learning
[17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27]. The
Cluster-Contrast framework [28], [29] introduces a strategy
that utilizes unique centroid representations for cluster-level
contrastive learning, supported by a momentum update mechanism to maintain feature consistency. However, using a single
proxy to represent an entire cluster may introduce biases. To
address this limitation, multi-proxy methods [27], [30] have
been developed to enhance robustness. Given the inherent
noise in pseudo-labels, label refinement strategies [17], [31],
[32] have been proposed to improve their reliability. Although
these techniques perform well in single-modality unsupervised
ReID, their direct application to unsupervised VI-ReID faces
significant obstacles due to the large cross-modality gap.

10573

III. M ETHODOLOGY
A. Notation Definition
Given an unlabeled visible-infrared person re-identification
dataset D = {DV , DR }, where DV = {xiv | i = 1, 2, . . ., N v }
represents the unlabeled visible dataset with N v visible samples and DR = {xir | i = 1, 2, . . ., N r } denotes the unlabeled
infrared dataset with N r infrared samples. For the USVI-ReID
task, the objective is to train a robust network fθ to map
an instance xit from D into an embedding space F, where
t ∈ {v, r} denotes the visible and infrared modality. Therefore,
we can employ the encoder fθ to extract d dimensional visible
features F v = { fiv | i = 1, 2, . . ., N v } and infrared features
F r = { fir | i = 1, 2, . . ., N r }, where fit ∈ Rd .
B. Overview

B. Unsupervised Visible-Infrared Person ReID
Unsupervised visible-infrared person re-identification
(USVI-ReID) has drawn significant attention for its capability
to learn both modality-specific and modality-invariant
features without requiring cross-modality annotations. Most
USVI-ReID methods [11], [12], [13], [33], [34], [35],
[36], [37] follow the DCL [15] framework, which typically
involves two steps: (1) generating pseudo-labels through
clustering and (2) leveraging these pseudo-labels to establish
cross-modality correspondences. Methods like PGM [11]
and MBCCM [33] employ multi-stage graph matching by
constructing bipartite graphs, while OTLA [38] and DOTLA
[12] use Optimal Transport to map pseudo-labels between
modalities at the instance level. However, the inherent noise
in pseudo-labels often leads to unreliable cross-modality
correspondences, underscoring the need for strategies that
produce higher-quality pseudo-labels for USVI-ReID tasks.
C. Learning With Noisy Labels
Label noise has been shown to negatively impact the training of deep neural networks [39], [40], [41], [42]. Existing
strategies for addressing noisy labels can be broadly categorized into two approaches: label correction and sample
selection. Label correction methods [43], [44], [45], [46] focus
on using model predictions to refine noisy labels. For instance,
SMP [47] introduces an iterative learning framework to relabel
noisy samples and train the network directly on the noisy
dataset without additional clean data. Similarly, [48] employs
back-propagation to probabilistically update and correct image
labels alongside network training. In contrast, sample selection
methods [49], [50], [51] aim to identify and retain clean
samples while excluding noisy ones during training. NCE [52]
filters clean samples based on neighbor information, while
CBS [53] employs confidence-based sample augmentation to
improve the reliability of selected clean data. For the USVIReID task, pseudo-labels generated by clustering algorithms
are inherently noisy. Therefore, refining these noisy pseudolabels is essential for enhancing the model’s performance in
this domain. In this paper, we employ the calibration-thensampling strategy to shield the model from the effect of noisy
data.

The overall framework of our APPD is illustrated in Fig. 2.
Initially, we utilize the DBSCAN [54] algorithm to cluster visible and infrared features, respectively. Following the clustering
process, pseudo-label yti ∈ {1, 2, . . ., Y t } is assigned to the i-th
image from modality t, where Y t represents the total number
of clusters. Given the inherent noise in pseudo-labels, we
introduce an effective calibration-then-division module called
Adaptive Pseudo-label Calibration and Division (APCD) to
calibrate and sample cleaner pseudo-labels. First, we refine
the noisy pseudo-labels into more robust ones and assign
these calibrated pseudo-labels ŷti for each sample to obtain
Nv
Nr
the “labeled” dataset D̃V = {(xiv , ŷvi )}i=1
and D̃R = {(xir , ŷri )}i=1
.
Since the training data may still contain some noise even
after calibration, relying solely on all the data for training can
significantly impair the model’s generalization and robustness.
Therefore, we divide the visible and infrared pseudo-labels
into clean and noisy subsets, denoted as S v = S cv ∪ S nv and
S r = S cr ∪ S nr , respectively. After that, the clean set S ct is used
to perform Debiased Memory Hybrid Learning introduced in
Sec. III-E while the noisy set S nt is used to carry out Neighbor
Relation Learning introduced in Sec. III-F.
Notably, the pseudo-labels generated by two separate
clusterings for visible and infrared samples reveal a misalignment. To resolve this, we introduce the Optimal Transport
Prototype Matching (OTPM) module, which aligns visible
and infrared prototypes through optimal transport at clusterlevel, ensuring more accurate cross-modality correspondences.
Learning modality-invariant features is essential for effective
cross-modality matching. To further exploit modality-invariant
information and reduce sampling bias, we propose the Debiased Memory Hybrid Learning (DMHL) module. In this
module, we merge visible and infrared prototypes into new
modality-hybrid prototypes to better mitigate the substantial
cross-modality gaps. To address sampling bias introduced by
noisy pseudo-labels, we integrate Debiased Contrastive Learning, enabling noise-tolerant contrastive learning for improved
robustness. Moreover, DMHL does not account for potential
interactions among all noisy samples. To address this limitation, we introduce the Neighbor Relation Learning (NRL)
module, specifically designed to capture and model the complex interactions across the noisy sample set.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112622

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Adversarial ﬂow-based generative models for visible-to-Infrared person
re-Identiﬁcation
Honghu Pan
a

a , Yongyong Chen a , Xin Li b , Zhenyu He a,∗

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, Shenzhen, 518055, China

b Peng Cheng Laboratory, Shenzhen, 518055, China

a r t i c l e

i n f o

Keywords:
Visible-to-Infrared person re-Identiﬁcation
Data augmentation
Flow-based generative model
Adversarial training

a b s t r a c t
The task of visible-to-infrared (V2I) person re-identiﬁcation (ReID) presents greater challenges than visible-tovisible (V2V) ReID, primarily due to limited availability of training samples and signiﬁcant cross-modal discrepancy. To address these challenges, we propose Flow2Flow, a uniﬁed framework designed to simultaneously
expand training samples and generate cross-modal images for V2I person ReID. Flow2Flow operates by learning bijective transformations from both the visible and infrared image domains to a shared isotropic Gaussian
domain, utilizing invertible ﬂow-based generators for each modality. This framework enables the generation
of training samples by transforming latent Gaussian noise into visible or infrared images and the generation of
cross-modal images by transforming existing modality images through the latent Gaussian space into the target
modality. To ensure proper identity and modality alignment of the generated images, we devise two adversarial
training strategies. Speciﬁcally, we design an image encoder and a modality discriminator for each modality. The
image encoder enhances the similarity between generated images and real images of the same identity through
identity adversarial training, while the modality discriminator ensures the generated images are indistinguishable from real images through modality adversarial training. Experimental results on the SYSU-MM01 and RegDB
datasets demonstrate that both training sample generation and cross-modal image generation substantially improve V2I ReID accuracy.

1. Introduction
Person re-identiﬁcation (ReID) aims to match pedestrian images captured by non-overlapping cameras, which is achieved by training a deep
neural network to learn discriminative pedestrian representations, enabling cross-camera matching through feature similarity comparison.
Recent advancements [1,2] in person ReID have achieved human-level
accuracy on large-scale datasets [3,4]. However, these methods typically assume that pedestrian images are captured by visible-spectrum
cameras in well-lit environments, limiting their eﬀectiveness in nighttime surveillance scenarios. Given that infrared radiation is unaﬀected
by lighting conditions, visible-to-infrared (V2I) person ReID [5–10], a
cross-spectrum or cross-modal matching task, has garnered signiﬁcant
attention within the computer vision community. V2I person ReID is
a critical task for 24/7 intelligent surveillance systems, enabling crossmodal matching between daytime and nighttime images. It has a wide
range of applications in public security (tracking suspects or miss-

ing persons across diﬀerent lighting conditions), autonomous driving
(enhancing pedestrian detection in low-light environments), and smart
cities (enabling seamless person retrieval in day-night surveillance networks).
Unlike traditional single-modality ReID, V2I ReID must overcome
signiﬁcant cross-modal discrepancies, making it more challenging yet
practically indispensable. Despite recent progress in V2I ReID, the task
remains highly challenging for two primary reasons. First, the modality
discrepancy between the visible and infrared spectra is substantial. Visible (RGB) and infrared (grayscale, heat-based) images exhibit vast differences in texture, color, and illumination, making feature alignment
diﬃcult. Existing V2I ReID methods [11–14] mainly focus on reducing this cross-modal discrepancy through cross-modal image generation,
often employing generative adversarial networks (GANs) to generate
the target modality from the existing one. However, it is challenging
for GAN-based generators to transform infrared images, which contain
limited information, into visible images, which are information-rich.

∗ Corresponding author.

E-mail address: 19b951002@stu.hit.edu.cn, zhenyuhe@hit.edu.cn (Z. He).

https://doi.org/10.1016/j.patcog.2025.112622
Received 12 February 2025; Received in revised form 14 October 2025; Accepted 15 October 2025
Available online 29 October 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112622

H. Pan et al.

This challenge motivates the need for learning bijective transformations between the visible and infrared modalities. Second, the number of
training images in V2I datasets [5,6] is insuﬃcient to train deep models
eﬀectively. Compared to V2V ReId, most datasets in this task contain
few visible and infrared images, restricting supervised learning. For instance, the training sets of SYSU-MM01 [5] and RegDB [6] contain only
9929 and 2060 infrared images, respectively. Obtaining pedestrian images in real scenarios requires lots of manual collections and annotations, thereby, this paper proposes to expand training datasets by generative models.
In this paper, we introduce Flow2Flow, a uniﬁed framework that facilitates both cross-modal generation and training sample generation.
Our framework comprises two ﬂow-based generative models [15,16],
i.e., a visible ﬂow and an infrared ﬂow, each of which learns bijective transformations from the visible or infrared image domain to a
shared isotropic Gaussian domain. This design allows for the generation of visible or infrared training samples via forward ﬂow propagation from the latent noise domain to the respective image domain.
Additionally, target-modality images can be generated from sourcemodality images by transforming them from the source-modality domain through the Gaussian noise domain to the target-modality domain,
enabling cross-modal generation. Fig. 1 provides a schematic overview
of the training sample generation and cross-modal image generation
processes.
To ensure invertibility and exact log-likelihood computation, existing ﬂow models [15,16] typically employ multiple 1 × 1 convolutional
layers and linear coupling layers, which leads to insuﬃcient nonlinearity. To address this, we introduce an additional invertible activation layer in the ﬁnal block of both the visible and infrared ﬂows to
enhance model nonlinearity. Furthermore, we propose an identity adversarial training strategy and a modality adversarial training strategy
to ensure that the generated images align with speciﬁc identities and
modalities. For adversarial training, we implement two discriminators
for each modality: an image encoder for identity alignment and a modality discriminator for modality alignment. To achieve identity alignment
between real and generated images, we minimize the distance between
their encoded features during generator training, while maximizing it
during discriminator training. Meanwhile, the modality discriminators
work to distinguish whether images are generated or belong to a speciﬁc
real modality.
Experimental results on SYSU-MM01 [5] and RegDB [6] demonstrate that our method can eﬀectively improve the V2I ReID performance. Models trained on real and generated samples outperform models trained on real samples; meanwhile, models trained on dual-modal
images are superior to those trained on single-modal images. The main
contributions of this paper are threefold:

Fig. 1. Diagram illustrating (a) training sample generation and (b) (c) crossmodal image generation. In these diagrams, images outlined by blue boxes represent fake training samples, while those by red boxes represent cross-modal
images. The proposed Flow2Flow framework consists of a visible ﬂow and an
infrared ﬂow, each of which learns bijective transformation from the visible or
infrared image domain to a shared isotropic Gaussian domain. In (a), training
sample generation is accomplished through the forward propagation of ﬂow
models from the Gaussian domain to the image domains. In (b) and (c), crossmodal image generation is achieved by transforming existing modality images
(𝑥𝑉𝑖 or 𝑥𝑅
) through the Gaussian noise space to the target modality images (𝑥̂ 𝑉𝑖 2𝑅
𝑖
or 𝑥̂ 𝑅2𝑉
). (For interpretation of the references to colour in this ﬁgure legend, the
𝑖
reader is referred to the web version of this article.)

2. Related works
2.1. Visible-to-visible person reid
The V2V person ReID [17,18] is a single-modality image retrieval
task, which aims to enlarge the inter-class variance and reduce the intraclass variance. To this end, existing methods mainly consider three levels of factors: objective-level, network-level, and data-level. For the objectives or loss functions, TriNet [2] proposed the hard triplet mining
strategy on the basis of triplet loss to learn pedestrian representations;
BoT [19] combined the cross entropy loss and triplet loss to train network; moreover, the center loss [20] and angular loss [21] have also
been successfully applied in the V2V person ReID. For the network, early
works [2] learned the global features from pedestrian images via a single CNN branch. Next, the multi-branch architecture has been adopted
to learn the multi-granularity or part-level features [22,23]. Furthermore, data augmentation or generation [24,25] could also improve the
ReID accuracy, which belongs to the data-based category. For example,
PN-GAN [25] generated multi-pose pedestrian images via GAN model,
which could reduce the pedestrian view variance; DG-Net [26] disentangled the pedestrian images into appearance and pose structure, and
generated multi-pose pedestrian images for each sample by changing
the pose structure; JVTC [24] conducted the online data augmentation
for contrastive learning, in which the mesh projections were taken as
the references to generate multi-view images.

•

We propose Flow2Flow, a uniﬁed framework, to jointly generate
training samples and cross-modal images, which leverages a visible
ﬂow and an infrared ﬂow to learn bijective transformations from
image domains to a shared Gaussian domain;
• For the purpose of identity alignment and modality alignment of generated images, we develop an image encoder and a modality discriminator for each modality to perform the identity adversarial training
and modality adversarial training, respectively;
• Experimental results show that both the training sample generation
and cross-modal generation can eﬀectively improve the performance
of existing V2I ReID baselines, demonstrating the eﬀectiveness and
generalization of Flow2Flow.

2.2. Visible-to-infrared person reid
The remainder of this paper is organized as follows: Section 2 introduces recent literature related to this paper; Section 3 simply reviews
theoretical backgrounds of the ﬂow-based generative models; Section 4
elaborates the Flow2Flow model in detail; Section 5 presents the ablation studies, visualizations and comparisons with the SOTA; Section 6
draws brief conclusions.

The V2I person ReID enables cross-spectrum pedestrian retrieval,
whose crux is to reduce the large cross-modal discrepancy. Existing
V2I ReID methods mainly have two techniques to reduce the modal
discrepancy: 1) learning the modality-shared pedestrian representation
and 2) compensating information of target modality via generative
2

Pattern Recognition 172 (2026) 112622

H. Pan et al.

Fig. 2. Framework of Flow2Flow, in which the blue arrows denote training sample generation (𝑧𝑉𝑖 → 𝑥̂ 𝑉𝑖 2𝑉 and 𝑧𝑅
→ 𝑥̂ 𝑅2𝑅
), and the red arrows denote cross-modal
𝑖
𝑖
𝑅2𝑉
𝑅 →𝑥
generation (𝑥𝑉𝑖 →𝑧𝑉𝑖 →𝑥̂ 𝑉𝑖 2𝑅 and 𝑥𝑅
→𝑧
̂
).
It
consists
of
visible
and
infrared
ﬂow-based
generators

and

,
𝑉
𝑅 visible and infrared encoders 𝑉 and 𝑅 , visible
𝑖
𝑖
𝑖
and infrared modality discriminators 𝑉 and 𝑅 : 𝑉 or 𝑅 learns a bijective transformation from the visible domain 𝑃 (𝑋 𝑉 ) or infrared domain 𝑃 (𝑋 𝑅 ) to the
latent Gaussian domain Π(𝑍);  and  encourage the generated images to match true identity and modality by identity and modality adversarial training. (For
interpretation of the references to colour in this ﬁgure legend, the reader is referred to the web version of this article.)

models [27–29]. The modality-shared ones [7,30–37] projected the visible and infrared pedestrian images into a shared Euclidean space, in
which the intra-class similarity and inter-class similarity are maximized
and minimized, respectively. For example, DDAG [38] proposed a dualattentive aggregation learning method to mine both intra-modality partlevel and cross-modality graph-level contextual cues. MPANet [7] aimed
to capture the nuances of cross-modal images via a modality alleviation module and a pattern alignment module. SGIEL [32] leveraged
the body shape of pedestrians as the signiﬁcant modality-shared cues,
and devised a shape-erased feature learning paradigm to decorrelate
modality-shared features in two orthogonal subspaces. The modality
compensation ones [11,12,39,40] usually generated target modality information from existing modality data: DDRL [41] proposed an imagelevel sub-network based on GAN model, which could translate a visible
(infrared) image to a corresponding infrared (visible) one; cmPIG [12]
employed the set-level alignment information to generate instance alignment cross-modal paired-images; FMCNet [42] utilized the feature-level
modality compensation to reduce modality discrepancy, which generated the cross-modal features rather than images.

whether the data is true or fake to beat the generator. Recently, the
GAN architectures have been heavily reﬁned to adapt various application scenarios. For instance, the Conditional GAN [46,47] could generate samples corresponding to speciﬁc condition labels; CycleGAN [28]
enabled the unpaired cross-domain image translation by the cycle consistency loss. Meanwhile, the GAN model also showed its priority in
the V2I person ReID [11,41,42] and V2I person ReID areas [24,25].
Unlike the ﬂow-based model [15,16] which could exactly compute the
log-likelihood of true data, GAN model implicitly minimizes the KL divergence between the true data and data generated from noises. To make
the generated data indistinguishable from the real data, training a GAN
model pursues an equilibrium between the generator and discriminator,
which requires careful experimental setup tuning.
3. Preliminaries
The ﬂow-based generative model aims to learn a bijective transformation from a complex distribution 𝑋 ∼ 𝑃 (𝑋) to a simple distribution
𝑍 ∼ Π(𝑍) with a known probability density function, in which 𝑋 denotes the true training data and Π(𝑍) is usually a Gaussian distribution.
For the purpose of bijective mapping, the ﬂow-based model consists of
a sequence of invertible generators  = 1 ∙ ⋯ ∙ 𝐿 :

2.3. Flow-based generative models
The ﬂow-based generative model constructs an invertible mapping
from the complex distribution of true data to a simple distribution
(e.g., isotropic Gaussian distribution). Layers in the ﬂow-based model
should be carefully designed to match the goal of invertibility and exact log-likelihood computation. RealNVP [15] proposed the aﬃne coupling layer, which could easily compute the determinant of Jacobian
matrix; Glow [16] presented an invertible 1 × 1 convolution layer, meanwhile the LU decomposition was utilized to speed up the computation of determinants; cAttnFlow [43] introduced the invertible attentions to increase the nonlinearity of ﬂow-based model. Recently, a great
number of works have extended the ﬂow-based model image generation [16,44,45]. For example, on the image super-resolution ﬁeld, SRFlow [44] and HCFlow [45] took the low-resolution images as the condition, and thus learned the high-resolution images via a conditional
ﬂow.

𝑥𝑖 = (𝑧𝑖 ), 𝑧𝑖 = −1 (𝑥𝑖 ).

(1)

By the change of variable formula, 𝑃 (𝑋) and Π(𝑍) satisfy the following
transformation:
|
|
𝑃 (𝑋) = Π(𝑍)|det(𝐽−1 )|,
|
|

(2)

where det(𝐽−1 ) denotes the determinant of Jacobian matrix. Then the
objective of max{log(𝑃 (𝑋))} can be converted to:
𝐿
∑
∑
|
|
max{ log(Π(𝑧𝑖 )) +
log ||det(𝐽−1 )||}.
𝑙
|
|
𝑖
𝑙=1

(3)

From Eq. (1), Eq. (2) and Eq. (3), we could know that the training
process of the ﬂow-based model follows the reverse propagation, and
the inference or generation process follows the forward propagation.
A standard ﬂow-based model mainly contains two categories of layers: invertible 1 × 1 convolution layer [16] and aﬃne coupling layer [15,
48]. For a single generator 𝑙 in , the reverse and forward projection
of the 1 × 1 convolution layer has the following expression:

2.4. Generative adversarial network
The ﬁrst GAN model was proposed in [27], which consists of a generator and a discriminator, and they could improve each other by the
adversarial training. In GAN model, the generator generates samples
from noise variables with a known probability density function (PDF)
and tries to fool the discriminator, and the discriminator distinguishes

<𝑙>
𝑧<𝑙−1>
= 𝑊𝑙 𝑧<𝑙>
= 𝑊𝑙−1 𝑧<𝑙−1>
,
𝑖
𝑖 , 𝑧𝑖
𝑖

(4)

where 𝑍 <0> and 𝑍 <𝐿> denote 𝑍 and 𝑋, respectively. The design of the
aﬃne coupling layer should allow 1) invertible transformation and 2)
3


 succeeded in 0ms:
Pattern Recognition 163 (2025) 111443

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Adaptive transformer with Pyramid Fusion for cloth-changing Person
Re-Identification
Guoqing Zhang a,b , Jieqiong Zhou a , Yuhui Zheng a , Gaven Martin c , Ruili Wang b,d,e ,∗
a

School of Computer Science, Nanjing University of Information Science and Technology, Nanjing, China
School of Mathematical and Computational Sciences, Massey University, Auckland, New Zealand
c
Institute for Advanced Study, Massey University, Auckland, New Zealand
d
School of Computer Science, University of Nottingham, Ningbo, China
e
School of Data Science and Artificial Intelligence , Wenzhou University of Technology, Wenzhou, China
b

ARTICLE

INFO

Keywords:
Cloth changing
Person re-identification
Vision transformer

ABSTRACT
Recently, Transformer-based methods have made great progress in person re-identification (Re-ID), especially
in handling identity changes in clothing-changing scenarios. Most current studies usually use biometric
information-assisted methods such as human pose estimation to enhance the local perception ability of clotheschanging Re-ID. However, it is usually difficult for them to establish the connection between local biometric
information and global identity semantics during training, resulting in the lack of local perception ability
during the inference phase, which limits the improvement of model performance. In this paper, we propose
a Transformer-based Adaptive-Aware Attention and Pyramid Fusion Network (𝐴3 𝑃 𝐹 𝑁) for CC Re-ID, which
can capture and integrate multi-scale visual information to enhance recognition ability. Firstly, to improve the
information utilization efficiency of the model in cloth-changing scenarios, we propose a Multi-Layer Dynamic
Concentration module (MLDC) to evaluate the importance features at each layer in real time and reduce the
computational overlap between related layers. Secondly, we propose a Local Pyramid Aggregation Module
(LPAM) to extract multi-scale features, aiming to maintain global perceptual capability and focus on key local
information. In this module, we also combine the Fast Fourier Transform (FFT) with self-attention mechanism
to more effectively identify and analyze pedestrian gait and other structural details in the frequency domain
and reduce the computational complexity of processing high-dimensional data in the self-attention mechanism.
Finally, we build a new dataset incorporating diverse atmospheric conditions (for instance wind and rain) to
more realistically simulate natural scenarios for the changing of clothes. Extensive experiments on multiple
cloth-changing datasets clearly confirm the superior performance of 𝐴3 𝑃 𝐹 𝑁. The dataset and related code are
available on the website: https://github.com/jieqiongz1999/vcclothes-w-r.

1. Introduction
Person Re-Identification (Re-ID) strives to identify the same person across different cameras and plays a vital role in public safety.
However, to date most person Re-ID methods [1–3] use clothing as
discriminative information to deal with obstacles such as item occlusion
and perspective changes. However, in a real-world scenario, such as
criminal tracking, clothing change is a common evasion strategy, and
traditional short-term Re-ID technology cannot effectively deal with
this, as shown in Fig. 1. Therefore, it is important to study more
targeted cloth-changing Person Re-ID methods.

The CC Re-ID task aims to extract identity information unaffected
by clothing changes [4–6]. One category of approaches focus on identifying clothing-independent features, such as body outlines, posture
key-points and gait information. For example, Yang et al. [7] proposed
a network that can adapt to clothing changes around pedestrian silhouette sketches, but is affected by environmental factors such as lighting
and occlusion, and may ignore key details such as facial features.
The other category of approaches focus on separating identity and
clothing features such as GAN and semantic-guided clothing erasure
network [8]. However this usually brings challenges such as additional
computational overhead, high computational requirements, and strong
dependence on data quality.

∗ Corresponding author at: School of Mathematical and Computational Sciences, Massey University, Auckland, New Zealand.

E-mail addresses: guoqingzhang@nuist.edu.cn (G. Zhang), jieqiongz331@nuist.edu.cn (J. Zhou), zheng_yuhui@nuist.edu.cn (Y. Zheng),
G.J.Martin@massey.ac.nz (G. Martin), ruili.wang@massey.ac.nz (R. Wang).
https://doi.org/10.1016/j.patcog.2025.111443
Received 16 July 2024; Received in revised form 15 January 2025; Accepted 5 February 2025
Available online 12 February 2025
0031-3203/© 2025 Published by Elsevier Ltd.

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Fig. 1. Visualization of the Top-8 ranking lists generated by MGN [3] on the MSMT17 and Celeb-reID datasets. Images with red boxes indicate incorrect matches.

Recently, Vision Transformer [9] (ViT) has demonstrate remarkable
performance in various computer vision tasks [10,11] with its multihead self-attention mechanism to effectively capture inter dependencies
within an image. However, a common limitation of the Transformer
architecture is tendency to utilize the output of a specific layer for
representation, often neglecting other valuable information embedded
in the other layers. Fig. 2 shows the attention map visualization of the
first three layers and the last three layers of the ViT model. It can be
clearly observed that there are significant differences in the focus of
attention in different layers. In addition, although existing methods
often improve the local perception ability of pedestrian Re-ID after
changing clothes through strategies such as posture estimation, but how
to effectively coordinate local details with global semantic information
remains a challenge.
To mitigate these limitations, we propose a Transformer based
Adaptive-Aware Attention and Pyramid Fusion Network (𝐴3 𝑃 𝐹 𝑁) for
CC Re-ID. We firstly design a Multi-Layer Dynamic Concentration module (MLDC) to integrate the characteristics of each layer of ViT and
reduce the redundancy between layers. MLDC fuses different layer features through weighting and adjusts the importance of each layer in real
time. Subsequently, recognizing that each layer of the ViT model concentrates on different aspects of the image, we propose a Local Pyramid
Aggregation Module (LPAM) to extract multi-scale features, thereby
maintaining attention to global perception and key local information.
In this module, we also innovatively integrate a Fast Fourier Transform
(FFT) into the self-attention mechanism to effectively identify subtle
pedestrian differences in the frequency domain (such things as gait
and clothing texture) to improve both computational efficiency and
accuracy. Finally, since the existing Re-ID datasets do not consider the
impact of weather, we propose the VC-Clothes-W&R dataset to fill this
gap by introducing wind and rain elements.
Our primary contributions are the following:

Fig. 2. Attention maps of the first three layers (first row) and the last three layers
(second row) of ViT.

2. Related works
2.1. Classical person Re-ID
Current research has focused on solving problems such as lighting
changes [12], occlusion [13], and cross resolution [14]. Jiang et al.
[11] proposed a novel cross-modal Transformer (CMT) that jointly
explores modal-level alignment modules and instance-level modules for
visible-infrared person Re-ID, aiming to alleviate the loss of modalityspecific information caused by existing methods integrating different
modalities into a unified feature space. A Pose-guided Feature Decoupling (PFD) method proposed by Wang et al. [13] utilizes pose
information to effectively decouple semantic components (such as human body or joint parts), and aligns unoccluded parts accordingly.
Zhang et al. [14] proposed a Deep High-Resolution Pseudo-Siamese
Framework (PS-HRNet), which introduces the VDSR-CA module to
restore the resolution of low-resolution images and fully utilize the
different channel information of feature maps, while using the new
representation in HRNet to extract distinguishing features, thereby
achieving excellent performance in cross-resolution scenarios. In addition, unsupervised Re-ID is also a key research focus: DHA [15]
proposed an auto encoder-based method to generate deep latent attributes without extensive annotations, thus enhancing the ability to
extract features from sparse but discriminative data to identify individuals within clues and reduce reliance on labeled data. IPES-GAN
[16] adopts loop generation to adaptively balance environment and
identity features to achieve domain adaptation, which significantly
improves the robustness to environmental changes and camera settings
in different domains.

∙ We propose a Transformer-based Adaptive Aware Attention and
Pyramid Fusion Network for CC Re-ID;
∙ We integrate the Fast Fourier Transform into the self-attention
mechanism to improve the model’s ability of identifying pedestrian features in the frequency domain and optimize computing
efficiency;
∙ We propose the VC-Clothes-W&R dataset, which fills the missing
natural weather factors in existing pedestrian re-identification
datasets by introducing wind and rain elements.
The remainder of the paper is organized as follow: Section 2
presents some related works and the details of our proposed framework
are described in Section 3. Section 4 outlines the experimental setup
and presents the results of extensive experiments on diverse datasets.
Ablation studies are reviewed in Section 5, and Section 6 presents our
conclusions.
2

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

2.2. Person Re-ID under intensive cloth variations

3.1. Multi-layer dynamical concentration module

As public safety concerns become increasingly prominent, especially in the fields of monitoring and safety, there is a pressing need
for effective identification of potential threats. Therefore, accurate
identification of individuals who change their attire becomes crucial
to promptly detect and intervene in potential security risks. These
concerns have spurred many scholars to conduct in-depth research
on CC Re-ID. In recent years, some related cloth-changing datasets
have been released, such as VC-Clothes [17], Celeb-reID [18], CelebreID-light [19] and NKUP [20]. In these datasets, the same individual
switches among multiple outfits, and wears various accessories, such
as sunglasses, scarves, backpacks, etc. Frequent changes of clothing
greatly reduce the reliability of traditional appearance-based matching
methods.
To cope with the challenges brought by changing clothes, some
works learn clothing-independent features with the help of identityrelated auxiliary biological cues. For example, Hong et al. [4] proposed
a shape–appearance mutual learning framework (FSAM), which is a
dual-stream structure that acquires the detailed discriminative body
shape information in shape stream and enriches the appearance stream
with non-fabric-related details. Zhang et al. [21] proposed a novel
Multi-Biometric Unified Network (MBUNet), which applies adaptive
graph convolution to obtain relevant information between key points
of the human body, and combines multiple biological features such as
the person’s head, neck, shoulders to mitigate the influence of clothing
alterations. However, these methods have high requirements on image
quality, and when the image is affected by occlusion, low illumination
and so forth, this will limit the extraction of identity-related features,
thus limiting the performance of the model. To further reduce the
dependence on collecting a large amount of clothing change data, PosNeg [22] introduced an innovative data augmentation strategy, using
positive augmentation and negative augmentation techniques to enrich
the ID feature space and generate out-of-distribution synthetic samples,
thereby enhancing the model’s robustness to clothing changes.
Another very common methods seek to segregate clothing-related
features from irrelevant features, enabling the model to concentrate
on acquiring clothing-independent identity information. Xu et al. [8]
proposed AFD-Net, which uses GAN and semantic perception models
to distinguish the appearance and structural features of pedestrian
images to achieve the separation of identity and clothing features,
thereby enabling the model to learn identity Discriminating features.
Similarly, SAVS [23] first locates the human body and clothing area
according to the human body semantic segmentation, and introduces
the human body semantic attention module to emphasize the human
body information. Furthermore, it shields the clothing area to make the
model focus on the extraction of visual semantic information unrelated
to clothing. However, these kinds of methods generally face a challenge: in the process of separating clothing features from non-clothing
features, distorted details are inevitably generated and the accurate
expression of cloth-irrelevant features may be weakened, resulting in
unstable training processes and poor model performance. Considering
the limitation of the above two types of methods, we do not use any
biological auxiliary branches or feature decoupling to help distinguish
individuals, but make full use of the differences in features of each layer
of Transformer to learn identity-related features. Specific introduction
will be shown in the next section.

In image processing, the Transformer architecture builds a visual
feature hierarchy layer-by-layer, from edge and texture detection at
the primary layer to scene comprehension at the high-level layer.
However, previous Re-ID models often only focus on the information
of the terminal layer, while ignoring the fine details of the primary
and intermediate layers. To make up for this deficiency, we propose
the Multi-Layer Dynamical Concentration Module (MLDC) (Fig. 3).
This model dynamically synthesizes features across layers and also
includes the key visual information from each layer in the final feature
representation.
Calculation of weights. In order to effectively perform multilayer feature fusion, in our method, we assign a weight coefficient
𝑤𝑖 (i = 1...12) to each layer, the purpose of which is to evaluate the
feature importance of each layer in real time and reduce the similarity
redundancy of related layers, and the specific calculation process of 𝑤𝑖
is as follows:
)
(
𝐿
|⟨𝐹𝑖 , 𝐹𝑗 ⟩|
𝑒𝑥𝑝 𝑓𝑖 − 𝛼 𝛴𝑗=1,𝑗≠𝑖
(1)
𝑤𝑖 =
),
(
𝐿 𝑒𝑥𝑝 𝑓 − 𝛼 𝛴 𝐿
|⟨𝐹𝑘 , 𝐹𝑚 ⟩|
𝛴𝑘=1
𝑘
𝑚=1,𝑚≠𝑘
where 𝐹𝑖 ∈ R𝑁×𝐷 represents the output of the 𝑖th layer, N is the
number of image blocks and D is the feature dimension of each token,
⟨⋅, ⋅⟩ is the inner product, which measures the feature correlation of
different layers, 𝛼 is a regularization coefficient used to scale the impact
of orthogonality constraints and reduce feature overlap between layers,
𝐿 is the total number of layers. And 𝑓𝑖 is a one-dimensional scalar that
represents the importance of the output feature 𝐹𝑖 of each layer, the
specific calculation formula is as follows:
1∑
𝑚𝑒𝑎𝑛(𝐴𝑖𝑡 ),
ℎ 𝑡=1
ℎ

𝑓𝑖 =

(2)

where h represents the number of attention heads in each layer, 𝑚𝑒𝑎𝑛(⋅)
represents the mean of all elements, 𝐴𝑖𝑡 ∈ R𝑁×𝑁 represents the
attention score matrix of the 𝑡th head in the 𝑖th layer (t, i = 1,2, . . . ,12),
and the formula is as follows:
(
)
𝑄𝑖𝑡 𝐾𝑖𝑡𝑇
𝐴𝑖𝑡 = softmax
,
(3)
√
𝑑𝑡
where 𝑄𝑖𝑡 ∈ R𝑁×𝑑𝑡 and 𝐾𝑖𝑡 ∈ R𝑁×𝑑𝑡 are the query and key matrices of
is the dimension size of each
the 𝑡th head in the 𝑖th layer, and 𝑑𝑡 = 𝐷
ℎ
head, which is used to scale the dot product result to prevent too large
values from affecting the gradient of the 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(⋅) function.
Enhanced Feature Fusion With Regularization. To mitigate the
risk of model over-fitting that may occur due to the undue influence
of specific layers, we incorporate an 𝐿2 regularization term into our
feature fusion formula:
𝐿
𝐹𝑓 𝑢𝑠𝑖𝑜𝑛 = 𝛴𝑖=1
𝑤𝑖 𝐹𝑖 − 𝜆 ⋅ ‖𝑊 ‖2𝐹 ,

(4)

where 𝜆 is a non-negative regularization parameter employed to mitigate over-fitting by constraining the magnitude of the weights within
the model. ‖𝑊 ‖2𝐹 is the Frobenius norm of the weight matrix 𝑊 and
is the sum of the squares of all layer weights.
3.2. Local pyramid aggregation module

3. The proposed method

Although the adaptive weighted average method is very effective in
integrating multi-layer features to achieve comprehensive representation, its global fusion method may ignore local information.
In order to capture image details at various scales, we design a
local pyramid aggregation module (LPAM), as shown in Fig. 3. The
module adopts a pyramid structure and gradually fuses the output
of four different layers 𝐹𝑙1 , 𝐹𝑙2 , 𝐹𝑙3 , 𝐹𝑙4 to achieve the extraction of
different fine-grained information, the specific layer selection is provided in the ablation study section. For example, in the local feature

This section elaborates on our proposed approach. We first introduce a Multi-Layer Dynamical Concentration Module to evaluate the
significance of features at each layer in real time while minimizing computational redundancy among highly correlated layers in Section 3.1. In
Section 3.2, we further adopt a Local Pyramid Aggregation Module to
enhance multi-scale features and integrate Fast Fourier Transform (FFT)
to optimize the self-attention mechanism. Finally, the optimization of
the overall framework is described in Section 3.3.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Adversarial perturbation and defense for generalizable person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 186 (2025) 107287

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Adversarial perturbation and defense for generalizable person
re-identification
Hongchen Tan a , Kaiqiang Xu b , Pingping Tao c , Xiuping Liu b ,∗
a

Institute of Future Technology, Dalian University of Technology, Dalian, Dalian 116024, China
School of Mathematical Sciences, Dalian University of Technology, Dalian 116024, China
c
Shandong University, Weihai 264209, China
b

ARTICLE

INFO

Keywords:
Generalizable person Re-ID
Hard matching samples
Semantics purification
Metric perturbation

ABSTRACT
In the Domain Generalizable Person Re-Identification (DG Re-ID) task, the quality of identity-relevant
descriptor is crucial for domain generalization performance. However, for hard-matching samples, it is
difficult to separate high-quality identity-relevant feature from identity-irrelevant feature. It will inevitably
affect the domain generalization performance. Thus, in this paper, we try to enhance the model’s ability
to separate identity-relevant feature from identity-irrelevant feature of hard matching samples, to achieve
high-performance domain generalization. To this end, we propose an Adversarial Perturbation and Defense
(APD) Re-identification Method. In the APD, to synthesize hard matching samples, we introduce a MetricPerturbation Generation Network (MPG-Net) grounded in the concept of metric adversariality. In the MPG-Net,
we try to perturb the metric relationship of samples in the latent space, while preserving the essential visual
details of the original samples. Then, to capture high-quality identity-relevant feature, we propose a Semantic
Purification Network (SP-Net). The hard matching samples synthesized by MPG-Net is used to train the SPNet. In the SP-Net, we further design the Semantic Self-perturbation and Defense (SSD) Scheme, to better
disentangle and purify identity-relevant feature from these hard matching samples. Above all, through extensive
experimentation, we validate the effectiveness of the APD method in the DG Re-ID task.

1. Introduction
Person re-identification (Re-ID) endeavors to accurately identify a
specific individual across non-overlapping cameras, despite variations
in viewpoints, times, and locations (Liu, Feng, Chen, & Hu, 2023;
Ning, Wang, Wang, Zhang, & Ning, 2023; Wang, Huang, Yang, Tiwari,
& Zhang, 2024; Zhu et al., 2024). While employing deep learning
method, person Re-ID techniques (Li, Zhang, Tian, Wang, & Gao, 2022;
Tan, Liu, Bian, Wang, & Yin, 2022; Tan, Liu, Yin, & Li, 2023) have
exhibited significant performance on various publicly available benchmarks. However, a pivotal limitation remains; most of these methods
are trained and evaluated on identical datasets/domains. Consequently,
when deployed in practical scenarios involving new environments or
unseen domains, their performance often suffers a substantial decline.
This performance degradation is primarily attributed to the significant
discrepancy between the source/training and target/test domains.
To realize effective Re-ID in new target domains, recent DG ReID methods (Dai, Li, Liu, Tong, & yu Duan, 2021; Li, Zhang, Hu,
Zhang, & Yu, 2024; Liao & Shao, 2022; Zhuang et al., 2020) integrate
multiple source samples or multiple domain-specific models to cover

data distribution from various domains as much as possible. This
approach attempts to improve domain generalization by increasing the
amount of data/models to approximate or cover the distribution of
the target scenario. In addition, disentanglement-based methods (Eom
& Ham, 2019; Jin, Lan, Zeng, Chen, & Zhang, 2020; Zhang et al.,
2021; Yi-Fan Zhang et al., 2021; Zou, Yang, Yu, Kumar, & Kautz,
2020) believe that the key obstacle to domain generalization is the
identity-irrelevant feature. To this, they (Eom & Ham, 2019; Jin et al.,
2020; Zhang, Lan, et al., 2021; Zhang, Zhang, et al., 2021; Zou et al.,
2020) rely on a hypothesis that identity-irrelevant features are those whose
intra-class spacing is greater than their inter-class spacing, to disentangle
the identity-relevant and -irrelevant feature, and only use the former
for DG Re-ID. However, their assumption is correct only when the
feature space is well-designed. Specifically, before a well-trained model
and effective feature space are obtained, it is difficult to differentiate identity-relevant feature from identity-irrelevant ones. Especially
when there are many hard matching samples, their identity-relevant
and -irrelevant components are heavily entangled and hard to separate. Since this, the identity-relevant descriptor, which contains some

∗ Corresponding author.

E-mail addresses: tanhongchenphd@bjut.edu.cn (H. Tan), 459553299@qq.com (K. Xu), pingping.tao@sdu.edu.cn (P. Tao), xpliu@dlut.edu.cn (X. Liu).
https://doi.org/10.1016/j.neunet.2025.107287
Received 15 August 2024; Received in revised form 10 December 2024; Accepted 13 February 2025
Available online 22 February 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 186 (2025) 107287

H. Tan et al.

Fig. 1. Flowchart of the APD method. We first train the MPG-Net using Source Domain data. MPG-Net adopts the metric adversariality strategy to synthesize hard matching
samples (i.e. ‘‘New Source Domain’’ in Figure) based on Source Domain data. Secondly, SP-Net accommodates the training of hard matching samples synthesize by MPG-Net and
achieve high-quality person feature capturing capabilities. Finally, SP-Net performs person matching on the Unseen Domain.

misclassified identity-irrelevant feature, is prone to interference from
non-key factors of target domain such as background feature. Recently,
a serious of outstanding methods (Chen, Li, Wu, Liang, & Jha, 2021;
Wang, Wang, Zhu, & Wang, 2022; Zhu et al., 2023) have demonstrated
that enhancing the model’s ability to perceive and capture effective
semantics from difficult samples can effectively improve the model’s
generalization capability. Therefore, inspired by them (Chen et al.,
2021; Wang et al., 2022; Zhu et al., 2023), we try to enhance the ability
of the Re-ID model to separate identity-relevant feature from identityirrelevant feature of hard matching samples, to capture high-quality
identity-relevant feature applications in the face of unseen domains.
Above all, we propose a novel Adversarial Perturbation and Defense (APD) Re-identification Method for the Domain Generalization
person Re-identification (DG Re-ID) task. In the APD, we firstly try
to synthesize hard matching samples; we use these hard matching
samples to train the Re-ID model; Secondly, we try to separate and
purify the identity-relevant feature. Flowchart of the APD method is
shown in Fig. 1. For synthesizing hard matching samples, we design a
Metric-Perturbation Generation Network (MPG-Net). In the MPG-Net,
we aim to achieve a scenario where the inter-class distances between
samples of different person IDs are smaller than the intra-class distances
between samples of the same person ID, essentially adopting an inverse
strategy to metric learning. Moreover, during the generation process,
we adopt a shallow content consistency strategy to maintain minimal
variations in the person appearance. To capture high-quality identityrelevant feature, we propose a Semantic Purification Network (SP-Net).
And hard matching samples synthesized by MPG-Net is used to the
SP-Net. In the SP-Net, we propose a Semantic Self-perturbation and
Defense (SSD) Scheme to separate identity-relevant and -irrelevant
feature; and we purify the identity-relevant feature through a feature
attack manner. Specifically, based on the constraint that the intra-class
distance is less than the inter-class distance, the feature captured by the
model is the identity-relevant feature; otherwise, it is identity-irrelevant
feature; We further use identity-irrelevant feature to perform a disorder
perturb on identity-relevant feature to purify and improve the quality of
identity-relevant feature. This work makes the following contributions:
(i) We design a Metric-Perturbation Generation Network (MPG-Net)
to synthesize hard matching person samples.
(ii) We design a Semantic Purification Network (SP-Net) to perceive
and separate identity-relevant and -irrelevant feature from these hard
matching samples.
(iii) Extensive experiments on the many datasets demonstrate that
our APD achieves the competitive performance. A series of ablation studies have also validated that each design component significantly improves the performance of Domain Generalization person
Re-identification (DG Re-ID) task.

data to bolster the model’s adaptability across domains. Style Transferbased approaches (Chen, Zhu, & Gong, 2019; Huang, Wu, Xu, & Zhong,
2019; Lin, Wu, Yan, Xu, & Yang, 2020; Zhong, Zheng, Zheng, Li,
& Yang, 2019) initially leverage Generative Adversarial Networks
(GANs) (Goodfellow et al., 2014) to transform the style of unlabeled
target data to match the source data. This style-translated source data
is then used to train the Re-ID model, improving its robustness to
domain shifts. Attribute Recognition-based methods (Li, Chen, Tao, Yu, &
Qi, 2021; Wang, Zhu, Gong, & Li, 2018; Xu, Luo, & Hu, 2021) initially
utilize a Re-ID model to extract person descriptors. Subsequently, an
attribute detector is employed to extract attribute embeddings. These
person descriptors are then combined with the attribute embeddings
to facilitate person matching in the target domain, leveraging the
discriminative power of attributes. Pseudo Label-based methods (Dai, Liu,
Bai, Tong, & Duan, 2021; Feng et al., 2021; Zheng, Liu, He, Mei, Luo,
& Zha, 2021; Zheng et al., 2021) typically begin by utilizing a trained
Re-ID model to perform sample clustering through various clustering
strategies. Pseudo-labels are then predicted for unlabeled target data,
enabling the utilization of these pseudo-labeled target domain data to
fine-tune the Re-ID model. However, these methods still heavily rely on
data collection and often perform poorly in unseen scenarios, limiting
their practical applicability and generalization capabilities.
2.2. Domain generalization person Re-ID
Due to their practical relevance, Domain Generalization (DG) person
Re-ID methods have garnered increasing attention in recent years.
Meta Learning-based approaches (Choi, Kim, Jeong, Park, & Kim,
2021; Zhao et al., 2021) incorporate meta-learning and normalization
techniques (Huang & Belongie, 2017) to mimic real train–test domain
shifts and bolster generalization capabilities. The normalization technique, in particular, can mitigate identity-independent style variations
and extract identity-relevant descriptors, effectively mitigating the impact of different scene styles on generalization performance. However,
these methods often overlook the influence of different normalization
techniques and their positioning on the corresponding person image
features. Additionally, normalization techniques can potentially lose
parts of identity-relevant feature.
Ensemble Learning-based methods (Dai, Li, et al., 2021; Liao & Shao,
2022; Mancini, Bulò, Caputo, & Ricci, 2018; Zhuang et al., 2020) leverage multiple source domain data to train the Re-ID model, breaking
the confinement of a single source domain’s style. For instance, Song,
Yang, Song, Xiang, and Hospedales (2019) constructed a large-scale
domain generalization person Re-ID database and employed a metalearning training scheme (Finn, Abbeel, & Levine, 2017) along with a
proposed domain invariant mapping network. However, the additional
mapping network can slow down inference. In response, Tamura and
Murakami (2019) proposed a lightweight data augmentation selection
strategy that is easily applicable to other models or tasks. Nevertheless, these methods heavily rely on the collection and construction of
databases, making it challenging to predict whether the collected data
will adequately cover the target data distribution.

2. Related work
2.1. Domain adaptation person Re-ID
To enhance the domain generalization capability of the model,
Domain Adaptation Re-ID methods strive to collect unlabeled target
2

Neural Networks 186 (2025) 107287

H. Tan et al.

main visual content of samples unchanged, under perturbing sample
metric process. Next, we will gradually achieve the aforementioned
objective through specific formulations and equations.
First, we describe the sample partitioning strategy. As shown in
Fig. 2-(a), we denote the training dataset as 𝑋 = {𝑥1 , 𝑥2 , … , 𝑥𝑁 }, where
𝑁 is the number of person samples. For each sample 𝑥𝑖 ∈ 𝑋, we build
its negative sample subset 𝐴𝑖 by randomly picking 𝑀 persons whose ID
are different from 𝑥𝑖 . Namely, 𝐴𝑖 contains 𝑀 negative samples, and is
⋃ ⋃ ⋃
denoted as 𝐴𝑖 = {𝑥𝑖𝑗 |𝑗 = 1, 2, … , 𝑀} (𝐴 = 𝐴1 𝐴2 𝐴3 ⋯). Note
that 𝑥𝑖𝑗 ∈ 𝐴𝑖 is the negative sample of 𝑥𝑖 , and 𝐴𝑖 is the set of 𝑀
randomly selected negative samples of 𝑥𝑖 .
Second, we adopt GAN to push visual semantics of anchor sample
𝑥𝑖 to approximate its negative samples 𝐴𝑖 = {𝑥𝑖𝑗 |𝑗 = 1, 2, … , 𝑀}. The
generator loss between sample 𝑥𝑖 and negative sample 𝑥𝑖𝑗 is defined as:
1
𝐿𝐺 = − E𝑥𝑖 [𝑙𝑜𝑔 𝐷(𝐺(𝑥𝑖 ))].
(1)
2

Fig. 2. Schematic diagram of the MPG-Net. In sub-figure (a), we present the planning
of the samples and the negative sample set to prepare for the subsequent modification
of sample feature. The sub-figure (b) shows that we aim to reduce the distance between
samples of the same person ID and increase the distance between samples of different
person IDs in the deep latent space. In sub-figure (c), while the metric/distances changes
between samples, the person’s appearance feature should not change dramatically.

𝐿𝐺 drives anchor sample 𝐺(𝑥𝑖 ) to approximate the negative sample
set 𝐴𝑖 , aiming to fool the discriminator 𝐷. The discriminator is trained
to classify the input image into the ‘‘Fake/True’’ category by
1
1
𝐿𝐷 = − E𝑥𝑖 [𝑙𝑜𝑔(1 − 𝐷(𝐺(𝑥𝑖 )))] − E𝑥𝑖𝑗 [𝑙𝑜𝑔(𝐷(𝑥𝑖𝑗 ))].
(2)
2
2

Disentanglement-based methods (Eom & Ham, 2019; Jin et al., 2020;
Zhang, Lan, et al., 2021; Zhang, Zhang, et al., 2021; Zou et al., 2020)
strive to separate identity-relevant features from identity-irrelevant
ones, utilizing only the former for person Re-ID. However, as mentioned
in Section 1, under their current framework, identity-relevant feature
are often misinterpreted as identity-irrelevant, leading to inaccuracies.
To address this, based on the semantic disentanglement, we try to
purify identity-relevant feature through the semantics perturbation
strategy. Furthermore, we also adopt hard matching samples to further
push our proposed Re-ID model to refine and improve person descriptor
extraction.

For achieving the metric perturbation, we intentionally make (see
Fig. 2-(b)): (1) the distance between the synthesized sample 𝐺(𝑥𝑖 ) and
the negative sample 𝑥𝑖𝑗 to be as close as possible, and (2) the distance
between the synthesized sample 𝐺(𝑥𝑖 ) and its original sample 𝑥𝑖 to be
as far as possible. Therefore, we define a metric perturbation loss 𝐿𝑀 𝑃
by revising the triplet loss (Schroff, Kalenichenko, & Philbin, 2015):
𝐿𝑀 𝑃 = 𝑆 𝑃 (𝑑(𝜓(𝐺(𝑥𝑖 )), 𝜓(𝑥𝑖𝑗 )) − 𝑑(𝜓(𝐺(𝑥𝑖 )), 𝜓(𝑥𝑖 ))).

(3)

Here 𝑆 𝑃 (𝑥) = 𝑙𝑛(1 + 𝑒𝑥 ), distance 𝑑(𝑎, 𝑏) is the euclidean distance
between vectors 𝑎 and 𝑏, and 𝜓(⋅) is the deep neural layer to extract person semantics. To improve the quality of deep semantics, we adopted
ResNet-50 pre-trained on ImageNet (Jia et al., 2009) to construct the
𝜓(⋅). Note that the 𝜓(⋅) parameter is frozen and does not participate
in model training. We use ‘‘ResNet-50 Stage 4’’ to capture the deep
semantics. With the design of this strategy, in the deep latent space,
the distances between samples of the same ID will be increased, while
the distances between samples of different IDs will be reduced.
When using 𝐿𝑀 𝑃 to adjust the inter-/intra-class spacing in the latent
space, the content of the sample 𝐺(𝑥𝑖 ) could get modified and distorted
through this adjustment. Third, we would like to keep basic content
including the structure and body silhouette of the person relatively
stable (see Fig. 2-c). We know that in deep models, shallow features
primarily contains the content aspects of a person’s appearance. Thus,
we design a Content Consistency loss

3. Method
Flowchart of the APD method is shown in Fig. 1. To this end,
our APD contains two components: a Metric-Perturbation Generation
Network (MPG-Net) and a Semantic Purification Network (SP-Net).
The MPG-Net is designed to adopt the metric adversariality strategy
to synthesize hard matching samples. The SP-Net is designed to better accommodate the training of hard matching samples and achieve
high-quality person feature capturing capabilities.
3.1. MPG-Net
As delineated in Section 1, our first objective is to synthesize hard
matching samples. Consequently, in this section, we strive to realize a
scenario where the inter-class distances between samples of different
person IDs are smaller than the intra-class distances between samples
of the same person ID. It essentially reverses the conventional metric learning strategy. In addition, while the metric/distances changes
between samples, the person’s appearance feature should not change
dramatically. To achieve the goal, we propose the Metric-Perturbation
Generation Network (MPG-Net).
As an excellent generation model, Generative Adversarial Networks
(GANs) (Goodfellow et al., 2014) have outstanding performance in
modifying content and semantics. Therefore, we build our MPG-Net
based on GANs. However, simply using GAN does not effectively modify the metric space of samples from the person ID perspective. To this,
we try to design a sample partitioning strategy to build a negative sample set for each sample. And then GAN is used to push visual feature of
each sample to approximate its negative samples. To perturb the sample
metric, we further design the metric perturbation constraint to increase
intra-class spacing and narrow inter-class spacing. However, such a
strategy will seriously distort or change the key semantics of persons,
which is not conducive to the capture of high-quality descriptors. Thus,
we further to introduce the content consistency constraint to keep the

𝐿𝐶 = ‖𝜑(𝐺(𝑥𝑖 )) − 𝜑(𝑥𝑖 )‖1 ,

(4)

to penalize basic content change after modification. The 𝜑(⋅) is a
shallow neural layer to encode the basic content. We adopted ‘‘ResNet50 Stage 2’’ pre-trained on ImageNet (Jia et al., 2009) to construct the
𝜑(⋅). Note that the 𝜑(⋅) parameter is frozen and does not participate in
model training.
In addition, we also need to ensure the stability of samples’ visual semantics. We hope that if 𝐺(𝑥𝑖 ) does further sample synthesis
(i.e., 𝐺(𝐺(𝑥𝑖 ))), the visual semantics should be consistent with 𝐺(𝑥𝑖 ).
Thus, the loss can be defined as
𝐿𝑆 = ‖𝐺(𝐺(𝑥𝑖 )) − 𝐺(𝑥𝑖 )‖1 .

(5)

Finally, generator has the combined loss 𝐿𝑀 𝑃 𝐺 = 𝐿𝐺 + 𝜆1 𝐿𝑀 𝑃 +
𝜆2 𝐿𝑆 + 𝜆3 𝐿𝐶 . And the discriminator adopts the cross-entropy loss,
𝐿𝐷𝑀 𝑃 𝐺 = 𝐿𝐷 . Based on control variable strategy of parameter selection,
we set 𝜆1 = 2.0, 𝜆2 = 1.0, and 𝜆3 = 1.0.
3


codex
前 5 篇里已经出现几个高频模式：遮挡论文还在围绕“局部对齐加遮挡权重”，红外论文围绕“跨模态生成或伪标签净化”，泛化论文把“困难样本”重新定义成度量关系被扰动的样本。后面 4 篇我继续按同样口径抽取，最后只保留对我们能发 B 类有用的反推结论。
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
/bin/zsh -lc "pdftotext -l 3 'Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Attribute Guidance With Inherent Pseudo-label For
Occluded Person Re-identification
Rui Zhia , Zhen Yanga and Haiyang Zhanga, *

arXiv:2508.04998v1 [cs.CV] 7 Aug 2025

a Beijing University of Post and Telecommunication

Abstract. Person re-identification (Re-ID) aims to match person
images across different camera views, with occluded Re-ID addressing scenarios where pedestrians are partially visible. While pretrained vision-language models have shown effectiveness in Re-ID
tasks, they face significant challenges in occluded scenarios by focusing on holistic image semantics while neglecting fine-grained
attribute information. This limitation becomes particularly evident
when dealing with partially occluded pedestrians or when distinguishing between individuals with subtle appearance differences.
To address this limitation, we propose Attribute-Guide ReID (AGReID), a novel framework that leverages pre-trained models’ inherent capabilities to extract fine-grained semantic attributes without
additional data or annotations. Our framework operates through a
two-stage process: first generating attribute pseudo-labels that capture subtle visual characteristics, then introducing a dual-guidance
mechanism that combines holistic and fine-grained attribute information to enhance image feature extraction.
Extensive experiments demonstrate that AG-ReID achieves stateof-the-art results on multiple widely-used Re-ID datasets, showing
significant improvements in handling occlusions and subtle attribute
differences while maintaining competitive performance on standard
Re-ID scenarios.

1

Introduction

Person Re-Identification (ReID) is a computer vision task that aims
to match person images across different camera views [27]. With the
rapid development of smart cities and surveillance systems, ReID
has become increasingly crucial in various real-world applications,
processing millions of surveillance videos daily. It requires the extraction of unique features from images to recognize the same person, even when there are alterations in pose, lighting, and viewpoint.
Among various challenges in ReID, occlusion presents a particularly
significant obstacle. This necessitates advanced algorithms capable
of effectively managing partial occlusions while maintaining precise
identification capabilities.
Existing approaches to address occlusion in ReID can be categorized based on their feature extraction strategies. Global methods,
which are commonly used in general ReID tasks, learn a single feature to represent the entire image. While effective in standard scenarios, these methods often fail to handle occluded regions effectively
as they cannot distinguish between visible and occluded parts. To
address this limitation, researchers have proposed part-based techniques [20, 11, 17] and attribute-based methods. Part-based tech∗ Corresponding Author. Email: zhhy@bupt.edu.cn

Figure 1. Comparison of retrieval results between CLIP-ReID and
AG-ReID on challenging cases. Case1: A person wearing a black-red jacket
with subtle color variations. Case2: A person with an umbrella in occluded
scenario. For each case, the first column shows the query image, followed by
top-10 retrieval results. Incorrect matches (marked as ’0’) are outlined in red,
while correct matches (marked as ’1’) are outlined in green.

niques focus on comparing visible body parts between images, which
helps to identify individuals even when some parts are occluded. Recent studies [4, 28, 26] have shown that attribute-based methods can
significantly enhance model performance by leveraging detailed visual characteristics such as clothing style, accessories, and physical
features. These two approaches have shown promising results in handling occluded scenarios, but they inevitably require additional supervision signals, such as pose estimation data, human parsing labels,
or manually annotated attributes. This dependence on extra annotation data significantly limits their practical application in large-scale
scenarios.
Recent advances in pre-trained vision-language models present a
promising direction for addressing these limitations. Studies [10, 28]
have established that models like CLIP (Contrastive LanguageImage Pre-training) [14] are highly effective in extracting comprehensive semantic information from images, achieving significant performance improvements in ReID tasks. By learning combined representations of text and visuals through contrastive learning on massive image-text pairs, these models excel at grasping complex semantic concepts and transferring knowledge to downstream tasks.
However, in the context of occlusion ReID tasks, these models face
similar challenges as global methods. Relying solely on holistic in-

formation from textual prompts to guide image feature extraction
has significant limitations. This approach leads to the omission of
fine-grained semantic information within the image, causing the pretrained model to focus only on primary features that may be partially visible or occluded. As a result, performance degrades substantially in occlusion scenarios. As illustrated in Figure 1, the tendency
of CLIP-ReID to focus on holistic features while overlooking finegrained details significantly impacts its ability to distinguish between
individuals with similar appearances or identify partially occluded
persons.
This paper explores the potential of mining fine-grained feature
semantic information from the pre-trained vision-language model
CLIP to improve its performance in occluded ReID tasks. Our approach leverages attribute guidance, thereby eliminating the necessity for additional data or capabilities and representing a promising avenue for enhancing the model’s efficacy in this challenging
domain. We propose a novel attribute-guided method called AGReID that supports this challenging task through a two-stage training process. In the first stage, AG-ReID acquires image attribute
pseudo-labels by leveraging the detailed semantic information within
CLIP through context optimization. In the second stage, it guides
the extraction of image features with both holistic and fine-grained
attribute information, improving the retrieval performance of the
model. Specifically, the dual guidance includes: 1) Attribute-prompt
guidance: using the overall attribute prompt text feature to guide image features through contrastive learning. 2) Fine-grained attribute
pseudo-label guidance: the learnable tokens are implicitly trained
through the CoOp [31] method to obtain fine-grained semantics,
thereby guiding the extraction of image features. Furthermore, we
propose an attribute encoder for aligning image features with attribute pseudo-labels, and an attribute loss for measuring the semantic difference between them. To handle inconsistent features in occluded scenarios, we introduce a noise-masking mechanism that selectively considers attribute pairs based on their semantic similarity.
The efficacy of AG-ReID was assessed through experimentation on
multiple well-known occluded and holistic ReID datasets. The results demonstrated that AG-ReID outperformed a number of existing
methods. The primary contributions of this work are summarized as
follows:

• To the best of our knowledge, this is the first attempt to improve occluded ReID by embedding fine-grained attribute semantic into image features through the inherent capabilities of the
CLIP model, without the need for extra data or annotations, significantly enhancing model performance in occlusion scenarios.
• We present a novel attribute dual-guidance ReID framework
called AG-ReID, which effectively guides the extraction of image features through both holistic text features and fine-grained attribute pseudo-labels, improving feature accuracy and robustness.
• We propose an innovative method that implicitly trains attribute
pseudo-labels through context optimization, and design corresponding attribute encoder module and attribute loss to achieve
effective alignment between image features and attribute pseudolabels.
• We conduct evaluations on multiple challenging datasets, including Occluded-ReID, P-Duke and MSMT17, demonstrating that
AG-ReID achieves state-of-the-art performance in both occluded
and holistic scenarios.

2

Related Works

2.1 Pre-trained Vision-Language Learning
Pre-trained vision-language models are a class of machine learning
models trained on large-scale datasets to understand and process both
visual and textual data. These models can understand the relationships between images and text, enabling them to perform a variety
of tasks such as image captioning, visual question answering, and
image-text retrieval. The power of these models lies in their ability
to generalize well to various downstream tasks. This is achieved by
pre-training on extensive datasets, which enables the models to learn
a wide range of visual-textual relationships. Once pre-trained, these
models can be fine-tuned on specific tasks, making them versatile
and effective in real-world applications.
The dual encoder architecture is a prevalent pre-trained visionlanguage model architecture that employs two separate unimodal
encoders to independently process images and text. It utilizes shallow attention layers or dot products to align the embeddings of both
modalities into a unified semantic space, enhancing efficiency in
tasks like image-text retrieval. Nonetheless, the limited depth of interaction between the modalities can pose challenges in complex
visual-language understanding tasks. Both the widely used CLIP and
ALIGN [7] models incorporate this dual encoder architecture.
Contrastive Language–Image Pre-training (CLIP) is a pre-trained
model introduced by OpenAI that efficiently learns visual concepts
from natural language supervision and is adaptable to various downstream tasks [14]. It consists of two encoders, an Image Encoder
and a Text Encoder. Image encoders, with architectures like ViT
[3] or ResNet50 [5], are designed to transform images from a highdimensional RGB space into a low-dimensional embedding space.
The text encoder converts each word in the given prompt to a unique
numeric ID, maps IDs into embedding vectors, and finally encodes
them to a text feature that contains prompt semantic information.
During the training phase, CLIP optimizes a symmetric cross entropy loss to achieve the target of maximizing the cosine similarity
for matched pairs while minimizing the cosine similarity for all other
unmatched pairs.
One of the challenges in applying pre-trained models to downstream tasks is the time-consuming and domain-expertise-required
prompt engineering. Context Optimization (CoOp) automates this
process by modeling prompt context words using learnable vectors
while keeping the pre-trained parameters frozen [31]. CoOp significantly enhances prompt engineering performance and demonstrates robust domain generalization capabilities compared to manual
prompts. In essence, CoOp transforms static text prompts into learnable text templates. It acquires and learns text descriptions directly
through the intrinsic multi-modal abilities of the pre-trained model,
avoiding intricate manual word tuning. Specifically, the prompt embedding given to the text encoder is designed with trainable tokens
and fixed tokens, where the number and position of trainable tokens
can be adjusted according to the requirements of downstream tasks.

2.2 Attribute-based Person ReID
Person ReID has been extensively studied due to its critical applications in surveillance and security. Traditional person ReID methods
primarily focus on visual features extracted from images, and combine additional information such as pose, body mask, and visible infrared to address occlusion, light changes, and more [20, 19, 17, 22].
Similarly, recent research has explored the use of fine-grained at-

tributes, such as clothing color, accessories, and physical characteristics, to enhance ReID performance.
Attribute-based Person ReID leverages these descriptive attributes
to provide additional semantic information, improving the robustness
and accuracy of person matching. Notable work in this area includes
artificially annotating image attribute features in existing datasets [9],
using language models to automatically generate and utilize attribute
descriptions [26], and integrating multi-modal data to bridge the gap
between textual and visual information [28]. Recent advances have
also explored prompt-guided approaches for feature disentangling in
occluded scenarios [2] and text-based multi-granularity contrastive
learning for occluded person ReID [24].
In addition, recent work has also explored the interpretability of
attribute-guided methods. For example, AMD method [1] provides
post-mortem explanations for existing ReID models by identifying
and quantifying the contributions of different attributes.

3

Method

3.1 Overview
This section provides a detailed introduction to the AG-ReID model,
which leverages the rich semantic information inherent in pre-trained
models to enhance image retrieval performance for occluded person
ReID tasks. Our framework operates through a two-stage process
that aligns visual and textual modalities. In the first stage, we establish a semantic bridge between images and their fine-grained attributes, generating attribute pseudo-labels that capture subtle visual
characteristics. In the second stage, we introduce a dual-guidance
mechanism that combines holistic attribute-prompt features and finegrained attribute pseudo-labels to enhance image feature extraction.
To handle inconsistent features in occluded scenarios, we propose
a noise-masking mechanism that selectively focuses on reliable attribute matches while filtering out those affected by occlusions.

3.2 Cross-Modal Alignment Stage
3.2.1

Preliminary

To adapt the text encoder for specific downstream tasks without
extensive prompt engineering, we leverage a technique that incorporates learnable context vectors into the prompt embedding. Specifically, the input Ti to the text encoder ET is structured to include r
trainable vectors [v] alongside k − r fixed tokens [t]. The number r
and the position of [v] can be adjusted based on task requirements:
Ti = [t]i1 [v]i1 ...[v]ir [t]ik−r

During fine-tuning on a downstream task, only these learnable vectors [v] are optimized, utilizing the semantic knowledge encoded
within the frozen pre-trained model parameters to capture taskrelevant information.

3.2.2

Ti = [t]i1 [t]i2 ...[t]ik = Embd(pi )

(1)

fiT = ET (Ti )

(2)

During pre-training, such architectures typically optimize a symmetric contrastive loss to align the image and text representations
in a shared embedding space. This objective maximizes the cosine
similarity for matched image-text pairs while minimizing it for unmatched pairs:
s(fM , fT ) = L2(projM (fM )) · L2(projT (fT ))T

(3)

where projM and projT are projection layers, L2 denotes L2normalization, and t is a learned temperature parameter.

Attribute Prompt Template

To effectively capture fine-grained attributes of person images, we
design a set of attribute prompt templates that cover various visual
characteristics. These templates are constructed using a combination
of fixed and learnable tokens, following the CoOp framework. The
attribute prompt templates are designed to describe different aspects
of a person’s appearance, including:
• Clothing attributes (e.g., color, style, pattern)
• Accessories (e.g., bags, hats, glasses)
• Body characteristics (e.g., height, build)
• Pose and movement
Each attribute prompt template follows the structure:
Tattr = [t]prefix [v]1 [v]2 ...[v]r [t]suffix

(5)

where [t]prefix and [t]suffix are fixed tokens that provide context, and
[v]1 through [v]r are learnable tokens that adapt to capture specific attribute information. The number of learnable tokens r is determined
based on the complexity of the attribute being described.

3.2.3

Our approach is built upon a dual-encoder architecture designed to
process visual and textual information. Specifically, i ∈ {1...B}
denotes the index within a batch. Given the image batch M =
{m1 , m2, ..., mB } and their text descriptions P = {p1 , p2, ..., pB },
an image encoder EM maps images to 512-d features fiM =
EM (mi ). For text prompts pi , each prompt is first tokenized (e.g.,
using byte pair encoding) and then embedded into 512-d word tokens
Ti via an embedding layer Embd. These tokens are subsequently processed by a text encoder ET to produce the final text feature fiT .

(4)

Attribute Pseudo-label Generation

Inspired by the positive impact of fine-grained attributes in ReID, we
utilize the trainable attribute prompt structure defined above. Unlike
holistic prompts used in some prior work [10], our attribute prompts
feature learnable tokens dispersed throughout the text, each intended
to capture a specific fine-grained feature. This design is critical for
recognizing subtle differences, especially under occlusion, setting
the stage for effective pseudo-label generation as described next.
In our framework, attribute pseudo-labels refer to learnable token
representations automatically derived from CLIP and CoOp, requiring no manual annotation. These pseudo-labels capture fine-grained
visible attribute features (e.g., hair style, clothing color, accessories)
and serve as auxiliary supervision signals to guide the image encoder towards learning more discriminative attribute-aware features.
Unlike traditional labels that require human annotation, our pseudolabels are semantically trained to describe the collective attribute
characteristics of all images belonging to the same identity.
Our selection of attributes is informed by prior work demonstrating their effectiveness in person ReID. Specifically, we adopt key attribute categories identified as discriminative in ATPM [26], encompassing aspects like gender, hairstyle, posture, and clothing characteristics. Furthermore, the designed prompt template module offers
inherent flexibility; it facilitates the straightforward incorporation of
alternative attribute sets or descriptive contexts, such as explicitly


exec
/bin/zsh -lc "pdftotext -l 3 'Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Base-Detail Feature Learning Framework for Visible-Infrared Person
Re-Identification

arXiv:2505.03286v1 [cs.CV] 6 May 2025

Zhihao Gong1 , Lian Wu2 , Yong Xu3,∗
1
Harbin Institute of Technology (Shenzhen)
2
GuiZhou Education University
3
Harbin Institute of Technology (Shenzhen)
gongzhh888@gmail.com, wulian best@163.com, laterfall@hit.edu.cn
Abstract
Visible-infrared person re-identification (VIReID)
provides a solution for ReID tasks in 24-hour scenarios; however, significant challenges persist in
achieving satisfactory performance due to the substantial discrepancies between visible (VIS) and infrared (IR) modalities. Existing methods inadequately leverage information from different modalities, primarily focusing on digging distinguishing
features from modality-shared information while
neglecting modality-specific details. To fully utilize differentiated minutiae, we propose a BaseDetail Feature Learning Framework (BDLF) that
enhances the learning of both base and detail
knowledge, thereby capitalizing on both modalityshared and modality-specific information. Specifically, the proposed BDLF mines detail and base
features through a lossless detail feature extraction module and a complementary base embedding
generation mechanism, respectively, supported by
a novel correlation restriction method that ensures
the features gained by BDLF enrich both detail and
base knowledge across VIS and IR features. Comprehensive experiments conducted on the SYSUMM01, RegDB, and LLCM datasets validate the
effectiveness of BDLF.

1

Introduction

Person re-identification (ReID) aims to retrieve a target identity from gallery images captured by different cameras [Liu
et al., 2022] and has recently demonstrated significant advancements in the fields of security and public surveillance
[Ye et al., 2022a]. However, most existing methods [Cao
et al., 2023][Wang et al., 2022][Yan et al., 2021] primarily focus on utilizing RGB images captured by visible (VIS)
cameras during the daytime, which are inadequate for accommodating 24-hour scenarios that involve infrared (IR)
images captured by IR cameras. To address the substantial
cross-modality gap and facilitate operation in all-day scenarios, visible-infrared person re-identification (VIReID) methods [Chen et al., 2022][Park et al., 2021] have been developed, enabling the matching of IR (RGB) images given an
interest in a specific RGB (IR) pedestrian image.

(a) Features Learning with align- (b) Features Learning with crossing cross-modalities knowledge modalities knowledge compensation

(c) Feature Learning with the proposed BDLF
Figure 1: Motivation of the proposed BDLF, which focuses on sufficiently mining the modality-shared and modality-specific knowledge simultaneously and are not applicable for additional auxiliary
data.

The existing research on VIReID can generally be categorized into two principal methods: extracting distinguishing
modality-shared features from VIS and IR modalities[Park
et al., 2021][Zhang and Wang, 2023] and compensating for
modality-specific or modality-shared features [Zhang et al.,
2022a]. As shown in Figure 1(a), the former method aims to
reduce cross-modality discrepancies by aligning comprehensive cross-modality features into a common semantic space.
However, it neglects to leverage modality-specific and shared
cues, which inevitably leads to performance bottlenecks. The
latter approach, depicted in Figure 1(b) can be further divided into embedding-level and image-level methods. These
methods generate compensatory knowledge in the embedding space and at the pixel level respectively, using auxiliary models(e.g., GANs[Goodfellow et al., 2014], segmentation networks, part alignment networks, etc.). However, these
methods typically introduce losses and noise into the generated features or require additional data processing by other
models, making them less effective and convenient. Consequently, advancing the development of VIReID to a more

comprehensive level remains a significant challenge.
Inspired by the analyses presented above, it is essential to
recognize that modality-shared information, such as the contour and movement characteristics of pedestrians, can be considered base features. In contrast, modality-specific information, including the color and texture details of the RGB
modality and the thermal characteristics of the IR modality,
can be regarded as detail features. Both types of them should
be integrated and utilized effectively together. Therefore, in
this paper, we propose a novel Base-Detail Feature Learning
Framework (BDLF), as shown in Figure 1(c). This framework is designed to extract modality-shared base features
and modality-specific detail features from the original images
with minimal additional computational costs, while jointly
optimizing modality-shared, modality-specific, and comprehensive features.
The proposed BDLF comprises a modality-specific detail
feature extraction (DFE) module and a modality-shared base
embedding generation (BEG) block, which ultimately combine the optimized features collected. Inspired by [Zhao
et al., 2023], we designed the DFE module to mine the
modality-specific detail information losslessly. Subsequently,
the BEG block derives modality-shared base features. To
fully capture both specific and shared information, we proposed a novel specific-shared knowledge distillation(SKD)
loss. It encourages the detail (base) features to effectively
incorporate modality-specific (modality-shared) knowledge
by imposing a constraint on the correlation that the crossmodality detail and base features should exhibit. Specifically,
it ensures that the correlations across RGB and IR modalities are indistinct and notable, respectively. Perspectives in
[Feng et al., 2023] explain that the independent decomposition of features can maximize the mutual information of subfeatures; therefore, we introduced an independence constraint
in the semantic space between the derived detail and base
features. This indicates that the base feature exclusively encompasses modality-shared knowledge, while the detail feature contains modality-specific information. In summary, the
main contributions of our work are as follows:
• A novel correlation optimization method is proposed
that effectively generates both modality-shared and
modality-specific features using a non-parametric approach, rather than relying on classifiers.
• We propose an end-to-end Base-Detail Feature Learning Framework (BDLF) for VIReID that integrates extracts of modality-shared base knowledge and modalityspecific detail knowledge.
• Extensive experiments have demonstrated that the proposed BDLF outperforms other state-of-the-art methods
for the VIReID task on the SYSU-MM01, RegDB, and
LLCM datasets.

2

Related Work

The main idea for solution VI-ReID task is decreasing the
notable discrepence across VIS and IR modalities, thereby
the existing methods consist of aligning the cross-modality
features and utilizing the auxiliary data or features generated
by other models.

The alignment of feature representation methods seeks to
convert cross-modality features into a unified semantic space
through either metric learning techniques [Liu et al., 2022]
[Park et al., 2021] [Luo et al., 2019] or by enhancing networks with more effective feature extraction components
[Zhang and Wang, 2023] [Sarker and Zhao, 2024]. However, these approaches ultimately encounter performance bottlenecks due to the loss of modality-specific information.
The methods for utilizing auxiliary information produced
by other models are proposed to enhance identifiable knowledge. GAN-based methods [Zhang et al., 2022a]d[Wang et
al., 2020] generate compensatory features at either the image level or the embedding level to simulate features from
another modality. XIV [Li et al., 2020] introduces the Xmodality generated by a lightweight auxiliary network to decrease discrepancies between the two modalities. LUPI [Alehdaghi et al., 2022] establishes an intermediate domain between VIS and IR modalities. Furthermore, it generates images that belong to this intermediate domain to guide the
network in acquiring more discernible information. SGIEL
[Feng et al., 2023] innovatively adopts the shape knowledge
of identity generated by segmentation models to enrich supplementary information. TMD [Lu et al., 2024] generates
style-aligned images to minimize differences at the image
level, subsequently aligning cross-modality features to eliminate discrepancies in feature distribution and instance features. However, this remains a challenging field of research
because these methods either inevitably introduce information distortion during the generation process or fail to completely capture modality-specific and modality-shared information.

3

Methodology

3.1

Overall Framework

The pipeline of our proposed method, referred to as BDLF, is
illustrated in Figure 2. This method utilizes a single-stream
ResNet-50 network[He et al., 2016a] as its backbone. The intermediate features Z M ∈ RB×C×H×W , which pass through
a portion of the backbone, are fed into the proposed detail
feature extraction (DFE) module to yield detail features Z D .
Additionally, the base feature Z B is generated by inputting
the output Z ∈ RB×C from the backbone into the proposed
base embedding generation (BEG) block. A novel specificshared knowledge distillation (SKD) loss is proposed to ensure that the generated detail(base) features contain as much
modality-specific (modality-shared) knowledge as possible,
thereby effectively leveraging modality-specific and shared
information. Furthermore, we construct a modality-shared
feature Z F using a cross-modality feature fusion method to
optimally supplement the base features. During the inference phase, only the comprehensive feature Z yielded by the
backbone is used for performance evaluation. This is because
the proposed DFE and BEG modules effectively enhance the
comprehensive feature by incorporating additional detail and
base information.
Given an identity image from either the visible or infrared modality, VIReID intends to identify the most similar sequence of that identity in another modality. Let

cat

Z M (1 : C)

ZD

Z̄ D

GAP

lorth
Independence Restrict

Z̄ B
I −P
Z

ZFB

Projection matrix

Detail Feature

Base Feature

lokl
lDF E

RGB Feature

lBEG
lf bkl

CLSB

lskd
ZFB
Base Subspace

Z̄FB

Base Embedding Generation

Comprehensive Feature

ltri

Z̄ B

Batch Attention Channel Attention

Middle Feature

CLSD

ZkM (c + 1 : C)

P

lid

lapp

ZD

lcorr

Z1M (c + 1 : C)

Z̄ D

Correlation Restrict

Gap

Cross Attention

INN Block

Detail Subspace

ZkM (1 : c)

Layer Normalization

INN Block

ZM

Layer Normalization

Detail Feature Extraction
Z1M (1 : c)

Conv Block 4

Conv Block 1
Conv Block 2
Conv Block 3

Backbone Network

IR Feature

Multiplication

Addition

Figure 2: The pipeline of the proposed Base-Detail Feature Learning Framework (BDLF), which consists of a Detail Feature Extraction
(DFE) module and a Base Embedding Generation (BEG) block, and jointly optimizes the extracted detail, base, and comprehensive features.

the training set {XV , XI } consist of B identities, with
each identity including P samples. Therefore, XV =

xV b,p , b = 1, ..., B; p = 1,
 ..., P symbolizes the set of visible images, while XI = xI b,p , b = 1, ..., B; p = 1, ..., P
denotes the set of infrared images. As illustrated in Figure 2,
the VIS and IR images are processed through the backbone
network, i.e,
ZVM/I =E f ore (XV /I )
ZV /I =E rear (ZVM/I )
Z = cat(ZV , ZI )

(1)

where E f ore (·) and E rear (·) are the former and latter
parts of the backbone network, the embeddings ZV /I M ∈
B

B

R 2 ×C×H×W and ZV /I ∈ R 2 ×C denote the intermediate
and complete outputs from the backbone for the VIS and IR
modalities, cat(·) refers to the concatenation operation along
the batch dimension.

3.2

Specific-shared Knowledge Distillation

We observe that the similarity of base information, such as
contours and movements, between the VIS and IR modalities is noticeable. In contrast, the similarity of detail information including color, texture, and thermal details between
the two modalities is suppressed. Inspired by [Zhao et al.,
2023], as shown in Figure 3, the base and detail features can
be generated by increasing and reducing the correlation between the two modalities respectively. Based on this, we
propose a novel specific-shared knowledge distillation (SKD)

loss, which is numerically smoother and easier to optimize,
formulated as follows:
log[Corr(ZVB , ZIB )]
lskd = q
3
log[Corr(ZVD , ZID )] + γ

(2)

in which ZVB/I denotes the base features generated by the
proposed BEG block, and ZVD/I denotes the detail features
extracted from the proposed DFE module. Corr(·) is the
Pearson correlation coefficient operation, while γ represents a
constant that ensures the denominator remains non-zero. According to optimize the SKD loss, the correlation between
the VIS and IR modalities of both base and detail features(i.e,
Corr(ZVB , ZIB ) and Corr(ZVD , ZID ) in formula (2)) is simultaneously increased and decreased. This approach allows the
proposed DFE module to extract embeddings rich in detailed
knowledge. Consequently, the proposed BEG block is capable of generating base embeddings that contain a greater
amount of modality-shared knowledge.

3.3

Detail Feature Extraction

The proposed DFE module aims to acquire detail features
that imply modality-specific information from the intermediate embedding ZVM/I by utilizing a series of invertible neural network (INN) blocks[Zhao et al., 2023][Dinh et al.,
2017][Zhou et al., 2022], which can effectively preserves detailed characteristics and mitigates information loss during
feature extraction by making its input and output embeddings
are mutually generated. Taking the VIS case as an exam-


exec
/bin/zsh -lc "pdftotext -l 3 'Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10887789

Attribute Conditional Diffusion-Augmented Person
Re-Identification
Shijie Nie1 , Ziqiang Shi1 , Rujie Liu1 , Song Guo1 , Meng Zhang1 , Mengjiao Wang1 ,
Kazuki Osamura2 , Lina Septiana2 , Abe Narishige2
1 Fujitsu Research & Development Center, China. 2 Fujitsu Limited, Japan
nieshijie@fujitsu.com
Abstract—Due to privacy and cost issues, the lack of largescale labeled datasets limits the advancement of person reidentification. Existing methods use generative adversarial networks or game engine rendering for data augmentation to improve re-identification performance. However, these approaches
struggle to maintain realistic images. This paper introduces a
novel approach called Identity Diffuser, which uses diffusion
models to generate synthetic data for the same identity with
different poses. Our proposed framework incorporates identityspecific embeddings and target poses into the diffusion process,
enabling the generation of realistic and diverse images that
consistently preserve identity features. Guided by pretrained
re-identification net and target pose heatmap, the framework
learns transformation trajectories through forward and backward denoising steps in the diffusion models. This approach
effectively maintains key pedestrian attributes across various
poses. Experimental results on the Market1501 and DukeMTMC
datasets demonstrate a notable improvement in performance,
with a 1.73%/0.80% mAp increase in Market1501/DukeMTMC
datasets compared with current state-of-the-art method. When
less real data is included, the increment can be 5.1%/1.5%,
separately.
Index Terms—Person re-identification, conditional diffusion
models, attribute augmentation

poses. Due to one-step generation process, these methods
face challenges with image diversity and quality, and GANs
often struggle with convergence instability and mode collapse.
For example, PG-GAN often produces striped noise or blurry
images. Recently, diffusion models [6] have shown promise
in generating high-quality images [7] and have been applied
to various tasks [8]. Diffusion-based methods for pedestrian
synthesis, such as PIDM [9] (Fig. 1(c)), PCDM [10], and
IMAGPose [11] have made progress. However, controlling
the generation process to ensuring identity consistency across
different poses for re-identification remains challenging.

(a) Randperson
Rendering based methods

(b) PG-GAN
GAN based methods

(c) PIDM
Diffusion based methods

(d) Identity Diffuser
Our methods

I. I NTRODUCTION
Person re-identification involves matching individuals
across different cameras, a task complicated by variations
in pose, low-resolution surveillance footage, occlusions, and
other factors. While deep learning has driven significant
progress in this field [1], the shortage of large-scale labeled
datasets remains a significant challenge. Privacy concerns and
the high cost of manual annotation further complicate data
collection.
To address this issue, synthetic data generation methods
have been explored, broadly categorized into model-based
rendering and data-driven generative approaches. Model-based
rendering methods, such as PersonX [2] and Randperson [3]
shown in Fig. 1(a), create large-scale synthetic data by simulating diverse human poses and environments. While these
datasets improve re-identification performance when combined
with real data, they rely on expensive and complex 3D models
for high-quality results.
On the other hand, data-driven generative methods, especially those using GANs, offer an alternative approach.
Techniques like FD-GAN [4] and PG-GAN [5] (Fig. 1(b))
focus on augmenting data by generating images in various

Fig. 1. A visual comparison of various typical re-identification synthetic
data generation methods highlights specific challenges. The PIDM dataset
occasionally suffers from background noise and loss of cloth texture detail.
GAN-based methods often produce artifacts, such as striped images, while
rendering engine-based methods face issues with coarse 3D human modeling.

In this work, we propose the Identity Diffuser framework,
which takes the first step in integrating prior knowledge
from the identity embedding space into diffusion models
to generate high-quality pedestrian images with consistent
identity features. In scenarios where less real data per person
is available, our approach can generate novel images given
arbitrary poses and a source identity image. Our approach uses
a pre-trained encoder to guide the diffusion process, ensuring
identity consistency across pose variations.
Our contributions include the development of the Identity
Diffuser model, which outperforms state-of-the-art methods
in generating pedestrian images. Experiments on public reidentification datasets, such as Market1501 and DukeMTMC,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.

Real dataset

Reid
model

Train 1

Synthetic data
Finetune 4

ID embedding 2

Image pairs 2

Augmentation

Pre-trained
Reid-net

C

xs

Input
1

2

Diffusion model
3

4

Fig. 2. The proposed person re-identification framework consists of four
steps: training a re-identification model on a real dataset, using pose and
image embeddings to train a diffusion model, generating synthetic images
with the diffusion model, and pre-training the re-identification model with
these synthetic images before fine-tuning on the real dataset.

demonstrate that our model improves mAP by 1.73%/0.8%,
with even larger gains of up to 5.1%/1.5% when less real data
is available.
II. M ETHODS
In this section, we first present an overview of our generative
pipeline for pedestrian synthetic data generation, followed by
a detailed explanation of the model architecture and training
process.
A. Overview
Fig. 2 illustrates the pipeline. Starting with a real pedestrian dataset, We then randomly sample source/target image
pairs with same identity for training diffusion models. In
the following sections, we will not distinguish between the
source ID and target ID, as they are the same. We train a reidentification model [12] and use its backbone to extract source
identity embedding. OpenPose [13] is applied to extract pose
information from the target image. By combining the source
image, target pose map and identity embedding as input, we
train a diffusion model to generate synthetic pedestrian images
that matches the target image. Finally, we create a synthetic
dataset conditioned on arbitrary poses and source images to
pre-train the re-identification model, which is later fine-tuned
with the real dataset. Note that in the training stage, pose and
target image is aligned, while in the inference stage is not. The
detailed model architecture and training process are shown in
Fig. 3.
B. Human Attribute-conditioned Diffusion Model
In the detailed framework shown in Fig. 3, a pre-trained
ResNet model is used to extract identity embeddings, which
are then processed by an ID adapter and expanded to match
the dimensions of the source image. These embeddings are
combined with the pose heatmap and the source image using
channel-wise concatenation. The combined inputs are further
refined through global source image feature extractor using
cross-attention. The architecture is designed to estimate noise
and progressively update the image across multiple timesteps
during the diffusion process.

Timesteps

t

ID adapter

3

Pose 2
Condition
Random pose pool

xid

t+1
θ (yt , t, xp , xs )

yt

xp

Cross-Attention Estimated noise

Global feature
Extractor

Fig. 3. The Identity Diffuser model uses a pre-trained re-identification net to
extract identity embeddings, processes them with an ID adapter, and combines
them with the pose heatmap and source image. These inputs are refined
through a global feature extractor through cross-attention, and the model
estimates noise to iteratively denoise the image during the diffusion process.

To formulate this process, our objective is to train a diffusion
model G. Given a source image xs , a target pose xp , and
a target image y, the goal is to generate a synthetic image
xt that matches the pose of xp while preserving the identity
embedding of xs . The model is composed of three main
components: an offline pose extractor Ep , an identity encoder
Eid , and a diffusion model generator G. Let Pi represent the
distribution of all images corresponding to identity si with
arbitrary poses, and let Yi = {yij ∈ Pi }N
j=1 denote the set of
target images that share the same identity as the source images
but with different poses. Our objective is to train the generator
G such that it can produce synthetic images conditioned on
the identity representation and poses extracted from Yi . The
problem can be defined as follows:
G(ϵ, Eid (xs ), Ep (Yi )) ∼ Pi

(1)

where ϵ is random noise, starting with pure Gaussian noise
ϵ ∼ N (0, I). Let G = pθ (y|xs , xp , zid ) represent a diffusion
model conditioned on the target pose xp , source image xs , and
source identity embedding zid , which is extracted by Eid :
zid = Eid (xs )

(2)

Suppose the denoising diffusion probabilistic model
(DDPM) adds noise from y0 ∼ q(y0 ) to an isotropic Gaussian
noise yT ∼ N (0, I) in T steps. The forward process is:
q(yt |yt−1 ) = N (yt ;

p

1 − βt yt−1 , βt I)

(3)

where t ∼ [1, T ] and β1 , β2 , ..., βT is a fixed variance schedule
with
Qt βt ∈ (0, 1). Using the notation αt = 1 − βt and ᾱt =
i=1 αi , we can sample from q(yt |y0 ) in a closed form at an
arbitrary timestep t:
√
q(yt |y0 ) = N (yt ; ᾱt y0 , 1 − ᾱt I)
√
√
= ᾱt y0 + 1 − ᾱt ϵ

(4)

In the denosing process, the slightly denoised yt−1 is
sampled from yt , from the distribution pθ (yt−1 |yt ). This can
be approximated by a deep neural network to predict the mean
and variance, and is parameterized as:

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.

pθ (yt−1 |yt , xp , xs , zid ) = N (yt−1 ; µθ (yt , t, xp , xs , zid ),
Σθ (yt , t, xp , xs , zid )).

feature extractor, based on ResNet, maps inputs to specific
resolutions: 32x32, 16x16, and 8x8, which are then combined
with predicted noise using attention blocks.

(5)
Instead of directly predict the mean and variance, we predict
the ϵ in Eq. 4. The loss function is defined as:
2

Lmse = Eyt ,t,xp ,xs ,zid ,ϵ ∥ϵ − ϵθ (yt , t, xp , xs , zid )∥ .
The mean µθ can be calculated from ϵθ as:


1 − αt
1
yt − √
µθ (yt , t) = √
ϵθ (yt , t) .
α
1 − αt

(6)

(7)

We have ommitted various conditions in Eq. 6. In DDPM,
the variance Σθ is fixed, but in the improved DDPM [14], the
variance is learnable by an additional term Lvlb with relatively
small weight. Substitute mean and variance into Eq. 5, the
iterative denoising equation is:
yt−1 = µθ (yt , t) + Σθ ϵ,

ϵ ∼ N (0, I).

(8)

As shown in Fig. 3, we propose to use pose xp and id
embedding zid as the condition for the diffusion model. These
conditions are found to be effective in controlling the diffusion
models, with detailed experiment results shown in Sec. III.
a) Classifier-free Guidance (CFG): CFG [15] is an extended version of classifier-guidance [7], where constraint
information is incorporated during training, enhancing the
ability to generate detailed image structures. As is well known,
CFG performs better than classifier-guidance, but requires a
extra training stage. Given the relatively small scale of the
re-id dataset, we introduce pose and ID embedding guidance
during the training stage to ensure stricter alignment. Inspired
by CFG to achieve pose and identity jointly sampling during
training, we use the following equation to compute ϵθ :
ϵθ = ω1 ϵθ (yt , t, xp , xs , zid ) + ω2 ϵθ (yt , t, xp )
+ (1 − ω1 − ω2 )ϵθ (yt , t).

III. E XPERIMENTS
A. Dataset
We evaluate using two public re-identification datasets:
Market-1501 [18] and DukeMTMC-reID [19]. Market-1501
has 12,936 training images and 19,732 gallery images across
1,501 identities from six cameras. DukeMTMC-reID contains
36,411 images of 1,404 identities from eight cameras. We
apply the standard training/testing split for both datasets. Poses
are extracted with OpenPose [20], and we use a pretrained
ResNet50 [17] with default settings from BoT [12] for identity
feature extraction. About 52k synthetic images are generated
by 4x NVIDIA A40 GPUs , the estimated time usage is about
4 hours.
B. Visual and Quantitative Evaluation
We visually compare Identity Diffuser with the current stateof-the-art PIDM [9], as shown in Fig. 4. Each row depicts a
different person synthesized with random poses. Our method
produces more stable and consistent pedestrian images. We
use the DDIM sampling method with 100 sampling steps,
consistent with PIDM’s settings.
Identity Diffuser

PIDM

(9)

In the code, we set zero tensor for condition that is not used.
ω1 = 0.1 and ω2 = 0.9 are the hyperparameters to control the
weight of the conditions.
b) Network Architecture: The components in our framework shown in Fig.3 are primarily based on Guided
Diffusion[7], U-Net [16], and ResNet50 [17]. All input images are resized to a resolution of 128x64 pixels. A U-Net
structured network maps the input image, conditions, and
timestep embeddings to predict noise at specific timesteps.
The pre-trained reid-net in the figure serves as the backbone of
ResNet50, mapping inputs to a 2048-dimensional embedding
vector. This net is freezed during training. The ID adapter
consists of several bilinear upsampling and convolution layers,
mapping the embedding vector back to the original image size
with a channel number of N = 10. The pose is processed
as a heatmap of the same size as the input image, with the
number of channels equal to the keypoints number. The global

Fig. 4. Visual comparison of Identity Diffuser and PIDM[9] for market1501
generation

Frechet Inception Distance (FID) are used to evaluate the
realism of the generated images. FID measures how close the
distribution of generated images is to the real. The FID of our
method and other methods are shown in Table. I. Comparing
our method with the current methods, the realism (FID) is
superior. This finding suggests that our approach is capable of
producing more authentic human images.
C. Re-Identification Performance
ReID accuracy, measured by mAP (mean Average Precision), is used to evaluate the impact of generated data on
person reID tasks. A random test image and 20 poses from
real training data are fed into the diffusion model to generate
a synthetic training set Dsyn .
We initialize a ResNet50 network with Dsyn and randomly
select 20%, 40%, 60%, and 80% of the real re-identification

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Beyond geometry - The power of texture in interpretable 3D person ReID.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 261 (2025) 104517

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

Beyond geometry: The power of texture in interpretable 3D person ReID
Huy Nguyen ∗, Kien Nguyen, Akila Pemasiri, Sridha Sridharan, Clinton Fookes
Signal Processing, Artificial Intelligence and Vision Technologies (SAIVT), Queensland University of Technology (QUT), Brisbane, QLD 4000, Australia

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
3D person re-identification
Texture UV mapping
3D explainability

ABSTRACT
This paper presents FusionTexReIDNet, a robust framework for 3D person re-identification that uniquely
leverages UVTexture to enhance both performance and explainability. Unlike existing 3D person ReID
approaches that simply overlay textures on point clouds, our method exploits the full potential of UVTexture
through its high resolution and normalized coordinate properties. The framework consists of two main
streams: a UVTexture stream that processes appearance features and a 3D stream that handles geometric
information. These streams are fused through an effective combination of KNN, attribute-based, and explainable
re-ranking strategies. Our approach introduces explainability to 3D person ReID through the visualization of
activation maps on UVTextures, providing insights into the model’s decision-making process by highlighting
discriminative regions. By incorporating the Intersection-Alignment Score derived from activation maps and
visible clothing masks, we further improve the ReID accuracy. Extensive experiments demonstrate that
FusionTexReIDNet achieves state-of-the-art performance across various scenarios, with Rank-1 accuracies of
98.5% and 89.7% Rank-1 on benchmark datasets, while providing interpretable results through its explainable
component.

1. Introduction
Person re-identification (ReID) is the task of recognizing individuals across different non-overlapping camera views in a surveillance
system (Nguyen et al., 2022, 2023). This task involves matching individuals based on their visual characteristics, such as appearance, body
shape, and clothing, across different camera views. The importance
of person ReID has been highlighted by its security surveillance applications, which has spurred significant progress in the field, driven
by the availability of larger and more diverse datasets (Zheng et al.,
2015; Ristani et al., 2016; Nguyen et al., 2024) and advancements in
deep learning techniques for person representation (Khatun et al., 2020;
Hafner et al., 2022; Liu et al., 2023b).
While 2D person ReID methods have advanced significantly (Zhang
et al., 2023; Lin et al., 2023; Weng et al., 2023), they miss a crucial
aspect: humans are inherently three-dimensional (3D) entities. Traditional 2D ReID approaches often struggle with unpredictable viewpoint
changes and geometric deformations caused by camera positioning,
which we identify as significant factors responsible for poor generalization performance across domains. Shifting person ReID from 2D to
3D space can enhance the field by leveraging the additional information
provided by 3D human models. These models can normalize variations
in appearance due to human movements, dynamics, camera distances,
and lighting conditions, providing a consistent identity representation (Loper et al., 2015; Xu and Loy, 2021). Moreover, 3D models

can match identities across different viewing modalities, such as aerial
versus ground-based perspectives, by accounting for the changes in visible body parts and explicitly addressing viewpoint variations (Nguyen
et al., 2023, 2024).
However, state-of-the-art (SOTA) 3D person ReID approaches still
lag behind their 2D counterparts in performance and face significant
challenges in handling unpredictable viewpoint changes across domains. Most existing 3D person ReID methods (Wang et al., 2023;
Zheng et al., 2020; Chen et al., 2021a) simply reconstruct 3D models
from input images and directly perform classification on them without
explicitly addressing the viewpoint normalization problem. For example, PointReIDNet (Wang et al., 2023), 3DInvarReID (Liu et al., 2023a)
and OG-Net (Zheng et al., 2020) first infer an SMPL-based 3D human
model (Loper et al., 2015) from the input image, overlay the 3D model
with texture, encode the overlayed 3D point cloud, then classify the
encoded vector for person ReID. However, these approaches fail to
leverage the explicit viewpoint alignment capabilities that 3D reconstruction can provide, missing the opportunity to normalize geometric
deformations and appearance variations caused by camera positioning.
Unlike existing methods that simply overlay textures on 3D models
or ignore viewpoint variations, our work addresses these fundamental
limitations through explicit 3D viewpoint alignment and robust texture
representation. We propose FusionTexReIDNet, an innovative method

∗ Corresponding author.

E-mail address: nguyet91@qut.edu.au (H. Nguyen).
https://doi.org/10.1016/j.cviu.2025.104517
Received 9 December 2024; Received in revised form 20 August 2025; Accepted 18 September 2025
Available online 26 September 2025
1077-3142/© 2025 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license
(http://creativecommons.org/licenses/by/4.0/).

H. Nguyen, K. Nguyen, A. Pemasiri et al.

Computer Vision and Image Understanding 261 (2025) 104517

• We pioneer the integration of explainability into 3D person ReID
by developing a visualization technique that projects activation maps onto UVTextures and 3D models in canonical viewpoints. This provides clear insights into discriminative features
free from viewpoint-induced distortions, enabling better understanding of model decisions and facilitating performance improvements through explainable re-ranking strategies.
• We achieve unprecedented performance across seven diverse
datasets spanning ground–ground, aerial–aerial, and aerial–ground
scenarios, with our method reaching 98.5% Rank-1 accuracy
on Market-1501 and 89.7% on challenging AG-ReID.v2. These
results demonstrate the first instance where 3D person ReID
methods surpass their 2D counterparts through explicit viewpoint
alignment.
2. Related work
This section reviews and discusses prior work in person ReID in the
3D space and explainability in person ReID.
Person ReID in 3D. Literature of existing approaches in 3D person
ReID can generally be organized into two main categories based on
their utilization of the reconstructed 3D human models:
(1) Direct 3D model-based classification: Leveraging 3D information
in person ReID provides an effective approach by utilizing shape and
spatial depth features, which are particularly robust to variations in
texture. Point clouds, recognized for their depth information, add a
valuable dimension to data representation (Qi et al., 2016). Innovations
such as OG-Net by Zheng et al. (2020) convert 2D images into a 3D
data framework, incorporating structural and appearance information.
This approach is further explored by Liu et al. (2023a) and Wang
et al. (2023), who delve into distinguishing identity-specific features
from changeable aspects like clothing and posture in 3D shape representations. These advancements highlight the significant potential of
3D information in enhancing person ReID beyond the specific challenges of occlusion. However, these methods often struggle to capture
fine-grained texture details, which are crucial for accurate person ReID.
(2) 3D model-based data augmentation: Addressing occlusions in 3D
Re-ID, contributions such as PersonX (Sun and Zheng, 2018) and Wang
et al. (2022a) enhance data representation by utilizing 3D scanning
and UV mapping techniques, further enriched by the integration of
point clouds to incorporate depth information. Works like ASSP (Chen
et al., 2021a) and JGCL (Chen et al., 2020) show advancements in
merging 2D and 3D data, applying adversarial and contrastive learning
strategies, while Zhang et al.’s 3DT model (Zhang et al., 2022a) tackles
the challenge of group ReID in occluded environments through 3D
transformations. TranSG (Rao and Miao, 2023) introduces a focus on
utilizing 3D skeleton data for Re-ID, emphasizing skeletal graphs and
spatial–temporal semantics. However, it potentially overlooks crucial
details like body size, hairstyle, and clothing, pointing to the nuanced
challenges of clothing representation and diverse body movements in
3D person ReID research. Notably, while there is significant progress,
studies focusing on 3D ReID using video inputs, such as those by Liao
et al. (2019), Han et al. (2022), outline the breadth of methodologies
beyond the current discussion’s scope, highlighting the dynamic and
evolving nature of the field. Despite the advancements in synthesizing additional data, these methods often lack the ability to generate
highly realistic and diverse textures, which limits their effectiveness in
real-world scenarios.
Explainability in Person ReID. The blackbox nature of CNN models (Somers et al., 2022; Wang et al., 2021) has led to a growing
interest in developing explainable approaches for person ReID. Existing
works attempt to explain CNN models through various techniques,
such as visualizing salient maps (Selvaraju et al., 2019), distilling
knowledge (Chen et al., 2019a), or learning with decision trees (Zhang
et al., 2019a). In the person ReID domain, attention learning has been
investigated as a means to explain the model’s predictions (Zhang et al.,

Fig. 1. Overview of the proposed explainable 3D person ReID approach.
From person images captured at different viewpoints, our method reconstructs
3D human models and generates high-resolution UVTexture representations.
The framework leverages three key components: (1) UVTexture maps that
provide viewpoint-invariant appearance representation, (2) visible clothing
masks that identify non-occluded regions, and (3) activation map alignment
that highlights discriminative features. Unlike 2D methods that suffer from
viewpoint-dependent activation patterns and background bias, our 3D approach enables spatially consistent feature analysis across different camera
angles, providing interpretable insights into which body regions contribute to
identity matching.

that leverages 3D human models to perform explicit viewpoint normalization, projecting pedestrian images from arbitrary camera views into
canonical viewpoints to mitigate geometric deformations and appearance discrepancies. Our approach combines UVTexture representation
with 3D viewpoint alignment to achieve improved performance and
explainability, as illustrated in Fig. 1. By utilizing the high-resolution
and normalized coordinate properties of UVTexture, our method captures fine-grained appearance details while enabling precise localization of discriminative features on the human body. Additionally, we
introduce a transformer-based fusion module that compensates for
reconstruction errors by aligning and fusing visual cues from original
and canonical view images, ensuring robust performance even when
3D reconstruction is imperfect.
With these contributions, we make 3D person ReID performance
surpass 2D counterparts for the first time while introducing explainability through viewpoint-aware feature analysis. Our work addresses the
critical challenge of unpredictable viewpoint changes in cross-domain
scenarios and sets a new benchmark for 3D person ReID systems. The
core contributions are summarized as follows:
• We introduce a novel 3D viewpoint alignment framework that
explicitly addresses unpredictable camera view changes by projecting pedestrian images from arbitrary viewpoints into canonical views. This approach mitigates geometric deformations and
appearance variations caused by viewpoint changes, achieving
superior generalization performance across domains with up to
7.7% improvement in Rank-1 accuracy over state-of-the-art 3D
approaches.
• We develop a dual-stream architecture that strategically combines
explicit viewpoint normalization with high-resolution UVTexture
representation. Our transformer-based fusion module enables effective alignment and information compensation between original and canonical view images, addressing reconstruction errors
while preserving discriminative features for accurate identification.
2

H. Nguyen, K. Nguyen, A. Pemasiri et al.

Computer Vision and Image Understanding 261 (2025) 104517

Fig. 2. The proposed model has two streams: UVTextureNet extracts appearance features 𝐹𝐴 from UVTexture images, while 3DReIDNet processes 3D rendered
point clouds using KNN graphs to extract geometry features 𝐹𝑆 . Each stream produces distance matrices (𝐷1 , 𝐷2 ), fused to yield joint distance matrix 𝐷. The
Explainable Module overlays attention maps on UVTexture, visualizing model focus and incorporating visibility masks for enhanced explainability. It computes
Intersection-Alignment Score (IAS) from IoU between attention maps and visible clothes masks, plus activation alignment scores. Three re-ranking methods (KNN,
attribute, explainable) refine distance matrix 𝐷, with IAS improving ReID performance.

2020; Nguyen et al., 2023). Some notable examples of explainable
techniques in the 2D domain include Grad-CAM (Selvaraju et al.,
2019), which highlights areas of interest in images, and attributebased methods that visualize differences or similarities between person
matching (Chen et al., 2021b; Nguyen et al., 2023). Despite the progress
made in explainable person ReID, the majority of these approaches
focus on 2D image-based methods, and there are limitations to their
effectiveness. For instance, 2D explainable methods often struggle to
capture the complex spatial relationships and depth information that is
crucial for understanding person appearance and behavior in real-world
scenarios. The problem of domain generalization in person ReID has
gained increasing attention (Bhuiyan et al., 2024; Liu et al., 2024), particularly when dealing with varying viewpoints and camera conditions.
Additionally, 2D methods may be sensitive to variations in viewpoint,
occlusion, and illumination, which can hinder their ability to provide
reliable explanations.
In contrast, 3D data offers potential for explainability that 2D
does not, presenting a more comprehensive view of the spatial relationships and depth information that are crucial for understanding
complex scenes. Our paper investigates this potential, exploring how
3D data can enhance the explainability of person ReID models by
leveraging the additional dimension to provide deeper insights into
the model’s decision-making processes. The limitations of existing 2D
explainable methods and the lack of established 3D explainable approaches motivate our proposed UVTexture-inspired explainable person
ReID method, which aims to bridge this gap and provide a more
comprehensive understanding of the ReID process. By utilizing the
rich information available in 3D data, our approach addresses the
limitations of 2D methods and offers explainability in person ReID.

3.1. Preliminaries: UVTexture and 3D reconstruction
Inputs for our model are 3D human models and UVTexture. To
reconstruct 3D human models and UVTexture for our inputs, we employ
RSC-Net (Xu et al., 2021) and Texformer (Xu and Loy, 2021). It is worth
noting that other models and approaches can also be used. RSC-Net
is an algorithm designed to address the challenges of estimating 3D
human pose from low-resolution images and videos (Xu et al., 2021,
2020). RSC-Net achieves impressive results even when working with
low-resolution input data by integrating a resolution-aware network
that adapts to different resolutions, employing self-supervision loss, and
leveraging contrastive learning for high-quality 3D reconstructions.
For creating 3D textures via UVTexture maps for humans, we adopt
a transformer-based framework, Texformer (Xu and Loy, 2021). Texformer leverages global information, enabling the generation of realistic texture maps from input images. Recent advances in transformer
architectures have shown promising results in person ReID (Li et al.,
2024; Lin et al., 2024), particularly in handling complex visual relationships and feature extraction. Unlike traditional UV maps, which may
lack spatial consistency across different individuals and conditions (Xiu
et al., 2023; Pumarola et al., 2019), the UVTexture maps produced
by Texformer are particularly effective for supporting the downstream
task of explainable 3D person ReID due to their enhanced spatial
consistency.
3.2. Framework overview
This section presents the overall architecture of FusionTexReIDNet, a comprehensive framework for 3D person re-identification that
uniquely leverages both 3D structural information and high-resolution
UVTexture details. The proposed framework addresses the fundamental
limitations of existing 3D person ReID methods by introducing explicit
viewpoint normalization and robust texture representation through a
dual-stream architecture. As illustrated in Fig. 2, FusionTexReIDNet
consists of four main components working in concert: (1) 3D reconstruction and UVTexture generation — the preprocessing stage that transforms
2D pedestrian images into 3D human models with corresponding UVTexture maps using RSC-Net and Texformer; (2) Dual-stream feature extraction — comprising Stream 1 (UVTextureNet) for appearance feature
extraction and Stream 2 (3DReIDNet) for geometric feature processing; (3) Multi-level fusion module — integrating distance matrices from
both streams through weighted combination and multiple re-ranking
strategies; and (4) Explainable module — providing interpretability
through UVTexture-based activation map visualization and explainable
re-ranking using the Intersection-Alignment Score (IAS). A detailed
breakdown of the method components is provided in Table 1.

3. Methodology
In this section, we will elaborate on the proposed FusionTexReIDNet
framework and its key components in detail. First, we will introduce
the preliminaries of UVTexture and 3D reconstruction methods in
Section 3.1, which establish the foundational concepts essential for
understanding our approach. Then, we will provide a comprehensive
overview of the overall framework architecture in Section 3.2. Subsequently, we will detail the two main processing streams: the UVTextureemphasized ReID stream in Section 3.3 and the 3D point cloud processing stream in Section 3.4. Finally, we will present our novel explainable module that provides interpretability through UVTexture-based
visualization in Section 3.5.
3


codex
我只读了每篇前 3 页。结论先说：这批论文里最值得借鉴的不是某个模块，而是三种发 B 类的写法，分别是把困难样本重新定义清楚，把跨模态差异拆成“该对齐”和“不该抹掉”的两部分，把 3D 从辅助信息提升成“可观测证据”的组织方式。

**逐篇反推**

1. `Adaptive Occlusion-Aware Network`
创新类型是工程组合偏新机制。能发是因为它抓住了手工分块错位、姿态估计噪声大且忽略背包等非人体细节这两个 gap，用自适应部位提取和遮挡权重讲通。story 是“不依赖重姿态分支，也能找到可见局部并抑制遮挡局部”。

2. `Adaptive Pseudo-Label Purification and Debiasing`
创新类型是新机制。它把无监督可见光红外的伪标签噪声拆成三个后果，记忆库错误、跨模态对应错误、负样本采样偏置，然后分别设计净化、最优传输匹配、去偏学习。story 是“先校准再分流，干净标签对齐，噪声标签只学邻域关系”。

3. `Adaptive transformer with Pyramid Fusion`
创新类型是新机制加新数据设定，但模块味较重。能发主要靠换衣场景下局部细节和全局语义协调不足，以及新加风雨换衣数据集。story 是“换衣不只是不看衣服，还要动态融合 Transformer 多层和频域局部线索”。

4. `Adversarial flow-based generative models`
创新类型是新机制和生成增广。gap 是可见光红外数据少、模态差异大，GAN 非可逆且红外到可见光困难。它用双 flow 共享高斯空间，同时做扩样和跨模态翻译。story 是“可逆潜空间把跨模态生成和训练样本生成统一起来”。

5. `Adversarial perturbation and defense`
创新类型是问题重定义加新机制。它把泛化 ReID 的困难点定义成“身份相关和无关特征在困难匹配样本中纠缠”，再用度量关系反转但外观尽量保持的生成器制造困难样本。story 是“先主动制造最难的度量样本，再训练模型防御和净化身份语义”。

6. `Attribute Conditional Diffusion-Augmented`
创新类型是生成增广机制。gap 是标注数据少，渲染和 GAN 的真实感、身份一致性不足。它用 ReID 身份嵌入和姿态热图条件扩散，生成同身份不同姿态样本。story 是“身份保持的任意姿态扩散增广，低数据时更有价值”。

7. `Attribute Guidance with Inherent Pseudo-Label`
创新类型是新机制，但赛道拥挤。gap 是 CLIP 这类视觉语言模型偏全局语义，遮挡下忽略细粒度属性，而人工属性标注成本高。story 是“不加人工属性，从 CLIP 内部挖细粒度属性伪标签，补全全局提示的盲点”。

8. `Base-Detail Feature Learning Framework`
创新类型是问题重定义加新机制。它最有启发：跨模态不是把差异全抹掉，模态共享的轮廓和运动是 base，颜色、纹理、热特征是 detail，二者都要保留。story 是“对齐共享基底，同时保留模态特异细节”。

9. `Beyond geometry`
创新类型是问题重定义加新机制。它直接撞我们的 3D 和 aerial-ground 资产，核心是 3D ReID 不该只用点云几何，应该把纹理放到标准 UV 坐标里，并用可解释激活图和可见衣物 mask 辅助。story 是“3D 的价值不是几何本身，而是标准表面坐标里的纹理和可解释比较”。

**强候选方向**

1. Headline：ReID 应该比较“共同可观测的身份证据”，而不是强行比较所有 2D 区域。  
挂靠资产：SMPL 3D 几何、pose 热图门控、aerial-ground、SOLIDER-Swin。  
区别：它和 AOANet 的区别是不用 2D 自适应分块讲遮挡，而是用 SMPL 表面可见性定义哪些证据可比较。它和 `Beyond geometry` 的区别是避开完整 UVTexture 和重排序，重点做低清航拍到地面里的可观测性约束，目标是训练出单 embedding 或表面 token embedding。  
cheap kill-switch：零训练先用现有 SOLIDER token 加 SMPL 可见表面 mask，在 AG-ReID.v2 或 CARGO 上只看大视角差、低可见重叠子集。如果共同可见区域相似度不能减少明显误匹配，这条先降级。

2. Headline：aerial-ground 跨视角不是“消除视角差异”，而是学习 3D base 和视角 detail 的可控分工。  
挂靠资产：SOLIDER-Swin、SMPL mesh/joints、aerial-ground。  
区别：最像 BDLF，但 BDLF 是可见光红外，用相关性约束拆 base/detail；我们用物理 3D 可见性和视角投影拆“人体几何基底”和“视角特异纹理残差”。也区别于 `Beyond geometry`，我们不是把 UV 纹理当主要表征，而是让 3D 几何决定哪些视觉细节该保留、哪些该去偏。  
cheap kill-switch：冻结 SOLIDER，训练很小的两头线性探针，一个头吃 SMPL 几何或可见性统计，一个头吃视觉残差。若 base-only 和 detail-only 在 hard subset 上没有互补性，融合也吃不掉原 SOLIDER 的错误，这条不要硬做。

3. Headline：最有价值的困难样本不是随机遮挡或风格扰动，而是 3D 相机视角导致的身份证据反转。  
挂靠资产：SMPL 3D、aerial-ground、pose 热图门控。  
区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
cheap kill-switch：按 SMPL 可见性做 token dropout 或 surface dropout，小规模训练 5 到 10 个 epoch。必须证明它比随机擦除、随机 token dropout 更能改善极端视角子集，否则只是普通增广。

4. Headline：伪标签可靠性应该由“几何上是否看得到同一身份证据”决定，而不只由特征置信度决定。  
挂靠资产：SMPL 3D、pose 热图门控、aerial-ground，必要时接 SOLIDER 特征。  
区别：最像 APPD 和 AG-ReID，但它们净化伪标签主要看特征、属性或语义置信度。我们用 3D 可见性判断一个跨视角伪匹配是否有足够共同证据，尤其适合无监督 aerial-ground。  
cheap kill-switch：在有标注 AG 数据上模拟无监督聚类，比较“特征置信度”和“几何可观测性分数”对伪标签正确性的预测能力。如果几何分数的干净样本筛选纯度没有明显提升，这条不做主线。

我建议优先压第 1 和第 2。第 1 的问题定义最清楚，和团队的 pose gating、SMPL、aerial-ground 都能挂上；第 2 的论文叙事更像 Pattern Recognition/TMM，能把“不要过度对齐”讲成机制。第 3 可以作为训练机制配套，第 4 更像无监督扩展，除非 cheap kill-switch 很强，否则不要先押。
tokens used
80,691
我只读了每篇前 3 页。结论先说：这批论文里最值得借鉴的不是某个模块，而是三种发 B 类的写法，分别是把困难样本重新定义清楚，把跨模态差异拆成“该对齐”和“不该抹掉”的两部分，把 3D 从辅助信息提升成“可观测证据”的组织方式。

**逐篇反推**

1. `Adaptive Occlusion-Aware Network`
创新类型是工程组合偏新机制。能发是因为它抓住了手工分块错位、姿态估计噪声大且忽略背包等非人体细节这两个 gap，用自适应部位提取和遮挡权重讲通。story 是“不依赖重姿态分支，也能找到可见局部并抑制遮挡局部”。

2. `Adaptive Pseudo-Label Purification and Debiasing`
创新类型是新机制。它把无监督可见光红外的伪标签噪声拆成三个后果，记忆库错误、跨模态对应错误、负样本采样偏置，然后分别设计净化、最优传输匹配、去偏学习。story 是“先校准再分流，干净标签对齐，噪声标签只学邻域关系”。

3. `Adaptive transformer with Pyramid Fusion`
创新类型是新机制加新数据设定，但模块味较重。能发主要靠换衣场景下局部细节和全局语义协调不足，以及新加风雨换衣数据集。story 是“换衣不只是不看衣服，还要动态融合 Transformer 多层和频域局部线索”。

4. `Adversarial flow-based generative models`
创新类型是新机制和生成增广。gap 是可见光红外数据少、模态差异大，GAN 非可逆且红外到可见光困难。它用双 flow 共享高斯空间，同时做扩样和跨模态翻译。story 是“可逆潜空间把跨模态生成和训练样本生成统一起来”。

5. `Adversarial perturbation and defense`
创新类型是问题重定义加新机制。它把泛化 ReID 的困难点定义成“身份相关和无关特征在困难匹配样本中纠缠”，再用度量关系反转但外观尽量保持的生成器制造困难样本。story 是“先主动制造最难的度量样本，再训练模型防御和净化身份语义”。

6. `Attribute Conditional Diffusion-Augmented`
创新类型是生成增广机制。gap 是标注数据少，渲染和 GAN 的真实感、身份一致性不足。它用 ReID 身份嵌入和姿态热图条件扩散，生成同身份不同姿态样本。story 是“身份保持的任意姿态扩散增广，低数据时更有价值”。

7. `Attribute Guidance with Inherent Pseudo-Label`
创新类型是新机制，但赛道拥挤。gap 是 CLIP 这类视觉语言模型偏全局语义，遮挡下忽略细粒度属性，而人工属性标注成本高。story 是“不加人工属性，从 CLIP 内部挖细粒度属性伪标签，补全全局提示的盲点”。

8. `Base-Detail Feature Learning Framework`
创新类型是问题重定义加新机制。它最有启发：跨模态不是把差异全抹掉，模态共享的轮廓和运动是 base，颜色、纹理、热特征是 detail，二者都要保留。story 是“对齐共享基底，同时保留模态特异细节”。

9. `Beyond geometry`
创新类型是问题重定义加新机制。它直接撞我们的 3D 和 aerial-ground 资产，核心是 3D ReID 不该只用点云几何，应该把纹理放到标准 UV 坐标里，并用可解释激活图和可见衣物 mask 辅助。story 是“3D 的价值不是几何本身，而是标准表面坐标里的纹理和可解释比较”。

**强候选方向**

1. Headline：ReID 应该比较“共同可观测的身份证据”，而不是强行比较所有 2D 区域。  
挂靠资产：SMPL 3D 几何、pose 热图门控、aerial-ground、SOLIDER-Swin。  
区别：它和 AOANet 的区别是不用 2D 自适应分块讲遮挡，而是用 SMPL 表面可见性定义哪些证据可比较。它和 `Beyond geometry` 的区别是避开完整 UVTexture 和重排序，重点做低清航拍到地面里的可观测性约束，目标是训练出单 embedding 或表面 token embedding。  
cheap kill-switch：零训练先用现有 SOLIDER token 加 SMPL 可见表面 mask，在 AG-ReID.v2 或 CARGO 上只看大视角差、低可见重叠子集。如果共同可见区域相似度不能减少明显误匹配，这条先降级。

2. Headline：aerial-ground 跨视角不是“消除视角差异”，而是学习 3D base 和视角 detail 的可控分工。  
挂靠资产：SOLIDER-Swin、SMPL mesh/joints、aerial-ground。  
区别：最像 BDLF，但 BDLF 是可见光红外，用相关性约束拆 base/detail；我们用物理 3D 可见性和视角投影拆“人体几何基底”和“视角特异纹理残差”。也区别于 `Beyond geometry`，我们不是把 UV 纹理当主要表征，而是让 3D 几何决定哪些视觉细节该保留、哪些该去偏。  
cheap kill-switch：冻结 SOLIDER，训练很小的两头线性探针，一个头吃 SMPL 几何或可见性统计，一个头吃视觉残差。若 base-only 和 detail-only 在 hard subset 上没有互补性，融合也吃不掉原 SOLIDER 的错误，这条不要硬做。

3. Headline：最有价值的困难样本不是随机遮挡或风格扰动，而是 3D 相机视角导致的身份证据反转。  
挂靠资产：SMPL 3D、aerial-ground、pose 热图门控。  
区别：最像 APD，但 APD 在潜空间里做度量扰动，物理含义弱。我们把困难样本定义成 SMPL 相机、可见面、投影尺度改变后产生的“同人不像、异人很像”的物理困难样本。也不同于扩散增广论文，因为第一版可以只做 mask/token 级视角扰动，不碰高成本图像生成。  
cheap kill-switch：按 SMPL 可见性做 token dropout 或 surface dropout，小规模训练 5 到 10 个 epoch。必须证明它比随机擦除、随机 token dropout 更能改善极端视角子集，否则只是普通增广。

4. Headline：伪标签可靠性应该由“几何上是否看得到同一身份证据”决定，而不只由特征置信度决定。  
挂靠资产：SMPL 3D、pose 热图门控、aerial-ground，必要时接 SOLIDER 特征。  
区别：最像 APPD 和 AG-ReID，但它们净化伪标签主要看特征、属性或语义置信度。我们用 3D 可见性判断一个跨视角伪匹配是否有足够共同证据，尤其适合无监督 aerial-ground。  
cheap kill-switch：在有标注 AG 数据上模拟无监督聚类，比较“特征置信度”和“几何可观测性分数”对伪标签正确性的预测能力。如果几何分数的干净样本筛选纯度没有明显提升，这条不做主线。

我建议优先压第 1 和第 2。第 1 的问题定义最清楚，和团队的 pose gating、SMPL、aerial-ground 都能挂上；第 2 的论文叙事更像 Pattern Recognition/TMM，能把“不要过度对齐”讲成机制。第 3 可以作为训练机制配套，第 4 更像无监督扩展，除非 cheap kill-switch 很强，否则不要先押。
