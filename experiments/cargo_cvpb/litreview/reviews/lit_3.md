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
session id: 019ef71f-580a-7f31-8a72-7b1901fc3a2b
--------
user
你是 ReID 论文创新挖掘员 3/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(3/20):
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
ERROR: Reconnecting... 1/5
codex
我先按项目规则确认当前目标和实验上下文；如果父目录受限读不到，我会说明限制，然后只按你给的论文清单读取前三页做创新反推。
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
/bin/zsh -lc "sed -n '1,220p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

 succeeded in 0ms:
# AGENTS.md · ReID 工作区（Codex 接手版）

本文是 Codex 会自动加载的项目规则。真正来源仍然是 `CLAUDE.md` 和 `.claude/rules/`，如果本文和它们冲突，以 `CLAUDE.md` 为准。因为 Codex 默认只自动读 `AGENTS.md`，所以这里把 Claude 那套最重要的规则展开写清楚，方便 Claude 没有额度时由 Codex 接手。

`.claude/rules/` 是给人和 agent 看的 Markdown 规则。Codex 的 `.codex/rules/` 是命令权限规则，格式不同，不要把这些 Markdown 文件直接放进去。当前项目已经有 `.codex/hooks.json` 和 `.codex/hooks/`，用于在 Codex 执行命令前做训练审查和监控检查。

## 每次接手先做什么

每次接手、每个大步骤开始之前，都先读 `GOAL.md`。这是当前目标的唯一来源，由用户来写。用户可能直接改这个文件来调整方向，所以不要只看对话历史。

接着读 `CLAUDE.md`、本文件、`experiments/STATUS.md`。如果要启动实验、审查实验、改训练代码、解析结果，继续读相关的 `.claude/rules/*.md`。其中最重要的是 `result_discipline.md` 和 `experiment_protocol.md`。

如果 `GOAL.md` 的主目标被清空，或者写成暂停，就停下来等用户。不要自己找新方向。

## 你在这个项目里的角色

第一，做训练前两轮独立审查里的 Codex 这一轮。一个改了方法的实验，在开始训练之前，主 agent 会用 `codex exec` 起你来做一次独立代码审查。你看不到另一个审查者的结论，也不知道这是第几轮，也不要假设自己知道改了什么。审查范围是设计文档、新增和改动代码、配置、对照组变量隔离、评测协议和数据流。

审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。

第二，做独立讨论和探索。Claude 额度紧张时，能独立完成的子任务可以交给你，比如讨论方向、读代码、查事实、解析日志、检查实验记录、做差距分析。

## 工作区是什么

这是一个新的 ReID 研究工作区，目标是做一篇 CCF-B 级别、真正新颖、有效、又能讲清楚的行人重识别工作。具体子方向由调研和实验判断，不要一开始就把方向写死。

`SOLIDER-REID/` 是干净的上游代码，来源是 `github.com/tinyvision/SOLIDER-REID`，版本是 `8c08e1c`。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这个目录里从零写。

`experiments/` 是实验记录，从 `exp000` 开始编号。`.claude/rules/` 是详细规则。`.claude/hooks/` 和 `.codex/hooks/` 是用来强制执行纪律的检查脚本。

## 铁律

数字只认日志。所有指标都要用代码从日志文件里解析出来，不能凭记忆、凭印象写，也不要手抄。

凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。

rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。

永远不要挑随机种子，也不要挑表现最好的那个 epoch。一律上报最后一个 epoch 的结果，不要用 `best_model`。

评测口径是冻结的。要改评测口径，必须先问用户。

每涨一次点，都要换一个挑刺的角度重新核对一遍，看它是不是噪声、是不是数据泄漏、是不是评测口径前后不一致。

正式训练之前，先用很小的规模快跑一遍，确认不会崩，模块确实在起作用。

做好实验记录。`experiments/decisions.md`、`experiments/results.md`、每个实验的 `monitor.md` 都要及时更新。同样的配置加同样的种子，不要重复跑。

## 三条研究纪律

判定一个方向走死之前，要先定好标准，并且有足够证据。开始跑之前就把“什么样的结果才算这条路走死了”写进 `design.md`。只有一两个负结果时，只能写“还需要再试”，不能判定整条方向死掉。

自己写的评测脚本或分析脚本，要先用它复现一个已知的基线成绩。对得上之后，才能用它的结果下结论。新的评测口径、新的度量、新的脚本都按这条执行。

“贡献”是个有门槛的词。一个结果只有同时满足下面几条，才能叫贡献，才能说可以投稿：满足对新方向的要求；跑了三个种子并报告均值和标准差；涨幅超过正常波动范围；和最接近的已有工作区分清楚。在那之前，只能叫“信号”或“探索”。

## 一个新方向值不值得做

ReID 是活跃领域，不要一上来就觉得能做的都被做完了。先读论文、做差距分析，再决定方向。

一个新方向至少要满足下面三条里的两条，否则不作为主线。

1. 问题上有新意。不是加一个模块，而是重新定义或者更准确地刻画一个真实存在的问题。
2. 机制上有新意。是过去工作没有清楚写出来，而且代码上能实现的机制。
3. 证据上讲得清。能设计出干净的对照和消融，能回答它为什么有效。

方向必须和最接近的已有工作区分清楚。方向定下来之前，要和 Codex 或子代理讨论核实，确认它确实是新的。不能拿测试时的小技巧当主要贡献，比如重排序、特征归一化、翻转测试。不能用“比基线高了零点几”来定义创新。

## 实验命名和目录

实验目录叫 `exp{编号}_{简短描述}`，例如 `exp000_baseline`、`exp012_new_method`。

同一个实验的不同变体用字母后缀区分，例如 `exp012a`、`exp012b`、`exp012c`。判断标准是：核心方法相同，只是配置、种子、环境或超参不同，就属于同一个实验的变体。所有变体共用一个 `experiments/exp{编号}/` 目录。

每个实验用独立的 `OUTPUT_DIR`，例如 `./log/<数据集>/exp{编号}`。

训练命令一般使用这个形式：

```bash
python train.py --config_file xxx.yml \
  SOLVER.SEED NN \
  TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR ./log/.../expNNN \
  <其它覆盖项>
```

`TEST.IMS_PER_BATCH 64` 建议都加上，因为测试集大、又开了翻转测试时，默认 256 容易把显存撑爆。后台跑用 `setsid nohup python train.py ... </dev/null > /path/uniq.log 2>&1 &`。日志文件名必须唯一，不要互相覆盖。

## design.md 格式

开始训练前必须写 `experiments/exp{编号}/design.md`。格式如下：

```markdown
# 实验 exp{编号}: {名称}

## 动机
为什么做？解决什么问题？基于前面哪些实验或者论文？

## 核心假设
一句话说清楚。

## 技术方案
改了哪些文件？加了哪些模块？数据从输入到输出怎么走的？关键超参怎么定的？

## 对照组
和哪个基线比？只改了哪一个变量？

## 什么算走死
什么样的结果算“还需要再试”，什么样的结果算“这条路走死了”。

## 预期结果和失败解释
假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。

## 需要训练前审查
需要训练前审查：是
```

改了方法的实验填“需要训练前审查：是”。纯复现实验，也就是只改随机种子的，把这行改成“需要训练前审查：否”，检查脚本会放行。

## 训练前独立审查

任何改了模型或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查。一轮由 Claude 做，一轮由 Codex 做。两个审查者互相看不到对方结论，也不知道这是第几轮。

Codex 这一轮要完整审一遍，不是只看某几处。要看设计文档、新增和改动的代码、配置、对照组、变量隔离和评测协议。结论写进 `experiments/exp{编号}/review-codex.md`。

实质问题修好之后，必须再审一轮。某一轮里两个审查都没有实质问题，才算放行。

检查脚本 `.codex/hooks/check_design.sh` 会在包含 `train.py` 的命令执行前检查：设计文档在不在，`review-claude.md` 是不是通过且至少三十行，`review-codex.md` 是不是通过。如果 `design.md` 里写了“需要训练前审查：否”，就只检查设计文档。

## 代码原则

新模块要插件式实现，放在 `model/` 下，用配置开关控制。默认配置必须能复现基线。

每次只改一个核心变量。可以是一个模块、一个损失、一种 pooling、一个训练机制。如果要组合几个东西，必须写清楚组合了哪些已经验证过的模块，以及为什么现在适合组合。

配置、随机种子、commit 号或代码状态都要记下来。这个工作区当前顶层不一定是 Git 仓库，所以如果没有 commit 号，就记录文件状态、命令和关键改动。

不要用有破坏性的 git 命令，不要覆盖用户已经做的改动。

## 文档纪律

没有文档的实验，等于没做过。

每个实验从头到尾要维护这些文件：`experiments/exp{编号}/design.md`、`monitor.md`、需要时的 `review-claude.md` 和 `review-codex.md`。总记录放在 `experiments/results.md`、`experiments/decisions.md`、`experiments/STATUS.md`。

每次看日志，都要更新 `monitor.md`。至少写当前到第几个 epoch、进度如何、关键损失值、评测指标，以及一句判断：继续、盯着，还是停掉，并说明原因。

同一个实验在不同文档里的数字必须完全一致。新结果推翻旧判断时，直接改文档里的措辞，不要只在对话里说。

实际实现和 `design.md` 不一致的地方，在 `monitor.md` 里写清楚。一个实验结束后，先把文档补完整，再开下一个。

论文素材要一直维护在 `experiments/paper_materials/`。`story.md` 记录当前方法主线、候选贡献、支撑实验和推翻旧说法的结果。表格和图放在 `tables/` 和 `figures/`。

## 决策记录

重大决策追加到 `experiments/decisions.md`，格式如下：

```markdown
### [年-月-日 时:分] 决策 #编号
上下文：在什么情况下做的这个决策
选项：A. 方案和预期；B. 方案和预期
选择：A 还是 B
理由：为什么
执行结果：（后面补）
```

重大决策前，尽量让两个独立视角辩一辩。一个为方案 A 辩护，一个为方案 B 辩护，从技术可行性、创新性、论文价值、风险、成本几方面讲，并给出信心分。最后综合判断，把结论写进决策记录。

自己试过并判定为负的方向，要记进 `experiments/decisions.md`，免得以后重复跑。要重新走一个已经判负的方向，先写清楚为什么这次不一样。判负必须有足够证据，不能用一两个负结果判整条方向。

## 监控和等待

不要用长 `sleep` 反复读日志。优先用 Monitor 或后台 Bash 等待器。启动长任务后，用后台方式运行，并让完成事件或完成标志来唤醒后续检查。

如果必须执行 180 秒以上的 `sleep`，先更新某个 `experiments/exp*/monitor.md`。`.codex/hooks/check_monitor.sh` 会检查这一点。

第 1 到 5 个 epoch 要勤看，确认不会崩，模块在起作用。第 6 到 30 个 epoch 中等频率。30 个 epoch 以后可以看得稀一点。每次看完都更新 `monitor.md`。

出现 NaN 或 Inf，立刻停，先查原因，再决定是降学习率还是回退。显存溢出时，先减小模块复杂度，或者把 `TEST.IMS_PER_BATCH` 调小，不要随便改训练 batch size。长时间没进展时，先把证据写下来，再决定要不要停。

DataLoader 的子进程也会显示成 `python train.py`，不要把它误当成重复训练进程。停训练时只杀主进程。

## 机器和网络

你在 Mac 上跑，能联网。服务器只有国内网，装包用清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`，下模型用 `https://hf-mirror.com`。统一用 `~/.ssh/config` 里的别名连接，不要用 `sshpass`。

三台 GPU：

1. `hyy-5060ti-double`：恒源云，两块 5060Ti 16G，用户 root，直连。`/hy-tmp` 只有 50G。训练命令里用 `--gpu 0` 或 `--gpu 1` 选卡。每次训练只保留最后一个 epoch 的 checkpoint，中间的和 `best_model` 都删掉。
2. `lab-3090-d`：实验室 RTX3090 24G，单卡，在 docker 容器 `abdslm-common` 里，经 `lab-3090` 跳板连。容器一重启就会丢掉 sshd 和 IP，要在主机上执行 `docker exec abdslm-common /usr/sbin/sshd` 重新启动 sshd。IP 变了就更新 `~/.ssh/config`。
3. `lab-4090`：实验室 RTX4090D 24G，单卡，共享机器。只能用 `afr` 自己的空间，数据放 `/mnt1/afrdata`，代码、日志、虚拟环境放 `/home/afr/` 下的项目目录。绝对不要碰 `/root`、`/hy-tmp` 和共享 conda。

传大文件用 `rsync -az --partial`，放后台跑，再监控日志。本地到远程的带宽大概 2 MB/s。

如果一条远程命令反复失败，但单独拿出来测又正常，先怀疑日志没刷新，或者读到的文件不是最新的，不要急着改代码。

绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。

## SOLIDER-REID 代码底子

上游只带了 Market-1501 和 MSMT17 的配置，主干有 Swin Tiny、Small、Base。要用别的 benchmark，比如 Occluded-Duke、Occluded-ReID，需要自己加配置和数据集读取代码。

用 SOLIDER 的预训练权重。从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint。Mac 能连 Google Drive。

`SEMANTIC_WEIGHT` 默认是 0.2。

主干顺序是先用 Swin-Tiny 快速验证。有信号了再放大到 Small、Base。创新点先在 Tiny 上看有没有效果，确实有效再放大，不要在没效果的点上用大主干硬磨。

## Python 和环境

如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。

服务器上也不要动共享 conda。`lab-4090` 尤其要注意，虚拟环境放在 `/home/afr/` 下自己的项目目录。

## 写中文

所有给人看的文字，包括文档、注释、提交信息和给用户的回复，都用平实、完整的中文句子。

不要把判断标准、检查步骤生造成奇怪的简称。不要用箭头把几个词串起来代替句子。不要用破折号来连接或停顿。不要在中文里夹不必要的英文术语。真正的专有名词、命令、文件名、配置字段可以保留原文。


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

codex
目标文件和项目状态已经读到，当前任务明确是文献差距分析，不开实验。下面我按你的要求逐篇只抽前三页，重点看摘要、引言里的问题设定、机制卖点和证据链。
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
exec
/bin/zsh -lc "pdftotext -l 3 'Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 161 (2025) 111301

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Bidirectional modality information interaction for Visible–Infrared Person
Re-identification
Xi Yang a

,∗, Huanling Liu b , Nannan Wang a , Xinbo Gao c,d

a

Xidian University, Xi’an 710071, China
Hangzhou Institute of Technology, Xidian University, Hangzhou 311231, China
c
State Key Laboratory of Integrated Services Networks, School of Electronic Engineering, Xidian University, Xi’an 710071, China
d
The Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing 400065, China
b

ARTICLE

INFO

ABSTRACT

Keywords:
Dynamic aggregation
Feature intersection
Multi-weight loss

Due to the distinct imaging principles of visible and infrared images, significant challenges arise for visible–
infrared person re-identification (VI-ReID). Current methods either establish intermediate modalities to reduce
image differences or use various techniques to extract better shared features. However, despite the modality
differences in VI-ReID, significant similarities in contours and postures exist, so focusing on learning feature
representation can fundamentally solve the problem. To this end, this paper proposes a bidirectional modality
information interaction network to reduce the difficulty of inter-modality recognition and enhance intramodality feature extraction. To fully exploit cross-modality information, the Dynamic Aggregation (DA) module
iteratively projects features of both modalities onto each other, embedding image attributes from both, enabling
mutual learning, improving feature representation, and reducing modality differences. To enhance intramodality feature representation, the Feature Intersection (FI) module combines two pooling styles in a novel
way, ensuring global connection of local features and regional highlights. Lastly, we propose a Multi-Weight
(MW) loss to increase the center distance of same identity features within the same modality, helping the model
better learn identity attributes. Extensive experiments on SYSU-MM01 and RegDB confirm the superiority of
our approach.

1. Introduction

visible images to generate grayscale images as a bridge between IR
and visible images [8–13]; (3) decoupling features into shared and
specific features to enhance the former or to transfer the two complementarily [14–18]. However, the infrared and visible modalities are
independent and do not communicate with each other, whether the
intermediate modalities are set outside the modalities for easier metrics
or the shared-specific features are extracted inside the modalities. Previous approaches have not considered reducing modality discrepancy
by directly enhancing the communication between the two modalities,
which undoubtedly increases the difficulty of modality recognition.
Besides the modality discrepancy issue, how to better extract rich and
comprehensive pedestrian attribute information within the modality
is also a problem that cannot be ignored. As we all know, global
average pooling (GAP) [10,19,20] focuses on the global area while
global max pooling (GMP) [21] focuses on the local significant area.
Furthermore, the fusion operation (GAP+ GMP) [22,23] is adopted to
fuse the features from the last network layer, simultaneously focusing
on both global and local information [24]. However, a simple fusion

Person Re-identification (ReID) aims to match target individuals
across images captured by different cameras and at different times,
and recent systems in this field have achieved significant success [1–3].
As person re-identification develops, some limitations are revealed. For
example, most crimes happen at night, but traditional Re-ID does not
match visible (RGB) images well with infrared (IR) images. Therefore,
given an IR image of a pedestrian, searching for the pedestrian target
in visible light is necessary. However, RGB images and IR images are
fundamentally different, visible light and infrared light have different wavelength ranges. Thus, Visible–Infrared Person Re-Identification
(VI-ReID) was born.
The most obvious concern for VI-RelD is the large modality discrepancy between visible images and the infrared images. Existing
approaches bridging the modality differences fall into the following categories: (1) utilizing GAN networks to generate intermediate modalities
to reduce modality variance by bringing visible (RGB) and infrared (IR)
modality closer to the intermediate modality [4–7]; (2) transforming

∗ Corresponding author.

E-mail address: yangx@xidian.edu.cn (X. Yang).
https://doi.org/10.1016/j.patcog.2024.111301
Received 29 July 2024; Received in revised form 15 November 2024; Accepted 16 December 2024
Available online 24 December 2024
0031-3203/© 2024 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 161 (2025) 111301

X. Yang et al.

time, to comprehensively extract pedestrian information within the
modality, the Feature Intersection (FI) module is introduced. It adopts a
simple and efficient structure that combines the two pooling methods,
maximizing the potential of both while avoiding their drawbacks.
Specifically, we set up two branches, one branch performs GAP to
enhance the global connections and then utilizes local partitioning to
obtain local features with contextual information. The other branch
conducts local partitioning first and then carries out GMP for each
local feature to highlight the local focus. Finally, the local features from
the two branches are fused in the channel dimension. The advantages
of both pooling approaches are fully exploited through feature intersection within the modalities. Finally, to address both cross-modality
variances within classes and inter-class similarity, we propose a novel
multi-weight loss. It is worth noting that different pedestrians have
different similarities, and pedestrians with higher similarities cannot be
strongly supervised with differentiation. And to alleviate the nuisance
caused by different pedestrians with different similarities, we first
rank the distances after calculating the obtained inter-class distances.
Different weights are assigned from the most similar pedestrians to the
most distant pedestrians to relieve the concern that the most similar
pedestrians cannot be adequately trained.
In summary, our contributions are as follows:

Fig. 1. Illustration of the proposed method. The Dynamic Aggregation focuses on
bidirectional interaction of information between modalities, reducing modality differences. The Feature Intersection module focuses on intra-modality information exchange,
enhancing feature representation abilities. (For interpretation of the references to color
in this figure legend, the reader is referred to the web version of this article.)

• To make two modalities interact with each other and absorb helpful information to reduce the recognition difficulty, we propose a
dynamic aggregation module to fully exchange the information
between two modalities utilizing dynamic projection.
• A simple but efficient feature intersection module is designed to
introduce local features and take full advantage of GAP and GMP
to enrich the information of intra-modality feature extraction.
• To alleviate the trouble that pedestrian samples with high similarity cannot be adequately trained, we propose multi-weight loss
to perform strong/weak supervision to make a good balance of
inter-class distance.

operation does not fully exploit the advantages of both, but rather
weakens the focus of GMP on the salient areas. In addition to these,
triplet loss [25] is employed as a common loss function for VI-ReID,
which is only easy to increase the differential pedestrians between
modalities, and it is difficult to improve the similarity of cross-modality
features within classes. Further, HC loss [26] is adopted to reduce the
inter-modality identity center distance to mitigate the heterogeneity.
However, HC loss only focuses on the variability of the same pedestrian
across modalities, while ignoring the similarity of different pedestrians
in the same modality. Therefore, we propose a bidirectional modality
information interaction network. As shown in Fig. 1, the blue and red
circles respectively represent features of visible and infrared pedestrians. The Dynamic Aggregation (DA) module allows for mutual learning
of information between different modalities, aiming to compensate for
the limitations in feature expression caused by modality differences
and thereby reducing the difficulty of inter-modality recognition. In
contrast, the Feature Intersection (FI) module focuses on enhancing the
representation of features within the same modality, aiming to improve
the expression capability of the fused modality features.
The Dynamic Aggregation (DA) module that directly interacts with
and learns from features of the two modalities to generate two new
modality features. Firstly, the module defines the two modalities as
the primary and auxiliary modalities. Before interaction, multi-head
attention mechanisms enhance the important regions of the modality features. Next, to fully utilize the potential impact of the auxiliary modality, we convert the features of the auxiliary modality from
one-dimensional to higher-dimensional matrices. Valuable information
from the auxiliary modality is then weighted and transferred to the
primary modality. This weighted mapping adaptively and dynamically
adjusts the features of the primary modality, enhancing its responsiveness to the auxiliary modality. This process is bidirectional and involves
multiple iterations, with weights directly derived from the auxiliary
modality features and repeatedly executed. As a result, the entire
architecture engages in dynamic information exchange, completing
the information distribution process. Dynamic Aggregation preserves
the original primary modality information while providing useful information from the auxiliary modality, instead of merging the two
modality features into a single new modality feature. This adaptive
approach improves the modality representation capacity. At the same

2. Related work
2.1. Person re-identification
Person re-identification [27] is a technique that involves preprocessing person images captured across different video surveillance cameras
and locations, annotating the target individual, and analyzing the
person features in multi-camera images to retrieve the corresponding person. With the advancement of deep neural networks, existing
methods [3,28–31] have achieved inspiring performance in singlemodality person ReID. However, most real-world criminal incidents
occur at night. In order to achieve comprehensive real-time surveillance, it is necessary for visible and infrared video surveillance cameras to work together. Due to factors such as imaging mechanisms,
shooting angles, and environmental conditions, person re-identification
faces challenges in dealing with significant color variations and pose
changes between person images, making it difficult to perform identity
recognition in real-world scenarios. Therefore, Visible–Infrared Person
Re-identification (VI-ReID) [32] technology combines visible and infrared video surveillance cameras to retrieve target persons, thereby
overcoming this challenge.
2.2. Visible–infrared person re-identification
Visible–Infrared Person Re-Identification (VI-ReID) retrieves pedestrians from different types of cameras (visible and infrared). Due to
the differences in imaging principles, there is a significant modality
gap, making it difficult to achieve satisfactory matching results between
visible and infrared images. At the image level, reducing modality
differences is often achieved using auxiliary images. For example,
D2 RL [33] and AlignGAN [5] convert visible images to infrared images
2

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Fig. 2. A general overview of the proposed framework. It contains two branches of visible and infrared modalities, which are represented by different colors. Among them, the
dynamic interactions are carried out multiple times recursively, and ‘‘M’’ denotes the number of recursions. The ‘‘1-d to 2-d’’ indicates the feature dimension transformation. (For
interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

and vice versa. MMN [34] adopted lightweight networks to generate
auxiliary modalities for cross-modal searches. PMT [35] used grayscale
images as auxiliary modalities and introduces a progressive learning
strategy. GC-IFS [36] generated image pairs using a generative network, creating high-quality cross-modal paired images and blending
information from both modalities. However, generating cross-modality
images inevitably comes with noise. At the feature space level [32,37–
40], Ye et al. [20] designed a dual-stream CNN network to extract
deep features and integrated identity loss and contrastive loss to learn
multi-modality shared features. To extract shared features and transfer knowledge of different types of information, [41] introduced an
intermediate shared dual-stream network and a collaborative ensemble
learning mechanism. Zhu et al. [26] proposed the hetero-center loss,
which introduced the feature center of the same identity to VI-ReID
for the first time. This effectively reduced the distance between feature
centers of the same identity. DCLNet [42] encouraged positive pixels
with the same semantic information to be closer together while pushing negative pixels apart. MRCN [16] adopted a modality restoration
module and a modality compensation module to disentangle modalityindependent and modality-dependent features. Although these methods
have made significant progress, they overlook the interaction between
modality features and fail to fully leverage the advantages of both
modalities.

the pedestrian, its representation is invariant to modality changes.
Information about body structure is modality invariant, which can be
used to learn modality-shared feature representations. Therefore, we
propose a two-stream network, which consists of feature extractor and
feature embedding.
Feature extractor captures information from heterogeneous images
to form final feature descriptors. Since the input of the network includes
visible images and infrared images, we use two independant branches
to extract information from these two modalities. Given the limited
data, each branch contains a pre-trained model that inherits the architecture of ResNet-50 [43] prior to the global average pooling layer, but
with a slight variation. The difference is that we have removed the last
downsampling operation in ResNet-50, which can expand the receiving
field area and enrich the granularity of features.
The feature map from the trunk is then evenly divided horizontally
into 𝑝 stripes. Each strip is averaged into a local feature vector. And
we utilize the fully connected (FC) layer to reduce the dimension of
each local feature vector. To extract exclusive information from these
two modalities, the corresponding fully connected layer in the two
branches do not share the same weight. For each FC layer, we adopt a
Leaky ReLU activation layer and a batch normalization layer to solve
the internal covariance shift problem. In the test phase, the images are
input into the corresponding branch according to the modalities. Then,
L2 normalization is performed for each local feature vector. Finally, all
the feature vectors are connected to form the final feature descriptor.
During the test phase, given a probe image, we extract the feature
descriptor of the probe and all the heterogeneous gallery images. Next,
we sort the gallery images according to the euclidean distance between
the probe and the feature descriptors of all the gallery images. Under
ideal conditions, heterogeneous intra-class gallery images have the
highest similarity.
The purpose of feature embedding is to supervise the learning of
feature representation by the network, which achieves two goals of
enlarging the inter-class difference as much as possible and improving
the intra-class similarity from the two modalities. Therefore, we employ
multi-weight loss and classification loss to supervise network training.
For multi-weight loss, the features are normalized by L2 and the loss
value is calculated. For the classification loss, a local feature vector is
input into a classifier consisting of a FC layer and a softmax activation
layer. There are 𝑝 local feature vectors that need to be input into

3. Our approach
The proposed modality information fusion network consists of three
main parts, dynamic aggregation, feature intersection, and multiweight loss. Here, we employ an independent dual-stream structure
(Fig. 2) as the feature extraction network, where each modality has
its own areas of interest due to the variation in imaging principles.
We utilize two independent (no parameter sharing) ResNet-50 [43]
networks to preserve the respective modality information during the
feature extraction phase.
3.1. Two-stream network
The typical approach in VI-ReID is to divide pedestrians into horizontal stripes, extract local features, and connect them to represent
body structure. Since the body structure is an inherent property of
3


 succeeded in 0ms:
Pattern Recognition 161 (2025) 111217

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Camera-aware graph multi-domain adaptive learning for unsupervised
person re-identification
Zhidan Ran, Xiaobo Lu ∗, Xuan Wei, Wei Liu
School of Automation, Southeast University, Nanjing 210096, China
Key Laboratory of Measurement and Control of Complex Systems of Engineering, Ministry of Education, Nanjing 210096, China

ARTICLE

INFO

Keywords:
Unsupervised person re-identification
Heterogeneous graph learning
Adversarial learning

ABSTRACT
Recently, unsupervised person re-identification (Re-ID) has gained much attention due to its important practical
significance in real-world application scenarios without pairwise labeled data. A key challenge for unsupervised
person Re-ID is learning discriminative and robust feature representations under cross-camera scene variation.
Contrastive learning approaches treat unsupervised representation learning as a dictionary look-up task.
However, existing methods ignore both intra- and inter-camera semantic associations during training. In this
paper, we propose a novel unsupervised person Re-ID framework, Camera-Aware Graph Multi-Domain Adaptive
Learning (CGMAL), which can conduct multi-domain feature transfer with semantic propagation for learning
discriminative domain-invariant representations. Specifically, we treat each camera as a distinct domain and
extract image samples from every camera domain to form a mini-batch. A heterogeneous graph is constructed
for representing the relationships between all instances in a mini-batch. Then a Graph Convolutional Network
(GCN) is employed to fuse the image samples into a unified space and implement promising semantic transfer
for providing ideal feature representations. Subsequently, we construct the memory-based non-parametric
contrastive loss to train the model. In particular, we design an adversarial training scheme for transferring the
knowledge learned by GCN to the feature extractor. Experimental experiments on three benchmarks validate
that our proposed approach is superior to the state-of-the-art unsupervised methods.

1. Introduction
Person re-identification (Re-ID) aims at retrieving the identical
pedestrian across multiple non-overlapping cameras with a given query
image [1]. It attracts great attention in the computer vision community
and plays a pivotal role in an increasing number of applications in
video surveillance, such as finding people of interest (e.g., criminals or
lost kids), multi-camera tracking, and cross-scene crowd counting. In
recent years, the performance of fully-supervised person Re-ID has obtained significant improvements, principally because of the widespread
adoption of deep convolutional neural networks (CNNs) [2]. However, supervised models largely depend on substantial pairwise labeled
training data, which limits the feasibility and scalability for real-world
applications, since manually annotating cross-camera identity labels is
a time-consuming and cumbersome task. In this context, we focus on
developing an unsupervised scheme for pedestrian Re-ID.
Traditional unsupervised approaches attempt to learn invariant
components, i.e., saliency, dictionary, or metric, resulting in limited
scalability and Re-ID matching performance [1]. For deeply unsupervised person Re-ID, most of the existing methods are based on

unsupervised domain adaptation (UDA) [3], which leverages the prior
knowledge learned from other identity-labeled source domains. Unfortunately, these UDA methods require a large amount of annotated data,
and the performance is closely related to the domain gap. In this case,
recent research has focused on solving the person Re-ID problem using
fully unsupervised approaches with more flexibility [4]. However, the
performance is still far from satisfactory. One of the main challenges is
that, without the supervision of pairwise labels, it is a rather arduous
assignment to learn the identity-discriminative information due to the
high inter-class similarity under the same camera and the significant
intra-class variations across different cameras.
To illustrate this further, we show an illustrative example of the
cross-camera scene variations in Fig. 1. Typically, person Re-ID is considered a retrieval ranking problem. The similarity between the interidentity feature pair should ideally be lower than the intra-identity
feature pair. Nevertheless, the variations of pedestrian images captured
from different cameras (e.g., illumination, background clutter, resolution) lead to domain shift in the feature space and result in inconsistent
pairwise similarity distributions that decrease the Re-ID performance.

∗ Corresponding author at: School of Automation, Southeast University, Nanjing 210096, China.

E-mail addresses: zhidan_r@163.com (Z. Ran), xblu2013@126.com (X. Lu), wx1204@seu.edu.cn (X. Wei), liu_wei@seu.edu.cn (W. Liu).
https://doi.org/10.1016/j.patcog.2024.111217
Received 1 May 2024; Received in revised form 16 August 2024; Accepted 20 November 2024
Available online 28 November 2024
0031-3203/© 2024 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

2. Related work
2.1. Unsupervised person Re-ID
With the rapid development of deep learning, person Re-ID has
achieved impressive progress. Most previous Re-ID approaches are in a
supervised manner, where the cumbersome annotation process hinders
the scalability. In order to reduce the reliance on a large amount of
manually annotated supervised data, unsupervised models have been
widely studied for the person Re-ID task, and the related approaches
can be classified into two categories, unsupervised domain adaptation (UDA) [6] and fully unsupervised Re-ID [4]. The former learns
discriminative feature representations on the source labeled dataset
and fine-tunes the pre-trained model on the target unlabeled dataset,
while the latter attempts to learn rich feature representations from
the completely unlabeled data. Among UDA-based methods, In [7], a
Dissimilarity-based Maximum Mean Discrepancy (D-MMD) loss is proposed for aligning the source and target data in the dissimilarity space
via exploiting the structure of intra- and inter-class distributions. All
these UDA-based methods require an external source domain and rely
on the similarity between the source domain and the target domain.
Fully unsupervised methods learn directly from unlabeled images
and thus have better flexibility and scalability in real-world deployments. A memory-based multi-label classification loss is introduced
in [8] for boosting the Re-ID model training efficiency in multilabel classification. Lin et al. [4] propose a framework based on
softened similarity learning, which replaces clustering-based pseudo
labels with similarity-based softened labels. Unsupervised asymmetric
metric learning [9] aims to mitigate view-specific bias by learning
the specific projection for each view, resulting in better matching
performance. Li et al. [10] employ a pure transformer network for
extracting multi-grained feature representations. Relation-Preserving
Feature Embedding (RPE) [11] is presented to improve the performance
of the unsupervised person Re-ID by learning structure relations among
samples. In order to improve the accuracy of pseudo-label computation
across cameras, Xuan et al. [12] introduce a learning framework to
decompose the similarity computation into two phases.
In recent years, some unsupervised visual representation methods
based on contrastive learning have been applied to person Re-ID and
achieved impressive performance. A self-paced contrastive learning
method with hybrid memory is presented in [13] for gradually creating more reliable clusters to refine the memory and learning targets.
Zhao et al. [14] present a memory-based identification loss that is
non-parametric and coordinates with meta-learning. In [15], ICE is
proposed to enhance the consistency between augmented and original
instances by utilizing pairwise similarity scores as soft pseudo labels.
However, the variations of person images captured by different cameras (e.g., illumination, background clutter, resolution) make learning
discriminative features for Re-ID still a challenging task. Several works
leverage an unsupervised style transfer model based on StarGAN to generate style-transferred training images with different camera styles [5].
Different from these generative approaches, our work introduces a
unified fully unsupervised person Re-ID framework that enhances the
robustness of features against the cross-camera scene variation from the
perspective of semantic propagation.

Fig. 1. Illustration of cross-camera scene variations in person Re-ID. Two images in
the same row represent the same person. In the feature space, different shapes indicate
different identities and different colors represent different camera IDs. The distance
between the intra-identity feature pair should be smaller than the inter-identity feature
pair (For interpretation of the references to color in this figure legend, the reader is
referred to the web version of this article.)

In order to mitigate the influence caused by camera style disparities,
many previous approaches focus on enhancing robustness to input
changes in a data-augmented manner [5]. The process of generating
images with different camera styles, however, is relatively isolated from
the discriminative Re-ID learning and may introduce additional errors.
In contrast to the recent direction of adopting generative models
to enhance input diversity, we mainly focus on the crucial role of
information transfer in addressing the semantic alignment problem
across multiple camera-level sub-domains. In this paper, we propose
a novel unsupervised person Re-ID framework, termed Camera-Aware
Graph Multi-Domain Adaptive Learning (CGMAL), which can conduct
multi-domain feature transfer with semantic propagation for learning
discriminative domain-invariant representations. In particular, we train
a shared feature extractor network for mapping all input images of
unlabeled training data to feature representations. We treat unsupervised representation learning as a dictionary look-up issue and leverage
clustering algorithm to generate pseudo-labels. Notably, a heterogeneous graph is constructed for representing the relationships between
instances from multiple camera domains. Then, we design a graph convolutional network (GCN) to fuse the features into a unified space and
learn more crucial information between multiple camera domains with
semantic propagation. In this way, the GCN can implement promising
semantic transfer and generate ideal domain-invariant representations
for assisting the training. Finally, we design an adversarial training
scheme for transferring the knowledge learned by GCN to the feature extractor. Experiments demonstrate that our method can reach
state-of-the-art results on three large-scale Re-ID datasets.
To summarize, the main contributions of this work are fourfold:
• We propose a Camera-Aware Graph Multi-Domain Adaptive Learning (CGMAL) framework for unsupervised person re-identification,
in which can conduct multi-domain feature transfer with semantic propagation for learning discriminative domain-invariant
representations.
• A novel camera-aware graph is proposed for representing the
relationships between instances from multiple camera-level subdomains.
• An adversarial training scheme is designed for transferring the
knowledge learned by GCN to the feature extractor.
• Extensive experiments validate that the proposed method is superior to the state-of-the-art unsupervised methods on Market1501, PersonX, and MSMT17 datasets. Notably, our method also
achieves a higher performance than the one-shot supervisionbased and UDA methods relying on additional annotations.

2.2. Graph learning for Re-ID
In a growing number of real-world applications, data are generated from non-Euclidean spaces and are usually represented as graphs
with complex relationships and interdependency among objects. In
the past decades, researchers have been working on how to extend
convolutional operations to non-Euclidean data and determine the
appropriate receptive field for each vertex to enhance graph representation learning. Liu et al. [16] introduce a graph learning approach for
2

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

extracting both temporal and structural information. In [17], a temporal graph clustering framework is proposed for suiting the temporal
graph patterns.
Benefiting from the feature aggregation ability of GCN, graphbased solutions achieve promising performance in Re-ID tasks. Zhang
et al. [18] propose a part-guided graph convolution network to learn
the inter-local relationship of the corresponding parts among person
images and the intra-local relationship between neighboring parts in
one person image. In [19], a heterogeneous convolutional network
is proposed for learning the appearance information of person images and the correlations among person images simultaneously. Bai
et al. [20] propose a novel UDA scheme, Hierarchical ConnectivityCentered (HCC) clustering, which utilizes GCNs to exploit vertex-level
and cluster-level connectivity. In our works, graph convolutional networks are applied to purely unsupervised person Re-ID. We design a
graph convolutional network to fuse the features into a unified space
and learn more crucial information between multiple camera domains
with semantic propagation.

updated in repeated iterations. In stage two, we extract image samples
from every camera domain to compose a mini-batch and feed them into
the shared feature extractor network. In this manner, we can obtain
the up-to-data features. Subsequently, we construct a multi-domain
graph to model the relationships between all samples in the minibatch. The GCN is applied to fuse the image samples into a unified
space and learn key information between multiple camera domains
with semantic propagation. It is to be noted here that image features
with the same identity between cameras and different identities within
the same camera contain critical semantic information that can be
transferred between multiple camera domains via the multi-domain
graph and GCN. Finally, we construct two contrastive losses (𝑐 𝑙𝑠 and
𝑔𝑐 𝑛 ) based on the memory dictionary to supervise the training of
the feature extractor network and GCN. Particularly, we propose an
training scheme with a camera adversarial loss 𝑎𝑑 𝑣 for transferring
the knowledge learned by GCN to the feature extractor. Consequently,
the combined loss for feature extractor is:

3. Proposed method

𝑒𝑥 = 𝑐 𝑙𝑠 + 𝜆𝑎𝑑 𝑣 ,

3.1. Problem definition

where 𝜆 is the parameter that regulates the contribution of the two loss
terms.
Overall, our proposed framework is capable of (1) treating unsupervised representation learning as a dictionary look-up issue and (2)
learning discriminative and domain-invariant representations by implementing semantic information propagation between multiple camera
domains. Our method is summarized in Algorithm 1. We next describe
our method in detail from these two aspects.

In the unsupervised person Re-ID scenario, we are provided with an
unlabeled dataset  𝑡𝑟 = {1 , 2 , … , 𝑉 } captured from a surveillance
camera network consisting of 𝑉 cameras, where the subset 𝑗 =
{𝑥𝑗1 , 𝑥𝑗2 , … , 𝑥𝑗𝑁 } of 𝑁𝑗 unlabeled images, and thus there are 𝑁 = 𝑁1 +
𝑗
⋯ + 𝑁𝑉 images in total as training samples. Treating each camera
as a sub-domain, we assume that the sub-domains have the same
label space, and the underlying data distributions between camera subdomains are different due to the cross-camera scene variations. Our
goal is to learn a shared feature extractor network 𝜙 for all the camera
domains via using the data in {𝑗 }𝑉𝑗=1 . Person Re-ID is considered as
a ranking task in this work. Ideally, images with the same identity as
the probe person get a higher rank than other images with different
identities in the gallery. Given a query person image 𝑥𝑞 , the feature
extractor network is expected to extract feature vectors from a gallery
set  𝑔 for retrieving the person image 𝑥∗ with the same identity as 𝑥𝑞 ,
i.e.,
𝑥∗ = ar g min𝑔 𝑑(𝜙(𝑥𝑞 ), 𝜙(𝑥𝑖 )),
𝑥𝑖 ∈

(2)

3.3. Contrastive learning
This paper adopts contrastive learning for unsupervised representation learning in the person re-identification task.
(1) Memory Initialization: We design a memory-based feature dictionary  to store cluster representations {𝑚1 , 𝑚2 , … , 𝑚𝑁 𝑚 }, where 𝑁 𝑚
denotes the number of clusters. It is worth noting that 𝑁 𝑚 changes
dynamically as the iterative optimization of the model. The dictionary
is initialized with the average representation of all instances belonging
to each cluster, i.e.,
∑
1
𝑚𝑘 =
𝑓,
(3)
∣ 𝑘 ∣ 𝑓 ∈ 𝑖

(1)

where 𝜙(𝑥𝑖 ) is the 𝑑-dimensional feature vector extracted by the feature
extractor network 𝜙, 𝑑(⋅) is the Euclidean distance in feature space,
i.e., 𝑑(𝜙(𝑥𝑞 ), 𝜙(𝑥𝑖 )) = ‖𝜙(𝑥𝑞 ) − 𝜙(𝑥𝑖 )‖2 .

𝑖

𝑘

where 𝑘 represents the set consisting of all features in the 𝑘th cluster
and ∣ ⋅ ∣ denotes the number of features belonging to the set.

3.2. Overview of the method

(2) Memory Updating: After each training iteration, the feature
vectors in each mini-batch are used for updating the representations
in the memory dictionary. Formally, we update the 𝑘th representation
vector by:

Unsupervised re-identification is a very challenging and arduous
task since it suffers from the impact of camera shift and the lack of
manual annotation. The appearance of persons across diverse cameras
is affected by the changes of illumination, perspective, and other environmental factors, resulting in large differences between within-class
features. To address these challenges, we propose a novel framework
for tackling unsupervised Re-ID tasks across multiple cameras.
The framework of the proposed method is presented in Fig. 2, which
can be summarized into two-stage iterations: Pseudo-Label Generation
and Multi-Domain Semantic Propagation. In the first stage, we use a
feature extractor to extract feature vectors  of all the samples in the
training set  𝑡𝑟 . ResNet-50 [2] is used as the backbone along with
the average pooling layer and the batch normalization layer to form
the feature extractor network. The clustering algorithm DBSCAN [21]
is applied to feature vectors for generating identity pseudo-labels .
Note that the unclustered outlier samples are discarded during the
model training. These generated pseudo-labels are used to optimize
the model. Specifically, we construct a cluster-level memory dictionary,
which stores a single feature vector for each cluster and is dynamically

𝑚𝑇𝑘 = 𝛼 ⋅ 𝑚𝑇𝑘 −1 + (1 − 𝛼) ⋅ 𝑓 ,

(4)

where 𝛼 ∈ [0, 1) is the momentum updating coefficient, 𝑓 ∈ 𝑘 is
the feature vector corresponding to 𝑚𝑘 in the mini-batch, and the
superscript 𝑇 represents the 𝑇 th training epoch.
(3) Loss Function: Given a representation 𝑓𝑞 = 𝜙(𝑥𝑞 ) encoded by the
feature extractor, we compute a non-parametric classification loss with
one positive representation and all the negative representations in the
memory to supervise the training of 𝜙:
exp(𝑓𝑞 ⋅ 𝑚𝑓𝑞 ∕𝜏1 )
𝑐 𝑙𝑠 = E[− log ∑𝑁 𝑚
],
(5)
𝑖=1 exp(𝑓𝑞 ⋅ 𝑚𝑖 ∕𝜏1 )
where 𝑚𝑓𝑞 denotes the cluster representation of the class that 𝑓𝑞 belongs
to, 𝜏1 is a temperature hyper-parameter.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 174 (2026) 112947

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Bridging the gap : Learning adaptive knowledge transition for lifelong
person re-identiﬁcation
Jinjia Peng
a

a,b , Chengjun Wang

c,∗, Jican Tan

a,b , Huibing Wang

d , Xianping Fu

d

School of Cyber Security and Computer, Hebei University, Baoding, 071000, Hebei, China

b Hebei Machine Vision Engineering Research Center, Baoding, Hebei, 071000, China
c

School of Computer Science and Engineering, South China University of Technology, Guangzhou, Guangdong, 511400, China

d School of Computer Science and Technology, Dalian Maritime University, Dalian, Liaoning, 116024, China

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Lifelong learning
Knowledge distillation

a b s t r a c t
Lifelong person re-identiﬁcation (LReID) is a crucial task that meets the demand for deployment with the continuous data stream in real-world scenarios. Since all person data cannot be accessed at the same time, existing
methods mainly focus on directly building cross-domain constraints to maintain the previous knowledge. However, it is uncontrollable and ineﬃcient which hinders the generalization of the LReID model to new tasks due to
the varied domain bias. To address the problem, this paper proposes an Adaptive Knowledge Transition (AKT)
framework, which weakens the inhibition of generalization capacities by adaptively transferring knowledge.
Speciﬁcally, the Transition Factor Estimation (TFE) is designed to fetch domain-speciﬁc styles from diﬀerent
domains, which conducts appropriate knowledge transfer paths for progressive distillation. Then the Knowledge
Integration Module (KIM) merges the styles between domains to generate the intermediate transition knowledge. Moreover, Multi-Relation Distillation (MRD) is proposed to explore the relationship across domains from
diﬀerent perspectives for ﬂexible knowledge distillation. It eﬀectively preserves the learned knowledge by establishing distillation constraints in samples and classes separately. Comprehensive experiment results demonstrate
an advanced performance on benchmark datasets compared to state-of-the-art LReID methods.

1. Introduction
Person re-identiﬁcation (ReID) aims to identify the target person
across diﬀerent views or cameras. It has been widely investigated as a
subtask of image retrieval [1,2], which prefers to enhance the discriminative ability of appearance features such as shape or color. Although
advanced performances have been achieved recently, these methods often assume that the training process is conducted in an ideal environment where all the data is always accessible. In fact, person data are
continuously captured over time from diﬀerent locations or cameras,
which poses a key challenge for the deployment of Re-ID in real-world
scenarios.
To address the problem, Lifelong Person ReID (LReID) is proposed,
which is adapted to the continuous data stream. It requires that the
model not only adapts to the incoming identities but also maintains a
stable distinguishability in the previous person. Recently, existing LReID
methods have explored several solutions to address the key challenge of
catastrophic forgetting. In particular, replayer-based methods [3,4] are
the most intuitive approach, which employs a data replayer to store a
few samples after training the current task. While tackling the incoming

tasks, it actively provides the model with some cues to assist in recalling
previous data distributions. Although data replayer has proven eﬀective
in preventing catastrophic forgetting, it is not suﬃcient to completely
solve the LReID problem.
In addition, knowledge distillation-based methods [5,6], which have
made signiﬁcant progress, are currently the mainstream approach to
LReID. Inspired by the teacher-student network [7] framework, these
methods freeze and store model parameters at the end of previous task
training, designating it as a reference model. It usually enforces the consistency of the outputs between the reference model and the current
model to prevent catastrophic forgetting. However, distilling knowledge
directly through the features from the reference model to the current
model is uncontrollable and ineﬃcient, which may hinder its ability to
improve generalization in new tasks.
In light of the above discussion, improving the generalization performance of LReID models can be summarized as two main challenges: (1)
how to eﬀectively control the degree and direction of knowledge distillation and (2) exploit what kind of information to facilitate the knowledge
distillation. To address the problems, this paper proposes an Adaptive
Knowledge Transition (AKT) framework, which consists of two main

∗ Corresponding author.

E-mail address: cswangchengjun@mail.scut.edu.cn (C. Wang).
https://doi.org/10.1016/j.patcog.2025.112947
Received 20 November 2024; Received in revised form 21 October 2025; Accepted 16 December 2025
Available online 23 December 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 174 (2026) 112947

J. Peng et al.

2. Related work
2.1. Lifelong learning
Lifelong learning, or continual learning [8,9], aims to train a neural
network model with memory capabilities that adapts new task knowledge while retaining previously learned knowledge. With the advancement of deep neural networks, existing subtasks such as object detection,
image generation, and image retrieval have shown signiﬁcant performance improvements when pre-trained on speciﬁc domain data. However, the conventional training approach requires the retraining of a
completely new model for each speciﬁc data domain, which leads to a
signiﬁcant waste of resources.
Therefore, recent research has focused on lifelong learning that
can generalize to multiple tasks through a single training process. The
key challenge in lifelong learning is catastrophic forgetting, where
the performance of the model on previous tasks deteriorates signiﬁcantly as it is trained on new tasks. To address this problem, existing methods can be categorized into four types: replayer-based methods, regularization-based methods, parameter isolation-based methods,
knowledge distillation-based methods and multi modality-based methods.
The replayer-based method utilizes a sample replay buﬀer to store
a small amount of old data, such as original images or feature vectors
extracted by the model and provides hints from the old data during
training on new tasks. RM [10] proposes a per-sample classiﬁcation
uncertainty replayer strategy, which focuses on storing samples with
boundary ambiguity in each class. SSD [11] proposes to summarize the
knowledge from streaming data into more informative samples by extracting training features from real images. However, the replayer-based
method can only store a small number of samples from old tasks, which
may lead to a signiﬁcant bias in the weight of the classiﬁer.
To alleviate the pronounced bias in classiﬁer weights, regularizationbased methods are introduced, which utilize regularization of the loss
[12] to balance the weights of the classiﬁer. Hou et al. [13] introduce
cosine normalization to enhance the weight of old knowledge, while
Zhao et al. [14] employ bias correction to rectify the bias in classiﬁer
weights.
Furthermore, the parameter isolation-based method explicitly avoids
catastrophic forgetting by separating the parameters of diﬀerent tasks.
For instance, RPSNet [15] pre-allocates multiple parallel networks to
construct candidate paths and selects the optimal path for each task.
MEAT [16] utilizes binary masks to select task-speciﬁc neurons or parameters, isolating training parameters for each task. While the parameter isolation-based method eﬀectively prevents catastrophic forgetting,
it requires the storage of multiple frozen models, leading to signiﬁcant
resource consumption.
The knowledge distillation-based method is currently the mainstream approach in lifelong learning, inspired by the teacher-student
network [7]. It acquires knowledge cues for old tasks from frozen old
models through knowledge distillation. TA [17] designed a teacher
model adaptive framework that adapts to the data distribution of new
tasks by dynamically updating the batch normalization layer statistics
of the teacher model. DFKD [18] proposed a data-free knowledge distillation method that maintains a dynamic collection of generated samples
over time.
Recently, with the widespread adoption of multimodal large models,
researchers are struggle to explore the advantages of pretrained multimodal model such as CLIP [19] in continual learning. GMM [20] demonstrates that generative multimodal model is able to serve as excellent
incremental learners, eﬀectively avoids the classiﬁer bias in traditional
classiﬁer. Meanwhile, LVP-CLIP [21] enhances the continual learning
capability of CLIP models by constructing a label vector pool, preserving learned knowledge eﬀectively. Furthermore, C-CLIP [22] employs

Fig. 1. Comparison in knowledge transfer strategy between baseline and the
proposed Knowledge Transition Learning (KTL). To facilitate the knowledge
transfer, KTL explores an approximate path by estimating the relation between
the previous and current knowledge and generates the intermediate knowledge.

components: Knowledge Transition Learning (KTL) and Multi-Relation
Distillation (MRD).
To tackle the ﬁrst problem, the KTL is designed to explore suitable
paths for facilitating the adaptive knowledge transition. As shown in
Fig. 1, KTL aims to construct the simulated distribution from two distant
knowledge domains. In particular, KTL employs Transition Factors Estimation (TFE) to fetch the domain-speciﬁc style from diﬀerent domains,
which adaptively explores the direction of knowledge transfer. Combined with transition factors, the Knowledge Integration Module (KIM)
merges the style across domains to construct the progressive knowledge
distillation.
Furthermore, MRD is proposed to address the second problem,
which explores latent relationships from diﬀerent perspectives to further facilitate the knowledge transition. It leverages the distance between samples and class centroids to promote knowledge refreshment
in centroid bias distillation. Moreover, the relationship between samples is utilized in instance interaction distillation to mitigate the impact of catastrophic forgetting. These informative relationships are
employed to build the constraint, which exhibits greater relevance
and eﬀectively enhances the knowledge distillation. By establishing
these modules, the model not only addresses the problem of catastrophic forgetting but also improves generalization to new tasks eﬀectively. The primary contributions of this paper can be summarized as
follows:

•

An Adaptive Knowledge Transition framework is proposed to tackle
the challenge of LReID by facilitating the knowledge adaptive transfer. This framework not only addresses the detrimental eﬀects of
catastrophic forgetting on models but also improves the generalization capabilities to new tasks.
• To eﬀectively control the direction and degree of knowledge distillation, Knowledge Transition Learning(KTL) is designed, which conducts the suitable paths for facilitating knowledge distillation. Different from existing methods, it conducts an intermediate knowledge transition, which promotes the knowledge transfer between
models.
• To further facilitate the knowledge transfer, Multi-Relation Distillation is proposed, which aims to explore latent relationships for ﬂexible knowledge transition. It eﬀectively preserves the learned knowledge by building the distillation constraints in samples and classes
respectively.

2

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Fig. 2. Overview of the Adaptive Knowledge Transition (AKT) framework. It mainly consists of Knowledge Transition Learning (KTL) and Multi-Relation Distillation
(MRD). In KTL, the feature maps extracted from the current and reference model are ﬁrst fed into TFE to obtain the transition factor by multi-scale feature extraction.
Then the KIM generates the simulated features 𝑓 𝑚 by transition factor to construct the intermediate knowledge transition. The MRD explores the centroid-to-sample
and sample-to-sample relationship for each feature to further enforce knowledge consolidation.

low-rank adaptation update strategy to eﬀectively mitigate catastrophic
forgetting.

The current mainstream Lifelong Person Re-identiﬁcation (LReID)
methods employ a reference model of knowledge distillation to construct a strong baseline method. In particular, AKA [5] generates a fully
connected graph based on the relationships to improve knowledge consolidation for old tasks, which transfers the knowledge from the reference model to the current model through graph convolution. Building on
this, MEGE [25] introduces ranking consistency distillation in a diﬀerentiable manner to reﬁne knowledge and further prevent catastrophic forgetting. GwFReID [3] improves knowledge transfer by enforcing consistency in feature representation and prediction between the two models.
PKD [27] samples random patches from images, which construct the relation patch relation distillation to improve the informative knowledge
transfer.
In addition, other methods are devoted to exploring eﬀective knowledge transfer for LReID. To avoid erroneous or harmful knowledge from
aﬀecting the generalization of the model, LSTKC [28] utilizes a correction matrix to ﬁlter and correct the knowledge of the reference model.
Building upon this framework, LSTKC+ [29] incorporates a knowledgeguided parameter fusion strategy to dynamically balance novel and historical knowledge, improving long-term knowledge retention.
Further, KRKC [30] removes parameter freezing of the reference
model and improves knowledge transfer between models by jointly
training the reference and current models. C2R [31] designs a continual compatible transfer network to consolidate the learned feature and
expand it into the new feature space. DKP [32] utilizes the instancelevel diversity of each sample to transfer comprehensive ﬁne-grained
information for facilitating knowledge preservation. In this paper, AKT
leverages adaptive knowledge transfer to not only maximize the generalization ability on new tasks but also to mitigate catastrophic forgetting.

2.2. Person re-identiﬁcation
Person Re-ID recently attracts considerable attention and makes remarkable progress. Current Re-ID research includes four main ﬁelds:
Fully Supervised Learning (FSL), Unsupervised Domain Adaptation
(UDA), Purely Unsupervised Learning (PUL), and Lifelong Learning.
Recently, FSL [2] focuses on exploring the discriminative representation to eﬀectively identify the target person, which remains the advanced performance. However, it is impractical to annotate the numerous person images manually because of the vast consumption of
resources. To address the challenge, UDA [23] is explored, which leverages labeled source domain data to transfer knowledge, eﬀectively
reducing the dependence on manual labels in the target domain. It
primarily emphasizes aligning the features between the two distinct
domains to improve the model’s ability of generalization in the target
domain. PUL [24] is another approach to address the dependence on
manual labels. It does not depend on any manual labels, making it more
challenging yet widely studied. PUL Re-ID mainly improves the generalization ability by increasing the precision of clustering in generating
pseudo-labels.
Indeed, the aforementioned research ﬁelds have eﬀectively facilitated the deployment of Re-ID in real-world scenarios. However, in
the real world, Re-ID also needs to cope with the increasing data ﬂow
over time. Particularly, after training on a dataset from one domain,
the Re-ID model needs to train on added domains as deployment expands. In this situation, without proper constraints, the Re-ID model
will gradually forget the initial data distribution of the domain, which
may hinder its practical deployment in real-world scenarios. Therefore,
recent research eﬀorts have focused on Lifelong Person Re-ID [25,26].
It aims to train a Re-ID model that not only exhibits excellent generalization capabilities for new tasks but also maintains its performance on
old tasks.

3. Adaptive knowledge transition
3.1. Problem statement
In this section, this paper introduces the problem statement and the
setting under the baseline of LReID. Given 𝑇 dataset tasks  = { 𝑡 }𝑇𝑡=1 ,
3


codex
前三篇的共同点已经很清楚：它们都把常见干扰拆成“域、相机、模态、历史任务”这类结构变量，再用中间状态或图传播把直接对齐改成可控转移。这个套路对我们有用，但单纯照搬到强 SOLIDER backbone 上大概率会变成工程组合。
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
exec
/bin/zsh -lc "pdftotext -l 3 'Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Categorical Attention: Fine-grained Language-guided Noise Filtering Network for
Occluded Person Re-Identification
Minghui Chen1,2 , Dayan Wu1∗ , Chenxu Yang1,2 , Qinghang Su1,2 , Zheng Lin1
1
Institute of Information Engineering, Chinese Academy of Sciences
2
School of Cyber Security, University of Chinese Academy of Sciences
{chenminghui, wudayan, yangchenxu, suqinghang, linzheng}@iie.ac.cn
Abstract

Visual Encoder

Person Re-Identification (ReID) aims to match individuals across different camera views, but occlusions in real-world scenarios, such as vehicles
or crowds, hinder feature extraction and matching. Current occluded ReID methodologies typically leverage visual augmentation techniques in
an attempt to mitigate the disruptive effects of
occlusion-induced noise. However, relying solely
on visual data fail to effectively filter out occlusion noise. In this paper, we introduce the Finegrained Language-guided Noise Filtering Network
(FLaN-Net) for occluded ReID. FLaN-Net innovatively employs categorical attention mechanism to
generate adaptive tokens that capture the following
three distinct types of visual information: comprehensive descriptions of individuals, detailed visible
attributes, and characteristics of occluding objects.
Subsequently, a cross-attention mechanism aligns
these prompts with the image, guiding the model to
focus on relevant regions. To generate robust and
discriminative features for occluded pedestrians,
we further introduce a dynamic weighting fusion
module that integrates visual, textual, and crossattention features based on their reliability. Experimental results demonstrate that FLaN-Net outperforms existing methods on occluded ReID benchmarks, offering a robust solution for challenging
real-world conditions.

1

Text Encoder
(a) PFD

Textual Features
Only visual features

Pose estimation

(b) CLIP-ReID

Split

Have textual prompts
Coarse-grained

A photo of a [𝑿]𝟏 [𝑿]𝟐 … [𝑿]𝑴 𝐩𝐞𝐫𝐬𝐨𝐧

Constrastive Loss
(c) FLaN-Net (Ours)

Fine-grained prompts
More effective interaction

A photo of a [𝑺∗ ] person with 𝑨∗ partially
occluded by 𝑶∗

Multi-modal Feature
Enhancement

Figure 1: Comparison of baselines and our fine-grained languageguided noise filtering network. (a) PFD, (b) CLIP-ReID, (c) Our
proposed FLaN-Net method, which incorporates fine-grained textual descriptions and a more effective multi-modal interaction module, enables the model to achieve more robust noise suppression.

Hou et al., 2021], various strategies have been proposed to
mitigate the effects of noisy information caused by occlusion. Common approaches, such as auxiliary models [Hou
et al., 2021; Wang et al., 2022a; Dou et al., 2023] and
attention mechanisms [He et al., 2021; Tan et al., 2022;
Jia et al., 2023] help the model distinguish key information
from occlusion-induced noise. PFD [Wang et al., 2022a] is
a notable method that integrates the auxiliary model and attention mechanism, employing pose-guided feature disentangling to reduce occlusion noise by associating features with
human body parts. While auxiliary models provide external
information and attention mechanisms adaptively focus on
unobstructed regions, their reliance solely on visual data often fails to completely filter out occlusion noise. This leaves
occlusions still being misinterpreted as identity-relevant features, yielding less discriminative representations. As shown
in Fig. 1, the attention maps reveal the varying performance
of different methods under diverse occlusion scenarios. It can

Introduction

Person Re-Identification (ReID) aims to identify and match
the same target individual across different and nonoverlapping camera views [Ye et al., 2021]. However, people and objects often move randomly, and surveillance devices typically cover wide areas in the real-world scenario,
which leads to a high likelihood of individuals being partially occluded. This occlusion creates a major challenge for
person re-identification, as it introduces significant noise during feature extraction and feature matching. To cope with
occluded ReID [Zheng et al., 2015b; Zhuo et al., 2018;
∗

Visual Features

Corresponding Author

801

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

• We propose an advanced multi-modal feature enhancement paradigm that combines a cross-attention mechanism with a dynamic weighting fusion module. This
integration is designed to produce robust and discriminative representations of occluded pedestrians.
• Experimental results demonstrate that FLaN-Net significantly improves retrieval performance on both occluded
and holistic ReID benchmarks, outperforming state-ofthe-art methods.

be observed from Fig. 1(a) that the attention maps generated
by PFD highlight many irrelevant regions.
Inspired by the potential of vision-language models, recent works have explored the integration of textual descriptions with visual features to enhance feature extraction. As
a representative work, CLIP-ReID [Li et al., 2023] deploys
the vision-language model CLIP [Radford et al., 2021] to
enhance the learning of visual features by training a set of
tokens for each pedestrian ID. However, existing languageguided ReID methods are not well-suited for the occluded
ReID task, as their prompts tend to capture global information from images, failing to provide the necessary contextual
information in occluded scenarios. Furthermore, textual information in these methods is not fully utilized, as it is primarily employed for contrastive loss with images rather than
explicitly guiding feature extraction. As shown in Fig. 1(b),
although CLIP-ReID helps reduce noise in occlusion scenarios, its attention maps still frequently emphasize occluding
objects and irrelevant background regions, retaining significant noise in the extracted features.
To address these limitations, we propose the Fine-grained
Language-guided Noise Filtering Network (FLaN-Net) for
Occluded Person Re-Identification. FLaN-Net innovatively
employs a categorical attention mechanism to generate adaptive tokens that capture three distinct types of visual information: comprehensive descriptions of individuals, detailed
visible attributes, and characteristics of occluding objects.
Specifically, our method transforms image information into
subject and detail tokens, and then represents each image
with a occlusion-aware fine-grained textual prompt in the format:”A photo of a [S ∗ ] person with [A∗ ] partially occluded
by [O∗ ].” Here, S ∗ denotes the primary subject token, which
encapsulates the essential identity of the pedestrian. The attribute token A∗ and the context token O∗ serve as detail tokens, capturing identity-relevant attributes and occluding objects respectively. FLaN-Net learns detailed information for
each pedestrian through the guidance of multiple learnable
queries. Moreover, to ensure effective utilization of these
prompts, FLaN-Net incorporates a cross-attention mechanism that dynamically aligns textual tokens with image patch
tokens. This alignment enables the image encoder to focus on
semantically relevant and visible regions of the pedestrian,
filtering out noise caused by occlusions. Finally, FLaN-Net
combines the features from the image encoder, text encoder,
and cross-attention in a dynamic weighting fusion module
to generate a robust feature representation for each occluded
pedestrian. The dynamic fusion module assigns weights to
each feature based on its uncertainty, highlighting the influence of more reliable features. As shown in Fig. 1(c), our
method effectively highlights the visible regions of the pedestrian while filtering out noise from occluding objects, demonstrating improved focus and robustness in occluded scenarios.
We summarize our contributions as follows:
• We introduce FLaN-Net, a novel method that employs
a fine-grained language-guided mechanism to construct
adaptive occlusion-aware prompts for occluded ReID.
This technique enables meticulously detailed descriptions of the visible aspects of an individual, while effectively filtering out noise caused by occlusions.

2

Related Work

2.1

Occluded Person Re-identification

Occluded Person Re-Identification (Occluded ReID) presents
significant challenges due to the noise introduced by occlusions, which hinder the model’s ability to extract and
match features accurately. One common approach to mitigate this issue is the incorporation of auxiliary information, such as pose estimation [Hou et al., 2021; Wang et
al., 2022a] and human parsing models [Gao et al., 2020a;
Dou et al., 2023]. Another widely used approach involves
attention mechanisms [He et al., 2021; Tan et al., 2022;
Jia et al., 2023; Li et al., 2024], which enhance robustness by
adaptively focusing on relevant regions of an image. To further aid the attention learning process, various data augmentation strategies, such as random erasing [Wang et al., 2022b]
and artificially generated occlusions [Chen et al., 2021; Xia
et al., 2024; Tan et al., 2024], have been incorporated to help
the model better handle occlusions across diverse scenarios.
However, these methods rely only on visual features, neglecting the potential of text information to help filter out noise.
In recent years, language-guided methods [Li et al., 2023;
Yang et al., 2024] have been explored to address the challenges of ReID. CLIP-ReID [Li et al., 2023] is the pioneering work that uses the CLIP [Radford et al., 2021] model to
integrate textual prompts with visual features.

2.2

Image-to-Word Mapping

In the field of text-to-image generation, [Gal et al., 2022]
was the first to use novel pseudo-words in the word embedding space to represent an object or a style. Recently, this
technique, known as textual inversion, has been widely applied to zero-shot compositional image retrieval tasks [Saito
et al., 2023; Suo et al., 2024]. These methods map a reference image to a pseudo-token in the CLIP embedding space,
which is then combined with a descriptive query to facilitate
text-to-image retrieval. However, a limitation of these models is that they map the entire image to a single pseudo-token,
which can introduce noise and overlook important details relevant to the retrieval task. Recent studies have proposed using
learnable queries to capture fine-grained features within an
image. For example, in instance segmentation task, learnable
queries are used to explicitly represent an object’s class, location, and mask [Dong et al., 2021]. In object detection task,
learnable queries are used to capture object relationships and
global context for parallel predictions [Carion et al., 2020].
Inspired by these works, this paper proposes incorporating
learnable queries into occluded ReID tasks to perform finegrained textual inversion.

802

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

(b) Multi-modal Feature Enhancement

(a) Occlusion-aware Fine-grained Prompt Construction
A

photo

of

a

𝑺∗

person

with

∅𝒔
Input Image

𝑨∗

vl1
vl2
…

vlm
Trainable

occluded

𝑶∗

by

∅𝒂

vg
Visual
Encoder

partially

×n

𝝀 𝒗𝒈

∅𝒐

Average

Average

…

…

+

×n

𝐿𝐴𝑙𝑖𝑔𝑛

larger

𝝀 𝒕𝒑

Feed-Forward

+

+

Cross-Attention

Cross-Attention

…

×

greater
uncertainty

+

Feed-Forward

Learnable Queries

vg

Text
Encoder

Learnable Queries

×
less
uncertainty
smaller

K
V

Cross
Attention

+

𝝀𝒔𝒂𝒕𝒕

Q

…

Frozen

tp

𝐿 𝑇𝑟𝑖𝑝𝑙𝑒𝑡
+
𝐿𝐼𝐷
+
𝐿𝐴𝑟𝑐𝐹𝑎𝑐𝑒

satt

×

Figure 2: The framework of FLaN-Net: (a) Occlusion-aware Fine-grained Prompt Construction, which generates subject and detail tokens to
capture identity-specific features; (b) Multi-modal Feature Enhancement, which aligns textual prompts with visual regions via cross-attention
mechanism and combines visual, textual, and cross-attention features through a dynamic weighting fusion module.

3

n
v = fv (I) = {vg , vl1 , . . . , vl
}. Here vg ∈ Rd×1 denotes
n
the global visual feature and vli i=1 ∈ Rd×n represent the
local patch features, where d is the feature dimension and n
is the number of patches. Then we apply a simple mapping
network ϕs to transform the global image feature vg into a
subject-focused pseudo-word token. Formally, we define:
S ∗ = ϕs (vg ),
(1)
where ϕs is a three-layered fully-connected network and S ∗
serves as a comprehensive description of the main subject.

Methodology

We introduce the Fine-grained Language-guided Noise
Filtering Network (FLaN-Net) for Occluded Person ReIdentification, which consists of two key components, as
shown in Fig. 2. The first component, Occlusion-aware
Fine-grained Prompt Construction, generates adaptive tokens that capture three types of visual information: comprehensive descriptions of individuals, detailed visible attributes,
and characteristics of occluding objects. These tokens are
used to generate fine-grained textual descriptions, which provide a comprehensive representation of the pedestrian. The
second component, Multi-modal Feature Enhancement, integrates visual and textual information to improve feature extraction. This is achieved through a cross-attention mechanism that aligns textual prompts with specific image regions,
and a dynamic weighting fusion module, which adaptively
combines the features to ensure that the most relevant and reliable features are given larger weight.

3.1

Detail-focused Token Generation
The detail tokens are divided into two types: attribute tokens A∗ , which describe identity-relevant visible attributes,
and context tokens O∗ , which represent characteristics about
the occluding object. Both A∗ and O∗ are generated using
the same network architecture, leveraging learnable queries
to extract relevant features from the input image. To illustrate
this process, we take the generation of A∗ as an example and
provide a detailed
 nexplanation of its network structure.
Let Vl = vli i=1 ∈ Rd×n represent the local patch features. Next, these patch features are fed into a fine-grained
noise filtering network. This network interacts with a set of
d×m
m learnable queries X = {xi }m
through crossi=1 ∈ R
attention, allowing these queries to capture attribute information from the corresponding semantic regions in the image.
For A∗ , this process enables these queries to focus on specific visual details, capturing exposed attribute features such
as clothing colors, accessory types, and hairstyles. Similarly,
for O∗ , these queries could extract information about the occluding objects. Specifically, we compute the query Qd , key
Kd , and value Vd matrices as follows:
Qd = XWQ , Kd = [X, Vl ] WK , Vd = [X, Vl ] WV ,
(2)
where WQ , WK , WV are different linear transformations and
[X, Vl ] denotes the concatenation of the learnable queries and

Occlusion-aware Fine-grained Prompt
Construction

To effectively represent occluded images, our method
projects the image into pseudo-word tokens, including a
subject-focused token S ∗ for the individual’s essential identity and detail-focused tokens A∗ and O∗ for visible attributes
and occluding objects. These tokens are then combined to
construct a personalized prompt:”A photo of a [S ∗ ] person
with [A∗ ] partially occluded by [O∗ ].” This occlusion-aware
prompt enhances the model’s ability to distinguish individuals in occluded ReID tasks.
Subject-focused Token Generation
In order to get the subject-focused token, we leverage the image encoder fv of the pre-trained CLIP model. Specifically,
given an image I, the visual encoder extracts visual feature

803


 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

1

Causal Clothes-Invariant Feature Learning for
Cloth-Changing Person Re-ID
Xulin Li , Yan Lu , Bin Liu , Member, IEEE, Jiaze Li , Yating Liu , Qi Chu , Member, IEEE, Mang Ye ,
Senior Member, IEEE, Wanli Ouyang , Senior Member, IEEE, Nenghai Yu

Abstract—In cloth-changing person re-identification (CCReID), it is critical to learn clothes-invariant feature, which can
provide discriminative ID features that remain robust against
clothing changes. However, a spurious correlation currently limits
existing ReID methods from effectively extracting these clothinginvariant features. This spurious correlation arises from clothing
ownership: clothing is rarely shared across different identities, so
models tend to memorize clothing cues for identity recognition,
and this strategy generalizes poorly to unseen clothing. In this
paper, we propose Causal Clothes-Invariant Learning (CCIL),
which explicitly shifts CC-ReID from likelihood learning P (Y |X)
to causal intervention learning P (Y |do(X)) to block the clothing
shortcut. CCIL realizes this intervention through three modules:
a Confounder Dictionary, an Intervention Module, and Disentangle Regularization. The causality-based modeling makes the
entire model naturally clothes-invariant, effectively preventing
the capture of spurious correlations in feature learning. Extensive
experiments validate the effectiveness of CCIL. On PRCC and
DeepChange datasets, CCIL achieves Rank-1 accuracies of 66.4%
and 59.2%, outperforming state-of-the-art methods by 1.4 and
4.1 percentage points, respectively.
Index Terms—Cloth-changing person re-identification, Causal
intervention, Invariant feature learning

I. I NTRODUCTION

P

ERSON re-identification (ReID) aims to retrieve specific
pedestrians across different cameras, which is widely used
in city surveillance, intelligent security, and related areas. It
is challenging due to the existence of frequent occlusion [1],
background interference [2], and illumination variations [3].
Significant progress [4]–[6] has been observed in standard
person ReID, which assumes persons do not change their
clothes throughout the entire retrieval process. To meet the
long-term retrieval requirements of real-world scenarios, clothchanging ReID (CC-ReID) [7], [8] takes into consideration the
extra challenge of clothing variations.
This work is supported by the National Natural Science Foundation of
China (Grant No. 62272430).
Xulin Li, Jiaze Li, Bin Liu, Qi Chu and Nenghai Yu are with School of Cyber Science and Technology, University of Science and Technology of China,
Hefei 230026, China and Anhui Province Key Laboratory of Digital Security
(e-mail: (lxlkw, jz li)@mail.ustc.edu.cn; (flowice, qchu, ynh)@ustc.edu.cn)
Yating Liu is with School of Data Science, University of Science and Technology of China, Hefei 230026, China (e-mail: liuyat@mail.ustc.edu.cn) Yan
Lu, and Wanli Ouyang are with Shanghai Artificial Intelligence Laboratory,
Shanghai, China (e-mail: luyan@pjlab.org.cn; wanli.ouyang@sydney.edu.au)
MangYe is with Wuhan University, Wuhan 430072, China (e-mail: yemang@whu.edu.cn)
Xulin Li and Yan Lu contributed equally to this work. Corresponding
author: Bin Liu.

Clothes Ownership
ID 1

Clothing Shortcut

ID 2

ID 3

×

×

×

×

×

×

×

×

×

Feature Learning

ID 1
P(Y|do(X))
Clothes
Ownership
Causal Relationship

ID 1

×
(a) Training
Clothing Shortcut
Model Inference

Clothes
Ownership

Incorrect
Match

×

×

Clothes
Ownership

Causal Relationship

Correct
Match

(b) Testing with unseen IDs and clothing

Fig. 1. The spurious correlation between clothing and IDs can lead to a
clothing shortcut of identifying persons based on their clothing. This shortcut
is not a robust recognition process as it fails when encountering IDs and
clothing not in the training set. Our approach uses the causal intervention
P (Y |do(X)) to enable the deep model to solely learn the causal relationship
based on discriminative ID clues, thus demonstrating good generalizability in
testing scenarios.

Whatever specific ReID tasks are, invariant feature learning
is critical, which guides the model to extract invariant cues
and adapt to complex scenarios. In the CC-ReID task, clothesinvariant feature learning holds great importance as it requires
discriminative features that are robust to clothing changes.
Compared with other interfering factors (e.g., scene, illumination, camera, and viewpoint variations), clothing changes are
particularly challenging for invariant feature learning. This is
because clothing and human IDs exhibit strong spurious correlations, while correlations between other interfering factors
and identity are typically much weaker.
As shown in Figure 1, the spurious correlation is caused
by a fact that each outfit is only worn by its owner and
hardly shared with others. This kind of clothes ownership
is naturally reflected in existing CC-ReID datasets [7]–[9],
resulting in a clothing shortcut that recognizes people by
identifying their wearing clothes. Existing methods inevitably

Copyright © 2026 IEEE. Personal use of this material is permitted. However, permission to use this material for any other purposes must be obtained from the IEEE by
sending an email to pubs-permissions@ieee.org.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

2

capture this shortcut because it would help the model achieve
training targets more directly.
However, this clothing shortcut is an unreliable relationship.
When we deploy such a model in real applications, facing
unknown people wearing unseen clothes before, the model
still attempts to utilize the aforementioned clothing shortcut
to recognize people, leading to bad re-identification results.
To address this challenge, we proposed a Causal ClothesInvariant Learning (CCIL), which focuses on learning causal
relationships. This new causality-based training framework can
avoid the clothing shortcut during training, which promotes
invariant feature extraction. The main idea of the proposed
CCIL is to model the causal intervention probability [10], [11],
denoted as P (Y |do(X)), rather than the likelihood probability
P (Y |X) commonly modeled by most existing methods, where
Y means the human ID and X means the input image. As
shown in Figure 1, the P (Y |do(X)) only models the causal
relationships from X to Y and does not include the clothing
shortcut, while P (Y |X) models all relationships between X
and Y . The causal relationship reflects the stable association
of how to infer human identity from a given image by using
discriminative ID clues, making the models applicable to
various complex scenarios. So the CCIL which models the
P (Y |do(X)) could capture more stable causal patterns and
neglect the spurious correlation, leading to better clothesinvariant representation for CC-ReID.
To make the CCIL model the P (Y |do(X)) better within a
CC-ReID framework, we design our CCIL by following three
novel modules. Firstly, we learn the representation distribution
of clothing in the dataset and store them in a Confounder
Dictionary to support subsequent causal intervention implementation. Secondly, the Intervention Module employs a novel
and effective approach to model the backdoor adjustment formula in causal theory, leading to the derivation P (Y |do(x)).
Lastly, the Disentangle Regularization improves the modeling
of clothing in the Confounder Dictionary, thereby further
enhancing the effectiveness of the Intervention Module. The
overall feature learning process is combined within the causal
intervention, resulting in clothes-invariant feature extraction.
Our main contributions are summarized as follows:
• We analyze the barrier of clothes-invariant feature learning
in CC-ReID, which is the clothing shortcut in the training
set, and propose a novel framework dubbed Causal ClothesInvariant Learning (CCIL) via the causal intervention view.
Our CCIL first provides a new causal solution based on
backdoor adjustment for the CC-ReID task, which focuses on
eliminating the clothing shortcut.
• To achieve the CCIL, we present three causal modules
to implement confounder modeling, causal intervention, and
disentangled regularization, to jointly model P (Y |do(X)).
With our CCIL, deep models are guided to overlook the
clothing shortcut and instead capture more discriminative ID
clues.
• Extensive experiments on multiple CC-ReID datasets validate the effectiveness and superiority of our method against
the state-of-the-art causal and non-causal ReID methods in
mitigating the clothing shortcut.

II. R ELATED W ORK
Cloth-Changing Person ReID. Person re-identification
(ReID) is the task of retrieving persons of interest across nonoverlapping cameras. However, standard ReID methods [12],
[13] face limitations in long-term scenarios where individuals
change their clothes over time. Consequently, there has been
a growing interest in the field of cloth-changing person reidentification (CC-ReID) [7]–[9], [14]–[17] in recent years.
Most CC-ReID methods primarily use extra clothingagnostic modalities data to guide model training. Yang et
al. [8] used pure contour sketches for discriminative feature learning. Chen et al. [18] directly extracted a textureinsensitive 3D shape embedding from a 2D image by adding
3D body reconstruction as an auxiliary task. Hong et al. [19]
used 2D silhouettes, Qian et al. [7] introduced key points, Jin
et al. [20] and Lu et al. [21] utilized gait, Cui et al. [22]
and Xiong et al. [23] used human parsing, Li et al. [24]
introduced clothes templates to assist robust features learning
to clothes change. Other methods only use the original RGB
image to solve CC-ReID. Huang et al. [15] and Shu et al. [16]
improved CC-ReID from the network architecture and ranking
loss perspectives, respectively. Gu et al. [25] introduced an
adversarial loss to decouple clothes-irrelevant features from
the RGB modality. Han et al. [26] proposed clothing-change
augmentation methods to address the limitation of the insufficient number and variation of clothing in training data.
These methods are essentially likelihood-based and optimize
P (Y |X), which inevitably captures the clothing shortcut. Even
methods using clothing-agnostic auxiliary modalities do not
discard RGB images, and the impact of spurious correlation
is diluted but not eliminated. In contrast, our CCIL models
P (Y |do(X)) via backdoor adjustment to directly cut off the
clothing shortcut through causal intervention.
Causal Inference in Person ReID. Thanks to the powerful
ability of causal inference to remove bias and pursue causal
effects, existing research has explored the incorporation of
causal inference in person ReID. Rao et al. [27] proposed
a method for counterfactual attention learning to enhance the
attention module. Li et al. [28] utilized the total indirect effect
(TIE) to emphasize the significance of graph topology in
cross-modality ReID task. These methods focus on enhancing
specific ReID modules, such as attention and graph modules, while our method is model-agnostic. Zhang et al. [29]
achieved domain-invariant representation learning through approximated causal interventions. Domain-invariant learning
cannot be applied to achieve clothes-invariant learning due to
their involving different confounders, domains, and clothing.
Consequently, we adopted different techniques for modeling
the confounder and implementing the causal intervention.
In CC-ReID, Yang et al. [30] utilized the total direct effect
(TDE) to highlight the direct effect (image → identity). In
contrast, we employ the backdoor adjustment method to guide
the model in capturing purer clothing-invariant features and
abandoning clothing-related shortcuts (image ← clothing →
identity) as much as possible, which results in more robust
CC-ReID results.
Causal Inference in Other Tasks. Recently, improving

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

3

X
C

C

Y

Images

Person ID

C Confounder
X

Y
𝐚 𝑃(𝑌|𝑋)

X

Y
𝐛 𝑃(𝑌|𝑑𝑜 𝑋 )

Clothing
Shortcut
Causal
Relation

Fig. 2. Causal graphs for CC-ReID. (a) The clothing shortcut X ← C → Y
captured by the likelihood-based methods that directly utilize P (Y |X) to
model the relationships from X to Y . (b) Our method models the relationships
by the causal intervention probability P (Y |do(X)), which only captures the
causal relation X → Y and removes the clothing shortcut.

deep learning through causal inference [31], [32] has received
increasing attention. It has been applied to various fields,
including categorization [33], visual question answering [34],
semantic segmentation [35], object detection [36], large language models [37]. The backdoor adjustment [10], [11] is
a causal inference theory to achieve the causal intervention
P (Y |do(X)). Some deep learning methods [35], [36], [38]–
[41] model confounders using the mean of features, and subsequently simulate backdoor adjustment through cross-attention
to mitigate the effects of confounders from their corresponding
tasks. We employed the backdoor adjustment algorithm for the
first time in CC-ReID with a novel approach for confounder
modeling and causal intervention implementation to perform
invariant feature learning.
III. C AUSAL A NALYSIS
In this section, we introduce causal theory to analyze the
negative effect caused by the clothing shortcut in the CC-ReID
task. We demonstrated that causal intervention can be achieved
by modeling P (Y |do(X)), which theoretically tackles the
clothing shortcut problem, thereby facilitating the extraction
of clothes-invariant feature.

B. Analysis of Existing Likelihood-Based Methods
The likelihood-based method is influenced by the clothesidentity spurious correlation, thus capturing a clothing shortcut
for identifying persons through their clothing. As shown in
Figure 2 (a) likelihood-based methods directly learn the probability P (Y |X) to model the correlation relationship between
the X and Y by deep models. However, the clothing shortcut
and the causal relationship are entangled together, directly
learning P (Y |X) will capture the clothing shortcut unavoidably. Besides, because of such kind of abstract modeling
approach, the corresponding patterns of the clothing shortcut
will be just represented in an implicit and unexplainable way.
Therefore, once the clothing shortcut X ← C → Y is captured
by the model, removing the corresponding patterns from the
model without affecting the useful knowledge related to the
causal relationships proves to be quite challenging.
C. Causal Intervention Helps Invariant Feature Learning
In theory, causal intervention can cut off the clothing
shortcut and learn the relationship unaffected by clothing,
which is consistent with the clothes-invariant objective. The
intervention operation can be defined at the Do-operation [42]
do(·), denoting the causal relationship between X and Y as
P (Y |do(X)). As shown in Figure 2 (b), the intervention disrupts the potential correlation between the intervening variable
X and its cause C, depicted as the removed arrow X ← C.
In this case, there is only the causal relation X → Y between
the X and Y . However, achieving intervention is not straightforward. Fortunately, the ‘backdoor adjustment’ theory [10],
[11] allows us to compute the intervention probability1 :
X
P (Y |do(X)) =
P (Y |X, c) · P (c).
(1)
c

In comparison to the original likelihood P (Y |X) in the
Bayesian framework:
X
P (Y |X) =
P (Y |X, c) · P (c|X),
(2)
c

A. Structural Causal Model For CC-ReID
We analyze the causality in the CC-ReID task by using a
Structural Causal Model (SCM) [10], [11]. The SCM is built
to depict the causal relationships among the variables ‘images’
X, ‘human ID’ Y , and confounder ‘clothes’ C. As shown in
Figure 2 (a), the solid arrows denote the causal relationships:
cause → effect. X → Y denotes the labeling process from the
given image, as an image is labeled for its content. An ideal
CC-ReID model only identifies X → Y , which is unbiased.
C → X indicates that different clothes result in diverse image
contents. C → Y implies that the identity can be inferred from
the clothes, depicting the presence of the spurious correlation
caused by data collection. Other interfering factors (e.g., scene,
camera, and viewpoint) usually cannot be used to infer identity,
and thus do not open backdoor paths. Therefore, they are not
considered confounders under the causal inference framework.
In summary, there are two relationships from X to Y : a
clothing shortcut X ← C → Y (also known as the backdoor
path) and a causal relation X → Y .

it is evident that the intervention modified P (C = c|X) into a
prior probability P (C = c), which is equal to making clothes
C statistically independent of images X. In this case, the
backdoor path X ← C → Y is removed. As a result, the
clothing shortcut that identifies persons based on their attire is
not applicable and the model will learn undisturbed to capture
clothes-invariant features.
IV. M ETHOD
Problem Formulation. For a CC-ReID dataset G =
Nimg
{(xi , yi , ci )}i=1
, the i-th data sample in G can be denoted as
a triplet (xi , yi , ci ), where xi , yi , ci denotes the image, identity label and the clothing label, respectively. The CC-ReID
task focuses on extracting clothes-invariant and discriminative
image features for inference.
Method Overview. The main pipeline of our Causal
Clothes-Invariant Learning (CCIL) is shown in Figure 3.
1 The detailed proof is provided in the Supplementary Material.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -l 3 'Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

8119

Camera-Proxy Enhanced Identity-Recalibration
Learning for Unsupervised Visible-Infrared
Person Re-Identification
Run-Sen Xia , Xue-Yan Wang, Si-Bao Chen , Member, IEEE, Jin Tang , and Bin Luo , Senior Member, IEEE

Abstract— Visible-Infrared person Re-Identification (VI-ReID)
involves querying images of the same person across visible
and infrared modalities. To minimize annotation costs, Unsupervised Visible-Infrared person Re-Identification (UVI-ReID)
using pseudo-label contrastive learning has emerged. Traditional
UVI-ReID approaches often neglected camera domain information and relied on inadequate update strategies during training,
only using cosine distance for testing, which led to incorrect mapping of cross-modal relationships. To address these issues, we propose Camera-proxy Enhanced Identity-recalibration Learning
(CEIL). It consists of two main stages: first, it employs intramodal contrastive learning in conjunction with the camera-proxy,
updates the memory bank using our innovative Difficulty-aware
Cluster-based Memory Updating (DCMU) strategy, and applies
Camera Domain-driven Local correlation (CDL) Loss to enhance
the learning process. Then utilizes cross-modal contrastive learning, featuring our Proxy-enhanced Cross-modal Mapping (PCM)
module, to recalibrate the identity relationships between different
modalities. Graph network-based Camera constraint adjustment
Re-ranking (GCR) method is adopted during test, utilizing
camera domain information to recalibrate the correspondence
between identities. Extensive experiments have demonstrated
that CEIL achieving state-of-the-art performance on the SYSUMM01, RegDB, and LLCM datasets and the GCR, as a general
unsupervised re-ranking method, can further enhance performance of model on these datasets. The code will be released at
https://github.com/maybeextra/CEIL.
Index Terms— Cross-modal contrastive learning,
re-identification, camera-proxy, identity-recalibration.

person

I. I NTRODUCTION
N RECENT years, the demand for intelligent video
surveillance has driven progress in person Re-Identification
(ReID) [1], [2], [3], [4], which matches pedestrians across
non-overlapping camera views. While supervised and unsupervised methods [5], [6], [7], [8] excel in well-lit environments,
they falter in low-light or nighttime conditions due to limited
information. Efforts such as optimizing loss functions [9], [10],

I

Received 10 December 2024; revised 17 February 2025; accepted 4 March
2025. Date of publication 6 March 2025; date of current version 6 August
2025. This work was supported in part by the NSFC Key Project of Joint
Fund for Enterprise Innovation and Development under Grant U20B2068 and
Grant U24A20342 and in part by the National Natural Science Foundation
of China under Grant 61976004. This article was recommended by Associate
Editor G. Xu. (Corresponding author: Si-Bao Chen.)
The authors are with the MOE Key Laboratory of ICSP, the IMIS Laboratory
of Anhui, Anhui Provincial Key Laboratory of Multimodal Cognitive Computation, and the Zenmorn-AHU AI Joint Laboratory, School of Computer
Science and Technology, Anhui University, Hefei 230601, China (e-mail:
2966461966@qq.com; 518049@qq.com; sbchen@ahu.edu.cn; tj@ahu.edu.cn;
luobin@ahu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3548939

Fig. 1. Illustration of the dilemma. Images of the same identification exhibit
significant differences across multiple modalities, and within each modality,
variations arise due to different camera positions and times. This ultimately
leads to inconsistent clustering and ambiguous cross-modal label association.

[11] have been made, but single-modal approaches remain
inadequate, necessitating more universal solutions.
With the development of technology, recently deployed
cameras can seamlessly transition to far or near-infrared
mode during nighttime operations [12]. Taking advantage
of this capability, alongside the development of extensive
cross-modal datasets [13], [14], [15] annotated by humans,
various techniques [5], [16], [17], [18], [19], [20], [21],
[22], [23] have been proposed for Visible-Infrared person
Re-identification (VI-ReID). In contrast to the annotation of
single-modal ReID datasets, annotating cross-modal datasets
demands considerably more resources. It is undeniable that
the current high-performing VI-ReID methods heavily rely
on annotated training datasets. In response to this challenge,
Unsupervised Visible-Infrared person Re-Identification (UVIReID) methods [24] offers a promising solution.
Existing UVI-ReID methods focus on unsupervised contrastive learning using pseudo-labels generated by clustering. However, this learning strategy still has flaws. The
visible-infrared image space is characterized by substantial
modal gaps and pedestrian images within the same modality
exist high intra-Identification variance and inter-Identification
similarity. This resulted in suboptimal modal cohesive classes
and incorrect cross modal mappings as shown in Fig.1.
To tackle these challenges, it’s important to reassess the
UVI-ReID process. Variations in image sources, even within
the same modality, can complicate identification. Fortunately,

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

8120

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Fig. 2. Illustration of using camera information to segment the generation
of pseudo-labels. The four sections delineated by class boundaries represent
the distinct categories generated by DBSCAN clustering. On this foundation,
camera information is used to further divide and differentiate the camera
proxies. In the diagram, different shapes represent different cameras, and each
intra-class camera domain encapsulates a distinct camera proxy.

acquiring camera domain information is easier than obtaining true labels. It allows clusters to be subdivided into
camera proxies, as shown in Fig.2. Re-ID methods like
ICE [25] use this approach, and DCCL [26] applies it to
UVI-ReID with camera-aware and class-based contrastive
learning. However, introducing too many training objectives
when the foundation is unstable can be burdensome. Crossmodal identity learning faces similar challenges, as successful
cross-modal clustering relies on reliable intra-modal clustering first. Therefore, we propose a two-stage Camera-proxy
Enhanced Identity-recalibration Learning (CEIL) framework,
depicted in Fig.3.
To effectively integrate camera proxies throughout the training process while minimizing interference from excessive
information, we employ a two-stage approach. In the first
stage, we conduct contrastive learning exclusively at the
camera proxy-level. We introduce a Camera Domain-driven
Local correlation (CDL) Loss. In single-modal scenarios,
camera variations mainly stem from lighting and position
changes. However, in UVI-ReID, visible and infrared cameras
exhibit distinct sensor artifacts like noise patterns and thermal drift. It’s crucial to reduce camera domain interference
and compute losses separately for each domain. Leveraging
camera information not only addresses single-modal issues
but also provides a solid foundation for cross-modal learning.
Additionally, local loss computation slows training, mitigating
negative impacts from clustering errors. In the second stage,
we incorporate a class-level loss based on pseudo-labels.
Drawing inspiration from PGM [27], we develop a Proxyenhanced Cross-modal Mapping (PCM) module. This module
employs multi-level contrastive learning to bridge modality
gaps, according to dual consistency [26] in cross-modal correspondences between camera proxies and classes, to help
correct errors and enhance performance. In single-modal
scenarios, camera information mainly aids cross-camera feature alignment. For UVI-ReID tasks, the primary challenge
becomes cross-modal matching, as category-level matching
alone is often unstable. By incorporating camera information

as auxiliary features, a camera proxy hierarchy can be established beyond the category level, enabling the construction
of cross-modal matching graphs both within proxies and
categories. This approach significantly differs from traditional
single-modal uses of camera information. The multi-level
mapping consistency mechanism [26] across categories and
camera proxies effectively mitigates modal discrepancies,
improving the accuracy and robustness of cross-modal matching. Traditional methods such as those described in AGW [5],
which use momentum updates with memory banks for contrastive learning, assume spherical sample distributions. This
assumption conflicts with reality. In fact, clusters generated by
clustering algorithms often exhibit a ring-shaped distribution.
While RTMem [28] introduced real-time memory updates,
it overlooked challenging samples. To address this, our
Difficulty-aware Cluster-based Memory Updating (DCMU)
strategy dynamically adjusts based on the relationship between
samples and proxy memory, ensuring more effective learning. Upon completing the two-stage training, re-ranking can
significantly improve ReID performance, as shown in studies
like [29] and [30]. However, its reliance on CPU computation
can be time-consuming, reducing its application in recent
research. Although GCN [31] enhances speed, it lacks the
integration of prior camera information, often prioritizing
speed over accuracy. To overcome these limitations, we propose a Graph network-based Camera constraint adjustment
Re-ranking (GCR) method. This method combines camera
information with features and utilizes GPU-accelerated graph
convolution, significantly reducing computation time while
maintaining strong performance.
Our main contributions can be summarized as follows:
• Proposed the CDL Loss, using local negative samples
within the camera domain to improve domain consistency,
generalization, and discriminative ability.
• Introduced the PCM module to align multimodal clusters
at class and camera levels, improving robustness.
• Developed the DCMU strategy, updating the memory
bank based on data difficulty to enhance performance.
• Created the GCR method to integrate camera information and perform graph propagation, thereby significantly
reducing time waste and achieving a leap in performance.
Extensive experiments on the SYSU-MM01, RegDB, and
LLCM datasets demonstrate that the CEIL achieves state-ofthe-art (SOTA) performance compared to recent methods.
II. R ELATED W ORK
A. Contrastive Learning in Person Re-Identification
Contrastive Learning serves as a technical basis for pseudolabel-based unsupervised person ReID, attracting considerable
attention for its critical role. This approach learns discriminative feature representations through contrastive optimization in
the feature space. It constructs positive and negative sample
pairs, simultaneously minimizing the distance between positive
pairs while maximizing the separation of negative pairs.
Advancements in contrastive learning have significantly
propelled the development of unsupervised person ReID. The
seminal work of SimCLR [32] established a breakthrough

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

XIA et al.: CAMERA-PROXY ENHANCED IDENTITY-RECALIBRATION LEARNING FOR UVI-ReID

8121

Fig. 3. The overall architecture of the proposed CEIL. Includes a difficulty-aware cluster-based memory update (DCMU) strategy, an proxy-enhanced
cross-modal mapping (PCM) module, and a camera domain-driven local correlation (CDL) loss. At the start of each epoch, clustering is initialized, and
clusters are divided by camera source for contrastive learning. During the first training stage, CDL loss is applied within the same modality based on camera
agents, and MMD loss is applied between modalities. In the second stage, not only CDL loss is applied within modalities, but also InfoNCE loss is applied
based on clusters, and the PCM is used for cross-modal feature learning, with an equivalent loss calculated in the cross modality for each sample.

framework by integrating data augmentation, non-linear
projection heads, and large-batch training to learn visual
representations without supervision, with its core principle
being the maximization of similarity between augmented
views of the same images. Parallelly, MoCo [33] addressed
the limitations of negative samples through its innovative
momentum encoder and dynamic dictionary, thereby enhancing feature consistency. These foundational contributions
paved the way for modern pseudo-label-based unsupervised
person ReID methodologies. Building upon these principles,
SSL [34] pioneered the application of contrastive learning
in unsupervised person re-identification by optimizing feature
similarity through positive and negative sample pairs. This
approach was further refined by SpCL [35], which incorporated self-paced learning and a hybrid memory mechanism,
marking substantial progress in the field. SSFT [14] extended
these concepts by proposing a novel framework for UVIReID, achieving remarkable improvements in cross-modality
matching performance through shared-specific feature learning
and self-paced contrastive learning, and established the theoretical and methodological foundation for our proposed CEIL
framework.
B. Visible-Infrared Person Re-Identification
1) Supervised Methods: The issue of visible-infrared crossmodal pedestrian re-identification was initially introduced by
Wu et al. [36]. Ye et al. [5] later introduced the comprehensive SYSU-MM01 dataset, establishing a standardized
benchmark with the AGW method and proposing a novel
evaluation metric called mINP. Supervised VI-ReID [17],
[37], [38], [39] has focused on bridging the gap between
the visible and infrared modalities and learning robust representations to effectively handle these modal differences.
GECNet [40] proposes a grayscale enhancement coloring network that compensates for the modality gap by preserving the
information-rich structures contained in color images. Methods
such as MAUM [17] and MSCLNet [18] have concentrated

on learning modality-specific features and integrating them to
reduce noise and enhance identity discrimination. PMT [20]
and CycleTrans [21], introduced auxiliary modalities and
lightweight modules to capture richer semantic information.
Further advancements such as PartMix [19] and DMA [22]
have leveraged data augmentation techniques and optical characteristics to mitigate cross-modal differences and preserve
identity features. HOS-Net [23] has focused on learning a
common feature space by using aligned features from different
modalities. These advancements not only improved the performance of supervised VI-ReID but also provided valuable
insights for developing unsupervised methods in this domain.
2) Unsupervised Methods: UVI-ReID remained an area of
extensive research, focusing on bridging the gap between
visible and infrared modalities through innovative approaches.
OTLA [41] utilized optimal transport for pseudo label assignment across modalities. ADCA [24] enhanced inter-modality
representation learning via a dual-contrastive framework.
Recent works in unsupervised visible-infrared person reidentification have focused on bridging the modality gap
and learning modality-invariant features through various
cross-modal alignment and clustering strategies. Methods like
CCLNet [42] and Sun et al. [43] leveraged CLIP to generate cluster-aware prompts and extract deep text information,
facilitating the creation of text-image alignment modules
that bridged the visible-infrared gap. Approaches such as
PGM [27], MMM [44], and RPNR [45] established reliable
cross-modality correspondences through progressive graph
matching, cross-modality clustering, and optimal transport
prototype matching, respectively. Several works, including
CHCR [46] and MIMR [47], focused on refining or eliminating noisy pseudo-labels by analyzing clustering results or
refining labels in ambiguous samples. Mutual learning and
modality-specific information extraction were addressed by
methods like CMAM [48] and PCAL [49], which incorporated cross-modal asymmetric mutual learning algorithms and
modality-invariant contrastive learning to reduce modal differ-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.


codex
中间三篇给了两个更直接的信号：遮挡论文已经在用语言把“人、可见属性、遮挡物”拆开；换衣论文把衣服捷径上升成因果问题。我们要避开的就是“再做一个提示词、再做一个去衣服分支”，可以借它们的问题表述，但机制要换到我们自己的姿态、三维几何或航拍地面资产上。
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
exec
/bin/zsh -lc "pdftotext -l 3 'CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CCFL: Customized Client Federated Learning for
Unsupervised Person Re-identification
YI ZHENG, School of Information and Electronics Engineering, Jiangsu Vocational Institute of
Architectural Technology, Xuzhou, China
YONG ZHOU, School of Computer Science and Technology, China University of Mining and Technology,
Xuzhou, China
FAYAO LIU, Agency for Science, Technology and Research, Singapore, Singapore
JIAQI ZHAO, HANCHENG ZHU, and WENLIANG DU, School of Computer Science and
Technology, China University of Mining and Technology, Xuzhou, China
Federated learning-based person re-identification (Re-ID) aims to address the issue of data silos in surveillance systems caused by increasingly stringent regulations on sensitive data. However, due to differences
in data collection locations, times, and scales, severe non-independent and identically distributed (non-IID)
characteristics exist across different Re-ID datasets. Existing federated learning-based Re-ID methods often
adopt a unified model structure, which prevents the model from adapting well to diverse data environments,
thereby significantly degrading the overall Re-ID performance. To address the challenges of training neural
networks on non-IID data across different datasets, we propose a customizable federated learning framework.
First, customizable clients allow each organization to freely select suitable neural network training methods
and model architectures based on local data scales and prior knowledge, thus improving training outcomes.
Second, since traditional federated learning frameworks cannot achieve knowledge fusion through parameter
exchange between models with different architectures, we introduce an independent model, referred to as
the interaction model, specifically designed for knowledge exchange among clients. The interaction model
learns parameters (knowledge) from local models on each client through distillation learning. Subsequently,
the interaction model is uploaded to the server, where it undergoes parameter fusion (knowledge exchange)
with interaction models from other clients. Finally, the interaction model, enriched with knowledge from
other clients, guides local model training through knowledge distillation. It is worth noting that selecting
a lightweight interaction model, while potentially impacting Re-ID performance, can significantly reduce
communication costs between the server and clients.
This work was done by Y. Zheng while visiting Institute for Infocomm Research, A*STAR, Singapore.
This work was supported by the National Natural Science Foundation of China (Grant No. 62272461), and by the China
Scholarship Council (Grant No. 202206420034) which awarded Y. Zheng a scholarship for 1 year of study abroad at the
Agency for Science, Technology and Research.
Authors’ Contact Information: Yi Zheng, School of Information and Electronics Engineering, Jiangsu Vocational Institute
of Architectural Technology, Xuzhou, China; e-mail: yizheng@jsjzi.edu.cn; Yong Zhou (corresponding author), School of
Computer Science and Technology, China University of Mining and Technology, Xuzhou, China; e-mail: yzhou@cumt.edu.cn;
Fayao Liu, Agency for Science, Technology and Research, Singapore, Singapore; e-mail: Liu_Fayao@i2r.a-star.edu.sg; Jiaqi
Zhao, School of Computer Science and Technology, China University of Mining and Technology, Xuzhou, China; e-mail:
jiaqizhao@cumt.edu.cn; Hancheng Zhu, School of Computer Science and Technology, China University of Mining and
Technology, Xuzhou, China; e-mail: zhuhancheng@cumt.edu.cn; Wenliang Du, School of Computer Science and Technology,
China University of Mining and Technology, Xuzhou, China; e-mail: wldu@cumt.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/8-ART225
https://doi.org/10.1145/3735134
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:2

Y. Zheng et al.

CCS Concepts: • Computing methodologies → Visual content-based indexing and retrieval; Object
identification; Cooperation and coordination;
Additional Key Words and Phrases: person re-identification, federated learning, non-IID data, knowledge
transfer, knowledge distillation
ACM Reference format:
Yi Zheng, Yong Zhou, Fayao Liu, Jiaqi Zhao, Hancheng Zhu, and Wenliang Du. 2025. CCFL: Customized Client
Federated Learning for Unsupervised Person Re-identification. ACM Trans. Multimedia Comput. Commun.
Appl. 21, 8, Article 225 (August 2025), 21 pages.
https://doi.org/10.1145/3735134

1

Introduction

Person Re-identification (Re-ID) targets the identification of specific pedestrians in extensive
surveillance data [54]. Since Re-ID data is derived from video surveillance, it involves personal and
organizational privacy information (e.g., clothing, appearance, travel patterns), which imposes strict
requirements on data security. Moreover, due to variations in shooting times, locations, devices, and
angles, different Re-ID datasets exhibit significant disparities, leading to the Non-independent and
Identically Distributed (non-IID) problem, which is prevalent in cross-domain Re-ID research
[28, 37, 39].
In recent years, academic research has increasingly focused on minimizing the overexposure of
person Re-ID data due to privacy concerns. Many Re-ID datasets now require researchers to declare
their academic research purposes and sign relevant usage agreements. Simultaneously, researchers
have begun to shift away from supervised learning paradigms, exploring semi-supervised and
unsupervised learning approaches for Re-ID. However, methods based on semi-supervised learning
paradigms (e.g., domain adaptation and domain generalization [6, 33, 34]) remain inherently
dependent on labeled data (annotated source domain data). Moreover, due to the non-IID nature of
Re-ID datasets, unsupervised learning methods [52, 53] often exhibit inconsistent performance,
as they lack tailored training strategies and model architectures for different datasets. To achieve
higher performance in Re-ID models, researchers often adopt centralized training on multiple
Re-ID datasets to increase data scale and learn more generalizable features. However, as mentioned
earlier, due to the security requirements of Re-ID data, collecting sensitive surveillance information
from multiple organizations is generally unacceptable. In this context, the introduction of federated
learning [32] provides a solution to cross-domain learning in Re-ID data.
The core of federated learning algorithms is to enable independent training of model parameters
across multiple clients with private data sources. After a certain number of training iterations, each
client uploads its locally trained model parameters to a centralized server, instead of sharing raw
data across the network. The server then aggregates the parameters from multiple models using
model aggregation methods to obtain a fused model. Subsequently, the fused model parameters
are distributed to each client for the next round of training. This distributed learning approach
allows multiple clients to collaboratively train a shared network model using different data, while
ensuring that the original data remain stored locally, thus safeguarding the security of local data.
Currently, federated learning algorithms have been widely applied in handling data with higher
security sensitivity, such as financial and medical data [1, 14, 46].
In response to the growing concerns about data security in person Re-ID, researchers have
explored the potential of utilizing federated learning frameworks for Re-ID tasks [45, 49, 50].
As an experimental study, Zhuang et al. [56] conducted supervised training across nine largescale datasets to develop a robust Re-ID model. However, supervised training strategies rely
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:3

heavily on large amounts of labeled data, and the associated labeling costs limit their applicability.
Moreover, the labeling process itself could be perceived as a threat to the security of Re-ID data.
To address the limitations of manual data handling, they further proposed a federated Re-ID
method based on unsupervised learning, called FedUReID [55], which reduces dependency on data
labels. Nevertheless, these methods did not consider the statistical heterogeneity between different
datasets, which ultimately resulted in lower Re-ID accuracy.
Moreover, although some studies have attempted to mitigate the non-IID problem by designing
model architectures, parameters, and functional modules specifically tailored for handling heterogeneous data [38], these approaches still rely on a unified training strategy. First, according to the
performance summary table of unsupervised person Re-ID methods on various datasets, collected
and organized by the web site “Paper with Code,” different person Re-ID datasets often achieve
optimal performance with different methods. Second, different clients may possess varying levels of
computational resources, and large-scale computing devices, such as those used in TransReID-SSL
[30], are not commonly available.
To address the aforementioned issues, we propose a customizable federated unsupervised person
Re-ID framework. In this work, we utilize two unsupervised Re-ID training strategies based on
the DBSCAN clustering algorithm and the hierarchical clustering algorithm. Additionally, when
training large neural network models on small-scale datasets, these models, due to their large
number of parameters and finer feature descriptions, are more prone to overfitting. Therefore,
based on the number of training samples, we employ two representative models, ResNet-50 and
ResNet-34. It is worth noting that among existing clustering algorithms, DBSCAN and hierarchical
clustering are the most widely used and facilitate finding comparative methods, while requiring
only minimal prior data parameters. ResNet-50 and ResNet-34 are merely examples of complex and
simple models used to demonstrate the compatibility and scalability of the proposed method; the
choice of models is not limited to these in practical applications.
Due to the use of different model architectures, the model parameter averaging method commonly
used in federated learning, such as FedAvg [32], cannot be directly applied to the framework
proposed in this article. Therefore, we introduce an additional network model, referred to as the
interactive model, specifically designed to facilitate cross-client knowledge transfer. Specifically,
the interactive model is trained under the guidance of local models from each client to acquire
network parameters that closely approximate the performance of the local models. The interactive
model then conducts cross-client knowledge exchange using model parameter averaging methods.
Finally, the updated parameters from the interactive model are used to guide the training of the
local models, enabling them to learn more generalized and robust pedestrian features. Additionally,
if a slight sacrifice in Re-ID performance is acceptable, selecting a smaller-scale interactive model
can reduce the amount of data transmitted during cross-client communication, thereby improving
interaction efficiency.
In summary, the main contributions of this article are as follows:
— We propose a novel customizable client federated learning method that allows each client to
adopt independent training strategies and model structures.
— We introduce an innovative client information exchange method, where knowledge exchange
between neural network models with different structures is achieved through an additional
interactive model.
— By accepting a tradeoff in Re-ID performance, selecting a smaller model as the interactive
model reduces communication costs between clients and the server.
The remainder of this article is organized as follows. Section 2 provides a comprehensive review
of related work on unsupervised person Re-ID and federated learning. In Section 3, we analyze the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-identification Models.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209843

CFPER: Coarse-to-Fine Part-Experts Retrieval for
Efficient Person Re-identification
1st Shiyu Wang

2nd Mingming Lu⋆

School of Computer Science and Engineering
Central South University
ChangSha, China
224711075@csu.edu.cn

School of Computer Science and Engineering
Central South University
ChangSha, China
mingminglu@csu.edu.cn

Abstract—Most existing person re-identification (ReID) methods focus on improving retrieval accuracy by refining features,
which fails to balance accuracy with inference efficiency. We
observe that query difficulty varies: global features suffice for
simple cases, while fine-grained part features are required for
challenging cases, such as occlusion. However, current methods
typically use the same feature extraction network for all queries,
which may limit accuracy on difficult queries or waste computational resources on easier ones. To address this, we propose
a two-stage Coarse-to-Fine Dynamic Retrieval mechanism that
adaptively allocates resources based on query difficulty. For
“easy” queries, only global features are used in the coarse
stage, and inference terminates early. For “hard” queries, part
features are extracted in the fine stage for detailed matching.
To further reduce computational costs, we introduce Mixture
of Experts for part feature extraction, where a router assigns
patches to part experts using topology annotations, and only activates body-relevant experts, enabling accurate part identification
with significant computation reduction. Extensive experiments
demonstrate that our method achieves competitive performance
while significantly reducing computational costs compared to
state-of-the-art methods.
Index Terms—Transformer-based Person Re-identification,
Computational Efficiency, Early-Exit, Mixture of Experts.

I. I NTRODUCTION
Person re-identification (ReID) aims to retrieve a specific
person from a large database of person images captured by
diverse non-overlapping cameras, which is widely applied in
many domains, such as criminal investigation and smart city
[1]. Most existing person ReID methods primarily focus on
improving retrieval accuracy, often overlooking the importance
of computational efficiency. Specifically, these methods [1]–
[7] process all queries with the same network and use uniform
features for retrieval, without accounting for the varying
difficulty of queries, which may limit accuracy on challenging
queries or consume needless computational resources on easier
ones. In real-world applications, the retrieval difficulty of different queries varies: some query images exhibit distinct discriminative features, and global features alone are sufficient for
accurate retrieval. For these “easy” queries, excessive reliance
on fine-grained part features may lead to false matches, as
different pedestrians may share similar body part appearances.
In contrast, “hard” queries, such as those affected by occlusion,
pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.

TABLE I
P ERFORMANCE AND FLOP S OF T RANS R E ID WITH D IFFERENT
R ETRIEVAL F EATURES DURING I NFERENCE . “G” D ENOTES THE C ASE
USING ONLY THE G LOBAL F EATURE . “G+P” D ENOTES THE C ASE USING
THE C ONCATENATED F EATURE OF G LOBAL F EATURE AND F OUR PART
F EATURES .

Dataset
Feature
mAP (%)
Rank-1 (%)
FLOPs (G)

Market-1501
G
G+P
87.1
88.2
94.6
95.0
11.35
12.29

DukeMTMC
G
G+P
79.6
80.6
89.0
89.6
11.35
12.29

Occluded-Duke
G
G+P
53.8
55.7
61.6
64.2
11.35
12.29

To verify this, we conducted experiments on three datasets
with increasing retrieval difficulty: Market-1501 (holistic) [8],
DukeMTMC (holistic) [9], and Occluded-DukeMTMC (occluded) [2]. We tested two TransReID [4] configurations: (1)
global features, and (2) concatenated global and four part
features, and report mAP, Rank-1 and FLOPs in Table I. The
results show that introducing part features increases computational cost by 1.1× in terms of FLOPs, but provides only minor
improvements in mAP (1.1% and 1.0%) and Rank-1 (0.4% and
0.6%) on two holistic datasets, while resulting in significant
improvements (1.9% mAP, 3.1% Rank-1) on the more challenging occluded dataset. This indicates that global features
are generally sufficient to support accurate matching for “easy”
queries with distinctive visual features, while fine-grained part
features are more beneficial for challenging queries. Inspired
by the above observations, we propose a two-stage Coarse-toFine Dynamic Retrieval (CFDR) mechanism that adaptively
allocates computational resources based on the difficulty of
queries to balance performance and efficiency. Specifically, an
early-exit threshold is computed to evaluate query difficulty.
For “easy” queries, only global features are used for fast
retrieval in the coarse stage, and inference terminates early to
avoid unnecessary computational resource consumption. Only
“hard” queries proceed to the fine stage, where fine-grained
part features are extracted for more precise matching.
For part feature extraction in the fine inference stage of
CFDR, existing methods face several limitations. Some methods [5] rely on external models (e.g., pose estimation and
human parsing) to extract part features, introducing additional
computational costs. Other methods [4], [10] divide spatially
adjacent patches or pixels into fixed-size groups to learn

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
embeddings Z are extracted. An early-exit decision is made based on the threshold ⌈N α⌉: if met, the query is considered “easy”, only global feature is used
for retrieval and inference terminates. Otherwise, the query is considered “hard” and proceeds to the fine stage to extract part features using the Part-aware
Mixture of Experts (PMoE). PMoE includes a Patch-to-Part Router (PPR) and M Part Experts (PE). Each patch is routed to the corresponding body part via
PPR, and the fine-grained part feature ei of each body part is learned by the dedicated PE. Here, PWAP refers to Probability Weighted Average Pooling.

part features but ignore human topology priors, leading to
inaccurate part detection. Therefore, to further reduce computational costs and improve part identification accuracy, we
introduce Mixture of Experts and human topology priors to
propose a Part-aware Mixture of Experts (PMoE). The PMoE
consists of a Patch-to-Part Router (PPR) and a set of Part
Experts (PE). The PPR routes patches to corresponding body
parts based on identity labels and human topology annotations, enabling accurate part identification without introducing
additional computational costs. Each body part is assigned
to a dedicated Part Expert for fine-grained feature learning.
And during inference, PMoE only activates body-relevant
experts based on binary visibility routing weights to reduce
computational costs without hurting the performance.
Finally, we combine CFDR and PMoE to construct our
Coarse-to-Fine Part-Experts Retrieval model (CFPER).
The main contributions of our work can be summarized as
follows:
• We propose a novel Coarse-to-Fine dynamic retrieval
mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
on the query difficulty.
• We innovatively apply MoE in part feature extraction,
which uses human topology annotations to guide the
router for accurate part identification without extra computational costs, and only body-relevant experts are ac-

tivated during inference, significantly reducing inference
costs without sacrificing performance.
• The proposed CFPER achieves competitive performance
compared to SOTA methods on both holistic and occluded person ReID datasets, while also improving inference efficiency.
II. M ETHODOLOGY
In this section, we introduce the proposed Coarse-to-Fine
Part-Experts Retrieval (CFPER) in detail. An overview of
CFPER is shown in Fig. 1.
A. Coarse Inference Stage
Coarse Feature Extractor. We use the pre-trained Vision
Transformer (ViT) [11] as our feature extractor. Given a person
image X, the output of the encoder can be divided into two
parts: a global feature G ∈ R1×D and N patch embeddings
Z = [z1 , . . . , zN ] ∈ RN ×D .
Early-exit strategy. In the coarse inference stage, we
introduce an early-exit strategy to balance performance and
efficiency. Traditional early-exit strategies [12] are typically
used in classification tasks, relying on classifier scores. However, in person ReID, the training and testing identities are
inconsistent and the number of testing identities is unknown,
making the classifier score-based exit metric unsuitable. To
address this, we propose an early-exit strategy for person

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 2. Illustration of the early-exit strategy based on the first-order differences
of similarity scores.

ReID, which determines whether to exit early by evaluating
the query difficulty based on the first-order differences of
similarity scores.
To assess query difficulty, our intuition is to count the
number of body-related patches. As shown in Fig. 2, we
first compute the cosine similarity between global feature
G and N patch embeddings Z, obtaining similarity scores
S = [s1 , . . . , sN ]. The scores S are sorted in descending
D
order to obtain S D = [sD
1 , . . . , sN ]. We then compute the
first-order differences D = [D1 , . . . , DN −1 ], where Di =
D
sD
i − si+1 . We assume that body and background/occlusion
features are separate classes in the feature space, so there
should be a noticeable feature transition when a body feature
shifts to a background/occlusion feature. Therefore, we select
the maximum value in D and use its corresponding index
η = arg maxi (D) as the split point between the body and
background/occlusion regions. Patches with the top-η similarity scores are considered as body regions, while the rest
are considered as background/occlusion regions. Finally we
introduce an early-exit threshold α ∈ [0, 1]. If η ≥ ⌈N α⌉, the
query is considered “easy” with sufficient visible body regions
for high-confidence retrieval. In this case, only global features
G are used for retrieval, and the whole inference terminates.
Otherwise, the query is considered “hard” and proceeds to the
fine inference stage to extract fine-grained part features.
Supervision Loss of Coarse Inference Stage. We adopt
cross-entropy loss as ID loss to supervise the learning of
encoder:
LCOARSE = LID (G),
(1)
B. Fine Inference Stage
In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
retrieval. PMoE consists of a Patch-to-Part Router and a set
of Part Experts.
1) Patch-to-Part Router. The PPR module learns to route
patches embeddings Z to M+1 parts C = [c0 , c1 , . . . , cM ],
where c0 represents the background, and [c1 , . . . , cM ] corresponds to M body parts. PPR adopts a standard MoE router
structure, consisting of a fully-connected layer with parameters
θ ∈ R(M +1)×D followed by a softmax layer, which produces
the probabilities P ∈ RN ×(M +1) of each patch belonging to
the background and M body parts:
P = {pi ∈ RN | i = 0, 1, ..., M } = Sof tmax(Z · θT ), (2)

where pi represents the probabilities of patches embeddings
Z belonging to part ci .
Next, the router needs to assign patches to each expert based
on P. Traditional MoE patch-level routers [13] commonly
use the Top-k mechanism, where each expert is assigned k
patches (k ≪ N ). However, this fixed Top-k mechanism
is unsuitable for person ReID due to the varying sizes of
body parts. If k is too small, larger body regions (e.g., the
torso) may not be fully covered, leading to information loss.
Conversely, if k is too large, smaller body parts (e.g., the
head) may be polluted by surrounding background/occlusion
patches, introducing noise. To address this, our PPR employs
a soft routing mechanism. First, we aggregate M body parts
probabilities [p1 , . . . , pM ] to compute a foreground probability
pf ∈ RN : pjf = sum(pj1 , . . . , pjM ), where pji represents the
probability of patch embedding zj belonging to body part
ci . We then perform Probability Weighted Average Pooling
(PWAP) on patch embeddings Z to obtain a foreground
feature rf , a background feature r0 and M body part features
[r1 , . . . , rM ]:
PN
j
j=1 zj · pi
(3)
ri = PN j , ∀i ∈ {f, 0, 1, ..., M }
j=1 pi
where zj denotes the j-th patch embedding from the coarse
inference stage.
Supervision Loss of PPR. We introduce human topology
priors in the form of coarse body part labels. For each patch
j ∈ RN , its part label yj ∈ {0, 1, ..., M }. The cross-entropy
loss Lh with label smoothing is computed as:
Lh = −

M
X N
X

qi · log(pji ),

i=0 j=1

with qi =

(
1 − B−1
B ε
ε
B

(4)
if yj = i
otherwise

where B is batch size, ε is label smoothing regularization rate.
Additionally, we propose a Push Loss Lpush to separate the
body regions from background/occlusion noise:
E
D
t
t
B
r
,
r
X
0 f
1
,
(5)
Lpush =
B t=1 ||r0t || · ||rft ||
t
where r0|f
is the background/foreground feature of the t-th
image in a batch.
Finally, we follow the Global-identity Local-triplet (GiLt)
[7] strategy to supervise the training of PPR:

LP P R = λh Lh + Lpush + Lce (rf ) + Lce (rc )
+ Lpart
tri (r1 , . . . , rM ),

(6)

where rc = Concat(r1 , . . . , rM ), Lce is cross-entrophy loss,
Lpart
tri is part-averaged triplet loss [7].
2) Part Experts. Although Transformer excels at capturing
global information, it is less effective at capturing fine-grained
image details. To compensate for this limitation, we assign a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.


 succeeded in 0ms:
CCUP: A Controllable Synthetic Data Generation
Pipeline for Pretraining Cloth-Changing Person
Re-Identification Models

arXiv:2410.13567v3 [cs.CV] 30 Mar 2025

Yujian Zhao1 , Chengru Wu2 , Yinong Xu2 , Xuanzheng Du2 , Ruiyu Li2 , Guanglin Niu1*
1
School of Artificial Intelligence, Beihang University
2
Shen Yuan Honors College, Beihang University
{yjzhao1019, chengru wu, yinong xu, xuanzheng du, ruiyu li, beihangngl}@buaa.edu.cn

Abstract—Due to the high cost of constructing Cloth-changing
person reidentification (CC-ReID) data, the existing data-driven
models are hard to train efficiently on limited data, which
causes the issue of overfitting. To address this challenge, we
propose a low-cost and efficient pipeline specific to CC-ReID
tasks for generating controllable and high-quality synthetic data
simulating the surveillance scenarios. Particularly, we construct
a new self-annotated CC-ReID dataset named Cloth-Changing
Unreal Person (CCUP), containing 6,000 IDs, 1,179,976 images,
100 cameras, and 26.5 outfits per individual. Based on this largescale dataset, we introduce an effective and scalable pretrainfinetune framework for enhancing the generalization of the
traditional CC-ReID models. The extensive experimental results
demonstrate that our framework could improve the original
models such as two typical models TransReID and FIRe2 after
pretraining on CCUP and finetuning on a benchmark, and
outperform other state-of-the-art models. The dataset is available
at: https://github.com/yjzhao1019/CCUP.
Index Terms—Cloth-changing Person Re-identification, Lowcost Synthetic Dataset, Pretrain-finetune Framework

I. I NTRODUCTION
Person re-identification (ReID) aims to identify gallery
images containing persons with the same identity as the query
image in a cross-camera scenario. Furthermore, cloth-changing
person re-identification (CC-ReID) is a more challenging task
to identify the same person but with different clothes in realworld scenarios at large spatial and temporal scales.
In recent years, deep learning-based models [13]–[15] have
been widely used to learn the discriminative features of person
images for ReID and its extended task CC-ReID [16] .
However, there are two main challenges for the CC-ReID task.
Challenge 1: the high cost of sampling and labeling
real CC-ReID images limits the size of existing datasets,
causing low performance of training models due to the
lack of sufficient ground truth for supervision. Building
a ReID dataset requires a complex environmental setup of
places, devices and pedestrians, as well as manual labeling
without violating privacy (DukeMTMC-ReID has been retracted due to privacy concerns). In particular, the complexity
and costs of generating a CC-ReID dataset further increase
* Correspinding author. This work was supported by the National Natural
Science Foundation of China (No. 62376016).

significantly since it is difficult to capture images of the same
person wearing various outfits on a large spatial and temporal
scale. In contrast, synthetic datasets are emerging to reduce
costs and address privacy concerns. As shown in Tab. I, we
provide the statistic of identities (#IDs), images (#Images),
cameras (#Cam) and average outfits per identity (#avgClo)
of some typical CC-ReID and synthetic datasets. We could
observe that the whole size and especially #avgClo of all
the previous CC-ReID benchmark datasets such as PRCC,
LTCC and VC-Clothes are obviously limited, and Celeb-reID
and LaST are even created from celebrity street photography
and movies rather than real surveillance scenes. Besides, few
existing commonly-used synthetic datasets are not designed
for cloth-changing scenarios and therefore lack rich clothchanging ground truth. To address these issues, we propose
a controllable and low-cost pipeline for generating largescale synthetic data more suitable for CC-ReID tasks.
Challenge 2: cloth-irrelevant features are hard to be
extracted via the existing models straightly trained on
a limited CC-ReID dataset. Specific to the CC-ReID task,
the most pivotal purpose is to extract discriminative clothirrelevant features. Therefore, CAL [17] is proposed to extract
cloth-irrelevant features from original RGB images by penalizing the predictive power of the ReID model. AIM [18] is
proposed to analyze the impact of clothing on model inference
and eliminate clothing bias during training . Besides, various
auxiliary information such as gait [19], skeleton [20], and 3D
shape [21] could be exploited for supplementing more clothirrelevant features. However, all the previous CC-ReID models
suffer from extremely scarce training data, limiting their
performance specifically some advanced visual transformerbased models. To address this challenge, we employ a
scalable pretrain-finetune framework leveraging our largescale synthetic dataset to enhance the model performance
of CC-ReID.
Overall, the contributions of our work are three-fold:
• We construct a high-quality synthetic CC-ReID dataset
named CCUP with our low-cost and controllable data
generation pipeline, which is the first large-scale (over
1,000,000 images) dataset for the CC-ReID task.
• We exploit a scalable pretrain-finetune framework, which

TABLE I
S TATISTIC OF CC-R E ID AND SYNTHETIC DATASETS . H YPHENS REPRESENT THE NUMBER OF OUTFITS IS NOT PROVIDED .
Characteristic

#avgClo

3
12
17
15
29

Surveillance
Simulate
✓
✗
✗
✓
✓
✗
✓
✓

273,456
1,801,816
1,256,381
887,766

6
19
34
24

✗
✓
✓
✓

-

19.060
1,179,976

4
100

✓
✓

2.07
26.5

Dataset

#IDs

#Images

#Cam

PRCC [1]
Celeb-reID [2]
Celeb-reID-light [2]
LTCC [3]
DeepChange [4]
LaST [5]
NKUP [6]
NKUP+ [7]

221
1,052
590
152
1,124
10,862
107
361

33,698
34,186
10,842
17,138
178,407
228,000
9,738
40,217

Synthetic for ReID

PersonX [8]
RandPerson [9]
UnrealPerson [10]
ClonedPerson [11]

1,266
8,000
6,799
5,621

Synthetic for CC-ReID

VC-Clothes [12]
CCUP (ours)

512
6000

Real for CC-ReID

could improve the performance of CC-ReID via finetuning the same model pretrained on our large-scale
synthetic dataset CCUP.
• The extensive experimental results on multiple benchmark datasets including LTCC, VC-Clothes and NKUP
illustrate that our framework outperforms other state-ofthe-art baseline models significantly and consistently.
II. R ELATED W ORK
A. Cloth-changing person re-identification
Traditional ReID studies are highly dependent on clothing
appearance, which is not available to address unstrained clothing changes in real scenarios. Thus, there are an increasing
number of researches related to Cloth-Changing person ReIdentification (CC-ReID). Researchers first recognize the importance of data for the task and therefore construct many
benchmark datasets [1]–[7], [12]. Then, many innovative approaches have been proposed, the core idea of which is to
focus on cloth-irrelevant features. TransReID [16] propose
a pure transformer-based object ReID framework containing
novel modules such as jigsaw patch module and side information embeddings. FSAM [22] propose a two-stream framework
that learns discriminative body shape knowledge and transfers
it to complement the cloth-unrelated knowledge. Pos-Net [23]
reinforces the feature learning process by designing powerful complementary data augmentation strategies. IGCL [24]
proposed a novel framework where the human semantics are
leveraged and the identity is unchangeable to guide collaborative learning. IRM [25] propose a new instruct-ReID task
and a large-scale OmniReID benchmark as well as adaptive
triplet loss. Pixel sampling [26] propose a semantic-guided
approach that forces the model to automatically learn clothirrelevant signals by randomly changing clothes pixels. ISGAN [27] disentangles identity-related and unrelated features
from person images through an identity-shuffling technique
that exploits identification labels. FIRe2 [28] designs a finegrained feature mining module and presents a fine-grained

2.00
3.14
2.62
-

attribute recomposition module by recomposing image features
with different attributes. IFD [29] proposes an Identity-aware
Feature Decoupling learning framework to mine identityrelated features. However, these models have never been able
to identify cloth-changing person well due to the limited data.
B. Dataset synthesis
Techniques for dataset synthesis in CC-ReID domain are
mainly categorized into traditional graphics methods and deep
learning methods. Traditional graphics methods first generate
3D meshes, which are then imported into a physics engine to configure animations, add scenes, and simulate real
surveillance [8]–[12]. However, these methods still have some
shortcomings due to the inability of existing software to
generate high-resolution meshes on large scales. Deep learning
methods commonly use GAN [30] for continuous iteration to
synthesize dataset. CCPG [31] proposes a GAN based model
for clothing and pose transfer across identities to augment
images of more clothing variations. AFD-Net [32] proposes
a novel framework containing intra-class reconstruction and
inter-class adversary to disentangle the identity-related and
identity-unrelated features. What’s more, although diffusion
model [33] has received a lot of attention in the field of image
synthesis in recent years, there is no more mature work for
CC-ReID.
III. M ETHODOLOGY
A. Dataset generation
Considering the numerous advantages of synthetic data,
such as good controllability, a high degree of automation,
significantly lower costs compared to capturing real-world
data, and the excellent ability to simulate various real-world
environments, we prioritize synthetic data generation for CCReID tasks. Accordingly, we propose a low-cost, high-quality,
large-scale, and controllable data generation pipeline along
with a new dataset CCUP for pretraining CC-ReID models.
Specifically, the CCUP generation process consists of three
main procedures: (1) generating skeletal meshes of realistic

Fig. 1. Pipeline of our work. We first generate the skeletal mesh of person and provide a large number of clothing textures for cloth-changing. Then the
skeletal mesh is imported into the three scenarios to simulate surveillance and the person in the surveillance frames is detected. Finally, we construct a
large-scale dataset called CCUP containing 6,000 IDs, 100 cameras, and 1,1179,976 images. We pretrain the TransReID and FIRe2 two baseline models with
CCUP and finetune on PRCC, VC-Clothes, and NKUP benchmarks.

human characters, (2) simulating surveillance in diverse scenarios, and (3) producing self-labeled detection results for
cloth-changing pedestrian, as illustrated in Fig 1.
1) Generate person skeletal mesh: The skeletal mesh is the
basis for building synthetic data, which contains mesh data
and skeletal data. Mesh data stores vertex positions, normal
vectors and texture coordinates, etc. Skeletal data represents
the skeletal node hierarchy. Specifically, we construct the
person skeletal mesh using MakeHuman software, an opensource 3D human modeling software that helps users create
high-quality, vividly realistic human body models.
We modify Makehuman’s AssetDownloader plugin and employ it to collect assets from the MakeHuman community that
can be used to build 3D person models such as skin, hair,
clothes, etc. Besides, we modify MakeHuman’s MassProduce
plugin to create 6000 naked person skeletal meshes with
different physiological features. In this way, unique combinations of physiological parameters determine unique person
IDs, so we consider each naked person skeletal mesh to be
an ID and we get the set of IDs for the dataset ID =
{id1 , id2 , ..., idn }, where n is the number of identities in the
dataset. Benefiting from our modification of the AssetDownloader plugin, we collecte almost 3,000 clothing asset and
get different ensemble of outfits Clo = {clo1 , clo2 , ..., clom },
where m denotes the number of ensemble of outfits. In
turn, we can construct clothed skeleton meshes CSM =
{csm11 , csm12 , ..., csm1t1 , ..., csmn1 , csmn2 , ..., csmntn }:
csmij = {DressU p(idi , cloj ) | 1 ≤ i ≤ n, 1 ≤ j ≤ ti }, (1)
where ti (ti ≤ m) is the number of clothes for person idi and
DressU p(idi , cloj ) denotes wearing clothes cloj on person
idi . Inspired by Unrealperson [10] , we create more than
10,000 clothes textures, constructing the set of textures for the
https://github.com/makehumancommunity

dataset T = {t1 , t2 , ..., tr } for subsequent clothing changes,
where r denotes the number of textures of the dataset.
2) Simulate surveillance in scenarios: Unreal Engine is a
game engine that offers a wide range of rendering functions
and is popular in many applications such as game and movie
development. To generate the CC-ReID data, we employ
Unreal Engine (version 5.3.2) to simulate real surveillance
scenarios. We configure animations for the skeletal meshes
generated in section III-A1 so that they can walk around
the simulation scenarios. Then we replace the texture tk of
clothed skeleton meshed csm when they pass by different
cameras to simulate the more diverse cloth-changing, denoted
as RT (csm, t). Then, we choose three scenarios for simulation
in the epic games marketplace: an European alleyway, an office
building, and a park with 50, 25, and 25 cameras, respectively.
These set of three scenarios are denoted as S = {s1 , s2 , s3 }.
Particularly, we design travel routes of persons in three
scenarios and place cameras along the routes with diverse
viewpoints. Benefiting from a well-designed detection strategy,
the video of the person could be automatically captured if
this person passes under the camera. Then, the set of original
frames with automatically labeled person IDs, camera IDs and
cloth IDs are generated, denoted as OF = {of1 , ..., ofp }:
of = Sim(RT (csm, t), s), csm ∈ CSM, t ∈ T, s ∈ S, (2)
where Sim(RT (csm, t), s) denotes simulating the real
surveillance in scenarios s and obtaining the original frame for
RT (csm, t) of surveillance videos. To guarantee the quality
of labeled data, each image contains only one pedestrian by
adapting the starting time and the speed of this person.
3) Detect and label the bounding boxes: Based on the
surveillance video frames OF , we employ the advanced
https://www.unrealengine.com/


codex
我只按要求读了每篇前三页，下面判断只基于摘要和引言。

**逐篇反推**

1. `Bidirectional modality information interaction`
创新类型：新机制为主，带工程组合。  
为什么能发：VI-ReID 过去多是造中间模态或学共享特征，它说真正缺的是可见光和红外特征之间的双向信息交互。证据链是动态投影交互、池化交叉、损失约束分别消融。  
Story 套路：不是把两种模态拉到一个桥上，而是让两种模态互相补信息。

2. `AKT lifelong ReID`
创新类型：问题重定义加新机制。  
为什么能发：LReID 不是简单防遗忘，而是旧知识直接蒸馏会阻碍新域泛化。它用“知识过渡路径”替代硬拷贝旧模型。  
Story 套路：从“保旧知识”升级成“控制旧知识怎么过渡到新任务”。

3. `Camera-aware graph multi-domain adaptive learning`
创新类型：新机制。  
为什么能发：完全无监督 ReID 里，相机不是元数据，而是天然子域。它把每个相机当域，用图传播做跨相机语义对齐。  
Story 套路：不再靠风格增广修相机差异，而是显式建模相机间语义传播。

4. `CEIL`
创新类型：新机制加工程组合。  
为什么能发：UVI-ReID 的伪标签错在跨模态关系映射不稳。它把类别进一步拆成 camera proxy，先稳住模态内聚类，再做跨模态校准。  
Story 套路：跨模态身份不能直接对齐，要先在相机代理层面重新校准。

5. `FLaN-Net`
创新类型：问题重定义加新机制。  
为什么能发：遮挡 ReID 不是单纯看哪里可见，而是要分清人、可见属性、遮挡物。它用细粒度语言 token 显式拆这三类信息。  
Story 套路：遮挡噪声过滤从视觉注意力升级成“人和遮挡物的语义分解”。

6. `CCIL`
创新类型：强问题重定义加新机制。  
为什么能发：换衣 ReID 的核心不是缺衣服无关特征，而是训练集里衣服和身份有虚假相关。它用因果干预切断衣服捷径。  
Story 套路：不是削弱衣服特征，而是从因果上阻断模型学衣服身份捷径。

7. `CCFL`
创新类型：新数据设定加机制。  
为什么能发：隐私和非独立同分布下，各客户端强行统一模型不合理。它允许客户端自定义模型，用交互模型蒸馏交换知识。  
Story 套路：联邦 ReID 不该追求一个统一模型，而该让异构客户端通过中介模型交换知识。

8. `CCUP`
创新类型：新数据设定。  
为什么能发：换衣数据太小，现有方法被数据规模卡住。它用可控合成数据提供大量身份、相机和衣服变化，再做预训练微调。  
Story 套路：换衣泛化不是再调一个模块，而是先补足真实数据永远难补的变化维度。

9. `CFPER`
创新类型：新机制，偏效率问题。  
为什么能发：所有 query 用同样细粒度特征浪费计算，难 query 才需要部位专家。它用早退和部位专家动态分配计算。  
Story 套路：ReID 检索应该按 query 难度花计算，简单样本用全局特征，困难样本才上部位匹配。

**对我们有用的强候选**

1. Headline：航拍地面 ReID 的核心干扰不是普通域差异，而是三维视角混杂造成的身份捷径。  
挂团队资产：aerial-ground 数据，SMPL 3D 几何，SOLIDER-Swin。  
区别：不像 CCIL 切衣服捷径，也不像相机图方法只用 camera ID。我们把高度、视角、人体朝向、投影可见性定义成可干预的三维混杂变量。  
最低成本首验：冻结 SOLIDER 特征，在 AG-ReID.v2 或 CARGO 上测视角、相机、高度能否被线性预测；再做一个简单的视角原型残差化或对抗去视角头。如果视角可预测性很强，并且残差化后跨视角 mAP 有超过噪声的信号，就值得立项。

2. Headline：用 SMPL 定义“互相可见的身体表面”，让遮挡和航拍地面匹配只比较可靠身体区域。  
挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
最低成本首验：不改训练，先用已有姿态或 SMPL 投影估计每张图的部位可见性，统计错误检索是否集中在可见性不匹配样本。再做一个后处理距离，只加可见部位权重。如果零训练后处理都没信号，就先停。

3. Headline：把 camera proxy 升级成 geometry proxy，用三维姿态和视角代理修正无监督跨视角伪标签。  
挂团队资产：aerial-ground 数据，SMPL，pose 热图门控。  
区别：CEIL 和 CGMAL 的代理变量是相机或模态，我们的代理变量是物理几何，包括俯仰视角、人体朝向、尺度、关节可见性。这个切口更适合航拍地面，而不是普通监控相机。  
最低成本首验：用 SOLIDER 特征跑无监督聚类，比较按 camera proxy 和 geometry proxy 切分后的伪标签纯度、跨视角召回率。若 geometry proxy 不能减少明显的跨视角错配，这条不继续。

4. Headline：用三维可控合成数据预训练“视角过渡能力”，不是泛泛扩数据。  
挂团队资产：SMPL 基建，aerial-ground，SOLIDER-Swin。  
区别：CCUP 是为换衣生成大规模合成数据。我们的卖点不是数据规模，而是同一身份在地面、斜视、俯视、低清之间的可控过渡，用来补航拍地面极端视角缺口。  
最低成本首验：先生成很小规模，例如几百个身份、数个视角和尺度，只训练一个轻量投影头或最后几层，比较 AG-ReID.v2 上一颗 seed 的变化。若小规模都没有方向性信号，不投入大生成。

我最看好的顺序是 1、3、2、4。第一条问题层面最强，能把航拍地面的“极端视角”讲成因果混杂，而不是普通跨域。第三条最容易首验，成本低。第二条依赖 SMPL 质量，但一旦成立，机制会比二维姿态门控更硬。第四条成本最高，适合作为前面方向有信号后的数据增强证据，不建议单独当主线。
tokens used
107,048
我只按要求读了每篇前三页，下面判断只基于摘要和引言。

**逐篇反推**

1. `Bidirectional modality information interaction`
创新类型：新机制为主，带工程组合。  
为什么能发：VI-ReID 过去多是造中间模态或学共享特征，它说真正缺的是可见光和红外特征之间的双向信息交互。证据链是动态投影交互、池化交叉、损失约束分别消融。  
Story 套路：不是把两种模态拉到一个桥上，而是让两种模态互相补信息。

2. `AKT lifelong ReID`
创新类型：问题重定义加新机制。  
为什么能发：LReID 不是简单防遗忘，而是旧知识直接蒸馏会阻碍新域泛化。它用“知识过渡路径”替代硬拷贝旧模型。  
Story 套路：从“保旧知识”升级成“控制旧知识怎么过渡到新任务”。

3. `Camera-aware graph multi-domain adaptive learning`
创新类型：新机制。  
为什么能发：完全无监督 ReID 里，相机不是元数据，而是天然子域。它把每个相机当域，用图传播做跨相机语义对齐。  
Story 套路：不再靠风格增广修相机差异，而是显式建模相机间语义传播。

4. `CEIL`
创新类型：新机制加工程组合。  
为什么能发：UVI-ReID 的伪标签错在跨模态关系映射不稳。它把类别进一步拆成 camera proxy，先稳住模态内聚类，再做跨模态校准。  
Story 套路：跨模态身份不能直接对齐，要先在相机代理层面重新校准。

5. `FLaN-Net`
创新类型：问题重定义加新机制。  
为什么能发：遮挡 ReID 不是单纯看哪里可见，而是要分清人、可见属性、遮挡物。它用细粒度语言 token 显式拆这三类信息。  
Story 套路：遮挡噪声过滤从视觉注意力升级成“人和遮挡物的语义分解”。

6. `CCIL`
创新类型：强问题重定义加新机制。  
为什么能发：换衣 ReID 的核心不是缺衣服无关特征，而是训练集里衣服和身份有虚假相关。它用因果干预切断衣服捷径。  
Story 套路：不是削弱衣服特征，而是从因果上阻断模型学衣服身份捷径。

7. `CCFL`
创新类型：新数据设定加机制。  
为什么能发：隐私和非独立同分布下，各客户端强行统一模型不合理。它允许客户端自定义模型，用交互模型蒸馏交换知识。  
Story 套路：联邦 ReID 不该追求一个统一模型，而该让异构客户端通过中介模型交换知识。

8. `CCUP`
创新类型：新数据设定。  
为什么能发：换衣数据太小，现有方法被数据规模卡住。它用可控合成数据提供大量身份、相机和衣服变化，再做预训练微调。  
Story 套路：换衣泛化不是再调一个模块，而是先补足真实数据永远难补的变化维度。

9. `CFPER`
创新类型：新机制，偏效率问题。  
为什么能发：所有 query 用同样细粒度特征浪费计算，难 query 才需要部位专家。它用早退和部位专家动态分配计算。  
Story 套路：ReID 检索应该按 query 难度花计算，简单样本用全局特征，困难样本才上部位匹配。

**对我们有用的强候选**

1. Headline：航拍地面 ReID 的核心干扰不是普通域差异，而是三维视角混杂造成的身份捷径。  
挂团队资产：aerial-ground 数据，SMPL 3D 几何，SOLIDER-Swin。  
区别：不像 CCIL 切衣服捷径，也不像相机图方法只用 camera ID。我们把高度、视角、人体朝向、投影可见性定义成可干预的三维混杂变量。  
最低成本首验：冻结 SOLIDER 特征，在 AG-ReID.v2 或 CARGO 上测视角、相机、高度能否被线性预测；再做一个简单的视角原型残差化或对抗去视角头。如果视角可预测性很强，并且残差化后跨视角 mAP 有超过噪声的信号，就值得立项。

2. Headline：用 SMPL 定义“互相可见的身体表面”，让遮挡和航拍地面匹配只比较可靠身体区域。  
挂团队资产：SMPL 3D 几何，pose 热图门控，SOLIDER-Swin。  
区别：FLaN-Net 用语言分人、属性、遮挡物；CFPER 用二维部位专家。我们用三维人体表面和投影可见性定义匹配支持区域，重点解决俯视、低清、遮挡时二维部位顺序失真的问题。  
最低成本首验：不改训练，先用已有姿态或 SMPL 投影估计每张图的部位可见性，统计错误检索是否集中在可见性不匹配样本。再做一个后处理距离，只加可见部位权重。如果零训练后处理都没信号，就先停。

3. Headline：把 camera proxy 升级成 geometry proxy，用三维姿态和视角代理修正无监督跨视角伪标签。  
挂团队资产：aerial-ground 数据，SMPL，pose 热图门控。  
区别：CEIL 和 CGMAL 的代理变量是相机或模态，我们的代理变量是物理几何，包括俯仰视角、人体朝向、尺度、关节可见性。这个切口更适合航拍地面，而不是普通监控相机。  
最低成本首验：用 SOLIDER 特征跑无监督聚类，比较按 camera proxy 和 geometry proxy 切分后的伪标签纯度、跨视角召回率。若 geometry proxy 不能减少明显的跨视角错配，这条不继续。

4. Headline：用三维可控合成数据预训练“视角过渡能力”，不是泛泛扩数据。  
挂团队资产：SMPL 基建，aerial-ground，SOLIDER-Swin。  
区别：CCUP 是为换衣生成大规模合成数据。我们的卖点不是数据规模，而是同一身份在地面、斜视、俯视、低清之间的可控过渡，用来补航拍地面极端视角缺口。  
最低成本首验：先生成很小规模，例如几百个身份、数个视角和尺度，只训练一个轻量投影头或最后几层，比较 AG-ReID.v2 上一颗 seed 的变化。若小规模都没有方向性信号，不投入大生成。

我最看好的顺序是 1、3、2、4。第一条问题层面最强，能把航拍地面的“极端视角”讲成因果混杂，而不是普通跨域。第三条最容易首验，成本低。第二条依赖 SMPL 质量，但一旦成立，机制会比二维姿态门控更硬。第四条成本最高，适合作为前面方向有信号后的数据增强证据，不建议单独当主线。
