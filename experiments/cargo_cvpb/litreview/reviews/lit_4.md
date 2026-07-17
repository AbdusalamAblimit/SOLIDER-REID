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
session id: 019ef71f-5ff6-7012-ae3f-87ff924273b6
--------
user
你是 ReID 论文创新挖掘员 4/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(4/20):
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会先按项目规则快速读取目标和状态文件，然后只用 `pdftotext -l 3` 抽每篇前三页，重点反推可落到团队资产上的强创新点。当前环境是只读，我不会改任何文件。
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
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
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
目标文件已经读完，当前任务确实是文献差距分析，不开实验、不写设计文档。接下来我先核对这 9 个 PDF 是否都在目录里，再按用户指定的前三页抽取。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
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

exec
/bin/zsh -lc 'which pdftotext && pdftotext -v' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/opt/homebrew/bin/pdftotext
pdftotext version 26.04.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org
Copyright 1996-2011, 2022 Glyph & Cog, LLC

codex
PDF 列表正常，`pdftotext` 可用。现在我分批读取前三页，后面会只抽取和创新套路有关的信息，不做常规长摘要。
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
/bin/zsh -lc "pdftotext -l 3 'CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4100

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

CLIP-Based Camera-Agnostic Feature Learning for
Intra-Camera Supervised Person Re-Identification
Xuan Tan , Xun Gong , Member, IEEE, and Yang Xiang
Abstract— Contrastive Language-Image Pre-Training (CLIP)
model excels in traditional person re-identification (ReID) tasks
due to its inherent advantage in generating textual descriptions for pedestrian images. However, applying CLIP directly
to intra-camera supervised person re-identification (ICS ReID)
presents challenges. ICS ReID requires independent identity
labeling within each camera, without associations across cameras. This limits the effectiveness of text-based enhancements.
To address this, we propose a novel framework called CLIP-based
Camera-Agnostic Feature Learning (CCAFL) for ICS ReID.
Accordingly, two custom modules are designed to guide the model
to actively learn camera-agnostic pedestrian features: IntraCamera Discriminative Learning (ICDL) and Inter-Camera
Adversarial Learning (ICAL). Specifically, we first establish
learnable textual prompts for intra-camera pedestrian images
to obtain crucial semantic supervision signals for subsequent
intra- and inter-camera learning. Then, we design ICDL to
increase inter-class variation by considering the hard positive and hard negative samples within each camera, thereby
learning intra-camera finer-grained pedestrian features. Additionally, we propose ICAL to reduce inter-camera pedestrian
feature discrepancies by penalizing the model’s ability to predict the camera from which a pedestrian image originates,
thus enhancing the model’s capability to recognize pedestrians from different viewpoints. Extensive experiments on
popular ReID datasets demonstrate the effectiveness of our
approach. Especially, on the challenging MSMT17 dataset,
we arrive at 58.9% in terms of mAP accuracy, surpassing
state-of-the-art methods by 7.6%. Code is available at https://
gitee.com/swjtugx/classmate/tree/master/OurGroup/CCAFL.
Index Terms— Person re-identification, CLIP, intra-camera
supervision, camera-based adversarial loss.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) involves identifying the
same individual across different camera views. It has

Received 16 September 2024; revised 12 November 2024 and 2 December
2024; accepted 22 December 2024. Date of publication 24 December 2024;
date of current version 7 May 2025. This work was supported in part by the
National Natural Science Foundation of China under Grant 62376231; in part
by Sichuan Science and Technology Program under Grant 24NSFSC1070;
in part by the Science and Technology Research and Development Program
of China National Railway Group Company Ltd., under Grant K2023T003;
and in part by Tangshan Basic Research Science and Technology Program
under Grant 23130230E. This article was recommended by Associate Editor
Y. M. Ro. (Corresponding author: Xun Gong.)
Xuan Tan and Yang Xiang are with Tangshan Research Institute,
Southwest Jiaotong University, Tangshan 063000, China (e-mail: trangle@
my.swjtu.edu.cn; xiangyang@my.swjtu.edu.cn).
Xun Gong is with the School of Computing and Artificial Intelligence and
the Manufacturing Industry Chains Collaboration and Information Support
Technology Key Laboratory of Sichuan Province, Southwest Jiaotong University, Chengdu, Sichuan 610031, China, and also with the Engineering
Research Center of Sustainable Urban Intelligent Transportation, Ministry of
Education, Chengdu, Sichuan 610031, China (e-mail: xgong@swjtu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3522178

attracted significant attention because of its applications in
person tracking, security systems, and traffic monitoring.
Current research primarily focuses on two directions: fully
supervised [1], [2], [3], [4], [5], [6], [7] and unsupervised [8],
[9], [10]. With the advent of deep learning technologies,
fully supervised person ReID has seen significant performance improvements. However, the considerable annotation
cost associated with the increasing number of cameras and IDs
in real-world scenarios poses a significant challenge for the
practical deployment of ReID systems. Conversely, unsupervised person ReID does not require any label information but
tends to underperform in complex scenarios involving multiple
IDs. In recent years, to combine the advantages and mitigate
the drawbacks of supervised and unsupervised methods, the
Intra-camera supervision (ICS) approach has been proposed.
This approach assumes individual labeling of IDs within
each camera without establishing cross-camera identity links.
As a result, ICS supervision significantly reduces annotation
costs compared to full supervision while still maintaining
identification accuracy. Therefore, it is considered a more
practical setup for ReID scenarios.
However, the lack of cross-camera annotation information
poses a significant challenge for effectively learning pedestrian
features in ICS ReID. Specifically, the number of annotated
pedestrian training samples within each camera is significantly
lower than in fully supervised cross-camera person ReID
tasks. Additionally, due to factors such as varying viewpoints,
occlusion, and background noise, the absence of inter-camera
labels makes it difficult for models to learn variations in
pedestrian appearance across different views, as illustrated in
Fig. 1. Therefore, effectively utilizing intra-camera supervised
information to learn associations between cross-camera IDs is
crucial for addressing ICS ReID tasks.
In ICS ReID, a common approach is dividing model learning into two stages: intra-camera and inter-camera learning.
For instance, the multi-label learning strategy MATE [11]
constructs a Softmax parameter classifier for each camera to
classify pedestrians while associating cross-camera identity
labels. However, the variation in the number of pedestrian
samples within each camera may lead to suboptimal performance. To address this issue, Precise-ICS [12] constructs a
non-parametric classifier [13] for each camera and continues
to train the model by assigning pseudo-labels to highly similar cross-camera images. Despite these advancements, such
methods fail to fully utilize intra-camera sample annotation
information. Pseudo-labels obtained through simple clustering
of different camera angles are often inaccurate and lack flexibility. Notably, the recent large-scale vision-language model

1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
Authorized licensed use limited to:
Downloaded on June 09,2026 atfor
08:58:38
from IEEE Xplore. Restrictions apply.
SeeTIANJIN UNIVERSITY.
https://www.ieee.org/publications/rights/index.html
more UTC
information.

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

Fig. 1. Illustration of label settings under different person Re-ID data configurations. The light-blue areas represent the intra-camera and cross-camera
feature spaces, with different shapes corresponding to different identities.
(a) Conventional fully supervised training data requires unified identity
annotation across all cameras. (b) Intra-camera supervised (ICS) training
data only requires independent identity annotation within each camera view,
utilizing separate class spaces. In ICS ReID data, superscripts of identity labels
indicate camera view labels.

CLIP [14] has demonstrated inherent advantages in generating
image textual descriptions. We can leverage CLIP to describe
unseen pedestrian features, thus generating general descriptions of new pedestrian images without additional annotation
data.
However, this approach faces certain limitations in the
task of cross-camera person re-identification (ICS ReID).
Specifically, since only intra-camera identity labels are available, we initially generate implicit textual features based
on these labels to represent pedestrians. During intra-camera
learning, these textual features serve as effective supervisory
signals within each camera. However, in the cross-camera
learning phase, we rely on clustering algorithms to establish
cross-camera identity associations, generating corresponding
textual features for each pedestrian across cameras. While
these features provide additional semantic supervision for
cross-camera learning, pedestrian images are often affected by
factors such as background and viewpoint variations, which
compromise the stability of cross-camera textual features—
especially in complex environments. This noisy, cross-camera
textual information can introduce biases in model learning,
thereby limiting the full potential of CLIP-based methods in
semi-supervised person re-identification.
To further optimize cross-camera textual features, we propose leveraging intra-camera labeled image data and
camera-specific labels to improve the quality of crosscamera associations. To this end, we introduce a CLIP-based
Camera-Agnostic Feature Learning (CCAFL) method, which
progressively enhances feature discriminability through a
three-stage learning process. As illustrated in Fig. 2, our

4101

Fig. 2. The diagram illustrates our proposed approach, which leverages
CLIP and prompt learning to generate textual descriptions for person images
within each camera. Based on this, we combine the textual information with
intra-camera and inter-camera learning, enabling the model to focus better on
discriminative features.

approach’s components work together to address the challenges inherent in ICS ReID tasks. Initially, we employ prompt
learning to generate textual descriptions for each labeled
pedestrian within each camera, thus providing additional
supervisory signals for subsequent intra- and inter-camera
learning stages. By integrating our proposed Intra-Camera
Discriminative Learning (ICDL) and Inter-Camera Adversarial Learning (ICAL) modules, these textual descriptions
enable the model to more accurately capture the representative
features of the same pedestrian across varying viewpoints,
effectively mitigating the impact of noisy pseudo-labels and
enhancing the supervision efficacy of textual prompt learning
in cross-camera person re-identification.
In the intra-camera learning stage, to further enhance the
discriminative power of pedestrian features within each camera, we construct an independent hybrid feature memory
bank for each camera using annotated intra-camera IDs. This
memory bank stores the average features and all instance
features for each ID. Subsequently, we apply Intra-Camera
Discriminative Learning (ICDL), which considers hard-toclassify positive and negative samples within the same camera
while leveraging textual features obtained in the first stage to
further reduce intra-camera intra-class variance and inter-class
similarity.
In the inter-camera learning stage, we first use a
cross-camera association algorithm to link cross-camera
pedestrian IDs to improve the accuracy of cross-camera
pedestrian ID associations and enhance the model’s ability to
recognize pedestrians from different viewpoints. Using these
associated IDs, we construct a cross-camera feature memory
bank that stores their central features for contrastive learning. Then, to better reduce the data distribution differences
between different camera views, we propose an Inter-Camera

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

4102

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Adversarial Learning (ICAL) method. Specifically, we add
a multi-camera classifier after the backbone network of the
re-identification model and define ICAL as a multi-positive
class classification loss. During training, by minimizing
ICAL, we force the backbone network of the re-identification
model to learn camera-invariant features by penalizing the
model’s ability to predict the corresponding camera for the
same identity. Backpropagation enables the feature maps
to capture camera-invariant features. Finally, utilizing the
intra-camera supervisory semantic information obtained previously, we assign a textual description to each cluster based on
the generated pseudo-labels. These textual descriptions effectively summarize individual images and serve as additional
supervisory information for inter-camera learning.
Our main contributions can be summarized as follows:
• We propose a simple yet effective three-stage training
strategy, called CCAFL, that integrates CLIP-generated
textual information into the novel semi-supervised ICS
ReID task for subsequent learning.
• Two critical modules: Intra-Camera Discriminative
Learning (ICDL) and Inter-Camera Adversarial Learning
(ICAL) are introduced to compel the model to learn
camera-agnostic features. ICDL aims to extract intracamera fine-grained pedestrian features, while ICAL
reduces the inter-camera discrepancies in pedestrian feature distribution. These modules collectively enhance the
accuracy of cross-camera pedestrian identity recognition.
• Extensive experiments conducted on three popular person
re-identification benchmarks, Market-1501, DukeMTMCReID and MSMT17, demonstrate that our method
significantly outperforms the current state-of-the-art ICS
methods. Our performance even exceeds that of fully
supervised methods.
II. R ELATED W ORK
A. Intra-Camera Supervised Person ReID
With the increasing number of cameras and persons in realworld scenarios, the task of annotating a large-scale ID dataset
becomes prohibitively costly. To address this issue, a setup
known as Intra-camera supervision (ICS) ReID has been proposed, where annotations are performed independently across
various cameras, with labels only available for persons within
the same camera view. Previous methods approached the
ICS ReID problem from two angles: intra-camera supervised
learning and inter-camera ID association learning. In intracamera learning, PCSL [15] and ACAN [16] employ a direct
triplet loss [17] to train models, while MATE [11] constructs
a multi-branch classifier for each camera. However, when the
distribution of intra-camera ID samples is unbalanced and
scant, it can result in biased learning. In contrast, PreciseICS [12] uses a non-parametric classifier and undertakes joint
learning, but insufficient intra-camera learning can severely
impair the model when persons with high intra-camera similarity are present. For inter-camera learning, Precise-ICS
supervises learning through pseudo-labeling based on the similarity of person features across cameras. CMT [18] combines
contrastive learning with the Mean Teacher [19] paradigm to

construct a semi-supervised learning framework. However, the
methods above overlook the labeled instance features within
the same camera, leading to insufficient intra-camera learning
and consequently failing to effectively distinguish pedestrian
features within the same camera. PIRID [20] and DCL [21]
also leverage contrastive learning to learn pedestrian features.
However, as the number of pedestrians within each camera
increases, relying solely on camera-specific mean features
fails to capture sufficient discriminative characteristics, which
ultimately affects model accuracy. In contrast, we propose
a within-camera discriminative learning approach that combines mean features with instance features, enabling more
comprehensive learning of fine-grained pedestrian features
within each camera and thereby improving model performance. Additionally, these methods do not adequately consider
the disparities in data distribution across different cameras,
failing to fully capture the invariant features of pedestrians
across cameras.
Differently, we design Intra-Camera Discriminative Loss
(ICDL) and Inter-Camera Adversarial Loss (ICAL) methods
to effectively enhance the model’s ability to distinguish pedestrian ID features within and across cameras. Additionally,
we incorporate high-level semantic features generated by CLIP
for each person within a camera to further boost the model’s
performance.
B. Unsupervised Person ReID
In recent years, unsupervised person ReID [8], [9], [10],
[22], [23], [24], [25] tasks have attracted wide attention.
These tasks are primarily categorized based on whether additional related data are employed, encapsulating unsupervised
domain-adaptive (UDA) ReID and purely unsupervised learning (USL) ReID. The latter, pure unsupervised ReID, presents
greater challenges due to its independence from any external
data. However, with the successful application of contrastive
learning in the unsupervised domain, the performance of
USL ReID has significantly increased - notable methods
include SPCL [8]’s self-paced contrastive learning procedure
that builds a mixed memory bank, fully exploiting all available data. CAP [22] technique divides clusters into multiple
camera-perception proxies based on the camera ID to alleviate
discrepancies in ID features generated by camera perspective
alterations. ClusterContrast [24] directly establishes a simple
yet effective cluster-level memory bank, achieving decoupling
between feature update rates and the number of images.
RTMem [26] employs a real-time memory update strategy,
updating cluster centroids by randomly sampling current
mini-batch instance features without the need for momentum.
In contrast, LP [27] considers two types of additional features
from different local views and leverages the knowledge of an
offline teacher model to optimize the model. In this study, our
work is grounded in the framework of intra- and inter-camera
contrastive learning, which is a widely used and effective
representation learning method for unsupervised person ReID.
C. Vision-Language Models
Large-scale pre-trained vision-language models, integrating
copious amounts of textual and visual data, have proven

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 262 (2025) 104552

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

Channel-aware feature mining network for Visible–Infrared Person
Re-identification
Pengxia Li, Zhonghao Du, Linhui Zhang, Yanyi Lv, Yujie Liu ∗
Qingdao Institute of Software, College of Computer Science and Technology, China University of Petroleum (East China), Shandong Key Laboratory of Intelligent
Oil and Gas Industrial Software, China

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
Visible Infrared Person Re-identification
Feature learning
Channel-level processing

ABSTRACT
Visible–Infrared Person Re-identification (VI-ReID) aims to match the identities of pedestrians captured
by non-overlapping cameras in both visible and infrared modalities. The key to overcoming the VI-ReID
challenge lies in extracting diverse modality-shared features. Current methods mainly focus on channellevel operations during data preprocessing, with the aim of expanding the dataset. However, these methods
often overlook the complex relationships among channel features, leading to insufficient utilization of unique
information in each channel. To address this issue, we propose the Channel-Aware Feature Mining Network
(CAFMNet) to improve VI-ReID effectiveness. Specifically, we design three core modules: a Channel-Level
Feature Optimization (CLFO) module, which captures channel-level key features for identity recognition and
directly extracts identity-relevant information at the channel level; a Channel-Level Feature Refinement (CLFR)
module, which enhances channel-level features while retaining useful information—addressing the irrelevant
content in initially extracted features; a Multi-Dimensional Feature Optimization (MDFO) module, which
comprehensively processes multi-dimensional feature information to enhance the model’s ability to understand
and describe input data. Extensive experiments on the SYSU-MM01 and LLCM datasets demonstrate that
our CAFMNet outperforms existing approaches in terms of VI-ReID effectiveness. The code is available at
https://github.com/cobeibei/CAFMNet-1.

1. Introduction
Person re-identification (Re-ID) is a core technology in intelligent
video surveillance systems, as it enables the retrieval of images or
video sequences of specific pedestrians from multi-view data captured
by non-overlapping cameras. In practical urban security scenarios, the
demand for 24-hour continuous surveillance is increasing. However,
traditional Re-ID methods relying solely on RGB images suffer from
significant performance degradation in low-light, nighttime, or backlit
environments. Infrared (IR) cameras can effectively overcome lighting
limitations by capturing thermal radiation from objects, which has
attracted widespread attention among researchers to visible–infrared
person re-identification (VI-ReID).
The core goal of VI-ReID is to associate RGB and IR images of
the same pedestrian in camera data. Nevertheless, the inherent differences between the RGB and IR imaging mechanisms pose enormous
challenges to this task: RGB images contain rich color and texture information, but are limited in use under poor lighting conditions; IR images
reflect thermal radiation information, but have weak texture details and
lack color cues. These differences result in a prominent ‘‘modality gap’’:

Features extracted from the two modalities often exhibit drastically
different statistical distributions, making direct cross-modal matching
extremely difficult. Therefore, the core challenge of VI-ReID lies in
extracting modality-shared, identity-discriminative features that are
robust to cross-modal differences.
Recent advances in deep learning have demonstrated that finegrained feature extraction — particularly channel-level feature extraction — is crucial for enhancing the performance of Re-ID models. In
VI-ReID scenarios, channel-level features play a key role in capturing
identity-discriminative patterns that are invariant across modalities.
For instance, specific channels of RGB images may emphasize clothing
textures, while the single channel of IR images tends to focus on
presenting pedestrian contours. Despite this understanding, existing VIReID methods still have obvious shortcomings in effectively utilizing
channel-level features, and this issue has become a bottleneck hindering
further performance improvement.
Existing VI-ReID methods primarily address modality differences
through two strategies: auxiliary information-based methods and feature learning-based methods. Regrettably, neither strategy fully taps

∗ Corresponding author.

E-mail address: liuyujie@upc.edu.cn (Y. Liu).
https://doi.org/10.1016/j.cviu.2025.104552
Received 17 June 2025; Received in revised form 16 September 2025; Accepted 19 October 2025
Available online 24 October 2025
1077-3142/© 2025 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Feature Mining Network (CAFMNet). CAFMNet consists of three core
modules: the Channel-Level Feature Optimization (CLFO) module, the
Channel-Level Feature Refinement (CLFR) module, and the MultiDimensional Feature Optimization (MDFO) module. These modules
work collaboratively to overcome the limitations of existing methods. The CLFO module directly extracts useful feature information
for identity recognition at the channel level. Considering that the
initially extracted features may contain some irrelevant information,
we design a Channel-Level Feature Refinement (CLFR) module to
further enhance channel-level features, thereby achieving the goal of
strengthening these features while suppressing irrelevant information.
Finally, to enhance the model’s ability to understand and represent input data, we use the Multi-Dimensional Feature Optimization (MDFO)
module to comprehensively process feature information from multiple dimensions, which improves feature quality and strengthens key
features.
Our primary contributions are summarized as follows:
⋅ We propose a novel deep learning framework, the Channel-Aware
Feature Mining Network (CAFMNet), which focuses on accurately capturing the channel-level features most critical to identification for
efficient visible–infrared person re-identification (VI-ReID).
⋅ We propose a novel Channel-Level Feature Optimization (CLFO)
module that extracts identification-relevant features at the channel
level. Additionally, we introduce a Channel-Level Feature Refinement
(CLFR) module that captures critical discriminative information while
suppressing irrelevant data, thereby enhancing channel-level features.
⋅ We propose the Multi-Dimensional Feature Optimization (MDFO)
module to process feature information from various dimensions, enhancing critical features and improving the model’s understanding of
input data.
⋅ Experimental results on the SYSU-MM01 and LLCM datasets show
that our approach achieves a new state-of-the-art.

Fig. 1. The different channel activations of the feature maps are shown, with
each sub-map corresponding to a different channel. We randomly selected
a pedestrian image from the SYSU-MM01 dataset to visualize the different
channel activations of its original feature map.

into the potential of channel-level features. Auxiliary information-based
methods attempt to build a bridge between RGB and IR modalities by
introducing additional auxiliary data. Although they have contributed
to the field, they have inherent flaws: generating auxiliary information
typically requires extra processing steps, which increases computational
burden; some auxiliary data may also need manual annotation, further
raising data costs. More critically, these methods do not directly model
the channel-level features of raw RGB/IR data; instead, they rely on
auxiliary modalities to indirectly alleviate cross-modal differences, thus
missing the opportunity to capture unique discriminative cues in each
channel of the target modalities.
In contrast, feature learning-based methods aim to extract meaningful feature representations directly from raw multi-modal data without
relying on external auxiliary information. Their core goal is to reduce
cross-modal differences through techniques such as pixel-level feature
alignment and mapping multi-modal features to a shared feature space.
Among them, pixel-level feature alignment methods are extremely
sensitive to image noise and subtle color changes, which may damage
channel-level semantic information; methods that map multi-modal
features to a shared feature space ignore the ‘‘channel imbalance’’
phenomenon—different channels within a single modality (e.g., the
R, G, B channels of RGB images) and across modalities (e.g., RGB
channels vs. IR channels) contribute differently to identity recognition. Additionally, some feature learning-based methods only perform
channel-level operations during the data preprocessing stage, with the
sole purpose of expanding dataset scale. This approach overlooks the
valuable discriminative information inherent in each channel.
Through detailed feature map visualization (as shown in Fig. 1), it
can be observed that some channels of RGB/IR images contain unique
and information-rich identity cues (e.g., some channels highlight the
clothing texture or thermal contour of pedestrians); in contrast, other
channels contribute little to recognition and may even introduce noise.
This channel imbalance phenomenon, combined with the failure of
existing methods to explicitly model and refine channel-level features,
results in poor feature learning performance in VI-ReID tasks and limits
model performance. To address these urgent issues awaiting breakthroughs, we propose a novel Channel-Aware Feature Mining Network
(CAFMNet), which is specifically designed to extract and optimize
channel-level features crucial for VI-ReID.
In this paper, to tackle this challenge, we propose a new method for
extracting complex channel-level features, called the Channel-Aware

2. Related work
Visible–infrared person re-identification (VI-ReID) addresses modality discrepancies through two methods: using auxiliary information and
feature learning.
2.1. Using auxiliary information
In Visible–Infrared Person Re-identification (VI-ReID), the method
based on auxiliary information aims to effectively alleviate cross-modal
differences by introducing additional auxiliary modalities, which play
a crucial role as a bridge connecting the visible and infrared modalities. Common auxiliary modalities in this task include other generated
images, grayscale-transformed images, contour information, skeleton
information, and textual descriptions.
Align-GAN (Wang et al., 2019) uses GANs to convert RGB images to infrared, while AGPI2 (Alehdaghi et al., 2023) employs GANs
to create virtual intermediate images for improving RGB-infrared reidentification. Due to the absence of cross-modal pairs, these methods
may generate noisy images. CycleGAN (Zhu et al., 2017), which enables unsupervised image transformation without paired data, has been
applied in VI-ReID: for example, JSIA-ReID (Yang et al., 2020) uses
CycleGAN to generate cross-modal paired images from unpaired sets,
and another study (Xia et al., 2021) introduces a CycleGAN-based
IMT network to synthesize cross-modal images and expand the dataset.
However, CycleGAN-generated images may still have flaws such as
missing details, inaccurate colors, or unnatural textures, which affect
model performance.
In addition to GAN-generated images, other auxiliary modalities are
also widely used in VI-ReID: PMT (Lu et al., 2023), WF-CAMReVi (Sarker
and Zhao, 2024), and HAT (Ye et al., 2020b) utilize grayscale images to
help models learn modality-invariant features and minimize differences
between RGB and infrared images; SPOT (Chen et al., 2022) enhances
2

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

recognition through pose structure information; STAR (Jiang et al.,
2024) improves video-based VI-ReID accuracy by leveraging skeleton
data; CGMMNet (Xu and Zhao, 2024) addresses color discrepancies and
blurred boundaries using intermediate modality images and contour
maps; lastly, YYDS (Du et al., 2024) boosts re-identification performance by incorporating rough textual descriptions to fill in missing
color information in infrared images. Overall, using generated images as auxiliary information typically involves Generative Adversarial
Networks (GANs) for modality transformations (e.g., converting RGB
images to infrared images).
The introduction of auxiliary information can provide additional
contextual semantic information to compensate for modality differences, which indeed contributes significantly to improving model performance. However, this approach has inherent limitations: the generation of auxiliary information requires additional image processing
or natural language processing steps, which increases computational
burden; some auxiliary information also needs manual annotation,
raising data costs; furthermore, auxiliary information may introduce
redundant information or noise. In addition, modality differences between auxiliary information and target modalities lead to consistency
issues, which require additional alignment and fusion strategies.

channels of an RGB image with a single channel" as auxiliary modalities
to reduce modality differences. However, both approaches only use
channel operations for data expansion or modality adaptation, without
deeply exploring the identity-discriminative information contained in
individual channels. CHCR (Pang et al., 2023) provides a new perspective for channel-level processing: its inter-channel pseudo-label
refinement method, based on the principle that the three RGB channels
of the same sample correspond to the same identity, performs crossmodal clustering on each of the three channels with the infrared
modality separately. It evaluates consistency using the Intersection
over Union (IoU) to eliminate unreliable pseudo-labels, which not
only mitigates the information loss caused by traditional single-channel
conversion but also verifies the performance gain brought by channellevel features. Our visualization experiments (as shown in Fig. 1)
also confirm that some channels indeed contain highly discriminative
features crucial for identity recognition. Therefore, more efforts should
be devoted to exploring and utilizing channel-level features to enhance
the model’s representational capability and cross-modal recognition
accuracy.

2.2. Feature learning

The network (as shown in Fig. 2) adopts a dual-stream ResNet50 (He et al., 2016; Ye et al., 2020a) architecture to separately process
features from RGB and infrared (IR) images, which effectively addresses
the matching challenges in visible–infrared person re-identification (VIReID) caused by modality differences. First, the input visible–infrared
(VIS-IR) features pass through the Channel-Level Feature Optimization (CLFO) module, which directly extracts channel-level key features
closely related to identity recognition. To further improve the quality
of these channel features, we design the Channel-Level Feature Refinement (CLFR) module to suppress redundant or irrelevant information
and enhance discriminative features, thereby improving the accuracy
and robustness of feature representation.
On this basis, to enhance the model’s ability to understand and
describe input data, we introduce the Multi-Dimensional Feature Optimization (MDFO) module, which further explores and integrates feature information across multiple dimensions and layers. Through the
sequential processing of these three modules, the network can extract
richer and highly relevant key features from the original multi-modal
data, significantly strengthening the model’s discriminative ability.

3. Methodology

Feature learning methods aim to extract and learn meaningful feature representations directly from raw multi-modal data, rather than
relying on image transformations or additional auxiliary information.
Their core objective is to reduce discrepancies between different modalities through specific techniques — such as aligning features at the
pixel level or mapping multi-modal features directly into a shared
feature space — thereby improving the model’s generalization ability
and recognition accuracy. This approach emphasizes enhancing the
model’s understanding and processing capabilities for multi-source data
without introducing external information.
Pixel-level feature alignment methods operate directly on each pixel
in the image. For example, SAAI (Fang et al., 2023) achieves aggregation of potential semantic partial features by calculating the similarity
between pixel-level features and learnable prototypes; DCLNet (Sun
et al., 2022) proposes a dense contrastive learning network to perform
pixel-to-pixel dense alignment; CSL (Nie et al., 2024) designs a pixellevel color transformation module to learn the relationships between
different color channels. However, since these methods operate directly
at the pixel level, they are highly sensitive to image noise or subtle color
variations—this can significantly affect the model’s feature extraction
and recognition performance.
Another category of methods aims to project multi-modal features
into a shared feature space to learn a unified cross-modal representation. For example, MAUM (Liu et al., 2022) designs a one-way metric
learning approach that enhances memory capability by learning crossmodal metrics in two directions; RFM (Tan et al., 2023) introduces
a cross-modal center loss at the feature level to explore more compact intra-class distributions and employs a modality-aware spatial
attention module to better exploit texture regions. However, due to
significant differences between RGB and infrared images in information
capacity, representation, sharpness, and lighting conditions, simply
mapping them into the same feature space is insufficient to fully
eliminate modality gaps. This may also lead to the loss of important modality-specific information, which negatively impacts overall
recognition performance.
In addition, channel-level processing deserves attention in visible–
infrared person re-identification (VI-ReID), yet its significant value
remains underutilized. Current works mostly focus on preprocessing:
for example, Yang et al. (2022b), Wu and Ye (2023), Teng et al.
(2024), Dai et al. (2024), Zhang et al. (2024) generate color-invariant
images through random channel enhancement to expand the dataset;
CAJ (Ye et al., 2021a) uses images generated by "replacing the three

3.1. Channel-level feature optimization module
In the visible–infrared person re-identification (VI-ReID) task, RGB
and infrared (IR) images exhibit significant differences in channel
distribution due to their distinct imaging principles. For example, IR
images typically contain only a single thermal channel, while RGB images have three color channels. This modality asymmetry means certain
channels may carry stronger identity-discriminative information, while
others could be redundant or noisy.
To address this issue, we propose the Channel-Level Feature Optimization (CLFO) module. Its goal is to enhance the model’s ability
to extract discriminative identity features across modalities through
multi-level channel modeling and dynamic feature refinement. Unlike
traditional attention mechanisms (e.g., SE, CBAM) that focus solely on
channel importance estimation, CLFO integrates depthwise separable
convolution, group normalization, and a learnable residual connection
into a unified framework. These components work together to achieve
fine-grained channel-level feature modeling at the early stage of feature extraction, effectively mitigating the impact of channel imbalance
between RGB and IR images.
This design allows CLFO to not only adaptively highlight informative channels but also maintain computational efficiency and training
stability—key requirements for VI-ReID tasks. In what follows, we
3


exec
/bin/zsh -lc "pdftotext -l 3 'CLIP-driven fine-grained mining for text-based person search.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 267 (2026) 104741

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

CLIP-driven fine-grained mining for text-based person search
Xianwen Lin a , Xia Geng a ,∗, Shengli Wu b , Zhi Yang c
a

School of Computer Science and Communication Engineering, Jiangsu University, Zhenjiang, 212013, China
School of Computing, Ulster University, Belfast, United Kingdom
c
School of Management, Jiangsu University, Zhenjiang, 212013, China
b

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
Cross-modality
Text-based person search
Fine-grained representation learning

ABSTRACT
Text-based Person Search aims to retrieve images of pedestrians based on textual descriptions. The primary challenge lies in exploring intra-modal discrepancies while ensuring inter-modal consistency. Recently,
Contrastive Language-Image Pre-training (CLIP) has attracted extensive attention for its powerful semantic
understanding ability. However, existing approaches mainly rely on masked modeling tasks to achieve
implicit fine-grained relationship learning. In contrast, we propose a CLIP-Driven Fine-grained Mining (CDFM)
framework that explicitly mines fine-grained relationships without introducing external tools. CDFM extracts
well aligned fine-grained representations while maintaining global alignment. Specifically, we first design
an Attention Bias-based visual Forward (ABF) process that extracts local visual representations through
unidirectional information transfer of local patch sets. Secondly, to obtain fine-grained text representations,
we propose a Fine-grained Embedding Learning (FEL) module and a Text Extraction Strategy (TES). FEL
uses a cross-attention mechanism to aggregate fine-grained text information into a set of learnable labels.
To ensure the fine-grained text representations are well-aligned, TES uses the local visual representations as
supervision, transferring cosine similarity and Euclidean distance knowledge from the pre-training space into
the multimodal decoder. Experimental results demonstrate the superiority of our CDFM.

1. Introduction
Text-based Person Search (TBPS) (Li et al., 2017) aims to retrieve
a set of images most relevant to a given text description from a
person image library. It is a subtask of image–text retrieval and person
re-identification. The introduction of textual description results in significant modality heterogeneity and intra-identity variations, making
TBPS more demanding in terms of fine-grained information. Benefiting from the success of Visual-Language Pretraining models (VLP),
recent works (Jiang and Ye, 2023; Han et al., 2021; Yan et al., 2022)
apply their powerful knowledge to TBPS, alleviating the above problems and achieving impressive results. Compared to models pre-trained
on unimodal data, VLPs provide more robust, high-quality semantic
representations under natural language supervision, with Contrastive
Language-Image Pre-training (CLIP) (Radford et al., 2021) being the
most representative example.
Although IRRA (Jiang and Ye, 2023) has demonstrated the effectiveness of fine-tuning the full CLIP model on TBPS, TBPS requires the
model to distinguish subtle differences between pedestrians, whereas
CLIP primarily focuses on instance-level alignment (image-to-sentence)
during the pre-training stage. Upon reviewing recent works, we identify

several critical issues in applying the CLIP model to explicitly mine
fine-grained information.
Firstly, Reliance solely on global alignment may lead to potential
task gaps (Fig. 1(a)). Previous approaches (Chen et al., 2022; Gao
et al., 2021) to addressing this challenge typically employed the Partbased Convolutional Baseline (PCB) (Sun et al., 2018) to explicitly
obtain local visual features, but these works relied on CNN-based
visual encoders. H. Li et al. (2021) attempted to introduce PCB to a
transformer-based visual encoder and modified the partition method,
but did not achieve satisfactory performance. Secondly, simple modal
sharing may lead to suboptimal results. Zuo et al. (2024) proposed a
shared cross-modal decoder to mine fine-grained representations, but
neglected the compatibility between decoder and pre-training space.
Finally, introducing external tools makes the network rely on their
performance (Fig. 1(b)). For instance, Wang et al. (2020) employs the
Stanford POS tagger to parse text and uses HRNet to segment human
images. While this approach can enhance model performance, as pretraining and fine-tuning have emerged as a new paradigm, we suggest
focusing on exploring the knowledge of pre-training networks.
In light of the above observations, we propose a CLIP-Driven Finegrained Mining framework (CDFM). CDFM utilizes CLIP as the initialization to mitigate the misalignment of cross-modal features and

∗ Corresponding author.

E-mail addresses: sharven@163.com (X. Lin), gengxia@ujs.edu.cn (X. Geng), s.wu1@ulster.ac.uk (S. Wu), yangzhi@ujs.edu.cn (Z. Yang).
https://doi.org/10.1016/j.cviu.2026.104741
Received 6 October 2024; Received in revised form 7 February 2026; Accepted 13 March 2026
Available online 18 March 2026
1077-3142/© 2026 Published by Elsevier Inc.

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

Fine-grained Embedding Learning (FEL) module, without affecting global representations or introducing external tools.
• We design a text extraction strategy to transfer the knowledge
of CLIP into the proposed FEL without semantic annotations,
ensuring the fine-grained textual features are aligned with the
corresponding visual concepts.
• Extensive experiments on three public TBPS benchmarks, i.e.,
CUHK-PEDES (Li et al., 2017), ICFG-PEDES (Ding et al., 2021),
and RSTPReid (Zhu et al., 2021) demonstrate the effectiveness of
CDFM.
2. Related work
2.1. Text-based person search
Text-based person search was first proposed by Li et al. (2017),
who constructed the first benchmark dataset CUHK-PEDES. The main
challenge in TBPS is finding a suitable joint image–text embedding
space and aligning image and text embeddings. Early works focused
on exploring the association of global features between person image–
text pairs. Zheng et al. (2020) proposed Instance Loss to reduce the
cross-modal differences between two heterogeneous features, leading
to more discriminative embeddings. Zhang and Lu (2018) introduced
CMPM Loss and CMPC Loss to align cross-modal representations. Shu
et al. (2022) introduced a modality-shared network to align global features through two implicit semantic alignment paradigms. While these
approaches learn discriminative global cross-modal embedding representations, they overlook the high demand for fine-grained cross-modal
alignment in TBPS.
Recent works have attempted to extract fine-grained features. Wang
et al. (2020), Aggarwal et al. (2020) and Jing et al. (2020) introduce
external tools to explicitly exploit human segmentation, pose estimation, noun phrase extraction, and attributes prediction to facilitate
cross-modal alignment. Although these methods extract enhanced local
features explicitly rather than implicitly, the introduction of external
tools leads to an inefficient and time-consuming process during network
training. Gao et al. (2021) proposed NAFS, which improved the feature
extraction backbone to obtain representations at different scales and
introduced cross-modality cross-attention methods for joint alignment
over full-scale representations. However, the computational cost of
dynamically calculating local feature similarity during the inference
stage limits its applicability in real-world scenarios. Ding et al. (2021)
proposed SSAN, which extracts partially aligned embeddings from the
visual and textual modalities and introduces the word attention module
to infer the corresponding relationship between words and parts based
on the representation of each word. Since the attention mechanism is
solely employed in the text modality and ignores the interaction with
the other modality, the extracted text features do not align well with
the corresponding image features. Zuo et al. (2024) proposed a shared
cross-modal granularity decoder to achieve fine-grained alignment.
Nevertheless, due to the neglect of the influence of randomly initialized
transformer blocks on the pre-training backbone network, the model is
inclined to be trapped in the local minimum. In contrast, our approach
extracts fine-grained text features through cross-attention mechanisms
and modality-sharing decoder, and achieves cross-modal alignment
through visual supervision by proposed text extraction strategy without
requiring extra tools or annotations.
As vision-language pre-training models (Radford et al., 2021; J. Li
et al., 2021; Chen et al., 2020; Su et al., 2019) gradually demonstrate
strong cross-modal alignment capabilities, some works have attempted
to generalize CLIP model to TBPS. Han et al. (2021) proposed a
momentum contrastive learning framework to address the data-scarce
problem by transferring the knowledge learned by the CLIP model from
generic image–text pairs. Yan et al. (2022) proposed a CLIP-Driven
Fine-grained framework, which selects informative tokens based on
the [CLS] token’s attention map to transfer CLIP’s knowledge. Wang

Fig. 1. Motivation of our proposed CDFM. (a) Early global-matching methods directly align global image and text embeddings. (b) Recent local-matching
paradigms introduce external tools. (c) Our fine-grained mining method explicitly extracts and aligns fine-grained image and text embeddings without
affecting global representations or introducing external tools.

explicitly mine fine-grained differences as supplementary information
alongside global alignment. First, inspired by CLIP’s success in image
segmentation, we propose a novel Attention Bias-based visual Forward (ABF) process for extracting local visual features. Unlike previous
approaches (Chen et al., 2022; Wang et al., 2020), we group the
embedded patches according to the specific topology of the pedestrians,
prepend a local classification ([CLS]) token to each group, and forward
the token sequence with the assistance of attention bias matrix. The
state of local [CLS] tokens at the output of the visual encoder serves as
the local image representations. Next, we propose a Fine-grained Embedding Learning (FEL) module to extract fine-grained text embeddings
corresponding to the visual semantics. Specifically, we feed embedded
textual tokens into a multimodal decoder with a set of learnable tokens
to explore the semantic correlation between words and tokens. This
allows text embeddings to be implicitly combined with semantically
similar tokens without relying on external tools. We term the final
output as fine-grained text embeddings.
However, in the absence of semantic annotations, ensuring the
alignment between fine-grained text embeddings and the corresponding
local image representations is challenging. To address this, we design
a Text Extraction Strategy (TES) that correlates cosine similarity and
transfers Euclidean distance under visual supervision. The underlying
logic of TES is that modal sharing is effective for cross-modal tasks (Shu
et al., 2022; H. Li et al., 2021; Zuo et al., 2024), and the image–
text joint latent embedding space of the pre-training model should be
consistent. Therefore, we take the image modality as a bridge to align
the fine-grained image embeddings obtained from the modality-shared
FEL with the output of the ABF. In this way, the knowledge from the
CLIP latent space will be transferred to the FEL, ensuring that the
fine-grained text embeddings extracted by the FEL maintain consistent
semantic cues with their corresponding visual concepts.
Finally, to prevent the model from being trapped in the local
minimum due to the random initialization of FEL (Bai et al., 2023),
we draw inspiration from J. Li et al. (2021) and introduce Momentum
Distillation (MoD) during the training process of FEL. This approach
aims to learn more robust fine-grained embeddings from pseudo-targets
generated by momentum modality-shared learnable tokens.
Our main contributions and innovations are summarized as follows:
• We propose a CLIP-driven fine-grained mining framework that
explicitly leverages fine-grained embeddings through the proposed Attention Bias-based visual Forward process (ABF) and
2

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

et al. (2024) proposed a CLIP-guided fusion framework for pedestrian attribute recognition, exploring CLIP’s capability in understanding fine-grained information such as attribute labels. Jiang and Ye
(2023) directly fine-tuned the CLIP model, and the remarkable results demonstrated that the visual-language pretraining model has high
compatibility with TBPS.

The output of vision encoder is projected into the image–text joint
latent embedding space using a learnable image projection, denoted
as 𝐹𝑣 = {𝑓𝑣𝑐𝑙𝑠 , 𝑓𝑣1 , … , 𝑓𝑣𝑁 } ∈ 𝑅(𝑁+1)×𝐶 , where 𝑓𝑣𝑐𝑙𝑠 represents the global
embeddings of the input image.
Text Encoder. For text description, the input sentence 𝑇 is first
tokenized via a lower-cased Byte Pair Encoding (BPE) with a vocabulary size of 49,152, and then processed by a Transformer with masked
self-attention modified by Radford et al. (2021). Since the text description is of variable length, we prepend and postpend learnable [SOS]
and [EOS] tokens to the sequence, setting the maximum sequence
length to 77. Similar to image branch, a learnable text projection is
used to project text features into the latent space, denoted as 𝐹𝑡 =
{𝑓𝑡SOS , 𝑓𝑡1 , … , 𝑓𝑡𝑀 , 𝑓𝑡EOS } ∈ 𝑅(𝑀+2)×𝐶 , where 𝑓𝑡EOS serves as the global
text embedding, and {𝑓𝑡𝑖 }𝑀
are word-level embeddings.
𝑖=1
Global alignment. Our proposed CDFM mines fine-grained local
relations under the alignment of global image–text embeddings. We
optimize the global embedding via common ID Loss (Zheng et al., 2020)
and SDM Loss (Jiang and Ye, 2023). ID loss is a cross-entropy loss
that promotes the transfer of CLIP knowledge in TBPS tasks by treating
text-based person search problem as classification problem through a
modality-shared classification head. SDM loss utilizes KL divergence
to measure the difference between the cross-modal cosine similarity
matrix and the ground truth matrix in a mini-batch. This loss function
enlarges the similarity of matched pairs and reduces the similarity of
mismatched pairs. The SDM loss is formulated as:

2.2. Vision-language pre-training
Inspired by the success of unimodal pretraining models in Transformer-based language pretraining, such as BERT (Devlin et al.,
2018) and Vision Transformer (Dosovitskiy et al., 2020), many works
have attempted to exploit large-scale image–text datasets for pre-training to enhance the relevance of image–text modalities. Pre-training
and fine-tuning have emerged as a mainstream paradigm for learning
multimodal representations.
Existing VLP work can be categorized into single-stream, dualstream, and mixed architectures based on the backbone architecture:
• Single-stream framework: This consists of a single shared Transformer encoder (Shu et al., 2022; H. Li et al., 2021) where
images and texts are concatenated and fed into the encoder to
extract representations. This reduces the number of parameters
but introduces a large number of computations.
• Dual-stream framework: Although this framework may lack the
ability to model complex interactions across modalities, its independent encoders demonstrate remarkable performance in image–
text retrieval tasks. For example, CLIP utilizes 400 million image–
text pairs trained with cross-modal contrast loss, enabling its
visual encoder to contain textual semantic information.
• Mixed architecture: This approach first extracts and aligns the
representations of respective modalities with a dual-stream architecture, then feeds the image and text features into a multimodal
encoder for fusion. ALBEF (J. Li et al., 2021) employs this architecture, and BLIP (Li et al., 2022) proposes a multimodal
Mixture of Encoder–Decoder (MED) with both understanding and
generation capabilities for deeper interaction between image and
text modalities.

𝐿𝑠𝑑𝑚 =

𝐵 𝐵
𝐵 ∑
𝐵
𝑝𝐼2𝑇
𝑝𝑇𝑖,𝑗2𝐼
∑
𝑖,𝑗
1 ∑ ∑ 𝐼2𝑇
(
𝑝𝑖,𝑗 log(
)+
𝑝𝑇𝑖,𝑗2𝐼 log(
)),
𝐵 𝑖=1 𝑗=1
𝑦𝑖,𝑗 + 𝜖
𝑦𝑖,𝑗 + 𝜖
𝑖=1 𝑗=1

(1)

𝐼2𝑇 ∕𝑇 2𝐼

where 𝑝𝑖,𝑗
represents the proportion of the image-to-text or textto-image cosine similarity score in a mini-batch. 𝜖 is a small number
to avoid numerical problems, 𝐵 denotes the batch size and 𝑦𝑖,𝑗 is the
ground truth probability.
3.2. Local image embedding extraction
To explicitly leverage the fine-grained image information, most previous CNN-based works for text-based person search typically employ
hard horizontal slicing (Sun et al., 2018) to extract local visual features.
However, due to the characteristic of self-attention, using a Transformer as visual encoder inevitably integrates information from the
entire image. Thus, directly applying horizontal slicing to the output of
Vision Transformers (ViT) is suboptimal. According to the analysis of
attention distance in Ghiasi et al. (2022), certain attention heads in the
lower layers of ViT exhibit small attention distance, indicating some
degree of local attention. We suggest that in transformer-based networks, more consideration should be given to the impact of attention
mechanism on local feature extraction.
The masked attention was initially proposed by Veličković et al.
(2017) and became widely known when the Mask2Former model
(Cheng et al., 2021) adapted it as a constrained cross-attention module.
Inspired by advances in image segmentation (Jiao et al., 2023; Xu
et al., 2023; Cheng et al., 2021), we propose a novel Attention Biasbased Forward process (ABF) as illustrated in Fig. 3(a). Similar to
Mask2Former, ABF also extracts local features by modulating the attention matrix to limit attention within specific regions. The difference
is that Mask2Former introduces masked attention to alleviate the
slow convergence of query features caused by global context in the
cross-attention layer. It enhances the sensitivity of query features to
foreground information, aiming to extract region proposals of specific
types from an image. Whereas, our proposed ABF leverages global
contextual information to mine additional semantic clues within the
specified regions through unidirectional information transfer of local
patch sets.
Specifically, ABF does not modify the forward process of the first
𝐿 layers in the visual encoder, allowing the [CLS] token to capture

Some recent re-ID works, such as Yang et al. (2023), Zuo et al.
(2023), Jin et al. (2025) and Shu et al. (2021), attempt to put the spotlight on pre-training a model from scratch. These approaches capture
more fine-grained associations by constructing large-scale pedestrian
datasets and employing pre-training tasks related to alignment targets.
Given that pretraining a model from scratch is too expensive, we adopt
the CLIP model to initialize the encoders and fine-tune it entirely on
the TBPS task.
3. Method
In this section, we present our proposed CDFM framework. The
overview of CDFM is illustrated in Fig. 2, and the details are discussed
in the following subsections.
3.1. Revisiting CLIP’s dual-encoder and global alignment
With the advancement of VLP models, recent studies (Yan et al.,
2022; Jiang and Ye, 2023) have attempted to transfer the knowledge
of CLIP to text-based person search. We initialize the CDFM with the
full CLIP image and text encoders to leverage its powerful cross-modal
alignment capability.
Image Encoder. Given an input image 𝐼 ∈ 𝑅𝐻×𝑊 ×3 , we first divide
it into 𝑁 = 𝐻 × 𝑊 ∕𝑃 2 non-overlapping patches, where 𝑃 denotes the
size of each patch. These patches are then flattened and prepended
with a learnable [CLS] token to form an input sequence. We adopt
a 12-layer Vision Encoder to model correlations among the patches.
3


codex
前三篇的共性已经很明显：论文会把“已有强预训练模型不够细”包装成一个具体缺口，然后用很轻的监督重组织或细粒度挖掘来讲机制。现在继续读可见光红外、噪声和生成去衣服偏置这组。
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
/bin/zsh -lc "pdftotext -l 3 'CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 177 (2026) 113333

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

CLIP -powered modality centering with spiral training for visible-infrared
person re-identiﬁcation
Jianghao Xiong
a

a , Xiaohua Xie

a,b,c,d , Qinyu Feng

e , Jian-Huang Lai

a,b,d,∗

School of Computer Science and Engineering, Sun Yat-Sen University, Guangzhou, 510006, China

b Pazhou Lab (Huangpu), Guangzhou, 510555, China
c

Guangdong Province Key Laboratory of Information Security Technology, Sun Yat-Sen University, Guangzhou, 510006, China

d Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, Sun Yat-Sen University, Guangzhou, 510006, China
e

Digital Grid Research Institute, CSG, Guangzhou, 510700, China

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Visible-infrared
Modality alignment
CLIP

a b s t r a c t
Cross-modal person re-identiﬁcation poses signiﬁcant challenges due to the need for aligning features across
diﬀerent modalities, speciﬁcally visible and infrared in this context. Given CLIP’s powerful cross-modal learning
capabilities, we explore its potential to bridge these modality gaps. This paper introduces the Modality Centering
with Spiral Training Network (MCST). We enhance text prompts by employing separable descriptions to independently capture personal and modality-speciﬁc information, thus disentangling identity-speciﬁc features from
modality-speciﬁc ones at the semantic level. For optimizing the image encoder, we utilize only personal text
descriptions to guide identity-aware feature learning. To align features across modalities for the same identity,
we propose a text-text centering loss to minimize distance between visible and infrared text representations, and
an image-text centering loss to reduce discrepancies between image and text features. In addition, we introduce a
novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
realistic evaluation benchmark. Extensive experiments demonstrate that our approach achieves state-of-the-art
performance on multiple datasets. The code and dataset are available at https://github.com/WhollyOat/MCST.

1. Introduction
Visible-infrared person re-identiﬁcation (V-I ReID) focuses on identifying individuals across non-overlapping camera views that operate in
both visible and infrared modalities. By utilizing these complementary
modalities, V-I ReID facilitates reliable identity recognition under diverse lighting conditions, ranging from daylight to nighttime. This technology is particularly valuable in applications such as security surveillance, public safety, and intelligent transportation systems, where robust
person identiﬁcation is crucial across varying environmental conditions.
Due to the signiﬁcant diﬀerences in spectral characteristics and
distribution properties between visible and infrared images, directly
matching images from these two modalities is challenging. As a result,
V-I ReID primarily addresses the issue of modality diﬀerences [1–3].
Traditional V-I ReID methods focus on extracting image features from
both modalities and aligning the visible-image and infrared-image feature spaces.

As CLIP-ReID [4] integrate CLIP [5] into image-image ReID, a new
paradigm has been unlocked. By leveraging the capabilities of large
models to understand image semantics, the model can describe images
using text, focusing on the features of diﬀerent parts of the human body.
These textual descriptions then guide the model to extract discriminative image features. This approach does not require manual annotation
of text and simplify the training process. Instead, it harnesses the interpretive power of large models, oﬀering greater ﬂexibility and potential.
Despite the remarkable success of CLIP-based ReID methods [6,7] in
visible scenarios, their application to V-I ReID remains largely underexplored and presents unique challenges. Unlike conventional V-I ReID
frameworks that only consider two visual modalities, CLIP-based methods introduce an additional semantic modality through text prompts,
which expands the representation space from two modalities (visible
and infrared images) to four heterogeneous feature spaces, namely visible image, infrared image, visible text, and infrared text. The coexistence
of these heterogeneous spaces signiﬁcantly increases the complexity of

∗ Corresponding author.

E-mail addresses: xiongjh7@mail2.sysu.edu.cn (J. Xiong), xiexiaoh6@mail.sysu.edu.cn (X. Xie), fengqy1@csg.cn (Q. Feng), stsljh@mail.sysu.edu.cn (J.-H. Lai).
https://doi.org/10.1016/j.patcog.2026.113333
Received 9 January 2025; Received in revised form 13 February 2026; Accepted 16 February 2026
Available online 19 February 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 177 (2026) 113333

J. Xiong et al.

cross-modal alignment, and naive pairwise alignment strategies are insuﬃcient to achieve identity-preserving modality-invariant representations.
Furthermore, existing CLIP-based V-I ReID methods typically treat
textual prompts as static semantic anchors, which leads to semantic
drift when the image encoder is further optimized. In such a two-stage
training paradigm, text prompts are derived from initial image representations and remain ﬁxed during subsequent training, causing inconsistency between textual and visual embeddings as the image features
evolve. In addition, prior studies mainly focus on aligning visual modalities while overlooking the modality bias embedded in textual representations, which may further amplify cross-modal discrepancies.
These challenges indicate that existing CLIP-based V-I ReID methods lack a principled mechanism to jointly model heterogeneous multimodal feature spaces and dynamically maintain semantic consistency
during training. Therefore, a uniﬁed framework that can simultaneously
align visible image, infrared image, and their corresponding textual representations, while dynamically updating multi-modal representations,
is highly desirable.
Motivated by this insight, we propose a novel Modality Centering
with Spiral Training Network (MCST) for V-I ReID, as illustrated in
Fig. 1. The MCST framework comprises a Modality Centering (MC)
strategy to mitigate modality discrepancies and a Spiral Training (ST)
scheme to alternately updates of text prompts and image features.
Speciﬁcally, based on CLIP-ReID, MC enhances the text prompt by reducing the disparity between representations of visible and infrared text.
Regarding image encoder training, MC promotes proximity between representations of visible text and infrared images, and reciprocally, between infrared text and visible images. This strategy aims to converge

the four feature spaces towards a central modality. To maintain the
similarity between the text prompt and image features, ST dynamically
trains both the text prompt and the image encoder in alternating phases,
training one part for a small number of epochs before switching to the
other. Additionally, separable text prompts are utilized at the text level
to decouple features, continuously stripping away modality-speciﬁc features to extract modality-invariant features.
To advance research in V-I ReID, we introduce a large-scale imagebased visible-infrared ReID dataset, CMG-P. Compared to existing V-I
ReID datasets, CMG-P oﬀers two key advantages. The ﬁrst is its realworld diversity. The images are sourced from actual pedestrians and
include a wide range of challenging scenarios such as pedestrian occlusion, scale variation, clothing changes, and motion blur. This variety allows the dataset to better simulate complex real-world conditions, which
provides signiﬁcant beneﬁts for V-I ReID research. Additionally, CMG-P
contains 36,031 visible images and 36,144 infrared images of 1011 individuals, which is one of the most extensive datasets in the ﬁeld. Further
details about the dataset are provided in Section 5.1.1.
The main contributions of our work can be summarized as follows:
•

We propose a novel Modality Centering (MC) framework for VI ReID, which jointly aligns four heterogeneous feature spaces
(visible/infrared images and texts) toward a shared identitydiscriminative embedding space, addressing both cross-modal and
text-image modality gaps.
• We introduce a Spiral Training (ST) scheme that dynamically updates text prompts and image encoders, overcoming the semantic
drift problem caused by static prompts and maintaining consistent
multi-modal representations during training.

Fig. 1. Comparison of CLIP-ReID and our method. (a) CLIP-ReID uses a two-stage approach, ﬁrst optimizing the implicit text prompt and then training the image
encoder. (b) Our method for VI-ReID integrates text prompts to create and align four distinct feature spaces, i.e visible image, infrared image, visible text, and infrared
text. We connect these feature spaces, and alternately train the text prompt and image encoder.

2

Pattern Recognition 177 (2026) 113333

J. Xiong et al.
•

We design separable text prompts to disentangle identity-speciﬁc and
modality-speciﬁc semantics, enabling explicit removal of modality
bias at the semantic level and facilitating modality-invariant representation learning.
• We construct a large-scale V-I cross-modality ReID dataset ReID
dataset CMG-P, featuring clothing changes, occlusions, and realworld noise, which provides a challenging benchmark for future V-I
ReID research.

by CLIP, primarily targeting text-image ReID. In contrast, the latter
bypasses pre-annotated text and generates implicit text prompts using
prompt engineering techniques like CoOp [20], leveraging text features
to guide image feature alignment, mainly for image-image ReID. CLIPReID [4] is a pioneering method that utilizes CLIP’s vision-language
capabilities to learn discriminative image representations without explicit text labels, aligning visual features with learnable text prompts.
TF-CLIP [18] learns a text-free shared feature space from video data
for video-based ReID, enabling eﬀective cross-view and temporal feature alignment without relying on textual descriptions. PromptSG [19]
is an end-to-end method that generates text prompts from image features using an inversion network and employs a language-guided crossattention module to perform text-image semantic guidance. The use of
CLIP for VI-ReID has been studied in [21] and [22]. CLIP-MC [21] is
a one-stage method that enhances modality alignment by generating
ﬁne-grained image descriptions directly from image features to compensate for modality text context. It fuses original image features with
the compensated text features to create more robust representations.
CSDN [22] follows the CLIP-ReID pipeline by optimizing complementary language descriptions for visible and infrared images, integrating
them via an attention fusion module. The fused text representation then
guides modality-invariant image feature learning. While CLIP-MC and
CSDN employ an entangled approach to integrate modality information into the ﬁnal representation-potentially compromising discriminative power-our method adopts a disentangled strategy. By separating
modality information through distinct text descriptions and aligning
four feature spaces, we preserve modality-invariant features essential
for accurate person retrieval. Furthermore, our multi-stage framework
dynamically adapts to diverse datasets while preserving tight alignment
between text and image features.

2. Related work
2.1. Visible-infrared person re-identiﬁcation
To bridge the gap between visible and infrared modalities, current
methods can generally be divided into feature alignment [8–10] and
modality conversion [11–13] approaches. Feature alignment aims to
project the features of visible and infrared modalities into a shared feature space to minimize the modality gap. SEFL [8] eliminates identityspeciﬁc shape information from features and allows the model to focus on more general and modality-invariant appearance cues. HOS-Net
[9] uses high-order structural relationships and middle-layer features to
capture rich cross-modal discriminative information. IDKL [10] aligns
multi-modal features in a shared space through implicit discriminative knowledge learning. In contrast, modality conversion seeks to reduce cross-modal diﬀerences by transforming images from one modality
into the other or a third modality. RBDF [11] employs two generative
networks that serve as inverse mappings to learn to transform visible
images into infrared images and vice versa. By incorporating an additional association loss to bridge the gap between the intermediate representations of the two modalities. PartMix [12] generates augmented
samples by mixing part descriptors and learns to identify discriminative body parts across modalities. CAJ+ [13] augments spectral channels and eﬀectively integrate complementary information from visible
and infrared modalities. Both approaches aim to overcome modality
discrepancies and extract modality-invariant human semantics. However, feature alignment often struggles to selectively preserve identitydiscriminative semantics while reducing modality diﬀerences. Meanwhile, modality conversion is non-directional and tends to introduce
additional noise during the transformation process, which can adversely
impact model training. Our method combines these two approaches, utilizing CLIP’s ability to understand images to generate text modality related to the image semantics. The text modality serve as a bridge to help
align features between the visible and infrared domains.
In addition to supervised approaches, unsupervised V-I ReID has
attracted increasing attention because it reduces the reliance on
costly cross-modality identity annotations. ADCA [14] introduces an
augmented dual-contrastive aggregation framework that maintains
modality-speciﬁc memories and performs cross-modality memory aggregation at the cluster level to associate identities. From a diﬀerent
perspective, CHCR [15] adopts a hierarchical clustering and reﬁnement
strategy to progressively establish reliable pseudo labels across modalities under large modality discrepancies. Although these works focus on
label-free training, the CLIP-guided text modality in our method provides auxiliary cross-modal alignment signals and can be naturally integrated with pseudo-label or clustering-based pipelines, making our
framework compatible with future label-eﬃcient or semi-supervised extensions.

2.3. Vision-language models
Vision-Language Models (VLMs) are powerful models focused on integrating visual and linguistic understanding. VLMs align visual features
and language embeddings using large-scale image-text datasets, eﬀectively bridging the gap between visual and textual modalities to solve a
wide range of multi-modal tasks. CLIP [5] is a leading example of a VLM.
It uses contrastive learning to map images and text into a shared semantic space, allowing zero-shot reasoning across diverse vision-language
tasks. Other prominent VLMs, such as ALIGN [23], Flamingo [24], BLIP2 [25], BEIT-3 [26], and LLAVA [27], have demonstrated state-of-the-art
performance on tasks such as image captioning, visual question answering (VQA), cross-modal retrieval, and others.
3. Preliminary
3.1. Contrastive language-image pre-training (CLIP)
CLIP is trained on a large-scale dataset of diverse image-text pairs,
enabling it to learn a uniﬁed representation for both images and text.
This shared representation allows CLIP to align visual and textual information seamlessly and facilitates a common understanding between
the two modalities. Speciﬁcally, CLIP consists of two separate encoders:
an image encoder (⋅) and a text encoder  (⋅). The image encoder (⋅)
typically utilizes a Vision Transformer [28] or ResNet [29] architecture. Given an input image 𝑥 ∈ ℝ𝐻×𝑊 ×𝐶 , its feature representation is
extracted by (⋅) and then projected into the multi-modal embedding
space using a linear projection 𝐼 , i.e, 𝑓𝑥 = 𝐼 ((𝑥)). Similarly, the text
embedding 𝑓𝑡 of an input text 𝑡 is computed as 𝑓𝑡 = 𝑇 ( (𝑡)), where  (⋅)
is a Text Transformer and 𝑇 is a linear projection. For a batch of 𝑁
image-text pairs, the probability that 𝑥 and 𝑡 form a correct pair can be
computed by

2.2. CLIP-based person re-identiﬁcation
CLIP can augment visual information with textual features to address
the challenges of diverse ReID tasks. Existing CLIP-based ReID methods can be divided into explicit text descriptions [16,17] and implicit
text prompts [4,18,19] methods. The former involves obtaining text descriptions for each image through manual annotation or text generation
models, followed by aligning text features with image features extracted

exp(sim(𝑓𝑥 , 𝑓𝑡 )∕𝜏)
𝑃 (𝑦𝑡 |𝑥) = ∑𝑁
,
𝑘=1 exp(sim(𝑓𝑥 , 𝑓𝑡𝑘 )∕𝜏)
3

(1)


exec
/bin/zsh -lc "pdftotext -l 3 'Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10890718

Cloth-debiasing with Stable Diffusion in
Cloth-changing Person Re-identification
Haiyang Zhang

Xinshuang Wang

School of Computer Science
Beijing University Of Posts and Telecommunications

School of Computer Science
Beijing University Of Posts and Telecommunications

Abstract—In the current study of cloth-changing person reidentification (CC-ReID), the misidentification rate is significantly
high for different individuals wearing similar attire, due to biases
in clothing features. The generation model can unify the clothing
feature space, minimizing interference from clothing color and
type, and enabling model to concentrate on extracting clothingirrelevant features. However, the current use of Generative
Adversarial Networks (GANs) for changing clothes in CC-ReID
faces challenges in addressing complex discrepancies between the
generated images and the original images, resulting in unstable
outcomes when changing the same clothing for pedestrians with
varying postures and clothing types. Consequently, we generate
cloth-changing pedestrian images with consistent clothing based
on a stable diffusion model controlled by body keypoint information, ensuring the images conform to the geometric structure
of human bodies. Additionally, we employ threshold filters to
refine these images, aiming to construct a high-quality CC-ReID
dataset with consistent clothing styles. Meanwhile, we improve
the CC-ReID model by introducing centroid loss to increase
inter-class differences, thereby maximizing the model’s ability
to distinguish between pedestrians wearing similar clothing. Extensive experiments demonstrate that our approach outperforms
previous methods, achieving a 6% increase in Rank-1 and a 4.4%
increase in mAP on the PRCC dataset compared to the baseline.
Index Terms—cloth-changing person re-identification, stable
diffusion

I. I NTRODUCTION
Person re-identification (ReID) aims to identify individuals in surveillance videos across various locations and time
periods. Most existing studies [1]–[6] assume pedestrians’
clothing remains unchanged over short duration. However, the
challenge of changes in clothing not only exists in identifying a
pedestrian over a long period but also arises in some short-time
realistic scenarios, for instance, criminal suspects often change
their clothes to avoid being identified and tracked. Due to its
crucial role in intelligent monitoring systems, cloth-changing
ReID has garnered increasing attention in recent years.
Humans can recognize acquaintances, even if those acquaintances are wearing clothes that they have never seen before,
because human brains can decouple and make use of clothirrelevant features, such as body shape and gaits. Similarly in
CC-ReID, if the data is sufficiently complex, the data-driven
model can adapt to the variations in-class samples, and learn
cloth-irrelevant and discriminative features. However, humans
tend to identify strangers based on their different clothes. It
is quite similar to the way neural networks learn features of

pedestrians. In realistic scenarios of CC-ReID, a lack of data
diversity leads to classification bias during training. The model
tends to rely on the most easily distinguishable features for
recognition, which misleading the model to focus on nonidentifying parts that seem to be related to IDs and assign
high scores to images of people with similar clothing.
To mitigate this bias, the generative model can be used
to unify the clothing feature space, and change the clothing
of pedestrians to a consistent color and type. By utilizing
similar clothing data of pedestrians for training can reduce
the interference of clothing-related features and enhance the
model’s ability to distinguish pedestrians wearing similar
clothing. The most current cloth-changing method in CCReID involves GAN models [7]–[9]. In GANs, the inadequate
coverage of the feature space leads to generated samples that
are unable to fully capture the complex variation patterns
in high-dimensional clothing features. If there is a complex
variations, the generation outcomes may not be optimal. When
changing a pedestrian’s clothing type or altering the attire
of pedestrians with uncommon poses to generate consistent
clothing data, GANs tend to retain most of the original features
of pedestrian clothing and may generate low-quality images
with distorted postures, which interferes with ReID model
training.
To address the limitations of using GANs for clothingchange in CC-ReID, referring to the application of diffusion
model in virtual try-on field [10], we propose adopting a generation scheme based on the stable diffusion model to ensure
the quality of the generated clothing-change data. By gradually
varying during the generation process, the diffusion model [11]
can more stably control the geometric shapes and details of
the generated images than GANs [12]. We use segmentation
masks of pedestrians as masks for the inpainting method in
stable diffusion, ensuring the non-clothing parts of the human
body remain unchanged. In addition, the pose keypoints of
pedestrians are used to control the generated clothing-change
images, maintaining their original poses. Threshold filters
are used to select high-quality training data with consistent
clothing. To enhance the model’s ability to distinguish pedestrians wearing similar clothing, we introduce centroid loss
[13]. By reducing the distance between sample feature vectors
and their class centroids, centroid loss improves intra-class
compactness, ensuring same-class samples are closer in feature
space. This, in turn, amplifies differences between classes,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.

Fig. 1. Overall architecture of the proposed method. On the left, auxiliary information is generated to assist with the cloth-changing process. In the center,
the generate-filter clothing-change module(GFCC) produces pedestrian images with consistent clothing. On the right is the structure of our CCAL model that
needs to be trained.

enhancing inter-class separability. Without centroid loss, class
feature vectors may scatter and overlap, leading to confusion.
Our contributions can be summarized as three folds: (1) A
cloth-changing scheme based on the stable diffusion model is
employed to generate consistent clothing for pedestrians. The
masks for the stable diffusion inpainting method are derived
from body segmentation maps to maximize the retention
of human features, excluding clothing. To ensure that the
generated images conform to the geometric structure of human
bodies, the generation process is controlled by pedestrian
body keypoint information. (2) The threshold filters are used
to eliminate low-quality generated images, allowing us to
construct high-quality consistent clothing CC-ReID datasets.
(3) The consistent clothing CC-ReID datasets are used to train
the enhanced CC-ReID model, which introduces the centroid
loss. Experimental results demonstrate that our approach outperforms previous methods by a significant margin.
II. M ETHODOLOGY
A. Auxiliary information generation
The body segmentation map and pose keypoints of pedestrians are required to assist the generating phase, as illustrated in
the auxiliary information generation module of Fig.1. We use
self-correction human parsing(SCHP) model [14] to generate
segmentation masks that represent body parts and clothing
items. These segmentation maps serve as masks for the
inpainting process in II-B. We utilize the pre-trained SCHP
model on the ATR dataset to extract human parsing representations. The model includes 18 labels for body segmentation
masks, such as ’Upper-clothes’, ’Left-leg’, ’Right-leg’, ’Leftarm’, ’Right-arm’, ’Bag’ and so on. Additionally, we utilize
OpenPose model [15] to extract 18 pose keypoints of each
pedestrian in the images. The keypoints is used in II-B to
control the generated images with a natural pose.

B. Generating phase
The generating phase of the generate-filter clothing-change
module(GFCC) adopts LaDI-VTON method [10], as illustrated in Fig.1. The expanded Stable Diffusion inpainting
pipeline of LaDI-VTON is used to change clothing for pedestrian images in CC-ReID.
To be specific, the original spatial input γ of is formed by
1×h×w
concatenating a binary inpainting mask m ∈ {0, 1}
, the
masked image E( ˆ
I) which is encoded into a latent representation, and the denoising network input zt along the channel
dimension. Additionally, the spatial input γ ∈ R9×h×w is
expanded by appending two other components:the resized pose
map p ∈ R18×h×w and the encoded clothing E(Ĉ) ∈ R4×h×w .
The final spatial input of the inpainting denoising network is:
ˆ p; E(Ĉ)] ∈ R(9+18+4)×h×w
γ = [zt ; m; E(I);

(1)

Iˆ represents the pedestrian image I masked by a mask
1×H×W
M ∈ {0, 1}
. The mask m is resized from M . The
original inpainting mask M is derived from the segmentation
map of pedestrian. The inpainting area is specified by the mask
M . To ensure sufficient coverage of the intended clothing
regions, the method proposed in prior work [16] is applied.
Additional inputs are utilized to condition the inpainting
pipeline, including the textual prompt describing the clothing
[10], the pose map P , and the clothing Ĉ which is warped
according to the pedestrian body shape. The warped clothing
Ĉ is generated through a geometric matching module [17]
and a U-Net refinement model [18]. The geometric matching
module determines the correlation between the clothing C and
a clothing-irrelevant pedestrian representation, which includes
Iˆ and P , generating parameters θ. A thin-plate spline transformation [19] generates the coarse warped clothing C ′ from the
clothing C by C ′ = T P Sθ (C). The U-Net refines the warped
ˆ
clothing by Ĉ = U net(C ′ , P, I).

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.

C. Filtering phase
To address the issue of low-quality generated cloth-changing
pedestrian images, which may impact ReID performance, we
use two threshold filters, shown in Fig.1.
1) Similarity Threshold Filter: This filter compares the
ReID features of generated images with those of original
images, filtering based on similarity scores [20]. Generated
images with higher scores are retained, while those with lower
scores are discarded.
2) FID Threshold Filter: We divide the generated images
into small batches and compute the FID [21] value for each
batch compared to the real image set. If a batch’s FID value
exceeds the set threshold, it is discarded. To enhance dataset
quality, we refine the process by regrouping remaining images
into new batches and recalculating FID values.
Additionally, low-quality images are removed through manual inspection to ensure the final dataset meets high standards.
D. Centroid loss for CC-ReID
Our CCAL model utilizes the CAL model [22] as the
baseline, and introduces the centroid loss in prior work [13]
for training. Specifically, the L2 loss is computed using the
centroids. For each pedestrian identity i in the batch, the
samples are divided into two groups: PA , which includes
all samples of ID i, and PB , which comprises the samples
which are not belonging to ID i. The centroid embedding of
both groups is computed. The loss function is defined as the
euclidean distance between the centroids of the two groups.
This loss for each ID in the batch is summed. Within a batch,
each ID i has N samples. f (n) and Xn respectively represent
the ID of sample n and its feature embedding. PA for ID i
consists of N samples, and PB of this ID contains M samples.
The centroids for both clusters of ID i are derived as follows:
X
1
·
(Xn )
(2)
PA (i) =
N
n,f (n)=i

PB (i) =

1
·
M

X

(Xn )

(3)

n,f (n)̸=i

The centroid loss for id i is defined as:
LP (i) = ||PA (i) − PB (i)||2
For a batch, the centroid loss LP is calculated as
The final loss function is as follows:
L = αLcal + βLP

(4)
P

i LP (i).

Fig. 2. The examples show the effects of changing the clothing for pedestrians
with different postures and different original clothing types.

cameras. We report the recognition accuracy for PRCC in both
same clothes setting and clothing-change setting. For LTCC,
we provide the accuracy for general ReID and CC-ReID.
B. Implementation details
In the generating phase, we randomly select several images
of each pedestrian and change their clothing to a set of long
sleeves and pants, which cover most body areas (e.g., arms
and legs) and allow the model to focus on pedestrian features
that are typically visible despite clothing, such as the head and
neck, while retaining human contours. We use stable diffusion
inpainting pre-trained model [25] with 20 inference steps and
3 batch size. The EMASC module [10] is used to enable
better detail reconstruction in inpainting task. Example clothchanging images are shown in Fig. 2. After filtering, we add
4556 generated pedestrian images to the PRCC dataset and
1310 images to the LTCC dataset.
We train our model on our consistent clothing datasets. Following the CAL [22], ResNet50 [26] pre-trained on ImageNet
is involved in our ReID model, with its final downsampling
removed for enhanced detail. We utilize global average pooling
and global max pooling to combine the output feature map of
the backbone, which are then concatenated and normalized
using BatchNorm [27]. α and β of (5) in II-D are set as 1
and 0.1, respectively. We add Side Information Embeddings
(SIE) [28] layer to eliminate the interference of camera-related
features for LTCC dataset. The input images are resized to
384 × 192 with random horizontal flipping, cropping, and
erasing [29] . The batch size is set to 32. Our model is trained
for 200 epochs using the Adam [30] optimizer, applying
centroid loss between the 10th and 25th epoch, and CAL loss
after the 25th epoch. The learning rate is initialized to 3e−4
and divided by 10 after every 20 epochs.
C. Comparison with state-of-the-art methods
We compare our method with several existing ReID methods on LTCC and PRCC datasets. For the PRCC dataset

(5)

The parameters α and β are empirically chosen weights.
Lcal represents the original loss in our baseline.
III. E XPERIMENT
A. Datasets
We evaluate our approach on two widely used datasets
in CC-ReID: PRCC [23] and LTCC [24]. PRCC comprises
33698 images of 221 individuals captured by 3 cameras.
LTCC contains 17,138 images of 152 individuals taken by 12

TABLE I
C OMPARISON WITH SOTA METHODS ON PRCC DATASET (%).
Method
IANet [31]
RCSANet [32]
AIM [33]
DLAW [34]
IRM [35]
CAL(Baseline) [22]
Our

Standard
Rank-1 mAP
99.4
98.3
100
97.2
100
98.42
98.7
100
99.8
100
99.8

Cloth-Changing
Rank-1 mAP
46.3
45.9
50.2
48.6
57.9
58.3
56.2
57.1
54.2
52.3
55.2
55.8
61.2
60.2

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113873

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

CLNS: Camera-aware label noise suppression for unsupervised
visible–infrared person re-identification
Sicheng Zhao a , Wei Lu a , Shihao Li b , Sibao Chen a
Jin Tang a , Bin Luo a

,∗, Chris H.Q. Ding c , Futian Wang a ,∗,

a

The MOE Key Laboratory of ICSP, IMIS Laboratory of Anhui, Anhui Provincial Key Laboratory of Multimodal Cognitive Computation, Zenmorn-AHU AI Joint
Laboratory, School of Computer Science and Technology, Anhui University, Hefei, China
b The school of Artificial Intelligence, Anhui University, Hefei, China
c The School of Data Science (SDS), Chinese University of Hong Kong, Shenzhen 518172, China

ARTICLE

INFO

Keywords:
Unsupervised person re-identification
Cross-modality
Contrastive learning
Noisy labels

ABSTRACT
Unsupervised visible–infrared person re-identification (US-VI-ReID) retrieves pedestrian images across modalities without manual annotations. To address camera-specific biases that fragment identities and amplify label
noise, we propose the Camera-aware Label Noise Suppression (CLNS) framework, a coarse-to-fine pipeline
that progressively purifies noise. Specifically, the Camera-aware Prototype Calibration (CPC) module exploits
cross-camera consistency to rectify structural errors and construct reliable prototypes. Building on this,
Optimal Transport Prototype Matching (OTPM) achieves global cross-modality alignment. At the instance
level, Neighbor-guided Camera-domain Learning (NCL) densifies feature distributions using soft supervision,
while a Noise-aware Memory Updating (NMU) strategy prevents error accumulation. On the SYSU-MM01,
RegDB, and LLCM datasets, CLNS achieves Rank-1 (mAP) accuracies of 69.5% (65.6%), 94.8% (91.8%), and
54.6% (57.8%), respectively, significantly surpassing state-of-the-art methods. The code will be released at
https://github.com/apatientzsc/CLNS.

1. Introduction
Visible–Infrared Person Re-identification (VI-ReID), as investigated
in recent work [1], matches pedestrian images across visible and infrared modalities, playing a pivotal role in intelligent surveillance.
While supervised approaches have achieved remarkable progress, they
rely heavily on large-scale, labor-intensive annotations [2–6]. Consequently, Unsupervised VI-ReID (US-VI-ReID) has garnered increasing
attention for its scalability and real-world applicability. Pioneering
studies, such as those by Wu et al. [7] and Ye et al. [8], aim to learn
discriminative representations without manual labels.
A central challenge in US-VI-ReID lies in generating reliable pseudolabels. Mainstream approaches typically employ clustering algorithms
(e.g., DBSCAN) to generate pseudo-labels from global feature similarities, as seen in PGM [9]. While effective in general scenarios, these
methods often overlook a critical source of error: camera-specific bias.
As illustrated in Fig. 1, images of the same identity captured by the
same camera often exhibit higher visual similarity than those from
different cameras due to background and viewpoint consistencies. To
quantitatively illustrate the severity of this issue, we conducted a

preliminary analysis on the SYSU-MM01 dataset using the baseline
method. Specifically, the average cosine distance of inter-camera positive pairs (images of the same identity from different cameras) is up to
1.0615, which is significantly higher than that of intra-camera positive
pairs (0.7795). This camera-induced discrepancy misleads clustering
algorithms into splitting a single identity into multiple camera-specific
clusters (identity fragmentation) or merging distinct identities. Such
noise not only corrupts the memory bank but also severs cross-modality
correspondences, severely undermining training effectiveness.
Although several label refinement strategies [10,11] have been
proposed, they mostly operate on the final clustering results without
explicitly modeling or suppressing the underlying camera bias. Lacking this constraint, they are easily misled by the high intra-camera
similarity inherent in surveillance data, often exacerbating identity
fragmentation instead of resolving it. To address these challenges, we
propose Camera-aware Label Noise Suppression (CLNS), a synergistic
framework designed to progressively purify camera-induced noise from
a coarse-to-fine perspective. Our method operates through a coherent

∗ Corresponding authors.

E-mail addresses: 2069923681@qq.com (S. Zhao), luwei_ahu@qq.com (W. Lu), shli0603@foxmail.com (S. Li), sbchen@ahu.edu.cn (S. Chen),
chrisding@cuhk.edu.cn (C.H.Q. Ding), wft@ahu.edu.cn (F. Wang), tangjin@ahu.edu.cn (J. Tang), luobin@ahu.edu.cn (B. Luo).
https://doi.org/10.1016/j.patcog.2026.113873
Received 24 December 2025; Received in revised form 17 April 2026; Accepted 27 April 2026
Available online 30 April 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113873

S. Zhao et al.

Fig. 1. Impact of camera-induced label noise. (a) Intra-camera visual similarity causes identity fragmentation, leading to (b) unreliable prototypes and (c) incorrect
cross-modality alignment. In contrast, (a’–c’) CLNS calibrates camera biases to construct robust prototypes, establishing accurate correspondences.

pipeline. Specifically, we first introduce a Camera-aware Prototype Calibration (CPC) module. By exploiting cross-camera neighborhood consistency, CPC filters out unreliable samples to rectify structural errors
and construct robust prototypes. Subsequently, the Optimal Transport
Prototype Matching (OTPM) module formulates cross-modality alignment as a global optimal transport problem, establishing precise correspondences that are unattainable with noisy prototypes. Subsequently,
to refine feature distributions at the instance level, the Neighborguided Camera-domain Learning (NCL) module utilizes confidenceaware soft labels to suppress residual distribution noise. Finally, a
Noise-aware Memory Updating (NMU) strategy adaptively re-weights
sample contributions to prevent error accumulation in the memory
bank.
Our main contributions are as follows:

modality gap. Ye et al. [15] proposed a channel-level perturbation strategy (CA) to improve robustness against color variations, while Chen
et al. [16] developed a dual-modality alignment network to generate
coherent intermediate features. Despite their remarkable performance,
the heavy reliance on large-scale manual annotations restricts the
scalability of S-VI-ReID in real-world deployments, motivating the shift
towards unsupervised paradigms.
2.2. Unsupervised visible–infrared person re-ID
Unsupervised VI-ReID (US-VI-ReID) [9] aims to learn discriminative representations without identity labels. The mainstream paradigm
has evolved from GAN-based translation to clustering-based pseudolabeling.
To improve the quality of generated pseudo-labels, recent studies have focused on refining the clustering process. Shi et al. [17]
introduced a multi-memory matching (MMM) framework, which leverages multiple memory banks to capture diverse feature representations. Yang et al. [18] proposed SDCL, a shallow-to-deep collaborative
learning approach that progressively mines fine-grained relations from
coarse clusters. To further enhance label reliability, Dai et al. [10]
integrated similarity-based refinement strategies to correct noisy assignments during training. Beyond simple clustering, establishing robust cross-modality correspondence is critical for US-VI-ReID. Wang
et al. [19] pioneered the use of Optimal Transport (OT) in this field
(OTLA), formulating alignment as a global assignment problem to reduce semantic mismatching. Following this, Cheng et al. [20] extended
this idea with DOTLA, enforcing dual-level transport constraints. Furthermore, graph-based approaches have shown promise in modeling
intrinsic relationships. Wu et al. [9] proposed PGM, utilizing graph
optimization to refine instance-level affinities, while Ye et al. [21]
developed a dual-level matching strategy to filter outliers via graph
consistency.
However, these methods predominantly rely on global feature clustering, which is inherently sensitive to intra-class variance. Crucially,
they often overlook camera-specific bias, a primary factor that fragments identities into disjoint clusters. Without explicitly modeling this
bias, existing global alignment strategies often struggle to establish
accurate cross-modality matching.

• We propose the CLNS framework to address the camera-specific
bias in US-VI-ReID. By progressively purifying noise from coarselevel prototypes to fine-grained instances, CLNS ensures robust
cross-modality learning.
• We introduce the CPC and OTPM modules to act as structural
gatekeepers. CPC rectifies pseudo-label errors via cross-camera
consistency, while OTPM achieves global cross-modality alignment through optimal transport.
• We design the NCL module and NMU strategy to handle residual
noise. NCL compacts feature distributions using soft supervision,
and NMU prevents memory corruption from outliers.
Extensive experiments on SYSU-MM01, RegDB, and LLCM datasets
demonstrate that CLNS achieves state-of-the-art performance, validating the effectiveness of suppressing camera-aware noise.
2. Related work
2.1. Supervised visible–infrared person re-ID
Supervised Visible–Infrared Person Re-identification (S-VI-ReID) relies on cross-modality annotations to learn shared feature spaces. Existing approaches typically tackle the modality discrepancy through two
dominant paradigms: feature alignment and modality augmentation.
Feature alignment methods aim to bridge the modality gap by projecting features into a common subspace. For instance, Wang et al. [12]
proposed the D2RL framework, which optimizes feature embeddings
and modality translation jointly to reduce appearance discrepancies.
Similarly, Liu et al. [13] introduced MAUM to learn unidirectional
cross-modality metrics, effectively aligning distributions across different spectrums. More recently, Fang et al. [14] focused on fine-grained
alignment, utilizing semantic part prototypes to enhance local feature
invariance against modality shifts. Conversely, modality augmentation strategies generate intermediate representations to smooth the

2.3. Learning with noisy labels
Since unsupervised clustering inevitably introduces noise, mitigating noisy pseudo-labels is a central challenge. Existing solutions generally fall into two categories: label correction and noise-tolerant loss
functions.
Label correction methods aim to refine noisy targets using model
predictions or structural constraints. For example, Yin et al. [22] proposed RPNR, which exploits neighborhood consistency to select highconfidence samples for reliable cluster reconstruction. In their subsequent work, Yin et al. [23] introduced APPD, further enhancing
2

Pattern Recognition 179 (2026) 113873

S. Zhao et al.

Fig. 2. The proposed CLNS framework. It consists of five key components: (a) shows the initial clustering; (b) CPC calibrates camera-biased prototypes; (c) OTPM
aligns global cross-modality features; (d) NCL refines fine-grained instances; and (e) NMU updates memory based on reliability.

where |𝐶𝑖𝑡 | is the number of instances in cluster 𝑖. These prototypes are
updated using a momentum strategy:

label purification by adaptively adjusting the pseudo-label distribution.
Similarly, Pang et al. [24] utilized inter-channel validation in CHAR
to identify and eliminate unreliable labels based on cross-channel consensus. On the other hand, noise-tolerant loss functions aim to downweight the contribution of unreliable samples during optimization.
Fang et al. [25] employed a Robust Hybrid Loss (RHFL) to mitigate
the impact of hard outliers, while Yang et al. [26] designed DSCE to
adapt symmetric cross-entropy loss to evolving cluster structures.
Despite these advances, most existing works treat label noise as
instance-level outliers or feature-level inconsistencies. They largely
ignore the systematic camera-domain gap as a structural noise source.
Unlike random noise, camera-induced noise is structured and consistent, which requires specific handling. In contrast to previous approaches, our CLNS framework explicitly models and suppresses this
camera-induced noise, providing a more fundamental solution to label
refinement.

𝑡(𝛿−1)
𝑚𝑡(𝛿)
+ (1 − 𝛽)𝑓𝑞 ,
𝑞 ← 𝛽𝑚𝑞

(2)

where 𝛽 is the momentum coefficient, 𝑓𝑞 is the current feature, and 𝛿 is
the iteration index. The training objective consists of two components.
Intra-modality contrastive learning. For a query 𝑞 𝑡 , the InfoNCE loss pulls
it towards its assigned prototype 𝑚𝑡+ :
exp(𝑞 𝑡 ⋅ 𝑚𝑡+ ∕𝜏)
𝑡intra = − log ∑𝑁
,
𝑡
exp(𝑞 𝑡 ⋅ 𝑚𝑡𝑝 ∕𝜏)
𝑝=1

(3)

where 𝑁𝑡 is the number of clusters and 𝜏 is the temperature parameter.
Inter-modality contrastive learning. In the second stage, cross-modality
alignment is enforced by pulling the query 𝑞 𝑣 towards its matched
infrared prototype 𝑚𝑣→𝑟
+ :
exp(𝑞 𝑣 ⋅ 𝑚𝑣→𝑟
+ ∕𝜏)
𝑣inter = − log ∑𝑁
.
𝑟
𝑣 ⋅ 𝑚𝑟 ∕𝜏)
exp(𝑞
𝑗
𝑗=1

3. Method
3.1. Preliminaries and baseline

(4)

The total baseline loss is:
𝑏𝑎𝑠𝑒 = 𝑣intra + 𝑟intra + 𝑣inter + 𝑟inter .

Preliminaries. Let 𝑋 = {𝑋𝑣 , 𝑋𝑟 } denote the unlabeled datasets
from visible and infrared modalities, where 𝑋𝑣 = {𝑥𝑣1 , 𝑥𝑣2 , … , 𝑥𝑣𝑁 } and
𝑋𝑟 = {𝑥𝑟1 , 𝑥𝑟2 , … , 𝑥𝑟𝑀 }. We utilize a two-stream encoder with shared
parameters 𝜃 and modality-specific heads to extract features, denoted
𝑣 } and
𝑟 }. To generate
as 𝐹𝑣 = {𝑓1𝑣 , 𝑓2𝑣 , … , 𝑓𝑁
𝐹𝑟 = {𝑓1𝑟 , 𝑓2𝑟 , … , 𝑓𝑀
initial pseudo-labels, DBSCAN is applied independently on 𝐹𝑣 and 𝐹𝑟 ,
yielding cluster sets 𝑣 = {𝐶1𝑣 , 𝐶2𝑣 , … , 𝐶𝑆𝑣 } and 𝑟 = {𝐶1𝑟 , 𝐶2𝑟 , … , 𝐶𝐿𝑟 },
where 𝑆 and 𝐿 are the number of clusters. To explicitly model cameraspecific biases, we partition each cluster into finer camera domains
based on camera IDs. Let 𝑣 = {𝜙𝑣1 , … , 𝜙𝑣𝑆×𝐾 } and 𝑟 = {𝜙𝑟1 , … , 𝜙𝑟𝐿×𝐾 }
𝑣
𝑟
represent the sets of camera-domain centroids, computed by averaging
features within each camera-specific subset of a cluster, where 𝐾𝑣 and
𝐾𝑟 denote the number of cameras in each modality.
Baseline. We adopt CEIL [27] as our baseline. In the first stage,
CEIL generates initial pseudo-labels via independent clustering to initialize cluster prototypes:
1 ∑ 𝑡
𝑚𝑡𝑞 = 𝑡
𝑓 and 𝑡 ∈ {𝑣, 𝑟},
(1)
|𝐶𝑖 | 𝑠∈𝐶 𝑡 𝑠

(5)

Our proposed modules are integrated into the second stage to robustify this learning process. The overall framework of CLNS is illustrated
in Fig. 2.
3.2. Camera-aware prototype calibration
Standard clustering often yields noisy pseudo-labels. While neighborhood consistency strategies like RPNR [22] effectively filter random
noise, they fail in US-VI-ReID contexts. High visual similarity within
the same camera view (due to background and lighting) creates false
neighborhoods, causing standard methods to reinforce camera-specific
errors rather than correct them. To address this, we introduce the
Camera-aware Prototype Calibration (CPC) module, which exploits
cross-camera consistency to filter unreliable samples and construct
robust, camera-invariant prototypes.
Taking the visible modality as an example, let 𝑣 = {𝐶1𝑣 , … , 𝐶𝑆𝑣 }
denote the initial clusters. For a sample 𝑥𝑣𝑖 with feature 𝐟𝑖𝑣 , we suppress

𝑖

3


codex
这一组里更值得借的是“结构性噪声”与“四空间语义漂移”两种问题刻画；Stable Diffusion 那篇更像数据工程，除非我们把几何约束做成核心机制，否则很容易只是换生成器。最后三篇我继续读完。
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
/bin/zsh -lc "pdftotext -l 3 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

205

CMAG: Cross-Modal Attention and
Graph-Enhanced Memory for Unsupervised
Visible-Infrared Person Re-Identification
Yi-Feng Zhang , Can-Long Zhang , Member, IEEE, Jun-Wei Tian , Hai-Fei Ma ,
Zhi-Xin Li , Member, IEEE, and Zhi-Wen Wang

Abstract—Unsupervised
visible-infrared
person
reidentification (USL-VI-ReID) has garnered widespread attention
due to its surveillance application value in complex environments.
However, it faces four key challenges: modality discrepancy,
batch training limitations, pseudo-label noise, and camera view
bias. This paper proposes the CMAG (Cross-Modal Attention
and Graph-enhanced Memory) framework, which innovatively
combines circular topology structure with cross-modal attention
mechanisms to address these challenges. CMAG introduces
four core innovations: (1) applying circular topology structure
to provide pseudo-label verification through detecting circular
paths in feature space, effectively addressing the pseudo-label
noise problem; (2) designing a cross-modal attention mechanism
for Vision Transformers with residual fusion to balance
modality-specific and shared information, solving the modality
discrepancy issue; (3) constructing a graph-structured memory
enhancement module with adaptive graph construction and
multi-layer feature propagation to overcome batch training
limitations; and (4) integrating camera-specific clustering with
circular structure constraints to reduce camera background bias.
Extensive experiments on SYSU-MM01 and RegDB datasets
demonstrate the effectiveness of CMAG, achieving approximately
3.5% improvement in Rank-1 accuracy and 2.8% in mAP
on average compared to state-of-the-art methods, validating
our approach’s advantages in addressing key challenges in
unsupervised cross-modal person re-identification.Code is
available at https://github.com/hurryup186/CMAG

Received 23 April 2025; revised 4 July 2025; accepted 31 July 2025.
Date of publication 5 August 2025; date of current version 22 January
2026. This work was supported in part by the Natural Science Foundation
of China under Grant 62266009, in part by Guangxi Science and Technology
Program under Grant AB25069418, in part by Guangxi First-Class Undergraduate Course Construction Project under Grant 202103, in part by Guangxi
Engineering Research Center of Educational Intelligent Technology, and in
part by the Innovation Project of Guangxi Graduate Education under Grant
YCBZ2024104. This article was recommended by Associate Editor Z. Yang.
(Corresponding author: Can-Long Zhang.)
Yi-Feng Zhang, Jun-Wei Tian, and Hai-Fei Ma are with the
Key Laboratory of Education Blockchain and Intelligent Technology,
Ministry of Education, Guangxi Normal University, Guilin 541004,
China (e-mail: zhangyifeng186@stu.gxnu.edu.cn; junwtian@stu.gxnu.edu.cn;
mhf2022@mailbox.gxnu.edu.cn).
Can-Long Zhang and Zhi-Xin Li are with the Key Laboratory of Education
Blockchain and Intelligent Technology, Ministry of Education, Guangxi
Key Laboratory of Multi-Source Information Mining and Security, Guangxi
Normal University, Guilin 541004, China (e-mail: clzhang@gxnu.edu.cn;
lizx@gxnu.edu.cn).
Zhi-Wen Wang is with the School of Electronic Engineering, Guangxi
University of Science and Technology, Liuzhou 545006, China (e-mail:
wzw69@gxust.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3595846

Index Terms—Unsupervised cross-modal re-identification, circular topology structure, graph-structured memory, cross-modal
attention, vision transformer.

I. I NTRODUCTION

I

N RECENT years, visible-infrared cross-modal person reidentification (VI-ReID) has received widespread attention
due to its security applications [1], [2], [3], [4], [5] in nighttime
surveillance and complex lighting conditions. Compared to
traditional single-modal person re-identification [6], [7], [8],
[9], VI-ReID faces greater challenges: it must not only address
common issues such as pose variations, occlusions, and camera view differences, but also bridge the representation gap
caused by modality disparities [10], [11]. Although supervised
methods have made significant progress in this field [12], [13],
they rely on large amounts of annotated data, which is costly
and limited in scalability.
Unsupervised visible-infrared person re-identification
(USL-VI-ReID) faces four interconnected challenges
that fundamentally limit existing approaches: modality
discrepancy, batch training limitations, pseudo-label noise,
and camera view bias, as illustrated in Figure 1. The core
limitation of current methods stems from a fundamental
conceptual gap: they focus primarily on direct pairwise
relationships while neglecting rich structural information
in higher-order feature space relationships, particularly
problematic in cross-modal scenarios where direct similarity
measurements are inherently unreliable due to modality
disparities.
Existing methods fail to effectively address these challenges
due to specific technical limitations. For modality discrepancy, CNN-based approaches like DDAG [14] and Hi-CMD
[15] rely on simple projections without fine-grained attention
mechanisms to balance modality-specific preservation with
cross-modal alignment [16], [17]. For batch training limitations, current methods use either Cluster Memory (losing
instance details) or Instance Memory (ignoring structured
relationships) without effective global transfer mechanisms,
causing inconsistent representations [18], [19], [20]. For
pseudo-label noise, clustering methods like PGM [21] and
ADCA [15] lack robust noise detection and ignore higherorder structures, relying on unreliable direct similarities [22],
[23], [24]. For camera view bias, methods like OTLA [25]

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.

206

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

Fig. 1. Core challenges in cross-modal person ReID(recommended to
observe from a color perspective). Core challenges in cross-modal person reidentification. (a) Modality difference and identity relationship representation:
infrared (left) vs. visible (right) modalities with different identities (colorcoded). Ideally Dinter should be greater than Dintra , which is difficult to
maintain across modalities. (b) Batch training limitations in cross-modal ReID.
Left: Global distribution of complete RGB (stars) and IR (squares) samples,
with colors representing different identities. Right: Examples of first and Nth
batch sampling, showing how limited local sampling at different iterations
leads to inconsistent feature representation across batches and modalities.(c)
Conventional approaches assign pseudo-labels by selecting the most similar
cluster center for each sample; however, this strategy suffers from inherent
limitations. Specifically, both correct and incorrect samples demonstrate
similarity consistency, with pseudo-label noise causing ambiguity in the
prototypes derived from clustering.(d) Camera view bias: Same person appears
significantly different across camera viewpoints. Images A, B, C, D show the
same identity from front, right side, back, and left side views respectively
(top: visible, bottom: infrared). Traditional direct similarity measurements
struggle to associate these extreme pose variations, as A(front) and C(back)
exhibit very low similarity. Our circular topology structure addresses this
by forming bridging connections A−→B−→C−→D−→A, where intermediate
poses serve as transitional links to establish reliable identity consistency even
under extreme viewpoint changes.

and GUR [26] ignore camera information or use it as auxiliary
features, lacking camera-aware strategies [27]. Recent TCSVT
publications have made progress in addressing these challenges through camera-proxy mechanisms [28], asymmetric
learning strategies [29], and self-supervised approaches [30],
yet systematic solutions integrating multiple challenges remain
limited.
To address these interconnected challenges systematically,
we propose the CMAG (Cross-Modal Attention and Graphenhanced Memory) framework with four targeted innovations:
Dynamic Cross-modal Attention (addressing modality discrepancy), Graph-Structured Memory Enhancement (overcoming
batch limitations), Circular Topology Structure (reducing
pseudo-label noise), and Camera-Aware consistency constraints (mitigating camera bias). Unlike existing approaches
that treat these challenges independently, CMAG provides a
unified solution through novel integration of algebraic topology theory, Vision Transformer-specific attention mechanisms,
and camera-aware learning strategies. The core innovation

lies in examining feature space from a topological perspective, introducing circular topology structure (CATS) theory
into cross-modal feature learning for the first time. CATS
demonstrates significant advantages in resolving pose variation challenges. While direct similarity between front and
back views is typically low, circular paths introduce intermediate poses as “bridging points” (A−→B−→C−→D−→A),
establishing reliable identity consistency even under extreme
pose variations. We also designed a cross-modal attention
mechanism specifically for unsupervised visible-infrared person re-identification, called Dynamic Cross-modal Attention
Mechanism for ViT (DCAM-ViT). Our DCAM-ViT adopts
an innovative residual design that dynamically adjusts fusion
intensity, theoretically resolving the contradiction between
feature alignment and discriminability.
The Graph-Structured Memory Enhancement Module
(GSMEM) breaks through batch training limitations by maintaining a global feature memory bank and constructing
cross-batch sample relationship graphs. Unlike existing methods relying solely on Cluster or Instance Memory, GSMEM
organically integrates memory networks with graph neural networks. We also propose Camera-Aware consistency constraint
(CARC) mechanism integrating camera-specific clustering
with circular structure to reduce camera background bias,
achieving balance between cross-camera openness and strict
intra-camera verification. In summary, our main contributions
directly address the four identified challenges:
• Circular topology structure (CATS) introducing algebraic topology theory for cross-modal pseudo-label validation, effectively addressing pseudo-label noise through
transitivity verification.
• Vision Transformer-specific cross-modal attention
(DCAM-ViT) with class token level interaction and residual fusion, solving modality discrepancy by balancing
specific and shared information.
• Graph-structured memory enhancement (GSMEM)
integrating memory networks with graph neural networks,
overcoming batch training limitations through global consistency maintenance.
• Camera-aware consistency constraint (CARC) combining camera-specific clustering with circular structure,
reducing camera background bias through optimized clustering and adaptive filtering.
II. R ELATED W ORK
A. Unsupervised Cross-Modal Person Re-Identification
Unsupervised person re-identification (USL-ReID) aims
to learn discriminative features without labels through
pseudo-label generation and feature enhancement approaches.
For pseudo-label generation, clustering algorithms construct
pseudo-supervised signals [16], [31], while feature enhancement methods employ contrastive learning with memory
mechanisms. However, these single-modal approaches cannot
directly address cross-modal challenges.
Unsupervised cross-modal person re-identification (USLVI-ReID) extends this to visible-infrared scenarios. Yang et
al. [32] explored mutual information maximization, while

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID

Wang et al. [33] designed alternating clustering for crossmodal consistency. Recent advances include Yang et al. [15]
dual contrastive learning (ADCA), Wang et al. [25] optimal
transport (OTLA), and Wu and Ye [21] graph matching
(PGM). The most recent works employ token-level attention mechanisms [27] and dual-stream contrastive learning
approaches [34].
B. Cross-Modal Person Re-Identification
Supervised cross-modal person re-identification typically
employs feature alignment [12] or generative approaches.
Ye et al. [35] proposed dual-stream architectures, while Wu
et al. [13] designed mutual learning frameworks. These
methods require extensive labeled data, limiting practical
deployment.
Key limitations include: most methods rely on CNN architectures without leveraging Transformer advantages, and lack
effective global feature transfer mechanisms, resulting in significant cross-modal pseudo-label noise. While comprehensive
approaches like GUR [26] address hierarchical discrepancy,
our research targets specific bottlenecks through ViT-specific
attention, graph-enhanced memory, and topological structure
validation.
C. Graph Neural Networks in Feature Learning
Graph Neural Networks have been applied to person reidentification for structured modeling [36]. Memory-enhanced
approaches like Li et al. [37] introduced global feature memory banks, while recent works have explored cross-modal
graph applications. However, circular graph structures—paths
returning to starting points—remain largely unexplored despite
their rich topological information [38]. Our research introduces circular topology theory to USL-VI-ReID for enhanced
pseudo-label validation.
D. Vision Transformer-Based Cross-Modal Learning
Vision Transformer (ViT) demonstrates advantages in capturing long-distance dependencies [39]. For VI-ReID, Sarker
and Zhao [40] proposed cross-modal attention Transformers,
while Yang et al.’s SDCL [41] employed shallow-deep collaboration. However, existing methods typically adopt simple
fusion mechanisms, struggling with complex inter-modality
relationships. Our research focuses on ViT-specific crossmodal attention design through class token interaction and
residual fusion, operating at the attention mechanism level
rather than feature hierarchy collaboration.

207

A. Problem Definition and Framework Overview
1) Two-Stage Learning Strategy: Our CMAG framework
adopts a progressive two-stage learning approach to address
the complexity of unsupervised cross-modal person reidentification. Stage 1 (50 epochs) establishes robust feature
extraction through a shared ViT backbone with modalityspecific clustering, building discriminative representations for
both RGB and IR modalities. Stage 2 (100 epochs) activates our novel cross-modal components (CATS, DCAM-ViT,
GSMEM) for advanced modality alignment and unified feature
space learning.
In the USL-VI-ReID task, we have two unlabeled datasets:
Nrgb
visible light dataset XRGB = {xirgb }i=1
and infrared dataset
rgb
ir Nir
H×W×3
XIR = {xi }i=1 , where xi ∈ R
and xiir ∈ RH×W×1
represent the i-th visible and infrared images respectively, Nrgb
and Nir denote the number of visible and infrared training
samples, and H, W are the image height and width. The
objective is to learn a unified feature space under unsupervised
conditions.
The CMAG framework employs a multi-task learning
paradigm that integrates loss functions from all modules.
Each component contributes its specialized loss terms: circular
consistency loss from CATS, cross-modal alignment loss from
DCAM-ViT, graph-structured memory enhancement loss from
GSMEM, and camera-aware consistency loss from CARC.
These loss functions are jointly optimized with adaptive weight
adjustment to balance modality alignment, feature enhancement, and pseudo-label optimization throughout the training
process. This multi-stage collaborative optimization allows
the framework to efficiently handle modality differences, feature inconsistency, and camera view bias under unsupervised
conditions.
2) Framework Visualization Details: The visual elements
in Figure 2 are designed to illustrate the key mechanisms of
CMAG. In the clustering phase (a), blue and orange bars represent RGB and IR features respectively, with different colored
stars and circles in the clustering results denoting different
identity clusters. The cycle-aware structure (b) visualizes how
circular paths (thick green edges) connect features to validate
identity consistency, with the formula C p = (A p ) A computing
these paths. The memory module (c) shows both the temporal
queue update mechanism (left, with ‘ptr’ indicating current
position) and adaptive graph construction (right, where node
size reflects local density). Throughout the framework, blue
consistently represents RGB modality while orange represents
IR modality, facilitating cross-modal understanding.
B. Basic Feature Learning Network

III. P ROPOSED M ETHOD
Unsupervised visible-infrared person re-identification (USLVI-ReID) faces four key challenges: feature inconsistency
caused by modality differences, limited global information access due to batch training constraints, pseudo-label
noise issues, and camera view bias. This section introduces our proposed CMAG (Cross-Modal Attention and
Graph-enhanced Memory) framework, which addresses these
challenges through four innovative components.

To capture rich semantic information, we adopt a Vision
Transformer (ViT)-based multi-class token feature extraction
network. Inspired by recent multi-token approaches [27],
we employ K = 4 class tokens, significantly enhancing
the model’s ability to express multi-granularity features.
The feature extraction process is represented as shown in
Equation (1):
Zrgb = frgb (xrgb ) ∈ RK×d ,

Zir = fir (xir ) ∈ RK×d

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.

(1)


exec
/bin/zsh -lc "pdftotext -l 3 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 196 (2026) 108374

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

ColorSketchNet: Unifying color, sketch and texture for modality-agnostic
multi-modal person re-identiﬁcation
Manman Liu

a,b , Xu Cheng

a,b,c,∗, Yan Jiang

c , Baowei Wang

c

a School of Software, Nanjing University of Information Science and Technology, Nanjing, 210044, China
b

State Key Laboratory of Integrated Services Networks, Xidian University, Xi’an, 710071, China

c School of Computer Science, Nanjing University of Information Science and Technology, Nanjing, 210044, China

a r t i c l e

i n f o

Keywords:
Multi-modal person re-identiﬁcation
Uniﬁed framework
Attribute compensation
Auxiliary modality

a b s t r a c t
Person re-identiﬁcation (ReID) using text or sketch as descriptive query has made great eﬀorts in practical application. Existing multi-modal ReID methods usually map features from diﬀerent modalities into a common space
to learn cross-modal alignment. However, they ignore inherent modality attribute diﬀerences, resulting in limited shared feature representation. Moreover, unfair modality attributes exacerbate the imbalance in multi-task
learning, e.g., Text-RGB, Sketch-RGB, Text and Sketch-RGB. Towards these issues, we propose a ColorSketchNet
framework that can adaptively compensate for diﬀerent modality features and establish a fair attribute space for
cross-modal alignment. Speciﬁcally, we ﬁrst introduce a color sketch generator to generate the auxiliary modality
that includes rich color and texture sketch, which is utilized to narrow the modality gap with the target image.
Second, an attribute compensation and reﬁned module is presented to compensate for the missing information in
sketch and text with the help of auxiliary color sketch modality and suppress auxiliary modality noise, thereby
enhancing the reliability of shared features across modalities. Further, a uniﬁed collaborative alignment learning
scheme is developed to guide the model in learning identity representations of each modality. Extensive experiments on three multi-modal ReID datasets show that our ColorSketchNet greatly improves the retrieval accuracy
and generalization ability in both paired modality and tri-modality scenarios.

1. Introduction
Person re-identiﬁcation (ReID) has been extensively studied in recent
years due to its critical role in intelligent security surveillance, which
aims to match the same individuals under multiple non-overlapping
cameras (Ye et al., 2021). Existing ReID methods can be divided
into single-modal and multi-modal approaches. The former focuses on
matching RGB images (Liu et al., 2024b; Tan et al., 2025), while the
latter retrieves RGB pedestrian images using heterogeneous modalities,
i.e., text (Jiang and Ye, 2023a; Li et al., 2024a), sketch (Lin et al., 2023;
Pang et al., 2018a) and infrared (Wu et al., 2025a; Zhu et al., 2024). Especially in real-world criminal investigations, multi-modal ReID is indispensable, since high-quality photos are often missing and suspect identiﬁcation must rely on eyewitness descriptions or artist sketch (Ye et al.,
2015).
Most existing ReID works (Lin et al., 2024; Ren et al., 2025; Ye et al.,
2024) mainly focus on paired modality task (e.g., visible-text, visiblesketch), aiming to learn the shared feature to reduce the modality gap
between two inputs. Despite encouraging success, these methods still
face challenges in real-world criminal scenarios, where eyewitnesses

often provide both portraits (sketch) and verbal descriptions (text) simultaneously. In such complex scenarios, paired modality methods are
inherently incapable of fully exploiting the complementary cues across
modalities, leaving a critical gap between existing research and practical
forensic needs.
Recently, some works have also attempted to addressing this challenge. Chen et al. (2023a) proposed a uniﬁed framework (UNIReID) that
is capable of handling Text-RGB, Sketch-RGB, and Text and Sketch-RGB
tasks simultaneously. Although UNIReID covers all potential scenarios
of crime identiﬁcation in real-world contexts, it fails to fairly reduce
the cross-modality gap due to the varying modality diﬀerences between
diﬀerent subtasks. As shown in Fig. 1(a), in the Text-RGB task, text descriptions are usually subjective impressions and it is diﬃcult to accurately report detailed appearance information. Consequently, the model
can only focus on a limited set of shared attributes, such as color and
clothing type, which may result in the retrieval of multiple individuals with similar coarse-level features, such as pedestrians wearing red
tops and black pants. Similarly, in the Sketch-RGB task, sketch is abstract and has limited ability to represent color and style information.
Existing methods learn only the contour features between the sketch

∗ Corresponding author. Both Manman Liu and Xu Cheng contributed equally to this research.

E-mail addresses: lmanman2022@163.com (M. Liu), xcheng@nuist.edu.cn (X. Cheng), 202212200047@nuist.edu.cn (Y. Jiang), wbw.ﬁrst@163.com (B. Wang).
https://doi.org/10.1016/j.neunet.2025.108374
Received 8 May 2025; Received in revised form 2 November 2025; Accepted 24 November 2025
Available online 30 November 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 196 (2026) 108374

M. Liu et al.

Fig. 1. Illustration of existing methods and the proposed method. (a) Existing methods extract the features of diﬀerent modalities and directly project them into the
common space for alignment, which is prone to semantic ambiguity. (b) Our method can compensate for the features between modalities before mapping them to
common space by using color sketch, which improves the performance of cross-modality learning.

Table 1
The coverage of the three dimensions of color, contour, and pose in the Tri-ICFG-PEDES dataset was
quantitatively counted.
Modality

Color

Counter

Pose

Text
Sketch
RGB

86.0 %
0%
97.7 %

33.4 %
75.2 %
98.0 %

47.8 %
92.7 %
67.7 %

lacks ﬁne-grained structural cues, and RGB is the only modality that encompasses all attributes. This quantitative evidence conﬁrms that simply aligning features across modalities (without explicitly compensating
for these missing attributes) inevitably leads to biased or ambiguous retrieval results.
Motivated by these observations, we propose ColorSketchNet, a
novel modality-agnostic multi-modal ReID framework, meaning that our
approach can consistently maintain strong performance regardless of
whether the query is text, sketch, or a combination of both. In addition, by introducing an auxiliary colored sketch modality, we establish
a fair attribute space in which the inherent disparities among diﬀerent
modalities are explicitly compensated, thereby enabling uniﬁed representation learning across all three modalities. As illustrated in Fig. 1(b),
the generator ﬁrst produces a colored sketch corresponding to a pedestrian photo, which contains both contour, color and texture information.
This auxiliary modality allows the model to compensate for ﬁne-grained
visual details missing in text and for color or style information absent
in sketch, while aligning them with RGB images within the uniﬁed attribute space. Speciﬁcally, the generated colored sketch provides edge
cues required for text-to-RGB matching, color cues required for sketchto-RGB matching, and complementary signals for the joint text+sketchto-RGB task. In this way, the model is able to extract richer shared
features and achieve more robust cross-modal learning. By explicitly
addressing the limitations of prior paired modality and uniﬁed frameworks, ColorSketchNet represents a signiﬁcant step toward practical and
reliable multi-modal ReID.
Speciﬁcally, we ﬁrst design the color sketch generator (CSG) module to generate knowledge-rich color sketch that are closely aligned
with the target distribution, which consists of dynamic lighting modiﬁer
and color sketch auxiliary generator. The dynamic lighting modiﬁer is

and images, leading to potential semantic ambiguity and mismatches
between multiple images. Meanwhile, the learning process is restricted
due to the limited shared attributes among the three modalities in the
Text and Sketch-RGB task. These inherent diﬀerences cause large shifts
in feature distributions, leading to unreliable training. The fundamental challenge, therefore, is not just to design a uniﬁed framework, but
to explicitly compensate for the missing attributes of each modality-0a
problem largely overlooked by existing methods.
To further validate this observation, we conducted a quantitative
analysis on the ﬁrst 500 samples of the Tri-ICFG-PEDES dataset, focusing on the three modalities (text, sketch, and RGB). We evaluated the
presence of discriminative attributes in three key dimensions crucial for
person re-identiﬁcation: color, contour, and pose. As shown in Table 1,
RGB images provide near-complete coverage for both color and contour.
Sketch, while entirely lacking color cues, retain strong signals for both
contour and pose. In contrast, text-to-RGB samples oﬀer rich color coverage but weaker contour information. Furthermore, the analysis conﬁrms
that pose consistency is also highly modality-dependent.
Thus, these statistics, along with their illustration in Fig. 1, clearly
demonstrate the inherent modality diﬀerences: sketch lack color, text
2

Neural Networks 196 (2026) 108374

M. Liu et al.

introduced to prevent visual information loss caused by insuﬃcient
lighting or overexposure, allowing our generator to handle photos under complex lighting conditions. Then the Diﬀerence of Gaussian (DoG)
ﬁlter is exploited to extract contour information from the photo, and a
auxiliary color sketch modality is obtained by the YUV channel replication strategy. Second, an attribute compensation and reﬁned module (ACRM) is proposed to compensate for missing color information in
the sketch and missing texture details in text, which utilizes the auxiliary color sketch as a bridge to introduce additional visual details, expand the shared space that the model can learn, and alleviate the visual
diﬀerences caused by the inherent modality diﬀerences. Additionally,
ACRM remove the modality-irrelevant features and improve the reliability of shared features to reduce the potential noise in the auxiliary
color sketch modality generation. Further, we design a uniﬁed collaborative alignment learning strategy(UCA) to guide the model in learning
identity representations of each modality 𝑙𝑖𝑑𝑚 and establish structural
constraints 𝑙𝑐𝑚 between the auxiliary color sketch and the three modalities. The intra-class compactness loss 𝑙𝑖𝑐𝑐 introduced in UCA aims to
minimize the feature variance between samples of the same identity and
solve the problem of unbalanced feature distribution within the modality due to changes in perspective, illumination and partial occlusion.
Generally, the contributions of this paper can be summarized as follows:

using classiﬁcation loss and triple loss, but they exhibit limited discriminability for ﬁne-grained details. The latter methods (Ke et al., 2024;
Wu et al., 2024a) enhance cross-modality recognition ability by using
ﬁne-grained local feature alignment. Yan et al. (2023) used attention
mechanisms to associate textual keywords with corresponding image
regions, ACF-R+ (Gong et al., 2025) introduced an asymmetric perception mechanism to better model semantic mappings, and HKGR (Zeng
et al., 2025) eﬀectively mitigates cross-modal semantic noise through
scene graph parsing and multi-granularity semantic alignment. Despite
their strong performance in bi-modal retrieval, these methods heavily
rely on large-scale paired supervision and show limited robustness under modality absence, weak supervision, or heterogeneous multi-modal
inputs, thereby restricting their scalability to more complex tri-modal
scenarios.
Sketch-RGB person ReID. Compared with text-image tasks, sketchimage retrieval focuses more on visual attribute discrepancies, such as
the absence of color and texture details in sketch. Existing methods are
divided into direct cross-modal feature alignment methods (Gui et al.,
2020; Zhu et al., 2022) and auxiliary modality generation-based methods (Chen et al., 2022). The former emphasize domain-invariant feature learning, for instance, Pang et al. (2018a) introduced an adversarial learning model to reduce the domain gap between sketch and RGB,
while the MCGI framework (Zhang et al., 2025b) proposed modality collaboration and granularity interaction mechanisms to fuse complementary features. StyleGAN-based approaches (Koley et al., 2024a) further
leveraged sketch abstraction levels combined with diﬀerentiable ranking losses to improve cross-style alignment stability. The latter introduce
auxiliary modalities as a “bridge" to compensate for missing attributes,
for instance, Chen et al. (2022) introduce a "bridge" modality between
sketch and RGB to mitigate the modality gap. Later, Liu et al. (2024a)
proposed an auxiliary modality that is robust to background clutter and
style variation, which developed a multi-modal collaborative learning
scheme to align feature relationships and distributions. Despite their effectiveness in aligning visual modalities, most existing methods remain
limited to low-level visual feature compensation and often fail to capture semantic-level discrepancies. Moreover, existing generative methods typically produce black-and-white sketch as auxiliary modalities,
neglecting color distributions and texture details, which restricts their
alignment capacity with RGB images.
Tri-modality person ReID. This task aims to fully leverage the advantages of diﬀerent modalities. Zhai et al. (2022) ﬁrst proposed a symmetric disentangling scheme to promote adversarial alignment of descriptive features from sketch and text with RGB image features, which
improved the model’s semantic understanding. Chen et al. (2023a) presented a uniﬁed person ReID framework for cross-modal and multimodal tasks, introducing sketch and text as descriptive queries to
explore modality-agnostic re-identiﬁcation scenarios. However, these
methods primarily focus on mapping global information extracted from
diﬀerent modalities into a common space for modality alignment. Meanwhile, it also faces the problems of signiﬁcant attribute diﬀerences between modalities and incomplete or inaccurate information during the
alignment. In contrast, we propose a ColorSketchNet framework to alleviate the unfair modality attributes across the modalities.

•

We are the ﬁrst to introduce an auxiliary color sketch modality into
multi-modal ReID. This novel modality explicitly compensates for
the inherent attribute diﬀerences between text, sketch, and RGB, enabling uniﬁed representation learning across heterogeneous modalities that existing paired modality or uniﬁed frameworks cannot
achieve.
• We design a Color Sketch Generator (CSG) that not only produces
vivid and texture-rich color sketch, but also incorporates a dynamic
lighting modiﬁer to handle illumination variations-0addressing a key
limitation of prior ReID methods that often fail under complex lighting conditions.
• We propose an Attribute Compensation and Reﬁnement Module
(ACRM) that leverages the auxiliary color sketch as a bridge to build
a fair and balanced attribute space across modalities. Unlike conventional feature alignment, ACRM performs explicit attribute compensation and noise removal, thereby enhancing the reliability of shared
features.
• Extensive experiments on three challenging trimodal benchmark datasets (Tri-ICFG-PEDES, Tri-RSTPReid, and Tri-PKU-Sketch)
demonstrate that our ColorSketchNet delivers robust and superior
performance in both paired modality and tri-modality retrieval tasks.
Furthermore, additional evaluation on the SketchyCOCO dataset
conﬁrms its strong generalization ability and practical potential for
real-world forensic applications.
2. Related work
2.1. Cross-modal person re-identiﬁcation
Cross-modal ReID tasks can be divided into three categories: TextRGB person ReID (Xia et al., 2025), Sketch-RGB person ReID (Zhang et
al., 2022a) and Tri-modality person ReID (Li et al., 2024b).
Text-RGB person ReID. In Text-RGB person re-identiﬁcation, the
core challenge lies in the signiﬁcant cross-modal discrepancy between
textual descriptions and visual images, as well as the diﬃculty of modeling ﬁne-grained attributes such as clothing colors and patterns. In recent years, contrastive learning has gradually become the mainstream
framework for cross-modal alignment. Within this framework, existing
studies can be broadly divided into global feature modeling and local
feature modeling. The former methods (Cheng et al., 2024; Li et al.,
2023; Zheng et al., 2020a) extract global features from a dual-stream
encoder into a common space and then optimize vector distances by

2.2. Attribute information compensation
Attribute compensation aims to introduce additional information or
compensate for missing information in speciﬁc modalities (such as images, text, sketch, etc.) to alleviate the modality attributes imbalance.
In recent years, some studies (Chen et al., 2022; Wang & Cheng, 2024;
Yu et al., 2024) have been proposed to generate auxiliary modalities to
compensate for unique information in diﬀerent modalities.
In addition, Zhang et al. (2024) proposed a MSALNet to learn information compensation and fusion between visible light and infrared features by generating auxiliary branches, which utilized multi-stage auxiliary learning strategy to suppress interference information and improve
3


exec
/bin/zsh -lc "pdftotext -l 3 'Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Robotics and Automation (ICRA)
May 19-23, 2025. Atlanta, USA

Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle
Re-Identification with High-Fidelity Synthetic Data

2025 IEEE International Conference on Robotics and Automation (ICRA) | 979-8-3315-4139-2/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICRA55743.2025.11127895

Leyang Jin1 , Wei Ji2 , Tat-Seng Chua1 , Zhedong Zheng3†
Abstract— Due to the critical issues of privacy and partial
occlusion, license plate information is not always available
in vehicle recognition systems. Consequently, researchers have
increasingly turned towards vehicle re-identification (reID)
techniques to bridge the gap between cross-view camera
systems. Despite the growing interest, one major challenge
persists: the scarcity of authentic, large-scale training datasets.
To address this challenge, this paper introduces a coarse-to-fine
generation pipeline designed to synthesize high-fidelity vehicle
data, thereby facilitating subsequent vehicle representation
learning. Specifically, the proposed approach consists of three
stages: Prompt Processing, Diffusion Fine-tuning, and Semantic Filtering. First, we collect detailed prompts from vehicle
websites and companies with fine-grained vehicle prototype
attributes. Next, we leverage the prior knowledge of these
automotive prototypes to fine-tune diffusion models. Finally,
to ensure the quality of the synthesized data, we employ pretrained vision-language models to filter out substandard images.
Building upon the high-quality data generated by this pipeline,
we validate the effectiveness using vanilla models. Extensive experimental evaluations demonstrate that our approach achieves
competitive accuracy on public benchmarks such as VeRi-776,
VehicleID and CityFlowV2, and is compatible with various
model architectures.

I. I NTRODUCTION
Vehicle re-identification (reID) aims to match images of
the same vehicle across multiple cameras, which is crucial
for the deployment of autonomous vehicles [1] and intelligent traffic systems [2]. Given the minor intra-class differences between car models, vehicle reID is typically treated as
a fine-grained representation learning task [3], [4]. However,
privacy concerns [5] and annotation difficulties in multisensor systems [6], [7] result in a scarcity of realistic training
data. To address this issue, recent research [8], [9], [10]
has focused on generating synthetic data for vehicle reID.
Despite these efforts, generating large-scale, high-fidelity
training data that captures subtle inter-class discrepancies and
intra-class consistencies remains challenging.
Existing efforts on vehicle reID data generation can be
divided into two directions: 1) Graphics-engine-based methods, such as PAMTRI [8] and VehicleX [9]. They employ 3D
1 Leyang Jin and Tat-seng Chua are with School of Computing, National
University of Singapore, Singapore 117417 e0792447@u.nus.edu,

dcscts@nus.edu.sg
2 Wei Ji is with School of Intelligence Science and Technology, Nanjing
University, China 215163 weiji@nju.edu.cn
3 Zhedong Zheng is with the FST and ICI, University of Macau, China
999078 zhedongzheng@um.edu.mo † corresponding author
This work is supported by the University of Macau Start-up Research Grant SRG2024-00002-FST and Multi-Year Research Grant MYRGGRG2024-00077-FST-UMDF

979-8-3315-4139-2/25/$31.00 ©2025 IEEE

Fig. 1: We compare our Vehicle-Diff dataset to existing
synthetic datasets. The second and third rows of datasets
are based on 3D engines (PAMTRI [8] and VehicleX [9]),
while PTGAN [11] and VehicleGAN [10] adopt the datadriven structure, i.e., Generative Adversarial Networks [12].
We could observe that the proposed method is with a closer
visual appearance compared to the real dataset, i.e., VeRi776. Besides, the generated images by the proposed method
are associated with text captions, allowing for cross-modality
knowledge to guide generation.
CAD models to generate vehicle images. While these methods have made significant strides, they still face challenges.
There is a notable domain gap between rendered 3D CAD
vehicle images and actual real-world images. Additionally,
the process of generating the VehicleX dataset relies heavily
on a large amount of labeled vehicle re-identification data,
which is costly and raises privacy concerns. Similarly, synthetic data from PAMTRI needs to be combined with fully
labeled re-identification datasets. 2) Data-driven methods,
such as generative adversarial networks (GANs) [12]. For
instance, PTGAN [11] and VehicleGAN [10] explore GANs
to synthesize novel vehicle views. Although these methods
generate vehicle images with relatively good visual quality,
they under-explore the cross-modality guidance and thus the
fine-grained attributes of the same vehicle are often inconsistent, compromising the training process of the vehicle reID.
To address the aforementioned challenges, we propose
Vehicle-Diff, a new pipeline designed to synthesize large-

7319

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.

scale training data for vehicle re-identification, facilitating
the representation learning. In particular, the pipeline consists
of three primary stages: prompt processing, diffusion model
tuning, and semantic filtering. We first collect and process the
prompt for vehicles with a focus on the vehicle attribute. To
harness the pre-trained inherent knowledge of car prototypes,
we employ carefully crafted prompts. Then, we fine-tune
the diffusion model using only 1% of unlabeled target data
during the generation stage. It enables the diffusion model
to adapt to the target vehicle domain at both the content
and stylistic levels. In the subsequent filtering stage, we
apply sophisticated post-processing techniques to enhance
the semantic alignment of the generated data. Our pipeline
is scalable and adaptable to multiple downstream scenarios,
reducing labeling costs and privacy concerns. As shown in
Fig. 1, the generated vehicle images are much closer to the
real-world data. Finally, we construct a new labeled vehicle re-identification dataset, called Vehicle-Diff, comprising
149,472 images of 4,940 distinct vehicles. The efficacy of
Vehicle-Diff is substantiated through comparative evaluations
with synthetic datasets produced by existing approaches. In
summary, our paper makes the following contributions:
• A new coarse-to-fine cross-modality generation pipeline
by prompting the diffusion model to craft a synthetic
vehicle re-identification dataset tailored to a downstream
scene, with only about 1% unlabeled images in the
original dataset. To the best of our knowledge, our work
is among the early attempts for large-scale training data
generation with attributes for vehicle re-identification.
• Extensive experiments have validated that our pipeline
can minimize the gap between synthetic and real
data, facilitating the subsequential reID model learning.
The proposed method has achieved competitive performance, e.g., 83.79% mAP on the VeRi-776 dataset.
II. R ELATED WORK
Vehicle Re-Identification. Vehicle re-identification (reID)
involves retrieving vehicles of interest from a database of
images collected by traffic cameras. Previous studies [13],
[14], [15], [16] have achieved significant success using
supervised learning. However, this approach faces challenges
such as high annotation costs and privacy concerns when
collecting and labeling data. To mitigate these issues, some
works [17], [18] have explored unsupervised learning to
reduce annotation requirements. Despite these efforts, substantial real data is still needed for general vehicle reID
tasks [16], and attribute annotations remain preferable [19],
[20]. In contrast, we propose a multi-modality data synthesis
approach that significantly reduces the need for both real data
and annotations, addressing these limitations effectively.
Synthetic Datasets for Vehicle Re-Identification Task.
Synthetic data are increasingly used to address privacy concerns and high annotation costs in creating re-identification
datasets [21], [22]. Previous works [8], [9], [23], [24],
[25] have employed 3D engines to generate characters and
vehicles, but these assets suffer from the intrinsic domain gap
between virtual and real scenes and are time-consuming to

create. VehicleGAN [10] and PTGAN [11] deploy GANs for
data augmentation, with VehicleGAN focusing on AutoReconstruction and pose consistency, and PTGAN generating
novel vehicle views based on given poses. However, these
methods still require large labeled datasets for effective
training and are constrained by the quality and patterns of the
original data. In contrast, our multi-modality data synthesis
approach reduces the need for both real data and annotations,
addressing these limitations effectively.
Text-to-image Diffusion Models. Diffusion models [26],
[27] have recently emerged as promising generative models,
particularly for text-to-image generation, where they can
produce images based on textual descriptions. Recent advancements such as Stable Diffusion [28], Stable Diffusion
XL [29], and Midjourney [30] have demonstrated remarkable
results in this domain. Leveraging the power of these models,
methods like [31], [32], [33] have utilized diffusion models,
e.g., GLIDE [34], to generate synthetic data for image classification. Despite their impressive visual outcomes and applications, the potential of text-to-image diffusion models for
vehicle re-identification remains underexplored. In this paper,
we evaluate multiple state-of-the-art text-to-image models
and identify the optimal model for enhancing downstream
vehicle re-identification performance.
III. METHOD
An overview of Vehicle-Diff is provided in Fig. 2. VehicleDiff generates high-fidelity data in a coarse-to-fine manner
to enhance reID network training, comprising three stages:
(1) prompt processing, (2) diffusion fine-tuning, and (3)
semantic filtering. First, the prompt processing stage (§IIIA) constructs a prompt library and specifies vehicle attributes
such as models and colors for image generation. Next, during
the diffusion fine-tuning stage (§III-B), Vehicle-Diff finetunes the diffusion model using unlabeled vehicle images,
improving its adaptation to vehicle image generation. Finally,
in the semantic filtering stage (§III-C), Vehicle-Diff generates
vehicle images with different IDs using the prompt library
and fine-tuned model, followed by filtering these images
through off-the-shelf detection and cross-modality alignment.
A. Prompt Processing
The prompt processing stage aims to construct discriminative vehicle attribute prompts to guide image generation, thus
enhancing inter-class consistency and intra-class diversity.
We first filter the noisy online information to collect vehicle
attributes, i.e., brand, production year, and body style, for
different car models from an online car information website 1 . It is worth noting that color is an important attribute,
and we will use it again in the third stage for semantic
filtering. Moreover, inspired by alternating optimization [38]
and human-diffusion interaction [39], [40], [41], we also
develop a prompt template to improve the quality of the
generated images. Specifically, we adjusted one component
of the prompt template based on feedback from the diffusion
1 https://www.autoevolution.com/

7320
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.

Select
attribute
keywords

Diffusion
Model

Prompt: a green 2013
Chevrolet Impala vehicle

Diffusion
Model

poor cross-image consistency

Filtering
Vehicle model
information

Prompt: a green
Chevrolet Impala vehicle

Prompt Template Design:
Iterate till the best prompt template is found

Vehicle model
attribute library

good cross-image consistency

Optimized
prompt library

Text
embeddings

Gradient update

1% Unlabeled real images
Text
Encoder

Captioned 1% Text-image Pairs

a black car driving
down the road

Image
Captioner

LoRA
Layers

Diffusion Model

……

MSE
Loss

Apply noise × n
Diffusion Model Selection

Optimal
Fine-Tuned
Diffusion
Model

Noisy adapted images

Correct Front Grills
! $
× ×48
4 4

Additional Weights
Trainable Parameters

Stage2: Diffusion Fine-Tuning
Stage3: Semantic Filtering

H×W×3
Images

Linear Embedding

Stage1: Prompt Processing

4

×

$
4

Stage 2

Swin
Transformer
Block

×2

! $
×
×4&
16 16

! $
× ×2 &
8 8

×&

Stage 1

Patch Partition

Frozen Parameters

!

Filtered adapted images

Swin
Transformer
Block

×2

! $
×
×8&
32 32

Stage 3

Swin
Transformer
Block

×6

reID model

Stage 4

Swin
Transformer
Block

×2

Pri
ma
ry
Cla
ssif
ier

……

Semantic
Filtering

ier

Optimized
prompt library

Reference
prompt

Patch Merging

Diffusion Text
Encoder Frozen

Au
xilia
ry C
las
sif

Wrong Front Grills

Patch Merging

Reference
prompt

Patch Merging

Diffusion Text
Encoder Tuned

Fig. 2: An overview of our coarse-to-fine cross-modality pipeline Vehicle-Diff. It has three stages: Prompt Processing,
Diffusion Fine-tuning, and Semantic Filtering. (1) We first scrape and filter vehicle model information from online vehicle
websites. Given the diffusion model, we then select the prompt template according to the visual quality. (2) In the second
stage, we leverage the off-the-shelf image captioner to generate the pseudo caption. It is worth noting that the proposed
pipeline only requests a few unlabeled real images from the downstream dataset. After the data preparation, we fine-tune
the diffusion model via Mean Squared Error (MSE) loss. (3) In the third stage, using the refined prompts, we choose the
most effective diffusion model by comparing visual quality, such as consistency. Then, we create synthetic data for the
vehicle re-identification task. We use the cross-modality model to filter out semantically misaligned data. Finally, we feed
the high-fidelity data to train the reID model via cross-entropy loss [35], [36] and circle loss [37].
model. The final prompt template is designed as “a [color]
[production year] [brand] [car model] [body style] driving
down the road.” In the bottom of Fig. 1, we show several
examples of the prompt template and the resulting images.
B. Diffusion Fine-tuning
Vehicle-Diff leverages a text-to-image diffusion model
to generate vehicle images according to prompts. However, a pre-trained diffusion model still struggles to adapt
well to the real-world vehicle images, resulting in a domain gap between synthesized images and those in vehicle
reID datasets. Therefore, we further fine-tune the diffusion
model to mitigate the domain discrepancy while retaining
its generation capability. As shown in Fig. 2 (Stage 2), we
illustrate the step-by-step fine-tuning stage from the data
preparation to the model optimization. To be specific, we
first deploy an image captioner, i.e., BLIP-2 [42], to predict
text prompts for unlabeled vehicle images, and then employ

the generated image-text pairs to fine-tune the text-to-image
diffusion model. We incorporate additional weights [43] in
the decoder part, while keeping the pre-trained weights
unchanged. Therefore, the additional weights could adapt the
final visual style, while maintaining the generative capability.
The optimization objective is the mean squared error (MSE)
loss. It is worth noting that, our Vehicle-Diff could be trained
with only a few (1%) unlabeled images of the vehicle
dataset for fine-tuning, i.e., 378 images for VeRi-776 and
527 images for CityFlowV2, while previous methods either
require large-scale datasets (GAN-based methods [10], [11])
or rely on labeled images (graphics-engine-based methods
[8], [9]). Moreover, different from these methods, VehicleDiff harnesses the generative power of diffusion models,
enabling to generate more realistic images, as shown in
Fig. 1. Similarly, we fine-tune multiple candidate diffusion
models in preparation for the next stage, which involves

7321
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.


codex
我只读了每篇前三页。总判断是：这批论文真正可借鉴的不是具体模块，而是把“模态差异”拆成更具体的结构性问题，例如通道不平衡、相机诱导伪标签噪声、四个异质空间漂移、模态属性不公平。我们要发 B 类，最好沿这个写法把团队资产里的 3D 几何和航拍地面极端视角变成问题本身。

**逐篇反推**

1. **CAFMNet**  
创新类型：工程组合偏新机制。  
为什么能发：把 VI-ReID 的模态差异细化成通道贡献不均衡，证明不同通道含有不同身份线索，再用通道优化、通道精炼、多维特征优化串起来。  
story 套路：不是再生成辅助模态，而是直接挖 RGB 和 IR 原始通道里的身份信息。

2. **CCAFL**  
创新类型：新数据设定加新机制。  
为什么能发：Intra-camera supervised ReID 只标每个相机内部身份，不标跨相机关联，成本低但跨相机难学。它用 CLIP prompt 提供语义监督，再用相机内细粒度学习和相机间对抗学习压掉相机差异。  
story 套路：弱标注不是退而求其次，而是更现实的监督形式，核心是学相机无关身份特征。

3. **CDFM**  
创新类型：新机制。  
为什么能发：指出 CLIP 全局对齐不够，外部解析和姿态工具又带依赖，于是直接在 CLIP 内部挖细粒度图文关系。  
story 套路：不靠外部工具，从预训练空间里把细粒度局部语义“挖出来”。

4. **MCST**  
创新类型：问题重定义加新机制加新数据设定。  
为什么能发：CLIP 用到 VI-ReID 后，不再只是可见光和红外两个空间，而是可见图像、红外图像、可见文本、红外文本四个异质空间。它把问题讲成四空间对齐和静态 prompt 漂移，再用模态居中和交替训练解决。  
story 套路：CLIP 不是直接解决跨模态，反而引入新的异质空间，需要重新设计对齐机制。

5. **CLNS**  
创新类型：问题重定义加新机制。  
为什么能发：把无监督 VI-ReID 的伪标签噪声具体化为相机诱导的结构性噪声，不是普通随机噪声。证据是同身份跨相机距离明显大于同相机距离，导致身份碎裂。  
story 套路：先证明噪声有结构，再做从原型到实例的逐级净化。

6. **Stable Diffusion cloth-debiasing**  
创新类型：工程组合偏新数据生成。  
为什么能发：换衣 ReID 中模型依赖衣服，GAN 换衣质量不稳，所以用 Stable Diffusion 加人体解析和姿态控制生成统一衣服，再用过滤器筛数据。  
story 套路：把衣服空间统一，让模型被迫看非衣服身份线索。这个方向本身容易变成生成数据工程，除非几何约束成为核心贡献。

7. **CMAG**  
创新类型：新机制加工程组合。  
为什么能发：无监督 VI-ReID 同时有模态差异、批训练局限、伪标签噪声和相机偏置。它用环形拓扑验证伪标签，用跨模态注意力和图记忆补全全局关系。  
story 套路：直接相似度不可靠，高阶结构关系比两两距离更稳。

8. **Vehicle-Diff**  
创新类型：新数据生成设定。  
为什么能发：真实车辆 ReID 数据稀缺，3D 渲染有域差，GAN 属性一致性差。它用车辆属性 prompt、少量无标注目标域微调扩散模型，再用视觉语言模型过滤，构造大规模合成数据。  
story 套路：不是随便生成图，而是用属性和语义过滤保证身份细粒度一致性。

9. **ColorSketchNet**  
创新类型：问题重定义加新机制。  
为什么能发：它把 text、sketch、RGB 的差异讲成“模态属性不公平”，例如 text 有颜色但轮廓弱，sketch 有轮廓但没颜色。用彩色素描作为辅助模态补齐缺失属性。  
story 套路：跨模态对齐前，先让不同模态有公平的属性空间。

**给我们的强候选**

1. **可见表面公平匹配：航拍地面 ReID 的核心不是视角差，而是两端可见的身体表面不同。**  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
区别：最像 ColorSketchNet 和 MCST，但它们补的是文本、素描或红外的属性差异；我们补的是由 3D 视角造成的可见身体表面不对称。  
cheap kill-switch：冻结 SOLIDER，用 SMPL 或 2D pose 把特征池化到身体部位，只比较双方共同可见的部位。若在 CARGO 或 AG-ReID.v2 上不能超过全局特征的噪声线，就停。

2. **几何校准伪标签去噪：航拍地面无监督或少标注 ReID 的伪标签噪声是高度、俯仰角和相机结构造成的身份碎裂。**  
挂靠资产：aerial-ground 数据、SOLIDER 特征、SMPL 可见性。  
区别：最像 CLNS 和 CMAG，但它们只用相机一致性或拓扑关系；我们有物理几何，可判断两个样本是不是因为视角不可比才距离远。  
cheap kill-switch：先不训练，直接用 SOLIDER 聚类，统计同身份跨高度或跨俯仰角距离是否显著大于同视角距离，再做一个几何加权的原型合并。若碎裂不随几何变化，方向判弱。

3. **SMPL surface token memory：把 Swin 图像 token 映射到规范人体表面坐标，而不是在二维图像平面上硬对齐。**  
挂靠资产：SOLIDER-Swin 的 768 维特征、SMPL mesh、joints 和投影基建。  
区别：最像 CDFM 和 CAFMNet，但它们挖的是 2D 局部 token 或通道；我们把局部特征放到 3D 人体表面索引上，天然处理航拍俯视、地面侧视和遮挡可见性。  
cheap kill-switch：冻结 backbone，只训练一个很轻的 surface token 投影头，和水平条带、pose part pooling 对比。若规范表面特征没有更稳的跨视角同身份距离，就停。

4. **SMPL 约束的反事实视角生成：生成不是为了扩数据，而是为了制造同一 3D 身份在航拍和地面之间的可控视角反事实。**  
挂靠资产：SMPL、pose、aerial-ground。  
区别：最像 Stable Diffusion cloth-debiasing 和 Vehicle-Diff，但它们主要靠生成质量和过滤；我们的切开点是 3D 身份一致性和可见面一致性，生成只是验证几何反事实。  
cheap kill-switch：只做小规模，选几十个身份生成有限反事实样本，用现有 ReID 训练短跑对比普通增广。若过滤后保留率低，或收益被普通模糊、裁剪、随机擦除吃掉，就不继续。

我最建议优先押第 1 个，备选第 3 个。第 2 个适合做少标注或无监督扩展，第 4 个风险最高，只有在几何约束真的能稳定生成身份一致样本时才值得加大投入。
tokens used
107,805
我只读了每篇前三页。总判断是：这批论文真正可借鉴的不是具体模块，而是把“模态差异”拆成更具体的结构性问题，例如通道不平衡、相机诱导伪标签噪声、四个异质空间漂移、模态属性不公平。我们要发 B 类，最好沿这个写法把团队资产里的 3D 几何和航拍地面极端视角变成问题本身。

**逐篇反推**

1. **CAFMNet**  
创新类型：工程组合偏新机制。  
为什么能发：把 VI-ReID 的模态差异细化成通道贡献不均衡，证明不同通道含有不同身份线索，再用通道优化、通道精炼、多维特征优化串起来。  
story 套路：不是再生成辅助模态，而是直接挖 RGB 和 IR 原始通道里的身份信息。

2. **CCAFL**  
创新类型：新数据设定加新机制。  
为什么能发：Intra-camera supervised ReID 只标每个相机内部身份，不标跨相机关联，成本低但跨相机难学。它用 CLIP prompt 提供语义监督，再用相机内细粒度学习和相机间对抗学习压掉相机差异。  
story 套路：弱标注不是退而求其次，而是更现实的监督形式，核心是学相机无关身份特征。

3. **CDFM**  
创新类型：新机制。  
为什么能发：指出 CLIP 全局对齐不够，外部解析和姿态工具又带依赖，于是直接在 CLIP 内部挖细粒度图文关系。  
story 套路：不靠外部工具，从预训练空间里把细粒度局部语义“挖出来”。

4. **MCST**  
创新类型：问题重定义加新机制加新数据设定。  
为什么能发：CLIP 用到 VI-ReID 后，不再只是可见光和红外两个空间，而是可见图像、红外图像、可见文本、红外文本四个异质空间。它把问题讲成四空间对齐和静态 prompt 漂移，再用模态居中和交替训练解决。  
story 套路：CLIP 不是直接解决跨模态，反而引入新的异质空间，需要重新设计对齐机制。

5. **CLNS**  
创新类型：问题重定义加新机制。  
为什么能发：把无监督 VI-ReID 的伪标签噪声具体化为相机诱导的结构性噪声，不是普通随机噪声。证据是同身份跨相机距离明显大于同相机距离，导致身份碎裂。  
story 套路：先证明噪声有结构，再做从原型到实例的逐级净化。

6. **Stable Diffusion cloth-debiasing**  
创新类型：工程组合偏新数据生成。  
为什么能发：换衣 ReID 中模型依赖衣服，GAN 换衣质量不稳，所以用 Stable Diffusion 加人体解析和姿态控制生成统一衣服，再用过滤器筛数据。  
story 套路：把衣服空间统一，让模型被迫看非衣服身份线索。这个方向本身容易变成生成数据工程，除非几何约束成为核心贡献。

7. **CMAG**  
创新类型：新机制加工程组合。  
为什么能发：无监督 VI-ReID 同时有模态差异、批训练局限、伪标签噪声和相机偏置。它用环形拓扑验证伪标签，用跨模态注意力和图记忆补全全局关系。  
story 套路：直接相似度不可靠，高阶结构关系比两两距离更稳。

8. **Vehicle-Diff**  
创新类型：新数据生成设定。  
为什么能发：真实车辆 ReID 数据稀缺，3D 渲染有域差，GAN 属性一致性差。它用车辆属性 prompt、少量无标注目标域微调扩散模型，再用视觉语言模型过滤，构造大规模合成数据。  
story 套路：不是随便生成图，而是用属性和语义过滤保证身份细粒度一致性。

9. **ColorSketchNet**  
创新类型：问题重定义加新机制。  
为什么能发：它把 text、sketch、RGB 的差异讲成“模态属性不公平”，例如 text 有颜色但轮廓弱，sketch 有轮廓但没颜色。用彩色素描作为辅助模态补齐缺失属性。  
story 套路：跨模态对齐前，先让不同模态有公平的属性空间。

**给我们的强候选**

1. **可见表面公平匹配：航拍地面 ReID 的核心不是视角差，而是两端可见的身体表面不同。**  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。  
区别：最像 ColorSketchNet 和 MCST，但它们补的是文本、素描或红外的属性差异；我们补的是由 3D 视角造成的可见身体表面不对称。  
cheap kill-switch：冻结 SOLIDER，用 SMPL 或 2D pose 把特征池化到身体部位，只比较双方共同可见的部位。若在 CARGO 或 AG-ReID.v2 上不能超过全局特征的噪声线，就停。

2. **几何校准伪标签去噪：航拍地面无监督或少标注 ReID 的伪标签噪声是高度、俯仰角和相机结构造成的身份碎裂。**  
挂靠资产：aerial-ground 数据、SOLIDER 特征、SMPL 可见性。  
区别：最像 CLNS 和 CMAG，但它们只用相机一致性或拓扑关系；我们有物理几何，可判断两个样本是不是因为视角不可比才距离远。  
cheap kill-switch：先不训练，直接用 SOLIDER 聚类，统计同身份跨高度或跨俯仰角距离是否显著大于同视角距离，再做一个几何加权的原型合并。若碎裂不随几何变化，方向判弱。

3. **SMPL surface token memory：把 Swin 图像 token 映射到规范人体表面坐标，而不是在二维图像平面上硬对齐。**  
挂靠资产：SOLIDER-Swin 的 768 维特征、SMPL mesh、joints 和投影基建。  
区别：最像 CDFM 和 CAFMNet，但它们挖的是 2D 局部 token 或通道；我们把局部特征放到 3D 人体表面索引上，天然处理航拍俯视、地面侧视和遮挡可见性。  
cheap kill-switch：冻结 backbone，只训练一个很轻的 surface token 投影头，和水平条带、pose part pooling 对比。若规范表面特征没有更稳的跨视角同身份距离，就停。

4. **SMPL 约束的反事实视角生成：生成不是为了扩数据，而是为了制造同一 3D 身份在航拍和地面之间的可控视角反事实。**  
挂靠资产：SMPL、pose、aerial-ground。  
区别：最像 Stable Diffusion cloth-debiasing 和 Vehicle-Diff，但它们主要靠生成质量和过滤；我们的切开点是 3D 身份一致性和可见面一致性，生成只是验证几何反事实。  
cheap kill-switch：只做小规模，选几十个身份生成有限反事实样本，用现有 ReID 训练短跑对比普通增广。若过滤后保留率低，或收益被普通模糊、裁剪、随机擦除吃掉，就不继续。

我最建议优先押第 1 个，备选第 3 个。第 2 个适合做少标注或无监督扩展，第 4 个风险最高，只有在几何约束真的能稳定生成身份一致样本时才值得加大投入。
