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
session id: 019ef71f-6fb7-7000-9e7f-76c9e6e02ebd
--------
user
你是 ReID 论文创新挖掘员 6/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(6/20):
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我会按项目规则先尝试读取目标和状态文件；如果当前沙盒只允许读论文目录，我会说明限制并继续只基于这批 PDF 的前 3 页做判断。随后按你的要求逐篇用 `pdftotext -l 3` 提取摘要和引言。
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
我已经读到当前目标是“先读文献做差距分析，提出候选方向，先不要开实验”。现在先核对这 9 个 PDF 的文件名，然后分批提取前 3 页文本，重点抓摘要、引言里的问题定义、机制和证据组织方式。
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
Adversarial perturbation and defense for generalizable person re-identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
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
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
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

exec
/bin/zsh -lc 'pdftotext -v' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
pdftotext version 26.04.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org
Copyright 1996-2011, 2022 Glyph & Cog, LLC

codex
文件名都在目录里，`pdftotext` 也可用。接下来分三批读，避免一次输出太长导致摘要和引言被截断。
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
/bin/zsh -lc "pdftotext -l 3 'CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CVAF: A CLIP-Based View-Consistent Alignment Framework
for Aerial-Ground Person Re-Identification
DONGXU MAO, SHANGZHI TENG, and XUEQIANG LYU, School of Computer Science, Beijing
Information Science and Technology University, Beijing, China
With the increasing adoption of UAV platforms in areas such as public safety and smart cities, AerialGround Person Re-Identification (AGPReID) has emerged as a crucial yet highly challenging task, garnering
growing interest from the research community. While existing approaches have leveraged identity attributes
and viewpoint disentanglement strategies to improve cross-view matching, their heavy reliance on prior
knowledge often compromises model generalization. Furthermore, some methods that explicitly separate
viewpoints may unintentionally discard identity-related, view-invariant features, leading to incomplete
identity representations. To address these limitations, we propose a CLIP-based View-Consistent Alignment
Framework (CVAF) with two training stages. In the first stage, learnable text tokens are employed to represent
identity-aware textual descriptions. To promote consistent alignment across varying viewpoints, we introduce
a Text Consistency Loss (TCL) that regularizes the stability of text-token interactions with multi-view images.
In the second stage, we present a Semantic Filtering Module (SFM) that jointly modulates image patch tokens
along spatial and channel dimensions. A text-guided cross-attention mechanism generates spatial attention
maps to explicitly emphasize identity-relevant regions, while semantic matching between textual features
and visual tokens enables adaptive reweighting of image representations, effectively suppressing background
clutter and view-specific noise. Extensive experiments on multiple AGPReID datasets demonstrate that our
CVAF outperforms the state-of-the-art methods.
CCS Concepts: • Information systems → Information retrieval; • Computing methodologies → Image
representations;
Additional Key Words and Phrases: Vision-language Learning, Aerial-Ground View, Person Re-Identification,
Image Retrieval
ACM Reference format:
Dongxu Mao, Shangzhi Teng, and Xueqiang Lyu. 2026. CVAF: A CLIP-Based View-Consistent Alignment
Framework for Aerial-Ground Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 22, 3,
Article 85 (February 2026), 19 pages.
https://doi.org/10.1145/3785482

This work is supported by the National Natural Science Foundation of China (Grants Nos. 62202061 and 62171043), the
Beijing Natural Science Foundation (Grant Nos. 4232025 and 4254096), and the Research Program of Beijing Municipal
Education Commission (Grant No. KM202311232002).
Authors’ Contact Information: Dongxu Mao, School of Computer Science, Beijing Information Science and Technology
University, Beijing, China; e-mail: 2023020673@bistu.edu.cn; Shangzhi Teng (corresponding author), School of Computer
Science, Beijing Information Science and Technology University, Beijing, China; e-mail: tengshangzhi@bistu.edu.cn;
Xueqiang Lyu, School of Computer Science, Beijing Information Science and Technology University, Beijing, China; e-mail:
lxq@bistu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2026/2-ART85
https://doi.org/10.1145/3785482
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

85:2
1

D. Mao et al.

Introduction

Person Re-Identification (ReID) is a fundamental task in computer vision that aims to identify
and match pedestrian instances across images captured by multiple non-overlapping cameras.
Traditional ReID methods primarily focus on image sourced from ground-only cameras. With
the proliferation of Unmanned Aerial Vehicles (UAVs) [17, 20, 51], their high mobility and
top-down viewpoints offer valuable complements to ground cameras by enhancing coverage,
reducing occlusion, and enabling more flexible deployment. The integration of aerial and ground
perspectives thus facilitates the construction of more comprehensive and adaptive intelligent
surveillance systems. However, the substantial appearance variations and scale discrepancies
introduced by heterogeneous camera views in aerial-ground networks also pose new challenges
for robust person ReID.
Figure 1 presents a visual comparison between pedestrian images captured by ground-only
cameras and those captured under aerial-ground cameras. Ground-only cameras typically provide
frontal or side views of pedestrians, where clothing and appearance features are clearly visible. In
contrast, aerial cameras—due to their elevated viewpoints and top-down perspectives—primarily
observe the head and upper body regions. As a result, pedestrians appear smaller, with blurred
visual details, distorted poses, and inconsistent aspect ratios. Despite significant progress in person
ReID, existing methods [7, 8, 13, 30, 32, 34, 37] often fall short in aerial-ground scenarios due to their
reliance on datasets captured exclusively under ground-only camera settings. These approaches are
typically optimized for consistent viewpoints and limited intra-class variation, yet they encounter
challenges when applied to aerial-ground scenarios. In Aerial-Ground Person Re-Identification
(AGPReID) task, the same identity frequently appears across drastically different views, resulting
in substantial intra-class discrepancies and making cross-view matching particularly challenging.
To address this issue, AG-ReID [18] leverages identity attributes as auxiliary, view-invariant feature
to bridge the appearance gap. More recent efforts such as VDT [40] and ViT-based disentanglement
frameworks attempt to explicitly decouple viewpoint and identity information, enabling more
robust feature learning across heterogeneous camera views.
While both attribute-based and view-disentanglement approaches have shown promise in addressing the severe viewpoint discrepancies of AGPReID, they each exhibit inherent limitations.
Attribute-based methods depend on predefined or detector-generated soft-biometric cues such as
color, clothing type, or accessories. However, these cues become unreliable under aerial viewpoints
due to low resolution, missing body parts, and reduced visual detail. Such coarse and incomplete attributes restrict the model from autonomously discovering fine-grained, view-invariant
patterns—such as stable body proportions, global silhouette geometry, or consistent structural
layouts—that remain shared across aerial and ground views but fall outside the scope of manually defined labels. Conversely, view-disentanglement methods introduce explicit mechanisms
(e.g., view tokens, subtractive separation, orthogonality constraints) to isolate view-related factors from identity features. Yet, in aerial-ground scenarios, identity cues and viewpoint cues are
intrinsically entangled. Many cross-view shared cues—such as approximate body shape, global
contour, and structural transitions—are partially view-dependent and thus cannot be perfectly
separated. Over-aggressive disentanglement may inadvertently discard these shared cues, weakening discriminability and generalization across views. These limitations motivate the need for an
approach that preserves such view-consistent structural information while enabling the model to
learn robust, view-invariant representations without relying on coarse attributes or rigid factor
decomposition.
To address the aforementioned limitations, we leverage the strong cross-modal alignment capabilities of CLIP as the foundation of our method. CLIP is pretrained on a large-scale corpus of
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.

CVAF for Aerial-Ground Person Re-Identification

85:3

Fig. 1. (a) AGPReID vs. (b) traditional person ReID. AGPReID integrates views from aerial and ground
perspectives, introducing significant intra-class variations due to drastic viewpoint changes and posing
greater challenges for robust feature learning. In contrast, traditional person ReID focuses on ground-view
images from a single perspective, where intra-class variations are relatively small. AGPReID, Aerial-Ground
Person Re-Identification.

image-text pairs spanning diverse scenes, viewpoints, and visual appearances, which implicitly
endows it with the ability to associate semantically consistent content across different views. This
inherent generalization ability makes CLIP particularly suitable for the AGPReID task, where large
viewpoint and modality discrepancies pose significant challenges. Building upon this foundation,
we propose a CLIP-Based View-Consistent Alignment Framework (CVAF) with two training
stages for AGPReID task. CVAF leverages the powerful vision-language pretrained model CLIP [10]
to extract robust and view-invariant identity representations. In the first stage, we introduce a set of
learnable identity-aware text tokens, optimized via a supervised contrastive loss to align images and
textual descriptions across views. To ensure that these tokens capture consistent identity semantics,
we further propose a Text Consistency Loss (TCL), which enforces stable alignment between
multi-view image features and their shared identity description. This encourages the model to learn
stable, semantic-level representations that generalize across diverse viewpoints without relying on
manually labeled attributes. In the second stage, the learned text tokens are reused to guide image
feature refinement via a Semantic Filtering Module (SFM). SFM modulates image patch tokens
jointly across spatial and channel dimensions. It first applies text-guided cross-attention to generate
a spatial attention map, explicitly highlighting identity-relevant regions. Then, it reweights image
tokens based on their semantic similarity to the text features, suppressing background noise and
view-specific artifacts. Without the need for explicit viewpoint separation, SFM eliminates the risk
of identity feature loss while still extracting robust, view-invariant representations.
In summary, our main contributions are as follows:
— We present CVAF, a CLIP-based AGPReID framework that effectively aligns multimodal
representations and significantly improves cross-view matching performance.
— We introduce a novel TCL to enhance the stability and invariance of learned identity-aware
text tokens, and design a lightweight SFM to emphasize identity-relevant features.
— Extensive experiments on multiple AGPReID benchmarks demonstrate that CVAF consistently
outperforms existing state-of-the-art methods.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 3, Article 85. Publication date: February 2026.


exec
/bin/zsh -lc "pdftotext -l 3 'Cross-modality average precision optimization for visible thermal person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 164 (2025) 111489

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Cross-modality average precision optimization for visible thermal person
re-identification
Yongguo Ling a,e , Zhiming Luo b , Dazhen Lin b , Shaozi Li b , Min Jiang b , Nicu Sebe c ,
Zhun Zhong d ,∗
a

School of Computer, Electronics and Information, Guangxi University, Nanning 530004, China

b Department of Artificial Intelligence, Xiamen University, Xiamen 361005, China
c Department of Information Engineering and Computer Science, University of Trento, Trento 38123, Italy
d School of Computer Science and Information Engineering, Hefei University of Technology, 230009, Hefei, China
e

Guangxi Key Laboratory of Multimedia Communications and Network Technology, Nanning, 530004, Guangxi, China

ARTICLE

INFO

Keywords:
Person re-identification
Cross-modality
Average precision
Modality alignment

ABSTRACT
Metric learning has emerged as a popular approach for addressing the challenges of visible thermal person
re-identification (VT-ReID), such as the cross-modality discrepancy and intra-class variations. However, existing
metric learning-based methods often focus on optimizing the model for hard positive samples, neglecting the
importance of high-ranking ones, due to failing to consider the overall ranking order within a batch. To
overcome this limitation, we propose a novel approach called Cross-modality Average Precision (CAP) that
directly optimizes the cross-modality overall ranking order in VT-ReID. Unlike the recently introduced Smooth
Average Precision (Smooth-AP), which primarily corrects misordered samples at high ranks, CAP specifically
targets the main challenge of cross-modality discrepancy in VT-ReID. Our method involves setting a query
instance from one modality and calculating the CAP using galleries from another modality. CAP encompasses
two complementary aspects: CAP with Visible queries (CAPV) and CAP with Thermal queries (CAPT). By
jointly optimizing these two aspects, we can effectively improve the cross-modality overall ranking order.
Additionally, to enhance the effectiveness of CAP, we introduce two techniques. The first technique is Dynamic
Modality Alignment (DMA), which reduces the cross-modality discrepancy by adaptively adjusting the weights
of modality alignment. The second technique involves implementing CAP and DMA on the Global and Local
Features (GLF), enabling us to optimize the model at both global and local levels, further enhancing the
advantages of CAP and DMA. We conducted extensive experiments on two VT-ReID datasets, and the results
demonstrate the effectiveness of our proposed method, which achieves state-of-the-art performance.

1. Introduction
Person re-identification (ReID) is a task that involves matching a
specific query person from a set of gallery images captured by nonoverlapping cameras. Traditional ReID [1] assumes that both the query
and gallery images are obtained from RGB cameras. However, the
reliance on visible (RGB) images in illumination environments makes
them susceptible to significant changes under poor lighting conditions,
such as night-time. To address this issue, researchers have proposed
the collection of thermal images using thermal cameras in scenarios
with poor illumination [2]. In this paper, we focus on the problem of
matching person images between visible and thermal cameras, which
is commonly referred to as visible-thermal person re-identification (VTReID) in the research community [3]. Compared to traditional ReID,

VT-ReID presents more significant challenges. The primary difficulties
stem from the substantial inter-modality discrepancy caused by the utilization of different modalities and the presence of intra-class variations
resulting from environmental factors (e.g., illumination) and personal
changes (e.g., pose). (Fig. 1).
The primary objective of visible-thermal person re-identification
(VT-ReID) is to align features from different modalities and learn a
shared semantic embedding space that facilitates accurate matching
of person identities across modalities. To address this goal, several
metric learning methods [4,5] have been proposed to mitigate the
cross-modality discrepancy and intra-class variations. However, these
approaches typically perform gradient updates using a small number
of sample pairs, such as the triplet loss [6] (Fig. 2(a)) , leading to

∗ Corresponding author at: School of Computer Science and Information Engineering, Hefei University of Technology, 230009, Hefei, China.

E-mail addresses: ygling@gxu.edu.cn (Y. Ling), zhiming.luo@xmu.edu.cn (Z. Luo), dzlin@xmu.edu.cn (D. Lin), szlig@xmu.edu.cn (S. Li),
minjiang@xmu.edu.cn (M. Jiang), niculae.sebe@unitn.it (N. Sebe), zhunzhong@hfut.edu.cn (Z. Zhong).
https://doi.org/10.1016/j.patcog.2025.111489
Received 7 October 2023; Received in revised form 31 December 2024; Accepted 19 February 2025
Available online 28 February 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

Fig. 1. Illustration of the main challenges in VT-ReID. The inter-modality discrepancy
caused by different modalities and the intra-class variations caused by different
illumination, view, and pose.

Fig. 2. Illustration of the difference between triplet loss, original Smooth-AP, and
our Cross-modality Average Precision (CAP). Shapes indicate the identities and colors
represent the modalities (blue for visible and yellow for thermal). (a) Triplet loss
performs gradient updates with a couple of sample pairs, potentially encountering
difficulties in escaping local optima because of its constrained ability to capture the
overall ranking order. (b) Original Smooth-AP will prioritize optimizing the ranking
of the intra-modality instances (e.g. 𝑝1, 𝑝2) while ignoring the inter-modality ones
(e.g. 𝑝3) due to its inherent characteristics. (c) Our CAP considers handling the large
cross-modality gap in VT-ReID, which can directly optimize the cross-modality overall
ranking orders. (For interpretation of the references to color in this figure legend, the
reader is referred to the web version of this article.)

an over-focus on optimizing low-ranking (located at the rear of the
ranking list with lower similarity) positive samples at the expense
of high-ranking (located at the front of the ranking list with higher
similarity) ones, due to the failure to consider the overall ranking
order [7]. This issue is particularly problematic in VT-ReID due to
the simultaneous presence of cross-modality discrepancy and intraclass variations. Furthermore, these metric learning methods often rely
on intricate sampling strategies and pairing losses, requiring extensive
experimentation and empirical fine-tuning.
Recently, Smooth-AP [8] introduced an approximation method to
calculate differentiable average precision, which directly optimizes the
overall ranking order. However, the original Smooth-AP fails to address
the main challenge of cross-modality in VT-ReID. Specifically, SmoothAP prioritizes correcting misordered samples at high ranks [8], while
in the cross-modality VT-ReID task, the inter-modality distance is typically larger than the intra-modality distance. As a result, it prioritizes
optimizing the ranking of intra-modality instances (high rank 𝑝1, 𝑝2)
while disregarding the ranking of inter-modality instances (low rank
𝑝3) (Fig. 2(b)), leading to inferior performance.
To overcome these limitations, we propose a differentiable Crossmodality Average Precision (CAP) method that explicitly addresses
the substantial cross-modality discrepancy in VT-ReID. CAP directly
optimizes the cross-modality overall ranking order within a batch.
Specifically, we introduce a query instance from one modality and
calculate CAP using galleries from another modality. CAP incorporates
two aspects 2(c): CAP with Visible queries (CAPV) and CAP with
Thermal queries (CAPT). These two aspects mutually reinforce each
other and jointly optimize the cross-modality overall ranking order.
To further enhance the effectiveness of CAP, we introduce two
techniques. Firstly, recognizing that the large cross-modality discrepancy hinders CAP performance, we propose Dynamic Modality Alignment (DMA) to mitigate this discrepancy. DMA involves constructing
a dynamic Cross-modality Affiliated Matrix (CAM) that assigns higher
weights to cross-modality sample pairs exhibiting smaller non-modality
variations (e.g., same view and pose). By encouraging the network
to focus on reducing the cross-modality discrepancy while ignoring
non-modality variations during alignment, DMA mitigates the influence
of non-modality variations and achieves superior modality alignment.
On the second aspect, some elaborate works have demonstrated the
effectiveness of local features [9,10] in VT-ReID. Motivated by these
findings, we extract Global and Local Features (GLF) jointly to enhance
feature discrimination. CAP and DMA are then applied individually
to these features, further improving the performance of the proposed
method.

Our contributions can be summarized as follows:
• We develop a differentiable Cross-modality Average Precision
(CAP) to directly optimize the cross-modality overall ranking orders, which explicitly handle the large cross-modality discrepancy
in the VT-ReID task.
• We propose a Dynamic Modality Alignment (DMA) to reduce
the cross-modality discrepancy by constructing a dynamic Crossmodality Affiliated Matrix (CAM), which can alleviate the influence of the non-modality variations and achieve better modality
alignment, facilitating the effectiveness of CAP.
• We apply our CAP and DMA to both Global and Local Features (GLF) individually. By incorporating them into a carefully
designed global–local structure, we achieve state-of-the-art performance on two datasets.
2. Related work
Object re-identification (ReID) [11] is a sub-task within the broader
field of image retrieval. It encompasses various domains, such as
building retrieval [12], drone-based geo-localization [13], vehicle reidentification [14,15], and person re-identification [16].
2.1. VT-ReID
Visible-thermal person re-identification (VT-ReID) was first introduced by [2], which aims to match the query person of one modality
from the gallery of another one. Since then, many methods have been
proposed for VT-ReID, which can be mainly divided into four groups.
(1) Feature extractor based methods aim to design a cross-modality
structure to extract modality-invariant and discriminative features. For
example,
Ye et al. [17] introduced a modality-aware collaborative ensemble learning method and middle-level sharable two-stream network
to handle modality discrepancies at both feature and classifier levels.
Some methods [18,19] focus on extracting the identity-related feature
2

Pattern Recognition 164 (2025) 111489

Y. Ling et al.

by explicitly removing the irrelevant information. Fu et al. [20] used
a neural architecture search method to automatically search for the
best segmentation scheme, and determine which BN layer needs to be
segmented. To improve the feature discrimination, the global and local
features are applied to the cross-modality matching [10,21].
(2) Metric learning based methods aim to learn an embedding space
by explicitly reducing the distances between intra-identify samples of
two modalities. Hao et al. [4] map the features of two modalities into a
hypersphere manifold, and then constrain the intra-modality variations
and inter-modality discrepancy in this manifold. Ye et al. [5] introduce
a triplet loss with a bidirectional exponential angle to optimize the
angle discriminative features of two modalities samples. In addition,
Liu et al. [22] proposed dual-granularity triplet loss hierarchically
integrates sample-based and center-based triplet losses using simple
configurations like pooling and batch normalization.
(3) Distribution alignment based methods are designed to learn
modality-invariant features by decreasing the distribution discrepancy
of two modalities. Wu et al. [23] propose a modality alleviation
structure and a pattern alignment structure to align two modalities.
Zhao et al. [24] leverages color-irrelevant consistency learning to
extract color-agnostic features, and identity-aware modality adaptation
to align feature distributions at the identity level. In order to explore
nuances of information, Zhang et al. [25] introduce a method to embed
two modalities images into a 3D public space, and use a contrastive
association structure to learn contrastive features. Park et al. [26]
introduce a dense correspondence relationship between visible and
thermal modalities to match the corresponding pedestrian parts of the
two modalities.
(4) Image generation based methods attempt to bridge the modality
gap in the image-level by image translation techniques. Li et al. [27]
and Wei et al. [28] utilize a lightweight network to generate images
from one modality to another modality, and learn modality-invariant
embedding representation from these three modalities. In order to
reduce the influence of id-unrelated factors in features, some methods [29,30] use variational autoencoders and generative adversarial
networks to decompose features into two factors, id-related and idunrelated features, where id-related features with rich identify information are used for cross-modality retrieval. Liu et al. [31] attempt to
generate high-quality images to smooth the large inter-modality gap.

identity classification (𝐿ID ), the losses of Cross-modality Average Precision (𝐿CAP ), and the losses of Dynamic Modality Alignment (𝐿DMA )
with each type of feature. These combined losses are jointly used as the
objective function to optimize the network in an end-to-end manner.
Specifically, the 𝐿ID is mainly used to reduce intra-class variations, the
𝐿CAP can directly optimize the cross-modality overall ranking orders in
a batch (Fig. 4(a → c)), and the 𝐿DMA can alleviate the influence of
non-modality variations and effectively smooth the inter-modality gap
(Fig. 4(a → b)).
3.1. Cross-modality average precision
Existing metric learning methods will over-focus on optimizing lowranking positive samples at the expense of high-ranking ones due to
a lack of consideration of the overall ranking orders. Moreover, these
methods usually require to be well-designed for jointly handling crossmodality discrepancy and intra-class variations, which require huge
experimentation and empirical practice. Inspired by Smooth-AP [8], we
propose a Cross-modality Average Precision (CAP) to directly optimize
the cross-modality overall ranking orders. We next introduce CAP in
detail.
True average precision. We set 𝑠𝑖 as the cosine similarity between
the query and the sample 𝑖. The ranking of sample 𝑖 in any set 𝑆 can
be defined as:
∑
𝑅(𝑖, 𝑆) = 1 +
𝑢(𝑠𝑖 − 𝑠𝑗 ),
(1)
𝑗∈𝑆 ,𝑗≠𝑖

where 𝑢(𝑥) is a Heaviside step function, which will be set to 1 when
𝑥 > 0, otherwise set to 0.
Given a query 𝑓 𝑞 , and gallery set 𝐺𝑞 = {𝑓 𝑖 , 𝑖 = 1, … , 𝑁}. For
each query 𝑓 𝑞 , the gallery set can split into a positive set 𝑃 𝑞 and a
negative set 𝑁 𝑞 , which are formed by samples with the same ID and
with different IDS, respectively. The average precision of a query 𝑓 𝑞 is
defined as:
1 ∑ 𝑅 (𝑖, 𝑃 𝑞 )
𝐴𝑃 (𝑓 𝑞 ) = 𝑞
|𝑃 | 𝑖∈𝑃 𝑞 𝑅 (𝑖, 𝐺𝑞 )
∑
(2)
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑢(𝑠𝑖𝑗 )
1 ∑
,
= 𝑞
∑
∑
|𝑃 | 𝑖∈𝑃 𝑞 1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑢(𝑠𝑖𝑗 ) + 𝑗∈𝑁 𝑞 𝑢(𝑠𝑖𝑗 )
where 𝑠𝑖𝑗 = 𝑠𝑖 − 𝑠𝑗 , |𝑃 𝑞 | is the instance number of 𝑃 𝑞 set. It is noticed
that the derivative of the Heaviside step function (𝛿(𝑥) = 𝑑 𝑢(𝑥)∕𝑑 𝑥) is
flat with zero except 𝑥 = 0 point (Fig. 5(a)), and thus cannot be used
to optimize the model.
Approximate differentiable average precision. To address this
issue, we use an approximate method to calculate the average precision
by replacing the Heaviside step function 𝑢(𝑥) with a sigmoid-derived
function 𝑔(𝑥), which is defined as:
1
(3)
𝑔(𝑥) =
−𝑥 ,
1+𝑒 𝜏

2.2. Optimizing average precision
Average precision is a standard metric for retrieval tasks. Recently,
directly optimizing average precision based methods [32–34] have
been proposed to address the challenge of non-differentiability average
precision in the retrieval community. Such as using an approximation
derived from distance quantization [35], and a histogram binning
approximation [7], relaxing indicator function with a sigmoid function [8]. Recently, Ramziet al. [36] used an upper bound to optimize the average precision. Li et al. [37] propose PNP to optimize
the negative instances before the positive ones. Distinguishing itself
from previous research efforts, our approach explicitly addresses the
formidable challenge posed by the substantial cross-modality discrepancy. We introduce a novel concept termed ‘‘Cross-modality Average
Precision’’, which allows us to directly optimize the global ranking
orders across different modalities, specifically involving queries and
galleries. This marks a pioneering endeavor in the realm of addressing
the VT-ReID problem by directly optimizing cross-modality average
precision.

where 𝜏 is a hyper-parameter that adjusts the sharpness. The sigmoid
derived function 𝑔(𝑥) and its derivative 𝑑 𝑔(𝑥)∕𝑑 𝑥 are shown in Fig. 5(b).
Therefore, the average precision in Eq. (2) can be approximated as:
∑
1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 )
1 ∑
𝐴𝑃 (𝑓 𝑞 ) ≈ 𝑞
.
(4)
∑
∑
|𝑃 | 𝑖∈𝑃 𝑞 1 + 𝑗∈𝑃 𝑞 ,𝑗≠𝑖 𝑔(𝑠𝑖𝑗 ) + 𝑗∈𝑁 𝑞 𝑔(𝑠𝑖𝑗 )
This approximate calculation can be directly optimized using
gradient-descent methods.
Cross-modality average precision. For VT-ReID, the main challenge is the coexistence of inter-modality discrepancy and intramodality variations, and the former is usually larger than the latter.
So the original Smooth-AP will prioritize optimizing the ranking of
intra-modality instances while neglecting to optimize the ranking of
inter-modality ones due to its inherent attributes. To handle the issue,
we calculate Cross-modality Average Precision (CAP) to handle the
main challenge of the cross-modality VT-ReID, in which queries and

3. Proposed method
In this section, we provide a detailed description of our proposed
framework. As illustrated in Fig. 3, we adopt a Global–Local Structure
(GLS) to extract both Global and Local Features (GLF) from visible and
thermal modality images, respectively. Then we calculate the losses of
3


exec
/bin/zsh -lc "pdftotext -l 3 'CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 259 (2025) 104406

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

CSGN:CLIP-driven semantic guidance network for Clothes-Changing Person
Re-Identification
Yang Lu a ,∗, Bin Ge a , Chenxing Xia a , Junming Guan b
a College of Computer Science and Engineering, Anhui University of Science and Technology, Huainan, 232000, Anhui, China
b

School of Information Engineering, Huang Shan University, Huangshan, 245000, Anhui, China

ARTICLE

INFO

Communicated by William Robson Schwartz
Keywords:
Clothes-Changing Person Re-identification
CLIP
Prompt engineering
Metric learning
Heterogeneous semantic fusion

ABSTRACT
Clothes-Changing Person Re-identification (CCReID) aims to match identities across images of individuals in
different attires. Due to the significant appearance variations caused by clothing changes, distinguishing the
same identity becomes challenging, while the differences between distinct individuals are often subtle. To
address this, we reduce the impact of clothing information on identity judgment by introducing linguistic
modalities. Considering CLIP’s (Contrastive Language-Image Pre-training) ability to align high-level semantic
information with visual features, we propose a CLIP-driven Semantic Guidance Network (CSGN), which consists
of a Multi-Description Generator (MDG), a Visual Semantic Steering module (VSS), and a Heterogeneous
Semantic Fusion loss (HSF). Specifically, to mitigate the color sensitivity of CLIP’s text encoder, we design the
MDG to generate pseudo-text in both RGB and grayscale modalities, incorporating a combined loss function
for text-image mutuality. This helps reduce the encoder’s bias towards color. Additionally, to improve the
CLIP visual encoder’s ability to extract identity-independent features, we construct the VSS, which combines
ResNet and ViT feature extractors to enhance visual feature extraction. Finally, recognizing the complementary
nature of semantics in heterogeneous descriptions, we use HSF, which constrains visual features by focusing
not only on pseudo-text derived from RGB but also on pseudo-text derived from grayscale, thereby mitigating
the influence of clothing information. Experimental results show that our method outperforms existing
state-of-the-art approaches.

1. Introduction
With the acceleration of urbanization and population growth, Person Re-identification (ReID) technology has garnered widespread attention. As an essential automated pedestrian retrieval technique in video
surveillance systems, ReID aims to link multiple images of the same
person captured by different cameras or taken by the same camera
at different times. In large-scale video surveillance systems, person reidentification compensates for the visual limitations of fixed cameras
and holds significant application value in areas such as smart cities,
surveillance security, and criminal investigations. Despite notable advancements in traditional ReID methods, challenges still persist in
Clothes-Changing Person Re-identification (CCReID). Traditional ReID
methods mostly rely on factors such as clothing appearance, pose, and
body type to distinguish between different identities. However, as time
passes, clothing changes pose significant challenges to model accuracy.
In the case of clothing changes, existing methods (Shu et al., 2021;
Yu et al., 2020; Yang et al., 2019; Li et al., 2020, 2021b; Yang et al.,
2023b; Gu et al., 2022) can be broadly categorized into two approaches.

The first category (Shu et al., 2021; Yu et al., 2020; Yang et al., 2019)
uses auxiliary modalities such as gait, silhouette, and contour to mitigate the effects of garment transformations. The second category (Yang
et al., 2023b; Gu et al., 2022) focuses on adversarial learning and metric
functions to learn discriminative features that are resistant to clothing
interference. However, as shown in Fig. 1, these methods primarily
rely on visual modality information, which places the clothes-changing
person re-identification network in a visual bottleneck. Fortunately,
vision-language learning paradigms have garnered increasing attention
due to their ability to learn semantically rich visual representations.
Several studies (Li et al., 2023b; He et al., 2023; Lin et al., 2023)
have shown that combining visual content with corresponding language
descriptions using CLIP enables the model to perceive high-level semantic information related to the target pedestrian. However, these
studies have mainly focused on clothing-consistent scenarios, and the
application of CLIP to CCReID has not been fully explored. We observe
that CLIP’s language descriptions of character images tend to focus on
a person’s clothing, and language descriptions can selectively mask the

∗ Corresponding author.

E-mail address: 2022201255@aust.edu.cn (Y. Lu).
https://doi.org/10.1016/j.cviu.2025.104406
Received 7 October 2024; Received in revised form 30 April 2025; Accepted 27 May 2025
Available online 2 June 2025
1077-3142/© 2025 Published by Elsevier Inc.

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 1. Superiority of CLIP over traditional methods.

interference of color information. This motivates us to adapt CLIP to
the CCReID task, using semantic information as a bridge to guide visual
representations and thereby eliminate clothing effects.
Therefore, we propose a two-stage CLIP-based Semantic Guidance
Network (CSGN). Specifically, as shown in Fig. 2, in the first stage, we
design a Multi-Description Generator (MDG), which generates pseudotext in both RGB and grayscale modalities, combining mutual loss
between text and image to mitigate the text encoder’s sensitivity to
color. In the second stage, we design a Visual Semantic Steering module
(VSS), which enhances visual features by parallelizing ResNet and ViT
feature extractors. Given the complementarity of these two descriptions, we further propose a Heterogeneous Semantic Fusion (HSF) loss
to provide more comprehensive constraints on the individual’s features.
Our main contributions are summarized as follows:

(2021b) introduced a body representation model that captures body
shape information to mitigate the impact of garment variations. However, these shape estimation methods often suffer from inaccuracies
due to the inherent unreliability of inferring body contours from 2D
images, especially in the context of CC-ReID where clothing variation
is significant and can obscure true body shape cues.
In addition to shape-based features, gait information has also been
explored as auxiliary identity-related information (Fan et al., 2020; Jin
et al., 2022). Some researchers have reformulated the person ReID task
as a gait recognition problem. However, this direction remains in its
early stages due to the limited availability of reliable video datasets
and a relatively small body of results.
Nonetheless, due to the lack of explicit guidance from prior knowledge, existing methods often fall into a visual bottleneck. In this paper,
we propose the CSGN framework, which leverages a vision-language
model to generate pseudo-text descriptions. These descriptions provide
semantic guidance for the distribution of visual features, effectively
helping the model to overcome the visual bottleneck.

∙ We propose the CSGN network, which integrates the powerful
linguistic capabilities of CLIP into the CCReID task.
∙ We design an MDG module that reduces the text encoder’s sensitivity to color through bimodal prompt engineering and an
image-text mutual loss.
∙ We introduce a VSS module that enhances visual features by parallelizing ResNet and ViT extractors, along with an HSF loss that
guides CSGN to learn more comprehensive individual features.
∙ Extensive experiments on four widely used datasets demonstrate
the effectiveness and superiority of our method over state-of-theart approaches.

2.2. Visual language learning
Compared to supervised pre-training on ImageNet, visual-language
pre-training (VLP) significantly improves the performance of many
downstream tasks by training image-text alignments. With the advancement of deep learning, large-scale visual-language models have
emerged. For example, Radford et al. (2021) first trained CLIP on a
dataset of 400 million (image, text) pairs collected from the internet,
achieving impressive performance. Jia et al. (2021) utilized a noisy
dataset consisting of over 1 billion image-text pairs and employed a
simple dual-encoder architecture to learn visual and linguistic representations, achieving strong results on benchmark datasets like Flickr30K
and MSCOCO for image-text retrieval. Li et al. (2021a) focused on
large-scale image-text pairs collected from the internet, which are often
noisy, and addressed this by employing fine-grained alignment and
cross-modal interaction with momentum distillation, improving the
model’s capacity to learn from such noisy data. Wang et al. (2021)
proposed a minimalist pre-training framework that reduces training
complexity by leveraging large-scale weak supervision and uses a single
prefixed language modeling target for end-to-end training. However,
both pre-training and fine-tuning these large models demand substantial storage and computational resources. As a result, some research
has begun exploring how to apply these large models to downstream
tasks with smaller parameter sets. CoOp (Zhou et al., 2022c) introduces
a learnable prompt paradigm to effectively utilize the powerful prior
knowledge of a vision-language model (e.g., CLIP) for downstream
tasks while keeping the model parameters fixed. CoCoOp (Zhou et al.,
2022b) builds upon CoOp by adding a lightweight neural network
to generate a conditional sequence for each image, which is incorporated into the learnable vectors in the original CoOp framework.

2. Related work
2.1. Clothes-Changing Person Re-identification
Clothes-changing person re-identification (ReID), also known as
long-term person ReID, is a challenging task that involves more complex influencing factors than traditional short-term ReID. Given these
long-term challenges, several benchmark datasets have been proposed,
including PRCC (Yang et al., 2019), Celebrities-reID (Huang et al.,
2019), LTCC (Qian et al., 2020), and VC-Clothes (Wan et al., 2020).
Meanwhile, various methods have been developed to learn clothinginvariant representations. For instance, Shu et al. (2021) employed a
human parsing model to extract body parts as auxiliary information
and introduced a pixel sampling strategy to modify the garment regions
in pedestrian images. Yu et al. (2020) designed a garment detector
that utilizes garment templates as auxiliary information to refine appearance features. Yang et al. (2019) proposed a polar-coordinate
method to extract a fan-shaped receptive field from contour sketches,
enabling the model to learn body shape information and better handle
moderate clothing changes. Li et al. (2020) used human keypoints
to extract garment-independent and shape-related features. Li et al.
2

Y. Lu, B. Ge, C. Xia et al.

Computer Vision and Image Understanding 259 (2025) 104406

Fig. 2. Overview of the proposed method. Our CSGN includes two stages. Stage 1: Training MDG using RGB maps and gray-scale maps with Meta-Insert skill. Stage 2: Training
VSS to fuse local and global features, and reduce the feature bias by the HSF loss function.

In contrast to the static prompt in CoOp, the dynamic prompt in
CoCoOp is instance-adaptive, making it more robust to class migration.
Adapter (Houlsby et al., 2019) offers a different approach by preserving
the original pre-trained model parameters while introducing a small set
of trainable parameters. During fine-tuning, only these new parameters
are trained, thus avoiding excessive computational costs.
However, in a ReID scenario, all IDs inherently belong to the same
category, making it challenging to generate distinct and appropriate
text descriptions for each image. To address this challenge, we introduce a novel framework named CSGN, which consists of three core
components: MDG, VSS, and HSF. In the first stage, our MDG module
generates two learnable prompt tokens specific to each identity in
the training set. This process incorporates a mutual loss mechanism
between text and image representations, effectively mitigating the
text encoder’s sensitivity to color variations. In the second stage, VSS
enhances the visual feature representation by leveraging the complementary strengths of both ResNet and ViT feature extractors in parallel.
This parallel processing ensures a richer and more nuanced visual understanding. Finally, recognizing the synergy between these two types
of representations, the HSF loss is applied, providing a comprehensive
and constraining force that further strengthens the identity of each
individual. In summary, our CSGN framework offers a holistic approach
to generating discriminative textual and visual representations for ReID
scenarios.

sets {𝑇1 , 𝑇2 , … , 𝑇𝑁 } and {𝐼1 , 𝐼2 , … , 𝐼𝑁 } are one-to-one matched, where
when i = j, 𝑇𝑖 and 𝐼𝑗 form positive sample pairs, and otherwise,
𝑇𝑖 and 𝐼𝑗 are negative sample pairs. Next, we calculate the cosine
similarity between 𝑇𝑖 and 𝐼𝑗 ; the greater the cosine similarity, the
stronger the correspondence between 𝑇𝑖 and 𝐼𝑗 , and vice versa. Finally,
we maximize the cosine similarity for the 𝑁 positive samples and
minimize the cosine similarity for the N(N-1) negative samples by
training the parameters of the Text Encoder and Image Encoder. This
process ensures a one-to-one correspondence between images and texts.
We next briefly review CLIP-ReID (Li et al., 2023b), which consists
of an image encoder I(⋅) and a language encoder T(⋅), both pre-trained
on CLIP. For the image encoder I(⋅), ViT-B/16 or ResNet-50 are commonly used to extract feature vectors from images. For the language
encoder T(⋅), ViT-B/16 is implemented as a Transformer, which generates a representation from a sentence. Specifically, given a description
such as ‘‘A photo of a [class]’’, where [class] is typically replaced by
concrete text labels, T(⋅) first converts each word into a unique numeric
ID using lower-cased byte pair encoding (BPE) with a vocabulary size of
49,152. Each ID is then mapped to a 512-dimensional word embedding.
To enable parallel computation, each text sequence has a fixed length
of 77 tokens, including the start [SOS] and end [EOS] tokens. After
passing through a 12-layer model with 8 attention heads, the [EOS]
token is considered as the feature representation of the text, which
is then layer-normalized and linearly projected into the cross-modal
embedding space.
CLIP-ReID consists of two training phases. In the first phase, both
I(⋅) and T(⋅) are frozen, and the language prompt is optimized by 𝐿𝑖2𝑡
and 𝐿𝑡2𝑖 in Eq. (6). In the second phase, T(⋅) and the language prompt
are fixed, while I(⋅) is trained using 𝐿𝑖𝑑 , 𝐿𝑡𝑟𝑖 , and 𝐿𝑖2𝑡𝑐𝑒 in Eq. (7) to
achieve the best alignment between the visual and text features.

3. The proposed methods
A Preliminaries: Overview of CLIP and CLIP-ReID
We first briefly review CLIP (Radford et al., 2021). The CLIP model
structure consists of two parts: the Text Encoder and the Image Encoder. The Text Encoder is based on a Transformer model, while two
models are used for the Image Encoder: one is ResNet based on CNN,
and the other is ViT based on Transformer. Assuming that a batch in
the dataloader contains 𝑁 (text-image) pairs, we first encode these 𝑁
texts into one-dimensional vectors of length d using the Text Encoder.
This batch of text data is represented as {𝑇1 , 𝑇2 , … , 𝑇𝑁 }. Similarly,
the 𝑁 images are encoded by the Image Encoder as {𝐼1 , 𝐼2 , … , 𝐼𝑁 },
where each image vector has the same length as the text vector. The

𝐿𝑖2𝑡 = −

𝑁
exp(𝑠(𝒇𝑖 , 𝒇𝑖𝑡 ))
1 ∑
log( ∑𝑛
),
𝑏
𝑁 𝑖=1
exp(𝑠(𝒇𝑖 , 𝒇𝑗𝑡 ))
𝑗=1

𝐿𝑡2𝑖 = −

𝑁
exp(𝑠(𝒇𝑝𝑖 , 𝒇𝑦𝑡 ))
∑
1 ∑ 1
𝑖
log( ∑𝑛
),
𝑏
𝑁 𝑖=1 |𝑝(𝑦𝑖 )| 𝑝 ∈𝑝(𝑦 )
exp(𝑠(𝒇
,
𝒇𝑦𝑡 ))
𝑗
𝑗=1
𝑖
𝑖

(1)

(2)

𝑖

𝑛𝑏

𝐿𝑖𝑑 = −
3

1 ∑
𝑞 𝑙𝑜𝑔(𝑾 (𝒇𝑖 )),
𝑛𝑏 𝑖=1 𝑖

(3)


codex
第一批里已经能看到一个共同套路：把已有损失或 CLIP 对齐改成任务特定的排序、语义、视角一致性目标，然后用“原始方法会错优化什么”作为卖点。现在读第二批，重点看它们怎样把中性特征、文本嵌入和样本可靠性包装成问题。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
DATE: Dual Asymmetric Textual Embedding
guided Person Re-Identification
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209501

Pengqi Yin1,2,3 , Hantao Yao4,∗ , Changsheng Xu1,2
1

State Key Laboratory of Multimodal Artificial Intelligence Systems
2
Institute of Automation, Chinese Academy of Sciences
3
University of Chinese Academy of Sciences School of Artificial Intelligence
4
School of Information Science and Technology, University of Science and Technology of China
yinpengqi2022@ia.ac.cn, yaohantao@ustc.edu.cn, csxu@nlpr.ia.ac.cn

Abstract—Inspired by the development of the Visual-Language
Models(VLM), the textual embedding generated from the learnable prompt or the textual-level description is explored to boost
the visual embedding by synchronizing the fusion and alignment
strategy for person re-identification(ReID). However, the synchronization strategy treats the learnable-based and description-based
textual embedding equally, leading to the generated visual representation easily affected by the noise contained in each textual
embedding, especially for the description-based textual embedding. To address the above shortcoming, we propose a novel
Dual Asymmetric Textual Embedding(DATE) that uses learnablebased and description-based textual embedding to asymmetricly
guide person representation learning. Since the description-based
textual embeddings are controlled mainly by the automatically
extracted textual-level descriptions, they contain a lot of noise
and have less discriminative ability. Therefore, DATE treats
the description-based textual embeddings as auxiliary clues to
boost visual and textual representation learning. Moreover, the
Textual-to-Visual Adapter and Textual-to-Textual Adapter are
proposed to inject the description-based textual embedding into
the learnable-based textual embedding and visual embedding.
To reduce the effect of noise in textual description, the identityaware description-based textual embeddings are generated by
averaging the description-based textual embeddings belonging
to the same identity, which is used to boost the discriminative
to infer the learnable-based textual space used for aligning the
visual representation learning. Extensive evaluations on Market1501, DukeMTMC, and MSMT-17 validate the effectiveness of
the proposed method.
Index Terms—Person Re-identification, Multi-Modal Recognition, Multi-Modal Fusion

I. I NTRODUCTION
Person Re-Identification(ReID) aims to retrieve query images from large-scale gallery images. Previous methods generate the visual embedding based on the metric-based learning [1], part-based methods [2], [3], and self-attention learning [4], [5]. However, the above-mentioned methods only consider the visual clues inferred from the given images, lacking
clues from other modalities. Inspired by the development of
the Visual-Language Models(VLM), the textual embedding
generated from the VLM has been explored to boost the person
representation learning in person ReID [6], denoted as multimodal ReID.
*Corresponding author

Among all multi-modal ReID methods, CLIP-ReID [7] is
the first method using identity-level textual space to align the
visual embedding. Unlike the traditional metric term used in
existing visual-based ReID methods, CLIP-ReID applies an
additional contrastive loss between the visual embeddings and
the textual-level identity-aware embeddings inferred from the
identity-aware prompts, denoted as the learnable-based textual
embedding. However, learnable-based textual embedding does
not capture the text description containing human priors,
leading to a limited ability to describe the specific information
of each person’s image. Recently, multiple types of textual
embeddings have been employed to enhance the diversity of
textual space [3], [8], [9], e.g., MP-ReID [3] firstly applies
MLLM to extract the image-level description, which is further
fused with the learnable textual tokens and visual tokens by a
synchronization strategy. We define the embedding generated
by the learnable tokens and the generated description as
the learnable-based textual embedding and description-based
textual embedding, respectively. However, the synchronization strategy treats the learnable-based and description-based
textual embedding equally, leading to the generated visual
representation easily affected by the noise contained in each
textual embedding, especially for the description-based textual
embedding. Consequently, it is critical for multi-modal ReID
methods to propose a reasonable fusion strategy for jointly
considering the benefits of learnable-based and descriptionbased textual embedding.
To address the above shortcomings, we introduce an asymmetrical strategy that fuses learnable-based and descriptionbased textual embedding with visual embedding. Although
description-based textual embedding can provide essential
clues containing the human prior, low-quality text description
would lead to the description-based textual embedding being
less discriminative. Note that low-quality images and meaningless descriptions like noise can easily generate low-quality text
descriptions. Therefore, it is a reasonable motivation to treat
the description-based textual embeddings as auxiliary clues
to boost visual and textual representation learning rather than
decisive information. Consequently, we inject the descriptionbased textual embedding into the learnable-based and visual
embedding with the fuse module. To reduce the effect of noise

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.

in textual description, the identity-aware description-based
textual embedding is generated by averaging the descriptionbased textual embeddings belonging to the same identity,
which is used to boost the discriminative to infer the learnablebased textual space used for aligning the visual representation
learning.
In this work, we propose a novel Dual Asymmetric Textual
Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning, as shown in Figure 1. We utilize MLLM
to generate consistent-granularity descriptions for each image
under specific prompts, which are then processed through a
Text Encoder to obtain description-based textual embeddings.
Meanwhile, similar to CLIP-ReID, we generate visual embedding and learnable-based textual embedding. After that,
Textual-to-Visual Adapter and Textual-to-Textual Adapter are
proposed to fuse the visual embeddings and learnable-based
textual embeddings with the description-based textual embeddings for injecting the human-level knowledge inferred from
the textual description. We then implement a cross-granularity
training approach, where identity-level learnable tokens are
supervised using image-level description-based textual embeddings and visual embeddings, and image-level visual embeddings are supervised using identity-level description-based
textual embeddings. This method asymmetrically leverages the
description-based textual space and the learnable-based textual
space to facilitate the visual space in learning more robust
visual representations.
Our main contributions are summarized as follows:
1) An asymmetrical fusion strategy is introduced to jointly
consider the benefits of different types of textual embeddings, such as learnable-based and description-based
textual embeddings.
2) We propose a novel Dual Asymmetric Textual Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person
representation learning.
3) Evaluation on several benchmarks verify the effectiveness of the proposed method, e.g., obtaining 90.7%,
83.6%, and 74.7% for Market-1501, DukeMTMC, and
MSMT17, respectively.
II. R ELATED W ORKS
A. Prompt Learning
Inspired by the NLP realm where researchers use prompt
learning methods to avoid the high costs of directly training
models [10]. The rise of ChatGPT and GPT-4 offers a idea
of using instruction-tuning methodology to transfer its ability
to downstream tasks, e.g., InstructGPT [11], FLAN-T5 [12],
which shows the ability to use prompt learning method on
zero-shot and few-shot tasks. Some works transferred this idea
to the vision realm. BLIP-2 [13], Flamingo [14] learns from
image-text pairs and has shown promising ability. CoOp [15]
uses a set of global prompts to help the model fit in downstream tasks. CLIP-ReID [7] trains an identity-level prompt

to cooperate with the training of the visual encoder. Our
method uses learnable-based textual embeddings to boost the
performance of visual space.
B. Person Re-identification
Person Re-Identification (ReID) is a pivotal task in computer
vision. The attention mechanism shows its feature extraction
ability, enhancing its performance on ReID tasks [4]. Vision
Transformer (ViT) application for image feature extraction
introduces innovative approaches to the ReID task [16]. The
trivial idea is to leverage ViT’s robust feature aggregation capability to consolidate hierarchical and regional features [17].
The development of Vision-Language models [18] have also
provided solutions for ReID tasks. CLIP-ReID [7] uses a
prompt learning strategy [15] to generate a set of discriminative text embeddings and then use it to enhance the visual space by minimizing InfoNCE loss [19] between visual
embeddings and textual embeddings. Instruct-ReID [9] uses
different prompts to help the model shift to different tasks.
MLLMReID [8] uses an MLLM to directly produce visual
embeddings, effectively applying the powerful generalization
ability of MLLM to the ReID tasks.
Previous cross-modal ReID methods have predominantly
employed synchronous fusion of multiple modalities, overlooking the noise within the description-based textual embeddings. In our approach, we achieve a more effective model by
treating the description-based textual embeddings as auxiliary
clues to boost visual and textual representation learning.
III. M ETHODS
A. Overall Framework
This work proposes a novel Dual Asymmetric Textual
Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning. As shown in Figure 1, the proposed
DATE consists of three types of embedding spaces: visual
space, learnable-based textual space, and description-based
textual space. The description-based textual space first applies
MLLM to extract the textual description for each image and
then uses the frozen Text Encoder to generate the imagelevel description-based embedding. After that, a Textual-toVisual Adapter At2v is proposed to fuse the description-based
embedding and the image’s visual embedding extracted by the
frozen Visual Encoder. Moreover, a Textual-to-Textual Adapter
At2t injects the description-based textual embedding into the
learnable-based textual embedding generated by feeding the
identity-aware prompt into the frozen Text Encoder. Through a
cross-granularity training method we can boost the robustness
within the visual space.
B. Dual Asymmetric Textual Embedding
Previous methods applies adversarial learning between the
visual and textual embeddings inferred from the identity-aware
prompts, denoted as the learnable-based textual embedding.
However, learnable-based textual embedding does not capture
the text description containing human priors, leading to a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.

“The person in
the image is
wearing a white
T-shirt and black
pants”

Matmul

Matmul

Softmax

Enhanced
Visual
Embeddings ෡
𝑭

ℒ𝑐𝑙𝑠

ℒ𝑐𝑜𝑛

…

Learnable-based
Textual
Embeddings 𝑻𝒍

Retrieval

GeLU

Softmax

Matmul

Matmul

…

Text
Encoder 𝓔

ℒ𝑑

ℒ𝑐𝑒

…
…

…

Learnable
-based
Textual
Learnable
Space
Prompts 𝔼

𝕋

…

Textual-to-Visual Adapter

ID-level
Descriptionbased
Textual
Embeddings
𝕋෢𝒅

…
…

…

…
…

Visual
Space

…

Visual
Encoder 𝓥

Visual Embeddings
𝑭

GeLU

𝐌𝐋𝐋𝐌
𝒢

Image-level
Descriptionbased
Textual
Embeddings
𝒅

…

“Please describe the
appearance of the
person in the picture.
Do not add any other
sentences.”

Text
Encoder 𝓔

Semantic
Descriptions
𝑻𝒍

MLLM Prompts

…

Description
-based
Textual
Space

Enhanced
Learnable-based
Textual
Embeddings 𝑻෡𝒍

Textual-to-Textual Adapter

Fig. 1. The framework of DATE. DATE employs a description-based textual space as complementary to the visual and learnable-based textual space. We
use two cross-attention adapters to fuse description-based textual descriptions with visual and learnable-based textual embedding, respectively. We conduct a
cross-granularity training method, guiding the construction of visual space with learnable-based and description-based textual embeddings.

limited ability to describe the specific information of each
image. Moreover, the text description generated by MLLM
can be used to construct a complementary textual space to
enhance its diversity. Therefore, a novel Dual Asymmetric
Textual Embedding(DATE) is proposed to use learnable-based
and description-based textual embedding to asymmetrically
guide person representation learning.
Given the training dataset D = {xi , yi }N
i=1 sampled from
M identities, for the learnable-based textual space, CLIP-ReID
defines a set of learnable prompts E = {E 1 , E 2 , · · · , E M },
where E i denotes the identity-aware prompt of i-th identity. Specifically, the learnable prompt E i is designed as
“A photo of a [e1i ][e2i ] · · · [eni ] person.”, where eni stands for
learnable tokens and n is the length of prompt. After that,
the learnable-based textual embeddings of all M identities
T l = {tl1 , ..., tlM } can be generated by feeding all learnable
prompts E into the Text Encoder E:
T l = E(E),

(1)

where E are learnable tokens needed for optimization.
Meanwhile, we apply the frozen Visual Encoder V to extract
the visual embedding f i for the image xi ,
f i = V(xi ),

(2)

where V denotes for the visual encoder.
The proposed Dual Asymmetric Textual Embedding D aims
to take the visual embedding f i , the image xi , and MLLM

G as input, for generating the enhanced visual embeddings
F̂ = {f̂ 1 , ..., f̂ N }, which can formulated as Eq. (3),
f̂ i = D(xi , f i , G).

(3)

Formally, given the image xi , LLaVA(G) is employed to
generate the image-level description Tdi based on the prompt
such as “{Please describe the appearance of the person in the
picture,...}”. After that, the Text Encoder E is applied to generate the corresponding description-based textual embedding
tdi = E(Tdi ). Once obtaining the pair of visual embedding f i
and the description-based textual embedding tdi , a Textual-toVisual Adapter At2v is proposed to fuse those two types of
embeddings,
f̂ i = At2v (f i , tdi ).
(4)
To better fuse the visual and textual embedding, a multihead attention module and two fully connected layers implement the Textual-to-Visual Adapter At2v . Since the visual
embedding f i contains enough discriminative information,
it is treated as the value and key terms in the multi-head
attention module, and the description-based textual embedding
is served as the query term for providing the complementary
clues to generated the fused visual embedding. After that, two
additional fully connected layers are employed to project the
fused visual embedding. Finally, the Textual-to-Visual Adapter
At2v can be formulated as Eq. (5),
d

At2v (f i , ti ) = F2 (GeLU(F1 (softmax

⊤
td
i fi
√
d

!
f i + f i ))) + f i ,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.

(5)


exec
/bin/zsh -lc "pdftotext -l 3 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

5469

CycleTrans: Learning Neutral Yet Discriminative
Features via Cycle Construction for VisibleInfrared Person Re-Identification
Qiong Wu, Jiaer Xia , Pingyang Dai , Yiyi Zhou , Member, IEEE,
Yongjian Wu, and Rongrong Ji , Senior Member, IEEE

Abstract— Visible-infrared person re-identification (VI-ReID)
is the task of matching the same individuals across the visible and infrared modalities. Its main challenge lies in the
modality gap caused by the cameras operating on different
spectra. Existing VI-ReID methods mainly focus on learning
general features across modalities, often at the expense of
feature discriminability. To address this issue, we present a novel
cycle-construction-based network for neutral yet discriminative
feature learning, termed CycleTrans. Specifically, CycleTrans
uses a lightweight knowledge capturing module (KCM) to capture rich semantics from the modality-relevant feature maps
according to pseudo anchors. Afterward, a discrepancy modeling
module (DMM) is deployed to transform these features into
neutral ones according to the modality-irrelevant prototypes.
To ensure feature discriminability, another two KCMs are further
deployed for feature cycle constructions. With cycle construction,
our method can learn effective neutral features for visible
and infrared images while preserving their salient semantics.
Extensive experiments on SYSU-MM01 and RegDB datasets
validate the merits of CycleTrans against a flurry of state-ofthe-art (SOTA) methods, +1.88% on rank-1 in SYSU-MM01
and +1.1% on rank-1 in RegDB. Our code is available at
https://github.com/DoubtedSteam/CycleTrans.
Index Terms— Cross-modality retrieval, deep learning, person
re-identification.
Manuscript received 12 April 2023; revised 20 December 2023; accepted
20 March 2024. Date of publication 9 April 2024; date of current version
1 March 2025. This work was supported in part by the National Key Research
and Development Program of China under Grant 2022ZD0118202; in part by
the National Science Fund for Distinguished Young Scholars under Grant
62025603; in part by the National Natural Science Foundation of China
under Grant U21B2037, Grant U22B2051, Grant 62176222, Grant 62176223,
Grant 62176226, Grant 62072386, Grant 62072387, Grant 62072389, Grant
62002305, and Grant 62272401; and in part by the Natural Science Foundation
of Fujian Province of China under Grant 2021J01002 and Grant 2022J06001.
(Corresponding author: Pingyang Dai.)
Qiong Wu and Yiyi Zhou are with the Key Laboratory of Multimedia
Trusted Perception and Efficient Computing, Ministry of Education of China,
Xiamen 361005, China, and also with the Institute of Artificial Intelligence,
Xiamen University, Xiamen 361005, China (e-mail: qiong@stu.xmu.edu.cn;
zhouyiyi@xmu.edu.cn).
Jiaer Xia and Pingyang Dai are with the Key Laboratory of Multimedia
Trusted Perception and Efficient Computing, Ministry of Education of China,
Xiamen University, Xiamen 361005, China (e-mail: xiajiaer@stu.xmu.edu.cn;
pydai@xmu.edu.cn).
Yongjian Wu is with the Youtu Laboratory, Tencent Company Ltd., Shanghai
200233, China (e-mail: littlekenwu@tencent.com).
Rongrong Ji is with the Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, and Fujian
Engineering Research Center of Trusted Artificial Intelligence Analysis and
Application, Institute of Artificial Intelligence, Xiamen University, Xiamen
361005, China, and also with the Peng Cheng Laboratory, Shenzhen 518066,
China (e-mail: rrji@xmu.edu.cn).
Digital Object Identifier 10.1109/TNNLS.2024.3382937

I. I NTRODUCTION

V

ISIBLE-INFRARED person re-identification (VI-ReID)
[1] aims at matching visible and infrared images of
pedestrians with the same identity, which are captured by the
cameras operating on different spectra. As more and more
infrared cameras are deployed in real-world scenarios, the
research of VI-ReID has attracted increasing attention from
both academia and industry [1], [2], [3], [4], [5], [6], [7],
[8]. In addition to the intrinsic challenges of traditional Re-ID
tasks, such as the variations of viewpoints and body poses,
VI-ReID also suffers from the obvious appearance difference
between pedestrian images of different modalities [9], [10],
[11], [12]. Meanwhile, besides the blur of image [13] and
occlusion of human body [14], feature extraction is also
hindered by the characteristics of cameras, e.g., the appearance
of the same person in different modalities only has limited
shared information.
This issue is also coined as modality gap [1], [15], [16],
[17], as illustrated in Fig. 1(a). Specifically, under different
types of cameras, the pedestrian will exhibit notable differences in visual characteristics, e.g., the color and texture of
clothes. And this gap will be further reflected in the features
extracted by deep neural networks, as shown in Fig. 1(b).
In this case, the traditional Re-ID methods [18], [19], [20],
[21], which identify pedestrians mainly based on the appearance, often fail to accomplish this task.
In recent years, a bunch of methods have been proposed
for VI-ReID and achieved remarkable progress [2], [22], [23],
[24], [25], [26], [27], [28], [29], [30], [31]. The prevalent
solution [22], [23], [24], [25], [32] to modality gap is aligning
the feature or pixel distributions of two modalities, which,
however, usually comes at the expense of feature discriminability. To explain, the feature alignment needs to cluster
the samples of the same modality in the joint semantic space.
This optimization process also reduces the semantic distances
between the samples of different identities, as shown in Fig. 1.
Meanwhile, the salient semantics of pedestrian images tend to
be lost during alignment, e.g., the details of cloths, which also
greatly reduce the descriptive power of learned features. In this
case, how to make a trade-off between the generality and
discriminability of multi-modal features is the key to VI-ReID.
To address this issue, we propose a novel cycleconstruction-based network (CycleTrans) for VI-ReID.

2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.

5470

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

KCM and DMM, which can help the model learn
discriminative yet neural features.
3) The proposed CycleTrans achieves new SOTA performance on multiple benchmark datasets, e.g., 76.58%
on Rank-1 in SYSU-MM01 under all-search single-shot
setting. And the experimental results also well validate
its effectiveness toward the modality gap.
II. R ELATED W ORK

Fig. 1. Illustrations of the examples of VI-ReID and the feature spaces of
different Re-ID methods. (a) In VI-ReID, pedestrians with the same identity
exhibit notable appearance differences between visible and infrared images,
which is often termed modality gap. (b) Traditional methods (left) often fail
to match pedestrians across modalities, and only cross-modality alignment
(middle) often narrows down the decision boundaries between samples of
different identities. So, the idea semantic space for VI-ReID (right) should be
neutral yet discriminative.

The main principle of CycleTrans is to enhance the descriptive
power of transformed neutral features via semantical cycle
reconstructions. As shown in Fig. 2, the proposed CycleTrans
consists of three knowledge capturing modules (KCMs)
sharing the same parameters, and a discrepancy modeling
module (DMM). Specifically, the first KCM extracts
discriminative semantics from convolution feature maps
according to modality-specific anchors. Afterward, DMM
is applied to transform these features into neutral ones
for visible and infrared images, which is achieved by
using modality-irrelevant prototypes as the transfer targets.
To ensure discriminability, another two KCMs are further
applied to reconstruct the modality-relevant features learned
before. Based on this, two cycle constructions are built. The
cycle construction ends with the original modality, which
can benefit the discriminability. When cycle construction
ends with another modality, it helps alleviate the modality
gap. Through these cycle construction process, the proposed
method can well model general features across modalities
while preserving their salient semantics for fine-grained
pedestrian identification.
To validate the proposed CycleTrans, we conduct extensive
experiments on two benchmarks, namely SYSU-MM01 [1]
and RegDB [17]. The experimental results not only show its
obvious performance gains over the state-of-the-art (SOTA)
methods, e.g., +1.88% Rank-1 on SYSU-MM01 and +1.1%
Rank-1 on RegDB than DEEN [33], but also greatly confirm
its effectiveness in bridging the modality gap.
Overall, our main contributions are threefold.
1) We propose a novel cycle-construction-based network
for VI-ReID, termed CycleTrans. CycleTrans applies
shared prototypes as transferring targets to mitigate
the modality gap, and adopts the cycle construction to
enhance feature discriminability.
2) To alleviate the modality gap while preserving salient
semantics, two novel modules are proposed, namely

VI-ReID is an essential task that aims to match individuals
across the visible and infrared modalities, effectively compensating for the deficiencies of visible cameras in low-light
conditions. This task introduces unique challenges beyond
those found in traditional ReID, such as varying viewpoints,
illumination, and body poses, while also contending with
the modality gap–the marked appearance differences when
captured by different camera types [34], [35], [36], [37].
To address these initial challenges, Wu et al. [1] introduced
the foundational SYSU-MM01 dataset and proposed a deep
zero-padding network specifically designed for cross-modality
matching. Building on this foundation, two-stream models
were explored to process each modality independently, aiming
to minimize variations at both the feature and prediction
levels [38], [39], [40]. These methods set the stage for more
advanced strategies like MSO [41] and CoAL [42], which
further honed the capture of intra-modality information and
enhanced feature discriminability. The integration of GANs
marked a significant evolution in the field, with CmGAN [2]
being the first to employ these networks for VI-ReID. Subsequent innovations followed, including AlignGAN [23] and
JSIA [24], which leveraged GANs to generate images for
the missing modality and align cross-modal distributions at
multiple levels. In parallel, D2 RL [22] proposed a novel
four-dimensional image space that encompasses both RGB and
infrared data. As the field progressed, researchers introduced
the concept of an intermediate modality with works like
X-modality [25], cm-SSFT [5], SFANet [43], and MSA [44],
which served as a bridge between the visible and infrared
spectra. Besides, PartMix [45] generates the middle modality
through a novel data augmentation way. However, a drawback
emerged in that cm-SSFT required additional modality information even during the testing phase. In an effort to circumvent
this issue, FBP-AL [46] and FMCNet [47] concentrated on
extracting features that transcend modalities, with FMCNet
utilizing a memory bank approach and MAUM [48] focusing
on information aggregation from alternate-modality memory
banks. Further refining this approach, MSCLNet [49] sought
to combine representations from both modalities to increase
discriminability and suppress noise. Meanwhile, MPANet [50]
delved into the subtleties of inter-modality differences without
supplementary supervision. Most recently, DEEN [51] and
SMCL [33] have innovated by generating diverse embeddings
and applying modality mixup constraints, respectively, to mitigate the modality gap while preserving discriminability. In the
realm of part-based approaches, SCS+ [52] and MHSA-Net
[53] brought new insights by focusing on the comparison
of identical body parts, with the former using a clustering

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.

WU et al.: CycleTrans: LEARNING NEUTRAL YET DISCRIMINATIVE FEATURES VIA CYCLE CONSTRUCTION

5471

Fig. 2. Overview of the proposed CycleTrans. Given an image of arbitrary modality, CycleTrans first uses the proposed KCM to gather salient yet task-related
semantics from convolution feature maps based on the modality-relevant pseudo anchors. Afterward, the DMM is deployed to transform these features into
neutral ones via modeling the discrepancy to modality-irrelevant prototypes. To ensure feature discriminability, a cycle construction stage is implemented
(bottom), where another two KCMs are used to transform neutral features into the original modality-relevant representations.

algorithm for part detection and the latter ensuring feature
consistency within the same head. Distinguishing itself from
the former methods, the proposed CycleTrans method sets
a new precedent by transforming features of both modalities onto a shared distribution, guided by modality-irrelevant
prototypes. It maintains discriminability through innovative
semantic-cycle constructions, offering a novel perspective on
the persistent challenge of the modality gap in VI-ReID.
III. P RELIMINARY
N
Let D = {(xi , yi , mi )}i=1
denotes the VI-ReID dataset
which has N samples in total. For each example, denoted as
(xi , yi , mi ), the image xi has a corresponding identity label
Np
yi ∈ Y = {y j } j=1
and a modality label mi ∈ M = {v, r },
where N p is the number of identities, and v and r denote the
visible and infrared modalities, respectively.
Given a pedestrian image, VI-ReID aims to match the same
person in the other modality by ranking the similarity to
instances in the gallery set,1 and its objective can be defined
as
X


argmin
I d2 xi , x j > d2 (xi , xr ) ,
2

i, j,k

where, yi = y j , yi ̸ = yr , mi ̸ = m j , mi ̸ = mr .

(1)

Here, I (·) is an indicator function that returns 1 if the condition
is satisfied and 0 otherwise. d2 (·, ·) measures distance between
two features extracted by the model with parameters 2.

Specifically, for a visible or infrared image xi , we first apply
a convolutional backbone to extract its feature map, denoted
as F I ∈ Rh×w×d , where h × w denotes the resolution and d is
dimensionality. Afterward, we use the proposed KCM to mine
rich semantics from F I
F′I = KCM(F I , C)

(2)

where C ∈ Rk×d denotes the trainable pseudo anchors of
the corresponding modality. After the process of KCM, the
obtained features F′I ∈ Rk×d contain descriptive semantics for
Re-ID, but it is still modality-relevant.
To this end, we further transform F′I into neutral features
via a novel DMM

F N = DMM P, F′I
(3)
where P ∈ Rn×d are modality-irrelevant prototypes. Neutral
features F N ∈ Rk×d are further flattened to a representation
vector and then used for cross-modal retrieval.
To ensure the discriminability of the transformed F N , we use
it to reconstruct the modality-relevant features F′I via another
two KCMs. To keep the model compact, three KCMs share
the same parameters.
Overall, through this cycle-construction paradigm, the proposed CycleTrans can well capture salient semantics from each
modality, while learning effective neutral representations for
cross-modal retrieval.
B. Knowledge Capturing Module

IV. M ETHOD
A. Overview
The overall structure of the proposed cycle-constructionbased network (CycleTrans) is depicted in Fig. 2. Its main
principle is to maintain the descriptive power of the transformed neutral features via feature cycle constructions.
1 In testing, gallery set contain a series of pedestrian images whose identity
is known.

KCM is a novel and lightweight module for learning
discriminative and task-related semantics from convolutional
feature maps.
Concretely, given the feature map of an arbitrary modality
F I ∈ Rh×w×d , we first reshape it to a 2-d tensor F̂ I ∈ Rhw×d .
Then, we apply a dot-product attention to refine the features
by aggregating semantics from similar regions


T 
F̃ I = Softmax norm F̂ I norm F̂ I
F̂ I
(4)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

1003

Dependability Feature Learning Based on Sample
Generation for Unsupervised Text-to-Image
Person Re-Identification
Chenglong Shao, Tongzhen Si , Xiaohui Yang , Member, IEEE, and Hui Yuan , Senior Member, IEEE

Abstract—Text-to-image person re-identification (TIReID)
aims to retrieve the target pedestrians according to specific
textual descriptions. Benefiting from abundant annotated training
data, current supervised TIReID methods have achieved impressive performance. However, annotating cross-modality data is
extremely time-consuming, which limits their application in
real-world scenarios. Several methods attempt to generate text
descriptions or pseudo-labels but neglect the dependability of
image-text matching relationships or identity information. To
this end, we propose a Dependability Feature Learning based
on Sample Generation (DFLSG) for unsupervised TIReID. First,
we introduce a dependable text generation method that leverages
multimodal large language models to generate diverse texts
and further filtrate dependable texts for establishing imagetext matching relationships. Second, we design an Error Sample
Filtering Module (ESFM) to eliminate abnormal samples and
obtain reliable identity labels. Furthermore, we develop a Multilevel Triplet Joint Learning (MTJL) process, which continuously
optimizes the cross-modality dependable feature from center and
instance views. Extensive experiments are implemented to assess
the proposed DFLSG on four mainstream TIReID databases.
Experimental results demonstrate that DFLSG achieves state-ofthe-art performance compared with other unsupervised methods.
Code will be available at: https://github.com/CLS-2001/DFLSG
Index Terms—Text-to-image person re-identification, unsupervised learning, deep learning.

I. I NTRODUCTION
EXT-TO-IMAGE person re-identification (TIReID) aims
to retrieve the target pedestrians from a large-scale image
gallery based on specific textual description queries [1], [2],
[3]. Different from single-modality person re-identification
methods [4], [5], [6], TIReID utilizes diverse text descriptions
as query information to match target images, which is considered a cross-modality task. In recent years, due to its potential
application value in smart cities and intelligent transportation

T

Received 16 May 2025; revised 4 August 2025; accepted 18 August 2025.
Date of publication 20 August 2025; date of current version 22 January
2026. This work was supported in part by the National Natural Science
Foundation of China under Grant 62222110 and Grant 61603151; in part by
Taishan Scholar Project of Shandong Province under Grant tsqn202103001;
in part by Shandong Provincial Natural Science Foundation under Grant
ZR2023LZH013, Grant ZR2024QF185, Grant ZR2022MF263, and Grant
ZR2023LZH006; and in part by the New Introduced Talents Program of
University of Jinan under Grant 1009569. This article was recommended by
Associate Editor G. Xu. (Corresponding authors: Tongzhen Si; Xiaohui Yang.)
Chenglong Shao, Tongzhen Si, and Xiaohui Yang are with Shandong Key
Laboratory of Ubiquitous Intelligent Computing, University of Jinan, Jinan
250022, China (e-mail: ise sitz@ujn.edu.cn; ise xhyang@ujn.edu.cn).
Hui Yuan is with the School of Control Science and Engineering, Shandong
University, Jinan 250061, China (e-mail: huiyuan@sdu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3600759

Fig. 1. (a) Supervised TIReID setting with matched image-text pairs and
identity labels. (b) Unsupervised TIReID settings without image-text match
relationship and identity labels. (c) Description of the proposed dependable
text generation process for building image-text match relationship.

systems, TIReID has received widespread attention. However,
the inherent modality differences cause the cross-modality
matching process to be difficult.
Several works leverage pedestrian identity information to
explore invariant features at different granularities for bridging
the modality gap [7], [8]. In addition, some methods generate
additional image-text pairs according to identity information
for enhancing the diversity of training samples [9], [10].
These methods require image-text pairs to possess explicitly
matching relationship and identity information, as shown in
Fig. 1 (a).
In practical scenarios, image samples are easily obtainable
and abundant, but text descriptions are typically manually
annotated. In addition, annotating identity labels for different
pedestrian samples consumes much time and is not feasible.
Unsupervised TIReID aims to explore internal pedestrian
information from unlabeled samples, which has broad application prospects. As for unlabeled image and text samples, they
not only lack the label information but also miss the matching
relationship as shown in Fig. 1 (b). This significantly increases

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.

1004

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026

the complexity of the task. Existing image-based unsupervised
learning methods usually perform the clustering operation to
assign pseudo-labels for different pedestrian images [11], [12].
However, these methods could only process image samples
and fail to construct image-text matching relationships. Moreover, they neglect error category samples during the clustering
process, which undermines the reliability of identity labels.
Therefore, they are not suitable for unsupervised TIReID task.
Multimodal Large Language Model (MLLM) possesses
powerful cross-modality understanding and generation capabilities [13], [14]. Some researchers utilize a kind of MLLM to
generate text descriptions with pedestrian attributes according
to given different prompts [15]. Nevertheless, the reliability of
these generated text descriptions remains doubtful. In addition,
some methods cluster sample features to generate identity
information and employ cross-modality matching losses to
enhance semantic consistency [16], [17]. When using all
pseudo-labels, many generated unreliable label information
hinders model optimization. Although these methods do not
employ the labeled text descriptions, they utilize the label
information of images or the image-text match relationship, which belongs to the weak-supervised learning task.
In contrast, we focus on addressing two important problems
for unsupervised TIReID. Firstly, how to obtain better text
descriptions to construct reliable image-text matching relationships. Secondly, how to utilize clustering information to
generate reliable identity labels.
To this end, we propose a Dependability Feature Learning
based on Sample Generation (DFLSG) method for unsupervised TIReID. Dependability is solved from two key aspects.
On the one hand, we introduce a dependable text generation
method to automatically obtain accurate text descriptions.
Considering that MLLM has a strong generation ability and
text samples are uncertain and diverse, we design different
prompts to guide two MLLMs to generate diverse text descriptions. Then, the dependable text is selected as the training data
based on the cross-modality correlation guided by the prior
knowledge from the TIReID model. In this way, we build
the image-text matching relationship as shown in Fig. 1 (c).
On the other hand, since incorrect pseudo-labels could affect
the training process, we propose an Error Sample Filtering
Module (ESFM) to eliminate abnormal samples. We perform
the clustering operation to assign pseudo-labels for image-text
pairs and compute the image class center features. Generally,
the similarity between correct category text descriptions and
the image class centers is higher than that of incorrect category.
Hence, we utilize the Interquartile Range (IQR) filtering algorithm to identify and filter out abnormal samples. In this way,
we refine the clustering results and obtain more dependable
image-text pairs with label information.
Moreover, images and texts belong to two different modalities and follow one-to-one matching relationships. Image-text
pairs with the same identity possess a large modality difference, which increases the intra-class variations. Hence, we
design the Multilevel Triplet Joint Learning (MTJL) process
to explore cross-modality dependable features from the center
and instance views. Firstly, we present a center-level matching
loss that minimizes the distance between class center feature

and instance features within the same category. Secondly, we
introduce an instance-level matching loss that continuously
optimizes the intra-modality and inter-modality feature distribution to reduce intra-class variations and increase inter-class
distances. The MTJL method effectively aggregates the two
levels of matching loss to optimize the feature distribution
across different modalities and facilitate the model to learn
dependability pedestrian features.
In this study, four major contributions are summarized as
follows.
(1) We propose a dependable text generation process that
leverages the complementary strengths of MLLMs to
generate and filter accurate descriptions for constructing
image-text matching relationships.
(2) We design an ESFM to integrate statistical method
and cross-modality correlation for eliminating abnormal
image-text pairs, which effectively enhances the reliability of identity labels.
(3) We construct an MTJL process to continuously learn
cross-modality dependability features from the center
and instance views.
(4) We conduct numerous experiments on three public
benchmark datasets, and experiment results demonstrate
that DFLSG achieves state-of-the-art performance for
unsupervised TIReID task.
II. R ELATED W ORK
A. Person Re-Identification
Person re-identification (ReID) aims to search and locate
target pedestrian images across cameras according to given
image queries [18]. With the development of deep learning
technologies, many researchers have focused on the person
identification task and proposed some advanced methods [5],
[19], [20], [21]. Some methods construct different network
structures to extract discriminative pedestrian features at different granularities [22], [23], [24]. In addition, some works
design different loss functions to continuously optimize feature
distribution from the representation learning or metric learning
views [25], [26], [27]. The above methods require pedestrian
images with identity information as training data, but identity
information annotation is extremely time-consuming.
Therefore, researchers have begun exploring unsupervised
ReID methods to mine pedestrian key attributes from unlabeled data. Unsupervised ReID methods can be divided into
two categories: unsupervised domain adaptation (UDA) methods and fully unsupervised learning (USL) methods. UDA
methods aim to transfer knowledge learned from labeled
source domain database to unlabeled target domain database
[28], [29]. For example, Wei et al. [30] construct a GAN-based
image style transfer network to change the pedestrian style
from the source domain to the target domain for reducing the
domain gap. Zhai et al. [31] design a mutual learning strategy
and introduce regularization mechanisms in the target domain
to adaptively learn different feature distributions. However,
UDA methods heavily rely on the data quality of the source
domain, and annotating the source domain data also generates
additional resource costs. Hence, some studies have started to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.

SHAO et al.: DFLSG FOR UNSUPERVISED TEXT-TO-IMAGE PERSON RE-IDENTIFICATION

design USL methods that do not require any identity information in the training process [32], [33]. For instance, Dai et al.
[11] employ clustering algorithms to assign pseudo-labels for
unlabeled images and design a clustering-level contrastive loss,
effectively reducing intra-class variations. Some researchers
[34] leverage similarity distributions as soft labels to explore
the relationships among unlabeled images, which promotes
similar images to have consistent feature representations.
These methods have achieved good performance in processing image-based unsupervised ReID task. Unfortunately, when
facing text descriptions, these methods fail to establish the
image-text matching relationship and cannot solve the crossmodality unsupervised learning problem. In this study, we
design an effective method to handle the unsupervised TIReID
task, which greatly improves the model performance.
B. Text-to-Image Person Re-Identification
Li et al. [1] first study the TIReID task, which aims to match
target pedestrian images according to given textual descriptions. They design a recurrent neural network that outputs the
overall sample correlations and publish the first cross-modality
database, CUHK-PEDES. Recently, this challenging task has
received significant attention due to its importance in practical
applications.
Some studies mine robust pedestrian features from the
global view to bridge the inter-modality gap [27], [35]. For
example, Chen et al. [36] design a cross-modality knowledge
adaptive framework to balance the information volume of
two modalities and enhance image-text semantic consistency.
Afterwards, Jiang and Ye [2] utilize implicit relation reasoning and similarity distribution loss to model the relationship
between visual and textual representations. In addition, some
methods introduce extra branches to learn pedestrian key
attributes for improving the fine-grained perception ability. For
example, Shu et al. [37] propose an Implicit Visual-Textual
(IVT) network that leverages multi-level alignment and bidirectional masking to explore sample fine-grained information.
Yan et al. [38] leverage K-Q matrix relations in the transformer
to select important local features and aggregate local attributes
to enhance final feature representation. To further enhance
data diversity, some researchers utilize identity information to
generate additional image-text pairs. For instance, Song et al.
[9] employ the diffusion model guided by clothing accessories
to reconstruct original image-text pairs.
Benefiting from image-text pairs with identity information,
existing supervised methods achieve relatively good performance [39], [40]. However, cross-modality data annotation
typically requires numerous resource consumption. To this
end, researchers have started exploring weakly supervised or
unsupervised TIReID tasks. For example, Bai et al. [15] finetune BLIP to generate text descriptions with key attributes
and introduce text confidence score to mitigate noise attribute
impact. In addition, Zhao et al. [16] utilize clustering operation
to assign pseudo-labels for unlabeled image-text pairs and
design a text-guided matching loss to learn discriminative
visual-textual joint embedding.
Differently, we extend a new method to address how to
obtain dependable text descriptions for building the image-text

1005

matching relationship. In addition, we design ESFM to eliminate abnormal samples for acquiring dependable identity
labels. Finally, we propose MTJL to extract dependability features from center and instance views, which could effectively
mitigate the inter-modality discrepancy. Hence, our method
significantly improves the retrieval performance over other
unsupervised TIReID methods.
III. A PPROACH
In this section, we construct a dependability feature learning
framework based on sample generation, termed DFLSG, to
address the unsupervised TIReID task. We first introduce
the overall structure of DFLSG. Then, the dependable text
generation process is described to construct the image-text
matching relationship and ESFM is illustrated how to obtain
dependability labels. Finally, we introduce MTJL that continuously optimizes intra-modality and inter-modality feature
distribution to bridge the inter-modality gap.
A. Structure of DFLSG
As shown in Fig. 2, we present the overall model structure of
DFLSG. Specifically, considering that CLIP [41] can capture
rich semantic information between image and text samples,
we adopt CLIP as the backbone network. Initially, to obtain
reliable image-text pairs for constructing the cross-modality
dataset, we propose a dependable text generation process.
Then, all input images are resized to 384 × 128, and the
maximum text length is limited to 77 tokens. The DBSCAN
clustering algorithm [42] is utilized to assign pseudo-labels
for the image-text pairs. To enhance the reliability of identity
labels, we design ESFM to identify and eliminate abnormal
samples. Based on refined clustering results, we randomly
select P pedestrian identities where each identity contains K
image-text pairs within each batch. Random padding, random
flipping, random erasing, and random masking are applied
as data augmentation strategies. After extracting pedestrian
features by CLIP encoders, the image and text features are
denoted as vi ∈ R512×1 and t i ∈ R512×1 (i = 1, 2, . . . , PK).
Finally, we construct MTJL to explore dependable feature
distribution from center and instance views.
B. Dependable Text Generation Process
Since texts require manual description, this process is
extremely time-consuming when facing large-scale datasets.
Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
descriptions. Hence, generating rich and reliable text descriptions for different pedestrian images is an urgent problem.
Previous methods fine-tune MLLM to generate text descriptions with key attributes according to specific prompts [15],
[43]. Due to the influence of illumination, occlusion, and
resolution, MLLM may generate hallucinations and output
erroneous attributes, which results in unreliable texts. To this
end, we propose a dependable text generation method to
build the image-text matching relationship and enhance the
dependability of training data. Specifically, ChatGPT [44] possesses strong text understanding and generation capabilities,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.


codex
第二批确认了这类论文的卖点很少是单纯加模块，而是把“可靠性”“非对称性”“中性但保判别”定义成新的训练约束。最后一批我会看视频可见红外、频率嵌入和偏应用型系统文，后面综合时会把偏工程组合的文章降权。
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
/bin/zsh -lc "pdftotext -l 3 'Deep intelligent technique for person Re-identification system in surveillance images.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 162 (2025) 111349

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Deep intelligent technique for person Re-identification system in
surveillance images
Ms. R. Mallika Alias Pandeeswari a,*, Dr． G. Rajakumar b
a
b

Research Scholar Anna University, Francis Xavier Engineering College, Vannarpettai, Tirunelveli 627 003, India
Professor, Department of Electronics and Communication Engineering, Francis Xavier Engineering College, Tirunelveli 627 003, India

A R T I C L E I N F O

A B S T R A C T

Keywords:
Person re-identification
Googlenet
Horned lizard optimization
Preprocessing Accuracy

Person re-identification is the system that aims to attain the re-identity of a particular person captured by
different surveillance cameras. However, it is still a challenging problem in the surveillance system. The more
considerable variation of light conditions, body poses, angles illumination, and occlusion makes it difficult for
the system to re-identify the persons. Recently, the study has been significantly improved by the use of deep
intelligence frameworks. However, it faces some limitations, such as insufficient features and poor accuracy.
Therefore, a novel Horned Lizard Googlenet Forecasting System (HLGFS) is developed in this research to better
result in person re-identification. The novelty of the research lies in integrating Horned Lizard optimization with
GoogleNet for fine-tuned and efficient forecasting to re-identify the person. Initially, the surveillance images
were preprocessed to filter the low-level noise features. Further, the relevant features were extracted based on
the Horned Lizard optimization function. Subsequently, by analyzing the extracted features, the re-identity of the
person is identified and received by matching and ranking. Moreover, the similarity percentage of the query and
identified images was measured through structure similarity. The process of the designed model is tested using
the CUHK03, Market1501, and DukeMTMC re-id dataset in the Python platform. Finally, the forecasting effi­
ciency of the approach is validated and related to existing techniques. The accuracy of HLGFS is 97.8 %, and the
mAP is 97.6 % for the CUHK03 dataset, with 97.68 % accuracy, and 98.87 % mAP for the Market1501 dataset
and for the DukeMTMC re-id dataset, the model achieved 96.65 % accuracy and 96.65 % mAP.

1. Introduction
Person re-identification employs visual data from surveillance film in
non-overlapping views to match persons to a query [1]. It is frequently
and extensively used in public security and video surveillance [2]. The
photographic photos of the same people recorded by several cameras
may change dramatically due to the background clutter and the sharp
variances in angles and lighting [3]. These features act as obstacles to
proper camera matching of pedestrians [4]. Since pedestrians come in a
variety of forms, a quadratic image patch that always includes some
backdrop regions is used to represent a pedestrian for person
re-identification [5]. The majority of low-quality photos are made with
various camera angles [6] and an unregulated backdrop, making faces
and features impractical [7]. In this case, various people are seen as
being quite similar or the same person’s picture seems different [8]. As a
result, creating an intelligent model to examine the security footage is
rapidly approaching [9].

The widespread machine learning and deep learning models for
pedestrian re-identification are employed [10]. Significant advance­
ments in this sector have been made possible by deep learning’s ongoing
development [11]. The learning architectures examined a variety of
factors, including deep visual information, semantic traits, and super­
ficial visual features, to classify people [12]. Significant advancements
in this sector have been developed to examine a variety of factors,
including deep visual information, semantic traits, and superficial visual
features, to classify people [13]. The presence of background noises
makes the model complex to provide an accurate prediction [14].
However, these models have gone through pose misalignment problems
[15]. To overcome this, an optimized intelligent prediction system is
employed. However, the proposed research gives equal importance to all
the parts that extract sufficient features and make it easier for the system
to notice the individual’s identity in the surveillance cameras, and by
selecting the optimal features it improves the prediction accuracy.

* Corresponding author.
E-mail addresses: mallikapandeeswari@francisxavier.ac.in (Ms.R.M.A. Pandeeswari), gmanly12@gmail.com (D.G. Rajakumar).
https://doi.org/10.1016/j.patcog.2025.111349
Received 14 March 2024; Received in revised form 4 December 2024; Accepted 6 January 2025
Available online 10 January 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Ms.R.M.A. Pandeeswari and D.G. Rajakumar

Pattern Recognition 162 (2025) 111349

1.1. Problem statement and objectives

formulation. Henceforth, the present features were analyzed, and the reperson identification was initiated. The Re-person identification was
executed by analyzing the body shape and capturing body or face fea­
tures from the specific test images. Finally, some traditional models
measured and validated the key robustness metrics. The unique contri­
bution and novelty of this study are integrating the Horned Lizard
optimization process with the Googlenet model to make a hybrid person
identification system. Here, before, this Hybrid novel HLGFS is not
tested for person-identification applications. The model includes pre­
processing and feature analysis modules to enhance re-identification
accuracy. The model removed low-level noise features that enhance
the image quality and make the images feasible for feature extraction.
The fitness procedure for the Horned Lizard selected and extracted
sufficient features such as body shape, face, and clothing information for
better prediction and also tuned and adjusted the hyperparameters of
the network model for the robust learning of the features, leading to
increased prediction accuracy. Several studies were done based on
intelligent models for re-person identification, and the models were used
in real-world environments. Each model showcases its benefits and also
exhibits some limitations that forbid it from being used in real-time
applications. The recent model surveyed in the related work section
has various limitations, such as lower identification accuracy due to the
lower image quality, insufficient features identification stage, compu­
tation complexity, increased cost, and increased time consumption. A
new model must be introduced to address all these limitations of the
prevailing person re-identification system. The recommended model
carried out robust preprocessing steps that enhance the quality of the
images for easy identification, and essential features are extracted by the
optimal feature selection process, reducing the time and computation
complexity.
The article is arranged as given: Section 2 reviews the traditional
literature, Section 3 explains the problems in the re-identification
frameworks, Section 4 describes the solution for the stated problems
with a new method, Section 5 discusses the results of the created solu­
tion, and Section 6 noted the conclusion of the research with future
work.

The urgency of public safety has increased surveillance systems; reperson identification is imperative in the intelligence surveillance sys­
tem. Re-person identification is a widely used method of retrieving
specific persons from different camera angles. It is one of the very
challenging tasks due to the different viewpoints of the particular per­
sons, varying illuminations, camera environments, occlusion, back­
ground, etc. The images from the surveillance camera are too noisy,
increasing the complexity score in identifying the persons who have
been recorded for crimes. Several bio-inspired models were imple­
mented in the past for this re-person identification system, but it could
not gain the expected optimal solutions, due to their lack of feature
selection. All optimization algorithms like particle swarm algorithm,
grey wolf, Ant colony, Hyena optimal model, etc., can executed up to the
optimal solution iteration, once, the optimal solution is found then the
optimization iteration is stopped. In this case, the features identifying
face challenges due to the stop iteration of optimal solution inaccurate
feature selection is recorded. HLGFS is developed to address several
critical challenges in person re-identification, particularly within sur­
veillance systems used for public safety. It incorporates horned lizard
optimization because it is ideal for hunting. Its hunting fitness is much
better than bio-inspired mechanics, due to their skin characteristics
changing behaviour. This behavior has attracted this research study for
identifying the features of every different characteristic video frame, by
changing its behavior fitness. Here, the skin color-changing fitness is
updated in the dense layer of Googlenet, which afforded the finest
feature selection outcome for all tested video frames. The main reason
for choosing the googlenet, it can train more and different format video
frames. Hence, incorporating the horned lizard features in the Googlenet
and making the hybrid prediction mechanism “HLGFS” is considered the
key novelty of this study. Practically, it is utilized for criminal or crime
identification by conducting the investigation on surveillance cameras
in public areas. In addition, for identifying the person in crowded scenes,
and dynamic lighting, the skin change behavior of the horned lizard was
utilized for recognizing the unique characteristics of the tested person
from the recorded crowed video frames.
The prime objectives of this work are re-person predictive framework
was introduced in the artificial intelligence field with the bioinspired
model to enhance the camera identification system. Here, the Re-person
was identified by processing the intelligent concepts. Hence, the specific
objectives of this study are:

2. Related work
Existing works related to the re-identification task are described
below,
Nguyen et al. [16] provided an enlarged dataset with wearable,
CCTV, and aerial technologies for aerial-ground ReID study. To assist
ongoing studies in this area, the company will make the baseline code
and enriched dataset freely available to the community. However, po­
tential bias in the data and reliance on the head region could prevent
generalization. Huang et al. [17] propose a sequential step learning
architecture (SSLA) that improves the feature extraction performance of
the re-identity network by co-segmentation. Focusing on large-scale
spatiotemporal human re-identification, the Large Scale Spatio Tempo­
ral (LaST) dataset is the most significant and efficient labeled re-ID
benchmark, by Xiujun et al. [18]. Long-term and cloth-changing cir­
cumstances demonstrate LaST’s strong generalization capacity.
Although it may not provide all the obstacles needed to explore
long-term scenarios, the dataset seeks to stimulate investigation in the
re-ID domain and promote the development of re-ID methods for
real-world scenarios. To re-identify people, Yiheng et al. [19] suggested
an end-to-end foreground-aware network (EFAN) that uses camera IDs
from pre-existing datasets to locate background regions and create a soft
foreground mask. This method promotes foreground and background
branches for more robust and discriminative feature representations,
using target improvement modules and attention loss. A deep
learning-based technique for surveillance system human recognition is
presented by Choudhary et al. [20]. It trains a multi-tasking model using
a Siamese architecture (SA) with similarity and classification con­
straints. The technique may not be as applicable in the actual world due

• Introduces the HLGFS hybrid system that combines horned lizardinspired optimization and the Googlenet architecture to enhance
person Re-identification capabilities.
• Improves feature extraction by the horned lizard optimization
mechanism so the adaptive identification of relevant features within
noisy, occluded, or dynamic conditions can be done by the system.
• Avoids the problems faced by bio-inspired optimization algorithms
by maintaining a continuously adapting fitness mechanism to ensure
a robust and accurate selection of features.
• Validates the performance of the HLGFS on widely recognized
datasets CUHK03, Market1501, and DukeMTMC to testify to its
robustness.
Research Question: The features of the traditional intelligent
approach are insufficient to estimate the recorded performance through
re-person identification. It increased the identification error with less
accuracy. To resolve these issues, the current work has focused on
building a novel tuned forecasting framework for Re-person identifica­
tion. The essential contribution is explained as follows,
The surveillance image databases were gathered and taken as Python
input during the prime process. Consequently, a novel HLGFS was built
with the required critical functional parameters. Here, the image noises
were analyzed and eliminated at the preprocessing stage by filtering
2

Ms.R.M.A. Pandeeswari and D.G. Rajakumar

Pattern Recognition 162 (2025) 111349

to its reliance on paired input during training. Zhu et al. [21] introduced
an automatic aligning transformer (AAT) for the person’s re-identity
prediction. However, its response is low for the background patches.
Chen et al. [22] used attention pyramid architecture (APS) to exploit the
attention area in the person re-identification. The model works based on
the split and merge principle. The computational cost and time is very
high. Dong et al. [23] proposed a framework that includes the
multi-view characteristics of the person images. It integrates feature
maps to describe the target pedestrian. The model achieved higher
performance benefits. However, it may not be optimal for superior in­
formation extraction. The attention mechanism may ignore essential
feature information. Wang et al. [24] proposed a multi-deep supervision
with attention features to address this issue. Multi-structure and deep
supervision have been used to remedy the essential global feature in­
formation loss. However, the model faced cross-modality issues due to
omitting local and salient features. Gupta et al. [25] suggested the re­
sidual neural network and transfer learning for the better learning of
visual features in the person-identification process. In addition, hyper­
parameters’ influence is explored in this model. The overall comparison
of the discussed related works is presented in Table 1.

Table 1
Advantages and limitations of related works.
Authors

Method

Advantages

Disadvantages

Nguyen
et al. [16]

Attribute-based,
three-stream
ReID technique

Both rank accuracy
and total
performance have
improved.

Huang et al.
[17]

Sequential step
learning
architecture

Xiujun et al.
[18]

LaST

Yiheng et al.
[19]

End-to-end
foregroundaware network

Choudhary
et al. [20]

A multi-tasking
model using a
Siamese
architecture

It achieves notable
performance gains
with more distinct
and temporal
invariant body
attributes.
Strong
generalization
capacity
The method
promotes
foreground and
background
branches for more
robust and
discriminative
feature
representations.
Increased accuracy
performance and
efficient validation
process

Potential bias in the
data and reliance on
the head region could
prevent
generalization.
Complexity,
computing expenses,
and parameterization
are increased.

Zhu et al.
[21]

AAT

Chen et al.
[22]

Attention
pyramid
architecture

Dong et al.
[23]

Multi-view
characteristicsbased framework

Wang et al.
[24]

Multi-deep
supervision with
attention
features
Residual neural
network and
transfer learning

3. Proposed methodology: HLGFS
The proposed architecture for the Re-ID works is based on integrating
the Horned Lizard optimization [26] and GoogleNet [27]. The horned
lizard optimization is the metaheuristic function designed based on the
adaptive and defensive behavior of the horned Lizard. This optimization
approach effectively balances the exploration and exploitation phases. It
converges robustly toward near-optimal solutions. By mimicking the
lizard’s adaptive color changing, it adapts its search strategy dynami­
cally according to the problem space to enable the re-identification
system to navigate diverse solution areas. Moreover, the use of an
adaptive mechanism based on iteration progress and solution quality
adjusts the step size to avoid the inefficiencies associated with a balance
between exploration and exploitation. Integration into the Horned Liz­
ard GoogLeNet Forecasting System achieves an improved ability to
extract critical features to identify people through the optimization
technique. Utilizing GoogLeNet’s inception layers for multi-scale feature
extraction coupled with HLO’s adaptive exploration, HLGFS provides a
more detailed and nuanced understanding of the input data. This syn­
ergy enables the capturing of subtle details of the features. HLGFS
thereby enhances identification and demonstrates the novelty of HLO in
feature extraction and re-identification systems. The block structure of
the proposed schema is visualized in Fig. 1.
The process starts with the data training function. The data input and
the learning process of the proposed HLGFS model are detailed in Eqn.
(1).
T(Sd ) = in (n = 1, 2, ...x)

Gupta et al.
[25]

(1)

It significantly beats
the other researched
models, even in
large datasets.
Better
understanding of the
features and
enhanced
identification
accuracy

Its adaptation to
different surroundings
may be limited due to
its dependence on
camera IDs.

The technique may
not be as applicable in
the actual world due
to its reliance on
paired input during
training.
Its response to the
background patches is
low.
The computational
cost and time is very
high.
It may not be optimal
for superior
information
extraction.
The model faced
cross-modality issues
due to omitting local
and salient features.
Influence of
hyperparameters

the noise tracing variable, f indicates the standard pedestrian features
and the term k denotes the low-level features. Based on this Eqn. (2), the
preprocessing layers eliminated the several noisy features for better
identification.

Here T indicates the data training function, Sd indicates the sur­
veillance image database, in represents the trained person images, and x
represents the total number of images present in the dataset. The layers
of the designed HLGFS model are visualized in Fig. 2.
The designed architecture comprises 11 layers: an input layer, 4
convolution layers (C1, C2, C3, and C4), 3 pooling layers (P1, P2, and
P3), an inception layer, a fully connected layer, and an output layer.

3.2. Feature extraction
The proposed HLGFS for re-person identification follows feature
extraction and comparing those features with the other images. The
success of the re-person identification relies on the extraction of better
features. In the proposed research, the best features were extracted
based on the cryptic behavior of the horned Lizard. In this phase, the
Lizard imitates the characteristics of the environment, such as color,
texture, etc., to blend with the surroundings. Using this behavior, the
presented system extracts a person’s body shape, face, and clothing at­
tributes. The features are earned by Eqn. (3).

3.1. Preprocessing
The collected surveillance images contain noises that make the sys­
tem complex to extract sufficient features and identify the re-person. The
preprocessing phase reduces the identification complexity and thereby
increases the model’s overall accuracy.
P = Sd × λ [(f, k) − (k)]

The system locates
the human and nonhuman regions at
the patch level in
self-manner
The model focuses
on the clothing
features such as
color, even in the
faded background.
The model achieved
higher performance
benefits

It doesn’t use for longterm scenarios

(2)

Here, the preprocessing step is defined as the variable P, λ denotes
3


exec
/bin/zsh -lc "pdftotext -l 3 'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1766

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

Discovering Multi-Frequency Embedding for
Visible-Infrared Person Re-Identification
Hongyang Gu , Xiaogang Yang , Ruitao Lu , Lei Pu , Siming Han, and Ming Wu

Abstract—Visible-Infrared Person Re-identification (VI-ReID)
is critical for round-the-clock surveillance systems yet is hindered
by significant modality discrepancies. Existing methods often fail
to fully exploit frequency domain information, focusing predominantly on spatial domain feature learning or limited frequency
decompositions. To address this, we propose the Multi-Frequency
Embedding Network (MFENet), a feature-level method that
operates in the frequency domain through multi-frequency
decomposition to learn discriminative and modality-invariant features. Specifically, the HiLo-Frequency Modulation (HiLo-FM)
module efficiently extracts low-frequency features via frequencydomain filtering and high-frequency details through lightweight
multiscale convolutions, followed by attention-based fusion. The
Frequency-Aware Diversity Enhancer (FADE) module further
enriches feature discriminability by weighting multi-frequency
components and learning diverse features through multi-branch
architectures. To further enhance the performance of our method,
we introduce two innovative loss functions. The Cross-Modality
Soft Retrieval (CMSR) loss prioritizes cross-modality consistency over intra-modality similarity, while the Cross-Modality
Ranking Regularization (CMRR) loss enhances feature diversity
through differentiable rank correlation optimization. Extensive
experiments demonstrate the state-of-the-art performance of our
method, achieving 61.06% Rank-1 and 67.75% mAP in the
challenging IR to VIS mode on the largest VI-ReID benchmark
LLCM, surpassing existing methods by significant margins without resorting to reranking or additional labeled data. Code is
available at https://github.com/GuHY777/MFENet-VIReID.
Index Terms—Person re-identification, cross-modality, visibleinfrared, Fourier transform, retrieval loss.

I. I NTRODUCTION

P

ERSON re-identification (ReID), a critical technology for
intelligent surveillance systems and urban management
[1], aims to learn discriminative features that maintain identity consistency across non-overlapping camera views. While
recent advances in visible-spectrum recognition [2], [3], [4]
have achieved remarkable progress, these methods remain

Received 30 May 2025; revised 26 August 2025; accepted 18 September
2025. Date of publication 22 September 2025; date of current version
5 February 2026. This work was supported in part by the National Natural
Science Foundation of China under Grant 62401609, in part by the Natural
Science Basic Research Plan in Shaanxi Province of China under Grant
2024JC-YBQN-0628 and Grant 2025JC-YBMS-730, and in part by the China
Postdoctoral Science Foundation under Grant 2024M754275. This article
was recommended by Associate Editor X. Shu. (Corresponding author:
Hongyang Gu.)
The authors are with the Rocket Force University of Engineering,
Xi’an 710025, China (e-mail: guhy7@outlook.com; doctoryxg@163.com;
lrt19880220@163.com; warmstoner@163.com; hansm119@outlook.com;
hyacinth531@163.com).
This article has supplementary downloadable material available at
https://doi.org/10.1109/TCSVT.2025.3612751, provided by the authors.
Digital Object Identifier 10.1109/TCSVT.2025.3612751

fundamentally limited to daylight applications due to their
dependence on optimal illumination conditions.
To enable round-the-clock surveillance capabilities, crossmodality visible-infrared person re-identification (VI-ReID)
[5] has emerged as a pivotal research direction, addressing the
significant modality gap between visible (VIS) and infrared
(IR) imaging. Current VI-ReID methods can be broadly categorized into two paradigms: 1) Image–level methods employ
generative models [6], [7], [8], [9], [10] or basic transformations [11], [12], [13], [14], [15], [16] to bridge the modality
gap through input-space alignment. However, generative models suffer from training instability and detail degradation, while
simple transformations prove inadequate for handling complex cross-modality variations. 2) Feature-level methods focus
on architectural innovations and specialized loss functions.
ResNet50 [17] remains the predominant backbone in VI-ReID
research, while emerging architectures like Vision Transformers (ViTs) [18] demonstrate comparable efficacy [19], [20],
[21]. Both frameworks achieve satisfactory performance, with
ResNet50-based methods dominating current implementations
and ViT methods showing growing promise for cross-modality
scenarios, especially with additional multimodal information
[22], [23]. For loss function design, in addition to the crossentropy loss [24] that implicitly aligns modalities, specialized
cross-modality losses [1], [25], [26] explicitly minimize intraclass distances while maximizing inter-class separability and
mitigating modality discrepancies. In practice, compared with
image-level methods, feature-level methods tend to deliver
stronger recognition performance. However, most existing
feature-level methods predominantly focus on spatial-domain
features, leaving the frequency domain insufficiently leveraged
for modality-invariant learning.
Recent advances in frequency domain analysis have
demonstrated remarkable success across various vision tasks,
including image deraining [27], denoising [28], and low-light
enhancement [29]. As shown in Fig. 2, Fourier analysis reveals that amplitude spectra (Fig. 2(b)) primarily
encode modality-specific characteristics, while phase spectra
(Fig. 2(c)) preserve structural information across modalities. These findings have motivated several frequency-based
VI-ReID methods. FDMNet [30] and FDNM [31] focus on
amplitude spectrum alignment while preserving phase information. In addition, DSSF3 [32] advocates joint learning of
both spectral components. These methods perform broadspectrum feature mining across the entire frequency band (top
left of Fig. 1), but overlook the distinct roles of low/highfrequency in modality-invariant feature learning.

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.

GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION

1767

Fig. 1. The comparison of existing frequency-based methods and our proposed
method. Existing methods either perform broad-spectrum feature mining
across the entire frequency band or rely on simple low/high-frequency decomposition, which often adopts coarse and inefficient feature learning in the
spatial domain. Conversely, our method achieves efficient low/high-frequency
feature learning through the meticulously designed HiLo-FM module
(Sec. III-B). Meanwhile, the FADE module (Sec. III-C) leverages multifrequency attention to generate richer, more robust, and modality-invariant
features.

Further decomposition into low-frequency (Fig. 2(d))
and high-frequency (Fig. 2(e)) components provides deeper
insights. The low-frequency component, capturing the overall
shape of a person, is heavily influenced by modality-specific
information, such as color and illumination in VIS images
or thermal radiation in IR images. In contrast, the highfrequency component effectively filters out modality-specific
styles, preserving shared structural details like edges and contours, thereby making the two modalities appear more similar
[19]. Notable methods for low/high-frequency decomposition
include FSDF [33] and BiFFN [34] (top right of Fig. 1).
FSDF leverages discrete cosine transform in conjunction with
fundamental 1 × 1 convolutional operators, whereas BiFFN
integrates wavelet transforms with resource-demanding Graph
Neural Networks (GNNs) [35]. Although these two methods
have demonstrated certain effectiveness, two critical challenges
persist. First, they predominantly execute low/high-frequency
feature learning in the spatial domain, thereby neglecting
the potential for efficient frequency-domain processing. Second, and more critically, they rely on limited frequency
decompositions, focusing solely on low/high-frequency components. This limitation is particularly suboptimal, as our
visual analysis in Fig. 2 reveals. While low-frequencies
(Fig. 2(d)) are dominated by modality-specific styles and highfrequencies (Fig. 2(e)) better preserve shared structures, a
person’s identity is inherently multi-scale. Crucial cues, from
overall posture to fine-grained textures (e.g., the logo on

Fig. 2. The visualization of frequency components for a pair of VIS (odd
rows) and IR (even rows) images. The figure shows the (a) original images
and the reconstructions from (b) amplitude (A), and (c) phase (P). The lowfrequency (L) and high-frequency (H) components, shown in (d) and (e), are
separated using a frequency mask as detailed in Eq.(4). The multi-frequency
decompositions in (f) are generated by partitioning the spectrum into multiple
non-overlapping bands, formally defined in Eq.(17). This visualization highlights how modality-invariant cues are distributed across different frequency
bands, motivating our method.

the person’s clothing), are scattered across various frequency
bands, with their prominence varying between modalities,
as shown in the multi-frequency decompositions (Fig. 2(f)).
Relying on a single, handcrafted frequency cutoff thus forces
a poor trade-off, either discarding valuable identity cues or
retaining excessive modality-specific noise. This underscores
the necessity of a more flexible, multi-frequency analysis to
capture these scattered yet discriminative features.
Motivated by the aforementioned findings, we propose the
Multi-Frequency Embedding Network (MFENet), a featurelevel method that learns modality-invariant representations
from the frequency domain via multi-frequency decompositions. Specifically, to address the challenge of efficient
low/high-frequency feature extraction, we introduce the HiLoFrequency Modulation (HiLo-FM) module. This module
employs the more flexible Fast Fourier Transform (FFT)
[36] for low/high-frequency decomposition, directly performs
efficient filtering on low-frequency features in the frequency
domain, and conducts lightweight multi-scale feature mining

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.

1768

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

on high-frequency features in the spatial domain. To mitigate
unavoidable noise, these low/high-frequency features are then
fused using spatial-attention-based modulation. To achieve
multi-frequency feature learning, we propose the FrequencyAware Diversity Enhancer (FADE) module, which can be
viewed as an extension of Instance Normalization (IN) [37]
in the frequency domain. By weighting multiple pre-set frequencies from low to high, the module accentuates feature
differences across bands and learns rich modality-invariant
features with a multi-branch structure. Additionally, considering the cross-modality retrieval characteristics of the VI-ReID
task, we propose two loss functions to further improve the
robustness and diversity of cross-modality features: CrossModality Soft Retrieval (CMSR) loss and Cross-Modality
Ranking Regularization (CMRR) loss. The CMSR loss is
specifically devised to accentuate cross-modality matching. It
enforces that the maximum inter-modality distance between
instances of the same class should be smaller than the
minimum intra-modality distance between instances of the
same class. This mechanism directs the focus more towards
cross-modality features rather than intra-modality features.
Meanwhile, the CMRR loss is dedicated to augmenting the
diversity of cross-modality features. It achieves this by maximizing the inconsistency in ranking among features extracted
by different branches in the FADE module. To tackle the
challenge posed by non-differentiable ranking, a differentiable
Spearman rank function is incorporated.
The contributions of our work are summarized as follows:
• We propose MFENet, a feature-level method that operates
in the frequency domain and introduces multi-frequency
decompositions to learn robust modality-invariant features
for VI-ReID. This method incorporates two key modules: the HiLo-FM module, which efficiently extracts and
fuses low/high-frequency features, and the FADE module, which effectively enhances feature discriminability
through diverse frequency responses.
• We propose two novel loss functions for cross-modality
retrieval tasks: CMSR loss, which enhances the robustness of cross-modality features by focusing more
on cross-modality features learning rather than intramodality features learning, and CMRR loss, which
promotes the diversity of cross-modality features by
maximizing the inconsistency in ranking among features
extracted by different branches.
• We conduct extensive experiments on VI-ReID benchmarks SYSU-MM01, RegDB, and LLCM, which validate
the effectiveness of our designed modules and loss functions, and demonstrate that MFENet outperforms most
state-of-the-art methods.
II. R ELATED W ORK
A. Visible-Infrared Person Re-Identification
The field of VI-ReID [5] enables 24-hour surveillance capabilities by matching cross-modality images, yet it is confronted
with significant modality discrepancies between visible and
infrared images. To address these disparities, existing methods
primarily focus on two aspects:

1) Image-level methods leverage Generative Adversarial
Networks (GANs) [38] and Diffusion Models (DMs) [39], or
employ basic operations and modules to enrich input images
and bridge the VIS-IR gap. Although GAN-based methods [6],
[7], [8] are widely used, they generally exhibit inferior image
generation quality when compared to DM-based methods [9],
[10]. Moreover, generative models typically require substantial
computational resources and time, and are prone to introducing
noise. Some methods either design intuitive enhancement
operations [14], [15], [16] or utilize simple modules [11], [13]
for intermediate modality generation. However, these methods
often rely heavily on domain-specific knowledge, are overly
simplistic, and struggle to adapt to the complex scenarios
inherent in VI-ReID tasks effectively.
2) Feature-level methods concentrate on developing
advanced network architectures or loss functions. The majority
of existing works utilize ResNet50 [17] as the backbone
network, and enhance performance through attention mechanisms [1], local feature mining [40], multi-scale learning [41],
and high-order structure [25]. These methods have achieved
promising results. With the advent of ViTs [18] in recent years,
numerous studies have explored ViT-based architectures [19],
[20], [21], [23], [40], [42]. Despite their good performance,
these architectures still lag behind ResNet50 under the same
ImageNet pre-trained model settings. Their high computational
complexity and large number of parameters also limit their
application in scenarios with limited computational resources.
In terms of loss function design, in addition to the commonly
used cross-entropy loss [24], various loss functions specifically
designed for cross-modality scenarios [1], [25], [26] have
been employed. These functions aim to minimize intra-class
distances, maximize inter-class distances, and account for the
differences between modalities.
In practice, feature-level methods tend to be more stable
and computationally economical than image-level methods.
Nevertheless, the vast majority of feature-level VI-ReID methods still learn in the spatial domain, and thus under-utilize
frequency-domain cues that can be modality-invariant. In this
work, we advance the feature-level line by explicitly operating
in the frequency domain, conducting multi-frequency feature
learning to mine robust and diverse modality-invariant representations.
B. Frequency-Domain Analysis in Deep Learning
Frequency-domain analysis [43], a subfield of image
processing, converts spatial domain images into frequency
features to uncover concealed patterns within image data.
Unlike spatial-domain analysis, which directly manipulates
pixel values, frequency-domain analysis dissects images into
various frequency components, thereby facilitating the differentiation and manipulation of subtle features such as noise and
texture. Recent advancements in frequency-domain analysis
have achieved remarkable success across a wide range of
vision tasks, including image deraining [27], denoising [28],
and low-light enhancement [29].
In the field of VI-ReID, several recent works [30], [31], [32],
[33], [34] have initiated the exploration of frequency-domain
feature learning for VI-ReID tasks. Notably, FDNM [31] and

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
DIRL: Learning Discriminative ID-Related Representations
for Video Visible-Infrared Person ReID
JIAHE WANG, XIZHAN GAO, SIJIE NIU, HUI ZHAO, and GUANG FENG, Shandong
Provincial Key Laboratory of Ubiquitous Intelligent Computing, School of Information Science and
Engineering, University of Jinan, Jinan, China
The core of Video-Based Visible-Infrared Person Re-Identification (VVI-ReID) lies in learning modal-sharing
features, namely ID-related feature representations, that are often mixed within modal-invariant features.
Existing methods use weight sharing networks to learn frame-level modal-invariant features, but fail to
consider modality invariance when computing sequence-level features, resulting in a significant gap between
modalities. Moreover, these methods do not explicitly separate the ID-related features and modal-related
features, which reduces the model’s discriminative ability and leads to interference in VVI-ReID. In this article,
we propose a Discriminative ID-Related Representation Learning (DIRL) network for VVI-ReID. DIRL network
consists of three key components, that is Two-Stream Backbone Module (TBM), Cross-Modality Interaction
Module (CIM), and Feature Decoupling Module (FDM). More specifically, the TBM is first constructed to
preliminarily capture frame-level modal-invariant features. Then, the CIM is designed to interact information
between modals and aggregate temporal features simultaneously, thereby obtaining sequence-level modalinvariant features. Finally, the FDM is designed to explicitly separate modal-related features from ID-related
ones within the modal-invariant features, thereby leaving only discriminative ID-related representations.
Through extensive benchmark experiments, our method demonstrates superior performance over state-ofthe-art approaches by significant margins. Our code will be available at https://github.com/JhSearch/DIRL.
CCS Concepts: • Information systems → Information retrieval; • Computing methodologies →
Computer vision tasks;
Additional Key Words and Phrases: Video-based visible-infrared person re-identification, feature decoupling,
id-related feature learning, cross-modality video retrieval
ACM Reference format:
Jiahe Wang, Xizhan Gao, Sijie Niu, Hui Zhao, and Guang Feng. 2025. DIRL: Learning Discriminative ID-Related
Representations for Video Visible-Infrared Person ReID. ACM Trans. Multimedia Comput. Commun. Appl. 21,
8, Article 238 (August 2025), 16 pages.
https://doi.org/10.1145/3745784
This work was supported by the National Natural Science Foundation of China under Grant Nos. 62101213 and 62471202.
Authors’ Contact Information: Jiahe Wang, Shandong Provincial Key Laboratory of Ubiquitous Intelligent Computing, School
of Information Science and Engineering, University of Jinan, Jinan, China; e-mail: 202221100455@stu.ujn.edu.cn; Xizhan Gao
(corresponding author), Shandong Provincial Key Laboratory of Ubiquitous Intelligent Computing, School of Information
Science and Engineering, University of Jinan, Jinan, China; e-mail: ise_gaoxz@ujn.edu.cn; Sijie Niu, Shandong Provincial Key
Laboratory of Ubiquitous Intelligent Computing, School of Information Science and Engineering, University of Jinan, Jinan,
China; e-mail: ise_niusj@ujn.edu.cn; Hui Zhao, Shandong Provincial Key Laboratory of Ubiquitous Intelligent Computing,
School of Information Science and Engineering, University of Jinan, Jinan, China; e-mail: ise_zhaohui@ujn.edu.cn; Guang
Feng, Shandong Provincial Key Laboratory of Ubiquitous Intelligent Computing, School of Information Science and
Engineering, University of Jinan, Jinan, China; e-mail: ise_fengg@ujn.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/8-ART238
https://doi.org/10.1145/3745784
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.

238:2
1

J. Wang et al.

Introduction

Person Re-Identification (ReID) is a crucial task in computer vision that focuses on identifying the
same individual across different times, cameras, or scenes. It plays a significant role in applications
such as video surveillance, intelligent security systems, crowd monitoring, and other related fields.
Although some deep end-to-end methods [1, 2] have achieved notable success. Most of these
approaches are highly dependent on adequate lighting conditions [3–5]. However, in real-world
applications, such optimal lighting conditions are not always available. For example, in low-light
environments such as nighttime, visible light images (RGB images) often struggle to capture clear
and accurate features of a person. In these cases, infrared images provide a practical alternative by
preserving essential person-related information, even when visible images fall short. Leveraging
this advantage, Visible-Infrared Person Re-Identification (VI-ReID) offers a viable approach
to tackle the challenges posed by such conditions.
VI-ReID aims to match the same individual from a still-image or video sequence across visible and
infrared camera views, resulting in Image-Based Visible-Infrared Person Re-Identification
(IVI-ReID) task [6–8] and Video-Based Visible-Infrared Person Re-Identification (VVIReID) task [9–11]. The IVI-ReID task primarily involves addressing the challenge of cross-modality
feature alignment. For example, Chen et al. [6] proposed a quadruplet deep network with an
attention mechanism to capture global spatial features, thereby reducing spatial misalignment and
semantic inconsistency across modalities. Hermans et al. [7] introduced a triplet loss to minimize
intra-class cross-modality distance while maximizing inter-class separation, enhancing identity
feature alignment. Wang et al. [8] proposed MPMN with modality-specific and shared branches to
disentangle modality noise and preserve identity semantics, effectively addressing feature alignment
at both global and local levels. However, these IVI-ReID methods are designed based on still images,
and their performance often degrades when applied to real-world scenarios (e.g., video data). Due
to variations in pose, movement, perspective, lighting conditions, and spatial placement over time,
video data poses unique challenges for accurate identification.
In contrast, VVI-ReID can effectively handle video data. VVI-ReID primarily focuses on addressing the challenges of spatial-temporal feature learning and cross-modality feature alignment.
For example, Li et al. [9] proposed the Intermediary-Guided Bidirectional Spatial–Temporal
Aggregation Network (IBAN), which uses Convolutional Neural Networks (CNNs) to capture
local features within individual frames, uses Long Short-Term Memory (LSTM) networks to
extract temporal cues from sequential data, and uses an intermediary-guided strategy to bridge
the gap between different modalities. However, when processing lengthy video sequences, LSTM
models struggle to effectively capture temporal nuances and long-range dependencies [10], due
to the sequential nature of their computation and the inherent forgetting mechanism. Moreover,
when handling scenarios involving modality fusion, especially in later video sequences with occlusions and background interference, the model tends to overlook modal invariance. This issue
is particularly evident in earlier video segments. This further exacerbates the gap between features
from different modalities, allowing modal-related information to interfere with the re-identification
task, ultimately diminishing the model’s discriminative ability. Similarly, Lin et al. [11] proposed
MITML, which uses a weight-shared CNN to construct spatio-temporal features, facilitating the
learning of frame-level modal-invariant features. It then incorporates a modal-invariant adversarial
loss to align cross-modality features, ensuring that these features preserve both ID-related and
modality-related information. However, this method only obtains frame-level modal invariance and
does not account for modality invariance when computing sequence-level features, exacerbating
the gap between sequential features of different modalities (as shown in Figure 1). Furthermore,
without further separating modal-invariant features, modal-related information can interfere with
the re-identification task. This interference ultimately reduces the model’s discriminative ability.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.

DIRL for Video Visible-Infrared Person ReID

238:3

Fig. 1. Compared to the existing methods, our approach emphasizes the interaction of information between
modalities when extracting sequence-level features, further reducing the feature gap between different
modalities.

To address the aforementioned issues, we propose a Discriminative ID-Related Representation Learning (DIRL) network for VVI-ReID. DIRL network consists of three key components,
that is, Two-Stream Backbone Module (TBM), Cross-Modality Interaction Module (CIM),
and Feature Decoupling Module (FDM). Due to the superiority of the CNN structure in capturing local features of pedestrians, we use two-stream CNN network as our backbone to capture
frame-level modal-invariant features. The CIM, an improved version of attention mechanisms,
enables modality interaction during the adaptive aggregation of temporal information, thereby
obtaining sequence-level modal-invariant features. The FDM is composed of two trainable feature
decoupling matrices, which explicitly separate ID-related information from modal-related information in the modal-invariant features, resulting in more discriminative and modal-sharing ID-related
representations. The contributions of our work are summarized as follows:
— We proposed a CIM to perform modality information interaction and temporal feature aggregation simultaneously, thereby obtaining sequence-level modal-invariant features.
— We developed an FDM and designed its corresponding loss function to effectively separate
ID-related information from modal-related information, thereby leaving only discriminative
ID-related representations.
— By integrating TBM, CIM, and FDM, we proposed the DIRL network for VVI-ReID. Extensive
experiments are conducted on two public video ReID datasets. The experimental results
demonstrate that the proposed method achieves competitive performance compared to some
State-of-the-Art (SOTA) methods.
2

Related Work

In this section, we will briefly review some related works on IVI-ReID and VVI-ReID methods.
2.1

IVI-ReID

IVI-ReID involves matching images in visible and infrared modalities. Existing IVI-ReID methods
can be categorized into four groups: modal-sharing feature learning-based methods [12–15], modalspecific information compensation-based methods [16–18], modality generation methods [19, 20],
data augmentation methods [21–24], and feature decoupling methods [25–27].
Modal-sharing feature learning-based methods focus on aligning features from different modalities in a shared feature space. They aim to reduce cross-modality discrepancies through feature-level
constraints. For example, Wei et al. [12] proposed a flexible body partition model-based adversarial learning method, incorporating fine-grained part representations and adversarial learning
with a modality classifier to reduce the cross-modality gap and enhance feature distinguishability.
Similarly, Feng et al. [13] proposed a shape-removed feature learning framework that decorrelates
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.


codex
已按要求只用 `pdftotext -l 3` 读了每篇前 3 页。下面是摘要和引言层面的反推，不代表全文细节。

**逐篇提取**
1. `Cross-modality average precision optimization...`
创新类型：新机制。把可见光和热红外 ReID 的优化目标从三元组局部配对改成跨模态可微 AP 排序。
为什么能发：指出 triplet 和 Smooth-AP 都没有直接优化跨模态检索排序。证据链是 CAPV、CAPT、动态模态对齐、全局局部特征逐项消融，加 SYSU-MM01 和 RegDB 对标。
story 套路：跨模态 ReID 不该只拉近难正样本，而要直接优化最终跨模态排序。

2. `CSGN...`
创新类型：新机制偏工程组合。CLIP 伪文本、RGB 和灰度双描述、ResNet 和 ViT 并行视觉引导、异构语义融合。
为什么能发：抓住换衣 ReID 中 CLIP 文本容易关注衣服颜色这个缺口。证据链是四个换衣数据集，对 MDG、VSS、HSF 做消融。
story 套路：用语言语义削弱衣服颜色捷径，让模型从视觉瓶颈里出来。

3. `CVAF...`
创新类型：问题重定义加新机制。把 aerial-ground ReID 说成“视角一致语义保留”，反对粗属性依赖和强行视角解耦。
为什么能发：AGPReID 里航拍低清、尺度小、视角极端，属性不可靠，过度解耦会丢身份线索。证据链是 CLIP 身份文本 token、文本一致性损失、语义过滤模块，多 AGPReID 数据集对标。
story 套路：跨视角不是把视角信息删掉，而是保留跨视角仍稳定的身份结构。

4. `CycleTrans...`
创新类型：新机制。用循环构造学习“中性但可区分”的可见红外特征。
为什么能发：现有模态对齐会牺牲判别性。证据链是 KCM、DMM、循环重构消融，加 SYSU-MM01 和 RegDB。
story 套路：跨模态特征要去模态化，但不能把细粒度身份语义一起抹掉。

5. `DATE...`
创新类型：新机制。把可学习文本嵌入和描述文本嵌入做非对称融合，描述文本只当辅助线索。
为什么能发：MLLM 自动描述有噪声，同步融合会污染视觉表示。证据链是两个 adapter、身份级描述均值、跨粒度训练，在 Market、Duke、MSMT 上验证。
story 套路：文本不是裁判，只能作为有噪声的辅助证据。

6. `Deep intelligent technique...`
创新类型：工程组合。Horned Lizard 优化加 GoogleNet 做特征选择和分类。
为什么能发：它包装的是噪声监控图像下的特征不足问题，但机制和证据都比较泛，报告指标也异常高。
story 套路：生物启发优化帮助选特征。对我们启发很弱，不建议沿这个方向学。

7. `Dependability Feature Learning...`
创新类型：新数据设定加新机制。无监督文本图像 ReID，既没有图文匹配关系，也没有身份标签，先生成并筛可靠样本。
为什么能发：真实场景下文本标注和身份标注都贵。证据链是多 MLLM 文本生成、可靠文本筛选、异常样本过滤、中心和实例两级三元组学习。
story 套路：先解决训练样本是否可信，再谈跨模态对齐。

8. `DIRL...`
创新类型：问题重定义加新机制。视频可见红外 ReID 不能只做帧级模态不变，还要做序列级交互和身份模态解耦。
为什么能发：视频序列里模态干扰会在时间聚合阶段重新混进来。证据链是 TBM、CIM、FDM 三件套和视频 VI-ReID 数据集对标。
story 套路：视频跨模态的关键不是每帧对齐，而是序列级保留身份、剥离模态。

9. `Discovering Multi-Frequency Embedding...`
创新类型：新机制。多频率嵌入，把 VI-ReID 从空间域扩展到频率域。
为什么能发：已有频率方法只做粗低频和高频划分，身份线索其实分散在多个频段。证据链是 HiLo-FM、FADE、CMSR、CMRR 消融，加 SYSU、RegDB、LLCM。
story 套路：跨模态身份线索不是单一频段，必须按频率结构挖出来。

**这批论文反推出的规律**
能投 B 类的核心不是“加一个模块”，而是先指出现有方法优化错了对象，例如局部配对不等于 AP、视角解耦会丢身份、文本描述不可靠、模态对齐会损判别性。然后把这个错误对象改成一个任务特定机制，再用干净消融证明每个机制确实解决了那个错误。

**候选方向**
1. Headline：Aerial-ground ReID 应该对齐“同一人体表面的可见身份线索”，而不是追求粗暴的视角不变。
挂的团队资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。
区别：最像 CVAF，但 CVAF 用 CLIP 文本 token 做视角一致语义过滤；我们用 SMPL 可见表面和投影关系定义哪些身体区域在航拍和地面视角中可比，避免文本语义和人工属性不可靠。
cheap kill-switch：在 CARGO 或 AG-ReID.v2 上用冻结 SOLIDER 特征，加 2D pose 或 SMPL 投影做可见表面池化。若零训练或小头训练不能明显改善跨视角正样本相似度和 mAP，直接停。

2. Headline：跨视角 ReID 的损失函数应直接优化“可见表面条件下的跨视角 AP”。
挂的团队资产：aerial-ground、pose 热图门控、SMPL、SOLIDER-Swin。
区别：最像 CAP 和 MFENet 的 CMSR，但它们解决可见红外模态排序；我们解决航拍和地面之间可见区域不对称导致的排序错误。正样本不是一律等价，而是按 3D 可见重叠和姿态可信度加权。
cheap kill-switch：只实现一个轻量损失头，保持 backbone 和训练协议不变，对比 ID 加 triplet。10 到 20 个 epoch 小跑如果没有超过正常波动，且 hard positive 排名没有改善，停。

3. Headline：航拍低清下的语言语义不是都可信，应由几何可见性决定哪些文本能蒸馏进 ReID 表征。
挂的团队资产：aerial-ground、SMPL、SOLIDER-Swin。
区别：最像 CSGN、DATE、DFLSG 和 CVAF，但这些方法主要从文本图像相关性或身份均值处理文本噪声。我们的切开点是视角和几何导致的语义可靠性，例如航拍中颜色、鞋子、背包可能不可见，身形和轮廓更稳定。
cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。

4. Headline：航拍到地面匹配中，身份线索会随视角和尺度在频段间迁移，频率选择应由人体几何控制。
挂的团队资产：aerial-ground、SMPL 3D 几何、SOLIDER-Swin。
区别：最像 MFENet，但 MFENet 是通用 VI-ReID 的多频率分支；我们的切开点是把频率可靠性绑定到视角、尺度和身体表面可见性，而不是固定低频高频或纯注意力学习。
cheap kill-switch：先不训练，直接做频段分析。把 CARGO 或 AG-ReID.v2 按视角差、尺度、可见身体区域分组，看不同频段的同身份和异身份间隔是否有稳定规律。没有规律就不做模块。
tokens used
91,380
已按要求只用 `pdftotext -l 3` 读了每篇前 3 页。下面是摘要和引言层面的反推，不代表全文细节。

**逐篇提取**
1. `Cross-modality average precision optimization...`
创新类型：新机制。把可见光和热红外 ReID 的优化目标从三元组局部配对改成跨模态可微 AP 排序。
为什么能发：指出 triplet 和 Smooth-AP 都没有直接优化跨模态检索排序。证据链是 CAPV、CAPT、动态模态对齐、全局局部特征逐项消融，加 SYSU-MM01 和 RegDB 对标。
story 套路：跨模态 ReID 不该只拉近难正样本，而要直接优化最终跨模态排序。

2. `CSGN...`
创新类型：新机制偏工程组合。CLIP 伪文本、RGB 和灰度双描述、ResNet 和 ViT 并行视觉引导、异构语义融合。
为什么能发：抓住换衣 ReID 中 CLIP 文本容易关注衣服颜色这个缺口。证据链是四个换衣数据集，对 MDG、VSS、HSF 做消融。
story 套路：用语言语义削弱衣服颜色捷径，让模型从视觉瓶颈里出来。

3. `CVAF...`
创新类型：问题重定义加新机制。把 aerial-ground ReID 说成“视角一致语义保留”，反对粗属性依赖和强行视角解耦。
为什么能发：AGPReID 里航拍低清、尺度小、视角极端，属性不可靠，过度解耦会丢身份线索。证据链是 CLIP 身份文本 token、文本一致性损失、语义过滤模块，多 AGPReID 数据集对标。
story 套路：跨视角不是把视角信息删掉，而是保留跨视角仍稳定的身份结构。

4. `CycleTrans...`
创新类型：新机制。用循环构造学习“中性但可区分”的可见红外特征。
为什么能发：现有模态对齐会牺牲判别性。证据链是 KCM、DMM、循环重构消融，加 SYSU-MM01 和 RegDB。
story 套路：跨模态特征要去模态化，但不能把细粒度身份语义一起抹掉。

5. `DATE...`
创新类型：新机制。把可学习文本嵌入和描述文本嵌入做非对称融合，描述文本只当辅助线索。
为什么能发：MLLM 自动描述有噪声，同步融合会污染视觉表示。证据链是两个 adapter、身份级描述均值、跨粒度训练，在 Market、Duke、MSMT 上验证。
story 套路：文本不是裁判，只能作为有噪声的辅助证据。

6. `Deep intelligent technique...`
创新类型：工程组合。Horned Lizard 优化加 GoogleNet 做特征选择和分类。
为什么能发：它包装的是噪声监控图像下的特征不足问题，但机制和证据都比较泛，报告指标也异常高。
story 套路：生物启发优化帮助选特征。对我们启发很弱，不建议沿这个方向学。

7. `Dependability Feature Learning...`
创新类型：新数据设定加新机制。无监督文本图像 ReID，既没有图文匹配关系，也没有身份标签，先生成并筛可靠样本。
为什么能发：真实场景下文本标注和身份标注都贵。证据链是多 MLLM 文本生成、可靠文本筛选、异常样本过滤、中心和实例两级三元组学习。
story 套路：先解决训练样本是否可信，再谈跨模态对齐。

8. `DIRL...`
创新类型：问题重定义加新机制。视频可见红外 ReID 不能只做帧级模态不变，还要做序列级交互和身份模态解耦。
为什么能发：视频序列里模态干扰会在时间聚合阶段重新混进来。证据链是 TBM、CIM、FDM 三件套和视频 VI-ReID 数据集对标。
story 套路：视频跨模态的关键不是每帧对齐，而是序列级保留身份、剥离模态。

9. `Discovering Multi-Frequency Embedding...`
创新类型：新机制。多频率嵌入，把 VI-ReID 从空间域扩展到频率域。
为什么能发：已有频率方法只做粗低频和高频划分，身份线索其实分散在多个频段。证据链是 HiLo-FM、FADE、CMSR、CMRR 消融，加 SYSU、RegDB、LLCM。
story 套路：跨模态身份线索不是单一频段，必须按频率结构挖出来。

**这批论文反推出的规律**
能投 B 类的核心不是“加一个模块”，而是先指出现有方法优化错了对象，例如局部配对不等于 AP、视角解耦会丢身份、文本描述不可靠、模态对齐会损判别性。然后把这个错误对象改成一个任务特定机制，再用干净消融证明每个机制确实解决了那个错误。

**候选方向**
1. Headline：Aerial-ground ReID 应该对齐“同一人体表面的可见身份线索”，而不是追求粗暴的视角不变。
挂的团队资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。
区别：最像 CVAF，但 CVAF 用 CLIP 文本 token 做视角一致语义过滤；我们用 SMPL 可见表面和投影关系定义哪些身体区域在航拍和地面视角中可比，避免文本语义和人工属性不可靠。
cheap kill-switch：在 CARGO 或 AG-ReID.v2 上用冻结 SOLIDER 特征，加 2D pose 或 SMPL 投影做可见表面池化。若零训练或小头训练不能明显改善跨视角正样本相似度和 mAP，直接停。

2. Headline：跨视角 ReID 的损失函数应直接优化“可见表面条件下的跨视角 AP”。
挂的团队资产：aerial-ground、pose 热图门控、SMPL、SOLIDER-Swin。
区别：最像 CAP 和 MFENet 的 CMSR，但它们解决可见红外模态排序；我们解决航拍和地面之间可见区域不对称导致的排序错误。正样本不是一律等价，而是按 3D 可见重叠和姿态可信度加权。
cheap kill-switch：只实现一个轻量损失头，保持 backbone 和训练协议不变，对比 ID 加 triplet。10 到 20 个 epoch 小跑如果没有超过正常波动，且 hard positive 排名没有改善，停。

3. Headline：航拍低清下的语言语义不是都可信，应由几何可见性决定哪些文本能蒸馏进 ReID 表征。
挂的团队资产：aerial-ground、SMPL、SOLIDER-Swin。
区别：最像 CSGN、DATE、DFLSG 和 CVAF，但这些方法主要从文本图像相关性或身份均值处理文本噪声。我们的切开点是视角和几何导致的语义可靠性，例如航拍中颜色、鞋子、背包可能不可见，身形和轮廓更稳定。
cheap kill-switch：抽一小批 aerial-ground 图，用 CLIP 或 MLLM 生成属性描述，统计同身份跨视角属性一致性是否能被视角、尺度、SMPL 可见性预测。预测不了就停；能预测，再做训练期蒸馏小实验。

4. Headline：航拍到地面匹配中，身份线索会随视角和尺度在频段间迁移，频率选择应由人体几何控制。
挂的团队资产：aerial-ground、SMPL 3D 几何、SOLIDER-Swin。
区别：最像 MFENet，但 MFENet 是通用 VI-ReID 的多频率分支；我们的切开点是把频率可靠性绑定到视角、尺度和身体表面可见性，而不是固定低频高频或纯注意力学习。
cheap kill-switch：先不训练，直接做频段分析。把 CARGO 或 AG-ReID.v2 按视角差、尺度、可见身体区域分组，看不同频段的同身份和异身份间隔是否有稳定规律。没有规律就不做模块。
