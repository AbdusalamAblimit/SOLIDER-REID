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
session id: 019ef835-a0db-7f52-9984-8312108140cd
--------
user
你是 ReID 创新方法论拆解员 12。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(12)==
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标文件能否读取，然后用 PDF 技能和 `pdftotext` 把每篇前 8 页的方法部分抽出来。之后只围绕“观察如何被包装成创新”拆，不会把重点放在复刻模块细节。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'STATUS.md' -g 'AGENTS.md' .. ../.. ../../.. ../../../.. 2>/dev/null | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 434ms:
../../../CLAUDE.md
../../../AGENTS.md
../../../GOAL.md
../../../experiments/STATUS.md
../../../../reid-paper/CLAUDE.md
../../../../reid-paper/AGENTS.md
../../../../reid-paper/GOAL.md
../../STATUS.md
../../../../thundera/AGENTS.md
../../../../SOLIDER-REID/CLAUDE.md
../../../../SOLIDER-REID/AGENTS.md
../../../../prcv/prcv-paper-writing/SOLIDER-REID/CLAUDE.md
../../../../prcv/prcv-paper-writing/SOLIDER-REID/AGENTS.md
../../../../reid-paper/experiments/STATUS.md
../../../../work/2606/CLAUDE.md
../../../../work/2606/reid-opl/CLAUDE.md
../../../../work/2606/reid-opl/AGENTS.md
../../../../browser-ext/thundera-github/AGENTS.md
../../../../prcv/SOLIDER-REID/CLAUDE.md
../../../../prcv/SOLIDER-REID/AGENTS.md
../../../../work/2606/reid-autolab/CLAUDE.md
../../../../work/2606/reid-autolab/GOAL.md
../../../../work/2606/reid-autolab/experiments/STATUS.md

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

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我已经确认当前目标仍是文献调研和方法方向构造，不涉及开实验。接下来我会只读本目录里的这些 PDF，优先从论文自己的方法论链条里抽“观察、改写问题、机制、证据、审稿人买账点”。
hook: PreToolUse
hook: PreToolUse
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
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
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
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
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
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf

codex
这批清单实际列出了 6 篇 PDF，而不是 12 篇。我会按列出的 6 篇逐篇拆完，不因为编号或数量不一致省略任何一篇。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209807

False Negatives Consensus Suppression for
Text-to-Image Person Re-identificatio
Ruigeng Zeng1,2,3 , Wentao Ma4,* , Qinglin Wang1,2,3* , Xinjun Mao3 and Jie Liu1,2,3
1

2

Laboratory of Digitizing Software for Frontier Equipment, National University of Defense Technology, Changsha, China
National Key Laboratory of Parallel and Distributed Computing, National University of Defense Technology, Changsha, China
3
College of Computer Science and Technology, National University of Defense Technology, Changsha, China
4
School of Information and Artificial Intelligence, Anhui Agricultural University, Hefei, China

Abstract—Text-Image Person Re-identification (TIReID) aims
to retrieve the relevant pedestrian images according to the
given textual query. Recent methods typically achieve this goal
through image-text contrastive learning, which assumes that
only paired images and texts from the same pedestrian are
considered positive samples. However, we observe that there
exist negative samples, termed false negatives, that are highly
semantically related to the anchor in practice. Training with
these false negatives may adversely affect feature representation
learning and semantic alignment between modalities. This work
proposed a false negative detection and suppression (FNDS)
method to mitigate their adverse impact. Our FNCD consists
of a False Negative Consensus Detection (FNCD) mechanism
and an Adaptive False Negative Suppression (AFNS) method.
FNCD combines dual-grained detection to consensually identify
potential false negatives, while AFNS assigns adaptive weights
to the false negative similarities for more robust suppression.
Extensive experiments conducted on three public benchmark
datasets demonstrate the effectiveness of the proposed method.
Index Terms—Text-image person Re-identification, false negative, cross-modal contrastive learning

I. I NTRODUCTION
Text-Image Person Re-identification (TIReID) [1], a subtask of Person Re-identification (ReID), aims to retrieve the
most semantically related pedestrian images from a large
candidate gallery based on the given text query. Due to its
practical relevance in the fields of public safety and smart
cities, TI-ReID has garnered increasing attention in recent
years. However, TIReID remains a challenging task as it
requires fine-grained feature representation of pedestrians’
complex semantic visual and textual information, as well as
accurate visual-textual alignment.
To tackle these challenges, most previous works adopt
Visual Semantic Embedding (VSE) methods to learn the
correspondence between the image and text modalities. These
methods [2]–[5] generally follow a common model structure: “image/text encoders + feature embedding”. In this
framework, image/text features are first extracted using the
respective encoders, and then these features are embedded
(model-specific) into a shared latent space for cross-modal
alignment. Image/text encoders typically use single-modality
* Corresponding authors(email: wtma@ahau.edu.cn, wangqinglin@
nudt.edu.cn).
Code is available at https://github.com/Ray-Zhen/FNDS.

Anchor: A man has his head bent down with arms at his sides. His
right leg is extended behind him. He wears a black, short-sleeve top,
blue shorts, ending below the knees, and dark sandals.

attract

Positive

repel

repel

repel

False Negative True Negative True Negative

Fig. 1: The illustration of false negatives. There exist negative
samples that share the same semantics with the anchor due to
the data noise. Repeling such false negative samples from the
anchor harms the representation learning.

networks initialized with pre-trained models (e.g., ViT [6]
on ImageNet for the image encoder and BERT [7] for the
text encoder) to facilitate cross-modal learning. The most
recent works use the Contrastive Language-Image Pre-training
(CLIP) model [8], pre-trained on large image-text datasets,
as image/text encoders to leverage multi-modal semantic correspondence, greatly enhancing retrieval performance over
single-modal methods.
Most TIReID methods employ image-text contrastive learning method to establish semantic correspondences between
image and text modalities. Specifically, considering the query
text as anchor, the paired person image is viewed as positive, while all other images in the mini-batch are treated
as negative samples. The optimization objective is to bring
the positive samples closer to the anchor while pushing the
negative samples farther away in the shared latent space.
Although image-text contrastive learning has demonstrated
impressive performance, these methods often overlook the
semantic relationships between image-text pairs of each individual. We observe that some negative samples share the
same semantic concept as the anchor, due to incorrect person
ID labeling and the semantic diversity of text descriptions,
where a single description may refer to multiple pedestrians.
A typical example is shown in Figure 1. During training, these
negative samples, defined as false negatives in this paper,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

are still treated as true negatives and pushed away from the
anchor, contradicting the optimization objective and adversely
impacting cross-modal representation learning and alignment.
As far as we know, there is litter work has been devoted
to studying the false negative problem in TIReID, and the
closest related work is in the task of image-text matching [9],
[10]. However, these works may be inefficient or ineffective in
TIReID because they rely solely on global image-text features
for false negative detection, neglecting fine-grained local cues,
where subtle differences in local details often exist between
pedestrians.
To address the false negative problem in TIReID, we
propose a false negative detection and suppression framework,
called FNDS, to identify potential false negative samples and
mitigate their adverse impact. Our FNDS consists of a False
Negative Consensus Detection (FNCD) mechanism and an
Adaptive False Negative Suppression (AFNS) method. FNCD
fuses dual-grained detection to consensually identify potential false negative samples, thereby providing more reliable
negative samples for robust contrastive learning. Specifically,
we leverage the similarity distribution differences between
false negatives and true negatives to screen out potential
false negatives. To enrich feature granularity, we propose a
dual-representation method that combines coarse-grained basic
global representation (BGR) with fine-grained token selection
representation (TSR) to enhance cross-modal correspondence
comprehensively. Based on this, we conduct consensus false
negative detection for more robust identification. The proposed
AFNS assigns adaptive weights, adjusted based on false negative confidence, to the similarities between the anchor and
detected false negative samples, enabling more robust false
negative suppression. Moreover, the Cross-modal Momentum
Contrastive (CM-MoC) module is introduced for a more
accurate estimation of false negative similarity distribution.
The main contribution can be summarized as follows: (1) We
reveal and investigate the inevitable false negative problem
in TIReID. We propose a FNDS framework to suppress the
adverse impact of false negatives through the False Negative Consensus Detection (FNCD) mechanism and Adaptive
False Negative Suppression (AFNS) method. (2) We introduce
Cross-modal Momentum Contrastive (CM-MoC) to expand
the training data in each training epoch, enabling a more accurate estimation of false negative similarities. (3) We conducted
extensive experiments on three widely used datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid. The comprehensive
results demonstrate that our method surpasses all current stateof-the-art approaches, confirming its effectiveness.
II. R ELATED WORK
A. Text-image Person Re-identification
TIReID, first introduced by [1], is a subtask of crossmodal retrieval with challenges in fine-grained alignment
due to intra- and inter-modal variations. According to the
alignment strategy, the existing approaches can be generally
classified into two categories: the global-matching methods

and local-matching methods. Global-matching methods focus on designing models or objective functions to learn
image-text correspondence within a shared latent space [1],
[11]. However, global-matching methods focus exclusively
on global-level feature representation, neglecting informative
local details, which hinders performance improvements. To
address this limitation, local-matching methods [2]–[4], [12]–
[14] have been proposed to capture fine-grained local crossmodal alignment between visual scenes and text descriptions.
Recently, CLIP [8], a landmark in visual-language pre-training
(VLP), has garnered remarkable success owing to its robust and comprehensive multi-modal representations. Consequently, numerous studies [12], [13] have integrated CLIP into
TIReID to improve cross-modal representation and alignment.
For instance, [12] introduced a CLIP-driven framework to
extract fine-grained visual information, while [13] utilized
both the visual and language encoders of CLIP to capture implicit fine-grained cross-modal relations. In this paper, we do
not aim to design elaborate cross-modal alignment strategies
or introduce powerful backbone networks. Instead, we focus
on addressing the inevitable and challenging false negative
problem in TIReID.
B. Learning with False Negative
Research on false negatives, a crucial issue in noisy data
tasks, has gained increasing attention in fields such as imagetext matching [9], [10], graph representation learning [15], [16]
and sound source localization [17]. To tackle the false negative
challenge, many approaches have been introduced, which can
generally be grouped into robust loss function methods [16],
[18] and sample selection methods [9], [10], [19]. The former
approaches focus on developing loss functions that are tolerant
to noise, aiming to reduce the negative impact of false negative
samples. In contrast, the sample selection approaches focus
on formulating effective techniques to identify false negative
samples and mitigate their impact. The methods mentioned
above have made considerable advancements in various tasks.
However, they are not specifically tailored for TIReID. Therefore, in this study, we propose a novel approach to mitigate
false negatives and tackle the false negative challenge in
TIReID.
III. T HE P ROPOSED M ETHOD
A. Feature Representation
Image-Text Feature Representation. Following the previous work [13], we adopt the CLIP backbone to extract imagetext features. For image feature representation, given an input
image Ii ∈ Rc×w×h , we adopt the visual encoder of CLIP
to extract token feature sequence fiv = {vgi , v1i , v2i , ..., vni v }
, with a total length of nv + 1. Where, vg is the image-level
v
global feature of [CLS] token, {vji }nj=1
is the patch-level local
features. For Text Feature Representation, given the input text
Ti , we obtain the text features fit = {tisos , ti1 , ti2 , ..., tint , tieos }
with textual encoder of CLIP. Where, nt denotes the text token
length, tisos and tieos are the specific token features for the
[SOS] and [EOS] tokens, respectively, tieos serve as text-level

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

negative
similarities

...
vg

FNDS

negative samples

true negatives

false negatives

tg

...

Text
Encoder

(a) The Model Framework

Token Selection

Momentum
Encoder

She is wearing a white
shirt and denim shorts
and black tennis shoes.

Momentum
Encoder

Token Selection

Image
Encoder

vtsr

GMM with TSR similarities
Component 1
Component 2
Mixture PDF

GMM with BGR similarities
Component 1
Component 2
Mixture PDF

−

ttsr

False Negative Consensus Detection
positive
+
+
−

−

+

+
−
−

ρ
+

+

−

−

true negative

−

false negative

similarity

Adaptive False Negative Suppression
(b) False Negative Detection and Suppression

Fig. 2: The overview of our method. (a) is the illustration
of the model framework, which consists of token selection
representation (TSR) module, False Negative Detection and
Suppression (FNCD) and Cross-modal Momentum Contrastive
(CM-MoC) module. (b) illustrates the core of FNDS, which
consists of False Negative Consensus Detection (FNCD)
mechanism and Adaptive False Negative Suppression (AFNS)
method.
t
global feature of tig , and {tij }nj=1
represent the word-level local

features.
Dual Representation. Most previous TIReID works [13],
[20], [21] only adopt vgi and tig token features as basic global
representation (BGR) for cross-modal alignment. However, using only BGR may overlook fine-grained intra-modal semantics, hindering detailed cross-modal alignment. To overcome
this, we aggregate local features from image patches and word
tokens for fine-grained feature representations.
Specifically, inspired by the previous works [12], [22], we
select and transform informative local features to generate finegrained token selection representation (TSR). In practice, take
the visual process for example, we first obtain the attention
map Avi ∈ R(1+nv )×(1+nv ) from the last transformer block
of the image encoder, which reflects the importance scores
between 1 + nv tokens. Then the correlation weight between
[CLS] token and local tokens avi = Avi [0, 1 :] ∈ Rnv are
used to select the top-K informative local token features fˆiv =
{vki v , vki v , ..., vki nv }, where knv = R × nv denotes the indices of
1
2
the selected local tokens and R is the token selection ratio. In
terms of textual procedure, we obtain the selected local text
features in a similar way as fˆit = {tikt , tikt , ..., tikt }. Finally,
n
1
2
the selected local visual and textual token features are linearly
transformed and aggregated via:
titsr = M axP ool(σ(BN (W1t fˆit ) + W2t fˆit ))
(1)
i
vtsr
= M axP ool(σ(BN (W1v fˆiv ) + W2v fˆiv ))
i
where vtsr
and titsr is the TSR for image and text,
M axP ool(·) is the max-pooling function, σ(·) is the ReLU
activation function, BN (·) is the batch normalization, and W
denotes the linear transformation parameter.

B. False Negative Consensus Detection
To alleviate the adverse impact of false negatives, the
primary challenge is to identify and remove the potential false

negative samples during training. Intuitively, false negative
samples are negative samples that exhibit high semantic similarity to positive samples, which can be treated as anomalies among negative samples. Building on this observation,
we exploit the differences in similarity distributions between
false negatives and true negatives to identify potential false
negatives. To this end, we employ a two-component Gaussian
Mixture Model (GMM) to fit the similarity distributions of
negative samples within the current mini-batch. Specifically,
we first compute the cosine similarity set S ∈ RB×B across
all image-text feature pairs {vi , ti }B
i=1 in a mini-batch:
B
S(vi , ti )|B
i=1 = {sim(vk , tl )}k,l=1

(2)

Where sim(vk , tl ) = vk tTl /||vk ||||tl ||, B is the batch size.
Then we collect the negative pair similarities, which are
B(1−B)
denoted as S − = {sim(vk , tl )}B
. At last,
k,l=1,k̸=l ∈ R
we fit the negative pair similarities by using the GMM to find
out the potential false negative samples:
p(s− | θ) =

2
X

βk ϕ(s− | k)

(3)

k=1

where βk is the mixture coefficient and ϕ(s− | k) is the
probability density of negative pair similarities of the k-th
component. We adopt the Expectation Maximization algorithm
to optimize the GMM. Subsequently, the posterior probability
−
p(k|s−
i ) = p(k)p(si |k) of i-th negative pair, as either a false
negative or true negative, can be computed, where k ∈ {0, 1},
k = 0 indicates a true negative with a lower similarity mean,
and k = 1 indicates a false negative with a higher similarity
mean. By setting a threshold δ for p(k = 1|s−
i ), the negative
pairs in the current mini-batch can be divided into true negative
and false negative subsets:
−
−
−
S f n = {s−
i |p(k = 1|si ) ≥ δ, ∀si ∈ S }
−
−
−
S tn = {s−
i |p(k = 0|si ) < δ, ∀si ∈ S }

(4)

where S f n denotes the false negative set, S tn denotes the
true negative set, and S = S tn ∪ S f n . Note that we adopt
both BGR and TSR for false negative identification, denoting
fn
fn
their respective false negative sets as Sbgr
and Stsr
. To obtain
a more reliable false negative set, we propose exploiting the
fn
fn
consistency between Sbgr
and Stsr
to identify their consensus
as the final false negative set. The process can be described
as follows:
tn
tn
Ŝ tn = Sbgr
∩ Stsr
fn
fn
Ŝ f n = Sbgr
∩ Stsr
u

S = S − (Ŝ

tn

(5)
tn

∪ Ŝ )

where, Ŝ tn is the final consensus true negative set, Ŝ f n is the
final consensus false negative set, S u is the uncertain set. For
the negative samples in uncertain set S u , we randomly labeled
them as true negative or false negative.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

C. Adaptive False Negative Suppression
After identifying potential false negatives, the challenge
shifts to minimizing their impact during training. However,
in the early stages of training, the backbone model is insufficiently trained, leading to unreliable image-text features. As a
result, the difference in similarity distributions between false
negatives and true negatives is not significant. This poses a
challenge, as the identified false negatives may be mixed with
true negative samples. A straightforward approach to eliminate
the detected false negative samples will lead to performance
degradation(as shown in Table III). To address this problem,
we propose an Adaptive False Negative Suppression (AFNS)
loss that adaptively adjusts the weights of identified false
negative similarities based on their false negative probabilities.
The loss function can be expressed as follows:
LAF N S =
−

B
X
i=1

−

B
X
i=1

s(vi , ti )
P
k∈Sif n ρi,k s(vi , tk ) (6)
j=1,j /
∈Sif n s(vi , tj ) +

log PB

s(ti , vi )
P
k∈S f n ρi,k s(ti , vk )
j=1,j /
∈S f n s(ti , vj ) +

log PB

i

i

where s(vi , ti ) = exp(sim(vi , ti )/τ ), Sif n is the set of
detected false negative samples, ρ(·) denotes the adaptive
weight function that is adaptively adjusted based on the false
negative probability, which is defined as follows:
ρi,k = exp(−p(k = 1|sim(ti , vk )))

(7)

where sim(ti , vk ) is the cosine similarity between the anchor
text ti and negative image vk . If the detected false negative
samples exhibit high similarity to the anchor, they are more
likely to be false negatives. Consequently, a small weight is
assigned to the similarity between the anchor and these false
negatives. Conversely, if the detected false negative samples
exhibit low similarity to the anchor, they are more likely to be
misidentified as false negatives, and a larger weight is assigned
to their similarity.
D. With Cross-modal Momentum Contrastive
To increase the availability of negative samples in each
iteration for precise GMM fitting, we propose the Cross-modal
Momentum Contrastive (CM-MoC) module, which leverages
dynamic queues to maintain a larger pool of negatives. CMMoC includes four feature queues for global and token selection features and two momentum-updated encoders for image
and text modalities. Technically, we maintain two distinct
types of feature queues to store recent global and token
i
selection image-text features (vgi , tig ) and (vtsr
, titsr ) from
momentum-updated encoders, progressively replacing older
features. These queued features are then leveraged to compute
cross-modal similarity computation with current mini-batch
features for loss calculation. The queue sizes are hyperparameters set larger than the mini-batch size. the momentum-updated

encoders are updated using an exponential moving average
(EMA):
Θm = mΘm + (1 − m)Θ
(8)
where Θm are the momentum encoder parameters, Θ are the
original encoder parameters updated via back-propagation.
E. Training and Inference
During the training stage, we utilize both BGR (vgi , tig ) and
i
TSR (vtsr
, titsr ) to achieve more reliable and comprehensive
false negative suppression. The final loss function is expressed
as follows:
NS
NS
LAF N S = LAF
+ LAF
(9)
tsr
bgr
NS
NS
where LAF
and LAF
are the AFNS losses computed
tsr
bgr
using Equation 6, with BGR and TSR, respectively. During
inference, we compute the average similarities of both features,
i
i.e., s = (sim(vgi , tig ) + sim(vtsr
, titsr ))/2.

IV. E XPERIMENTS
A. Settings
1) Datasets: We use the CUHK-PEDES [1], ICFGPEDES [4], and RSTPReid [3] datasets to train and evaluate
our model in the experiments.
2) Evaluation Metrics: We adopt extensively used metrics
to measure our model: Recall at K (R@K) and mean Average
Precision (mAP).
3) Implementation Details: We adopt the Adam optimizer
with an initial learning rate 1×10−5 for the pretraind backbone
and 1 × 10−3 for another randomly initialized model. The
model is trained for 25 epochs with cosine learning rate decay
and a 5 epoch warm-up, where the learning rate is linearly
increased from 1 × 10−6 to 1 × 10−5 . We set the batch
size set to 64, the temperature parameter τ is set to 0.03,
the token selection ratio R is set to 0.3, the false negative
probability threshold δ is 0.90, and the feature queue size is
set to 6144, 2048 and 512 for CUHK-PEDES, ICFG-PEDES
and RSTPReid, respectively.
B. Comparison with State-of-the-Art Methods
Table I presents the comparison between our approach
and the SoTAs on three benchmark datasets. We have the
following observations: (1) On the CUHK-PEDES dataset,
our proposed method consistently surpasses all current SoTA
approaches in R@1 and mAP metrics. Notably, our method
achieves performance improvements of +6.05% and +2.69%
on the R@1 metric compared to the strong baseline methods
CFine [12] and IRRA [13], respectively. In addition, when
compared to the recent method DCEL [27], which aims to
mitigate the negative impact of noisy data pairs, our method
also achieves an impressive improvement of +1.05% on the
R@1 metric. (2) For ICFG-PEDES dataset, our proposed
method significantly outperforms the most recent SoTA approaches. Specifically, our method outperforms the strong
baseline methods CFine [12] and IRRA [13] by +6.35%
and +3.72%, respectively, on the R@1 metric. Compared to
the best baseline method MUM [21], our proposed method

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

TABLE I: Performance comparisons with SoTA methods.
Methods

Ref

ViTAA [23]
DSSL [3]
SSAN [4]
TIPCB [2]
LGUR [24]
IVT [25]
UniPT [26]
CFine [12]
IRRA [13]
TBPS-CLIP [20]
MUM [21]
DCEL [27]
FNDS (Ours)

ECCV’20
MM’21
arXiv21
Neuro’22
MM’22
ECCV’22
ICCV’23
TIP’23
CVPR’23
AAAI’24
AAAI’24
MM’23
-

R@1
54.92
59.98
61.37
64.26
65.25
65.59
68.50
69.57
73.38
73.54
74.25
75.02
76.07

CUHK-PEDES
R@5 R@10
75.18 82.90
80.41 87.56
80.15 86.73
83.19 89.10
83.12 89.00
83.11 89.21
84.67 90.38
85.93 91.15
89.93 93.71
88.19 92.35
89.83 93.58
90.89 94.52
89.98 94.27

mAP
51.60
66.13
65.38
66.15
69.60

R@1
50.98
54.23
54.96
59.02
56.04
60.09
60.83
63.46
61.95
65.62
64.88
67.18

ICFG-PEDES
R@5 R@10
68.79 75.78
72.63 79.53
74.72 81.89
75.32 81.56
73.60 80.22
76.19 82.46
76.55 82.42
80.25 85.82
83.55 88.75
80.54 85.83
81.34 86.72
81.37 86.32

mAP
38.06
48.26
38.78
39.40

R@1
39.05
43.50
46.70
51.85
50.55
60.20
61.95
63.40
61.35
64.85

RSTPReid
R@5 R@10
62.60 73.95
67.80 77.15
70.00 78.80
74.85 82.85
72.50 81.60
81.30 88.20
83.55 88.75
83.30 90.30
83.95 90.45
83.95 89.70

mAP
47.17
48.26
49.28
50.47

achieves impressive performance improvements in the R@1,
R@5, and R@10 metrics by +1.56%, +0.83%, and +0.49%, respectively. (3) For ICFG-PEDES dataset, our proposed method
still outperforms all existing SoTA methods on RSTPReid.
We achieve an R@1 accuracy of 64.85%, outperforming the
most recent baseline method, MUM [21], by +1.45% in R@1
accuracy.

modules into the model. However, a performance gap persists
when compared to the full model. (3) The comparison No.
3 vs. No. 4 effectively demonstrates the effectiveness of
consensus false negative detection and suppression.
Effect of false negative suppression strategies. To validate the effectiveness of our proposed FNDS, we performed
ablation studies comparing it with the commonly used FNE
and FNA methods from [19], [28]. Experimental results in
TABLE II: Ablation studies on different components of our Table III reveal the following insights: 1) Our FNDS mehtod
method. We present the results for the R@1 metric.
outperforms FNE and FNA methods on all datasets, demonstrating the effectiveness of adaptive weights in mitigating the
Components
No. AF N S AF N S
CUHK-PEDES ICFG-PEDES RSTPReid
Lbgr
Ltsr
CM-MoC
impact of misclassified false negatives. 2) The FNE method
0
70.71
60.17
57.65
outperforms FNA, likely due to FNA’s sensitivity to the quality
1
✓
✓
73.99
64.26
62.75
of detected false negatives, performing well only with higher
✓
72.48
63.69
58.60
2
reliability.
3
✓
✓
72.58
63.771
59.20
4
5

✓

✓
✓

✓
✓

62.62
76.07

52.22
67.18

48.85
64.85
76.5

Method

65.36
74.61
67.18

63.80
62.40
64.85

C. Ablation Study
Effect of each component in FNDS. To investigate the
effects and contributions of each proposed component, we conducted ablation studies by comparing different combinations
of AFNS components. The results are presented in Table II.
The baseline model shown in No. 0 refers to the model
initialized with CLIP-ViT-B/16 and fine-tuned with the imagetext contrastive loss. We can observe from the experimental
results that: (1) The comparison No.0 vs. No.5 shows that
when incorporating the AFNS loss and CM-MoC module
into the baseline model, the model retrieval performance
achieves significant improvement, which highlights the overall
effectiveness of our approach. (2) The comparisons No.0 vs.
No.1, and No. 0 vs. No. 2 reveals the significant improvements
achieved by solely integrating the AFNS loss and CM-MoC

75.5
75.0
74.5
74.0

74.5

92
81
44
61
96
40

75.57
63.75
76.07

48
20

RSTP-REID
24
10

ICFG-PEDES

75.0

2
51

FNE
FNA
AFNS (Ours)

CUHK-PEDES

76.0

75.5

R@1 (%)

TABLE III: Ablation studies on different false negative suppression strategies. We present the results for the R@1 metric.

R@1 (%)

76.0

73.5

0.70 0.75 0.80 0.85 0.90 0.95

momentum queue size

(a)

(b)

Fig. 3: Parametric analysis of the hyper-parameters of momentum queue size and δ on the CUHK-PEDS dataset.
D. Parametric Analysis
In this section we analyze the impact of hyperparameters
δ (false negative probability threshold) and momentum queue
size on the CUHK-PEDES dataset. We can observe from the
Figure 3 that: (1) A larger queue size (e.g., 6144) enhances
performance by providing FNDS with more negatives samples
during training. However, excessively large queues can degrade performance, likely due to reduced feature consistency
caused by retaining too many outdated features, which may
mislead model learning. (2) When δ is set either too small
or too large, the model fails to achieve optimal retrieval
performance. Specifically, when δ is too large, most negative

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

(a) Query: A man
wearing a white shirt
with a picture, a pair
of colorful shorts
and a pair of yellow
sandals.

Positive

FN Prob: 96% FN Prob: 1% FN Prob: 1%

(b) Query: He is
wearing a white shirt
with a black backpack
and dark gray cargo
shorts with brown
sandals.

Positive

FN Prob: 96%

FN Prob: 96% FN Prob: 4%

Fig. 4: Illustration of the detected false negative samples on
the CUHK-PEDS dataset. The false negative probability are
denoted in the top of image.

samples are classified as true negatives, leaving the influence
of false negatives unresolved. Conversely, when δ is too small,
numerous true negatives are misclassified as false negatives
and are not sufficiently separated from the anchor, adversely
affecting the training process.
E. Qualitative Results
To better understand the effectiveness of our method, we
visualize the detected false negatives samples in Figure 4. We
can observe that some negative samples with high probability
indeed share the same semantics as the anchor and are
classified as false negatives, while those with low probability
share different semantics from the anchor and are classified as
true negatives.
V. C ONCLUSION
In this paper, we make the first attempt to explore the false
negatives problem in TIReID. To handle these tricky challenges, we proposed an effective FNCD framework to identify
and suppress the impact of false negatives. Additionally, a
CM-MoC module is introduced to enhance data enrichment
for more accurate similarity distribution estimation. Extensive
experiment results demonstrated the superiority and robustness
of our proposal.
ACKNOWLEDGMENT
This work was supported in part by the National Key
Research and Development Program of China under Grant
2023YFA1011704 and 2021YFB0300101, in part by the National Natural Science Foundation of China under Grant
62302144, 62303018, 31771679, 62306008, and 62301006.
R EFERENCES
[1] Shuang Li, Tong Xiao, et al., “Person search with natural language
description,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp.
1970–1979.
[2] Yuhao Chen, Guoqing Zhang, et al., “Tipcb: A simple but effective
part-based convolutional baseline for text-based person search,” Neurocomputing, vol. 494, pp. 171–181, 2022.
[3] Aichun Zhu, Zijie Wang, et al., “Dssl: Deep surroundings-person
separation learning for text-based person retrieval,” in Proceedings of
the ACM International Conference on Multimedia, 2021, pp. 209–217.
[4] Zefeng Ding, Changxing Ding, et al., “Semantically self-aligned network
for text-to-image part-aware person re-identification,” arXiv preprint
arXiv:2107.12666, 2021.

[5] Chenyang Gao, Guanyu Cai, et al., “Contextual non-local alignment over
full-scale representation for text-based person search,” arXiv preprint
arXiv:2101.03036, 2021.
[6] Alexey Dosovitskiy, Lucas Beyer, et al., “An image is worth 16x16
words: Transformers for image recognition at scale,” arXiv preprint
arXiv:2010.11929, 2020.
[7] Jacob Devlin, Ming-Wei Chang, et al., “Bert: Pre-training of deep
bidirectional transformers for language understanding,” arXiv preprint
arXiv:1810.04805, 2018.
[8] Alec Radford, , et al., “Learning transferable visual models from natural
language supervision,” in Proceedings of the International Conference
on Machine Learning, 2021, pp. 8748–8763.
[9] Zheng Li, Caili Guo, et al., “Integrating language guidance into imagetext matching for correcting false negatives,” IEEE Transactions on
Multimedia, vol. 26, pp. 103–116, 2023.
[10] Haoxuan Li, Yi Bin, et al., “Your negative may not be true negative:
Boosting image-text matching with false negative elimination,” in
Proceedings of the ACM International Conference on Multimedia, 2023,
pp. 924–934.
[11] Ying Zhang and Huchuan Lu, “Deep cross-modal projection learning
for image-text matching,” in Proceedings of the European Conference
on Computer Vision, 2018, pp. 686–701.
[12] Shuanglin Yan, Neng Dong, et al., “Clip-driven fine-grained text-image
person re-identification,” IEEE Transactions on Image Processing, 2023.
[13] Ding Jiang and Mang Ye, “Cross-modal implicit relation reasoning and
aligning for text-to-image person retrieval,” in IEEE Conf. Comput. Vis.
Pattern Recog., 2023, pp. 2787–2797.
[14] Xinyi Wu, Wentao Ma, et al., “Text-based occluded person reidentification via multi-granularity contrastive consistency learning,” in
AAAI, 2024.
[15] Han Zhao, Xu Yang, et al., “Graph debiased contrastive learning with
joint representation clustering.,” in Proceedings of the International
Joint Conference on Artificial Intelligence, 2021, pp. 3434–3440.
[16] Xiao Qin, Nasrullah Sheikh, et al., “Relation-aware graph attention
model with adaptive self-adversarial training,” in Proceedings of the
AAAI Conf. Artif. Intell., 2021, vol. 35, pp. 9368–9376.
[17] Weixuan Sun, Jiayi Zhang, et al., “Learning audio-visual source
localization via false negative aware contrastive learning,” in IEEE Conf.
Comput. Vis. Pattern Recog., 2023, pp. 6420–6429.
[18] Ching-Yao Chuang, Joshua Robinson, et al., “Debiased contrastive learning,” Proceedings of the Advances in Neural Information Processing
Systems, vol. 33, pp. 8765–8775, 2020.
[19] Tsai-Shien Chen, Wei-Chih Hung, et al., “Incremental false negative
detection for contrastive learning,” arXiv preprint arXiv:2106.03719,
2021.
[20] Min Cao, Yang Bai, et al., “An empirical study of clip for text-based
person search,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
vol. 38, pp. 465–473.
[21] Zhiwei Zhao, Bin Liu, Yan Lu, et al., “Unifying multi-modal uncertainty modeling and semantic alignment for text-to-image person reidentification,” in Proceedings of the AAAI Conf. Artif. Intell., 2024,
vol. 38, pp. 7534–7542.
[22] Yang Qin, Yingke Chen, et al., “Noisy-correspondence learning for textto-image person re-identification,” in IEEE Conf. Comput. Vis. Pattern
Recog., 2024, pp. 27197–27206.
[23] Zhe Wang, Zhiyuan Fang, et al., “Vitaa: Visual-textual attributes
alignment in person search by natural language,” in Proceedings of
the European Conference on Computer Vision, 2020, pp. 402–420.
[24] Zhiyin Shao, Xinyu Zhang, et al., “Learning granularity-unified representations for text-to-image person re-identification,” in Proceedings of
the ACM International Conference on Multimedia, 2022, pp. 5566–5574.
[25] Xiujun Shu, Wei Wen, et al., “See finer, see more: Implicit modality
alignment for text-based person retrieval,” in Proceedings of the
European Conference on Computer Vision, 2022, pp. 624–641.
[26] Zhiyin Shao, Xinyu Zhang, et al., “Unified pre-training with pseudo
texts for text-to-image person re-identification,” in Proceedings of the
IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 11174–11184.
[27] Shenshen Li, Xing Xu, Yang Yang, et al., “Dcel: Deep cross-modal
evidential learning for text-based person retrieval,” in Proceedings of
the ACM International Conference on Multimedia, 2023, pp. 6292–6300.
[28] Tri Huynh, Simon Kornblith, et al., “Boosting contrastive self-supervised
learning with false negative cancellation,” in Proceedings of the IEEE
Winter Conference on Applications of Computer Vision, 2022, pp. 2785–
2795.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Exploring Part-Informed Visual-Language Learning
for Person Re-Identification
Yin Lin1,2 , Yehansen Chen2 , Baocai Yin2 , Jinshui Hu2 , Bing Yin2 , Cong Liu2 , Zengfu Wang1 *
1 University of Science and Technology of China, Hefei, China
2 iFLYTEK Research, Hefei, China

arXiv:2308.02738v2 [cs.CV] 21 Mar 2025

lin5875@mail.ustc.edu.cn; zfwang@ustc.edu.cn;
{yinlin, yhschen, bcyin, jshu, bingyin, congliu2}@iflytek.com

Abstract—Recently, visual-language learning (VLL) has shown
great potential in enhancing visual-based person re-identification
(ReID). Existing VLL-based ReID methods typically focus on
image-text feature alignment at the whole-body level, while
neglecting supervision on fine-grained part features, thus lacking
constraints for local feature semantic consistency. To this end,
we propose Part-Informed Visual-language Learning (π-VL) to
enhance fine-grained visual features with part-informed language
supervisions for ReID tasks. Specifically, π-VL introduces a
human parsing-guided prompt tuning strategy and a hierarchical
visual-language alignment paradigm to ensure within-part feature semantic consistency. The former combines both identity
labels and human parsing maps to constitute pixel-level text
prompts, and the latter fuses multi-scale visual features with
a light-weight auxiliary head to perform fine-grained image-text
alignment. As a plug-and-play and inference-free solution, our
π-VL achieves performance comparable to or better than stateof-the-art methods on four commonly used ReID benchmarks.
Notably, it reports 91.0% Rank-1 and 76.9% mAP on the
challenging MSMT17 database, without bells and whistles.
Index Terms—Person re-identification, Visual-language learning, Fine-grained image-text alignment

I. I NTRODUCTION
Person re-identification (ReID) refers to the task of retrieving the query person-of-interest from large-scale gallery
databases captured by non-overlapping camera views [1].
Owing to its practical importance for intelligent video surveillance, ReID has gained ever-growing attention from both
academia and industry in recent years [2]–[4].
As appearance biometrics serve as the most fundamental
and well-studied cues for identity recognition [2], [5]–[7],
appearance-based ReID has achieved considerable success
across a wide range of applications. However, human body
semantics are not readily apparent in raw pixels, making it
challenging to learn semantic information under the single
supervision of one-hot or pair-wise identity labels [8].
Inspired by the recent success of visual-language models
[9], [10], CLIP-ReID [8] is one of the pioneer attempts
that leverages natural texts to specify visual concepts beyond
appearance. By tuning identity-specific text prompts [11],
it uses text representations generated by a powerful text
encoder [9] to deliver the image encoder a broader source
∗ Corresponding author

Learnable Identity-Specific Prompts

A photo of a [X]1[X]2[X]3[X]4 person

Text
Encoder

Text Embeddings
Inner Product

Image
Encoder

Identity Loss

Image Embeddings

(A) CLIP-ReID based on Global Image-Text Alignment
Learnable Identity-Specific Part Prompts
Head

Bag

…

Shoes

A photo of a [X]1[X]2[X]3[X]4 person’s head

Human
Parsing

Text
Encoder
Label
Guiding

Text Feature Map

MSE Loss

Image
Encoder
Visual Feature Map

(B) Our Part-Informed Visual-Language Learning

Fig. 1. Comparison of CLIP-ReID [8] and our part-informed visual-language
learning (π-VL) framework. (a) CLIP-ReID based on global image-text
alignment. (b) Our π-VL based on pixel-level image-text alignment.

of supervisions, leading to more discriminative global features. However, naively porting ideas from global image-text
alignment may not suffice for ReID. Several studies [3], [12]
have demonstrated that some non-salient details can be easily
overwhelmed, raising the within-part semantic inconsistency
issue (see Fig.2). And they also revealed that introducing partinformed identity supervisions is a promising solution to this
issue [3]. This motivates us to ask: Is learning fine-grained
body semantics as easy as global image-text alignment in ReID
task? An obstacle to addressing this issue lies in the ambiguous
boundaries between different parts of the human body. While
the human parsing task [13], [14] has effectively tackled
this problem, it introduces a new issue: supervision conflict.
Human parsing distinguishes identity-agnostic body part semantics, whereas ReID requires identity-specific discriminative
cues. This conflict can lead to reduced feature diversity and a
confused decision boundary for identity recognition.
In this paper, we address the above problems by introducing a Part-Informed Visual-Language learning framework,
termed π-VL, for person ReID tasks. Unlike existing works
that apply parsing maps for background elimination or body

alignment, we propose to construct pixel-level text prompts via
human parsing, and perform per-pixel image-text alignment to
enhance visual features. To alleviate the supervision conflict
problem, we combine both global-level identity labels and
pixel-level parsing semantics for contrastive prompt tuning,
leading to more discriminative part text embeddings. Furthermore, considering the hierarchical nature of visual backbones
[15], we propose a light-weight auxiliary head to fuse multistage visual features and design a parsing confidence weighted
alignment loss for robust semantic enhancement. It is worth
noting that our π-VL is a plug-and-play and loss functionbased solution, it is highly compatible with existing ReID
models. Experimental results on both CNN and ViT-based
backbones suggest that π-VL has the potential to be used as
a universal front-end capable of handling various model architectures. It achieves highly competitive results, i.e., 91.0%
Rank-1 and 76.9% mAP, on the MSMT17 benchmark, and
shows consistent improvements over mainstream person ReID
databases. Our contributions are summarized as follows:
• We propose a part-informed visual-language learning
framework, named π-VL, for person ReID. To our best
knowledge, this is one of the first attempts to introduce
fine-grained visual-language learning for ReID tasks.
• We present an identity-aware part-informed prompt tuning strategy based on human parsing. With this strategy,
we can generate pixel-level text prompts based on both
identity labels and parsing maps, strengthening the visual
encoder to spot more discriminative features.
• We design a novel fine-grained alignment mechanism for
ReID tasks. It integrates confidence scores from human
parsing to weight the alignment loss, leading to a more
semantically rich feature space for person image retrieval.
• Extensive experiments on mainstream ReID benchmarks
not only demonstrate the superior performance of the
proposed method, but also validate its generalization
ability to various visual encoders.
II. R ELATED W ORK
A. Appearance-based Person ReID
Appearance-based ReID aims to match a target pedestrian
across disjoint visible camera views at varying places and
times. It is challenging to learn suitable feature representations
robust enough to withstand large intra-class variations of
illumination, poses, and background clutter [2], [16].
Nowadays, deep learning methods show powerful capacity
of automatically extracting features from large-scale image
datasets and have achieved state-of-the-art results on RGBbased person ReID tasks [16]. Building on various sophisticated CNN architectures, deep ReID models are doing exceptionally well on visual matching by learning robust crosscamera feature representations and optimal distance metrics in
an end-to-end manner [1], [2]. To learn more discriminative
features, part information and contextual information are also
exploited in recent works [16], [17]. For example, methods
like PCB [3] and MGN [12] utilize hand-crafted partitioning

to split feature maps into grid cells or horizontal stripes for
local feature learning. Another line of researches adopt off-theshelf pose estimation or attention module [17] to extract the
human part aligned features. Although these approaches have
reported encouraging performance, the learned part features
still lack high-level semantics under the single supervision of
discrete identity labels [8].
B. Visual-language Pre-training
Over the past years, the emergence of visual-language pretraining models has led to substantial improvements to many
downstream tasks [9], [10]. Based on the idea of contrastive
image-text alignment, CLIP exploits a two-directional InfoNCE loss [9] to pre-train a pair of image and text encoders, leading to semantic meaningful visual representations
in harmony with manually-designed text prompts. To our best
knowledge, CLIP-ReID [8] is the first milestone that deals
with ReID tasks based on CLIP. By tuning identity-specific
text prompts [11], it uses text representations generated by a
powerful text encoder [9] to distill the image encoder, leading
to more discriminative global features. However, the local
part features still lack meaningful semantics under the only
supervision of global identity text embeddings.
III. M ETHODOLOGY
A. Preliminaries: Overview of CLIP-ReID
CLIP-ReID [8] is the first milestone that applies visuallanguage pre-training models to appearance-based ReID. As
one-hot identity labels used by ReID are meaningless to
construct high-quality text prompts, it proposes a two-stage
training procedure for visual-language learning.
The first training stage aims to optimize identityspecific text tokens with CLIP-style supervisions. By passing a pre-designed text description TD = ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person.’ and the corresponding person
image Ii through a frozen text encoder T (·) and a frozen
image encoder I(·) respectively, a text embedding Tyi and an
image embedding Vyi could be obtained, where [X] denotes
a learnable text token with the word embedding dimension,
M is the number of learnable text tokens, and yi indicates the person identity label. Then, CLIP-style contrastive
learning losses Li2t and Lt2i [9] are computed to optimize
[X]1 [X]2 [X]3 ...[X]M :
Lstage1 = Li2t + Lt2i .

(1)

In the second training stage, the learned identity-specific text
embeddings are treated as a classifier, and the image encoder
I(·) is fully optimized under the supervision of identity loss
Lid with label smoothing and triplet loss Ltri [2]:
Li2tce =

N
X

exp (Vi · Tyk )
−qk log PN
ya =1 exp (Vi · Tya )
k=1

Lstage2 = Lid + Ltri + Li2tce ,

(2)
(3)

where qk represents the soft label in the target distribution, N
is the number of identities, and i denotes the image index.

Head
Body
Shoes
(a) CLIP-ReID

Head
Body
Shoes
(b) Ours

But unlike [8], we reformulate the identity text prompt
at the pixel level using parsing maps, thereby generating fine-grained text prompts Tipart , such as, ‘A photo
of a [X]1 [X]2 [X]3 ...[X]M person’s head.’, ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person’s shoes.’. Then we can obtain
by passing fine-grained
a fine-grained text embedding tpart
i
text prompts Tipart through the tokenizer T :

Fig. 2. Illustration of within-part semantic inconsistency. Colors indicate
different body parts, symbols denote human identities, and the red dashed
line represents the decision boundary for identity recognition.

B. The Within-part Semantic Inconsistency Issue
CLIP-ReID is simple and effective, yet to be improved.
Since the natural language supervision is limited to the
whole-body scale (Eq.(2)), some non-salient or infrequent
part features can be easily overwhelmed, and still lack highlevel semantics [4]. As shown in Fig.2, we use t-SNE [18]
to visualize the pixel-level feature distributions produced by
CLIP-ReID, and adopt human parsing [13] to assign semantic
labels for each pixel. Here, colors indicate different body parts,
while symbols denote human identities.
We observe that although the decision boundary of identities
(the red dashed line) is generally clear, features of different
body parts are hard to distinguish. Furthermore, for several
confusing identities (e.g., identities denoted as crosses and
circles), the classification boundary of their part features is
even more difficult to be recognized. We term this issue as
within-part semantic inconsistency, which directly hinders the
performance of person retrieval.
C. Part-Informed Prompt Tuning
To address the issue of within-part semantic inconsistency,
an intuitive approach is to aggregate part features that share
the same semantics while separating those that are irrelevant.
This intuition, however, further raises two questions: (1) How
to identify the semantics of fine-grained part features? (2) How
to design the supervision signal for part distinction?
The first question has already been answered by state-of-theart human parsing models, which are robust to the ambiguous
boundaries between different body parts. Thus, we employ a
human parsing model H [13] to generate a pixel-level parsing
map P for person image Ii . Specifically, we follow the setup
from [13] and classify each pixel into N (N=20) semantic
categories, including ‘Background’, ‘Hat’, ‘Hair’, etc (see the
appendix for details). This allows us to generate per-pixel text
prompts based on the semantic labels and CLIP text encoder.
However, human parsing inherently introduces a new obstacle in addressing the second question. That is, human parsing only distinguishes identity-agnostic body part semantics,
whereas ReID requires learning identity-specific discriminative
cues. This conflict can suppress the diversity and discriminability of ReID features to some extent, leading to inferior
performance. Inspired by [11], we propose a part-informed
prompt tuning strategy to solve the supervision conflict issue.
As illustrated in Fig.3(a), similar to [8], we first learn identityspecific tokens with the text prompt Ti , i.e., ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person’, through optimizing Eq.(1).

tpart
= T (Tipart ).
i

(4)

Then, we align the spatial resolution of visual feature maps
and parsing map via downsampling, and rearrange the finegrained text embedding based on the spital arrangement of
parsing maps, leading to a ‘text embedding map’ (see appendix
for details), i.e.,
) tfi ull ∈ RH×W ,
tfi ull = rearrange(tpart
i

(5)

Next, we propose to learn our part-informed text prompts with
pixel-level dense contrastive learning. Specifically, let vif ull
denote the visual feature map extracted by the visual encoder,
we treat pixel-wise text embedding tji ∈ tfi ull and visual
embeddings vij ∈ vif ull of the j-th semantic label of the ij
th person (tji , vi+
), (vij , tji+ ) as positive pairs, while counting
j
the others as negative pairs (tji , vi−
), (vij , tji− ):

N
j
exp tji · vi+
/τ
1 X
log
,
 P
j
j
N j=1
/τ
exp tji · vi+
+ vj exp tji · vi−
i−

N
(6)
X
exp vij · tji+ /τ
Lpart
=
log

,
P
i2t
j
j
j
j
exp vi · ti+ + tj exp vi · ti− /τ
j=1

Lpart
t2i =

i−

part
Lpart = Lpart
t2i + Li2t ,

where τ is the temperature coefficient of the InfoNCE loss.
During the prompt tuning process, only the learnable text
tokens [X]1 [X]2 [X]3 ...[X]M are optimized, while the image
and text encoders are frozen (see the appendix for details).
Therefore, the loss function of the first training stage can
be defined as:
Lstage1′ = Lstage1 + Lpart ,

(7)

D. Part-Informed Visual-Language ReID
With part-informed prompt tuning, we are able to generate
identity-specific text embeddings with discriminative body
semantics. However, for image-text alignment, two issues
remain: 1) Which resolution of visual features is suitable for
supervision by language signals and 2) How to make the
alignment process robust against inevitable noises generated
by off-the-shelf parsing models?
For the first issue, downsampling operations can easily lead
to information loss, causing supervision signals to become inaccurate for small-scale features. Therefore, directly imposing
language supervision on all intermediate visual feature maps
is not advisable. To address this, we introduce an auxiliary
head that fuses multi-scale visual features to a relatively higher
resolution. This allows both low- and high-level visual features
to directly receive gradients from language guidance, thereby

Part-Informed Prompts

Part-Informed Prompts
…
Shoes
Head
Bag
A photo of a [X]1[X]2[X]3[X]4 person’s head

Text
Encoder

…
Shoes
Head
Bag
A photo of a [X]1[X]2[X]3[X]4 person’s head

Label
Guiding Text Feature Map

Human
Parsing

Weight-Sharing

Weight-Sharing

Label Text Feature Map
Guiding

Human
Parsing
Visual Feature Map

Image
Encoder

Text
Encoder

Visual Feature Map

Image
Encoder

Image Embeddings

Identity-Specific Prompts

Image Embeddings

Identity-Specific Prompts

A photo of a [X]1[X]2[X]3[X]4 person

Text
Encoder

Text Embeddings

A photo of a [X]1[X]2[X]3[X]4 person

Inner Product
Text
Encoder

Text Embeddings

(b) Stage 2: Part-Informed Visual-language ReID

(a) Stage 1: Part-Informed Prompt Tuning

Fig. 3. The proposed π-VL framework. To solve the within-part semantic inconsistency issue (Section III-B), it first learns identity-specific and part-informed
text prompts in a coarse-to-fine manner (Section III-C). Then it leverages a hierarchical fusion-based alignment strategy (Section III-D) to perform fine-grained
image-text alignment between part-informed text embeddings and multi-scale visual features.

During inference, we directly use the global features of the
visual encoder to conduct person retrieval (Fig.(4)).

Training
Person
Images

Backbone

Visual
Embeddings

Text
Embeddings

Text
Encoder
Shared

HFA

Visual
Feature Maps

Text
Feature Maps

Text
Encoder

IV. E XPERIMENTS

Parsing
prompts

A. Datasets and Evaluation Protocols

e.g., A photo of a
xxxx person’s shoes.

Inference
Person
Images

Identity
prompts
e.g., A photo of a
xxxx person.

Backbone

Visual
Embeddings

L2
Normalization

Person Retrieval

Fig. 4. Illustration of the hierarchical image-text alignment strategy. We
propose to fuse multi-scale features for image-text alignment.

creating a more semantically meaningful feature space for
person re-identification.
Figure 4 illustrates the implementation of our hierarchical visual-text alignment (HFA) strategy. For CNN-based
backbones, we simply employ a lightweight feature pyramid
network [19] to align the spatial and channel dimensions of
output features. For ViT-based backbones [20], we follow the
design principle proposed by [19] to fuse multi-scale visual
features for a plain backbone. It is worth noting that our HFA
is an inference-free solution and will therefore be discarded
during the inference process.
For the second issue, we empirically find that higher pixelwise confidence scores generated by human parsing often
represent accurate parsing results [13]. Motivated by this
finding, we propose a parsing confidence-weighted alignment
loss for fine-grained image-text alignment. For each pixel-level
image and text feature pairs (vif ull , tfi ull ), we downsample
the parsing confidence map P to the spatial scale of visual
feature maps (i.e., h × w), and perform softmax operation to
normalized it to a weight map P̂ , then we utilize P̂ as the
spatial weight of the pixel-wise mean squared error loss, i.e.,
Lalign = P̂ ∗ ||vif ull − tfi ull ||22 .

(8)

The overall learning objective of the second training stage
is a weighted summation of stage2 loss (Eq.(3)) and Lalign
(Eq.(8)), defined as:
Lstage2′ = Lstage2 + Lalign ,

(9)

We evaluate π-VL on four publicly available person ReID
benchmarks, including MSMT17 [12], Market-1501 [27],
CUHK03 [28], and Occluded-Duke [16]. We follow the general ReID evaluation protocol [27]. The standard cumulated
matching characteristics (CMC) curve and mean average precision (mAP) are used to evaluate the retrieval performance.
B. Comparisons with State-of-the-art Methods
In this subsection, we demonstrate the effectiveness of our
proposed method by comparing it with state-of-the-art ReID
algorithms. The compared approaches include global visual
representation learning methods [5], [6], [8], [21], [23], local
visual representation learning models [3], [4], [17], [21], and
ViT-based approaches [7], [8], [24]–[26]. Table I shows the
Rank-1 accuracy and mAP of various methods across four
datasets. Key observations include:
Visual-language learning benefits RGB-based ReID. As
demonstrated by CLIP-ReID, the feature interaction between
a pair of aligned image and text encoders brings substantial improvements to ReID performance. Specifically, on the
MSMT17 dataset, it shows a 4.4% absolute improvement over
TransReID. On one hand, the broad supervisions provided
by natural language effectively enhance the semantics of
visual features. Moreover, CLIP provides a better starting
point compared to the traditional ImageNet-supervised pretraining [15] used in ReID models. However, CLIP-ReID
only applies natural language supervision to global visual
features, thereby overlooking the fine-grained semantics of
part-informed features.
Part-Informed visual-language learning matters. In
Fig.2, it can be observed that different body part features
are difficult to distinguish using global image-text alignment
(Fig. 2(a)). In contrast, fine-grained visual-language learning
emerges as a promising solution (Fig. 2(b)). This demonstrates
that our π-VL effectively alleviates the semantic inconsistency issue. Beside, our π-VL performance highly comparable

TABLE I
C OMPARISON WITH STATE - OF - THE - ART METHODS . ‘ REPO ’ MEANS OUR REPRODUCED RESULTS .
Backbone

CNN

ViT

Methods

Venues

PCB [3]
MGN [4]
OSNet [6]
Auto-ReID [5]
HOReID [17]
CAL [21]
LTReID [22]
CAJ [23]
CLIP-ReID [8]
CLIP-ReID (repo)
π-VL (ours)
AAformer [24]
TransReID [7]
DCAL [25]
InstructReID [26]
CLIP-ReID [8]
CLIP-ReID (repo)
π-VL (ours)

ECCV 2018
MM 2018
ICCV 2019
ICCV 2019
CVPR 2020
ICCV 2021
TMM 2022
CVPR 2024
AAAI 2023
AAAI 2023
Arxiv 2021
ICCV 2021
CVPR 2022
CVPR 2024
AAAI 2023
AAAI 2023
-

MSMT17
mAP
Rank-1
52.9
78.7
52.5
78.2
56.2
79.5
58.6
81.0
44.3
75.1
63.0
84.4
62.5
84.0
64.2
85.8
63.2
83.6
67.4
85.3
64.0
83.1
72.4
86.9
75.8
89.7
75.0
88.7
76.9
91.0

to state-of-the-arts methods on all four experimental benchmarks. Specially, we achieve 91.0% Rank-1 and 76.9% mAP
for MSMT17, indicating that introducing part-informed text
prompts leads to more semantically meaningful visual features.
Notably, all the improvements are achieved under the singlequery mode without re-ranking or other bells and whistles,
and crucially, without increasing FLOPs or parameters. This
underscores the effectiveness of π-VL
TABLE II
E VALUATION OF EACH MODULE ON THE CUHK03 DATASET.
B: THE BASELINE MODEL , H: HUMAN PARSING - BASED PROMPTS , P :
IDENTITY- AWARE PART- INFORMED PROMPTS , W : PARSING CONFIDENCE
WEIGHTED ALIGNMENT LOSS .
B
✓
✓
✓
✓

H

P

W

✓
✓

✓

✓

Rank-1
81.2 (repo)
81.7 (+0.5)
82.5 (+1.3)
83.7 (+2.5)

mAP
77.8 (repo)
78.4 (+0.6)
78.7 (+0.9)
79.8 (+2.0)

C. Ablation Studies
In this section, we evaluate the effectiveness of different πVL components on CUHK03 dataset, with the results summarized in Table II. Here we adopt CLIP-ReID with the ResNet50 backbone as the baseline model. To assess the impact of
part-informed visual-language learning, we first only employ
the semantic labels generated by human parsing as the text
prompt H, e.g., ‘A photo of a person’s head’. Then we impose
the proposed identity-aware part-informed prompts P, e.g., ‘A
photo of a [X]1 [X]2 [X]3 ...[X]M person’s head’. Finally, we
evaluate the effectiveness of the parsing confidence weighted
alignment loss i.e., the full version of π-VL.
Effectiveness of part-informed semantic labels. As listed
in Table II, when only exploiting the human parsing labels as
text prompts to perform image-text alignment, it surprisingly
yields 0.5% gains of Rank-1 and 0.6% enhancement of mAP.
This suggests that fine-grained image-text alignment generally
enriches the semantics of visual features. Unlike other part
feature learning methods [3], [12], our fine-grained image-text
alignment brings no additional inference costs to B.
Effectiveness of identity-aware part prompts. A limitation of parsing label-based prompts is that they only distinguish the semantics of different body parts, while ReID

Market-1501
mAP
Rank-1
81.6
93.8
86.9
95.7
84.9
94.8
85.1
94.5
84.9
94.2
87.0
94.5
89.0
95.9
86.1
94.4
89.8
95.7
89.2
95.3
90.5
96.5
87.7
95.4
88.9
95.2
87.5
94.7
93.5
96.5
90.5
95.4
90.1
95.2
91.3
97.0

CUHK03
mAP
Rank-1
57.5
63.7
67.4
68.0
67.8
72.3
73.0
77.9
77.8
81.2
79.8
83.7
77.8
79.9
79.5
82.1
83.0
84.5

Occluded-Duke
mAP
Rank-1
43.8
55.1
53.5
61.0
53.2
60.7
54.5
62.3
58.2
67.0
59.2
66.4
60.3
67.2
59.7
66.5
61.4
69.4

tasks require learning identity-related cues for person retrieval.
To address this, we propose combining identity and parsing
prompts for image-text alignment. As evidenced by the 3rd
row of Table II, introducing identity labels results in a 0.8%
improvement in Rank-1 accuracy and a 0.3% increase in mAP.
These gains benefit from our coarse-to-fine strategy, which initially learns identity-specific prompts and subsequently refines
them with part-level semantics.
Effectiveness of the parsing confidence weighted alignment loss. In this paper, we propose a parsing confidence
weighted alignment loss (Eq. 8) to enhance the model robustness to noises generated by human parsing. As shown in the
last row of Table II, this loss function yields 0.6% gains of
Rank-1 and 0.6% enhancement of mAP. This indicates that
our weighted strategy effectively guides the model to focus
on pixels with higher parsing confidence.
TABLE III
G ENERALIZATION ABILITY TO DIFFERENT HUMAN PARSING MODELS .
Method
CLIP-ReID [8] (baseline)
SOLIDER [30]
SCHP [13]
InVPT [14]

mIOU on [29]
55.45
59.36
67.61

Rank-1
81.2
83.0
83.7
83.5

mAP
77.8
79.3
79.8
80.0

Influence of different human parsing models. A major
concern for parsing-based ReID models is that the performance is sensitive to the quality of parsing maps. Here, we
compare the ReID performance of π-VL with different human
parsing models, including SOLIDER [30], InvPT [14], and
SCHP [13], on the CUHK03 dataset. As shown in Table III,
our method consistently outperforms the baseline model across
various parsing approaches. Besides, π-VL demonstrates high
stability in ReID performance regardless of the parsing model
used. This stability likely stems from conducting image-text
alignment at the feature map level rather than the raw pixel
level. Furthermore, we also observe that higher MIOU on [29]
does not necessarily translate to a higher Rank-1 accuracy
on CUHK03. This discrepancy arises because the average
resolution of person images in ReID benchmarks is generally
lower than that in [29], creating a domain gap that affects
parsing quality. Moreover, since π-VL only involves human
parsing during training, we can generate high-quality parsing

maps offline, even using human annotations, to eliminate label
noise in visual-language learning.
V. C ONCLUSION
In this paper, we introduce one of the first attempts to
extend visual-language learning-based ReID from whole-body
to fine-grained part-level. To address within-part semantic
inconsistencies, we use human parsing for pixel-level labeling
and design identity-aware, part-informed text prompts. This
enables fine-grained image-text alignment, creating a more
semantically meaningful embedding space for person ReID.
We also develop a fusion module and a parsing confidenceweighted alignment loss to integrate features at different semantic levels. Building upon these innovations, our π-VL is a
plug-and-play, inference-free solution compatible with modern
backbones. Experiments on CNN and ViT-based models show
its superiority in general ReID tasks.
Acknowledgments. This work was supported by the National
Key R&D Program of China (2022YFB4500600)
R EFERENCES
[1] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and
Steven CH Hoi, “Deep learning for person re-identification: A survey
and outlook,” IEEE transactions on pattern analysis and machine
intelligence, vol. 44, no. 6, pp. 2872–2893, 2021. 1, 2
[2] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang, “Bag
of tricks and a strong baseline for deep person re-identification,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) Workshops, June 2019. 1, 2
[3] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline),” in Proceedings of the European Conference
on Computer Vision (ECCV), September 2018. 1, 2, 4, 5
[4] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, and Xi Zhou,
“Learning discriminative features with multiple granularities for person
re-identification,” in Proceedings of the 26th ACM international conference on Multimedia, 2018, pp. 274–282. 1, 3, 4, 5
[5] Ruijie Quan, Xuanyi Dong, Yu Wu, Linchao Zhu, and Yi Yang, “Autoreid: Searching for a part-aware convnet for person re-identification,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2019, pp. 3750–3759. 1, 4, 5
[6] Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, and Tao Xiang,
“Omni-scale feature learning for person re-identification,” in Proceedings of the IEEE/CVF international conference on computer vision,
2019, pp. 3702–3712. 1, 4, 5
[7] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021,
pp. 15013–15022. 1, 4, 5
[8] Siyuan Li, Li Sun, and Qingli Li, “Clip-reid: Exploiting vision-language
model for image re-identification without concrete text labels,” in
Proceedings of the AAAI Conference on Artificial Intelligence, 2023,
vol. 37, pp. 1405–1413. 1, 2, 3, 4, 5
[9] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin,
Jack Clark, et al., “Learning transferable visual models from natural
language supervision,” in International conference on machine learning.
PMLR, 2021, pp. 8748–8763. 1, 2
[10] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi, “Blip:
Bootstrapping language-image pre-training for unified vision-language
understanding and generation,” in International Conference on Machine
Learning. PMLR, 2022, pp. 12888–12900. 1, 2
[11] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu,
“Learning to prompt for vision-language models,” International Journal
of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022. 1, 2, 3
[12] Longhui Wei, Shiliang Zhang, Wen Gao, and Qi Tian, “Person transfer
gan to bridge domain gap for person re-identification,” in Proceedings of
the IEEE conference on computer vision and pattern recognition, 2018,
pp. 79–88. 1, 2, 4, 5

[13] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang, “Self-correction for
human parsing,” IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 44, no. 6, pp. 3260–3271, 2020. 1, 3, 4, 5
[14] Hanrong Ye and Dan Xu,
“Invpt: Inverted pyramid multitask transformer for dense scene understanding,” arXiv preprint
arXiv:2203.07997, 2022. 1, 5
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep
residual learning for image recognition,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp. 770–
778. 2, 4
[16] Jiaxu Miao, Yu Wu, Ping Liu, Yuhang Ding, and Yi Yang, “Pose-guided
feature alignment for occluded person re-identification,” in Proceedings
of the IEEE International Conference on Computer Vision, 2019, pp.
542–551. 2, 4
[17] Guan’an Wang, Shuo Yang, Huanyu Liu, Zhicheng Wang, Yang Yang,
Shuliang Wang, Gang Yu, Erjin Zhou, and Jian Sun, “High-order
information matters: Learning relation and topology for occluded person
re-identification,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2020, pp. 6449–6458. 2,
4, 5
[18] Laurens Van der Maaten and Geoffrey Hinton, “Visualizing data using
t-sne.,” Journal of machine learning research, vol. 9, no. 11, 2008. 3
[19] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He, “Exploring
plain vision transformer backbones for object detection,” in European
Conference on Computer Vision. Springer, 2022, pp. 280–296. 4
[20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani,
Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., “An image
is worth 16x16 words: Transformers for image recognition at scale,”
arXiv preprint arXiv:2010.11929, 2020. 4
[21] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan,
and Xilin Chen, “Clothes-changing person re-identification with rgb
modality only,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2022, pp. 1060–1069. 4,
5
[22] Pingyu Wang, Zhicheng Zhao, Fei Su, and Honying Meng, “Ltreid:
Factorizable feature generation with independent components for longtailed person re-identification,” IEEE Transactions on Multimedia, 2022.
5
[23] Yiyu Chen, Zheyi Fan, Zhaoru Chen, and Yixuan Zhu, “Ca-jaccard:
Camera-aware jaccard distance for person re-identification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024, pp. 17532–17541. 4, 5
[24] Kuan Zhu, Haiyun Guo, Shiliang Zhang, Yaowei Wang, Gaopan Huang,
Honglin Qiao, Jing Liu, Jinqiao Wang, and Ming Tang, “Aaformer:
Auto-aligned transformer for person re-identification,” arXiv preprint
arXiv:2104.00921, 2021. 4, 5
[25] Haowei Zhu, Wenjing Ke, Dong Li, Ji Liu, Lu Tian, and Yi Shan,
“Dual cross-attention learning for fine-grained visual categorization and
object re-identification,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2022, pp. 4692–4702. 4,
5
[26] Weizhen He, Yiheng Deng, Shixiang Tang, Qihao Chen, Qingsong
Xie, Yizhou Wang, Lei Bai, Feng Zhu, Rui Zhao, Wanli Ouyang,
et al., “Instruct-reid: A multi-purpose person re-identification task with
instructions,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 17521–17531. 4, 5
[27] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang,
and Qi Tian, “Scalable person re-identification: A benchmark,” in
Proceedings of the IEEE international conference on computer vision,
2015, pp. 1116–1124. 4
[28] Wei Li, Rui Zhao, Tong Xiao, and Xiaogang Wang, “Deepreid: Deep
filter pairing neural network for person re-identification,” in Proceedings
of the IEEE conference on computer vision and pattern recognition,
2014, pp. 152–159. 4
[29] Derek Hoiem, Santosh K Divvala, and James H Hays, “Pascal voc 2008
challenge,” World Literature Today, vol. 24, no. 1, 2009. 5
[30] Weihua Chen, Xianzhe Xu, Jian Jia, Hao Luo, Yaohua Wang, Fan Wang,
Rong Jin, and Xiuyu Sun, “Beyond appearance: a semantic controllable
self-supervised learning framework for human-centric visual tasks,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 15050–15061. 5


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

1801

FLAG: A Framework With Explicit Learning Based
on Appearance and Gait for Video-Based
Clothes-Changing Person Re-Identification
Hengjie Lu , Yilin Gao , and Shugong Xu , Fellow, IEEE
Abstract— Person re-identification (ReID) aims to search
for the target person among the non-overlapping surveillance
cameras. Video-based clothes-changing person re-identification
(VCC-ReID) has become an essential branch of ReID due to
the rich spatial and temporal information in the videos and
the broad application of the scenarios. Appearance and gait are
discriminative features in the video-based ReID, but appearance
information is limited due to the clothes changing, which makes
the VCC-ReID challenging. To solve this challenge, we propose a
Framework with explicit Learning based on Appearance and Gait
(FLAG), which can explicitly extract two corresponding types
of information and be combined with most existing video-based
ReID methods. The FLAG includes a multi-modal and multigranularities Architecture (MGA), which is a large model,
and a Cross-Modal Knowledge Distillation Scheme (CMKDS),
which has a small model. They can be applied to devices
with different computing resources. The MGA is designed to
simultaneously take the visible light and silhouette modalities
as input to explicitly learn the appearance and gait features,
respectively. The silhouette modalities are composed of several
levels of granularities to model global and local gait features and
independently serve as input for MGA. The Embedding-Based
parallel fusion module is proposed to fuse the appearance and
multi-granularities gait feature efficiently. The CMKDS is present
to distill the MGA to a small single-modal model that only uses
the visible light modality as input. The Embedding-Based direct
and indirect distillation strategies are designed in the CMKDS.
Experimental results demonstrate that the FLAG combined with
the existing video-based ReID methods can significantly improve
their performance. In addition, when FLAG is combined with
the AP3D method, the MGA can outperform state-of-the-art
accuracy by 4.2%.
Index Terms— Video-based person re-identification, clotheschanging person re-identification, multi-modal learning, knowledge distillation.

I. I NTRODUCTION

R

ECENTLY, Person Re-Identification (ReID), which aims
to match the same person from multiple non-overlapping
cameras, has become a popular research area because of its
wide application, such as intelligent surveillance, criminal
Received 12 April 2024; revised 11 September 2024; accepted 14 October
2024. Date of publication 18 October 2024; date of current version
13 February 2025. This work was supported in part by the National High
Quality Program under Grant TC220H07D, in part by the National Key
Research and Development Program of China under Grant 2022YFB2902002,
and in part by the Innovation Program of Shanghai Municipal Science and
Technology Commission under Grant 20511106603. This article was recommended by Associate Editor J. Shen. (Corresponding author: Shugong Xu.)
The authors are with the School of Communication and Information Engineering, Shanghai University, Shanghai 200444, China (e-mail:
luhengjie@shu.edu.cn; gaoyilin@shu.edu.cn; shugong@shu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3483265

investigation, and so on. With the rise of deep learning
techniques [1], [2], [3], [4], significant progress have been
achieved in ReID [5], [6], [7], [8], [9], [10]. Compared with
image-based ReID, video-based ReID, which belongs to the
video analysis, can perform better due to the rich spatial and
temporal information in the video sequences. Clothes changing
is a common scenario when we want to re-identify over a long
period of time. Besides, clothes changing also exists in some
short-time ReID cases, e.g., the clothes changing caused by a
criminal or hot weather. Overall, video-based clothes-changing
person re-identification (VCC-ReID) is crucial in computer
vision.
Most researchers focus on the video-based ReID with the
same-clothes setting instead of the clothes-changing setting
due to its significant challenges. Specifically, video-based
ReID mainly relies on the appearance and gait (the way
of walking) information of pedestrians, and the appearance
information is dominated by the clothes from the surveillance
camera. Clothes changing will result in limited appearance
information, which makes the VCC-ReID become a challenging task. In this challenging VCC-ReID, the core is how to
mine the limited remaining appearance information (e.g., the
human face) and the gait information. Although some existing
video-based ReID methods can simultaneously extract spatial
and temporal information in video sequences, decoupling and
extracting appearance and gait information still needs to be
improved.
Gait recognition is also a technique for identifying pedestrians, which typically takes silhouettes as input. As shown in
Fig. 1, the silhouettes are segmented by the semantic segmentation model from the visible light images. The silhouettes do
not contain appearance information and are not affected by
the clothes. When using the silhouettes as input, the model
can focus on the gait information. If the existing video-based
ReID methods, which only use visible light modality as input,
can be extended to take visible light and silhouette modalities
as input simultaneously, their performance on VCC-ReID
can be improved by explicitly extracting appearance and gait
information separately.
Based on this idea, we propose a Framework with explicit
Learning based on Appearance and Gait (FLAG), which can
extend most existing video-based ReID methods to extract
appearance and gait information explicitly. Specifically, our
FLAG includes a multi-Modal and multi-Granularities Architecture (MGA) which is a large model, and a Cross-Modal
Knowledge Distillation Scheme (CMKDS) which has a small

1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1802

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

Fig. 1.
Examples of the visible light and silhouette modalities. The
silhouettes are generated from the visible light modalities through the semantic
segmentation model.

model. They can be applied to devices with different computing power.
The MGA simultaneously uses the visible light and silhouette modalities as input to extract appearance and gait
information explicitly. The silhouette modality comprises
N levels of granularity, which help the model extract global
and local gait information. Therefore, the MGA has N+1
branches to process visible light and silhouette modalities.
N branches are used to process the N-granularities silhouette
modality (gait), and one branch is used to process the visible
light modality (appearance). Most existing video-based ReID
methods can used as the appearance and gait branches of the
MGA. Through the MGA, they can be extended to extract
appearance and gait information explicitly. To efficiently
fuse the appearance and multi-granularities gait features, the
Embedding-Based parallel fusion module is designed in the
MGA. The MGA’s performance is excellent but large and
needs silhouettes generated by the semantic segmentation
model. To expand our method to devices with limited computing power, the CMKDS is proposed to distill the MGA
to a small single-modal model that only uses the visible
light modality as input. For example, we can use AP3D [9],
which is a video-based ReID method, as the branches of
MGA and distill this MGA (teacher) to the AP3D (student). The performance of AP3D can be improved with
the explicit appearance and gait features from the MGA
and the inference cost of AP3D will not be increased.
The Embedding-Based direct and indirect distillation strategies are designed to realize efficient cross-modal knowledge
distillation.
Experimental results on CCVID [11], a VCC-ReID
dataset, demonstrate the generalization of our FLAG on
the existing ReID methods. Specifically, when combined
with the existing video-based ReID methods (AP3D [9] and
TCLNet [10]), the MGA can significantly improve their
performance, and the CMKDS can also improve their accuracy without increasing inference cost. In addition, when
combined with AP3D, the MGA can achieve state-of-the-art
performance.

The main contributions can be summarized as follows:
1) A framework with explicit learning based on appearance and gait (FLAG) is proposed to explicitly extract
appearance and gait information on VCC-ReID, and it
can be combined with most existing video-based ReID
methods.
2) A multi-modal and multi-granularities architecture
(MGA) in FLAG, which takes the visible light
and multi-granularities silhouette modalities as input,
is designed to explicitly extract appearance and gait
features and fuse them. The MGA can be applied to
devices with powerful computing power.
3) A cross-modal knowledge distillation scheme (CMKDS)
in FLAG is designed to distill the MGA to a small
single-modal model that only uses the visible light
modality as input. The small model from CMKDS can
be applied to devices with limited computing power.
4) Experimental results demonstrate the generalization
of our FLAG on the existing video-based ReID
methods. In addition, the MGA can outperform stateof-the-art accuracy by 4.2% when combined with
AP3D.
II. R ELATED W ORK
The video-based ReID can be divided into two categories:
the same-clothes setting and the clothes-changing setting.
The video-based ReID with the same-clothes setting assumes
people will not change their clothes. This kind of video-based
ReID is easier but has limited practicality, which is the
mainstream research direction. The video-based ReID with the
clothes-changing setting can be simultaneously applied in the
clothes-consistent and clothes-changing scenes. This kind of
video-based ReID is more practical but received less attention
due to its difficulty. In this section, we will introduce these
two kinds of video-based ReID.
A. Video-Based ReID With Same-Clothes Setting
The video-based ReID with the same-clothes setting attracts
much attention and performs well. Several datasets have been
published to support this task, such as PRID-2011 [12],
iLIDS-VID [13], MARS [14], DukeMTMC-VideoReID [15]
and LS-VID [16]. These datasets only contain the same-clothes
scenes, so the model trained with them can not applied to the
clothes-changing scenes.
The methods in the video-based ReID with the same-clothes
setting can be divided into two types according to how
they model temporal information. One type is directly
using the existing operator, such as CNN [9], [17], [18],
RNN [19], [20] GNN [21], [22], [23], [24] and Transformer [8], [25], [26], [27] to model temporal information.
The other type is designing the specialized module to model
temporal information, such as the module to mine interframe difference [10], [28], [29], [30], [31], [32], [33],
[34], [35], and the module to evaluate importance of frames
[36], [37], [38], [39], [40], [41], [42], [43], [44], [45].
Gu et al. [9] proposed an appearance-preserving module

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

to align the feature maps according to semantic similarity.
Hou et al. [18] proposed a bilateral complementary network
for spatial complementarity modeling and a temporal kernel
selection block to capture short-term and long-term temporal relations. Chen et al. [19] present competitive similarity
aggregation and co-attentive snippet embedding to reduce the
intra-person variation in each sample. McLaughlin et al. [20]
designed a video-based ReID system for wide area tracking
based on an RNN architecture. Yang et al. [21] proposed
a spatial-temporal graph convolutional network containing
spatial and temporal GCN branches. Li et al. [24] present
a spatial-temporal graph-guided global attention network that
can mine spatial-temporal knowledge through graph modeling. He et al. [26] designed a dense interaction learning
framework to reduce the difficulties of multi-grained spatialtemporal interaction modeling. Yang et al. [27] designed a
spatiotemporal interaction Transformer network to effectively
extract the discriminative robust representation. Tang et al. [8]
designed a novel multi-stage spatial-temporal aggregation
Transformer with two designed proxy embedding modules.
Hou et al. [10] designed a temporal complementary learning
network that extracts complementary information of consecutive frames. Chen et al. [30] designed a region-level
saliency and granularities mining network to discover temporal
coherence. Leng et al. [33] proposed a multi-granularities
occlusion aware framework to extract multi-granularities features by precisely erasing the occlusion. Eom et al. [37]
designed a spatial and temporal memory network to extract
robust person representations against spatial and temporal
distractors. Wang et al. [39] proposed a hierarchical mining
network to extract discriminative representations with high
integrity even over sequences where the characteristics of
pedestrians are not consecutive. Tao et al. [43] proposed an
adaptive interference removal framework to remove various
interference.
Significant progress has been achieved in the video-based
ReID with the same-clothes setting. For example, AP3D [9]
and TCLNet [10] are representation methods of the two types
mentioned above; they can achieve 97.2 and 96.9 on Rank-1
metric in the DukeMTMC-VideoReID [15] dataset, respectively. This means the existing methods have a strong practical
application in clothes-consistent scenarios. However, because
these methods are not optimized for the clothes-changing
scene, their performance in it still needs improvement.
As mentioned in Section I, clothes changing is a common scenario in the actual use of video-based person re-identification,
so the video-based ReID with the clothes-changing setting has
great research prospects.
B. Video-Based ReID With Clothes-Changing Setting
Recently, a few researchers have focused on the video-based
ReID with the clothes-changing setting. Gu et al. [11] proposed the first publicly available VCC-ReID dataset named
CCVID. This dataset contains clothes-changing and clothesconsistent scenes, so the models trained with it are more
practical.

1803

The methods in the video-based ReID with the clotheschanging setting can also be divided into two types according
to how they are optimized in clothes-changing scenes. One
type is introducing the additional input with information
that is irrelevant to clothes but relevant to identity, such
as the face [46] and gait (our work). The other type is
introducing the additional task at the output to assist in
decoupling the clothes-irrelevant identity features, such as
the clothes classification [11], [47] and human reconstruction
tasks [47], [48], [49]. Arkushin et al. [46] proposed a method
that combines pre-trained face recognition and ReID models
and created an enriched gallery from the given query and
gallery samples. Gu et al. [11] proposed a clothes-based
adversarial loss to force the backbone of the ReID model
to learn clothes-irrelevant features. Cui et al. [47] designed
a deep component reconstruction ReID framework to disentangle the clothes-irrelevant and the clothes-relevant features.
Liu et al. [48] proposed a joint two-layer shape and texture
representation of a 3D clothed human model to disentangle
identity from non-identity components of 3D clothed humans
and reconstruct accurate 3D clothed body shapes and learn
discriminative features of naked body shapes for person ReID
in a joint manner. Nguyen et al. [49] proposed a temporal
3D shape modeling that can leverage human 3D shape to
assist ReID.
The first publicly available VCC-ReID dataset
(CCVID [11]) is proposed in 2022. After that, some
progress has been achieved in this task. Due to the short
development time and the task’s difficulty, the performance of
current methods in this field still needs to be improved. For
example, GEFF [46] and DCR-ReID [47] are representation
methods of the two types mentioned above; they can
only achieve 89.2 and 84.7 on the Rank-1 metric in the
CCVID [11] dataset, respectively. Such performance is still a
certain distance from actual use. The weaker performance of
current methods and the more practical application value of
VCC-ReID mean that this task has great research prospects.
Therefore, we will focus on the VCC-ReID in this paper.
As mentioned above, in the video-based ReID task, more
progress has been achieved on the same-clothes setting
compared to the clothes-changing setting. So, the methods
designed for the same-clothes setting are extensive and have
potential, as they have yet to be optimized for the clotheschanging setting. Therefore, we propose the FLAG, which
can combined with the existing video-based ReID methods,
to transfer the methods designed for the same-clothes setting
to the clothes-changing setting. Through our FLAG, we can
significantly improve the performance of the methods designed
for the same-clothes setting in the clothes-changing setting and
fully utilize these methods. Specifically, we will combine our
FLAG with AP3D [9] and TCLNet [10], which are designed
for the same-clothes setting. As mentioned in Section II-A,
AP3D and TCLNet represent two types of methods in the
video-based ReID with the same-clothes setting according
to how they model temporal information. Combining with
them can comprehensively demonstrate the generality of
our FLAG.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1804

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

Fig. 3. The multi-modal and multi-granularity architecture. We also use
N-granularities (N=2) silhouettes as an example.

Fig. 2. The overall framework of our FLAG. In this example, MGA uses
visible light and two-granularities silhouette modalities as input. The CMKDS
is used to distill the large MGA to the small baseline model.

III. M ETHOD
A. Overall Framework
Our work aims to improve the performance of the existing
video-based ReID models in the task of VCC-ReID by modeling pedestrian appearance and gait features. To realize this
goal, we propose a Framework with explicit Learning based on
Appearance and Gait (FLAG). As shown in Fig. 2, our FLAG
includes a multi-Modal and multi-Granularities Architecture
(MGA) and a Cross-Modal Knowledge Distillation Scheme
(CMKDS). The MGA uses visible light and N-granularities
(N = 2 in Fig. 2 as an example) silhouette modalities as
inputs to explicitly extract appearance and N-granularities gait
features, respectively. With the help of N-granularities gait
features, the MGA performs better but is larger and needs more
modality than the baseline model we use, such as AP3D [9]
and TCLNet [10]. Therefore, we use the CMKDS to distill
our MGA to the baseline model. Through the knowledge
(appearance and N-granularities gait) from the MGA, the
baseline model’s performance can be improved without any
additional inference cost. Finally, we can get a large MGA
and a small improved baseline model for different computing
power devices. Our FLAG has generality, so most existing
video-based ReID models can be used as the baseline model.
B. Multi-Modal and Multi-Granularities Architecture
As shown in Fig. 3, we use N+1 branches to process
visible light and N-granularities silhouette modalities. The
silhouettes of different granularity represent different numbers
of segmentation parts for the human body. For example,
in Fig. 3, the first-granularity silhouettes mean dividing the
human body into one part. In our MGA, the maximum
number of granularities of silhouettes can be set to five, which
means there are at most six branches in the MGA. However,
when the number of granularities reaches the threshold, the

gait information will be fully modeled, and increasing more
granularities at this time will lead to a decrease in MGA’s
performance, and the more granularity, the more parameters
the MGA will have. The specific analysis will be shown in
Section IV-C. Each branch uses a baseline model, such as
AP3D and TCLNet, to explicitly extract the corresponding
appearance and N-granularities gait features. Most existing
video-based ReID models can be taken as our baseline model.
The baseline models in N+1 branches have the same structure
but independent parameters.
The MGA’s training process can be divided into two stages.
In the first stage, we will pre-train each branch on its corresponding modality. In this stage, the appearance embedding
(E A ) and N-granularities gait embeddings can be generated.
We take N = 2 as an example to introduce, the appearance
embedding (E A ), first-granularity gait embedding (E G1 ) and
second-granularity gait embedding (E G2 ) can be generated as
follows:
E A = BV (VV ),
E G1 = B S1 (VS1 ),
E G2 = B S2 (VS2 ),

(1)

where E A , E G1 , E G2 ∈ R2048 . The VV , VS1 , and VS2 represent
the video clips of visible light, first-granularity silhouette, and
second-granularity silhouette, respectively. The BV , B S1 , B S2
represent the visible light branch, first-granularity silhouette
branch, and second-granularity silhouette branch, respectively.
In the second stage, we will freeze each branch and train the
Embedding-Based Parallel Fusion Module (EPFM) proposed
to parallel fuse each modality’s video embedding. In this stage,
the final embedding(E F ∈ R2048 ) can be fused from E A , E G1
and E G2 as follows:
E F = E P F M(E A , E G1 , E G2 ).

(2)

Introducing silhouette modalities can force the model to
learn gait information. Moreover, multi-granularities silhouette modalities can help the model capture global and local
gait features, effectively complementing appearance features.
Extracting features through independent branches helps to
avoid interference between modalities. Pre-training and freezing each branch can increase the model’s interpretability and
further ensure the complementarity of the branches’ features.
Our EPFM can be understood as a late fusion of feature
fusion, which can effectively fuse complementary features
while maintaining low complexity.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

1805

Fig. 4. Different granularities and forms of silhouettes. In the nineteen parts silhouettes, due to the detailed division of body parts, not every human image
will include these parts, so the number of body parts is less than nineteen. Distinguishing the differences among different granularities in gray continuous
through human eyes is difficult.

1) Multi-Granularities Silhouettes: We use SCHP [2] to
segment silhouettes from the visible light images. SCHP is
a human parsing (a fine-grained semantic segmentation task)
model, which can segment the human body into at most
nineteen parts.
By combining nineteen body parts, we design five different
granularities of silhouettes. As shown in Fig. 4, the human
body is divided into single, two, three, four, and nineteen
parts, respectively in them. The N-granularities silhouettes
mentioned above represent that we choose N kinds of granularities from them. Therefore, the N can be set to a maximum
of five. Through them, the model can be constrained to extract
gait features from different granularities. For example, the
single-part silhouettes can help the model extract global gait
features, and the silhouettes of the other granularities can help
the model extract local gait features, and the more body parts
there are, the more detailed the model focuses on.
In addition to five different granularities silhouettes, we also
designed three forms of silhouettes: gray equal, pseudo color,
and gray continuous. Gray equal and gray continuous mean
human parts and background are shown in grayscale images.
In gray equal, the grayscale values of body parts and background are taken at equal intervals. In gray continuous, the
grayscale values of body parts and background are continuously taken. Pseudo color means human parts and background
are shown in pseudo color. Different granularities silhouettes
require different forms to help the model better extract gait
features. In addition, when the combination of selected granularities differs, the required forms also vary. We will discuss
this in Section IV-E1.
The main-stream silhouette scheme in gait recognition is
single-granularity, which is a single part and gray equal. In our
MGA, we expand it to multi-granularities to extract global and
local gait features. In addition, to ensure the better presentation
of each granularity’s silhouette, we design three forms for
them.

Fig. 5.
The embedding-based parallel fusion module. We also use
N-granularities gaits (N=2) as an example.

2) Embedding-Based Parallel Fusion Module: As shown
in Fig. 5, EPFM uses the appearance embedding (E A ), firstgranularity gait embedding (E G1 ) and second-granularity gait
embedding (E G2 ) as inputs.
First, E A , E G1 , and E G2 are concatenated and fed into
the attention layers. The attention layers consist of two fully
connected layers and a Softmax layer. After attention layers,
three corresponding attention weights (W A , WG1 , WG2 ∈
R2048 ) can be get. The process can be formulated as:
[W A ′ , WG1 ′ , WG2 ′ ] = FC(FC([E A , E G1 , E G2 ])),
W A , WG1 , WG2 = So f tmax(W A ′ , WG1 ′ , WG2 ′ ),

(3)

where W A ′ , WG1 ′ and WG2 ′ are the weights before normalization. [·] means the concatenation operation. The FC means
the fully connected layer.
Second, three adapters transform E A , E G1 , E G2 into the
same feature space. Three adapters have the same structure
and independent parameters, which consist of a fully connected layer and an activation function (Tanh). After adapters,
E A ′ , E G1 ′ , E G2 ′ ∈ R2048 can be get. The process can be

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1806

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

TABLE I

formulated as:
E A ′ = T anh(FC(E A )),
E G1 ′ = T anh(FC(E G1 )),
E G2 ′ = T anh(FC(E G2 )).

PARAMETER AND C OMPUTATIONAL C OMPLEXITY C OMPARISON
B ETWEEN BASELINE M ODELS AND MGA

(4)

Then we fuse E A ′ , E G1 ′ , E G2 ′ with their corresponding
weights. A batch norm layer is added at the end, and the final
video embedding (E F ∈ R2048 ) can be obtained from it. The
process can be formulated as:
X
EF = B N(
Wi × E i ′ ),
(5)
i∈S

where S represents the set of the appearance, first-granularity
gait, and second-granularity gait, and the B N represents the
batch norm layer.
3) Loss: We combine the cross entropy loss and triplet
loss [50] as the overall loss when pre-training each branch
and training the EPFM in the MGA.
Cross Entropy Loss is usually used in the classification task.
When used in the ReID task, each identify can be regarded
as a class. When using this loss, we will add a classification
layer after the final video embedding (E F ) to transform it to
the prediction. Its formulation is defined as:
LC E = −

B
X

pi log qi ,

(6)

i=1

where B is the mini-batch size, pi is the one-hot ground truth,
and qi is the prediction probability.
Triplet loss is widely used in the ReID task, and it aims to
ensure that the maximal distance between an anchor and all
positive samples is smaller than the minimal distance between
this anchor and all negative samples. Its formulation is defined
as:
L T ri =

B
X
[m + max
i=1

e p ∈Po(ei )

d(ei , e p )
−

min

en ∈N e(ei )

d(ei , en )]+ ,

(7)

where m is the margin hyper-parameter, [·]+ = max(0, ·),
ei is the i th final video embedding (E F ), and distance function
d(·, ·) here is cosine distance. Po(ei ) and N e(ei ) are positive
and negative sets for ei , respectively.
The Overall Loss of MGA is defined as:
L Overall = λ1 L C E + λ2 L T ri ,

(8)

where λ1 and λ2 are the weights to balance two losses.
C. Cross-Modal Knowledge Distillation Scheme
Table I shows the parameter and computational complexity
(MACs, multiply-add operations) comparison between Baseline Models and MGA. In Table I, we use AP3D and TCLNet
as our baseline model, and we use the MGA with single,
two, three, four, and five granularities silhouettes to compare.
The 1-G Silhouette in Table I means the single-granularity
silhouette; the others are similar. Our MGA can achieve
better performance but has more parameters and computational

Fig. 6. The cross-modal knowledge distillation scheme. The core of CMKDS
is the Embedding-Based direct and indirect distillation strategies. We also use
N-granularities (N=2) silhouettes as an example.

complexity and needs more modalities. So, our MGA is
more suitable for devices with powerful computing power.
To expand our work to more devices with limited computing
power, we design the CMKDS.
As shown in Fig. 6, our CMKDS uses the MGA as a
teacher model and the baseline model as a student model.
Although the baseline model only uses visible light modality
as input, the knowledge with appearance and gait features,
which are explicitly extracted, can help the baseline model
improve performance.
In our CMKDS, the distillation process can be divided into
two stages. First, we pre-train the MGA and baseline model
(M B ). This stage can be formulated as:
E T = M G A(VV , VS1 , VS2 ),
E S = M B (VV ),

(9)

where E T , E S ∈ R2048 represent the teacher and student
embedding, respectively.
Second, we freeze the MGA and distill the baseline model
with our Embedding-Based Direct and Indirect Distillation
Strategies (EDIDS).
Through our CMKDS, we can get an improved baseline
model, which can used on devices with limited computing
power. It has better performance and the same parameter
quantity, computational complexity, and input modality as the
original baseline model. Our CMKDS also has generality,
which can be combined with most existing video-based ReID
methods.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

1807

1) Embedding-Based Direct and Indirect Distillation
Strategies: The EDIDS can be divided into direct and indirect
distillation strategies, and they all distill the embedding from
the student model. The direct distillation strategy directly uses
the embedding from MGA (teacher) to distill the student
embedding, which is the existing main-stream knowledge
distillation scheme. The indirect distillation strategy uses the
pre-trained and frozen MGA (teacher) classification layer as
the classifier for the student model. Through this way, when
we use cross entropy loss to supervise the student model,
the student embedding will be indirectly constrained to fit
the teacher embedding. Both distillation strategies are used
after an adapter consisting of a fully connected layer. We only
need the baseline model (student) without the adapter when
inference.
The input modalities of teacher and student models are
inconsistent, so we use the embedding located deep in the
model with less relationship with the input as the distillation
object to reduce the impact of cross-modal distillation. The
goals of direct and indirect distillation strategies are consistent. Both aim to ensure that the student embedding can fit
the teacher embedding, thereby fully extracting appearance
and gait features. The adapter can reduce the difference in
feature space between teacher and student embedding, thereby
reducing the impact of cross-modal distillation.
2) Loss: As mentioned in Section III-B3, we use cross
entropy loss and triplet loss to pre-train the MGA and baseline
model. When distilling the baseline model, the mean square
error loss and cross entropy loss are used in direct and
indirect distillation strategies, respectively. In addition, the
triplet loss [50] is also used to supervise the baseline model.
As shown in Fig. 6, the mean square error loss and cross
entropy loss are used after the adapter, and the triplet loss is
used before the adapter.
Mean Square Error Loss can be used to calculate the
difference between two embedding. This loss in distillation
stage is formulated as follows:
E S ′ = FC(E S )
B

L MSE =

1 X
(eˆi − ei )2 ,
B

(10)

i=1

where E S ′ represents the student video embedding after the
adapter. The eˆi and ei are the i th E S ′ and E T , respectively.
The cross entropy loss in distillation stage is formulated as
follows:
LC E ′ = −

B
X

pi log qi ,

(11)

i=1

where B is the mini-batch size, pi is the one-hot ground truth.
The qi is the prediction probability generated from the E S ′ and
MGA (teacher) classifier.
The triplet loss in distillation stage is formulated as follows:
L T ri ′ =

B
X
[m + max

e p ∈Po(ei )

i=1

−

min

en ∈N e(ei )

d(ei , e p )
d(ei , en )]+ ,

(12)

where m is the margin hyper-parameter, [·]+ = max(0, ·), ei is
the i th student video embedding (E S ), and distance function
d(·, ·) here is cosine distance. Po(ei ) and N e(ei ) are positive
and negative sets for ei , respectively.
The Overall Loss of CMKDS in distillation stage is defined
as:
L Overall ′ = α1 L M S E + α2 L C E ′ + α3 L T ri ′ ,

(13)

where α1 , α2 , and α3 are the weights to balance three losses.
IV. E XPERIMENTS
A. Dataset and Settings
1) Dataset: CCVID [11] is the first publicly available
VCC-ReID dataset and is widely used by researchers. In addition, CCVID contains clothes-changing and clothes-consistent
scenes and has more excellent practical application value.
For these reasons, we evaluate our FLAG on the CCVID.
CCVID contains 2,856 sequences and 226 identities. Each
identity wears 2 to 5 suits of clothes. The length of each
sequence changes from 27 to 410 frames, with an average
length of 122. This dataset is split into 75 identities for training
and 151 identities for test. In the test set, 834 sequences are
used as the query set, and the other 1074 sequences form the
gallery set.
2) Evaluation Protocol: Following existing practices, the
Rank-k matching accuracy and the mean Average Precision
(mAP) are used to evaluate performance. Rank-k matching
accuracy means the probability of a correct match appearing in
the top-k ranked retrieved results. The mAP reflects the overall
performance of the algorithm with precision and recall. The
precision indicates the accuracy of the algorithm’s prediction,
and the recall indicates the completeness of the algorithm’s
prediction. In addition, two kinds of test settings are defined:
general setting and clothes-changing setting. The general
setting means both clothes-changing and clothes-consistent
ground truth samples are used to calculate accuracy. The
clothes-changing setting (CC) means only clothes-changing
ground truth samples are used to calculate accuracy.
3) Implementation Details: When training, we randomly
sample 8 frames with a stride of 4 to form a video clip, and
each frame is resized to 256 × 128, and only the horizontal
flip is used for data augmentation. When testing, each video
sample is divided into a series of 8-frame clips with a stride
of 4, and the averaged embedding of these clips is used
as the final representation of the video for testing. These
implementation details are the same as [11] to realize fair
comparisons.
4) Abbreviation: Some abbreviations will be used in the
following tables, and we will explain them here. As used
in Section III-C, N-G Silhouette means the N-granularities
silhouettes. As mentioned in Section III-B1, we design five
kinds of granularities and three kinds of forms. By combining
different granularities and forms, we can obtain 15 kinds
of silhouettes. We number them as abbreviations: 1 represents Single Part & Gray Equal, 2 represents Single Part &
Pseudo Color, 3 represents Single Part & Gray Continuous,
4 represents Two Parts & Gray Equal, 5 represents Two

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1808

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

TABLE II

TABLE III

P ERFORMANCE C OMPARISON OF O UR M ETHOD
W ITH S TATE - OF - THE -A RT M ETHODS

G ENERALIZATION E XPERIMENTS ON MGA

Parts & Pseudo Color, 6 represents Two Parts & Gray Continuous, 7 represents Three Parts & Gray Equal, 8 represents
Three Parts & Pseudo Color, 9 represents Three Parts &
Gray Continuous, 10 represents Four Parts & Gray Equal,
11 represents Four Parts & Pseudo Color, 12 represents Four
Parts & Gray Continuous, 13 represents Nineteen Parts &
Gray Equal, 14 represents Nineteen Parts & Pseudo Color,
15 represents Nineteen Parts & Gray Continuous.
B. Comparison With State-of-the-Art Methods
Table II shows the performance comparison of our method
with state-of-the-art methods. The MGA in Table II has
four branches (visible and three-granularities silhouettes), and
each branch uses AP3D to extract the corresponding features
explicitly. Except for the results of AP3D [9] and TCLNet [10]
which we reproduce, the rest results all come from [11], [46],
[47], [48], and [49].
Our MGA achieves state-of-the-art performance on all evaluation metrics and test settings. Compared with the existing
state-of-the-art method GEFF [46], our MGA outperforms it
by 4.2% and 2.8% for the Rank-1 metric on the general and
CC settings, respectively. In addition, our MGA significantly
improves the performance of existing methods (AP3D). Compared with the baseline model (AP3D), our MGA outperforms
it by 4.8% and 6.1% for the Rank-1 metric on the general and
CC settings, respectively.
We will explain why our MGA offers better results than
other methods and can improve the performance of existing
methods. As mentioned in Section I, the core in the VCC-ReID
is how to mine the limited remaining appearance information
(e.g., the human face) and the gait information. Existing
video-based ReID methods often only take visible light as
input. On the one hand, we cannot determine whether the
features extracted in this situation can contain both appearance and gait information. On the other hand, even though
both appearance and gait information can be extracted, their
decoupling is inadequate. However, our MGA, which can
combined with other existing methods, can simultaneously
take visible light and silhouette modalities as input and process
these modalities using separate branches, which can explicitly

extract appearance and gait features and achieve sufficient
decoupling.
C. Generalization Experiments on MGA
Table III shows the results of generalization experiments on
MGA. As mentioned in Section II, combining with AP3D [9]
and TCLNet [10] can comprehensively demonstrate the generality of our FLAG, so we use them as our baseline models
in Table III. As mentioned in Section III-B1, the number of
granularities can be set to a maximum of five, so we conduct
experiments with visible light and N-granularities (N = 1, 2, 3,
4, 5) silhouette modalities. The choice of different granularities
and forms will affect performance, and we use the choice
with the best performance to compare. For example, the best
choice in visible light and 3-granularities silhouette modalities
is using visible light, Single Part & Gray Equal silhouette,
Three Parts & Pseudo Color silhouette, and Four Parts &
Gray Equal silhouette. The ablation experiments on the choice
of granularities and forms will be shown in Section IV-E1.
We also compare the MGA with the single-modality models.
As shown in Table III, comparing our MGA with the singlemodality models, performance can be improved by 3.12%
∼ 3.6% and 2.64% ∼ 5.76% on Rank-1 with AP3D and
TCLNet, respectively. This means that no matter what the
value of N is in MGA, the performance can be significantly
improved on two baseline models, which can demonstrate the
generalization of our MGA. In addition, when the number
of granularities is set to three, the best performance can be
obtained. This demonstrates that silhouettes with more granularities can help the model capture more detailed gait features,
thereby improving performance. However, when the number

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ESTI: An Efficient Spatial-Temporal Interaction
Network For Video-Based Person Re-Identification
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209571

Guquan Jing1,2 , Peng Gao1,2 , Yiyang Hu1,2 , Yujian Lee1,2 , Hui Zhang2 ∗
1

2

Department of Computer Science, Hong Kong Baptist University, Hong Kong, China
Department of Computer Science, Beijing Normal-Hong Kong Baptist University, Guangdong, China
{guquanjing1, gaopeng1225, yiyanghu124, yujianlee1119}@gmail.com, amyzhang@uic.edu.cn

Abstract—Video-based person re-identification (Re-ID) aims
to identify the target pedestrian from video sequences. However, redundant information exist in input frames. Extracting
spatial-temporal features in whole adjacent frames can introduce
additional computational overhead. Furthermore, this process
leads to the loss of critical spatial and temporal details, causing
suboptimal representations. To mitigate these issues, we propose
an Efficient Spatial-Temporal Interaction (ESTI) network, which
processes half of the input sequence separately through spatial
and temporal branches, extracting high-level discriminative features across multiple layers and avoiding redundancy computations. In particular, we propose a Feature Enhancement Module
(FEM) for the spatial branch to focus on enhancing spatial dependencies adaptively, and a Temporal Interaction Module (TIM)
for temporal branch to capture temporal correlations effectively.
Spatial-temporal interaction is performed at the final layer to
generate distinctive representations. Extensive experiments on
three challenging video Re-ID datasets show that our ESTI
achieves competitive results while maintaining low computational
complexity.
Index Terms—Video-based person re-identification, spatialtemporal information

I. I NTRODUCTION
Video-based person Re-Identification (Re-ID) [1]–[4] aims
to identify the same pedestrian from video sequences across
non-overlapping cameras, which is a crucial task in intelligent
surveillance and video retrieval. Different from the imagebased Re-ID that relies on single-shot images, video-based
Re-ID offers richer spatial-temporal information. Efficiently
leveraging these spatial-temporal cues is significant for achieving robust performance.
Early methods to video Re-ID adapt models from other
video tasks, such as 3D CNN [5] and RNN [6], to learn
video temporal information, which are not suitable for video
Re-ID. Recent studies [1]–[4], [7]–[16] first extract framelevel features, then aggregate them temporally to learn spatialtemporal representations. However, these methods face significant limitations due to the redundancy within video sequences.
Figure 1 shows two sampled sequences through the Restricted
Random Sampling (RRS) strategy [7] from the iLIDS-VID
∗ Corresponding author.
This work is supported in part by the Natural Science Foundation of
China (62076029); in part by the National Key R&D Program of China
(2022YFE0201400); in part by the Guangdong Provincial Key Laboratory
of Interdisciplinary Research and Application for Data Science, BNU-HKBU
United International College (2022B1212010006).

Spatial Feature

Discriminative
Spatial-temporal Representation

Temporal Feature

Spatial-temporal Representation
Multi-Layer
Enhancement

Spatial Feature

Temporal Feature

Multi-Layer
Interaction

Spatial-Temporal Module
Similar Feature Representations

Similar Frames
Fig. 1. Two sampled sequences from iLIDS-VID (the image sequence above)
and MARS (the image sequence below) dataset with previous video Re-ID
methods (black boxes and lines) and our method (red boxes and lines).

[17] and the MARS [18] datasets. Input frames in video ReID often exhibit minimal variation, as pedestrians perform
limited actions with subtle frame differences. Recent methods
aggregate features across such whole adjacent frames, leading
to computational inefficiency due to redundant computations
that contribute minimally to the overall representation but
still require full processing. Moreover, existing aggregation
strategies on similar frames often cause spatial and temporal
information loss. As shown in Figure 1, recent methods tend to
aggregate similar features temporally, causing an overemphasis
on redundant areas while ignoring critical local regions. These
inefficiencies dilute key features and restrict the generation of
high-level spatial and temporal representations.
To tackle aforementioned issues, we propose an Efficient Spatial-Temporal Interaction (ESTI) network for videobased person Re-ID (see Figure 1 for a conceptual illustration). Specifically, the network divides input frames into
two branches: a spatial branch and a temporal branch, each
processing half of the sequence to extract high-level spatial
and temporal features, respectively. This half-sequence extraction strategy balances computational efficiency and pedestrian

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

representation quality through capturing comprehensive spatial
and temporal information without over-processing spatialtemporal aggregation of similar content. Inspired by [3], a
pyramid structure is adopted to generate distinctive features
across layers. Our design significantly reduces computational
cost and mitigates redundancy often introduced by the spatialtemporal aggregation across whole adjacent frames. By extracting critical spatial and temporal features from separate
halves of the input sequence and performing spatial-temporal
interaction at the final stage, the ESTI maintains efficiency
while preserving crucial details. To obtain high-level spatial
and temporal dependencies, we propose the Feature Enhancement Module (FEM) for the spatial branch and the Temporal
Interaction Module (TIM) for the temporal branch. The FEM
emphasizes spatial cues and local details, while the TIM
models temporal correlations across adjacent features to capture both local and global dynamics. Both modules leverage
deformable attention [19] to adaptively extract crucial pedestrian information while reducing computational overhead. To
facilitate effective training, a Feature Aggregation (FA) module
is designed to generate pedestrian representations at each layer
under supervision for feature refinement. Consequently, highlevel spatial and temporal features are aggregated via the
Spatial-Temporal Interaction Module (STIM), which shares
a similar structure with the TIM. By incorporating these
components, the ESTI efficiently captures the interaction of
spatial and temporal information, generating discriminative
video-level representations.
The contributions of this paper can be summarized as
follows:
• We propose an Efficient Spatial-Temporal Interaction
(ESTI) Network, incorporating spatial and temporal
branches to minimize redundancy and efficiently obtain
high-level spatial-temporal dependencies.
• We propose a Feature Enhancement Module (FEM) to enhance spatial features and a Temporal Interaction Module
(TIM) to capture temporal correlations. The high-level
spatial and temporal features are generated for further
spatial-temporal interaction. A Feature Aggregation (FA)
module is designed to create pedestrian representations
at each layer under supervision, facilitating effective
training.
• Extensive experiments on three challenging video Re-ID
datasets demonstrate that our network achieve a competitive performance with a low computational cost compared
to state-of-the-art methods.
II. R ELATED W ORKS
Video-based person re-identification (Re-ID) aims to retrieve the target pedestrian from video sequences. Early methods primarily utilize models for other video tasks, such as
RNN [6] and 3D CNN [5], to directly model temporal information. For example, Eom et al. [6] exploit RNNs to encode a
sequence temporally, enabling access to the temporal memory.
Gu et al. [5] propose a network that uses 3D convolutions
to model temporal information while preserving appearance

information. However, these models are not suitable for video
Re-ID task as they are not designed specifically for this
domain and introduce a mass of parameters. Recent methods
[1]–[4], [7]–[16] obtain features in each frame, subsequently
aggregate multi-frame features. For instance, Wang et al.
[3] propose a feature aggregation framework with a pyramid
structure to aggregate frame-level features temporally. Wu et
al. [16] enhance extracted features based on the pedestrian
relative state before aggregating them. Despite the remarkable
progress achieved, these methods fail to mitigate the impact
of redundant information in video Re-ID data, which can lead
to computational overhead and critical information loss.
III. M ETHOD
A. overview
As shown in Figure 2 (left), the overall structure of our
proposed network consists of a spatial branch and a temporal
branch. Specifically, given a video tracklet with T frames V =
{It }Tt=1 , it is first fed into a backbone network (e.g., ResNet50 [20]) to extract frame-level features F 0 = {Ft0 }Tt=1 ,
where Ft0 ∈ RC×H×W . We separate these features along the
time axis, with the spatial branch focusing on the first half
{F10 ...F T0 } to enhance spatial dependencies and the temporal
2
branch processing the rest {F T0 +1 ...FT0 } to capture temporal
2
correlations. Inspired by [3], we utilize a pyramid structure to
extract multi-layer features while mitigating irrelevant information. For each branch, the proposed Feature Enhancement
Module (FEM) and Temporal Interaction Module (TIM) are
applied at each layer to extract distinctive spatial and temporal
features F n = {Fln }L
l=1 , where n denotes the index of layer
starts from 1, and L = 2Tn . At the final layer, the SpatialTemporal Interaction Module (STIM) integrates the high-level
spatial and temporal features, capturing comprehensive spatialtemporal correlations. Similar to [3], the ESTI supervises
multi-layer features for effective learning. However, we enhance this process with a Feature Aggregation (FA) module,
which combines features across layers for discriminative representations. The details are presented below.
B. Spatial-Temporal Interaction
1) Spatial Branch: We extract spatial features from half of
the input sequence to focus on discriminative spatial representations. To achieve this, we propose the Feature Enhancement
Module (FEM), which emphasizes the target pedestrian in the
spatial domain using deformable attention [19], as shown in
Figure 2 (right). In detail, given the feature map Ft of the
t-th feature in the spatial branch, we first generate embedded
features Qt and Vt through embedding layers θs and θs′ , and a
linear layer φs . A set of 2-D reference points rt is derived from
Qt via a linear projection φr , indicating sampled positions on
Vt . We can express these as:
Qt = θs (Ft ), Vt = φs (θs′ (Ft )), rt = φr (Qt ).

(1)

Afterward, we define j index both the feature Qt and reference
t
points rt . The learnable offset ∆rmjk
for each reference point

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

2-th Layer
1-th Frame

TIM

Temporal Interaction Module

FEM

Feature Enhancement Module

STIM

Spatial-temporal Interaction Module

AG

0.1 0.4 0.3 0.2

Reference Point
Addition

FA
AG

FA

Feature Aggregation

Aggregation

Residual
RCB Connection Block

The Spatial-Temporal Feature

Layer-2
FA

RCB

STIM
0.1 0.4 0.3 0.2
Layer-1

AG

FEM

TIM

FA

Feature Enhancement
Module (FEM)

Layer-0

FEM

FEM

TIM

TIM
AG
0 0.5 0.3 0.2

RCB

Temporal Branch

Spatial Branch
Image Backbone

0.1 0.4 0.3 0.2
AG

Temporal Interaction
Module (TIM)

Input Frames

Fig. 2. The overall architecture of our proposed ESTI (left) and the illustration of the Feature Enhancement Module (FEM) and the Temporal Interaction
Module (TIM) (right). In the overall architecture (left), we use eight frames (T = 8) as an example.

rtj , and attention weight Atmjk (j-th Qt to k-th Vt at m-th
head) are acquired from two linear projections (θo and θa ) and
a softmax function to search local crucial positions around rtj
and generate distinctive features. These can be expressed as:
t
∆rmjk
= θo (Qjt ), Atmjk = softmax(θa (Qjt )).

(2)

Subsequently, the sampled positions Vtk are aggregated with
the attention weight Atmjk . The enhancement process E(·) is
formulated as:
E(Qjt , rtj , Vt ) =
M
X

Wm

m=1

"K
X

′
t
Atmjk · Wm Vt (rtj + ∆rmjk
)

k=1

#

,

(3)

Qt+1 = θt+1 (Ft+1 ), Vt = φt (θt′ (Ft )),

′

where Wm and Wm are learnable weights. M and K are the
total attention head and sampling point number, respectively.
Therefore, an enhanced feature map F̂tS in the spatial branch
is obtained by implementing enhancement process to Qt ,
followed by a shortcut connection to the initial feature map
Ft . This processing enables the FEM to adaptively focus on
local relevant regions, thereby enhancing the discriminative
capacity of spatial features. Eventually, we aggregate the
adjacent feature maps by applying the element-wise addition
and a Residual Connection Block (RCB) [20]. The feature for
the input of next layer FS in the spatial branch is calculated
by:
S
FS = RCB(F̂tS + F̂t+1
).

By leveraging these enhancements, the FEM ensures that our
network captures critical spatial details, contributing to the
generation of discriminative spatial representations.
2) Temporal Branch: For the remaining feature maps in
the input sequence, we apply the Temporal Interaction Module
(TIM) to capture temporal correlations, as shown in Figure 2
(right). Similar to the FEM, the TIM utilizes the deformable
attention [19] to efficiently model temporal dependencies.
Differently, the TIM focuses on interaction between adjacent
feature maps. These feature maps inquire each other to generate distinctive one. Formally, given adjacent feature maps
Ft and Ft+1 in the temporal branch, the temporal interaction
process I(·) that inquires the temporal information in Ft using
Ft+1 can be formulated as:

(4)

rt+1 = φt+1
r (Qt+1 ),
j
I(Qjt+1 , rt+1
, Vt ) =
"K
#
M
X
X
′
j
t+1
t+1
Amjk · Wm Vt (rt+1 + ∆rmjk ) ,
Wm
m=1

(5)

(6)

k=1

where θt′ and θt+1 are embedding layers. φt and φt+1
are
r
t+1
linear layers. The sampling offsets ∆rmjk
and the attention
j
weight At+1
mjk are generated from Qt+1 . This allows our
network to establish temporal correlations between adjacent
frames. We also inquires the temporal information in Ft+1
using Ft . By performing mutual inquiry between Ft and Ft+1
during the interaction and incorporating a shortcut connection

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

Global
Average Pooling
Temporal
TAP Average Pooling
C
Concatenation

TABLE I
P ERFORMANCE (%) COMPARISON OF OUR METHOD AND
STATE - OF - THE - ART METHODS ON MARS, I LIDS-VID AND PRID-2011.
T HE BEST RESULTS ARE MARKED AS BOLD . T HE SECOND - BEST RESULTS
ARE UNDERLINED .

GAP
FEM

RCB

GAP

From Layer 1
FEM

RCB

GAP

C

TAP

GAP

C

TAP

From Layer 2
Spatial-Temporal
Feature

RCB

Methods

Fig. 3. The illustration of the Feature Aggregation (FA) module.

for both Ft and Ft+1 , two interaction features F̂tT em and
T em
F̂t+1
in the temporal branch are generated, as shown in
Figure 2 (right). This process enables the TIM to effectively
model temporal relationships while mitigating redundancy
from similar frames. Same as the spatial branch, the feature for
next layers’ input FT em in the temporal branch is calculated
by:
T em
FT em = RCB(F̂tT em + F̂t+1
).

(7)

The TIM combines temporal cues from adjacent frames to
capture both local and global dynamic details, thereby ensuring
the discriminative temporal feature generation.
3) Interaction: We aggregate the spatial and temporal features at the final layer using the Spatial-Temporal Interaction
Module (STIM), which shares the same structure as the
TIM. However, it takes both the spatial feature FS and the
temporal feature FT em at the final layer as input, performing
mutual inquiry between them to obtain comprehensive spatialtemporal representations that include both distinctive parts of
spatial and temporal features.
C. Feature Aggregation Module
We generate pedestrian representations at each layer via
a Feature Aggregation (FA) module. As shown in Figure 3,
log2 T representations are generated and supervised. Features
from both the spatial and temporal branches at each layer are
initially aggregated through the element-wise addition. FEMs
are then applied, followed by RCBs to generate the feature
FRn at the n-th layer. This feature are then fed into a global
average pooling (GAP) layer. The process can be formulated
as:
FRn = RCB(FEM(F1n + · · · + FLn )),
(8)
H

Rn =

W

1 1 XX n
fh,w ,
HW
w=1

MGH [1]
MG-RAFA [2]
NL-AP3D [5]
PSTA [3]
DenseIL [4]
RAPA [8]
SGWCNN [9]
MFA [10]
SINet [11]
CaViT [12]
PiT [13]
RGCN [14]
TMT [15]
TCViT [16]
Ours

MARS
Rank-1 mAP
90.0
85.8
88.8
85.9
90.7
85.6
91.5
85.8
90.8
87.0
88.7
82.8
90.0
85.7
90.9
86.2
91.0
86.2
90.8
87.2
90.2
86.8
91.1
86.5
91.8
86.5
91.7
87.6
91.9
87.2

(9)

h=1

n
where n and L are introduced in the overview section, fh,w
∈
C
n
R denotes the vector of FR at the position (h, w). Rn is the

generated representation at the n-th layer. Note that the spatialtemporal feature is processed without applying FEM. The
representation from the first layer is directly supervised, while
representations from subsequent layers and spatial-temporal
feature are concatenated with the previous layer’s representations. Concatenated features are then processed through a
temporal average pooling (TAP) layer to produce representations under supervision. The FA module refines features at
each layer for distinctive representation generation.

iLIDS-VID
Rank-1 Rank-5
85.6
97.1
88.6
98.0
88.7
91.5
98.1
92.0
98.0
89.6
87.8
96.0
93.3
99.3
92.5
93.3
98.0
92.1
98.9
90.2
98.5
91.3
98.6
94.3
99.3
94.0
99.3

Ours (31.4M)

PRID-2011
Rank-1
Rank-5
94.8
99.3
95.9
99.7
95.6
98.9
95.2
96.6
100
96.5
95.5
98.9
96.6
98.9

TMT (60.4M)

PSTA (35.4M)

NL-AP3D (31.6M)
SGWCNN (41.0M)

Fig. 4. Comparison of Rank-1 accuracy and model size between our ESTI
and state-of-the-art methods on the MARS dataset.

D. Loss Function
We use the cross-entropy loss Lcls with label smoothing
and the triplet loss Ltri with batch hard mining as our Re-ID
loss Lid :
Lid = Lcls + Ltri .

(10)

IV. E XPERIMENTS
A. Datasets and Evaluation Metrics
We evaluate our network on three challenging video-based
Re-ID datasets including MARS [18], iLIDS-VID [17] and
PRID-2011 [21]. The mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) curve are adopted
as our evaluation metrics.
B. Implementation Details
We sample 8 frames from the video sequence via RRS
strategy as input, each of them are resized to 256 × 128.
The ResNet-50 pre-trained on ImageNet is utilized as our
image backbone and we remove the last down-sampling operation. During training, random horizontal flipping and random
erasing are applied for data augmentation. Adam is used to
optimize our network with weight decay 5 × 10−4 . We set the
initial learning rate as 3.5 × 10−4 and follow the learning rate
decay strategy [22]. The epoch number is 500 in total. During
testing, the cosine similarity is used to measure the similarity
between query and gallery.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

TABLE II
A BLATION RESULTS OF COMPONENTS OF OUR PROPOSED NETWORK ON MARS DATASET INCLUDING S PATIAL BRANCH , T EMPORAL BRANCH ,
S PATIAL -T EMPORAL I NTERACTION (STI) AND F EATURE AGGREGATION (FA) MODULE .

Baseline
✓
✓
✓
✓
✓
✓
✓

Components
Spatial
Temporal
Branch
Branch
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓

MARS
STIM

FA

mAP

Rank-1

✓
✓

✓
✓

84.5
85.6
86.4
86.9
87.0
87.0
87.2

88.4
90.4
91.0
91.3
91.6
91.5
91.9

Parameters (M)

GFLOPs (G)

25.63
27.52
28.29
30.18
30.88
30.73
31.43

34.44
35,29
35.40
35.71
35.81
35.86
35.95

TABLE III

TABLE IV

A BLATION RESULTS OF DIFFERENT
METHODS FOR THE SPATIAL BRANCH
ON MARS.

A BLATION RESULTS OF DIFFERENT

Methods
Addition + RCBs
Self-Attention
Ours

Rank-2

Rank-3

Rank-4

Methods
Addition + RCBs
Cross-Attention
Ours

Rank-5

Ours

Baseline

Ours

Baseline

Query Feature Map Rank-1

MARS
mAP
Rank-1
86.5
91.0
86.7
91.3
87.2
91.9

METHODS FOR THE TEMPORAL
BRANCH ON MARS.

Fig. 5. Visualization of retrieve results and feature maps of query images
using the baseline and our ESTI. Images with green and red bounding boxes
indicate the correct and incorrect matches.

C. Comparison with State-of-the-art Methods
We compare the ESTI with recent state-of-the-art methods [1]–[5], [8]–[16] on MARS, iLIDS-VID and PRID-2011
datasets, as shown in Table I. Note that some methods involving cross-modality are excluded for a fair comparison. We
also indicate the efficient of the ESTI by comparing its model
size with them, as shown in Figure 4. Our ESTI achieves the
highest Rank-1 accuracy (91.9%) on the MARS dataset and the
highest Rank-5 accuracy (99.3%) on the iLIDS-VID dataset.
On the PRID-2011 dataset, our ESTI and MFA [10] share the
best Rank-1 accuracy. While we achieve the second-best mAP
on the MARS dataset and Rank-1 accuracy on iLIDS-VID, our
model size is significantly smaller than that of TCViT [16],
which attains the best mAP on MARS and the best Rank-1
accuracy on iLIDS-VID, since they use a vision transformer

MARS
mAP
Rank-1
86.3
90.7
86.8
91.4
87.2
91.9

TABLE V
A BLATION RESULTS OF THE F EATURE
AGGREGATION MODULE WITH AND
WITHOUT THE F EATURE
E NHANCEMENT M ODULE (FEM) ON
MARS.
Methods
w/o FEM
wo FEM

MARS
mAP
Rank-1
87.1
91.7
87.2
91.9

backbone with massive parameter size. As presented in Figure
4, our ESTI achieves a superior trade-off between efficiency
and effectiveness, obtaining the highest Rank-1 accuracy while
maintaining a smaller model size.
D. Ablation Study
1) Analysis of Different Components: We conduct ablation
experiments to validate the effectiveness of each component
in the ESTI on MARS by adding the component gradually, as
shown in Table II. Significant improvements are observed in
the 2-th and 3-th rows when the spatial and temporal branches
are employed individually. Specifically, the spatial branch
increases 1.1% mAP and 2.0% Rank-1 accuracy compare to
the baseline, while the temporal branch contributes a 1.9%
mAP and 2.6% Rank-1 accuracy improvement. Combining
both spatial and temporal branches achieve a 2.4% mAP and
2.9% Rank-1 accuracy increase over the baseline. We further
verify the effectiveness of the Spatial-Temporal Interaction
Module (STIM) at the final layer and the Feature Aggregation (FA) module. As shown in the in 5-th row, the FA
module provides an additional 0.1% mAP and 0.3% Rank-1
accuracy improvement. Incorporating the STIM boosts performance by 0.1% mAP and 0.2% Rank-1 accuracy. Eventually,
ESTI obtains the best overall performance by combining all
components. The small amount of additional parameter size
and computational complexity (FLOPs) introduced by each
component demonstrates the efficiency of our ESTI. These
results show that the ESTI effectively captures the interaction
of high-level spatial and temporal information while balancing
computational efficiency.
2) Analysis of Methods in the Spatial Branch: Table III
presents the experimental results of different enhancement
strategies, including feature addition with Residual Connection
Blocks (RCBs), Self-Attention and our method. Our method
outperforms the Self-Attention by 0.5% in mAP and 0.6%

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

in Rank-1 accuracy, and surpasses the addition with RCBs
by 0.7% in mAP and 0.9% in Rank-1 accuracy. This shows
that the Feature Enhancement Module (FEM) in the spatial
branch adaptively extract distinctive spatial dependencies by
emphasizing the target pedestrian within the spatial domain,
thereby enabling the generation of high-level spatial features.
3) Analysis of Methods in the Temporal Branch: Table IV
shows the experimental results of various interaction methods,
including feature addition with Residual Connection Blocks
(RCBs), Cross-Attention and the Temporal Interaction Module
(TIM). The TIM significantly exceeds the addition with RCBs
by 0.9% in mAP and 1.2% in Rank-1 accuracy. Furthermore, it
achieves a 0.4% mAP and 0.5% Rank-1 accuracy improvement
over the Cross-Attention mechanism. This demonstrates that
our TIM focuses on interactions between adjacent features,
modeling the temporal correlations effectively while capturing
both local and global dynamic details.
4) Analysis of FEM in the FA Module: We evaluate the
effectiveness of the FEM within the FA module, as shown in
Table V. The performance improvements of 0.1% mAP and
0.2 Rank-1 accuracy demonstrate that the FEM enhances the
pedestrian representations. Consequently, these enhanced representations are under supervision can better refine the features
generated from each layer, contributing to more discriminative
outputs.
E. Visualization
To further demonstrate the effectiveness of our proposed
ESTI, we visualize retrieve results and feature maps of query
images between the baseline and our network, as shown
in Figure 5. It can be observed that the ESTI significantly
highlights the focus on discriminative regions of the pedestrian
compared to the baseline. Moreover, the baseline tends to
retrieve negative samples with similar appearance, while our
ESTI consistently identifies the correct matches from Rank-1
to Rank-5. This demonstrates the superiority of ESTI in capturing distinctive spatial-temporal features, leading to improved
retrieval accuracy.
V. C ONCLUSION
This paper proposes an Efficient Spatial-Temporal Interaction (ESTI) network for video-based person Re-ID to mitigate
the redundancy computations and further prevent the loss of
spatial and temporal information. The network divides the
input sequence, utilizing half for the spatial branch and half
for the temporal branch. A Feature Enhancement Module
(FEM) and a Temporal Interaction Module (TIM) are designed for extract enhanced spatial features adaptively and
capture temporal correlations effectively. Features from each
layer are aggregated and supervised for improved training via
the Feature Aggregation (FA) module. We perform spatialtemporal interaction at the final layer to generate distinctive
representations. Our ESTI captures the interaction of spatial
and temporal information. Extensive experiments on three
challenging video Re-ID datasets demonstrate the efficiency
and effectiveness of our ESTI network.

R EFERENCES
[1] Yichao Yan, Jie Qin, Jiaxin Chen, Li Liu, Fan Zhu, Ying Tai, and Ling
Shao, “Learning multi-granular hypergraphs for video-based person reidentification,” in CVPR, 2020, pp. 2899–2908.
[2] Zhizheng Zhang, Cuiling Lan, Wenjun Zeng, and Zhibo Chen, “Multigranularity reference-aided attentive feature aggregation for video-based
person re-identification,” in CVPR, 2020, pp. 10407–10416.
[3] Yingquan Wang, Pingping Zhang, Shang Gao, Xia Geng, Hu Lu, and
Dong Wang, “Pyramid spatial-temporal aggregation for video-based
person re-identification,” in ICCV, 2021, pp. 12026–12035.
[4] Tianyu He, Xin Jin, Xu Shen, Jianqiang Huang, Zhibo Chen, and XianSheng Hua, “Dense interaction learning for video-based person reidentification,” in ICCV, 2021, pp. 1490–1501.
[5] Xinqian Gu, Hong Chang, Bingpeng Ma, Hongkai Zhang, and Xilin
Chen, “Appearance-preserving 3d convolution for video-based person
re-identification,” in ECCV. Springer, 2020, pp. 228–243.
[6] Chanho Eom, Geon Lee, Junghyup Lee, and Bumsub Ham, “Videobased person re-identification with spatial and temporal memory networks,” in ICCV, 2021, pp. 12016–12025.
[7] Shuang Li, Slawomir Bak, Peter Carr, and Xiaogang Wang, “Diversity regularized spatiotemporal attention for video-based person reidentification,” in CVPR, 2018, pp. 369–378.
[8] Guoqing Zhang, Yuhao Chen, Yang Dai, Yuhui Zheng, and Yi Wu,
“Reference-aided part-aligned feature disentangling for video person reidentification,” in ICME, 2021, pp. 1–6.
[9] Yingmao Yao, Xiaoyan Jiang, Hamido Fujita, and Zhijun Fang, “A
sparse graph wavelet convolution neural network for video-based person
re-identification,” Pattern Recognition, vol. 129, pp. 108708, 2022.
[10] Xinqian Gu, Hong Chang, Bingpeng Ma, and Shiguang Shan, “Motion
feature aggregation for video-based person re-identification,” IEEE
Transactions on Image Processing, vol. 31, pp. 3908–3919, 2022.
[11] Shutao Bai, Bingpeng Ma, Hong Chang, Rui Huang, and Xilin Chen,
“Salient-to-broad transition for video person re-identification,” in CVPR,
2022, pp. 7339–7348.
[12] Jinlin Wu, Lingxiao He, Wu Liu, Yang Yang, Zhen Lei, Tao Mei, and
Stan Z Li, “Cavit: Contextual alignment vision transformer for video
object re-identification,” in ECCV. Springer, 2022, pp. 549–566.
[13] Xianghao Zang, Ge Li, and Wei Gao, “Multidirection and multiscale
pyramid in transformer for video-based pedestrian retrieval,” IEEE
Transactions on Industrial Informatics, vol. 18, no. 12, pp. 8776–8785,
2022.
[14] Honghu Pan, Qiao Liu, Yongyong Chen, Yunqi He, Yuan Zheng, Feng
Zheng, and Zhenyu He, “Pose-aided video-based person re-identification
via recurrent graph convolutional network,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7183–
7196, 2023.
[15] Xuehu Liu, Pingping Zhang, Chenyang Yu, Xuesheng Qian, Xiaoyun
Yang, and Huchuan Lu, “A video is worth three views: Trigeminal transformers for video-based person re-identification,” IEEE Transactions on
Intelligent Transportation Systems, vol. 25, no. 9, pp. 12818–12828,
2024.
[16] Pengfei Wu, Le Wang, Sanping Zhou, Gang Hua, and Changyin
Sun, “Temporal correlation vision transformer for video person reidentification,” in AAAI, 2024, vol. 38, pp. 6083–6091.
[17] Taiqing Wang, Shaogang Gong, Xiatian Zhu, and Shengjin Wang,
“Person re-identification by video ranking,” in ECCV. Springer, 2014,
pp. 688–703.
[18] Liang Zheng, Zhi Bie, Yifan Sun, Jingdong Wang, Chi Su, Shengjin
Wang, and Qi Tian, “Mars: A video benchmark for large-scale person
re-identification,” in ECCV. Springer, 2016, pp. 868–884.
[19] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng
Dai, “Deformable detr: Deformable transformers for end-to-end object
detection,” arXiv preprint arXiv:2010.04159, 2020.
[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep
residual learning for image recognition,” in CVPR, 2016, pp. 770–778.
[21] Martin Hirzer, Csaba Beleznai, Peter M Roth, and Horst Bischof, “Person re-identification by descriptive and discriminative classification,”
in Image Analysis: 17th Scandinavian Conference, SCIA 2011, Ystad,
Sweden, May 2011. Proceedings 17. Springer, 2011, pp. 91–102.
[22] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang, “Bag of
tricks and a strong baseline for deep person re-identification,” in CVPR
Workshops, June 2019.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'FDGReID - Federated Domain Generalization for Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Machine Learning (2026) 115:22
https://doi.org/10.1007/s10994-025-06974-z

FDGReID: Federated Domain Generalization for Person Reidentification
Ke Niu1 · Haiyang Yu1,3 · Teng Fu1 · Mengyang Zhao1 · Bin Li1 · Xuelin Qian2 ·
Xiangyang Xue1
Received: 31 May 2025 / Revised: 17 September 2025 / Accepted: 18 December 2025 /
Published online: 13 January 2026
© The Author(s), under exclusive licence to Springer Science+Business Media LLC, part of Springer Nature 2026

Abstract
Person re-identification (Re-ID) has become a critical task in cross-camera retrieval systems. While deep learning-based approaches have made significant strides under controlled conditions, real-world deployment remains hindered by two major challenges: domain drift and data privacy. To address these challenges, we propose FDGReID, a novel
federated learning framework designed to achieve domain generalization in person Re-ID
without compromising user privacy. FDGReID introduces two core components: style
information sharing (SIS) and viewpoint-aware contrastive learning (VCL). SIS diversifies stylistic exposure among distributed clients by sharing style representations during
federated training, improving resilience to visual appearance changes. VCL, in contrast,
mitigates spatial viewpoint shifts by enforcing identity consistency via contrastive objectives across varied perspectives at each client. Together, these modules enable FDGReID
to learn robust, domain-invariant person representations without direct data exchange. We
conduct extensive experiments on widely-used cross-domain Re-ID benchmarks, demonstrating that FDGReID consistently outperforms existing federated and generalizable
Re-ID baselines. Moreover, it ensures strict data privacy compliance by keeping all raw
images localized. Our results highlight FDGReID’s effectiveness and practicality in building scalable, privacy-preserving Re-ID systems for real-world applications.
Keywords Person re-identification · Federated learning

1 Introduction
Person re-identification (Re-ID) (Zheng et al., 2017a, b) is a critical cross-camera retrieval
task with broad real-world applications in intelligent transportation, smart cities, and public
safety. The goal of Re-ID is to retrieve images of the same pedestrian captured by differ-

Ke Niu and Haiyang Yu contributed equally to this work
Editors: Chun-Yi Lee, Andy Song, Jhih-Ciang Wu, Hung Guei
Extended author information available on the last page of the article

13

22 Page 2 of 20

Machine Learning (2026) 115:22

ent non-overlapping cameras. In recent years, significant progress has been achieved, with
many state-of-the-art methods attaining remarkable accuracy on standard benchmarks.
Despite these advancements, deploying Re-ID systems in real-world environments
remains highly challenging. One of the most fundamental obstacles is the issue of domain
drift between the training and inference phases, which transforms the Re-ID task into a
zero-shot learning problem. This challenge stems from limited data availability and the
complexity of real-world operational environments. Specifically, while training datasets
are typically collected under controlled settings with pre-defined identities, practical Re-ID
systems cannot anticipate or pre-collect data for target identities at deployment time. Consequently, identities encountered during inference are entirely unseen. Furthermore, dynamic
factors such as illumination changes, diverse camera angles, and environmental conditions
exacerbate domain divergence, leading to significant degradation in model performance.
To tackle this, recent research has explored domain generalizable Re-ID (DG-ReID) (Choi
et al., 2021b; Ni et al., 2023), where models are trained across multiple source domains to
generalize to unseen target domains. However, most DG-ReID approaches rely heavily on
direct data or feature alignment, introducing privacy concerns–a critical barrier in sensitive
or regulated scenarios.
To address the privacy issue, federated learning (FL) (Zhuang et al., 2020) emerges as a
promising solution. FL enables collaborative model training across distributed clients without sharing raw data, thus preserving user privacy. While FL-based Re-ID methods primarily enhance training strategies (e.g., improving aggregation schemes), they often overlook
domain drift–resulting in biased local models. Aggregating these biased models without
addressing distributional shifts leads to suboptimal generalization in the global model.
Through our analysis of popular Re-ID datasets, we identify that domain drift arises
primarily from two factors: (1) stylistic discrepancies such as image tone and brightness,
and (2) viewpoint inconsistencies due to varied camera placements. As illustrated in Fig. 1,
datasets like iLIDs (Zheng et al., 2009) predominantly capture side-view images, while
GRID (Loy et al., 2010) suffers from color bias and severe distortion. These inconsistencies
hinder effective generalization across domains.

Fig. 1 Illustration of the heterogeneity across person Re-ID datasets

13

Machine Learning (2026) 115:22

Page 3 of 20 22

In this paper, we propose FDGReID, a novel decentralized Re-ID framework that integrates federated learning with domain generalization principles to simultaneously tackle
privacy and domain drift challenges. FDGReID introduces two key components: Style
Information Sharing (SIS): A mechanism that promotes stylistic diversity across federated clients by enabling early-layer style information exchange. This encourages models to
become more robust to varied visual styles. Viewpoint-aware Contrastive Learning (VCL):
A contrastive learning strategy that enforces viewpoint-invariant identity representations,
improving model robustness to spatial domain shifts. Extensive experiments across multiple
Re-ID benchmarks validate that FDGReID significantly enhances cross-domain generalization while safeguarding user data, demonstrating its practical value for real-world Re-ID
deployments. Our main contributions are summarized as follows:
● We propose a novel Style Information Sharing (SIS) module within the FL framework,
which explicitly increases style diversity among clients to mitigate style-related domain
drift.
● We introduce Viewpoint-aware Contrastive Learning (VCL), a client-side contrastive
learning approach that improves viewpoint invariance in person embeddings.
● We conduct comprehensive experiments showing that FDGReID achieves superior generalization performance on unseen domains while ensuring privacy preservation, making it well-suited for deployment in privacy-sensitive real-world scenarios.

2 Related Work
2.1 Supervised Person Re-ID
Most existing person re-identification (Re-ID) approaches adopt supervised learning paradigms, typically employing classification loss to derive discriminative global representations (Zheng et al., 2017a, b). Recent advancements have explored feature invariance
across various factors, including image resolution (Li et al., 2019a), human pose (Li et al.,
2019b), domain shift (Huang et al., 2019; Jin et al., 2020), and clothing changes (Qian et
al., 2020). However, viewpoint variation remains an under-addressed yet critical challenge
in person Re-ID. Filax and Ortmeier (2021) provide empirical evidence that differences in
camera viewpoints significantly impact the performance of metric-based learning models.
Similarly, Sun and Zheng (2019) highlight substantial performance degradation when query
and gallery images exhibit viewpoint discrepancies. Motivated by these observations, we
propose a Viewpoint-aware Contrastive Learning (VCL) module that explicitly tackles the
domain drift caused by viewpoint variations. VCL encourages identity consistency across
diverse viewpoints by enforcing contrastive objectives during local client training, thereby
enhancing the robustness of the learned representations in federated Re-ID settings.

2.2 Domain-Generalizable Re-ID
Existing domain generalizable (DG) person Re-ID approaches can be broadly categorized
into three main paradigms: (1) Meta-Learning: Choi et al. (2021b) simulate generalization
failures within a meta-learning framework, enabling normalization layers to adapt more

13

22 Page 4 of 20

Machine Learning (2026) 115:22

effectively to unseen domains. (2) Domain Alignment: Ni et al. (2023) propose Cross-ID
Similarity Learning (CSL), which extracts shared local visual cues across different identities
to learn domain-invariant representations. (3) Mixture of Experts: Dai et al. (2021) present
a decorrelation loss to preserve the distinctiveness and complementarity of each source
domain by decoupling their feature distributions. While these approaches improve crossdomain generalization, they typically rely on explicit representation alignment or data-level
transformations, which may conflict with privacy requirements in decentralized settings. In
contrast, our method operates within a federated learning framework and avoids direct data
sharing, making it inherently more privacy-preserving while still addressing domain drift.

2.3 Federated Learning for Person Re-ID
Federated Learning (FL) (Yang et al., 2019; Zhao et al., 2018; Kairouz et al., 2021) is a
decentralized machine learning paradigm that enables collaborative model training across
multiple edge devices while preserving the privacy of locally stored data. Its privacypreserving nature makes FL particularly suitable for sensitive applications such as person
re-identification (Re-ID), where direct data sharing is often infeasible due to ethical and
regulatory constraints. To facilitate research in this direction, Zhuang et al. (2020) introduced FedReID, a benchmarking framework for evaluating Re-ID models under federated
settings. Building on this, the same authors proposed FedUReID (Zhuang et al., 2021), the
first federated unsupervised Re-ID system, representing a significant step toward privacypreserving person Re-ID. Despite these advances, designing FL-based Re-ID models with
strong generalization capabilities remains a critical challenge. One of the key issues lies
in the aggregation of local models, where heterogeneous data distributions across clients
often lead to biased and suboptimal global models. This challenge is further compounded
by domain drift–stemming from variations in camera viewpoints and image styles–which
severely hampers generalization to unseen domains. Liu et al. (2024) attempted to mitigate data imbalance via dynamic aggregation weights, yet their method does not explicitly
address the domain shift problem in a domain generalization context. To bridge this gap,
we propose FDGReID, a novel federated learning framework that explicitly tackles domain
drift in person Re-ID by addressing style and viewpoint variations across decentralized data
sources.

3 Method
3.1 Overview
We propose FDGReID, a federated learning-based framework that integrates two novel
components–Style Information Sharing (SIS) and Viewpoint-aware Contrastive Learning
(VCL)–to jointly address domain drift and data privacy challenges in person re-identification (Re-ID). As illustrated in Fig. 2, FDGReID operates under a decentralized learning paradigm. In each FL round, the server distributes the global model to participating
clients. During local training, clients first invoke SIS (Sect. 3.2) to exchange style-relevant
features in a privacy-preserving manner, improving robustness to domain-specific appearance variations. Subsequently, each client trains with VCL (Sect. 3.3), which enforces

13

Page 5 of 20 22

Machine Learning (2026) 115:22

Server
Style Information Sharing mechanism

Client k

Client 2

Client 1

PAR

PAR

PAR
AFF

...

Backbone
Concat
Conv

AFF

AFF
Conv

PFE
Pose
Estimation

Refinement
module

RGAT

PFE

PFE

Fig. 2 An overview of the proposed FDGReID framework under the federated learning setting. PAR person appearance representation, PFE pose-guided feature embedding

identity consistency across viewpoints via a contrastive learning objective. SIS and VCL
are complementary: SIS handles appearance-related domain shifts, while VCL mitigates
spatial discrepancies due to camera perspectives. Together, they enhance robustness and
generalization across heterogeneous, unseen environments. We detail both modules in the
following sections and demonstrate their joint contributions to privacy-preserving, domaingeneralizable federated Re-ID.

3.2 Style Information Sharing Mechanism
The style information sharing (SIS) mechanism is designed to enhance cross-client style
diversity and mitigate domain drift caused by visual appearance discrepancies. Each clik
ent k computes a style descriptor from its local dataset Dk = {(xki , yik )}N
i=1 , summarizing
stylistic characteristics such as color distribution, brightness, and texture. These descriptors are then securely transmitted to the central server. At the beginning of each federated
round, the server distributes both the current global model parameters θt and the collected
style descriptors to all participating clients. Each client receives a style descriptor randomly
sampled from a different client, ensuring inter-client style augmentation without direct data
sharing. Upon receiving the redistributed descriptor, the client applies a style transfer operation to its local mini-batch, aligning its data to the newly assigned style. This cross-style
augmentation enhances the model’s robustness to diverse stylistic environments. To further
increase style diversity and reduce overfitting to limited local styles, we introduce a Style
Extrapolation technique. This strategy perturbs the received style descriptors to generate
extrapolated variations, enriching the stylistic space explored during training. Following
style augmentation, each client performs a local training step using its adapted dataset and
updates the local model parameters θkt+1 . These updates are then sent back to the server,
where they are aggregated to produce the new global model θt+1 . This process is repeated
iteratively until convergence. By decoupling style distribution from raw data and promoting
style diversity through inter-client transfer and extrapolation, SIS significantly improves the
generalization ability of the global model f to unseen test domains T. In what follows, we
detail the three key components of the SIS mechanism: Style Information Representation,

13

22 Page 6 of 20

Machine Learning (2026) 115:22

Selective Style-Shifting, and Style Extrapolation, each targeting the limitations of insufficient style variability in decentralized settings.

3.2.1 Style Information Representation
In convolutional neural networks (CNNs), the style of an image can be effectively captured by the channel-wise statistics of intermediate feature maps. Given a feature map
θ ∈ RC×H×W , where C, H, and W denote the number of channels, height, and width
respectively, we define the style descriptor using the channel-wise mean µ(θ) ∈ RC and
standard deviation σ(θ) ∈ RC as:
H

µc (θ) =

W

∑∑
1
θc,i,j 
H ×W

(1)

i=1 j=1




σc (θ) = 

H

W


1
2
(θc,i,j − µc (θ)) 
H ×W

(2)

i=1 j=1

where the subscript c indexes the feature channel. These statistics encode the global appearance properties (e.g., color tone and texture) of an image and are widely used as style
descriptors in image generation and domain adaptation.
For each client k, we compute its overall style representation by aggregating these
descriptors across its local dataset Sk , which contains ρk samples. The average channelwise mean µk and its element-wise variance Σk (µ) are computed as:
µk =

1 ∑
µ(ski )
ρk
ski ∈Sk

Σk (µ) =

1 ∑
2
(µ(ski ) − µk ) 
ρk
ski ∈Sk

(3)

(4)

Here, ski denotes the feature representation of the i-th sample in client k’s dataset, and the
squaring operation is applied element-wise. The resulting µk and Σk (µ) collectively form
the style descriptor for client k, which characterizes the statistical appearance distribution
across its local samples. This compact representation is used for inter-client style sharing in
our SIS mechanism.

3.2.2 Selective Style-Shifting
To promote inter-client stylistic diversity, each client k transmits its aggregated style statistics–channel-wise mean µk and variance Σk (µ)–to the server. The server then redistributes
these style descriptors such that each client receives a style profile from a different client
k ′ ̸= k . This cross-client style exchange effectively introduces unseen appearance variations
and mitigates the limited style diversity inherent in local datasets.

13

Page 7 of 20 22

Machine Learning (2026) 115:22

To selectively apply style transfer, we adopt a clustering-based strategy. For each minibatch of B samples, we apply k-means++ clustering on the extracted style vectors to identify
B/2 centroids representing the dominant style modes. The B/2 samples closest to these centroids are retained in their original style to preserve content consistency. The remaining B/2
samples are transformed via Adaptive Instance Normalization (AdaIN) (Huang & Belongie,
2017) using the received style descriptor:
AdaIN(s) = (σk′ + εσ · Σ′k (σ)) ·

(

s − µ(s)
σ(s)

)

+ (µ′k + εµ · Σ′k (µ))

(5)

Here, s denotes the original feature vector, while µ(s) and σ(s) are its channel-wise mean
and standard deviation. The target style parameters µ′k and σk′ are derived from the redistributed style descriptor. The perturbation terms εµ and εσ are sampled from a standard normal
distribution and scaled by the variance of the style descriptors Σ′k (·) to introduce controlled
randomness and further enhance stylistic diversity.
This selective style-shifting process allows clients to augment their local data distribution in a targeted and privacy-preserving manner, improving the generalization of the model
to stylistically diverse environments.

3.2.3 Style Extrapolation
To further enhance stylistic diversity and improve generalization, we introduce a Style
Extrapolation mechanism that adaptively perturbs the statistical style descriptors of feature representations. Specifically, we define an extrapolation function E(µ, σ, α), where
the extrapolation factor α controls the extent of deviation from the original style statistics.
Selecting an appropriate α is crucial, as it directly affects the degree of stylistic transformation and, consequently, the model’s performance.
Conventional style interpolation techniques typically rely on fixed statistical parameters (means and variances), neglecting the actual distributional structure of features in the
embedding space. This can limit the discriminative capability of the model in person Re-ID
scenarios. To address this issue, we propose a feature-aware dynamic extrapolation method
that tailors the extrapolation factor α based on pairwise distances between feature vectors in
a mini-batch, thereby capturing the intrinsic variability in style.
Given a mini-batch of features s̃n , we first compute the Euclidean distance between all
pairs of samples:


Dij = s̃in − s̃jn 
(6)
j∗
We then identify the pair of samples (s̃i∗
n , s̃n ) with the largest distance:
j∗
(s̃i∗
n , s̃n ) = arg max Dij 
i,j

(7)

This pair represents the most stylistically dissimilar samples in the mini-batch and serves
as a reference for dynamic extrapolation. We compute a normalized extrapolation factor as:

13

22 Page 8 of 20

Machine Learning (2026) 115:22

αij =

Dij − Dmin

Dmax − Dmin

(8)

where Dmin and Dmax are the minimum and maximum values in the distance matrix D,
respectively. This normalized factor ensures that style extrapolation is proportional to the
diversity observed in the current mini-batch.
Next, we use αij to extrapolate the style statistics of each sample s̃in . The updated channel-wise mean and standard deviation are computed as:

(

)

µnew (s̃in ) = µ(s̃in ) + αij · µ(s̃in ) − µn (sn ) 

(

)

σnew (s̃in ) = σ(s̃in ) + αij · σ(s̃in ) − σn (sn ) 

(9)
(10)

Here, µ(s̃in ) and σ(s̃in ) denote the original style statistics of sample s̃in , while µn (sn ) and
σn (sn ) represent the batch-wise statistics of the concatenated batch ŝn = [sn , s̃n ]. The
extrapolated statistics are then applied to transform the features via adaptive normalization.
By generating these extrapolated style features, the model is exposed to a broader range
of stylistic variations during training. This not only improves the robustness of the learned
representations but also enhances the model’s ability to generalize to previously unseen
domains exhibiting substantial style shifts.
The SIS module is designed to be lightweight and computationally efficient. It operates entirely at the feature level, where only channel-wise statistics–namely, the mean and
standard deviation–are computed for each instance. These statistics are then perturbed using
simple linear extrapolation to generate augmented style descriptors. The style-shifting is
implemented via AdaIN, which is applied directly to the intermediate feature maps and
incurs negligible overhead on modern edge GPUs.

3.3 Viewpoint-Aware Contrastive Learning
The Viewpoint-aware Contrastive Learning (VCL) module is designed to enhance the
model’s robustness to viewpoint variations–a key source of domain drift in person Re-ID.
It comprises two complementary branches: the Person Appearance Representation (PAR)
branch and the Pose-guided Feature Embedding (PFE) branch. The PAR branch focuses on
extracting rich appearance-based features that serve as stable and discriminative identity
cues. These features are especially valuable in federated learning (FL) settings, where local
data may exhibit highly diverse visual styles. In contrast, the PFE branch leverages human
pose estimation to encode skeletal shape representations, which are inherently invariant
to appearance factors such as clothing and lighting, thus offering complementary robustness across different viewpoints. To encourage the model to learn viewpoint-invariant yet
identity-discriminative embeddings, we introduce a viewpoint-aware contrastive loss that
minimizes intra-class variance across diverse viewpoints while maximizing inter-class
separability within the same viewpoint. This dual objective facilitates the learning of more
generalizable representations across camera angles and environments. Finally, we propose
an Adaptive Feature Fusion (AFF) strategy to effectively integrate appearance and pose
features. By dynamically weighting and merging the outputs from both branches, AFF pro-

13


 succeeded in 255ms:
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

19879

Find Hidden Modality Divergence: Adversarial
Aware Learning for Unsupervised Visible–Infrared
Person Re-Identification
Yuxuan Liu , Hongwei Ge , Yong Luo , and Chunguo Wu

Abstract—Unsupervised visible–infrared person re-identification (Unsupervised VI-ReID) aims to learn discriminative
identity features under the large modality gap without any
labeled data. Currently, the state-of-the-art methods optimize
cross-modality differences by using contrastive learning as the
underlying paradigm. However, they neglect the problem of
modality divergence during the cross-modality optimization process. This problem means that the interclass instances between
the cross-modality intraclass gaps can make cross-modality
intraclass instances difficult to get closer to each other in
the feature space due to the effect of contrastive learning on
these interclass instances. To alleviate the negative impact of
the modality divergence problem, we propose an adversarial
aware learning (ADAL) framework to explore the instances
that generate modal divergence and adversarially optimize these
explored instances. Specifically, on the one hand, we explore the
optimization directions of each cluster during the cross-modality
optimization process, and the cluster centroids generating
positive optimization are facilitated, while the others generating negative optimization are penalized. On the other hand,
we further consider the instance-level optimization process,
which increases the affinities of the positive instance pairs
with large cross-modality gaps to further improve the centroidlevel optimization. Extensive experiments conducted on the
visible–infrared person Re-ID datasets show that the proposed
method is used as a universally applicable plug-in module to add
the existing unsupervised VI-ReID methods, which outperforms
the existing state-of-the-art approaches.
Index Terms—Adversarial aware learning (ADAL), person reidentification (Re-ID), unsupervised learning, visible–infrared.

P

I. I NTRODUCTION
ERSON re-identification (Re-ID) aims at matching the
consistent pedestrians in different cameras by learning

Received 19 August 2024; revised 18 January 2025 and 10 May 2025;
accepted 13 July 2025. Date of publication 29 July 2025; date of current
version 31 October 2025. This work was supported in part by the National
Natural Science Foundation of China under Grant 61976034, in part by Dalian
Science and Technology Innovation Fund under Grant 2022JJ12GX013, and
in part by Liaoning Natural Science Foundation under Grant 2022-YGJC-20.
(Corresponding author: Hongwei Ge.)
Yuxuan Liu and Yong Luo are with the School of Computer Science and
Technology, Dalian University of Technology, Dalian 116024, China (e-mail:
lyx8880lzc@mail.dlut.edu.cn).
Hongwei Ge is with the School of Computer Science and Technology,
Dalian University of Technology, Dalian 116023, China, and also with the
Key Laboratory of Social Computing and Cognitive Intelligence, Ministry of
Education, Dalian University of Technology, Dalian 116024, China (e-mail:
hwge@dlut.edu.cn).
Chunguo Wu is with the Key Laboratory of Symbolic Computation and
Knowledge Engineering of Ministry of Education, College of Computer
Science and Technology, Jilin University, Changchun 130012, China.
Digital Object Identifier 10.1109/TNNLS.2025.3591116

diverse pedestrian feature representations [1], [2], [3], which
can serve as the continuous learning system apply in the intelligent surveillance and security [4], [5], [6]. Current approaches
consider that pedestrians are matched during the daytime by
intelligent surveillance systems and primarily depend on rich
visual texture information to address the single visible modality problems in person Re-ID. However, these single-modality
techniques have weak capabilities for retrieving persons under
poor lighting conditions, limiting their applicability in realworld surveillance scenarios.
Currently, a growing number of approaches have focused
on the visible–infrared person Re-ID tasks and made many
progresses [7], [8], [9]. However, the current visible–infrared
person re-identification (VI-ReID) methods are mainly trained
in supervised settings, which require large amounts of
labeled data. Since the identity annotations of visible to
infrared modality are more costly than single modal person Re-ID annotations, limiting the scalability of supervised
visible–infrared person Re-ID methods in real-world surveillance systems. Currently, unsupervised person Re-ID methods
[10], [11], [12] have attracted increasing attention. Current
unsupervised visible-modality person Re-ID methods widely
utilize cluster-based methods to obtain clustering results and
continuously optimize the clusters to generate precise pseudo
labels. These visible-modality methods can generate better
initial clustering results due to primarily depending on rich
clothes texture information, making the subsequent optimization process easier based on initial clustering results. In
contrast, it is more difficult to optimize intraclass variance on
unsupervised visible–infrared person Re-ID task [13], [14],
[15] because of the huge cross-modality intraclass gaps.
Due to the identity labels in two modalities being unseen
in the unsupervised settings, the state-of-the-art methods [14],
[15] address the unsupervised visible–infrared person Re-ID
problem by exploring cross-modality correspondences of
the same identity. However, there are huge cross-modality
intraclass gaps in the unsupervised VI-ReID, which will
undoubtedly negatively impact on exploring the cross-modality
correspondences of the same identity. Therefore, reducing
the cross-modality intraclass gaps through effective crossmodality optimization methods can further improve the
ability to explore cross-modality correspondences of the same
identity.
The existing unsupervised VI-ReID methods primarily use
contrastive learning to perform cross-modality optimization,

2162-237X © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19880

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

Fig. 1. Representation graph of motivation. Colors and shapes represent the pedestrians and modalities, respectively. The instances within the
black circle are denoted as representative cross-modality intraclass instances.
(a) Cross-modality intraclass instances are directly pulled closer to each other
during the training process, resulting in a smaller cross-modality intraclass
gap. (b) Feature distribution under the modality divergence problem, where
the interclass instances within the cross-modality gaps hinder cross-modality
intraclass instances from drawing closer, leading to a larger cross-modality
intraclass gap.

aiming to pull cross-modality intraclass instances closer while
pushing all interclass instances farther apart. Due to the
large cross-modality intraclass gaps, a considerable number
of interclass instances must exist between the cross-modality
intraclass gaps. In this case, the modality divergence problem
arises, meaning that these interclass instances between the
cross-modality intraclass gaps hinder cross-modality intraclass
instances from moving closer in the feature space. This occurs
due to the influence of contrastive learning on these interclass
instances, resulting in a larger cross-modality intraclass gap, as
shown in Fig. 1(b). In contrast, when the interclass instances
do not exist between the cross-modality intraclass gaps, the
intraclass instances are directly pulled closer to each other
by the effect of contrastive learning on positives, which
further reduces the cross-modality intraclass gaps, as shown
in Fig. 1(a). This effectively promotes modality alignment and
enhances robustness to cross-modality variations. Therefore,
such interclass instances between the cross-modality intraclass
gaps can deteriorate cross-modality intraclass instances from
converging in the feature space during the contrastive learning
process, which demonstrates the importance of the proposed
modality divergence problem in the unsupervised VI-ReID
task.
Based on the above motivation, we propose an adversarial
aware learning (ADAL) to alleviate the negative impact of
the modality divergence problem on unsupervised VI-ReID.
Specifically, as shown in Fig. 2(a), we explore the cluster
instances that contribute to negative and positive optimization. Negative optimization hinders cross-modality intraclass
clusters from converging, while positive optimization promotes
the alignment of intraclass clusters across visible and infrared
modalities. Therefore, we facilitate the positive optimization

Fig. 2.
Representation graph of our ADAL. We facilitate the positive
optimization process and penalizes the negative optimization process to
increase the cross-modality intraclass compactness. (a) Exploration process
in negative and positive optimization. (b) Adversarial optimization process
for negative and positive optimization.

process by further pushing these cluster instances away from
each other and penalize the negative optimization process by
pulling these cluster instances closer toward each other, as
shown in Fig. 2(b), which increases cross-modality intraclass
compactness.
However, the above cross-modality optimization process
only focuses on cluster-level optimization but is weak at
optimizing those hard positive instances at the cluster edges.
In general, the hard positive instances are the important
cause of generating large cross-modality intraclass variance.
Therefore, we further consider the instance-level optimization
process, increasing the intraclass affinities in large crossmodality gaps. Specifically, we merge the highest similarity
clusters in two modalities based on the similarity ranking of
cross-modality cluster centroids. Then, the easiest and hardest
positive instance pairs are explored from the merged clusters.
Since easy positive instance distribution can represent the most
compact degree of each cluster, we further enforce the model
to learn the ability to converge all cross-modality hard feature
distribution into easy feature distribution, increasing the crossmodality intraclass compactness.
We summarize our contributions as follows.
1) We raise a new modality divergence problem during
the cross-modality optimization process. To alleviate
the negative impact of the modality divergence problem on unsupervised VI-ReID, we propose an ADAL
framework, which can adversarially optimize the clusters
that generate the modality divergence, improving the
intraclass compactness of the cross-modality.
2) To further improve the cluster-level optimization process, we enforce the model to learn the ability that
converges all cross-modality hard positive feature distribution into easy positive feature distribution, which

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

increases the affinities of the intraclass instance pairs
with large cross-modality gaps.
3) We conduct extensive experiments in the large-scale
datasets to verify the effectiveness of the proposed
ADAL framework. Our framework achieves the state-ofthe-art performance on widely adopted VI-ReID person
datasets.
II. R ELATED W ORK
A. Visible–Infrared Person Re-ID
1) Supervised Visible–Infrared Person Re-ID: has recently
received increasing attention due to its potential for all-day
surveillance [16], [17], [18]. The key technical challenge is the
large domain gap between visible and infrared images from the
same individuals [19]. Wang et al. [20] introduce a dual-level
visible–infrared person Re-ID framework, which decomposes
the mixed discrepancies and handles them separately. Additionally, Liu et al. [21] propose a memory-augmented learning
framework, which learns and enhances cross-modality metrics. However, the supervised VI-ReID methods require large
amounts of labeled data in cross-modality, limiting the scalability of supervised methods in real-world deployments.
2) Unsupervised Visible–Infrared Person Re-ID: aims at
learning discriminative identity features under the large
modality gaps without any labeled data [13], [14], [22].
Liang et al. [13] first propose a two-stage method named
homogeneous-to-heterogeneous learning to address the unsupervised visible–infrared person Re-ID problem. Based on the
two-stage optimization method, the state-of-the-art unsupervised VI-ReID methods [14], [15] reduce huge cross-modality
intraclass gaps by finding cross-modality correspondences
of the same identity. Yang et al. [14] associate positive
cross-modality identities to learn the intramodality person
representation. Wu and Ye [15] formulate correspondence
mining as a graph-matching process to explore cross-modality
correspondences.
However, these approaches ignore the negative impact of the
huge cross-modality intraclass gaps on exploring the crossmodality correspondences of the same identity. The key to
reduce cross-modality intraclass gaps is addressing the modality divergence problem during the cross-modality optimization
process. The modality divergence problem means that simultaneously optimizing intra- and interclass gaps of cross-modality
will mutually generate negative optimization directions. The
proposed ADAL can effectively explore the instances that
generate modality divergence, alleviating the negative impact
of the modality divergence problem on unsupervised VI-ReID
by adversarially optimizing these explored instances.
B. Unsupervised Person Re-ID
The unsupervised learning method aims to learn discriminative feature representations without relying on any
labels. Current methods have emerged in the field of person
Re-ID research. Two interest areas in unsupervised person
Re-ID methods are unsupervised domain adaptation (UDA)
approaches [11], [23], [24] and fully unsupervised approaches
[10], [25], [26]. UDA-based approaches aim to minimize the

19881

domain gap by learning domain-invariant features from labeled
datasets and unlabeled target datasets. In contrast, fully unsupervised approaches can direct training on unlabeled target
datasets, enhancing the efficiency of real-world surveillance
systems. However, current fully unsupervised methods are
used for the single visible modality. Due to the large visible
and infrared modality differences, the existing unsupervised
methods are unsuitable for visible–infrared person Re-ID that
requires the learning of cross-modality invariant features in
the interference of the large modality differences.

C. Adversarial Learning Method for Person Re-ID
Adversarial learning is initially introduced in the generative
adversarial network (GAN) [27], which can produce realistic
images by training the generative model. This technique has
found applications in diverse tasks, such as domain adaptation
[28] and representation learning [29]. Over recent years, adversarial learning has been applied in the field of person Re-ID.
Gu et al. [30] propose a clothes-based adversarial loss, which
makes the trained clothing classifier unable to distinguish
between instances with the same identity and different clothes.
In contrast, our method aims to penalize the instances that
generate the negative optimization, which imposes the opposite
optimization direction on these negative instances to reduce
the cross-modality gaps. It is more suitable for unsupervised
visible–infrared person Re-ID tasks.
III. M ETHOD
A. Modality Divergence Problem Formulation
The modality divergence problem manifests as follows:
during the cross-modal optimization process, the interclass
instances in the cross-modal intraclass gaps can make crossmodal intraclass instances difficult to get closer to each other
in the feature space due to the effect of contrastive learning
on these interclass instances, thus negatively affecting the
modality alignment.
Specifically, given a pair of cross-modal intraclass instance
normalization features f (Xvi ) ∈ V and f (Xri ) ∈ R to be
optimized by contrastive learning, where V and R denote the
set of visible and infrared instance features, respectively. The
cross-modal intraclass gap can be defined by the following
formula:


Dintra = f Xvi − f Xri 2
(1)
where k · k2 denotes the Euclidean distance.
In the feature space, it is assumed that there exists an interclass instance feature f (X j ) ∈ R ∪ V between the cross-modal
intraclass instance pair f (Xvi ) and f (Xri ), and the interclass
instance feature (hardest negative instance) with minimum
Euclidean distance to the instance pair can be formulated as




f X j = arg min f Xvi − f (X) 2 + f Xri − f (X) 2
f (X)∈N

(2)
where N denotes the set of all negative instances. It is worth
noting that the above formula uses the similarity calculation
to define the hardest negative interclass instance that generates

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19882

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

the maximum divergence interference, which serves as an
example to intuitively display the modality divergence problem. However, the full instances that generated the modality
divergence problem require progressive exploration by using
our method and we propose the adversarial aware cross-modal
optimization process for these explored instances to alleviate
the modality divergence problem.
The cross-modal interclass gap can be defined by the
following formula:




Dinter = f Xvi − f X j 2 + f Xri − f X j 2 .
(3)
The cross-modal optimization process aims at continuously
reducing Dintra to improve cross-modal intraclass compactness
and simultaneously increasing Dinter to expand interclass differences. The optimization objective can be defined by the
following formula:
Finter = max ( Dinter )
Fintra = min ( Dintra ) .

(4)
(5)

In this case, the process of expanding interclass difference
can inevitably interfere with the process of improving the
cross-modal intraclass compactness, thus negatively affecting
the modality alignment. This phenomenon suggests that the
modality divergence problem is a key challenge in the unsupervised visible–infrared person Re-ID task.
B. Overall Framework
The proposed ADAL framework is illustrated in Fig. 3.
We first utilize the intramodality optimization module in
Section III-C as the baseline to learn intramodality discriminative features. Furthermore, we mainly raise a new modality
divergence problem during cross-modality optimization on the
unsupervised VI-ReID task, and our framework emphasizes
its novel cross-modality optimization module to alleviate the
negative impact generated by the modality divergence problem
in Section III-D.
Fig. 3(a) shows the pipeline of our framework. The visible
and infrared images are first inputted to the backbone for
feature extraction, and the clustering algorithm is used to
classify the extracted visible and infrared instance features,
generating the visible and infrared cluster instance features,
respectively. Then, we combine the visible and infrared cluster
instance features to obtain the joint cluster instance features,
which are input into the adversarial aware cross-modality
optimization (ADAC) module. Simultaneously, we calculate
the cluster centroids by taking the mean instance feature of
each cluster in the visible and infrared cluster instance features
to construct visible and infrared memory, which are then input
into the ADAC module.
Fig. 3(b) shows the ADAC module, which consists of an
upper and a lower part. In the upper part, we randomly
sample the joint cluster instance features to construct the minibatch, and the memory and guide memory are constructed
by fusing the instance features in each mini-batch and its
corresponding cluster centroids. We calculate the similarity
between the instance features with the memory and guide
memory, respectively, and the similarity difference values are

further calculated to determine the negative or positive memory. Finally, we calculate the adversarial aware optimization
loss. In the lower part, the visible and infrared cluster centroids
are matched to each other to calculate their corresponding
maximum similarity centroids. We then merge the matched
clusters and explore the hardest and easiest positive instances
in each merged cluster. Finally, we calculate the instance-level
cross-modality optimization loss.
C. Intramodality Optimization
At the beginning of each training epoch, the visible instance
features and infrared instance features are extracted by the
backbone network. Then, the clustering algorithm DBSCAN
[31] is used to, respectively, classify the extracted visible and
infrared instance features, generating the cluster instance features, as shown in Fig. 3. We further construct visible memory
Cv and infrared memory Cr based on the cluster instance
features to store the cluster centroids that are represented by
the mean instance feature of each cluster, respectively.
We randomly sample the infrared instance features f (Xr )
and the visible instance features f (Xv ) in a mini-batch. The
visible and infrared memory is updated by the corresponding
modality instance features, which are as follows:
Cv [i] ← αCv [i] + (1 − α) f (Xv )
 
 
Cr j ← αCr j + (1 − α) f (Xr )

(6)
(7)

where Cv [i] and Cr [ j] denote the ith and jth cluster centroid
in visible and infrared memory, respectively. α denotes the
memory updating rate.
The intramodality contrastive loss function can be identified
by the following formula:

exp h f (xr [i]) , Cr [i]i /τ
(8)
Lr = − log PPr
˝
 ˛ 
r
r
j /τ
j=1 exp f (x [i]) , C

exp h f (xv [i]) , Cv [i]i /τ
Lv = − log PPv
(9)
˝
 ˛ 
v
v
j /τ
j=1 exp f (x [i]) , C
where τ is a temperature hyperparameter [32], and Pv and Pr
denote the number of cluster centroids in visible and infrared
memory, respectively.
Moreover, for the visible and infrared modality, we use
the interclass similarity constraint loss Lvs and Lrs to further
extend the interclass gaps of the intramodality during the
intramodality optimization process, which is as follows:
D
E
2
1
−
Lvs =
FN f (xv [i]) , Cv
− Z0
(10)
F
Mv
D
E
2
 
1
−
Lrs =
FN f xr j , Cr
− Z0
(11)
F
Mr
where h·, ·i represents the inner product between two feature
vectors, which measures their similarity. FN() denotes the
normalization operation, which normalizes the input vector
according to the L2 norm to make the length of the vector
equal to 1, thus ensuring that the vectors have the same length.
Z 0 denotes the zero matrix. Mv and Mr represent the number
−
−
of elements in the matrix. Cv and Cr denote all the negative
cluster centroids without the ith or jth cluster centroid. The

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

19883

Fig. 3. ADAL framework for unsupervised visible–infrared person Re-ID. The ADAC module in the ADAL framework is proposed to explore the clusters that
generate negative and positive optimization by determining whether the instance features are near or far from each cluster centroid at the next iteration. Then,
the ADAC increases cross-modality intraclass compactness by further facilitating the positive optimization process and penalizing the negative optimization
process. Moreover, the ADAC explores the easiest and hardest positive instance pairs and further enforces the model to learn the ability that converges all
hard feature distributions into easy feature distributions, which further improves cross-modality intra-class compactness. (a) ADAL framework. (b) Adversarial
aware cross-modality optimization module.

total intramodality optimization loss function Lintra can be
identified by the following formula:
Lintra = Lr + Lv + θ (Lvs + Lrs )

(12)

where θ is used to control the degree of the intramodality
interclass similarity constraint. By adjusting θ, the model
can control the optimization of interclass similarity in the
intramodality, enhancing the ability to distinguish interclass
features while ensuring the compactness of intraclass features
in the intramodality. The total intramodality optimization loss
Lintra is used to optimize the intramodality instance feature
distribution.
D. Adversarial Aware Cross-Modality Optimization
Since directly reducing cross-modality intraclass gaps can
inevitably interfere with the intramodality instance feature

distributions, current approaches [13], [14] perform crossmodality optimization by contrastive learning to simultaneously improve intramodality and cross-modality intraclass
compactness, which jointly achieve the optimal intraclass
feature distributions. However, the modality divergence is an
inevitable problem in such a necessary optimization process,
which means that the interclass instances between the crossmodality intraclass gaps can make cross-modality intraclass
instances difficult to get closer to each other in the feature
space due to the effect of contrastive learning on these interclass instances, as shown in Fig. 1(b). Therefore, exploring and
optimizing such instances that produce modality divergence to
alleviate the negative impact of this problem is key to address
unsupervised VI-ReID tasks.
We first construct the guide visible memory Mv and guide
infrared memory Mr , which are updated by the features
extracted from the next iteration of the model, respectively.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19884

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

This process is as follows:
Mv [i] ← αMv [i] + (1 − α) f Xvt+1
 

 
Mr j ← αMr j + (1 − α) f Xrt+1


(13)
(14)

where t denotes the number of iterations for the current model
training. f (Xvt+1 ) and f (Xrt+1 ) denote the visible instance
features and infrared instance features extracted from the
model at the (t + 1) iteration, respectively. α represents the
memory updating rate.
Then, we, respectively, compute the similarity between the
extracted instance features at the tth iteration and each cluster
centroid in the cluster memory C at the tth iteration, and
the similarity between the above extracted instance features
at the tth iteration and each cluster centroid in the guide
memory M at the (t + 1)th iteration is further computed. The
similarity difference between the (t + 1)th and tth iteration can
be identified by the following formula:
˝

˛ ˝

˛
(15)
Dv = f Xvt , Mv − f Xvt , Cv
˝

˛ ˝
 r˛
r
r
r
r
D = f Xt , M − f Xt , C
(16)
where h·, ·i is the inner product to measure similarity. Dv and
Dr denote the similarity difference in the visible and infrared
streams, respectively. In that case, we can determine whether
the extracted instance features are near or far from each cluster
centroid at the next iteration by judging whether the similarity
difference values are positive or negative, which is as follows:
(
CNegative , Dv > 0
C=
(17)
CPositive , Dv < 0
where Dr is calculated in the same formula as Dv . This way
can filter which cluster centroids will generate negative and
positive optimization directions by determining whether the
instances are near or far from each cluster centroid at the next
iteration.
Since intraclass instances in two modalities will be close
to each other during the cross-modality optimization process,
once the extracted instances are close to other cluster centroids
at the next iteration, the cluster instances belonging to these
cluster centroids are between the cross-modality intraclass
gaps and produce negative optimization, as shown in the pink
area of Fig. 2(a). Conversely, once the extracted instances are
further away from other cluster centroids at the next iteration,
the cluster instances belonging to these cluster centroids are
outside of the cross-modality intraclass gaps and produce
positive optimization, as shown in the green area of Fig. 2(a).
We further facilitate the positive optimization process by
enlarging the distance between the instance features f (Xt )
and the cluster centroids CPositive that produces the positive
optimization, and the negative optimization is penalized by
reducing the distance between the instance features f (Xt )
and the cluster centroid CNegative that produces the negative
optimization. Such optimization process is as follows:
˝
˛

1
2
FN f Xvt , CPositive − Z 0 F
(18)
Lp =
Mp
˝

˛
1
2
Ln =
FN f Xvt , CNegative − Z 1 F
(19)
Mn

where the infrared stream is calculated in the same formula.
Z 0 denotes the zero matrix. Z 1 denotes the one matrix. FN()
denotes the normalization operation, which normalizes the
input vector according to the L2 norm to make the length
of the vector become 1, thus ensuring that the vectors have
the same length. k · k2F denotes Frobenius norm. M p and Mn
are the number of elements in the matrix.
The adversarial aware optimization loss Lao can be identified by the following formula:
Lao = ηL p + Ln

(20)

where η is the balance parameter between promoting positive
optimization and penalizing negative optimization in the crossmodality optimization process. Specifically, increasing the
value of η enables the model to focus more on pushing
instance features away from the cluster centroids that produce
positive optimization. In contrast, decreasing the value of η
can enable the model to pay more attention to reducing the
distance between instance features and the cluster centroids
that produce negative optimization.
However, the above cross-modality optimization process
only focuses on centroid-level optimization but neglects
interinstance affinities. In general, the hard positive instances
at the cluster edges in visible and infrared modality are the
important cause of generating large cross-modality intraclass
variance. Increasing the affinities of the positive instance pairs
with the large cross-modality gaps can further improve the
cross-modality intraclass compactness.
The ith cluster centroid Cv [i] as the query centroid is
sampled randomly in the visible memory, and the jth infrared
cluster centroid Cr [ j] with the maximum similarity to Cv [i] is
explored

 
(21)
Cr j = argmax Cv [i] · Cr
Cr

where Cr denotes the infrared memory containing all the
cluster centroids in the infrared modality.
We select the jth infrared cluster Sr [ j] indexed by the
explored jth cluster centroid Cr [ j], and we further merge Sr [ j]
with the visible cluster Sv [i] indexed by the cluster centroid
Cv [i]
 
S0 = Sv [i] ∪ Sr j .
(22)
Finally, we use the visible instance features f (Xv ) to explore
the easiest and hardest positive instances from the set S0

f (Xve ) = argmax f (Xv ) · S0
(23)
S0


f Xvh = argmin f (Xv ) · S0
(24)
S0

where the easiest positive instances are denoted as f (Xve )
and the hardest positive instances are denoted as f (Xvh ).
The instance-level cross-modality optimization loss Lco is as
follows:
 2
1
f (Xve ) − f Xvh F
(25)
Lco =
Mco
where k · k2F denotes the Frobenius norm. Mco is the number of elements in the matrix. Since easy positive instance
distribution can represent the most compact degree of each
cluster, the instance-level cross-modality optimization loss Lco

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

can enforce the model to learn the ability that converges all
hard feature distribution into easy feature distribution, which
further improves the cross-modality intraclass compactness.
The total cross-modality optimization loss function Lcross
combines Lao and Lco is as follows:
Lcross = Lao + βLco

The total complexity of both the proposed adversarial learning method and the baseline is as follows:
FLOPs (total) = (nc + T ) × Fr + (2 × d + 4) × nq × nc .
(32)

(26)

where β is used to adjust the weight of instance-level crossmodality optimization. Specifically, increasing the value of β
allows the model to focus more on optimizing the intraclass
compactness of cross-modality instances. In contrast, decreasing the value of β can enable the model to focus more on the
centroid-level optimization process, which aims to improve the
similarity between modalities to reduce the modality gaps.
The overall loss containing the intramodality optimization
loss function Lintra and the cross-modality optimization loss
function Lcross is as follows:
Ltotal = Lintra + Lcross .

19885

(27)

E. Analysis of Computational Complexity
The complexity of the proposed adversarial learning Lao
execution process is elaborated as follows: the computational
complexity of the backbone model that extracts a single picture
is denoted as Fr , and T denotes the number of training dataset
instances. Then, the complexity of the backbone is used to
represent the baseline in subsequent comparisons, which are
as follows:
FLOPs (baseline) = T × Fr .
(28)
Our adversarial learning process is divided into three
processes: instance memory construction process, sampling
process, and adversarial optimization process. In the instance
memory construction process, the number of cluster centroids
from instance memory is indicated by nc , which is calculated
by the mean of cluster instances in the training dataset. The
computational complexity is as follows:

FLOPs memory = nc × Fr .
(29)
For the process of instance sampling, we primarily explore
the similarity between the extracted features with the original
memory and the guide memory, respectively, and then, we
sample the negative cluster centroids and the positive cluster
centroids based on the similarity differences. Specifically, the
number of extracted features indicates nq and the extracted
feature dimension is denoted by d. Considering that multiplication, subtraction, and addition have participated in the
floating-point operation, respectively, the computational complexity of the sampling process is

FLOPs sampling = (2 × d + 1) × nq × nc .
(30)
In the adversarial optimization process, we perform the
adversarial learning process by simultaneously optimizing the
explored positive feature pairs and negative feature pairs.
The computational complexity of the adversarial optimization
process is

FLOPs optimization = 3 × nq × nc .
(31)

The computational complexity of the proposed adversarial
learning method is as follows:
FLOPs (adversarial) = nc × Fr + (2d + 4) × nq × nc

(33)

where the number of cluster nc is much smaller than the
number of instances T in the training dataset, and the second
half (2d + 4) × nq × nc is much smaller than the baseline
T × Fr because backbone model can produce the largest
complexity during the feature extraction. For this reason, the
proposed adversarial learning method has low computational
complexity, and the main computational complexity comes
from the backbone model in the extracting feature process.
Furthermore, the proposed adversarial learning method only
serves as a training strategy and does not participate in
testing. Thus, it does not degrade the speed of the inference
process.
IV. E XPERIMENTS
A. Dataset and Evaluation Metrics
1) Datasets Details: We utilize three extensively adopted
visible–infrared datasets, namely, SYSU-MM01 [44], [45],
RegDB [46], and LLCM [47].
2) SYSU-MM01 Dataset: The training set and test set of
the SYSU-MM01 dataset are strictly independent and have no
identity overlap. The training set contains 395 identities with
a total of 22 258 RGB images and 11 909 infrared images
captured by six cameras, including four RGB cameras and
two infrared cameras. The test set contains 96 identities with
301 RGB images and 3803 infrared images.
3) RegDB Dataset: The training set and test set of the
RegDB dataset are strictly independent and have no identity
overlap. The RegDB dataset contains a total of 412 identities,
and each identity contains ten visible light images and ten
infrared images. The RegDB dataset is randomly divided into
two parts: 206 identities for the training set with a total of
2060 RGB images and 2060 infrared images, and images of
the remaining 206 identities are used for the test set with a
total of 2060 RGB images and 2060 infrared images.
4) LLCM Dataset: The LLCM dataset is divided into a
training set and a testing set at a ratio of approximately 2:1,
which are strictly independent and have no identity overlap.
The training set contains 713 identities with a total of 16 946
person images from the visible modality and 13 975 person
images from the infrared modality, and the testing set contains
351 identities with a total of 8680 images from the visible
modality and 7166 images from the infrared modality.
5) Evaluation Metrics: This article adopts the cumulative matching characteristics (CMC), mean average precision
(mAP), mean inverse negative penalty (mINP) [39], and
modality divergence reduction rate (MDRR), where the quantitative metric process of MDRR is illustrated as follows.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19886

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

In the testing process, we sample visible and infrared
intraclass instance pairs f (Y vi ) and f (Y ri ) from the testing minibatch, and the cross-modal intraclass gap can be defined by
the following formula:


(34)
D0intra = f Y vi − f Y ri 2
where k · k2 denotes the Euclidean distance.
We explore the interclass centroid V j (hardest negative
centroid) with minimum Euclidean distance between the crossmodal intraclass instances f (Y vi ) and f (Y ri ). The class centroid
is calculated by the arithmetic mean of the class instances. The
explored interclass centroid can be defined by the following
formula:



(35)
V j = arg min f Y vi − V 2 + f Y ri − V 2
V∈U

where U denotes the set of all negative class centroids.
The cross-modal interclass gap under the modality divergence problem can be defined by the following formula:


(36)
D0inter = f Y vi − V j 2 + f Y ri − V j 2 .
Then, we calculate the metric formula for modality divergence. Since alleviating modality divergence problem mainly
lies in increasing D0inter without interfering with reducing
D0intra , the larger Dmodality indicates the smaller negative effect
of modality divergence problem. This process can be defined
by the following formula:
Dmodality = D0inter − D0intra .

(37)

Then, the MDRR is calculated by the following formula:
!
init
Dfinal
modality − Dmodality
(38)
Rmodality = mean
Dfinal
modality
PN
where mean(x) = (1/N) i=1
xi , and i denotes the index
value of the visible and infrared instance pair. N denotes the
number of the sampled test instance pairs. Dinit
modality represents
the distance difference at the start of training, and Dfinal
modality
represents the distance difference at the end of training. The
quantitative metric Rmodality measures the percentage of modality divergence reduction during the cross-modal optimization.
A high value of metric Rmodality indicates that the proposed
method effectively reduces modality divergence.
B. Implementation Details
Following AGW [39], the backbone contains a nonlocal
module enhanced network, which uses the ResNet50 [48]
pretrained on ImageNet [49] as the feature extractor. All
pedestrian images are resized to a consistent 288 × 144
resolution and apply channel augmentation [37] and random
horizontal flipping, erasing, and cropping. The model uses
the Adam [50] optimizer. Following [13], [14], the training
process is divided into two stages. The first stage executes
the intramodality optimization module in Section III-C for
50 training epochs, and ADAC module in Section III-D is
executed for 50 training epochs in the second stage, where
the interclass similarity constraint losses Lvs and Lrs in the
intramodality optimization module are deleted in the second

stage. We adopt DBSCAN [31] for pseudo-label assignment,
where the distance threshold is set 0.6 on SYSU-MM01, 0.3
on RegDB, and 0.6 on LLCM. The memory updating rate α
is set to 0.2. Parameter θ is set to 40 on SYSU-MM01 and
0.1 on RegDB. Parameter β is set to 1.3. The parameter η in
adversarial aware optimization loss is set to 0.5. In ablation
experiments, we perform ablation studies based on the results
of adding to the PGM [15].
C. Comparison With the State-of-the-Art Methods
We mainly raise a new modality divergence problem during
cross-modality optimization in the unsupervised VI-ReID task,
and our method lays emphasis on its novel cross-modality
optimization process. Furthermore, our method has universal applicability that can be used as a plug-in module to
add the existing unsupervised VI-ReID methods. We compare the recently developed supervised visible–infrared person
re-identification (supervised VI-ReID) methods and unsupervised visible–infrared person re-identification (unsupervised
VI-ReID) methods, as shown in Tables I–III.
1) Comparison With Unsupervised VI-ReID Methods: We
compare the proposed method with the six existing unsupervised VI-ReID methods, where the state-of-the-art methods
ADCA [14], PGM [15], and GUR [41] address the unsupervised visible–infrared person Re-ID problem by finding
cross-modality correspondences of the same identity. However,
there are huge cross-modality intraclass gaps in the unsupervised VI-ReID, which undoubtedly have a severe negative
impact on exploring the cross-modality correspondences of
the same identity. Furthermore, the state-of-the-art method
PCLHD [42] aims to explore the hard and dynamic prototypes
in data distributions and utilizes prototype-based contrastive
learning to preserve the intrinsic variety in sample features,
and SDCL [43] aims to focus on the collaboration of the
extracted shallow and deep features to capture modalityinvariant features and counter the cross-modality discrepancy.
In contrast, our ADAL framework emphasizes its novel crossmodality optimization process, which aims to alleviate the
negative impact of the modality divergence problem on unsupervised VI-ReID to reduce the huge cross-modality intraclass
gaps and thus further improve the exploration ability of
cross-modality correspondences. It can be seen that the proposed ADAL outperforms the state-of-the-art unsupervised
visible–infrared person Re-ID methods PCLHD and SDCL
when used as a plug-in module to add these state-of-the-art
methods.
Furthermore, we conduct the comparison experiment on the
LLCM dataset, as shown in Table III. Since all unsupervised
person VI-ReID methods use the SYSU-MM01 and RegDB
datasets, we reproduce the state-of-the-art and closely related
methods ADCA and PGM to compare with our ADAL. It can
be seen that our ADAL has universal applicability that can be
used as a plug-in module to add the existing unsupervised
VI-ReID methods to address the new modality divergence
problem in unsupervised VI-ReID and significantly surpasses
current unsupervised VI-ReID methods on the larger and
complex visible–infrared person dataset LLCM.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.


codex
前 8 页已经覆盖了 ESTI、π-VL、FNDS 的完整方法主体，也覆盖了 ADAL 和 FLAG 的核心方法主体。FDGReID 是长篇期刊稿，方法在第 8 页还没结束，我会继续读后续方法和消融部分，避免只按前 8 页截断造成误判。
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
/bin/zsh -lc "pdftotext -f 8 -l 13 'FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 8 -l 13 'FDGReID - Federated Domain Generalization for Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1808

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

TABLE II

TABLE III

P ERFORMANCE C OMPARISON OF O UR M ETHOD
W ITH S TATE - OF - THE -A RT M ETHODS

G ENERALIZATION E XPERIMENTS ON MGA

Parts & Pseudo Color, 6 represents Two Parts & Gray Continuous, 7 represents Three Parts & Gray Equal, 8 represents
Three Parts & Pseudo Color, 9 represents Three Parts &
Gray Continuous, 10 represents Four Parts & Gray Equal,
11 represents Four Parts & Pseudo Color, 12 represents Four
Parts & Gray Continuous, 13 represents Nineteen Parts &
Gray Equal, 14 represents Nineteen Parts & Pseudo Color,
15 represents Nineteen Parts & Gray Continuous.
B. Comparison With State-of-the-Art Methods
Table II shows the performance comparison of our method
with state-of-the-art methods. The MGA in Table II has
four branches (visible and three-granularities silhouettes), and
each branch uses AP3D to extract the corresponding features
explicitly. Except for the results of AP3D [9] and TCLNet [10]
which we reproduce, the rest results all come from [11], [46],
[47], [48], and [49].
Our MGA achieves state-of-the-art performance on all evaluation metrics and test settings. Compared with the existing
state-of-the-art method GEFF [46], our MGA outperforms it
by 4.2% and 2.8% for the Rank-1 metric on the general and
CC settings, respectively. In addition, our MGA significantly
improves the performance of existing methods (AP3D). Compared with the baseline model (AP3D), our MGA outperforms
it by 4.8% and 6.1% for the Rank-1 metric on the general and
CC settings, respectively.
We will explain why our MGA offers better results than
other methods and can improve the performance of existing
methods. As mentioned in Section I, the core in the VCC-ReID
is how to mine the limited remaining appearance information
(e.g., the human face) and the gait information. Existing
video-based ReID methods often only take visible light as
input. On the one hand, we cannot determine whether the
features extracted in this situation can contain both appearance and gait information. On the other hand, even though
both appearance and gait information can be extracted, their
decoupling is inadequate. However, our MGA, which can
combined with other existing methods, can simultaneously
take visible light and silhouette modalities as input and process
these modalities using separate branches, which can explicitly

extract appearance and gait features and achieve sufficient
decoupling.
C. Generalization Experiments on MGA
Table III shows the results of generalization experiments on
MGA. As mentioned in Section II, combining with AP3D [9]
and TCLNet [10] can comprehensively demonstrate the generality of our FLAG, so we use them as our baseline models
in Table III. As mentioned in Section III-B1, the number of
granularities can be set to a maximum of five, so we conduct
experiments with visible light and N-granularities (N = 1, 2, 3,
4, 5) silhouette modalities. The choice of different granularities
and forms will affect performance, and we use the choice
with the best performance to compare. For example, the best
choice in visible light and 3-granularities silhouette modalities
is using visible light, Single Part & Gray Equal silhouette,
Three Parts & Pseudo Color silhouette, and Four Parts &
Gray Equal silhouette. The ablation experiments on the choice
of granularities and forms will be shown in Section IV-E1.
We also compare the MGA with the single-modality models.
As shown in Table III, comparing our MGA with the singlemodality models, performance can be improved by 3.12%
∼ 3.6% and 2.64% ∼ 5.76% on Rank-1 with AP3D and
TCLNet, respectively. This means that no matter what the
value of N is in MGA, the performance can be significantly
improved on two baseline models, which can demonstrate the
generalization of our MGA. In addition, when the number
of granularities is set to three, the best performance can be
obtained. This demonstrates that silhouettes with more granularities can help the model capture more detailed gait features,
thereby improving performance. However, when the number

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

1809

TABLE IV
G ENERALIZATION E XPERIMENTS ON CMKDS

of granularities reaches the threshold, the gait information can
be comprehensively captured, and more granularities will add
more noise, leading to performance degradation.

TABLE V
P ERFORMANCE OF D IFFERENT C HOICES OF G RANULARITIES AND F ORMS

D. Generalization Experiments on CMKDS
Table IV shows the results of generalization experiments
on CMKDS. As mentioned in Section III-C, CMKDS aims to
distill the large MGA with the multi-modalities to the small
baseline model with the single modality. We compare the
improved baseline models (students) with CMKDS with the
original baseline models without CMKDS on AP3D [9] and
TCLNet [10]. When using CMKDS, we use the MGA with
visible Light & N-granularities (N = 1, 2, 3, 4, 5) silhouette
modalities as teachers. The teachers’ performance is shown in
Table III.
As shown in Table IV, compared with original baseline
models, the improved baseline models (students)’ performances are improved by 0.48% ∼ 1.2% and 0.24% ∼
0.84% on Rank-1 with AP3D and TCLNet, respectively.
The parameters, computational complexity (MACs), and input
modalities of original baseline models and improved baseline
models are the same. This demonstrates that our CMKDS
can improve baseline models’ performance without increasing any inference cost, which can be used on devices with
limited computing power. No matter what the value of N
is in MGA (teacher) when using CMKDS, the performance
can be improved on two baseline models, demonstrating the
generalization of CMKDS. In addition, when we use the MGA
with visible light & 3-granularities silhouette modalities as the
teacher in CMKDS, the improved baseline models can achieve
better performance. This is mainly because the MGA with
visible light & 3-granularities silhouette modalities can perform best. When other conditions are consistent, the student’s
performance largely depends on the teacher’s performance.
E. Ablation Study
In this section, we will conduct ablation experiments on the
choices of granularities and forms, EPFM, fusion strategy in
the MGA, and EDIDS in the CMKDS. We use AP3D [9] as
our baseline model in the following experiments.
1) Choices of Granularities and Forms: Table V shows the
ablation experiments on the choices of granularities and forms
in the MGA. We show the results of all the visible & singlegranularity Silhouette in Table V. The performance of visible

& single-granularity silhouette can reflect the complementarity
of 15 kinds of silhouettes (combinations of different granularities and forms) with visible light. So, we use this performance
as a basis to choose a combination of granularities and
forms when we use visible Light & N-granularities silhouette
(N = 2, 3, 4, 5) in the MGA. Specifically, for N granularities
(N = 2, 3, 4, 5), we choose the top N granularities with the
best performance of visible & single-granularity in the same
and different forms to conduct experiments.
As shown in Table V, the best choice in visible light &
single-granularity silhouette is visible light and Three Parts &
Pseudo Color, the best choice in visible light & twogranularities silhouette is visible light, Single Part & Gray
Equal, and Four Parts & Gray Equal, the best choice in visible
light & three-granularities silhouette is visible light, Single

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1810

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

TABLE VI

TABLE VIII

P ERFORMANCE OF E MBEDDING -BASED PARALLEL
F USION M ODULE AND OTHER F USION M ETHODS

P ERFORMANCE OF D IFFERENT F USION S TRATEGIES

TABLE IX
P ERFORMANCE OF D IFFERENT D ISTILLATION S TRATEGIES
TABLE VII
P ERFORMANCE OF PARALLEL AND S ERIAL D ESIGN IN THE
E MBEDDING -BASED PARALLEL F USION M ODULE

Part & Gray Equal, Three Parts & Pseudo Color, and Four
Parts & Gray Equal, the best choice in visible light & fourgranularities silhouette is visible light, Single Part & Gray
Continuous, Two Parts & Gray Continuous, Three Parts &
Gray Continuous, and Four Parts & Gray Continuous, the best
choice in visible light & five-granularities silhouette is visible
light, Single Part & Gray Equal, Two Parts & Gray Equal,
Three Parts & Gray Equal, and Four Parts & Gray Equal, and
Nineteen Parts & Gray Equal. In addition, the main-stream
silhouette in gait recognition is Single Part & Gray Equal;
when we use it in visible light & single-granularity, its
performance on Rank-1 is only 91.73. Our best choice of
visible light & N-granularities (N = 1, 2, 3, 4, 5) can exceed
it by 0.48% ∼ 1.68%. This can demonstrate the effectiveness
of our multi-granularities and forms.
2) Comparison of Embedding-Based Parallel Fusion Module With Other Fusion Methods: Table VI shows the
performance of EPFM and other multi-modal fusion methods.
Our EPFM is designed to fuse embedding, so the methods we compare are also proposed to fuse embedding. The
Embedding-Based fusion methods we compare are designed
to fuse two modalities, so we only compare them with our
EPFM on the MGA with visible light & single-granularity
silhouette.
As shown in Table VI, our EPFM can achieve the best
performance and exceed other methods by 0.72% ∼ 5.4% on
Rank-1, which can demonstrate the effectiveness of our EPFM.
3) Effectiveness of Parallel Design in the Embedding-Based
Parallel Fusion Module: Table VII shows the performance of
parallel and serial design in the EPFM. We use the parallel
fusion method in our MGA, which means parallel fuse N+1
branches’ embedding. The serial fusion method means we
first fuse N branches’ embedding (N-granularities gait) to the
whole gait embedding, then fuse the whole gait embedding
and appearance embedding. Parallel and serial fusion methods
are different only when using visible light & N-granularities

silhouette (N = 2, 3, 4, 5) in the MGA, so we conduct
experiments on them with the best choice of granularities and
forms.
As shown in Table VII, the parallel design is better than
the serial design, which can demonstrate the effectiveness
of the parallel design. This is mainly because the parallel
fusion method can better utilize the complementarity among
appearance and multi-granularities gait information.
4) Effectiveness of Fusion Strategy: Table VIII shows
the performance of different fusion strategies in the MGA.
As mentioned in Section III-B, our fusion strategy in the MGA
is that we first pre-train each branch on its corresponding
modality and then freeze each branch and train the EPFM.
We compare the other two fusion strategies in Table VIII.
They are each branch is not pre-trained and not frozen, and
each branch is pre-trained and not frozen.
As shown in Table VIII, our fusion strategy can perform
best, demonstrating our fusion strategy’s effectiveness in the
MGA. This is mainly because pre-training can ensure that
each branch can extract the appearance and multi-granularities
gait features of the corresponding modality, and freezing
can ensure that these features are still maintained during
training EPFM.
5) Effectiveness of Embedding-Based Direct and Indirect
Distillation Strategies: Table IX shows the performance of
different distillation strategies in the CMKDS. As mentioned
in Section III-C1, we use EDIDS in the CMKDS. The core of
EDIDS is the direct and indirect distillation strategies, so we
compare it with the strategies of only direct distillation and
only indirect distillation. We use the MGA with visible light
& N-granularities silhouette (N = 1, 2, 3, 4, 5) of best choice
on granularities and forms as teachers.
As shown in Table IX, no matter which MGA is used
as a teacher, our EDIDS can achieve the best performance,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

demonstrating our EDIDS’s effectiveness. With the dual
constraints of direct and indirect distillation, the student’s
embedding will approach the teacher’s embedding to the
greatest extent possible, thereby learning the appearance and
multi-granularities gait knowledge explicitly extracted by the
teacher.
V. A NALYSIS
In this section, we will analyze our method’s advantages
and limitations and provide future improvement plans for its
limitations. We will also provide an analysis of our method’s
real-world applications.
We will discuss our advantages over different dimensions
against various existing methods. First, in terms of the
effectiveness of feature extraction, our method can explicitly decouple and extract appearance and gait features by
inputting visible light and silhouette modalities and processing them with independent branches. In contrast, the
existing methods often only use visible light modalities as
input, which cannot fully decouple and extract appearance
and gait features. Second, In terms of interpretability, our
method uses independent branch processing visible light and
silhouette modalities; the semantic information contained in
the extracted features of each branch is interpretable. For
example, the features extracted by the silhouette branches
contain gait-related semantic information. Existing methods
that only take visible light as input cannot determine what
kind of features they extract. Third, in terms of generality,
our method can improve the performance of most existing
video-based person re-identification methods by forcing them
to explicitly extract appearance and gait information. However,
the existing methods often cannot be combined with other
methods. Finally, in terms of the diversity of models, our
method can provide a large MGA and a small model for
different computing power devices. However, existing methods
often only provide a model with a specific computational
complexity.
Our method also has limitations that need to be improved in
the future. The limitation of our work is the complexity of the
training process. Specifically, our FLAG includes MGA and
CMKDS. The MGA’s training process can be divided into two
stages. In the first stage, we will pre-train each branch of MGA
on its corresponding modality. In the second stage, we will
freeze each branch and train the EPFM proposed to parallel
fuse each modality’s video embedding. In the CMKDS, the
distillation process can also be divided into two stages. First,
we pre-train the MGA and baseline model. Second, we freeze
the MGA and distill the baseline model with our EDIDS.
The two-stage training processes of MGA and CMKDS are
complex. In the future, we will simplify the two-stage training
processes of MGA and CMKDS. For example, we can try to
train the MGA end-to-end with intermediate supervision to
each branch. We can also try to train the CMKDS with online
distillation instead of the offline distillation used now.
As mentioned in Section I, the VCC-ReID that our method
focuses on is practical in the real world. Our method has two
advantages in real-world applications. First, it can decouple
and extract appearance and gait features explicitly and achieve

1811

higher robustness and accuracy in real-world applications.
Second, our method designs two types of models that can
be applied to devices with different computing power. This
also helps our method be applied in the real world. When
applying our method in the real world, we first need to
confirm the device’s computing power. If the device has
powerful computing power, such as a cloud server, we can
use our MGA. We need to use the pedestrian detector and
tracker to get the pedestrian tracklets (visible light) first. Then,
we use a human parsing model to segment silhouettes from
the visible light pedestrian tracklets. Finally, we take visible
light and silhouette modalities tracklets as our MGA’s input.
If the device has limited computing power, such as edge
devices, we can use the small model in the CMKDS. In this
situation, we do not need the human parsing model; the person
re-identification model is also smaller than MGA.
VI. C ONCLUSION
In this paper, to tackle the challenge of the limited
appearance information due to clothes changing in the VCCReID, we propose a Framework with explicit Learning based
on Appearance and Gait (FLAG). The FLAG comprises a
multi-Modal and multi-Granularities Architecture (MGA) and
a Cross-Modal Knowledge Distillation Scheme (CMKDS).
The MGA is designed to simultaneously take the visible
light and multi-granularities silhouette modalities as input to
explicitly extract the appearance and multi-granularities gait
features, respectively. The MGA is large and needs more
modalities, so the CMKDS is designed to distill it to a small
model that only takes visible light as input. The appearance
and multi-granularities gait knowledge explicitly extracted by
the teacher can help the student improve performance. The
FLAG is suitable for devices with different computing power
through the large MGA and the small model from CMKDS.
In addition, our FLAG is generalizable, which means the
MGA and CMKDS can be combined with most existing
video-based ReID methods. Experimental results demonstrate
that the FLAG can help existing video-based ReID methods
improve performance on VCC-ReID. In addition, when FLAG
is combined with the AP3D [9], the MGA can outperform
state-of-the-art accuracy by 4.2%. Ablation experiments also
demonstrate the effectiveness of our MGA and CMKDS
designs. In summary, in addition to contributing to the development of the VCC-ReID, our method can also be understood
as integrating the two fields of video-based ReID and gait
recognition, which proposes a new direction for developing
these two fields.
R EFERENCES
[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2016, pp. 770–778.
[2] P. Li, Y. Xu, Y. Wei, and Y. Yang, “Self-correction for human parsing,”
IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 3260–3271,
Jun. 2022.
[3] J. Wu et al., “Neighbor-guided consistent and contrastive learning
for semi-supervised action recognition,” IEEE Trans. Image Process.,
vol. 32, pp. 2215–2227, 2023.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1812

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

[4] J. Shen and N. Robertson, “BBAS: Towards large scale effective
ensemble adversarial attacks against deep neural network learning,” Inf.
Sci., vol. 569, pp. 469–478, Aug. 2021.
[5] Z. Pang, C. Wang, L. Zhao, Y. Liu, and G. Sharma, “Cross-modality
hierarchical clustering and refinement for unsupervised visible-infrared
person re-identification,” IEEE Trans. Circuits Syst. Video Technol.,
vol. 34, no. 4, pp. 2706–2718, Apr. 2024.
[6] Y. Shi, Z. Wei, H. Ling, Z. Wang, J. Shen, and P. Li, “Person retrieval
in surveillance videos via deep attribute mining and reasoning,” IEEE
Trans. Multimedia, vol. 23, pp. 4376–4387, 2021.
[7] L. Wu et al., “Pseudo-pair based self-similarity learning for unsupervised person re-identification,” IEEE Trans. Image Process., vol. 31,
pp. 4803–4816, 2022.
[8] Z. Tang, R. Zhang, Z. Peng, J. Chen, and L. Lin, “Multi-stage spatiotemporal aggregation transformer for video person re-identification,”
IEEE Trans. Multimedia, vol. 25, pp. 7917–7929, 2023.
[9] X. Gu, H. Chang, B. Ma, H. Zhang, and X. Chen, “Appearancepreserving 3D convolution for video-based person re-identification,”
in Computer Vision–(ECCV). Cham, Switzerland: Springer, 2020,
pp. 228–243.
[10] R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen, “Temporal complementary learning for video person re-identification,” in Computer
Vision–(ECCV). Cham, Switzerland: Springer, 2020, pp. 388–405.
[11] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen,
“Clothes-changing person re-identification with RGB modality only,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2022,
pp. 1060–1069.
[12] M. Hirzer, C. Beleznai, P. M. Roth, and H. Bischof, “Person reidentification by descriptive and discriminative classification,” in Image
Analysis. Berlin, Germany: Springer, 2011, pp. 91–102.
[13] T. Wang, S. Gong, X. Zhu, and S. Wang, “Person re-identification
by video ranking,” in Computer Vision–(ECCV). Cham, Switzerland:
Springer, 2014, pp. 688–703.
[14] L. Zheng et al., “MARS: A video benchmark for large-scale person
re-identification,” in Computer Vision–(ECCV). Cham, Switzerland:
Springer, 2016, pp. 868–884.
[15] Y. Wu, Y. Lin, X. Dong, Y. Yan, W. Ouyang, and Y. Yang, “Exploit
the unknown gradually: One-shot video-based person re-identification
by stepwise learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., Jun. 2018, pp. 5177–5186.
[16] J. Li, S. Zhang, J. Wang, W. Gao, and Q. Tian, “Global-local temporal
representations for video person re-identification,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 3957–3966.
[17] P. Li, P. Pan, P. Liu, M. Xu, and Y. Yang, “Hierarchical temporal
modeling with mutual distance matching for video based person reidentification,” IEEE Trans. Circuits Syst. Video Technol., vol. 31, no. 2,
pp. 503–511, Feb. 2021.
[18] R. Hou, H. Chang, B. Ma, R. Huang, and S. Shan, “BiCnet-TKS:
Learning efficient spatial-temporal representation for video person reidentification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2021, pp. 2014–2023.
[19] D. Chen, H. Li, T. Xiao, S. Yi, and X. Wang, “Video person
re-identification with competitive snippet-similarity aggregation and coattentive snippet embedding,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit., Jun. 2018, pp. 1169–1178.
[20] N. McLaughlin, J. M. del Rincon, and P. Miller, “Video person reidentification for wide area tracking based on recurrent neural networks,”
IEEE Trans. Circuits Syst. Video Technol., vol. 29, no. 9, pp. 2613–2626,
Sep. 2019.
[21] J. Yang, W.-S. Zheng, Q. Yang, Y.-C. Chen, and Q. Tian, “Spatialtemporal graph convolutional network for video-based person reidentification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2020, pp. 3286–3296.
[22] Y. Yan et al., “Learning multi-granular hypergraphs for video-based
person re-identification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), Jun. 2020, pp. 2896–2905.
[23] Y. Wu, O. E. F. Bourahla, X. Li, F. Wu, Q. Tian, and X. Zhou, “Adaptive
graph representation learning for video person re-identification,” IEEE
Trans. Image Process., vol. 29, pp. 8821–8830, 2020.
[24] X. Li, W. Wang, Q. Li, and J. Zhang, “Spatial-temporal graph-guided
global attention network for video-based person re-identification,” Mach.
Vis. Appl., vol. 35, no. 1, p. 8, Jan. 2024.
[25] T. Zhang et al., “Spatiotemporal transformer for video-based person reidentification,” 2021, arXiv:2103.16469.

[26] T. He, X. Jin, X. Shen, J. Huang, Z. Chen, and X.-S. Hua, “Dense
interaction learning for video-based person re-identification,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 1490–1501.
[27] F. Yang, W. Li, B. Liang, and J. Zhang, “Spatiotemporal interaction
transformer network for video-based person reidentification in Internet
of Things,” IEEE Internet Things J., vol. 10, no. 14, pp. 12537–12547,
Jul. 2023.
[28] Z. Liu, Y. Wang, and A. Li, “Hierarchical integration of rich features for
video-based person re-identification,” IEEE Trans. Circuits Syst. Video
Technol., vol. 29, no. 12, pp. 3646–3659, Dec. 2019.
[29] R. Hou, B. Ma, H. Chang, X. Gu, S. Shan, and X. Chen, “IAUnet: Global
context-aware feature learning for person reidentification,” IEEE Trans.
Neural Netw. Learn. Syst., vol. 32, no. 10, pp. 4460–4474, Oct. 2021.
[30] C. Chen, M. Ye, M. Qi, J. Wu, Y. Liu, and J. Jiang, “Saliency and
granularity: Discovering temporal coherence for video-based person reidentification,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 9,
pp. 6100–6112, Sep. 2022.
[31] S. Bai, B. Ma, H. Chang, R. Huang, S. Shan, and X. Chen, “SANet:
Statistic attention network for video-based person re-identification,”
IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 6, pp. 3866–3879,
Jun. 2022.
[32] H. Fu, K. Zhang, H. Li, J. Wang, and Z. Wang, “Spatial temporal and
channel aware network for video-based person re-identification,” Image
Vis. Comput., vol. 118, Feb. 2022, Art. no. 104356.
[33] J. Leng, H. Wang, X. Gao, Y. Zhang, Y. Wang, and M. Mo,
“Where to look: Multi-granularity occlusion aware for video person reidentification,” Neurocomputing, vol. 536, pp. 137–151, Jun. 2023.
[34] X. Liu, P. Zhang, and H. Lu, “Video-based person re-identification with
long short-term representation learning,” in Image and Graphics. Cham,
Switzerland: Springer, 2023, pp. 55–67.
[35] J. Wang, Q. Zhao, D. Jia, Z. Huang, M. Zhang, and X. Ren, “Spatialtemporal aware network for video-based person re-identification,”
Multimedia Tools Appl., vol. 83, no. 12, pp. 36355–36373, Sep. 2023.
[36] Z. Chen, Z. Zhou, J. Huang, P. Zhang, and B. Li, “Frame-guided regionaligned representation for video person re-identification,” Proc. AAAI
Conf. Artif. Intell., Apr. 2020, vol. 34, no. 7, pp. 10591–10598.
[37] C. Eom, G. Lee, J. Lee, and B. Ham, “Video-based person reidentification with spatial and temporal memory networks,” in Proc.
IEEE Int. Conf. Comput. Vis., Jul. 2021, pp. 12036–12045.
[38] L. Liu, X. Yang, N. Wang, and X. Gao, “Viewing from frequency
domain: A DCT-based information enhancement network for video
person re-identification,” in Proc. 29th ACM Int. Conf. Multimedia,
New York, NY, USA, 2021, pp. 227–235.
[39] Z. Wang et al., “Robust video-based person re-identification by hierarchical mining,” IEEE Trans. Circuits Syst. Video Technol., vol. 32,
no. 12, pp. 8179–8191, Dec. 2022.
[40] F. Yang, X. Wang, X. Zhu, B. Liang, and W. Li, “Relation-based globalpartial feature learning network for video-based person re-identification,”
Neurocomputing, vol. 488, pp. 424–435, Jun. 2022.
[41] C. Zhang, P. Chen, T. Lei, Y. Wu, and H. Meng, “What-where-when
attention network for video-based person re-identification,” Neurocomputing, vol. 468, pp. 33–47, Jan. 2022.
[42] C. Gao, R. Yao, Y. Zhou, J. Zhao, L. Fang, and F. Hu, “Efficient lightweight video person re-identification with online difference
discrimination module,” Multimedia Tools Appl., vol. 81, no. 14,
pp. 19169–19181, Jun. 2022.
[43] H. Tao, Q. Duan, and J. An, “An adaptive interference removal framework for video person re-identification,” IEEE Trans. Circuits Syst. Video
Technol., vol. 33, no. 9, pp. 5148–5159, Sep. 2023.
[44] H. Pan, Y. Chen, and Z. He, “Multi-granularity graph pooling for
video-based person re-identification,” Neural Netw., vol. 160, pp. 22–33,
Mar. 2023.
[45] M. Kim, M. Cho, and S. Lee, “Feature disentanglement learning with
switching and aggregation for video-based person re-identification,” in
Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Jan. 2023,
pp. 1603–1612.
[46] D. Arkushin, B. Cohen, S. Peleg, and O. Fried, “GEFF: Improving any
clothes-changing person ReID model using gallery enrichment with face
features,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)
Workshops, Jan. 2024, pp. 152–162.
[47] Z. Cui, J. Zhou, Y. Peng, S. Zhang, and Y. Wang, “DCR-ReID: Deep
component reconstruction for cloth-changing person re-identification,”
IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 8, pp. 4415–4428,
Aug. 2023.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

[48] F. Liu, M. Kim, Z. Gu, A. Jain, and X. Liu, “Learning clothing
and pose invariant 3D shape representation for long-term person reidentification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV),
Oct. 2023, pp. 19617–19626.
[49] V. D. Nguyen, P. Mantini, and S. K. Shah, “Temporal 3D shape
modeling for video-based cloth-changing person re-identification,” in
Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV) Workshops,
Jan. 2024, pp. 173–182.
[50] A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for
person re-identification,” 2017, arXiv:1703.07737.
[51] J. Carreira and A. Zisserman, “Quo vadis, action recognition? A new
model and the kinetics dataset,” in Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jul. 2017, pp. 4724–4733.
[52] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural
networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
Jun. 2018, pp. 7794–7803.
[53] M. Wieczorek, B. Rychalska, and J. Dabrowski, “On the unreasonable
effectiveness of centroids in image retrieval,” in Neural Information
Processing. Cham, Switzerland: Springer, 2021, pp. 212–223.
[54] S. Shon, T. Oh, and J. Glass, “Noise-tolerant audio-visual online person
verification using an attention-based neural network fusion,” in Proc.
IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2019,
pp. 3995–3999.
[55] Y. Qian, Z. Chen, and S. Wang, “Audio-visual deep neural network for
robust person verification,” IEEE/ACM Trans. Audio, Speech, Language
Process., vol. 29, pp. 1079–1092, 2021.
[56] B. Wu, H. Lu, Z. Chen, C. Zhu, and S. Xu, “ERDBF: Embeddingregularized double branches fusion for multi-modal age estimation,”
IEEE Access, vol. 11, pp. 47608–47618, 2023.

Hengjie Lu received the B.Eng. degree in communication engineering from Shanghai University,
Shanghai, China, in 2020, where he is currently
pursuing the Ph.D. degree in information and
communication engineering. His research interests
include person re-identification, face attribute recognition, depth completion, and deep learning.

1813

Yilin Gao received the B.Eng. degree from
the Department of Communication Engineering,
Shanghai University, Shanghai, China, in 2021,
where he is currently pursuing the Ph.D. degree
with the School of Communication and Information Engineering. His research interests encompass
optical character recognition (OCR), facial analysis,
autonomous driving, and embodied intelligence.

Shugong Xu (Fellow, IEEE) received the Graduate
degree from Wuhan University, China, in 1990,
and the master’s degree in pattern recognition and
intelligent control and the Ph.D. degree in EE from
Huazhong University of Science and Technology
(HUST), China, in 1993 and 1996, respectively.
He is currently a Professor with Shanghai University.
He was the Center Director and an Intel Principal Investigator of the Intel Collaborative Research
Institute for Mobile Networking and Computing
(ICRI-MNC), prior to December 2016, where he
joined Shanghai University. Before joining Intel in September 2013, he was
a Research Director and a Principal Scientist with the Communication Technologies Laboratory, Huawei Technologies. He was also the Chief Scientist
and a PI of China National 863 Project on End-to-End Energy Efficient
Networks. He was one of the co-founders of the Green Touch consortium
together with Bell Laboratories. He served as the Co-Chair for the Technical
Committee for three terms in this international consortium. Prior to joining
Huawei in 2008, he was with the Sharp Laboratories of America as a Senior
Research Scientist. Before that, he conducted research as a Research Fellow
with the City College of New York, Michigan State University, and Tsinghua
University. He has published over 160 peer-reviewed research papers in top
international conferences and journals. He has over 50 patents granted. His
current research interests include machine learning, pattern recognition and
intelligent machine, and wireless communication systems. He was awarded
“National Innovation Leadership Talent” by China government in 2013, was
elevated to an IEEE Fellow in 2015 for contributions to the improvement of
wireless networks efficiency. He is also the Winner of the 2017 Award for
Advances in Communication from IEEE Communications Society.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.


 succeeded in 0ms:
22 Page 8 of 20

Machine Learning (2026) 115:22

αij =

Dij − Dmin

Dmax − Dmin

(8)

where Dmin and Dmax are the minimum and maximum values in the distance matrix D,
respectively. This normalized factor ensures that style extrapolation is proportional to the
diversity observed in the current mini-batch.
Next, we use αij to extrapolate the style statistics of each sample s̃in . The updated channel-wise mean and standard deviation are computed as:

(

)

µnew (s̃in ) = µ(s̃in ) + αij · µ(s̃in ) − µn (sn ) 

(

)

σnew (s̃in ) = σ(s̃in ) + αij · σ(s̃in ) − σn (sn ) 

(9)
(10)

Here, µ(s̃in ) and σ(s̃in ) denote the original style statistics of sample s̃in , while µn (sn ) and
σn (sn ) represent the batch-wise statistics of the concatenated batch ŝn = [sn , s̃n ]. The
extrapolated statistics are then applied to transform the features via adaptive normalization.
By generating these extrapolated style features, the model is exposed to a broader range
of stylistic variations during training. This not only improves the robustness of the learned
representations but also enhances the model’s ability to generalize to previously unseen
domains exhibiting substantial style shifts.
The SIS module is designed to be lightweight and computationally efficient. It operates entirely at the feature level, where only channel-wise statistics–namely, the mean and
standard deviation–are computed for each instance. These statistics are then perturbed using
simple linear extrapolation to generate augmented style descriptors. The style-shifting is
implemented via AdaIN, which is applied directly to the intermediate feature maps and
incurs negligible overhead on modern edge GPUs.

3.3 Viewpoint-Aware Contrastive Learning
The Viewpoint-aware Contrastive Learning (VCL) module is designed to enhance the
model’s robustness to viewpoint variations–a key source of domain drift in person Re-ID.
It comprises two complementary branches: the Person Appearance Representation (PAR)
branch and the Pose-guided Feature Embedding (PFE) branch. The PAR branch focuses on
extracting rich appearance-based features that serve as stable and discriminative identity
cues. These features are especially valuable in federated learning (FL) settings, where local
data may exhibit highly diverse visual styles. In contrast, the PFE branch leverages human
pose estimation to encode skeletal shape representations, which are inherently invariant
to appearance factors such as clothing and lighting, thus offering complementary robustness across different viewpoints. To encourage the model to learn viewpoint-invariant yet
identity-discriminative embeddings, we introduce a viewpoint-aware contrastive loss that
minimizes intra-class variance across diverse viewpoints while maximizing inter-class
separability within the same viewpoint. This dual objective facilitates the learning of more
generalizable representations across camera angles and environments. Finally, we propose
an Adaptive Feature Fusion (AFF) strategy to effectively integrate appearance and pose
features. By dynamically weighting and merging the outputs from both branches, AFF pro-

13

Page 9 of 20 22

Machine Learning (2026) 115:22

duces a unified embedding that captures both appearance consistency and geometric stability, enhancing the overall discriminability of the learned representation.

3.3.1 Pose-Guided Feature Embedding Branch (PFE)
The Pose-guided Feature Embedding (PFE) branch aims to capture fine-grained, appearance-invariant body shape features that are robust across different viewpoints. To this end,
we employ OpenPose (Cao et al., 2021), a state-of-the-art pose estimator widely used for
human pose estimation tasks. Given an input image x ∈ X , OpenPose outputs a set of p
body joints J = {ji }pi=1 , where each joint ji corresponds to a key anatomical landmark
represented by its 2D spatial coordinates (xi , yi ). These joints form a skeleton-based structural representation of the human body. To model both local part features and their relational structure, we adopt a Relational Graph Attention Network (RGAT) (Busbridge et
al., 2019) to embed the skeleton into a shape-aware feature space. RGAT is particularly
well-suited for this task due to two main advantages: (1) its attention mechanism adaptively assigns importance to each node, which is crucial as different body regions contribute
unequally to identity cues; and (2) it explicitly captures the semantic relationships among
joints, which enhances viewpoint robustness in body shape representation. The initial joint
features extracted by OpenPose are passed through a refinement module comprising fully
connected layers, resulting in enhanced node embeddings J ′ = {ji′ }pi=1 . These are stacked
into a matrix J ′ = [j1′ , . . . , jp′ ] ∈ Rp×F , where F is the feature dimension.
For each relation type r ∈ R, we apply a learnable linear transformation:
G(r) = J ′ W (r) ∈ Rp×F 
′

(11)
(r)

(r)

(r)

where W (r) ∈ RF ×F is a learnable weight matrix, and G(r) = [g1 , g2 , . . . , gp ] denotes
the intermediate node features under relation r.
The attention score between node i and node j under relation r is computed as:
′

(r)

(r)

αi,j = softmax(hi
(r)

(r)

where hi and hj
across all nodes.

(r)

· hj )

(12)

are projected node embeddings, and softmax ensures normalization

The updated embedding for node i is then computed via relational attention-based
aggregation:



h′i = σ 

∑ ∑

r∈R j∈Nir



′
(r) (r)
αi,j gj  ∈ RF 

(13)

Here, Nir denotes the set of neighbors of node i under relation r, and σ is an activation function (we use sigmoid in our implementation).
Finally, the global shape representation is obtained by averaging over all node
embeddings:

13

22 Page 10 of 20

Machine Learning (2026) 115:22
p

hshape =

1∑ ′
hi 
p

(14)

i=1

This compact vector hshape encodes the structural pose information of the individual and
is later fused with appearance-based features to construct a viewpoint-invariant identity
representation.

3.3.2 Person Appearance Representation (PAR)
The Person Appearance Representation (PAR) branch is responsible for extracting highlevel semantic features that capture the visual appearance of individuals. In federated
learning (FL) scenarios for person re-identification (Re-ID), appearance features serve as
relatively stable and discriminative identity cues, playing a crucial role in distinguishing
individuals under diverse environmental conditions.
We employ a CNN backbone to extract the appearance feature vector fa , which encodes
global visual attributes such as clothing, color, and texture. The Style Information Sharing
(SIS) mechanism is integrated into the early layers of the backbone to retain style-relevant
representations and enhance robustness to inter-client stylistic variations.
To address the domain drift induced by camera viewpoint changes, we introduce two contrastive learning objectives: the appearance viewpoint-aware contrastive loss Lappearance
CV
and the shape viewpoint-aware contrastive loss Lshape
.
These
losses
are
designed
to enforce
CV
viewpoint invariance while maintaining identity discrimination.
For each appearance feature fia ∈ F A , we define positive samples as those sharing the
same identity but captured from different viewpoints, and negative samples as those from
different identities but captured under the same viewpoint. The appearance viewpoint-aware
contrastive loss is formulated as:
∑
exp (dij /τ )
N
i
∑
j∈S
+
(15)
, with dij = d(fa (i), fa (j))
Lappearance
=−
log ∑
CV
exp (dik /τ )
i=1

k̸=i

i
where S+
denotes the set of positive samples for the i-th instance, d(·, ·) is a distance function, and τ is a temperature scaling parameter.
Analogously, to enforce viewpoint-aware consistency in shape space, we define the
shape viewpoint-aware contrastive loss Lshape
CV , applied to shape feature vectors fs (i). In
this setting, positive pairs consist of samples with the same viewpoint but different identities,
encouraging the model to learn shape features that are independent of viewpoint-specific
appearance. The loss is given by:
∑
exp(sij /τ )
N
i
∑
j∈S+
shape
(16)
, with sij = d(fs (i), fs (j))
LCV = −
log ∑
exp(sik /τ )
i=1

13

k̸=i

Page 11 of 20 22

Machine Learning (2026) 115:22

These two losses jointly enhance the model’s ability to disentangle identity-related information from viewpoint-dependent variations, contributing to more robust and generalizable
person representations in federated Re-ID settings.

3.3.3 Adaptive Feature Fusion (AFF)
We propose an Adaptive Feature Fusion (AFF) module to effectively integrate appearance
features fai and shape features fsi into a unified, discriminative representation. Given their
complementary nature–appearance features capturing visual semantics and shape features
offering viewpoint-invariant cues–adaptive fusion enables the model to leverage both
modalities dynamically based on their contextual importance. The fusion process begins by
projecting both feature types into a shared latent space using learnable transformation functions. Specifically, the transformed features fˆai and fˆsi are computed as:
fˆai = ϕa (fai ),

fˆsi = ϕs (fsi ),

(17)

where ϕa (·) and ϕs (·) are projection functions that normalize and align both feature vectors
within the same embedding space, facilitating effective interaction and comparison.
To adaptively determine the relative importance of each feature type, we introduce a
self-attention mechanism. The transformed features are concatenated and passed through
separate convolutional layers to compute attention weights:
wai = Conva (fˆai ∥fˆsi ),

wsi = Convs (fˆai ∥fˆsi ),

(18)

where ∥ denotes feature concatenation. The outputs wai and wsi represent the adaptive
importance of appearance and shape information, respectively, and are used to guide the
fusion process.
The final global feature representation fi for input image xi is computed via elementwise weighted summation:
fi = wai ⊙ fˆai + wsi ⊙ fˆsi ,

(19)

where ⊙ denotes element-wise multiplication. This fused representation fi captures both
identity-preserving visual cues and robust geometric structure.
The fused features F = {fi }N
i=1 are supervised by the identification loss LID , which
consists of a cross-entropy classification loss Lce and a pairwise triplet loss Ltri :
LID = Lce + Ltri .

(20)

Finally, the overall training objective integrates appearance and shape consistency across
viewpoints via the total loss:
L = LID + Lappearance
+ Lshape
CV
CV .

(21)

13

22 Page 12 of 20

Machine Learning (2026) 115:22

This unified optimization objective ensures that the model simultaneously achieves identity
discrimination, viewpoint invariance, and modality-aware fusion, leading to improved generalization in real-world federated Re-ID scenarios.
The Viewpoint-aware Contrastive Learning (VCL) module incorporates a lightweight
Relational Graph Attention Network (RGAT) to capture pose-structured identity information. The input to RGAT consists of compact 2D keypoint coordinates, typically extracted as
14 body joints per person. This results in a small graph with only 14 nodes, where attention
computation is limited to local joint interactions. Importantly, pose estimation using OpenPose is performed only once during the data preprocessing phase. Therefore, no runtime
pose inference is required during either training or inference.

4 Experiments
4.1 Experimental Setups
4.1.1 Datasets
We evaluate our proposed framework on nine publicly available person Re-ID datasets,
including: Market-1501 (Zheng et al., 2015), MSMT17 (Wei et al., 2018), CUHK02 (Li &
Wang, 2013), CUHK03 (Li et al., 2014), CUHK-SYSU (Xiao et al., 2016), PRID (Hirzer
et al., 2011), GRID (Loy et al., 2010), VIPeR (Gray & Tao, 2008), and iLIDs (Zheng et
al., 2009). These datasets vary significantly in terms of the number of identities, image
samples, camera views, and visual characteristics, reflecting diverse real-world scenarios. A
summary of their key attributes is provided in Table 1. Notably, we use the ’labeled’ subset
of CUHK03, following the protocol in Dai et al. (2021). For brevity, we use the following
abbreviations throughout the paper: MSMT17 as MS, Market-1501 as M, CUHK02 as C2,
CUHK03 as C3, and CUHK-SYSU as CS.
Table 1 Summary of all the
datasets

13

Datasets
Market1501
(M) Zheng et al.
(2015)
MSMT17 (MS)
Wei et al. (2018)
CUHK02 (C2) Li
and Wang (2013)
CUHK03 (C3) Li
et al. (2014)
CUHK-SYSU
(CS) Xiao et al.
(2016)
PRID Hirzer et al.
(2011)
GRID Loy et al.
(2010)
VIPeR Gray and
Tao (2008)
iLIDs Zheng et al.
(2009)

IDs
1501

Images
32,217

Cameras
6

Environment
Campus

4101

126,441

15

Campus

1816

7264

10

Campus

1467

14,096

2

Campus

11,934

34,574

1

Campus

749

949

2

Street

1,025

1275

8

632

1264

2

Underground
station
Street

300

4515

2

Airport

Page 13 of 20 22

Machine Learning (2026) 115:22

4.1.2 Hyperparameters
We selecte key hyperparameters such as the temperature parameter τ (set to 0.1), the style
extrapolation factor α (0.5), the batch size (64), and the learning rate (5 × 10−4 ) based on
widely adopted default settings in prior contrastive learning and federated learning literature, as well as preliminary tuning on a small validation subset.

4.1.3 Evaluation Protocols
Since the DukeMTMC-reID (Zheng et al., 2017) dataset has been prohibited for use in
academic research, despite its prior adoption in several FL Re-ID studies (Liu et al., 2024),
we establish a new evaluation framework tailored for FDGReID, as outlined in Table 2. In
Domain Generalization Protocol-1 (DG P1), model training is conducted using all training
images from multiple source domains. For evaluation, we adopt four target datasets–PRID,
GRID, VIPeR, and iLIDs–and report results as the average over 10 randomly generated
query/gallery splits. This protocol reflects typical domain generalization (DG) scenarios in
person Re-ID. Domain Generalization Protocol-2 (DG P2) extends DG P1 by incorporating
both training and testing images from the source domains, thereby increasing the training
data volume. This protocol is used to evaluate the robustness and scalability of the model
under data-rich source domains. In addition, we define a Federated Re-ID Protocol to simulate a fully supervised FL setting. All nine datasets are treated as separate clients. The final
global model, obtained through federated aggregation, is evaluated using mean Average
Precision (mAP) and Rank-1 accuracy. Performance results in this section are reported in
the format (Rank-1 / mAP) for consistency across experiments.

4.2 Implementation Details
We implement FDGReID in Python using the FedReID (Zhuang et al., 2023) framework,
which is built on PyTorch and employs ResNet-50 as the backbone network. The experiments are conducted in a federated learning setting with one central server and multiple clients, where each client is assigned a distinct dataset. These datasets originate from multiple
camera views, simulating realistic edge-camera deployments where data is locally stored
and processed. The server runs on an Intel(R) Xeon(R) Silver 4130 CPU, while each client
is executed on an individual NVIDIA GeForce RTX 4090 GPU. Model synchronization
is performed using PyTorch’s built-in communication backend for both aggregation and
parameter updates. In each training round, we evaluate both the local models and the aggregated global model. For final reporting, we record the best performance achieved on each
dataset throughout all training rounds. Unless otherwise specified, we adopt the following default hyperparameters: batch size B = 64, total training rounds R = 20, and training
Table 2 Evaluation protocols
DG
Protocol-1
DG
Protocol-2
Federated Fully
Supervised

Training Sets
M+C2+C3+CS

Testing Sets
PRID,GRID,VIPeR,iLIDs

Full-(M+C2+C3+CS)

PRID,GRID,VIPeR,iLIDs

M,MS,C3
PRID,VIPeR,iLIDs

M,MS,C3
PRID,VIPeR,iLIDs

13


exec
/bin/zsh -lc "pdftotext -f 4 -l 10 'Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
19882

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

the maximum divergence interference, which serves as an
example to intuitively display the modality divergence problem. However, the full instances that generated the modality
divergence problem require progressive exploration by using
our method and we propose the adversarial aware cross-modal
optimization process for these explored instances to alleviate
the modality divergence problem.
The cross-modal interclass gap can be defined by the
following formula:




Dinter = f Xvi − f X j 2 + f Xri − f X j 2 .
(3)
The cross-modal optimization process aims at continuously
reducing Dintra to improve cross-modal intraclass compactness
and simultaneously increasing Dinter to expand interclass differences. The optimization objective can be defined by the
following formula:
Finter = max ( Dinter )
Fintra = min ( Dintra ) .

(4)
(5)

In this case, the process of expanding interclass difference
can inevitably interfere with the process of improving the
cross-modal intraclass compactness, thus negatively affecting
the modality alignment. This phenomenon suggests that the
modality divergence problem is a key challenge in the unsupervised visible–infrared person Re-ID task.
B. Overall Framework
The proposed ADAL framework is illustrated in Fig. 3.
We first utilize the intramodality optimization module in
Section III-C as the baseline to learn intramodality discriminative features. Furthermore, we mainly raise a new modality
divergence problem during cross-modality optimization on the
unsupervised VI-ReID task, and our framework emphasizes
its novel cross-modality optimization module to alleviate the
negative impact generated by the modality divergence problem
in Section III-D.
Fig. 3(a) shows the pipeline of our framework. The visible
and infrared images are first inputted to the backbone for
feature extraction, and the clustering algorithm is used to
classify the extracted visible and infrared instance features,
generating the visible and infrared cluster instance features,
respectively. Then, we combine the visible and infrared cluster
instance features to obtain the joint cluster instance features,
which are input into the adversarial aware cross-modality
optimization (ADAC) module. Simultaneously, we calculate
the cluster centroids by taking the mean instance feature of
each cluster in the visible and infrared cluster instance features
to construct visible and infrared memory, which are then input
into the ADAC module.
Fig. 3(b) shows the ADAC module, which consists of an
upper and a lower part. In the upper part, we randomly
sample the joint cluster instance features to construct the minibatch, and the memory and guide memory are constructed
by fusing the instance features in each mini-batch and its
corresponding cluster centroids. We calculate the similarity
between the instance features with the memory and guide
memory, respectively, and the similarity difference values are

further calculated to determine the negative or positive memory. Finally, we calculate the adversarial aware optimization
loss. In the lower part, the visible and infrared cluster centroids
are matched to each other to calculate their corresponding
maximum similarity centroids. We then merge the matched
clusters and explore the hardest and easiest positive instances
in each merged cluster. Finally, we calculate the instance-level
cross-modality optimization loss.
C. Intramodality Optimization
At the beginning of each training epoch, the visible instance
features and infrared instance features are extracted by the
backbone network. Then, the clustering algorithm DBSCAN
[31] is used to, respectively, classify the extracted visible and
infrared instance features, generating the cluster instance features, as shown in Fig. 3. We further construct visible memory
Cv and infrared memory Cr based on the cluster instance
features to store the cluster centroids that are represented by
the mean instance feature of each cluster, respectively.
We randomly sample the infrared instance features f (Xr )
and the visible instance features f (Xv ) in a mini-batch. The
visible and infrared memory is updated by the corresponding
modality instance features, which are as follows:
Cv [i] ← αCv [i] + (1 − α) f (Xv )
 
 
Cr j ← αCr j + (1 − α) f (Xr )

(6)
(7)

where Cv [i] and Cr [ j] denote the ith and jth cluster centroid
in visible and infrared memory, respectively. α denotes the
memory updating rate.
The intramodality contrastive loss function can be identified
by the following formula:

exp h f (xr [i]) , Cr [i]i /τ
(8)
Lr = − log PPr
˝
 ˛ 
r
r
j /τ
j=1 exp f (x [i]) , C

exp h f (xv [i]) , Cv [i]i /τ
Lv = − log PPv
(9)
˝
 ˛ 
v
v
j /τ
j=1 exp f (x [i]) , C
where τ is a temperature hyperparameter [32], and Pv and Pr
denote the number of cluster centroids in visible and infrared
memory, respectively.
Moreover, for the visible and infrared modality, we use
the interclass similarity constraint loss Lvs and Lrs to further
extend the interclass gaps of the intramodality during the
intramodality optimization process, which is as follows:
D
E
2
1
−
Lvs =
FN f (xv [i]) , Cv
− Z0
(10)
F
Mv
D
E
2
 
1
−
Lrs =
FN f xr j , Cr
− Z0
(11)
F
Mr
where h·, ·i represents the inner product between two feature
vectors, which measures their similarity. FN() denotes the
normalization operation, which normalizes the input vector
according to the L2 norm to make the length of the vector
equal to 1, thus ensuring that the vectors have the same length.
Z 0 denotes the zero matrix. Mv and Mr represent the number
−
−
of elements in the matrix. Cv and Cr denote all the negative
cluster centroids without the ith or jth cluster centroid. The

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

19883

Fig. 3. ADAL framework for unsupervised visible–infrared person Re-ID. The ADAC module in the ADAL framework is proposed to explore the clusters that
generate negative and positive optimization by determining whether the instance features are near or far from each cluster centroid at the next iteration. Then,
the ADAC increases cross-modality intraclass compactness by further facilitating the positive optimization process and penalizing the negative optimization
process. Moreover, the ADAC explores the easiest and hardest positive instance pairs and further enforces the model to learn the ability that converges all
hard feature distributions into easy feature distributions, which further improves cross-modality intra-class compactness. (a) ADAL framework. (b) Adversarial
aware cross-modality optimization module.

total intramodality optimization loss function Lintra can be
identified by the following formula:
Lintra = Lr + Lv + θ (Lvs + Lrs )

(12)

where θ is used to control the degree of the intramodality
interclass similarity constraint. By adjusting θ, the model
can control the optimization of interclass similarity in the
intramodality, enhancing the ability to distinguish interclass
features while ensuring the compactness of intraclass features
in the intramodality. The total intramodality optimization loss
Lintra is used to optimize the intramodality instance feature
distribution.
D. Adversarial Aware Cross-Modality Optimization
Since directly reducing cross-modality intraclass gaps can
inevitably interfere with the intramodality instance feature

distributions, current approaches [13], [14] perform crossmodality optimization by contrastive learning to simultaneously improve intramodality and cross-modality intraclass
compactness, which jointly achieve the optimal intraclass
feature distributions. However, the modality divergence is an
inevitable problem in such a necessary optimization process,
which means that the interclass instances between the crossmodality intraclass gaps can make cross-modality intraclass
instances difficult to get closer to each other in the feature
space due to the effect of contrastive learning on these interclass instances, as shown in Fig. 1(b). Therefore, exploring and
optimizing such instances that produce modality divergence to
alleviate the negative impact of this problem is key to address
unsupervised VI-ReID tasks.
We first construct the guide visible memory Mv and guide
infrared memory Mr , which are updated by the features
extracted from the next iteration of the model, respectively.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19884

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

This process is as follows:
Mv [i] ← αMv [i] + (1 − α) f Xvt+1
 

 
Mr j ← αMr j + (1 − α) f Xrt+1


(13)
(14)

where t denotes the number of iterations for the current model
training. f (Xvt+1 ) and f (Xrt+1 ) denote the visible instance
features and infrared instance features extracted from the
model at the (t + 1) iteration, respectively. α represents the
memory updating rate.
Then, we, respectively, compute the similarity between the
extracted instance features at the tth iteration and each cluster
centroid in the cluster memory C at the tth iteration, and
the similarity between the above extracted instance features
at the tth iteration and each cluster centroid in the guide
memory M at the (t + 1)th iteration is further computed. The
similarity difference between the (t + 1)th and tth iteration can
be identified by the following formula:
˝

˛ ˝

˛
(15)
Dv = f Xvt , Mv − f Xvt , Cv
˝

˛ ˝
 r˛
r
r
r
r
D = f Xt , M − f Xt , C
(16)
where h·, ·i is the inner product to measure similarity. Dv and
Dr denote the similarity difference in the visible and infrared
streams, respectively. In that case, we can determine whether
the extracted instance features are near or far from each cluster
centroid at the next iteration by judging whether the similarity
difference values are positive or negative, which is as follows:
(
CNegative , Dv > 0
C=
(17)
CPositive , Dv < 0
where Dr is calculated in the same formula as Dv . This way
can filter which cluster centroids will generate negative and
positive optimization directions by determining whether the
instances are near or far from each cluster centroid at the next
iteration.
Since intraclass instances in two modalities will be close
to each other during the cross-modality optimization process,
once the extracted instances are close to other cluster centroids
at the next iteration, the cluster instances belonging to these
cluster centroids are between the cross-modality intraclass
gaps and produce negative optimization, as shown in the pink
area of Fig. 2(a). Conversely, once the extracted instances are
further away from other cluster centroids at the next iteration,
the cluster instances belonging to these cluster centroids are
outside of the cross-modality intraclass gaps and produce
positive optimization, as shown in the green area of Fig. 2(a).
We further facilitate the positive optimization process by
enlarging the distance between the instance features f (Xt )
and the cluster centroids CPositive that produces the positive
optimization, and the negative optimization is penalized by
reducing the distance between the instance features f (Xt )
and the cluster centroid CNegative that produces the negative
optimization. Such optimization process is as follows:
˝
˛

1
2
FN f Xvt , CPositive − Z 0 F
(18)
Lp =
Mp
˝

˛
1
2
Ln =
FN f Xvt , CNegative − Z 1 F
(19)
Mn

where the infrared stream is calculated in the same formula.
Z 0 denotes the zero matrix. Z 1 denotes the one matrix. FN()
denotes the normalization operation, which normalizes the
input vector according to the L2 norm to make the length
of the vector become 1, thus ensuring that the vectors have
the same length. k · k2F denotes Frobenius norm. M p and Mn
are the number of elements in the matrix.
The adversarial aware optimization loss Lao can be identified by the following formula:
Lao = ηL p + Ln

(20)

where η is the balance parameter between promoting positive
optimization and penalizing negative optimization in the crossmodality optimization process. Specifically, increasing the
value of η enables the model to focus more on pushing
instance features away from the cluster centroids that produce
positive optimization. In contrast, decreasing the value of η
can enable the model to pay more attention to reducing the
distance between instance features and the cluster centroids
that produce negative optimization.
However, the above cross-modality optimization process
only focuses on centroid-level optimization but neglects
interinstance affinities. In general, the hard positive instances
at the cluster edges in visible and infrared modality are the
important cause of generating large cross-modality intraclass
variance. Increasing the affinities of the positive instance pairs
with the large cross-modality gaps can further improve the
cross-modality intraclass compactness.
The ith cluster centroid Cv [i] as the query centroid is
sampled randomly in the visible memory, and the jth infrared
cluster centroid Cr [ j] with the maximum similarity to Cv [i] is
explored

 
(21)
Cr j = argmax Cv [i] · Cr
Cr

where Cr denotes the infrared memory containing all the
cluster centroids in the infrared modality.
We select the jth infrared cluster Sr [ j] indexed by the
explored jth cluster centroid Cr [ j], and we further merge Sr [ j]
with the visible cluster Sv [i] indexed by the cluster centroid
Cv [i]
 
S0 = Sv [i] ∪ Sr j .
(22)
Finally, we use the visible instance features f (Xv ) to explore
the easiest and hardest positive instances from the set S0

f (Xve ) = argmax f (Xv ) · S0
(23)
S0


f Xvh = argmin f (Xv ) · S0
(24)
S0

where the easiest positive instances are denoted as f (Xve )
and the hardest positive instances are denoted as f (Xvh ).
The instance-level cross-modality optimization loss Lco is as
follows:
 2
1
f (Xve ) − f Xvh F
(25)
Lco =
Mco
where k · k2F denotes the Frobenius norm. Mco is the number of elements in the matrix. Since easy positive instance
distribution can represent the most compact degree of each
cluster, the instance-level cross-modality optimization loss Lco

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

can enforce the model to learn the ability that converges all
hard feature distribution into easy feature distribution, which
further improves the cross-modality intraclass compactness.
The total cross-modality optimization loss function Lcross
combines Lao and Lco is as follows:
Lcross = Lao + βLco

The total complexity of both the proposed adversarial learning method and the baseline is as follows:
FLOPs (total) = (nc + T ) × Fr + (2 × d + 4) × nq × nc .
(32)

(26)

where β is used to adjust the weight of instance-level crossmodality optimization. Specifically, increasing the value of β
allows the model to focus more on optimizing the intraclass
compactness of cross-modality instances. In contrast, decreasing the value of β can enable the model to focus more on the
centroid-level optimization process, which aims to improve the
similarity between modalities to reduce the modality gaps.
The overall loss containing the intramodality optimization
loss function Lintra and the cross-modality optimization loss
function Lcross is as follows:
Ltotal = Lintra + Lcross .

19885

(27)

E. Analysis of Computational Complexity
The complexity of the proposed adversarial learning Lao
execution process is elaborated as follows: the computational
complexity of the backbone model that extracts a single picture
is denoted as Fr , and T denotes the number of training dataset
instances. Then, the complexity of the backbone is used to
represent the baseline in subsequent comparisons, which are
as follows:
FLOPs (baseline) = T × Fr .
(28)
Our adversarial learning process is divided into three
processes: instance memory construction process, sampling
process, and adversarial optimization process. In the instance
memory construction process, the number of cluster centroids
from instance memory is indicated by nc , which is calculated
by the mean of cluster instances in the training dataset. The
computational complexity is as follows:

FLOPs memory = nc × Fr .
(29)
For the process of instance sampling, we primarily explore
the similarity between the extracted features with the original
memory and the guide memory, respectively, and then, we
sample the negative cluster centroids and the positive cluster
centroids based on the similarity differences. Specifically, the
number of extracted features indicates nq and the extracted
feature dimension is denoted by d. Considering that multiplication, subtraction, and addition have participated in the
floating-point operation, respectively, the computational complexity of the sampling process is

FLOPs sampling = (2 × d + 1) × nq × nc .
(30)
In the adversarial optimization process, we perform the
adversarial learning process by simultaneously optimizing the
explored positive feature pairs and negative feature pairs.
The computational complexity of the adversarial optimization
process is

FLOPs optimization = 3 × nq × nc .
(31)

The computational complexity of the proposed adversarial
learning method is as follows:
FLOPs (adversarial) = nc × Fr + (2d + 4) × nq × nc

(33)

where the number of cluster nc is much smaller than the
number of instances T in the training dataset, and the second
half (2d + 4) × nq × nc is much smaller than the baseline
T × Fr because backbone model can produce the largest
complexity during the feature extraction. For this reason, the
proposed adversarial learning method has low computational
complexity, and the main computational complexity comes
from the backbone model in the extracting feature process.
Furthermore, the proposed adversarial learning method only
serves as a training strategy and does not participate in
testing. Thus, it does not degrade the speed of the inference
process.
IV. E XPERIMENTS
A. Dataset and Evaluation Metrics
1) Datasets Details: We utilize three extensively adopted
visible–infrared datasets, namely, SYSU-MM01 [44], [45],
RegDB [46], and LLCM [47].
2) SYSU-MM01 Dataset: The training set and test set of
the SYSU-MM01 dataset are strictly independent and have no
identity overlap. The training set contains 395 identities with
a total of 22 258 RGB images and 11 909 infrared images
captured by six cameras, including four RGB cameras and
two infrared cameras. The test set contains 96 identities with
301 RGB images and 3803 infrared images.
3) RegDB Dataset: The training set and test set of the
RegDB dataset are strictly independent and have no identity
overlap. The RegDB dataset contains a total of 412 identities,
and each identity contains ten visible light images and ten
infrared images. The RegDB dataset is randomly divided into
two parts: 206 identities for the training set with a total of
2060 RGB images and 2060 infrared images, and images of
the remaining 206 identities are used for the test set with a
total of 2060 RGB images and 2060 infrared images.
4) LLCM Dataset: The LLCM dataset is divided into a
training set and a testing set at a ratio of approximately 2:1,
which are strictly independent and have no identity overlap.
The training set contains 713 identities with a total of 16 946
person images from the visible modality and 13 975 person
images from the infrared modality, and the testing set contains
351 identities with a total of 8680 images from the visible
modality and 7166 images from the infrared modality.
5) Evaluation Metrics: This article adopts the cumulative matching characteristics (CMC), mean average precision
(mAP), mean inverse negative penalty (mINP) [39], and
modality divergence reduction rate (MDRR), where the quantitative metric process of MDRR is illustrated as follows.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19886

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

In the testing process, we sample visible and infrared
intraclass instance pairs f (Y vi ) and f (Y ri ) from the testing minibatch, and the cross-modal intraclass gap can be defined by
the following formula:


(34)
D0intra = f Y vi − f Y ri 2
where k · k2 denotes the Euclidean distance.
We explore the interclass centroid V j (hardest negative
centroid) with minimum Euclidean distance between the crossmodal intraclass instances f (Y vi ) and f (Y ri ). The class centroid
is calculated by the arithmetic mean of the class instances. The
explored interclass centroid can be defined by the following
formula:



(35)
V j = arg min f Y vi − V 2 + f Y ri − V 2
V∈U

where U denotes the set of all negative class centroids.
The cross-modal interclass gap under the modality divergence problem can be defined by the following formula:


(36)
D0inter = f Y vi − V j 2 + f Y ri − V j 2 .
Then, we calculate the metric formula for modality divergence. Since alleviating modality divergence problem mainly
lies in increasing D0inter without interfering with reducing
D0intra , the larger Dmodality indicates the smaller negative effect
of modality divergence problem. This process can be defined
by the following formula:
Dmodality = D0inter − D0intra .

(37)

Then, the MDRR is calculated by the following formula:
!
init
Dfinal
modality − Dmodality
(38)
Rmodality = mean
Dfinal
modality
PN
where mean(x) = (1/N) i=1
xi , and i denotes the index
value of the visible and infrared instance pair. N denotes the
number of the sampled test instance pairs. Dinit
modality represents
the distance difference at the start of training, and Dfinal
modality
represents the distance difference at the end of training. The
quantitative metric Rmodality measures the percentage of modality divergence reduction during the cross-modal optimization.
A high value of metric Rmodality indicates that the proposed
method effectively reduces modality divergence.
B. Implementation Details
Following AGW [39], the backbone contains a nonlocal
module enhanced network, which uses the ResNet50 [48]
pretrained on ImageNet [49] as the feature extractor. All
pedestrian images are resized to a consistent 288 × 144
resolution and apply channel augmentation [37] and random
horizontal flipping, erasing, and cropping. The model uses
the Adam [50] optimizer. Following [13], [14], the training
process is divided into two stages. The first stage executes
the intramodality optimization module in Section III-C for
50 training epochs, and ADAC module in Section III-D is
executed for 50 training epochs in the second stage, where
the interclass similarity constraint losses Lvs and Lrs in the
intramodality optimization module are deleted in the second

stage. We adopt DBSCAN [31] for pseudo-label assignment,
where the distance threshold is set 0.6 on SYSU-MM01, 0.3
on RegDB, and 0.6 on LLCM. The memory updating rate α
is set to 0.2. Parameter θ is set to 40 on SYSU-MM01 and
0.1 on RegDB. Parameter β is set to 1.3. The parameter η in
adversarial aware optimization loss is set to 0.5. In ablation
experiments, we perform ablation studies based on the results
of adding to the PGM [15].
C. Comparison With the State-of-the-Art Methods
We mainly raise a new modality divergence problem during
cross-modality optimization in the unsupervised VI-ReID task,
and our method lays emphasis on its novel cross-modality
optimization process. Furthermore, our method has universal applicability that can be used as a plug-in module to
add the existing unsupervised VI-ReID methods. We compare the recently developed supervised visible–infrared person
re-identification (supervised VI-ReID) methods and unsupervised visible–infrared person re-identification (unsupervised
VI-ReID) methods, as shown in Tables I–III.
1) Comparison With Unsupervised VI-ReID Methods: We
compare the proposed method with the six existing unsupervised VI-ReID methods, where the state-of-the-art methods
ADCA [14], PGM [15], and GUR [41] address the unsupervised visible–infrared person Re-ID problem by finding
cross-modality correspondences of the same identity. However,
there are huge cross-modality intraclass gaps in the unsupervised VI-ReID, which undoubtedly have a severe negative
impact on exploring the cross-modality correspondences of
the same identity. Furthermore, the state-of-the-art method
PCLHD [42] aims to explore the hard and dynamic prototypes
in data distributions and utilizes prototype-based contrastive
learning to preserve the intrinsic variety in sample features,
and SDCL [43] aims to focus on the collaboration of the
extracted shallow and deep features to capture modalityinvariant features and counter the cross-modality discrepancy.
In contrast, our ADAL framework emphasizes its novel crossmodality optimization process, which aims to alleviate the
negative impact of the modality divergence problem on unsupervised VI-ReID to reduce the huge cross-modality intraclass
gaps and thus further improve the exploration ability of
cross-modality correspondences. It can be seen that the proposed ADAL outperforms the state-of-the-art unsupervised
visible–infrared person Re-ID methods PCLHD and SDCL
when used as a plug-in module to add these state-of-the-art
methods.
Furthermore, we conduct the comparison experiment on the
LLCM dataset, as shown in Table III. Since all unsupervised
person VI-ReID methods use the SYSU-MM01 and RegDB
datasets, we reproduce the state-of-the-art and closely related
methods ADCA and PGM to compare with our ADAL. It can
be seen that our ADAL has universal applicability that can be
used as a plug-in module to add the existing unsupervised
VI-ReID methods to address the new modality divergence
problem in unsupervised VI-ReID and significantly surpasses
current unsupervised VI-ReID methods on the larger and
complex visible–infrared person dataset LLCM.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

19887

TABLE I
C OMPARISON R ESULTS ON SYSU-MM01. S INCE A LL THE C OMPARED M ETHODS D O N OT U SE C AMERA I NFORMATION , FOR A FAIR C OMPARISON , O UR
M ETHOD I S C OMPARED W ITH GUR * T HAT D OES N OT U SE C AMERA I NFORMATION . I T C ONSISTS OF T WO G ROUPS , I . E ., S UPERVISED VI-R E ID
M ETHODS AND U NSUPERVISED VI-R E ID M ETHODS . T HE H IGHEST S CORES A RE S HOWN IN B OLD

TABLE II
C OMPARISON R ESULTS W ITH THE S TATE - OF - THE -A RT M ETHODS ON R EG DB. S INCE A LL THE C OMPARED M ETHODS D O N OT U SE C AMERA
I NFORMATION , FOR A FAIR C OMPARISON , O UR M ETHOD I S C OMPARED W ITH GUR * T HAT D OES N OT U SE C AMERA I NFORMATION . I T C ONSISTS
OF T WO G ROUPS , I . E ., S UPERVISED VI-R E ID M ETHODS AND U NSUPERVISED VI-R E ID M ETHODS . T HE
H IGHEST S CORES A RE S HOWN IN B OLD

D. Ablation Study
In this section, we carry out ablation studies, as shown in
Table IV, to demonstrate the role of the ADAC module in
Section III-D. In Table IV, L p , Ln , and Lco are the components
of ADAC module.
1) Effectiveness of ADAC: In Table IV, when the ADAC
module components L p , Ln , and Lco are, respectively, applied
to the baseline, the significant improvements can be observed.
It can be seen that the ADAC module improves performance

by penalizing the instance features that produce negative
optimization and facilitating the instance features that produce
positive optimization.
Instance-level cross-modality optimization loss Lco is proposed to further consider the instance-level optimization
based on the cluster-level optimization process L p + Ln ,
which achieves obvious gains over the baseline + (L p +
Ln ), as shown in Table IV. This benefit from that Lco can
improve the affinities of the positive instance pairs with large

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19888

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

TABLE III
C OMPARISON R ESULTS W ITH THE S TATE - OF - THE -A RT M ETHODS ON LLCM. T HE R ESULTS OF C OMPARISON M ETHODS A RE R EPRODUCED BY U S . T HE
H IGHEST S CORES A RE S HOWN IN B OLD

TABLE IV
P ERFORMANCE (%) C OMPARISON OF D IFFERENT S CHEMES OF THE P ROPOSED ADAL A PPROACH ON THE SYSU-MM01 AND R EG DB. T HE H IGHEST
S CORES A RE S HOWN IN B OLD

Fig. 4. (a)–(d) Performance (%) impact of parameter distance threshold,
memory updating rate α, balance parameter η, and balance parameter β in the
ADAC module.

cross-modality gaps to effectively improve cross-modality intraclass compactness.

2) Effectiveness of Distance Threshold in ADAC: We experiment with different distance threshold settings and discuss
their impact, as shown in Fig. 4(a). We verify the model on
different threshold parameter values (0.2, 0.4, 0.6, and 0.8) in
the SYSU-MM01 dataset. A smaller distance threshold (such
as 0.2) may cause the clustering to be scattered and cannot
effectively classify similar pedestrian instances into the same
cluster. Conversely, a larger distance threshold (such as 0.8)
may easily misclassify instances with different identities as the
same cluster, leading to deterioration of model performance.
Since almost all unsupervised visible–infrared person Re-ID
methods use DBSCAN for clustering, the distance threshold
is determined empirically by previous methods. Therefore, our
method serves as a plug-and-play method, which defaults to
adopt the distance threshold parameter 0.6 in SYSU-MM01,
consistent with the baseline method for fair comparison.
3) Effectiveness of Memory Updating Rate α in ADAC:
To study the effect of the memory updating rate α, we set it
to incremental values in Fig. 4(b). We use α to control the
updated degree of memory and guide memory in the ADAC
module. We change it over a range (0, 0.2, 0.4, 0.6, 0.8) and
keep the other parameters unchanged on SYSU-MM01. Since
a smaller α increases the update ratio of the extracted features
and such memory contains more changes in newly extracted
features, which improves to reflect the distribution of the feature during the training process. Our ADAC module requires
memory and guide memory to determine the convergence
direction of the extracted features in the next iteration. Thus,
we empirically set α to a lower value of 0.2 based on the
experiments, allowing the memories to adapt to the changes
in the newly extracted features.
4) Effectiveness of Balance Parameter η in ADAC: To
study the effect of the balance parameter η, we incrementally adjusted its value, as illustrated in Fig. 4(c). η
is used to control the ratio between promoting positive
optimization and penalizing negative optimization in the cross-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.


codex
FDGReID 后续确认了方法不是单纯“联邦加 ReID”，它把域漂移拆成风格漂移和视角漂移两个来源，所以后面要特别看它的消融是不是证明了这两个来源，而不是只证明组件堆叠涨点。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 14 -l 19 'FDGReID - Federated Domain Generalization for Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
22 Page 14 of 20

Machine Learning (2026) 115:22

duration of 120 epochs. All images are resized to 256 × 128, and standard data augmentation techniques including horizontal flipping and random erasing are applied. For viewpoint
estimation, we utilize MEBOW (Wu et al., 2020), applying inference directly on the training
set using its publicly available pre-trained model. For pose estimation, we adopt OpenPose (Cao et al., 2021) to extract 19 keypoints in COCO format. Since facial landmarks
such as nose, eyes, and ears are not essential for our application, we average the five facial
keypoints into a single representative point, resulting in a total of 14 keypoints per image.
Model training is conducted using the Adam optimizer (Kingma & Ba, 2014) with an initial
learning rate of 5 × 10−4 , momentum of 0.9, and an L2 weight decay of 1 × 10−6 . The
learning rate is reduced by a factor of 0.1 every 30 epochs. The temperature parameter τ
1
used in contrastive loss is set to 16
. For the largest client (i.e., the MSMT17 dataset), each
round of local training (with 6 epochs) takes approximately 25.3 min. A complete federated
training run (R = 20 rounds, totaling 120 epochs) requires roughly 8 to 9 h, including a small
amount of communication overhead. In contrast, the inference time for each dataset is very
short, on the order of a few minutes.

4.3 Comparison with State-of-the-Arts
Generalization across unseen domains.
Table 3 presents the performance of our proposed FDGReID framework under both DG
Protocol-1 and DG Protocol-2. We evaluate the model on four target datasets (PRID, GRID,
VIPeR, and iLIDs) and report the average of mean Average Precision (mAP) and Rank-1
accuracy to quantify overall generalization ability. The upper section of Table 3 compares
FDGReID with existing domain generalization (DG) Re-ID methods. Under DG Protocol-1, our model performs slightly below several specialized DG methods on average. This
is expected, as most DG methods explicitly perform representation alignment or sophisticated data augmentation, which facilitates domain invariance. For example, the state-of-theart method META (Xu et al., 2021) employs a Mixture-of-Experts (MoE) framework that
shares all network parameters except for batch normalization layers across domains. However, such methods violate federated learning principles, as they rely on domain-specific
parameter sharing that may compromise data privacy. In contrast, FDGReID adheres strictly
to privacy constraints and avoids any cross-domain parameter coupling. Although a direct
comparison with DG methods is not entirely fair due to differing assumptions, FDGReID
still achieves competitive results on challenging datasets such as VIPeR and iLIDs–outperforming certain DG baselines. Moreover, under DG Protocol-2, which increases training
data by incorporating both training and test sets from source domains, our model exhibits
substantial performance gains. These results highlight the strong generalization capability
of FDGReID as training data diversity increases.

4.3.1 Performance in Federated Settings
The lower section of Table 3 presents results under the standard federated learning (FL)
setting. FDGReID consistently outperforms prior FL-based Re-ID approaches, notably surpassing the state-of-the-art FFReID (Zhang et al., XXX) with an average improvement of
8.0% in mAP and 7.2% in Rank-1 accuracy. This superior performance demonstrates that
naive parameter aggregation in FL settings–where clients possess highly heterogeneous and

13

Machine Learning (2026) 115:22

Page 15 of 20 22

Table 3 In comparison with state-of-the-art methods under DG protocol-1
Method
PRID Hirzer et GRID Loy et
VIPeR Gray and iLIDs Zheng et
Average
al. (2011)
al. (2010)
Tao (2008)
al. (2009)
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
QAConv Liao
62.2/52.3
57.4/48.6
66.3/57.0
81.9/75.0
67.0/58.2
and Shao (2020)
50.5/40.0
68.2/60.8
74.3/65.0
64.6/55.2
M3 L Zhao et al. 65.3/55.0
(2021)
MetaBIN Choi et 70.8/61.2
57.9/50.2
64.3/55.9
82.7/74.7
68.9/60.5
al. (2021a)
META Xu et al.
71.7/61.9
60.1/52.4
68.4/61.5
83.5/79.2
70.9/63.8
(2021)
FedReID McMa- 25.2/28.3
7.1/5.4
32.7/31.5
58.9/55.7
31.0/30.2
han et al. (2016)
FedPaV Zhuang
27.1/30.2
12.0/8.5
38.7/41.2
62.2/61.0
35.0/35.2
et al. (2020)
FedAvg McMa34.1/39.6
19.4/14.4
43.2/47.7
68.5/72.6
41.3/43.6
han et al. (2016)
FFReID Zhang et 50.6/49.1
39.1/38.5
58.9/57.2
73.6/72.2
55.6/54.3
al. (XXX)
FDGReID(Ours) 58.2/57.1
48.2/43.7
66.4/64.8
81.7/80.3
63.6/61.5
FDGReID(Ours)* 61.4/57.7
54.5/53.6
72.6/71.7
84.4/85.3
68.2/67.0
The symbol ‘*’ represents the performance under DG protocol-2. The upper portion of the table displays
the cross-domain person Re-ID performance attained by the DG person Re-ID approaches. The lower part
of the table highlights the model outcomes obtained through the FL person RE-ID methods

domain-shifted data–often leads to sub-optimal global models. In contrast, our proposed
components, such as Style Information Sharing and Viewpoint-aware Contrastive Learning,
effectively alleviate local model bias and enhance global model generalization. Importantly,
under DG Protocol-2, FDGReID achieves an additional gain of 4.6% in mAP and 5.5% in
Rank-1 accuracy. This improvement is primarily attributed to the enlarged training data
scope, enabling the model to learn more robust viewpoint- and style-invariant representations. These findings further confirm that FDGReID is well-suited for real-world federated
Re-ID applications, particularly in settings with significant domain diversity.
Experiments under the Federated Fully Supervised Re-ID Setting.***
As shown in Table 4, our FDGReID framework achieves consistent and substantial
improvements across multiple datasets in the federated fully supervised setting. Notably,
on the CUHK03 dataset, our method yields a remarkable improvement of +15.8% in mAP
and +15.5% in Rank-1 accuracy over the previous state-of-the-art, representing the most
significant gain across all evaluated datasets. This result underscores the effectiveness of our
approach in mitigating challenges associated with data volume imbalance and distribution
heterogeneity among local clients in federated environments. Even on smaller datasets such
as iLIDs, FDGReID demonstrates clear benefits, achieving a +7.5% increase in mAP and
a +5.5% improvement in Rank-1 accuracy. These results confirm that our method remains
robust and beneficial, even when data availability is limited. On the widely-used Market-1501 dataset–which already presents strong baseline performance–our method achieves
an additional +3.1% mAP and a notable +10.8% improvement in Rank-1 accuracy. Given
that FFReID has already shown competitive results on this dataset, these gains further validate the generalization capacity of our approach, especially in scenarios with high domain

13

22 Page 16 of 20

Machine Learning (2026) 115:22

Table 4 In comparing our methods with state-of-the-art approaches in federated full supervision, we compare
our methods with FFReID, FedAvg, FedPaV,and FedReID. Specifically, FedAvg utilizes ViT as the backbone
architecture, while the other methods adopt ResNet50 as the backbone
Method
FDGReID(Ours) FFReFedAvg
FedPaV Zhuang FedReID
ID Zhang et
McMahan et et al. (2020)
McMahan et
al. (XXX)
al. (2016)
al. (2016)
Datasets
Rank-1/mAP
Rank-1/mAP Rank-1/mAP Rank-1/mAP
Rank-1/mAP
Market1501 Zheng 92.8/85.6
89.7/74.8
72.5/47.0
83.4/60.6
79.0/52.1
et al. (2015)
MSMT17 Wei et
53.1/39.9
52.9/37.1
40.4/25.3
41.0/21.4
35.5/19.6
al. (2018)
CUHK03 Li et al. 69.2 /65.4
53.4/49.9
23.6/21.3
31.7/27.8
16.0/14.5
(2014)
PRID Hirzer et al. 79.3/82.0
76.0/79.6
50.0/56.2
37.6/42.1
3.0/5.4
(2011)
VIPeR Gray and
72.9/73.6
67.4/72.2
39.8/45.7
48.1/52.5
30.3/35.6
Tao (2008)
iLIDSZheng et al. 94.2/90.3
86.7/84.8
76.5/71.3
79.9/76.4
76.5/66.6
(2009)
Average
76.9/72.8
71.0/66.4
50.5/44.5
53.6/46.8
40.1/32.3

diversity and large-scale samples. Collectively, these results highlight the robustness and
scalability of FDGReID in fully supervised federated Re-ID tasks, making it well-suited
for deployment in real-world surveillance systems with heterogeneous, privacy-constrained
client data.

4.4 Ablation Studies
Table 5 presents a comprehensive ablation analysis of the proposed Style Information Sharing (SIS) and Viewpoint-aware Contrastive Learning (VCL) components. Additionally, we
investigate the impact of the proposed viewpoint-aware contrastive losses by comparing
models trained with and without them. From Table 5, we observe that removing both SIS
and VCL significantly compromises the model’s generalization performance. In particular,
the model performs poorly on datasets with lower image quality, such as GRID, where
Rank-1 accuracy drops below 10%, and the average mAP remains around 30%. These
results indicate that the model, without any domain adaptation strategies, fails to generalize
effectively across domains. When evaluating the individual contribution of SIS and VCL,
it is evident that VCL plays a more critical role. The model equipped solely with VCL
achieves an average mAP of 13.6% and Rank-1 accuracy of 11.8% across all four test sets.
Notably, the improvement is more pronounced on challenging datasets such as GRID and
VIPeR. This confirms our hypothesis that viewpoint variation is a dominant factor in crossdomain performance degradation. A deeper investigation further reveals that SIS alone has a
limited effect when compared to VCL. We attribute this to the unique nature of person Re-ID
datasets, which not only differ in lighting conditions, backgrounds, and viewpoints but also
exhibit significant stylistic diversity, including variations from real-world, synthetic, and
cartoon images. Moreover, Table 5 shows the critical impact of integrating viewpoint-aware
contrastive losses within VCL. Without these losses, the model’s performance degrades
notably, with a 4.9% drop in average mAP and an 8.5% decline in Rank-1 accuracy. This
highlights the importance of learning robust viewpoint-invariant features and validates the

13

Machine Learning (2026) 115:22

Page 17 of 20 22

Table 5 Ablation studies of (1) the Style Information Sharing mechanism and the Viewpoint-aware Contrastive Learning (2) the viewpoint-aware contrastive losse. ‘SIS’ refers to the style information sharing mechanism, ‘VCL’ denotes the Viewpoint-aware Contrastive Learning, and Lvcl represents the viewpoint-aware
contrastive losse. The symbol ‘*’ represents the performance under DG protocol-2
Method
PRID Hirzer et GRID Loy et
VIPeR Gray and iLIDs Zheng et
Average
al. (2011)
al. (2010)
Tao (2008)
al. (2009)
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
Rank-1/mAP
w/o SIS, w/o
25.2/28.3
7.1/5.4
32.7/31.5
58.9/55.7
31.0/30.2
VCL
w SIS, w/o VCL 34.5/35.5
19.6/17.2
37.5/38.2
64.1/57.2
38.9/37.0
w/o SIS, w VCL 47.5/45.0
37.7/32.2
55.4/53.0
69.5/65.0
52.5/48.8
49.4/47.5
37.3/35.6
59.9/57.3
72.1/75.9
54.7/49.0
VCL, Lvc
FDGReID(Ours) 58.2/57.1
48.2/43.7
66.4/64.8
81.7/80.3
63.6/61.5
FDGReID(Ours)* 61.4/57.7
54.5/53.6
72.6/71.7
84.4/85.3
68.2/67.0

effectiveness of our contrastive learning design in modeling intra-class consistency and
inter-class separation across varying viewpoints.

5 Conclusion
In this paper, we propose FDGReID, a federated domain generalizable framework for person re-identification (Re-ID) tailored to address domain shifts and data privacy challenges in
real-world scenarios. It integrates two novel components: Style Information Sharing (SIS)
and Viewpoint-aware Contrastive Learning (VCL). SIS boosts local model robustness by
enabling collaborative, privacy-preserving feature sharing among federated clients to introduce diverse image style representations. VCL, conversely, mitigates viewpoint-induced
domain drift by learning viewpoint-invariant identity representations via contrastive learning. Extensive experiments on multiple benchmark datasets show FDGReID achieves stateof-the-art generalization performance while ensuring strict data privacy. Ablation studies
confirm the effectiveness of each component: VCL significantly enhances cross-view discriminability, and SIS complementarily addresses style diversity. Overall, FDGReID offers
a scalable, privacy-aware, and highly generalizable solution for deploying person Re-ID
systems in practical federated environments.
Acknowledgements This work was supported by Shanghai Platform for Neuromorphic and AI Chip under
Grant 17DZ2260900 (NeuHelium) and in part by the National Natural Science Foundation of China under
Grant 62406252. The authors would like to thank the anonymous reviewers for their valuable suggestions
and constructive criticisms.
Author Contributions Ke Niu, Haiyang Yu Teng Fu and Mengyang Zhao mainly conducted experiments and
wrote this manuscript. Bin Li guided the method design and experiments. Xuelin Qian and Xiangyang Xue
put forward suggestions for method improvement. All authors read and approved this manuscript.
Funding This work was supported by Shanghai Platform for Neuromorphic and AI Chip under Grant
17DZ2260900 (NeuHelium) and in part by the National Natural Science Foundation of China under Grant
62406252.
Data Availability No datasets were generated or analysed during the current study.
Code Availability The code will be available after this paper is accepted.

13

22 Page 18 of 20

Machine Learning (2026) 115:22

Declarations
Conflict of interest The authors declare that they have no conflict of interest.
Ethical Approval Not applicable.
Consent to Participate Not applicable.
Consent for Publication Not applicable.

References
Busbridge, D., Sherburn, D., Cavallo, P., & Hammerla, N. Y. (2019). Relational graph attention networks.
arXiv preprint arXiv:1904.05811
Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2021). Openpose: Realtime multi-person 2d
pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence,43, 172–186.
Choi, S., Kim, T., Jeong, M., & Park, H.,& Kim, C. (2021a). Meta batch-instance normalization for generalizable person re-identification. In 2021 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR).
Choi, S., Kim, T., Jeong, M., Park, H., & Kim, C. (2021b). Meta batch-instance normalization for generalizable person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, (pp. 3425–3435).
Dai, Y., Li, X., Liu, J., Tong, Z., & Duan, L. -Y. (2021). Generalizable person re-identification with relevanceaware mixture of experts. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR).
Filax, M., & Ortmeier, F. (2021). On the influence of viewpoint change for metric learning. In 2021 17th
International Conference on Machine Vision and Applications (MVA) (pp. 1–4). IEEE.
Gray, D., & Tao, H. (2008). Viewpoint invariant pedestrian recognition with an ensemble of localized features (pp. 262–275).
Hirzer, M., Beleznai, C., Roth, P. M., & Bischof, H. (2011). Person re-identification by descriptive and discriminative classification (pp. 91–102.)
Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization.
In Proceedings of the IEEE international conference on computer vision, (pp. 1501–1510).
Huang, Y., Wu, Q., Xu, J., & Zhong, Y. (2019). Sbsgan: Suppression of inter-domain background shift for
person re-identification. In ICCV (pp. 9527–9536).
Jin, X., Lan, C., Zeng, W., Chen, Z., & Zhang, L. (2020). Style normalization and restitution for generalizable
person re-identification. In CVPR (pp. 3143–3152).
Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z.,
Cormode, G., Cummings, R., et al. (2021). Advances and open problems in federated learning. Foundations and trends®. Machine Learning,14(1–2), 1–210.
Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv Learning
Li, W., & Wang, X. (2013). Locally aligned feature transforms across views. In 2013 IEEE conference on
computer vision and pattern recognition.
Li, Y. -J., Chen, Y. -C., Lin, Y. -Y., Du, X., & Wang, Y. -C. F. (2019). Recover and identify: A generative dual
model for cross-resolution person re-identification. In ICCV (pp. 8090–8099).
Li, Y. -J., Lin, C. -S., Lin, Y. -B., & Wang, Y. -C. F. (2019). Cross-dataset person re-identification via unsupervised pose disentanglement and adaptation. In ICCV (pp. 7919–7929).
Li, W., Zhao, R., Xiao, T., & Wang, X. (2014). Deepreid: Deep filter pairing neural network for person reidentification. In 2014 IEEE Conference on Computer Vision and Pattern Recognition.
Liao, S., & Shao, L. (2020). Interpretable and generalizable person re-identification with query-adaptive
convolution and temporal lifting (pp. 456–474).
Liu, F., Ye, M., & Du, B. (2024). Domain generalized federated learning for person re-identification. Computer Vision and Image Understanding,241, Article 103969.
Loy, C. C., Xiang, T., & Gong, S. (2010). Time-delayed correlation analysis for multi-camera activity understanding. International Journal of Computer Vision,90, 106–129.

13

Machine Learning (2026) 115:22

Page 19 of 20 22

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. (2016). Communication-efficient learning of deep networks from decentralized data. arXiv Learning.
Ni, H., Li, Y., Gao, L., Shen, H. T., & Song, J. (2023). Part-aware transformer for generalizable person
re-identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp.
11280–11289).
Qian, X., Wang, W., Zhang, L., Zhu, F., Fu, Y., Xiang, T., Jiang, Y. -G., & Xue, X. (2020). Long-term clothchanging person re-identification. In ACCV.
Sun, X., & Zheng, L. (2019). Dissecting person re-identification from the viewpoint of viewpoint. In CVPR,
(pp. 608–617).
Wei, L., Zhang, S., Gao, W., & Tian, Q. (2018). Person transfer gan to bridge domain gap for person reidentification. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition.
Wu, C., Chen, Y., Luo, J., Su, C. -C., Dawane, A., Hanzra, B., Deng, Z., Liu, B., Wang, J. Z., & Kuo, C. -H.
(2020). Mebow: Monocular estimation of body orientation in the wild. In 2020 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR).
Xiao, T., Li, S., Wang, B., Li, L., & Wang, X. (2016). End-to-end deep learning for person search. arXiv
computer vision and pattern recognition.
Xu, B., Liang, J., He, L., & Sun, Z. (2021). Mimic embedding via adaptive aggregation: Learning generalizable person re-identification.
Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated machine learning: Concept and applications. ACM
Transactions on Intelligent Systems and Technology (TIST),10(2), 1–19.
Zhang, P., Yan, H., Wu, W., Wang, S., & Improv, I.: Improving federated person re-identification through
feature-aware proximity and aggregation.
Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-iid data.
arXiv preprint arXiv:1806.00582
Zhao, Y., Zhong, Z., Yang, F., Luo, Z., Lin, Y., Li, S., & Sebe, N. (2021). Learning to generalize unseen
domains via memory-based multi-source meta-learning for person re-identification. In 2021 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR).
Zheng, W.-S., Gong, S., & Xiang, T. (2009). Associating groups of people. In Procedings of the British
Machine Vision Conference 2009.
Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., & Tian, Q. (2015). Scalable person re-identification: A
benchmark. In 2015 IEEE International Conference on Computer Vision (ICCV).
Zheng, L., Zhang, H., Sun, S., Chandraker, M., Yang, Y., & Tian, Q. (2017). Person re-identification in the
wild. In CVPR, (pp. 1367–1376).
Zheng, Z., Zheng, L., & Yang, Y. (2017). Unlabeled samples generated by gan improve the person re-identification baseline in vitro. In 2017 IEEE International Conference on Computer Vision (ICCV).
Zheng, Z., Zheng, L., & Yang, Y. (2017). A discriminatively learned cnn embedding for person reidentification. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM),14(1),
1–20.
Zhuang, W., Wen, Y., & Zhang, S. (2021). Joint optimization in edge-cloud continuum for federated unsupervised person re-identification. In Proceedings of the 29th ACM International Conference on Multimedia, (pp. 433–441).
Zhuang, W., Wen, Y., Zhang, X., Gan, X., Yin, D., Zhou, D., Zhang, S., & Yi, S. (2020). Performance optimization of federated person re-identification via benchmark analysis. In Proceedings of the 28th ACM
International Conference on Multimedia, (pp. 955–963).
Zhuang, W., Wen, Y., Zhang, X., Gan, X., Yin, D., Zhou, D., Zhang, S., & Yi, S. (2020). Performance optimization of federated person re-identification via benchmark analysis. In Proceedings of the 28th ACM
International Conference on Multimedia.
Zhuang, W., Gan, X., Wen, Y., & Zhang, S. (2023). Optimizing performance of federated person re-identification: Benchmarking and analysis. ACM Transactions on Multimedia Computing, Communications
and Applications,19(1s), 1–18.
Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and
institutional affiliations.
Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a
publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

13


codex
按你这条消息列出的 6 篇拆完。没有额外联网，PDF 的方法和相关工作已经足够判断它们怎么构造创新。

**1. ESTI**
**触发观察：** 视频 ReID 里相邻帧变化很小，很多方法仍然对整段相邻帧做完整时空聚合。作者抓住两个后果：一是重复计算，二是相似帧聚合会放大冗余区域，反而稀释关键局部和动态细节。

**重定义动作：** 它没有把问题说成“时序建模不够强”，而是改写成“全帧时空聚合本身有冗余，并导致关键信息丢失”。关键词是 redundant information、redundancy computations、critical spatial and temporal details、efficient spatial-temporal interaction。

**机制怎么长出来：** 如果冗余来自整段都做同一种聚合，那么自然做法就是少处理、分开处理、晚交互。于是它把序列一分为二，一半走空间分支，一半走时间分支，再用 FEM 强化空间局部，用 TIM 建模相邻时序关系，最后用 STIM 做空间时间交互。机制和叙事绑定中等偏紧，真正紧的是“分支减冗余”，FEM/TIM 用 deformable attention 更像工程实现选择。

**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。

**reviewer 为什么买账：** 这篇主要卖“视频帧冗余下的高效时空交互视角”，不是卖 deformable attention 本身。审稿人买账点在于它把常见的“更强时序模块”换成“避免过度处理相似帧”，且性能和复杂度同时说得过去。

**2. Exploring Part-Informed Visual-Language Learning**
**触发观察：** CLIP-ReID 的全局图文对齐能让全局特征变强，但作者用 t-SNE 和 parsing 标注发现，像素级或部位级特征仍然混在一起。身份边界可能清楚，但头、身体、鞋等部位的语义不一致，特别是相似身份的局部特征更难分。

**重定义动作：** 它把问题从“ReID 需要语言监督”推进成“全身级语言监督不足以约束局部语义一致性”。关键词是 within-part semantic inconsistency、part-informed visual-language learning、pixel-level text prompts、fine-grained image-text alignment。它还补了一层冲突定义：human parsing 是身份无关的，而 ReID 要身份相关判别，直接用 parsing 会和身份学习冲突。

**机制怎么长出来：** 既然问题是部位语义不一致，就用 parsing 给每个像素部位名，再构造“某身份的头、鞋、包”等身份感知部位文本提示。既然 parsing 有噪声，就用置信度加权。既然低层和高层特征尺度不同，就加轻量辅助头做多尺度融合对齐。这个机制和重定义绑定很紧，每一块都对应一个被定义出来的障碍。

**证据闭环：** 最关键的是 Fig.2 的特征分布可视化，证明全局 CLIP-ReID 下部位语义确实混。消融里从普通 parsing prompt，到 identity-aware part prompt，再到 parsing confidence weighted loss，逐步证明“部位语义”和“身份约束”都必要。不同 parsing 模型的实验也在回答 reviewer 对 parsing 质量敏感性的疑问。

**reviewer 为什么买账：** 这篇卖的是视角多于模块。真正 novelty 是把 CLIP-ReID 从全局图文对齐改写成“部位级、身份感知、训练期语言监督”，不是简单加 parsing。它的强点是概念链完整：发现局部语义不一致，指出 parsing 与 ReID 的监督冲突，再给出对应机制。

**3. False Negatives Consensus Suppression**
**触发观察：** 文本到图像 ReID 的对比学习默认“非配对样本都是负样本”。作者指出这个假设在 TIReID 里不成立，因为文本描述可能适配多个行人，标注或描述粒度也会造成语义相近的假负样本。把这些样本推远会破坏跨模态对齐。

**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。

**机制怎么长出来：** 如果假负样本是高相似负样本，就用负样本相似度分布找异常。因为只看全局可能误判，所以同时用 BGR 全局表示和 TSR 局部 token 选择表示，取二者共识。因为早期模型不可靠，不能直接删掉疑似假负样本，所以用 AFNS 按假负概率自适应降权。因为 batch 内负样本少，GMM 拟合不稳，所以加跨模态 momentum queue。机制基本是从问题定义自然推出的。

**证据闭环：** 它有三类证据：组件消融证明 BGR/TSR/CM-MoC 都有贡献；和 FNE、FNA 这类简单删样本或改标签策略比较，证明“自适应抑制”比硬删除稳；可视化显示高概率假负样本确实和文本语义相近。这个闭环比较完整，因为它直接验证“假负样本存在且会伤训练”。

**reviewer 为什么买账：** 这篇卖的是一个训练假设错误，而不是新 backbone。reviewer 容易买账的点是：假负样本问题在 TIReID 里直观存在，而且方法可以作为对比学习插件加到强模型上。novelty 来自“把负样本污染作为核心问题”，不是来自 GMM 或 queue 本身。

**4. FDGReID**
**触发观察：** 真实 ReID 部署同时有两个矛盾：跨域泛化需要多域信息，但隐私限制不允许直接共享数据；联邦学习能保隐私，但普通 FL 只是聚合本地偏置模型，没有解决域漂移。作者进一步把漂移观察拆成两类：风格差异，比如亮度、色调、纹理；视角差异，比如摄像头角度不同。

**重定义动作：** 它把问题从“联邦 ReID”改写成“联邦域泛化 ReID”。关键词是 federated domain generalization、style information sharing、viewpoint-aware contrastive learning、privacy-preserving domain drift。它的核心话术是：DG 方法不够隐私，FL 方法不够泛化，所以要在 FL 约束内做 DG。

**机制怎么长出来：** 风格漂移对应 SIS：客户端不传图，只传 feature mean/std 这类风格统计，服务器重分发风格描述，客户端用 AdaIN 做跨客户端风格迁移和外推。视角漂移对应 VCL：用 appearance 分支提外观，用 OpenPose 加 RGAT 提姿态结构，再用 viewpoint-aware contrastive loss 学视角不变表征，最后 AFF 融合。机制和重定义的高层绑定清楚，但组件本身偏拼装，AdaIN、pose、graph attention 都不是新东西。

**证据闭环：** 最关键是 ablation：去掉 SIS 和 VCL 性能很差；只加 VCL 的收益明显大于只加 SIS，支撑“视角漂移是重要来源”；去掉 viewpoint-aware contrastive loss 后明显下降，说明不是只靠多分支。它还在 DG 协议和 federated fully supervised 协议下比较，承认不一定打过专门 DG 方法，但在 FL 设定里明显强于 FedReID、FedAvg、FFReID。

**reviewer 为什么买账：** 这篇卖的是场景定义和系统视角。novelty 不在单个模块，而在“隐私约束下把域漂移拆成风格和视角两类再分别处理”。如果 reviewer 接受 federated ReID 是实际部署问题，它就有价值；如果只看纯方法新意，机制会显得常规。

**5. Find Hidden Modality Divergence**
**触发观察：** 无监督可见光红外 ReID 里，大家用对比学习拉近跨模态同类、推远异类。但作者指出，由于跨模态同类间隔很大，中间常常夹着异类样本。对比学习推远这些异类时，会阻碍真正同类跨模态样本靠近。

**重定义动作：** 它没有泛泛说“modality gap 大”，而是把问题改写成“跨模态优化过程中的 modality divergence”。关键词是 hidden modality divergence、negative optimization、positive optimization、adversarial aware learning。这个重定义把静态差距变成了动态优化冲突：同类聚合和异类分离在某些位置会互相打架。

**机制怎么长出来：** 既然问题是优化方向冲突，就要找出哪些 cluster centroid 正在产生正向或负向优化。ADAL 用当前 memory 和下一步 guide memory 的相似度变化判断某个 centroid 会让实例靠近还是远离。对正向优化继续促进，对负向优化反向惩罚。然后再补一个 instance-level easy-hard positive 约束，让跨模态 hard positive 向 easy positive 收拢。这个机制和重定义绑定很紧，因为它不是一般对齐，而是在处理“错误优化方向”。

**证据闭环：** 论文除了常规 SOTA，还设计了 MDRR 这种 modality divergence reduction rate，试图直接量化“divergence 是否被减少”。消融里分别加 Lp、Ln、Lco，证明促进正向、惩罚负向、easy-hard 收拢都有效。作为插件加到已有无监督 VI-ReID 方法上也能涨，说明它不是依赖某个特定 backbone。

**reviewer 为什么买账：** 这篇的真正 novelty 是视角和机制都相对绑定。它把老问题“跨模态 gap”讲成“对比学习内部的优化冲突”，再用 adversarial aware 的方式处理冲突。reviewer 买的是这个诊断，不只是最终指标。

**6. FLAG**
**触发观察：** 视频换衣 ReID 中，外观信息被衣服主导，换衣后外观可靠性下降；但视频里还有 gait。现有视频 ReID 方法虽然会建模时空信息，却没有显式把 appearance 和 gait 分开提取。silhouette 不含衣服纹理，天然更适合学 gait。

**重定义动作：** 它把 VCC-ReID 从“视频 ReID 的困难版本”改写成“有限 appearance 信息和 gait 信息如何显式学习的问题”。关键词是 explicit learning based on appearance and gait、multi-modal and multi-granularities、cross-modal knowledge distillation。它还把贡献包装成“把同衣视频 ReID 方法迁移到换衣设定的框架”，而不是单个新网络。

**机制怎么长出来：** 如果 appearance 和 gait 要显式解耦，就用 visible branch 学 appearance，用 silhouette branch 学 gait。因为 gait 既有全局轮廓也有局部身体结构，所以做多粒度 silhouette。因为多模态大模型推理成本高，所以再用 CMKDS 把 MGA teacher 蒸馏到只用 visible 输入的小模型。机制和问题绑定很清楚，尤其是“大模型高性能，小模型可部署”这一层很实用。

**证据闭环：** 它的关键证据不是单一 SOTA，而是泛化和消融：MGA 接到 AP3D、TCLNet 都涨，证明框架不是只适配一个方法；不同 silhouette 粒度显示三粒度最好，粒度过多会引入噪声，支撑“多粒度 gait”而不是越多越好；EPFM、并行融合、预训练冻结、直接加间接蒸馏都有消融；CMKDS 证明不增加推理成本也能涨。

**reviewer 为什么买账：** 这篇卖的是“显式 appearance/gait 分解”的框架视角，加上换衣视频这个明确应用场景。模块未必很新，但叙事强，因为 silhouette/gait 与换衣鲁棒性之间有天然因果关系。真正 novelty 是把视频 ReID 和 gait recognition 接起来，并用蒸馏解决部署问题。

**共同套路**
这批论文共同不是先发明模块，而是先抓一个旧范式里的错误默认：相邻帧都值得聚合、全局 VLM 足够、所有负样本都真负、FL 聚合自然泛化、对比学习方向总是有益、换衣后还能靠普通外观。然后它们把这个错误默认命名成一个新失败模式，并设计至少一个能直接验证该失败模式的消融、可视化或指标。我们要学的是这个结构：先找“大家以为 X，其实 Y”的可证伪观察，再让机制成为 Y 的自然修正，而不是先堆一个模块再倒找理由。
tokens used
122,329
按你这条消息列出的 6 篇拆完。没有额外联网，PDF 的方法和相关工作已经足够判断它们怎么构造创新。

**1. ESTI**
**触发观察：** 视频 ReID 里相邻帧变化很小，很多方法仍然对整段相邻帧做完整时空聚合。作者抓住两个后果：一是重复计算，二是相似帧聚合会放大冗余区域，反而稀释关键局部和动态细节。

**重定义动作：** 它没有把问题说成“时序建模不够强”，而是改写成“全帧时空聚合本身有冗余，并导致关键信息丢失”。关键词是 redundant information、redundancy computations、critical spatial and temporal details、efficient spatial-temporal interaction。

**机制怎么长出来：** 如果冗余来自整段都做同一种聚合，那么自然做法就是少处理、分开处理、晚交互。于是它把序列一分为二，一半走空间分支，一半走时间分支，再用 FEM 强化空间局部，用 TIM 建模相邻时序关系，最后用 STIM 做空间时间交互。机制和叙事绑定中等偏紧，真正紧的是“分支减冗余”，FEM/TIM 用 deformable attention 更像工程实现选择。

**证据闭环：** 关键证据是组件消融：空间分支、时间分支、STIM、FA 逐步加上都有收益；再用 RCB、自注意力、交叉注意力对比 FEM/TIM；还有模型大小和 Rank-1 的效率图，说明不是靠堆参数。弱点是它没有特别强地证明“一半帧就足够”这个核心重定义，只是间接由效率和性能成立。

**reviewer 为什么买账：** 这篇主要卖“视频帧冗余下的高效时空交互视角”，不是卖 deformable attention 本身。审稿人买账点在于它把常见的“更强时序模块”换成“避免过度处理相似帧”，且性能和复杂度同时说得过去。

**2. Exploring Part-Informed Visual-Language Learning**
**触发观察：** CLIP-ReID 的全局图文对齐能让全局特征变强，但作者用 t-SNE 和 parsing 标注发现，像素级或部位级特征仍然混在一起。身份边界可能清楚，但头、身体、鞋等部位的语义不一致，特别是相似身份的局部特征更难分。

**重定义动作：** 它把问题从“ReID 需要语言监督”推进成“全身级语言监督不足以约束局部语义一致性”。关键词是 within-part semantic inconsistency、part-informed visual-language learning、pixel-level text prompts、fine-grained image-text alignment。它还补了一层冲突定义：human parsing 是身份无关的，而 ReID 要身份相关判别，直接用 parsing 会和身份学习冲突。

**机制怎么长出来：** 既然问题是部位语义不一致，就用 parsing 给每个像素部位名，再构造“某身份的头、鞋、包”等身份感知部位文本提示。既然 parsing 有噪声，就用置信度加权。既然低层和高层特征尺度不同，就加轻量辅助头做多尺度融合对齐。这个机制和重定义绑定很紧，每一块都对应一个被定义出来的障碍。

**证据闭环：** 最关键的是 Fig.2 的特征分布可视化，证明全局 CLIP-ReID 下部位语义确实混。消融里从普通 parsing prompt，到 identity-aware part prompt，再到 parsing confidence weighted loss，逐步证明“部位语义”和“身份约束”都必要。不同 parsing 模型的实验也在回答 reviewer 对 parsing 质量敏感性的疑问。

**reviewer 为什么买账：** 这篇卖的是视角多于模块。真正 novelty 是把 CLIP-ReID 从全局图文对齐改写成“部位级、身份感知、训练期语言监督”，不是简单加 parsing。它的强点是概念链完整：发现局部语义不一致，指出 parsing 与 ReID 的监督冲突，再给出对应机制。

**3. False Negatives Consensus Suppression**
**触发观察：** 文本到图像 ReID 的对比学习默认“非配对样本都是负样本”。作者指出这个假设在 TIReID 里不成立，因为文本描述可能适配多个行人，标注或描述粒度也会造成语义相近的假负样本。把这些样本推远会破坏跨模态对齐。

**重定义动作：** 它把问题从“图文细粒度对齐不够”改写成“负样本集合被污染”。关键词是 false negatives、consensus detection、adaptive suppression、negative may not be true negative。这个改写很聪明，因为它不和一堆局部对齐方法正面撞，而是攻击对比学习的基础假设。

**机制怎么长出来：** 如果假负样本是高相似负样本，就用负样本相似度分布找异常。因为只看全局可能误判，所以同时用 BGR 全局表示和 TSR 局部 token 选择表示，取二者共识。因为早期模型不可靠，不能直接删掉疑似假负样本，所以用 AFNS 按假负概率自适应降权。因为 batch 内负样本少，GMM 拟合不稳，所以加跨模态 momentum queue。机制基本是从问题定义自然推出的。

**证据闭环：** 它有三类证据：组件消融证明 BGR/TSR/CM-MoC 都有贡献；和 FNE、FNA 这类简单删样本或改标签策略比较，证明“自适应抑制”比硬删除稳；可视化显示高概率假负样本确实和文本语义相近。这个闭环比较完整，因为它直接验证“假负样本存在且会伤训练”。

**reviewer 为什么买账：** 这篇卖的是一个训练假设错误，而不是新 backbone。reviewer 容易买账的点是：假负样本问题在 TIReID 里直观存在，而且方法可以作为对比学习插件加到强模型上。novelty 来自“把负样本污染作为核心问题”，不是来自 GMM 或 queue 本身。

**4. FDGReID**
**触发观察：** 真实 ReID 部署同时有两个矛盾：跨域泛化需要多域信息，但隐私限制不允许直接共享数据；联邦学习能保隐私，但普通 FL 只是聚合本地偏置模型，没有解决域漂移。作者进一步把漂移观察拆成两类：风格差异，比如亮度、色调、纹理；视角差异，比如摄像头角度不同。

**重定义动作：** 它把问题从“联邦 ReID”改写成“联邦域泛化 ReID”。关键词是 federated domain generalization、style information sharing、viewpoint-aware contrastive learning、privacy-preserving domain drift。它的核心话术是：DG 方法不够隐私，FL 方法不够泛化，所以要在 FL 约束内做 DG。

**机制怎么长出来：** 风格漂移对应 SIS：客户端不传图，只传 feature mean/std 这类风格统计，服务器重分发风格描述，客户端用 AdaIN 做跨客户端风格迁移和外推。视角漂移对应 VCL：用 appearance 分支提外观，用 OpenPose 加 RGAT 提姿态结构，再用 viewpoint-aware contrastive loss 学视角不变表征，最后 AFF 融合。机制和重定义的高层绑定清楚，但组件本身偏拼装，AdaIN、pose、graph attention 都不是新东西。

**证据闭环：** 最关键是 ablation：去掉 SIS 和 VCL 性能很差；只加 VCL 的收益明显大于只加 SIS，支撑“视角漂移是重要来源”；去掉 viewpoint-aware contrastive loss 后明显下降，说明不是只靠多分支。它还在 DG 协议和 federated fully supervised 协议下比较，承认不一定打过专门 DG 方法，但在 FL 设定里明显强于 FedReID、FedAvg、FFReID。

**reviewer 为什么买账：** 这篇卖的是场景定义和系统视角。novelty 不在单个模块，而在“隐私约束下把域漂移拆成风格和视角两类再分别处理”。如果 reviewer 接受 federated ReID 是实际部署问题，它就有价值；如果只看纯方法新意，机制会显得常规。

**5. Find Hidden Modality Divergence**
**触发观察：** 无监督可见光红外 ReID 里，大家用对比学习拉近跨模态同类、推远异类。但作者指出，由于跨模态同类间隔很大，中间常常夹着异类样本。对比学习推远这些异类时，会阻碍真正同类跨模态样本靠近。

**重定义动作：** 它没有泛泛说“modality gap 大”，而是把问题改写成“跨模态优化过程中的 modality divergence”。关键词是 hidden modality divergence、negative optimization、positive optimization、adversarial aware learning。这个重定义把静态差距变成了动态优化冲突：同类聚合和异类分离在某些位置会互相打架。

**机制怎么长出来：** 既然问题是优化方向冲突，就要找出哪些 cluster centroid 正在产生正向或负向优化。ADAL 用当前 memory 和下一步 guide memory 的相似度变化判断某个 centroid 会让实例靠近还是远离。对正向优化继续促进，对负向优化反向惩罚。然后再补一个 instance-level easy-hard positive 约束，让跨模态 hard positive 向 easy positive 收拢。这个机制和重定义绑定很紧，因为它不是一般对齐，而是在处理“错误优化方向”。

**证据闭环：** 论文除了常规 SOTA，还设计了 MDRR 这种 modality divergence reduction rate，试图直接量化“divergence 是否被减少”。消融里分别加 Lp、Ln、Lco，证明促进正向、惩罚负向、easy-hard 收拢都有效。作为插件加到已有无监督 VI-ReID 方法上也能涨，说明它不是依赖某个特定 backbone。

**reviewer 为什么买账：** 这篇的真正 novelty 是视角和机制都相对绑定。它把老问题“跨模态 gap”讲成“对比学习内部的优化冲突”，再用 adversarial aware 的方式处理冲突。reviewer 买的是这个诊断，不只是最终指标。

**6. FLAG**
**触发观察：** 视频换衣 ReID 中，外观信息被衣服主导，换衣后外观可靠性下降；但视频里还有 gait。现有视频 ReID 方法虽然会建模时空信息，却没有显式把 appearance 和 gait 分开提取。silhouette 不含衣服纹理，天然更适合学 gait。

**重定义动作：** 它把 VCC-ReID 从“视频 ReID 的困难版本”改写成“有限 appearance 信息和 gait 信息如何显式学习的问题”。关键词是 explicit learning based on appearance and gait、multi-modal and multi-granularities、cross-modal knowledge distillation。它还把贡献包装成“把同衣视频 ReID 方法迁移到换衣设定的框架”，而不是单个新网络。

**机制怎么长出来：** 如果 appearance 和 gait 要显式解耦，就用 visible branch 学 appearance，用 silhouette branch 学 gait。因为 gait 既有全局轮廓也有局部身体结构，所以做多粒度 silhouette。因为多模态大模型推理成本高，所以再用 CMKDS 把 MGA teacher 蒸馏到只用 visible 输入的小模型。机制和问题绑定很清楚，尤其是“大模型高性能，小模型可部署”这一层很实用。

**证据闭环：** 它的关键证据不是单一 SOTA，而是泛化和消融：MGA 接到 AP3D、TCLNet 都涨，证明框架不是只适配一个方法；不同 silhouette 粒度显示三粒度最好，粒度过多会引入噪声，支撑“多粒度 gait”而不是越多越好；EPFM、并行融合、预训练冻结、直接加间接蒸馏都有消融；CMKDS 证明不增加推理成本也能涨。

**reviewer 为什么买账：** 这篇卖的是“显式 appearance/gait 分解”的框架视角，加上换衣视频这个明确应用场景。模块未必很新，但叙事强，因为 silhouette/gait 与换衣鲁棒性之间有天然因果关系。真正 novelty 是把视频 ReID 和 gait recognition 接起来，并用蒸馏解决部署问题。

**共同套路**
这批论文共同不是先发明模块，而是先抓一个旧范式里的错误默认：相邻帧都值得聚合、全局 VLM 足够、所有负样本都真负、FL 聚合自然泛化、对比学习方向总是有益、换衣后还能靠普通外观。然后它们把这个错误默认命名成一个新失败模式，并设计至少一个能直接验证该失败模式的消融、可视化或指标。我们要学的是这个结构：先找“大家以为 X，其实 Y”的可证伪观察，再让机制成为 Y 的自然修正，而不是先堆一个模块再倒找理由。
