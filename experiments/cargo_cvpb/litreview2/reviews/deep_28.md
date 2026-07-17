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
session id: 019ef836-1eb4-78e3-9baa-419362f53a00
--------
user
你是 ReID 创新方法论拆解员 28。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(28)==
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和工作区纪律，然后直接用 `pdftotext` 抽取这 5 篇的前 8 页方法部分。这个任务涉及 PDF 阅读，我会只取和拆解方法论相关的文本，不改任何文件。
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
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "sed -n '1,220p' ../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../GOAL.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc ls in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../CLAUDE.md: No such file or directory

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
我刚才确认到 `experiments/STATUS.md` 在上两级，`GOAL.md` 和 `CLAUDE.md` 不在那个位置，应该在项目根目录。我先补读根目录文件，然后开始抽取这批 PDF 的方法段。
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
/bin/zsh -lc "pdfinfo 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           
Subject:         
Keywords:        
Author:          
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.25
CreationDate:    Tue Apr 15 09:39:11 2025 CST
ModDate:         Tue Apr 15 09:39:11 2025 CST
Custom Metadata: yes
Metadata Stream: no
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           16
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       2417519 bytes
Optimized:       no
PDF version:     1.5

exec
/bin/zsh -lc "pdfinfo 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           
Subject:         
Creator:         LaTeX with hyperref
Producer:        LuaHBTeX, Version 1.24.0 (MiKTeX 26.1)
CreationDate:    Thu Mar 12 18:13:49 2026 CST
ModDate:         Thu Mar 12 18:13:49 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           16
Encrypted:       no
Page size:       486 x 720 pts
Page rot:        0
File size:       21972539 bytes
Optimized:       no
PDF version:     2.0

exec
/bin/zsh -lc "pdfinfo 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders
Subject:         -  Information systems  ->  Information retrieval.
Keywords:        visible-infrared person re-identification, text prompts, knowledge distillation
Author:          Hongde Zhang
Creator:         LaTeX with acmart 2024/12/28 v2.12 Typesetting articles for the Association for Computing Machinery and hyperref 2024-01-20 v7.01h Hypertext links for LaTeX
Producer:        XeTeX version 0.999996
CreationDate:    Wed Apr 30 16:47:24 2025 CST
ModDate:         Wed May 28 21:23:37 2025 CST
Custom Metadata: no
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            AcroForm
JavaScript:      no
Pages:           5
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       932240 bytes
Optimized:       no
PDF version:     1.5

exec
/bin/zsh -lc 'find ../../.. -name AGENTS.md -maxdepth 3 -print' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
../../../AGENTS.md

codex
PDF 技能的建议是需要版式时再渲染；这次用户明确要方法论文本，所以我按指定的 `pdftotext -f 1 -l 8` 走。接下来先定位每篇的方法标题和实验标题，避免只读摘要或漏掉方法段。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,260p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

不要堆砌小标题、加粗和表格来充字数。能用一段话讲清楚就用一段话。

不要用比喻性的网络口头禅。思考和写作都尽量用完整、清楚的中文句子。

exec
/bin/zsh -lc "pdfinfo 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Visible-Infrared Person Re-Identification With Real-World Label Noise
Subject:         IEEE Transactions on Circuits and Systems for Video Technology;2025;35;5;10.1109/TCSVT.2025.3526449
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.24; modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Sun Apr 27 10:36:14 2025 CST
ModDate:         Tue May  6 06:25:51 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           13
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       7673415 bytes
Optimized:       no
PDF version:     1.5

exec
/bin/zsh -lc "pdfinfo 'When Large Vision-Language Models Meet Person Re-Identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           When Large Vision-Language Models Meet Person Re-Identification
Author:          Qizao Wang; Bin Li; Xiangyang Xue
Creator:         arXiv GenPDF (tex2pdf:a6404ea)
Producer:        pikepdf 8.15.1
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           6
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       1699143 bytes
Optimized:       no
PDF version:     1.7

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
22:CCS Concepts: • Computing methodologies → Visual content-based indexing and retrieval;
23:Additional Key Words and Phrases: Person re-identification, modality discrepancies, feature decoupling
32:202021111218@mail.scuec.edu.cn; Zongze Li, Beidou Intelligent Technology (Shenzhen) Company Limited, Shenzhen,
45:ACM Reference format:
51:1
96:265:3
113:The research work presented in this article can be summarized in the following three aspects:
124:2
188:3
190:The Proposed Method
198:265:5
241:3.1 CDM
270:1 Õ
280:1 Õ
293:1 Õ
301:3.2
309:IN operation can be expressed as:
323:265:7
353:3.3
377:1 ÕÕ 𝑗
379:2𝐾 𝑖=1 𝑗=1 𝑖
389:2
404:The final adversarial decoupling loss can be expressed as:
412:3.4

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:1
8:Abstract—Text-based Person Retrieval (TPR) as a multimodal task, which aims to retrieve the target person from a
31:4.7% parameters. Code is available at https://github.com/LiuYating/UP-Person.
35:I. I NTRODUCTION
49:518071, China (e-mail: liuyatin21@mails.tsinghua.edu.cn).
54:518071, China (e-mail: lanxy@pcl.ac.cn; liuzm@pcl.ac.cn).
114:74.17%
120:68.19%
122:7.4M
127:150M
211:2
254:4.7% parameters.
256:II. R ELATED W ORK
310:3
399:III. P RELIMINARIES
421:4
452:V.
471:IV. M ETHOD
488:5
520:1
539:1
552:2
584:3
620:2
667:2
715:6
931:7
979:4(c), and their structures overlap and interact, which is prone
1037:1 XX
1055:V. E XPERIMENTAL S ETTINGS
1059:34, 054 images and 68, 126 texts. Both the validation set and
1061:078 images and 6, 158 texts, and the latter has 3, 074 images
1065:8
1152:4,102 identities. Each image corresponds to one description.
1189:VI. E XPERIMENTAL R ESULTS AND ANALYSIS
1202:2.38% and 3.15% on R@1 across three datasets with less finetuned parameters. Furthermore, with the CLIP-Large model

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
15:Abstract
33:Experiments on two widely used datasets show that the proposed
36:Expect answer: A boy wearing a green shortsleeved shirt, black trousers and grey shoes.
69:ACM Reference Format:
79:1 Introduction
98:2118
114:2𝑖 + 1). Using 𝑊𝑡𝑒𝑥𝑡 (𝑊𝑖𝑚𝑔 ) to represent the parameters of text
167:2i-th epoch
209:2
223:2.2
250:2.1
260:2119
264:1
279:2.3
296:Methods
322:34.9
323:35.9
324:38.1
325:36.9
326:43.8
327:52.9
328:42.4
329:40.7
330:45.9
331:54.3
332:65.34
333:62.25
334:69.42
335:74.63
336:66.82
337:64.88
338:72.64
339:76.19
340:68.05
341:65.17
342:72.43
343:77.58
344:68.72
345:66.29
346:72.52
347:78.17
348:70.59
349:67.49
350:75.98
351:80.24
352:71.68
353:68.79
354:76.97
355:81.94
356:74.66
357:71.73
358:79.69
359:83.68
360:74.7
361:71.8
362:80.3
363:83.3
364:75.18
365:78.40
366:70.12
367:81.82
368:66.34
369:62.51
370:68.15
371:74.09
372:67.53
373:64.98
374:71.66
375:76.52
376:74.44
377:71.17
378:78.98
379:82.75
380:75.2
381:71.8
382:82.0
383:85.0
384:78.64
385:83.12
386:74.22
387:85.03
412:70.93
413:66.04
414:48.1
415:48.9
416:57.9
417:53.6
418:80.35
419:72.46
420:90.61
421:84.53
422:91.66
423:85.96
424:83.60
425:75.67
426:85.34
427:76.39
428:87.87
429:85.09
430:94.51
431:88.67
432:91.1
433:85.1
434:91.07
435:85.23
436:89.12
437:84.43
438:84.83
439:76.55
440:84.71
441:83.28
442:89.0
443:84.7
444:95.14
445:89.62
450:48.5
451:49.3
452:56.3
453:53.4
454:79.37
455:72.26
456:88.77
457:81.61
458:92.97
459:86.08
460:81.97
461:73.78
462:83.88
463:75.16
464:86.95
465:84.34
466:93.64
467:87.61
468:89.5
469:83.4
470:92.18
471:86.59
472:88.38
473:83.86
474:84.16
475:75.13
476:87.18
477:84.72
478:88.2
479:82.8
480:94.33
481:89.01
483:Table 2: Ablation studies of each component in MPT network.
487:73.19
488:68.34
489:75.4
490:71.03
491:77.09
492:73.26
493:78.64
494:74.22
496:Methods
505:78.58
506:80.12
507:80.14
508:82.37
509:82.15
510:84.32
511:83.12
512:85.03
547:1 − 𝜖,
568:2.4
605:2120
613:3 Experiments
614:3.1 Datasets and Evaluation Protocols
621:3.2
626:Comparison with State-of-the-art Methods
628:3.4
644:3.3
666:Ablation Study
686:4
688:Conclusion
707:2121
714:2020. Hi-CMD: Hierarchical cross-modality disentanglement for visible-infrared
730:1835–1843.
755:2017. RGB-infrared cross-modality person re-identification. In Proceedings of the
773:2022. Fmcnet: Feature-level modality compensation for visible-infrared person
781:2122

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | nl -ba | sed -n '79,613p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    79	1 Introduction
    80	In this paper, we propose the Mutual Promotion Training (MPT)
    81	network for this issue. Generally, prompts that focus on person
    82	The Visible-Infrared person Re-IDentification (VI-ReID) task aims
    83	information can better assist in optimizing the image encoders.
    84	to match the visible images with the infrared images according
    85	Conversely, prompts optimized with the constraints of trained en∗
    86	Corresponding Author.
    87	coders will pay more attention to person information. Based on
    88	this idea, we attempt to introduce more person-related information into prompts. Specifically, we adopt an alternating training
    89	This work is licensed under a Creative Commons Attribution 4.0 International License.
    90	strategy which alternately optimizes the text prompts and image
    91	ICMR ’25, Chicago, IL, USA
    92	encoders in different epochs. When optimizing text prompts, we
    93	© 2025 Copyright held by the owner/author(s).
    94	design the modality knowledge propagation loss. By aligning preACM ISBN 979-8-4007-1877-9/2025/06
    95	https://doi.org/10.1145/3731715.3733496
    96	dicted class probability of features, the loss can enable prompts to
    97	
    98	2118
    99	
   100	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   101	
   102	Hongde Zhang and Bingpeng Ma
   103	
   104	focus on person-related information during the training process
   105	and encoders can benefit from continuously optimized prompts.
   106	However, because the optimization objectives of prompts and
   107	encoders are different, optimizing prompts and image encoders simultaneously may cause mutual interference. For example, when
   108	prompts learn person-related information from image encoders,
   109	the parameters of encoders are supposed not to be affected. To
   110	avoid the mutual interference, we propose the alternating training
   111	strategy to implement the mutual reinforcement between prompts
   112	and encoders. Specifically, we fix image encoders (prompts) and
   113	optimize the prompts (image encoders) when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 (𝑒𝑝𝑜𝑐ℎ =
   114	2𝑖 + 1). Using 𝑊𝑡𝑒𝑥𝑡 (𝑊𝑖𝑚𝑔 ) to represent the parameters of text
   115	prompts (image encoders) and 𝐹𝑡𝑒𝑥𝑡 (𝐹𝑖𝑚𝑔 ) to represent the optimization process of text prompts (image encoders), the optimization process can be formalized as:
   116	
   117	MPT Module
   118	Infrared
   119	Image
   120	Encoder
   121	
   122	I1
   123	I2
   124	
   125	Infrared
   126	Image
   127	Encoder
   128	
   129	IB
   130	A photo of [X]1 [X]2
   131	[X]3 [X]4 person.
   132	
   133	Text
   134	Encoder
   135	
   136	T1 T2
   137	
   138	prompts
   139	
   140	Visible
   141	Image
   142	Encoder
   143	
   144	IB
   145	Modality
   146	T B Knowledge
   147	Propagation
   148	Loss
   149	
   150	A photo of [X]1 [X]2
   151	[X]3 [X]4 person.
   152	
   153	prompts
   154	
   155	V1
   156	V2
   157	
   158	Text
   159	Encoder
   160	
   161	Visible
   162	Image
   163	Encoder
   164	
   165	VB
   166	
   167	2i-th epoch
   168	
   169	I1
   170	I2
   171	
   172	T1 T2
   173	
   174	Modality
   175	T B Alignment
   176	Loss
   177	
   178	V1
   179	V2
   180	VB
   181	
   182	(2i+1)-th epoch
   183	
   184	Fig. 2: The MPT network introduces an alternating training strategy,
   185	in which the modality knowledge propagation loss is designed for
   186	optimizing text prompts and the modality alignment loss is used
   187	to optimize image encoders. The lock represents fixed parameters,
   188	while the star indicates parameters being optimized.
   189	
   190	𝑓𝑖𝑥
   191	𝑊𝑡𝑒𝑥𝑡 = 𝐹𝑡𝑒𝑥𝑡 (𝑊𝑡𝑒𝑥𝑡 , 𝑊𝑖𝑚𝑔
   192	),
   193	
   194	learn more person-related information from the encoders. This significantly alleviates the problem that prompts lack person-related
   195	information. When training the image encoders, we propose the
   196	modality alignment loss. The loss treats the text features as a special intermediate modality, and aligns both visible and infrared
   197	modalities to the text modality. In this way, prompts can serve
   198	as the bridge for eliminating the discrepancies between the two
   199	modalities, helping the encoders generate better features. With the
   200	help of MPT, text prompts can help image encoders become better,
   201	and better image encoders can in turn promote prompts to learn
   202	more person-related information. Under such a positive feedback,
   203	both text prompts and image encoders will continue to improve.
   204	We evaluate the MPT network on two widely used datasets to
   205	demonstrate its effectiveness. Extensive experimental results show
   206	that the MPT network indeed solves the problems existing in the
   207	current methods and outperforms state-of-the-art methods.
   208	
   209	2
   210	
   211	𝑓𝑖𝑥
   212	𝑊𝑖𝑚𝑔 = 𝐹𝑖𝑚𝑔 (𝑊𝑖𝑚𝑔 , 𝑊𝑡𝑒𝑥𝑡
   213	),
   214	
   215	(1)
   216	
   217	(𝐸𝑝𝑜𝑐ℎ 2𝑖 + 1)
   218	
   219	(2)
   220	
   221	where the superscript 𝑓𝑖𝑥 indicates that the weight is fixed.
   222	
   223	2.2
   224	
   225	Modality Knowledge Propagation Loss
   226	
   227	The image encoders are trained on person images with labels, so
   228	they can capture details related to person identity. And such ability
   229	is exactly what prompts need most. Therefore, when optimizing
   230	prompts, our purpose is to propagate the knowledge contained in
   231	image encoders to prompts.
   232	The class probability distribution of a sample contains dark knowledge about the inter-class relationships. For example, classes with
   233	similar predicted probabilities for the same sample may be close to
   234	each other. Under such a premise, we propose the modality knowledge propagation loss. The loss constrains the predicted class probability of text features to be similar to that of image features. In
   235	this way, person-related knowledge is propagated from image encoders into text prompts. The loss function can be expressed as:
   236	
   237	The proposed method
   238	
   239	Some studies follow the two-stage training strategy in CLIP ReID
   240	to generate text prompts, but such strategy can only generate prompts
   241	which do not focus on person-related information. And the reason
   242	is that the CLIP model is unfamiliar to the ReID task, so it does not
   243	pay attention to details related to person identity. Therefore, how
   244	to generate prompts that better focus on person-related information remains an urgent problem to be solved.
   245	In this case, we propose the MPT network, which improves both
   246	prompts and image encoders by alternately optimizing them. The
   247	structure of MPT is shown in the Fig. 2. Specifically, the text encoder takes prompts as input and outputs corresponding text features, while image encoders take images as input and output corresponding image features. When 𝑒𝑝𝑜𝑐ℎ = 2𝑖, we fix image encoders and optimize prompts with modality knowledge propagation loss. And when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 + 1, we fix prompts and optimize
   248	image encoders with modality alignment loss.
   249	
   250	2.1
   251	
   252	(𝐸𝑝𝑜𝑐ℎ 2𝑖)
   253	
   254	Alternating Training Strategy
   255	
   256	Generally, encoders trained on VI-ReID datasets can help prompts
   257	focus on person information, while prompts which focus on person
   258	information can also help encoders extract better features. Therefore, the target of our network is that prompts can increasingly
   259	
   260	2119
   261	
   262	𝐿𝑚𝑘𝑝 =
   263	
   264	1
   265	∑ 𝐾𝐿(𝐹 𝐶(𝐼𝑖𝑘 ), 𝐹 𝐶(𝑇𝑖 ))
   266	|𝑃 (𝑦𝑖 )| 𝑘∈𝑃 (𝑦 )
   267	
   268	(3)
   269	
   270	𝑖
   271	
   272	where 𝐼𝑖𝑘 is the 𝑘-th image feature with the 𝑖-th identity and 𝑇𝑖
   273	is the text feature with the 𝑖-th identity (each identity only has
   274	one text prompt, so there is one text feature for each identity). 𝐹 𝐶
   275	denotes the shared classifier, and 𝐾𝐿(⋅) is the Kullback-Leibler divergence between two probabilities. 𝑃 (𝑦𝑖 ) is the set of indices of
   276	all positives image features for 𝑇𝑖 in the mini-batch and | ⋅ | is its
   277	cardinality.
   278	
   279	2.3
   280	
   281	Modality Alignment Loss
   282	
   283	Due to the fact that the high-level semantics contained in the text
   284	are often modality independent (such as body shape, hairstyle, etc.),
   285	we can consider optimized prompts as a special intermediate modality. When optimizing encoders, we treat the text features generated by prompts as the “bridge” and use them to connect the two
   286	modalities. In this way, the modality discrepancies can be effectively reduced.
   287	The proposed modality alignment loss first calculates the similarity between text features and image features, then utilizes contrastive learning loss to align the image features of both modalities
   288	
   289	Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders
   290	
   291	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   292	
   293	Table 1: Comparisons with SOTA methods on SYSU-MM01 and RegDB datasets. We report Rank at 𝑟 accuracy (%) and mAP (%).
   294	Dataset
   295	Setting
   296	Methods
   297	Hi-CMD [3]
   298	JSIA [14]
   299	AlignGAN [13]
   300	SPOT [1]
   301	PMWGCN [12]
   302	WF-CAMReViT [11]
   303	DART [17]
   304	MSSFCS [18]
   305	MAUM [7]
   306	CAL [16]
   307	DEEN [23]
   308	SEFL [4]
   309	FMCNet [22]
   310	PMT [8]
   311	ACD [10]
   312	CSDN [21]
   313	MPT (ours)
   314	
   315	SYSU-MM01
   316	All-search
   317	Indoor-search
   318	r=1
   319	mAP
   320	r=1
   321	mAP
   322	34.9
   323	35.9
   324	38.1
   325	36.9
   326	43.8
   327	52.9
   328	42.4
   329	40.7
   330	45.9
   331	54.3
   332	65.34
   333	62.25
   334	69.42
   335	74.63
   336	66.82
   337	64.88
   338	72.64
   339	76.19
   340	68.05
   341	65.17
   342	72.43
   343	77.58
   344	68.72
   345	66.29
   346	72.52
   347	78.17
   348	70.59
   349	67.49
   350	75.98
   351	80.24
   352	71.68
   353	68.79
   354	76.97
   355	81.94
   356	74.66
   357	71.73
   358	79.69
   359	83.68
   360	74.7
   361	71.8
   362	80.3
   363	83.3
   364	75.18
   365	78.40
   366	70.12
   367	81.82
   368	66.34
   369	62.51
   370	68.15
   371	74.09
   372	67.53
   373	64.98
   374	71.66
   375	76.52
   376	74.44
   377	71.17
   378	78.98
   379	82.75
   380	75.2
   381	71.8
   382	82.0
   383	85.0
   384	78.64
   385	83.12
   386	74.22
   387	85.03
   388	
   389	Venue
   390	CVPR-20
   391	AAAI-20
   392	ICCV-19
   393	TIP-22
   394	TIFS-24
   395	PR-24
   396	CVPR-22
   397	TMM-24
   398	CVPR-22
   399	ICCV-23
   400	CVPR-23
   401	CVPR-23
   402	CVPR-22
   403	AAAI-23
   404	TIFS-24
   405	arxiv-24
   406	-
   407	
   408	RegDB
   409	VIS to IR
   410	r=1
   411	mAP
   412	70.93
   413	66.04
   414	48.1
   415	48.9
   416	57.9
   417	53.6
   418	80.35
   419	72.46
   420	90.61
   421	84.53
   422	91.66
   423	85.96
   424	83.60
   425	75.67
   426	85.34
   427	76.39
   428	87.87
   429	85.09
   430	94.51
   431	88.67
   432	91.1
   433	85.1
   434	91.07
   435	85.23
   436	89.12
   437	84.43
   438	84.83
   439	76.55
   440	84.71
   441	83.28
   442	89.0
   443	84.7
   444	95.14
   445	89.62
   446	
   447	IR to VIS
   448	r=1
   449	mAP
   450	48.5
   451	49.3
   452	56.3
   453	53.4
   454	79.37
   455	72.26
   456	88.77
   457	81.61
   458	92.97
   459	86.08
   460	81.97
   461	73.78
   462	83.88
   463	75.16
   464	86.95
   465	84.34
   466	93.64
   467	87.61
   468	89.5
   469	83.4
   470	92.18
   471	86.59
   472	88.38
   473	83.86
   474	84.16
   475	75.13
   476	87.18
   477	84.72
   478	88.2
   479	82.8
   480	94.33
   481	89.01
   482	
   483	Table 2: Ablation studies of each component in MPT network.
   484	All-search
   485	Rank-1
   486	mAP
   487	73.19
   488	68.34
   489	75.4
   490	71.03
   491	77.09
   492	73.26
   493	78.64
   494	74.22
   495	
   496	Methods
   497	Baseline
   498	two-stage [21]
   499	MPT w/ 𝐿𝑀𝑃 𝑇
   500	MPT
   501	
   502	Indoor-search
   503	Rank-1
   504	mAP
   505	78.58
   506	80.12
   507	80.14
   508	82.37
   509	82.15
   510	84.32
   511	83.12
   512	85.03
   513	
   514	Baseline
   515	
   516	to the text features. Specifically, before calculating similarity, we
   517	use linear layers to project them into a common space. And the
   518	similarity between text and image features can be expressed as:
   519	
   520	Fig. 3: Illustration of visualization results using t-SNE. Features
   521	in visible (infrared) modality are represented by circles (triangles),
   522	while the various colors are used to distinguish different identities.
   523	
   524	(4)
   525	
   526	𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑖 ) = (𝑊𝐼 ⋅ 𝐼𝑖 ) ⋅ (𝑊𝑇 ⋅ 𝑇𝑖 ),
   527	
   528	where 𝑊𝐼 and 𝑊𝑇 are linear layers. Then the modality alignment
   529	loss can be expressed as:
   530	𝑁
   531	
   532	𝐿𝑚𝑎 = ∑ −𝑞𝑘 𝑙𝑜𝑔
   533	𝑘=1
   534	
   535	𝑒𝑥𝑝(𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑘 ))
   536	𝑁
   537	
   538	∑𝑎 𝑒𝑥𝑝(𝑠𝑖𝑚(𝐼𝑖 , 𝑇𝑎 ))
   539	
   540	,
   541	
   542	(5)
   543	
   544	where 𝑁 is both the number of identities in the dataset and the
   545	number of text prompts. 𝑞𝑘 is the value in the target distribution,
   546	and it can be expressed as:
   547	1 − 𝜖,
   548	𝑞𝑘 = { 𝜖
   549	𝑁,
   550	
   551	𝑦𝑘 = 𝑦 𝑖
   552	,
   553	𝑦𝑘 ≠ 𝑦 𝑖
   554	
   555	Initialization
   556	
   557	(6)
   558	
   559	where 𝑦𝑘 is the identity of the 𝑘-th image, and 𝜖 is a hyper-parameter
   560	used for the smoothing operation and is set to 0.1 as [6].
   561	Therefore, the mutual promotion loss is expressed as:
   562	𝐿𝑀𝑃 𝑇 = 𝐿𝑚𝑘𝑝 + 𝐿𝑚𝑎
   563	
   564	(7)
   565	
   566	With the help of the above loss functions, text prompts and image encoders can mutually promote each other, ultimately improving the performance of cross-modality retrieval.
   567	
   568	2.4
   569	
   570	Baseline
   571	
   572	Objective Functions
   573	
   574	MPT
   575	
   576	Fig. 4: The distribution of three models (i.e. Initialization, Baseline,
   577	and MPT network) between cross-modality features. Blue and green
   578	color are used to represent the intra-class distances and inter-class
   579	distances, respectively. Red lines represent the mean of inter-class
   580	distances and the mean of inter-class distances, and 𝛿𝑖 is the difference between the two type of means.
   581	
   582	𝐿𝑡𝑟𝑖 = [𝑚𝑎𝑥(𝑑𝑝 − 𝑑𝑛 + 𝜌)]+ ,
   583	
   584	(9)
   585	
   586	where 𝑝𝑘 is the prediction logits of 𝑘-th class, and 𝑑𝑝 (𝑑𝑛 ) denotes
   587	feature distance of positive (negative) pair. 𝜌 is the margin which
   588	is set to 0.3 following [6].
   589	Therefore, the total loss can be formulated as follow:
   590	
   591	To ensure the performance of the model, we employ the commonly
   592	used triplet loss and identity loss, which can be calculated as:
   593	𝑁
   594	
   595	𝐿𝑖𝑑 = ∑ −𝑞𝑘 𝑙𝑜𝑔(𝑝𝑘 ),
   596	
   597	MPT
   598	
   599	(8)
   600	
   601	𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑖𝑑 + 𝐿𝑡𝑟𝑖 + 𝐿𝑀𝑃 𝑇
   602	
   603	𝑘=1
   604	
   605	2120
   606	
   607	(10)
   608	
   609	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   610	
   611	Hongde Zhang and Bingpeng Ma
   612	
   613	3 Experiments

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '51,430p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    51	1
    52	
    53	Introduction
    54	
    55	Person re-identification aims to match a query image of an individual with gallery images captured
    56	by multiple non-overlapping cameras. This technology has significant practical value in surveillance
    57	tasks, making it a research hotspot in the field of computer vision. Although most existing methods
    58	primarily focus on visible light scenarios, utilizing color information as an important cue for
    59	identification, these methods are limited in low-light conditions as visible light cameras struggle to
    60	capture effective color information, thus restricting their applicability.
    61	With the advancement of surveillance technology, an increasing number of new cameras are
    62	equipped with both visible light and infrared dual modalities, allowing them to automatically
    63	switch modes between day and night, thereby achieving around-the-clock monitoring. To address
    64	this emerging technological demand, Wu et al. [40] pioneer the study of cross-modality person
    65	re-identification between visible and infrared images, thereby establishing this research field and
    66	significantly expanding the application scenarios of person re-identification. However, despite
    67	the progress made, several challenges remain unresolved. One of the key issues is the significant
    68	modality gap between visible light and infrared images, which makes it difficult to effectively
    69	match person identities between these different modalities. The appearance differences caused
    70	by differences in lighting conditions, background, and the inherent properties of visible light and
    71	infrared imaging complicate feature extraction and alignment. Therefore, how to effectively bridge
    72	the modality gap between visible light and infrared images becomes the key to improving the
    73	accuracy and robustness of cross-modality person re-identification.
    74	In modal-shared feature spaces, identity information is always inevitably coupled with other
    75	identity-irrelated information. Camera-related features, induced by differences in imaging mechanisms, viewpoints, lighting, and backgrounds, lack identity consistency and often interfere with
    76	identity learning. In contrast, camera-irrelated features robustly capture identity information and
    77	remain stable in cross cameras and modalities. Additionally, due to the differences in imaging principles between visible and infrared spectra, cross-modality person re-identification must contend
    78	not only with feature variations within a single modality but also with cross-modality feature
    79	heterogeneity arising from differences in reflectance in the visible spectrum and emissivity [27, 35]
    80	in the infrared spectrum.
    81	Most existing works [1, 38, 41, 46] focused on extracting identity-related features or rely on a
    82	single decoupling strategy to address the effects of modal differences, to achieve identity-based
    83	matching and similarity ranking. However, in the modality-shared feature space, identity information is often intertwined with camera noise and modality-sensitive redundancy, resulting in the
    84	weakening of discriminative features. This means that a single-stage decoupling strategy may find
    85	it difficult to simultaneously mitigate the impact of modality and camera differences, resulting in a
    86	decrease in the discriminative ability of modality-shared features, thereby affecting the performance
    87	of cross-modal person re-identification.
    88	To address the above challenges, this study proposes an innovative cross-modality person
    89	reidentification framework—the Hierarchical Decoupling Network (HDNet), which includes
    90	two main modules: Camera Decoupling Module (CDM) and Modality-Invariant Identity
    91	Decoupling Module (MIDM). This framework aims to progressively decouple camera difference
    92	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
    93	
    94	Visible-Infrared Person Re-Identification
    95	
    96	265:3
    97	
    98	features, modality difference features, and identity-irrelevant redundant features, and significantly
    99	enhance the discriminative performance of cross-modality person re-identification through this
   100	systematic feature purification mechanism. Specifically, HDNet introduces several innovative
   101	aspects to address the challenges in visible-infrared person re-identification. First, it adopts a
   102	hierarchical feature decoupling approach. CDM decouples camera-related features and camerairrelated features from modality-shared features to mitigate the impact of camera variations. This
   103	is achieved by adopting an adversarial approach with Gradient Reversal Layers (GRL) [8] to
   104	minimize the identity information in camera-related features, allowing the network to focus more
   105	on identity-specific features. Then, the MIDM further decouples identity-dependent and identityindependent features from camera-irrelated features, using residual features to recover the lost
   106	identity information. In MIDM, an adversarial decoupling loss is designed to optimize the feature
   107	decoupling process and promote a better separation of identity features and redundant features. This
   108	loss function encourages identity-dependent features to be clustered more tightly and redundant
   109	features to be clustered more dispersedly, enhancing the discriminative power of identity features.
   110	And, a Feature Cross-Modal Recombination Strategy (FCRS) is introduced to further reduce
   111	the modality differences of identity features, which guides the network to focus on the shared
   112	identity information across modalities, improving the robustness and accuracy of the system.
   113	The research work presented in this article can be summarized in the following three aspects:
   114	— HDNet, which includes CDM and MIDM, is an innovative end-to-end feature purification
   115	framework that utilizes a multi-level decoupling strategy (camera-modality-identity), ensuring
   116	maximal preservation of identity information at each level, while effectively eliminating
   117	interference.
   118	— CDM decouples camera-related and camera-irrelated features from modality-shared features
   119	by adopting an adversarial approach with the GRL and minimizes the identity information
   120	in camera-related features, thereby allowing the network to focus more on identity-specific
   121	features.
   122	— MIDM utilizes adversarial decoupling loss to optimize identity disentanglement information and introduces the FCRS to focus on the shared identity information by pairing and
   123	recombining residual identity features from different modalities.
   124	2
   125	
   126	Related Work
   127	
   128	Person Re-Identification. Person re-identification refers to the problem of pedestrian retrieval across
   129	multiple surveillance cameras [57]. The main challenges arise from variations in camera environments, pose changes, and viewpoint differences. Traditional person re-identification methods
   130	are based on visible light cameras and can be divided into two categories [15, 29, 45, 56]. One
   131	is based on feature extraction methods [25, 31], which usually introduce the ID-discriminative
   132	Embedding model [58] and regard each identity as a different category to obtain fine-grained
   133	global features; or combining local features with full-body representation to make it robust to
   134	misalignment changes. Cheng et al. [5] design a multi-channel parts-aggregated deep convolutional
   135	network by integrating local body part features and global full-body features. The other is the
   136	metric learning-based method [2, 12, 30], which aims to learn discriminative features by reducing
   137	the feature distance between pedestrian images. Hermans et al. [12] demonstrate that using a
   138	variant of the triplet loss is beneficial for discriminative Re-ID model learning. Zhou et al. [60]
   139	present a point to set similarity for deep metric learning, which can jointly minimize the intra-class
   140	distance and maximize the inter-class distance. Although these methods have achieved effective
   141	performance in traditional person re-identification, they are not fully applicable to cross-modal
   142	person re-identification, thus requiring additional methods to handle modality-specific information.
   143	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   144	
   145	265:4
   146	
   147	H. Ding et al.
   148	
   149	Visible-Infrared Person Re-Identification. Visible-infrared person re-identification aims to match
   150	pedestrian images from different modalities despite significant spectral differences, focusing on
   151	feature embedding, optimization, and metric learning [38, 51, 52]. Existing methods include the
   152	deep zero-padding single-stream network by Wu et al. [40], which adapts to different modalities
   153	by zero-padding, capturing modality-specific information. Ye et al. [48] propose a dual-stream
   154	CNN, where each stream processes visible and infrared images, using a shared backbone to extract
   155	modality-shared features. Zhuang et al. [61] design camera-based batch normalization to eliminate
   156	feature distribution differences caused by different cameras. Wang et al. [36] propose an attentive
   157	modality-consistent network that enhances feature learning through a residual attention module.
   158	Zhang et al. [54] propose a non-linear Middle Modality Generator to unify visible and infrared
   159	images, reducing modality discrepancy, with Distribution Consistency Loss ensuring distribution consistency. Ye et al. [47] introduce a channel-enhanced joint learning strategy to improve
   160	robustness against color variations. Li et al. [17] propose a Dynamic Weighted Gradient Reversal
   161	Network that leverages adversarial training and adaptive dynamic weights to reduce cross-modality
   162	discrepancies and improve visible-infrared person re-identification accuracy. Zhang et al. [49]
   163	propose a Hybrid Modality Metric Learning Framework, which addresses various variations in
   164	cross-modality person re-identification by applying similarity constraints at both the class and
   165	modality levels. However, most existing methods fail to fully consider the impact of camera differences. Although current approaches employ shared networks to learn modality-shared features,
   166	they struggle to simultaneously mitigate the effects of both modality and camera differences,
   167	leading to a reduction in the discriminative power of modality-shared features and, consequently,
   168	affecting the performance of cross-modality person re-identification.
   169	Feature Decoupling. Feature disentanglement aims to extract different factors from input data for
   170	meaningful feature representation, improving person matching. In visible-infrared re-identification,
   171	Wang et al. [37] first use disentanglement to generate cross-modal paired images and map them to a
   172	unified space. Choi et al. [6] propose a Hierarchical Cross-Modal Disentanglement (Hi-CMD)
   173	network to separate identity-discriminative and identity-excluding factors. Pu et al. [24] introduce a
   174	Dual-Gaussian VAE to decompose identity-related and identity-ambiguous subspaces, while Kansal
   175	et al. [16] propose a spectral disentanglement method that separates modality-specific spectral
   176	information. Lu et al. [22] use tri-level disentanglement to address differences at different levels,
   177	and Hu and Hu [14] employ dual-adversarial learning to disentangle modality-specific features
   178	effectively. Feng et al. [7] propose a Shape-Guided dIverse fEature Learning (SGIEL), which
   179	decomposes shape-related and shape-erased features into two orthogonal subspaces, enhancing
   180	feature diversity. Chen et al. [3] suggest Identity Feature Disentanglement, which reduced the
   181	modality gap in visible-infrared person re-identification by separating identity-relevant and identityirrelevant information.
   182	Existing decoupling methods do not fully consider the complex relationships between modalityshared features and other redundant features, which are not simply linearly combined. As a result,
   183	relying on a single decoupling strategy to simultaneously address modality and camera differences
   184	is quite challenging. To overcome this, this article proposes an end-to-end feature purification
   185	framework that employs a multi-level decoupling strategy (camera-modality-identity). This ensures
   186	that identity information is preserved as much as possible at each level while effectively removing interference, thus avoiding the incomplete or overly simplified decoupling issues present in
   187	existing methods.
   188	3
   189	
   190	The Proposed Method
   191	
   192	As shown in Figure 1, HDNet adopts a dual-stream network with visible and infrared branches [48]
   193	to extract image features. The dual-stream network splits ResNet50 into modality-specific layers
   194	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   195	
   196	Visible-Infrared Person Re-Identification
   197	
   198	265:5
   199	
   200	Fig. 1. Framework of HDNet. HDNet uses a dual-stream network with visible and infrared branches to extract
   201	image features. Pedestrian images X are processed to obtain a feature map 𝐹 , which is decoupled in the
   202	CDM network into camera-related features 𝐹𝑐𝑟 and camera-irrelated features 𝐹𝑐𝑖 . A GRL minimizes identity
   203	information in 𝐹𝑐𝑟 , while undergoes Instance Normalization (IN) to produce normalized features 𝐹e𝑐𝑖 . Residual
   204	feature 𝑅 is used to retain identity information, with positive residual features 𝑅 + and negative residual 𝑅 −
   205	features separated and recombined with 𝐹e𝑐𝑖 to generate identity features 𝐹𝑖𝑑 and redundant features 𝐹𝑛 .
   206	Adversarial decoupling loss 𝐿𝑎𝑑𝑣 and recombination center loss 𝐿𝑟 further reduce cross-modality differences
   207	and enhance the consistency of identity information.
   208	
   209	(Block1, Block2) and modality-shared layers (Block3, Block4, Block5). For a given pedestrian image
   210	X, the feature map obtained after being input into the feature extraction network is denoted as
   211	𝐹 ∈ Rℎ×𝑤×𝑐 , where the height and width of the feature map 𝐹 are ℎ and 𝑤, respectively, and the
   212	number of channels is 𝑐.
   213	The obtained feature map is input into the CDM network, where camera-related features 𝐹𝑐𝑟
   214	and camera-irrelated 𝐹𝑐𝑖 features are decoupled. In CDM, camera-related features, 𝐹𝑐𝑟 , are those
   215	that can correctly classify different cameras and do not contain identity information. 𝐹𝑐𝑖 called
   216	camera-irrelated features are those that are related to identity information after the initial extraction
   217	of shared features, and the camera differences are removed. Subsequently, 𝐹𝑐𝑖 is further refined
   218	through the following network. MIDM first performs Instance Normalization (IN) to obtain normalized features 𝐹e𝑐𝑖 . Although IN alleviates modality discrepancies, it may filter out some identity
   219	information from the original features. To compensate 𝐹e𝑐𝑖 , the residual features between the original
   220	features 𝐹𝑐𝑖 and the IN features 𝐹e𝑐𝑖 are used as the decoupling target. From these residual features
   221	𝑅, positive residual features 𝑅 + (containing identity-related information) and negative residual
   222	features 𝑅 − (containing identity-unrelated information) are separated, 𝑅 + and 𝑅 − are recombined
   223	with 𝐹e𝑐𝑖 to obtain identity features 𝐹𝑖𝑑 and redundant features 𝐹𝑛 . An adversarial decoupling loss
   224	𝐿𝑎𝑑𝑣 is designed to optimize the effectiveness of feature decoupling. The positive residual feature
   225	from the residual feature contains visible modality and infrared modality. In order to further reduce
   226	the differences between cross-modality features under the same modality, the positive residual
   227	feature 𝑅 + and the normalized feature 𝐹e𝑐𝑖 are first split by modality into 𝐹e𝑐𝑖𝑣 , 𝐹e𝑐𝑖𝑡 , 𝑅𝑣+ , and 𝑅𝑡+ ,
   228	followed by exchange and recombination. FCRS exchanges the positive residual features 𝑅𝑣+ and
   229	𝑅𝑡+ from the two modalities and recombines them with the split features 𝐹˜𝑐𝑖𝑣 and 𝐹˜𝑐𝑖𝑡 to generate
   230	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   231	
   232	265:6
   233	
   234	H. Ding et al.
   235	
   236	cross-modal recombinant features. Meanwhile, the positive residual features from the unexchanged
   237	modality combined with the corresponding split features are referred to as same-modality recombinant features. The two recombinant features are optimized using the recombinant center loss 𝐿𝑟 ,
   238	enabling the model to explore the commonalities of identity information across modalities more
   239	deeply and further reduce modality differences. Additionally, 𝐿𝑏𝑎𝑠𝑒 represents the commonly used
   240	the sum of cross-modal identification loss 𝐿𝑖𝑑 and 𝐿𝑏ℎ𝑡𝑟𝑖 in the re-identification process.
   241	3.1 CDM
   242	We are inspired by SE-Net [13] and hope to use channel masks 𝑀 to decouple features 𝐹 and adaptively separate camera-related features 𝐹𝑐𝑟 and camera-irrelated information 𝐹𝑐𝑖 . From Equations
   243	𝐶
   244	𝐶
   245	(1) to (3), 𝑔 (𝐹 ) represents global average pooling, 𝑊1 ∈ R 𝑟 ×𝐶 and 𝑊2 ∈ R 𝑟 ×𝐶 represent two fully
   246	connected layers, and 𝛿 is Sigmoid function. To improve adaptability and reduce the number of
   247	parameters, the dimension reduction ratio is set to 16.
   248	𝑀 = 𝑊2𝛿 (𝑊1𝑔 (𝐹 )),
   249	
   250	(1)
   251	
   252	𝐹𝑐𝑟 = 𝑀 × 𝐹,
   253	
   254	(2)
   255	
   256	𝐹𝑐𝑖 = (1 − 𝑀) × 𝐹 .
   257	
   258	(3)
   259	
   260	In order to constrain the mask to extract the features we need, we leverage camera information
   261	labels to guide the feature learning, ensuring a stronger focus on camera-related classification.
   262	Meanwhile, the other branch is trained with identity labels to enforce identity-specific feature
   263	constraints. As in Equations (4) and (5), where 𝑁 is the batch size, we first train the Camera Classifier
   264	𝐶𝑐𝑎𝑚 and Identity Classifier 𝐶𝑐𝑖𝑑 to correctly predict the camera labels 𝑦𝑐𝑎𝑚 and identity labels 𝑦𝐼 𝐷 ,
   265	respectively.
   266	𝑁
   267	
   268	𝐿𝑐𝑖𝑑 = −
   269	
   270	1 Õ
   271	log (𝑦𝐼 𝐷 k𝐶𝑐𝑖𝑑 (𝐹𝑐𝑖 )),
   272	𝑁 𝑖=1
   273	
   274	(4)
   275	
   276	𝑁
   277	
   278	𝐿𝑐𝑎𝑚 = −
   279	
   280	1 Õ
   281	log (𝑦𝑐𝑎𝑚 k𝐶𝑐𝑎𝑚 (𝐹𝑐𝑟 )).
   282	𝑁 𝑖=1
   283	
   284	(5)
   285	
   286	As in Equation (5), to enhance the adversarial nature between two features and retain as much
   287	identity information as possible in the camera-related feature 𝐹𝑐𝑟 , second, we apply an identity
   288	classification loss with the GRL for 𝐹𝑐𝑖 .
   289	𝑁
   290	
   291	𝐿𝑔𝑟𝑙 = −
   292	
   293	1 Õ
   294	log (𝑦𝐼 𝐷 |𝐺𝑅𝐿(𝐶𝑐𝑖𝑑 (𝐹𝑐𝑖 ))).
   295	𝑁 𝑖=1
   296	
   297	(6)
   298	
   299	Thus, the total camera decoupling loss is:
   300	𝐿𝑐 = 𝐿𝑐𝑖𝑑 + 𝐿𝑐𝑎𝑚 + 𝐿𝑔𝑟𝑙 .
   301	3.2
   302	
   303	(7)
   304	
   305	MIDM
   306	
   307	To further alleviate modality discrepancies, we apply IN to the camera-irrelated features 𝐹𝑐𝑖 . IN can
   308	reduce the variance between instance features and enhance the model’s generalization ability. The
   309	IN operation can be expressed as:
   310	
   311	
   312	𝐹𝑐𝑖 − 𝜇 (𝐹𝑐𝑖 )
   313	𝐹˜𝑐𝑖 = IN(𝐹𝑐𝑖 ) = 𝜏
   314	+ 𝜖,
   315	(8)
   316	𝜎 (𝐹𝑐𝑖 )
   317	where 𝜇 and 𝜎 represent the mean and standard deviation calculated across the spatial dimensions
   318	for each independent instance feature. The scaling parameter 𝜏 ∈ R𝐶 and the shift parameter
   319	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   320	
   321	Visible-Infrared Person Re-Identification
   322	
   323	265:7
   324	
   325	Fig. 2. The optimization objective of the adversarial decoupling loss. When 𝐹˜𝑐𝑖 is recombined with 𝑅 + , it
   326	results in tighter identity clusters, while recombining with 𝑅 − leads to more dispersed clusters. Solid circles
   327	represent different identities, stars indicate identity centers, “intra” represents intra-class distance, and “inter”
   328	denotes inter-class distance.
   329	
   330	𝜖 ∈ R𝐶 are learnable. IN ensures the independence of each instance feature and reduces the mutual
   331	influence between features from different modalities.
   332	Although IN can filter out some mode-specific information, it may also neglect identity-related
   333	information embedded in instance features, thereby affecting the identity-discriminative ability of
   334	instance features. Hence, using IN on 𝐹𝑐𝑖 may lead to a loss of identity information. We consider
   335	the residual features of 𝐹𝑐𝑖 as the decoupling target to recover the lost identity information. The
   336	residual features 𝑅 ∈ R𝐻 ×𝑊 ×𝐶 can be represented as:
   337	𝑅 = 𝐹𝑐𝑖 − 𝐹˜𝑐𝑖 .
   338	
   339	(9)
   340	
   341	Following the decoupling method in Section 3.1, the residual features are split into positive
   342	residual features 𝑅 + (identity-related information) and negative residual features 𝑅 − (identityunrelated information).
   343	Next, we recombine 𝐹˜𝑐𝑖 , 𝑅 + , and 𝑅 − to obtain the final identity and redundant features as follows:
   344	𝐹𝑖𝑑 = 𝐹˜𝑐𝑖 + 𝑅 +,
   345	𝐹𝑛 = 𝐹˜𝑐𝑖 + 𝑅 − .
   346	
   347	(10)
   348	(11)
   349	
   350	In this way, we decouple the residual features and recombine the identity and redundant features.
   351	In essence, 𝐹𝑖𝑑 and 𝐹𝑛 represent the identity-related features and redundant features decoupled
   352	from the camera-irrelated features 𝐹˜𝑐𝑖 , respectively.
   353	3.3
   354	
   355	Adversarial Decoupling Loss
   356	
   357	To guide the channel mask 𝑀 to adaptively extract identity information from 𝐹˜𝑐𝑖 , we propose
   358	an adversarial decoupling loss. Figure 2 illustrates the optimization goal of this loss. When 𝐹˜𝑐𝑖 is
   359	recombined with 𝑅 + , it contains more identity-related information, resulting in tighter clusters. In
   360	contrast, when 𝐹˜𝑐𝑖 is recombined with 𝑅 − , the identity information mixes with modality-irrelevant
   361	features, causing the clusters to become more dispersed. In the figure, solid circles of different
   362	colors represent samples of different IDs, the star represents the center of the samples for that ID,
   363	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   364	
   365	265:8
   366	
   367	H. Ding et al.
   368	
   369	“intra” denotes the intra-class distance for the same ID, and “inter” indicates the inter-class distance
   370	between different ID centers.
   371	𝑃
   372	
   373	𝐷 intra (f) =
   374	
   375	𝐾
   376	
   377	1 ÕÕ 𝑗
   378	kf − 𝜇𝑐𝑖 k 22,
   379	2𝐾 𝑖=1 𝑗=1 𝑖
   380	𝑃 −1
   381	
   382	𝐷 inter (f) =
   383	
   384	(12)
   385	
   386	𝑃
   387	
   388	Õ Õ
   389	2
   390	k𝜇𝑘 − 𝜇 𝑗 k 2 .
   391	𝑃 (𝑃 − 1)
   392	
   393	(13)
   394	
   395	𝑘=1 𝑗=𝑘+1
   396	
   397	𝑃 represents the number of classes, 𝐾 is the number of samples per class. f𝑖𝑗 is the feature of the
   398	𝑗th sample from class 𝑖, and 𝜇𝑐𝑖 is the center of class 𝑖. 𝜇𝑘 and 𝜇 𝑗 represent the class centers of class
   399	𝑘 and class 𝑗, respectively.
   400	𝐿𝑎𝑑𝑣 (𝑎, 𝑏) = softplus(𝐷 intra (𝑎) − 𝐷 intra (𝑏)) + softplus(𝐷 inter (𝑏) − 𝐷 inter (𝑎)).
   401	
   402	(14)
   403	
   404	The final adversarial decoupling loss can be expressed as:
   405	𝐿𝑎𝑑𝑣 = 𝐿𝑎𝑑𝑣 (𝐹𝑖𝑑 , 𝐹˜𝑐𝑖 ) + 𝐿𝑎𝑑𝑣 ( 𝐹˜𝑐𝑖 , 𝐹𝑛 ).
   406	
   407	(15)
   408	
   409	In summary, the feature decoupling module not only decouples identity-discriminative features
   410	with stronger robustness, reducing the degree of interference from identity-unrelated information during training, but also alleviates the modality discrepancies between visible and infrared
   411	modalities.
   412	3.4
   413	
   414	Feature Cross-Modality Recombination Strategy
   415	
   416	To further reduce the impact of modality differences and guide the network to focus on shared
   417	features, we decompose 𝐹e𝑐𝑖 and 𝑅 + by modality into 𝐹˜𝑐𝑖𝑣 , 𝐹˜𝑐𝑖𝑡 , 𝑅𝑣+ , and 𝑅𝑡+ , and then cross recombine
   418	them as follows:
   419	𝐹˜𝑣_𝑣 = 𝐹˜𝑐𝑖𝑣 + 𝑅𝑣+,
   420	𝐹˜𝑡 _𝑡 = 𝐹˜𝑐𝑖𝑡 + 𝑅𝑡+,
   421	
   422	(16)
   423	
   424	𝐹˜𝑡 _𝑣 = 𝐹˜𝑐𝑖𝑡 + 𝑅𝑣+,
   425	𝐹˜𝑣_𝑡 = 𝐹˜𝑐𝑖𝑣 + 𝑅𝑡+ .
   426	
   427	(18)
   428	
   429	(17)
   430	(19)

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*|.*Dataset.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
33:1. INTRODUCTION
90:14] enhance LLMs by incorporating visual perception and understanding, demonstrating considerable potential in multi-modal learning tasks. However, integrating LVLMs with person ReID remains
114:4x
158:semantic token to enhance identity representations. (3) Experimental results show that, without requiring additional annotations, our
161:2. METHODOLOGY
162:2.1. Overview of LVLM
191:4
196:2.2. Pedestrian Semantic Token Generation
230:2.3. Semantic-Guided Interaction
253:2.4. Optimization and Inference
293:78.4
294:74.8
295:75.5
296:79.0
297:78.6
298:76.8
299:76.4
301:88.7
302:86.6
303:87.9
304:89.0
305:89.7
306:88.6
307:87.2
309:86.9
310:86.0
311:88.0
312:88.2
313:88.9
314:88.4
315:86.0
316:87.0
318:95.7
319:94.8
320:96.1
321:95.7
322:95.2
323:96.1
324:95.1
325:94.5
327:67.4
328:76.4
329:76.9
330:75.6
331:77.4
334:68.0
335:80.1
336:78.9
337:77.9
338:81.1
353:80.6
354:78.2
355:80.1
356:80.0
357:82.5
359:89.6
360:88.8
361:89.0
362:90.1
363:90.0
365:88.2
366:88.0
367:87.5
368:88.0
369:89.6
371:95.0
372:95.4
373:94.7
374:95.4
375:95.5
377:79.0
378:80.3
380:80.3
381:81.6
383:55.7
384:53.6
385:58.2
386:59.5
388:64.2
389:64.5
390:67.1
391:67.1
397:82.8
399:92.2
401:89.2
403:95.6
405:82.3
407:84.6
409:59.8
411:68.1
415:Methods
436:3. EXPERIMENTS
437:3.1. Experimental Settings
438:Datasets. We evaluate our methods on four person ReID datasets:
445:14, respectively, resulting in n = 50. In other words, 50 visual
454:3.2. Comparison with State-of-the-Art Methods
458:Ablation studies of our key two components on DukeMTMC-reID
461:Table 3. Ablation of the camera semantic supplementation
482:79.0
483:80.9
484:79.0
486:90.2
487:91.0
488:90.0
490:87.3
491:88.3
492:87.3
494:94.7
495:95.0
496:94.5
498:Methods
502:82.8
504:92.2
506:89.2
508:95.6
514:Methods
528:81.6
529:82.3
530:82.8
532:91.4
533:92.1
534:92.2
536:89.1
537:88.4
538:89.2
540:95.2
541:95.3
542:95.6
560:3.3. Ablation Studies
577:Ablation of the camera semantic supplementation strategy. In
616:4. CONCLUSION
622:rich semantics of pedestrians. Experimental results show the significance of LVLM-based semantic guidance in enhancing visual representations, paving the way for future research in this direction.
637:2023.
647:1034.
659:2023.
677:17–35.
688:282.
703:3195.
731:2023.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '399,1058p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   399	III. P RELIMINARIES
   400	PETL keeps the pre-trained model frozen and tunes a small
   401	number of learnable parameters. Several state-of-the-art PETL
   402	methods are introduced in the following.
   403	Adapter. Adapter [37] inserts small modules into transformer layers, which typically employs a down-projection
   404	Wdown ∈ Rd×m to map the input x to a lower-dimensional
   405	space defined by the bottleneck dimension m, followed by a
   406	nonlinear activation function f like ReLU function and an upprojection with Wup ∈ Rm×d . Finally, adapter is incorporated
   407	with a residual connection, the output x′ is formulated as:
   408	x′ ← x + f (xWdown ) Wup .
   409	
   410	(1)
   411	
   412	The vanilla sequential structure [37] positions two above
   413	adapters in series within a layer of the transformer: one
   414	following the MHA sublayer and another following the MLP
   415	sublayer. He et al. [58] have proposed an alternative adapter
   416	variant that is parallel with MHA or MLP sublayer:
   417	h′ ← h (x) + f (xWdown ) Wup ,
   418	
   419	(2)
   420	
   421	4
   422	
   423	where h(x) is the output of the original x by MHA or MLP,
   424	and h′ represents the final output with a parallel adapter.
   425	LoRA. LoRA [21] incorporates low-rank trainable matrices into transformer layers with the aim of providing an
   426	approximation to weight updates. For a pre-trained weight
   427	matrix W ∈ Rd×k , LoRA can update it with a low-rank
   428	decomposition:
   429	W + ∆W = W + Wdown Wup ,
   430	
   431	(3)
   432	
   433	where Wdown ∈ Rd×r and Wup ∈ Rr×k are learnable.
   434	For the input x to a linear projection y = xW , LoRA alters
   435	the output y = xW to y ′ in the following manner:
   436	y ′ ← x (W + s · Wdown Wup ) = xW + s · xWdown Wup , (4)
   437	where s is a learnable scalar hyper-parameter. It is noteworthy
   438	that LoRA as a reparameterized method, can merge the weight
   439	updates into original weights during inference stage, which
   440	decreases computation costs.
   441	Prefix and Prompt Tuning. They incorporate tunable
   442	tokens, where the former is added to the input of a transformer
   443	block, the latter is prepended to the keys and values of
   444	attention. We will separately explicate the two methods.
   445	Formally, for the input x ∈ Rn×d of attention module, the
   446	original query, key, and value are denoted as Q = xWq , K =
   447	xWk , V = xWv . Attention (Attn) is formulated as:
   448	
   449	
   450	QK T
   451	√
   452	V.
   453	Attn (Q, K, V ) = softmax
   454	d
   455	Prefix Tuning (Pre-T) [22] prepends two prefix tokens
   456	Pk , Pv ∈ Rl×d to K, V ∈ Rn×d respectively. Therefore,
   457	attention is modified as:
   458	y ′ = Attn (Q, [Pk ; K] , [Pv ; V ]) ,
   459	
   460	(5)
   461	
   462	where [·; ·] stands for concatenation operation. The multi-head
   463	attention (MHA) with h heads is omitted for brevity.
   464	Prompt Tuning (Pro-T) [59] prepends prompts P ∈ Rl×d
   465	to the input tokens x ∈ Rn×d , which is equivalent to
   466	concatenate the same prompt P W to xW , denoted as:
   467	y ′ = Attn([P Wq ; Q], [P Wk ; K], [P Wv ; V ]).
   468	
   469	(6)
   470	
   471	IV. M ETHOD
   472	In this section, we illustrate each component of our method
   473	in detail. First, we provide a detailed description of our image
   474	and text encoder in Section IV-A. Then, the architecture of
   475	the unified PETL is explained in Section IV-B. Finally, we
   476	present the proposed submodules S-Prefix and L-Adapter in
   477	Section IV-C.
   478	A. Feature Extraction
   479	For a set of person images I = {I1 , I2 , . . . , In } paired
   480	with corresponding text queries T = {T1 , T2 , . . . , Tn }, textbased person image retrieval is carried out by evaluating the
   481	similarity between each text query and every image, and then
   482	the target person image is returned with the highest similarity
   483	score. The initial phase involves feature extraction for both
   484	vision and language branches.
   485	
   486	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   487	
   488	5
   489	
   490	L-Adapter
   491	
   492	Fine-tuned
   493	
   494	LoRA
   495	
   496	UP-Person block
   497	
   498	Concatenate
   499	
   500	L-Adapter
   501	𝑊𝑢𝑝
   502	
   503	MLP
   504	
   505	Layer Norm
   506	
   507	LayerNorm
   508	Nonlinear
   509	
   510	+
   511	
   512	𝑊𝑑𝑜𝑤𝑛
   513	
   514	LayerNorm
   515	
   516	L-Adapter
   517	
   518	× Multiply
   519	
   520	1
   521	
   522	+
   523	MLP
   524	
   525	Multi-Head Attention
   526	
   527	Layer Norm
   528	
   529	PETL
   530	
   531	Gradient Control
   532	
   533	S-Prefix
   534	
   535	×12
   536	
   537	Frozen
   538	
   539	1
   540	
   541	L-Adapter
   542	
   543	+
   544	
   545	Image Input
   546	
   547	𝑆𝑝
   548	
   549	Attention
   550	Image Encoder
   551	
   552	2
   553	
   554	SDM
   555	
   556	×
   557	
   558	×12
   559	𝑄
   560	
   561	𝑃𝑘
   562	
   563	PETL
   564	
   565	LoRA
   566	
   567	𝑉
   568	
   569	+
   570	𝑊𝑘
   571	
   572	LoRA
   573	
   574	Prefix
   575	
   576	× 𝑆𝑝
   577	
   578	𝑊𝑣
   579	
   580	𝑃𝑣
   581	
   582	𝑉
   583	
   584	3
   585	
   586	Softmax
   587	
   588	Hidden States
   589	
   590	MLP
   591	
   592	Layer Norm
   593	
   594	Multi-Head Attention
   595	
   596	Text Input
   597	
   598	𝑃𝑣
   599	
   600	𝐾
   601	
   602	+
   603	𝑊𝑞
   604	
   605	Layer Norm
   606	
   607	A lady with long black
   608	hair is wearing a white
   609	shirt with short blue
   610	shorts, multi-colored
   611	New Balance tennis
   612	shoes, and is carrying
   613	a red backpack.
   614	
   615	Prefix Part
   616	
   617	Attention
   618	Value
   619	
   620	2
   621	
   622	×
   623	
   624	Multi-Head Attention
   625	
   626	+
   627	L-Adapter
   628	
   629	LayerNorm
   630	
   631	Prefix
   632	𝑄
   633	
   634	Text Encoder
   635	
   636	Embedding
   637	
   638	𝑃𝑘
   639	
   640	𝐾𝑇
   641	
   642	S-Prefix
   643	
   644	Image / Text Encoder
   645	
   646	Fig. 2. Overview of the proposed UP-Person framework. Left is the overall backbone of UP-Person, which consists of image encoder and text encoder
   647	based on CLIP, two PETL modules for both encoders, and one parameter-free loss function constraint SDM as optimization objective. Only a few parameters
   648	in PETL modules are fine-tuned in training phase, while the other original full backbone of CLIP is frozen. Right is the implementation details of one
   649	transformer block for both image and text encoders. In addition to prefix tokens in the keys and values of MHA, S-Prefix proposes a Sp factor in attention
   650	calculator to enhance gradient propagation of prefix tokens. L-Adapter is proposed in two normalization layers to adjust the overall distribution and avoid
   651	submodule conflicts. LoRA is inserted to update the weights of keys and values. Overall, L-Adapter helps transfer global pedestrian features, while LoRA
   652	and S-Prefix, working together in MHA, focus on attention to promote local knowledge transferring for TPR. All blocks with dashed borderlines represent
   653	the fine-tuned modules. On the far right are the more specific implementation details of our L-Adapter and S-Prefix.
   654	
   655	Image Encoder. We adopt the visual backbone of CLIP
   656	(ViT-B/16) [60] as our image encoder. We first partition the
   657	image I ∈ RH×W ×C into a sequence of N = H ×W/P 2 nonoverlapping patches, where P is the patch size. The patches are
   658	then mapped to embeddings with a linear projection and added
   659	with positional embeddings to enhance spatial information.
   660	Subsequently, a [CLS] token is introduced at the beginning
   661	of the embeddings to denote the overall global information
   662	of the image. The sequence of P 2 + 1 tokens is then fed
   663	into a series of transformer blocks to capture the correlations
   664	within these patches. A transformer block typically consists
   665	of a MHA and a MLP, where layer normalization [61] is
   666	omitted for simplicity in the later formulation. The input image
   667	2
   668	features are represented as Xi,j,l ∈ R(P +1)×D , where l is the
   669	layer index, D is the hidden dimension of a patch, and P 2 + 1
   670	is the length of the sequence embeddings. The calculation of
   671	l-th layer is formulated as:
   672	X̂i,j,l = MHA (Xi,j,l−1 ) + Xi,j,l−1
   673	
   674	
   675	Xi,j,l = MLP X̂i,j,l + X̂i,j,l
   676	
   677	(7)
   678	(8)
   679	
   680	Text Encoder. The language backbone of CLIP is utilized
   681	as text encoder, which is also a 12-layer transformer. The
   682	computation within a single transformer block mirrors that
   683	of the image encoder. For the input text T , we tokenize the
   684	input description to embeddings f by a simple tokenizer with
   685	a 49152 vocab size [62]. For data augmentation, we randomly
   686	mask 15% of the tokens and replace them with the [MASK]
   687	
   688	token following BERT. f then adds [BOS] as the start of the
   689	sequence and [EOS] as the end flag. Thus, the overall sequence
   690	can be denoted as {fbos , f1 , . . . , feos } and then fed into the
   691	transformer as above image encoder by masked MHA, where
   692	the output of feos is the global representation in language
   693	branch.
   694	B. Unified PETL Architecture
   695	Motivation. As described in Section III, we can conclude
   696	that different PETL methods generally focus on different parts
   697	in the transformer block. Specifically, Adapter [20] is inserted
   698	through a residual connection to adapt the output information of MLP and MHA. LoRA [21] incorporates low-rank
   699	matrices to update weights, which typically represent inherent
   700	characteristics of a model for a specific downstream task.
   701	Prefix Tuning [22] operates at the forefront of the attention
   702	module, which guides the model to focus on more relevant
   703	parts of the person image or text description by learnable prefix
   704	tokens, thereby extracting more useful information from the
   705	input of each layer. Intuitively, the functions and processed
   706	information of the three most representative PETL modules
   707	should complement with each other if we carry on careful and
   708	proper module design. Therefore, we infer that a unified PETL
   709	framework can exhibit a more powerful expressive capability
   710	in text-based person retrieval.
   711	Unified PETL Framework. Inspired by the above observation and analysis, as shown in Figure 2, we propose UP-
   712	
   713	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   714	
   715	6
   716	
   717	Person framework based on CLIP backbone, which designs
   718	and optimizes multiple lightweight PETL methods: Prefix,
   719	LoRA and Adapter within image and text encoders. (1)
   720	Prefix is optimized to Salable Prefix (S-Prefix) to enhance
   721	the adaptation ability of the prefix embeddings for text-based
   722	person retrieval task. S-Prefix concatenates key and value in
   723	attention of MHA sublayer, which complements the taskspecific information and steers model to focus on crucial TPRspecific content. (2) Low-rank matrices are designed in weight
   724	modules of key and value related to MHA to learn more inherently local information about TPR. (3) Layernorm Adapter
   725	(L-Adapter) is proposed to adjust the overall distribution of
   726	feature representations in parallel with Layer Normalization
   727	(LN), which can adapt shift and bias in a nonlinear way.
   728	Theoretical Analysis of UP-Person. We then provide a
   729	more in-depth theoretical justification to explain the function
   730	of each module and why the combination of Prefix, LoRA,
   731	and L-Adapter works effectively together in TPR task.
   732	First, we analyze LoRA component for TPR:
   733	
   734	we further introduce two our novel submodules: S-Prefix
   735	and L-Adapter.
   736	
   737	
   738	
   739	
   740	T
   741	QK ′ V ′ = Q K T + ∆K T
   742	V T + ∆V T
   743	
   744	However, during the training phase, it was observed that
   745	the convergence rate of the vanilla prefix was notably slow
   746	in TPR task, resulting in a poor retrieval performance. Our
   747	analysis indicates that this phenomenon is mainly caused by
   748	relatively small gradient values of prefix tokens. Specifically,
   749	the gradient of Pv [63] can be denoted as:
   750	
   751	⊤
   752	 
   753	⊤ ∂L
   754	∂L
   755	∂h
   756	∂L
   757	= λ(x) σ xWq Pk ⊤
   758	, (14)
   759	=
   760	∂Pv
   761	∂Pv
   762	∂h
   763	∂h
   764	
   765	= QK T V + QK T ∆V + Q∆K T V T + Q∆K T ∆V T ,
   766	|
   767	{z
   768	}
   769	| {z }
   770	vanilla attention
   771	
   772	local information of TPR
   773	
   774	(9)
   775	k
   776	k
   777	where ∆K = X∆Wk = X(Wdown
   778	Wup
   779	), and ∆V =
   780	v
   781	v
   782	X∆Wv = X(Wdown Wup ). The second additional term that
   783	modifies the attention mechanism enables the model to capture
   784	more nuanced and local features, and relationships in TPR that
   785	the original weight matrices of CLIP cannot fully represent.
   786	Next, the embeddings of Prefix Pk and Pv can be concatenated on K and V to injecting task-specific information
   787	prompts for TPR, which helps the attention mechanism focus
   788	on task-relevant information. The attention with Prefix can be
   789	expressed as:
   790	
   791	T
   792	
   793	T
   794	
   795	Q[Pk ; K ′ ] [Pv ; V ′ ] = QK ′ V ′ + QPk T Pv
   796	= QK T V + QK T ∇V + Q∇K T V T + Q∇K T ∇V T +
   797	| {z }
   798	|
   799	{z
   800	}
   801	vanilla attention
   802	QPk T Pv .
   803	
   804	|
   805	
   806	{z
   807	
   808	local information of TPR
   809	
   810	}
   811	
   812	task-specific information of TPR
   813	
   814	(10)
   815	
   816	Finally, Adapter component can be utilized to adjust the
   817	overall distribution in layernorm, fine-tuning intermediate features from a global perspective:
   818	y → y + f (y · Wdown )Wup ,
   819	{z
   820	}
   821	|
   822	
   823	(11)
   824	
   825	global information of TPR
   826	T
   827	
   828	where y = LN (Q[Pk ; K ′ ] [Pv ; V ′ ]), and y represents the
   829	normalized intermediate features.
   830	The above PETL submodules do not conflict in terms of
   831	functionality or spatial distribution. Theoretically, the different
   832	components work synergistically to enhance performance.
   833	Therefore, by introducing these additional components to
   834	fine-tune the original features from coarse to fine, we can
   835	cohesively enhance global distribution, local dependencies,
   836	and TPR-specific prompt information.
   837	
   838	C. S-Prefix
   839	Compared with the prompt-based submodule in CSKT [27],
   840	Prefix is more flexible and efficient when attached to multiple
   841	layers, since it does not change sequence length. According
   842	to the study [58], [63], Prefix can be represented as a form
   843	similar to adapter, which can be viewed as working on the
   844	original head attention output h:
   845	
   846	h ← (1 − λ(x))h + λ(x)softmax xWq PkT Pv ,
   847	(12)
   848	where λ(x) is formulated as:
   849	
   850	exp xWq Pk⊤ i
   851	i
   852	
   853	 .
   854	P
   855	λ(x) = P
   856	exp xWq Pk⊤ i + exp xWq Wk⊤ x⊤ j
   857	P
   858	
   859	i
   860	
   861	(13)
   862	
   863	j
   864	
   865	where σ is the softmax function. Since the length of Pk is
   866	much less than input x, λ(x) is a small value tending to 0,
   867	which dramatically reduces the convergence speed of training,
   868	and further significantly impacts the retrieval performance of
   869	TPR task. A similar conclusion can be drawn on Pk .
   870	To solve this issue, we propose an improved module Salable
   871	Prefix (S-Prefix) to enhance gradient propagation of prefix
   872	tokens, which introduces a novel salable factor on prefixrelated attention to optimize original prefix. As shown in
   873	Figure 3, we first convert the N input tokens X ∈ RN ×D into
   874	queries Q, keys K and values V . Then, the L prefix tokens are
   875	inserted into K and V , and the transformed keys and values are
   876	′
   877	′
   878	denoted as K , V ∈ R(N +L)×D . After applying the attention
   879	′
   880	mechanism, QK T ∈ RN ×(N +L) . S-Prefix separates attention
   881	matrix into two parts: prefix matrix and original matrix. we
   882	design a salable factor Sp to boost the attention related to
   883	the prefix part and further speed up the convergence, which
   884	is a lightweight multiplier for the attention module of prefix.
   885	In order to demonstrate how and where Sp works effectively,
   886	S-Prefix is equivalent to inserting Sp to Equation (12), finally
   887	denoted as:
   888	
   889	h ← (1 − λ(x))h + Sp · λ(x) · softmax xWq PkT Pv . (15)
   890	Subsequently, the improved gradient of prefix is as follows:
   891	
   892	⊤
   893	 
   894	⊤ ∂L
   895	∂h
   896	∂L
   897	∂L
   898	=
   899	= Sp · λ(x) σ xWq Pk ⊤
   900	, (16)
   901	∂Pv
   902	∂Pv
   903	∂h
   904	∂h
   905	where Sp is a learnable factor designed to address the issue
   906	of excessively small gradients for Pv caused by λ(x).
   907	
   908	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   909	
   910	Fine-tuned × Multiply
   911	
   912	+ Concatenate
   913	
   914	Frozen
   915	
   916	…
   917	
   918	𝑵 × (𝑵 + 𝑳)
   919	
   920	+
   921	Scalable Factor 𝑺𝒑
   922	
   923	×
   924	𝑵×𝑵
   925	
   926	𝑵×𝑳
   927	…
   928	
   929	𝑵 × (𝑵 + 𝑳)
   930	
   931	7
   932	
   933	Inspired by the above, we believe that further exploring the
   934	powerful transferring ability of layernorm is promising. As
   935	depicted in Figure 4(d), Layernorm Adapter (L-Adapter) is
   936	finally designed in parallel with layernorm to adjust overall
   937	distribution in a nonlinear way, which is more flexible and
   938	capable of addressing more complex situations compared to
   939	linear methods such as LN-tuning, and can effectively avoid
   940	module conflicts compared with the vanilla parallel adapter. It
   941	can be formulated as:
   942	h ← LayerNorm (x) + s · Adapter (x) .
   943	
   944	Attention
   945	
   946	(17)
   947	
   948	E. Optimization Objective
   949	
   950	…
   951	
   952	𝑵
   953	
   954	𝑳
   955	
   956	A
   957	
   958	Prefix Tokens
   959	
   960	lady
   961	
   962	…
   963	
   964	backpack
   965	
   966	Person Image Patch or Word Tokens
   967	
   968	Fig. 3. Illustration of S-Prefix. We utilize Sp to denote the salable
   969	factor about attention of prefix to accelerate the convergence rate. S-Prefix
   970	submodules are inserted in all transformer layers of two branches.
   971	
   972	D. L-Adapter.
   973	The vanilla adapters [37], [40] typically act on modules
   974	containing MLP or MHA, either in a sequential or parallel
   975	manner as Figure 4(a)(c). However, in the unified framework,
   976	PETL submodules such as S-Prefix and LoRA are designed
   977	within MHA or MLP for functional diversity on different
   978	locations. The original parallel adapter and other PETL submodules simultaneously do fine-tuning on input as Figure
   979	4(c), and their structures overlap and interact, which is prone
   980	to disrupting the intended optimization direction and causes
   981	component conflicts. Unlike them, in this paper, we focus
   982	on Layer Normalization to get rid of causing conflicts from
   983	space and explore appropriate connection manner including
   984	sequential and parallel.
   985	Layer Normalization (LN) [61], also known as layernorm,
   986	is used to standardize the distributions of intermediate layers. This normalization process enhances the smoothness of
   987	gradients during training, accelerates the training process, and
   988	improves generalization accuracy. The procedure unfolds in
   989	two main steps: (1) Normalization of x by mean and variance.
   990	This helps in centering and scaling the values, bringing x to a
   991	standard form. (2) Scaling and shifting operations with gain g
   992	and bias b. Following the normalization step, the standardized
   993	values undergo a scaling and shifting process. The scaling
   994	operation, governed by the gain term g, allows for adjusting the
   995	spread of the values. The shifting operation, regulated by the
   996	bias term b, enables the network to introduce necessary variations. In essence, layernorm not only ensures that intermediate
   997	layers have consistent statistical properties but also introduces
   998	adaptability through the scaling and shifting operations. we
   999	can conclude that layernorm is important in fine-tuning to
  1000	downstream tasks, and unfreezing layernorm (LN-tuning) like
  1001	[64] has been explored. However, LN-tuning as Figure 4(e)
  1002	directly fine-tunes the shift and scale parapmeters, which
  1003	damages the inherent features of CLIP in a linear way.
  1004	
  1005	A parameter-free loss function is adopted in training phase
  1006	termed as Similarity Distribution Matching (SDM) [13], which
  1007	integrates the cosine similarity distributions of the N × N
  1008	embeddings for image-text pairs into the KL divergence to
  1009	build up the connection of two modalities.
  1010	For a mini-batch containing N image-text
  1011	a
  1012	 pairs, we form
  1013	N
  1014	set of image-text representation pairs as fiv , fjt , yi,j j=1 ,
  1015	where yi,j = 1 represents a matched pair from the same
  1016	person, and yi,j = 0 indicates an unmatched pair. The
  1017	probability of a matching pair pi,j is calculated with a softmax
  1018	function as follows:
  1019	 
  1020	exp sim fiv , fjt /τ
  1021	,
  1022	(18)
  1023	pi,j = PN
  1024	v
  1025	t
  1026	k=1 exp (sim (fi , fk ) /τ )
  1027	where sim(f v , f t ) denotes cosine similarity between text embedding f v and image embedding f t , and τ is a temperature
  1028	hyper-parameter that controls the sharpness
  1029	PN of the probability
  1030	distribution. We denote qi,j = yi,j / k=1 yi,k as the true
  1031	matching probability. Finally, the SDM loss from image to text
  1032	is computed based on above probabilities and KL divergence:
  1033	
  1034	
  1035	N N
  1036	pi,j
  1037	1 XX
  1038	pi,j log
  1039	, (19)
  1040	Li2t = KL (pi ∥qi ) =
  1041	N i=1 j=1
  1042	qi,j + ϵ
  1043	where ϵ is a small number to avoid potential issues with
  1044	numerical calculations. Then, the bidirectional SDM loss is
  1045	formulated as:
  1046	Lsdm = Li2t + Lt2i ,
  1047	
  1048	(20)
  1049	
  1050	where Li2t denotes image-to-text matching for the input text,
  1051	and Lt2i denotes text-to-image matching for the input image,
  1052	similar to Equation (19). Both weights of the bidirectional
  1053	loss functions are set to 1 equally to enhance cross-modal
  1054	alignment ability.
  1055	V. E XPERIMENTAL S ETTINGS
  1056	This section introduces the three official datasets, implementation details and evaluation metrics.
  1057	CUHK-PEDES as the most commonly used dataset, contains 40,206 images and 80,412 textual descriptions for 13,003
  1058	identities. The training set consists of 11,003 identities with

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | rg -n \""'^(Abstract|[0-9]+'"\\.?|[IVX]+\\.|[A-Z][A-Za-z ]{2,}:|.*Method.*|.*Approach.*|.*Proposed.*|.*Experiment.*|.*Ablation.*|.*Conclusion.*|.*Dataset.*)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3:4857
10:Abstract— In recent years, growing needs for advanced security
37:24
39:2024;
41:5
43:2024;
68:I. I NTRODUCTION
90:There are three primary causes of noisy labels in this context:
115:1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
117:See https://www.ieee.org/publications/rights/index.html for more information.
118:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
195:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
201:II. R ELATED W ORK
249:4859
306:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
317:III. M ETHODOLOGY
360:1 if yi = y j
366:0 if yi ̸ = y j .
394:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
517:4861
535:1) Robust Identification Loss: As described in section III-C,
542:A sorting weight is defined as follows:
569:2
586:2
593:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
630:3
631:1X
649:2) Robust Retrieval Loss: Combined with the subsets
659:Lhet definite as follows:
671:1− ŷicj
680:1− ŷicj
687:1− ŷik
691:2
711:Lhom is defined as shown below:
739:IV. E XPERIMENTS
740:A. Experiments Setting
741:1) Datasets: Our proposed methodology was evaluated
746:34,166 images, including 22,257 visible and 11,909 nearinfrared images. These images were captured using 4 visible
757:10 visible and 10 infrared images. Despite its relatively smaller
770:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
774:4863
782:50%, R ESPECTIVELY. T HE B EST R ESULTS A RE H IGHLIGHTED
789:24.07% for infrared and 23.63% for visible. It can be observed
804:2) Implementation Details: In implementing the proposed
825:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
834:3) Evaluation Metrics: In our experimental settings,
851:1) Compared VI-ReID Methods: To illustrate the advancement of our method, we compared it with several stateof-the-art models on the datasets SYSU-MM01 and RegDB
875:2) Results on Instance-Dependent Label Noise: It can be
883:13.36%, 8.61%, 4.13%, 11.29%, and 10.52% enhancements
901:3) Results on Real-World Noise: As shown in Tab. III, the
916:4) Results on Random Noise: Given that the label noise in
920:Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '35,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    35	I. I NTRODUCTION
    36	
    37	T
    38	
    39	EXT-based Person Retrieval (TPR) [1] aims to locate the
    40	person of interest from a large pool of candidates given
    41	a pedestrian description, which is a cross-task that integrates
    42	Manuscript initially received on May 2, 2024. Revised version received on
    43	October 16, 2024, January 31, 2025. This work was supported by the National
    44	Natural Science Foundation of China NSFC under U23B2030, and the Major
    45	Key Project of Peng Cheng Laboratory under PCL2023A08. (Corresponding
    46	author: Zimo Liu and Qingmin Liao.)
    47	Yating Liu is with Shenzhen International Graduate School, Tsinghua
    48	University, Shenzhen 518071, China and Peng Cheng Laboratory, Shenzhen
    49	518071, China (e-mail: liuyatin21@mails.tsinghua.edu.cn).
    50	Yaowei Li is with School of ECE, Peking University, Shenzhen 518071,
    51	China and Peng Cheng Laboratory, Shenzhen 518071, China (e-mail:
    52	ywl@stu.pku.edu.cn).
    53	Xiangyuan Lan and Zimo Liu are with Peng Cheng Laboratory, Shenzhen
    54	518071, China (e-mail: lanxy@pcl.ac.cn; liuzm@pcl.ac.cn).
    55	Wenming Yang and Qingmin Liao are with Shenzhen International
    56	Graduate School, Tsinghua University, Shenzhen 518071, China (e-mail:
    57	yang.wenming@sz.tsinghua.edu.cn; liaoqm@tsinghua.edu.cn).
    58	
    59	Knowledge of CLIP (KC)
    60	TPR-specific Knowledge of CLIP (PKC)
    61	TPR-specific Knowledge from Training datasets (PKT)
    62	
    63	(a)
    64	
    65	Fine-tuned
    66	Frozen
    67	
    68	PKC + PKT
    69	
    70	…
    71	
    72	Image
    73	Encoder
    74	
    75	Text
    76	Encoder
    77	
    78	PETL
    79	module
    80	
    81	Parameter-Efficient Transfer Learning
    82	
    83	>
    84	
    85	arXiv:2504.10084v1 [cs.CV] 14 Apr 2025
    86	
    87	Yating Liu, Yaowei Li, Xiangyuan Lan,
    88	Wenming Yang, Senior Member, IEEE, Zimo Liu, and Qingmin Liao, Senior Member, IEEE
    89	
    90	PKT
    91	
    92	Image
    93	Encoder
    94	
    95	Text
    96	Encoder
    97	
    98	Full-tuning
    99	
   100	Model
   101	
   102	Knowledge
   103	
   104	TPR-specific Knowledge
   105	
   106	(b)
   107	
   108	R@1
   109	
   110	Ours
   111	Full Tuning
   112	
   113	Ours
   114	74.17%
   115	
   116	+5.98%
   117	
   118	Full Tuning
   119	
   120	68.19%
   121	
   122	7.4M
   123	
   124	-95.1%
   125	
   126	Fine-tuned Params
   127	150M
   128	
   129	Fig. 1. The motivation for our proposed method. (a) shows how PETLbased methods can transfer TPR-specific knowledge from both CLIP and
   130	training data, whereas full-tuning relies solely on the training data as its
   131	knowledge source. Full-tuning (lower) only utilizes the knowledge of the
   132	pre-trained CLIP at initialization and almost loses the original knowledge,
   133	which thus only retains the knowledge of TPR from training datasets (PKT).
   134	PETL (upper) fine-tunes a small parameters and keeps CLIP backbone frozen,
   135	which means that it can integrate both TPR-specific within CLIP (PKC) due
   136	to the retained parameters of original CLIP and TPR-specific knowledge
   137	from training data (PKT). Therefore, PETL methods can incorporate more
   138	knowledge compared to full-tuning if rationally designed. (b) On CUHKPEDES, our approach reduces 95.1% training parameters and gains an
   139	improvement by 5.98% on R@1 compared to the full-tuning CLIP.
   140	
   141	person re-identification (Re-ID) [2] with cross-modal retrieval
   142	[3]. The core of TPR is to establish the matching relationship
   143	between person images and texts. Compared to conventional
   144	image-based person retrieval (Re-ID) [2], [4]–[6] and attributebased person retrieval [7], text-based person retrieval [8] provides a more intuitive and convenient way by forming queries
   145	with natural language descriptions, thus attracts increasing
   146	attention from both academia and industry, benefiting a variety
   147	of applications, such as security surveillance and intelligent
   148	transportation.
   149	As large foundation models expand, the embedded knowledge becomes increasingly abundant. Consequently, it is crucial to explore how to effectively transfer these pre-trained
   150	models to downstream tasks in order to maximize knowledge
   151	utilization. The great success in Vision-Language Pre-training
   152	(VLP) has recently shown a strong cross-modal transfer capability in various vision-language understanding tasks [9] ,
   153	where the most representative work is Contrastive Language-
   154	
   155	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   156	
   157	Image Pre-training (CLIP) [10]. With 400M web-crawled
   158	trainable image-text pairs, CLIP contains abundant generic
   159	knowledge learned from the large-scale dataset. Besides, the
   160	pre-trained encoders of CLIP have greater cross-modal matching potential due to its two-branch contrastive architecture
   161	than uni-modal encoders, i.e., ViT [11] for vision, BERT [12]
   162	for language. Therefore, several works [13], [14] adopt CLIP
   163	as the backbone, and propose multi-level matching modules
   164	to achieve CLIP-based cross-modal interactions from coarse
   165	to fine with fully fine-tuning, which leads to a significant
   166	improvement compared to many previous uni-modal retrieval
   167	frameworks [15]–[18]. However, this full-tuning paradigm
   168	faces two fatal issues: (1) it has a high risk of overfitting
   169	on limited task-specific training datasets as the scale of large
   170	pre-trained models continues to increase, and (2) training
   171	and storing a completely new large model for each dataset
   172	independently is expensive in practice.
   173	An elegant solution to the above-mentioned problems
   174	is Parameter-Efficient Transfer Learning (PETL) [19], i.e.,
   175	Adapter [20], LoRA [21], Prefix [22], Prompt [23] and other
   176	related variants [24]–[26], which can achieve comparable or
   177	superior performance only with fine-tuning a few parameters of large models. As shown in Figure 1(a), full-tuning
   178	(lower) only utilizes the knowledge of the pre-trained CLIP
   179	at initialization stage and almost loses the original knowledge
   180	embedded in network. On the contrary, the CLIP backbone
   181	is frozen and preserved in PETL paradigm (upper). Guided
   182	by PETL-related components, TPR-specific knowledge within
   183	CLIP is effectively transferred as the parameters of the original
   184	CLIP remain unchanged. The rich knowledge of CLIP about
   185	visual objects and textual descriptions can serve as knowledge
   186	complementarity for fine-grained TPR task. Consequently,
   187	PETL facilitates learning both task-specific knowledge from
   188	general CLIP and knowledge from the training dataset, making
   189	it superior to full-tuning in terms of knowledge retention,
   190	particularly in our scenarios with limited training data.
   191	However, PETL paradigm is not well explored in TPR.
   192	CSKT [27] makes the first attempt to explore CLIP with
   193	PETL-related methods on TPR and proposes a novel bidirectional multi-modal prompt-tuning, which attains superior
   194	performance only with fine-tuning 7.4% parameters of CLIP.
   195	This existing PETL-based method only focuses on the global
   196	feature representation, and has not sufficiently transferred both
   197	global and local information from multiple views of CLIP. The
   198	depth of exploration by PETL on TPR still remains limited.
   199	This prompts us to consider whether it is feasible to design a
   200	unified PETL framework for TPR from a more comprehensive
   201	perspective.
   202	In this paper, our target is to investigate how to design a simple, effective and parameter-efficient unified transfer learning architecture based on multiple lightweight PETL methods.
   203	A serious concern on unifying various PETL methods is
   204	component conflict, where the performance drops significantly
   205	in practice when a single PETL component is incorporated into
   206	a unified framework. This occurs as the original structures of
   207	multiple PETL components overlap and interact, potentially
   208	disrupting the intended optimization direction. Consequently,
   209	it is crucial to design a unified framework where different
   210	
   211	2
   212	
   213	components do not interfere with each other and can work
   214	cooperatively. Another issue we observe is that when the
   215	vanilla prefix component is adapted to TPR, it shows a poor
   216	performance compared to the approximate PETL submodule
   217	such as prompt-tuning in CSKT [27]. Thus, we consider
   218	whether we could optimize the vanilla PETL components for
   219	better synergy in the unified framework.
   220	To address the aforementioned issues, we propose a novel
   221	Unified Parameter-Efficient Transfer Learning (PETL) method
   222	for Text-based Person Retrieval (UP-Person). UP-Person implements a comprehensive PETL-based method to enhance
   223	knowledge transferring without requiring any additional complex cross-modal interaction modules. Specifically, as shown
   224	in Figure 2, in a transformer block of CLIP, we design and
   225	unify three submodules based on Prefix, LoRA and Adapter
   226	to adapt to TPR task. LoRA is incorporated to modify the
   227	weights of Multi-Head Attention (MHA), allowing the model
   228	to capture more nuanced and local features and relationships
   229	in TPR that the original CLIP cannot fully uncover. Prefix
   230	is prepended to the keys and values of MHA to enrich taskspecific information for TPR. Adapter is designed in layer normalization (layernorm) to adjust the overall distribution from
   231	a global perspective. Meanwhile, we propose two improved
   232	submodules: Salable Prefix (S-Prefix) and Layernorm Adapter
   233	(L-Adapter). S-Prefix is introduced to enhance gradient backward propagation of prefix embeddings, which improves both
   234	the flexibility and performance of the vanilla prefix. L-Adapter
   235	is designed in parallel with layernorm, alongside the residual
   236	connection, to get rid of component conflicts. Our method
   237	demonstrates greater advantages in data-scarce scenarios especially in RSTPReid dataset. It achieves better performance
   238	and parameter-efficiency with negligible sacrifice in inference
   239	efficiency, and reduces computation and storage costs. Our
   240	contributions can be summarized as follows:
   241	• We propose a unified parameter-efficient transfer learning
   242	method for text-based person retrieval. To the best of our
   243	knowledge, our study is the first attempt to investigate
   244	the unified PETL framework for TPR, which effectively
   245	transfers both global and local knowledge, along with
   246	task-specific knowledge, to TPR task with very fewer
   247	computation and storage costs.
   248	• To optimize the PETL components for better synergy,
   249	we further develop two improved PETL components,
   250	respectively: (1) a scalable prefix in attention named SPrefix, and (2) a novel adapter L-Adapter designed in
   251	parallel with layernorm of blocks.
   252	• Extensive experiments show that UP-Person achieves
   253	superior performance compared with the prior state-ofthe art on three public datasets while merely fine-tuning
   254	4.7% parameters.
   255	
   256	II. R ELATED W ORK
   257	In this section, we will briefly review the most relevant study including vision-language pre-training, parameterefficient transfer learning, and text-based person retrieval.
   258	
   259	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   260	

codex
前三篇的方法段已经能看出各自的“问题重写方式”：UP-Person 把全量微调改写成“知识保留不足”，HDNet 把跨模态差异拆成“相机、模态、身份冗余的层级污染”，MPT 把文本提示从静态辅助改写成“能和图像编码器互相强化的中间模态”。我继续补齐剩下两篇的方法段，并把消融和可视化也读出来，用来回答“证据闭环”。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '414,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   414	Feature Cross-Modality Recombination Strategy
   415	
   416	To further reduce the impact of modality differences and guide the network to focus on shared
   417	features, we decompose 𝐹e𝑐𝑖 and 𝑅 + by modality into 𝐹˜𝑐𝑖𝑣 , 𝐹˜𝑐𝑖𝑡 , 𝑅𝑣+ , and 𝑅𝑡+ , and then cross recombine
   418	them as follows:
   419	𝐹˜𝑣_𝑣 = 𝐹˜𝑐𝑖𝑣 + 𝑅𝑣+,
   420	𝐹˜𝑡 _𝑡 = 𝐹˜𝑐𝑖𝑡 + 𝑅𝑡+,
   421	
   422	(16)
   423	
   424	𝐹˜𝑡 _𝑣 = 𝐹˜𝑐𝑖𝑡 + 𝑅𝑣+,
   425	𝐹˜𝑣_𝑡 = 𝐹˜𝑐𝑖𝑣 + 𝑅𝑡+ .
   426	
   427	(18)
   428	
   429	(17)
   430	(19)
   431	
   432	Here, 𝐹˜𝑣_𝑣 and 𝐹˜𝑡 _𝑡 represent the same-modality features, and Equation (10) 𝐹𝑖𝑑 is composed of
   433	these two features, while 𝐹˜𝑣_𝑡 and 𝐹˜𝑡 _𝑣 represent the cross-modality identity features. Subscripts
   434	𝑣 and 𝑡 denote visible and infrared modalities, respectively. In the process of cross-modality
   435	recombination, we adopt a random combination approach. For the same identity, by pairing the
   436	positive residual features of different modalities, the features are recombined into their respective
   437	modalities. This method avoids the strict requirement of recombining the same modality features,
   438	allowing the model to effectively transfer the identity information across modalities, rather than
   439	being limited to local or narrow recombination.
   440	In the cross-modality recombination process, a random combination strategy is employed.
   441	Residual identity features from different modalities are paired and recombined. For the same identity,
   442	the model is not restricted to recombining only features from the same modality (e.g., visible-visible
   443	or thermal-thermal), but rather allows for cross-modality pairings (e.g., visible-thermal).
   444	To further optimize decoupling loss, we apply metric learning to the four recombined features
   445	𝐹˜𝑣_𝑣 , 𝐹˜𝑡 _𝑡 , 𝐹˜𝑡 _𝑣 , and 𝐹˜𝑣_𝑡 . Equation (20) defines the feature center for each ID for the four types of
   446	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   447	
   448	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '68,200p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    68	I. I NTRODUCTION
    69	
    70	P
    71	
    72	EDESTRIAN re-identification utilizes extracted image
    73	features to facilitate identity localization across different
    74	cameras, playing a crucial role in public safety and traffic
    75	detection [1], [2], [3], [4], [5], [6]. The scope of person
    76	re-identification has recently expanded to include visibleinfrared cross-modal scenarios [7], [8], [9], [10], [11],
    77	[12]. Incorporating infrared images enables effective reidentification in poorly lit environments, enhancing its
    78	practicality in real-world scenarios. However, visible-infrared
    79	person identification (VI-ReID) encounters notable challenges
    80	due to the modality gap [13] and the quality of infrared images
    81	[14]. To overcome these challenges, previous methods have
    82	focused on uncovering shared features across modalities [15],
    83	[16], [17], [18], employing data augmentations [19], [20],
    84	[21], and leveraging auxiliary information [22], [23], [24].
    85	Despite these advancements, most VI-ReID models, have not
    86	adequately addressed the adverse effects of noisy labels on
    87	their performance.
    88	In the real world, label noise is a common issue [25], [26],
    89	[27], [28], and this is particularly true in the VI-ReID task.
    90	There are three primary causes of noisy labels in this context:
    91	• Instantiated Identity within Single Modality: The
    92	classification of identities often involves transferring
    93	categories from semantically distant entities, leading to
    94	confusion between identities. The main reason for this
    95	issue is the small gap between the characteristics of the
    96	same entity in different identities.
    97	• Instantiated Identity between Two Modalities: Infrared
    98	imaging often loses key features such as color and texture,
    99	making it challenging to distinguish between different
   100	identities. Consequently, it is hard to match infrared
   101	images with the same identity and to correlate infrared
   102	images with their visible counterparts.
   103	• Annotator Errors: While primarily stemming from
   104	the first two reasons, errors due to the sloppiness or
   105	carelessness of annotators can also contribute to noisy
   106	labels. These errors, which can sometimes be random,
   107	are often exacerbated by difficult annotation tasks.
   108	Currently, only one method [19] has discussed the issue
   109	of noisy labels in VI-ReID. Although the existing study
   110	DART [19] has considered the impact of noisy labels on
   111	VI-ReID, it primarily focuses on random noise, such as
   112	arbitrarily switching image labels to incorrect ones. This noise
   113	generation approach does not fully represent the complex
   114	
   115	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
   116	and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   117	See https://www.ieee.org/publications/rights/index.html for more information.
   118	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   119	
   120	4858
   121	
   122	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   123	
   124	Fig. 1. The processing of visible-infrared re-identification dataset annotation. From left to right, these are random assignment, instance-dependent algorithmic
   125	generation, and real-world annotation. Circles represent visible images, and triangles represent infrared images. The outline of each instance is its potential
   126	real label (identity), and the center is filled with annotation labels, with different colors indicating different identities.
   127	
   128	forms of noise encountered in real-world scenarios, as outlined
   129	in the previously discussed points. To better simulate realworld conditions, we propose using instance-dependent label
   130	noise (IDN), which varies with each instance and reflects
   131	specific characteristics, offering a more realistic representation
   132	of label noise than the random noise used in the existing
   133	method [19]. As depicted in the middle part of Fig. 1, our
   134	approach follows existing studies [29], [30], employing the
   135	IDN generation algorithm to create synthetic datasets with
   136	noisy labels. However, it is important to acknowledge that even
   137	this advanced method has limitations. That is, it cannot fully
   138	capture the unpredictable and complex nature of real-world
   139	label noise in practical applications of VI-ReID.
   140	Considering the aforementioned analysis, we identify two
   141	primary reasons for the discrepancy between IDN and realworld noise: Firstly, as an algorithmically synthesized noise,
   142	IDN is unable to accurately mimic annotator errors, as shown
   143	in the real-world scenarios depicted in Fig. 1. Specifically,
   144	it does not account for instances of annotator carelessness or
   145	inattention. Secondly, the robustness of VI-ReID algorithms
   146	still requires real-world labelling for corroboration and cannot
   147	rely on simulated security challenges alone. Indeed, relying
   148	on synthetic algorithms for noise simulation might lead
   149	to a distorted evaluation of the methods’ effectiveness.
   150	To establish a more robust foundation for validating VIReID methods against real-world label noise, we have
   151	re-labeled the SYSU-MM01 dataset, creating an updated
   152	version called SYSU-Real that includes genuine real-world
   153	annotations.
   154	To tackle the challenges of increasingly complex label
   155	noise above, we introduce a robust hybrid loss (RHL)
   156	framework that categorizes data into three segments: clean,
   157	indistinguishably noisy, and obviously noisy. For the clean
   158	data, we apply the standard cross-entropy (CE) loss, given
   159	that its labels are reliable. The second segment contains
   160	samples with ambiguous label quality, posing a significant
   161	
   162	challenge to label distinction. We note a bias in the CE loss
   163	function, particularly its tendency to assign disproportionately
   164	negative values to samples with noisy labels, leading to skewed
   165	optimization. To counteract this, we utilize the Generalized
   166	Cross-Entropy (GCE) loss function [31], which aims to
   167	reduce this bias and ensure fair treatment of these samples.
   168	In the third segment, where noise is evident, we discard the
   169	noisy labels altogether. Instead, we leverage the preheating
   170	model’s generalization capability, using the confidence level
   171	as the basis for loss calculation. Additionally, recognizing the
   172	inadequacy of general retrieval loss, we adopt the pair division
   173	method and triplet loss from DART to enhance the retrieval
   174	process’s robustness.
   175	In summary, our contributions are shown as follows:
   176	• We present, to our knowledge, the first consideration
   177	of real-world label noise in VI-ReID. To overcome
   178	the challenges of obtaining real-world labeled datasets,
   179	we have implemented two strategies. Firstly, we synthesized datasets with an instance-dependent label
   180	noise generation algorithm. Secondly, we manually
   181	re-labeled the SYSU-MM01 dataset, creating the
   182	human-annotated SYSU-Real dataset. This approach
   183	significantly enhances the realism of label noise, yet it
   184	also adds to the complexity and challenge of VI-ReID
   185	studies.
   186	• We introduce a Robust Hybrid Loss (RHL) function
   187	that effectively manages instance-dependent label noise
   188	and real-world noise. This function utilizes a threecomponent mixture, allowing distinct loss calculation
   189	strategies for different noise subsets, thereby enhancing
   190	the method’s robustness and adaptability.
   191	• We conducted comparative experiments with six stateof-the-art methods on the SYSU-MM01 and RegDB
   192	datasets, incorporating both random and instancedependent label noise. Additionally, we evaluated
   193	real-world noise handling on the relabeled SYSU-
   194	
   195	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   196	
   197	ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE
   198	
   199	Real dataset. Our results demonstrate the superior
   200	effectiveness and robustness of our proposed method.

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '436,616p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '33,160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '161,292p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   161	2. METHODOLOGY
   162	2.1. Overview of LVLM
   163	Overall framework. A typical LVLM consists of three key components: a visual encoder, a vision-language connector, and an LLM.
   164	The visual encoder extracts rich visual representations from images,
   165	which are then processed by the vision-language connector that converts visual features into the word embedding space. The LLM,
   166	trained for next-word prediction, generates text based on the encoded
   167	visual content. In this work, we leverage Qwen2-VL [14], one of
   168	
   169	the most advanced LVLMs, known for its superior capabilities in
   170	instruction-following, semantic understanding, and text generation
   171	across diverse tasks. Qwen2-VL combines a Vision Transformer
   172	(ViT) [17] as the visual encoder and the Qwen2 [11] as the LLM.
   173	The vision-language connector between the two components is one
   174	MLP layer that also compresses the extracted visual tokens.
   175	Visual token extraction. Before inputting a pedestrian image into
   176	the LLM, the image is first encoded and compressed by the visual
   177	encoder. Specifically, each input RGB image x ∈ RH×W ×3 , where
   178	H and W are its height and width, is first divided into patches of size
   179	P ×P . These patches are then embedded and flattened into a feature
   180	vector xp ∈ RN ×d , where N = H × W/P 2 represents the number of patches, and d is the embedding dimension. The resulting
   181	patch embeddings are processed through multiple layers of Transformer self-attention blocks [18], producing visual representations
   182	f ∈ RN ×d . To enhance the model’s ability to capture spatial dependencies, Multimodal Rotary Position Embedding (M-RoPE) [14] is
   183	used in the process. Afterward, a simple MLP layer compresses adjacent 2 × 2 tokens into a single token, producing the final visual
   184	tokens V , which is formulated as:
   185	V = MLP(f ) = [v1 ; v2 ; · · · ; vn ] ∈ Rn×d ,
   186	
   187	(1)
   188	
   189	N
   190	. Notably, instead of using the traditional [class]
   191	4
   192	
   193	where n =
   194	token [17], the image is transformed into a set of visual tokens.
   195	These visual tokens will then be passed to the LLM for further processing and interaction.
   196	2.2. Pedestrian Semantic Token Generation
   197	We aim to integrate the advanced visual semantic understanding and
   198	generation capabilities of LVLM into the feature extraction pipeline,
   199	by guiding the ReID model to generate one semantic token that encapsulates instructive information of the pedestrian. To achieve this,
   200	we use instructions to direct the LVLM to summarize the pedestrian’s visual appearance:
   201	<|vision start|> V <|vision end|> Summarize
   202	the person image into one word, focusing on age, gender,
   203	clothing, and biometric features.
   204	
   205	where V represents the extracted visual tokens, while the special
   206	tokens <|vision start|> and <|vision end|> are used to
   207	mark the beginning and end of the visual token sequence. With this
   208	instruction, the LVLM is guided to focus on the appearance-related
   209	semantics in the image, and then generate a semantic token that summarizes the relevant identity features. We denote this generated token as <REID>, which serves as a compact representation of the
   210	pedestrian’s visual appearance.
   211	Camera semantic supplementation. The semantic token generation process overlooks the influence of camera variations. To improve pedestrian semantic consistency across cameras, we explicitly model and account for these camera-induced feature variations.
   212	Specifically, we assign a unique learnable embedding vector to each
   213	camera, which allows the model to learn the inherent feature shifts
   214	caused by cameras. We denote the set of learnable camera embedi
   215	dings as Vcam = {vcam
   216	|i = 1, 2, · · · , N c }, where N c is the total
   217	number of cameras. One direct implementation is to supplement the
   218	generated pedestrian semantic token with the camera semantics as
   219	follows:
   220	yc
   221	v̄reid = vreid + vcam
   222	,
   223	(2)
   224	where vreid is the encoding of the <REID> token, y c is the camera
   225	ID corresponding to the image x. However, this late supplementation
   226	strategy may affect the visual model weakly. We thus try to transfer
   227	the usage of camera embeddings to the input of the visual model,
   228	where the camera embeddings are added to the patch embeddings
   229	xp . We evaluate and discuss the two variants in Sec. 3.3.
   230	2.3. Semantic-Guided Interaction
   231	We design the Semantic-Guided Interaction (SGI) module to facilitate bidirectional interaction between the generated semantic token
   232	and the visual tokens. Specifically, the generated semantic token is
   233	first concatenated with the visual tokens. Formally,
   234	z = [vreid ; v1 ; v2 ; · · · ; vn ] ∈ R(n+1)×d .
   235	
   236	(3)
   237	
   238	This concatenated token sequence is then passed through 4
   239	layers of Transformer blocks, each consisting of a multi-head selfattention layer and a feed-forward network. The module refines
   240	the visual features to capture identity-relevant information under
   241	the guidance of the semantic token. Meanwhile, the semantic token, serving as the pivot for information aggregation, distills more
   242	discriminative features from the visual representations, enhancing
   243	the overall understanding of the pedestrian’s identity. Through
   244	the semantic-guided interaction module, the model produces the
   245	reinforced representation as:
   246	ẑ = [v̂reid ; v̂1 ; v̂2 ; · · · ; v̂n ] = SGI(z).
   247	
   248	(4)
   249	
   250	Then, the reinforced semantic token representation v̂reid is used
   251	to compute the Re-ID losses, i.e., identity classification loss [2] and
   252	triplet loss [4].
   253	2.4. Optimization and Inference
   254	During training, we optimize the parameters of both the visual model
   255	and the SGI module. The parameters of LLM are frozen, but gradients backpropagate through it to update other learnable components.
   256	By leveraging the generated <REID> token in conjunction with the
   257	SGI module, we achieve a joint end-to-end training that harnesses
   258	the strengths of LVLM in instruction-following and visual semantic understanding. This process allows for the integration of rich
   259	
   260	Table 1. Comparison with the state-of-the-art methods. The results of our method and the best results of comparison methods are
   261	shown in bold.
   262	DukeMTMC-reID
   263	
   264	Market-1501
   265	
   266	mAP
   267	
   268	Rank-1
   269	
   270	mAP
   271	
   272	Rank-1
   273	
   274	mAP
   275	
   276	Rank-1
   277	
   278	mAP
   279	
   280	Rank-1
   281	
   282	CNN
   283	
   284	MGN [19]
   285	DG-Net [20]
   286	SAN [5]
   287	Pyramid [21]
   288	Relation-Net [22]
   289	RGA-SC [23]
   290	CDNet [6]
   291	CAL [7]
   292	

 succeeded in 0ms:
    33	1. INTRODUCTION
    34	Person re-identification (ReID) is a crucial task in computer vision,
    35	aimed at accurately matching pedestrians across different camera
    36	views [1]. With the continuous advancements in deep learning techniques, person ReID methods have evolved significantly [2, 3]. In
    37	the past decade, many research has significantly improved ReID
    38	accuracy by optimizing the distances between features [4] and designing refined modules [5, 6, 7], following the paradigm shown in
    39	Fig. 1 (a). Due to the difficulty of learning rich pedestrian semantics
    40	from a single modality, cross-modal learning has received close attention recently. For example, in the context of the development of
    41	pre-trained Vision-Language Models (VLMs), CLIP-ReID [8] based
    42	on the representative VLM model CLIP [9] to leverage text semantics. As shown in Fig. 1 (b), it enhances visual features through
    43	cross-modal contrastive learning with image-text pairs. Meanwhile,
    44	Large Language Models (LLMs) [10, 11] have attracted widespread
    45	This work was supported by the National Natural Science Foundation of
    46	China under Grant No. 62576110.
    47	† Corresponding author.
    48	
    49	ReID Loss
    50	
    51	ReID Loss
    52	
    53	Visual
    54	Encoder
    55	
    56	Visual
    57	Encoder
    58	
    59	Contrastive Loss
    60	
    61	Text
    62	Encoder
    63	
    64	A photo of [X]1 [X]2
    65	… [X]M person.
    66	
    67	(a) Conventional ReID
    68	
    69	(b) CLIP-ReID
    70	
    71	Generated
    72	Semantic Token
    73	Summarize the
    74	person image into
    75	one word, …
    76	
    77	ReID
    78	Loss
    79	
    80	Instruction
    81	
    82	LVLM
    83	Visual Tokens
    84	
    85	(c) LVLM-ReID
    86	
    87	Fig. 1. Comparison of different person ReID frameworks.
    88	attention due to their powerful capabilities in text generation and
    89	comprehension. Large Vision-Language Models (LVLMs) [12, 13,
    90	14] enhance LLMs by incorporating visual perception and understanding, demonstrating considerable potential in multi-modal learning tasks. However, integrating LVLMs with person ReID remains
    91	an underexplored challenge.
    92	LVLMs typically operate on a generative paradigm, training and
    93	functioning by predicting the next word in a sequence. Thanks to
    94	pre-training and instruction tuning, LVLMs can follow instructions
    95	and converse with humans. As a result, a direct approach might be
    96	to have the model to identify the input person images. However,
    97	ReID gallery databases are usually very large (comprising tens of
    98	thousands of pedestrian images) [15, 16]. The time and cost of comparing identities one by one with LVLMs are substantial. Processing
    99	multiple images simultaneously would also lead to an unacceptable
   100	increase in visual tokens. Therefore, we are motivated to leverage
   101	the reasoning and understanding capabilities of LVLMs, while adhering to the mainstream ReID paradigm of feature extraction combined with feature similarity-based retrieval [1]. A potential solution involves using LVLMs to describe pedestrian images and finetuning the visual encoder via tasks such as image-text matching or
   102	image caption prediction. However, it presents several limitations:
   103	(1) High-quality and diverse text annotations are expensive to obtain.
   104	(2) The goals of image-text matching or image caption prediction
   105	tasks may not align well with those of image-based ReID. (3) During
   106	the inference phase, the potential of LVLMs is often underutilized,
   107	as they are not effectively integrated with the visual features.
   108	
   109	<REID>
   110	
   111	FC
   112	
   113	Camera Semantic Supplementation
   114	4x
   115	
   116	LLM
   117	
   118	Visual Encoder
   119	V
   120	
   121	Pedestrian Semantic Token Generation
   122	
   123	Feed-Forward Network
   124	
   125	K
   126	
   127	Multi-Head Self-Attention
   128	
   129	Summarize the person image
   130	into one word, focusing on
   131	age, gender, clothing, and
   132	biometric features.
   133	
   134	Q
   135	
   136	Semantic-Guided Interaction
   137	
   138	Fig. 2. Framework of our LVLM-ReID. It leverages instructions to guide the frozen LLM toward focusing on particular visual semantics
   139	within pedestrian images, resulting in the generation of one semantic token that encapsulates the pedestrian’s appearance information. Subsequently, an efficient interaction module is designed to facilitate refinement between the generated token and the visual tokens. Finally, the
   140	reinforced semantic token is optimized and employed for person retrieval.
   141	To address these issues, we propose a new ReID framework
   142	called LVLM-ReID to leverage the superior semantic understanding
   143	and generation ability of LVLMs. Specifically, as shown in Fig. 1 (c),
   144	we use instruction to guide the LVLM to focus on specific visual
   145	semantics in pedestrian images, generating a semantic token representing the pedestrian’s appearance information. We then design
   146	an effective interaction module between the generated token and visual tokens, refining the visual representations of pedestrians while
   147	reinforcing the semantic token as a discriminative identity representation. Ultimately, the reinforced semantic token is optimized and
   148	used during inference to achieve person retrieval. Our framework
   149	integrates the generative process of LVLMs into the ReID model,
   150	eliminating the need for additional image caption annotations and
   151	enabling end-to-end effective learning. More importantly, during
   152	the inference phase, we continue to leverage the generative power
   153	of LVLMs to enhance visual features adaptively. Our experiments
   154	show that one LVLM-generated semantic token can effectively facilitate the learning of pedestrian representations. Our contributions
   155	are summarized as follows: (1) We propose a novel framework
   156	that incorporates LVLMs into the person ReID task, offering a new
   157	perspective on using generative language models to assist discriminative visual models. (2) We propose to utilize the generative capability of LVLMs to produce a semantic token for pedestrians and design a semantic-guided interaction module leveraging the generated
   158	semantic token to enhance identity representations. (3) Experimental results show that, without requiring additional annotations, our
   159	method effectively improves the discriminability of identity features
   160	and achieves competitive results across multiple datasets.

 succeeded in 0ms:
   436	3. EXPERIMENTS
   437	3.1. Experimental Settings
   438	Datasets. We evaluate our methods on four person ReID datasets:
   439	DukeMTMC-reID [16], Market-1501 [15], CUHK03 [28], and
   440	Occluded-Duke [29].
   441	Evaluation metrics. We follow the common practices to adopt Cumulative Matching Characteristics (CMC) at Rank-1 and mean Average Precision (mAP) for performance evaluation.
   442	Implementation details. We employ Qwen2-VL-2B [14] considering its efficiency with limited resources, while larger model sizes
   443	such as 7B and 72B have better LLM capabilities. The model adopts
   444	BFloat16 mixed precision. H, W , and P are set to 280, 140, and
   445	14, respectively, resulting in n = 50. In other words, 50 visual
   446	tokens are included in the input of LLM and our SGI module. Following [2], random horizontal flipping, padding, random cropping,
   447	and random erasing [30] are used for data augmentation. 16 identities and 4 images per person are randomly sampled to constitute a
   448	training batch. Adam optimizer with weight decay of 3 × 10−4 is
   449	adopted, with the warmup strategy that linearly increases the learning rate from 3 × 10−5 to 3 × 10−4 in the first 10 epochs. We train
   450	the model for 60 epochs, with a learning rate decay factor of 0.1 at
   451	the 30th epoch. α1 and α2 are set to 0.25 and 1 following [8]. The
   452	margin m of triplet loss is set to 0.3. Our method is implemented
   453	using PyTorch and on one NVIDIA A800 GPU.
   454	3.2. Comparison with State-of-the-Art Methods
   455	As shown in Tab. 1, methods based on CNNs achieve solid performance by designing elaborate modules for person ReID, while
   456	
   457	Table 2.
   458	Ablation studies of our key two components on DukeMTMC-reID
   459	and Market-1501.
   460	
   461	Table 3. Ablation of the camera semantic supplementation
   462	(CSS) strategy. CSS-vreid and
   463	CSS-xp denote adding the camera embedding to vreid and xp ,
   464	respectively.
   465	
   466	DukeMTMC-reID
   467	
   468	Market-1501
   469	
   470	mAP
   471	
   472	Rank-1
   473	
   474	mAP
   475	
   476	Rank-1
   477	
   478	Baseline
   479	Ours w/o PSTG
   480	Ours w/o SGI
   481	
   482	79.0
   483	80.9
   484	79.0
   485	
   486	90.2
   487	91.0
   488	90.0
   489	
   490	87.3
   491	88.3
   492	87.3
   493	
   494	94.7
   495	95.0
   496	94.5
   497	
   498	Methods
   499	
   500	Ours
   501	
   502	82.8
   503	
   504	92.2
   505	
   506	89.2
   507	
   508	95.6
   509	
   510	w/o CSS
   511	CSS-vreid
   512	CSS-xp
   513	
   514	Methods
   515	
   516	DukeMTMC-reID
   517	
   518	Market-1501
   519	
   520	mAP
   521	
   522	Rank-1
   523	
   524	mAP
   525	
   526	Rank-1
   527	
   528	81.6
   529	82.3
   530	82.8
   531	
   532	91.4
   533	92.1
   534	92.2
   535	
   536	89.1
   537	88.4
   538	89.2
   539	
   540	95.2
   541	95.3
   542	95.6
   543	
   544	TransReID [24] explores the potential of Transformer [18, 17] in
   545	ReID, establishing itself as a strong baseline with superior capability.
   546	Rather than designing elaborate modules for interactions between
   547	image pairs [26], or leveraging part-level features [27, 25] based
   548	on ViT, we introduce LVLM’s advanced understanding and generative processes into the ReID framework. On the DukeMTMC-reID
   549	dataset, which is known for variations in appearance, LVLM-ReID
   550	achieves an mAP of 82.8% and a Rank-1 accuracy of 92.2%, surpassing previous advanced methods. On the CUHK03 dataset,
   551	LVLM-ReID also significantly outperforms the advanced method
   552	CLIP-ReID [8]. LVLM-ReID achieves competitive results on the
   553	Market-1501 dataset and also performs well on the challenging
   554	Occluded-Duke dataset, showing its robustness and generalization
   555	ability in occlusion scenarios. Note that CLIP-ReID [8] leverages
   556	a VLM pre-trained on large-scale image-text pairs, and it discards
   557	the text encoder during inference. Differently, LVLM-ReID integrates LVLM into ReID training and inference stages in a novel
   558	paradigm. The strong performance of LVLM-ReID across datasets
   559	demonstrates its capability as a powerful LVLM-based baseline.
   560	3.3. Ablation Studies
   561	Effectiveness of the generated pedestrian semantic token. (1)
   562	Our baseline only uses the visual model of the LVLM, and the visual tokens are averaged to compute loss and feature similarity during training and inference. The baseline only uses the visual model,
   563	overlooking the role of LVLM in visual semantic understanding and
   564	achieving inferior performance. (2) In the variant “Ours w/o PSTG”,
   565	we replace the LVLM-generated semantic token with a learnable token, similar to the design of the [class] token [17], to integrate
   566	visual information. As shown in Tab. 2, this substitution leads to a
   567	substantial performance drop since the randomly initialized learnable token lacks rich semantic cues. This result underscores the
   568	importance of our PSTG mechanism, which contributes to a more
   569	comprehensive understanding of pedestrian images.
   570	Effectiveness of the SGI module. In the “Ours w/o SGI” variant,
   571	we remove the SGI module and rely solely on the LVLM-generated
   572	semantic token for ReID. As shown in Tab. 2, this configuration
   573	achieves reasonably good performance, suggesting that our PSTG
   574	effectively captures essential pedestrian semantic information. However, the variant struggles to outperform the baseline, emphasizing
   575	the importance of the SGI module in leveraging the generated semantic token. It represents a novel paradigm of enhancing identity
   576	representations with the LVLM-generated semantic token.
   577	Ablation of the camera semantic supplementation strategy. In
   578	Tab. 3, the result of CSS-vreid shows that camera semantics can
   579	improve the representation ability of the generated token. When
   580	transferring the usage of camera embeddings to the input of the visual model (denoted by CSS-xp ), we observe a better performance.
   581	This design helps to improve the robustness of visual features and
   582	
   583	(a)
   584	
   585	Query
   586	
   587	Rank-1 --------------------------------------------------------------------------------------> Rank-8
   588	
   589	(b)
   590	
   591	Query
   592	
   593	Rank-1 --------------------------------------------------------------------------------------> Rank-8
   594	
   595	Fig. 3. Visualization of retrieval results. For each query, the first
   596	and the second rows show the top-8 retrieval results of the baseline
   597	and our method on CUHK03, respectively. Retrieved images with
   598	green and red boxes are correct and incorrect results, respectively.
   599	For each query in the figure, there are four matched person images
   600	in the gallery. Best viewed in color and zoomed in.
   601	the LVLM-generated semantic token, further improving the model’s
   602	ability to match pedestrians across cameras.
   603	Inference time. Our method requires generating only a single semantic token using LVLM, which can be accomplished in a single
   604	forward pass during inference, eliminating the need for iterative autoregressive generation while ensuring practicality. The inference
   605	time for processing 64 images in parallel is 784 ms for the visualonly baseline and 906 ms for our model on a single GPU with 12GB
   606	of memory. FLOPs are 128G and 257G, respectively. The marginal
   607	increase in inference time is acceptable and can be further optimized
   608	through techniques such as flash attention [31].
   609	Visualization of retrieval results. As shown in Fig. 3, the baseline model often relies on coarse-grained and vulnerable features,
   610	such as general color, similar clothing, and similar pose, leading
   611	to false positives. In contrast, guided by the semantic token, our
   612	method effectively captures nuanced identity-specific features, such
   613	as distinctive patterns and accessories, enabling superior differentiation between individuals. Our method also showcases robustness in
   614	scenarios involving occlusions, whether from objects (Fig. 3 (a)) or
   615	other pedestrians (Fig. 3 (b)).
   616	4. CONCLUSION

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '317,739p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   317	III. M ETHODOLOGY
   318	In this section, we focus on the proposed method for VIReID with instance-dependent noisy labels. Firstly, we follow
   319	the twin noisy labels in [19] to present the problem definition
   320	of VI-ReID with instance-dependent label noise in III-A.
   321	Subsequently, the pipeline of our method is provided in III-B.
   322	Furthermore, we present the anchor pairs selection module
   323	based on Gaussian-Mixture-Model(GMM) in III-C to divide
   324	samples into clean and noisy subsets. Ultimately, we provide
   325	a hybrid loss function to achieve a robust training process.
   326	A. Problem Definition
   327	The objective of the VI-ReID task is to retrieve the
   328	corresponding identity for each sample in the query set from
   329	the gallery set, thereby facilitating precise individual identification. Generally, each image x within the query/gallery
   330	set has an ID label y and a modality index m, which
   331	belongs to the visible subset V or infrared subset I, e.g.
   332	nm
   333	M = {xim , yim }i=0
   334	, m ∈ {v, i}. Under the consideration of
   335	noisy labels, the authentic label of each sample is substituted
   336	with the noisy label, resulting in a dataset comprising each
   337	sample represented as {xim , ỹim }. The noisy label ỹ pertains to
   338	random noise, instance-dependent label noise, or real-world
   339	noise, and we get different noisy labels through different
   340	generation pathways. More generation details of noise labels
   341	are provided in IV-A. In the realm of VI-ReID, noisy
   342	labels induce a special harmful phenomenon called noisy
   343	correspondence. It introduces confusing correlations between
   344	g
   345	q
   346	the gallery sample xi and the query sample x j . For example,
   347	we defined the correspondence label as yicj ∈ {0, 1}. If gallery
   348	g
   349	q
   350	sample xi is corresponding to query sample x j , then the
   351	c
   352	c
   353	correspondence label has yi j = 1, conversely yi j = 0. This
   354	means that we can use the category labels y q and y g of
   355	
   356	samples x q and x g to represent the corresponding label y c :
   357	(
   358	g
   359	q
   360	1 if yi = y j
   361	c
   362	(1)
   363	yi j =
   364	q
   365	g
   366	0 if yi ̸ = y j .
   367	Similarly, the corresponding label y c is replaced with a noisy
   368	version ỹ c . We consider cross-modal retrieval with noisy
   369	labelling and noisy correspondence cases.
   370	B. Co-Training Pipeline
   371	As Fig. 2(a) showcases, our method utilizes a co-training
   372	model. In view of previous findings, the structure of cotraining helps to enhance the robustness of the model. This
   373	is because by co-training two identical modules, the model
   374	can fit the data from different perspectives, thus enhancing
   375	generalisation and reducing the effect of noise annotations.
   376	Initially, the two modules, each with a different initialization,
   377	undergo a warm-up phase to endow the model with the latent
   378	capability to discern authentic labels. Subsequently, identity
   379	classification is performed for each sample, utilizing the crossentropy loss function L id to compute the loss value L, which
   380	will be employed in Section III-C to discriminate the clean and
   381	noisy subsets of samples. Thereafter, leveraging the identified
   382	subsets and the classification confidence, anchor points are
   383	selected to facilitate the partitioning into triplets. Finally, upon
   384	this foundation, robust hybrid loss is calculated, and it is
   385	utilized to make the model robust to re-identification training.
   386	Specifically, we instantiate Encoders E = {E V , E I },
   387	to extract deep features, FV and FI , from visible samples
   388	x v and infrared samples x i , respectively. The elicited features
   389	fulfil a bifurcated function: on the one hand, they are entered
   390	into the Classifiers C = {C V , C I } for the precise delineation
   391	and identification of specific identities; on the other hand,
   392	they play a pivotal role in the formulation of the correlative
   393	
   394	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   395	
   396	ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE
   397	
   398	matrix Tc , which is integral to the process of retrieval and pair
   399	division.
   400	C. Anchor Pairs Selection
   401	As mentioned in section III-B, the selection process can
   402	be divided into two steps: coarse-grained selection for clean
   403	subset and fine-grained selection for triplets.
   404	Due to the complexity and variability of the patterns of
   405	noise, it is difficult for the model to directly point to clean
   406	samples in the entire dataset for noisy labels, especially
   407	those that are instance-dependent. Thanks to previous work,
   408	it was realised that the model fits clean data first rather than
   409	noisy data in the warm-up phase, which makes the approach
   410	by modelling per-sample loss value distribution a viable
   411	option. This approach follows the small loss principle, which
   412	considers samples that have small loss values as clean samples
   413	and vice versa as noise samples. According to the warmed-up
   414	module parameters, we can calculate the discrimination loss
   415	values for each sample:
   416	N
   417	N
   418	ℓdis (θ ) = {ℓi }i=1
   419	= {Ldis (xi , yi )}i=1
   420	.
   421	
   422	(2)
   423	
   424	Considering the re-identification dataset is a classification
   425	dataset in substance, we choose cross-entropy loss as the
   426	discrimination loss function. As shown in Fig. 2(b), the Ldis
   427	in Eq. 2 is defined as follows:
   428	Ldis (xi , yi ) = −logP(yi | C(E(xi ))).
   429	
   430	p(ℓ) =
   431	
   432	γk p(ℓ|k),
   433	
   434	(4)
   435	
   436	k=1
   437	
   438	where γk and p(ℓ|k) are the mixture coefficient and the
   439	probability density of the k-th component, respectively.
   440	Through the loss and the confidence value, we can directly
   441	find the obvious noisy label set S O N with a threshold ε1 .
   442	S O N = {(xi , yi ) | ℓi > ε1 },
   443	
   444	(5)
   445	
   446	where ℓi is the normalized loss of i-th sample. We treat the
   447	component with the lower mean as the clean one. To divide the
   448	clean set S C and the indistinguishable noise set S I N , we make
   449	use of the posterior probability p(kl |ℓi ):
   450	S = {(xi , yi ) | ℓi ≤ ε1 , p(kl |ℓi ) > ε2 },
   451	
   452	(6)
   453	
   454	= {(xi , yi ) | ℓi ≤ ε1 , p(kl |ℓi ) ≤ ε2 },
   455	
   456	(7)
   457	
   458	C
   459	
   460	S
   461	
   462	IN
   463	
   464	pair and contains a positive indication (yicj = 1) or a negative
   465	indication (yicj = 0). To filter the anchors and further segment
   466	the samples, we set a threshold δ measuring classification
   467	confidence. Thus, the following clean correspond dataset Dc
   468	and noisy correspond dataset Dn are available:
   469	n
   470	
   471	o
   472	
   473	Dc = xi , x j , yicj | p (kl |ℓi ) > δ, p kl |ℓ j > δ , (8)
   474	n
   475	
   476	o
   477	
   478	Dn = xi , x j , yicj | p (kl |ℓi ) > δ, p kl |ℓ j ≤ δ . (9)
   479	Since it can’t be correctly divided, we opt to discard the
   480	sample pairs whose confidence levels are both lower than δ.
   481	Note that subset S is used for identification, while subset D
   482	is used for retrieval. Then, we refine the corresponding label
   483	by Eq. 10, which in turn obtains the corrected corresponding
   484	label.
   485	
   486	
   487	ŷicj = I yicj ∈ Dc ⊙ yicj ,
   488	(10)
   489	where the I(·) is an indicator, the value is 1 if the condition
   490	in the round bracket is satisfied, otherwise it is 0. ⊙ is
   491	the xnor operation, intended to examine the validity of
   492	the
   493	corresponding label. Eventually, it can be divided into
   494	
   495	
   496	c = 0 , FP-FN ŷ c = 0 , ŷ c = 1 , TPTP-TN ŷicj = 1, ŷik
   497	i
   498	j
   499	ik
   500	
   501	
   502	
   503	
   504	c = 1 , FP-TN ŷ c = 0 , ŷ c = 0 with their
   505	FN ŷicj = 1, ŷik
   506	ij
   507	ik
   508	rectified correspondence.
   509	
   510	(3)
   511	
   512	We employ a two-component Gaussian Mixture Model to
   513	fit the distribution of per-sample losses as the prior methods:
   514	K
   515	X
   516	
   517	4861
   518	
   519	where the posterior probability p(kl |ℓi ) = p(kl ) p(ℓi |kl )/ p(li )
   520	is the clean probability of i-th sample. For simplicity, we set
   521	the threshold ε1 to 0.95 and ε2 to 0.5 through all experiments.
   522	With the benefit of coarse-grained filtering, we can construct
   523	triplets using the attributive pseudo-labels of samples to clean
   524	and noisy subsets for fine-grained selection. Firstly, to realize
   525	that sample pairs appear cross-modal, we need to mix the
   526	two-modal datasets, with reference to [82]. Each sample pair
   527	(xim 1 , x mj 2 , yicj ) contains two image samples xim 1 , x mj 2 and a
   528	correspondence label yicj . The correspondence label reflects
   529	the correspondence between the two samples in the sample
   530	
   531	D. Robust Hybrid Loss
   532	To classify different identities, we prompt a robust hybrid
   533	loss. It contains two parts, one for person identification and
   534	the other for cross-modality retrieval.
   535	1) Robust Identification Loss: As described in section III-C,
   536	we have three subsets for identity labels, and we treat different
   537	subsets with different loss functions.
   538	a) Obvious noisy set: The obvious noise set is caused
   539	by an annotation accident. The label is unbelievable, so it can
   540	be an unsupervised task. According to [83], the classification
   541	confidence of the classifier serves as a soft pseudo-label.
   542	A sorting weight is defined as follows:
   543	an = φn ( f θ (xi )),
   544	
   545	(11)
   546	
   547	where φn (·) returns the identity of the n-th in the classification
   548	confidence ranking. Thus we have the loss function:
   549	LO N = −
   550	
   551	K1
   552	N
   553	X
   554	X
   555	an
   556	log( f θ (xi )),
   557	α
   558	N
   559	i=1
   560	
   561	(12)
   562	
   563	n=1
   564	
   565	where f θ (·) is the classifier with characters θ , and the α is
   566	used to normalize weights over N classes of identities
   567	P N and
   568	an
   569	2
   570	it should be 1+N
   571	due to the sum of sorting weight n=1
   572	N
   573	)N
   574	ON,
   575	equal to (1+N
   576	.
   577	x
   578	comes
   579	from
   580	the
   581	obvious
   582	noise
   583	set
   584	S
   585	i
   586	2
   587	and K 1 is the number of the samples in it. Considering
   588	the advantages of early learning [84], the empirical and
   589	generalization capabilities possessed by the warm-up method
   590	allow obvious noisy labels to be correctly optimized
   591	by Eq. 12.
   592	
   593	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   594	
   595	4862
   596	
   597	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   598	
   599	b) Clean set: For the clean subset, we are confident
   600	enough that their labels are not noisy. Thus, we can train
   601	models on this subset with a vanilla cross-entropy loss.
   602	LC = −
   603	
   604	K2
   605	X
   606	
   607	yi log( f θ (xi )),
   608	
   609	(13)
   610	
   611	i=1
   612	
   613	where sample pairs (xi , yi ) come from subset S C , and K 2 is
   614	the number of the samples in S C .
   615	c) Indistinguishable noisy set: In this part, the set
   616	contains the most indistinguishable samples, i.e., the set
   617	mixes cleanly labelled samples and noisily labelled samples.
   618	In this section, the set contains the most indistinguishable
   619	samples, i.e., the set mixes cleanly labelled samples and noisily
   620	labelled samples. Since distinguishing the two becomes nearly
   621	impossible and carries a high risk, we consider employing a
   622	robust loss function.
   623	To avoid excessive loss values for samples with noisy labels,
   624	we employ GCE [31] as the indistinguishable noisy loss L I N
   625	to deal with the difficult samples in subset S I N :
   626	K
   627	
   628	LI N =
   629	
   630	3
   631	1X
   632	q
   633	(1 − yi f θ (xi )).
   634	q
   635	
   636	(14)
   637	
   638	i=1
   639	
   640	To summarize, our robust identification loss consists of three
   641	components, balanced by three adaptive hyper-parameters
   642	β1 , β2 and β3 . The adaptive hyper-parameters are computed
   643	from the proportion of the subsets S C , S I N and S O N to the
   644	full set S, respectively.
   645	Lid = β1 LC + β2 L I N + β3 L O N .
   646	
   647	(15)
   648	
   649	2) Robust Retrieval Loss: Combined with the subsets
   650	delineated in Section III-C, we can solve the noisy
   651	correspondence problem in retrieval concerning [19]. The
   652	robust retrieval loss we employ is a variant of the triplet loss,
   653	which considers each of the four subsets of cases:
   654	Lr e = m + Lhet + Lhom ,
   655	
   656	(16)
   657	
   658	where m is the fixed margin to avoid outlier values and the
   659	Lhet definite as follows:
   660	
   661	
   662	Lhet =
   663	
   664	(−1)
   665	
   666	c
   667	ŷicj ⊗ ŷik
   668	
   669	
   670	
   671	1− ŷicj
   672	
   673	
   674	(−1)
   675	
   676	
   677	
   678	di j + (−1)
   679	
   680	1− ŷicj
   681	
   682	
   683	
   684	(
   685	
   686	c
   687	1− ŷik
   688	
   689	)
   690	
   691	2
   692	
   693	
   694	c
   695	ŷicj ⊗ ŷik
   696	(1− ŷikc )
   697	
   698	c
   699	ŷicj ⊙ ŷik
   700	
   701	dik
   702	
   703	,
   704	
   705	(17)
   706	where ⊙ and ⊗ denote the xnor and xor operations,
   707	respectively. di j and dik represent the distances between
   708	features (xi , x j ) and between features (xi , xk ), respectively.
   709	TP-TN and FP-FN pairs are heterogeneous pairs, and both
   710	use Eq. 17 to achieve identity space optimization.
   711	Lhom is defined as shown below:
   712	
   713	
   714	c c
   715	c
   716	Lhom = (−1) ŷi j ŷik ŷicj ⊙ ŷik
   717	dis ,
   718	(18)
   719	considering TP-FN and FP-TN pairs are homogeneous pairs,
   720	an extra pair (xi , xs ) will be sampled and utilize the distance
   721	dis between anchor xi and extra instance xs to calculate the
   722	homogeneous retrieval losses.
   723	
   724	Fig. 3. Label distribution of the re-labelled dataset SYSU-Real. Different
   725	colours represent different partitions, and we utilize this property to show the
   726	proportion of noise within the infrared and visible subsets. Note that only
   727	two items, visible noise ratio and infrared noise ratio, are proportions to the
   728	number of samples in the modal subsets, all others are proportions to the
   729	number of samples in the full dataset.
   730	
   731	Ultimately, our robust loss is a hybrid of the person
   732	identification loss and the cross-modality retrieval loss.
   733	Lhyb = Lid + βLr e ,
   734	
   735	(19)
   736	
   737	where the β is a hyper-parameter for balancing identification
   738	and retrieval.
   739	IV. E XPERIMENTS

codex
有两篇排版把方法和表格混在同一页，文本顺序被列布局打乱了。我会按页和行号再补一次，重点补 HDNet 的跨模态重组末尾、LVLM-ReID 的训练和推理段，以及各篇消融证据。
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
/bin/zsh -lc "pdftotext -f 9 -l 10 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Visible-Infrared Person Re-Identification
     2	
     3	265:9
     4	
     5	features:
     6	𝐾
     7	
     8	𝑖
     9	𝑐𝑚
    10	=
    11	
    12	1 Õ 𝑖
    13	𝐹𝑚,𝑗 ,
    14	𝐾
    15	
    16	(20)
    17	
    18	𝑘=1
    19	𝑖
    20	where 𝑚 ∈ {𝑣_𝑣, 𝑡_𝑡, 𝑣_𝑡, 𝑡_𝑣 }, 𝐹𝑚,𝑗
    21	is the feature of the 𝑗th sample from class 𝑖, and 𝐾 represents
    22	the number of images for the same identity within a batch. The loss function for center alignment
    23	is defined as:
    24	𝑁
    25	
    26	1 Õ 𝑖
    27	𝐿𝑟 =
    28	k𝑐 𝑣_𝑣 − 𝑐𝑡𝑖 _𝑡 k 2 − k𝑐 𝑖𝑣_𝑡 − 𝑐𝑡𝑖 _𝑣 k 2 + 𝜌 ,
    29	(21)
    30	𝑁 𝑖=1
    31	
    32	where 𝑁 denotes the total number of identities in a batch, and 𝜌 is a pre-defined margin. This center
    33	alignment loss helps guide the model to focus on shared identity information across modalities
    34	while minimizing the modality discrepancy in the recombined features.
    35	3.5
    36	
    37	Total Loss Function
    38	
    39	The total loss function of HDNet is defined as:
    40	𝐿 = 𝐿𝑏𝑎𝑠𝑒 + 𝛼𝐿𝑐 + 𝛽𝐿𝑎𝑑𝑣 + 𝛾𝐿𝑟 ,
    41	
    42	(22)
    43	
    44	where 𝛼, 𝛽, and 𝛾 are hyperparameters used to balance the contributions of each loss term.
    45	4
    46	
    47	Experiments
    48	
    49	4.1 Dataset and Experimental Setup
    50	Datasets. We evaluated the proposed method on two publicly available datasets, SYSU-MM01 [40]
    51	and RegDB [23].
    52	— SYSU-MM01 is a large dataset captured by four visible light cameras and two near-infrared
    53	cameras, covering indoor and outdoor environments. The training set consists of 395 pedestrian
    54	identities with 22,258 visible light images and 11,909 infrared images. The test set comprises
    55	query and gallery sets, totaling 96 pedestrian identities. The query set includes 3,803 infrared
    56	images, and the gallery set contains 301 (3,010) randomly sampled visible light images for
    57	Single-Shot (Multi-Shot) scenarios.
    58	— RegDB is a dataset captured by one visible light camera and one thermal infrared camera. It
    59	consists of a total of 412 identities, each with 10 visible light images and 10 infrared images,
    60	resulting in a total of 4,120 images for both modalities. The dataset is randomly split into
    61	training and test sets, each containing images of 206 pedestrian identities.
    62	Evaluation Metrics. To ensure fair comparison with existing methods, all experiments follow
    63	common evaluation settings used in existing cross-modal person re-identification methods. SYSUMM01 has two different test modes: All-Search and Indoor-Search. In All-Search, the gallery set
    64	includes all visible light images, while in Indoor-Search, the gallery set only contains indoor visible
    65	light images. RegDB has two test modes: the query set contains visible light images, and the
    66	gallery set contains infrared images, and vice versa. For both datasets, we use Cumulated Matching
    67	Characteristics and Mean Average Precision (mAP) to evaluate the model’s performance.
    68	Implementation Details. HDNet is trained using PyTorch on a single Tesla A40 GPU. We use
    69	ResNet-50 [11] pretrained on ImageNet as the backbone network, with the convolutional stride of
    70	the last layer set to 1. During the training phase, the aspect ratio of input images is uniformly set to
    71	288 × 144, and the following data augmentation methods are applied: horizontal flipping, random
    72	cropping, color jittering, and random erasing [59] with a probability of 50%. In each training batch,
    73	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
    74	
    75	265:10
    76	
    77	H. Ding et al.
    78	Table 1. The Ablation Experiment Results of Our Method on the SYSU-MM01 Dataset (%)
    79	
    80	Method
    81	𝐵 (𝐿𝑖𝑑 + 𝐿𝑏ℎ𝑡𝑟𝑖 )
    82	𝐵 + 𝐶𝐷𝑀
    83	𝐵 + 𝑀𝐼𝐷𝑀 ∗
    84	𝐵 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣
    85	𝐵 + 𝐶𝐷𝑀 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣
    86	𝐵 + 𝐶𝐷𝑀 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣 + 𝐿𝑟
    87	
    88	All-Search
    89	
    90	Indoor-Search
    91	
    92	R1
    93	
    94	R10
    95	
    96	R20
    97	
    98	mAP
    99	
   100	R1
   101	
   102	R10
   103	
   104	R20
   105	
   106	mAP
   107	
   108	64.94
   109	66.42
   110	58.86
   111	67.94
   112	72.65
   113	74.84
   114	
   115	94.30
   116	95.27
   117	91.58
   118	94.92
   119	95.82
   120	97.56
   121	
   122	97.81
   123	97.03
   124	96.22
   125	98.70
   126	98.66
   127	98.90
   128	
   129	61.58
   130	61.58
   131	56.35
   132	65.51
   133	67.13
   134	68.68
   135	
   136	70.36
   137	69.84
   138	61.81
   139	73.31
   140	77.10
   141	81.14
   142	
   143	97.29
   144	95.65
   145	95.53
   146	98.05
   147	98.76
   148	98.98
   149	
   150	99.45
   151	98.41
   152	98.61
   153	99.58
   154	99.61
   155	99.81
   156	
   157	75.20
   158	75.20
   159	69.33
   160	77.52
   161	78.45
   162	82.20
   163	
   164	Table 2. The Ablation Experiment Results of Our Method on the RegDB Dataset (%)
   165	
   166	Method
   167	𝐵 (𝐿𝑖𝑑 + 𝐿𝑏ℎ𝑡𝑟𝑖 )
   168	𝐵 + 𝑀𝐼𝐷𝑀 ∗
   169	𝐵 + 𝐶𝐷𝑀
   170	𝐵 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣
   171	𝐵 + 𝐶𝐷𝑀 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣
   172	𝐵 + 𝐶𝐷𝑀 + 𝑀𝐼𝐷𝑀 ∗ + 𝐿𝑎𝑑𝑣 + 𝐿𝑟
   173	
   174	All-Search
   175	
   176	Indoor-Search
   177	
   178	R1
   179	
   180	R10
   181	
   182	R20
   183	
   184	mAP
   185	
   186	R1
   187	
   188	R10
   189	
   190	R20
   191	
   192	mAP
   193	
   194	84.96
   195	83.54
   196	84.91
   197	86.71
   198	90.55
   199	92.19
   200	
   201	95.53
   202	95.29
   203	95.94
   204	97.57
   205	95.97
   206	97.67
   207	
   208	97.33
   209	97.48
   210	97.61
   211	98.64
   212	98.20
   213	98.93
   214	
   215	76.65
   216	76.03
   217	77.63
   218	80.55
   219	78.75
   220	81.08
   221	
   222	81.65
   223	77.96
   224	82.15
   225	85.39
   226	83.88
   227	87.48
   228	
   229	94.12
   230	92.96
   231	95.87
   232	96.50
   233	95.74
   234	97.48
   235	
   236	96.91
   237	97.18
   238	97.67
   239	98.69
   240	97.57
   241	98.79
   242	
   243	73.13
   244	71.11
   245	73.70
   246	77.47
   247	75.84
   248	77.92
   249	
   250	𝑃 = 8 pedestrian identities are randomly sampled, and 𝐾 = 4 images of visible light and infrared
   251	randomly sampled under the same identity are used, resulting in a total of 2𝑁 = 2𝑃𝐾 = 64 original
   252	images. Both original and augmented images are input to the network, with a total of 128 images
   253	in each training batch. We use SGD for model optimization, with a momentum parameter set to
   254	0.9, an initial learning rate of 0.01, and a linear increase to 0.1 after 10 iterations. The learning rate
   255	is then decayed to 0.01 at the 20th iteration, and further decayed to 0.001 at the 50th iteration, with
   256	a total of 80 iterations. In addition, the parameter 𝜌 in the 𝐿𝑟 is set to 0.7. The hyperparameters 𝛼,
   257	𝛽, and 𝛾 are set to 1, 1.2, and 0.8, respectively.
   258	4.2
   259	
   260	Ablation Experiments

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | rg -n \"Ablation|ablation|MPT|two-stage|prompts|visualization|t-SNE|alternating|knowledge propagation|alignment|mutual|baseline\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
19:prompts generated by original CLIP can not accurately describe
21:generate prompts which are more suitable for VI-ReID. Based on
22:such idea, we design a novel network that helps prompts focus
24:prompts and image encoders. Specifically, when optimizing prompts,
25:we introduce modality knowledge propagation loss. The loss aligns
27:the knowledge in image encoders is transferred to prompts. When
28:optimizing encoders, we design modality alignment loss. The loss
31:the mutual reinforcement of two parts, the quality of both prompts
55:prompts to help alleviate inter-modality discrepancies. These works
57:by two-stage training strategy [5]. They fix image encoders to optimize text prompts in the first stage and fix text prompts to optimize
59:image encoders in the second stage. The generated text prompts
64:However, the text prompts generated by multimodal large language models can not effectively describe identity information of
65:visible-infrared person re-identification, text prompts, knowledge
75:tion, prompts generated directly by the original CLIP also can not
78:to improve the quality of text prompts.
80:In this paper, we propose the Mutual Promotion Training (MPT)
81:network for this issue. Generally, prompts that focus on person
85:Conversely, prompts optimized with the constraints of trained en∗
88:this idea, we attempt to introduce more person-related information into prompts. Specifically, we adopt an alternating training
90:strategy which alternately optimizes the text prompts and image
92:encoders in different epochs. When optimizing text prompts, we
94:design the modality knowledge propagation loss. By aligning preACM ISBN 979-8-4007-1877-9/2025/06
96:dicted class probability of features, the loss can enable prompts to
105:and encoders can benefit from continuously optimized prompts.
106:However, because the optimization objectives of prompts and
107:encoders are different, optimizing prompts and image encoders simultaneously may cause mutual interference. For example, when
108:prompts learn person-related information from image encoders,
110:avoid the mutual interference, we propose the alternating training
111:strategy to implement the mutual reinforcement between prompts
112:and encoders. Specifically, we fix image encoders (prompts) and
113:optimize the prompts (image encoders) when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 (𝑒𝑝𝑜𝑐ℎ =
115:prompts (image encoders) and 𝐹𝑡𝑒𝑥𝑡 (𝐹𝑖𝑚𝑔 ) to represent the optimization process of text prompts (image encoders), the optimization process can be formalized as:
117:MPT Module
138:prompts
153:prompts
184:Fig. 2: The MPT network introduces an alternating training strategy,
185:in which the modality knowledge propagation loss is designed for
186:optimizing text prompts and the modality alignment loss is used
194:learn more person-related information from the encoders. This significantly alleviates the problem that prompts lack person-related
196:modality alignment loss. The loss treats the text features as a special intermediate modality, and aligns both visible and infrared
197:modalities to the text modality. In this way, prompts can serve
200:help of MPT, text prompts can help image encoders become better,
201:and better image encoders can in turn promote prompts to learn
203:both text prompts and image encoders will continue to improve.
204:We evaluate the MPT network on two widely used datasets to
206:that the MPT network indeed solves the problems existing in the
229:is exactly what prompts need most. Therefore, when optimizing
230:prompts, our purpose is to propagate the knowledge contained in
231:image encoders to prompts.
234:each other. Under such a premise, we propose the modality knowledge propagation loss. The loss constrains the predicted class probability of text features to be similar to that of image features. In
235:this way, person-related knowledge is propagated from image encoders into text prompts. The loss function can be expressed as:
239:Some studies follow the two-stage training strategy in CLIP ReID
240:to generate text prompts, but such strategy can only generate prompts
244:to generate prompts that better focus on person-related information remains an urgent problem to be solved.
245:In this case, we propose the MPT network, which improves both
246:prompts and image encoders by alternately optimizing them. The
247:structure of MPT is shown in the Fig. 2. Specifically, the text encoder takes prompts as input and outputs corresponding text features, while image encoders take images as input and output corresponding image features. When 𝑒𝑝𝑜𝑐ℎ = 2𝑖, we fix image encoders and optimize prompts with modality knowledge propagation loss. And when 𝑒𝑝𝑜𝑐ℎ = 2𝑖 + 1, we fix prompts and optimize
248:image encoders with modality alignment loss.
256:Generally, encoders trained on VI-ReID datasets can help prompts
257:focus on person information, while prompts which focus on person
258:information can also help encoders extract better features. Therefore, the target of our network is that prompts can increasingly
285:we can consider optimized prompts as a special intermediate modality. When optimizing encoders, we treat the text features generated by prompts as the “bridge” and use them to connect the two
287:The proposed modality alignment loss first calculates the similarity between text features and image features, then utilizes contrastive learning loss to align the image features of both modalities
313:MPT (ours)
483:Table 2: Ablation studies of each component in MPT network.
498:two-stage [21]
499:MPT w/ 𝐿𝑀𝑃 𝑇
500:MPT
520:Fig. 3: Illustration of visualization results using t-SNE. Features
528:where 𝑊𝐼 and 𝑊𝑇 are linear layers. Then the modality alignment
545:number of text prompts. 𝑞𝑘 is the value in the target distribution,
561:Therefore, the mutual promotion loss is expressed as:
566:With the help of the above loss functions, text prompts and image encoders can mutually promote each other, ultimately improving the performance of cross-modality retrieval.
574:MPT
577:and MPT network) between cross-modality features. Blue and green
597:MPT
624:show the original image followed by feature maps extracted by twostage model and MPT network.
634:The MPT network outperforms all methods. The results suggest
635:that prompts with person-related information can more effectively
637:reason for achieving such high performance is that we treat prompts
639:two modalities to text features. This effectively reduces the discrepancies between the two modalities. More importantly, we utilize the knowledge distillation technique to help prompts learn
640:person-related information, making these prompts better serve as
641:the bridge. The MPT network also outperforms all existing stateof-the-art methods on RegDB dataset, which proves the previous
648:Feature Distribution. As shown in the Fig. 3, we display the distribution of features extracted by the baseline network and the
649:MPT network. It is obvious that features with the same identity but
650:different modalities are more clustered in the MPT network than
651:in the baseline network, indicating that the MPT network greatly
655:in Fig. 4. As observed, the MPT network significantly enlarges the
656:gap between the means of inter-class and intra-class distances compared to the other two models (𝛿1 < 𝛿2 < 𝛿3 ). Therefore, MPT network effectively reduces intra-class discrepancies, which indicates
657:that the prompts indeed bridge the inter-modality gap.
658:Focus Areas. To verify that prompts generated by MPT focus
659:on person-related information, we visualized feature maps of twostage model and MPT network. During training, 𝑊𝐼 ⋅𝐼𝑖 and 𝑊𝑇 ⋅𝑇𝑖
662:of MPT pay more attention to body areas, which means that the
663:prompts generated by MPT are more sensitive to identity-related
666:Ablation Study
669:the effectiveness of key components in the MPT network. The experiments are conducted under both all-search and indoor modes,
671:results under all-search mode for analysis. Compared to the baseline network, MPT improves 5.45% on Rank-1 and 5.88% on mAP.
672:The result means that text prompts and image encoders can indeed promote each other, ultimately enhancing the model’s ability
673:for cross-modality retrieval. When using the same network structure and loss functions, the alternating training strategy (MPT w/
674:𝐿𝑀𝑃 𝑇 ) improves 1.69% and 2.23% over two-stage training strategy. This indicates that by improving the quality of generated text
675:prompts, CLIP can play a more significant role in VI-ReID. What’s
677:(the contrastive learning loss) indicate that the designed losses effectively help prompts and encoders mutually reinforce each other.
678:In addition, under the same experimental setup, the two-stage
680:MPT network requires 12 hours and 12 minutes. This indicates
681:that MPT can significantly improve performance with only a slight
690:In this paper, we propose the MPT network for VI-ReID, in which
691:image encoders and text prompts mutually promote each other. In
692:this way, we obtain text prompts with more person-related information, which better align visible and infrared modalities at semantic level. In addition, we design the dedicated loss functions
693:for optimizing prompts and encoders respectively to ensure the
695:the superior performance of the MPT network compared with the
748:and feature alignment. In Proceedings of the IEEE/CVF international conference on
758:concordant attention via target-aware alignment for visible-infrared person reidentification. In Proceedings of the IEEE/CVF international conference on computer vision. 11122–11131.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | nl -ba | sed -n '253,435p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   253	2.4. Optimization and Inference
   254	During training, we optimize the parameters of both the visual model
   255	and the SGI module. The parameters of LLM are frozen, but gradients backpropagate through it to update other learnable components.
   256	By leveraging the generated <REID> token in conjunction with the
   257	SGI module, we achieve a joint end-to-end training that harnesses
   258	the strengths of LVLM in instruction-following and visual semantic understanding. This process allows for the integration of rich
   259	
   260	Table 1. Comparison with the state-of-the-art methods. The results of our method and the best results of comparison methods are
   261	shown in bold.
   262	DukeMTMC-reID
   263	
   264	Market-1501
   265	
   266	mAP
   267	
   268	Rank-1
   269	
   270	mAP
   271	
   272	Rank-1
   273	
   274	mAP
   275	
   276	Rank-1
   277	
   278	mAP
   279	
   280	Rank-1
   281	
   282	CNN
   283	
   284	MGN [19]
   285	DG-Net [20]
   286	SAN [5]
   287	Pyramid [21]
   288	Relation-Net [22]
   289	RGA-SC [23]
   290	CDNet [6]
   291	CAL [7]
   292	
   293	78.4
   294	74.8
   295	75.5
   296	79.0
   297	78.6
   298	76.8
   299	76.4
   300	
   301	88.7
   302	86.6
   303	87.9
   304	89.0
   305	89.7
   306	88.6
   307	87.2
   308	
   309	86.9
   310	86.0
   311	88.0
   312	88.2
   313	88.9
   314	88.4
   315	86.0
   316	87.0
   317	
   318	95.7
   319	94.8
   320	96.1
   321	95.7
   322	95.2
   323	96.1
   324	95.1
   325	94.5
   326	
   327	67.4
   328	76.4
   329	76.9
   330	75.6
   331	77.4
   332	-
   333	
   334	68.0
   335	80.1
   336	78.9
   337	77.9
   338	81.1
   339	-
   340	
   341	-
   342	
   343	-
   344	
   345	ViT
   346	
   347	TransReID [24]
   348	PAT [25]
   349	DCAL [26]
   350	AAformer [27]
   351	CLIP-ReID [8]
   352	
   353	80.6
   354	78.2
   355	80.1
   356	80.0
   357	82.5
   358	
   359	89.6
   360	88.8
   361	89.0
   362	90.1
   363	90.0
   364	
   365	88.2
   366	88.0
   367	87.5
   368	88.0
   369	89.6
   370	
   371	95.0
   372	95.4
   373	94.7
   374	95.4
   375	95.5
   376	
   377	79.0
   378	80.3
   379	
   380	80.3
   381	81.6
   382	
   383	55.7
   384	53.6
   385	58.2
   386	59.5
   387	
   388	64.2
   389	64.5
   390	67.1
   391	67.1
   392	
   393	LVLM
   394	
   395	LVLM-ReID (Ours)
   396	
   397	82.8
   398	
   399	92.2
   400	
   401	89.2
   402	
   403	95.6
   404	
   405	82.3
   406	
   407	84.6
   408	
   409	59.8
   410	
   411	68.1
   412	
   413	Backbone
   414	
   415	Methods
   416	
   417	CUHK03
   418	
   419	Occluded-Duke
   420	
   421	semantic cues into the visual representations, improving the accuracy of pedestrian identity recognition. The overall training loss is a
   422	weighted combination of the identity classification loss Lid [2] and
   423	the triplet loss Ltri [4], which is expressed as follows:
   424	L = α1 Lid + α2 Ltri ,
   425	
   426	(5)
   427	
   428	where α1 and α2 are balancing factors that control the contribution
   429	of each loss term.
   430	During inference, the LVLM is also used to generate the
   431	<REID> token for each input image. Then, the reinforced semantic token representation, v̂reid , is used to compute the cosine
   432	similarity between different person images. These similarity scores
   433	are employed for identity matching, allowing the model to identify
   434	pedestrians. Note that the identity representations of persons in the
   435	large gallery databases need be extracted only once in applications.

exec
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | rg -n \"Ablation|S-Prefix|L-Adapter|LoRA|Prefix|Adapter|component|conflict|parameters|Visualization|visualization|Full Tuning|full-tuning|Knowledge\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12:and fully fine-tune the entire network, which have shown notable performance improvements compared to uni-modal pretraining models. However, full-tuning a large model is prone
17:three lightweight PETL components including Prefix, LoRA and
18:Adapter, where Prefix and LoRA are devised together to mine
20:Adapter is designed to adjust global feature representations.
22:to the unified architecture of TPR. For one thing, S-Prefix is
25:performance of the vanilla prefix. For another thing, L-Adapter is
27:distribution, which can resolve conflicts caused by overlap and
31:4.7% parameters. Code is available at https://github.com/LiuYating/UP-Person.
59:Knowledge of CLIP (KC)
60:TPR-specific Knowledge of CLIP (PKC)
61:TPR-specific Knowledge from Training datasets (PKT)
102:Knowledge
104:TPR-specific Knowledge
111:Full Tuning
118:Full Tuning
130:training data, whereas full-tuning relies solely on the training data as its
134:PETL (upper) fine-tunes a small parameters and keeps CLIP backbone frozen,
136:to the retained parameters of original CLIP and TPR-specific knowledge
138:knowledge compared to full-tuning if rationally designed. (b) On CUHKPEDES, our approach reduces 95.1% training parameters and gains an
139:improvement by 5.98% on R@1 compared to the full-tuning CLIP.
167:frameworks [15]–[18]. However, this full-tuning paradigm
175:Adapter [20], LoRA [21], Prefix [22], Prompt [23] and other
177:superior performance only with fine-tuning a few parameters of large models. As shown in Figure 1(a), full-tuning
182:by PETL-related components, TPR-specific knowledge within
183:CLIP is effectively transferred as the parameters of the original
189:it superior to full-tuning in terms of knowledge retention,
194:performance only with fine-tuning 7.4% parameters of CLIP.
204:component conflict, where the performance drops significantly
205:in practice when a single PETL component is incorporated into
207:multiple PETL components overlap and interact, potentially
213:components do not interfere with each other and can work
215:vanilla prefix component is adapted to TPR, it shows a poor
218:whether we could optimize the vanilla PETL components for
225:unify three submodules based on Prefix, LoRA and Adapter
226:to adapt to TPR task. LoRA is incorporated to modify the
229:in TPR that the original CLIP cannot fully uncover. Prefix
230:is prepended to the keys and values of MHA to enrich taskspecific information for TPR. Adapter is designed in layer normalization (layernorm) to adjust the overall distribution from
232:submodules: Salable Prefix (S-Prefix) and Layernorm Adapter
233:(L-Adapter). S-Prefix is introduced to enhance gradient backward propagation of prefix embeddings, which improves both
234:the flexibility and performance of the vanilla prefix. L-Adapter
236:connection, to get rid of component conflicts. Our method
248:• To optimize the PETL components for better synergy,
249:we further develop two improved PETL components,
250:respectively: (1) a scalable prefix in attention named SPrefix, and (2) a novel adapter L-Adapter designed in
254:4.7% parameters.
277:structure, which have less parameters compared with dualstream models. In addition to VLP models that solely employ
306:language models like BERT [12] with 110 million parameters
307:to Yi-VL-34B [35] with 34 billion parameters and llama 3 [36]
308:with 70 billion parameters, full fine-tuning will be more and
314:of full-tuning, which can be broadly categorized into three
317:as Adapter [37], Prompt [23] and Prefix [22] were initially
319:specific downstream tasks by inserting additional parameters to
320:models. LoRA [21] as a representative method for reparameterized fine-tuning methods, utilizes low-rank decomposition
322:to reduce the number of fine-tuned parameters by selecting a
323:subset of pre-trained parameters, such as Bitfit [38].
325:VPT [39] and AdapterFormer [40] have emerged to address
333:parameters and training time while achieving comparable
338:while fine-tuning fewer parameters. We focus on three most
339:representative PETL approaches, including Adapter, LoRA
340:and Prefix.
385:PETL-based paradigm on CLIP, only training 12M parameters
386:while outperforming the performance of full-tuning CLIP. In
401:number of learnable parameters. Several state-of-the-art PETL
403:Adapter. Adapter [37] inserts small modules into transformer layers, which typically employs a down-projection
425:LoRA. LoRA [21] incorporates low-rank trainable matrices into transformer layers with the aim of providing an
427:matrix W ∈ Rd×k , LoRA can update it with a low-rank
434:For the input x to a linear projection y = xW , LoRA alters
438:that LoRA as a reparameterized method, can merge the weight
441:Prefix and Prompt Tuning. They incorporate tunable
455:Prefix Tuning (Pre-T) [22] prepends two prefix tokens
472:In this section, we illustrate each component of our method
476:present the proposed submodules S-Prefix and L-Adapter in
490:L-Adapter
494:LoRA
500:L-Adapter
516:L-Adapter
533:S-Prefix
541:L-Adapter
565:LoRA
572:LoRA
574:Prefix
615:Prefix Part
627:L-Adapter
631:Prefix
642:S-Prefix
647:based on CLIP, two PETL modules for both encoders, and one parameter-free loss function constraint SDM as optimization objective. Only a few parameters
649:transformer block for both image and text encoders. In addition to prefix tokens in the keys and values of MHA, S-Prefix proposes a Sp factor in attention
650:calculator to enhance gradient propagation of prefix tokens. L-Adapter is proposed in two normalization layers to adjust the overall distribution and avoid
651:submodule conflicts. LoRA is inserted to update the weights of keys and values. Overall, L-Adapter helps transfer global pedestrian features, while LoRA
652:and S-Prefix, working together in MHA, focus on attention to promote local knowledge transferring for TPR. All blocks with dashed borderlines represent
653:the fine-tuned modules. On the far right are the more specific implementation details of our L-Adapter and S-Prefix.
697:in the transformer block. Specifically, Adapter [20] is inserted
698:through a residual connection to adapt the output information of MLP and MHA. LoRA [21] incorporates low-rank
701:Prefix Tuning [22] operates at the forefront of the attention
718:and optimizes multiple lightweight PETL methods: Prefix,
719:LoRA and Adapter within image and text encoders. (1)
720:Prefix is optimized to Salable Prefix (S-Prefix) to enhance
722:person retrieval task. S-Prefix concatenates key and value in
724:modules of key and value related to MHA to learn more inherently local information about TPR. (3) Layernorm Adapter
725:(L-Adapter) is proposed to adjust the overall distribution of
730:of each module and why the combination of Prefix, LoRA,
731:and L-Adapter works effectively together in TPR task.
732:First, we analyze LoRA component for TPR:
734:we further introduce two our novel submodules: S-Prefix
735:and L-Adapter.
786:Next, the embeddings of Prefix Pk and Pv can be concatenated on K and V to injecting task-specific information
788:on task-relevant information. The attention with Prefix can be
816:Finally, Adapter component can be utilized to adjust the
830:The above PETL submodules do not conflict in terms of
832:components work synergistically to enhance performance.
833:Therefore, by introducing these additional components to
838:C. S-Prefix
840:Prefix is more flexible and efficient when attached to multiple
842:to the study [58], [63], Prefix can be represented as a form
871:Prefix (S-Prefix) to enhance gradient propagation of prefix
880:mechanism, QK T ∈ RN ×(N +L) . S-Prefix separates attention
886:S-Prefix is equivalent to inserting Sp to Equation (12), finally
935:depicted in Figure 4(d), Layernorm Adapter (L-Adapter) is
940:module conflicts compared with the vanilla parallel adapter. It
942:h ← LayerNorm (x) + s · Adapter (x) .
958:Prefix Tokens
968:Fig. 3. Illustration of S-Prefix. We utilize Sp to denote the salable
969:factor about attention of prefix to accelerate the convergence rate. S-Prefix
972:D. L-Adapter.
976:PETL submodules such as S-Prefix and LoRA are designed
981:component conflicts. Unlike them, in this paper, we focus
982:on Layer Normalization to get rid of causing conflicts from
1070:Adapter
1087:Adapter
1110:Adapter
1121:Adapter
1136:(a) Sequential Adapter
1138:(b) Sequential L-Adapter
1140:(c) Parallel Adapter
1142:(d) Parallel L-Adapter
1146:Fig. 4. Illustration of L-Adapter. (a) Sequential Adapter is connected behind MLP or MHA. (b) Sequential L-Adapter is connected behind layernorm. (c)
1147:Parallel Adapter always spans layernorm and MLP or MHA which contains other PETL submodules. (d) Parallel L-Adapter is inserted into layernorm with
1161:including S-Prefix, LoRA and L-Adapter. The image is resized
1169:layers, incorporates S-Prefix, L-Adapter, and LoRA components as illustrated in our framework. Specifically, S-Prefix
1170:and LoRA are integrated into MHA. Meanwhile, L-Adapter
1172:The bottleneck dimension b of the L-Adapter is set to 8 across
1174:rank of LoRA r is set to 32, while for RSTPReid, due to its
1176:Regarding the length of S-Prefix l, the values are set to 10 for
1195:respectively while only fine-tuning a few parameters. Moreover, we demonstrate that full-tuning method IRRA has a
1200:with the current best-performing PETL-based method DMAdapter, UP-Person achieves a significant improvement in
1201:terms of all metrics. It outperforms DM-Adapter by 2.00%,
1202:2.38% and 3.15% on R@1 across three datasets with less finetuned parameters. Furthermore, with the CLIP-Large model
1207:Figure 5 depicts the trade-off between fine-tuned parameters
1208:and retrieval performance on CUHK-PEDES among CLIPbased methods, where UP-Person achieves optimal performance with the minimum number of parameters. Our approach
1209:only stores 7.4M extra parameters in one scenario under the
1223:DM-Adapter
1242:Fig. 5. R@1 and parameters of different CLIP-based methods on CUHKPEDES.The horizontal coordinate denotes the number of fine-tuned parameters. The gray numbers and the radius of the circles both represent the entire
1249:model parameters, indicating lower computational complexity.
1251:in UP-Person, LoRA modifies the inner weights directly and
1253:While the inclusion of S-Prefix and L-Adapter components
1266:B. Ablation Study
1267:Ablations on PETL components. We analyze the effectiveness and contribution of each PETL module and synergetic
1270:Individually, LoRA, L-Adapter, and S-Prefix outperform
1274:S-Prefix (63.69% R@1) improves attention by adding contextspecific embeddings, surpassing the prompt-based submodule
1275:of CSKT (62.82% R@1) by 0.86%, validating its effectiveness. LoRA (72.56% R@1) adjusts attention weights to highlight subtle person features. L-Adapter optimizes the global
1281:Hence, L-Adapter enhances global knowledge transferring,
1282:LoRA refines attention to capture local dependencies and
1283:features, and S-Prefix strengthens the prior relevant knowledge
1285:To thoroughly validate the synergistic effectiveness of different submodules, we first integrate LoRA and L-Adapter
1289:mAP. Moreover, it is observed that the combination of SPrefix and L-Adapter (No.6) works better on critical metrics
1291:effectiveness of individual component. Integrating LoRA and
1292:S-Prefix also achieves superior performance on all metrics
1293:compared to individual component, demonstrating that these
1296:Finally, the combination of the three submodules with LAdapter, S-Prefix and LoRA (No.7) surpasses other combinations on the most important retrieval metrics including R@1
1298:features through L-Adapter, emphasizes fine-grained features
1299:in attention via S-Prefix and LoRA, and achieves effective
1300:synergistic functions within our three components.
1304:three submodules shows no significant conflicts. Thus, their
1306:rather than single component across multiple metrics.
1307:Hyper-parameters. As shown in Figure 6, to optimize
1309:hyper-parameters based on dataset characteristics, where a
1311:bottleneck reduction b of L-Adapter is consistently set to 8
1312:across all three TPR datasets to balance performance and efficiency. For LoRA, its rank r is adjusted according to dataset
1317:maintaining effective adaptation. Similarly, the length of SPrefix l is tailored to dataset-specific needs: CUHK-PEDES
1322:Ablations on LoRA and its variants. As shown in Tablse
1323:VI, we further analyze the ablation experiment on LoRA
1326:LoRA component outperforms those with DoRA and WoRA in
1327:overall metrics, both of which introduce additional parameters
1328:compared to LoRA. Therefore, we ultimately choose LoRA
1486:DM-Adapter [73]
1570:Fig. 6. The analysis of hyper-parameters at R@1 on CUHK-PEDES, ICFG-PEDES and RSTPReid.
1572:C. Analysis of S-Prefix.
1573:Table VII demonstrates the effectiveness of S-Prefix compared to vanilla prefix based on the CLIP backbone and
1574:UP-Person framework. First, we take CLIP (S-Prefix) as an
1577:to CLIP (Vanilla Prefix), CLIP (S-Prefix) gains a dramatic
1653:DM-Adapter [73]
1745:DM-Adapter [73]
1787:L-Adapter significantly surpass two other vanilla adapters,
1788:which denotes that layernorm is extremely important in finetuning to TPR due to scale and shift parameters. Compared
1789:to LN-tuning, parallel L-Adapter gains an overwhelming improvement by +1.64% on R@1. We speculate that unfreezing
1791:Analysis on conflict mitigation. As shown in Table IX, we
1796:MHA L-Adapter and MLP L-Adapter (No.1) outperforms all
1799:slightly worse than the L-Adapter configurations, with lower
1801:This indicates that the presence of an L-Adapter, particularly
1804:also underperform to our full L-Adapter configuration, further
1805:suggesting that L-Adapter is better suited for the unified structure. We speculate that the vanilla adapter, spanning multiple
1806:network layers (MHA or MLP), introduces conflicts with other
1807:PETL components in the optimization space, whereas the LAdapter, which focuses solely on adjusting the global distribution through LayerNorm, avoids these spatial conflicts without
1816:converge. In the unified framework which incorporates LoRA
1817:and L-Adapter, UP-Person (S-Prefix) has a significant gain
1818:than UP-Person (Vanilla Prefix) by +0.53% on R@1 when
1821:D. Analysis of L-Adapter.
1823:vanilla adapter and our L-Adapter in UP-Person, and further
1825:in Figure 4(a), Sequential Adapter is inserted behind MLP
1827:Sequential L-Adapter is connected behind layernorm in series.
1828:Similarly, parallel adapters span specific network components
1832:parameters while keeping other parameters frozen.
1834:is far lower than that of parallel L-Adapter (-1.17% on R@1).
1848:cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
1870:DM-Adapter
1916:S-Prefix LoRA L-Adapter
1929:+S-Prefix
1930:+LoRA
1931:+L-Adapter
1932:+LoRA +L-Adapter
1933:+S-Prefix +LoRA
1934:+S-Prefix +L-Adapter
2004:Prefix Type
2009:CLIP (Vanilla Prefix)
2014:CLIP (S-Prefix)
2020:CLIP (S-Prefix)
2026:CLIP (S-Prefix)
2032:CLIP (S-Prefix)
2038:CLIP (S-Prefix)
2044:CLIP (S-Prefix)
2046:UP-Person (Vanilla Prefix)
2051:UP-Person (S-Prefix)
2057:UP-Person (S-Prefix)
2063:UP-Person (S-Prefix)
2072:Adapter Type
2074:UP-Person (Sequential Adapter)
2075:UP-Person (Sequential L-Adapter)
2076:UP-Person (Parallel Adapter)
2077:UP-Person (Parallel L-Adapter)
2115:Adapter Type
2116:UP-Person (MHA L-Adapter, MLP L-Adapter)
2117:UP-Person (MHA L-Adapter, MLP Adapter)
2118:UP-Person (MHA Adapter, MLP Adapter)
2119:UP-Person (MHA Adapter, MLP L-Adapter)
2162:UP-Person (LoRA [21])
2227:and RSTPReid, respectively. Moreover, it is obvious that UPPerson is superior than IRRA-CLIP (full-tuning) in all metrics
2229:We conjecture that our method effectively alleviate overfitting compared to full-tuning-based methods since it merely
2230:fine-tunes a few parameters of simple PETL components. This
2234:V. Thus, full-tuning-based methods such as IRRA and IRRACLIP can cause the more severe overfitting when lacking of
2237:G. Visualization
2248:infer that the full-tuning method IRRA, which only utilizes
2249:pre-trained parameters at initialization process, may lose part
2253:“unseen” objects may fail. Overall, the visualization vividly
2404:scale, UP-Person with only 7.4M parameters outperforms
2405:BEIT-3 which has 1.9B trainable parameters on all metrics
2432:parameters by a large margin, such as IMEB [3] and DSMD
2445:across multiple pedestrian retrieval scenarios. Instead of storing a separate large foundation model for each specific scenario, UP-Person only occupies a few parameters tailored for
2507:within CLIP without bells and whistles. This unified framework adjusts global features with our L-Adapter while capturing fine-grained features in attention with out optimized SPrefix and LoRA. The significant performance improvements
2617:[22] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | rg -n \"Ablation|ablation|CDM|MIDM|FCRS|Adversarial|Visualization|visualization|t-SNE|decoupling|camera|modality|recombination\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12:in different modalities. Since these images are taken from varying viewpoints by different cameras, the
13:cross-modal detection task must address both modality discrepancies and camera variations. Many existing
14:approaches primarily focus on minimizing inter-modality differences to enhance retrieval accuracy, often
15:overlooking the impact of camera viewpoint differences. To tackle these challenges, this article introduces
16:a hierarchical feature decoupling network. First, the network decouples and extracts camera-related and
17:camera-irrelated features separately to mitigate the effects of camera variations. Second, it addresses modality
18:differences by extracting modality-independent features. Additionally, an adversarial decoupling loss is
23:Additional Key Words and Phrases: Person re-identification, modality discrepancies, feature decoupling
56:by multiple non-overlapping cameras. This technology has significant practical value in surveillance
59:identification, these methods are limited in low-light conditions as visible light cameras struggle to
61:With the advancement of surveillance technology, an increasing number of new cameras are
64:this emerging technological demand, Wu et al. [40] pioneer the study of cross-modality person
68:modality gap between visible light and infrared images, which makes it difficult to effectively
72:the modality gap between visible light and infrared images becomes the key to improving the
73:accuracy and robustness of cross-modality person re-identification.
76:identity learning. In contrast, camera-irrelated features robustly capture identity information and
77:remain stable in cross cameras and modalities. Additionally, due to the differences in imaging principles between visible and infrared spectra, cross-modality person re-identification must contend
78:not only with feature variations within a single modality but also with cross-modality feature
82:single decoupling strategy to address the effects of modal differences, to achieve identity-based
83:matching and similarity ranking. However, in the modality-shared feature space, identity information is often intertwined with camera noise and modality-sensitive redundancy, resulting in the
84:weakening of discriminative features. This means that a single-stage decoupling strategy may find
85:it difficult to simultaneously mitigate the impact of modality and camera differences, resulting in a
86:decrease in the discriminative ability of modality-shared features, thereby affecting the performance
88:To address the above challenges, this study proposes an innovative cross-modality person
90:two main modules: Camera Decoupling Module (CDM) and Modality-Invariant Identity
91:Decoupling Module (MIDM). This framework aims to progressively decouple camera difference
98:features, modality difference features, and identity-irrelevant redundant features, and significantly
99:enhance the discriminative performance of cross-modality person re-identification through this
102:hierarchical feature decoupling approach. CDM decouples camera-related features and camerairrelated features from modality-shared features to mitigate the impact of camera variations. This
104:minimize the identity information in camera-related features, allowing the network to focus more
105:on identity-specific features. Then, the MIDM further decouples identity-dependent and identityindependent features from camera-irrelated features, using residual features to recover the lost
106:identity information. In MIDM, an adversarial decoupling loss is designed to optimize the feature
107:decoupling process and promote a better separation of identity features and redundant features. This
110:And, a Feature Cross-Modal Recombination Strategy (FCRS) is introduced to further reduce
111:the modality differences of identity features, which guides the network to focus on the shared
114:— HDNet, which includes CDM and MIDM, is an innovative end-to-end feature purification
115:framework that utilizes a multi-level decoupling strategy (camera-modality-identity), ensuring
118:— CDM decouples camera-related and camera-irrelated features from modality-shared features
120:in camera-related features, thereby allowing the network to focus more on identity-specific
122:— MIDM utilizes adversarial decoupling loss to optimize identity disentanglement information and introduces the FCRS to focus on the shared identity information by pairing and
129:multiple surveillance cameras [57]. The main challenges arise from variations in camera environments, pose changes, and viewpoint differences. Traditional person re-identification methods
130:are based on visible light cameras and can be divided into two categories [15, 29, 45, 56]. One
142:person re-identification, thus requiring additional methods to handle modality-specific information.
153:by zero-padding, capturing modality-specific information. Ye et al. [48] propose a dual-stream
155:modality-shared features. Zhuang et al. [61] design camera-based batch normalization to eliminate
156:feature distribution differences caused by different cameras. Wang et al. [36] propose an attentive
157:modality-consistent network that enhances feature learning through a residual attention module.
159:images, reducing modality discrepancy, with Distribution Consistency Loss ensuring distribution consistency. Ye et al. [47] introduce a channel-enhanced joint learning strategy to improve
161:Network that leverages adversarial training and adaptive dynamic weights to reduce cross-modality
164:cross-modality person re-identification by applying similarity constraints at both the class and
165:modality levels. However, most existing methods fail to fully consider the impact of camera differences. Although current approaches employ shared networks to learn modality-shared features,
166:they struggle to simultaneously mitigate the effects of both modality and camera differences,
167:leading to a reduction in the discriminative power of modality-shared features and, consequently,
168:affecting the performance of cross-modality person re-identification.
175:et al. [16] propose a spectral disentanglement method that separates modality-specific spectral
177:and Hu and Hu [14] employ dual-adversarial learning to disentangle modality-specific features
181:modality gap in visible-infrared person re-identification by separating identity-relevant and identityirrelevant information.
182:Existing decoupling methods do not fully consider the complex relationships between modalityshared features and other redundant features, which are not simply linearly combined. As a result,
183:relying on a single decoupling strategy to simultaneously address modality and camera differences
185:framework that employs a multi-level decoupling strategy (camera-modality-identity). This ensures
186:that identity information is preserved as much as possible at each level while effectively removing interference, thus avoiding the incomplete or overly simplified decoupling issues present in
193:to extract image features. The dual-stream network splits ResNet50 into modality-specific layers
202:CDM network into camera-related features 𝐹𝑐𝑟 and camera-irrelated features 𝐹𝑐𝑖 . A GRL minimizes identity
206:Adversarial decoupling loss 𝐿𝑎𝑑𝑣 and recombination center loss 𝐿𝑟 further reduce cross-modality differences
209:(Block1, Block2) and modality-shared layers (Block3, Block4, Block5). For a given pedestrian image
213:The obtained feature map is input into the CDM network, where camera-related features 𝐹𝑐𝑟
214:and camera-irrelated 𝐹𝑐𝑖 features are decoupled. In CDM, camera-related features, 𝐹𝑐𝑟 , are those
215:that can correctly classify different cameras and do not contain identity information. 𝐹𝑐𝑖 called
216:camera-irrelated features are those that are related to identity information after the initial extraction
217:of shared features, and the camera differences are removed. Subsequently, 𝐹𝑐𝑖 is further refined
218:through the following network. MIDM first performs Instance Normalization (IN) to obtain normalized features 𝐹e𝑐𝑖 . Although IN alleviates modality discrepancies, it may filter out some identity
220:features 𝐹𝑐𝑖 and the IN features 𝐹e𝑐𝑖 are used as the decoupling target. From these residual features
223:with 𝐹e𝑐𝑖 to obtain identity features 𝐹𝑖𝑑 and redundant features 𝐹𝑛 . An adversarial decoupling loss
224:𝐿𝑎𝑑𝑣 is designed to optimize the effectiveness of feature decoupling. The positive residual feature
225:from the residual feature contains visible modality and infrared modality. In order to further reduce
226:the differences between cross-modality features under the same modality, the positive residual
227:feature 𝑅 + and the normalized feature 𝐹e𝑐𝑖 are first split by modality into 𝐹e𝑐𝑖𝑣 , 𝐹e𝑐𝑖𝑡 , 𝑅𝑣+ , and 𝑅𝑡+ ,
228:followed by exchange and recombination. FCRS exchanges the positive residual features 𝑅𝑣+ and
237:modality combined with the corresponding split features are referred to as same-modality recombinant features. The two recombinant features are optimized using the recombinant center loss 𝐿𝑟 ,
239:deeply and further reduce modality differences. Additionally, 𝐿𝑏𝑎𝑠𝑒 represents the commonly used
241:3.1 CDM
242:We are inspired by SE-Net [13] and hope to use channel masks 𝑀 to decouple features 𝐹 and adaptively separate camera-related features 𝐹𝑐𝑟 and camera-irrelated information 𝐹𝑐𝑖 . From Equations
260:In order to constrain the mask to extract the features we need, we leverage camera information
261:labels to guide the feature learning, ensuring a stronger focus on camera-related classification.
264:𝐶𝑐𝑎𝑚 and Identity Classifier 𝐶𝑐𝑖𝑑 to correctly predict the camera labels 𝑦𝑐𝑎𝑚 and identity labels 𝑦𝐼 𝐷 ,
287:identity information as possible in the camera-related feature 𝐹𝑐𝑟 , second, we apply an identity
299:Thus, the total camera decoupling loss is:
305:MIDM
307:To further alleviate modality discrepancies, we apply IN to the camera-irrelated features 𝐹𝑐𝑖 . IN can
325:Fig. 2. The optimization objective of the adversarial decoupling loss. When 𝐹˜𝑐𝑖 is recombined with 𝑅 + , it
335:the residual features of 𝐹𝑐𝑖 as the decoupling target to recover the lost identity information. The
341:Following the decoupling method in Section 3.1, the residual features are split into positive
352:from the camera-irrelated features 𝐹˜𝑐𝑖 , respectively.
355:Adversarial Decoupling Loss
358:an adversarial decoupling loss. Figure 2 illustrates the optimization goal of this loss. When 𝐹˜𝑐𝑖 is
360:contrast, when 𝐹˜𝑐𝑖 is recombined with 𝑅 − , the identity information mixes with modality-irrelevant
404:The final adversarial decoupling loss can be expressed as:
409:In summary, the feature decoupling module not only decouples identity-discriminative features
410:with stronger robustness, reducing the degree of interference from identity-unrelated information during training, but also alleviates the modality discrepancies between visible and infrared
416:To further reduce the impact of modality differences and guide the network to focus on shared
417:features, we decompose 𝐹e𝑐𝑖 and 𝑅 + by modality into 𝐹˜𝑐𝑖𝑣 , 𝐹˜𝑐𝑖𝑡 , 𝑅𝑣+ , and 𝑅𝑡+ , and then cross recombine
432:Here, 𝐹˜𝑣_𝑣 and 𝐹˜𝑡 _𝑡 represent the same-modality features, and Equation (10) 𝐹𝑖𝑑 is composed of
433:these two features, while 𝐹˜𝑣_𝑡 and 𝐹˜𝑡 _𝑣 represent the cross-modality identity features. Subscripts
434:𝑣 and 𝑡 denote visible and infrared modalities, respectively. In the process of cross-modality
435:recombination, we adopt a random combination approach. For the same identity, by pairing the
437:modalities. This method avoids the strict requirement of recombining the same modality features,
439:being limited to local or narrow recombination.
440:In the cross-modality recombination process, a random combination strategy is employed.
442:the model is not restricted to recombining only features from the same modality (e.g., visible-visible
443:or thermal-thermal), but rather allows for cross-modality pairings (e.g., visible-thermal).
444:To further optimize decoupling loss, we apply metric learning to the four recombined features
481:while minimizing the modality discrepancy in the recombined features.
499:— SYSU-MM01 is a large dataset captured by four visible light cameras and two near-infrared
500:cameras, covering indoor and outdoor environments. The training set consists of 395 pedestrian
505:— RegDB is a dataset captured by one visible light camera and one thermal infrared camera. It
525:Table 1. The Ablation Experiment Results of Our Method on the SYSU-MM01 Dataset (%)
611:Table 2. The Ablation Experiment Results of Our Method on the RegDB Dataset (%)
707:Ablation Experiments
709:To evaluate the effectiveness of the proposed method, ablation experiments are conducted on two
714:where 𝑀𝐼𝐷𝑀 ∗ denotes the 𝑀𝐼𝐷𝑀 without using the adversarial decoupling loss and FCRS, and
717:Comparing the feature decoupling module with and without the adversarial decoupling loss,
718:the model’s Rank values and mAP showed significant improvements when the adversarial decoupling loss was applied. This indicates that relying solely on the channel masking mentioned in
719:Section 3.3 makes it difficult to achieve effective decoupling of residual features, and it may even
727:Fig. 3. Heatmap visualization of decoupled features.
729:adversarial decoupling loss plays in the decoupling process, effectively promoting the separation of
731:decoupling approach proposed in Section 4.2.
733:Heatmap Visualization. To validate the decoupling effect of the feature decoupling method, we
738:indicating that although 𝐹e𝑐𝑖 can alleviate modality differences, it also loses a considerable amount
742:human body region, demonstrating that the recombination of 𝐹e𝑐𝑖 with the positive residual feature
743:𝑅 + can capture more discriminative identity information. Therefore, the heatmap visualization
744:experiment convincingly illustrates that our feature decoupling method effectively guides the model
745:to achieve a good decoupling effect between identity features and redundant features in pedestrian
747:Feature Distribution Visualization. To further analyze the role of the adversarial decoupling loss,
748:we randomly selected 10 identities and employed t-SNE [32] to visualize the feature distributions
751:overlap between different classes and unsatisfactory intra-class compactness and inter-class separability. The 𝐹𝑐𝑖 figure shows the feature distribution after CDM, where the camera-irrelated feature
752:decoupling effectively reduces the differences between different modalities, resulting in better
754:is significantly improved. The 𝐹𝑖𝑑 figure shows the distribution after MIDM, where the identity
755:feature decoupling further enhances intra-class compactness and inter-class separability, with each
763:Fig. 4. Visualization of feature distribution by t-SNE. Where (a) represents the original feature distribution
764:from pretrained ResNet50; (b) represents the feature distribution visualization after passing through the AGW
765:[48] baseline; (c) is the feature distribution after CDM; (d) is the feature distribution after MIDM.
767:Overall, as the multi-level feature decoupling progresses, the camera-irrelated feature decoupling
769:feature decoupling further enhances intra-class compactness and inter-class separability, achieving
776:on SYSU-MM01. Among them, SDL [16] proposed a non-adversarial feature decoupling method,
777:Hi-CMD [6], JSIA [34], TSME [21] also draw on the decoupling idea to deal with the modal style
778:information, in which the cross-modal paired images had to be acquired by Generate Adversarial
780:volatility, making training more difficult. In comparison to the aforementioned decoupling-based
791:In this article, we propose a hierarchical feature decoupling network, called HDNet. Our approach
792:aims to alleviate camera and modality discrepancies while reducing the interference of redundant
794:constructs a multi-level feature decoupling module to decouple camera-related information, identity
795:features, and redundant features. An adversarial decoupling loss is then employed to promote
907:better feature separation. Additionally, a cross-modality feature recombination strategy is used
908:to mitigate the impact of modality differences on the decoupled identity features. Experimental
930:[6] Seokeon Choi, Sumin Lee, Youngeun Kim, Taekyung Kim, and Changick Kim. 2020. Hi-CMD: Hierarchical crossmodality disentanglement for visible-infrared person re-identification. In Proceedings of the IEEE/CVF Conference on
937:[10] Soonyong Gwon, Sejun Kim, and Kisung Seo. 2024. Balanced and essential modality-specific and modality-shared
947:[15] Houjing Huang, Dangwei Li, Zhang Zhang, Xiaotang Chen, and Kaiqi Huang. 2018. Adversarially occluded samples for
959:[20] Jialun Liu, Yifan Sun, Feng Zhu, Hongbin Pei, Yi Yang, and Wenhui Li. 2022. Learning memory-augmented unidirectional metrics for cross-modality person re-identification. In Proceedings of the IEEE/CVF Conference on Computer
961:[21] Jianan Liu, Jialiang Wang, Nianchang Huang, Qiang Zhang, and Jungong Han. 2022. Revisiting modality-specific
964:[22] Zefeng Lu, Ronghao Lin, and Haifeng Hu. 2024. Tri-level modality-information disentanglement for visible-infrared
967:images from visible light and thermal cameras. Sensors 17, 3 (2017), 605.
992:[32] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of Machine Learning
995:modality-aware multiple granularity pre-training for RGB-infrared person re-identification. IEEE Transactions on
998:Cross-modality paired-images generation for RGB-infrared person re-identification. In Proceedings of the 34th AAAI
1003:[36] Hanzheng Wang, Jiaqi Zhao, Yong Zhou, Rui Yao, Ying Chen, and Silin Chen. 2021. AMC-Net: Attentive modalityconsistent network for visible-infrared person re-identification. Neurocomput 463, C (2021), 226–236.
1007:[38] Ziyu Wei, Xi Yang, Nannan Wang, and Xinbo Gao. 2021. Syncretic modality collaborative learning for visible infrared
1011:[40] Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. 2017. RGB-infrared cross-modality
1016:2021. Discover cross-modality nuances for visible-infrared person re-identification. In Proceedings of the IEEE/CVF
1018:[43] Song Wu, Shihao Shan, Guoqiang Xiao, Michael S. Lew, and Xinbo Gao. 2025. Implicit modality knowledge alignment
1034:Wang. 2022. Hybrid modality metric learning for visible-infrared person re-identification. ACM Transactions on
1036:[50] Qiang Zhang, Changzhou Lai, Jianan Liu, Nianchang Huang, and Jungong Han. 2022. Fmcnet: Feature-level modality
1042:difference: Cross-modality person re-identification via contrastive correlation. IEEE Transactions on Image Processing
1050:[53] Yukang Zhang and Hanzi Wang. 2023. Diverse embedding expansion network and low-light cross-modality benchmark
1053:[54] Yukang Zhang, Yan Yan, Yang Lu, and Hanzi Wang. 2021. Towards a unified middle modality learning for visibleinfrared person re-identification. In Proceedings of the 29th ACM International Conference on Multimedia. Association

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | rg -n \"Ablation|ablation|RHL|Hybrid|GCE|clean|indistinguishable|obvious|real-world|SYSU-Real|noise|robust|visualization|distribution\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
14:noise, an issue that becomes even more pronounced in crossmodal scenarios due to an increased likelihood of data confusion.
16:often overlook the complexities of instance-dependent and realworld noise, creating a disconnect from the practical applications
18:analyzes the primary sources of label noise in real-world settings,
21:we develop a Robust Hybrid Loss function (RHL) that enables
24:categorises data into three sets: clean, obviously noisy, and
29:method, we have re-annotated a real-world dataset, SYSU-Real.
31:various noise ratios of random and instance-dependent label
32:noise, demonstrate the generalized robustness and effectiveness
34:Index Terms— Person re-identification, visible infrared, crossmodality, instance-dependent label noise, robust deep learning.
78:practicality in real-world scenarios. However, visible-infrared
88:In the real world, label noise is a common issue [25], [26],
111:VI-ReID, it primarily focuses on random noise, such as
112:arbitrarily switching image labels to incorrect ones. This noise
116:and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
125:generation, and real-world annotation. Circles represent visible images, and triangles represent infrared images. The outline of each instance is its potential
128:forms of noise encountered in real-world scenarios, as outlined
130:noise (IDN), which varies with each instance and reflects
132:of label noise than the random noise used in the existing
138:capture the unpredictable and complex nature of real-world
139:label noise in practical applications of VI-ReID.
141:primary reasons for the discrepancy between IDN and realworld noise: Firstly, as an algorithmically synthesized noise,
143:in the real-world scenarios depicted in Fig. 1. Specifically,
145:inattention. Secondly, the robustness of VI-ReID algorithms
146:still requires real-world labelling for corroboration and cannot
148:on synthetic algorithms for noise simulation might lead
150:To establish a more robust foundation for validating VIReID methods against real-world label noise, we have
152:version called SYSU-Real that includes genuine real-world
155:noise above, we introduce a robust hybrid loss (RHL)
156:framework that categorizes data into three segments: clean,
157:indistinguishably noisy, and obviously noisy. For the clean
166:Cross-Entropy (GCE) loss function [31], which aims to
168:In the third segment, where noise is evident, we discard the
174:process’s robustness.
177:of real-world label noise in VI-ReID. To overcome
178:the challenges of obtaining real-world labeled datasets,
180:noise generation algorithm. Secondly, we manually
182:human-annotated SYSU-Real dataset. This approach
183:significantly enhances the realism of label noise, yet it
186:• We introduce a Robust Hybrid Loss (RHL) function
187:that effectively manages instance-dependent label noise
188:and real-world noise. This function utilizes a threecomponent mixture, allowing distinct loss calculation
189:strategies for different noise subsets, thereby enhancing
190:the method’s robustness and adaptability.
192:datasets, incorporating both random and instancedependent label noise. Additionally, we evaluated
193:real-world noise handling on the relabeled SYSU-
200:effectiveness and robustness of our proposed method.
204:and noise label learning that are pertinent to this paper.
252:Label noise has emerged as a significant challenge in the
257:which can inadvertently introduce label noise.
264:[67], [68] to mitigate the impact of label noise, and others
265:design robust loss functions [31], [69], [70], [71], [72] to
268:noise, they have predominantly been applied to classification
270:The study of label noise in multi-modality domains,
273:instance, Huang et al. [73] introduced the concept of noise
279:of noise correspondence between modalities, leading to
282:noise-robustness.
284:discussion of label noise to encompass both infrared and
286:integrates elements of sample selection, a robust framework,
287:and a robust loss function, tailored to the unique challenges
291:Label noise significantly impacts person re-identification
297:emerging studies have focused on label noise in text-to-image
299:focus on symmetric or random noise, overlooking the
300:complexity of real-world noise. Recognizing this gap, our
301:paper concentrates on real-world noisy labels. We undertake
303:dataset to better reflect the challenges posed by real-world
314:feature extraction, and the results and loss values are used to divide subsets to ultimately obtain a robust hybrid loss. It is noteworthy that the dashed line
320:of VI-ReID with instance-dependent label noise in III-A.
324:samples into clean and noisy subsets. Ultimately, we provide
325:a hybrid loss function to achieve a robust training process.
338:random noise, instance-dependent label noise, or real-world
339:noise, and we get different noisy labels through different
340:generation pathways. More generation details of noise labels
372:model. In view of previous findings, the structure of cotraining helps to enhance the robustness of the model. This
375:generalisation and reducing the effect of noise annotations.
380:will be employed in Section III-C to discriminate the clean and
384:this foundation, robust hybrid loss is calculated, and it is
385:utilized to make the model robust to re-identification training.
402:be divided into two steps: coarse-grained selection for clean
405:noise, it is difficult for the model to directly point to clean
408:it was realised that the model fits clean data first rather than
410:by modelling per-sample loss value distribution a viable
412:considers samples that have small loss values as clean samples
413:and vice versa as noise samples. According to the warmed-up
441:find the obvious noisy label set S O N with a threshold ε1 .
447:component with the lower mean as the clean one. To divide the
448:clean set S C and the indistinguishable noise set S I N , we make
467:confidence. Thus, the following clean correspond dataset Dc
513:fit the distribution of per-sample losses as the prior methods:
520:is the clean probability of i-th sample. For simplicity, we set
523:triplets using the attributive pseudo-labels of samples to clean
531:D. Robust Hybrid Loss
532:To classify different identities, we prompt a robust hybrid
538:a) Obvious noisy set: The obvious noise set is caused
581:obvious
582:noise
590:allow obvious noisy labels to be correctly optimized
599:b) Clean set: For the clean subset, we are confident
616:contains the most indistinguishable samples, i.e., the set
617:mixes cleanly labelled samples and noisily labelled samples.
618:In this section, the set contains the most indistinguishable
619:samples, i.e., the set mixes cleanly labelled samples and noisily
622:robust loss function.
624:we employ GCE [31] as the indistinguishable noisy loss L I N
640:To summarize, our robust identification loss consists of three
652:robust retrieval loss we employ is a variant of the triplet loss,
724:Fig. 3. Label distribution of the re-labelled dataset SYSU-Real. Different
726:proportion of noise within the infrared and visible subsets. Note that only
727:two items, visible noise ratio and infrared noise ratio, are proportions to the
731:Ultimately, our robust loss is a hybrid of the person
744:SYSU-MM01 called SYSU-Real.
750:the robustness and adaptability of our method across varied
761:Since both the random noise and the instance-dependent
762:label noise are generated by code, we additionally upset
764:dataset with real-world labelled noise we call SYSU-Real. For
785:label distribution of this relabelled dataset is shown in Fig. 3.
786:The total label noise ratio is 23.79%, out of which, 15.34%
788:for infrared data. For the noise ratio within each mode, it is
790:that the proportion of noise is slightly higher in infrared, which
795:For the SYSU-MM01 and SYSU-Real datasets, we utilize
798:sets. In contrast, the RegDB dataset is assessed under Infraredto-visible and visible-to-infrared modes. To ensure robustness,
807:on GeForce RTX 3090 GPUs. The RHL framework is aligned
816:model performance. For the generation of random noise,
822:training set, while the test set labels remain clean to assess the
823:algorithm’s robustness.
846:model’s robustness against hard negative samples, assessing
852:with noise rates of 0.1, 0.3 and 0.5. Models with which
858:the network. (c) ADP [21] improves the robustness of the
873:label noise in the field of VI-ReID.
877:outperforms previous methods on all metrics with all noise
886:the data is more distinguishable and the effect of noise
892:noise rate. However, for retrieval of visible images using
895:and the fact that the instance-dependent label noise is based
896:on the quality of the image instances, the noise brings about
898:of Infrared Images). With the noise rate of 0.5, the model
902:results of our method conducted on the SYSU-Real dataset
915:demonstrating the robustness of our model in the face of noise.
916:4) Results on Random Noise: Given that the label noise in
917:DART is generated randomly, we also present the robustness
918:results under the condition of random noise. As illustrated
930:Fig. 4. Hyper-Parameter Sensitivity Curves. The impact of robustness parameter q, balance parameter β, and threshold parameter ε1 on the main evaluation
935:scenarios of SYSU-MM01, as the noise rate increases from
941:significant challenge. Our method, at a noise rate of 30%,
944:DART. These results unequivocally demonstrate the robustness
947:C. Ablation Study
948:1) Analysis of the Different Components of RHL: As
949:demonstrated in Tab. VI, we conducted an ablation study of
950:various modules in our method on the SYSU-Real dataset. The
954:suppression of label noise by GCE. As shown in the final
958:samples, under the restraint of GCE, do not cause a biased
971:Fig. 5. Histogram of the effect of different types of noise on the degradation
972:of the metrics. We present it using the non-robust SOTA method DSCNet [8]
986:I IMPACTS OF D IFFERENT L OSS F UNCTION C OMBINATIONS ON SYSU-Real
994:of robustness in classification performance for RHL, while
1002:primarily ablate the robustness parameter q, balance parameter
1009:enhances robustness; however, excessively strong robustness
1013:The impact of various types of noise on the learned data
1014:distribution by models varies significantly. Taking the stateof-the-art method DSCNet as an example, we examine the
1015:differential effects of different noise types on performance
1016:metrics under equitably imposed noise rates. Within the
1019:the issue of real-world noise annotation presents a formidable
1020:challenge. As depicted in Fig. 5, all types of label noise
1022:IDN and real-world noise exhibit exacerbated degradation in
1023:comparison to random noise, primarily due to the association
1024:of their noise labels with specific instances, rendering them
1025:more misleading. On the Rank-1 metric, real-world noise
1027:Consequently, the investigation of real-world noise emerges as
1030:demonstrate the noise resistance of our method, we conducted a similarity comparison using the extracted features.
1035:illustrative, we opted to use the difference between RHL and
1048:Heatmap of the difference in similarity matrices between RHL
1052:of RHL and that of the state-of-the-art (SOTA) baseline, namely DART.
1054:performance of our method when dealing with label noise. The matrix
1057:the frame representing inter-class similarities. Notably, RHL significantly
1073:elevation in similarity levels caused by RHL’s enhancement
1074:of overall similarity (a byproduct of its strong robustness
1075:and improved feature extractor capabilities in resisting noise
1085:Embedding visualization on the extracted features. As depicted
1090:subsets; the use of CE loss on clean subsets ensures the speed
1091:of optimisation, while losses on both obviously noisy and
1092:indistinguishably noisy subsets greatly enhance the robustness
1099:and used for t-SNE visualisation with the model trained on SYSU-Real. The
1100:left side shows the results of the SOTA robust model DART and the right
1101:side shows the results of our model RHL.
1109:time compared to non-robust methods for designing complex
1110:networks. Specifically, RHL is only 0.015s slower compared
1111:to the baseline method (DART), and on average, RHL is 0.12s
1112:faster compared to other non-robust SOTA algorithms.
1115:the pivotal issue of label noise, particularly in cross-modal
1116:scenarios. We have innovatively developed the Robust Hybrid
1117:Loss function (RHL), tailored to optimize identification
1118:and retrieval in VI-ReID amidst various noise challenges,
1121:on re-annotated SYSU-Real and tests on SYSU-MM01 and
1122:RegDB datasets, demonstrates remarkable robustness and
1125:for future research in handling label noise in complex, realworld environments.
1180:transformer with distribution alignment for visible-infrared person reidentification,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 11,
1217:scores make instance-dependent label-noise learning possible,” in Proc.
1222:[29] X. Xia et al., “Part-dependent label noise: Towards instance-dependent
1223:label noise,” in Proc. Adv. Neural Inf. Process. Syst., Jan. 2020,
1309:labels more robust,” in Proc. Int. Conf. Mach. Learn. (ICML), 2020,
1322:examples for robust deep learning,” in Proc. Int. Conf. Mach. Learn.,
1342:[67] A. Vahdat, “Toward robustness against label noise in training deep
1358:[71] Y. Xu, P. Cao, Y. Kong, and Y. Wang, “L_DMI: A novel informationtheoretic loss function for training deep nets robust to label noise,”
1362:without knowing noise rates,” in Proc. Int. Conf. Mach. Learn., 2020,
1380:refining for person re-identification with label noise,” IEEE Trans. Image
1382:[79] F. Yang et al., “Joint noise-tolerant learning and meta camera shift adaptation for unsupervised person re-identification,” in

codex
现在方法段已读完，剩下是把“作者怎样证明这个新问题成立”找准。这里关键不是 SOTA 表，而是能反推他们重定义是否成立的消融、特征分布图、热力图和噪声分布图。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1260,1332p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1260	UP-Person ranks among the top two in multiple metrics
  1261	and achieves a favorable trade-off between performance and
  1262	inference efficiency. Therefore, considering metrics such as
  1263	computational efficiency, storage efficiency, model complexity,
  1264	and memory usage, UP-Person exhibits a notable advantage
  1265	over other methods.
  1266	B. Ablation Study
  1267	Ablations on PETL components. We analyze the effectiveness and contribution of each PETL module and synergetic
  1268	effects by conducting a series of ablation experiments on
  1269	CUHK-PEDES in Table V.
  1270	Individually, LoRA, L-Adapter, and S-Prefix outperform
  1271	zero-shot CLIP by a large margin. The TPR task requires
  1272	understanding both global and local relationships between
  1273	textual descriptions and person image features. Specifically,
  1274	S-Prefix (63.69% R@1) improves attention by adding contextspecific embeddings, surpassing the prompt-based submodule
  1275	of CSKT (62.82% R@1) by 0.86%, validating its effectiveness. LoRA (72.56% R@1) adjusts attention weights to highlight subtle person features. L-Adapter optimizes the global
  1276	feature distribution in LayerNorm, achieving superior perfor-
  1277	
  1278	9
  1279	
  1280	mance on the coarse evaluation criterion R@10 with 93.81%.
  1281	Hence, L-Adapter enhances global knowledge transferring,
  1282	LoRA refines attention to capture local dependencies and
  1283	features, and S-Prefix strengthens the prior relevant knowledge
  1284	representation for TPR tasks.
  1285	To thoroughly validate the synergistic effectiveness of different submodules, we first integrate LoRA and L-Adapter
  1286	(No.4), which yields superior performance compared to their
  1287	separate utilization (No.2 and No.3), gaining an improvement
  1288	by 0.92% and 1.16% on comprehensive retrieval criterion
  1289	mAP. Moreover, it is observed that the combination of SPrefix and L-Adapter (No.6) works better on critical metrics
  1290	R@1 and mAP where the combined result surpasses the
  1291	effectiveness of individual component. Integrating LoRA and
  1292	S-Prefix also achieves superior performance on all metrics
  1293	compared to individual component, demonstrating that these
  1294	two submodules can work together to optimize MHA layer,
  1295	enhancing the model’s task-specific information while focusing on fine-grained features for attention.
  1296	Finally, the combination of the three submodules with LAdapter, S-Prefix and LoRA (No.7) surpasses other combinations on the most important retrieval metrics including R@1
  1297	(74.17 %) and mAP (65.91 %), This integration adjusts global
  1298	features through L-Adapter, emphasizes fine-grained features
  1299	in attention via S-Prefix and LoRA, and achieves effective
  1300	synergistic functions within our three components.
  1301	In overall, ablations on the three modules further validate
  1302	the rationality of our architecture, as outlined in the theoretical
  1303	analysis of Section IV-B. The performance optimization of the
  1304	three submodules shows no significant conflicts. Thus, their
  1305	collaborative interaction results in better retrieval performance
  1306	rather than single component across multiple metrics.
  1307	Hyper-parameters. As shown in Figure 6, to optimize
  1308	performance across different datasets, we carefully select
  1309	hyper-parameters based on dataset characteristics, where a
  1310	grid search strategy is employed for each TPR dataset. The
  1311	bottleneck reduction b of L-Adapter is consistently set to 8
  1312	across all three TPR datasets to balance performance and efficiency. For LoRA, its rank r is adjusted according to dataset
  1313	size: CUHK-PEDES and ICFG-PEDES, being relatively larger
  1314	datasets, use a rank of 32 to ensure sufficient representational
  1315	capacity, whereas RSTPReid, with its smaller dataset size,
  1316	benefits from a reduced rank of 16 to prevent overfitting while
  1317	maintaining effective adaptation. Similarly, the length of SPrefix l is tailored to dataset-specific needs: CUHK-PEDES
  1318	(10), ICFG-PEDES (14), and RSTPReid (2). This reflects
  1319	that for small datasets, choosing a smaller l helps control the
  1320	model’s parameter size and improves its performance on the
  1321	limited data.
  1322	Ablations on LoRA and its variants. As shown in Tablse
  1323	VI, we further analyze the ablation experiment on LoRA
  1324	and its variants, i.e., DoRA [74], and WoRA [75] in our
  1325	architecture. We observe that the method incorporating the
  1326	LoRA component outperforms those with DoRA and WoRA in
  1327	overall metrics, both of which introduce additional parameters
  1328	compared to LoRA. Therefore, we ultimately choose LoRA
  1329	submodule in our UP-Person.
  1330	
  1331	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
  1332	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1568,1820p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1568	∗ indicates our replication results after a minor bug correction, also viewed as data augmentation in vanilla IRRA.
  1569	
  1570	Fig. 6. The analysis of hyper-parameters at R@1 on CUHK-PEDES, ICFG-PEDES and RSTPReid.
  1571	
  1572	C. Analysis of S-Prefix.
  1573	Table VII demonstrates the effectiveness of S-Prefix compared to vanilla prefix based on the CLIP backbone and
  1574	UP-Person framework. First, we take CLIP (S-Prefix) as an
  1575	
  1576	example to clarify the individual effect of Sp . Compared
  1577	to CLIP (Vanilla Prefix), CLIP (S-Prefix) gains a dramatic
  1578	improvement of 5.64% when Sp = 150, demonstrating that
  1579	integration Sp can significantly improve retrieval performance.
  1580	
  1581	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
  1582	
  1583	w/ CLIP
  1584	
  1585	w/o CLIP
  1586	
  1587	TABLE II
  1588	C OMPARISON ON ICFG-PEDES.
  1589	Method
  1590	CMPM/C [49]
  1591	ViTAA [65]
  1592	SSAN [50]
  1593	SAF [67]
  1594	TIPCB [16]
  1595	IVT [15]
  1596	TGDA [68]
  1597	Wu et al. [70]
  1598	LGUR [69]
  1599	APTM (w/o MALS) [71]
  1600	APTM (w/ MALS) [71]
  1601	
  1602	R@1
  1603	43.51
  1604	50.98
  1605	54.23
  1606	54.86
  1607	54.96
  1608	56.04
  1609	57.26
  1610	57.69
  1611	59.02
  1612	57.49
  1613	68.51
  1614	
  1615	R@5
  1616	65.44
  1617	68.79
  1618	72.63
  1619	72.13
  1620	74.72
  1621	73.60
  1622	75.19
  1623	75.79
  1624	75.32
  1625	75.84
  1626	82.99
  1627	
  1628	R@10
  1629	74.26
  1630	75.78
  1631	79.53
  1632	79.13
  1633	81.89
  1634	80.22
  1635	81.80
  1636	82.67
  1637	81.56
  1638	82.60
  1639	87.56
  1640	
  1641	mAP
  1642	32.76
  1643	32.27
  1644	36.07
  1645	32.41
  1646	41.22
  1647	
  1648	CFine [14]
  1649	IRRA-CLIP [13]
  1650	IRRA∗ [13]
  1651	IRRA [13]
  1652	CSKT [27]
  1653	DM-Adapter [73]
  1654	UP-Person (Ours, ViT-16/B)
  1655	UP-Person (Ours, ViT-14/L)
  1656	
  1657	60.83
  1658	56.74
  1659	61.36
  1660	63.46
  1661	58.90
  1662	62.64
  1663	65.02
  1664	65.98
  1665	
  1666	76.55
  1667	75.72
  1668	78.66
  1669	80.25
  1670	77.31
  1671	79.53
  1672	80.98
  1673	81.94
  1674	
  1675	82.42
  1676	82.26
  1677	84.60
  1678	85.82
  1679	83.56
  1680	85.32
  1681	86.16
  1682	87.05
  1683	
  1684	31.84
  1685	37.95
  1686	38.06
  1687	33.87
  1688	36.50
  1689	38.32
  1690	40.12
  1691	
  1692	w/ CLIP
  1693	
  1694	w/o CLIP
  1695	
  1696	TABLE III
  1697	C OMPARISON ON RSTPR EID .
  1698	Method
  1699	DSSL [66]
  1700	SSAN [50]
  1701	SAF [67]
  1702	IVT [15]
  1703	TGDA [68]
  1704	APTM (w/o MALS) [71]
  1705	APTM (w/ MALS) [71]
  1706	
  1707	R@1
  1708	32.43
  1709	43.50
  1710	44.05
  1711	46.70
  1712	48.35
  1713	47.20
  1714	67.50
  1715	
  1716	R@5
  1717	55.08
  1718	67.80
  1719	67.30
  1720	70.00
  1721	73.15
  1722	70.85
  1723	85.70
  1724	
  1725	R@10
  1726	63.19
  1727	77.15
  1728	76.25
  1729	78.80
  1730	80.30
  1731	80.00
  1732	91.45
  1733	
  1734	mAP
  1735	36.81
  1736	37.96
  1737	36.36
  1738	52.56
  1739	
  1740	CFine [14]
  1741	IRRA-CLIP [13]
  1742	IRRA∗ [13]
  1743	IRRA [13]
  1744	CSKT [27]
  1745	DM-Adapter [73]
  1746	UP-Person (Ours, ViT-16/B)
  1747	UP-Person (Ours, ViT-14/L)
  1748	
  1749	50.55
  1750	54.05
  1751	57.50
  1752	60.20
  1753	57.75
  1754	60.00
  1755	63.15
  1756	64.45
  1757	
  1758	72.50
  1759	80.70
  1760	80.15
  1761	81.30
  1762	81.30
  1763	82.10
  1764	83.45
  1765	84.75
  1766	
  1767	81.60
  1768	88.00
  1769	87.05
  1770	88.20
  1771	88.35
  1772	87.90
  1773	89.75
  1774	90.95
  1775	
  1776	43.41
  1777	44.31
  1778	47.17
  1779	46.43
  1780	47.37
  1781	48.15
  1782	51.17
  1783	
  1784	11
  1785	
  1786	Meanwhile, it is obvious that both sequential and parallel
  1787	L-Adapter significantly surpass two other vanilla adapters,
  1788	which denotes that layernorm is extremely important in finetuning to TPR due to scale and shift parameters. Compared
  1789	to LN-tuning, parallel L-Adapter gains an overwhelming improvement by +1.64% on R@1. We speculate that unfreezing
  1790	layernorm will directly destroy the knowledge of the pretrained CLIP.
  1791	Analysis on conflict mitigation. As shown in Table IX, we
  1792	compare our UP-Person (No.1) and other three combinations
  1793	of adapter including the mixed adapter with two types of
  1794	adapters (No.2 and No.4) and vanilla adapter (No.3).
  1795	The experimental results reveal that UP-Person with both
  1796	MHA L-Adapter and MLP L-Adapter (No.1) outperforms all
  1797	other combinations in terms of all evaluation metrics: R@1,
  1798	R@5, R@10, and mAP, showcasing superior retrieval performance. The vanilla adapter configuration (No.3) performs
  1799	slightly worse than the L-Adapter configurations, with lower
  1800	values for R@1, R@5, R@10, and mAP compared to No.1.
  1801	This indicates that the presence of an L-Adapter, particularly
  1802	in both the MHA and MLP layers, is crucial for boosting
  1803	performance. Other mixed configurations (No.2 and No.4)
  1804	also underperform to our full L-Adapter configuration, further
  1805	suggesting that L-Adapter is better suited for the unified structure. We speculate that the vanilla adapter, spanning multiple
  1806	network layers (MHA or MLP), introduces conflicts with other
  1807	PETL components in the optimization space, whereas the LAdapter, which focuses solely on adjusting the global distribution through LayerNorm, avoids these spatial conflicts without
  1808	entangling, and optimizes the structure more effectively.
  1809	E. Analysis of loss functions.
  1810	
  1811	With an increment of Sp , all metrics are enhanced, indicating
  1812	that extending the attention values corresponding to the prefix
  1813	part can effectively alleviate the problem of slow gradient
  1814	changes. However, excessively large Sp such as Sp = 500
  1815	leads to gradient exploding, and then the algorithm fails to
  1816	converge. In the unified framework which incorporates LoRA
  1817	and L-Adapter, UP-Person (S-Prefix) has a significant gain
  1818	than UP-Person (Vanilla Prefix) by +0.53% on R@1 when
  1819	Sp = 10. We finally choose the appropriate parameter Sp = 10
  1820	in our UP-Person.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf' - | nl -ba | sed -n '707,782p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   707	Ablation Experiments
   708	
   709	To evaluate the effectiveness of the proposed method, ablation experiments are conducted on two
   710	public datasets using the All-Search and Indoor-Search as the test modes for the SYSU-MM01
   711	dataset, and Visible-Infrared and Infrared-Visible as the test modes for the RegDB dataset. The
   712	experiments use 𝑅𝑎𝑛𝑘1 (𝑅1), 𝑅𝑎𝑛𝑘10 (𝑅10), 𝑅𝑎𝑛𝑘20 (𝑅20), and 𝑚𝐴𝑃 as the evaluation metrics to
   713	measure the model’s performance. Detailed experimental results are presented in Tables 1 and 2,
   714	where 𝑀𝐼𝐷𝑀 ∗ denotes the 𝑀𝐼𝐷𝑀 without using the adversarial decoupling loss and FCRS, and
   715	𝐵 represents the AGW [48] baseline network with ResNet50 as the backbone, trained only with
   716	identity loss and batch hard triplet loss, without incorporating the proposed method.
   717	Comparing the feature decoupling module with and without the adversarial decoupling loss,
   718	the model’s Rank values and mAP showed significant improvements when the adversarial decoupling loss was applied. This indicates that relying solely on the channel masking mentioned in
   719	Section 3.3 makes it difficult to achieve effective decoupling of residual features, and it may even
   720	hinder the optimization of pedestrian features. Moreover, this demonstrates the crucial role that
   721	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   722	
   723	Visible-Infrared Person Re-Identification
   724	
   725	265:11
   726	
   727	Fig. 3. Heatmap visualization of decoupled features.
   728	
   729	adversarial decoupling loss plays in the decoupling process, effectively promoting the separation of
   730	identity features from redundant features, thereby validating the effectiveness of the global feature
   731	decoupling approach proposed in Section 4.2.
   732	4.3 Visual Analysis
   733	Heatmap Visualization. To validate the decoupling effect of the feature decoupling method, we
   734	selected two individuals and utilized the Grad-CAM [28] method to showcase the activation
   735	heatmaps of their instance-normalized features 𝐹e𝑐𝑖 , and identity features 𝐹𝑖𝑑 in different modalities,
   736	as illustrated in Figure 3. The first row represents visible light images, and the second row represents
   737	infrared images. From Figure 3, it is observed that the activation area of 𝐹e𝑐𝑖 is relatively limited,
   738	indicating that although 𝐹e𝑐𝑖 can alleviate modality differences, it also loses a considerable amount
   739	of identity information. When 𝐹e𝑐𝑖 is combined with the negative residual feature 𝑅 − to reconstruct
   740	the redundant feature 𝐹𝑛 , the corresponding heatmap mainly exhibits higher activation values in
   741	the background region. In contrast, the activation area of 𝐹𝑖𝑑 is predominantly concentrated in the
   742	human body region, demonstrating that the recombination of 𝐹e𝑐𝑖 with the positive residual feature
   743	𝑅 + can capture more discriminative identity information. Therefore, the heatmap visualization
   744	experiment convincingly illustrates that our feature decoupling method effectively guides the model
   745	to achieve a good decoupling effect between identity features and redundant features in pedestrian
   746	images, reducing the interference of redundant information on the discriminative identity features.
   747	Feature Distribution Visualization. To further analyze the role of the adversarial decoupling loss,
   748	we randomly selected 10 identities and employed t-SNE [32] to visualize the feature distributions
   749	extracted by HDNet, including features 𝐹 , 𝐵𝑎𝑠𝑒, 𝐹𝑐𝑖 , and 𝐹𝑖𝑑 . In Figure 4, 𝐹 , as the original feature
   750	distribution from pretrained ResNet50, has the least ideal effect. The 𝐵𝑎𝑠𝑒 shows the feature distribution extracted by the AGW [48] baseline, where the clustering effect is poor, with significant
   751	overlap between different classes and unsatisfactory intra-class compactness and inter-class separability. The 𝐹𝑐𝑖 figure shows the feature distribution after CDM, where the camera-irrelated feature
   752	decoupling effectively reduces the differences between different modalities, resulting in better
   753	separation of different classes. Compared to the baseline network features, the clustering effect
   754	is significantly improved. The 𝐹𝑖𝑑 figure shows the distribution after MIDM, where the identity
   755	feature decoupling further enhances intra-class compactness and inter-class separability, with each
   756	class forming tight and distinct clusters.
   757	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 265. Publication date: September 2025.
   758	
   759	265:12
   760	
   761	H. Ding et al.
   762	
   763	Fig. 4. Visualization of feature distribution by t-SNE. Where (a) represents the original feature distribution
   764	from pretrained ResNet50; (b) represents the feature distribution visualization after passing through the AGW
   765	[48] baseline; (c) is the feature distribution after CDM; (d) is the feature distribution after MIDM.
   766	
   767	Overall, as the multi-level feature decoupling progresses, the camera-irrelated feature decoupling
   768	has significantly improved the clustering effect compared to the baseline model, and the identity
   769	feature decoupling further enhances intra-class compactness and inter-class separability, achieving
   770	more precise identity differentiation for cross-modal pedestrian images.
   771	4.4
   772	
   773	Comparison with the State-of-the-Art Methods
   774	
   775	As shown in Table 3, our proposed HDNet outperforms the existing state-of-the-art methods
   776	on SYSU-MM01. Among them, SDL [16] proposed a non-adversarial feature decoupling method,
   777	Hi-CMD [6], JSIA [34], TSME [21] also draw on the decoupling idea to deal with the modal style
   778	information, in which the cross-modal paired images had to be acquired by Generate Adversarial
   779	Network, and the process of image generation consumes more time and space, and had greater
   780	volatility, making training more difficult. In comparison to the aforementioned decoupling-based
   781	methods, HDNet adopts an end-to-end training approach, eliminating the need for image generation,
   782	thereby improving training efficiency and achieving superior results. Among existing methods,

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf' - | nl -ba | sed -n '613,688p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   613	3 Experiments
   614	3.1 Datasets and Evaluation Protocols
   615	SYSU-MM01 [15]. The dataset contains two test modes (i.e., allsearch and indoor-search), in which the all-search mode is more
   616	challenging. We evaluate the method on specified test set and randomly generate gallery for 10 times following [13, 15].
   617	RegDB [9]. The RegDB dataset contains 412 identities. The proposed method is evaluated under both the Visible to Infrared mode
   618	and the Infrared to Visible mode. We use the same evaluation protocol as [17, 19, 20], in which the training/testing split procedure
   619	is repeated 10 times.
   620	
   621	3.2
   622	
   623	Fig. 5: Visualization results of feature maps. For each image, we
   624	show the original image followed by feature maps extracted by twostage model and MPT network.
   625	
   626	Comparison with State-of-the-art Methods
   627	
   628	3.4
   629	
   630	We compare the proposed method with the state-of-the-art methods, including the method based on GAN, the methods based on
   631	different structures and loss functions, and the methods based on
   632	the intermediate modality.
   633	The comparisons on the SYSU-MM01 dataset are shown in Tab. 1.
   634	The MPT network outperforms all methods. The results suggest
   635	that prompts with person-related information can more effectively
   636	bridge the gap between visible and infrared modalities. And the
   637	reason for achieving such high performance is that we treat prompts
   638	as the bridge between two modalities, and align features of the
   639	two modalities to text features. This effectively reduces the discrepancies between the two modalities. More importantly, we utilize the knowledge distillation technique to help prompts learn
   640	person-related information, making these prompts better serve as
   641	the bridge. The MPT network also outperforms all existing stateof-the-art methods on RegDB dataset, which proves the previous
   642	analysis.
   643	
   644	3.3
   645	
   646	Visualization
   647	
   648	Feature Distribution. As shown in the Fig. 3, we display the distribution of features extracted by the baseline network and the
   649	MPT network. It is obvious that features with the same identity but
   650	different modalities are more clustered in the MPT network than
   651	in the baseline network, indicating that the MPT network greatly
   652	reduces inter-modality discrepancies.
   653	Following the work [24], we visualize the feature distribution
   654	with the frequency of inter-class and intra-class distances, as shown
   655	in Fig. 4. As observed, the MPT network significantly enlarges the
   656	gap between the means of inter-class and intra-class distances compared to the other two models (𝛿1 < 𝛿2 < 𝛿3 ). Therefore, MPT network effectively reduces intra-class discrepancies, which indicates
   657	that the prompts indeed bridge the inter-modality gap.
   658	Focus Areas. To verify that prompts generated by MPT focus
   659	on person-related information, we visualized feature maps of twostage model and MPT network. During training, 𝑊𝐼 ⋅𝐼𝑖 and 𝑊𝑇 ⋅𝑇𝑖
   660	in Eq. 4 are projected into the common space, so we visualized
   661	feature maps of 𝑊𝐼 ⋅ 𝐼𝑖 to approximate that of 𝑊𝑇 ⋅ 𝑇𝑖 (text feature maps cannot be visualized). As shown in Fig. 5, feature maps
   662	of MPT pay more attention to body areas, which means that the
   663	prompts generated by MPT are more sensitive to identity-related
   664	information.
   665	
   666	Ablation Study
   667	
   668	We conduct experiments on the SYSU-MM01 dataset to evaluate
   669	the effectiveness of key components in the MPT network. The experiments are conducted under both all-search and indoor modes,
   670	and Tab. 2 shows the performance of each variant. And we use the
   671	results under all-search mode for analysis. Compared to the baseline network, MPT improves 5.45% on Rank-1 and 5.88% on mAP.
   672	The result means that text prompts and image encoders can indeed promote each other, ultimately enhancing the model’s ability
   673	for cross-modality retrieval. When using the same network structure and loss functions, the alternating training strategy (MPT w/
   674	𝐿𝑀𝑃 𝑇 ) improves 1.69% and 2.23% over two-stage training strategy. This indicates that by improving the quality of generated text
   675	prompts, CLIP can play a more significant role in VI-ReID. What’s
   676	more, the superior results obtained when using 𝐿𝑀𝑃 𝑇 over 𝐿𝐶𝐿
   677	(the contrastive learning loss) indicate that the designed losses effectively help prompts and encoders mutually reinforce each other.
   678	In addition, under the same experimental setup, the two-stage
   679	training model method requires 11 hours and 56 minutes, while
   680	MPT network requires 12 hours and 12 minutes. This indicates
   681	that MPT can significantly improve performance with only a slight
   682	increase in training time. Meanwhile, since the text modality is
   683	not involved in the inference process, the two methods require the
   684	same amount of time during the testing phase.
   685	
   686	4
   687	
   688	Conclusion

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '947,1102p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   947	C. Ablation Study
   948	1) Analysis of the Different Components of RHL: As
   949	demonstrated in Tab. VI, we conducted an ablation study of
   950	various modules in our method on the SYSU-Real dataset. The
   951	combination of Lr e and LC constitutes the DART baseline.
   952	After the incorporation of L I N , a significant improvement
   953	of 6.79% in Rank-1 is observed, attributable to the powerful
   954	suppression of label noise by GCE. As shown in the final
   955	row, our method achieves a 7.23% increase in Rank-1, a
   956	6.94% increase in mAP, and a 7.48% increase in mINP. L O N
   957	addresses the evidently erroneous samples. Although these
   958	samples, under the restraint of GCE, do not cause a biased
   959	
   960	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   961	
   962	4866
   963	
   964	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   965	
   966	TABLE V
   967	C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON THE R EG DB
   968	DATASET U NDER THE R ANDOM N OISE R ATIO OF 10%, 30% AND 50%,
   969	R ESPECTIVELY. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD
   970	
   971	Fig. 5. Histogram of the effect of different types of noise on the degradation
   972	of the metrics. We present it using the non-robust SOTA method DSCNet [8]
   973	as an example.
   974	
   975	may impede convergence. Thus, a value around 0.7 yields
   976	optimal effects. (b) Concerning β, it balances the losses from
   977	classification and retrieval. Since the metric prioritizes retrieval
   978	performance, suboptimal performance is observed when β is
   979	less than 1; whereas, a value of 3 achieves a better balance.
   980	(c) As for ε1 , a value that is too large leads to the misalignment
   981	of gradients in L O N due to stochastic factors, while a value
   982	that is too small causes the network to overlook learning
   983	correct labels.
   984	
   985	TABLE VI
   986	I IMPACTS OF D IFFERENT L OSS F UNCTION C OMBINATIONS ON SYSU-Real
   987	
   988	optimization direction in the model, they evidently blunt the
   989	loss, resulting in performance degradation. With the addition
   990	of L O N , there is a 0.44% increase in Rank-1, a 1.53% increase
   991	in mAP, and a 2.1% increase in mINP. The reason why the
   992	contribution to performance of L O N is not as significant as
   993	that of L I N lies in the fact that L I N is the primary source
   994	of robustness in classification performance for RHL, while
   995	L O N deals with only a small portion of the data. When L O N
   996	works alone, as in typical VI-ReID methods, its performance
   997	is subpar. This is due to its fractional operation on gradient
   998	values, causing the loss to be distributed across various soft
   999	labels of categories. Consequently, it can only fine-tune the
  1000	gradient direction but struggles to form an effective loss value.
  1001	2) Analysis of the Hyper-Parameter Sensitivities: We
  1002	primarily ablate the robustness parameter q, balance parameter
  1003	β, and threshold parameter ε1 . For the constituent loss
  1004	parameters β1 , β2 and β3 , we adopted a strategy that adaptively
  1005	calculates based on the sample proportion. As illustrated in
  1006	Fig. 4, we present the optimal values for each parameter.
  1007	Specifically, the principles underlying the roles of these three
  1008	hyper-parameters are distinct: (a) For q, a higher value
  1009	enhances robustness; however, excessively strong robustness
  1010	
  1011	D. Visualization Analysis
  1012	1) Indicator Degradation Under Different Noise Types:
  1013	The impact of various types of noise on the learned data
  1014	distribution by models varies significantly. Taking the stateof-the-art method DSCNet as an example, we examine the
  1015	differential effects of different noise types on performance
  1016	metrics under equitably imposed noise rates. Within the
  1017	context of VI-ReID, wherein the discrimination of identity
  1018	categories is notably challenging, even for human observers,
  1019	the issue of real-world noise annotation presents a formidable
  1020	challenge. As depicted in Fig. 5, all types of label noise
  1021	induce substantial performance degradation. Notably, both
  1022	IDN and real-world noise exhibit exacerbated degradation in
  1023	comparison to random noise, primarily due to the association
  1024	of their noise labels with specific instances, rendering them
  1025	more misleading. On the Rank-1 metric, real-world noise
  1026	may even lead to performance decrements surpassing 50%.
  1027	Consequently, the investigation of real-world noise emerges as
  1028	an inherently challenging and profoundly significant endeavor.
  1029	2) Analysis of Noise Resistance Against DART: To
  1030	demonstrate the noise resistance of our method, we conducted a similarity comparison using the extracted features.
  1031	As illustrated in Fig. 6, the similarity matrix coordinates are
  1032	constructed from sixteen samples of two distinct identities.
  1033	Considering the exemplary performance of DART, which
  1034	makes the absolute values of the similarity matrix less
  1035	illustrative, we opted to use the difference between RHL and
  1036	DART as a surrogate measure. Red indicates an increase in
  1037	similarity, while blue signifies a decrease. It is evident from
  1038	the heatmap, especially within the black frames, that there
  1039	is a notable increase in similarity, underscoring our method’s
  1040	significant improvement in counteracting the degradation
  1041	caused by noisy labels. Specifically, the matrix is arranged in
  1042	
  1043	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
  1044	
  1045	ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE
  1046	
  1047	Fig. 6.
  1048	Heatmap of the difference in similarity matrices between RHL
  1049	and baseline. The left side displays 16 sample identities, each represented
  1050	by four modalities, effectively showcasing two distinct identities. On the
  1051	right side, the heatmap depicts the disparity between the similarity matrix
  1052	of RHL and that of the state-of-the-art (SOTA) baseline, namely DART.
  1053	This selection of differences is aimed to prominently highlight the enhanced
  1054	performance of our method when dealing with label noise. The matrix
  1055	coordinates correspond to the sample numbers on the left, with positions
  1056	within the black frame indicating intra-class similarities, and those outside
  1057	the frame representing inter-class similarities. Notably, RHL significantly
  1058	improves intra-class similarity while reducing inter-class similarity.
  1059	
  1060	a sequence of visible - infrared - visible - infrared, allowing
  1061	us to discuss four scenarios:
  1062	(a) Intra-class, same modality, i.e., areas near the diagonal.
  1063	For visible light, there is a pronounced enhancement, while
  1064	for infrared, the increase is somewhat weaker. This is
  1065	attributed not only to the data itself but also to the inherently
  1066	high similarity within the infrared modality, making further
  1067	improvement challenging.
  1068	(b) Intra-class, different modalities, located near the offdiagonal areas within the black frame. Here too, there is
  1069	a noticeable increase in similarity, which is of paramount
  1070	importance for re-identification.
  1071	(c) Inter-class, same modality. For visible light images, there
  1072	is an unexpected rise in similarity, likely due to a global
  1073	elevation in similarity levels caused by RHL’s enhancement
  1074	of overall similarity (a byproduct of its strong robustness
  1075	and improved feature extractor capabilities in resisting noise
  1076	labels). This results in visible light modalities, which contain
  1077	more detailed features, experiencing a relative increase.
  1078	In contrast, the similarity in the infrared modality significantly
  1079	decreases.
  1080	(d) Inter-class, different modalities. This scenario exhibits
  1081	considerable randomness, but overall, there tends to be a
  1082	decline in similarity.
  1083	3) t-SNE Visualization of Classification Robust: To accentuate the discernment of categorical information facilitated
  1084	by our method, we employ t-distributed Stochastic Neighbor
  1085	Embedding visualization on the extracted features. As depicted
  1086	in Fig. 7, our method demonstrates superior clustering efficacy
  1087	compared to DART. Specifically, intra-class cohesion is
  1088	enhanced, while inter-class distinctiveness is accentuated. The
  1089	main reason for this lies in the different strategies for different
  1090	subsets; the use of CE loss on clean subsets ensures the speed
  1091	of optimisation, while losses on both obviously noisy and
  1092	indistinguishably noisy subsets greatly enhance the robustness
  1093	of the method.
  1094	
  1095	4867
  1096	
  1097	Fig. 7. The t-SNE visualisation plot of the extracted features. 10 identities are
  1098	randomly selected and the features of each identity of the sample are extracted
  1099	and used for t-SNE visualisation with the model trained on SYSU-Real. The
  1100	left side shows the results of the SOTA robust model DART and the right
  1101	side shows the results of our model RHL.
  1102	TABLE VII

exec
/bin/zsh -lc "pdftotext 'When Large Vision-Language Models Meet Person Re-Identification.pdf' - | rg -n \"PSTG|SGI|CSS|semantic token|learnable token|visual-only|Visualization|retrieval|inference time|ablation|Ablation\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
23:one semantic token that encapsulates key appearance semantics
25:Semantic-Guided Interaction (SGI) module, establishing a reciprocal interaction between the semantic token and visual tokens.
26:Ultimately, the reinforced semantic token serves as the representation of pedestrian identity. Our framework integrates the semantic
101:the reasoning and understanding capabilities of LVLMs, while adhering to the mainstream ReID paradigm of feature extraction combined with feature similarity-based retrieval [1]. A potential solution involves using LVLMs to describe pedestrian images and finetuning the visual encoder via tasks such as image-text matching or
139:within pedestrian images, resulting in the generation of one semantic token that encapsulates the pedestrian’s appearance information. Subsequently, an efficient interaction module is designed to facilitate refinement between the generated token and the visual tokens. Finally, the
140:reinforced semantic token is optimized and employed for person retrieval.
145:semantics in pedestrian images, generating a semantic token representing the pedestrian’s appearance information. We then design
147:reinforcing the semantic token as a discriminative identity representation. Ultimately, the reinforced semantic token is optimized and
148:used during inference to achieve person retrieval. Our framework
154:show that one LVLM-generated semantic token can effectively facilitate the learning of pedestrian representations. Our contributions
157:perspective on using generative language models to assist discriminative visual models. (2) We propose to utilize the generative capability of LVLMs to produce a semantic token for pedestrians and design a semantic-guided interaction module leveraging the generated
158:semantic token to enhance identity representations. (3) Experimental results show that, without requiring additional annotations, our
199:by guiding the ReID model to generate one semantic token that encapsulates instructive information of the pedestrian. To achieve this,
209:semantics in the image, and then generate a semantic token that summarizes the relevant identity features. We denote this generated token as <REID>, which serves as a compact representation of the
211:Camera semantic supplementation. The semantic token generation process overlooks the influence of camera variations. To improve pedestrian semantic consistency across cameras, we explicitly model and account for these camera-induced feature variations.
218:generated pedestrian semantic token with the camera semantics as
231:We design the Semantic-Guided Interaction (SGI) module to facilitate bidirectional interaction between the generated semantic token
232:and the visual tokens. Specifically, the generated semantic token is
241:the guidance of the semantic token. Meanwhile, the semantic token, serving as the pivot for information aggregation, distills more
246:ẑ = [v̂reid ; v̂1 ; v̂2 ; · · · ; v̂n ] = SGI(z).
250:Then, the reinforced semantic token representation v̂reid is used
255:and the SGI module. The parameters of LLM are frozen, but gradients backpropagate through it to update other learnable components.
257:SGI module, we achieve a joint end-to-end training that harnesses
431:<REID> token for each input image. Then, the reinforced semantic token representation, v̂reid , is used to compute the cosine
446:tokens are included in the input of LLM and our SGI module. Following [2], random horizontal flipping, padding, random cropping,
458:Ablation studies of our key two components on DukeMTMC-reID
461:Table 3. Ablation of the camera semantic supplementation
462:(CSS) strategy. CSS-vreid and
463:CSS-xp denote adding the camera embedding to vreid and xp ,
479:Ours w/o PSTG
480:Ours w/o SGI
510:w/o CSS
511:CSS-vreid
512:CSS-xp
560:3.3. Ablation Studies
561:Effectiveness of the generated pedestrian semantic token. (1)
564:achieving inferior performance. (2) In the variant “Ours w/o PSTG”,
565:we replace the LVLM-generated semantic token with a learnable token, similar to the design of the [class] token [17], to integrate
567:substantial performance drop since the randomly initialized learnable token lacks rich semantic cues. This result underscores the
568:importance of our PSTG mechanism, which contributes to a more
570:Effectiveness of the SGI module. In the “Ours w/o SGI” variant,
571:we remove the SGI module and rely solely on the LVLM-generated
572:semantic token for ReID. As shown in Tab. 2, this configuration
573:achieves reasonably good performance, suggesting that our PSTG
575:the importance of the SGI module in leveraging the generated semantic token. It represents a novel paradigm of enhancing identity
576:representations with the LVLM-generated semantic token.
577:Ablation of the camera semantic supplementation strategy. In
578:Tab. 3, the result of CSS-vreid shows that camera semantics can
580:transferring the usage of camera embeddings to the input of the visual model (denoted by CSS-xp ), we observe a better performance.
595:Fig. 3. Visualization of retrieval results. For each query, the first
596:and the second rows show the top-8 retrieval results of the baseline
601:the LVLM-generated semantic token, further improving the model’s
603:Inference time. Our method requires generating only a single semantic token using LVLM, which can be accomplished in a single
607:increase in inference time is acceptable and can be further optimized
609:Visualization of retrieval results. As shown in Fig. 3, the baseline model often relies on coarse-grained and vulnerable features,
611:to false positives. In contrast, guided by the semantic token, our
625:instructions, multiple semantic tokens, or larger LVLM variants.

codex
证据部分也基本齐了。最后我会按用户要的五个问题逐篇写，不把它们拆成“模块清单”，而是强调从观察到问题重写、再到机制和证据的逻辑链。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
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
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1810,1855p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1810	
  1811	With an increment of Sp , all metrics are enhanced, indicating
  1812	that extending the attention values corresponding to the prefix
  1813	part can effectively alleviate the problem of slow gradient
  1814	changes. However, excessively large Sp such as Sp = 500
  1815	leads to gradient exploding, and then the algorithm fails to
  1816	converge. In the unified framework which incorporates LoRA
  1817	and L-Adapter, UP-Person (S-Prefix) has a significant gain
  1818	than UP-Person (Vanilla Prefix) by +0.53% on R@1 when
  1819	Sp = 10. We finally choose the appropriate parameter Sp = 10
  1820	in our UP-Person.
  1821	D. Analysis of L-Adapter.
  1822	Table VIII shows the effectiveness of LN-tuning [64],
  1823	vanilla adapter and our L-Adapter in UP-Person, and further
  1824	explores the sequential and parallel connection. As depicted
  1825	in Figure 4(a), Sequential Adapter is inserted behind MLP
  1826	and MHA, which is the original adapter structure as in [37].
  1827	Sequential L-Adapter is connected behind layernorm in series.
  1828	Similarly, parallel adapters span specific network components
  1829	which belong to vanilla CLIP, which are demonstrated in
  1830	Figure 4(c) and (d). LN-tuning in Figure 4(e) fine-tunes
  1831	the gain (for scale operation) and bias (for shift operation)
  1832	parameters while keeping other parameters frozen.
  1833	We observe that the performance of parallel vanilla adapter
  1834	is far lower than that of parallel L-Adapter (-1.17% on R@1).
  1835	
  1836	Table X provides ablation studies on various loss functions,
  1837	including the ITC (Image-Text Contrastive) loss, ID loss (for
  1838	person identity classification) and MLM loss (for masked
  1839	implicit reasoning) based on IRRA [13]. It shows that SDM
  1840	achieves superior results in all metrics compared to ITC loss
  1841	by +2.13%, +1.33%, +0.51%, and +1.20%, respectively. The
  1842	overall Rank metrics with ID loss are lower than those of SDM
  1843	alone, where ID loss primarily focuses on classification and
  1844	plays a supplementary role in TPR task. The combination of
  1845	SDM+ID+MLM improves R@5, R@10 and mAP metric due
  1846	to its integration with the cross-attention module for implicit
  1847	reasoning. However, this slight gain from MLM comes at the
  1848	cost of an dramatic increase in model size and trainable parameters. Therefore, UP-Person with SDM offers an optimal tradeoff between complexity and performance without bells and
  1849	whistles, and its simplicity provides a scalable foundational
  1850	framework, making it easy to combine with different types of
  1851	losses based on objectives.
  1852	F. Domain Generalization Performance Comparison
  1853	In Table XI, we conduct domain generalization experiments between different datasets derived from CUHK-PEDES,
  1854	ICFG-PEDES and RSTPReid. We train models on the dataset
  1855	of source domain and then transfer them to the target domain

exec
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '2220,2258p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  2220	7.4M
  2221	7.4M
  2222	13.1M
  2223	52.3M
  2224	
  2225	advantage compared with IRRA when source domain is RSTPReid, which gains a significant improvement on R@1 by
  2226	+4.21% and +6.2% when target domain is CUHK-PEDES
  2227	and RSTPReid, respectively. Moreover, it is obvious that UPPerson is superior than IRRA-CLIP (full-tuning) in all metrics
  2228	by a large margin.
  2229	We conjecture that our method effectively alleviate overfitting compared to full-tuning-based methods since it merely
  2230	fine-tunes a few parameters of simple PETL components. This
  2231	advantage is more pronounced when the training dataset is
  2232	scarce. We observe that the size of RSTPReid is much less
  2233	than other datasets as elaborated in the beginning of Section
  2234	V. Thus, full-tuning-based methods such as IRRA and IRRACLIP can cause the more severe overfitting when lacking of
  2235	training data and further largely reduce generalization performance. All results demonstrate the powerful generalization
  2236	ability of our UP-Person.
  2237	G. Visualization
  2238	Figure 7 compares the top-10 retrieval results from the
  2239	IRRA (the first row) and our proposed UP-Person (the second
  2240	row). The matched and unmatched images are marked in
  2241	green and red, respectively. The yellow box denotes the
  2242	key differentiated retrieval objects between correct and false
  2243	retrieval. It can be seen that UP-Person can retrieve the corresponding pedestrian images for a query text more accurately.
  2244	For example, IRRA cannot correctly recognize the phrase bob
  2245	style in Query 1, white collared dark shirt and bag over his
  2246	right shoulder in Query 2, which represent fine-grained or
  2247	unusual objects or words compared to common instances. We
  2248	infer that the full-tuning method IRRA, which only utilizes
  2249	pre-trained parameters at initialization process, may lose part
  2250	of the original abundant knowledge of vanilla CLIP during
  2251	training. If the model of IRRA does not fully “see” the relevant
  2252	knowledge in training phase, inference when encountering the
  2253	“unseen” objects may fail. Overall, the visualization vividly
  2254	demonstrates the effectiveness of UP-Person.
  2255	H. Coarse-grained Text-to-Image Retrieval
  2256	We utilize two datasets for the coarse-grained retrieval task:
  2257	Flickr30K [76] and MSCOCO [77]. Unlike TPR datasets
  2258	

exec
/bin/zsh -lc "pdftotext 'UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1908,1942p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1908	195M
  1909	12M
  1910	16M
  1911	7.4M
  1912	
  1913	TABLE V
  1914	A BLATION STUDY ON EACH COMPONENT OF UP-P ERSON .
  1915	Components
  1916	S-Prefix LoRA L-Adapter
  1917	
  1918	No. Methods
  1919	0
  1920	1
  1921	2
  1922	3
  1923	4
  1924	5
  1925	6
  1926	7
  1927	
  1928	Zero-shot CLIP
  1929	+S-Prefix
  1930	+LoRA
  1931	+L-Adapter
  1932	+LoRA +L-Adapter
  1933	+S-Prefix +LoRA
  1934	+S-Prefix +L-Adapter
  1935	UP-Person (Ours)
  1936	
  1937	CUHK-PEDES
  1938	R@1 R@5 R@10
  1939	12.61 27.08 35.48
  1940	63.68 83.74 89.54
  1941	72.56 89.26 93.18
  1942	72.09 89.07 93.81

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '920,947p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   920	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   921	
   922	ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE
   923	
   924	4865
   925	
   926	TABLE IV
   927	C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON THE SYSU-MM01 DATASET U NDER THE R ANDOM N OISE W ITH N OISE R ATIOS OF 10%, 30%
   928	AND 50%, R ESPECTIVELY. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD
   929	
   930	Fig. 4. Hyper-Parameter Sensitivity Curves. The impact of robustness parameter q, balance parameter β, and threshold parameter ε1 on the main evaluation
   931	metrics. Different colors correspond to different metrics, with the orange line representing the hyper-parameter values in the optimal scenario.
   932	
   933	in Tab. V and IV, our approach is capable of achieving
   934	state-of-the-art performance in all scenarios. In the Indoor
   935	scenarios of SYSU-MM01, as the noise rate increases from
   936	30% to 50%, our method only experiences marginal declines
   937	of 0.73%, 0.37%, and 0.32% in the metrics of Rank-1, mean
   938	Average Precision (mAP), and mean Inverse Negative Penalty
   939	(mINP), respectively. In the case of RegDB, retrieving Visible
   940	images from Infrared images under noisy conditions poses a
   941	significant challenge. Our method, at a noise rate of 30%,
   942	achieves enhancements of 7.86%, 9.98%, and 12.33% in the
   943	metrics of Rank-1, mAP, and mINP, respectively, compared to
   944	DART. These results unequivocally demonstrate the robustness
   945	of our method.
   946	
   947	C. Ablation Study

exec
/bin/zsh -lc "pdftotext 'Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf' - | nl -ba | sed -n '739,920p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   739	IV. E XPERIMENTS
   740	A. Experiments Setting
   741	1) Datasets: Our proposed methodology was evaluated
   742	rigorously on the most prominent visible-infrared datasets:
   743	SYSU-MM01 [85], RegDB [86], and a modified version of
   744	SYSU-MM01 called SYSU-Real.
   745	SYSU-MM01, an extensive dataset, comprises a total of
   746	34,166 images, including 22,257 visible and 11,909 nearinfrared images. These images were captured using 4 visible
   747	and 2 infrared cameras, strategically positioned in diverse
   748	settings, encompassing both indoor and outdoor environments.
   749	This dataset offers a comprehensive platform for assessing
   750	the robustness and adaptability of our method across varied
   751	lighting and background conditions.
   752	Conversely, RegDB, while smaller in scale, presents a
   753	unique dataset configuration. It has been meticulously assembled using two aligned cameras, one visible and one infrared,
   754	contributing to a more controlled data acquisition process. The
   755	dataset encompasses 412 distinct person identities, with each
   756	identity represented by a set of 20 images, equally divided into
   757	10 visible and 10 infrared images. Despite its relatively smaller
   758	size and consequently lower complexity, RegDB serves as an
   759	essential benchmark for evaluating the effectiveness of our
   760	approach in a more constrained and uniform setting.
   761	Since both the random noise and the instance-dependent
   762	label noise are generated by code, we additionally upset
   763	the SYSU-MM01 dataset to re-labelled artificially, and this
   764	dataset with real-world labelled noise we call SYSU-Real. For
   765	relabelling, we simulate a cross-modal Re-ID process, where
   766	samples are labelled sequentially in id order. There must exist
   767	a certain amount of same-id samples in each packet, and both
   768	modalities will be found at the same time. Specifically, the
   769	
   770	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   771	
   772	ZHANG et al.: VISIBLE-INFRARED PERSON RE-IDENTIFICATION WITH REAL-WORLD LABEL NOISE
   773	
   774	4863
   775	
   776	TABLE I
   777	C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON THE SYSU-MM01 DATASET U NDER THE IDN W ITH
   778	N OISE R ATIOS OF 10%, 30% AND 50%, R ESPECTIVELY. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD
   779	
   780	TABLE II
   781	C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON THE R EG DBDATASET U NDER THE IDN W ITH N OISE R ATIO OF 10%, 30% AND
   782	50%, R ESPECTIVELY. T HE B EST R ESULTS A RE H IGHLIGHTED
   783	IN B OLD
   784	
   785	label distribution of this relabelled dataset is shown in Fig. 3.
   786	The total label noise ratio is 23.79%, out of which, 15.34%
   787	are noisy labels for visible data and 8.44% are noisy labels
   788	for infrared data. For the noise ratio within each mode, it is
   789	24.07% for infrared and 23.63% for visible. It can be observed
   790	that the proportion of noise is slightly higher in infrared, which
   791	is in line with the assumption that it will be more difficult to
   792	
   793	annotate infrared than visible samples because of issues such
   794	as low image quality in infrared modality.
   795	For the SYSU-MM01 and SYSU-Real datasets, we utilize
   796	both ‘all-search’ and ‘indoor-search’ modes, differentiating
   797	between visible and infrared images in the gallery and query
   798	sets. In contrast, the RegDB dataset is assessed under Infraredto-visible and visible-to-infrared modes. To ensure robustness,
   799	we follow the approach in [21] by randomly splitting the
   800	dataset into training and testing sets 10 times, reporting the
   801	overall average results. This rigorous approach allows for
   802	a comprehensive and accurate assessment of our model’s
   803	performance.
   804	2) Implementation Details: In implementing the proposed
   805	method, we utilized PyTorch version 1.10.0 and CUDA 11.0 as
   806	the core computing platforms. All experiments were conducted
   807	on GeForce RTX 3090 GPUs. The RHL framework is aligned
   808	with the Advanced Dual Path (ADP) architecture, and its
   809	coding structure is informed by the DART implementation,
   810	reflecting state-of-the-art practices in the field. On the cotraining architecture, our method’s output mechanism parallels
   811	that of other contemporary co-training approaches. We average
   812	the outputs from the two distinct modules A and B within
   813	the co-training system. Such an approach ensures a balanced
   814	and comprehensive representation of the learning process,
   815	leveraging the strengths of both modules to enhance overall
   816	model performance. For the generation of random noise,
   817	we followed the generation rules of DART, while the
   818	generation of IDNs refers to the generation algorithm in [29].
   819	The training and test sets are divided before generating the
   820	noisy annotations, ensuring that the i.i.d. assumption remains
   821	valid for the data itself. Noisy labels are applied only to the
   822	training set, while the test set labels remain clean to assess the
   823	algorithm’s robustness.
   824	
   825	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.
   826	
   827	4864
   828	
   829	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025
   830	
   831	TABLE III
   832	C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON THE SYSU-R EALDATASET. T HE B EST R ESULTS A RE H IGHLIGHTED IN B OLD
   833	
   834	3) Evaluation Metrics: In our experimental settings,
   835	we evaluate the performance on both datasets using three
   836	key metrics: Cumulative Matching Characteristic (CMC),
   837	Mean Average Precision (mAP), and Mean Inverse Negative
   838	Penalty (mINP) as specified in [82]. The CMC metric
   839	primarily focuses on rank accuracy, evaluating how effectively
   840	a model ranks the correct match within a list of potential
   841	candidates. This reflects the model’s precision in retrieval
   842	tasks. In contrast, the mAP metric measures the mean precision
   843	across all queries, offering a broader view of the model’s
   844	overall retrieval quality, including both precision and recall
   845	aspects. Lastly, the mINP metric provides insight into the
   846	model’s robustness against hard negative samples, assessing
   847	its performance in more challenging scenarios where incorrect
   848	matches are closely similar to the correct ones.
   849	
   850	B. Comparison With State-of-the-Arts
   851	1) Compared VI-ReID Methods: To illustrate the advancement of our method, we compared it with several stateof-the-art models on the datasets SYSU-MM01 and RegDB
   852	with noise rates of 0.1, 0.3 and 0.5. Models with which
   853	to compare include the following: (a) DDAG [17] utilizes
   854	Graph Structured Attention to enhance the feature by
   855	incorporating the neighbourhood information across two
   856	modalities. (b) AGW [82] employs a weighted regularized
   857	triple (WRT) loss to improve the retrieval performance of
   858	the network. (c) ADP [21] improves the robustness of the
   859	network to the effects of modal changes through operations
   860	such as colour enhancement. (d) DSCNet [8] focuses on
   861	channel-level semantic inconsistency to reduce the modality
   862	discrepancy. (e) DDEN [87] bridges the modal gap between
   863	visible and infrared with a novel enhancement network.
   864	(f) MSALNet [5] extracts shared features across modalities
   865	by using feature compensation, enhancing similarity, and
   866	applying center alignment loss to improve performance while
   867	reducing computational cost. (g) HOS-Net [88] lies in its highorder structure learning module, which models relationships
   868	across local features using a whitened hypergraph to improve
   869	feature representation. (h) IDKL [1] enhances modality-shared
   870	features by distilling identity-aware discriminative information
   871	from modality-specific features while reducing modality style
   872	discrepancies. (i) DART [19] is the first model to consider the
   873	label noise in the field of VI-ReID.
   874	
   875	2) Results on Instance-Dependent Label Noise: It can be
   876	observed in Tab. I and Tab. II that our method significantly
   877	outperforms previous methods on all metrics with all noise
   878	rate cases. Comparing with DART on SYSU-MM01, under
   879	IDN-50%, our method improves 6.39%, 2.37%, 0.94%, 5.61%
   880	and 5.76% on Rank-1, Rank-10, Rank-20, mAP and mINP,
   881	respectively. The superior performance of our method is
   882	even more evident in the indoor case of SYSU-MM01, with
   883	13.36%, 8.61%, 4.13%, 11.29%, and 10.52% enhancements
   884	in the five metrics, respectively, and also under IDN-50%.
   885	We analyzed it as a better light environment indoors, where
   886	the data is more distinguishable and the effect of noise
   887	is suppressed. This allows the GMM to better distinguish
   888	between noisy labels, which in turn improves the performance
   889	of the metrics. Compared with DART on RegDB, our model
   890	improves the performance on the metrics Rank-1, mAP, and
   891	mINP by 6.16%, 8.23%, and 11.78%, respectively, with 30%
   892	noise rate. However, for retrieval of visible images using
   893	infrared images in the RegDB dataset, things get complicated.
   894	Owing to the inherent unrecognizability of infrared images
   895	and the fact that the instance-dependent label noise is based
   896	on the quality of the image instances, the noise brings about
   897	catastrophic metrics degradation (compare Visible Retrieval
   898	of Infrared Images). With the noise rate of 0.5, the model
   899	will be completely obfuscated, yet our model still maintains a
   900	performance improvement of 1.82% than DART.
   901	3) Results on Real-World Noise: As shown in Tab. III, the
   902	results of our method conducted on the SYSU-Real dataset
   903	are similarly significant. Under the All-Search scenario,
   904	our approach demonstrates a notable improvement over the
   905	previously established optimal baseline, DART. Specifically,
   906	there is a 7.23% increase in Rank-1 accuracy, a 6.94%
   907	enhancement in mAP, and even a 7.48% rise in the mINP.
   908	In the Indoor-Search scenario, although the improvements
   909	are not as substantial as those in the All-Search scenario,
   910	due to the reduced complexity of the task, the Rank-20
   911	accuracy reaches 99.28%. This suggests that our model is
   912	almost certainly capable of retrieving samples of the same
   913	class within the top 20 results. Additionally, the Rank-1, mAP,
   914	and mINP reach 69.34%, 74.97%, and 71.02% respectively,
   915	demonstrating the robustness of our model in the face of noise.
   916	4) Results on Random Noise: Given that the label noise in
   917	DART is generated randomly, we also present the robustness
   918	results under the condition of random noise. As illustrated
   919	
   920	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:50 UTC from IEEE Xplore. Restrictions apply.

codex
我按本地 PDF 的方法段、消融和可视化段整理。没有额外联网，因为这 5 篇内部的相关工作和证据已经足够拆方法论。

**1. UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval**

1. **触发观察**  
作者注意到一个反直觉点：在文本行人检索里，全量微调 CLIP 不一定是最好的。全量微调虽然能适配训练集，但会破坏或遗忘 CLIP 原本的大规模视觉语言知识，而且每个数据集都要存一套大模型。相反，参数高效微调保留 CLIP 主干，可能同时利用“CLIP 里已有的行人相关知识”和“训练集学到的任务知识”。第二个观察更具体：把 Prefix、LoRA、Adapter 直接堆在一起会互相干扰，普通 Prefix 在 TPR 上收敛慢、效果差。

2. **重定义动作**  
它没有把问题讲成“怎样加一个更强的跨模态模块”，而是改写成“怎样在不破坏 CLIP 原知识的前提下，把全局、局部、任务特定知识一起迁移到 TPR”。关键词是 knowledge retention、TPR-specific knowledge within CLIP、unified PETL、component conflict、global/local/task-specific knowledge。这个重定义把“省参数”从工程优点提升成方法必要性：少调参数不是为了便宜，而是为了保留预训练知识。

3. **机制怎么长出来**  
机制基本是从重定义推出来的。既然要保留 CLIP，就冻结主干；既然要同时迁移不同层次知识，就把不同 PETL 放到不同位置。LoRA 改 attention 的 key/value 权重，负责细粒度局部关系；S-Prefix 给 attention 加任务提示，并用可学习缩放因子解决 Prefix 梯度小、收敛慢的问题；L-Adapter 放在 LayerNorm 旁边调全局分布，避免和 MHA、MLP 里的 LoRA、Prefix 抢同一优化空间。绑定比较紧，但 LoRA 本身不是新东西，新意主要在“每个组件对应一种知识迁移角色”和“避免组件冲突”。

4. **证据闭环**  
关键证据不是单纯 SOTA 表，而是组件消融和冲突消融。单独的 LoRA、S-Prefix、L-Adapter 都比 zero-shot CLIP 强，三者组合在 CUHK-PEDES 上达到最高 R@1 和 mAP，说明“全局、局部、任务提示”确实互补。S-Prefix 对 vanilla Prefix 有明显提升，并且缩放因子过大时会梯度爆炸，证明他们不是随便加参数，而是在修 Prefix 梯度问题。L-Adapter 对比 vanilla Adapter、LN-tuning 和混合 Adapter，支持“普通 Adapter 会和其他 PETL 冲突，LayerNorm 旁边更适合调全局分布”的说法。可视化里 UP-Person 比全量微调的 IRRA 更能识别细粒度或少见描述，也服务于“全量微调丢失 CLIP 原知识”的叙事。

5. **reviewer 为什么买账**  
这篇卖的主要是视角，其次才是机制。机制都是常见 PETL 家族，但作者把“参数高效”讲成了“知识保留和知识迁移结构化分工”，这比“我少调参数还涨点”高级。reviewer 会买账，是因为它把工程约束、过拟合风险、预训练知识保留、组件冲突和消融证据连成了一条线。

**2. Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement**

1. **触发观察**  
作者注意到 VI-ReID 里大家常说“模态差异”，但实际共享特征里混着不止一种干扰。身份信息和相机差异、视角、光照、背景、模态冗余纠缠在一起。已有方法多半只做单阶段解耦或只盯模态差异，不能同时处理相机变化和可见光红外差异。还有一个具体观察是 Instance Normalization 能缓解模态差异，但会丢掉身份信息。

2. **重定义动作**  
它把旧问题“减小 visible-infrared modality gap”改写成“对共享特征做层级净化”。关键词是 hierarchical feature decoupling、feature purification、camera-modality-identity、camera-irrelated features、identity-related residual、redundant features、cross-modality recombination。这样一来，问题不再是单个模态对齐损失，而是先把相机干扰拿掉，再处理模态冗余，再把被归一化丢掉的身份信息补回来。

3. **机制怎么长出来**  
机制和重定义绑定得很直接。第一层 CDM 用通道掩码把特征分成 camera-related 和 camera-irrelated，分别用相机标签、身份标签和对抗约束来分开。第二层 MIDM 对 camera-irrelated 特征做 IN，得到模态更稳定的特征，但承认 IN 会丢身份，于是把原特征和 IN 后特征的残差拿出来，再拆成正残差和负残差。正残差和归一化特征重组成身份特征，负残差重组成冗余特征。然后用 adversarial decoupling loss 让身份特征类内更紧、类间更远，让冗余特征相反。最后 FCRS 把同身份不同模态的正残差交叉重组，用中心约束进一步逼模型关注跨模态共享身份信息。

4. **证据闭环**  
最关键的是 MIDM 单独用时会掉点或不稳，加入 adversarial decoupling loss 后提升，说明“只做掩码或 IN 不够，残差需要被正确分离”。SYSU-MM01 和 RegDB 的消融把 CDM、MIDM、对抗解耦和重组中心损失逐步加上，形成层级收益链。热力图也对准了重定义：IN 后特征激活区域有限，说明确实丢身份；负残差重组后更多激活背景，说明冗余被分出去；身份特征更多聚焦人体区域。t-SNE 从原始特征、AGW 基线、CDM 后特征到 MIDM 后特征逐步更紧、更可分，正好证明“层级净化”不是只在指标上有效。

5. **reviewer 为什么买账**  
这篇卖的是机制和问题拆分的结合。它没有提出一个全新外部信息源，而是把“模态差异”细分成相机、模态、身份冗余三层污染。reviewer 买账的点在于机制路径清楚，消融和可视化都能对应每一层的作用。但它的新颖性边界也比较窄，因为解耦、IN、GRL、残差重组都不是新概念，真正的 novelty 来自层级组织方式。

**3. Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders**

1. **触发观察**  
作者抓住的是 CLIP-ReID 类方法里的一个弱点：两阶段生成的 text prompts 往往不关注行人身份细节。原因是原始 CLIP 对 VI-ReID 不熟，直接生成的提示更偏通用语义，不一定关心跨模态行人身份。另一个观察是，如果同时优化 prompts 和 image encoders，二者目标不同，会互相干扰。

2. **重定义动作**  
它把 prompt 从“给图像编码器用的静态文本辅助”改写成“可学习的中间模态”。关键词是 mutual promotion、mutual reinforcement、person-related prompts、text features as bridge、modality knowledge propagation、modality alignment、alternating training。旧问题是“怎么用文本帮助可见光和红外对齐”，新问题是“提示和图像编码器怎样互相变好”。

3. **机制怎么长出来**  
机制很自然。既然 prompts 缺少身份信息，就让训练好的图像编码器把身份知识传给 prompts。作者用分类概率分布里的 dark knowledge，通过 KL 让文本特征的类别概率接近图像特征的类别概率。既然高质量 prompts 可以作为模态无关桥梁，就在优化图像编码器时，把可见光和红外图像特征都对齐到文本特征。既然同时优化会互相干扰，就交替训练：偶数 epoch 固定图像编码器训 prompts，奇数 epoch 固定 prompts 训图像编码器。这个设计和“互相强化”的叙事绑定很紧。

4. **证据闭环**  
它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。

5. **reviewer 为什么买账**  
这篇主要卖视角。模块上就是 KL 蒸馏、对齐损失、交替优化，但它把 prompt 的角色从“静态文本”翻成“被图像编码器教育、再反过来教育图像编码器的中间模态”。这比单纯说“引入文本提示”更像一个方法故事。reviewer 买账的原因是机制简单，问题清楚，证据能直接回答“prompt 是否真的更关注身份”。

**4. Visible-Infrared Person Re-Identification With Real-World Label Noise**

1. **触发观察**  
作者注意到 VI-ReID 真实标注噪声被低估了。已有 DART 讨论过噪声，但主要是随机换标签。真实 VI-ReID 里噪声更复杂：同模态身份相似会标错，红外缺少颜色和纹理导致跨模态难匹配，标注者也会粗心。随机噪声不能代表这些情况，算法合成的 instance-dependent noise 也不能完全模拟真实标注错误。

2. **重定义动作**  
它把旧问题“VI-ReID 要抗模态差异”改写成“VI-ReID 要抗真实身份标签噪声和检索对应关系噪声”。关键词是 real-world label noise、instance-dependent label noise、SYSU-Real、noisy correspondence、clean subset、indistinguishable noisy subset、obvious noisy subset、Robust Hybrid Loss。最重要的重定义动作是把噪声样本分成三种，而不是把所有低可信样本一刀切。

3. **机制怎么长出来**  
机制几乎是从三类噪声定义直接长出来的。先用 co-training 和 warm-up 利用 early learning，小损失样本更可能干净。再用 GMM 拟合单样本 CE loss，把样本分成 clean、indistinguishably noisy、obviously noisy。干净样本用 CE，因为标签可信；难分辨噪声用 GCE，降低噪声样本的大损失偏置；明显噪声直接不信原标签，用分类置信度构造软伪标签。检索侧还把样本对按置信度划成 clean/noisy correspondence，修正 triplet 里的正负关系，形成 robust retrieval loss。最后把识别损失和检索损失混成 RHL。

4. **证据闭环**  
这篇的证据闭环很强，因为它先造了 SYSU-Real，报告真实噪声比例是 23.79%，并且红外内部噪声比例略高，这直接支撑“真实噪声值得研究”。然后它在 random noise、IDN 和 SYSU-Real 三种条件下都测，证明不是只吃某一种噪声。RHL 消融里，加入处理 indistinguishable noisy subset 的 GCE 带来最大提升，完整方法在 SYSU-Real 上比 DART 提升 7.23% Rank-1、6.94% mAP 和 7.48% mINP，正好支持“三类样本要分治”。可视化也服务于重定义：不同噪声类型的退化柱状图说明 IDN 和真实噪声比随机噪声更伤；相似度矩阵差异图显示 RHL 提高类内相似、降低类间相似；t-SNE 显示比 DART 聚类更清楚。

5. **reviewer 为什么买账**  
这篇卖的第一是问题，第二才是方法。RHL 的零件不稀奇，GMM、小损失原则、GCE、co-training 都有先例。但“真实 VI-ReID 标签噪声”这个问题更接地气，并且作者实际重标了 SYSU-Real，再证明随机噪声和真实噪声不是一回事。reviewer 会买账，是因为它不是只说现实复杂，而是把现实复杂性落实成数据、噪声分类、损失分治和可视化证据。

**5. When Large Vision-Language Models Meet Person Re-Identification**

1. **触发观察**  
作者注意到 LVLM 很强，但直接用于 ReID 很别扭。让 LVLM 逐个判断 query 和 gallery 是否同人，成本和 token 数都不可接受。让 LVLM 生成 caption 再做 image-text matching 也有问题：高质量文本标注贵，caption 目标和 ReID 目标不一致，推理阶段通常没有真正利用 LVLM 的生成能力。CLIP-ReID 类方法虽然用了文本，但往往训练后丢掉文本编码器。

2. **重定义动作**  
它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。

3. **机制怎么长出来**  
机制也比较自然。为了不逐对比对，就仍然保留 ReID 的特征提取加相似度检索范式。为了用 LVLM 的语义理解能力，就用指令让 Qwen2-VL 针对行人图像生成一个 `<REID>` 语义 token，要求关注年龄、性别、衣着、生物特征。为了让这个 token 不只是孤立文本输出，设计 SGI，把语义 token 和视觉 tokens 拼起来经过 4 层 Transformer 交互，最后用强化后的语义 token 作为身份表示。为了处理相机变化，又加 camera semantic supplementation，最好的是把相机 embedding 加到视觉 patch embedding 上。训练用 ID loss 和 triplet loss，LLM 冻结但允许梯度穿过它更新视觉模型和 SGI。

4. **证据闭环**  
关键消融是三组。第一，只用 LVLM 的 visual model 做 baseline 效果低，说明单纯视觉 tokens 不够。第二，把 LVLM 生成的 semantic token 换成随机 learnable token 后明显掉点，说明 `<REID>` token 的语义来源有用，不只是多了一个 class token。第三，去掉 SGI 后效果也掉，说明 token 需要和视觉 tokens 交互，而不是直接拿生成 token 检索。CSS 的消融显示把相机语义放到输入 patch embedding 比放到 `<REID>` token 后面更好，说明相机偏移要早处理。检索可视化显示 baseline 容易被颜色、衣服、姿态误导，而语义 token 引导后能抓住图案、配饰和遮挡下的细节。推理时间从 784ms 到 906ms 处理 64 张图，用来缓解“LVLM 太慢”的质疑。

5. **reviewer 为什么买账**  
这篇卖的是视角，非常明显。它不是一个小模块故事，而是“LVLM 如何进入 ReID 范式”的第一类尝试。真正 novelty 是把生成式 LVLM 压缩成一个可检索的身份语义 token，并且训练和推理都使用它。风险也明显：计算成本高，且“one word/token 能否稳定代表身份”需要更多验证。但作为方法稿，它的视角足够新，reviewer 容易认为这是一个值得建立的 baseline paradigm。

最后总结这批论文的共同套路：它们都不是先从模块出发，而是先把一个常见说法拆开，比如“模态差异”“用 CLIP”“用 prompt”“噪声标签”“用 LVLM”，然后改写成一个更具体的新问题。机制通常只是把这个新问题逐项工程化，最后用消融和可视化证明“新问题的拆法是对的”，而不只是证明模型涨点。
tokens used
130,334
我按本地 PDF 的方法段、消融和可视化段整理。没有额外联网，因为这 5 篇内部的相关工作和证据已经足够拆方法论。

**1. UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval**

1. **触发观察**  
作者注意到一个反直觉点：在文本行人检索里，全量微调 CLIP 不一定是最好的。全量微调虽然能适配训练集，但会破坏或遗忘 CLIP 原本的大规模视觉语言知识，而且每个数据集都要存一套大模型。相反，参数高效微调保留 CLIP 主干，可能同时利用“CLIP 里已有的行人相关知识”和“训练集学到的任务知识”。第二个观察更具体：把 Prefix、LoRA、Adapter 直接堆在一起会互相干扰，普通 Prefix 在 TPR 上收敛慢、效果差。

2. **重定义动作**  
它没有把问题讲成“怎样加一个更强的跨模态模块”，而是改写成“怎样在不破坏 CLIP 原知识的前提下，把全局、局部、任务特定知识一起迁移到 TPR”。关键词是 knowledge retention、TPR-specific knowledge within CLIP、unified PETL、component conflict、global/local/task-specific knowledge。这个重定义把“省参数”从工程优点提升成方法必要性：少调参数不是为了便宜，而是为了保留预训练知识。

3. **机制怎么长出来**  
机制基本是从重定义推出来的。既然要保留 CLIP，就冻结主干；既然要同时迁移不同层次知识，就把不同 PETL 放到不同位置。LoRA 改 attention 的 key/value 权重，负责细粒度局部关系；S-Prefix 给 attention 加任务提示，并用可学习缩放因子解决 Prefix 梯度小、收敛慢的问题；L-Adapter 放在 LayerNorm 旁边调全局分布，避免和 MHA、MLP 里的 LoRA、Prefix 抢同一优化空间。绑定比较紧，但 LoRA 本身不是新东西，新意主要在“每个组件对应一种知识迁移角色”和“避免组件冲突”。

4. **证据闭环**  
关键证据不是单纯 SOTA 表，而是组件消融和冲突消融。单独的 LoRA、S-Prefix、L-Adapter 都比 zero-shot CLIP 强，三者组合在 CUHK-PEDES 上达到最高 R@1 和 mAP，说明“全局、局部、任务提示”确实互补。S-Prefix 对 vanilla Prefix 有明显提升，并且缩放因子过大时会梯度爆炸，证明他们不是随便加参数，而是在修 Prefix 梯度问题。L-Adapter 对比 vanilla Adapter、LN-tuning 和混合 Adapter，支持“普通 Adapter 会和其他 PETL 冲突，LayerNorm 旁边更适合调全局分布”的说法。可视化里 UP-Person 比全量微调的 IRRA 更能识别细粒度或少见描述，也服务于“全量微调丢失 CLIP 原知识”的叙事。

5. **reviewer 为什么买账**  
这篇卖的主要是视角，其次才是机制。机制都是常见 PETL 家族，但作者把“参数高效”讲成了“知识保留和知识迁移结构化分工”，这比“我少调参数还涨点”高级。reviewer 会买账，是因为它把工程约束、过拟合风险、预训练知识保留、组件冲突和消融证据连成了一条线。

**2. Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement**

1. **触发观察**  
作者注意到 VI-ReID 里大家常说“模态差异”，但实际共享特征里混着不止一种干扰。身份信息和相机差异、视角、光照、背景、模态冗余纠缠在一起。已有方法多半只做单阶段解耦或只盯模态差异，不能同时处理相机变化和可见光红外差异。还有一个具体观察是 Instance Normalization 能缓解模态差异，但会丢掉身份信息。

2. **重定义动作**  
它把旧问题“减小 visible-infrared modality gap”改写成“对共享特征做层级净化”。关键词是 hierarchical feature decoupling、feature purification、camera-modality-identity、camera-irrelated features、identity-related residual、redundant features、cross-modality recombination。这样一来，问题不再是单个模态对齐损失，而是先把相机干扰拿掉，再处理模态冗余，再把被归一化丢掉的身份信息补回来。

3. **机制怎么长出来**  
机制和重定义绑定得很直接。第一层 CDM 用通道掩码把特征分成 camera-related 和 camera-irrelated，分别用相机标签、身份标签和对抗约束来分开。第二层 MIDM 对 camera-irrelated 特征做 IN，得到模态更稳定的特征，但承认 IN 会丢身份，于是把原特征和 IN 后特征的残差拿出来，再拆成正残差和负残差。正残差和归一化特征重组成身份特征，负残差重组成冗余特征。然后用 adversarial decoupling loss 让身份特征类内更紧、类间更远，让冗余特征相反。最后 FCRS 把同身份不同模态的正残差交叉重组，用中心约束进一步逼模型关注跨模态共享身份信息。

4. **证据闭环**  
最关键的是 MIDM 单独用时会掉点或不稳，加入 adversarial decoupling loss 后提升，说明“只做掩码或 IN 不够，残差需要被正确分离”。SYSU-MM01 和 RegDB 的消融把 CDM、MIDM、对抗解耦和重组中心损失逐步加上，形成层级收益链。热力图也对准了重定义：IN 后特征激活区域有限，说明确实丢身份；负残差重组后更多激活背景，说明冗余被分出去；身份特征更多聚焦人体区域。t-SNE 从原始特征、AGW 基线、CDM 后特征到 MIDM 后特征逐步更紧、更可分，正好证明“层级净化”不是只在指标上有效。

5. **reviewer 为什么买账**  
这篇卖的是机制和问题拆分的结合。它没有提出一个全新外部信息源，而是把“模态差异”细分成相机、模态、身份冗余三层污染。reviewer 买账的点在于机制路径清楚，消融和可视化都能对应每一层的作用。但它的新颖性边界也比较窄，因为解耦、IN、GRL、残差重组都不是新概念，真正的 novelty 来自层级组织方式。

**3. Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders**

1. **触发观察**  
作者抓住的是 CLIP-ReID 类方法里的一个弱点：两阶段生成的 text prompts 往往不关注行人身份细节。原因是原始 CLIP 对 VI-ReID 不熟，直接生成的提示更偏通用语义，不一定关心跨模态行人身份。另一个观察是，如果同时优化 prompts 和 image encoders，二者目标不同，会互相干扰。

2. **重定义动作**  
它把 prompt 从“给图像编码器用的静态文本辅助”改写成“可学习的中间模态”。关键词是 mutual promotion、mutual reinforcement、person-related prompts、text features as bridge、modality knowledge propagation、modality alignment、alternating training。旧问题是“怎么用文本帮助可见光和红外对齐”，新问题是“提示和图像编码器怎样互相变好”。

3. **机制怎么长出来**  
机制很自然。既然 prompts 缺少身份信息，就让训练好的图像编码器把身份知识传给 prompts。作者用分类概率分布里的 dark knowledge，通过 KL 让文本特征的类别概率接近图像特征的类别概率。既然高质量 prompts 可以作为模态无关桥梁，就在优化图像编码器时，把可见光和红外图像特征都对齐到文本特征。既然同时优化会互相干扰，就交替训练：偶数 epoch 固定图像编码器训 prompts，奇数 epoch 固定 prompts 训图像编码器。这个设计和“互相强化”的叙事绑定很紧。

4. **证据闭环**  
它的核心证据是 baseline、two-stage、MPT w/ loss、完整 MPT 的对比。完整 MPT 在 SYSU-MM01 all-search 上比 baseline 提升 5.45% Rank-1 和 5.88% mAP；在相同结构和损失下，交替训练比 two-stage 更好，说明不是“多一个文本分支”就行，而是 co-evolve 的训练方式有效。再加上 L_MPT 优于普通 contrastive loss，证明他们设计的两个损失确实服务于 mutual reinforcement。可视化也围绕重定义展开：t-SNE 里同身份跨模态更聚；距离分布里类间和类内均值差更大；feature map 显示 MPT 更关注人体区域，支持“prompt 更 person-related”。

5. **reviewer 为什么买账**  
这篇主要卖视角。模块上就是 KL 蒸馏、对齐损失、交替优化，但它把 prompt 的角色从“静态文本”翻成“被图像编码器教育、再反过来教育图像编码器的中间模态”。这比单纯说“引入文本提示”更像一个方法故事。reviewer 买账的原因是机制简单，问题清楚，证据能直接回答“prompt 是否真的更关注身份”。

**4. Visible-Infrared Person Re-Identification With Real-World Label Noise**

1. **触发观察**  
作者注意到 VI-ReID 真实标注噪声被低估了。已有 DART 讨论过噪声，但主要是随机换标签。真实 VI-ReID 里噪声更复杂：同模态身份相似会标错，红外缺少颜色和纹理导致跨模态难匹配，标注者也会粗心。随机噪声不能代表这些情况，算法合成的 instance-dependent noise 也不能完全模拟真实标注错误。

2. **重定义动作**  
它把旧问题“VI-ReID 要抗模态差异”改写成“VI-ReID 要抗真实身份标签噪声和检索对应关系噪声”。关键词是 real-world label noise、instance-dependent label noise、SYSU-Real、noisy correspondence、clean subset、indistinguishable noisy subset、obvious noisy subset、Robust Hybrid Loss。最重要的重定义动作是把噪声样本分成三种，而不是把所有低可信样本一刀切。

3. **机制怎么长出来**  
机制几乎是从三类噪声定义直接长出来的。先用 co-training 和 warm-up 利用 early learning，小损失样本更可能干净。再用 GMM 拟合单样本 CE loss，把样本分成 clean、indistinguishably noisy、obviously noisy。干净样本用 CE，因为标签可信；难分辨噪声用 GCE，降低噪声样本的大损失偏置；明显噪声直接不信原标签，用分类置信度构造软伪标签。检索侧还把样本对按置信度划成 clean/noisy correspondence，修正 triplet 里的正负关系，形成 robust retrieval loss。最后把识别损失和检索损失混成 RHL。

4. **证据闭环**  
这篇的证据闭环很强，因为它先造了 SYSU-Real，报告真实噪声比例是 23.79%，并且红外内部噪声比例略高，这直接支撑“真实噪声值得研究”。然后它在 random noise、IDN 和 SYSU-Real 三种条件下都测，证明不是只吃某一种噪声。RHL 消融里，加入处理 indistinguishable noisy subset 的 GCE 带来最大提升，完整方法在 SYSU-Real 上比 DART 提升 7.23% Rank-1、6.94% mAP 和 7.48% mINP，正好支持“三类样本要分治”。可视化也服务于重定义：不同噪声类型的退化柱状图说明 IDN 和真实噪声比随机噪声更伤；相似度矩阵差异图显示 RHL 提高类内相似、降低类间相似；t-SNE 显示比 DART 聚类更清楚。

5. **reviewer 为什么买账**  
这篇卖的第一是问题，第二才是方法。RHL 的零件不稀奇，GMM、小损失原则、GCE、co-training 都有先例。但“真实 VI-ReID 标签噪声”这个问题更接地气，并且作者实际重标了 SYSU-Real，再证明随机噪声和真实噪声不是一回事。reviewer 会买账，是因为它不是只说现实复杂，而是把现实复杂性落实成数据、噪声分类、损失分治和可视化证据。

**5. When Large Vision-Language Models Meet Person Re-Identification**

1. **触发观察**  
作者注意到 LVLM 很强，但直接用于 ReID 很别扭。让 LVLM 逐个判断 query 和 gallery 是否同人，成本和 token 数都不可接受。让 LVLM 生成 caption 再做 image-text matching 也有问题：高质量文本标注贵，caption 目标和 ReID 目标不一致，推理阶段通常没有真正利用 LVLM 的生成能力。CLIP-ReID 类方法虽然用了文本，但往往训练后丢掉文本编码器。

2. **重定义动作**  
它把 LVLM 从“生成描述的外部工具”改写成“产生身份语义 token 的特征提取器组成部分”。关键词是 LVLM-ReID、Pedestrian Semantic Token Generation、`<REID>` token、Semantic-Guided Interaction、reinforced semantic token、generative language models assist discriminative visual models。旧问题是“怎么给 ReID 加文本语义”，新问题是“怎样把 LVLM 的生成过程变成 ReID 的身份表示”。

3. **机制怎么长出来**  
机制也比较自然。为了不逐对比对，就仍然保留 ReID 的特征提取加相似度检索范式。为了用 LVLM 的语义理解能力，就用指令让 Qwen2-VL 针对行人图像生成一个 `<REID>` 语义 token，要求关注年龄、性别、衣着、生物特征。为了让这个 token 不只是孤立文本输出，设计 SGI，把语义 token 和视觉 tokens 拼起来经过 4 层 Transformer 交互，最后用强化后的语义 token 作为身份表示。为了处理相机变化，又加 camera semantic supplementation，最好的是把相机 embedding 加到视觉 patch embedding 上。训练用 ID loss 和 triplet loss，LLM 冻结但允许梯度穿过它更新视觉模型和 SGI。

4. **证据闭环**  
关键消融是三组。第一，只用 LVLM 的 visual model 做 baseline 效果低，说明单纯视觉 tokens 不够。第二，把 LVLM 生成的 semantic token 换成随机 learnable token 后明显掉点，说明 `<REID>` token 的语义来源有用，不只是多了一个 class token。第三，去掉 SGI 后效果也掉，说明 token 需要和视觉 tokens 交互，而不是直接拿生成 token 检索。CSS 的消融显示把相机语义放到输入 patch embedding 比放到 `<REID>` token 后面更好，说明相机偏移要早处理。检索可视化显示 baseline 容易被颜色、衣服、姿态误导，而语义 token 引导后能抓住图案、配饰和遮挡下的细节。推理时间从 784ms 到 906ms 处理 64 张图，用来缓解“LVLM 太慢”的质疑。

5. **reviewer 为什么买账**  
这篇卖的是视角，非常明显。它不是一个小模块故事，而是“LVLM 如何进入 ReID 范式”的第一类尝试。真正 novelty 是把生成式 LVLM 压缩成一个可检索的身份语义 token，并且训练和推理都使用它。风险也明显：计算成本高，且“one word/token 能否稳定代表身份”需要更多验证。但作为方法稿，它的视角足够新，reviewer 容易认为这是一个值得建立的 baseline paradigm。

最后总结这批论文的共同套路：它们都不是先从模块出发，而是先把一个常见说法拆开，比如“模态差异”“用 CLIP”“用 prompt”“噪声标签”“用 LVLM”，然后改写成一个更具体的新问题。机制通常只是把这个新问题逐项工程化，最后用消融和可视化证明“新问题的拆法是对的”，而不只是证明模型涨点。
