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
session id: 019ef835-c84d-79b3-a23e-20cbbf9cecc5
--------
user
你是 ReID 创新方法论拆解员 17。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(17)==
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则读取当前目标和纪律文件，然后逐篇用 `pdftotext -f 1 -l 8` 抽取前 8 页方法部分。这个任务只需要读文件和必要的背景核对，不会改动工作区。
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
/bin/zsh -lc "sed -n '1,260p' ../../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../CLAUDE.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' ../../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../GOAL.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,240p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

**核心结果(reliable_retrieval_eval.py，zero-training，主表 paper_materials/reliable_retrieval_results.md)**：三 backbone×数据集一致——RCIR 式边际名义覆盖达标但 s_top1 难层严重欠覆盖(α0.2 worst-cov 0.31-0.42)→我们分层 group-conditional 修到 0.76-0.88，候选集几乎不变甚至更小(Occ-PoseTrack α0.1 12.8→7.58)。selective mAP@80 +6.75~+10.1，s_top1 Spearman 0.59-0.75，null 对照(random/s_gap)nAURC≈1 正确。**诚实**:worst-group 覆盖修复=评价粒度/部署风险，非标准 mAP 提升；不吹"超越 RCIR 算法"，写"揭示边际 RCIR 在 ReID 隐藏 worst-group 欠覆盖+标准修复"。

**投稿打磨(2026-06-09 Codex)**：已通读 v2、红队意见和核心结果，写成 `paper_materials/paper_final_polish.md`。已小改 `paper_draft_v2.md` 三处：分组边界改为 calibration split 确定，三种子说明排除 SOLIDER Occ-Duke 单 seed 补充点，表 2 caption 同步。**已补**：codex 写出 `tables.tex`(主表+selective+单点假阳性+hard_expand 代价，未跑格子用 -- 标边界)、`refs.bib`(RCIR/DistributionNet/SSPEM/KPR/QPM/ProFD/k-reciprocal/Vovk/Barber/conformal-risk-control 10 篇)、`appendix_implementation.md`(same_camera/合法 gallery/nonconformity/split-conformal 分位数/4 层分组/20 split/AURC/random null/复现校验)。待补重点剩：图编号 caption 对齐、Market 同口径(下面在跑)、英文化、选会议、投稿格式。

**整夜自主推进中(cron 9859ade0 每30min + 等待器)**：
- **论文素材一致性+诚实性审计已做(已提交 34b50b8)**：codex 审计主表数字全对上源真值、无硬越界；修了图号错位(只3张真图却引用图1-8→归正+未画的标待绘)、tables.tex 无源单 seed 曲线点(删→--、52.60→52.57)、Market 数回填源真值标 exp011 旧口径、两处过强措辞软化、results.md 内部 α 标注矛盾。详见 consistency_audit-codex.md。
- **AURC + 完整 selective 曲线：5 个 backbone×数据集全部捕获(填用户待办#1/#2，已提交 b770118/70b1de0/4582dbd/b5b624b)**：原 eval stdout 没持久化、@80 之外全丢，重跑捕获完整 SELECTIVE 行。SOLIDER Occ-PoseTrack 三种子(AURC 0.113±0.001)、ImageNet-Swin Occ-PoseTrack 单seed(0.133)、ImageNet-Swin Occ-Duke 三种子(0.335)、ViT Occ-Duke 三种子(0.247±0.001)、SOLIDER Occ-Duke 单seed(0.228)。**每项 worst-group/Spearman/no_reject/@80 全对上★主表**(强自洽)。AURC 按难度 0.113-0.335 排序、nAURC 全<1(null random/s_gap nAURC≈1.0-1.5)。tables.tex selective 表 5 行齐、fig3 五条真实多点曲线、日志全存 eval_logs/。
- **exp014 Market 同口径对照三种子定稿(决策#41 完成，已提交 96ee1a9)**：复现通过(三种子训练 mAP 91.6/91.6/91.6 对上 exp000)。**核心结果(三种子均值)**：α0.2 边际 RCIR worst-cov 0.218±0.001→我们分层 0.774±0.005；α0.1 0.605±0.003→0.883±0.001；Spearman 0.364±0.004、selective@80 +3.14、AURC 0.043。即非遮挡 Market 上边际 conformal 条件失效比遮挡(0.30-0.42)更严重(0.218)、标准分层同样修复。**"条件失效非遮挡特有=否"的同口径三种子直接证据**。中途 seed2 等待器抢跑致 OOM 已重启恢复；Occ-Duke AURC 改在空闲 GPU1 即时跑省空转。
- **★ 至此可靠检索线实验数据全部采集完毕**：主 worst-group 表(5 组合三种子)、selective+AURC 表(5 组合完整曲线)、Market 同口径对照(三种子)、单点假阳性两划分两数据集、hard_expand 代价、正交容量律。eval_logs/ 14 个日志入库作证据。3 张配图(fig1 容量律/fig2 worst-group/fig3 selective 5曲线)。论文 v1/v2+final_polish+tables.tex+refs.bib+appendix+一致性审计全在。
- **最终红队(完整论文，redteam_final-codex.md)已做**：核心张力=数据齐全后越诚实技术增量越薄，能站"ReID 可靠检索协议+风险诊断"不能站方法稿。命中重估 **PRCV 0.52-0.66 / ICPR 0.36-0.50 / ICME 0.30-0.42**(数据补全降低"实验不足"风险,但 Market 三种子让"遮挡特有"防线消失,故 ICME/ICPR 不上调,PRCV 略稳)。3 最该补(全措辞/已有数据,无需新实验)：①拆通用 vs 遮挡贡献框架 ②代价进主叙述 ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标)。
- **我已做的非框架补充(已提交)**：hard_expand 表补全 6 组合困难组候选集绝对大小(RCIR 0.3-5.7→我们 4-95,Occ-Duke 偏大但~gallery 0.5%)，回应红队 fix#2 的数据部分。**框架/定位类(拆贡献、降级强说法、叙述 placement、选会议、英文化、投稿)全留用户决策，不擅自改大方向**。
- **conformal 方法学有效性审查(conformal_validity_review-codex.md)已做，关键正面结论**：以 conformal 理论审稿人角度逐行审 reliable_retrieval_eval.py，**无代码硬伤**，核心诊断(边际达标但 s_top1 难层欠覆盖、标准分组可修复)在实现层站得住；每组 calibration query ~276-421，worst-group 0.218/0.88 不是小样本崩坏；覆盖数字全对上日志。需收窄/限定的全是理论口径(身份可交换+固定gallery假设、RCIR-style marginal 非原文、worst 是4预固定组经验最小值、α0.2 worst 0.76-0.77 写"接近")，多数 draft 已做。总评：作经验诊断+标准分组校准闭环可投稿，作"无条件分布无关 conformal 定理/超越RCIR新算法"站不住=正是诚实定位。
- **★ 这条线的自主可做事项已全部完成**：实验数据全齐+论文素材全齐+一致性审计+两轮独立红队(通用CCF-B novelty + conformal理论validity，都无硬伤)+代价绝对量级数据。剩余纯属用户战略决策。三台机器实验程序已完，正确闲置(无非冗余线内活)。
- **④四组覆盖表已补(2026-06-09,提交 1c3e967)**：给 eval 加 GROUPS 打印(只输出已聚合的逐层覆盖/样本量,不改计算),重跑 4 代表组合,RCIR worst-group+SELECTIVE 与重跑前完全一致。新机制图景:边际 RCIR 把易层过覆盖到 0.99 凑总体 0.80、难层只 0.22-0.42,我们分组校准四组全拉到 ~0.80;每组 calib 276-422 充足(非小样本噪声)。见 results.md"四组逐层覆盖"节。
- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。

---

## ★★★ 旧（2026-06-08 08:25，选项1 执行中）★★★

codex
`experiments/STATUS.md` 已读到，当前项目主线是从论文库里拆别人怎么构造方法创新，这和你的任务一致。我刚才把根目录层级多退了一层，接下来按正确路径补读 `GOAL.md` 和 `CLAUDE.md`，然后开始抽 PDF 方法部分。
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
/bin/zsh -lc 'which pdftotext' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,260p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/opt/homebrew/bin/pdftotext

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

codex
我现在先定位每篇的 `Method` 或 `Approach` 边界，避免只看摘要。随后会逐篇读从方法开头到实验前后的完整段落，重点抓“观察怎么变成问题、问题怎么推出机制、证据怎么把这个问题讲实”。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
237:Methodology
252:Our Proposed MMET Framework
306:Algorithm 1: The Training Procedure of Our Proposed MMET Method

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
55:1. Introduction
139:2. Related work
304:3. Methodology
465:4. Experiment

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
230:A. Overall Framework

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
296:3 The Proposed Method

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
195:Figure 2: Overview of Our Proposed LVI-ReID Method. Initially, the input image (visible or infrared) is processed into image

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | rg -n \"("'^[0-9]+'"\\.?\\s+|"'^[IVX]+'"\\.?\\s+)(Introduction|Related|Method|Approach|Proposed|Experiment|Experiments|Evaluation|Methodology|Framework|Preliminary|Problem)|Methodology|Proposed|Framework\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
70:1. Introduction
175:2. Related work
280:3. Methodology
988:4. Experiments

codex
方法边界已经基本找到了：前三篇是标准 `Methodology` 结构，后几篇的标题格式不统一，需要按正文行号截取。下面我会分批读取，避免一次把六篇的全文混在一起。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf' - | nl -ba | sed -n '220,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   220	importantly, these methods trend to ignore the challenge where one modality or more modalities
   221	are missing. Consequently, current methods are still far from satisfactory in generalization for
   222	practical person Re-ID.
   223	To solve these problems, in this work, we take a big step forward and design a novel MMET
   224	framework for more robust visual-semantic embedding learning on visual, textual, and visualtextual tasks. On the basis of it, a dynamic masking mechanism called MMM is introduced to mask
   225	both the image patches and the text tokens, which can jointly work on multimodal or unimodal
   226	data and significantly boost the performance of generalizable person Re-ID. It is worth mentioning
   227	that our MMET model is designed to be able to take advantage of unpaired image data or text
   228	data along with paired image-text pairs. While previous methods fail to adapt in the multimodal
   229	scenarios, especially for those cross-modal and multimodal vision-and-language tasks. To the best
   230	of our knowledge, this is the first attempt to adopt a foundational language and vision alignment
   231	model that explicitly targets vision, language, and their multimodal combination all at once. We
   232	hope that our method will serve as a strong baseline for visual-semantic embedding, and shed light
   233	into potential tasks for the community to move forward.
   234	3
   235	3.1
   236	
   237	Methodology
   238	Problem Formulation
   239	
   240	We begin with a formal description of the DG Re-ID problem. We assume that we are given
   241	𝐾
   242	K source domains D = {D𝑘 }𝑘=1
   243	. Each source domain contains its own image-label pairs D𝑘 =
   244	 𝑘 𝑘 𝑘  N𝑘
   245	𝒙 𝑖 , 𝒙 𝑡 , 𝑦𝑖 𝑖=1 , where N𝑘 is the number of images in the source domain D𝑘 . Each sample 𝒙 𝑘𝑖 ∈ X𝑘
   246	is associated with an identity label 𝑦𝑖𝑘 ∈ Y𝑘 = {1, 2, . . . , M𝑘 } and text caption 𝒙 𝑘𝑡 ∈ X𝑡 , where M𝑘 is
   247	the number of identities in the source domain D𝑘 . During the training phase, we train a DG model
   248	using the aggregated image-label pairs of all source domains. In the testing phase, we perform a
   249	retrieval task on unseen target domains without additional model updating.
   250	3.2
   251	
   252	Our Proposed MMET Framework
   253	
   254	In this section, we propose a MMET network for multimodal feature learning, which can be flexibly
   255	adapted to the visual task, textual task, and visual-textual task, respectively. In fact, traditional
   256	single-stream structure requires early fusion of the two modalities, which cannot be directly
   257	adapted in the scenario where the attributes of the benchmark dataset are missing. To address
   258	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   259	
   260	278:6
   261	
   262	S. Xiang et al.
   263	
   264	Fig. 2. The framework of our proposed MMET method, which contains: (1) Image encoder to capture unimodal
   265	image representations; (2) Text encoder to process unimodal text information; (3) Multimodal encoder that
   266	takes as input the encoded unimodal image and text and integrates their representations for multimodal
   267	reasoning.
   268	
   269	this problem, we introduce MMET, a foundational language and vision alignment model that
   270	explicitly targets unimodal vision-only or language-only, and their multimodal combination all at
   271	once for generalizable person Re-ID event, which contains Image encoder, Multi-modal encoder, and
   272	Multi-modal encoder, the detailed architecture of our approach is illustrated in Figure 2.
   273	Image Encoder. The image encoder mainly extracts visual model features from two-dimensional
   274	image input, and the structure still adopts the basic construction of visual transformers. Given an
   275	input image, we resize it to a fixed size, and divide the image into M mean image patches, which
   276	are then linearly projected through linear projection layer, with bit-set padding and an extra image
   277	classification token [CLS_I] is sent to the Transformer layer. The output of the image encoder is a
   278	sequence of hidden state vectors, which can be denoted as set {ℎ𝐼 }. Each of these corresponds to
   279	an image patch, and there is an additional global feature output ℎ𝐶𝐿𝑆_𝐼 corresponding to the image
   280	classification token [CLS_I].
   281	Text Encoder. In order to ensure the overall balance of the MMET model, the text encoder here is
   282	different from the general model in the field of natural language processing. Instead, we basically
   283	adopt the same architecture as the visual Transformer, only the parameters are different. Giving the
   284	caption from a pedestrian, we first tokenize the text according to the word sentence. Continuously,
   285	we embed it into the word vector sequence according to the BERT model [5], then we apply a
   286	transformer model over the word vectors to encode them into a list of hidden state vectors {ℎ𝑇 },
   287	including ℎ𝐶𝐿𝑆_𝑇 for the text classification [CLS_T] token.
   288	Multi-Modal Encoder. Different from the general two-stream structure, in this work, we adopt a
   289	separate Transformer to fuse the hidden state sequence of images and text, as shown in Figure 2.
   290	Specifically, we apply two linear maps to each hidden state vector in the sets of {ℎ𝐼 } and {ℎ𝑇 },
   291	which are then concatenated into a full sequence while adding an additional token [CLS_M]. As
   292	shown in Figure 2, this concatenated list is fed into the multimodal encoder transformer, allowing
   293	cross-attention between the projected unimodal image and text representations and fusing the two
   294	modalities. It is worth mentioning that the output value of this part is also a set of hidden state
   295	sequences ℎ𝑀 , each of these corresponds to a unimodal vector from ℎ𝐼 or ℎ𝑇 , and a vector ℎ𝐶𝐿𝑆,𝑀
   296	corresponding to [CLS_M].
   297	To be more specific, masked image modeling L M I M loss [1] and masked language modeling
   298	L M L M loss [5] are applied onto the image and text encoders over a single image or a text piece,
   299	respectively, while MMM L M M M and image-text matching L I T M are used over paired image-text
   300	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   301	
   302	Learning Visual-Semantic Embedding for Generalizable Person Re-identification
   303	
   304	278:7
   305	
   306	Algorithm 1: The Training Procedure of Our Proposed MMET Method
   307	
   308	data [26, 35], as illustrated in Figure 2. And Global Contrastive (GC) loss L G C resembles that of
   309	CLIP [33] during the pretraining process. The detailed information of masked image modeling and
   310	masked language modeling is formulated as follows:
   311	
   312	
   313	LMIM = E (𝐼,𝑇 )∼𝐷 ℓmim 𝑓𝜃 ( ˆ𝐼,𝑇 ), 𝑓𝜃¯𝑣 (𝐼,𝑇 ) ,
   314	(1)
   315	
   316	
   317	LMLM = E (𝐼,𝑇 )∼𝐷 ℓmlm 𝑓𝜃 (𝐼, 𝑇ˆ ),𝑇 ,
   318	(2)
   319	 
   320	 
   321	ˆ
   322	where (𝐼,𝑇 ) denotes the image-text pair, 𝐼,𝑇
   323	and 𝐼, 𝑇ˆ represent two masked views by randomly
   324	masking a portion of the input, which are then sent into our network f parameterized by 𝜃 , to get
   325	their multimodal representations.
   326	Network Updating. In this work, we pretrain the entire model from scratch with synthetic dataset
   327	FineGPR [49], which contains more than 2 million image-text pairs. For the generalizable person
   328	Re-ID task, as shown in Algorithm 1, we apply the classifier head on top of the multimodal ℎ𝐶𝐿𝑆,𝑀
   329	from the multi-modal encoder during training phase. Similarly, for visual recognition and language
   330	understanding tasks, we apply a classifier head on top of ℎ𝐶𝐿𝑆,𝐼 from image encoder and ℎ𝐶𝐿𝑆,𝑇 from
   331	text encoder, respectively. During the model optimization phase, we pretrain the MMET model
   332	once, and evaluate it separately on each downstream Re-ID task. For a single-label N classification
   333	task, the identification loss (cross-entropy loss) is written as
   334	L𝐼 𝐷 = −
   335	
   336	1
   337	𝑀𝑏𝑎𝑡𝑐ℎ
   338	
   339	𝑀Õ
   340	𝑏𝑎𝑡𝑐ℎ 𝑁
   341	Õ
   342	𝑖=1
   343	
   344	𝑦𝑖 𝑗 log 𝑦ˆ𝑖 𝑗 ,
   345	
   346	(3)
   347	
   348	𝑗=1
   349	
   350	where 𝑀𝑏𝑎𝑡𝑐ℎ is the number of labeled training images in a batch, 𝑦ˆ𝑖 𝑗 is the predicted probability
   351	of the input belonging to ground-truth class 𝑦𝑖 𝑗 . In essence, many previous works [27, 45] have
   352	been found that performing training with multiple losses has great potential to learn a robust and
   353	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   354	
   355	278:8
   356	
   357	S. Xiang et al.
   358	
   359	Fig. 3. An example of masked image with our masked modality modeling strategy, in which we mainly mask
   360	the pedestrian region instead of whole part of input images.
   361	
   362	generalizable Re-ID model. Inspired from this, we also adopt triplet loss to mine the relationship
   363	of training samples during training, which can minimize the distance among positive pairs and
   364	maximize the distance between negative pairs. And our triplet loss is defined as
   365	
   366	L𝑡𝑟𝑖𝑝𝑙𝑒𝑡 = 𝑑𝑎,𝑝 − 𝑑𝑎,𝑛 + 𝑚 + ,
   367	(4)
   368	where 𝑑𝑎,𝑝 , 𝑑𝑎,𝑛 denote the feature distances of positive pair and negative pairs, respectively, m
   369	represents the margin of our triplet loss, (𝑧)+ denotes max(z,0). To this end, we propose to jointly
   370	learn robust visual-semantic embedding using classification loss [57] and Triplet loss [15] in a
   371	training batch, which can be expressed as
   372	L𝑡𝑜𝑡𝑎𝑙 = L𝐼 𝐷 + L𝑇 𝑟𝑖𝑝𝑙𝑒𝑡 .
   373	3.3
   374	
   375	(5)
   376	
   377	MMM Strategy
   378	
   379	In fact, most of the previous visual-language modeling approaches focus on the masked language
   380	modeling of the texts in multimodal input by reconstructing the mask labeling unit, while neglecting
   381	the mask feature learning at the pixel level for single image modality. In this work, we introduce
   382	a novel MMM strategy to perform pretraining based on the MMET model, which can effectively
   383	achieve the goal of performing mask loss calculations on image and text modality.
   384	As shown in Figure 3, we mainly mask the pedestrian region instead of whole part of input
   385	images, since the human part contains more discriminative feature for representation learning, and
   386	it has no effect on the prediction performance of the model when the masked region mainly focus on
   387	the background of pedestrian samples. Motivated from this, we propose a dynamic strategy called
   388	MMM, which can be significantly applied to pretraining tasks on text encoders for monolingual
   389	datasets. Following the default settings in the field of natural language processing task [5], here we
   390	also randomly mask a certain fraction of image blocks and text labeling units, respectively. Then
   391	we adopt the classifier to reconstruct from other image blocks and tokenized units on the hidden
   392	state output ℎ𝑀 of the image-text modality. In contrast to denoising auto-encoders [38], we only
   393	predict the masked image and words rather than reconstructing the entire text inputs.
   394	During the pretraining, masked image modeling is applied into the image encoder over unpaired
   395	image data, while masked language modeling is applied into the text encoder over unpaired text
   396	data. Specifically, given an image and text data, we perform masking on image encoder and text
   397	encoder, respectively. Finally, we add an image-text matching loss L I T M following prior visionand-language pretraining literature [26]. On the basis of it, we introduce Image-Text Matching
   398	pretraining task to learn the relationship between image and text samples, and then apply a classifier
   399	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   400	
   401	
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '285,475p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   285	new data, we propose an adaptability retention (AdRet) strategy. This
   286	strategy employs a knowledge distillation loss to speciﬁcally train the
   287	parameter predictor within the DyAdap module, constraining it to adapt
   288	to new training data while maintaining its ability to adapt to previously
   289	trained domains.
   290	
   291	3.1. Problem deﬁnition and formulation
   292	
   293	3.3. Dynamic adaptation module
   294	
   295	For clariﬁcation, we ﬁrst give the deﬁnition of LReID task. Assume
   296	the sequence of datasets is denoted as  = {𝑇1 , 𝑇2 , … , 𝑇𝑚 }. The ReID
   297	model is required to perform continuous training starting from 𝑇1 and
   298	sequentially on each subsequent dataset. After each training phase, the
   299	
   300	In our DKAR model, we ﬁrstly design a dynamic adaptation module to adapt our model to diverse domains. In the DyAdap module as
   301	shown in Fig. 3, we ﬁrstly employ an instance normalization layer to
   302	explicitly eliminate the feature distribution divergence across diﬀerent
   303	
   304	3. Methodology
   305	
   306	4
   307	
   308	Neural Networks 198 (2026) 108566
   309	
   310	Z. Chen et al.
   311	
   312	domains, thereby facilitating the adaptation of our model. In addition
   313	to the instance normalization layer, in our DyAdap module, we deploy
   314	a dynamic convolutional layer, which receives the domain-aware information within input image features as guidance to ﬂexibly adapt our
   315	model toward diverse domains. Practically, given input features 𝑓 , we
   316	ﬁrst send it into an instance normalization (IN) layer. The IN is a widely
   317	adopted distribution alignment strategy. It achieves feature distribution
   318	alignment across diﬀerent domains by calculating and normalizing the
   319	statistics of each channel in the input image features. Formally, the process of instance normalization could be written as,
   320	𝑓 − 𝜇(𝑓 )
   321	𝑓′ =
   322	(1)
   323	𝜎(𝑓 )
   324	𝑓𝑖𝑛 = 𝛾𝑓 ′ + 𝛽
   325	
   326	(2)
   327	
   328	where 𝑓 represents features before being instance normalized, 𝑓𝑖𝑛 represents normalized features, 𝜇 and 𝜎 represent channel-wise mean and
   329	standard deviation of 𝑓 , 𝛾 and 𝛽 are aﬃne parameters. Thereafter, we
   330	send the normalized feature 𝑓𝑖𝑛 into a dynamic convolution layer. Specifically, there are two major modules inside the dynamic convolution
   331	layer, namely, a parameter generator and a convolutional module. The
   332	parameter generator is designed to generate dynamic parameters based
   333	on the input features. In terms of structure, the parameter generator
   334	consists of two fully connection layers inserted with a ReLU activation
   335	function. Formally, the dynamic parameters could be summarized as,
   336	𝜔 = 𝑓 𝑐(𝑟𝑒𝑙𝑢(𝑓 𝑐(𝑝𝑜𝑜𝑙((𝑓 )), 𝜃1 )), 𝜃2 ),
   337	
   338	(3)
   339	
   340	where 𝜔 indicates generated dynamic parameters; 𝑓 𝑐(, 𝜃𝑖 ) represents
   341	fully connection layer. Notably, the parameter predictor receives the
   342	features before normalization layer as inputs, which contains more instance and domain-aware information. After that, we feed the normalized feature and the generated dynamic parameter into a convolution
   343	module, which could be written as,
   344	𝑓 𝐷𝑦𝐴𝑑𝑎𝑝 = 𝑐𝑜𝑛𝑣(𝑓 𝐼𝑁 ; 𝜔)
   345	
   346	Fig. 3. The illustration about the DyAdap model, which combines instance normalization used to eliminate the inﬂuence brought by feature distribution divergence across domains with dynamic convolution used to adapt our model
   347	toward diverse domains.
   348	
   349	(4)
   350	
   351	where 𝐹 𝐷𝑦𝐴𝑑𝑎𝑝 is the output of DyAdap module; 𝑐𝑜𝑛𝑣 represents convolution operation; 𝜔 denotes generated dynamic parameters. Through
   352	such dynamic operation, our model could ﬂexibly adapt to diverse target
   353	domains. In terms of structure, we use our DyAdap module to replace the
   354	3 × 3 convolutional layer inside each residual block in resnet-50 model
   355	to achieve adaptation.
   356	The DyAdap module integrates dynamic convolution structures into
   357	traditional models, endowing them with the capability to dynamically
   358	adjust parameters in response to data from unseen domains. This dynamic convolution structure allows the DyAdap module to leverage
   359	knowledge acquired from prior domains, facilitating adaptive parameter
   360	adjustments to generate an optimal feature distribution for the current
   361	domain during training across diverse datasets. However, during lifelong learning across multiple domains, signiﬁcant updates occur within
   362	the model’s feature extraction network, potentially resulting in the loss
   363	of feature extraction capabilities learned from previous domains. Although IN method enhances the model’s domain generalization, the hypernetwork structure within the DyAdap module, responsible for generating dynamic convolution parameters, remains limited by the distribution of the current training domain. Consequently, both the feature
   364	extraction network and the DyAdap module are susceptible to catastrophic forgetting. In essence, the DyAdap module does not directly mitigate catastrophic forgetting but instead improves the model’s dynamic
   365	adaptability and domain generalization to a certain extent. Although
   366	this enhancement does not fundamentally resolve the issue of knowledge retention across domains, it establishes a structural foundation for
   367	achieving optimal performance in various domains, particularly in ReID
   368	models.
   369	
   370	Retention (AdRet) strategy. Speciﬁcally, we apply knowledge distillation losses to the features processed in the DyAdap module, as well as
   371	to the predicted parameters generated by the module. This ensures that
   372	the knowledge learned in the old domain and the knowledge in the current domain jointly guide the training process, allowing the model to
   373	eﬀectively adapt back to previously seen domains.
   374	Practically, in our model, during each training iteration, we preserve
   375	the version of the model trained on the last domain as a teacher network,
   376	which is frozen and used to guide the current model in adapting back
   377	to previously seen domains. Let the model being trained in the current
   378	iteration be denoted as Φ𝑡 , where 𝑡 indicates the index of the current
   379	training stage. The preserved teacher model is then denoted as Φ𝑡−1 .
   380	For each image, we simultaneously feed it into Φ𝑡 and Φ𝑡−1 , guiding the
   381	predictions of Φ𝑡 to align with those of Φ𝑡−1 . This ensures that our model
   382	retains the ability to adapt back to the previous domain. Speciﬁcally,
   383	during the training of Φ𝑡 , the features input to the parameter predictor of
   384	the DyAdap module are guided to align with the corresponding features
   385	at the same position in Φ𝑡−1 . This ensures that our model achieves a
   386	balance between retaining the ability to adapt to previous domains and
   387	learning new knowledge. Given the features extracted by Φ𝑡 and Φ𝑡−1 ,
   388	we ﬁrst apply spatial average pooling to them. The pooled features are
   389	denoted as 𝑓 𝑡 ∈ ℝ𝐶 and 𝑓 𝑡−1 ∈ ℝ𝐶 , where 𝐶 is the channel dimension.
   390	The knowledge distillation applied to these features can then be expressed as:
   391	.𝐸
   392	𝐿𝐹𝐾𝐷
   393	=
   394	
   395	𝐶
   396	∑
   397	𝑖=1
   398	
   399	𝑓𝑖𝑡 log
   400	
   401	𝑓𝑖𝑡
   402	𝑓𝑖𝑡−1
   403	
   404	(5)
   405	
   406	.𝐸 indicate the knowledge distillation loss; the 𝑓 𝑡 indicates
   407	where the 𝐿𝐹𝐾𝐷
   408	𝑖
   409	the element within 𝑖-th channel of 𝑓 𝑡 .
   410	In addition to applying knowledge distillation loss to the features fed
   411	into the parameter generator, we also apply a parameter regularization
   412	loss to the dynamic parameters generated by the parameter predictor
   413	
   414	3.4. Adaptability retention strategy
   415	To prevent our model from losing its ability to adapt to previously
   416	trained domains when trained on new data, we propose an Adaptability
   417	5
   418	
   419	Neural Networks 198 (2026) 108566
   420	
   421	Z. Chen et al.
   422	
   423	Fig. 4. Four datasets for lifelong learning.
   424	
   425	within the DyAdap module. Given the dynamic parameters 𝜔𝑡 ∈ ℝ𝐾
   426	and 𝜔𝑡−1 ∈ ℝ𝐾 , generated by Φ𝑡 and Φ𝑡−1 respectively, where 𝐾 represents the channel dimension, the parameter regularization loss can be
   427	expressed as:
   428	√
   429	√
   430	2
   431	√∑
   432	√𝐾
   433	𝑡
   434	𝑡−1
   435	√
   436	𝐿𝑟𝑒𝑔 =
   437	(𝜔𝑗 − 𝜔𝑗 )
   438	(6)
   439	
   440	where 𝐿 is the overall loss function, and 𝐿𝑐𝑒 and 𝐿𝑡𝑟𝑖. represent the identity classiﬁcation loss and triplet loss, respectively.
   441	After each training iteration, we perform a Batch Normalization Fusion (BNF) by merging the ﬁnal batch normalization (BN) layers of Φ𝑡−1
   442	and Φ𝑡 . This process helps retain more knowledge learned from previous domains while minimizing the impact on performance in the current
   443	domain. The speciﬁc fusion approach can be represented as:
   444	
   445	𝑗=1
   446	
   447	𝜓 𝑓 = 𝜓 𝑡 + 𝜌𝜓 𝑡−1
   448	
   449	where 𝐿𝑟𝑒𝑔 represents the knowledge distillation loss, and 𝜔𝑡𝑗 denotes the
   450	element in the 𝑗-th channel of 𝜔𝑡 . By guiding our model with knowledge
   451	from previous domains, we can ensure that it preserves the ability to
   452	adapt back to previously seen domains.
   453	The AdRet strategy designs speciﬁc knowledge distillation loss functions for both the feature extraction network and the DyAdap module,
   454	enhancing the model’s ability to mitigate catastrophic forgetting while
   455	dynamically adapting to new domains. During continual learning across
   456	multiple domains, the ReID model with the DyAdap module adjusts its
   457	parameter distribution to adapt to new domains, leading to the potential forgetting of information learned from previous domains. To address
   458	this, the AdRet strategy constrains the parameter update process of the
   459	feature extraction network and the hypernetwork component within the
   460	DyAdap module. This enables the model to retain its feature extraction
   461	and generalization capabilities from previous domains while dynamically adjusting parameter distributions to achieve optimal performance
   462	across domains, thereby mitigating the impact of catastrophic forgetting
   463	on ReID model.
   464	
   465	4. Experiment
   466	In this section, a series of experiments were conducted to validate
   467	our proposed DKAR model. In the experimental setup, we introduce the
   468	employed datasets, evaluation metrics, and the conﬁguration of hyperparameters. Within the experimental results, we analyze and compare
   469	the continual learning performance of the DKAR model with other LReID
   470	models under the same setting, evaluating our method’s eﬀectiveness
   471	on seen domains and unseen domains. For ablation experiments, we designed experiments to separately evaluate the improvement brought by
   472	our designed modules.
   473	
   474	3.5. Training and test procedure
   475	In this work, we apply an identity classiﬁcation loss and a triplet

codex
前两篇的方法主体已经读到。第一篇的关键不是单个模块，而是把“图像、文本、图文同时可用或缺失”统一成一个视觉语义嵌入预训练问题；第二篇把终身 ReID 的遗忘重说成“动态适应能力本身会遗忘”，所以约束的是动态参数生成器而不只是普通特征。
codex
第一篇还缺 `MMM Strategy` 后半段和训练细节，我补一段；第二篇方法在抽取文本里有排版乱序，我也补 `Training and test procedure` 的后半段，避免漏掉损失组合。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf' - | nl -ba | sed -n '395,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   395	image data, while masked language modeling is applied into the text encoder over unpaired text
   396	data. Specifically, given an image and text data, we perform masking on image encoder and text
   397	encoder, respectively. Finally, we add an image-text matching loss L I T M following prior visionand-language pretraining literature [26]. On the basis of it, we introduce Image-Text Matching
   398	pretraining task to learn the relationship between image and text samples, and then apply a classifier
   399	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   400	
   401	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '470,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   470	models under the same setting, evaluating our method’s eﬀectiveness
   471	on seen domains and unseen domains. For ablation experiments, we designed experiments to separately evaluate the improvement brought by
   472	our designed modules.
   473	
   474	3.5. Training and test procedure
   475	In this work, we apply an identity classiﬁcation loss and a triplet
   476	loss to the output features of our model. Additionally, we employ two
   477	knowledge distillation losses, as deﬁned in Eqs. (5) and (6). Formally,
   478	the overall loss function of our model can be written as:
   479	.𝐸
   480	𝐿 = 𝐿𝑐𝑒 + 𝐿𝑡𝑟𝑖. + 𝐿𝐹𝐾𝐷
   481	+ 𝐿𝑟𝑒𝑔
   482	
   483	(8)
   484	
   485	where 𝜓 𝑓 denotes the fused parameters within the BN layer, 𝜓 𝑡 and
   486	𝜓 𝑡−1 represent the parameters of the BN layer in the trained model and
   487	the frozen model, respectively. 𝜌 is the trade-oﬀ parameter used in the
   488	fusion process, which we set to 0.25 in our experiments to retain more
   489	information from the old domain. Our complete training algorithm procedure is shown in Algorithm 1.
   490	During the test stage, we perform spatial average pooling on the
   491	features output by our DKAR model and use the resulting features for
   492	inference.
   493	
   494	4.1. Experiment setting
   495	4.1.1. Datasets
   496	As our proposed DKAR model is designed to enhance the continual learning capabilities of ReID models, we selected four commonly
   497	
   498	(7)
   499	6
   500	
   501	Neural Networks 198 (2026) 108566
   502	
   503	Z. Chen et al.
   504	
   505	Fig. 5. Three datasets for generalization performance testing.
   506	
   507	The VIPeR (Gray & Tao, 2008) dataset comprises a total of 1, 264 images of 632 persons captured by two cameras. As illustrated in Fig. 4,
   508	each person is represented by two images from diﬀerent camera views,
   509	with signiﬁcant viewpoint variations and challenging lighting conditions. The images have dimensions of 128 × 48 pixels, presenting substantial diﬃculties for feature matching due to the extreme visual discrepancies between camera views. During model training, each person’s
   510	two images are split into training and testing sets based on the camera
   511	identiﬁer.
   512	Market1501 (Zheng et al., 2015) contains 32, 668 images of 1, 501
   513	persons captured by six cameras in a campus environment. Fig. 4 demonstrates the dataset’s characteristics, including possible misalignments
   514	and occlusions. The training set includes 751 persons with 12, 936 images, while the test set consists of 750 persons with 19, 732 images, with
   515	no identity overlap between sets.
   516	The CUHK-SYSU (Xiao et al., 2016) dataset comprises 18, 184 images
   517	of 8, 432 persons with 99, 809 annotated bounding boxes, collected from
   518	street snapshots and movie screenshots. As shown in Fig. 4, the dataset
   519	features diverse scenes with both indoor and outdoor environments. The
   520	training set includes 11, 206 images of 5, 532 persons, while the test set
   521	contains 6, 978 images of 2, 900 persons.
   522	The MSMT17 (Wei et al., 2018) dataset consists of 126, 441 images of
   523	4, 101 persons captured under diﬀerent environmental conditions during morning, noon, and evening, using 15 cameras. Fig. 4 highlights
   524	the dataset’s complexity, featuring signiﬁcant illumination variations,
   525	weather changes, and diverse backgrounds across diﬀerent times of day.
   526	To ensure fair and rigorous evaluation of continual learning performance, we adopt a standardized dataset partitioning strategy across
   527	all experiments. For each dataset, we strictly separate the training and
   528	testing sets at the person identity level, ensuring no identity overlap
   529	between training and evaluation phases. Speciﬁcally, our partitioning
   530	strategy follows these principles:
   531	Identity separation: All person identities are randomly split into
   532	mutually exclusive training and testing sets, with no identity appearing
   533	in both sets.
   534	
   535	Algorithm 1 Training procedure of DKAR.
   536	Require: Input:
   537	𝑡-th Task dataset 𝐷𝑡 ,
   538	Feature Extraction Network 𝐹
   539	Initialization:
   540	Model parameters 𝜃Feat , 𝜃DyAdap ,
   541	Hyper-parameters 𝛾1 , 𝛾2
   542	∗ , 𝜃∗
   543	Ensure:
   544	Optimized model parameters 𝜃Feat
   545	DyAdap
   546	1: while Task list is not empty do
   547	2: Get training data:
   548	3: 𝑥 ∼ 𝐷𝑡
   549	// Sample a batch of input images
   550	4: Feature Extraction:
   551	5: 𝑓 ← 𝐹 (𝑥; 𝜃Feat )
   552	// Extract original features
   553	6: Dynamic Adjustment with DyAdap:
   554	7: 𝑓̂ ← DyAdap(𝑓 ; 𝜃DyAdap )
   555	// Apply dynamic adaptation
   556	8: Compute Loss:
   557	.𝐸 (Eq. 5)
   558	9:
   559	Compute feature extraction loss 𝐿𝐹𝐾𝐷
   560	10: Compute hyperparameter loss 𝐿𝑟𝑒𝑔 (Eq. 6)

codex
第一篇的前 8 页在 `MMM Strategy` 里被截断了，说明方法段跨过第 8 页。我已经按你的要求先用了前 8 页抽取，但为了不漏“完整方法部分”，我会额外补读后续一两页，只用于把方法收完整。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 11 'Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf' - | nl -ba | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Learning Visual-Semantic Embedding for Generalizable Person Re-identification
     2	
     3	278:9
     4	
     5	to decide if an input image and text match each other on the basis of ℎ𝐶𝐿𝑆,𝑀 from the multimodal
     6	encoder. It is worth mentioning that masked image modeling loss L M I M [1] and masked language
     7	modeling loss L M L M [5] are applied onto the image and text encoders over a single image or a
     8	text piece, respectively. While MMM loss L M M M [35] are used over paired image-text data.
     9	For downstream tasks, classification heads are applied on the outputs from the image, text,
    10	and multimodal encoders, respectively, for image retrieval tasks. According to the output ℎ𝑀 of
    11	the multimodal encoder, a multi-layer perception is applied to perform predictive computation
    12	on masked partial image patches and text labeling units. For the loss function, L M M M can be
    13	regarded as a relatively complete mask training target in the field of visual language multimodality
    14	because it combines the mask of images and text at the same time. In fact, our L M M M loss can be
    15	seen as a supplement to the general GC loss L G C [33].
    16	To sum up, our MMET framework is expected to be applicable to multimodal or unimodal
    17	scenarios for various vison tasks, which has the great potential to solve the incomplete modality
    18	for generalizable Re-ID event. In addition, MMM strategy can mask the local or discriminative
    19	pedestrian region instead of whole part of input image, which can significantly enhance the
    20	robustness of visual-semantic embedding learning. The intuition for choosing Transformer in this
    21	work is not only because it represents a new trend of traditional vision tasks, but also considering
    22	that Transformer has its own architectural advantages in the use of text modality.
    23	4
    24	4.1
    25	
    26	Experimental Results
    27	Datasets
    28	
    29	In this article, we conduct experiments on several large-scale public datasets, which include Market1501 [56], MSMT17 [42], and CUHK03 [21] datasets, as well as multimodal Re-ID dataset CUHKPEDES [20] and FineGPR [20].
    30	Market-1501 [56] contains 32,668 labeled images of 1,501 identities captured from campus in
    31	Tsinghua University. Each identity is captured by at most six cameras. The training set contains
    32	12,936 images from 751 identities and the test set contains 19,732 images from 750 identities.
    33	MSMT17 [42] has 126,441 labeled images belonging to 4,101 identities and contains 32,621 training
    34	images from 1,041 identities. For the testing set, 11,659 bounding boxes are used as query images
    35	and the other 82,161 bounding boxes are used as gallery images.
    36	CUHK03 [21] contains 14,097 images of 1,467 identities. Following the CUHK03-NP protocol
    37	[59], it is divided into 7,365 images of 767 identities as the training set, and the remaining 6,732
    38	images of 700 identities as the testing set.
    39	CUHK-PEDES [20] contains 40,206 images of 13,003 persons from five existing person Re-ID
    40	datasets, as the subjects for language descriptions. And each image was annotated with two sentence
    41	descriptions and a total of 80,412 sentences were collected.
    42	FineGPR [20] contains 2,028,600 synthesized person images of 1,150 identities, including its corresponding text description with fine-grained attribute annotations. Images in this dataset generally
    43	contain different attributes in a large scope, e.g., viewpoint, weather, illumination, background, and
    44	ID-level annotations, also including many hard samples with occlusion.
    45	RandPerson [41] contains 1,801,816 synthesized person images of 8,000 identities, which is the
    46	first synthetic person Re-ID dataset with a set-up similar to real video surveillance systems. Similar
    47	with FineGPR, images in this dataset generally contain different viewpoints, poses, illuminations,
    48	backgrounds, occlusions, and resolutions.
    49	In our experiments, we follow the standard evaluation protocol [56] used in Re-ID task, and
    50	adopt Mean Average Precision (mAP) and Cumulative Matching Characteristics at Rank-1 and
    51	Rank-5 for performance evaluation on downstream Re-ID task.
    52	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
    53	
    54	278:10
    55	
    56	S. Xiang et al.
    57	Table 1. Ablation Studies of MMM on the Basis of MMET Method, Which Is Trained
    58	on FineGPR Dataset
    59	
    60	Methods
    61	Baseline
    62	MMET w/ RM
    63	MMET w/ MMM (ours)
    64	
    65	FineGPR → Market-1501
    66	
    67	FineGPR → MSMT17
    68	
    69	mAP ↑
    70	
    71	Rank-1 ↑
    72	
    73	mAP ↑
    74	
    75	Rank-1 ↑
    76	
    77	15.4
    78	26.7
    79	28.8
    80	
    81	38.9
    82	51.1
    83	59.2
    84	
    85	3.9
    86	16.6
    87	22.6
    88	
    89	12.5
    90	35.1
    91	50.7
    92	
    93	“Baseline” represents traditional ViT method [7] for DG Re-ID task, and MMM indicates the masked
    94	multimodal modeling strategy proposed in this work, and RM denotes the random masking.
    95	
    96	4.2
    97	
    98	Implementation Details
    99	
   100	During the pretraining of MMET model, we take the ViT-B/16 [7] as the backbone for the image
   101	encoder, text encoder, and multi-modal encoder, respectively, and the hyper-parameter of masking
   102	ratio is empirically set as 15%. Following the training procedure in [35], we adopt SGD with
   103	momentum of 0.9 and weight decay of 0.1. The learning rate is initialized as 1e-3 with cosine
   104	learning rate decay, and training iteration is empirically set as 10,000. Additionally, all the images
   105	are resized to the 256 × 128, and the length of the segmented image block is 12 × 12. The batch
   106	size of training samples is set as 64. As for triplet selection, we randomly select 16 persons and
   107	sampled 4 images for each identity. Adam method and warmup learning strategy are also adopted
   108	to optimize the model. All the experiments are performed on PyTorch [29] with one Nvidia GeForce
   109	RTX 3090 GPU on a server equipped with a Intel Xeon Gold 6240 CPU.
   110	4.3
   111	
   112	Ablation Study
   113	
   114	MMET Is a Strong Baseline for Visual-Semantic Embedding. In this section, we further validate the
   115	effectiveness of our MMM strategy. We give some results of our MMET model on Market-1501
   116	and MSMT17 dataset, respectively. The detailed results are reported in Table 1. Compared with
   117	the baseline model, our proposed MMET can increase the mAP performance on two benchmark
   118	datasets from 15.4%, 3.9% to 26.7% (+11.3%), 16.6% (+12.7%), respectively. This indicates that MMET
   119	increases the discriminative ability of the feature. The structure of MMET is as concise as that
   120	of ViT-B/16, and training MMET requires nothing more than training a canonical classification
   121	network. We hope it will serve as a strong baseline for multimodal image retrieval task.
   122	MMM Improves MMET Especially in Rank-1 Accuracy. According to Table 1, while MMET already
   123	has a high accuracy, MMM brings further improvement to it. On these two datasets, the improvement
   124	in Rank-1 accuracy is +8.1% and +9.6%, respectively; the improvement in mAP is +2.1% and +4.7%,
   125	respectively. The improvement in Rank-1 is larger than in mAP accuracy. In fact, Rank-1 accuracy
   126	characterizes the ability to retrieve the easiest match in the camera network, while mAP indicates
   127	the ability to find all the matches. So the results indicate that MMM strategy is especially beneficial
   128	in finding more closer and easier matches at the first sight, which is more applicable in real-world
   129	scenarios.
   130	The Effectiveness of MMM Strategy. According to Table 2, we can obviously observe that the
   131	performance of Image and Text (MMM strategy) is much superior than single modality (e.g., Image
   132	or Text) (MIM or MLM strategy), which demonstrates that there exists a mutual benefits between
   133	images and text data. It is worth mentioning that CUHK-PEDES dataset contains rich multimodal
   134	labels (e.g., Image and Text), which can serve as a strong foundation for our visual-semantic
   135	embedding learning. However, from Tables 1 and 2, we can also observe that the performance of
   136	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   137	
   138	Learning Visual-Semantic Embedding for Generalizable Person Re-identification
   139	
   140	278:11
   141	
   142	Table 2. Validation Experiment Results Trained on FineGPR Datasets, While Tested
   143	on CUHK-PEDES Dataset, Which Contains Rich Multimodal Labels (e.g., Image and
   144	Text) for Multimodal Fusion
   145	
   146	Methods
   147	
   148	Image data Text data
   149	
   150	MMET + MIM
   151	MMET + MLM
   152	MMET + MMM
   153	
   154	Ø
   155	×
   156	Ø
   157	
   158	FineGPR → CUHK-PEDES
   159	mAP ↑ Rank-1 ↑ Rank-5 ↑
   160	24.6
   161	14.4
   162	25.6
   163	
   164	×
   165	Ø
   166	Ø
   167	
   168	36.7
   169	27.8
   170	38.8
   171	
   172	47.6
   173	39.5
   174	50.4
   175	
   176	MIM, MLM, and MMM represent the masked image modeling, masked language modeling, and
   177	MMM strategy, respectively.
   178	
   179	Table 3. Performance Comparison with the State-of-the-Art UDA Methods on
   180	Market-1501, MSMT17, and CUHK03 Datasets, Respectively
   181	
   182	Methods
   183	LOMO [22]
   184	BOW [56]
   185	PTGAN [42]
   186	SPGAN [4]
   187	PUL [8]
   188	QAConv [23]
   189	QAConv-GS [25]
   190	CDTnet [12]
   191	TransMatcher [24]
   192	Instruct-ReID [14]
   193	MMET (ours)
   194	
   195	Market-1501
   196	
   197	MSMT17
   198	
   199	CUHK03
   200	
   201	mAP
   202	
   203	Rank-1
   204	
   205	mAP
   206	
   207	Rank-1
   208	
   209	mAP
   210	
   211	Rank-1
   212	
   213	8.0
   214	14.8
   215	15.7
   216	22.8
   217	20.5
   218	17.8
   219	49.1
   220	45.5

codex
接下来读第三篇终身可见光红外 ReID。这篇方法段很长，包含重放样本重构、域和模态混合、跨域认知网络几个部件，我会按方法小节分块读。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '260,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   260	efforts (Kong et al., 2023; Li and Hoiem, 2018; Kirkpatrick et al.,
   261	2016; Aljundi et al., 2018; Rannen et al., 2017) lean on regularization
   262	methods, expecting the teacher model to instruct the student model via
   263	knowledge distillation, thereby helping the student model to combat
   264	catastrophic forgetting effectively. Some methods (Serra et al., 2018;
   265	Yoon et al., 2017) adopt parameter isolation methods, they reserves
   266	the most valuable parameters for different tasks, thereby enhancing
   267	the accuracy of the model in performing earlier tasks. Additionally,
   268	addressing sample imbalance during Lifelong Learning is a concern for
   269	a part of approaches (Feng et al., 2023; Wu et al., 2019; Hou et al.,
   270	2019), ensuring fairness in the multi-step training.
   271	
   272	In the field of person re-identification (ReID), Graph Convolutional
   273	Networks (GCNs) have seen extensive application, such as in supervised
   274	person ReID (Shen et al., 2018), unsupervised multi-source domain
   275	adaptation for person ReID (Bai et al., 2021), and lifelong person
   276	ReID (Pu et al., 2021).
   277	However, there is a lack of research concerning GCN architectures
   278	that effectively reconcile the reduction of modal disparities and the
   279	concurrent accumulation of knowledge for LVI-ReID task.
   280	3. Methodology
   281	3.1. Problem definition and formulation
   282	In Lifelong Visible–Infrared Person Re-Identification (LVI-ReID), the
   283	model needs to learn knowledge from a continual data stream containing 𝑇 domains. Suppose there is a stream of datasets 𝐷 = {𝐷𝑖 , 𝐶 𝑖 }𝑇𝑖=1 ,
   284	𝑖
   285	𝑖 } contains the training and testing set and
   286	where 𝐷𝑖 = {𝐷𝑡𝑟𝑎𝑖𝑛
   287	, 𝐷𝑡𝑒𝑠𝑡
   288	𝑖
   289	𝑖,𝑚
   290	𝐶 = {𝐶 }, 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟} involves the visible and infrared cameras
   291	set of the 𝑖th domain. Specifically, at each stage 𝑡, the 𝑡th training
   292	|𝐷𝑡
   293	
   294	|
   295	
   296	𝑡
   297	𝑡 𝑡,𝑚
   298	𝑡𝑟𝑎𝑖𝑛
   299	set 𝐷𝑡𝑟𝑎𝑖𝑛
   300	= {(𝑥𝑡,𝑚
   301	, 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟} contains the images
   302	𝑖 , 𝑦𝑖 , 𝑐𝑖 )}𝑖=1
   303	
   304	3
   305	
   306	X. Zhu, G. Xiao, M.S. Lew et al.
   307	
   308	Computer Vision and Image Understanding 254 (2025) 104328
   309	
   310	Fig. 2. The overview of our proposed methods. In the LVI-ReID task, a training stage comprises two phases, an adaptation phase (in red) and a rehearsal phase (in blue).
   311	During the adaptation phase, a mini-batch sampled from current domain is processed initially through the backbone network. The resultant features are subsequently fed into our
   312	camera-aware Cross-domain Cognitive Network, tasked with accumulating knowledge, ensuring intra-modal consistency, and mining cross-modal similarities. The adaptability of
   313	our model is refined through a collaborative optimization process steered by the loss functions 𝑎𝑑𝑎𝑝𝑡 and 𝑐𝑑𝑐𝑛 . Transitioning into the Rehearsal Phase, a mini-batch of images
   314	retrieved from the memory bank is reconstructed via our Replay Samples Domain-Modal-Mix Reconstruction. This operation facilitates cross-modal and cross-domain integration.
   315	The resultant rehearsal instances are mapped into confidence scores independently by both a temporally frozen model, retained from the previous stage, and the current training
   316	model. Knowledge preservation is enforced through 𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 , which enables a distilled knowledge transfer from the frozen to the training model, mitigating the issue of catastrophic
   317	forgetting. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
   318	
   319	𝑥𝑡,𝑚
   320	from either the visible or infrared modality in the training set of
   321	𝑖
   322	the 𝑡th domain, 𝑦𝑡𝑖 is the corresponding identity label and 𝑐𝑖𝑡,𝑚 is the
   323	corresponding camera label belonging to 𝐶 𝑡,𝑚 . As the setting in previous
   324	𝑡
   325	lifelong person re-identification works, 𝐷𝑡𝑟𝑎𝑖𝑛
   326	is available only at the
   327	𝑡th stage, and we build up an exemplar memory 𝑀 𝑡 to store a limited
   328	number of samplers from each stage for rehearsal. For inference, we
   329	will evaluate the adaptability and anti-forgetting ability of the trained
   330	𝑖 }𝑡
   331	model using all testing sets from all seen domains {𝐷𝑡𝑒𝑠𝑡
   332	in the
   333	𝑖=1
   334	LVI-ReID task.
   335	
   336	Cross-Domain Same-Modality Mixing Samples, (4) Cross-Domain CrossModality Mixing Samples. Through channel-level Domain-ModalityMixing, the diversity of replay samples is significantly augmented,
   337	thereby profoundly mitigating catastrophic forgetting issues induced
   338	by both domain transfer and cross-modal disparity. As a result, this approach substantially enhances the model’s anti-forgetting ability. Then,
   339	the reconstructed samples are mapped onto a confidence interval by the
   340	frozen model from the previous stage and the current training model.
   341	Furthermore, knowledge distillation is performed using the rehearsal
   342	loss function 𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 , encouraging the model to combat catastrophic
   343	forgetting effectively.
   344	
   345	3.2. Method overview
   346	
   347	3.3. Baseline approach
   348	
   349	As depicted in Fig. 2 , our method consists of two main processes
   350	when training the 𝑡th domain at the 𝑡th stage: the adaptation phase
   351	(red) and the rehearsal phase (blue). During the adaptation phase,
   352	𝑡
   353	a batch of samples from the training set 𝐷𝑡𝑟𝑎𝑖𝑛
   354	will first be mapped
   355	into high-dimensional features by the Backbone network, followed
   356	by projection onto a confidence interval by the Classifier, and subjected to constraints imposed by an adaptation loss function 𝑎𝑑𝑎𝑝𝑡 .
   357	Subsequently, we propose leveraging a Cross-domain Cognitive Network (CDCN) to facilitate model learning of intra-modal consistency,
   358	meanwhile constructing distinct cross-modal similarity graphs for each
   359	data domain based on their respective sets of cameras 𝐶 𝑡 , thereby
   360	mitigating the impact of modality semantic gaps and enhancing the
   361	capability of the learned model to recognize cross-modal similarities.
   362	Moreover, a shared GCN layer is employed to alleviate the domain
   363	semantic gap and reinforce the cross-domain generalization ability and
   364	adaptability of the learned model. During the rehearsal phase, we
   365	capitalize on the inherent strengths of a memory bank and propose
   366	the Domain-Modality-Mixing (DMM) based replay samples strategy,
   367	which performs cross-domain and cross-modal sample reconstruction
   368	of replay samples. As shown in Fig. 3, the strategy of DMM commences
   369	by vertically partitioning a three-channel replay sample into several
   370	blocks, where, for each block, it first opts for one of three channel
   371	mixing ways: (1) Single-channel Mixing, (2) Dual-channel Mixing, (3)
   372	All-channel Mixing. Following the selection of channel mixing strategies, DMM proceeds with a certain probability to implement any of the
   373	following four blending approaches: (1) Intra-Domain Same-Modality
   374	Mixing Samples, (2) Intra-Domain Cross-Modality Mixing Samples, (3)
   375	
   376	We establish a baseline LVI-ReID approach by incorporating knowledge distillation (Li and Hoiem, 2018) and utilizing a replay-based
   377	strategy underpinned by iCaRL (Rebuffi et al., 2017) methods. The
   378	baseline model consists of a feature extractor 𝐻(⋅; 𝜃) with parameters
   379	𝜃 and an incremental classifier 𝐺(⋅; 𝜙) with parameters 𝜙 whose output
   380	dimension can expand as the number of total seen identities. Similarly
   381	to previous VI-ReID works, we add a Batch Normalization layer before
   382	the classifier, denoted as 𝐵𝑁(⋅). So, the whole network defined as
   383	𝐹 (⋅; 𝜃; 𝜙) = 𝐺(𝐵𝑁(𝐻(⋅; 𝜃)); 𝜙) is the mapping from the input space to
   384	confidence scores. When training new samples, we use cross-entropy
   385	loss 𝑖𝑑 and triplet loss 𝑡𝑟𝑖 to conduct the adaptation loss function
   386	𝑎𝑑𝑎𝑝𝑡 and optimize the adaptability of the model:
   387	𝑁𝑛
   388	
   389	𝑖𝑑 = −
   390	
   391	𝑃
   392	
   393	𝑏 𝑁
   394	1 ∑∑
   395	𝑦̂ log 𝜎(𝐹 𝑛 (𝑦𝑖𝑗 |𝑥𝑖 ; 𝜃; 𝜙))
   396	𝑛
   397	𝑁𝑏 𝑖=1 𝑗=1 𝑖𝑗
   398	
   399	(1)
   400	
   401	where 𝑁𝑏𝑛 represents the mini-batch size during the adaptation phase,
   402	while 𝑁 𝑃 represents the total number of classes. The one-hot encoded
   403	label for sample 𝑖 is given by 𝑦𝑖𝑗 . Here, 𝜎 is softmax function, and
   404	𝐹 𝑛 (𝑦̂𝑖𝑗 |𝑥𝑖 ; 𝜃; 𝜙) represents the training model’s predicted probability that
   405	sample 𝑥𝑖 belongs to class 𝑗.
   406	𝑁𝑛
   407	
   408	𝑏 [
   409	( (
   410	)
   411	1 ∑
   412	𝑡𝑟𝑖 = 𝑛
   413	𝑚𝑎𝑥 𝑑 𝐻 𝑛 (𝑎𝑖 ), 𝐻 𝑛 (𝑝𝑖 )
   414	𝑁𝑏 𝑖=1
   415	)]
   416	(
   417	)
   418	− 𝑑 𝐻 𝑛 (𝑎𝑖 ), 𝐻 𝑛 (𝑟𝑖 ) + 𝑚, 0
   419	
   420	4
   421	
   422	(2)
   423	
   424	X. Zhu, G. Xiao, M.S. Lew et al.
   425	
   426	Computer Vision and Image Understanding 254 (2025) 104328
   427	
   428	inappropriate labeling of mixed images, ensuring that the model learns
   429	accurately even from such composite data.
   430	To address the problems mentioned above in LVI-ReID, we draw
   431	inspiration from the methods of PCB (Sun et al., 2018), CutMix (Yun
   432	et al., 2019), and CA (Ye et al., 2021). Under the constraint of maintaining the same memory size, we propose the Domain-Modality-Mix
   433	(DMM) based replay samples reconstruction strategy to achieve the
   434	cross-domain sample mix and improve the diversity of replay samples,
   435	thereby effectively mitigating the issue of catastrophic forgetting in the
   436	learned model. Our DMM strategy involves two main selection strategies: Channel Mixing Strategies and Sample Mixing Strategies. The
   437	channel mixing operation can be denoted as 𝐶𝐻(⋅, ⋅; 𝑐ℎ; 𝑐𝑡), where the
   438	first parameter refers to the primary image block, while the second parameter signifies the alternative sample block, 𝑐ℎ ∈ {1, 2, 3} represents
   439	different channel mixing tactics and 𝑐𝑡 ∈ {1, 2, 3} denotes the different
   440	channel types selected. For any two sample blocks with the same
   441	vertical position 𝑥𝑏𝑘,0 = {𝑥𝑅
   442	, 𝑥𝐺 , 𝑥𝐵 } and 𝑥𝑏𝑘,1 = {𝑥𝑅
   443	, 𝑥𝐺 , 𝑥𝐵 },
   444	𝑏𝑘,0 𝑏𝑘,0 𝑏𝑘,0
   445	𝑏𝑘,1 𝑏𝑘,1 𝑏𝑘,1
   446	each consisting of three RGB channels (with identical RGB channel
   447	values for infrared samples), the channel mixing operation can be
   448	mathematically expressed as follows:
   449	
   450	Fig. 3. The Replay Samples Domain-Modal-Mix Strategy. The original replay images
   451	from different domains and modalities will be mixed according to certain rules to
   452	enhance the diversity of replay samples, thereby reducing the performance degradation
   453	of the model during training caused by domain gaps and modal gaps.
   454	
   455	where 𝑁𝑏𝑛 is the mini-batch size during the adaptation phase, while 𝑑
   456	represents the Euclidean distance. 𝐻 𝑛 (⋅) is the training feature extractor, 𝑎𝑖 stands for the anchor sample, 𝑝 denotes a positive sample of the
   457	anchor, 𝑟 denotes a negative sample of the anchor and 𝑚 is a constant.
   458	Then, we can obtain the optimization adaptation loss function 𝑎𝑑𝑎𝑝𝑡 in
   459	the adaptation phase of model learning:
   460	𝑎𝑑𝑎𝑝𝑡 = 𝑖𝑑 + 𝑡𝑟𝑖
   461	
   462	(4)
   463	
   464	where 𝑁𝑏𝑜 is the size of a rehearsal mini-batch, and 𝑇 represents
   465	the temperature coefficient. 𝐾𝐿(⋅ ∥ ⋅) denotes the Kullback–Leibler
   466	divergence, 𝐹 𝑛 (⋅) is the training network’s output, and 𝐹 𝑜 (⋅) is the
   467	frozen old network’s output with gradient updates stopped. Ultimately,
   468	we can get the total optimization object for the Baseline:
   469	𝑏𝑎𝑠𝑒 = 𝑎𝑑𝑎𝑝𝑡 + 𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙
   470	
   471	(6)
   472	
   473	⎧ {𝑥𝑅 , 𝑥𝐺 , 𝑥𝐵 },
   474	𝑏𝑘,1
   475	𝑏𝑘,1
   476	𝑏𝑘,0
   477	⎪
   478	, 𝑥𝐺
   479	, 𝑥𝐵
   480	},
   481	𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑏𝑘,1 ; 2; 𝑐𝑡) = ⎨ {𝑥𝑅
   482	𝑏𝑘,0
   483	𝑏𝑘,1
   484	𝑏𝑘,1
   485	⎪ {𝑥𝑅 , 𝑥𝐺 , 𝑥𝐵 },
   486	⎩
   487	𝑏𝑘,1
   488	𝑏𝑘,0
   489	𝑏𝑘,1
   490	
   491	𝑐𝑡 = 1
   492	𝑐𝑡 = 2
   493	𝑐𝑡 = 3
   494	
   495	(7)
   496	
   497	(8)
   498	
   499	where 𝑐ℎ = 1 represents single-channel mixing across the three RGB
   500	channels, 𝑐ℎ = 2 signifies the random selection of two out of the three
   501	RGB channels for mixing, and 𝑐ℎ = 3 indicates a complete replacement
   502	using 𝑥𝑏𝑘,1 .
   503	After selecting channel mixing strategies, DMM will adopt sample
   504	mixing strategies. The sample mixing strategy can be represented as
   505	𝑆𝐷𝑀(⋅, {}; 𝑠𝑑; 𝑠𝑚; 𝑠𝑡; 𝑐ℎ; 𝑐𝑡), where the initial argument is the primary
   506	sample block, followed by a set of alternative sample blocks. Here,
   507	𝑠𝑑 ∈ {0, 1} indicates whether cross-domain mixing is to be performed,
   508	while 𝑠𝑚 ∈ {0, 1} signifies if cross-modal mixing will perform. The
   509	variable 𝑠𝑡 denotes the number of domains whose samples are stored
   510	in the memory bank 𝑀 𝑡 , while 𝑐ℎ and 𝑐𝑡 are channel mixing ways and
   511	channel types, respectively. Assume that when 𝑠𝑡 < 2, for any given
   512	sample block 𝑥𝑏𝑘,0 , there exists a set of sample blocks comprising the
   513	same-domain and same-modality block 𝑥𝑠𝑠
   514	and the same-domain and
   515	𝑏𝑘,1
   516	cross-modality block 𝑥𝑠𝑐
   517	, both aligned vertically. Conversely, when
   518	𝑏𝑘,2
   519	𝑠𝑡 ⩾ 2, for any sample block 𝑥𝑏𝑘,0 , the associated set encompasses
   520	the same-domain and same-modality block 𝑥𝑠𝑠

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '520,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '760,1010p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   520	the same-domain and same-modality block 𝑥𝑠𝑠
   521	, the same-domain and
   522	𝑏𝑘,1
   523	cross-modality block 𝑥𝑠𝑐
   524	, the cross-domain and same-modality block
   525	𝑏𝑘,2
   526	𝑥𝑐𝑠
   527	, and the cross-domain and cross-modality block 𝑥𝑐𝑐
   528	, all sharing
   529	𝑏𝑘,3
   530	𝑏𝑘,4
   531	the same vertical alignment (it is noteworthy that the same-domain
   532	and same-modality block 𝑥𝑠𝑠
   533	could be 𝑥𝑏𝑘,0 itself, implying no actual
   534	𝑏𝑘,1
   535	mixing in this instance). The sample mixing strategy can then be
   536	mathematically formalized as follows:
   537	
   538	𝑁𝑏𝑜
   539	
   540	𝑇2 ∑
   541	𝐾𝐿(𝐹 𝑛 (𝑥𝑖 )||𝐹 𝑜 (𝑥𝑖 ))
   542	𝑁𝑏𝑜 𝑖=1
   543	
   544	𝑐𝑡 = 1
   545	𝑐𝑡 = 2
   546	𝑐𝑡 = 3
   547	
   548	𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑏𝑘,1 ; 3; 𝑐𝑡) = 𝑥𝑏𝑘,1 , 𝑐𝑡 ∈ {1, 2, 3}
   549	
   550	(3)
   551	
   552	During the rehearsal phase, initially, we follow the replay sampling
   553	strategy from iCaRL (Rebuffi et al., 2017), updating the memory bank
   554	after the completion of training at each incremental stage. Furthermore,
   555	we apply knowledge distillation (Li and Hoiem, 2018) to the replay
   556	samples and conduct the rehearsal loss function 𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 , aiming to
   557	bolster the baseline model’s resilience against forgetting:
   558	𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 =
   559	
   560	⎧ {𝑥𝑅 , 𝑥𝐺 , 𝑥𝐵 },
   561	𝑏𝑘,1
   562	𝑏𝑘,0
   563	𝑏𝑘,0
   564	⎪
   565	, 𝑥𝐺
   566	, 𝑥𝐵
   567	},
   568	𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑏𝑘,1 ; 1; 𝑐𝑡) = ⎨ {𝑥𝑅
   569	𝑏𝑘,0
   570	𝑏𝑘,1
   571	𝑏𝑘,0
   572	⎪ {𝑥𝑅 , 𝑥𝐺 , 𝑥𝐵 },
   573	⎩
   574	𝑏𝑘,0
   575	𝑏𝑘,0
   576	𝑏𝑘,1
   577	
   578	(5)
   579	
   580	3.4. Replay samples domain-modality-mix reconstruction
   581	In this section, we will introduce the details of the proposed Replay Samples Domain-Modality-Mix Reconstruction. Given that VI-ReID
   582	constitutes a fine-grained problem, overfitting the old data domain is
   583	prone to occur when relying solely on a small, repetitive set of replayed
   584	samples, exacerbating catastrophic forgetting. By contrast, a diverse
   585	set of replay samples has the potential to cross a broader range of
   586	feature spaces, thereby enabling the model to revisit critical features
   587	and patterns across various tasks while continuously learning new
   588	ones, thereby enhancing its ability to retain and discriminate among
   589	multiple categories learned in the past. Moreover, previous LReID tasks
   590	overlooked the potential knowledge in the memory bank containing
   591	replay samples from multiple domains. The implementation of crossdomain sample mixing during sample replay not only serves to mitigate
   592	further the detrimental effects of domain semantic gaps but also, by
   593	leveraging frozen legacy models and knowledge distillation techniques,
   594	eliminates concerns regarding the misguidance of the model due to
   595	
   596	𝑆𝐷𝑀(𝑥𝑏𝑘,0 , { 𝑥𝑠𝑠
   597	, 𝑥𝑠𝑐 }; 𝑠𝑑; 𝑠𝑚; 𝑠𝑡 < 2; 𝑐ℎ; 𝑐𝑡)
   598	𝑏𝑘,1 𝑏𝑘,2
   599	{
   600	𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑠𝑠
   601	; 𝑐ℎ; 𝑐𝑡), 𝑠𝑑 = 0, 𝑠𝑚 = 0
   602	𝑏𝑘,1
   603	=
   604	𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑠𝑐
   605	; 𝑐ℎ; 𝑐𝑡), 𝑠𝑑 = 0, 𝑠𝑚 = 1
   606	𝑏𝑘,2
   607	
   608	(9)
   609	
   610	𝑆𝐷𝑀(𝑥𝑏𝑘,0 , { 𝑥𝑠𝑠
   611	, 𝑥𝑠𝑐 , 𝑥𝑐𝑠 , 𝑥𝑐𝑐 }; 𝑠𝑑; 𝑠𝑚; 𝑠𝑡 ⩾ 2; 𝑐ℎ; 𝑐𝑡)
   612	𝑏𝑘,1 𝑏𝑘,2 𝑏𝑘,3 𝑏𝑘,4
   613	⎧𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑠𝑠 ; 𝑐ℎ; 𝑐𝑡),
   614	𝑏𝑘,1
   615	⎪
   616	; 𝑐ℎ; 𝑐𝑡),
   617	⎪𝐶𝐻(𝑥𝑏𝑘,0 , 𝑥𝑠𝑐
   618	𝑏𝑘,2
   619	=⎨
   620	𝑐𝑠 ; 𝑐ℎ; 𝑐𝑡),
   621	𝐶𝐻(𝑥
   622	,
   623	𝑥
   624	𝑏𝑘,0 𝑏𝑘,3
   625	⎪
   626	⎪𝐶𝐻(𝑥 , 𝑥𝑐𝑐 ; 𝑐ℎ; 𝑐𝑡),
   627	⎩
   628	𝑏𝑘,0 𝑏𝑘,4
   629	5
   630	
   631	𝑠𝑑 = 0, 𝑠𝑚 = 0
   632	𝑠𝑑 = 0, 𝑠𝑚 = 1
   633	𝑠𝑑 = 1, 𝑠𝑚 = 0
   634	𝑠𝑑 = 1, 𝑠𝑚 = 1
   635	
   636	(10)
   637	
   638	X. Zhu, G. Xiao, M.S. Lew et al.
   639	
   640	Computer Vision and Image Understanding 254 (2025) 104328
   641	
   642	During the rehearsal phase, we select multiple replay samples from
   643	the memory bank 𝑀 𝑡 to form a rehearsal mini-batch equivalent to an
   644	adaptation mini-batch. For each replay sample 𝑥𝑡,𝑚
   645	𝑟 belonging to 𝑡th domain and 𝑚 modality. DMM first divide it into 𝑘 blocks vertically 𝑥𝑡,𝑚
   646	𝑟 =
   647	𝑘 . Subsequently, DMM proceeds to perform domain mix with
   648	{𝑥𝑡,𝑚
   649	}
   650	𝑟,𝑗 𝑗=1
   651	probability 𝑝𝑐𝑑 and modality mix with probability 𝑝𝑐𝑚 for each block
   652	𝑡,𝑚
   653	𝑥𝑟,𝑗 . The algorithmic process of DMM is illustrated in Algorithm 1.
   654	
   655	3.5.1. Knowledge accumulation
   656	For our cognitive nodes’ knowledge acquiring, we use movement
   657	average to accumulate knowledge from samples belonging to the same
   658	camera in each mini-batch. In the course of either the adaptation phase
   659	or the inference of stage 𝑡, we gain access to the camera label set
   660	|𝐶 𝑡,𝑚 |
   661	𝐶 𝑡,𝑚 = {𝑐𝑗𝑡,𝑚 }𝑗=1 , 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟} belonging to the 𝑡th domain. Specifically,
   662	during the adaptation phase, for each individual camera 𝑐𝑗𝑡,𝑚 , we systematically generate and permanently store a dedicated cognitive node
   663	𝑡,𝑚
   664	𝑛𝑡,𝑚
   665	𝑗 equipped with a fully-connected layer denoted as 𝐹 𝐶𝑗 . Moreover,
   666	
   667	Algorithm 1: Replay Samples Domain-Modality-Mix Reconstruction
   668	
   669	𝑁
   670	
   671	when considering a set of sample instances {𝑥𝑖 }𝑖=1𝑐 collectively associated with camera 𝑐𝑗𝑡,𝑚 , the value 𝑣𝑡,𝑚
   672	𝑗 of the corresponding cognitive
   673	node 𝑛𝑡,𝑚
   674	undergoes
   675	an
   676	update
   677	governed
   678	by the momentum-averaging
   679	𝑗
   680	mechanism:
   681	
   682	𝑡
   683	Input : Replay Sample 𝑥𝑡,𝑚
   684	𝑟 , Memory Bank 𝑀 ,
   685	Number of Domains 𝑠𝑡 in Memory Bank ,
   686	Number of Blocks per Sample 𝑘,
   687	Modality Mix Probability 𝑝𝑐𝑚 ,
   688	Domain Mix Probability 𝑝𝑐𝑑 ,
   689	Mixing Operation 𝑆𝐷𝑀(⋅, {}; 𝑠𝑑; 𝑠𝑚; 𝑠𝑡; 𝑐ℎ; 𝑐𝑡)
   690	Output: Reconstruction Sample 𝑥𝑟𝑠𝑑𝑚𝑚
   691	𝑟
   692	// Divide into 𝑘 blocks
   693	𝑡,𝑚 𝑘
   694	𝑡,𝑚
   695	{𝑥𝑟,𝑗 }𝑗=1 ← 𝑥𝑟 ;
   696	
   697	𝑣𝑡,𝑚
   698	𝑗 = (1 − 𝑚𝑡𝑚) ×
   699	
   700	𝑁𝑐
   701	∑
   702	
   703	𝑤𝑖 𝐻(𝑥𝑖 ; 𝜃) + 𝑚𝑡𝑚 × 𝑣𝑡,𝑚
   704	𝑗
   705	
   706	(11)
   707	
   708	𝑖=1
   709	
   710	where 𝐻(𝑥𝑖 ; 𝜃) symbolizes the feature extractor. The weight assigned
   711	to the extracted feature 𝐻(𝑥𝑖 ; 𝜃) is denoted by 𝑤𝑖 , which is derived
   712	through the fully connected layer 𝐹 𝐶𝑗𝑡,𝑚 and 𝑚𝑡𝑚 is the moving coefficient which conventionally set to a value of 0.9. In this way, we can
   713	constantly accumulate knowledge and get a group of updated values
   714	|𝐶 𝑡,𝑚 |
   715	𝑉𝑡𝑆 = {𝑣𝑡,𝑚
   716	𝑗 }𝑗=1 , 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟} consisting of all cognitive nodes of the
   717	𝑡th domain. It should be noted that the cognitive nodes themselves do
   718	not learn cross-modality similarity directly, they are the anchors for the
   719	GCN layer to learn cross-modality similarity, we design cognitive nodes
   720	to accumulate knowledge in the camera level and help GCN layer to
   721	optimize parameters and learn cross-modality similarity.
   722	
   723	for 𝑗 = 1 to 𝑘 do
   724	𝑐ℎ ← 𝑅𝑎𝑛𝑑𝑜𝑚_𝐶ℎ𝑜𝑖𝑐𝑒 (1, 2, 3);
   725	𝑐𝑡 ← 𝑅𝑎𝑛𝑑𝑜𝑚_𝐶ℎ𝑜𝑖𝑐𝑒 (1, 2, 3);
   726	𝑠𝑑 ← 𝑅𝑎𝑛𝑑𝑜𝑚_𝐶ℎ𝑜𝑖𝑐𝑒_𝑤𝑖𝑡ℎ_𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 (0, 1; 𝑝𝑐𝑑 );
   727	𝑠𝑚 ← 𝑅𝑎𝑛𝑑𝑜𝑚_𝐶ℎ𝑜𝑖𝑐𝑒_𝑤𝑖𝑡ℎ_𝑃 𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 (0, 1; 𝑝𝑐𝑚 );
   728	if 𝑠𝑡 ⩾ 2 then
   729	𝐴𝑡𝑠𝑒𝑡 ← {𝑥𝑠𝑠
   730	, 𝑥𝑠𝑐 , 𝑥𝑐𝑠 , 𝑥𝑐𝑐 } ∈ 𝑀 𝑡 ;
   731	𝑏𝑘,1 𝑏𝑘,2 𝑏𝑘,3 𝑏𝑘,4
   732	else
   733	𝐴𝑡𝑠𝑒𝑡 ← {𝑥𝑠𝑠
   734	, 𝑥𝑠𝑐 } ∈ 𝑀 𝑡 ;
   735	𝑏𝑘,1 𝑏𝑘,2
   736	end
   737	𝑥𝑟𝑠𝑑𝑚𝑚
   738	← 𝑆𝐷𝑀(𝑥𝑡,𝑚
   739	𝑟,𝑗
   740	𝑟,𝑗 , 𝐴𝑡𝑠𝑒𝑡; 𝑠𝑑; 𝑠𝑚; 𝑠𝑡; 𝑐ℎ; 𝑐𝑡);
   741	
   742	3.5.2. Cognitive graph construct
   743	The primary goal of the CDCN is to learn the intra-modal similarities
   744	and cross-modal consistencies. To achieve this, we employ a dualgraph architecture  = (𝐴, 𝑉 ) consisting of two unique components:
   745	an instance-based Intra-Modal Consistency Graph (IMCG), designed to
   746	integrate and harmonize knowledge within the same modality, and
   747	a domain-specific Cross-Modal Similarity Graph (CMSG), designed to
   748	reconcile disparities between visible and infrared modalities within the
   749	same domain.
   750	Intra-Modal Consistency Graph. Given a mini-batch from the
   751	adaptation or inference phase, the 𝑁𝑚 features extracted from either
   752	the visible or infrared modality are defined as 𝑉 𝐶,𝑚 = 𝐻(𝑋 𝑚 ; 𝜃), where
   753	𝑁𝑚
   754	𝑋 𝑚 = {𝑥𝑚
   755	𝑖 }𝑖=1 , 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟}. Inspired by AKA (Pu et al., 2021), we
   756	employ the IMCG 𝐶 (𝐴𝐶 , 𝑉 𝐶 ) to discover the intra-modal relationship
   757	among instances across visible and infrared modalities, respectively.
   758	More specifically, we define the edge set of each modality as 𝐴𝐶,𝑚 ,
   759	which is fully connected, while 𝑉 𝐶,𝑚 represents the feature values
   760	extracted from the backbone consisting of instances belonging to 𝑚

 succeeded in 0ms:
   760	extracted from the backbone consisting of instances belonging to 𝑚
   761	modal. We regard the contributions from instance 𝑖𝑚 to instance 𝑗𝑚 as
   762	equal in the intra-modal, so we set the equal edge weights between
   763	different instances in the same modality:
   764	
   765	end
   766	
   767	// Combine blocks
   768	𝑥𝑟𝑠𝑑𝑚𝑚
   769	← {𝑥𝑟𝑠𝑑𝑚𝑚
   770	}𝑘𝑗=1 ;
   771	𝑟
   772	𝑟,𝑗
   773	return 𝑥𝑟𝑠𝑑𝑚𝑚
   774	𝑟
   775	3.5. Cross-domain cognitive network
   776	In this section, we will introduce the details of the proposed Crossdomain Cognitive Network (CDCN). In the LVI-ReID task, alleviating the performance degradation caused by the modality semantic
   777	gap poses a significant challenge. Since Graph Convolutional Networks (Jiang et al., 2019) have powerful information aggregation and
   778	transmission ability, they are employed to enhance our model’s crossmodal and cross-domain cognition ability. Inspired by AKA (Pu et al.,
   779	2021), we first construct consistency graphs within visible and infrared
   780	modalities to encourage the model to learn intra-modal consistency.
   781	Furthermore, given the abundance of information encapsulated within
   782	cameras in VI-ReID tasks, we adopt the Chunking Theory (Gobet et al.,
   783	2001) from cognitive psychology to conduct the cognitive network.
   784	We aggregate information from within individual cameras into cognitive nodes, thereby packaging camera-specific details, and establish
   785	cross-modal similarity graphs via these cognitive nodes. This strategy facilitates cross-modal information transfer and promotes learning
   786	of cross-modal invariance while accumulating knowledge. Additionally, leveraging domain-specific cognitive nodes, we construct tailored
   787	cross-modal similarity graphs for each domain, encouraging the model
   788	to address intra-domain cross-modal cognitive disparities specifically.
   789	However, focusing solely on intra-domain cross-modal cognition is
   790	insufficient, and the performance decrement induced by the domain
   791	semantic gap must also be addressed. To this end, we employ a shared
   792	GCN Layer to facilitate the integration of cross-domain information.
   793	Finally, we design a camera-aware cross-domain cognitive network
   794	(equipping with a batch-norm layer 𝑏𝑛(), a classifier 𝑔(⋅; 𝜙𝑓 ), and a
   795	shared GCN Layer) for each stage. The proposed CDCN framework
   796	contains three processes: (1) knowledge accumulation, (2) cognitive
   797	graph construct, and (3) knowledge transfer.
   798	
   799	𝐴𝐶,𝑚
   800	𝑖𝑚 𝑗𝑚 = 1.0, 𝑖𝑚 𝑗𝑚 ∈ [1, 𝑁𝑚 ], 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟}
   801	[ 𝐶,𝑣𝑖𝑠
   802	]
   803	[ 𝐶,𝑣𝑖𝑠 ]
   804	𝐴
   805	0
   806	𝑉
   807	𝐶
   808	𝐴𝐶 =
   809	,
   810	𝑉
   811	=
   812	0
   813	𝐴𝐶,𝑖𝑟
   814	𝑉 𝐶,𝑖𝑟
   815	
   816	(12)
   817	
   818	where 𝐴𝐶,𝑚 ∈ R𝑁𝑚 ×𝑁𝑚 , and 𝐴𝐶 ∈ R2𝑁𝑚 ×2𝑁𝑚 gives the adjacency matrix
   819	of IMCG  , while 𝑉 𝐶 ∈ R2𝑁𝑚 ×𝑑 is concatenated by 𝑑 dimension feature
   820	values 𝑉 𝐶,𝑚 ∈ R𝑁𝑚 ×𝑑 from both visible and infrared modalities.
   821	Cross-modal Similarity Graph. In order to address disparities
   822	between visible and infrared modalities within the same domain, we
   823	adopt the Cross-modal Similarity Graph (CMSG)  (𝐴𝐾 , 𝑉 𝐾 ) to learn
   824	the cross-modal invariance based on visible and infrared cognitive
   825	nodes. Specifically, in the 𝑡th domain, for a mini-batch of size 𝑁𝑣𝑖𝑠
   826	sampled from visible modality, each instance is connected to every
   827	infrared cognitive node of the 𝑡th domain:
   828	𝐴𝐾,𝑣𝑖𝑠
   829	= 1.0, 𝑖𝑣𝑖𝑠 ∈ [1, 𝑁𝑣𝑖𝑠 ], 𝑘𝑖𝑟 ∈ [1, |𝐶 𝑡,𝑖𝑟 |]
   830	𝑖 𝑘
   831	𝑣𝑖𝑠 𝑖𝑟
   832	
   833	6
   834	
   835	(13)
   836	
   837	X. Zhu, G. Xiao, M.S. Lew et al.
   838	
   839	Computer Vision and Image Understanding 254 (2025) 104328
   840	
   841	where 𝑖𝑣𝑖𝑠 is the index of a single instance in a visible mini-batch,
   842	and 𝑘𝑖𝑟 is the index of an infrared cognitive node in the domain, with
   843	|𝐶 𝑡,𝑖𝑟 | representing the total number of infrared cognitive nodes in the
   844	𝑡th domain. It is similar to a mini-batch of size 𝑁𝑖𝑟 sampled from the
   845	infrared modality:
   846	𝐴𝐾,𝑖𝑟
   847	= 1.0, 𝑖𝑖𝑟 ∈ [1, 𝑁𝑖𝑟 ], 𝑘𝑣𝑖𝑠 ∈ [|𝐶 𝑡,𝑖𝑟 |, |𝐶 𝑡,𝑖𝑟 | + |𝐶 𝑡,𝑣𝑖𝑠 |]
   848	𝑖 𝑘
   849	𝑖𝑟 𝑣𝑖𝑠
   850	
   851	3.5.5. The inference of CDCN
   852	This section details the distinctions between the training and inference phases of the CDCN framework. The inference phase is distinctive
   853	as it omits the accumulation of new knowledge and optimization procedures, and it primarily focuses on utilizing the accrued knowledge
   854	for predictions. By the way, during the rehearsal phase, CDCN does
   855	not participate actively due to the risk of incorporating misleading
   856	information from mixed samples, which could potentially corrupt its
   857	learned knowledge.
   858	Note that, during the inference phase, as we deal with only a
   859	single modality for both the query and gallery sets, the edge set for
   860	𝐶,𝑚 ∈ R𝑁𝑒 ×𝑁𝑒 , the
   861	a mini-batch of size 𝑁𝑒 is represented as 𝐴𝐶
   862	𝑒 = 𝐴
   863	corresponding value vector as 𝑉𝑒𝐶 = 𝑉 𝐶,𝑚 ∈ R𝑁𝑒 ×𝑑 , while the edge
   864	𝐾
   865	set 𝐴𝐾
   866	𝑒 , and values vector 𝑉𝑒 of CSG are similar to adaptation phase.
   867	Ultimately, we can get the edge set and values vector of 𝑒 (𝐴𝑒 , 𝑉𝑒 ):
   868	[
   869	]
   870	[ 𝐶,𝑚 ]
   871	𝐴𝐶,𝑚
   872	𝐴𝐾,𝑣𝑖𝑠
   873	𝑉𝑒
   874	𝑒
   875	𝑒
   876	𝐴𝑒 =
   877	,
   878	𝑉
   879	=
   880	(19)
   881	𝑒
   882	𝑉𝑒𝐾
   883	𝐴𝐾,𝑖𝑟
   884	𝐴𝐾,𝑒𝑦𝑒
   885	𝑒
   886	𝑒
   887	
   888	(14)
   889	
   890	where 𝑘𝑣𝑖𝑠 is the index of a visible cognitive node in the domain and
   891	|𝐶 𝑡,𝑣𝑖𝑠 | represents the total number of visible cognitive nodes in the 𝑡th
   892	domain. Subsequently, we can get the adjacency matrix 𝐴𝐾 of CMSG
   893	𝐾 :
   894	[
   895	]
   896	0
   897	𝐴𝐾,𝑣𝑖𝑠
   898	𝐴𝐾 =
   899	, 𝑉 𝐾 = 𝑉𝑡𝑆
   900	(15)
   901	𝐾,𝑖𝑟
   902	𝐾,𝑒𝑦𝑒
   903	𝐴
   904	𝐴
   905	𝑡
   906	
   907	𝑡
   908	
   909	where 𝐴𝐾,𝑣𝑖𝑠 ∈ R2𝑁𝑚 ×|𝐶 | , 𝐴𝐾,𝑖𝑟 ∈ R|𝐶 |×2𝑁𝑚 gives the relationship be𝑡
   910	tween instances and cross-modal cognitive nodes, 𝑉𝑡𝑆 ∈ R|𝐶 |×𝑑 repre𝑡
   911	sents all values of cognitive nodes in the camera set 𝐶 of 𝑡th domain,
   912	and 𝐴𝐾,𝑒𝑦𝑒 is an identity matrix meaning that every cognitive node is
   913	directly connected to itself.
   914	
   915	𝑡
   916	
   917	3.5.3. Knowledge transfer
   918	Aiming to stimulate the model’s capacity to capture both intramodal consistency and cross-modal similarity, we deploy a shared GCN
   919	layer to instantiate an environment where intra-modal instance nodes
   920	engage in knowledge sharing and interact with cross-modal cognitive
   921	nodes, all within a context of cross-domain knowledge integration.
   922	Cognitive Graph. By merging 𝐴𝐶 and 𝐴𝐾 , 𝑉 𝐶 and 𝑉 𝐾 , we acquire
   923	the edge set 𝐴 and value vector 𝑉 of joint graph :
   924	[
   925	]
   926	[ 𝐶 ]
   927	𝐴𝐶
   928	𝐴𝐾,𝑣𝑖𝑠
   929	𝑉
   930	𝐴=
   931	,
   932	𝑉
   933	=
   934	(16)
   935	𝐴𝐾,𝑖𝑟 𝐴𝐾,𝑒𝑦𝑒
   936	𝑉𝐾
   937	𝑡
   938	
   939	 = 𝑎𝑑𝑎𝑝𝑡 + 𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 + 𝑐𝑑𝑐𝑛
   940	
   941	𝑡
   942	
   943	4.1. New benchmark for LVI-ReID
   944	To evaluate the adaptability and resistance to forgetting the proposed model, we devise a continual learning data stream scheme
   945	RegDB→SYSU→LLCM→VCM based on four visible–infrared person
   946	re-identification datasets:
   947	RegDB (Nguyen et al., 2017) comprises 412 unique individual identities, each including ten pairs of visible and infrared modality images.
   948	Following the practice in prior works, we randomly select 206 identities
   949	for training and reserve the remaining identities for testing.
   950	SYSU-MM01 (Wu et al., 2017b) includes 491 distinct identities
   951	collected by four visible and two infrared cameras. The training set
   952	consists of 395 identities with 22258 visible images and 11909 infrared
   953	images. In the testing phase, this dataset provides two different search
   954	modes: Indoor Search and All-Search. Notably, we will exclusively
   955	employ the All-Search mode, wherein all images captured by visible
   956	cameras are incorporated into the gallery set.
   957	LLCM (Zhang and Wang, 2023) is a low-light cross-modal dataset
   958	encompassing 1064 objects with unique identity labels. The image data
   959	was continuously collected over more than 100 days from January to
   960	April by nine RGB cameras and nine infrared cameras, spanning various
   961	weather conditions and clothing styles. Of these, 713 identities are
   962	allocated to the training set, while the remaining 351 identities form
   963	the test set (the number of query samples is 8680).
   964	HITSZ-VCM (Lin et al., 2022) is a video-based visible–infrared
   965	person re-identification dataset that captures behavioral trajectories
   966	of 927 identities using six RGB cameras and six infrared cameras.
   967	Each identity sequence is constructed as a tracklet consisting of 24
   968	consecutive frames, resulting in a total of 11785 visible tracklets and
   969	
   970	(17)
   971	
   972	3.5.4. The optimization of CDCN
   973	Utilizing residual connections, we obtain the final feature vectors
   974	𝑉 𝐹 of CDCN outputs, employing the knowledge representation vector
   975	𝑉̄ and feature values 𝑉 𝐶 of instances for the process. Subsequently,
   976	we input the feature vectors 𝑉 𝑂𝐹 sequentially into the CDCN’s shared
   977	batch normalization layer 𝑏𝑛() and shared classifier 𝑔(⋅; 𝜙𝑐 ) to obtain
   978	confidence scores, and optimize CDCN using cross-entropy loss:
   979	𝑃
   980	
   981	2𝑁𝑚 𝑁
   982	1 ∑∑
   983	𝑦̂ log 𝜎(𝑔(𝑏𝑛(𝑉𝑖𝐹 ); 𝜙𝑐 ))
   984	2𝑁𝑚 𝑖=1 𝑗=1 𝑖𝑗
   985	
   986	(20)
   987	
   988	4. Experiments
   989	
   990	where 𝑉̂ is the vertex embedding after one-layer graph convolutional
   991	operation, 𝑊 is a learnable weight matrix of the GCN layer, and 𝛿 is
   992	a non-linear function, e.g., LeakyReLu (Xu et al., 2015). We employed
   993	only a single layer to accomplish knowledge transmission both within
   994	and across modalities, yet it is feasible to seamlessly superimpose
   995	multiple layers of GCN. After completing the knowledge transmission,
   996	we obtain the knowledge representation vector 𝑉̄ from top 2𝑁𝑚 rows
   997	of 𝑉̂ , denoted as 𝑉̄ = {𝑉̂𝑖 |𝑖 ∈ [1, 2𝑁𝑚 ]}.
   998	
   999	𝑐𝑑𝑐𝑛 = −
  1000	
  1001	𝑡
  1002	
  1003	3.5.6. Multi-loss optimization
  1004	It is worth noting that we utilize a single shared GCN layer throughout all training stages to learn and accumulate knowledge from different domains. For each domain, new cognitive nodes are created for
  1005	each camera to memorize domain-specific knowledge at the camera
  1006	level. During inference, these domain-specific cognitive nodes and the
  1007	shared GCN layer work together to leverage the learned cross-domain
  1008	knowledge. Finally, the optimization function of our model can be
  1009	described as follows:
  1010	

codex
第三篇的 `pdftotext` 把双栏内容交叉了，但方法小节已经能还原：基线重放蒸馏、DMM 重构、CDCN 图认知和多损失优化。最后一段多损失公式被实验段夹住，我再补几行确认结尾。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '998,1045p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   998	
   999	𝑐𝑑𝑐𝑛 = −
  1000	
  1001	𝑡
  1002	
  1003	3.5.6. Multi-loss optimization
  1004	It is worth noting that we utilize a single shared GCN layer throughout all training stages to learn and accumulate knowledge from different domains. For each domain, new cognitive nodes are created for
  1005	each camera to memorize domain-specific knowledge at the camera
  1006	level. During inference, these domain-specific cognitive nodes and the
  1007	shared GCN layer work together to leverage the learned cross-domain
  1008	knowledge. Finally, the optimization function of our model can be
  1009	described as follows:
  1010	
  1011	where 𝐴 ∈ R(2𝑁𝑚 +|𝐶 |)×(2𝑁𝑚 +|𝐶 |) indicates the correlation of intramodal instances, and the relationship between instances and cross𝑡
  1012	modal cognitive nodes, with 𝑉 ∈ R(2𝑁𝑚 +|𝐶 |)×𝑑 representing the value
  1013	vector.
  1014	Knowledge Transfer. After constructing the joint graph , we
  1015	transfer the intra-modal and cross-modal knowledge via the GCN,
  1016	which is formulated as:
  1017	𝑉̂ = 𝛿(𝐴(𝑉 𝑊 ))
  1018	
  1019	𝑡
  1020	
  1021	where 𝐴𝑒 ∈ R(𝑁𝑒 +|𝐶 |)×(𝑁𝑒 +|𝐶 |) , 𝑉𝑒 ∈ R(𝑁𝑒 +|𝐶 |)×𝑑 and 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟}
  1022	represents the instances belonging to different modality from either the
  1023	query set or the gallery set under different test mode.
  1024	
  1025	(18)
  1026	
  1027	where 2𝑁𝑚 is the size of mini-batch, 𝑁 𝑃 is the total number of classes,
  1028	𝑔(𝑏𝑛(𝑉𝑖𝑂𝐹 ); 𝜙𝑓 ) is the confidence scores, and 𝜎(⋅) is the softmax function.
  1029	7
  1030	
  1031	
codex
现在读第四篇提示池终身可见光红外 ReID。它没有标准 `Methodology` 行，所以我先按正文中的框架图、问题定义和提示池小节定位。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | nl -ba | rg -n \"Method|method|Approach|approach|Overview|overview|Problem|problem|Prompt|prompt|Pool|Generator|Loss|Training|Algorithm|Experiments|Experiment\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:     1	Lifelong Visible-Infrared Person Re-Identification with Prompt
2:     2	Pool and Instance-level Prompt Generator
36:    36	methods primarily rely on fixed datasets for training, which struggle to accommodate continuously evolving cross-domain data, thus
41:    41	we propose a prompt-based dynamic learning framework to address the challenges inherent in LVI-ReID. Specifically, we design a
42:    42	Prompt Pool (PP) module to encapsulate shared knowledge across
43:    43	tasks or domains. In addition, we propose an instance-level prompt
45:    45	domain-specific knowledge, overcoming the limitations of a fixedsize prompt pool. For task-agnostic inference during the LVI-ReID
47:    47	the most relevant prompt by evaluating the similarity between
48:    48	query tokens and keys, thereby addressing the nuanced requirements of varying tasks. Extensive experimental evaluations demonstrate the superiority of our proposed prompt learning-based PPIPG framework over state-of-the-art methods in both lifelong learnings, lifelong person re-identification (LReID), and LVI-ReID settings. These results underscore the efficacy and practicality of our
50:    50	and cross-domains. The source code of our designed PP-IPG method
54:    54	methodologies → Computer vision.
57:    57	Lifelong learning, Visible-infrared person re-identification, Prompt
61:    61	Visible-Infrared Person Re-Identification with Prompt Pool and Instancelevel Prompt Generator. In Proceedings of the 2025 International Conference on
79:    79	Relevant studies [21, 58] have proposed an online learning method
82:    82	datasets in these approaches presents fewer challenges in maintaining learned knowledge while enhancing generalization capabilities.
106:   106	to propose an adaptive knowledge accumulation strategy, encompassing both knowledge representation and knowledge manipulation aspects. However, existing methods and tasks primarily focus
118:   118	features. Existing approaches, while capable of learning aligned
127:   127	problem, where identities in the training and testing sets are entirely disjoint. Second, unlike LReID, LVI-ReID must tackle not only
129:   129	To effectively overcome these challenges in the LVI-ReID, we propose a novel prompt learning-based framework. Specifically, our
130:   130	framework incorporates two distinct prompt generation modules
131:   131	to address the challenges of LVI-ReID. The first is a Prompt Pool
133:   133	grouped prompts as model inputs. The PP module effectively captures shared knowledge across tasks or domains while maintaining
135:   135	diverse scenarios. The second is an Instance-level Prompt Generation (IPG) module, which dynamically generates instance-specific
136:   136	prompts from input tokens without relying on a fixed-size prompt
157:   157	token and selects the associated prompt and classifier for effective
159:   159	The main contributions of our prompt learning-based PP-IPG
161:   161	• We introduce a prompt learning-based approach to tackle
162:   162	the LVI-ReID problem. A Prompt Pool (PP) module is designed to enable flexible retrieval of grouped prompts for
164:   164	task-specific prompts through parameter-isolated groups,
167:   167	• An Instance-Level Prompt Generator (IPG) is proposed to
169:   169	decomposing input token semantics and dynamically generating instance-level prompts. It Integrates task identifierguided domain encoding to achieve fine-grained prompt
172:   172	select the most suitable prompts and classifiers during the
174:   174	• Extensive experiments demonstrate that our PP-IPG approach
175:   175	outperforms state-of-the-art lifelong learning methods, LReID
176:   176	methods, and existing LVI-ReID solutions on the task of LVIReID. To the best of our knowledge, our work is the first
177:   177	attempt to integrate prompt learning with lifelong learning in the context of visible-infrared ReID, addressing the
187:   187	data distributions and catastrophic forgetting. Existing approaches
191:   191	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
195:   195	Figure 2: Overview of Our Proposed LVI-ReID Method. Initially, the input image (visible or infrared) is processed into image
196:   196	patches with positional embeddings, which are then fed into the IPG (Instance-specific Prompt Generator) module and the
197:   197	pre-trained model. The class token output by the pre-trained model retrieves highly relevant prompts from the prompt pool,
199:   199	task identifier, which guides the IPG module in generating instance-level prompts. Subsequently, the instance-level prompts,
200:   200	selected prompts, class token, and image tokens are concatenated and fed into the self-attention module. Notably, each layer of
201:   201	the attention module is independently equipped with an IPG module to compute distinct instance-level prompts.
202:   202	are categorized into distillation-based methods [7, 11, 22, 25, 34, 46,
203:   203	59], regularization-based methods [15, 18, 19, 55], and structurebased methods [8, 14, 23, 33, 35, 49, 54]. Distillation-based methods preserve cross-task performance via knowledge distillation:
206:   206	models for storage-free optimization; Regularization-based methods protect critical knowledge by constraining parameter updates,
208:   208	Bayesian integration; Structure-based methods explicitly construct
211:   211	shared parameter spaces, structure-based methods achieve superior
217:   217	different times by the same camera correspond to the same individual. Early neural networks, primarily designed for classification tasks, naturally extracted global features from an input image when applied to ReID tasks [62]. Traditional ReID methods
219:   219	scenarios. Based on deep neural networks, numerous methods
222:   222	parts or simple vertical region partitioning) and viewpoint information. These methods are tailored specifically for visible-spectrum
224:   224	the Visible-Infrared Person Re-identification (VI-ReID) methods
226:   226	framework to adaptively learn modality-sharable features for VIReID. Typical methods[9, 12, 50, 51, 64] adopt a two-stream network,
229:   229	layer to extract modality-invariant features, thus resolving the differences within and across modalities. Recently, VI-ReID methods
239:   239	feature loss to guide the Transformer to infer reliable identity information according to different modalities. Methods [6, 37, 38, 40]
241:   241	the design of network architectures or loss functions. These methods use Generative Adversarial Networks (GANs) to generate one
245:   245	methods depends on the high quality of the images generated.
246:   246	Prompt learning. Inspired by the success of text prompting
247:   247	techniques [2, 20] in Natural Language Processing, prompt learning
248:   248	methods have also been widely applied in visual tasks and have
250:   250	learning that combines prompting mechanisms with Vision Transformers is developing rapidly. L2P [42] first introduced the concept
251:   251	of the prompt pool to dynamically select prompts for continual
253:   253	the learned representations. DualPrompt [41] uses two types of
254:   254	prompts to address catastrophic forgetting, where G-Prompt is
255:   255	used for general knowledge, and E-Prompt is used for task-specific
256:   256	knowledge. S-prompt [39] introduces an independent cross-domain
257:   257	prompting learning scheme and employs the K-Nearest Neighbor
259:   259	[17] adopts an adaptive prompt generator to generate appropriate
260:   260	prompts for each instance, thereby solving the domain scalability
261:   261	problem caused by the prompt pool. In this work, our method utilizes both the prompt pool and the instance-level prompt generator
263:   263	visual prompts in the LVI-ReID domain.
276:   276	mechanism to automatically select prompts 𝑃𝐾 from the prompt
278:   278	Each prompt in the prompt pool can maintain a certain degree of
280:   280	domain-specific knowledge. Meanwhile, an instance-level prompt
282:   282	the task identifier 𝑡 as the supervise information to linearly transform the image tokens into the adaptive prompts 𝑃𝐼 , which can
292:   292	3.1 Problem formulation
328:   328	evaluate the performance of our proposed method on the LVI-ReID
375:   375	Approach overview
377:   377	The overview of our proposed PP-IPG framework for LVI-ReID is illustrated in Fig.2. In each iteration of the training process, 𝑁 images
387:   387	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
399:   399	prompts from the prompt pool. Based on the features of the input
400:   400	images, we select the 𝐾 most similar prompts 𝑃𝐾 to participate in
401:   401	parameter updates. The prompt selection process can be regarded
402:   402	as a clustering operation to some extent. Input data with similar features tend to activate the same or high-similarity prompt
404:   404	way of matching corresponding prompt sets based on similarity
405:   405	can better adapt to the prompt requirements of different task inputs. Furthermore, Li and Liang [24] demonstrated that mapping
406:   406	the prompt parameters through a multi-layer perceptron (MLP)
409:   409	that directly optimizing the prompt 𝑃𝐾 could result in instability
411:   411	parameters of the prompt often lacks generalizability across different tasks or datasets, thereby limiting its applicability. To address
413:   413	into the design of the prompt pool. The residual connections in
414:   414	our prompt pool not only preserve the original information of the
415:   415	prompt, ensuring that the foundational knowledge remains intact,
418:   418	designed prompt pool can be expressed as follows:
489:   489	Instance-level Prompt Generator
491:   491	Conventional prompt-based lifelong learning approaches suffer
493:   493	on fixed prompt pool architectures necessitates prior hyperparameter tuning for target domains, which contradicts the practical
496:   496	as the prompt pool expands dynamically. To address these challenges, we propose an instance-aware adaptive prompting strategy,
499:   499	design an Instance-level Prompt Generator (IPG) module that provides fine-grained guidance, assisting the model efficiently in leveraging learned representations and better capturing domain-specific
502:   502	instance-specific prompts based on input features, breaking the
503:   503	limitation of traditional fixed-size prompt pools and enabling the
510:   510	design principles of [17]. These generated vectors serve as instancelevel prompts 𝑃𝐼 , dynamically guiding the model to adapt to diverse
513:   513	Prompt Pool
516:   516	prompt pool has the ability to learn the shared knowledge among
519:   519	prompt learning has the advantages of simple operation and low
522:   522	consider leveraging the advantages of prompt learning to help the
524:   524	between modality-specific information. Thus, in this work, we introduce the prompt pool to address the catastrophic forgetting problem
525:   525	existing in LVI-ReID. At the same time, the selected prompts are
527:   527	representation correspondences among modality-specific information. Formally, the prompt pool consists of a set of learnable vectors,
535:   535	is a prompt with 𝐿𝑝 tokens and has the same
536:   536	dimension as 𝑥𝑒𝑚 . In our work, not all prompts in the prompt pool
559:   559	Figure 3: The inference stage of our LVI-ReID method. The
561:   561	match the corresponding prompt. Meanwhile, the image token is additionally processed through QAB(·) to calculate the
566:   566	To enable input instances to autonomously select prompts from
567:   567	the prompt pool and the corresponding classifiers, we design a
568:   568	query-key mechanism. The works of L2P [42] and DualPrompt
572:   572	query features for prompt selection. Additionally, we introduce a
577:   577	for prompts, and the image feature tokens 𝑓𝑖 output by SAB∗ (·) are
598:   598	re-identification method. To ensure that the HITSZ-VCM dataset
614:   614	where, 𝑄 𝑝 (𝑥𝑖 ) represents the query function for prompts, and
615:   615	𝑄𝑐 (𝑥𝑖 ) represents the query function for classifiers. Each prompt
622:   622	design a L𝑝𝑢𝑙𝑙 loss function to ensure they will independently approach 𝑓𝑝,𝑖 and 𝑓𝑐,𝑖 in the vector space, which is defined as follows:
650:   650	token. The class token is used to select the most relevant prompts
651:   651	from the prompt pool, while the image token, after being concatenated with the queue token 𝜃𝑞 , is fed into QAB(·) to compute the
656:   656	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
660:   660	Table 1: Comparison with State-of-the-Art Methods.All the results are implemented based on our baseline. The training
662:   662	Methods
776:   776	Experiments
784:   784	Prompt Pool
810:   810	center are saved. The size of the prompt pool is set to 8, and the
811:   811	length of each prompt is set to 20. In each training iteration, the
812:   812	top two most relevant prompts are selected. The weights of the
814:   814	dataset is completed, and an instance-level prompt with a length of
859:   859	experimental results demonstrate that our proposed PP-IPG method
861:   861	with other lifelong learning methods, such as LwF [17], iCaRL
862:   862	[27], WA [50], BiC [38], and an LReID method PTKP [8], our PPIPG method improves by 3.9% in 𝑠 R-1 and by 3.7% in 𝑠 mAP . These
863:   863	lifelong learning methods are specifically designed for the closedset problem of image classification, and this characteristic leads
865:   865	worth noting that our method outperforms ttqk by 6.5% in 𝑠 R-1
866:   866	and by 5.8% in 𝑠 mAP . This may be because the prompts enhance
867:   867	our model’s ability to extract knowledge. Our method has demonstrated the most excellent performance in four different datasets,
869:   869	of complex past knowledge memory and the processing requirements of new tasks. Moreover, as shown in Fig. 4, our method
870:   870	outperforms other methods in the last three training stages.It is
872:   872	Comparison with state-of-the-art methods
874:   874	We compared our proposed PP-IPG architecture with several stateof-the-art methods on the task of LVI-ReID. The performance of
878:   878	all methods were strictly adapted for this task based on their released codes, meanwhile, the backbone network, preprocessing
879:   879	methods, and training techniques of all methods are the same. The
888:   888	inherent parameter efficiency of the prompt learning mechanism,
892:   892	of tasks), our approach achieves a 91.4% reduction in parameters.
916:   916	the catastrophic forgetting problem without introducing a large
918:   918	uses the prompt pool module. In the first task stage, 𝑠 R-1 decreases
920:   920	extraction ability of the prompt pool is slightly inferior to that of
922:   922	ability of the prompt pool gradually emerges, achieving the best
924:   924	attributed to the better continuity of the prompt pool compared
925:   925	to the IPG module. The prompts in the prompt pool participate in
927:   927	prompts independently in each attention layer. After combining
953:   953	novel prompt-based framework. Specifically, the prompt pool is
955:   955	or domains. Meanwhile, each prompt maintains its independence,
957:   957	instance-level prompt generator can dynamically generate instancespecific prompts, thus breaking free from the limitation of a fixedsize prompt pool. These two modules are complementary and effectively work for robust feature representation learning. Ablation
964:   964	datasets illustrate the superiority of the proposed prompt learningbased PP-IPG method for the LVI-ReID task.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | nl -ba | sed -n '150,430p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | nl -ba | sed -n '430,720p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   150	within streaming domains. During the training phase, the
   151	model must both reduce the modality gap and prevent catastrophic forgetting of previous tasks. Some samples from
   152	previous tasks will be stored in a memory bank for replay
   153	purposes. It is worth noting that the models dealing with
   154	visible modality data and infrared modality data can either
   155	be shared or distinct.
   156	
   157	token and selects the associated prompt and classifier for effective
   158	task processing.
   159	The main contributions of our prompt learning-based PP-IPG
   160	framework for LVI-ReID are summarised as follows:
   161	• We introduce a prompt learning-based approach to tackle
   162	the LVI-ReID problem. A Prompt Pool (PP) module is designed to enable flexible retrieval of grouped prompts for
   163	diverse tasks or domains. This module dynamically routes
   164	task-specific prompts through parameter-isolated groups,
   165	effectively balancing cross-task knowledge sharing while
   166	preserving domain-specific discriminative features.
   167	• An Instance-Level Prompt Generator (IPG) is proposed to
   168	overcome the fixed size of proposed pool constraints by
   169	decomposing input token semantics and dynamically generating instance-level prompts. It Integrates task identifierguided domain encoding to achieve fine-grained prompt
   170	customization at the instance level. Furthermore, we design
   171	a query-key mechanism using query tokens to adaptively
   172	select the most suitable prompts and classifiers during the
   173	LVI-ReID inference phase.
   174	• Extensive experiments demonstrate that our PP-IPG approach
   175	outperforms state-of-the-art lifelong learning methods, LReID
   176	methods, and existing LVI-ReID solutions on the task of LVIReID. To the best of our knowledge, our work is the first
   177	attempt to integrate prompt learning with lifelong learning in the context of visible-infrared ReID, addressing the
   178	practical challenges of real-world ReID tasks.
   179	
   180	2
   181	
   182	RELATED WORK
   183	
   184	Lifelong Learning. Lifelong learning sequentially delivers multitask data to algorithms for dynamic knowledge acquisition under
   185	the constraint of accessing only current task data, aiming to learn
   186	a globally effective model while addressing challenges of evolving
   187	data distributions and catastrophic forgetting. Existing approaches
   188	
   189	954
   190	
   191	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
   192	
   193	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   194	
   195	Figure 2: Overview of Our Proposed LVI-ReID Method. Initially, the input image (visible or infrared) is processed into image
   196	patches with positional embeddings, which are then fed into the IPG (Instance-specific Prompt Generator) module and the
   197	pre-trained model. The class token output by the pre-trained model retrieves highly relevant prompts from the prompt pool,
   198	while the image tokens are concatenated with an additional query token and passed to the query attention block to compute a
   199	task identifier, which guides the IPG module in generating instance-level prompts. Subsequently, the instance-level prompts,
   200	selected prompts, class token, and image tokens are concatenated and fed into the self-attention module. Notably, each layer of
   201	the attention module is independently equipped with an IPG module to compute distinct instance-level prompts.
   202	are categorized into distillation-based methods [7, 11, 22, 25, 34, 46,
   203	59], regularization-based methods [15, 18, 19, 55], and structurebased methods [8, 14, 23, 33, 35, 49, 54]. Distillation-based methods preserve cross-task performance via knowledge distillation:
   204	iCaRL [34] combines limited historical data replay with current
   205	task training, while LwF [25] generates pseudo-labels from old
   206	models for storage-free optimization; Regularization-based methods protect critical knowledge by constraining parameter updates,
   207	such as EWC [18] slowing learning rates for task-critical parameters and Lee et al. [19] fusing old and new parameters through
   208	Bayesian integration; Structure-based methods explicitly construct
   209	task-specific parameters, exemplified by DyTox [8] dynamically
   210	expanding task-specific tokens with a shared backbone for crosstask feature generalization. Compared to task interference risks in
   211	shared parameter spaces, structure-based methods achieve superior
   212	knowledge isolation through parameter decoupling.
   213	Visible-Infrared Person Re-identification (ReID). Person
   214	re-identification is a task that aims to ascertain whether images
   215	of individuals captured from different camera viewpoints or at
   216	
   217	different times by the same camera correspond to the same individual. Early neural networks, primarily designed for classification tasks, naturally extracted global features from an input image when applied to ReID tasks [62]. Traditional ReID methods
   218	were insufficient for addressing the complexities of real-world
   219	scenarios. Based on deep neural networks, numerous methods
   220	[3, 28, 36, 44, 45, 56, 61, 63] have incorporated advanced feature
   221	learning strategies, including local features (e.g., pedestrian body
   222	parts or simple vertical region partitioning) and viewpoint information. These methods are tailored specifically for visible-spectrum
   223	image retrieval tasks, making them effective only in daytime scenarios and limiting their applicability in nighttime conditions. Thus,
   224	the Visible-Infrared Person Re-identification (VI-ReID) methods
   225	have been proposed. Wu et al. [43] designed a deep zero-padding
   226	framework to adaptively learn modality-sharable features for VIReID. Typical methods[9, 12, 50, 51, 64] adopt a two-stream network,
   227	where each modality has its own parameters in the shallow layer to
   228	extract modality-variant features and shares parameters in the deep
   229	layer to extract modality-invariant features, thus resolving the differences within and across modalities. Recently, VI-ReID methods
   230	
   231	955
   232	
   233	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   234	
   235	Zhenxi Luo, Guoqiang Xiao, Michael S. Lew, Song Wu
   236	
   237	based on Transformer have been proposed [4, 26, 30]. MSEL [30]
   238	combines a progressive learning strategy and a modality-shared
   239	feature loss to guide the Transformer to infer reliable identity information according to different modalities. Methods [6, 37, 38, 40]
   240	based on modality compensation differ from those focusing on
   241	the design of network architectures or loss functions. These methods use Generative Adversarial Networks (GANs) to generate one
   242	available modality into another modality and then fuse the two
   243	to reduce modality differences, thereby achieving the purpose of
   244	cross-modal image alignment. However, the performance of these
   245	methods depends on the high quality of the images generated.
   246	Prompt learning. Inspired by the success of text prompting
   247	techniques [2, 20] in Natural Language Processing, prompt learning
   248	methods have also been widely applied in visual tasks and have
   249	significantly improved their performance. Research on continual
   250	learning that combines prompting mechanisms with Vision Transformers is developing rapidly. L2P [42] first introduced the concept
   251	of the prompt pool to dynamically select prompts for continual
   252	learning tasks, guiding the pre-trained ViT backbone to utilize
   253	the learned representations. DualPrompt [41] uses two types of
   254	prompts to address catastrophic forgetting, where G-Prompt is
   255	used for general knowledge, and E-Prompt is used for task-specific
   256	knowledge. S-prompt [39] introduces an independent cross-domain
   257	prompting learning scheme and employs the K-Nearest Neighbor
   258	algorithm at the inference stage to obtain task identifiers. DAP
   259	[17] adopts an adaptive prompt generator to generate appropriate
   260	prompts for each instance, thereby solving the domain scalability
   261	problem caused by the prompt pool. In this work, our method utilizes both the prompt pool and the instance-level prompt generator
   262	to address catastrophic forgetting and explore the application of
   263	visual prompts in the LVI-ReID domain.
   264	
   265	second half comprises infrared modality images. In the rehearsal
   266	mechanism, the same setup applies to replay samples, denoted as
   267	𝐵𝑜 , where 𝑟 ∈ {0, 1, · · · , 𝑡 − 1}. We denote the input
   268	{(𝑥𝑖𝑚,𝑟 , 𝑦𝑖𝑟 )}𝑖=1
   269	image as 𝑥 𝑚 ∈ R𝐶 ×𝐻 ×𝑊 , where 𝐶, 𝐻 , and 𝑊 represent the channel, the height and the width dimension of the image, respectively,
   270	and 𝑚 represents the modality information. The input image 𝑥 𝑚 is
   271	transformed into image embeddings 𝑥𝑒𝑚 ∈ R𝐿×𝑑 through a linear
   272	projection, where 𝐿 represents the number of tokens and 𝑑 stands
   273	for the embedding dimension. Subsequently, the positional encoding information is added to the image embeddings 𝑥𝑒𝑚 . Then, it is
   274	concatenated with the class token for global semantic representation learning, resulting in 𝑥𝑒𝑚 ∈ R (𝐿+1) ×𝑑 . Before being fed into
   275	the Transformer encoder SAB(·) module, we design a query-key
   276	mechanism to automatically select prompts 𝑃𝐾 from the prompt
   277	pool that exhibits high similarity to the current input image 𝑥𝑒𝑚 .
   278	Each prompt in the prompt pool can maintain a certain degree of
   279	independence, thereby effectively differentiating task-specific or
   280	domain-specific knowledge. Meanwhile, an instance-level prompt
   281	generator is proposed, which takes the embedding information of
   282	the task identifier 𝑡 as the supervise information to linearly transform the image tokens into the adaptive prompts 𝑃𝐼 , which can
   283	enhance the model’s ability to capture intra - domain knowledge.
   284	We concatenate 𝑥𝑒𝑚 , 𝑃𝐾 and 𝑃𝐼 to obtain the feature representation
   285	of 𝑥𝑝𝑚 = [𝑃𝐼 ; 𝑃𝐾 ; 𝑥𝑒𝑚 ], which is the input of the SAB(·) module. After passing through the transformer encoder SAB(·) module, the
   286	class token is extracted and used as the feature to be fed into the
   287	classifier.
   288	
   289	3.3
   290	
   291	3 METHODOLOGY
   292	3.1 Problem formulation
   293	In this section, we provide a formal definition of the LVI-ReID
   294	task. The dataset of continual tasks in the LVI-ReID is represented
   295	as {𝐷 𝑖 }𝑇𝑖=1 , where 𝑖 denotes the task identifier, and 𝐷 𝑖 consists of
   296	𝑖
   297	𝑖
   298	both the training dataset 𝐷𝑡𝑟𝑎𝑖𝑛
   299	and the test dataset 𝐷𝑡𝑒𝑠𝑡
   300	. During the training process of LVI-ReID, let the current task identi𝑡
   301	fier be 𝑡, and the training dataset is formally defined as 𝐷𝑡𝑟𝑎𝑖𝑛
   302	=
   303	|𝐷 𝑡
   304	
   305	𝐵𝑛∑︁
   306	+𝐵𝑜
   307	∑︁ 𝑝 𝑝 ∑︁
   308	1
   309	𝑛 𝑛
   310	L𝑤𝑟𝑡 = 𝑛
   311	log(1 + exp( 𝑤𝑖 𝑗 𝑑𝑖 𝑗 −
   312	𝑤𝑖𝑘
   313	𝑑𝑖𝑘 ))
   314	𝑜
   315	𝐵 + 𝐵 𝑖=1
   316	𝑖𝑗
   317	𝑖𝑘
   318	
   319	|
   320	
   321	𝑝
   322	
   323	{(𝑥𝑖𝑚,𝑡 , 𝑦𝑖𝑡 )}𝑖=1𝑡𝑟𝑎𝑖𝑛 , where 𝑥𝑖𝑚,𝑡 represents the images from visible
   324	and infrared modality 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟 }, and 𝑦𝑖𝑡 indicates the corresponding identity label. Our proposed PP-IPG framework for the
   325	LVI-ReID task is based on a rehearsal mechanism, where a small
   326	amount of significant data from previous tasks is retained in the task
   327	𝑡 to facilitate lifelong learning. During the inference process, we
   328	evaluate the performance of our proposed method on the LVI-ReID
   329	𝑖
   330	𝑡 from all previous tasks.
   331	task using the test dataset {𝐷𝑡𝑒𝑠𝑡
   332	}𝑖=1
   333	
   334	3.2
   335	
   336	Baseline
   337	
   338	The baseline model comprises self-attention blocks SAB(·), a batch
   339	normalization (BN) layer, and a task-specific classifier 𝐶 (·). Notably, both the batch normalization layer and the classifier expand
   340	dynamically when the number of tasks increases. To learn distinctive high-level features, we use the weighted regularized triplet
   341	loss function L𝑤𝑟𝑡 [52] for the training dataset, the formula is as
   342	follows:
   343	
   344	exp(𝑑𝑖 𝑗 )
   345	
   346	𝑛)
   347	exp(𝑑𝑖𝑘
   348	
   349	(1)
   350	
   351	𝑝
   352	𝑛
   353	𝑤𝑖 𝑗 = Í
   354	𝑛
   355	𝑝 , 𝑤𝑖𝑘 = Í 𝑛
   356	𝑝
   357	𝑑𝑖𝑘 ∈ N𝑖 exp(𝑑𝑖𝑘 )
   358	𝑑𝑖 𝑗 ∈ P𝑖 exp(𝑑𝑖 𝑗 )
   359	
   360	In each training batch, for every anchor sample 𝑥𝑖𝑚 , (𝑖, 𝑗, 𝑘) denotes
   361	a triplet. It should be noted that in the channel-mixed learning
   362	strategy, 𝑗 and 𝑘 can be from either the same modality or different
   363	modalities. For the anchor 𝑥𝑖𝑚 , P𝑖 stands for the corresponding
   364	𝑝 𝑛
   365	positive set, and N𝑖 represents the negative set. 𝑑𝑖 𝑗 /𝑑𝑖𝑘
   366	indicates
   367	the pairwise distance of a positive/negative sample pair. Here, 𝑑𝑖 𝑗
   368	represents the Euclidean distance between two samples, which is
   369	expressed as 𝑑𝑖 𝑗 = ∥𝑆𝐴𝐵(𝑥𝑖𝑚 ) − 𝑆𝐴𝐵(𝑥 𝑚
   370	𝑗 )∥ 2 . The prediction output
   371	for a given task is expressed as 𝑃 𝑛 (·) = 𝜎 (C(BN(SAB(·)))), where
   372	𝜎 is the softmax activation function. The classification results for
   373	both rehearsal data and newly introduced data are evaluated using
   374	
   375	Approach overview
   376	
   377	The overview of our proposed PP-IPG framework for LVI-ReID is illustrated in Fig.2. In each iteration of the training process, 𝑁 images
   378	𝑡
   379	per modality from 𝐾 identities in the training dataset 𝐷𝑡𝑟𝑎𝑖𝑛
   380	are ran𝑚,𝑡 𝑡 𝐵𝑛
   381	𝑛
   382	domly selected and denoted as {(𝑥𝑖 , 𝑦𝑖 )}𝑖=1 , with 𝐵 = 2 ×𝐾 × 𝑁 ,
   383	where the first half consists of visible modality images and the
   384	
   385	956
   386	
   387	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
   388	
   389	the cross-entropy loss function L𝐼 𝐷 :
   390	L𝐼 𝐷 = −
   391	
   392	𝐵𝑛∑︁
   393	+𝐵𝑜
   394	
   395	𝑦𝑖 · log(𝑃 𝑛 (𝑥𝑖𝑚 ))
   396	
   397	participate in the training process during each training iteration.
   398	Specifically, we designed a Query-Key mechanism to select the
   399	prompts from the prompt pool. Based on the features of the input
   400	images, we select the 𝐾 most similar prompts 𝑃𝐾 to participate in
   401	parameter updates. The prompt selection process can be regarded
   402	as a clustering operation to some extent. Input data with similar features tend to activate the same or high-similarity prompt
   403	sets when processed by the designed Query-Key mechanism. This
   404	way of matching corresponding prompt sets based on similarity
   405	can better adapt to the prompt requirements of different task inputs. Furthermore, Li and Liang [24] demonstrated that mapping
   406	the prompt parameters through a multi-layer perceptron (MLP)
   407	significantly improves model performance by introducing a more
   408	structured optimization process. However, their work also revealed
   409	that directly optimizing the prompt 𝑃𝐾 could result in instability
   410	during the training process, making it challenging to achieve reliable performance, and highlighted that directly reorganizing the
   411	parameters of the prompt often lacks generalizability across different tasks or datasets, thereby limiting its applicability. To address
   412	these challenges, we propose incorporating residual connections
   413	into the design of the prompt pool. The residual connections in
   414	our prompt pool not only preserve the original information of the
   415	prompt, ensuring that the foundational knowledge remains intact,
   416	but also facilitate enhanced feature fusion, thereby improving the
   417	model’s ability to adapt and generalize. Formally, the process of the
   418	designed prompt pool can be expressed as follows:
   419	
   420	(2)
   421	
   422	𝑖=1
   423	
   424	where, 𝑥𝑖𝑚 represents the rehearsal data and current data of the
   425	two modalities, and 𝑦𝑖 is its corresponding label. In the rehearsal
   426	mechanism, we design a L𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 loss function to preserve the
   427	similarity of the replay samples feature representations from the
   428	old model and the new model. We also design a L𝑘𝑑 loss function
   429	to align the feature representations in the latent space of the two
   430	modalities by bringing the prediction probabilities of the visible

 succeeded in 0ms:
   430	modalities by bringing the prediction probabilities of the visible
   431	modality data closer to those of the infrared modality data. Let
   432	𝑃 𝑜 (·) be the prediction probability of the old model, and 𝑃 𝑛 (·) be
   433	the prediction probability of the new model, and the formula of
   434	L𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 and L𝑘𝑑 are as follows:
   435	𝐵𝑜
   436	
   437	
   438	𝑇 2 ∑︁
   439	L𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙 = 𝑜
   440	JS 𝑃 𝑛 (𝑥𝑖𝑟 ) ∥ SD 𝑃 𝑜 (𝑥𝑖𝑟 )
   441	𝐵 𝑖=1
   442	L𝑘𝑑 =
   443	
   444	𝐵𝑛 /2
   445	
   446	
   447	2 · 𝑇 2 ∑︁  𝑛 𝑣𝑖𝑠,𝑡
   448	𝑖𝑟,𝑡
   449	JS 𝑃 (𝑥𝑖 ) ∥ SD 𝑃 𝑛 (𝑥𝑖+𝐵
   450	𝑛 /2 )
   451	𝑛
   452	𝐵
   453	𝑖=1
   454	
   455	(3)
   456	
   457	(4)
   458	
   459	where 𝑆𝐷 (·) represents the stop gradient operation, 𝐽𝑆 (· ∥ ·) denotes the Jensen-Shannon divergence, and 𝑇 is the temperature
   460	parameter used to control the sharpness of the probability distribution. Additionally, 𝑥𝑖𝑣𝑖𝑠,𝑡 and 𝑥 𝑖𝑟,𝑡
   461	𝑗 refer to the visible and infrared
   462	images, respectively. Therefore, the total loss function for the baseline model is formulated as:
   463	L𝑏𝑎𝑠𝑒 = (1 − 𝜆) · (L𝐼 𝐷 + L𝑤𝑟𝑡 + L𝑘𝑑 ) + 𝜆 · L𝑟𝑒ℎ𝑒𝑎𝑟𝑠𝑎𝑙
   464	|𝑁
   465	
   466	′
   467	
   468	𝑃𝐾 = MLP(𝑃𝐾 ) + 𝑃𝐾
   469	
   470	(7)
   471	
   472	(5)
   473	
   474	|
   475	
   476	3.5
   477	
   478	where, 𝜆 = |𝑁𝑜𝑙𝑑 | , and |𝑁𝑜𝑙𝑑 | denotes the number of identities
   479	𝑎𝑙𝑙
   480	from domains 1 to 𝑡 − 1, and |𝑁𝑎𝑙𝑙 | represents the total number
   481	of identities across domains 1 to 𝑡. The parameter 𝜆 reflects the
   482	proportion of old identities (up to domain 𝑡 − 1) relative to the total
   483	number of identities across all domains (up to domain 𝑡).
   484	
   485	3.4
   486	
   487	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   488	
   489	Instance-level Prompt Generator
   490	
   491	Conventional prompt-based lifelong learning approaches suffer
   492	from two critical limitations. Firstly, their inherent dependency
   493	on fixed prompt pool architectures necessitates prior hyperparameter tuning for target domains, which contradicts the practical
   494	assumption of lifelong learning where prior domain knowledge
   495	is unavailable .Secondly, the memory footprint increases linearly
   496	as the prompt pool expands dynamically. To address these challenges, we propose an instance-aware adaptive prompting strategy,
   497	which enhances cross-domain knowledge retention while enabling
   498	fine-grained instance-level feature adaptation. To achieve this, we
   499	design an Instance-level Prompt Generator (IPG) module that provides fine-grained guidance, assisting the model efficiently in leveraging learned representations and better capturing domain-specific
   500	knowledge, thereby significantly improving the model’s domain
   501	generalization capability. The IPG module dynamically generates
   502	instance-specific prompts based on input features, breaking the
   503	limitation of traditional fixed-size prompt pools and enabling the
   504	model to adapt to more diverse input scenarios and complex tasks.
   505	As illustrated in Fig. 2, the IPG module consists of a transposed
   506	operation (T), layer normalization (LN), a multilayer perceptron
   507	(MLP), and a linear transformation (LT). By processing the input
   508	image features, the IPG module transforms specific instance-level
   509	information into a set of learnable vectors, which aligns with the
   510	design principles of [17]. These generated vectors serve as instancelevel prompts 𝑃𝐼 , dynamically guiding the model to adapt to diverse
   511	input instances, thereby enhancing the model’s adaptability and
   512	
   513	Prompt Pool
   514	
   515	Previous works [41, 42] have demonstrated that the design of the
   516	prompt pool has the ability to learn the shared knowledge among
   517	all tasks while retaining the specific knowledge of particular tasks.
   518	Moreover, numerous studies [1, 5, 16] have proved that visual
   519	prompt learning has the advantages of simple operation and low
   520	additional resources requirement, and it can potentially drive the
   521	original model to extract significant visual features. Therefore, we
   522	consider leveraging the advantages of prompt learning to help the
   523	model focus on potential semantic representation correspondences
   524	between modality-specific information. Thus, in this work, we introduce the prompt pool to address the catastrophic forgetting problem
   525	existing in LVI-ReID. At the same time, the selected prompts are
   526	used to guide the model in preserving and leveraging the semantic
   527	representation correspondences among modality-specific information. Formally, the prompt pool consists of a set of learnable vectors,
   528	and it can be expressed as:
   529	𝑃 = {𝑃 1, 𝑃2, · · · , 𝑃𝑀 }
   530	(6)
   531	𝐿
   532	×𝑑
   533	𝑝
   534	where 𝑃 𝑗 ∈ R
   535	is a prompt with 𝐿𝑝 tokens and has the same
   536	dimension as 𝑥𝑒𝑚 . In our work, not all prompts in the prompt pool
   537	
   538	957
   539	
   540	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   541	
   542	Zhenxi Luo, Guoqiang Xiao, Michael S. Lew, Song Wu
   543	
   544	generalization performance. The process of the proposed IPG module can be formally expressed as follows:
   545	𝑃𝐼 = 𝐿𝑇 (𝑀𝐿𝑃 (𝐿𝑁 (𝑥𝑒𝑚 ) T );𝜓 (𝑡)) T
   546	= (𝛾𝑡 · 𝑀𝐿𝑃 (𝐿𝑁 (𝑥𝑒𝑚 ) T ) + 𝛽𝑡 ) T ∈ R𝐿𝑝 ×𝑑
   547	
   548	(8)
   549	
   550	where, the 𝐿𝑇 represents an affine transformation operation that
   551	computes the scaling parameter 𝛾𝑡 and the bias parameter 𝛽𝑡 . Specifically, the 𝐿𝑇 maps the task identifier information 𝑡 into the scaling
   552	parameter 𝛾𝑡 and the bias parameter 𝛽𝑡 through an affine transformation layer, which can be formally expressed as [𝛾𝑡 , 𝛽𝑡 ] = 𝜓 (𝑡).
   553	Subsequently, an affine transformation is performed on the instancelevel image embeddings based on the two parameters, where the
   554	task identifier information 𝑡 is obtained through a query-key mechanism. Notably, each Transformer layer in our model independently
   555	possesses an IPG module.
   556	
   557	3.6
   558	
   559	Figure 3: The inference stage of our LVI-ReID method. The
   560	frozen pre-trained model uses the class token as a query to
   561	match the corresponding prompt. Meanwhile, the image token is additionally processed through QAB(·) to calculate the
   562	task identifier, which guides the output of the IPG module.
   563	
   564	Query-Key Mechanism
   565	
   566	To enable input instances to autonomously select prompts from
   567	the prompt pool and the corresponding classifiers, we design a
   568	query-key mechanism. The works of L2P [42] and DualPrompt
   569	[41] have demonstrated that the query-key mechanism can impact
   570	the model’s lifelong learning ability. Building on this insight, we
   571	utilize the frozen pre-trained self-attention block SAB∗ (·) to extract
   572	query features for prompt selection. Additionally, we introduce a
   573	trainable attention block, denoted as the query attention block
   574	QAB(·), which participates in gradient backpropagation further
   575	to refine query features for classifier selection and task identifier.
   576	Specifically, we take the class token output by SAB∗ (·) as the query
   577	for prompts, and the image feature tokens 𝑓𝑖 output by SAB∗ (·) are
   578	passed into QAB(·) after an additional query token is embedded.
   579	The resulting query token 𝜃𝑞 from QAB(·) is then utilized as the
   580	query for selecting classifiers. The formalized process is described
   581	as follows:
   582	𝑐𝑙𝑠, 𝑓𝑖 = SAB∗ (𝑥𝑖𝑚,𝑡 )
   583	𝑓𝑝,𝑖 = 𝑄 𝑝 (𝑥𝑖 ) = 𝑐𝑙𝑠 ∈ R𝑑
   584	
   585	final loss function is defined as follows:
   586	L = L𝑏𝑎𝑠𝑒 + L𝑝𝑢𝑙𝑙
   587	
   588	where, L𝑏𝑎𝑠𝑒 can be seen in Equation 5, and L𝑝𝑢𝑙𝑙 can be seen in
   589	Equation 10.
   590	
   591	4 EXPERIMENTS
   592	4.1 Datasets & Evaluation protocols
   593	Dataset. To evaluate the performance of the proposed framework,
   594	four visible-infrared datasets were adopted, namely RegDB [31],
   595	SYSU-MM01 [43], LLCM [57], and HITSZ-VCM [27]. During the
   596	data preprocessing stage, when dealing with the RegDB and SYSUMM01 datasets, we followed the guidelines of data preprocessing outlined in [53]. For the LLCM dataset, we carried out operations in accordance with the procedures of data preprocessing
   597	described in [57]. HITSZ-VCM is a video-based cross-modal person
   598	re-identification method. To ensure that the HITSZ-VCM dataset
   599	can apply to the LVI-ReID task, we randomly selected one frame
   600	from each pedestrian tracklet as input, thereby converting the video
   601	data format of HITSZ-VCM into an image format.
   602	Evaluation protocol. We use the mean Average Precision (mAP)
   603	and the Cumulative Matching Characteristic curve (CMC) as the
   604	evaluation metrics for these four datasets for the task of LVI-ReID.
   605	For the metric of CMC, we calculated the proportion of the number of correctly retrieved images using the Rank-1 accuracy (R-1).
   606	We also report the average incremental accuracy, denoted as 𝑠 𝑅−1 ,
   607	𝑠𝑚𝐴𝑃 .
   608	
   609	(9)
   610	𝑑
   611	
   612	𝑓𝑐,𝑖 = 𝑄𝑐 (𝑥𝑖 ) = QAB(𝜃𝑞 ; 𝑓𝑖 ) ∈ R
   613	
   614	where, 𝑄 𝑝 (𝑥𝑖 ) represents the query function for prompts, and
   615	𝑄𝑐 (𝑥𝑖 ) represents the query function for classifiers. Each prompt
   616	𝑀 ∈ R𝑀 ×𝑑
   617	and classifier has a learnable key, denoted as {𝑘𝑒𝑦𝑝 }𝑖=1
   618	𝑡
   619	𝑡
   620	×𝑑
   621	and {𝑘𝑒𝑦𝑐 }𝑖=1 ∈ R , respectively. During the training stage, we
   622	design a L𝑝𝑢𝑙𝑙 loss function to ensure they will independently approach 𝑓𝑝,𝑖 and 𝑓𝑐,𝑖 in the vector space, which is defined as follows:
   623	𝐵𝑛∑︁
   624	+𝐵𝑜
   625	1
   626	L𝑝𝑢𝑙𝑙 = 𝑛
   627	((1 − CosSim(𝑓𝑝,𝑖 , 𝑘𝑒𝑦𝑝 ))
   628	𝐵 + 𝐵𝑜 𝑖=1
   629	
   630	(11)
   631	
   632	(10)
   633	
   634	4.2
   635	
   636	+ (1 − CosSim(𝑓𝑐,𝑖 , 𝑘𝑒𝑦𝑐 )))
   637	
   638	Implementation details
   639	
   640	As shown in Fig. 2, we adopted 10 self-attention blocks pre-trained
   641	on ImageNet as the backbone network. All these modules have
   642	12 attention heads, and their embedding dimension is set to 768.
   643	To ensure that the knowledge of continuous learning tasks is not
   644	prematurely leaked, the frozen pre-trained model is trained only on
   645	the dataset RegDB of the first task. In terms of data augmentation,
   646	images are resized to 256×128 through operations such as zeropadding, random cropping, and random horizontal flipping. In each
   647	
   648	where, CosSim(·) represents the cosine similarity. During the inference phase, as illustrated in Fig. 3, the input image is processed by
   649	the frozen pre-trained model to obtain the class token and image
   650	token. The class token is used to select the most relevant prompts
   651	from the prompt pool, while the image token, after being concatenated with the queue token 𝜃𝑞 , is fed into QAB(·) to compute the
   652	task identifier and select the appropriate classifier. As a result, our
   653	
   654	958
   655	
   656	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
   657	
   658	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   659	
   660	Table 1: Comparison with State-of-the-Art Methods.All the results are implemented based on our baseline. The training
   661	sequence is as follows: RegDB → SYSU-MM01 → LLCM → HITSZ-VCM.
   662	Methods
   663	JointTrain
   664	FineTune
   665	LWF[25]
   666	iCaRL[34]
   667	WA[60]
   668	BIC[47]
   669	PTKP[10]
   670	TTQK[48]
   671	Ours
   672	
   673	RegDB
   674	R-1 mAP
   675	70.7
   676	71.1
   677	3.1
   678	6.4
   679	27.7
   680	28.0
   681	52.0
   682	54.0
   683	51.2
   684	53.7
   685	41.8
   686	43.7
   687	53.4
   688	56.1
   689	55.4
   690	55.7
   691	59.0 58.8
   692	
   693	SYSU-MM01
   694	R-1 mAP
   695	40.9
   696	44.4
   697	10.9
   698	15.2
   699	29.3
   700	30.6
   701	29.8
   702	31.7
   703	28.5
   704	28.4
   705	29.3
   706	30.4
   707	32.2
   708	33.9
   709	24.0
   710	28.2
   711	32.7
   712	35.2
   713	
   714	LLCM
   715	R-1 mAP
   716	50.4
   717	56.3
   718	12.3
   719	16.9
   720	31.3

codex
第四篇的机制读完了。它的核心问题定义很直接：传统提示池能存共享知识但固定容量不够，实例级生成能适配个体但缺长期共享，所以用提示池和实例级生成器互补，并用查询键机制在测试时无任务标签选择提示和分类器。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -n \"("'^'"\\s*[0-9]+\\s+[0-9]+\\.?\\s+|Proposed|Method|Framework|Local|Residual|Attention|Experiment|Ablation|Loss|Overview|module|Module|Transformer)\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:     1	Local-Aware Residual Attention Vision Transformer for
13:    13	and robustness of cross-modal pedestrian matching. To solve the above problem, we propose the LocalAware Residual Attention Vision Transformer (LAReViT) to enhance the learning of fine-grained local and
15:    15	the Local-Aware Residual (LAR) Module, which uses a novel Local Residual Attention (LRA) mechanism, is
17:    17	fine-grained local information lost in lower-level visual features, the LRA in the LAR module adopts novel
18:    18	attention residual connections. Additionally, we propose a Positional Channel Reconstruction (PCR) Module
21:    21	Finally, the novel Center Aggregation Loss (CAL) is designed to reduce modality discrepancies moderately
53:    53	Xuecheng Hua, Ke Cheng, Gege Zhu, Hu Lu, Yuanquan Wang, and Shitong Wang. 2025. Local-Aware Residual
54:    54	Attention Vision Transformer for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput.
90:    90	the network, the attention maps become increasingly globalized. Vision Transformer (ViT)
97:    97	Figure 2 presents a visual representation of the feature similarity matrices for ViT and LocalAware Residual Attention Vision Transformer (LAReViT) across different modalities. The
103:   103	Local-Aware Residual Attention Vision Transformer for VI-ReID
113:   113	precisely captures effective cross-modal person feature. CNN, convolutional neural network; LAReViT, LocalAware Residual Attention Vision Transformer; ViT, vision transformer.
131:   131	designed a Local-Aware Residual (LAR) Module with a Local Residual Attention (LRA)
139:   139	proposed a Positional Channel Reconstruction (PCR) Module that reweights each patch block
164:   164	pedestrian features across different modalities through the proposed Center Aggregation Loss
167:   167	Local-Aware Residual Attention Vision Transformer for VI-ReID
177:   177	— The LAR Module, which is based on a novel LRA mechanism, is proposed to enhance the
180:   180	— The PCR Module is designed to promote the network to focus on effective patches through
215:   215	Transformer in Person ReID
217:   217	Transformer [47] was a revolutionary model designed to be applied across various tasks in natural language processing. Han et al. [15] and Khan et al. [22] investigated the application of the
239:   239	Transformer in VI-ReID
261:   261	Combining CNNs and Transformer in Person Re-ID
265:   265	address. Consequently, many studies have proposed foundational CNN-Transformer hybrid models
268:   268	field of ReID, SPH [13] was the first to propose the end-to-end CNN-Transformer Hybrid framework
271:   271	incorporates both a CNN-based stream and a Transformer-based stream. The Transformer-based
275:   275	Local-Aware Residual Attention Vision Transformer for VI-ReID
279:   279	Fig. 3. Overview of our LAReViT for VI-ReID. The PCR Module is designed to reweight the input patches
281:   281	LAR Module on the local-based input sequence to enhance the exploration and capture of low-level informative cues, aiming to alleviate the issue of over-globalization in transformer features. Finally, the CAL
285:   285	CNN structure and global information of the Transformer structure to mine feature saliency
289:   289	and Transformers inevitably increases model complexity and computational costs. Additionally,
292:   292	to CNN-Transformer hybrid models, our proposed LAReViT utilizes a pure ViT as its backbone,
293:   293	without incorporating additional CNNs. Though utilizing the proposed PCR and LAR modules,
294:   294	we enhance the local perception capabilities of Transformer and achieve improved performance
296:   296	3 The Proposed Method
300:   300	proposed PCR Module, described in Section 3.1, reweights features within patches at the channel
302:   302	Module, based on the LRA mechanism in transformer, is described in Section 3.2. This module
319:   319	PCR Module
329:   329	Consequently, the proposed PCR Module enables a more comprehensive extraction of cross-modal
336:   336	is set to 364 and serves as the effective input sequence length for the Transformer. The Transformer

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '270,560p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   270	feature learning. Subsequently, in the Clothing-Change person re-identification field, CT-Net [52]
   271	incorporates both a CNN-based stream and a Transformer-based stream. The Transformer-based
   272	stream to encourage the CNN-based stream to explore more detailed ID-unique features when
   273	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   274	
   275	Local-Aware Residual Attention Vision Transformer for VI-ReID
   276	
   277	146:7
   278	
   279	Fig. 3. Overview of our LAReViT for VI-ReID. The PCR Module is designed to reweight the input patches
   280	of images and enhance the expression of effective fine-grained local information. Then, we employ the
   281	LAR Module on the local-based input sequence to enhance the exploration and capture of low-level informative cues, aiming to alleviate the issue of over-globalization in transformer features. Finally, the CAL
   282	is applied to facilitate the integration of different modalities and enhance the representation of crossmodal data.
   283	
   284	the clothing information is no reliable. In VI-ReID, CMIT [10] aggregates local features of the
   285	CNN structure and global information of the Transformer structure to mine feature saliency
   286	representation. MTECN [29] is designed to adequately extract and fuse multi-modal information
   287	from both local and global perspectives, thereby effectively reducing the disparities between
   288	modalities. The aforementioned methods are outstanding. However, the combination of CNNs
   289	and Transformers inevitably increases model complexity and computational costs. Additionally,
   290	models that integrate these two distinct architectures face challenges in hyperparameter selection
   291	and training stability, which can lead to decreased efficiency in feature extraction. In contrast
   292	to CNN-Transformer hybrid models, our proposed LAReViT utilizes a pure ViT as its backbone,
   293	without incorporating additional CNNs. Though utilizing the proposed PCR and LAR modules,
   294	we enhance the local perception capabilities of Transformer and achieve improved performance
   295	without introducing extra parameters.
   296	3 The Proposed Method
   297	In this section, we present the technical details of the proposed LAReViT. The structure diagrams
   298	of the model and its components are shown in Figures 3 and 4. LAReViT encompasses three
   299	innovations that work together to extract more comprehensive discriminative features. Firstly, the
   300	proposed PCR Module, described in Section 3.1, reweights features within patches at the channel
   301	level and reinforces the advantages of local receptive fields in convolution. Additionally, the LAR
   302	Module, based on the LRA mechanism in transformer, is described in Section 3.2. This module
   303	enables fine-grained learning of local features and explores and captures low-level informative
   304	clues from shallow-layer networks. Finally, the CAL, introduced in Section 3.3, aims to moderate
   305	the differences between modalities, which promotes the integration of different modalities and
   306	enhances the representation of cross-modal data.
   307	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   308	
   309	146:8
   310	
   311	X. Hua et al.
   312	
   313	Fig. 4. Details of PCR. It reconstructs the weights of each patch through operations such as pooling and fully
   314	connected layers. This allows the network to emphasize effective channels, thereby effective the learning of
   315	fine-grained local information.
   316	
   317	3.1
   318	
   319	PCR Module
   320	
   321	Existing transformer models employ convolutional kernels to perceive the visual field within
   322	each 16 × 16 patch. However, these models often lean towards representation learning based on
   323	global attention between patches, thereby underutilizing the advantages of local receptive fields
   324	brought by convolutional kernels. For cross-modal matching, this local information often contains
   325	crucial person discriminative information. We aim to emphasize certain patches that contain key
   326	clues, and thus, the PCR is proposed to guide the network by effectively mining fine-grained local
   327	contextual features. Different from existing transformer models, PCR reweights each patch block in
   328	ViT at the channel level, further enhancing the alignment and learning of effective local features.
   329	Consequently, the proposed PCR Module enables a more comprehensive extraction of cross-modal
   330	person features.
   331	First, we set the dataset to S. According to the standard process of ViT [7], we reshape the
   332	2
   333	dataset image S ∈ R𝐵×𝐻 ×𝑊 ×𝐶 into a sequence of flattened 2D patches S𝑝 ∈ R𝐵×𝑁 × (𝑃 ·𝐶 ) , where
   334	B represents the batch size, (𝐻,𝑊 ) is the resolution of the original image, C is the number of
   335	channels, (𝑃, 𝑃) is the resolution of each image patch. N is the resulting number of patches, which
   336	is set to 364 and serves as the effective input sequence length for the Transformer. The Transformer
   337	uses constant latent vector size 𝐷 = 𝑃 2 · 𝐶 through all of its layers, so we flatten the patches and
   338	map to D dimensions with a trainable linear projection. We refer to the output of this projection as
   339	the patch embeddings and denote it as the input for the PCR.
   340	Specifically, given the patch embeddings S𝑝 , we denote them as X. As shown in Figure 4, the
   341	spatial relationship weight of patch embedding is efficiently modeled by applying Average Pooling
   342	(AVG), Maximum Pooling (MAP), and Projection Operation E along channel dimension from
   343	network input. We facilitate the interaction of information from different patches across diverse
   344	spatial descriptors, to obtain dynamic selection values W. Features from different patches are
   345	adaptively emphasized by utilizing these selection values to reweight them:
   346	W𝑎𝑣𝑔 = 𝐴𝑉 𝐺 (X),
   347	
   348	(1)
   349	
   350	W𝑚𝑎𝑝 = 𝑀𝐴𝑃 (X),
   351	
   352	(2)
   353	
   354	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   355	
   356	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '560,820p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
codex
第五篇的前 8 页只读到方法开头和 PCR 模块前半段，方法显然跨过第 8 页。为保证完整，我继续补读后续页，把 PCR、LAR 和 CAL 三个部分收完。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 9 -l 13 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,340p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Local-Aware Residual Attention Vision Transformer for VI-ReID
     2	
     3	146:9
     4	
     5	W𝜒 = E (𝛼 · W𝑎𝑣𝑔 ⊕ (1 − 𝛼) · W𝑚𝑎𝑝 ),
     6	
     7	(3)
     8	
     9	𝑃𝐶𝑅(X) = W𝜒 ⊗ X + X,
    10	
    11	(4)
    12	
    13	where the input X ∈ R [𝐵,𝑃,𝐷 ] , B represents the batch size, P is the number of patches divided into
    14	images, and D denote the dimensions of the embedding. We set parameter 𝛼 equal to 0.3. Projection
    15	Operation E is composed of linear layers and activation functions.
    16	PCR guides Transformer models in handling local information more effectively in cross-modal
    17	matching tasks, particularly by placing emphasis on patches that contain key clues. This contributes
    18	to enhancing the capabilities of the model to capture effective information across different modalities
    19	and guiding the learning of discriminative fine-grained person features.
    20	3.2
    21	
    22	LAR Module in LAReViT
    23	
    24	For VI-ReID tasks, comprehensive pedestrian global features are crucial. Similarly, discriminative
    25	fine-grained local features should also be valued. ViT demonstrates outstanding performance in
    26	addressing image tasks with long-range dependencies and global contextual information. Therefore,
    27	by incorporating the transformer into VI-ReID task, we can effectively utilize its self-attention
    28	mechanism to model the global relationships present in cross-modal images. However, when
    29	applying ViT to VI-ReID domain, there are two limitations due to the inherent constraints of the
    30	network structure: (1) The self-attention mechanism in ViT is characterized by feature collapse
    31	in deeper layers leading to excessive similarity among feature patches, causing the vanishing of
    32	some shallow fine-grained visual feature information [46, 75]. In VI-ReID, shallow-layer low-level
    33	cues typically represent details such as color and texture [71]. However, these low-level cues are
    34	crucial for cross-modal matching tasks. (2) As the network depth increases, ViT aggregates features
    35	between patches at a global scale, resulting in lost focus on local information and capturing extra
    36	irrelevant information [36]. Consequently, the loss of certain effective local features ultimately
    37	results in the over-globalization of ViT feature. In this context, we propose LAR as a solution to
    38	address the issues of over-globalization and feature collapse in transformer attention mechanism.
    39	LAR aims to study local features, such as the head, trunk, and legs of individuals, which are
    40	obtained through splitting and masking operations based on the proposed Regional Division
    41	(RD). Formally, by performing the PCR Module with the RD technique, we divide the global patch
    42	embedding feature X into local parts X L :
    43	[ Xℎ𝑒𝑎𝑑 , X𝑡𝑟𝑢𝑛𝑘 , X𝑙𝑒𝑔 ] = RD ( X ),
    44	
    45	(5)
    46	
    47	X L = [ Xℎ𝑒𝑎𝑑 , X𝑡𝑟𝑢𝑛𝑘 , X𝑙𝑒𝑔 ],
    48	
    49	(6)
    50	
    51	where X represents the dataflow within the network, which currently denotes the embedding
    52	feature output by the PCR. The superscript head, trunk, leg represents the embedding feature for
    53	different areas of the person, while L denotes input data is local. Subsequently, we incorporate the
    54	global data X together with the local data X L , which is followed by the addition of a class token
    55	and positional embeddings to obtain the input for the subsequent LAReViT network.
    56	The architecture of LAReViT is illustrated in Figure 3. For global person data X, we utilized the
    57	original transformer encoder to learn the global representation of pedestrians. For local person data
    58	X L , the LAR Module is proposed to learn person local feature. Regarding the LAR, we construct
    59	innovative attention residual connections between its consecutive Transformer Block layers, which
    60	forms the core of the LRA mechanism. The LRA facilitates the propagation and accumulation
    61	of attention to cross-modal person feature from shallow to deep layers. As shown on the right
    62	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
    63	
    64	146:10
    65	
    66	X. Hua et al.
    67	
    68	side of Figure 3, this expansion red arrow, which represents the proposed novel attention residual
    69	connection, complements the skip connections that are already inside each Transformer Block.
    70	In fact, while existing skip connections already propagate shallow-layer low-level features
    71	to deeper layers internal the Transformer Block, the proposed LRA serves as a complementary
    72	process. The LRA with novel Residual Attention (RA) connection further propagates query
    73	information from adjacent Transformer Blocks. It defines the relationships between patches utilized
    74	for feature extraction, thereby enabling a new attention capability to consider previously extracted
    75	relationships while learning to extract new ones. We apply this concept to the input of local
    76	cross-modal person information within the network, forming a Local-aware Attention Residual
    77	Connection. Formally, the proposed LRA is achieved by modifying the calculation of the query
    78	matrix defined in the equation. Specifically, the equation combines the Q matrix of the current
    79	Transformer Block aggregated with the input X L of the block before the previous block, as follows:
    80	𝑄𝑖 , 𝐾𝑖 , 𝑉𝑖 = 𝑊𝑖 X𝑖L , 𝑊𝑖𝐾 X𝑖L , 𝑊𝑖𝑉 X𝑖L ,
    81	(
    82	𝑄𝑖
    83	𝑖 =0
    84	𝑄𝑖 = 𝑄𝑖 +X𝑖L−1
    85	,
    86	𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
    87	2
    88	!
    89	𝑄𝑖 · 𝐾𝑖𝑇
    90	L
    91	𝐿𝑅𝐴𝑖 (X𝑖 ) = 𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥
    92	· 𝑉𝑖 ,
    93	√
    94	𝑑𝑖
    95	
    96	(7)
    97	
    98	L
    99	X𝑖+1
   100	= 𝐿𝑅𝐴𝑖 (X𝑖L ) + X𝑖L ,
   101	
   102	(10)
   103	
   104	𝑄
   105	
   106	(8)
   107	
   108	(9)
   109	
   110	where i indicates the ith LAR and W is the weight matrix of query, key, and value to attention
   111	mechanism. The information flow among adjacent LRA is extended beyond the standard flow and
   112	propagates the attention used to aggregate information between patches in previous transformer
   113	blocks. However, propagating and accumulating such information may prevent the network from
   114	learning high-level representations while amplifying the attention related to low-level features. To
   115	avoid this, we balance the quantity of RA to propagate between adjacent LRA. The query matrix Q is
   116	implemented as a combination of the current layer and previous layer information. So, considering
   117	the implementation of LRA, it can be easily integrated into existing VI-ReID transformer models
   118	while maintaining similar computational costs. Finally, the proposed LAR Module, which is based
   119	on the LRA, effectively addresses the issues of feature excessive globalization and feature collapse
   120	in Transformers.
   121	3.3
   122	
   123	CAL
   124	
   125	Due to the modality difference between RGB and IR images, there are discrepancies in the representation of features belonging to the same individual in the feature space. Therefore, for VI-ReID,
   126	our goal is to compel the model to prioritize the consideration of modality-shared features, to
   127	mitigate the challenge posed by the modality gap in person matching. However, existing center
   128	loss methods directly aggregate all cross-modal features of the same identity and constrain the
   129	inter-class distance, which leads to overly strong constraints on the learning of modality-shared
   130	features. This constraint can enable the network to learn person identification features rapidly,
   131	but may result in insufficient learning of modality-shared information, which poses challenges
   132	in utilizing abundant identity information thoroughly. Therefore, we propose the CAL, which
   133	promotes modal alignment and learning of comprehensive features by way of moderately reducing
   134	the modality gap.
   135	Specifically, for the global and local output dataflow X and X L from the LAReViT, which share
   136	one classification token. Then, we utilize the shared classification token as the person feature
   137	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   138	
   139	Local-Aware Residual Attention Vision Transformer for VI-ReID
   140	
   141	146:11
   142	
   143	extracted by LAReViT, which we formulate as 𝑓𝑝 . For the embedding feature 𝑓𝑝 of LAReViT, we
   144	categorize it into features 𝑓 𝑉 and 𝑓 𝑇 under the visible and IR modalities, respectively. Then, we
   145	calculate the centers 𝑐𝑉 and 𝑐𝑇 of person features for different modalities by summation and
   146	averaging:
   147	𝐾
   148	
   149	𝑐𝑖V =
   150	
   151	1 Õ V
   152	𝑓 ,
   153	𝐾 𝑝=1 𝑝
   154	
   155	𝑐𝑖T =
   156	
   157	1 Õ T
   158	𝑓 ,
   159	𝐾 𝑝=1 𝑝
   160	
   161	(11)
   162	
   163	𝐾
   164	
   165	(12)
   166	
   167	where V and T represent the features belonging to the visible and IR modalities, respectively. K
   168	represents the number of samples for the same ID. Finally, we obtain the sample center c for each
   169	identity, where i denotes the ID of the feature center.
   170	As shown on the left side of Figure 3, CAL aims to mitigate modal discrepancy and learn more
   171	comprehensive cross-modal features by minimizing the distance across modality while moderately
   172	pushing the distance between negative samples. Furthermore, to further constrain the network
   173	optimization and learning smoothly and stably, we incorporate the inter-modal constraint term with
   174	the negative sample constraint as the denominator, which appropriately alleviates the constraints
   175	between negative samples. This moderate constraint can guide the LAReViT to learn cross-modal
   176	features more thoroughly. Specifically, CAL can be formulated as follows:
   177	D𝑖𝑚𝑜𝑑𝑎𝑙 = 𝑐𝑖V − 𝑐𝑖T 2 ,
   178	
   179	(13)
   180	
   181	D𝑖,𝑗 = 𝑓 𝑗 − 𝑐𝑖V 2 + 𝑓 𝑗 − 𝑐𝑖T 2,
   182	
   183	(14)
   184	
   185	𝑛𝑒𝑔
   186	
   187	Í𝑁
   188	
   189	𝐿𝐶𝐴𝐿 =
   190	
   191	1
   192	2(𝑁 −1)
   193	
   194	𝑚𝑜𝑑𝑎𝑙
   195	𝑖=1 D𝑖
   196	,
   197	Í𝑁
   198	𝑛𝑒𝑔
   199	𝑚𝑜𝑑𝑎𝑙
   200	𝑖,𝑗=1 D𝑖,𝑗 + 𝑖=1 D𝑖
   201	
   202	Í𝑁
   203	
   204	(15)
   205	
   206	∀𝑖≠𝑗
   207	
   208	in which D is defined as the distances between the cross-modal features. The L2 norm, denoted by
   209	k · k, represents the Euclidean distance. The subscripts i and j represent the identity of the feature
   210	or the feature center of a person. N represents the total number of person identities. We impose
   211	the constraints between each person feature 𝑓 𝑗 and feature centers 𝑐𝑉𝑖 , 𝑐𝑇𝑖 within each identity i
   212	and j.
   213	CAL facilitates the integration of different modalities and bridges the modality gap. It moderately
   214	constrains the distance between cross-modal features, thereby promoting better cross-modal
   215	alignment in LAReViT and extracting comprehensive person features.
   216	3.4
   217	
   218	Objective Function Optimization
   219	
   220	Finally, we incorporate commonly used ID loss, triplet loss, and the designed CAL to enhance
   221	the ability of LAReViT to extract discriminative features. The overall loss function is defined as
   222	follows:
   223	𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑖𝑑 + 𝐿𝑡𝑟𝑖 + 𝜆𝐿𝐶𝐴𝐿 ,
   224	
   225	(16)
   226	
   227	where the parameter 𝜆 is set to 0.5 in order to balance the terms CAL. Moreover, Algorithm 1
   228	depicts the training procedure of our proposed methodology.
   229	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   230	
   231	146:12
   232	
   233	X. Hua et al.
   234	
   235	Algorithm 1: The Training Process of LAReViT
   236	
   237	4
   238	4.1
   239	
   240	Experimental Result
   241	Datasets and Evaluation Metrics
   242	
   243	The SYSU-MM01 dataset [50] furnishes a collection of 287,628 visible images and 15,792 IR images
   244	captured by four visible cameras and two IR cameras. This dataset comprises 491 distinct identities
   245	and encompasses both All-search and Indoor-search modes. We utilize a query set consisting of
   246	3,803 IR images, and we randomly select 301 images from the remaining visible images to form the
   247	gallery set.
   248	The RegDB dataset [37] comprises a collection of 4,120 paired visible and IR images that were
   249	captured using overlapping cameras. This dataset encompasses 412 unique identities, with each
   250	identity associated with 10 visible images and 10 IR images. During the training phase, we randomly
   251	select all the images from 206 identities, while the remaining 206 identities are exclusively used for
   252	testing purposes.
   253	The LLCM dataset [69] employs a 9-camera network deployed in low-light environments, allowing
   254	for the capture of visible (VIS) images during the daytime and IR images at night. This dataset
   255	comprises 46,767 bounding boxes belonging to 1,064 distinct identities, covering a wide range of
   256	climate conditions and clothing styles.
   257	We utilize Cumulative Matching Characteristics, Mean Average Precision (mAP), and
   258	Mean Inverse Negative Penalty (mINP) [61] as our primary evaluation metrics.
   259	4.2
   260	
   261	Implementation Details
   262	
   263	Our model is implemented using the PyTorch library and trained on an NVIDIA 3090 GPU. Initially,
   264	we adopted ViT-B/16 as our baseline architecture. For data augmentation, each image is resized
   265	to 288 × 144 with Zero-Padding, and we apply random horizontal flipping, random erasing [74],
   266	and random grayscale. To address lighting and resolution variations, we also apply Gaussian blur
   267	and color jittering techniques to the IR images. During the training phase, we randomly select four
   268	visible (VIS) images and four IR images from six identities. The model optimization is performed
   269	using the AdamW optimizer with cosine learning rate. The learning rate is set to 3e-4, and the
   270	weight decay is set to 1e-4. We train the model for 32 epochs on 3 VI-ReID datasets. Additionally,
   271	we set the parameters 𝜆 = 0.5, 𝛼 = 0.3 and resolution of each image patch (𝑃, 𝑃) is 16 × 16.
   272	4.3
   273	
   274	Comparison with the State-of-the-Arts
   275	
   276	4.3.1 Results on SYSU-MM01 Datasets. Based on the experimental results presented in Table 1,
   277	it can be observed that the proposed LAReViT method surpasses other state-of-the-art approaches
   278	on the SYSU-MM01 dataset. LAReViT achieves 76.71% Rank-1 and 72.95% mAP in the All-search
   279	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   280	
   281	Local-Aware Residual Attention Vision Transformer for VI-ReID
   282	
   283	146:13
   284	
   285	Table 1. Comparison with the State-of-the-Arts on SYSU-MM01 Dataset
   286	Method
   287	
   288	Publish
   289	
   290	All search
   291	Indoor search
   292	Rank-1 Rank-10 Rank-20 mAP Rank-1 Rank-10 Rank-20 mAP
   293	
   294	Zero-Pad [50]
   295	HCML [58]
   296	AliGAN [48]
   297	DDAG [60]
   298	AGW [61]
   299	HAT [62]
   300	
   301	ICCV 17
   302	AAAI 18
   303	ICCV 19
   304	ECCV 20
   305	TPAMI 21
   306	TIFS 21
   307	
   308	14.80
   309	14.32
   310	42.40
   311	54.75
   312	47.50
   313	55.29
   314	
   315	54.12
   316	53.16
   317	85.00
   318	90.39
   319	84.39
   320	92.14
   321	
   322	71.33
   323	69.17
   324	93.70
   325	95.81
   326	92.14
   327	97.36
   328	
   329	15.95
   330	16.16
   331	40.70
   332	53.02
   333	47.65
   334	53.89
   335	
   336	20.58
   337	24.52
   338	45.90
   339	61.02
   340	54.17

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 13 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '340,680p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   340	54.17
   341	62.10
   342	
   343	68.38
   344	73.25
   345	87.60
   346	94.06
   347	91.14
   348	95.75
   349	
   350	85.79
   351	86.73
   352	94.40
   353	98.41
   354	95.98
   355	99.20
   356	
   357	26.92
   358	30.08
   359	54.30
   360	67.98
   361	62.97
   362	69.37
   363	
   364	SPOT [2]
   365	FMCNet [67]
   366	DCLNet [44]
   367	MAUM [33]
   368	DSCNet [68]
   369	
   370	TIP 22
   371	CVPR 22
   372	ACM MM 22
   373	CVPR 22
   374	TIFS 22
   375	
   376	65.34
   377	66.34
   378	70.80
   379	71.68
   380	73.89
   381	
   382	92.73
   383	96.27
   384	
   385	97.04
   386	98.84
   387	
   388	62.25
   389	62.51
   390	65.30
   391	68.79
   392	69.47
   393	
   394	69.42
   395	68.15
   396	73.50
   397	76.97
   398	79.35
   399	
   400	96.22
   401	98.32
   402	
   403	99.12
   404	99.77
   405	
   406	74.63 Transformer
   407	74.09
   408	CNN
   409	76.80
   410	CNN
   411	81.94
   412	CNN
   413	82.65
   414	CNN
   415	
   416	SFANet [31]
   417	PMT [34]
   418	DGRNet [26]
   419	DEEN [69]
   420	
   421	TNNLS 23
   422	AAAI 23
   423	TOMM 23
   424	CVPR 23
   425	
   426	65.74
   427	67.53
   428	71.53
   429	74.70
   430	
   431	92.98
   432	95.36
   433	96.06
   434	97.60
   435	
   436	97.05
   437	98.64
   438	98.62
   439	99.20
   440	
   441	60.83
   442	64.98
   443	68.04
   444	71.80
   445	
   446	71.60
   447	71.66
   448	77.49
   449	80.30
   450	
   451	96.60
   452	96.73
   453	98.61
   454	99.00
   455	
   456	99.45
   457	99.25
   458	99.79
   459	99.80
   460	
   461	80.05
   462	CNN
   463	76.52 Transformer
   464	81.51
   465	CNN
   466	83.30
   467	CNN
   468	
   469	SDCL [54]
   470	MMM [43]
   471	RPNR [63]
   472	MPMN [56]
   473	MIAM [4]
   474	TRR [73]
   475	
   476	CVPR 24
   477	ECCV 24
   478	ACM MM 24
   479	TMM 24
   480	JVCIR 24
   481	Inform Fusion 24
   482	
   483	64.49
   484	65.90
   485	65.20
   486	70.59
   487	74.34
   488	74.44
   489	
   490	96.22
   491	97.76
   492	-
   493	
   494	98.77
   495	99.29
   496	-
   497	
   498	63.24
   499	61.80
   500	60.00
   501	67.49
   502	71.84
   503	70.56
   504	
   505	71.37
   506	70.30
   507	68.90
   508	75.98
   509	82.93
   510	81.32
   511	
   512	98.12
   513	98.82
   514	-
   515	
   516	99.62
   517	99.95
   518	-
   519	
   520	76.90
   521	CNN
   522	74.90
   523	CNN
   524	74.40
   525	CNN
   526	80.24
   527	CNN
   528	84.80
   529	CNN
   530	83.22 Transformer
   531	
   532	LAReViT
   533	
   534	Ours
   535	
   536	76.71
   537	
   538	97.33
   539	
   540	99.05
   541	
   542	72.95 84.22
   543	
   544	99.02
   545	
   546	99.85
   547	
   548	86.26 Transformer
   549	
   550	Backbone
   551	CNN
   552	CNN
   553	CNN
   554	CNN
   555	CNN
   556	CNN
   557	
   558	Rank-k accuracy (%) and mAP (%) are reported. The highest value of each evaluation metrics is underlined.
   559	
   560	mode. In addition, in the Indoor-search mode, LAReViT achieves 84.22% Rank-1 and 86.26% mAP.
   561	By leveraging the strengths of both convolution kernel and transformer architectures, LAReViT
   562	explores both local and global features to extract more comprehensive representations. In the
   563	All-search mode, our method demonstrates superiority over CNN-based approaches [26, 44, 56, 69]
   564	in terms of Rank-n accuracy and mAP. Our method outperforms DGRNet [26] that utilizes dynamic
   565	weighted gradient reversal by 5.18% in Rank-1 accuracy and 4.91% in mAP. Additionally, compared
   566	to the MPMN [56] method that utilized modality shared-specific features cooperative separation,
   567	the proposed LAReViT achieves a performance gain of 6.12% in Rank-1 accuracy and 5.46% in mAP.
   568	The advantage of LAReViT in extracting global features contributes to its superior performance
   569	compared to CNN-based methods. Moreover, compared to transformer-based methods such as
   570	SPOT [2], PMT [34], and TRR [73], our approach exhibits superior performance across various
   571	evaluation metrics. Specifically, our method outperforms the state-of-the-art TRR by 2.27% in
   572	Rank-1 accuracy and 2.39% in mAP. While TRR considered the correlation between modalities
   573	during the feature fusion stage, the aforementioned transformer-based methods lack exploration of
   574	fine-grained features. Therefore, our method achieves better performance by incorporating shallow
   575	local information and leveraging the advantages of convolution receptive fields. Additionally, our
   576	model outperforms DDAG [60], a CNN-based approach that combines transformer components for
   577	feature extraction. In the Indoor-search mode, our method also demonstrates superior performance
   578	in terms of Rank-n accuracy and mAP. These analyses evidence the effectiveness of our proposed
   579	approach in extracting more comprehensive modality-shared features.
   580	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   581	
   582	
codex
第五篇方法读完：作者的叙事不是“再做局部特征”，而是抓住 ViT 在 VI-ReID 中越深越全局化、低层局部线索被淹没这个观察，用不加 CNN 的纯 ViT 改造来卖。现在读第六篇遮挡 ReID 的 mask-aware hierarchical aggregation。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '520,840p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '210,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | nl -ba | rg -n \"Overall|Framework|Method|Proposed|Mask|Aggregation|Transformer|Experiment|Ablation|Loss|Hierarchical|Attention|Block|Module|Training|Approach\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:     5	Mask-Aware Hierarchical Aggregation Transformer
14:    14	local features. To achieve these we propose a Mask-Aware
15:    15	Hierarchical Aggregation TrAnsforMer (MAHATMA) method
17:    17	Mask Information Embedding (MIE) module, which directs the
21:    21	a Hierarchical Feature Aggregation (HFA) module that mines
101:   101	call Mask-Aware Hierarchical Aggregation TrAnsforMer
103:   103	extractor, a Mask Information Embeddings (MIE) module,
104:   104	a Hierarchical Feature Aggregation (HFA) module, and a
140:   140	Mask-Aware Hierarchical Aggregation TrAnsforMer
155:   155	A. Visual Transformer
156:   156	Transformer [16] is a revolutionary neural network architecture, initially introduced in the study of Natural Language
160:   160	transformer has recently been transplanted into diverse computer vision tasks. Vision Transformer has demonstrated
162:   162	sequential data. For example, Vision Transformer (ViT) [17]
165:   165	that serve as tokens similar to words in NLP. Swin Transformer [18] designed a hierarchical transformer with shifting
195:   195	Xu et al. [11] proposed a Feature Recovery Transformer
209:   209	Zhang et al. [32] introduced a Hierarchical Aggregation
210:   210	Transformers (HAT) framework, which proposed a feature calibration module based on transformer to integrate multi-scale
212:   212	information. Fan et al. [9] proposed a novel Skip Connection Aggregated Transformer (SCAT) network, incorporating
215:   215	Aggregated Transformer (MFAT) network, which constructs
230:   230	A. Overall Framework
297:   297	B. Mask Information Embeddings Module
333:   333	Fig. 2. Our proposed framework for occluded person Re-ID. Mask-Aware Hierarchical Aggregation TrAnsforMer (MAHATMA) consists of a mask extractor,
371:   371	C. Hierarchical Feature Aggregation Module
445:   445	D. Diverse Feature Completion Module
502:   502	E. Loss Function and Optimization
646:   646	C. Comparison With the State-of-the-Art Methods
753:   753	D. Ablation Study
822:   822	2) Effectiveness of Mask Extractor Architecture: To verify

 succeeded in 0ms:
   210	Transformers (HAT) framework, which proposed a feature calibration module based on transformer to integrate multi-scale
   211	features by leveraging global perspectives and enhancing local
   212	information. Fan et al. [9] proposed a novel Skip Connection Aggregated Transformer (SCAT) network, incorporating
   213	elements from different layers to compose pedestrian feature
   214	representations. Tan et al. [33] designed a Multi-level Feature
   215	Aggregated Transformer (MFAT) network, which constructs
   216	an aggregation framework from global and local perspectives
   217	to obtain more comprehensive and discriminative attention
   218	regions. However, these multi-scale feature aggregation methods do not effectively capture significant variations in pattern
   219	information among pixels within a patch and cannot effectively
   220	model similar pattern information between pixels in different
   221	patches when processing patch feature representations in the
   222	transformer, resulting in the loss of many fine-grained features
   223	and context information.
   224	In contrast to the methods mentioned above, our proposed MAHATMA can not only realizes the perception of
   225	non-occluded body parts in a variety of occlusion scenarios,
   226	but also has the ability to capture more fine-grained information. This advantage improves the feature robustness of the
   227	model under occlusion conditions and significantly enhances
   228	the matching accuracy.
   229	III. P ROPOSED M ETHOD
   230	A. Overall Framework
   231	Fig 2 presents the framework of our approach, which
   232	employs a pre-trained ViT [17] as the feature extractor to capture feature embeddings from input images. Let x ∈ R H ×W ×C
   233	be an image input, with H, W, C representing its height, width,
   234	and channel dimensions. Initially,
   235	n we partition the oinput image
   236	x into N fixed-size patches x ip |i = 1, 2, . . . , N through a
   237	sliding window approach. The stride and dimension of each
   238	image patch are denoted as S and P. The resultant count of
   239	generated patches N can be depicted as follows:
   240	 
   241	
   242	
   243	W +S−P
   244	H+S−P
   245	×
   246	,
   247	(1)
   248	N=
   249	S
   250	S
   251	where ⌊.⌋ is the floor function. Since the input to the
   252	transformer encoder must be in the form of a sequence,
   253	a trainable linear transformation function f (.) is applied to
   254	flatten patches, mapping them to D dimensions after which
   255	patch embeddings E p ∈ R N ×D are obtained. A learnable
   256	classification token E g is added before patch embeddings,
   257	with output classification token serving as the global feature
   258	representations for encoder. To retain the positional information, we use learnable position encodings PE . Considering
   259	that feature representations are highly sensitive to camera
   260	variations, we adopt the approach proposed in [7] for acquiring
   261	camera viewpoint information Cid . To solve the problem that
   262	MSA in transformer is susceptible to occlusion interference,
   263	we design a mask information embeddings (MIE) module
   264	to obtain the foreground semantic embeddings E m through
   265	encoding the mask semantic information of body parts, and
   266	then integrate E m and E p to direct the model’s attention
   267	
   268	5823
   269	
   270	towards non-occluded body parts of the person. In the end,
   271	the sequence forwarded as input to the transformer encoder is
   272	represented as:
   273	E input = {E g ; E p ; E m } + PE + λcm Cid ,
   274	
   275	(2)
   276	
   277	where PE represents position embeddings, Cid ∈ R (N +1)×D
   278	represents the camera embeddings and remains unchanged for
   279	the same image, λcm is a hyper-parameter used to adjust
   280	the weights of the camera embeddings. Then, the input
   281	embedding E input undergoes processing by L transformer
   282	layers to generate the ultimate feature outputs. To better deal
   283	with occlusion scenes, inspired by [34], the representation
   284	for image patches at the 2nd , 4th , 10th , and 12th stages are
   285	concatenated and sent to the hierarchical feature aggregation
   286	(HFA) module to capture fine-grained local features. To further
   287	strengthen the robustness of feature learning with transformer
   288	framework and improve the model’s global perception ability,
   289	we design a diverse feature completion (DFC) module, which
   290	makes full use of dilated convolutional networks to obtain
   291	discriminating global features by learning visual cues from
   292	different receptive fields. The current methods [9], [12] of
   293	directly using global features for identity recognition often
   294	fail to produce satisfactory results, and so we combine the
   295	above-mentioned global features with local features for person
   296	matching to achieve more precise classification outcomes.
   297	B. Mask Information Embeddings Module
   298	It is known that precise focusing on non-occluded body
   299	parts is a necessary step for occluded person Re-ID. Despite
   300	the remarkable performance achieved by transformer-based
   301	strong baseline in occluded person Re-ID, as indicated by
   302	the visualization of attention maps of TransReID in Fig 1(b),
   303	the MSA mechanism may be susceptible to interference
   304	from background information and occlusion noise, making it
   305	challenging to precisely focus on non-occluded body parts.
   306	Because of this problem, we design a MIE module that incorporates semantic information of body parts into the embedding
   307	representation to mitigate the impact of occlusion interference
   308	on the precise positioning of the model.
   309	Inspired by position and side information embeddings,
   310	which encode both positional and camera information using
   311	learnable embeddings, we introduce the foreground semantic
   312	embeddings to retain the semantic information of body parts.
   313	This design constitutes the core principle of the MIE module,
   314	which significantly mitigates the negative impact of occlusion on feature extraction by integrating foreground semantic
   315	information. Specifically, we first generate the semantic mask
   316	of pedestrians through a mask extractor. Considering the
   317	excellent performance of HRNet [35] in human semantic
   318	parsing, we use it as mask extractor backbone network to
   319	extract pedestrian features, and these features are processed
   320	through a 1 × 1 convolutional layer followed by a softmax
   321	to generate human semantic mask m. In addition, we use the
   322	parsing labels generated by the human body parsing model
   323	PifPaf [36] to supervise m to further optimize its quality.
   324	Subsequently, we follow the methods provided in [37] to learn
   325	five categories of semantic information for body parts and one
   326	
   327	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   328	
   329	5824
   330	
   331	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   332	
   333	Fig. 2. Our proposed framework for occluded person Re-ID. Mask-Aware Hierarchical Aggregation TrAnsforMer (MAHATMA) consists of a mask extractor,
   334	a mask information embeddings (MIE) module, a hierarchical feature aggregation (HFA) module and a diverse feature completion (DFC) module. After
   335	obtaining mask semantic information for body parts using the mask extractor, the MIE module encodes mask semantic information of body parts into
   336	embedding representations, which input along with patch embeddings, positional embeddings, and camera-related information into the transformer encoder,
   337	directing the model’s attention towards non-occluded body parts. Then the HFA module and DFC module are used to obtain both fine-grained and robust
   338	feature representations. The former can mine more exploitable high-quality detail information by aggregating hierarchical image patch representations to
   339	capture fine-grained local features. The latter obtains robust global feature representations by integrating multi-path features.
   340	
   341	category of semantic information for background. To solve the
   342	occlusion problem, only the mask of the human body, which
   343	we call the foreground mask is used, while the background
   344	information may be disregarded. Similar to the processing
   345	of input image x,n the foreground mask
   346	o is partitioned into
   347	i
   348	N mask patches m p |i = 1, 2, . . . , N . Subsequently, each
   349	mask patch is flattened and linearly projected, resulting in
   350	foreground semantic embeddings E m ∈ R N ×D . As shown in
   351	Eq.(2), the foreground semantic embeddings E m will be fused
   352	with patch embeddings E p , which can effectively mitigate
   353	the interference of occlusion and direct the model’s attention
   354	towards non-occluded body parts.
   355	Here, we only demonstrate the use of mask patches to obtain
   356	foreground semantic embeddings, and then fuse the foreground
   357	semantic embeddings with the patch embeddings to lessen the
   358	impact of occlusion noise in locating non-occluded body parts.
   359	In practice, mask patches can also be used to guide visible
   360	features in updating occluded features. In addition, mask
   361	patches are used to represent semantic categories to allow for
   362	the extraction of effective local features by rearranging token
   363	features belonging to similar semantic characteristics. It is
   364	worth noting that the problem of video person Re-ID under
   365	occlusion conditions can be further explored by combining
   366	continuous semantic segmentation mask blocks to provide
   367	more fine-grained classification results. Related research [38]
   368	has demonstrated that this approach is more effective in
   369	dealing with real-world challenges.
   370	
   371	C. Hierarchical Feature Aggregation Module
   372	Through the MIE module, we direct the model to concentrate on the visible body parts of the individual, so mitigating
   373	interference from different types of occlusions. However,
   374	transformer-based baselines also face the difficulty of capturing fine-grained information. In other words, because of the
   375	absence of fine-grained features, pre-trained models can fail to
   376	distinguish different people with similar appearances. Therefore, designing a model that can capture fine-grained discriminative features and distinguish among challenging samples is
   377	crucial. This principle is further demonstrated in [39].
   378	In this paper, we design a HFA module which mines
   379	more exploitable high-quality detail information by aggregating hierarchical image patch representations to capture
   380	fine-grained local features. The core of the HFA module is
   381	to efficiently model the pattern information between different
   382	patches, and achieve the capture of fine-grained features of the
   383	target person by searching the feature representation that best
   384	represents the identity of the person. In most scenarios, the
   385	pattern information of an image is conveyed by pixels that are
   386	densely distributed in space and form into connected components. Similarly, for occluded person images, the target person
   387	will be identified from pixels that are densely distributed
   388	in space and form into connected components away from
   389	the occluded part. In a convolution neural network, feature
   390	representation of each channel is viewed as a reaction to particular pattern information. Based on these prior knowledge,
   391	
   392	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   393	
   394	ZHANG et al.: MAHATMA FOR OCCLUDED PERSON RE-IDENTIFICATION
   395	
   396	5825
   397	
   398	we recognize that using a sliding window approach based on
   399	local regions, to measure the key pattern information is an
   400	important way to extract fine-grained information. Therefore,
   401	in order to provide high-quality local features, we apply a
   402	convolutional-based HFA module to aggregate this diverse
   403	information. Additionally, we incorporate a global max pooling (GMP) layer to ensure translation and magnitude stability
   404	of body part features.
   405	As the network depth increases, feature representations of
   406	each patch will become smoother and increasingly similar. It is
   407	difficult for HFA to effectively model diverse detailed information in the presence of high similarity of inputs. Therefore,
   408	adopting a hierarchical structure to integrate shallow-layer
   409	image patch representations is a reasonable solution. Specifically, after the l th transformer block, we follow the method
   410	in [18] to aggregate reshaped image patch representations after
   411	the 2nd , 4th , 10th and 12th transformer block as the input of
   412	HFA. The final feature generation can be described as follows:
   413	f p = Max pool(C(G ∗ ( f 1 , f 2 , . . . , f L ))),
   414	
   415	(3)
   416	
   417	where fl = r eshape[x lp1 , x lp2 , . . . , x lpN ] with l ∈ {1, 2, . . . , L},
   418	N represents the number of patches, C represents the convolutional layers of the HFA module, and G is the binary gate
   419	that selects the input for HFA.
   420	To bolster the discriminative power of the HFA-generated
   421	features, we introduce a local contrastive loss whose core idea
   422	is to learn a representation that can distinguish between positive pairs (instances from the same class) and negative pairs
   423	(instances from different classes) and maximize the similarity
   424	between positive pairs. To facilitate the implementation of the
   425	local contrastive loss, we employ the following “distance” in
   426	this paper.
   427	N
   428	
   429	p
   430	
   431	
   432	1 X
   433	log 1 + exp(|| f p − f t ||22 − || f p − f n ||22 ) ,
   434	L lc =
   435	Np
   436	
   437	i=1
   438	
   439	(4)
   440	where N p denotes the number of positive instance pairs from
   441	the same category, f p denotes the generated local features,
   442	f t represents the characteristics of the target instance, which
   443	corresponds to the positive instance, while f n represents the
   444	negative characteristics of the instance.
   445	D. Diverse Feature Completion Module
   446	Combining global features with local features significantly
   447	enriches feature representation. However, for occluded person
   448	Re-ID, because of the spatial discontinuity caused by partial
   449	occlusion, relying solely on the classification token from the
   450	last transformer layer as the global representation of the
   451	entire input sequence does not adequately capture detailed
   452	information of the occluded person. This often leads to the loss
   453	of key features, thus affecting the accuracy of Re-ID. Therefore, to obtain discriminative global features, a DFC module
   454	is designed to conform multi-path features to compensate
   455	global features through a multi-branch parallel convolutional
   456	network. Specifically, for each branch of the DFC module,
   457	1 , ϕ2 ,
   458	we first use three 3×3 dilated convolutional layers ϕ3×3
   459	3×3
   460	
   461	3
   462	and ϕ3×3
   463	with different dilation rates (1, 2, 3) to sample the
   464	input in parallel and concatenate the results together, followed
   465	by another 1 × 1 convolution layer θ1×1 to make sure that
   466	the result is consistent with the size of the input. Finally, the
   467	feature mappings of each branch are combined to produce the
   468	ultimate feature representations. The feature mappings of each
   469	branch are represented as:
   470	1
   471	2
   472	3
   473	f gi = θ1×1 (Concat[ϕ3×3
   474	(zl ), ϕ3×3
   475	(zl ), ϕ3×3
   476	(zl )]),
   477	
   478	(5)
   479	
   480	where zl = r eshape(x gl ). Furthermore, to ensure that the
   481	feature embeddings generated by our designed convolution
   482	branches can capture more discriminative features, we incorporate the cosine similarity loss to constrain it. Our objective
   483	is to make the feature vectors output from the model closer to
   484	the target feature vectors in terms of cosine similarity, thereby
   485	improving the model performance. Through the ablation experiments in the fourth section, we find that the introduction
   486	of this loss function has shown a significant performance
   487	enhancement for person Re-ID problems, particularly when
   488	considering occluded person Re-ID tasks. The cosine similarity loss function takes the following form:
   489	L cs = 1 −
   490	
   491	< f g , ft >
   492	,
   493	f g · ∥ ft ∥
   494	
   495	(6)
   496	
   497	where f g represents the feature representation output by the
   498	DFC module, f t denotes the target feature vector, which is
   499	calculated by utilizing the average features extracted from each
   500	branch. <, > is the usual dot product operation on vectors and
   501	∥·∥ is the L 2 -norm.
   502	E. Loss Function and Optimization
   503	In our method, we design corresponding loss functions
   504	for local features and global features respectively, and use
   505	triplet loss and identity loss to constrain the global feature
   506	representations. Combined with the cosine similarity loss, the
   507	loss function for global features can be articulated as follows:
   508	L G = L tri + L cls + λL cs .
   509	
   510	(7)
   511	
   512	As an example, when extracting local features, pedestrians
   513	with different identities may have similar occlusions and this
   514	causes identity information to be weakened and inter-class
   515	differences to be blurred. It is challenging for the widely
   516	used softmax loss to achieve sufficient intra-class compactness
   517	under these conditions. Following [8], we incorporate an
   518	additional angular margin into the premier softmax loss to
   519	emphasize differences between local features to better capture
   520	information in the local region. The L p−cls with an additional

 succeeded in 0ms:
   520	information in the local region. The L p−cls with an additional
   521	angular margin can be represented as:
   522	s(WT f +m)
   523	
   524	B
   525	
   526	L p−cls = −
   527	
   528	e yi p
   529	1 X
   530	log
   531	,
   532	T
   533	s(W f +m))
   534	B
   535	e yi p
   536	+ Dinter
   537	
   538	(8)
   539	
   540	i=1
   541	
   542	where Dinter =
   543	
   544	C
   545	P
   546	
   547	e
   548	
   549	s(WyT f p )
   550	j
   551	
   552	, B and C are the batch
   553	
   554	j=1, j̸ = y i
   555	
   556	size and the number of classes, respectively. m represents
   557	the margin in angular distance and s is a hyperparameter for
   558	
   559	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   560	
   561	5826
   562	
   563	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   564	
   565	scaling. Combining this with the local contrastive loss, the loss
   566	function for local features can be articulated as follows:
   567	L P = L p−cls + γ L lc .
   568	
   569	(9)
   570	
   571	The comprehensive target loss function can be articulated as:
   572	
   573	TABLE I
   574	B ELOW A RE THE C OMPREHENSIVE D ETAILS R EGARDING THE DATASETS .
   575	W E A SSESS O UR A PPROACH U SING F IVE P UBLICLY AVAILABLE
   576	DATASETS , I NCLUDING T WO O CCLUSION DATASETS , O CCLUDED D UKE AND O CCLUDED -REID, A LONG W ITH T HREE H OLIS TIC DATASETS , M ARKET-1501, D UKE MTMC- RE ID AND
   577	MSMT17
   578	
   579	L = L G + L P = L tri + L cls + λL cs + L p−cls + γ L lc . (10)
   580	where λ and γ represent hyperparameters used to adjust the
   581	importance of L cs and L lc respectively.
   582	IV. E XPERIMENTS
   583	A. Datasets and Evaluation Metrics
   584	To illustrate the efficiency of our approach, adequate experiments are performed on five commonly shared datasets,
   585	encompassing two occluded person Re-ID datasets and three
   586	holistic person Re-ID datasets. The datasets detail information
   587	is summarized in Table I.
   588	Occluded-Duke [40] is an extensive dataset gathered
   589	specifically for occluded person Re-ID by transforming
   590	DukeMTMC. This training set comprises 15,618 images
   591	depicting 702 individuals. The test set comprises 2,210 query
   592	images representing 519 individuals and 17,661 gallery images
   593	depicting 1,110 individuals. Because of its extensive scale,
   594	Occluded-Duke continues to be regarded as a prominent
   595	datasets for occluded person Re-ID tasks.
   596	Occluded-REID [21] encompasses images of individuals
   597	in occluded scenarios taken by moving cameras. It contains
   598	2,000 images originating from 200 individuals, each individual
   599	is represented by 5 occluded person images and 5 full-body
   600	images. The framework employed in conducting experiments
   601	on this dataset undergoes training with the Marker-1501 [41]
   602	training dataset.
   603	Market-1501 [41] stands as an extensively utilized holistic
   604	person Re-ID dataset taken by 6 cameras. It comprises 12,936
   605	training images belonging to 751 individuals, along with
   606	3,368 query images from 750 individuals, and 19,732 gallery
   607	images representing 750 individuals. Non-occluded images
   608	predominantly constitute the dataset.
   609	DukeMTMC-reID [42] is comprised of 36,411 images
   610	depicting 1,404 individuals, which is taken by 8 cameras.
   611	A subset of 16,522 images representing 702 individuals are
   612	selected at random as the training set from the dataset, while
   613	the rest of the images are distributed into 2,228 query images
   614	and 17,661 gallery images for testing purposes, ensuring that
   615	gallery images do not overlap with the training set.
   616	MSMT17 [43] comprises 125,441 images from 4,101 identities gathered from 15 camera perspectives. This incorporates
   617	32,621 training images. Throughout the inference procedure,
   618	11,659 images are randomly selected as queries from a pool
   619	of 93,820 images, while the remainder of the images are
   620	designated as gallery images.
   621	Evaluation metrics: For consistency in evaluation, we also
   622	measure the effectiveness of our proposed method using
   623	widely recognized metrics, including mean Average Precision
   624	(mAP) and Cumulative Matching Characteristics (CMC).
   625	
   626	B. Implementation Details
   627	We adopt the ViT [17] pretrained on the ImageNet as
   628	the baseline structure. If no other instructions are provided,
   629	all images used are resized to 256 × 128, and the training
   630	samples are augmented through techniques including random
   631	erasing [44], random horizontal flipping, random cropping, and
   632	padding. To avoid domain shift during testing in the OccludedREID, additional colour jitter augmentation is used based
   633	on [29]. Throughout the training process, every mini-batch
   634	is composed of 64 images, with each image selected from
   635	4 distinct identities. SGD is used as the optimizer using a
   636	momentum value of 0.9 and weight decay set to 1e-4. The
   637	learning rate starts at 0.008 and decays following a cosine
   638	learning rate decay schedule. When training on the OccludedDuke, the hyper-parameters s and m in the Arcface loss are
   639	configured to 30 and 0.5, respectively. While training on the
   640	Occluded-REID, considering the potential risk of overfitting
   641	due to the differences in domain between Market-1501 and
   642	Occluded-REID, we reduce m to 0.3 during training. To obtain
   643	mask semantic information from person images, we employ
   644	HRNet [35] pretrained on the ImageNet. All experiments were
   645	performed using an RTX 3090 GPU.
   646	C. Comparison With the State-of-the-Art Methods
   647	We conduct comparisons between our approach and the
   648	current state-of-the-art approach on five different datasets,
   649	which include occluded person Re-ID as well as holistic
   650	person Re-ID.
   651	1) Results on Occluded Datasets: To comprehensively
   652	demonstrate the efficiency of our proposed approach, we contrast it with state-of-the-art methods on Occluded-Duke
   653	and Occluded-REID datasets. Specific comparison results
   654	are depicted in Table II. We compare three distinct categories of methods: (1) holistic Re-ID methods, such as
   655	FD-GAN [45] and PCB [46], (2) occluded Re-ID methods
   656	including DSR [47], PGFA [40], FPR [28], PVPM [48], and
   657	HOReID [29], (3) transformer-based occluded Re-ID methods
   658	including TransReID [7], FED [49], DPM [8], PAT [5],
   659	PFD [6], FMD [50], AMG [10], FCFormer [12] and SAP [51].
   660	From Table II, it is evident that the transformer-based
   661	architecture has advantages in solving occlusion situations.
   662	
   663	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   664	
   665	ZHANG et al.: MAHATMA FOR OCCLUDED PERSON RE-IDENTIFICATION
   666	
   667	5827
   668	
   669	TABLE II
   670	
   671	TABLE IV
   672	
   673	C OMPARISON OF P ERFORMANCE W ITH S TATE - OF - THE -A RT A PPROACHES
   674	ON O CCLUDED -D UKE , O CCLUDED -REID. T HOSE SOTA A PPROACHES
   675	A RE G ROUPED I NTO T HREE C ATEGORIES F ROM T OP TO B OTTOM : H AND -C RAFTED S PLITTING BASED , O CCLUDED R E -ID
   676	BASED , AND T RANSFORMER BASED . “∗ ” D ENOTES THE
   677	E NCODER A DOPTING A S MALL S TRIDE S LIDING W IN DOW S ETTING . T HE H IGHEST P ERFORMANCE M ETRIC IN E ACH C OLUMN I S H IGHLIGHTED IN B OLD ,
   678	W HILE THE S ECOND -B EST I S U NDERSCORED
   679	
   680	C OMPARISON OF P ERFORMANCE W ITH S TATE - OF - THE -A RT A PPROACHES
   681	ON THE MSMT17 DATASET. T HE H IGHEST P ERFORMANCE M ETRIC IN E ACH C OLUMN I S H IGHLIGHTED IN B OLD , W HILE THE
   682	S ECOND -B EST I S U NDERSCORED
   683	
   684	TABLE III
   685	C OMPARISON OF P ERFORMANCE W ITH S TATE - OF - THE -A RT A PPROACHES
   686	ON M ARKET-1501 AND D UKE MTMC- RE ID DATASETS . T HE H IGHEST
   687	P ERFORMANCE M ETRIC IN E ACH C OLUMN I S H IGHLIGHTED IN
   688	B OLD , W HILE THE S ECOND -B EST I S U NDERSCORED
   689	
   690	MAHATMA achieves better performance on different datasets
   691	due to its ability to accurately locate non-occluded body parts
   692	and capture discriminative fine-grained features, achieving a
   693	73.3%/85.8% Rank-1 accuracy and a 62.3%/79.5% mAP on
   694	Occluded-Duke and Occluded-REID datasets, respectively.
   695	
   696	Our proposed MAHATMA approach demonstrates superior performance on occluded datasets, and the reasons can
   697	be attributed to the following points. Firstly, in comparison
   698	to CNNs, transformer exhibits superior feature representation capabilities, enabling them to focus more effectively
   699	on discriminative features. Secondly, we fully recognize that
   700	the transformer itself has limitations in accurately locating non-occluded body parts and extracting discriminative
   701	fine-grained features when solving occluded person Re-ID
   702	task. Therefore, our proposed method combines mask priors, hierarchical aggregation strategy, and feature completion
   703	strategy to learn robust and informative person feature representations. Finally, the various loss functions we proposed
   704	allow each module to effectively play its role by optimizing
   705	the model training objectives, thus significantly enhancing the
   706	generalization capability of our overall architecture.
   707	2) Results on Holistic Datasets: To further validate the
   708	efficiency of our approach for the holistic person Re-ID
   709	task, experiments are carried out on three holistic datasets,
   710	comprising Market1501, DukeMTMC-reID and MSMT17.
   711	Table III depicts the experimental results of our approach
   712	on Market-1501 and DukeMTMC-reID datasets. We compare
   713	three distinct categories of methods: (1) part-based Re-ID
   714	methods including PCB [46], DSR [47], and VPM [23], (2)
   715	occluded Re-ID methods including HOReID [29], PGFA [40],
   716	and FPR [28], (3) transformer-based Re-ID methods including
   717	TransReID [7], DPM [8], FED [49], PFD [6], AMG [10],
   718	PAT [5], FCFormer [12] and SAP [51]. From Table III,
   719	observations reveal that our approach attains a 95.2%/91.2%
   720	Rank-1 accuracy and a 88.2%/81.2% mAP on the Market1501
   721	and DukeMTMC-reID datasets respectively. We additionally
   722	carry out experiments on the MSMT17 dataset to assess
   723	the efficiency of our proposed approach. Several methods
   724	are compared, including PCB [46], OSNet [52], RGASC [25],TransReID [7], PFD [6], SAP [51]. From Table IV,
   725	we can see that our approach attains a Rank-1 accuracy
   726	of 85.6% and a mAP of 68.1% on MSMT17. From the
   727	experimental results in above two tables, it can be concluded
   728	that although the model we proposed is to solve the occlusion
   729	problem rather than the holistic problem, we still achieve
   730	
   731	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   732	
   733	5828
   734	
   735	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   736	
   737	TABLE V
   738	
   739	TABLE VI
   740	
   741	E VALUATION OF E ACH M ODULE ’ S I MPACT W ITHIN MAHATMA ON
   742	THE O CCLUDED -D UKE DATASET. H ERE , MIE R EFERS TO THE M ASK
   743	I NFORMATION E MBEDDINGS M ODULE , HFA R EFERS TO THE H IER ARCHICAL F EATURE AGGREGATION M ODULE , DFC R EFERS TO
   744	THE D IVERSE F EATURE C OMPLETION M ODULE , L lc R EFERS
   745	TO THE L OCAL C ONTRASTIVE L OSS , L cs R EFERS TO THE
   746	C OSINE S IMILARITY L OSS
   747	
   748	A NALYSIS OF THE M ASK E XTRACTOR A RCHITECTURE
   749	
   750	comparable performance on holistic datasets compared to
   751	other holistic Re-ID methods, demonstrating the resilience of
   752	the proposed approach.
   753	D. Ablation Study
   754	To next showcase the efficiency of MAHATMA, we carry
   755	out extensive ablation studies on the Occluded-Duke dataset.
   756	We initially assess the contribution of each component to the
   757	model performance and visualize the distribution of features
   758	extracted by the model under the action of different components, and then proceed to scrutinize the effectiveness of the
   759	mask extractor architecture and the impact of different parameter configurations (the sole additional hyperparameter in our
   760	model). Finally, we deliberate on an appropriate configuration
   761	for the number of branches within a multi-branch completion
   762	network tailored explicitly for occluded person Re-ID task.
   763	1) Effectiveness of Each Component: The overall setup
   764	remains the same, with only the evaluated module being
   765	used or removed from MAHATMA. As depicted in Table V,
   766	we assess the effect of the mask information embedddings
   767	(MIE) module, hierarchical feature aggregation (HFA) module, diverse feature completion (DFC) module, and the loss
   768	constraints specifically applied to HFA and DFC modules.
   769	Index-1 represents our baseline architecture, which is a pure
   770	transformer encoder architecture. From index-2 to index-4,
   771	we can see that adding various modules on top of the original
   772	baseline brings significant performance improvements. Specifically, with the incorporation of the MIE module, resulting in a
   773	3.5% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that MIE can avoid being affected
   774	by noise information and successfully direct the model’s
   775	attention towards non-occluded body parts. After adding the
   776	HFA module, resulting in a 5.6% enhancement in Rank-1
   777	accuracy and a 4.5% enhancement in mAP, indicating that the
   778	HFA can correctly aggregate high-quality features at different
   779	levels and focus on extracting discriminative fine-grained local
   780	features. With the introduce of the DFC module, resulting in a
   781	3.6% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that the DFC can successfully learn
   782	
   783	enough context information to help the model obtain robust
   784	global features. By comparing Index-5 and Index-6, it can
   785	be found that although DFC can capture rich global features,
   786	the overall accuracy is low due to the lack of extraction of
   787	local fine-grained features. The combination of MIE and HFA
   788	models not only uses the MIE module to locate non-occluded
   789	parts, but also aggregates fine-grained local features through
   790	the HFA module, thereby improving the model performance.
   791	Additionally, it becomes apparent that MIE, HFA, and DFC
   792	complement each other, and their combination on top of the
   793	baseline structure brought great performance enhancement to
   794	MAHATMA with a 71.9% Rank-1 accuracy and a 61.4% mAP.
   795	Moreover, from index-7 and index-8, it is evident that our
   796	comprehensive model demonstrates peak performance with
   797	a Rank-1 accuracy of 73.3% and a mAP of 62.3%, which
   798	indicates the efficiency of our proposed local contrastive loss
   799	and cosine similarity loss.
   800	To further verify the effectiveness of the proposed components, we use t-SNE [53] to visualize the distribution of
   801	features extracted by the model under different components,
   802	and the results are shown in Fig 3. The circles in the figure
   803	represent image features randomly selected from the train set
   804	of the Occluded-Duke dataset, and different colors represent
   805	different identities. Specifically, Fig 3(a) shows the distribution
   806	of features extracted by the baseline model. It is obvious
   807	that a large number of outlier features are generated due
   808	to the influence of occlusion. These outlier features mainly
   809	due to the existence of similar occlusion between different
   810	identities. Fig 3(b) shows the distribution of features after the
   811	introduction of the MIE module. With the accurate positioning of non-occluded body parts, the boundaries inter-classes
   812	gradually become clear. Fig 3 (c) and Fig 3(d) show the distribution of features after adding the HFA and DFC modules,
   813	respectively. By jointly obtaining fine-grained local features
   814	and rich global features, the influence of occlusion on the
   815	feature extraction ability of the model is significantly reduced,
   816	the intra-class distance is more compact, and the boundaries
   817	inter-classes are clearer. Finally, Fig 3(e) shows the feature
   818	distribution of the complete MAHATMA model. It can be
   819	seen that the outlier features almost completely disappear,
   820	further proving the excellent performance of the proposed
   821	components.
   822	2) Effectiveness of Mask Extractor Architecture: To verify
   823	the effectiveness of HRNet as the backbone network of the
   824	mask extractor, we conduct a comparative experiment on the
   825	effects of different mask extractor architectures. The specific
   826	comparison settings are shown in TableVI, we additionally
   827	use ResNet-50 and ResNet-50-ibn as comparison backbone
   828	
   829	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   830	
   831	
codex
为了把“触发观察”和“重定义动作”写准，我会补看几篇引言末尾的动机段。方法本身已经读完了，但这些论文通常在引言里把旧问题改名成新问题，这正是你要学的部分。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '55,140p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    55	1. Introduction
    56	The person re-identiﬁcation task aims to recognize the same individual from a large-scale database based on a given query image (Liu
    57	et al., 2023a,b; Wang et al., 2019). This technology is primarily applied
    58	in intelligent video surveillance systems, where it processes visual person imagery captured by distributed camera networks to enhance public security and monitoring eﬃciency. Recently, many re-identiﬁcation
    59	models have been trained oﬄine, meaning they are trained and deployed within a ﬁxed scenario. However, in real-world applications, reidentiﬁcation models are often trained online and continually deployed
    60	in diverse, novel environments. Therefore, the lifelong re-identiﬁcation
    61	
    62	(LReID) task (Cui et al., 2024; Pu et al., 2023; Zhang et al., 2023) is proposed, which aims to construct a robust ReID model (Jiao et al., 2023;
    63	Nguyen et al., 2024) that can ﬁt into online training and adapt to individual training domains.
    64	In LReID, models are required to undergo iterative training on sequentially provided new training data, with previous training data becoming inaccessible in subsequent iterations. This gives rise to the classic continual learning challenge of catastrophic forgetting, where models tend to overﬁt the current training domain, resulting in signiﬁcant performance degradation on previously learned domains. Existing LReID methods primarily address this issue from two perspectives:
    65	data replay and knowledge distillation. Among methods based on data
    66	
    67	⋆
    68	This work is supported by the Guangdong Basic and Applied Basic Research Foundation (No.2025A1515011465), the National Natural Science Foundation of
    69	China (No. 62502387), the China Postdoctoral Science Foundation (No. BX20250486), and the Natural Science Basic Research Program of Shaanxi (No. 2025JCYBQN-861), the National Natural Science Foundation of China (No. U23B2013)
    70	∗
    71	Corresponding author.
    72	E-mail addresses: erichwang1225@mail.nwpu.edu.cn, ggchideaki@gmail.com (Z. Chen), bingliang.jiao@mail.nwpu.edu.cn (B. Jiao), wxwang@nwpu.edu.cn (W.
    73	Wang), peng.wang@nwpu.edu.cn (P. Wang).
    74	1
    75	The ﬁrst two authors contribute equally.
    76	
    77	https://doi.org/10.1016/j.neunet.2026.108566
    78	Received 17 March 2025; Received in revised form 22 December 2025; Accepted 7 January 2026
    79	Available online 9 January 2026
    80	0893-6080/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    81	
    82	Neural Networks 198 (2026) 108566
    83	
    84	Z. Chen et al.
    85	
    86	Fig. 1. In this ﬁgure, the upper part illustrates a static LReID model trained through continual learning, where parameter units are restricted from updating to
    87	prevent overﬁtting to any single domain, with the goal of enhancing overall performance across all training domains. However, this strategy may result in suboptimal
    88	performance within speciﬁc domains. The lower part presents our designed dynamic LReID model, which incorporates partially dynamic parameter units capable of
    89	ﬂexibly adapting to the current distribution based on domain-related information within the input images. This enables the model to ﬂexibly adapt to diverse target
    90	domains, thereby achieving optimal performance across diﬀerent domains.
    91	
    92	replay, iCaRL (Rebuﬃ et al., 2017) eﬀectively mitigates catastrophic
    93	forgetting by preserving representative exemplars from historical data
    94	and incorporating a nearest-mean classiﬁcation strategy, although it incurs ongoing storage overhead and potential privacy risks. PTKP (Ge
    95	et al., 2022) further enhances knowledge retention through prototype
    96	transfer and knowledge graph construction mechanisms; however, its
    97	computational complexity increases substantially as the number of tasks
    98	grows. In the direction of knowledge distillation, LwF (Li & Hoiem,
    99	2017) employs distillation loss to maintain consistency in model outputs but struggles with continuously arriving tasks exhibiting large domain shifts. UCIR (Hou et al., 2019) preserves historical knowledge by
   100	maintaining inter-class similarity relationships, yet it faces limitations
   101	in balancing performance between old and new tasks. Although existing methods have achieved progress in speciﬁc scenarios, methods based
   102	on data replay generally impose additional storage costs, while methods
   103	based on knowledge distillation often fail to achieve an optimal balance
   104	between old and new domains.
   105	In this work, we propose the DKAR model, a novel ReID model
   106	designed to address these challenges. As shown in Fig. 1, unlike previous methods that prevent model ﬁtting to novel domains to avoid
   107	knowledge forgetting, our model generates dynamic parameters to construct customized feature extraction paradigms for each domain. This
   108	enables our model to readapt to each training domain, ensuring recognition accuracy across all domains. Speciﬁcally, the DKAR model consists of two major components: a Dynamic Adaptation (DyAdap) module
   109	and an Adaptability Retention (AdRet) strategy. The DyAdap module
   110	is designed to adapt our model to diverse domains via dynamic convolution operations (Han et al., 2024; Zhang et al., 2020). Considering that feature distribution gaps among diverse domains can hinder
   111	model adaptation, the DyAdap module ﬁrst employs an instance-level
   112	normalization layer to eliminate these distributional divergences. The
   113	normalized features are then fed into a dynamic convolution module,
   114	where the convolutional parameters are generated based on the input
   115	features themselves. By leveraging domain- and instance-aware information within the input features, our DyAdap module eﬀectively constructs customized feature extractors for individual domains, helping
   116	our ReID model achieve optimal adaptation and promising identiﬁcation performance across these domains.
   117	Furthermore, to prevent our model from losing its ability to adapt to
   118	previously trained domains when trained on new data, we propose an
   119	Adaptability Retention (AdRet) strategy. This strategy focuses on train-
   120	
   121	ing the DyAdap module’s parameter predictor. By retaining the parameter predictor from the previous stage, it imposes a correlation constraint
   122	between the outputs of the current and previous stage predictors when
   123	processing the same image input, encouraging the model to retain its
   124	adaptability to previously trained domains. Through extensive experiments, the eﬀectiveness of our designed module and training strategy is
   125	thoroughly evaluated. Additionally, empowered by our designed components, our DKAR model achieves promising performance in the mainstream lifelong re-identiﬁcation setting.
   126	Our contributions can be summarized as follows.
   127	•
   128	
   129	In this work, we propose a novel approach to tackle the lifelong person re-identiﬁcation task by constructing a dynamic framework that
   130	enables the model to adapt ﬂexibly to diﬀerent training domains,
   131	thereby preventing knowledge forgetting caused by overﬁtting.
   132	• We propose a novel DKAR model for the lifelong ReID task. In our
   133	model, a dynamic adaptation module is designed to adapt the ReID
   134	model to diverse domains through dynamic convolution operations.
   135	Additionally, we introduce an adaptability retention strategy to preserve the model’s ability to adapt to previously seen domains while
   136	training on novel ones.
   137	• Through extensive ablation experiments, we thoroughly evaluate the
   138	eﬀectiveness of our designed module and strategy.
   139	2. Related work
   140	In this section, we conducted a survey on the common research approaches and achievements in the ReID and LReID problems. Additionally, our proposed DKAR model not only mitigates the catastrophic forgetting phenomenon of ReID models in the lifelong learning (Zhao et al.,

exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '70,140p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    70	Re-ID. Nevertheless, in practical situations, as illustrated in
    71	Fig. 1(a), individuals are frequently prone to occlusion by
    72	obstacles (e.g., pedestrians, cars, trees, roadblocks), making
    73	the accurate matching of individuals with incomplete and
    74	obscured body parts a challenging task. Consequently, the
    75	research on occluded person Re-ID holds substantial practical
    76	significance.
    77	Recently, benefiting from the multi-head self-attention
    78	(MSA) mechanism that drives a model to focus on different
    79	body parts and captures long-range dependencies, transformerbased methods [5], [6], [7], [8], [9] have shown promising
    80	advancements in the occluded person Re-ID task. However,
    81	these methods still have following limitations: (i) Due to the
    82	diversity of occlusion types, existing methods cannot precisely
    83	focus on non-occluded body parts [7], [10]. As shown in
    84	Fig 1(b), MSA can guide model (TransReID) to focus on a
    85	discriminative part, but it may also introduce background and
    86	occlusion information in the feature embedding (as indicated
    87	by the red box). To address this issue, some studies [11], [12],
    88	[13] have utilized additional semantic information to enhance
    89	feature robustness and achieved some effective results, but
    90	these methods still fail to completely avoid the limitation that
    91	MSA is highly sensitive to background and occlusion information, which in turn limits the ability of accurate positioning.
    92	(ii) Current methods fail to effectively capture fine-grained
    93	features in images [9], [14], [15]. The main reason is that
    94	most of these methods rely on MSA to assist the model in
    95	capturing global dependencies, while ignoring fine modeling
    96	of details between pixels in different patches, resulting in the
    97	loss of fine-grained features and contextual information.
    98	We design an architecture tailored specifically for the
    99	occluded person Re-ID task, which can not only precisely focus on non-occluded body parts, but also extract
   100	fine-grained feature information for pedestrian identity matching. Fig 2 illustrates our proposed framework, which we
   101	call Mask-Aware Hierarchical Aggregation TrAnsforMer
   102	(MAHATMA). Specifically, MAHATMA consists of a mask
   103	extractor, a Mask Information Embeddings (MIE) module,
   104	a Hierarchical Feature Aggregation (HFA) module, and a
   105	Diverse Feature Completion (DFC) module. Firstly, we direct
   106	the model to specifically focus on non-occluded body parts in
   107	occlusion scenarios by designing the MIE module to encode
   108	the mask semantic information of body parts obtained through
   109	a mask extractor, mitigating the impact of occlusion on the
   110	
   111	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
   112	and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   113	See https://www.ieee.org/publications/rights/index.html for more information.
   114	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:11 UTC from IEEE Xplore. Restrictions apply.
   115	
   116	5822
   117	
   118	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   119	
   120	Fig. 1. Visualization of images with different occlusion types. The first, second and third columns represent the occluded input images, attention heatmaps
   121	of TransReID and MAHATMA, respectively. Compared to TransReID, the
   122	proposed MAHATMA can effectively mitigate the impact of occlusions and
   123	focus on non-occluded body parts, as indicated by the red box, thus extracting
   124	discriminative feature representations.
   125	
   126	precise positioning of the model. Secondly, our proposed
   127	HFA module is to mine more exploitable high-quality detail
   128	information by aggregating hierarchical image patch representations to capture fine-grained local features. To further
   129	alleviate the problem of feature loss under occlusion conditions and enhance the model’s global perception capability,
   130	we propose the DFC module that is able to compensate
   131	for the possible loss of global features through a multi-path
   132	feature integration approach. We perform experiments on
   133	diverse datasets, encompassing both holistic (Market-1501,
   134	DukeMTMC-reID, MSMT17) and occluded (Occlued-Duke,
   135	Occlued-REID) scenarios. A wealth of experimental results
   136	corroborate the validity of our approach.
   137	The new contributions in this paper are summarised as
   138	follows:
   139	• A novel framework for occluded person Re-ID, called
   140	Mask-Aware Hierarchical Aggregation TrAnsforMer

exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '80,170p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    80	the visible light modality in performing image retrieval under low-light conditions, it is typically
    81	necessary to develop more advanced models for VI-ReID to address the differences in cross-modal
    82	features and reduce modality discrepancies.
    83	Existing outstanding works proposed in VI-ReID mainly concentrated on convolutional neural
    84	networks (CNNs) [20, 61, 64]. However, as shown in Figure 1(c), due to the inherent local receptive
    85	field properties of CNNs, the network emphasizes certain local features of images while failing to
    86	recognize long-range dependencies, resulting in the loss of some global features. To address this
    87	limitation, transformers have been widely adopted in computer vision. This attention mechanism
    88	can model extensive dependencies between tokens in data sequences, thereby enhancing the
    89	ability of the network to capture global features. However, as the token proceeds deeper into
    90	the network, the attention maps become increasingly globalized. Vision Transformer (ViT)
    91	aggregates features between patches at a global scale, resulting in lost focus on local information
    92	and capturing extra irrelevant information [36], as shown in Figure 1(b). Therefore, one of our aims
    93	is to enable the network to learn comprehensive global features, while emphasizing certain effective
    94	discriminative local features. In addition, the self-attention mechanism in ViT is characterized by
    95	feature collapse in deeper layers leading to the vanishing of low-level visual features [46]. To provide
    96	evidence for the phenomenon of feature collapse that occurs as the network deepens in VI-ReID,
    97	Figure 2 presents a visual representation of the feature similarity matrices for ViT and LocalAware Residual Attention Vision Transformer (LAReViT) across different modalities. The
    98	matrices were constructed by calculating the cosine similarity between distinct feature patches
    99	extracted by models. We observe that as the features progress from the shallower to the deeper
   100	layers of the model, the similarity between distinct patches gradually increases. The increased
   101	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   102	
   103	Local-Aware Residual Attention Vision Transformer for VI-ReID
   104	
   105	146:3
   106	
   107	Fig. 1. Motivation: Analyze feature maps learned from GradCAM [42] algorithm. CNN-based networks often
   108	emphasize local information, which results in the loss of global features. ViT is devoted to capturing global
   109	features but faces the challenge of over-globalization, leading to the redundancy of irrelevant information
   110	such as redundant background. Although global features are crucial, it is equally important to emphasize
   111	discriminative local features, which assist in guiding the network to express comprehensive discriminative
   112	features. Therefore, exploring valuable local information on transformer is crucial, and our proposed LAReViT
   113	precisely captures effective cross-modal person feature. CNN, convolutional neural network; LAReViT, LocalAware Residual Attention Vision Transformer; ViT, vision transformer.
   114	
   115	similarity among distinct patches means the loss of feature diversity, resulting in feature collapsing.
   116	The proposed LAReViT, through an innovative residual connection between attention layers,
   117	effectively alleviated this phenomenon. The excessive similarity among patches is caused by
   118	feature collapse, resulting in the loss of low-level fine-grained feature information [6, 23, 75]. In
   119	VI-ReID, low-level fine-grained cues typically represent details such as color and texture [71].
   120	Significantly, these low-level cues, typically found in shallower network layers, are crucial for
   121	cross-modal matching. Therefore, another of our aims is to optimal the utilization of those shallow
   122	low-level cues.
   123	In this article, we leverage the strengths of transformer in modeling global information, while
   124	enhancing the advantages of convolution operations in patch embedding regarding the local
   125	receptive field. Additionally, we thoroughly consider the utilization of shallow-layer low-level
   126	features. Consequently, the LAReViT is proposed to capture and preserve local low-level features,
   127	addressing the issue of transformer features over-globalization and collapse in VI-ReID. This
   128	approach aims to bridge the modal gap within a semantically richer high-level feature space,
   129	thereby enhancing the comprehensiveness and discriminativeness of person features across the
   130	visible and IR modals. Different from the existing pure transformer-based VI-ReID method, we
   131	designed a Local-Aware Residual (LAR) Module with a Local Residual Attention (LRA)
   132	mechanism therein to enhance the transmission and accumulation of attention-related information
   133	from shallow to deep layers. LAR allows the network to provide more low-level detailed information
   134	about local elements of pedestrians and capture and retain crucial low-level pedestrian features
   135	across different modalities. Through utilizing those detailed crucial elements, we guide the model in
   136	learning more comprehensive feature representations. Moreover, we addressed the issue of the overglobalization of transformer features by leveraging fine-grained local information from different
   137	parts of the human body, which enables transformer to extract local features more effectively. Finally,
   138	taking into account the advantages of convolution kernel in local receptive fields, we innovatively
   139	proposed a Positional Channel Reconstruction (PCR) Module that reweights each patch block
   140	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   141	
   142	146:4
   143	
   144	X. Hua et al.
   145	
   146	Fig. 2. Feature similarity matrix across various layers for ViT and LAReViT under the visible and IR model.
   147	The color scale indicates the similarity between feature patches, with a higher degree of green denoting
   148	greater similarity among the patches.
   149	
   150	in ViT. Different from the CNN-based Squeeze-and-Excitation (SE) block in SEnet [19], our
   151	proposed PCR merely enhances the advantages of convolution used in patch embeddings within
   152	ViT. The former adaptively recalibrates channel-wise feature responses by explicitly modeling
   153	interdependencies between channels through global average pooling to generate channel-wise
   154	statistics. The PCR adopts a weighted combination of Average Pooling and Maximum Pooling
   155	methods. We ensure that the network can focus more effectively on identification information while
   156	suppressing less useful patch embedding. The novel PCR strengthens the learning of effective local
   157	information by transformer and maximizes the advantages of the convolution in feature patches
   158	along with the attention mechanism in transformer block.
   159	In summary, by combining LAR and PCR, the proposed LAReViT demonstrates innovative
   160	capabilities in preserving shallow-layer low-level features and emphasizing local features, while
   161	retaining the original ability of the transformer to extract global context. We endow the attention
   162	mechanism with the capability to capture low-level effective local feature, thereby enabling the
   163	learned person features to exhibit greater diversity. Furthermore, we bridge the gap between
   164	pedestrian features across different modalities through the proposed Center Aggregation Loss
   165	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
   166	
   167	
exec
/bin/zsh -lc "pdftotext -f 1 -l 4 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '70,175p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    70	1. Introduction
    71	The Visible–Infrared Person Re-Identification (VI-ReID) task aims to
    72	identify the individual with the most matching identity from a database
    73	composed of images captured under both visible and infrared modalities. Unlike visible ReID tasks, which are confined to daytime applications, the VI-ReID task integrates both the visible and infrared modality,
    74	enabling all-day intelligent surveillance for security purposes, given its
    75	significant practical value. Many academic studies have approached
    76	enhancing the performance of VI-ReID from various perspectives (Wang
    77	et al., 2019b; Zhang and Wang, 2023; Li et al., 2020).
    78	However, in practical applications of VI-ReID, there often exists an
    79	expansion and scalability requirement for the camera-based databases;
    80	meanwhile, these models trained solely on static datasets are not
    81	suitable in the real world where data continually pours in. As depicted
    82	in Figs. 1(c) and 1(d) under the joint learning scenario, if every new
    83	
    84	data domain necessitates retraining, it would incur substantial time
    85	and memory resource expenditures. Moreover, as depicted in Fig.
    86	1(b), the model attains heightened generalization capability and adaptability through the continuous accrual of knowledge. Consequently,
    87	the VI-ReID model necessitates the capacity for lifelong learning to
    88	accommodate the ever-changing data domains.
    89	To achieve the aforementioned objective, as depicted in Fig. 1(a),
    90	TTQK (Xing et al., 2024) introduced the Lifelong Visible–Infrared Person Re-Identification (LVI-ReID). This task requires that the model can
    91	adapt to continuously arriving new data domains encompassing both
    92	visible and infrared modalities, while also maintaining its identification
    93	capabilities on previously encountered domains. We refer to these two
    94	key characteristics as domain adaptability and anti-forgetting ability.
    95	Addressing the requirements and characteristics of LVI-ReID, we face
    96	
    97	∗ Corresponding author at: College of Computer and Information Science, Southwest University, Chongqing, 400715, China.
    98	
    99	E-mail address: songwuswu@swu.edu.cn (S. Wu).
   100	https://doi.org/10.1016/j.cviu.2025.104328
   101	Received 13 May 2024; Received in revised form 28 January 2025; Accepted 24 February 2025
   102	Available online 4 March 2025
   103	1077-3142/© 2025 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
   104	
   105	X. Zhu, G. Xiao, M.S. Lew et al.
   106	
   107	Computer Vision and Image Understanding 254 (2025) 104328
   108	
   109	address in its design. Secondly, as illustrated in Fig. 1(f), addressing the
   110	dual deterioration in adaptability and anti-forgetting ability caused by
   111	the domain semantic gap is another pressing challenge. Finally, TTQK
   112	does not leverage more fine-grained information, so how to fully utilize
   113	the potential of the learned model within environments characterized
   114	by dynamic expansion of camera networks and limited storage space is
   115	also an equally important challenge.
   116	Prior studies have partly focused on realizing lifelong person reidentification under a single visible modality. For instance, AKA (Pu
   117	et al., 2021) implemented knowledge accumulation and manipulation
   118	by deploying the AKA network, while PTKP (Ge et al., 2022) employed pseudo-labeling techniques to mitigate the domain semantic gap
   119	between continuous learning data domains. However, due to the significant discrepancies across different data modalities and domains, these
   120	previous researches have not adequately addressed the multifaceted
   121	challenges inherent in LVI-ReID.
   122	Thus, we propose an innovative LVI-ReID framework that aims to
   123	address these challenges from both the image and feature levels. At the
   124	image level, inspired by the principles of PCB (Sun et al., 2018) and
   125	CutMix (Yun et al., 2019), coupled with the inherent advantages of
   126	Memory Banks, we introduce the Domain-Modality-Mix (DMM) based
   127	replay samples reconstruction strategy. This strategy involves randomly
   128	mixing and reconstructing stored replay sample images across different
   129	partitioned regions with a certain probability, crossing domains and
   130	modalities. Operating the Domain-Modality-Mix strategy within a fixed
   131	storage space can amplify the diversity of replay samples, thereby
   132	effectively mitigating catastrophic forgetting caused by cross-domain
   133	and cross-modality semantic gaps.
   134	Moreover, at the feature level, drawing inspiration from Chunking
   135	Theory (Gobet et al., 2001) in cognitive psychology and leveraging the
   136	robust feature aggregation properties of Graph Convolutional Networks
   137	(GCNs) (Jiang et al., 2019), we designed a novel Cross-domain Cognitive Network (CDCN) to promote the extraction of cross-modal invariant features. By implementing a camera-aware, dynamically expanding
   138	GCN framework, the cross-domain cognitive network harmoniously integrates visible and infrared data. This integration not only bridges the
   139	cross-modal disparity across disparate camera types but also inherently
   140	consolidates multi-domain information, leading to a substantial boost
   141	in the LVI-ReID performance of the learned model. Extensive experiments show the effectiveness of our proposed methodology. The main
   142	contributions of our proposed method can be summarized as follows:
   143	• Replay Samples Domain-Modality-Mix Reconstruction: During the rehearsal phase in LVI-ReID, we innovatively designed a
   144	Domain-Modality-Mix (DMM) based replay sample reconstruction
   145	strategy. The DMM can generate diversified cross-domain and
   146	cross-modality samples as data augmentation, effectively serving
   147	as a countermeasure against catastrophic forgetting in LVI-ReID.
   148	• Cross-domain Cognitive Network: To enhance the domain adaptability of the learned LVI-ReID model for the continuous crossdomain data streams, we designed a Cross-domain Cognitive Network (CDCN) framework, which is based on a camera-extended
   149	graph convolutional network to fuse cross-modal information and
   150	integrate cross-domain information at the feature level, having
   151	the advantages of real-world applications.
   152	• LVI-ReID Task Performance: We tackle the LVI-ReID task at
   153	a more fine-grained level and explore a highly extensible baseline for the LVI-ReID task. Extensive experiments demonstrate
   154	that our proposed DMM strategy and CDCN framework have remarkable domain adaptability and robust resistance to forgetting
   155	and outperform multiple state-of-the-art methods in comparative
   156	assessments of the performance of LVI-ReID.
   157	
   158	Fig. 1. Motivation of Our Work. To mitigate the substantial computational cost in
   159	terms of time and memory associated with retraining models at different stages (1(c)
   160	and 1(d)), we propose the LVI-ReID task (1(a)), endowing models with lifelong learning
   161	abilities that cumulatively accumulate knowledge for enhanced generalization and
   162	adaptability (1(b)). During the process of continual learning, we need to simultaneously
   163	manage the decline in model adaptability and catastrophic forgetting caused by modal
   164	gaps (1(e)) and domain gaps (1(f)).
   165	
   166	multiple challenges: Firstly, we need to tackle the dual decline in model
   167	adaptability and anti-forgetting ability due to significant cross-modal
   168	discrepancies (as shown in Fig. 1(e)), which TTQK did not specifically
   169	2
   170	
   171	X. Zhu, G. Xiao, M.S. Lew et al.
   172	
   173	Computer Vision and Image Understanding 254 (2025) 104328
   174	
   175	2. Related work

codex
我还会补第一篇引言后半段，因为这类统一视角论文的“重定义动作”通常写在引言里，而不是方法公式里。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 5 'Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf' - | nl -ba | sed -n '95,235p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    95	this problem, we provide a new perspective on learning human representations on the basis of
    96	visual-semantic feature, even in a scenario with incomplete modality (e.g., unimodal vision-only or
    97	language-only data), and propose a Multi-Modal Equivalent Transformer (MMET) for more
    98	robust visual-semantic embedding learning for Re-ID event. On the basis of it, a dynamic masking
    99	mechanism called Masked Multimodal Modeling (MMM) is introduced to mask both the image
   100	patches and the text tokens, which can jointly work on unimodal visual or textual modality and
   101	significantly boost the performance of generalizable person Re-ID. As illustrated in Figure 1, the
   102	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   103	
   104	Learning Visual-Semantic Embedding for Generalizable Person Re-identification
   105	
   106	278:3
   107	
   108	Fig. 1. We present MMET, a language and vision alignment model that learns strong representations from
   109	image-text pairs and unpaired images/texts under a common Transformer model architecture.
   110	
   111	key idea is that our model is trained on a set of source domain based on the unpaired images/texts
   112	or image-text pairs, and should generalize to any new unseen datasets for effective Re-ID without
   113	any model updating. To the best of our knowledge, this is the first work aiming at investigating the
   114	DG with multiple modalities on Re-ID events.
   115	In essence, the concept of MMM appears to have similar notions explored in recent works [52,
   116	55]. For example, Yu et al. [52] propose a modified encoder-decoder architecture CoCa, which is
   117	trained with both contrastive loss and captioning loss from scratch on both Web-scale alt-text
   118	data and annotated images; Zhao et al. [55] introduce a jointly MMM method named MAMO
   119	to learn fine-grained multimodal representations, which performs joint masking on image-text
   120	input and integrates both implicit and explicit targets for the masked signals to recover. However,
   121	only multimodal data are required as image-text input for CoCa and MAMO method respectively.
   122	Compared with traditional multimodal feature fusion strategies [17, 43, 52, 55], our MMET is
   123	different from them in terms of two perspectives: (1) The input of traditional method is unimodal
   124	vision-only or language-only data, while our MMET method works on a wide range of tasks in the
   125	vision, language, and vision-and-language domains, sometimes even when one modality or more
   126	modalities are missing, which allows our model to be more flexible and adaptable in real-world
   127	scenarios; (2) Prior works mainly investigate the robustness of multimodal transformers to modalityincomplete test data, with the requirement to finetune the entire model using modality-complete
   128	training data. In contrast, our work studies a more general scenario where various modality-missing
   129	cases would occur differently not only for each data sample but also learning phases (training,
   130	testing, or both).
   131	As a consequence, the major contributions of our work can be summarized into three-fold:
   132	— We propose a MMET framework for more robust visual-semantic embedding learning.
   133	— Based on it, a dynamic masking mechanism called MMM is introduced to further boost the
   134	performance of generalizable person Re-ID.
   135	— Comprehensive experiments show that our MMET method matches or exceeds the performance of existing methods with a clear margin, which reveals the applicability of visualsemantic based pretraining with new insights.
   136	In the rest of the article, we first review some related works of person Re-ID methods and
   137	previous semantic-based methods in Section 2. Then in Section 3, we give more details about
   138	the learning procedure of the proposed MMET method. Extensive evaluations compared with
   139	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   140	
   141	278:4
   142	
   143	S. Xiang et al.
   144	
   145	state-of-the-art methods and comprehensive analyses of the proposed approach are elaborated in
   146	Section 4. Conclusion and Future Works are given in Section 5.
   147	2
   148	
   149	Related Works
   150	
   151	In this section, we have a brief review on the related works of traditional person Re-ID methods and
   152	Transformer-based approaches. The mainstream idea of the existing methods is to learn a robust
   153	model for feature representation.
   154	2.1
   155	
   156	CNN-Based Person Re-ID Methods
   157	
   158	Actually, there are mainly two kinds of feature learning paradigms for person Re-ID tasks: (1)
   159	Hand-crafted-based method and (2) Deep learning-based approach, which are introduced as follows.
   160	Traditional research works [6, 9, 54] related to hand-crafted systems for person Re-ID aim to
   161	design or learn discriminative representation or pedestrian features. For example, Farenzena et al.
   162	[9] proposed an appearance-based method for these situations where the number of candidates
   163	varies continuously. Ding et al. [6] presented a scalable deep feature learning model for person Re-ID
   164	via distance comparison. Besides directly using mid-level color and texture features, some methods
   165	[54] also explore different discriminative abilities of local patches for better discriminative power
   166	and generalization ability. Unfortunately, these hand-crafted feature-based approaches always fail
   167	to produce competitive results on large-scale datasets. The main reason is that these early works
   168	are mostly based on heuristic design, and thus they could not learn optimal discriminative features
   169	on current large-scale dataset.
   170	Recently, benefited from the advances of deep neural networks and availability of large-scale
   171	datasets, person Re-ID performance in supervised learning has been significantly boosted to a
   172	new level [44, 49], e.g., Xiang et al. [44] propose a feature fusion strategy based on traditional
   173	convolutional neural network with attention mechanism, which learns robust feature extraction
   174	and reliable metric learning in an end-to-end manner. Gu et al. [11] propose a clothes-based
   175	adversarial loss to mine clothes-irrelevant features from the original RGB images by penalizing
   176	the predictive power of Re-ID model. Wang et al. [39] design a neighbor transformer network to
   177	explicitly model interactions across all input images for discriminative representations. Recently,
   178	Zhong et al. [58] propose a grayscale enhancement colorization network called GECNet to bridge
   179	the modality gap by retaining the structure of the colored image which contains rich information;
   180	Xie et al. [50] also present a new SRC method to address the unsupervised video person Re-ID
   181	problem, which jointly considers discriminative feature learning and label estimation for video
   182	person Re-ID. In addition, Xu et al. [51] also design a differentiable retrieval-sort loss to optimize
   183	the Re-ID model by ranking each positive sample ahead of the negative samples according to the
   184	distance and sorting the positive samples according to the angle. Even though above approaches
   185	can promote the performance of Re-ID task to a certain extent, these methods either focus on
   186	multimodal representation learning or optimizing the order of positive samples with low rank
   187	in terms of metric learning. More importantly, they always ignore the impacts of each frame
   188	on clustering and model training, especially when camera information is not provided. Besides,
   189	some recent works [4, 42] attempt to address UDA base on Generative Adversarial Network
   190	(GAN) model. Unfortunately, these approaches always require abundant computing resources to
   191	achieve satisfactory performance, and leveraging GAN network is unable to guarantee the quality
   192	of generated images.
   193	2.2
   194	
   195	Transformer-Based Methods
   196	
   197	Transformer model is proposed in [37] to handle sequential data in the field of natural language
   198	processing, which has been applied into computer vision to explore long-range dependencies with
   199	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 10, Article 278. Publication date: October 2025.
   200	
   201	Learning Visual-Semantic Embedding for Generalizable Person Re-identification
   202	
   203	278:5
   204	
   205	multi-head self-attention strategy on person Re-ID task. For example, He et al. [13] make the first
   206	attempt to apply Transformer architecture on this event, and propose a pure transformer-based
   207	object Re-ID framework with critical improvement. Liao et al. [24] also adopt Vision Transformer
   208	(ViT) for efficient image matching and metric learning tasks.
   209	As for the robust feature extraction, Chen et al. [3] design a hybrid backbone Res-Transformer
   210	based on ResNet-50 and Transformer block for effective identify information. Lai et al. [17] propose
   211	an adaptive part division model to better extract local features for person Re-ID. In addition, Xiang
   212	et al. [43] propose a deep multimodal fusion network to elaborate rich semantic knowledge for
   213	assisting in representation learning during the pretraining. As for the modality-agnostic Re-ID
   214	task, Li et al. [19] design a new multimodal Re-ID framework to tackle the challenges inherent
   215	in zero-shot multimodal Re-ID, which creatively integrates a pre-trained foundation model and
   216	a multimodal tokenizer into Re-ID tasks. Besides, Chen et al. [2] propose a unified person ReID architecture called UNIReID to mine and fuse visual and textual modality information, and
   217	then effectively adapt to cross-modality and multi-modality tasks. Although these methods can
   218	improve the performance of generalizable person Re-ID in some degree, the potential of deep
   219	multimodal feature between visual and semantic feature is always being underestimated. More
   220	importantly, these methods trend to ignore the challenge where one modality or more modalities
   221	are missing. Consequently, current methods are still far from satisfactory in generalization for
   222	practical person Re-ID.
   223	To solve these problems, in this work, we take a big step forward and design a novel MMET
   224	framework for more robust visual-semantic embedding learning on visual, textual, and visualtextual tasks. On the basis of it, a dynamic masking mechanism called MMM is introduced to mask
   225	both the image patches and the text tokens, which can jointly work on multimodal or unimodal
   226	data and significantly boost the performance of generalizable person Re-ID. It is worth mentioning
   227	that our MMET model is designed to be able to take advantage of unpaired image data or text
   228	data along with paired image-text pairs. While previous methods fail to adapt in the multimodal
   229	scenarios, especially for those cross-modal and multimodal vision-and-language tasks. To the best
   230	of our knowledge, this is the first attempt to adopt a foundational language and vision alignment
   231	model that explicitly targets vision, language, and their multimodal combination all at once. We
   232	hope that our method will serve as a strong baseline for visual-semantic embedding, and shed light
   233	into potential tasks for the community to move forward.
   234	3
   235	3.1

codex
证据闭环不能只凭方法段猜，我会额外定位每篇的消融或可视化段，尤其是 DKAR、PP-IPG、LAReViT 和 MAHATMA。这里不改任何文件，只抽关键验证设计。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | rg -n \"Ablation|ablation|DyAdap|AdRet|BNF|visual|Visualization|Table|effectiveness|Effectiveness\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
58:    58	in intelligent video surveillance systems, where it processes visual person imagery captured by distributed camera networks to enhance public security and monitoring eﬃciency. Recently, many re-identiﬁcation
108:   108	enables our model to readapt to each training domain, ensuring recognition accuracy across all domains. Speciﬁcally, the DKAR model consists of two major components: a Dynamic Adaptation (DyAdap) module
109:   109	and an Adaptability Retention (AdRet) strategy. The DyAdap module
111:   111	model adaptation, the DyAdap module ﬁrst employs an instance-level
115:   115	features themselves. By leveraging domain- and instance-aware information within the input features, our DyAdap module eﬀectively constructs customized feature extractors for individual domains, helping
119:   119	Adaptability Retention (AdRet) strategy. This strategy focuses on train-
121:   121	ing the DyAdap module’s parameter predictor. By retaining the parameter predictor from the previous stage, it imposes a correlation constraint
137:   137	• Through extensive ablation experiments, we thoroughly evaluate the
184:   184	with lower correlation to the training data and achieving excellent performance on unseen domains. In our proposed DyAdap module, we take
240:   240	a cross-domain and cross-modal ReID task that incorporates visual and
248:   248	Fig. 2. The whole framework of our proposed DKAR model along with BNF strategy. The DyAdap module enables our model to adapt to diverse unseen domains by
249:   249	importing a dynamic convolution network. The AdRet strategy uses knowledge distillation to help the DyAdap module retain its ability to adapt to previous domains
250:   250	while training on novel domains. The BNF strategy retains more information learned from previous domains while minimizing its impact on the performance of the
276:   276	In this section, we will provide a detailed introduction to our proposed DKAR model, which consists of a Dynamic Adaptation (DyAdap)
277:   277	module, an Adaptability Retention (AdRet) strategy.
282:   282	DKAR model, we ﬁrst design a dynamic adaptation module (DyAdap)
285:   285	new data, we propose an adaptability retention (AdRet) strategy. This
287:   287	parameter predictor within the DyAdap module, constraining it to adapt
300:   300	In our DKAR model, we ﬁrstly design a dynamic adaptation module to adapt our model to diverse domains. In the DyAdap module as
313:   313	to the instance normalization layer, in our DyAdap module, we deploy
346:   346	Fig. 3. The illustration about the DyAdap model, which combines instance normalization used to eliminate the inﬂuence brought by feature distribution divergence across domains with dynamic convolution used to adapt our model
351:   351	where 𝐹 𝐷𝑦𝐴𝑑𝑎𝑝 is the output of DyAdap module; 𝑐𝑜𝑛𝑣 represents convolution operation; 𝜔 denotes generated dynamic parameters. Through
353:   353	domains. In terms of structure, we use our DyAdap module to replace the
356:   356	The DyAdap module integrates dynamic convolution structures into
358:   358	adjust parameters in response to data from unseen domains. This dynamic convolution structure allows the DyAdap module to leverage
363:   363	of feature extraction capabilities learned from previous domains. Although IN method enhances the model’s domain generalization, the hypernetwork structure within the DyAdap module, responsible for generating dynamic convolution parameters, remains limited by the distribution of the current training domain. Consequently, both the feature
364:   364	extraction network and the DyAdap module are susceptible to catastrophic forgetting. In essence, the DyAdap module does not directly mitigate catastrophic forgetting but instead improves the model’s dynamic
370:   370	Retention (AdRet) strategy. Speciﬁcally, we apply knowledge distillation losses to the features processed in the DyAdap module, as well as
384:   384	the DyAdap module are guided to align with the corresponding features
425:   425	within the DyAdap module. Given the dynamic parameters 𝜔𝑡 ∈ ℝ𝐾
441:   441	After each training iteration, we perform a Batch Normalization Fusion (BNF) by merging the ﬁnal batch normalization (BN) layers of Φ𝑡−1
453:   453	The AdRet strategy designs speciﬁc knowledge distillation loss functions for both the feature extraction network and the DyAdap module,
456:   456	multiple domains, the ReID model with the DyAdap module adjusts its
458:   458	this, the AdRet strategy constrains the parameter update process of the
460:   460	DyAdap module. This enables the model to retain its feature extraction
471:   471	on seen domains and unseen domains. For ablation experiments, we designed experiments to separately evaluate the improvement brought by
509:   509	with signiﬁcant viewpoint variations and challenging lighting conditions. The images have dimensions of 128 × 48 pixels, presenting substantial diﬃculties for feature matching due to the extreme visual discrepancies between camera views. During model training, each person’s
540:   540	Model parameters 𝜃Feat , 𝜃DyAdap ,
545:   545	DyAdap
553:   553	6: Dynamic Adjustment with DyAdap:
554:   554	7: 𝑓̂ ← DyAdap(𝑓 ; 𝜃DyAdap )
571:   571	16: 𝜃DyAdap
572:   572	← 𝜃DyAdap − 𝛾2 ∇𝜃DyAdap 
577:   577	, 𝜃DyAdap
580:   580	To provide a comprehensive visual understanding of the dataset characteristics and challenges, we include representative sample images from
684:   684	As shown in Table 1, our proposed DKAR achieved the best average
710:   710	The DyAdap module we propose equips the model with the ability to dynamically adjust its parameters in response to domain variations. As a result, after continuous learning across multiple domains,
720:   720	Table 1
871:   871	Table 2
939:   939	Table 3
968:   968	shown in Table 3, the Rank-1 diﬀerences between training and testing sets remain consistently small across all datasets. The maximum observed gap does not exceed 5%, indicating that the model does not rely
978:   978	According to the results in Table 2, our proposed method outperforms all other compared methods across the three datasets. Speciﬁcally,
982:   982	that the proposed DyAdap module eﬀectively enhances the model’s domain generalization ability by dynamically generating parameter distributions better suited to the domain data encountered by the model.
1001:  1001	4.3. Ablation experiment
1003:  1003	both of the DyAdap module and AdRet strategy. Besides, we also design
1008:  1008	To evaluate the eﬀectiveness of our proposed DyAdap module along
1009:  1009	with AdRet strategy, we gradually insert them into the baseline model
1010:  1010	and compare the performance improvement. From the results in Table 4,
1015:  1015	4.3.2. Eﬀectiveness of BNF strategy
1016:  1016	To evaluate the performance of our proposed BNF strategy in addressing catastrophic forgetting, we integrated the Batch Normalization
1024:  1024	Table 4
1025:  1025	The eﬀectiveness of our designed DyAdap module, AdRet strategy, and BNF strategy.
1028:  1028	DyAdap
1030:  1030	AdRet
1032:  1032	BNF
1036:  1036	Base+DyAdap
1037:  1037	Base+AdRet
1039:  1039	Base+BNF
1065:  1065	Base+DKAR+BNF
1173:  1173	Table 5
1279:  1279	learning process. According to the results exhibited in the Table 4, it is
1280:  1280	evident that the introduced BNF strategy averagely brings 0.6% 𝑚𝐴𝑃
1281:  1281	improvement across these four datasets. This indicates that the BNF
1285:  1285	used for balancing information in the feature extraction backbone network, without handling the head network, our proposed BNF strategy proves beneﬁcial in further preserving information learned in old
1289:  1289	The results in Table 5 illustrate that diﬀerent training sequences result in minor variations in the performance of our proposed framework.
1299:  1299	4.3.3. Eﬀectiveness of knowledge distillation losses in AdRet strategy
1300:  1300	Within the AdRet strategy, there are two loss functions used for
1312:  1312	Table 6
1410:  1410	Table 7
1458:  1458	4.3.5. Ablation about hyperparameters
1462:  1462	experimental results in Table 6 reveals that the model’s performance
1483:  1483	ﬁrst dataset. Based on the experimental results exhibited in Table 1, we
1495:  1495	the model, directly indicating its memory footprint and storage requirements. According to the results presented in Table 7, it can be concluded
1499:  1499	although the DyAdap module increases the number of learnable parameters by approximately 8% by replacing the standard 3 × 3 convolution
1590:  1590	we propose the DKAR model, where a DyAdap module is designed to
1592:  1592	an AdRet strategy to help the dynamic adaptation module retain its ability to adapt to previous domains while training on novel domains. Extensive experiments demonstrate that our proposed method achieves
1595:  1595	Zhiyu Chen: Writing – original draft, Visualization, Software,

exec
/bin/zsh -lc "pdftotext 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | nl -ba | rg -n \"Ablation|ablation|Prompt Pool|Instance-level|IPG|PP|Effectiveness|Table 2|Table 3|Figure 4|outperforms|full\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2:     2	Pool and Instance-level Prompt Generator
42:    42	Prompt Pool (PP) module to encapsulate shared knowledge across
44:    44	generator (IPG) to further enhance the model’s ability to capture
48:    48	query tokens and keys, thereby addressing the nuanced requirements of varying tasks. Extensive experimental evaluations demonstrate the superiority of our proposed prompt learning-based PPIPG framework over state-of-the-art methods in both lifelong learnings, lifelong person re-identification (LReID), and LVI-ReID settings. These results underscore the efficacy and practicality of our
50:    50	and cross-domains. The source code of our designed PP-IPG method
51:    51	is at https://github.com/SWU-CSMediaLab/PP-IPG.
61:    61	Visible-Infrared Person Re-Identification with Prompt Pool and Instancelevel Prompt Generator. In Proceedings of the 2025 International Conference on
90:    90	for profit or commercial advantage and that copies bear this notice and the full citation
131:   131	to address the challenges of LVI-ReID. The first is a Prompt Pool
132:   132	(PP) module, which allows each task or domain to flexibly retrieve
133:   133	grouped prompts as model inputs. The PP module effectively captures shared knowledge across tasks or domains while maintaining
135:   135	diverse scenarios. The second is an Instance-level Prompt Generation (IPG) module, which dynamically generates instance-specific
137:   137	pool. The IPG module also utilizes task identifiers to dynamically
139:   139	The combination of PP and IPG modules significantly enhances
159:   159	The main contributions of our prompt learning-based PP-IPG
162:   162	the LVI-ReID problem. A Prompt Pool (PP) module is designed to enable flexible retrieval of grouped prompts for
167:   167	• An Instance-Level Prompt Generator (IPG) is proposed to
174:   174	• Extensive experiments demonstrate that our PP-IPG approach
175:   175	outperforms state-of-the-art lifelong learning methods, LReID
191:   191	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
196:   196	patches with positional embeddings, which are then fed into the IPG (Instance-specific Prompt Generator) module and the
199:   199	task identifier, which guides the IPG module in generating instance-level prompts. Subsequently, the instance-level prompts,
201:   201	the attention module is independently equipped with an IPG module to compute distinct instance-level prompts.
324:   324	and infrared modality 𝑚 ∈ {𝑣𝑖𝑠, 𝑖𝑟 }, and 𝑦𝑖𝑡 indicates the corresponding identity label. Our proposed PP-IPG framework for the
377:   377	The overview of our proposed PP-IPG framework for LVI-ReID is illustrated in Fig.2. In each iteration of the training process, 𝑁 images
387:   387	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
489:   489	Instance-level Prompt Generator
499:   499	design an Instance-level Prompt Generator (IPG) module that provides fine-grained guidance, assisting the model efficiently in leveraging learned representations and better capturing domain-specific
501:   501	generalization capability. The IPG module dynamically generates
505:   505	As illustrated in Fig. 2, the IPG module consists of a transposed
508:   508	image features, the IPG module transforms specific instance-level
513:   513	Prompt Pool
544:   544	generalization performance. The process of the proposed IPG module can be formally expressed as follows:
555:   555	possesses an IPG module.
562:   562	task identifier, which guides the output of the IPG module.
656:   656	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
775:   775	Table 2: Ablation Study of Individual Components. The average incremental accuracy after each stage is reported.
782:   782	IPG
784:   784	Prompt Pool
813:   813	MLP in the IPG module are frozen after the training of the first
855:   855	Figure 4: Incremental average Rank-1 after each training
859:   859	experimental results demonstrate that our proposed PP-IPG method
862:   862	[27], WA [50], BiC [38], and an LReID method PTKP [8], our PPIPG method improves by 3.9% in 𝑠 R-1 and by 3.7% in 𝑠 mAP . These
865:   865	worth noting that our method outperforms ttqk by 6.5% in 𝑠 R-1
870:   870	outperforms other methods in the last three training stages.It is
874:   874	We compared our proposed PP-IPG architecture with several stateof-the-art methods on the task of LVI-ReID. The performance of
903:   903	Ablation study
905:   905	We conducted an ablation experiment to evaluate the effectiveness
906:   906	of the proposed components and their interactions. The performance results are presented in Table 2, where we report the incremental average precision performance of the model at each task
907:   907	stage. The following conclusions can be drawn from Table 2: Initially, with the support of the IPG module, Exp-2 shows a significant
915:   915	the proposed IPG module can improve the model’s ability to address
921:   921	the IPG module. However, as the task stages increase, the memory
925:   925	to the IPG module. The prompts in the prompt pool participate in
926:   926	the training of all task stages, while the IPG generates different
941:   941	select their corresponding keys. This fully demonstrates that our
957:   957	instance-level prompt generator can dynamically generate instancespecific prompts, thus breaking free from the limitation of a fixedsize prompt pool. These two modules are complementary and effectively work for robust feature representation learning. Ablation
964:   964	datasets illustrate the superiority of the proposed prompt learningbased PP-IPG method for the LVI-ReID task.
983:   983	Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator
1059:  1059	Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
1124:  1124	Instance-level Prompts for Rehearsal-free Continual Learning. In 2023 IEEE/CVF

exec
/bin/zsh -lc "pdftotext 'Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|MIE|HFA|DFC|t-SNE|attention|heatmap|Effectiveness|Table V|TABLE V|visual|Fig 3|Fig. 3\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
17:    17	Mask Information Embedding (MIE) module, which directs the
21:    21	a Hierarchical Feature Aggregation (HFA) module that mines
25:    25	(DFC) module, which is able to complete global features through
77:    77	Recently, benefiting from the multi-head self-attention
103:   103	extractor, a Mask Information Embeddings (MIE) module,
104:   104	a Hierarchical Feature Aggregation (HFA) module, and a
105:   105	Diverse Feature Completion (DFC) module. Firstly, we direct
107:   107	occlusion scenarios by designing the MIE module to encode
120:   120	Fig. 1. Visualization of images with different occlusion types. The first, second and third columns represent the occluded input images, attention heatmaps
127:   127	HFA module is to mine more exploitable high-quality detail
130:   130	we propose the DFC module that is able to compensate
144:   144	• We propose a mask information embeddings (MIE) module that encodes mask semantic information of body parts
148:   148	• We propose a hierarchical feature aggregation (HFA)
151:   151	• A diverse feature completion (DFC) module that can
159:   159	many NLP tasks. Inspired by the self-attention mechanism,
168:   168	image dimensions. Carion et al. [19] performed cross-attention
181:   181	methods based on attention [25], [26], [27], and (iii) methods
184:   184	field and single-head self-attention mechanism of CNN, they
217:   217	to obtain more comprehensive and discriminative attention
263:   263	we design a mask information embeddings (MIE) module
266:   266	then integrate E m and E p to direct the model’s attention
286:   286	(HFA) module to capture fine-grained local features. To further
289:   289	we design a diverse feature completion (DFC) module, which
291:   291	discriminating global features by learning visual cues from
302:   302	the visualization of attention maps of TransReID in Fig 1(b),
306:   306	Because of this problem, we design a MIE module that incorporates semantic information of body parts into the embedding
313:   313	This design constitutes the core principle of the MIE module,
334:   334	a mask information embeddings (MIE) module, a hierarchical feature aggregation (HFA) module and a diverse feature completion (DFC) module. After
335:   335	obtaining mask semantic information for body parts using the mask extractor, the MIE module encodes mask semantic information of body parts into
337:   337	directing the model’s attention towards non-occluded body parts. Then the HFA module and DFC module are used to obtain both fine-grained and robust
353:   353	the interference of occlusion and direct the model’s attention
372:   372	Through the MIE module, we direct the model to concentrate on the visible body parts of the individual, so mitigating
378:   378	In this paper, we design a HFA module which mines
380:   380	fine-grained local features. The core of the HFA module is
402:   402	convolutional-based HFA module to aggregate this diverse
407:   407	difficult for HFA to effectively model diverse detailed information in the presence of high similarity of inputs. Therefore,
412:   412	HFA. The final feature generation can be described as follows:
418:   418	N represents the number of patches, C represents the convolutional layers of the HFA module, and G is the binary gate
419:   419	that selects the input for HFA.
420:   420	To bolster the discriminative power of the HFA-generated
453:   453	of key features, thus affecting the accuracy of Re-ID. Therefore, to obtain discriminative global features, a DFC module
456:   456	network. Specifically, for each branch of the DFC module,
485:   485	improving the model performance. Through the ablation experiments in the fourth section, we find that the introduction
498:   498	DFC module, f t denotes the target feature vector, which is
737:   737	TABLE V
739:   739	TABLE VI
742:   742	THE O CCLUDED -D UKE DATASET. H ERE , MIE R EFERS TO THE M ASK
743:   743	I NFORMATION E MBEDDINGS M ODULE , HFA R EFERS TO THE H IER ARCHICAL F EATURE AGGREGATION M ODULE , DFC R EFERS TO
753:   753	D. Ablation Study
755:   755	out extensive ablation studies on the Occluded-Duke dataset.
757:   757	model performance and visualize the distribution of features
763:   763	1) Effectiveness of Each Component: The overall setup
765:   765	used or removed from MAHATMA. As depicted in Table V,
767:   767	(MIE) module, hierarchical feature aggregation (HFA) module, diverse feature completion (DFC) module, and the loss
768:   768	constraints specifically applied to HFA and DFC modules.
772:   772	baseline brings significant performance improvements. Specifically, with the incorporation of the MIE module, resulting in a
773:   773	3.5% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that MIE can avoid being affected
775:   775	attention towards non-occluded body parts. After adding the
776:   776	HFA module, resulting in a 5.6% enhancement in Rank-1
778:   778	HFA can correctly aggregate high-quality features at different
780:   780	features. With the introduce of the DFC module, resulting in a
781:   781	3.6% enhancement in Rank-1 accuracy and a 3.8% enhancement in mAP, indicating that the DFC can successfully learn
785:   785	be found that although DFC can capture rich global features,
787:   787	local fine-grained features. The combination of MIE and HFA
788:   788	models not only uses the MIE module to locate non-occluded
790:   790	the HFA module, thereby improving the model performance.
791:   791	Additionally, it becomes apparent that MIE, HFA, and DFC
800:   800	To further verify the effectiveness of the proposed components, we use t-SNE [53] to visualize the distribution of
802:   802	and the results are shown in Fig 3. The circles in the figure
805:   805	different identities. Specifically, Fig 3(a) shows the distribution
810:   810	identities. Fig 3(b) shows the distribution of features after the
811:   811	introduction of the MIE module. With the accurate positioning of non-occluded body parts, the boundaries inter-classes
812:   812	gradually become clear. Fig 3 (c) and Fig 3(d) show the distribution of features after adding the HFA and DFC modules,
817:   817	inter-classes are clearer. Finally, Fig 3(e) shows the feature
822:   822	2) Effectiveness of Mask Extractor Architecture: To verify
835:   835	Fig. 3. Visual analysis of feature distribution on the Occluded-Duke dataset. Circles represent sample features and different colors represent different identities.
836:   836	(a) The feature distribution of the baseline model, (b) the feature distribution of the baseline model after adding the MIE module, (c) the feature distribution of
837:   837	the baseline model after adding the MIE and HFA modules, (d) the feature distribution of the baseline model after adding the MIE, HFA and DFC modules,
840:   840	TABLE VII
841:   841	S TUDY A BOUT H OW M ANY B RANCHES A RE S UITABLE FOR DFC
852:   852	helping the MIE module to remove occlusion interference and
855:   855	3) Effectiveness of Loss Function Weights: As depicted in
863:   863	of ablation experiments to gauge the model’s efficiency under
865:   865	γ , we utilize the MIE module and the HFA module as the
870:   870	the value of γ is insufficiently small, HFA has not converged
877:   877	λ. As depicted in Fig 4(b), upon integrating the DFC module
881:   881	and Rank-1 are both optimal. When λ surpasses 1, the DFC
884:   884	we choose to set λ to 1, so that DFC successfully learns rich
886:   886	4) Effectiveness of How Many Branches are More Suitable
887:   887	for DFC: The proposed DFC module utilizes a multi-branch
895:   895	suitable for DFC. As depicted in Table VII, augmenting
896:   896	the number of DFC branches from 1 to 2, more embeddings are generated to mitigate the impact of occlusion
899:   899	exceeds 2, DFC generates too many redundant features, some
900:   900	of which may carry the influence of occlusion noise, leading to a drop in performance. Therefore, DFC with two
903:   903	2 branches for DFC.
923:   923	results are listed in detail in Table VIII. From the results,
946:   946	TABLE VIII
963:   963	Information Embeddings (MIE) module to alleviate the impact
966:   966	we propose a Hierarchical Feature Aggregation (HFA) module
970:   970	Feature Completion (DFC) module to help global features
1052:  1052	attention for person re-identification,” in Proc. IEEE/CVF Conf. Comput.
1058:  1058	via attribute-based shift attention,” IEEE Trans. Circuits Syst. Video
1134:  1134	[51] M. Jia, Y. Sun, Y. Zhai, X. Cheng, Y. Yang, and Y. Li, “Semi-attention
1140:  1140	[53] L. van der Maaten and G. E. Hinton, “Visualizing data using t-SNE,”
1172:  1172	visual tracking, and pattern recognition.

exec
/bin/zsh -lc "pdftotext 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|ablation|PCR|LAR|CAL|Feature similarity|GradCAM|t-SNE|effectiveness|Effectiveness|Table [0-9]|Figure 2|visual\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
13:    13	and robustness of cross-modal pedestrian matching. To solve the above problem, we propose the LocalAware Residual Attention Vision Transformer (LAReViT) to enhance the learning of fine-grained local and
15:    15	the Local-Aware Residual (LAR) Module, which uses a novel Local Residual Attention (LRA) mechanism, is
17:    17	fine-grained local information lost in lower-level visual features, the LRA in the LAR module adopts novel
18:    18	attention residual connections. Additionally, we propose a Positional Channel Reconstruction (PCR) Module
19:    19	that takes advantage of the local receptive field benefits of convolution. PCR reweights features within patches
21:    21	Finally, the novel Center Aggregation Loss (CAL) is designed to reduce modality discrepancies moderately
24:    24	is available at https://github.com/Hua-XC/LAReViT.
95:    95	feature collapse in deeper layers leading to the vanishing of low-level visual features [46]. To provide
97:    97	Figure 2 presents a visual representation of the feature similarity matrices for ViT and LocalAware Residual Attention Vision Transformer (LAReViT) across different modalities. The
107:   107	Fig. 1. Motivation: Analyze feature maps learned from GradCAM [42] algorithm. CNN-based networks often
112:   112	features. Therefore, exploring valuable local information on transformer is crucial, and our proposed LAReViT
113:   113	precisely captures effective cross-modal person feature. CNN, convolutional neural network; LAReViT, LocalAware Residual Attention Vision Transformer; ViT, vision transformer.
116:   116	The proposed LAReViT, through an innovative residual connection between attention layers,
126:   126	features. Consequently, the LAReViT is proposed to capture and preserve local low-level features,
131:   131	designed a Local-Aware Residual (LAR) Module with a Local Residual Attention (LRA)
133:   133	from shallow to deep layers. LAR allows the network to provide more low-level detailed information
139:   139	proposed a Positional Channel Reconstruction (PCR) Module that reweights each patch block
146:   146	Fig. 2. Feature similarity matrix across various layers for ViT and LAReViT under the visible and IR model.
151:   151	proposed PCR merely enhances the advantages of convolution used in patch embeddings within
154:   154	statistics. The PCR adopts a weighted combination of Average Pooling and Maximum Pooling
156:   156	suppressing less useful patch embedding. The novel PCR strengthens the learning of effective local
159:   159	In summary, by combining LAR and PCR, the proposed LAReViT demonstrates innovative
171:   171	(CAL). As shown in Figure 1(d), the proposed method captures the pedestrian body more accurately
175:   175	and a novel transformer-based method (LAReViT) is proposed with a CAL, which aims to
177:   177	— The LAR Module, which is based on a novel LRA mechanism, is proposed to enhance the
180:   180	— The PCR Module is designed to promote the network to focus on effective patches through
183:   183	— Our proposed LAReViT demonstrates superior performance compared to other state-of-theart methods in the VI-ReID task, as evidenced by extensive evaluations of the SYSU-MM01,
228:   228	to the ReID field. Subsequently, CSL [38] was designed to mine local visual information shared
246:   246	ViT as the feature extractor and selected visual cues to filter out identity-irrelevant information.
247:   247	The aforementioned work has proven the effectiveness of employing ViT in VI-ReID. However, they
253:   253	results in the vanishing of low-level visual features [6], and ViT also emphasizes the extraction of
255:   255	low-level visual features at the local level that arise from the emphasis on extracting global features
256:   256	in ViT. Moreover, we propose a novel LAR attention learning method to enhance the learning of
279:   279	Fig. 3. Overview of our LAReViT for VI-ReID. The PCR Module is designed to reweight the input patches
281:   281	LAR Module on the local-based input sequence to enhance the exploration and capture of low-level informative cues, aiming to alleviate the issue of over-globalization in transformer features. Finally, the CAL
292:   292	to CNN-Transformer hybrid models, our proposed LAReViT utilizes a pure ViT as its backbone,
293:   293	without incorporating additional CNNs. Though utilizing the proposed PCR and LAR modules,
297:   297	In this section, we present the technical details of the proposed LAReViT. The structure diagrams
298:   298	of the model and its components are shown in Figures 3 and 4. LAReViT encompasses three
300:   300	proposed PCR Module, described in Section 3.1, reweights features within patches at the channel
301:   301	level and reinforces the advantages of local receptive fields in convolution. Additionally, the LAR
304:   304	clues from shallow-layer networks. Finally, the CAL, introduced in Section 3.3, aims to moderate
313:   313	Fig. 4. Details of PCR. It reconstructs the weights of each patch through operations such as pooling and fully
319:   319	PCR Module
321:   321	Existing transformer models employ convolutional kernels to perceive the visual field within
326:   326	clues, and thus, the PCR is proposed to guide the network by effectively mining fine-grained local
327:   327	contextual features. Different from existing transformer models, PCR reweights each patch block in
329:   329	Consequently, the proposed PCR Module enables a more comprehensive extraction of cross-modal
339:   339	the patch embeddings and denote it as the input for the PCR.
371:   371	PCR guides Transformer models in handling local information more effectively in cross-modal
377:   377	LAR Module in LAReViT
387:   387	some shallow fine-grained visual feature information [46, 75]. In VI-ReID, shallow-layer low-level
392:   392	results in the over-globalization of ViT feature. In this context, we propose LAR as a solution to
394:   394	LAR aims to study local features, such as the head, trunk, and legs of individuals, which are
396:   396	(RD). Formally, by performing the PCR Module with the RD technique, we divide the global patch
407:   407	feature output by the PCR. The superscript head, trunk, leg represents the embedding feature for
410:   410	and positional embeddings to obtain the input for the subsequent LAReViT network.
411:   411	The architecture of LAReViT is illustrated in Figure 3. For global person data X, we utilized the
413:   413	X L , the LAR Module is proposed to learn person local feature. Regarding the LAR, we construct
465:   465	where i indicates the ith LAR and W is the weight matrix of query, key, and value to attention
473:   473	while maintaining similar computational costs. Finally, the proposed LAR Module, which is based
478:   478	CAL
487:   487	in utilizing abundant identity information thoroughly. Therefore, we propose the CAL, which
490:   490	Specifically, for the global and local output dataflow X and X L from the LAReViT, which share
498:   498	extracted by LAReViT, which we formulate as 𝑓𝑝 . For the embedding feature 𝑓𝑝 of LAReViT, we
525:   525	As shown on the left side of Figure 3, CAL aims to mitigate modal discrepancy and learn more
530:   530	between negative samples. This moderate constraint can guide the LAReViT to learn cross-modal
531:   531	features more thoroughly. Specifically, CAL can be formulated as follows:
568:   568	CAL facilitates the integration of different modalities and bridges the modality gap. It moderately
570:   570	alignment in LAReViT and extracting comprehensive person features.
575:   575	Finally, we incorporate commonly used ID loss, triplet loss, and the designed CAL to enhance
576:   576	the ability of LAReViT to extract discriminative features. The overall loss function is defined as
582:   582	where the parameter 𝜆 is set to 0.5 in order to balance the terms CAL. Moreover, Algorithm 1
590:   590	Algorithm 1: The Training Process of LAReViT
631:   631	4.3.1 Results on SYSU-MM01 Datasets. Based on the experimental results presented in Table 1,
632:   632	it can be observed that the proposed LAReViT method surpasses other state-of-the-art approaches
633:   633	on the SYSU-MM01 dataset. LAReViT achieves 76.71% Rank-1 and 72.95% mAP in the All-search
640:   640	Table 1. Comparison with the State-of-the-Arts on SYSU-MM01 Dataset
887:   887	LAReViT
915:   915	mode. In addition, in the Indoor-search mode, LAReViT achieves 84.22% Rank-1 and 86.26% mAP.
916:   916	By leveraging the strengths of both convolution kernel and transformer architectures, LAReViT
922:   922	the proposed LAReViT achieves a performance gain of 6.12% in Rank-1 accuracy and 5.46% in mAP.
923:   923	The advantage of LAReViT in extracting global features contributes to its superior performance
933:   933	in terms of Rank-n accuracy and mAP. These analyses evidence the effectiveness of our proposed
940:   940	Table 2. Comparison with the State-of-the-Arts on RegDB Dataset
1102:  1102	LAReViT
1117:  1117	4.3.2 Results on RegDB Datasets. According to Table 2, LAReViT also outperforms the latest
1118:  1118	research on the RegDB dataset. In the V-I mode, LAReViT achieves a Rank-1 accuracy of 90.4% and
1119:  1119	an mAP of 84.7%. In the I-V mode, LAReViT achieves a Rank-1 accuracy of 90.5% and an mAP of
1123:  1123	proposed LAReViT in the V-I search mode. However, in the I-V search mode, their Rank-1 and
1124:  1124	mAP are more than 1% lower than the LAReViT. We attribute this to the imbalance in expressing
1126:  1126	not only superior but also very close in both search modes. This further highlights that LAReViT
1129:  1129	4.3.3 Results on LLCM Datasets. Finally, Table 3 demonstrates that LAReViT also performs exceptionally well on the LLCM dataset, showcasing its superior performance even under challenging
1138:  1138	Table 3. Comparison with the State-of-the-Arts on LLCM Dataset
1204:  1204	LAReViT
1221:  1221	Ablation Study and Analysis
1223:  1223	In this section, we conducted ablation studies to evaluate the individual contributions of each
1224:  1224	component within our proposed LAReViT framework. All experiments were conducted on the SYSUMM01 dataset, utilizing the same baseline approach under the full search mode. Hyperparameters
1227:  1227	4.4.1 Effectiveness of Each Component. Here, we analyze the key components of LAReViT, including the PCR Module, LAR Module, and CAL. For a fair comparison, we used a pure transformerbased model as the baseline (Base) for all experiments. As summarized in Table 4, each component
1228:  1228	contributes to performance improvement. Starting from the baseline, the addition of PCR enhances performance, indicating that PCR effectively enhances the utilization of local information
1231:  1231	transformer layers. Furthermore, with the addition of LAR, we observe a significant performance
1232:  1232	improvement, confirming that LAR effectively mines local fine-grained features and fully utilizes
1234:  1234	and feature collapse in transformer. Finally, the experiments demonstrate that CAL enables the
1237:  1237	these components enhances the performance of LAReViT.
1238:  1238	4.4.2 Parameters and Computational Cost Analysis. In Table 5, we compared the computational
1239:  1239	cost of the proposed LAReViT with existing methods on the SYSU-MM01 dataset. As shown
1240:  1240	in Table 5, our method requires a total of 85.72M parameters and 64.88G Floating Point of
1244:  1244	DEEN. During the testing phase, our LAReViT utilizes only global features for retrieval. At this
1252:  1252	Table 4. Ablation Study of PCR, LAR, CAL on the All-Search Mode of SYSU-MM01
1265:  1265	PCR
1267:  1267	LAR
1270:  1270	CAL
1332:  1332	Table 5. Computational Consumption Comparison on SYSU-MM01 Dataset
1340:  1340	+PCR
1341:  1341	+LAR
1347:  1347	LAReViT Transformer
1379:  1379	proposed PCR and LAR achieve significant performance improvements with minimal additional
1380:  1380	parameter overhead. The experimental results indicate that our LAReViT maintains a moderate
1381:  1381	computational cost compared to existing methods. The superiority of our LAReViT can be attributed
1384:  1384	4.4.3 Effectiveness of RD of Local Features. Firstly, our network adopts the input of locally
1385:  1385	partitioned features using the local feature action network. To demonstrate the effectiveness of
1386:  1386	the RD of individual head, trunk, and leg areas proposed in LAR, we compare it with a similar
1402:  1402	Fig. 5. The accuracy obtained with different numbers of the divided blocks and RD on LAReViT.
1409:  1409	trunk, and leg parts, resulting in the highest accuracy. Therefore, for the LAR module, the RD
1410:  1410	approach we adopt for enhancing local features helps LAReViT better learn local information,
1412:  1412	in local features. The above analysis demonstrates the effectiveness of our RD approach.
1413:  1413	4.4.4 The Impact of Attention with Novel Residual Connection. To validate the effectiveness of
1417:  1417	which enhances the visual feature diversity and model robustness. The proposed network captures
1424:  1424	4.4.5 Analysis of Parameter 𝛼 in PCR. We evaluate the influence of the parameter 𝛼 in Equation (3)
1427:  1427	accuracy for the proposed PCR. The PCR guide model emphasizes patches that contain key clues,
1434:  1434	Fig. 7. The accuracy obtained with different parameters 𝛼 of PCR.
1436:  1436	Fig. 8. The accuracy obtained with different parameters 𝜆 of CAL.
1440:  1440	4.4.6 Analysis of Parameter 𝜆 of Loss Function. The Rank-1 accuracy and mAP results of LAReViT
1442:  1442	to 0.5, indicating that appropriately adjusting the degree of constraint on CAL contributes to better
1443:  1443	model fitting and higher accuracy. CAL achieves a balance between intra-class and inter-class
1450:  1450	To analyze the visual effectiveness of our proposed model, we present several representative visual
1453:  1453	generated by the CNN-based model displayed on the left side, the heatmaps generated by LAReViT
1455:  1455	of the baseline model in the middle, the heatmaps generated by LAReViT focus on more fine
1464:  1464	result. Left: the CNN-based model retrieval result, middle: the baseline retrieval result, and right: the LAReViT
1467:  1467	Fig. 10. The t-SNE visualization result of CNN-based, baseline, and LAReViT.
1469:  1469	indicates that LAReViT is designed to capture more comprehensive person features and addresses
1471:  1471	result, demonstrating that LAReViT learns highly discriminative identity features among different
1473:  1473	As illustrated in Figure 10, the t-distributed Stochastic Neighbor Embedding (t-SNE) visualization of the identity embeddings learned by the CNN-based model reveals that features
1476:  1476	innovatively designed LAReViT model achieves better constraint effects, enabling each identity to
1478:  1478	Finally, in Figure 11, we visualize the intra-class and inter-class identity feature distances.
1480:  1480	that LAReViT successfully reduces modality differences and extracts more discriminative identity
1482:  1482	indicates that our LAReViT achieves a more balanced optimization of feature distances. Moreover,
1483:  1483	although the inter-class distance decreases, the increased discrepancy between intra-class and interclass distances suggests that the features extracted by LAReViT possess stronger discrimination.
1490:  1490	Fig. 11. The feature distances of intra-class and inter-class visualization.
1496:  1496	In this article, we propose a novel transformer-based framework called LAReViT for VI-ReID.
1499:  1499	PCR and LAR module effectively capture and explore low-level fine-grained cues, thereby fully
1501:  1501	worth noting that our PCR and LAR can be integrated with existing VI-ReID models or even other
1503:  1503	validate the outstanding performance of LAReViT and the efficacy of its components. In the future,
1630:  1630	Local features coupling global representations for visual recognition. In Proceedings of the IEEE/CVF International
1670:  1670	[53] Bin Yang, Jun Chen, and Mang Ye. 2023. Top-K visual tokens transformer: Selecting tokens for visible-infrared person

codex
我已经定位到关键消融。下面补几张表附近的文字和数值，不是为了报榜单，而是看作者怎样用消融证明他们的“问题重定义”而不只是堆模块。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '1000,1085p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1000	
  1001	4.3. Ablation experiment
  1002	We designed a series of experiments to evaluate the eﬀectiveness of
  1003	both of the DyAdap module and AdRet strategy. Besides, we also design
  1004	experiments to explore the most suitable hyperparameters within our
  1005	model. The resnet-50 model trained with ID loss and triplet loss are used
  1006	as baseline model.
  1007	4.3.1. Eﬀectiveness of designed components
  1008	To evaluate the eﬀectiveness of our proposed DyAdap module along
  1009	with AdRet strategy, we gradually insert them into the baseline model
  1010	and compare the performance improvement. From the results in Table 4,
  1011	it can be seen that these components respectively brings 15.2% and 4.9%
  1012	𝑚𝐴𝑃 improvement to the Baseline model on average, which denotes
  1013	their eﬀectiveness. Besides, we could ﬁnd that simultaneously inserted
  1014	these modules into baseline model further brings 34.7% 𝑚𝐴𝑃 improvement, which denotes their compatibility.
  1015	4.3.2. Eﬀectiveness of BNF strategy
  1016	To evaluate the performance of our proposed BNF strategy in addressing catastrophic forgetting, we integrated the Batch Normalization
  1017	9
  1018	
  1019	Neural Networks 198 (2026) 108566
  1020	
  1021	Z. Chen et al.
  1022	
  1023	Fig. 6. Overﬁtting prevention analysis of the proposed DKAR model.
  1024	Table 4
  1025	The eﬀectiveness of our designed DyAdap module, AdRet strategy, and BNF strategy.
  1026	VIPeR
  1027	
  1028	DyAdap
  1029	
  1030	AdRet
  1031	
  1032	BNF
  1033	
  1034	𝑅-1
  1035	Baseline
  1036	Base+DyAdap
  1037	Base+AdRet
  1038	Base+DKAR
  1039	Base+BNF
  1040	
  1041	×
  1042	✓
  1043	×
  1044	✓
  1045	×
  1046	
  1047	×
  1048	×
  1049	✓
  1050	✓
  1051	×
  1052	
  1053	×
  1054	×
  1055	×
  1056	×
  1057	✓
  1058	
  1059	9.5
  1060	35.0
  1061	11.4
  1062	75.5
  1063	10.2
  1064	
  1065	Base+DKAR+BNF
  1066	
  1067	✓
  1068	
  1069	✓
  1070	
  1071	✓
  1072	
  1073	76.0
  1074	
  1075	Market1501
  1076	
  1077	CUHK-SYSU
  1078	
  1079	MSMT17
  1080	
  1081	Average
  1082	
  1083	𝑚𝐴𝑃
  1084	
  1085	𝑅-1

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf' - | nl -ba | sed -n '770,945p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   770	38.3
   771	36.3
   772	36.2
   773	42.8 42.0
   774	
   775	Table 2: Ablation Study of Individual Components. The average incremental accuracy after each stage is reported.
   776	Experiments
   777	1
   778	2
   779	3
   780	4
   781	
   782	IPG
   783	
   784	Prompt Pool
   785	
   786	!
   787	!
   788	
   789	!
   790	!
   791	
   792	stage1
   793	𝑠 R-1 𝑠 mAP
   794	70.6
   795	70.2
   796	72.7
   797	71.4
   798	72.0
   799	70.7
   800	75.0 73.1
   801	
   802	training iteration, 6 samples of different identities are randomly
   803	selected for the current task and past tasks. Each identity contains
   804	3 visible-light images and 3 infrared images. When constructing
   805	the memory bank, we adapted the strategy proposed by PTKP [10]
   806	to meet the requirements of cross-modality research. We randomly
   807	select one-third of the identity categories from the samples of the
   808	original task. For each category, two category centers are retained
   809	(one for each modality), and the 3 samples farthest from the category
   810	center are saved. The size of the prompt pool is set to 8, and the
   811	length of each prompt is set to 20. In each training iteration, the
   812	top two most relevant prompts are selected. The weights of the
   813	MLP in the IPG module are frozen after the training of the first
   814	dataset is completed, and an instance-level prompt with a length of
   815	20 is generated for each instance. We use AdamW as the optimizer
   816	and combine it with a cosine annealing learning rate scheduler for
   817	training. For the first task, the initial learning rate is set to 5 × 10 −4 ,
   818	the weight decay is set to 1 × 10 −4 , and the training lasts for 20
   819	epochs. The initial learning rate for the subsequent two tasks is set
   820	to 5 × 10 −5 , and the training lasts 10 epochs. For the final task, the
   821	initial learning rate returns to 5 × 10 −4 , and the total training lasts
   822	for 30 epochs. The temperature parameter is set to 2.
   823	
   824	4.3
   825	
   826	stage2
   827	𝑠 R-1 𝑠 mAP
   828	47.9
   829	50.2
   830	51.2
   831	52.0
   832	53.6 54.8
   833	51.8
   834	54.0
   835	
   836	stage3
   837	𝑠 R-1 𝑠 mAP
   838	39.3
   839	42.9
   840	44.0
   841	47.3
   842	46.6 49.7
   843	46.5 49.8
   844	
   845	stage4
   846	𝑠 R-1 𝑠 mAP
   847	34.3
   848	34.8
   849	39.9
   850	39.6
   851	41.6
   852	40.9
   853	42.8 42.0
   854	
   855	Figure 4: Incremental average Rank-1 after each training
   856	stage (left).Incremental average mAP after each training
   857	stage (right).
   858	
   859	experimental results demonstrate that our proposed PP-IPG method
   860	achieves the best performance on the LVI-ReID task. Compared
   861	with other lifelong learning methods, such as LwF [17], iCaRL
   862	[27], WA [50], BiC [38], and an LReID method PTKP [8], our PPIPG method improves by 3.9% in 𝑠 R-1 and by 3.7% in 𝑠 mAP . These
   863	lifelong learning methods are specifically designed for the closedset problem of image classification, and this characteristic leads
   864	to their insufficient adaptability when facing open-set tasks. It is
   865	worth noting that our method outperforms ttqk by 6.5% in 𝑠 R-1
   866	and by 5.8% in 𝑠 mAP . This may be because the prompts enhance
   867	our model’s ability to extract knowledge. Our method has demonstrated the most excellent performance in four different datasets,
   868	indicating that it has the best ability to cope with the challenges
   869	of complex past knowledge memory and the processing requirements of new tasks. Moreover, as shown in Fig. 4, our method
   870	outperforms other methods in the last three training stages.It is
   871	
   872	Comparison with state-of-the-art methods
   873	
   874	We compared our proposed PP-IPG architecture with several stateof-the-art methods on the task of LVI-ReID. The performance of
   875	comparison results are shown in Table 1. "Joint-Train" represents
   876	joint training on all datasets, while "FineTune" refers to sequential
   877	training without specific measures. To make a fair comparison,
   878	all methods were strictly adapted for this task based on their released codes, meanwhile, the backbone network, preprocessing
   879	methods, and training techniques of all methods are the same. The
   880	
   881	959
   882	
   883	ICMR ’25, June 30-July 3, 2025, Chicago, IL, USA
   884	
   885	Zhenxi Luo, Guoqiang Xiao, Michael S. Lew, Song Wu
   886	
   887	noteworthy that our model introduces only 0.45M additional trainable parameters beyond the backbone network, benefiting from the
   888	inherent parameter efficiency of the prompt learning mechanism,
   889	while maintaining task-agnostic parameter scalability. Compared
   890	with the current state-of-the-art LVI-ReID model TTQK [48], which
   891	requires 𝑡 · 7M additional parameters (where 𝑡 denotes the number
   892	of tasks), our approach achieves a 91.4% reduction in parameters.
   893	Notably, this parameter efficiency advantage expands as the task
   894	scale t increases. Such task-independent parameter characteristics
   895	endow our model with superior scalability and enhanced practical
   896	deployment potential.
   897	
   898	4.4
   899	
   900	Figure 5: Histogram of the selection frequencies of keys corresponding to the classifier. Selection of different keys before training (left). Selection of different keys after training
   901	(right). The results are obtained from the test set.
   902	
   903	Ablation study
   904	
   905	We conducted an ablation experiment to evaluate the effectiveness
   906	of the proposed components and their interactions. The performance results are presented in Table 2, where we report the incremental average precision performance of the model at each task
   907	stage. The following conclusions can be drawn from Table 2: Initially, with the support of the IPG module, Exp-2 shows a significant
   908	improvement in both the memory ability and feature extraction
   909	ability of the model compared to Exp-1. Although the performance
   910	gap between Exp-2 and Exp-1 in the first task stage is not large,
   911	with a difference of 2.1% in 𝑠 R-1 and 1.2% in 𝑠 mAP , as the data is
   912	continuously fed into the model, the gap in their memory abilities
   913	gradually becomes apparent. In the fourth task stage, the difference
   914	in 𝑠 R-1 reaches 5.6% and in𝑠 mAP reaches 4.8%. This indicates that
   915	the proposed IPG module can improve the model’s ability to address
   916	the catastrophic forgetting problem without introducing a large
   917	number of additional parameters. Compared to Exp-2, Exp-3 only
   918	uses the prompt pool module. In the first task stage, 𝑠 R-1 decreases
   919	by 0.7% and 𝑠 mAP decreases by 0.7%. This suggests that the feature
   920	extraction ability of the prompt pool is slightly inferior to that of
   921	the IPG module. However, as the task stages increase, the memory
   922	ability of the prompt pool gradually emerges, achieving the best
   923	performance in the second task stage. We analyze that this may be
   924	attributed to the better continuity of the prompt pool compared
   925	to the IPG module. The prompts in the prompt pool participate in
   926	the training of all task stages, while the IPG generates different
   927	prompts independently in each attention layer. After combining
   928	these two modules, the best performance is achieved in Exp-4. Compared to Exp-3, it shows an improvement of 1.2% in 𝑠 R-1 and 1.1% in
   929	𝑠 mAP . The results in the first task stage indicate its optimal feature
   930	learning ability and the performance in the subsequent task stages
   931	shows that its memory ability is further enhanced.
   932	
   933	4.5
   934	
   935	the parameters of the keys are randomly initialized at the beginning,
   936	and there is no established connection between the samples from
   937	various domains and the keys. This affects the performance of the
   938	model in selecting the classifier and calculating the task identifier
   939	during the inference phase. However, the histogram of selection
   940	frequencies shown in the right-hand plot becomes significantly regular after training. Samples from different domains can adaptively
   941	select their corresponding keys. This fully demonstrates that our
   942	designed query-key mechanism enables the model to adaptively
   943	recognize the tasks to which the input data belong, thus effectively
   944	preserving knowledge across different domains.
   945	

exec
/bin/zsh -lc "pdftotext 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '1295,1335p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1295	variations can be observed. Despite these dataset-speciﬁc ﬂuctuations,
  1296	these results conﬁrm that our method robustly mitigates catastrophic
  1297	forgetting across various training orders.
  1298	
  1299	4.3.3. Eﬀectiveness of knowledge distillation losses in AdRet strategy
  1300	Within the AdRet strategy, there are two loss functions used for
  1301	knowledge distillation: one applied to the raw features extracted by the
  1302	.𝐸 , and the other applied to
  1303	feature extraction network, denoted as 𝐿𝐹𝐾𝐷
  1304	the generated dynamic parameters within the dynamic convolution network, denoted as 𝐿𝑟𝑒𝑔 . To assess the eﬀectiveness of these two loss func.𝐸
  1305	tions in addressing catastrophic forgetting, we separately apply 𝐿𝐹𝐾𝐷
  1306	10
  1307	
  1308	Neural Networks 198 (2026) 108566
  1309	
  1310	Z. Chen et al.
  1311	
  1312	Table 6
  1313	The inﬂuence of diﬀerent hyperparameter settings on the performance of our proposed framework.
  1314	Ratio
  1315	1:1
  1316	2:1
  1317	1:2
  1318	5:1
  1319	1:5
  1320	
  1321	VIPeR
  1322	
  1323	Market1501
  1324	
  1325	CUHK-SYSU
  1326	
  1327	MSMT17
  1328	
  1329	𝑅−1
  1330	
  1331	𝑚𝐴𝑃
  1332	
  1333	𝑅−1
  1334	
  1335	𝑚𝐴𝑃

exec
/bin/zsh -lc "pdftotext 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1221,1295p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1221	Ablation Study and Analysis
  1222	
  1223	In this section, we conducted ablation studies to evaluate the individual contributions of each
  1224	component within our proposed LAReViT framework. All experiments were conducted on the SYSUMM01 dataset, utilizing the same baseline approach under the full search mode. Hyperparameters
  1225	were meticulously fine-tuned for each experiment. The outcomes of these experiments are elaborated
  1226	upon as follows.
  1227	4.4.1 Effectiveness of Each Component. Here, we analyze the key components of LAReViT, including the PCR Module, LAR Module, and CAL. For a fair comparison, we used a pure transformerbased model as the baseline (Base) for all experiments. As summarized in Table 4, each component
  1228	contributes to performance improvement. Starting from the baseline, the addition of PCR enhances performance, indicating that PCR effectively enhances the utilization of local information
  1229	within patches at each position in the transformer. This improvement can be attributed to the
  1230	re-weighting of channel importance, which emphasizes discriminative features for subsequent
  1231	transformer layers. Furthermore, with the addition of LAR, we observe a significant performance
  1232	improvement, confirming that LAR effectively mines local fine-grained features and fully utilizes
  1233	low-level semantic clues of shallow networks. It addresses the challenges of excessive globalization
  1234	and feature collapse in transformer. Finally, the experiments demonstrate that CAL enables the
  1235	network to extract more discriminative features of individuals and moderately reduces modality
  1236	disparities, resulting in a more stable distribution of features. In conclusion, the combination of
  1237	these components enhances the performance of LAReViT.
  1238	4.4.2 Parameters and Computational Cost Analysis. In Table 5, we compared the computational
  1239	cost of the proposed LAReViT with existing methods on the SYSU-MM01 dataset. As shown
  1240	in Table 5, our method requires a total of 85.72M parameters and 64.88G Floating Point of
  1241	Operations (FLOPs). Compared to CNN-based methods, our model, which is based on the ViT,
  1242	exhibits higher computational consumption and parameter requirements. However, our model
  1243	demonstrates faster convergence, resulting in significantly reduced training times compared to
  1244	DEEN. During the testing phase, our LAReViT utilizes only global features for retrieval. At this
  1245	stage, its computational cost is comparable to that of BASE and lower than that of CNN-based
  1246	state-of-the-art. Furthermore, compared to other transformer-based approaches such as PMT, the
  1247	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 5, Article 146. Publication date: May 2025.
  1248	
  1249	146:16
  1250	
  1251	X. Hua et al.
  1252	Table 4. Ablation Study of PCR, LAR, CAL on the All-Search Mode of SYSU-MM01
  1253	Dataset
  1254	
  1255	Method
  1256	BASE
  1257	3
  1258	3
  1259	3
  1260	3
  1261	3
  1262	3
  1263	3
  1264	
  1265	PCR
  1266	
  1267	LAR
  1268	
  1269	SYSU-MM01
  1270	CAL
  1271	
  1272	3
  1273	3
  1274	3
  1275	3
  1276	3
  1277	
  1278	3
  1279	3
  1280	3
  1281	
  1282	3
  1283	3
  1284	
  1285	Rank-1
  1286	62.2
  1287	64.0
  1288	72.2
  1289	64.6
  1290	73.7
  1291	75.1
  1292	76.7
  1293	
  1294	Rank-10
  1295	91.3

codex
LAReViT 的消融表刚好截在 mAP 前，我补最后几行。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1282,1325p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1282	3
  1283	3
  1284	
  1285	Rank-1
  1286	62.2
  1287	64.0
  1288	72.2
  1289	64.6
  1290	73.7
  1291	75.1
  1292	76.7
  1293	
  1294	Rank-10
  1295	91.3
  1296	98.5
  1297	95.4
  1298	92.7
  1299	97.3
  1300	96.8
  1301	97.3
  1302	
  1303	Rank-20
  1304	95.9
  1305	98.8
  1306	98.2
  1307	96.6
  1308	98.9
  1309	99.0
  1310	99.0
  1311	
  1312	mAP
  1313	57.5
  1314	61.5
  1315	69.0
  1316	60.6
  1317	69.4
  1318	71.8
  1319	72.9
  1320	
  1321	mINP
  1322	42.3
  1323	47.9
  1324	56.1
  1325	46.0

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf' - | nl -ba | sed -n '1080,1165p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1080	
  1081	Average
  1082	
  1083	𝑚𝐴𝑃
  1084	
  1085	𝑅-1
  1086	
  1087	𝑚𝐴𝑃
  1088	
  1089	𝑅-1
  1090	
  1091	𝑚𝐴𝑃
  1092	
  1093	𝑅-1
  1094	
  1095	𝑚𝐴𝑃
  1096	
  1097	𝑅-1
  1098	
  1099	𝑚𝐴𝑃
  1100	
  1101	16.3
  1102	46.0
  1103	19.9
  1104	81.5
  1105	17.5
  1106	
  1107	19.7
  1108	49.5
  1109	39.2
  1110	71.2
  1111	20.6
  1112	
  1113	5.9
  1114	25.0
  1115	15.5
  1116	48.3
  1117	7.7
  1118	
  1119	57.2
  1120	57.6
  1121	64.3
  1122	75.5
  1123	57.5
  1124	
  1125	50.0
  1126	52.6
  1127	58.0
  1128	70.7
  1129	49.6
  1130	
  1131	13.9
  1132	32.6
  1133	8.2
  1134	33.6
  1135	11.0
  1136	
  1137	3.6
  1138	13.2
  1139	2.0
  1140	14.1
  1141	3.7
  1142	
  1143	25.1
  1144	43.7
  1145	30.8
  1146	64.0
  1147	24.8
  1148	
  1149	19.0
  1150	34.2
  1151	23.9
  1152	53.7
  1153	19.6
  1154	
  1155	82.4
  1156	
  1157	72.8
  1158	
  1159	51.2
  1160	
  1161	76.8
  1162	
  1163	72.3
  1164	
  1165	37.3

codex
还需要第三篇 DMM 和 CDCN 的消融证据，我单独定位它的 ablation 段。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | rg -n \"Ablation|ablation|DMM|CDCN|Diverse|Domain-Modality|Cross-domain Cognitive|Effectiveness|Table|Fig|forgetting|adaptability|anti-forgetting\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
57:    57	challenges of cross-modality and cross-domain variations. Thus, in this paper, we proposed a Domain-ModalityMix (DMM) based replay samples reconstruction strategy and Cross-domain Cognitive Network (CDCN) to
60:    60	diverse replay samples, we enhance the anti-forgetting ability of our model by the Domain-Modality-Mix
62:    62	alleviating catastrophic forgetting induced by modality and domain variations. Finally, guided by the Chunking
63:    63	Theory in cognitive psychology, we designed a Cross-domain Cognitive Network, which incorporates a
66:    66	demonstrate that our proposed method has remarkable adaptability and robust resistance to forgetting and
68:    68	The source code of our designed method is at https://github.com/SWU-CS-MediaLab/DMM-CDCN.
82:    82	in Figs. 1(c) and 1(d) under the joint learning scenario, if every new
85:    85	and memory resource expenditures. Moreover, as depicted in Fig.
86:    86	1(b), the model attains heightened generalization capability and adaptability through the continuous accrual of knowledge. Consequently,
89:    89	To achieve the aforementioned objective, as depicted in Fig. 1(a),
94:    94	key characteristics as domain adaptability and anti-forgetting ability.
109:   109	address in its design. Secondly, as illustrated in Fig. 1(f), addressing the
110:   110	dual deterioration in adaptability and anti-forgetting ability caused by
126:   126	Memory Banks, we introduce the Domain-Modality-Mix (DMM) based
130:   130	modalities. Operating the Domain-Modality-Mix strategy within a fixed
132:   132	effectively mitigating catastrophic forgetting caused by cross-domain
137:   137	(GCNs) (Jiang et al., 2019), we designed a novel Cross-domain Cognitive Network (CDCN) to promote the extraction of cross-modal invariant features. By implementing a camera-aware, dynamically expanding
143:   143	• Replay Samples Domain-Modality-Mix Reconstruction: During the rehearsal phase in LVI-ReID, we innovatively designed a
144:   144	Domain-Modality-Mix (DMM) based replay sample reconstruction
145:   145	strategy. The DMM can generate diversified cross-domain and
147:   147	as a countermeasure against catastrophic forgetting in LVI-ReID.
148:   148	• Cross-domain Cognitive Network: To enhance the domain adaptability of the learned LVI-ReID model for the continuous crossdomain data streams, we designed a Cross-domain Cognitive Network (CDCN) framework, which is based on a camera-extended
154:   154	that our proposed DMM strategy and CDCN framework have remarkable domain adaptability and robust resistance to forgetting
158:   158	Fig. 1. Motivation of Our Work. To mitigate the substantial computational cost in
162:   162	adaptability (1(b)). During the process of continual learning, we need to simultaneously
163:   163	manage the decline in model adaptability and catastrophic forgetting caused by modal
167:   167	adaptability and anti-forgetting ability due to significant cross-modal
168:   168	discrepancies (as shown in Fig. 1(e)), which TTQK did not specifically
201:   201	data domains. In contrast, our DMM fusion strategy has been specifically designed to handle data augmentation tasks that involve multiple
228:   228	catastrophic forgetting and assimilate new knowledge in continuous
237:   237	combating catastrophic forgetting, and they have not fully exploited
256:   256	we denote as adaptability and anti-forgetting ability. The field’s strategies can be broadly grouped into three categories: By employing an
264:   264	catastrophic forgetting effectively. Some methods (Serra et al., 2018;
310:   310	Fig. 2. The overview of our proposed methods. In the LVI-ReID task, a training stage comprises two phases, an adaptation phase (in red) and a rehearsal phase (in blue).
312:   312	camera-aware Cross-domain Cognitive Network, tasked with accumulating knowledge, ensuring intra-modal consistency, and mining cross-modal similarities. The adaptability of
317:   317	forgetting. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
329:   329	will evaluate the adaptability and anti-forgetting ability of the trained
336:   336	Cross-Domain Same-Modality Mixing Samples, (4) Cross-Domain CrossModality Mixing Samples. Through channel-level Domain-ModalityMixing, the diversity of replay samples is significantly augmented,
337:   337	thereby profoundly mitigating catastrophic forgetting issues induced
338:   338	by both domain transfer and cross-modal disparity. As a result, this approach substantially enhances the model’s anti-forgetting ability. Then,
343:   343	forgetting effectively.
349:   349	As depicted in Fig. 2 , our method consists of two main processes
357:   357	Subsequently, we propose leveraging a Cross-domain Cognitive Network (CDCN) to facilitate model learning of intra-modal consistency,
364:   364	adaptability of the learned model. During the rehearsal phase, we
366:   366	the Domain-Modality-Mixing (DMM) based replay samples strategy,
368:   368	of replay samples. As shown in Fig. 3, the strategy of DMM commences
372:   372	All-channel Mixing. Following the selection of channel mixing strategies, DMM proceeds with a certain probability to implement any of the
386:   386	𝑎𝑑𝑎𝑝𝑡 and optimize the adaptability of the model:
432:   432	et al., 2019), and CA (Ye et al., 2021). Under the constraint of maintaining the same memory size, we propose the Domain-Modality-Mix
433:   433	(DMM) based replay samples reconstruction strategy to achieve the
435:   435	thereby effectively mitigating the issue of catastrophic forgetting in the
436:   436	learned model. Our DMM strategy involves two main selection strategies: Channel Mixing Strategies and Sample Mixing Strategies. The
450:   450	Fig. 3. The Replay Samples Domain-Modal-Mix Strategy. The original replay images
503:   503	After selecting channel mixing strategies, DMM will adopt sample
557:   557	bolster the baseline model’s resilience against forgetting:
581:   581	In this section, we will introduce the details of the proposed Replay Samples Domain-Modality-Mix Reconstruction. Given that VI-ReID
584:   584	samples, exacerbating catastrophic forgetting. By contrast, a diverse
645:   645	𝑟 belonging to 𝑡th domain and 𝑚 modality. DMM first divide it into 𝑘 blocks vertically 𝑥𝑡,𝑚
647:   647	𝑘 . Subsequently, DMM proceeds to perform domain mix with
653:   653	𝑥𝑟,𝑗 . The algorithmic process of DMM is illustrated in Algorithm 1.
667:   667	Algorithm 1: Replay Samples Domain-Modality-Mix Reconstruction
743:   743	The primary goal of the CDCN is to learn the intra-modal similarities
776:   776	In this section, we will introduce the details of the proposed Crossdomain Cognitive Network (CDCN). In the LVI-ReID task, alleviating the performance degradation caused by the modality semantic
795:   795	shared GCN Layer) for each stage. The proposed CDCN framework
851:   851	3.5.5. The inference of CDCN
852:   852	This section details the distinctions between the training and inference phases of the CDCN framework. The inference phase is distinctive
854:   854	for predictions. By the way, during the rehearsal phase, CDCN does
944:   944	To evaluate the adaptability and resistance to forgetting the proposed model, we devise a continual learning data stream scheme
972:   972	3.5.4. The optimization of CDCN
974:   974	𝑉 𝐹 of CDCN outputs, employing the knowledge representation vector
976:   976	we input the feature vectors 𝑉 𝑂𝐹 sequentially into the CDCN’s shared
978:   978	confidence scores, and optimize CDCN using cross-entropy loss:
1035:  1035	Table 1
1041:  1041	the model’s adaptability, we proceed by resuming directly from the model trained in the first stage using the Baseline, thereby rendering the
1047:  1047	Ablation
1241:  1241	Base. + CDCN
1242:  1242	Base. + DMM
1243:  1243	Base. + CDCN + DMM
1295:  1295	Table 2
1296:  1296	Anti-forgetting performance under VIS (query) to IR (gallery) test mode at the last training stage (Red denotes the Top-1, while Green represents
1300:  1300	same as Table 1, ‘‘Single-Learning’’ means training one dataset only using a single model, ‘‘Joint-Learning’’ represents training all datasets by
1306:  1306	Ablation
1465:  1465	Base. + CDCN
1466:  1466	Base. + DMM
1467:  1467	Base. + CDCN + DMM
1548:  1548	Fig. 4. Performances in adaptability 4(a) and anti-forgetting ability 4(b) under both VIS to IR and IR to VIS test modes. Our method achieves optimal stage-wise adaptation and
1551:  1551	Adaptation Performance. Our method (Base. + CDCN + DMM)
1553:  1553	of adaptability (in Table 1). Specifically, under the VIS to IR test
1557:  1557	5.37 mAP. Under the IR to VIS test mode (Table 3), our method
1563:  1563	Anti-forgetting Performance. With regard to anti-forgetting ability, our method (Base. + CDCN + DMM) also demonstrates remarkable improvements (in Table 2). Under the VIS to IR test mode, our
1570:  1570	mode (in Table 3). This observed superiority of our method likely
1571:  1571	arises from addressing the issue of catastrophic forgetting induced by
1574:  1574	Stage-wise Performance. As depicted in Fig. 4, we evaluated the
1575:  1575	𝑠̄ and 𝑚̄ metrics for each model at every stage. Regarding adaptability,
1577:  1577	respect to anti-forgetting capability, while our method did not yield the
1582:  1582	substantially boosts its anti-forgetting ability as training progresses, and
1593:  1593	the memory bank will permanently retain stored samples. In our DMM
1601:  1601	the comprehensive adaptability of models. For anti-forgetting performance, we employ 𝑚̄ (weighted average performance on previously
1602:  1602	seen domains before the current training domain) to measure the antiforgetting ability of models. Both 𝑠̄ and 𝑚̄ are measured with Rank-1
1610:  1610	et al., 2022), and KRKC (Yu et al., 2023). The adaptation and antiforgetting performances of each method are presented in Table 1 and
1611:  1611	Table 2, respectively. The term ‘‘Single-Learning’’ denotes only training
1624:  1624	Table 3
1625:  1625	Performances in adaptability 𝑠̄ and anti-forgetting ability 𝑚̄ under IR (query) to VIS
1627:  1627	Table 1 and 2.
1632:  1632	Table 4
1633:  1633	Structure analysis for DMM strategy. The training order and evaluation procedures are
1634:  1634	the same as Table 1 and 2.
1700:  1700	Base. + CDCN
1701:  1701	Base. + DMM
1702:  1702	Base. + CDCN + DMM
1758:  1758	4.5. Ablation studies
1759:  1759	We performed ablation experiments to evaluate the individual impacts of distinct modules on the comprehensive enhancement of the
1760:  1760	model, as evidenced in Tables 1 and 2. Our research outcomes illustrate
1761:  1761	that integrating the replay sample Domain-Modal-Mixed (DMM) reconstruction strategy into the baseline model notably bolsters its resistance
1762:  1762	against catastrophic forgetting, although this comes with a trade-off
1763:  1763	of moderately compromising the model’s adaptability. Simultaneously,
1764:  1764	incorporating the Cross-domain Cognitive Network (CDCN) results in
1765:  1765	a more conspicuous elevation of the model’s adaptability throughout
1768:  1768	a result where the whole exceeds the sum of its parts. The CDCN
1769:  1769	compensates for the adverse effect of DMM on adaptation, while DMM,
1770:  1770	in turn, reinforces the anti-forgetting memory efficacy of the CDCN.
1772:  1772	Fig. 5. Parameter Analysis of 𝑝𝑐𝑚 and 𝑝𝑐𝑑 . The Average-mAP indicates the mean
1776:  1776	4.5.1. The effectiveness of CDCN
1777:  1777	As demonstrated in Table 1, following the integration of the CDCN
1778:  1778	module, the model (Base.+ CDCN) exhibits a notably enhanced adaptability across all stages compared to the baseline, with improvements
1782:  1782	compared to other methods, our CDCN achieves enhancements in the
1785:  1785	Rank-1 and 4.45 mAP under the IR to VIS test mode (in Table 3),
1787:  1787	adaptability. Furthermore, our CDCN also facilitates an augmented
1788:  1788	resistance to forgetting in the model. With respect to the weighted average anti-forgetting performance ̄
1790:  1790	our method (Base.+CDCN) achieves an elevation of 3.8 Rank-1 and
1794:  1794	forgetting across different modalities and domains learning tasks.
1798:  1798	adaptation phase and the rehearsal phase, as detailed in Table 4. It
1800:  1800	of a mixed strategy leads to a decline in the model’s adaptability,
1801:  1801	potentially due to the CDCN’s requirement to consolidate information
1805:  1805	previously learned knowledge, thereby severely undermining its antiforgetting ability. Consequently, the exclusive application of the mixed
1807:  1807	Parameter Analysis. As depicted in Fig. 5, we employ hold-off
1809:  1809	DMM. In order to strike an optimal balance between adaptability and
1810:  1810	anti-forgetting ability, we initially select 𝑝𝑐𝑚 to maximize the mean
1814:  1814	excellent performance in terms of both adaptability and anti-forgetting
1817:  1817	4.5.2. The effectiveness of DMM
1818:  1818	After applying DMM to the baseline, regarding the weighted average
1819:  1819	anti-forgetting performance ̄
1822:  1822	the VIS to IR test mode (in Table 2), and further accomplishes increases of 4.17 Rank-1 and 3.07 mAP when tested under the IR to VIS
1823:  1823	mode (in Table 3), thereby validating its robust capacity in alleviating
1824:  1824	catastrophic forgetting in the LVI-ReID task.
1826:  1826	DMM, we conducted comparative tests across three scenarios: adding
1829:  1829	As shown Table 5, we test four different training order and found
1841:  1841	Table 5
1883:  1883	Table 6
1898:  1898	Fig. 6. Comparative visualization of extracted features. From each dataset, we randomly selected 8192 infrared and 8192 visible images, with points of the same color
1938:  1938	multiple cameras within the domain, our CDCN mitigates the effects
1942:  1942	designed for the lifelong learning task, our CDCN tailors its fully
1947:  1947	forgetting. This might be caused by two factors: first, the HITSZ-VCM
1949:  1949	Table 1, the model’s best performance on the HITSZ-VCM dataset is
1954:  1954	large gap between the anti-forgetting performance and the basic level
1958:  1958	to large, on the final results, as shown in Table 6. When the memory
1959:  1959	size is too small (size = 128), the model’s ability to resist forgetting
1964:  1964	5.2. The respective pros and cons of DMM and CDCN
1965:  1965	How to balance adaptability and anti-forgetting ability in lifelong
1967:  1967	DMM is designed to tackle the reduction in anti-forgetting performance
1969:  1969	anti-forgetting ability. Just like a student, If you help him review more
1971:  1971	new knowledge to some extent, so DMM alone can improve the antiforgetting ability while cause a negative impact on adaptability. As for
1972:  1972	the CDCN, it is designed to help model adapt new knowledge mainly,
1978:  1978	accumulated across domains by cognitive nodes, CDCN also becomes
1979:  1979	a extend memory module to improve the anti-forgetting ability of the
1986:  1986	each domain). As depicted in Fig. 6, we conducted a comparative
1996:  1996	5.1. Similarities and differences between CDCN and AKA
1997:  1997	Despite the remarkable progress achieved by our proposed DMM
1998:  1998	in combating catastrophic forgetting, further exploration into the potential of mixing samples across domains and modalities remains a
1999:  1999	valuable issue. In stage-wise experiments (in Fig. 4), our model did
2002:  2002	cross-domain images. Additionally, applying too strong a DMM (by
2006:  2006	Our Cross-domain Cognitive Network (CDCN) and AKA (Pu et al.,
2072:  2072	In this paper, we address the Lifelong Visible–Infrared Person ReIdentification (LVI-ReID) problem from a more fine-grained perspective, aiming to solve the critical issues of catastrophic forgetting and
2073:  2073	poor adaptability to new tasks encountered by models when learning cross-modal and cross-domain data. To effectively mitigate the
2074:  2074	detrimental effects of catastrophic forgetting, we introduce the Replay
2076:  2076	decay caused by cross-modal and cross-domain differences. Concurrently, we propose a Camera-aware Cross-domain Cognitive Network
2079:  2079	model’s adaptability but also significantly reinforce its retention of
2087:  2087	coordinated anti-forgetting and adaptation. In: 2022 IEEE/CVF Conference on
2098:  2098	Kumaran, D., Hadsell, R., 2016. Overcoming catastrophic forgetting in neural
2105:  2105	Li, Z., Hoiem, D., 2018. Learning without forgetting. IEEE Trans. Pattern Anal. Mach.
2164:  2164	Serra, J., Suris, D., Miron, M., Karatzoglou, A., 2018. Overcoming catastrophic forgetting with hard attention to the task. In: Dy, J., Krause, A. (Eds.), Proceedings of
2190:  2190	Wu, G., Gong, S., 2021. Generalising without forgetting for lifelong person
2229:  2229	Zhang, Y., Wang, H., 2023. Diverse embedding expansion network and low-light crossmodality benchmark for visible-infrared person re-identification. In: Proceedings of

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '1548,1585p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1548	Fig. 4. Performances in adaptability 4(a) and anti-forgetting ability 4(b) under both VIS to IR and IR to VIS test modes. Our method achieves optimal stage-wise adaptation and
  1549	enhanced long-term robustness in LVI-ReID.
  1550	
  1551	Adaptation Performance. Our method (Base. + CDCN + DMM)
  1552	significantly outperforms other state-of-the-art approaches in the term
  1553	of adaptability (in Table 1). Specifically, under the VIS to IR test
  1554	mode, our method has achieved enhancements of 11.53 and 9.39 in
  1555	Rank-1 and mAP, respectively, in 𝑠̄ when compared to LL methods,
  1556	while outperforming the Lifelong ReID methods by 6.95 Rank-1 and
  1557	5.37 mAP. Under the IR to VIS test mode (Table 3), our method
  1558	achieves improvements of 6.62 Rank-1 and 4.73 mAP over the average
  1559	adaptation performance 𝑠̄ of LL and LReID approaches. This may be
  1560	attributed to the fact that LL approaches are primarily designed for
  1561	closed datasets, whereas LReID methods tend to lack considerations for
  1562	cross-modal knowledge learning.
  1563	Anti-forgetting Performance. With regard to anti-forgetting ability, our method (Base. + CDCN + DMM) also demonstrates remarkable improvements (in Table 2). Under the VIS to IR test mode, our
  1564	method demonstrates notable improvements compared to LL methods,
  1565	achieving increases of 8.63 Rank-1 and 7.82 mAP, respectively, in ̄
  1566	𝑚.
  1567	Furthermore, compared with LReID methods, our method surpasses
  1568	them with enhancements of 8.41 Rank-1 and 7.05 mAP. While we
  1569	achieve 6.54 Rank-1 and 4.52 mAP increases under the IR to VIS test
  1570	mode (in Table 3). This observed superiority of our method likely
  1571	arises from addressing the issue of catastrophic forgetting induced by
  1572	the modal gap, and this deficiency is common to both LL and LReID
  1573	approaches.
  1574	Stage-wise Performance. As depicted in Fig. 4, we evaluated the
  1575	𝑠̄ and 𝑚̄ metrics for each model at every stage. Regarding adaptability,
  1576	our approach achieved the best performance across all stages. With
  1577	respect to anti-forgetting capability, while our method did not yield the
  1578	optimal results in the second phase, it went on to achieve the best performance in the long run as the training progressed through subsequent
  1579	stages. This discrepancy in the second phase is likely attributed to the
  1580	fact that, during this phase, the memory bank comprised data from a
  1581	single domain only, precluding the mix of cross-domain images. Consequently, the model was unable to unleash its potential fully. Our model
  1582	substantially boosts its anti-forgetting ability as training progresses, and
  1583	the memory bank encompasses a more diverse dataset. This accentuates
  1584	the efficacy of our approach in facilitating lifelong learning, underlining
  1585	its prowess in sustaining the continuous acquisition of new knowledge

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf' - | nl -ba | sed -n '1758,1826p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1758	4.5. Ablation studies
  1759	We performed ablation experiments to evaluate the individual impacts of distinct modules on the comprehensive enhancement of the
  1760	model, as evidenced in Tables 1 and 2. Our research outcomes illustrate
  1761	that integrating the replay sample Domain-Modal-Mixed (DMM) reconstruction strategy into the baseline model notably bolsters its resistance
  1762	against catastrophic forgetting, although this comes with a trade-off
  1763	of moderately compromising the model’s adaptability. Simultaneously,
  1764	incorporating the Cross-domain Cognitive Network (CDCN) results in
  1765	a more conspicuous elevation of the model’s adaptability throughout
  1766	various stages, while concurrently bolstering its memory retention
  1767	capabilities. The synergistic application of these two components yields
  1768	a result where the whole exceeds the sum of its parts. The CDCN
  1769	compensates for the adverse effect of DMM on adaptation, while DMM,
  1770	in turn, reinforces the anti-forgetting memory efficacy of the CDCN.
  1771	
  1772	Fig. 5. Parameter Analysis of 𝑝𝑐𝑚 and 𝑝𝑐𝑑 . The Average-mAP indicates the mean
  1773	performance of both 𝑠̄ and ̄
  1774	𝑚.
  1775	
  1776	4.5.1. The effectiveness of CDCN
  1777	As demonstrated in Table 1, following the integration of the CDCN
  1778	module, the model (Base.+ CDCN) exhibits a notably enhanced adaptability across all stages compared to the baseline, with improvements
  1779	including an increase of 15.00 Rank-1 and 15.24 mAP for RegDB, 6.12
  1780	Rank-1 and 5.34 mAP for SYSU-MM01, 4.1 Rank-1 and 3.57 mAP for
  1781	LLCM, and 3.89 Rank-1 and 3.02 mAP for HITSZ-VCM. Meanwhile,
  1782	compared to other methods, our CDCN achieves enhancements in the
  1783	average adaptation performance ̄𝑠, realizing a boost of 6.67 Rank-1
  1784	and 5.43 mAP under the VIS to IR test mode, and an increase of 6.14
  1785	Rank-1 and 4.45 mAP under the IR to VIS test mode (in Table 3),
  1786	thereby demonstrating significant advancements in the cross-domain
  1787	adaptability. Furthermore, our CDCN also facilitates an augmented
  1788	resistance to forgetting in the model. With respect to the weighted average anti-forgetting performance ̄
  1789	𝑚, in comparison to other approaches,
  1790	our method (Base.+CDCN) achieves an elevation of 3.8 Rank-1 and
  1791	3.34 mAP under the VIS to IR test mode, along with an enhancement
  1792	of 2.83 Rank-1 and 1.64 mAP when tested under the inverse IR to
  1793	VIS mode, thereby substantiating its efficacy in mitigating catastrophic
  1794	forgetting across different modalities and domains learning tasks.
  1795	
  1796	mixed strategies solely during the adaptation phase, incorporating them
  1797	only in the rehearsal phase, and applying mixed strategies in both the
  1798	adaptation phase and the rehearsal phase, as detailed in Table 4. It
  1799	has been observed that during the adaptation phase, the employment
  1800	of a mixed strategy leads to a decline in the model’s adaptability,
  1801	potentially due to the CDCN’s requirement to consolidate information
  1802	from the same camera source when accruing knowledge. The implementation of the mixed strategy during the adaptation phase results in
  1803	the accumulation of muddled knowledge, which not only diminishes
  1804	the model’s focus on learning the current domain but also disrupts
  1805	previously learned knowledge, thereby severely undermining its antiforgetting ability. Consequently, the exclusive application of the mixed
  1806	strategy during the replay phase emerges as the optimal strategy.
  1807	Parameter Analysis. As depicted in Fig. 5, we employ hold-off
  1808	validation data to ascertain two hyper-parameters 𝑝𝑐𝑚 and 𝑝𝑐𝑑 in the
  1809	DMM. In order to strike an optimal balance between adaptability and
  1810	anti-forgetting ability, we initially select 𝑝𝑐𝑚 to maximize the mean
  1811	performance of both 𝑠̄ and ̄
  1812	𝑚. Subsequently, based on the selected 𝑝𝑐𝑚 ,
  1813	we determine the optimal value for 𝑝𝑐𝑑 . Ultimately, our model achieves
  1814	excellent performance in terms of both adaptability and anti-forgetting
  1815	ability with 𝑝𝑐𝑚 = 0.05 and 𝑝𝑐𝑑 = 0.5.
  1816	
  1817	4.5.2. The effectiveness of DMM
  1818	After applying DMM to the baseline, regarding the weighted average
  1819	anti-forgetting performance ̄
  1820	𝑚, in comparison to other approaches, our
  1821	method yields enhancements of 4.76 Rank-1 and 4.43 mAP under
  1822	the VIS to IR test mode (in Table 2), and further accomplishes increases of 4.17 Rank-1 and 3.07 mAP when tested under the IR to VIS
  1823	mode (in Table 3), thereby validating its robust capacity in alleviating
  1824	catastrophic forgetting in the LVI-ReID task.
  1825	Structure Analysis. To further evaluate the effectiveness of our
  1826	DMM, we conducted comparative tests across three scenarios: adding

codex
我按你指定的 `pdftotext -f 1 -l 8` 读了方法段；其中几篇方法跨过第 8 页，我又补读后续页把方法收完整。

## 1. Learning Visual-Semantic Embedding for Generalizable Person Re-Identification

1. 触发观察：普通泛化 ReID 主要还是图像特征学习，视觉语言方法又常要求图文配对完整。现实里可能只有图像、只有文本、或图文都有，现有方法没有统一处理“模态缺失”的能力。

2. 重定义动作：作者把问题从“做更强 DG ReID”改成“学习一个视觉语义统一嵌入”。关键词是 `visual-semantic embedding`、`unimodal or multimodal`、`incomplete modality`、`unified perspective`。这让图像 ReID、文本 ReID、图文 ReID 被讲成同一个问题。

3. 机制怎么长出来：既然要同时处理图像、文本、图文，就自然长出图像编码器、文本编码器、多模态编码器。既然模态会缺失，就用图像掩码、文本掩码、图文联合掩码来预训练。`MMM` 不是单纯随机遮挡，而是强调行人区域，服务于“视觉语义嵌入要关注人本体”的叙事。

4. 证据闭环：关键消融是 ViT 基线、MMET 加随机掩码、MMET 加 MMM。MMM 在 Market 和 MSMT 上明显优于随机掩码。另一个闭环是 MIM、MLM、MMM 的对比，证明图像和文本联合掩码确实比单模态掩码更贴合“统一视觉语义”的定义。

5. 审稿人为什么买账：这篇卖的主要是视角，不是某个新模块。真正的新意是把 ReID 泛化、图文预训练、模态缺失统一到一个训练框架里。机制很多来自视觉语言预训练，但套到 ReID 的“模态不完整泛化”上，有清楚的问题重定义。

## 2. Lifelong person re-identification via dynamically knowledge adaptation and retention

1. 触发观察：终身 ReID 里，很多方法为了不忘旧域，会限制模型适应新域。这样虽然缓解遗忘，但会让模型在每个具体域上都不是最优。作者抓住的反直觉点是：防遗忘不应该等于少适应。

2. 重定义动作：作者把问题从“如何保留旧知识”改成“如何保留模型对旧域的适应能力”。关键词是 `dynamic adaptation`、`adaptability retention`、`readapt to each domain`。这比普通蒸馏更进一步，因为蒸馏对象变成了“动态参数生成能力”。

3. 机制怎么长出来：如果每个域都需要不同适应方式，就用 DyAdap 动态卷积，根据输入特征生成卷积参数；如果动态适应器本身也会忘，就用 AdRet 约束当前模型和旧模型在特征以及动态参数上的一致性；如果批归一化带有域统计，就加 BNF 融合旧新 BN。

4. 证据闭环：关键消融是逐步加入 DyAdap、AdRet、BNF。文中报告 DyAdap 平均带来 15.2 mAP，AdRet 带来 4.9 mAP，同时加入带来 34.7 mAP，BNF 还有小幅增益。这个消融证明“动态适应”和“保留适应能力”是两个不同但互补的轴。

5. 审稿人为什么买账：这篇卖的是“防遗忘不等于冻结适应”的视角。机制不是简单再蒸馏一次，而是把蒸馏落到动态参数生成器上，和重定义绑定比较紧。

## 3. Lifelong visible-infrared person re-identification via DMM and CDCN

1. 触发观察：LVI-ReID 不只是终身学习，也不只是可见光红外 ReID。连续到来的新域同时带来域差异和模态差异，小 memory bank 反复重放还会过拟合旧样本，导致适应性和抗遗忘能力一起下降。

2. 重定义动作：作者把问题拆成两个指标轴：`domain adaptability` 和 `anti-forgetting ability`。再进一步说，二者的下降来自 `domain semantic gap` 和 `modality semantic gap`。这让方法可以自然分成一个管适应、一个管遗忘。

3. 机制怎么长出来：DMM 在图像层面对 memory bank 样本做域和模态混合，增加有限重放样本的多样性，主攻抗遗忘。CDCN 在特征层面建相机认知节点和图卷积，建模同模态一致性、跨模态相似性、跨域共享，主攻适应性。一个在 rehearsal 阶段用，一个在 adaptation 阶段用，逻辑上刚好对应两类退化。

4. 证据闭环：关键不是只报最终涨点，而是分别证明 DMM 和 CDCN 的角色。消融显示 DMM 提升抗遗忘但会牺牲适应性，CDCN 提升适应性并改善记忆，二者组合互相补偿。另一个关键证据是 DMM 只放在重放阶段最好，放在适应阶段会污染相机认知节点。

5. 审稿人为什么买账：这篇卖的是“二维问题拆解”。DMM 和 CDCN 单看都不算特别新，但它们分别对应抗遗忘和适应性，组合后形成闭环，所以贡献来自问题结构化，而不是单个零件。

## 4. Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator

1. 触发观察：终身提示学习里，固定 prompt pool 能记共享知识，但容量和任务数绑定；实例级 prompt 适应性强，但长期共享和连续记忆弱。LVI-ReID 还要求测试时没有任务标签，只能自己判断当前输入该用什么提示和分类器。

2. 重定义动作：作者把 LVI-ReID 从“持续更新模型”改成“任务无关的提示选择和实例级提示生成”。关键词是 `Prompt Pool`、`Instance-level Prompt Generator`、`query-key mechanism`、`task-agnostic inference`、`parameter scalability`。

3. 机制怎么长出来：Prompt Pool 保存跨任务共享和域相关提示，IPG 根据输入 token 和推断出的任务标识生成实例级提示。查询键机制用类 token 选 prompt，用额外 query token 选分类器和任务标识。这样固定池负责连续性，实例生成器负责细粒度适应。

4. 证据闭环：表 2 是核心。IPG 单独用时，后期比基线明显好，说明实例提示抗遗忘有效；Prompt Pool 单独用时，任务越多记忆优势越明显；两者组合最好。图 5 的 key 选择频率从训练前随机变成训练后有结构，证明模型确实学会了无任务标签的路由。

5. 审稿人为什么买账：这篇卖的是 prompt learning 在 LVI-ReID 中的可扩展用法。新意不在 prompt 这个概念，而在把固定池和实例生成做成互补结构，并用 task-agnostic inference 解释现实部署价值。

## 5. Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification

1. 触发观察：CNN 有局部性但缺全局依赖，ViT 有全局建模但越深越全局化，patch 之间越来越相似，浅层颜色、纹理、局部细节被冲淡。作者用 GradCAM 和特征相似矩阵先把这个现象可视化出来。

2. 重定义动作：作者把 VI-ReID 的困难从“模态差异”重说成“Transformer 过度全局化和特征坍缩导致局部低层线索丢失”。关键词是 `over-globalization`、`feature collapse`、`low-level cues`、`Local Residual Attention`。

3. 机制怎么长出来：既然 patch embedding 里有卷积局部感受野，就用 PCR 对 patch 做通道重权重；既然低层局部关系会在深层消失，就用 LAR 把局部区域的注意力关系从浅层残差传到深层；既然 VI-ReID 还有模态差异，就用 CAL 适度拉近可见光和红外中心，而不是过强压缩。

4. 证据闭环：最关键的是图 2 的相似矩阵，它直接支撑“深层 patch 变得过度相似”的重定义。表 4 中基线 62.2/57.5，加入 PCR、LAR、CAL 后逐步提升，完整模型到 76.7/72.9。后续还有局部分区、注意力残差、热力图、t-SNE 和类内类间距离可视化。

5. 审稿人为什么买账：这篇卖的是诊断视角加机制。它不是泛泛地说“加局部特征”，而是先证明 ViT 在 VI-ReID 中会丢失低层局部差异，再让 PCR 和 LAR 顺着这个诊断长出来。

## 6. Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification

1. 触发观察：遮挡 ReID 中，Transformer 的多头注意力虽然能看不同部位，但也会被背景和遮挡物吸引。与此同时，patch 级处理会丢掉像素和局部模式之间的细粒度信息。

2. 重定义动作：作者把遮挡问题从“身体部位缺失”改成“注意力被遮挡污染，并且细粒度局部模式没有被保住”。关键词是 `mask-aware`、`non-occluded body parts`、`foreground semantic embeddings`、`hierarchical feature aggregation`。

3. 机制怎么长出来：如果注意力会看错，就用 HRNet 解析出人体前景 mask，把 mask patch 投影成语义嵌入，和图像 patch、位置、相机嵌入一起送入 Transformer。若高层 patch 太平滑，就聚合第 2、4、10、12 层的 patch 表征做 HFA。若遮挡导致全局特征断裂，就用多分支空洞卷积 DFC 补全不同感受野下的全局信息。

4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。

5. 审稿人为什么买账：这篇更偏机制稿，但它有清楚的观察闭环。MIE 对应注意力定位错误，HFA 对应细粒度信息丢失，DFC 对应遮挡下全局特征不完整，三个部件虽然多，但各自有明确病因。

共同套路：这些论文都不是先说“我加了一个模块”，而是先找一个具体失败现象，再把它升级成一个新问题词汇。能发的方法稿通常让机制看起来像是从这个新问题自然长出来的，然后用一个消融或可视化证明“这个新问题确实存在”，而不是只证明最终涨点。
tokens used
161,345
我按你指定的 `pdftotext -f 1 -l 8` 读了方法段；其中几篇方法跨过第 8 页，我又补读后续页把方法收完整。

## 1. Learning Visual-Semantic Embedding for Generalizable Person Re-Identification

1. 触发观察：普通泛化 ReID 主要还是图像特征学习，视觉语言方法又常要求图文配对完整。现实里可能只有图像、只有文本、或图文都有，现有方法没有统一处理“模态缺失”的能力。

2. 重定义动作：作者把问题从“做更强 DG ReID”改成“学习一个视觉语义统一嵌入”。关键词是 `visual-semantic embedding`、`unimodal or multimodal`、`incomplete modality`、`unified perspective`。这让图像 ReID、文本 ReID、图文 ReID 被讲成同一个问题。

3. 机制怎么长出来：既然要同时处理图像、文本、图文，就自然长出图像编码器、文本编码器、多模态编码器。既然模态会缺失，就用图像掩码、文本掩码、图文联合掩码来预训练。`MMM` 不是单纯随机遮挡，而是强调行人区域，服务于“视觉语义嵌入要关注人本体”的叙事。

4. 证据闭环：关键消融是 ViT 基线、MMET 加随机掩码、MMET 加 MMM。MMM 在 Market 和 MSMT 上明显优于随机掩码。另一个闭环是 MIM、MLM、MMM 的对比，证明图像和文本联合掩码确实比单模态掩码更贴合“统一视觉语义”的定义。

5. 审稿人为什么买账：这篇卖的主要是视角，不是某个新模块。真正的新意是把 ReID 泛化、图文预训练、模态缺失统一到一个训练框架里。机制很多来自视觉语言预训练，但套到 ReID 的“模态不完整泛化”上，有清楚的问题重定义。

## 2. Lifelong person re-identification via dynamically knowledge adaptation and retention

1. 触发观察：终身 ReID 里，很多方法为了不忘旧域，会限制模型适应新域。这样虽然缓解遗忘，但会让模型在每个具体域上都不是最优。作者抓住的反直觉点是：防遗忘不应该等于少适应。

2. 重定义动作：作者把问题从“如何保留旧知识”改成“如何保留模型对旧域的适应能力”。关键词是 `dynamic adaptation`、`adaptability retention`、`readapt to each domain`。这比普通蒸馏更进一步，因为蒸馏对象变成了“动态参数生成能力”。

3. 机制怎么长出来：如果每个域都需要不同适应方式，就用 DyAdap 动态卷积，根据输入特征生成卷积参数；如果动态适应器本身也会忘，就用 AdRet 约束当前模型和旧模型在特征以及动态参数上的一致性；如果批归一化带有域统计，就加 BNF 融合旧新 BN。

4. 证据闭环：关键消融是逐步加入 DyAdap、AdRet、BNF。文中报告 DyAdap 平均带来 15.2 mAP，AdRet 带来 4.9 mAP，同时加入带来 34.7 mAP，BNF 还有小幅增益。这个消融证明“动态适应”和“保留适应能力”是两个不同但互补的轴。

5. 审稿人为什么买账：这篇卖的是“防遗忘不等于冻结适应”的视角。机制不是简单再蒸馏一次，而是把蒸馏落到动态参数生成器上，和重定义绑定比较紧。

## 3. Lifelong visible-infrared person re-identification via DMM and CDCN

1. 触发观察：LVI-ReID 不只是终身学习，也不只是可见光红外 ReID。连续到来的新域同时带来域差异和模态差异，小 memory bank 反复重放还会过拟合旧样本，导致适应性和抗遗忘能力一起下降。

2. 重定义动作：作者把问题拆成两个指标轴：`domain adaptability` 和 `anti-forgetting ability`。再进一步说，二者的下降来自 `domain semantic gap` 和 `modality semantic gap`。这让方法可以自然分成一个管适应、一个管遗忘。

3. 机制怎么长出来：DMM 在图像层面对 memory bank 样本做域和模态混合，增加有限重放样本的多样性，主攻抗遗忘。CDCN 在特征层面建相机认知节点和图卷积，建模同模态一致性、跨模态相似性、跨域共享，主攻适应性。一个在 rehearsal 阶段用，一个在 adaptation 阶段用，逻辑上刚好对应两类退化。

4. 证据闭环：关键不是只报最终涨点，而是分别证明 DMM 和 CDCN 的角色。消融显示 DMM 提升抗遗忘但会牺牲适应性，CDCN 提升适应性并改善记忆，二者组合互相补偿。另一个关键证据是 DMM 只放在重放阶段最好，放在适应阶段会污染相机认知节点。

5. 审稿人为什么买账：这篇卖的是“二维问题拆解”。DMM 和 CDCN 单看都不算特别新，但它们分别对应抗遗忘和适应性，组合后形成闭环，所以贡献来自问题结构化，而不是单个零件。

## 4. Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator

1. 触发观察：终身提示学习里，固定 prompt pool 能记共享知识，但容量和任务数绑定；实例级 prompt 适应性强，但长期共享和连续记忆弱。LVI-ReID 还要求测试时没有任务标签，只能自己判断当前输入该用什么提示和分类器。

2. 重定义动作：作者把 LVI-ReID 从“持续更新模型”改成“任务无关的提示选择和实例级提示生成”。关键词是 `Prompt Pool`、`Instance-level Prompt Generator`、`query-key mechanism`、`task-agnostic inference`、`parameter scalability`。

3. 机制怎么长出来：Prompt Pool 保存跨任务共享和域相关提示，IPG 根据输入 token 和推断出的任务标识生成实例级提示。查询键机制用类 token 选 prompt，用额外 query token 选分类器和任务标识。这样固定池负责连续性，实例生成器负责细粒度适应。

4. 证据闭环：表 2 是核心。IPG 单独用时，后期比基线明显好，说明实例提示抗遗忘有效；Prompt Pool 单独用时，任务越多记忆优势越明显；两者组合最好。图 5 的 key 选择频率从训练前随机变成训练后有结构，证明模型确实学会了无任务标签的路由。

5. 审稿人为什么买账：这篇卖的是 prompt learning 在 LVI-ReID 中的可扩展用法。新意不在 prompt 这个概念，而在把固定池和实例生成做成互补结构，并用 task-agnostic inference 解释现实部署价值。

## 5. Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification

1. 触发观察：CNN 有局部性但缺全局依赖，ViT 有全局建模但越深越全局化，patch 之间越来越相似，浅层颜色、纹理、局部细节被冲淡。作者用 GradCAM 和特征相似矩阵先把这个现象可视化出来。

2. 重定义动作：作者把 VI-ReID 的困难从“模态差异”重说成“Transformer 过度全局化和特征坍缩导致局部低层线索丢失”。关键词是 `over-globalization`、`feature collapse`、`low-level cues`、`Local Residual Attention`。

3. 机制怎么长出来：既然 patch embedding 里有卷积局部感受野，就用 PCR 对 patch 做通道重权重；既然低层局部关系会在深层消失，就用 LAR 把局部区域的注意力关系从浅层残差传到深层；既然 VI-ReID 还有模态差异，就用 CAL 适度拉近可见光和红外中心，而不是过强压缩。

4. 证据闭环：最关键的是图 2 的相似矩阵，它直接支撑“深层 patch 变得过度相似”的重定义。表 4 中基线 62.2/57.5，加入 PCR、LAR、CAL 后逐步提升，完整模型到 76.7/72.9。后续还有局部分区、注意力残差、热力图、t-SNE 和类内类间距离可视化。

5. 审稿人为什么买账：这篇卖的是诊断视角加机制。它不是泛泛地说“加局部特征”，而是先证明 ViT 在 VI-ReID 中会丢失低层局部差异，再让 PCR 和 LAR 顺着这个诊断长出来。

## 6. Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification

1. 触发观察：遮挡 ReID 中，Transformer 的多头注意力虽然能看不同部位，但也会被背景和遮挡物吸引。与此同时，patch 级处理会丢掉像素和局部模式之间的细粒度信息。

2. 重定义动作：作者把遮挡问题从“身体部位缺失”改成“注意力被遮挡污染，并且细粒度局部模式没有被保住”。关键词是 `mask-aware`、`non-occluded body parts`、`foreground semantic embeddings`、`hierarchical feature aggregation`。

3. 机制怎么长出来：如果注意力会看错，就用 HRNet 解析出人体前景 mask，把 mask patch 投影成语义嵌入，和图像 patch、位置、相机嵌入一起送入 Transformer。若高层 patch 太平滑，就聚合第 2、4、10、12 层的 patch 表征做 HFA。若遮挡导致全局特征断裂，就用多分支空洞卷积 DFC 补全不同感受野下的全局信息。

4. 证据闭环：图 1 的注意力热图直接证明 TransReID 会看遮挡和背景，而 MAHATMA 更关注未遮挡人体。表 V 中 MIE、HFA、DFC 分别带来明显提升，完整模型在 Occluded-Duke 达到 73.3 rank1 和 62.3 mAP。图 3 的 t-SNE 进一步显示，随着 MIE、HFA、DFC 加入，遮挡造成的离群特征逐步消失。

5. 审稿人为什么买账：这篇更偏机制稿，但它有清楚的观察闭环。MIE 对应注意力定位错误，HFA 对应细粒度信息丢失，DFC 对应遮挡下全局特征不完整，三个部件虽然多，但各自有明确病因。

共同套路：这些论文都不是先说“我加了一个模块”，而是先找一个具体失败现象，再把它升级成一个新问题词汇。能发的方法稿通常让机制看起来像是从这个新问题自然长出来的，然后用一个消融或可视化证明“这个新问题确实存在”，而不是只证明最终涨点。
