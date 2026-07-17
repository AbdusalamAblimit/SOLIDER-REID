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
session id: 019ef835-61a8-7560-bb43-41f98c840972
--------
user
你是 ReID 创新方法论拆解员 4。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(4)==
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按你列出的 6 篇处理，因为标题清单实际有 6 个 PDF。先核对本目录文件和项目目标文件的可读性，然后按你指定的 `pdftotext -f 1 -l 8` 读方法部分。
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
/bin/zsh -lc "sed -n '1,200p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,160p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "pwd && rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,200p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我会用 PDF 技能处理这批论文，但这里主要按你的要求走 `pdftotext` 文本抽取。刚读过 `GOAL.md` 和当前状态，任务和现在的文献差距分析目标一致。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Dynamic Weighting Fusion Module
To further enhance feature representation, we introduce a dynamic weighting fusion module that assigns adaptive weights
to each feature based on prediction uncertainty, allowing the
model to prioritize more reliable features. Specifically, the
global visual feature vg , the textual feature tp , and the crossattention feature satt are fed into the fusion module. For each
feature fm (m = 1, . . . , M , where M = 3, representing the
visual, textual, and cross-attention features), the associated
weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
distribution. The uncertainty Em is computed as follows:

local patch features, facilitating their interaction. The crossattention output H i for the learnable queries in the i-th attention block is then computed as:


Qd Kd⊤
i
√
H = CrossAttn(Qd , Kd , Vd ) = softmax
Vd .
d
(3)
Subsequently, we feed H i into a two-layer feed-forward network FFN(·), producing X̃ i , which represents the updated
features of the learnable queries after the i-th attention block.
This process is formulated as:


X̃ i = FFN H i + X̃ i−1 + H i .
(4)

Em = −
Afterwards, we perform average pooling over the refined
query embeddings after multiple transformer blocks. The result is then passed through a simple mapping network ϕa to
derive the final attribute token A∗ . It can be formulated as:
A∗ = ϕa (AvgPool(X̃f inal )),

pm (k) log pm (k),

(8)

k=1

where pm (k) is the softmax probability assigned to the k-th
individual’s feature fm , and C represents the total number of
individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
that feature during the fusion process. The weight λm for
feature fm is then determined as follows:


exp
M ax (Em ) − Em
m=1,...,M
λm = M
(9)

.
P
exp
M ax (Em ) − Eq

(5)

where X̃f inal is the final output query embeddings from multiple transformer blocks, AvgPool(·) denotes average pooing, and ϕa denotes a three-layer feed-forward network.
The above network produces the refined attribute token A∗ .
Similarly, a parallel network with the same structure but different parameters is employed to generate the corresponding
context token O∗ . In this network, ϕo serves as the associated
mapping function.

3.2

C
X

m=1,...,M

q=1

The final fused representation fˆ is formulated as:
fˆ =

Multi-modal Feature Enhancement

M
X

λm f m ,

(10)

m=1

To effectively capture the intricate relationships between
multi-modal information, we integrate a cross-attention
mechanism with a dynamic fusion strategy, enhancing the
model’s capacity to represent nuanced identity features.

where fm represents each prediction feature. This adaptive
fusion approach enables balanced contributions from the visual, textual, and cross-attention features.

3.3

Cross-Attention Mechanism
The cross-attention mechanism allows the model to interactively align specific textual cues with corresponding visual regions, focusing on subject and detail tokens in the constructed
prompt. Let T represent the constructed prompt ”A photo of
a [S ∗ ] person with [A∗ ] partially occluded by [O∗ ]”, which
is then fed into the frozen CLIP text encoder ft to obtain the
textual representation tp . Formally, this process can be expressed as follows:
tp = ft (T ).
(6)

Loss Function and Inference

Our framework incorporates four loss functions: Crossmodal Contrastive Loss LAlign [Radford et al., 2021], Triplet
Loss LTriplet [Hermans et al., 2017], ID Classification Loss
LID [Zheng et al., 2017], and ArcFace Loss LArcFace [Deng
et al., 2019].
Cross-modal Contrastive Loss
To align visual and textual representations of each individual, we employ a cross-modal contrastive loss that encourages high similarity between images and their corresponding
prompts in the embedding space. It is formulated as:

To implement cross-attention, we treat the textual embedding
tp as the query Qc , while the visual feature v extracted from
the image serves as both the key Kc and value Vc . The output
of the cross-attention mechanism can be expressed as:


Qc Kc⊤
√
Vc .
satt = CrossAttn(Qc , Kc , Vc ) = softmax
d
(7)
By leveraging the cross-attention mechanism, the textual tokens dynamically guide the image encoder to focus on the visible and discriminative features of the pedestrian while suppressing noise from occluded regions.

LAlign = Li2t + Lt2i ,
X
exp (sim (vi , tp+ ) /τ )
,
Li2t (i) = −
log N
P
p+ ∈P (n)
exp (sim (vi , tn ) /τ )

(11)
(12)

n=1

Lt2i (i) = −

exp (sim (ti , vp+ ) /τ )
log N
,
P
p+ ∈P (n)
exp (sim (ti , vn ) /τ )
X

(13)

n=1

where P (n) denotes the set of positive samples that correspond to the same identity i and τ is a temperature parameter.

804

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Triplet Loss
The triplet loss is employed to increase the distinction between identities by minimizing the distance between positive
pairs while maximizing the distance between negative pairs.
This loss is formulated as:
LTriplet = max (dp − dn + m, 0) ,
(14)
where dp and dn are the distances of the positive and negative
pairs respectively and m is the margin.

Methods
PVPM (CVPR 20)
HOReID (CVPR 20)
RFCnet (TPAMI 21)
HCGA (TIP 23)
PAT (CVPR 21)
TransReID (ICCV 21)
DRL-Net (TMM 22)
PFD (AAAI 22)
DPM (ACM MM 22)
SAP (AAAI 23)
OAT (TIP 24)
OAMN (ICCV 21)
FED (CVPR 22)
CAAO (TIP 23)
ADP (AAAI 24)
DPM-SPT (AAAI 24)
CLIP-ReID (AAAI 23)
FLaN-Net (Ours)

ID Classification Loss
The ID classification loss ensures correct identification of
each individual and is defined as:
N
X
LID = −
qk log (yk ) ,
(15)
k=1

where yk is the predicted probability of an individual, and qk
is the corresponding ground truth label.
ArcFace Loss
ArcFace loss improves feature discrimination by introducing
an angular margin to enhance the separation between different individuals and compact intra-individual representations:
s cos θy +m))
PN
e ( ( k
LArcFace = − N1 k=1 log s cos θ +m
,
yk
(
(
)) +Pn
e
es cos θj

Occ-Duke
R-1 mAP
47.0 37.7
55.1 43.8
63.9 54.5
70.2 57.5
64.5 53.6
66.4 59.2
65.8 53.9
69.5 61.8
71.4 61.8
70.0 62.2
71.8 62.2
62.6 46.1
68.1 56.4
68.5 59.5
74.5 63.8
74.7 63.0
67.1 59.5
75.2 65.5

Occ-REID
R-1 mAP
66.8 59.5
80.3 70.2
81.6 72.1
81.5 83.0
85.5 79.7
83.0 76.8
82.6 78.2
86.3 79.3
87.1 83.4
89.2 85.1
87.8 81.1
92.6 89.5

Table 1: Performance comparison on Occluded-Duke and OccludedREID datasets. The compared methods are grouped into four categories: auxiliary model-based, transformer-based, data augmentation and language-guided.

j=1,j̸=yk

(16)
where θj denotes the angle between a feature and the weight
vector of the j-th identity. The angular margin m improves
discrimination, and the scale factor s stabilizes optimization.

4

Experiments

Implementation Details
In this work, we employ the ViT-B/16 pretrained on CLIP as
the visual encoder, and the pre-trained CLIP text transformer
as the text encoder. We use two independent fine-grained
noise filtering networks to get A∗ and O∗ , each consisting of
3 learnable queries and 6 cross-attention blocks. The model
is trained using a batch size of 64, consisting of 16 identities,
each with 4 images. All input images are resized to 256 × 128
pixels. For optimization, we use the Adam optimizer with a
base learning rate of 5e-5 for the randomly initialized modules and 1e-5 for the visual encoder. The model is trained for
60 epochs, with the learning rate decaying by a factor of 0.1
at epochs 20 and 40. The Triplet Loss uses the margin m =
0.3, and the ArcFace Loss is configured with the margin m
= 0.5 and the scale factor s = 30. The optimizer for the Arcface Loss function is separately initialized with SGD, using
a learning rate of 0.1 and a weight decay of 5e-4. The λ in
Eq.17 is set to 0.5 for all datasets. All components are trained
on a single NVIDIA RTX3090 GPU.

4.1

Experimental Settings

4.2

Total Loss
Overall, the loss function used in FLaN-Net is defined as:
L = λLAlign + LTriplet + LID + LArcFace ,
(17)
where λ is a hyper-parameter that balances the contribution
of LAlign to the total loss.
Inference
During inference, following CLIP-ReID [Li et al., 2023], we
rely solely on features extracted from the image encoder for
person retrieval. The Euclidean distance is computed between the feature of the query image and those in the gallery
set to identify the closest matches. The proposed components
work together to enhance the model’s ability to accurately
identify the target subject, ultimately optimizing the feature
representation generated by the image encoder.

Datasets and Evaluation Protocols
We evaluate the proposed FLaN-Net method on two categories of datasets: occluded datasets, including OccludedDuke [Miao et al., 2019] and Occluded-REID [Zhuo et al.,
2018], and holistic datasets, comprising Market-1501 [Zheng
et al., 2015a], DukeMTMC-reID [Zheng et al., 2017] and
CUHK03-NP [Li et al., 2014]. As the Occluded-REID
dataset lacks a dedicated training set, we utilize Market1501 for training, consistent with other methods to maintain
a fair basis for comparison. To assess the effectiveness of
our approach, we adopt Cumulative Matching Characteristic
(CMC) curves and the mean Average Precision (mAP).

Comparison with State-of-the-Art Methods

Experimental Results on Occluded ReID Datasets
To evaluate the effectiveness of our proposed FLaN-Net,
we conducted extensive comparisons with various state-ofthe-art methods on the occluded ReID datasets, including
Occluded-Duke and Occluded-REID, and show the results in
Tab. 1. SOTA methods are divided into four mainstreams:
• Auxiliary model-based methods: PVPM [Gao et al.,
2020b]; HOReID [Wang et al., 2020]; RFCnet [Hou et al.,
2021]; HCGA [Dou et al., 2023].
• Transformer-based methods: PAT [Li et al., 2021]; TransReID [He et al., 2021]; DRL-Net [Jia et al., 2022];

805

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Methods
SAN (AAAI 20)
TransReID (ICCV 21)
HAT (ACM MM 21)
DCAL (CVPR 22)
AAformer (TNNLS 23)
PHA (CVPR 23)
CLIP-ReID (AAAI 23)
RFCnet (TPAMI 21)
PFD (AAAI 22)
FED (CVPR 22)
DPM (ACM MM 22)
CAAO (TIP 23)
HCGA (TIP 23)
SAP (AAAI 23)
ADP (AAAI 24)
DPM-SPT (AAAI 24)
OAT (TIP 24)
FLaN-Net (Ours)

Market-1501
R-1 mAP
96.1
88.0
95.2
88.9
95.8
89.8
94.7
87.5
95.4
88.0
96.1
90.2
95.5
89.6
95.2
89.2
95.5
89.7
95.0
86.3
95.5
89.7
95.3
88.0
95.2
88.4
96.0
90.5
95.6
89.5
95.5
89.4
95.7
89.9
95.7
90.5

DukeMTMC
R-1 mAP
87.9
75.5
90.7
82.0
90.4
81.4
89.0
80.1
90.1
80.9
90.0
82.5
90.7
80.7
91.2
83.2
89.4
78.0
91.0
82.6
89.8
80.9
91.2
83.1
91.1
82.4
91.2
82.3
92.1
83.6

Methods
RGA-SC (CVPR 20)
HAT (ACM MM 21)
NetVLAD-M (TIFS 22)
MPN (TPAMI 22)
AAformer (TNNLS 23)
PHA (CVPR 23)
OAT (TIP 24)
FLaN-Net (Ours)

Labeled
R-1 mAP
81.1 77.4
82.6 80.0
80.4 76.7
85.0 81.1
80.3 79.0
84.5 83.0
83.9 81.5
88.1 86.7

Detected
R-1 mAP
79.6 74.5
79.1 75.5
79.7 74.8
83.4 79.1
78.1 77.2
83.2 80.3
80.6 78.0
87.1 84.8

Table 3: Performance comparison on CUHK03-NP.
Index
Prompts
R-1 mAP
1
”A photo of a person”
72.6 63.7
2
”A photo of a [S ∗ ] person”
73.4 64.4
”A photo of a [S ∗ ] person with [A∗ ]”
74.6 65.2
3
∗
∗
∗
4 ”A photo of a [S ] person with [A ] partially occluded by [O ]” 75.2 65.5

Table 4: Ablation study of different prompts on Occluded-Duke.

it by +0.2%/+0.9% Rank-1 accruacy/mAP on Market-1501
and +2.1%/+1.1% Rank-1 accruacy/mAP on DukeMTMC.
Additionally, we evaluate FLaN-Net’s performance on
the CUHK03-NP dataset under both manually labeled and
auto-detected bounding box settings. Several methods are
compared, including RGA-SC [Zhang et al., 2020], HAT,
NetVLAD-M [Zhang et al., 2022], MPN [Ding et al., 2022],
AAformer, PHA, OAT. As shown in Tab. 3, FLaN-Net still
surpasses all other methods with significant margins. Specifically, it achieves at least +3.1%/+3.7% improvements in
Rank-1 accuracy/mAP on the labeled setting and at least
+3.7%/+4.5% on the detected setting. Though FLaN-Net is
not explicitly designed for holistic ReID tasks, it still guarantees a comparable performance with most of holistic methods,
underscoring its robustness and generalization capabilities.

Table 2:
Performance comparison on Market-1501 and
DukeMTMC-reID. The compared methods are grouped into
two categories: holistic methods and occluded methods.

PFD [Wang et al., 2022a]; DPM [Tan et al., 2022];
SAP [Jia et al., 2023]; OAT [Li et al., 2024].
• Data augmentation methods: OAMN [Chen et al., 2021];
FED [Wang et al., 2022b]; CAAO [Zhao et al., 2023];
ADP [Xia et al., 2024]; DPM-SPT [Tan et al., 2024].
• Language-guided methods: CLIP-ReID [Li et al., 2023].
The experimental results demonstrate that FLaN-Net
achieves outstanding performance on both occluded ReID
datasets. For the Occluded-Duke dataset, FLaN-Net achieves
a Rank-1 accuracy of 75.2% and an mAP of 65.5%, outperforming the second best method, DPM-SPT, by +0.5%
and +2.5%, respectively. On the Occluded-REID dataset,
FLaN-Net achieves the best performance with at least
+3.4% Rank-1 accuracy and +4.4% mAP compared to other
methods. These results indicate that FLaN-Net effectively
addresses the noise caused by occlusions.

4.3

Ablation Study

Ablation Study on Prompt Variations
We design a series of prompts to evaluate their impact on
model performance, as shown in Tab. 4. From index 1 to
2, the addition of subject-specific S ∗ improves Rank-1 accuracy from 72.6% to 73.4% and mAP from 63.7% to 64.4%,
demonstrating the benefit of incorporating identity-specific
information. Expanding the prompts with attribute details
A∗ (index 3) further enhanced Rank-1 accuracy to 74.6% and
mAP to 65.2%, underscoring the importance of fine-grained
identity attributes. Finally, introducing occlusion details O∗
(index 4) yields the highest performance, with Rank-1 reaching 75.2% and mAP 65.5%. These results validate the effectiveness of occlusion-aware fine-grained prompts in guiding
feature extraction under occluded scenarios.

Experimental Results on Holistic ReID Datasets
We also experiment our proposed method on holistic person ReID datasets, including Market-1501, DukeMTMCreID and CUHK03-NP. Tab. 2 shows the results on Market1501 and DukeMTMC-reID datasets. We compare FLaN-Net
with two categories of methods:
• Holistic ReID methods: SAN [Jin et al., 2020]; TransReID; HAT [Zhang et al., 2021]; DCAL [Zhu et al., 2022];
AAformer [Zhu et al., 2023]; PHA [Zhang et al., 2023];
CLIP-ReID.
• Occluded ReID methods: RFCnet; PFD; FED; DPM;
CAAO; HCGA; SAP; ADP; DPM-SPT; OAT.
We observe that our FLaN-Net achieves competitive results on the Market1501 dataset and achieves SOTA performance on the DukeMTMC-reID dataset. Compared with
language-guided method CLIP-ReID, our method surpasses

Ablation Study on Model Components
In Tab. 5, we evaluate the contribution of occlusion-aware
fine-grained prompt (F), cross-attention mechanism (C), and
dynamic weighting fusion module (D) on Occluded-Duke.
From index 1 to 2, the performance improves by +2.9%
in rank-1 accuracy and +1.8% in mAP, demonstrating the
importance of utilizing fine-grained descriptions to capture

806

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Index
1
2
3
4
5

F
✓
✓
✓

C
✓
✓
✓

D
✓

R-1
70.0
72.9
72.1
74.3
75.2

R-5
83.7
85.0
84.3
85.7
86.3

R-10 mAP
88.0 61.6
88.8 63.4
88.4 62.7
89.4 64.9
89.8 65.5

Table 5: Ablation study of occlusion-aware fine-grained prompt (F ),
cross-attention mechanism (C), and dynamic weighting fusion module (D) on Occluded-Duke.

Occluded-Duke
R-1
mAP
Average Weighting 74.6
64.9
74.3
65.3
Summation
Concatenation
73.2
64.2
Dynamic Fusion 75.2
65.5

Figure 3: Impact of learnable query numbers on Occluded-Duke
performance.

DukeMTMC-reID
R-1
mAP
91.8
82.8
91.7
83.1
91.9
83.0
92.1
83.6

(a)

(b)

(c)

(d)

(a)

(b)

(c)

(d)

(a)

(b)

(c)

(d)

(a)

(b)

(c)

(d)

Table 6: Comparison of different fusion methods.

identity-relevant features. From index 1 to 3, cross-attention
also shows its effectiveness. For index 4, combining F and
C further boosts performance by +4.3% in rank-1 accuracy
and +3.3% in mAP, highlighting the necessity of utilizing
fine-grained prompts through cross-attention. Notably, Using fine-grained prompts alone (index 2) or applying crossattention with simple prompts (index 3) proves insufficient
for optimal performance. Finally, from index 4 to 5, the addition of D results in an additional improvement of +0.9% in
rank-1 accuracy and +0.6% in mAP, demonstrating the effectiveness of dynamically fusing features.

4.4

Figure 4: Visualization of attention maps. (a) Input images, (b) PFD,
(c) CLIP-ReID, (d) FLaN-Net.

5

Visualization

To evaluate the model’s ability to handle occluded images,
we visualize attention maps generated by different methods,
as shown in Fig. 4. The PFD is heavily distracted by occlusions, leading to less effective attention. CLIP-ReID demonstrates improved attention but still captures irrelevant areas
in some cases. In contrast, our proposed FLaN-Net focuses
precisely on the visible and identity-relevant regions of the
pedestrian, avoiding interference from occlusions. This visualization highlights the robustness of FLaN-Net in handling
diverse occlusion scenarios and its ability to focus on the most
discriminative features for person re-identification.

Model Analysis

Explore the Optimal Number of Learnable Queries
We investigate the impact of varying the number of learnable
queries on model performance, as illustrated in Fig. 3. The
results indicate that using 3 learnable queries yields the best
performance, with both mAP and Rank-1 reaching their highest values. The performance initially drops as the number of
learnable queries increases from 3 to 5. However, the performance begins to improve again as the number rises from 5 to
7 queries. Despite this, the performance gain remains modest compared to the increase in computational cost. Consequently, selecting 3 learnable queries strikes the optimal balance between performance and computational efficiency.

6

Conclusions

In this paper, we propose FLaN-Net, an innovative framework specifically designed to address the challenges of occluded person re-identification. By employing a categorical attention mechanism, FLaN-Net generates fine-grained
prompts capturing individual descriptions, visible attributes,
and occluding object characteristics. The integration of crossattention mechanisms and a dynamic weighting fusion module enables the model to focus on core identity while mitigating the impact of occlusions. Experimental results across
multiple datasets demonstrate that FLaN-Net achieves stateof-the-art performance, underscoring its robustness and effectiveness in challenging real-world scenarios. This work paves
the way for the development of more advanced occlusionaware ReID solutions and highlights the potential of integrating vision and language for robust identity recognition.

Effectiveness of Dynamic Weighting Fusion Module
To assess the effectiveness of our proposed dynamic weighting fusion module, we compare it against three alternative
fusion methods: average weighting, summation, and concatenation. In the average weighting approach, each of the three
features is assigned an equal weight of 1/3, disregarding their
individual importance. The summation method combines the
three features through element-wise addition, while the concatenation approach merges features along a specified axis.
As shown in Tab. 6, our dynamic fusion method outperforms
all these methods by adaptively assigning weights to each feature based on the entropy of its prediction distribution.

807

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Acknowledgments

[Hermans et al., 2017] Alexander Hermans, Lucas Beyer,
and Bastian Leibe. In defense of the triplet loss for person
re-identification. arXiv preprint arXiv:1703.07737, 2017.

This work was supported by the National Natural Science
Foundation of China under Grant 62472420.

[Hou et al., 2021] Ruibing Hou, Bingpeng Ma, Hong Chang,
Xinqian Gu, Shiguang Shan, and Xilin Chen. Feature completion for occluded person re-identification. IEEE Transactions on Pattern Analysis and Machine Intelligence,
44(9):4894–4912, 2021.

References
[Carion et al., 2020] Nicolas Carion, Francisco Massa,
Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov,
and Sergey Zagoruyko. End-to-end object detection with
transformers. In European conference on computer vision,
pages 213–229. Springer, 2020.
[Chen et al., 2021] Peixian Chen, Wenfeng Liu, Pingyang
Dai, Jianzhuang Liu, Qixiang Ye, Mingliang Xu, Qi’an
Chen, and Rongrong Ji. Occlude them all: Occlusionaware attention network for occluded person re-id. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 11833–11842, 2021.
[Deng et al., 2019] Jiankang Deng, Jia Guo, Niannan Xue,
and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 4690–4699, 2019.
[Ding et al., 2022] Changxing Ding, Kan Wang, Pengfei
Wang, and Dacheng Tao. Multi-task learning with coarse
priors for robust part-aware person re-identification. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 44(3):1474–1488, 2022.
[Dong et al., 2021] Bin Dong, Fangao Zeng, Tiancai Wang,
Xiangyu Zhang, and Yichen Wei. Solq: Segmenting objects by learning queries. Advances in Neural Information
Processing Systems, 34:21898–21909, 2021.
[Dou et al., 2023] Shuguang Dou, Cairong Zhao, Xinyang
Jiang, Shanshan Zhang, Wei-Shi Zheng, and Wangmeng
Zuo. Human co-parsing guided alignment for occluded
person re-identification. IEEE Transactions on Image Processing, 32:458–470, 2023.
[Gal et al., 2022] Rinon Gal, Yuval Alaluf, Yuval Atzmon,
Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel
Cohen-Or. An image is worth one word: Personalizing
text-to-image generation using textual inversion. arXiv
preprint arXiv:2208.01618, 2022.
[Gao et al., 2020a] Lishuai Gao, Hua Zhang, Zan Gao, Weili
Guan, Zhiyong Cheng, and Meng Wang. Texture semantically aligned with visibility-aware for partial person reidentification. In Proceedings of the 28th ACM international conference on multimedia, pages 3771–3779, 2020.
[Gao et al., 2020b] Shang Gao, Jingya Wang, Huchuan Lu,
and Zimo Liu. Pose-guided visible part matching for occluded person reid. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
11744–11752, 2020.
[He et al., 2021] Shuting He, Hao Luo, Pichao Wang, Fan
Wang, Hao Li, and Wei Jiang. Transreid: Transformerbased object re-identification. In Proceedings of the
IEEE/CVF international conference on computer vision,
pages 15013–15022, 2021.

[Jia et al., 2022] Mengxi Jia, Xinhua Cheng, Shijian Lu, and
Jian Zhang. Learning disentangled representation implicitly via transformer for occluded person re-identification.
IEEE Transactions on Multimedia, 25:1294–1305, 2022.
[Jia et al., 2023] Mengxi Jia, Yifan Sun, Yunpeng Zhai, Xinhua Cheng, Yi Yang, and Ying Li. Semi-attention partition
for occluded person re-identification. In Proceedings of
the AAAI Conference on Artificial Intelligence, volume 37,
pages 998–1006, 2023.
[Jin et al., 2020] Xin Jin, Cuiling Lan, Wenjun Zeng, Guoqiang Wei, and Zhibo Chen. Semantics-aligned representation learning for person re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11173–11180, 2020.
[Li et al., 2014] Wei Li, Rui Zhao, Tong Xiao, and Xiaogang
Wang. Deepreid: Deep filter pairing neural network for
person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
152–159, 2014.
[Li et al., 2021] Yulin Li, Jianfeng He, Tianzhu Zhang, Xiang Liu, Yongdong Zhang, and Feng Wu. Diverse part discovery: Occluded person re-identification with part-aware
transformer. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 2898–
2907, 2021.
[Li et al., 2023] Siyuan Li, Li Sun, and Qingli Li. Clipreid: exploiting vision-language model for image reidentification without concrete text labels. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1405–1413, 2023.
[Li et al., 2024] Yanping Li, Yizhang Liu, Hongyun
Zhang, Cairong Zhao, Zhihua Wei, and Duoqian Miao.
Occlusion-aware transformer with second-order attention
for person re-identification. IEEE Transactions on Image
Processing, 33:3200–3211, 2024.
[Miao et al., 2019] Jiaxu Miao, Yu Wu, Ping Liu, Yuhang
Ding, and Yi Yang. Pose-guided feature alignment for
occluded person re-identification. In Proceedings of the
IEEE/CVF international conference on computer vision,
pages 542–551, 2019.
[Radford et al., 2021] Alec Radford, Jong Wook Kim, Chris
Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack
Clark, et al. Learning transferable visual models from natural language supervision. In International conference on
machine learning, pages 8748–8763. PMLR, 2021.

808


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

8122

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

ences and learn invariant representations. Additionally, Teng
et al. [50] and Shi et al. [51] proposed strategies to preserve
cross-modal correlations and used progressive contrastive
learning with dynamic prototypes to establish relationships
between different modalities. GUR [52] first introduced
camera proxies into UVI-ReID, systematically exploring intracamera, inter-camera, and cross-modal domain information
through a bottom-up strategy. DCCL [26] advanced this by
proposing a dual consistency constraint framework combining camera proxies and class constraints, integrating label
refinement with feature interaction learning for improved
performance. However, these methods overlook the training
instability caused by multi-objective optimization, a critical
issue in UVI-ReID. SCA-RCP [53] groups cross-modal proxies of the same identity, it remains an extension of class-level
cross-modal training, making it prone to errors from incorrect
class mappings.
Our method divides contrastive learning into two phases.
The first phase establishes robust intra-modal relationships
through camera proxy-based contrastive learning, effectively
addressing multi-objective training challenges with a novel
camera-specific loss function that surpasses previous multiloss approaches. The second phase integrates camera proxy
information with class-level constraints, generating independent cross-modal mapping graphs at both levels. Furthermore,
we resolve the cluster distribution-DBSCAN preset mismatch [28] in UVI-ReID by refining the memory bank update
strategy.
C. Re-Ranking in Person Re-Identification
Re-ranking is a common post-processing method in image
retrieval, generally divided into two approaches. Learningbased methods [54] incurred a significant amount of additional computation, making non-learning-based methods a
more favorable choice. Recently, MCJA [55] proposed a
multi-spectral enhancement ranking strategy to compute distances between multi-spectral embeddings during testing, but
it could only operate in a supervised environment. Hu et al.
[56] introduced a quantum-inspired interactive ranking aggregation (QI-IRA) method that uses quantum theory to model
and aggregate multiple basic rankings, achieving performance
close to or better than fully supervised methods with lower
labeling costs. However, it still requires labeled data, making it unsuitable for unsupervised scenarios. In unsupervised
settings, complex multi-peak relationships cause neighbor isolation issues. CMRR [57] addressed this with self-modal and
cyclic-modal searching, but these methods increased computational overhead. CMKR [58] used intra-modal and inter-modal
queries to tackle neighbor isolation, further raising computational costs. Zhang et al. [59] proposed a decentralized feature
propagation algorithm for parallel or distributed computing,
extended to cross-camera retrieval with improved feature propagation. However, incorporating camera information to this
feature propagation will added extra time consumption.
To address these issues, we combine camera domain information with extracted features in the post-processing stage,
replacing the simple k-nearest neighbor graph. This creates a

more robust basis for graph propagation and avoids the time
cost of integrating camera information during propagation.
The method can be GPU-accelerated, significantly improving
performance with minimal added computational time.
III. T HE P ROPOSED M ETHODS
In this section, we build upon the baseline framework to
introduce our proposed two-stage training network, CEIL. The
training process is shown in Fig.3, which comprises three
components: the DCMU strategy, the CDL loss, and the PCM
module. In the first stage, we exclusively employ camera proxy
contrastive learning. In the second stage, PCM is integrated to
enable cross-modal feature learning at both the camera proxy
and class levels. Throughout the entire process, we use DCMU
as the memory update strategy and apply CDL loss as the loss
function for the camera proxy. After training, the GCR method
is introduced to further optimize the results.
A. Baseline
As a pioneering pseudo-label-based method in the UVIReID field, ADCA [24] serves as our baseline. To ensure
fair comparison with our CEIL, we exclude the cross-modal
memory aggregation while incorporating Multikernel Maximum Mean Discrepancy (MMD) [60] loss into the objective
function. Unlike DG-VAE [61], which employs distinct feature extractors and encoders for different modalities before
VAE decoding, ADCA demonstrates the redundancy of such
architecture in pseudo-label-based contrastive learning. It only
utilizes modality-specific first layers in the pre-trained network
to mitigate cross-modal discrepancies.
At the beginning of each epoch, features are extracted from
both infrared and visible modal samples using the backbone.
Subsequently, the DBSCAN [62] is applied for clustering these
features. Following the clustering process, the memory for
each cluster is initialized by computing the average feature
values within each respective cluster, as described below:
1 X s
f ,
(1)
m qe =
Cqe s∈Cqe
where e = {v, r } denote the visible and infrared modalities,
respectively. Cqe represents the q-th cluster set under modality
e, | · | denotes the number of instances within each cluster, and
f s represents the features extracted from the image s.
During training, the modality-specific memories are updated
through a momentum update strategy, as follows:
m qe(δ) ← βm qe(δ−1) + (1 − β) f q ,

(2)

where f q represents the features obtained from the training
batch corresponding to m q , β denotes the momentum updating
factor, and δ signifies the iteration number. The loss for each
modality is calculated using the InfoNCE loss [63], as follows:

exp q e · m e+ /τ
e

,
LInfoNCE = − log P e
(3)
N
e
e
p=0 exp q · m q /τ
where m e+ is the memory corresponding to the pseudo label
of the query q e , τ is a temperature parameter, and N e is the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

XIA et al.: CAMERA-PROXY ENHANCED IDENTITY-RECALIBRATION LEARNING FOR UVI-ReID

number of clusters for modality e. The MMD loss, using a
Gaussian kernel, aligns the feature distributions:
 2



(4)
LMMD = Ev φ FvN − Er φ FrN H ,
k

where φ(·) is a feature mapping function, and Hk represents
the Reproducing Kernel Hilbert Space. The image features are
denoted by FvN and FrN . Finally, the total loss combines the
proxy losses of each modality and MMD losses:
Lbase = LvInfoNCE + LrInfoNCE + LMMD .

(5)

8123

can vary significantly due to factors like shooting angles
and lighting conditions. By calculating losses independently
for each camera domain, the model can better handle these
variations, improving its ability to generalize across different
domains. Secondly, selecting negative samples with higher
similarity to the current sample allows the model to learn more
subtle differences. When negative samples are very similar to
positive samples, the model is forced to learn more detailed
features, enhancing its discriminative ability.

B. Camera Domain-Driven Local Association Loss

C. Difficulty-Aware Cluster-Based Memory Updating

The contrastive loss, using one-hot pseudo-labels [35], fails
to accurately adjust relationships between camera proxies
within the same class and those in other classes. PPCL [11]
incorporates camera information to enhance the learning process. For each training sample, it mines credible proxies from
camera agents that do not belong to the same class and are
captured by non-self cameras. However, in cross-modal scenarios, the significant differences within and between modalities
can lead to unstable clustering. When the clustering within
the same camera is unreliable, attempting to perform credible mining across cameras becomes even more impractical.
To address this limitation, we propose the CDL Loss (Lcdl ).
Unlike PPCL, it conducts local contrastive learning within
each camera domain, which avoids interference across camera
domains while reducing the impact of unreliable clustering.
Such a design is particularly well-suited for effectively handling feature learning in multi-camera scenarios.
For a given query feature, similarity is assessed with all
camera proxies in the same class. By calculating the distance
to both positive and negative proxies within each camera
domain separately, reduces disturbances from domain differences and brings camera domains of the same class closer
together. Given a query q, the loss L(q, c) under camera c as
follows:
exp(Sim+ (q, c))
, (6)
L(q, c) =
Pn
+
exp(Sim (q, c)) + i=1
exp(Simi− (q, c))

Traditional momentum update methods facilitate
cross-iteration information exchange by sequentially updating
the corresponding centroids in the memory bank with fixed
weights for each sample, However, which relies too much
on randomness, often neglecting hard samples. A common
issue is that, after multiple iterations, easy samples cluster
well, but hard samples remain problematic. Shi et al. [51]
selected a fixed number of hard samples from all clusters for
updates, which performed poorly in the face of imbalanced
cluster sizes. To address this limitation, we propose the
DCMU strategy. It dynamically selects hard samples within
aggregation clusters, generates weights based on difficulty and
a smoothing factor, and then replaces the cluster centroids
by cluster. As illustrated in Figure 4, this mechanism
forces easy samples to converge towards hard samples,
thereby significantly enhancing the model’s ability to handle
challenging samples.
During forward propagation, DCMU updates in units of
clusters. For the i-th input feature vector xic belonging to
cluster c in the current input batch and the corresponding
cluster center feature fc , the cosine distance is defined as:

where Sim+ (q, c) is the similarity between a query q and its
positive proxy in camera domain c, Simi− (q, c) is the similarity
between q and the i-th nearest negative proxy in this domain,
and n is the number of nearest negative proxies considered
per domain. The positive proxy is the proxy corresponding
to the class of query q within the current camera domain.
The nearest negative proxies are chosen based on the cosine
similarity between query q and other proxies in the camera
domain, with “nearest” meaning sorted in descending order of
similarity. The overall loss Lcdl (q) is obtained by averaging
over all effective camera domains as follows:
1 X
− log (L(q, c)) ,
(7)
Lcdl (q) =
|Cq |
c∈Cq

where Cq denotes the set of valid camera domains for the
query q, and |Cq | represents their number.
Calculating losses based on local negative samples across
different camera domains is crucial. Firstly, it helps maintain
domain consistency. Images from different camera domains

dic = 1 − cos(θ ) = 1 −

xic · fc
,
∥xic ∥∥fc ∥

(8)

where dic is the cosine distance between xic and fc , with ∥ · ∥
denotes the L2 norm. The hard samples for cluster c, denoted
by S c , are selected using the binary indicator F c as follows:
(
1 if di ≥ µc
c
Fi =
,
0 otherwise
Nc

1 X c
µ = c
di ,
N
c

i=1

S c = {s | Fsc = 1},

(9)

where µc represents the mean distance of all N c samples in
the current cluster. Then weights for hard samples wc are
calculated using a softmax and smoothed with a factor α:
w cj = Softmax(d cj ) · α + δargmax(d c ) ( j) · (1 − α),
exp(d cj )
Softmax(d cj ) = P
,
(10)
c
s∈S c exp(ds )
where j denotes the j-th index in S c , and δargmax(d c ) is a binary
list corresponding to d c , with 1 assigned to the maximum
value in d c and 0 to all others. Finally, the pseudo-cluster

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

8124

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Fig. 4. Illustration of the purpose of DCMU. Cluster centers calculated by
means and updated with momentum rely on randomness, leading to unfocused
results. DCMU focuses on difficult samples and uses pseudo-centers to push
nearby samples towards these challenging cases, resulting in tighter clusters.

c

center f pseudo for cluster c is obtained by applying weighted
aggregation to the selected features:
P c c
ws xs
s∈S c
pseudoc
.
(11)
f
=
P c c
ws xs

performance, we propose GCR. The core innovation of it
compared to GCN lies in incorporating camera prior knowledge into the graph construction process and ensuring stability
through multiple symmetric normalization operations during
the iterative phase. Experimental results demonstrate that
GCR achieves significant performance improvements without
substantially increasing computation time. Detailed results are
analyzed in the experiments section.
The original features of the combined query and gallery
sets are denoted by X u , with n images in total. We define a
graph G = (V, E) where each vertex vi in V = {v1 , . . . , vn }
corresponds to an image, and the edges E ∈ Rn×n represent the similarities between these images. Initially, cosine
distances are calculated to establish a similarity ranking R for
the images. Then, E is adjusted based on camera information:


if cami = cam j and j ∈ Ri
 λ,
E i j = 1/λ, if cami ̸ = cam j and j ∈ Ri ,
(14)


0,
otherwise

(13)

where sample j is a top k1 nearest neighbor of sample i, with
cami and cam j as their respective cameras. λ adjusts weights,
emphasizing inter-camera samples when smaller. The ranking
R and the matrix S, which stores the squared similarity scores
of the top k2 neighbors, are saved for use in the second stage.
In the second phase, to enhance retrieval performance, feature aggregation is necessary. Traditional re-ranking methods
use local query expansion [29], while GCR, like GCN [31],
performs iterative propagation on a GNN. Unlike GCN, which
only performs the conversion to an undirected graph and
normalization after initialization, we execute this conversion
process before each iteration, laying a solid foundation for
matrix propagation, as shown below:
 


1
⊤
·
E
+
(E)
,
(15)
Ẽ (0) = diag
∥E + (E)⊤ ∥1

Updating cluster centers using pseudo-centroids achieves
dual objectives. First, computing cosine distances between current samples and hard sample clusters from previous iterations
accelerates the convergence of easy samples toward hard sample regions. Second, formed through weighted aggregations of
historical hard instances, pseudo-centroids naturally lie closer
to hard samples, thereby recalibrating the hardness levels of
previously easy samples. This dynamic tug-of-war process
drives the model to learn noise-resistant feature representations
while reducing sensitivity to outliers.

where E (0) represents the adjacency matrix at the initial
iteration, and its transpose is denoted by (E (0) )⊤ . ∥ · ∥1
denotes the L1 norm, which calculates the sum of the elements
in each row of the matrix. diag(·) converts a vector into a
diagonal matrix, ensuring that each element of the vector
symmetrically scales the corresponding row of the matrix. This
process results in the matrix Ẽ (0) , which is both symmetrized
and normalized, ensuring uniformity and balance in node
relationships. The neighbor aggregation process for updating
the adjacency matrix to E (t) at the t-th iteration is as follows:

s∈S c

During the back propagation phase, the gradient of the loss
function L with respect to the input xic is computed as:
∂L
∂L ∂fc
∂L
= c · c ≈ c · fc ,
c
∂xi
∂f ∂xi
∂f

(12)

∂L
where ∂x
c refer to the gradient of the loss function L with
i
respect to the input feature xic , can be obtained by considering
both the direct gradient ∂∂fLc with respect to the cluster center
feature fc and the gradient of fc with respect to xic , denoted as
∂fc
, which represents fc . After completing the loss computa∂xic
tion, the cluster centers are updated as follows:
c

fc ← f pseudo .

D. Graph Network-Based Camera Constraint Adjustment
Re-Ranking
Re-ranking techniques improve retrieval accuracy by refining initial results. In the UVI-ReID task, traditional re-ranking
methods [29] can effectively enhance performance, but they
often come with high computational costs. Graph Convolutional Network [64] (GNN) based methods, GCN [31],
leverage GPU acceleration to significantly reduce computation
time, yet they do not achieve further performance improvements. To balance low computational cost and high retrieval

E (t) [i, j] =

kX
2 −1



S[i, m] · Ẽ (t−1) R(i, m), j ,

(16)

m=0

where E (t) [i, j] denotes the updated value between nodes i
and j. The aggregation sums over neighboring nodes, where
S[i, m] is the similarity weight between node i and its m-th
neighbor. R(i, m) gives the index of the m-th neighbor of node
i. Ẽ (t−1) [r (i, m), j] retrieves the symmetrized and normalized
adjacency value from the previous iteration. To facilitate
understanding, the complete process of post-processing using
GCR is provided in Algorithm 1.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

XIA et al.: CAMERA-PROXY ENHANCED IDENTITY-RECALIBRATION LEARNING FOR UVI-ReID

Algorithm 1 Re-Ranking Process of the Proposed GCR
Input: The union of query and gallery features X u and their
corresponding camera domains Cu .
Parameter: Hyper-parameters k1 , k2 , and T .
Output: Final distance matrix E.
1: Calculate the similarity matrix S and similarity ranking R
using X u , k1 and k2 .
2: Compute the adjacency matrix E using R and C u according to Eq.(14).
3: Initialize t = 0.
4: while t < T do
5:
Translate the adjacency matrix E into an undirected
graph and perform row normalization, by using Eq.(15).
6:
Update matrix E through message propagation with
GNN using S according to Eq.(16).
7:
Increment t
8: end while
9: return E.

E. Proxy-Enhanced Cross-Modal Mapping
Using only MMD for aligning feature distributions often
results in identity misalignment between visible and infrared
modalities. Previous methods, like PGM [27], address this
by clustering at the class level and establishing mappings
between modalities, but they overlook camera proxy information. To overcome this, we introduce the PCM module in the
second training stage. Building on the foundation of relatively
stable and reliable modal cohesive classes, this module constructs cross-modal correspondences at both the camera proxy
and class levels, significantly enhancing robustness.
First, we construct independent bipartite graph structures
for the proxy clusters and identity clusters obtained from
the infrared and visible modalities. This allows each proxy
cluster and identity cluster within the infrared modality to
find a corresponding cluster within the visible modality. For
the proxy clusters and identity clusters that do not form
pairs in the initial matching, we repeat the above process
until all clusters have corresponding matches. In class-based
matching, PGM has demonstrated that cross-modal bipartite
graph matching is essentially a binary linear programming
problem with linear constraints. The matching cost penalizes
feature differences between matched clusters across modalities, meaning that more similar cluster features result in
lower costs. The proposed PCM, which introduces additional
cross-modal camera proxies, also adheres to this principle.
Following PGM, we utilize the Hungarian algorithm [65] for
matching, and maintain its cost matrix as follows:
1
Cost (i, j) = −
,
exp(Dis(i, j))

(17)

where Cost (i, j) represents the cost function between the
proxies or classes of i and j, and Dis(i, j) denotes the distance measure between them, obtained from our GCR, which
takes additional consideration of camera domain information,
which can provide a more comprehensive measure. In Fig.5
provides a simplified diagram of the mapping process.

8125

Fig. 5.
Illustration of cross-modal mapping in PCM. PCM creates
cross-modal mappings between classes and camera proxies, significantly
boosting error robustness through leveraging consistency between these elements.

p

Secondly, after obtaining the mapping relationships Ti2r ,
between the two domains regarding the
proxy clusters and class clusters, using these mapping relationships to perform modal transformation on the pseudo labels
Pv , Pr , Cv , and Cr obtained through clustering:
p
c , and T c
Tr 2i , Ti2r
r 2i

p

Pv2r = Ti2r (Pv ),
Cv2r = Trc2i (Cv ),

p

Pr 2v = Tr 2i (Pr ),
Cr 2v = Trc2i (Cr ),

(18)

during stage 2, we not only use the pseudo-labels generated
by clustering to train within the same modality but also
utilize the generated mapping pseudo-labels to train across
modalities. Since the incorporation of the class hierarchy, the
calculation of the loss differs from stage 1. To begin with,
we compute the intra-modal loss, which is composed of two
distinct components: The proxy-level loss is performed using
our proposed Lintra
cdl , which utilizes Pv and Pr , while the
class-level loss is accomplished through Lintra
InfoNCE , which takes
Cv and Cr as inputs, as shown below:
intra
Lintra = Lintra
cdl + LInfoNCE ,

(19)

this combination of losses at different levels helps our model
learn better. The calculation for cross-modal loss and intramodal loss is essentially symmetrical, which ensures that the
updates of the infrared and visible light modes are balanced
during training. The difference lies in replacing the class-level
labels with Cv2r and Cr 2v , and the proxy-level labels with
Pv2r and Pr 2v . Algorithm 2 illustrates the operations of our
proposed PCM during the second phase of CEIL.
F. Unified Loss
We use Lcdl and LMMD to define the first stage loss as:
Lstage1 = Lvcdl + Lrcdl + LMMD ,

(20)

where the superscript of the Lcdl indicating its modality. And
define the loss of the second stage as follows:
Lstage2 = Lintra + Lcross ,

(21)

where Lintra and Lcr oss are incorporated into the computations
for both the visible and infrared modalities.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

8126

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Algorithm 2 Training Process for CEIL: Stage 2
Iv

Nv
{xiv }i=1

Ir

Nr
,
{xir }i=1

and
=
Input: Unlabeled image
=
feature extractor E trained after stage 1, memory bank M.
Parameter: Training epochs and iters.
Output: Feature extractor E trained post-stage 2.
1: for i = 1 to epochs do
2:
Use E to extract features from I v and I r , obtain the
feature sets F v and F r .
3:
Apply DBSCAN to cluster the features F v and F r for
each modality Cv and Cr .
4:
Partition the camera proxies into Pv and Pr according
to their respective camera sources.
p
p
c , and T c
5:
Obtain cross-modal mappings Ti2r , Tr 2i , Ti2r
r 2i
between camera proxies and class proxies by PCM.
6:
Obtain cross-modal mapping labels Pv2r , Pr 2v , Cv2r ,
and Cr 2v according to Eq.(18).
7:
for j = 1 to iters do
8:
Update M based on DCMU.
9:
Using P v , P r , Cv and Cr , calculate the intra-modal
loss Lintra according to Eq.(19).
10:
Similarly, using Pv2r , Pr 2v , Cv2r , and Cr 2v , calculate
the cross-modal loss Lcross .
11:
Optimize E to minimize the loss defined in Eq.(21).
12:
end for
13: end for
14: return E.

IV. E XPERIMENTS
A. Experimental Setting
1) Datasets: We evaluate our CEIL using three public ReID
datasets: SYSU-MM01 [14], RegDB [13], and LLCM [15].
The SYSU-MM01 dataset includes 30,071 visible images
and 15,792 near-infrared images captured by 4 visible and
2 infrared cameras. The dataset is divided into a training
set with 395 identities and a test set with 96 identities.
There are two matching modes: “All-search”, which matches
images across all cameras, and “Indoor-search”, which restricts
matches to indoor cameras only. In both modes, infrared
images are matched with visible images. During testing,
one image per identity per camera is randomly selected to
form a gallery set for evaluation. This process is repeated
10 times with different random gallery splits, and the average
performance is reported.
The RegDB dataset comprises 412 identities, each with
10 visible images and 10 infrared images. According to the
protocol by Ye et al. [66], the dataset is evenly split, with half
of the identities used for training and the other for testing.
There are two matching modes: “VIS to IR” for matching
visible to infrared images, and “IR to VIS” for the reverse.
The LLCM dataset, collected under low-light conditions,
includes a training set with 713 identities (16,946 visible images and 13,975 infrared images) and a test set
with 351 identities (8,680 visible images and 7,166 infrared
images). Similar to the RegDB dataset, both “VIS to IR” and
“IR to VIS” modes are employed to evaluate the model’s

performance. The selection of samples during the evaluation
process is similar to that used for the SYSU-MM01 dataset.
2) Evaluation Metrics: We use Cumulative Matching Characteristics (CMC), Mean Average Precision (mAP), and Mean
Inverse Negative Penalty (mINP) [5] as the main evaluation
metrics to validate the effectiveness of our proposed CEIL.
3) Implementation Details: Our model uses the same
feature extractor as CCLNet [42]. Specifically, this feature
extractor is a dual-stream ResNet that has been pre-trained
on ImageNet [67]. All images are resized to 288 × 144 and
using data augmentation such as horizontal flipping, random
erasing, and channel augmentation [16], following the paper
of PGM [27]. Based on the magnitude of the data, we have
selected different types of training minibatch. For the SYSUMM01 and LLCM datasets, we selected 8 identities, each with
10 visible light images and 10 infrared images. For the RegDB
dataset, both the number of identities per modality and the
corresponding number of images are halved. We use warm-up
to update the learning rate from 3.5e-6 to 3.5e-4 during the
first 10 epochs. At the 20th and 50th epochs, the learning rates
are adjusted to 3.5e-5 and 3.5e-6. We employ the Adam optimizer [68] to train for a total of 90 epochs, entering the second
stage after the 50st epoch. The implementation is carried out
in PyTorch and runs on an NVIDIA RTX 3090 GPU.
B. Comparison With SOTA Methods
The results for the SYSU-MM01 and RegDB datasets are
shown in Table I. Meanwhile, the results for the LLCM dataset
are presented in Table II. Among these, CCLNet, PGM, and
SDCL were reproduced based on local experiments due to the
availability of comprehensive code provided for these methods.
In both tables, the symbol “∗ ” indicates methods that have been
post-processed using GCR. The sections above the horizontal
line highlight the performance of supervised learning methods,
whereas the sections below showcase our CEIL method along
with other unsupervised learning approaches.
1) Comparison With VI-ReID Methods: It’s encouraging
that our CEIL, even without post-processing, matches the performance of supervised VI-ReID methods like AGW and PMT.
When enhanced with GCR, it significantly outperforms top
supervised methods on the SYSU-MM01 and RegDB datasets.
The LLCM dataset, which presents challenges such as low
light and climate changes, which difficult for unsupervised
learning. However, with the addition of GCR, CEIL remains
competitive with SOTA models like HOS-Net. Although
there is a noticeable performance gap between CEIL without
GCR and the current state-of-the-art supervised methods on
more demanding datasets, the significant improvement brought
about by post-processing highlights CEIL’s potential and effectiveness. This also emphasizes the critical role of GCR, which
helps CEIL catch up with minimal extra overhead.
2) Comparison With UVI-ReID Methods: Even without our
proposed GCR, the CEIL model we developed significantly
outperforms the current state-of-the-art CMAM model on
the SYSU-MM01, RegDB, and LLCM datasets, achieving a
comprehensive lead. This clearly highlights the effectiveness
of leveraging readily accessible camera information. Obtaining

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

Pattern Recognition 161 (2025) 111301

X. Yang et al.

different classifiers with independent parameters. The classifier then
predicts the identity of each feature vector separately. For each branch,
we calculate the classification loss based on the classifier’s predicted
value and the identity of the input image. The loss of each branch is
used to update the parameters of the corresponding branch during the
training phase.
Fig. 3. Change of channel width in dynamic interaction. 𝑑𝑖 and 𝑑ℎ represent the
input/output channel and hidden channel (M = 3).

3.2. Dynamic aggregation
After extracting features through a dual-stream network, we propose
a dynamic aggregation module to help the current modality absorb
information from the other modality as a way to reduce the difficulty
of inter-modality recognition. Specifically, we employ multi-head attention to enhance intra-modality information, and then design the
dynamic aggregation module to enhance modality representations. Our
structure is designed as a recursive architecture since more dynamic
projections can lead to better interactions. The whole framework not
only learns to extract high-quality visual contextual cues but also
attempts to focus on optimal attention through image representations
based on cross-modality feature guidance. Essentially, our dynamic
interactions define two modalities as primary/auxiliary modalities,
transforming the auxiliary modality from a single dimension to a higher
and broader dimension on which the projection weights required for
the primary modality are based. Since the projection weights for the
information interaction are based on the auxiliary modality features
and not fixed values, this process is referred to as dynamic interaction.
The recursive architecture is designed to accommodate more weight
projections, leading to better information interaction. Dynamic interaction enables direct communication between modalities and adaptively
improves modality representation, thereby reducing the difficulty of
inter-modality recognition.
From the modality perspective, we conduct two interactions: one
where the visible light image serves as the primary modality 𝑋𝑣𝑝 and
the infrared image as the auxiliary modality 𝑋𝑖𝑎 ,; and another where
the infrared image is the primary modality 𝑋𝑖𝑝 and the visible light
image is the auxiliary modality 𝑋𝑣𝑎 Firstly, we apply a multi-head
attention mechanism to enhance the important regions within each
modality, optimizing the attention of features in critical areas for internal information enhancement. The interaction process can be defined as
(𝑋𝑣𝑝 , 𝑋𝑖𝑎 ) and (𝑋𝑖𝑝 , 𝑋𝑣𝑎 ). Secondly, the dynamic aggregation
module projects the primary modality features using the auxiliary
modality information, dynamically generating projection weights based
on the auxiliary modality features. It takes the primary and auxiliary
modality features as input and outputs the primary modality projection
features for one iteration. First, the weights of the dynamic projection
values are generated based on the auxiliary modality features:
 = 𝐿1 (𝑋𝑎 ),

Fig. 4. The new visible features and infrared features generated through dynamic
aggregation.

where 𝑋𝑣𝑛𝑒𝑤 and 𝑋𝑖𝑛𝑒𝑤 stand for the new visible and infrared features
obtained after dynamic interaction. (⋅) represents dynamic interaction
and (⋅) refers to the multi-head attention mechanism.
3.3. Feature intersection
To represent the whole human image, previous VI-ReID methods
adopt GAP or GMP. GAP covers the whole body part of the human
image, but it is easily distracted by background clutter and occlusion.
GMP overcomes this problem by aggregating features from the most
differentiated parts helpful for VI-ReID, while discarding background
clutter. However, GMP does not include information from the whole
body part. For this reason, we propose a simple structure but take full
advantage of GAP and GMP, while also introducing local features. The
difference with GAP, GMP, and (GAP+GMP) is shown in Fig. 5. Here,
we divide into two branches, pooling (GMP) the 18 × 9 global features extracted from the backbone network to obtain an N × 1 feature
(without considering the batch size and channels) and then dividing
it into N times 1 × 1 local features. The other branch is first divided
to obtain N times 18∕𝑁 × 9 local features, and then pooled (GAP) to
get N times 1 × 1 local features. Finally, the local features from the
two branches are concatenated in the channel dimension. It is worth
noting that each of the two branches focuses on different regions. The
first branch first adopts GAP and then divides, so that the obtained
local features synthesize the information of the whole human body and
effectively consider the contextual cue relationships. On the other hand,
the second branch performs GMP after division, which can better focus
on the more discriminative local features in each region.
The whole process can be described as:

(1)

where  represents the weight of the projected value and 𝐿1 (⋅) refers to
the linear layers, which implement the conversion of 1-d features into
2-d matrices. The next dynamic interaction can be defined as follows:
𝑋𝑛𝑒𝑤 = 𝐿2 (𝑋𝑝 ∗ ),

(2)

where 𝐿2 (⋅) represents the multiplication of the two, reverting from 2-d
to 1-d, corresponding to 𝐿1 (⋅).
Recursive structure is adopted to optimize modal feature representation by successive iterations. Notably, 𝑀 recursive blocks of dynamic
interactions are identical except for their channel dimensions. We
create a bottleneck architecture by assigning a hidden channel 𝑑ℎ to
the intermediate dynamic interaction block than the original input
dimension 𝑑𝑖 , as shown in Fig. 3. To finally obtain the new bi-modality
features after the interaction, 𝑀 is set to 2, as shown in Fig. 4. The
interaction process can be defined as:
𝑋𝑣𝑛𝑒𝑤 = ((𝑋𝑣𝑝 , 𝑋𝑖𝑎 )),

(3)

𝑋𝑖𝑛𝑒𝑤 = ((𝑋𝑖𝑝 , 𝑋𝑣𝑎 )),

(4)

𝑋𝑛𝑔𝑎𝑝 = (𝐺𝐴𝑃 (𝑋𝑖 , 𝑋𝑣 )),

(5)

𝑋𝑛𝑔𝑚𝑝 = 𝐺𝑀 𝑃 ((𝑋𝑖 , 𝑋𝑣 )),

(6)

𝑋𝑛 = 𝐶 𝑜𝑛𝑐 𝑎𝑡(𝑋𝑛𝑔𝑎𝑝 , 𝑋𝑛𝑔𝑚𝑝 ),

(7)

where  represents partial division, 𝑋𝑛𝑔𝑎𝑝 is the feature map obtained
by branch 1, 𝑋𝑛𝑔𝑚𝑝 is the feature map obtained by branch 2, and 𝑋𝑛 is
the feature map obtained by splicing two branch channels.
3.4. Multi-weight loss
In the VI-ReID task, each person consists of visible images and
infrared images. To learn with identity attribute information, we propose a new multi-weight loss to reduce the center distance of the
4

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Fig. 5. Comparisons of common feature extraction methods with our FI. The blue and pink rectangles denote the visible and infrared features respectively. (a) GAP. (b) GMP. (c)
GAP+GMP. (D) FI(ours). (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

4. Experiments
We conduct extensive experiments on two publicly available
datasets SYSU-MM01 and RegDB. First, in Section 4.1, two evaluated
datasets are introduced. Then, details on training of the experiments are
described in Section 4.2, followed by an introduction to the evaluation
metrics in Section 4.3. Next, the experimental results are analyzed and
compared to state-of-the-art methods in Section 4.4. Ablation studies
are conducted in Section 4.5, and finally, a comprehensive discussion
of the overall experiments is presented in Section 4.6.
Fig. 6. A simple description about multi-weight loss. The distance between negative
and positive samples represents the degree of similarity between different pedestrians.
The more distant pedestrians are less similar. Hierarchical supervision is achieved by
setting different weights for pedestrians with different similarities. (For interpretation
of the references to color in this figure legend, the reader is referred to the web version
of this article.)

4.1. Datasets
SYSU-MM01 is the first large-scale dataset specifically for VI-ReID
task. It includes six cameras (including four RGB cameras and two IR
cameras) for a total of 491 pedestrians. Cameras 3 and 6 shoots infrared
light in an unlit environment. Cameras 4 and 5 are placed in the two
outdoor scenes of the gate and garden, respectively.
RegDB [44] contains 412 pedestrians, and each person has 20
images, including 10 RGB images and 10 IR images, among which 254
are female and 158 are male. And 156 of the 412 pedestrians are taken
from the front and 256 pedestrians are taken from the back. Following
the previous work [20], we randomly select 206 IDs for training and
the remaining 206 IDs for testing.

same identity feature in different modalities and increase the center
distance of different identity features within the same modality, as
shown in Fig. 6. For increasing the discrepancy of different pedestrians
within a modality, the similarity between different pedestrians is not
the same, which easily causes the pedestrians with higher similarity
to be insufficiently trained. Here, we divide the negative samples 2
levels according to the similarity distance, and different levels are given
different loss weights to flexibly impose different penalties for different
identity center distances.
The whole multi-weight loss can be described as:
‖2
𝑀 𝑊 = ‖
‖(𝐶𝑣 , 𝐶𝑖 )‖2 − 𝜆1 𝑎𝑛𝑘1 − 𝜆2 𝑎𝑛𝑘2 ,

4.2. Implementation details

(8)
Our method is implemented on the Pytorch, using NVIDIA TiTAN
Xp GPU. Following the previous method [10], the images are resized to
384 × 192. We adopt the SGD optimization strategy. The weight decay
is set to 0.0005 and the momentum is set to 0.9. We utilize random
cropping and horizontal flips for data enhancement. Our learning rate
starts at 0.1 using a warm-up strategy. The batch size is set to 64.
We randomly select 4 identities. And each identity contains 4 RGB
images and 4 IR images. Our training epoch is 60. As always, we utilize
ResNet-50 [43] as our backbone pre-trained on ImageNet.
The feature map of the input feature intersection module is partitioned horizontally into 𝑁 = 6 components. Through a linear layer and
batch normalization, the dimension of the feature is reduced to 512,
which is used in 𝑀 𝑊 . After passing the dual-classifier, the dimension
of the feature is reduced to the number of categories, which is utilized
to predict the labels of the images.

where 𝐶𝑣 and 𝐶𝑖 denote visible and infrared modality center. 𝜆1 and 𝜆2
are the weights corresponding to different similarity levels. 𝑎𝑛𝑘1 and
𝑎𝑛𝑘2 represent two levels of ranking based on the similarity between
different pedestrians. 𝑎𝑛𝑘1 typically corresponds to pedestrian pairs
with higher similarity, while 𝑎𝑛𝑘2 corresponds to pedestrian pairs
with lower similarity.
At the same time, the cross-entropy loss is utilized for identity
prediction in the paper. The classification loss 𝑐 𝑙𝑠 is calculated as:
𝑁𝑟𝑔 𝑏+𝑖𝑟
(
)
∑
1
𝑐 𝑙 𝑠 =
𝑐 𝑒 𝑋𝑖𝑟𝑔 𝑏+𝑖𝑟 , 𝑌𝑖𝑟𝑔 𝑏+𝑖𝑟 ,
(9)
𝑁𝑟𝑔 𝑏+𝑖𝑟 𝑖=1
where 𝑁𝑟𝑔 𝑏+𝑖𝑟 denotes all images from visible and infrared modalities,
𝑐 𝑒 is the cross-entropy loss function. 𝑌𝑖 is the ground truth label.
𝑋𝑖 represents the feature vector of the 𝑖th image used for identity
prediction.
5

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Table 1
Comparison of CMC (%) and mAP (%) with the state-of-the-art methods on SYSU-MM01 and RegDB datasets. Our results show the best results in terms of Rank-1 accuracy and
mAP.
Methods

SYSU-MM01

RegDB

All-search

Zero-Padding [32]
D2 RL [33]
Hi-CMD [45]
JSIA-ReID [6]
AlignGAN [5]
XIV-ReID [9]
DDAG [46]
NFS [47]
DML [48]
CM-NAS [49]
FMCNet [14]
SMCL [15]
PMT [50]
DART [35]
MRCN [16]
CAJL [11]
MMN [10]
DCLNet [42]
Ours

Indoor-search

Visible to thermal

Thermal to visible

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

Rank-1

mAP

Rank-1

mAP

14.80
28.90
34.94
38.10
42.40
49.92
54.75
56.91
58.40
61.99
66.34
67.39
67.53
68.70
68.90
69.88
70.60
70.80
71.30

54.12
70.60
77.58
80.70
85.00
89.79
90.39
91.34
91.20
92.87
–
92.87
95.36
96.40
95.20
95.71
96.20
–
95.54

71.33
82.40
–
89.90
93.70
95.96
95.81
96.52
96.90
97.25
–
96.76
98.64
99.00
98.40
98.46
99.00
–
98.78

15.95
29.20
35.94
36.90
40.70
50.73
53.02
55.45
56.10
60.02
62.51
61.78
64.98
66.30
65.50
66.89
66.90
65.30
67.73

20.58
–
–
43.80
45.90
–
61.02
62.79
62.40
–
68.15
68.84
71.66
72.50
76.00
76.26
76.20
73.50
77.16

68.38
–
–
86.20
87.60
–
94.06
96.53
95.20
–
–
96.55
96.73
97.80
98.30
97.88
97.20
–
98.06

85.79
–
–
94.20
94.40
–
98.41
99.07
98.70
–
–
98.77
99.52
99.50
99.70
99.49
99.30
–
99.18

26.92
–
–
52.90
54.30
–
67.98
69.79
69.50
–
74.09
75.56
76.52
78.20
79.80
80.37
79.60
76.80
80.55

16.70
43.40
70.93
48.10
57.90
62.21
69.34
80.54
77.60
84.54
89.10
83.90
84.83
83.60
91.40
85.03
91.60
81.20
94.32

17.90
44.10
66.04
48.90
53.60
60.18
63.46
72.10
84.30
80.32
84.40
79.80
76.55
75.70
84.60
79.14
84.10
74.30
89.02

17.80
–
–
48.50
56.30
–
68.06
77.95
77.00
82.57
88.40
83.10
84.16
82.00
88.30
84.75
87.50
78.00
93.27

18.90
–
–
49.30
53.40
–
61.80
69.79
83.60
78.31
83.90
78.60
75.13
73.80
81.90
77.82
80.50
70.60
88.77

4.3. Evaluation metrics

Table 2
Analysis about the influence of each component in terms of CMC (%) at rank 1 and
mAP (%). ‘‘DA’’ denotes the dynamic aggregation module. ‘‘FI’’ denotes the feature
intersection module. ‘‘MW’’ denotes the multi-weight loss.

The mean Average Precision (mAP) and Cumulative Matching Characteristic (CMC) are utilized in our experiments. The CMC (Rank-k
accuracy) metric quantifies the probability of a correct cross-modality
person image appearing in the top-k retrieved results. On the other
hand, mAP assesses the retrieval performance when multiple matching
images are present in the gallery set.

Baseline

DA

✓
✓
✓
✓
✓
✓

4.4. Comparison with state-of-the-arts
We compare our method with several other methods on SYSUMM01 dataset and RegDB dataset. Among them, there is one basic method (Zero-Padding) [32], five methods based on intermediate
modality (PMT [50], XIV-ReID [9], one method based on dual-attention
mechanism (DDAG [46]), SMCL [15], CAJL [11]), MMN [10], one
method based on noise annotation and correlation (DART [35]), two
methods based on neural architecture search (NFS [47], CM-NAS [49]),
one mutual learning method (DML [48]), two modality compensation methods (MRCN [16], FMCNet [14]), four GAN-based methods
(D2 RL [33], AlignGAN [5], Hi-CMD [45], JSIA-ReID [6]), and one
method based on intensive contrast learning (DCLNet [42]).

FI

MW

Rank-1

mAP

✓

63.59
68.19
65.48
64.36
69.74
71.30

59.34
64.96
61.84
60.23
65.82
67.73

✓
✓
✓
✓
✓

✓
✓

SYSU-MM01

Table 3
Analyze the impact of 𝑑𝑖 and 𝑑ℎ .
SYSU-MM01

𝑑𝑖

𝑑ℎ

Rank-1

Rank-10

Rank-20

mAP

2048
2048

32
64

70.22
71.30

96.01
95.54

99.11
98.78

66.69
67.73

256
512
1024
2048

64
64
64
64

69.97
68.61
69.93
71.30

95.92
95.77
95.76
95.54

98.94
98.81
98.98
98.78

66.79
65.81
66.23
67.73

4.4.2. Results on RegDB datasets
In the same way, we evaluate our method on the RegDB dataset, as
shown in Table 1. We adopt visible-to-thermal and thermal-to-visible
modes. Visible to thermal means that visible images are query images
and thermal images are gallery images. Thermal to visible is just the
opposite. In the visible-to-thermal mode, our method achieves 94.32%
and 89.02% on Rank-1 and mAP, outperforming the state-of-the-art
method MMN by 2.92% and 4.42% on Rank-1 and mAP. In the thermalto-visible mode, our method achieves 93.27% and 88.77% on Rank-1
and mAP, an improvement of 4.89% and 4.91% compared to the
state-of-the-art model FMCNet, respectively. In a nutshell, compared
to other methods, our method achieves excellent results in both modes
according to the evaluation metrics.
In general, through comparisons with various methods on both
datasets, our method has demonstrated its superiority in the VI-ReID
task.

4.4.1. Results on SYSU-MM01 datasets
The comparison of the experimental results of the above methods
is shown in Table 1. In the all-search mode, our method implements
Rank-1 of 71.30% and mAP of 67.73%. Compared to the state-of-the-art
MMN method, our method improves the Rank-1 accuracy by 0.70% and
the mAP by 0.83%. Although DART outperforms our results in Rank10 and Rank-20, the margin is not substantial. On the contrary, we
excel more in mAP and Rank-1. In the Indoor-search mode, our method
achieves 77.16% and 80.55% on Rank-1 and mAP, improving the Rank1 by 0.90% and the mAP by 0.18% compared to the state-of-the-art
CAJL method. MRCN achieved 99.7% in Rank-20, only surpassing us
by 0.52%. Our method improves the performance of Rank-1 and mAP,
validating the effectiveness of the proposed method in fusing useful
information and synthesizing the pooling approach while accurately
distinguishing pedestrians with high similarity.
6

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Table 4
Analysis of the impact of different allocation ratios of 𝜆1 and 𝜆2 on accuracy.
‘999999’ represents the order of sorting similar distances from low to high. The
number of ‘8’ represents the proportion of weights 𝜆1 and 𝜆2 set within the white
‘999999’.
Distribution of 𝜆1

Distribution of 𝜆2

999999
899999
889999
888999
888899
888889
888888

888888
988888
998888
999888
999988
999998
999999

SYSU-MM01
Rank-1

mAP

69.74
69.02
71.30
70.39
68.80
68.91
69.13

65.82
65.39
67.73
66.86
65.54
65.72
66.40

Table 5
Performance comparison of different feature extraction methods in terms of CMC (%)
and mAP (%) on SYSU-MM01. (Setting: Baseline + DA+MW.)

Fig. 7. Analysis about the influence of the weight coefficient 𝜆1 in terms of CMC (%)
and mAP (%) on SYSU-MM01.

Structure

Rank-1

Rank-10

Rank-20

mAP

GAP
GMP
GAP+GMP

68.85
68.89
69.82

95.83
95.99
95.80

98.86
99.12
98.97

65.58
66.00
66.29

FI(ours)

71.30

95.54

98.78

67.73

4.5. Ablation study
Our main contribution consists of three key components: a dynamic
aggregation module (DA), a feature intersection module (FI), and a
multi-weight loss (MW). To evaluate the contribution of each component in our model. We perform an ablation experiment on SYSU-MM01
by introducing each component step-by-step, as shown in Table 2.
It can be seen that the performance is significantly improved by
adding DA to the baseline, which proves that DA can effectively exchange helpful information between modalities and reduce the difficulty of modality recognition. Secondly, the introduction of FI is
beneficial to the network, which proves that the proposed method
can reasonably exploit the advantages of both pooling approaches and
avoid their shortcomings. On this basis, the addition of MW makes the
different pedestrian distances well-balanced and facilitates the network
to extract discriminative features, while making the difficult samples
get stronger supervision.

SYSU-MM01

4.6.3. Hierarchical weights 𝜆1 and 𝜆2
In our setting, we calculate the similarity distances for six different
pedestrians from a batch of four distinct identities. According to Eq. (8),
we have set two levels of weight parameters, 𝜆1 and 𝜆2 and thus
need to divide the six different similarity distances into two groups.
These similarity distances are sorted in ascending order, with smaller
distances (indicating higher similarity) assigned higher weights, and
larger distances (indicating lower similarity) assigned smaller weights.
As shown in Table 4, when 𝜆1 is applied to the first two smaller
distances and 𝜆2 to the remaining four larger distances, the model’s
accuracy reaches its maximum. This suggests that high weights are
not suitable for all pedestrian pairs, which aligns with our previous
assumption that only highly similar pedestrian pairs may not have
received sufficient attention and training during the learning process.
Here, we argue that two different pedestrian pairs in a batch size belong
to hard samples,high similarity pedestrians, and need to attach more
weights.

4.6. Discussions
4.6.1. Channel dimension
We analyze the impact of input and hidden channel widths on
dynamic interactions in terms of performance. Specifically, we explore
their effects by fixing one and adjusting the other in Table 3. The
experiments were conducted based on SYSU-MM01. The results show
that 𝑑𝑖 = 2048 and 𝑑ℎ = 64 significantly obtain more stable accuracy compared to other dimensional combinations. Among them, a
performance advantage is attained in Rank-1 and mAP when 𝑑𝑖 is held
constant and 𝑑ℎ is configured to 64. At the same time, 𝑑ℎ is fixed as 64,
a significant performance is obtained when 𝑑𝑖 reaches the maximum.
We believe that the larger width retains more favorable representations
due to the effect of modality differences, both in the input and hidden
channel dimensions.

4.6.4. Comparison with different feature extraction methods
In this section, we compare the effects of different feature extraction
methods on network performance under the all-search mode on the
SYSU-MM01 dataset. As shown in Table 5, we compare with the common GAP, GMP, and the combination of GAP and GMP(GAP+GMP).
Our proposed feature extraction method gains performance advantages
in terms of Rank-1 and mAP. GAP focuses on the global region and
captures overall features, while GMP focuses on the significant region
and captures local features. We note that both of them are comparable
in performance although each has its own advantages. For the fusion
method GAP+GMP, the results are significantly better than the first
two. However, at the same time, the simple summation weakens GMP’s
focus on significant regions. Therefore, the proposed FI selects the
splicing in the channel dimension well to retain the advantages of both
and outperforms GAP+GMP in terms of performance.

4.6.2. The impact of 𝜆1
In this section, we control the smaller weight 𝜆2 = 0.01 in the
multi-weight loss and explore the effect of the larger weight 𝜆1 . 𝜆1
varies from 0.01 to 0.05 and the experimental results are shown in
Fig. 7. When 𝜆1 = 0.01, it means 𝜆2 = 𝜆1 for the state without setting
hierarchical weights. It can be seen that when 𝜆1 > 0.01, the effects are
all significantly better than the original state (𝜆1 = 0.01), which means
that we give heavier weights to pedestrians with higher similarity by
hierarchical weights, so as to make the distances between different
pedestrians evenly distinguished. If mAP increases as 𝜆1 increases,
it suggests that the weight setting may have a positive impact on
model performance. Conversely, if mAP decreases, it may indicate
that excessively high weights lead to overfitting or other negative
effects. Notably, the trends in Rank-1 and mAP align, with the best
experimental results observed when 𝜆1 = 0.02.

4.6.5. Visualization analysis
We compare the t-SNE visualization results on the baseline and
the proposed method. To ensure fairness, we randomly select several
images of ten identities from all cameras. For each individual, 20
visible images and 20 infrared images are randomly chosen. In the
visualization, different colors represent different identities, and the asterisks denote the person features, with the distances between asterisks
7

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Fig. 8. t-SNE visualization of baseline and the proposed method on SYSU-MM01 dataset. Different colors denote different identities. (For interpretation of the references to color
in this figure legend, the reader is referred to the web version of this article.)

Data availability

representing the two-dimensional feature distances. As shown in Fig. 8,
it can be seen from the results of the benchmark method that the
distance between different image features under the same identity is
far, and there are large modality differences. After the introduction of
the proposed method, the red, blue and other regions are displayed
smaller, and the feature distance of the same identity is significantly
reduced. It is proved that the proposed method reduces the difficulty
of inter-modality recognition, enriches the representation information
within the modes, enhances the overall representation ability of the
modes, and thus reduces the modality differences. The visualization
results confirm the effectiveness of the proposed method.

Data will be made available on request.

References
[1] Y. Cho, W.J. Kim, S. Hong, S.-E. Yoon, Part-based pseudo label refinement for
unsupervised person re-identification, in: CVPR, 2022, pp. 7308–7318.
[2] H. Gu, J. Li, G. Fu, C. Wong, X. Chen, J. Zhu, Autoloss-gms: Searching
generalized margin-based softmax loss function for person re-identification, in:
CVPR, 2022, pp. 4744–4753.
[3] H. Li, M. Ye, C. Wang, B. Du, Pyramidal transformer with conv-patchify for
person re-identification, in: ACM MM, 2022, pp. 7317–7326.
[4] P. Dai, R. Ji, H. Wang, Q. Wu, Y. Huang, Cross-modality person re-identification
with generative adversarial training., in: IJCAI, Vol. 1, 2018, p. 2.
[5] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, Z. Hou, Rgb-infrared cross-modality
person re-identification via joint pixel and feature alignment, in: ICCV, 2019, pp.
3623–3632.
[6] G.-A. Wang, T. Zhang, Y. Yang, J. Cheng, J. Chang, X. Liang, Z.-G. Hou, Crossmodality paired-images generation for RGB-infrared person re-identification, in:
AAAI, Vol. 34, 2020, pp. 12144–12151.
[7] Y. Hao, J. Li, N. Wang, X. Gao, Modality adversarial neural network for
visible-thermal person re-identification, Pattern Recognit. 107 (2020) 107533.
[8] H. Liu, S. Ma, D. Xia, S. Li, Sfanet: A spectrum-aware feature augmentation
network for visible-infrared person reidentification, IEEE Trans. Neural Netw.
Learn. Syst. (2021).
[9] D. Li, X. Wei, X. Hong, Y. Gong, Infrared-visible cross-modal person
re-identification with an x modality, in: AAAI, Vol. 34, 2020, pp. 4610–4617.
[10] Y. Zhang, Y. Yan, Y. Lu, H. Wang, Towards a unified middle modality learning
for visible-infrared person re-identification, in: ACM MM, 2021, pp. 788–796.
[11] M. Ye, W. Ruan, B. Du, M.Z. Shou, Channel augmented joint learning for
visible-infrared recognition, in: ICCV, 2021, pp. 13567–13576.
[12] Y. Ling, Z. Zhong, Z. Luo, P. Rota, S. Li, N. Sebe, Class-aware modality mix
and center-guided metric learning for visible-thermal person re-identification,
in: ACM MM, 2020, pp. 889–897.
[13] Y. Lu, Y. Wu, B. Liu, T. Zhang, B. Li, Q. Chu, N. Yu, Cross-modality person re-identification with shared-specific feature transfer, in: CVPR, 2020, pp.
13379–13389.
[14] Q. Zhang, C. Lai, J. Liu, N. Huang, J. Han, FMCNet: Feature-level modality
compensation for visible-infrared person re-identification, in: CVPR, 2022, pp.
7349–7358.
[15] Z. Wei, X. Yang, N. Wang, X. Gao, Syncretic modality collaborative learning for
visible infrared person re-identification, in: ICCV, 2021, pp. 225–234.
[16] Y. Zhang, Y. Yan, J. Li, H. Wang, MRCN: a novel modality restitution and
compensation network for visible-infrared person re-identification, in: AAAI, Vol.
37, 2023, pp. 3498–3506.
[17] P.K. Sarker, Q. Zhao, Enhanced visible–infrared person re-identification based
on cross-attention multiscale residual vision transformer, Pattern Recognit. 149
(2024) 110288.
[18] S. Wu, S. Shan, G. Xiao, M.S. Lew, X. Gao, Implicit modality knowledge
alignment and uncertainty estimation for visible-infrared person re-identification,
Expert Syst. Appl. 259 (2025) 125291.
[19] M. Ye, Z. Wang, X. Lan, P.C. Yuen, Visible thermal person re-identification via
dual-constrained top-ranking., in: IJCAI, Vol. 1, 2018, p. 2.
[20] M. Ye, X. Lan, J. Li, P. Yuen, Hierarchical discriminative learning for visible
thermal person re-identification, in: AAAI, Vol. 32, 2018.
[21] G. Wang, Y. Yuan, X. Chen, J. Li, X. Zhou, Learning discriminative features
with multiple granularities for person re-identification, in: ACM MM, 2018, pp.
274–282.

5. Conclusion
In this paper, we propose a novel modality information fusion
network to enhance overall modality representation for VI-ReID. First,
to reduce the difficulty of inter-modality recognition, the proposed
dynamic aggregation module re-represents the primary modality features by taking the projections of the auxiliary modality in higher
dimensions as weights. Second, we design a simple but efficient feature intersection module, which takes full advantage of both pooling
approaches and enriches intra-modality information extraction. Finally,
a multi-weight loss is proposed to achieve a good balance of interclass distances through hierarchical supervision, effectively solving the
issue that pedestrian samples with high similarity cannot be adequately
trained. Extensive experiments have shown that the proposed method
outperforms the current state-of-the-art approaches by a significant
margin in visible–infrared person re-identification task.
CRediT authorship contribution statement
Xi Yang: Writing – original draft, Software, Methodology, Funding
acquisition. Huanling Liu: Writing – original draft, Software, Data
curation. Nannan Wang: Writing – original draft, Software, Methodology. Xinbo Gao: Writing – review & editing, Supervision, Methodology,
Funding acquisition.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Acknowledgments
This work was supported in part by the National Natural Science
Foundation of China under Grant 62372348 and Grant 62036007, in
part by the Key Research and Development Program of Shaanxi under
Grant 2024GXZDCYL-02-10, in part by Shaanxi Outstanding Youth Science Fund Project under Grant 2023-JC-JQ-53, in part by the Shaanxi
Province Core Technology Research and Development Project under
grant 2024QY2-GJHX-11, in part by the Fundamental Research Funds
for the Central Universities under Grant QTZX24080 and QTZX23042.
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Fig. 2. Illustration of the proposed framework. At the beginning of each epoch, we utilize a feature extractor network to obtain features for all samples in the dataset and then
assign pseudo-labels to them using DBSCAN. A memory dictionary is built to store a single feature vector for each cluster and is dynamically updated in repeated iterations.
Notably, we construct a camera-aware graph to represent the relationships between instances from multiple camera domains. Then, we design a graph convolutional network
(GCN) is designed to fuse the features into a unified space and learn more crucial information between multiple camera domains with semantic propagation. In this way, the
GCN can implement promising semantic transfer and generate ideal domain-invariant representations for assisting the training. Our model is trained with two contrastive losses
(𝑐 𝑙𝑠 and 𝑔𝑐 𝑛 ) and a camera adversarial loss 𝑎𝑑 𝑣 . The discriminator is trained to identify whether the features are from the feature extractor or the GCN. After training, only the
feature vectors output by the feature extractor are used for the inference.

respectively, and 𝑐𝑖 , 𝑐𝑗 be the corresponding camera IDs. We assign
values to the elements in 𝐴̂ based on identity and camera information:
⎧ 1, 𝑖𝑓 𝑦𝑖 = 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 ≠ 𝑐𝑗 ,
⎪
𝑎̂𝑖𝑗 = ⎨−1, 𝑖𝑓 𝑦𝑖 ≠ 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 = 𝑐𝑗 ,
(7)
⎪
0,
𝑜𝑡ℎ𝑒𝑟𝑠.
⎩

3.4. Camera-aware graph multi-domain adaptive learning
Benefiting from the feature aggregation property, the GCNs can
achieve promising semantic transfer between different camera domains
and exploit the knowledge across multiple related camera domains
simultaneously. Consequently, in this work, we propose the adversarial
training scheme that employs feature aggregation with GCNs to provide ideal feature representations for optimizing the feature extractor
network.
In the 𝑡th iteration of an epoch, we sample 𝑁 𝑏 images from each
camera domain to form a mini-batch which can be formulated as
𝑏
𝑡 = {𝑡1 , 𝑡2 , … , 𝑡𝑉 } = {𝑏𝑡1,𝑖 , 𝑏𝑡2,𝑖 , … , 𝑏𝑡𝑉 ,𝑖 }𝑁
, where 𝑡𝑖 ⊂ 𝑖 . The
𝑖=1
mini-batch of images is first fed-forward into the feature extractor 𝜙 to
obtain the up-to-date features  𝑡 = 𝜙(𝑡 ). We argue that image features
with the same identity across various cameras and different identities
within the same camera contain critical semantic information. In order
to exploit this correlation pattern, we design a heterogeneous graph
to represent the identity and camera domain relationships between
all instances in a mini-batch. Then, a graph convolutional network
(GCN) is designed for fusing the features into a unified space and
learn more crucial information between multiple camera domains with
semantic propagation. In this way, the GCN can implement promising
semantic transfer and generate ideal domain-invariant representations
for assisting the training. Accordingly, our model can adapt to various
camera style differences with leveraging simultaneously the knowledge
across multiple related camera-level sub-domains. Finally, we design an
adversarial training scheme for transferring the knowledge learned by
GCN to the feature extractor.
Specifically, all instances in a mini-batch can construct a directed
graph (, ), where  indicates the set of edges,  denotes the set of
nodes, and each node 𝑣𝑖 ∈  is associated with an image feature. In
practice, the weights of  are represented by a matrix:
𝑎
𝑎
𝐴̂ = (𝑎̂𝑖𝑗 ) ∈ R𝑁 ×𝑁 ,

Then, we add self-connections and normalize the matrix as:
1

1

𝐴 = 𝐷− 2 (𝐴̂ + 𝐼)𝐷− 2 ,

(8)

where 𝐼 denotes the identity matrix, and 𝐷 is the degree matrix.
To generate ideal representations with feature aggregation, we employ the graph convolutional network 𝐺 to update the node features on
(, ). The matrix 𝐴 and the set of nodes  are input into the GCN,
and the output is a new set of node features 𝑡 . The propagation rule
for updating the node features at the 𝑙-layer is defined as,
 (𝑙) = 𝐺(𝑙) ([𝐴 (𝑙−1) ∥  (𝑙−1) ]),

(9)

𝑡 } and 𝑡 =  (𝐿) . 𝐿 is the depth of GCN.
with  (0) =  𝑡 = {𝑓1𝑡 , 𝑓2𝑡 , … , 𝑓𝑁
𝑎
∥ is the concatenation operation. 𝐺(𝑙) is the network with two convblocks of the form Conv-BatchNorm-LeakyReLU. Note that the features
in 𝑡 have the same dimension as the features in  𝑡 .
For training the GCN, we consider maximizing the similarity between instances in the same cluster with InfoNCE loss [22]:
exp(𝑧𝑞 ⋅ 𝑚𝑧𝑞 ∕𝜏2 )
𝑔 𝑐 𝑛 = E[− log ∑𝑁 𝑚
],
(10)
𝑖=1 exp(𝑧𝑞 ⋅ 𝑚𝑖 ∕𝜏2 )

where 𝑧𝑞 ∈ 𝑡 , 𝑚𝑧𝑞 denotes the positive cluster feature vector to query
instance 𝑧𝑞 .
In addition, we design a discriminator network 𝐷 that is optimized
in an alternating manner with 𝜙 to address the adversarial min–max
problem:
min max E𝑧∼𝑝 [log 𝐷(𝑧)] + E𝑥∼𝑝 𝑡𝑟 [log(1 − 𝐷(𝜙(𝑥)))],
𝜙

(6)



𝐷

𝐺(𝜙( 𝑡𝑟 )), and 𝑝

(11)

where  =
 , 𝑝 𝑡𝑟 represent the distribution of the
samples in ,  𝑡𝑟 . Note that we block the gradient propagation between
𝜙 and 𝐺, 𝐺 and 𝐷. We train 𝐷 to maximize the probability of assigning

where 𝑁 𝑎 = 𝑉 × 𝑁 𝑏 represents the size of the mini-batch. Let 𝑦𝑖 , 𝑦𝑗 be
the pseudo-labels of the 𝑖th and 𝑗th image samples in the mini-batch,
4

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Algorithm 1 The Camera-Aware Graph Multi-Domain Adaptive
Learning Framework
Require:
Unlabeled training data  𝑡𝑟 ;
Momentum coefficient 𝛼;
Weight balance parameter 𝜆;
Temperature parameters 𝜏1 , 𝜏2 ;
Number of epochs 𝑁𝑒𝑝𝑜𝑐 ℎ ;
Number of iterations 𝑁𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛 ;
Initialize the feature extractor 𝜙 with ResNet-50 pre-trained on
ImageNet;
1: for 𝑘1 in [1,𝑁𝑒𝑝𝑜𝑐 ℎ ] do
2:
Extract feature vectors  for  𝑡𝑟 with 𝜙;
3:
Perform DBSCAN on  for generating identity pseudo-labels;
4:
Construct and initialize a cluster-level memory dictionary 
with Eq. (3);
5:
for 𝑘2 in [1,𝑁𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛 ] do
6:
Sample 𝑉 × 𝑁 𝑏 images from  𝑡𝑟 to form 𝑘2 ;
7:
Construct 𝐴 with Eq. (8);
8:
Update 𝐺 with Eq. (10);
9:
Update 𝜙 with Eq. (2);
10:
Update  with Eq. (4);
11:
end for
12: end for
13: After training, only 𝜙 is used for the inference.

Table 1
Properties of the three challenging datasets.
Datasets

#images

#IDs

#cameras

Market-1501
MSMT17
PersonX

32,217
126,441
45,792

1501
4101
1266

6
15
6

bounding boxes of 1041 identities for training, 11,659 probe bounding
boxes, and 82,161 gallery bounding boxes of 3060 identities for testing.
PersonX [24] consists of 45,792 synthetic images of 1266 identities
under 6 camera views. The dataset is divided into a training set and a
test set: the former contains 9840 bounding boxes of 410 identities,
while the latter consists of a gallery set and a query set containing
30,816 and 5136 images from 856 identities, respectively.
Evaluation Protocol. For person Re-ID, mean average precision
(mAP) and the Cumulative Matching Characteristic (CMC) curves are
adopted as metrics to evaluate the performance of each algorithm. The
mAP is the mean value of average precision (AP) across all queries,
and the CMC curves represent the probability that a correct match
appears in candidate lists of different sizes. We report the Rank1,
Rank5, and Rank10 accuracy of CMC curves in this paper. For a fair
comparison, we do not utilize any post-processing, such as multi-query
fusion or re-ranking, and all experiments are performed in a single
query setting [1].
4.2. Implementation details

the correct set labels to the feature vectors while training 𝜙 to minimize log(1 − 𝐷(𝜙(𝑥))). In other words, we train the discriminator 𝐷
to discriminate whether the feature vectors come from the extractor
network 𝜙 or the graph convolutional network 𝐺, while 𝜙 is trained to
fool 𝐷. In this way, the distributions of multiple camera domains can
be consistent with the unified distribution learned from the GCN. For
𝜙, the adversarial loss 𝑎𝑑 𝑣 is defined based on the probabilities of the
discriminator 𝐷(𝜙(𝑏)) over one mini-batch as:

The feature extractor in our framework adopts ResNet-50 [2] as the
backbone network, which is initialized with the parameters pre-trained
on ImageNet. Specifically, all modules after layer4 are removed and are
replaced with an average pooling layer, a batch normalization layer,
and an L2 normalization layer, which can produce 1024-dimensional
feature vectors. For the graph convolutional network, we set the depth
of the GCN 𝐿 to 1 in all experiments. The discriminator is a binary
classifier, where the input dimension is 1024, the output dimension
is 2, the dimensionality of two hidden layers is 1024, and the batch
normalization is inserted in convolutional layers. The entire model is
optimized by the Adam optimizer with 𝛽 = (0.9, 0.99). The learning rate
is initialized as 3.5 × 10−4 , 1 × 10−4 , and 3 × 10−4 for the feature extractor
network, discriminator, and GCN, respectively, and then reduced to
one-tenth of the previous value every 20 epochs.
Unless otherwise specified, all the input person images are resized
to 256 × 128. During training, random erasing, random horizontal
flipping, and random cropping are adopted for data augmentation. The
memory dictionary is updated with a momentum coefficient 𝛼 = 0.1.
For DBSCAN, the distance threshold is set to 0.6, and the maximum
distance between two samples is set to 0.4. During the training phase,
we set 𝜏1 = 𝜏2 = 0.05 and 𝑁𝑒𝑝𝑜𝑐 ℎ = 50. Our experiments are conducted
on the PyTorch platform with 2 RTX 3090 GPUs and 1 Intel i9-10980XE
CPU.

𝑏

𝑎𝑑 𝑣 =

𝑁
𝑉 ∑
∑
1
− log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )),
𝑏
𝑉 × 𝑁 𝑐=1 𝑖=1

(12)

where 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )) denotes the probability that the feature vector generated by 𝜙 comes from 𝐺. Notably, we construct the loss based on
− log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )) instead of [1 − log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 ))].
4. Experiments
4.1. Datasets and protocols
To demonstrate the effectiveness and superiority of the proposed
method, we conduct extensive experiments on three large-scale person
Re-ID benchmarks: Market-1501 [23], PersonX [24], and MSMT17 [25].
The properties of all the datasets are summarized in Table 1.
Market-1501 [23] is composed of 32,217 pedestrian images observed under six manually configured cameras for 1501 different identities. All images in the dataset are captured using the Deformable
Part Model (DPM) person detector [23]. We divide the dataset into a
training set containing 12,936 images of 751 identities and a test set
comprising 19,281 of the remaining 750 identities. Further, the test set
is made up of a gallery set containing 15,913 images and a query set
containing 3368 images.
MSMT17 [25] is collected from outdoor and indoor scenarios in the
morning, midday, and afternoon. As a newly released large-scale person
Re-ID benchmark, it is currently the most challenging image-based
dataset. MSMT17 is composed of 126,441 bounding boxes of 4101
identities, which are collected by a fifteen-camera system. Following
the official evaluation protocol, the dataset is divided into 32,621

4.3. Comparison with state-of-the-art methods
In Tables 2–4, we report the performance comparison with the stateof-the-art unsupervised learning and transfer learning approaches on
Market1501, PersonX, and MSMT17 datasets.
(1) Comparison on Market1501 and PersonX: Our proposed CGMAL
achieves 86.8% mAP and 90.5% mAP on Market1501 and PersonX,
outperforming all the competitive methods, including UDA methods:
D-MMD [7], MMCL [8], DG-Net++ [26], ECN+ [5], HCN [19], HCCMMT [20], TAL-MIRN [28], PREST [27], MDJL [3] and MLC [6], and
fully unsupervised methods: MMCL [8], SpCL [13], MC-DSCE [29],
PPLR [32], IIDS [12], AdaMG [34], CCL [22], ISE [21], ACAN [31],
LEL [35], FPLP [36], C3 AB [30], IIDCL [40], ADNMT [37], FCM [38],
5

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Table 2
Comparison with state-of-the-art methods on Market-1501. MMCL* denotes MMCL under the transfer learning setting. UDA means Unsupervised
Domain Adaptation Re-ID, and USL means Fully Unsupervised Re-ID. CP: Camera Prior Information.
Method

Reference

CP

Market1501
mAP

Rank1

Rank5

Rank10

UDA

D-MMD [7]
MMCL* [8]
DG-Net++ [26]
ECN+ [5]
PREST [27]
HCN [19]
HCC-MMT [20]
TAL-MIRN [28]
MLC [6]
MDJL [3]

ECCV2020
CVPR2020
ECCV2020
TPAMI2020
TIP2021
TCSVT2021
TIP2021
TCSVT2022
PR2022
PR2023

×
×
×
×
✓
×
×
✓
×
×

48.8
60.4
61.7
63.8
62.4
70.5
78.9
40.0
65.9
59.8

70.6
84.4
82.1
84.1
82.5
90.7
91.2
73.1
85.6
80.3

87.0
92.8
90.2
92.8
92.1
–
96.7
86.3
93.9
87.4

91.5
95.0
92.7
95.4
94.9
–
97.9
–
96.0
89.9

USL

MMCL [8]
SpCL [13]
MC-DSCE [29]
IIDS [12]
C3 AB [30]
ACAN [31]
PPLR [32]
ISE [21]
TransCL [33]
AdaMG [34]
LEL [35]
CCL [22]
FPLP [36]
RPE [11]
ADNMT [37]
FCM [38]
ACFL [2]
DCReID [39]

CVPR2020
NeurIPS2020
CVPR2021
TPAMI2022
PR2022
TCSVT2022
CVPR2022
CVPR2022
IJCNN2022
TCSVT2023
TIFS2023
TCSVT2023
TIP2023
TMM2023
NeuralNetw2024
AAAI2024
PR2024
CVDL2024

×
×
✓
✓
×
✓
×
×
×
×
×
✓
×
×
✓
×
×
×

45.5
73.1
61.7
78.0
77.0
50.6
81.5
84.7
82.9
84.6
83.4
85.3
85.8
82.4
73.4
83.5
85.3
85.5

80.3
88.1
83.9
91.2
90.0
73.3
92.8
94.0
93.0
93.9
92.9
94.1
94.5
92.6
89.0
93.2
94.3
94.2

89.4
95.1
92.3
96.2
96.4
87.6
97.1
97.8
97.3
97.9
97.1
97.8
97.8
97.1
95.0
96.7
98.0
97.7

92.3
97.0
–
97.7
97.6
91.8
98.1
98.8
98.3
98.9
97.8
98.8
98.7
97.9
96.7
97.6
98.8
98.5

CGMAL(Ours)

This paper

✓

86.8

94.9

98.2

98.9

Table 3
Unsupervised person Re-ID performance comparison with state-of-the-art methods on MSMT17. MMCL* denotes MMCL under the transfer
learning setting. UDA means Unsupervised Domain Adaptation Re-ID, and USL means Fully Unsupervised Re-ID. CP: Camera Prior Information.
Method

Reference

CP

MSMT17
mAP

Rank1

Rank5

Rank10

UDA

D-MMD [7]
ECN+ [5]
MMCL* [8]
DG-Net++ [26]
TAL-MIRN [28]
PREST [27]
HCN [19]
HCC-MMT [20]
MLC [6]
MDJL [3]

ECCV2020
TPAMI2020
CVPR2020
ECCV2020
TCSVT2021
TIP2021
TCSVT2021
TIP2021
PR2022
PR2023

×
×
×
×
✓
✓
×
×
×
×

15.3
16.0
16.2
22.1
14.2
18.5
29.9
28.4
18.0
13.4

34.4
42.5
43.6
48.8
39.0
43.8
58.7
57.4
46.4
34.3

51.1
55.9
54.3
60.9
51.5
57.5
–
70.3
57.9
44.5

58.5
61.5
58.9
65.9
–
63.6
–
75.2
62.7
50.6

USL

MMCL [8]
SpCL [13]
MC-DSCE [29]
IIDS [12]
ACAN [31]
PPLR [32]
ISE [21]
TransCL [33]
AdaMG [34]
LEL [35]
FPLP [36]
ADNMT [37]
DCReID [39]
FCM [38]
ACFL [2]

CVPR2020
NeurIPS2020
CVPR2021
TPAMI2022
TCSVT2022
CVPR2022
CVPR2022
IJCNN2022
TCSVT2023
TIFS2023
TIP2023
NeuralNetw2024
CVDL2024
AAAI2024
PR2024

×
×
✓
✓
✓
×
×
×
×
×
×
✓
×
×
×

11.2
19.1
15.5
35.1
11.2
31.4
35.0
41.3
38.0
42.6
39.5
24.0
33.5
37.2
39.0

35.4
42.3
35.2
64.4
27.1
61.1
64.7
68.6
66.3
68.2
67.9
54.5
62.4
64.5
67.7

44.8
55.6
48.3
76.2
40.9
73.4
75.5
79.3
76.9
77.9
78.0
66.5
71.5
75.0
78.6

49.8
61.2
–
80.5
47.3
77.8
79.4
83.0
80.6
81.4
81.6
70.9
76.9
79.9
82.1

CGMAL(Ours)

This paper

✓

44.1

69.7

79.9

83.2

DCReID [39], TransCL [33], and ACFL [2]. In FCM [38], we set the
backbone to ResNet-50 [2] for a fair comparison.

released dataset, there are fewer methods reported results on MSMT17.

(2) Comparison on MSMT17: Table 3 shows the quantitative results
on a more large-scale and challenging dataset containing complicated
illumination and background variations, i.e., MSMT17. As it is a newly

framework has superiority over the compared methods (unsupervised

The results indicate that the performance of our proposed unsupervised
learning and transfer learning methods). We produce the best performance with rank1 accuracy of 69.7% and mAP score of 44.1%,
6

Pattern Recognition 161 (2025) 111217

Z. Ran et al.
Table 4
Comparison with state-of-the-art methods on PersonX. CP: Camera Prior Information.
Method

Reference

CP

PersonX
mAP

Rank1

Rank5

Rank10

TransCL [33]
AdaMG [34]
CCL [22]
IIDCL [40]
FCM [38]

IJCNN2022
TCSVT2023
TCSVT2023
TOMM2024
AAAI2024

×
×
✓
✓
×

89.2
87.6
90.1
88.4
88.1

95.4
95.0
96.1
95.2
94.8

99.0
98.8
99.0
99.0
98.8

99.4
99.5
99.6
99.6
99.4

CGMAL(Ours)

This paper

✓

90.5

96.1

99.3

99.7

Table 5
Ablation studies on Market1501 and MSMT17. CGMAL1−3 indicate three variants of the proposed framework. ‘Param’ and ‘GFLOPs’ denote the number of parameters and floating
point operations.
Method

Baseline
CGMAL1
CGMAL2
CGMAL3

Setting

Market1501

MSMT17

𝑐 𝑙 𝑠

inter-camera

intra-camera

mAP

Rank1

Rank5

mAP

Rank1

Rank5

✓
✓
✓
✓

×
✓
×
✓

×
×
✓
✓

82.2
86.5
86.3
86.8

92.3
94.7
94.5
94.9

95.7
97.8
97.5
98.2

35.9
43.8
43.5
44.1

61.3
69.2
68.9
69.7

73.8
79.5
79.4
79.9

Param

GFLOPs

23.5M
25.6M
25.6M
25.6M

31.74
31.77
31.77
31.77

Table 6
The results with various architectures on Market1501.

respectively. Compared to ACFL [2], the proposed CGMAL gains 5.1%
mAP and 2.0% rank1 accuracy improvements. Compared with DGNet++ [26], the performance is improved from 22.1% to 44.1% for mAP
and 48.8% to 69.7% for rank1 accuracy, respectively.
(3) Comparison with UDA Approaches: UDA approaches usually rely
on source domain labeled data to train a model. Specifically, D-MMD,
and TAL-MIRN try to align the feature distribution of source and target
domains. MMCL generates pseudo-labels via formulating certain rules,
then train the Re-ID model with those pseudo-labels. DG-Net++ performs image-to-image style translation based on Generative Adversarial
Networks (GANs), converting the source images to the target style.
In contrast, without any identity annotation, our proposed method is
superior to all of them.
(4) Comparison with USL Approaches: Under the purely unsupervised
setting, our method achieves the best performance on the two datasets
in Table 2. Specifically, CGMAL obtains rank1 accuracy = 94.9%
and mAP = 86.8% on Market-1501, and achieves rank1 accuracy =
69.7% and mAP = 44.1% on MSMT17. Compared with the secondbest method [35], CGMAL obtains clear gains of 2.1% and 1.5% for
mAP on Market-1501 and MSMT17, demonstrating the advantage of
the proposed learning framework. Notably, our method significantly
outperforms SpCL, which is based on proxy contrastive learning.
(5) Comparison with Transformer-Based Approaches: CGMAL achieves
comparable or superior performance with those transformer-based approaches [33]. In fact, our backbone network is considered the fundamental component. The superiority may be caused by multi-domain
feature transfer with semantic propagation. Therefore, we believe this
research will inspire transformer-based work.
(6) Comparison with GCN-Based Approaches: Our proposed approach
also suppresses existing GCN-based methods [19,20]: 14.2% mAP and
11.0% rank1 accuracy improvement on MSMT17. This demonstrates
the superiority of our camera-aware graph and adversarial training
scheme.

Architecture

Method

Market1501
mAP

Rank1

GoogLeNet+GCN

Baseline
CGMAL

78.2
82.6

86.3
89.7

DenseNet-121+GCN

Baseline
CGMAL

79.8
83.5

89.4
92.3

MobileNetV2+GCN

Baseline
CGMAL

80.7
84.9

91.2
93.6

ResNet-50+GCN

Baseline
CGMAL

82.2
86.8

92.3
94.9

ResNet-50+GAT

Baseline
CGMAL

82.2
87.1

92.3
94.7

Table 7
The results with different clustering algorithms on Market1501. SP and AHC indicates
spectral clustering and agglomerative hierarchical clustering, respectively.
Method

KMeans (k = 500)
KMeans (k = 750)
KMeans (k = 1000)
AHC
SP
DBSCAN

Market1501
mAP

Rank1

77.9
78.2
77.5
74.3
79.8
82.2

85.1
86.5
86.4
85.9
90.1
92.3

4.4. Ablation studies

proposed Camera-Aware Graph Multi-Domain Adaptive Learning (CGMAL) assisted model, which are named CGMAL1−3 .
(1) Effectiveness of the Inter-Camera Correlation: We investigate the
effect of inter-camera feature aggregation. For CGMAL1 , we adapt
the assignment rule of 𝐴̂ in the proposed full model (CGMAL3 ) as
follows:
{
1, 𝑖𝑓 𝑦𝑖 = 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 ≠ 𝑐𝑗 ,
𝑎̂𝑖𝑗 =
(13)
0, 𝑜𝑡ℎ𝑒𝑟𝑠.

In this section, we conduct the experiments by evaluating the performance contribution of different model settings on the three datasets
to investigate the effectiveness of the proposed approach. The experimental results of each setting are summarized in Tables 5–7.
We first utilize 𝑐 𝑙𝑠 to train the feature extractor with the clusterlevel memory dictionary and use it as our baseline. As shown in
Table 5, the baseline method achieves competitive performance. Specifically, the baseline of mAP for Market-1501 and MSMT17 are 82.2%
and 35.9%, respectively. Then, we investigate three variants of the

On Market-1501, CGMAL1 obtains an obvious improvement with the
mAP/Rank1 accuracy increasing from 82.2%/92.3% to 86.5%/94.7%
compared to the baseline. On MSMT17, CGMAL1 provides
+7.9%/+7.9% for mAP/Rank1 accuracy improvements compared to
baseline. The performance gain demonstrates that the knowledge across
multiple related camera domains can boost the representation learning
for unsupervised person Re-ID.
(2) Effectiveness of the Intra-Camera Correlation: We further analyze
the effect of intra-camera feature aggregation. Similarly, the assignment
7

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Fig. 3. Analysis of hyper parameters on Market-1501. From left to right: (a) The impact of 𝜆 in Eq. (2). (b) The impact of 𝛼 in Eq. (4). (c) The impact of the batch size.

rule for 𝐴̂ is adjusted to:
{
−1, 𝑖𝑓 𝑦𝑖 ≠ 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 = 𝑐𝑗 ,
𝑎̂𝑖𝑗 =
0, 𝑜𝑡ℎ𝑒𝑟𝑠.

(14)

As shown in Table 5, CGMAL2 presents superior performance compared
with the baseline method. Furthermore, we can observe a performance
degradation when removing the intra-camera feature aggregation from
the full model (CGMAL3 ), validating its effectiveness.
(3) Effectiveness of the Camera-Aware Graph Multi-Domain Adaptive
Learning: The proposed full model combines both inter-camera and
intra-camera feature aggregation. Experimental results show that the
model using only inter-camera feature aggregation performs better than
when using only intra-camera feature aggregation. Particularly, combining the two improvements provides better results than when each
improvement is used separately. Fig. 4 shows the Re-ID performance
during iterations. This indicates that the proposed approach gradually learns discriminative feature representations from multi-domain
style images. Moreover, several retrieval results for the baseline and
CGMAL are visualized in Fig. 6. It shows that the baseline is hard
to distinguish target objects affected by the cross-camera scene variations. Conversely, with the help of camera-aware graph multi-domain
adaptive learning, our method can identify them successfully. Notably, CGMAL increases negligible computational cost, which is mainly
caused by the graph convolutional network and discriminator.
(4) The Generalization of CGMAL Across Various Architectures: As
shown in Table 6, our proposed CGMAL is not limited to specific
CNN Backbone. For instance, CGMAL can boost the Rank1 accuracy of
GoogLeNet baseline from 86.3% to 89.7% on Market1501. Moreover,
we adopt GAT [1] to assign attention weights for neighbor nodes
and aggregate information from them. GAT achieves comparable or
superior performance with GCN. Those results demonstrate that our
CGMAL has strong generalization ability on various architectures.
(5) Comparison Among Different Clustering Algorithms: In order to
compare the performance of different clustering algorithms, we leverage the baseline model with four clustering algorithms to perform
Re-ID. As shown in Table 7, DBSCAN outperforms the remaining clustering algorithms by a large margin and can get rid of the dependence
on manually setting the cluster number. Therefore, we choose DBSCAN
as our default clustering algorithm.

Fig. 4. The mAP and Rank1 accuracy on Market-1501.

Fig. 5. The performance of CGMAL with various L for training on Market1501.

4.5. Parameter analysis
In this section, we investigate four important hyper-parameters,
i.e., the loss weight 𝜆 in Eq. (2), the momentum updating coefficient
𝛼 in Eq. (4), the batch size 𝑁 𝑎 , and the depth of GCN 𝐿. In order to
clearly analyze the effect of each parameter, we evaluate one parameter
and fix the other ones. Experiments for all parameters are evaluated on
the Market-1501 dataset.
(1) Loss Weight 𝜆: Fig. 3(a) shows how the hyper-parameter 𝜆, the
weight that determines the importance of 𝑐 𝑙𝑠 and 𝑎𝑑 𝑣 in the overall
loss 𝑒𝑥 , affects the person Re-ID performance. We can observe that the
performance first shows an upward trend as 𝜆 increases. Our method

Fig. 6. Visualization of retrieval results for the baseline and our CGMAL.

8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

4

(a) Framework of the CCIL

(b) Confounder Dictionary

Two-stream Network

Disentangle
Regularization

𝒙𝒊

𝑳𝒑𝒐

𝑭𝒄𝒍𝒕

𝑳kl

𝒇𝒄𝒍𝒕

𝑭𝒄𝒍𝒕

𝒇𝒄𝒍𝒕

Confounder
Dictionary

…

𝑴𝒄𝒍𝒕

𝑳𝒇𝒐

𝒄𝟏

𝒄𝟐

Intervention Module
Inference

Distance

𝝁𝒄𝒋

Sampling
𝑵(𝝁𝒄𝒋 , 𝝈𝟐𝒄𝒋 )

𝒇𝒄𝟏𝟏 ~𝒇𝒄𝒏𝑲

𝝍𝟏 ⋅

𝑭𝒊𝒎𝒈

𝒇𝒊𝒎𝒈

(d) Disentangle Regularization
Feature
center

element-wise add

Feature

Σ weighted sum

𝑷(𝒄𝒋𝟏 )~𝑷(𝒄𝒋𝑲 )

Causal
Classifier

…

𝒇𝒊𝒎𝒈

…
Update

element-wise product

𝝍𝟐 ⋅
Σ

𝑷(𝒄𝒏 )

…

ID-Clothe
Classifier

𝑷(𝒄𝟏𝟏 )~𝑷(𝒄𝒏𝑲 )

…

𝒄n

ID

𝑷(𝒄𝒋 )

Softmax

𝑷(𝒄𝟐 )

𝑴𝒊𝒎𝒈

Backdoor
Adjustment

(c) Intervention Module
𝒇𝒄j𝟏 ~𝒇𝒄𝒋𝑲

𝑷(𝒄𝟏 )

𝟏 − 𝑴𝒊𝒎𝒈

𝑭𝒊𝒎𝒈

𝑳int

𝒄𝟐

𝝁𝒄𝒍𝒕

𝒄𝒏

𝝈𝟐𝒄𝒍𝒕

𝒇𝒊𝒎𝒈

𝒄𝟏

𝑳int

𝑃(𝑌|𝑑𝑜 𝑋 )

𝒇𝒄𝒍𝒕

𝑳𝒑𝒐
𝑳𝒇𝒐
𝑳kl

Pull
close

𝒇𝒊𝒎𝒈

Push
away

Fig. 3. (a) The framework of the proposed Causal Clothes-Invariant Learning (CCIL). Three causal modules are complementary to each other and train
the model under the causal intervention framework, achieving better clothes-invariant features. (b) The Confounder Dictionary models clothing to support
the implementation of the subsequent causal intervention. (c) The Intervention Module collaborates with the established Confounder Dictionary to derive the
P (Y |do(x)). (d) The Disentangle Regularization improves the modeling of clothing in the Confounder Dictionary, thereby further enhancing the effectiveness
of the Intervention Module.

• During training, a given image would be fed into a twostream network to obtain image features and clothing features.
The clothing features are utilized to establish a Confounder
Dictionary iteratively (Sec. IV-A).
• With this, the Confounder Dictionary stores richer information about each cloth, which then interacts with the image
features to model causal intervention probability P (Y |do(X))
(Sec. IV-B).
• To further make the modeled intervention P (Y |do(X)) more
accurate, three losses are proposed to achieve clothes and identity disentangling, which improves the modeling of clothing in
the Confounder Dictionary and enhances the effectiveness of
the Intervention Module (Sec. IV-C).
The aforementioned three modules collectively implement
the causal intervention during training, leading to discriminative clothes-invariant feature learning.
A. Confounder Dictionary
We construct a Confounder Dictionary D to represent all
Nclt pieces of clothing in the training set, supporting the implementation of causal interventions in subsequent Equation 8.
As shown in Figure 3 (b), the Confounder Dictionary models
each clothing item ci as a multivariate Gaussian distribution
N (µci , σc2i ), where µci ∈ Rd and σc2i ∈ Rd represent the
mean and variance, respectively. Modeling the distribution of
each clothing item can significantly reduce storage overhead
compared to storing the clothing features of all images, and
it also provides more information than simply storing the

mean clothing features. Moreover, the Confounder Dictionary
also stores the probability of each clothing, denoted as P (ci ),
which can be calculated in the training set. The Confounder
Dictionary can be defined as follows:
clt
D = {(N (µci , σc2i ), P (ci )}N
i=1 .

(3)

2
of clothing
We calculate the mean µclt and variance σclt
features fclt within the training batch.
X
1
µclt =
fclt ,
N (c = ci ) c=c
i
(4)
X
1
2
(fclt − µclt )2 ,
σclt
=
N (c = ci ) c=c
i

where N (c = ci ) represents the number of samples with
clothing label ci in each training batch. Subsequently, based on
the clothing labels, we update the relevant µci and σc2i items of
the Confounder Dictionary by the exponential moving average
(EMA) scheme:
µci = α · µci + (1 − α) · µclt ,
2
σc2i = α · σc2i + (1 − α) · σclt
,

(5)

where α denotes the memory coefficient.
In addition, the clothing features fclt ∈ Rd used to construct
the Confounder Dictionary are extracted through spatial attention, which distinguish them from identity features fimg ∈ Rd :
fclt = P ool(Fclt ⊙ M1 (Fclt ) ⊙ (1 − M2 (Fimg ))) ,
fimg = P ool(Fimg ⊙ M2 (Fimg ) ,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

(6)

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

5

where Fclt ∈ Rd×h×w and Fimg ∈ Rd×h×w are the clothing
feature map and identity feature map output by the two-stream
network. M1 and M2 are implemented by a 2D convolutional
layer with an output dimension of 1, followed by sigmoid
activation. ⊙ denotes the element-wise product, P ool denote
the pooling operation applied along the spatial axes. This
strategy of multiplying the clothing feature map by the reverse
attention mask of the identity feature aids in accurately modeling confounders, ensuring that subsequent causal interventions
do not mistakenly eliminate identity-related information.
B. Intervention Module
To eliminate the interference of the clothing shortcut, an
intervention loss Lint is constructed by maximizing the intervention probability:

where pjk represents the sampling probabilities of different
fcik , which is computed by the probability density function
(PDF) of the multivariate Gaussian distribution:
K

X
1
1
exp(− DM (µcj , fcjk )),
pjk = exp(− DM (µcj , fcjk ))/
2
2
k=1

DM (µcj , fcjk ) = (fcjk − µcj )Σ−1 (fcjk − µcj )T ,
(11)

where DM (·, ·) is the Mahalanobis distance, Σ ∈ Rd×d
denotes the diagonal covariance matrix with diagonal elements
equal to σc2j .
To enhance the computational efficiency of Equation 8,
we utilize the Normalized Weighted Geometric Mean
(NWGM) [45] to approximate moving the classifier out of
accumulation operation:
X
P (Y |do(X = xi )) =
Cls[g(fimg , N (µcj , σc2j ))] · P (cj ),
cj

Lint = E [−log (P (Y = yi |do(X = xi ))] .

(7)

The intervention reflects the causality between variables X
and Y , uncorrelated with the confounder, which is consistent
with the clothes-invariant objective.
To achieve the intervention optimization in Equation 7, we
design an Intervention Module to implement the P (Y |do(X))
well within the CC-ReID framework.
As shown in Figure 3 (c), the Intervention Module takes the
image feature fimg of image xi and the Confounder Dictionary
D as its inputs, and then calculates the intervention probability
based on backdoor adjustment theory [10], [11] that we review
in Equation 1. With this module, the intervention probability
is calculated as follows:
X
P (Y |do(X = xi )) =
P (Y |X = xi , cj ) · P (cj )
cj

=

X

Cls[g(fimg , N (µcj , σc2j ))] · P (cj ),

cj

(8)
where N (µcj , σc2j ) and P (cj ) are achieved in the Confounder
Dictionary. Cls(·) = Sof tmax(Linear(·)) is a classifier
defined as a linear layer followed by a softmax activation
function. g(·) is used to produce conditional probability
P (Y |X, cj ) based on cj . Inspired by condition injection
methods [43], [44], we design g as follows:
g(fimg , N (µcj , σc2j )) = fimg ⊙ ψ1 (µcj ) + ψ2 (µcj ),

(9)

where ⊙ denotes the element-wise product, ψ1 and ψ2 are two
independent linear layers.
Equation 9 only utilizes the mean representation µcj of each
clothing item cj as a condition, which overlooks the variations
in clothing representation arise from changes in lighting, viewpoint and other factors in different environments. Therefore,
we sample K diverse features [fci1 ...fciK ] for each clothing
item from the distribution N (µcj , σc2j ) to comprehensively
account for the influence of clothing conditions. The improved
g(·) is represented as follows:
g(fimg , N (µcj , σc2j )) =

K
X
k=1

X
≈ Cls[
g(fimg , N (µcj , σc2j )) · P (cj )].
cj

(12)
This scheme reduces multiple classifications to a single operation, which significantly simplifies the computation of
intervention probability.
In conclusion, the Intervention Module can eliminate the influence of the clothing shortcut by simulating causal intervention P (Y |do(X)), ensuring clothes-invariant feature learning.
C. Disentangle Regularization
To achieve the expectation of modeling the intervention
probability P (Y |do(X)) by a deep learning implementation,
it is crucial to ensure the validity of Equation 8 as much as
possible. Therefore, as shown in Figure 3 (d), we designed
a disentangle regularization to ensure that the clothing representations are discriminative and do not contain identity clues.
This allows for the causal intervention to accurately cut off
the clothing shortcut without compromising the extraction of
identity features.
We propose a clothes-identity probabilistic orthogonal loss
Lpo to replace the original classification loss. Specifically, we
construct an ID-Clothes classifier with a total of Nclt + Nid
categories, where Nid represents the total human identity
categories and Nclt denotes the total clothing categories.
Subsequently, the image feature fimg and the clothing feature
fclt are fed into the ID-Clothes classifier to get the predicted
probability distributions:
pimg = Sof tmax([W1 ; W2 ]fimg ),
pcls = Sof tmax([W2 ; W1 ]fclt ),

where W1 ∈ Rd×Nid and W2 ∈ Rd×Nclt are the weights of
the classifier, and [·; ·] denotes concatenation. Finally, clothesidentity probabilistic orthogonal loss Lpo minimizes the crossentropy of the predicted probabilities and the identity labels
or clothing labels:
Lpo = E[−yi log(pimg )] + E[−ci log(pclt )].

(fimg ⊙ ψ1 (fcjk ) + ψ2 (fcjk ))pjk ,
(10)

(13)

(14)

This strategy results in clothes and images becoming negative categories of each other, with their features belonging to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

6

different spaces. Consequently, the clothing features contain
distinct information from the image features, facilitating better
disentanglement.
To further disentangle clothing and identity, we employ an
additional KL-divergence loss Lkl :
′
′
Lkl = E[Dkl (Pimg ||Pimg
) + Dkl (Pclt ||Pclt
)],

(15)

where Dkl (·||·) denotes the KL divergence. Pimg is computed
by averaging the classification probabilities pimg from half of
the samples with the same identity label in the training batch,
′
while Pimg
denotes the average classification probabilities
′
from the other half of the samples. Pclt and Pclt
has a similar
meaning and calculation pipeline. So this loss term means that
the classification probabilities among positive samples should
be as similar as possible and reduce the interference of noisy
samples through probability averaging.
In addition, we design a metric learning loss, clothesidentity feature separation loss Lf s , that directly constrains
the Euclidean distance between features:
a
p
a
n
Lf s = E[Deu (fm
, fm
)] + E[ρ − Deu (fm
, fm
)]+ ,

(16)

where Deu (·, ·) is Euclidean distance, [·]+ = max(0, ·), ρ is
the margin parameter. And m ∈ {img, clt}, fm represents the
image/clothing feature centers from half of the samples with
the same identity/clothing label in the current mini-batch. f p
is the feature center of the positive samples for f a and f n is
the negative one.
Specifically, this loss aims to cluster positive features
closely together and distance negative features from each other.
Similar to our proposed Lpo loss, the Lf s loss conducts feature
learning in a shared metric space, where clothes and images
serve as negative categories for each other, emphasizing their
distinction. This strategy further enhances that the clothes
features should have different information from the image
ones, leading to better disentanglement.
D. Optimization
The whole model is trained end-to-end and the total loss
Ltotal of our method is defined as:
Ltotal = Lint + Lpo + Lkl + Lf s ,

(17)

We uniformly assigned a weight of 1 to all losses, without the
need for hyper-parameter searches, which has yielded good
results.
In addition, considering that the Confounder Dictionary may
not be accurate in the early stages of training, we do not apply
the intervention loss Lint during the warmup stage (first 10
epochs) to stabilize the training.
V. E XPERIMENTS
We focus on evaluating the CC-ReID performance of CCIL
under multiple scenarios, verifying the effectiveness of causal
intervention in suppressing the clothing shortcut, and assessing
the robustness of CCIL under noisy clothing labels. The Supplementary Material reports additional analyses on complexity,
occlusion, and other settings.

TABLE I
B RIEF STATISTICS OF DATASETS USED IN THIS WORK .
Dataset

Source

Images

IDs Cameras Cloth-Labels

PRCC
VC-Clothes
LTCC
DeepChange
LaST
Celeb-reID-light

Surveillance 33,698
221
Synthetic
19,060
512
Surveillance 17,119
152
Surveillance 178,407 1,121
Movie
228,156 10,862
Internet
10,842
590

3
4
12
17
-

!
!
!
%
!
!

A. Datasets and Evaluation Protocol
Dataset Details. For primary evaluation, we use six widely
used public CC-ReID datasets: PRCC [8], VC-Clothes [46],
LTCC [7], DeepChange [9], Celeb-reID-light [15] and
LaST [16]. Table I gives brief statistics of the datasets used
in this work. The PRCC dataset is collected from 3 cameras and each identity has 2 pieces of clothing. It contains
33,698 images from 221 identities. The VC-Clothes dataset
is a synthetic CC-ReID dataset rendered by the GTA5 game
engine. It contains 512 virtual identities of 19,060 images in 4
different cameras. The LTCC dataset contains 17,119 images
of 152 identities captured by 12 cameras. The DeepChange
dataset is a large-scale long-term ReID dataset, which consists
of 178,407 images of 1,121 identities from 17 cameras and
only includes the recording date without clothing labels. The
LaST dataset is also a long-term ReID benchmark collected
from more than 2,000 movies in 8 countries, containing 10,862
identities and 228,156 images. The Celeb-reID-light dataset
is collected from snapshots of celebrities on the Internet. It
contains 10,842 images of 590 identities. In this dataset, each
clothing item has only one image, so image IDs can be used
as clothing IDs.
Evaluation Protocol. Our experiments follow the evaluation protocol in existing CC-ReID benchmarks. The Rank-k
accuracy and mean average precision (mAP) are adopted as the
evaluation metrics. We employ a cloth-changing evaluation
setting for PRCC, VC-Clothes, and LTCC datasets. In this setting, the clothing between each individual’s query and gallery
images is different. We also report the result under the clothunchanging setting for PRCC and VC-Clothes datasets, which
means images are all cloth-consistent for each identity. For
the PRCC dataset, we follow [25], [47] and report results on
cameras A/C for cloth-changing and A/B for cloth-unchanging
settings. For the VC-Clothes dataset, we follow [25], [47] and
report results on cameras 3/4 for cloth-changing and 2/3 for
cloth-unchanging settings. For the LTCC dataset, the accuracy
is calculated only using cloth-changing ground-truth samples
in the cloth-changing setting [19], [30]. For the DeepChange
dataset, we follow [9], [25] and use true matches from different
times and trajectories. For Celeb-reID-light and LaST, we
follow their official standard ReID evaluation protocols.
B. Implementation Details
Network. We utilize the ResNet-50 model [48] as the
backbone of our two-stream network. Following the widely
used Re-ID methods [49] the last convolutional stride is set to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

7

1 and the BNNeck is added. The first bottleneck of ResNet is
designated as the shared module, while other bottlenecks are
configured as branch-specific modules. We also employed the
ViT-Base model [50] pre-trained on ImageNet with patch size
16, as the backbone of our two-stream network to further test
the generalizability of our method. The first four layers of ViT
are designated as shared modules, while the other layers are
configured as branch-specific modules.
Training. The model is trained for 120 epochs with the
SGD optimizer. In the first 10 epochs, the learning rate linearly
increased from 0.001 to 0.01 for the ResNet-50 backbone
and increased from 0.0008 to 0.008 for the ViT backbone.
Afterwards, the learning rate was decayed to 0 following a
cosine decay schedule. The batch size is set to 64 with 8
identities and 1 or 2 clothing items. Following [7], [25], [26],
[30], images are resized to 384×192 and are augmented with
random horizontal flipping, padding, random cropping, and
random erasing [51] in training. Following [7], [18], [22], [25],
[26], [30], clothing labels are used during training. The hyperparameter α in Equation 5 is set to 0.9, K in Equation 10 is
set to 4, and ρ in Equation 16 is set to 0.6 for the ResNet-50
backbone and set to 0.9 for the ViT backbone.
Inference. In the inference stage, only the identity feature
extraction stream is activated to extract fimg as a human
signature, and we directly measure cosine similarities across
images to obtain the retrieval results. The clothing stream and
causal modules are used only for training, so their additional
cost does not affect inference; detailed parameter and runtime
comparisons are provided in the Supplementary Material.

shortcuts, while other causality-based ReID methods fail to
do so. Method [27] employs counterfactual interventions to
optimize the total direct effect (TDE) of attention maps.
This approach attempts to eliminate all potential spurious
correlations in spatial attention without specifically addressing
the influence of clothing, thus failing to effectively eliminate
clothing shortcuts. TDE-based methods, such as AIM [30],
highlight clothes-unrelated features within individual images
through factual-counterfactual comparison. However, they do
not explicitly adjust the clothing distribution associated with
each image. In contrast, for each clothing item, backdoor
adjustment blocks the shortcut path X ← C → Y by replacing
P (C|X) with P (C). Since clothing shortcuts in CC-ReID
mainly stem from distribution-level clothing-identity spurious
correlations in the training data, this global adjustment directly
addresses the source of the bias and therefore suppresses such
shortcuts in a more targeted manner.

C. Comparison with Causality-based ReID Methods

D. Comparison with State-of-the-art Methods

We compared CCIL with causality-based ReID methods [27], [30] on the PRCC dataset. Following [36], [52],
in addition to common accuracy metrics, we introduced the
neural causation coefficient (NCC) to quantitatively verify the
effectiveness of approaches in eliminating clothing shortcuts.
NCC is a causal discovery model pretrained on synthetic observation samples that can directly operate on feature vectors
to evaluate causal relationships in visual images. Specifically,
we input identity features and clothing features into the pretrained NCC model to obtain the NCC(C → X) score, which
ranges from (0,1) and represents the relative causality intensity
from clothes C to image X. Due to factors such as blurriness,
occlusion, and variations in lighting, some samples exhibit
insignificant clothing shortcuts prior to intervention. Therefore,
we report the average values of the NCC(C → X) score for
the top 10% samples to highlight the methods’ effectiveness in
mitigating clothing shortcuts. Causal interventions can cut off
the clothing shortcut by disrupting the potential correlation
between the intervention variable X and its cause C, as
illustrated by the removed arrow C → X. Consequently, a
lower NCC(C → X) indicates a more effective prevention of
interference caused by clothing shortcuts.
As shown in Table II, our method achieves the best
NCC scores and Rank-1 accuracy, significantly outperforming
the baseline and other causality-based ReID methods. This
suggests that our approach effectively eliminates clothing

We compare the performance of CCIL with state-of-the-art
methods on PRCC, VC-Clothes, LTCC, DeepChange, LaST
and Celeb-reID-light datasets in Table III, Table IV and
Table V. The PRCC dataset provides standardized evaluation,
while VC-Clothes, LTCC, and DeepChange datasets respectively consider the effectiveness of the methods under the
conditions of clothing sharing, some pedestrians not changing
outfits, and the absence of clothing labels. LaST and CelebreID-light further evaluate generalization in non-surveillance
scenarios.
Results on Ideal Conditions. PRCC is a standard CC-ReID
dataset, in which each identity is associated with two sets
of clothing and appears in all cameras, making it well-suited
for investigating the pure impact of clothing. As depicted in
Table III, CCIL achieves 66.4% Rank-1 accuracy and 65.2%
mAP accuracy on the PRCC dataset. Compared with the methods using auxiliary modality data, our method surpasses HPRNet [23] by a large margin, with 4.1% absolute improvement
in Rank-1 accuracy, along with 5.1% enhancements in mAP
accuracy. Although these existing methods incorporate clothesinvariant modality information such as poses and contours,
most of them use the RGB modality as the main input, resulting in the persistence of clothing shortcuts that interfere with
feature learning. Our method achieves superior results without
the need for additional modality data. Compared with the
methods using RGB modality only, our method outperforms

TABLE II
C OMPARISON WITH CAUSALITY- BASED R E ID METHODS ON THE PRCC
DATASET. T HE NEURAL CAUSATION COEFFICIENT (NCC) IS USED TO
QUANTITATIVELY VERIFY THE EFFECTIVENESS OF APPROACHES IN
ELIMINATING CLOTHING SHORTCUTS . T HE TERMS ”TDE” AND ”BA”
REFER TO TOTAL DIRECT EFFECT AND BACKDOOR ADJUSTMENT.

Method

Causal Technology

NCC↓

Rank1↑

CAL [27]
AIM [30]

TDE (Counterfactual)
TDE (Counterfactual)

0.32
0.26

55.0
57.9

baseline
CCIL (Ours)

No Causal
BA (Intervention)

0.34
0.13

54.1
66.4

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

8

TABLE III
C OMPARISON WITH THE STATE - OF - THE - ART METHODS ON THREE CC-R E ID DATASETS . T HE TERMS ” POSE ”, ”2D”, AND ”3D” REFER TO HUMAN
POSES , 2D SILHOUETTES , AND 3D SHAPE INFORMATION , RESPECTIVELY. ”-” DENOTES THAT THE ORIGINAL PAPER WAS NOT REPORTED .
PRCC
Method

Venue

Modality

VC-Clothes

LTCC

Cloth-changing

Cloth-unchanging Cloth-changing Cloth-unchanging Cloth-changing

Rank1

mAP

Rank1

mAP

Rank1

mAP

Rank1

mAP

Rank1

mAP

SPT+ASE [8]
GI-ReID [20]
UCAD [53]
3DSL [18]
FSAM [19]
DCR-ReID [22]
CCPG [54]
HPRNet [23]

TPAMI’2019
CVPR’2022
IJCAI’2022
CVPR’2021
CVPR’2021
TCSVT’2023
CVPR’2024
TCSVT’2025

Contour
RGB+2D
RGB+2D
RGB+Pose+2D+3D
RGB+Pose+2D
RGB+Parsing+Contour
RGB+2D
RGB+Parsing

34.4
37.6
45.3
51.3
54.5
57.2
61.8
62.3

57.4
58.3
60.1

64.2
80.0
96.5
98.8
100.0
100.0
100.0

99.7
99.6
99.6

64.5
82.4
79.9
78.6
-

57.8
73.8
81.2
78.9
-

92.6
94.7
-

81.1
94.8
-

23.7
32.5
31.2
38.5
41.1
46.2
45.9

10.4
15.1
14.8
16.2
20.4
22.9
19.2

IANet [55]
mAPLoss [16]
CAL [25]
ACID [56]
AIM [30]
CCFA [26]
Instruct-ReID [57]
FIRe2 [58]

CVPR’2019
TCSVT’2021
CVPR’2022
TIP’2023
CVPR’2023
CVPR’2023
CVPR’2024
TIFS’2024

RGB
RGB
RGB
RGB
RGB
RGB
RGB
RGB

46.3
57.5
55.2
55.4
57.9
61.2
54.2
65.0

46.9
54.7
55.8
58.3
58.4
52.3
63.1

99.4
100.0
99.1
100.0
99.6
100.0

98.3
99.8
99.9
98.7
99.5

81.4
84.3
82.1
89.7
-

81.7
74.2
81.9
78.9
-

95.1
95.1
95.0
-

95.3
94.7
95.1
-

25.0
40.1
29.1
40.6
45.3
44.6

12.6
18.0
14.5
19.1
22.1
19.1

CCIL

Ours

RGB

66.4

65.2

100.0

99.2

89.6

88.2

96.0

95.7

46.0

22.2

TABLE IV
C OMPARISON WITH THE STATE - OF - THE - ART METHODS ON THE
D EEP C HANGE DATASET WITHOUT CLOTHES LABELS .
DeepChange
Method
BoT ResNet-50 [59]
ReIDCaps [15]
ViT B16 [50]
SCNet [60]
CAL [25]
IMS+GEP [61]
baseline (Ours)
CCIL (Ours)

Rank1

Rank5

Rank10

mAP

47.5
44.3
49.7
53.5
54.0
55.1
53.6
59.2

59.5
56.4
61.8
64.9
64.2
69.5

65.2
62.0
67.4
69.6
69.3
74.3

13.0
13.3
15.0
18.7
19.0
18.3
17.2
20.8

TABLE V
C OMPARISON WITH THE STATE - OF - THE - ART METHODS ON THE L A ST
AND C ELEB - RE ID- LIGHT DATASETS .
LaST
Method

Celeb-reID-light
Method

Rank1

mAP

Rank1

mAP

mAPLoss [16]
IMS+GEP [61]
CAL [25]
FIRe2 [58]
baseline (ours)

71.0
73.2
73.7
75.0
72.8

28.0
29.8
28.8
32.2
27.9

mAPLoss [16]
RCSANet [47]
CAL [25]
3DInvarReID [62]
baseline (ours)

29.0
29.3
33.6
37.0
31.5

16.3
16.7
18.5
21.8
17.0

CCIL (Ours)

76.8

32.0

CCIL (Ours)

38.5

22.4

the second-best method FIRe2 [58] by 1.4% and 2.1% in Rank1 and mAP accuracy. From our results, it can be seen that it
is still challenging for the likelihood-based methods to drop
clothes-related cues while keeping the identity discriminative,
further proving the effectiveness of our causality-based work.
Furthermore, under the cloth-unchanging setting, our approach achieves the best Rank-1 and comparable mAP accuracy among all the state-of-the-art methods. The cloth-

unchanging setting is based on the strong assumption that
people keep their clothes unchanged, allowing the utilization
of clothing information to bring gains. Our method aims
to capture reliable identity cues while excluding unreliable
clothing information, enabling generalization across diverse
scenarios, regardless of whether the clothing is changed.
Results with Shared Clothing Conditions. VC-Clothes
is a synthetic CC-ReID dataset in which a small number
of clothing items are shared, resembling the low-probability
scenario in the real world where two individuals wear the same
outfit. Since most clothing is not shared, the clothing shortcuts
for inferring identity based on clothing will still significantly
interfere with the learning of clothes-invariant features. Furthermore, shared clothing does not affect the effectiveness of
the Confounder Dictionary and the Intervention Module, as
it equates to storing multiple representations for a clothing
item. As shown in Table III, CCIL achieves the best mAP
and Rank-1 accuracy among all the state-of-the-art competitors
on the VC-Clothes dataset under both the cloth-changing and
cloth-unchanging settings. The results indicate that our method
remains effective in the presence of clothing sharing. We also
point out that if the majority of clothing is shared among most
identities, the clothing shortcuts become very weak, although
this is unlikely to occur. In this extreme case, P (Y |do(X))
remains the correct objective, but the improvement effect is
limited, as P (Y |X) is already very close to P (Y |do(X)).
Results with Limited Clothing Diversity. LTCC is a small
CC-ReID dataset that contains a total of 77 identities in the
training set, with 31 identities having only one set of clothing.
As depicted in Table III, CCIL achieves better or comparable
performance compared to FIRe2 [58] and CCFA [26] on the
LTCC dataset. Compared with AIM [30], CCIL also achieves
higher Rank-1 and mAP accuracy on the Cloth-changing
setting of PRCC. We further illustrate the reasons for the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

Pattern Recognition 174 (2026) 112947

J. Peng et al.

LReID aims to sequentially train these data domains and maximize their
generalization ability across all tasks. The 𝑡-th stage dataset 𝑋 𝑡 consists
of the training set and testing set. Notably, the classes of the training set
and testing set are non-overlapping. Only the training set is available
during the 𝑡-th training. LReID evaluates all the testing sets for each data
domain after ﬁnishing all the training processes of the training set. Since
| 𝑡 |
the subsequent work is based on the training phase,  𝑡 = {(𝑥𝑡𝑖 , 𝑦𝑡𝑖 )}𝑖=1
mainly represents the training set of the current data domain in this
paper, where | 𝑡 | is the number of the training set.
Following the baseline method, the reference models and data replayer are also employed. To be speciﬁc, data replayer randomly selects
and stores a small number of samples based on person identity from the
training data, which is formulated as:
 𝑡+1 ←  𝑡+1 ∪ {(𝑥𝑡1 , 𝑦𝑡𝑗 ), (𝑥𝑡2 , 𝑦𝑡𝑗 ), .(𝑥𝑡𝑝 , 𝑦𝑡𝑗 )}𝐶
,
𝑗=1

transfer. The framework mainly consists of two modules: Transition Factors Estimation (TFE) and Knowledge Integration Module (KIM).
The lower left part of Fig. 2 illustrates the process of the TFE. It uses
hidden layer feature maps from both the current and reference models
as input, and transition factors are obtained through a series of convolution and pooling operations. Especially, given the input image 𝑥𝑡 , the
deep feature maps extracted by the residual block of models are fed into
transition factors estimation:
ℎ𝑘 = 𝜑𝑘 (𝑥𝑡 ),

where the deep feature maps ℎ𝑘 are extracted by residual block 𝜑𝑘 , 𝑘 ∈
{𝑛, 𝑜} represents the branch of the current model and reference model,
respectively. To fetch the domain style from the deep feature maps, the
TFE employs shared 1 × 1 convolution and 3 × 3 convolution block to
reﬁne common characteristics, which can be formulated as:

(1)

where 𝐶 represents the total number of identities in the current data,
and 𝑝 denotes the number of samples randomly selected per identity.
The data replayer is merged into the new task dataset  𝑡+1 and fed into
the model for joint training in the subsequent task.
In addition, the model retains diverse information about identities
after the training of the current datasets, which holds vital importance
for knowledge recall in subsequent training sessions. Therefore, the current model parameters are frozen and saved as a reference model for the
next task:
𝜃𝑛𝑡+1 ← 𝐹 𝑟𝑜𝑧𝑒𝑛(𝜃𝑜𝑡 ),

(3)

𝛾 𝑘 = 𝐶𝑎𝑡(𝐶𝑜𝑛𝑣1×1 (ℎ𝑘 ), 𝐶𝑜𝑛𝑣3×3 (ℎ𝑘 )),

(4)

where 𝐶𝑜𝑛𝑣1×1 and 𝐶𝑜𝑛𝑣3×3 represent the 1 × 1 convolution and 3 × 3
convolution operation, respectively, and 𝐶𝑎𝑡 denotes the matrix concatenation operation, 𝑘 ∈ {𝑛, 𝑜}. By reducing the dimensionality through
convolution at two scales, a more informative feature is obtained for
both the current model and the reference model. Furthermore, the fused
style feature 𝛾 𝑚 can be obtained adaptively by adding the style feature
of the current model and the reference model, which can be represented
as:

(2)

𝛾𝑚 = 𝛾𝑛 + 𝛾𝑜

where 𝐹 𝑟𝑜𝑧𝑒𝑛 represents the frozen the parameter, 𝜃𝑛𝑡+1 is the parameters
of reference model for the task 𝑡 + 1 and 𝜃𝑜𝑡 is the parameters of current

(5)

In order to obtain informative transition factors for diﬀerent models, the fused style feature 𝛾 𝑚 are fed into pooling layer and multi-layer
perception (MLP), which is formulated as:
[ 𝑛 ]
𝛼
= 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝑀𝐿𝑃 (𝑃 𝑜𝑜𝑙(𝛾 𝑚 ))),
(6)
𝛼𝑜

model for the task 𝑡.
3.2. Overview
To improve the generalization and anti-forgetting ability of the
LReID model, this paper proposes an Adaptive Knowledge Transition
(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
be summarized as Fig. 2. Its training phase is mainly separated into two
situations: 𝑡 = 1 and 𝑡 > 1.
For the ﬁrst task 𝑡 = 1, person images are randomly sampled from
the current task data based on identity and fed into the model. After extracting features from the person images by the model, the Re-ID losses
are computed and backpropagated. Finally, the model parameters are
updated and trained with a speciﬁed number of epochs until convergence.
In other situations, for the remaining tasks 𝑡 > 1, samples are ﬁrst extracted based on identity from the current task data and data replayer.
Then they are fed into both the current model and the reference model
to extract deep feature maps. Transition factors are obtained by fetching
the style for diﬀerent domains with deep feature maps in the Transition
Factors Estimation (TFE). Combined with transition factors, the Knowledge Integrate Module (KIM) merges the features and constructs the
intermediate transition knowledge, which is performed as the bridge to
facilitate the knowledge transition. In Multi-Relation Distillation (MRD),
Centroid Bias Distillation (CBD) and Instance Interaction Distillation
(IID) are constructed to explore latent relationships from diﬀerent perspectives, which further enhance knowledge consolidation. The above
steps are continuously iterated until ﬁnishing training all the tasks to
remain a discriminative and anti-forgetting LReID model.

where 𝑀𝐿𝑃 represents the multi-layer perception and 𝑃 𝑜𝑜𝑙 denotes the
average pooling layer. 𝛼 𝑛 , 𝛼 𝑜 ∈ ℝ represent the transition factors from
the current model and the reference model, respectively.
Combined with the transition factors, the knowledge integration
module generates intermediate futures to further promote the knowledge transfer. To be speciﬁc, it takes the one-dimension feature of the
model as input, which is represented as:
( )
𝑓 𝑘 = 𝐺𝑘 𝑥𝑡 ,
(7)
where 𝑘 ∈ {𝑛, 𝑜} represents the branch of the current model and the reference model and 𝑓 𝑘 ∈ ℝ𝑑 is the one-dimension features from the current model 𝐺𝑛 and the reference model 𝐺𝑜 . 𝑑 is the dimension of the feature vector extracted by deep neural model. Based on transition factors,
the knowledge integration module generates the intermediate feature
𝑓 𝑚 ∈ ℝ𝑑 by multiple and added operations, which can be formulated
as:
𝑓 𝑚 = 𝛼𝑛 ⊙ 𝑓 𝑛 + 𝛼𝑜 ⊙ 𝑓 𝑜 ,

(8)

where ⊙ represents the vector-matrix multiplication of the vectors. The
intermediate feature 𝑓 𝑚 encompasses the domain styles of both the current model and the reference model, providing crucial cues for knowledge accumulation and refresh. By constructing simulated distribution,
the knowledge can adaptively transfer between models, which eﬀectively weakens the inhibition of generalization capacities to new tasks.
3.4. Multi-relation distillation

3.3. Knowledge transition learning

To further enhance the knowledge consolidation of previous
identities, the Multi-Relation Distillation (MRD) is designed in this section. It aims to explore the latent relationships across domains from different perspectives, which exhibit greater relevance and are less prone
to be forgotten. The two main relationships are included: (1) Instance
Interaction Distillation (IID) and (2) Centroid Bias Distillation (CBD),
which are shown in Fig. 3

To conduct the adaptive transfer knowledge, Knowledge Transition Learning (KTL) is proposed in this section. In the context of
two vastly distinct knowledge domains, directly transferring knowledge
prone to be ineﬃcient and uncontrollable. Therefore, KTL adaptively
constructs a simulated distribution to achieve a progressive knowledge
4

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Fig. 3. Illustration of Centroid Bias Distillation (CBD) and Instance Interaction Distillation (IID). (b) shows the previous samples and centroids have a shift when
expanding to new tasks. While (a) illustrates the centroids of the previous task are employed to narrow the intra-class distance in CBD and (c) presents the relation
of previous samples deployed to enhance the knowledge consolidation in IID.

Feature-based knowledge distillation only emphasizes the consistency of features on the same sample across models. However, the latent
relationships between samples are more informative for knowledge distillation. Analogous to the human brain, we usually construct basic skills
with the relationships between objects to generalize new tasks. Therefore, the IID is proposed to recall previous identities by exploring the
relationship between samples. It ﬁrstly construct the feature matrix as
input:
𝐹 (𝑘) = [𝑓1𝑘 , 𝑓2𝑘 , … , 𝑓𝐵𝑘 ] ∈ ℝ𝐵×𝑑 ,

the centroid and samples to establish the constrain of knowledge refresh
for the previous identities:
𝐿𝑐𝑏𝑑 = 𝐷𝐾𝐿 (𝑆 𝑛 ∕𝜏||𝑆 𝑚 ∕𝜏) + 𝐷𝐾𝐿 (𝑆 𝑚 ∕𝜏||𝑆 𝑜 ∕𝜏).

The centroid bias distillation loss eﬀectively mitigates the disruption to the feature representation of previous samples, increasing the
ability of the model to generalization to remaining tasks. By building
the above distillation constraints through simulated features, the model
signiﬁcantly promotes knowledge preservation while mitigating the inhibition of the generalization to new tasks.

(9)

where 𝐵 is the number of person images in a batch, while 𝑑 denotes the
dimension of the feature vector. The feature matrix 𝐹 (𝑘) ∈ {𝑛, 𝑜, 𝑚} contains a batch of features from the branch of the current model, reference
model, and KTL, respectively. To capture the relationships between features, cosine distance is utilized in this paper to calculate the similarity
matrix:
𝑇

𝑉 𝑘 = 𝐹 (𝑘) 𝐹 (𝑘) ,

3.5. Optimization
During the training of the ﬁrst task, the optimization objective of the
model is to maximize its discriminative ability for person Re-ID. At this
stage, this paper mainly employs cross-entropy loss 𝐿𝑐𝑒 , triplet loss 𝐿𝑡𝑟 ,
and contrastive loss 𝐿𝑐 to constrain the training of the model.

(10)

𝐿𝑟𝑒𝑖𝑑 = 𝐿𝑐𝑒 + 𝐿𝑡𝑟 + 𝐿𝑐 .

where the 𝑉 𝑘 ∈ ℝ𝐵×𝐵 , which contains the similarity between samples
within a batch. IID then utilizes the similarity matrix to build the constrain for knowledge review of previous tasks, which is formulated as:
𝐿𝑖𝑖𝑑 = 𝐷𝐾𝐿 (𝑉 𝑛 ∕𝜏||𝑉 𝑚 ∕𝜏) + 𝐷𝐾𝐿 (𝑉 𝑚 ∕𝜏||𝑉 𝑜 ∕𝜏),

(14)

When training the remaining tasks, to prevent catastrophic forgetting
caused by excessive learning, the optimization objective of model must
incorporate mechanisms for reviewing previously learned knowledge.
Therefore the knowledge refresh loss is employed, which is formulated
as:

(11)

where 𝐷𝐾𝐿 represents the Kullback-Leibler Divergence [33], an asymmetric measure of diﬀerence between two probability distributions. 𝜏 is
the temperature parameter used to control the smoothness of the distribution. Diﬀerent from identity relationships, the relationships between
samples exhibit more information and eﬀectively mitigate catastrophic
forgetting in LReID tasks.
In addition to utilizing relation distillation for intra or interrelationships, this paper also proposes CBD, which considers the relationship from centroids. When the model is extended to new tasks, the
weights of the classiﬁer inevitably are biased towards the new task due
to the unbalance of samples between current and previous identities.
It may disrupt the feature representation and intra-class compactness of
the previous classes. Therefore, the CBD aims to reduce the centroid bias
of the previous class while expanding to new tasks. Given the feature 𝑓 𝑘 ,
the distance from the centroid is calculated by cosine similarity:
𝑆 𝑘 = 𝑐𝑗 𝑇 𝑓 𝑘 ,

(13)

𝐿𝑟𝑒𝑓 𝑟𝑒𝑠ℎ = 𝐿𝑐𝑏𝑑 + 𝐿𝑖𝑖𝑑 ,

(15)

where 𝐿𝑐𝑡 is centroid bias distillation loss calculated by Eq. 13 while 𝐿𝑟𝑒
is instance interaction distillation loss derived from Eq. 11. Finally, the
overall loss function can be described as:
𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑟𝑒𝑖𝑑 + 𝐿𝑟𝑒𝑓 𝑟𝑒𝑠ℎ .

(16)

4. Experiment
4.1. Datasets and evaluation metric
To verify the performance of our method with LReID tasks, this paper
conduct extensive experiments in six datasets: VIPeR [34], Market-1501
[35], Cuhk-Sysu [36], MSMT17 [37], Cuhk03 [38] and DukeMTMC
[39]. Table 1 shown the detailed information of these six benchmark
person Re-ID datasets. There are two challenging training orders are
employed in the proposed method, including Order-1: VIPeR → Market1501 → Cuhk-Sysu → MSMT17 and Order-2: DukeMTMC → MSMT17 →
Market-1501 → Cuhk-Sysu → Chuk03.

(12)

where 𝑐𝑗 is the mean vector of the feature 𝑓 𝑘 stored in the memory bank,
and 𝑗 indicates the identy of 𝑓 𝑘 . While 𝑆 𝑘 are the distances between the
centroid and the features, 𝑘 ∈ {𝑛, 𝑜, 𝑚}. CBD employs similarity between
5

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Table 1
Detail information of four benchmark datasets. “Ident.”
represents the number of identities, while “C/V” denotes
the number of cameras or views.
Datasets

Year

Identities

C/V

Samples

VIPeR [34]
Market-1501 [35]
DukeMTMC [35]
Cuhk-sysu [36]
MSMT17 [37]
Cuhk03 [38]

2007
2015
2017
2017
2018
2014

632
1501
1404
8432
4101
1467

2
6
8
2
15
2

1264
32,217
36,411
18,184
36,441
13,164

mental learning such as LwF [42], iCarL [43], UCIR [13], BiC [44] and
WA [14] have demonstrated advanced performance on small datasets
like ImageNet. These methods adopt data partitioning to frame the task
as class-incremental learning within a single dataset. In contrast, LReID
focuses on continual learning across datasets, making it inherently more
challenging. Therefore, the performance of these methods remains moderate when tested on person Re-ID datasets, with Rank-1/mAP reaching
only about 60 %/50 %. This phenomenon highlights the uniqueness of
the LReID task.
To address the catastrophic forgetting issue in LReID tasks, other
methods like AKA [5], PTKP [6], and KRKC [30] are proposed to explore further approaches. AKA [5] and PTKP [6] aim to utilize the relational graph matrix to improve knowledge accumulation and distillation
through graph convolution. While these methods provide a strong baseline for lifelong person re-identiﬁcation research, the performance still
needs to be improved. The latest KRKC [30] suggests that unfreezing
the parameters of the reference model leads to signiﬁcant performance
gains. However, this approach requires the maintenance of two models
for feature space consolidation during the training and testing phases,
which consumes more time and GPU resources. In contrast, our method
employs the same framework without freezing the parameters of the reference model, which achieves superior performance by adaptive knowledge transition. It eﬀectively facilitates knowledge transferring through
transition knowledge learning and multi-relation distillation. Compared
to the results of KRKC without feature space consolidation, our method
demonstrates an average performance improvement of 5.9 %/6.5 % in
terms of Rank-1/mAP. These outstanding results further highlight the
eﬀectiveness of the proposed method, providing compelling evidence
for its superiority in LReID tasks .
Performance on Training Order-2. To further validate the robustness of the proposed method, we conducted incremental training under
Order-2 in Table 3, alongside comparisons with state-of-the-art methods.
The experimental results show that our method maintains a leading position in both average mAP and Rank-1 metrics. Similar to Order-1, traditional incremental learning methods such as LwF [42] and SPD [45] limited performance under Order-2 due to their inability to generalize eﬀectively in addressing the unique challenges of LReID scenarios. To tackle
these limitations, AKA [5] and MEGE [25] have investigated the challenges of LReID and proposed robust baselines. While these approaches
show notable improvements over traditional methods, they still fail
to comprehensively address the inherent challenges of LReID. Furthermore, compared to selective knowledge distillation methods such as
PKD [27], CRFL [46] and C2R [31], our approach demonstrates clear advantages. The ﬁltering modules in these methods often introduce uncontrollable and noisy features during the distillation process. In contrast,
our method leverages raw features directly and adaptively regulates the
extent of knowledge transfer through the knowledge transition learning
module. This eﬀectively mitigates the impact of redundant information,
leading to a more substantial improvement in accuracy. Recently, LSTKC
[28] introduce a diﬀerent approach by employing rectiﬁcation module
to adjust the fusion ratio between the new model and old model, achieving remarkable performance. However, the proposed method utilizing
multi-relation distillation for multi-scale knowledge transfer still outperforms it by +2.1 %/2.0 % in mAP/Rank-1. These remarkable results
under training Order-2 further illustrate the superiority of the proposed
method. By consistently achieving outstanding performance across different training order, our method demonstrates its robustness and adaptability, eﬀectively addressing the challenges in LReID.

ViPeR is a classic person Re-ID dataset with small scale and simple
scene settings. It consists of images of 632 individuals captured from two
diﬀerent viewpoints, with each person having one image per viewpoint,
for a total of 1264 images.
Market-1501 is a widely used benchmark dataset in the ﬁeld of person Re-ID, consisting of 32,688 images of persons collected at Tsinghua
University. This dataset includes images of over 1500 individuals from
six cameras, amounting to over 30,000 images.
DukeMTMC is also a benchmark datasets includes images captured
by 8 cameras on the Duke University campus, providing diverse scenarios for person re-identiﬁcation research. It includes 36,411 images of
1404 identities, with 702 identities for training and 702 for testing.
Cuhk-sysu collects person images from ﬁlm scenes and street
footage, with over 4000 individuals from two scenarios and a total of
over 100,000 images. Each person has multiple images in diﬀerent scenarios, with annotations including individual identity and scene numbers.
MSMT17 is a large-scale multi-camera person Re-ID dataset provided by Peking University. It contains over 4000 individuals with more
than 250,000 images captured by 15 cameras covering diverse scenes
and environments.
Cuhk03 is a comprehensive person re-identiﬁcation dataset with
1467 identities and 13,164 images. It oﬀers both manually labeled and
automatically detected bounding boxes for realistic evaluation.
Evaluation Metric. To evaluate the performance of our method in
the LReID setting, this paper mainly employs two evaluation metrics: Cumulative Matching Characteristics [40] (CMC) and mean Average Precision [41] (mAP). CMC evaluates system performance by displaying
the probability of retrieving the correct match within the top 𝑁 matching results. Another evaluation metric is mAP, which considers both the
retrieval accuracy and recall rate of the system.
4.2. Implementation details
In this paper, the ResNet50 model pre-trained on ImageNet is used
as the backbone. The proposed TFE is embedded before the ﬁrst stage
of ResNet50 to extract transition factors. The data replayer stores two
images for each identity and extracts 64 images for each epoch. All images are resized to 256 × 128 and each batch contains 128 images from
the new task. The Adam optimizer is employed in this paper to update
the model, with a weight decay of 5e-4. The temperature parameter 𝜏
for knowledge distillation is set to 1. The total number of epochs is 60,
with an initial learning rate of 3.5e-4 that decays after 20 epochs. The
experiments are implemented by Pytorch with an Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz and two NVIDIA A4000 GPUs.
4.3. Comparison with state-of-the-arts

4.4. Ablation study
Performance on Training Order-1. To demonstrate the eﬀectiveness of the proposed method, this section conducts incremental training
on Order-1 and compares it with existing methods, as shown in Table 2.
The experimental results show that our method outperforms most existing approaches under equivalent conditions, achieving leading performance on all datasets. Notably, existing methods which focus on incre-

To validate the eﬀectiveness of the proposed method in this paper, a series of ablation experiments are conducted on the various
modules. Fig. 4 shows the CMC curves about diﬀerent methods on
the benchmark, and Table 4 illustrates the ablation results of mAP
and Rank-1 on the benchmark. The experiments demonstrate that by
6

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Table 2
Comparison of Rank-1(%) and mAP(%) performance with state-of-the-arts on seen domains for Training Order-1. *
represents feature space consolidation, which consumes more time and GPU resources.
VIPeR

Method

Reference

mAP
JointTrain
iCarL [43]
UCIR [13]
BiC [44]
WA [14]
LwF [42]
AKA [5]
PTKP [6]
KRKC [30]
KRKC* [30]

–
CVPR17
CVPR19
CVPR19
CVPR20
CVPR21
CVPR21
AAAI22
AAAI23
AAAI23

85.7
67.0
66.8
61.2
58.0
54.6
61.7
66.3
69.9
76.3

Our

this paper

72.2

Market-1501

Cuhk-Sysu

MSMT17

Average

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

78.8
56.6
56.3
50.4
48.1
45.3
50.6
56.0
58.5
67.7

85.4
56.9
45.5
47.4
50.9
53.9
28.3
58.3
47.9
64.4

93.7
78.7
65.9
68.9
70.3
74.6
50.7
77.4
72.3
82.5

88.1
80.0
68.8
71.6
70.4
81.1
76.9
77.1
84.4
88.9

89.8
82.9
70.5
72.5
71.9
83.7
79.6
78.8
86.4
90.7

49.4
10.2
12.8
22.4
18.6
14.6
13.4
25.2
46.1
43.3

74.7
24.1
29.3
42.1
38.8
32.3
28.0
48.0
69.9
67.1

84.2
53.5
48.5
50.7
49.5
51.1
45.1
56.7
62.1
68.2

77.2
60.6
55.5
58.5
57.3
59.0
52.2
65.1
71.8
77.0

62.3

60.2

80.4

84.9

86.9

57.0

79.0

68.6

77.2

Table 3
Comparison of Rank-1(%) and mAP(%) performance with state-of-the-arts on seen domains for Training Order-2.
DukeMTMC

MSMT17

Market-1501

Cuhk-Sysu

Chuk03

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

60.4

75.7

24.6

48.9

68.1

85.2

81.4

83.8

42.7

43.6

55.4

67.5

CVPR19
CVPR21
CVPR21
MM22
TPAMI23
PR23
MM24
AAAI24
CVPR24
CVPR24

28.5
42.7
42.2
58.3
30.1
34.4
49.4
49.9
59.7
53.4

48.5
61.7
60.1
74.1
46.1
51.3
67.0
67.6
75.0
70.5

3.7
5.1
5.4
6.4
5.7
7.6
14.5
14.6
7.3
14.5

11.5
14.0
15.1
17.4
16.4
20.1
33.8
34.0
19.2
33.3

32.3
34.4
37.2
43.2
33.1
61.6
56.0
55.0
42.4
60.6

57.4
58.6
59.8
67.4
56.5
80.4
77.6
76.7
66.5
81.0

62.1
69.9
71.2
74.5
77.6
82.8
83.2
82.3
76.0
83.0

65.0
73.0
73.9
76.9
80.5
85.1
84.9
83.8
77.8
84.9

43.0
34.1
36.9
33.7
44.1
49.0
45.3
46.3
37.8
45.0

45.2
34.1
37.9
34.8
45.3
50.1
47.1
48.1
39.3
46.1

33.9
37.2
38.6
43.2
38.1
47.1
49.7
49.6
44.7
51.3

45.5
48.4
49.4
54.1
49.1
57.4
62.1
62.1
55.6
63.2

this paper

59.8

73.3

13.9

32.5

49.8

73.9

81.9

84.5

53.1

56.5

51.7

64.1

Method

Reference

JointTrain

–

SPD [45]
LwF [42]
AKA [5]
PKD [27]
MEGE [25]
CRFL [46]
CKP [47]
LSTKC [28]
C2R [31]
DKP [32]
Our

Average

Table 4
Experimental results of ablation studies. AKT means adaptive knowledge transition, while CBD and IID represent
centroid bias distillation and instance interaction distillation, respectively.
VIPeR
KTL

✓
✓
✓
✓

CBD

IID

✓

✓
✓

✓
✓

✓

Market-1501

CUHK-SYSU

MSMT17

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

49.1
50.9
56.6
60.8
62.3
62.3

53.2
60.7
68.3
69.8
71.7
72.2

71.2
73.0
80.0
80.3
84.8
80.4

35.4
40.7
59.0
59.2
59.4
60.2

81.7
82.9
81.6
84.8
90.2
86.9

83.1
83.2
79.8
82.9
84.3
84.9

74.9
76.1
76.6
76.7
63.3
79.0

36.9
40.9
53.3
55.6
56.7
57.0

69.2
70.7
73.7
75.7
75.2
77.2

52.2
56.4
65.1
66.9
68.0
68.6

incorporating the proposed modules, the Rank-1 and mAP of the baseline are improved by 8.0 % and 16.4 %, respectively. Particularly, the
most signiﬁcant performance improvement is achieved when the KTL
is integrated, with a 9.5 % increase in mAP. Notably, on the Market1501, there is a remarkable increase of 23.6 % in mAP. These results indicate that AKT promotes progressive knowledge transfer, effectively combating catastrophic forgetting while mitigating the inhibitory eﬀects of knowledge distillation when generalizing to new
tasks.
Furthermore, Rank-1 improves signiﬁcantly on the ﬁrst three
datasets when CBD is integrated, while remaining a drop on the last
dataset. This suggests that the CBD eﬀectively reduces the bias on the
classiﬁer on the current task and gains signiﬁcant performance on previous tasks. In order to mitigate the degradation in the last dataset, the IID
is cooperated. It rebalances the weights of the classiﬁers by exploring the
relationship between the samples, leading to an advanced performance
both in CMC score and mAP.
To further validate the eﬀectiveness of the KTL module, this section
also conducted detailed ablation experiments. Fig. 5 (a) illustrates the
various performances on four datasets after training with four diﬀerent

Average

methods. The experiment results indicate that “Linear” gains a slight
improvement compared to the baseline method. However, there still remains a gap with the proposed method. This is attributed to the more
informative acquisition of transition factors by convolutional layers of
diﬀerent scales. In addition, the experimental results demonstrate that
the utilization of all hidden layers fails to improve the generalization capability of the model. This suggests that the shallow hidden layer in the
model contains a more domain-speciﬁc style, and the rational selection
of hidden layers is essential for the positive eﬀect of the TFE. Regarding
the selection of hidden layers, a detailed analysis is also presented in
Section 4.5.
In this section, the diﬀerent scales of convolution layers are
also explored to demonstrate the eﬀectiveness of KTL, which is
shown in Table 5 (b). The experiment results illustrate that the
proposed “1 × 1 + 3 × 3” convolution layer for TFE gains the best
performance. In particular, the accuracy improves obviously when
“1 × 1” convolution is employed. This suggests that the “1 × 1” convolution layer plays an important role in extracting the domain
style of the model. Furthermore, the performance is gained signiﬁcantly while two scales of convolution are constructed. This is
7

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Fig. 4. CMC curves of comparison experiments on four datasets about the diﬀerence methods. They illustrate the changes in CMC scores from rank-1 to rank-50.
The CMC scores of AKT both achieve the leading performance against the baseline method.

Fig. 5. Ablation study of the proposed TFE module. (a) and (b) depict the structures of TFE and Convolution, respectively.

attributed that multi-scale convolution provides diversity in the deep
feature extraction to ensure the quality of the generated transition
factors.
To further validate the eﬀectiveness of CBD and IID, this section constructs a detailed analysis about diﬀerent knowledge distillation strategies, which is shown in Table 5. Where “SD” represents only sample distillation and “DCD” denotes building the centroids in the batch rather
overall samples. The experiments results demonstrates that both SD and
DCD exhibit signiﬁcant performance gaps compared to our proposed
method. Therefore, the proposed CBD and IID achieve substantial accuracy improvement by acquiring a acceptable increase in computational
consumption.

dataset, as illustrated in Fig. 6. The backbone network for this method
is based on ResNet50, which consists of four stages of residual blocks.
We embed the TFE into each stage and compare their diﬀerent performance. Speciﬁcally, ‘layer0’, ‘layer1’, ‘layer2’, and ‘layer3’ represent the
embedding of TFE before the residual block of stage-1, stage-2, stage-3,
and stage-4 of the ResNet50 model, respectively. While ‘layer4’ indicates
the embedding of the TFE after all stages of residual block.
The results indicate that the accuracy gradually decreases as the
embedding position deepens within the backbone network. This phenomenon arises from the fact that deep-level feature extraction in the
backbone network captures more focus on domain-shared features,
while shallow networks are better at capturing domain-speciﬁc features.
Therefore, embedding in shallow hidden layers allows for capturing
more speciﬁc domain-style features, which facilitates the generation of
appropriate simulated features and promotes the knowledge transition.
Replayer batch size analysis. The inﬂuence about the size of the
replayer is explored in this section. The size of the replayer is critical to

4.5. Parameters analysis
Embedding position analysis. This section explores the impact of
embedding TFE in diﬀerent locations on the performance of the ﬁrst
8


codex
第一页到第八页已经抽出来了，但几篇的正文很长，工具输出把方法段中间截断了。下面我会只按方法章节标题重新抽取，避免只根据残缺片段做判断。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf' - | sed -n '/3\\. Our approach/,/4\\.3\\. Evaluation metrics/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | rg -n '"'^(III|A'"\\.|B\\.|C\\.|D\\.|E\\.|F\\.|IV|Algorithm|Fig\\. 3|Camera|Difficulty|Graph|Proxy|Unified)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Camera-Proxy Enhanced Identity-Recalibration
19:Camera Domain-driven Local correlation (CDL) Loss to enhance
147:Difficulty-aware Cluster-based Memory Updating (DCMU)
173:A. Contrastive Learning in Person Re-Identification
189:Fig. 3. The overall architecture of the proposed CEIL. Includes a difficulty-aware cluster-based memory update (DCMU) strategy, an proxy-enhanced
213:B. Visible-Infrared Person Re-Identification
283:C. Re-Ranking in Person Re-Identification
306:III. T HE P ROPOSED M ETHODS
318:A. Baseline
400:B. Camera Domain-Driven Local Association Loss
402:C. Difficulty-Aware Cluster-Based Memory Updating
649:D. Graph Network-Based Camera Constraint Adjustment
683:Algorithm 1 Re-Ranking Process of the Proposed GCR
704:E. Proxy-Enhanced Cross-Modal Mapping
797:F. Unified Loss
818:Algorithm 2 Training Process for CEIL: Stage 2
874:IV. E XPERIMENTS
875:A. Experimental Setting
923:B. Comparison With SOTA Methods

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf' - | sed -n '/3\\. Proposed method/,/4\\. Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Fig. 2. Illustration of the proposed framework. At the beginning of each epoch, we utilize a feature extractor network to obtain features for all samples in the dataset and then
assign pseudo-labels to them using DBSCAN. A memory dictionary is built to store a single feature vector for each cluster and is dynamically updated in repeated iterations.
Notably, we construct a camera-aware graph to represent the relationships between instances from multiple camera domains. Then, we design a graph convolutional network
(GCN) is designed to fuse the features into a unified space and learn more crucial information between multiple camera domains with semantic propagation. In this way, the
GCN can implement promising semantic transfer and generate ideal domain-invariant representations for assisting the training. Our model is trained with two contrastive losses
(𝑐 𝑙𝑠 and 𝑔𝑐 𝑛 ) and a camera adversarial loss 𝑎𝑑 𝑣 . The discriminator is trained to identify whether the features are from the feature extractor or the GCN. After training, only the
feature vectors output by the feature extractor are used for the inference.

respectively, and 𝑐𝑖 , 𝑐𝑗 be the corresponding camera IDs. We assign
values to the elements in 𝐴̂ based on identity and camera information:
⎧ 1, 𝑖𝑓 𝑦𝑖 = 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 ≠ 𝑐𝑗 ,
⎪
𝑎̂𝑖𝑗 = ⎨−1, 𝑖𝑓 𝑦𝑖 ≠ 𝑦𝑗 𝑎𝑛𝑑 𝑐𝑖 = 𝑐𝑗 ,
(7)
⎪
0,
𝑜𝑡ℎ𝑒𝑟𝑠.
⎩

3.4. Camera-aware graph multi-domain adaptive learning
Benefiting from the feature aggregation property, the GCNs can
achieve promising semantic transfer between different camera domains
and exploit the knowledge across multiple related camera domains
simultaneously. Consequently, in this work, we propose the adversarial
training scheme that employs feature aggregation with GCNs to provide ideal feature representations for optimizing the feature extractor
network.
In the 𝑡th iteration of an epoch, we sample 𝑁 𝑏 images from each
camera domain to form a mini-batch which can be formulated as
𝑏
𝑡 = {𝑡1 , 𝑡2 , … , 𝑡𝑉 } = {𝑏𝑡1,𝑖 , 𝑏𝑡2,𝑖 , … , 𝑏𝑡𝑉 ,𝑖 }𝑁
, where 𝑡𝑖 ⊂ 𝑖 . The
𝑖=1
mini-batch of images is first fed-forward into the feature extractor 𝜙 to
obtain the up-to-date features  𝑡 = 𝜙(𝑡 ). We argue that image features
with the same identity across various cameras and different identities
within the same camera contain critical semantic information. In order
to exploit this correlation pattern, we design a heterogeneous graph
to represent the identity and camera domain relationships between
all instances in a mini-batch. Then, a graph convolutional network
(GCN) is designed for fusing the features into a unified space and
learn more crucial information between multiple camera domains with
semantic propagation. In this way, the GCN can implement promising
semantic transfer and generate ideal domain-invariant representations
for assisting the training. Accordingly, our model can adapt to various
camera style differences with leveraging simultaneously the knowledge
across multiple related camera-level sub-domains. Finally, we design an
adversarial training scheme for transferring the knowledge learned by
GCN to the feature extractor.
Specifically, all instances in a mini-batch can construct a directed
graph (, ), where  indicates the set of edges,  denotes the set of
nodes, and each node 𝑣𝑖 ∈  is associated with an image feature. In
practice, the weights of  are represented by a matrix:
𝑎
𝑎
𝐴̂ = (𝑎̂𝑖𝑗 ) ∈ R𝑁 ×𝑁 ,

Then, we add self-connections and normalize the matrix as:
1

1

𝐴 = 𝐷− 2 (𝐴̂ + 𝐼)𝐷− 2 ,

(8)

where 𝐼 denotes the identity matrix, and 𝐷 is the degree matrix.
To generate ideal representations with feature aggregation, we employ the graph convolutional network 𝐺 to update the node features on
(, ). The matrix 𝐴 and the set of nodes  are input into the GCN,
and the output is a new set of node features 𝑡 . The propagation rule
for updating the node features at the 𝑙-layer is defined as,
 (𝑙) = 𝐺(𝑙) ([𝐴 (𝑙−1) ∥  (𝑙−1) ]),

(9)

𝑡 } and 𝑡 =  (𝐿) . 𝐿 is the depth of GCN.
with  (0) =  𝑡 = {𝑓1𝑡 , 𝑓2𝑡 , … , 𝑓𝑁
𝑎
∥ is the concatenation operation. 𝐺(𝑙) is the network with two convblocks of the form Conv-BatchNorm-LeakyReLU. Note that the features
in 𝑡 have the same dimension as the features in  𝑡 .
For training the GCN, we consider maximizing the similarity between instances in the same cluster with InfoNCE loss [22]:
exp(𝑧𝑞 ⋅ 𝑚𝑧𝑞 ∕𝜏2 )
𝑔 𝑐 𝑛 = E[− log ∑𝑁 𝑚
],
(10)
𝑖=1 exp(𝑧𝑞 ⋅ 𝑚𝑖 ∕𝜏2 )

where 𝑧𝑞 ∈ 𝑡 , 𝑚𝑧𝑞 denotes the positive cluster feature vector to query
instance 𝑧𝑞 .
In addition, we design a discriminator network 𝐷 that is optimized
in an alternating manner with 𝜙 to address the adversarial min–max
problem:
min max E𝑧∼𝑝 [log 𝐷(𝑧)] + E𝑥∼𝑝 𝑡𝑟 [log(1 − 𝐷(𝜙(𝑥)))],
𝜙

(6)



𝐷

𝐺(𝜙( 𝑡𝑟 )), and 𝑝

(11)

where  =
 , 𝑝 𝑡𝑟 represent the distribution of the
samples in ,  𝑡𝑟 . Note that we block the gradient propagation between
𝜙 and 𝐺, 𝐺 and 𝐷. We train 𝐷 to maximize the probability of assigning

where 𝑁 𝑎 = 𝑉 × 𝑁 𝑏 represents the size of the mini-batch. Let 𝑦𝑖 , 𝑦𝑗 be
the pseudo-labels of the 𝑖th and 𝑗th image samples in the mini-batch,
4

Pattern Recognition 161 (2025) 111217

Z. Ran et al.

Algorithm 1 The Camera-Aware Graph Multi-Domain Adaptive
Learning Framework
Require:
Unlabeled training data  𝑡𝑟 ;
Momentum coefficient 𝛼;
Weight balance parameter 𝜆;
Temperature parameters 𝜏1 , 𝜏2 ;
Number of epochs 𝑁𝑒𝑝𝑜𝑐 ℎ ;
Number of iterations 𝑁𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛 ;
Initialize the feature extractor 𝜙 with ResNet-50 pre-trained on
ImageNet;
1: for 𝑘1 in [1,𝑁𝑒𝑝𝑜𝑐 ℎ ] do
2:
Extract feature vectors  for  𝑡𝑟 with 𝜙;
3:
Perform DBSCAN on  for generating identity pseudo-labels;
4:
Construct and initialize a cluster-level memory dictionary 
with Eq. (3);
5:
for 𝑘2 in [1,𝑁𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛 ] do
6:
Sample 𝑉 × 𝑁 𝑏 images from  𝑡𝑟 to form 𝑘2 ;
7:
Construct 𝐴 with Eq. (8);
8:
Update 𝐺 with Eq. (10);
9:
Update 𝜙 with Eq. (2);
10:
Update  with Eq. (4);
11:
end for
12: end for
13: After training, only 𝜙 is used for the inference.

Table 1
Properties of the three challenging datasets.
Datasets

#images

#IDs

#cameras

Market-1501
MSMT17
PersonX

32,217
126,441
45,792

1501
4101
1266

6
15
6

bounding boxes of 1041 identities for training, 11,659 probe bounding
boxes, and 82,161 gallery bounding boxes of 3060 identities for testing.
PersonX [24] consists of 45,792 synthetic images of 1266 identities
under 6 camera views. The dataset is divided into a training set and a
test set: the former contains 9840 bounding boxes of 410 identities,
while the latter consists of a gallery set and a query set containing
30,816 and 5136 images from 856 identities, respectively.
Evaluation Protocol. For person Re-ID, mean average precision
(mAP) and the Cumulative Matching Characteristic (CMC) curves are
adopted as metrics to evaluate the performance of each algorithm. The
mAP is the mean value of average precision (AP) across all queries,
and the CMC curves represent the probability that a correct match
appears in candidate lists of different sizes. We report the Rank1,
Rank5, and Rank10 accuracy of CMC curves in this paper. For a fair
comparison, we do not utilize any post-processing, such as multi-query
fusion or re-ranking, and all experiments are performed in a single
query setting [1].
4.2. Implementation details

the correct set labels to the feature vectors while training 𝜙 to minimize log(1 − 𝐷(𝜙(𝑥))). In other words, we train the discriminator 𝐷
to discriminate whether the feature vectors come from the extractor
network 𝜙 or the graph convolutional network 𝐺, while 𝜙 is trained to
fool 𝐷. In this way, the distributions of multiple camera domains can
be consistent with the unified distribution learned from the GCN. For
𝜙, the adversarial loss 𝑎𝑑 𝑣 is defined based on the probabilities of the
discriminator 𝐷(𝜙(𝑏)) over one mini-batch as:

The feature extractor in our framework adopts ResNet-50 [2] as the
backbone network, which is initialized with the parameters pre-trained
on ImageNet. Specifically, all modules after layer4 are removed and are
replaced with an average pooling layer, a batch normalization layer,
and an L2 normalization layer, which can produce 1024-dimensional
feature vectors. For the graph convolutional network, we set the depth
of the GCN 𝐿 to 1 in all experiments. The discriminator is a binary
classifier, where the input dimension is 1024, the output dimension
is 2, the dimensionality of two hidden layers is 1024, and the batch
normalization is inserted in convolutional layers. The entire model is
optimized by the Adam optimizer with 𝛽 = (0.9, 0.99). The learning rate
is initialized as 3.5 × 10−4 , 1 × 10−4 , and 3 × 10−4 for the feature extractor
network, discriminator, and GCN, respectively, and then reduced to
one-tenth of the previous value every 20 epochs.
Unless otherwise specified, all the input person images are resized
to 256 × 128. During training, random erasing, random horizontal
flipping, and random cropping are adopted for data augmentation. The
memory dictionary is updated with a momentum coefficient 𝛼 = 0.1.
For DBSCAN, the distance threshold is set to 0.6, and the maximum
distance between two samples is set to 0.4. During the training phase,
we set 𝜏1 = 𝜏2 = 0.05 and 𝑁𝑒𝑝𝑜𝑐 ℎ = 50. Our experiments are conducted
on the PyTorch platform with 2 RTX 3090 GPUs and 1 Intel i9-10980XE
CPU.

𝑏

𝑎𝑑 𝑣 =

𝑁
𝑉 ∑
∑
1
− log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )),
𝑏
𝑉 × 𝑁 𝑐=1 𝑖=1

(12)

where 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )) denotes the probability that the feature vector generated by 𝜙 comes from 𝐺. Notably, we construct the loss based on
− log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 )) instead of [1 − log 𝐷(𝜙(𝑏𝑡𝑐 ,𝑖 ))].
4. Experiments

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf' - | sed -n '/3\\. Adaptive knowledge transition/,/4\\. Experiment/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Adaptive knowledge transition
3.1. Problem statement
In this section, this paper introduces the problem statement and the
setting under the baseline of LReID. Given 𝑇 dataset tasks  = { 𝑡 }𝑇𝑡=1 ,
3

Pattern Recognition 174 (2026) 112947

J. Peng et al.

LReID aims to sequentially train these data domains and maximize their
generalization ability across all tasks. The 𝑡-th stage dataset 𝑋 𝑡 consists
of the training set and testing set. Notably, the classes of the training set
and testing set are non-overlapping. Only the training set is available
during the 𝑡-th training. LReID evaluates all the testing sets for each data
domain after ﬁnishing all the training processes of the training set. Since
| 𝑡 |
the subsequent work is based on the training phase,  𝑡 = {(𝑥𝑡𝑖 , 𝑦𝑡𝑖 )}𝑖=1
mainly represents the training set of the current data domain in this
paper, where | 𝑡 | is the number of the training set.
Following the baseline method, the reference models and data replayer are also employed. To be speciﬁc, data replayer randomly selects
and stores a small number of samples based on person identity from the
training data, which is formulated as:
 𝑡+1 ←  𝑡+1 ∪ {(𝑥𝑡1 , 𝑦𝑡𝑗 ), (𝑥𝑡2 , 𝑦𝑡𝑗 ), .(𝑥𝑡𝑝 , 𝑦𝑡𝑗 )}𝐶
,
𝑗=1

transfer. The framework mainly consists of two modules: Transition Factors Estimation (TFE) and Knowledge Integration Module (KIM).
The lower left part of Fig. 2 illustrates the process of the TFE. It uses
hidden layer feature maps from both the current and reference models
as input, and transition factors are obtained through a series of convolution and pooling operations. Especially, given the input image 𝑥𝑡 , the
deep feature maps extracted by the residual block of models are fed into
transition factors estimation:
ℎ𝑘 = 𝜑𝑘 (𝑥𝑡 ),

where the deep feature maps ℎ𝑘 are extracted by residual block 𝜑𝑘 , 𝑘 ∈
{𝑛, 𝑜} represents the branch of the current model and reference model,
respectively. To fetch the domain style from the deep feature maps, the
TFE employs shared 1 × 1 convolution and 3 × 3 convolution block to
reﬁne common characteristics, which can be formulated as:

(1)

where 𝐶 represents the total number of identities in the current data,
and 𝑝 denotes the number of samples randomly selected per identity.
The data replayer is merged into the new task dataset  𝑡+1 and fed into
the model for joint training in the subsequent task.
In addition, the model retains diverse information about identities
after the training of the current datasets, which holds vital importance
for knowledge recall in subsequent training sessions. Therefore, the current model parameters are frozen and saved as a reference model for the
next task:
𝜃𝑛𝑡+1 ← 𝐹 𝑟𝑜𝑧𝑒𝑛(𝜃𝑜𝑡 ),

(3)

𝛾 𝑘 = 𝐶𝑎𝑡(𝐶𝑜𝑛𝑣1×1 (ℎ𝑘 ), 𝐶𝑜𝑛𝑣3×3 (ℎ𝑘 )),

(4)

where 𝐶𝑜𝑛𝑣1×1 and 𝐶𝑜𝑛𝑣3×3 represent the 1 × 1 convolution and 3 × 3
convolution operation, respectively, and 𝐶𝑎𝑡 denotes the matrix concatenation operation, 𝑘 ∈ {𝑛, 𝑜}. By reducing the dimensionality through
convolution at two scales, a more informative feature is obtained for
both the current model and the reference model. Furthermore, the fused
style feature 𝛾 𝑚 can be obtained adaptively by adding the style feature
of the current model and the reference model, which can be represented
as:

(2)

𝛾𝑚 = 𝛾𝑛 + 𝛾𝑜

where 𝐹 𝑟𝑜𝑧𝑒𝑛 represents the frozen the parameter, 𝜃𝑛𝑡+1 is the parameters
of reference model for the task 𝑡 + 1 and 𝜃𝑜𝑡 is the parameters of current

(5)

In order to obtain informative transition factors for diﬀerent models, the fused style feature 𝛾 𝑚 are fed into pooling layer and multi-layer
perception (MLP), which is formulated as:
[ 𝑛 ]
𝛼
= 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝑀𝐿𝑃 (𝑃 𝑜𝑜𝑙(𝛾 𝑚 ))),
(6)
𝛼𝑜

model for the task 𝑡.
3.2. Overview
To improve the generalization and anti-forgetting ability of the
LReID model, this paper proposes an Adaptive Knowledge Transition
(AKT) framework, which conducts the simulated distribution for facilitating the adaptive knowledge distillation. The framework of AKT can
be summarized as Fig. 2. Its training phase is mainly separated into two
situations: 𝑡 = 1 and 𝑡 > 1.
For the ﬁrst task 𝑡 = 1, person images are randomly sampled from
the current task data based on identity and fed into the model. After extracting features from the person images by the model, the Re-ID losses
are computed and backpropagated. Finally, the model parameters are
updated and trained with a speciﬁed number of epochs until convergence.
In other situations, for the remaining tasks 𝑡 > 1, samples are ﬁrst extracted based on identity from the current task data and data replayer.
Then they are fed into both the current model and the reference model
to extract deep feature maps. Transition factors are obtained by fetching
the style for diﬀerent domains with deep feature maps in the Transition
Factors Estimation (TFE). Combined with transition factors, the Knowledge Integrate Module (KIM) merges the features and constructs the
intermediate transition knowledge, which is performed as the bridge to
facilitate the knowledge transition. In Multi-Relation Distillation (MRD),
Centroid Bias Distillation (CBD) and Instance Interaction Distillation
(IID) are constructed to explore latent relationships from diﬀerent perspectives, which further enhance knowledge consolidation. The above
steps are continuously iterated until ﬁnishing training all the tasks to
remain a discriminative and anti-forgetting LReID model.

where 𝑀𝐿𝑃 represents the multi-layer perception and 𝑃 𝑜𝑜𝑙 denotes the
average pooling layer. 𝛼 𝑛 , 𝛼 𝑜 ∈ ℝ represent the transition factors from
the current model and the reference model, respectively.
Combined with the transition factors, the knowledge integration
module generates intermediate futures to further promote the knowledge transfer. To be speciﬁc, it takes the one-dimension feature of the
model as input, which is represented as:
( )
𝑓 𝑘 = 𝐺𝑘 𝑥𝑡 ,
(7)
where 𝑘 ∈ {𝑛, 𝑜} represents the branch of the current model and the reference model and 𝑓 𝑘 ∈ ℝ𝑑 is the one-dimension features from the current model 𝐺𝑛 and the reference model 𝐺𝑜 . 𝑑 is the dimension of the feature vector extracted by deep neural model. Based on transition factors,
the knowledge integration module generates the intermediate feature
𝑓 𝑚 ∈ ℝ𝑑 by multiple and added operations, which can be formulated
as:
𝑓 𝑚 = 𝛼𝑛 ⊙ 𝑓 𝑛 + 𝛼𝑜 ⊙ 𝑓 𝑜 ,

(8)

where ⊙ represents the vector-matrix multiplication of the vectors. The
intermediate feature 𝑓 𝑚 encompasses the domain styles of both the current model and the reference model, providing crucial cues for knowledge accumulation and refresh. By constructing simulated distribution,
the knowledge can adaptively transfer between models, which eﬀectively weakens the inhibition of generalization capacities to new tasks.
3.4. Multi-relation distillation

3.3. Knowledge transition learning

To further enhance the knowledge consolidation of previous
identities, the Multi-Relation Distillation (MRD) is designed in this section. It aims to explore the latent relationships across domains from different perspectives, which exhibit greater relevance and are less prone
to be forgotten. The two main relationships are included: (1) Instance
Interaction Distillation (IID) and (2) Centroid Bias Distillation (CBD),
which are shown in Fig. 3

To conduct the adaptive transfer knowledge, Knowledge Transition Learning (KTL) is proposed in this section. In the context of
two vastly distinct knowledge domains, directly transferring knowledge
prone to be ineﬃcient and uncontrollable. Therefore, KTL adaptively
constructs a simulated distribution to achieve a progressive knowledge
4

Pattern Recognition 174 (2026) 112947

J. Peng et al.

Fig. 3. Illustration of Centroid Bias Distillation (CBD) and Instance Interaction Distillation (IID). (b) shows the previous samples and centroids have a shift when
expanding to new tasks. While (a) illustrates the centroids of the previous task are employed to narrow the intra-class distance in CBD and (c) presents the relation
of previous samples deployed to enhance the knowledge consolidation in IID.

Feature-based knowledge distillation only emphasizes the consistency of features on the same sample across models. However, the latent
relationships between samples are more informative for knowledge distillation. Analogous to the human brain, we usually construct basic skills
with the relationships between objects to generalize new tasks. Therefore, the IID is proposed to recall previous identities by exploring the
relationship between samples. It ﬁrstly construct the feature matrix as
input:
𝐹 (𝑘) = [𝑓1𝑘 , 𝑓2𝑘 , … , 𝑓𝐵𝑘 ] ∈ ℝ𝐵×𝑑 ,

the centroid and samples to establish the constrain of knowledge refresh
for the previous identities:
𝐿𝑐𝑏𝑑 = 𝐷𝐾𝐿 (𝑆 𝑛 ∕𝜏||𝑆 𝑚 ∕𝜏) + 𝐷𝐾𝐿 (𝑆 𝑚 ∕𝜏||𝑆 𝑜 ∕𝜏).

The centroid bias distillation loss eﬀectively mitigates the disruption to the feature representation of previous samples, increasing the
ability of the model to generalization to remaining tasks. By building
the above distillation constraints through simulated features, the model
signiﬁcantly promotes knowledge preservation while mitigating the inhibition of the generalization to new tasks.

(9)

where 𝐵 is the number of person images in a batch, while 𝑑 denotes the
dimension of the feature vector. The feature matrix 𝐹 (𝑘) ∈ {𝑛, 𝑜, 𝑚} contains a batch of features from the branch of the current model, reference
model, and KTL, respectively. To capture the relationships between features, cosine distance is utilized in this paper to calculate the similarity
matrix:
𝑇

𝑉 𝑘 = 𝐹 (𝑘) 𝐹 (𝑘) ,

3.5. Optimization
During the training of the ﬁrst task, the optimization objective of the
model is to maximize its discriminative ability for person Re-ID. At this
stage, this paper mainly employs cross-entropy loss 𝐿𝑐𝑒 , triplet loss 𝐿𝑡𝑟 ,
and contrastive loss 𝐿𝑐 to constrain the training of the model.

(10)

𝐿𝑟𝑒𝑖𝑑 = 𝐿𝑐𝑒 + 𝐿𝑡𝑟 + 𝐿𝑐 .

where the 𝑉 𝑘 ∈ ℝ𝐵×𝐵 , which contains the similarity between samples
within a batch. IID then utilizes the similarity matrix to build the constrain for knowledge review of previous tasks, which is formulated as:
𝐿𝑖𝑖𝑑 = 𝐷𝐾𝐿 (𝑉 𝑛 ∕𝜏||𝑉 𝑚 ∕𝜏) + 𝐷𝐾𝐿 (𝑉 𝑚 ∕𝜏||𝑉 𝑜 ∕𝜏),

(14)

When training the remaining tasks, to prevent catastrophic forgetting
caused by excessive learning, the optimization objective of model must
incorporate mechanisms for reviewing previously learned knowledge.
Therefore the knowledge refresh loss is employed, which is formulated
as:

(11)

where 𝐷𝐾𝐿 represents the Kullback-Leibler Divergence [33], an asymmetric measure of diﬀerence between two probability distributions. 𝜏 is
the temperature parameter used to control the smoothness of the distribution. Diﬀerent from identity relationships, the relationships between
samples exhibit more information and eﬀectively mitigate catastrophic
forgetting in LReID tasks.
In addition to utilizing relation distillation for intra or interrelationships, this paper also proposes CBD, which considers the relationship from centroids. When the model is extended to new tasks, the
weights of the classiﬁer inevitably are biased towards the new task due
to the unbalance of samples between current and previous identities.
It may disrupt the feature representation and intra-class compactness of
the previous classes. Therefore, the CBD aims to reduce the centroid bias
of the previous class while expanding to new tasks. Given the feature 𝑓 𝑘 ,
the distance from the centroid is calculated by cosine similarity:
𝑆 𝑘 = 𝑐𝑗 𝑇 𝑓 𝑘 ,

(13)

𝐿𝑟𝑒𝑓 𝑟𝑒𝑠ℎ = 𝐿𝑐𝑏𝑑 + 𝐿𝑖𝑖𝑑 ,

(15)

where 𝐿𝑐𝑡 is centroid bias distillation loss calculated by Eq. 13 while 𝐿𝑟𝑒
is instance interaction distillation loss derived from Eq. 11. Finally, the
overall loss function can be described as:
𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑟𝑒𝑖𝑑 + 𝐿𝑟𝑒𝑓 𝑟𝑒𝑠ℎ .

(16)

4. Experiment

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf' - | sed -n '/3 Methodology/,/4 Experiments/p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
 succeeded in 109ms:
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

Pattern Recognition 161 (2025) 111301

X. Yang et al.

different classifiers with independent parameters. The classifier then
predicts the identity of each feature vector separately. For each branch,
we calculate the classification loss based on the classifier’s predicted
value and the identity of the input image. The loss of each branch is
used to update the parameters of the corresponding branch during the
training phase.
Fig. 3. Change of channel width in dynamic interaction. 𝑑𝑖 and 𝑑ℎ represent the
input/output channel and hidden channel (M = 3).

3.2. Dynamic aggregation
After extracting features through a dual-stream network, we propose
a dynamic aggregation module to help the current modality absorb
information from the other modality as a way to reduce the difficulty
of inter-modality recognition. Specifically, we employ multi-head attention to enhance intra-modality information, and then design the
dynamic aggregation module to enhance modality representations. Our
structure is designed as a recursive architecture since more dynamic
projections can lead to better interactions. The whole framework not
only learns to extract high-quality visual contextual cues but also
attempts to focus on optimal attention through image representations
based on cross-modality feature guidance. Essentially, our dynamic
interactions define two modalities as primary/auxiliary modalities,
transforming the auxiliary modality from a single dimension to a higher
and broader dimension on which the projection weights required for
the primary modality are based. Since the projection weights for the
information interaction are based on the auxiliary modality features
and not fixed values, this process is referred to as dynamic interaction.
The recursive architecture is designed to accommodate more weight
projections, leading to better information interaction. Dynamic interaction enables direct communication between modalities and adaptively
improves modality representation, thereby reducing the difficulty of
inter-modality recognition.
From the modality perspective, we conduct two interactions: one
where the visible light image serves as the primary modality 𝑋𝑣𝑝 and
the infrared image as the auxiliary modality 𝑋𝑖𝑎 ,; and another where
the infrared image is the primary modality 𝑋𝑖𝑝 and the visible light
image is the auxiliary modality 𝑋𝑣𝑎 Firstly, we apply a multi-head
attention mechanism to enhance the important regions within each
modality, optimizing the attention of features in critical areas for internal information enhancement. The interaction process can be defined as
(𝑋𝑣𝑝 , 𝑋𝑖𝑎 ) and (𝑋𝑖𝑝 , 𝑋𝑣𝑎 ). Secondly, the dynamic aggregation
module projects the primary modality features using the auxiliary
modality information, dynamically generating projection weights based
on the auxiliary modality features. It takes the primary and auxiliary
modality features as input and outputs the primary modality projection
features for one iteration. First, the weights of the dynamic projection
values are generated based on the auxiliary modality features:
 = 𝐿1 (𝑋𝑎 ),

Fig. 4. The new visible features and infrared features generated through dynamic
aggregation.

where 𝑋𝑣𝑛𝑒𝑤 and 𝑋𝑖𝑛𝑒𝑤 stand for the new visible and infrared features
obtained after dynamic interaction. (⋅) represents dynamic interaction
and (⋅) refers to the multi-head attention mechanism.
3.3. Feature intersection
To represent the whole human image, previous VI-ReID methods
adopt GAP or GMP. GAP covers the whole body part of the human
image, but it is easily distracted by background clutter and occlusion.
GMP overcomes this problem by aggregating features from the most
differentiated parts helpful for VI-ReID, while discarding background
clutter. However, GMP does not include information from the whole
body part. For this reason, we propose a simple structure but take full
advantage of GAP and GMP, while also introducing local features. The
difference with GAP, GMP, and (GAP+GMP) is shown in Fig. 5. Here,
we divide into two branches, pooling (GMP) the 18 × 9 global features extracted from the backbone network to obtain an N × 1 feature
(without considering the batch size and channels) and then dividing
it into N times 1 × 1 local features. The other branch is first divided
to obtain N times 18∕𝑁 × 9 local features, and then pooled (GAP) to
get N times 1 × 1 local features. Finally, the local features from the
two branches are concatenated in the channel dimension. It is worth
noting that each of the two branches focuses on different regions. The
first branch first adopts GAP and then divides, so that the obtained
local features synthesize the information of the whole human body and
effectively consider the contextual cue relationships. On the other hand,
the second branch performs GMP after division, which can better focus
on the more discriminative local features in each region.
The whole process can be described as:

(1)

where  represents the weight of the projected value and 𝐿1 (⋅) refers to
the linear layers, which implement the conversion of 1-d features into
2-d matrices. The next dynamic interaction can be defined as follows:
𝑋𝑛𝑒𝑤 = 𝐿2 (𝑋𝑝 ∗ ),

(2)

where 𝐿2 (⋅) represents the multiplication of the two, reverting from 2-d
to 1-d, corresponding to 𝐿1 (⋅).
Recursive structure is adopted to optimize modal feature representation by successive iterations. Notably, 𝑀 recursive blocks of dynamic
interactions are identical except for their channel dimensions. We
create a bottleneck architecture by assigning a hidden channel 𝑑ℎ to
the intermediate dynamic interaction block than the original input
dimension 𝑑𝑖 , as shown in Fig. 3. To finally obtain the new bi-modality
features after the interaction, 𝑀 is set to 2, as shown in Fig. 4. The
interaction process can be defined as:
𝑋𝑣𝑛𝑒𝑤 = ((𝑋𝑣𝑝 , 𝑋𝑖𝑎 )),

(3)

𝑋𝑖𝑛𝑒𝑤 = ((𝑋𝑖𝑝 , 𝑋𝑣𝑎 )),

(4)

𝑋𝑛𝑔𝑎𝑝 = (𝐺𝐴𝑃 (𝑋𝑖 , 𝑋𝑣 )),

(5)

𝑋𝑛𝑔𝑚𝑝 = 𝐺𝑀 𝑃 ((𝑋𝑖 , 𝑋𝑣 )),

(6)

𝑋𝑛 = 𝐶 𝑜𝑛𝑐 𝑎𝑡(𝑋𝑛𝑔𝑎𝑝 , 𝑋𝑛𝑔𝑚𝑝 ),

(7)

where  represents partial division, 𝑋𝑛𝑔𝑎𝑝 is the feature map obtained
by branch 1, 𝑋𝑛𝑔𝑚𝑝 is the feature map obtained by branch 2, and 𝑋𝑛 is
the feature map obtained by splicing two branch channels.
3.4. Multi-weight loss
In the VI-ReID task, each person consists of visible images and
infrared images. To learn with identity attribute information, we propose a new multi-weight loss to reduce the center distance of the
4

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Fig. 5. Comparisons of common feature extraction methods with our FI. The blue and pink rectangles denote the visible and infrared features respectively. (a) GAP. (b) GMP. (c)
GAP+GMP. (D) FI(ours). (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

4. Experiments
We conduct extensive experiments on two publicly available
datasets SYSU-MM01 and RegDB. First, in Section 4.1, two evaluated
datasets are introduced. Then, details on training of the experiments are
described in Section 4.2, followed by an introduction to the evaluation
metrics in Section 4.3. Next, the experimental results are analyzed and
compared to state-of-the-art methods in Section 4.4. Ablation studies
are conducted in Section 4.5, and finally, a comprehensive discussion
of the overall experiments is presented in Section 4.6.
Fig. 6. A simple description about multi-weight loss. The distance between negative
and positive samples represents the degree of similarity between different pedestrians.
The more distant pedestrians are less similar. Hierarchical supervision is achieved by
setting different weights for pedestrians with different similarities. (For interpretation
of the references to color in this figure legend, the reader is referred to the web version
of this article.)

4.1. Datasets
SYSU-MM01 is the first large-scale dataset specifically for VI-ReID
task. It includes six cameras (including four RGB cameras and two IR
cameras) for a total of 491 pedestrians. Cameras 3 and 6 shoots infrared
light in an unlit environment. Cameras 4 and 5 are placed in the two
outdoor scenes of the gate and garden, respectively.
RegDB [44] contains 412 pedestrians, and each person has 20
images, including 10 RGB images and 10 IR images, among which 254
are female and 158 are male. And 156 of the 412 pedestrians are taken
from the front and 256 pedestrians are taken from the back. Following
the previous work [20], we randomly select 206 IDs for training and
the remaining 206 IDs for testing.

same identity feature in different modalities and increase the center
distance of different identity features within the same modality, as
shown in Fig. 6. For increasing the discrepancy of different pedestrians
within a modality, the similarity between different pedestrians is not
the same, which easily causes the pedestrians with higher similarity
to be insufficiently trained. Here, we divide the negative samples 2
levels according to the similarity distance, and different levels are given
different loss weights to flexibly impose different penalties for different
identity center distances.
The whole multi-weight loss can be described as:
‖2
𝑀 𝑊 = ‖
‖(𝐶𝑣 , 𝐶𝑖 )‖2 − 𝜆1 𝑎𝑛𝑘1 − 𝜆2 𝑎𝑛𝑘2 ,

4.2. Implementation details

(8)
Our method is implemented on the Pytorch, using NVIDIA TiTAN
Xp GPU. Following the previous method [10], the images are resized to
384 × 192. We adopt the SGD optimization strategy. The weight decay
is set to 0.0005 and the momentum is set to 0.9. We utilize random
cropping and horizontal flips for data enhancement. Our learning rate
starts at 0.1 using a warm-up strategy. The batch size is set to 64.
We randomly select 4 identities. And each identity contains 4 RGB
images and 4 IR images. Our training epoch is 60. As always, we utilize
ResNet-50 [43] as our backbone pre-trained on ImageNet.
The feature map of the input feature intersection module is partitioned horizontally into 𝑁 = 6 components. Through a linear layer and
batch normalization, the dimension of the feature is reduced to 512,
which is used in 𝑀 𝑊 . After passing the dual-classifier, the dimension
of the feature is reduced to the number of categories, which is utilized
to predict the labels of the images.

where 𝐶𝑣 and 𝐶𝑖 denote visible and infrared modality center. 𝜆1 and 𝜆2
are the weights corresponding to different similarity levels. 𝑎𝑛𝑘1 and
𝑎𝑛𝑘2 represent two levels of ranking based on the similarity between
different pedestrians. 𝑎𝑛𝑘1 typically corresponds to pedestrian pairs
with higher similarity, while 𝑎𝑛𝑘2 corresponds to pedestrian pairs
with lower similarity.
At the same time, the cross-entropy loss is utilized for identity
prediction in the paper. The classification loss 𝑐 𝑙𝑠 is calculated as:
𝑁𝑟𝑔 𝑏+𝑖𝑟
(
)
∑
1
𝑐 𝑙 𝑠 =
𝑐 𝑒 𝑋𝑖𝑟𝑔 𝑏+𝑖𝑟 , 𝑌𝑖𝑟𝑔 𝑏+𝑖𝑟 ,
(9)
𝑁𝑟𝑔 𝑏+𝑖𝑟 𝑖=1
where 𝑁𝑟𝑔 𝑏+𝑖𝑟 denotes all images from visible and infrared modalities,
𝑐 𝑒 is the cross-entropy loss function. 𝑌𝑖 is the ground truth label.
𝑋𝑖 represents the feature vector of the 𝑖th image used for identity
prediction.
5

Pattern Recognition 161 (2025) 111301

X. Yang et al.

Table 1
Comparison of CMC (%) and mAP (%) with the state-of-the-art methods on SYSU-MM01 and RegDB datasets. Our results show the best results in terms of Rank-1 accuracy and
mAP.
Methods

SYSU-MM01

RegDB

All-search

Zero-Padding [32]
D2 RL [33]
Hi-CMD [45]
JSIA-ReID [6]
AlignGAN [5]
XIV-ReID [9]
DDAG [46]
NFS [47]
DML [48]
CM-NAS [49]
FMCNet [14]
SMCL [15]
PMT [50]
DART [35]
MRCN [16]
CAJL [11]
MMN [10]
DCLNet [42]
Ours

Indoor-search

Visible to thermal

Thermal to visible

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

Rank-1

mAP

Rank-1

mAP

14.80
28.90
34.94
38.10
42.40
49.92
54.75
56.91
58.40
61.99
66.34
67.39
67.53
68.70
68.90
69.88
70.60
70.80
71.30

54.12
70.60
77.58
80.70
85.00
89.79
90.39
91.34
91.20
92.87
–
92.87
95.36
96.40
95.20
95.71
96.20
–
95.54

71.33
82.40
–
89.90
93.70
95.96
95.81
96.52
96.90
97.25
–
96.76
98.64
99.00
98.40
98.46
99.00
–
98.78

15.95
29.20
35.94
36.90
40.70
50.73
53.02
55.45
56.10
60.02
62.51
61.78
64.98
66.30
65.50
66.89
66.90
65.30
67.73

20.58
–
–
43.80
45.90
–
61.02
62.79
62.40
–
68.15
68.84
71.66
72.50
76.00
76.26
76.20
73.50
77.16

68.38
–
–
86.20
87.60
–
94.06
96.53
95.20
–
–
96.55
96.73
97.80
98.30
97.88
97.20
–
98.06

85.79
–
–
94.20
94.40
–
98.41
99.07
98.70
–
–
98.77
99.52
99.50
99.70
99.49
99.30
–
99.18

26.92
–
–
52.90
54.30
–
67.98
69.79
69.50
–
74.09
75.56
76.52
78.20
79.80
80.37
79.60
76.80
80.55

16.70
43.40
70.93
48.10
57.90
62.21
69.34
80.54
77.60
84.54
89.10
83.90
84.83
83.60
91.40
85.03
91.60
81.20
94.32

17.90
44.10
66.04
48.90
53.60
60.18
63.46
72.10
84.30
80.32
84.40
79.80
76.55
75.70
84.60
79.14
84.10
74.30
89.02

17.80
–
–
48.50
56.30
–
68.06
77.95
77.00
82.57
88.40
83.10
84.16
82.00
88.30
84.75
87.50
78.00
93.27

18.90
–
–
49.30
53.40
–
61.80
69.79
83.60
78.31
83.90
78.60
75.13
73.80
81.90
77.82
80.50
70.60
88.77

4.3. Evaluation metrics

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf' - | rg -n '"'^(III|IV|A'"\\.|B\\.|C\\.|D\\.|V|Fig\\. 3|Problem|Method|Confounder|Intervention|Disentangle|Optimization)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
311:III. C AUSAL A NALYSIS
319:B. Analysis of Existing Likelihood-Based Methods
332:C. Causal Intervention Helps Invariant Feature Learning
357:A. Structural Causal Model For CC-ReID
384:IV. M ETHOD
385:Problem Formulation. For a CC-ReID dataset G =
392:Method Overview. The main pipeline of our Causal
411:Disentangle
428:Confounder
441:Intervention Module
553:Fig. 3. (a) The framework of the proposed Causal Clothes-Invariant Learning (CCIL). Three causal modules are complementary to each other and train
571:A. Confounder Dictionary
650:B. Intervention Module
747:C. Disentangle Regularization
844:D. Optimization
858:V. E XPERIMENTS
876:VC-Clothes
909:A. Datasets and Evaluation Protocol
946:B. Implementation Details
963:ViT-Base model [50] pre-trained on ImageNet with patch size
1007:C. Comparison with Causality-based ReID Methods
1009:D. Comparison with State-of-the-art Methods
1063:Method
1108:Method
1110:Venue
1114:VC-Clothes
1353:Method
1356:ViT B16 [50]
1407:Method
1410:Method

codex
目前已经拿到前三篇的完整方法段。后面三篇的标题格式混有期刊排版空格，我先用行号定位章节，再按行号切出方法，不用依赖标题匹配。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf' - | rg -n '"'^(3|4|Methodology|Occlusion|Subject|Detail|Multi|Cross|Dynamic|Loss|Inference|Experiments)'"'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
60:Multi-modal Feature
262:Cross-Attention
264:Cross-Attention
294:Cross
323:3
339:Methodology
351:3.1
353:Detail-focused Token Generation
377:Occlusion-aware Fine-grained Prompt
387:Subject-focused Token Generation
395:Dynamic Weighting Fusion Module
461:3.2
473:Multi-modal Feature Enhancement
492:3.3
494:Cross-Attention Mechanism
503:Loss Function and Inference
506:Loss LTriplet [Hermans et al., 2017], ID Classification Loss
509:Cross-modal Contrastive Loss
620:47.0 37.7
663:4
665:Experiments
672:3 learnable queries and 6 cross-attention blocks. The model
685:4.1
689:4.2
697:Inference
868:3
872:4 ”A photo of a [S ] person with [A ] partially occluded by [O ]” 75.2 65.5
911:4.3
953:3
954:4
1005:Dynamic Fusion 75.2
1065:4.4
1127:44(9):4894–4912, 2021.
1189:Occlusion-aware transformer with second-order attention

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | sed -n '306,873p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
III. T HE P ROPOSED M ETHODS
In this section, we build upon the baseline framework to
introduce our proposed two-stage training network, CEIL. The
training process is shown in Fig.3, which comprises three
components: the DCMU strategy, the CDL loss, and the PCM
module. In the first stage, we exclusively employ camera proxy
contrastive learning. In the second stage, PCM is integrated to
enable cross-modal feature learning at both the camera proxy
and class levels. Throughout the entire process, we use DCMU
as the memory update strategy and apply CDL loss as the loss
function for the camera proxy. After training, the GCR method
is introduced to further optimize the results.
A. Baseline
As a pioneering pseudo-label-based method in the UVIReID field, ADCA [24] serves as our baseline. To ensure
fair comparison with our CEIL, we exclude the cross-modal
memory aggregation while incorporating Multikernel Maximum Mean Discrepancy (MMD) [60] loss into the objective
function. Unlike DG-VAE [61], which employs distinct feature extractors and encoders for different modalities before
VAE decoding, ADCA demonstrates the redundancy of such
architecture in pseudo-label-based contrastive learning. It only
utilizes modality-specific first layers in the pre-trained network
to mitigate cross-modal discrepancies.
At the beginning of each epoch, features are extracted from
both infrared and visible modal samples using the backbone.
Subsequently, the DBSCAN [62] is applied for clustering these
features. Following the clustering process, the memory for
each cluster is initialized by computing the average feature
values within each respective cluster, as described below:
1 X s
f ,
(1)
m qe =
Cqe s∈Cqe
where e = {v, r } denote the visible and infrared modalities,
respectively. Cqe represents the q-th cluster set under modality
e, | · | denotes the number of instances within each cluster, and
f s represents the features extracted from the image s.
During training, the modality-specific memories are updated
through a momentum update strategy, as follows:
m qe(δ) ← βm qe(δ−1) + (1 − β) f q ,

(2)

where f q represents the features obtained from the training
batch corresponding to m q , β denotes the momentum updating
factor, and δ signifies the iteration number. The loss for each
modality is calculated using the InfoNCE loss [63], as follows:

exp q e · m e+ /τ
e

,
LInfoNCE = − log P e
(3)
N
e
e
p=0 exp q · m q /τ
where m e+ is the memory corresponding to the pseudo label
of the query q e , τ is a temperature parameter, and N e is the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

XIA et al.: CAMERA-PROXY ENHANCED IDENTITY-RECALIBRATION LEARNING FOR UVI-ReID

number of clusters for modality e. The MMD loss, using a
Gaussian kernel, aligns the feature distributions:
 2



(4)
LMMD = Ev φ FvN − Er φ FrN H ,
k

where φ(·) is a feature mapping function, and Hk represents
the Reproducing Kernel Hilbert Space. The image features are
denoted by FvN and FrN . Finally, the total loss combines the
proxy losses of each modality and MMD losses:
Lbase = LvInfoNCE + LrInfoNCE + LMMD .

(5)

8123

can vary significantly due to factors like shooting angles
and lighting conditions. By calculating losses independently
for each camera domain, the model can better handle these
variations, improving its ability to generalize across different
domains. Secondly, selecting negative samples with higher
similarity to the current sample allows the model to learn more
subtle differences. When negative samples are very similar to
positive samples, the model is forced to learn more detailed
features, enhancing its discriminative ability.

B. Camera Domain-Driven Local Association Loss

C. Difficulty-Aware Cluster-Based Memory Updating

The contrastive loss, using one-hot pseudo-labels [35], fails
to accurately adjust relationships between camera proxies
within the same class and those in other classes. PPCL [11]
incorporates camera information to enhance the learning process. For each training sample, it mines credible proxies from
camera agents that do not belong to the same class and are
captured by non-self cameras. However, in cross-modal scenarios, the significant differences within and between modalities
can lead to unstable clustering. When the clustering within
the same camera is unreliable, attempting to perform credible mining across cameras becomes even more impractical.
To address this limitation, we propose the CDL Loss (Lcdl ).
Unlike PPCL, it conducts local contrastive learning within
each camera domain, which avoids interference across camera
domains while reducing the impact of unreliable clustering.
Such a design is particularly well-suited for effectively handling feature learning in multi-camera scenarios.
For a given query feature, similarity is assessed with all
camera proxies in the same class. By calculating the distance
to both positive and negative proxies within each camera
domain separately, reduces disturbances from domain differences and brings camera domains of the same class closer
together. Given a query q, the loss L(q, c) under camera c as
follows:
exp(Sim+ (q, c))
, (6)
L(q, c) =
Pn
+
exp(Sim (q, c)) + i=1
exp(Simi− (q, c))

Traditional momentum update methods facilitate
cross-iteration information exchange by sequentially updating
the corresponding centroids in the memory bank with fixed
weights for each sample, However, which relies too much
on randomness, often neglecting hard samples. A common
issue is that, after multiple iterations, easy samples cluster
well, but hard samples remain problematic. Shi et al. [51]
selected a fixed number of hard samples from all clusters for
updates, which performed poorly in the face of imbalanced
cluster sizes. To address this limitation, we propose the
DCMU strategy. It dynamically selects hard samples within
aggregation clusters, generates weights based on difficulty and
a smoothing factor, and then replaces the cluster centroids
by cluster. As illustrated in Figure 4, this mechanism
forces easy samples to converge towards hard samples,
thereby significantly enhancing the model’s ability to handle
challenging samples.
During forward propagation, DCMU updates in units of
clusters. For the i-th input feature vector xic belonging to
cluster c in the current input batch and the corresponding
cluster center feature fc , the cosine distance is defined as:

where Sim+ (q, c) is the similarity between a query q and its
positive proxy in camera domain c, Simi− (q, c) is the similarity
between q and the i-th nearest negative proxy in this domain,
and n is the number of nearest negative proxies considered
per domain. The positive proxy is the proxy corresponding
to the class of query q within the current camera domain.
The nearest negative proxies are chosen based on the cosine
similarity between query q and other proxies in the camera
domain, with “nearest” meaning sorted in descending order of
similarity. The overall loss Lcdl (q) is obtained by averaging
over all effective camera domains as follows:
1 X
− log (L(q, c)) ,
(7)
Lcdl (q) =
|Cq |
c∈Cq

where Cq denotes the set of valid camera domains for the
query q, and |Cq | represents their number.
Calculating losses based on local negative samples across
different camera domains is crucial. Firstly, it helps maintain
domain consistency. Images from different camera domains

dic = 1 − cos(θ ) = 1 −

xic · fc
,
∥xic ∥∥fc ∥

(8)

where dic is the cosine distance between xic and fc , with ∥ · ∥
denotes the L2 norm. The hard samples for cluster c, denoted
by S c , are selected using the binary indicator F c as follows:
(
1 if di ≥ µc
c
Fi =
,
0 otherwise
Nc

1 X c
µ = c
di ,
N
c

i=1

S c = {s | Fsc = 1},

(9)

where µc represents the mean distance of all N c samples in
the current cluster. Then weights for hard samples wc are
calculated using a softmax and smoothed with a factor α:
w cj = Softmax(d cj ) · α + δargmax(d c ) ( j) · (1 − α),
exp(d cj )
Softmax(d cj ) = P
,
(10)
c
s∈S c exp(ds )
where j denotes the j-th index in S c , and δargmax(d c ) is a binary
list corresponding to d c , with 1 assigned to the maximum
value in d c and 0 to all others. Finally, the pseudo-cluster

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

8124

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Fig. 4. Illustration of the purpose of DCMU. Cluster centers calculated by
means and updated with momentum rely on randomness, leading to unfocused
results. DCMU focuses on difficult samples and uses pseudo-centers to push
nearby samples towards these challenging cases, resulting in tighter clusters.

c

center f pseudo for cluster c is obtained by applying weighted
aggregation to the selected features:
P c c
ws xs
s∈S c
pseudoc
.
(11)
f
=
P c c
ws xs

performance, we propose GCR. The core innovation of it
compared to GCN lies in incorporating camera prior knowledge into the graph construction process and ensuring stability
through multiple symmetric normalization operations during
the iterative phase. Experimental results demonstrate that
GCR achieves significant performance improvements without
substantially increasing computation time. Detailed results are
analyzed in the experiments section.
The original features of the combined query and gallery
sets are denoted by X u , with n images in total. We define a
graph G = (V, E) where each vertex vi in V = {v1 , . . . , vn }
corresponds to an image, and the edges E ∈ Rn×n represent the similarities between these images. Initially, cosine
distances are calculated to establish a similarity ranking R for
the images. Then, E is adjusted based on camera information:


if cami = cam j and j ∈ Ri
 λ,
E i j = 1/λ, if cami ̸ = cam j and j ∈ Ri ,
(14)


0,
otherwise

(13)

where sample j is a top k1 nearest neighbor of sample i, with
cami and cam j as their respective cameras. λ adjusts weights,
emphasizing inter-camera samples when smaller. The ranking
R and the matrix S, which stores the squared similarity scores
of the top k2 neighbors, are saved for use in the second stage.
In the second phase, to enhance retrieval performance, feature aggregation is necessary. Traditional re-ranking methods
use local query expansion [29], while GCR, like GCN [31],
performs iterative propagation on a GNN. Unlike GCN, which
only performs the conversion to an undirected graph and
normalization after initialization, we execute this conversion
process before each iteration, laying a solid foundation for
matrix propagation, as shown below:
 


1
⊤
·
E
+
(E)
,
(15)
Ẽ (0) = diag
∥E + (E)⊤ ∥1

Updating cluster centers using pseudo-centroids achieves
dual objectives. First, computing cosine distances between current samples and hard sample clusters from previous iterations
accelerates the convergence of easy samples toward hard sample regions. Second, formed through weighted aggregations of
historical hard instances, pseudo-centroids naturally lie closer
to hard samples, thereby recalibrating the hardness levels of
previously easy samples. This dynamic tug-of-war process
drives the model to learn noise-resistant feature representations
while reducing sensitivity to outliers.

where E (0) represents the adjacency matrix at the initial
iteration, and its transpose is denoted by (E (0) )⊤ . ∥ · ∥1
denotes the L1 norm, which calculates the sum of the elements
in each row of the matrix. diag(·) converts a vector into a
diagonal matrix, ensuring that each element of the vector
symmetrically scales the corresponding row of the matrix. This
process results in the matrix Ẽ (0) , which is both symmetrized
and normalized, ensuring uniformity and balance in node
relationships. The neighbor aggregation process for updating
the adjacency matrix to E (t) at the t-th iteration is as follows:

s∈S c

During the back propagation phase, the gradient of the loss
function L with respect to the input xic is computed as:
∂L
∂L ∂fc
∂L
= c · c ≈ c · fc ,
c
∂xi
∂f ∂xi
∂f

(12)

∂L
where ∂x
c refer to the gradient of the loss function L with
i
respect to the input feature xic , can be obtained by considering
both the direct gradient ∂∂fLc with respect to the cluster center
feature fc and the gradient of fc with respect to xic , denoted as
∂fc
, which represents fc . After completing the loss computa∂xic
tion, the cluster centers are updated as follows:
c

fc ← f pseudo .

D. Graph Network-Based Camera Constraint Adjustment
Re-Ranking
Re-ranking techniques improve retrieval accuracy by refining initial results. In the UVI-ReID task, traditional re-ranking
methods [29] can effectively enhance performance, but they
often come with high computational costs. Graph Convolutional Network [64] (GNN) based methods, GCN [31],
leverage GPU acceleration to significantly reduce computation
time, yet they do not achieve further performance improvements. To balance low computational cost and high retrieval

E (t) [i, j] =

kX
2 −1



S[i, m] · Ẽ (t−1) R(i, m), j ,

(16)

m=0

where E (t) [i, j] denotes the updated value between nodes i
and j. The aggregation sums over neighboring nodes, where
S[i, m] is the similarity weight between node i and its m-th
neighbor. R(i, m) gives the index of the m-th neighbor of node
i. Ẽ (t−1) [r (i, m), j] retrieves the symmetrized and normalized
adjacency value from the previous iteration. To facilitate
understanding, the complete process of post-processing using
GCR is provided in Algorithm 1.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

XIA et al.: CAMERA-PROXY ENHANCED IDENTITY-RECALIBRATION LEARNING FOR UVI-ReID

Algorithm 1 Re-Ranking Process of the Proposed GCR
Input: The union of query and gallery features X u and their
corresponding camera domains Cu .
Parameter: Hyper-parameters k1 , k2 , and T .
Output: Final distance matrix E.
1: Calculate the similarity matrix S and similarity ranking R
using X u , k1 and k2 .
2: Compute the adjacency matrix E using R and C u according to Eq.(14).
3: Initialize t = 0.
4: while t < T do
5:
Translate the adjacency matrix E into an undirected
graph and perform row normalization, by using Eq.(15).
6:
Update matrix E through message propagation with
GNN using S according to Eq.(16).
7:
Increment t
8: end while
9: return E.

E. Proxy-Enhanced Cross-Modal Mapping
Using only MMD for aligning feature distributions often
results in identity misalignment between visible and infrared
modalities. Previous methods, like PGM [27], address this
by clustering at the class level and establishing mappings
between modalities, but they overlook camera proxy information. To overcome this, we introduce the PCM module in the
second training stage. Building on the foundation of relatively
stable and reliable modal cohesive classes, this module constructs cross-modal correspondences at both the camera proxy
and class levels, significantly enhancing robustness.
First, we construct independent bipartite graph structures
for the proxy clusters and identity clusters obtained from
the infrared and visible modalities. This allows each proxy
cluster and identity cluster within the infrared modality to
find a corresponding cluster within the visible modality. For
the proxy clusters and identity clusters that do not form
pairs in the initial matching, we repeat the above process
until all clusters have corresponding matches. In class-based
matching, PGM has demonstrated that cross-modal bipartite
graph matching is essentially a binary linear programming
problem with linear constraints. The matching cost penalizes
feature differences between matched clusters across modalities, meaning that more similar cluster features result in
lower costs. The proposed PCM, which introduces additional
cross-modal camera proxies, also adheres to this principle.
Following PGM, we utilize the Hungarian algorithm [65] for
matching, and maintain its cost matrix as follows:
1
Cost (i, j) = −
,
exp(Dis(i, j))

(17)

where Cost (i, j) represents the cost function between the
proxies or classes of i and j, and Dis(i, j) denotes the distance measure between them, obtained from our GCR, which
takes additional consideration of camera domain information,
which can provide a more comprehensive measure. In Fig.5
provides a simplified diagram of the mapping process.

8125

Fig. 5.
Illustration of cross-modal mapping in PCM. PCM creates
cross-modal mappings between classes and camera proxies, significantly
boosting error robustness through leveraging consistency between these elements.

p

Secondly, after obtaining the mapping relationships Ti2r ,
between the two domains regarding the
proxy clusters and class clusters, using these mapping relationships to perform modal transformation on the pseudo labels
Pv , Pr , Cv , and Cr obtained through clustering:
p
c , and T c
Tr 2i , Ti2r
r 2i

p

Pv2r = Ti2r (Pv ),
Cv2r = Trc2i (Cv ),

p

Pr 2v = Tr 2i (Pr ),
Cr 2v = Trc2i (Cr ),

(18)

during stage 2, we not only use the pseudo-labels generated
by clustering to train within the same modality but also
utilize the generated mapping pseudo-labels to train across
modalities. Since the incorporation of the class hierarchy, the
calculation of the loss differs from stage 1. To begin with,
we compute the intra-modal loss, which is composed of two
distinct components: The proxy-level loss is performed using
our proposed Lintra
cdl , which utilizes Pv and Pr , while the
class-level loss is accomplished through Lintra
InfoNCE , which takes
Cv and Cr as inputs, as shown below:
intra
Lintra = Lintra
cdl + LInfoNCE ,

(19)

this combination of losses at different levels helps our model
learn better. The calculation for cross-modal loss and intramodal loss is essentially symmetrical, which ensures that the
updates of the infrared and visible light modes are balanced
during training. The difference lies in replacing the class-level
labels with Cv2r and Cr 2v , and the proxy-level labels with
Pv2r and Pr 2v . Algorithm 2 illustrates the operations of our
proposed PCM during the second phase of CEIL.
F. Unified Loss
We use Lcdl and LMMD to define the first stage loss as:
Lstage1 = Lvcdl + Lrcdl + LMMD ,

(20)

where the superscript of the Lcdl indicating its modality. And
define the loss of the second stage as follows:
Lstage2 = Lintra + Lcross ,

(21)

where Lintra and Lcr oss are incorporated into the computations
for both the visible and infrared modalities.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:15 UTC from IEEE Xplore. Restrictions apply.

8126

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Algorithm 2 Training Process for CEIL: Stage 2
Iv

Nv
{xiv }i=1

Ir

Nr
,
{xir }i=1

and
=
Input: Unlabeled image
=
feature extractor E trained after stage 1, memory bank M.
Parameter: Training epochs and iters.
Output: Feature extractor E trained post-stage 2.
1: for i = 1 to epochs do
2:
Use E to extract features from I v and I r , obtain the
feature sets F v and F r .
3:
Apply DBSCAN to cluster the features F v and F r for
each modality Cv and Cr .
4:
Partition the camera proxies into Pv and Pr according
to their respective camera sources.
p
p
c , and T c
5:
Obtain cross-modal mappings Ti2r , Tr 2i , Ti2r
r 2i
between camera proxies and class proxies by PCM.
6:
Obtain cross-modal mapping labels Pv2r , Pr 2v , Cv2r ,
and Cr 2v according to Eq.(18).
7:
for j = 1 to iters do
8:
Update M based on DCMU.
9:
Using P v , P r , Cv and Cr , calculate the intra-modal
loss Lintra according to Eq.(19).
10:
Similarly, using Pv2r , Pr 2v , Cv2r , and Cr 2v , calculate
the cross-modal loss Lcross .
11:
Optimize E to minimize the loss defined in Eq.(21).
12:
end for
13: end for
14: return E.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf' - | sed -n '311,858p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

4

(a) Framework of the CCIL

(b) Confounder Dictionary

Two-stream Network

Disentangle
Regularization

𝒙𝒊

𝑳𝒑𝒐

𝑭𝒄𝒍𝒕

𝑳kl

𝒇𝒄𝒍𝒕

𝑭𝒄𝒍𝒕

𝒇𝒄𝒍𝒕

Confounder
Dictionary

…

𝑴𝒄𝒍𝒕

𝑳𝒇𝒐

𝒄𝟏

𝒄𝟐

Intervention Module
Inference

Distance

𝝁𝒄𝒋

Sampling
𝑵(𝝁𝒄𝒋 , 𝝈𝟐𝒄𝒋 )

𝒇𝒄𝟏𝟏 ~𝒇𝒄𝒏𝑲

𝝍𝟏 ⋅

𝑭𝒊𝒎𝒈

𝒇𝒊𝒎𝒈

(d) Disentangle Regularization
Feature
center

element-wise add

Feature

Σ weighted sum

𝑷(𝒄𝒋𝟏 )~𝑷(𝒄𝒋𝑲 )

Causal
Classifier

…

𝒇𝒊𝒎𝒈

…
Update

element-wise product

𝝍𝟐 ⋅
Σ

𝑷(𝒄𝒏 )

…

ID-Clothe
Classifier

𝑷(𝒄𝟏𝟏 )~𝑷(𝒄𝒏𝑲 )

…

𝒄n

ID

𝑷(𝒄𝒋 )

Softmax

𝑷(𝒄𝟐 )

𝑴𝒊𝒎𝒈

Backdoor
Adjustment

(c) Intervention Module
𝒇𝒄j𝟏 ~𝒇𝒄𝒋𝑲

𝑷(𝒄𝟏 )

𝟏 − 𝑴𝒊𝒎𝒈

𝑭𝒊𝒎𝒈

𝑳int

𝒄𝟐

𝝁𝒄𝒍𝒕

𝒄𝒏

𝝈𝟐𝒄𝒍𝒕

𝒇𝒊𝒎𝒈

𝒄𝟏

𝑳int

𝑃(𝑌|𝑑𝑜 𝑋 )

𝒇𝒄𝒍𝒕

𝑳𝒑𝒐
𝑳𝒇𝒐
𝑳kl

Pull
close

𝒇𝒊𝒎𝒈

Push
away

Fig. 3. (a) The framework of the proposed Causal Clothes-Invariant Learning (CCIL). Three causal modules are complementary to each other and train
the model under the causal intervention framework, achieving better clothes-invariant features. (b) The Confounder Dictionary models clothing to support
the implementation of the subsequent causal intervention. (c) The Intervention Module collaborates with the established Confounder Dictionary to derive the
P (Y |do(x)). (d) The Disentangle Regularization improves the modeling of clothing in the Confounder Dictionary, thereby further enhancing the effectiveness
of the Intervention Module.

• During training, a given image would be fed into a twostream network to obtain image features and clothing features.
The clothing features are utilized to establish a Confounder
Dictionary iteratively (Sec. IV-A).
• With this, the Confounder Dictionary stores richer information about each cloth, which then interacts with the image
features to model causal intervention probability P (Y |do(X))
(Sec. IV-B).
• To further make the modeled intervention P (Y |do(X)) more
accurate, three losses are proposed to achieve clothes and identity disentangling, which improves the modeling of clothing in
the Confounder Dictionary and enhances the effectiveness of
the Intervention Module (Sec. IV-C).
The aforementioned three modules collectively implement
the causal intervention during training, leading to discriminative clothes-invariant feature learning.
A. Confounder Dictionary
We construct a Confounder Dictionary D to represent all
Nclt pieces of clothing in the training set, supporting the implementation of causal interventions in subsequent Equation 8.
As shown in Figure 3 (b), the Confounder Dictionary models
each clothing item ci as a multivariate Gaussian distribution
N (µci , σc2i ), where µci ∈ Rd and σc2i ∈ Rd represent the
mean and variance, respectively. Modeling the distribution of
each clothing item can significantly reduce storage overhead
compared to storing the clothing features of all images, and
it also provides more information than simply storing the

mean clothing features. Moreover, the Confounder Dictionary
also stores the probability of each clothing, denoted as P (ci ),
which can be calculated in the training set. The Confounder
Dictionary can be defined as follows:
clt
D = {(N (µci , σc2i ), P (ci )}N
i=1 .

(3)

2
of clothing
We calculate the mean µclt and variance σclt
features fclt within the training batch.
X
1
µclt =
fclt ,
N (c = ci ) c=c
i
(4)
X
1
2
(fclt − µclt )2 ,
σclt
=
N (c = ci ) c=c
i

where N (c = ci ) represents the number of samples with
clothing label ci in each training batch. Subsequently, based on
the clothing labels, we update the relevant µci and σc2i items of
the Confounder Dictionary by the exponential moving average
(EMA) scheme:
µci = α · µci + (1 − α) · µclt ,
2
σc2i = α · σc2i + (1 − α) · σclt
,

(5)

where α denotes the memory coefficient.
In addition, the clothing features fclt ∈ Rd used to construct
the Confounder Dictionary are extracted through spatial attention, which distinguish them from identity features fimg ∈ Rd :
fclt = P ool(Fclt ⊙ M1 (Fclt ) ⊙ (1 − M2 (Fimg ))) ,
fimg = P ool(Fimg ⊙ M2 (Fimg ) ,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

(6)

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

5

where Fclt ∈ Rd×h×w and Fimg ∈ Rd×h×w are the clothing
feature map and identity feature map output by the two-stream
network. M1 and M2 are implemented by a 2D convolutional
layer with an output dimension of 1, followed by sigmoid
activation. ⊙ denotes the element-wise product, P ool denote
the pooling operation applied along the spatial axes. This
strategy of multiplying the clothing feature map by the reverse
attention mask of the identity feature aids in accurately modeling confounders, ensuring that subsequent causal interventions
do not mistakenly eliminate identity-related information.
B. Intervention Module
To eliminate the interference of the clothing shortcut, an
intervention loss Lint is constructed by maximizing the intervention probability:

where pjk represents the sampling probabilities of different
fcik , which is computed by the probability density function
(PDF) of the multivariate Gaussian distribution:
K

X
1
1
exp(− DM (µcj , fcjk )),
pjk = exp(− DM (µcj , fcjk ))/
2
2
k=1

DM (µcj , fcjk ) = (fcjk − µcj )Σ−1 (fcjk − µcj )T ,
(11)

where DM (·, ·) is the Mahalanobis distance, Σ ∈ Rd×d
denotes the diagonal covariance matrix with diagonal elements
equal to σc2j .
To enhance the computational efficiency of Equation 8,
we utilize the Normalized Weighted Geometric Mean
(NWGM) [45] to approximate moving the classifier out of
accumulation operation:
X
P (Y |do(X = xi )) =
Cls[g(fimg , N (µcj , σc2j ))] · P (cj ),
cj

Lint = E [−log (P (Y = yi |do(X = xi ))] .

(7)

The intervention reflects the causality between variables X
and Y , uncorrelated with the confounder, which is consistent
with the clothes-invariant objective.
To achieve the intervention optimization in Equation 7, we
design an Intervention Module to implement the P (Y |do(X))
well within the CC-ReID framework.
As shown in Figure 3 (c), the Intervention Module takes the
image feature fimg of image xi and the Confounder Dictionary
D as its inputs, and then calculates the intervention probability
based on backdoor adjustment theory [10], [11] that we review
in Equation 1. With this module, the intervention probability
is calculated as follows:
X
P (Y |do(X = xi )) =
P (Y |X = xi , cj ) · P (cj )
cj

=

X

Cls[g(fimg , N (µcj , σc2j ))] · P (cj ),

cj

(8)
where N (µcj , σc2j ) and P (cj ) are achieved in the Confounder
Dictionary. Cls(·) = Sof tmax(Linear(·)) is a classifier
defined as a linear layer followed by a softmax activation
function. g(·) is used to produce conditional probability
P (Y |X, cj ) based on cj . Inspired by condition injection
methods [43], [44], we design g as follows:
g(fimg , N (µcj , σc2j )) = fimg ⊙ ψ1 (µcj ) + ψ2 (µcj ),

(9)

where ⊙ denotes the element-wise product, ψ1 and ψ2 are two
independent linear layers.
Equation 9 only utilizes the mean representation µcj of each
clothing item cj as a condition, which overlooks the variations
in clothing representation arise from changes in lighting, viewpoint and other factors in different environments. Therefore,
we sample K diverse features [fci1 ...fciK ] for each clothing
item from the distribution N (µcj , σc2j ) to comprehensively
account for the influence of clothing conditions. The improved
g(·) is represented as follows:
g(fimg , N (µcj , σc2j )) =

K
X
k=1

X
≈ Cls[
g(fimg , N (µcj , σc2j )) · P (cj )].
cj

(12)
This scheme reduces multiple classifications to a single operation, which significantly simplifies the computation of
intervention probability.
In conclusion, the Intervention Module can eliminate the influence of the clothing shortcut by simulating causal intervention P (Y |do(X)), ensuring clothes-invariant feature learning.
C. Disentangle Regularization
To achieve the expectation of modeling the intervention
probability P (Y |do(X)) by a deep learning implementation,
it is crucial to ensure the validity of Equation 8 as much as
possible. Therefore, as shown in Figure 3 (d), we designed
a disentangle regularization to ensure that the clothing representations are discriminative and do not contain identity clues.
This allows for the causal intervention to accurately cut off
the clothing shortcut without compromising the extraction of
identity features.
We propose a clothes-identity probabilistic orthogonal loss
Lpo to replace the original classification loss. Specifically, we
construct an ID-Clothes classifier with a total of Nclt + Nid
categories, where Nid represents the total human identity
categories and Nclt denotes the total clothing categories.
Subsequently, the image feature fimg and the clothing feature
fclt are fed into the ID-Clothes classifier to get the predicted
probability distributions:
pimg = Sof tmax([W1 ; W2 ]fimg ),
pcls = Sof tmax([W2 ; W1 ]fclt ),

where W1 ∈ Rd×Nid and W2 ∈ Rd×Nclt are the weights of
the classifier, and [·; ·] denotes concatenation. Finally, clothesidentity probabilistic orthogonal loss Lpo minimizes the crossentropy of the predicted probabilities and the identity labels
or clothing labels:
Lpo = E[−yi log(pimg )] + E[−ci log(pclt )].

(fimg ⊙ ψ1 (fcjk ) + ψ2 (fcjk ))pjk ,
(10)

(13)

(14)

This strategy results in clothes and images becoming negative categories of each other, with their features belonging to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:07 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3700883

6

different spaces. Consequently, the clothing features contain
distinct information from the image features, facilitating better
disentanglement.
To further disentangle clothing and identity, we employ an
additional KL-divergence loss Lkl :
′
′
Lkl = E[Dkl (Pimg ||Pimg
) + Dkl (Pclt ||Pclt
)],

(15)

where Dkl (·||·) denotes the KL divergence. Pimg is computed
by averaging the classification probabilities pimg from half of
the samples with the same identity label in the training batch,
′
while Pimg
denotes the average classification probabilities
′
from the other half of the samples. Pclt and Pclt
has a similar
meaning and calculation pipeline. So this loss term means that
the classification probabilities among positive samples should
be as similar as possible and reduce the interference of noisy
samples through probability averaging.
In addition, we design a metric learning loss, clothesidentity feature separation loss Lf s , that directly constrains
the Euclidean distance between features:
a
p
a
n
Lf s = E[Deu (fm
, fm
)] + E[ρ − Deu (fm
, fm
)]+ ,

(16)

where Deu (·, ·) is Euclidean distance, [·]+ = max(0, ·), ρ is
the margin parameter. And m ∈ {img, clt}, fm represents the
image/clothing feature centers from half of the samples with
the same identity/clothing label in the current mini-batch. f p
is the feature center of the positive samples for f a and f n is
the negative one.
Specifically, this loss aims to cluster positive features
closely together and distance negative features from each other.
Similar to our proposed Lpo loss, the Lf s loss conducts feature
learning in a shared metric space, where clothes and images
serve as negative categories for each other, emphasizing their
distinction. This strategy further enhances that the clothes
features should have different information from the image
ones, leading to better disentanglement.
D. Optimization
The whole model is trained end-to-end and the total loss
Ltotal of our method is defined as:
Ltotal = Lint + Lpo + Lkl + Lf s ,

(17)

We uniformly assigned a weight of 1 to all losses, without the
need for hyper-parameter searches, which has yielded good
results.
In addition, considering that the Confounder Dictionary may
not be accurate in the early stages of training, we do not apply
the intervention loss Lint during the warmup stage (first 10
epochs) to stabilize the training.
V. E XPERIMENTS

codex
还有一篇 FLaN-Net 的方法段已定位到 323 到 665 行，我继续抽出来。之后我会把每篇拆成你指定的五个问题，不写成模块复述。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf' - | sed -n '323,665p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Dynamic Weighting Fusion Module
To further enhance feature representation, we introduce a dynamic weighting fusion module that assigns adaptive weights
to each feature based on prediction uncertainty, allowing the
model to prioritize more reliable features. Specifically, the
global visual feature vg , the textual feature tp , and the crossattention feature satt are fed into the fusion module. For each
feature fm (m = 1, . . . , M , where M = 3, representing the
visual, textual, and cross-attention features), the associated
weight λm is determined according to the uncertainty of feature fm , which is quantified by the entropy of its prediction
distribution. The uncertainty Em is computed as follows:

local patch features, facilitating their interaction. The crossattention output H i for the learnable queries in the i-th attention block is then computed as:


Qd Kd⊤
i
√
H = CrossAttn(Qd , Kd , Vd ) = softmax
Vd .
d
(3)
Subsequently, we feed H i into a two-layer feed-forward network FFN(·), producing X̃ i , which represents the updated
features of the learnable queries after the i-th attention block.
This process is formulated as:


X̃ i = FFN H i + X̃ i−1 + H i .
(4)

Em = −
Afterwards, we perform average pooling over the refined
query embeddings after multiple transformer blocks. The result is then passed through a simple mapping network ϕa to
derive the final attribute token A∗ . It can be formulated as:
A∗ = ϕa (AvgPool(X̃f inal )),

pm (k) log pm (k),

(8)

k=1

where pm (k) is the softmax probability assigned to the k-th
individual’s feature fm , and C represents the total number of
individuals in the dataset. A lower Em reflects less uncertainty in the prediction, resulting in an increased weight for
that feature during the fusion process. The weight λm for
feature fm is then determined as follows:


exp
M ax (Em ) − Em
m=1,...,M
λm = M
(9)

.
P
exp
M ax (Em ) − Eq

(5)

where X̃f inal is the final output query embeddings from multiple transformer blocks, AvgPool(·) denotes average pooing, and ϕa denotes a three-layer feed-forward network.
The above network produces the refined attribute token A∗ .
Similarly, a parallel network with the same structure but different parameters is employed to generate the corresponding
context token O∗ . In this network, ϕo serves as the associated
mapping function.

3.2

C
X

m=1,...,M

q=1

The final fused representation fˆ is formulated as:
fˆ =

Multi-modal Feature Enhancement

M
X

λm f m ,

(10)

m=1

To effectively capture the intricate relationships between
multi-modal information, we integrate a cross-attention
mechanism with a dynamic fusion strategy, enhancing the
model’s capacity to represent nuanced identity features.

where fm represents each prediction feature. This adaptive
fusion approach enables balanced contributions from the visual, textual, and cross-attention features.

3.3

Cross-Attention Mechanism
The cross-attention mechanism allows the model to interactively align specific textual cues with corresponding visual regions, focusing on subject and detail tokens in the constructed
prompt. Let T represent the constructed prompt ”A photo of
a [S ∗ ] person with [A∗ ] partially occluded by [O∗ ]”, which
is then fed into the frozen CLIP text encoder ft to obtain the
textual representation tp . Formally, this process can be expressed as follows:
tp = ft (T ).
(6)

Loss Function and Inference

Our framework incorporates four loss functions: Crossmodal Contrastive Loss LAlign [Radford et al., 2021], Triplet
Loss LTriplet [Hermans et al., 2017], ID Classification Loss
LID [Zheng et al., 2017], and ArcFace Loss LArcFace [Deng
et al., 2019].
Cross-modal Contrastive Loss
To align visual and textual representations of each individual, we employ a cross-modal contrastive loss that encourages high similarity between images and their corresponding
prompts in the embedding space. It is formulated as:

To implement cross-attention, we treat the textual embedding
tp as the query Qc , while the visual feature v extracted from
the image serves as both the key Kc and value Vc . The output
of the cross-attention mechanism can be expressed as:


Qc Kc⊤
√
Vc .
satt = CrossAttn(Qc , Kc , Vc ) = softmax
d
(7)
By leveraging the cross-attention mechanism, the textual tokens dynamically guide the image encoder to focus on the visible and discriminative features of the pedestrian while suppressing noise from occluded regions.

LAlign = Li2t + Lt2i ,
X
exp (sim (vi , tp+ ) /τ )
,
Li2t (i) = −
log N
P
p+ ∈P (n)
exp (sim (vi , tn ) /τ )

(11)
(12)

n=1

Lt2i (i) = −

exp (sim (ti , vp+ ) /τ )
log N
,
P
p+ ∈P (n)
exp (sim (ti , vn ) /τ )
X

(13)

n=1

where P (n) denotes the set of positive samples that correspond to the same identity i and τ is a temperature parameter.

804

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Triplet Loss
The triplet loss is employed to increase the distinction between identities by minimizing the distance between positive
pairs while maximizing the distance between negative pairs.
This loss is formulated as:
LTriplet = max (dp − dn + m, 0) ,
(14)
where dp and dn are the distances of the positive and negative
pairs respectively and m is the margin.

Methods
PVPM (CVPR 20)
HOReID (CVPR 20)
RFCnet (TPAMI 21)
HCGA (TIP 23)
PAT (CVPR 21)
TransReID (ICCV 21)
DRL-Net (TMM 22)
PFD (AAAI 22)
DPM (ACM MM 22)
SAP (AAAI 23)
OAT (TIP 24)
OAMN (ICCV 21)
FED (CVPR 22)
CAAO (TIP 23)
ADP (AAAI 24)
DPM-SPT (AAAI 24)
CLIP-ReID (AAAI 23)
FLaN-Net (Ours)

ID Classification Loss
The ID classification loss ensures correct identification of
each individual and is defined as:
N
X
LID = −
qk log (yk ) ,
(15)
k=1

where yk is the predicted probability of an individual, and qk
is the corresponding ground truth label.
ArcFace Loss
ArcFace loss improves feature discrimination by introducing
an angular margin to enhance the separation between different individuals and compact intra-individual representations:
s cos θy +m))
PN
e ( ( k
LArcFace = − N1 k=1 log s cos θ +m
,
yk
(
(
)) +Pn
e
es cos θj

Occ-Duke
R-1 mAP
47.0 37.7
55.1 43.8
63.9 54.5
70.2 57.5
64.5 53.6
66.4 59.2
65.8 53.9
69.5 61.8
71.4 61.8
70.0 62.2
71.8 62.2
62.6 46.1
68.1 56.4
68.5 59.5
74.5 63.8
74.7 63.0
67.1 59.5
75.2 65.5

Occ-REID
R-1 mAP
66.8 59.5
80.3 70.2
81.6 72.1
81.5 83.0
85.5 79.7
83.0 76.8
82.6 78.2
86.3 79.3
87.1 83.4
89.2 85.1
87.8 81.1
92.6 89.5

Table 1: Performance comparison on Occluded-Duke and OccludedREID datasets. The compared methods are grouped into four categories: auxiliary model-based, transformer-based, data augmentation and language-guided.

j=1,j̸=yk

(16)
where θj denotes the angle between a feature and the weight
vector of the j-th identity. The angular margin m improves
discrimination, and the scale factor s stabilizes optimization.

4

Experiments

codex
CEIL 和 CCIL 的前 8 页把方法讲完整了，但关键消融在后面页。为了回答“证据闭环”，我再读它们的实验消融段，只取相关部分。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf' - | rg -n 'Ablation|ablation|NCC|shortcut|Intervention|Confounder|Disentangle|noisy|clothing|Table|Fig|causal'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
373:further proving the effectiveness of our causality-based work.
378:of clothing information to bring gains. Our method aims
380:clothing information, enabling generalization across diverse
381:scenarios, regardless of whether the clothing is changed.
384:of clothing items are shared, resembling the low-probability
386:outfit. Since most clothing is not shared, the clothing shortcuts
387:for inferring identity based on clothing will still significantly
388:interfere with the learning of clothes-invariant features. Furthermore, shared clothing does not affect the effectiveness of
389:the Confounder Dictionary and the Intervention Module, as
390:it equates to storing multiple representations for a clothing
391:item. As shown in Table III, CCIL achieves the best mAP
395:remains effective in the presence of clothing sharing. We also
396:point out that if the majority of clothing is shared among most
397:identities, the clothing shortcuts become very weak, although
403:training set, with 31 identities having only one set of clothing.
404:As depicted in Table III, CCIL achieves better or comparable
472:LTCC only includes 46 identities with multiple sets of clothing, which provides fewer clothing items, the modeling quality
473:of our Confounder Dictionary may be somewhat affected.
474:Results without Clothing Labels. DeepChange is a largescale CC-ReID dataset that does not provide clothing labels.
476:clothing labels. The results on DeepChange shown in Table IV
483:images). As shown in Table V, compared with the baseline,
488:Table VI, we combine CCIL with transformer architecture
496:E. Ablation Study
497:In this section, we conduct ablation studies on several CCReID datasets to evaluate the effectiveness of each detailed
498:part of the CCIL, including the Intervention Module, the
499:Confounder Dictionary, and the Disentangle Regularization.
500:Effectiveness of the Intervention Module. As shown in
501:Table VII, in the 1-st row, we establish a baseline, which is a
502:two-stream network without any causal modules and trained
504:clothing features. In the 2nd ∼ 6th rows, we introduce causal
514:Intervention Module
528:No Causal Intervention
585:Index Confounder Dictionary
661:Disentangle
749:the advantages of pursuing pure causal effects as opposed
754:Confounder Dictionary. Ultimately, compared to the baseline,
758:Effectiveness of the Confounder Dictionary. To further
759:prove that the reason for the performance gain is causality
761:of further experiments in Table VIII. In the 1st row, we use
827:No Causal Intervention
897:randomly initialized vectors as the values in the Confounder
899:treat the Confounder Dictionary as learnable parameters. The
901:we remove the causal meaning of intervention but keep the
902:training scheme unchanged, treating it as a feature enhancement without causal meaning. The results show that they
904:PRCC and DeepChange datasets compared with our causal
905:intervention method. When the Confounder Dictionary lacks
906:valid constraints for deriving clothing representations, the
907:optimization objective of the Intervention Module deviates
908:from causal intervention. In contrast, the results of the 3rd and
910:effectively constrain clothing representations is beneficial, as
912:causal intervention. Furthermore, in the 5th and 6th rows,
913:we attempt to directly store the clothing features fclt of all
921:Effectiveness of the Disentangle Regularization. The
922:experiments conducted in Table IX explore the effectiveness
923:of the Disentangle Regularization. Rows 2∼4 evaluate each
924:disentanglement loss independently under the same causalintervention framework. Compared with the baseline using
928:probabilistic orthogonality between identity and clothing categories, Lkl stabilizes the predicted distributions among positive samples, while the Lf s loss separates the clothing and
931:datasets, demonstrating their complementarity. With the Disentangle Regularization, the network can learn pure clothing
933:Consequently, the Confounder Dictionary can better model
934:the clothing C, and the Intervention Module can also better
991:Fig. 4. Effect of the hyper-parameter α, K and ρ on PRCC in the clothchanging setting. We first evaluated the α and K without the Disentangle
994:worse than the full model. This demonstrates that disentanglement cannot replace causal intervention. All these ablation
997:F. Robustness of Causal Intervention to Clothing Labels
998:In Section V-D, we use shooting dates as pseudo clothing
1000:sensitivity of causal intervention to clothing-label quality. The
1001:evaluated settings include within-ID clothing-label randomization at different ratios and generating pseudo clothing labels
1002:via clustering of clothing features. Since clothing labels in CCReID are defined per identity (distinguishing different outfits
1006:As shown in Table X, compared with no causal intervention, all settings achieve substantial improvements. Even
1007:under 100% within-ID clothing-label randomization, the performance only drops by 1.8 percentage points in Rank-1
1008:compared to using ground-truth clothing labels. This indicates
1011:each identity. Based on this, we use a simpler fully clothinglabel-free alternative: using identity labels as grouping units,
1014:identity to recover finer-grained clothing groups, achieving
1015:60.5% Rank-1 and 59.4% mAP, which is close to the groundtruth performance. These results demonstrate that our causal
1016:intervention framework has weak dependence on clothinglabel quality.
1017:Note that the Disentangle Regularization is designed to extract clothing representations and is more sensitive to clothinglabel quality. When clothing labels are unavailable, a pretrained human parsing model can be used to assist clothing
1051:Fig. 5. t-SNE [70] visualization of the distributions of image features fimg and clothes features fclt on the PRCC dataset. Different colors represent different
1058:Equation 5 to control the update speed of the Confounder
1062:0.999} and K from {1, 2, 4, 8, 16} without the Disentangle
1064:Figure 4 (a), setting α = 0.9 yields the highest performance. It
1066:in Figure 4 (b), setting K = 4 is sufficient, and a larger K is
1069:effectiveness of the Disentangle Regularization. We evaluated
1072:they require different margin parameters. As shown in Figure 4
1102:Fig. 6. Visualization of the activation feature maps on PRCC, and LTCC
1124:Figure 5 shows t-SNE [70] visualization results for image
1125:features fimg and clothing features fclt on the PRCC dataset.
1127:features (circles) and clothing features (triangles and stars) of
1131:distinguishes person features and clothing features, and it is
1136:Fig. 7. Visualization of the retrieval ranking lists on PRCC and LTCC datasets.
1140:In addition, some clusters of the baseline model are influenced by clothing, resulting in their division into two
1154:As shown in Figure 6, we visualize activation feature maps
1161:to stable causal relationships in recognition, which remain
1162:consistent regardless of variations in clothing or scenes.
1164:clothing-related regions, particularly prominent patterns on
1165:clothing. Meanwhile, our method still allocates slight attention
1166:to certain areas of clothing, as these regions also contain
1168:ignoring the clothing areas may result in the loss of certain
1171:in learning clothing-invariant features.
1173:Figure 7 illustrates some retrieval results of our proposed
1182:This paper studies the cloth-changing person Reidentification (CC-ReID) task from a novel causal
1184:clothing and identity in the dataset may interfere with the
1190:under the causal intervention framework, achieving better
1220:sketch under moderate clothing change,” IEEE Transactions on Pattern
1247:[17] Z. Pang, J. Wang, L. Zhao, and C. Wang, “Identity-clothing similarity
1248:modeling for unsupervised clothing change person re-identification,” in
1307:causal inference,” ACM Transactions on Knowledge Discovery from
1310:the good and removing the bad momentum causal effect,” arXiv preprint
1321:large language models: Opening a new frontier for causality,” arXiv
1324:conditions adaptation for object detection via causal intervention,” IEEE
1326:[39] Y. Li, X. Yang, X. Shang, and T.-S. Chua, “Interventional video relation
1329:[40] X. Lin, Z. Wu, G. Chen, G. Li, and Y. Yu, “A causal debiasing framework
1371:“Discovering causal signals in images,” in Proceedings of the IEEE/CVF
1375:the influence of clothing: universal clothing attribute disentanglement
1379:[54] V. D. Nguyen, P. Mantini, and S. K. Shah, “Contrastive clothing and pose
1403:[62] F. Liu, M. Kim, Z. Gu, A. Jain, and X. Liu, “Learning clothing and pose
1413:“Interventional video grounding with dual contrastive learning,” in
1417:causal intervention,” Advances in Neural Information Processing Systems, vol. 34, pp. 22 158–22 170, 2021.
1419:“Show, deconfound and tell: Image captioning with causal inference,”
1422:[68] T. Lin, Z. Yu, H. Hu, Y. Xu, and C.-W. Chen, “Interventional bag multiinstance learning on whole-slide pathological images,” in Proceedings of

exec
/bin/zsh -lc "pdftotext -f 8 -l 14 'Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | rg -n 'Ablation|ablation|CDL|DCMU|PCM|GCR|camera|proxy|re-ranking|component|Table|Fig'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
32:Partition the camera proxies into Pv and Pr according
33:to their respective camera sources.
40:between camera proxies and class proxies by PCM.
47:Update M based on DCMU.
67:2 infrared cameras. The dataset is divided into a training
70:images across all cameras, and “Indoor-search”, which restricts
71:matches to indoor cameras only. In both modes, infrared
73:one image per identity per camera is randomly selected to
112:shown in Table I. Meanwhile, the results for the LLCM dataset
113:are presented in Table II. Among these, CCLNet, PGM, and
117:post-processed using GCR. The sections above the horizontal
123:When enhanced with GCR, it significantly outperforms top
127:learning. However, with the addition of GCR, CEIL remains
130:GCR and the current state-of-the-art supervised methods on
132:about by post-processing highlights CEIL’s potential and effectiveness. This also emphasizes the critical role of GCR, which
135:proposed GCR, the CEIL model we developed significantly
139:of leveraging readily accessible camera information. Obtaining
151:I N A LL TABLES , “∗” D ENOTES GCR P OST-P ROCESSING , AND “†” I NDICATES THE M ETHOD U SING C AMERA P ROXY. U NMARKED
161:C. Ablation Studies
166:logical step. With the application of our GCR technique, the
170:applying GCR, we achieved over 60% mAP on the particularly
173:We conducted ablation studies on the RegDB, LLCM, and
175:component in the training of the CEIL model. The ablation
177:Table III, while the experiments on LLCM are shown in
178:Table IV. In these tables, “Baseline” refers to the baseline
180:In Table III and Table IV, “Time” denotes training time.
181:In Table VII, it refers to post-processing time during testing.
182:1) Effectiveness of PCM: The comparative analysis of Strategy 2 versus Strategy 4, and Strategy 3 versus Strategy 5,
183:substantiates the efficacy of the PCM module in enhancing
185:modality gap. To corroborate the utility of camera proxy
187:SYSU-MM01 dataset, as delineated in Table V. The experimental results demonstrate that combining camera proxies
190:2) Effectiveness of DCMU: The comparisons between
200:T HE I NFLUENCE OF E ACH C OMPONENT ON THE P ERFORMANCE OF THE CEIL ON SYSU-MM01 AND R EG DB. “PCM” D ENOTES P ROXYE NHANCED C ROSS -M ODAL M APPING . “DCMU” D ENOTES D IFFICULTY-AWARE C LUSTER -BASED M EMORY U PDATING . “Lcdl ” D ENOTES
201:C AMERA D OMAIN -D RIVEN L OCAL A SSOCIATION L OSS . “GCR” D ENOTES G RAPH N ETWORK -BASED
215:P ROPOSED GCR
222:AND “PCM” D EMONSTRATES THE P ERFORMANCE OF
226:traditional momentum update strategy with DCMU, whether
228:all datasets. The robust capability of DCMU in identifying
231:3) Effectiveness of CDL Loss: The comparison between
232:Baseline and Strategy 2 shows that adding the CDL loss alone
234:samples are captured by a single camera and thus lack the
236:benefits of camera information. Despite this, the CDL loss still
239:4) Effectiveness of GCR During Test: GCR post-processing
243:plug-and-play re-ranking module, GCR is highly versatile
244:in existing methods. Table VI shows consistent performance
247:for post-processing validated the effectiveness of GCR by
249:As shown in Table VII. Like GCN [31], GCR uses GPU
251:Incorporating camera information, GCR builds a more reliable
266:GCR, I NCLUDING P ERFORMANCE AND C OMPUTATIONAL T IME . A LL
269:I LLUSTRATION OF H YPERPARAMETERS : F OR THE DCMU, Lcdl , AND
270:GCR. T HE S UPERSCRIPTS I NDICATE U SAGE P HASES
273:A BLATION S TUDY OF “GCR” C OMPONENTS . “MSN” R EFERS TO THE
278:ablation studies on the components of GCR as shown in
279:Table VIII, where “MSN” indicates symmetric normalization
281:incorporation of camera information. The results highlight the
282:positive impact of each component on GCR.
283:5) Cost Analysis: During training, CDL Loss transforms
284:class domains into camera domains for local smoothing,
289:single-camera domains such as RegDB, it introduces a slight
290:overhead of approximately 30 seconds. Compared to traditional momentum updates, DCMU creates pseudo-centroids
293:Baseline and Strategy 1. PCM introduces a second training
298:During testing, GCR based on cosine distance constructs
300:utilizing camera information. Both the time complexity and
308:our proposed GCR significantly outperforms traditional cosine
311:6) Synergy: The efficacy of the CDL loss hinges on the
313:different cameras. In more challenging datasets, this becomes
315:InfoNCE loss with the CDL loss in the LLCM does not yield
318:scenarios, the CDL loss functions more like a local smoothing
319:loss. However, when combined with DCMU, which enhances
320:the model’s clustering capability, the CDL loss is better aligned
322:between Strategy 3 and Strategy 1, where integrating the CDL
323:loss with DCMU results in a substantially greater performance
324:boost than introducing the CDL loss alone. This effect is even
326:7) Limitation: The quality of initial clustering significantly influences the performance gains of GCR. Ablation
327:experiments show that while GCR post-processing consistently improves performance across strategies 1∗ to 5∗ , the
330:to greater post-optimization gains. GCR exhibits computational complexity comparable to cosine distance, enabling its
335:model performance. Table IX shows the optimal hyperparameter values for the SYSU-MM01, RegDB, and LLCM datasets.
340:1) Study the Hyperparameter of CDL Loss: The hyperparameter n represents the number of nearest negative proxies
342:impact of varying values of n. As shown in Fig.6(a), the
358:Fig. 6. Illustration of hyperparameters adjustment. Including the hyperparameter n for CDL loss, the hyperparameter α for DCMU, and the hyperparameters
359:k1 , k2 , and λ for GCR, mAP is calculated by averaging across two retrieval modes, using the first partition of the test set as a validation set for optimization.
361:2) Study the Hyperparameter of DCMU: In DCMU, the
367:samples. Our experiments, as shown in Fig.6(b), indicate that
369:3) Study the Hyperparameters of GCR: The hyperparameters k1 , k2 , and λ are key to optimizing GCR’s performance.
372:adjustments between inter-camera and intra-camera samples.
376:Fig.6(c) shows the impact of adjusting λ, and Fig.6(d-f) shows
390:Fig. 7. Visualization of feature distribution. The first row displays t-SNE
402:Fig.7(a-c) for compare the baseline method, the first stage of
411:cross-modal training at both the camera proxy and class levels,
413:we apply GCR for post-processing to optimize the test results,
414:with GCR recalibrating cross-modal identity matching. Extensive experiments validate the exceptional performance of our
421:Fig. 8. Visualization of retrieval results. The leftmost image is an infrared
428:remained suboptimal. After introducing the PCM module for
433:negative pairs for these models, as shown in Fig. 7(d-f).
442:of our proposed CEIL and GCR, we obtained the retrieval
444:them using GCR, as shown in Fig.8. Despite the significant
447:GCR for post-processing, the features extracted by CEIL are
448:enhanced by incorporating prior camera information and iteratively processing on a graph network. This greatly enriches
453:first stage, we use CDL loss and the DCMU strategy to train
454:only at the camera proxy level. This ensures the establishment
455:of reasonable intra-modal clustering associations. In the second stage, building on the first stage, we incorporate PCM for
487:[11] D. Zheng, J. Xiao, M. Sun, H. Bai, and J. Hou, “Plausible proxy mining
495:cameras,” Sensors, vol. 17, no. 3, p. 605, Mar. 2017.
551:cross neighborhood re-ranking,” in Proc. IEEE/CVF Conf. Comput. Vis.
554:“Understanding image retrieval re-ranking: A graph neural network
639:convolution based efficient re-ranking for visual retrieval,” IEEE Trans.

codex
我按清单实际的 6 篇拆。括号里写“4”应该是笔误。

**1. Bidirectional modality information interaction for Visible-Infrared Person Re-identification**

1. 触发观察：作者抓住的不是“可见光和红外差异大”这个老问题，而是现有方法虽然在缩小模态差异，却基本把两个模态当成各自独立的流来处理。生成中间模态、灰度桥、共享特征和私有特征解耦，本质都是绕开模态差异，而不是让两个模态直接交换有用信息。同时他们又补了两个更细的观察：GAP 和 GMP 各有偏好，简单相加会削弱局部显著性；同模态内长得很像的不同人，在普通 triplet 或中心损失里监督不够。

2. 重定义动作：他们把“跨模态差异”重说成“跨模态信息没有双向交互”。这一步很关键，旧问题是对齐，新问题是交互；旧关键词是 modality gap、shared feature，新关键词变成 bidirectional modality information interaction、dynamic aggregation、feature intersection。这样一改，论文不再只是说“我要减少差异”，而是说“两个模态本来有轮廓、姿态等可共享线索，应该互相帮助”。

3. 机制怎么长出来：Dynamic Aggregation 是从这个重定义里比较自然长出来的。既然问题是两个模态不沟通，那就把一个模态当 primary，另一个当 auxiliary，用 auxiliary 动态生成投影权重去重表征 primary，再反向做一次。Feature Intersection 和 Multi-Weight loss 的逻辑绑定弱一些，它们分别来自“全局和局部信息没有正确组合”和“相似负样本监督不足”，更像是围绕主机制补强特征质量和判别边界。

4. 证据闭环：关键证据是三块消融：DA 单独加上能明显提升，说明“模态交互”不是空话；FI 对 GAP、GMP、GAP+GMP 的对比说明交叉式组合确实优于朴素池化；MW 的权重分组实验说明相似负样本需要更强监督。t-SNE 可视化用来证明同身份跨模态距离缩小、模态分裂减弱。它的证据够支撑“机制有效”，但对“交互真的传递了互补语义”证明还不算很硬，如果我们学它，最好再加单向交互、随机辅助模态、错配辅助模态这类破坏性对照。

5. reviewer 为什么买账：这篇主要卖机制，视角有但不算特别深。reviewer 会买，是因为它把老的 VI-ReID 模态差异问题换成了一个容易接受的新动作：不要只对齐，要双向交换。真正 novelty 来自“直接用另一模态生成动态投影权重”，不是 GAP/GMP 或 weighted loss。

**2. Bridging the gap: Learning adaptive knowledge transition for lifelong person re-identification**

1. 触发观察：作者注意到 LReID 里现有知识蒸馏大多是把上一阶段模型冻结成 reference model，然后让当前模型和旧模型输出一致。这个做法能抗遗忘，但很粗暴，因为不同阶段的数据域偏差很大，直接蒸馏会把旧知识硬压到新任务上，反过来抑制新任务泛化。也就是说，问题不是“有没有旧知识”，而是“旧知识怎么过渡到新域”。

2. 重定义动作：他们把 catastrophic forgetting 重定义成 adaptive knowledge transition 问题。旧说法是 preserve previous knowledge，新说法是 control the degree and direction of knowledge distillation。关键词从 distillation、forgetting，换成 transition path、intermediate knowledge、adaptive transfer。这样一来，论文的中心不再是“再加一个蒸馏损失”，而是“旧知识和新知识之间需要一座可学习的桥”。

3. 机制怎么长出来：KTL 很自然。既然直接从旧模型到新模型太硬，就先用 TFE 从 current model 和 reference model 的浅层特征里估计 domain-specific style，再用 KIM 融合成 intermediate feature，作为中间过渡知识。MRD 也顺着这个思路走：不只蒸馏单点 feature，而是蒸馏样本间关系和样本到类中心关系。机制和重定义绑定较紧，尤其是“中间知识”这一步，是论文叙事的骨架。

4. 证据闭环：最关键的消融是 KTL 带来最大 mAP 增益，这直接支撑“transition 比直接蒸馏更好”。另一个很好的证据是 TFE 嵌入位置分析，越靠浅层效果越好，说明他们说的 domain-specific style 不是事后硬编，而是和浅层风格信息对应。CBD/IID 的消融说明样本关系和中心关系都在补足普通 feature distillation。两种训练顺序的结果则证明不是某个 domain order 的偶然。

5. reviewer 为什么买账：这篇卖的是视角加机制。novelty 的核心不是“又一个 lifelong 蒸馏模块”，而是把旧知识保存讲成“知识过渡路径选择”。这个叙事很适合 LReID，因为它正好解释了稳定性和可塑性的矛盾。

**3. Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification**

1. 触发观察：作者抓的是无监督 ReID 里一个很实在的问题：聚类和对比学习生成了伪标签，但训练时忽略了相机之间和相机内部的语义关系。跨相机同身份会因为光照、背景、分辨率产生大域偏移；同相机内不同身份又可能非常相似。过去用相机风格迁移做数据增强，但生成图像和 ReID 判别学习是割裂的，还可能引入噪声。

2. 重定义动作：他们把无监督 ReID 从“伪标签不准”重定义成“多相机子域之间缺少语义传播”。每个 camera 被定义成一个 domain，训练不只是 cluster contrastive learning，而是 camera-aware multi-domain adaptive learning。关键词是 camera-aware graph、multi-domain、semantic propagation、domain-invariant representation。

3. 机制怎么长出来：机制很顺。既然每个相机是一个 domain，就在 mini-batch 里每个相机采样，构建异质图。跨相机同伪身份给正边，同相机不同伪身份给负边，然后用 GCN 把这些关系传播到统一空间。最后用对抗训练把 GCN 学到的“理想域不变表示”迁回普通 feature extractor。这个机制和重定义绑定很紧，图的边定义就是论文问题定义的具体实现。

4. 证据闭环：关键消融是 inter-camera correlation、intra-camera correlation 和完整 CGMAL 的比较。跨相机边单独有效，说明“相机间语义传播”是主收益；同相机负关系也有效，说明“相机内相似身份区分”不是摆设。不同 backbone、GCN/GAT 替换、检索可视化共同证明它不是某个网络结构偶然涨点，而是 camera-aware graph 这个结构在起作用。

5. reviewer 为什么买账：这篇卖的是视角。GCN 不新，memory contrastive 也不新，但“把 camera 当成多个无标签子域，并用图显式传播伪身份关系”是清楚的。reviewer 买账的点在于它把 camera metadata 从普通 nuisance 变成了可用结构。

**4. Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification**

1. 触发观察：作者看到 UVI-ReID 里有两层不稳定：第一，模态差异大，跨模态类别映射容易错；第二，同一模态内部还有 camera domain 差异，导致类内不紧、类间混。直接做跨模态聚类或类别级匹配太早，会在基础还不稳定时引入错误关系。还有一个工程层面的观察：普通 memory momentum update 容易被 easy samples 主导，hard samples 长期没被修正。

2. 重定义动作：他们把问题从“无监督跨模态伪标签噪声”重定义成“identity relationship 需要 camera-proxy 逐级校准”。旧问题是 class-level cross-modal mapping，新问题是先建立模态内 camera proxy，再做 class 和 camera proxy 双层跨模态映射。关键词是 camera-proxy、identity-recalibration、local association、proxy-enhanced cross-modal mapping。

3. 机制怎么长出来：两阶段训练是从重定义自然推出来的。第一阶段只做 camera proxy 级别的模态内对比，用 CDL 在每个相机域内找局部 hard negatives，先把单模态结构稳住。DCMU 更新 memory 时偏向 hard samples，让簇中心不被 easy samples 拖走。第二阶段再用 PCM 通过匈牙利匹配建立可见光和红外之间的 class/proxy 双层对应。GCR 测试时再把 camera 信息放进图重排序，属于对“camera 信息全流程使用”的延伸。

4. 证据闭环：核心消融是 CDL、DCMU、PCM、GCR 分别加入后的提升。PCM 的证据对应“只做类级匹配不够，proxy 级一致性有用”；DCMU 的证据对应“hard samples 主导更新能改善簇”；CDL 的证据对应“camera-local contrastive 比全局乱拉更稳”。可视化部分包括 t-SNE 和检索列表，说明第一阶段让模态内结构更稳，第二阶段和 GCR 改善跨模态身份匹配。这里证据闭环比较工程化，但和每个问题点基本对应。

5. reviewer 为什么买账：这篇卖的是系统化视角，不是单一模块。camera proxy 之前有人用过，跨模态图匹配也有人做过，所以 novelty 不在“第一次使用 camera 信息”，而在“把 camera proxy 作为 UVI-ReID 身份校准的中间层，并做成两阶段流程”。reviewer 买账的前提是每个组件的消融都能说明它解决了一个具体不稳定源。

**5. Categorical Attention: Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification**

1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。

2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。

3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。

4. 证据闭环：最强证据是 prompt ablation：从普通 “a person”，到加 S*，再加 A*，最后加 O*，性能逐步提升，直接证明这三个语义类别不是装饰。组件消融 F、C、D 说明细粒度 prompt、cross-attention、动态融合各自有效。attention map 可视化也很关键，PFD 和 CLIP-ReID 仍会看遮挡物，而 FLaN-Net 更集中在可见人体区域。这类可视化正好证明“重定义是对的”，不是只看涨点。

5. reviewer 为什么买账：这篇主要卖视角。CLIP 用在 ReID 已经不是新鲜事，learnable token 也不是新鲜事，但它把遮挡拆成“主体、可见属性、遮挡物”三个语言槽位，这个问题刻画很清楚。reviewer 买账的是“语言不只是全局语义监督，而是细粒度噪声过滤器”。

**6. Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID**

1. 触发观察：作者抓住的是换衣 ReID 最本质的捷径：训练集中衣服通常属于某个固定身份，衣服几乎不被不同身份共享。模型为了最小化训练损失，会自然学到 clothing shortcut，也就是用衣服认人。现有方法即使用轮廓、解析、3D、gait 等辅助信息，也往往没有完全抛弃 RGB，所以只是稀释衣服捷径，而不是切断它。

2. 重定义动作：这是这批里最强的重定义之一。他们不再说“我们要学 clothes-invariant feature”，而是说普通训练在学 P(Y|X)，这个目标本身会吸收衣服和身份的伪相关；真正应该学的是 P(Y|do(X))，通过 causal intervention 切断 X <- C -> Y 的 backdoor path。关键词是 clothing ownership、clothing shortcut、spurious correlation、causal intervention、backdoor adjustment、confounder dictionary。

3. 机制怎么长出来：机制和重定义绑定非常紧。既然衣服 C 是 confounder，就要建 Confounder Dictionary，存每件衣服的分布和先验 P(c)。既然目标是 P(Y|do(X))，Intervention Module 就按 backdoor adjustment 对所有衣服条件求和，而不是只看当前图像衣服。既然 intervention 依赖“衣服表示必须是衣服而不是身份”，就加 Disentangle Regularization，把 identity feature 和 clothing feature 拉开。这里不是先有模块再找故事，而是每个模块都能对应 causal graph 里的变量。

4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。

5. reviewer 为什么买账：这篇卖的是视角，而且视角强于模块。模块本身是 dictionary、feature injection、disentanglement，不算罕见；真正 novelty 是把 CC-ReID 的失败原因从“衣服变化太大”改写成“训练目标被衣服所有权伪相关污染”。reviewer 会买，因为它解释了为什么很多 clothes-agnostic 方法仍然会失败，并且给了能验证 shortcut 是否减弱的指标和可视化。

最后看这批论文的共同套路：它们不是从“我要加一个模块”开始，而是先抓一个具体失败现象，然后把老问题换成一个更有结构的新问题，例如交互、过渡、语义传播、身份校准、噪声过滤、因果干预。机制好发的关键，是新问题一旦说清楚，模块几乎像是在实现这个新问题里的变量和关系。最值得学的是证据闭环：不要只报涨点，要有能证明“这个新问题定义确实成立”的消融、破坏性对照或可视化。
tokens used
143,732
我按清单实际的 6 篇拆。括号里写“4”应该是笔误。

**1. Bidirectional modality information interaction for Visible-Infrared Person Re-identification**

1. 触发观察：作者抓住的不是“可见光和红外差异大”这个老问题，而是现有方法虽然在缩小模态差异，却基本把两个模态当成各自独立的流来处理。生成中间模态、灰度桥、共享特征和私有特征解耦，本质都是绕开模态差异，而不是让两个模态直接交换有用信息。同时他们又补了两个更细的观察：GAP 和 GMP 各有偏好，简单相加会削弱局部显著性；同模态内长得很像的不同人，在普通 triplet 或中心损失里监督不够。

2. 重定义动作：他们把“跨模态差异”重说成“跨模态信息没有双向交互”。这一步很关键，旧问题是对齐，新问题是交互；旧关键词是 modality gap、shared feature，新关键词变成 bidirectional modality information interaction、dynamic aggregation、feature intersection。这样一改，论文不再只是说“我要减少差异”，而是说“两个模态本来有轮廓、姿态等可共享线索，应该互相帮助”。

3. 机制怎么长出来：Dynamic Aggregation 是从这个重定义里比较自然长出来的。既然问题是两个模态不沟通，那就把一个模态当 primary，另一个当 auxiliary，用 auxiliary 动态生成投影权重去重表征 primary，再反向做一次。Feature Intersection 和 Multi-Weight loss 的逻辑绑定弱一些，它们分别来自“全局和局部信息没有正确组合”和“相似负样本监督不足”，更像是围绕主机制补强特征质量和判别边界。

4. 证据闭环：关键证据是三块消融：DA 单独加上能明显提升，说明“模态交互”不是空话；FI 对 GAP、GMP、GAP+GMP 的对比说明交叉式组合确实优于朴素池化；MW 的权重分组实验说明相似负样本需要更强监督。t-SNE 可视化用来证明同身份跨模态距离缩小、模态分裂减弱。它的证据够支撑“机制有效”，但对“交互真的传递了互补语义”证明还不算很硬，如果我们学它，最好再加单向交互、随机辅助模态、错配辅助模态这类破坏性对照。

5. reviewer 为什么买账：这篇主要卖机制，视角有但不算特别深。reviewer 会买，是因为它把老的 VI-ReID 模态差异问题换成了一个容易接受的新动作：不要只对齐，要双向交换。真正 novelty 来自“直接用另一模态生成动态投影权重”，不是 GAP/GMP 或 weighted loss。

**2. Bridging the gap: Learning adaptive knowledge transition for lifelong person re-identification**

1. 触发观察：作者注意到 LReID 里现有知识蒸馏大多是把上一阶段模型冻结成 reference model，然后让当前模型和旧模型输出一致。这个做法能抗遗忘，但很粗暴，因为不同阶段的数据域偏差很大，直接蒸馏会把旧知识硬压到新任务上，反过来抑制新任务泛化。也就是说，问题不是“有没有旧知识”，而是“旧知识怎么过渡到新域”。

2. 重定义动作：他们把 catastrophic forgetting 重定义成 adaptive knowledge transition 问题。旧说法是 preserve previous knowledge，新说法是 control the degree and direction of knowledge distillation。关键词从 distillation、forgetting，换成 transition path、intermediate knowledge、adaptive transfer。这样一来，论文的中心不再是“再加一个蒸馏损失”，而是“旧知识和新知识之间需要一座可学习的桥”。

3. 机制怎么长出来：KTL 很自然。既然直接从旧模型到新模型太硬，就先用 TFE 从 current model 和 reference model 的浅层特征里估计 domain-specific style，再用 KIM 融合成 intermediate feature，作为中间过渡知识。MRD 也顺着这个思路走：不只蒸馏单点 feature，而是蒸馏样本间关系和样本到类中心关系。机制和重定义绑定较紧，尤其是“中间知识”这一步，是论文叙事的骨架。

4. 证据闭环：最关键的消融是 KTL 带来最大 mAP 增益，这直接支撑“transition 比直接蒸馏更好”。另一个很好的证据是 TFE 嵌入位置分析，越靠浅层效果越好，说明他们说的 domain-specific style 不是事后硬编，而是和浅层风格信息对应。CBD/IID 的消融说明样本关系和中心关系都在补足普通 feature distillation。两种训练顺序的结果则证明不是某个 domain order 的偶然。

5. reviewer 为什么买账：这篇卖的是视角加机制。novelty 的核心不是“又一个 lifelong 蒸馏模块”，而是把旧知识保存讲成“知识过渡路径选择”。这个叙事很适合 LReID，因为它正好解释了稳定性和可塑性的矛盾。

**3. Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification**

1. 触发观察：作者抓的是无监督 ReID 里一个很实在的问题：聚类和对比学习生成了伪标签，但训练时忽略了相机之间和相机内部的语义关系。跨相机同身份会因为光照、背景、分辨率产生大域偏移；同相机内不同身份又可能非常相似。过去用相机风格迁移做数据增强，但生成图像和 ReID 判别学习是割裂的，还可能引入噪声。

2. 重定义动作：他们把无监督 ReID 从“伪标签不准”重定义成“多相机子域之间缺少语义传播”。每个 camera 被定义成一个 domain，训练不只是 cluster contrastive learning，而是 camera-aware multi-domain adaptive learning。关键词是 camera-aware graph、multi-domain、semantic propagation、domain-invariant representation。

3. 机制怎么长出来：机制很顺。既然每个相机是一个 domain，就在 mini-batch 里每个相机采样，构建异质图。跨相机同伪身份给正边，同相机不同伪身份给负边，然后用 GCN 把这些关系传播到统一空间。最后用对抗训练把 GCN 学到的“理想域不变表示”迁回普通 feature extractor。这个机制和重定义绑定很紧，图的边定义就是论文问题定义的具体实现。

4. 证据闭环：关键消融是 inter-camera correlation、intra-camera correlation 和完整 CGMAL 的比较。跨相机边单独有效，说明“相机间语义传播”是主收益；同相机负关系也有效，说明“相机内相似身份区分”不是摆设。不同 backbone、GCN/GAT 替换、检索可视化共同证明它不是某个网络结构偶然涨点，而是 camera-aware graph 这个结构在起作用。

5. reviewer 为什么买账：这篇卖的是视角。GCN 不新，memory contrastive 也不新，但“把 camera 当成多个无标签子域，并用图显式传播伪身份关系”是清楚的。reviewer 买账的点在于它把 camera metadata 从普通 nuisance 变成了可用结构。

**4. Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification**

1. 触发观察：作者看到 UVI-ReID 里有两层不稳定：第一，模态差异大，跨模态类别映射容易错；第二，同一模态内部还有 camera domain 差异，导致类内不紧、类间混。直接做跨模态聚类或类别级匹配太早，会在基础还不稳定时引入错误关系。还有一个工程层面的观察：普通 memory momentum update 容易被 easy samples 主导，hard samples 长期没被修正。

2. 重定义动作：他们把问题从“无监督跨模态伪标签噪声”重定义成“identity relationship 需要 camera-proxy 逐级校准”。旧问题是 class-level cross-modal mapping，新问题是先建立模态内 camera proxy，再做 class 和 camera proxy 双层跨模态映射。关键词是 camera-proxy、identity-recalibration、local association、proxy-enhanced cross-modal mapping。

3. 机制怎么长出来：两阶段训练是从重定义自然推出来的。第一阶段只做 camera proxy 级别的模态内对比，用 CDL 在每个相机域内找局部 hard negatives，先把单模态结构稳住。DCMU 更新 memory 时偏向 hard samples，让簇中心不被 easy samples 拖走。第二阶段再用 PCM 通过匈牙利匹配建立可见光和红外之间的 class/proxy 双层对应。GCR 测试时再把 camera 信息放进图重排序，属于对“camera 信息全流程使用”的延伸。

4. 证据闭环：核心消融是 CDL、DCMU、PCM、GCR 分别加入后的提升。PCM 的证据对应“只做类级匹配不够，proxy 级一致性有用”；DCMU 的证据对应“hard samples 主导更新能改善簇”；CDL 的证据对应“camera-local contrastive 比全局乱拉更稳”。可视化部分包括 t-SNE 和检索列表，说明第一阶段让模态内结构更稳，第二阶段和 GCR 改善跨模态身份匹配。这里证据闭环比较工程化，但和每个问题点基本对应。

5. reviewer 为什么买账：这篇卖的是系统化视角，不是单一模块。camera proxy 之前有人用过，跨模态图匹配也有人做过，所以 novelty 不在“第一次使用 camera 信息”，而在“把 camera proxy 作为 UVI-ReID 身份校准的中间层，并做成两阶段流程”。reviewer 买账的前提是每个组件的消融都能说明它解决了一个具体不稳定源。

**5. Categorical Attention: Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification**

1. 触发观察：作者先抓住两个失败现象。传统遮挡 ReID 依赖姿态、人体解析、attention 或遮挡增广，但它们都只看视觉，容易把遮挡物也当成身份特征。CLIP-ReID 引入语言后仍然不够，因为 prompt 是全局粗粒度的，文本主要用于对比监督，并没有直接指导模型区分“人、可见属性、遮挡物”。

2. 重定义动作：他们把遮挡问题重定义成 language-guided noise filtering，而不是简单的 visible part matching。更关键的是，他们把图像内容拆成三个语义类别：主体身份、可见属性、遮挡物。关键词是 categorical attention、fine-grained prompt、occlusion-aware prompt、subject token、attribute token、context token。这个重定义把“遮挡”从一个空间缺失问题变成了一个语义分类和噪声过滤问题。

3. 机制怎么长出来：机制基本是顺着这个重定义长出来的。主体 token S* 来自全局图像特征，表示“这是谁”；属性 token A* 由 learnable queries 从 patch 中提取可见身份属性；上下文 token O* 提取遮挡物特征。三者组成 “A photo of a [S*] person with [A*] partially occluded by [O*]” 这样的细粒度 prompt，再用文本作为 query 对视觉 patch 做 cross-attention，引导图像特征看可见人体区域、压掉遮挡物。动态融合用预测不确定性给 visual、text、cross-attention feature 分配权重，是对可靠性问题的补强。

4. 证据闭环：最强证据是 prompt ablation：从普通 “a person”，到加 S*，再加 A*，最后加 O*，性能逐步提升，直接证明这三个语义类别不是装饰。组件消融 F、C、D 说明细粒度 prompt、cross-attention、动态融合各自有效。attention map 可视化也很关键，PFD 和 CLIP-ReID 仍会看遮挡物，而 FLaN-Net 更集中在可见人体区域。这类可视化正好证明“重定义是对的”，不是只看涨点。

5. reviewer 为什么买账：这篇主要卖视角。CLIP 用在 ReID 已经不是新鲜事，learnable token 也不是新鲜事，但它把遮挡拆成“主体、可见属性、遮挡物”三个语言槽位，这个问题刻画很清楚。reviewer 买账的是“语言不只是全局语义监督，而是细粒度噪声过滤器”。

**6. Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID**

1. 触发观察：作者抓住的是换衣 ReID 最本质的捷径：训练集中衣服通常属于某个固定身份，衣服几乎不被不同身份共享。模型为了最小化训练损失，会自然学到 clothing shortcut，也就是用衣服认人。现有方法即使用轮廓、解析、3D、gait 等辅助信息，也往往没有完全抛弃 RGB，所以只是稀释衣服捷径，而不是切断它。

2. 重定义动作：这是这批里最强的重定义之一。他们不再说“我们要学 clothes-invariant feature”，而是说普通训练在学 P(Y|X)，这个目标本身会吸收衣服和身份的伪相关；真正应该学的是 P(Y|do(X))，通过 causal intervention 切断 X <- C -> Y 的 backdoor path。关键词是 clothing ownership、clothing shortcut、spurious correlation、causal intervention、backdoor adjustment、confounder dictionary。

3. 机制怎么长出来：机制和重定义绑定非常紧。既然衣服 C 是 confounder，就要建 Confounder Dictionary，存每件衣服的分布和先验 P(c)。既然目标是 P(Y|do(X))，Intervention Module 就按 backdoor adjustment 对所有衣服条件求和，而不是只看当前图像衣服。既然 intervention 依赖“衣服表示必须是衣服而不是身份”，就加 Disentangle Regularization，把 identity feature 和 clothing feature 拉开。这里不是先有模块再找故事，而是每个模块都能对应 causal graph 里的变量。

4. 证据闭环：它的证据设计很完整。首先和其他 causal ReID 方法比较，并用 NCC(C -> X) 这种因果强度指标证明 clothing shortcut 被压低，而不只是 rank1 提升。其次消融 Intervention Module、Confounder Dictionary、Disentangle Regularization，说明不是普通 feature enhancement。再用随机化衣服标签、聚类伪衣服标签测试对 clothing label 质量的鲁棒性。t-SNE 显示身份特征和衣服特征分开，activation map 显示模型不再过度看衣服纹理，检索图显示换衣匹配更稳。这些证据直接围绕“切断衣服捷径”闭环。

5. reviewer 为什么买账：这篇卖的是视角，而且视角强于模块。模块本身是 dictionary、feature injection、disentanglement，不算罕见；真正 novelty 是把 CC-ReID 的失败原因从“衣服变化太大”改写成“训练目标被衣服所有权伪相关污染”。reviewer 会买，因为它解释了为什么很多 clothes-agnostic 方法仍然会失败，并且给了能验证 shortcut 是否减弱的指标和可视化。

最后看这批论文的共同套路：它们不是从“我要加一个模块”开始，而是先抓一个具体失败现象，然后把老问题换成一个更有结构的新问题，例如交互、过渡、语义传播、身份校准、噪声过滤、因果干预。机制好发的关键，是新问题一旦说清楚，模块几乎像是在实现这个新问题里的变量和关系。最值得学的是证据闭环：不要只报涨点，要有能证明“这个新问题定义确实成立”的消融、破坏性对照或可视化。
