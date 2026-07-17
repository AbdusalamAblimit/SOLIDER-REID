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
session id: 019ef835-c064-7282-bae6-d23473a21072
--------
user
你是 ReID 创新方法论拆解员 16。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(16)==
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按你的拆解框架逐篇读方法部分，优先用你指定的 `pdftotext -f 1 -l 8` 提取正文。先按项目规则补读目标和状态文件；如果当前只读沙箱拿不到上级文件，我会继续处理本目录论文。
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
/bin/zsh -lc "sed -n '1,220p' ../../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../GOAL.md: No such file or directory

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
/bin/zsh -lc "sed -n '1,260p' ../../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../CLAUDE.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' ../../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../AGENTS.md: No such file or directory

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
PDF 技能说明已经确认，和你的要求兼容：这次主要做文本抽取，不做版面改动。刚才上级路径多退了一层，`experiments/STATUS.md` 已读到；我现在补读项目根目录的目标和总规则。
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

hook: PreToolUse
hook: PreToolUse
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

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'ls -1 *.pdf' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
目标文件确认是 6 篇，不是 16 篇论文；“16”应是这轮拆解员编号。接下来我会逐篇抽取前 8 页并定位 method 或 approach 段落，重点看作者从观察到机制的叙事链，而不是复述模块结构。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|sketch|feedback|text\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:Interactive Sketch-based Person Re-Identification
2:with Text Feedback
15:Abstract—Sketch-based Person Re-identification (Sketch ReID)
17:hand-drawn sketches as queries. A significant issue is the limited
18:structural clues of sketch queries, which fail to convey highlevel semantic retrieval intentions, such as colors and genders.
19:Existing works typically combine sketches and texts for multimodal retrieval, which inevitably introduces modality interference and relies heavily on expensive tri-modal datasets. In this
21:sketch-based person retrieval framework that incorporates user
22:feedback to refine the sketch person retrieval ranking without
23:text training. A lightweight vision-to-text converting network
24:is proposed to represent sketches with equivalent pseudo-word
25:tokens, which aims to provide context assistance for interactive
26:retrieval. Then, the sketch token can be seamlessly integrated
27:with text feedback tokens within CLIP’s textual space for
28:explicit sketch-text compositionality, thus achieving feedbackguided ranking refinement. Extensive experiments underscore
31:Index Terms—Sketch-based person ReID, Interactive retrieval
37:image-to-image ReID methods [2]–[4] use photos of targets
41:readily accessible. Therefore, Sketch ReID [5] was introduced
42:to match candidate pedestrian photos using sketch queries,
43:where sketches play a role as user-provided clues to capture
45:However, as Figure 1 (a) shows, although sketch and image
46:modalities are both visual expressions, the sketch-based query
48:and gender. The bottleneck is that a single sketch query can
54:Fig. 1. (a) Sketch query is inherently inadequate in representing high-level
57:(b) Existing works conduct simple combination between sketch and text for
60:person retrieval through context assistance and feedback guidance, enabling
61:sketch and text to query in synergy within the textual domain.
69:combination may distort the optimal sketch-text composed
71:datasets, making the overall framework less flexible.
73:an interactive and flexible sketch person retrieval framework
74:that can integrate the user’s text feedback to refine the sketch
75:person retrieval ranking without text training. Figure 1 (c)
76:shows a detailed interaction process. Merely using a sketch
78:to the missing semantic intents. In our framework, users can
79:provide text feedback with high-level semantic details based
83:through Context Assistance and Feedback Guidance. Retrieval
87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
89:utilize the sketch information for retrieval, inspired by the
90:implicit grammatical composition capability of CLIP’s text
91:encoder [8], [9], our idea is to represent the sketch as a
92:pseudo-word token to provide context assistance for interactive
93:retrieval. The sketch token can be seamlessly and flexibly
94:integrated with user-guided text feedback through textual token
95:concatenation, enabling sketch and text to query in synergy
96:within the image-text-aligned embedding space.
97:The above explicit sketch-text compositionality can ingeniously avoid modality interference and also achieve optimal composed semantic mining with CLIP’s textual domain.
98:Meanwhile, a vision-to-text converting network is trained
99:to identify semantic-equivalent tokens for sketch modalities.
101:sketch and retrieval alignment between sketches and images,
102:eliminating the need for extensive textual descriptions.
104:• We introduce a novel interactive person retrieval framework for sketch ReID, flexibly integrating the user’s text
105:feedback with sketch queries for ranking refinement.
106:• InteractReID enables interactive retrieval through context assistance and feedback guidance, which not only
108:the training reliance on text modality.
110:sketch-based and interactive person retrieval scenarios
113:A. Sketch-based Person Re-identification
114:Sketch ReID was first introduced in [5] along with the
115:proposal of the PKUSketch dataset. Most works [10]–[12] are
119:modality generation. Lin et al. [13] focuses on the sketch
121:Market-Sketch-1K dataset. Zhai et al. [6] explores the complementary semantics of sketch and text, and conducts descriptive
123:Chen et al. [7] proposes a unified person ReID framework
126:However, existing sketch+text composition methods focus
129:the optimal sketch-text composed semantics. However, our
131:through modality early-fusion (i.e., representing sketch as a
133:the text modality), achieving interactive and flexible sketch
134:person retrieval with text feedback.
136:B. Interactive Image Retrieval with user feedback
138:by incorporating user feedback with their intentions, which is
140:works focused on learning with feedback in various forms,
141:such as sketches [14], spatial layouts [15], attributes [16],
142:or texts [17]–[21]. Among them, natural language is widely
143:utilized. A classic approach is to design an image-text composition module that generates composed queries similar
151:retrieval field and conduct modality late-fusion. Our InteractReID firstly introduces the idea of feedback refinement into
152:the sketch person retrieval field. Different from the above
153:innovations, we represent sketches as pseudo-word tokens to
154:achieve sketch-text synergy during interactive inference, where
155:the sketch tokens can be seamlessly integrated with userprovided text feedback tokens with the help of CLIP’s implicit
160:400 million image-text pairs [8] with a multi-class N -pair
163:paired image and text embeddings is maximized, while the
168:text encoder T conducts similar processing on input word
172:produces final textual feature wt = T(W ) ∈ Rd .
179:large-scale sketch-image-text tri-modal dataset Tri-PEDES [7],
202:Context
242:Sketch-based
260:CLIP Text
275:CLIP Text
278:Pseudo-word Token Learning for Sketch
280:CLIP Text
282:Feedback
287:Interactive Sketch Person Retrieval with Text Feedback
289:Fig. 2. Training and inference of our proposed InteractReID framework. Left: We first fine-tune CLIP on the multi-modal ReID task for downstream knowledge
291:CLIP, we aim to find the sketch’s equivalent mapping in textual space for interactive retrieval, where two contrastive losses LSCC and LALIGN are utilized
292:for sketch’s pseudo-word token generation. Right: During inference, basic sketch person retrieval is conducted through similarity calculation. To interactively
293:refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
297:As Figure 2 shows, given a batch of B sketch-imagetext pairs, we equip with a cross-modal matching loss to
301:Using text-to-image matching as an example, for each
319:where τ is a temperature parameter that controls the probability distribution peaks. Then the matching loss from text to
337:Symmetrically, the matching loss from image to text Li2t ,
338:from sketch to image Ls2i , and from image to sketch Li2s can
340:modality pairs. The total cross-modal matching loss for finetuning CLIP to align sketch-image-text can be formulated as:
349:C. Pseudo-word Token Learning for Sketch
350:To achieve interactive sketch person retrieval with user’s
351:text feedback, based on the vision-text joint embedding space
353:accurately capture the sketch semantics for context assistance,
354:thus achieving explicit sketch-text compositionality through
355:flexible textual token concatenation.
356:Concretely, we train a lightweight vision-to-text converting
358:sketch retrieval datasets to achieve the equivalent mapping.
359:Given an input sketch query S, we first obtain its visual
366:pass it through CLIP’s text encoder to obtain the sketch’s
368:Revisiting our training objective, we aim to bring the sketch
369:features in the textual space st as close as possible to those
411:P ERFORMANCE COMPARISONS WITH S OTA METHODS ON
413:Methods
417:Sketch-based
422:Sketch+Text
423:Sketch-based
438:P ERFORMANCE COMPARISONS WITH S OTA METHODS ON PKUS KETCH .
450:Meanwhile, the learned token embedding st for sketch
461:We update the parameters of the proposed vision-to-text
463:downstream sketch person retrieval datasets Market-Sketch1K [13] and PKUSketch [5] without paired textual descriptions, while keeping the CLIP text encoder T frozen to fully
464:utilize its grammatical composition capability for sketch and
465:text’s token concatenation.
467:The retrieval practicality of our proposed InteractReID
468:framework can be evaluated from not only sketch-based person
470:Firstly, the sketch query’s feature sv is extracted and compared with gallery features ivg for similarity calculation, thus
473:In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
474:Specifically, we introduce context assistance through the welltrained vision-to-text converting network Φv2w for sketch’s
476:Then the feedback guidance is achieved through textual token
478:query feature can be generated through the CLIP text encoder:
483:A. Experimental Settings
488:datasets, where sketches are generated according to RGB
490:identities, 115233 RGB images, and sketches with 175972 text
493:Methods
499:Sketch-based
504:Sketch+Text
506:Sketch-based
536:PKUSketch [5] is a basic dataset for sketch ReID, which
538:images and a sketch. 150 persons are randomly selected for
540:We conduct 10 experiments on randomly partitioned datasets
542:Market-Sketch-1K [13] is a new sketch ReID dataset with
544:4,763 sketches of 996 identities and 32,668 photos of 1,501
545:identities in total. Sketches are divided into 6 groups according
554:maximum length of the textual token sequence is 77. τ is set
557:trained on Market-Sketch-1K or PKUSketch with 20 epochs
559:for V and 1e-4 for Φv2w training. During inference, textual
561:simulate users’ text feedback. Note that our InteractReID does
562:not involve any text annotations while training Φv2w .
563:B. Comparison with State-of-the-art Methods
564:We compare the proposed InteractReID with existing reproducible baselines on Market-Sketch-1K and PKUSketch
565:datasets, focusing on two realistic retrieval scenarios using different queries: sketch-based and interactive retrieval. As shown
568:Concretely, when first using the sketch-based query for person retrieval, it can be observed that our paradigm outperforms
576:Fig. 3. A typical interactive person retrieval process of our InteractReID. Top-10 initial and refined ranking results on the Market-Sketch-1K dataset are
577:compared through sketch-based and interactive person retrieval with text feedback. Gallery samples in green boxes match the query, while in red boxes
596:Sketch-based
613:person retrieval through context assistance and text feedback guidance. Compared to existing sketch+text composition
614:methods which adopt modality late-fusion for retrieval (i.e.,
616:early-fusion that achieves explicit sketch-text compositionality
617:through textual token concatenation. Thanks to the interaction
618:capability achieved by integrating sketch’s pseudo-word tokens
619:with user-provided text feedback in CLIP’s language token
621:methods with a R@1 of 92.00% on PKUSketch and 49.24%
622:on Market-Sketch-1K for retrieval ranking refinement.
624:We comprehensively evaluate the effectiveness of the proposed InteractReID on Market-Sketch-1K from the following
630:to No.5 in both the sketch-based and interactive retrieval
636:A BLATIONS ON THE DESIGN OF V ISION - TO - TEXT C ONVERTING
654:Sketch-based
677:Tri-ICFG-PEDES, and Tri-RSTPReid. Experimental results
681:2) Is Vision-to-text Converting Network effective? The
683:mapping from visual to textual space. To verify the above
686:the text encoder without network converting (No.1), mapping
690:necessary to map visual features into the textual embedding
697:We also experimented with other variants, such as varying
703:3) Does pseudo-word tokens really capture sketch semantics? To analyze the effectiveness of the pseudo-word token
704:in capturing sketch information, we evaluate the well-trained
705:model using relative validation datasets. Concretely, we utilize sketches’ pseudo-word tokens as queries to retrieve in
706:the gallery which solely consists of input sketches. R@1 of
707:95.64% and R@5 of 99.76% on Market-Sketch-1K demonstrate that the semantic effectiveness of pseudo-word tokens.
709:A typical interactive person retrieval process is demonstrated in Figure 3 to verify the practicality of our InteractReID. Firstly, basic sketch-based person retrieval is conducted, which tends to match persons only with similar
711:framework can support users to provide discriminative semantic feedback for interactive and flexible retrieval. Concretely,
712:when mapping the sketch into its equivalent pseudo-word
713:token and combining it with feedback tokens in the textual
716:In this paper, we propose a novel interactive sketch-based
717:person retrieval framework InteractReID, which can flexibly
718:integrate sketch queries with user’s text feedback to achieve
720:knowledge transfer. Based on this, a vision-to-text converting
721:network is trained to represent sketch as a pseudo-word token
722:so that the CLIP text encoder can flexibly compose the
723:sketch token and text feedback tokens for interactive retrieval.
724:Extensive quantitative and qualitative results on both sketchbased and interactive person retrieval scenarios demonstrate
726:to validate the effectiveness of the proposed InteractReID
727:framework. In the future, we aim to explore LLM-based
729:based on user’s text feedback.
744:model for image re-identification without concrete text labels,” in Proc.
752:“Cross-domain adversarial feature learning for sketch reidentification,” in Proc. of ACM MM, 2018, pp. 609–617.
760:[9] Alberto Baldrati et al., “Zero-shot composed image retrieval with textual
763:for sketch re-identification,” IEEE TIFS, vol. 17, pp. 3421–3432, 2022.
765:semantic consistent feature construction for sketch re-identification,” in
768:sketch re-identification,” in Proc. of AAAI, 2024, pp. 3747–3755.
770:Satoh, “Beyond domain gap: Exploiting subjectivity in sketch-based
773:A sketch and text duet for fine-grained image retrieval,” in Proc. of
775:[15] Arko Barman and Shishir K Shah, “A graph-based approach for making
782:[18] Yanbei Chen et al., “Image search with text feedback by visiolinguistic
784:[19] Seungmin Lee, Dongwan Kim, and Bohyung Han, “Cosmo: Contentstyle modulation for image retrieval with text feedback,” in Proc. of
787:transformer for fashion retrieval with feedback,” in Proc. of CVPR,
789:[21] Dafeng Li and Yingying Zhu, “Visual-linguistic alignment and composition for image retrieval with text feedback,” in Proc. of ICME. IEEE,
797:aligning for text-to-image person retrieval,” in Proc. of CVPR, 2023,
799:[26] Xinyi Wu, Wentao Ma, et al., “Text-based occluded person reidentification via multi-granularity contrastive consistency learning,” in
812:sketch re-identification,” Neurocomputing, vol. 403, pp. 294–303, 2020.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|mask|pose|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:Instant pose extraction based on mask transformer for occluded person
21:Instant pose extraction
22:Mask aware module
25:ABSTRACT
26:Re-Identification (Re-ID) of obscured pedestrians is a daunting task, primarily due to the frequent occlusion
28:we propose a novel approach named Instant Pose Extraction based on Mask Transformer (MTIPE), tailored
29:specifically for occluded person Re-ID. MTIPE consists of several new modules: a Mask Aware Module (MAM)
31:Module (MACM) to enrich the feature representation; a Pose Aggregation Module (PAM) to separate useful
32:human information from the occlusion noise; a Feature Matching Module (FMM) in matching non-occluded
33:parts; introduction of learnable local prototypes in the defined local prototype-based transformer decoder; a
35:local contextual information; and Pose Key-points Loss to better match non-occluded body parts. Through
36:comprehensive experimental evaluations and comparisons, MTIPE demonstrates encouraging performance
38:of current state-of-the-art methods in various aspects, highlighting its potential advantages and promising
41:1. Introduction
42:The purpose of person Re-Identification (Re-ID) is to match specific people in different cameras under different scenes, lighting, and
46:application scenarios, the occlusion of people by different obstacles
52:random occlusions of body parts by different obstacles will change the
55:Some existing approaches use an auxiliary model with artificial key
57:points and parsed information to address the occlusion challenge. Wang
58:et al. [4] proposed a graph convolution network (GCN) to generate
59:part features using pose guided key-points and apply graph matching to
63:proposed the Reasoning and Tuning Graph Attention Network, which
66:proposed a part and pose guided part feature alignment to select useful
71:been made using CNN-related methods to solve such problems, CNNs
94:occlusion challenges in person Re-ID, which integrates global
95:feature maps with specific pose information to highlight visible
96:body parts and effectively captures pose features, ensuring robust
97:alignment of occluded human body images by generating a prototype mask. This alignment enhances the accuracy of matching
98:even under significant occlusion scenarios.
99:• In order to better retrieve and match the masked person, MAM
101:average pooling to generate masks that capture local information,
108:• To demonstrate the effectiveness of our method, we perform experiments on occluded and holistic Re-ID datasets. Experimental
109:results also verify our proposed method reaches a new state of the
113:Fig. 1. An illustration of the application of using MTIPE method in occluded person
128:Sun et al. [10] uniformly partition the feature map and learn partiallevel features by multiple classifiers. However, these methods are for
135:To solve the above problems, we propose Instant Pose Extraction based on Mask Transformer (MTIPE) for Occluded Person Re-ID,
136:as shown in Fig. 1. During the training process, less masked images are assigned as high-quality images, and their loss scale will
137:be lower than that of lower-quality images with more masking, thus
139:consistent as possible. Specifically, the proposed MTIPE consists of
140:a Mask based Transformer Encoder,including a Mask Aware Module
143:the transformer encoder–decoder architecture, a Pose Feature Aggregation Module (PFAM) and a Feature Matching Module (FMM). One
146:mask.
147:Firstly, using the pre-trained mask-based transformer block [3,7],
149:network and generates a mask by global average pooling. In order for
152:integrating global feature map information with pose information from
156:prototype corresponds to the most recognizable part of the occluded human image, and generates a prototype mask focused on local features of
157:the human body. Most existing attention-based approaches [8,9] tend
160:varying occlusions, our approach captures diverse person appearances,
161:offering Re-ID insights. Therefore we propose a new attention module
163:evaluate the effectiveness of the proposed MTIPE, we conduct a series
164:of experiments on both occluded and holistic Re-ID benchmarks. The
171:misalignment. Gao et al. [11] proposed a Pose-guided Visible Part
172:Matching (PVPM) method that jointly learns discriminating features
173:with pose guided attention and self-mines partial visibility in an endto-end framework. Miao et al. [6] introduced a new method called
174:Pose Guided Feature Alignment (PGFA) that uses pose landmarks to
175:separate useful information from occlusion noise. Zhao et al. [12]
176:proposed a novel Content-Adaptive Auto-Occlusion Network (CAAO),
177:which can dynamically select appropriate occlusion regions in an image
179:Tan et al. [13] proposed a more intuitive and effective data-driven
183:designed an Occlusion Generator (OGor) to automatically generate artificially occluded images from generic surveillance images, minimizing
184:the negative impact of occlusion on feature learning and developed a
185:novel multi-granularity contrastive consistency alignment framework,
193:Fig. 2. The framework of the proposed MTIPE model which consists of a mask-based transformer encoder, a prototype-based transformer decoder and a feature matching module.
194:The mask-based transformer encoder includes a Multi-headed Attention Constraint Module (MACM) and a Mask Aware Module (MAM). For masked human images, the MACM
196:by integrating global feature map information and posture information from the target person. For different images MAM generates a dynamic mask that selects subspace from
199:introduced an Attention Disturbance Mask (ADM) module that generates interference noise similar to real occluders, serving as a more
200:complex form of occlusion. They developed a Dual-Path Constraint
201:(DPC) module, which obtains better supervision information from holistic images through dual-path interaction. Although the above methods
202:can solve the occlusion problem to some extent, most of them rely
203:heavily on off-the-shelf artificial parsing models or pose estimators.
204:In summary, holistic person Re-ID methods have achieved notable
206:extract discriminative representations. However, these methods often
208:complete body information. On the other hand, recent approaches to
209:occluded Re-ID, including pose estimation and manual parsing, provide
211:can be limiting. Our proposed method stands out by employing a transformer encoder–decoder architecture to effectively extract and align
212:human pose information, thus preserving global context and improving
213:performance in both holistic and occluded scenarios. This novel approach addresses the limitations of existing methods and demonstrates
238:are used to learn the feature representation. Our proposed MACM is
240:the class-token representations will be used to train an overall prototype for each class. For the masked images, we send the representation
242:a dynamic prototype mask for each input image.
252:3. Proposed method
253:In this section, we present the proposed Instant Pose Extraction
254:based on Mask Transformer (MTIPE) in detail. As shown in Fig. 2, our
255:MTIPE consists of four main components, the Mask based Transformer
256:Encoder, the Local Prototype based Transformer Decoder, the Pose
265:3.1. Mask based transformer encoder
287:3.1.1. Mask Aware Module (MAM)
288:We propose MAM with the aim of learning a dynamic mask to cut
292:prototype mask is generated after global average pooling, using the
299:Fig. 3. The structure of the Mask Aware Module (MAM). The MAM is a convolution-based dynamic mask generator that considers the neighbor nodes of each patch, evaluates
300:the weights of each channel based on the correlation of local information, and thus cuts the overall prototype and selects valid subspace for matching by generating masks.
308:useful information is concentrated in the mask image, the final mask is
345:but also facilitates the matching between the prototype mask and the
356:it difficult for MAM to generate a valid prototype mask. Therefore,
369:3.2. Pose feature aggregation module
375:uses a pose estimator to detect human key-points from human images.
376:Given an input image 𝐼, MTIPE extracts 𝐾 landmarks using the pose
378:confidence score are obtained by the pose estimator. To classify the
408:Fig. 4. As the network structure deepens, the feature representation of each patch becomes smooth and the prototype mask generated by MAM is difficult to function in the case
409:of highly similar inputs, so we propose the Multi-Head Attention Constraint Module (MACM) to enrich the feature representation by aggregating different feature representations
424:global feature information and the pose information from the person in
425:the image, the pose feature aggregation module multiplies the heat map
426:𝐻 with the feature 𝑓𝑔𝑒 and outputs the pose feature map 𝑃𝑖 . Since each
428:human body, the purpose of the pose feature aggregation module map
431:pose feature aggregation module has encoded information from different parts of the human body, we still want to find the part of the 𝑓𝑔𝑒 that
459:different confidence levels to obtain 𝑓𝑝𝑘ℎ and 𝑓𝑝𝑘𝑙 . We then propose the
484:can obtain M local masks.
537:According to the standard architecture of transformer, we propose
550:Evaluation Protocol. To verify fair comparison with other methods,
553:the evaluation settings provided by existing occluded methods.
572:We achieve our MTIPE with PyTorch and conduct all experiments on a
578:𝑉 , respectively. We propose to use
599:Results on Occluded Re-ID Datasets. Table 1 shows the performance of our model and previous methods on the occluded datasets.
600:Four methods are compared, namely, the Re-ID method based on
601:manual segmentation, the holistic Re-ID methods based on key-points
602:information, the partial methods and occluded Re-ID methods. The
603:Rank-1/mAP of this method achieves 66.5% Rank-1 accuracy and
604:57.8%mAP on the Occluded-Duke, establishing a new SOTA performance. Compared to the manual segmentation PCB [10] method, our
607:masks to suppress the noisy information from the occluded region while
611:this advantage and outperforms other Transformer-based methods. In
612:addition, our MTIPE also outperforms other mask-based [23,24] learning strategies, which fully demonstrates the effectiveness of our fusion
613:of two learning methods, transformer encoder–decoder and attention
625:4. Experiments
626:In this section, to validate the effectiveness of our proposed method,
627:we conduct experiments on four Re-ID datasets, which include occluded
629:effectiveness of our proposed module through ablation experiments and
633:our proposed method on the holistic Re-ID datasets, we conducted
634:experiments on three holistic Re-ID datasets, including Market-1501,
635:DukeMTMC-ReID, and MSMT17. For better comparison, four methods
636:were chosen, including (1) partial feature-based methods [10,29]. (2)
637:Global feature-based methods [30,31,33]. (3) Additional cue-based
638:methods [4,6,34]. (4) Transformer-based approachs [17,26]. Table 2
642:performance. The proposed MTIPE compares with the transformerbased method (Pirt [27]), our method surpasses it by +2.7% mAP and
655:Performance comparison with state-of-the-art methods on Occluded-Duke.
656:Methods
718:Methods
732:Pose-guided re-ID [35]
880:Methods
884:Effectiveness of the MTE. As shown in Index 2, the +5.2% mAP improves the performance when only the mask-based transformer encoder
889:prototype while generating dynamic masks. Therefore, the encoder is
922:that the prototype mask generated by the prototype can have a better
924:the effect of more noisy backgrounds such as occlusion. From the
927:simultaneous use of encoder–decoder can be better for pose information
930:+1.0% Rank1 accuracy on DukeMTMC-ReID. We also experimented
931:our MTIPE model on the MSMT17, comparing several methods, including [30,33,36], 𝑒𝑡𝑐. The results are shown in Table 3, and our
933:method achieves 80.4% Rank-1 accuracy and 61.1% mAP on MSMT17.
934:It can be seen that our proposed method is equally robust, both for the
940:proposed attention module can extract more distinguishing features,
943:the introduction of the attention module in the encoder module. The
949:to analyze the effectiveness of each component of our proposed MTIPE,
950:including the Mask-based Transformer Encoder (MTE), the Pooling
961:Chen: Writing – review & editing, Writing – original draft, Supervision, Resources, Project administration, Methodology, Funding acquisition. Li-Li Huang: Writing – review & editing, Supervision, Methodology, Funding acquisition. Bin Luo: Supervision, Resources, Funding
965:Fig. 7. Visualization on features generated by our proposed MTIPE module.
998:[6] J. Miao, Y. Wu, P. Liu, Y. Ding, Y. Yang, Pose-guided feature alignment for
1006:them all: Occlusion-aware attention network for occluded person re-id, in:
1015:[11] S. Gao, J. Wang, H. Lu, Z. Liu, Pose-guided visible part matching for occluded
1018:[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
1037:We visualize the features generated by the proposed MTIPE. As
1038:shown in Fig. 7, the proposed MACM and MAM can locate the occluded body parts well, which proves the effectiveness of our proposed
1039:method.
1041:5. Conclusion
1042:In this paper, we propose an Instant Pose Extract Based on Mask
1043:Transformer (MTIPE) method for Occluded Person Re-Identification,
1044:which includes a mask based transformer encoder (including MAM
1046:PAM), a pose feature aggregation module (PFAM), a feature matching
1048:in the encoder generates dynamic masks to select appropriate subspace
1049:to match the occluded human images, and further proposes MACM
1052:map information and the pose information from the target person. In
1054:distinguishable body parts and propose the FMM module to enhance
1055:the features of the encoder. Finally, an extensive experimental evaluation of occlusion and whole-person Re-ID benchmarks shows that
1056:MTIPE outperforms existing methods.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Internal-External Context Interaction Network for Person Re-Identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|context|internal|external\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Internal–External Context Interaction Network for
10:Abstract—Capturing discriminative cues with attention mechanisms is crucial for solving the high inter-class similarity problem
12:own contextual information within a single sample using selfaffinity between elements, and some works have demonstrated its
16:this paper, we propose an internal-external context interaction
18:of inter-sample latent context information and intra-sample
19:local context information to enhance the feature representation
24:an internal-external context interaction network (IEINet) for
27:a region-diverse loss to constrain the network. Many experiments
29:approach attains state-of-the-art performance on several largescale person Re-ID datasets.
30:Index Terms—Person re-identification, internal-external context interaction, discriminative features, region-diverse loss.
65:to address the challenge. To extract subtle discriminative information, some methods [1], [2], [3], [4] have been proposed to
70:stream of approaches [5], [6], [7], [8] aims to accurately
87:researchers have begun to utilize element-specific context for
90:robust character feature representations by modeling contextual relationships between patches, while [22] jointly mines
93:excels in global context modeling, its inherent single-sample
106:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
110:Fig. 1. (a) presents the schematic comparison between frameworks for the proposed IEINet and typical related methods. Splitting-based methods [1], [2]
111:learn multiple part features containing subtle visual cues from strips, while attention and splitting-based methods [13], [26] introduce attention modules to
113:Multidimensional attention-based methods [15], [21], [22], [25] with the single feature space make learning between different dimensions of attention interfere
126:take full advantage of the local contextual information between
128:In this paper, we propose an internal-external context interaction (IEI) attention mechanism for person Re-ID, which
131:context (external information) of different samples with the
132:internal local context information of a sample. Specifically,
134:robust self-context information of the sample, while storing
135:potential semantic features between different pedestrian samples through an external learnable memory, which guides the
143:mechanism, we further design an internal-external context
155:Compared with existing methods, IEINet is innovative in
156:two aspects: first, compared with multidimensional attentionbased methods [15], [21], [22], [23], [24], [25] (e.g., hybrid
161:approaches [13], [26], [27], [28] introduce attention before
167:using the interactions between the latent contextual
169:contextual information of each element within a sample.
187:3) Extensive experiments are carried out on publicly available person datasets (Market1501 [29], CUHK03 [30],
188:MSMT17 [31] and PersonX [32] to confirm the effectiveness of our proposed IEI mechanism and IEINet. The
189:experimental results demonstrate that our method attains
203:response to this challenge, some methods [1], [2], [3], [4],
218:Other approaches [5], [6], [7], [8], [20], [36], [37] use
231:data distribution. Compared to the above methods, our IEINet
240:the channel weights using the global context of channels. Reference [44] restricts channel interactions to a localized range,
263:[19] and [20] use SA to directly model the global context
271:only utilise own contextual information within a single sample
275:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
279:Fig. 2. Overall architecture of IEINet. IEI-C is the channel internal-external context interaction module, and IEI-S is the spatial internal-external context
286:them, our proposed IEI attention mechanism is constructed
287:based on both inter-sample context and intra-sample context,
288:which utilises inter-sample context to directly capture subtle differences between persons and additional visual cues,
290:the above approaches compute pairwise affinity between all
293:robust intra-sample own context information and restrict the
294:interaction of intra-sample and inter-sample contexts to a local
298:multidimensional attention-based approaches [15], [21], [22],
303:with each other. Attention and splitting-based approaches [13],
309:semantically continuous part information. Compared with multidimensional attention-based methods as well as attention
310:and splitting-based methods, on the one hand, IEINet learns
318:There are some methods [48], [49] in the field of person
326:relationships, our method directly learns potential relationships
337:covers three parts. Firstly, we present the overall framework
392:B. Spatial Internal-External Context Interaction Module
394:contextual information of different samples and the spatial
395:local context in a sample to infer the important positions
401:a spatially robust contextual representation of a single sample,
405:of a location. Q s reflects the robust activation contextual
411:spatial contextual information between samples, which is a
412:learnable external memory that is independent of the individual samples, and it learns the spatial contextual information
415:by interacting the inter-sample context with the local selfinformation within the sample. The local context information
420:slide on Q s to obtain a local context query matrix Q s0 with
424:i ∈ {1, 2, . . . , H W} of Q s0 is a local context query vector for
429:Next, we project the local context Qis0 at the ith location
430:onto the inter-sample spatial context K s , and obtain the k × k
432:projected onto K s to realize the internal and external context
443:context information under the guidance of the inter-sample
444:context. This method emphasizes the important identity-related
475:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
479:local context query matrix and the inter-sample context key
497:C. Channel Internal-External Context Interaction Module
500:interaction of inter-sample contextual information with intrasample channel local context. The structure of this module is
503:contextual representation of a sample in terms of channels,
506:reflects the activation contextual information of the feature
509:we introduce a key matrix K c ∈ Rk×k that stores intersample channel context information and let it interact with
510:the intra-sample channel local context information to generate
511:aggregated weights for each channel. The local contextual
515:of 1 to slide C times on Qc to obtain a local context query
517:Qc0j , j ∈ {1, 2, . . . , C} is the local context query vector for
522:Next, we establish the internal and external contextual
527:channel, which is obtained by projecting the local context Qc0j
528:onto the inter-sample channel context K c and implementing
561:employed in person Re-ID approaches [1], [12], [57]. In addition, to enable IEINet to extract more different discriminative
671:We evaluate the proposed method on four large-scale person
687:while the CUHK03-Detected dataset utilizes DPM for detecting bounding boxes of person images, similar to the approach
709:5) Evaluation Metrics: In all experiments, we use the mean
713:Following previous person Re-ID methods [1], [19], [57],
730:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
750:above are employed in our experiments conducted on the four
752:C. Comparison With the State-of-the-Art Methods
753:To demonstrate the effectiveness of our proposed IEINet,
754:we compare it with current state-of-the-art methods on the
759:comparison between our method and the current state-ofthe-art methods on the Market1501 dataset. We can observe
760:that our method achieves optimal performance in mAP. First,
774:comparison of our IEINet and the current state-of-the-art methods on CUHK03-Labeled and CUHK03-Detected datasets.
775:The results in the table show that our method outperforms the
790:In contrast, our method achieves superior performance to them
793:comparison of our IEINet with the state-of-the-art methods

codex
前三篇的章节位置已经定位：第一篇的方法从第 3 节开始，第二篇的核心在 pseudo-word token 和交互检索流程，第三篇的核心在内部外部上下文交互模块。现在我读这些段落的完整文本，避免只凭摘要和图注判断。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf' - | nl -ba | sed -n '179,483p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   179	large-scale sketch-image-text tri-modal dataset Tri-PEDES [7],
   180	
   181	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:07 UTC from IEEE Xplore. Restrictions apply.
   182	
   183	CLIP Fine-tuning
   184	
   185	Training
   186	
   187	Inference
   188	CLIP Image
   189	Encoder
   190	
   191	CLIP Image
   192	Patch
   193	Encoder
   194	Embedding
   195	
   196	CLIP Image
   197	Patch
   198	Embedding Encoder
   199	
   200	Gallery images
   201	
   202	Context
   203	Assistance
   204	
   205	sharing
   206	
   207	Equivalent
   208	Mapping
   209	
   210	CLIP Image
   211	Encoder
   212	
   213	Patch
   214	Embedding
   215	
   216	Prompt
   217	
   218	Converting
   219	Network
   220	
   221	User
   222	
   223	a photo of
   224	
   225	(2)
   226	Interactive
   227	Retrieval
   228	
   229	“ He is wearing a
   230	black and white
   231	hooded sweatshirt,
   232	jeans, and
   233	Tokenize
   234	primarily white
   235	sneakers with Embedding
   236	black. He is also
   237	wearing a wrist
   238	watch on his left
   239	wrist.”
   240	
   241	(1)
   242	Sketch-based
   243	Retrieval
   244	
   245	Converting
   246	Network
   247	
   248	Initial
   249	Ranking
   250	R@1
   251	
   252	R@5
   253	
   254	Refined
   255	Ranking
   256	
   257	a photo of
   258	“Similar striped
   259	
   260	CLIP Text
   261	Encoder
   262	
   263	Task-oriented Knowledge Adaption
   264	
   265	Tokenized
   266	embeddings
   267	
   268	T-Shirts. But the
   269	target is a man in
   270	black brown short
   271	pants, and wearing
   272	a red shoes not red
   273	or blue pants”
   274	
   275	CLIP Text
   276	Encoder
   277	
   278	Pseudo-word Token Learning for Sketch
   279	
   280	CLIP Text
   281	Encoder
   282	Feedback
   283	Guidance
   284	
   285	Gallery images
   286	
   287	Interactive Sketch Person Retrieval with Text Feedback
   288	
   289	Fig. 2. Training and inference of our proposed InteractReID framework. Left: We first fine-tune CLIP on the multi-modal ReID task for downstream knowledge
   290	adaptation, where the cross-modal matching loss LCM = Lt2i + Li2t + Ls2i + Li2s is used for modality alignment. Based on the knowledge-adapted
   291	CLIP, we aim to find the sketch’s equivalent mapping in textual space for interactive retrieval, where two contrastive losses LSCC and LALIGN are utilized
   292	for sketch’s pseudo-word token generation. Right: During inference, basic sketch person retrieval is conducted through similarity calculation. To interactively
   293	refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
   294	
   295	aiming to further utilize CLIP’s modality-aligned knowledge
   296	in the downstream multi-modal ReID tasks.
   297	As Figure 2 shows, given a batch of B sketch-imagetext pairs, we equip with a cross-modal matching loss to
   298	pull positive pairs together and push negative pairs apart.
   299	Specifically, different modalities are associated by embedding
   300	their cosine similarity distributions into the KL divergence.
   301	Using text-to-image matching as an example, for each
   302	pair wit , ivj , we model its matching probability through the
   303	feature’s cosine similarity, which can be calculated by:
   304	 
   305	exp sim wit , ivj /τ
   306	pi,j = PB
   307	,
   308	(1)
   309	t v
   310	k=1 exp (sim (wi , ik ) /τ )
   311	
   312	(wit )⊤ ivj
   313	,
   314	sim wit , ivj =
   315	∥wit ∥ ivj
   316	
   317	(2)
   318	
   319	where τ is a temperature parameter that controls the probability distribution peaks. Then the matching loss from text to
   320	image can be computed by:
   321	B
   322	
   323	B
   324	
   325	1 XX
   326	Lt2i = DKL (q̂i,j ∥pi,j ) = −
   327	q̂i,j log pi,j ,
   328	|B| i=1 j=1
   329	
   330	(3)
   331	
   332	where q̂i,j is the true matching probability, which is the
   333	PB
   334	normalized ground-truth labels qi,j / k=1 qi,k . (qi,j = 1
   335	means a matched pair with the same identity, and qi,j = 0
   336	indicates the unmatched pair).
   337	Symmetrically, the matching loss from image to text Li2t ,
   338	from sketch to image Ls2i , and from image to sketch Li2s can
   339	also be calculated by modeling matching probability between
   340	modality pairs. The total cross-modal matching loss for finetuning CLIP to align sketch-image-text can be formulated as:
   341	min
   342	
   343	{V,T}
   344	
   345	LCM = Lt2i + Li2t + Ls2i + Li2s .
   346	
   347	(4)
   348	
   349	C. Pseudo-word Token Learning for Sketch
   350	To achieve interactive sketch person retrieval with user’s
   351	text feedback, based on the vision-text joint embedding space
   352	provided by CLIP, we aim to find a pseudo-word token that can
   353	accurately capture the sketch semantics for context assistance,
   354	thus achieving explicit sketch-text compositionality through
   355	flexible textual token concatenation.
   356	Concretely, we train a lightweight vision-to-text converting
   357	network Φv2w with 1-layer MLP on downstream realistic
   358	sketch retrieval datasets to achieve the equivalent mapping.
   359	Given an input sketch query S, we first obtain its visual
   360	embedding through CLIP’s vision encoder: sv = V(S) ∈
   361	Rd , which will be sent to the converting network Φv2w
   362	to generate its equivalent pseudo-word token embedding as
   363	sw = Φv2w (sv ) ∈ Rd . To maintain the semantic integrity
   364	and compositionality, inspired by the popular prompt learning paradigm [27], we attach sw at the end of a universal
   365	prompt sentence pw (e.g., “a photo of”, “an image of”) and
   366	pass it through CLIP’s text encoder to obtain the sketch’s
   367	final language-equivalent feature st = T({pw ; sw }) ∈ Rd .
   368	Revisiting our training objective, we aim to bring the sketch
   369	features in the textual space st as close as possible to those
   370	in the visual space sv . To achieve this, we propose a selfcycle contrastive loss to impose training constraints on the
   371	converting network Φv2w , i.e.,
   372	min
   373	
   374	{Φ,V}
   375	
   376	LSCC = Lcst (sv , st ) + Lcst (st , sv ),
   377	
   378	
   379	exp (svi )⊤ sti /τ
   380	1 X
   381	,
   382	Lcst (s , s ) = −
   383	log P
   384	v ⊤ t
   385	|B|
   386	j∈B exp (si ) sj /τ
   387	i∈B
   388	
   389	exp (sti )⊤ svi /τ
   390	1 X
   391	t v
   392	.
   393	Lcst (s , s ) = −
   394	log P
   395	t ⊤ v
   396	|B|
   397	j∈B exp (si ) sj /τ
   398	i∈B
   399	v
   400	
   401	t
   402	
   403	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:07 UTC from IEEE Xplore. Restrictions apply.
   404	
   405	(5)
   406	(6)
   407	
   408	(7)
   409	
   410	TABLE I
   411	P ERFORMANCE COMPARISONS WITH S OTA METHODS ON
   412	M ARKET-S KETCH -1K. † MEANS RE - IMPLEMENTATION BY US .
   413	Methods
   414	Retrieval Type
   415	DCLNet [28]
   416	DSCNet [29]
   417	Sketch-based
   418	DEEN [30]
   419	BDG [13]
   420	UniReID† [7]
   421	UniReID† [7]
   422	Sketch+Text
   423	Sketch-based
   424	InteractReID
   425	Interactive
   426	
   427	R@1
   428	12.24
   429	13.84
   430	12.11
   431	18.10
   432	8.65
   433	27.01
   434	37.30
   435	50.42
   436	
   437	TABLE II
   438	P ERFORMANCE COMPARISONS WITH S OTA METHODS ON PKUS KETCH .
   439	
   440	R@5 R@10 mAP mINP
   441	29.20 39.58 13.45
   442	30.55 40.34 14.73
   443	25.44 30.94 12.62
   444	38.95 50.75 19.61
   445	19.45 26.88 11.16 7.52
   446	53.97 67.68 31.48 23.33
   447	61.81 73.80 39.65 30.49
   448	79.70 88.61 54.98 46.88
   449	
   450	Meanwhile, the learned token embedding st for sketch
   451	should also be aligned with its paired images iv = V(I) ∈ Rd
   452	through the following feature alignment loss:
   453	min
   454	
   455	{Φ,V}
   456	
   457	LALIGN = Lcst (st , iv ) + Lcst (iv , st ).
   458	
   459	(8)
   460	
   461	We update the parameters of the proposed vision-to-text
   462	converting network Φv2w and CLIP image encoder V on
   463	downstream sketch person retrieval datasets Market-Sketch1K [13] and PKUSketch [5] without paired textual descriptions, while keeping the CLIP text encoder T frozen to fully
   464	utilize its grammatical composition capability for sketch and
   465	text’s token concatenation.
   466	D. Inference
   467	The retrieval practicality of our proposed InteractReID
   468	framework can be evaluated from not only sketch-based person
   469	retrieval but also interactive person retrieval.
   470	Firstly, the sketch query’s feature sv is extracted and compared with gallery features ivg for similarity calculation, thus
   471	generating the initial retrieval ranking results, which are then
   472	returned to users for retrieval accuracy evaluation.
   473	In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
   474	Specifically, we introduce context assistance through the welltrained vision-to-text converting network Φv2w for sketch’s
   475	equivalent pseudo-word token generation: sw = Φv2w (sv ).
   476	Then the feedback guidance is achieved through textual token
   477	concatenation among pw , sw , and tw . Therefore, the composed
   478	query feature can be generated through the CLIP text encoder:
   479	stq = T ({pw ; sw ; tw }) ∈ Rd . Lastly, we conduct refined
   480	retrieval ranking by comparing the cosine similarity between
   481	stq and gallery visual features ivg .
   482	IV. E XPERIMENTS
   483	A. Experimental Settings

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '252,570p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   252	3. Proposed method
   253	In this section, we present the proposed Instant Pose Extraction
   254	based on Mask Transformer (MTIPE) in detail. As shown in Fig. 2, our
   255	MTIPE consists of four main components, the Mask based Transformer
   256	Encoder, the Local Prototype based Transformer Decoder, the Pose
   257	Feature Aggregation Module and a Feature Matching Module.
   258	
   259	1 ∑
   260	 (𝑃 (𝑓𝑔𝑖𝑙 ))
   261	𝐺 𝑖 𝐼𝐷
   262	𝐺
   263	
   264	𝑒𝑛 = 𝐼 𝐷 (𝑃 (𝑓𝑔𝑒 )) +
   265	3.1. Mask based transformer encoder
   266	
   267	1 ∑
   268	 (𝑓 𝑖 ) + 𝛼𝑜𝑟𝑡ℎ𝑜 ,
   269	𝐺 𝑖 𝑇 𝑔𝑙
   270	𝐺
   271	
   272	+𝑇 (𝑓𝑔 𝑒 ) +
   273	
   274	The original features are extracted using a pure transformer-based
   275	model [7] to segment the image into non-overlapping blocks, which
   276	may lose the local neighborhood information of the blocks. Therefore,
   277	MTIPE uses a shifted window [16] to obtain patches with overlapping
   278	pixels. Give an image 𝐼 ∈ R𝐻×𝑊 ×𝐶 , where 𝐻 , 𝑊 , 𝐶 denote its height,
   279	width, and number of channels, respectively. We use a sliding window
   280	to segment the image into 𝐾 fixed-size (ℎ × 𝑤) patches, denote the step
   281	size as 𝑆, and the patch size of each image as 𝑃 . Then the number of
   282	patches 𝐾 can be expressed as:
   283	
   284	(3)
   285	
   286	where 𝑜𝑟𝑡ℎ𝑜 will be illustrated in Eq. (8).
   287	3.1.1. Mask Aware Module (MAM)
   288	We propose MAM with the aim of learning a dynamic mask to cut
   289	the entire prototype and selecting appropriate subspace for matching,
   290	which is one of the key ideas of MTIPE. As shown in Fig. 3, MAM
   291	uses the reshaped image representation as 𝑓𝑟 ∈ R𝐻×𝑊 ×𝐶 , and a
   292	prototype mask is generated after global average pooling, using the
   293	3
   294	
   295	Pattern Recognition 159 (2025) 111082
   296	
   297	T.-T. Yuan et al.
   298	
   299	Fig. 3. The structure of the Mask Aware Module (MAM). The MAM is a convolution-based dynamic mask generator that considers the neighbor nodes of each patch, evaluates
   300	the weights of each channel based on the correlation of local information, and thus cuts the overall prototype and selects valid subspace for matching by generating masks.
   301	
   302	advantages of convolution neural networks, the weight of each channel
   303	is evaluated through the correlation of local information. Since the
   304	pure converter cannot focus on the information of the neighbor nodes,
   305	we divide it into feature mapping tables to non-overlapping Windows
   306	according to the structure of the Swin-Transformer [16]. The module
   307	can take into account the neighbor nodes of each patch, so that more
   308	useful information is concentrated in the mask image, the final mask is
   309	generated as:
   310	𝑓𝑟 = 𝑅𝑒𝑠ℎ𝑎𝑝𝑒[𝑥𝑟1 ; 𝑥𝑟2 ; ...; 𝑥𝑟𝐾 ],
   311	(4)
   312	𝑓 ′ = 𝐹𝑐 𝑜𝑛𝑣 (𝐶 ∗ (𝑓1 , 𝑓2 , … , 𝑓𝑟 )),
   313	𝐹𝑀 𝐴𝑀 = 𝜎(𝐴𝑣𝑔 𝑝𝑜𝑜𝑙(𝑓 ′ )),
   314	
   315	MACM, we only use class token as a global representation, so when
   316	ignoring the key vector of class token itself, we can get the attention
   317	mapping between class token and image patch. As follows:
   318	√
   319	𝑞 𝑐 𝑙𝑠 (𝐾𝐿𝑖𝑚𝑔 )𝑇
   320	𝑐
   321	𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥( 𝐿
   322	), 𝑑 =
   323	,
   324	(7)
   325	𝑑
   326	𝑁
   327	where 𝑁 is the number of headers, 𝑞𝐿𝑐 𝑙𝑠 is the query vector of class
   328	tokens in the transformer block, 𝐾𝐿𝑖𝑚𝑔 is the query vector of image
   329	patches in the transformer block.
   330	A standard orthogonality constraint is applied to separate the attention force of each head:
   331	
   332	(5)
   333	
   334	2
   335	
   336	𝑜𝑟𝑡ℎ𝑜 = ‖𝑊 𝑇 𝑊 − 𝐼𝑁 ‖𝐹 ,
   337	
   338	where, 𝜎 denotes the sigmoid function, 𝐹𝑐 𝑜𝑛𝑣 denotes the convolution
   339	layer, and C ∈ R1×𝐿 is for a gate control unit that selects the MAM
   340	input.
   341	
   342	where ‖.‖2𝐹 is Frobenius norm, the 𝐼𝑁 is the identity matrix, W is a
   343	normalized matrix with each row being 𝐿2 normalized. The use of
   344	orthogonal matrices not only makes the network structure more stable,
   345	but also facilitates the matching between the prototype mask and the
   346	holistic prototype.
   347	
   348	3.1.2. Multi-head Attention Constraint Module (MACM)
   349	Multi-headed self-attention provides multiple representation subspace for the model’s attention layer while enabling the model to
   350	focus on multiple focal regions. In the original transformer block, by
   351	calculating the cosine similarity, we found that the features of each
   352	patch become more similar as we go deeper into the network structure.
   353	However,some parts of obscured pedestrian, such as head, shoulder and
   354	neck should be assigned more attention. The original high-similarity
   355	feature input not only causes a waste of resources, but also makes
   356	it difficult for MAM to generate a valid prototype mask. Therefore,
   357	we introduce MACM to promote class tagging in multiple heads to
   358	aggregate different information and enrich the feature representation
   359	starting from a shallow structure as shown in Fig. 4. Subsequently, a
   360	large number of feature representations of the rich dataset will be used
   361	to train the overall prototype for each identity.
   362	Usually, the long side pays attention to the use of query matrix q
   363	∈ R 𝑑𝑝 , key matrix k ∈ R 𝑑𝑘 , value matrix v ∈ R 𝑑𝑣 , each attention head
   364	is calculated as:
   365	ℎ𝑖 = 𝑓 (𝑊𝑖(𝑞) 𝑞 , 𝑊𝑖(𝑘) 𝑘, 𝑊𝑖(𝑣) 𝑣) ∈ R𝑃𝑣 .
   366	
   367	(8)
   368	
   369	3.2. Pose feature aggregation module
   370	
   371	Posture Estimation. In order to better match the image, the model
   372	should pay more attention to the unblocked part of the image in the
   373	image input stage. Because there is less body information available
   374	in the occluded image, the matching performance is degraded. MTIPE
   375	uses a pose estimator to detect human key-points from human images.
   376	Given an input image 𝐼, MTIPE extracts 𝐾 landmarks using the pose
   377	estimator. For each extracted landmark its position coordinates and
   378	confidence score are obtained by the pose estimator. To classify the
   379	landmarks into high and low confidence levels, we set a threshold to
   380	filter out the landmarks with confidence scores less than 𝛽. Expressed
   381	as:
   382	{
   383	(𝑥𝑖 , 𝑦𝑖 ), 𝑖𝑓 𝑆 𝑐 𝑜𝑟𝑒𝑖 ≥ 𝛽 ,
   384	𝐿𝐾𝑖 =
   385	(9)
   386	0, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒.
   387	
   388	(6)
   389	
   390	𝑃𝑞 ×𝑑𝑞
   391	Among the parameters that can be learned are W(𝑞)
   392	, W(𝑘)
   393	𝑖 ∈R
   394	𝑖 ∈R
   395	𝑃𝑘 ×𝑑𝑘 , W(𝑣) ∈R 𝑃𝑣 ×𝑑𝑣 , and the functions representing attention pooling
   396	𝑖
   397	
   398	Here, 𝐿𝐾𝑖 denotes the 𝑖th landmark, and (𝑥𝑖 , 𝑦𝑖 ) denote the coordinates of the 𝑖th landmark, 𝑆 𝑐 𝑜𝑟𝑒𝑖 is the confidence score and 𝛽 is the
   399	threshold. The landmarks are then used to generate heat maps 𝐻. The
   400	
   401	can be additive attention and scaled point-product attention. In the
   402	4
   403	
   404	Pattern Recognition 159 (2025) 111082
   405	
   406	T.-T. Yuan et al.
   407	
   408	Fig. 4. As the network structure deepens, the feature representation of each patch becomes smooth and the prototype mask generated by MAM is difficult to function in the case
   409	of highly similar inputs, so we propose the Multi-Head Attention Constraint Module (MACM) to enrich the feature representation by aggregating different feature representations
   410	in multiple heads in the transformer block.
   411	
   412	maximum response point of each heat map corresponds to a human key
   413	point.
   414	
   415	𝑄𝑖 = 𝑃𝑖 𝑊𝑞 , 𝐾𝑗 = 𝑓𝑑𝑗 𝑒 𝑊𝑘 , 𝑉𝑗 = 𝑓𝑑𝑗 𝑒 𝑊𝑣 ,
   416	
   417	(10)
   418	
   419	where 𝑖 = 1, 2, … , 𝑝𝑖 , 𝑗 = 1, 2, … , 𝐷, linear projections 𝑊𝑞 ∈ R𝐷×𝑑𝑘 , 𝑊𝑘 ∈
   420	R𝐷×𝑑𝑘 , 𝑊𝑣 ∈ R𝐷×𝑑𝑣 are applied to learnable parameters and features
   421	𝑓𝑑 𝑒 , respectively.
   422	
   423	Feature Aggregation Module. In order to integrate the extracted
   424	global feature information and the pose information from the person in
   425	the image, the pose feature aggregation module multiplies the heat map
   426	𝐻 with the feature 𝑓𝑔𝑒 and outputs the pose feature map 𝑃𝑖 . Since each
   427	heat map 𝐻 shows the information focus on different regions of the
   428	human body, the purpose of the pose feature aggregation module map
   429	is to suppress the interference information from the occluded regions
   430	by focusing on the non-occluded part of the target person. Although the
   431	pose feature aggregation module has encoded information from different parts of the human body, we still want to find the part of the 𝑓𝑔𝑒 that
   432	contributes most to the information of the obscured person’s body parts.
   433	{
   434	}
   435	Therefore, firstly, the feature aggregation set 𝑆 = 𝑆𝑖 |𝑖 = 1, 2, … , 𝐾 ,
   436	and in the feature matching module, we use matching with the local
   437	feature view set output by the decoder through the calculation of the
   438	ensemble similarity measure.
   439	
   440	Feature Matching Module. Since there are many interfering elements
   441	in the occluded picture, various noises are introduced to lead to mismatch. Meanwhile, the occluded picture may have similar features to
   442	different parts of the human body. Local prototype learning tends to
   443	focus on the same different parts of the human body in the picture,
   444	which may lead to matching failure. In order to obtain better information related to key points of human body, we set a feature matching
   445	module. Since every feature of the feature cluster S output by attitude
   446	feature matching is related to a certain key point information of human
   447	body, we can find the local prototype related to a certain key point of
   448	human body by calculating the similarity between the local prototype
   449	set and the feature cluster.
   450	
   451	3.3. Local prototype based transformer decoder
   452	
   453	Decoder Supervision Loss. The probability of containing human body
   454	parts information in the image can be judged according to the confidence level of key-points, and divided into key-points with high
   455	𝑖
   456	confidence 𝐅𝑘ℎ = {𝑓𝑘ℎ
   457	= |𝑖 = 1, 2, … , 𝐿} and the rest form keypoints with low confidence 𝐅𝑘𝑙 = {𝑓𝑘𝑙𝑖 = |𝑖 = 1, 2, … , 𝑃𝑖 − 𝐿}. We
   458	first perform the operation of averaging pooling for two features with
   459	different confidence levels to obtain 𝑓𝑝𝑘ℎ and 𝑓𝑝𝑘𝑙 . We then propose the
   460	Posture Key-points Loss function as:
   461	⟨
   462	⟩
   463	𝑖 ,𝑓 𝑖
   464	𝐵
   465	𝑓𝑝𝑘ℎ
   466	∑
   467	𝑝𝑘𝑙
   468	1
   469	𝑝𝑘 =
   470	,
   471	(11)
   472	𝐵 𝑖 ‖𝑓 𝑖 ‖‖𝑓 𝑖 ‖
   473	𝑝𝑘ℎ
   474	𝑝𝑘𝑙
   475	
   476	In the local part prototype based transformer decoder, in order to
   477	learn differentiated body parts, we introduce a set of learnable local
   478	prototypes so that we can match body parts in a robust manner. First,
   479	{ }𝑀
   480	we introduce a set of local prototypes 𝑃 = 𝑝𝑖 𝑖=1 , 𝑝𝑖 ∈ R1×𝑑 denotes
   481	a local classifier. It is used in the subsequent feature matching module
   482	to determine whether the learned distinguishing feature belongs to a
   483	part of the human body 𝑖. By computing all the local prototypes, we
   484	can obtain M local masks.
   485	These local prototypes are set as learnable parameters and added
   486	as queries to each attention layer. As shown in Fig. 2, the averaging
   487	pooling layer is applied to the heat map H and then multiplied by 𝑓𝑒𝑛 ,
   488	with the combination of outputs as keys and values, finally output the
   489	𝑓𝑑 𝑒 ∈ R(𝐾+1)×𝐷 . The averaging pooling operation causes the feature
   490	vector to fuse the information of visible body parts and ignore the
   491	occluded parts and redundant partial information. Formally, queries,
   492	keys and values can be formulated as:
   493	
   494	where 𝐵 denotes the training batch size. < ., . > denotes the inner
   495	product. This loss function is useful to compare the similarity of human
   496	body parts and non-human body parts. As shown in Eq. (11), if they
   497	are similar, the value of this loss function will be large. To obtain the
   498	decoder global features, we add the identity loss and triplet loss to the
   499	decoder loss.
   500	5
   501	
   502	Pattern Recognition 159 (2025) 111082
   503	
   504	T.-T. Yuan et al.
   505	
   506	1 ∑
   507	𝑖
   508	 (𝑃 (𝑓𝑘ℎ
   509	))
   510	𝐿 𝑖 𝐼𝐷
   511	
   512	set consists of 3368 images, and the gallery set consists of 19,732
   513	images.
   514	
   515	𝐿
   516	
   517	𝑑 𝑒 = 𝐿𝐼 𝐷 (𝑃 (𝑓𝑝𝑘ℎ )) +
   518	
   519	1 ∑
   520	+𝑇 (𝑓𝑝𝑘ℎ ) +
   521	 (𝑓 𝑖 ) + 𝛼𝑜𝑟𝑡ℎ𝑜 .
   522	𝐿 𝑖 𝑇 ℎ
   523	𝐿
   524	
   525	(12)
   526	
   527	MSMT17 [2] contains 126,441 images of 4101 IDs captured from 15
   528	camera network. The training set has 32,621 images of 1041 identities,
   529	and the testing set has 93,820 images of 3060 identities.
   530	Duke-MTMC [20] contains of 36,411 images of 1812 persons from 8
   531	cameras. 16,522 images of 702 persons are randomly selected from
   532	the datasets as the training set, and the remaining images are divided
   533	into the testing set containing 2228 query images and 17,661 gallery
   534	images.
   535	
   536	3.4. Pooling Attention Module (PAM)
   537	According to the standard architecture of transformer, we propose
   538	to use a PAM to extract the input image 𝐼 local context information and
   539	merge it into the local prototype. We can get the updated set of local
   540	{
   541	}𝑀
   542	prototypes 𝑝𝑎𝑡𝑡𝑛
   543	. The input 𝐼 is reshaped, and then the average
   544	𝑖
   545	𝑖=1
   546	pooling layer is applied on the reshaped 𝐼 to obtain 𝑃𝑖 . Then 𝑃𝑖 is
   547	provided to the depth convolution of the relative position encoding:
   548	𝑃𝑖𝑒𝑛 = 𝐷𝑊 𝐶 𝑜𝑛𝑣(𝐴𝑣𝑔 𝑃 𝑜𝑜𝑙𝑖𝑛𝑔(𝐼𝑖 )) + 𝑃𝑖 , 𝑖 = 1, 2, … , 𝑛.
   549	
   550	Evaluation Protocol. To verify fair comparison with other methods,
   551	we adopt the widely used Cumulative Matching Characteristic (CMC)
   552	and mean Average Precision (mAP) as evaluation metrics and follow
   553	the evaluation settings provided by existing occluded methods.
   554	
   555	(13)
   556	
   557	4.2. Implementation details
   558	
   559	Afterwards, the feature map is flattened and connected.
   560	𝑃 = 𝐿𝑎𝑦𝑒𝑟𝑁 𝑜𝑟𝑚(𝐶 𝑜𝑛𝑐 𝑎𝑡(𝑃𝑖𝑒𝑛 )).
   561	
   562	We use ViT [7], which was pre-trained on ImageNet [21], as the
   563	backbone network. Both training and test images were resized to
   564	256 × 128 and the commonly used horizontal flip, fill, random crop
   565	and random erase [22] were used as data enhancements. Additional
   566	color dithering enhancement was used to avoid domain variance when
   567	testing in the Occluded-REID. With the success of TransReID [17], we
   568	used a lower step size and set 𝜆 to 3.0. The transformer decoder was
   569	the same as in [18]. During the training stage, the batch size was set to
   570	64 with 4 images per ID. The SGD is utilized as the optimizer, in which

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Internal-External Context Interaction Network for Person Re-Identification.pdf' - | nl -ba | sed -n '337,670p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   337	covers three parts. Firstly, we present the overall framework
   338	of IEINet. Secondly, we provide detailed information about
   339	the spatial and channel modules that are developed using the
   340	IEI attention mechanism. Lastly, we describe the loss function
   341	used for IEINet.
   342	A. Network Architecture
   343	As shown in Figure 2, IEINet is a multi-branch network
   344	structure, and this network consists of a ResNet-50 backbone
   345	and three branches (Branch-C, Branch-S1 and Branch-S2). We
   346	select ResNet-50 [52], the CNN that is most widely employed
   347	in the Re-ID field, as the basis for feature extraction of
   348	IEINet and change the step size of the spatial down-sampling
   349	operation in the res conv5 layer of this network from 2 to 1,
   350	thus allowing our IEI modules to extract important information
   351	at a higher spatial resolution. Following previously developed
   352	person Re-ID models [12], [53], we use the residual layers
   353	before the res conv5 layer of the ResNet-50 network as the
   354	IEINet backbone. Then, the res conv5 layer is duplicated into
   355	
   356	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
   357	
   358	736
   359	
   360	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   361	
   362	2
   363	
   364	Fig. 3. Diagram of IEI-S. CAP denotes channel average pooling. denotes
   365	denotes the sum of all
   366	the element-wise multiplication operation, and
   367	products. ⊕ denotes the element-wise sum operation.
   368	
   369	three parts, and an IEI module is added after each part to form
   370	three separate branches.
   371	Adding the IEI-C module to Branch-C allows this branch to
   372	highlight important channels that are useful for discriminating
   373	identities. The feature map output by the channel module is
   374	subjected to a generalized-mean pooling [54] operation to
   375	produce a 2048-dim feature vector that serves as the final
   376	feature embedding for this branch. IEI-S modules are added
   377	to Branch-S1 and Branch-S2, and the feature maps output
   378	by these two modules are subjected to a generalized-mean
   379	pooling operation to obtain two 2048-dim feature vectors.
   380	To obtain compact feature representations, we execute the
   381	dimensionality reduction operation consisting of a 1 × 1
   382	convolution operation, a batch normalization (BN) operation
   383	and a rectified linear unit (ReLU) activation operation on these
   384	two feature vectors to reduce their dimensions to 1024 and
   385	obtain the final feature embeddings of these two branches. Furthermore, to make the two spatial branches capture different
   386	meaningful information in the spatial dimension, we introduce
   387	a region-diverse loss (Lrd ) to make their two spatial modules
   388	focus on different important regions, as described in Section
   389	III-D. During the testing phase, we concatenate the feature
   390	embeddings of all branches to form the feature representation
   391	of a person image.
   392	B. Spatial Internal-External Context Interaction Module
   393	The IEI-S module aims to use the interactions between the
   394	contextual information of different samples and the spatial
   395	local context in a sample to infer the important positions
   396	related to the sample identity and refine the representations of
   397	all positions; the structure of this module is shown in Figure 3.
   398	Let the feature map X ∈ RC×H×W of a sample be the input
   399	tensor of the module, where C, H, and W are the number of
   400	channels, spatial height, and width, respectively. To construct
   401	a spatially robust contextual representation of a single sample,
   402	we implement a channel average pooling (CAP) operation on
   403	the input feature map to obtain a matrix Q s of size H × W,
   404	where one element represents the overall activation response
   405	of a location. Q s reflects the robust activation contextual
   406	information of the feature map in space.
   407	
   408	2
   409	
   410	We introduce a key matrix K s ∈ Rk ×k that stores the
   411	spatial contextual information between samples, which is a
   412	learnable external memory that is independent of the individual samples, and it learns the spatial contextual information
   413	between samples that is most discriminative regarding the
   414	identities of all samples in the entire dataset. The aggregation weights of each position within a sample are obtained
   415	by interacting the inter-sample context with the local selfinformation within the sample. The local context information
   416	for each location within a sample is extracted by implementing
   417	an unfolding operation, which is similar to the convolution
   418	operation.
   419	Specifically, we use a window with the size of k × k to
   420	slide on Q s to obtain a local context query matrix Q s0 with
   421	the size of HW × k2 , where the sliding step is 1, and the
   422	information in the k × k window of one position is extracted
   423	by sliding at a time, with a total of HW times. Each row Qis0 ,
   424	i ∈ {1, 2, . . . , H W} of Q s0 is a local context query vector for
   425	one location. Note that similar to the convolution operation,
   426	the window size k can be set to an odd number (3, 5, 7, etc.)
   427	to maintain a constant number of positions and to facilitate
   428	filling.
   429	Next, we project the local context Qis0 at the ith location
   430	onto the inter-sample spatial context K s , and obtain the k × k
   431	aggregation weights of the ith position via the softmax normalization operation. The local information of all positions is
   432	projected onto K s to realize the internal and external context
   433	2
   434	interaction and generate the attention matrix A s ∈ RHW×k .
   435	The matrix A s is calculated as follows:
   436	
   437	A s = softmax Q s0 ⊗ K s
   438	= softmax (Unfold (Q s ) ⊗ K s )
   439	(1)
   440	where ⊗ denotes the matrix multiplication operation and each
   441	row Ais of A s is the aggregation weights of a position. The
   442	aggregation weights of each location are learned from its local
   443	context information under the guidance of the inter-sample
   444	context. This method emphasizes the important identity-related
   445	locations in the sample, and makes use of the rich information
   446	in the neighborhood of each location in the sample.
   447	After obtaining the attention matrix A s , we refine the representations of all locations based on the aggregated weights
   448	of each location within the sample. We first implement an
   449	embedding function θ(·) for X to obtain a feature map V s
   450	with the same size as X. The embedding function θ(·) is
   451	implemented as a 1 × 1 convolution with a BN layer and
   452	a ReLU activation function. The feature map V s is then
   453	0
   454	2
   455	expanded and deformed into a tensor V s ∈ RHW×k ×C by an
   456	s0
   457	k2 ×C
   458	unfolding operation, where Vi ∈ R
   459	is the value matrix of
   460	position i, and there are H × W value matrices. The matrix
   461	multiplication operation of Ais and Vis0 is performed to obtain
   462	the refined output at position i. The refined outputs for all
   463	positions can be expressed as:
   464	V ms = A s ⊗ V s0 = A s ⊗ Unfold(θ(X))
   465	
   466	(2)
   467	
   468	where A s is changed into a tensor with the shape of HW×1×k2
   469	before performing matrix multiplication.
   470	Finally, the shape of V ms is resized to C × H × W,
   471	and Gaussian error linear unit (GeLU) activation and BN
   472	
   473	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
   474	
   475	LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
   476	
   477	737
   478	
   479	local context query matrix and the inter-sample context key
   480	matrix. The matrix Ac is calculated as:
   481	
   482	Ac = softmax Qc0 ⊗ K c
   483	= softmax (Unfold (Qc ) ⊗ K c )
   484	(4)
   485	
   486	Fig. 4. Diagram of IEI-C. SAP represents spatial average pooling.
   487	represents the element-wise multiplication operation between vectors and
   488	represents summation. ⊕ represents element-wise addition operation.
   489	
   490	operations are implemented on it. Then the final output
   491	feature map X s of the module is obtained by elementwise adding operation. These operations can be defined as
   492	follows:
   493	X s = BN (GeLU (V ms )) + X
   494	
   495	(3)
   496	
   497	C. Channel Internal-External Context Interaction Module
   498	As a complement to the IEI-S module, the IEI-C module
   499	aims to emphasize important channels related to sample identities and refines the representations of all channels via the
   500	interaction of inter-sample contextual information with intrasample channel local context. The structure of this module is
   501	shown in Figure 4. We use the feature map X ∈ RC×H×W
   502	as the input tensor for this module. To construct a robust
   503	contextual representation of a sample in terms of channels,
   504	we implement a spatial average pooling (SAP) operation on
   505	the input feature map X to obtain a C-dim vector Qc , which
   506	reflects the activation contextual information of the feature
   507	map in the channel dimension.
   508	To learn discriminative information in terms of channels,
   509	we introduce a key matrix K c ∈ Rk×k that stores intersample channel context information and let it interact with
   510	the intra-sample channel local context information to generate
   511	aggregated weights for each channel. The local contextual
   512	information for each channel within a sample is extracted by
   513	applying an unfolding operation to Qc .
   514	Specifically, we use a window with size of k × 1 and step
   515	of 1 to slide C times on Qc to obtain a local context query
   516	matrix Qc0 for channels with size of C × k, where the jth row
   517	Qc0j , j ∈ {1, 2, . . . , C} is the local context query vector for
   518	the jth channel, which comes from the information in the k
   519	neighborhood of the jth element in Qc . Similarly, to keep the
   520	number of channels constant as well as to facilitate filling, k
   521	is defined in the same way as in the spatial module.
   522	Next, we establish the internal and external contextual
   523	interactions of the sample and obtain the channel attention
   524	matrix Ac ∈ RC×k by multiplying the intra-sample channel
   525	
   526	where the jth row Acj is the aggregation weights of the jth
   527	channel, which is obtained by projecting the local context Qc0j
   528	onto the inter-sample channel context K c and implementing
   529	the softmax normalization operation. This interaction process
   530	takes full advantage of the neighborhood information of channels in a sample and the potential correlation information
   531	between samples to generate the aggregation weights.
   532	After obtaining the channel attention matrix Ac , we refine
   533	the representations of all channels based on the aggregated
   534	weights of each channel within the sample. We first implement
   535	an embedding function δ(·) similar to θ(·) for X to obtain a feature map V c with the same shape as X. Then, the feature map
   536	V c is expanded and deformed into a tensor V c0 ∈ RC×k×HW
   537	k×HW
   538	is the value
   539	by an unfolding operation, where V c0
   540	j ∈ R
   541	matrix of channel j, and there are C value matrices. After
   542	that, the aggregated weights of each channel are multiplied
   543	by the corresponding value matrix to refine the outputs of all
   544	channels, and this process can be expressed as:
   545	Vmc = Ac ⊗ V c0 = Ac ⊗ Unfold(δ(X))
   546	
   547	(5)
   548	
   549	Finally, the shape of V mc is resized to C × H × W, and
   550	GeLU activation and BN operations are implemented on it.
   551	Then the final output feature map X c of the module is obtained
   552	by element-wise adding operation. These operations can be
   553	defined as follows:
   554	Xc = BN (GeLU (V mc )) + X
   555	
   556	(6)
   557	
   558	D. Loss Functions
   559	During the network’s training process, we use the crossentropy loss with label smoothing regularization (LSR) [55]
   560	and the batch hard triplet loss [56], which are commonly
   561	employed in person Re-ID approaches [1], [12], [57]. In addition, to enable IEINet to extract more different discriminative
   562	features, we also propose a region-diverse loss.
   563	1) Cross-Entropy Loss With LSR: The cross-entropy loss
   564	function is frequently utilized in Re-ID tasks, and LSR is
   565	implemented to avoid overfitting of the training data by the
   566	network during training. This loss can be defined as:
   567	8
   568	N−1
   569	N
   570	<
   571	X
   572	ε, i = y
   573	qi = 1 −
   574	(7)
   575	Lid = −
   576	qi log (pi )
   577	N
   578	:q = ε/ , i , y
   579	i=1
   580	
   581	i
   582	
   583	N
   584	
   585	where N is the number of persons in the training set, and y is
   586	the ground-truth label of the image input into the network, and
   587	pi represents the probability of the input image being classified
   588	as the ith person. qi is the smoothed label and ε is a small
   589	constant that is set to 0.1.
   590	2) Batch Hard Triplet Loss: To improve the ranking performance of the network and to increase the differences between
   591	different persons, we select the batch hard triplet loss for
   592	network metric optimization. To satisfy the computational
   593	requirements of this loss, we randomly select P persons and
   594	
   595	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
   596	
   597	738
   598	
   599	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   600	
   601	TABLE II
   602	D ETAILED C HARACTERISTICS OF F OUR DATASETS
   603	
   604	K images of each person from the training set to form a minibatch. The expression of this loss function is:
   605	P X
   606	K 
   607	X
   608	
   609	
   610	Ltriplet =
   611	α + max D f xai , f xip
   612	i=1 a=1
   613	
   614	p=1,...,K
   615	
   616	3
   617	−
   618	
   619	min
   620	
   621	n=1,...,K
   622	j=1,...,P, j,i
   623	
   624	D f xa , f xn 5
   625	
   626	i
   627	
   628	
   629	j
   630	
   631	(8)
   632	+
   633	
   634	where f (xai ), f (xip ), and f (xnj ) are features extracted from the
   635	anchor, positive, and negative samples, respectively, and the
   636	margin α indicates the minimum boundary distance between
   637	the positive and negative sample pairs. D(·, ·) represents the
   638	Euclidean distance between two features. The max and min
   639	functions are utilized to obtain the hardest positive and negative sample pairs, respectively.
   640	3) Region-Diverse Loss: To improve the performance of
   641	IEINet in the person Re-ID task, we design a loss that is used
   642	between the two spatial branches of IEINet (Branch-S1 and
   643	Branch-S2) to differentiate the features that they learn, which
   644	allows the network to capture more information that is useful
   645	for identification. The loss is named the region-diverse loss,
   646	and its expression is:
   647	T
   648	
   649	W s1 X s1 ⊗ (W s2 X s2 )
   650	Lrd =
   651	||W s1 X s1 ||2 · ||W s2 X s2 ||2
   652	
   653	(9)
   654	
   655	where X s1 ∈ RC×HW and X s2 ∈ RC×HW are the feature
   656	tensors output from the IEI-S modules in Branch-S1 and
   657	Branch-S2, respectively. W s1 and W s2 ∈ R1×C are both
   658	linear transformation matrices, and || · ||2 represents the L2norm. This loss reflects the similarity degree between the
   659	important regions of interest for these two branches. The lower
   660	the loss value, the more the difference between the regions
   661	they emphasize. Therefore, IEINet is able to capture more
   662	meaningful information after being trained and optimised by
   663	this loss.
   664	4) Total Loss: The total loss of IEINet is calculated as
   665	follows:
   666	Ltotal = Lid + Ltriplet + λLrd
   667	(10)
   668	where λ is a weight hyperparameter for adjusting the importance of the region-diverse loss (Lrd ) in the overall loss.
   669	IV. E XPERIMENTS
   670	A. Datasets and Evaluation Metrics

codex
前三篇的完整方法正文已经读完。第一篇是典型“遮挡导致部位误匹配，所以把姿态和动态掩码合起来”的路子；第二篇更像把任务交互方式重定义；第三篇则是把注意力从单样本上下文扩到跨样本潜在上下文。继续读剩下三篇。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|diffusion|inpainting|re-ranking|occlusion\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6:Latent Diffusion-Guided Feature Inpainting for Occluded Person
7:Re-Identification With Hybrid Re-Ranking
10:Abstract—Occlusion remains a persistent challenge in person
11:re-identification (ReID). Existing approaches either rely on
13:make models resistant to occlusion through real or synthetic
14:occlusion augmentations. While these methods suppress the effect
15:of occlusion, they do not solve the underlying problem, since
17:directly address this gap, we formulate occlusion as a feature-level
18:distortion and propose a Latent Diffusion guided De-Occluder
22:occlusion-free priors, our method eliminates the need for the
23:backbone to model occlusion explicitly, leading to inherently more
26:known at test time, we introduce a Hybrid Re-Ranking (HRR)
28:re-ranking by leveraging centroid-based identity anchors to
29:refine k-reciprocal re-ranking, thus boosting retrieval precision
30:while suppressing noise. Extensive experiments on standard and
31:occlusion-focused benchmarks confirm that our approach not
32:only overcomes the shortcomings of existing occlusion handling
35:merely resisting occlusion.
38:requiring neither explicit occlusion masks nor paired cleanoccluded images. By removing occlusion-induced artifacts, the
40:To achieve our goal of feature inpainting, we introduce a latent
41:Diffusion guided De-Occluder (DDO), a generative framework that reconstructs clean identity-preserving embeddings
42:from partially occluded inputs. Unlike previous approaches
44:occupancy estimation [5]–[7] our method leverages the
45:generative capabilities of latent diffusion models [8] to
47:space, enabling occlusion-aware representations to remain
54:varying viewpoints, lighting conditions, and occlusion across
57:two fundamental limitations remain: (i) the heavy reliance on Fig. 1: Illustration of occlusion-induced distortion in feature
60:occlusion-induced corruption at the source, and (ii) naïve denotes its occluded counterpart Fo . The distortion is modeled
61:(T ) from Fclean to Fo , which our method
75:two novel innovations. First, we propose a novel method to
76:tackle occlusion by reconstructing clean feature embeddings
77:Unlike standard Denoising Diffusion Probabilistic Models
78:from occluded inputs in the latent space. Treating occlusion (DDPM) [9] and Denoising Score Matching (DSM) [10] that
80:problem as a Feature Inpainting task and employ a diffusion- explicit paired supervision between clean and occluded features,
90:inpainting in ReID where identity consistency is paramount.
103:Second, we propose a novel Hybrid Re-Ranking scheme
117:re-ranking to centroid-curated candidate sets, thereby reducing
119:re-ranking as a centroid-guided refinement over the manifold
123:and evaluate it on several standard benchmark datasets including Market-1501 [14], CUHK03 [15] and occlusion-focused
125:and P-ETHZ [18]. Our approach achieves new state-of-theart performance across the board, in terms of both mean
127:generalizability of our approach.
129:• We formulate the occlusion problem as a feature-level
130:inpainting task, wherein occluded regions in the latent space are reconstructed to recover clean, identitypreserving feature embeddings. These refined features are
132:retrieval under severe occlusion.
133:• We introduce the first latent Diffusion guided DeOccluder (DDO) module for person ReID, capable
136:is designed as a plug-and-play occlusion module that can
138:• We propose a Hybrid Re-Ranking (HRR) algorithm
140:re-ranking process, improving retrieval precision and
141:efficiency. This unified retrieval framework achieves stateof-the-art results on multiple standard and occluded ReID
147:spatial discrimination via horizontal stripes, yet fixed partitioning is brittle under pose variation and occlusion. Bak et al. [20]
149:but rigid patch layouts still fail under heavy occlusion.
150:Recent occlusion-aware ReID methods mainly suppress or
152:use adaptive position extraction with occlusion weighting, while
153:Dong et al. [22] disentangle identity cues from occlusion noise
160:occlusion.
162:ReID [48] and SPReID [41] incorporate human pose or semantic parsing to improve alignment. However, these methods
167:Generative methods for occlusion handling include GANbased approaches [5], [47] that operate in pixel space to
169:occluded samples to improve robustness. While visually plausible, these methods are prey to mode collapse and vanishing
172:implicitly learn to handle occlusions), often failing to improve
173:downstream matching under severe test-time occlusion. In
174:contrast, latent diffusion models [8] offer a principled generative
175:framework for feature-space reconstruction, where iterative
176:denoising [9] enables fine-grained inpainting of corrupted
179:for re-ranking [53] improve ranking quality by exploiting local
182:outliers. Conversely, centroid-based retrieval methods [13] use
186:To systematically position our approach within the broader
187:landscape of occlusion handling, Table I categorizes the major
188:methodological directions in the literature, contrasting their
189:core principles and occlusion treatment strategies with our
191:In contrast to these approaches, our method addresses
192:occlusion at its source by learning a generative inverse mapping
197:Early person ReID methods largely used CNN backbones. guidance through diffusion-based inpainting, not implicit
208:TABLE I: Principles of different methods for handling occlusion in person ReID. The Comments column contrasts each approach
209:with our latent Diffusion based De-Occluder (DDO) and Hybrid Re-Ranking (HRR) formulation.
218:Mathematical modeling of occlusion
225:Theoretical framework for occlusion effects but
242:Improve spatial granularity but embeddings remain corrupted under occlusion; no explicit
266:sensitive to occlusion; augmentation improves
267:tolerance, not de-occlusion.
281:external estimator quality. No explicit occlusion
284:Occlusion aware modules - augmentation and attention/masking for robustness
296:Train backbones to suppress/reweight occlusion
310:DDO - Explicit de-occlusion in feature
326:training. Novel algorithm to restrict re-ranking
334:Inpainting
346:re-ranking
348:Fig. 2: Methodology Overview: Image features Fo are extracted using a ViT encoder and processed through the pretrained
355:Our approach models occluded and unoccluded features as
356:Furthermore, unlike prior centroid-based methods [54], [55]
358:graph-theoretic framework where centroids perform topology than adapting the ReID model to tolerate occlusion-induced
360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
361:guarantees denser reciprocal neighborhoods with reduced false reconstruction of identity-preserving embeddings by inpainting
364:problem inherent in standard re-ranking when gallery sets
365:exhibit high intra-class variance. Together, these contributions in Figure 2, directly addresses partial occlusions using a Latent
366:yield consistent state-of-the-art performance across all evaluated Diffusion guided De-Occluder (DDO). Given an occluded
368:occlusion corruption rather than adapting models to tolerate it. module learns a mapping Fclean = DDO(Fo ) where Fclean
383:Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
391:for k-reciprocal re-ranking, rather than relying solely on noisy
397:The standard denoising objective of a diffusion model is
398:performance over prior approaches by explicitly modeling oc- given by:
405:A. Feature Inpainting via Diffusion guided De-Occluder (DDO)
413:Prior approaches make ReID models resistant to occlusion
416:address this by formulating occlusion as a feature inpainting
426:diffusion denoising
428:We conceptualize occlusion as a structured distortion in
431:by Fclean , are transformed into occlusion-corrupted counterparts
455:of T −1 (Equation 1), unifying diffusion-based denoising
459:ReID modules with clean, occlusion-free priors, thereby
462:While DDO is built upon the latent diffusion framework, its
466:This perspective aligns naturally with the theory of diffusion
467:forward diffusion process by predicting noise, requiring only
478:integrates the standard diffusion denoising objective with
479:Analogously, occlusion-induced corruption is treated as a
482:as a conditional denoising diffusion model where Lrecon acts as
486:clean priors rather than requiring them to be occlusion-aware.
495:applications of diffusion models and enables task-specific
497:feature inpainting. To highlight the distinct nature of our task
499:specific Latent Diffusion Model (LDM) from standard diffusion
500:to reconstruct the denoised embedding. A forward diffusion
501:frameworks, we compare the compute requirements of our
526:(Forward Diffusion):
556:Diffusion
566:Fig. 3: Latent Diffusion guided De-Occluder (DDO) Module Training. The ViT encoders are kept frozen throughout training
567:to ensure that the latent diffusion model focuses exclusively on the denoising task without altering the input representations.
576:diffusion models. DDO is designed for feature-space inpainting,
582:Stable Diffusion [8]
772:C. Hybrid Re-Ranking (HRR)
773:Standard re-ranking methods [51] compute pairwise distances
776:We propose Hybrid Re-Ranking (HRR), combining robust
778:By restricting re-ranking strictly to semantically relevant
783:graph for robust matching. This method is explained in Figure
788:for re-ranking. Lines between query and centroids are labeled
792:Hybrid Re-Ranking (HRR).
795:for Jaccard-based re-ranking.
799:compared to standard re-ranking approaches.
812:Geometrically, this procedure restricts re-ranking to identity
832:Standard re-ranking computes the Jaccard distance dJ (q, g)
857:We construct Cq = {A, B, C} and restrict re-ranking to gallery
900:Re-ranking can be interpreted as transductive inference over a
902:In standard re-ranking, the graph includes all gallery samples,
923:We evaluate our method on eleven standard and occlusionfocused short term and cloth changing ReID benchmark
924:datasets (Table III), consistently outperforming state-of-theart approaches. These results validate the effectiveness of our
925:latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
947:Stage 2: Local Re-ranking within Filtered Gallery:
984:inter-identity noise, yielding a more robust re-ranking graph and
986:more semantically meaningful subgraphs, improving re-ranking
989:re-ranking in this manner.
1085:We evaluate our method under two distinct person reidentification scenarios:
1098:Our method achieves state-of-the-art performance across all
1099:gallery. This is the general surveillance scenario which evaluated benchmark datasets, consistently outperforming exallows the presentation of unknown individuals to the isting approaches on both occluded and holistic Re-ID datasets.
1106:prototype-guided re-ranking to enhance retrieval accuracy further confirm that HRR effectively leverages identity-level
1109:The efficacy of the proposed DDO and HRR modules is
1113:we introduce our second contribution—Hybrid Re-Ranking benchmark ReID backbones, specifically TransReID [27] and
1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
1123:classify methods utilizing such techniques under the closed- produces superior performance across benchmarks and target
1125:only those methods that either employ centroid-based retrieval operating on a frozen ImageNet-pretrained encoder ensures it
1126:learns dataset-agnostic occlusion inversion (T −1 ) rather than
1131:Together, these metrics offer a robust evaluation of both function as a universal, plug-and-play occlusion handler. A
1134:proposed model.
1149:All experiments were conducted using 1 NVIDIA RTX domain adaptation [57], supports generalization to real-world
1150:3080 GPU, equipped with 10 GB of VRAM. The final ReID ReID. Therefore, DDO learns to target latent-space inpainting
1155:unchanged for all datasets, the compute specifications remain pipeline and our DDO-enhanced approach in Figure 6. Ocuniform throughout. To facilitate full reproducibility, Section clusion in the baseline model distorts identity cues, leading

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|yourself|others|unsupervised|visible|infrared\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Learning From Yourself to Others for Unsupervised
6:Visible-Infrared Re-Identification
9:Abstract—Unsupervised
10:visible-infrared
15:alleviating the significant modality gap between visible and
16:infrared modalities. Existing methods mainly focus on mitigating
21:framework for US-VI-ReID, which proactively identifies noisy
25:principle of learning from yourself to others. For intra-modality
32:(HCM) is proposed to establish local instance-level label
37:experiments demonstrate that our HCCL achieves competitive
41:Index Terms—Person re-identification, cross-modality, unsupervised learning, noisy labels.
65:ISIBLE-INFRARED person re-identification (VI-ReID)
66:aims to match visible and infrared images captured from
71:visible and infrared modalities. Meanwhile, it also encounters
83:surveillance systems amass vast amounts of data daily, making manual annotation impractical. Therefore, unsupervised
84:visible infrared person re-identification (US-VI-ReID) has garnered widespread attention in the community, which directly
90:etc. These methods primarily consider how to mine crossmodality correspondences based on pseudo-labels generated
102:samples. Existing methods [22], [23], [24] optimize noisy
115:Fig. 1. Illustration of our motivation. (a) Existing methods overlook the
118:our method proactively identifies noisy labels during forward propagation,
126:most approaches directly use noisy labels without effective
132:Although current advanced methods [25], [27], [28] attempt
142:of learning from yourself to others, where intra-modality
145:own hardness. Driven by this motivation, we propose a Hierarchical Centrality Collaborative Learning (HCCL) framework,
150:show strong associative connections with others in the same
161:the proposed HCM employs a novel local-to-global hierarchical matching paradigm, reinforced by the bidirectional
172:• We propose a novel Hierarchical Centrality Collaborative Learning (HCCL) framework for US-VI-ReID that
174:dependable cross-modality correspondences. This framework not only promotes the learning of identity-consistent
182:• Hierarchical Consistency Matching (HCM) is proposed
187:• Extensive experiments on the SYSU-MM01 and RegDB
190:some recently supervised VI-ReID methods. Moreover,
194:A. Unsupervised Single Modality Person ReID
195:Unsupervised single-modality person re-identification
197:visible cameras without relying on manually labeled
201:supervised methodologies. Specifically, Cluster Contrast
207:JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
212:model robustness. Cheng et al. [39] proposed a Neighbour
219:during the 24-hour retrieval, cameras will capture infrared
221:raise important cross-modality ReID problems, such as visibleinfrared person re-identification, which are difficult to address
222:with existing single-modality ReID methods due to the large
223:discrepancies between visible and infrared modalities.
224:B. Supervised Visible-Infrared Person ReID
225:Visible-infrared person re-identification (VI-ReID) has
229:between visible and infrared modalities, which can be broadly
237:rich semantic information from visible and infrared images.
238:In addition, Zhang et al. [41] proposed Adaptive Middlemodality Alignment Learning (AMMA), which employs an
243:feature space. Specifically, Zhang and Wang [18] proposed
246:Alignment and Affinity Inference framework (SAAI) [19] was
247:proposed to align semantic part features with the learnable
249:However, the impressive performance of the above methods
253:systems. Therefore, unsupervised visible-infrared person reidentification (US-VI-ReID) has been intensively researched
255:C. Unsupervised Visible-Infrared Person ReID
256:Unsupervised Visible-Infrared Person Re-Identification
262:making it highly applicable in real-world surveillance scenarios. Existing approaches [21], [24], [26], [39], [45], [46],
268:Yang et al. [22] introduced a novel Augmented DualContrastive Aggregation (ADCA) learning framework that
269:aggregates heterogeneous memory with cluster-level similarity, serving as a baseline for unsupervised VI-ReID. PGM
276:et al. [50] proposed a progressive contrastive learning with
277:hard and dynamic prototypes method, focusing on commonality, divergence and variety. GUR [24] adopted a bottom-up
279:against the hierarchical discrepancy. Yang et al. [26] proposed
280:a Shallow-Deep Collaborative Learning (SDCL) framework
283:Nevertheless, these methods lack an in-depth exploration
289:delays. In contrast, we propose the HCCL framework, which
293:In this section, the proposed Hierarchical Centrality Collaborative Learning (HCCL) is explained in detail. First, we
295:we introduce the proposed Closeness Centrality Selection
300:1, 2, . . . , Nv } and X r = {xir |i = 1, 2, . . . , Nr } denote the visible
301:and infrared sets with Nv and Nr images, respectively. The
305:The pipeline of the proposed HCCL is shown in Fig. 2,
314:Fig. 2. Illustration of our proposed HCCL. (a) Clustering stage. HCCL generates intra-modality pseudo-labels by DBSCAN, consistent with [22]. Different
315:shapes denote different identities. (b) is our proposed Closeness Centrality Selection (CCS), detailed in Sec. III-B. The darker the color of the instance, the
317:is our proposed reliable homogeneous contrastive loss. (c) is our proposed Hierarchical Consistency Matching (HCM), and will be described in Sec. III-C.
318:LDRHL denotes our proposed discriminative reliable heterogeneous loss.
325:of clusters in the visible and infrared modalities. To mitigate
334:recent approaches in US-VI-ReID [22], [24], [25], [26], [27].
357:1, 2, . . . kc } denote the c-th cluster of visible modality with kcv
359:in the infrared modality with kcr vertices.
360:For clear understanding, taking the visible modality as an
361:example, we omit the similar processing for the infrared
392:JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
468:l-th cluster in the infrared modality.
469:In general, the proposed CCE quantifies the reliability
475:in memory as done in existing methods, we store each cluster’s
477:the visible and infrared memory. Mv = {mv1 , mv2 , . . ., mvPv } and
481:feature dimension) denote the reliable memory in the visible
482:and infrared modality, respectively. We treat the memory
504:where mvc represents the c-th prototype stored in the visible
506:the infrared memory. Ωvc denotes the set of reliable homogeneous instances in the c-th cluster of the visible modality,
540:During the training, we update visible and infrared memories by a momentum updating strategy:
632:To establish cross-modality correspondences, existing methods [27], [28] construct a cost matrix by calculating Euclidean
642:within each cluster. These approaches suffer from two major
644:the method of centroid computation inherently incorporates
649:hierarchical paradigm. For local matching, Bidirectional Consistency Local Matching (BCLM) is proposed to associate
654:(DRHL) is proposed to learn modality-invariant and discriminative features and resist the huge cross-modality discrepancy.
668:i-th instance in c-th cluster from the visible modality, I(·) is
673:visible modality and F̃ r = { f˜ir | i = 1, 2, . . . , Pr } for the infrared
676:anchor instance f˜cv in the visible modality and all the anchor
677:instances F̃ r from the infrared modality to identify the most
678:relevant infrared instances for pairing, which is defined as:
684:f˜lr . The top-k candidates Nk ( f˜cv→r ) from the infrared modality
691:Existing methods [22], [26] establish cross-modality correspondences by using similarity metric, but they solely rely on
695:visible to the infrared modality. In reality, the red circle and
707:visible candidates to obtain Nk ( f˜lr→v ). Finally, local crossmodality positive label pairs R are established when f˜cv satisfies
726:anchor instances of the c-th and l-th clusters in the visible and
727:infrared modalities, respectively. It ensures that f˜cv and f˜lr are
733:2) Global Prototype Reliability Alignment: Existing methods [25], [28], [29], [30], [49] construct the cost matrix by
751:cost at the indices corresponding to the visible cluster containing f˜cv and the infrared cluster containing f˜lr is set to zero.
770:with matching the visible reliability prototype mvc and the
771:infrared reliability prototype mrl . In this way, the cost matrix
776:visible cluster and its corresponding infrared cluster while
780:JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
803:of visible and infrared prototypes, respectively; h·i denotes
878:denotes the corresponding label in the infrared
880:modality for the given sample fiv from the visible modality.
882:represents the visible pseudo-label matched
884:with fir from the infrared modality.
908:[27], we evaluate our method on two widely used crossmodality ReID datasets: SYSU-MM01 [58] and RegDB [59].
912:SYSU-MM01 [58] contains 22,258 visible images and 11,909
913:infrared images of 395 identities, captured by 4 RGB cameras
914:and 2 infrared cameras. Following official protocols [22],
919:is represented by 10 infrared images and 10 visible images.
920:There are two testing modes: infrared to visible and visible to
921:infrared. The former uses infrared images as the query set and
922:visible images as the gallery set, and the latter vice versa.
929:3) Implementation Details: Our HCCL framework is
945:epochs, followed by the training of our HCCL framework in
950:experimentally.
952:To comprehensively evaluate the effectiveness of our proposed method, we compare our methods with supervised and
953:unsupervised learning on SYSU-MM01 [58] and RegDB [59].
954:The experimental results are shown in Table I and Table II.
955:1) Comparison With Unsupervised Methods: We compared
956:existing unsupervised methods, among which ICE [33], IICS
958:unsupervised ReID approaches. However, these methods struggle to reduce cross-modality discrepancies, resulting in poor
960:As shown in Table I and Table II, our proposed HCCL
964:more impressively on RegDB (visible to infrared), attaining
980:gap (0.2%) suggests both methods have achieved highly superior performance in US-VI-ReID. Furthermore, we observe
994:Existing US-VI-ReID methods establish cross-modality correspondences through similarity metrics such as ADCA, DCCL
1001:approaches introduce latency and may lead to the establishment of incorrect cross-modality correspondences. In contrast,
1002:our method discards noisy samples during the forward propagation, allowing for immediate feature adjustments without
1010:visible and infrared prototypes, learning modality-invariant

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf' - | rg -n -i \"abstract|introduction|method|approach|proposed|framework|experiment|conclusion|granularity|transformer|visible|infrared\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:Learning multi-granularity representation with transformer for
9:visible-infrared person re-identification
35:Visible-infrared person re-identification
36:Multi-granularity
37:Transformer
40:ABSTRACT
41:Visible-infrared person re-identification (VI-ReID) aims to match pedestrian images from visible and nearinfrared modalities. The pedestrian images of two modalities contain discriminative features in different sizes
42:and positions, e.g., the global color of the cloth, the body’s local pose, and the shoe’s pixel size. However, existing methods mainly capture features at a single granularity, ignoring multi-granularity information contributing
43:to pedestrian identification. Therefore, we propose a cross-modality multi-granularity Transformer (CM2 GT)
44:framework to solve this issue. CM2 GT learns coarse-to-fine feature representations and integrates discriminative
46:alignment caused by matching single granularity features. Specifically, we first design a multi-granularity
47:feature extractor (MGFE) module based on Transformer to capture the global-patch-pixel level features of each
48:modality, which can flexibly represent semantic information at multiple scales. Secondly, a multi-granularity
49:fusion Transformer (MGFT) module mines the hierarchical relationships between multi-granularity features
50:by a saliency-enhanced Transformer, which ensures the identity-wise saliency consistency across different
53:between the anchor sample and its cross-modality nearest neighbor. Extensive experiments demonstrate that
54:our approach outperforms state-of-the-art methods.
56:1. Introduction
58:methods [1–3] concentrate on single-modality person re-identification
59:using visible images, leading to limited performance for pedestrian
60:matching in nighttime scenes. Therefore, visible-infrared person reidentification (VI-ReID) [4,5] is designed to retrieve pedestrian images
61:captured by both visible and near-infrared (NIR) cameras and promote
64:In the homogeneous context of visible-to-visible (VV-ReID) ReID,
65:multi-granularity features are extracted exclusively from RGB images,
68:both intra-modality and inter-modality multi-granularity features, in
71:For the VI-ReID task, it is critical to extract multi-granularity features. On the one hand, when two pedestrians only have a back
72:view, and the height and gender are the same, these multi-granularity
87:Fig. 1. (a) Global-level based methods capture the most salient region, (b) Part-level based methods consider part features, (c) Our approach combines multi-granularity features
90:granularity features can complement each other to contribute to the
91:identification of pedestrians when the features of the single granularity
94:extremities. Multi-granularity features can facilitate the discrimination
98:However, existing methods heavily rely on single-granularity feature extraction, and may result in irrelevant matching and ambiguous
99:alignment. For global-level based methods [6,7], the regions of interest
100:in cross-modality images of these methods may be irrelevant, which
101:leads to irrelevant matching. As shown in Fig. 1(a-1), in the visible
104:irrelevant matching problem. For part-level based methods [8,9], they
107:modality and the visible modality contain head features, but the visible
111:In this paper, we propose utilizing flexible multi-granularity features to solve these problems. First, multi-granularity information
112:is extracted for multi-level matching, instead of relying on singlegranularity features which may be semantically different. Second, the
113:cross-modality multi-granularity alignment could flexibly align each
117:we propose the cross-modality multi-granularity Transformer (CM2 GT)
118:framework, including the multi-granularity feature extractor (MGFE)
119:module and multi-granularity fusion Transformer (MGFT) module.
122:Transformer, to adequately obtain multiple granularities. MGFT module mines the hierarchical aggregation relationship between patch-level
123:and pixel-level features by interaction with multi-granularity features,
125:enhancing the discriminative ability of features. Specifically, we design a saliency-enhanced Transformer which preserves the saliency
136:(1) To the best of our knowledge, it is an early work on combining multi-granularity features based on Transformer structure in the
138:(2) We design a cross-modality multi-granularity Transformer
139:(CM2 GT) framework to capture multi-granularity features containing
142:(3) The cross-modality nearest-neighbor clustering (CNC) loss function is proposed to reduce the difference among intra-class crossmodality samples and promote more compactness among them.
143:(4) Experimental results on the large-scale public SYSU-MM01 [10]
144:and RegDB [11] datasets demonstrate that our approach outperforms
145:the state-of-the-art methods.
149:in a large gallery set using a query image from visible cameras. Lin
150:et al. [1] proposed an attribute-person recognition framework to learn a
154:et al. [3] proposed an incremental generative occlusion adversarial
161:3. Our approach
165:learning part-level features. These methods are a single-modality identification task, where only the intra-modality discrepancy is measured.
166:However, these methods cannot be applied to nighttime surveillance
167:scenarios due to the large difference between visible and near-infrared
168:modalities. In this paper, we propose a cross-modality multi-granularity
169:Transformer to learn coarse-to-fine feature representation and reduce
172:In this section, we introduce cross-modality multi-granularity Transformer (CM2 GT) framework for VI-ReID. Firstly, we give a brief
173:overview of the network structure. Secondly, we present a multigranularity feature extractor (MGFE) to capture global-, patch-, and
174:pixel-level features for each modality. Thirdly, a multi-granularity
175:fusion Transformer (MGFT) is introduced to fuse cross-modality information and interact with multi-granularity features. Finally, we
179:2.2. Visible-infrared person re-identification
180:Visible-infrared person re-identification attempts to match visible
181:images and near-infrared images of a person under disjoint cameras.
182:Different from previous visible–visible ReID task, VI-ReID not only
185:intrinsically different imaging processes of visible and near-infrared
187:Existing VI-ReID methods can be broadly classified into two main
188:categories, including modality compensation methods and feature
189:learning-based methods. In the first category of methods, Li et al. [12]
190:introduced an auxiliary modality to reduce the inter-modality discrepancy. Wang et al. [13] proposed model consists of a pixel generator,
192:of methods, Wei et al. [8] exploited part representations to learn more
195:These main methods effectively extracted single granularity features
196:by focusing on coarse or fine regions, leading to irrelevant matching and ambiguous alignment. In this paper, we capture multiplegranularity features based on the Transformer structure to solve these
200:The overview of our approach is shown in Fig. 2. Firstly, in the feature extractor, ResNet50 [20] pre-trained on ImageNet [21] is adopted
203:which learn modality-specific features. The visible modality feature
204:𝑋 𝑣𝑖𝑠 and near-infrared modality feature 𝑋 𝑛𝑖𝑟 are obtained by a twostream network. Secondly, MGFE module utilizes 𝑋 𝑣𝑖𝑠 and 𝑋 𝑛𝑖𝑟 to
225:MGFT fuses multi-granularity features (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 )
229:3.2. Multi-granularity feature extractor
230:As shown in Fig. 2, the multi-granularity feature extractor (MGFE)
233:each modality. The pixel branch uses Transformer on each local part
234:to capture pixel granularity features, which focuses on finer and richer
236:other local parts. These three-level granularity features can complement
238:The features from the visible and NIR modalities are processed
241:process of the visible modality is described in detail for simplicity.
243:2.3. Multi-granularity in computer vision field
244:In the field of computer vision, multi-granularity based methods
245:have achieved better performance. For example, In terms of multigranularity feature capture, [14] elucidated skeletal dynamics within
246:various contexts. Our approach extracts features across the granularity,
248:and comprehensive data feature map. At the multi-granularity feature
249:interaction level, [15] introduced a unified perception framework to
251:via attention mechanisms. Our approach captures features at multiple granularities and emphasizes their interaction and fusion across
255:As shown in Fig. 2, given visible modality feature 𝑋 𝑣𝑖𝑠 , we utilize
258:capture global granularity features 𝑋 𝑔 𝑙𝑜𝑏𝑎𝑙 , as follows:
260:2.4. Transformer in VI-ReID
261:ViT [17] is the first Transformer-based image classification model
262:that promotes the application of Transformers in the field of image classification. In VI-ReID, the Transformer models [18,19] show advantages
264:information directly, and proposed a modality-aware enhancement loss
268:These methods demonstrate that Transformers have shown superior
269:capability and great potential for different tasks. In our approach, we
270:propose a saliency-enhanced Transformer to preserve the saliency consistency across modalities and granularities and perform cross-modality
271:cross-granularity features alignment.
280:To capture long-range dependencies within each modality, we design a patch branch based on the Transformer module to obtain the
282:patch-level granularity feature 𝑋 𝑝𝑎𝑡𝑐 ℎ .
305:Fig. 2. Illustration of our cross-modality multi-granularity framework, including multi-granularity feature extractor (MGFE), multi-granularity fusion Transformer (MGFT), and total
306:loss function. MGFE consists of global, patch, and pixel branches, which aims to capture three-level granularity features. MGFT interacts feature pairs in the patch branch and
314:Transformer encoder to explore intra-modality relationships between
316:The Transformer encoder [17] contains a multi-head self-attention
334:capture pixel granularity features, and refer each pixel as a patch, so
336:Firstly, for visible modality image 𝑋 𝑣𝑖𝑠 ∈ R𝐶×𝐻×𝑊 , we use uniform
381:Thirdly, the 𝑚th sequence is fed into the Transformer encoder
382:to obtain spatially-aware pixel granularity representation 𝑥𝑣𝑖𝑠
384:Transformer encoder here is similar to the Transformer encoder in
387:granularity features is 𝑋 𝑝𝑖𝑥𝑒𝑙 , as follows:
417:3.3. Multi-granularity fusion transformer
418:The ambiguous alignment in single-granularity based methods can
420:a multi-granularity fusion Transformer (MGFT) module that effectively aligns features across multiple modalities and granularities with
421:saliency-enhanced Transformer.
422:The modality and granularity discrepancy results in different regions of interest. To preserve the saliency consistency across modalities
423:and granularities, we propose a saliency-enhanced Transformer including saliency-consist attention (SCA) component different with the
424:general Transformer. So we elaborate on its detailed attention design
455:In this way, the patch-level features 𝑋 𝑝𝑎𝑡𝑐 ℎ of visible modality have
464:granularity.
473:image into stripes and perform the pixel-level Transformer on each
521:the distributions of visible modality and near-infrared modality.
526:and 𝐿𝑚𝑎 is proposed to reduce the difference between modalities and
568:Thus, the output 𝑂 of saliency-enhanced Transformer (ST) is defined
574:We use saliency-enhanced Transformer to fuse inter-modality intragranularity and intra-modality inter-granularity features as:
601:inter-modality intra-granularity fusion intra-modality inter-granularity fusion
670:LayerNorm in Transformer for simplicity.
705:where 𝑁 𝑣𝑖𝑠 and 𝑁 𝑛𝑖𝑟 are the set of visible modality images and near-
713:infrared modality images.
714:The neighbor consistency loss 𝐿𝑛𝑐 is proposed to guarantee that the
891:from four visible cameras. In indoor-search mode, the gallery images are
892:captured by two visible light cameras.
894:visible and one infrared camera, which contains 8240 images of 412
895:identities. Each identity consists of 10 visible images and 10 infrared
896:images. The training set and testing set contain 2060 visible images
897:and 2060 infrared images of 206 identities. It contains two evaluation
898:settings (visible-search-infrared (V - I) and infrared-search-visible (I V)) [4,5]. The evaluation process is repeated 10 times with random
900:experimental results.
905:The training process of our approach is shown in Algorithm 1.
915:Algorithm 1 The training procedure of our approach
947:All experiments are deployed on an NVIDIA GeForce 3090 GPU, and
948:the implementation of our approach is based on the Pytorch framework.
949:Following the existing VI-ReID methods [4,5], the input images are
951:is used to optimize our network. In all experiments, we set the initial
955:4. Experiments
959:4.1. Datasets and experimental settings
963:parameters. In Tables 1 and 2 we conduct experiments on SYSU-MM01
974:SYSU-MM01 dataset [10] is the largest public VI-ReID dataset, captured by four visible cameras and two near-infrared cameras, including
977:composed of 22,258 visible images and 11,909 infrared images. The
979:infrared images for the query and 301 visible images for the gallery. In
1474:𝑠2 yields optimal results for our approach, indicating the network’s capability to capture features specific to each modality and those shared
1477:SYSU-MM01 and RegDB datasets. In Table 5, experimental results show
1480:model is able to learn from a wider perspective, and the method not
1490:optimal experimental performance is achieved when the patch sizes
1494:mechanism in Transformer that can effectively capture rich pedestrian
1501:our approach, we conduct ablation experiments on the SYSU-MM01
1507:improvement shows that compared to using single-granularity features,
1511:different numbers of stripes and the depth in the Transformer on the
1845:Evaluation of each granularity on the SYSU-MM01 dataset.
1948:multi-granularity features. Comparing Index 2 with Index 3, these improved results prove that fusing features between multiple granularities
1952:our approach includes MGFE, MGFT, and CNC, the performance yields
1953:69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
1960:in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
1962:between multi-granularity features can reduce the irrelevant matching and ambiguous alignment problems caused by single-granularity
1967:granularity for testing is lower than that of using multiple granularity fusion. This phenomenon can be attributed to the fact that each
1968:granularity only contains partial discriminative information related to
1973:By fusing these multi-granularity features, our approach can more
1976:4.3.2. Effectiveness of three-granularity features
1984:than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity

codex
后三篇的定位也清楚了：扩散补全那篇把遮挡说成特征层污染并加检索后处理，HCCL 那篇把无监督跨模态的核心问题改成伪标签可靠性传播，CM2GT 那篇把可见红外错配说成单粒度对齐失败。下面我读它们的方法细节和关键证据段。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | nl -ba | sed -n '348,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   348	Fig. 2: Methodology Overview: Image features Fo are extracted using a ViT encoder and processed through the pretrained
   349	DDO module to reconstruct clean feature embeddings Fclean . The downstream ReID module utilizes Fclean to compute a
   350	preliminary distance matrix, which the HRR module subsequently reranks to produce the final distance matrix for retrieval.
   351	
   352	formulation enables DDO to function as a plug-and-play
   353	III. M ETHODOLOGY
   354	module compatible with any ViT-based ReID architecture.
   355	Our approach models occluded and unoccluded features as
   356	Furthermore, unlike prior centroid-based methods [54], [55]
   357	that use centroids for direct retrieval, HRR formulates a novel distinct but correlated representations in the latent space. Rather
   358	graph-theoretic framework where centroids perform topology than adapting the ReID model to tolerate occlusion-induced
   359	pruning to construct high-confidence subgraphs before local k- distortions, we explicitly learn the transformation from clean
   360	reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
   361	guarantees denser reciprocal neighborhoods with reduced false reconstruction of identity-preserving embeddings by inpainting
   362	positives (Equations 15–16), addressing the noise amplification the corrupted regions back to their original, informative form.
   363	Our person re-identification pipeline, schematically explained
   364	problem inherent in standard re-ranking when gallery sets
   365	exhibit high intra-class variance. Together, these contributions in Figure 2, directly addresses partial occlusions using a Latent
   366	yield consistent state-of-the-art performance across all evaluated Diffusion guided De-Occluder (DDO). Given an occluded
   367	benchmarks, validating the effectiveness of explicitly removing feature embedding Fo extracted by the ViT encoder, the DDO
   368	occlusion corruption rather than adapting models to tolerate it. module learns a mapping Fclean = DDO(Fo ) where Fclean
   369	approximates the clean, unoccluded embedding. This mapping
   370	is optimized via denoising objectives and a reconstruction loss
   371	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
   372	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   373	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   374	
   375	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   376	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3681512
   377	
   378	4
   379	
   380	over latent space, improving robustness without altering the and the reverse process iteratively denoises zt through T steps
   381	backbone architecture, making DDO a plug-and-play module. using a trainable noise predictor ϵθ . After T timesteps, this
   382	For retrieval, we formalize our Hybrid centroid-based yields a denoised latent z0 ≈ Dθ (zt ), which is decoded to
   383	Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
   384	The original clean features F from a frozen ViT encoder
   385	identity
   386	P clusters Gi , we compute centroid embeddings Ci =
   387	1
   388	act as supervision. The reconstruction loss is defined as
   389	g∈Gi g and leverage these centroids to filter candidates
   390	|Gi |
   391	for k-reciprocal re-ranking, rather than relying solely on noisy
   392	Lrecon = λ∥Fclean − F∥22 + (1 − λ) (1 − cos(Fclean , F)) , (2)
   393	pairwise distances over all gallery samples. This aggregated
   394	filtering reduces intra-class variance, after which localized re- where λ balances MSE and cosine similarity, ensuring semantic
   395	ranking is applied to refine top matches. Together, DDO and fidelity.
   396	HRR form a compact, effective system achieving superior
   397	The standard denoising objective of a diffusion model is
   398	performance over prior approaches by explicitly modeling oc- given by:
   399	h
   400	i
   401	2
   402	clusion correction and leveraging robust identity-level filtering.
   403	Ldiff = Ez,ϵ,t ϵ − ϵθ (zt , t) 2 ,
   404	(3)
   405	A. Feature Inpainting via Diffusion guided De-Occluder (DDO)
   406	
   407	where ϵθ is the noise predictor parameterized by θ and ᾱt
   408	is the cumulative product of the noise schedule coefficients.
   409	To ensure semantic alignment with clean identity features, we
   410	augment this with the reconstruction loss from Eq. 2. The final
   411	DDO training loss is thus:
   412	
   413	Prior approaches make ReID models resistant to occlusion
   414	through pose priors, semantic parsing, or data augmentation,
   415	but do not remove the underlying feature-space corruption. We
   416	address this by formulating occlusion as a feature inpainting
   417	LDDO =
   418	Ldiff
   419	+
   420	γLrecon
   421	problem that explicitly recovers clean, identity-preserving
   422	|{z}
   423	| {z }
   424	(4)
   425	embeddings from occluded counterparts.
   426	diffusion denoising
   427	id-preserving supervision
   428	We conceptualize occlusion as a structured distortion in
   429	where γ controls the guidance strength of the reconstruction
   430	feature space, where clean identity-preserving features, denoted term.
   431	by Fclean , are transformed into occlusion-corrupted counterparts
   432	
   433	
   434	Fo through an unknown mapping T . This notion is illustrated
   435	F̂
   436	=
   437	D
   438	(F
   439	),
   440	θ
   441	=
   442	arg
   443	min
   444	L
   445	(5)
   446	clean
   447	θ
   448	o
   449	DDO
   450	in Figure 1, where each colored dot represents a clean feature
   451	θ
   452	embedding Fclean , and the corresponding x mark within the
   453	This formulation establishes DDO as the operational realizadashed circle depicts its occluded version Fo = T (Fclean ).
   454	tion
   455	of T −1 (Equation 1), unifying diffusion-based denoising
   456	Rather than adapting the ReID system to tolerate such distortions, we propose to learn the inverse transformation T −1 that with feature-level reconstruction. By explicitly learning to
   457	inpaint corrupted embeddings, DDO provides downstream
   458	maps corrupted features back to their clean form:
   459	ReID modules with clean, occlusion-free priors, thereby
   460	F̂clean = T −1 (Fo ),
   461	(1) improving retrieval robustness.
   462	While DDO is built upon the latent diffusion framework, its
   463	where F̂clean is the reconstructed embedding used for downtraining paradigm differs fundamentally from unsupervised
   464	stream retrieval.
   465	DDPM [9] and DSM [10]. Standard DDPM reverses the
   466	This perspective aligns naturally with the theory of diffusion
   467	forward diffusion process by predicting noise, requiring only
   468	models, where clean data x0 is gradually corrupted by noise
   469	the clean distribution q(x0 ); likewise, DSM learns the score
   470	via a forward process q(xt | x0 ), and a neural network is
   471	function from noisy observations in an unsupervised manner.
   472	trained to approximate the reverse process pθ (xt−1 | xt ) to
   473	In contrast, DDO leverages explicit paired supervision. Clean
   474	recover x0 . Specifically:
   475	features F from a frozen ViT encoder serve as ground-truth for
   476	x0 ≈ pθ (x0 | xT ) = Dθ (xT ), xT ∼ N (0, I).
   477	occluded features Fo . The combined loss LDDO (Equation 4)
   478	integrates the standard diffusion denoising objective with
   479	Analogously, occlusion-induced corruption is treated as a
   480	identity-preserving reconstruction. We interpret this formulation
   481	deterministic analogue of this stochastic process, and T −1
   482	as a conditional denoising diffusion model where Lrecon acts as
   483	is learned to "inpaint" missing semantic content in the latent
   484	a guidance term steering the reverse process toward identityfeature space, thereby supplying downstream modules with
   485	consistent reconstructions, analogous to classifier guidance [11]
   486	clean priors rather than requiring them to be occlusion-aware.
   487	but implemented via direct supervision. This yields a deterTo realize this concept, we introduce the Latent Diffuministic reconstruction operator D(Fo ) ≈ T −1 (Fo ) rather than
   488	sion guided De-Occluder (DDO), a generative module that
   489	a stochastic sampler, which is essential for ReID where we
   490	reconstructs clean features from Fo in the latent space. As
   491	must recover a specific clean embedding to preserve identity.
   492	illustrated in Figure 3, Fo is first projected into a compact
   493	This supervised paradigm distinguishes DDO from generic
   494	latent space z via a convolutional encoder. The encoder maps
   495	applications of diffusion models and enables task-specific
   496	ViT features of shape (768 × 14 × 14) into a 256-channel
   497	feature inpainting. To highlight the distinct nature of our task
   498	latent representation, while the decoder mirrors this structure
   499	specific Latent Diffusion Model (LDM) from standard diffusion
   500	to reconstruct the denoised embedding. A forward diffusion
   501	frameworks, we compare the compute requirements of our
   502	process then corrupts z with Gaussian noise ϵ ∼ N (0, I):
   503	entire ReID system with DDO to DDPM [9] and DSM [56]
   504	√
   505	√
   506	zt = ᾱt z + 1 − ᾱt ϵ,
   507	in Table II. We train the model using synthetically generated
   508	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
   509	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   510	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   511	
   512	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   513	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3681512
   514	
   515	5
   516	
   517	)
   518	
   519	Encoder
   520	
   521	ViT
   522	Encoder
   523	
   524	εₜ◌̂
   525	Add Noise
   526	(Forward Diffusion):
   527	
   528	ε̂ₜ
   529	
   530	Linear Noise Schedule
   531	
   532	ViT
   533	Encoder
   534	Occluded
   535	Image
   536	
   537	Loss function:
   538	: Frozen parameters
   539	Cleaned
   540	Feature Map
   541	
   542	Unoccluded
   543	Feature Map (F)
   544	
   545	Unoccluded
   546	Image
   547	
   548	‘T’ Timesteps
   549	
   550	Sample Noise:
   551	(0, I)
   552	
   553	Decoder
   554	
   555	Reverse
   556	Diffusion
   557	
   558	DDO
   559	
   560	Occluded
   561	Feature Map
   562	
   563	: Trainable parameters
   564	)
   565	
   566	Fig. 3: Latent Diffusion guided De-Occluder (DDO) Module Training. The ViT encoders are kept frozen throughout training
   567	to ensure that the latent diffusion model focuses exclusively on the denoising task without altering the input representations.
   568	The model is optimized using the standard denoising objective (See Equation 3) guided by the reconstruction loss Lrecon (see
   569	Equation 2), where the unoccluded feature map F serves as the ground-truth. This combined training objective (See Equation 4)
   570	enables DDO to effectively learn to reconstruct clean and identity-preserving feature maps from their occluded counterparts.
   571	
   572	occluded images constructed via silhouette-guided background
   573	patching and Fourier domain adaptation [57] (Full details in
   574	Supplementary Material Section I).
   575	TABLE II: Comparison of DDO efficiency with standard
   576	diffusion models. DDO is designed for feature-space inpainting,
   577	while standard models operate on high-resolution images.
   578	Model
   579	
   580	Params (M) FLOPs (G) Dimension
   581	
   582	Stable Diffusion [8]
   583	DDPM [9]
   584	DSM [56]
   585	Baseline + DDO (Ours)
   586	
   587	1070+
   588	431
   589	300
   590	188
   591	
   592	520
   593	430
   594	450
   595	24.3
   596	
   597	4×64×64
   598	512×512×3
   599	512×512×3
   600	768×14×14
   601	
   602	B. ReID Architecture
   603	Our person ReID model is built upon a ViT backbone. We
   604	incorporate a multi-branch local feature extraction, a GELT
   605	module for global-local fusion, and part-based classification
   606	heads. We integrate it with our pretrained frozen DDO and
   607	train it using a contrastive loss (Modified Centroid Triplet Loss)
   608	[13] and a classification loss (Cross-Entropy Loss). The model
   609	is designed to extract robust feature representations even under
   610	occluded conditions and to separate identity features effectively
   611	in the embedding space.
   612	Given an input image x ∈ R3×224×224 , we first extract
   613	features using a pretrained Vision Transformer (ViT) encoder
   614	E. Simultaneously, the image is passed through DDO that has
   615	been trained to reconstruct clean embeddings from occluded
   616	features (see Section III-A). The Clean Feature Map Fclean
   617	and the Occluded Feature Map Fo are fused via a multi-head
   618	cross-attention module:
   619	Q = Fclean WQ , K = Fo WK ,
   620	
   621	
   622	QK⊤
   623	Fattn = Softmax √
   624	V,
   625	dk
   626	
   627	V = Fo WV ,
   628	
   629	(6)
   630	(7)
   631	
   632	where WQ , WK , and WV are learnable projection matrices,
   633	and dk is the dimensionality of the keys.
   634	The Fused Feature Map Fattn is re-tokenized and passed
   635	through the final normalization layer of the ViT encoder to
   636	obtain refined token embeddings. From this, both the [CLS]
   637	token and spatial tokens are extracted. The spatial tokens are
   638	reshaped into a 2D spatial map S ∈ R768×14×14 and processed
   639	using horizontal part-based pooling across p equal parts. To
   640	enhance local-global interaction, we use a Global-Enhanced
   641	Local Token (GELT) fusion strategy [58]. For each part feature
   642	si , we incorporate the [CLS] token c as:
   643	c + α · si
   644	s̃i =
   645	,
   646	(8)
   647	1+α
   648	where α is a learnable or fixed fusion coefficient. This improves
   649	the semantic expressiveness of local part features by injecting
   650	global context.
   651	Each part i is classified using a separate classification head
   652	Ci :
   653	ŷi = Ci (s̃i ), for i = 1, 2, . . . , p,
   654	(9)
   655	where s̃i ∈ R768 is the pooled embedding of part i.
   656	a) Loss Function: To train our model, we employ a
   657	modified triplet loss formulation [55], where intra-class and
   658	inter-class distances are weighted differently. The key idea is
   659	to prioritize the separation of negative samples by assigning a
   660	higher weight to the inter-class distance term.
   661	Let fa , fp , and fn denote the anchor, the closest positive
   662	centroid, and the farthest negative centroid, respectively. The
   663	modified triplet loss is defined as:
   664	
   665	Lmtriplet = max 0, w1 · ∥fa − fp ∥22 − w2 · ∥fa − fn ∥22 + α ,
   666	(10)
   667	where w1 < w2 are weighting coefficients for intra-class and
   668	inter-class distances, respectively, and α is the margin. This
   669	asymmetric weighting encourages the model to emphasize
   670	
   671	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
   672	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   673	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   674	
   675	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   676	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3681512
   677	
   678	6
   679	
   680	Multi Headed Cross
   681	Attention
   682	Q
   683	KV
   684	ViT
   685	Encoder
   686	
   687	..
   688	
   689	Q
   690	KV
   691	
   692	Tokenization and
   693	Normalization
   694	
   695	DDO
   696	
   697	Occluded
   698	Feature Map
   699	
   700	Cleaned Feature
   701	Map
   702	)
   703	
   704	)
   705	
   706	Fused Feature
   707	Map
   708	)
   709	
   710	Modified Centroid Triplet Loss
   711	
   712	Globally
   713	Enhanced
   714	Local
   715	Avg
   716	Tokens
   717	(GELT) pool
   718	Generation
   719	
   720	Averaged
   721	GELT
   722	
   723	Cross-Entropy
   724	Loss
   725	FC Classifier 1
   726	FC Classifier 2
   727	FC Classifier 3
   728	FC Classifier 4
   729	FC Classifier 13
   730	FC Classifier 14
   731	
   732	Classifier Heads
   733	(
   734	
   735	Fig. 4: Overview of our ReID architecture: The DDO module parameters (trained separately on a synthetically occluded
   736	dataset (Supplementary Section I)) are kept frozen during ReID training. The two loss functions used to train the ReID model
   737	are: 1) The Modified Centroid Triplet Loss on the Fused Feature Map (Fattn ) (Equation 10) and 2) Cross-entropy loss on the
   738	outputs of each of the p classifier heads (Equation 11).
   739	
   740	pushing negative samples further apart in the feature space
   741	rather than excessively collapsing similar embeddings.
   742	For classification, we use standard Cross-Entropy loss on
   743	the outputs of the Classifier heads (Ci ) defined as:
   744	
   745	
   746	C
   747	X
   748	LCE (ŷi , y) = −ŷi,y + log 
   749	eŷi,j 
   750	
   751	(11)
   752	
   753	j=1
   754	
   755	The final training objective combines the classification loss
   756	over all p parts with the modified triplet loss:
   757	Ltotal =
   758	
   759	p
   760	X
   761	
   762	LCE (ŷi , y) + Lmtriplet ,
   763	
   764	(12)
   765	
   766	i=1
   767	
   768	This formulation leads to a more structured and robust
   769	feature space, enabling improved discrimination, especially
   770	under occluded and cluttered visual conditions.
   771	
   772	C. Hybrid Re-Ranking (HRR)
   773	Standard re-ranking methods [51] compute pairwise distances
   774	across the entire gallery, incurring high computational costs
   775	and leaving Jaccard distances vulnerable to outlier noise.
   776	We propose Hybrid Re-Ranking (HRR), combining robust
   777	centroid-based anchors [54] with local manifold refinement.
   778	By restricting re-ranking strictly to semantically relevant
   779	identity clusters, HRR simultaneously reduces computational
   780	overhead and actively suppresses outlier-induced corruption.
   781	This centroid-guided pruning eliminates irrelevant gallery
   782	variations, yielding a denser, structurally stronger retrieval
   783	graph for robust matching. This method is explained in Figure
   784	5. Each colored cluster represents gallery embeddings from
   785	a distinct identity, with triangle markers denoting identity
   786	centroids. The query embedding (black cross) is compared
   787	against all centroids using Euclidean distance, and only the top3 nearest identities (within the circular boundary) are retained
   788	for re-ranking. Lines between query and centroids are labeled
   789	with their respective distances.
   790	
   791	Fig. 5: Illustration of centroid-guided gallery filtering for
   792	Hybrid Re-Ranking (HRR).
   793	This filtering removes noisy or irrelevant identities and
   794	leads to more compact, semantically aligned neighborhoods
   795	for Jaccard-based re-ranking.
   796	1) Theoretical motivation for Centroid filtering and HRR:
   797	In this section, we provide a theoretical analysis demonstrating
   798	why HRR yields a superior graph structure for retrieval
   799	compared to standard re-ranking approaches.
   800	Let Nq denote the reciprocal neighborhood of query q.
   801	Hypothesis: Eg∈Nqhybrid [w(q, g)] > Eg∈Nqstandard [w(q, g)]
   802	where w(q, g) is the similarity score.
   803	Proof: Figure 5 provides an intuitive visualization of our
   804	centroid filtering scheme. Each colored cluster corresponds to
   805	the feature embeddings of a single gallery identity, and the
   806	black triangle denotes the centroid µc of that identity, computed
   807	as the mean of its sample embeddings. The black cross (X)
   808	marks the query image q. The Euclidean distances between
   809	the query and all centroids ∥q − µc ∥2 are drawn as annotated
   810	lines. We select the top-n identities with the lowest centroid
   811	distances to construct the filtered identity set Cq .
   812	Geometrically, this procedure restricts re-ranking to identity
   813	regions lying within a compact ellipsoidal neighborhood
   814	centered at q. This filtering avoids distant or noisy identities
   815	
   816	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
   817	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   818	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   819	
   820	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   821	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3681512
   822	
   823	7
   824	
   825	and focuses the refinement process on semantically relevant
   826	gallery clusters. Mathematically, it defines a pruned subgraph
   827	Gq where the node set is:
   828	Vq = {gi ∈ G | yi ∈ Cq }
   829	
   830	(13)
   831	
   832	Standard re-ranking computes the Jaccard distance dJ (q, g)
   833	over k-reciprocal neighbors:
   834	dJ (q, g) = 1 −
   835	
   836	|Rq ∩ Rg |
   837	|Rq ∪ Rg |
   838	
   839	(14)
   840	
   841	where Rq and Rg are the sets of k-reciprocal neighbors for
   842	q and g, respectively. In traditional settings, these sets are
   843	constructed over the full gallery G, resulting in high overlap
   844	with unrelated identities. This leads to inflated denominators
   845	and poor alignment of reciprocal sets.
   846	Numerical illustration. From Figure 5, the query q lies at
   847	(2, 2) and the centroids (triangles) of six identities are located
   848	around various positions. Let us denote three closest centroids
   849	(yellow, blue and green) as identities A, B, C. Their distances
   850	from q are:
   851	∥q − µA ∥2 = 0.83,
   852	
   853	∥q − µB ∥2 = 0.53,
   854	
   855	∥q − µC ∥2 = 0.88
   856	
   857	We construct Cq = {A, B, C} and restrict re-ranking to gallery
   858	samples with labels in Cq .
   859	Assume:
   860	Rstandard
   861	= {A1 , B1 , C1 , D1 , E1 },
   862	q
   863	
   864	Rstandard
   865	= {B1 , C1 , F1 }
   866	g
   867	
   868	then:
   869	
   870	2
   871	= 0.666
   872	6
   873	In HRR, since only identities A, B, C are retained:
   874	dstandard
   875	(q, g) = 1 −
   876	J
   877	
   878	Rhybrid
   879	= {A1 , B1 , C1 },
   880	q
   881	then:
   882	dhybrid
   883	(q, g) = 1 −
   884	J
   885	
   886	|Rhybrid
   887	∪ Rhybrid
   888	|
   889	q
   890	g
   891	
   892	>
   893	
   894	(18)
   895	
   896	where dorig is the original distance (e.g., Euclidean or cosine),
   897	dJ is the Jaccard distance, and λr balances the two.
   898	Here, dJ (q, g) is the Jaccard distance computed over the
   899	k-reciprocal neighbors in Gq .
   900	Re-ranking can be interpreted as transductive inference over a
   901	graph where nodes represent samples and edges are similarities.
   902	In standard re-ranking, the graph includes all gallery samples,
   903	leading to cluttered and noisy neighborhoods. In HRR, we
   904	prune the graph by restricting it to samples from centroidmatched identities.
   905	
   906	(15)
   907	
   908	Thus, the pruned Jaccard distance is lower and reflects better
   909	alignment.
   910	This implies a more meaningful graph structure Ghybrid
   911	=
   912	q
   913	(Vq , Eq ) with enhanced similarity alignment.
   914	The
   915	resulting
   916	
   917	edge weights: w(q, g) = exp −∥q − g∥22
   918	are higher on average in the hybrid subgraph:
   919	Eg∈Nqhybrid [w(q, g)] > Eg∈Nqstandard [w(q, g)]
   920	
   921	drerank (q, g) = (1 − λr ) dorig (q, g) + λr · dJ (q, g)
   922	
   923	We evaluate our method on eleven standard and occlusionfocused short term and cloth changing ReID benchmark
   924	datasets (Table III), consistently outperforming state-of-theart approaches. These results validate the effectiveness of our
   925	latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
   926	challenging conditions.
   927	
   928	2
   929	= 0.333
   930	3
   931	
   932	|Rstandard
   933	∩ Rstandard
   934	|
   935	q
   936	g
   937	standard |
   938	|Rstandard
   939	∪
   940	R
   941	q
   942	g
   943	
   944	where Top-N denotes the operator that ranks all gallery
   945	centroids based on their similarity to the query and selects the
   946	N most similar centroids for subsequent retrieval.
   947	Stage 2: Local Re-ranking within Filtered Gallery:
   948	From C q , we define the subset of gallery samples G q = {gi ∈
   949	G | yi ∈ C q }, which contains only the samples belonging
   950	to the top-selected identities. We then compute the re-ranked
   951	distance between the query and each gallery sample as:
   952	
   953	IV. E XPERIMENTS
   954	
   955	Rhybrid
   956	= {B1 , C1 }
   957	g
   958	
   959	This proves:
   960	|Rhybrid
   961	∩ Rhybrid
   962	|
   963	q
   964	g
   965	
   966	Stage 1: Centroid-based Identity Filtering: Let the
   967	gallery features be denoted as G = {g1 , g2 , . . . , gGL } with
   968	identity labels yg ∈ {1, . . . , C} (GL = Total number of
   969	gallery samples).
   970	For each identity c, we compute the centroid
   971	P
   972	µc = |G1c | gi ∈G c gi , where G c is the set of gallery samples
   973	for identity c. We then select the top-n centroids most similar
   974	to the query q using cosine similarity:
   975	
   976	
   977	C
   978	C q = Top-N {cos(q, µc )}c=1
   979	(17)
   980	
   981	(16)
   982	
   983	Thus, HRR increases the intra-cluster cohesion and reduces
   984	inter-identity noise, yielding a more robust re-ranking graph and
   985	improved retrieval accuracy. The hybrid graph yields denser and
   986	more semantically meaningful subgraphs, improving re-ranking
   987	robustness. To our knowledge, no prior work has integrated
   988	identity-level centroids into the graph structure of k-reciprocal
   989	re-ranking in this manner.
   990	The HRR algorithm has two stages as enumerated below.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf' - | nl -ba | sed -n '293,908p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   293	In this section, the proposed Hierarchical Centrality Collaborative Learning (HCCL) is explained in detail. First, we
   294	present the pipeline of HCCL in Section III-A. Subsequently,
   295	we introduce the proposed Closeness Centrality Selection
   296	(CCS) in Section III-B. Finally, we present the Hierarchical
   297	Consistency Matching (HCM) in Section III-C.
   298	A. Overview
   299	Given a VI-ReID dataset D = {X v , X r }, where X v = {xiv |i =
   300	1, 2, . . . , Nv } and X r = {xir |i = 1, 2, . . . , Nr } denote the visible
   301	and infrared sets with Nv and Nr images, respectively. The
   302	purpose of US-VI-ReID is to learn the modality-invariant
   303	and identity-discriminative representations without using any
   304	annotated labels.
   305	The pipeline of the proposed HCCL is shown in Fig. 2,
   306	we utilize a two-stream encoder to extract pedestrian features
   307	
   308	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
   309	
   310	9826
   311	
   312	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025
   313	
   314	Fig. 2. Illustration of our proposed HCCL. (a) Clustering stage. HCCL generates intra-modality pseudo-labels by DBSCAN, consistent with [22]. Different
   315	shapes denote different identities. (b) is our proposed Closeness Centrality Selection (CCS), detailed in Sec. III-B. The darker the color of the instance, the
   316	higher its confidence level, and vice versa. Instances with confidence levels below the predefined positional threshold are identified as noisy samples. LRHCL
   317	is our proposed reliable homogeneous contrastive loss. (c) is our proposed Hierarchical Consistency Matching (HCM), and will be described in Sec. III-C.
   318	LDRHL denotes our proposed discriminative reliable heterogeneous loss.
   319	
   320	of two modalities, obtaining F v = { fiv |i = 1, 2, . . . , Nv } and
   321	F r = { fir |i = 1, 2, . . . , Nr }. Then, following existing works [22],
   322	[24], [25], [27], the corresponding pseudo-labels Y v = {yvi |i =
   323	1, 2, . . . , Pv } and Y r = {yri |i = 1, 2, . . . , Pr } are generated by
   324	using DBSCAN [38], where Pv and Pr are the total number
   325	of clusters in the visible and infrared modalities. To mitigate
   326	the inherent noise in the generated pseudo-labels, we propose
   327	the Closeness Centrality Selection (CCS), which quantifies
   328	and identifies reliable samples within each cluster to serve
   329	as reliable prototypes. Further, we propose the Hierarchical
   330	Consistency Matching (HCM) to establish cross-modality correspondences by adopting a novel local-to-global hierarchical
   331	paradigm.
   332	B. Closeness Centrality Selection
   333	DBSCAN is a commonly used clustering algorithm for
   334	recent approaches in US-VI-ReID [22], [24], [25], [26], [27].
   335	It groups points in a data set that are closely packed together
   336	based on their distance to other points. However, inherent
   337	intra-modality variations may lead to two problems: features
   338	of different pedestrians showing similar densities, or features of the same pedestrian exhibiting varied densities. In
   339	such conditions, feature representation will lead to incorrect
   340	pseudo-label assignments, namely noisy labels. Moreover, subsequent cross-modality matching directly relies on these noisy
   341	labels, leading to the establishment of incorrect cross-modality
   342	
   343	correspondences. To mitigate the issue of noisy labels, we
   344	propose a Closeness Centrality Selection (CCS) to proactively quantify sample confidence during forward propagation,
   345	selecting high-confidence samples as reliable modality-specific
   346	memories for homogeneous contrastive learning. The CCS
   347	is described as a two-step process, comprising Closeness
   348	Centrality Exploration (CCE) and Reliable Homogeneous
   349	Contrastive Learning (RHCL). The detailed architecture is
   350	shown in Fig. 2 (b).
   351	1) Closeness Centrality Exploration: In CCE, each cluster (after DBSCAN) is modeled as a connected weighted
   352	undirected graph, where the vertices represent the individual
   353	features within the cluster, and the edges denote the simiv
   354	larity between these features. Specifically, let Gvc = { fc,i
   355	|i =
   356	v
   357	1, 2, . . . kc } denote the c-th cluster of visible modality with kcv
   358	vertices. Similarly, Grc = { fc,r j | j = 1, 2, . . . kcr } is the c-th cluster
   359	in the infrared modality with kcr vertices.
   360	For clear understanding, taking the visible modality as an
   361	example, we omit the similar processing for the infrared
   362	modality. For the c-th cluster, we first compute the similarity
   363	for the i-th feature within the Gvc as follows:
   364	v
   365	v
   366	v
   367	v
   368	v
   369	v
   370	v
   371	S c,i
   372	= [s( fc,i
   373	, fc,1
   374	), s( fc,i
   375	, fc,2
   376	), . . . , s( fc,i
   377	, fc,k
   378	v )],
   379	c
   380	
   381	(1)
   382	
   383	v
   384	where fc,i
   385	and fc,v j denote the i-th and j-th features within the cv
   386	th cluster, respectively. S c,i
   387	denotes the similarity between the
   388	i-th feature and the remaining features within the c-th cluster.
   389	
   390	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
   391	
   392	JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
   393	
   394	v
   395	The similarity between two features fc,i
   396	and fc,v j is calculated
   397	as follows:
   398	v
   399	fc,i
   400	· fc,v j
   401	v
   402	s( fc,i
   403	, fc,v j ) = v
   404	, { f v , f v } ∈ Gvc .
   405	(2)
   406	|| fc,i ||2 || fc,v j ||2 c,i c, j
   407	
   408	For each feature in the c-th cluster, we compute its similarity via Eq. (1). The similarity of all features within the
   409	cluster is concatenated to obtain the similarity matrix S cv ,
   410	v
   411	v
   412	v
   413	T
   414	v
   415	i.e., S cv = [S c,1
   416	, S c,2
   417	, . . . , S c,k
   418	v ] . The matrix S c represents the
   419	c
   420	cosine similarity between feature pairs, which reflects their
   421	positive correlation. To transform the similarity into a measure
   422	of dissimilarity that reflects the negative correlation between
   423	v
   424	v
   425	features, the distance matrix Dvc ∈ Rkc ×kc is computed as:
   426	Dvc = 1 − S cv .
   427	
   428	(3)
   429	
   430	Based on the Dvc , we further compute the feature score Avc,i
   431	which is defined as the inverse of the average shortest path
   432	distance [52], [53] from the node to all other nodes within the
   433	cluster. It is formulated as:
   434	n
   435	X
   436	1
   437	v
   438	,
   439	(4)
   440	Ac,i =
   441	v
   442	D
   443	(
   444	j)
   445	+ D̄vc,i
   446	c,i
   447	j=1
   448	where Dvc,i denotes the i-th row of the distance matrix Dvc ,
   449	corresponding to the i-th vertex within graph Gvc . We sort
   450	the distances in ascending order and select the top n smallest
   451	distance, denoted as Dtc,i ( j)( j = 1, 2, . . . n), where j indicates
   452	v
   453	the j-th smallest distance between fc,i
   454	and its nearest neighbors
   455	v
   456	v
   457	within the same cluster Gc . D̄c,i denotes the mean value of
   458	the i-th node within the c-th graph Gvc . The higher the score,
   459	the shorter the average distance to other nodes in the cluster,
   460	indicating stronger associations with other nodes in the same
   461	cluster, which are referred to as reliable samples.
   462	Further, the score of a cluster Avc is obtained by concatenating the scores calculated from all features within the cluster:
   463	Avc = [Avc,1 , Avc,2 , . . . , Avc,kcv ].
   464	
   465	(5)
   466	
   467	Analogously, Arl represents the corresponding score of the
   468	l-th cluster in the infrared modality.
   469	In general, the proposed CCE quantifies the reliability
   470	of the node within the cluster and mines reliable samples,
   471	thereby establishing a foundation for the subsequent reliable
   472	homogeneous contrastive learning.
   473	2) Reliable Homogeneous Contrastive Learning: Instead of
   474	directly averaging all features within a cluster and storing them
   475	in memory as done in existing methods, we store each cluster’s
   476	reliable homogeneous representation, based on the CCE, in
   477	the visible and infrared memory. Mv = {mv1 , mv2 , . . ., mvPv } and
   478	v
   479	r
   480	Mr = {mr1 , mr2 , . . ., mrPr } (Mv/r ∈ Rd×P /P where d is the
   481	feature dimension) denote the reliable memory in the visible
   482	and infrared modality, respectively. We treat the memory
   483	center as a reliable prototype, which represents the mean
   484	feature of all reliable samples within the same cluster, serving
   485	as a compact and robust cluster-level representation as follows:
   486	1 X v
   487	mvc = v
   488	f ,
   489	(6)
   490	|Ωc | v v c,i
   491	fc,i ∈Ωc
   492	
   493	mrl =
   494	
   495	1 X r
   496	f ,
   497	|Ωrl | r r l, j
   498	fl, j ∈Ωl
   499	
   500	(7)
   501	
   502	9827
   503	
   504	where mvc represents the c-th prototype stored in the visible
   505	memory while mrl corresponds to the l-th prototype stored in
   506	the infrared memory. Ωvc denotes the set of reliable homogeneous instances in the c-th cluster of the visible modality,
   507	which can be obtained by:
   508	ˇ ˇ
   509	v
   510	(8)
   511	Ωvc = { fc,i
   512	|Rank(Avc ) ≤ γ · ˇΩvc ˇ},
   513	ˇ
   514	ˇ
   515	rˇ
   516	r
   517	r
   518	r
   519	ˇ
   520	(9)
   521	Ω = { f |Rank(A ) ≤ γ · Ω },
   522	l
   523	
   524	l, j
   525	
   526	l
   527	
   528	l
   529	
   530	where Rank (·) is a rank function that refers to the ranking of
   531	the features within the set; γ represents the selection ratio. A
   532	certain proportion of features with high confidence are retained
   533	based on the ranking of feature scores Avc and Arl , while
   534	those that do not meet the predefined positional threshold are
   535	identified as noisy samples. They are directly discarded during
   536	forward propagation, ensuring that the reliable prototypes help
   537	mitigate the impact of noisy samples. This process allows for
   538	immediate feature adjustments without any delay, improving
   539	the consistency of the learned feature representations.
   540	During the training, we update visible and infrared memories by a momentum updating strategy:
   541	v
   542	mv(δ)
   543	← βmv(δ−1)
   544	+ (1 − β) fc,i
   545	,
   546	c
   547	c
   548	
   549	(10)
   550	
   551	mr(δ)
   552	← βmr(δ−1)
   553	+ (1 − β) fl,r j ,
   554	l
   555	l
   556	
   557	(11)
   558	
   559	v
   560	where fc,i
   561	and fl,r j are the features from the c-th and the l-th
   562	cluster, respectively. β denotes the momentum updating rate;
   563	δ indicates the number of training iterations.
   564	Based on the reliable memory Mv and Mr , we design a
   565	reliable homogeneous contrastive loss to minimize the distance
   566	between the given sample and the positive prototype for each
   567	modality, which is defined as:
   568	 
   569	NB
   570	exp( fiv · Mv yvi /τ)
   571	1 X
   572	v
   573	h i
   574	,
   575	(12)
   576	log P v
   577	LRHCL = −
   578	P
   579	NB
   580	exp( f v · Mv yv /τ)
   581	i=1
   582	
   583	j=1
   584	
   585	i
   586	
   587	j
   588	
   589	NB
   590	X
   591	
   592	 
   593	exp( fir · Mr yri /τ)
   594	1
   595	h i
   596	LrRHCL = −
   597	log P r
   598	,
   599	P
   600	NB
   601	r
   602	r yr /τ)
   603	exp(
   604	f
   605	·
   606	M
   607	i=1
   608	j=1
   609	i
   610	j
   611	
   612	(13)
   613	
   614	where NB denote
   615	the total
   616	
   617	  number of samples in a training
   618	batch. Mv yvi and Mr yri denote the corresponding positive
   619	reliable prototype of the given features fiv and fir stored in the
   620	memory. τ is a temperature hyper-parameter.
   621	The total loss of reliable homogeneous contrastive learning
   622	is formulated as:
   623	LRHCL = LvRHCL + LrRHCL .
   624	
   625	(14)
   626	
   627	LRHCL selects reliable samples within each cluster as prototypes for cluster representation, learning modality-specific
   628	information and optimizing the feature extractor. This process
   629	leverages backward propagation as a secondary safeguard to
   630	enhance robustness against noisy labels.
   631	C. Hierarchical Consistency Matching
   632	To establish cross-modality correspondences, existing methods [27], [28] construct a cost matrix by calculating Euclidean
   633	distances between cluster centroids of different modalities
   634	where the centroids are obtained by averaging all features
   635	
   636	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
   637	
   638	9828
   639	
   640	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025
   641	
   642	within each cluster. These approaches suffer from two major
   643	limitations: noisy labels and scale sensitivity. Specifically,
   644	the method of centroid computation inherently incorporates
   645	noisy labels into the cost matrix, increasing the difficulty
   646	of matching. Furthermore, the Euclidean distance is sensitive to scale variations, posing limitations in cross-modality
   647	similarity measurement. Therefore, we propose a Hierarchical
   648	Consistency Matching (HCM) to mine the underlying crossmodality correspondences by adopting a novel local-to-global
   649	hierarchical paradigm. For local matching, Bidirectional Consistency Local Matching (BCLM) is proposed to associate
   650	the local positive cross-modality identities through bidirectional consistency. For global matching, Global Prototype
   651	Reliability Alignment (GPRA) is designed to establish global
   652	cross-modality correspondences by integrating local matches.
   653	In addition, Discriminative Reliable Heterogeneous Learning
   654	(DRHL) is proposed to learn modality-invariant and discriminative features and resist the huge cross-modality discrepancy.
   655	The detailed architecture is shown in Fig. 2 (c).
   656	1) Bidirectional Consistency Local Matching: First, we
   657	select the instance with the highest closeness centrality score
   658	by the CCS as the centrality anchor instance, which is formulated as:
   659	v
   660	f˜cv = { fc,i
   661	| I(Avc,i > Avc, j ,
   662	
   663	∀i , j) = 1},
   664	
   665	(15)
   666	
   667	where Avc,i denotes the closeness centrality score of the
   668	i-th instance in c-th cluster from the visible modality, I(·) is
   669	an indicator function. Each centrality anchor instance serves
   670	as a robust representative of all features within the identity.
   671	Subsequently, the centrality anchor instances from each cluster
   672	are concatenated to obtain F̃ v = { f˜iv | i = 1, 2, . . . , Pv } for the
   673	visible modality and F̃ r = { f˜ir | i = 1, 2, . . . , Pr } for the infrared
   674	modality.
   675	Then we compute the similarity between the c-th centrality
   676	anchor instance f˜cv in the visible modality and all the anchor
   677	instances F̃ r from the infrared modality to identify the most
   678	relevant infrared instances for pairing, which is defined as:
   679	S̃ cv→r = [s( f˜cv , f˜1r ), s( f˜cv , f˜2r ), . . . , s( f˜cv , f˜Prr )],
   680	
   681	(16)
   682	
   683	where s( f˜cv , f˜lr ) denotes the cosine similarity between f˜cv and
   684	f˜lr . The top-k candidates Nk ( f˜cv→r ) from the infrared modality
   685	are identified, which is formulated as follows:
   686	Nk ( f˜cv→r ) = { f˜lr | l ∈ arg max S̃ cv→r }.
   687	l1 ,l2 ,...,lk
   688	
   689	(17)
   690	
   691	Existing methods [22], [26] establish cross-modality correspondences by using similarity metric, but they solely rely on
   692	unidirectional similarity which can result in incorrect heterogeneous neighbor selection and negative model optimization.
   693	As depicted wrong match in Fig. 2 (c), the blue samples within
   694	the red dashed box represent unidirectional matches from the
   695	visible to the infrared modality. In reality, the red circle and
   696	the blue triangle do not belong to the same class, indicating the
   697	issue of incorrect matching caused by unidirectional similarity.
   698	Therefore, we propose the bidirectional consistency to mitigate such errors, where the underlying rationale is that the
   699	images from different modalities belonging to the same class
   700	
   701	should be bidirectional neighbor for each other [54], [55], [56].
   702	This bidirectional correspondence captures the symmetry and
   703	consistency of cross-modality matching, serving as a strong
   704	prior for establishing reliable correspondences. Specifically,
   705	we traverse through all instances in Nk ( f˜cv→r ), and for each
   706	instance f˜lr within this set, we compute its corresponding
   707	visible candidates to obtain Nk ( f˜lr→v ). Finally, local crossmodality positive label pairs R are established when f˜cv satisfies
   708	the following criterion:
   709	k
   710	) | f˜cv ∈ Nk ( f˜lr→v )|ll=l
   711	},
   712	R = {(ŷcv→r , ŷr→v
   713	l
   714	1
   715	
   716	(18)
   717	
   718	where ŷv→r
   719	and ŷr→v
   720	are the pseudo-labels of f˜lr and f˜cv ,
   721	c
   722	l
   723	v
   724	v
   725	respectively. f˜c ∈ F̃ and f˜lr ∈ F̃ r denote the centrality
   726	anchor instances of the c-th and l-th clusters in the visible and
   727	infrared modalities, respectively. It ensures that f˜cv and f˜lr are
   728	mutually reliable counterparts, thereby establishing the correct
   729	local cross-modality correspondence. Moreover, to enhance
   730	computational efficiency, we terminate the traversal process for
   731	each centrality anchor instance once the first reliable positive
   732	cross-modality label is found.
   733	2) Global Prototype Reliability Alignment: Existing methods [25], [28], [29], [30], [49] construct the cost matrix by
   734	calculating the similarity between samples and incorporate
   735	various algorithms to globally mine cross-modality correspondences. However, the presence of noisy samples undermines
   736	the similarity measures between instances, leading to misleading associations in the cost matrix, thereby posing a significant
   737	challenge to establishing correct correspondences.
   738	To address the above issues, we try to utilize centrality
   739	anchor instance to represent the entire identity, enabling the
   740	establishment of local cross-modality correspondences through
   741	BCLM. However, the conditions for associating the label of
   742	two modalities in BCLM are too stringent, and due to large
   743	heterogeneous modality discrepancies, many features fail to
   744	meet these conditions. Inspired by the local-to-global idea,
   745	we leverage highly correlated local bidirectional matchings to
   746	guide the global alignment, designing a simple yet effective
   747	v
   748	r
   749	cost expression P ∈ RP ×P .
   750	Specifically, for feature pairs belonging to R, the matching
   751	cost at the indices corresponding to the visible cluster containing f˜cv and the infrared cluster containing f˜lr is set to zero.
   752	For other pairs, the cost is represented by the dissimilarity
   753	between reliable homogeneous prototypes. It can be defined
   754	as follows:
   755	8
   756	<0,
   757	if (ŷcv→r , ŷr→v
   758	) ∈ R,
   759	l
   760	1
   761	(19)
   762	Pc,l =
   763	r→v
   764	, if (ŷv→r
   765	) < R,
   766	:
   767	c , ŷl
   768	exp(s(mvc , mrl ))
   769	where each element Pc,l in P denotes the cost associated
   770	with matching the visible reliability prototype mvc and the
   771	infrared reliability prototype mrl . In this way, the cost matrix
   772	jointly incorporates both the cluster-level feature disparities
   773	and the instance-level local matches. Subsequently, following
   774	[29] and [49], the optimal transport algorithm is applied to
   775	establish global cross-modality correspondences between each
   776	visible cluster and its corresponding infrared cluster while
   777	
   778	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
   779	
   780	JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
   781	
   782	minimizing the total cost of the transportation plan Q as
   783	follows.
   784	minhQ, Pi + λH(Q),
   785	Q
   786	8
   787	ˆ
   788	<Q1 = 1 · 1 ,
   789	Pv
   790	s.t.
   791	1
   792	ˆ
   793	T
   794	:Q 1 = 1 · ,
   795	Pr
   796	v
   797	
   798	(20)
   799	
   800	r
   801	
   802	where Q ∈ RP ×P represents the transport plan for crossmodality prototype matching; Pv and Pr are the number
   803	of visible and infrared prototypes, respectively; h·i denotes
   804	the Frobenius dot-product. H(Q) is an entropy regularization
   805	term to avoid excessively matching; λ is a regularization
   806	parameter; 1 is an all-one vector. Finally, Q is refined using the
   807	Sinkhorn-Knopp algorithm [57] to globally associate positive
   808	cross-modality identities by selecting the highest probability
   809	values from its rows and columns.
   810	3) Discriminative Reliable Heterogeneous Learning: When
   811	relying solely on reliable homogeneous contrastive learning,
   812	the model becomes confined to learning modality-specific
   813	features, which hinders it from learning the distribution of
   814	data with identical labels across different modalities in the
   815	feature space by leveraging established cross-modality correspondences. Therefore, we propose a discriminative reliable
   816	heterogeneous loss to mine modality-invariant features and
   817	reduce the cross-modality differences, which is defined as:
   818	
   819	
   820	NB
   821	/τ)
   822	exp( fiv · Mr ŷv→r
   823	1 X
   824	i
   825	v→r
   826	h
   827	i
   828	, (21)
   829	LDRHL = −
   830	log P r
   831	P
   832	NB
   833	exp( f v · Mr ŷv→r /τ)
   834	i=1
   835	
   836	j=1
   837	
   838	i
   839	
   840	NB
   841	X
   842	
   843	j
   844	
   845	
   846	
   847	/τ)
   848	exp( fir · Mv ŷr→v
   849	1
   850	i
   851	h
   852	i
   853	Lr→v
   854	=
   855	−
   856	log
   857	,
   858	DRHL
   859	P Pv
   860	NB
   861	r→v
   862	r
   863	v
   864	exp(
   865	f
   866	·
   867	M
   868	ŷ
   869	/τ)
   870	i=1
   871	j=1
   872	i
   873	j
   874	
   875	(22)
   876	
   877	where ŷv→r
   878	denotes the corresponding label in the infrared
   879	i
   880	modality for the given sample fiv from the visible modality.
   881	Analogously, ŷr→v
   882	represents the visible pseudo-label matched
   883	i
   884	with fir from the infrared modality.
   885	Inspired by [25], we alternately use two unidirectional
   886	metric losses to obtain LDRHL , which can be formulated as
   887	follows:
   888	(
   889	Lv→r
   890	DRHL , epoch%2 = 0,
   891	(23)
   892	LDRHL =
   893	Lr→v
   894	DRHL , epoch%2 = 1,
   895	where LDRHL minimizes the distance between the given sample
   896	and their corresponding cross-modality prototype, while simultaneously maximizing the discrepancy between the sample and
   897	all other reliable prototypes.
   898	Finally, the overall loss is summarized as a combination of
   899	LRHCL and LDRHL , which is defined as follows:
   900	L = LRHCL + µLDRHL ,
   901	
   902	(24)
   903	
   904	where µ is the loss balance factor.
   905	IV. E XPERIMENTS
   906	A. Datasets and Evaluation Protocols
   907	1) Datasets: Following existing US-VI-ReID works [25],
   908	[27], we evaluate our method on two widely used crossmodality ReID datasets: SYSU-MM01 [58] and RegDB [59].

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '161,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   161	3. Our approach
   162	
   163	suppression network to learn a more discriminative and robust feature.
   164	Sun et al. [9] conducted a uniform partition on the conv-layer for
   165	learning part-level features. These methods are a single-modality identification task, where only the intra-modality discrepancy is measured.
   166	However, these methods cannot be applied to nighttime surveillance
   167	scenarios due to the large difference between visible and near-infrared
   168	modalities. In this paper, we propose a cross-modality multi-granularity
   169	Transformer to learn coarse-to-fine feature representation and reduce
   170	differences between modalities.
   171	
   172	In this section, we introduce cross-modality multi-granularity Transformer (CM2 GT) framework for VI-ReID. Firstly, we give a brief
   173	overview of the network structure. Secondly, we present a multigranularity feature extractor (MGFE) to capture global-, patch-, and
   174	pixel-level features for each modality. Thirdly, a multi-granularity
   175	fusion Transformer (MGFT) is introduced to fuse cross-modality information and interact with multi-granularity features. Finally, we
   176	propose a cross-modality nearest neighbor clustering (CNC) loss function to make the distance between intra-class cross-modality samples
   177	closer.
   178	
   179	2.2. Visible-infrared person re-identification
   180	Visible-infrared person re-identification attempts to match visible
   181	images and near-infrared images of a person under disjoint cameras.
   182	Different from previous visible–visible ReID task, VI-ReID not only
   183	handles the intra-modality variations caused by changes in the viewing
   184	angle, but also reduces the inter-modality discrepancy caused by the
   185	intrinsically different imaging processes of visible and near-infrared
   186	cameras.
   187	Existing VI-ReID methods can be broadly classified into two main
   188	categories, including modality compensation methods and feature
   189	learning-based methods. In the first category of methods, Li et al. [12]
   190	introduced an auxiliary modality to reduce the inter-modality discrepancy. Wang et al. [13] proposed model consists of a pixel generator,
   191	a feature generator and a joint discriminator. In the second category
   192	of methods, Wei et al. [8] exploited part representations to learn more
   193	fine-grained information. Ye et al. [4] dynamically learn the structural
   194	features of the graph between modalities.
   195	These main methods effectively extracted single granularity features
   196	by focusing on coarse or fine regions, leading to irrelevant matching and ambiguous alignment. In this paper, we capture multiplegranularity features based on the Transformer structure to solve these
   197	problems.
   198	
   199	3.1. Overview
   200	The overview of our approach is shown in Fig. 2. Firstly, in the feature extractor, ResNet50 [20] pre-trained on ImageNet [21] is adopted
   201	as the backbone of the two-stream network. 𝑆0 and 𝑆1 are the stages
   202	that learn modality-shared features. 𝑆2 , 𝑆3 and 𝑆4 are the stages
   203	which learn modality-specific features. The visible modality feature
   204	𝑋 𝑣𝑖𝑠 and near-infrared modality feature 𝑋 𝑛𝑖𝑟 are obtained by a twostream network. Secondly, MGFE module utilizes 𝑋 𝑣𝑖𝑠 and 𝑋 𝑛𝑖𝑟 to
   205	𝑣𝑖𝑠
   206	𝑛𝑖𝑟
   207	capture global-level features 𝑋 𝑔𝑙𝑜𝑏𝑎𝑙 and 𝑋 𝑔𝑙𝑜𝑏𝑎𝑙 , patch-level features
   208	𝑣𝑖𝑠
   209	
   210	𝑛𝑖𝑟
   211	
   212	𝑣𝑖𝑠
   213	
   214	𝑛𝑖𝑟
   215	
   216	𝑋 𝑝𝑎𝑡𝑐 ℎ and 𝑋 𝑝𝑎𝑡𝑐 ℎ , and pixel-level features 𝑋 𝑝𝑖𝑥𝑒𝑙 and 𝑋 𝑝𝑖𝑥𝑒𝑙 . Thirdly,
   217	𝑣𝑖𝑠
   218	
   219	𝑛𝑖𝑟
   220	
   221	𝑣𝑖𝑠
   222	
   223	𝑛𝑖𝑟
   224	
   225	MGFT fuses multi-granularity features (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 )
   226	across modalities. Finally, features from the global, patch, and pixel
   227	branches are aggregated together to obtain output feature 𝑋𝑜𝑢𝑡𝑝𝑢𝑡 , and
   228	are used to calculate the total loss function.
   229	3.2. Multi-granularity feature extractor
   230	As shown in Fig. 2, the multi-granularity feature extractor (MGFE)
   231	includes three branches: global branch, patch branch, and pixel branch.
   232	The global branch can capture the most salient features of the pedestrian image. The patch branch effectively explores on patch discriminative information and long-range dependency relationships within
   233	each modality. The pixel branch uses Transformer on each local part
   234	to capture pixel granularity features, which focuses on finer and richer
   235	discriminative information and filters out information interference on
   236	other local parts. These three-level granularity features can complement
   237	each other to obtain rich feature representation in images.
   238	The features from the visible and NIR modalities are processed
   239	by parameter-sharing MGFE. Considering the processing of features
   240	from both modalities is similar in each branch of MGFE, the
   241	process of the visible modality is described in detail for simplicity.
   242	
   243	2.3. Multi-granularity in computer vision field
   244	In the field of computer vision, multi-granularity based methods
   245	have achieved better performance. For example, In terms of multigranularity feature capture, [14] elucidated skeletal dynamics within
   246	various contexts. Our approach extracts features across the granularity,
   247	from global to patch to pixel levels, potentially yielding a more detailed
   248	and comprehensive data feature map. At the multi-granularity feature
   249	interaction level, [15] introduced a unified perception framework to
   250	identify behavioral patterns across granularities. Meanwhile, [16] concentrated on enhancing feature learning within and across modalities
   251	via attention mechanisms. Our approach captures features at multiple granularities and emphasizes their interaction and fusion across
   252	modalities.
   253	
   254	3.2.1. Global branch
   255	As shown in Fig. 2, given visible modality feature 𝑋 𝑣𝑖𝑠 , we utilize
   256	global max-pooling (GMP) and Batch Normalization (BN) layers to
   257	𝑣𝑖𝑠
   258	capture global granularity features 𝑋 𝑔 𝑙𝑜𝑏𝑎𝑙 , as follows:
   259	
   260	2.4. Transformer in VI-ReID
   261	ViT [17] is the first Transformer-based image classification model
   262	that promotes the application of Transformers in the field of image classification. In VI-ReID, the Transformer models [18,19] show advantages
   263	in capturing long-range dependencies and extracting discriminative features. Liang et al. [18] fused with token embeddings to encode modality
   264	information directly, and proposed a modality-aware enhancement loss
   265	to enhance representation of modality embeddings. Chen et al. [19]
   266	unified structure-related appearance learning and part-level interaction
   267	learning to enhance the semantically.
   268	These methods demonstrate that Transformers have shown superior
   269	capability and great potential for different tasks. In our approach, we
   270	propose a saliency-enhanced Transformer to preserve the saliency consistency across modalities and granularities and perform cross-modality
   271	cross-granularity features alignment.
   272	
   273	𝑣𝑖𝑠
   274	
   275	𝑋 𝑔𝑙𝑜𝑏𝑎𝑙 = 𝐵 𝑁(𝐺𝑀 𝑃 (𝑋 𝑣𝑖𝑠 ))
   276	
   277	(1)
   278	
   279	3.2.2. Patch branch
   280	To capture long-range dependencies within each modality, we design a patch branch based on the Transformer module to obtain the
   281	𝑣𝑖𝑠
   282	patch-level granularity feature 𝑋 𝑝𝑎𝑡𝑐 ℎ .
   283	𝑣𝑖𝑠
   284	Firstly, we reshape 𝑋
   285	∈ R𝐶×𝐻1 ×𝑊1 into a sequence of patches
   286	𝑣𝑖𝑠
   287	𝑁1 ×(𝑃ℎ ×𝑃𝑤 ×𝐶) , where 𝐶, 𝐻 , and 𝑊 denote the feature
   288	𝑋𝑝𝑎𝑡𝑐
   289	∈
   290	R
   291	1
   292	1
   293	ℎ
   294	dimension of the channel, width and height, respectively. 𝑃ℎ × 𝑃𝑤
   295	represents the resolution of each patch. 𝑁1 = (𝐻1 × 𝑊1 )∕(𝑃ℎ × 𝑃𝑤 )
   296	denotes the number of patches. Then, the class token [CLS] is added
   297	to beginning of the sequence to capture the global feature of the
   298	whole image. Spatial information is incorporated by adding learnable
   299	3
   300	
   301	Pattern Recognition 164 (2025) 111510
   302	
   303	Y. Feng et al.
   304	
   305	Fig. 2. Illustration of our cross-modality multi-granularity framework, including multi-granularity feature extractor (MGFE), multi-granularity fusion Transformer (MGFT), and total
   306	loss function. MGFE consists of global, patch, and pixel branches, which aims to capture three-level granularity features. MGFT interacts feature pairs in the patch branch and
   307	pixel branch. The total loss function is employed to supervise the whole training process, which improves intra-class cross-modality sample compactness.
   308	
   309	position embedding for each patch, resulting in the sequence with a
   310	𝑣𝑖𝑠
   311	size of 𝑋𝑝𝑎𝑡𝑐
   312	∈ R(𝑁1 +1)×(𝑃ℎ ×𝑃𝑤 ×𝐶) . Finally, this sequence enters the
   313	ℎ
   314	Transformer encoder to explore intra-modality relationships between
   315	patch features.
   316	The Transformer encoder [17] contains a multi-head self-attention
   317	(MHSA), layer normalization (LN) and a feed-forward network (FFN).
   318	𝑣𝑖𝑠 first is feed into three linear projections to generate query 𝑄, key
   319	𝑋𝑝𝑎𝑡𝑐
   320	ℎ
   321	𝐾 and value 𝑉 . Then it is fed into MHSA, and the input and output of
   322	MHAS are connected through the residual and LN. Finally, the input
   323	and output of FFN are also connected through the residual and sent to
   324	𝑣𝑖𝑠
   325	LN to obtain X𝑝𝑎𝑡𝑐 ℎ .
   326	𝑣𝑖𝑠
   327	Specifically, 𝑋𝑝𝑎𝑡𝑐
   328	is feed into three linear projections to generate
   329	ℎ
   330	query 𝑄, key 𝐾, and value 𝑉 . The weight sum of 𝑉 by attention weights
   331	can be formulated by:
   332	
   333	information in each strip. Specifically, we design the pixel branch to
   334	capture pixel granularity features, and refer each pixel as a patch, so
   335	the following process is essentially a pixel-level feature extraction.
   336	Firstly, for visible modality image 𝑋 𝑣𝑖𝑠 ∈ R𝐶×𝐻×𝑊 , we use uniform
   337	partition strategy [9] to obtain 𝑗 non-overlapping stripe features. The
   338	𝑣𝑖𝑠 .
   339	set of all stripes is 𝑋𝑎𝑙
   340	𝑙
   341	𝑣𝑖𝑠
   342	𝑣𝑖𝑠 𝑣𝑖𝑠
   343	𝑣𝑖𝑠
   344	𝑋𝑎𝑙
   345	𝑙 = {𝑥1 , 𝑥2 , … , 𝑥𝑗 }
   346	
   347	Secondly, we reshape each stripe feature 𝑥𝑣𝑖𝑠
   348	∈ R𝐶×𝐻2 ×𝑊2 , 𝑚 ∈
   349	𝑚
   350	[1, 2, … , 𝑗] into a sequence of fixed-size pixel patches, then add class token [CLS] and position embedding to obtain 𝑥𝑣𝑖𝑠
   351	∈ R(𝑁2 +1)×(𝑃 ×𝑃 ×𝐶) .
   352	𝑚,𝑝𝑖𝑥𝑒𝑙
   353	𝑃 ×𝑃 represents the resolution of each pixel patch. The set of sequences
   354	𝑣𝑖𝑠 , as follows:
   355	on all stripes is 𝑋𝑝𝑖𝑥𝑒𝑙
   356	𝑣𝑖𝑠
   357	𝑋𝑝𝑖𝑥𝑒𝑙
   358	= {𝑥𝑣𝑖𝑠
   359	, 𝑥𝑣𝑖𝑠 , … , 𝑥𝑣𝑖𝑠
   360	𝑗 ,𝑝𝑖𝑥𝑒𝑙 }
   361	1,𝑝𝑖𝑥𝑒𝑙 2,𝑝𝑖𝑥𝑒𝑙
   362	
   363	𝐴𝑡𝑡𝑛(𝑄, 𝐾 , 𝑉 ) = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝑄𝐾 𝑇 )𝑉 ,
   364	𝑣𝑖𝑠
   365	𝑣𝑖𝑠
   366	𝑣𝑖𝑠
   367	𝑄 = 𝑋𝑝𝑎𝑡𝑐
   368	ℎ 𝑊𝑞 , 𝐾 = 𝑋𝑝𝑎𝑡𝑐 ℎ 𝑊𝑘 , 𝑉 = 𝑋𝑝𝑎𝑡𝑐 ℎ 𝑊𝑣
   369	
   370	(6)
   371	
   372	(2)
   373	
   374	(7)
   375	
   376	where 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥 function is adopted to normalize the obtained attention
   377	weight. 𝑊𝑞 , 𝑊𝑘 and 𝑊𝑣 are learnable parameters. MHSA includes ℎ
   378	paralleled heads. Each head contains an independent scaled dot product
   379	attention function. The MHSA functions can be computed by:
   380	
   381	Thirdly, the 𝑚th sequence is fed into the Transformer encoder
   382	to obtain spatially-aware pixel granularity representation 𝑥𝑣𝑖𝑠
   383	𝑚,𝑝𝑖𝑥𝑒𝑙 . The
   384	Transformer encoder here is similar to the Transformer encoder in
   385	the Patch Branch, so we do not repeat it. Finally, the set of all pixel
   386	𝑣𝑖𝑠
   387	granularity features is 𝑋 𝑝𝑖𝑥𝑒𝑙 , as follows:
   388	
   389	𝑣𝑖𝑠
   390	𝑜
   391	𝑋𝑀
   392	𝐻 𝑆 𝐴 = [ℎ𝑒𝑎𝑑1 , ℎ𝑒𝑎𝑑2 , … , ℎ𝑒𝑎𝑑ℎ ]𝑊 ,
   393	
   394	𝑣𝑖𝑠
   395	𝑣𝑖𝑠
   396	𝑋 𝑝𝑖𝑥𝑒𝑙 = {𝑥𝑣𝑖𝑠
   397	1,𝑝𝑖𝑥𝑒𝑙 , 𝑥2,𝑝𝑖𝑥𝑒𝑙 , … , 𝑥𝑗 ,𝑝𝑖𝑥𝑒𝑙 }
   398	
   399	𝑣𝑖𝑠
   400	
   401	(3)
   402	
   403	ℎ𝑒𝑎𝑑𝑖 = 𝐴𝑡𝑡𝑛(𝑄𝑖 , 𝐾𝑖 , 𝑉𝑖 )
   404	
   405	where 𝑄𝑖 , 𝐾𝑖 , 𝑉𝑖 are the feature slices of query, key and value of the
   406	𝑖th head, and 𝑊 𝑜 ∈ Rℎ∗𝑑ℎ ×𝑑 . 𝑑ℎ = 𝑑∕ℎ is the dimension of the output
   407	features for each head.
   408	The outputs and inputs of MHSA are connected via residual connections and layer normalization (LN), which can be formulated by
   409	Eq. (4).
   410	𝑣𝑖𝑠
   411	𝑣𝑖𝑠
   412	𝑣𝑖𝑠
   413	𝑋𝐿𝑁
   414	= 𝐿𝑁(𝑋𝑝𝑎𝑡𝑐
   415	ℎ + 𝑋𝑀 𝐻 𝑆 𝐴 )
   416	
   417	3.3. Multi-granularity fusion transformer
   418	The ambiguous alignment in single-granularity based methods can
   419	lead to decreased performance. To mitigate this issue, we present
   420	a multi-granularity fusion Transformer (MGFT) module that effectively aligns features across multiple modalities and granularities with
   421	saliency-enhanced Transformer.
   422	The modality and granularity discrepancy results in different regions of interest. To preserve the saliency consistency across modalities
   423	and granularities, we propose a saliency-enhanced Transformer including saliency-consist attention (SCA) component different with the
   424	general Transformer. So we elaborate on its detailed attention design
   425	in Fig. 3.
   426	𝑣𝑖𝑠
   427	𝑛𝑖𝑟
   428	Given a set of inputs 𝑋𝑞′ , 𝑋𝑘′ , and 𝑋𝑣′ , {𝑋𝑞′ , 𝑋𝑘′ , 𝑋𝑣′ } ∈ {𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ ,
   429	
   430	(4)
   431	
   432	The feed-forward network (FFN) consisting of two linear projection
   433	layers and a GELU activation function is applied after the MHSA, as
   434	follows:
   435	𝑣𝑖𝑠
   436	
   437	𝑣𝑖𝑠
   438	𝑣𝑖𝑠
   439	𝑋 𝑝𝑎𝑡𝑐 ℎ = 𝐿𝑁(𝑋𝐿𝑁
   440	+ 𝐹 𝐹 𝑁(𝑋𝐿𝑁
   441	))
   442	
   443	(8)
   444	
   445	𝑣𝑖𝑠
   446	𝑛𝑖𝑟
   447	𝑣𝑖𝑠
   448	In this way, we obtain 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ of patch branch and 𝑋 𝑝𝑖𝑥𝑒𝑙 ,
   449	𝑛𝑖𝑟
   450	𝑋 𝑝𝑖𝑥𝑒𝑙 of the pixel branch from two modalities.
   451	
   452	(5)
   453	𝑣𝑖𝑠
   454	
   455	In this way, the patch-level features 𝑋 𝑝𝑎𝑡𝑐 ℎ of visible modality have
   456	been captured.
   457	
   458	𝑣𝑖𝑠
   459	
   460	𝑛𝑖𝑟
   461	
   462	𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 }, 𝑋𝑞′ and 𝑋𝑘′ come from different modalities or different
   463	granularities, and 𝑋𝑘′ and 𝑋𝑣′ come from the same modality and same
   464	granularity.
   465	For the general saliency-consist attention, the query input 𝑋𝑞′ is
   466	′
   467	first decomposed to class token [CLS] 𝑋𝑐′ 𝑙𝑠 and spatial tokens 𝑋𝑡𝑜𝑘𝑒𝑛𝑠
   468	.
   469	
   470	3.2.3. Pixel branch
   471	To more effectively capture the pixel-level details of pedestrian
   472	images, we leverage the uniform partitioning strategy to divide the
   473	image into stripes and perform the pixel-level Transformer on each
   474	strip, which allowing the model to focus more on learning fine-grained
   475	4
   476	
   477	Pattern Recognition 164 (2025) 111510
   478	
   479	Y. Feng et al.
   480	
   481	Fig. 4. The illustration of the CNC loss function.
   482	
   483	Through MGFE and MGFT, we obtain the final output feature 𝑋𝑜𝑢𝑡𝑝𝑢𝑡
   484	of the global, patch, and pixel branches, as follows:
   485	𝑣𝑖𝑠
   486	
   487	𝑛𝑖𝑟
   488	
   489	𝑂𝑔𝑙𝑜𝑏𝑎𝑙 = {𝑋 𝑔𝑙𝑜𝑏𝑎𝑙 , 𝑋 𝑔𝑙𝑜𝑏𝑎𝑙 },
   490	Fig. 3. The illustration of the saliency-consist attention.
   491	
   492	𝑣𝑖𝑠
   493	𝑛𝑖𝑟
   494	𝑂𝑝𝑎𝑡𝑐 ℎ = {𝑂𝑝𝑎𝑡𝑐
   495	ℎ , 𝑂𝑝𝑎𝑡𝑐 ℎ },
   496	𝑣𝑖𝑠
   497	𝑛𝑖𝑟
   498	𝑂𝑝𝑖𝑥𝑒𝑙 = {𝑂𝑝𝑖𝑥𝑒𝑙
   499	, 𝑂𝑝𝑖𝑥𝑒𝑙
   500	},
   501	
   502	𝑋𝑜𝑢𝑡𝑝𝑢𝑡 = {𝑂𝑔𝑙𝑜𝑏𝑎𝑙 , 𝑂𝑝𝑎𝑡𝑐 ℎ , 𝑂𝑝𝑖𝑥𝑒𝑙 }
   503	
   504	Originally, the global feature [CLS] is used for attention as 𝑋𝑞 , which is
   505	considered as global-wise standard attention. However, this learnable
   506	class embedding lacks spatial awareness, which is unable to align
   507	spatial regions. To highlight the spatial task-salient features across
   508	granularities, we use the average value of spatial tokens as saliency
   509	score 𝑠 to refine the attention score:
   510	
   511	3.4. Cross-modality nearest-neighbor clustering loss function
   512	In VI-ReID task, the triplet-based loss functions [4,5] has been
   513	widely used. When the distance between intra-class samples is smaller
   514	than a predefined threshold, traditional triplet-based loss functions
   515	provide weak constraints on intra-class clustering. It is difficult for
   516	VI-ReID to make intra-class cross-modality distance close due to the
   517	modality discrepancy. Therefore, we design a cross-modality nearestneighbor clustering loss function 𝐿𝑐 𝑛𝑐 including neighbor consistency
   518	loss 𝐿𝑛𝑐 and modality-alignment loss 𝐿𝑚𝑎 to further pull cross-modality
   519	intra-class samples. 𝐿𝑛𝑐 aims to reduce the differences between each
   520	sample and its nearest-neighbor. 𝐿𝑚𝑎 measures the deviation between
   521	the distributions of visible modality and near-infrared modality.
   522	As shown in Fig. 4, for each sample, we first find the neighboring
   523	features within the same class across different modalities. Then, 𝐿𝑛𝑐
   524	is designed to shorten the distance between samples within a class
   525	and promote consistency between intra-class cross-modality samples,
   526	and 𝐿𝑚𝑎 is proposed to reduce the difference between modalities and
   527	enhance cross-modality alignment.
   528	̃ 𝑣𝑖𝑠
   529	Given an anchor sample 𝑋
   530	∈ 𝑋𝑜𝑢𝑡𝑝𝑢𝑡 , the nearest-neighbor
   531	𝑎𝑛𝑐 ℎ𝑜𝑟
   532	𝑛𝑖𝑟
   533	̃ , 𝑡 ∈ [1, 2, … , 𝑘] is selected from intra-class cross-modality
   534	sample 𝑋
   535	𝑡
   536	̃ 𝑛𝑖𝑟 = {𝑋
   537	̃ 𝑛𝑖𝑟 , 𝑋
   538	̃ 𝑛𝑖𝑟 , … , 𝑋
   539	̃ 𝑛𝑖𝑟 }. 𝑘 denotes the number of intrasamples 𝑋
   540	𝑘
   541	1
   542	2
   543	class cross-modality samples in a mini-batch. We use Euclidean distance
   544	to calculate distance between anchor sample and its cross-modality
   545	nearest-neighbor samples.
   546	
   547	ℎ𝑒𝑎𝑑 = 𝐴𝑡𝑡𝑛((𝑋𝑞 + 𝑠)𝑊𝑞 , 𝑋𝑘 𝑊𝑘 , 𝑋𝑣 𝑊𝑣 )
   548	= 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥((𝑄 + 𝑠𝑊𝑞 )𝐾 𝑇 )𝑉
   549	
   550	(9)
   551	
   552	≈ 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥((𝑄𝐾 𝑇 )𝑉 ) + 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥((𝑠𝑊𝑞 𝐾 𝑇 )𝑉 )
   553	⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟ ⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟
   554	standard attention
   555	
   556	saliency-enhanced attention
   557	
   558	where 𝑄, 𝐾 , 𝑉 denote query, key and value respectively. 𝑊𝑞 , 𝑊𝑘 , 𝑊𝑣 are
   559	the learnable weights for projecting 𝑋𝑞 , 𝑋𝑘 , 𝑋𝑣 features to query, key,
   560	value space respectively. The output 𝑍 ′ of a saliency-consist attention
   561	(SCA) module of a set of inputs 𝑋𝑞′ , 𝑋𝑘′ , and 𝑋𝑣′ is defined as follows.
   562	𝑋𝑐′′𝑙𝑠 = 𝑋𝑐′ 𝑙𝑠 + 𝑆 𝐶 𝐴(𝑋𝑞′ , 𝑋𝑘′ , 𝑋𝑣′ ),
   563	
   564	(10)
   565	′
   566	𝑍 ′ = [𝑋𝑐′′𝑙𝑠 ; 𝑋𝑡𝑜𝑘𝑒𝑛
   567	]
   568	Thus, the output 𝑂 of saliency-enhanced Transformer (ST) is defined
   569	as1 :
   570	𝑂 = 𝑆 𝑇 (𝑋𝑞′ , 𝑋𝑘′ , 𝑋𝑣′ ) = ℎ𝑒𝑎𝑑(𝑋𝑞′ , 𝑋𝑘′ , 𝑋𝑣′ )
   571	
   572	(11)
   573	
   574	We use saliency-enhanced Transformer to fuse inter-modality intragranularity and intra-modality inter-granularity features as:
   575	𝑣𝑖𝑠
   576	𝑂𝑝𝑎𝑡𝑐
   577	ℎ =
   578	
   579	𝑣𝑖𝑠
   580	
   581	𝑛𝑖𝑟
   582	
   583	𝑛𝑖𝑟
   584	
   585	𝑆 𝑇 (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ )
   586	⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟
   587	
   588	+
   589	
   590	𝑣𝑖𝑠
   591	
   592	𝑣𝑖𝑠
   593	
   594	(13)
   595	
   596	𝑣𝑖𝑠
   597	
   598	𝑆 𝑇 (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 )
   599	⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟
   600	
   601	inter-modality intra-granularity fusion intra-modality inter-granularity fusion
   602	𝑛𝑖𝑟
   603	𝑣𝑖𝑠
   604	𝑣𝑖𝑠
   605	𝑛𝑖𝑟
   606	𝑛𝑖𝑟
   607	𝑛𝑖𝑟
   608	𝑛𝑖𝑟
   609	𝑂𝑝𝑎𝑡𝑐 ℎ = 𝑆 𝑇 (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ ) + 𝑆 𝑇 (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 )
   610	𝑣𝑖𝑠
   611	𝑛𝑖𝑟
   612	𝑛𝑖𝑟
   613	𝑣𝑖𝑠
   614	𝑣𝑖𝑠
   615	𝑣𝑖𝑠
   616	𝑣𝑖𝑠
   617	𝑂𝑝𝑖𝑥𝑒𝑙
   618	= 𝑆 𝑇 (𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 ) + 𝑆 𝑇 (𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ )
   619	𝑛𝑖𝑟
   620	𝑣𝑖𝑠
   621	𝑣𝑖𝑠
   622	𝑛𝑖𝑟
   623	𝑛𝑖𝑟
   624	𝑛𝑖𝑟
   625	𝑛𝑖𝑟
   626	𝑂𝑝𝑖𝑥𝑒𝑙
   627	= 𝑆 𝑇 (𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 ) + 𝑆 𝑇 (𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ )
   628	
   629	̃ 𝑣𝑖𝑠 − 𝑋
   630	̃ 𝑛𝑖𝑟 ‖,
   631	{‖𝑋
   632	𝑎𝑛𝑐 ℎ𝑜𝑟
   633	1
   634	̃ 𝑣𝑖𝑠 − 𝑋
   635	̃ 𝑛𝑖𝑟 ‖, … ,
   636	{𝐷1 , 𝐷2 , … , 𝐷𝑘 } =‖𝑋
   637	𝑎𝑛𝑐 ℎ𝑜𝑟
   638	2
   639	
   640	(14)
   641	
   642	̃ 𝑣𝑖𝑠 − 𝑋
   643	̃ 𝑛𝑖𝑟 ‖}
   644	‖𝑋
   645	𝑎𝑛𝑐 ℎ𝑜𝑟
   646	𝑘
   647	̃ 𝑣𝑖𝑠
   648	The index 𝑡 of the nearest-neighbor sample of anchor sample 𝑋
   649	𝑎𝑛𝑐 ℎ𝑜𝑟
   650	can be obtained by:
   651	
   652	(12)
   653	
   654	𝑡 = ar g min(𝐷1 , 𝐷2 , … , 𝐷𝑘 ), 𝑡 ∈ [1, 2, … , 𝑘]
   655	
   656	(15)
   657	
   658	̃ 𝑛𝑖𝑟 of anchor
   659	By this way, the cross-modality nearest-neighbor sample 𝑋
   660	𝑡
   661	̃ 𝑣𝑖𝑠
   662	sample 𝑋
   663	can
   664	be
   665	selected.
   666	𝑎𝑛𝑐 ℎ𝑜𝑟
   667	
   668	1
   669	We represent it in single-head version and omit Feedforward Network and
   670	LayerNorm in Transformer for simplicity.
   671	
   672	5
   673	
   674	Pattern Recognition 164 (2025) 111510
   675	
   676	Y. Feng et al.
   677	
   678	Table 1
   679	The results of the shared and individual circumstances on the SYSU-MM01 dataset.
   680	
   681	Then, we obtain probability distributions 𝐴 and 𝑃 of anchor sample
   682	and nearest-neighbor by 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥 function.
   683	
   684	Settings
   685	
   686	𝑣𝑖𝑠
   687	̃ 𝑣𝑖𝑠 ), 𝑋
   688	̃ 𝑣𝑖𝑠
   689	𝐴 = 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥(𝑋
   690	𝑎𝑛𝑐 ℎ𝑜𝑟
   691	𝑎𝑛𝑐 ℎ𝑜𝑟 ∈ 𝑁
   692	
   693	All-search
   694	
   695	(16)
   696	
   697	̃ 𝑛𝑖𝑟 ), 𝑋
   698	̃ 𝑛𝑖𝑟 ∈ 𝑁 𝑛𝑖𝑟
   699	𝑃 = 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥(𝑋
   700	𝑡
   701	𝑡
   702	
   703	SYSU-MM01 dataset
   704	
   705	where 𝑁 𝑣𝑖𝑠 and 𝑁 𝑛𝑖𝑟 are the set of visible modality images and near-
   706	
   707	𝑠0
   708	𝑠1
   709	𝑠2
   710	𝑠3
   711	𝑠4
   712	
   713	infrared modality images.
   714	The neighbor consistency loss 𝐿𝑛𝑐 is proposed to guarantee that the
   715	classification results of 𝐴 and 𝑃 are consistent, which can be computed
   716	by:
   717	1 ∑∑
   718	𝐿𝑛𝑐 = −
   719	log⟨𝐴, 𝑃 ⟩, 𝑁 = {𝑁 𝑣𝑖𝑠 ∥ 𝑁 𝑛𝑖𝑟 }
   720	(17)
   721	|𝑁|
   722	
   723	Indoor-search
   724	
   725	R1
   726	
   727	R10
   728	
   729	R20
   730	
   731	mAP
   732	
   733	R1
   734	
   735	R10
   736	
   737	R20
   738	
   739	mAP
   740	
   741	66.29
   742	67.74
   743	69.79
   744	67.61
   745	65.48
   746	
   747	91.45
   748	91.14
   749	95.45
   750	94.35
   751	93.11
   752	
   753	96.66
   754	96.27
   755	98.71
   756	97.74
   757	97.45
   758	
   759	60.96
   760	60.44

codex
方法部分已经够用了，但你的第 4 点要求看“证据闭环”，这通常在实验和消融部分。我会继续用 `pdftotext` 定位各篇的 ablation、visualization、effectiveness 段落，只取和“作者如何证明问题重定义成立”直接相关的部分。
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
/bin/zsh -lc "pdftotext 'Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf' - | rg -n -i \"ablation|effectiveness|visual|typical interactive|pseudo-word|converting|feedback|ranking refinement|Table|Fig\\. 3|Does\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2:with Text Feedback
22:feedback to refine the sketch person retrieval ranking without
23:text training. A lightweight vision-to-text converting network
24:is proposed to represent sketches with equivalent pseudo-word
27:with text feedback tokens within CLIP’s textual space for
28:explicit sketch-text compositionality, thus achieving feedbackguided ranking refinement. Extensive experiments underscore
40:such photo queries — the target’s visual photos are often not
44:intricate visual details with similar structures.
46:modalities are both visual expressions, the sketch-based query
60:person retrieval through context assistance and feedback guidance, enabling
74:that can integrate the user’s text feedback to refine the sketch
79:provide text feedback with high-level semantic details based
83:through Context Assistance and Feedback Guidance. Retrieval
87:ranking refinement is conducted on CLIP’s excellent imagetext-aligned embedding space [8], where a textual feedbackguided query is provided to retrieve and rank gallery images
92:pseudo-word token to provide context assistance for interactive
94:integrated with user-guided text feedback through textual token
98:Meanwhile, a vision-to-text converting network is trained
105:feedback with sketch queries for ranking refinement.
106:• InteractReID enables interactive retrieval through context assistance and feedback guidance, which not only
132:pseudo-word token and conducting token concatenation with
134:person retrieval with text feedback.
136:B. Interactive Image Retrieval with user feedback
138:by incorporating user feedback with their intentions, which is
140:works focused on learning with feedback in various forms,
151:retrieval field and conduct modality late-fusion. Our InteractReID firstly introduces the idea of feedback refinement into
153:innovations, we represent sketches as pseudo-word tokens to
155:the sketch tokens can be seamlessly integrated with userprovided text feedback tokens with the help of CLIP’s implicit
167:transformer to obtain its visual feature iv = V(I) ∈ Rd . The
218:Converting
245:Converting
278:Pseudo-word Token Learning for Sketch
282:Feedback
287:Interactive Sketch Person Retrieval with Text Feedback
292:for sketch’s pseudo-word token generation. Right: During inference, basic sketch person retrieval is conducted through similarity calculation. To interactively
293:refine the retrieval ranking, users’ text feedback tw can be integrated with the sketch’s pseudo-word token sw through context and feedback guidance.
349:C. Pseudo-word Token Learning for Sketch
351:text feedback, based on the vision-text joint embedding space
352:provided by CLIP, we aim to find a pseudo-word token that can
356:Concretely, we train a lightweight vision-to-text converting
359:Given an input sketch query S, we first obtain its visual
361:Rd , which will be sent to the converting network Φv2w
362:to generate its equivalent pseudo-word token embedding as
370:in the visual space sv . To achieve this, we propose a selfcycle contrastive loss to impose training constraints on the
371:converting network Φv2w , i.e.,
410:TABLE I
437:TABLE II
462:converting network Φv2w and CLIP image encoder V on
473:In order to refine the retrieval ranking results, users can provide any textual feedback tw for interactive communication.
474:Specifically, we introduce context assistance through the welltrained vision-to-text converting network Φv2w for sketch’s
475:equivalent pseudo-word token generation: sw = Φv2w (sv ).
476:Then the feedback guidance is achieved through textual token
481:stq and gallery visual features ivg .
555:to 0.07. In the pseudo-word token learning process, 1-layer
561:simulate users’ text feedback. Note that our InteractReID does
566:in Table I and Table II, our InteractReID has demonstrated its
571:is likely due to the effectiveness of our cross-modal matching
576:Fig. 3. A typical interactive person retrieval process of our InteractReID. Top-10 initial and refined ranking results on the Market-Sketch-1K dataset are
577:compared through sketch-based and interactive person retrieval with text feedback. Gallery samples in green boxes match the query, while in red boxes
579:TABLE III
613:person retrieval through context assistance and text feedback guidance. Compared to existing sketch+text composition
618:capability achieved by integrating sketch’s pseudo-word tokens
619:with user-provided text feedback in CLIP’s language token
622:on Market-Sketch-1K for retrieval ranking refinement.
623:C. Ablation Study
624:We comprehensively evaluate the effectiveness of the proposed InteractReID on Market-Sketch-1K from the following
627:shown in Table III, we first eliminate the knowledge adaptation
628:fine-tuning process and directly apply CLIP’s pre-trained parameters for downstream converting network training (No.1).
632:knowledge adaptation. Furthermore, we conduct ablation studies on the adaptation datasets (No.2 - No.5 in Table III),
635:TABLE IV
681:2) Is Vision-to-text Converting Network effective? The
682:converting network is designed to achieve feature-equivalent
683:mapping from visual to textual space. To verify the above
684:motivation, we conduct ablation studies in Table IV from the
685:following three designs: directly sending visual features into
686:the text encoder without network converting (No.1), mapping
688:MLP designs for the converting network (No.3 - No.11).
690:necessary to map visual features into the textual embedding
693:the converting network. (3) Considering the balance between
703:3) Does pseudo-word tokens really capture sketch semantics? To analyze the effectiveness of the pseudo-word token
705:model using relative validation datasets. Concretely, we utilize sketches’ pseudo-word tokens as queries to retrieve in
707:95.64% and R@5 of 99.76% on Market-Sketch-1K demonstrate that the semantic effectiveness of pseudo-word tokens.
709:A typical interactive person retrieval process is demonstrated in Figure 3 to verify the practicality of our InteractReID. Firstly, basic sketch-based person retrieval is conducted, which tends to match persons only with similar
711:framework can support users to provide discriminative semantic feedback for interactive and flexible retrieval. Concretely,
712:when mapping the sketch into its equivalent pseudo-word
713:token and combining it with feedback tokens in the textual
714:domain, InteractReID can achieve retrieval ranking refinement.
718:integrate sketch queries with user’s text feedback to achieve
719:retrieval ranking refinement. Concretely, Task-oriented Knowledge Adaptation is first conducted for CLIP’s pre-trained
720:knowledge transfer. Based on this, a vision-to-text converting
721:network is trained to represent sketch as a pseudo-word token
723:sketch token and text feedback tokens for interactive retrieval.
725:our superiority. Comprehensive ablation studies are conducted
726:to validate the effectiveness of the proposed InteractReID
729:based on user’s text feedback.
757:[8] Alec Radford, Jong Wook Kim, et al., “Learning transferable visual
782:[18] Yanbei Chen et al., “Image search with text feedback by visiolinguistic
784:[19] Seungmin Lee, Dongwan Kim, and Bohyung Han, “Cosmo: Contentstyle modulation for image retrieval with text feedback,” in Proc. of
787:transformer for fashion retrieval with feedback,” in Proc. of CVPR,
789:[21] Dafeng Li and Yingying Zhu, “Visual-linguistic alignment and composition for image retrieval with text feedback,” in Proc. of ICME. IEEE,

exec
/bin/zsh -lc "pdftotext 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|effect of|Table 4|Table 5|Fig\\. 7|MAM|MACM|PAM|FMM\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
29:specifically for occluded person Re-ID. MTIPE consists of several new modules: a Mask Aware Module (MAM)
31:Module (MACM) to enrich the feature representation; a Pose Aggregation Module (PAM) to separate useful
32:human information from the occlusion noise; a Feature Matching Module (FMM) in matching non-occluded
34:Pooling Attention Module (PAM) instead of traditional self-attention module to better extract and propagate
99:• In order to better retrieve and match the masked person, MAM
102:while MACM facilitates the integration of nuanced feature representations across the image. The inclusion of local prototypes in
108:• To demonstrate the effectiveness of our method, we perform experiments on occluded and holistic Re-ID datasets. Experimental
141:(MAM) and a Multi-headed Attention Constraint Module (MACM), a
142:Local Prototype based Transformer Decoder, including a Pooling Attention Module (PAM) to perform different parts recognition through
143:the transformer encoder–decoder architecture, a Pose Feature Aggregation Module (PFAM) and a Feature Matching Module (FMM). One
148:the MAM extracts local information through a convolutional neural
150:the multi-headed attention module to aggregate more feature representations with nuances throughout the image, we design a MACM to
154:to distinguish body parts and introduce FMM to strengthen encoder
163:evaluate the effectiveness of the proposed MTIPE, we conduct a series
186:leveraging intra-/inter-granularity of visual-text representations for semantic alignment of occluded visuals and query texts. Xia et al. [15]
194:The mask-based transformer encoder includes a Multi-headed Attention Constraint Module (MACM) and a Mask Aware Module (MAM). For masked human images, the MACM
196:by integrating global feature map information and posture information from the target person. For different images MAM generates a dynamic mask that selects subspace from
197:the overall prototype for matching. ‘‘Pooling-Attention’’ denotes the Pooling Attention Module (PAM).
238:are used to learn the feature representation. Our proposed MACM is
241:connections of the different image patches to the MAM, thus generating
287:3.1.1. Mask Aware Module (MAM)
288:We propose MAM with the aim of learning a dynamic mask to cut
290:which is one of the key ideas of MTIPE. As shown in Fig. 3, MAM
299:Fig. 3. The structure of the Mask Aware Module (MAM). The MAM is a convolution-based dynamic mask generator that considers the neighbor nodes of each patch, evaluates
315:MACM, we only use class token as a global representation, so when
339:layer, and C ∈ R1×𝐿 is for a gate control unit that selects the MAM
348:3.1.2. Multi-head Attention Constraint Module (MACM)
356:it difficult for MAM to generate a valid prototype mask. Therefore,
357:we introduce MACM to promote class tagging in multiple heads to
408:Fig. 4. As the network structure deepens, the feature representation of each patch becomes smooth and the prototype mask generated by MAM is difficult to function in the case
409:of highly similar inputs, so we propose the Multi-Head Attention Constraint Module (MACM) to enrich the feature representation by aggregating different feature representations
536:3.4. Pooling Attention Module (PAM)
538:to use a PAM to extract the input image 𝐼 local context information and
594:self-attention [18], PAM uses average pooling operation, which not
612:addition, our MTIPE also outperforms other mask-based [23,24] learning strategies, which fully demonstrates the effectiveness of our fusion
626:In this section, to validate the effectiveness of our proposed method,
629:effectiveness of our proposed module through ablation experiments and
630:provide more visualization results at the end.
632:Results on Holistic Datasets. To demonstrate the effectiveness of
658:Table 4
659:Ablation study analysis of different modules.
754:PAM
876:Table 4.
884:Effectiveness of the MTE. As shown in Index 2, the +5.2% mAP improves the performance when only the mask-based transformer encoder
886:MTE includes MACM and MAM modules. MACM not only trains on
887:overall prototype with more different features can also help the MAM
919:Effectiveness of the LPTD. From Index 1 and Index 3, it can be seen
924:the effect of more noisy backgrounds such as occlusion. From the
937:Effectiveness of the PAM. As can be seen from Index 1 and Index
941:proving the effectiveness of propagating local contextual information
947:4.4. Ablation study
948:In this section, we perform an ablation study on the Occluded-Duke
949:to analyze the effectiveness of each component of our proposed MTIPE,
951:Attention Module (PAM), and the Local Prototype based Transformer
965:Fig. 7. Visualization on features generated by our proposed MTIPE module.
1036:4.5. Visualization
1037:We visualize the features generated by the proposed MTIPE. As
1038:shown in Fig. 7, the proposed MACM and MAM can locate the occluded body parts well, which proves the effectiveness of our proposed
1044:which includes a mask based transformer encoder (including MAM
1045:and MACM), a local prototype-based transformer decoder (including
1046:PAM), a pose feature aggregation module (PFAM), a feature matching
1047:module (FMM), and a Posture Key-points Loss. Among them, the MAM
1049:to match the occluded human images, and further proposes MACM
1054:distinguishable body parts and propose the FMM module to enhance
1171:Trans. on PAMI, Pattern Recognition, Pattern Recognition Letters, International Journal

exec
/bin/zsh -lc "pdftotext 'Internal-External Context Interaction Network for Person Re-Identification.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|IEI|IEI-S|IEI-C|Region-Diverse|external|context|Table|Fig\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Internal–External Context Interaction Network for
12:own contextual information within a single sample using selfaffinity between elements, and some works have demonstrated its
14:semantic cues and additional visual cues such as backpacks,
16:this paper, we propose an internal-external context interaction
17:(IEI) attention mechanism, which aims to exploit the interaction
18:of inter-sample latent context information and intra-sample
19:local context information to enhance the feature representation
21:differences between persons and additional visual cues using
24:an internal-external context interaction network (IEINet) for
27:a region-diverse loss to constrain the network. Many experiments
28:validate the effectiveness of our IEINet and demonstrate that our
30:Index Terms—Person re-identification, internal-external context interaction, discriminative features, region-diverse loss.
64:information such as the texture of clothes and extrinsic accessory information such as backpacks are important visual cues
67:subtle visual cues from each part, as shown in Figure 1(a).
87:researchers have begun to utilize element-specific context for
90:robust character feature representations by modeling contextual relationships between patches, while [22] jointly mines
93:excels in global context modeling, its inherent single-sample
99:texture, accessory details, and so forth (see Figure 1 (b), rows
106:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
110:Fig. 1. (a) presents the schematic comparison between frameworks for the proposed IEINet and typical related methods. Splitting-based methods [1], [2]
111:learn multiple part features containing subtle visual cues from strips, while attention and splitting-based methods [13], [26] introduce attention modules to
114:with each other. Our IEINet makes it easier to learn multidimensional attention information as well as semantically continuous part features through multiple
115:independent feature spaces and the region-diverse constraint. (b) presents the comparison between heatmaps of IEI-S and SA. IEI not only focuses on more
121:relevance to the subject region (see Figure 1(b), rows 3-4).
126:take full advantage of the local contextual information between
128:In this paper, we propose an internal-external context interaction (IEI) attention mechanism for person Re-ID, which
131:context (external information) of different samples with the
132:internal local context information of a sample. Specifically,
134:robust self-context information of the sample, while storing
135:potential semantic features between different pedestrian samples through an external learnable memory, which guides the
143:mechanism, we further design an internal-external context
144:interaction network (IEINet), which is a CNN-based threebranch network, where each branch is equipped with an IEI
146:a channel IEI (IEI-C) module is added to one branch, and
148:a spatial IEI (IEI-S) module is added to each of the other
150:feature decoupling learning through multiple independent feature spaces. In addition, to motivate different IEI-S branches
155:Compared with existing methods, IEINet is innovative in
158:shown in Figure 1(a), the multi-branch structure of IEINet provides independent optimization space for different dimensional
162:or after rigid splitting to enhance the robustness of partlevel features, as shown in Figure 1(a), IEINet can learn
164:multi-branch structure and the region-diverse constraint. In
167:using the interactions between the latent contextual
169:contextual information of each element within a sample.
181:2) We propose an IEINet. This network contains IEI-S and
182:IEI-C modules for mining the discriminative features in
183:the spatial and channel dimensions, respectively. In addition, for IEINet to capture the discriminative information
184:from more key regions, we propose a region-diverse loss
185:to constrain the IEI-S modules in different branches of
188:MSMT17 [31] and PersonX [32] to confirm the effectiveness of our proposed IEI mechanism and IEINet. The
192:TABLE I
205:to learn subtle visual cues. MGN [1] splits the feature map
231:data distribution. Compared to the above methods, our IEINet
232:can capture subtle visual cues without additional labels and
233:models with the help of IEI mechanism.
240:the channel weights using the global context of channels. Reference [44] restricts channel interactions to a localized range,
263:[19] and [20] use SA to directly model the global context
269:As shown in Table I, [19], [20], [22] use the SA mechanism
271:only utilise own contextual information within a single sample
275:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
279:Fig. 2. Overall architecture of IEINet. IEI-C is the channel internal-external context interaction module, and IEI-S is the spatial internal-external context
280:interaction module. They mine discriminative information from the channel and spatial dimensions, respectively. Lrd is a region-diverse loss under which two
281:different color-marked IEI-S modules can focus on different important regions. During the testing/inference phase of the network, all feature embeddings/vectors
286:them, our proposed IEI attention mechanism is constructed
287:based on both inter-sample context and intra-sample context,
288:which utilises inter-sample context to directly capture subtle differences between persons and additional visual cues,
293:robust intra-sample own context information and restrict the
294:interaction of intra-sample and inter-sample contexts to a local
297:In terms of diverse feature learning, as shown in Table I,
310:and splitting-based methods, on the one hand, IEINet learns
313:information. On the other hand, IEINet not only can learn
315:the multiple independent feature spaces and the region-diverse
327:among all samples, thus capturing clues related to the identities of samples. In addition, our IEI attention mechanism
335:multiple dimensions, we propose an IEINet, and the whole
336:architecture of this network is shown in Figure 2. This section
338:of IEINet. Secondly, we provide detailed information about
340:IEI attention mechanism. Lastly, we describe the loss function
341:used for IEINet.
343:As shown in Figure 2, IEINet is a multi-branch network
348:IEINet and change the step size of the spatial down-sampling
350:thus allowing our IEI modules to extract important information
354:IEINet backbone. Then, the res conv5 layer is duplicated into
364:Fig. 3. Diagram of IEI-S. CAP denotes channel average pooling. denotes
369:three parts, and an IEI module is added after each part to form
371:Adding the IEI-C module to Branch-C allows this branch to
376:feature embedding for this branch. IEI-S modules are added
387:a region-diverse loss (Lrd ) to make their two spatial modules
392:B. Spatial Internal-External Context Interaction Module
393:The IEI-S module aims to use the interactions between the
394:contextual information of different samples and the spatial
395:local context in a sample to infer the important positions
397:all positions; the structure of this module is shown in Figure 3.
401:a spatially robust contextual representation of a single sample,
405:of a location. Q s reflects the robust activation contextual
411:spatial contextual information between samples, which is a
412:learnable external memory that is independent of the individual samples, and it learns the spatial contextual information
415:by interacting the inter-sample context with the local selfinformation within the sample. The local context information
420:slide on Q s to obtain a local context query matrix Q s0 with
424:i ∈ {1, 2, . . . , H W} of Q s0 is a local context query vector for
429:Next, we project the local context Qis0 at the ith location
430:onto the inter-sample spatial context K s , and obtain the k × k
432:projected onto K s to realize the internal and external context
443:context information under the guidance of the inter-sample
444:context. This method emphasizes the important identity-related
475:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
479:local context query matrix and the inter-sample context key
486:Fig. 4. Diagram of IEI-C. SAP represents spatial average pooling.
497:C. Channel Internal-External Context Interaction Module
498:As a complement to the IEI-S module, the IEI-C module
500:interaction of inter-sample contextual information with intrasample channel local context. The structure of this module is
501:shown in Figure 4. We use the feature map X ∈ RC×H×W
503:contextual representation of a sample in terms of channels,
506:reflects the activation contextual information of the feature
509:we introduce a key matrix K c ∈ Rk×k that stores intersample channel context information and let it interact with
510:the intra-sample channel local context information to generate
511:aggregated weights for each channel. The local contextual
515:of 1 to slide C times on Qc to obtain a local context query
517:Qc0j , j ∈ {1, 2, . . . , C} is the local context query vector for
522:Next, we establish the internal and external contextual
527:channel, which is obtained by projecting the local context Qc0j
528:onto the inter-sample channel context K c and implementing
561:employed in person Re-ID approaches [1], [12], [57]. In addition, to enable IEINet to extract more different discriminative
562:features, we also propose a region-diverse loss.
601:TABLE II
640:3) Region-Diverse Loss: To improve the performance of
641:IEINet in the person Re-ID task, we design a loss that is used
642:between the two spatial branches of IEINet (Branch-S1 and
645:for identification. The loss is named the region-diverse loss,
656:tensors output from the IEI-S modules in Branch-S1 and
661:they emphasize. Therefore, IEINet is able to capture more
664:4) Total Loss: The total loss of IEINet is calculated as
668:where λ is a weight hyperparameter for adjusting the importance of the region-diverse loss (Lrd ) in the overall loss.
674:MSMT17 [31] and PersonX [32]. Table II presents the divisions of training and test data, along with the number of
718:on ImageNet to initialize the ResNet-50 backbone of IEINet
730:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
732:TABLE III
737:TABLE V
740:TABLE VI
743:TABLE IV
746:algorithm is used to optimize the parameters of IEINet. The
753:To demonstrate the effectiveness of our proposed IEINet,
757:Tables III, IV, V and VI.
758:1) Results on Market1501: Table III shows the performance
764:representation capabilities, while our IEINet uses the modules
765:designed based on the IEI mechanism to learn discriminative
767:results in Table III, we can observe that our IEINet exceeds
769:IEINet are multi-branch networks. MGN divides a feature map
771:outperforms IEINet by 0.2% in Rank-1 accuracy using more
772:loss calculations, IEINet still outperforms it by 2.0% in mAP.
773:2) Results on CUHK03: Table IV shows the performance
774:comparison of our IEINet and the current state-of-the-art methods on CUHK03-Labeled and CUHK03-Detected datasets.
775:The results in the table show that our method outperforms the
781:CUHK03-Labeled dataset, our IEINet outperforms them by
783:respectively. On the CUHK03-Detected dataset, our IEINet
792:3) Results on MSMT17: Table V shows the performance
793:comparison of our IEINet with the state-of-the-art methods
811:different sites. The Rank-1 accuracy and mAP of IEINet reach
814:4) Results on PersonX: Table VI shows the performance
815:comparison of our IEINet with the state-of-the-art methods on
821:Table VI, we can see that our IEINet outperforms them on
824:In addition, we find that IEINet (ViT) does not perform
825:as well as IEINet (ResNet-50) on the above datasets, but the
833:D. Ablation Study
834:To verify the effectiveness of the key components in the
835:IEINet and find its optimal structure, we create a baseline
838:the IEI-C module is removed. We perform a wide range
839:of ablation experiments on the Market1501 and CUHK03Labeled datasets.
840:1) Determination of the Local Context Range: The hyperparameter k of the IEI-S and IEI-C modules determines the
841:range of local contextual information extracted for each element (location/channel). To ensure that the unfolding operation
846:single-branch networks by adding an IEI-S module and an
847:IEI-C module after the res conv5 layer of Baseline, respectively, and the experimental results of these two single-branch
848:networks with different values of k are shown in Figure 5.
849:We can observe that the performances of “Baseline+IEIS” and “Baseline+IEI-C” on Market1501(CUHK03-Labeled)
851:TABLE VII
855:gradually. The mAP of “Baseline+IEI-S” is higher than that
857:mAP of “Baseline+IEI-C” is higher than that of Baseline.
867:of both the IEI-S and IEI-C modules, we finally determine the
868:value of hyperparameter k to be 3 for both the IEI-S and IEI-C
870:2) Validity of the Local Context Modeling: In the IEI-S
871:and IEI-C modules, we implement a channel/spatial dimension
873:robust intra-sample self-context information and further extract
874:local context information for each element while reducing
875:the computational effort of the model. To verify the effectiveness of this operation in these two modules, we use the
877:“Baseline+IEI-S” and “Baseline+IEI-C” attention modules.
878:“Baseline+IEI-S (w/o CAP)” replaces the CAP operation
879:in “Baseline+IEI-S” with a 3 × 3 convolution, and the tensor
884:inter-sample contextual information. Similarly, “Baseline+IEIC (w/o SAP)” replaces the SAP operation in “Baseline+IEI-C”
888:According to the results in Table VII, we observe that
889:the Rank-1 accuracy and mAP of “Baseline+IEI-S (w/o
890:CAP)” and “Baseline+IEI-C (w/o SAP)” exceed those of
892:efficacious to use the local context information of elements
893:to interact with external information to reinforce the feature representation ability. Moreover, on the Market1501 and
894:CUHK03-Labeled datasets, “Baseline+IEI-S (w/o CAP)” and
895:“Baseline+IEI-C (w/o SAP)” are inferior to “Baseline+IEIS” and “Baseline+IEI-C” in Rank-1 accuracy and mAP,
901:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
905:Fig. 5. Comparison of the performance of Baseline with IEI-S and IEI-C under different k values. (a) The mAP of IEI-S on Market1501. (b) The mAP of
906:IEI-S on CUHK03-Labeled. (c) The mAP of IEI-C on Market1501. (d) The mAP of IEI-C on CUHK03-Labeled.
907:TABLE VIII
909:IEIN ET C OMPONENTS
911:can offer the IEI modules with more efficient and robust local
912:contextual information.
913:3) IEI-S vs. SA(Internal-External Interaction vs. IntraSample Interaction): To further verify the efficacy of internalexternal context interaction, we implement a “Baseline+SA”
916:with our IEI-S module. The SA module and IEI-S module
920:own information within a single sample. In contrast, our IEI-S
922:between the potential inter-sample contextual information and
923:the local contextual information within a sample.
924:From the results in Table VIII, it can be seen that the
925:IEI-S module outperforms the SA module by 0.7% Rank-1
929:TABLE IX
936:the context information between different samples.
939:complexity of the SA module, the IEI-S module and the
940:IEI-C module are respectively:
942:Ω(IEI − S ) = HWC 2 + HWk2C + HWk4 , k = 3
943:Ω(IEI − C) = HWC 2 + Ck2 + HWCk, k = 3
950:In contrast, both our IEI-S module and IEI-C module achieve
953:To further validate the computational efficiency of IEINet,
955:GFLOPs. As shown in Table IX, IEINet significantly reduces
963:Fig. 6. Heat maps of feature maps. Every four images form a group, in which the activation maps of Baseline, “Baseline+SA” and “Baseline+IEI-S” are
967:with cutting-edge methods. For example, IEINet requires only
970:(87.09 M / 22.07 G), respectively, and IEINet’s mAP is
973:of IEINet is reduced by 1.7%, but Params and GFLOPs of
974:IEINet are further reduced by 33.1% and 60.0%. These results
975:show that IEINet achieves efficient feature learning at a much
977:5) Heat Map Visualization Analysis: In Figure 6, we
991:Second, compared with the SA module, the IEI-S module is
997:The inter-sample context information in the IEI-S module is
999:intra-sample locale context and inter-sample context interact,
1005:the aggregation of features in the IEI-S module fully utilize
1006:the intra-sample local self-context, compared to the Baseline
1010:comparisons further confirm the validity of the IEI attention
1012:6) IEINet Structure Study: The results of the ablation
1014:Table VIII. From the table, we can observe that the IEI-C
1015:and IEI-S modules are effective not only when implemented
1018:(CUHK03-Labeled) dataset, the mAP of “Baseline+IEI-S” and
1019:“Baseline+IEI-C” exceeds that of Baseline by 1.2%+(1.0%+).
1020:The “RB+CB+SB-1” in Table VIII is a two-branch network consisting of ResNet-50 backbone, Branch-C and
1021:Branch-S1. Compared with the “Baseline+IEI-C”, it achieves
1023:two datasets. This is because the amalgamation of IEI-S and
1024:IEI-C modules can simultaneously learn rich discriminative
1026:IEINet (w/o Lrd )” formed by adding Branch-S2 to “RB +
1030:7) Impact of the Region-Diverse Loss: The ablation experimental results of the region-diverse loss function Lrd are shown
1031:in Table VIII. IEINet with Lrd outperforms “IEINet (w/o
1034:discriminative regions focused on by the IEI-S modules of
1038:the impact of the region-diverse loss Lrd in the overall
1044:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
1048:Fig. 7. Performance comparison of distinct values of λ. (a) The Rank-1 accuracy and mAP on Market1501. (b) The Rank-1 accuracy and mAP on CUHK03Labeled.
1050:Fig. 8. Retrieval results for six query images. The first three belong to the Market1501 dataset and the remaining three are from the CUHK03-Labeled dataset.
1051:For each query image, the top-5 retrieved results of both Baseline and IEINet are shown in each row. The image embedded in the green box has the same
1054:performance. We can observe from Figure 7 that the trends
1055:of Rank-1 accuracy of IEINet on Market1501 and CUHK03Labeled datasets are similar. As the value of λ increases,
1059:can also observe that IEINet achieves the maximum mAP on
1070:region-diverse loss in the overall loss.
1071:8) Sorted Visualization Analysis: Figure 8 displays and
1072:compares the top-5 retrieval results of Baseline and IEINet.
1075:our IEINet is still able to more accurately identify samples that
1079:Baseline, IEINet can make the best of this detailed information
1082:IEINet can better identify the person images based on this
1085:In this paper, we propose an IEINet to capture the discriminative features of persons from multiple dimensions. The key
1086:components of IEINet are its IEI modules and the regiondiverse loss. The IEI-S (IEI-C) module stores the potential
1087:contextual information between different person samples in
1088:an external learnable memory, and lets the local context of
1091:IEINet uses IEI modules and the region-diverse loss to mine
1093:effectiveness of our proposed method.
1202:LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION

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
/bin/zsh -lc "pdftotext 'Instant pose extraction based on mask transformer for occluded person re-identification.pdf' - | nl -ba | sed -n '876,1040p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   876	Table 4.
   877	
   878	Table 3
   879	Performance comparison with state-of-the-art models on MSMT17.
   880	Methods
   881	
   882	MTE
   883	
   884	Effectiveness of the MTE. As shown in Index 2, the +5.2% mAP improves the performance when only the mask-based transformer encoder
   885	is used compared to using the baseline model. Due to the fact that
   886	MTE includes MACM and MAM modules. MACM not only trains on
   887	overall prototype with more different features can also help the MAM
   888	module to select the appropriate subspace for matching from the overall
   889	prototype while generating dynamic masks. Therefore, the encoder is
   890	essential to improve the performance of the model.
   891	
   892	MSMT-17
   893	mAP
   894	
   895	Rank-1
   896	
   897	46.3
   898	47.6
   899	52.9
   900	46.8
   901	42.9
   902	–
   903	57.5
   904	55.7
   905	59.6
   906	61.1
   907	
   908	71.3
   909	73.6
   910	78.7
   911	75.5
   912	72.8
   913	72.0
   914	80.3
   915	79.2
   916	81.0
   917	80.4
   918	
   919	Effectiveness of the LPTD. From Index 1 and Index 3, it can be seen
   920	that when the local prototype based transformer decoder structure is
   921	added, the performance is improved by +6.3% mAP, which indicates
   922	that the prototype mask generated by the prototype can have a better
   923	effect on the matching of the occluded human images and can reduce
   924	the effect of more noisy backgrounds such as occlusion. From the
   925	comparison between Index 1 and Index 3, the performance is improved
   926	by +8.4% mAP when the decoder is added. This indicates that the
   927	simultaneous use of encoder–decoder can be better for pose information
   928	extraction and feature aggregation.
   929	
   930	+1.0% Rank1 accuracy on DukeMTMC-ReID. We also experimented
   931	our MTIPE model on the MSMT17, comparing several methods, including [30,33,36], 𝑒𝑡𝑐. The results are shown in Table 3, and our
   932	MTIPE model is equally competitive with other models, specifically, our
   933	method achieves 80.4% Rank-1 accuracy and 61.1% mAP on MSMT17.
   934	It can be seen that our proposed method is equally robust, both for the
   935	holistic Re-ID task and for the occluded Re-ID task.
   936	
   937	Effectiveness of the PAM. As can be seen from Index 1 and Index
   938	4, our overall model achieves the best performance with +9.4% mAP
   939	improvement compared to baseline, which is due to the fact that our
   940	proposed attention module can extract more distinguishing features,
   941	proving the effectiveness of propagating local contextual information
   942	among the prototypes. As can be seen from Index 3 and Index 4, where
   943	the introduction of the attention module in the encoder module. The
   944	performance of the model is also improved by +1.1% mAP compared
   945	to Index 4. The prototype after passing the attention module contains
   946	
   947	4.4. Ablation study
   948	In this section, we perform an ablation study on the Occluded-Duke
   949	to analyze the effectiveness of each component of our proposed MTIPE,
   950	including the Mask-based Transformer Encoder (MTE), the Pooling
   951	Attention Module (PAM), and the Local Prototype based Transformer
   952	7
   953	
   954	Pattern Recognition 159 (2025) 111082
   955	
   956	T.-T. Yuan et al.
   957	
   958	CRediT authorship contribution statement
   959	Ting-Ting Yuan: Writing – review & editing, Writing – original
   960	draft. Qing-Ling Shu: Writing – review & editing, Software. Si-Bao
   961	Chen: Writing – review & editing, Writing – original draft, Supervision, Resources, Project administration, Methodology, Funding acquisition. Li-Li Huang: Writing – review & editing, Supervision, Methodology, Funding acquisition. Bin Luo: Supervision, Resources, Funding
   962	acquisition.
   963	Declaration of competing interest
   964	All authors disclosed no relevant relationships.
   965	Fig. 7. Visualization on features generated by our proposed MTIPE module.
   966	
   967	Data availability
   968	different local contextual information and can focus on more useful
   969	parts of the occluded human image.
   970	
   971	The code will be shared soon.
   972	
   973	Analysis of the Number of Local Prototypes. The number of local
   974	prototypes determines the granularity of the features. For this we find
   975	the number 𝑀 that gives the highest performance of the model on the
   976	Occluded-Duke and Market-1501. As shown in Fig. 5, the performance
   977	keeps improving on the Occluded-Duke until 𝑀 reaches 16, and the
   978	best performance is shown in Fig. 6 for 𝑀 of 9 on the Market-1501.
   979	Therefore, it can be concluded that local prototypes work better in
   980	Occluded-Duke because finer-grained features are more useful for the
   981	identification of people in more complex scenarios.
   982	
   983	References
   984	[1] C. Eom, B. Ham, Learning disentangled representation for robust person
   985	re-identification, Adv. Neural Inf. Process. Syst. 32 (2019).
   986	[2] L. Wei, S. Zhang, W. Gao, Q. Tian, Person transfer gan to bridge domain gap
   987	for person re-identification, in: Proceedings of the IEEE Conference on Computer
   988	Vision and Pattern Recognition, 2018, pp. 79–88.
   989	[3] Y. Li, D. Miao, H. Zhang, J. Zhou, C. Zhao, Multi-granularity cross transformer
   990	network for person re-identification, Pattern Recognit. 150 (2024) 110362.
   991	[4] G. Wang, S. Yang, H. Liu, Z. Wang, Y. Yang, S. Wang, G. Yu, E. Zhou,
   992	J. Sun, High-order information matters: Learning relation and topology for
   993	occluded person re-identification, in: Proceedings of the IEEE/CVF Conference
   994	on Computer Vision and Pattern Recognition, 2020, pp. 6449–6458.
   995	[5] M. Huang, C. Hou, Q. Yang, Z. Wang, Reasoning and tuning: Graph attention
   996	network for occluded person re-identification, IEEE Trans. Image Process. 32
   997	(2023) 1568–1582.
   998	[6] J. Miao, Y. Wu, P. Liu, Y. Ding, Y. Yang, Pose-guided feature alignment for
   999	occluded person re-identification, in: Proceedings of the IEEE/CVF International
  1000	Conference on Computer Vision, 2019, pp. 542–551.
  1001	[7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner,
  1002	M. Dehghani, M. Minderer, G. Heigold, S. Gelly, An image is worth 16x16
  1003	words: Transformers for image recognition at scale, in: International Conference
  1004	on Learning Representations, 2021.
  1005	[8] P. Chen, W. Liu, P. Dai, J. Liu, Q. Ye, M. Xu, Q. Chen, R. Ji, Occlude
  1006	them all: Occlusion-aware attention network for occluded person re-id, in:
  1007	Proceedings of the IEEE/CVF International Conference on Computer Vision,
  1008	2021, pp. 11833–11842.
  1009	[9] P.K. Sarker, Q. Zhao, Enhanced visible–infrared person re-identification based
  1010	on cross-attention multiscale residual vision transformer, Pattern Recognit. 149
  1011	(2024) 110288.
  1012	[10] Y. Sun, L. Zheng, Y. Yang, Q. Tian, S. Wang, Beyond part models: Person retrieval
  1013	with refined part pooling (and a strong convolutional baseline), in: Proceedings
  1014	of the European Conference on Computer Vision, ECCV, 2018, pp. 480–496.
  1015	[11] S. Gao, J. Wang, H. Lu, Z. Liu, Pose-guided visible part matching for occluded
  1016	person reid, in: Proceedings of the IEEE/CVF Conference on Computer Vision
  1017	and Pattern Recognition, 2020, pp. 11744–11752.
  1018	[12] C. Zhao, Z. Qu, X. Jiang, Y. Tu, X. Bai, Content-adaptive auto-occlusion network
  1019	for occluded person re-identification, IEEE Trans. Image Process. 32 (2023)
  1020	4223–4236.
  1021	[13] L. Tan, J. Xia, W. Liu, P. Dai, Y. Wu, L. Cao, Occluded person re-identification
  1022	via saliency-guided patch transfer, in: Proceedings of the AAAI Conference on
  1023	Artificial Intelligence, 38, (5) 2024, pp. 5070–5078.
  1024	[14] X. Wu, W. Ma, D. Guo, T. Zhou, S. Zhao, Z. Cai, Text-based occluded person
  1025	re-identification via multi-granularity contrastive consistency learning, in: Proceedings of the AAAI Conference on Artificial Intelligence, 38, (6) 2024, pp.
  1026	6162–6170.
  1027	[15] J. Xia, L. Tan, P. Dai, M. Zhao, Y. Wu, L. Cao, Attention disturbance and dualpath constraint network for occluded person re-identification, in: Proceedings of
  1028	the AAAI Conference on Artificial Intelligence, 38, (6) 2024, pp. 6198–6206.
  1029	[16] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer:
  1030	Hierarchical vision transformer using shifted windows, in: Proceedings of the
  1031	IEEE/CVF International Conference on Computer Vision, 2021, pp. 10012–10022.
  1032	[17] S. He, H. Luo, P. Wang, F. Wang, H. Li, W. Jiang, Transreid: Transformer-based
  1033	object re-identification, in: Proceedings of the IEEE/CVF International Conference
  1034	on Computer Vision, 2021, pp. 15013–15022.
  1035	
  1036	4.5. Visualization
  1037	We visualize the features generated by the proposed MTIPE. As
  1038	shown in Fig. 7, the proposed MACM and MAM can locate the occluded body parts well, which proves the effectiveness of our proposed
  1039	method.
  1040	

exec
/bin/zsh -lc "pdftotext 'Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf' - | nl -ba | sed -n '623,715p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   623	C. Ablation Study
   624	We comprehensively evaluate the effectiveness of the proposed InteractReID on Market-Sketch-1K from the following
   625	three questions.
   626	1) Is Task-oriented Knowledge Adaptation necessary? As
   627	shown in Table III, we first eliminate the knowledge adaptation
   628	fine-tuning process and directly apply CLIP’s pre-trained parameters for downstream converting network training (No.1).
   629	This leads to significant performance degradation compared
   630	to No.5 in both the sketch-based and interactive retrieval
   631	scenarios, which highlight the importance of task-oriented
   632	knowledge adaptation. Furthermore, we conduct ablation studies on the adaptation datasets (No.2 - No.5 in Table III),
   633	where Tri-PEDES is a combination of Tri-CUHK-PEDES,
   634	
   635	TABLE IV
   636	A BLATIONS ON THE DESIGN OF V ISION - TO - TEXT C ONVERTING
   637	N ETWORK . l DENOTES THE LAYER OF MLP BLOCKS , AND h MEANS THE
   638	HIDDEN UNITS OF MLP. T HE UNIT FOR “T RAIN PARAMS ” IS ‘M’.
   639	No.
   640	1
   641	2
   642	3
   643	4
   644	5
   645	6
   646	7
   647	8
   648	9
   649	10
   650	11
   651	
   652	Network
   653	Train
   654	Sketch-based
   655	Interactive
   656	Design
   657	Params R@1 mAP mINP R@1 mAP mINP
   658	86.19 28.86 30.92 21.06 33.84 39.26 31.63
   659	Linear only
   660	86.46 34.77 37.72 28.40 45.95 50.97 42.48
   661	l=1, h=512
   662	86.72 36.29 39.20 27.74 49.24 53.06 44.85
   663	l=1, h=768
   664	86.98 37.30 39.65 30.49 50.42 54.98 46.88
   665	l=1, h=1024 87.24 38.31 40.38 30.88 46.79 51.74 44.08
   666	l=2, h=512
   667	86.98 33.97 36.91 28.13 44.43 50.12 42.14
   668	l=2, h=768
   669	87.57 37.76 40.11 30.94 46.71 51.76 43.97
   670	l=2, h=1024 88.29 38.95 40.90 31.29 46.62 50.54 42.08
   671	l=3, h=512
   672	87.24 29.75 33.88 25.72 44.35 49.17 40.82
   673	l=3, h=768
   674	88.16 35.19 37.91 28.94 48.06 52.74 44.21
   675	l=3, h=1024 89.34 34.60 38.37 29.54 45.49 50.11 41.78
   676	
   677	Tri-ICFG-PEDES, and Tri-RSTPReid. Experimental results
   678	demonstrate that the complexity of adaptation datasets is
   679	strongly associated with the model’s generalization ability in
   680	downstream retrieval tasks.
   681	2) Is Vision-to-text Converting Network effective? The
   682	converting network is designed to achieve feature-equivalent
   683	mapping from visual to textual space. To verify the above
   684	motivation, we conduct ablation studies in Table IV from the
   685	following three designs: directly sending visual features into
   686	the text encoder without network converting (No.1), mapping
   687	with a linear-only layer without activation (No.2), different
   688	MLP designs for the converting network (No.3 - No.11).
   689	There are three observations: (1) Learnable parameters are
   690	necessary to map visual features into the textual embedding
   691	space due to the significant modality gap. (2) Non-linear ReLU
   692	activation is critical for enhancing the expressing capability of
   693	the converting network. (3) Considering the balance between
   694	retrieval performance and training efficiency, 1-layer MLP
   695	with 768 hidden units (No.4) is selected as the optimal network
   696	configuration.
   697	We also experimented with other variants, such as varying
   698	the dropout rate, but did not observe any significant improve-
   699	
   700	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:07 UTC from IEEE Xplore. Restrictions apply.
   701	
   702	ments.
   703	3) Does pseudo-word tokens really capture sketch semantics? To analyze the effectiveness of the pseudo-word token
   704	in capturing sketch information, we evaluate the well-trained
   705	model using relative validation datasets. Concretely, we utilize sketches’ pseudo-word tokens as queries to retrieve in
   706	the gallery which solely consists of input sketches. R@1 of
   707	95.64% and R@5 of 99.76% on Market-Sketch-1K demonstrate that the semantic effectiveness of pseudo-word tokens.
   708	D. Qualitative Results
   709	A typical interactive person retrieval process is demonstrated in Figure 3 to verify the practicality of our InteractReID. Firstly, basic sketch-based person retrieval is conducted, which tends to match persons only with similar
   710	structures. When the initial ranking results are returned, our
   711	framework can support users to provide discriminative semantic feedback for interactive and flexible retrieval. Concretely,
   712	when mapping the sketch into its equivalent pseudo-word
   713	token and combining it with feedback tokens in the textual
   714	domain, InteractReID can achieve retrieval ranking refinement.
   715	V. C ONCLUSION AND F UTURE WORKS

exec
/bin/zsh -lc "pdftotext 'Internal-External Context Interaction Network for Person Re-Identification.pdf' - | nl -ba | sed -n '833,1082p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   833	D. Ablation Study
   834	To verify the effectiveness of the key components in the
   835	IEINet and find its optimal structure, we create a baseline
   836	network called Baseline, which comprises the ResNet-50
   837	backbone and the channel branch (Branch-C) from which
   838	the IEI-C module is removed. We perform a wide range
   839	of ablation experiments on the Market1501 and CUHK03Labeled datasets.
   840	1) Determination of the Local Context Range: The hyperparameter k of the IEI-S and IEI-C modules determines the
   841	range of local contextual information extracted for each element (location/channel). To ensure that the unfolding operation
   842	extracts local information with the same neighborhood size
   843	centered on each element for all elements, this parameter can
   844	be set as an odd number such as 3, 5, 7, etc. To determine
   845	the value of parameter k in each module, we construct two
   846	single-branch networks by adding an IEI-S module and an
   847	IEI-C module after the res conv5 layer of Baseline, respectively, and the experimental results of these two single-branch
   848	networks with different values of k are shown in Figure 5.
   849	We can observe that the performances of “Baseline+IEIS” and “Baseline+IEI-C” on Market1501(CUHK03-Labeled)
   850	
   851	TABLE VII
   852	C OMPARISON E XPERIMENTS C ONCERNING THE D IMENSIONAL C OMPRES SION O PERATION
   853	
   854	follow similar trends. As the value of k increases, they decrease
   855	gradually. The mAP of “Baseline+IEI-S” is higher than that
   856	of Baseline for all values of k. When k =3 and k =5, the
   857	mAP of “Baseline+IEI-C” is higher than that of Baseline.
   858	This shows that refining the representations of elements with
   859	the detailed information in the smaller neighborhood can
   860	enhance the feature representation ability, because the smaller
   861	the neighborhood is, the more relevant the elements in the
   862	neighborhood are, and the subtle identification clues can be
   863	enhanced. On the other hand, a larger value of k implies
   864	that the computational cost and number of parameters of
   865	the network are larger. Therefore, based on the impact of
   866	hyperparameter k on the performances and computational costs
   867	of both the IEI-S and IEI-C modules, we finally determine the
   868	value of hyperparameter k to be 3 for both the IEI-S and IEI-C
   869	modules.
   870	2) Validity of the Local Context Modeling: In the IEI-S
   871	and IEI-C modules, we implement a channel/spatial dimension
   872	compression operation on the input feature map to capture
   873	robust intra-sample self-context information and further extract
   874	local context information for each element while reducing
   875	the computational effort of the model. To verify the effectiveness of this operation in these two modules, we use the
   876	convolution operation to replace the pooling operation in the
   877	“Baseline+IEI-S” and “Baseline+IEI-C” attention modules.
   878	“Baseline+IEI-S (w/o CAP)” replaces the CAP operation
   879	in “Baseline+IEI-S” with a 3 × 3 convolution, and the tensor
   880	size output from the convolution is HW × k2 , where a location
   881	feature vector contains information in the 3 × 3 neighborhood
   882	of the location. The aggregation weights are generated by
   883	letting the feature vector at each location interact with the
   884	inter-sample contextual information. Similarly, “Baseline+IEIC (w/o SAP)” replaces the SAP operation in “Baseline+IEI-C”
   885	with a 1D convolution of size 3 that captures the information
   886	of adjacent channels, and the convolution yields a tensor of
   887	size C × k that is used to calculate the aggregation weights.
   888	According to the results in Table VII, we observe that
   889	the Rank-1 accuracy and mAP of “Baseline+IEI-S (w/o
   890	CAP)” and “Baseline+IEI-C (w/o SAP)” exceed those of
   891	Baseline on both person datasets, which suggests that it is
   892	efficacious to use the local context information of elements
   893	to interact with external information to reinforce the feature representation ability. Moreover, on the Market1501 and
   894	CUHK03-Labeled datasets, “Baseline+IEI-S (w/o CAP)” and
   895	“Baseline+IEI-C (w/o SAP)” are inferior to “Baseline+IEIS” and “Baseline+IEI-C” in Rank-1 accuracy and mAP,
   896	respectively. This suggests that the dimensional compression
   897	operation conducted on the input feature map, although simple,
   898	
   899	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
   900	
   901	LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
   902	
   903	741
   904	
   905	Fig. 5. Comparison of the performance of Baseline with IEI-S and IEI-C under different k values. (a) The mAP of IEI-S on Market1501. (b) The mAP of
   906	IEI-S on CUHK03-Labeled. (c) The mAP of IEI-C on Market1501. (d) The mAP of IEI-C on CUHK03-Labeled.
   907	TABLE VIII
   908	C OMPARISON E XPERIMENTS FOR A BLATION S TUDIES I NVOLVING THE
   909	IEIN ET C OMPONENTS
   910	
   911	can offer the IEI modules with more efficient and robust local
   912	contextual information.
   913	3) IEI-S vs. SA(Internal-External Interaction vs. IntraSample Interaction): To further verify the efficacy of internalexternal context interaction, we implement a “Baseline+SA”
   914	network by assembling the SA [18] module behind the
   915	res conv5 layer of Baseline to conduct a fair comparison
   916	with our IEI-S module. The SA module and IEI-S module
   917	refine the feature representation at each location based on the
   918	aggregated weights generated by interaction. The aggregation
   919	weights of the SA module are generated by the interaction of
   920	own information within a single sample. In contrast, our IEI-S
   921	module generates aggregated weights from the interactions
   922	between the potential inter-sample contextual information and
   923	the local contextual information within a sample.
   924	From the results in Table VIII, it can be seen that the
   925	IEI-S module outperforms the SA module by 0.7% Rank-1
   926	accuracy and 0.9% Rank-1 accuracy on the Market1501 and
   927	CUHK03-Labeled datasets, respectively, with 0.54 GFLOPs
   928	
   929	TABLE IX
   930	C OMPARISONS OF M ODEL C OMPLEXITY ON M ARKET 1501. GFLOP S : T HE
   931	N UMBER OF F LOATING P OINT O PERATIONS FOR AN I NPUT I MAGE ;
   932	PARAMS : T HE PARAMETER N UMBER OF THE M ODELS
   933	
   934	less computational volume. This indicates that it is effective
   935	to refine the feature representation of each location by using
   936	the context information between different samples.
   937	4) Comparison of Computational Complexity: Taking the
   938	input of size H × W × C as an example, the computational
   939	complexity of the SA module, the IEI-S module and the
   940	IEI-C module are respectively:
   941	Ω(S A) = 4HWC 2 + 2(HW)2C
   942	Ω(IEI − S ) = HWC 2 + HWk2C + HWk4 , k = 3
   943	Ω(IEI − C) = HWC 2 + Ck2 + HWCk, k = 3
   944	
   945	(11)
   946	
   947	where k is the size of the window and the neighborhood, which
   948	both have a value of 3. The complexity of the SA module is
   949	O((HW)2 ), which is the square of the number of locations HW.
   950	In contrast, both our IEI-S module and IEI-C module achieve
   951	a complexity of O(HW), which is linear to the number of
   952	locations HW.
   953	To further validate the computational efficiency of IEINet,
   954	we compare it with existing methods in terms of Params and
   955	GFLOPs. As shown in Table IX, IEINet significantly reduces
   956	
   957	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
   958	
   959	742
   960	
   961	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   962	
   963	Fig. 6. Heat maps of feature maps. Every four images form a group, in which the activation maps of Baseline, “Baseline+SA” and “Baseline+IEI-S” are
   964	sequentially arranged behind the original image.
   965	
   966	the computational complexity while remaining competitive
   967	with cutting-edge methods. For example, IEINet requires only
   968	66.04M Params and 9.54 GFLOPs, which are 24.2% fewer
   969	Params and 56.8% fewer GFLOPs than IBLSFormer [81]
   970	(87.09 M / 22.07 G), respectively, and IEINet’s mAP is
   971	1.7% less than that of BMCE [82] (98.65 M / 23.81 G).
   972	Compared with BMCE [82] (98.65 M / 23.81 G), the mAP
   973	of IEINet is reduced by 1.7%, but Params and GFLOPs of
   974	IEINet are further reduced by 33.1% and 60.0%. These results
   975	show that IEINet achieves efficient feature learning at a much
   976	lighter computational cost while maintaining competitive performance.
   977	5) Heat Map Visualization Analysis: In Figure 6, we
   978	present the activation maps of the feature maps output by
   979	different networks for some of the same images in the
   980	Market1501 and CUHK03 databases. We have the following
   981	observations. First, Baseline’s heat maps respond at key points
   982	such as shoes, shoulders, and knees, focusing mainly on the
   983	most salient information in the whole image. The heat maps of
   984	the SA module have high responses in the lower body region.
   985	Compared with Baseline, it tends to learn the features of the
   986	pants, ignoring the learning of complex semantics in the upper
   987	body region. Usually, the color and texture of a pair of pants
   988	are consistent across the whole, so the SA module enhances
   989	the representation of this region when aggregating features
   990	based on semantic similarity.
   991	Second, compared with the SA module, the IEI-S module is
   992	able to learn discriminative features not only from the lower
   993	body region but also from the upper body region, and it is
   994	able to learn not only pedestrian semantic information but
   995	also extrinsic semantic information such as backpacks. For
   996	example, the backpacks in groups (b) and (f) are highlighted.
   997	The inter-sample context information in the IEI-S module is
   998	learnt from all samples in a dataset, which indicates differential features between samples in the dataset. Thus, when
   999	intra-sample locale context and inter-sample context interact,
  1000	samples will tend to emphasise cues such as backpacks that
  1001	are relevant to their identities, regardless of whether these
  1002	cues are derived from the semantics of the human body or
  1003	
  1004	not. In addition, since both the generation of weights and
  1005	the aggregation of features in the IEI-S module fully utilize
  1006	the intra-sample local self-context, compared to the Baseline
  1007	and SA module, it also enhances the learning of detailed
  1008	information such as the texture of clothes. For example, the
  1009	texture of the T-shirt in group (e) is highlighted. The above
  1010	comparisons further confirm the validity of the IEI attention
  1011	mechanism.
  1012	6) IEINet Structure Study: The results of the ablation
  1013	experiments regarding the network structure are shown in
  1014	Table VIII. From the table, we can observe that the IEI-C
  1015	and IEI-S modules are effective not only when implemented
  1016	individually on a single branch but also when implemented
  1017	simultaneously on a two-branch network. On the Market1501
  1018	(CUHK03-Labeled) dataset, the mAP of “Baseline+IEI-S” and
  1019	“Baseline+IEI-C” exceeds that of Baseline by 1.2%+(1.0%+).
  1020	The “RB+CB+SB-1” in Table VIII is a two-branch network consisting of ResNet-50 backbone, Branch-C and
  1021	Branch-S1. Compared with the “Baseline+IEI-C”, it achieves
  1022	0.4%+/2.0%+ improvement in Rank-1 accuracy/mAP on these
  1023	two datasets. This is because the amalgamation of IEI-S and
  1024	IEI-C modules can simultaneously learn rich discriminative
  1025	information from two different dimensions. The network “
  1026	IEINet (w/o Lrd )” formed by adding Branch-S2 to “RB +
  1027	CB + SB-1” outperforms the two-branch network, because the
  1028	discriminative regions concerned by the two spatial interaction
  1029	modules are different.
  1030	7) Impact of the Region-Diverse Loss: The ablation experimental results of the region-diverse loss function Lrd are shown
  1031	in Table VIII. IEINet with Lrd outperforms “IEINet (w/o
  1032	Lrd )” on both datasets, especially in Rank-1 accuracy, which
  1033	confirms that the loss can increase the difference between the
  1034	discriminative regions focused on by the IEI-S modules of
  1035	Branch-S1 and Branch-S2, thus helping the network capture
  1036	discriminative information in more important regions.
  1037	Eq. (10) includes a loss weight hyperparameter λ to adjust
  1038	the impact of the region-diverse loss Lrd in the overall
  1039	loss. We carry out experiments on the Market1501 and
  1040	CUHK03-Labeled datasets to study the effect of λ on network
  1041	
  1042	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
  1043	
  1044	LIU et al.: INTERNAL–EXTERNAL CONTEXT INTERACTION NETWORK FOR PERSON RE-IDENTIFICATION
  1045	
  1046	743
  1047	
  1048	Fig. 7. Performance comparison of distinct values of λ. (a) The Rank-1 accuracy and mAP on Market1501. (b) The Rank-1 accuracy and mAP on CUHK03Labeled.
  1049	
  1050	Fig. 8. Retrieval results for six query images. The first three belong to the Market1501 dataset and the remaining three are from the CUHK03-Labeled dataset.
  1051	For each query image, the top-5 retrieved results of both Baseline and IEINet are shown in each row. The image embedded in the green box has the same
  1052	identity as the query, while the image embedded in the red box has a different identity with the query.
  1053	
  1054	performance. We can observe from Figure 7 that the trends
  1055	of Rank-1 accuracy of IEINet on Market1501 and CUHK03Labeled datasets are similar. As the value of λ increases,
  1056	
  1057	the metric first increases, and then gradually decreases after
  1058	reaching the maximum value of 95.5%/78.9% (λ =0.5). We
  1059	can also observe that IEINet achieves the maximum mAP on
  1060	
  1061	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:36 UTC from IEEE Xplore. Restrictions apply.
  1062	
  1063	744
  1064	
  1065	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
  1066	
  1067	both datasets when λ = 0.5, with 88.9% mAP and 76.4%
  1068	mAP on the Market1501 and CUHK03-Labeled datasets,
  1069	respectively. We choose λ = 0.5 as the loss weight of the
  1070	region-diverse loss in the overall loss.
  1071	8) Sorted Visualization Analysis: Figure 8 displays and
  1072	compares the top-5 retrieval results of Baseline and IEINet.
  1073	From the retrieval results, we can see that even though the
  1074	query images are taken in complex real-world environments,
  1075	our IEINet is still able to more accurately identify samples that
  1076	belong to the same person as the query. For example, for the
  1077	third query image, detailed information such as the texture of
  1078	pedestrians’ clothes is an important identity clue. Compared to
  1079	Baseline, IEINet can make the best of this detailed information
  1080	to better identify the samples. For the fifth and sixth query
  1081	images, the backpack is also an important identity cue, and
  1082	IEINet can better identify the person images based on this

codex
证据段里有个明显分化：有些论文真的用消融直接验证视角，例如 InteractReID 问“伪词是否捕获草图语义”，IEINet 比较自注意力和内部外部交互；也有论文更像堆模块涨点，视角证明弱一些。现在补读后三篇的消融和可视化。
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
/bin/zsh -lc "pdftotext 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|DDO|HRR|inpainting|re-ranking|feature-space|clean|occluded|Table|Fig\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
6:Latent Diffusion-Guided Feature Inpainting for Occluded Person
7:Re-Identification With Hybrid Re-Ranking
16:the corrupted feature representation itself remains occluded. To
19:(DDO) that learns to inpaint corrupted feature embeddings
20:and reconstruct clean, identity-preserving representations in the
26:known at test time, we introduce a Hybrid Re-Ranking (HRR)
27:scheme. HRR directly addresses the limitations of standard
28:re-ranking by leveraging centroid-based identity anchors to
29:refine k-reciprocal re-ranking, thus boosting retrieval precision
34:validating the effectiveness of de-occluding features rather than
38:requiring neither explicit occlusion masks nor paired cleanoccluded images. By removing occlusion-induced artifacts, the
40:To achieve our goal of feature inpainting, we introduce a latent
41:Diffusion guided De-Occluder (DDO), a generative framework that reconstructs clean identity-preserving embeddings
42:from partially occluded inputs. Unlike previous approaches
48:discriminative under severe visual clutter.
57:two fundamental limitations remain: (i) the heavy reliance on Fig. 1: Illustration of occlusion-induced distortion in feature
58:suppression strategies that make downstream models resistant space. Each colored dot represents a clean feature embedding
59:to occluded features, rather than explicitly removing the Fclean , while the corresponding x mark within the dashed radius
60:occlusion-induced corruption at the source, and (ii) naïve denotes its occluded counterpart Fo . The distortion is modeled
61:(T ) from Fclean to Fo , which our method
76:tackle occlusion by reconstructing clean feature embeddings
78:from occluded inputs in the latent space. Treating occlusion (DDPM) [9] and Denoising Score Matching (DSM) [10] that
79:as structured noise as illustrated in Figure 1, we formulate the learn denoising in an unsupervised manner, DDO leverages
80:problem as a Feature Inpainting task and employ a diffusion- explicit paired supervision between clean and occluded features,
87:a distribution, making it uniquely suited for feature-space
90:inpainting in ReID where identity consistency is paramount.
100:Notably, DDO is designed as a plug-and-play module that
103:Second, we propose a novel Hybrid Re-Ranking scheme
104:(HRR) that directly leverages the closed-world nature of the
115:representations [12], [13]. Building on this, HRR computes
117:re-ranking to centroid-curated candidate sets, thereby reducing
119:re-ranking as a centroid-guided refinement over the manifold
124:datasets such as Occluded-Duke [16] and Occluded ReID [17]
126:Average Precision (mAP) and Cumulative Matching Characteristic (CMC) Rank-1 scores, validating the effectiveness and
130:inpainting task, wherein occluded regions in the latent space are reconstructed to recover clean, identitypreserving feature embeddings. These refined features are
133:• We introduce the first latent Diffusion guided DeOccluder (DDO) module for person ReID, capable
135:occluded inputs in a fully self-supervised manner. DDO
138:• We propose a Hybrid Re-Ranking (HRR) algorithm
140:re-ranking process, improving retrieval precision and
141:efficiency. This unified retrieval framework achieves stateof-the-art results on multiple standard and occluded ReID
144:datasets we have trained and tested our model on (Market1501, CUHK-03, MSMT17, Occluded-Reid, Occluded
166:mechanism for occluded features.
169:occluded samples to improve robustness. While visually plausible, these methods are prey to mode collapse and vanishing
170:gradients [49], [50] and lack strong feature-space guidance
175:framework for feature-space reconstruction, where iterative
176:denoising [9] enables fine-grained inpainting of corrupted
177:embeddings aligned with the clean feature distribution.
179:for re-ranking [53] improve ranking quality by exploiting local
187:landscape of occlusion handling, Table I categorizes the major
193:that reconstructs clean feature representations before they
196:level attention mechanisms. We provide explicit feature-space
197:Early person ReID methods largely used CNN backbones. guidance through diffusion-based inpainting, not implicit
208:TABLE I: Principles of different methods for handling occlusion in person ReID. The Comments column contrasts each approach
209:with our latent Diffusion based De-Occluder (DDO) and Hybrid Re-Ranking (HRR) formulation.
245:Motion/spatio-temporal cues to suppress occluded crops
294:MSMT17, Occluded-ReID
297:via attention/masking; embeddings remain occluded at test time; still operate on corrupted
308:feature-space supervision limits matching improvements.
310:DDO - Explicit de-occlusion in feature
311:space. HRR - Centroid based filtering
318:OccludedReID,
319:Occluded-Duke,
324:Learns explicit inverse transform to reconstruct clean embeddings from occluded features before ReID. Plug-and-play module; no
326:training. Novel algorithm to restrict re-ranking
332:DDO
334:Inpainting
341:HRR
346:re-ranking
348:Fig. 2: Methodology Overview: Image features Fo are extracted using a ViT encoder and processed through the pretrained
349:DDO module to reconstruct clean feature embeddings Fclean . The downstream ReID module utilizes Fclean to compute a
350:preliminary distance matrix, which the HRR module subsequently reranks to produce the final distance matrix for retrieval.
352:formulation enables DDO to function as a plug-and-play
355:Our approach models occluded and unoccluded features as
357:that use centroids for direct retrieval, HRR formulates a novel distinct but correlated representations in the latent space. Rather
359:pruning to construct high-confidence subgraphs before local k- distortions, we explicitly learn the transformation from clean
360:reciprocal re-ranking [51]. This hybrid approach mathematically to corrupted features caused by occlusion. This enables the
361:guarantees denser reciprocal neighborhoods with reduced false reconstruction of identity-preserving embeddings by inpainting
364:problem inherent in standard re-ranking when gallery sets
365:exhibit high intra-class variance. Together, these contributions in Figure 2, directly addresses partial occlusions using a Latent
366:yield consistent state-of-the-art performance across all evaluated Diffusion guided De-Occluder (DDO). Given an occluded
367:benchmarks, validating the effectiveness of explicitly removing feature embedding Fo extracted by the ViT encoder, the DDO
368:occlusion corruption rather than adapting models to tolerate it. module learns a mapping Fclean = DDO(Fo ) where Fclean
369:approximates the clean, unoccluded embedding. This mapping
381:backbone architecture, making DDO a plug-and-play module. using a trainable noise predictor ϵθ . After T timesteps, this
383:Re-Ranking (HRR) as follows: given a gallery set G with recover F̂clean .
384:The original clean features F from a frozen ViT encoder
391:for k-reciprocal re-ranking, rather than relying solely on noisy
392:Lrecon = λ∥Fclean − F∥22 + (1 − λ) (1 − cos(Fclean , F)) , (2)
395:ranking is applied to refine top matches. Together, DDO and fidelity.
396:HRR form a compact, effective system achieving superior
405:A. Feature Inpainting via Diffusion guided De-Occluder (DDO)
409:To ensure semantic alignment with clean identity features, we
411:DDO training loss is thus:
415:but do not remove the underlying feature-space corruption. We
416:address this by formulating occlusion as a feature inpainting
417:LDDO =
421:problem that explicitly recovers clean, identity-preserving
425:embeddings from occluded counterparts.
430:feature space, where clean identity-preserving features, denoted term.
431:by Fclean , are transformed into occlusion-corrupted counterparts
446:clean
449:DDO
450:in Figure 1, where each colored dot represents a clean feature
452:embedding Fclean , and the corresponding x mark within the
453:This formulation establishes DDO as the operational realizadashed circle depicts its occluded version Fo = T (Fclean ).
457:inpaint corrupted embeddings, DDO provides downstream
458:maps corrupted features back to their clean form:
459:ReID modules with clean, occlusion-free priors, thereby
460:F̂clean = T −1 (Fo ),
462:While DDO is built upon the latent diffusion framework, its
463:where F̂clean is the reconstructed embedding used for downtraining paradigm differs fundamentally from unsupervised
468:models, where clean data x0 is gradually corrupted by noise
469:the clean distribution q(x0 ); likewise, DSM learns the score
473:In contrast, DDO leverages explicit paired supervision. Clean
477:occluded features Fo . The combined loss LDDO (Equation 4)
486:clean priors rather than requiring them to be occlusion-aware.
488:sion guided De-Occluder (DDO), a generative module that
490:reconstructs clean features from Fo in the latent space. As
491:must recover a specific clean embedding to preserve identity.
492:illustrated in Figure 3, Fo is first projected into a compact
493:This supervised paradigm distinguishes DDO from generic
497:feature inpainting. To highlight the distinct nature of our task
503:entire ReID system with DDO to DDPM [9] and DSM [56]
507:in Table II. We train the model using synthetically generated
534:Occluded
539:Cleaned
542:Unoccluded
545:Unoccluded
558:DDO
560:Occluded
566:Fig. 3: Latent Diffusion guided De-Occluder (DDO) Module Training. The ViT encoders are kept frozen throughout training
569:Equation 2), where the unoccluded feature map F serves as the ground-truth. This combined training objective (See Equation 4)
570:enables DDO to effectively learn to reconstruct clean and identity-preserving feature maps from their occluded counterparts.
572:occluded images constructed via silhouette-guided background
575:TABLE II: Comparison of DDO efficiency with standard
576:diffusion models. DDO is designed for feature-space inpainting,
582:Stable Diffusion [8]
585:Baseline + DDO (Ours)
606:heads. We integrate it with our pretrained frozen DDO and
610:occluded conditions and to separate identity features effectively
614:E. Simultaneously, the image is passed through DDO that has
615:been trained to reconstruct clean embeddings from occluded
616:features (see Section III-A). The Clean Feature Map Fclean
617:and the Occluded Feature Map Fo are fused via a multi-head
619:Q = Fclean WQ , K = Fo WK ,
695:DDO
697:Occluded
700:Cleaned Feature
735:Fig. 4: Overview of our ReID architecture: The DDO module parameters (trained separately on a synthetically occluded
770:under occluded and cluttered visual conditions.
772:C. Hybrid Re-Ranking (HRR)
773:Standard re-ranking methods [51] compute pairwise distances
776:We propose Hybrid Re-Ranking (HRR), combining robust
778:By restricting re-ranking strictly to semantically relevant
779:identity clusters, HRR simultaneously reduces computational
783:graph for robust matching. This method is explained in Figure
788:for re-ranking. Lines between query and centroids are labeled
791:Fig. 5: Illustration of centroid-guided gallery filtering for
792:Hybrid Re-Ranking (HRR).
795:for Jaccard-based re-ranking.
796:1) Theoretical motivation for Centroid filtering and HRR:
798:why HRR yields a superior graph structure for retrieval
799:compared to standard re-ranking approaches.
803:Proof: Figure 5 provides an intuitive visualization of our
812:Geometrically, this procedure restricts re-ranking to identity
832:Standard re-ranking computes the Jaccard distance dJ (q, g)
846:Numerical illustration. From Figure 5, the query q lies at
857:We construct Cq = {A, B, C} and restrict re-ranking to gallery
873:In HRR, since only identities A, B, C are retained:
900:Re-ranking can be interpreted as transductive inference over a
902:In standard re-ranking, the graph includes all gallery samples,
903:leading to cluttered and noisy neighborhoods. In HRR, we
924:datasets (Table III), consistently outperforming state-of-theart approaches. These results validate the effectiveness of our
925:latent Diffusion-based De-Occluder (DDO) and Hybrid ReRanking (HRR) scheme in robustly improving accuracy under
947:Stage 2: Local Re-ranking within Filtered Gallery:
983:Thus, HRR increases the intra-cluster cohesion and reduces
984:inter-identity noise, yielding a more robust re-ranking graph and
986:more semantically meaningful subgraphs, improving re-ranking
989:re-ranking in this manner.
990:The HRR algorithm has two stages as enumerated below.
992:TABLE III: Datasets used for training and evaluation. Datasets
1006:Occluded-ReID [17]
1007:Occluded-Duke [16]
1099:gallery. This is the general surveillance scenario which evaluated benchmark datasets, consistently outperforming exallows the presentation of unknown individuals to the isting approaches on both occluded and holistic Re-ID datasets.
1101:The quantitative results are summarized in Tables IV (Protocol
1105:individuals and allows for the use of identity-aware Occluded-Duke (+15.86% mAP open-set, +12.5% mAP closedretrieval strategies, including centroid-based retrieval and set). The consistent improvements in closed-set evaluation
1106:prototype-guided re-ranking to enhance retrieval accuracy further confirm that HRR effectively leverages identity-level
1109:The efficacy of the proposed DDO and HRR modules is
1112:identification is conventionally treated as a closed-set problem, dimensions. Table VI validates their effectiveness on established
1113:we introduce our second contribution—Hybrid Re-Ranking benchmark ReID backbones, specifically TransReID [27] and
1114:(HRR)—within this setting to effectively leverage the available Pass-ViT [65]. Table VII quantifies the contributions of DDO
1115:identity annotations during retrieval. We categorize general (feature-space de-occlusion) and HRR (centroid-guided rere-ranking methods (e.g., Jaccard) [51] as open-set protocols, ranking). The combined DDO+HRR configuration consistently
1119:To ensure a fair comparison, we present our results under their complementary nature: DDO provides clean priors while
1120:these two scenarios separately. All prior benchmarks that do HRR exploits closed-set gallery structure. Cross-dataset evalunot incorporate identity-level knowledge during retrieval are ation results are presented in Table VIII, where models trained
1121:grouped under the open-set protocol (Table IV). Given that on Market1501 and CUHK03 undergo zero-shot evaluation
1122:centroid-based retrieval is a relatively recent development, we across five distinct ReID datasets. DDO + HRR consistently
1124:set protocol. Accordingly, these results (Table V) includes domains. This generalization stems from two factors. First,
1130:(CMC) Rank-1 accuracy and mean Average Precision (mAP). mitigates domain-specific camera biases, allowing DDO to
1132:the ranking quality and overall retrieval effectiveness of our holistic analysis of the computational overhead with subsequent
1138:D. Ablation Studies
1141:a) Effect of DDO: Our DDO module reconstructs clean
1142:(patch16, 224) architecture initialized with standard ImageNet latent feature embeddings from occluded inputs, trained on
1145:dataset without configuration tuning. So, for example, the ideal work [75], [81], long-range retrieval relies mainly on clothing
1146:Top-N for Market = 90 (See Figure 8) but our reported results appearance and coarse pose cues rather than facial biometrics.
1150:3080 GPU, equipped with 10 GB of VRAM. The final ReID ReID. Therefore, DDO learns to target latent-space inpainting
1151:model, after construction but before data loading, occupies of occluded regions to preserve semantic fidelity, making it
1152:approximately 1.014 GB of memory. For the Market-1501 inherently well suited for occluded person ReID.
1153:dataset, each training iteration with a batch size of 32 takes apTo evaluate the effectiveness of DDO (De-Occluder), we
1155:unchanged for all datasets, the compute specifications remain pipeline and our DDO-enhanced approach in Figure 6. Ocuniform throughout. To facilitate full reproducibility, Section clusion in the baseline model distorts identity cues, leading
1156:II of the Supplementary Material details all hyperparameters, to attention on irrelevant regions and retrieval errors. DDO
1158:reconstructs features approximating unoccluded counterparts,
1168:TABLE IV: Comparison of state-of-the-art person re-identification models across multiple datasets (Protocol 1 - OS). For
1315:Occluded-ReID
1317:Occluded-Duke
1697:DDO (Ours)
1728:TABLE V: Comparison of state-of-the-art person re-identification models across multiple datasets (Protocol 2 - CS). Empty
1746:Occluded-ReID Occluded-Duke
2189:DDO & HRR (Ours)
2192:regions. For example, DDO restores focus to distinctive lowerbody clothing in the first case, redirects attention from a
2198:attending indiscriminately to background and occluded regions.
2199:DDO correctly identifies the white wall as an occlusion,
2201:parts) for accurate retrieval. These results demonstrate DDO’s
2222:Figure 7(a) illustrates the variation of Mean Squared Error
2228:We evaluate structural similarity (SSIM) between occluded
2229:feature maps Fo and unoccluded features F before and after
2230:applying DDO (Fclean ). For each occluded input, we extract
2231:ViT features and compute SSIM against the clean embedding
2233:occluded input through the DDO (Fclean and F ). Figure 7
2234:(b) plots results over 100 held-out Occluded SHHQ test
2235:images (unseen during DDO training). The x-axis shows
2246:TABLE VI: Ablation study showing the effect of incorporating the proposed DDO and HRR into existing ReID pipelines
2263:Occluded-Duke
2267:Occluded-ReID
2276:TransReID + DDO
2277:TransReID + DDO + HRR
2328:Pass-ViT + DDO
2329:Pass-ViT + DDO + HRR
2380:conditions. The addition of HRR further boosts retrieval
2381:accuracy by effectively guiding the re-ranking process, thereby
2382:establishing DDO as a robust feature inpainting module and
2383:HRR as a powerful complement for retrieval refinement.
2385:with increasing values of top-N , reaching a peak at a datasetdependent value before plateauing, as shown in Figure 8.
2387:the structure of the centroid-based hybrid re-ranking approach.
2391:on re-ranking within this subset GN . A larger gallery implies
2395:intra-class samples or the best-quality matches for robust reFig. 6: Qualitative comparison of attention maps and retrieval
2397:results. (a) Probe, (b) Heatmap without DDO, (c) Heatmap
2398:relevant positive samples before re-ranking.
2399:with DDO, (d) the Rank-1 retrieval without DDO, and (e)
2401:the Rank-1 retrieval with DDO. Green and red boxes indicate
2410:leaving minimal improvement room; DDO gains are small. explaining the peak performance at larger N in such settings.
2412:contrast, DDO-enhanced features (Fclean vs. F ) maintain ranking performance. Figure 7 (c) shows that larger galleries
2417:We further validate the effectiveness of our modules by alignment. Occluded-Duke exhibits the most pronounced
2418:incorporating DDO and HRR into two representative ViT- improvement: it has the largest gallery and highest sample
2420:cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
2421:to obtain de-occluded embeddings, which are then enhanced heuristic for optimal N selection is present in Section V of the
2423:downstream model, while HRR refines retrieval by leveraging
2424:gallery centroids during re-ranking. As shown in Table VI, E. Limitations
2425:DDO consistently improves performance across all datasets,
2427:with particularly large gains on occlusion-heavy benchmarks proposed DDO and HRR frameworks. Identifying these boundsuch as Occluded-Duke and Occluded-ReID, confirming its ary conditions provides a holistic assessment of our algorithm’s
2437:Fig. 7: (a) Plot showing the Mean Squared Error (MSE) and Cosine Error between reconstructed and original features across
2438:varying numbers of diffusion steps in the denoising process. (b) Comparison of SSIM similarity between occluded and clean
2439:features, before and after applying the DDO module. (c) Relationship between gallery size and mAP gain achieved through
2440:re-ranking across five ReID datasets. A linear trend is fitted, and Spearman correlation is reported to quantify the dependency.
2441:TABLE VII: Comparison of mAP scores with and without DDO and HRR. Bold represents best results and underline represents
2442:the second best. "Baseline" represents our ReID model without DDO or HRR.
2451:Occluded-ReID
2453:Occluded-Duke
2488:Baseline + HRR
2489:Baseline + DDO
2490:Baseline + DDO + HRR
2492:TABLE VIII: Cross-dataset evaluation. Each cell shows mAP and Rank-1 (%) for models trained on the specified dataset and
2493:evaluated on the target datasets. Bold represents the best results. "Baseline" represents our ReID model without DDO or HRR.
2525:Baseline + HRR
2526:Baseline + DDO
2527:Baseline + DDO + HRR
2592:Baseline + HRR
2593:Baseline + DDO
2594:Baseline + DDO + HRR
2658:1) DDO: Limitations of Feature Reconstruction: While the
2659:proposed Latent Diffusion guided De-Occluder (DDO) effectively recovers identity-preserving representations under partial
2668:sensor noise. Extending the feature inpainting mechanism to
2669:handle these compounded, multi-factor visual degradations
2671:2) HRR: Limitations of Centroid based Filtering: When
2672:gallery samples per identity are few and heavily occluded,
2674:causing minimally occluded queries to align closer to
2675:less-occluded identities and leading to retrieval errors. This
2677:is evident in Table VII on Occluded-ReID (mAP drops from
2678:92.3% to 90.1%) and Table VIII on Partial-ReID (Rank-1
2690:no of unoccluded features after occlusion. Then, the number
2735:validated in Table VII, where HRR significantly improves
2736:performance on Occluded-Duke (1000 gallery identities, mAP:
2737:81.8% → 90.6%), but exhibits diminished effectiveness on
2738:Occluded-ReID (100 gallery identities, mAP: 92.3% → 90.1%),
2744:distortion in the latent feature space and learnt an inverse mapping to recover clean embeddings. We framed the problem as a
2745:feature-level inpainting task. This novel approach has enabled
2749:have proposed a latent Diffusion guided De-Occluder (DDO)
2752:Through extensive experiments on multiple occluded and
2755:reconstructing clean or pseudo-clean feature representations
2756:from occluded inputs. Additionally, we have introduced a
2757:Fig. 8: Retrieval Performance (mAP) vs top-N centroid filtering novel Hybrid Re-Ranking (HRR) scheme that successfully
2758:for different datasets during the hybrid re-ranking.
2760:confirmed the effectiveness of both the de-occlusion module
2762:and the re-ranking strategy. Additionally, the proposed diffusionLet each gallery feature gi for identity c be expressed as: based feature inpainting framework extends naturally to other
2763:gi = µc + ϵi + δi where µc is the clean identity-preserving occlusion-prone tasks such as object detection, segmentation,
2764:embedding. ϵi ∼ (0, σ 2 I) represents zero-mean random noise. and tracking, where reconstructing clean embeddings could
2768:adaptations of DDO and hybrid centroid-based refinement
2819:Consequently, a minimally occluded query may be closer in
2834:is reflected in our experiments as shown in Table VII on
2835:Occluded-ReID, where the mAP drops from 92.3% (without
2836:HRR) to 90.1% (with HRR), and Table VIII on Partial-ReID
2837:where the Rank-1 retrieval drops from 58.3% (without HRR) to
2838:56.7% (with HRR). As occlusion becomes more severe, larger
2855:them all: Occlusion-aware attention network for occluded person re-id,”
2882:effectiveness of centroids in image retrieval,” in Proceedings of the 28th
2896:information matters: Learning relation and topology for occluded person
2900:“Occluded person re-identification,” in 2018 26th ACM International
2913:[21] X. Liu, J. Guo, H. Chen, Q. Miao, Y. Xi, and R. Liu, “Adaptive occlusionaware network for occluded person re-identification,” IEEE Transactions
2917:and noising defense network for occluded person re-identification,” IEEE
2921:semantic perception and cnn-transformer hybrid network for occluded
2925:enhancement for occluded person re-identification,” IEEE Transactions on
2980:based on 3d occupancy estimation for occluded regions,” in Proceedings
3001:[47] H. Huang, W. Yang, X. Chen, X. Zhao, K. Huang, and J. Lin, “Adversarially occluded samples for person re-identification,” in Proceedings
3005:for occluded person re-identification,” in Proceedings of the IEEE/CVF
3013:[51] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with k-reciprocal encoding,” in Proceedings of the IEEE
3027:for re-ranking person re-identification,” IEEE Transactions on Image
3081:[69] L. He, J. Liao, Z. Zhang, C. Li, Z. Yang, and T.-S. Chua, “Foregroundaware pyramid reconstruction for alignment-free occluded person reidentification,” in Proceedings of the IEEE/CVF International Conference
3093:[73] X. Cheng, M. Jia, Q. Wang, and J. Zhang, “More is better: Multisource dynamic parsing attention for occluded person re-identification,”
3106:learning framework for human-centric visual tasks,” in Proceedings of
3113:and S. W. Baik, “Visual appearance and soft biometrics fusion for person
3146:in Intelligent Systems (1998-2019) and the Visualization and Intelligent
3158:and video databases, graphics and visualization, robotics, human-computer

exec
/bin/zsh -lc "pdftotext 'Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|granularity|MGFE|MGFT|CNC|saliency|nearest|Table|Fig\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:Learning multi-granularity representation with transformer for
36:Multi-granularity
42:and positions, e.g., the global color of the cloth, the body’s local pose, and the shoe’s pixel size. However, existing methods mainly capture features at a single granularity, ignoring multi-granularity information contributing
43:to pedestrian identification. Therefore, we propose a cross-modality multi-granularity Transformer (CM2 GT)
46:alignment caused by matching single granularity features. Specifically, we first design a multi-granularity
47:feature extractor (MGFE) module based on Transformer to capture the global-patch-pixel level features of each
48:modality, which can flexibly represent semantic information at multiple scales. Secondly, a multi-granularity
49:fusion Transformer (MGFT) module mines the hierarchical relationships between multi-granularity features
50:by a saliency-enhanced Transformer, which ensures the identity-wise saliency consistency across different
52:space, we design a cross-modality nearest-neighbor clustering (CNC) loss function to minimize the distance
53:between the anchor sample and its cross-modality nearest neighbor. Extensive experiments demonstrate that
65:multi-granularity features are extracted exclusively from RGB images,
68:both intra-modality and inter-modality multi-granularity features, in
71:For the VI-ReID task, it is critical to extract multi-granularity features. On the one hand, when two pedestrians only have a back
72:view, and the height and gender are the same, these multi-granularity
87:Fig. 1. (a) Global-level based methods capture the most salient region, (b) Part-level based methods consider part features, (c) Our approach combines multi-granularity features
90:granularity features can complement each other to contribute to the
91:identification of pedestrians when the features of the single granularity
94:extremities. Multi-granularity features can facilitate the discrimination
98:However, existing methods heavily rely on single-granularity feature extraction, and may result in irrelevant matching and ambiguous
101:leads to irrelevant matching. As shown in Fig. 1(a-1), in the visible
106:divide multiple body parts and results in ambiguous alignment in crossmodality matching. In Fig. 1(b-1), the first stripes of both the NIR
111:In this paper, we propose utilizing flexible multi-granularity features to solve these problems. First, multi-granularity information
112:is extracted for multi-level matching, instead of relying on singlegranularity features which may be semantically different. Second, the
113:cross-modality multi-granularity alignment could flexibly align each
114:region in hierarchical levels, regardless of the inflexible assignment resulting from uniform division. Generally, as shown in Fig. 1(c), in both
117:we propose the cross-modality multi-granularity Transformer (CM2 GT)
118:framework, including the multi-granularity feature extractor (MGFE)
119:module and multi-granularity fusion Transformer (MGFT) module.
121:from two modalities, MGFE simultaneously captures global-level feature by global max pooling, and patch- and pixel-level features by
122:Transformer, to adequately obtain multiple granularities. MGFT module mines the hierarchical aggregation relationship between patch-level
123:and pixel-level features by interaction with multi-granularity features,
125:enhancing the discriminative ability of features. Specifically, we design a saliency-enhanced Transformer which preserves the saliency
130:than a predefined threshold, these loss functions provide weak constraints, resulting in less effective clustering of cross-modality intraclass samples. In this paper, we design a cross-modality nearestneighbor clustering (CNC) loss function to cluster the intra-class crossmodality samples. Specifically, we obtain the intra-class cross-modality
131:nearest-neighbor of the anchor samples by computing the similarity
134:sample and nearest-neighbor sample.
136:(1) To the best of our knowledge, it is an early work on combining multi-granularity features based on Transformer structure in the
138:(2) We design a cross-modality multi-granularity Transformer
139:(CM2 GT) framework to capture multi-granularity features containing
142:(3) The cross-modality nearest-neighbor clustering (CNC) loss function is proposed to reduce the difference among intra-class crossmodality samples and promote more compactness among them.
168:modalities. In this paper, we propose a cross-modality multi-granularity
172:In this section, we introduce cross-modality multi-granularity Transformer (CM2 GT) framework for VI-ReID. Firstly, we give a brief
173:overview of the network structure. Secondly, we present a multigranularity feature extractor (MGFE) to capture global-, patch-, and
174:pixel-level features for each modality. Thirdly, a multi-granularity
175:fusion Transformer (MGFT) is introduced to fuse cross-modality information and interact with multi-granularity features. Finally, we
176:propose a cross-modality nearest neighbor clustering (CNC) loss function to make the distance between intra-class cross-modality samples
195:These main methods effectively extracted single granularity features
196:by focusing on coarse or fine regions, leading to irrelevant matching and ambiguous alignment. In this paper, we capture multiplegranularity features based on the Transformer structure to solve these
200:The overview of our approach is shown in Fig. 2. Firstly, in the feature extractor, ResNet50 [20] pre-trained on ImageNet [21] is adopted
204:𝑋 𝑣𝑖𝑠 and near-infrared modality feature 𝑋 𝑛𝑖𝑟 are obtained by a twostream network. Secondly, MGFE module utilizes 𝑋 𝑣𝑖𝑠 and 𝑋 𝑛𝑖𝑟 to
225:MGFT fuses multi-granularity features (𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑎𝑡𝑐 ℎ , 𝑋 𝑝𝑖𝑥𝑒𝑙 , 𝑋 𝑝𝑖𝑥𝑒𝑙 )
229:3.2. Multi-granularity feature extractor
230:As shown in Fig. 2, the multi-granularity feature extractor (MGFE)
234:to capture pixel granularity features, which focuses on finer and richer
236:other local parts. These three-level granularity features can complement
239:by parameter-sharing MGFE. Considering the processing of features
240:from both modalities is similar in each branch of MGFE, the
243:2.3. Multi-granularity in computer vision field
244:In the field of computer vision, multi-granularity based methods
245:have achieved better performance. For example, In terms of multigranularity feature capture, [14] elucidated skeletal dynamics within
246:various contexts. Our approach extracts features across the granularity,
248:and comprehensive data feature map. At the multi-granularity feature
255:As shown in Fig. 2, given visible modality feature 𝑋 𝑣𝑖𝑠 , we utilize
258:capture global granularity features 𝑋 𝑔 𝑙𝑜𝑏𝑎𝑙 , as follows:
270:propose a saliency-enhanced Transformer to preserve the saliency consistency across modalities and granularities and perform cross-modality
271:cross-granularity features alignment.
282:patch-level granularity feature 𝑋 𝑝𝑎𝑡𝑐 ℎ .
305:Fig. 2. Illustration of our cross-modality multi-granularity framework, including multi-granularity feature extractor (MGFE), multi-granularity fusion Transformer (MGFT), and total
306:loss function. MGFE consists of global, patch, and pixel branches, which aims to capture three-level granularity features. MGFT interacts feature pairs in the patch branch and
334:capture pixel granularity features, and refer each pixel as a patch, so
382:to obtain spatially-aware pixel granularity representation 𝑥𝑣𝑖𝑠
387:granularity features is 𝑋 𝑝𝑖𝑥𝑒𝑙 , as follows:
417:3.3. Multi-granularity fusion transformer
418:The ambiguous alignment in single-granularity based methods can
420:a multi-granularity fusion Transformer (MGFT) module that effectively aligns features across multiple modalities and granularities with
421:saliency-enhanced Transformer.
422:The modality and granularity discrepancy results in different regions of interest. To preserve the saliency consistency across modalities
423:and granularities, we propose a saliency-enhanced Transformer including saliency-consist attention (SCA) component different with the
425:in Fig. 3.
464:granularity.
465:For the general saliency-consist attention, the query input 𝑋𝑞′ is
481:Fig. 4. The illustration of the CNC loss function.
483:Through MGFE and MGFT, we obtain the final output feature 𝑋𝑜𝑢𝑡𝑝𝑢𝑡
490:Fig. 3. The illustration of the saliency-consist attention.
508:granularities, we use the average value of spatial tokens as saliency
511:3.4. Cross-modality nearest-neighbor clustering loss function
517:modality discrepancy. Therefore, we design a cross-modality nearestneighbor clustering loss function 𝐿𝑐 𝑛𝑐 including neighbor consistency
520:sample and its nearest-neighbor. 𝐿𝑚𝑎 measures the deviation between
522:As shown in Fig. 4, for each sample, we first find the neighboring
530:∈ 𝑋𝑜𝑢𝑡𝑝𝑢𝑡 , the nearest-neighbor
545:nearest-neighbor samples.
556:saliency-enhanced attention
560:value space respectively. The output 𝑍 ′ of a saliency-consist attention
568:Thus, the output 𝑂 of saliency-enhanced Transformer (ST) is defined
574:We use saliency-enhanced Transformer to fuse inter-modality intragranularity and intra-modality inter-granularity features as:
601:inter-modality intra-granularity fusion intra-modality inter-granularity fusion
648:The index 𝑡 of the nearest-neighbor sample of anchor sample 𝑋
659:By this way, the cross-modality nearest-neighbor sample 𝑋
678:Table 1
682:and nearest-neighbor by 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥 function.
789:Table 2
883:The total loss function contains the identity loss function 𝐿𝑖𝑑 , tripletcenter loss function 𝐿𝑡𝑐 [4,5,22], and cross-modality nearest-neighbor
920:the MGFE parameters 𝛩2 and MGFT parameters 𝛩3 .
935:MGFE. // (1)-(8)
937:Obtain 𝑂𝑔𝑙𝑜𝑏𝑎𝑙 , 𝑂𝑝𝑎𝑡𝑐 ℎ , and 𝑂𝑝𝑖𝑥𝑒𝑙 by MGFT. // (9)-(13)
963:parameters. In Tables 1 and 2 we conduct experiments on SYSU-MM01
966:From Tables 1 and 2, we note that when set to 𝑠4 , the performance
967:decreases in both datasets. This occurs because, in this configuration,
986:Table 3
1177:Table 4
1368:Table 5
1477:SYSU-MM01 and RegDB datasets. In Table 5, experimental results show
1483:best performance are achieved when depth is set to 1. In Table 6, the
1489:number of heads, and the head embedding dimensions. In Table 3,
1495:information. In Table 4, this occurrence is similarly observed in the
1498:4.3. Ablation study
1499:4.3.1. Effectiveness of each module
1500:To verify the effectiveness of each module (MGFE, MGFT, CNC) in
1501:our approach, we conduct ablation experiments on the SYSU-MM01
1504:Table 7 lists the results in all-search and indoor-search modes. In
1505:Index 1 and Index 2, the results of MGFE are 6.26% and 4.47% higher
1507:improvement shows that compared to using single-granularity features,
1517:Table 6
1622:Table 7
1628:MGFE
1630:MGFT
1632:CNC
1733:Table 8
1739:MGFE
1741:MGFT
1743:CNC
1844:Table 9
1845:Evaluation of each granularity on the SYSU-MM01 dataset.
1948:multi-granularity features. Comparing Index 2 with Index 3, these improved results prove that fusing features between multiple granularities
1950:Index 2 with Index 4, CNC loss function can effectively make distance
1952:our approach includes MGFE, MGFT, and CNC, the performance yields
1953:69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
1954:As shown in Table 8, we find that significant improvements also occur
1960:in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
1962:between multi-granularity features can reduce the irrelevant matching and ambiguous alignment problems caused by single-granularity
1965:In Tables 11 and 12, we have added the results of testing at different
1967:granularity for testing is lower than that of using multiple granularity fusion. This phenomenon can be attributed to the fact that each
1968:granularity only contains partial discriminative information related to
1973:By fusing these multi-granularity features, our approach can more
1976:4.3.2. Effectiveness of three-granularity features
1977:We evaluate the effectiveness of using a combination of global-,
1978:patch-, and pixel-level features. In Tables 9 and 10, ‘‘Global’’, ‘‘Patch’’,
1981:Table 9 reports the results on the SYSU-MM01 dataset in all-search
1984:than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity
1990:Table 10
1991:Evaluation of each granularity on the REGDB dataset.
2093:Table 11
2257:Table 12
2293:Fig. 5. Performance (%) comparison with different values for parameters 𝛼 and 𝛽 on the SYSU-MM01 and RegDB datasets.
2296:outline features using the nearest-neighbor algorithm.
2300:𝛽 on the SYSU-MM01 and RegDB datasets. In Fig. 5(a), we increase
2303:achieved by setting 𝛼 = 3.0. Meanwhile, in Fig. 5(b), we fix 𝛼 = 3.0,
2306:In Fig. 5(c)(d), when 𝛼 = 3.0 and 𝛽 = 0.3, our approach also achieved
2309:4.5.2. Evaluation of different metrics in CNC loss function
2310:From Tables 13 and 14, we can observe that the performance
2312:the following reasons: (1) Stable measurement of common features:
2325:4.5. Effectiveness of CNC loss function
2326:4.5.1. Visual analysis of nearest neighbor features in CNC loss function
2327:As shown in Fig. 6, we demonstrate the results of anchor sample
2328:searching for the nearest-neighbor cross-modality samples within the
2329:same class in CNC. It can be observed that our model not only effectively extracts the discriminative regions of the images but also matches
2331:the nearest-neighbor algorithm. For example, as shown in Fig. 6(b), our
2339:Fig. 6. Visualization of neighbor features in CNC loss. The first row is the original image, and the second row is the features obtained by the model. The red box represents the
2340:features of the nearest neighbor.
2342:Table 13
2429:Table 14
2495:the covariance matrix, which can be inaccurate when feature distributions differ greatly, leading to unstable measurements. For example,
2501:As shown in Table 16, our approach significantly outperforms existing methods in all-search and indoor-search modes.
2511:In Table 15, we verify the stability of our method by comparing the standard deviation with WF-CAMReViT [23] and LAB [24]
2520:In Table 17, our approach significantly outperforms existing methods in both evaluation modes. Our method outperforms the metric
2527:our approach shows notable leads of 3.12% and 6.97% in Rank-1
2529:In Tables 16 and 17, we compare with state-of-the-art methods
2544:Table 15
2648:Table 16
2941:Table 17
3234:First, as shown in Tables 16 and 17, our method achieves superior
3239:multi-granularity features, compensates for missing single-granularity
3246:Table 18
3360:Second, in Table 16, we observed that on the SYSU-MM01 dataset,
3364:multi-granularity features for fusion. This ensures algorithm effectiveness without the need for additional models. Compared to PartMix,
3366:dataset scale and computational complexity, our method uses a multigranularity feature extraction algorithm based on the original dataset
3368:Third, the results on the RegDB dataset in Table 17 indicate that
3370:We believe this is due to the RegDB dataset being small and pedestrian features being relatively complete. Single-granularity features
3372:advantages of multi-granularity features in our method less effective. In
3374:in scenes, and has significant background noise interference, the advantages of multi-granularity are effectively reflected. The experimental
3377:Fig. 7. Top-10 search results for five randomly selected images on the SYSU-MM01
3382:To further verify the effectiveness of our proposed method, we
3385:contains 46,767 bounding boxes of 1064 identities. Table 18 shows that
3389:single granularity fails to discern pedestrian identities effectively, the
3392:4.8. Visualization
3393:4.8.1. Visualization of qualitative results
3394:As shown in Fig. 7, we visualize the Rank-10 search results for five
3395:random selected pedestrian queries. In Fig. 7(a–c), our approach can
3397:facial information. In Fig. 7(d), when facial features are missing, our
3399:walking posture. In Fig. 7(e), when pedestrians have similar facial features, arm movements, and clothing logs, another granularity feature
3406:Fig. 8. Visualization result of heatmaps on the SYSU-MM01 dataset. (a) Input images,
3410:4.8.2. Visualization of feature heatmaps
3411:We visualize feature maps from global branch, patch branch, pixel
3413:Fig. 8. In Fig. 8(b), we observe that the global branch focuses on large
3414:regions of interest in pedestrian images, ignoring other granularity
3421:Fig. 9. The visualization results of feature distributions from the baseline and our approach by t-SNE on the SYSU-MM01 dataset, where each color represents an identity. The
3425:focuses on the body region of person 1, ignoring the most discriminative facial feature. In Fig. 8(c), the patch branch pays attention
3428:of person 4 are captured. In Fig. 8(d), we can observe that more
3430:posture and arm movements of person 5 and person 8. In Fig. 8(e),
3431:our approach can effectively cover multi-granularity discriminative
3434:not only addresses the limitations of relying on single-granularity features, but also enhances the model’s ability to focus on pedestrians
3435:with discriminative information at multi-granularity in a variety of
3448:Visualization, Validation, Software, Resources, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Feng Chen:
3449:Writing – review & editing, Writing – original draft, Visualization,
3457:4.8.3. Visualization of feature distribution
3458:We visualize the feature distributions using t-SNE on the SYSUMM01 dataset in Fig. 9. Different colors represent different persons,
3460:infrared modality images. Fig. 9(a) shows the distribution of features
3461:extracted by baseline method (trained by 𝐿𝑖𝑑 and 𝐿𝑡𝑐 ), and Fig. 9(b)
3463:red arrow indicates the difference between modalities in Fig. 9. We
3464:can observe that the length of the arrow in Fig. 9(b) is much smaller
3465:than that in Fig. 9(a). This phenomenon is because the baseline clusters
3478:In this paper, we design a cross-modality multi-granularity
3479:(CM2 MG) Transformer framework including a multi-granularity feature
3480:extractor (MGFE) module and a multi-granularity fusion Transformer
3481:(MGFT) module. MGFE aims to capture multi-granularity level features from two modalities to explore the complementarity of multigranularity features. MGFT fuses multi-granularity feature pairs to mine
3482:hierarchical relationship between multi-granularity features across
3483:modalities. In addition, the cross-modality nearest-neighbor clustering
3484:(CNC) function is designed to further pull cross-modality intra-class
3487:approach outperforms most state-of-the-art methods. Ablation studies
3489:Meanwhile, in the section of visualization, it can be observed that our
3584:[15] M. Cao, R. Yan, X. Shu, J. Zhang, J. Wang, G.-S. Xie, MUP: Multi-granularity

exec
/bin/zsh -lc "pdftotext 'Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf' - | rg -n -i \"ablation|effectiveness|visual|visualization|CCS|HCM|BCLM|GPRA|DRHL|RHCL|centrality|noisy|Table|Fig\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
17:the effects of noisy labels through loss functions during
18:backward propagation. However, these noisy labels already
20:a Hierarchical Centrality Collaborative Learning (HCCL)
21:framework for US-VI-ReID, which proactively identifies noisy
26:learning, we propose a Closeness Centrality Selection (CCS),
27:quantifying sample confidence using closeness centrality to
28:identify noisy samples. By discarding the noisy samples during
29:forward propagation, CCS mitigates their adverse effects
32:(HCM) is proposed to establish local instance-level label
41:Index Terms—Person re-identification, cross-modality, unsupervised learning, noisy labels.
95:introduces noise into the pseudo-labels, i.e., noisy labels,
100:similar features, such as clothing or body shapes, as illustrated by the noisy labels in Fig. 1 (a). The pedestrian
101:features assigned the wrong pseudo-labels are termed noisy
102:samples. Existing methods [22], [23], [24] optimize noisy
115:Fig. 1. Illustration of our motivation. (a) Existing methods overlook the
116:issue of noisy labels caused by intra-modality variation during the clustering
118:our method proactively identifies noisy labels during forward propagation,
126:most approaches directly use noisy labels without effective
129:correspondence in Fig. 1 (a). This confusion complicates
135:effectiveness, they overlook a critical flaw: these noisy samples
140:whether US-VI-ReID can directly discard noisy samples during intra-modality learning to obtain reliable samples within
145:own hardness. Driven by this motivation, we propose a Hierarchical Centrality Collaborative Learning (HCCL) framework,
146:incorporating the Closeness Centrality Selection (CCS) and
147:the Hierarchical Consistency Matching (HCM), as shown in
148:Fig. 1 (b). Specifically, CCS calculates the closeness centrality score for each sample within a cluster according to
151:cluster, as reflected by a high closeness centrality score, they
156:threshold are regarded as noisy samples and subsequently
161:the proposed HCM employs a novel local-to-global hierarchical matching paradigm, reinforced by the bidirectional
163:alignment. At the local level, HCM associates cross-modality
166:identified by CCS, and 2) bidirectional affinity should be
172:• We propose a novel Hierarchical Centrality Collaborative Learning (HCCL) framework for US-VI-ReID that
177:• We design a Closeness Centrality Selection (CCS) to
179:by using closeness centrality, serving as the reliable
182:• Hierarchical Consistency Matching (HCM) is proposed
188:datasets demonstrate the effectiveness of HCCL, which
284:of noisy labels generated during the clustering process,
287:loss functions to address noisy samples during backward propagation, but by that stage, the adverse effects of noisy samples
290:directly discards noisy samples during forward propagation,
293:In this section, the proposed Hierarchical Centrality Collaborative Learning (HCCL) is explained in detail. First, we
295:we introduce the proposed Closeness Centrality Selection
296:(CCS) in Section III-B. Finally, we present the Hierarchical
297:Consistency Matching (HCM) in Section III-C.
305:The pipeline of the proposed HCCL is shown in Fig. 2,
314:Fig. 2. Illustration of our proposed HCCL. (a) Clustering stage. HCCL generates intra-modality pseudo-labels by DBSCAN, consistent with [22]. Different
315:shapes denote different identities. (b) is our proposed Closeness Centrality Selection (CCS), detailed in Sec. III-B. The darker the color of the instance, the
316:higher its confidence level, and vice versa. Instances with confidence levels below the predefined positional threshold are identified as noisy samples. LRHCL
317:is our proposed reliable homogeneous contrastive loss. (c) is our proposed Hierarchical Consistency Matching (HCM), and will be described in Sec. III-C.
318:LDRHL denotes our proposed discriminative reliable heterogeneous loss.
327:the Closeness Centrality Selection (CCS), which quantifies
330:Consistency Matching (HCM) to establish cross-modality correspondences by adopting a novel local-to-global hierarchical
332:B. Closeness Centrality Selection
340:pseudo-label assignments, namely noisy labels. Moreover, subsequent cross-modality matching directly relies on these noisy
343:correspondences. To mitigate the issue of noisy labels, we
344:propose a Closeness Centrality Selection (CCS) to proactively quantify sample confidence during forward propagation,
346:memories for homogeneous contrastive learning. The CCS
348:Centrality Exploration (CCE) and Reliable Homogeneous
349:Contrastive Learning (RHCL). The detailed architecture is
350:shown in Fig. 2 (b).
351:1) Closeness Centrality Exploration: In CCE, each cluster (after DBSCAN) is modeled as a connected weighted
535:identified as noisy samples. They are directly discarded during
537:mitigate the impact of noisy samples. This process allows for
577:LRHCL = −
596:LrRHCL = −
623:LRHCL = LvRHCL + LrRHCL .
627:LRHCL selects reliable samples within each cluster as prototypes for cluster representation, learning modality-specific
630:enhance robustness against noisy labels.
643:limitations: noisy labels and scale sensitivity. Specifically,
645:noisy labels into the cost matrix, increasing the difficulty
648:Consistency Matching (HCM) to mine the underlying crossmodality correspondences by adopting a novel local-to-global
649:hierarchical paradigm. For local matching, Bidirectional Consistency Local Matching (BCLM) is proposed to associate
651:Reliability Alignment (GPRA) is designed to establish global
654:(DRHL) is proposed to learn modality-invariant and discriminative features and resist the huge cross-modality discrepancy.
655:The detailed architecture is shown in Fig. 2 (c).
657:select the instance with the highest closeness centrality score
658:by the CCS as the centrality anchor instance, which is formulated as:
667:where Avc,i denotes the closeness centrality score of the
669:an indicator function. Each centrality anchor instance serves
671:Subsequently, the centrality anchor instances from each cluster
675:Then we compute the similarity between the c-th centrality
693:As depicted wrong match in Fig. 2 (c), the blue samples within
725:respectively. f˜c ∈ F̃ and f˜lr ∈ F̃ r denote the centrality
731:each centrality anchor instance once the first reliable positive
735:various algorithms to globally mine cross-modality correspondences. However, the presence of noisy samples undermines
738:To address the above issues, we try to utilize centrality
741:BCLM. However, the conditions for associating the label of
742:two modalities in BCLM are too stringent, and due to large
829:LDRHL = −
858:DRHL
886:metric losses to obtain LDRHL , which can be formulated as
890:DRHL , epoch%2 = 0,
892:LDRHL =
894:DRHL , epoch%2 = 1,
895:where LDRHL minimizes the distance between the given sample
899:LRHCL and LDRHL , which is defined as follows:
900:L = LRHCL + µLDRHL ,
927:sets 10 times, and the results are averaged to obtain a stable
952:To comprehensively evaluate the effectiveness of our proposed method, we compare our methods with supervised and
954:The experimental results are shown in Table I and Table II.
960:As shown in Table I and Table II, our proposed HCCL
972:TABLE I
999:exploration of noisy labels, addressing the issue through
1002:our method discards noisy samples during the forward propagation, allowing for immediate feature adjustments without
1006:superiority and effectiveness of our HCCL. In addition, we
1021:TABLE II
1033:by directly discarding noisy samples during forward propagation, it prevents the issue where noisy samples are
1037:we will continue to explore approaches to address noisy labels
1043:TABLE III
1052:As shown in Table III, it is evident that our proposed HCCL
1061:TABLE IV
1082:D. Ablation Study
1083:To evaluate the effectiveness of each component introduced
1084:in this paper, we conduct ablation experiments on SYSUMM01 and RegDB datasets. LADCL and LACCL indicate the
1088:as the baseline. While this work facilitates unsupervised VIReID learning, it overlooks the adverse impact of noisy labels.
1091:2) Effectiveness of CCS:
1092:a) Effectiveness of CCE: The CCE is proposed to
1096:RegDB datasets, respectively as shown in Table IV (see index
1098:is that it captures reliable samples with closeness centrality
1099:during forward propagation, alleviating the issue of noisy
1101:b) Effectiveness of LRHCL : The LRHCL is designed to
1103:extractor. As shown in Table IV, compared to LADCL , the performance is improved to 3% in Rank-1 and 1% in mAP on the
1110:3) Effectiveness of HCM:
1111:a) Effectiveness of BCLM: The BCLM is introduced
1115:most confident samples as centrality anchors and introduces
1118:b) Effectiveness of LDRHL : Notably, LDRHL cannot
1121:network training and optimization. When combined LDRHL
1122:with GPRA, it achieved significant improvements on the
1125:LDRHL makes efficient use of the cross-modality correspondences, enabling the model to learn modality-invariant and
1126:discriminative features. When comparing LDRHL with LACCL
1134:parameter on SYSU-MM01 dataset, as shown in Fig. 3.
1137:centrality score. Through experiments, we find that when
1154:Fig. 3. The effect of hyper-parameter n, γ and µ with different values on SYSU-MM01 under all-search mode.
1156:Fig. 4. The t-SNE visualization of 10 randomly selected identities from SYSU-MM01. Circles and crosses represent features of visible and infrared images,
1159:Fig. 5. The visualization of intra-class and inter-class distances of crossmodality features on SYSU-MM01.
1167:to combine LRHCL and LDRHL . When µ is too small, the
1175:F. Visualization
1176:To demonstrate the effectiveness of HCCL, we first visualize
1179:Fig. 6. Visualization of top-5 retrieval results on SYSU-MM01. Green and
1182:as illustrated in Fig. 4. The intra- and inter-class distances
1184:presented in Fig. 5. Additionally, the ranking list for the
1185:SYSU-MM01 dataset is shown in Fig. 6.
1186:1) T-SNE Visualization: In Fig. 4 (a), we observe that
1191:of noisy labels. Fig. 4 (b), Fig. 4 (c) and Fig. 4 (d) present the
1202:converging. However, the issue of noisy labels remains unresolved.
1203:In contrast, as shown in Fig. 4 (e), samples of the same
1207:Moreover, the issue of noisy labels is effectively mitigated.
1211:2) Intra- and Inter-Class Visualization: Meanwhile, as
1212:shown in Fig. 5 (a) and Fig. 5 (b), it is obvious that the means
1218:and discarding noisy samples during training, resulting in a
1221:3) Ranking Visualization: We conduct the top-5 retrieval
1223:setting of infrared to visible, as shown in Fig. 6. We observe
1236:In this paper, we propose a novel Hierarchical Centrality
1239:learning, we design a Closeness Centrality Selection (CCS)
1241:centrality. Noisy samples are directly discarded during forward propagation, ensuring immediate adjustment of features
1243:a Hierarchical Consistency Matching (HCM) to establish
1247:HCM establishes dependable cross-modality correspondences,
1419:[52] K. Okamoto, W. Chen, and X.-Y. Li, “Ranking of closeness centrality
1464:with twin noisy labels for visible-infrared person re-identification,”
1486:[74] L. Van der Maaten and G. Hinton, “Visualizing data using t-SNE,”

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
exec
/bin/zsh -lc "pdftotext 'Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf' - | nl -ba | sed -n '1138,1162p;2188,2240p;2380,2441p;2658,2684p;2735,2739p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1138	D. Ablation Studies
  1139	
  1140	For our backbone feature extractor, we selected the ViT-Base
  1141	a) Effect of DDO: Our DDO module reconstructs clean
  1142	(patch16, 224) architecture initialized with standard ImageNet latent feature embeddings from occluded inputs, trained on
  1143	pretraining. Once the hyperparameters were finalised using a 40,000 synthetic high-resolution frontal images with diverse
  1144	thorough Grid search, identical settings were used for each poses (see Section I of Supplementary). As in prior ReID
  1145	dataset without configuration tuning. So, for example, the ideal work [75], [81], long-range retrieval relies mainly on clothing
  1146	Top-N for Market = 90 (See Figure 8) but our reported results appearance and coarse pose cues rather than facial biometrics.
  1147	are all at Top-N = 50.
  1148	Pose diversity in our training dataset, together with FDA-based
  1149	All experiments were conducted using 1 NVIDIA RTX domain adaptation [57], supports generalization to real-world
  1150	3080 GPU, equipped with 10 GB of VRAM. The final ReID ReID. Therefore, DDO learns to target latent-space inpainting
  1151	model, after construction but before data loading, occupies of occluded regions to preserve semantic fidelity, making it
  1152	approximately 1.014 GB of memory. For the Market-1501 inherently well suited for occluded person ReID.
  1153	dataset, each training iteration with a batch size of 32 takes apTo evaluate the effectiveness of DDO (De-Occluder), we
  1154	proximately 4.5 minutes to complete. Since the model remains present a qualitative comparison between the baseline ReID
  1155	unchanged for all datasets, the compute specifications remain pipeline and our DDO-enhanced approach in Figure 6. Ocuniform throughout. To facilitate full reproducibility, Section clusion in the baseline model distorts identity cues, leading
  1156	II of the Supplementary Material details all hyperparameters, to attention on irrelevant regions and retrieval errors. DDO
  1157	loss functions, and training strategies.
  1158	reconstructs features approximating unoccluded counterparts,
  1159	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
  1160	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
  1161	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
  1162	
  2188	SEAS [80]
  2189	DDO & HRR (Ours)
  2190	
  2191	guiding cross-attention toward visible, identity-discriminative
  2192	regions. For example, DDO restores focus to distinctive lowerbody clothing in the first case, redirects attention from a
  2193	nearby occluder to the subject’s face and posture in the second,
  2194	and shifts focus from ambiguous hair to unique clothing
  2195	in the third. Notably, in the last two CUHK03 cases, the
  2196	baseline erroneously interprets the white wall as foreground,
  2197	hallucinating person-like features across the entire image and
  2198	attending indiscriminately to background and occluded regions.
  2199	DDO correctly identifies the white wall as an occlusion,
  2200	refocusing attention on salient regions (clothing, visible body
  2201	parts) for accurate retrieval. These results demonstrate DDO’s
  2202	ability to suppress distractions, recover identity-consistent
  2203	features, and improve ReID under occlusion.
  2204	
  2205	URL
  2206	URL
  2207	URL
  2208	URL
  2209	URL
  2210	–
  2211	URL
  2212	URL
  2213	URL
  2214	–
  2215	–
  2216	URL
  2217	URL
  2218	URL
  2219	–
  2220	–
  2221	
  2222	Figure 7(a) illustrates the variation of Mean Squared Error
  2223	(MSE) and Cosine Error between reconstructed and original
  2224	features as a function of diffusion steps. Both metrics improve
  2225	significantly up to 100 steps, beyond which further gains are
  2226	marginal, indicating an optimal trade-off between accuracy and
  2227	computational cost.
  2228	We evaluate structural similarity (SSIM) between occluded
  2229	feature maps Fo and unoccluded features F before and after
  2230	applying DDO (Fclean ). For each occluded input, we extract
  2231	ViT features and compute SSIM against the clean embedding
  2232	(Fo and F ). This process is repeated after processing the
  2233	occluded input through the DDO (Fclean and F ). Figure 7
  2234	(b) plots results over 100 held-out Occluded SHHQ test
  2235	images (unseen during DDO training). The x-axis shows
  2236	
  2237	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
  2238	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
  2239	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
  2240	
  2380	conditions. The addition of HRR further boosts retrieval
  2381	accuracy by effectively guiding the re-ranking process, thereby
  2382	establishing DDO as a robust feature inpainting module and
  2383	HRR as a powerful complement for retrieval refinement.
  2384	b) Effect of Top-N Centroid Selection on Retrieval Performance: We observe that retrieval performance improves
  2385	with increasing values of top-N , reaching a peak at a datasetdependent value before plateauing, as shown in Figure 8.
  2386	Datasets with larger gallery sizes tend to achieve peak performance at higher N values. This behavior can be explained by
  2387	the structure of the centroid-based hybrid re-ranking approach.
  2388	Let G denote the full gallery, and let GN ⊂ G represent the
  2389	subset of gallery samples associated with the Top-N centroids
  2390	most similar to a query feature q. The final ranking is based
  2391	on re-ranking within this subset GN . A larger gallery implies
  2392	that each centroid ID represents more samples, with broader
  2393	variations in pose, occlusion, and viewpoint. Hence, if |G| is
  2394	large and N is small, the subset GN may not contain enough
  2395	intra-class samples or the best-quality matches for robust reFig. 6: Qualitative comparison of attention maps and retrieval
  2396	ranking. Increasing N improves the likelihood of including
  2397	results. (a) Probe, (b) Heatmap without DDO, (c) Heatmap
  2398	relevant positive samples before re-ranking.
  2399	with DDO, (d) the Rank-1 retrieval without DDO, and (e)
  2400	Let P ⊂ G be the set of gallery samples sharing the query’s
  2401	the Rank-1 retrieval with DDO. Green and red boxes indicate
  2402	identity, and let PN = P ∩ GN denote the subset within the
  2403	correct and incorrect matches, respectively.
  2404	Top-N centroid candidates. The recall of positive matches is
  2405	defined as: rN = |PN |/|P|. The expected retrieval quality is
  2406	then approximated as: E[mAP] ∝ E[rank-quality(PN )] · rN .
  2407	In large galleries, rN increases more slowly for small
  2408	patch-to-image ratio, directly indicating occlusion severity. At
  2409	low occlusion, baseline SSIM (Fo vs. F ) is already high, N , requiring higher N to achieve sufficient coverage—thus
  2410	leaving minimal improvement room; DDO gains are small. explaining the peak performance at larger N in such settings.
  2411	We analyze how gallery size and sample density affect reAs occlusion increases, baseline SSIM sharply degrades. In
  2412	contrast, DDO-enhanced features (Fclean vs. F ) maintain ranking performance. Figure 7 (c) shows that larger galleries
  2413	significantly higher SSIM under heavy occlusion, demonstrating yield greater mAP improvements with perfect monotonic
  2414	active reconstruction of corrupted content and realignment correlation (r = 1.00, p = 0.000), likely due to enhanced
  2415	noisy-match suppression and better mutual nearest-neighbor
  2416	toward true identity representation.
  2417	We further validate the effectiveness of our modules by alignment. Occluded-Duke exhibits the most pronounced
  2418	incorporating DDO and HRR into two representative ViT- improvement: it has the largest gallery and highest sample
  2419	based pipelines, TransReID [27] and Pass-ViT [65]. In both density per identity, enabling more effective centroid-based
  2420	cases, the ViT-extracted features are passed through DDO filtering and local re-ranking under occlusion. An actionable
  2421	to obtain de-occluded embeddings, which are then enhanced heuristic for optimal N selection is present in Section V of the
  2422	by multi-headed self-attention before being fed into the Supplementary Material.
  2423	downstream model, while HRR refines retrieval by leveraging
  2424	gallery centroids during re-ranking. As shown in Table VI, E. Limitations
  2425	DDO consistently improves performance across all datasets,
  2426	This subsection outlines the operational limitations of the
  2427	with particularly large gains on occlusion-heavy benchmarks proposed DDO and HRR frameworks. Identifying these boundsuch as Occluded-Duke and Occluded-ReID, confirming its ary conditions provides a holistic assessment of our algorithm’s
  2428	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:05 UTC from IEEE Xplore. Restrictions apply.
  2429	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
  2430	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
  2431	
  2432	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
  2433	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3681512
  2434	
  2435	11
  2436	
  2437	Fig. 7: (a) Plot showing the Mean Squared Error (MSE) and Cosine Error between reconstructed and original features across
  2438	varying numbers of diffusion steps in the denoising process. (b) Comparison of SSIM similarity between occluded and clean
  2439	features, before and after applying the DDO module. (c) Relationship between gallery size and mAP gain achieved through
  2440	re-ranking across five ReID datasets. A linear trend is fitted, and Spearman correlation is reported to quantify the dependency.
  2441	TABLE VII: Comparison of mAP scores with and without DDO and HRR. Bold represents best results and underline represents
  2658	1) DDO: Limitations of Feature Reconstruction: While the
  2659	proposed Latent Diffusion guided De-Occluder (DDO) effectively recovers identity-preserving representations under partial
  2660	occlusion, its performance naturally degrades under extreme
  2661	occlusion scenarios (e.g., > 80% coverage). In such cases, the
  2662	severely limited visible context provides insufficient semantic
  2663	guidance for accurate latent reconstruction. Furthermore, our
  2664	current framework is explicitly formulated to model and invert
  2665	spatial occlusion distortions. It is not currently designed to
  2666	disentangle or mitigate simultaneous environmental artifacts,
  2667	such as severe illumination changes, motion blur, or heavy
  2668	sensor noise. Extending the feature inpainting mechanism to
  2669	handle these compounded, multi-factor visual degradations
  2670	remains a valuable direction for future research.
  2671	2) HRR: Limitations of Centroid based Filtering: When
  2672	gallery samples per identity are few and heavily occluded,
  2673	centroids become biased toward incomplete representations,
  2674	causing minimally occluded queries to align closer to
  2675	less-occluded identities and leading to retrieval errors. This
  2676	
  2677	is evident in Table VII on Occluded-ReID (mAP drops from
  2678	92.3% to 90.1%) and Table VIII on Partial-ReID (Rank-1
  2679	drops from 58.3% to 56.7%). Heavy occlusion in person
  2680	re-identification (Re-ID) leads to two simultaneous effects: a
  2681	reduction in the number of visible discriminative features, and
  2682	a structured bias in gallery centroid embeddings. Together,
  2683	these factors significantly degrade retrieval performance under
  2684	sparse or occlusion-heavy conditions. We use the methodology
  2735	validated in Table VII, where HRR significantly improves
  2736	performance on Occluded-Duke (1000 gallery identities, mAP:
  2737	81.8% → 90.6%), but exhibits diminished effectiveness on
  2738	Occluded-ReID (100 gallery identities, mAP: 92.3% → 90.1%),
  2739	indicating that centroid-based filtering requires sufficient gallery

exec
/bin/zsh -lc "pdftotext 'Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '760,905p;1498,1985p;2325,2340p;3393,3435p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   760	60.44
   761	63.50
   762	61.59
   763	55.29
   764	
   765	68.71
   766	69.47
   767	73.41
   768	69.93
   769	66.76
   770	
   771	93.29
   772	93.75
   773	97.29
   774	95.11
   775	93.25
   776	
   777	97.84
   778	98.47
   779	99.37
   780	98.60
   781	97.28
   782	
   783	68.27
   784	73.02
   785	76.63
   786	71.25
   787	68.05
   788	
   789	Table 2
   790	The results of the shared and individual circumstances on the REGDB dataset.
   791	
   792	where ⟨⋅⟩ represents the dot product operator.
   793	Meanwhile, to reduce the modality gap, we propose modalityalignment loss 𝐿𝑚𝑎 using Kullback–Leibler divergence to preserve feature correlations between modalities.
   794	1 ∑∑
   795	𝐿𝑚𝑎 =
   796	(𝐾 𝐿(𝐴 ∥ 𝑃 ) + 𝐾 𝐿(𝑃 ∥ 𝐴))
   797	(18)
   798	|𝑁|
   799	
   800	Settings
   801	
   802	RegDB dataset
   803	V–I
   804	
   805	𝑠0
   806	𝑠1
   807	𝑠2
   808	𝑠3
   809	𝑠4
   810	
   811	In summary, the 𝐿𝑐 𝑛𝑐 can be represented by:
   812	
   813	I–V
   814	
   815	R1
   816	
   817	R10
   818	
   819	R20
   820	
   821	mAP
   822	
   823	R1
   824	
   825	R10
   826	
   827	R20
   828	
   829	mAP
   830	
   831	85.29
   832	85.53
   833	86.72
   834	83.88
   835	77.52
   836	
   837	95.53
   838	95.29
   839	96.26
   840	95.89
   841	93.35
   842	
   843	97.67
   844	97.91
   845	98.30
   846	97.67
   847	96.94
   848	
   849	74.37
   850	77.45
   851	77.79
   852	74.37
   853	67.52
   854	
   855	85.05
   856	86.34
   857	86.47
   858	82.77
   859	73.40
   860	
   861	96.17
   862	96.80
   863	96.12
   864	95.29
   865	91.02
   866	
   867	98.64
   868	98.50
   869	97.77
   870	97.38
   871	94.90
   872	
   873	76.08
   874	77.13
   875	77.51
   876	74.08
   877	63.88
   878	
   879	(19)
   880	
   881	𝐿𝑐 𝑛𝑐 = 𝐿𝑚𝑐 + 𝐿𝑚𝑎
   882	
   883	The total loss function contains the identity loss function 𝐿𝑖𝑑 , tripletcenter loss function 𝐿𝑡𝑐 [4,5,22], and cross-modality nearest-neighbor
   884	clustering loss function 𝐿𝑐 𝑛𝑐 . 𝐿𝑖𝑑 aims to learn discriminative feature
   885	representations by using modality-specific information, which encourages identity-invariant feature representations and distinguishes the
   886	identities of different persons. 𝐿𝑡𝑐 ensures intra-class feature compactness by minimizing the cross-modality positive center pairwise
   887	distance.
   888	The total loss can be represented as:
   889	
   890	indoor-search mode. In all-search mode, the gallery images are captured
   891	from four visible cameras. In indoor-search mode, the gallery images are
   892	captured by two visible light cameras.
   893	RegDB dataset [11] is captured by a dual-camera including one
   894	visible and one infrared camera, which contains 8240 images of 412
   895	identities. Each identity consists of 10 visible images and 10 infrared
   896	images. The training set and testing set contain 2060 visible images
   897	and 2060 infrared images of 206 identities. It contains two evaluation
   898	settings (visible-search-infrared (V - I) and infrared-search-visible (I V)) [4,5]. The evaluation process is repeated 10 times with random
   899	gallery and probe sets, and the performance is based on the average
   900	experimental results.
   901	
   902	(20)
   903	
   904	𝐿𝑡𝑜𝑡𝑎𝑙 = 𝐿𝑖𝑑 + 𝛼 𝐿𝑡𝑐 + 𝛽 𝐿𝑐 𝑛𝑐
   905	The training process of our approach is shown in Algorithm 1.
  1498	4.3. Ablation study
  1499	4.3.1. Effectiveness of each module
  1500	To verify the effectiveness of each module (MGFE, MGFT, CNC) in
  1501	our approach, we conduct ablation experiments on the SYSU-MM01
  1502	and RegDB datasets. ‘‘B’’ denotes the baseline with ResNet50, 𝐿𝑖𝑑 , and
  1503	𝐿𝑡𝑐 .
  1504	Table 7 lists the results in all-search and indoor-search modes. In
  1505	Index 1 and Index 2, the results of MGFE are 6.26% and 4.47% higher
  1506	than baseline in Rank-1 accuracy on two modes, respectively. This
  1507	improvement shows that compared to using single-granularity features,
  1508	
  1509	4.2.3. Pixel branch setting
  1510	On the pixel branch, we analyze the performance influence of
  1511	different numbers of stripes and the depth in the Transformer on the
  1512	7
  1513	
  1514	Pattern Recognition 164 (2025) 111510
  1515	
  1516	Y. Feng et al.
  1517	Table 6
  1518	The settings of pixel branch on the RegDB dataset.
  1519	Settings
  1520	
  1521	RegDB
  1522	V–I
  1523	
  1524	Pixel
  1525	
  1526	I–V
  1527	
  1528	Stripes
  1529	
  1530	Depth
  1531	
  1532	R1
  1533	
  1534	R10
  1535	
  1536	R20
  1537	
  1538	mAP
  1539	
  1540	R1
  1541	
  1542	R10
  1543	
  1544	R20
  1545	
  1546	mAP
  1547	
  1548	1
  1549	3
  1550	6
  1551	
  1552	1
  1553	1
  1554	1
  1555	
  1556	82.91
  1557	85.42
  1558	86.72
  1559	
  1560	94.13
  1561	95.63
  1562	96.26
  1563	
  1564	97.14
  1565	97.23
  1566	98.30
  1567	
  1568	73.09
  1569	77.08
  1570	77.79
  1571	
  1572	80.63
  1573	84.08
  1574	86.47
  1575	
  1576	94.08
  1577	94.27
  1578	96.12
  1579	
  1580	96.80
  1581	97.04
  1582	97.77
  1583	
  1584	71.06
  1585	74.70
  1586	77.51
  1587	
  1588	6
  1589	6
  1590	
  1591	2
  1592	3
  1593	
  1594	84.66
  1595	84.08
  1596	
  1597	94.42
  1598	95.53
  1599	
  1600	96.55
  1601	97.62
  1602	
  1603	77.18
  1604	75.56
  1605	
  1606	84.61
  1607	84.51
  1608	
  1609	95.87
  1610	95.53
  1611	
  1612	97.96
  1613	97.57
  1614	
  1615	74.79
  1616	74.73
  1617	
  1618	𝑤 ℎ
  1619	
  1620	[1 1]
  1621	
  1622	Table 7
  1623	Evaluation of each module on the SYSU-MM01 dataset.
  1624	Index
  1625	
  1626	B
  1627	
  1628	MGFE
  1629	
  1630	MGFT
  1631	
  1632	CNC
  1633	
  1634	SYSU-MM01 dataset
  1635	All-search
  1636	
  1637	Indoor-search
  1638	
  1639	R1
  1640	
  1641	R10
  1642	
  1643	R20
  1644	
  1645	mAP
  1646	
  1647	R1
  1648	
  1649	R10
  1650	
  1651	R20
  1652	
  1653	mAP
  1654	
  1655	1
  1656	2
  1657	3
  1658	
  1659	✔
  1660	✔
  1661	✔
  1662	
  1663	✗
  1664	✔
  1665	✔
  1666	
  1667	✗
  1668	✗
  1669	✔
  1670	
  1671	✗
  1672	✗
  1673	✗
  1674	
  1675	58.48
  1676	64.74
  1677	66.56
  1678	
  1679	88.11
  1680	90.77
  1681	93.75
  1682	
  1683	92.64
  1684	93.90
  1685	96.36
  1686	
  1687	55.82
  1688	59.28
  1689	62.75
  1690	
  1691	62.16
  1692	66.63
  1693	69.42
  1694	
  1695	90.80
  1696	90.72
  1697	94.11
  1698	
  1699	93.79
  1700	95.88
  1701	97.16
  1702	
  1703	64.51
  1704	70.39
  1705	72.26
  1706	
  1707	4
  1708	
  1709	✔
  1710	
  1711	✔
  1712	
  1713	✔
  1714	
  1715	✔
  1716	
  1717	69.79
  1718	
  1719	95.45
  1720	
  1721	98.71
  1722	
  1723	63.50
  1724	
  1725	73.41
  1726	
  1727	97.29
  1728	
  1729	99.37
  1730	
  1731	76.63
  1732	
  1733	Table 8
  1734	Evaluation of each module on the REGDB dataset.
  1735	Index
  1736	
  1737	B
  1738	
  1739	MGFE
  1740	
  1741	MGFT
  1742	
  1743	CNC
  1744	
  1745	RegDB dataset
  1746	V–I
  1747	
  1748	I–V
  1749	
  1750	R1
  1751	
  1752	R10
  1753	
  1754	R20
  1755	
  1756	mAP
  1757	
  1758	R1
  1759	
  1760	R10
  1761	
  1762	R20
  1763	
  1764	mAP
  1765	
  1766	1
  1767	2
  1768	3
  1769	
  1770	✔
  1771	✔
  1772	✔
  1773	
  1774	✗
  1775	✔
  1776	✔
  1777	
  1778	✗
  1779	✗
  1780	✔
  1781	
  1782	✗
  1783	✗
  1784	✗
  1785	
  1786	74.61
  1787	83.15
  1788	86.36
  1789	
  1790	85.83
  1791	92.90
  1792	95.58
  1793	
  1794	90.53
  1795	93.38
  1796	97.52
  1797	
  1798	75.14
  1799	75.68
  1800	77.20
  1801	
  1802	72.63
  1803	82.58
  1804	87.72
  1805	
  1806	81.96
  1807	92.08
  1808	97.18
  1809	
  1810	88.56
  1811	94.94
  1812	98.45
  1813	
  1814	70.97
  1815	74.73
  1816	78.75
  1817	
  1818	4
  1819	
  1820	✔
  1821	
  1822	✔
  1823	
  1824	✔
  1825	
  1826	✔
  1827	
  1828	86.72
  1829	
  1830	96.26
  1831	
  1832	98.30
  1833	
  1834	77.79
  1835	
  1836	86.47
  1837	
  1838	96.12
  1839	
  1840	97.77
  1841	
  1842	77.51
  1843	
  1844	Table 9
  1845	Evaluation of each granularity on the SYSU-MM01 dataset.
  1846	Index
  1847	
  1848	Global
  1849	
  1850	Patch
  1851	
  1852	Pixel
  1853	
  1854	SYSU-MM01 dataset
  1855	All-search
  1856	
  1857	Indoor-search
  1858	
  1859	R1
  1860	
  1861	R10
  1862	
  1863	R20
  1864	
  1865	mAP
  1866	
  1867	R1
  1868	
  1869	R10
  1870	
  1871	R20
  1872	
  1873	mAP
  1874	
  1875	1
  1876	2
  1877	3
  1878	
  1879	✔
  1880	✔
  1881	✔
  1882	
  1883	✗
  1884	✔
  1885	✗
  1886	
  1887	✗
  1888	✗
  1889	✔
  1890	
  1891	60.01
  1892	66.19
  1893	67.55
  1894	
  1895	89.09
  1896	91.66
  1897	92.64
  1898	
  1899	93.22
  1900	95.69
  1901	97.00
  1902	
  1903	57.96
  1904	61.55
  1905	60.11
  1906	
  1907	64.35
  1908	67.30
  1909	68.34
  1910	
  1911	91.39
  1912	94.34
  1913	94.88
  1914	
  1915	94.88
  1916	96.83
  1917	96.15
  1918	
  1919	67.30
  1920	72.02
  1921	71.26
  1922	
  1923	4
  1924	
  1925	✔
  1926	
  1927	✔
  1928	
  1929	✔
  1930	
  1931	69.79
  1932	
  1933	95.45
  1934	
  1935	98.71
  1936	
  1937	63.50
  1938	
  1939	73.41
  1940	
  1941	97.29
  1942	
  1943	99.37
  1944	
  1945	76.63
  1946	
  1947	we can capture comprehensively discriminative information by using
  1948	multi-granularity features. Comparing Index 2 with Index 3, these improved results prove that fusing features between multiple granularities
  1949	can further enhance the discrimination ability of features. Comparing
  1950	Index 2 with Index 4, CNC loss function can effectively make distance
  1951	between intra-class cross-modality samples close. In Index 5, when
  1952	our approach includes MGFE, MGFT, and CNC, the performance yields
  1953	69.79% and 73.41% in Rank-1 accuracy. These significant improvements demonstrate the effectiveness of each module in our approach.
  1954	As shown in Table 8, we find that significant improvements also occur
  1955	on the RegDB dataset.
  1956	
  1957	features can complement each other to improve the discrimination of
  1958	features. Similar improvements also appear in the comparison of Index
  1959	1 and Index 3. In Index 4, the performance can reach 69.79%/73.41%
  1960	in Rank-1/mAP when our approach includes three-granularity features. In Table 10, similar improvements have been achieved on the
  1961	RegDB dataset. These improvements prove that the complementarity
  1962	between multi-granularity features can reduce the irrelevant matching and ambiguous alignment problems caused by single-granularity
  1963	feature matching.
  1964	4.3.3. The results of testing at different granularities
  1965	In Tables 11 and 12, we have added the results of testing at different
  1966	granularities. We can observe that the performance of using a single
  1967	granularity for testing is lower than that of using multiple granularity fusion. This phenomenon can be attributed to the fact that each
  1968	granularity only contains partial discriminative information related to
  1969	a pedestrian’s identity, which is insufficient for effective identification.
  1970	For example, global features include information about the pedestrian’s
  1971	torso, patch features contain information about the pedestrian’s legs,
  1972	and pixel features include information about the pedestrian’s eyes.
  1973	By fusing these multi-granularity features, our approach can more
  1974	comprehensively and effectively identify the pedestrian’s identity.
  1975	
  1976	4.3.2. Effectiveness of three-granularity features
  1977	We evaluate the effectiveness of using a combination of global-,
  1978	patch-, and pixel-level features. In Tables 9 and 10, ‘‘Global’’, ‘‘Patch’’,
  1979	and ‘‘Pixel’’ represents using the global branch, patch branch, and pixel
  1980	branch, respectively.
  1981	Table 9 reports the results on the SYSU-MM01 dataset in all-search
  1982	and indoor-search modes. Comparing Index 1 with Index 2 in all-search
  1983	mode, the result using global and patch-level features is 6.18% better
  1984	than using only global-level features in Rank-1 accuracy. This improvement is largely attributed to the fact that global- and patch-granularity
  1985	8
  2325	4.5. Effectiveness of CNC loss function
  2326	4.5.1. Visual analysis of nearest neighbor features in CNC loss function
  2327	As shown in Fig. 6, we demonstrate the results of anchor sample
  2328	searching for the nearest-neighbor cross-modality samples within the
  2329	same class in CNC. It can be observed that our model not only effectively extracts the discriminative regions of the images but also matches
  2330	the most similar cross-modality features to the anchor samples using
  2331	the nearest-neighbor algorithm. For example, as shown in Fig. 6(b), our
  2332	model first captures the outline of the visible light pedestrian and then
  2333	9
  2334	
  2335	Pattern Recognition 164 (2025) 111510
  2336	
  2337	Y. Feng et al.
  2338	
  2339	Fig. 6. Visualization of neighbor features in CNC loss. The first row is the original image, and the second row is the features obtained by the model. The red box represents the
  2340	features of the nearest neighbor.
  3393	4.8.1. Visualization of qualitative results
  3394	As shown in Fig. 7, we visualize the Rank-10 search results for five
  3395	random selected pedestrian queries. In Fig. 7(a–c), our approach can
  3396	retrieve the true match images by exploiting the most discriminative
  3397	facial information. In Fig. 7(d), when facial features are missing, our
  3398	method is still able to search for the true match by backpack and
  3399	walking posture. In Fig. 7(e), when pedestrians have similar facial features, arm movements, and clothing logs, another granularity feature
  3400	(shoe colors) is used to achieve correct matching. Through a detailed
  3401	examination of these results, it becomes evident that our approach not
  3402	only maintains stability but also exhibits remarkable robustness against
  3403	a wide array of variations, including changes in pose, action, lighting,
  3404	and environmental conditions
  3405	
  3406	Fig. 8. Visualization result of heatmaps on the SYSU-MM01 dataset. (a) Input images,
  3407	(b) Results via the global branch, (c) Results via the patch branch, (d) Results via the
  3408	pixel branch, (e) Our approach.
  3409	
  3410	4.8.2. Visualization of feature heatmaps
  3411	We visualize feature maps from global branch, patch branch, pixel
  3412	branch, and combination of the three branches with Grad-CAM in
  3413	Fig. 8. In Fig. 8(b), we observe that the global branch focuses on large
  3414	regions of interest in pedestrian images, ignoring other granularity
  3415	12
  3416	
  3417	Pattern Recognition 164 (2025) 111510
  3418	
  3419	Y. Feng et al.
  3420	
  3421	Fig. 9. The visualization results of feature distributions from the baseline and our approach by t-SNE on the SYSU-MM01 dataset, where each color represents an identity. The
  3422	circles and asterisks indicate the features extracted from the visible and infrared modalities.
  3423	
  3424	features with discriminative ability. For example, the global branch
  3425	focuses on the body region of person 1, ignoring the most discriminative facial feature. In Fig. 8(c), the patch branch pays attention
  3426	to pedestrians’ contours and detailed regions, rather than large-scale
  3427	non-discriminative regions. For example, the face area, arms, and legs
  3428	of person 4 are captured. In Fig. 8(d), we can observe that more
  3429	comprehensive person features are paid attention to, e.g., the walking
  3430	posture and arm movements of person 5 and person 8. In Fig. 8(e),
  3431	our approach can effectively cover multi-granularity discriminative
  3432	features to obtain complete semantic information, which improves the
  3433	performance of the model. These results demonstrate that our approach
  3434	not only addresses the limitations of relying on single-granularity features, but also enhances the model’s ability to focus on pedestrians
  3435	with discriminative information at multi-granularity in a variety of

exec
/bin/zsh -lc "pdftotext 'Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf' - | nl -ba | sed -n '1082,1225p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1082	D. Ablation Study
  1083	To evaluate the effectiveness of each component introduced
  1084	in this paper, we conduct ablation experiments on SYSUMM01 and RegDB datasets. LADCL and LACCL indicate the
  1085	intra-modality loss in [22] and the cross-modality loss in [25],
  1086	respectively which achieve promising performance on the USVI-ReID. The results are reported in Tabel IV.
  1087	1) Baseline Settings: We employ the DCL framework [22]
  1088	as the baseline. While this work facilitates unsupervised VIReID learning, it overlooks the adverse impact of noisy labels.
  1089	As a result, the method achieves 36.18% and 45.00% in rank-1
  1090	on the SYSU-MM01 and RegDB datasets, respectively.
  1091	2) Effectiveness of CCS:
  1092	a) Effectiveness of CCE: The CCE is proposed to
  1093	seek reliable samples for intra-modality and cross-modality
  1094	learning. Compared to the baseline, the CCE improves the
  1095	performance of 6% and 7% in Rank-1 on SYSU-MM01 and
  1096	RegDB datasets, respectively as shown in Table IV (see index
  1097	1 & index 2). The main reason for such a great improvement
  1098	is that it captures reliable samples with closeness centrality
  1099	during forward propagation, alleviating the issue of noisy
  1100	labels.
  1101	b) Effectiveness of LRHCL : The LRHCL is designed to
  1102	learn modality-specific information and optimize the feature
  1103	extractor. As shown in Table IV, compared to LADCL , the performance is improved to 3% in Rank-1 and 1% in mAP on the
  1104	
  1105	SYSU-MM01, respectively. The main gain is achieved by the
  1106	design of reliable modality-specific memory, which captures
  1107	the reliable feature distribution of each pedestrian and achieves
  1108	immediate feature adjustment without any delay, enhancing the
  1109	learning of identity-consistent feature representation.
  1110	3) Effectiveness of HCM:
  1111	a) Effectiveness of BCLM: The BCLM is introduced
  1112	to establish reliable cross-modality correspondences between
  1113	two modalities, improving performance by 3% in Rank-1 and
  1114	3% in mAP on the SYSU-MM01, respectively. It selects the
  1115	most confident samples as centrality anchors and introduces
  1116	bidirectional consistency to dig up reliable cross-modality correspondences, thereby improving the accuracy and robustness
  1117	of cross-modality matching.
  1118	b) Effectiveness of LDRHL : Notably, LDRHL cannot
  1119	function independently, as it relies on the cross-modality
  1120	correspondences established by the previous module to guide
  1121	network training and optimization. When combined LDRHL
  1122	with GPRA, it achieved significant improvements on the
  1123	SYSU-MM01, demonstrating an increase of 17% in Rank-1
  1124	and 14% in mAP (see index 3 & index 5). This implies that
  1125	LDRHL makes efficient use of the cross-modality correspondences, enabling the model to learn modality-invariant and
  1126	discriminative features. When comparing LDRHL with LACCL
  1127	(see index 4 & index 5, index 6 & index 7), the former shows
  1128	superior performance. This can be attributed to our designed
  1129	memory which captures reliable features and exhibits robust
  1130	resistance to multiple cross-modality discrepancies.
  1131	E. Hyper-Parameter Analysis
  1132	To analyze the effect of hyper-parameters n, γ and µ, we
  1133	evaluate their performance by selecting various values for each
  1134	parameter on SYSU-MM01 dataset, as shown in Fig. 3.
  1135	1) Analysis of n: In Eq. (4), n determines the number of
  1136	nearest neighbors considered when calculating the closeness
  1137	centrality score. Through experiments, we find that when
  1138	n is set to values smaller than 11 or greater than 15, the
  1139	performance of the model significantly degrades. Therefore,
  1140	we present the results for different values of n ranging from
  1141	11 to 15 and observe that the model achieves the best performance when n = 13. A larger n considers more neighboring
  1142	information but potentially introduces noise, while a smaller n
  1143	focuses on the closest neighbors and fails to accurately reflect
  1144	the reliability of samples within clusters.
  1145	2) Analysis of γ: The parameter γ in Eq. (8) and Eq. (9)
  1146	controls the proportion of reliable samples selected from
  1147	
  1148	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
  1149	
  1150	JI et al.: LEARNING FROM YOURSELF TO OTHERS FOR UNSUPERVISED VISIBLE-INFRARED RE-IDENTIFICATION
  1151	
  1152	9833
  1153	
  1154	Fig. 3. The effect of hyper-parameter n, γ and µ with different values on SYSU-MM01 under all-search mode.
  1155	
  1156	Fig. 4. The t-SNE visualization of 10 randomly selected identities from SYSU-MM01. Circles and crosses represent features of visible and infrared images,
  1157	respectively. Different colors represent different identities.
  1158	
  1159	Fig. 5. The visualization of intra-class and inter-class distances of crossmodality features on SYSU-MM01.
  1160	
  1161	each cluster. Our experiments show that γ = 0.55 yields
  1162	the best performance on SYSU-MM01. Larger values of γ
  1163	may incorporate more low-quality samples, thus reducing the
  1164	overall reliability of selected instances, while smaller values
  1165	of γ could compromise the model’s generalization ability.
  1166	3) Analysis of µ: In Eq. (24), µ is the weighting parameter
  1167	to combine LRHCL and LDRHL . When µ is too small, the
  1168	model focuses primarily on intra-modality loss during training,
  1169	resulting in insufficient cross-modality matching ability and
  1170	hindering the establishment of robust cross-modality correspondences. Conversely, a higher µ causes the model to place
  1171	greater emphasis on cross-modality loss, making it challenging
  1172	to distinguish differences between persons within the same
  1173	modality. We observe that the model achieves the best accuracy when µ = 0.25, effectively coordinating the learning of
  1174	modality-invariant and identity-discriminative features.
  1175	F. Visualization
  1176	To demonstrate the effectiveness of HCCL, we first visualize
  1177	the distribution of the learned features by using t-SNE [74],
  1178	
  1179	Fig. 6. Visualization of top-5 retrieval results on SYSU-MM01. Green and
  1180	red boxes denote correct and incorrect matches, respectively.
  1181	
  1182	as illustrated in Fig. 4. The intra- and inter-class distances
  1183	of cross-modality features on the SYSU-MM01 dataset are
  1184	presented in Fig. 5. Additionally, the ranking list for the
  1185	SYSU-MM01 dataset is shown in Fig. 6.
  1186	1) T-SNE Visualization: In Fig. 4 (a), we observe that
  1187	there exist considerable distances between samples of the
  1188	same identity across different modalities, indicating significant cross-modality discrepancy. Furthermore, samples from
  1189	different identities are incorrectly assigned to the same class
  1190	(see the orange and blue dashed boxes), highlighting the issue
  1191	of noisy labels. Fig. 4 (b), Fig. 4 (c) and Fig. 4 (d) present the
  1192	distribution of ADCA [22], PGM [25] and RPLN [49], respectively, which are currently among the most influential works in
  1193	
  1194	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:55:27 UTC from IEEE Xplore. Restrictions apply.
  1195	
  1196	9834
  1197	
  1198	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025
  1199	
  1200	US-VI-ReID. They demonstrate encouraging improvement
  1201	with the samples between the two modalities gradually
  1202	converging. However, the issue of noisy labels remains unresolved.
  1203	In contrast, as shown in Fig. 4 (e), samples of the same
  1204	identity within the same modality are more tightly clustered in
  1205	the feature space, while samples from different modalities but
  1206	the same identity exhibit a more compact feature distribution.
  1207	Moreover, the issue of noisy labels is effectively mitigated.
  1208	This improvement indicates that our proposed HCCL effectively addresses this challenge, allowing the model to learn
  1209	identity-consistent intra-modality representations and establish
  1210	reliable cross-modality correspondences.
  1211	2) Intra- and Inter-Class Visualization: Meanwhile, as
  1212	shown in Fig. 5 (a) and Fig. 5 (b), it is obvious that the means
  1213	of intra-class and inter-class feature distributions are pushed
  1214	away (δ1 < δ2 ). This observation demonstrates that our method
  1215	effectively enhances the discriminative capability of crossmodality features. Moreover, the intra-class distance of HCCL
  1216	is significantly reduced compared with the baseline. This is
  1217	attributed to HCCL proactively identifying reliable samples
  1218	and discarding noisy samples during training, resulting in a
  1219	more compact distribution of intra-class samples in the feature
  1220	space and mitigating cross-modality discrepancy.
  1221	3) Ranking Visualization: We conduct the top-5 retrieval
  1222	ranking of some example queries on SYSU-MM01 with the
  1223	setting of infrared to visible, as shown in Fig. 6. We observe
  1224	that compared to the baseline retrieval results, our method
  1225	yields a greater number of correctly matched positive samples,

codex
我按本地 PDF 提取并读了方法和关键消融。列表里实际是 6 篇，下面逐篇拆。

**1. Instant Pose Extraction Based on Mask Transformer for Occluded Person ReID**

1. 触发观察：遮挡不是简单少了一块图像，而是会让可见人体部位和遮挡噪声混在一起，导致局部匹配错位。作者还观察到 Transformer 越深，patch 特征越相似，动态掩码很难生成有效的部位原型。
2. 重定义动作：他们把遮挡 ReID 从“抑制遮挡区域”改写成“即时提取可用姿态和可见部位原型”。关键词是 dynamic mask、prototype mask、valid subspace、non-occluded body parts、pose key-points。
3. 机制怎么长出来：如果问题是“遮挡下可见部位需要被重新定位和匹配”，那么动态掩码、姿态热图、局部原型、非遮挡部位匹配就能顺着长出来。但这篇机制绑定不算特别紧，因为模块较多，MAM、MACM、PAM、FMM 都在承担一部分解释，容易显得是组合拳。
4. 证据闭环：主要证据是 Occluded-Duke 上分模块消融，MTE 带来约 +5.2 mAP，LPTD 带来约 +6.3 mAP，完整模型约 +9.4 mAP；可视化显示 MAM 和 MACM 能定位遮挡相关部位；局部原型数量在 Occluded-Duke 上更细粒度更好。
5. reviewer 为什么买账：更买机制完整性和遮挡场景表现，不完全是视角。真正 novelty 来自“用动态原型掩码加姿态聚合做可见部位匹配”的组合，而不是单个模块。对我们有用的教训是，遮挡论文如果只是“加部位模块”，必须有强可视化和分模块消融支撑，否则视角不够硬。

**2. Interactive Sketch-Based Person ReID with Text Feedback**

1. 触发观察：草图能表达轮廓和结构，但表达不了颜色、性别、衣服语义等高层意图。已有 sketch+text 方法把草图和文本简单组合，既有模态干扰，又依赖昂贵的三模态数据。
2. 重定义动作：他们把 Sketch ReID 从“一次性草图检索”改写成“用户反馈驱动的交互检索”。关键词是 context assistance、feedback guidance、pseudo-word token、explicit sketch-text compositionality、textual token concatenation。
3. 机制怎么长出来：这个绑定很紧。既然用户反馈是文本，CLIP 文本空间又擅长组合语义，那么自然做法就是把草图映射成一个伪词 token，再和用户反馈 token 拼接，让 CLIP 文本编码器完成组合查询。
4. 证据闭环：他们不是只报涨点，而是问了三个对故事很关键的问题。任务适配 CLIP 是否必要，视觉到文本转换网络是否必要，伪词 token 是否真的保留草图语义。最关键的是用草图伪词去检索草图库，R@1 达到 95.64%，证明伪词不是随机向量，而确实携带草图语义。
5. reviewer 为什么买账：主要买视角。它把检索流程从静态查询改成了可交互流程，把草图和文本统一到文本 token 空间，机制很轻但叙事很顺。对我们有用的是，好的创新不一定复杂，关键是把“真实用户流程里的缺口”定义出来。

**3. Internal-External Context Interaction Network for Person ReID**

1. 触发观察：传统注意力主要利用单个样本内部上下文，自注意力只看样本自身的元素关系，容易只盯显著区域，忽略背包、衣服纹理等细微但有身份区分力的外部线索。
2. 重定义动作：他们把注意力问题从“样本内部哪里重要”改写成“样本内部局部上下文应该和跨样本潜在上下文交互”。关键词是 internal-external context interaction、inter-sample latent context、intra-sample local context、external learnable memory。
3. 机制怎么长出来：绑定比较紧。既然要引入跨样本上下文，就需要一个外部可学习记忆；既然还要保留样本自身结构，就用局部窗口提取内部上下文；然后把局部 query 投影到外部 key 上生成空间和通道权重。
4. 证据闭环：最关键消融是 IEI-S 对比普通 self-attention。两者都做位置特征重加权，但 IEI-S 多了跨样本上下文，结果在 Market1501 和 CUHK03 上更好且计算量更低。热图也对应了重定义，普通注意力偏下半身，IEI-S 能同时关注上身、背包、T 恤纹理等身份线索。
5. reviewer 为什么买账：视角和机制都能买账。它不是泛泛说“注意力更强”，而是明确指出自注意力的参考范围太窄，并用外部记忆修正这个范围。对我们有用的是，如果要做注意力类方法，最好改变“注意力依据什么信息产生”，而不是只换注意力形式。

**4. Latent Diffusion-Guided Feature Inpainting for Occluded Person ReID**

1. 触发观察：很多遮挡方法只是让模型适应遮挡或压低遮挡影响，但最终 embedding 仍然是被遮挡污染过的。标准重排序在大图库里还会被离群样本和噪声邻域放大错误。
2. 重定义动作：他们把遮挡从“需要鲁棒特征”改写成“特征空间里的结构化污染，需要反变换恢复干净身份表征”。关键词是 feature-level inpainting、inverse transform、clean identity-preserving embedding、de-occlusion、centroid-guided pruning。
3. 机制怎么长出来：如果接受“遮挡是特征污染”，扩散式去噪补全就很自然。DDO 用冻结 ViT 的干净特征作监督，把遮挡特征恢复成干净特征。HRR 则利用身份中心先筛候选，再做局部重排序。不过 HRR 依赖闭集身份结构，协议风险比 DDO 大。
4. 证据闭环：证据比较直接。可视化显示 DDO 把注意力从白墙、遮挡物拉回衣服、脸和可见身体；MSE、余弦误差和 SSIM 显示补全后特征更接近干净特征；DDO 插到 TransReID 和 Pass-ViT 也有增益。HRR 的证据更复杂，Occluded-Duke 上 mAP 从 81.8 到 90.6，但 Occluded-ReID 上会从 92.3 降到 90.1，作者也承认图库身份样本少或重遮挡时中心会偏。
5. reviewer 为什么买账：主要买问题重定义，“不要忍受遮挡，而是恢复被污染的特征”。但这篇也最容易被挑协议和监督来源，尤其是合成遮挡配对、闭集重排序、中心标签使用。对我们有用的是，“特征污染反演”是强视角，但测试期重排序不适合作主贡献。

**5. Learning From Yourself to Others for Unsupervised Visible-Infrared ReID**

1. 触发观察：无监督 VI-ReID 里，伪标签噪声不是只在 loss 反传时造成问题。噪声样本已经在前向传播、聚类原型、跨模态匹配阶段污染了表示。已有方法常用损失函数补救，但补救太晚。
2. 重定义动作：他们把问题从“如何缩小可见红外模态差异”改写成“先判断哪些伪标签可信，再从单模态可靠样本走向跨模态匹配”。标题里的 yourself to others 就是这个叙事。关键词是 noisy labels、forward propagation、closeness centrality、reliable homogeneous prototype、local-to-global hierarchical matching。
3. 机制怎么长出来：绑定很紧。既然要先学自己，就在每个单模态 cluster 内用 closeness centrality 找可靠样本，低可靠样本前向阶段直接丢掉。既然再学别人，就用中心锚点做双向一致局部匹配，再把可靠局部匹配送进全局最优传输，最后用异质对比损失学跨模态不变特征。
4. 证据闭环：消融非常贴问题定义。CCE 在 SYSU-MM01 和 RegDB 上 Rank-1 分别约 +6 和 +7；BCLM 约 +3 Rank-1 和 +3 mAP；GPRA 加 LDRHL 在 SYSU-MM01 上带来约 +17 Rank-1 和 +14 mAP。t-SNE 也不是泛泛展示聚类更好，而是专门显示已有方法虽然模态靠近了，但噪声标签还在，而 HCCL 缓解了这个问题。
5. reviewer 为什么买账：主要买视角。它抓住“噪声样本必须在前向阶段处理”这个具体观察，再把单模态可靠性和跨模态匹配串成自然流程。对我们最有参考价值的是，它的每个模块都对应一句可检验的话，而不是先有模块再找解释。

**6. Learning Multi-Granularity Representation with Transformer for VI-ReID**

1. 触发观察：可见红外两种模态的有效身份线索大小和位置不一致，可能是全局衣服颜色、局部姿态，也可能是鞋子或脸部这类很小区域。单一全局或单一部位粒度会造成 irrelevant matching 和 ambiguous alignment。
2. 重定义动作：他们把 VI-ReID 从“模态差异问题”改写成“跨模态、跨粒度的对齐问题”。关键词是 global-patch-pixel、multi-granularity、saliency consistency、cross-granularity alignment、nearest-neighbor clustering。
3. 机制怎么长出来：机制基本顺着重定义生长。既然单粒度不够，就抽 global、patch、pixel 三个层级；既然不同模态和粒度关注区域不一致，就用 saliency-enhanced Transformer 做同粒度跨模态融合和同模态跨粒度融合；既然同类跨模态距离仍远，就用 CNC 拉近最近邻同类样本。
4. 证据闭环：MGFE 在 SYSU-MM01 两种模式下 Rank-1 分别约 +6.26 和 +4.47；MGFT 和 CNC 继续提升；三粒度组合明显优于单粒度或两粒度。热图也服务于视角，global 关注大区域，patch 关注轮廓和局部，pixel 捕捉姿态和手臂动作，组合后覆盖更完整语义。
5. reviewer 为什么买账：买的是“单粒度导致错配”的视角加充分消融，但机制本身偏常规。它的创新来源不是 Transformer 分支，而是把 VI-ReID 的失败解释成粒度不匹配，并用三粒度实验证据把这个解释坐实。

这批论文共同套路很清楚：先抓一个具体失败现象，再把它改写成一个更高层的问题定义，让机制看起来像自然结果。强论文的关键不是模块多，而是有一个消融或可视化能直接证明“这个新问题定义是对的”，例如伪词确实有草图语义、外部上下文确实比自注意力多看身份线索、前向丢噪声确实缓解伪标签污染。对我们做 ReID 方法稿，最该学的是先找一个可被证伪的失败解释，再设计一个如果失败就会推翻叙事的证据闭环。
tokens used
229,658
我按本地 PDF 提取并读了方法和关键消融。列表里实际是 6 篇，下面逐篇拆。

**1. Instant Pose Extraction Based on Mask Transformer for Occluded Person ReID**

1. 触发观察：遮挡不是简单少了一块图像，而是会让可见人体部位和遮挡噪声混在一起，导致局部匹配错位。作者还观察到 Transformer 越深，patch 特征越相似，动态掩码很难生成有效的部位原型。
2. 重定义动作：他们把遮挡 ReID 从“抑制遮挡区域”改写成“即时提取可用姿态和可见部位原型”。关键词是 dynamic mask、prototype mask、valid subspace、non-occluded body parts、pose key-points。
3. 机制怎么长出来：如果问题是“遮挡下可见部位需要被重新定位和匹配”，那么动态掩码、姿态热图、局部原型、非遮挡部位匹配就能顺着长出来。但这篇机制绑定不算特别紧，因为模块较多，MAM、MACM、PAM、FMM 都在承担一部分解释，容易显得是组合拳。
4. 证据闭环：主要证据是 Occluded-Duke 上分模块消融，MTE 带来约 +5.2 mAP，LPTD 带来约 +6.3 mAP，完整模型约 +9.4 mAP；可视化显示 MAM 和 MACM 能定位遮挡相关部位；局部原型数量在 Occluded-Duke 上更细粒度更好。
5. reviewer 为什么买账：更买机制完整性和遮挡场景表现，不完全是视角。真正 novelty 来自“用动态原型掩码加姿态聚合做可见部位匹配”的组合，而不是单个模块。对我们有用的教训是，遮挡论文如果只是“加部位模块”，必须有强可视化和分模块消融支撑，否则视角不够硬。

**2. Interactive Sketch-Based Person ReID with Text Feedback**

1. 触发观察：草图能表达轮廓和结构，但表达不了颜色、性别、衣服语义等高层意图。已有 sketch+text 方法把草图和文本简单组合，既有模态干扰，又依赖昂贵的三模态数据。
2. 重定义动作：他们把 Sketch ReID 从“一次性草图检索”改写成“用户反馈驱动的交互检索”。关键词是 context assistance、feedback guidance、pseudo-word token、explicit sketch-text compositionality、textual token concatenation。
3. 机制怎么长出来：这个绑定很紧。既然用户反馈是文本，CLIP 文本空间又擅长组合语义，那么自然做法就是把草图映射成一个伪词 token，再和用户反馈 token 拼接，让 CLIP 文本编码器完成组合查询。
4. 证据闭环：他们不是只报涨点，而是问了三个对故事很关键的问题。任务适配 CLIP 是否必要，视觉到文本转换网络是否必要，伪词 token 是否真的保留草图语义。最关键的是用草图伪词去检索草图库，R@1 达到 95.64%，证明伪词不是随机向量，而确实携带草图语义。
5. reviewer 为什么买账：主要买视角。它把检索流程从静态查询改成了可交互流程，把草图和文本统一到文本 token 空间，机制很轻但叙事很顺。对我们有用的是，好的创新不一定复杂，关键是把“真实用户流程里的缺口”定义出来。

**3. Internal-External Context Interaction Network for Person ReID**

1. 触发观察：传统注意力主要利用单个样本内部上下文，自注意力只看样本自身的元素关系，容易只盯显著区域，忽略背包、衣服纹理等细微但有身份区分力的外部线索。
2. 重定义动作：他们把注意力问题从“样本内部哪里重要”改写成“样本内部局部上下文应该和跨样本潜在上下文交互”。关键词是 internal-external context interaction、inter-sample latent context、intra-sample local context、external learnable memory。
3. 机制怎么长出来：绑定比较紧。既然要引入跨样本上下文，就需要一个外部可学习记忆；既然还要保留样本自身结构，就用局部窗口提取内部上下文；然后把局部 query 投影到外部 key 上生成空间和通道权重。
4. 证据闭环：最关键消融是 IEI-S 对比普通 self-attention。两者都做位置特征重加权，但 IEI-S 多了跨样本上下文，结果在 Market1501 和 CUHK03 上更好且计算量更低。热图也对应了重定义，普通注意力偏下半身，IEI-S 能同时关注上身、背包、T 恤纹理等身份线索。
5. reviewer 为什么买账：视角和机制都能买账。它不是泛泛说“注意力更强”，而是明确指出自注意力的参考范围太窄，并用外部记忆修正这个范围。对我们有用的是，如果要做注意力类方法，最好改变“注意力依据什么信息产生”，而不是只换注意力形式。

**4. Latent Diffusion-Guided Feature Inpainting for Occluded Person ReID**

1. 触发观察：很多遮挡方法只是让模型适应遮挡或压低遮挡影响，但最终 embedding 仍然是被遮挡污染过的。标准重排序在大图库里还会被离群样本和噪声邻域放大错误。
2. 重定义动作：他们把遮挡从“需要鲁棒特征”改写成“特征空间里的结构化污染，需要反变换恢复干净身份表征”。关键词是 feature-level inpainting、inverse transform、clean identity-preserving embedding、de-occlusion、centroid-guided pruning。
3. 机制怎么长出来：如果接受“遮挡是特征污染”，扩散式去噪补全就很自然。DDO 用冻结 ViT 的干净特征作监督，把遮挡特征恢复成干净特征。HRR 则利用身份中心先筛候选，再做局部重排序。不过 HRR 依赖闭集身份结构，协议风险比 DDO 大。
4. 证据闭环：证据比较直接。可视化显示 DDO 把注意力从白墙、遮挡物拉回衣服、脸和可见身体；MSE、余弦误差和 SSIM 显示补全后特征更接近干净特征；DDO 插到 TransReID 和 Pass-ViT 也有增益。HRR 的证据更复杂，Occluded-Duke 上 mAP 从 81.8 到 90.6，但 Occluded-ReID 上会从 92.3 降到 90.1，作者也承认图库身份样本少或重遮挡时中心会偏。
5. reviewer 为什么买账：主要买问题重定义，“不要忍受遮挡，而是恢复被污染的特征”。但这篇也最容易被挑协议和监督来源，尤其是合成遮挡配对、闭集重排序、中心标签使用。对我们有用的是，“特征污染反演”是强视角，但测试期重排序不适合作主贡献。

**5. Learning From Yourself to Others for Unsupervised Visible-Infrared ReID**

1. 触发观察：无监督 VI-ReID 里，伪标签噪声不是只在 loss 反传时造成问题。噪声样本已经在前向传播、聚类原型、跨模态匹配阶段污染了表示。已有方法常用损失函数补救，但补救太晚。
2. 重定义动作：他们把问题从“如何缩小可见红外模态差异”改写成“先判断哪些伪标签可信，再从单模态可靠样本走向跨模态匹配”。标题里的 yourself to others 就是这个叙事。关键词是 noisy labels、forward propagation、closeness centrality、reliable homogeneous prototype、local-to-global hierarchical matching。
3. 机制怎么长出来：绑定很紧。既然要先学自己，就在每个单模态 cluster 内用 closeness centrality 找可靠样本，低可靠样本前向阶段直接丢掉。既然再学别人，就用中心锚点做双向一致局部匹配，再把可靠局部匹配送进全局最优传输，最后用异质对比损失学跨模态不变特征。
4. 证据闭环：消融非常贴问题定义。CCE 在 SYSU-MM01 和 RegDB 上 Rank-1 分别约 +6 和 +7；BCLM 约 +3 Rank-1 和 +3 mAP；GPRA 加 LDRHL 在 SYSU-MM01 上带来约 +17 Rank-1 和 +14 mAP。t-SNE 也不是泛泛展示聚类更好，而是专门显示已有方法虽然模态靠近了，但噪声标签还在，而 HCCL 缓解了这个问题。
5. reviewer 为什么买账：主要买视角。它抓住“噪声样本必须在前向阶段处理”这个具体观察，再把单模态可靠性和跨模态匹配串成自然流程。对我们最有参考价值的是，它的每个模块都对应一句可检验的话，而不是先有模块再找解释。

**6. Learning Multi-Granularity Representation with Transformer for VI-ReID**

1. 触发观察：可见红外两种模态的有效身份线索大小和位置不一致，可能是全局衣服颜色、局部姿态，也可能是鞋子或脸部这类很小区域。单一全局或单一部位粒度会造成 irrelevant matching 和 ambiguous alignment。
2. 重定义动作：他们把 VI-ReID 从“模态差异问题”改写成“跨模态、跨粒度的对齐问题”。关键词是 global-patch-pixel、multi-granularity、saliency consistency、cross-granularity alignment、nearest-neighbor clustering。
3. 机制怎么长出来：机制基本顺着重定义生长。既然单粒度不够，就抽 global、patch、pixel 三个层级；既然不同模态和粒度关注区域不一致，就用 saliency-enhanced Transformer 做同粒度跨模态融合和同模态跨粒度融合；既然同类跨模态距离仍远，就用 CNC 拉近最近邻同类样本。
4. 证据闭环：MGFE 在 SYSU-MM01 两种模式下 Rank-1 分别约 +6.26 和 +4.47；MGFT 和 CNC 继续提升；三粒度组合明显优于单粒度或两粒度。热图也服务于视角，global 关注大区域，patch 关注轮廓和局部，pixel 捕捉姿态和手臂动作，组合后覆盖更完整语义。
5. reviewer 为什么买账：买的是“单粒度导致错配”的视角加充分消融，但机制本身偏常规。它的创新来源不是 Transformer 分支，而是把 VI-ReID 的失败解释成粒度不匹配，并用三粒度实验证据把这个解释坐实。

这批论文共同套路很清楚：先抓一个具体失败现象，再把它改写成一个更高层的问题定义，让机制看起来像自然结果。强论文的关键不是模块多，而是有一个消融或可视化能直接证明“这个新问题定义是对的”，例如伪词确实有草图语义、外部上下文确实比自注意力多看身份线索、前向丢噪声确实缓解伪标签污染。对我们做 ReID 方法稿，最该学的是先找一个可被证伪的失败解释，再设计一个如果失败就会推翻叙事的证据闭环。
