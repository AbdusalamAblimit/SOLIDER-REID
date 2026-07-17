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
session id: 019ef71f-a71a-71a3-a0f4-8df18197fbeb
--------
user
你是 ReID 论文创新挖掘员 13/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(13/20):
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Multi-year long-term person re-identification using gait and HAR features.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则核对当前目标和状态文件，然后只用 `pdftotext -l 3` 读取这批论文的摘要和引言，不展开全文。
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
/bin/zsh -lc "sed -n '1,220p' ../../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
sed: ../../../../GOAL.md: No such file or directory

exec
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

 exited 1 in 0ms:
sed: ../../../../CLAUDE.md: No such file or directory

codex
状态文件读到的是当前主线已经转向更高层的 ReID 问题设定，不是简单堆模块；这和这次“从论文里反推强创新点”的目标一致。我刚才相对路径多退了一层，补读顶层目标后再开始抽取论文。
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
/bin/zsh -lc "rg --files -g 'Multi*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf

codex
下面进入论文抽取。我会按文件逐篇只取前三页，重点看它把问题怎么定义、机制怎么包装、证据链怎么搭，而不是复述方法细节。
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
/bin/zsh -lc "pdftotext -l 3 'Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -l 3 'Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ECAI 2025
I. Lynce et al. (Eds.)
© 2025 The Authors.
This article is published online with Open Access by IOS Press and distributed under the terms
of the Creative Commons Attribution Non-Commercial License 4.0 (CC BY-NC 4.0).
doi:10.3233/FAIA250870

707

Multi-Branch Clothes-Agnostic Feature Learning
for Cloth-Changing Person Re-Identification
Shijuan Huanga, * and Hefei Lingb, **
a Huazhong University of Science and Technology
b Huazhong University of Science and Technology

ORCID (Shijuan Huang): https://orcid.org/0009-0000-2177-5110, ORCID (Hefei Ling):
https://orcid.org/0000-0001-6797-7412
Abstract. Person Re-Identiﬁcation (Re-ID) is crucial for video
surveillance and multi-camera tracking, yet traditional methods
struggle with clothing changes that undermine their reliability. This
paper introduces a novel multi-branch clothes-agnostic feature learning framework to address cloth-changing person re-identiﬁcation
(CC-ReID), which comprises two key modules: Multi-grained
Clothes Caption Generation (MCG), and Multi-Branch ClothesAgnostic Feature Extraction (MAE). MCG leverages Large VisionLanguage Models to generate diverse coarse-to-ﬁne clothing descriptions, reducing the impact of clothing on feature extraction. MAE
employs a dual-branch architecture combining Semantic-Guided
Feature Extraction (SGE) and Parsing Image Feature Extraction
(PIE) to focus on identity-related features while minimizing dependence on clothing characteristics. Extensive experiments on benchmark datasets demonstrate that our method achieves state-of-the-art
performance for CC-ReID tasks, showcasing our method’s effectiveness in real-world applications.

1

Introduction

Person Re-identiﬁcation (Re-ID) [23], a cornerstone technology for
multi-camera surveillance systems and cross-view tracking, focuses
on robustly associating individuals across non-overlapping camera
networks. Conventional Re-ID methodologies predominantly rely on
appearance-based features—such as clothing color distributions, texture patterns, and stylistic attributes—operating under the assumption that subjects retain consistent attire within spatially or temporally constrained environments. While effective in controlled scenarios, these approaches face critical limitations in practical deployments: pedestrians frequently alter clothing across extended time
spans or diverse geographical contexts, rendering appearance-centric
features unstable and severely compromising the practical utility of
traditional models. Consequently, cloth-changing Re-ID (CC-ReID)
has become a critical research area, focusing on accurately identifying individuals despite changes in clothing.
To address the challenge of clothing variations, researchers have
pursued two primary strategies: suppressing clothing interference
and extracting clothes-agnostic features to improve recognition accuracy and robustness. The ﬁrst line of work employs feature disentanglement frameworks to isolate apparel-related attributes from
∗ Corresponding Author. Email: shijuan_huang@hust.edu.cn
∗∗ Corresponding Author. Email: lhefei@hust.edu.cn

identity-discriminative representations. For instance, CAL [10] introduces a clothes-agnostic learning paradigm that penalizes clothing
classiﬁcation accuracy through adversarial training, thereby forcing
the model to discard transient apparel cues while preserving stable
identity semantics. The second strategy leverages multi-modal biometric features insensitive to clothing changes. GI-ReID [17] use gait
features, 3DSL [2] reconstructs 3D shape features, and FSAM [13]
and PRCC [35] utilize silhouette information. However, these methods are complex and often eliminate useful color information. CCAF
[19] uses textual descriptions and pre-trained models like CLIP [26]
to extract clothes-agnostic features, but the semantic gap between
text and image introduces noise.
Text descriptions can capture ﬁne-grained features from images,
making them valuable for CC-ReID. However, using CLIP introduces a semantic gap between text and image, reducing recognition
effectiveness. Furthermore, these methods focus on non-clothing descriptions such as body shape and posture, but these descriptions are
often imprecise, which can weaken model performance. To address
this, we leverage generated clothing descriptions through an attention mechanism and use image parsing techniques to obtain clothingirrelevant features. This approach helps the model learn more robust
identity features while minimizing dependence on clothing-speciﬁc
characteristics.
First, we use various Large Vision-Language Models (LVLMs)
to generate clothing descriptions with different granularities, from
coarse to ﬁne. To ensure accuracy, we apply human parsing algorithms [18] to isolate clothing regions in images, allowing LVLMs
to focus speciﬁcally on these areas. Given that generated descriptions may contain noise, we ﬁlter them using cosine similarity. This
process yields multi-granular, precise clothing descriptions for each
image, which are crucial for reducing the inﬂuence of clothing on
feature extraction.
Next, we adopt a Multi-Branch Clothes-Agnostic Feature Extraction approach to minimize dependence on clothing features and focus
on identity-related features. This approach contains two main components: Semantic-Guided Feature Extraction (SGE) and Parsing Image Feature Extraction (PIE). In SGE, we input generated clothing
captions into a text encoder to extract textual features, then apply an
attention mechanism to reduce the inﬂuence of clothing characteristics on image features. In PIE, we use parsing algorithms to identify
and mask clothing regions, creating additional training samples for
the image encoder. This multi-branch design encourages the model

708

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

to focus on persistent features like body structure and posture, rather
than speciﬁc clothing colors and styles.
Our contributions are summarized as follows:
• We use LVLMs to generate precise coarse-to-ﬁne captions for the
clothes of each image, mitigating the impact of clothing changes
on recognition.
• We employ a multi-branch architecture: semantic-guided feature
extraction and parsing image feature extraction, diminishing the
inﬂuence of clothing in feature extraction.
• Extensive experiments on two widely used CC-ReID datasets
demonstrate that our method achieves state-of-the-art performance.

2
2.1

Related work
Person Re-identiﬁcation

Person Re-identiﬁcation (Re-ID) is a fundamental computer vision
task focused on retrieving target individuals across distributed camera networks with non-overlapping ﬁelds of view. Early works [8]
establish early methodologies for viewpoint-invariant feature extraction, emphasizing handcrafted descriptors robust to pose variations and illumination changes. The ﬁeld underwent a paradigm shift
with the integration of deep neural networks, which enabled end-toend learning of discriminative representations. A milestone work by
Zheng et al. demonstrates this potential through a dedicated CNN
architecture optimized for Re-ID tasks.
Subsequently, researchers turn their attention to addressing challenges like domain adaptation and cross-view matching. Wei et al.
[32] pioneer GAN-based frameworks to align feature distributions
between source and target domains, effectively mitigating performance degradation in unseen environments. Concurrently, Luo et al.
[22] tackle fragmented label spaces across domains through a novel
cross-domain mixup strategy, which interpolates both features and
labels to synthesize intermediate samples, thereby enhancing model
robustness to domain shifts.

fusion mechanisms that balance semantic guidance with visual speciﬁcity in CC-ReID systems.

2.3

In recent years, LVLMs have signiﬁcantly improved multimodal data
processing by leveraging the capabilities of Large Language Models,
enabling efﬁcient image-to-text conversion and generation. QwenVL [1] excels in object detection and localization, providing robust cross-language visual understanding. InternVL2.5 [4] optimizes
cross-modal interactions, making it suitable for advanced applications. MiniGPT-4 [42] aligns a pre-trained vision encoder with LLMs
via ﬁne-tuning a single linear layer, achieving efﬁcient multimodal
understanding. MiniGPT-5 [41] combines Stable Diffusion mechanism with LLMs through special visual tokens known as "generative vokens", introducing a proﬁcient multi-modal generation. BLIP3
[34] introduces a 3-stage training paradigm and uses next token prediction to unify the training objective, enhancing both training efﬁciency and model performance. LLaVA [21] enhances performance
in complex tasks through instruction tuning with an open-source
dataset.

3

Cloth-Changing Person Re-identiﬁcation

The core challenge in CC-ReID lies in extracting identity-consistent
features resilient to apparel variations. Some methods employ adversarial learning to decouple clothing attributes from identity semantics. For instance, CAL [10] employs an adversarial loss function
to penalize clothing classiﬁcation, extracting clothing-irrelevant features. Other studies exploit complementary biometric modalities inherently invariant to clothing changes. Gait-based approaches such
as GI-ReID [17] analyze temporal motion dynamics in walking sequences, while MBUNet [38] fuses posture estimation with spatial
attention to capture anatomical invariants. Contour-driven methods
like FSAM [13] employ semantic parsing networks to align silhouette boundaries, and 3DSL [2] reconstructs 3D mesh representations to model body shape topology. However, these techniques often
demand complex preprocessing pipelines (e.g., pose estimation or
depth sensing) and risk discarding discriminative color/texture features crucial for short-term identiﬁcation. To mitigate these limitations, hybrid frameworks like CCAF [19] integrate CLIP’s visionlanguage capabilities [26] with textual prompts to guide clothesagnostic feature learning. Nevertheless, the inherent semantic gap
between coarse textual descriptions and ﬁne-grained visual patterns
introduces alignment noise. This underscores the need for adaptive

Method

In this section, we introduce our proposed method, which consists of
two modules: the Multi-grained Clothes Caption Generation (MCG)
and the Multi-Branch Clothes-Agnostic Feature Extraction (MAE).
Figure 1 provides an overview of the framework. The MCG module
generates coarse-to-ﬁne clothes captions to guide the MAE module
in suppressing clothing-related features. The MAE module employs
a multi-branch architecture to extract clothes-agnostic features.

3.1
2.2

Large Vision-Language Models

Multi-grained Clothes Caption Generation

Previous studies attempt to extract clothes-agnostic features by introducing text descriptions, focusing on inherent characteristics like
body shape while avoiding clothing details. However, these descriptions often lack speciﬁcity and unique identifying features. To address this, we leverage LVLMs to generate detailed clothing descriptions from images. We then use a text encoder to extract clothing features from these descriptions and apply an attention mechanism to reduce the inﬂuence of clothing characteristics on image features. This
approach enables a more accurate and robust extraction of clothesagnostic features.
We use human parsing algorithms to isolate clothing areas in images and direct the LVLMs to focus on these areas. To enhance description diversity, we employ multiple open-source LVLMs, including Qwen2-VL, BLIP3, MiniGPT-4, and InternVL2.5.
To generate descriptions with varying levels of detail, we use
prompts ranging from broad to speciﬁc. A broad prompt like "Please
brieﬂy summarize the person’s clothing in one sentence." leads
LVLMs to produce succinct summaries, highlighting the most prominent clothing features while omitting ﬁner details. In contrast, a detailed prompt such as "Provide a detailed description of the outﬁt,
specifying its color, type, style, and material" encourages rich and
comprehensive clothing descriptions.

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

Figure 1.

709

Overview of our proposed framework, which consists of two modules: the Multi-grained Clothes Caption Generation (MCG) module and the
Multi-Branch Clothes-Agnostic Feature Extraction (MAE) module.

description. Thus, we select the two descriptions with the highest
scores for subsequent model training.

Example Clothes Caption with VLM
Broad Clothes Description
Input: Please brieﬂy summarize the person’s clothing in one
sentence.

3.2

LVLM: A white short-sleeved T-shirt paired with black
cropped pants and red sneakers.

Multi-Branch Clothes-Agnostic Feature
Extraction

Detailed Clothes Description
Input: Provide a detailed description of the outﬁt, specifying
its color, type, style, and material.
LVLM: A white cotton short-sleeve crewneck T-shirt
(minimalist design) paired with black mid-calf cotton trousers
and red mesh-paneled sneakers with white rubber soles,
creating a casual monochromatic contrast.
Using four LVLMs and two prompts, we generate eight distinct
clothing descriptions for each image. However, due to the inherent unpredictability of LVLMs, these captions may contain mismatches or inaccuracies. To select clean clothing descriptions, we
input the descriptions and their corresponding images into the pretrained CLIP model’s text and image encoders, generating text features Ft and image features Fv . We calculate the cosine similarity to
obtain a matching score s as follows:
s = sim(Fv , Ft ) =

F v · Ft
.
Fv  × Ft 

(1)

The higher its matching score with the image, the more accurate the

We employ a multi-branch architecture comprising two components:
Semantic-Guided Feature Extraction (SGE) and Parsing Image Feature Extraction (PIE).
Semantic-Guided Feature Extraction. Given a person image
I ∈ RW ×H×C , the image is divided into non-overlapping patches,
with the number of patches N = H × W/P 2 , where P is the
patch size. A trainable linear projection layer maps these patches
to one-dimensional vectors {fiv }N
i=1 . A learnable [CLS] vector is
prepended to this sequence. The sequence is then processed by
the EVA-02 image encoder [7] to extract visual features Fv =
v
v
v
{fcls
, f1v , . . . , fN
}, where fcls
represents the global feature and
v
v
{f1 , . . . , fN } represent local features.
For each person image, we obtain two clothing descriptions from
the MCG module. Each description is tokenized using lower-cased
byte pair encoding (BPE) with [SOS] and [EOS] tokens added at
the beginning and end, respectively. These tokenized sequences are
processed through CLIP’s text encoder to generate textual features.
The text features from the two descriptions are averaged to produce
t
t
t
the ﬁnal text feature Ft = {fsos
, f1t , . . . , fM
, feos
} for the image.
First, we use the LayerNorm to normalize the visual features Fv

and the text features Ft , resulting in the normalized features Fv and


 succeeded in 0ms:
ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3315-6701-9/26/$31.00 ©2026 IEEE | DOI: 10.1109/ICASSP55912.2026.11464029

MULTI-GRANULARITY ATTRIBUTE PROMPT LEARNING FOR CLOTH-CHANGING
PERSON RE-IDENTIFICATION
Xu Li, Hefei Ling, Shijuan Huang, Chenxing Zhao
Huazhong University of Science and Technology, Wuhan, China
ABSTRACT
Cloth-changing person re-identification (CC-ReID) addresses
the practical need to recognize pedestrians across cameras after clothing changes. Existing methods use biological traits
and clothing labels to extract invariant features but underutilize semantic attributes and require costly manual annotation. To overcome these limitations, we propose a Multigranularity Attribute Prompt Learning (MAPL) method.
MAPL incorporates visual deep prompts in a dual-branch
network (raw and clothing-erased images) to learn clothingirrelevant representations. It generates pseudo clothing labels
through attribute recognition to reduce annotation dependency. Using these pseudo-labels and attribute texts, it enhances identity-related attributes while suppressing clothingrelated ones. This is achieved through feature disentanglement at both the caption and attribute levels. Experiments
show that MAPL achieves state-of-the-art performance,
outperforming the second-best method by 3.2%/13.0% in
Rank-1 and 4.0%/18.3% in mAP on PRCC and LTCC under
clothing-changing settings. The source code is released at
https://github.com/IPMD2025/MAPL.
Index Terms— Cloth-Changing Person ReID, MultiModal Fusion, Attribute, Prompt Learning.
1. INTRODUCTION
Person re-identification (ReID) aims to identify individuals across non-overlapping camera views in surveillance scenarios, playing an essential role in smart city development
and intelligent security systems [1, 2, 3]. Traditional ReID
methods primarily focus on short-term scenarios under the
assumption that the same individual retains the same clothing. However, clothing changes are common in real-world
scenarios and may even be used intentionally to evade surveillance. Most ReID models also over-rely on apparel features
[4]. Therefore, CC-ReID, which operates over longer time
spans, is both more challenging and practically significant.
This work is supported in part by the Natural Science Foundation of
China under Grant 62372203 and 62302186, in part by the Major Scientific and Technological Project of Shenzhen (202316021), in part by the National key research and development program of China(2022YFB2601802),
in part by the Major Scientific and Technological Project of Hubei Province
(2022BAA046, 2022BAA042).

979-8-3315-6701-9/26/$31.00 ©2026 IEEE

Existing methods can be broadly categorized into unimodal and multimodal approaches. Unimodal methods
[5, 6, 7, 8] disentangle clothing-related features solely from
RGB images. Multimodal methods [9, 10, 11, 12, 13] enhance clothing-invariant feature extraction by incorporating
additional cues like sketches, face, gait, clothing-erased images, attributes, and textual prompts. However, they fail to
effectively utilize attributes, particularly neglecting attribute
texts. AD ViT [14] utilizes only clothing-related attributes,
while MADE [15] employs identity-relevant attributes. Both
are limited to one-hot attribute labels and overlook attribute
texts containing richer semantic information. CCAF [16] and
MIPL [17] generate text from images but lack attribute information. Moreover, their reliance on clothing labels increases
manual annotation costs.
In this paper, We propose a dual-branch architecture based
on visual deep prompt tuning [18], which leverages attribute
labels and attribute texts to compel the model to focus on
identity-relevant attributes while suppressing clothing-related
information. First, to enhance generalization and achieve
pixel-level disentanglement, clothing-erased images and visual deep prompt tuning (VLP deep) are introduced as the
baseline. Second, to address the lack of high-level semantic
guidance in pixel-level disentanglement, an attribute textual disentanglement module is implemented by reinforcing
identity-relevant attributes and suppressing clothing-related
attributes. A BNNeck layer [2] is incorporated to separate
image tuning from attribute disentanglement, preventing corruption of the alignment knowledge.
Our main contributions can be summarized as follows:(1)
A visual deep prompt tuning method is introduced, and a
separation mechanism is incorporated to buffer potential misalignment caused by image tuning and attribute disentanglement. (2) A multi-granularity attribute disentanglement module is proposed, which enhances identity-relevant attributes
while suppressing clothing-related information. This compensates for the lack of high-level semantics in pixel-level disentanglement and reduces the reliance on manually annotated
clothing labels. (3) Extensive experimental results demonstrate that the proposed MAPL achieves new state-of-the-art
(SOTA) performance.

10942

ICASSP 2026

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

Mask images

2. METHOD

Lmap , Lman , Lmsp , Lmsn

...

Transformer

BN

1

0
Sneaker
Attributes

Input

Fvt
CAT

Fv

A pedestrian in a short-sleeved top
A pedestrian in sneakers

Identity- related

Head

Clothing- related

Feature extraction

Lce

Lap
Lan

Fvt

[cls]G token
Prompt token

Lsp

The man under the age of 30
A Pedestrian with short hair, no
attachments, short sleeves,
trousers and sneakers.

Lmce

Identity
classification

BN

Clothing- related
Text Encoder

Jacket

A male pedestrian
A pedestrian under the age of 30

...

0

Head

Ltri

...

where, Mic is the mask matrix, and ⊙ denotes the Hadamard
product. It is worth noting that certain regions corresponding
to attributes such as shoes and hair are not masked. Compared to other clothing attributes, these regions exhibit lower
variability, and directly erasing them could impair ReID performance. However, they may still retain clothing style information (e.g., short sleeves vs. long sleeves, hair). Therefore,
we adopt coarse-grained label-level disentanglement instead
of fine-grained pixel masking in subsequent modules.
Attribute Recognition. AAPAR [20] achieves SOTA
performance on standard PAR datasets including PETA, PA100K, RAPv2, as well as the competition dataset UPAR,
demonstrating strong cross-dataset generalization capability. Therefore, we employ the AAPAR model pre-trained on
PETA to identify attributes. Following the standard configuration, we recognize 7 attribute groups (35 attributes). The
attribute groups consist of age, gender, headwear, upper-body
clothing, lower-body clothing, shoes, and attachments. Age
and gender are considered identity-relevant attributes, while
the others are categorized as clothing-related attributes. The
a
attribute labels are defined as yij
, where i ∈ {1, 2, . . . , N } , j ∈
{1, 2, . . . , 35}.
Pseudo label generation. CC-ReID requires recognizing
different individuals while disregarding clothing variations

Less30

Caption-level

(1)

1

BN

Lmtri

Mutual
Learning

Identity- related

Attribute-level

xsi = xi ⊙ Mic ,

Male

...

Given a training dataset D = (xi , yiid )|N
i=1 , where xi denotes the i-th image, N represents the number of pedestrian
images, and yiid is the corresponding person identity label.
Clothing-erased images generation. The SCHP [19]
model is employed to parse pedestrian images into 20 semantic categories, including background, hair, upper clothing,
etc. Regions corresponding to hair, face, arms, legs, and
shoes remain unmasked. All other regions are defined as
clothing-related areas, and their pixels are set to zero to generate the masked image.

AAPAR

pseudo-label

BN

 24

...

2.2. Auxiliary Information Generation

...

Transformer

Raw images

The proposed MAPL is a dual-branch network consisting of a raw image branch and a masked image branch. ,
As shown in Figure.1, it comprises three main components:
1) Auxiliary information generation; 2) Multimodal feature
extraction module; and 3) A classification module. In the
auxiliary information generation stage, clothing-erased images, one-hot pedestrian attribute labels, pseudo-labels, and
attribute texts are generated. In the feature extraction module, features are extracted separately from the raw images,
masked images, caption-level texts, and attribute-level texts.
The classification module comprises person identity recognition, attribute disentanglement, and branch interaction.

Prompt

SCHP

Transformer

Prompt

2.1. Preliminaries

Attribute Decoupling for
Masked Images

Image Encoder

...

Patch token
CAT

Lsn

Attribute Decoupling
for Raw Images
Classification

Fig. 1. Overview structure of the MAPL framework.
within the same identity. Therefore, identity labels are integrated during pseudo-label generation. They are concatenated
with attribute labels to form attribute-level pseudo-labels, and
with all clothing-related attribute labels to form caption-level
pseudo-labels.
Attribute Text Generation. The attribute labels are converted into attribute texts using hand-crafted templates. For
each image, 5 invariant attribute-level texts Tap are generated (e.g., “A pedestrian under the age of 30”), along with
30 clothing-related attribute-level texts Tan (e.g., “A pedestrian without leather shoes”). Each image is also associated
with one positive caption-level text Tsp that includes age and
gender (e.g., “A man under the age of 30”), and one negative
caption-level text Tsn that describes other clothing attributes
(e.g., “A pedestrian with short hair, no attachments, ...”).
2.3. Multi modal feature extraction
2.4. Overview
Image feature extraction. Visual deep prompts are introduced to enhance generalization capability. Specifically,
learnable random prompt tokens xjprompt are appended to the
input of each Transformer layer:
h
i
xjin = xjcls , xjprompt , xjimg , j ∈ {0, 1, . . . , L − 1} , (2)
where xjprompt ∈ RL×M ×1×C , the notation [, ] indicates the
concatenation operation and M is the length of the prompt
tokens. During the iterative computation of the Transformer
layers, xj+1
prompt replaces the prompt tokens output from the
j-th Transformer layer. The output of the final layer is passed
through layer normalization and a modality-specific projection to obtain the pedestrian image features. Its class token,
used for person classification, is denoted as Fvt , while the
m
features of the masked image are defined as Fvt
. In this
work, a BNNeck layer is used to separate image classification

10943
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

from modal alignment, preventing the disruption of alignment
knowledge during image tuning.
Fv = BN N eck(Fvt )

Table 1. Comparison with SOTA Methods on PRCC, LTCC.
Methods

(3)
AFL [6]
FIRe2 [22]
MADE [15]
CAL [5]
AIM [7]
MCSC [23]
CCAF [16]
DIFIF [24]
ours(MAPL)

Text feature extraction. We employ a frozen CLIP [21]
text encoder to extract textual features at both the caption
level and the attribute level, encompassing both identityrelevant and clothing-relevant attributes, denoted as FT sp ,
FT sn , FT ap , and FT an .

PRCC
CC
SC
R1 mAP
R1
mAP
57.4 56.5
100
99.8
65
63.1
100
99.5
67.5 64.2
100
98.9
55.2 55.8
100
99.8
57.9 58.3
100
99.9
57.8 57.3
99.8
99.8
70.4 63.7
99.9
98.4
71.3 67.0
100
99.5
74.5 71.0
99.17 98.36

LTCC
CC
General
R1 mAP R1
mAP
42.1 18.4 74.4 39.1
44.6 19.1 75.9 39.9
46.9 25.0
82.2 49.3
40.1 18.0 74.2 40.8
40.6 19.1 76.3 41.1
42.2 19.4 73.9 40.2
42.9 20.1 75.3 41.3
50.5 25.1
80.9 46.2
63.5 43.4
83.5 62.73

2.5. Classification Module
Identity classification. To ensure that the pedestrian
features retain identity discriminability, the hard example
mining-based triplet loss and the cross-entropy loss are applied to constrain image features. Lid and Lm
id denote the
identity classification losses for the raw image and the masked
image, respectively.
Attribute decoupling. A multi-granularity attribute disentanglement mechanism is introduced in the multimodal
space to further eliminate clothing-related interference. It
comprises focusing on identity-relevant attributes and suppressing clothing-relevant attributes. 1) Focusing on identityrelevant attributes. Since feature fusion can enhance the representation of consistent positional biases and enable precise
intervention [7], for the original images, we concatenate the
image features with the identity-relevant attribute texts. The
triplet loss based on attribute pseudo-labels is then applied to
enforce focus on the invariant attributes of the pedestrian.
5

Lap =

1X
(Ltri ([Fv , FT ap ] , yipse a ))
5 i=1

(4)

To enrich the granularity of semantic descriptions, captionlevel focusing on identity-relevant attributes is introduced in
the same manner.

Lsp = Ltri [Fv , FT sp ] , yiid
(5)
The sum of these two losses is denoted as Lp . For the
clothing-erased images, considering the minimal residual
clothing information, only label-based constraints are applied.
5
X
(6)
Lm
=
(Ltri (Fv , yipse a ))
ap
i=1

Similarly, the final loss for the masked image branch is
denoted as Lm
2) Suppression of clothing-relevant atp .
tributes. By explicitly suppressing clothing-relevant attributes, clothing-related interference is further eliminated.
The specific calculation method is consistent with that used
for focusing on identity-relevant attributes, but suppression is
achieved through a negative loss based on clothing pseudolabels. The masked branch still employs only one-hot labels

for adversarial training. The final adversarial losses, denoted
as Ln and Lm
n . For example:
Lsn = −Ltri ([Fv , FT sp ] , yipse s ) .

(7)

Branch interaction. Mutual learning [16] is adopted to
ensure semantic consistency between the image features of
the two branches.
P

Lvcon =

1 X
2
∥ci − cm
i ∥ ,
P i=1

(8)

where, ci and cm
i represent the within-batch centroids of the
image features BN (Fv ) from the two branches, respectively.
m
The image features in the multimodal space, Fvt and Fvt
, also
vt
participate in mutual learning, yielding the loss Lcon . The
sum of the losses from the two spaces yields Lcon .
The overall training loss for optimizing the network parameters is defined as follows:
m
m
Ltotal = Lid +Lm
id +α×(Lp +Lp +Ln +Ln )+Lcon , (9)

where α is a hyperparameter, set to 0.1 on the PRCC dataset
and 0.2 on the LTCC dataset. During inference we only adopt
the image-modal space features from the original image
branch.
3. EXPERIMENTS
3.1. Experimental Settings
We conduct experiments on standard CC-ReID datasets,
PRCC [9] and LTCC [4], using Rank-1 accuracy and mean
Average Precision (mAP) for evaluation. Following prior
studies, the experiments include three settings: 1) Clothing
Change (CC): indicating that the same identity wears different clothing in the query and gallery images; 2) Same
Clothing (SC): indicating that the same identity wears identical clothing in the query and gallery images; 3) General:
indicating that the same identity may appear in either the
same or different clothing across query and gallery images.

10944
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Multi Queue for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10889622

Multi Queue for Unsupervised Person
Re-identification
Zhenyuan Lin †

Shengyong Xie †

Danhua Liu *

Guangzhou Institute of Technology School of Computer Science and information Security School of Artificial Intelligence
Xidian University
Guilin University Of Electronic Technology
Xidian University
Xian, China
Guilin, China
Guangzhou, China
dhliu@xidian.edu.cn
2416057194@qq.com
linzhenyuan@stu.xidian.edu.cn

Weikun Li

Ang Gao

Yubo Dong

School of Computer Science and information Security
Guilin University Of Electronic Technology
Guilin, China
liweikun1105 @163.com

School of Artificial Intelligence
Xidian University
Xian, China
anggao@stu.xidian.edu.cn

School of Artificial Intelligence
Xidian University
Xian, China
ybdong@stu.xidian.edu.cn

Abstract—Recently, cluster-based methods have achieved significant success in unsupervised re-ID tasks. The hierarchical
clustering algorithm, exemplified by SpCL, has been widely
adopted in unsupervised cross-domain adaptation and unsupervised learning. The momentum-based feature update mechanism
in SpCL has been integrated into various algorithms, achieving
notable results in subsequent studies. In this paper, we propose
a multi-queue feature updating algorithm that stores feature
vectors corresponding to person IDs in multiple queues. Random
sampling is then applied to construct the negative sample matrix
for contrastive loss, addressing the limitations of momentumbased updating methods. Additionally, we replace the static
temperature coefficient in contrastive loss with a trainable temperature coefficient, enabling the model to automatically balance
sensitivity between easy and hard samples. The code is available
at https://github.com/bmfarer/multi-queue.git.
Index Terms—unsupervised re-ID tasks; hierarchical clustering algorithm;updating feature vectors; multi queue features
updating algorithm;

I. I NTRODUCTION
The unsupervised person re-identification (re-ID) task is
particularly challenging due to the need to distinguish finegrained visual differences between individuals [1] and manage long-tailed data distributions [2], where certain identities
dominate the dataset. These challenges hinder the direct application of unsupervised classification algorithms, requiring
the development of specialized techniques to learn effective
representations without labels .
Current approaches for training on unlabeled data can be
broadly categorized into two main strategies: unsupervised
cross-domain adaptation (UDA) and unsupervised learning
(USL). UDA transfers knowledge from a labeled source domain to an unlabeled target domain, leveraging existing labeled
data to enhance performance in the target domain. In contrast,
USL methods aim to extract useful information directly from
the unlabeled data of the target domain, building effective
† : Equal contribution, * : Corresponding author
This work was supported by the Natural Science Foundation (NSF) of
China (No. 62476206, 62293483, 62101398), the National Key R&D Program
of China (2022YFB2902900)and the project of Pazhou Lab (Huangpu) (No.
2022K0904).

models without relying on external supervision or labeled
examples.
General UDA methods can be classified into generative style
transfer algorithms [3] and pseudo-label generation algorithms.
The latter includes mutual learning algorithms, which use
siamese networks to construct teacher-student models [4], [5],
K-nearest neighbor sampling algorithms [6], [7], and clusterbased algorithms [8]–[10].
Compared to UDA, USL is more challenging to train directly on unlabeled data due to the lack of pretraining support
from source domain data. MMCL [7] generates pseudo labels
through clustering and stores historical features in a memory
bank for learning. PIL [11] also generates pseudo-labels, but
uses linear interpolation to place these labels at the centers of
two clusters, guiding the learning process.
Cluster-based contrastive learning algorithms are widely
employed in both UDA [9], [12], [13] and USL [9], [14]. These
methods typically utilize InfoNCE [15] as the foundational
loss function and incorporate a memory bank [16] to store
negative sample features. This memory bank is often used to
refine pseudo labels in various works.
Therefore, this paper makes the following improvements to
memory bank and momentum update (Fig.1):
• Multi-queue is applied to store and manage historical
features.
• The features updating method of momentum updating is
removed and the operations of queue are used to update
features of the memory bank;
• Replace the static temperature coefficient InfoNCE [15]
with trainable temperature coefficient, so that the algorithm can automatically balance the attention of difficult
samples in the training process.
II. R ELATE W ORK
Self-supervised learning. Most self-supervised learning
methods primarily use contrastive loss [15], [19] as the core
loss function. These include siamese networks for constructing dissimilarities, step-by-step model representation learning
during asymmetric training [20], [21], clustering algorithms

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

Features
Enqueue

Cluster
Algorithm

F(θ)

Encoder
Unlabeled
Dataset

Pseudo Labels
and features

Features

Multi Queue
Memory Bank

Stage 1：Cluster Features

Pseudo Labels
Enqueue

Resample

Contrastive
Loss

F(θ)

Sample
features

Backward
Encoder
Pseudo Labels
Dataset

Features

Backward

Stage 2：Train model with pseudo labels

Fig. 1. Architecture of the multi-queue algorithm. The proposed algorithm is divided into two stages. In the first stage, pseudo labels are generated by
clustering algorithm, which are placed in the memory bank and the corresponding pseudo-labels are affixed to the image. In the second stage, based on the
pseudo labels id, the classification sampling is performed and a feature is taken from each id queue in the memory bank to form a negative sample feature
matrix K, which is computed and compared with the feature Q predicted by the current round and backpropagated. After that, Q is put into the corresponding
id queue according to the pseudo labels.

for generating pseudo labels [22], memory banks for storing
negative sample features [16], and momentum updates for
refreshing frozen encoders with large negative sample dictionaries to aid encoder learning [23]. However, due to the specific characteristics of person re-ID datasets, self-supervised
learning methods tend to experience model collapse [26], [36]
when directly applied to unsupervised re-ID. Therefore, these
methods are not directly applicable to the tasks addressed in
this paper.
Unsupervised re-ID. BUC [37] introduces a bottom-up
hierarchical clustering approach that progressively merges
smaller clusters into larger, more coherent ones, thereby
enhancing identity representation through improved cluster
quality. CycAs [38] and TSSL [39] improve pseudo label
accuracy by integrating continuous tracking predictions over
time, which enhances the consistency and reliability of identity
assignment. CCE [6] employs a soft similarity mechanism
to learn robust image representations across different camera views, facilitating better cross-camera identification and
minimizing discrepancies between camera-specific features.
Cluster-based contrastive algorithms [9], [12]–[14] modify
the InfoNCE [15] loss function by incorporating a memory
bank to store negative samples, updating these samples using
a momentum-based approach, and applying clustering algorithms to generate accurate pseudo labels.
III. A NALYSIS AND M ETHODS
A. Preliminary
Suppose that for a single sample, the predicted vector is
q and the sequence of negative sample matrices drawn from
the memory bank is K = {k1 , k2 , ..., kc }, in which c is the
number of pseudo label classes generated by the clustering.
Denote k+ as the positive feature vector corresponding to q
taken from the memory bank, then InfoNCE is as in Eq.1:

features

𝐿𝑚𝑎𝑥

Multi queue memory bank

Fig. 2. Methods for initializing and updating a multi queue. After the features
are obtained, they are separated according to the pseudo labels and enqueue
to cover the old features.

exp(q·k+ /τ )
Lq = −log PK
i=0 exp(q·ki /τ )

(1)

B. Multi Queue
At the beginning of each epoch, the extracted features
are first enqueue according to the pseudo labels. Denote the
maximum length of the queue to Lmax and the current length
of the queue L. If L > Lmax , discard redundant features; Els
enqueue the features. Since the sampling method employed
in this algorithm is the id-resampling method [14], this is to
ensure the basic stability of the queue length so that historical
features can be dequeueed in a timely manner and avoid them
from being resampled, which can reduce the overfitting of the
model.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

C. Trainable Temperature InfoNCE
Based on InfoNCE [15], since the multi queues method for
constructing memory bank is available in this algorithm. If
only update the parameters of the model with features from
the memory bank, the model will be prone to overfitting,
especially on low difficulty datasets. Therefore, in this paper,
we propose to use variable temperature to balance the sample
difficulty and suppress model overfitting. Following Eq.1, the
basic skeleton of the contrastive loss function for variable
temperature is shown in Eq.1. But the τ is a variable parameter.
In the experiments of our work, Adam is used to update the
parameters, so the updated formula for the τ is given as Eq.2:
Fig. 3. Ablation of Queue Length Maximum.

λ
m̂t
vˆt + ϵ

(2)

m̂t =

mt
1 − β1t

(3)

vˆt =

vt
1 − β2t

(4)

τ ←τ−√

λ = lα

100 and 400, respectively. Adam is the optimizer with a weight
decay factor of 5e-4. Step learning rate decay is adjusted every
20 epochs with a decay rate of 0.2. The number of re-samples
per ID and the length and number of re-samples for a single
queue are set to 16. The hyperparameters of DBSCAN and
Infomap are the same as those of Cluster Contrast [14].

(5)

B. Comparison with Other USL Methods.
l is the total learning rate and α is a multiple of the learning
rate, which is used to automatically adjust learning rates of
τ . The reason why λ = l is not set directly is that, if λ =
l, then the gradient update of the model to the temperature
will generate redundant label noise due to the excessive speed,
which will affect the optimization of the model. Meanwhile,
making λ ̸= l properly keeps the learning rate of model and τ
out of sync so that the model can find more possible solutions.

The results of the horizontal comparison are given in Tab.II.
From the experimental results, Market1501 [33],
DukeMTMC-reID [35] and PersonX [36] have achieved
better results than other algorithms, and the mAP is 0.2%,
0.4% and 1.4% higher than the highest algorithm in the
corresponding dataset, respectively. On MSMT17 [34],
however, the effect is slightly lower than that of the highest
algorithm, named Cluster Contrast [14], at -0.5%.

IV. E XPERIMENT
C. Ablation Experiments
TABLE I
S TATISTICS OF DATASETS USED IN THE EXPERIMENTAL SECTION .
Dataset Object
MSMT17
PersonX
Market-1501
DukeMTMC-reID

train IDs
1,041
410
751
702

train images
62,621
9,840
12,936
16,522

test IDs
3,060
856
750
702

query images
11,659
5,136
3,368
2,228

total images
126,441
45,792
32,668
36,441

cameras
15
6
6
8

To verify the robustness of the proposed algorithm, we
evaluate our mothod on Market1501 [33], MSMT17 [34],
DukeMTMC [35] and Personx [36], whose details are given
in Tab.I. The test setup and environment are two GTX1080s
with CUDA 11.3.
A. Implementation Details.
In this paper, Resnet50 [37] pre-trained by Imagenet [38] is
applied as the experimental model, modules from first to four
adopted, the Average Pooling layer between each two adjacent
modules removed, and the GAP (Global Average Pooling) [14]
adopted at the tail of the model. DBSCAN [32] and Infomap
[26] are used to test the effect of the clustering algorithm.
The image input size for this experiment is 256 × 128. The
employed data augmentation are random horizontal flipping,
10-pixel padding, and random erasing [39].The batch size,
learning rate, epochs and iterations are set to 128, 1.75e-4,

The baseline implementation details as shown in Sec.IV-A.
While different from the Sec.IV-B, the test setup and environment are four Titan-Xps with CUDA 10.2.
1) Queue Length: The MSMT17 dataset was utilized for
the ablation experiment conducted on the cohort. Initially,
the maximum queue lengths were established at 4, 8, 12,
16, 20, 24, and 28, respectively. Analysis of Figure 3 reveals that within the range of 4 ≤ Lmax ≤ 20, the mAP
values consistently increase; conversely, within the range of
20 ≤ Lmax ≤ 28, the mAP values initially rise before
subsequently declining. It is noted that 16 corresponds to the
resampling frequency, indicating that each time pedestrian data
associated with ID B is resampled, the 16 feature vectors of
B within the multi-queue are updated. Consequently, when
Lmax ≤ 16, the queue exclusively contains features derived
from the most recent resampling (new features), whereas when
Lmax > 16, the queue incorporates features from the previous
two resampling events (old features). The data suggest that
retaining a portion of old features within the queue enhances
the model’s representational capacity.
2) Wether apply trainable temperature: Initially, we performed ablation experiments on the datasets Market1501,
DukeMTMC-reID, and MSMT17, both with and without a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 190 (2025) 107617

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Multi-feature balanced network for clothes-changing person
re-identificationI
Mengqing Mei a,b , Chun Ye a , Zhiwei Ye a,b
Liye Mei a,b , Ziyuan Chen a
a

,∗, Fangyi Liu c , Mang Ye c , LingYu Yan a,b

,

School of Computer Science, Hubei University of Technology, Wuhan, 430068, China

b Hubei Key Laboratory of Green Intelligent Computing Power Network, Hubei University of Technology, Wuhan, 430068, China
c School of Computer Science, Wuhan University, Wuhan 430072, China

ARTICLE

INFO

Keywords:
Deep learning
Machine learning
Robustness
Cloth-changing person re-identification
Fine-grained

ABSTRACT
Clothes-changing person re-identification (CC-ReID) is pivotal in long-term scenes, especially when involving
significant variations in clothes. The principal challenge of this task lies in the extraction of clothes-irrelevant
features. Currently, most methods alleviate the interference caused by clothing changes through separating
and identifying body regions. However, these approaches could not fully utilize all the useful information in
pedestrian images. In this work, a novel Multi-features Balanced Network (MBNet) is proposed for improving
the robustness of the CC-ReID model by exploiting clothing-unrelated features, consisting of a global branch, a
clothing-unrelated branch, and a mask branch. Specifically, to highlight clothing-unrelated clues, a knowledge
transfer module (KTM) is first designed. Then, the clothing-unrelated branch only receives images that are
unaffected or less affected clothing to resist clothes-changing. Besides, a feature attention module (FAM) is
introduced in this branch, which can suppress background clutter and extract discriminative fine-grained
features. Finally, a cross fusion module (CFM) is used to integrate more contextual information and mine
more clothing-independent and pose features in the mask branch. Three branches are combined to perform
CC-ReID. Extensive experiments on three popular synthetic and realistic datasets show that the superiority of
the proposed approach, achieving a Rank-1/mAP accuracy of 44.6%/22.7%, 58.3%/57.9%, 87.2%/84.0%.

1. Introduction
Person re-identification (ReID) (Ye et al., 2022) is an image retrieval
task aimed at matching target pedestrians across different viewpoints
and multiple cameras to find the movement trajectory of pedestrians,
which has been broadly applied in real-life scenarios, including suspect tracking, searching missing persons, unmanned supermarkets, and
autonomous driving. In recent years, the rise of deep learning has
propelled the rapid development of ReID methods, making it a research
hotspot. Most previous ReID methods mainly match target pedestrians
in a short-term scenario. However, these methods overlook an important issue: the possibility of pedestrians changing their clothing. In a
realistic long-term scenario, pedestrian often changes their clothes at
different times, making the clothing appearance information tends to
be unreliable. Compared to short-term person ReID, clothes-changing
person ReID (CC-ReID) is suitable for more complex scenarios and is

more flexible. Therefore, in the paper, a more realistic and practical
clothes-changing person ReID task is concerned.
Most existing ReID methods (Yang, Liu et al., 2023; Zhong et al.,
2024) have primarily addressed conventional challenges. These challenges include overcoming occlusions, visible infrared scenarios (Qian,
Lin, & Du, 2025), camera view variations, and pose changes. However,
when confronted with scenarios involving clothing changes, traditional
ReID methods (Peng, Tang, Lin, & Zheng, 2024; Si, He, Li, & Gao,
2023; Ye et al., 2022) heavily rely on clothing-related information for
identification, which leads to a sharp decline in model performance
under the circumstance of clothes-changing. In addition, such information may mislead the model when people wear different clothes
or when different people wear similar clothes. Therefore, the clothing
appearance information is invalidated in this case. The main challenge
of cloth-changing ReID is to overcome the variation of clothes and
extract discriminative clothing-irrelevant features.

I This research was supported by National Natural Science Foundation of China (Grant Nos. 62376089, U23A20318, 62302154, 62472149), and Young and
Middle-aged Scientific and Technological Innovation Team Plan in Higher Education Institutions in Hubei Province, China (Grant No. T2023007).
∗ Corresponding author.
E-mail address: hgcsyzw@hbut.edu.cn (Z. Ye).

https://doi.org/10.1016/j.neunet.2025.107617
Received 23 November 2024; Received in revised form 13 February 2025; Accepted 11 May 2025
Available online 29 May 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 190 (2025) 107617

M. Mei et al.

To handle with the problems above, researchers have gradually
shifted their focus to cloth-changing ReID (Hong, Wu, Wu, Han, &
Zheng, 2021; Qian et al., 2020; Yang, Wu, & Zheng, 2021). The current
CC-ReID methods mainly pursue learning discriminative cloth-invariant
feature representations and could be roughly divided into two categories. The former (Liu, Kim, Gu, Jain, & Liu, 2023; Zhang, Liu, Chen,
Zheng, & Zhang, 2023) is based on additional auxiliary knowledge to
dig out cloth-irrelevant features, such as keypoints, 3D shapes, semantic
masks, and human parsing techniques to resist the interference caused
by clothing changes. For example, DeSKPro (Wu, Liu, Shi, Tang, &
Guo, 2022) used human parsing knowledge to eliminate the distraction
of clothing appearance, and mined identity-sensitive cues from the
recovered facial details. However, these facial features are not clear
enough due to the occlusion of masks or hats in surveillance videos.
Hence, it becomes impractical in real-world applications. The latter (Gu
et al., 2022; Hong et al., 2021; Xu et al., 2021) designs disentanglement strategies to separate clothing-related and clothing-unrelated
features from pedestrian images, in order to mitigate the impact of
clothing changes. It includes using Generative Adversarial Networks
(GAN) (Wang, Zhang, Lu, Fang and Wang, 2022; Yu et al., 2022) or randomly changing the pixels of clothing regions (Shu, Li, Wang, Ruan, &
Tian, 2021) to force the model to learn cloth-irrelevant features which
can reduce the differences in clothing color and texture information in
the image. For example, DG-Net (Zheng et al., 2019) proposed a joint
learning framework that utilizes discriminative and generative modules
separately, learning the appearance and structure codes to generate
high-quality cross-id composed images. However, ensuring the quality
of the generated images is challenging, and there is no guarantee that
the generated images will not damage cloth-irrelevant features.
In person re-identification, particularly in clothes-changing scenarios, relying on only one type of feature often yields poor results.
This is due to an over-reliance on a specific feature, which makes
the model susceptible to environmental changes such as lighting, occlusion, and perspective. For example, PCB (Part-based Convolutional
Baseline) (Sun, Zheng, Yang, Tian, & Wang, 2018) relied on features of
color and texture that showed significant performance degradation in
clothes-changing scenarios due to its sensitivity to clothing variations.
While appearance features provide basic identification information,
they may fail under varying lighting conditions or partial occlusion.
Similarly, the effectiveness of the posture feature is also limited when
the pose of the individual changes. These issues will indicate that
single-feature methods could result in a decrease in model identification accuracy when faced with complex factors such as perspective,
occlusion, and illumination. Zhang et al. (2023) pointed out that utilizing multiple biological features, including the head, shoulders, and
neck, can significantly alleviate the issue of clothing variation. Jin
et al. (2022) introduced gait recognition as auxiliary information by
leveraging person unique gait characteristics to drive the ReID model
to learn cloth-agnostic representations. Liu, Liu et al. (2024) proposed
using pose-guided attention learning to study and extract invariant
features of pedestrians under different conditions. Therefore, it is particularly important to explore how to effectively combine multiple
features to enhance robustness and accuracy of the model. In this
paper, a novel multi-feature balanced network (MBNet) is proposed
that combines multiple features such as head, leg, appearance, posture,
and background to enhance the performance of the clothes-changing
ReID model.
The proposed MBNet utilizes different source images as input, allowing it to learn invariant features under variations of the conditions.
By leveraging the complementary nature of multiple features and balancing them, MBNet can dynamically learn and adapt to different
condition changes. By merging the multimodal features from the input
image, more comprehensive information can be obtained in the same
scenario compared to methods that rely on a single feature. For example, head features can effectively distinguish individuals, while dynamic information about movements provided in pose features can play

an important role when static appearance features fail. In MBNet, different features are adjusted to ensure that the most favorable features
could be selected for identification in complex environments, independent of the failure of individual features. It includes a global branch, a
cloth-unrelated branch, and a mask branch, as shown in Fig. 1. Specifically, MBNet applies a knowledge transfer module in the global branch
to transfer the knowledge learned from the clothing-unrelated branch
to the global branch, which aims to emphasize clothing-unrelated areas.
To further extract fine-grained clothes-irrelevant features, the clothingunrelated branch uses cropped RGB images as input to focus on identity
information. Besides, MBNet adds a feature attention mechanism in
this branch to reduce the local noise and enhance the robustness of
clothes-changing problems. In addition, a background constraint loss
is embedded to suppress the learned background features, focusing on
the foreground features of pedestrians. To resist clothing changes, in the
mask branch, MBNet takes the silhouette mask as input to effectively
learn their body features. However, it overlooks important appearance
clues. To integrate more contextual information, MBNet introduces a
cross fusion module.
In summary, the main contributions of this paper could be outlined
as follows:
• A Multi-feature Balanced Network (MBNet) is proposed to learn
multiple clothing-irrelevant features to overcome the interference caused by cloth-changing. The core idea is to improve the
robustness of the ReID model by balancing multiple identity
features.
• To highlight clothing-unrelated areas, a knowledge transfer module (KTM) is proposed for resisting clothes-changing problems.
• A feature attention module (FAM) is proposed that reduces the
local noise and extracts more purely clothing independent features. Meanwhile, a background constraint loss is embedded to
strengthen identity-related features and resist the interference of
irrelevant features.
• A cross fusion module (CFM) is proposed that utilizes a twoway cross attention mechanism to integrate mask features and
clothing-irrelevant features of pedestrians to cope with situations
where different people wear similar clothing.
• More extensive evaluations of our method are conducted across
cloth-changing scenarios. The experimental results demonstrate
the effectiveness of our proposed framework MBNet performs well
on cloth-changing ReID datasets.
The rest of this paper is organized as follows. Section 2 surveys
recent work related to general person ReID and cloth-changing person
ReID. Section 3 describes the framework of the proposed method in
detail. Section 4 presents experimental results and analysis to show the
superiority of our method. Section 5 concludes the paper.
2. Related work
2.1. General person re-identification
Benefiting from the development of convolutional neural networks
in recent years, ReID methods have achieved considerable success
and widely explored. Existing person ReID mainly addressed some
traditional problems: overcoming occlusions (Li et al., 2021; Wang
et al., 2020), visible infrared scenarios (Ye, Shen, & Shao, 2021; Zhu
et al., 2024), camera view variations (Peng et al., 2024; Su, Zhang,
Xing, Gao, & Tian, 2016), and pose changes (Nguyen, Mantini, &
Shah, 2024). Gao, Wang, Lu, and Liu (2020) used pose-guided to
overcome occlusion and detected the non-occluded body parts and
then matched the correspondents separately, which aimed to exploit
more discriminative local features. Wang et al. (2020) viewed the
learned semantic features of key points in images as nodes of the
graph, and used graph convolutional layers to pass relation information
2

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 1. The framework of our proposed Multi-feature Balance Network (MBNet). It consists of three branches:the global branch, the clothing-unrelated branch and the mask branch.
The global branch uses the Knowledge Transfer Module (KTM) which aims to transfer the clothes information from clothing-unrelated branch to this branch. The clothing-unrelated
branch uses the Feature Attention Module (FAM) to further enhance its ability to learn clothing-irrelevant fine-grained features. The mask branch uses the Cross Fusion Module
(CFM) to obtain the fusion feature 𝐹𝑎𝑝 .

between nodes, learning high-order information and topological information for discriminative features and robust alignment. Zhong et al.
(2022) utilized intermediate grayscale images as auxiliary information
to colorize the single-modality infrared images and eliminate crossmodality variations. Su et al. (2016) proposed a three-stage attribute
enhancement ReID method. First, they used a dataset labeled with
attributes for training, then fine-tuned the network using attributes and
identity labels, and finally fused the predicted attributes.
Although these methods have made significant progress, they heavily rely on clothing appearance information, which would become
ineffective or even misleading when faced with changing scenes.

neighbor structure in a two-branches network. Wu et al. (2022) proposed an effective identity-sensitive knowledge propagation framework
to mitigate the resolution degradation issues and learn cloth-irrelevant
features.
The second type is to separate clothing-related and clothingunrelated features from pedestrian images, encouraging the model
to learn clothes-invariant features. For example, Shu et al. (2021)
proposed a method of randomly changing clothing pixels. They used
a pre-trained human parsing model to obtain body parts and randomly
exchanged the pixels of clothes or pants among different pedestrians,
which makes the model less dependent on clothing colors and textures.
Yu et al. (2022) employed GAN to generate multiple images of the same
identity wearing different clothes, simulating a cloth-changing scene.
Hong et al. (2021) proposed a two-stream mutual learning framework
that utilizes shape and appearance features to explore body shape
cues of clothing changes. It adopts a dense interactive mutual learning
strategy, encouraging the appearance stream to learn fine-grained discriminative body shape knowledge from the shape stream. Cui, Zhou,
Peng, Zhang, and Wang (2023) proposed a novel disentanglement
framework that utilizes human component region reconstruction and
a deep assembled module to separate clothing-relevant and clothingirrelevant features. Gu et al. (2022) mined clothing-irrelevant features
from the original RGB images. They collected clothes labels and used
clothes classifiers to learn clothing-relevant features, which forced
the model to learn clothes-irrelevant features through the clothes
adversarial loss function. However, collecting clothes labels is very
time-consuming and not feasible in real life. Zhao et al. (2024) utilized
instance-level features from each identity sample to synthesize novel
and diverse clothing styles to train a robust model. Simultaneously,
enhancing samples through style to construct informative and robust
prototypes.
To sum up, the former methods introduce additional auxiliary information to overcome the clothes-changing, but this often incurs high
computational costs. The latter methods may destroy the most discriminative information from the original RGB image. And it is unclear

2.2. Cloth-changing person re-identification
In recent years, a lot of work (Bansal, Foresti, & Martinel, 2022;
Li, Xu, Wei, & Yang, 2022; Yao, Kusakunniran, Wu, Xu, & Zhang,
2022) has emerged for clothes-changing person ReID. The core idea
of the work is to learn clothes-irrelevant features. The first type of
these methods are to introduce clothing-unrelated auxiliary information for feature mining, including keypoints (Qian et al., 2020; Zhang
et al., 2023), 3D shapes (Chen et al., 2021; Liu et al., 2023), semantic
masks (Li, Xu, Li, & Guo, 2023), and human parsing techniques (Wu
et al., 2022). Zhang et al. (2023) proposed a novel multi-biological
unified network that utilizes multi-biometric features to mitigate the
influence of clothing. It used a key points estimation model and a
graph convolution layer to capture more discriminative pose features,
aiming to enhance the robustness of the cloth-changing ReID model.
Qian et al. (2020) leveraged the human keypoints to encode shape
embedding, distilling the identity-relevant shape feature, which aims to
focus on the body shape information. Liu et al. (2023) used 3D shapes
to reconstruct 3D clothed body shapes, disentangling identity from
non-identity components. It learns discriminative features of naked
body shapes in a joint manner. Li et al. (2023) utilized unsupervised
contrastive learning framework that combines RGB and semantic mask
features to learn cross-clothes invariance from a hierarchically semantic
3


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113676

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Multi-granularity collaborative constraint feature alignment network for
unsupervised person re-identification
Yanbing Chen a ,1 , Lingyi Guo b ,∗,1 , Zhixin Tie a , Yinghong Xu b , Hao Sheng c
a School of Computer Science and Technology, Zhejiang Sci-Tech University, Hangzhou, 310018, Zhejiang, China
b
c

College of Science, Zhejiang Sci-Tech University, Hangzhou, 310018, Zhejiang, China
State Key Laboratory of Virtual Reality Technology and Systems, School of Computer Science and Engineering, Beihang University, 100191, Beijing, China

ARTICLE

INFO

Keywords:
Person re-identification
Unsupervised learning
Contrastive learning
Fine-grained feature learning

ABSTRACT
Unsupervised Person Re-identification (UReID) aims to match pedestrians across non-overlapping cameras
without annotations. Existing methods generally focus on mining multi-granularity feature complementarity,
yet lack an explicit cross-granularity alignment mechanism, thus struggling to ensure stable semantic consistency across granularities during training. To address this issue, we propose a Multi-Granularity Collaborative
Constraint Feature Alignment Network (MCCAN) for unsupervised person re-identification. As a multi-output,
coarse-to-fine hierarchical alignment framework, MCCAN enables explicit alignment of multi-granularity
features at all learning stages. Specifically, we design a multi-granularity constraint loss function to align local
and global features explicitly, thus preserving their semantic consistency throughout training. Furthermore, we
introduce a Center-Constrained Filtering (CCF) module, which leverages global feature centroids to constrain
local features, achieving multi-granularity feature alignment within the memory bank. Extensive experiments
on four public UReID datasets fully validate the effectiveness of the proposed method.

1. Introduction
Person re-identification (ReID) strives to match specific individuals
across non-overlapping cameras. This technology has emerged as a
prominent research hotspot in recent years due to its critical application
value in numerous real-world scenarios, such as intelligent surveillance, public security, and pedestrian behavior analysis. Although traditional supervised ReID methods have achieved significant performance
breakthroughs, they rely heavily on manual annotations of large-scale
cross-camera data. This reliance not only results in prohibitively high
annotation costs and extensive time consumption but also ultimately
limits their large-scale deployment in real-world scenarios. To address
this core bottleneck, unsupervised person re-identification (UReID) has
emerged as a highly promising alternative [1–4]. By eliminating the
need for manual labeling, it demonstrates superior scalability and
generalization capabilities.
Existing UReID approaches can be broadly categorized into unsupervised domain adaptation (UDA) ReID and fully unsupervised learning
(USL) ReID. UDA ReID methods learn transferable knowledge from a
labeled source domain and adapt it to an unlabeled target domain [5,6].
In contrast, USL ReID methods directly learn from the unlabeled target
dataset [7–9], making them more suitable for real-world applications.

Most pioneering USL ReID methods currently adopt memory bankbased image features for contrastive learning [10–13]. These methods
typically follow a two-stage training strategy: first, image features are
clustered using algorithms such as DBSCAN [14], and pseudo-labels
are assigned; second, the model is trained based on these pseudolabels. These methods generally rely on global features to complete
label assignment and feature learning. Although they have achieved
excellent performance, due to the lack of targeted learning of local
detailed features, they are prone to feature bias during the training
process, ultimately resulting in performance bottlenecks [15,16].
Inspired by fine-grained feature learning methods in the supervised
person re-identification field [17–20], state-of-the-art methods for USL
ReID in recent years have focused on guiding models to learn finegrained local features to mitigate the inherent biases of global features.
These methods focus on exploring the complementary information between global and local features. For instance, PPLR [21] leverages the
complementarity between global and local features to suppress label
noise. LP [15] supplements the discriminative details easily overlooked
by global features by adding a dedicated local feature learning branch.
GL-NCPLR [22] constructs an interactive complementary mechanism

∗ Corresponding author.
1

E-mail address: 2023220105009@mails.zstu.edu.cn (L. Guo).
Contributed equally to this work.

https://doi.org/10.1016/j.patcog.2026.113676
Received 3 February 2026; Received in revised form 13 March 2026; Accepted 2 April 2026
Available online 12 April 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

2. Related work

between global and local soft labels. CMLR [23] synergistically enhances pseudo-label reliability through a complementary optimization
strategy of ’global cross-camera + local intra-camera’.
Although the above-mentioned methods have achieved significant
performance improvements, they still mainly rely on an implicit learning paradigm to maintain semantic consistency among multi-granularity
features, without adopting explicit alignment strategies. There exists an essential difference in optimization objectives between implicit learning and explicit alignment of semantic consistency: Implicit
learning centers on mining complementary information among multigranularity features, aiming to enhance feature discriminability, where
the preservation of semantic consistency is mostly a byproduct of the
discriminative learning process. In contrast, explicit alignment takes
the preservation of cross-granularity semantic consistency as one of the
core optimization objectives, on an equal footing with discriminative
learning. Nevertheless, for unsupervised learning, such an implicit
learning mechanism inherently suffers from the following drawbacks:
(1) Unsupervised models inherently suffer from the problem of pseudolabel noise. When pseudo-label noise coexists with noise or redundancy
in local features, the model will learn a large amount of spurious
complementary information, and the semantic consistency learning
based on this will undergo severe deviation, thus exacerbating the
semantic misalignment between features of different granularities.
(2) Fine-grained features themselves have inherent limitations. Their
semantics are not only prone to misalignment with global features but
also suffer from semantic mismatches between different fine-grained
features. Relying solely on the model’s implicit learning capability is far
from sufficient to constrain complex fine-grained semantic information.
To fully address the aforementioned issues, we propose a MultiGranularity Collaborative Constraint Feature Alignment Network (MCCAN). The network adopts a coarse-to-fine multi-granularity learning
architecture that effectively captures hierarchical local complementary
details. Based on this architecture, we design a novel multi-granularity
constraint loss function. Unlike existing methods such as PPLR [21],
LP [15], GL-NCPLR [22], CMLR [23], and PFCD [16], which rely solely
on the model to learn semantic consistency implicitly, our proposed loss
explicitly aligns semantic information across features of different granularities during training, thereby enabling the model to preserve stable
semantic consistency throughout the learning process. Furthermore, we
introduce a Center-Constrained Filtering (CCF) module. Distinct from
traditional approaches that directly construct a multi-granularity memory bank [15,16,22], the CCF module utilizes the centroids of global
features to constrain and explicitly align the semantics of fine-grained
features prior to memory bank initialization. By filtering out semantically inconsistent local features, this module ultimately achieves a more
robust initialization of the memory bank. Our main contributions are
as follows:

2.1. Unsupervised person ReID
Contemporary unsupervised person re-identification methods are
primarily categorized into two paradigms: unsupervised domain adaptation ReID [28,29] and fully unsupervised learning ReID [30,31].
The core idea of UDA ReID is to transfer knowledge acquired from
source domain data to the target domain [32], with its fundamental
premise that the distribution discrepancy between the source and target
domains is small. If this premise is not satisfied, the model is highly
prone to overfitting. In contrast, USL ReID methods perform model
training directly on the unlabeled target domain without requiring
any annotation information, thus exhibiting stronger data adaptability
and scalability [1]. Since USL ReID aligns better with our research
objectives, we focus on this setting.
State-of-the-art USL person re-identification methods typically consist of three core stages: pseudo-label generation, memory bank initialization, and neural network training. Existing studies mostly focus on optimizing these three stages to improve model performance.
In particular, BUC [33] designed a bottom-up clustering strategy,
which achieved effective extraction of highly discriminative features
by exploring cross-identity differences and intra-identity similarities of
pedestrians. ICE [31] minimized intra-class variance by mining hard
negative samples and using pairwise similarity scores as soft pseudolabels. ISE [7] adopted a progressive linear interpolation strategy to
address the data inconsistency problem during memory bank updates.
DCCT [34] constructed a dual-network architecture to generate pseudolabels, thereby supporting the co-training of models. DKD-MPL [35]
proposed a dual knowledge distillation framework, which could effectively reduce pseudo-label noise. Notably, the key improvements
proposed in our paper cover all the three aforementioned core stages.
Relevant technical details and research backgrounds will be elaborated
in subsequent sections.
2.2. Fine-grained contrastive learning
Traditional USL ReID methods mostly rely on global features for
feature learning, yet they often neglect fine-grained local detailed information. This drawback is prone to inducing viewpoint bias, thereby
preventing the model from achieving optimal performance. To address this issue, the focus of relevant research has gradually shifted
toward contrastive learning paradigms based on fine-grained features
in recent years. Specifically, PPLR [21] introduced a cross agreement
score that purified the pseudo-labels of global features by integrating
the predictions of local features, thereby collectively alleviating the
noise problem in global feature clustering. LP [15] fused local and
global feature perspectives and leveraged the guidance mechanism of a
teacher–student model to further reduce noise interference. PFCD [16]
combined global–local feature fusion, confidence-driven denoising, and
modular knowledge distillation strategies to effectively enhance the
model’s feature representation capability and anti-noise robustness.
CMLR [23] integrated camera awareness and multi-level label refinement mechanisms while learning fine-grained features, thus further
boosting the contrastive learning effect and feature discriminability.
GL-NCPLR [22] relied on label propagation and global–local feature
interaction optimization strategies to achieve dual improvement in
pseudo-label quality and feature discrimination ability. DSFNet [36]
fused the local and global features of individuals through a weighted
assignment and integration strategy, ensuring the integrity of instance
features.
Nevertheless, most existing fine-grained feature-based USL ReID
methods rely primarily on a part-based learning paradigm. This design
often causes the model to overfit to fixed local detail perspectives,
thereby limiting its generalization ability. In contrast, the hierarchical fine-grained features extracted by MCCAN provide a more robust

• We design a novel multi-output, coarse-to-fine multi-granularity
feature learning framework. Particularly importantly, we propose
a corresponding multi-granularity constraint loss function that
explicitly aligns semantics between fine-grained and global features. This design effectively compensates for the limitations of
traditional fine-grained learning methods, enabling the model to
learn more comprehensive and reliable fine-grained features.
• We propose a CCF module, which leverages the constraint relationship between global feature centroids and local features to
filter out unreliable local features, effectively aligning the multigranularity features in the memory bank and thereby significantly
enhancing the robustness of the memory bank.
• Extensive experiments on the Market-1501 [24], MSMT17 [25],
PersonX [26] and Veri776 [27] datasets demonstrate that the
proposed method outperforms various existing state-of-the-art
approaches on the UReID task, demonstrating strong cross-dataset
adaptability.
2

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 1. Overview of the MCCAN framework. The upper half illustrates the main framework, which encompasses feature extraction, clustering, and the two loss
functions designed for this framework. The lower half presents the two core components: the multi-granularity memory bank and the CCF module.

representation of pedestrian details. In the field of cross-modal ReID,
CM2 GT [37] further enhanced the robust representation of pedestrian
details via a global-patch-pixel hierarchical fine-grained feature design,
and verified the effectiveness of such a design as well. Furthermore,
existing USL ReID methods mostly focus on feature fusion and the
complementary learning of multi-granularity information but generally
lack explicit alignment mechanisms. This limitation prevents the model
from maintaining stable semantic consistency across features of varying
granularities. To address these issues, MCCAN effectively resolves the
aforementioned problems by introducing a constraint loss and the CCF
module.

To be specific, the 𝐿 − 𝑡ℎ layer of the model is duplicated to construct
a dual-branch output architecture, where each branch extracts finegrained features at distinct hierarchical levels, thereby enhancing the
representational capacity of fine-grained features.
Fig. 1 illustrates our framework and its individual modules. The MCCAN outputs two types of fine-grained features at different hierarchical
levels, which are generated by Branch 1 and Branch 2, respectively.
In particular, Branch 1 calculates the dot product between each patch
token and the class token, and ranks them to obtain a Rank-list. We
select the top K ranks to form the Top-K list 𝐾 :
(
)
𝑑𝑜𝑡𝑖 = 𝑥𝑖 ⊤ 𝑐𝑙𝑠1 ,  = rank {𝑑𝑜𝑡𝑖 }𝑛𝑖=1 ,  = 𝜉(),
(1)

3. Methodology

𝐾 = {𝑑𝑜𝑡𝑖 ∈  ∣ 𝑖 ≤ 𝐾}, 𝐾 = {𝑥𝑖 ∈  ∣ 𝑖 ≤ 𝐾},

3.1. Multi-granularity collaborative constraint feature alignment network
framework

(2)

where 𝑥𝑖 is the 𝑖th patch token output from Branch 1, with a total of
𝑛 patch tokens. 𝑐𝑙𝑠1 is the class token of Branch 1, 𝑟𝑎𝑛𝑘(⋅) represents
the sorting operation in descending order, and 𝜉(⋅) is the mapping from
the dot product list to the feature list. 𝑑𝑜𝑡𝑖 denotes the dot product
result. We apply a softmax function to the features in the obtained TopK list and compute a weighted sum to encourage the model’s contextual
awareness of local details. The final fine-grained features based on
patch tokens are obtained as:

The unlabeled pedestrian dataset is denoted as 𝐷 = {𝑥𝑖 ∣ 𝑖 =
1, 2, … , 𝑁}, where 𝑥𝑖 represents the 𝑖th image and 𝑁 denotes the total
number of samples in the dataset. The goal of USL ReID is to train a
deep neural network  = 𝑓 (𝜃; 𝑥) that maximizes the feature distance
between different individuals and minimizes the feature discrepancy
between the same individual.  = 𝑓 (𝜃; 𝑥) maps an image to a specific
embedding feature 𝒇 ∈ R𝑑 . To mine more refined discriminative
features, existing methods such as PPLR [21] and GL-NCPLR [22]
typically perform horizontal partitioning on feature maps prior to the
network output layer, so as to construct part-based fine-grained feature
representations. However, such a single partitioning strategy still suffers from the limitation of insufficient representation granularity, and
is prone to induce the model to overfit to the fixed perspective of part
division. In contrast, to achieve multi-granularity feature extraction
across hierarchical levels ranging from global to local, we employ a
Transformer network with shared weights across the first 𝐿 − 1 layers.

𝑓patch =

∑
𝑥𝑖 ∈𝐾

∑

exp(𝑑𝑜𝑡𝑖 )
𝑑𝑜𝑡𝑗 ∈𝐾 exp(𝑑𝑜𝑡𝑗 )

𝑥𝑖 .

(3)

Inspired by MGN [18], we perform a reshape operation on the 𝑛
D-dimensional patch tokens from Branch 2, reconstructing them into a
feature map 𝑓2 with the size of 𝐻
×𝑊
×D, where 𝑃 is the patch size. We
𝑃
𝑃
then uniformly divide the reconstructed feature map horizontally into
𝑀 parts and apply GeM pooling to obtain the part-based fine-grained
features:
(
(
))
𝑓𝑝𝑎𝑟𝑡 = 𝐶𝑜𝑛𝑐𝑎𝑡(𝐺𝑒𝑀 𝑆𝑝𝑙𝑖𝑡 𝑓2 , 𝑀 ),
(4)
3


codex
前五篇的共同模式已经很清楚：很多论文把“多粒度、多分支、多队列”作为机制卖点，但真正能发的部分通常是把一个训练不稳定或语义错配的问题重新定义清楚。接下来读剩下四篇，重点找能和航拍、姿态、SMPL 或强主干资产相接的缺口。
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
/bin/zsh -lc "pdftotext -l 3 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Journal Pre-proof

Multi-Granularity Dynamic Hierarchical Graphs for Video-Based
Person Re-Identification
Wei Zhao, Bingyi Zhou, Yongquan Wang, Yuanpeng Fan,
Guoyou Wang
PII:
DOI:
Reference:

S0893-6080(26)00699-4
https://doi.org/10.1016/j.neunet.2026.109238
NN 109238

To appear in:

Neural Networks

Received date:
Revised date:
Accepted date:

20 June 2025
25 May 2026
6 June 2026

Please cite this article as: Wei Zhao, Bingyi Zhou, Yongquan Wang, Yuanpeng Fan, Guoyou Wang,
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification, Neural Networks (2025), doi: https://doi.org/10.1016/j.neunet.2026.109238

This is a PDF of an article that has undergone enhancements after acceptance, such as the addition of a cover page and metadata, and formatting for readability. This version will undergo additional copyediting, typesetting and review before it is published in its final form. As such, this version is no longer the Accepted Manuscript, but it is not yet the definitive Version of Record; we are
providing this early version to give early visibility of the article. Please note that Elsevier’s sharing
policy for the Published Journal Article applies to this version, see: https://www.elsevier.com/about/
policies-and-standards/sharing#4-published-journal-article. Please also note that, during the production process, errors may be discovered which could affect the content, and all legal disclaimers that
apply to the journal pertain.
© 2026 Published by Elsevier Ltd.

Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person
Re-Identification
Wei Zhaoa , Bingyi Zhoua , Yongquan Wanga , Yuanpeng Fana , Guoyou Wanga,∗
a School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, 1037 Luoyu

Road, 430074, Wuhan, China

Abstract
Video-based person re-identification (Re-ID) aims to identify video sequences of specific pedestrians captured in a
distributed camera system. The key to tackling this challenging task is to model rich temporal features in video
sequences. However, existing methods usually capture only fixed-length global temporal clues, making it difficult to
eliminate the effect of occlusion and accurately model the motion information of pedestrians. To address this limitation,
we propose a novel graph-based framework, namely dynamic hierarchical graph network(DHGN), to model temporal
features by adaptively capturing multi-granularity temporal clues. Specifically, we adaptively segment the video features
horizontally into several regions and construct graphs for the features at the same level. The features of each frame
at the same level are considered as graph nodes, which are then adaptively connected based on feature similarity.
By dynamically aggregating features from neighboring nodes in the same graph, DHGN is able to adaptively capture
temporal cues from different body parts. Furthermore, in order to obtain more robust matching results, we propose
a similarity weighted inference module (SWIM), which utilizes gallery-gallery similarity to modify the query-gallery
similarity matrix. Extensive experiments on four benchmarks clearly demonstrate the effectiveness of the proposed
method.
Keywords: video-based person re-identification, graph neural network, feature inference
1. Introduction
Person re-identification (Re-ID) aims to match the
same person in different cameras with great potential in
areas such as criminal investigation, forensics, and security. Existing Re-ID methods are mainly divided into two
categories, i.e.,image-based methods [1, 2, 3] and videobased methods [4, 5]. Image-based methods primarily rely
on appearance features, such as the color and texture of
clothing, and do not consider temporal information. In
contrast, video-based methods take tracklets as input for
more robust Re-ID by exploiting richer spatio-temporal
features.
Video-based methods critically focus on how to model
the temporal features of pedestrians. Generally, temporal
clues of different lengths in the video play distinct roles in
identification [6]. Long-term temporal clues can mitigate
the effects of occlusion, while short-term temporal clues
can model detailed motion patterns [7]. Furthermore, because of differing movement patterns and viewpoints, various hierarchical body parts of pedestrians exhibit differences in movement intensity and occlusion. For example,
∗ Corresponding author

Email addresses: wzhao97@hust.edu.cn (Wei Zhao),
zhou_bingyi@hust.edu.cn (Bingyi Zhou),
wangyongquan@hust.edu.cn (Yongquan Wang), ypfan@hust.edu.cn
(Yuanpeng Fan), gywang@mail.hust.edu.cn (Guoyou Wang)
Preprint submitted to Elsevier

as shown in the Figure 1 (a), some body parts (e.g. legs)
varied greatly during the walk, while other body parts (e.g.
torso) remained stable. Concurrently, various body parts
of the individual are intermittently occluded in different
frames, and these occluded parts subsequently reappear
in other frames. Therefore, processing these parts separately and adaptively capturing temporal clues can model
temporal features more efficiently.
However, most of the existing methods [8, 6, 9, 4] can
only model fixed-length global temporal features, thus failing to eliminate the effect of occlusion and accurately extract motion information of body parts. Accordingly, some
methods [7, 10] use uniform horizontal segmentation and
graph neural networks to model fine-grained temporal features. While these methods can adaptively extract temporal clues of different lengths, they cannot dynamically
segment features based on occlusion and body parts. In
addition, these methods have a fixed number of neighboring nodes, which can lead to the introduction of some lowcorrelation features.
Following the above discussion, we propose a novel
graph-based framework for video-based person Re-ID,
named dynamic hierarchical graph network (DHGN). As
shown in Figure 1 (b), DHGN dynamically divides global
features into segments of varying sizes based on visual
content and adaptively models the temporal features of
different body parts. The core component of DHGN is
June 7, 2026

the graph convolutional (GC) layer, which constructs individual graphs for different body parts. Subsequently, the
GC layer adaptively captures temporal clues for each body
part by aggregating features from adjacent nodes within
the same graph.
Specifically, for each frame, the feature map is first
evenly divided into fine-grained horizontal blocks, which
are then clustered based on feature similarity into adaptive
semantic regions of different sizes. These clustered regions
are treated as graph nodes. Graphs are constructed within
the same body-part level, rather than across arbitrary regions, to maintain semantic consistency. Edges are established using a mutual nearest neighbor criterion, so that
only mutually similar nodes are connected. This strategy yields a similarity-adaptive and dynamically updated
graph topology, enabling the GC layer to aggregate reliable
temporal information from semantically corresponding regions across frames.
In addition, due to the differences in pedestrians’ movement patterns and occlusion areas, a fixed number of
division is not appropriate. Thus, we apply a multigranularity segmentation scheme and construct multiple
graphs at each granularity. Finally, we interact information on the graph structure via graph neural networks
(GNNs) to adaptively model the corresponding temporal features. Moreover, most of the approaches use only
query-gallery similarities for matching and ignore gallerygallery relationships. Therefore, we design a similarity
weighted inference module (SWIM) which utilizes the relationship between query-gallery and gallery-gallery to optimize pedestrian matching.
Compared with other methods, DHGN has several advantages: (1) Adaptive capture of temporal clues from
different body parts. DHGN can adaptively aggregate
feature patches of varying sizes based on similarity, thus
avoiding the interference of factors such as occlusion, and
effectively modeling the motion information of pedestrians.
(2) The number of features involved in extracting temporal
information dynamically changes. When some nodes are
excessively different from other nodes, DHGN is able to dynamically adjust the number of neighbours, thus avoiding
information interference. (3) The GC layer can be added
to any convolutional neural network and specific granularity of graphs can be deployed depending on the resolution
of the features. As the depth of the network increases, the
resolution of the features decreases and the semantic information becomes more abstract. Therefore, feature maps
with low resolution are difficult to localize fine-grained targets. Accordingly, DHGN captures finer-grained temporal
clues in high-resolution features and fuses them with the
original features through residual connection. The main
contributions of this paper can be summarized as follows:

Figure 1: The temporal clue capture method. (a) Most methods can
only process Euclidean data, capturing fixed-length global temporal
cues such as long-term (fixed-frame intervals) and short-term (adjacent frames). (b) Our method employs graph neural networks that
can cope with non-Euclidean data and adaptively capture temporal
clues of different lengths for each body part.

mation of each part.
• We propose a similarity weighted inference module
(SWIM) to exploit pedestrian relationships in gallery
for matching.
• We evaluated DHGN and SWIM on four widely used
video re-identification benchmarks. The evaluation
demonstrates that the proposed method significantly
improves the baseline performance and outperforms
other state-of-the-art algorithms.

2. Related work
2.1. Video person Re-ID
Video person Re-ID aims to retrieval the target person
from the video gallery set. Compared with image-based
methods, video-based methods can extract rich spatiotemporal information in sequences, which helps to achieve
more robust Re-ID. Existing video-based methods mainly
use RNN [11, 12], 3D-CNN [13, 14], Transformer [15, 16]
and GNN [10, 17] to model temporal information. RNNbased methods typically aggregate spatial features by feeding CNN features into an RNN [18] or its variant LSTM
[19]. While RNNs are capable of capturing long-term temporal clues, RNN-based methods are difficult to train and
have limited effectiveness in modelling temporal information in Re-ID tasks [8]. 3D-CNN-based methods [6, 4]
mainly extract spatio-temporal information by 3D convolution. By employing multiple 3D convolutions with different temporal kernels, these methods can capture both

• We propose a novel framework, DHGN, for videobased person re-identification. DHGN adaptively
captures temporal clues at different granularities to
cope with occlusions and to model the motion infor2


exec
/bin/zsh -lc "pdftotext -l 3 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Multi-Scale Dynamic Fusion for Visible-Infrared Person
Re-Identification
SHEN WANG, YU WANG, and RENJIE QIAO, College of Intelligent Systems Science and
Engineering, Harbin Engineering University, Harbin, China
KEJUN WU, School of Electronic Information and Communications, Huazhong University of Science
and Technology, Wuhan, China
CHIA-WEN LIN, Department of Electrical Engineering, National Tsing Hua University, Hsinchu, Taiwan
CHENGTAO CAI, College of Intelligent Systems Science and Engineering, Harbin Engineering
University, Harbin, China, Heilongjiang Provincial Key Laboratory of Environment Intelligent Perception,
Harbin, China, and Key laboratory of Intelligent Technology and Application of Marine Equipment, Harbin
Engineering University, Harbin, China
Visible-infrared person re-identification (VI-ReID) aims to match persons across visible and infrared modalities;
however, its performance is prone to complex dynamic scenes, such as occlusions, background shifts, and
pose changes. In this paper, we propose a Multi-scale Dynamic Fusion Network (MDFN) to address these
challenges in the VI-ReID task. Specifically, the proposed MDFN consists of the Dynamic Feature Fusion
(DFF), Dynamic Perception Enhancement (DPE), and Feature Reweighting with Similarity (FRS) modules. The
DFF module dynamically extracts local and long-range dependencies among features to obtain finer-grained
discriminative features. The DPE module extracts multi-scale features from both visible and infrared modalities
to generate diverse embeddings. The FRS module mitigates the impact of information imbalance between
modalities, thereby further improving performance. Extensive experiments on the SYSU-MM01 and RegDB
datasets show that our MDFN outperforms other state-of-the-art methods, especially in complex dynamic
scenes with occlusions, background shifts, and pose changes.
CCS Concepts: • Computing methodologies → Computer vision;
Additional Key Words and Phrases: Visible-Infrared Person Re-Identification, Modal Alignment, Feature
Enhancement, Dynamic Challenges, Modal Perception Enhancement

This work was supported in part by the Key Project of Heilongjiang Natural Science Foundation (ZD2022F001).
Authors’ Contact Information: Shen Wang (corresponding author), College of Intelligent Systems Science and Engineering,
Harbin Engineering University, Harbin, China; e-mail: watsono@hrbeu.edu.cn; Yu Wang, College of Intelligent Systems
Science and Engineering, Harbin Engineering University, Harbin, China; e-mail: ywang1509@163.com; Renjie Qiao, College
of Intelligent Systems Science and Engineering, Harbin Engineering University, Heilongjiang Harbin, China; e-mail: qiaorenjie@hrbeu.edu.cn; Kejun Wu, School of Electronic Information and Communications, Huazhong University of Science and
Technology, Wuhan, China; e-mail: wukejun_hust@163.com; Chia-Wen Lin, Department of Electrical Engineering, National
Tsing Hua University, Hsinchu, Taiwan; e-mail: cwlin@ee.nthu.edu.tw; Chengtao Cai (corresponding author), College
of Intelligent Systems Science and Engineering, Harbin Engineering University, Harbin, China, Heilongjiang Provincial
Key Laboratory of Environment Intelligent Perception, Harbin, China, and Key laboratory of Intelligent Technology and
Application of Marine Equipment, Harbin Engineering University, Harbin, China; e-mail: caichengtao@hrbeu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/3-ART101
https://doi.org/10.1145/3715330
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.

101:2

S. Wang et al.

ACM Reference format:
Shen Wang, Yu Wang, Renjie Qiao, Kejun Wu, Chia-Wen Lin, and Chengtao Cai. 2025. Multi-Scale Dynamic
Fusion for Visible-Infrared Person Re-Identification. ACM Trans. Multimedia Comput. Commun. Appl. 21, 3,
Article 101 (March 2025), 24 pages.
https://doi.org/10.1145/3715330

1

Introduction

Person re-identification (ReID) focuses primarily on matching images captured by visible
(VIS) light cameras, essentially addressing single-modal person-matching tasks. However, under
low-light conditions, general surveillance cameras struggle to capture clear person information.
Advanced surveillance systems can automatically switch from VIS light mode to infrared (IR)
mode under insufficient lighting conditions, such as at nightfall, thereby capturing clearer images
of individuals’ characteristics. There are significant differences between the images captured in
the two modalities. VIS light ( Red, Green, Blue) images have three channels containing the VIS
light color information of red, green, and blue [39, 40], whereas IR images have only one channel
containing the intensity information of near-IR light. Additionally, the wavelength ranges for
VIS and IR light differ, resulting in distinct sharpness and lighting conditions, which can produce
very different results in the images. Therefore, a new paradigm of visible-infrared ReID (VIReID) [38] has emerged and has been rapidly deployed for video surveillance applications in
recent years.
VI-ReID is essential for surveillance and security systems, particularly in complex, dynamic
environments. In scenarios such as public security monitoring [20], search and rescue operations,
and night-time surveillance, VI-ReID enables consistent monitoring under variable lighting and
weather conditions, allowing reliable identification even when VIS cameras lack detail under
low-light conditions [41]. However, implementing VI-ReID in real-world applications remains
challenging, especially due to occlusions, pose changes, and background shifts (illustrated in
Figure 1) that hinder accurate cross-modal matching. Different from traditional single-modal
methods, the fusion of multi-modal data requires finding a suitable method to bridge the gap
between the two modalities [12]. These modal differences can lead to significant matching difficulties
due to differences in imaging principles and environmental factors such as illumination conditions
and sensor characteristics. Therefore, while VI-ReID holds great promise for improving security and
surveillance, addressing challenges such as modality gaps, occlusion, and environmental variability
is critical to achieving reliable performance in diverse, dynamic scenarios.
To address the aforementioned challenges, we propose a multi-scale dynamic fusion network
(MDFN), which comprises three modules: dynamic feature fusion (DFF), dynamic perceptual enhancement (DPE), and feature reweighting with similarity (FRS). The DFF module
incorporates a local unit that uses adaptive group convolution (AGC) to capture local features,
alongside a global unit enhanced with an improved multi-head self-attention mechanism [33].
The DPE module employs deformable dilated convolution with various dilation rates to capture
multi-scale spatial information and enhance feature embeddings through the modality-aware
hierarchical contrast (MHC) loss. Additionally, to address inaccuracies in the query library affinity matrix, we introduce the FRS module, which optimizes matching performance by leveraging
similarities between internal gallery features. By combining these modules, fine-grained features
can be extracted efficiently to address dynamic challenges such as lighting variations, occlusion,
pose changes, and complex background shifts.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.

Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification

101:3

Fig. 1. The challenges of VI-ReID: the samples (a), (b), and (c) provide three visible-infrared paired images of
persons. Visible images suffer from occlusion, background shifts and pose changes, while infrared images
yield low clarity and high noises. Developing an effective VI-ReID solution robust to these dynamic changes
remains a challenge to be addressed.

Our main contributions are summarized as follows:
— To address the inherent challenges of complex dynamic scenes in VI-ReID, we propose an
MDFN framework to improve feature extraction and enhancement, as well as modality
alignment from VIS and IR images.
— Our MDFN framework integrates the DFF module to extract fine-grained features by fusing
the extracted local and long-range dependencies among features, the DPE module to enhance
multi-scale spatial dependencies and feature representation, and the FRS module to generate
accurate feature vectors for retrieval by utilizing higher-order similarity information in the
query and gallery sets. Therefore, our MDFN framework is capable of tackling complex
dynamic changes across VIS and IR modalities.
— Extensive experiments demonstrate that our proposed MDFN outperforms the existing stateof-the-art VI-ReID methods overall in terms of the Rank-1 and mAP metrics on the commonly
used SYSU-MM01 and RegDB datasets.
2
2.1

Related Work
Single-Modality ReID

ReID is the identification of a target person in an existing video sequence of possible sources with
nonoverlapping camera fields of view. It is widely considered a subproblem of image retrieval.
Given a monitored person image, the image of that person across devices is retrieved.
Current research focuses on discovering discriminative representations that are invariant to
light, resolution, human pose, occlusion, and other confounding factors in VIS lighting for use
in person ReID tasks. SEAS [58] enhances appearance-based ReID by using an implicit 3D body
shape representation as a pixel-level guide. This approach incorporates body shape knowledge in
addition to appearance features for identity feature extraction enhancement with no additional
computational cost. EIE [24] proposes an enhancement, integration and extension approach aimed
at activating the representation of detailed features in occluded ReID. The robustness and feature
representation of the model are improved by means of a self-enhancement module, a cointegration module, and a matching extension module. SecureReID [50] proposes a federated learning
reversible anonymization framework that generates full-body anonymized images with little or no
impact on re-identification performance. In addition, the authors introduce an identity-specific
encryption-decryption architecture to enhance security, where anonymized images are encrypted
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.


exec
/bin/zsh -lc "pdftotext -l 3 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 11, NOVEMBER 2025

10891

Multi-Model Synergy Perception for Open-World
Person Re-Identification
Zhipu Liu

and Lei Zhang , Senior Member, IEEE

Abstract—Open-world person re-identification aims to train
a model on source domains and generalize well on unseen
domains. Existing domain generalizable person re-identification
methods primarily employ the equality training paradigm to
train the model on multi-source domains. However, in open-world
scenarios, domain imbalance often causes domain bias issue that
leads to sub-optimal generalization ability, which is seriously
overlooked. In this paper, we propose a Multi-model Synergy
Perception (MSP) framework equipped with an Asynchronous
Training Paradigm (ATP) on biased domains to maintain the
domain balance for exploring the domain-invariant features.
With the philosophy of divide and conquer, we divide the biased
source domains into multiple debiased sub-source domains and
employ a multi-network architecture to learn these sub-source
domains in parallel. Additionally, to better generalize knowledge
across these sub-source domains, we propose a Structure Synergy
Perception (SSP) module that constructs the feature relationship
distribution for each sub-domain and aligns them to map the
unique knowledge to each other. Furthermore, considering the
consistency of sub-source domains, we further propose a Synergy
Distillation Perception (SDP) to improve the model both semantic
and domain generalization ability. The main idea of SDP is to
use the center guided soft label and the part based triplet graph
to distill each submodel, which can facilitate the network to
explore domain-invariant representations of images. Extensive
experiments demonstrate that our method outperforms state-ofthe-arts for open-domain person ReID.
Index Terms—Person re-identification, multi-model synergy
perception, asynchronous training paradigm, structure synergy
perception, synergy distillation perception.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) aims to match people
across non-overlapped camera views. Recent works in
ReID primarily focus on the supervised and unsupervised
domain adaptive ReID and have achieved promising performance on a seen domain (i.e., a domain with training data)
[1], [2]. However, directly applying these well-trained models

Received 29 September 2024; revised 25 February 2025 and 22 April 2025;
accepted 31 May 2025. Date of publication 6 June 2025; date of current
version 30 October 2025. This work was supported in part by the Scientific
Research Foundation of Chongqing University of Technology, in part by the
National Natural Science Fund of China under Grant 62271090, in part by
Chongqing Natural Science Fund under Grant cstc2021jcyj-jqX0023, and in
part by the National Youth Talent Project. This article was recommended by
Associate Editor J. Cai. (Corresponding author: Lei Zhang.)
Zhipu Liu is with the School of Computer Science and Engineering,
Chongqing University of Technology, Chongqing 400054, China (e-mail:
zpliu@cqut.edu.cn).
Lei Zhang is with the School of Microelectronics and Communication Engineering, Chongqing University, Chongqing 400044, China (e-mail:
leizhang@cqu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3577279

TABLE I
T HE S TATISTICS OF THE DATASETS . D OMAIN I MBALANCE I S S HOWN

to open domains suffers from drastic performance degradation
due to domain shift [3], [4]. In open-world applications, how
to guarantee the open-domain generalization of a ReID model
is an intractable challenge.
To meet this goal, domain generalization (DG) has recently
emerged, which follows the training protocol with one or
multiple labeled source domains [5], [6]. Most previous DG
methods [7], [8], [9] assume a homogeneous environment
where the source and target domains share the same label
space. However, the task of DG for person re-identification
(DG Re-ID) faces a more challenging open-set setting without
ID overlap across domains [10], [11], [12], [13]. To meet
the challenge, a leading problem is faced: identity imbalance and domain bias among different sources. As shown in
Table I, obvious domain imbalance among public benchmarks
is observed. This has been a long-standing problem that deep
learning models are easily biased to majority classes [14],
[15], [16]. Based on this observation, we reasonably infer that
DG model is easily biased to large-scale source domain and
thus neglects domain-invariant learning of small-scale source
datasets.
To verify that, we conduct an exploratory experiment by
training an vanilla DG model (ERM) on 4 source domains,
as shown in Fig. 1. From Fig. 1(a), we see that small-scale
domain 1&2 converges faster than large-scale domain 3&4.
With further iterations, as is shown in Fig. 1(b), overfitting
happens and the performances of domain 1&2 and the open
test domain are degraded. This indicates the domain imbalance
leads to a collapse risk of DG ReID model, which, however, is
seriously overlooked. To solve this issue, an intuitive way is to
re-weight the samples, a widely-used strategy in the long-tailed
recognition task [17], [18], [19]. However, such re-weighting
leads to inadequate learning of domain-invariant representation
by rudely reducing importance of some domains. This is
because each domain is valuable in domain-invariant learning,
as a rather challenging task.

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.

10892

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 11, NOVEMBER 2025

Fig. 1. An exploratory experiment by training a standard DG model on 4
source domains, where domain 1&2 represent small-scale datasets and domain
3&4 represent large-scale datasets.

In this paper, we propose a Multi-model Synergy Perception (MSP) framework to construct balanced domains for
exploring domain-invariant features. For this purpose, we
first develop an Asynchronous Training Paradigm (ATP) that
divides the biased source domains into multiple debiased
subdomains and each subdomain has rich domain diversity.
To fully exploit multiple subdomains for domain-invariant
features learning, inspired by the philosophy of divide and
conquer, we construct a multi-network architecture to learn all
subdomains in parallel. Specifically, we use multiple parallel
sub-models to learn different subdomains to exploit different
batch knowledge of the large-scale domain, while sharing
the same batch knowledge of the small-scale domain for
knowledge asynchronous learning. To keep domain balance,
different sub-models are independent of each other during
training process. Based on the idea of multi-submodel training,
another problem is faced: naively aggregating all parallel
local sub-models as federated learning may lead to inferior
asynchronous optimization due to lacking synergy perception
among sub-models. Considering that the distance relationship
of features can represent the retrieval model knowledge, we
propose a Structure Synergy Perception (SSP) module that
constructs the feature relationship distribution for each subdomain and aligns them to map the unique knowledge to
each other for synergy learning. Specifically, the structure
distribution is built based on the feature relationship of each
subdomain and then we align structure distribution of different subdomains to better generalize knowledge across these
subdomains.
Additionally, each submodel can only exploit singleview generalizable knowledge, and the conventional multisubmodel training strategies are unable to complement the
insufficient view with each other due to lacking interaction
learning among sub-models. Considering the consistency of
sub-source domains, we further propose a Synergy Distillation
Perception (SDP) to aggregate views of different submodels to
improve the model both semantic and domain generalization
ability. The SDP primarily comprises semantic based cross-

model central distillation (CCD) and part graph distillation
(PGD). The main idea of SDP is to use the center guided
soft label and the part based triplet graph to distill each
subnetwork, which facilitates the network to explore domaininvariant representations of images. Specifically, considering
the importance of logits in semantic classification, CCD first
aggregates the output logits of different submodels to obtain
multi-view knowledge. And then this value is used as guided
soft labels to distill each submodel for semantic generalization
learning. Furthermore, existing ReID loss functions primarily
leverage the image-level visual similarities of IDs to explore
the discriminative knowledge, which, however, is unable to
eliminate the domain noise as this noise may exist in all
features extracted from the same model. Considering the visual
difference of different local parts, we develop a Parts Graph
Distillation (PGD) that uses this valuable prior knowledge to
explore the domain invariant features. The PGD divides the
image into several local parts and leverages the parts visual
similarities to construct the parts graph for each subdomain.
And then we employ intra- and inter-subdomain distillation
based on the parts graph to further improve the model domain
generalization ability.
Due to the limitation of small-scale domain, each sub-model
can only acquire partial knowledge of large-scale domain.
And different knowledge allows the sub-models to possess
varying discriminative ability. To obtain the optimal final
model, we propose an adaptive-weight method to aggregate
all submodels during testing process. The testing images are
augmented randomly and we employ contrastive loss function
to compute its similarity, which serves as the weights for the
corresponding submodel. And to improve the reliability of the
learned weights, we augment the image with different ways
and employ cross-computation among the augmented images
for weighs learning.
In summary, the contributions of this paper are three-fold.
• We propose a Multi-model Synergy Perception (MSP)
framework equipped with an Asynchronous Training
Paradigm (ATP) for open-world ReID. With the philosophy of divide and conquer, we divide the biased
source domains into multiple debiased subdomains and
employ a multi-network architecture to learn these subdomains in parallel, which facilitates domain-invariant
representations.
• To reinforce the collaborative learning among submodels,
we construct a Structure Synergy Perception (SSP) module based on the feature relationship. SSP constructs the
feature relationships distribution of each subdomain and
aligns them to map the unique knowledge to each other
for synergy learning.
• Since each submodel can only exploit single-view generalizable knowledge, we further propose a Synergy
Distillation Perception (SDP) to aggregate views of different submodels. The SDP comprises semantic based
cross-model central distillation (CCD) and parts graph
distillation (PGD), which uses the center guided soft label
and the part based triplet graph respectively to distill each
subnetwork for improving the model both semantic and
domain generalization ability.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.

LIU AND ZHANG: MULTI-MODEL SYNERGY PERCEPTION FOR OPEN-WORLD PERSON RE-IDENTIFICATION

II. R ELATED W ORK
A. Supervised Person ReID
In the last decade, deep learning based methods have
achieved great progress for fully-supervised person Re-ID
[20], [21]. The fully-supervised methods primarily explore
the discriminative features to improve the retrieval results,
which can generally grouped into three categories including:
(1) global image based algorithms [22], [23], (2) part based
methods [20], [24] and (3) attention based methods [21], [25].
Hermans et al. [23] propose a hard sample based triplet loss
function to increase the differences among classes. And to
exploit the details for fine-grained retrieval, several methods
[20], [24] propose to parts division strategies for learning more
reliable features. Then to guide the model to focus on the
discriminative regions, several methods [21], [25] leverage the
attention mechanism to relieve the inference of background.
However, these methods are usually trained and tested on a
single domain, which may suffer from dramatic degradation
if applied on an open domain that is previously not accessed
during training due to domain shifts [4], [26], [27].
B. Unsupervised Domain Adaptation (UDA) for Person ReID
UDA assumes that a labeled source data and an unlabeled
target data can be accessed to train the model. Early semisupervised and weakly-supervised methods require the pseudo
label of target data [28], [29], [30]. Since the target labels
are hard to access in some environments, some methods [31],
[32], [33] propose to leverage the clustering algorithms to get
and update the pseudo label of target data, which achieves
promising results that is close to the labeled environment.
And considering the domain shift, some methods relies on
generative adversarial networks (GAN) [34], [35], [36] to
transfer the style of images from source domain to target
domain. Although promising results are achieved by UDA,
the acquisition target data is prohibitive in some environment
and the model adaptation is time-consuming. Another more
challenging yet practical setting of domain generalization
(DG) is initiated, where only labeled source data can be
accessed during training.
C. Domain Generalization (DG)
DG methods aim to train a generalizable model by labeled
source domains and the trained model is required to be directly
applied to any unseen domain. The recent DG methods can
be roughly grouped into three categories: Domain-invariant
representation learning [37], [38], [39], domain augmentation
[40], [41], [42], [43], [44] and meta-learning [45], [46], [47].
Some methods [39], [48] employ adversarial training strategy
to disentangle the domain-invariant features. The performance
of these methods is limited to the diversity of training data,
so many methods leverage the GAN to increase the source
domain diversity, which achieves promising performance and
can be inserted to other models. In recent years, Metalearning technique is introduced for DG [45], [46], [47],
which simulates the train-test pipeline of model training to
improve the generalization ability. These methods achieves

10893

great progress for DG classification tasks, where the source
and target domains share the same label space (close-set).
However, the DG ReID assumes a more challenging openset setting that there is no ID overlap across domains, which
makes it challenging to directly apply these DG classification
methods to DG ReID.
D. Domain Generalization for Person ReID (DG ReID)
DG ReID assumes a more challenging open-set setting that
there is no ID overlap across domains [49], [50]. The domain
guided dropout (DGD) [51] is the first multi-dataset domain
generalization work in person ReID. Then, some methods [10],
[52] explore the domain-invariant features by relieving the
style shift of training and testing domains. DualNorm [52]
inserts instance normalization (IN) [53] to reduce domain
bias. However, this method can eliminate some important
discriminative information. To relieve this issue, [10] proposes
a Style Normalization and Restitution (SNR) module to filter
out style variations by IN. Recently, the meta-learning strategy is also introduced into the DG ReID [13], [54]. Zhao
et al. [54] propose a memory-based multi-source meta-learning
(M3L) network, which randomly splits the source domains
into meta-train and meta-test sets to simulate the train-test
process of DG. Choi et al. [13] proposes a Meta BatchInstance Normalization (MetaBIN). The MetaBIN reconsiders
the under-style effect of batch normalization (BN) and overstyle effect of instance normalization (IN) and proposes a
learnable balancing parameter between BN and IN to learn
generalizable and discriminative features. And to improve the
generalization of balancing parameter, the MetaBIN employs
the meta-learning strategy to learn this parameter. Moreover,
a novel Meta Distribution Alignment (MDA) method [55]
is proposed to enable the model to share similar distribution in a test-time-training fashion. Ni et al. [56] propose
a pure Transformer model (termed Part-aware Transformer)
for DG-ReID by designing a proxy task to mine local visual
information shared by different IDs. Chen et al. [57] propose
a multi-scale query-adaptive convolution framework to encode
patches for generalization learning. Jiao et al. [58] develop a
new normalization scheme called Dynamically Transformed
Instance Normalization (DTIN) to alleviate the drawback of
IN (i.e., the limitation of eliminating discriminative patterns).
However, domain imbalance is seriously overlooked by DG
and DG Re-ID communities.
E. Model Ensemble
Model ensemble aims to mix multiple models to improve
the overall model’s capability. Reference [59] is the first
method to leverage multi-model to learn a system composed of
many separated networks. Then, [11] proposes a multi-experts
architecture, which leverages different models to handle different source domain, and uses a vote technique to aggregate
multiple models. Additionally, several methods [60], [61]
leverage multiple models to explore both domain-invariant and
domain-specific features. Reference [60] proposes a Mimic
Embedding via adapTive Aggregation (META) method, which
shares all the parameters except for the batch normalization

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Multi-year long-term person re-identification using gait and HAR features.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112627

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Multi-year long-term person re-identiﬁcation using gait and HAR features
David Freire-Obregón ∗, Oliverio J. Santana , Javier Lorenzo-Navarro
Daniel Hernández-Sosa, Modesto Castrillón-Santana

,

Universidad de Las Palmas de Gran Canaria, Institute of Intelligent Systems and Numeric Applications in Engineering, Las Palmas de Gran Canaria, 35007, Spain

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Biometrics
Gait
Human action recognition

a b s t r a c t
We propose a two-stream person re-identiﬁcation (Re-ID) framework that integrates gait and human action recognition (HAR) through cross-attention fusion. The model processes gait sequences via a BiLSTM-based encoder to
capture temporal motion dynamics. At the same time, HAR embeddings are extracted using pre-trained video
backbones and distilled into compact behavioral features. These two modalities are fused using a cross-attention
mechanism, enriching gait-based identity representations with context-aware activity cues. We evaluate our
method on a newly curated long-term spatio-temporal dataset of ultra-distance runners captured in natural outdoor settings across multiple locations spanning three years (2020 to 2023). Experimental results demonstrate
that integrating HAR signiﬁcantly enhances gait-based Re-ID performance. Compared to gait-only models, our approach yields a 12 % improvement in mean Average Precision (mAP) in cross-year scenarios and up to an 11.6 %
gain in same-year evaluations. The HAR-enhanced models also exhibit faster convergence and higher Rank-1
accuracy, establishing the eﬀectiveness of multi-modal motion-based representations for long-term, real-world
person Re-ID.

1. Introduction
Humans can recognize familiar individuals across diverse contexts
and times. Biometrics automates this process using physical, chemical,
or behavioral traits [1]. Typically, biometric veriﬁcation systems assume a known gallery of identities created during registration. In contrast, when identities are not pre-registered, the task shifts to linking
observations of the same individual across time and space, regardless
of their true name. This problem, known as person re-identiﬁcation
(Re-ID), involves retrieving an individual across diﬀerent cameras or
time spans [2]. In this work, we focus on supervised Re-ID, where labeled identities guide the learning of embeddings. Most research emphasizes short-term image-based Re-ID, where appearance remains stable. Real deployments, however, require long-term Re-ID, where clothing and visual cues change over time. Video provides richer temporal
information, with gait serving as a stable biometric foundation. Nevertheless, benchmarks for long-term, video-based Re-ID remain scarce.
DeepChange [3] is among the few, though its imbalance in identity frequency poses challenges, and its use in video-based scenarios is limited.
Long-term Re-ID in crowded, dynamic environments is particularly difﬁcult: individuals may look alike, appear brieﬂy, or undergo occlusions,
motion blur, and domain shifts. Appearance-based cues often fail under
such conditions. Motion signals such as gait, together with human ac-

tion recognition (HAR), oﬀer complementary stability and behavioral
context. Despite this, they are rarely combined in trainable, end-to-end
systems.
Our main contributions are as follows. First, we propose a twostream architecture that combines gait and HAR features for person ReID. A gait dynamics is learned in a structured way by a BiLSTM branch,
and a light-weight and well-cited baseline in video recognition is obtained by the HAR branch, producing compact clip-level embeddings by
global-temporal average pooling, in order to keep the auxiliary branch
light-weight. Then, we incorporate a cross-attention scheme enabling interactive mixing of the two feature streams, allowing for a ﬂexible combination of activity and motion cues. We utilized a triplet loss function
to uncover discriminative and generalizable representations. Second, we
introduce a new dataset collected in unconstrained environments, where
ultra-distance race participants were recorded at two locations in 2020
and at two additional locations in 2023. This design facilitates evaluation under both short-term conditions (same-day/within-year) and longterm conditions (across multiple years), thereby capturing realistic variations in appearance, viewpoint, and environment.
Experiments show that fusing gait and HAR outperforms singlestream and visual-only baselines, yielding more robust embeddings. Results highlight the potential of motion-based multimodal representations
for Re-ID in unconstrained conditions.

∗ Corresponding author.

E-mail address: david.freire@ulpgc.es (D. Freire-Obregón).
https://doi.org/10.1016/j.patcog.2025.112627
Received 2 June 2025; Received in revised form 12 September 2025; Accepted 18 October 2025
Available online 23 October 2025
0031-3203/© 2025 The Author(s). Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

Pattern Recognition 172 (2026) 112627

D. Freire-Obregón et al.

Fig. 1. Pipeline of the proposed two-stream architecture. Gait and HAR features are extracted independently, aligned in dimension, and fused via a cross-attention
mechanism where gait attends to HAR. The fused embedding is used for identity representation. This design prioritizes gait as the primary cue while enriching it
with complementary HAR context.

2. Related work

26,000 subjects in unconstrained environments. Other datasets, such
as FVG [17], contribute to advancing cross-view and in-the-wild gait
recognition by oﬀering high-quality visual gait data under naturalistic
conditions.
However, these datasets typically lack long-term variability, as they
do not include recordings of the same individuals across multiple years.
In contrast, one of our key contributions is a new dataset comprising
real-world race footage recorded over three years (2020 and 2023) and
from diﬀerent distant locations within the same year. This enables the
evaluation of gait-based Re-ID under realistic conditions of appearance
change, long-term temporal gaps, and spatial variability, a scenario not
addressed in existing benchmarks.

Person Re-ID has traditionally focused on visual appearance cues
such as color, clothing, and facial features. However, in real-world longterm scenarios involving large-scale outdoor events or cross-year comparisons, appearance cues often fall short due to lighting variations,
occlusions, and clothing changes [4]. To address these challenges, researchers have explored motion-based modalities such as gait as complementary or alternative identity signals [5]. Our work builds on this
line of research by jointly modeling gait and HAR in a uniﬁed architecture for spatio-temporal Re-ID [6].
Re-ID in time and space. Spatio-temporal Re-ID aims to recognize individuals across signiﬁcant temporal and spatial gaps, where
appearance-based models are particularly vulnerable to domain shift
and context changes [7]. Existing methods often address these issues by
incorporating temporal modeling [8] or leveraging soft biometrics such
as gait [9]. In this regard, gait is especially promising in long-term ReID due to its inherent stability and robustness against changes in clothing and viewpoint. However, many current models still struggle when
faced with unconstrained real-world variability, such as that found in
long-term datasets.
Appearance-based gait recognition. Gait recognition methods can
be broadly categorized into skeleton-based and appearance-based approaches [9]. Our work considers 2D silhouettes used to extract discriminative motion patterns. More recent deep learning-based models,
such as GaitSet [10], use set-based learning to handle unaligned input
frames. GaitPart [11] introduces part-based modeling to capture local
motion features better. These approaches have demonstrated strong performance under controlled conditions but often lack robustness in dynamic, real-world environments.
HAR and multi-modal cues. HAR provides a higher-level understanding of behavior that can complement gait in identity reasoning. Although HAR has been widely studied for activity classiﬁcation
tasks [12], its use in person Re-ID remains limited. Recently, pre-trained
HAR models for person Re-ID in ultra-distance sports scenarios have
been explored, emphasizing the impact of fatigue-induced movement
changes on recognition performance [13]. However, their approach
does not explicitly incorporate gait analysis. We aim to bridge this gap
by jointly embedding HAR and gait cues, enabling richer, context-aware
representations for spatio-temporal Re-ID.
Datasets. Several benchmark datasets support gait-based person
Re-ID research. CASIA-B [14] remains one of the most widely used
datasets, featuring gait sequences from 124 subjects under varying viewpoints and conditions (e.g., normal, bag-carrying, and coat-wearing).
The OU-ISIR gait datasets, including the large-scale OU-MVLP [15], provide extensive samples across diﬀerent age groups, clothing variations,
and viewpoints. OU-MVLP, in particular, includes over 10,000 subjects
captured from 14 view angles, making it one of the most extensive
multi-view gait datasets available. GREW [16] is a recent large-scale
dataset collected from real-world surveillance footage containing over

3. Methodology
This section describes the proposed two-stream architecture for
spatio-temporal person Re-ID (see Fig. 1), which integrates gait and human action information through pre-trained backbones. We ﬁrst formalize the problem, then detail the structure and roles of the gait and HAR
backbones, the fusion strategy based on cross-attention, and ﬁnally, the
training objective using triplet loss with semi-hard negative sampling.
3.1. Problem formulation
Let  = {(𝑣𝑖 , 𝑦𝑖 )}𝑁
be a dataset of 𝑁 raw video recordings, where:
𝑖=1
𝑣𝑖 denotes the 𝑖th input video captured at a particular time and location,
• 𝑦 ∈ {1, … , 𝐶} is the identity label associated with 𝑣 .
𝑖
𝑖
•

Each video 𝑣𝑖 is processed through two modality-speciﬁc pipelines
to extract temporal embeddings for gait and HAR.
Gait Processing Pipeline. The gait modality aims to model the subtle motion patterns unique to each individual. To extract these cues, the
raw video 𝑣𝑖 is ﬁrst processed into a sequence of binary silhouettes by
combining person detection and pose estimation. Speciﬁcally, we use
YOLOv8 [18] and Bot-SORT [19] for robust multi-object tracking, enabling consistent localization of the subject across frames. Silhouettes
are later generated using SAMURAI [20]. The resulting silhouette sequence serves as input to a pre-trained gait encoder GAIT , which has
been trained on large-scale public datasets such as CASIA-B, OUMVLP,
and GREW.
The encoder produces a feature tensor from the silhouette sequence:
𝐸𝑖GAIT = GAIT (𝑣silhouettes
) ∈ ℝ𝐷𝑔 ×𝑃
𝑖
Where 𝐷𝑔 is the feature dimension and 𝑃 represents the number of horizontal partitions of the body used during Horizontal Pyramid Pooling
(HPP). This output captures part-level information rather than temporal dynamics; each row corresponds to a distinct horizontal region of
the body (e.g., upper torso, lower legs), not a time step.
2

Pattern Recognition 172 (2026) 112627

D. Freire-Obregón et al.

To ensure stability, each part embedding vector is ﬁrst standardized:
𝐸̃ 𝑖 [∶, 𝑝] =

Since the output dimensions of the gait and HAR streams may diﬀer
(𝐷𝑔 ≠ 𝐷ℎ ), we apply a projection layer to transform the HAR representation into the common dimension 𝐷:

𝐸𝑖GAIT [∶, 𝑝] − 𝜇train
𝜎train

ℎHAR
= 𝜙proj (ℎ̄ HAR
) ∈ ℝ𝐷
𝑖
𝑖

and then L2-normalized:
𝐸̂ 𝑖 [∶, 𝑝] =

Both ℎGAIT
∈ ℝ𝐷 and ℎHAR
∈ ℝ𝐷 are projected into a shared embed𝑖
𝑖
ding space of dimension 𝐷, where 𝐷 denotes the common projection
size (i.e., 𝐷𝑔 = 𝐷), ensuring compatibility for subsequent fusion. The
resulting pair of embeddings (ℎGAIT
, ℎHAR
) are aligned in dimension and
𝑖
𝑖
subsequently used for cross-attention-based fusion and identity embedding computation.

𝐸̃ 𝑖 [∶, 𝑝]
‖𝐸̃ 𝑖 [∶, 𝑝]‖2

This results in a sequence of normalized part-based embeddings
𝐸̂ 𝑖GAIT ∈ ℝ𝐷𝑔 ×𝑃 . Although they form a sequence-like structure, the order of parts corresponds to spatial locations rather than chronological
time.
Part-wise Encoding. To capture the structured spatial information encoded in the body parts, the normalized part descriptors 𝐸̂ 𝑖GAIT are interpreted as a sequence and passed through a BiLSTM. While this sequence
does not represent time, the recurrent architecture allows for context
modeling across adjacent body regions. This allows the model to capture structured co-movement patterns (e.g., torso-leg coordination) that
are informative for identity. We use a bidirectional LSTM to aggregate
such spatial dependencies. Let 𝐸̃ 𝑖 = (𝐸̂ 𝑖GAIT )⊤ ∈ ℝ𝑃 ×𝐷𝑔 , where the part
dimension 𝑃 is treated as the temporal axis.

Rationale for temporal aggregation. In the HAR stream, we use global
temporal average pooling in order to obtain clip-level embeddings. This
keeps the HAR stream lightweight, computationally robust, and easy to
train in parallel with the combining module, aligning with typical practice in recent video recognition backbones (e.g., C2D [21], I3D [22],
SlowFast [23], X3D [24]). Shallow recurrent layers and temporal 1D
convolutions in our experiments did not show consistent improvement
under long-term Re-ID, while average pooling kept accuracy with a modest cost. Even though more intricate aggregation methods are potentially
available (e.g., self-attention, Transformer pooling), we intentionally select simplicity and robustness in order to uncover the isolated contribution of HAR cues towards gait-based recognition.

(
(
(
)))
ℎGAIT
= 𝜙 BiLSTM256 Dropout BiLSTM128 (𝐸̃ 𝑖 )
𝑖
Here, 𝜙 denotes a dense layer with ReLU activation. This conﬁguration models spatial body part embeddings as a pseudo-temporal sequence, allowing the BiLSTM to capture part-to-part relational dynamics
relevant to identity.
HAR Processing Pipeline. The HAR stream is designed to capture
high-level activity patterns from the subject’s movement. However, raw
videos may contain multiple actors, visual clutter, or background distractions To ensure that the extracted embeddings focus solely on the
subject of interest, we apply a context-constrained preprocessing step.
Step 1: Video Preprocessing - Context Constraint. We use the silhouettes
previously computed to crop a tight region around the individual in each
frame. The rest of the frame is suppressed by superimposing the silhouette onto a static mode frame 𝑓 . For an individual 𝑖 at time 𝑡 ∈ [0, 𝑇 ], let
𝑆𝑖𝑙(𝑖) (𝑡) be the silhouette, and 𝐹 (𝑖) (𝑡) the raw frame. The preprocessed
frame is deﬁned as:
(
)
𝐹 ′(𝑖) (𝑡) = Crop(𝐹 (𝑖) (𝑡), 𝑆𝑖𝑙(𝑖) (𝑡)) + 1 − 1𝑆𝑖𝑙(𝑖) (𝑡) ⋅ 𝑓

3.2. Cross-attention fusion
We apply a cross-attention mechanism to fuse both modalities, where
the gait embedding attends to the HAR representation. Given the aligned
embeddings (ℎGAIT
, ℎHAR
) for sample 𝑖, we deﬁne:
𝑖
𝑖
𝑄𝑖 = ℎGAIT
∈ ℝ1×𝐷
𝑖
𝐾𝑖 = 𝑉𝑖 = ℎHAR
∈ ℝ1×𝐷
𝑖
The attention weights and attended HAR embedding are computed
as:

′

where 𝑇 ′ is the number of temporal segments or frames output by the
HAR backbone and 𝐷ℎ is the dimensionality of each HAR embedding
vector.
After extracting the sequence of embeddings from the HAR backbone, we apply normalization frame-wise before pooling. First, the embeddings are standardized using the mean and standard deviation computed across the training set, and then, each frame is L2-normalized:

HAR
𝜎train

,

𝐸̂ 𝑖HAR =

)
,

𝑧𝑖 = 𝛼𝑖 ⋅ 𝑉𝑖

where [ ; ] denotes concatenation and 𝜙proj is a dense projection layer.
In this setup, gait is deﬁned as the querying modality (serving as
the Query in the attention mechanism) because the primary objective
of the framework is identity recognition rather than activity categorization. Gait embeddings are explicitly trained to discriminate identities
and thus form the base representation to be preserved. HAR features, in
turn, provide the Keys and Values, oﬀering complementary contextual
cues that enrich the gait signal without shifting the focus toward action
recognition. Through the attention mechanism, the model selectively
integrates relevant HAR information into the gait embedding. The resulting attended representation is then combined with the original gait
embedding, ensuring that gait remains the dominant identity signal. Finally, a projection layer fuses both sources into a uniﬁed identity representation in a common embedding space, where samples can be directly
compared using L2 distance for Re-ID.

𝐸𝑖HAR = HAR (𝑣context
) ∈ ℝ𝑇 ×𝐷ℎ
𝑖

HAR
𝐸𝑖HAR − 𝜇train

𝑄𝑖 𝐾𝑖⊤
√
𝐷

We then concatenate the original gait embedding with the attended
HAR vector and apply a projection layer 𝜙 to obtain the ﬁnal fused
identity representation:
(
)
𝑓𝑖 = 𝜙proj [ℎGAIT
; 𝑧𝑖 ] ∈ ℝ𝐷
𝑖

This produces a context-constrained video 𝑣context
= {𝐹 ′(𝑖) (𝑡)}𝑇𝑡=0 fo𝑖
cused solely on the individual.
Step 2: HAR Feature Extraction. The processed video 𝑣context
is then
𝑖
passed through a pre-trained action recognition backbone HAR , which
outputs a sequence of temporal embeddings:

𝐸̃ 𝑖HAR =

(

𝛼𝑖 = softmax

𝐸̃ 𝑖HAR
‖𝐸̃ HAR ‖2
𝑖

3.3. Triplet loss for metric learning

Although temporal modeling techniques, such as those used in gait
recognition, aim to capture patterns, we observed that naive temporal
processing alone did not yield signiﬁcant improvements in Re-ID performance. Then, average pooling is applied over the temporal dimension
to aggregate frame-level features into a sequence-level representation:

To teach the model how to tell diﬀerent people apart, we use a triplet
loss function. It compares three examples at a time:
•

ℎ̄ HAR
= AvgPool(𝐸̂ 𝑖HAR ) ∈ ℝ𝐷ℎ
𝑖

•

3

an anchor (𝑎), a reference video of a person,
a positive sample (𝑝), another video of the same person,


codex
基于每篇前三页读到的摘要和引言，下面只提炼对找创新点有用的信息。

**逐篇提取**
1. `Multi Queue for Unsupervised Person Re-identification`  
类型：新机制偏工程。把无监督 ReID 的 memory bank 更新问题改成每个伪身份维护多队列，再配可学习温度。  
能发点：抓住 momentum 更新里历史特征过旧、负样本构造不稳的问题，用队列长度和温度消融搭证据。  
套路：不是再换聚类，而是说“memory bank 的更新策略本身就是训练信号”。

2. `Multi-Branch Clothes-Agnostic Feature Learning`  
类型：工程组合加语义机制。用 LVLM 生成衣服粗细粒度描述，再用文本注意力和解析掩码压低衣服影响。  
能发点：现有文本法多描述非衣服属性，语义不准；它反过来精确描述衣服，然后把衣服从特征里扣掉。  
套路：先“认识衣服”，再“忽略衣服”。

3. `Multi-feature balanced network`  
类型：工程组合。全局、无衣服区域、轮廓掩码三分支，再做知识转移、注意力和交叉融合。  
能发点：单一无衣服特征在姿态、遮挡、光照下不稳，多线索互补能补短板。  
套路：换衣 ReID 不是找一个永远可靠的特征，而是动态平衡多个弱可靠特征。

4. `Multi-Granularity Attribute Prompt Learning`  
类型：新机制。把属性识别、属性文本、衣服擦除图和视觉深提示放在双分支里做多粒度解耦。  
能发点：已有方法用 one-hot 属性或衣服标签，没用属性文本，也依赖人工衣服标注；它用伪属性和文本语义补高层监督。  
套路：从像素、标签、文本三个层次同时增强身份属性、抑制衣服属性。

5. `Multi-granularity collaborative constraint feature alignment network`  
类型：新机制。无监督 ReID 中不只挖全局和局部互补，而是显式约束不同粒度语义对齐。  
能发点：伪标签噪声加局部冗余会产生假的互补信息，中心约束过滤能让 memory bank 里的局部特征更干净。  
套路：多粒度不是越多越好，关键是不同粒度必须语义一致。

6. `Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification`  
类型：问题重定义加新机制。把视频 ReID 的时间建模改成不同身体部位的可变长度时间线索建模。  
能发点：固定全局时间窗口不能处理遮挡和部位运动差异；动态分割身体区域、同层建图、互近邻连边，证据链比较清楚。  
套路：视频 ReID 不是整个人的一条时间序列，而是多个身体部位各自有不同节奏和可靠性。

7. `Multi-Model Synergy Perception for Open-World Person Re-Identification`  
类型：问题重定义。把开放世界 ReID 的失败归因到源域不平衡和域偏置，而不是笼统域泛化。  
能发点：先用实验说明小域过拟合、大域主导会拖垮未知域，再用异步多模型和结构蒸馏平衡知识。  
套路：开放世界泛化不是多源一起训就行，源域大小本身会制造偏见。

8. `Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification`  
类型：工程组合。局部和全局动态融合、多尺度增强、相似度重加权处理可见光和红外差异。  
能发点：把 VI-ReID 的遮挡、背景、姿态变化统一包装成复杂动态场景，证据来自 SYSU-MM01 和 RegDB。  
套路：跨模态对齐之外，还要处理动态场景下哪些尺度、哪些特征更可信。

9. `Multi-year long-term person re-identification using gait and HAR features`  
类型：新数据设定加问题重定义。三年跨度、真实户外跑者数据，用步态和动作识别特征做长期 ReID。  
能发点：长期视频 ReID 数据少，衣服外观跨年失效，步态和行为上下文能补稳定身份线索。  
套路：长期 ReID 应该从外观检索转成行为生物特征检索。

**这批给我们的直接启发**
“多分支、多粒度、多尺度”本身已经很拥挤。能发 B 类的不是再拼一个分支，而是先指出一个被忽略的真实失败机制，再让机制服务这个问题。对我们来说，最值得借的是三类写法：显式对齐、可靠性建模、源域或视角不平衡。衣服文本和普通多分支换衣方向不建议正面进入，除非能被 SMPL 或航拍地面的几何问题重新定义。

**强创新点候选**
1. 一句话：把航拍到地面的跨视角 ReID 重新定义为人体三维表面坐标上的检索，而不是二维图像域适配。  
挂的资产：CARGO / AG-ReID.v2，SMPL，SOLIDER-Swin。  
区别：最像 MCCAN 和 DHGN 的多粒度对齐，但它们仍在二维 patch 或视频部位上对齐；我们用 SMPL 把 SOLIDER token 投到规范人体表面，只比较共同可见且几何可信的身体表面区域。也不同于换衣 ReID 里的 3D shape，它不是拿 3D 当身份线索，而是拿 3D 当跨视角坐标系。  
廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。

2. 一句话：用姿态热图的不确定性做跨视角部位可靠性估计，让模型知道哪些身体区域在航拍低清条件下不能信。  
挂的资产：PSG / LGPA-D 姿态热图门控，SMPL，航拍地面数据。  
区别：PSG 类方法把姿态当注意力提示，默认姿态可靠；DHGN 按相似度动态连图。这个方向把热图熵、关节置信度、SMPL 可见性作为可靠性变量，学习“这个视角下这个部位该不该参与匹配”。  
廉价首验：不训练模型，直接用已有姿态热图置信度给 SOLIDER 局部特征加权，重点看航拍低清和严重俯视样本。如果错误样本的低置信部位确实更容易误导，并且置信度加权带来超过噪声的提升，再做学习版。

3. 一句话：把开放世界 ReID 的域不平衡收窄成航拍地面 ReID 的视角稀有性问题。  
挂的资产：CARGO / AG-ReID.v2，SOLIDER-Swin。  
区别：MSP 处理的是多数据集大小不平衡；我们处理的是同一个跨视角系统内部的视角、分辨率、俯仰角和高度分布不平衡。核心不是多模型集成，而是防止模型被大量清晰地面视角牵走，忽略少量但关键的航拍视角。  
廉价首验：按视角高度、分辨率、人体框面积、俯视程度分桶，解析 SOLIDER 基线在各桶的 AP 和错配。如果少数航拍桶明显塌陷，再用一个最小的分桶均衡采样或分桶 BN 跑短训。若分桶差异不明显，或均衡采样没有收益，就不做。

4. 一句话：用 SMPL 生成跨视角反事实可见性监督，让模型学习“换视角后哪些身份证据应该保留，哪些证据本来就会消失”。  
挂的资产：SMPL，SOLIDER-Swin，航拍地面跨视角数据，姿态门控。  
区别：换衣论文是擦衣服或描述衣服来压制衣服捷径；VI-ReID 论文是用动态卷积处理复杂场景。这个方向用几何反事实定义跨视角不变性，不是随机遮挡，也不是普通数据增强。  
廉价首验：用 SMPL 可见性生成俯视和地面视角的身体可见区域掩码，只做小规模训练或冻结特征蒸馏，对比随机擦除、普通低清增强和人体水平分块。如果几何可见性监督不能超过这些廉价对照，就停。

我会优先排 1 和 4。它们最能把团队已有的 SMPL、航拍地面数据和 SOLIDER 强主干连成一个问题层面的故事，也最容易和这批“多分支、多粒度”工作切开。
tokens used
89,320
基于每篇前三页读到的摘要和引言，下面只提炼对找创新点有用的信息。

**逐篇提取**
1. `Multi Queue for Unsupervised Person Re-identification`  
类型：新机制偏工程。把无监督 ReID 的 memory bank 更新问题改成每个伪身份维护多队列，再配可学习温度。  
能发点：抓住 momentum 更新里历史特征过旧、负样本构造不稳的问题，用队列长度和温度消融搭证据。  
套路：不是再换聚类，而是说“memory bank 的更新策略本身就是训练信号”。

2. `Multi-Branch Clothes-Agnostic Feature Learning`  
类型：工程组合加语义机制。用 LVLM 生成衣服粗细粒度描述，再用文本注意力和解析掩码压低衣服影响。  
能发点：现有文本法多描述非衣服属性，语义不准；它反过来精确描述衣服，然后把衣服从特征里扣掉。  
套路：先“认识衣服”，再“忽略衣服”。

3. `Multi-feature balanced network`  
类型：工程组合。全局、无衣服区域、轮廓掩码三分支，再做知识转移、注意力和交叉融合。  
能发点：单一无衣服特征在姿态、遮挡、光照下不稳，多线索互补能补短板。  
套路：换衣 ReID 不是找一个永远可靠的特征，而是动态平衡多个弱可靠特征。

4. `Multi-Granularity Attribute Prompt Learning`  
类型：新机制。把属性识别、属性文本、衣服擦除图和视觉深提示放在双分支里做多粒度解耦。  
能发点：已有方法用 one-hot 属性或衣服标签，没用属性文本，也依赖人工衣服标注；它用伪属性和文本语义补高层监督。  
套路：从像素、标签、文本三个层次同时增强身份属性、抑制衣服属性。

5. `Multi-granularity collaborative constraint feature alignment network`  
类型：新机制。无监督 ReID 中不只挖全局和局部互补，而是显式约束不同粒度语义对齐。  
能发点：伪标签噪声加局部冗余会产生假的互补信息，中心约束过滤能让 memory bank 里的局部特征更干净。  
套路：多粒度不是越多越好，关键是不同粒度必须语义一致。

6. `Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification`  
类型：问题重定义加新机制。把视频 ReID 的时间建模改成不同身体部位的可变长度时间线索建模。  
能发点：固定全局时间窗口不能处理遮挡和部位运动差异；动态分割身体区域、同层建图、互近邻连边，证据链比较清楚。  
套路：视频 ReID 不是整个人的一条时间序列，而是多个身体部位各自有不同节奏和可靠性。

7. `Multi-Model Synergy Perception for Open-World Person Re-Identification`  
类型：问题重定义。把开放世界 ReID 的失败归因到源域不平衡和域偏置，而不是笼统域泛化。  
能发点：先用实验说明小域过拟合、大域主导会拖垮未知域，再用异步多模型和结构蒸馏平衡知识。  
套路：开放世界泛化不是多源一起训就行，源域大小本身会制造偏见。

8. `Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification`  
类型：工程组合。局部和全局动态融合、多尺度增强、相似度重加权处理可见光和红外差异。  
能发点：把 VI-ReID 的遮挡、背景、姿态变化统一包装成复杂动态场景，证据来自 SYSU-MM01 和 RegDB。  
套路：跨模态对齐之外，还要处理动态场景下哪些尺度、哪些特征更可信。

9. `Multi-year long-term person re-identification using gait and HAR features`  
类型：新数据设定加问题重定义。三年跨度、真实户外跑者数据，用步态和动作识别特征做长期 ReID。  
能发点：长期视频 ReID 数据少，衣服外观跨年失效，步态和行为上下文能补稳定身份线索。  
套路：长期 ReID 应该从外观检索转成行为生物特征检索。

**这批给我们的直接启发**
“多分支、多粒度、多尺度”本身已经很拥挤。能发 B 类的不是再拼一个分支，而是先指出一个被忽略的真实失败机制，再让机制服务这个问题。对我们来说，最值得借的是三类写法：显式对齐、可靠性建模、源域或视角不平衡。衣服文本和普通多分支换衣方向不建议正面进入，除非能被 SMPL 或航拍地面的几何问题重新定义。

**强创新点候选**
1. 一句话：把航拍到地面的跨视角 ReID 重新定义为人体三维表面坐标上的检索，而不是二维图像域适配。  
挂的资产：CARGO / AG-ReID.v2，SMPL，SOLIDER-Swin。  
区别：最像 MCCAN 和 DHGN 的多粒度对齐，但它们仍在二维 patch 或视频部位上对齐；我们用 SMPL 把 SOLIDER token 投到规范人体表面，只比较共同可见且几何可信的身体表面区域。也不同于换衣 ReID 里的 3D shape，它不是拿 3D 当身份线索，而是拿 3D 当跨视角坐标系。  
廉价首验：冻结 SOLIDER，在一小批航拍地面配对上估计 SMPL 或关键点，把 patch 特征池化到 8 到 16 个身体表面区域，比较原始全局特征和表面对齐特征的正负样本间隔、rank1 和 mAP。若航拍 query 没有明显改善，或者改善被普通水平分块吃掉，就停。

2. 一句话：用姿态热图的不确定性做跨视角部位可靠性估计，让模型知道哪些身体区域在航拍低清条件下不能信。  
挂的资产：PSG / LGPA-D 姿态热图门控，SMPL，航拍地面数据。  
区别：PSG 类方法把姿态当注意力提示，默认姿态可靠；DHGN 按相似度动态连图。这个方向把热图熵、关节置信度、SMPL 可见性作为可靠性变量，学习“这个视角下这个部位该不该参与匹配”。  
廉价首验：不训练模型，直接用已有姿态热图置信度给 SOLIDER 局部特征加权，重点看航拍低清和严重俯视样本。如果错误样本的低置信部位确实更容易误导，并且置信度加权带来超过噪声的提升，再做学习版。

3. 一句话：把开放世界 ReID 的域不平衡收窄成航拍地面 ReID 的视角稀有性问题。  
挂的资产：CARGO / AG-ReID.v2，SOLIDER-Swin。  
区别：MSP 处理的是多数据集大小不平衡；我们处理的是同一个跨视角系统内部的视角、分辨率、俯仰角和高度分布不平衡。核心不是多模型集成，而是防止模型被大量清晰地面视角牵走，忽略少量但关键的航拍视角。  
廉价首验：按视角高度、分辨率、人体框面积、俯视程度分桶，解析 SOLIDER 基线在各桶的 AP 和错配。如果少数航拍桶明显塌陷，再用一个最小的分桶均衡采样或分桶 BN 跑短训。若分桶差异不明显，或均衡采样没有收益，就不做。

4. 一句话：用 SMPL 生成跨视角反事实可见性监督，让模型学习“换视角后哪些身份证据应该保留，哪些证据本来就会消失”。  
挂的资产：SMPL，SOLIDER-Swin，航拍地面跨视角数据，姿态门控。  
区别：换衣论文是擦衣服或描述衣服来压制衣服捷径；VI-ReID 论文是用动态卷积处理复杂场景。这个方向用几何反事实定义跨视角不变性，不是随机遮挡，也不是普通数据增强。  
廉价首验：用 SMPL 可见性生成俯视和地面视角的身体可见区域掩码，只做小规模训练或冻结特征蒸馏，对比随机擦除、普通低清增强和人体水平分块。如果几何可见性监督不能超过这些廉价对照，就停。

我会优先排 1 和 4。它们最能把团队已有的 SMPL、航拍地面数据和 SOLIDER 强主干连成一个问题层面的故事，也最容易和这批“多分支、多粒度”工作切开。
